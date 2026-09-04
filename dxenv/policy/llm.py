"""Model-backed policies, and the backends they run on.

Two backends, one interface:

  `VLLMBackend`   -- the real thing. vLLM with guided decoding and a LoRA adapter.
                     Imported lazily; a CUDA host only.
  `RandomBackend` -- samples uniformly from the grammar. Not a mock in the usual sense:
                     it exercises the identical prompt-build -> constrain -> parse ->
                     step path, so every commit tests that path even though the fast
                     suite has no GPU. It is also the honest "format-valid but
                     uninformed" reference, which is a different floor from the prior
                     policy and worth reporting separately.

One turn = one independent (prompt, completion) pair
----------------------------------------------------
The policy is stateless across turns: each turn rebuilds the prompt from the current
observation rather than appending to a growing chat transcript. Three reasons, in order
of weight:

  1. The observation already carries the full case state -- every revealed result, the
     ledger, the turn count. A transcript would restate it, at quadratic token cost.
  2. Every turn becomes an independently trainable example, which is what lets GRPO
     broadcast one episode-level advantage across the turns that produced it without
     worrying about which prefix each turn saw.
  3. A transcript accumulates the model's own past reasoning, and de-leaked reasoning
     that was clean at turn 2 can become a leak at turn 7 by being read alongside a
     result it now explains. Rebuilding from the typed observation cannot drift.

The cost of the choice is real and worth naming: the model cannot remember a plan it
formed at turn 1 except through the actions it took. That is a deliberate restriction to
what is verifiable from the record.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import ActionMenu, build_menu
from dxenv.env.episode import DiagnosticEpisode
from dxenv.env.schemas import Action, Observation
from dxenv.policy.decoding import (
    DEFAULT_MAX_TOKENS,
    DecodingError,
    action_json_schema,
    parse_action,
    render_wire,
    sample_wire_action,
)
from dxenv.policy.prompt import chat_messages


class BackendError(RuntimeError):
    """The backend could not produce a generation. Never caught inside `dxenv.policy`."""


@dataclass(frozen=True, slots=True)
class Generation:
    """One sampled completion. `text` is what the parser sees."""

    text: str
    prompt: str = ""
    token_ids: tuple[int, ...] = ()
    finish_reason: str = "stop"


class Backend(Protocol):
    def generate(
        self,
        conversations: Sequence[Sequence[dict[str, str]]],
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        seed: int | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> list[list[Generation]]:
        """`seeds` supplies one seed PER conversation.

        That is what lets a batched call stay exactly as reproducible as the sequential
        calls it replaces -- vLLM accepts a list of SamplingParams aligned with the
        conversations, so throughput does not have to be traded against determinism.
        `seed` remains for the single-conversation case.
        """
        ...


@dataclass(slots=True)
class RandomBackend:
    """Uniform over the grammar. Deterministic given its seed [I10]."""

    seed: int = 0
    menu: ActionMenu = field(default_factory=build_menu)
    taxonomy: Taxonomy = field(default_factory=load_taxonomy)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def generate(
        self,
        conversations: Sequence[Sequence[dict[str, str]]],
        n: int = 1,
        temperature: float = 1.0,  # noqa: ARG002 - a uniform sampler has no temperature
        max_tokens: int = DEFAULT_MAX_TOKENS,  # noqa: ARG002
        seed: int | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> list[list[Generation]]:
        if seeds is not None:
            return [
                self.generate([c], n=n, seed=sd)[0]
                for c, sd in zip(conversations, seeds, strict=True)
            ]
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        out: list[list[Generation]] = []
        for conv in conversations:
            prompt = "\n".join(m["content"] for m in conv)
            out.append(
                [
                    Generation(
                        text=render_wire(
                            sample_wire_action(rng, menu=self.menu, taxonomy=self.taxonomy)
                        ),
                        prompt=prompt,
                    )
                    for _ in range(n)
                ]
            )
        return out


@dataclass(slots=True)
class VLLMBackend:
    """vLLM with JSON-schema guided decoding and an optional LoRA adapter.

    Constructed lazily and held for the process lifetime: engine startup dominates a
    short run, and re-creating it per GRPO step would spend most of the wall clock
    loading weights it already had.
    """

    model: str
    lora_path: str | None = None
    max_model_len: int = 16384
    """Context window. 8192 left only ~540 tokens of margin.

    Measured: the prompt is ~10.5k characters at turn 0 and ~14.3k with every analyte
    revealed, which at a pessimistic 3 chars/token is 4,777 prompt tokens; add a
    truncation retry at 2,876 and the worst case reaches 7,653 against a 8,192 limit. The
    menu and the 149-label list dominate and are full of underscored slugs, which
    tokenize worse than prose. Overflow is a hard failure mid-run, and KV cache is not
    the scarce resource on an 80GB card running a 7B.
    """
    gpu_memory_utilization: float = 0.55
    """Deliberately below vLLM's own default.

    In a GRPO run this engine shares a device with the trainer, which holds the 7B in
    bf16 plus activations and optimizer state. vLLM PREALLOCATES its KV cache at startup,
    so a utilization tuned for a pure-inference server leaves the trainer to OOM on the
    first backward pass -- after the engine has loaded, which is the most expensive
    possible moment to find out. Raise it for a standalone eval sweep, where nothing else
    is on the card.
    """
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    _engine: Any = field(default=None, init=False)
    _schema: dict[str, Any] = field(default_factory=action_json_schema, init=False)
    _lora_version: int = field(default=1, init=False)
    """Bumped on every reload. vLLM caches an adapter BY ID, so pushing new weights to a
    path the engine has already loaded under id 1 is a no-op -- the engine keeps serving
    the old adapter and nothing says so. That failure is silent and it turns a GRPO run
    into rejection sampling against a frozen policy."""

    use_flashinfer_sampler: bool = False
    """Whether to let vLLM use FlashInfer's sampler.

    Off by default. FlashInfer JIT-COMPILES its sampling kernel during engine warmup and
    needs nvcc; on a cluster whose compute nodes carry a driver but no CUDA toolkit that
    aborts engine startup outright:

        RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
                      doesn't exist

    The PyTorch sampler is numerically equivalent and compiles nothing, so the default
    costs some throughput and buys independence from whether a toolkit happens to be
    installed. Set True once one is confirmed working.
    """

    def _lazy_engine(self) -> Any:  # pragma: no cover - requires CUDA
        if self._engine is None:
            # Set BEFORE importing vllm, which reads it at import time. Done here rather
            # than only in slurm/env.sh so the backend is correct when used directly --
            # a library that needs a sibling shell script to work is a library with an
            # undocumented dependency.
            if not self.use_flashinfer_sampler:
                os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            try:
                from vllm import LLM
            except ImportError as exc:
                raise BackendError(
                    "vLLM is not installed. On a CUDA host: pip install -e '.[gpu]'"
                ) from exc
            kwargs: dict[str, Any] = {
                "model": self.model,
                "dtype": self.dtype,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "tensor_parallel_size": self.tensor_parallel_size,
                "enable_lora": self.lora_path is not None,
                "max_lora_rank": 64,
            }
            # Compact JSON. vLLM permits arbitrary whitespace between tokens by default,
            # and the model duly spends them pretty-printing -- observed emitting
            # '{\n  "kind": ...'. Indentation carries nothing the parser uses, and
            # whitespace the grammar allows is another unbounded channel.
            #
            # `disable_any_whitespace` is rejected unless the backend is named explicitly
            # ("only supported for xgrammar and guidance"), so both go together.
            #
            # Tried in descending order of preference and falling back on ANY failure,
            # because every config-shape mismatch here has cost a full queue cycle to
            # discover, and the last option -- vLLM's own defaults -- is always correct,
            # merely more token-hungry. This is engine configuration, not policy: the
            # fallback changes how output is formatted, never which action is chosen, and
            # it says loudly which rung it landed on.
            # No `disable_any_whitespace`. It was accepted by the engine and the model
            # still emitted '{"kind": "order_test", ...' with spaces, so it was not doing
            # what it claimed; and now that SFT targets are rendered with the model's own
            # spacing (see decoding.WIRE_KEY_ORDER), forcing compact output would create
            # the very train/inference mismatch that wrecked the first SFT run.
            attempts: list[dict[str, Any]] = [{"backend": "xgrammar"}, {}]
            supports_cfg = (
                "structured_outputs_config" in inspect.signature(LLM.__init__).parameters
            )
            last: Exception | None = None
            for cfg in attempts:
                try:
                    self._engine = LLM(
                        **kwargs,
                        **({"structured_outputs_config": cfg} if cfg and supports_cfg else {}),
                    )
                    print(f"[dxenv] vLLM structured_outputs_config={cfg or 'defaults'}")
                    break
                except Exception as exc:
                    last = exc
                    print(f"[dxenv] structured_outputs_config={cfg} rejected "
                          f"({type(exc).__name__}: {str(exc)[:160]}); trying simpler")
            if self._engine is None:
                raise BackendError(
                    f"vLLM engine would not start under any structured-output "
                    f"configuration. Last error: {last}"
                ) from last
        return self._engine

    def _structured_output_kwargs(self) -> dict[str, Any]:  # pragma: no cover - CUDA only
        """Build the JSON-schema constraint for whichever vLLM API is installed.

        vLLM renamed this: `guided_decoding=GuidedDecodingParams(...)` became
        `structured_outputs=StructuredOutputsParams(...)`. This code was written against
        ~0.6 and the cluster runs 0.28, where the old name raises ImportError.

        Detected rather than pinned. A version check would encode today's cutover and
        break at the next rename in a place nobody would look; asking the installed
        SamplingParams which keyword it actually accepts stays correct across both, and
        fails with the available parameter names rather than an ImportError if a third
        spelling ever appears.
        """
        import inspect

        from vllm import SamplingParams
        from vllm import sampling_params as sp

        accepted = set(inspect.signature(SamplingParams).parameters)
        for kwarg, cls_name in (
            ("structured_outputs", "StructuredOutputsParams"),
            ("guided_decoding", "GuidedDecodingParams"),
        ):
            if kwarg in accepted and hasattr(sp, cls_name):
                return {kwarg: getattr(sp, cls_name)(json=dict(self._schema))}
        raise BackendError(
            "this vLLM exposes neither `structured_outputs` nor `guided_decoding` on "
            f"SamplingParams (it accepts: {sorted(accepted)}). Constrained decoding is "
            "not optional here -- without it the grammar is unenforced and I3 stops "
            "holding -- so fix the mapping in VLLMBackend rather than falling back to "
            "free generation."
        )

    def generate(  # pragma: no cover - requires CUDA
        self,
        conversations: Sequence[Sequence[dict[str, str]]],
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        seed: int | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> list[list[Generation]]:
        from vllm import SamplingParams

        engine = self._lazy_engine()
        constraint = self._structured_output_kwargs()

        def _params(one_seed: int | None) -> Any:
            return SamplingParams(
                n=n, temperature=temperature, max_tokens=max_tokens,
                seed=one_seed, **constraint,
            )

        # vLLM accepts a LIST of SamplingParams aligned with the conversations, so a
        # batched call still gives each sequence its own seed. Without that, batching
        # would buy throughput by giving up per-episode reproducibility, and those two
        # are not worth trading against each other.
        params: Any = [_params(x) for x in seeds] if seeds is not None else _params(seed)
        lora = None
        if self.lora_path is not None:
            from vllm.lora.request import LoRARequest

            lora = LoRARequest(
                f"policy-v{self._lora_version}", self._lora_version, self.lora_path
            )
        outputs = engine.chat(
            [list(c) for c in conversations], params, lora_request=lora, use_tqdm=False
        )
        # Defensive attribute access on vLLM's output objects. `text` is load-bearing and
        # must be present; the rest are metadata this code can do without, and the last
        # four failures were all API-shape drift on this surface -- there is no reason to
        # let a renamed `finish_reason` cost another queue cycle when its absence only
        # loses the truncation check.
        return [
            [
                Generation(
                    text=c.text,
                    prompt=getattr(o, "prompt", "") or "",
                    token_ids=tuple(getattr(c, "token_ids", ()) or ()),
                    finish_reason=str(getattr(c, "finish_reason", "stop") or "stop"),
                )
                for c in o.outputs
            ]
            for o in outputs
        ]


    def reload_lora(self, path: str) -> None:  # pragma: no cover - requires CUDA
        """Point the sampler at freshly trained weights, under a NEW adapter id."""
        self.lora_path = path
        self._lora_version += 1


@dataclass(slots=True)
class LLMPolicy:
    """A `Policy` over any `Backend`. Records every generation for later training.

    `generations` is keyed by turn so a rollout can be replayed as training examples
    without re-running the model -- rollouts are the expensive artifact of the whole
    project (CLAUDE.md 4).
    """

    backend: Backend
    temperature: float = 1.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int | None = None
    menu: ActionMenu = field(default_factory=build_menu)
    taxonomy: Taxonomy = field(default_factory=load_taxonomy)
    generations: list[dict[str, Any]] = field(default_factory=list)

    def reset_log(self) -> None:
        self.generations = []

    def act(self, episode: DiagnosticEpisode, obs: Observation) -> Action:
        conv = chat_messages(obs, menu=self.menu, taxonomy=self.taxonomy)
        turn_seed = None if self.seed is None else self.seed + obs.turn
        gen = self.backend.generate(
            [conv], n=1, temperature=self.temperature,
            max_tokens=self.max_tokens, seed=turn_seed,
        )[0][0]

        # A generation stopped by the token limit is a PREFIX of valid JSON -- correct
        # structure, unfinished string -- and reporting that as "the grammar was not
        # applied" sends the reader looking in exactly the wrong place. Retry once with a
        # larger budget: finishing the same constrained generation is not choosing a
        # different action, so it does not substitute anything the way a fallback would.
        if gen.finish_reason == "length":
            gen = self.backend.generate(
                [conv], n=1, temperature=self.temperature,
                max_tokens=self.max_tokens * 2, seed=turn_seed,
            )[0][0]
            if gen.finish_reason == "length":
                raise DecodingError(
                    f"turn {obs.turn} of {episode.record.patient_id}: the model did not "
                    f"finish a valid action within {self.max_tokens * 2} tokens, so the "
                    f"output is a truncated prefix rather than malformed JSON. Raise "
                    f"LLMPolicy.max_tokens, or shorten the reasoning the prompt asks for "
                    f"-- do NOT add a fallback action, which would silently replace what "
                    f"the policy was going to do. Got: {gen.text[:200]!r}"
                )
        try:
            action = parse_action(gen.text, self.menu, self.taxonomy)
        except DecodingError as exc:
            raise DecodingError(
                f"turn {obs.turn} of {episode.record.patient_id}: {exc}. Under guided "
                "decoding this cannot happen, so it means the schema was not applied to "
                "this call -- fix the backend rather than adding a fallback action."
            ) from exc
        self.generations.append(
            {
                "turn": obs.turn,
                "prompt": [dict(m) for m in conv],
                "completion": gen.text,
                "finish_reason": gen.finish_reason,
            }
        )
        return action


def batched_act(
    policies: Sequence[LLMPolicy],
    episodes: Sequence[DiagnosticEpisode],
    observations: Sequence[Observation],
    strict: bool = True,
) -> list[Action | None]:
    """One backend call for many episodes' current turns.

    This is what makes the project run at all. Driving episodes one at a time issues a
    single-sequence request per turn, which wastes almost all of a GPU: measured against
    the plan, Gate B needs 4,800-12,800 calls and a GRPO run needs ~640,000, which is
    ~356 hours sequentially. vLLM's whole advantage is batching, and the episodes in a
    group -- and across patients in a GRPO step -- are independent at any given turn.

    Every policy must share one backend, since the point is to reach it once. Per-episode
    seeds are preserved through `seeds`, so this is faster than the sequential path
    without being less reproducible than it.
    """
    if not policies:
        return []
    backend = policies[0].backend
    if any(p.backend is not backend for p in policies):
        raise BackendError(
            "batched_act needs every policy to share one backend; separate backends mean "
            "separate engines, which is the thing this exists to avoid."
        )

    convs = [
        chat_messages(obs, menu=p.menu, taxonomy=p.taxonomy)
        for p, obs in zip(policies, observations, strict=True)
    ]
    seeds = [
        None if p.seed is None else p.seed + obs.turn
        for p, obs in zip(policies, observations, strict=True)
    ]
    budget = max(p.max_tokens for p in policies)
    gens = [g[0] for g in backend.generate(
        convs, n=1, temperature=policies[0].temperature, max_tokens=budget, seeds=seeds
    )]

    # Retry only the truncated ones, and only once. Same reasoning as the single-episode
    # path: finishing a constrained generation is not choosing a different action.
    stuck = [i for i, g in enumerate(gens) if g.finish_reason == "length"]
    if stuck:
        retry = backend.generate(
            [convs[i] for i in stuck], n=1, temperature=policies[0].temperature,
            max_tokens=budget * 2, seeds=[seeds[i] for i in stuck],
        )
        for slot, out in zip(stuck, retry, strict=True):
            gens[slot] = out[0]

    # Alignment, checked rather than trusted. A batched call returns a list, and if it
    # were ever out of order -- by a backend change, a scheduler reordering, or a bug here
    # -- episode i would be handed episode j's action. Nothing downstream could detect
    # that: every action is individually legal, every episode still terminates, and the
    # rewards would simply be attached to the wrong trajectories. It would look like a
    # weak policy rather than a broken harness.
    #
    # Every prompt names its own case ("CASE <patient_ref> (turn N"), so the returned
    # prompt is a witness of which conversation produced it, when the backend supplies it.
    for obs, gen in zip(observations, gens, strict=True):
        marker = f"CASE {obs.patient_ref}"
        if gen.prompt and marker not in gen.prompt:
            raise BackendError(
                f"batched outputs are misaligned: the generation returned for "
                f"{obs.patient_ref} carries a different case in its prompt. Every action "
                f"would be legal and attached to the wrong episode, so this must halt "
                f"rather than be scored."
            )

    actions: list[Action | None] = []
    for i, (policy, episode, obs, gen) in enumerate(
        zip(policies, episodes, observations, gens, strict=True)
    ):
        if gen.finish_reason == "length":
            message = (
                f"turn {obs.turn} of {episode.record.patient_id}: no valid action within "
                f"{budget * 2} tokens; the output is a truncated prefix. Raise "
                f"max_tokens or shorten the reasoning the prompt asks for. Do NOT add a "
                f"fallback action. Got: {gen.text[:200]!r}"
            )
            if strict:
                raise DecodingError(message)
            print(f"[dxenv] {message}", flush=True)
            actions.append(None)
            continue
        try:
            action = parse_action(gen.text, policy.menu, policy.taxonomy)
        except DecodingError as exc:
            if strict:
                raise DecodingError(
                    f"turn {obs.turn} of {episode.record.patient_id}: {exc}"
                ) from exc
            # Not strict: record the failure and let the caller end this episode. A
            # 1,600-episode sweep must not die because one generation degenerated -- and
            # the information is not lost, because Gate B scores schema_valid_fraction
            # against a threshold of 1.0, so a single failure still fails the gate. Loud,
            # but not fatal to the run that would have reported it.
            print(f"[dxenv] decode failure, {episode.record.patient_id} turn {obs.turn}: "
                  f"{gen.text[:120]!r}", flush=True)
            actions.append(None)
            continue
        policy.generations.append(
            {
                "turn": obs.turn,
                "prompt": [dict(m) for m in convs[i]],
                "completion": gen.text,
                "finish_reason": gen.finish_reason,
            }
        )
        actions.append(action)
    return actions
