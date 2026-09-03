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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import ActionMenu, build_menu
from dxenv.env.episode import DiagnosticEpisode
from dxenv.env.schemas import Action, Observation
from dxenv.policy.decoding import (
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
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> list[list[Generation]]: ...


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
        max_tokens: int = 512,  # noqa: ARG002
        seed: int | None = None,
    ) -> list[list[Generation]]:
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
    max_model_len: int = 8192
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

    def _lazy_engine(self) -> Any:  # pragma: no cover - requires CUDA
        if self._engine is None:
            try:
                from vllm import LLM
            except ImportError as exc:
                raise BackendError(
                    "vLLM is not installed. On a CUDA host: pip install -e '.[gpu]'"
                ) from exc
            self._engine = LLM(
                model=self.model,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                enable_lora=self.lora_path is not None,
                max_lora_rank=64,
            )
        return self._engine

    def generate(  # pragma: no cover - requires CUDA
        self,
        conversations: Sequence[Sequence[dict[str, str]]],
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> list[list[Generation]]:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        engine = self._lazy_engine()
        params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            guided_decoding=GuidedDecodingParams(json=self._schema),
        )
        lora = None
        if self.lora_path is not None:
            from vllm.lora.request import LoRARequest

            lora = LoRARequest(
                f"policy-v{self._lora_version}", self._lora_version, self.lora_path
            )
        outputs = engine.chat(
            [list(c) for c in conversations], params, lora_request=lora, use_tqdm=False
        )
        return [
            [
                Generation(
                    text=c.text,
                    prompt=o.prompt or "",
                    token_ids=tuple(c.token_ids or ()),
                    finish_reason=str(c.finish_reason),
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
    max_tokens: int = 512
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
