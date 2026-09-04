"""SFT dataset construction with Bayes-posterior soft labels, and the LoRA fine-tune.

CLAUDE.md 8.4, and the sentence the whole module is built around:

> **Label targets with the Bayes posterior, not one-hot.** SFT on winning trajectories
> otherwise teaches the model to say 0.99 every time, destroying the calibration the
> Brier score exists to reward -- before RL even starts.

Which is why `build_examples` recomputes the target distribution from the posterior over
what was VISIBLE at that turn, rather than copying whatever the demonstrating policy
happened to report. Even a policy that reports its own posterior can drift from it
(top-k truncation, rounding); recomputing makes the target a property of the evidence
rather than of the demonstrator.

Precisely what the target is
----------------------------
The posterior's top `DEFAULT_MAX_LABELS`, at their exact posterior values, plus the
max-entropy completion of the remaining mass -- NOT the posterior itself. The grammar
caps how many labels a report may name, so the ordering of the unnamed tail is
genuinely lost, and saying "the target is the posterior" would overstate it. Measured on
the diagnosis turns of a 60-patient teacher run, mean total-variation distance from the
true posterior is 0.018 and the worst case is 0.218, both bounded by the unnamed tail
mass. `DEFAULT_MAX_LABELS` carries the measurement that set the cap.

Abstention is seeded, not filtered in
-------------------------------------
An action the SFT set never contains is an action RL never samples, and an action RL
never samples is one it cannot discover. Abstention is the specific casualty: rejection
sampling drops abstentions almost by construction (they score below a good diagnosis, so
they never clear the reward bar), and the result is a policy that structurally cannot
abstain. `seed_abstentions` adds them back from the cases where abstaining is genuinely
the right call -- a near-flat posterior at an exhausted budget -- rather than by
injecting random ones, which would teach the model to abstain arbitrarily.

Deliberately undertrained
-------------------------
1-2 epochs, low LR, LoRA, stop early. The goal is a competent prior, not a finished
policy: GRPO needs within-group variation to compute advantages from, and a model
sharpened onto its SFT set produces identical rollouts and therefore zero gradient. The
defaults here are chosen to leave entropy on the table.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.bayes import entropy, posterior
from dxenv.env.obs_model import ObservationModel, build_observation_model
from dxenv.env.schemas import Diagnose, Observation, OrderTest
from dxenv.policy.baselines import evidence_from_observation
from dxenv.policy.decoding import DEFAULT_MAX_LABELS, complete_distribution, render_wire
from dxenv.policy.prompt import chat_messages
from dxenv.policy.teacher import TeacherTrace, TeacherTurn

DEFAULT_ENTROPY_FLOOR: Final = 0.25
"""Minimum mean target entropy, in nats, over the diagnosis examples.

A one-hot target has entropy 0. This floor is what `test_sft_targets_not_onehot` asserts
against; it is set low deliberately, because a confident target on an easy patient is
correct and only a set that is confident EVERYWHERE indicates collapse."""

MIN_ABSTAIN_FRACTION: Final = 0.02
"""Floor under the abstention share of the SFT set (`test_abstain_present_in_sft_set`)."""


class SFTError(ValueError):
    """Malformed example or dataset. Never caught inside `dxenv.policy`."""


@dataclass(frozen=True, slots=True)
class SFTExample:
    """One (prompt, completion) pair, plus what it was derived from."""

    messages: tuple[dict[str, str], ...]
    completion: str
    patient_id: str
    turn: int
    kind: str
    target_entropy: float = 0.0
    source: str = "teacher"

    def as_dict(self) -> dict[str, Any]:
        return {
            "messages": [dict(m) for m in self.messages],
            "completion": self.completion,
            "patient_id": self.patient_id,
            "turn": self.turn,
            "kind": self.kind,
            "target_entropy": self.target_entropy,
            "source": self.source,
        }

    def distribution(self, taxonomy: Taxonomy | None = None) -> dict[str, float]:
        """The full distribution this example actually trains toward.

        Parses the completion the way the environment will at rollout time, residual
        mass included -- so this is the target as the SCORER sees it, not as the JSON
        literally reads. The two differ, and the difference is the whole reason
        `complete_distribution` exists.
        """
        obj = json.loads(self.completion)
        if obj.get("kind") != "diagnose":
            raise SFTError(f"example {self.patient_id}:{self.turn} is not a diagnosis")
        return complete_distribution(obj["diagnosis"], taxonomy)


def soft_label_wire(
    belief: npt.NDArray[np.float64],
    reasoning: str,
    taxonomy: Taxonomy | None = None,
    max_labels: int = DEFAULT_MAX_LABELS,
) -> dict[str, Any]:
    """A `diagnose` wire object carrying the posterior, top-`max_labels` of it.

    The tail is not renormalised away: `complete_distribution` spreads the unnamed mass
    uniformly at parse time, so a target naming 8 of 149 labels still trains the model
    toward "and 0.14 of my belief is elsewhere" rather than toward a false confidence
    manufactured by truncation.
    """
    tax = taxonomy or load_taxonomy()
    order = np.argsort(-belief)[:max_labels]
    return {
        "kind": "diagnose",
        "reasoning": reasoning,
        "diagnosis": [
            # 9 dp, not 6: at 16 named labels, 6 dp accumulates ~8e-6 of rounding error,
            # which is larger than the unnamed tail on a confident posterior and makes the
            # target measurably not-the-posterior for no benefit.
            {"condition": tax.slugs[int(i)], "probability": round(float(belief[int(i)]), 9)}
            for i in order
        ],
    }


def _example_from_turn(
    turn: TeacherTurn,
    patient_id: str,
    taxonomy: Taxonomy,
    model: ObservationModel,
    source: str,
) -> SFTExample:
    messages = tuple(chat_messages(turn.observation, taxonomy=taxonomy))
    if isinstance(turn.action, Diagnose):
        belief = posterior(evidence_from_observation(turn.observation), model)
        wire = soft_label_wire(belief, turn.reasoning, taxonomy)
        target_h = entropy(np.array(list(complete_distribution(wire["diagnosis"], taxonomy)
                                          .values()), dtype=np.float64))
        return SFTExample(
            messages=messages, completion=render_wire(wire), patient_id=patient_id,
            turn=turn.turn, kind="diagnose", target_entropy=target_h, source=source,
        )
    kind = "order_test" if isinstance(turn.action, OrderTest) else turn.action.kind
    return SFTExample(
        messages=messages, completion=render_wire(turn.wire()), patient_id=patient_id,
        turn=turn.turn, kind=kind, source=source,
    )


def build_examples(
    traces: Sequence[TeacherTrace],
    taxonomy: Taxonomy | None = None,
    model: ObservationModel | None = None,
    source: str = "teacher",
) -> list[SFTExample]:
    """De-leaked traces -> training examples, one per turn.

    Raises on a privileged trace rather than filtering it: reaching this function with
    privilege intact means the pipeline skipped `deleak`, and quietly dropping the trace
    would hide that the step is missing.
    """
    tax = taxonomy or load_taxonomy()
    m = model or build_observation_model()
    out: list[SFTExample] = []
    for tr in traces:
        if tr.privileged:
            raise SFTError(
                f"trace {tr.patient_id} is still privileged. Run `teacher.deleak` and "
                "`teacher.filter_traces` first -- leaked reasoning in SFT data is worse "
                "than no SFT data (CLAUDE.md 8.2)."
            )
        out.extend(_example_from_turn(t, tr.patient_id, tax, m, source) for t in tr.turns)
    return out


def abstention_reasoning(obs: Observation, h: float) -> str:
    return (
        f"After {len(obs.revealed_results)} result(s) my belief is still diffuse "
        f"(entropy {h:.2f} nats) and {obs.remaining_budget:g} of budget remains, which "
        f"buys less separation than it costs. Naming a distribution here would state a "
        f"confidence I do not have, so I abstain."
    )


def seed_abstentions(
    traces: Sequence[TeacherTrace],
    fraction: float = 0.05,
    entropy_quantile: float = 0.85,
    taxonomy: Taxonomy | None = None,
    model: ObservationModel | None = None,
) -> list[SFTExample]:
    """Abstention examples drawn from the genuinely ambiguous cases.

    Selected by posterior entropy, not at random: abstaining is the right action when the
    evidence does not separate the hypotheses, and an SFT set that abstains at random
    teaches the model to abstain at random. The top `entropy_quantile` of first-turn
    posteriors is the pool; `fraction` of the traces are drawn from it, highest entropy
    first, so the seeded behaviour is "abstain when lost" rather than "abstain sometimes".
    """
    tax = taxonomy or load_taxonomy()
    m = model or build_observation_model()
    scored: list[tuple[float, TeacherTrace]] = []
    for tr in traces:
        if not tr.turns:
            continue
        obs = tr.turns[0].observation
        scored.append((entropy(posterior(evidence_from_observation(obs), m)), tr))
    if not scored:
        return []
    scored.sort(key=lambda kv: -kv[0])
    cutoff = float(np.quantile([h for h, _ in scored], entropy_quantile))
    pool = [(h, tr) for h, tr in scored if h >= cutoff]
    n = max(1, round(len(traces) * fraction))
    out: list[SFTExample] = []
    for h, tr in pool[:n]:
        obs = tr.turns[0].observation
        wire = {"kind": "abstain", "reasoning": abstention_reasoning(obs, h)}
        out.append(
            SFTExample(
                messages=tuple(chat_messages(obs, taxonomy=tax)),
                completion=render_wire(wire),
                patient_id=tr.patient_id,
                turn=obs.turn + 1,
                kind="abstain",
                source="seeded_abstention",
            )
        )
    return out


@dataclass(slots=True)
class SFTDataset:
    examples: list[SFTExample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.examples:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return counts

    def abstain_fraction(self) -> float:
        return self.kind_counts().get("abstain", 0) / max(len(self.examples), 1)

    def mean_target_entropy(self) -> float:
        h = [e.target_entropy for e in self.examples if e.kind == "diagnose"]
        return float(np.mean(h)) if h else 0.0

    def validate(
        self,
        entropy_floor: float = DEFAULT_ENTROPY_FLOOR,
        min_abstain: float = MIN_ABSTAIN_FRACTION,
    ) -> None:
        """Refuse a dataset that would train the pathologies Phase 3 exists to avoid."""
        if not self.examples:
            raise SFTError("empty SFT set")
        h = self.mean_target_entropy()
        if h < entropy_floor:
            raise SFTError(
                f"mean diagnosis target entropy {h:.3f} nats is below the floor "
                f"{entropy_floor:.3f}. The targets have collapsed toward one-hot, which "
                "destroys the calibration the Brier score exists to reward -- before RL "
                "starts. Check that soft labels are being recomputed from the posterior."
            )
        frac = self.abstain_fraction()
        if frac < min_abstain:
            raise SFTError(
                f"abstentions are {frac:.3%} of the set, below {min_abstain:.1%}. An "
                "action the SFT set never contains is one RL never samples, and one RL "
                "never samples is one it cannot discover. Call `seed_abstentions`."
            )

    def write_jsonl(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for e in self.examples:
                fh.write(json.dumps(e.as_dict(), separators=(",", ":"), sort_keys=True) + "\n")
        return len(self.examples)

    @classmethod
    def read_jsonl(cls, path: Path) -> SFTDataset:
        examples = []
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                o = json.loads(raw)
                examples.append(
                    SFTExample(
                        messages=tuple(o["messages"]), completion=o["completion"],
                        patient_id=o["patient_id"], turn=o["turn"], kind=o["kind"],
                        target_entropy=float(o.get("target_entropy", 0.0)),
                        source=o.get("source", "teacher"),
                    )
                )
        return cls(examples)

    def summary(self) -> dict[str, Any]:
        return {
            "n": len(self.examples),
            "kinds": self.kind_counts(),
            "abstain_fraction": self.abstain_fraction(),
            "mean_target_entropy": self.mean_target_entropy(),
            "n_patients": len({e.patient_id for e in self.examples}),
        }


# ------------------------------------------------------------------- the fine-tune ----


@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Deliberately undertrained. Every default here leaves entropy on the table."""

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: Path = Path("runs/sft")
    epochs: float = 1.0
    learning_rate: float = 1e-5
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    batch_size: int = 4
    grad_accum: int = 8
    max_seq_len: int = 4096
    warmup_ratio: float = 0.03
    seed: int = 0
    bf16: bool = True

    def __post_init__(self) -> None:
        if self.epochs > 2.0:
            raise SFTError(
                f"{self.epochs} epochs. CLAUDE.md 8.4 says 1-2 and means it: a model "
                "sharpened onto its SFT set produces identical rollouts, identical "
                "rollouts give zero advantage, and GRPO then has no gradient to work "
                "with. Undertraining here is the point, not a compromise."
            )


def _trl_config(trl_config_cls: Any, cfg: SFTConfig, bf16: bool) -> Any:
    """Build TRL's SFTConfig, passing only the fields this TRL version accepts.

    TRL renames these often -- `max_seq_length` became `max_length`, `completion_only_loss`
    arrived mid-series -- and an unknown keyword is a TypeError twenty minutes into a job
    holding a GPU. Filtering against the installed signature turns that into a printed
    line naming what was dropped.

    `completion_only_loss` is the one that matters and is checked explicitly: without it
    the run trains on the prompt, and the prompt is a 13k-character menu and label list the
    model never has to generate. That would not crash -- it would just waste the run.
    """
    import inspect

    wanted = {
        "output_dir": str(cfg.output_dir),
        "num_train_epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "max_length": cfg.max_seq_len,
        "warmup_ratio": cfg.warmup_ratio,
        "bf16": bf16,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "seed": cfg.seed,
        "report_to": [],
        "completion_only_loss": True,
    }
    accepted = set(inspect.signature(trl_config_cls).parameters)
    dropped = sorted(k for k in wanted if k not in accepted)
    if dropped:
        print(f"[dxenv] TRL {trl_config_cls.__name__} does not accept {dropped}; dropping")
    if "completion_only_loss" in dropped:
        print(
            "[dxenv] WARNING: this TRL cannot restrict loss to the completion, so the run "
            "will also train on the prompt -- a 13k-character menu the model never has to "
            "generate. Check TRL's current name for that option before trusting the "
            "resulting adapter."
        )
    if "max_length" in dropped and "max_seq_length" in accepted:
        wanted["max_seq_length"] = wanted.pop("max_length")  # the older spelling
    return trl_config_cls(**{k: v for k, v in wanted.items() if k in accepted})


def train_lora(dataset: SFTDataset, cfg: SFTConfig) -> Path:  # pragma: no cover - GPU only
    """LoRA SFT over the examples. Imports torch/peft/trl lazily; a CUDA host only.

    Returns the adapter directory, which `policy.llm.VLLMBackend(lora_path=...)` loads
    and `train/grpo.py` takes as its reference policy for the KL term.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig as TRLConfig
        from trl import SFTTrainer
    except ImportError as exc:
        raise SFTError(
            "SFT needs the GPU extra: pip install -e '.[gpu]' on a CUDA host. The "
            "dataset itself is plain JSONL and builds anywhere."
        ) from exc

    dataset.validate()
    tok = AutoTokenizer.from_pretrained(cfg.model)
    # PROMPT-COMPLETION format, not a single `messages` list. `completion_only_loss`
    # applies to prompt-completion datasets; with a conversational messages-only dataset
    # TRL cannot tell where the prompt ends, so it either ignores the flag or raises.
    # Getting it wrong would train on the prompt -- and the prompt is a 13k-character menu
    # and label list that the model never has to generate, so most of the gradient would
    # go into reproducing fixed text.
    rows = [
        {
            "prompt": [dict(m) for m in e.messages],
            "completion": [{"role": "assistant", "content": e.completion}],
        }
        for e in dataset.examples
    ]
    ds = Dataset.from_list(rows)
    trainer = SFTTrainer(
        model=cfg.model,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
        args=_trl_config(TRLConfig, cfg, bf16=cfg.bf16 and torch.cuda.is_available()),
    )
    trainer.train()
    out = cfg.output_dir / "final"
    trainer.save_model(str(out))
    return out
