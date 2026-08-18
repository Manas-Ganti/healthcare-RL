"""Curriculum staging, and the guarded training data loader [I12].

Curriculum: single-condition -> comorbid, short horizon -> full budget. Both exist to
stop "order everything" becoming a survival strategy while the model is still confused --
at full budget with a confused policy, exhaustive testing genuinely is the best available
strategy, and that habit is stubborn once trained in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar

import yaml

_CONFIG_DIR: Final = Path(__file__).resolve().parents[1] / "configs"

T = TypeVar("T")


class CurriculumError(ValueError):
    """Malformed stage definition. Never caught inside `dxenv.train`."""


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    comorbid: bool
    max_turns: int
    advance_criterion: float = 0.0
    """Mean reward required to advance. Zero means "advance on schedule"."""


@dataclass(frozen=True, slots=True)
class Curriculum:
    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise CurriculumError("curriculum has no stages")

    def index_of(self, name: str) -> int:
        for i, s in enumerate(self.stages):
            if s.name == name:
                return i
        raise CurriculumError(f"unknown stage {name!r}")

    def next_stage(self, current: str, mean_reward: float) -> str:
        """Advance at most ONE stage, and only on the criterion.

        Skipping stages defeats the point: the whole reason for staging is that the
        policy learns short-horizon behaviour before it is trusted with a full budget.
        """
        i = self.index_of(current)
        if i + 1 >= len(self.stages):
            return current
        if mean_reward < self.stages[i].advance_criterion:
            return current
        return self.stages[i + 1].name


def load_curriculum(path: Path | None = None) -> Curriculum:
    with (path or _CONFIG_DIR / "env.yaml").open() as fh:
        raw = yaml.safe_load(fh)
    stages = tuple(
        Stage(
            name=str(s["name"]),
            comorbid=bool(s["comorbid"]),
            max_turns=int(s["max_turns"]),
            advance_criterion=float(s.get("advance_criterion", 0.0)),
        )
        for s in raw["curriculum"]["stages"]
    )
    return Curriculum(stages=stages)


def load_training_ids(
    ids: Sequence[str], loader: Callable[[list[str]], list[T]]
) -> list[T]:
    """The ONLY sanctioned path from training code to patient data.

    Routing every training read through one function is what makes the eval guard
    enforceable; a loader called directly bypasses it, so do not add one.
    """
    return loader(list(ids))


def stage_episode_overrides(stage: Stage) -> dict[str, Any]:
    return {"max_turns": stage.max_turns, "comorbid": stage.comorbid}
