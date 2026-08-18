"""Training monitors. The ceiling assertion is the automatic reward-hacking detector.

A trip is treated as a LEAK until proven otherwise, not as a tuning nuisance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


class CeilingViolation(AssertionError):
    """Episode reward exceeded the hard ceiling [I9]. Halts the run."""


class EntropyCollapse(AssertionError):
    """Group advantage variance fell to zero: no gradient, and no signal to recover from."""


def assert_below_ceiling(
    reward: float, ceiling: float, patient_id: str, trajectory: dict[str, Any],
    tolerance: float = 1e-6,
) -> None:
    """Halt and dump the offending trajectory. Per-episode, so use the HARD ceiling.

    Asserting the expected ceiling here would fire on luck -- a proper scoring rule only
    guarantees truthful reporting wins on average -- and a detector that cries wolf gets
    switched off, which is worse than not having one.
    """
    if reward > ceiling + tolerance:
        raise CeilingViolation(
            f"patient {patient_id}: reward {reward:.6f} exceeds the hard ceiling "
            f"{ceiling:.6f}. Treat this as a LEAK until proven otherwise -- it means the "
            f"agent scored better than a perfectly confident correct answer, which is "
            f"impossible without information it should not have.\nTrajectory: {trajectory}"
        )


@dataclass(slots=True)
class RunningCeilingMonitor:
    """Mean reward against the EXPECTED ceiling. The statistical hacking detector."""

    window: int = 512
    tolerance: float = 0.02
    rewards: deque[float] = field(default_factory=deque)
    ceilings: deque[float] = field(default_factory=deque)

    def update(self, reward: float, expected_ceiling: float) -> None:
        self.rewards.append(reward)
        self.ceilings.append(expected_ceiling)
        while len(self.rewards) > self.window:
            self.rewards.popleft()
            self.ceilings.popleft()

    @property
    def breached(self) -> bool:
        if len(self.rewards) < self.window // 2:
            return False
        return float(np.mean(self.rewards)) > float(np.mean(self.ceilings)) + self.tolerance

    def report(self) -> dict[str, float]:
        if not self.rewards:
            return {"n": 0.0, "mean_reward": 0.0, "mean_ceiling": 0.0, "gap": 0.0}
        mr, mc = float(np.mean(self.rewards)), float(np.mean(self.ceilings))
        return {"n": float(len(self.rewards)), "mean_reward": mr, "mean_ceiling": mc,
                "gap": mc - mr}


def group_advantages(
    rewards: npt.NDArray[np.float64], eps: float = 1e-8
) -> npt.NDArray[np.float64]:
    """GRPO advantages: standardised within the group."""
    r = np.asarray(rewards, dtype=np.float64)
    return np.asarray((r - r.mean()) / (r.std() + eps), dtype=np.float64)


def assert_group_has_variance(rewards: npt.NDArray[np.float64], floor: float = 1e-6) -> None:
    """Zero spread means zero advantage means no gradient. Fail loudly, not silently."""
    if float(np.std(rewards)) < floor:
        raise EntropyCollapse(
            f"group reward std {np.std(rewards):.3e} below floor {floor:.1e}: every "
            "rollout scored the same, so the advantage is zero and this batch teaches "
            "nothing. Raise sampling temperature or check for a degenerate policy."
        )
