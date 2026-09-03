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


class CostCollapse(AssertionError):
    """The cost distribution collapsed to an endpoint. Both ends are failures."""


class DegenerateGroups(AssertionError):
    """Too many groups had no within-group spread to compute an advantage from."""


@dataclass(slots=True)
class CostDistributionMonitor:
    """Watch the test count for collapse to zero or to the budget cap.

    Both endpoints are failure modes and they look nothing alike:

      collapse to ZERO   the agent has learned that tests never pay, which is I5 working
                         too well -- lambda is too high and the environment has become
                         "guess from vitals". The cost-accuracy frontier is then a point.
      collapse to the CAP  "order everything" became a survival strategy, which is what
                         the curriculum exists to prevent. Usually appears early, while
                         the policy is still confused and exhaustive testing genuinely is
                         its best available move.

    Neither halts the run on its own -- a stage that legitimately trains at zero tests
    exists -- so this reports and the caller decides. `assert_healthy` is available for
    callers that want the halt.
    """

    window: int = 512
    counts: deque[int] = field(default_factory=deque)
    budget_capped: deque[bool] = field(default_factory=deque)

    def update(self, n_tests: int, hit_budget_cap: bool) -> None:
        self.counts.append(int(n_tests))
        self.budget_capped.append(bool(hit_budget_cap))
        while len(self.counts) > self.window:
            self.counts.popleft()
            self.budget_capped.popleft()

    def report(self) -> dict[str, float]:
        if not self.counts:
            return {"n": 0.0, "mean_tests": 0.0, "zero_fraction": 0.0, "capped_fraction": 0.0}
        arr = np.array(self.counts, dtype=np.float64)
        return {
            "n": float(len(arr)),
            "mean_tests": float(arr.mean()),
            "p90_tests": float(np.quantile(arr, 0.9)),
            "zero_fraction": float((arr == 0).mean()),
            "capped_fraction": float(np.mean(self.budget_capped)),
        }

    def assert_healthy(self, max_zero: float = 0.95, max_capped: float = 0.5) -> None:
        if len(self.counts) < self.window // 2:
            return
        r = self.report()
        if r["zero_fraction"] > max_zero:
            raise CostCollapse(
                f"{r['zero_fraction']:.1%} of episodes ordered NO tests. The agent has "
                "concluded that testing never pays, so the cost-accuracy frontier has "
                "collapsed to a point and the environment is now 'guess from vitals'. "
                "Check costs.lambda against the marginal value measurement in "
                "configs/reward.yaml before touching the policy."
            )
        if r["capped_fraction"] > max_capped:
            raise CostCollapse(
                f"{r['capped_fraction']:.1%} of episodes spent the whole budget. "
                "'Order everything' has become the survival strategy -- shorten the "
                "curriculum stage or the horizon rather than raising lambda, which "
                "would also punish the policies that test well."
            )


@dataclass(slots=True)
class DegenerateGroupMonitor:
    """Fraction of groups with no reward spread, over a window.

    Asserting per group would halt on a legitimately easy patient where every sample
    lands in the same place -- that is the environment being clear, not the policy being
    collapsed. What matters is the RATE: a run where a third of groups are degenerate is
    training on two-thirds of its batch and paying for all of it.
    """

    window: int = 128
    floor: float = 1e-6
    max_fraction: float = 0.5
    flags: deque[bool] = field(default_factory=deque)

    def update(self, rewards: npt.NDArray[np.float64]) -> bool:
        degenerate = float(np.std(rewards)) < self.floor
        self.flags.append(degenerate)
        while len(self.flags) > self.window:
            self.flags.popleft()
        return degenerate

    @property
    def fraction(self) -> float:
        return float(np.mean(self.flags)) if self.flags else 0.0

    def assert_healthy(self) -> None:
        if len(self.flags) < self.window // 2:
            return
        if self.fraction > self.max_fraction:
            raise DegenerateGroups(
                f"{self.fraction:.1%} of the last {len(self.flags)} groups had zero "
                f"reward spread, above the {self.max_fraction:.0%} ceiling. Those "
                "batches contribute exactly nothing to the gradient. Raise sampling "
                "temperature, or check whether the policy has collapsed onto one action."
            )

    def report(self) -> dict[str, float]:
        return {"n": float(len(self.flags)), "degenerate_fraction": self.fraction}


def kl_k3(logp_policy: npt.NDArray[np.float64], logp_ref: npt.NDArray[np.float64]) -> float:
    """Schulman's k3 estimator of KL(policy || reference), per token, averaged.

        k3 = exp(r) - r - 1,   r = logp_ref - logp_policy

    Unbiased, and non-negative on every sample -- unlike the naive `logp_policy -
    logp_ref`, which is unbiased but takes negative values on individual tokens and so
    produces a KL penalty that occasionally PAYS the policy for moving away from the
    reference. That is a small effect on average and a very odd one to debug.

    This is the reference implementation. The torch path in `train/grpo.py` computes the
    same quantity and `test_kl_matches_reference_implementation` asserts they agree on a
    fixed batch.
    """
    a = np.asarray(logp_policy, dtype=np.float64)
    b = np.asarray(logp_ref, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"logprob shapes differ: {a.shape} vs {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite log-probabilities [I11]")
    r = b - a
    return float(np.mean(np.exp(r) - r - 1.0))


def clipped_surrogate(
    logp: npt.NDArray[np.float64],
    logp_old: npt.NDArray[np.float64],
    advantages: npt.NDArray[np.float64],
    clip_eps: float = 0.2,
) -> float:
    """PPO-style clipped objective, per token. Reference implementation; higher is better.

    `advantages` is broadcast across every token of the sequence it belongs to -- one
    episode-level advantage, assigned uniformly to the tokens that produced the episode.
    That is the standard GRPO credit assignment and it is worth naming as an assumption
    rather than an accident: it says a good episode makes every one of its turns slightly
    more likely, including the turns that were incidental to why it was good.
    """
    ratio = np.exp(np.asarray(logp) - np.asarray(logp_old))
    adv = np.asarray(advantages, dtype=np.float64)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return float(np.mean(np.minimum(unclipped, clipped)))
