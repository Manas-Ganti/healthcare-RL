"""Budget sweep -> the cost-accuracy frontier.

The environment is budget-conditioned: B ~ p(B) per episode and B is exposed in the
observation, so ONE policy spans the whole frontier and eval traces it by sweeping B.
A policy whose test count does not respond to B is ignoring the constraint and getting
its reward some other way -- `test_policy_behavior_varies_with_budget` is the check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from dxenv.data.corpus import PatientRecord
from dxenv.env.episode import DiagnosticEpisode, EpisodeConfig, load_episode_config
from dxenv.policy.baselines import Policy, run_episode
from dxenv.reward.engine import GroundTruth, RewardConfig, load_reward_config, score_trajectory


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    budget: float
    mean_total: float
    mean_diagnosis: float
    mean_tests: float
    mean_spend: float


@dataclass(frozen=True, slots=True)
class ParetoCurve:
    points: tuple[ParetoPoint, ...]

    def is_broadly_monotone(self, tolerance: float = 0.05) -> bool:
        """Accuracy must not DECREASE with more budget, beyond noise.

        Tolerance, not strict monotonicity: these are sample means, and demanding exact
        monotonicity would fail on noise and train the reader to ignore the check.
        """
        d = [p.mean_diagnosis for p in self.points]
        return all(b >= a - tolerance for a, b in pairwise(d))

    def render(self) -> str:
        rows = [f"{'budget':>8}{'total':>10}{'diagnosis':>11}{'tests':>8}{'spend':>9}"]
        rows += [
            f"{p.budget:>8.0f}{p.mean_total:>10.4f}{p.mean_diagnosis:>11.4f}"
            f"{p.mean_tests:>8.2f}{p.mean_spend:>9.1f}"
            for p in self.points
        ]
        return "\n".join(rows)


def sweep(
    records: Sequence[PatientRecord],
    policy: Policy,
    budgets: Sequence[float] | None = None,
    cfg: EpisodeConfig | None = None,
    rcfg: RewardConfig | None = None,
    seed: int = 0,
) -> ParetoCurve:
    c = cfg or load_episode_config()
    r = rcfg or load_reward_config()
    grid = list(budgets) if budgets is not None else list(c.budget_support)

    points: list[ParetoPoint] = []
    for b in grid:
        totals, diags, tests, spends = [], [], [], []
        for i, rec in enumerate(records):
            ep = DiagnosticEpisode(rec, seed=seed + i, config=c, budget=float(b))
            traj = run_episode(ep, policy)
            br = score_trajectory(
                traj, GroundTruth(rec.condition, rec.analytes, rec.allergies), r
            )
            totals.append(br.total)
            diags.append(br.diagnosis)
            tests.append(br.n_tests_charged)
            spends.append(float(traj["spent"]))
        points.append(ParetoPoint(
            budget=float(b),
            mean_total=float(np.mean(totals)),
            mean_diagnosis=float(np.mean(diags)),
            mean_tests=float(np.mean(tests)),
            mean_spend=float(np.mean(spends)),
        ))
    return ParetoCurve(points=tuple(points))
