"""Rejection sampling for the SFT set.

CLAUDE.md 8.3, and the sentence that determines the whole design:

> **Filtering on correct diagnosis alone selects for lucky guesses and shotgun
> test-ordering** -- a trajectory that got it right after 40 tests is a bad
> demonstration, and that habit is stubborn once trained in.

So nothing here filters on correctness. Four filters, each closing a different way of
being accidentally right:

  reward           the full reward, costs included. A correct answer after 40 tests
                   scores below a correct answer after 3, and the filter sees that.
  process          did each test move the posterior toward the truth? Computed from the
                   evidence sequence alone -- it cannot see the final report, which is
                   what makes it a check on *reasoning* rather than a second check on
                   the outcome.
  reproducibility  correct once in eight is luck. A patient contributes nothing unless
                   several of its k samples clear the bar.
  balance          otherwise the SFT set is whatever the generator emits most, and the
                   model learns the prior instead of the reasoning.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.obs_model import ObservationModel, ResultValue, build_observation_model
from dxenv.policy.rollout import Rollout

DEFAULT_MIN_REPRODUCIBLE: Final = 3
"""Of k samples, how many must clear the reward bar before the patient contributes.

At k=8 this is a 3-in-8 requirement. One-in-eight is luck and the filter must say so
(`test_filter_rejects_lucky_single_sample`); eight-in-eight would keep only the patients
the policy already finds easy, which is the half of the corpus with nothing left to
teach."""


class RejectionError(ValueError):
    """Malformed filter configuration. Never caught inside `dxenv.policy`."""


@dataclass(frozen=True, slots=True)
class RejectionConfig:
    min_reward: float = 0.0
    min_process_fraction: float = 0.5
    """Fraction of charged tests that must have moved the posterior toward the truth.

    Not 1.0. A competent diagnostician orders a test that comes back reassuringly normal
    and correctly rules something out -- for the true condition that step moves mass the
    wrong way while being exactly the right action. Demanding every step help would
    select for trajectories that got lucky with their evidence."""

    max_tests: int = 8
    min_reproducible: int = DEFAULT_MIN_REPRODUCIBLE
    max_per_condition: int | None = None
    require_diagnosis: bool = True
    """Abstentions are seeded separately (CLAUDE.md 8.4) rather than filtered in here,
    because an abstention that clears a reward bar is usually a hard patient rather than
    a demonstration of good judgment."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_process_fraction <= 1.0:
            raise RejectionError("min_process_fraction must be in [0, 1]")
        if self.min_reproducible < 1:
            raise RejectionError("min_reproducible must be >= 1")


@dataclass(frozen=True, slots=True)
class Verdict:
    accepted: bool
    reasons: tuple[str, ...] = ()
    process_fraction: float = 0.0

    def line(self) -> str:
        return ("ACCEPT" if self.accepted else "REJECT " + "; ".join(self.reasons))


def ordered_evidence(
    trajectory: Mapping[str, Any], catalog: Catalog | None = None
) -> list[tuple[str, ResultValue]]:
    """The analytes the agent PAID for, in the order it revealed them.

    Reads only the `order_test` steps. It never touches the terminal report -- that
    separation is the point of `test_process_filter_uses_posterior_not_outcome`, and it
    is structural here rather than a promise made in a comment.
    """
    cat = catalog or load_catalog()
    out: list[tuple[str, ResultValue]] = []
    for step in trajectory["steps"]:
        action = step["action"]
        if action["kind"] != "order_test" or float(step.get("cost_charged", 0.0)) <= 0.0:
            continue
        revealed = {r["analyte"]: r for r in step.get("revealed", [])}
        for analyte in cat.test(action["test_key"]).analytes:
            r = revealed.get(analyte)
            if r is None:
                continue
            value: ResultValue = (
                float(r["value_number"]) if r.get("value_number") is not None
                else str(r["value_code"])
            )
            out.append((analyte, value))
    return out


def process_fraction(
    evidence: Sequence[tuple[str, ResultValue]],
    true_condition: str,
    taxonomy: Taxonomy | None = None,
    model: ObservationModel | None = None,
) -> float:
    """Fraction of paid tests after which the true condition's posterior mass ROSE.

    Takes the evidence sequence, not the trajectory, so there is no argument through
    which the declared distribution could arrive. A process filter that can see the
    outcome is an outcome filter wearing a lab coat.

    Returns 1.0 for an empty sequence: a policy that ordered nothing has no process to
    fault, and its reward already reflects whether that was wise.
    """
    if not evidence:
        return 1.0
    from dxenv.env.bayes import posterior_sequence

    tax = taxonomy or load_taxonomy()
    m = model or build_observation_model()
    idx = tax.index(true_condition)
    beliefs = posterior_sequence(list(evidence), model=m)
    moves = [
        float(beliefs[i + 1][idx]) > float(beliefs[i][idx]) for i in range(len(beliefs) - 1)
    ]
    return float(sum(moves) / len(moves))


def judge(
    rollout: Rollout,
    cfg: RejectionConfig,
    taxonomy: Taxonomy | None = None,
    catalog: Catalog | None = None,
    model: ObservationModel | None = None,
) -> Verdict:
    """Per-rollout filters. Reproducibility and balance are group-level; see below."""
    reasons: list[str] = []
    if cfg.require_diagnosis and not rollout.diagnosed:
        reasons.append(f"terminated on {rollout.breakdown.termination_reason}")
    if rollout.reward < cfg.min_reward:
        reasons.append(f"reward {rollout.reward:+.3f} below {cfg.min_reward:+.3f}")
    if rollout.n_tests > cfg.max_tests:
        reasons.append(
            f"{rollout.n_tests} tests exceeds {cfg.max_tests} -- a correct answer after "
            "a shotgun sweep is a bad demonstration, not a good one"
        )
    frac = process_fraction(
        ordered_evidence(rollout.trajectory, catalog), rollout.condition, taxonomy, model
    )
    if frac < cfg.min_process_fraction:
        reasons.append(
            f"only {frac:.2f} of its tests moved the posterior toward the truth "
            f"(need {cfg.min_process_fraction:.2f})"
        )
    return Verdict(accepted=not reasons, reasons=tuple(reasons), process_fraction=frac)


@dataclass(frozen=True, slots=True)
class GroupDecision:
    """What survived from one patient's k samples, and why the rest did not."""

    patient_id: str
    condition: str
    accepted: tuple[Rollout, ...]
    n_sampled: int
    n_passing: int
    reproducible: bool
    reasons: tuple[str, ...] = ()

    @property
    def best(self) -> Rollout | None:
        return max(self.accepted, key=lambda r: r.reward) if self.accepted else None


def filter_group(
    rollouts: Sequence[Rollout],
    cfg: RejectionConfig,
    taxonomy: Taxonomy | None = None,
    catalog: Catalog | None = None,
    model: ObservationModel | None = None,
    keep: int = 1,
) -> GroupDecision:
    """Apply every filter to one patient's group. Reproducibility gates the whole group.

    A patient whose k samples clear the bar only once contributed a coin flip, and
    training on the winning flip teaches the model that the flip was skill.
    """
    if not rollouts:
        raise RejectionError("empty group")
    verdicts = [judge(r, cfg, taxonomy, catalog, model) for r in rollouts]
    passing = [r for r, v in zip(rollouts, verdicts, strict=True) if v.accepted]
    reproducible = len(passing) >= cfg.min_reproducible
    reasons: list[str] = []
    if not reproducible:
        reasons.append(
            f"only {len(passing)}/{len(rollouts)} samples passed; "
            f"{cfg.min_reproducible} required -- one success in k is luck"
        )
    accepted = tuple(sorted(passing, key=lambda r: -r.reward)[:keep]) if reproducible else ()
    return GroupDecision(
        patient_id=rollouts[0].patient_id,
        condition=rollouts[0].condition,
        accepted=accepted,
        n_sampled=len(rollouts),
        n_passing=len(passing),
        reproducible=reproducible,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class BalanceReport:
    counts: dict[str, int]
    dropped: int
    target: int

    @property
    def imbalance(self) -> float:
        """max/mean over the conditions actually present. 1.0 is perfectly flat."""
        if not self.counts:
            return 1.0
        v = np.array(list(self.counts.values()), dtype=np.float64)
        return float(v.max() / v.mean())


def balance_conditions(
    rollouts: Sequence[Rollout],
    rng: np.random.Generator,
    max_per_condition: int | None = None,
    taxonomy: Taxonomy | None = None,
) -> tuple[list[Rollout], BalanceReport]:
    """Cap each condition's share of the SFT set.

    Without this the set is dominated by whatever the generator emits most, and the model
    learns the prior instead of the reasoning -- it will report `essential_hypertension`
    at a rate that looks like calibration on the training distribution and is simply the
    marginal.

    Capping keeps the HIGHEST-reward examples per condition rather than a random subset:
    the cap is there to flatten the label distribution, not to throw away the best
    demonstrations of the labels that happen to be common.
    """
    tax = taxonomy or load_taxonomy()
    by_condition: dict[str, list[Rollout]] = {}
    for r in rollouts:
        by_condition.setdefault(r.condition, []).append(r)
    if max_per_condition is None:
        present = len(by_condition) or 1
        max_per_condition = max(1, math.ceil(len(rollouts) / present))

    kept: list[Rollout] = []
    counts: dict[str, int] = {}
    for condition, group in sorted(by_condition.items()):
        # Sort by reward, then by seed, so the cap is deterministic given the inputs
        # [I10] -- `rng` is here for callers who want a random tiebreak and is used only
        # to shuffle the final order, never to choose which examples survive.
        chosen = sorted(group, key=lambda r: (-r.reward, r.seed))[:max_per_condition]
        kept.extend(chosen)
        counts[condition] = len(chosen)
    unmapped = sorted(set(counts) - set(tax.slugs))
    if unmapped:
        raise RejectionError(f"rollouts carry labels outside the taxonomy: {unmapped}")
    order = rng.permutation(len(kept))
    return [kept[int(i)] for i in order], BalanceReport(
        counts=counts, dropped=len(rollouts) - len(kept), target=max_per_condition
    )


@dataclass(slots=True)
class RejectionStats:
    """Why things were dropped. A rejection rate without reasons is not actionable."""

    groups: int = 0
    groups_reproducible: int = 0
    rollouts: int = 0
    rollouts_passing: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)

    def note(self, verdict: Verdict) -> None:
        self.rollouts += 1
        if verdict.accepted:
            self.rollouts_passing += 1
        for r in verdict.reasons:
            head = r.split(" ")[0]
            self.reason_counts[head] = self.reason_counts.get(head, 0) + 1

    def render(self) -> str:
        rate = self.rollouts_passing / max(self.rollouts, 1)
        lines = [
            f"rejection sampling: {self.rollouts_passing}/{self.rollouts} rollouts passed "
            f"({rate:.1%}); {self.groups_reproducible}/{self.groups} patients reproducible",
        ]
        lines.extend(
            f"  dropped on {k}: {v}"
            for k, v in sorted(self.reason_counts.items(), key=lambda kv: -kv[1])
        )
        return "\n".join(lines)
