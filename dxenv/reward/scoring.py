"""Terminal diagnosis scoring: Brier, plus severity weights [I7].

Brier, not log-loss. Log-loss is unbounded below, so one rollout that puts ~0 on the
truth produces a huge negative that wrecks GRPO's advantage normalisation for the whole
group [I11]. Brier is bounded on both sides and that boundedness is the point.

The rule is shifted so that reporting the uniform distribution scores exactly 0:

    score(p, c) = (1 - 1/n) - sum_i (p_i - [i == c])^2

A positive shift is an affine transform with positive slope, so the rule stays STRICTLY
PROPER -- the unique maximiser of expected score is reporting your true belief. That is
what rules out hedging mathematically rather than penalising it heuristically, and
`test_brier_is_proper` is the check that it survived every refactor.

Range, for n = 149:
    confident and correct : +0.993
    uniform              :  0.000
    confident and wrong  : -1.007
"""

from __future__ import annotations

from collections.abc import Callable

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import yaml

from dxenv.data.taxonomy import Taxonomy, load_taxonomy

_CONFIG_DIR: Final = Path(__file__).resolve().parents[1] / "configs"


class ScoringError(ValueError):
    """Malformed belief or severity table. Never caught inside `dxenv.reward`."""


@dataclass(frozen=True, slots=True)
class SeverityTable:
    weights: dict[int, float]

    def weight(self, urgency: int) -> float:
        try:
            return self.weights[urgency]
        except KeyError as exc:
            raise ScoringError(f"no severity weight for urgency tier {urgency}") from exc


@lru_cache(maxsize=4)
def load_severity(path: Path | None = None) -> SeverityTable:
    with (path or _CONFIG_DIR / "severity.yaml").open() as fh:
        raw = yaml.safe_load(fh)
    weights = {int(t): float(spec["weight"]) for t, spec in raw["tiers"].items()}
    if not weights:
        raise ScoringError("severity.yaml declares no tiers")
    if any(w <= 0.0 for w in weights.values()):
        raise ScoringError("severity weights must be positive")
    return SeverityTable(weights=weights)


def uniform_offset(n: int) -> float:
    """The shift that puts a uniform report at exactly 0."""
    if n < 2:
        raise ScoringError("scoring needs at least 2 labels")
    return 1.0 - 1.0 / n


def brier_score(p: npt.NDArray[np.float64], true_idx: int) -> float:
    """Strictly proper, bounded, uniform-report-is-zero. Higher is better."""
    if p.ndim != 1:
        raise ScoringError("belief must be 1-D")
    n = p.size
    if not 0 <= true_idx < n:
        raise ScoringError(f"true index {true_idx} out of range for {n} labels")
    if not np.isfinite(p).all():
        raise ScoringError("belief contains NaN or inf [I11]")
    if (p < -1e-12).any():
        raise ScoringError("belief has negative mass")
    total = float(p.sum())
    if abs(total - 1.0) > 1e-6:
        raise ScoringError(f"belief sums to {total}, not 1")
    loss = float(np.sum(p**2)) - 2.0 * float(p[true_idx]) + 1.0
    return uniform_offset(n) - loss


def distribution_to_vector(
    distribution: Mapping[str, float], taxonomy: Taxonomy | None = None
) -> npt.NDArray[np.float64]:
    """Map a slug->probability report onto the canonical label ordering.

    Unlisted labels get exactly zero -- that is the agent's assertion, not a default.
    An unknown slug raises: silently dropping it would renormalise the report behind the
    agent's back and score something it never said.
    """
    tax = taxonomy or load_taxonomy()
    vec = np.zeros(len(tax), dtype=np.float64)
    for slug, prob in distribution.items():
        vec[tax.index(slug)] = float(prob)
    total = float(vec.sum())
    if abs(total - 1.0) > 1e-6:
        raise ScoringError(f"reported distribution sums to {total}, not 1")
    return vec


def severity_weight(
    condition: str, taxonomy: Taxonomy | None = None, severity: SeverityTable | None = None
) -> float:
    """Weight keyed on the TRUE condition.

    Keying on the true condition rather than the reported one is what stops an agent
    inflating its score by preferentially naming high-urgency conditions.
    """
    tax = taxonomy or load_taxonomy()
    sev = severity or load_severity()
    return sev.weight(tax.get(condition).urgency)


def terminal_diagnosis_score(
    distribution: Mapping[str, float],
    true_condition: str,
    taxonomy: Taxonomy | None = None,
    severity: SeverityTable | None = None,
) -> float:
    tax = taxonomy or load_taxonomy()
    vec = distribution_to_vector(distribution, tax)
    raw = brier_score(vec, tax.index(true_condition))
    return raw * severity_weight(true_condition, tax, severity)


def score_bounds(
    taxonomy: Taxonomy | None = None, severity: SeverityTable | None = None
) -> tuple[float, float]:
    """(min, max) attainable terminal diagnosis score. Used to assert finiteness [I11]."""
    tax = taxonomy or load_taxonomy()
    sev = severity or load_severity()
    n = len(tax)
    w_max = max(sev.weights.values())
    best = uniform_offset(n) * w_max
    worst = (uniform_offset(n) - 2.0) * w_max
    return worst, best


def weighted_score_fn(
    taxonomy: Taxonomy | None = None, severity: SeverityTable | None = None
) -> Callable[[npt.NDArray[np.float64], int], float]:
    """The terminal scoring rule as a `(report, true_idx) -> float` callable.

    This is the function `env/bayes.py` takes as its injected `ScoreFn`, and it is what
    every ceiling in the project is computed against. It exists so there is exactly ONE
    severity-weighted scorer rather than a copy at each of the four call sites -- a copy
    that drifts turns the ceiling from a bound into a number that resembles one.
    """
    tax = taxonomy or load_taxonomy()
    sev = severity or load_severity()
    weights = np.array([sev.weight(lab.urgency) for lab in tax.labels], dtype=np.float64)

    def fn(report: npt.NDArray[np.float64], true_idx: int) -> float:
        return brier_score(report, true_idx) * float(weights[true_idx])

    return fn
