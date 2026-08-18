"""I7: terminal scoring uses a strictly proper rule over a fixed flat label set."""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.data.taxonomy import check_flat
from dxenv.reward.scoring import (
    ScoringError,
    brier_score,
    distribution_to_vector,
    score_bounds,
    severity_weight,
    terminal_diagnosis_score,
)


def test_brier_is_proper() -> None:
    """THE most important test in the repo.

    For random true beliefs q, the expected score under q must be strictly higher for
    reporting q than for reporting any p != q. This is what mathematically rules out
    hedging, rather than penalising it heuristically -- and it is what licenses
    `bayes.bayes_optimal_value` skipping the maximisation entirely.
    """
    rng = np.random.default_rng(0)
    n = 40
    for _ in range(50):
        q = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 3.0))
        eq = float(sum(q[i] * brier_score(q, i) for i in range(n)))
        for _ in range(20):
            mix = rng.uniform(0.05, 0.95)
            p = mix * q + (1 - mix) * rng.dirichlet(np.ones(n))
            ep = float(sum(q[i] * brier_score(p, i) for i in range(n)))
            assert ep < eq + 1e-12, "reporting a different distribution scored higher"


def test_brier_bounded_and_finite(taxonomy) -> None:
    n = len(taxonomy)
    rng = np.random.default_rng(1)
    lo, hi = -1.0 - 1.0 / n, 1.0
    for _ in range(300):
        p = rng.dirichlet(np.ones(n) * 0.3)
        s = brier_score(p, int(rng.integers(n)))
        assert np.isfinite(s)
        assert lo - 1e-9 <= s <= hi + 1e-9


def test_uniform_report_scores_exactly_zero(taxonomy) -> None:
    n = len(taxonomy)
    assert brier_score(np.full(n, 1.0 / n), 0) == pytest.approx(0.0, abs=1e-12)


def test_more_mass_on_truth_scores_higher(taxonomy) -> None:
    n = len(taxonomy)
    prev = -np.inf
    for mass in (0.01, 0.1, 0.3, 0.6, 0.9, 0.999):
        p = np.full(n, (1.0 - mass) / (n - 1))
        p[5] = mass
        s = brier_score(p, 5)
        assert s > prev
        prev = s


def test_flat_distribution_scores_below_confident_correct(taxonomy) -> None:
    n = len(taxonomy)
    flat = np.full(n, 1.0 / n)
    confident = np.zeros(n)
    confident[7] = 1.0
    assert brier_score(flat, 7) < brier_score(confident, 7)


def test_severity_weight_orders_correctly(taxonomy) -> None:
    """Missing a high-urgency condition costs more than missing a benign one."""
    by_tier: dict[int, str] = {}
    for lab in taxonomy.labels:
        by_tier.setdefault(lab.urgency, lab.slug)

    # Hold the report's shape RELATIVE TO THE TRUTH fixed across tiers: 0.02 on the true
    # condition, 0.98 on one decoy. The raw Brier value is then identical for every tier,
    # so the only thing that varies is the severity weight -- which is what we are
    # testing. Varying the distribution too would confound the two.
    penalties = {}
    for tier, slug in sorted(by_tier.items()):
        decoy = next(s for s in taxonomy.slugs if s != slug)
        penalties[tier] = terminal_diagnosis_score({slug: 0.02, decoy: 0.98}, slug)
    assert all(v < 0 for v in penalties.values()), "the probe report must be a BAD one"
    tiers = sorted(penalties)
    from itertools import pairwise

    for a, b in pairwise(tiers):
        assert penalties[b] < penalties[a], (
            f"missing a tier-{b} condition must cost more than a tier-{a} one"
        )
    assert severity_weight(by_tier[4]) > severity_weight(by_tier[1])


def test_label_set_is_flat(taxonomy) -> None:
    """Hierarchy is what lets an agent hedge upward and collect partial credit."""
    assert check_flat(taxonomy) == []


def test_label_set_frozen(taxonomy) -> None:
    taxonomy.assert_frozen()


def test_unknown_label_in_report_raises(taxonomy) -> None:
    """Silently dropping it would renormalise behind the agent's back."""
    with pytest.raises(Exception, match="unknown condition slug"):
        distribution_to_vector({"not_a_real_condition": 1.0}, taxonomy)


def test_unnormalised_report_raises(taxonomy) -> None:
    with pytest.raises(ScoringError, match="sums to"):
        distribution_to_vector({taxonomy.slugs[0]: 0.4, taxonomy.slugs[1]: 0.4}, taxonomy)


def test_score_bounds_are_finite(taxonomy) -> None:
    lo, hi = score_bounds(taxonomy)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < 0 < hi
