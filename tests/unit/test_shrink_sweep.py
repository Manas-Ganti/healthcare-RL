"""The shrink sweep, and the detector inside it.

The sweep exists to split a diagnosis deficit into miscalibration and wrong ranking. Its
whole value is the three-way reading, so the reading is what gets tested -- including the
two cases where it must DECLINE to fire, which are the ones it got wrong first.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest


def _mod() -> Any:
    spec = importlib.util.spec_from_file_location("shrink_sweep", "scripts/shrink_sweep.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sweep(points: dict[float, float]) -> list[dict[str, float]]:
    return [{"alpha": a, "mean_total": v, "mean_diagnosis": v} for a, v in sorted(points.items())]


def test_shrink_endpoints_are_the_report_and_the_prior() -> None:
    m = _mod()
    prior = {"a": 0.5, "b": 0.3, "c": 0.2}
    report = {"a": 0.9, "b": 0.1, "c": 0.0}

    at_one = m.shrink(report, 1.0, prior)
    at_zero = m.shrink(report, 0.0, prior)
    assert at_one == pytest.approx(report)
    assert at_zero == pytest.approx(prior)


def test_shrink_is_a_normalised_convex_combination() -> None:
    m = _mod()
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 12
        prior = dict(zip("abcdefghijkl", rng.dirichlet(np.ones(n)), strict=True))
        report = dict(zip("abcdefghijkl", rng.dirichlet(np.ones(n)), strict=True))
        alpha = float(rng.uniform())
        out = m.shrink(report, alpha, prior)
        assert sum(out.values()) == pytest.approx(1.0)
        assert all(v >= 0.0 for v in out.values())
        for k, pk in prior.items():
            assert out[k] == pytest.approx((1 - alpha) * pk + alpha * report[k], abs=1e-9)


def test_shrink_covers_the_whole_label_set_not_only_named_labels() -> None:
    """A report naming two labels must still be mixed against the full prior."""
    m = _mod()
    prior = {"a": 0.5, "b": 0.3, "c": 0.2}
    out = m.shrink({"a": 1.0}, 0.5, prior)
    assert set(out) == set(prior)
    assert out["c"] == pytest.approx(0.1)


def test_reweighted_leaves_a_trajectory_without_a_report_alone() -> None:
    """An abstain contributes the same score at every alpha rather than distorting."""
    m = _mod()
    traj = {"steps": [{"turn": 0, "action": {"kind": "abstain", "action_id": "x"}}]}
    assert m.reweighted(traj, 0.0, {"a": 1.0}) is traj


def test_reweighted_does_not_mutate_its_input() -> None:
    m = _mod()
    prior = {"a": 0.5, "b": 0.5}
    traj = {
        "steps": [{"turn": 0,
                   "action": {"kind": "diagnose", "action_id": "d",
                              "distribution": {"a": 1.0, "b": 0.0}}}],
    }
    before = dict(traj["steps"][0]["action"]["distribution"])
    out = m.reweighted(traj, 0.0, prior)
    assert traj["steps"][0]["action"]["distribution"] == before
    assert out["steps"][0]["action"]["distribution"]["a"] == pytest.approx(0.5)


def test_reading_fires_on_an_informative_but_overconfident_policy() -> None:
    """The case the sweep is FOR: a real interior peak clearing both endpoints."""
    m = _mod()
    verdict, beats_prior, beats_as_ran = m.classify(
        _sweep({0.0: -0.02, 0.3: 0.20, 0.5: 0.35, 0.8: 0.10, 1.0: -0.35})
    )
    assert verdict == "calibration"
    assert beats_prior == pytest.approx(0.37)
    assert beats_as_ran == pytest.approx(0.70)


def test_reading_declines_on_a_policy_with_no_model_behind_it() -> None:
    """`random_schema`'s real shape: peaks at alpha=0.1, worth +0.004 over the prior.

    Read literally that is an interior peak. Reported as "the ranking carries real
    information" it would be a detector firing on the one row built to contain nothing,
    which is how a detector stops being believed.
    """
    m = _mod()
    verdict, _, _ = m.classify(
        _sweep({0.0: -0.062, 0.1: -0.058, 0.5: -0.127, 1.0: -0.325})
    )
    assert verdict == "knowledge"


def test_reading_declines_on_exact_bayes() -> None:
    """`greedy_bayes`'s real shape: peaks at alpha=0.9, worth +0.001 over its own report.

    Exact Bayes over the true observation model is the best-calibrated policy in the repo.
    Calling it overconfident would invert the finding.
    """
    m = _mod()
    verdict, _, _ = m.classify(
        _sweep({0.0: -0.199, 0.5: 0.611, 0.9: 0.840, 1.0: 0.839})
    )
    assert verdict == "ranking"


def test_reading_requires_both_endpoints_to_be_cleared() -> None:
    """Clearing one endpoint is not enough; the margin is deliberately two-sided."""
    m = _mod()
    # Beats the prior handsomely, but only ties its own report: confidence is fine.
    assert m.classify(_sweep({0.0: -0.30, 0.5: 0.401, 1.0: 0.40}))[0] == "ranking"
    # Beats its own report handsomely, but only ties the prior: the ranking is the problem.
    assert m.classify(_sweep({0.0: 0.400, 0.5: 0.401, 1.0: -0.30}))[0] == "knowledge"


def test_alpha_grid_must_contain_both_endpoints() -> None:
    """Every reading is stated against alpha=0 and alpha=1, so both must be measured."""
    m = _mod()
    with pytest.raises(StopIteration):
        m.classify(_sweep({0.2: 0.1, 0.5: 0.2}))
