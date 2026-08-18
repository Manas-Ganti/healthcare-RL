"""Audit-suite tests. Every probe is both a test and a reported result (CLAUDE.md 10)."""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.eval.audit import (
    CLINICAL_SPOT_CHECKS,
    probe_bayes_ceiling,
    probe_blank_record_baseline,
    probe_counterfactual_perturbation,
    probe_held_out_modules,
    probe_leakage_ablation,
    probe_no_test_ablation,
    probe_shuffled_labels,
    run_audit,
)


@pytest.fixture(scope="module")
def sample(request):
    from dxenv.data.corpus import generate_corpus

    return generate_corpus(30, seed=4242)


def test_blank_record_baseline(sample, episode_config, reward_config) -> None:
    assert probe_blank_record_baseline(sample, episode_config, reward_config).passed


def test_no_test_ablation(sample, episode_config, reward_config) -> None:
    o = probe_no_test_ablation(sample, episode_config, reward_config)
    assert o.passed, o.detail
    assert o.metrics["gain"] > 0.2


def test_leakage_ablation(sample, episode_config, reward_config) -> None:
    o = probe_leakage_ablation(sample, episode_config, reward_config)
    assert o.passed, o.detail
    assert o.metrics["blocked_surviving"] == 0.0


def test_leak_probe_positive_control_is_large(sample, episode_config, reward_config) -> None:
    """The probe must be able to SEE a leak, or its pass means nothing."""
    o = probe_leakage_ablation(sample, episode_config, reward_config)
    assert o.metrics["control_gain"] > 1.0


def test_shuffled_labels(sample, episode_config, reward_config) -> None:
    o = probe_shuffled_labels(sample, episode_config, reward_config)
    assert o.passed, o.detail
    assert o.metrics["shuffled"] < 0.1


def test_counterfactual_perturbation(obs_model, catalog) -> None:
    from dxenv.data.corpus import generate_corpus

    o = probe_counterfactual_perturbation(generate_corpus(10, seed=1), obs_model, catalog)
    assert o.passed, o.detail
    assert o.metrics["generic_rate"] == 1.0


def test_clinical_spot_checks_reference_real_things(taxonomy, catalog) -> None:
    for analyte, _value, condition in CLINICAL_SPOT_CHECKS:
        catalog.analyte(analyte)
        taxonomy.index(condition)


def test_bayes_ceiling_probe(sample, episode_config, reward_config) -> None:
    o = probe_bayes_ceiling(sample, episode_config, reward_config)
    assert o.passed, o.detail
    assert o.metrics["headroom"] >= -1e-9


def test_held_out_modules_reports_a_gap(sample, episode_config, reward_config) -> None:
    o = probe_held_out_modules(sample, episode_config, reward_config)
    assert o.passed
    assert "gap" in o.metrics or "n_held" in o.metrics


def test_audit_suite_runs_end_to_end_on_fixture(episode_config, reward_config) -> None:
    """10-patient fixture, fast, runs in CI."""
    from dxenv.data.corpus import generate_corpus

    report = run_audit(generate_corpus(10, seed=99), episode_config, reward_config)
    assert len(report.outcomes) == 7
    assert isinstance(report.render(), str)


def test_lazy_policy_scores_below_working_policy(sample, episode_config, reward_config) -> None:
    """Both directions of the gap, as CLAUDE.md 7.7 requires.

    Guess-the-prior must be clearly worse than doing the work -- but not CATASTROPHICALLY
    worse, because a huge penalty for not-knowing produces reckless overconfidence early
    in training, when not-knowing is the correct state.
    """
    from dxenv.eval.audit import _score
    from dxenv.policy.baselines import GreedyBayesPolicy, PriorPolicy

    lazy = _score(sample, PriorPolicy(), episode_config, reward_config)
    working = _score(sample, GreedyBayesPolicy(), episode_config, reward_config)
    assert working["total"] > lazy["total"] + 0.5, "doing the work must pay"
    assert lazy["total"] > -3.0, (
        "the lazy policy is punished so hard that early training would be pushed toward "
        "reckless confident guessing"
    )


def test_abstain_priced_between_correct_and_incorrect(reward_config, taxonomy) -> None:
    from dxenv.reward.scoring import terminal_diagnosis_score

    slug = taxonomy.slugs[0]
    decoy = taxonomy.slugs[1]
    correct = terminal_diagnosis_score({slug: 0.95, decoy: 0.05}, slug, taxonomy)
    wrong = terminal_diagnosis_score({decoy: 0.95, slug: 0.05}, slug, taxonomy)
    abstain = reward_config.abstain_value - reward_config.abstain_penalty
    assert wrong < abstain < correct


def test_policy_behavior_varies_with_budget(episode_config, reward_config) -> None:
    """A policy whose test count ignores B is getting its reward another way."""
    from dxenv.data.corpus import generate_corpus
    from dxenv.eval.pareto import sweep
    from dxenv.policy.baselines import GreedyBayesPolicy

    curve = sweep(generate_corpus(20, seed=8), GreedyBayesPolicy(max_tests=12),
                  budgets=[10.0, 50.0, 200.0], cfg=episode_config, rcfg=reward_config)
    counts = [p.mean_tests for p in curve.points]
    assert counts[-1] > counts[0] + 1.0, f"test count barely moved with budget: {counts}"


def test_pareto_sweep_covers_budget_range(episode_config, reward_config) -> None:
    from dxenv.data.corpus import generate_corpus
    from dxenv.eval.pareto import sweep
    from dxenv.policy.baselines import VitalsOnlyPolicy

    curve = sweep(generate_corpus(8, seed=3), VitalsOnlyPolicy(), cfg=episode_config,
                  rcfg=reward_config)
    assert [p.budget for p in curve.points] == list(episode_config.budget_support)


def test_pareto_is_broadly_monotone(episode_config, reward_config) -> None:
    from dxenv.data.corpus import generate_corpus
    from dxenv.eval.pareto import sweep
    from dxenv.policy.baselines import GreedyBayesPolicy

    curve = sweep(generate_corpus(25, seed=11), GreedyBayesPolicy(max_tests=12),
                  budgets=[10.0, 25.0, 50.0, 100.0], cfg=episode_config, rcfg=reward_config)
    assert curve.is_broadly_monotone(), curve.render()


def test_calibration_metrics_match_reference() -> None:
    """Synthetic sets with KNOWN calibration."""
    from dxenv.eval.calibration import expected_calibration_error, sharpen

    rng = np.random.default_rng(0)
    n, k = 4000, 5

    # Perfectly calibrated: sample the label FROM the reported distribution.
    probs = rng.dirichlet(np.ones(k) * 0.6, size=n)
    truth = [int(rng.choice(k, p=probs[i])) for i in range(n)]
    report = expected_calibration_error(probs, truth)
    assert report.ece < 0.05, report.render()

    # Overconfident: report one-hot on the argmax of the same distribution.
    over = expected_calibration_error(sharpen(probs), truth)
    assert over.ece > report.ece
    assert over.overconfident
    assert over.mean_confidence == pytest.approx(1.0)


def test_calibration_rejects_bad_input() -> None:
    from dxenv.eval.calibration import expected_calibration_error

    with pytest.raises(ValueError, match="NaN"):
        expected_calibration_error(np.array([[np.nan, 1.0]]), [0])
    with pytest.raises(ValueError, match="differ in length"):
        expected_calibration_error(np.array([[0.5, 0.5]]), [0, 1])
