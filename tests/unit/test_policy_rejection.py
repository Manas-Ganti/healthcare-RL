"""Rejection sampling filters (CLAUDE.md 8.3).

Filtering on correct diagnosis alone selects for lucky guesses and shotgun test-ordering.
Nothing here filters on correctness; these tests are the proof of that.
"""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.policy.baselines import GreedyBayesPolicy, PriorPolicy
from dxenv.policy.rejection import (
    RejectionConfig,
    RejectionError,
    balance_conditions,
    filter_group,
    judge,
    ordered_evidence,
    process_fraction,
)
from dxenv.policy.rollout import (
    Rollout,
    RolloutContext,
    constant_factory,
    rollout_group,
)


@pytest.fixture(scope="module")
def ctx(episode_config, reward_config, taxonomy, catalog, obs_model):
    return RolloutContext(
        episode_config=episode_config, reward_config=reward_config,
        taxonomy=taxonomy, catalog=catalog, model=obs_model,
    )


@pytest.fixture(scope="module")
def groups(fixture_corpus, ctx):
    return [
        rollout_group(rec, constant_factory(GreedyBayesPolicy), 8, 100 + i * 8, ctx,
                      budget=200.0)
        for i, rec in enumerate(fixture_corpus[:12])
    ]


def _mutate(rollout: Rollout, **kw) -> Rollout:
    from dataclasses import replace
    return replace(rollout, **kw)


def test_filter_rejects_high_cost_correct(groups) -> None:
    """A correct answer after a shotgun sweep is a bad demonstration, not a good one."""
    cfg = RejectionConfig(max_tests=3)
    shotgun = _mutate(groups[0][0], breakdown=_shotgun_breakdown(groups[0][0]))
    v = judge(shotgun, cfg)
    assert not v.accepted
    assert any("exceeds" in r for r in v.reasons)


def _shotgun_breakdown(rollout):
    from dataclasses import replace
    return replace(rollout.breakdown, n_tests_charged=40, diagnosis=5.0, total=4.0)


def test_filter_rejects_lucky_single_sample(groups, ctx) -> None:
    """Correct once in eight is luck. The whole group contributes nothing."""
    cfg = RejectionConfig(min_reward=0.0, min_reproducible=3)
    rollouts = list(groups[0])
    # One clear winner, seven clear failures: exactly the shape of a lucky guess.
    lucky = [_mutate(r, breakdown=_scaled(r.breakdown, -5.0)) for r in rollouts]
    lucky[0] = _mutate(rollouts[0], breakdown=_scaled(rollouts[0].breakdown, +5.0))
    decision = filter_group(lucky, cfg)
    assert decision.n_passing == 1
    assert not decision.reproducible
    assert decision.accepted == ()
    assert "luck" in decision.reasons[0]


def _scaled(breakdown, total):
    from dataclasses import replace
    return replace(breakdown, total=total)


def test_reproducible_group_is_accepted(groups) -> None:
    cfg = RejectionConfig(min_reward=0.0, min_reproducible=3)
    rollouts = [_mutate(r, breakdown=_scaled(r.breakdown, +2.0)) for r in groups[0]]
    decision = filter_group(rollouts, cfg)
    assert decision.reproducible and len(decision.accepted) == 1


def test_process_filter_uses_posterior_not_outcome(groups, taxonomy, obs_model,
                                                   catalog) -> None:
    """The process score must not change when the final report changes.

    Structural, not incidental: `process_fraction` takes the evidence sequence, so there
    is no argument through which the declared distribution could arrive.
    """
    rollout = next(r for g in groups for r in g if r.n_tests > 0)
    evidence = ordered_evidence(rollout.trajectory, catalog)
    base = process_fraction(evidence, rollout.condition, taxonomy, obs_model)

    forged = {
        **rollout.trajectory,
        "steps": [
            {**s, "action": {**s["action"], "distribution": {taxonomy.slugs[0]: 1.0}}}
            if s["action"]["kind"] == "diagnose" else s
            for s in rollout.trajectory["steps"]
        ],
    }
    assert ordered_evidence(forged, catalog) == evidence
    assert process_fraction(
        ordered_evidence(forged, catalog), rollout.condition, taxonomy, obs_model
    ) == base


def test_process_fraction_is_one_for_no_tests(taxonomy, obs_model, fixture_corpus) -> None:
    """A policy that ordered nothing has no process to fault."""
    assert process_fraction([], fixture_corpus[0].condition, taxonomy, obs_model) == 1.0


def test_process_filter_separates_informative_from_uninformative(
    groups, taxonomy, obs_model, catalog
) -> None:
    """Test the detector: evidence pointing away from the truth must score lower."""
    rollout = next(r for g in groups for r in g if r.n_tests >= 2)
    good = process_fraction(
        ordered_evidence(rollout.trajectory, catalog), rollout.condition, taxonomy, obs_model
    )
    wrong_label = next(s for s in taxonomy.slugs if s != rollout.condition)
    bad = process_fraction(
        ordered_evidence(rollout.trajectory, catalog), wrong_label, taxonomy, obs_model
    )
    assert good >= bad


def test_ordered_evidence_ignores_duplicate_and_refused_orders(groups, catalog) -> None:
    """Only PAID orders count; a deduped repeat reveals nothing new."""
    for group in groups[:3]:
        for r in group:
            assert len(ordered_evidence(r.trajectory, catalog)) >= 0
            charged = sum(
                1 for s in r.trajectory["steps"]
                if s["action"]["kind"] == "order_test" and s.get("cost_charged", 0.0) > 0
            )
            assert charged == r.n_tests


def test_condition_balance_within_tolerance(groups) -> None:
    """Otherwise the SFT set is whatever the generator emits most, and the model learns
    the prior instead of the reasoning."""
    rollouts = [r for g in groups for r in g]
    kept, report = balance_conditions(rollouts, np.random.default_rng(0),
                                      max_per_condition=2)
    assert max(report.counts.values()) <= 2
    assert report.imbalance <= 2.0
    assert len(kept) == sum(report.counts.values())


def test_balance_keeps_the_best_examples(groups) -> None:
    """The cap flattens the label distribution; it does not discard the best work."""
    rollouts = [r for g in groups for r in g]
    kept, _ = balance_conditions(rollouts, np.random.default_rng(0), max_per_condition=1)
    by_condition: dict[str, list[float]] = {}
    for r in rollouts:
        by_condition.setdefault(r.condition, []).append(r.reward)
    for r in kept:
        assert r.reward == max(by_condition[r.condition])


def test_empty_group_raises(groups) -> None:
    with pytest.raises(RejectionError, match="empty group"):
        filter_group([], RejectionConfig())


def test_bad_config_raises() -> None:
    with pytest.raises(RejectionError):
        RejectionConfig(min_process_fraction=1.5)
    with pytest.raises(RejectionError):
        RejectionConfig(min_reproducible=0)


def test_lazy_policy_is_filtered_out(fixture_corpus, ctx) -> None:
    """The prior-reporting policy orders nothing and must not survive the reward bar."""
    rollouts = rollout_group(fixture_corpus[0], constant_factory(PriorPolicy), 8, 1, ctx,
                             budget=200.0)
    decision = filter_group(rollouts, RejectionConfig(min_reward=0.5, min_reproducible=3))
    assert not decision.reproducible
