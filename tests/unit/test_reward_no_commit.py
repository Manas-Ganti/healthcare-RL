"""Pricing of episodes the policy did not end: timeout and decode failure.

GRPO run 1 learned to run out the clock because a timeout cost exactly what a deliberate
abstain cost. These tests pin the ordering that closes that hedge.
"""

from __future__ import annotations

import dataclasses

import pytest
from dxenv.reward.engine import (
    NO_COMMIT_REASONS,
    GroundTruth,
    RewardError,
    score_trajectory,
    validate_reward_config,
)


def _ending(reason, reward_config, taxonomy, catalog, obs_model, distribution=None):
    truth = taxonomy.slugs[0]
    steps = []
    if distribution is not None:
        steps = [{"action": {"kind": "diagnose", "distribution": distribution}}]
    traj = {"steps": steps, "termination_reason": reason}
    return score_trajectory(
        traj, GroundTruth(condition=truth, analytes={}), reward_config,
        taxonomy=taxonomy, catalog=catalog, model=obs_model,
    )


@pytest.mark.parametrize("reason", sorted(NO_COMMIT_REASONS))
def test_no_commit_scores_below_abstain(reason, reward_config, taxonomy, catalog,
                                        obs_model) -> None:
    timeout = _ending(reason, reward_config, taxonomy, catalog, obs_model).diagnosis
    abstain = _ending("abstain", reward_config, taxonomy, catalog, obs_model).diagnosis
    assert timeout < abstain, "running out the clock must not be as cheap as abstaining"
    assert timeout == pytest.approx(abstain - reward_config.no_commit_penalty)


def test_prior_report_beats_no_commit(reward_config, taxonomy, catalog, obs_model) -> None:
    """The honest fallback at the horizon is to report the prior, never to time out.

    This is the property that keeps the penalty from teaching recklessness: under a proper
    rule a calibrated broad report is always available and, in expectation over the
    prior, beats failing to decide by a clear margin.
    """
    from dxenv.reward.scoring import terminal_diagnosis_score

    w = taxonomy.prior()
    report = {s: float(p) for s, p in zip(taxonomy.slugs, w, strict=True)}
    ev = sum(
        float(p) * terminal_diagnosis_score(report, s, taxonomy, reward_config.severity)
        for s, p in zip(taxonomy.slugs, w, strict=True)
    )
    timeout = _ending("max_turns", reward_config, taxonomy, catalog, obs_model).diagnosis
    abstain = _ending("abstain", reward_config, taxonomy, catalog, obs_model).diagnosis
    assert ev > abstain > timeout
    assert ev - timeout > 0.25, "the hedge must be closed by a margin, not a rounding error"


def test_no_commit_far_from_confident_wrong_floor(reward_config, taxonomy, catalog,
                                                  obs_model) -> None:
    """Not catastrophic: a timeout must still beat a confident wrong answer."""
    truth, decoy = taxonomy.slugs[0], taxonomy.slugs[1]
    wrong = _ending("diagnose", reward_config, taxonomy, catalog, obs_model,
                    {decoy: 0.95, truth: 0.05})
    timeout = _ending("max_turns", reward_config, taxonomy, catalog, obs_model)
    assert wrong.diagnosis < timeout.diagnosis


def test_unknown_termination_reason_raises(reward_config, taxonomy, catalog,
                                           obs_model) -> None:
    with pytest.raises(RewardError, match="unknown termination_reason"):
        _ending("budget_vibes", reward_config, taxonomy, catalog, obs_model)


def test_negative_no_commit_penalty_refused(reward_config) -> None:
    with pytest.raises(RewardError, match="no_commit"):
        validate_reward_config(dataclasses.replace(reward_config, no_commit_penalty=-0.1))


def test_no_commit_penalty_is_in_the_hash(reward_config) -> None:
    """Otherwise a run could resume across the change and mix two reward regimes."""
    changed = dataclasses.replace(reward_config, no_commit_penalty=0.0)
    assert changed.hash() != reward_config.hash()
