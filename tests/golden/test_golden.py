"""Golden tests against frozen fixtures.

The Bayes solver and the reward engine will both be refactored, and both are easy to
break subtly -- a sign flip, a renormalisation in the wrong place, an off-by-one in the
label ordering. None of those show up as an exception; they show up as slightly
different numbers that still look plausible. That is what these catch.

Regenerate with `python scripts/regenerate_golden.py`, deliberately, and READ THE DIFF.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dxenv.reward.engine import GroundTruth, score_trajectory

GOLDEN = Path("tests/golden")


@pytest.fixture(scope="module")
def episodes() -> list[dict]:
    return json.loads((GOLDEN / "episodes.json").read_text())


@pytest.fixture(scope="module")
def fingerprints() -> dict:
    return json.loads((GOLDEN / "fingerprints.json").read_text())


def test_label_set_hash_matches_golden(taxonomy, fingerprints) -> None:
    assert taxonomy.hash() == fingerprints["label_set_hash"]
    assert len(taxonomy) == fingerprints["n_labels"]


def test_menu_fingerprint_matches_golden(menu, fingerprints) -> None:
    """A changed fingerprint invalidates every stored trajectory. Never silent."""
    assert menu.fingerprint() == fingerprints["menu_fingerprint"]
    assert len(menu.test_actions()) == fingerprints["n_tests"]
    assert len(menu.treatment_actions()) == fingerprints["n_treatments"]


def test_config_hashes_match_golden(episode_config, reward_config, fingerprints) -> None:
    assert episode_config.hash() == fingerprints["episode_config_hash"]
    assert reward_config.hash() == fingerprints["reward_config_hash"]


def test_posterior_matches_hand_computed() -> None:
    """The 2-condition, 2-test case worked out by hand in env/bayes.py's docstring."""
    toy = json.loads((GOLDEN / "bayes_toy.json").read_text())
    prior = np.array(toy["prior"])
    lik1 = np.array(toy["likelihood_plus"])
    post1 = prior * lik1
    post1 /= post1.sum()
    assert np.allclose(post1, toy["posterior_after_plus"], atol=1e-12)

    lik2 = np.array(toy["likelihood_plus_2"])
    post2 = prior * lik1 * lik2
    post2 /= post2.sum()
    assert np.allclose(post2, toy["posterior_after_both"], atol=1e-12)

    # Order invariance, on the hand-computed case specifically.
    other = prior * lik2 * lik1
    other /= other.sum()
    assert np.allclose(post2, other, atol=1e-15)


def test_rescoring_stored_trajectory_matches(episodes, reward_config) -> None:
    """Rescoring a frozen trajectory must reproduce the frozen reward, term by term."""
    from dxenv.data.corpus import generate_corpus

    records = {r.patient_id: r for r in generate_corpus(5, seed=314159)}
    for ep in episodes:
        rec = records[ep["patient_id"]]
        assert rec.condition == ep["condition"], "corpus generation drifted"
        gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
        actual = score_trajectory(ep["trajectory"], gt, reward_config).as_dict()
        expected = ep["reward"]
        for key, want in expected.items():
            got = actual[key]
            if isinstance(want, float):
                assert got == pytest.approx(want, abs=1e-9), f"{key}: {got} != {want}"
            else:
                assert got == want, f"{key}: {got} != {want}"


def test_golden_trajectories_are_wellformed(episodes) -> None:
    for ep in episodes:
        traj = ep["trajectory"]
        assert traj["terminated"]
        assert traj["termination_reason"] == "diagnose"
        assert traj["spent"] <= traj["budget"]
        kinds = [s["action"]["kind"] for s in traj["steps"]]
        assert kinds.count("diagnose") == 1
        assert kinds.index("diagnose") == len(kinds) - 1, "diagnose must terminate"


def test_golden_covers_every_reward_term(episodes) -> None:
    """A golden that exercises only some terms silently stops guarding the others."""
    keys = set()
    for ep in episodes:
        keys |= set(ep["reward"])
    for term in ("diagnosis", "test_cost", "turn_penalty", "treatment", "shaping",
                 "verify", "total"):
        assert term in keys
    assert any(ep["reward"]["test_cost"] != 0.0 for ep in episodes)
    assert any(ep["reward"]["verify"] != 0.0 for ep in episodes)
    assert any(ep["reward"]["treatment"] != 0.0 for ep in episodes)
