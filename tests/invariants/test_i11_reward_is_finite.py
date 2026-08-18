"""I11: reward is finite and bounded. NaN/inf is a hard failure, never clipped away."""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.env.bayes import BayesError, posterior
from dxenv.reward.engine import GroundTruth, InvariantViolation, reward_bounds, score_trajectory
from dxenv.reward.scoring import ScoringError, brier_score


def _random_trajectory(rec, rng, menu, episode_config, catalog, taxonomy):
    from dxenv.env.actions import ActionKind, action_id
    from dxenv.env.episode import DiagnosticEpisode
    from dxenv.env.schemas import Abstain, Diagnose, OrderTest, Prescribe

    ep = DiagnosticEpisode(rec, seed=int(rng.integers(1 << 30)), config=episode_config,
                           menu=menu, catalog=catalog)
    ep.reset()
    while not ep.state.done:
        roll = rng.random()
        if roll < 0.55:
            key = str(rng.choice(list(catalog.test_keys)))
            pred = str(rng.choice(["low", "normal", "high", "normal_categorical",
                                   "abnormal_categorical"]))
            ep.step(OrderTest(action_id=menu.id_for_test(key), test_key=key, prediction=pred))
        elif roll < 0.75:
            t = str(rng.choice(list(catalog.treatment_keys)))
            ep.step(Prescribe(action_id=menu.id_for_treatment(t), treatment_key=t))
        elif roll < 0.9:
            k = int(rng.integers(1, 6))
            idx = rng.choice(len(taxonomy), size=k, replace=False)
            w = rng.dirichlet(np.ones(k))
            dist = {taxonomy.slugs[int(i)]: float(x) for i, x in zip(idx, w, strict=True)}
            total = sum(dist.values())
            ep.step(Diagnose(action_id=action_id(ActionKind.DIAGNOSE, "diagnose"),
                             distribution={k2: v / total for k2, v in dist.items()}))
        else:
            ep.step(Abstain(action_id=action_id(ActionKind.ABSTAIN, "abstain")))
    return ep.trajectory()


def test_reward_finite_over_random_policies(fixture_corpus, menu, episode_config, catalog,
                                            taxonomy, reward_config) -> None:
    rng = np.random.default_rng(17)
    lo, hi = reward_bounds(reward_config, taxonomy)
    for rec in fixture_corpus:
        traj = _random_trajectory(rec, rng, menu, episode_config, catalog, taxonomy)
        gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
        b = score_trajectory(traj, gt, reward_config)
        assert np.isfinite(b.total), f"non-finite reward for {rec.patient_id}"
        assert lo <= b.total <= hi, f"reward {b.total} outside bounds ({lo}, {hi})"


@pytest.mark.slow
def test_reward_finite_over_full_corpus(full_corpus, menu, episode_config, catalog,
                                        taxonomy, reward_config) -> None:
    rng = np.random.default_rng(18)
    for rec in full_corpus[:400]:
        traj = _random_trajectory(rec, rng, menu, episode_config, catalog, taxonomy)
        gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
        assert np.isfinite(score_trajectory(traj, gt, reward_config).total)


def test_nan_belief_raises_rather_than_being_clipped(taxonomy) -> None:
    n = len(taxonomy)
    bad = np.full(n, 1.0 / n)
    bad[0] = np.nan
    with pytest.raises(ScoringError):
        brier_score(bad, 0)


def test_inf_belief_raises(taxonomy) -> None:
    n = len(taxonomy)
    bad = np.zeros(n)
    bad[0] = np.inf
    with pytest.raises(ScoringError):
        brier_score(bad, 0)


def test_posterior_stays_finite_under_extreme_evidence(obs_model, catalog, taxonomy) -> None:
    """Piling on evidence must not underflow the posterior to NaN.

    This is what the categorical smoothing and the log-sum-exp floor are for.
    """
    rng = np.random.default_rng(5)
    for _ in range(20):
        cond = str(rng.choice(taxonomy.slugs))
        ev = {k: obs_model.sample(k, cond, rng) for k in catalog.all_analyte_keys}
        p = posterior(ev, obs_model)
        assert np.isfinite(p).all()
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()


def test_posterior_rejects_non_finite_prior(obs_model, taxonomy) -> None:
    bad = np.full(len(taxonomy), np.nan)
    with pytest.raises(BayesError):
        posterior({}, obs_model, prior_log=bad)


def test_engine_raises_on_non_finite_total(fixture_corpus, menu, episode_config, catalog,
                                           reward_config, monkeypatch) -> None:
    """Test the detector: an injected NaN must halt, not be silently clipped."""
    import dxenv.reward.engine as eng

    rec = fixture_corpus[0]
    traj = _random_trajectory(rec, np.random.default_rng(1), menu, episode_config, catalog,
                              eng.load_taxonomy())
    monkeypatch.setattr(
        eng, "terminal_diagnosis_score", lambda *_a, **_k: float("nan")
    )
    traj["termination_reason"] = "diagnose"
    traj["steps"] = [s for s in traj["steps"] if s["action"]["kind"] != "abstain"]
    if not any(s["action"]["kind"] == "diagnose" for s in traj["steps"]):
        traj["steps"].append({
            "turn": 1,
            "action": {"kind": "diagnose", "action_id": "x", "distribution": {rec.condition: 1.0}},
            "revealed": [], "was_duplicate": False, "cost_charged": 0.0,
        })
    gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
    with pytest.raises(InvariantViolation, match="not finite"):
        score_trajectory(traj, gt, reward_config)
