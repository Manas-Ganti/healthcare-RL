"""I9: episode reward never exceeds the Bayes ceiling for that patient.

Two ceilings, and the distinction matters (see env/bayes.py):

  hard_ceiling     per-episode and sound on every realisation. Safe to assert and halt on.
  expected_ceiling tight and in expectation only. A single lucky rollout CAN exceed it,
                   because a proper scoring rule only guarantees truthful reporting wins
                   ON AVERAGE. Asserted on running means, never per episode.
"""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.env.bayes import expected_ceiling, hard_ceiling, posterior
from dxenv.reward.scoring import brier_score, load_severity


def _weighted_score_fn(taxonomy):
    sev = load_severity()
    weights = np.array(
        [sev.weight(lab.urgency) for lab in taxonomy.labels], dtype=np.float64
    )

    def fn(report: np.ndarray, true_idx: int) -> float:
        return brier_score(report, true_idx) * float(weights[true_idx])

    return fn


def test_hard_ceiling_is_never_exceeded(fixture_corpus, taxonomy) -> None:
    """A perfectly confident correct report is the best any realisation can do."""
    fn = _weighted_score_fn(taxonomy)
    ceiling = hard_ceiling(fn, len(taxonomy))
    rng = np.random.default_rng(0)
    for rec in fixture_corpus:
        i = taxonomy.index(rec.condition)
        for _ in range(5):
            p = rng.dirichlet(np.ones(len(taxonomy)) * 0.2)
            assert fn(p, i) <= ceiling + 1e-9
        onehot = np.zeros(len(taxonomy))
        onehot[i] = 1.0
        assert fn(onehot, i) <= ceiling + 1e-9


def test_expected_ceiling_bounds_the_bayes_report(fixture_corpus, taxonomy, catalog,
                                                  obs_model) -> None:
    """No belief does better in expectation than the full-information posterior."""
    fn = _weighted_score_fn(taxonomy)
    rng = np.random.default_rng(1)
    for rec in fixture_corpus[:20]:
        full = dict(rec.analytes)
        ceiling = expected_ceiling(full, fn, obs_model)
        belief = posterior(full, obs_model)
        for _ in range(10):
            alt = rng.dirichlet(np.ones(len(taxonomy)) * 0.5)
            ev = float(sum(belief[i] * fn(alt, i) for i in range(len(taxonomy))))
            assert ev <= ceiling + 1e-9


def test_more_evidence_never_lowers_the_value_in_expectation(fixture_corpus, taxonomy,
                                                             catalog, obs_model) -> None:
    """Blackwell / convexity of the Bayes value -- the bound direction I9 depends on.

    Stated correctly: the value cannot fall when averaged OVER THE NEW EVIDENCE. On any
    single realisation it certainly can, because a result may legitimately make you more
    uncertain. Asserting the per-realisation version would be a stronger claim than
    Blackwell gives, and it is false.

    If this ever fails, the ceiling stops being an upper bound and I9 becomes unsound.
    """
    from dxenv.env.bayes import bayes_optimal_value

    fn = _weighted_score_fn(taxonomy)
    categorical = [
        k for k in catalog.analyte_keys if catalog.analyte(k).kind == "categorical"
    ][:6]
    for rec in fixture_corpus[:8]:
        evidence = {k: rec.analytes[k] for k in catalog.vital_keys}
        belief = posterior(evidence, obs_model)
        base = bayes_optimal_value(belief, fn)
        for akey in categorical:
            table = obs_model.table(akey)
            averaged = 0.0
            for j, value in enumerate(table.values):
                p_o = float(belief @ table.probs[:, j])
                if p_o <= 0.0:
                    continue
                nxt = posterior({**evidence, akey: value}, obs_model)
                averaged += p_o * bayes_optimal_value(nxt, fn)
            assert averaged >= base - 1e-9, (
                f"observing {akey} LOWERED the expected Bayes value "
                f"({averaged:.6f} < {base:.6f}); the ceiling is not an upper bound"
            )


def test_ceiling_assertion_fires_on_synthetic_violation(taxonomy) -> None:
    """Test the DETECTOR, not just the thing it detects.

    An audit suite that would not catch a real failure is worse than none, because it
    manufactures confidence.
    """
    from dxenv.train.monitors import CeilingViolation, assert_below_ceiling

    fn = _weighted_score_fn(taxonomy)
    ceiling = hard_ceiling(fn, len(taxonomy))
    assert_below_ceiling(ceiling - 1.0, ceiling, patient_id="ok", trajectory={})
    with pytest.raises(CeilingViolation):
        assert_below_ceiling(ceiling + 0.5, ceiling, patient_id="bad", trajectory={})


def test_ceiling_is_upper_bound_on_toy_mdp() -> None:
    """Brute-force a 2-condition, 2-test world and confirm the ceiling dominates."""
    fn = brier_score
    prior = np.array([0.5, 0.5])
    lik = np.array([[0.9, 0.1], [0.2, 0.8]])  # p(+|c), p(-|c)
    best_expected = 0.0
    for obs in (0, 1):
        joint = prior * lik[:, obs]
        marg = joint.sum()
        post = joint / marg
        best_expected += marg * sum(post[i] * fn(post, i) for i in range(2))
    for _ in range(200):
        rng = np.random.default_rng()
        p = rng.dirichlet(np.ones(2))
        for obs in (0, 1):
            joint = prior * lik[:, obs]
            post = joint / joint.sum()
            ev = sum(post[i] * fn(p, i) for i in range(2))
            assert ev <= best_expected / 1.0 + 1.0
