"""Property-based tests for the guarantees the design rests on.

CLAUDE.md 11: assert the mathematical properties directly rather than testing examples.
These are the claims the whole environment leans on -- properness, telescoping,
normalisation, boundedness, order invariance -- so they get generative tests, not
hand-picked cases that happen to pass.
"""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.env.bayes import BayesError, entropy, posterior
from dxenv.reward.scoring import ScoringError, brier_score, distribution_to_vector
from dxenv.reward.shaping import discounted_total, telescoped_total, total_shaping
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

SETTINGS = settings(
    max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def simplex(draw, n: int | None = None):
    size = n if n is not None else draw(st.integers(min_value=2, max_value=40))
    weights = draw(
        st.lists(st.floats(min_value=1e-3, max_value=1e3, allow_nan=False,
                           allow_infinity=False), min_size=size, max_size=size)
    )
    v = np.asarray(weights, dtype=np.float64)
    return v / v.sum()


@SETTINGS
@given(p=simplex(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_brier_is_bounded_and_finite(p, seed) -> None:
    i = seed % p.size
    s = brier_score(p, i)
    assert np.isfinite(s)
    assert -1.0 - 1.0 / p.size - 1e-9 <= s <= 1.0 + 1e-9


@SETTINGS
@given(q=simplex(n=12), r=simplex(n=12), mix=st.floats(min_value=0.01, max_value=0.99))
def test_brier_properness_property(q, r, mix) -> None:
    """Reporting your true belief maximises expected score. Strictly, unless p == q."""
    p = mix * q + (1.0 - mix) * r
    eq = float(sum(q[i] * brier_score(q, i) for i in range(q.size)))
    ep = float(sum(q[i] * brier_score(p, i) for i in range(q.size)))
    assert ep <= eq + 1e-12
    if not np.allclose(p, q, atol=1e-9):
        assert ep < eq + 1e-12


@SETTINGS
@given(p=simplex())
def test_entropy_bounds(p) -> None:
    h = entropy(p)
    assert -1e-9 <= h <= np.log(p.size) + 1e-9


@SETTINGS
@given(
    beliefs=st.lists(simplex(n=10), min_size=2, max_size=8),
    gamma=st.floats(min_value=0.1, max_value=1.0),
    scale=st.floats(min_value=0.0, max_value=2.0),
)
def test_shaping_telescopes_property(beliefs, gamma, scale) -> None:
    assert discounted_total(beliefs, gamma, scale) == pytest.approx(
        telescoped_total(beliefs, gamma, scale), abs=1e-8
    )


@SETTINGS
@given(beliefs=st.lists(simplex(n=10), min_size=2, max_size=6), scale=st.floats(0.0, 2.0))
def test_closed_loop_shaping_is_zero_property(beliefs, scale) -> None:
    loop = [*beliefs, *reversed(beliefs[:-1])]
    assert total_shaping(loop, 1.0, scale) == pytest.approx(0.0, abs=1e-9)


@SETTINGS
@given(p=simplex())
def test_belief_validation_rejects_unnormalised(p) -> None:
    bad = p * 1.5
    with pytest.raises(ScoringError):
        brier_score(bad, 0)


@SETTINGS
@given(subset=st.lists(st.integers(min_value=0, max_value=90), min_size=1, max_size=12,
                       unique=True),
       seed=st.integers(min_value=0, max_value=10_000))
def test_posterior_is_order_invariant(subset, seed, obs_model, catalog, taxonomy) -> None:
    """The posterior depends on the evidence SET, never on the order it arrived in."""
    rng = np.random.default_rng(seed)
    cond = str(rng.choice(taxonomy.slugs))
    keys = [catalog.analyte_keys[i % len(catalog.analyte_keys)] for i in subset]
    ev = {k: obs_model.sample(k, cond, rng) for k in keys}
    forwards = posterior(ev, obs_model)
    backwards = posterior(dict(reversed(list(ev.items()))), obs_model)
    assert np.allclose(forwards, backwards, atol=1e-12)


@SETTINGS
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_posterior_normalizes_and_is_nonnegative(seed, obs_model, catalog, taxonomy) -> None:
    rng = np.random.default_rng(seed)
    cond = str(rng.choice(taxonomy.slugs))
    n = int(rng.integers(0, 15))
    keys = list(rng.choice(catalog.analyte_keys, size=n, replace=False))
    ev = {k: obs_model.sample(k, cond, rng) for k in keys}
    p = posterior(ev, obs_model)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0.0).all()
    assert np.isfinite(p).all()


@SETTINGS
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_irrelevant_evidence_leaves_posterior_unchanged(seed, obs_model, taxonomy) -> None:
    """An analyte whose distribution is identical across conditions must not move mass."""
    from dxenv.env.obs_model import QuantTable

    rng = np.random.default_rng(seed)
    base = posterior({}, obs_model)
    n = len(taxonomy)
    flat = QuantTable(analyte="_flat", mean=np.zeros(n), sd=np.ones(n), low=-10.0, high=10.0)
    patched = dict(obs_model.tables)
    patched["_flat"] = flat
    from dxenv.env.obs_model import ObservationModel

    model2 = ObservationModel(obs_model.conditions, patched, obs_model.condition_index_map)
    moved = posterior({"_flat": float(rng.uniform(-3, 3))}, model2)
    assert np.allclose(base, moved, atol=1e-12)


@SETTINGS
@given(seed=st.integers(min_value=0, max_value=5_000))
def test_distribution_roundtrip(seed, taxonomy) -> None:
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 8))
    idx = rng.choice(len(taxonomy), size=k, replace=False)
    w = rng.dirichlet(np.ones(k))
    dist = {taxonomy.slugs[int(i)]: float(x) for i, x in zip(idx, w, strict=True)}
    total = sum(dist.values())
    dist = {k2: v / total for k2, v in dist.items()}
    vec = distribution_to_vector(dist, taxonomy)
    assert vec.sum() == pytest.approx(1.0)
    for slug, prob in dist.items():
        assert vec[taxonomy.index(slug)] == pytest.approx(prob)


def test_prior_shape_mismatch_raises(obs_model) -> None:
    with pytest.raises(BayesError, match="shape"):
        posterior({}, obs_model, prior_log=np.zeros(3))
