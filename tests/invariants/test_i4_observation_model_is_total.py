"""I4: every orderable test returns a value for every patient.

No "unavailable", no None, no default-to-normal. The point is that the SPARSITY PATTERN
of a record carries no information -- `test_no_side_channel` is the direct check.
"""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.env.catalog import CategoricalAnalyte, QuantitativeAnalyte
from dxenv.env.obs_model import ObsModelError, build_observation_model


def test_every_pair_has_a_distribution(obs_model, catalog, taxonomy) -> None:
    assert obs_model.pair_count == len(catalog.all_analyte_keys) * len(taxonomy)
    for akey in catalog.all_analyte_keys:
        table = obs_model.table(akey)
        if isinstance(table.__class__, type) and hasattr(table, "mean"):
            assert table.mean.shape == (len(taxonomy),)
            assert table.sd.shape == (len(taxonomy),)
        else:
            assert table.probs.shape[0] == len(taxonomy)


def test_lookup_miss_raises_rather_than_defaulting(obs_model) -> None:
    """'Return normal if the pair is not found' is exactly the bug I4 exists to prevent."""
    with pytest.raises(ObsModelError, match="no distribution for analyte"):
        obs_model.table("analyte_that_does_not_exist")


def test_sampling_is_total(obs_model, catalog, taxonomy) -> None:
    """Sample every pair; nothing is None, nothing raises, all finite."""
    rng = np.random.default_rng(0)
    for akey in catalog.all_analyte_keys:
        for cond in taxonomy.slugs:
            v = obs_model.sample(akey, cond, rng)
            assert v is not None
            if isinstance(v, float):
                assert np.isfinite(v)
            else:
                assert isinstance(v, str) and v


def test_distributions_normalize(obs_model, catalog) -> None:
    for akey in catalog.all_analyte_keys:
        a = catalog.analyte(akey)
        if isinstance(a, CategoricalAnalyte):
            sums = obs_model.table(akey).probs.sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-9)


def test_no_zero_likelihood_anywhere(obs_model, catalog) -> None:
    """Smoothing means no categorical likelihood is exactly zero.

    A zero likelihood makes log p = -inf, which propagates NaN through the posterior and
    into the reward [I11], and hands the agent a one-test oracle for any finding unique
    to one condition.
    """
    for akey in catalog.all_analyte_keys:
        if isinstance(catalog.analyte(akey), CategoricalAnalyte):
            assert (obs_model.table(akey).probs > 0.0).all()


def test_sampled_values_in_plausible_range(obs_model, catalog, taxonomy) -> None:
    rng = np.random.default_rng(7)
    for akey in catalog.all_analyte_keys:
        a = catalog.analyte(akey)
        for cond in taxonomy.slugs[::7]:
            for _ in range(5):
                v = obs_model.sample(akey, cond, rng)
                if isinstance(a, QuantitativeAnalyte):
                    assert a.bounds[0] <= v <= a.bounds[1], f"{akey}|{cond} out of bounds"
                else:
                    assert v in a.values


def test_deterministic_under_seed(obs_model, catalog, taxonomy) -> None:
    keys = catalog.all_analyte_keys[:20]
    def draw() -> list[object]:
        rng = np.random.default_rng(1234)
        return [obs_model.sample(k, taxonomy.slugs[3], rng) for k in keys]
    assert draw() == draw()


def test_every_patient_has_every_analyte(fixture_corpus, catalog) -> None:
    """The structural form of I4: there is no such thing as a missing result."""
    expected = set(catalog.all_analyte_keys)
    for rec in fixture_corpus:
        assert set(rec.analytes) == expected
        assert all(v is not None for v in rec.analytes.values())


def test_no_side_channel(fixture_corpus, catalog) -> None:
    """A classifier trained on WHICH tests returned values must perform at chance.

    This is the direct test that I4 worked. Because every analyte is always present, the
    presence-pattern feature matrix is constant, so there is nothing to learn -- we
    assert the matrix is genuinely constant rather than training a model to discover it.
    """
    presence = {
        tuple(sorted(k for k, v in rec.analytes.items() if v is not None))
        for rec in fixture_corpus
    }
    assert len(presence) == 1, (
        "patients differ in WHICH analytes are present; the sparsity pattern is a "
        "channel to the label"
    )
    assert set(presence.pop()) == set(catalog.all_analyte_keys)


@pytest.mark.slow
def test_no_side_channel_learned(full_corpus, catalog, taxonomy) -> None:
    """The empirical version: fit a classifier on presence bits and confirm chance."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    keys = catalog.all_analyte_keys
    x = np.array([[1.0 if k in rec.analytes else 0.0 for k in keys] for rec in full_corpus])
    y = np.array([taxonomy.index(rec.condition) for rec in full_corpus])
    keep = np.isin(y, [c for c in set(y.tolist()) if (y == c).sum() >= 5])
    x, y = x[keep], y[keep]
    learned = cross_val_score(LogisticRegression(max_iter=200), x, y, cv=3).mean()
    baseline = cross_val_score(DummyClassifier(strategy="prior"), x, y, cv=3).mean()
    assert learned <= baseline + 0.01, f"presence bits leak: {learned:.3f} vs {baseline:.3f}"


def test_model_build_is_cached_and_identical() -> None:
    assert build_observation_model() is build_observation_model()
