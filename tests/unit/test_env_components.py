"""Unit tests for the env-side components (CLAUDE.md 6)."""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.data.taxonomy import Taxonomy, TaxonomyError
from dxenv.env.catalog import CatalogError, load_catalog
from dxenv.env.filter import global_vocabulary, scrub_text


def test_vocabulary_is_global(fixture_corpus, catalog, menu) -> None:
    """The exemption in assert_no_label_leak is sound only if this holds.

    If this test is deleted, the leak audit silently stops guarding record-derived text,
    because every string would look like vocabulary.
    """
    from tests.conftest import observe

    vocab = global_vocabulary()
    per_patient: set[frozenset[str]] = set()
    for rec in fixture_corpus[:30]:
        obs = observe(rec, catalog, menu)
        codes = {r.value_code for r in obs.revealed_results if r.value_code}
        assert codes <= vocab
        per_patient.add(frozenset(r.analyte for r in obs.revealed_results))
    assert len(per_patient) == 1, "the set of analytes varies by patient; that is a channel"


def test_vocabulary_does_not_depend_on_any_patient() -> None:
    assert global_vocabulary() is global_vocabulary()


def test_scrub_redacts_longest_form_first() -> None:
    """Short-form-first would leave "type 2 [redacted] mellitus", which still names it."""
    out = scrub_text("Patient has Type 2 diabetes mellitus documented.")
    assert "diabetes" not in out.lower()
    assert "[redacted]" in out


def test_scrub_is_case_insensitive_and_word_bounded() -> None:
    assert "[redacted]" in scrub_text("history of MIGRAINE")
    # A word merely CONTAINING a label form must survive: over-scrubbing destroys signal.
    assert scrub_text("premigraineous aura") == "premigraineous aura"


def test_scrub_is_idempotent() -> None:
    once = scrub_text("Known asthma and gout.")
    assert scrub_text(once) == once


def test_taxonomy_rejects_duplicate_slugs() -> None:
    from dxenv.data.taxonomy import Label

    lab = Label("a", "A", "sys", 1, 1.0, ())
    with pytest.raises(TaxonomyError, match="duplicate"):
        Taxonomy([lab, lab])


def test_taxonomy_rejects_bad_urgency() -> None:
    from dxenv.data.taxonomy import Label

    with pytest.raises(TaxonomyError, match="urgency"):
        Taxonomy([Label("a", "A", "sys", 9, 1.0, ())])


def test_taxonomy_rejects_zero_prior() -> None:
    from dxenv.data.taxonomy import Label

    with pytest.raises(TaxonomyError, match="prior weight"):
        Taxonomy([Label("a", "A", "sys", 1, 0.0, ())])


def test_unknown_slug_raises(taxonomy) -> None:
    with pytest.raises(TaxonomyError, match="unknown condition slug"):
        taxonomy.index("nope")


def test_catalog_rejects_unreachable_analyte(tmp_path, catalog) -> None:
    """An analyte no test can order inflates the ceiling above anything attainable."""
    import yaml

    raw = yaml.safe_load((__import__("pathlib").Path("dxenv/env/catalog.yaml")).read_text())
    raw["analytes"].append({
        "key": "unreachable", "display": "Unreachable", "kind": "quantitative",
        "unit": "x", "ref_low": 0.0, "ref_high": 1.0, "bounds": [0.0, 2.0],
        "healthy": {"mean": 0.5, "sd": 0.1},
    })
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(CatalogError, match="reachable by no orderable test"):
        load_catalog(p)


def test_catalog_rejects_unnormalised_categorical(tmp_path) -> None:
    import pathlib

    import yaml

    raw = yaml.safe_load(pathlib.Path("dxenv/env/catalog.yaml").read_text())
    for a in raw["analytes"]:
        if a["kind"] == "categorical":
            a["healthy"] = {a["values"][0]: 0.5}
            break
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(CatalogError, match="sums to"):
        load_catalog(p)


def test_categorical_normal_value_is_first(catalog) -> None:
    """The predict-then-verify bucketing relies on values[0] being the reference result."""
    prefixes = ("normal", "negative", "no_growth", "nonreactive", "undetectable",
                "nonpregnant_normal")
    for key in catalog.analyte_keys:
        a = catalog.analyte(key)
        if a.kind == "categorical":
            assert a.values[0].startswith(prefixes), f"{key}: {a.values[0]}"


def test_observation_model_rejects_unknown_condition(tmp_path, taxonomy) -> None:
    import pathlib

    from dxenv.env.obs_model import ObsModelError, build_observation_model

    p = tmp_path / "ov.yaml"
    p.write_text("not_a_condition:\n  troponin: {mean: 10.0, sd: 1.0}\n")
    build_observation_model.cache_clear()
    try:
        with pytest.raises(ObsModelError, match="not in the taxonomy"):
            build_observation_model(p)
    finally:
        build_observation_model.cache_clear()
        build_observation_model()
    assert pathlib.Path("dxenv/env/obs_overrides.yaml").exists()


def test_posterior_moves_correct_direction(obs_model, taxonomy) -> None:
    """Informative abnormal evidence raises mass on conditions with higher likelihood."""
    base = obs_model
    prior_p = np.exp(np.log(taxonomy.prior()))
    for analyte, value in (("troponin", 1200.0), ("lipase", 1400.0), ("tsh", 18.0)):
        from dxenv.env.bayes import posterior

        after = posterior({analyte: value}, base)
        ll = base.log_likelihood_vector(analyte, value)
        favoured = np.argsort(-ll)[:10]
        assert after[favoured].sum() > prior_p[favoured].sum()


def test_action_menu_rejects_duplicate_ids() -> None:
    from dxenv.env.actions import Action, ActionError, ActionKind, ActionMenu

    a = Action("dup", ActionKind.DIAGNOSE, "x", "X")
    with pytest.raises(ActionError, match="collision"):
        ActionMenu((a, a))


def test_episode_config_rejects_default_cost(tmp_path) -> None:
    import pathlib

    from dxenv.env.episode import EpisodeError, load_episode_config

    p = tmp_path / "costs.yaml"
    p.write_text("version: 1\ndefault: 0.0\ntests:\n  cbc: 5.0\n")
    with pytest.raises(EpisodeError, match="declares a default"):
        load_episode_config(pathlib.Path("dxenv/configs/env.yaml"), p)


def test_budget_weights_must_normalise(tmp_path) -> None:
    import pathlib

    import yaml
    from dxenv.env.episode import EpisodeError, load_episode_config

    raw = yaml.safe_load(pathlib.Path("dxenv/configs/env.yaml").read_text())
    raw["budget"]["weights"] = [0.5] * len(raw["budget"]["support"])
    p = tmp_path / "env.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(EpisodeError, match="weights sum"):
        load_episode_config(p, pathlib.Path("dxenv/configs/costs.yaml"))
