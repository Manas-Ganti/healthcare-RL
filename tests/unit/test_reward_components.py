"""Unit tests for the reward sub-modules named in CLAUDE.md 7."""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.reward.treatment import (
    TreatmentError,
    contraindication_violations,
    load_treatment_config,
    treatment_score,
)
from dxenv.reward.verify import VerifyError, actual_bucket, bucket_prior, verify_term

HEALTHY = {"egfr": 95.0, "hcg_quant": 0.0, "potassium": 4.1, "platelets": 250.0}


@pytest.fixture(scope="module")
def tcfg():
    return load_treatment_config()


def _score(prescribed, declared, truth, analytes=None, allergies=(), cfg=None):
    return treatment_score(
        prescribed, declared, truth, analytes or HEALTHY, allergies,
        coherence_scale=0.5, correctness_scale=1.0, contraindication_penalty=4.0, cfg=cfg,
    )[0]


# ----------------------------------------------------------------------- treatment --


def test_contraindication_penalty_dominates_suboptimal(tcfg) -> None:
    """A contraindicated prescription must score strictly worse than suboptimal-but-safe.

    Otherwise "shotgun everything plausible" dominates, and the shotgun includes the harm.
    """
    dx = {"gout": 1.0}
    safe_but_suboptimal = _score(["acetaminophen"], dx, "gout", cfg=tcfg)
    contraindicated = _score(["ibuprofen"], dx, "gout", {**HEALTHY, "egfr": 18.0}, cfg=tcfg)
    assert contraindicated < safe_but_suboptimal
    assert contraindicated < 0.0


def test_contraindication_dominates_even_when_the_drug_is_first_line(tcfg) -> None:
    """The penalty must beat the appropriateness credit, not merely offset it."""
    dx = {"gout": 1.0}
    correct_but_harmful = _score(["ibuprofen"], dx, "gout", {**HEALTHY, "egfr": 18.0}, cfg=tcfg)
    do_nothing = _score([], dx, "gout", cfg=tcfg)
    assert correct_but_harmful < do_nothing


def test_lucky_treatment_with_wrong_dx_scores_low(tcfg) -> None:
    """The right drug under a confidently wrong diagnosis collects almost nothing."""
    lucky = _score(["insulin_infusion"], {"migraine": 0.97, "diabetic_ketoacidosis": 0.03},
                   "diabetic_ketoacidosis", cfg=tcfg)
    earned = _score(["insulin_infusion"], {"diabetic_ketoacidosis": 0.97, "migraine": 0.03},
                    "diabetic_ketoacidosis", cfg=tcfg)
    assert lucky < earned
    assert lucky < 0.1


def test_coherent_treatment_scores_above_incoherent(tcfg) -> None:
    """At FIXED diagnostic accuracy, acting on your own stated belief must pay."""
    dx = {"community_acquired_pneumonia": 0.8, "influenza": 0.2}
    coherent = _score(["amoxicillin"], dx, "community_acquired_pneumonia", cfg=tcfg)
    incoherent = _score(["levothyroxine"], dx, "community_acquired_pneumonia", cfg=tcfg)
    assert coherent > incoherent


def test_allergy_is_detected(tcfg) -> None:
    v = contraindication_violations(["amoxicillin"], HEALTHY, ("penicillin",), tcfg)
    assert [x[1] for x in v] == ["allergy"]


def test_pregnancy_contraindication(tcfg) -> None:
    v = contraindication_violations(["lisinopril"], {**HEALTHY, "hcg_quant": 5000.0}, (), tcfg)
    assert any(x[1] == "pregnancy" for x in v)


def test_drug_interaction_needs_both_drugs(tcfg) -> None:
    assert not contraindication_violations(["thrombolysis"], HEALTHY, (), tcfg)
    v = contraindication_violations(["thrombolysis", "apixaban"], HEALTHY, (), tcfg)
    assert any(x[1] == "interaction" for x in v)


def test_prescribing_nothing_is_never_negative(tcfg) -> None:
    assert _score([], {"gout": 1.0}, "gout", cfg=tcfg) == 0.0


def test_violations_are_order_independent(tcfg) -> None:
    a = contraindication_violations(["thrombolysis", "apixaban"], HEALTHY, (), tcfg)
    b = contraindication_violations(["apixaban", "thrombolysis"], HEALTHY, (), tcfg)
    assert sorted(a) == sorted(b)


def test_unknown_treatment_in_config_raises(tmp_path) -> None:
    bad = tmp_path / "t.yaml"
    bad.write_text(
        "first_line:\n  gout: [not_a_real_drug]\nacceptable: {}\nacceptable_fraction: 0.4\n"
        "contraindications:\n  allergy: {scale: 1.0}\n"
        "  renal: {egfr_below: 30.0, treatments: [], scale: 1.0}\n"
        "  pregnancy: {hcg_above: 25.0, treatments: [], scale: 1.0}\n"
        "  hyperkalemia: {potassium_above: 5.5, treatments: [], scale: 1.0}\n"
        "  bleeding: {platelets_below: 50.0, treatments: [], scale: 1.0}\n"
        "  interactions: []\n"
    )
    with pytest.raises(TreatmentError, match="unknown treatments"):
        load_treatment_config(bad)


# -------------------------------------------------------------------------- verify --


def test_commit_is_mandatory() -> None:
    """Structural: the schema rejects a test order without a prediction."""
    from dxenv.env.schemas import OrderTest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OrderTest(action_id="x", test_key="cbc")


def test_result_not_visible_before_commit(menu, catalog, episode_config, fixture_corpus) -> None:
    """Structural: the action object carries the prediction and cannot reach the result.

    The result does not exist until `episode._do_order` runs, which is after the action
    has been constructed and handed over.
    """
    from dxenv.env.schemas import OrderTest

    action = OrderTest(action_id=menu.id_for_test("cbc"), test_key="cbc", prediction="low")
    assert not hasattr(action, "revealed")
    assert not hasattr(action, "result")
    assert set(OrderTest.model_fields) == {"kind", "action_id", "test_key", "prediction"}


def test_verify_score_zero_for_chance_predictions(catalog, obs_model, taxonomy) -> None:
    """Averaged over the prior, a fixed prediction earns exactly zero."""
    rng = np.random.default_rng(0)
    prior = taxonomy.prior()
    for key in ("troponin", "urinalysis", "ecg"):
        analyte = catalog.test(key).analytes[0]
        for prediction in (("normal_categorical",) if analyte.endswith("finding")
                           else ("low", "normal", "high")):
            total = 0.0
            n = 4000
            for _ in range(n):
                cond = taxonomy.slugs[int(rng.choice(len(taxonomy), p=prior))]
                revealed = {analyte: obs_model.sample(analyte, cond, rng)}
                total += verify_term(key, prediction, revealed, order_cost=1.0, fraction=0.25,
                                     catalog=catalog)
            assert abs(total / n) < 0.02, f"{key}/{prediction} earns {total / n:+.4f} at chance"


def test_verify_is_zero_when_nothing_was_revealed(catalog) -> None:
    assert verify_term("cbc", "normal", {}, order_cost=1.0, fraction=0.25, catalog=catalog) == 0.0


def test_verify_rejects_fraction_at_or_above_one(catalog) -> None:
    with pytest.raises(VerifyError, match="fraction"):
        verify_term("cbc", "normal", {"hemoglobin": 14.0}, order_cost=1.0, fraction=1.0,
                    catalog=catalog)


def test_bucket_classification(catalog) -> None:
    assert actual_bucket("hemoglobin", 5.0, catalog) == "low"
    assert actual_bucket("hemoglobin", 14.0, catalog) == "normal"
    assert actual_bucket("hemoglobin", 20.0, catalog) == "high"
    assert actual_bucket("ecg_finding", "normal_sinus", catalog) == "normal_categorical"
    assert actual_bucket("ecg_finding", "st_elevation", catalog) == "abnormal_categorical"


def test_bucket_priors_sum_to_one(catalog) -> None:
    for key in catalog.analyte_keys:
        a = catalog.analyte(key)
        buckets = (("normal_categorical", "abnormal_categorical")
                   if a.kind == "categorical" else ("low", "normal", "high"))
        assert sum(bucket_prior(key, b) for b in buckets) == pytest.approx(1.0, abs=1e-6)


def test_wrong_value_type_raises(catalog) -> None:
    with pytest.raises(VerifyError):
        actual_bucket("hemoglobin", "not_a_number", catalog)
