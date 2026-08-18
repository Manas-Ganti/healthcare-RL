"""I1: ground truth never enters an observation."""

from __future__ import annotations

import pytest
from dxenv.data.corpus import PatientRecord, PatientView
from dxenv.env.filter import (
    LabelLeakError,
    assert_no_label_leak,
    build_observation,
    observation_strings,
)
from dxenv.env.schemas import Observation
from tests.conftest import observe, orderable_results


def test_patient_view_cannot_carry_the_condition() -> None:
    """Structural: the type the filter accepts has no field for the label."""
    assert "condition" not in PatientView.__slots__
    assert "condition" in PatientRecord.__slots__


def test_observation_schema_has_no_label_field() -> None:
    """Structural: the type the filter RETURNS has no field for the label either.

    Also asserts there is no free-text field. Free text is where a label re-enters after
    every structural precaution has been taken.
    """
    fields = set(Observation.model_fields)
    for forbidden in ("condition", "diagnosis", "reason", "note", "conclusion", "text"):
        assert forbidden not in fields, f"Observation grew a {forbidden!r} field"


def test_no_label_string_in_observation_fixture(fixture_corpus, menu, catalog) -> None:
    fp = menu.fingerprint()
    for rec in fixture_corpus:
        obs = build_observation(rec.view(), orderable_results(rec, catalog), 0, 100.0, 20, fp)
        assert_no_label_leak(obs, rec.condition)


@pytest.mark.slow
def test_no_label_string_in_observation_full_corpus(full_corpus, menu, catalog) -> None:
    """Corpus-wide, with EVERY analyte revealed -- the maximally-exposed observation."""
    fp = menu.fingerprint()
    for rec in full_corpus:
        obs = build_observation(rec.view(), orderable_results(rec, catalog), 0, 100.0, 20, fp)
        assert_no_label_leak(obs, rec.condition)


@pytest.mark.slow
def test_no_label_string_for_every_condition(one_per_condition, menu, catalog) -> None:
    """One patient per condition: the rare labels get checked too."""
    fp = menu.fingerprint()
    for rec in one_per_condition:
        obs = build_observation(rec.view(), orderable_results(rec, catalog), 0, 100.0, 20, fp)
        assert_no_label_leak(obs, rec.condition)


def test_leak_detector_actually_fires(fixture_corpus, menu) -> None:
    """Test the detector, not just the thing it detects.

    An audit that would not catch a real leak is worse than none, because it
    manufactures confidence.
    """
    rec = fixture_corpus[0]
    obs = build_observation(rec.view(), {}, 0, 100.0, 20, menu.fingerprint())
    display = obs.__class__.model_validate(
        {
            **obs.model_dump(),
            "family_history": (f"history of {rec.condition.replace(chr(95), chr(32))}",),
        }
    )
    with pytest.raises(LabelLeakError):
        assert_no_label_leak(display, rec.condition)


def test_observation_strings_covers_every_string_field(fixture_corpus, menu, catalog) -> None:
    """If a new string field is added, the leak audit must see it.

    Without this, adding a field to Observation silently shrinks the audit's coverage
    while every leak test stays green.
    """
    rec = fixture_corpus[0]
    obs = observe(rec, catalog, menu)
    seen = set(observation_strings(obs))
    dumped = obs.model_dump()
    missed = []
    for name, value in dumped.items():
        if isinstance(value, str) and value and value not in seen:
            missed.append(name)
        if isinstance(value, tuple | list):
            for item in value:
                if isinstance(item, str) and item and item not in seen:
                    missed.append(name)
    assert not missed, f"observation_strings misses string fields: {sorted(set(missed))}"
