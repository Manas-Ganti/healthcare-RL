"""I2: the observation is built by allowlist; unknown things are refused, not passed."""

from __future__ import annotations

import pytest
from dxenv.data.corpus import BLOCKED_RESOURCE_TYPES
from dxenv.env.filter import ALLOWED_RESOURCE_FIELDS, FilterError, filter_resources
from dxenv.env.schemas import Observation
from tests.conftest import observe


def test_unknown_resource_type_raises_in_strict_mode() -> None:
    with pytest.raises(FilterError, match="unrecognised resource type"):
        filter_resources([{"resourceType": "GenomicStudy", "data": "x"}], strict=True)


def test_unknown_resource_type_dropped_when_not_strict() -> None:
    assert filter_resources([{"resourceType": "GenomicStudy"}], strict=False) == ()


def test_no_blocked_resource_survives_the_filter(fixture_corpus) -> None:
    for rec in fixture_corpus:
        kept = filter_resources(rec.resources, strict=False)
        types = {r["resourceType"] for r in kept}
        assert not (types & BLOCKED_RESOURCE_TYPES)


def test_every_blocked_type_is_actually_present_in_the_source(fixture_corpus) -> None:
    """The filter tests are only meaningful if the leaky resources exist to be removed.

    A filter that removes things that were never there is the most comfortable kind of
    green test and the least informative.
    """
    present = {r["resourceType"] for rec in fixture_corpus for r in rec.resources}
    overlap = present & BLOCKED_RESOURCE_TYPES
    assert len(overlap) >= 6, f"source records only contain {sorted(overlap)}"


def test_allowlist_and_blocklist_are_disjoint() -> None:
    assert not (set(ALLOWED_RESOURCE_FIELDS) & BLOCKED_RESOURCE_TYPES)


def test_permitted_resources_drop_unlisted_fields() -> None:
    kept = filter_resources(
        [
            {
                "resourceType": "Encounter",
                "class": "amb",
                "reasonCode": [{"text": "diabetes"}],
            }
        ],
        strict=True,
    )
    assert kept == ({"resourceType": "Encounter", "class": "amb"},)


def test_observation_schema_forbids_extra_fields() -> None:
    """The allowlist must fail closed at the schema too, not only at the resource filter."""
    assert Observation.model_config["extra"] == "forbid"


def test_filter_is_idempotent(fixture_corpus) -> None:
    for rec in fixture_corpus[:20]:
        once = filter_resources(rec.resources, strict=False)
        assert filter_resources(once, strict=False) == once


def test_observation_is_json_serializable_and_stable(fixture_corpus, menu, catalog) -> None:
    import json


    rec = fixture_corpus[0]
    a = observe(rec, catalog, menu, budget=50.0)
    b = observe(rec, catalog, menu, budget=50.0)
    assert json.dumps(a.model_dump(mode="json")) == json.dumps(b.model_dump(mode="json"))
