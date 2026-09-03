"""Constrained decoding (CLAUDE.md 8.1) and the prompt builder.

Format is not an SFT problem: the grammar makes invalid output impossible, which keeps
the format reward at exactly zero and preserves the entropy GRPO needs.
"""

from __future__ import annotations

import json

import jsonschema
import numpy as np
import pytest
from dxenv.env.schemas import Abstain, Diagnose, OrderTest, Prescribe
from dxenv.policy.decoding import (
    DecodingError,
    action_json_schema,
    complete_distribution,
    parse_action,
    render_wire,
    sample_wire_action,
)
from dxenv.policy.prompt import build_prompt, chat_messages, render_menu


@pytest.fixture(scope="module")
def schema():
    return action_json_schema()


def test_schema_is_a_valid_json_schema(schema) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


def test_constrained_decoding_always_valid(schema, menu, taxonomy) -> None:
    """1000 samples, 100% schema-valid and 100% parseable into a legal action."""
    rng = np.random.default_rng(0)
    validator = jsonschema.Draft202012Validator(schema)
    for _ in range(1000):
        wire = sample_wire_action(rng, menu=menu, taxonomy=taxonomy)
        validator.validate(wire)
        action = parse_action(render_wire(wire), menu, taxonomy)
        assert action.action_id in menu.ids


def test_probabilities_sum_to_one_post_decode(menu, taxonomy) -> None:
    """Sum-to-one is not expressible in JSON Schema, so it is repaired after decoding."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        wire = sample_wire_action(rng, menu=menu, taxonomy=taxonomy, kind="diagnose")
        action = parse_action(render_wire(wire), menu, taxonomy)
        assert isinstance(action, Diagnose)
        assert abs(sum(action.distribution.values()) - 1.0) < 1e-9


def test_grammar_rejects_off_menu_actions(schema, menu, taxonomy) -> None:
    """An off-menu action is UNREPRESENTABLE, not merely rejected downstream [I3]."""
    validator = jsonschema.Draft202012Validator(schema)
    for wire in (
        {"kind": "order_test", "reasoning": "x", "test_key": "whole_body_mri_with_answer",
         "prediction": "high"},
        {"kind": "prescribe", "reasoning": "x", "treatment_key": "unlisted_drug"},
        {"kind": "diagnose", "reasoning": "x",
         "diagnosis": [{"condition": "not_a_real_label", "probability": 1.0}]},
        {"kind": "teleport", "reasoning": "x"},
    ):
        assert not validator.is_valid(wire), f"grammar admitted {wire!r}"
        with pytest.raises(DecodingError):
            parse_action(json.dumps(wire), menu, taxonomy)


def test_action_id_is_derived_not_generated(schema, menu) -> None:
    """The model never emits an action_id, so it can never disagree with the test_key."""
    props = {v["properties"]["kind"]["const"]: v["properties"] for v in schema["oneOf"]}
    for variant in props.values():
        assert "action_id" not in variant


def test_residual_mass_is_spread_not_renormalised(taxonomy) -> None:
    """A report naming 0.7 across eight labels claims 0.3 elsewhere. Honour that.

    Renormalising would convert "0.3 of my belief is elsewhere" into a confidence the
    agent never claimed, and a proper scoring rule would duly reward it when right.
    """
    pairs = [{"condition": s, "probability": 0.1} for s in taxonomy.slugs[:7]]
    dist = complete_distribution(pairs, taxonomy)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    named = sum(dist[s] for s in taxonomy.slugs[:7])
    assert abs(named - 0.7) < 1e-6, "named mass was renormalised away"
    tail = [dist[s] for s in taxonomy.slugs[7:]]
    assert max(tail) - min(tail) < 1e-6, "residual mass is not uniform over the unnamed"


def test_overclaimed_mass_is_renormalised(taxonomy) -> None:
    """The one case with no residual to place."""
    pairs = [{"condition": s, "probability": 0.5} for s in taxonomy.slugs[:4]]
    dist = complete_distribution(pairs, taxonomy)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert abs(dist[taxonomy.slugs[0]] - 0.25) < 1e-9


def test_duplicate_conditions_are_summed(taxonomy) -> None:
    pairs = [
        {"condition": taxonomy.slugs[0], "probability": 0.3},
        {"condition": taxonomy.slugs[0], "probability": 0.2},
    ]
    assert abs(complete_distribution(pairs, taxonomy)[taxonomy.slugs[0]] - 0.5) < 1e-9


def test_degenerate_reports_raise(taxonomy) -> None:
    with pytest.raises(DecodingError, match="zero mass"):
        complete_distribution([{"condition": taxonomy.slugs[0], "probability": 0.0}], taxonomy)
    with pytest.raises(DecodingError, match="names no conditions"):
        complete_distribution([], taxonomy)
    with pytest.raises(DecodingError, match="outside the frozen taxonomy"):
        complete_distribution([{"condition": "nope", "probability": 1.0}], taxonomy)


def test_non_json_output_raises_rather_than_defaulting(menu, taxonomy) -> None:
    """No fallback action. A silent default becomes a quiet distributional shift."""
    with pytest.raises(DecodingError, match="not JSON"):
        parse_action("I would order a CBC.", menu, taxonomy)


def test_parse_round_trips_every_action_kind(menu, taxonomy) -> None:
    rng = np.random.default_rng(3)
    kinds = {"order_test": OrderTest, "prescribe": Prescribe,
             "diagnose": Diagnose, "abstain": Abstain}
    for kind, cls in kinds.items():
        wire = sample_wire_action(rng, menu=menu, taxonomy=taxonomy, kind=kind)
        assert isinstance(parse_action(render_wire(wire), menu, taxonomy), cls)


# --------------------------------------------------------------------------- prompt --


def _scrub_global_vocabulary(text: str) -> str:
    """Remove every patient-independent catalog string from the prompt.

    `filter.assert_no_label_leak` exempts the global vocabulary by whole-string equality,
    which is exact. A prompt is one concatenated blob, so the exemption has to be applied
    by removal -- and removal has to be on WORD BOUNDARIES, longest entry first. The
    vocabulary contains "s" and "%", and a plain substring removal of "s" turns "disease"
    into "di ea e", which scrubs away the very leak the positive control injects.

    The exemption matters more than it looks. A respiratory PCR panel returning
    `influenza_positive` for a patient with influenza is not a leak -- it is the intended
    channel, and that value is a possible result for every patient, so its presence
    carries no information about who this is. Without the exemption the check fires on
    the environment working correctly, which is the fastest route to a leak test nobody
    trusts.
    """
    import re

    from dxenv.env.filter import global_vocabulary

    out = text.lower().replace("_", " ")
    for v in sorted(global_vocabulary(), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(v.lower().replace('_', ' '))}\b", " ", out)
    return out


def test_prompt_contains_no_label_string(fixture_corpus, catalog, menu, taxonomy) -> None:
    """The prompt inherits I1 from the observation; check it anyway, over the corpus."""
    from tests.conftest import observe

    for rec in fixture_corpus:
        obs = observe(rec, catalog, menu)
        # The menu and label set are global and name every condition by construction, so
        # the check is on the case sheet -- the only part derived from this record.
        text = _scrub_global_vocabulary(
            build_prompt(obs, menu=menu, taxonomy=taxonomy, include_menu=False)
        )
        for form in taxonomy.get(rec.condition).leak_strings:
            if len(form) < 4:
                continue
            assert f" {form} " not in f" {text} ", (
                f"{form!r} leaked into the prompt for {rec.patient_id}"
            )


def test_prompt_leak_check_actually_fires(fixture_corpus, catalog, menu, taxonomy) -> None:
    """Test the detector. A leak check that cannot fail manufactures confidence."""
    rec = fixture_corpus[0]
    from tests.conftest import observe

    obs = observe(rec, catalog, menu)
    display = taxonomy.get(rec.condition).display
    leaked = obs.model_copy(update={"presenting_complaint": f"known {display}"})
    text = _scrub_global_vocabulary(
        build_prompt(leaked, menu=menu, taxonomy=taxonomy, include_menu=False)
    )
    assert any(
        f" {form} " in f" {text} "
        for form in taxonomy.get(rec.condition).leak_strings
        if len(form) >= 4
    ), "an injected leak went undetected"


def test_prompt_is_a_function_of_the_observation_alone(fixture_corpus, catalog, menu) -> None:
    """Two records sharing an observation must produce the same prompt.

    This is the structural half of I1 for the prompt: the leak check above says no label
    string got through, and this says there is no channel through which one could -- the
    prompt depends on nothing but the typed observation.
    """
    from tests.conftest import observe

    a, b = fixture_corpus[0], fixture_corpus[1]
    assert a.condition != b.condition
    obs_a = observe(a, catalog, menu)
    obs_b = observe(b, catalog, menu).model_copy(
        update={k: getattr(obs_a, k) for k in type(obs_a).model_fields}
    )
    assert build_prompt(obs_a, menu=menu) == build_prompt(obs_b, menu=menu)


def test_menu_in_prompt_is_identical_across_patients(fixture_corpus, catalog, menu) -> None:
    """[I3] If the rendered menu varied with the patient, the menu would be the diagnosis."""
    rendered = {render_menu(menu, catalog) for _ in fixture_corpus[:20]}
    assert len(rendered) == 1


def test_chat_messages_have_system_and_user_roles(fixture_corpus, catalog, menu) -> None:
    from tests.conftest import observe

    msgs = chat_messages(observe(fixture_corpus[0], catalog, menu))
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert all(m["content"] for m in msgs)
