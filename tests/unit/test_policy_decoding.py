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


# ------------------------------------------------------------- vLLM API compatibility --


def _fake_vllm(kwarg: str, cls_name: str):
    """A stand-in vLLM exposing one generation of the structured-output API."""
    import inspect
    import types

    sp = types.ModuleType("vllm.sampling_params")

    class Params:
        def __init__(self, json):
            self.json = json

    Params.__name__ = cls_name
    setattr(sp, cls_name, Params)

    # Name mirrors vLLM's class, which is what makes the fake a faithful stand-in.
    def SamplingParams(**kw):  # noqa: N802  # pragma: no cover - only its signature is read
        return kw

    SamplingParams.__signature__ = inspect.Signature(
        [
            inspect.Parameter(p, inspect.Parameter.KEYWORD_ONLY, default=None)
            for p in ("n", "temperature", "max_tokens", "seed", kwarg)
        ]
    )
    v = types.ModuleType("vllm")
    v.SamplingParams = SamplingParams
    v.sampling_params = sp
    return v, sp


@pytest.mark.parametrize(
    ("kwarg", "cls_name"),
    [
        ("structured_outputs", "StructuredOutputsParams"),
        ("guided_decoding", "GuidedDecodingParams"),
    ],
)
def test_structured_output_api_is_detected_not_pinned(monkeypatch, kwarg, cls_name) -> None:
    """vLLM renamed this between the version this was written for and the one deployed.

    A version check would encode today's cutover and break at the next rename somewhere
    nobody would look. Asking the installed SamplingParams which keyword it accepts stays
    correct across both, which is what this asserts -- on fakes, because the real check
    needs a CUDA host and this has to run in the fast suite.
    """
    from dxenv.policy.llm import VLLMBackend

    v, sp = _fake_vllm(kwarg, cls_name)
    monkeypatch.setitem(__import__("sys").modules, "vllm", v)
    monkeypatch.setitem(__import__("sys").modules, "vllm.sampling_params", sp)

    got = VLLMBackend(model="x")._structured_output_kwargs()
    assert list(got) == [kwarg]
    assert type(got[kwarg]).__name__ == cls_name


def test_unknown_structured_output_api_raises_rather_than_degrading(monkeypatch) -> None:
    """Constrained decoding is not optional: unenforced, the grammar stops holding I3.

    So an unrecognised API must fail loudly rather than fall back to free generation,
    which would silently produce off-menu actions and a format reward that is no longer
    zero.
    """
    from dxenv.policy.llm import BackendError, VLLMBackend

    v, sp = _fake_vllm("nothing_relevant", "SomethingElse")
    monkeypatch.setitem(__import__("sys").modules, "vllm", v)
    monkeypatch.setitem(__import__("sys").modules, "vllm.sampling_params", sp)

    with pytest.raises(BackendError, match="neither"):
        VLLMBackend(model="x")._structured_output_kwargs()


def test_token_budget_covers_the_widest_action(taxonomy) -> None:
    """The budget must fit a full `diagnose`, which is the worst case by a wide margin.

    512 did not, and the resulting truncated prefix was reported as "the grammar was not
    applied" -- structurally valid JSON, unfinished string, and a diagnosis pointing at
    the wrong component entirely.
    """
    import json

    import numpy as np
    from dxenv.policy.decoding import (
        DEFAULT_MAX_LABELS,
        DEFAULT_MAX_TOKENS,
        MAX_REASONING_CHARS,
    )
    from dxenv.policy.sft import soft_label_wire

    belief = np.full(len(taxonomy), 1.0 / len(taxonomy))
    widest = soft_label_wire(belief, "x" * MAX_REASONING_CHARS, taxonomy, DEFAULT_MAX_LABELS)
    serialized = json.dumps(widest, separators=(",", ":"))
    # ~4 characters per token is the usual rule of thumb for English + JSON punctuation.
    assert len(serialized) / 4 < DEFAULT_MAX_TOKENS, (
        f"widest action is ~{len(serialized) // 4} tokens against a budget of "
        f"{DEFAULT_MAX_TOKENS}"
    )


def test_truncated_generation_is_reported_as_truncation(fixture_corpus, catalog, menu,
                                                        episode_config) -> None:
    """Test the diagnosis, not just the failure: a length-stop must not read as a grammar
    failure, and must not be silently replaced by a fallback action."""
    from dxenv.env.episode import DiagnosticEpisode
    from dxenv.policy.decoding import DecodingError
    from dxenv.policy.llm import Generation, LLMPolicy

    class AlwaysTruncates:
        def generate(self, conversations, **kw):  # noqa: ARG002 - Backend protocol
            return [[Generation(text='{"kind":"order_test","reason', finish_reason="length")]]

    policy = LLMPolicy(backend=AlwaysTruncates(), menu=menu)
    episode = DiagnosticEpisode(fixture_corpus[0], seed=0, config=episode_config,
                                catalog=catalog, budget=100.0)
    obs = episode.reset()
    with pytest.raises(DecodingError, match="truncated prefix"):
        policy.act(episode, obs)


def test_truncation_is_retried_once_before_failing(fixture_corpus, catalog, menu,
                                                   episode_config) -> None:
    """Finishing the same constrained generation is not choosing a different action."""
    from dxenv.env.episode import DiagnosticEpisode
    from dxenv.policy.llm import Generation, LLMPolicy

    class TruncatesOnce:
        def __init__(self):
            self.calls = 0

        def generate(self, conversations, **kw):  # noqa: ARG002 - Backend protocol
            self.calls += 1
            if self.calls == 1:
                return [[Generation(text='{"kind":"abst', finish_reason="length")]]
            return [[Generation(text='{"kind":"abstain","reasoning":"budget exhausted"}',
                                finish_reason="stop")]]

    backend = TruncatesOnce()
    policy = LLMPolicy(backend=backend, menu=menu)
    episode = DiagnosticEpisode(fixture_corpus[0], seed=0, config=episode_config,
                                catalog=catalog, budget=100.0)
    action = policy.act(episode, episode.reset())
    assert backend.calls == 2
    assert action.kind == "abstain"


def test_reasoning_length_is_bounded_by_the_grammar_not_just_documented(schema) -> None:
    """`maxLength` is advisory; `pattern` is compiled into the grammar and actually binds.

    This distinction cost two GPU jobs. With only maxLength the decoder happily ran past
    2876 tokens mid-sentence, because grammar backends constrain structure and not string
    length -- so the cap has to be expressed as something the grammar can enforce.
    """
    import re

    from dxenv.policy.decoding import MAX_REASONING_CHARS

    for variant in schema["oneOf"]:
        prop = variant["properties"]["reasoning"]
        assert "pattern" in prop, "reasoning is length-bounded only by a comment"
        rx = re.compile(prop["pattern"])
        assert rx.match("Ordering a CBC to separate anemia from infection.")
        assert rx.match("x" * MAX_REASONING_CHARS)
        assert not rx.match("x" * (MAX_REASONING_CHARS + 1))


def test_sampled_actions_still_satisfy_the_bounded_schema(menu, taxonomy, schema) -> None:
    """The grammar sampler must keep producing schema-valid output under the new pattern."""
    rng = np.random.default_rng(7)
    validator = jsonschema.Draft202012Validator(schema)
    for _ in range(200):
        wire = sample_wire_action(rng, menu=menu, taxonomy=taxonomy)
        validator.validate(wire)


def test_reasoning_pattern_excludes_control_characters(schema) -> None:
    """A raw newline inside a JSON string is invalid JSON, and the first pattern allowed it.

    `[^"\\\\]` bounds length and blocks quotes, and still lets the model emit a literal
    newline -- structurally fine to the grammar, rejected by every JSON parser. Excluding
    backslash also means the model cannot write the escape, so the characters themselves
    have to be forbidden.
    """
    import re

    for variant in schema["oneOf"]:
        rx = re.compile(variant["properties"]["reasoning"]["pattern"])
        assert rx.match("Ordering a CBC to separate anemia from infection.")
        assert rx.match("其他内容")  # non-ASCII prose is fine
        assert not rx.match("text\nmore")
        assert not rx.match("text\tmore")
        assert not rx.match("text\rmore")


def test_control_characters_do_not_kill_a_run(menu, taxonomy) -> None:
    """Belt to the pattern's braces: backends vary in how much regex they honour.

    A raw newline in a reasoning string is unambiguous, so parsing it changes nothing
    about which action was chosen -- unlike a fallback action, which would substitute a
    decision the policy never made and is still refused.
    """
    raw = '{"kind": "abstain", "reasoning": "belief is diffuse\nand the budget is spent"}'
    assert parse_action(raw, menu, taxonomy).kind == "abstain"
