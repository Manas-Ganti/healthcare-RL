"""The privileged teacher and the de-leaking pass (CLAUDE.md 8.2).

Leaked reasoning in SFT data is worse than no SFT data: it trains the model to assert
conclusions it has no evidence for, which is precisely the pathology the environment
exists to prevent. Every test here is about whether the de-leaker actually worked, and
several are about whether the DETECTORS would notice if it had not.
"""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.env.bayes import posterior
from dxenv.env.schemas import Diagnose, OrderTest
from dxenv.policy.baselines import evidence_from_observation
from dxenv.policy.teacher import (
    MIN_REASONING_CHARS,
    LeakDetector,
    audit_trace,
    audit_trace_grounded,
    check_grounding,
    deleak,
    deleak_ablation,
    deleak_is_label_blind,
    filter_traces,
    mentioned_conditions,
    privileged_trace,
    word_boundary_hit,
)

N = 40


@pytest.fixture(scope="module")
def privileged(fixture_corpus):
    return [
        privileged_trace(r, seed=i, budget=150.0) for i, r in enumerate(fixture_corpus[:N])
    ]


@pytest.fixture(scope="module")
def deleaked(privileged):
    return [deleak(t) for t in privileged]


# ------------------------------------------------------------- the positive control --


def test_privileged_trace_leaks_and_the_detector_fires(privileged) -> None:
    """The teacher MUST leak, or the de-leaker is being tested against nothing.

    A filter that removes things which were never there is the most comfortable kind of
    green test and the least informative. `data/corpus.py` makes the same choice about
    leaky FHIR resources, for the same reason.
    """
    findings = sum(len(audit_trace(t)) for t in privileged)
    assert findings > 0, "the privileged teacher did not leak; it is not privileged"


def test_grounding_check_fires_on_privileged_traces(privileged) -> None:
    """The SFT filter must also catch the privileged trace, not only the literal check."""
    assert sum(len(audit_trace_grounded(t)) for t in privileged) > 0


# ------------------------------------------------------------------ after de-leaking --


def test_no_trace_mentions_condition_before_diagnosis_turn(deleaked) -> None:
    """CLAUDE.md 8.2, in the form that is actually checkable.

    The literal reading -- the reasoning must never contain the condition's name -- fails
    on contact: a de-leaked trace names the leading hypotheses from the VISIBLE posterior,
    and the visible posterior is usually right. Enforcing it literally rejects almost
    every clean trace and forbids the model from ever writing a differential.

    What must hold instead is that every condition named was one the visible evidence
    ranked, WITH its posterior probability attached. An assertion of fact carries no
    probability; a hypothesis the evidence does not support ranks too low.
    """
    for trace in deleaked:
        assert audit_trace_grounded(trace) == [], (
            f"{trace.patient_id}: {[f.line() for f in audit_trace_grounded(trace)][:2]}"
        )


def test_deleaked_traces_all_survive_the_filter(deleaked) -> None:
    clean, rejected = filter_traces(deleaked)
    assert len(clean) == len(deleaked), [r[1][0].line() for r in rejected[:3]]


def test_deleak_is_label_blind_exactly(privileged) -> None:
    """The strongest form: swap the true label, get byte-identical reasoning."""
    assert all(deleak_is_label_blind(t) for t in privileged)


def test_deleak_ablation_gap_is_near_zero(deleaked) -> None:
    """The statistical backstop, with the null drawn from the visible posterior.

    Holding rank fixed is the whole point. A uniformly random null reports a gap of +0.6
    on a de-leaker that is label-blind by construction, because the true condition is
    usually near the top of the posterior -- that is the environment working, not
    privilege leaking, and a check that fails on correct behaviour gets ignored.
    """
    result = deleak_ablation(deleaked, np.random.default_rng(0))
    assert abs(result.gap) < 0.15, result.line()


def test_deleak_ablation_detects_the_privileged_trace(privileged) -> None:
    """Test the detector: the ablation must separate privileged from de-leaked."""
    result = deleak_ablation(privileged, np.random.default_rng(0))
    assert result.gap > 0.3, result.line()


def test_deleaked_trace_still_justifies_action(deleaked) -> None:
    """De-leaking that collapses reasoning to a stub removed the demonstration, not a leak."""
    for trace in deleaked:
        for turn in trace.turns:
            assert len(turn.reasoning) >= MIN_REASONING_CHARS, (
                f"{trace.patient_id} turn {turn.turn}: {turn.reasoning!r}"
            )


def test_deleaking_preserves_the_action_sequence(privileged, deleaked) -> None:
    """WHICH test was ordered is the teacher's contribution and is not a leak."""
    for a, b in zip(privileged, deleaked, strict=True):
        assert [type(t.action) for t in a.turns] == [type(t.action) for t in b.turns]
        for x, y in zip(a.turns, b.turns, strict=True):
            if isinstance(x.action, OrderTest):
                assert x.action.test_key == y.action.test_key


# ------------------------------------------------------------------ the teacher itself --


def test_teacher_report_is_the_posterior_not_one_hot(deleaked, taxonomy, obs_model) -> None:
    """One-hot targets destroy calibration before RL starts (CLAUDE.md 8.4)."""
    from dxenv.env.bayes import entropy

    entropies = []
    for trace in deleaked:
        for turn in trace.turns:
            if isinstance(turn.action, Diagnose):
                belief = posterior(evidence_from_observation(turn.observation), obs_model)
                declared = np.array(
                    [turn.action.distribution[s] for s in taxonomy.slugs], dtype=np.float64
                )
                assert np.allclose(declared, belief, atol=1e-9)
                entropies.append(entropy(declared))
    assert float(np.mean(entropies)) > 0.1, "the teacher's reports collapsed to one-hot"


def test_teacher_trajectories_respect_action_schema(privileged, menu) -> None:
    for trace in privileged:
        for turn in trace.turns:
            assert turn.action.action_id in menu.ids


def test_teacher_does_not_shotgun(privileged) -> None:
    """A trajectory that got it right after 40 tests is a bad demonstration."""
    for trace in privileged:
        orders = [t for t in trace.turns if isinstance(t.action, OrderTest)]
        assert len(orders) <= 5


# ------------------------------------------------------------------------ the checks --


def test_word_boundary_matching_does_not_fire_inside_words() -> None:
    """The synonym "mi" inside "commit" rejected three-quarters of a clean SFT set."""
    assert word_boundary_hit("i commit to normal", "mi") is None
    assert word_boundary_hit("suspected mi on the ecg", "mi") is not None


def test_mentioned_conditions_finds_real_mentions(taxonomy) -> None:
    slug = taxonomy.slugs[0]
    assert slug in mentioned_conditions(f"leading hypothesis is {slug} 0.40", taxonomy)
    assert mentioned_conditions("nothing here at all", taxonomy) == set()


def test_grounding_rejects_an_assertion_without_a_probability(deleaked, taxonomy,
                                                              obs_model) -> None:
    """Test the detector: "the patient has X" must be caught even when X ranks first."""
    from dxenv.policy.teacher import TeacherTurn

    trace = deleaked[0]
    turn = trace.turns[0]
    belief = posterior(evidence_from_observation(turn.observation), obs_model)
    top = taxonomy.slugs[int(np.argmax(belief))]
    forged = TeacherTurn(
        turn=turn.turn, observation=turn.observation, action=turn.action,
        reasoning=f"The patient has {top} and I am ordering confirmation.",
        privileged=False,
    )
    findings = check_grounding(forged, belief, taxonomy)
    assert [f.kind for f in findings] == ["ungrounded_assertion"]


def test_grounding_rejects_an_unranked_mention(deleaked, taxonomy, obs_model) -> None:
    from dxenv.policy.teacher import TeacherTurn

    trace = deleaked[0]
    turn = trace.turns[0]
    belief = posterior(evidence_from_observation(turn.observation), obs_model)
    worst = taxonomy.slugs[int(np.argmin(belief))]
    forged = TeacherTurn(
        turn=turn.turn, observation=turn.observation, action=turn.action,
        reasoning=f"I think this is {worst} 0.99 despite the evidence.",
        privileged=False,
    )
    assert [f.kind for f in check_grounding(forged, belief, taxonomy)] == ["unearned_mention"]


def test_leak_detector_similarity_arm_catches_paraphrase(taxonomy) -> None:
    """Substring matching sails past a reordered phrase; the n-gram arm should not."""
    det = LeakDetector(taxonomy=taxonomy, threshold=0.4)
    slug = "type_2_diabetes" if "type_2_diabetes" in taxonomy.slugs else taxonomy.slugs[0]
    display = taxonomy.get(slug).display
    scrambled = " ".join(reversed(display.split()))
    if scrambled.lower() == display.lower():
        pytest.skip("single-word display name; nothing to reorder")
    assert det.check(f"findings suggest {scrambled} here", slug)
