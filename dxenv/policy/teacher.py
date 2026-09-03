"""The privileged teacher, and the de-leaking pass that makes its traces usable.

CLAUDE.md 8.2: the teacher sees ground truth and produces expert test sequences. Then
**strip the privilege** -- regenerate the reasoning conditioned only on what was visible
at that turn, and filter hard for traces that reference the answer before earning it.

> Leaked reasoning in SFT data is worse than no SFT data. It trains the model to assert
> conclusions it has no evidence for, which is precisely the pathology the environment
> exists to prevent.

Why the first pass leaks ON PURPOSE
-----------------------------------
`privileged_trace` writes reasoning that names the condition. That is not sloppiness; it
is what a ground-truth-conditioned teacher actually produces, and generating a clean
trace directly would make `test_no_trace_mentions_condition_before_diagnosis_turn` a test
that a filter removes things that were never there -- the most comfortable kind of green
test and the least informative. `dxenv/data/corpus.py` makes the same choice about leaky
FHIR resources, for the same reason. The privileged trace is the positive control for the
detector, and the suite asserts the detector fires on it.

What "de-leaked" means here, precisely
--------------------------------------
The de-leaked reasoning is REGENERATED, not redacted. Redaction leaves the shape of the
argument intact -- "the elevated ferritin confirms [REDACTED]" still asserts a conclusion
the visible evidence does not support, and a model trained on it learns to assert. So the
turn's reasoning is rebuilt from the posterior over *visible* evidence alone: the same
hypotheses the agent could actually have entertained, with the weights it could actually
have computed. What survives is an argument; what is removed is the answer.

"Never mentions the condition" is not the check you want -- and here is why
---------------------------------------------------------------------------
CLAUDE.md 8.2 asks for a trace that does not "reference the answer before earning it",
and the obvious reading -- the reasoning must never contain the condition's name before
the diagnosis turn -- fails on contact. A de-leaked trace names the leading hypotheses
from the VISIBLE posterior, and the visible posterior is usually right; on a 20-patient
sample this repo's own de-leaker trips a literal string check on 19 of 20 traces, every
one of them clean. Enforcing the literal rule would forbid the model from ever writing a
differential, which is the entire content of diagnostic reasoning.

The distinction that actually matters is counterfactual: is the condition named *because
the teacher knew the answer*, or because the visible evidence put it there? That is
checkable, and this module checks it three ways, in increasing strength:

  `LeakDetector`       literal substring + similarity. The right tool for free text and
                       for the POSITIVE CONTROL -- it must fire on the privileged trace.
  `check_grounding`    every condition named must (a) rank inside the visible posterior's
                       top-n and (b) appear next to a probability that matches that
                       posterior. An assertion of fact carries no probability and is
                       rejected; a hypothesis the evidence does not support ranks too low
                       and is rejected. This is the filter the SFT set actually runs.
  `deleak_ablation`    the statistical backstop, and the only one that survives swapping
                       in an LLM teacher: run the de-leaker on the same observation under
                       a SHUFFLED label and compare mention rates. A label-blind
                       generator mentions the true condition exactly as often as it
                       mentions a randomly assigned one. Any gap is privilege leaking.

The teacher's terminal report is the Bayes posterior, NOT one-hot
-----------------------------------------------------------------
Even though the teacher knows the answer. One-hot targets are how a Phase 3 destroys the
calibration the Brier score exists to reward, before RL starts (CLAUDE.md 8.4). The
teacher's advantage is meant to show up in WHICH TESTS it orders, not in a confidence it
had no way to earn.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from dxenv.data.corpus import PatientRecord
from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import ActionKind, ActionMenu, action_id, build_menu
from dxenv.env.bayes import posterior
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.episode import DiagnosticEpisode, EpisodeConfig, load_episode_config
from dxenv.env.obs_model import ObservationModel, ResultValue, build_observation_model
from dxenv.env.schemas import Action, Diagnose, Observation, OrderTest
from dxenv.policy.baselines import evidence_from_observation
from dxenv.policy.decoding import DEFAULT_MAX_LABELS
from dxenv.reward.verify import actual_bucket, headline_analyte

NGRAM_N: Final = 4
DEFAULT_SIMILARITY_THRESHOLD: Final = 0.55
MIN_REASONING_CHARS: Final = 60
"""Floor under de-leaked reasoning length.

De-leaking that collapses a trace to "ordering a test" has not removed a leak, it has
removed the demonstration. `test_deleaked_trace_still_justifies_action` asserts against
this; CLAUDE.md 8.2 asks for exactly that check.
"""


class TeacherError(ValueError):
    """The teacher could not produce a usable trace. Never caught inside `dxenv.policy`."""


# --------------------------------------------------------------------- leak checks ----


def word_boundary_hit(text_lower: str, form: str) -> int | None:
    """Index of `form` in `text_lower` as a WHOLE WORD, or None.

    Raw substring matching is what the observation scrubber uses, and there it is right:
    an observation is machine-generated, so a false positive costs a scrubbed display
    string and a false negative is a leak. Prose is the opposite case. The synonym list
    contains short forms -- "mi", "dm", "af" -- and `"mi" in "commit"` is true, which
    made the first version of this filter reject 45 of 60 clean traces for mentioning
    acute myocardial infarction inside the word "commit". A filter that rejects
    three-quarters of a clean SFT set does not get tightened; it gets switched off.
    """
    for m in re.finditer(rf"\b{re.escape(form)}\b", text_lower):
        return int(m.start())
    return None


def _char_ngrams(text: str, n: int = NGRAM_N) -> set[str]:
    s = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))} or {s}


def ngram_similarity(a: str, b: str, n: int = NGRAM_N) -> float:
    """Character-n-gram Jaccard. The default stand-in for an embedding check.

    Deliberately cheap and deliberately *conservative in the right direction*: it fires on
    near-misses ("diabetes mellitus type 2" vs "type 2 diabetes") that exact substring
    matching sails past, at the price of some false positives. A false positive drops one
    SFT trace; a false negative trains the model to assert. Those costs are not symmetric.

    Swap in a real sentence embedder via `LeakDetector.embedder` when one is available --
    the interface is (str, str) -> float in [0, 1].
    """
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    inter = len(ga & gb)
    union = len(ga | gb)
    return float(inter / union) if union else 0.0


@dataclass(frozen=True, slots=True)
class LeakFinding:
    turn: int
    kind: str
    matched: str
    score: float
    excerpt: str

    def line(self) -> str:
        return (
            f"turn {self.turn}: {self.kind} on {self.matched!r} "
            f"(score {self.score:.2f}) -- {self.excerpt!r}"
        )


@dataclass(slots=True)
class LeakDetector:
    """String and similarity checks against a condition's every known surface form.

    Runs over the WHOLE SFT set and fails on any hit (CLAUDE.md 8.2), not over a sample:
    a leak in 2% of traces is still a leak, and it is the 2% the model will learn the
    pathology from.
    """

    taxonomy: Taxonomy = field(default_factory=load_taxonomy)
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    embedder: Callable[[str, str], float] = ngram_similarity

    def check(self, text: str, condition: str, turn: int = 0) -> list[LeakFinding]:
        """Every leak of `condition` in `text`. Empty list means clean."""
        if not text:
            return []
        lowered = text.lower()
        label = self.taxonomy.get(condition)
        findings: list[LeakFinding] = []
        for form in label.leak_strings:
            idx = word_boundary_hit(lowered, form) if form else None
            if idx is not None:
                findings.append(
                    LeakFinding(
                        turn=turn, kind="substring", matched=form, score=1.0,
                        excerpt=text[max(0, idx - 30) : idx + len(form) + 30],
                    )
                )
        if findings:
            return findings
        # Similarity is checked per sentence: a 1200-character trace dilutes any single
        # phrase below any sensible whole-document threshold.
        for sentence in re.split(r"[.;\n]", text):
            s = sentence.strip()
            if len(s) < 8:
                continue
            for form in label.leak_strings:
                score = self.embedder(s, form)
                if score >= self.threshold:
                    findings.append(
                        LeakFinding(turn=turn, kind="similarity", matched=form,
                                    score=score, excerpt=s)
                    )
        return findings


# ------------------------------------------------------------------------- traces ----


@dataclass(frozen=True, slots=True)
class TeacherTurn:
    turn: int
    observation: Observation
    action: Action
    reasoning: str
    privileged: bool

    def wire(self) -> dict[str, Any]:
        """The turn as the model would have emitted it, for SFT."""
        a = self.action
        if isinstance(a, OrderTest):
            return {
                "kind": "order_test", "reasoning": self.reasoning,
                "test_key": a.test_key, "prediction": a.prediction,
            }
        if isinstance(a, Diagnose):
            top = sorted(a.distribution.items(), key=lambda kv: -kv[1])[:DEFAULT_MAX_LABELS]
            return {
                "kind": "diagnose", "reasoning": self.reasoning,
                "diagnosis": [
                    {"condition": k, "probability": round(float(v), 6)} for k, v in top
                ],
            }
        return {"kind": a.kind, "reasoning": self.reasoning}


@dataclass(frozen=True, slots=True)
class TeacherTrace:
    patient_id: str
    condition: str
    turns: tuple[TeacherTurn, ...]
    trajectory: dict[str, Any]
    privileged: bool

    @property
    def diagnosis_turn(self) -> int:
        for t in self.turns:
            if isinstance(t.action, Diagnose):
                return t.turn
        return len(self.turns)


# ------------------------------------------------------------------ the teacher ----


@dataclass(slots=True)
class PrivilegedTeacher:
    """Picks tests using the true condition. Reports the posterior it actually earned.

    Test choice is by likelihood ratio AT THE VALUE THIS PATIENT WILL RETURN -- how much
    the result the patient is going to give raises the true condition's log-posterior,
    per unit cost. That is privilege used the way it should be: to demonstrate WHICH
    tests were worth ordering for a case like this one, which is the thing a student
    policy cannot work out for itself and the only thing worth imitating.
    """

    max_tests: int = 5
    min_gain: float = 0.15
    """Stop once the next-best test buys less than this in log-odds per unit cost.

    A teacher that orders until the budget runs out demonstrates shotgun testing, and
    that habit is stubborn once trained in (CLAUDE.md 8.3)."""

    taxonomy: Taxonomy = field(default_factory=load_taxonomy)
    catalog: Catalog = field(default_factory=load_catalog)
    model: ObservationModel = field(default_factory=build_observation_model)
    menu: ActionMenu = field(default_factory=build_menu)
    cost_weight: float = 0.5

    def _log_gain(self, analyte: str, value: ResultValue, true_idx: int,
                  belief: np.ndarray) -> float:
        """Increase in the true condition's log-posterior from this exact result."""
        ll = self.model.log_likelihood_vector(analyte, value)
        shifted = ll - ll.max()
        w = belief * np.exp(shifted)
        total = float(w.sum())
        if total <= 0.0:
            return 0.0
        return float(np.log(max(w[true_idx] / total, 1e-300)) -
                     np.log(max(belief[true_idx], 1e-300)))

    def choose(
        self, record: PatientRecord, episode: DiagnosticEpisode, obs: Observation
    ) -> tuple[Action, float]:
        """The teacher's action for this turn, plus the gain that justified it."""
        belief = posterior(evidence_from_observation(obs), self.model)
        true_idx = self.taxonomy.index(record.condition)
        if len(episode.state.ordered) >= self.max_tests:
            return self._report(belief), 0.0

        best: tuple[float, str] | None = None
        for key in self.catalog.test_keys:
            if key in episode.state.ordered:
                continue
            cost = episode.config.cost_of(key)
            if cost > episode.remaining_budget:
                continue
            analyte = self.catalog.test(key).analytes[0]
            gain = self._log_gain(analyte, record.analytes[analyte], true_idx, belief)
            per_cost = gain / max(cost, 1e-9) ** self.cost_weight
            if best is None or per_cost > best[0]:
                best = (per_cost, key)
        if best is None or best[0] < self.min_gain:
            return self._report(belief), 0.0
        key = best[1]
        analyte = headline_analyte(key, self.catalog)
        # The teacher's predict-then-verify commitment is made from the value it can see.
        # That IS privileged, and it is one of the things de-leaking has to handle: a
        # student cannot copy the commitment, only the habit of making one.
        prediction = actual_bucket(analyte, record.analytes[analyte], self.catalog)
        return OrderTest(
            action_id=self.menu.id_for_test(key), test_key=key, prediction=prediction
        ), best[0]

    def _report(self, belief: np.ndarray) -> Diagnose:
        """The posterior, not a one-hot. See the module docstring."""
        order = np.argsort(-belief)
        raw = {self.taxonomy.slugs[int(i)]: float(belief[int(i)]) for i in order}
        total = sum(raw.values())
        return Diagnose(
            action_id=action_id(ActionKind.DIAGNOSE, "diagnose"),
            distribution={k: v / total for k, v in raw.items()},
        )


def _privileged_reasoning(
    record: PatientRecord, action: Action, gain: float, taxonomy: Taxonomy
) -> str:
    """Reasoning as a ground-truth-conditioned teacher writes it. LEAKS, deliberately."""
    display = taxonomy.get(record.condition).display
    if isinstance(action, OrderTest):
        return (
            f"The patient has {display}. To make that case from the record I order "
            f"{action.test_key}, which for {display} is the most discriminative result "
            f"available here (log-odds gain {gain:.2f} per unit cost), and I expect it to "
            f"come back {action.prediction}."
        )
    return (
        f"The findings are those of {display}, so I report a distribution concentrated "
        f"on {display} to the extent the evidence gathered so far supports it."
    )


def _deleaked_reasoning(
    obs: Observation, action: Action, belief: np.ndarray, taxonomy: Taxonomy
) -> str:
    """Reasoning rebuilt from the visible posterior alone. Never sees the condition.

    Takes `belief` -- which is `posterior(evidence_from_observation(obs))`, computable by
    any agent from the same observation -- and nothing else about the patient.
    """
    order = np.argsort(-belief)[:3]
    top = ", ".join(
        f"{taxonomy.slugs[int(i)]} {belief[int(i)]:.2f}" for i in order
    )
    seen = len(obs.revealed_results)
    context = (
        f"{obs.demographics.age_years}y {obs.demographics.sex} presenting with "
        f"{obs.presenting_complaint}"
    )
    if isinstance(action, OrderTest):
        return (
            f"{context}. On {seen} result(s) so far the leading hypotheses are {top}. "
            f"These are not yet separated by what I can see, so I order {action.test_key} "
            f"and commit to {action.prediction} for its headline result; if that "
            f"commitment is wrong the ordering above is wrong too, and I will know it "
            f"next turn. Remaining budget {obs.remaining_budget:g}."
        )
    return (
        f"{context}. After {seen} result(s) my belief is {top}. Further testing would "
        f"cost more than the sharpening it buys at this budget, so I report the "
        f"distribution I actually hold rather than committing harder than the evidence "
        f"supports."
    )


def privileged_trace(
    record: PatientRecord,
    seed: int,
    teacher: PrivilegedTeacher | None = None,
    config: EpisodeConfig | None = None,
    budget: float | None = None,
) -> TeacherTrace:
    """Run the teacher with its privilege intact. The output LEAKS; de-leak it before use."""
    t = teacher or PrivilegedTeacher()
    cfg = config or load_episode_config()
    episode = DiagnosticEpisode(record, seed=seed, config=cfg, catalog=t.catalog, budget=budget)
    obs = episode.reset()
    turns: list[TeacherTurn] = []
    done = False
    while not done:
        action, gain = t.choose(record, episode, obs)
        turns.append(
            TeacherTurn(
                turn=obs.turn + 1,
                observation=obs,
                action=action,
                reasoning=_privileged_reasoning(record, action, gain, t.taxonomy),
                privileged=True,
            )
        )
        obs, done, _ = episode.step(action)
    return TeacherTrace(
        patient_id=record.patient_id,
        condition=record.condition,
        turns=tuple(turns),
        trajectory=episode.trajectory(),
        privileged=True,
    )


def deleak(
    trace: TeacherTrace,
    model: ObservationModel | None = None,
    taxonomy: Taxonomy | None = None,
) -> TeacherTrace:
    """Regenerate every turn's reasoning from what was visible at that turn.

    The action sequence is kept -- that is the teacher's contribution and it is not a
    leak, because *which* test was ordered is a decision the student is meant to imitate.
    What is discarded is every justification that referenced the answer.
    """
    m = model or build_observation_model()
    tax = taxonomy or load_taxonomy()
    turns = tuple(
        TeacherTurn(
            turn=t.turn,
            observation=t.observation,
            action=t.action,
            reasoning=_deleaked_reasoning(
                t.observation,
                t.action,
                posterior(evidence_from_observation(t.observation), m),
                tax,
            ),
            privileged=False,
        )
        for t in trace.turns
    )
    return TeacherTrace(
        patient_id=trace.patient_id,
        condition=trace.condition,
        turns=turns,
        trajectory=trace.trajectory,
        privileged=False,
    )


def audit_trace(trace: TeacherTrace, detector: LeakDetector | None = None) -> list[LeakFinding]:
    """Literal leak check over every turn. The POSITIVE CONTROL for the de-leaker.

    Must fire on a privileged trace. It also fires on clean traces that name a supported
    differential, which is why it is not the SFT filter -- see `check_grounding`.
    """
    det = detector or LeakDetector()
    out: list[LeakFinding] = []
    for t in trace.turns:
        out.extend(det.check(t.reasoning, trace.condition, turn=t.turn))
    return out


def mentioned_conditions(text: str, taxonomy: Taxonomy) -> set[str]:
    """Every taxonomy label whose surface form appears in `text`."""
    lowered = text.lower()
    return {
        lab.slug
        for lab in taxonomy.labels
        if any(form and word_boundary_hit(lowered, form) is not None for form in lab.leak_strings)
    }


_PROB_NEAR: Final = 60
"""Characters after a mention within which its probability must appear.

Wide enough for "iron_deficiency_anemia 0.12" and for a sentence that puts the number
after a clause; narrow enough that a probability belonging to a different hypothesis
three items down the list does not launder an ungrounded assertion."""


def check_grounding(
    turn: TeacherTurn,
    belief: np.ndarray,
    taxonomy: Taxonomy,
    top_n: int = 3,
    tolerance: float = 0.02,
) -> list[LeakFinding]:
    """Every condition named must be one the visible evidence actually ranked.

    Two failure modes, one check each:

      unearned_mention     the condition is named but sits outside the visible
                           posterior's top-n. The evidence did not put it there, so
                           something else did.
      ungrounded_assertion the condition is named with no probability attached, or with
                           one that does not match the posterior. "The patient has X" is
                           an assertion of fact; a model trained on assertions learns to
                           assert, which is the pathology the whole environment exists to
                           prevent.

    `belief` must be the posterior over the turn's OWN observation -- what the agent
    could have computed -- not the full-information posterior.
    """
    ranked = [taxonomy.slugs[int(i)] for i in np.argsort(-belief)[:top_n]]
    findings: list[LeakFinding] = []
    lowered = turn.reasoning.lower()
    for slug in sorted(mentioned_conditions(turn.reasoning, taxonomy)):
        idx = taxonomy.index(slug)
        hits = [
            word_boundary_hit(lowered, f) for f in taxonomy.get(slug).leak_strings if f
        ]
        at = min(h for h in hits if h is not None)
        window = turn.reasoning[at : at + _PROB_NEAR]
        numbers = [float(m) for m in re.findall(r"\d*\.\d+", window)]
        if slug not in ranked:
            findings.append(
                LeakFinding(turn.turn, "unearned_mention", slug, float(belief[idx]),
                            window.strip())
            )
        elif not any(abs(n - float(belief[idx])) <= tolerance for n in numbers):
            findings.append(
                LeakFinding(turn.turn, "ungrounded_assertion", slug, float(belief[idx]),
                            window.strip())
            )
    return findings


def audit_trace_grounded(
    trace: TeacherTrace,
    model: ObservationModel | None = None,
    taxonomy: Taxonomy | None = None,
    top_n: int = 3,
) -> list[LeakFinding]:
    """`check_grounding` over a whole trace, excluding the terminal report.

    The diagnosis turn is exempt: naming the condition IS the action there, and the
    distribution it reports is scored by a proper rule rather than filtered by a string
    check.
    """
    m = model or build_observation_model()
    tax = taxonomy or load_taxonomy()
    out: list[LeakFinding] = []
    for t in trace.turns:
        if isinstance(t.action, Diagnose):
            continue
        belief = posterior(evidence_from_observation(t.observation), m)
        out.extend(check_grounding(t, belief, tax, top_n=top_n))
    return out


def filter_traces(
    traces: Sequence[TeacherTrace],
    model: ObservationModel | None = None,
    taxonomy: Taxonomy | None = None,
    top_n: int = 3,
) -> tuple[list[TeacherTrace], list[tuple[str, list[LeakFinding]]]]:
    """Split into (clean, rejected-with-reasons). Rejection is loud, never silent.

    Returning the reasons rather than a count is the difference between "we dropped 4%"
    and "we dropped 4% because the de-leaker misses possessives", and only one of those
    can be acted on.
    """
    m = model or build_observation_model()
    tax = taxonomy or load_taxonomy()
    clean: list[TeacherTrace] = []
    rejected: list[tuple[str, list[LeakFinding]]] = []
    for tr in traces:
        findings = audit_trace_grounded(tr, m, tax, top_n=top_n)
        short = [t for t in tr.turns if len(t.reasoning) < MIN_REASONING_CHARS]
        if short:
            findings = [
                *findings,
                LeakFinding(short[0].turn, "stub_reasoning", "", 0.0, short[0].reasoning),
            ]
        if findings:
            rejected.append((tr.patient_id, findings))
        else:
            clean.append(tr)
    return clean, rejected


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Mention rates under the true label and under a shuffled one."""

    true_rate: float
    shuffled_rate: float
    n: int

    @property
    def gap(self) -> float:
        """Excess mentions attributable to privilege. Zero for a label-blind de-leaker."""
        return self.true_rate - self.shuffled_rate

    def line(self) -> str:
        return (
            f"de-leak ablation over {self.n} pre-diagnosis turns: true condition "
            f"mentioned {self.true_rate:.3f}, shuffled condition {self.shuffled_rate:.3f}, "
            f"gap {self.gap:+.3f}"
        )


def deleak_ablation(
    traces: Sequence[TeacherTrace],
    rng: np.random.Generator,
    model: ObservationModel | None = None,
    taxonomy: Taxonomy | None = None,
) -> AblationResult:
    """Is the de-leaker label-blind? The check that survives swapping in an LLM teacher.

    Choosing the null is the whole difficulty, and the obvious null is wrong. Comparing
    against a UNIFORMLY random condition reports a gap of +0.63 on a de-leaker that is
    label-blind by construction, because the true condition is usually near the top of
    the visible posterior -- which is the environment working, not privilege leaking. A
    check that fails on correct behaviour teaches you to ignore it.

    The right null holds RANK fixed: draw the comparison condition from the visible
    posterior itself. Then both rates answer the same question -- "how often does a
    condition of this much posterior mass get named" -- and the only thing that can
    separate them is the generator knowing which one is true.

    See `deleak_is_label_blind` for the exact version, which this module's own
    deterministic de-leaker can satisfy outright.
    """
    m = model or build_observation_model()
    tax = taxonomy or load_taxonomy()
    true_hits = null_hits = n = 0
    for tr in traces:
        for t in tr.turns:
            if isinstance(t.action, Diagnose):
                continue
            belief = posterior(evidence_from_observation(t.observation), m)
            mentioned = mentioned_conditions(t.reasoning, tax)
            drawn = str(rng.choice(np.array(tax.slugs), p=belief))
            true_hits += int(tr.condition in mentioned)
            null_hits += int(drawn in mentioned)
            n += 1
    if n == 0:
        raise TeacherError("no pre-diagnosis turns to ablate over")
    return AblationResult(true_hits / n, null_hits / n, n)


def deleak_is_label_blind(
    trace: TeacherTrace,
    taxonomy: Taxonomy | None = None,
    model: ObservationModel | None = None,
) -> bool:
    """Exact check: does the de-leaked reasoning change if the true label changes?

    Stronger than any rate comparison, and available here because `deleak` is
    deterministic and takes the observation alone. Re-running it against a trace whose
    condition has been swapped must produce byte-identical reasoning -- if it does not,
    the label reached the generator through some path, and the ablation above is then the
    only tool left.
    """
    tax = taxonomy or load_taxonomy()
    other = next(s for s in tax.slugs if s != trace.condition)
    swapped = TeacherTrace(
        patient_id=trace.patient_id, condition=other, turns=trace.turns,
        trajectory=trace.trajectory, privileged=trace.privileged,
    )
    a = deleak(trace, model=model, taxonomy=tax)
    b = deleak(swapped, model=model, taxonomy=tax)
    return [t.reasoning for t in a.turns] == [t.reasoning for t in b.turns]
