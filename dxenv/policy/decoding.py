"""Constrained decoding: the JSON grammar, and the wire -> `Action` parser.

CLAUDE.md 8.1: format is not an SFT problem. A JSON schema per action type makes invalid
output *impossible*, which keeps the format reward at exactly zero and -- the part that
actually matters -- **preserves the entropy GRPO needs**. SFT on format burns diversity,
and GRPO computes advantages from within-group variation; identical rollouts give zero
gradient.

The wire schema is deliberately NARROWER than `env.schemas.Action`
--------------------------------------------------------------------
The model never emits an `action_id`. It emits a `test_key` (or `treatment_key`) drawn
from an enum of exactly the menu's keys, and `parse_action` derives the content-addressed
id from the menu. Two consequences, both load-bearing:

  * an off-menu action is not merely rejected, it is **unrepresentable** -- the grammar
    has no path that produces one (`test_grammar_rejects_off_menu_actions`);
  * the `action_id`/`test_key` disagreement that `episode._do_order` raises on cannot
    occur, because only one of the two is ever generated.

Sum-to-one is not expressible in JSON Schema
--------------------------------------------
So it is repaired after decoding, not asserted during it. `complete_distribution` is the
whole of that repair and it is worth reading closely, because the obvious version of it is
wrong (see below).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import ActionKind, ActionMenu, action_id, build_menu
from dxenv.env.schemas import Abstain, Action, Diagnose, OrderTest, Prescribe

DEFAULT_MAX_LABELS: Final = 16
"""How many conditions a report may name explicitly.

Not a truncation of the belief -- see `complete_distribution`, which completes the tail
rather than renormalising it away. But the completion is max-entropy, so whatever the
posterior knew about the ordering of the unnamed labels IS lost, and how much is lost
depends on this number.

Chosen from measurement rather than taste. Unnamed posterior mass at the initial
vitals-only observation, over 300 patients -- the most diffuse the belief ever is:

    k= 4   mean 0.204   p90 0.523   max 0.690
    k= 8   mean 0.107   p90 0.331   max 0.512
    k=16   mean 0.044   p90 0.152   max 0.327
    k=32   mean 0.011   p90 0.018   max 0.150

16 roughly halves the p90 tail against 8 for about 200 extra tokens on the one diagnose
turn of an episode. 32 halves it again but the returns have clearly turned, and the token
cost lands on every rollout of every GRPO group. Emitting all 149 pairs would spend most
of a rollout's budget on mass a competent policy has already ruled out.
"""

MAX_REASONING_CHARS: Final = 700
"""Cap on the reasoning string, in characters.

Advisory, not enforced. Grammar backends generally do not implement JSON Schema's
`maxLength` for strings -- xgrammar constrains structure, not length -- so this bounds the
token budget below and is repeated as a prompt instruction, which is what actually
shortens the output.
"""


def max_completion_tokens(max_labels: int = DEFAULT_MAX_LABELS) -> int:
    """A generous upper bound on the tokens one action needs.

    Derived rather than guessed, because guessing produced a real failure: at 512 the 7B
    ran out mid-string on an `order_test` and the truncated prefix -- structurally valid
    JSON, just unfinished -- surfaced as "the grammar was not applied", which is exactly
    backwards. The grammar was applied; the budget was not enough to finish inside it.

    The worst case is `diagnose`: reasoning, plus `max_labels` entries each carrying a
    condition slug (up to ~35 chars) and a 9-decimal probability. Roughly 4 chars/token,
    ~30 tokens per entry, and doubled for headroom -- the cost of being generous is some
    latency, and the cost of being tight is a dead six-hour job.
    """
    reasoning = MAX_REASONING_CHARS // 4
    entries = max_labels * 30
    return 2 * (reasoning + entries + 64)

QUANT_PREDICTIONS: Final = ("low", "normal", "high")
CAT_PREDICTIONS: Final = ("normal_categorical", "abnormal_categorical")
PREDICTIONS: Final = QUANT_PREDICTIONS + CAT_PREDICTIONS


class DecodingError(ValueError):
    """The model's output could not be turned into a legal action. Never swallowed.

    Under constrained decoding this should be unreachable; it is raised rather than
    defaulted to `abstain` precisely so that an unreachable case that turns out to be
    reachable shows up as a crash instead of as a quietly rising abstention rate.
    """


# ------------------------------------------------------------------------- schema ----


def action_json_schema(
    menu: ActionMenu | None = None,
    taxonomy: Taxonomy | None = None,
    max_labels: int = DEFAULT_MAX_LABELS,
) -> dict[str, Any]:
    """The grammar. One `oneOf` over the four action kinds, with every key enumerated."""
    m = menu or build_menu()
    tax = taxonomy or load_taxonomy()

    reasoning = {
        "type": "string",
        "maxLength": MAX_REASONING_CHARS,
        # `maxLength` alone does not bound anything: grammar backends constrain STRUCTURE,
        # not string length, so a chatty model will fill whatever token budget it is
        # given. A 7B ran past 2876 tokens mid-sentence with the cap set to 700.
        #
        # `pattern` IS compiled into the grammar, so the bound becomes structural: the
        # decoder cannot emit a 701st character. Excluding quote and backslash keeps the
        # regex simple and costs only escapes inside the reasoning, which is prose.
        # Excludes the C0 control characters as well as quote and backslash. The first
        # version excluded only quote and backslash, which let the model emit a RAW
        # newline inside the string -- structurally fine to the grammar, and invalid JSON,
        # because JSON requires control characters to be escaped. Excluding backslash
        # means it cannot write the escape either, so the only consistent choice is to
        # forbid the characters themselves.
        "pattern": f'^[^"\\\\\\x00-\\x1f]{{0,{MAX_REASONING_CHARS}}}$',
        "description": (
            f"Why this action, given only what is on the case sheet. "
            f"One or two sentences, at most {MAX_REASONING_CHARS} characters."
        ),
    }

    def variant(kind: str, props: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"kind": {"const": kind}, "reasoning": reasoning, **props},
            "required": ["kind", "reasoning", *props],
            "additionalProperties": False,
        }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "dxenv_action",
        "oneOf": [
            variant(
                "order_test",
                {
                    "test_key": {"enum": sorted(a.key for a in m.test_actions())},
                    "prediction": {"enum": list(PREDICTIONS)},
                },
            ),
            variant(
                "prescribe",
                {"treatment_key": {"enum": sorted(a.key for a in m.treatment_actions())}},
            ),
            variant(
                "diagnose",
                {
                    "diagnosis": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max_labels,
                        "items": {
                            "type": "object",
                            "properties": {
                                "condition": {"enum": list(tax.slugs)},
                                "probability": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["condition", "probability"],
                            "additionalProperties": False,
                        },
                    }
                },
            ),
            variant("abstain", {}),
        ],
    }


def schema_fingerprint(schema: Mapping[str, Any]) -> str:
    """Hash of the grammar, pinned into a run alongside the menu fingerprint."""
    import hashlib

    blob = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------- distribution ----


def complete_distribution(
    pairs: Sequence[Mapping[str, Any]], taxonomy: Taxonomy | None = None
) -> dict[str, float]:
    """Turn the named pairs into a full distribution over the label set.

    The obvious implementation -- renormalise the named probabilities to sum to 1 -- is
    wrong, and wrong in the direction that flatters the policy. An agent that names eight
    conditions totalling 0.7 has said "0.3 of my belief is elsewhere"; renormalising
    silently converts that into "1.0 of my belief is in these eight", which is a
    confidence the agent never claimed and which a proper scoring rule will duly reward
    when it happens to be right.

    So the residual mass is spread UNIFORMLY over the labels the agent did not name --
    the maximum-entropy completion consistent with what it actually said. It is also the
    fix for the floor-depression problem in `policy/baselines.TOP_K_REPORT`: a report
    truncated to k costs an uninformed policy far more than an informed one, but a
    truncated report with a uniform tail costs it almost nothing.

    Over-claiming (named mass > 1) is the one case that IS renormalised: there is no
    residual to place, and the agent has stated relative weights that sum badly.

    Duplicated conditions are summed, not rejected. Mass is mass, and rejecting would
    turn a formatting slip into a scored failure.
    """
    tax = taxonomy or load_taxonomy()
    named: dict[str, float] = {}
    for item in pairs:
        slug = str(item["condition"])
        p = float(item["probability"])
        if not np.isfinite(p) or p < 0.0:
            raise DecodingError(f"probability for {slug!r} is {p!r}; must be finite and >= 0")
        named[slug] = named.get(slug, 0.0) + p
    unknown = sorted(set(named) - set(tax.slugs))
    if unknown:
        raise DecodingError(f"report names labels outside the frozen taxonomy: {unknown}")
    if not named:
        raise DecodingError("diagnosis names no conditions")

    total = sum(named.values())
    if total <= 0.0:
        raise DecodingError("diagnosis puts zero mass everywhere; it is not a distribution")

    unnamed = [s for s in tax.slugs if s not in named]
    if total > 1.0 or not unnamed:
        return {k: v / total for k, v in named.items()}

    residual = (1.0 - total) / len(unnamed)
    out = dict(named)
    for slug in unnamed:
        out[slug] = residual
    # Float error accumulates over 149 terms; land it on the largest entry, where it is
    # relatively smallest. The Diagnose schema rejects anything off by more than 1e-6.
    drift = 1.0 - sum(out.values())
    top = max(out, key=lambda k: out[k])
    out[top] += drift
    return out


# --------------------------------------------------------------------------- parse ----


def parse_wire(obj: Mapping[str, Any], menu: ActionMenu, taxonomy: Taxonomy) -> Action:
    """Wire object -> validated `Action`, with ids derived from the menu."""
    kind = obj.get("kind")
    if kind == ActionKind.ORDER_TEST.value:
        key = str(obj["test_key"])
        aid = menu.id_for_test(key)
        if aid not in menu.ids:
            raise DecodingError(f"test_key {key!r} is not on the global menu [I3]")
        return OrderTest(action_id=aid, test_key=key, prediction=obj["prediction"])
    if kind == ActionKind.PRESCRIBE.value:
        key = str(obj["treatment_key"])
        aid = menu.id_for_treatment(key)
        if aid not in menu.ids:
            raise DecodingError(f"treatment_key {key!r} is not on the global menu [I3]")
        return Prescribe(action_id=aid, treatment_key=key)
    if kind == ActionKind.DIAGNOSE.value:
        dist = complete_distribution(obj["diagnosis"], taxonomy)
        return Diagnose(action_id=action_id(ActionKind.DIAGNOSE, "diagnose"), distribution=dist)
    if kind == ActionKind.ABSTAIN.value:
        return Abstain(action_id=action_id(ActionKind.ABSTAIN, "abstain"))
    raise DecodingError(f"unknown action kind {kind!r}")


def parse_action(
    text: str,
    menu: ActionMenu | None = None,
    taxonomy: Taxonomy | None = None,
) -> Action:
    """Decoded text -> `Action`. Raises on anything the environment would reject."""
    m = menu or build_menu()
    tax = taxonomy or load_taxonomy()
    try:
        # strict=False accepts literal control characters inside strings. Belt to the
        # pattern's braces: grammar backends vary in how much of a regex they honour, and
        # a raw newline in a reasoning string is unambiguous -- accepting it parses the
        # same content and changes nothing about which action was chosen. That is a
        # different thing from a fallback ACTION, which would substitute a decision the
        # policy never made, and is still refused below.
        obj = json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        raise DecodingError(
            f"model output is not JSON ({exc}). Under constrained decoding this is "
            f"unreachable, so reaching it means the grammar was not applied: {text[:200]!r}"
        ) from exc
    if not isinstance(obj, dict):
        raise DecodingError(f"expected a JSON object, got {type(obj).__name__}")
    try:
        return parse_wire(obj, m, tax)
    except (KeyError, TypeError) as exc:
        raise DecodingError(f"wire object is missing a required field: {exc}") from exc


def reasoning_of(text: str) -> str:
    """The `reasoning` string, for the de-leaking filters. Empty if unparseable."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return ""
    return str(obj.get("reasoning", "")) if isinstance(obj, dict) else ""


# ------------------------------------------------------------------------ sampling ----


def sample_wire_action(
    rng: np.random.Generator,
    menu: ActionMenu | None = None,
    taxonomy: Taxonomy | None = None,
    max_labels: int = DEFAULT_MAX_LABELS,
    kind: str | None = None,
) -> dict[str, Any]:
    """A uniformly random schema-valid wire object.

    This is what a perfectly-constrained, perfectly-uninformed decoder emits. It backs
    `test_constrained_decoding_always_valid` without a GPU, and it is the fake backend
    the episode-loop tests run against -- the grammar and the parser are exercised on
    every commit even though the model that will use them is not.
    """
    m = menu or build_menu()
    tax = taxonomy or load_taxonomy()
    k = kind or str(rng.choice(["order_test", "prescribe", "diagnose", "abstain"]))
    base: dict[str, Any] = {"kind": k, "reasoning": f"sampled action of kind {k}"}
    if k == "order_test":
        keys = sorted(a.key for a in m.test_actions())
        base["test_key"] = str(rng.choice(keys))
        base["prediction"] = str(rng.choice(list(PREDICTIONS)))
    elif k == "prescribe":
        keys = sorted(a.key for a in m.treatment_actions())
        base["treatment_key"] = str(rng.choice(keys))
    elif k == "diagnose":
        n = int(rng.integers(1, max_labels + 1))
        slugs = rng.choice(np.array(tax.slugs), size=n, replace=False)
        w = rng.dirichlet(np.ones(n))
        base["diagnosis"] = [
            {"condition": str(s), "probability": float(p)} for s, p in zip(slugs, w, strict=True)
        ]
    return base


WIRE_KEY_ORDER: Final = ("kind", "reasoning", "test_key", "prediction", "treatment_key",
                        "diagnosis")
"""Canonical key order: the order `action_json_schema` DECLARES the properties in.

Not alphabetical. This serialisation is what SFT trains the model to emit, and a
grammar-constrained decoder emits properties in declaration order -- so if the two
disagree, every token after the first divergence is off the distribution the model was
trained on.

That is not a theoretical concern. `sort_keys=True` here taught the model
`{"kind":...,"prediction":...,"reasoning":...}` while the decoder forces
`{"kind":...,"reasoning":...}`, and the SFT'd 7B responded by rambling, drifting into
Chinese mid-string, and emitting fragments like `basePath/000022/bedside/urinalysis`. The
adapter was fine; it was being decoded into a shape it had never seen.
"""


def render_wire(obj: Mapping[str, Any]) -> str:
    """Serialise a wire object exactly as the constrained decoder emits one.

    Key order follows the schema, and the separators are Python's defaults -- which is
    what the model is observed to produce, and what its own pre-training prior favours.
    Fighting that prior buys a few tokens and costs distribution match.
    """
    ordered = {k: obj[k] for k in WIRE_KEY_ORDER if k in obj}
    extra = {k: v for k, v in obj.items() if k not in ordered}
    if extra:
        raise DecodingError(f"wire object has keys outside the schema: {sorted(extra)}")
    return json.dumps(ordered)


# --------------------------------------------------------------------------- vLLM ----


def guided_decoding_params(schema: Mapping[str, Any] | None = None) -> Any:
    """vLLM `GuidedDecodingParams` for this grammar. Imported lazily; GPU-only.

    Kept behind a function so that importing `dxenv.policy` on a laptop -- which is where
    the invariant suite runs -- does not require vLLM to be installed.
    """
    try:
        from vllm import sampling_params as sp
    except ImportError as exc:  # pragma: no cover - exercised only on a GPU host
        raise DecodingError(
            "vLLM is not installed. Install the infer extra (`pip install -e "
            "'.[infer]'`) on a CUDA host; the grammar itself is pure data and needs "
            "nothing."
        ) from exc
    # Renamed in vLLM: GuidedDecodingParams -> StructuredOutputsParams. Detected rather
    # than pinned; see VLLMBackend._structured_output_kwargs for the reasoning.
    for name in ("StructuredOutputsParams", "GuidedDecodingParams"):
        if hasattr(sp, name):
            return getattr(sp, name)(json=dict(schema or action_json_schema()))
    raise DecodingError(
        "this vLLM exposes neither StructuredOutputsParams nor GuidedDecodingParams"
    )


DEFAULT_MAX_TOKENS: Final = max_completion_tokens()
"""Token budget one action needs, from `max_completion_tokens`. 512 was not enough."""
