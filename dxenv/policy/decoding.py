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

DEFAULT_MAX_LABELS: Final = 8
"""How many conditions a report may name explicitly.

Not a truncation of the belief -- see `complete_distribution`. Emitting all 149 pairs on
every rollout would triple the token cost of a GRPO group for mass that is, in a
competent policy, almost entirely in the tail.
"""

MAX_REASONING_CHARS: Final = 1200

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
        "description": "Why this action, given only what is on the case sheet.",
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
        obj = json.loads(text)
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


def render_wire(obj: Mapping[str, Any]) -> str:
    """Serialise a wire object the way a decoder would emit it."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


# --------------------------------------------------------------------------- vLLM ----


def guided_decoding_params(schema: Mapping[str, Any] | None = None) -> Any:
    """vLLM `GuidedDecodingParams` for this grammar. Imported lazily; GPU-only.

    Kept behind a function so that importing `dxenv.policy` on a laptop -- which is where
    the invariant suite runs -- does not require vLLM to be installed.
    """
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError as exc:  # pragma: no cover - exercised only on a GPU host
        raise DecodingError(
            "vLLM is not installed. Install the GPU extra (`pip install -e '.[gpu]'`) on "
            "a CUDA host; the grammar itself is pure data and needs nothing."
        ) from exc
    return GuidedDecodingParams(json=dict(schema or action_json_schema()))
