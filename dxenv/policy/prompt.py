"""Observation -> prompt text. The only place model input is constructed.

Everything here reads a typed `Observation` and the two global objects (the menu, the
taxonomy). It cannot reach a `PatientRecord`, so the prompt inherits I1 from the
observation rather than re-earning it: there is no argument through which the label could
arrive. `test_prompt_contains_no_label_string` asserts that over the full corpus anyway,
because "structurally impossible" claims are exactly the ones worth checking.

The menu is rendered from `build_menu()`, which takes no patient [I3]. Rendering only the
*affordable* tests would be a per-patient menu wearing a disguise -- the agent would read
"this test is not offered" as evidence -- so the full menu is always printed and the
budget is stated separately as a number the agent has to reason about.
"""

from __future__ import annotations

from typing import Final

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import ActionKind, ActionMenu, build_menu
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.schemas import AnalyteResult, Observation

SYSTEM_PROMPT: Final = """\
You are a diagnostician working through a case one turn at a time.

Each turn you take exactly one action:
  order_test  -- order a test or panel. It costs budget. You must commit to a coarse
                 prediction of the headline result BEFORE seeing it.
  prescribe   -- prescribe a treatment.
  diagnose    -- terminate by reporting a probability distribution over conditions.
  abstain     -- terminate without committing to a diagnosis.

You are scored once, at the end, by a proper scoring rule over your reported
distribution. Reporting your honest belief is the score-maximising thing to do:
overstating confidence is punished, and so is hedging toward a flat distribution. Tests
subtract from your score, always, so order one only when the answer it gives is worth
more than it costs. Running out of turns without deciding is scored like abstaining.

Reply with a single JSON object and nothing else.\
"""


def _fmt_result(r: AnalyteResult) -> str:
    if r.value_number is not None:
        ref = (
            f" (ref {r.ref_low:g}-{r.ref_high:g})"
            if r.ref_low is not None and r.ref_high is not None
            else ""
        )
        unit = f" {r.unit}" if r.unit else ""
        return f"  {r.display}: {r.value_number:.4g}{unit}{ref}"
    return f"  {r.display}: {r.value_code}"


def render_menu(menu: ActionMenu | None = None, catalog: Catalog | None = None) -> str:
    """The global menu, grouped by category. Identical for every patient [I3]."""
    m = menu or build_menu()
    cat = catalog or load_catalog()
    by_category: dict[str, list[str]] = {}
    for a in m.test_actions():
        spec = cat.test(a.key)
        by_category.setdefault(spec.category, []).append(f"{a.key} ({spec.display})")
    lines = ["TESTS (test_key, then what it measures):"]
    for category in sorted(by_category):
        lines.append(f" {category}:")
        lines.extend(f"   {entry}" for entry in sorted(by_category[category]))
    lines.append("TREATMENTS (treatment_key):")
    lines.extend(
        f"   {a.key} ({cat.treatment(a.key).display})"
        for a in sorted(m.treatment_actions(), key=lambda x: x.key)
    )
    return "\n".join(lines)


def render_label_set(taxonomy: Taxonomy | None = None) -> str:
    tax = taxonomy or load_taxonomy()
    return "CONDITIONS you may name in a diagnosis:\n" + "\n".join(
        f"   {slug}" for slug in tax.slugs
    )


def render_observation(obs: Observation) -> str:
    """The per-turn case state. Reads the observation and nothing else."""
    d = obs.demographics
    parts = [
        f"CASE {obs.patient_ref}  (turn {obs.turn}, {obs.turns_remaining} turns left)",
        f"Budget remaining: {obs.remaining_budget:g}",
        f"Patient: {d.age_years}-year-old {d.sex}",
        f"Presenting complaint: {obs.presenting_complaint}",
    ]
    parts.append("Vitals:")
    parts.extend(_fmt_result(r) for r in obs.vitals)
    parts.append(
        "Family history: " + (", ".join(obs.family_history) if obs.family_history else "none coded")
    )
    parts.append("Allergies: " + (", ".join(obs.allergies) if obs.allergies else "none recorded"))
    if obs.revealed_results:
        parts.append("Results so far:")
        parts.extend(_fmt_result(r) for r in obs.revealed_results)
    else:
        parts.append("Results so far: none; you have ordered nothing.")
    return "\n".join(parts)


def build_prompt(
    obs: Observation,
    menu: ActionMenu | None = None,
    taxonomy: Taxonomy | None = None,
    catalog: Catalog | None = None,
    include_menu: bool = True,
) -> str:
    """The full user-turn prompt. `include_menu=False` is for the blank-record probe."""
    blocks = [render_observation(obs)]
    if include_menu:
        blocks.append(render_menu(menu, catalog))
        blocks.append(render_label_set(taxonomy))
    blocks.append("Your action, as a single JSON object:")
    return "\n\n".join(blocks)


def chat_messages(obs: Observation, **kwargs: object) -> list[dict[str, str]]:
    """Chat-template form. vLLM and transformers both take this shape."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(obs, **kwargs)},  # type: ignore[arg-type]
    ]


def menu_action_kinds() -> tuple[str, ...]:
    return tuple(k.value for k in ActionKind)
