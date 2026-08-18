"""The observation builder: an allowlist that fails closed [I1, I2].

Two separate jobs, deliberately not merged:

  `build_observation`  takes a `PatientView` -- a projection that structurally lacks the
                       condition -- and emits an `Observation`, a model with no field
                       able to hold a label. I1 holds because ground truth is out of
                       scope here, not because this code is careful.

  `filter_resources`   applies the (resource_type, field) allowlist to raw Synthea FHIR.
                       Unknown resource types RAISE in strict mode. This is the path that
                       needs the allowlist, because raw bundles are where the label hides.

Why each blocked type is blocked (CLAUDE.md 6.2):

  Condition                    the label, written out
  MedicationRequest            metformin IS a diabetes diagnosis; the one people forget
  Encounter.reasonCode         the record's own explanation of the visit
  CarePlan                     named after the condition
  Procedure                    dialysis implies renal failure
  DiagnosticReport.conclusion  the answer, already written
  CareTeam                     "oncology team" narrows things considerably

Strings leak even when fields do not
------------------------------------
A lab value is fine; a lab display reading "HbA1c - diabetes monitoring" is not. So
display strings in the observation come from the CATALOG, never from the record, and any
text that does survive from the record is scrubbed.

The global-vocabulary rule
--------------------------
Result values such as `st_elevation` or `influenza_positive` are drawn from a vocabulary
declared in catalog.yaml that is IDENTICAL for every patient. They are exempt from the
label-string scrub, for the same reason the action menu is safe under I3: a vocabulary
that does not vary with the patient cannot encode which patient it is. The information
such a value carries reaches the agent through the likelihood -- the intended channel --
and mangling the names would cost fidelity while closing nothing.

Scrubbing therefore targets RECORD-DERIVED text, where the string was written by an
author who knew the answer. `assert_no_label_leak` enforces exactly that distinction, and
`test_result_vocabulary_is_global` is what licenses the exemption.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Final

from dxenv.data.corpus import BLOCKED_RESOURCE_TYPES, PatientView
from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.catalog import Catalog, CategoricalAnalyte, QuantitativeAnalyte, load_catalog
from dxenv.env.obs_model import ResultValue
from dxenv.env.schemas import AnalyteResult, Demographics, Observation

REDACTED: Final = "[redacted]"

ALLOWED_RESOURCE_FIELDS: Final[dict[str, frozenset[str]]] = {
    "Patient": frozenset({"id", "gender", "birthDate"}),
    "Observation": frozenset({"resourceType", "category", "code", "valueQuantity", "effectiveDateTime"}),
    "FamilyMemberHistory": frozenset({"resourceType", "relationship", "condition"}),
    # Encounter is permitted, but ONLY its non-explanatory fields. reasonCode and
    # reasonReference are the record telling you why the visit happened, which is the
    # diagnosis; they are excluded here rather than by blocking Encounter wholesale,
    # because the visit class and date are legitimate context.
    "Encounter": frozenset({"resourceType", "class", "period", "type"}),
}
"""(resource_type -> permitted fields). Everything else is dropped; an unrecognised
resource type raises in strict mode. Note `Observation.note` is absent -- free text
inside a permitted resource is still free text."""


class FilterError(ValueError):
    """Allowlist violation. Never caught inside `dxenv.env`."""


class LabelLeakError(AssertionError):
    """A label string reached an observation. Always a bug, never a tolerable condition."""


@lru_cache(maxsize=1)
def global_result_vocabulary(catalog: Catalog | None = None) -> frozenset[str]:
    """Every categorical result value the environment can ever emit.

    Global and patient-independent, which is precisely what makes these strings safe to
    exempt from the label scrub.
    """
    cat = catalog or load_catalog()
    out: set[str] = set()
    for key in cat.all_analyte_keys:
        a = cat.analyte(key)
        if isinstance(a, CategoricalAnalyte):
            out.update(a.values)
    return frozenset(out)


def scrub_text(text: str, taxonomy: Taxonomy | None = None) -> str:
    """Redact any condition name or synonym, for EVERY condition in the taxonomy.

    Scrubbing against the whole label set rather than against this patient's label is not
    an oversight -- it is the only version that is safe. A scrubber that knew which label
    to remove would need the label, and the pattern of what it removed would itself be a
    channel.
    """
    tax = taxonomy or load_taxonomy()
    out = text
    forms = sorted(
        {s for lab in tax.labels for s in lab.leak_strings if len(s) >= 4},
        key=len,
        reverse=True,
    )
    for form in forms:
        out = re.sub(rf"(?<![A-Za-z]){re.escape(form)}(?![A-Za-z])", REDACTED, out, flags=re.I)
    return out


def filter_resources(
    resources: tuple[dict[str, Any], ...] | list[dict[str, Any]], strict: bool = True
) -> tuple[dict[str, Any], ...]:
    """Apply the allowlist to raw FHIR. Fails closed [I2].

    In strict mode an unrecognised resource type raises rather than passing through --
    the whole point of an allowlist is that new, unreviewed resource types are refused by
    default. A future Synthea release adding a resource we have never seen must break the
    build, not silently enter observations.
    """
    kept: list[dict[str, Any]] = []
    for res in resources:
        rtype = res.get("resourceType")
        if not isinstance(rtype, str):
            raise FilterError(f"resource without a resourceType: {sorted(res)[:5]}")
        if rtype in BLOCKED_RESOURCE_TYPES:
            continue
        allowed = ALLOWED_RESOURCE_FIELDS.get(rtype)
        if allowed is None:
            if strict:
                raise FilterError(
                    f"unrecognised resource type {rtype!r}. Add it to "
                    "ALLOWED_RESOURCE_FIELDS or BLOCKED_RESOURCE_TYPES after deciding "
                    "which it is; the allowlist does not guess."
                )
            continue
        kept.append({k: v for k, v in res.items() if k in allowed})
    return tuple(kept)


def build_result(key: str, value: ResultValue, catalog: Catalog) -> AnalyteResult:
    """Build one result. `display` comes from the CATALOG, never from the record."""
    a = catalog.analyte(key)
    if isinstance(a, QuantitativeAnalyte):
        if not isinstance(value, (int, float)):
            raise FilterError(f"analyte {key!r} is quantitative but got {value!r}")
        return AnalyteResult(
            analyte=key,
            display=a.display,
            unit=a.unit,
            value_number=round(float(value), 4),
            ref_low=a.ref_low,
            ref_high=a.ref_high,
        )
    if not isinstance(value, str):
        raise FilterError(f"analyte {key!r} is categorical but got {value!r}")
    return AnalyteResult(analyte=key, display=a.display, value_code=value)


def build_observation(
    view: PatientView,
    revealed: dict[str, ResultValue],
    turn: int,
    remaining_budget: float,
    turns_remaining: int,
    menu_fingerprint: str,
    catalog: Catalog | None = None,
    taxonomy: Taxonomy | None = None,
) -> Observation:
    """The agent's view of the world at one turn.

    `view` cannot carry the condition and `Observation` cannot hold it, so there is no
    path by which the label enters. Vitals and the presenting complaint are free; every
    other analyte appears only if it is in `revealed`.
    """
    cat = catalog or load_catalog()
    tax = taxonomy or load_taxonomy()

    complaint = view.analytes.get("presenting_complaint")
    if complaint is None:
        raise FilterError(
            "patient has no presenting_complaint. Every patient has one by construction "
            "(I4); a missing value here means the corpus was built with a stale catalog."
        )
    if not isinstance(complaint, str):
        raise FilterError("presenting_complaint must be categorical")

    vitals = tuple(
        build_result(k, view.analytes[k], cat)
        for k in cat.vital_keys
        if k != "presenting_complaint"
    )

    unknown = sorted(set(revealed) - set(cat.analyte_keys))
    if unknown:
        raise FilterError(f"revealed analytes not on the orderable catalog: {unknown}")

    results = tuple(build_result(k, revealed[k], cat) for k in sorted(revealed))
    family = tuple(scrub_text(f, tax) for f in view.family_history)

    return Observation(
        patient_ref=view.patient_id,
        turn=turn,
        demographics=Demographics(age_years=view.age_years, sex=view.sex),  # type: ignore[arg-type]
        presenting_complaint=complaint,
        vitals=vitals,
        family_history=family,
        revealed_results=results,
        remaining_budget=round(float(remaining_budget), 4),
        turns_remaining=turns_remaining,
        menu_fingerprint=menu_fingerprint,
    )


def observation_strings(obs: Observation) -> list[str]:
    """Every string an agent could read off an observation. Used by the leak audit."""
    out = [obs.patient_ref, obs.presenting_complaint, *obs.family_history]
    for r in (*obs.vitals, *obs.revealed_results):
        out.extend([r.analyte, r.display, r.unit])
        if r.value_code is not None:
            out.append(r.value_code)
    return [s for s in out if s]


def assert_no_label_leak(
    obs: Observation, condition: str, taxonomy: Taxonomy | None = None
) -> None:
    """Raise if any RECORD-DERIVED string in `obs` names this patient's condition.

    Values from the global result vocabulary are exempt; see the module docstring. The
    exemption is only sound while that vocabulary really is patient-independent, which
    `test_result_vocabulary_is_global` checks separately -- if that test is ever deleted,
    this function silently stops guarding anything.
    """
    tax = taxonomy or load_taxonomy()
    vocab = global_result_vocabulary()
    forms = [s for s in tax.get(condition).leak_strings if len(s) >= 4]
    for text in observation_strings(obs):
        if text in vocab:
            continue
        haystack = f" {text.replace('_', ' ').lower()} "
        for form in forms:
            if f" {form} " in haystack:
                raise LabelLeakError(
                    f"observation for {obs.patient_ref} contains {form!r} (condition "
                    f"{condition!r}) in the string {text!r}"
                )
