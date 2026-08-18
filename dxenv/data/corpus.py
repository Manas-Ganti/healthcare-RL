"""Patient records: Synthea parsing, and a synthetic generator for when it is absent.

Two entry points:

  `parse_synthea_bundle`  -- real Synthea FHIR output. Maps SNOMED to the flat taxonomy
                             and RAISES on any unmapped code (a silent drop biases the
                             prior, which Gate A would not catch).
  `generate_corpus`       -- self-contained generator driven by env/obs_model.py. Exists
                             so the environment, the invariant suite and the Bayes ceiling
                             are all runnable without a Java toolchain, and so the toy
                             cases in the golden tests have a fixed source.

The generated records deliberately INCLUDE the leaky resources -- Condition,
MedicationRequest, CarePlan, Encounter.reasonCode, Procedure,
DiagnosticReport.conclusion, CareTeam -- carrying the real condition name. Without them
the filter tests would be checking that a filter removes things that were never there,
which is the most comfortable kind of green test and the least informative.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from dxenv.data.taxonomy import Taxonomy, TaxonomyError, load_taxonomy, map_snomed
from dxenv.env.catalog import Catalog, CategoricalAnalyte, load_catalog
from dxenv.env.obs_model import ObservationModel, ResultValue, build_observation_model

BLOCKED_RESOURCE_TYPES: Final = frozenset(
    {
        "Condition",
        "MedicationRequest",
        "MedicationStatement",
        "CarePlan",
        "Procedure",
        "DiagnosticReport",
        "CareTeam",
        "Goal",
        "ImagingStudy",
        "Claim",
        "ExplanationOfBenefit",
    }
)
"""Resource types that are the label in disguise. See env/filter.py for the reasoning
per type; this set is the single source of truth and both modules read it."""

FAMILY_HISTORY_POOL: Final = (
    "fh_cardiac_first_degree",
    "fh_metabolic_first_degree",
    "fh_oncologic_first_degree",
    "fh_neurovascular_first_degree",
    "fh_autoimmune_first_degree",
)
"""Coded at the ORGAN-SYSTEM level, not the condition level.

Free-text family history ("type 2 diabetes in a first-degree relative") names a
condition, and for the patient who has that condition it is a label string sitting in
the observation. Coding one level up keeps the weak prior signal that family history
genuinely carries without ever printing a label."""

ALLERGY_POOL: Final = (
    "penicillin",
    "sulfonamide",
    "macrolide",
    "nsaid",
    "opioid",
    "cephalosporin",
)
"""Drug classes a patient may be allergic to.

Sampled INDEPENDENTLY of the condition -- deliberately, and the independence is tested.
Allergies are visible to the agent (unlike Condition or MedicationRequest) because they
do not name the diagnosis and because prescribing safely is impossible without them;
the contraindication penalties in reward/treatment.py are only meaningful if the agent
could have avoided the harm."""


class CorpusError(ValueError):
    """Malformed record. Never caught inside `dxenv.data`."""


@dataclass(frozen=True, slots=True)
class PatientView:
    """Everything about a patient that an observation may legally be built from.

    Note what is NOT here: the condition. `filter.build_observation` accepts only this
    type, so the observation builder has no access to ground truth at all -- I1 holds
    because the label is out of scope, not because the builder is careful.
    """

    patient_id: str
    age_years: int
    sex: str
    family_history: tuple[str, ...]
    allergies: tuple[str, ...]
    analytes: dict[str, ResultValue]
    """Every analyte, already sampled. Revealing a subset is the episode's job.

    All 105 are present for every patient, which is what makes I4 structural: there is no
    such thing as a test that "has no result" for this patient."""


@dataclass(frozen=True, slots=True)
class PatientRecord:
    """A patient plus the hidden label and the raw, leaky source resources."""

    patient_id: str
    age_years: int
    sex: str
    condition: str
    """GROUND TRUTH. Never passed to anything in env/filter.py."""

    analytes: dict[str, ResultValue]
    family_history: tuple[str, ...] = ()
    allergies: tuple[str, ...] = ()
    resources: tuple[dict[str, Any], ...] = field(default=())
    synthea_module: str = "synthetic"

    def view(self) -> PatientView:
        """The de-labelled projection. The only thing the filter is allowed to see."""
        return PatientView(
            patient_id=self.patient_id,
            age_years=self.age_years,
            sex=self.sex,
            family_history=self.family_history,
            allergies=self.allergies,
            analytes=dict(self.analytes),
        )


def _sample_demographics(rng: np.random.Generator) -> tuple[int, str]:
    age = int(np.clip(rng.normal(52.0, 21.0), 0, 100))
    sex = str(rng.choice(["female", "male"]))
    return age, sex


def _leaky_resources(
    condition_display: str, rng: np.random.Generator
) -> tuple[dict[str, Any], ...]:
    """Synthea-shaped resources that give the answer away, in the seven usual places."""
    return (
        {
            "resourceType": "Condition",
            "code": {"text": condition_display},
            "clinicalStatus": "active",
        },
        {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": f"therapy for {condition_display}"},
            "reasonReference": {"display": condition_display},
        },
        {
            "resourceType": "CarePlan",
            "title": f"{condition_display} care plan",
        },
        {
            "resourceType": "Encounter",
            "class": "ambulatory",
            "reasonCode": [{"text": condition_display}],
            "period": {"start": "2026-01-01"},
        },
        {
            "resourceType": "Procedure",
            "code": {"text": f"procedure for {condition_display}"},
        },
        {
            "resourceType": "DiagnosticReport",
            "conclusion": f"Findings consistent with {condition_display}.",
        },
        {
            "resourceType": "CareTeam",
            "name": f"{condition_display} management team",
        },
        {
            "resourceType": "Observation",
            "category": "vital-signs",
            "code": {"text": "Heart rate"},
            "note": f"reviewed in the context of {condition_display}",
            "valueQuantity": {"value": float(rng.normal(78.0, 8.0)), "unit": "bpm"},
        },
    )


def generate_patient(
    patient_id: str,
    rng: np.random.Generator,
    taxonomy: Taxonomy | None = None,
    model: ObservationModel | None = None,
    catalog: Catalog | None = None,
    condition: str | None = None,
) -> PatientRecord:
    """One synthetic patient. Deterministic given the generator's state."""
    tax = taxonomy or load_taxonomy()
    m = model or build_observation_model()
    cat = catalog or load_catalog()

    if condition is None:
        condition = str(rng.choice(tax.slugs, p=tax.prior()))
    elif condition not in tax.slugs:
        raise CorpusError(f"unknown condition {condition!r}")

    age, sex = _sample_demographics(rng)
    analytes: dict[str, ResultValue] = {
        akey: m.sample(akey, condition, rng) for akey in cat.all_analyte_keys
    }
    n_fh = int(rng.integers(0, 3))
    fh = tuple(rng.choice(FAMILY_HISTORY_POOL, size=n_fh, replace=False).tolist())
    # Drawn from a condition-independent distribution: see ALLERGY_POOL.
    n_allergy = int(rng.binomial(2, 0.18))
    allergies = tuple(rng.choice(ALLERGY_POOL, size=n_allergy, replace=False).tolist())

    return PatientRecord(
        patient_id=patient_id,
        age_years=age,
        sex=sex,
        condition=condition,
        analytes=analytes,
        family_history=fh,
        allergies=allergies,
        resources=_leaky_resources(tax.get(condition).display, rng),
        synthea_module=tax.get(condition).system,
    )


def generate_corpus(
    n: int,
    seed: int,
    taxonomy: Taxonomy | None = None,
    model: ObservationModel | None = None,
) -> list[PatientRecord]:
    """`n` patients, reproducible from `seed` alone [I10]."""
    tax = taxonomy or load_taxonomy()
    m = model or build_observation_model()
    cat = load_catalog()
    rng = np.random.default_rng(seed)
    return [
        generate_patient(f"synth-{seed}-{i:06d}", rng, tax, m, cat) for i in range(n)
    ]


# ------------------------------------------------------------- Synthea ingestion ----


def _iter_bundle_resources(bundle: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for entry in bundle.get("entry", []):
        res = entry.get("resource")
        if isinstance(res, dict):
            yield res


def parse_synthea_bundle(
    path: Path, taxonomy: Taxonomy | None = None
) -> PatientRecord:
    """Parse one Synthea FHIR bundle into a `PatientRecord`.

    Raises rather than guessing: an unmapped SNOMED code, a missing Patient resource, or
    a bundle with no mappable condition all fail loudly. `map_snomed` has no fallback
    bucket by design (data/taxonomy.py).

    The analyte panel is NOT taken from the bundle. Synthea emits labs only where its
    modules chose to, and that sparsity pattern is itself a channel to the label [I4]. We
    keep the bundle's demographics and its condition, and generate the full panel from
    the observation model instead. That is a deliberate substitution, and it is why the
    environment's difficulty is a property of obs_model.py rather than of Synthea.
    """
    tax = taxonomy or load_taxonomy()
    with path.open() as fh:
        bundle = json.load(fh)

    patient: dict[str, Any] | None = None
    resources: list[dict[str, Any]] = []
    condition_codes: list[str] = []
    for res in _iter_bundle_resources(bundle):
        resources.append(res)
        rtype = res.get("resourceType")
        if rtype == "Patient" and patient is None:
            patient = res
        elif rtype == "Condition":
            for coding in res.get("code", {}).get("coding", []):
                if "snomed" in str(coding.get("system", "")).lower():
                    condition_codes.append(str(coding.get("code")))

    if patient is None:
        raise CorpusError(f"{path}: bundle has no Patient resource")
    if not condition_codes:
        raise CorpusError(f"{path}: bundle has no SNOMED-coded Condition")

    mapped: list[str] = []
    unmapped: list[str] = []
    for code in condition_codes:
        try:
            mapped.append(map_snomed(code))
        except TaxonomyError:
            unmapped.append(code)
    if not mapped:
        raise CorpusError(
            f"{path}: no Condition mapped to the taxonomy (unmapped codes: {unmapped}). "
            "Populate dxenv/data/snomed_map.yaml before ingesting real Synthea output."
        )

    # Highest-urgency mapped condition becomes the label. Recorded here rather than left
    # implicit: with comorbid records this choice shapes the whole label distribution.
    condition = max(mapped, key=lambda slug: tax.get(slug).urgency)

    rng = np.random.default_rng(abs(hash(str(patient.get("id", path.name)))) % (2**32))
    m = build_observation_model()
    cat = load_catalog()
    analytes: dict[str, ResultValue] = {
        akey: m.sample(akey, condition, rng) for akey in cat.all_analyte_keys
    }
    birth = str(patient.get("birthDate", "1970-01-01"))[:4]
    age = max(0, min(120, 2026 - int(birth) if birth.isdigit() else 50))
    sex = str(patient.get("gender", "other"))
    if sex not in {"female", "male", "other"}:
        sex = "other"

    return PatientRecord(
        patient_id=str(patient.get("id", path.stem)),
        age_years=age,
        sex=sex,
        condition=condition,
        analytes=analytes,
        family_history=(),
        allergies=(),
        resources=tuple(resources),
        synthea_module="synthea",
    )


def categorical_analyte_keys(catalog: Catalog | None = None) -> tuple[str, ...]:
    cat = catalog or load_catalog()
    return tuple(
        k for k in cat.all_analyte_keys if isinstance(cat.analyte(k), CategoricalAnalyte)
    )
