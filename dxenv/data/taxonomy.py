"""Fixed flat condition label set and the mappings into it.

The label set is *flat* by construction (CLAUDE.md 6.1): no label is an ancestor of
another, so an agent cannot hedge upward into a superclass and collect partial credit
everywhere. `system` groupings exist for reporting only and are never visible to
`dxenv.reward`.

The set is frozen: `LABEL_SET_HASH` is committed and `Taxonomy.hash()` must match it.
Adding or renaming a label is a deliberate, breaking change that invalidates every stored
trajectory, every ceiling, and the eval split hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import yaml

_LABELS_PATH: Final = Path(__file__).with_name("labels.yaml")
_SNOMED_PATH: Final = Path(__file__).with_name("snomed_map.yaml")

# Frozen hash of the sorted label slugs. See tests/unit/test_taxonomy.py.
# Regenerate deliberately with: python -m dxenv.data.taxonomy --print-hash
LABEL_SET_HASH: Final = "2f762a7a0f71e5db9356550730e3974feceefe4ccd3b88829b7296b3fbc7b8f3"

URGENCY_TIERS: Final = (1, 2, 3, 4)


class TaxonomyError(ValueError):
    """Raised on any unmapped code, unknown slug, or malformed label file.

    Never caught inside `dxenv.data`; a silent default here is how a leak gets in.
    """


@dataclass(frozen=True, slots=True)
class Label:
    """One diagnostic label. Immutable and hashable."""

    slug: str
    display: str
    system: str
    urgency: int
    prior_weight: float
    synonyms: tuple[str, ...]

    @property
    def canonical_forms(self) -> tuple[str, ...]:
        """The forms that define this label's *identity*, for the flatness check.

        Deliberately narrow: display name and slug only. `leak_strings` is deliberately
        wide. Checking flatness against the wide list produces false positives ("stroke"
        is a legitimate scrub term for ischemic stroke and also a substring of heat
        stroke) and would pressure someone into trimming the scrub list, which is the
        one list that must never be trimmed.
        """
        return (self.display.lower(), self.slug.replace("_", " ").lower())

    @property
    def leak_strings(self) -> tuple[str, ...]:
        """Every surface form that would give this label away in an observation.

        Used by the observation scrubber and by `test_no_label_string_in_observation`.
        Under-listing a synonym is how a leak survives the suite, so this deliberately
        includes the slug's word forms as well as the curated synonyms.
        """
        forms = {self.display, self.slug, self.slug.replace("_", " "), *self.synonyms}
        return tuple(sorted(f.lower() for f in forms if f))


class Taxonomy:
    """The frozen flat label set.

    Construct via `load_taxonomy()`, which caches. Every accessor raises on a miss --
    there is no "unknown" bucket, because an unknown bucket silently absorbs the
    conditions the environment is worst at.
    """

    def __init__(self, labels: list[Label]) -> None:
        if not labels:
            raise TaxonomyError("empty label set")
        slugs = [lab.slug for lab in labels]
        dupes = {s for s in slugs if slugs.count(s) > 1}
        if dupes:
            raise TaxonomyError(f"duplicate label slugs: {sorted(dupes)}")
        # Canonical order is sorted-by-slug. Everything downstream -- posterior vectors,
        # Brier targets, stored trajectories -- indexes into this order, so it must not
        # depend on file order.
        self._labels: Final = tuple(sorted(labels, key=lambda lab: lab.slug))
        self._index: Final = {lab.slug: i for i, lab in enumerate(self._labels)}
        bad = [lab.slug for lab in self._labels if lab.urgency not in URGENCY_TIERS]
        if bad:
            raise TaxonomyError(f"urgency outside {URGENCY_TIERS}: {bad}")
        nonpos = [lab.slug for lab in self._labels if lab.prior_weight <= 0.0]
        if nonpos:
            raise TaxonomyError(f"prior weight must be > 0 (a zero-prior label is unreachable "
                                f"but still scoreable): {nonpos}")

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Any:
        return iter(self._labels)

    @property
    def labels(self) -> tuple[Label, ...]:
        return self._labels

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(lab.slug for lab in self._labels)

    def index(self, slug: str) -> int:
        """Position of `slug` in the canonical ordering. Raises on unknown slug."""
        try:
            return self._index[slug]
        except KeyError as exc:
            raise TaxonomyError(f"unknown condition slug: {slug!r}") from exc

    def get(self, slug: str) -> Label:
        return self._labels[self.index(slug)]

    def prior(self) -> npt.NDArray[np.float64]:
        """Normalised prevalence prior over the canonical label ordering."""
        w = np.array([lab.prior_weight for lab in self._labels], dtype=np.float64)
        return np.asarray(w / w.sum(), dtype=np.float64)

    def urgency_vector(self) -> npt.NDArray[np.float64]:
        return np.array([lab.urgency for lab in self._labels], dtype=np.int64)

    def hash(self) -> str:
        """Content hash of the frozen label set.

        Covers slug, urgency and prior -- anything whose change would silently alter
        scoring. Display strings and synonyms are excluded so the leak-scrubber list can
        be tightened without invalidating stored trajectories.
        """
        payload = [[lab.slug, lab.urgency, round(lab.prior_weight, 6)] for lab in self._labels]
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def assert_frozen(self) -> None:
        actual = self.hash()
        if actual != LABEL_SET_HASH:
            raise TaxonomyError(
                f"label set hash drifted: {actual} != committed {LABEL_SET_HASH}. "
                "Changing the label set invalidates stored trajectories, ceilings and the "
                "eval split. If intentional, update LABEL_SET_HASH in the same commit."
            )

    def all_leak_strings(self) -> dict[str, tuple[str, ...]]:
        return {lab.slug: lab.leak_strings for lab in self._labels}


def _parse_labels(raw: object) -> list[Label]:
    if not isinstance(raw, list):
        raise TaxonomyError("labels.yaml must contain a list of label mappings")
    out: list[Label] = []
    required = {"slug", "display", "system", "urgency", "prior"}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TaxonomyError(f"label entry {i} is not a mapping")
        missing = required - item.keys()
        if missing:
            raise TaxonomyError(f"label entry {i} missing fields: {sorted(missing)}")
        unknown = item.keys() - (required | {"synonyms"})
        if unknown:
            raise TaxonomyError(f"label entry {i} has unknown fields: {sorted(unknown)}")
        out.append(
            Label(
                slug=str(item["slug"]),
                display=str(item["display"]),
                system=str(item["system"]),
                urgency=int(item["urgency"]),
                prior_weight=float(item["prior"]),
                synonyms=tuple(str(s) for s in item.get("synonyms", ())),
            )
        )
    return out


@lru_cache(maxsize=1)
def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Load and cache the frozen taxonomy."""
    p = path or _LABELS_PATH
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    return Taxonomy(_parse_labels(raw))


@lru_cache(maxsize=1)
def load_snomed_map(path: Path | None = None) -> dict[str, str]:
    """SNOMED CT concept id -> label slug.

    PROVISIONAL. The shipped map covers the codes emitted by the Synthea modules this
    repo has been exercised against; it is deliberately incomplete and is kept in a
    separate file from `labels.yaml` so that terminology churn cannot alter the frozen
    label-set hash. `map_snomed` raises on any code not present -- see CLAUDE.md 6.1
    `test_every_corpus_condition_mapped`. Do NOT add a fallback bucket.
    """
    p = path or _SNOMED_PATH
    if not p.exists():
        return {}
    with p.open() as fh:
        raw = yaml.safe_load(fh) or {}
    codes = raw.get("codes", {})
    if not isinstance(codes, dict):
        raise TaxonomyError("snomed_map.yaml: 'codes' must be a mapping")
    tax = load_taxonomy()
    mapping = {str(code): str(slug) for code, slug in codes.items()}
    unknown = sorted({slug for slug in mapping.values() if slug not in tax.slugs})
    if unknown:
        raise TaxonomyError(f"snomed_map.yaml points at slugs not in the taxonomy: {unknown}")
    return mapping


def map_snomed(code: str) -> str:
    """Map a SNOMED concept id to a label slug. Raises if unmapped -- by design."""
    mapping = load_snomed_map()
    try:
        return mapping[str(code)]
    except KeyError as exc:
        raise TaxonomyError(
            f"unmapped SNOMED code {code!r}. Add it to snomed_map.yaml or exclude the "
            "condition from the corpus; silently dropping it biases the prior."
        ) from exc


def check_flat(tax: Taxonomy) -> list[tuple[str, str]]:
    """Return label pairs that look like an ancestor/descendant relation.

    Structural proxy only: true SNOMED ancestry needs a terminology release, which this
    repo does not vendor. It catches the realistic authoring mistake -- adding a broad
    label ("diabetes mellitus") alongside a specific one ("type 2 diabetes mellitus") --
    by flagging any canonical form that is a whole-phrase subphrase of another's.
    Checked against `canonical_forms`, never `leak_strings`. A true ancestry check
    belongs in a nightly job with a real SNOMED release.
    """
    forms: list[tuple[str, set[str]]] = [(lab.slug, set(lab.canonical_forms)) for lab in tax.labels]
    offenders: list[tuple[str, str]] = []
    for slug_a, forms_a in forms:
        for slug_b, forms_b in forms:
            if slug_a >= slug_b:
                continue
            for fa in forms_a:
                for fb in forms_b:
                    if fa == fb or len(fa) < 4 or len(fb) < 4:
                        continue
                    short, long_, sa, sb = (
                        (fa, fb, slug_a, slug_b) if len(fa) < len(fb) else (fb, fa, slug_b, slug_a)
                    )
                    if f" {short} " in f" {long_} ":
                        offenders.append((sa, sb))
    return sorted(set(offenders))


if __name__ == "__main__":  # pragma: no cover - maintenance helper
    t = load_taxonomy()
    print(f"labels: {len(t)}")
    print(f"hash:   {t.hash()}")
