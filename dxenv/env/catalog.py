"""Loader for the global catalog of analytes, orderable tests and treatments.

Pure data access. Knows nothing about any patient -- that is what makes the global
action menu (I3) structurally guaranteed rather than merely tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

import yaml

_CATALOG_PATH: Final = Path(__file__).with_name("catalog.yaml")

AnalyteKind = Literal["quantitative", "categorical"]


class CatalogError(ValueError):
    """Malformed catalog, or a lookup miss. Never caught inside `dxenv.env`."""


@dataclass(frozen=True, slots=True)
class QuantitativeAnalyte:
    key: str
    display: str
    unit: str
    ref_low: float
    ref_high: float
    bounds: tuple[float, float]
    healthy_mean: float
    healthy_sd: float

    kind: AnalyteKind = "quantitative"


@dataclass(frozen=True, slots=True)
class CategoricalAnalyte:
    key: str
    display: str
    values: tuple[str, ...]
    healthy: tuple[float, ...]  # aligned with `values`

    kind: AnalyteKind = "categorical"

    @property
    def unit(self) -> str:
        return ""


Analyte = QuantitativeAnalyte | CategoricalAnalyte


@dataclass(frozen=True, slots=True)
class TestSpec:
    """An orderable investigation. `analytes` is what the result contains."""

    key: str
    display: str
    category: str
    analytes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TreatmentSpec:
    key: str
    display: str
    drug_class: str


@dataclass(frozen=True, slots=True)
class Catalog:
    analytes: dict[str, Analyte]
    """Orderable analytes: revealed only by paying for a test."""

    vitals: dict[str, Analyte]
    """Auto-revealed analytes: present in the t=0 observation, never on the menu."""

    tests: dict[str, TestSpec]
    treatments: dict[str, TreatmentSpec]

    def analyte(self, key: str) -> Analyte:
        """Look up an analyte, orderable or vital. Raises on a miss."""
        found = self.analytes.get(key) or self.vitals.get(key)
        if found is None:
            raise CatalogError(f"unknown analyte {key!r}")
        return found

    def test(self, key: str) -> TestSpec:
        try:
            return self.tests[key]
        except KeyError as exc:
            raise CatalogError(f"unknown test {key!r}") from exc

    def treatment(self, key: str) -> TreatmentSpec:
        try:
            return self.treatments[key]
        except KeyError as exc:
            raise CatalogError(f"unknown treatment {key!r}") from exc

    @property
    def analyte_keys(self) -> tuple[str, ...]:
        """Orderable analytes only."""
        return tuple(sorted(self.analytes))

    @property
    def vital_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.vitals))

    @property
    def all_analyte_keys(self) -> tuple[str, ...]:
        """Every random variable the observation model must cover (I4)."""
        return tuple(sorted(set(self.analytes) | set(self.vitals)))

    @property
    def test_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.tests))

    @property
    def treatment_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.treatments))


def _parse_analyte(item: dict[str, Any]) -> Analyte:
    kind = item.get("kind")
    key = str(item.get("key", "<missing>"))
    if kind == "quantitative":
        lo, hi = (float(x) for x in item["bounds"])
        if not lo < hi:
            raise CatalogError(f"analyte {key}: bounds must be increasing")
        healthy = item["healthy"]
        if float(healthy["sd"]) <= 0.0:
            raise CatalogError(f"analyte {key}: healthy sd must be > 0")
        return QuantitativeAnalyte(
            key=key,
            display=str(item["display"]),
            unit=str(item["unit"]),
            ref_low=float(item["ref_low"]),
            ref_high=float(item["ref_high"]),
            bounds=(lo, hi),
            healthy_mean=float(healthy["mean"]),
            healthy_sd=float(healthy["sd"]),
        )
    if kind == "categorical":
        values = tuple(str(v) for v in item["values"])
        if len(set(values)) != len(values):
            raise CatalogError(f"analyte {key}: duplicate result values")
        healthy_map = {str(k): float(v) for k, v in item["healthy"].items()}
        unknown = sorted(healthy_map.keys() - set(values))
        if unknown:
            raise CatalogError(f"analyte {key}: healthy mass on undeclared values {unknown}")
        # Missing values get exactly zero -- explicit, not a default. A categorical result
        # an agent can never observe under health is a legitimate modelling choice.
        probs = tuple(healthy_map.get(v, 0.0) for v in values)
        total = sum(probs)
        if abs(total - 1.0) > 1e-9:
            raise CatalogError(f"analyte {key}: healthy distribution sums to {total}, not 1")
        return CategoricalAnalyte(
            key=key, display=str(item["display"]), values=values, healthy=probs
        )
    raise CatalogError(f"analyte {key}: unknown kind {kind!r}")


@lru_cache(maxsize=1)
def load_catalog(path: Path | None = None) -> Catalog:
    with (path or _CATALOG_PATH).open() as fh:
        raw = yaml.safe_load(fh)
    for section in ("analytes", "vitals", "tests", "treatments"):
        if section not in raw:
            raise CatalogError(f"catalog missing section {section!r}")

    analytes: dict[str, Analyte] = {}
    for item in raw["analytes"]:
        a = _parse_analyte(item)
        if a.key in analytes:
            raise CatalogError(f"duplicate analyte key {a.key!r}")
        analytes[a.key] = a

    vitals: dict[str, Analyte] = {}
    for item in raw["vitals"]:
        v = _parse_analyte(item)
        if v.key in vitals or v.key in analytes:
            raise CatalogError(
                f"vital {v.key!r} collides with another analyte key. A variable that is both "
                "free and orderable would let an agent pay for what it already has."
            )
        vitals[v.key] = v

    tests: dict[str, TestSpec] = {}
    for item in raw["tests"]:
        keys = tuple(str(a) for a in item["analytes"])
        if not keys:
            raise CatalogError(f"test {item['key']!r} returns no analytes; I4 forbids an "
                               "orderable action that can return nothing")
        missing = [k for k in keys if k not in analytes]
        if missing:
            raise CatalogError(f"test {item['key']!r} references unknown analytes {missing}")
        spec = TestSpec(
            key=str(item["key"]),
            display=str(item["display"]),
            category=str(item["category"]),
            analytes=keys,
        )
        if spec.key in tests:
            raise CatalogError(f"duplicate test key {spec.key!r}")
        tests[spec.key] = spec

    treatments: dict[str, TreatmentSpec] = {}
    for item in raw["treatments"]:
        spec_t = TreatmentSpec(
            key=str(item["key"]), display=str(item["display"]), drug_class=str(item["drug_class"])
        )
        if spec_t.key in treatments:
            raise CatalogError(f"duplicate treatment key {spec_t.key!r}")
        treatments[spec_t.key] = spec_t

    orphans = sorted(set(analytes) - {a for t in tests.values() for a in t.analytes})
    if orphans:
        raise CatalogError(
            f"analytes reachable by no orderable test: {orphans}. An unreachable analyte "
            "silently inflates the Bayes ceiling above anything an agent can attain."
        )
    return Catalog(analytes=analytes, vitals=vitals, tests=tests, treatments=treatments)
