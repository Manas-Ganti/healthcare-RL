"""Treatment appropriateness and contraindications.

Scored twice (CLAUDE.md 7.5):

  coherence   -- appropriateness conditional on the DECLARED diagnosis. Rewards an agent
                 whose actions follow from its own stated belief, even when that belief
                 turns out to be wrong.
  correctness -- appropriateness against the TRUE condition, GATED on the declared
                 probability mass the agent put on that condition. This is what makes a
                 lucky-correct treatment under a wrong diagnosis collect nothing: the
                 gate is near zero exactly when the agent did not believe it.

Contraindication penalties are asymmetric and large, and every one of them is checkable
by the agent from information already in its observation -- the allergy list, eGFR,
potassium, platelets, beta-hCG. A penalty the agent could not have avoided would just be
noise in the reward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.obs_model import ResultValue

_CONFIG_DIR: Final = Path(__file__).resolve().parents[1] / "configs"


class TreatmentError(ValueError):
    """Malformed treatment config. Never caught inside `dxenv.reward`."""


@dataclass(frozen=True, slots=True)
class TreatmentConfig:
    first_line: dict[str, tuple[str, ...]]
    acceptable: dict[str, tuple[str, ...]]
    acceptable_fraction: float
    allergy_scale: float
    renal: tuple[float, tuple[str, ...], float]
    pregnancy: tuple[float, tuple[str, ...], float]
    hyperkalemia: tuple[float, tuple[str, ...], float]
    bleeding: tuple[float, tuple[str, ...], float]
    interactions: tuple[tuple[frozenset[str], float], ...]

    def appropriateness(self, treatment: str, condition: str) -> float:
        """1.0 first-line, `acceptable_fraction` if defensible, else 0."""
        if treatment in self.first_line.get(condition, ()):
            return 1.0
        if treatment in self.acceptable.get(condition, ()):
            return self.acceptable_fraction
        return 0.0


@lru_cache(maxsize=4)
def load_treatment_config(path: Path | None = None) -> TreatmentConfig:
    with (path or _CONFIG_DIR / "treatments.yaml").open() as fh:
        raw = yaml.safe_load(fh)
    cat = load_catalog()
    known = set(cat.treatment_keys)

    def _check(keys: list[str], where: str) -> tuple[str, ...]:
        bad = sorted(set(keys) - known)
        if bad:
            raise TreatmentError(f"{where}: unknown treatments {bad}")
        return tuple(keys)

    first_line = {c: _check(v, f"first_line[{c}]") for c, v in raw["first_line"].items()}
    acceptable = {c: _check(v, f"acceptable[{c}]") for c, v in raw["acceptable"].items()}
    ci = raw["contraindications"]

    def _rule(name: str, threshold_key: str) -> tuple[float, tuple[str, ...], float]:
        block = ci[name]
        return (
            float(block[threshold_key]),
            _check(block["treatments"], f"contraindications[{name}]"),
            float(block["scale"]),
        )

    interactions = tuple(
        (frozenset(_check(r["pair"], "interactions")), float(r["scale"]))
        for r in ci["interactions"]
    )
    return TreatmentConfig(
        first_line=first_line,
        acceptable=acceptable,
        acceptable_fraction=float(raw["acceptable_fraction"]),
        allergy_scale=float(ci["allergy"]["scale"]),
        renal=_rule("renal", "egfr_below"),
        pregnancy=_rule("pregnancy", "hcg_above"),
        hyperkalemia=_rule("hyperkalemia", "potassium_above"),
        bleeding=_rule("bleeding", "platelets_below"),
        interactions=interactions,
    )


def contraindication_violations(
    prescribed: list[str],
    analytes: Mapping[str, ResultValue],
    allergies: tuple[str, ...],
    cfg: TreatmentConfig | None = None,
    catalog: Catalog | None = None,
) -> list[tuple[str, str, float]]:
    """(treatment, rule, scale) for every violation. Deterministic and order-independent."""
    c = cfg or load_treatment_config()
    cat = catalog or load_catalog()
    allergy_classes = set(allergies)
    out: list[tuple[str, str, float]] = []

    def _num(key: str, default: float) -> float:
        v = analytes.get(key, default)
        return float(v) if isinstance(v, (int, float)) else default

    egfr, hcg = _num("egfr", 100.0), _num("hcg_quant", 0.0)
    potassium, platelets = _num("potassium", 4.0), _num("platelets", 250.0)

    thresholds = (
        ("renal", egfr < c.renal[0], c.renal[1], c.renal[2]),
        ("pregnancy", hcg > c.pregnancy[0], c.pregnancy[1], c.pregnancy[2]),
        ("hyperkalemia", potassium > c.hyperkalemia[0], c.hyperkalemia[1], c.hyperkalemia[2]),
        ("bleeding", platelets < c.bleeding[0], c.bleeding[1], c.bleeding[2]),
    )
    for t in prescribed:
        if cat.treatment(t).drug_class in allergy_classes:
            out.append((t, "allergy", c.allergy_scale))
        for name, active, treatments, scale in thresholds:
            if active and t in treatments:
                out.append((t, name, scale))
    given = set(prescribed)
    for pair, scale in c.interactions:
        if pair <= given:
            out.append(("+".join(sorted(pair)), "interaction", scale))
    return out


def treatment_score(
    prescribed: list[str],
    declared: Mapping[str, float],
    true_condition: str,
    analytes: Mapping[str, ResultValue],
    allergies: tuple[str, ...],
    coherence_scale: float,
    correctness_scale: float,
    contraindication_penalty: float,
    cfg: TreatmentConfig | None = None,
    catalog: Catalog | None = None,
) -> tuple[float, list[tuple[str, str, float]]]:
    """Returns (score, violations). Prescribing nothing scores exactly 0, never negative."""
    c = cfg or load_treatment_config()
    if not prescribed:
        return 0.0, []

    declared_dx = max(declared, key=lambda k: declared[k]) if declared else None
    mass_on_truth = float(declared.get(true_condition, 0.0))

    coherence = 0.0
    if declared_dx is not None:
        coherence = sum(c.appropriateness(t, declared_dx) for t in prescribed) / len(prescribed)

    # Gated on the agent's OWN stated confidence in the truth. A shotgun prescription
    # that happens to include the right drug under a confident wrong diagnosis collects
    # approximately nothing here.
    correctness = (
        mass_on_truth
        * sum(c.appropriateness(t, true_condition) for t in prescribed)
        / len(prescribed)
    )

    violations = contraindication_violations(prescribed, analytes, allergies, c, catalog)
    penalty = contraindication_penalty * sum(scale for _, _, scale in violations)

    return coherence_scale * coherence + correctness_scale * correctness - penalty, violations
