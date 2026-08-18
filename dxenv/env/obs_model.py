"""Generative observation model: p(result | condition), for EVERY pair [I4].

The whole cross product of (analyte x condition) is materialised when the model is
built. There is no runtime fallback and no "unavailable" result: every orderable test
returns a value for every patient, so the *sparsity pattern* of a record carries no
information about the label. That is the point of I4, and `test_no_side_channel` is the
direct check that it worked.

Inheritance of the healthy baseline happens exactly once, at build time, in
`build_observation_model`. After that, a lookup miss raises. "Return normal if the pair
isn't found" is the bug this module is shaped to prevent.

Layout is vectorised per analyte -- means/sds or a probability matrix indexed by
condition -- because the Bayes posterior updates over all 149 conditions at once and a
per-condition dict lookup there would dominate the DP in env/bayes.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import yaml
from scipy.special import ndtr, ndtri

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.catalog import (
    CategoricalAnalyte,
    QuantitativeAnalyte,
    load_catalog,
)

_OVERRIDES_PATH: Final = Path(__file__).with_name("obs_overrides.yaml")

CATEGORICAL_EPSILON: Final = 1e-4
"""Uniform smoothing mixed into every categorical likelihood at build time.

Two reasons, both load-bearing:
  * A likelihood of exactly zero makes log p = -inf, and one such term anywhere in a
    trajectory propagates NaN through the posterior and into the reward [I11].
  * Without it, a single categorical result can render a condition *literally*
    impossible. That is not robust modelling -- it hands the agent a one-test oracle for
    any finding unique to one condition, and inflates the Bayes ceiling to match.
Smoothing is applied once, at build time, so the stored likelihoods are the smoothed
ones and nothing downstream has to remember to do it.
"""

ResultValue = float | str


class ObsModelError(ValueError):
    """Unknown pair, or a malformed override. Never caught inside `dxenv.env`."""


@dataclass(frozen=True, slots=True)
class QuantTable:
    """Truncated-normal parameters for one analyte, indexed by condition."""

    analyte: str
    mean: np.ndarray  # (n_conditions,)
    sd: np.ndarray  # (n_conditions,)
    low: float
    high: float

    def sample(self, condition_idx: int, rng: np.random.Generator) -> float:
        """Inverse-CDF sample from the truncated normal. Never returns None [I4]."""
        mu = float(self.mean[condition_idx])
        sigma = float(self.sd[condition_idx])
        alpha = ndtr((self.low - mu) / sigma)
        beta = ndtr((self.high - mu) / sigma)
        u = rng.random()
        x = mu + sigma * float(ndtri(alpha + u * (beta - alpha)))
        # Guard against ndtri saturating at the tails; bounds are a hard contract.
        return float(np.clip(x, self.low, self.high))

    def log_likelihood(self, value: float) -> np.ndarray:
        """log p(value | c) for every condition c, as a vector."""
        z = (value - self.mean) / self.sd
        log_phi = -0.5 * z * z - 0.5 * np.log(2.0 * np.pi) - np.log(self.sd)
        mass = ndtr((self.high - self.mean) / self.sd) - ndtr((self.low - self.mean) / self.sd)
        return np.asarray(log_phi - np.log(mass), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CatTable:
    """Categorical probabilities for one analyte: (n_conditions, n_values), rows sum to 1."""

    analyte: str
    values: tuple[str, ...]
    probs: np.ndarray

    def sample(self, condition_idx: int, rng: np.random.Generator) -> str:
        row = self.probs[condition_idx]
        return self.values[int(rng.choice(len(self.values), p=row))]

    def log_likelihood(self, value: str) -> np.ndarray:
        try:
            j = self.values.index(value)
        except ValueError as exc:
            raise ObsModelError(
                f"analyte {self.analyte!r} has no result value {value!r}"
            ) from exc
        return np.log(self.probs[:, j])


AnalyteTable = QuantTable | CatTable


@dataclass(frozen=True, slots=True)
class ObservationModel:
    """The materialised model. Every (analyte, condition) pair is present, by construction."""

    conditions: tuple[str, ...]
    tables: dict[str, AnalyteTable]
    condition_index_map: dict[str, int]

    def table(self, analyte: str) -> AnalyteTable:
        try:
            return self.tables[analyte]
        except KeyError as exc:
            raise ObsModelError(
                f"no distribution for analyte {analyte!r}. This is a build error, not a "
                "runtime condition -- do not add a fallback here (I4)."
            ) from exc

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)

    @property
    def pair_count(self) -> int:
        return len(self.tables) * len(self.conditions)

    def sample(self, analyte: str, condition: str, rng: np.random.Generator) -> ResultValue:
        idx = self.condition_index(condition)
        return self.table(analyte).sample(idx, rng)

    def log_likelihood_vector(self, analyte: str, value: ResultValue) -> np.ndarray:
        t = self.table(analyte)
        if isinstance(t, QuantTable):
            if not isinstance(value, (int, float)):
                raise ObsModelError(f"analyte {analyte!r} is quantitative; got {value!r}")
            return t.log_likelihood(float(value))
        if not isinstance(value, str):
            raise ObsModelError(f"analyte {analyte!r} is categorical; got {value!r}")
        return t.log_likelihood(value)

    def condition_index(self, condition: str) -> int:
        try:
            return self.condition_index_map[condition]
        except KeyError as exc:
            raise ObsModelError(f"unknown condition {condition!r}") from exc


def _smooth(probs: np.ndarray, eps: float = CATEGORICAL_EPSILON) -> np.ndarray:
    k = probs.shape[-1]
    return (1.0 - eps) * probs + eps / k


def _load_overrides(path: Path | None = None) -> dict[str, dict[str, dict[str, float]]]:
    with (path or _OVERRIDES_PATH).open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ObsModelError("obs_overrides.yaml must be a mapping of condition -> analyte -> spec")
    return raw


@lru_cache(maxsize=1)
def build_observation_model(
    overrides_path: Path | None = None, taxonomy: Taxonomy | None = None
) -> ObservationModel:
    """Materialise p(result | condition) for the full cross product.

    Raises on any override that names an unknown condition or analyte, puts mass on an
    undeclared categorical value, or fails to normalise. Failing closed here is what
    makes the totality guarantee (I4) real rather than aspirational.
    """
    tax = taxonomy or load_taxonomy()
    cat = load_catalog()
    overrides = _load_overrides(overrides_path)

    conditions = tax.slugs
    n = len(conditions)

    unknown_conds = sorted(set(overrides) - set(conditions))
    if unknown_conds:
        raise ObsModelError(f"overrides for conditions not in the taxonomy: {unknown_conds}")

    tables: dict[str, AnalyteTable] = {}
    for akey in cat.all_analyte_keys:
        analyte = cat.analyte(akey)
        if isinstance(analyte, QuantitativeAnalyte):
            mean = np.full(n, analyte.healthy_mean, dtype=np.float64)
            sd = np.full(n, analyte.healthy_sd, dtype=np.float64)
            for cond, block in overrides.items():
                spec = block.get(akey)
                if spec is None:
                    continue
                i = tax.index(cond)
                if set(spec) != {"mean", "sd"}:
                    raise ObsModelError(
                        f"{cond}.{akey}: quantitative override needs exactly mean and sd"
                    )
                m, s = float(spec["mean"]), float(spec["sd"])
                lo, hi = analyte.bounds
                if not lo <= m <= hi:
                    raise ObsModelError(f"{cond}.{akey}: mean {m} outside bounds {analyte.bounds}")
                if s <= 0.0:
                    raise ObsModelError(f"{cond}.{akey}: sd must be > 0")
                mean[i], sd[i] = m, s
            tables[akey] = QuantTable(
                analyte=akey, mean=mean, sd=sd, low=analyte.bounds[0], high=analyte.bounds[1]
            )
        elif isinstance(analyte, CategoricalAnalyte):
            probs = np.tile(np.asarray(analyte.healthy, dtype=np.float64), (n, 1))
            for cond, block in overrides.items():
                spec = block.get(akey)
                if spec is None:
                    continue
                i = tax.index(cond)
                bad = sorted(set(spec) - set(analyte.values))
                if bad:
                    raise ObsModelError(f"{cond}.{akey}: mass on undeclared values {bad}")
                row = np.array([float(spec.get(v, 0.0)) for v in analyte.values])
                total = row.sum()
                if abs(total - 1.0) > 1e-9:
                    raise ObsModelError(f"{cond}.{akey}: sums to {total}, not 1")
                if (row < 0.0).any():
                    raise ObsModelError(f"{cond}.{akey}: negative probability")
                probs[i] = row
            tables[akey] = CatTable(
                analyte=akey, values=analyte.values, probs=_smooth(probs)
            )
        else:  # pragma: no cover - the union is closed
            raise ObsModelError(f"analyte {akey!r} has an unhandled kind")

    model = ObservationModel(
        conditions=conditions,
        tables=tables,
        condition_index_map={c: i for i, c in enumerate(conditions)},
    )
    expected = len(cat.all_analyte_keys) * n
    if model.pair_count != expected:
        raise ObsModelError(f"materialised {model.pair_count} pairs, expected {expected}")
    return model
