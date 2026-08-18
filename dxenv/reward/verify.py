"""Predict-then-verify [I5-safe per-step scoring].

The agent commits to a coarse prediction of a test's result BEFORE the result is
revealed. The commitment lives on the action schema (`OrderTest.prediction`), so a test
order without a prediction is rejected by pydantic -- `test_commit_is_mandatory` cannot
be satisfied by a runtime check someone later forgets to call, and
`test_result_not_visible_before_commit` is structural: the action object is constructed
by the policy, and the result object does not exist until `episode._do_order` runs.

Scored against the PRIOR marginal for that bucket, so a prediction made at chance has
expected score exactly zero. Predicting "normal" on everything therefore earns nothing.

Side effect worth naming: an agent that can predict a result exactly has learned that
the test is redundant, and the cost term then pushes it to stop ordering that test. The
verify term is not there to make testing pay -- it cannot, see below.

I5 SAFETY, by construction: the verify reward for an order is a FRACTION of that same
order's own cost. `fraction < 1` therefore makes cost + verify strictly negative for
every test individually, not merely for the cheapest one. Scaling by a flat constant
instead would force the constant below the cheapest test's price, which would make the
term meaningless for the expensive tests where a committed prediction matters most.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final, Literal

import numpy as np

from dxenv.data.taxonomy import load_taxonomy
from dxenv.env.catalog import CategoricalAnalyte, Catalog, QuantitativeAnalyte, load_catalog
from dxenv.env.obs_model import ResultValue, build_observation_model

Bucket = Literal["low", "normal", "high", "abnormal_categorical", "normal_categorical"]

QUANT_BUCKETS: Final = ("low", "normal", "high")
CAT_BUCKETS: Final = ("normal_categorical", "abnormal_categorical")


class VerifyError(ValueError):
    """Malformed prediction or unknown analyte. Never caught inside `dxenv.reward`."""


def headline_analyte(test_key: str, catalog: Catalog | None = None) -> str:
    """The analyte a prediction is judged against.

    Panels return several analytes; the prediction is scored against the FIRST one
    listed in catalog.yaml, which is by convention the analyte the panel is ordered for.
    Scoring against all of them would make "low" meaningless for a mixed panel.
    """
    cat = catalog or load_catalog()
    return cat.test(test_key).analytes[0]


def actual_bucket(analyte: str, value: ResultValue, catalog: Catalog | None = None) -> Bucket:
    """Coarsen a revealed result into the bucket vocabulary."""
    cat = catalog or load_catalog()
    a = cat.analyte(analyte)
    if isinstance(a, QuantitativeAnalyte):
        if not isinstance(value, (int, float)):
            raise VerifyError(f"{analyte!r} is quantitative but got {value!r}")
        if value < a.ref_low:
            return "low"
        if value > a.ref_high:
            return "high"
        return "normal"
    if isinstance(a, CategoricalAnalyte):
        if not isinstance(value, str):
            raise VerifyError(f"{analyte!r} is categorical but got {value!r}")
        # Convention, enforced in the catalog: values[0] is the reference/normal result.
        return "normal_categorical" if value == a.values[0] else "abnormal_categorical"
    raise VerifyError(f"analyte {analyte!r} has an unhandled kind")


@lru_cache(maxsize=1)
def _bucket_priors() -> dict[tuple[str, str], float]:
    """P(bucket | analyte) marginalised over the taxonomy prior.

    Derived entirely from committed data files, so it is deterministic and carries no
    RNG, clock or run state -- reward stays a pure function of its arguments [I8].
    """
    tax = load_taxonomy()
    cat = load_catalog()
    model = build_observation_model()
    prior = tax.prior()
    out: dict[tuple[str, str], float] = {}
    for key in cat.analyte_keys:
        a = cat.analyte(key)
        if isinstance(a, CategoricalAnalyte):
            t = model.table(key)
            probs = np.asarray(t.probs)  # type: ignore[union-attr]
            p_normal = float(prior @ probs[:, 0])
            out[(key, "normal_categorical")] = p_normal
            out[(key, "abnormal_categorical")] = 1.0 - p_normal
        elif isinstance(a, QuantitativeAnalyte):
            t = model.table(key)
            mean = np.asarray(t.mean)  # type: ignore[union-attr]
            sd = np.asarray(t.sd)  # type: ignore[union-attr]
            from scipy.special import ndtr

            p_low = float(prior @ ndtr((a.ref_low - mean) / sd))
            p_below_high = float(prior @ ndtr((a.ref_high - mean) / sd))
            out[(key, "low")] = p_low
            out[(key, "normal")] = max(0.0, p_below_high - p_low)
            out[(key, "high")] = max(0.0, 1.0 - p_below_high)
    return out


def bucket_prior(analyte: str, bucket: str) -> float:
    try:
        return _bucket_priors()[(analyte, bucket)]
    except KeyError as exc:
        raise VerifyError(f"no prior for bucket {bucket!r} of analyte {analyte!r}") from exc


def verify_term(
    test_key: str,
    prediction: str,
    revealed: dict[str, ResultValue],
    order_cost: float,
    fraction: float,
    catalog: Catalog | None = None,
) -> float:
    """`fraction * order_cost * (1{correct} - P(bucket))`, bounded by the order's own cost.

    Zero in expectation for a prediction made at chance, which is what makes it
    unfarmable by a constant policy: predicting "normal" on everything earns nothing.
    """
    if not 0.0 <= fraction < 1.0:
        raise VerifyError(
            f"verify fraction must be in [0, 1); got {fraction}. At 1.0 a correct "
            "prediction would exactly cancel the test's cost and make testing free [I5]."
        )
    if order_cost < 0.0:
        raise VerifyError("order cost must be non-negative")
    scale = fraction * order_cost
    cat = catalog or load_catalog()
    analyte = headline_analyte(test_key, cat)
    if analyte not in revealed:
        # A refused or duplicate order reveals nothing new; there is nothing to verify.
        return 0.0
    a = cat.analyte(analyte)
    valid = CAT_BUCKETS if isinstance(a, CategoricalAnalyte) else QUANT_BUCKETS
    if prediction not in valid:
        # A category error (predicting "low" for an imaging finding) scores the worst
        # available outcome rather than raising: it is a legal schema value, just a
        # nonsensical one for this test, and the agent should learn that.
        return -scale * bucket_prior(analyte, valid[0])
    correct = actual_bucket(analyte, revealed[analyte], cat) == prediction
    return float(scale * ((1.0 if correct else 0.0) - bucket_prior(analyte, prediction)))


def max_verify_gain(fraction: float, order_cost: float) -> float:
    """Largest verify reward one order can produce: strictly below that order's cost."""
    return float(fraction * order_cost)
