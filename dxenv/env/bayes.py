"""Exact posterior over the flat label set, and the Bayes ceiling [I9].

    2-condition, 2-test worked example (the golden test asserts this exact arithmetic)
    ------------------------------------------------------------------------------
    Conditions A, B with prior (0.5, 0.5). One binary analyte with

        p(+|A) = 0.9   p(+|B) = 0.2

    Observing '+':
        unnormalised  = (0.5*0.9, 0.5*0.2) = (0.45, 0.10)
        posterior     = (0.45/0.55, 0.10/0.55) = (0.8181818..., 0.1818181...)

    Observing '+' then a second, independent analyte with p(+|A)=0.3, p(+|B)=0.6 and
    seeing '+' again:
        unnormalised  = (0.45*0.3, 0.10*0.6) = (0.135, 0.060)
        posterior     = (0.6923076..., 0.3076923...)
    Order-invariant: multiplying the same two likelihood vectors in the other order gives
    the same normalised result.

Why the scoring rule is injected
--------------------------------
`env/` must not import `reward/` (CLAUDE.md 3), but the ceiling is defined in terms of
the terminal scoring rule. So the scoring rule arrives as a callable. The caller passes
`dxenv.reward.scoring.brier_score`; there is exactly one implementation of the rule and
this module does not get its own copy to drift from.

On the two ceilings
-------------------
`expected_ceiling` is the Bayes-optimal *expected* score given maximal information. It is
an upper bound on expected reward, and it is the number worth reporting -- but a single
lucky rollout CAN exceed it, because a proper scoring rule only guarantees that truthful
reporting maximises the score *in expectation*. Asserting it per episode would fire on
luck, and a detector that cries wolf gets switched off, which is worse than not having
one.

`hard_ceiling` is the per-episode bound: the score of a report that puts all mass on the
true condition. Nothing an agent does can beat it on any realisation, so exceeding it is
a genuine impossibility -- a leak, a scoring bug, or a NaN.

Use both. `hard_ceiling` for the per-episode halt; `expected_ceiling` for the running-mean
check that actually detects reward hacking. See train/monitors.py.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import numpy as np
import numpy.typing as npt

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.obs_model import ObservationModel, ResultValue, build_observation_model

Evidence = Mapping[str, ResultValue]
"""analyte key -> observed result. The posterior depends on this SET, never on order."""

ScoreFn = Callable[[npt.NDArray[np.float64], int], float]
"""(reported distribution, true condition index) -> score. Higher is better."""

_LOG_ZERO_FLOOR: Final = -700.0
"""Clamp for log-likelihoods, below which exp underflows to exactly 0 in float64.

The categorical tables are smoothed at build time (obs_model.CATEGORICAL_EPSILON) so no
single term is -inf, but a long trajectory of many analytes can still sum past the
underflow point for badly-fitting conditions. Clamping the *sum* rather than dropping the
condition keeps the posterior strictly positive and the log-sum-exp finite [I11].
"""


class BayesError(ValueError):
    """Malformed evidence or belief. Never caught inside `dxenv.env`."""


def log_prior(taxonomy: Taxonomy | None = None) -> npt.NDArray[np.float64]:
    tax = taxonomy or load_taxonomy()
    return np.log(tax.prior())


def _normalise_log(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Softmax with the standard max-shift. Returns a strictly positive, normalised vector."""
    if not np.isfinite(logits).all():
        raise BayesError("non-finite log-posterior; a likelihood term was inf or NaN [I11]")
    shifted = logits - logits.max()
    shifted = np.maximum(shifted, _LOG_ZERO_FLOOR)
    w = np.exp(shifted)
    total = w.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise BayesError("posterior failed to normalise")
    return np.asarray(w / total, dtype=np.float64)


def posterior(
    evidence: Evidence,
    model: ObservationModel | None = None,
    prior_log: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Exact Bayesian update over the flat label set.

    Conditional independence of analytes given the condition is the modelling assumption
    the environment is built on -- it is what makes the posterior exact rather than
    approximate, and it is honest here because the observation model generates each
    analyte independently given the condition. If a correlated analyte pair is ever
    added, this becomes an approximation and the ceiling stops being sound.
    """
    m = model or build_observation_model()
    logits = np.array(prior_log if prior_log is not None else log_prior(), dtype=np.float64)
    if logits.shape != (m.n_conditions,):
        raise BayesError(f"prior has shape {logits.shape}, expected {(m.n_conditions,)}")
    for analyte, value in evidence.items():
        logits = logits + m.log_likelihood_vector(analyte, value)
    return _normalise_log(logits)


def entropy(p: npt.NDArray[np.float64]) -> float:
    """Shannon entropy in nats. The shaping potential is its negation (reward/shaping.py)."""
    check_belief(p)
    nz = p[p > 0.0]
    return float(-(nz * np.log(nz)).sum())


def check_belief(p: npt.NDArray[np.float64]) -> None:
    if p.ndim != 1:
        raise BayesError("belief must be a 1-D vector")
    if not np.isfinite(p).all():
        raise BayesError("belief contains NaN or inf [I11]")
    if (p < 0.0).any():
        raise BayesError("belief has negative mass")
    if abs(float(p.sum()) - 1.0) > 1e-8:
        raise BayesError(f"belief sums to {p.sum()}, not 1")


def expected_score(
    belief: npt.NDArray[np.float64],
    report: npt.NDArray[np.float64],
    score_fn: ScoreFn,
) -> float:
    """E_{c ~ belief}[ score_fn(report, c) ]."""
    check_belief(belief)
    check_belief(report)
    return float(sum(belief[i] * score_fn(report, i) for i in range(belief.size)))


def bayes_optimal_value(belief: npt.NDArray[np.float64], score_fn: ScoreFn) -> float:
    """Best expected score obtainable while holding `belief`.

    By properness of the scoring rule this is attained by reporting `belief` itself, so
    no maximisation is needed. `test_brier_is_proper` is what licenses that shortcut --
    if the rule ever stops being proper, this function silently becomes wrong, which is
    why that test is the most important one in the repo.
    """
    return expected_score(belief, belief, score_fn)


def expected_ceiling(
    full_evidence: Evidence,
    score_fn: ScoreFn,
    model: ObservationModel | None = None,
    prior_log: npt.NDArray[np.float64] | None = None,
) -> float:
    """Upper bound on the expected terminal score, given every analyte revealed for free.

    Bound direction, explicitly (I9 is unsound if this is not an *upper* bound):
      1. More evidence never lowers the attainable expected score under a proper scoring
         rule (Blackwell sufficiency: any policy on a subset of the evidence can be
         simulated by one holding all of it). So the full-information Bayes value bounds
         the diagnostic score of every policy.
      2. Every test cost is non-positive [I5] and the turn penalty is non-positive, so
         dropping them only raises the bound.
      3. Potential-based shaping telescopes to gamma^T Phi(s_T) - Phi(s_0), which the
         caller must add separately if shaping is active -- this function does NOT
         include it. See `combined_expected_ceiling`.

    It is therefore a bound, not the exact optimum: a real agent is budget-limited and
    pays for evidence. Reporting it as "the ceiling" is honest only with that caveat.
    """
    belief = posterior(full_evidence, model=model, prior_log=prior_log)
    return bayes_optimal_value(belief, score_fn)


def hard_ceiling(score_fn: ScoreFn, n_conditions: int | None = None) -> float:
    """Per-episode bound: the score of a perfectly confident, correct report.

    Sound on every realisation, including a lucky guess, which is exactly what makes it
    safe to assert per episode and halt on violation.
    """
    n = n_conditions if n_conditions is not None else len(load_taxonomy())
    best = -np.inf
    for i in range(n):
        onehot = np.zeros(n, dtype=np.float64)
        onehot[i] = 1.0
        best = max(best, score_fn(onehot, i))
    return float(best)


def posterior_sequence(
    ordered_evidence: list[tuple[str, ResultValue]],
    model: ObservationModel | None = None,
    prior_log: npt.NDArray[np.float64] | None = None,
) -> list[npt.NDArray[np.float64]]:
    """Posterior after each successive observation, starting from the prior.

    Used by the shaping potential (which needs Phi at every state) and by the rejection
    sampler's process filter, which asks whether each test moved the posterior toward the
    truth -- a question about the *path*, not just the endpoint.
    """
    m = model or build_observation_model()
    logits = np.array(prior_log if prior_log is not None else log_prior(), dtype=np.float64)
    out = [_normalise_log(logits)]
    for analyte, value in ordered_evidence:
        logits = logits + m.log_likelihood_vector(analyte, value)
        out.append(_normalise_log(logits))
    return out
