"""Potential-based shaping [I6]: F(s,a,s') = gamma * Phi(s') - Phi(s).

Ng, Harada & Russell (1999): for ANY potential Phi, this form leaves the optimal policy
set unchanged. That guarantee is the entire reason shaping is permitted in an
environment whose whole purpose is to not be gameable -- and it holds only for this
form. No other per-step bonus exists, and none may be added.

Phi = -H(posterior), in nats: the potential rises as the agent becomes more certain.

Note what this does NOT do: it does not reward "informative" tests. The sum over any
trajectory telescopes to gamma^T Phi(s_T) - Phi(s_0), so a path that wanders through
low-entropy states and returns collects nothing. `test_closed_loop_shaping_is_zero` is
the property that makes it unfarmable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dxenv.env.bayes import entropy


class ShapingError(ValueError):
    """Malformed belief sequence. Never caught inside `dxenv.reward`."""


def potential(belief: npt.NDArray[np.float64]) -> float:
    """Phi(s) = -H(belief). Bounded in [-log n, 0]."""
    return -entropy(belief)


def shaping_terms(
    beliefs: list[npt.NDArray[np.float64]],
    gamma: float = 1.0,
    scale: float = 1.0,
) -> list[float]:
    """One term per transition. `beliefs` is the posterior at each state, s_0 first."""
    if len(beliefs) < 2:
        return []
    if not 0.0 < gamma <= 1.0:
        raise ShapingError(f"gamma must be in (0, 1], got {gamma}")
    if scale < 0.0:
        raise ShapingError("scale must be non-negative")
    phis = [potential(b) for b in beliefs]
    return [scale * (gamma * phis[i + 1] - phis[i]) for i in range(len(phis) - 1)]


def total_shaping(
    beliefs: list[npt.NDArray[np.float64]],
    gamma: float = 1.0,
    scale: float = 1.0,
) -> float:
    """PLAIN sum of the shaping terms -- what the reward engine actually adds up.

    Equals the telescoped closed form only when gamma == 1 (which is the shipped config).
    For gamma < 1 the telescoping identity is about the DISCOUNTED sum; use
    `discounted_total` for that comparison. Conflating the two is an easy mistake: the
    plain sum of gamma*Phi(s') - Phi(s) does NOT telescope when gamma < 1.
    """
    return float(sum(shaping_terms(beliefs, gamma, scale)))


def discounted_total(
    beliefs: list[npt.NDArray[np.float64]], gamma: float = 1.0, scale: float = 1.0
) -> float:
    """sum_i gamma^i * F_i -- the quantity the telescoping identity is about."""
    terms = shaping_terms(beliefs, gamma, scale)
    return float(sum(gamma**i * f for i, f in enumerate(terms)))


def telescoped_total(
    beliefs: list[npt.NDArray[np.float64]], gamma: float = 1.0, scale: float = 1.0
) -> float:
    """The closed form, computed from the endpoints only.

    Exists so `test_shaping_telescopes` can compare two independent computations rather
    than asserting a function equals itself.
    """
    if len(beliefs) < 2:
        return 0.0
    t = len(beliefs) - 1
    return float(scale * (gamma**t * potential(beliefs[-1]) - potential(beliefs[0])))


def max_shaping_gain(n_conditions: int, scale: float) -> float:
    """Largest shaping reward any single transition can produce.

    Used by the reward-config validator to prove no test step can be net-positive [I5].
    The extreme case is going from a uniform posterior (H = log n) to certainty (H = 0).
    """
    return float(scale * np.log(n_conditions))
