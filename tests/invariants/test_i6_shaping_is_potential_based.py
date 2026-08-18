"""I6: any shaping term is potential-based, F(s,a,s') = gamma*Phi(s') - Phi(s)."""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.reward.shaping import (
    ShapingError,
    discounted_total,
    potential,
    shaping_terms,
    telescoped_total,
    total_shaping,
)


def _beliefs(n: int, k: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.dirichlet(np.ones(n) * 0.5) for _ in range(k)]


@pytest.mark.parametrize("gamma", [1.0, 0.99, 0.5])
def test_shaping_telescopes(gamma: float) -> None:
    """The DISCOUNTED sum equals gamma^T Phi(s_T) - Phi(s_0), path-independent.

    Stated for the discounted sum on purpose: the plain sum of gamma*Phi(s') - Phi(s)
    does not telescope when gamma < 1, and asserting it does would quietly pass only
    because the shipped config uses gamma = 1.
    """
    beliefs = _beliefs(20, 7)
    assert discounted_total(beliefs, gamma, 0.1) == pytest.approx(
        telescoped_total(beliefs, gamma, 0.1), abs=1e-9
    )


def test_plain_sum_matches_telescoped_at_gamma_one() -> None:
    """The shipped config uses gamma = 1, where the engine's plain sum is the closed form."""
    beliefs = _beliefs(20, 7)
    assert total_shaping(beliefs, 1.0, 0.1) == pytest.approx(
        telescoped_total(beliefs, 1.0, 0.1), abs=1e-12
    )


def test_closed_loop_shaping_is_zero() -> None:
    """Return to a previously visited state and net shaping is zero.

    This is the property that makes shaping unfarmable: no cycle pays.
    """
    b = _beliefs(20, 4, seed=3)
    loop = [b[0], b[1], b[2], b[3], b[2], b[1], b[0]]
    assert total_shaping(loop, gamma=1.0, scale=0.1) == pytest.approx(0.0, abs=1e-12)


def test_shaping_depends_only_on_endpoints() -> None:
    """Two different paths between the same endpoints collect identical shaping."""
    start, end = _beliefs(20, 2, seed=5)
    mid_a, mid_b = _beliefs(20, 2, seed=6)
    a = total_shaping([start, mid_a, end], 1.0, 0.1)
    b = total_shaping([start, mid_b, mid_a, end], 1.0, 0.1)
    assert a == pytest.approx(b, abs=1e-12)


def test_potential_is_negative_entropy_and_bounded() -> None:
    n = 20
    uniform = np.full(n, 1.0 / n)
    sharp = np.zeros(n)
    sharp[0] = 1.0
    assert potential(uniform) == pytest.approx(-np.log(n))
    assert potential(sharp) == pytest.approx(0.0, abs=1e-12)
    assert potential(uniform) < potential(sharp)


def test_shaping_rejects_bad_gamma() -> None:
    with pytest.raises(ShapingError):
        shaping_terms(_beliefs(5, 3), gamma=1.5, scale=0.1)
    with pytest.raises(ShapingError):
        shaping_terms(_beliefs(5, 3), gamma=0.0, scale=0.1)


def test_no_shaping_for_a_single_state() -> None:
    assert shaping_terms(_beliefs(5, 1)) == []


@pytest.mark.slow
def test_shaping_preserves_optimal_policy() -> None:
    """On a toy MDP small enough to solve exactly, argmax policy is unchanged.

    Ng et al. (1999) guarantees this for any Phi. Slow, but it is the theorem the whole
    mechanism rests on, so it is asserted rather than cited.
    """
    rng = np.random.default_rng(11)
    n_states, n_actions = 6, 3
    trans = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    reward = rng.normal(size=(n_states, n_actions))
    phi = rng.normal(size=n_states)
    gamma = 0.9

    def solve(r: np.ndarray) -> np.ndarray:
        v = np.zeros(n_states)
        for _ in range(3000):
            q = r + gamma * trans @ v
            v = q.max(axis=1)
        return (r + gamma * trans @ v).argmax(axis=1)

    shaped = reward + gamma * (trans @ phi) - phi[:, None]
    assert (solve(reward) == solve(shaped)).all()
