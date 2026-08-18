"""Test costs and the turn penalty. Nothing here is ever positive [I5].

I5 is the invariant most likely to be broken by a well-meaning future edit -- someone
adds a "reward informative tests" term and the agent promptly finds tests that maximise
entropy reduction under the belief model without improving the answer. A test pays for
itself ONLY by improving the terminal score enough to cover its cost.

The price list is `dxenv/configs/costs.yaml`, shared with env/episode.py so the ledger
and the reward cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml

_CONFIG_DIR: Final = Path(__file__).resolve().parents[1] / "configs"


class CostError(ValueError):
    """Missing price or malformed table. Never caught inside `dxenv.reward`."""


@dataclass(frozen=True, slots=True)
class CostTable:
    prices: dict[str, float]

    def price(self, test_key: str) -> float:
        try:
            return self.prices[test_key]
        except KeyError as exc:
            raise CostError(
                f"no cost for test {test_key!r}. There is deliberately no default: a "
                "free test is an infinite-value test and the agent will find it."
            ) from exc

    @property
    def cheapest(self) -> float:
        return min(self.prices.values())


@lru_cache(maxsize=4)
def load_cost_table(path: Path | None = None) -> CostTable:
    with (path or _CONFIG_DIR / "costs.yaml").open() as fh:
        raw = yaml.safe_load(fh)
    if raw.get("default") is not None:
        raise CostError("costs.yaml declares a default; remove it (see module docstring)")
    prices = {str(k): float(v) for k, v in raw["tests"].items()}
    if not prices:
        raise CostError("costs.yaml lists no tests")
    bad = sorted(k for k, v in prices.items() if v <= 0.0)
    if bad:
        raise CostError(f"non-positive test prices would make a test free or profitable: {bad}")
    return CostTable(prices=prices)


def test_cost_term(test_key: str, lam: float, table: CostTable | None = None) -> float:
    """The cost contribution of ONE charged order. Always <= 0 [I5]."""
    t = table or load_cost_table()
    if lam < 0.0:
        raise CostError("lambda must be non-negative; a negative lambda pays for testing")
    return -lam * t.price(test_key)


def turn_penalty_term(n_turns: int, mu: float) -> float:
    """Always <= 0. Prices deliberation, not investigation -- keep mu small."""
    if mu < 0.0:
        raise CostError("mu must be non-negative")
    if n_turns < 0:
        raise CostError("n_turns must be non-negative")
    return -mu * float(n_turns)
