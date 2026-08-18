"""Gym-style diagnostic episode: turn loop, budget ledger, dedup, termination.

`step` returns an observation and a done flag. It does NOT return a reward.

That is deliberate and structural. The environment produces trajectories; `dxenv.reward`
scores them, offline, as a pure function [I8]. Keeping the two apart is what makes it
free to rescore a stored corpus under new weights -- and you will change the weights
repeatedly. An env that returned rewards would quietly couple every stored rollout to the
config that generated it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from dxenv.data.corpus import PatientRecord
from dxenv.data.taxonomy import load_taxonomy
from dxenv.env.actions import ActionMenu, build_menu
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.filter import build_observation, build_result
from dxenv.env.obs_model import ResultValue
from dxenv.env.schemas import Abstain, Action, Diagnose, Observation, OrderTest, Prescribe, Step

_CONFIG_DIR: Final = Path(__file__).resolve().parents[1] / "configs"


class EpisodeError(ValueError):
    """Invalid action or malformed config. Never caught inside `dxenv.env`."""


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    max_turns: int
    dedup_repeat_orders: bool
    expose_remaining_budget: bool
    budget_support: tuple[float, ...]
    budget_weights: tuple[float, ...]
    costs: dict[str, float]

    def cost_of(self, test_key: str) -> float:
        """Price of an order. Raises on a miss -- a free test is an infinite-value test."""
        try:
            return self.costs[test_key]
        except KeyError as exc:
            raise EpisodeError(
                f"no cost for test {test_key!r}. Add it to configs/costs.yaml; there is "
                "deliberately no default."
            ) from exc

    def hash(self) -> str:
        payload = {
            "max_turns": self.max_turns,
            "dedup": self.dedup_repeat_orders,
            "budget_support": list(self.budget_support),
            "budget_weights": list(self.budget_weights),
            "costs": dict(sorted(self.costs.items())),
        }
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


def load_episode_config(
    env_path: Path | None = None, costs_path: Path | None = None
) -> EpisodeConfig:
    with (env_path or _CONFIG_DIR / "env.yaml").open() as fh:
        env_cfg = yaml.safe_load(fh)
    with (costs_path or _CONFIG_DIR / "costs.yaml").open() as fh:
        cost_cfg = yaml.safe_load(fh)

    if cost_cfg.get("default") is not None:
        raise EpisodeError(
            "costs.yaml declares a default. Remove it: a defaulted cost hides a missing "
            "entry, and a mispriced-to-zero test is the cheapest reward hack available."
        )
    support = tuple(float(b) for b in env_cfg["budget"]["support"])
    weights = tuple(float(w) for w in env_cfg["budget"]["weights"])
    if len(support) != len(weights):
        raise EpisodeError("budget support and weights differ in length")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise EpisodeError(f"budget weights sum to {sum(weights)}, not 1")

    return EpisodeConfig(
        max_turns=int(env_cfg["episode"]["max_turns"]),
        dedup_repeat_orders=bool(env_cfg["episode"]["dedup_repeat_orders"]),
        expose_remaining_budget=bool(env_cfg["episode"]["expose_remaining_budget"]),
        budget_support=support,
        budget_weights=weights,
        costs={str(k): float(v) for k, v in cost_cfg["tests"].items()},
    )


def sample_budget(cfg: EpisodeConfig, rng: np.random.Generator) -> float:
    """B ~ p(B), per episode. One policy spans the whole cost-accuracy frontier."""
    return float(rng.choice(cfg.budget_support, p=cfg.budget_weights))


@dataclass(slots=True)
class EpisodeState:
    turn: int = 0
    spent: float = 0.0
    budget: float = 0.0
    ordered: set[str] = field(default_factory=set)
    revealed: dict[str, ResultValue] = field(default_factory=dict)
    prescribed: list[str] = field(default_factory=list)
    done: bool = False
    termination_reason: str | None = None


class DiagnosticEpisode:
    """One patient, one episode. Deterministic given (patient, seed, config hash) [I10]."""

    def __init__(
        self,
        record: PatientRecord,
        seed: int,
        config: EpisodeConfig | None = None,
        menu: ActionMenu | None = None,
        catalog: Catalog | None = None,
        budget: float | None = None,
    ) -> None:
        self.record = record
        self.seed = seed
        self.config = config or load_episode_config()
        self.menu = menu or build_menu()
        self.catalog = catalog or load_catalog()
        self._rng = np.random.default_rng(
            [seed, int.from_bytes(record.patient_id.encode()[-8:], "little")]
        )
        self._forced_budget = budget
        self.state = EpisodeState()
        self.steps: list[Step] = []

    # -------------------------------------------------------------------- lifecycle --
    def reset(self) -> Observation:
        self.state = EpisodeState()
        self.state.budget = (
            self._forced_budget
            if self._forced_budget is not None
            else sample_budget(self.config, self._rng)
        )
        self.steps = []
        return self.observe()

    def observe(self) -> Observation:
        return build_observation(
            view=self.record.view(),
            revealed=dict(self.state.revealed),
            turn=self.state.turn,
            remaining_budget=self.remaining_budget if self.config.expose_remaining_budget else 0.0,
            turns_remaining=self.config.max_turns - self.state.turn,
            menu_fingerprint=self.menu.fingerprint(),
            catalog=self.catalog,
        )

    @property
    def remaining_budget(self) -> float:
        return self.state.budget - self.state.spent

    # ------------------------------------------------------------------------ step --
    def step(self, action: Action) -> tuple[Observation, bool, dict[str, Any]]:
        """Advance one turn. Returns (observation, done, info) -- never a reward."""
        if self.state.done:
            raise EpisodeError("episode already terminated")
        if action.action_id not in self.menu.ids:
            raise EpisodeError(f"action {action.action_id!r} is not on the global menu [I3]")

        info: dict[str, Any] = {}
        self.state.turn += 1

        if isinstance(action, OrderTest):
            info = self._do_order(action)
        elif isinstance(action, Prescribe):
            self.state.prescribed.append(action.treatment_key)
            self.steps.append(Step(turn=self.state.turn, action=action))
        elif isinstance(action, Diagnose):
            self._check_distribution(action)
            self.steps.append(Step(turn=self.state.turn, action=action))
            self._terminate("diagnose")
        elif isinstance(action, Abstain):
            self.steps.append(Step(turn=self.state.turn, action=action))
            self._terminate("abstain")
        else:  # pragma: no cover - Action is a closed union
            raise EpisodeError(f"unhandled action type {type(action)!r}")

        if not self.state.done and self.state.turn >= self.config.max_turns:
            self._terminate("max_turns")

        return self.observe(), self.state.done, info

    def _check_distribution(self, action: Diagnose) -> None:
        """Reject a report over labels that are not in the frozen taxonomy.

        Caught here rather than in the reward engine so a malformed report fails at the
        turn that produced it, with that turn's context, instead of surfacing as a
        scoring anomaly hours later during offline rescoring.
        """
        if not action.distribution:
            raise EpisodeError("diagnose with an empty distribution")
        slugs = set(load_taxonomy().slugs)
        unknown = sorted(set(action.distribution) - slugs)
        if unknown:
            raise EpisodeError(f"diagnosis names labels outside the taxonomy: {unknown}")

    def _do_order(self, action: OrderTest) -> dict[str, Any]:
        key = self.menu.test_key_for(action.action_id)
        if key != action.test_key:
            raise EpisodeError(
                f"action_id {action.action_id!r} names test {key!r}, payload says "
                f"{action.test_key!r}"
            )
        spec = self.catalog.test(key)
        cost = self.config.cost_of(key)

        if self.config.dedup_repeat_orders and key in self.state.ordered:
            # Second order of the same test: free, and returns the cached result.
            # Without this the agent finds the cheapest test and spams it.
            cached = tuple(
                build_result(a, self.record.analytes[a], self.catalog) for a in spec.analytes
            )
            self.steps.append(
                Step(turn=self.state.turn, action=action, revealed=cached,
                     was_duplicate=True, cost_charged=0.0)
            )
            return {"duplicate": True, "cost": 0.0}

        if cost > self.remaining_budget + 1e-9:
            # Refused, not silently truncated: the ledger must never go negative
            # (test_budget_never_exceeded). The turn is still consumed, so refusing is
            # not a free retry.
            self.steps.append(
                Step(turn=self.state.turn, action=action, revealed=(), cost_charged=0.0)
            )
            return {"refused": True, "reason": "unaffordable", "cost": cost}

        self.state.spent += cost
        self.state.ordered.add(key)
        for a in spec.analytes:
            # Every analyte has a value for every patient -- no None, no "unavailable",
            # no default-to-normal [I4].
            self.state.revealed[a] = self.record.analytes[a]
        revealed = tuple(
            build_result(a, self.record.analytes[a], self.catalog) for a in spec.analytes
        )
        self.steps.append(
            Step(turn=self.state.turn, action=action, revealed=revealed, cost_charged=cost)
        )
        return {"cost": cost, "analytes": list(spec.analytes)}

    def _terminate(self, reason: str) -> None:
        self.state.done = True
        self.state.termination_reason = reason

    # ----------------------------------------------------------------- trajectories --
    def trajectory(self) -> dict[str, Any]:
        """The persisted record of this episode, ready for offline scoring [I8]."""
        return {
            "patient_id": self.record.patient_id,
            "seed": self.seed,
            "config_hash": self.config.hash(),
            "menu_fingerprint": self.menu.fingerprint(),
            "budget": self.state.budget,
            "spent": self.state.spent,
            "steps": [s.model_dump(mode="json") for s in self.steps],
            "terminated": self.state.done,
            "termination_reason": self.state.termination_reason or "max_turns",
        }
