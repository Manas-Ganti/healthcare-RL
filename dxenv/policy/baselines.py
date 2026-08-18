"""Reference policies to report every result against.

CLAUDE.md 10 requires baselines: prompted, SFT-only, greedy heuristic, Bayes-optimal.
The two that need no model live here, plus the blank-record floor.

Every policy here reads ONLY the observation. That is not a stylistic choice: a baseline
that peeks at the record would silently raise the bar the learned policy is measured
against, and the comparison would be meaningless in the flattering direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import ActionKind, ActionMenu, action_id, build_menu
from dxenv.env.bayes import entropy, posterior
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.episode import DiagnosticEpisode
from dxenv.env.obs_model import (
    CatTable,
    ObservationModel,
    QuantTable,
    ResultValue,
    build_observation_model,
)
from dxenv.env.schemas import Action, Diagnose, Observation, OrderTest

TOP_K_REPORT: int | None = None
"""How many labels a report names; None reports the full distribution.

Defaults to None deliberately. Truncating to a top-k costs an UNINFORMED policy far more
than an informed one -- the prior's mass is spread across the whole label set, so a top-8
prior report throws away most of it and the blank-record floor comes out artificially
low. That would make every other policy look better than it is, measured against a
floor that was depressed by a serialisation choice."""


class Policy(Protocol):
    def act(self, episode: DiagnosticEpisode, obs: Observation) -> Action: ...


def evidence_from_observation(obs: Observation) -> dict[str, ResultValue]:
    """Everything the agent legitimately knows, read off the observation alone."""
    ev: dict[str, ResultValue] = {"presenting_complaint": obs.presenting_complaint}
    for r in (*obs.vitals, *obs.revealed_results):
        ev[r.analyte] = r.value_number if r.value_number is not None else str(r.value_code)
    return ev


def _report(
    belief: npt.NDArray[np.float64], tax: Taxonomy, top_k: int | None = TOP_K_REPORT
) -> Diagnose:
    idx = np.argsort(-belief) if top_k is None else np.argsort(-belief)[:top_k]
    raw = {tax.slugs[int(i)]: float(belief[int(i)]) for i in idx}
    total = sum(raw.values())
    return Diagnose(
        action_id=action_id(ActionKind.DIAGNOSE, "diagnose"),
        distribution={k: v / total for k, v in raw.items()},
    )


@dataclass(slots=True)
class PriorPolicy:
    """Report the prevalence prior. Orders nothing.

    The lazy attractor, and the floor every other policy must clear by a real margin.
    """

    taxonomy: Taxonomy = field(default_factory=load_taxonomy)

    def act(self, episode: DiagnosticEpisode, obs: Observation) -> Action:  # noqa: ARG002
        # Both parameters are unused BY DESIGN -- ignoring the observation entirely is
        # what makes this the blank-record floor. They stay for Policy conformance.
        return _report(self.taxonomy.prior(), self.taxonomy)


@dataclass(slots=True)
class VitalsOnlyPolicy:
    """Bayes-update on the free observation, then report. Still orders nothing.

    This is the honest "no-test" baseline: it uses everything available for free, so the
    gap to a testing policy measures what TESTS bought, not what Bayes bought.
    """

    taxonomy: Taxonomy = field(default_factory=load_taxonomy)
    model: ObservationModel = field(default_factory=build_observation_model)

    def act(self, episode: DiagnosticEpisode, obs: Observation) -> Action:  # noqa: ARG002
        return _report(posterior(evidence_from_observation(obs), self.model), self.taxonomy)


@dataclass(slots=True)
class GreedyBayesPolicy:
    """Order the test with the lowest expected posterior entropy per unit cost, then report.

    NOT the optimal policy and not the ceiling: it is myopic, one step of lookahead, and
    it ignores what it would do with the answer. It is a strong, cheap reference point.

    The entropy calculation chooses actions only. It never enters the reward -- that
    would be an information-gain bonus, which I5 prohibits precisely because an agent
    will then find tests that maximise entropy reduction without improving the answer.
    """

    max_tests: int = 6
    taxonomy: Taxonomy = field(default_factory=load_taxonomy)
    catalog: Catalog = field(default_factory=load_catalog)
    model: ObservationModel = field(default_factory=build_observation_model)
    menu: ActionMenu = field(default_factory=build_menu)
    cost_weight: float = 0.5
    """Exponent on cost when ranking. 0 ignores price; 1 is strict value-per-unit-cost."""

    def _expected_entropy(self, belief: npt.NDArray[np.float64], analyte: str) -> float:
        table = self.model.table(analyte)
        if isinstance(table, CatTable):
            joint = belief[:, None] * table.probs
            marg = joint.sum(0)
            post = joint / np.maximum(marg, 1e-300)
            ent = -(post * np.log(np.maximum(post, 1e-300))).sum(0)
            return float(marg @ ent)
        assert isinstance(table, QuantTable)
        grid = np.linspace(float(table.mean.min()), float(table.mean.max()), 9)
        total = weight = 0.0
        for x in grid:
            ll = table.log_likelihood(float(x))
            ll = ll - ll.max()
            w = belief * np.exp(ll)
            s = float(w.sum())
            if s <= 0.0:
                continue
            total += s * entropy(w / s)
            weight += s
        return float(total / max(weight, 1e-300))

    def act(self, episode: DiagnosticEpisode, obs: Observation) -> Action:
        belief = posterior(evidence_from_observation(obs), self.model)
        if len(episode.state.ordered) >= self.max_tests:
            return _report(belief, self.taxonomy)

        best: tuple[float, str] | None = None
        for key in self.catalog.test_keys:
            if key in episode.state.ordered:
                continue
            cost = episode.config.cost_of(key)
            if cost > episode.remaining_budget:
                continue
            headline = self.catalog.test(key).analytes[0]
            score = self._expected_entropy(belief, headline) * (cost**self.cost_weight)
            if best is None or score < best[0]:
                best = (score, key)
        if best is None:
            return _report(belief, self.taxonomy)
        key = best[1]
        return OrderTest(
            action_id=self.menu.id_for_test(key), test_key=key, prediction="normal"
        )


def run_episode(episode: DiagnosticEpisode, policy: Policy) -> dict[str, object]:
    """Drive one episode to termination and return the stored trajectory."""
    obs = episode.reset()
    done = False
    while not done:
        obs, done, _ = episode.step(policy.act(episode, obs))
    return episode.trajectory()
