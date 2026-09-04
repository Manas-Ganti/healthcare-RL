"""Rollout collection: one episode, or a group of k, scored and ready to store.

This is the seam between the environment (which produces trajectories and no reward) and
everything that consumes reward -- rejection sampling, SFT dataset construction, GRPO.
It lives in `policy/` rather than `train/` because Phase 3 needs it before a training
loop exists, and `reward/` may not import `policy/` but `policy/` may import `reward/`.

Every rollout carries its two ceilings with it. Computing them here rather than in the
training loop means an offline rescoring pass can check I9 without re-deriving the Bayes
posterior, and it means a stored corpus is self-describing: a line that scored above its
own recorded ceiling is visible in the file, not only in a monitor that was running at
the time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from dxenv.data.corpus import PatientRecord
from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.bayes import expected_ceiling, hard_ceiling
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.episode import (
    DiagnosticEpisode,
    EpisodeConfig,
    load_episode_config,
    sample_budget,
)
from dxenv.env.obs_model import ObservationModel, build_observation_model
from dxenv.policy.baselines import Policy, run_episode
from dxenv.policy.llm import LLMPolicy, batched_act
from dxenv.reward.engine import (
    GroundTruth,
    RewardBreakdown,
    RewardConfig,
    load_reward_config,
    score_trajectory,
)
from dxenv.reward.scoring import weighted_score_fn


@dataclass(frozen=True, slots=True)
class Rollout:
    """One scored episode, with everything needed to store, filter or train on it."""

    patient_id: str
    condition: str
    """GROUND TRUTH. Present because rejection sampling and the ceiling need it; it is
    never rendered into a prompt. `dxenv.data.store` keeps it out of the model-safe
    projection."""

    seed: int
    budget: float
    trajectory: dict[str, Any]
    breakdown: RewardBreakdown
    generations: tuple[dict[str, Any], ...]
    hard_ceiling: float
    expected_ceiling: float

    @property
    def reward(self) -> float:
        return self.breakdown.total

    @property
    def n_tests(self) -> int:
        return self.breakdown.n_tests_charged

    @property
    def diagnosed(self) -> bool:
        return self.breakdown.termination_reason == "diagnose"

    @property
    def headroom(self) -> float:
        """How far below the per-episode hard ceiling this rollout landed."""
        return self.hard_ceiling - self.reward

    def ground_truth_dict(self) -> dict[str, Any]:
        return {"condition": self.condition, "patient_id": self.patient_id}

    def tags(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "diagnosis": self.breakdown.diagnosis,
            "n_tests": self.n_tests,
            "termination_reason": self.breakdown.termination_reason,
            "hard_ceiling": self.hard_ceiling,
            "expected_ceiling": self.expected_ceiling,
            "generations": list(self.generations),
        }


@dataclass(slots=True)
class RolloutContext:
    """The objects every rollout needs, loaded once.

    Constructing an `ObservationModel` per rollout costs more than the rollout does on
    the RandomBackend and is invisible next to a 7B forward pass -- which is exactly how
    it survives unnoticed until a 20k-episode sweep takes an hour longer than it should.
    """

    episode_config: EpisodeConfig = field(default_factory=load_episode_config)
    reward_config: RewardConfig | None = None
    taxonomy: Taxonomy = field(default_factory=load_taxonomy)
    catalog: Catalog = field(default_factory=load_catalog)
    model: ObservationModel = field(default_factory=build_observation_model)
    _score_fn: Any = None
    _hard: float | None = None
    _expected: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reward_config is None:
            self.reward_config = load_reward_config()
        self._score_fn = weighted_score_fn(self.taxonomy, self.reward_config.severity)
        self._hard = hard_ceiling(self._score_fn, len(self.taxonomy))

    @property
    def score_fn(self) -> Any:
        return self._score_fn

    @property
    def hard(self) -> float:
        assert self._hard is not None
        return self._hard

    def expected_for(self, record: PatientRecord) -> float:
        """Full-information Bayes value for this patient. An upper bound; see env/bayes.

        Memoised per patient. It is a deterministic function of the record and the k
        rollouts of a group all share it, so recomputing a 149-way posterior over ~105
        analytes eight times per patient is pure waste -- invisible next to a 7B forward
        pass, and the dominant cost of a heuristic-policy sweep.
        """
        hit = self._expected.get(record.patient_id)
        if hit is None:
            hit = expected_ceiling(dict(record.analytes), self._score_fn, self.model)
            self._expected[record.patient_id] = hit
        return hit


PolicyFactory = Callable[[int], Policy]
"""seed -> a fresh policy. A factory, not a policy, so k samples of the same patient get
k independent samplers instead of sharing one generator's state and correlating."""


def constant_factory(build: Callable[[], Policy]) -> PolicyFactory:
    """A factory for a policy that does not sample, and so has no use for the seed.

    The heuristic baselines are deterministic: `GreedyBayesPolicy` returns the same action
    for the same observation every time. Wrapping them says that explicitly, instead of
    scattering `lambda _seed: GreedyBayesPolicy()` -- which reads like an oversight -- at
    every call site.
    """
    return lambda _seed: build()


def rollout_once(
    record: PatientRecord,
    policy_factory: PolicyFactory,
    seed: int,
    ctx: RolloutContext,
    budget: float | None = None,
) -> Rollout:
    """One episode, driven to termination and scored. Deterministic in `seed` [I10]."""
    assert ctx.reward_config is not None
    policy = policy_factory(seed)
    episode = DiagnosticEpisode(
        record, seed=seed, config=ctx.episode_config, catalog=ctx.catalog, budget=budget
    )
    trajectory = run_episode(episode, policy)
    breakdown = score_trajectory(
        trajectory,
        GroundTruth(record.condition, record.analytes, record.allergies),
        ctx.reward_config,
        taxonomy=ctx.taxonomy,
        catalog=ctx.catalog,
        model=ctx.model,
    )
    gens = tuple(getattr(policy, "generations", ()) or ())
    return Rollout(
        patient_id=record.patient_id,
        condition=record.condition,
        seed=seed,
        budget=float(trajectory["budget"]),
        trajectory=trajectory,
        breakdown=breakdown,
        generations=gens,
        hard_ceiling=ctx.hard,
        expected_ceiling=ctx.expected_for(record),
    )


def _score_episode(
    record: PatientRecord, episode: DiagnosticEpisode, policy: Any, seed: int,
    ctx: RolloutContext,
) -> Rollout:
    assert ctx.reward_config is not None
    trajectory = episode.trajectory()
    breakdown = score_trajectory(
        trajectory,
        GroundTruth(record.condition, record.analytes, record.allergies),
        ctx.reward_config,
        taxonomy=ctx.taxonomy, catalog=ctx.catalog, model=ctx.model,
    )
    return Rollout(
        patient_id=record.patient_id, condition=record.condition, seed=seed,
        budget=float(trajectory["budget"]), trajectory=trajectory, breakdown=breakdown,
        generations=tuple(getattr(policy, "generations", ()) or ()),
        hard_ceiling=ctx.hard, expected_ceiling=ctx.expected_for(record),
    )


def rollout_lockstep(
    specs: Sequence[tuple[PatientRecord, int, float]],
    policy_factory: PolicyFactory,
    ctx: RolloutContext,
) -> list[Rollout]:
    """Run many episodes together, one backend call per ROUND of turns.

    This is what makes the GPU path affordable. Driving episodes one at a time issues a
    single-sequence request per turn and wastes most of the device: Gate B needs
    4,800-12,800 calls and a 2000-step GRPO run needs ~640,000, which is ~356 hours
    sequentially. The episodes are independent at any given turn, so they batch.

    Episodes finish at different turns, so each round batches only the ones still running
    -- the batch shrinks as episodes terminate, which is exactly right: a finished episode
    must not be stepped again.

    Falls back to the sequential path for policies without a shared LLM backend; the
    heuristic baselines are pure CPU and gain nothing from batching.
    """
    episodes, policies, records, seeds = [], [], [], []
    for record, seed, budget in specs:
        policies.append(policy_factory(seed))
        episodes.append(DiagnosticEpisode(
            record, seed=seed, config=ctx.episode_config, catalog=ctx.catalog, budget=budget
        ))
        records.append(record)
        seeds.append(seed)

    # Batching is only a win when the policies share ONE engine. A factory that builds a
    # fresh backend per policy (the heuristic baselines, and the grammar sampler used in
    # tests) has nothing to gain and would break the shared-engine assumption, so it takes
    # the sequential path and behaves exactly as before.
    llm_policies = [p for p in policies if isinstance(p, LLMPolicy)]
    shareable = (
        len(llm_policies) == len(policies) > 0
        and len({id(p.backend) for p in llm_policies}) == 1
    )
    if not shareable:
        return [
            rollout_once(rec, policy_factory, seed, ctx, budget=budget)
            for rec, seed, budget in specs
        ]

    observations = [ep.reset() for ep in episodes]
    done = [False] * len(episodes)
    while not all(done):
        live = [i for i, d in enumerate(done) if not d]
        actions = batched_act(
            [cast(LLMPolicy, policies[i]) for i in live],
            [episodes[i] for i in live],
            [observations[i] for i in live],
        )
        for i, action in zip(live, actions, strict=True):
            observations[i], done[i], _ = episodes[i].step(action)

    return [
        _score_episode(records[i], episodes[i], policies[i], seeds[i], ctx)
        for i in range(len(episodes))
    ]


def rollout_group(
    record: PatientRecord,
    policy_factory: PolicyFactory,
    k: int,
    base_seed: int,
    ctx: RolloutContext,
    budget: float | None = None,
) -> list[Rollout]:
    """k independent samples of ONE patient -- the group GRPO takes advantages within.

    The budget is drawn once and shared across the group. Sampling it per rollout would
    put an easy episode and a hard one in the same group and score the difference as
    policy quality, which is the one thing an advantage is not allowed to measure.
    """
    if k < 1:
        raise ValueError(f"group size must be >= 1, got {k}")
    if budget is None:
        budget = sample_budget(ctx.episode_config, np.random.default_rng(base_seed))
    return rollout_lockstep(
        [(record, base_seed + i, budget) for i in range(k)], policy_factory, ctx
    )


def group_rewards(rollouts: Sequence[Rollout]) -> npt.NDArray[np.float64]:
    return np.array([r.reward for r in rollouts], dtype=np.float64)


def pass_at_k(rollouts: Sequence[Rollout], threshold: float) -> bool:
    """Did ANY of the k rollouts clear the bar? The Gate B pass@k statistic.

    Deliberately defined on reward rather than on correctness. Filtering on correct
    diagnosis alone selects for lucky guesses and shotgun test-ordering (CLAUDE.md 8.3);
    a trajectory that got it right after 40 tests is not a pass.
    """
    return any(r.reward >= threshold for r in rollouts)
