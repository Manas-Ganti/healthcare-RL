"""Lockstep rollouts: one backend call per ROUND of turns, not per turn.

Without this the GPU path is unaffordable. Measured against the plan, Gate B issues
4,800-12,800 single-sequence requests and a 2000-step GRPO run issues ~640,000, which is
roughly 356 hours sequentially. The episodes in a group are independent at any given turn,
so they batch -- and the batch must shrink as episodes terminate at different turns.
"""

from __future__ import annotations

import pytest
from dxenv.policy.llm import BackendError, LLMPolicy, RandomBackend, batched_act
from dxenv.policy.rollout import RolloutContext, rollout_group, rollout_lockstep


@pytest.fixture(scope="module")
def ctx(episode_config, reward_config, taxonomy, catalog, obs_model):
    return RolloutContext(
        episode_config=episode_config, reward_config=reward_config,
        taxonomy=taxonomy, catalog=catalog, model=obs_model,
    )


class CountingBackend(RandomBackend):
    """A grammar sampler that records how many times the engine was reached."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.calls = 0
        self.widths: list[int] = []
        self._depth = 0

    def generate(self, conversations, **kw):
        # Depth-guarded: RandomBackend's per-seed path fans out into one recursive call
        # per conversation, an artefact of the fake rather than of the batching. Counting
        # those made the first version of this test read a batched round of 8 as calls of
        # width 8, 1, 1, ... -- exactly backwards. VLLMBackend does not fan out; it builds
        # a list of SamplingParams and issues one engine.chat.
        if self._depth == 0:
            self.calls += 1
            self.widths.append(len(conversations))
        self._depth += 1
        try:
            return super().generate(conversations, **kw)
        finally:
            self._depth -= 1


def test_lockstep_batches_the_whole_group_per_round(fixture_corpus, ctx) -> None:
    """One call per round, each covering every episode still running."""
    backend = CountingBackend(seed=3)
    rollouts = rollout_group(
        fixture_corpus[0], lambda s: LLMPolicy(backend=backend, seed=s), 8, 100, ctx,
        budget=200.0,
    )
    assert len(rollouts) == 8
    assert backend.calls == len(backend.widths)
    # The first round must cover all 8; later rounds cover only the survivors.
    assert backend.widths[0] == 8
    assert backend.calls < sum(r.breakdown.n_turns for r in rollouts), (
        "batching saved nothing: as many calls as turns"
    )


def test_batch_shrinks_as_episodes_terminate(fixture_corpus, ctx) -> None:
    """A finished episode must not be stepped again."""
    backend = CountingBackend(seed=11)
    rollout_group(
        fixture_corpus[1], lambda s: LLMPolicy(backend=backend, seed=s), 8, 7, ctx,
        budget=200.0,
    )
    assert backend.widths == sorted(backend.widths, reverse=True)
    assert backend.widths[-1] >= 1


def test_lockstep_matches_sequential_exactly(fixture_corpus, ctx) -> None:
    """Batching must be a throughput change and nothing else.

    Per-episode seeds are carried through `seeds`, so a batched call is as reproducible as
    the sequential calls it replaces -- if that were not true, batching would be buying
    speed with silent drift in the results.
    """
    shared = RandomBackend(seed=5)
    batched = rollout_group(
        fixture_corpus[2], lambda s: LLMPolicy(backend=shared, seed=s), 6, 42, ctx,
        budget=200.0,
    )
    # A fresh backend per policy takes the sequential path by construction.
    sequential = rollout_group(
        fixture_corpus[2], lambda s: LLMPolicy(backend=RandomBackend(seed=5), seed=s),
        6, 42, ctx, budget=200.0,
    )
    assert [r.reward for r in batched] == [r.reward for r in sequential]
    assert [r.trajectory for r in batched] == [r.trajectory for r in sequential]


def test_heuristic_policies_take_the_sequential_path(fixture_corpus, ctx) -> None:
    """The baselines are pure CPU and have no backend to share."""
    from dxenv.policy.baselines import GreedyBayesPolicy
    from dxenv.policy.rollout import constant_factory

    rollouts = rollout_group(
        fixture_corpus[3], constant_factory(GreedyBayesPolicy), 3, 1, ctx, budget=200.0
    )
    assert len(rollouts) == 3
    assert all(r.breakdown.n_turns > 0 for r in rollouts)


def test_batched_act_refuses_unshared_backends(fixture_corpus, catalog, episode_config,
                                               menu) -> None:
    """Separate backends mean separate engines, which is what batching exists to avoid."""
    from dxenv.env.episode import DiagnosticEpisode

    policies = [LLMPolicy(backend=RandomBackend(seed=i), menu=menu) for i in range(2)]
    episodes = [
        DiagnosticEpisode(fixture_corpus[i], seed=i, config=episode_config,
                          catalog=catalog, budget=100.0)
        for i in range(2)
    ]
    observations = [e.reset() for e in episodes]
    with pytest.raises(BackendError, match="share one backend"):
        batched_act(policies, episodes, observations)


def test_lockstep_across_patients(fixture_corpus, ctx) -> None:
    """GRPO batches a whole step, not just one group -- different patients, one call."""
    backend = CountingBackend(seed=9)
    specs = [(rec, 1000 + i, 200.0) for i, rec in enumerate(fixture_corpus[:12])]
    rollouts = rollout_lockstep(specs, lambda s: LLMPolicy(backend=backend, seed=s), ctx)
    assert len(rollouts) == 12
    assert backend.widths[0] == 12
    assert {r.patient_id for r in rollouts} == {r.patient_id for r, _, _ in specs}
