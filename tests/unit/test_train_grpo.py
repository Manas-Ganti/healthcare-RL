"""Phase 4: the GRPO loop, its monitors, and the curriculum (CLAUDE.md 9).

Everything here runs without a GPU, because the failures this project is exposed to --
leakage, reward hacking, a monitor that would not have fired -- do not live in the
backward pass. `NullUpdater` runs the whole loop with no model at all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from dxenv.data.splits import EvalLeakError, make_splits
from dxenv.policy.llm import LLMPolicy, RandomBackend
from dxenv.train.curriculum import Curriculum, CurriculumError, Stage, load_curriculum
from dxenv.train.grpo import (
    GRPOConfig,
    GRPOTrainer,
    NullUpdater,
    TrainingSequence,
    sequences_from_rollouts,
)
from dxenv.train.monitors import (
    CeilingViolation,
    CostCollapse,
    CostDistributionMonitor,
    DegenerateGroupMonitor,
    DegenerateGroups,
    RunningCeilingMonitor,
    assert_below_ceiling,
    assert_group_has_variance,
    clipped_surrogate,
    group_advantages,
    kl_k3,
)


@pytest.fixture
def trainer(tmp_path, fixture_corpus, episode_config, reward_config, taxonomy, catalog,
            obs_model):
    from dxenv.policy.rollout import RolloutContext

    splits = make_splits(fixture_corpus, seed=7)
    cfg = GRPOConfig(run_id="t", k=4, patients_per_step=3, monitor_every=100,
                     stage_window=3, root=tmp_path, seed=1)
    ctx = RolloutContext(
        episode_config=episode_config, reward_config=reward_config,
        taxonomy=taxonomy, catalog=catalog, model=obs_model,
    )
    return GRPOTrainer(
        cfg, {r.patient_id: r for r in fixture_corpus}, splits,
        lambda s: LLMPolicy(backend=RandomBackend(seed=s), seed=s),
        NullUpdater(), ctx=ctx, verify_frozen=False,
    )


# ----------------------------------------------------------------------- advantages --


def test_advantage_zero_when_group_identical() -> None:
    """The degenerate case behaves as expected: no spread, no gradient."""
    adv = group_advantages(np.full(8, 1.234))
    assert np.allclose(adv, 0.0)


def test_advantages_are_standardised() -> None:
    adv = group_advantages(np.array([0.0, 1.0, 2.0, 3.0]))
    assert abs(float(adv.mean())) < 1e-9
    assert abs(float(adv.std()) - 1.0) < 1e-6


def test_assert_group_has_variance_fires_on_a_flat_group() -> None:
    from dxenv.train.monitors import EntropyCollapse

    with pytest.raises(EntropyCollapse):
        assert_group_has_variance(np.full(8, 0.5))
    assert_group_has_variance(np.array([0.0, 1.0]))


# ------------------------------------------------------------------------------ KL --


def test_kl_matches_reference_implementation() -> None:
    """k3 on a fixed batch, against the estimator written out longhand."""
    rng = np.random.default_rng(0)
    logp = rng.normal(-2.0, 0.5, size=64)
    ref = rng.normal(-2.0, 0.5, size=64)
    expected = float(np.mean([
        np.exp(b - a) - (b - a) - 1.0 for a, b in zip(logp, ref, strict=True)
    ]))
    assert abs(kl_k3(logp, ref) - expected) < 1e-12


def test_kl_is_zero_for_identical_policies() -> None:
    logp = np.array([-1.0, -2.0, -0.5])
    assert abs(kl_k3(logp, logp)) < 1e-12


def test_kl_is_never_negative() -> None:
    """The reason for k3 over the naive difference: the naive one occasionally PAYS the
    policy for leaving the reference."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        a, b = rng.normal(-3, 1, size=16), rng.normal(-3, 1, size=16)
        assert kl_k3(a, b) >= -1e-12


def test_kl_rejects_non_finite_logprobs() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        kl_k3(np.array([np.nan]), np.array([-1.0]))


def test_clipped_surrogate_clips_large_ratios() -> None:
    """Beyond the clip range, further movement earns nothing."""
    logp_old = np.zeros(4)
    adv = np.ones(4)
    inside = clipped_surrogate(np.full(4, 0.05), logp_old, adv, clip_eps=0.2)
    outside = clipped_surrogate(np.full(4, 5.0), logp_old, adv, clip_eps=0.2)
    assert outside == pytest.approx(1.2)
    assert inside < outside


# -------------------------------------------------------------------- the detectors --


def test_ceiling_assertion_fires_on_synthetic_violation() -> None:
    """Test the DETECTOR, not just the thing it detects [I9].

    An audit that would not catch a real failure is worse than none, because it
    manufactures confidence.
    """
    assert_below_ceiling(1.0, 2.0, "p", {})
    with pytest.raises(CeilingViolation, match="LEAK"):
        assert_below_ceiling(2.5, 2.0, "patient-x", {"steps": []})


def test_trainer_halts_on_an_injected_ceiling_violation(trainer, monkeypatch) -> None:
    """End to end: a rollout scoring above its hard ceiling must stop the run."""
    from dxenv.train import grpo

    def exploding(reward, ceiling, patient_id, trajectory, tolerance=1e-6):
        raise CeilingViolation(f"injected for {patient_id}")

    monkeypatch.setattr(grpo, "assert_below_ceiling", exploding)
    with pytest.raises(CeilingViolation, match="injected"):
        trainer.run_step()


def test_running_ceiling_monitor_detects_a_sustained_breach() -> None:
    """A lucky rollout may beat the EXPECTED ceiling; a running mean may not."""
    m = RunningCeilingMonitor(window=64, tolerance=0.02)
    for _ in range(64):
        m.update(reward=1.0, expected_ceiling=1.5)
    assert not m.breached
    m2 = RunningCeilingMonitor(window=64, tolerance=0.02)
    for _ in range(64):
        m2.update(reward=2.0, expected_ceiling=1.5)
    assert m2.breached


def test_cost_monitor_detects_collapse_to_zero_tests() -> None:
    m = CostDistributionMonitor(window=64)
    for _ in range(64):
        m.update(0, hit_budget_cap=False)
    with pytest.raises(CostCollapse, match="NO tests"):
        m.assert_healthy()


def test_cost_monitor_detects_collapse_to_the_budget_cap() -> None:
    m = CostDistributionMonitor(window=64)
    for _ in range(64):
        m.update(12, hit_budget_cap=True)
    with pytest.raises(CostCollapse, match="whole budget"):
        m.assert_healthy()


def test_cost_monitor_is_quiet_on_a_healthy_distribution() -> None:
    m = CostDistributionMonitor(window=64)
    rng = np.random.default_rng(0)
    for _ in range(64):
        m.update(int(rng.integers(1, 7)), hit_budget_cap=False)
    m.assert_healthy()


def test_degenerate_group_monitor_uses_a_rate_not_a_single_group() -> None:
    """Halting on one flat group would halt on a legitimately easy patient."""
    m = DegenerateGroupMonitor(window=64, max_fraction=0.5)
    for i in range(64):
        m.update(np.full(4, 1.0) if i < 20 else np.array([0.0, 1.0, 2.0, 3.0]))
    m.assert_healthy()
    for _ in range(64):
        m.update(np.full(4, 1.0))
    with pytest.raises(DegenerateGroups):
        m.assert_healthy()


# ------------------------------------------------------------------------ curriculum --


def test_curriculum_advances_on_criterion() -> None:
    c = load_curriculum()
    first = c.stages[0]
    assert c.next_stage(first.name, first.advance_criterion - 0.01) == first.name
    assert c.next_stage(first.name, first.advance_criterion + 0.01) == c.stages[1].name


def test_curriculum_does_not_skip_stages() -> None:
    """A huge reward advances ONE stage. Staging exists because the policy must learn
    short-horizon behaviour before it is trusted with a full budget."""
    c = load_curriculum()
    assert c.next_stage(c.stages[0].name, 1e6) == c.stages[1].name
    assert c.next_stage(c.stages[-1].name, 1e6) == c.stages[-1].name


def test_curriculum_rejects_an_unknown_stage() -> None:
    with pytest.raises(CurriculumError, match="unknown stage"):
        load_curriculum().index_of("not_a_stage")


def test_trainer_advances_only_after_a_full_window(trainer) -> None:
    """One step is a handful of patients; advancing on that advances on noise."""
    tiny = Curriculum((
        Stage("a", comorbid=False, max_turns=8, advance_criterion=-1e9),
        Stage("b", comorbid=False, max_turns=20),
    ))
    trainer.curriculum = tiny
    trainer.stage = tiny.stages[0]
    trainer._stage_configs = {s.name: trainer._config_for(s) for s in tiny.stages}
    trainer.stage_rewards = []
    for _ in range(trainer.config.stage_window - 1):
        trainer.stage_rewards.append(0.0)
        assert not trainer.maybe_advance_stage()
    trainer.stage_rewards.append(0.0)
    assert trainer.maybe_advance_stage()
    assert trainer.stage.name == "b"


# ------------------------------------------------------------------------- the loop --


def test_training_never_reads_eval_split(trainer) -> None:
    """[I12] Every training read routes through the guard; the guard raises on eval."""
    for _ in range(4):
        records = trainer.sample_patients()
        assert all(r.patient_id in set(trainer.splits.train) for r in records)
    from dxenv.data.splits import guard_training_access

    with pytest.raises(EvalLeakError):
        guard_training_access([trainer.splits.eval[0]], trainer.splits)


def test_trainer_refuses_to_start_without_a_frozen_eval_split(
    tmp_path, fixture_corpus, monkeypatch
) -> None:
    """An unfrozen eval split is not an eval split. Fail at startup, not after the run."""
    from dxenv.data import splits as splits_mod

    monkeypatch.setattr(splits_mod, "_FROZEN_PATH", tmp_path / "absent.json")
    with pytest.raises(splits_mod.SplitError, match="does not exist"):
        GRPOTrainer(
            GRPOConfig(root=tmp_path), {r.patient_id: r for r in fixture_corpus},
            make_splits(fixture_corpus, seed=7),
            lambda s: LLMPolicy(backend=RandomBackend(seed=s)), NullUpdater(),
            verify_frozen=True,
        )


def test_loop_runs_and_persists_every_rollout(trainer, tmp_path) -> None:
    """CLAUDE.md 4: persist every trajectory ever generated."""
    reports = trainer.run(steps=3)
    assert len(reports) == 3
    run_dir = tmp_path / "t"
    lines = (run_dir / "episodes.jsonl").read_text().strip().splitlines()
    expected = 3 * trainer.config.patients_per_step * trainer.config.k
    assert len(lines) == expected
    assert json.loads((run_dir / "meta.json").read_text())["phase"] == "grpo"
    steps = (run_dir / "steps.jsonl").read_text().strip().splitlines()
    assert len(steps) == 3


def test_stored_lines_declare_their_env_config(trainer, tmp_path) -> None:
    """A curriculum stage changes max_turns, hence the hash. Every one must be declared."""
    trainer.run(steps=2)
    declared = set(trainer.meta.declared_env_hashes)
    for raw in (tmp_path / "t" / "episodes.jsonl").read_text().strip().splitlines():
        assert json.loads(raw)["trajectory"]["config_hash"] in declared


def test_sequences_carry_the_episode_advantage(trainer) -> None:
    """One episode-level advantage, broadcast across every token the episode generated."""
    from dxenv.policy.rollout import rollout_group

    records = trainer.sample_patients()
    rollouts = rollout_group(records[0], trainer.policy_factory, 4, 5, trainer.ctx,
                             budget=200.0)
    adv = group_advantages(np.array([r.reward for r in rollouts]))
    seqs = sequences_from_rollouts(rollouts, adv)
    assert seqs, "no generations were recorded"
    by_patient = {s.advantage for s in seqs}
    assert by_patient <= set(adv.tolist())
    for s in seqs:
        assert isinstance(s, TrainingSequence)
        assert [m["role"] for m in s.messages] == ["system", "user"]


def test_rollouts_without_generations_contribute_nothing(trainer) -> None:
    """The heuristic baselines have no generations; the loop must still run on them."""
    from dxenv.policy.baselines import GreedyBayesPolicy
    from dxenv.policy.rollout import constant_factory, rollout_group

    records = trainer.sample_patients()
    rollouts = rollout_group(records[0], constant_factory(GreedyBayesPolicy), 2, 5,
                             trainer.ctx, budget=200.0)
    assert sequences_from_rollouts(rollouts, np.zeros(2)) == []


def test_group_shares_one_budget(trainer) -> None:
    """An easy episode and a hard one in the same group would make the advantage measure
    the budget draw rather than the policy."""
    from dxenv.policy.rollout import rollout_group

    records = trainer.sample_patients()
    rollouts = rollout_group(records[0], trainer.policy_factory, 6, 11, trainer.ctx)
    assert len({r.budget for r in rollouts}) == 1


def test_loop_is_deterministic_under_seed(tmp_path, fixture_corpus, episode_config,
                                          reward_config, taxonomy, catalog, obs_model) -> None:
    """[I10] Same seed, same run."""
    from dxenv.policy.rollout import RolloutContext

    def build(run_id):
        splits = make_splits(fixture_corpus, seed=7)
        cfg = GRPOConfig(run_id=run_id, k=3, patients_per_step=2, monitor_every=100,
                         root=tmp_path, seed=99)
        ctx = RolloutContext(
            episode_config=episode_config, reward_config=reward_config,
            taxonomy=taxonomy, catalog=catalog, model=obs_model,
        )
        return GRPOTrainer(
            cfg, {r.patient_id: r for r in fixture_corpus}, splits,
            lambda s: LLMPolicy(backend=RandomBackend(seed=s), seed=s),
            NullUpdater(), ctx=ctx, verify_frozen=False,
        )

    a = [r.mean_reward for r in build("a").run(steps=3)]
    b = [r.mean_reward for r in build("b").run(steps=3)]
    assert a == b
