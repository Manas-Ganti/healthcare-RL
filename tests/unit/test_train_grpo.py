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


def test_rollout_weights_are_synced_every_step(trainer) -> None:
    """On-policy means the sampler must see the trained weights before the next batch.

    Without a sync the rollouts keep coming from the reference policy while the trained
    adapter drifts away from it. Nothing crashes: the run just stops being GRPO, and the
    KL term grows for a reason that is very hard to find from the logs. This is the test
    that would have caught it -- `sync_rollout_weights` existed on the protocol and on
    both implementations, and nothing called it.
    """
    trainer.run(steps=3)
    assert trainer.updater.syncs == 3
    assert trainer.n_syncs == 3


def test_sync_cadence_is_configurable_and_respected(trainer) -> None:
    from dataclasses import replace

    trainer.config = replace(trainer.config, sync_every=2)
    trainer.run(steps=4)
    assert trainer.updater.syncs == 2


def test_torch_updater_refuses_to_sync_into_the_void() -> None:
    """A sync with no backend would save to disk and push nothing. Fail, and fail early.

    The check runs before the model load, so this is assertable without a GPU -- and more
    to the point, a misconfigured sync costs a second rather than two minutes of loading
    weights it is about to not use.
    """
    from dxenv.train.grpo import TorchLoRAUpdater, TrainingError

    updater = TorchLoRAUpdater(config=GRPOConfig(), backend=None)
    with pytest.raises(TrainingError, match="no backend to sync to"):
        updater.sync_rollout_weights()


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


# ------------------------------------------------------------------- checkpointing --


def test_resume_restores_step_stage_and_monitor_windows(trainer) -> None:
    """On a scheduler a long run is a chain of jobs. What is not saved silently resets.

    The monitor windows are the subtle one: refilled from empty, the ceiling and collapse
    detectors cannot fire until half a window has passed, so they are OFF for the first
    stretch of every job in the chain.
    """
    trainer.run(steps=3)
    before = (trainer.step_index, trainer.stage.name, list(trainer.stage_rewards),
              len(trainer.ceiling_monitor.rewards), len(trainer.cost_monitor.counts),
              len(trainer.degenerate_monitor.flags))
    assert before[3] > 0 and before[4] > 0

    fresh = GRPOTrainer(
        trainer.config, trainer.records, trainer.splits, trainer.policy_factory,
        NullUpdater(), ctx=trainer.ctx, verify_frozen=False,
    )
    assert fresh.step_index == 0 and not fresh.ceiling_monitor.rewards
    assert fresh.load_state()
    after = (fresh.step_index, fresh.stage.name, list(fresh.stage_rewards),
             len(fresh.ceiling_monitor.rewards), len(fresh.cost_monitor.counts),
             len(fresh.degenerate_monitor.flags))
    assert after == before


def test_resume_continues_the_rng_rather_than_repeating_it(trainer, tmp_path) -> None:
    """Without the RNG state each job draws the same patients in the same order."""
    trainer.run(steps=2)
    first_ids = {r.patient_id for r in trainer.sample_patients()}

    fresh = GRPOTrainer(
        trainer.config, trainer.records, trainer.splits, trainer.policy_factory,
        NullUpdater(), ctx=trainer.ctx, verify_frozen=False,
    )
    fresh.load_state()
    assert {r.patient_id for r in fresh.sample_patients()} == first_ids

    restarted = GRPOTrainer(
        trainer.config, trainer.records, trainer.splits, trainer.policy_factory,
        NullUpdater(), ctx=trainer.ctx, verify_frozen=False,
    )
    assert {r.patient_id for r in restarted.sample_patients()} != first_ids


def test_load_state_returns_false_with_nothing_to_resume(trainer) -> None:
    from dataclasses import replace

    fresh = GRPOTrainer(
        replace(trainer.config, run_id="never-run"), trainer.records, trainer.splits,
        trainer.policy_factory, NullUpdater(), ctx=trainer.ctx, verify_frozen=False,
    )
    assert fresh.load_state() is False


def test_resume_refuses_across_a_reward_config_change(trainer) -> None:
    """Monitor windows measured under old weights must not be mixed with new ones."""
    from dataclasses import replace

    from dxenv.train.grpo import TrainingError

    trainer.run(steps=2)
    fresh = GRPOTrainer(
        trainer.config, trainer.records, trainer.splits, trainer.policy_factory,
        NullUpdater(), ctx=trainer.ctx, verify_frozen=False,
    )
    fresh.meta = replace(fresh.meta, reward_config_hash="different")
    with pytest.raises(TrainingError, match="cannot resume"):
        fresh.load_state()


def test_checkpoint_survives_a_halted_run(trainer, monkeypatch) -> None:
    """A wall-clock kill or a tripped monitor must still leave something to resume from."""
    from dxenv.train import grpo

    trainer.run(steps=2)

    def exploding(*a, **kw):
        raise CeilingViolation("injected")

    monkeypatch.setattr(grpo, "assert_below_ceiling", exploding)
    with pytest.raises(CeilingViolation):
        trainer.run(steps=1)
    state = trainer.config.root / trainer.config.run_id / GRPOTrainer.STATE_FILE
    assert state.exists()


def test_token_weighted_accumulation_matches_a_single_pass() -> None:
    """Each chunk is scaled by its SHARE OF THE STEP'S TOKENS, not by 1/n_chunks.

    Scaling by 1/n_chunks would reweight the step toward short sequences -- every chunk
    contributing equally regardless of how many tokens it carries -- which silently
    changes what is optimised rather than how it is computed.
    """
    import numpy as np

    per_seq_loss = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    tokens = np.array([100.0, 10.0, 50.0, 10.0, 200.0, 5.0])

    one_pass = float((per_seq_loss * tokens).sum() / tokens.sum())
    micro = 2
    denom = tokens.sum()
    accumulated = sum(
        float((per_seq_loss[i : i + micro] * tokens[i : i + micro]).sum() / denom)
        for i in range(0, len(per_seq_loss), micro)
    )
    assert abs(accumulated - one_pass) < 1e-12

    naive = sum(
        float((per_seq_loss[i : i + micro] * tokens[i : i + micro]).sum()
              / max(tokens[i : i + micro].sum(), 1.0))
        / (len(per_seq_loss) // micro)
        for i in range(0, len(per_seq_loss), micro)
    )
    assert abs(naive - one_pass) > 0.1, "the wrong scaling should be visibly different"
