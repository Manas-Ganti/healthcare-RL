"""The GRPO loop.

7B + LoRA, vLLM for rollout generation, curriculum from short horizon to full budget.

The orchestration is separated from the gradient step
-----------------------------------------------------
`GRPOTrainer` owns everything that can be wrong without a GPU: which patients are
sampled, whether the eval split was touched, how advantages are formed, when a monitor
halts the run, what gets persisted. The gradient step arrives as an `Updater`, and
`NullUpdater` runs the entire loop with no model at all.

That is not a testing convenience bolted on afterwards. The failures this project is
actually exposed to are leakage, reward hacking, and a monitor that would not have fired
-- none of which live in the backward pass, and all of which would otherwise be
untestable without eight hours on an A100. `test_ceiling_assertion_fires_on_synthetic_
violation` and `test_training_never_reads_eval_split` both run in the fast suite because
of this split.

Credit assignment
-----------------
One episode-level advantage, broadcast uniformly across every token the episode
generated. Standard for multi-turn GRPO, and worth stating as an assumption rather than
inheriting as a default: it says a good episode makes each of its turns slightly more
likely, including the turns that were incidental to why it was good. The alternative --
per-turn credit from a learned value head -- reintroduces a learned model into a reward
pipeline whose entire premise is that reward is verifiable (CLAUDE.md 13, RLVR).

Monitors halt; they do not warn
-------------------------------
A reward above the hard ceiling halts the run and dumps the trajectory [I9]. Treat a trip
as a leak until proven otherwise. The temptation, at 3am on step 4000, is to raise the
tolerance -- which is why the tolerance is a constructor argument that gets logged into
the run metadata rather than a number edited in place.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from dxenv.data.corpus import PatientRecord
from dxenv.data.splits import Splits, assert_eval_frozen, guard_training_access
from dxenv.data.store import RunMeta, TrajectoryStore
from dxenv.data.taxonomy import load_taxonomy
from dxenv.env.episode import EpisodeConfig, load_episode_config, sample_budget
from dxenv.policy.rollout import Rollout, RolloutContext, group_rewards, rollout_group
from dxenv.train.curriculum import Curriculum, Stage, load_curriculum, load_training_ids
from dxenv.train.monitors import (
    CostDistributionMonitor,
    DegenerateGroupMonitor,
    RunningCeilingMonitor,
    assert_below_ceiling,
    group_advantages,
)


class TrainingError(ValueError):
    """Malformed training configuration. Never caught inside `dxenv.train`."""


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    run_id: str = "grpo"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    reference_adapter: Path | None = None
    """The SFT LoRA. The KL term is measured against it, so it is the thing the policy is
    allowed to drift FROM; without one, KL is measured against the base model."""

    k: int = 8
    patients_per_step: int = 8
    max_steps: int = 2000
    temperature: float = 1.0
    max_tokens: int = 512

    learning_rate: float = 1e-6
    kl_coef: float = 0.02
    clip_eps: float = 0.2
    lora_rank: int = 32
    lora_alpha: int = 64

    monitor_every: int = 10
    save_every: int = 100
    sync_every: int = 1
    """Steps between pushing the trained adapter back to the rollout sampler.

    1, because GRPO is on-policy: the advantages are only valid for the policy that
    produced the rollouts. Raising this trades correctness for throughput -- at
    sync_every=N the loop is doing N steps of off-policy updates against a stale sampler,
    which the clipped surrogate tolerates for small N and silently degrades for large
    ones. Set it deliberately, and watch the KL when you do.
    """
    ceiling_tolerance: float = 1e-6
    stage_window: int = 20
    """Steps of mean reward a stage is judged on before it may advance. One step is
    8 patients; advancing on that would advance on noise."""

    seed: int = 0
    root: Path = Path("runs")

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reference_adapter"] = str(self.reference_adapter) if self.reference_adapter else None
        d["root"] = str(self.root)
        return d


@dataclass(frozen=True, slots=True)
class TrainingSequence:
    """One turn's (prompt, completion) with the advantage of the episode that produced it."""

    messages: tuple[dict[str, str], ...]
    completion: str
    advantage: float
    patient_id: str
    turn: int


class Updater(Protocol):
    """The gradient step. Everything model-shaped lives behind this."""

    def update(self, batch: Sequence[TrainingSequence]) -> dict[str, float]: ...
    def sync_rollout_weights(self) -> None: ...
    def save(self, path: Path) -> None: ...


@dataclass(slots=True)
class NullUpdater:
    """Runs the loop without a model. Records what it was asked to train on.

    Used by the orchestration tests, and by `--dry-run`, which is the cheapest way to
    find out that a config would have halted on step 3 before spending a node-hour
    discovering it.
    """

    seen: list[TrainingSequence] = field(default_factory=list)
    steps: int = 0

    def update(self, batch: Sequence[TrainingSequence]) -> dict[str, float]:
        self.seen.extend(batch)
        self.steps += 1
        adv = np.array([b.advantage for b in batch], dtype=np.float64)
        return {
            "loss": 0.0,
            "kl": 0.0,
            "n_sequences": float(len(batch)),
            "mean_abs_advantage": float(np.abs(adv).mean()) if len(adv) else 0.0,
        }

    syncs: int = 0

    def sync_rollout_weights(self) -> None:
        self.syncs += 1

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "null_updater.json").write_text(json.dumps({"steps": self.steps}) + "\n")


@dataclass(slots=True)
class StepReport:
    step: int
    stage: str
    mean_reward: float
    mean_diagnosis: float
    mean_tests: float
    mean_group_std: float
    degenerate_fraction: float
    ceiling_gap: float
    n_sequences: int
    metrics: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"step {self.step:>5} [{self.stage}] R={self.mean_reward:+.3f} "
            f"dx={self.mean_diagnosis:+.3f} tests={self.mean_tests:.2f} "
            f"group_std={self.mean_group_std:.3f} "
            f"degen={self.degenerate_fraction:.1%} gap={self.ceiling_gap:+.3f} "
            f"seqs={self.n_sequences}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "stage": self.stage, "mean_reward": self.mean_reward,
            "mean_diagnosis": self.mean_diagnosis, "mean_tests": self.mean_tests,
            "mean_group_std": self.mean_group_std,
            "degenerate_fraction": self.degenerate_fraction,
            "ceiling_gap": self.ceiling_gap, "n_sequences": self.n_sequences,
            **self.metrics,
        }


def sequences_from_rollouts(
    rollouts: Sequence[Rollout], advantages: npt.NDArray[np.float64]
) -> list[TrainingSequence]:
    """Explode a scored group into per-turn training sequences.

    A rollout with no recorded generations contributes nothing rather than raising: the
    heuristic baselines have no generations, and being able to run the loop against them
    is how the orchestration gets tested without a model.
    """
    out: list[TrainingSequence] = []
    for r, adv in zip(rollouts, advantages, strict=True):
        for gen in r.generations:
            out.append(
                TrainingSequence(
                    messages=tuple(gen["prompt"]),
                    completion=str(gen["completion"]),
                    advantage=float(adv),
                    patient_id=r.patient_id,
                    turn=int(gen["turn"]),
                )
            )
    return out


class GRPOTrainer:
    """The loop. Owns sampling, scoring, monitoring, persistence and the curriculum."""

    def __init__(
        self,
        config: GRPOConfig,
        records: dict[str, PatientRecord],
        splits: Splits,
        policy_factory: Any,
        updater: Updater,
        ctx: RolloutContext | None = None,
        episode_config: EpisodeConfig | None = None,
        curriculum: Curriculum | None = None,
        verify_frozen: bool = True,
    ) -> None:
        # I12, before anything else happens. An unfrozen eval split is not an eval split,
        # and discovering that after a training run means the run cannot be reported.
        if verify_frozen:
            assert_eval_frozen(splits)
        self.config = config
        self.records = records
        self.splits = splits
        self.policy_factory = policy_factory
        self.updater = updater
        self.episode_config = episode_config or load_episode_config()
        self.ctx = ctx or RolloutContext(episode_config=self.episode_config)
        self.curriculum = curriculum or load_curriculum()
        self.stage: Stage = self.curriculum.stages[0]
        # Built once, for every stage, so the run can DECLARE every env config hash it
        # will emit before it emits the first one.
        self._stage_configs = {s.name: self._config_for(s) for s in self.curriculum.stages}

        self.rng = np.random.default_rng(config.seed)
        self.ceiling_monitor = RunningCeilingMonitor()
        self.cost_monitor = CostDistributionMonitor()
        self.degenerate_monitor = DegenerateGroupMonitor()
        self.stage_rewards: list[float] = []
        self.history: list[StepReport] = []
        self.step_index = 0
        self.n_syncs = 0

        assert self.ctx.reward_config is not None
        self.meta = RunMeta(
            run_id=config.run_id,
            env_config_hash=self.episode_config.hash(),
            env_config_hashes=tuple(
                sorted({c.hash() for c in self._stage_configs.values()})
            ),
            reward_config_hash=self.ctx.reward_config.hash(),
            menu_fingerprint=_menu_fingerprint(),
            taxonomy_hash=load_taxonomy().hash(),
            phase="grpo",
            policy=config.model,
            notes={"grpo": config.as_dict()},
        )
        self._store: TrajectoryStore | None = None

    # ------------------------------------------------------------------ sampling --
    def sample_patients(self) -> list[PatientRecord]:
        """Draw this step's patients from TRAIN only, through the guarded loader [I12]."""
        pool = list(self.splits.train)
        idx = self.rng.choice(len(pool), size=min(self.config.patients_per_step, len(pool)),
                              replace=False)
        ids = [pool[int(i)] for i in idx]
        guard_training_access(ids, self.splits)
        return load_training_ids(ids, lambda got: [self.records[p] for p in got])

    def _config_for(self, stage: Stage) -> EpisodeConfig:
        """The stage's horizon, applied to the episode config.

        Rebuilt per stage rather than mutated, because `EpisodeConfig.hash` is pinned into
        every stored trajectory and a mutated config would silently restamp old lines.
        """
        base = self.episode_config
        if base.max_turns == stage.max_turns:
            return base
        return EpisodeConfig(
            max_turns=stage.max_turns,
            dedup_repeat_orders=base.dedup_repeat_orders,
            expose_remaining_budget=base.expose_remaining_budget,
            budget_support=base.budget_support,
            budget_weights=base.budget_weights,
            costs=dict(base.costs),
        )

    def _stage_episode_config(self) -> EpisodeConfig:
        return self._stage_configs[self.stage.name]

    # ---------------------------------------------------------------------- step --
    def run_step(self) -> StepReport:
        cfg = self.config
        stage_cfg = self._stage_episode_config()
        ctx = RolloutContext(
            episode_config=stage_cfg,
            reward_config=self.ctx.reward_config,
            taxonomy=self.ctx.taxonomy,
            catalog=self.ctx.catalog,
            model=self.ctx.model,
        )
        records = self.sample_patients()

        all_rollouts: list[Rollout] = []
        all_sequences: list[TrainingSequence] = []
        group_stds: list[float] = []

        for j, rec in enumerate(records):
            base_seed = int(self.rng.integers(0, 2**31 - 1))
            budget = sample_budget(stage_cfg, np.random.default_rng(base_seed))
            rollouts = rollout_group(
                rec, self.policy_factory, cfg.k, base_seed + j * cfg.k, ctx, budget=budget
            )
            rewards = group_rewards(rollouts)

            for r in rollouts:
                # I9, per episode. The automatic reward-hacking detector: a reward above
                # a perfectly confident correct answer is impossible without information
                # the agent should not have.
                assert_below_ceiling(
                    r.reward, r.hard_ceiling, r.patient_id, r.trajectory,
                    tolerance=cfg.ceiling_tolerance,
                )
                self.ceiling_monitor.update(r.reward, r.expected_ceiling)
                self.cost_monitor.update(
                    r.n_tests, hit_budget_cap=r.trajectory.get("spent", 0.0) >= budget - 1e-9
                )
                if self._store is not None:
                    self._store.append(r.trajectory, r.ground_truth_dict(), step=self.step_index,
                                       stage=self.stage.name, **_slim(r.tags()))

            self.degenerate_monitor.update(rewards)
            group_stds.append(float(np.std(rewards)))
            all_sequences.extend(sequences_from_rollouts(rollouts, group_advantages(rewards)))
            all_rollouts.extend(rollouts)

        metrics = self.updater.update(all_sequences) if all_sequences else {}
        rewards_all = np.array([r.reward for r in all_rollouts], dtype=np.float64)
        report = StepReport(
            step=self.step_index,
            stage=self.stage.name,
            mean_reward=float(rewards_all.mean()),
            mean_diagnosis=float(np.mean([r.breakdown.diagnosis for r in all_rollouts])),
            mean_tests=float(np.mean([r.n_tests for r in all_rollouts])),
            mean_group_std=float(np.mean(group_stds)),
            degenerate_fraction=self.degenerate_monitor.fraction,
            ceiling_gap=float(np.mean([r.expected_ceiling for r in all_rollouts]) -
                              float(rewards_all.mean())),
            n_sequences=len(all_sequences),
            metrics={k: float(v) for k, v in metrics.items()},
        )
        self.stage_rewards.append(report.mean_reward)
        self.history.append(report)
        self.step_index += 1

        if self.step_index % cfg.monitor_every == 0:
            self.assert_monitors()
        if self.step_index % cfg.sync_every == 0:
            # Push the updated adapter to the sampler. Without this the rollouts keep
            # coming from the reference policy while the trained weights drift away from
            # it -- no crash, no error, just a run that is no longer GRPO and a KL term
            # that grows for a reason nobody can find.
            self.updater.sync_rollout_weights()
            self.n_syncs += 1
        self.maybe_advance_stage()
        return report

    def assert_monitors(self) -> None:
        """The every-N-steps checks. Each one halts; none of them warn."""
        self.degenerate_monitor.assert_healthy()
        self.cost_monitor.assert_healthy()
        if self.ceiling_monitor.breached:
            rep = self.ceiling_monitor.report()
            raise AssertionError(
                f"running mean reward {rep['mean_reward']:.4f} exceeds the mean EXPECTED "
                f"ceiling {rep['mean_ceiling']:.4f} over {int(rep['n'])} episodes. A "
                "single lucky rollout may beat the expected ceiling; a running mean may "
                "not. Treat this as a leak until proven otherwise -- run eval/audit.py "
                "before changing anything in the training loop."
            )

    def maybe_advance_stage(self) -> bool:
        """Advance at most one stage, on the criterion, over a window. Never skips."""
        if len(self.stage_rewards) < self.config.stage_window:
            return False
        mean = float(np.mean(self.stage_rewards[-self.config.stage_window :]))
        nxt = self.curriculum.next_stage(self.stage.name, mean)
        if nxt == self.stage.name:
            return False
        self.stage = self.curriculum.stages[self.curriculum.index_of(nxt)]
        self.stage_rewards = []
        return True

    # ----------------------------------------------------------------------- run --
    # ------------------------------------------------------------------ checkpoint --
    STATE_FILE = "trainer_state.json"

    def save_state(self) -> Path:
        """Everything a resumed run needs that is not in the adapter weights.

        On a scheduler with a wall clock -- SLURM, and every HPC allocation -- a long run
        is a chain of shorter jobs, and what is NOT saved here silently resets at each
        boundary. Each of these has a specific consequence if it does:

          rng            the same patients get drawn again in the same order every job,
                         so the run trains on a fraction of the split and reports a mean
                         over a biased sample
          stage/rewards  the curriculum restarts at stage 0, so a policy that had earned
                         a full horizon is put back on a short one
          monitors       the ceiling and collapse detectors refill their windows from
                         empty and cannot fire until half a window has passed -- the
                         detectors are OFF for the first stretch of every job
          step_index     checkpoints and logs overwrite each other

        The trajectory store needs nothing: it is append-only and its lines carry their
        own step number.
        """
        state = {
            "step_index": self.step_index,
            "n_syncs": self.n_syncs,
            "stage": self.stage.name,
            "stage_rewards": list(self.stage_rewards),
            "rng": self.rng.bit_generator.state,
            "ceiling_rewards": list(self.ceiling_monitor.rewards),
            "ceiling_ceilings": list(self.ceiling_monitor.ceilings),
            "cost_counts": list(self.cost_monitor.counts),
            "cost_capped": [bool(b) for b in self.cost_monitor.budget_capped],
            "degenerate_flags": [bool(b) for b in self.degenerate_monitor.flags],
            "config_hashes": sorted(self.meta.declared_env_hashes),
            "reward_config_hash": self.meta.reward_config_hash,
        }
        path = self.config.root / self.config.run_id / self.STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written via a temp file and renamed: a job killed mid-write would otherwise
        # leave a truncated state file, and the next job would fail to parse it and start
        # from zero -- which is the failure this whole mechanism exists to prevent.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str) + "\n")
        tmp.replace(path)
        return path

    def load_state(self) -> bool:
        """Restore from `trainer_state.json`. Returns False if there is nothing to resume.

        Refuses to resume across a reward-config change: the stored monitor windows and
        stage rewards were measured under the old weights, and mixing them with new ones
        produces a mean that is neither.
        """
        path = self.config.root / self.config.run_id / self.STATE_FILE
        if not path.exists():
            return False
        state = json.loads(path.read_text())
        if state.get("reward_config_hash") != self.meta.reward_config_hash:
            raise TrainingError(
                f"cannot resume run {self.config.run_id!r}: it was trained under reward "
                f"config {state.get('reward_config_hash')!r} and this process has "
                f"{self.meta.reward_config_hash!r}. The saved monitor windows and stage "
                "rewards were measured under the old weights. Start a new run_id, or "
                "rescore the stored trajectories instead (scripts/rescore.py)."
            )
        self.step_index = int(state["step_index"])
        self.n_syncs = int(state.get("n_syncs", 0))
        self.stage = self.curriculum.stages[self.curriculum.index_of(state["stage"])]
        self.stage_rewards = [float(x) for x in state["stage_rewards"]]
        self.rng.bit_generator.state = state["rng"]
        self.ceiling_monitor.rewards = deque(state["ceiling_rewards"])
        self.ceiling_monitor.ceilings = deque(state["ceiling_ceilings"])
        self.cost_monitor.counts = deque(int(c) for c in state["cost_counts"])
        self.cost_monitor.budget_capped = deque(bool(b) for b in state["cost_capped"])
        self.degenerate_monitor.flags = deque(bool(b) for b in state["degenerate_flags"])
        return True

    def run(self, steps: int | None = None) -> list[StepReport]:
        n = steps if steps is not None else self.config.max_steps
        store = TrajectoryStore(self.meta, root=self.config.root)
        with store as s:
            self._store = s
            log = (self.config.root / self.config.run_id / "steps.jsonl").open("a")
            try:
                for _ in range(n):
                    report = self.run_step()
                    log.write(json.dumps(report.as_dict(), sort_keys=True) + "\n")
                    log.flush()
                    if self.step_index % self.config.save_every == 0:
                        self.updater.save(
                            self.config.root / self.config.run_id / f"step-{self.step_index}"
                        )
                        self.save_state()
            finally:
                # In `finally`, so a wall-clock kill, an OOM or a halted monitor still
                # leaves a resumable checkpoint. Losing the last few steps is cheap;
                # losing the curriculum stage and the monitor windows is not.
                self.save_state()
                log.close()
                self._store = None
        return self.history


def _menu_fingerprint() -> str:
    from dxenv.env.actions import build_menu

    return build_menu().fingerprint()


def _slim(tags: dict[str, Any]) -> dict[str, Any]:
    """Drop the generations from the stored tags.

    They are large, they are already recoverable from the SFT/rollout artifacts, and
    writing them on every line turns a 200MB run into a 20GB one.
    """
    return {k: v for k, v in tags.items() if k != "generations"}


# ------------------------------------------------------------------ torch updater ----


@dataclass(slots=True)
class TorchLoRAUpdater:  # pragma: no cover - CUDA only
    """The real gradient step: LoRA + clipped surrogate + k3 KL to the SFT reference.

    Imported lazily so that everything above runs, and is tested, on a laptop.
    """

    config: GRPOConfig
    backend: Any = None
    """The rollout sampler, so trained weights can be pushed back to it every sync."""

    _model: Any = None
    _tok: Any = None
    _opt: Any = None

    def _lazy(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise TrainingError(
                "GRPO needs the GPU extra: pip install -e '.[gpu]' on a CUDA host. The "
                "loop itself runs against NullUpdater anywhere."
            ) from exc

        self._tok = AutoTokenizer.from_pretrained(self.config.model)
        base: Any = AutoModelForCausalLM.from_pretrained(
            self.config.model, dtype=torch.bfloat16, device_map="cuda"
        )
        if self.config.reference_adapter is not None:
            # MERGED, not held as an adapter. The KL is measured against the policy GRPO
            # started from; an unmerged adapter would leave the reference as the base
            # model, and the run would be free to drift away from SFT for nothing.
            base = PeftModel.from_pretrained(base, str(self.config.reference_adapter))
            base = base.merge_and_unload()
        # There is NO separate reference model. `get_peft_model` injects the LoRA layers
        # into `base` IN PLACE, so holding `self._ref = base` aliases the very modules
        # the adapter now lives in -- the reference forward pass would run with the
        # trainable adapter active, KL would be identically zero for the whole run, and
        # nothing would say so. Reference logprobs come from `disable_adapter()` instead,
        # which is also why this fits alongside a vLLM engine: one copy of the weights,
        # not two.
        self._model = get_peft_model(
            base,
            LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            ),
        )
        self._opt = torch.optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

    def _completion_logprobs(self, model: Any, batch: Sequence[TrainingSequence]) -> Any:
        import torch

        texts, starts = [], []
        for b in batch:
            prompt = self._tok.apply_chat_template(
                list(b.messages), tokenize=False, add_generation_prompt=True
            )
            starts.append(len(self._tok(prompt).input_ids))
            texts.append(prompt + b.completion)
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        logits = model(**enc).logits[:, :-1]
        labels = enc.input_ids[:, 1:]
        logp = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        # Mask the prompt: gradient on the completion only. Training on the prompt spends
        # most of it reproducing the 149-label menu, which is fixed text.
        mask = torch.zeros_like(logp)
        for i, s in enumerate(starts):
            mask[i, max(s - 1, 0) :] = enc.attention_mask[i, 1:][max(s - 1, 0) :]
        return logp, mask

    def update(self, batch: Sequence[TrainingSequence]) -> dict[str, float]:
        import torch

        self._lazy()
        adv = torch.tensor([b.advantage for b in batch], device="cuda").unsqueeze(-1)
        logp, mask = self._completion_logprobs(self._model, batch)
        with torch.no_grad(), self._model.disable_adapter():
            ref_logp, _ = self._completion_logprobs(self._model, batch)
        old_logp = logp.detach()

        # One inner epoch, so `old_logp` is this batch's own detached logprobs and the
        # ratio is identically 1 -- the clipping is INERT and this reduces to a plain
        # policy gradient. That is correct single-epoch GRPO, and it is written out
        # because the clip_eps knob otherwise looks like it is doing something. It starts
        # doing something the moment a second inner epoch is added.
        ratio = torch.exp(logp - old_logp)
        surrogate = torch.min(
            ratio * adv,
            torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv,
        )
        # k3: exp(r) - r - 1, r = ref - policy. Non-negative per token, unlike the naive
        # difference, which occasionally pays the policy for leaving the reference.
        r = ref_logp - logp
        kl = torch.exp(r) - r - 1.0
        denom = mask.sum().clamp(min=1.0)
        loss = -((surrogate - self.config.kl_coef * kl) * mask).sum() / denom

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._model.parameters() if p.requires_grad], 1.0
        )
        self._opt.step()
        return {
            "loss": float(loss.item()),
            "kl": float(((kl * mask).sum() / denom).item()),
            "n_sequences": float(len(batch)),
        }

    def sync_rollout_weights(self) -> None:
        """Save the adapter and tell the sampler to reload it under a new id.

        Writing the file is not enough on its own: vLLM caches an adapter by id, so a
        reload that reuses the id keeps serving the old weights silently. `backend` is
        wired in by the training script; leaving it None makes the save a no-op push and
        is logged as such rather than passing quietly.
        """
        # Checked BEFORE save(), which triggers the 7B load. A misconfigured sync should
        # cost a second, not two minutes of loading weights it is about to not use.
        if self.backend is None:
            raise TrainingError(
                "TorchLoRAUpdater has no backend to sync to, so the rollout sampler would "
                "keep serving the reference policy for the whole run. Pass the "
                "VLLMBackend when constructing the updater."
            )
        path = self.config.root / self.config.run_id / "current"
        self.save(path)
        self.backend.reload_lora(str(path))

    def save(self, path: Path) -> None:
        self._lazy()
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
