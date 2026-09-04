"""CLAUDE.md 8.1: the check that decides whether Phase 3 needs SFT at all.

> Before building anything else in this phase: run the prompted base model with
> constrained decoding on 200 patients. If it clears the blank-record floor with
> reasonable spread, you may not need SFT at all.

Reports, on the same patients and under the same reward config:

  prior          the blank-record floor. Everything is reported ABOVE this, not above 0.
  vitals_bayes   every free observation used optimally, nothing ordered. The Gate B bar.
  greedy_bayes   a strong heuristic reference.
  random_schema  the grammar with no policy behind it -- format-valid and uninformed.
  prompted       the base model under guided decoding, if --model is given.

`random_schema` is the row that matters when reading a weak `prompted` number: a prompted
model at that level is not doing anything a constrained sampler is not.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dxenv.data.corpus import generate_corpus
from dxenv.data.store import RunMeta, TrajectoryStore
from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import build_menu
from dxenv.env.episode import load_episode_config
from dxenv.policy.baselines import GreedyBayesPolicy, PriorPolicy, VitalsOnlyPolicy
from dxenv.policy.decoding import action_json_schema, schema_fingerprint
from dxenv.policy.llm import LLMPolicy, RandomBackend, VLLMBackend
from dxenv.policy.rollout import RolloutContext, constant_factory, rollout_group


def calibration_margin(rollouts: Sequence[Any], taxonomy: Taxonomy, score_fn: Any) -> float:
    """Reported-distribution score minus the same distribution collapsed onto its argmax.

    The Gate B calibration criterion, and the one that catches a Phase 3 that trained
    confidence instead of belief. A model that has learned to say 0.99 scores BETTER
    collapsed, because collapsing a near-one-hot report costs nothing and gains the last
    sliver of mass; a calibrated model scores worse collapsed, because it was holding
    real uncertainty that the collapse throws away.

    Positive is the healthy sign. Reported here rather than as an accuracy, because
    accuracy cannot distinguish the two cases at all.
    """
    margins = []
    for r in rollouts:
        declared = next(
            (s["action"]["distribution"] for s in reversed(r.trajectory["steps"])
             if s["action"]["kind"] == "diagnose"),
            None,
        )
        if not declared:
            continue  # abstained or ran out of turns: there is no report to calibrate
        vec = np.zeros(len(taxonomy), dtype=np.float64)
        for slug, p in declared.items():
            vec[taxonomy.index(slug)] = float(p)
        total = float(vec.sum())
        if total <= 0.0:
            continue
        vec /= total
        collapsed = np.zeros_like(vec)
        collapsed[int(np.argmax(vec))] = 1.0
        idx = taxonomy.index(r.condition)
        margins.append(score_fn(vec, idx) - score_fn(collapsed, idx))
    return float(np.mean(margins)) if margins else 0.0


def evaluate(
    name: str, factory: Any, records: list[Any], k: int, ctx: RolloutContext,
    store: TrajectoryStore, seed: int,
) -> dict[str, Any]:
    per_patient_best, all_rewards, group_stds, tests, first_sample = [], [], [], [], []
    ceilings, everything, schema_valid = [], [], []
    # Progress, because this runs for hours and prints nothing otherwise: vLLM's own
    # progress bar is off (use_tqdm=False, since one bar per batched call would be noise),
    # so without this a live job is indistinguishable from a hung one.
    started = time.monotonic()
    for i, rec in enumerate(records):
        rollouts = rollout_group(rec, factory, k, seed + i * k, ctx)
        if (i + 1) % max(5, len(records) // 20) == 0 or i + 1 == len(records):
            done = i + 1
            elapsed = time.monotonic() - started
            rate = elapsed / done
            print(
                f"  [{name}] {done}/{len(records)} patients · "
                f"{elapsed / 60:.1f} min elapsed · "
                f"{rate * (len(records) - done) / 60:.1f} min remaining · "
                f"mean R {np.mean(all_rewards + [r.reward for r in rollouts]):+.3f}",
                flush=True,
            )
        rewards = [r.reward for r in rollouts]
        for r in rollouts:
            store.append(r.trajectory, r.ground_truth_dict(), policy=name,
                         **{kk: vv for kk, vv in r.tags().items() if kk != "generations"})
            # Under guided decoding every generation parses by construction; this counts
            # it anyway, because "by construction" is a claim about the backend and the
            # gate is where that claim gets checked rather than assumed.
            schema_valid.extend(
                _parses(g["completion"], ctx) for g in r.generations
            )
        per_patient_best.append(max(rewards))
        first_sample.append(rewards[0])
        all_rewards.extend(rewards)
        group_stds.append(float(np.std(rewards)))
        tests.extend(r.n_tests for r in rollouts)
        ceilings.append(rollouts[0].expected_ceiling)
        everything.extend(rollouts)
    return {
        "policy": name,
        "mean_reward": float(np.mean(all_rewards)),
        "mean_best_of_k": float(np.mean(per_patient_best)),
        "mean_first_sample": float(np.mean(first_sample)),
        "mean_group_std": float(np.mean(group_stds)),
        "mean_tests": float(np.mean(tests)),
        "mean_expected_ceiling": float(np.mean(ceilings)),
        "calibration_margin": calibration_margin(everything, ctx.taxonomy, ctx.score_fn),
        "schema_valid_fraction": float(np.mean(schema_valid)) if schema_valid else 1.0,
        "per_patient_best": per_patient_best,
        "per_patient_first": first_sample,
        "group_stds": group_stds,
    }


def _parses(completion: str, ctx: RolloutContext) -> bool:
    from dxenv.policy.decoding import DecodingError, parse_action

    try:
        parse_action(completion, build_menu(), ctx.taxonomy)
    except DecodingError:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", default=None, help="HF id; omit to skip the model row")
    ap.add_argument("--lora", type=Path, default=None,
                    help="SFT adapter. With it the model row is named `sft` rather than "
                         "`prompted`, because the two runs answer different questions.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                    help="0.85 suits this standalone sweep, where nothing shares the "
                         "card. VLLMBackend defaults lower for the co-located GRPO run.")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to prompted_baseline.json, or sft_baseline.json "
                         "with --lora, so the pre-SFT measurement is not overwritten")
    args = ap.parse_args()

    # The row is named for WHICH question this run answers. Before SFT: "can the base
    # model do this at all, and is SFT even needed" (CLAUDE.md 8.1). After SFT: "did SFT
    # help without destroying the calibration and the diversity GRPO needs" -- which is
    # Gate B proper, the go/no-go into Phase 4. Same script, same thresholds, two
    # different decisions, and the results must not overwrite each other.
    subject_name = "sft" if args.lora else "prompted"
    # A UNIQUE run id per invocation. The store is append-only, so a fixed id makes every
    # attempt -- including the five that crashed while the vLLM path was being fixed --
    # accumulate into one file: 13,536 lines where this run contributes 3,800. That breaks
    # progress counting, and worse, it silently mixes episodes generated under different
    # GRAMMARS, since the schema changed between those attempts. Offline rescoring would
    # then average across incompatible decoders without saying so.
    run_tag = os.environ.get("SLURM_JOB_ID") or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = args.out or Path(f"runs/phase3/{'sft' if args.lora else 'prompted'}_baseline.json")

    records = generate_corpus(args.n, seed=args.seed)
    ctx = RolloutContext()
    assert ctx.reward_config is not None
    meta = RunMeta(
        run_id=f"phase3_{subject_name}-{run_tag}",
        env_config_hash=load_episode_config().hash(),
        reward_config_hash=ctx.reward_config.hash(),
        menu_fingerprint=build_menu().fingerprint(),
        taxonomy_hash=load_taxonomy().hash(),
        phase=f"phase3_{subject_name}_baseline",
        policy="mixed",
        notes={
            "n": args.n, "k": args.k, "seed": args.seed,
            # The grammar is as much a part of how a trajectory was produced as the
            # reward weights are of how it was scored. Without this, two runs under
            # different schemas are indistinguishable in the store.
            "grammar_fingerprint": schema_fingerprint(action_json_schema()),
        },
    )

    rows: list[dict[str, Any]] = []
    # DXENV_RUNS, not a hardcoded "runs": slurm/env.sh points it at scratch precisely
    # because a sweep writes hundreds of megabytes of JSONL, and home has a quota.
    runs_root = Path(os.environ.get("DXENV_RUNS", "runs"))
    print(f"trajectory store: {runs_root / meta.run_id}")
    with TrajectoryStore(meta, root=runs_root) as store:
        # k=1 for the deterministic policies: k identical samples would report a group
        # std of 0 and invite the reader to conclude something about spread.
        rows.append(evaluate("prior", constant_factory(PriorPolicy), records, 1, ctx, store, 1))
        rows.append(evaluate("vitals_bayes", constant_factory(VitalsOnlyPolicy), records, 1,
                             ctx, store, 2))
        rows.append(evaluate("greedy_bayes", constant_factory(GreedyBayesPolicy), records, 1,
                             ctx, store, 3))
        rows.append(evaluate(
            "random_schema",
            lambda s: LLMPolicy(backend=RandomBackend(seed=s), seed=s),
            records, args.k, ctx, store, 4,
        ))
        if args.model:
            backend = VLLMBackend(
                model=args.model,
                lora_path=str(args.lora) if args.lora else None,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            rows.append(evaluate(
                subject_name,
                lambda s: LLMPolicy(backend=backend, temperature=args.temperature, seed=s),
                records, args.k, ctx, store, 5,
            ))

    floor = next(r for r in rows if r["policy"] == "prior")["mean_reward"]
    bar = next(r for r in rows if r["policy"] == "vitals_bayes")["mean_reward"]
    subject = next((r for r in rows if r["policy"] == subject_name),
                   next(r for r in rows if r["policy"] == "random_schema"))
    payload = {
        "n": args.n, "k": args.k, "seed": args.seed, "temperature": args.temperature,
        "model": args.model,
        "lora": str(args.lora) if args.lora else None,
        "blank_record_floor": floor,
        "gate_b_pass_bar": bar,
        # Hoisted from the subject row so the gate checker reads one place. The subject is
        # the prompted model when there is one, and the grammar sampler otherwise -- which
        # measures the floor a prompted model has to beat, not a policy.
        "subject_policy": subject["policy"],
        "calibration_margin": subject["calibration_margin"],
        "mean_expected_ceiling": subject["mean_expected_ceiling"],
        "schema_valid_fraction": subject["schema_valid_fraction"],
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'policy':<16}{'mean R':>10}{'best@k':>10}{'grp std':>10}{'tests':>8}")
    for r in rows:
        print(f"{r['policy']:<16}{r['mean_reward']:>+10.3f}{r['mean_best_of_k']:>+10.3f}"
              f"{r['mean_group_std']:>10.3f}{r['mean_tests']:>8.2f}")
    print(f"\nblank-record floor {floor:+.3f}; Gate B pass bar (vitals-only Bayes) {bar:+.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
