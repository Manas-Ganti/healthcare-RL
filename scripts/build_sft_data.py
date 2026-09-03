"""Phase 3 data pipeline: teacher -> de-leak -> filter -> rejection-sample -> SFT set.

Every stage reports what it dropped and why. A pipeline that reports "kept 4,182 of
9,000" and nothing else cannot be debugged; the failure it is hiding is almost always
that one filter is doing all the work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dxenv.data.corpus import generate_corpus
from dxenv.data.splits import rebuild_frozen_splits
from dxenv.policy.baselines import GreedyBayesPolicy
from dxenv.policy.rejection import (
    RejectionConfig,
    RejectionStats,
    balance_conditions,
    filter_group,
    judge,
)
from dxenv.policy.rollout import RolloutContext, constant_factory, rollout_group
from dxenv.policy.sft import SFTDataset, build_examples, seed_abstentions
from dxenv.policy.teacher import (
    PrivilegedTeacher,
    audit_trace,
    deleak,
    deleak_ablation,
    filter_traces,
    privileged_trace,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="patients to run the teacher over")
    ap.add_argument("--k", type=int, default=8, help="rejection-sampling group size")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--abstain-fraction", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("runs/phase3/sft.jsonl"))
    ap.add_argument("--no-rejection", action="store_true",
                    help="teacher traces only; skip the sampled-rollout arm")
    ap.add_argument("--frozen-split", action="store_true",
                    help="draw from the committed train split instead of a fresh corpus")
    args = ap.parse_args()

    if args.frozen_split:
        splits = rebuild_frozen_splits()
        pool = set(splits.train)
        prov = json.loads(Path("dxenv/data/eval_split.json").read_text())["provenance"]
        corpus = generate_corpus(int(prov["corpus_n"]), seed=int(prov["corpus_seed"]))
        records = [r for r in corpus if r.patient_id in pool][: args.n]
    else:
        records = generate_corpus(args.n, seed=args.seed)

    # ---- teacher, then strip the privilege -------------------------------------
    teacher = PrivilegedTeacher()
    privileged = [
        privileged_trace(r, seed=args.seed + i, teacher=teacher, budget=args.budget)
        for i, r in enumerate(records)
    ]
    leaks_before = sum(len(audit_trace(t)) for t in privileged)
    deleaked = [deleak(t) for t in privileged]
    clean, rejected = filter_traces(deleaked)
    ablation = deleak_ablation(clean, np.random.default_rng(args.seed))

    print(f"teacher: {len(privileged)} traces")
    print(f"  literal leak findings BEFORE de-leaking: {leaks_before} "
          f"(positive control -- must be > 0, or the teacher is not privileged)")
    print(f"  grounding filter: kept {len(clean)}, rejected {len(rejected)}")
    for pid, findings in rejected[:5]:
        print(f"    {pid}: {findings[0].line()[:120]}")
    print(f"  {ablation.line()}")

    # ---- rejection sampling over sampled rollouts -------------------------------
    accepted = []
    stats = RejectionStats()
    if not args.no_rejection:
        ctx = RolloutContext()
        cfg = RejectionConfig()
        for i, rec in enumerate(records):
            rollouts = rollout_group(
                rec, constant_factory(GreedyBayesPolicy), args.k,
                args.seed + i * args.k, ctx, budget=args.budget,
            )
            for r in rollouts:
                stats.note(judge(r, cfg))
            decision = filter_group(rollouts, cfg)
            stats.groups += 1
            stats.groups_reproducible += int(decision.reproducible)
            accepted.extend(decision.accepted)
        print(stats.render())
        balanced, report = balance_conditions(accepted, np.random.default_rng(args.seed))
        print(f"  condition balance: {len(balanced)} kept, {report.dropped} dropped to the "
              f"cap of {report.target}/condition; imbalance {report.imbalance:.2f}x")

    # ---- assemble ---------------------------------------------------------------
    dataset = SFTDataset(
        build_examples(clean)
        + seed_abstentions(clean, fraction=args.abstain_fraction)
    )
    dataset.validate()
    n = dataset.write_jsonl(args.out)
    print(f"\nSFT set: {json.dumps(dataset.summary(), sort_keys=True)}")
    print(f"wrote {n} examples to {args.out}")


if __name__ == "__main__":
    main()
