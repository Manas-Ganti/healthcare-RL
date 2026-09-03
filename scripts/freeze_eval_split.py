"""Freeze the eval split. Run ONCE, commit `dxenv/data/eval_split.json`, never again.

I12: the eval split is frozen and hash-verified before any training run, and training
never reads it. `train/grpo.py` calls `assert_eval_frozen` at startup and refuses to
begin without this file.

Re-running this after training has started invalidates every number ever measured
against the old split, so the script refuses to overwrite an existing file unless
`--force` is passed, and says why.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dxenv.data.corpus import generate_corpus
from dxenv.data.splits import freeze_eval_split, make_splits

FROZEN = Path("dxenv/data/eval_split.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20_000, help="corpus size to cut the split from")
    ap.add_argument("--corpus-seed", type=int, default=20260901)
    ap.add_argument("--split-seed", type=int, default=11)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if FROZEN.exists() and not args.force:
        raise SystemExit(
            f"{FROZEN} already exists. Re-freezing makes every number measured against "
            "the old split incomparable with every number measured against the new one. "
            "Pass --force only if you have decided to discard the old results."
        )

    records = generate_corpus(args.n, seed=args.corpus_seed)
    splits = make_splits(records, seed=args.split_seed)
    digest = freeze_eval_split(
        splits,
        FROZEN,
        provenance={
            "corpus_n": args.n,
            "corpus_seed": args.corpus_seed,
            "split_seed": args.split_seed,
            "generator": "dxenv.data.corpus.generate_corpus",
        },
    )
    print(json.dumps({"eval_hash": digest, **splits.summary()}, indent=2))
    print(f"\nWrote {FROZEN}. Commit it now.")


if __name__ == "__main__":
    main()
