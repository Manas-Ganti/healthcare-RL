"""Rescore a stored run under the current reward config.

Reward is a pure function of (trajectory, ground_truth, reward_config) [I8], so this is
free -- which is the whole reason `data/store.py` exists. Regenerating the rollouts is
not free, and on a 7B policy it is the dominant cost of the project.

The scoring function is INJECTED into the store rather than imported by it: `dxenv.reward`
imports `dxenv.data`, so a store that reached back for the reward engine would close an
import cycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from dxenv.data.corpus import generate_corpus
from dxenv.data.store import read_episodes, read_meta
from dxenv.reward.engine import GroundTruth, load_reward_config, score_trajectory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="runs/<run_id>")
    ap.add_argument("--config", type=Path, default=None, help="a different reward.yaml")
    ap.add_argument("--corpus-n", type=int, default=None,
                    help="regenerate this many patients to recover analytes and allergies")
    ap.add_argument("--corpus-seed", type=int, default=None)
    args = ap.parse_args()

    meta = read_meta(args.run)
    cfg = load_reward_config(args.config)
    print(f"run {meta['run_id']} was scored under reward config {meta['reward_config_hash']}")
    print(f"rescoring under                                    {cfg.hash()}")

    # The stored line carries the condition; analytes and allergies are needed for the
    # treatment terms and are recovered from the corpus by provenance rather than stored
    # per line, where they would dominate the file size.
    analytes: dict[str, Any] = {}
    if args.corpus_n is not None and args.corpus_seed is not None:
        analytes = {
            r.patient_id: (r.analytes, r.allergies)
            for r in generate_corpus(args.corpus_n, seed=args.corpus_seed)
        }

    old, new = [], []
    for ep in read_episodes(args.run / "episodes.jsonl"):
        pid = ep.ground_truth["patient_id"]
        a, allergies = analytes.get(pid, ({}, ()))
        b = score_trajectory(
            ep.trajectory, GroundTruth(ep.ground_truth["condition"], a, allergies), cfg
        )
        new.append(b.total)
        if "reward" in ep.tags:
            old.append(float(ep.tags["reward"]))

    print(json.dumps({
        "n": len(new),
        "mean_rescored": float(np.mean(new)),
        "mean_as_stored": float(np.mean(old)) if old else None,
        "mean_delta": float(np.mean(new) - np.mean(old)) if old else None,
    }, indent=2))


if __name__ == "__main__":
    main()
