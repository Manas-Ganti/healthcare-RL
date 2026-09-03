"""Phase 4 entry point.

Startup order matters and is not negotiable:

  1. verify the eval split against its committed hash [I12] -- an unfrozen eval split is
     not an eval split, and finding that out after a run means the run cannot be reported;
  2. build the record index from the SAME provenance the split was cut from, so the ids
     in the split resolve to the patients the split meant;
  3. only then construct the trainer, which routes every data read through the guarded
     loader.

`--dry-run` runs the whole loop against `NullUpdater`: real rollouts, real rewards, real
monitors, no gradient. It is the cheapest way to find out that a config would have halted
on step 3, and it costs minutes rather than a node-hour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dxenv.data.corpus import generate_corpus
from dxenv.data.splits import load_frozen_provenance, rebuild_frozen_splits
from dxenv.policy.llm import LLMPolicy, RandomBackend, VLLMBackend
from dxenv.train.grpo import GRPOConfig, GRPOTrainer, NullUpdater, TorchLoRAUpdater


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="grpo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--reference-adapter", type=Path, default=None,
                    help="the SFT LoRA the KL term is measured against")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--patients-per-step", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--dry-run", action="store_true",
                    help="full loop, real rewards and monitors, no gradient step")
    args = ap.parse_args()

    splits = rebuild_frozen_splits()          # (1) and its verification
    prov = load_frozen_provenance()
    corpus = generate_corpus(int(prov["corpus_n"]), seed=int(prov["corpus_seed"]))  # (2)
    records = {r.patient_id: r for r in corpus}
    print(f"eval split verified against its committed hash; {json.dumps(splits.summary())}")

    cfg = GRPOConfig(
        run_id=args.run_id, model=args.model, reference_adapter=args.reference_adapter,
        k=args.k, patients_per_step=args.patients_per_step, max_steps=args.steps,
        temperature=args.temperature, learning_rate=args.lr, kl_coef=args.kl_coef,
        seed=args.seed, root=args.root,
    )

    if args.dry_run:
        updater = NullUpdater()
        def factory(seed: int) -> LLMPolicy:
            return LLMPolicy(backend=RandomBackend(seed=seed), seed=seed)
    else:
        updater = TorchLoRAUpdater(config=cfg)  # type: ignore[assignment]
        backend = VLLMBackend(
            model=args.model,
            lora_path=str(args.reference_adapter) if args.reference_adapter else None,
        )
        def factory(seed: int) -> LLMPolicy:
            return LLMPolicy(backend=backend, temperature=args.temperature, seed=seed)

    trainer = GRPOTrainer(cfg, records, splits, factory, updater)  # (3)
    for report in trainer.run(steps=args.steps):
        print(report.line())


if __name__ == "__main__":
    main()
