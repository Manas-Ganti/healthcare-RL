"""Frozen train / eval / held-out-module splits, with a hash that is verified [I12].

The eval split is frozen and hash-verified before any training run, and training never
reads it. "Never reads it" is enforced by `guard_training_access`, which raises if
training-side code touches eval ids -- a convention that is merely documented gets
violated the first time someone debugs a training loop at 2am.

The held-out-module split exists to measure generalisation across Synthea's disease
modules (here, organ systems): a policy that has memorised the cardiology module's
signature is a different thing from one that can diagnose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from dxenv.data.corpus import PatientRecord
from dxenv.data.taxonomy import load_taxonomy

_FROZEN_PATH: Final = Path(__file__).with_name("eval_split.json")

DEFAULT_EVAL_FRACTION: Final = 0.2
DEFAULT_HOLDOUT_SYSTEMS: Final = ("rheumatologic", "obstetric")
"""Systems withheld entirely from training, to measure the generalisation gap.

Chosen as one mid-frequency system and one small one: withholding a large system would
distort the training prior badly enough that the gap would measure the distortion rather
than the generalisation.
"""


class SplitError(ValueError):
    """A split is malformed, or training touched eval. Never caught."""


class EvalLeakError(AssertionError):
    """Training-side code reached the eval split [I12]. Always a bug."""


@dataclass(frozen=True, slots=True)
class Splits:
    train: tuple[str, ...]
    eval: tuple[str, ...]
    holdout_modules: tuple[str, ...]
    holdout_systems: tuple[str, ...]

    def __post_init__(self) -> None:
        pairs = (
            ("train/eval", set(self.train) & set(self.eval)),
            ("train/holdout", set(self.train) & set(self.holdout_modules)),
            ("eval/holdout", set(self.eval) & set(self.holdout_modules)),
        )
        for name, overlap in pairs:
            if overlap:
                raise SplitError(f"{name} splits overlap on {len(overlap)} patients")

    def eval_hash(self) -> str:
        blob = json.dumps(sorted(self.eval), separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "eval": len(self.eval),
            "holdout_modules": len(self.holdout_modules),
        }


def make_splits(
    records: list[PatientRecord],
    seed: int,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    holdout_systems: tuple[str, ...] = DEFAULT_HOLDOUT_SYSTEMS,
) -> Splits:
    """Deterministic from (records, seed). Held-out systems are removed FIRST.

    Order matters: partitioning before removing the held-out systems would leave those
    patients in train, and the generalisation gap would then be measured against data the
    policy had seen.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise SplitError(f"eval_fraction must be in (0, 1), got {eval_fraction}")

    tax = load_taxonomy()
    held: list[PatientRecord] = []
    rest: list[PatientRecord] = []
    for rec in records:
        (held if tax.get(rec.condition).system in holdout_systems else rest).append(rec)

    ids = np.array(sorted(r.patient_id for r in rest))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_eval = max(1, round(len(ids) * eval_fraction))
    eval_ids = tuple(sorted(ids[perm[:n_eval]].tolist()))
    train_ids = tuple(sorted(ids[perm[n_eval:]].tolist()))

    return Splits(
        train=train_ids,
        eval=eval_ids,
        holdout_modules=tuple(sorted(r.patient_id for r in held)),
        holdout_systems=holdout_systems,
    )


def freeze_eval_split(splits: Splits, path: Path | None = None) -> str:
    """Write the eval hash to disk. Call ONCE, then commit the file."""
    p = path or _FROZEN_PATH
    digest = splits.eval_hash()
    p.write_text(
        json.dumps(
            {
                "eval_hash": digest,
                "n_eval": len(splits.eval),
                "holdout_systems": list(splits.holdout_systems),
            },
            indent=2,
        )
        + "\n"
    )
    return digest


def assert_eval_frozen(splits: Splits, path: Path | None = None) -> None:
    """Verify the eval split against the committed hash. Run before ANY training run."""
    p = path or _FROZEN_PATH
    if not p.exists():
        raise SplitError(
            f"{p} does not exist. Freeze the eval split and commit it before training; "
            "an unfrozen eval split is not an eval split."
        )
    frozen = json.loads(p.read_text())
    actual = splits.eval_hash()
    if actual != frozen["eval_hash"]:
        raise SplitError(
            f"eval split drifted: {actual} != frozen {frozen['eval_hash']}. Every number "
            "measured against the new split is incomparable with every number measured "
            "against the old one."
        )


def guard_training_access(requested_ids: list[str], splits: Splits) -> None:
    """Raise if training-side code requested any eval patient [I12].

    Called by the training data loader. Enforcement beats convention: a rule that lives
    only in a docstring gets broken the first time someone debugs a training loop at 2am.
    """
    overlap = set(requested_ids) & set(splits.eval)
    if overlap:
        raise EvalLeakError(
            f"training requested {len(overlap)} eval patients, e.g. {sorted(overlap)[:3]}. "
            "The eval split is frozen and training never reads it [I12]."
        )
