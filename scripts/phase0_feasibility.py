"""Phase 0 -- feasibility. Does a learnable problem exist?

Deliberately a throwaway harness (CLAUDE.md 5): a signal detector, not the agent. It is
allowed to be ugly. It is not allowed to be skipped, and it is not allowed to run before
its thresholds are committed.

Three probes:
    F  blank record, no observation at all
    V  demographics, vitals, presenting complaint; no test results
    T  V plus every analyte

Plus a LEAKAGE POSITIVE CONTROL. In this repo the blocked resources never reach the
feature matrix, so the standard "strip them and check accuracy barely moves" ablation
passes trivially -- which tells you nothing about whether it would catch a real leak. So
we also run the probe WITH the leaky fields included, and require that accuracy jumps.
An ablation that cannot detect a leak it was handed is not evidence of no leak.

    python scripts/phase0_feasibility.py --n 10000 --seed 7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dxenv.data.corpus import PatientRecord, generate_corpus
from dxenv.data.taxonomy import load_taxonomy
from dxenv.env.catalog import CategoricalAnalyte, load_catalog
from dxenv.reward.scoring import brier_score, severity_weight

RESULTS_PATH = Path("runs/phase0/results.json")
GATE_PATH = Path("dxenv/configs/gate_a.yaml")


@dataclass
class ProbeResult:
    name: str
    balanced_accuracy: float
    top1_accuracy: float
    top5_accuracy: float
    mean_weighted_brier: float
    n_features: int


def _encode(records: list[PatientRecord], keys: list[str], leaky: bool = False):
    """Feature matrix plus the column roles.

    Categorical analytes are returned as CODES and one-hot encoded downstream, never as
    ordinals. The first version of this harness ordinal-encoded them and the probe came
    out at chance -- it was measuring the encoding, not the environment. An index into
    an unordered vocabulary is not a number.
    """
    cat = load_catalog()
    tax = load_taxonomy()
    numeric_idx: list[int] = [0, 1]
    categorical_idx: list[int] = []
    rows: list[list[float]] = []
    for rec in records:
        row: list[float] = [float(rec.age_years), 1.0 if rec.sex == "female" else 0.0]
        for k in keys:
            a = cat.analyte(k)
            v = rec.analytes[k]
            if isinstance(a, CategoricalAnalyte):
                row.append(float(a.values.index(str(v))))
            else:
                row.append(float(v))
        if leaky:
            # Positive control: what a Condition resource contributes if the allowlist
            # ever lets one through.
            row.append(float(tax.index(rec.condition)))
        rows.append(row)
    for j, k in enumerate(keys, start=2):
        (categorical_idx if isinstance(cat.analyte(k), CategoricalAnalyte)
         else numeric_idx).append(j)
    if leaky:
        categorical_idx.append(2 + len(keys))
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray([tax.index(r.condition) for r in records], dtype=np.int64)
    return x, y, numeric_idx, categorical_idx


def _run_probe(name: str, x: np.ndarray, y: np.ndarray, seed: int,
               numeric_idx: list[int], categorical_idx: list[int],
               blank: bool = False) -> ProbeResult:
    tax = load_taxonomy()
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=seed)
    model: Any
    if blank:
        model = DummyClassifier(strategy="prior")
    else:
        pre = ColumnTransformer([
            ("num", StandardScaler(), numeric_idx),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_idx),
        ])
        model = make_pipeline(pre, LogisticRegression(max_iter=1500))
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    proba = model.predict_proba(xte)

    classes = list(model.classes_)
    full = np.zeros((len(yte), len(tax)), dtype=np.float64)
    for j, c in enumerate(classes):
        full[:, int(c)] = proba[:, j]
    full = np.clip(full, 1e-12, None)
    full /= full.sum(axis=1, keepdims=True)

    briers = [
        brier_score(full[i], int(yte[i])) * severity_weight(tax.slugs[int(yte[i])])
        for i in range(len(yte))
    ]
    order = np.argsort(-full, axis=1)
    top5 = float(np.mean([yte[i] in order[i, :5] for i in range(len(yte))]))
    return ProbeResult(
        name=name,
        balanced_accuracy=float(balanced_accuracy_score(yte, pred)),
        top1_accuracy=float(np.mean(pred == yte)),
        top5_accuracy=top5,
        mean_weighted_brier=float(np.mean(briers)),
        n_features=int(x.shape[1]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = ap.parse_args()

    if not GATE_PATH.exists():
        raise SystemExit(f"{GATE_PATH} is missing. Pre-register Gate A before measuring.")

    cat = load_catalog()
    tax = load_taxonomy()
    print(f"generating {args.n} patients ...")
    records = generate_corpus(args.n, seed=args.seed)

    vital_keys = [k for k in cat.vital_keys]
    all_keys = vital_keys + list(cat.analyte_keys)

    probes: list[ProbeResult] = []
    xb, y, _, _ = _encode(records, [])
    probes.append(_run_probe("F", xb, y, args.seed, [], [], blank=True))
    xv, _, nv, cv = _encode(records, vital_keys)
    probes.append(_run_probe("V", xv, y, args.seed, nv, cv))
    xt, _, nt, ct = _encode(records, all_keys)
    probes.append(_run_probe("T", xt, y, args.seed, nt, ct))
    xl, _, nl, cl = _encode(records, all_keys, leaky=True)
    probes.append(_run_probe("T_with_leak", xl, y, args.seed, nl, cl))

    by = {p.name: p for p in probes}
    counts = np.bincount(y, minlength=len(tax))
    majority = float(counts.max() / counts.sum())

    summary = {
        "n_patients": args.n,
        "seed": args.seed,
        "n_labels": len(tax),
        "majority_class_rate": majority,
        "probes": {p.name: p.__dict__ for p in probes},
        "v_minus_f": by["V"].balanced_accuracy - by["F"].balanced_accuracy,
        "t_minus_v": by["T"].balanced_accuracy - by["V"].balanced_accuracy,
        "v_minus_f_top1": by["V"].top1_accuracy - by["F"].top1_accuracy,
        "t_minus_v_top1": by["T"].top1_accuracy - by["V"].top1_accuracy,
        "leak_control_gain": by["T_with_leak"].balanced_accuracy - by["T"].balanced_accuracy,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'probe':<14}{'bal.acc':>10}{'top1':>8}{'top5':>8}{'w.brier':>10}{'feats':>7}")
    for p in probes:
        print(f"{p.name:<14}{p.balanced_accuracy:>10.4f}{p.top1_accuracy:>8.4f}"
              f"{p.top5_accuracy:>8.4f}{p.mean_weighted_brier:>10.4f}{p.n_features:>7}")
    print(f"\nmajority-class rate : {majority:.4f}")
    print(f"V - F               : {summary['v_minus_f']:+.4f}")
    print(f"T - V               : {summary['t_minus_v']:+.4f}   <- the size of the prize")
    print(f"T - V (top-1)       : {summary['t_minus_v_top1']:+.4f}")
    print(f"leak control gain   : {summary['leak_control_gain']:+.4f}   "
          f"<- must be LARGE, or the leak probe is blind")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
