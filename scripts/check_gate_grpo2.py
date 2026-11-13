"""Evaluate a sweep results file against the PRE-REGISTERED GRPO2 gate.

Reads `dxenv/configs/gate_grpo2.yaml`. Never edits it. The comparison is PAIRED by
patient: every row in the results file was scored on the same 200 patients, so the
per-patient difference removes patient difficulty from the noise.

    python scripts/check_gate_grpo2.py --results runs/phase3/grpo2_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

GATE = Path("dxenv/configs/gate_grpo2.yaml")


def paired(
    subject: list[float], baseline: list[float], resamples: int, seed: int
) -> tuple[float, float, float]:
    """(mean difference, 2.5th percentile, 97.5th percentile) by patient bootstrap."""
    a, b = np.asarray(subject, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise SystemExit(f"rows are not paired: {a.shape} vs {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def evaluate(results: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = {r["policy"]: r for r in results["rows"]}
    label = gate["subject"]["label"]
    th = gate["thresholds"]
    out: list[dict[str, Any]] = []

    for need in (label, gate["baselines"]["primary"], gate["baselines"]["stretch"]):
        if need not in rows:
            raise SystemExit(f"results file has no {need!r} row; rows: {sorted(rows)}")
        if "per_patient_mean" not in rows[need]:
            raise SystemExit(
                f"row {need!r} has no per_patient_mean; re-run the sweep with the current "
                "scripts/phase3_prompted_baseline.py"
            )

    subj = rows[label]
    for role in ("primary", "stretch"):
        base = gate["baselines"][role]
        mean, lo, hi = paired(subj["per_patient_mean"], rows[base]["per_patient_mean"],
                              int(th["bootstrap_resamples"]), int(th["bootstrap_seed"]))
        ok = mean >= th["min_mean_margin"] and lo > th["ci_lower_bound_min"]
        out.append({
            "criterion": f"beats_{base}" + ("" if role == "primary" else " (stretch)"),
            "role": role, "passed": ok,
            "detail": f"paired diff {mean:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]; "
                      f"need >= {th['min_mean_margin']:+.3f} with lower bound > "
                      f"{th['ci_lower_bound_min']:+.3f}",
        })

    dx = float(subj.get("ending_fractions", {}).get("diagnose", 0.0))
    out.append({"criterion": "diagnose_fraction", "role": "guard",
                "passed": dx >= th["min_diagnose_fraction"],
                "detail": f"{dx:.1%} (min {th['min_diagnose_fraction']:.0%}); "
                          f"endings {subj.get('ending_fractions')}"})
    sv = float(subj["schema_valid_fraction"])
    out.append({"criterion": "schema_valid_fraction", "role": "guard",
                "passed": sv >= th["schema_valid_fraction_min"],
                "detail": f"{sv:.4f} (min {th['schema_valid_fraction_min']})"})
    cm = float(subj["calibration_margin"])
    out.append({"criterion": "calibration_margin", "role": "guard",
                "passed": cm >= th["calibration_margin_min"],
                "detail": f"{cm:.3f} (min {th['calibration_margin_min']})"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    gate = yaml.safe_load(GATE.read_text())
    verdicts = evaluate(json.loads(args.results.read_text()), gate)
    for v in verdicts:
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['criterion']}: {v['detail']}")
    decisive = [v for v in verdicts if v["role"] != "stretch"]
    passed = all(v["passed"] for v in decisive)
    print(f"GATE GRPO2: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
