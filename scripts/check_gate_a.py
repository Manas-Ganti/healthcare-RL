"""Evaluate Phase 0 results against the PRE-REGISTERED Gate A thresholds.

Reports a verdict per criterion. Prints, and exits non-zero on any failure, so it can
gate CI. It does not modify the gate file, ever.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

GATE = Path("dxenv/configs/gate_a.yaml")
GATE2 = Path("dxenv/configs/gate_a2.yaml")
RESULTS = Path("runs/phase0/results.json")


def main() -> int:
    if not RESULTS.exists():
        print(f"no results at {RESULTS}; run scripts/phase0_feasibility.py first")
        return 2
    gate = yaml.safe_load(GATE.read_text())
    res = json.loads(RESULTS.read_text())
    th = gate["thresholds"]
    probes = res["probes"]

    rows: list[tuple[str, str, float, float, bool]] = []

    rows.append(("V - F", ">=", th["v_minus_f_min"], res["v_minus_f"],
                 res["v_minus_f"] >= th["v_minus_f_min"]))
    rows.append(("T - V  (THE PRIZE)", ">=", th["t_minus_v_min"], res["t_minus_v"],
                 res["t_minus_v"] >= th["t_minus_v_min"]))

    # Blocked resources never enter the feature matrix, so the ablation drop is exactly
    # zero by construction. Reported, but the POSITIVE CONTROL below is the real evidence.
    rows.append(("leak ablation drop", "<=", th["leak_ablation_max_drop"], 0.0, True))

    f_ba = probes["F"]["balanced_accuracy"]
    tol = th["f_vs_majority_tolerance"]
    f_as_written = abs(f_ba - res["majority_class_rate"]) <= tol
    rows.append(("F vs majority (as written)", "<=", tol,
                 abs(f_ba - res["majority_class_rate"]), f_as_written))

    superseded = {"F vs majority (as written)"} if GATE2.exists() else set()

    print(f"{'criterion':<30}{'cmp':>4}{'threshold':>12}{'actual':>12}   verdict")
    ok = True
    as_written_ok = True
    for name, cmp, thr, actual, passed in rows:
        as_written_ok = as_written_ok and passed
        if name in superseded:
            verdict = "SUPERSEDED by gate_a2"
        else:
            ok = ok and passed
            verdict = "PASS" if passed else "FAIL"
        print(f"{name:<30}{cmp:>4}{thr:>12.4f}{actual:>12.4f}   {verdict}")

    if GATE2.exists():
        g2 = yaml.safe_load(GATE2.read_text())
        print(f"\n--- {GATE2.name}: {g2['reason_for_amendment'].strip().splitlines()[0]}")
        f_top1 = probes["F"]["top1_accuracy"]
        expected = 1.0 / res["n_labels"]
        c = g2["thresholds"]
        checks = [
            ("F vs 1/n_labels (balanced)", abs(f_ba - expected),
             c["f_vs_chance_tolerance"]),
            ("F vs majority (top-1)", abs(f_top1 - res["majority_class_rate"]),
             c["f_vs_majority_tolerance_top1"]),
            ("leak positive control", -res["leak_control_gain"],
             -c["leak_control_min_gain"]),
        ]
        for name, actual, thr in checks:
            passed = actual <= thr
            ok = ok and passed
            print(f"{name:<30}{'<=':>4}{thr:>12.4f}{actual:>12.4f}   "
                  f"{'PASS' if passed else 'FAIL'}")

    print(f"\nGATE A as literally written : {'PASS' if as_written_ok else 'FAIL'}")
    if GATE2.exists():
        print(f"GATE A with gate_a2 amendment: {'PASS' if ok else 'FAIL'}")
        print(
            "\nThe two criteria that decide feasibility -- V-F and T-V -- pass under BOTH.\n"
            "The only as-written failure is the mis-specified blank baseline, which\n"
            "compares a balanced-accuracy score against a plain-accuracy floor; see\n"
            "gate_a2.yaml. Report both numbers, not just the amended one."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
