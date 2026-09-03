"""Evaluate a Phase 3 results file against the PRE-REGISTERED Gate B thresholds.

Reads `dxenv/configs/gate_b.yaml`. Never edits it. If a threshold is wrong, record a new
gate and report both verdicts -- see the note at the top of gate_a2.yaml for the pattern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

GATE = Path("dxenv/configs/gate_b.yaml")


def verdict(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"criterion": name, "status": "PASS" if ok else "FAIL", "passed": bool(ok),
            "detail": detail}


def skipped(name: str, detail: str) -> dict[str, Any]:
    """A criterion the results file cannot answer.

    Reported explicitly, and it makes the overall verdict INCOMPLETE rather than PASS or
    FAIL. A gate that silently evaluates two of its four criteria and prints a verdict is
    the exact failure CLAUDE.md 11 warns about: an audit that would not catch a real
    failure is worse than none, because it manufactures confidence.
    """
    return {"criterion": name, "status": "SKIP", "passed": False, "detail": detail}


def evaluate(results: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    t = gate["thresholds"]
    rows = {r["policy"]: r for r in results["rows"]}
    subject = rows.get("prompted") or rows.get("random_schema")
    if subject is None:
        raise SystemExit("results contain neither a `prompted` nor a `random_schema` row")
    bar = float(results["gate_b_pass_bar"])

    best = np.array(subject["per_patient_best"], dtype=np.float64)
    first = np.array(subject["per_patient_first"], dtype=np.float64)
    pass_at_k = float((best >= bar).mean())
    pass_at_1 = float((first >= bar).mean())
    stds = np.array(subject["group_stds"], dtype=np.float64)

    out = [
        verdict(
            "pass@k above pass@1",
            pass_at_k - pass_at_1 >= t["pass_at_8_minus_pass_at_1_min"],
            f"pass@{results['k']}={pass_at_k:.3f} vs pass@1={pass_at_1:.3f} "
            f"(gap {pass_at_k - pass_at_1:+.3f}, need "
            f">= {t['pass_at_8_minus_pass_at_1_min']}) against a bar of {bar:+.3f}",
        ),
        verdict(
            "within-group reward variance",
            float(stds.mean()) >= t["mean_group_reward_std_min"],
            f"mean group std {stds.mean():.4f} (need >= {t['mean_group_reward_std_min']})",
        ),
        verdict(
            "degenerate group fraction",
            float((stds < t["degenerate_group_std"]).mean())
            <= t["max_fraction_degenerate_groups"],
            f"{(stds < t['degenerate_group_std']).mean():.1%} of groups below "
            f"std {t['degenerate_group_std']} "
            f"(allowed <= {t['max_fraction_degenerate_groups']:.0%})",
        ),
    ]
    if "calibration_margin" in results:
        out.append(verdict(
            "calibration survived",
            float(results["calibration_margin"]) >= t["calibration_margin_min"],
            f"reported distribution scores {results['calibration_margin']:+.4f} above the "
            f"same distribution collapsed onto its argmax "
            f"(need >= {t['calibration_margin_min']})",
        ))
    else:
        out.append(skipped("calibration survived",
                           "results file carries no `calibration_margin`"))

    if "mean_expected_ceiling" in results:
        headroom = float(results["mean_expected_ceiling"]) - float(subject["mean_reward"])
        out.append(verdict(
            "headroom below ceiling",
            headroom >= t["headroom_below_expected_ceiling_min"],
            f"mean reward sits {headroom:.4f} below the mean expected ceiling "
            f"{results['mean_expected_ceiling']:+.4f} "
            f"(need >= {t['headroom_below_expected_ceiling_min']})",
        ))
    else:
        out.append(skipped("headroom below ceiling",
                           "results file carries no `mean_expected_ceiling`"))

    if "schema_valid_fraction" in results:
        out.append(verdict(
            "schema-valid output",
            float(results["schema_valid_fraction"]) >= t["schema_valid_fraction_min"],
            f"{results['schema_valid_fraction']:.4f} of generations parsed "
            f"(need >= {t['schema_valid_fraction_min']})",
        ))
    else:
        out.append(skipped("schema-valid output",
                           "results file carries no `schema_valid_fraction`"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=Path("runs/phase3/prompted_baseline.json"))
    args = ap.parse_args()
    if not args.results.exists():
        raise SystemExit(f"{args.results} does not exist; run phase3_prompted_baseline.py")

    gate = yaml.safe_load(GATE.read_text())
    results = json.loads(args.results.read_text())
    rows = evaluate(results, gate)
    width = max(len(r["criterion"]) for r in rows)
    subject = results.get("subject_policy", "unknown")
    print(f"subject policy: {subject}\n")
    for r in rows:
        print(f"{r['status']:<5} {r['criterion']:<{width}}  {r['detail']}")

    skips = [r for r in rows if r["status"] == "SKIP"]
    fails = [r for r in rows if r["status"] == "FAIL"]
    if skips:
        print(f"\nGATE B: INCOMPLETE -- {len(skips)} of {len(rows)} criteria could not be "
              "evaluated from this results file. A partial gate is not a gate.")
    else:
        print(f"\nGATE B: {'PASS' if not fails else 'FAIL'}")
    if subject != "prompted":
        print(
            f"\nNOTE: the subject is `{subject}`, not a prompted model. That row measures "
            "the floor a prompted model has to beat -- a grammar with no policy behind it "
            "-- so a FAIL here says nothing about any model. Re-run with --model on a CUDA "
            "host to evaluate the gate as written."
        )
    if fails:
        print("\nDeclared actions on failure (gate_b.yaml, pre-registered):")
        for k, v in gate["on_failure"].items():
            print(f"  {k}:\n    {' '.join(v.split())}")
    raise SystemExit(0 if not fails and not skips else 1)


if __name__ == "__main__":
    main()
