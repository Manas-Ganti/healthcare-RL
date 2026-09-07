"""Plot the GRPO learning curve from a run's steps.jsonl.

The figure is the Phase 4 result. Kept as a script rather than a notebook so it
regenerates from committed data with one command and cannot drift from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PANELS = [
    ("mean_reward", "Episode reward", "the quantity being optimised"),
    ("mean_diagnosis", "Diagnosis score", "reward with costs and penalties removed"),
    ("mean_tests", "Tests ordered per episode", "SFT left this at 0.79"),
    ("mean_group_std", "Within-group reward std", "the spread GRPO needs; a floor at 0.05"),
    ("kl", "KL from the SFT reference", "how far the policy has moved"),
    ("ceiling_gap", "Gap below the Bayes ceiling", "lower is closer to optimal"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=Path, default=Path("runs/grpo/steps.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/grpo/curve.png"))
    ap.add_argument("--smooth", type=int, default=9, help="moving-average window")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(
        (json.loads(line) for line in args.steps.read_text().splitlines() if line.strip()),
        key=lambda r: r["step"],
    )
    step = np.array([r["step"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (key, title, subtitle) in zip(axes.ravel(), PANELS, strict=True):
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        ax.plot(step, y, lw=0.9, alpha=0.35, color="tab:blue")
        if args.smooth > 1 and len(y) >= args.smooth:
            # Raw and smoothed together: the raw series is what was measured, the
            # smoothed one is only there to make the direction legible.
            kernel = np.ones(args.smooth) / args.smooth
            smoothed = np.convolve(y, kernel, mode="valid")
            ax.plot(step[args.smooth - 1 :], smoothed, lw=2.0, color="tab:blue")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("GRPO step", fontsize=9)
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=8, alpha=0.7)
        ax.grid(alpha=0.25)
        if key == "mean_group_std":
            ax.axhline(0.05, ls="--", lw=1, color="tab:red")
        if key in {"mean_reward", "mean_diagnosis"}:
            ax.axhline(0.0, ls=":", lw=1, color="grey")

    fig.suptitle(
        f"GRPO on dxenv — {len(rows)} steps, Qwen2.5-7B + LoRA from the SFT checkpoint",
        fontsize=13,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
