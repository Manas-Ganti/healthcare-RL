"""Split a policy's diagnosis deficit into miscalibration and wrong ranking.

The question this answers: when a policy scores BELOW the blank-record floor, is it because
it knows things and overcommits, or because it is ranking the wrong conditions entirely?
Those have completely different fixes -- the first is a training problem, the second is an
environment problem -- and they are indistinguishable from the aggregate reward.

The probe: replace each reported distribution `p` with a shrunk one,

    p_alpha = (1 - alpha) * prior + alpha * p

and rescore. `alpha = 1` is the policy as it ran; `alpha = 0` throws away everything it
said and reports the prevalence prior. Everything else about the trajectory is untouched --
the same tests were ordered at the same cost -- so the cost term is constant across the
sweep and the whole curve moves with the diagnosis term alone.

Reading it:

  * a peak at an INTERIOR alpha means the policy's ranking carries real information that
    its confidence is destroying. The gap between that peak and alpha=1 is the score being
    lost to miscalibration, and it is recoverable in training.
  * a curve that rises monotonically as alpha falls, peaking at alpha=0, means the policy's
    ranking is worse than no information at all. No amount of tempering fixes that; the
    model does not know the environment's likelihoods and only the teacher can supply them.

This costs nothing to run. Reward is a pure function of (trajectory, ground_truth, config)
[I8], so the stored rollouts can be rescored offline as many times as you like -- no GPU,
no new generations. Regenerating rollouts on a 7B policy is the dominant cost of the
project; this is the reason the store exists.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from dxenv.data.corpus import generate_corpus
from dxenv.data.store import read_episodes
from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.schemas import Diagnose
from dxenv.reward.engine import GroundTruth, RewardConfig, load_reward_config, score_trajectory

DEFAULT_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

INTERIOR_PEAK_MARGIN = 0.02
"""How far an interior peak must beat the pure prior before it is read as informative.

Without it the sweep reports "the ranking carries real information" for `random_schema` --
a uniform sampler over the grammar with no model behind it -- because its optimum lands at
alpha=0.1 rather than exactly 0.0, worth +0.004. A detector that fires on the row built to
contain nothing is a detector nobody will believe on the row that matters.
"""


def shrink(
    distribution: dict[str, float], alpha: float, prior: dict[str, float]
) -> dict[str, float]:
    """Mix a reported distribution toward the prevalence prior.

    The result is renormalised only to absorb float drift: both inputs sum to 1, so a
    convex combination of them does too, and any residual is accumulated error over 149
    terms rather than a claim the policy made. The `Diagnose` schema rejects anything off
    by more than 1e-6, so this cannot paper over a real inconsistency.
    """
    mixed = {
        slug: (1.0 - alpha) * prior[slug] + alpha * distribution.get(slug, 0.0)
        for slug in prior
    }
    total = sum(mixed.values())
    if total <= 0.0:
        raise ValueError("shrunk distribution has no mass; prior or report is degenerate")
    return {k: v / total for k, v in mixed.items()}


def reweighted(
    trajectory: dict[str, Any], alpha: float, prior: dict[str, float]
) -> dict[str, Any]:
    """Rebuild a stored trajectory with its diagnosis shrunk. Everything else preserved.

    Trajectories are scored in their stored dict form -- the reward engine reads them that
    way -- so this rewrites one key of one step and leaves the rest aliased. Nothing here
    mutates the input.

    A trajectory that abstained or ran out of turns has no report to shrink and is returned
    unchanged, so it contributes the same score at every alpha: it flattens the curve rather
    than distorting it.

    The shrunk report is round-tripped through `Diagnose` so the sum-to-one validator runs.
    That check is cheap and it is the one thing worth keeping: a distribution that silently
    stopped summing to 1 would be scored anyway and the whole sweep would be quietly wrong.
    """
    steps = []
    touched = False
    for step in trajectory["steps"]:
        action = step.get("action", {})
        if action.get("kind") == "diagnose" and "distribution" in action:
            dist = shrink({str(k): float(v) for k, v in action["distribution"].items()},
                          alpha, prior)
            validated = Diagnose(action_id=str(action["action_id"]), distribution=dist)
            steps.append({**step, "action": {**action,
                                             "distribution": dict(validated.distribution)}})
            touched = True
        else:
            steps.append(step)
    if not touched:
        return trajectory
    return {**trajectory, "steps": steps}


def sweep_policy(
    episodes: list[Any],
    alphas: tuple[float, ...],
    prior: dict[str, float],
    cfg: RewardConfig,
    analytes: dict[str, Any],
) -> list[dict[str, float]]:
    out = []
    for alpha in alphas:
        totals, diags = [], []
        for ep in episodes:
            pid = ep.ground_truth["patient_id"]
            a, allergies = analytes.get(pid, ({}, ()))
            gt = GroundTruth(ep.ground_truth["condition"], a, allergies)
            b = score_trajectory(reweighted(ep.trajectory, alpha, prior), gt, cfg)
            totals.append(b.total)
            diags.append(b.diagnosis)
        out.append({
            "alpha": alpha,
            "mean_total": float(np.mean(totals)),
            "mean_diagnosis": float(np.mean(diags)),
        })
    return out


def classify(
    rows: list[dict[str, float]], margin: float = INTERIOR_PEAK_MARGIN
) -> tuple[str, float, float]:
    """Which of the three diagnoses the sweep supports, plus the two margins it rests on.

    Returns one of:

      "calibration" -- an interior peak clearing BOTH endpoints. The ranking carries
                       information the confidence is destroying. Fixable in training.
      "knowledge"   -- nothing beats the pure prior. The ranking is uninformative, so
                       tempering and further GRPO are both beside the point.
      "ranking"     -- shrinking buys nothing. The confidence already suits the ranking;
                       whatever is missing is in what the policy ranks first.

    An interior peak must clear both endpoints, and both guards were earned on real rows.
    `random_schema` -- a uniform sampler over the grammar with no model behind it -- peaks
    at alpha=0.1, worth +0.004 over the pure prior. `greedy_bayes` -- exact Bayes over the
    true observation model, the best-calibrated policy in the repo -- peaks at alpha=0.9,
    worth +0.001 over its own report. Without the margins this sweep reports that a random
    sampler has learned something and that exact Bayes is overconfident, which is a
    detector firing on the two rows built to be the controls.
    """
    as_ran = next(r for r in rows if r["alpha"] == 1.0)
    all_prior = next(r for r in rows if r["alpha"] == 0.0)
    best = max(rows, key=lambda r: r["mean_total"])
    beats_prior = best["mean_total"] - all_prior["mean_total"]
    beats_as_ran = best["mean_total"] - as_ran["mean_total"]

    if 0.0 < best["alpha"] < 1.0 and beats_prior >= margin and beats_as_ran >= margin:
        return "calibration", beats_prior, beats_as_ran
    if beats_prior < margin:
        return "knowledge", beats_prior, beats_as_ran
    return "ranking", beats_prior, beats_as_ran


def report(name: str, rows: list[dict[str, float]], floor: float | None) -> None:
    as_ran = next(r for r in rows if r["alpha"] == 1.0)
    all_prior = next(r for r in rows if r["alpha"] == 0.0)
    best = max(rows, key=lambda r: r["mean_total"])

    print(f"\n### {name}   (n rescored at each alpha)\n")
    print(f"{'alpha':>7}{'mean R':>10}{'diagnosis':>12}{'vs as-ran':>11}")
    for r in rows:
        mark = "  <- best" if r["alpha"] == best["alpha"] else ""
        print(f"{r['alpha']:>7.2f}{r['mean_total']:>+10.3f}{r['mean_diagnosis']:>+12.3f}"
              f"{r['mean_total'] - as_ran['mean_total']:>+11.3f}{mark}")

    gain = best["mean_total"] - as_ran["mean_total"]
    print(f"\nas it ran (alpha=1)      {as_ran['mean_total']:+.3f}")
    print(f"pure prior (alpha=0)     {all_prior['mean_total']:+.3f}")
    print(f"best (alpha={best['alpha']:.2f})          {best['mean_total']:+.3f}   "
          f"recoverable by tempering alone: {gain:+.3f}")
    if floor is not None:
        print(f"blank-record floor       {floor:+.3f}")

    print()
    verdict, beats_prior, beats_as_ran = classify(rows)
    interior = verdict == "calibration"

    if interior:
        print(f"READING: a real interior peak at alpha={best['alpha']:.2f}, clearing both")
        print("  endpoints. The policy's ranking carries information its confidence is")
        print(f"  destroying: {beats_as_ran:+.3f} of score is lost to overcommitment alone,")
        print("  with no change to which conditions it favours. That is recoverable in")
        print("  TRAINING -- it is the failure CLAUDE.md 8.4 exists to prevent, so check")
        print("  SFTDataset.mean_target_entropy and that soft labels come from the Bayes")
        print("  posterior over visible evidence rather than from the demonstrator.")
        print("  NOTE: a diagnostic, not a fix. Shipping the shrunk distribution would be a")
        print("  calibration head bolted onto a miscalibrated policy, which gate_b.yaml")
        print("  explicitly declines as a remedy.")
    elif beats_prior < INTERIOR_PEAK_MARGIN:
        print(f"READING: nothing beats the pure prior by more than {INTERIOR_PEAK_MARGIN}")
        print(f"  (best margin {beats_prior:+.3f}). The policy's RANKING carries no usable")
        print("  information -- reporting prevalence and discarding everything it said scores")
        print("  the same or better. This is a KNOWLEDGE problem, not a calibration one, and")
        print("  neither tempering nor more GRPO touches it: the model does not know this")
        print("  environment's likelihoods, and the privileged teacher is the only thing that")
        print("  has them. Spend the next hours on SFT data, not on the RL loop.")
    else:
        print(f"READING: shrinking buys at most {beats_as_ran:+.3f}, below the")
        print(f"  {INTERIOR_PEAK_MARGIN} margin, so the report is already at about the right")
        print("  confidence for its ranking. The deficit is NOT miscalibration. Whatever is")
        print("  missing is in which conditions the policy ranks first, not in how hard it")
        print("  commits to them -- so look at the evidence it is conditioning on, not at")
        print("  the entropy of its targets.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="runs/<run_id> containing episodes.jsonl")
    ap.add_argument("--policy", default=None,
                    help="only this policy tag; default sweeps every policy in the store")
    ap.add_argument("--config", type=Path, default=None, help="a different reward.yaml")
    ap.add_argument("--corpus-n", type=int, default=None,
                    help="regenerate this many patients to recover analytes and allergies")
    ap.add_argument("--corpus-seed", type=int, default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    ap.add_argument("--out", type=Path, default=None, help="write the sweep as JSON")
    args = ap.parse_args()

    path = args.run / "episodes.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} does not exist")

    alphas = tuple(sorted(set(args.alphas)))
    if not {0.0, 1.0} <= set(alphas):
        raise SystemExit("the alpha grid must include 0.0 and 1.0; they are the endpoints "
                         "every reading below is stated against")

    tax: Taxonomy = load_taxonomy()
    prior_vec = tax.prior()
    prior = {slug: float(prior_vec[i]) for i, slug in enumerate(tax.slugs)}
    cfg = load_reward_config(args.config)

    analytes: dict[str, Any] = {}
    if args.corpus_n is not None and args.corpus_seed is not None:
        analytes = {
            r.patient_id: (r.analytes, r.allergies)
            for r in generate_corpus(args.corpus_n, seed=args.corpus_seed)
        }

    by_policy: dict[str, list[Any]] = defaultdict(list)
    for ep in read_episodes(path):
        name = str(ep.tags.get("policy", "unknown"))
        if args.policy is None or name == args.policy:
            by_policy[name].append(ep)
    if not by_policy:
        raise SystemExit(f"no episodes matched --policy {args.policy!r}")

    floor = None
    if "prior" in by_policy:
        floor = float(np.mean([float(e.tags["reward"]) for e in by_policy["prior"]
                               if "reward" in e.tags])) or None

    payload = {}
    for name, eps in sorted(by_policy.items()):
        rows = sweep_policy(eps, alphas, prior, cfg, analytes)
        payload[name] = {"n": len(eps), "sweep": rows}
        report(f"{name}  (n={len(eps)})", rows, floor)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
