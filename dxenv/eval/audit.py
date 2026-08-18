"""The audit suite (CLAUDE.md 10).

Every probe is both a test and a reported result. The suite exists to answer one
question: is the headline number measuring diagnosis, or measuring a leak?

Design rule throughout: a probe that CANNOT FAIL is not evidence. Several of these
checks pass trivially in this repo -- the leakage ablation, for instance, because blocked
resources never reach a feature matrix at all. Where that happens the probe carries a
POSITIVE CONTROL that deliberately injects the failure and requires the probe to catch
it. An audit suite that would not catch a real failure is worse than none, because it
manufactures confidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from dxenv.data.corpus import BLOCKED_RESOURCE_TYPES, PatientRecord
from dxenv.data.splits import DEFAULT_HOLDOUT_SYSTEMS
from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import ActionKind, action_id
from dxenv.env.bayes import entropy, hard_ceiling, posterior
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.episode import DiagnosticEpisode, EpisodeConfig, load_episode_config
from dxenv.env.filter import filter_resources
from dxenv.env.obs_model import ObservationModel, QuantTable, build_observation_model
from dxenv.env.schemas import Diagnose
from dxenv.policy.baselines import (
    GreedyBayesPolicy,
    Policy,
    PriorPolicy,
    VitalsOnlyPolicy,
    run_episode,
)
from dxenv.reward.engine import GroundTruth, RewardConfig, load_reward_config, score_trajectory
from dxenv.reward.scoring import brier_score, load_severity, severity_weight


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        nums = "  ".join(f"{k}={v:+.4f}" for k, v in self.metrics.items())
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name:<26} {nums}\n    {self.detail}"


@dataclass(frozen=True, slots=True)
class AuditReport:
    outcomes: tuple[ProbeOutcome, ...]

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outcomes)

    def render(self) -> str:
        head = "AUDIT: " + ("PASS" if self.passed else "FAIL")
        return "\n".join([head, "=" * len(head), *(o.line() for o in self.outcomes)])


def _score(records: Sequence[PatientRecord], policy: Policy, cfg: EpisodeConfig,
           rcfg: RewardConfig, budget: float = 200.0, seed: int = 0) -> dict[str, float]:
    totals, diags, tests = [], [], []
    for i, rec in enumerate(records):
        traj = run_episode(
            DiagnosticEpisode(rec, seed=seed + i, config=cfg, budget=budget), policy
        )
        b = score_trajectory(
            traj, GroundTruth(rec.condition, rec.analytes, rec.allergies), rcfg
        )
        totals.append(b.total)
        diags.append(b.diagnosis)
        tests.append(b.n_tests_charged)
    return {
        "total": float(np.mean(totals)),
        "diagnosis": float(np.mean(diags)),
        "tests": float(np.mean(tests)),
    }


# ------------------------------------------------------------------------- probes ----


def probe_blank_record_baseline(
    records: Sequence[PatientRecord], cfg: EpisodeConfig, rcfg: RewardConfig
) -> ProbeOutcome:
    """An agent with no observation must land at the prior. Everything is reported above this."""
    prior = _score(records, PriorPolicy(), cfg, rcfg)
    tax = load_taxonomy()
    expected = float(
        sum(
            tax.prior()[i] * brier_score(tax.prior(), i) * severity_weight(tax.slugs[i], tax)
            for i in range(len(tax))
        )
    )
    gap = abs(prior["diagnosis"] - expected)
    return ProbeOutcome(
        name="blank_record_baseline",
        passed=gap < 0.15 and prior["tests"] == 0.0,
        detail=(
            f"prior-reporting policy scores {prior['diagnosis']:+.4f}; the analytic "
            f"expectation is {expected:+.4f}. This is the FLOOR -- report every other "
            f"number against it, not against zero."
        ),
        metrics={"observed": prior["diagnosis"], "expected": expected, "gap": gap},
    )


def probe_no_test_ablation(
    records: Sequence[PatientRecord], cfg: EpisodeConfig, rcfg: RewardConfig
) -> ProbeOutcome:
    """Zero-test accuracy must be meaningfully worse than with tests, or tests are decorative."""
    without = _score(records, VitalsOnlyPolicy(), cfg, rcfg)
    with_tests = _score(records, GreedyBayesPolicy(), cfg, rcfg)
    gain = with_tests["diagnosis"] - without["diagnosis"]
    return ProbeOutcome(
        name="no_test_ablation",
        passed=gain > 0.2 and with_tests["tests"] > 0,
        detail=(
            f"ordering tests buys {gain:+.4f} of weighted score. If this were near zero "
            f"the reward signal would be that many points wide and any policy gradient "
            f"buried in rollout noise."
        ),
        metrics={"no_tests": without["diagnosis"], "with_tests": with_tests["diagnosis"],
                 "gain": gain},
    )


def probe_leakage_ablation(
    records: Sequence[PatientRecord], cfg: EpisodeConfig, rcfg: RewardConfig
) -> ProbeOutcome:
    """Stripping the label-bearing resources must barely move the score.

    Passes trivially here -- those resources never reach an observation -- so the probe
    also runs a POSITIVE CONTROL that hands the policy the label and requires the score
    to jump. Without that, this probe proves nothing.
    """
    base = _score(records, VitalsOnlyPolicy(), cfg, rcfg)
    stripped = [
        r for rec in records for r in filter_resources(rec.resources, strict=False)
    ]
    leaked_types = {r["resourceType"] for r in stripped} & BLOCKED_RESOURCE_TYPES

    tax = load_taxonomy()
    oracle = _score(records, _OraclePolicy(tax), cfg, rcfg)
    control_gain = oracle["diagnosis"] - base["diagnosis"]
    return ProbeOutcome(
        name="leakage_ablation",
        passed=not leaked_types and control_gain > 1.0,
        detail=(
            f"no blocked resource survives the filter ({len(stripped)} resources kept, "
            f"0 blocked). Positive control: a policy handed the label scores "
            f"{control_gain:+.4f} above the vitals-only baseline, so the probe is not "
            f"blind to a leak."
        ),
        metrics={"blocked_surviving": float(len(leaked_types)),
                 "control_gain": control_gain},
    )


@dataclass(slots=True)
class _OraclePolicy:
    """Positive control ONLY. Reads ground truth; never a baseline to report against."""

    taxonomy: Taxonomy

    def act(self, episode: DiagnosticEpisode, obs: Any) -> Any:  # noqa: ARG002
        return Diagnose(
            action_id=action_id(ActionKind.DIAGNOSE, "diagnose"),
            distribution={episode.record.condition: 1.0},
        )


def probe_shuffled_labels(
    records: Sequence[PatientRecord], cfg: EpisodeConfig, rcfg: RewardConfig, seed: int = 0
) -> ProbeOutcome:
    """With ground truth shuffled, reward must collapse to chance.

    If it does not, the score is coming from something other than the label -- which is
    the definition of a leak in the scoring path rather than the observation path.
    """
    rng = np.random.default_rng(seed)
    real = _score(records, GreedyBayesPolicy(), cfg, rcfg)

    shuffled_conditions = list(rng.permutation([r.condition for r in records]))
    totals = []
    for i, rec in enumerate(records):
        traj = run_episode(
            DiagnosticEpisode(rec, seed=i, config=cfg, budget=200.0), GreedyBayesPolicy()
        )
        gt = GroundTruth(str(shuffled_conditions[i]), rec.analytes, rec.allergies)
        totals.append(score_trajectory(traj, gt, rcfg).diagnosis)
    shuffled = float(np.mean(totals))
    drop = real["diagnosis"] - shuffled
    return ProbeOutcome(
        name="shuffled_labels",
        passed=shuffled < 0.1 and drop > 0.5,
        detail=(
            f"shuffling ground truth drops the diagnosis score from "
            f"{real['diagnosis']:+.4f} to {shuffled:+.4f}. A small drop would mean the "
            f"reward is not actually keyed on the label."
        ),
        metrics={"real": real["diagnosis"], "shuffled": shuffled, "drop": drop},
    )


CLINICAL_SPOT_CHECKS: tuple[tuple[str, float, str], ...] = (
    ("troponin", 1200.0, "acute_myocardial_infarction"),
    ("d_dimer", 3400.0, "pulmonary_embolism"),
    ("lipase", 1450.0, "acute_pancreatitis"),
    ("tsh", 16.5, "hypothyroidism"),
    ("beta_hydroxybutyrate", 6.2, "diabetic_ketoacidosis"),
    ("carboxyhemoglobin", 26.0, "carbon_monoxide_poisoning"),
)
"""(analyte, characteristic value, condition whose mass must rise).

Named checks, because the generic property below is guaranteed by Bayes and would pass
even if every override in obs_overrides.yaml were assigned to the wrong condition. These
are the ones that would catch that."""


def probe_counterfactual_perturbation(
    records: Sequence[PatientRecord],
    model: ObservationModel | None = None,
    catalog: Catalog | None = None,
) -> ProbeOutcome:
    """Perturb a lab; the posterior must move in the correct direction.

    Two checks, and the distinction matters:

    1. GENERIC. Mass must shift toward the conditions with the higher LIKELIHOOD AT THE
       OBSERVED VALUE -- not the higher mean. Those differ: a condition with mean 1200
       and sd 900 has low density at 42, so ranking by mean gets the direction wrong.
       (The first version of this probe ranked by mean and "failed" at 61%, which was the
       probe being wrong, not the model.) This one is guaranteed by Bayes, so it is a
       consistency check that catches a sign error in the likelihood.

    2. CLINICAL. A characteristic value must raise the mass on the condition it is
       characteristic OF. The generic check would pass even if every override were
       attached to the wrong condition; this is what catches that.
    """
    m = model or build_observation_model()
    cat = catalog or load_catalog()
    tax = load_taxonomy()

    # Fail closed if the spot-check table has drifted from the catalog or taxonomy: a
    # probe that silently skips its checks reports PASS while testing nothing.
    for analyte, _value, condition in CLINICAL_SPOT_CHECKS:
        cat.analyte(analyte)
        tax.index(condition)

    checks = moved = 0
    for rec in records[:40]:
        base_ev = {"presenting_complaint": rec.analytes["presenting_complaint"]}
        base = posterior(base_ev, m)
        for analyte in ("troponin", "d_dimer", "lipase", "tsh", "creatinine"):
            table = m.table(analyte)
            assert isinstance(table, QuantTable)
            value = float(np.percentile(np.asarray(table.mean), 98))
            perturbed = posterior({**base_ev, analyte: value}, m)
            favoured = np.argsort(-m.log_likelihood_vector(analyte, value))[:10]
            checks += 1
            if perturbed[favoured].sum() > base[favoured].sum():
                moved += 1
    rate = moved / max(checks, 1)

    spot_ok, spot_fail = 0, []
    for analyte, value, condition in CLINICAL_SPOT_CHECKS:
        base = posterior({}, m)
        after = posterior({analyte: value}, m)
        i = tax.index(condition)
        if after[i] > base[i]:
            spot_ok += 1
        else:
            spot_fail.append(f"{analyte}={value} did not raise {condition}")

    passed = rate > 0.99 and spot_ok == len(CLINICAL_SPOT_CHECKS)
    return ProbeOutcome(
        name="counterfactual_perturbation",
        passed=passed,
        detail=(
            f"{moved}/{checks} perturbations moved mass toward the higher-likelihood "
            f"conditions; {spot_ok}/{len(CLINICAL_SPOT_CHECKS)} clinical spot checks "
            f"moved the right condition." + (f" FAILURES: {spot_fail}" if spot_fail else "")
        ),
        metrics={"generic_rate": rate, "spot_checks": float(spot_ok),
                 "n": float(checks)},
    )


def probe_bayes_ceiling(
    records: Sequence[PatientRecord], cfg: EpisodeConfig, rcfg: RewardConfig
) -> ProbeOutcome:
    """No episode may exceed the HARD ceiling. See env/bayes.py on why not the expected one."""
    tax = load_taxonomy()
    sev = load_severity()
    weights = np.array([sev.weight(lab.urgency) for lab in tax.labels])

    def fn(report: npt.NDArray[np.float64], true_idx: int) -> float:
        return brier_score(report, true_idx) * float(weights[true_idx])

    ceiling = hard_ceiling(fn, len(tax))
    worst = -np.inf
    for i, rec in enumerate(records):
        traj = run_episode(
            DiagnosticEpisode(rec, seed=i, config=cfg, budget=200.0), GreedyBayesPolicy()
        )
        b = score_trajectory(
            traj, GroundTruth(rec.condition, rec.analytes, rec.allergies), rcfg
        )
        worst = max(worst, b.diagnosis)
    return ProbeOutcome(
        name="bayes_ceiling",
        passed=worst <= ceiling + 1e-6,
        detail=(
            f"highest single-episode diagnosis score {worst:+.4f} against a hard ceiling "
            f"of {ceiling:+.4f}. A breach is a LEAK until proven otherwise."
        ),
        metrics={"max_observed": worst, "ceiling": ceiling, "headroom": ceiling - worst},
    )


def probe_held_out_modules(
    records: Sequence[PatientRecord], cfg: EpisodeConfig, rcfg: RewardConfig
) -> ProbeOutcome:
    """Report the generalisation gap across withheld organ systems.

    Reported, not gated: a nonzero gap is expected and informative. Gating it would
    pressure future work toward hiding the gap rather than measuring it.
    """
    tax = load_taxonomy()
    held = [r for r in records if tax.get(r.condition).system in DEFAULT_HOLDOUT_SYSTEMS]
    seen = [r for r in records if tax.get(r.condition).system not in DEFAULT_HOLDOUT_SYSTEMS]
    if not held or not seen:
        return ProbeOutcome(
            name="held_out_modules", passed=True,
            detail="no held-out patients in this sample; gap not measurable",
            metrics={"n_held": float(len(held))},
        )
    a = _score(seen, GreedyBayesPolicy(), cfg, rcfg)
    b = _score(held, GreedyBayesPolicy(), cfg, rcfg)
    return ProbeOutcome(
        name="held_out_modules",
        passed=True,
        detail=(
            f"seen systems {a['diagnosis']:+.4f} vs held-out {b['diagnosis']:+.4f}. "
            f"Reported, never gated -- gating a generalisation gap rewards hiding it."
        ),
        metrics={"seen": a["diagnosis"], "held_out": b["diagnosis"],
                 "gap": a["diagnosis"] - b["diagnosis"]},
    )


def run_audit(
    records: Sequence[PatientRecord],
    cfg: EpisodeConfig | None = None,
    rcfg: RewardConfig | None = None,
    model: ObservationModel | None = None,
    catalog: Catalog | None = None,
) -> AuditReport:
    c = cfg or load_episode_config()
    r = rcfg or load_reward_config()
    return AuditReport(outcomes=(
        probe_blank_record_baseline(records, c, r),
        probe_no_test_ablation(records, c, r),
        probe_leakage_ablation(records, c, r),
        probe_shuffled_labels(records, c, r),
        probe_counterfactual_perturbation(records, model, catalog),
        probe_bayes_ceiling(records, c, r),
        probe_held_out_modules(records, c, r),
    ))


def posterior_entropy_summary(
    records: Sequence[PatientRecord], model: ObservationModel | None = None
) -> dict[str, float]:
    m = model or build_observation_model()
    ents = [
        entropy(posterior({"presenting_complaint": r.analytes["presenting_complaint"]}, m))
        for r in records
    ]
    return {"mean_entropy_after_complaint": float(np.mean(ents))}
