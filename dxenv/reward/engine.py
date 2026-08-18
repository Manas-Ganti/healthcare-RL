"""Pure composition of the reward terms [I8].

    R = brier(p, c_true) * severity_weight(c_true)
        - lambda * sum(cost(t) for t in tests_charged)
        - mu * n_turns
        + treatment_score
        + sum(potential_shaping_terms)     # telescopes, I6
        + sum(predict_then_verify_terms)

`score_trajectory` is a pure function of (trajectory, ground_truth, config): no RNG, no
clock, no global mutable state. It reads no files at call time -- the config is passed in,
and the only lazy loads (bucket priors, the observation model) are cached derivations of
committed data files that are identical for every call. That is what makes rescoring a
stored trajectory corpus under new weights free, and you WILL change these weights
repeatedly.

`validate_reward_config` proves I5 arithmetically instead of trusting a test to notice:
it refuses any configuration in which shaping plus verify could exceed the cheapest test's
cost, because that is exactly the configuration in which ordering a test becomes
net-positive and the agent learns to order everything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.bayes import posterior_sequence
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.obs_model import ObservationModel, ResultValue, build_observation_model
from dxenv.reward.costs import (
    CostTable,
    load_cost_table,
    order_cost_term,
    turn_penalty_term,
)
from dxenv.reward.scoring import (
    SeverityTable,
    load_severity,
    score_bounds,
    terminal_diagnosis_score,
)
from dxenv.reward.shaping import max_shaping_gain, shaping_terms
from dxenv.reward.treatment import (
    TreatmentConfig,
    load_treatment_config,
    treatment_score,
)
from dxenv.reward.verify import verify_term

_CONFIG_DIR: Final = Path(__file__).resolve().parents[1] / "configs"


class RewardError(ValueError):
    """Malformed trajectory or config. Never caught inside `dxenv.reward`."""


class InvariantViolation(AssertionError):
    """An invariant failed at scoring time. Always a bug; never clipped away [I11]."""


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The hidden truth for one patient. Reward may see all of this; observations may not."""

    condition: str
    analytes: dict[str, ResultValue]
    allergies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RewardConfig:
    lam: float
    mu: float
    abstain_value: float
    abstain_penalty: float
    shaping_enabled: bool
    shaping_gamma: float
    shaping_scale: float
    verify_enabled: bool
    verify_fraction: float
    coherence_scale: float
    correctness_scale: float
    contraindication_penalty: float
    severity: SeverityTable
    costs: CostTable
    treatments: TreatmentConfig
    raw: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        payload = {
            k: getattr(self, k)
            for k in (
                "lam", "mu", "abstain_value", "abstain_penalty", "shaping_enabled",
                "shaping_gamma", "shaping_scale", "verify_enabled", "verify_fraction",
                "coherence_scale", "correctness_scale", "contraindication_penalty",
            )
        }
        payload["severity"] = dict(sorted(self.severity.weights.items()))
        payload["costs"] = dict(sorted(self.costs.prices.items()))
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()


def load_reward_config(path: Path | None = None) -> RewardConfig:
    with (path or _CONFIG_DIR / "reward.yaml").open() as fh:
        raw = yaml.safe_load(fh)
    cfg = RewardConfig(
        lam=float(raw["costs"]["lambda"]),
        mu=float(raw["costs"]["mu"]),
        abstain_value=float(raw["abstain"]["value"]),
        abstain_penalty=float(raw["abstain"]["penalty"]),
        shaping_enabled=bool(raw["shaping"]["enabled"]),
        shaping_gamma=float(raw["shaping"]["gamma"]),
        shaping_scale=float(raw["shaping"]["scale"]),
        verify_enabled=bool(raw["verify"]["enabled"]),
        verify_fraction=float(raw["verify"]["fraction_of_cost"]),
        coherence_scale=float(raw["treatment"]["coherence_scale"]),
        correctness_scale=float(raw["treatment"]["correctness_scale"]),
        contraindication_penalty=float(raw["treatment"]["contraindication_penalty"]),
        severity=load_severity(),
        costs=load_cost_table(),
        treatments=load_treatment_config(),
        raw=raw,
    )
    validate_reward_config(cfg)
    return cfg


def validate_reward_config(cfg: RewardConfig, taxonomy: Taxonomy | None = None) -> None:
    """Refuse any config under which a test order could be net-positive [I5].

    This is the guard on the invariant most likely to be violated by a well-meaning
    future edit. It is arithmetic, not a sample-based test, so it holds for every state
    rather than for the states a test happened to visit.
    """
    tax = taxonomy or load_taxonomy()
    if cfg.lam < 0.0 or cfg.mu < 0.0:
        raise RewardError("lambda and mu must be non-negative")
    if cfg.contraindication_penalty < 0.0:
        raise RewardError("contraindication penalty must be non-negative")

    if not 0.0 <= cfg.verify_fraction < 1.0:
        raise RewardError(
            f"verify.fraction_of_cost must be in [0, 1); got {cfg.verify_fraction}. "
            "At 1.0 a correct prediction exactly cancels the order's cost [I5]."
        )

    # The verify term is a fraction of each order's OWN cost, so cost + verify is
    # strictly negative for every test individually. Only shaping can break I5, and only
    # if its scale is large relative to the cheapest order.
    floor_cost = cfg.lam * cfg.costs.cheapest
    gain = max_shaping_gain(len(tax), cfg.shaping_scale) if cfg.shaping_enabled else 0.0
    if gain >= floor_cost * (1.0 - cfg.verify_fraction):
        raise RewardError(
            f"I5 VIOLATION IN CONFIG: one transition can gain up to {gain:.5f} from "
            f"shaping, but the cheapest order nets only "
            f"{floor_cost * (1.0 - cfg.verify_fraction):.5f} after its verify credit. "
            f"Under this config an agent profits from ordering tests it does not need. "
            f"Either lower shaping.scale to below "
            f"{floor_cost * (1.0 - cfg.verify_fraction) / max(np.log(len(tax)), 1e-9):.6f}, "
            f"raise costs.lambda, or set shaping.enabled: false. See the long comment in "
            f"configs/reward.yaml -- at a lambda calibrated for an interior optimum, "
            f"meaningful shaping and strict per-step I5 are not simultaneously achievable."
        )


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Every component, kept separate. Aggregate-only rewards hide which term got gamed."""

    diagnosis: float
    test_cost: float
    turn_penalty: float
    treatment: float
    shaping: float
    verify: float
    total: float
    n_tests_charged: int
    n_turns: int
    termination_reason: str
    contraindications: tuple[tuple[str, str, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "test_cost": self.test_cost,
            "turn_penalty": self.turn_penalty,
            "treatment": self.treatment,
            "shaping": self.shaping,
            "verify": self.verify,
            "total": self.total,
            "n_tests_charged": self.n_tests_charged,
            "n_turns": self.n_turns,
            "termination_reason": self.termination_reason,
            "contraindications": [list(v) for v in self.contraindications],
        }


def _step_value(result: dict[str, Any]) -> ResultValue:
    if result.get("value_number") is not None:
        return float(result["value_number"])
    code = result.get("value_code")
    if code is None:
        raise RewardError(f"result {result.get('analyte')!r} has neither a number nor a code")
    return str(code)


def score_trajectory(
    trajectory: dict[str, Any],
    ground_truth: GroundTruth,
    config: RewardConfig,
    taxonomy: Taxonomy | None = None,
    catalog: Catalog | None = None,
    model: ObservationModel | None = None,
) -> RewardBreakdown:
    """Score one episode. Pure in (trajectory, ground_truth, config)."""
    tax = taxonomy or load_taxonomy()
    cat = catalog or load_catalog()
    obs_model = model or build_observation_model()

    steps = trajectory["steps"]
    n_turns = len(steps)

    test_cost = 0.0
    verify_total = 0.0
    n_charged = 0
    prescribed: list[str] = []
    declared: dict[str, float] = {}
    ordered_evidence: list[tuple[str, ResultValue]] = []

    # s_0 already includes the free vitals; shaping must start from what the agent
    # actually knew, or the first test appears to explain the whole prior.
    for vk in cat.vital_keys:
        if vk in ground_truth.analytes:
            ordered_evidence.append((vk, ground_truth.analytes[vk]))
    n_initial = len(ordered_evidence)

    for step in steps:
        action = step["action"]
        kind = action["kind"]
        if kind == "order_test":
            key = action["test_key"]
            charged = float(step.get("cost_charged", 0.0)) > 0.0
            if charged:
                test_cost += order_cost_term(key, config.lam, config.costs)
                n_charged += 1
            revealed = {
                r["analyte"]: _step_value(r) for r in step.get("revealed", [])
            }
            if charged:
                for a in cat.test(key).analytes:
                    if a in revealed:
                        ordered_evidence.append((a, revealed[a]))
            if config.verify_enabled and charged:
                verify_total += verify_term(
                    key,
                    action["prediction"],
                    revealed,
                    order_cost=config.lam * config.costs.price(key),
                    fraction=config.verify_fraction,
                    catalog=cat,
                )
        elif kind == "prescribe":
            prescribed.append(action["treatment_key"])
        elif kind == "diagnose":
            declared = {str(k): float(v) for k, v in action["distribution"].items()}
        elif kind == "abstain":
            declared = {}
        else:
            raise RewardError(f"unknown action kind {kind!r} in trajectory")

    reason = trajectory.get("termination_reason", "max_turns")

    if reason == "diagnose":
        if not declared:
            raise RewardError("trajectory terminated on diagnose with no distribution")
        diagnosis = terminal_diagnosis_score(declared, ground_truth.condition, tax, config.severity)
    elif reason == "abstain":
        # Priced near the EV of a calibrated prior guess, minus a small penalty so that
        # abstaining on everything is strictly worse than reporting the prior.
        diagnosis = config.abstain_value - config.abstain_penalty
    else:
        # Ran out of turns or budget without committing. Treated as an abstention that
        # also failed to decide -- same value, same penalty. Scoring it as a wrong answer
        # would make running out of time worse than a confident guess, which teaches
        # recklessness near the horizon.
        diagnosis = config.abstain_value - config.abstain_penalty

    turn_pen = turn_penalty_term(n_turns, config.mu)

    treat, violations = treatment_score(
        prescribed=prescribed,
        declared=declared,
        true_condition=ground_truth.condition,
        analytes=ground_truth.analytes,
        allergies=ground_truth.allergies,
        coherence_scale=config.coherence_scale,
        correctness_scale=config.correctness_scale,
        contraindication_penalty=config.contraindication_penalty,
        cfg=config.treatments,
        catalog=cat,
    )

    shaping_total = 0.0
    if config.shaping_enabled and len(ordered_evidence) > n_initial:
        beliefs = posterior_sequence(ordered_evidence, model=obs_model)[n_initial:]
        shaping_total = float(
            sum(shaping_terms(beliefs, config.shaping_gamma, config.shaping_scale))
        )

    total = diagnosis + test_cost + turn_pen + treat + shaping_total + verify_total
    if not np.isfinite(total):
        raise InvariantViolation(
            f"reward is not finite (diagnosis={diagnosis}, cost={test_cost}, "
            f"turns={turn_pen}, treatment={treat}, shaping={shaping_total}, "
            f"verify={verify_total}). NaN is never clipped away [I11]."
        )

    return RewardBreakdown(
        diagnosis=diagnosis,
        test_cost=test_cost,
        turn_penalty=turn_pen,
        treatment=treat,
        shaping=shaping_total,
        verify=verify_total,
        total=total,
        n_tests_charged=n_charged,
        n_turns=n_turns,
        termination_reason=reason,
        contraindications=tuple(violations),
    )


def reward_bounds(config: RewardConfig, taxonomy: Taxonomy | None = None) -> tuple[float, float]:
    """Finite (min, max) the total reward can take. Used by the finiteness audit [I11]."""
    tax = taxonomy or load_taxonomy()
    lo_dx, hi_dx = score_bounds(tax, config.severity)
    worst_cost = -config.lam * sum(sorted(config.costs.prices.values()))
    max_shape = max_shaping_gain(len(tax), config.shaping_scale) if config.shaping_enabled else 0.0
    max_ver = (
        config.verify_fraction * config.lam * sum(config.costs.prices.values())
        if config.verify_enabled
        else 0.0
    )
    hi = hi_dx + config.coherence_scale + config.correctness_scale + max_shape + max_ver
    lo = (
        lo_dx
        + worst_cost
        - config.mu * 10_000
        - config.contraindication_penalty * len(config.costs.prices)
    )
    return lo, hi
