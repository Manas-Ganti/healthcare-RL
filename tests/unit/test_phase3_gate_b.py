"""Gate B pre-registration (CLAUDE.md 8), enforced the same way Gate A is.

A threshold chosen after seeing the result is not a threshold. The mechanical check that
the gate's commit precedes the results file is the only thing that makes it a gate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

GATE = Path("dxenv/configs/gate_b.yaml")
RESULTS = Path("runs/phase3/prompted_baseline.json")


def _git_commit_time(path: Path) -> int | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", str(path)],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(out.stdout.strip()) if out.stdout.strip() else None


def test_gate_b_config_exists_and_is_committed() -> None:
    assert GATE.exists(), "Gate B must be pre-registered before Phase 3 is measured"
    assert _git_commit_time(GATE) is not None, "gate_b.yaml is not committed"


def test_gate_b_thresholds_preregistered() -> None:
    gate_t = _git_commit_time(GATE)
    if gate_t is None:
        pytest.skip("not a git checkout")
    results_t = _git_commit_time(RESULTS)
    if results_t is None:
        return  # results not yet committed: the ordering cannot be violated
    assert gate_t < results_t, (
        f"gate_b.yaml was committed at {gate_t}, results at {results_t}. The gate must "
        "predate the measurement."
    )


def test_gate_b_declares_every_criterion_from_claude_md() -> None:
    """The four Gate B criteria, each with a number attached."""
    gate = yaml.safe_load(GATE.read_text())
    for key in (
        "pass_at_8_minus_pass_at_1_min",
        "mean_group_reward_std_min",
        "calibration_margin_min",
        "headroom_below_expected_ceiling_min",
    ):
        assert key in gate["thresholds"], f"gate is missing {key}"
        assert isinstance(gate["thresholds"][key], (int, float))


def test_gate_b_declares_failure_actions() -> None:
    """A gate without a declared failure action is a note."""
    gate = yaml.safe_load(GATE.read_text())
    assert gate["on_failure"]
    assert "Do NOT proceed to GRPO" in gate["on_failure"]["pass_at_8_not_above_pass_at_1"]
    # The instruction that matters most: no headroom is a suspected leak, not a success.
    assert "leak" in gate["on_failure"]["no_headroom_below_ceiling"].lower()


def test_gate_b_pass_bar_is_a_procedure_not_a_number() -> None:
    """A number written down today would be arbitrary or reverse-engineered from a pilot.

    Specifying the bar as "the vitals-only Bayes baseline on the same patients" is
    pre-registration that survives contact: it is deterministic, computed on the same
    data under the same reward config, and cannot be nudged after the fact.
    """
    gate = yaml.safe_load(GATE.read_text())
    assert "rule" in gate["pass_bar"]
    assert gate["pass_bar"]["policy"].startswith("dxenv.policy.baselines")


def test_gate_b_checker_evaluates_a_synthetic_result() -> None:
    """Test the checker, on a result constructed to pass and one constructed to fail."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_gate_b", "scripts/check_gate_b.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gate = yaml.safe_load(GATE.read_text())

    passing = {
        "k": 8,
        "gate_b_pass_bar": 0.5,
        "rows": [{
            "policy": "prompted",
            "mean_reward": 0.4,
            "per_patient_best": [1.0] * 80 + [0.0] * 20,
            "per_patient_first": [1.0] * 40 + [0.0] * 60,
            "group_stds": [0.3] * 100,
        }],
        "calibration_margin": 0.2,
        "mean_expected_ceiling": 1.0,
        "schema_valid_fraction": 1.0,
    }
    assert all(v["status"] == "PASS" for v in mod.evaluate(passing, gate))

    flat = {**passing, "rows": [{**passing["rows"][0], "group_stds": [0.0] * 100}]}
    verdicts = {v["criterion"]: v["passed"] for v in mod.evaluate(flat, gate)}
    assert not verdicts["within-group reward variance"]
    assert not verdicts["degenerate group fraction"]

    no_gap = {**passing,
              "rows": [{**passing["rows"][0], "per_patient_first": [1.0] * 80 + [0.0] * 20}]}
    assert not next(
        v for v in mod.evaluate(no_gap, gate) if v["criterion"] == "pass@k above pass@1"
    )["passed"]


def test_gate_b_checker_reports_missing_criteria_as_skipped() -> None:
    """A gate that silently evaluates half its criteria and prints a verdict is not a gate.

    The model-free sweep can answer every criterion, but a results file from an older
    run, or one produced by a different harness, may not -- and the difference between
    "passed" and "was never checked" has to survive into the output.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_gate_b", "scripts/check_gate_b.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gate = yaml.safe_load(GATE.read_text())

    bare = {
        "k": 8,
        "gate_b_pass_bar": 0.5,
        "rows": [{
            "policy": "prompted", "mean_reward": 0.4,
            "per_patient_best": [1.0] * 80 + [0.0] * 20,
            "per_patient_first": [1.0] * 40 + [0.0] * 60,
            "group_stds": [0.3] * 100,
        }],
    }
    statuses = {v["criterion"]: v["status"] for v in mod.evaluate(bare, gate)}
    assert statuses["calibration survived"] == "SKIP"
    assert statuses["headroom below ceiling"] == "SKIP"
    assert statuses["schema-valid output"] == "SKIP"
    assert statuses["pass@k above pass@1"] == "PASS"


def test_gate_b2_changes_no_substantive_threshold() -> None:
    """An amendment may fix a specification error; it may not move the goalposts."""
    gate2 = Path("dxenv/configs/gate_b2.yaml")
    if not gate2.exists():
        pytest.skip("no amendment recorded")
    gate, amend = yaml.safe_load(GATE.read_text()), yaml.safe_load(gate2.read_text())

    for key, value in amend["unchanged_from_gate_b"].items():
        assert gate["thresholds"][key] == value, (
            f"gate_b2 claims {key} is unchanged but gate_b says "
            f"{gate['thresholds'][key]} vs {value}"
        )
    # The substantive criteria -- the ones that decide whether GRPO has anything to
    # sharpen -- must not appear among the amended thresholds.
    for key in ("pass_at_8_minus_pass_at_1_min", "mean_group_reward_std_min",
                "calibration_margin_min", "headroom_below_expected_ceiling_min"):
        assert key not in amend["thresholds"], f"gate_b2 moved a substantive threshold: {key}"

    assert amend["amends"] == "gate_b.yaml"
    assert amend["what_was_wrong"]["original"] == gate["thresholds"]["schema_valid_fraction_min"]
    # Both verdicts must be recorded, not just the favourable one.
    assert amend["measured_2026_09_05"]["verdict_under_gate_b"] == "FAIL"
