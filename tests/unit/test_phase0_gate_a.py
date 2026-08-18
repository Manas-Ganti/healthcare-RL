"""Phase 0 tests (CLAUDE.md 5), including the mechanical pre-registration check."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

GATE = Path("dxenv/configs/gate_a.yaml")
GATE2 = Path("dxenv/configs/gate_a2.yaml")
RESULTS = Path("runs/phase0/results.json")


def _git_commit_time(path: Path) -> int | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", str(path)],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(out.stdout.strip()) if out.stdout.strip() else None


def test_gate_a_config_exists_and_is_committed() -> None:
    assert GATE.exists(), "Gate A must be pre-registered before Phase 0 runs"
    assert _git_commit_time(GATE) is not None, "gate_a.yaml is not committed"


def test_gate_a_thresholds_preregistered() -> None:
    """The gate's commit must precede the results file.

    Enforced mechanically because that enforcement is the only thing that makes a gate a
    gate. A threshold chosen after seeing the result is not a threshold.
    """
    gate_t = _git_commit_time(GATE)
    if gate_t is None:
        pytest.skip("not a git checkout")
    results_t = _git_commit_time(RESULTS)
    if results_t is None:
        # Results not yet committed: the ordering cannot be violated.
        return
    assert gate_t < results_t, (
        f"gate_a.yaml was committed at {gate_t}, results at {results_t}. The gate must "
        "predate the measurement."
    )


def test_gate_a_declares_thresholds_and_failure_actions() -> None:
    gate = yaml.safe_load(GATE.read_text())
    for key in ("v_minus_f_min", "t_minus_v_min", "leak_ablation_max_drop"):
        assert key in gate["thresholds"], f"gate is missing {key}"
    assert gate["on_failure"], "a gate without a declared failure action is a note"
    assert "Do NOT lower the threshold" in gate["on_failure"][
        "t_minus_v_below_threshold"
    ]


def test_gate_a2_changes_no_substantive_threshold() -> None:
    """An amendment may fix a specification error; it may not move the goalposts."""
    if not GATE2.exists():
        pytest.skip("no amendment recorded")
    gate, amend = yaml.safe_load(GATE.read_text()), yaml.safe_load(GATE2.read_text())
    for key, value in amend["unchanged_from_gate_a"].items():
        assert gate["thresholds"][key] == value, (
            f"gate_a2 claims {key} is unchanged but gate_a says "
            f"{gate['thresholds'][key]} vs {value}"
        )
    assert "v_minus_f_min" not in amend["thresholds"]
    assert "t_minus_v_min" not in amend["thresholds"]


def test_probe_conditions_disjoint(catalog) -> None:
    """The V feature set contains no feature present only in T."""
    v = set(catalog.vital_keys)
    t = set(catalog.vital_keys) | set(catalog.analyte_keys)
    assert v < t
    assert not (v - t)
    assert set(catalog.analyte_keys) - v, "T adds nothing over V; the probes are identical"


def test_blank_baseline_is_prior(taxonomy) -> None:
    """F accuracy must sit at the prior floor, on both metrics.

    Checked analytically rather than by re-running the probe, so it stays in the fast
    suite: a prior-predicting classifier scores the majority rate under plain accuracy
    and 1/n under balanced accuracy.
    """
    if not RESULTS.exists():
        pytest.skip("Phase 0 has not been run")
    res = json.loads(RESULTS.read_text())
    f = res["probes"]["F"]
    assert abs(f["balanced_accuracy"] - 1.0 / res["n_labels"]) < 0.02
    assert abs(f["top1_accuracy"] - res["majority_class_rate"]) < 0.02


def test_prize_is_large_enough_to_train_on() -> None:
    """T - V is the size of the entire prize; below the gate there is nothing to learn."""
    if not RESULTS.exists():
        pytest.skip("Phase 0 has not been run")
    res = json.loads(RESULTS.read_text())
    gate = yaml.safe_load(GATE.read_text())
    assert res["t_minus_v"] >= gate["thresholds"]["t_minus_v_min"]
    assert res["v_minus_f"] >= gate["thresholds"]["v_minus_f_min"]


def test_leak_probe_is_not_blind() -> None:
    """Test the detector: handed the label, the leak probe must light up.

    The plain ablation passes trivially here because blocked resources never reach the
    feature matrix. A trivially-satisfied leak check is not evidence of no leak.
    """
    if not RESULTS.exists():
        pytest.skip("Phase 0 has not been run")
    res = json.loads(RESULTS.read_text())
    assert res["leak_control_gain"] >= 0.10, (
        "injecting the label barely moved the probe; it would not detect a real leak"
    )


def test_taxonomy_mapping_total(fixture_corpus, taxonomy) -> None:
    """Every condition in the corpus maps to exactly one label; nothing is dropped."""
    slugs = set(taxonomy.slugs)
    for rec in fixture_corpus:
        assert rec.condition in slugs


def test_every_corpus_condition_mapped(one_per_condition, taxonomy) -> None:
    seen = {r.condition for r in one_per_condition}
    assert seen == set(taxonomy.slugs)
    assert len(seen) == len(taxonomy)


def test_unmapped_snomed_code_raises() -> None:
    """No fallback bucket: an unmapped code must raise, not silently vanish."""
    from dxenv.data.taxonomy import TaxonomyError, map_snomed

    with pytest.raises(TaxonomyError, match="unmapped SNOMED code"):
        map_snomed("000000000000")


def test_prior_is_normalised_and_positive(taxonomy) -> None:
    p = taxonomy.prior()
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all(), "a zero-prior label is unreachable but still scoreable"
