"""The GRPO2 gate: pre-registered, paired, and able to say no."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

GATE = Path("dxenv/configs/gate_grpo2.yaml")
RESULTS = Path("runs/phase3/grpo2_baseline.json")


def _checker():
    spec = importlib.util.spec_from_file_location("cg", "scripts/check_gate_grpo2.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_time(path: Path) -> int | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ct", "--", str(path)],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(out) if out else None


def test_gate_grpo2_preregistered() -> None:
    gate_t = _git_time(GATE)
    if gate_t is None:
        pytest.skip("gate not committed yet, or not a git checkout")
    results_t = _git_time(RESULTS)
    if results_t is None:
        return
    assert gate_t < results_t, "gate_grpo2.yaml must predate the measurement"


def _results(subject: list[float], prior: list[float], dx: float = 0.9) -> dict:
    def row(name, ppm, **kw):
        return {"policy": name, "per_patient_mean": ppm, "schema_valid_fraction": 1.0,
                "calibration_margin": 0.5, **kw}
    return {"rows": [
        row("grpo2", subject, ending_fractions={"diagnose": dx, "abstain": 1 - dx}),
        row("prior", prior),
        row("vitals_bayes", [p + 0.7 for p in prior]),
    ]}


def test_gate_fails_on_noise() -> None:
    """A policy equal to the prior plus zero-mean noise must not pass."""
    cg, gate = _checker(), yaml.safe_load(GATE.read_text())
    rng = np.random.default_rng(1)
    prior = list(rng.normal(0, 0.3, 200))
    subj = [p + e for p, e in zip(prior, rng.normal(0, 0.8, 200), strict=True)]
    v = {x["criterion"]: x for x in cg.evaluate(_results(subj, prior), gate)}
    assert not v["beats_prior"]["passed"]


def test_gate_passes_a_clear_win_and_fails_a_hedged_one() -> None:
    cg, gate = _checker(), yaml.safe_load(GATE.read_text())
    prior = [0.0] * 200
    subj = [0.3 + 0.1 * ((i % 7) - 3) for i in range(200)]
    v = {x["criterion"]: x for x in cg.evaluate(_results(subj, prior), gate)}
    assert v["beats_prior"]["passed"]
    hedged = {x["criterion"]: x for x in cg.evaluate(_results(subj, prior, dx=0.2), gate)}
    assert not hedged["diagnose_fraction"]["passed"], "a win by not committing is not a win"
