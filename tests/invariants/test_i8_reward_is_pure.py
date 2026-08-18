"""I8: reward is a pure function of (trajectory, ground_truth, reward_config)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import dxenv.reward.engine as engine_mod
import pytest
from dxenv.reward.engine import GroundTruth, score_trajectory

FORBIDDEN_CALLS = {"random", "randint", "shuffle", "time", "now", "today", "monotonic"}


def _traj(rec, menu, episode_config, catalog):
    from dxenv.env.actions import ActionKind, action_id
    from dxenv.env.episode import DiagnosticEpisode
    from dxenv.env.schemas import Diagnose, OrderTest, Prescribe

    ep = DiagnosticEpisode(rec, seed=1, config=episode_config, menu=menu, catalog=catalog,
                           budget=200.0)
    ep.reset()
    ep.step(OrderTest(action_id=menu.id_for_test("cbc"), test_key="cbc", prediction="normal"))
    ep.step(OrderTest(action_id=menu.id_for_test("troponin"), test_key="troponin",
                      prediction="high"))
    ep.step(Prescribe(action_id=menu.id_for_treatment("aspirin"), treatment_key="aspirin"))
    ep.step(Diagnose(action_id=action_id(ActionKind.DIAGNOSE, "diagnose"),
                     distribution={rec.condition: 0.7, "migraine": 0.3}
                     if rec.condition != "migraine" else {"migraine": 1.0}))
    return ep.trajectory()


def test_same_input_gives_identical_output(fixture_corpus, menu, episode_config, catalog,
                                           reward_config) -> None:
    for rec in fixture_corpus[:25]:
        traj = _traj(rec, menu, episode_config, catalog)
        gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
        a = score_trajectory(traj, gt, reward_config)
        b = score_trajectory(traj, gt, reward_config)
        assert a.as_dict() == b.as_dict()


def test_scoring_does_not_mutate_its_inputs(fixture_corpus, menu, episode_config, catalog,
                                            reward_config) -> None:
    import copy

    rec = fixture_corpus[0]
    traj = _traj(rec, menu, episode_config, catalog)
    gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
    before = copy.deepcopy(traj)
    score_trajectory(traj, gt, reward_config)
    assert traj == before


def test_rescoring_under_new_weights_changes_only_the_weighted_terms(
    fixture_corpus, menu, episode_config, catalog, reward_config
) -> None:
    """Rescoring a stored corpus under new weights must be free and correct.

    You will change these weights repeatedly; this is the property that makes that cheap.
    """
    import dataclasses

    rec = fixture_corpus[0]
    traj = _traj(rec, menu, episode_config, catalog)
    gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
    base = score_trajectory(traj, gt, reward_config)
    doubled = score_trajectory(
        traj, gt, dataclasses.replace(reward_config, lam=reward_config.lam * 2)
    )
    assert doubled.test_cost == pytest.approx(base.test_cost * 2)
    assert doubled.diagnosis == pytest.approx(base.diagnosis)


def test_engine_source_has_no_rng_or_clock() -> None:
    """Structural: no RNG, no clock, no I/O reachable from the scoring path."""
    src = Path(inspect.getfile(engine_mod)).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CALLS:
            raise AssertionError(f"engine.py references {node.attr!r}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in {"random", "time", "datetime"}, alias.name
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {"random", "time", "datetime"}, node.module


def _imported_modules(path: Path) -> set[str]:
    """Real imports only, via AST.

    Grepping the source would also flag a docstring that NAMES a module -- env/bayes.py
    documents that callers pass in `dxenv.reward.scoring.brier_score`, which is precisely
    the dependency inversion that keeps env/ clean. Prose about a module is not a
    dependency on it.
    """
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_reward_does_not_import_policy_or_train() -> None:
    """CLAUDE.md 3: reward/ must not import from policy/ or train/."""
    for path in Path("dxenv/reward").glob("*.py"):
        mods = _imported_modules(path)
        assert not any(m.startswith("dxenv.policy") for m in mods), f"{path} imports policy/"
        assert not any(m.startswith("dxenv.train") for m in mods), f"{path} imports train/"


def test_env_does_not_import_reward() -> None:
    """CLAUDE.md 3: env/ must not import reward/. This is what makes offline rescoring work."""
    for path in Path("dxenv/env").glob("*.py"):
        mods = _imported_modules(path)
        assert not any(m.startswith("dxenv.reward") for m in mods), f"{path} imports reward/"
