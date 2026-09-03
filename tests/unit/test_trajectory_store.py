"""The trajectory store (CLAUDE.md 4): persist every rollout, rescore for free."""

from __future__ import annotations

import pytest
from dxenv.data.store import (
    RunMeta,
    StoreError,
    TrajectoryStore,
    read_episodes,
    read_meta,
    rescore,
    stored_trajectory,
)
from dxenv.env.episode import DiagnosticEpisode
from dxenv.policy.baselines import GreedyBayesPolicy, run_episode


@pytest.fixture
def meta(episode_config, reward_config, menu, taxonomy) -> RunMeta:
    return RunMeta(
        run_id="test-run",
        env_config_hash=episode_config.hash(),
        reward_config_hash=reward_config.hash(),
        menu_fingerprint=menu.fingerprint(),
        taxonomy_hash=taxonomy.hash(),
    )


def _episodes(records, episode_config, catalog, n=5):
    out = []
    for i, rec in enumerate(records[:n]):
        traj = run_episode(
            DiagnosticEpisode(rec, seed=i, config=episode_config, catalog=catalog,
                              budget=200.0),
            GreedyBayesPolicy(),
        )
        out.append((rec, traj))
    return out


def test_store_round_trips(tmp_path, meta, fixture_corpus, episode_config, catalog) -> None:
    with TrajectoryStore(meta, root=tmp_path) as s:
        for rec, traj in _episodes(fixture_corpus, episode_config, catalog):
            s.append(traj, {"condition": rec.condition, "patient_id": rec.patient_id})
    read = list(read_episodes(tmp_path / "test-run" / "episodes.jsonl"))
    assert len(read) == 5
    assert read_meta(tmp_path / "test-run")["run_id"] == "test-run"


def _strip_declared_distribution(trajectory: dict) -> dict:
    """Blank out the agent's OWN reported distribution.

    A trajectory necessarily names conditions: the diagnose action IS a distribution over
    the label set. That is the agent talking, not the record leaking, and asserting "no
    condition string appears anywhere in the trajectory" would be asserting that the
    agent never diagnoses. What must hold is that nothing ELSE in the line identifies
    which of the 149 labels is true.
    """
    steps = []
    for step in trajectory["steps"]:
        action = dict(step["action"])
        if action.get("kind") == "diagnose":
            action["distribution"] = {}
        steps.append({**step, "action": action})
    return {**trajectory, "steps": steps}


def test_stored_trajectory_excludes_ground_truth(
    tmp_path, meta, fixture_corpus, episode_config, catalog, taxonomy
) -> None:
    """The model-safe projection must not identify the label [I1].

    An episodes.jsonl file is deliberately NOT safe to feed to a model -- ground truth
    lives on the line so the run is self-contained for rescoring. What must hold is that
    the accessor rollout code uses cannot reach it.
    """
    with TrajectoryStore(meta, root=tmp_path) as s:
        for rec, traj in _episodes(fixture_corpus, episode_config, catalog, n=30):
            s.append(traj, {"condition": rec.condition, "patient_id": rec.patient_id})
    for ep in read_episodes(tmp_path / "test-run" / "episodes.jsonl"):
        traj = stored_trajectory(ep)
        assert "ground_truth" not in traj and "condition" not in traj
        _assert_no_label_leak_in_trajectory(traj, ep.ground_truth["condition"], taxonomy)


def _assert_no_label_leak_in_trajectory(trajectory: dict, condition: str, taxonomy) -> None:
    """The trajectory analogue of `filter.assert_no_label_leak`, same two exemptions.

    Strings from the global catalog vocabulary are exempt because they are identical for
    every patient -- `depression_screen` is a menu item, and the fact that the agent
    ordered it says what the agent did, not what the patient has. A vocabulary that does
    not vary with the patient cannot encode which patient it is; `test_vocabulary_is_
    global` is what keeps that exemption honest. Matching is word-boundary for the same
    reason: without it the synonym "mi" fires inside every occurrence of "commit".
    """
    from dxenv.env.filter import global_vocabulary

    vocab = global_vocabulary()
    forms = [f for f in taxonomy.get(condition).leak_strings if len(f) >= 4]

    def walk(node) -> list[str]:
        if isinstance(node, str):
            return [node]
        if isinstance(node, dict):
            return [s for k, v in node.items() for s in (*walk(k), *walk(v))]
        if isinstance(node, list):
            return [s for v in node for s in walk(v)]
        return []

    for text in walk(_strip_declared_distribution(trajectory)):
        if text in vocab:
            continue
        haystack = f" {text.replace('_', ' ').lower()} "
        for form in forms:
            assert f" {form} " not in haystack, (
                f"{form!r} (condition {condition!r}) reached the model-safe projection "
                f"in the string {text!r}"
            )


def test_declared_distribution_shape_carries_no_information(
    tmp_path, meta, fixture_corpus, episode_config, catalog, taxonomy
) -> None:
    """The report names the SAME labels for every patient, so its key set says nothing.

    This is the other half of the claim above: the previous test exempts the declared
    distribution from the string check, so something has to show that the exemption is
    not a hole. If the key set varied with the truth, the sparsity pattern would leak the
    label exactly the way I4 exists to prevent for test results.
    """
    with TrajectoryStore(meta, root=tmp_path) as s:
        for rec, traj in _episodes(fixture_corpus, episode_config, catalog, n=20):
            s.append(traj, {"condition": rec.condition, "patient_id": rec.patient_id})
    seen = set()
    for ep in read_episodes(tmp_path / "test-run" / "episodes.jsonl"):
        for step in stored_trajectory(ep)["steps"]:
            if step["action"]["kind"] == "diagnose":
                seen.add(frozenset(step["action"]["distribution"]))
    assert len(seen) == 1, "the reported label set varies with the patient"
    assert next(iter(seen)) == frozenset(taxonomy.slugs)


def test_append_rejects_undeclared_env_config(
    tmp_path, meta, fixture_corpus, episode_config, catalog
) -> None:
    """A line generated under a config the run never declared is a bug, not a warning."""
    _, traj = _episodes(fixture_corpus, episode_config, catalog, n=1)[0]
    traj = {**traj, "config_hash": "0" * 64}
    with TrajectoryStore(meta, root=tmp_path) as s, pytest.raises(StoreError, match="declare"):
        s.append(traj, {"condition": "x", "patient_id": "y"})


def test_declared_extra_env_config_is_accepted(
    tmp_path, meta, fixture_corpus, episode_config, catalog
) -> None:
    """A curriculum stage changes max_turns, hence the hash. Declared, so it is allowed."""
    other = "1" * 64
    declared = RunMeta(**{**meta.as_dict(), "env_config_hashes": (other,)})
    _, traj = _episodes(fixture_corpus, episode_config, catalog, n=1)[0]
    with TrajectoryStore(declared, root=tmp_path) as s:
        s.append({**traj, "config_hash": other}, {"condition": "x", "patient_id": "y"})
    assert len(list(read_episodes(tmp_path / "test-run" / "episodes.jsonl"))) == 1


def test_reopening_with_drifted_hashes_raises(
    tmp_path, meta, fixture_corpus, episode_config, catalog
) -> None:
    """Mixing two configs into one file makes the file uninterpretable. Fail loudly."""
    with TrajectoryStore(meta, root=tmp_path) as s:
        _, traj = _episodes(fixture_corpus, episode_config, catalog, n=1)[0]
        s.append(traj, {"condition": "x", "patient_id": "y"})
    drifted = RunMeta(**{**meta.as_dict(), "reward_config_hash": "deadbeef"})
    with pytest.raises(StoreError, match="different pinned hashes"):
        TrajectoryStore(drifted, root=tmp_path).open()


def test_malformed_line_raises_rather_than_being_skipped(tmp_path, meta) -> None:
    """Test the detector. Silently dropping 3% of a run produces a selected mean."""
    root = tmp_path / "test-run"
    TrajectoryStore(meta, root=tmp_path).open().close()
    (root / "episodes.jsonl").write_text('{"trajectory": {}, "ground_truth": {}}\nnot json\n')
    with pytest.raises(StoreError, match="not valid JSON"):
        list(read_episodes(root / "episodes.jsonl"))


def test_missing_required_key_raises(tmp_path, meta) -> None:
    root = tmp_path / "test-run"
    TrajectoryStore(meta, root=tmp_path).open().close()
    (root / "episodes.jsonl").write_text('{"trajectory": {}}\n')
    with pytest.raises(StoreError, match="missing"):
        list(read_episodes(root / "episodes.jsonl"))


def test_rescoring_a_stored_run_reproduces_the_original_scores(
    tmp_path, meta, fixture_corpus, episode_config, reward_config, catalog
) -> None:
    """[I8] Reward is pure, so a stored corpus rescores to exactly what it scored."""
    from dxenv.reward.engine import GroundTruth, score_trajectory

    truth = {}
    with TrajectoryStore(meta, root=tmp_path) as s:
        for rec, traj in _episodes(fixture_corpus, episode_config, catalog, n=10):
            truth[rec.patient_id] = rec
            s.append(traj, {"condition": rec.condition, "patient_id": rec.patient_id})

    def score(trajectory, gt):
        rec = truth[gt["patient_id"]]
        return score_trajectory(
            trajectory, GroundTruth(gt["condition"], rec.analytes, rec.allergies),
            reward_config,
        ).total

    a = rescore(tmp_path / "test-run" / "episodes.jsonl", score)
    b = rescore(tmp_path / "test-run" / "episodes.jsonl", score)
    assert a == b and len(a) == 10
