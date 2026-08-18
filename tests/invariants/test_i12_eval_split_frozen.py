"""I12: the eval split is frozen and hash-verified; training never reads it."""

from __future__ import annotations

import json

import pytest
from dxenv.data.splits import (
    EvalLeakError,
    SplitError,
    assert_eval_frozen,
    freeze_eval_split,
    guard_training_access,
    make_splits,
)


@pytest.fixture
def splits(fixture_corpus):
    return make_splits(fixture_corpus, seed=7)


def test_splits_are_disjoint(splits) -> None:
    assert not set(splits.train) & set(splits.eval)
    assert not set(splits.train) & set(splits.holdout_modules)
    assert not set(splits.eval) & set(splits.holdout_modules)


def test_splits_are_deterministic(fixture_corpus) -> None:
    a = make_splits(fixture_corpus, seed=7)
    b = make_splits(fixture_corpus, seed=7)
    assert a.eval == b.eval and a.train == b.train


def test_holdout_systems_are_absent_from_train_and_eval(fixture_corpus, taxonomy) -> None:
    """Held-out systems must be removed BEFORE partitioning.

    Partitioning first would leave those patients in train, and the generalisation gap
    would then be measured against data the policy had already seen.
    """
    s = make_splits(fixture_corpus, seed=7)
    by_id = {r.patient_id: r for r in fixture_corpus}
    for pid in (*s.train, *s.eval):
        assert taxonomy.get(by_id[pid].condition).system not in s.holdout_systems


def test_eval_split_hash_matches(splits, tmp_path) -> None:
    path = tmp_path / "eval_split.json"
    digest = freeze_eval_split(splits, path)
    assert_eval_frozen(splits, path)
    assert json.loads(path.read_text())["eval_hash"] == digest


def test_drifted_eval_split_is_rejected(fixture_corpus, tmp_path) -> None:
    """Test the detector: a changed eval split must be caught, not silently accepted."""
    path = tmp_path / "eval_split.json"
    freeze_eval_split(make_splits(fixture_corpus, seed=7), path)
    with pytest.raises(SplitError, match="drifted"):
        assert_eval_frozen(make_splits(fixture_corpus, seed=8), path)


def test_missing_frozen_file_is_rejected(splits, tmp_path) -> None:
    with pytest.raises(SplitError, match="does not exist"):
        assert_eval_frozen(splits, tmp_path / "absent.json")


def test_training_never_reads_eval_split(splits) -> None:
    guard_training_access(list(splits.train), splits)
    with pytest.raises(EvalLeakError, match="eval patients"):
        guard_training_access([splits.eval[0]], splits)


def test_training_loader_is_guarded(fixture_corpus, splits, monkeypatch) -> None:
    """Monkeypatch the loader to raise on eval paths and confirm training survives.

    CLAUDE.md 9: the guard must be exercised, not merely present.
    """
    from dxenv.train import curriculum

    seen: list[str] = []

    def loader(ids: list[str]) -> list[str]:
        guard_training_access(ids, splits)
        seen.extend(ids)
        return ids

    assert curriculum.load_training_ids(list(splits.train), loader) == list(splits.train)
    assert seen == list(splits.train)
    with pytest.raises(EvalLeakError):
        curriculum.load_training_ids([*splits.train, splits.eval[0]], loader)


def test_eval_fraction_is_respected(fixture_corpus) -> None:
    s = make_splits(fixture_corpus, seed=1, eval_fraction=0.25)
    n = len(s.train) + len(s.eval)
    assert abs(len(s.eval) / n - 0.25) < 0.05


def test_bad_eval_fraction_raises(fixture_corpus) -> None:
    with pytest.raises(SplitError):
        make_splits(fixture_corpus, seed=1, eval_fraction=1.5)
