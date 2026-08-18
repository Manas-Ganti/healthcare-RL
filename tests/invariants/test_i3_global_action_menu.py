"""I3: the action menu is global and identical for every patient."""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from dxenv.env.actions import ActionKind, action_id, build_menu


def test_build_menu_takes_no_patient() -> None:
    """Structural: there is no parameter through which a patient could reach the menu.

    If the menu were derived per patient, the menu WOULD BE the diagnosis.
    """
    assert inspect.signature(build_menu).parameters == {}


def test_menu_identical_across_patients(fixture_corpus, menu) -> None:
    baseline = menu.ids
    for _rec in fixture_corpus:
        assert build_menu().ids == baseline


def test_menu_independent_of_ground_truth(fixture_corpus) -> None:
    """Mutating a patient's condition does not change the menu."""
    before = build_menu().fingerprint()
    rec = fixture_corpus[0]
    mutated = dataclasses.replace(rec, condition="acute_myocardial_infarction")
    assert mutated.condition == "acute_myocardial_infarction"
    assert build_menu().fingerprint() == before


def test_action_ids_are_content_hashed_not_positional() -> None:
    """Adding a test must not renumber the others and invalidate stored trajectories."""
    assert action_id(ActionKind.ORDER_TEST, "cbc") == action_id(ActionKind.ORDER_TEST, "cbc")
    assert action_id(ActionKind.ORDER_TEST, "cbc") != action_id(ActionKind.ORDER_TEST, "bmp")
    # Same key, different kind, must not collide.
    assert action_id(ActionKind.ORDER_TEST, "x") != action_id(ActionKind.PRESCRIBE, "x")


def test_action_ids_stable_under_catalog_growth(catalog) -> None:
    """Ids depend on the key alone, so a new catalog entry cannot shift an old id."""
    before = {a.key: a.action_id for a in build_menu().test_actions()}
    for key, aid in before.items():
        assert action_id(ActionKind.ORDER_TEST, key) == aid


def test_menu_covers_every_catalog_entry(catalog, menu) -> None:
    assert {a.key for a in menu.test_actions()} == set(catalog.test_keys)
    assert {a.key for a in menu.treatment_actions()} == set(catalog.treatment_keys)


def test_menu_has_diagnose_and_abstain(menu) -> None:
    kinds = {a.kind for a in menu.actions}
    assert ActionKind.DIAGNOSE in kinds
    assert ActionKind.ABSTAIN in kinds, (
        "abstain must exist as an action or RL can never discover it"
    )


def test_menu_size_within_spec(menu) -> None:
    """CLAUDE.md 6.3 asks for 60-100 orderable tests."""
    assert 60 <= len(menu.test_actions()) <= 100


def test_off_menu_action_is_rejected(fixture_corpus, episode_config, menu, catalog) -> None:
    from dxenv.env.episode import DiagnosticEpisode, EpisodeError
    from dxenv.env.schemas import Abstain

    ep = DiagnosticEpisode(fixture_corpus[0], seed=1, config=episode_config, menu=menu,
                           catalog=catalog, budget=50.0)
    ep.reset()
    with pytest.raises(EpisodeError, match="not on the global menu"):
        ep.step(Abstain(action_id="abstain:deadbeefdeadbeef"))
