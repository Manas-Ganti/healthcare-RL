"""I10: episodes are deterministic given (patient_id, seed, config_hash)."""

from __future__ import annotations

import pytest
from dxenv.data.corpus import generate_corpus
from dxenv.env.actions import ActionKind, action_id
from dxenv.env.episode import DiagnosticEpisode
from dxenv.env.schemas import Diagnose, OrderTest


def _run(rec, seed, config, menu, catalog):
    ep = DiagnosticEpisode(rec, seed=seed, config=config, menu=menu, catalog=catalog)
    ep.reset()
    for key, pred in (("cbc", "normal"), ("bmp", "high"), ("ecg", "normal_categorical")):
        if not ep.state.done:
            ep.step(OrderTest(action_id=menu.id_for_test(key), test_key=key, prediction=pred))
    if not ep.state.done:
        ep.step(Diagnose(action_id=action_id(ActionKind.DIAGNOSE, "diagnose"),
                         distribution={rec.condition: 1.0}))
    return ep.trajectory()


def test_episode_deterministic_under_seed(fixture_corpus, episode_config, menu, catalog) -> None:
    for rec in fixture_corpus[:20]:
        assert _run(rec, 42, episode_config, menu, catalog) == _run(
            rec, 42, episode_config, menu, catalog
        )


def test_different_seeds_can_differ(fixture_corpus, episode_config, menu, catalog) -> None:
    """Budget is drawn per episode, so the seed must actually do something."""
    rec = fixture_corpus[0]
    budgets = {
        DiagnosticEpisode(rec, seed=s, config=episode_config, menu=menu,
                          catalog=catalog).reset().remaining_budget
        for s in range(30)
    }
    assert len(budgets) > 1, "the seed has no effect; determinism is vacuous"


def test_corpus_generation_is_reproducible() -> None:
    a = generate_corpus(30, seed=555)
    b = generate_corpus(30, seed=555)
    assert [r.patient_id for r in a] == [r.patient_id for r in b]
    assert [r.condition for r in a] == [r.condition for r in b]
    assert [r.analytes for r in a] == [r.analytes for r in b]
    assert [r.allergies for r in a] == [r.allergies for r in b]


def test_config_hash_changes_when_config_changes(episode_config) -> None:
    import dataclasses

    other = dataclasses.replace(episode_config, max_turns=episode_config.max_turns + 1)
    assert other.hash() != episode_config.hash()


def test_config_hash_is_stable(episode_config) -> None:
    assert episode_config.hash() == episode_config.hash()


def test_trajectory_records_the_config_hash(fixture_corpus, episode_config, menu,
                                            catalog) -> None:
    """Without this, a stored rollout cannot be tied to the config that produced it."""
    traj = _run(fixture_corpus[0], 1, episode_config, menu, catalog)
    assert traj["config_hash"] == episode_config.hash()
    assert traj["menu_fingerprint"] == menu.fingerprint()


def test_budget_never_exceeded(fixture_corpus, episode_config, menu, catalog) -> None:
    """Property test over random policies: the ledger never goes negative."""
    import numpy as np

    rng = np.random.default_rng(9)
    keys = list(catalog.test_keys)
    for rec in fixture_corpus[:25]:
        ep = DiagnosticEpisode(rec, seed=int(rng.integers(1 << 30)), config=episode_config,
                               menu=menu, catalog=catalog)
        ep.reset()
        while not ep.state.done:
            key = str(rng.choice(keys))
            ep.step(OrderTest(action_id=menu.id_for_test(key), test_key=key,
                              prediction="normal"))
            assert ep.state.spent <= ep.state.budget + 1e-9
            assert ep.remaining_budget >= -1e-9


def test_remaining_budget_in_observation_matches_ledger(fixture_corpus, episode_config,
                                                        menu, catalog) -> None:
    rec = fixture_corpus[0]
    ep = DiagnosticEpisode(rec, seed=3, config=episode_config, menu=menu, catalog=catalog,
                           budget=300.0)
    obs = ep.reset()
    assert obs.remaining_budget == pytest.approx(ep.remaining_budget)
    for key in ("cbc", "ct_head", "urinalysis"):
        obs, _, _ = ep.step(OrderTest(action_id=menu.id_for_test(key), test_key=key,
                                      prediction="normal"))
        assert obs.remaining_budget == pytest.approx(ep.remaining_budget)


def test_max_turns_enforced(fixture_corpus, episode_config, menu, catalog) -> None:
    rec = fixture_corpus[0]
    ep = DiagnosticEpisode(rec, seed=3, config=episode_config, menu=menu, catalog=catalog,
                           budget=1e6)
    ep.reset()
    while not ep.state.done:
        ep.step(OrderTest(action_id=menu.id_for_test("cbc"), test_key="cbc",
                          prediction="normal"))
    assert ep.state.turn <= episode_config.max_turns
    assert ep.state.termination_reason == "max_turns"


def test_terminates_on_diagnose_and_abstain(fixture_corpus, episode_config, menu,
                                            catalog) -> None:
    from dxenv.env.schemas import Abstain

    rec = fixture_corpus[0]
    for action, expected in (
        (Diagnose(action_id=action_id(ActionKind.DIAGNOSE, "diagnose"),
                  distribution={rec.condition: 1.0}), "diagnose"),
        (Abstain(action_id=action_id(ActionKind.ABSTAIN, "abstain")), "abstain"),
    ):
        ep = DiagnosticEpisode(rec, seed=1, config=episode_config, menu=menu, catalog=catalog)
        ep.reset()
        _, done, _ = ep.step(action)
        assert done and ep.state.termination_reason == expected


def test_repeat_order_is_deduped(fixture_corpus, episode_config, menu, catalog) -> None:
    """The second order costs nothing AND returns the cached result.

    Without this the agent finds the cheapest test and spams it.
    """
    ep = DiagnosticEpisode(fixture_corpus[0], seed=1, config=episode_config, menu=menu,
                           catalog=catalog, budget=500.0)
    ep.reset()
    a = OrderTest(action_id=menu.id_for_test("cmp"), test_key="cmp", prediction="normal")
    ep.step(a)
    first = ep.steps[-1].revealed
    spent = ep.state.spent
    ep.step(a)
    assert ep.steps[-1].was_duplicate
    assert ep.steps[-1].cost_charged == 0.0
    assert ep.steps[-1].revealed == first
    assert ep.state.spent == spent


def test_stepping_after_termination_raises(fixture_corpus, episode_config, menu,
                                           catalog) -> None:
    from dxenv.env.episode import EpisodeError
    from dxenv.env.schemas import Abstain

    ep = DiagnosticEpisode(fixture_corpus[0], seed=1, config=episode_config, menu=menu,
                           catalog=catalog)
    ep.reset()
    ab = Abstain(action_id=action_id(ActionKind.ABSTAIN, "abstain"))
    ep.step(ab)
    with pytest.raises(EpisodeError, match="already terminated"):
        ep.step(ab)
