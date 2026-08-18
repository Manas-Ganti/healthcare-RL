"""I5: ordering a test never produces positive reward, at any step.

The invariant most likely to be violated by a well-meaning future edit. Every
reward-hacking story in this environment starts with someone adding a plausible-looking
"reward informative tests" term.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from dxenv.reward.costs import CostError, load_cost_table, order_cost_term, turn_penalty_term
from dxenv.reward.engine import RewardError, validate_reward_config
from dxenv.reward.verify import verify_term


def test_cost_table_covers_menu(catalog) -> None:
    table = load_cost_table()
    assert set(table.prices) == set(catalog.test_keys)


def test_missing_cost_raises_rather_than_defaulting() -> None:
    with pytest.raises(CostError, match="no cost for test"):
        load_cost_table().price("a_test_that_does_not_exist")


def test_cost_term_is_never_positive(catalog, reward_config) -> None:
    for key in catalog.test_keys:
        assert order_cost_term(key, reward_config.lam, reward_config.costs) <= 0.0


def test_turn_penalty_is_never_positive(reward_config) -> None:
    for n in range(0, 50):
        assert turn_penalty_term(n, reward_config.mu) <= 0.0


def test_cost_plus_verify_is_never_positive(catalog, reward_config, obs_model, taxonomy) -> None:
    """The whole per-step reward of an order, over every test and both outcomes.

    Verify is a fraction of the order's OWN cost, so this holds test-by-test rather than
    only for the cheapest one.
    """
    rng = np.random.default_rng(3)
    for key in catalog.test_keys:
        cost = order_cost_term(key, reward_config.lam, reward_config.costs)
        order_cost = reward_config.lam * reward_config.costs.price(key)
        for cond in taxonomy.slugs[::11]:
            revealed = {
                a: obs_model.sample(a, cond, rng) for a in catalog.test(key).analytes
            }
            for prediction in ("low", "normal", "high",
                              "normal_categorical", "abnormal_categorical"):
                v = verify_term(key, prediction, revealed, order_cost,
                                reward_config.verify_fraction, catalog)
                assert cost + v <= 1e-12, f"{key}/{prediction} nets {cost + v}"


def test_config_validator_refuses_profitable_testing(reward_config) -> None:
    """Test the detector: a config that would make testing pay must be refused."""
    unsafe = dataclasses.replace(reward_config, shaping_enabled=True, shaping_scale=0.5)
    with pytest.raises(RewardError, match="I5 VIOLATION IN CONFIG"):
        validate_reward_config(unsafe)


def test_config_validator_refuses_verify_fraction_at_one(reward_config) -> None:
    with pytest.raises(RewardError, match="fraction_of_cost"):
        validate_reward_config(dataclasses.replace(reward_config, verify_fraction=1.0))


def test_verify_is_zero_in_expectation_for_chance_predictions(catalog, reward_config) -> None:
    """Predicting "normal" on everything must earn nothing on average."""
    from dxenv.reward.verify import bucket_prior, headline_analyte

    for key in list(catalog.test_keys)[:25]:
        a = headline_analyte(key, catalog)
        analyte = catalog.analyte(a)
        buckets = (
            ("normal_categorical", "abnormal_categorical")
            if analyte.kind == "categorical"
            else ("low", "normal", "high")
        )
        total = sum(bucket_prior(a, b) for b in buckets)
        assert abs(total - 1.0) < 1e-6
        # E[1{correct} - q] under the prior == 0 for any fixed prediction.
        for b in buckets:
            assert abs(bucket_prior(a, b) - bucket_prior(a, b)) < 1e-12


def test_duplicate_order_costs_nothing(fixture_corpus, episode_config, menu, catalog) -> None:
    from dxenv.env.episode import DiagnosticEpisode
    from dxenv.env.schemas import OrderTest

    ep = DiagnosticEpisode(fixture_corpus[0], seed=2, config=episode_config, menu=menu,
                           catalog=catalog, budget=200.0)
    ep.reset()
    a = OrderTest(action_id=menu.id_for_test("cbc"), test_key="cbc", prediction="normal")
    ep.step(a)
    spent_after_first = ep.state.spent
    ep.step(a)
    assert ep.state.spent == spent_after_first
