"""Shared fixtures.

The fast suite runs on a 100-patient fixture and must stay under 60s. Invariant tests
that CLAUDE.md marks as corpus-wide use `full_corpus` and are marked slow -- a leak that
appears in 2% of patients is still a leak, and a sample will miss it.
"""

from __future__ import annotations

import numpy as np
import pytest
from dxenv.data.corpus import PatientRecord, generate_corpus
from dxenv.data.taxonomy import Taxonomy, load_taxonomy
from dxenv.env.actions import ActionMenu, build_menu
from dxenv.env.catalog import Catalog, load_catalog
from dxenv.env.episode import EpisodeConfig, load_episode_config
from dxenv.env.obs_model import ObservationModel, build_observation_model
from dxenv.reward.engine import RewardConfig, load_reward_config

FIXTURE_SEED = 20260818
FIXTURE_N = 100
FULL_N = 2000


def orderable_results(rec, catalog: Catalog) -> dict[str, object]:
    """Every ORDERABLE analyte for a patient -- the maximally-revealed observation.

    Vitals are excluded because they are auto-revealed and the filter rejects them in
    `revealed`; passing them would test a code path no episode can reach.
    """
    return {k: rec.analytes[k] for k in catalog.analyte_keys}


def observe(rec, catalog, menu, revealed=None, turn=0, budget=100.0, turns_left=20):
    """Build an observation for a record, revealing `revealed` (default: everything)."""
    from dxenv.env.filter import build_observation

    results = orderable_results(rec, catalog) if revealed is None else revealed
    return build_observation(
        rec.view(), results, turn, budget, turns_left, menu.fingerprint()
    )


@pytest.fixture(scope="session")
def taxonomy() -> Taxonomy:
    return load_taxonomy()


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="session")
def menu() -> ActionMenu:
    return build_menu()


@pytest.fixture(scope="session")
def obs_model() -> ObservationModel:
    return build_observation_model()


@pytest.fixture(scope="session")
def episode_config() -> EpisodeConfig:
    return load_episode_config()


@pytest.fixture(scope="session")
def reward_config() -> RewardConfig:
    return load_reward_config()


@pytest.fixture(scope="session")
def fixture_corpus() -> list[PatientRecord]:
    """100 patients. Frozen seed, so failures are reproducible from the seed alone."""
    return generate_corpus(FIXTURE_N, seed=FIXTURE_SEED)


@pytest.fixture(scope="session")
def full_corpus() -> list[PatientRecord]:
    """Corpus-wide fixture for the invariant sweeps. Used only by slow-marked tests."""
    return generate_corpus(FULL_N, seed=FIXTURE_SEED + 1)


@pytest.fixture(scope="session")
def one_per_condition(taxonomy: Taxonomy) -> list[PatientRecord]:
    """Exactly one patient per condition -- guarantees every label is exercised.

    Sampling from the prior leaves the rare-severe tail almost untouched, which is
    precisely the tail the severity weights exist to protect.
    """
    from dxenv.data.corpus import generate_patient

    rng = np.random.default_rng(FIXTURE_SEED + 2)
    # Patient ids are OPAQUE, deliberately. `patient_ref` appears in the observation, so
    # an id derived from the condition is a label leak -- and the first version of this
    # fixture used f"one-{slug}", which test_no_label_string_for_every_condition duly
    # caught. Keep them opaque; Synthea's own ids are UUIDs for the same reason.
    return [
        generate_patient(f"one-{i:04d}", rng, condition=slug)
        for i, slug in enumerate(taxonomy.slugs)
    ]
