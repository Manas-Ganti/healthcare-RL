"""Regenerate the frozen golden fixtures.

Run DELIBERATELY, never as part of a test run, and inspect the diff before committing.
A golden file that regenerates itself when it disagrees with the code is not a test --
it is a very slow way of asserting True.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dxenv.data.corpus import generate_corpus
from dxenv.data.taxonomy import load_taxonomy
from dxenv.env.actions import ActionKind, action_id, build_menu
from dxenv.env.bayes import posterior
from dxenv.env.episode import DiagnosticEpisode, load_episode_config
from dxenv.env.obs_model import build_observation_model
from dxenv.env.schemas import Diagnose, OrderTest, Prescribe
from dxenv.reward.engine import GroundTruth, load_reward_config, score_trajectory

GOLDEN = Path("tests/golden")
SEED = 314159


def main() -> None:
    tax, menu = load_taxonomy(), build_menu()
    ecfg, rcfg = load_episode_config(), load_reward_config()
    tcfg = rcfg.treatments
    model = build_observation_model()
    records = generate_corpus(5, seed=SEED)

    episodes = []
    for i, rec in enumerate(records):
        ep = DiagnosticEpisode(rec, seed=i, config=ecfg, menu=menu, budget=150.0)
        ep.reset()
        for key, pred in (("cbc", "normal"), ("troponin", "high"), ("ecg", "normal_categorical")):
            ep.step(OrderTest(action_id=menu.id_for_test(key), test_key=key, prediction=pred))

        evidence = {"presenting_complaint": rec.analytes["presenting_complaint"]}
        for step in ep.steps:
            for r in step.revealed:
                evidence[r.analyte] = (
                    r.value_number if r.value_number is not None else str(r.value_code)
                )
        belief = posterior(evidence, model)
        top = np.argsort(-belief)[:5]
        dist = {tax.slugs[int(j)]: float(belief[int(j)]) for j in top}
        total = sum(dist.values())
        dist = {k: v / total for k, v in dist.items()}

        # Prescribe COHERENTLY with what is about to be declared. Prescribing a fixed
        # drug regardless of the diagnosis leaves the treatment term at exactly zero for
        # every episode, and the golden then silently stops guarding that whole path.
        declared_dx = max(dist, key=lambda k: dist[k])
        drug = tcfg.first_line[declared_dx][0]
        ep.step(Prescribe(action_id=menu.id_for_treatment(drug), treatment_key=drug))
        ep.step(Diagnose(action_id=action_id(ActionKind.DIAGNOSE, "diagnose"),
                         distribution=dist))
        traj = ep.trajectory()
        gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
        breakdown = score_trajectory(traj, gt, rcfg)
        episodes.append({
            "patient_id": rec.patient_id,
            "condition": rec.condition,
            "trajectory": traj,
            "reward": breakdown.as_dict(),
        })

    (GOLDEN / "episodes.json").write_text(json.dumps(episodes, indent=2, sort_keys=True) + "\n")

    # Bayes solver goldens: the worked example from env/bayes.py's docstring.
    toy = {
        "prior": [0.5, 0.5],
        "likelihood_plus": [0.9, 0.2],
        "posterior_after_plus": [0.45 / 0.55, 0.10 / 0.55],
        "likelihood_plus_2": [0.3, 0.6],
        "posterior_after_both": [0.135 / 0.195, 0.060 / 0.195],
    }
    (GOLDEN / "bayes_toy.json").write_text(json.dumps(toy, indent=2) + "\n")

    fingerprints = {
        "label_set_hash": tax.hash(),
        "menu_fingerprint": menu.fingerprint(),
        "episode_config_hash": ecfg.hash(),
        "reward_config_hash": rcfg.hash(),
        "n_labels": len(tax),
        "n_tests": len(menu.test_actions()),
        "n_treatments": len(menu.treatment_actions()),
    }
    (GOLDEN / "fingerprints.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
    print(json.dumps(fingerprints, indent=2))


if __name__ == "__main__":
    main()
