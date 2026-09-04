"""SFT dataset construction (CLAUDE.md 8.4).

Label targets with the Bayes posterior, not one-hot; seed abstention explicitly;
deliberately undertrain.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from dxenv.env.bayes import entropy, posterior
from dxenv.policy.baselines import evidence_from_observation
from dxenv.policy.decoding import parse_action
from dxenv.policy.sft import (
    DEFAULT_ENTROPY_FLOOR,
    SFTConfig,
    SFTDataset,
    SFTError,
    build_examples,
    seed_abstentions,
    soft_label_wire,
)
from dxenv.policy.teacher import deleak, filter_traces, privileged_trace

N = 60


@pytest.fixture(scope="module")
def traces(fixture_corpus):
    deleaked = [
        deleak(privileged_trace(r, seed=i, budget=150.0))
        for i, r in enumerate(fixture_corpus[:N])
    ]
    clean, _ = filter_traces(deleaked)
    return clean


@pytest.fixture(scope="module")
def dataset(traces):
    return SFTDataset(build_examples(traces) + seed_abstentions(traces, fraction=0.08))


def test_soft_labels_match_posterior(traces, dataset, taxonomy, obs_model) -> None:
    """The NAMED probabilities are the posterior's, exactly, recomputed not copied.

    Stated at the right precision. The target is the posterior's top-k plus a
    max-entropy completion of the tail, because the grammar caps how many labels a report
    may name -- so "the target is the posterior" is true of the head and false of the
    tail. What must hold exactly is the head; what must hold of the whole is that the
    divergence is bounded by the mass the report did not name.
    """
    by_patient = {t.patient_id: t for t in traces}
    checked = 0
    for example in dataset.examples:
        if example.kind != "diagnose":
            continue
        trace = by_patient[example.patient_id]
        turn = next(t for t in trace.turns if t.turn == example.turn)
        belief = posterior(evidence_from_observation(turn.observation), obs_model)
        named = [d["condition"] for d in json.loads(example.completion)["diagnosis"]]
        got = example.distribution(taxonomy)

        # (1) the head is the posterior, to rounding.
        for slug in named:
            assert abs(got[slug] - float(belief[taxonomy.index(slug)])) < 1e-5, slug
        # (2) the head is the posterior's top-k, in order.
        top = [taxonomy.slugs[int(i)] for i in np.argsort(-belief)[: len(named)]]
        assert named == top
        # (3) the divergence from the posterior is bounded by the unnamed mass.
        tail = 1.0 - float(sum(belief[taxonomy.index(s)] for s in named))
        tv = 0.5 * sum(
            abs(got.get(s, 0.0) - float(belief[taxonomy.index(s)])) for s in taxonomy.slugs
        )
        # Slack covers the per-label rounding in the emitted JSON; the bound that matters
        # is the tail, which dominates it by orders of magnitude whenever it is non-zero.
        assert tv <= tail + 1e-6, f"{example.patient_id}: TV {tv:.6f} exceeds tail {tail:.6f}"
        checked += 1
    assert checked > 0


def test_unnamed_tail_is_max_entropy(dataset, taxonomy) -> None:
    """The completion spreads the residual uniformly -- it does not invent an ordering."""
    for example in dataset.examples:
        if example.kind != "diagnose":
            continue
        named = {d["condition"] for d in json.loads(example.completion)["diagnosis"]}
        dist = example.distribution(taxonomy)
        tail = [v for k, v in dist.items() if k not in named]
        if not tail:
            continue  # the named mass reached 1.0; there is no residual to place
        assert max(tail) - min(tail) < 1e-9


def test_soft_labels_normalize(dataset, taxonomy) -> None:
    for example in dataset.examples:
        if example.kind != "diagnose":
            continue
        dist = example.distribution(taxonomy)
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert all(v >= 0.0 for v in dist.values())
        # A subset, not equality: when the named mass already reaches 1.0 there is no
        # residual to place and the report names only what it named.
        assert set(dist) <= set(taxonomy.slugs)


def test_sft_targets_not_onehot(dataset) -> None:
    """Entropy floor. One-hot targets destroy calibration before RL starts."""
    assert dataset.mean_target_entropy() > DEFAULT_ENTROPY_FLOOR


def test_abstain_present_in_sft_set(dataset) -> None:
    """An action the SFT set never contains is one RL never samples."""
    assert dataset.kind_counts().get("abstain", 0) > 0
    assert dataset.abstain_fraction() >= 0.02


def test_abstentions_are_seeded_from_ambiguous_cases(traces, obs_model) -> None:
    """Abstaining at random teaches the model to abstain at random.

    The seeded cases must be the high-entropy tail, not a uniform sample.
    """
    seeded = seed_abstentions(traces, fraction=0.1)
    picked = {e.patient_id for e in seeded}
    entropies = {
        t.patient_id: entropy(posterior(evidence_from_observation(t.turns[0].observation),
                                        obs_model))
        for t in traces
    }
    chosen = [entropies[p] for p in picked]
    rest = [h for p, h in entropies.items() if p not in picked]
    assert min(chosen) >= max(rest) - 1e-9, "seeded abstentions are not the ambiguous cases"


def test_every_completion_parses_as_a_legal_action(dataset, menu, taxonomy) -> None:
    """The SFT targets must be exactly what the grammar admits at rollout time."""
    for example in dataset.examples:
        action = parse_action(example.completion, menu, taxonomy)
        assert action.action_id in menu.ids


def test_examples_carry_no_assistant_turn_in_the_prompt(dataset) -> None:
    for example in dataset.examples:
        assert [m["role"] for m in example.messages] == ["system", "user"]


def test_build_examples_refuses_a_privileged_trace(fixture_corpus) -> None:
    """Reaching this with privilege intact means `deleak` was skipped. Fail loudly."""
    trace = privileged_trace(fixture_corpus[0], seed=0, budget=150.0)
    with pytest.raises(SFTError, match="still privileged"):
        build_examples([trace])


def test_validate_rejects_collapsed_targets(dataset, taxonomy) -> None:
    """Test the detector: a one-hot dataset must be refused, not merely reported."""
    from dataclasses import replace

    collapsed = SFTDataset([
        replace(e, target_entropy=0.0) if e.kind == "diagnose" else e
        for e in dataset.examples
    ])
    with pytest.raises(SFTError, match="collapsed toward one-hot"):
        collapsed.validate()


def test_validate_rejects_a_set_with_no_abstentions(traces) -> None:
    no_abstain = SFTDataset(build_examples(traces))
    with pytest.raises(SFTError, match="abstentions"):
        no_abstain.validate()


def test_soft_label_wire_names_the_top_k(taxonomy) -> None:
    belief = np.zeros(len(taxonomy))
    belief[3] = 0.6
    belief[7] = 0.4
    wire = soft_label_wire(belief, "because", taxonomy, max_labels=2)
    assert [d["condition"] for d in wire["diagnosis"]] == [taxonomy.slugs[3], taxonomy.slugs[7]]


def test_dataset_round_trips_through_jsonl(dataset, tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    n = dataset.write_jsonl(path)
    back = SFTDataset.read_jsonl(path)
    assert len(back) == n
    assert back.summary() == dataset.summary()
    for line in path.read_text().splitlines():
        json.loads(line)


def test_sft_config_refuses_to_overtrain() -> None:
    """A model sharpened onto its SFT set gives GRPO zero advantage to work with."""
    with pytest.raises(SFTError, match="epochs"):
        SFTConfig(epochs=5.0)


def test_trl_config_drops_optional_fields_loudly(capsys) -> None:
    """TRL removes and renames these; on the cluster even warmup_ratio is gone."""
    import inspect
    from pathlib import Path

    from dxenv.policy.sft import SFTConfig, _trl_config

    class TrimmedTRLConfig:
        # Only the signature is inspected.
        def __init__(self, output_dir=None, num_train_epochs=None, learning_rate=None, per_device_train_batch_size=None, gradient_accumulation_steps=None, max_seq_length=None, seed=None):  # noqa: ARG002, E501
            self.output_dir, self.max_seq_length = output_dir, max_seq_length

    TrimmedTRLConfig.__signature__ = inspect.signature(TrimmedTRLConfig.__init__)

    built = _trl_config(TrimmedTRLConfig, SFTConfig(output_dir=Path("/tmp/x")), bf16=False)
    out = capsys.readouterr().out
    assert "accepts:" in out and "dropping unsupported" in out
    # The older spelling is used rather than silently losing the sequence cap.
    assert built.max_seq_length == SFTConfig().max_seq_len
    # Losing the loss restriction changes WHAT is trained, so it is called out separately.
    assert "WARNING" in out and "completion" in out


def test_trl_config_refuses_to_lose_a_field_that_defines_the_run() -> None:
    """Epochs and learning rate ARE 'deliberately undertrained' (CLAUDE.md 8.4).

    Falling back to TRL's defaults for those would produce an adapter that looks trained
    and is not the one intended -- and unlike a crash, nothing downstream would say so.
    """
    import inspect
    from pathlib import Path

    from dxenv.policy.sft import SFTConfig, SFTError, _trl_config

    class NoLearningRate:
        def __init__(self, output_dir=None, num_train_epochs=None, per_device_train_batch_size=None, gradient_accumulation_steps=None):  # noqa: E501
            pass

    NoLearningRate.__signature__ = inspect.signature(NoLearningRate.__init__)

    with pytest.raises(SFTError, match="learning_rate"):
        _trl_config(NoLearningRate, SFTConfig(output_dir=Path("/tmp/x")), bf16=False)


def test_sft_rows_are_prompt_completion_not_bare_messages(dataset) -> None:
    """`completion_only_loss` applies to prompt-completion datasets.

    With a conversational messages-only dataset TRL cannot tell where the prompt ends, so
    the flag is ignored or raises -- and training on the prompt would spend most of the
    gradient reproducing a 13k-character menu the model never has to generate.
    """
    rows = [
        {
            "prompt": [dict(m) for m in e.messages],
            "completion": [{"role": "assistant", "content": e.completion}],
        }
        for e in dataset.examples[:5]
    ]
    for row in rows:
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]
        assert [m["role"] for m in row["completion"]] == ["assistant"]
        assert row["completion"][0]["content"].startswith("{")
