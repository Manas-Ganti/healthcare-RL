"""The sampler must serve the weights the synced LoRA was trained against."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dxenv.train.grpo import GRPOConfig, TrainingError, assert_sampler_shares_base
from dxenv.train.merge import MARKER, MergeError, ensure_merged, merged_path_for


def _adapter(tmp_path: Path, weights: bytes = b"w1") -> Path:
    a = tmp_path / "sft" / "final"
    a.mkdir(parents=True)
    (a / "adapter_model.safetensors").write_bytes(weights)
    (a / "adapter_config.json").write_text("{}")
    return a


def _fake_merge(calls: list):
    def merge(base: str, adapter: Path, out: Path) -> None:
        calls.append((base, adapter))
        (out / "model.safetensors").write_bytes(b"merged")
    return merge


def test_merges_once_then_reuses(tmp_path) -> None:
    a, calls = _adapter(tmp_path), []
    out = ensure_merged("Qwen/x", a, _fake_merge(calls))
    assert out == merged_path_for(a) and (out / MARKER).exists()
    assert ensure_merged("Qwen/x", a, _fake_merge(calls)) == out
    assert len(calls) == 1


def test_changed_adapter_is_refused_not_reused(tmp_path) -> None:
    a = _adapter(tmp_path)
    ensure_merged("Qwen/x", a, _fake_merge([]))
    (a / "adapter_model.safetensors").write_bytes(b"retrained")
    with pytest.raises(MergeError, match="different base or adapter"):
        ensure_merged("Qwen/x", a, _fake_merge([]))


def test_changed_base_is_refused(tmp_path) -> None:
    a = _adapter(tmp_path)
    ensure_merged("Qwen/x", a, _fake_merge([]))
    with pytest.raises(MergeError, match="different base or adapter"):
        ensure_merged("Qwen/y", a, _fake_merge([]))


def test_half_written_merge_is_refused(tmp_path) -> None:
    a = _adapter(tmp_path)
    out = merged_path_for(a)
    out.mkdir()
    (out / "model-00001.safetensors").write_bytes(b"partial")
    with pytest.raises(MergeError, match="half-written"):
        ensure_merged("Qwen/x", a, _fake_merge([]))


def test_interrupted_merge_leaves_no_marker(tmp_path) -> None:
    a = _adapter(tmp_path)

    def boom(base, adapter, out):
        (out / "model.safetensors").write_bytes(b"partial")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ensure_merged("Qwen/x", a, boom)
    assert not (merged_path_for(a) / MARKER).exists()


def test_marker_records_provenance(tmp_path) -> None:
    a = _adapter(tmp_path)
    m = json.loads((ensure_merged("Qwen/x", a, _fake_merge([])) / MARKER).read_text())
    assert m["base_model"] == "Qwen/x" and len(m["adapter_sha256"]) == 64


def test_trainer_refuses_unmerged_adapter_with_a_sampler(tmp_path) -> None:
    """The detector for runs 1-2: SFT merged into the trainer but not the sampler."""
    cfg = GRPOConfig(reference_adapter=tmp_path / "sft")
    with pytest.raises(TrainingError, match="wrong weights"):
        assert_sampler_shares_base(cfg, backend=object())
    assert_sampler_shares_base(GRPOConfig(reference_adapter=None), backend=object())
    assert_sampler_shares_base(cfg, backend=None)
