"""Fold the SFT adapter into the base weights once, so sampler and trainer share a base.

Why this exists (GRPO runs 1 and 2). The trainer merges the SFT adapter into the base and
trains a NEW LoRA on top of that merged model. Every sync then saved that new LoRA and
loaded it into vLLM -- whose base was still the raw Qwen checkpoint. From step 1 onward the
rollouts came from `raw base + small delta`, i.e. the prompted model, not the SFT policy:
the SFT adapter was silently dropped at the first sync. Both runs collapsed into 96-97%
timeouts, which is what the prompted model does, and run 1's "the policy learned not to
commit" was this bug, not a reward-design finding.

A LoRA delta is only meaningful against the weights it was trained on. So the sampler
must serve the same merged weights the trainer holds. Merging once to disk and pointing
both at it is exact: no rank concatenation, no scaling arithmetic, nothing to drift.

The merged copy is cached next to the adapter and keyed on (base model, adapter bytes).
A stale cache is an error, never silently reused: serving the wrong base is precisely the
failure this module exists to remove.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

MARKER = "dxenv_merged.json"


class MergeError(RuntimeError):
    pass


def adapter_digest(adapter: Path) -> str:
    """sha256 over the adapter's weight and config files, in name order."""
    files = sorted(
        p for p in adapter.iterdir()
        if p.is_file() and (p.suffix in {".safetensors", ".bin"} or p.name == "adapter_config.json")
    )
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise MergeError(f"{adapter} holds no adapter weights (.safetensors or .bin)")
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def merged_path_for(adapter: Path) -> Path:
    return adapter.parent / f"{adapter.name}-merged"


def _merge_with_peft(base_model: str, adapter: Path, out: Path) -> None:  # pragma: no cover
    """On CPU: the GPUs are about to be claimed by vLLM and the trainer."""
    import torch  # noqa: PLC0415 - GPU extra, imported only when a merge is needed
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model: Any = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model = model.to(torch.bfloat16)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out))


def ensure_merged(
    base_model: str,
    adapter: Path,
    merge_fn: Callable[[str, Path, Path], None] | None = None,
) -> Path:
    """Return a directory holding `base_model` with `adapter` merged in. Builds it once."""
    adapter = Path(adapter)
    if not adapter.is_dir():
        raise MergeError(f"reference adapter {adapter} is not a directory")
    want = {"base_model": base_model, "adapter": str(adapter.resolve()),
            "adapter_sha256": adapter_digest(adapter)}
    out = merged_path_for(adapter)
    marker = out / MARKER
    if marker.exists():
        have = json.loads(marker.read_text())
        if have == want:
            return out
        raise MergeError(
            f"{out} was merged from a different base or adapter:\n  have {have}\n  want "
            f"{want}\nDelete it (it is a derived artifact) and rerun; it is never reused "
            "or overwritten silently."
        )
    if out.exists() and any(out.iterdir()):
        raise MergeError(
            f"{out} exists without {MARKER}, so it may be a half-written merge. Delete it "
            "and rerun."
        )
    out.mkdir(parents=True, exist_ok=True)
    (merge_fn or _merge_with_peft)(base_model, adapter, out)
    # Written LAST, so an interrupted merge leaves no marker and is refused next time.
    marker.write_text(json.dumps(want, indent=2, sort_keys=True) + "\n")
    return out
