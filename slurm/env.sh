#!/usr/bin/env bash
# Shared environment. Sourced by every job script AFTER it has located the repo root.
#
# Deliberately does NOT set -e and does NOT cd. Both used to live here, and when this file
# failed to load -- which is what happens if the job was submitted from the wrong
# directory -- the calling script lost its error handling at the same moment it lost its
# environment, and then reported success. Each job script sets -e itself, first.

# `exit` in a SOURCED script terminates the calling shell. Since this file is meant to be
# sourced -- by job scripts, and interactively when poking at things on a login node -- a
# failure here would close your terminal, and only ever at the moment something has just
# gone wrong. Fail with `return` when sourced, `exit` when run.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _dxenv_fail() { echo "$@" >&2; return 1; }
else
    _dxenv_fail() { echo "$@" >&2; exit 1; }
fi

# Scratch. HOME, not /projects: on ARC /projects is per-ALLOCATION and not writable by an
# individual user, while home is 640GB (409GB used as of 2026-09-03, so ~230GB free) --
# comfortably more than the ~15GB a 7B checkpoint needs. This mirrors what the sibling
# project on this cluster settled on; see docs/arc/runbook.md.
export DXENV_SCRATCH="${DXENV_SCRATCH:-$HOME/dxenv}"
export HF_HOME="${HF_HOME:-$DXENV_SCRATCH/hf}"
# HF_HUB_ENABLE_HF_TRANSFER is deprecated in current huggingface_hub and warns on
# every invocation; Xet replaced it.
export HF_XET_HIGH_PERFORMANCE=1
# Recommended by the allocator itself in the OOM this run hit: with a vLLM engine and a
# trainer sharing one card, the free memory is fragmented and a large contiguous request
# fails even when the total is available.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="$DXENV_SCRATCH/inductor"
export OUTLINES_CACHE_DIR="$DXENV_SCRATCH/outlines"

if ! mkdir -p "$HF_HOME" "$DXENV_SCRATCH" 2>/dev/null; then
    _dxenv_fail "DXENV_SCRATCH=$DXENV_SCRATCH is not writable. Set it somewhere with ~20GB."
    return 1 2>/dev/null || exit 1
fi

# Set by each job script's bootstrap, which walks up from $SLURM_SUBMIT_DIR.
export DXENV_REPO="${DXENV_REPO:-$PWD}"
export DXENV_MODEL="${DXENV_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DXENV_RUNS="${DXENV_RUNS:-$DXENV_SCRATCH/runs}"
mkdir -p "$DXENV_RUNS"

# Which virtualenv. vLLM and TRL cannot coexist (see pyproject.toml), so there are two:
#   .venv        infer  -- rollouts, Gate B, gpu_smoke, GRPO
#   .venv-train  train  -- SFT only
export DXENV_VENV="${DXENV_VENV:-$DXENV_REPO/.venv}"

module reset >/dev/null 2>&1 || true

# --- CUDA toolkit for JIT-compiled kernels ---------------------------------------------
# vLLM's default sampler is FlashInfer, which JIT-COMPILES a kernel during engine warmup
# and needs nvcc. On this cluster that fails with
#     RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
# because the compute nodes carry a driver but no toolkit at that path.
#
# Two independent fixes, both applied, because either alone can lapse:
#
#   1. Point CUDA_HOME at a real toolkit. Try the module system first, then the toolkit
#      pip installed alongside torch (nvidia-cuda-nvcc ships one inside site-packages),
#      which needs no module at all and matches the wheel's CUDA version by construction.
#   2. Turn the FlashInfer sampler off. It is a throughput optimisation; the PyTorch
#      sampler is numerically equivalent and compiles nothing. This is what actually makes
#      the run independent of whether a toolkit is present, so it is the default rather
#      than the fallback. Set DXENV_FLASHINFER=1 to opt back in once a toolkit is
#      confirmed working.
module load CUDA >/dev/null 2>&1 || module load cuda >/dev/null 2>&1 || true

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
fi
if [[ -z "${CUDA_HOME:-}" && -x "$DXENV_VENV/bin/python" ]]; then
    _pip_nvcc="$("$DXENV_VENV/bin/python" - <<'PY' 2>/dev/null || true
import pathlib, sys
root = pathlib.Path(sys.prefix)
for base in root.glob("lib/python*/site-packages/nvidia/cuda_nvcc"):
    if (base / "bin" / "nvcc").exists():
        print(base)
        break
PY
)"
    if [[ -n "$_pip_nvcc" ]]; then
        export CUDA_HOME="$_pip_nvcc"
        export PATH="$CUDA_HOME/bin:$PATH"
    fi
    unset _pip_nvcc
fi
[[ -n "${CUDA_HOME:-}" ]] && echo "[dxenv] CUDA_HOME=$CUDA_HOME"

if [[ "${DXENV_FLASHINFER:-0}" != "1" ]]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
fi
# ---------------------------------------------------------------------------------------

# --- interpreter, asserted -------------------------------------------------------------
# The sibling project lost a full node allocation to this class of bug twice: `source
# activate` reporting success in a non-interactive batch shell without switching
# interpreters, and separately a zero-byte python that exited 0 and printed nothing, so a
# GPU job "succeeded" in two seconds with an empty log. Neither failed loudly anywhere.
#
# So: an absolute path, put on PATH directly rather than activated, and then ASSERTED --
# both that the interpreter runs at all and that it is the one intended.
export PY="$DXENV_VENV/bin/python"
if [[ ! -x "$PY" ]]; then
    _dxenv_fail "no interpreter at $PY -- run slurm/setup_cpu.sh first"
    return 1 2>/dev/null || exit 1
fi
export PATH="$DXENV_VENV/bin:$PATH"
if ! "$PY" -c "
import sys, pathlib
# sys.prefix, NOT the resolved sys.executable: .venv/bin/python is a symlink to the base
# interpreter, so resolving it reports the base install and the check fails on a perfectly
# good venv. sys.prefix is the venv root and is what actually distinguishes them.
want = pathlib.Path('$DXENV_VENV')
have = pathlib.Path(sys.prefix)
assert want.samefile(have), f'sys.prefix is {have}, expected {want}'
print(f'[dxenv] python={sys.executable} ({sys.version.split()[0]}) prefix={have}')
"; then
    _dxenv_fail "interpreter check FAILED for $PY. A python that prints nothing and exits 0
is a zero-byte or broken install -- rebuild the venv rather than debugging the job."
    return 1 2>/dev/null || exit 1
fi
# ---------------------------------------------------------------------------------------

# Telegram credentials, if configured. Kept OUTSIDE the repo: this repo is public, and a
# committed bot token is a live credential rather than a config value. ~/.config/vrr is
# checked too, because a sibling project on this cluster already put a token there and
# there is no reason to make you create a second bot.
for _sec in "$HOME/.config/dxenv/telegram.env" "$HOME/.config/vrr/secrets.env"; do
    # shellcheck source=/dev/null
    [[ -f "$_sec" ]] && source "$_sec"
done
unset _sec
# shellcheck source=/dev/null
source "$DXENV_REPO/slurm/notify.sh"
