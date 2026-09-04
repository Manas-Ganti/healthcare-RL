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
