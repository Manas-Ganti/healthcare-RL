#!/usr/bin/env bash
# Shared environment. Sourced by every job script AFTER it has located the repo root.
#
# Deliberately does NOT set -e and does NOT cd. Both used to live here, and when this file
# failed to load -- which is exactly what happens if the job was submitted from the wrong
# directory -- the calling script lost its error handling at the same moment it lost its
# environment, and then reported success. Each job script now sets -e itself, before
# anything can fail.

# Scratch. A 7B checkpoint is ~15GB and the HF cache fills a home quota on first download,
# so nothing model-shaped is allowed to land in $HOME.
export DXENV_SCRATCH="${DXENV_SCRATCH:-/projects/$USER/dxenv}"
export HF_HOME="$DXENV_SCRATCH/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCHINDUCTOR_CACHE_DIR="$DXENV_SCRATCH/inductor"
export OUTLINES_CACHE_DIR="$DXENV_SCRATCH/outlines"
if ! mkdir -p "$HF_HOME" "$DXENV_SCRATCH" 2>/dev/null; then
    cat >&2 <<MSG
DXENV_SCRATCH=$DXENV_SCRATCH is not writable.

The default (/projects/\$USER/dxenv) is a guess at your project space. Point it at real
scratch and re-submit:

    export DXENV_SCRATCH=/path/to/your/project/space/dxenv

This has to be somewhere with room: the model cache alone is ~15GB for a 7B checkpoint,
and it must not be a home directory with a quota.
MSG
    exit 1
fi

# Set by each job script's bootstrap, which walks up from $SLURM_SUBMIT_DIR.
export DXENV_REPO="${DXENV_REPO:-$PWD}"
export DXENV_MODEL="${DXENV_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Rollout stores go to scratch: a long run writes hundreds of MB of JSONL, and they are
# the expensive artifact, so they must not sit somewhere with a quota.
export DXENV_RUNS="${DXENV_RUNS:-$DXENV_SCRATCH/runs}"
mkdir -p "$DXENV_RUNS"

# ARC uses Lmod. Adjust the versions to whatever `module spider python cuda` reports.
module reset >/dev/null 2>&1 || true
module load Python/3.11.5 2>/dev/null || module load python/3.11 2>/dev/null || true
module load CUDA/12.4.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true
source "$DXENV_REPO/.venv/bin/activate"

# Telegram credentials, if configured. Kept OUTSIDE the repo on purpose: this repo is
# public, and a committed bot token is a live credential, not a config value.
# See scripts/notify.py for how to obtain the token and chat id.
# shellcheck source=/dev/null
[[ -f "$HOME/.config/dxenv/telegram.env" ]] && source "$HOME/.config/dxenv/telegram.env"
# shellcheck source=/dev/null
source "$DXENV_REPO/slurm/notify.sh"
