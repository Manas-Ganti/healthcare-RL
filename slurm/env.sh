#!/usr/bin/env bash
# Shared environment. Sourced by every job script.
set -euo pipefail

# Scratch. A 7B checkpoint is ~15GB and the HF cache fills a home quota on first download,
# so nothing model-shaped is allowed to land in $HOME.
export DXENV_SCRATCH="${DXENV_SCRATCH:-/projects/$USER/dxenv}"
export HF_HOME="$DXENV_SCRATCH/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCHINDUCTOR_CACHE_DIR="$DXENV_SCRATCH/inductor"
export OUTLINES_CACHE_DIR="$DXENV_SCRATCH/outlines"
mkdir -p "$HF_HOME" "$DXENV_SCRATCH"

export DXENV_REPO="${DXENV_REPO:-$HOME/healthcare-RL}"
export DXENV_MODEL="${DXENV_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Rollout stores go to scratch: a long run writes hundreds of MB of JSONL, and they are
# the expensive artifact, so they must not sit somewhere with a quota.
export DXENV_RUNS="${DXENV_RUNS:-$DXENV_SCRATCH/runs}"
mkdir -p "$DXENV_RUNS"

cd "$DXENV_REPO"
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
