#!/usr/bin/env bash
#SBATCH --job-name=dxenv-fetch
#SBATCH --partition=normal_q
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/fetch-%j.out

# Pre-download the weights into scratch. Separate from the GPU jobs because it needs
# internet and not a GPU, and because on clusters where compute nodes are network-isolated
# a training job that tries to download will simply hang until the wall clock kills it.

# --- bootstrap -------------------------------------------------------------------
# set -e HERE, not in env.sh. The first version put it there, and when env.sh failed to
# load the script kept going and printed "GPU stack OK" after every step had failed --
# the error handling was inside the file that failed to load.
set -euo pipefail

# SLURM copies the submitted script into a spool directory, so $0 and BASH_SOURCE point
# at /cm/local/apps/slurm/... and are useless for finding the repo. Walk up from the
# SUBMISSION directory instead, so the job works whether it was submitted from the repo
# root or from inside slurm/.
DXENV_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ "$DXENV_ROOT" != "/" && ! -f "$DXENV_ROOT/pyproject.toml" ]]; do
    DXENV_ROOT="$(dirname "$DXENV_ROOT")"
done
if [[ ! -f "$DXENV_ROOT/pyproject.toml" ]]; then
    echo "cannot find the repo root above ${SLURM_SUBMIT_DIR:-$PWD}" >&2
    exit 1
fi
cd "$DXENV_ROOT"
export DXENV_REPO="$DXENV_ROOT"
echo "repo root: $DXENV_ROOT"
source slurm/env.sh
# ---------------------------------------------------------------------------------
dxenv_notify_on_exit "model fetch"

python - <<PY
from huggingface_hub import snapshot_download
import os
p = snapshot_download("$DXENV_MODEL")
print("cached at", p)
PY
du -sh "$HF_HOME"
