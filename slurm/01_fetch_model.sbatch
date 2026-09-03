#!/usr/bin/env bash
#SBATCH --job-name=dxenv-fetch
#SBATCH --account=ece-6524-spring2026
#SBATCH --partition=normal_q
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-user=manasganti@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/logs/fetch-%j.out
#
# FLAGS THAT ARE LOAD-BEARING, not boilerplate (docs/arc/runbook.md 5):
#
#   --qos       the single biggest lever on this cluster. Adding it took a sibling job's
#               priority from 1330 to 2312 and its pend reason from Priority to Resources.
#               "short" is misleading: it caps at a FULL DAY and has the highest priority.
#               Only a >24h run needs *_base. Verify the A100 names once with
#                   sacctmgr show qos format=name%28,priority,maxwall
#   --mem       explicit, NEVER 0. --mem=0 means all memory on the node, which can only be
#               satisfied where nobody holds a byte, so a small job can never backfill.
#   --gres      typed (gpu:a100:1). A100 is 80GB, H200 is 141GB; for H200 use
#               --partition=h200_normal_q --qos=tc_h200_normal_short --gres=gpu:h200:1.
#   --account   from `quota`. Do NOT use *_preemptable_q: normal preempts preemptable,
#               so moving there makes you the preemptee.
#
# #SBATCH directives cannot read environment variables, and sbatch copies this file at
# SUBMIT time -- a pending job will not pick up edits made after it was queued.

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
