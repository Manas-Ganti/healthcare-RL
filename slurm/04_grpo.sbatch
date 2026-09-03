#!/usr/bin/env bash
#SBATCH --job-name=dxenv-grpo
#SBATCH --partition=a100_normal_q
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/grpo-%j.out

# RESUMABLE. A 2000-step run does not fit in one allocation, so resubmit this until it
# reports that --steps has been reached. Each job restores the step index, curriculum
# stage, RNG and monitor windows from trainer_state.json; the trajectory store is
# append-only and needs nothing.
#
# Do not submit this until Gate B has a verdict. The gate's own pre-registered failure
# action says not to, and it is the instruction that exists to stop a node being spent on
# a run that cannot teach the model anything.

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
dxenv_notify_on_exit "GRPO"

REF_ADAPTER="${1:-}"
REF_ARG=""
[[ -n "$REF_ADAPTER" ]] && REF_ARG="--reference-adapter $REF_ADAPTER"

python scripts/train_grpo.py \
    --run-id grpo \
    --model "$DXENV_MODEL" \
    $REF_ARG \
    --steps 2000 \
    --k 8 --patients-per-step 8 \
    --root "$DXENV_RUNS" \
    --resume

# Requeue automatically while there are steps left. Comment out to drive it by hand.
STATE="$DXENV_RUNS/grpo/trainer_state.json"
if [[ -f "$STATE" ]]; then
  DONE=$(python -c "import json;print(json.load(open('$STATE'))['step_index'])")
  STAGE=$(python -c "import json;print(json.load(open('$STATE'))['stage'])")
  echo "completed $DONE / 2000 steps"
  if (( DONE < 2000 )); then
    echo "requeueing"
    NEXT=$(sbatch --parsable "$0" "$REF_ADAPTER")
    dxenv_notify --status start --title "GRPO $DONE/2000"         --text "stage ${STAGE} · requeued as job ${NEXT}"
  else
    dxenv_notify --status ok --title "GRPO complete"         --text "$DONE/2000 steps, final stage ${STAGE}"
  fi
fi
