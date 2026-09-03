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
source slurm/env.sh

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
  echo "completed $DONE / 2000 steps"
  if (( DONE < 2000 )); then
    echo "requeueing"
    sbatch "$0" "$REF_ADAPTER"
  fi
fi
