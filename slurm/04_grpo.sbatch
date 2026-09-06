#!/usr/bin/env bash
#SBATCH --job-name=dxenv-grpo
#SBATCH --account=ece-6524-spring2026
#SBATCH --partition=h200_normal_q
#SBATCH --qos=tc_h200_normal_short
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-user=manasganti@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/logs/grpo-%j.out
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
#   --gres      TWO h200s: vLLM on cuda:0, the trainer on cuda:1. H200 is 141GB against
#               the A100's 80GB, and the trainer OOMed at 77.7GB on a card it had to
#               itself -- the logits tensor is [T, 152064] and episodes get longer as the
#               policy learns to test, so the requirement grows over a run. 76% more
#               headroom for the same queue priority.
#   --account   from `quota`. Do NOT use *_preemptable_q: normal preempts preemptable,
#               so moving there makes you the preemptee.
#   --time      12h, deliberately, not the 24h that tc_a100_normal_short allows. This job
#               RESUMES and requeues itself, so wall time costs only a vLLM restart per
#               link, while a shorter request backfills far more easily -- partitions here
#               have MaxTime=UNLIMITED, so the scheduler cannot plan around the jobs ahead
#               of you and being small and short is the only reliable way through. Raising
#               it to 24h would also sit exactly ON the QOS cap, with no margin.
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
dxenv_notify_on_exit "GRPO"

REF_ADAPTER="${1:-}"
REF_ARG=""
[[ -n "$REF_ADAPTER" ]] && REF_ARG="--reference-adapter $REF_ADAPTER"

# Two GPUs: vLLM on cuda:0, the trainer on cuda:1. Sharing one card caused three OOMs --
# the engine reserves its share up front and the trainer takes what is left, so every fix
# on one side just moved the pressure to the other. A node here has gpu:a100:8, so the
# second card costs nothing in queue terms and removes the interaction entirely.
python scripts/train_grpo.py \
    --run-id grpo \
    --trainer-device "${DXENV_TRAINER_DEVICE:-cuda:1}" \
    --gpu-memory-utilization "${DXENV_VLLM_UTIL:-0.85}" \
    --model "$DXENV_MODEL" \
    $REF_ARG \
    --steps "${DXENV_GRPO_STEPS:-400}" \
    --k 8 --patients-per-step 8 \
    --root "$DXENV_RUNS" \
    --resume

# Requeue automatically while there are steps left. Comment out to drive it by hand.
STATE="$DXENV_RUNS/grpo/trainer_state.json"
if [[ -f "$STATE" ]]; then
  TARGET="${DXENV_GRPO_STEPS:-400}"
  DONE=$(python -c "import json;print(json.load(open('$STATE'))['step_index'])")
  STAGE=$(python -c "import json;print(json.load(open('$STATE'))['stage'])")
  echo "completed $DONE / $TARGET steps"
  if (( DONE < TARGET )); then
    echo "requeueing"
    NEXT=$(sbatch --parsable "$0" "$REF_ADAPTER")
    dxenv_notify --status start --title "GRPO $DONE/$TARGET"         --text "stage ${STAGE} · requeued as job ${NEXT}"
  else
    dxenv_notify --status ok --title "GRPO complete"         --text "$DONE/$TARGET steps, final stage ${STAGE}"
  fi
fi
