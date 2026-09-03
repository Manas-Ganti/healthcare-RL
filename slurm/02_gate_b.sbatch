#!/usr/bin/env bash
#SBATCH --job-name=dxenv-gateb
#SBATCH --partition=a100_normal_q
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/gateb-%j.out

# CLAUDE.md 8.1 -- the measurement that decides whether SFT is needed at all, and the
# gate into Phase 4. Inference only, no gradient.
#
# Pass an adapter as $1 to measure the SFT'd policy instead of the base model; the results
# then land in sft_baseline.json rather than overwriting the pre-SFT run.

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
dxenv_notify_on_exit "Gate B"

LORA_ARG=""
RESULTS="runs/phase3/prompted_baseline.json"
if [[ -n "${1:-}" ]]; then
  LORA_ARG="--lora $1"
  RESULTS="runs/phase3/sft_baseline.json"
fi

python scripts/phase3_prompted_baseline.py \
    --n 200 --k 8 --model "$DXENV_MODEL" $LORA_ARG

# A FAIL is a result, not an error -- the gate is meant to be able to say no.
python scripts/check_gate_b.py --results "$RESULTS" | tee /tmp/gateb-$$.txt || true

# The verdict is what you actually want on your phone; a log tail buries it.
VERDICT=$(grep -E "^GATE B:" /tmp/gateb-$$.txt || echo "GATE B: no verdict parsed")
dxenv_notify --status ok --title "Gate B verdict"     --text "$(cat /tmp/gateb-$$.txt)" --log-lines 0
rm -f /tmp/gateb-$$.txt
echo "$VERDICT"

echo
echo "Commit the verdict so it can be read off the repo:"
echo "  git add $RESULTS && git commit -m 'Gate B verdict' && git push"
