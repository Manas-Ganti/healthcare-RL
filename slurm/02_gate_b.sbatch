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
source slurm/env.sh

LORA_ARG=""
RESULTS="runs/phase3/prompted_baseline.json"
if [[ -n "${1:-}" ]]; then
  LORA_ARG="--lora $1"
  RESULTS="runs/phase3/sft_baseline.json"
fi

python scripts/phase3_prompted_baseline.py \
    --n 200 --k 8 --model "$DXENV_MODEL" $LORA_ARG

python scripts/check_gate_b.py --results "$RESULTS" || true   # a FAIL is a result, not an error

echo
echo "Commit the verdict so it can be read off the repo:"
echo "  git add $RESULTS && git commit -m 'Gate B verdict' && git push"
