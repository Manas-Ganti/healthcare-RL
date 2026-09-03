#!/usr/bin/env bash
#SBATCH --job-name=dxenv-sft
#SBATCH --partition=a100_normal_q
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/sft-%j.out

# Run this ONLY if 02_gate_b said the base model cannot sample the behaviour. Deliberately
# undertrained: 1 epoch, low LR, LoRA. A model sharpened onto its SFT set produces
# identical rollouts, and identical rollouts give GRPO zero advantage to work with.

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
dxenv_notify_on_exit "SFT"

python scripts/build_sft_data.py --n 2000 --frozen-split --out runs/phase3/sft.jsonl

python - <<PY
from pathlib import Path
from dxenv.policy.sft import SFTConfig, SFTDataset, train_lora
ds = SFTDataset.read_jsonl(Path("runs/phase3/sft.jsonl"))
print(ds.summary())
ds.validate()   # refuses one-hot targets and a set with no abstentions
out = train_lora(ds, SFTConfig(model="$DXENV_MODEL", output_dir=Path("$DXENV_RUNS/sft")))
print("adapter:", out)
PY

echo "Next: sbatch slurm/02_gate_b.sbatch $DXENV_RUNS/sft/final"
