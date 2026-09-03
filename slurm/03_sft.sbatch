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
source slurm/env.sh
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
