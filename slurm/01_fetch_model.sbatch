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
source slurm/env.sh
dxenv_notify_on_exit "model fetch"

python - <<PY
from huggingface_hub import snapshot_download
import os
p = snapshot_download("$DXENV_MODEL")
print("cached at", p)
PY
du -sh "$HF_HOME"
