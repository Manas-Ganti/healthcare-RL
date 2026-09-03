#!/usr/bin/env bash
#SBATCH --job-name=dxenv-gpucheck
#SBATCH --partition=a100_normal_q       # check: sinfo -o "%P %G %l"
#SBATCH --account=CHANGE_ME             # check: sacctmgr show assoc user=$USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/gpucheck-%j.out

# Installs the GPU stack and proves it imports. Short and cheap on purpose: this is the
# step most likely to fail, none of the GPU code in this repo has ever executed, and
# finding a resolver conflict here costs 30 minutes rather than a training allocation.
source slurm/env.sh

pip install -e ".[dev,gpu]"

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "torch cannot see the GPU. Usually a CPU-only wheel pulled in by the resolver. "
    "Fix: pip uninstall -y torch && pip install vllm  (let vLLM pick its torch), "
    "then pip install -e '.[dev,gpu]'"
)
print("device:", torch.cuda.get_device_name(0))
import vllm, peft, trl, transformers
print("vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| trl", trl.__version__)
PY

echo "GPU stack OK."
