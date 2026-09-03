#!/usr/bin/env bash
#SBATCH --job-name=dxenv-gpucheck
#SBATCH --partition=a100_normal_q       # check: sinfo -o "%P %G %l"
#SBATCH --account=CHANGE_ME             # check: sacctmgr show assoc user=$USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/gpucheck-%j.out

# Installs the GPU stack, then drives ONE episode through it end to end.
#
# Short and cheap on purpose. None of the GPU code in this repo has ever executed, and the
# parts most likely to have rotted are vLLM's guided-decoding API (the import path for
# GuidedDecodingParams, and whether SamplingParams still takes `guided_decoding`) and the
# signature of LLM.chat(). An import check would not touch any of that -- it would pass,
# and the first real exercise would then be inside the four-hour Gate B job.
#
# So this loads a small model and generates one constrained action. If the API has moved,
# it fails here, in minutes, with a traceback.

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
dxenv_notify_on_exit "GPU check"

pip install -e ".[dev,gpu]"

echo "=== 1. does the stack import, and can torch see the GPU? ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "torch cannot see the GPU. Usually a CPU-only wheel pulled in by the resolver. "
    "Fix: pip uninstall -y torch && pip install vllm  (let vLLM pick its torch), "
    "then pip install -e '.[dev,gpu]'"
)
print("device:", torch.cuda.get_device_name(0))
print("memory:", round(torch.cuda.get_device_properties(0).total_memory / 2**30), "GiB")
import peft, transformers, trl, vllm
print("vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| trl", trl.__version__)
PY

echo
echo "=== 2. does the constrained-decoding path actually work? ==="
# A small model on purpose: this is testing OUR plumbing against vLLM's API, not the
# model's diagnostic ability. Override with DXENV_SMOKE_MODEL if 0.5B is unavailable.
export DXENV_SMOKE_MODEL="${DXENV_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
python scripts/gpu_smoke.py --model "$DXENV_SMOKE_MODEL" --n 2

echo
echo "GPU stack OK. Next: sbatch slurm/02_gate_b.sbatch"
