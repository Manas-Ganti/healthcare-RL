#!/usr/bin/env bash
#SBATCH --job-name=dxenv-gpucheck
#SBATCH --account=ece-6524-spring2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --mail-user=manasganti@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/logs/gpucheck-%j.out
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
#   --gres      typed (gpu:a100:1). A100 is 80GB, H200 is 141GB; for H200 use
#               --partition=h200_normal_q --qos=tc_h200_normal_short --gres=gpu:h200:1.
#   --account   from `quota`. Do NOT use *_preemptable_q: normal preempts preemptable,
#               so moving there makes you the preemptee.
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
dxenv_notify_on_exit "GPU check"

pip install -e ".[dev,infer]"

echo "=== 1. does the stack import, and can torch see the GPU? ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "torch cannot see the GPU. Usually a CPU-only wheel pulled in by the resolver. "
    "Fix: pip uninstall -y torch && pip install vllm  (let vLLM pick its torch), "
    "then pip install -e '.[dev,infer]'"
)
print("device:", torch.cuda.get_device_name(0))
print("memory:", round(torch.cuda.get_device_properties(0).total_memory / 2**30), "GiB")
# NOT trl -- it lives in the train env and importing it here would be the exact
# cross-environment mistake that cost a sibling project a full node allocation.
import peft, transformers, vllm
print("vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__)
PY

echo
echo "=== 2. does the constrained-decoding path actually work? ==="
# A small model on purpose: this is testing OUR plumbing against vLLM's API, not the
# model's diagnostic ability. Override with DXENV_SMOKE_MODEL if 0.5B is unavailable.
export DXENV_SMOKE_MODEL="${DXENV_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
python scripts/gpu_smoke.py --model "$DXENV_SMOKE_MODEL" --n 2

echo
echo "GPU stack OK. Next: sbatch slurm/02_gate_b.sbatch"
