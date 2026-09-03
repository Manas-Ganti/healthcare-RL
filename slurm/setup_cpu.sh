#!/usr/bin/env bash
# Login node. No GPU needed, and deliberately so: this proves the environment is sane
# while the dependency graph is still simple. If something breaks after the GPU install,
# you will know it was the GPU stack.
set -euo pipefail
cd "$(dirname "$0")/.."

module reset >/dev/null 2>&1 || true
module load Python/3.11.5 2>/dev/null || module load python/3.11 2>/dev/null || true

python3 -c 'import sys; assert sys.version_info >= (3,11), sys.version' || {
  echo "Python 3.11+ required; check 'module spider python'"; exit 1; }

# Two venvs, because vLLM and TRL cannot coexist (see pyproject.toml). Only the base
# dev deps are installed here -- the GPU extras go in on a compute node, where a CUDA
# wheel can actually be validated.
#   .venv        infer  -- rollouts, Gate B, gpu_smoke, GRPO
#   .venv-train  train  -- SFT only
python3 -m venv .venv
python3 -m venv .venv-train
for env in .venv .venv-train; do
    "$env/bin/pip" install -q -U pip
    "$env/bin/pip" install -q -e ".[dev]"
    echo "installed dev deps into $env"
done
source .venv/bin/activate

echo "--- fast suite ---"; pytest -q
echo "--- lint and types ---"; ruff check dxenv tests scripts && mypy
echo "--- frozen eval split [I12] ---"
python -c "from dxenv.data.splits import rebuild_frozen_splits; print(rebuild_frozen_splits().summary())"
echo "--- full loop, no gradient ---"
python scripts/train_grpo.py --dry-run --steps 3 --k 4 --patients-per-step 3 --run-id setupcheck
rm -rf runs/setupcheck
echo
echo "CPU setup OK."
echo
echo "Before the first GPU submission, verify the two cluster facts this repo guesses at:"
echo "  sacctmgr show assoc user=\$USER format=account,partition   # the --account name"
echo "  sacctmgr show qos format=name%28,priority,maxwall | grep a100"
echo
echo "Next: sbatch slurm/00_check_gpu.sbatch"
