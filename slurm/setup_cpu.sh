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

# Only the base dev deps are installed here -- the GPU extras go in on a compute node,
# where a CUDA wheel can actually be validated.
#   .venv        infer  -- rollouts, Gate B, gpu_smoke, GRPO
#   .venv-train  train  -- SFT only
make_venv() {
    local env="$1"
    # `python3 -m venv` can exit 0 having produced no pip -- ensurepip is stripped from
    # some HPC and conda pythons. That is how .venv-train came out empty while .venv,
    # created by an earlier run under a different module environment, looked fine.
    if [[ ! -x "$env/bin/python" ]]; then
        python3 -m venv "$env"
    fi
    # `python -m pip`, not the bin/pip shim: the shim can be absent even when the module
    # is importable, and the module is what actually does the install.
    if ! "$env/bin/python" -m pip --version >/dev/null 2>&1; then
        echo "  $env has no pip; bootstrapping with ensurepip"
        if ! "$env/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
            cat >&2 <<MSG
$env was created without pip, and ensurepip is unavailable in $(command -v python3).

That python cannot build a usable virtualenv. Load a real module and re-run:

    module spider python          # find an available build
    module load Python/3.11.5     # or whatever it lists
    rm -rf $env && bash slurm/setup_cpu.sh

You are currently on: $(python3 -c 'import sys; print(sys.executable)')
MSG
            exit 1
        fi
    fi
    "$env/bin/python" -m pip install -q -U pip
    "$env/bin/python" -m pip install -q -e ".[dev]"
    # Assert rather than assume: a silent no-op install is exactly the class of failure
    # this script exists to catch early.
    "$env/bin/python" -c "import dxenv, pytest" || {
        echo "$env installed but cannot import dxenv" >&2; exit 1; }
    echo "installed dev deps into $env ($("$env/bin/python" -V))"
}

# Two venvs, because vLLM and TRL cannot coexist (see pyproject.toml).
make_venv .venv
make_venv .venv-train

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
