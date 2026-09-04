#!/usr/bin/env bash
#SBATCH --job-name=dxenv-sft
#SBATCH --account=ece-6524-spring2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-user=manasganti@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/logs/sft-%j.out
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
# SFT is the ONE stage that needs TRL, and TRL cannot share an environment with vLLM
# (see pyproject.toml). Selected BEFORE env.sh, which asserts the interpreter.
export DXENV_VENV="${DXENV_VENV:-$DXENV_ROOT/.venv-train}"
echo "repo root: $DXENV_ROOT"
source slurm/env.sh
# ---------------------------------------------------------------------------------
dxenv_notify_on_exit "SFT"

pip install -q -e ".[dev,train]"

# Skip the rebuild if the set is already there. It costs 2000 teacher traces and 16000
# rollouts, and a resubmission after a TRL-API failure should not pay for it twice.
# Reuse only if the targets still match the CURRENT grammar. A stale set is not a
# harmless saving: the first SFT run trained on alphabetical key order while the decoder
# forces schema order, and produced an adapter that rambled in three languages. Rebuilding
# costs 30 minutes; training on a mismatched set costs 95 minutes plus the Gate B run that
# measures it.
if [[ -s runs/phase3/sft.jsonl ]] && python -c "
import json, sys
from dxenv.policy.decoding import action_json_schema
declared = {v['properties']['kind']['const']: list(v['properties'])
            for v in action_json_schema()['oneOf']}
with open('runs/phase3/sft.jsonl') as fh:
    for line in fh:
        obj = json.loads(json.loads(line)['completion'])
        emitted = list(obj)
        if emitted != [k for k in declared[obj['kind']] if k in emitted]:
            sys.exit(1)
"; then
    echo "reusing runs/phase3/sft.jsonl ($(wc -l < runs/phase3/sft.jsonl) examples, targets match the grammar)"
else
    echo "building the SFT set (absent, or its targets do not match the current grammar)"
    python scripts/build_sft_data.py --n 2000 --frozen-split --out runs/phase3/sft.jsonl
fi

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
