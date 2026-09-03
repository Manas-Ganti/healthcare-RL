# Running on a SLURM cluster (Virginia Tech ARC)

Three things about a scheduler change how this repo is run, and all three have bitten
projects like this one before:

1. **GPUs are allocated, not present.** Nothing on a login node can see a GPU, so the
   install must be split: the CPU install and the whole test suite run on the login node,
   and only the vLLM/torch import check needs an allocation.
2. **Jobs have a wall clock.** A 2000-step GRPO run does not fit in one allocation, so a
   long run is a *chain* of jobs. `train_grpo.py --resume` exists for this: it restores the
   step index, the curriculum stage, the RNG, and the monitor windows. Without it every
   job in the chain restarts the curriculum and refills the detector windows from empty,
   which leaves the ceiling and collapse monitors **off** for the first stretch of each job.
3. **Home directories have quotas.** A 7B checkpoint is ~15GB and the HF cache will fill a
   home quota on the first download. Every script here points `HF_HOME` at scratch.

Set `DXENV_SCRATCH` before submitting, or edit the default in `env.sh`.

## Logs

Each job writes one combined file (stdout and stderr merged) to
`slurm/logs/<name>-<jobid>.out`. The job id in the name means a resubmission never
overwrites its predecessor, which matters for `04_grpo.sbatch` because it requeues itself
and you want one file per link in the chain.

**Submit from the repo root.** `#SBATCH` directives are parsed by `sbatch` before any
shell runs, so `--output` cannot contain a variable and has to be a relative path --
relative to the directory you submit *from*, not to the script. Submitting from elsewhere
either scatters the logs or makes the job fail to launch outright, because SLURM will not
start a job whose output directory does not exist. If you need to submit from elsewhere,
use `sbatch --chdir=/path/to/healthcare-RL slurm/02_gate_b.sbatch`.

The logs themselves are gitignored; `slurm/logs/.gitkeep` is committed so the directory
survives a clone.

## Order

```bash
# login node -- no allocation needed
bash slurm/setup_cpu.sh                 # venv + CPU deps + full test suite

# one short GPU job to prove the stack imports and CUDA is visible
sbatch slurm/00_check_gpu.sbatch

# pre-download the model on a node with internet, if compute nodes are isolated
sbatch slurm/01_fetch_model.sbatch

# the measurement that gates everything (CLAUDE.md 8.1)
sbatch slurm/02_gate_b.sbatch

# only after Gate B has a verdict
sbatch slurm/03_sft.sbatch              # if and only if 8.1 says SFT is needed
sbatch slurm/04_grpo.sbatch             # resumable; resubmit to continue the chain
```

`04_grpo.sbatch` is safe to resubmit as many times as it takes: each job resumes where the
last one stopped and exits early once `--steps` is reached.

## Adjust before submitting

`--account`, `--partition` and the GPU type in each `.sbatch` are placeholders. Check what
your allocation actually offers with `sinfo -o "%P %G %l"` and `sacctmgr show assoc user=$USER`.
