# ARC quick reference

The minimum an agent needs to run this project on VT ARC. Traps and war stories
live in [`arc_runbook.md`](arc_runbook.md); results in
[`../results/`](../results). This file is just names, paths and the command shape.

## Identities

| | |
|---|---|
| SLURM account | `ece-6474-spring2026` (always pass `--account`; the `ARC_ACCOUNT` var in `arc_env.sh` is dead) |
| mail | `--mail-user=manasganti@vt.edu` (`--mail-type` is already in the launchers) |
| branch | `rebuild/visual-reasoning-rlvr` |
| remote | `github.com/Manas-Ganti/Visual-reasoning-rlvr` |
| ARC checkout | `/home/manasganti/ondemand/data/VLM-RL-aicontent-detection` |
| Mac checkout | `/Users/manasganti/portfolio-projects/VLM-RL-aicontent-detection` |

**Two checkouts, one branch.** Edits on the Mac reach ARC only via push + pull.
`.slurm` files are copied by `sbatch` at SUBMIT time — a pending job will NOT pick
up launcher edits. Python files and `arc_env.sh` are read at RUN time and will.

## Conda environments — crossing them fails

| env | path | for |
|---|---|---|
| `vrr` | `/home/manasganti/miniconda3/envs/vrr` | anything via `arc_infer.slurm` — gates, budget probe, distill, eval, groupvar (vLLM 0.8.5, transformers 4.51) |
| `vrr-train` | `/home/manasganti/.conda/envs/vrr-train` | `arc_sft.slurm`, `arc_grpo.slurm` (trl 1.9.2, transformers 5.15) |
| `vrr-gen` | `/home/manasganti/miniconda3/envs/vrr-gen` | image generation via `arc_synth.slurm` (diffusers) |

`vrr` and `vrr-train` cannot be merged: vLLM pins transformers ~4.51, trl needs
≥4.56. Using `vrr` for SFT dies with `cannot import name 'is_trackio_available'`.

## Always-set variables

```
HF_HOME=/home/manasganti/hf_cache          # NOT arc_env.sh's /projects default; token lives here too
WANDB_DIR=/home/manasganti/wandb           # /projects/$USER is not writable
VRR_DATASET=synth1024                      # namespaces data/, checkpoints/, logs/, results/
OVERVIEW_LONG_EDGE=56                      # MUST match across every stage of a run
CONDA_ENV=<one of the three above>
```

`NCCL_SOCKET_IFNAME=eth0` on H200 (`tc-xe*`) only. **Leave it unset on A100** —
`arc_env.sh` probes for an addressed interface. `ib0` exists on H200 but has no
IP, which is a silent 10-minute failure.

## Partitions and QOS

| partition | nodes | short QOS (prio / cap) | long QOS |
|---|---|---|---|
| `h200_normal_q` | 6 · `tc-xe[001-006]` · 141 GB | `tc_h200_normal_short` (2000 / 1 day) | `tc_h200_normal_base` (1000 / 7 d) |
| `a100_normal_q` | 14 · `tc-dgx[001-010]`,`tc-gpu[001-004]` · 80 GB | `tc_a100_normal_short` (1500 / 1 day) | `tc_a100_normal_base` (1000 / 7 d) |

"short" is a **full day** and the highest priority — use it for everything under
24 h. A100s are 80 GB, so 32B training there needs
`DS=configs/deepspeed_zero3.json` (ZeRO-2 replicates 66 GB per GPU and OOMs).

## Command shape

Env vars before `sbatch`; sbatch flags; script; then any passthrough args to the
Python entry point.

```bash
HF_HOME=... WANDB_DIR=... CONDA_ENV=... OVERVIEW_LONG_EDGE=56 VRR_DATASET=synth1024 \
JOB=<stage> MODEL=32b TP=1 \
sbatch --account=ece-6474-spring2026 --partition=h200_normal_q \
       --qos=tc_h200_normal_short --gres=gpu:h200:1 --cpus-per-task=8 --mem=96G \
       --time=00:30:00 --mail-user=manasganti@vt.edu \
       scripts/arc_infer.slurm --some-python-flag
```

Never `--mem=0` on a small job — the launchers default to it (right for full-node
work) and it can only be satisfied on a wholly idle node.

## Stage order

```
1  manifest_stats            data sanity — every predictor ~0.5, else the gates measure the file
2  JOB=budget  AUC=1         the ceiling AT THE AGENT'S BUDGET; budget 0 is the floor. Gap >=0.15
3  JOB=distill               rejection-sampled traces -> data/<ds>/sft_traces.jsonl
4  arc_sft.slurm             vrr-train. TARGET_AI_FRAC sets the class prior HERE and nowhere else
5  JOB=groupvar              gate 2 vs the SFT ckpt; want usable_groups >= 0.40
6  arc_grpo.slurm            vrr-train, GRAD_ACCUM=8, --max-steps to fit the wall
7  JOB=eval                  twice: VRR_DATASET=synth1024 and =synth1024flux (unseen generator)
```

`ADAPTER=` / `SFT_CKPT=` paths are `checkpoints/<ds>/<stage>-qwen2.5-vl-32b`. A
GRPO run killed by walltime leaves only `checkpoint-N/` subdirectories — the
top-level adapter is written by the final save, which a timeout skips.

`MAX_INSPECTS` must be identical in steps 5, 6 and 7 or the numbers are not comparable.

## Monitoring

```bash
squeue -u $USER -o "%.10i %.12j %.9T %.11M %.11L %.22R %N"
python tools/watch_grpo.py --dataset synth1024      # verdict balance, confidence, coherence
srun --jobid=<id> --overlap nvidia-smi              # is it actually computing
```

Telegram notification of the finished log is configured via
`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in `~/.config/vrr/secrets.env`.
