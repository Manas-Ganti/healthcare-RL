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

## Notifications

Jobs report themselves to Telegram so you do not have to hold an ssh session open. Each
job sends on start, on success (with a tail of its log), and on failure (with a longer
tail). Gate B additionally sends its verdict, and the GRPO chain reports which step and
curriculum stage it reached each time it requeues.

Setup, once:

1. Message **@BotFather** on Telegram, send `/newbot`, keep the token.
2. Message your new bot once -- a bot cannot open a conversation with you.
3. Find your chat id:

       TELEGRAM_BOT_TOKEN=<your-token> python scripts/notify.py --discover-chat-id

   It prints the ids of every chat that has messaged the bot. An empty result almost
   always means step 2 was skipped.
4. Store both **outside the repo**:

       mkdir -p ~/.config/dxenv
       cat > ~/.config/dxenv/telegram.env <<'EOF'
       export TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
       export TELEGRAM_CHAT_ID=987654321
       EOF
       chmod 600 ~/.config/dxenv/telegram.env

   `env.sh` sources it if present. It lives outside the repo because this repo is public
   and a committed bot token is a live credential, not a config value. `.gitignore`
   refuses `*.env` as a second line of defence.
5. Test from the login node:

       python scripts/notify.py --title "hello from ARC" --require

   `--require` makes a failed send exit non-zero. Without it, and everywhere inside a job,
   a failed send is silent by design: a notifier that can turn a successful twenty-hour
   run into a failed one because an HTTPS call timed out is worse than no notifier.

**If step 5 works on the login node but no messages arrive from jobs**, the compute nodes
have no route to the internet -- common, and the same reason `01_fetch_model.sbatch`
exists. Two options: submit jobs with `--mail-user` for plain SLURM state emails, or run a
watcher on the login node that polls `sacct` and sends the Telegram messages from there:

```bash
# login node, inside tmux/screen
while true; do
  sacct -u "$USER" --format=JobID,JobName%20,State,Elapsed --noheader -X \
    | grep -Ev "RUNNING|PENDING" > /tmp/dxenv-sacct.now
  if ! diff -q /tmp/dxenv-sacct.{last,now} >/dev/null 2>&1; then
    python scripts/notify.py --title "SLURM update" --text "$(cat /tmp/dxenv-sacct.now)"
    cp /tmp/dxenv-sacct.{now,last}
  fi
  sleep 120
done
```

The wall-clock case is the one worth having either way: SLURM sends SIGTERM before killing
a job, the trap catches it, and you get told. Without it a run that hits its time limit
simply vanishes with no error message at all.

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

## Cluster facts this is built around

From `docs/arc/`, written for a sibling project on the same hardware. Each was paid for
with a wasted allocation:

- **vLLM and TRL cannot share an environment.** vLLM pins transformers ~4.51; TRL needs
  >=4.56. Hence two venvs (`.venv` for inference/GRPO, `.venv-train` for SFT) and two
  extras (`[infer]`, `[train]`). Do not try to unify them.
- **`--qos` is the biggest scheduling lever.** Adding it moved a sibling job from priority
  1330 to 2312. "short" caps at a full day and has the highest priority; only a >24h run
  needs `*_base`.
- **Never `--mem=0`** on a small job -- it means all node memory and can only be satisfied
  on a wholly idle node, so it can never backfill.
- **`/projects` is per-allocation**, not per-user, and not writable by you. Home is 640GB
  with ~230GB free, which is where `DXENV_SCRATCH` points.
- **Assert the interpreter.** `source activate` can report success without switching
  interpreters, and a zero-byte python exits 0 printing nothing -- a GPU job then
  "succeeds" in two seconds with an empty log. `env.sh` checks `sys.prefix` before any
  work happens.

Both were confirmed on 2026-09-03: `ece-6524-spring2026` is a real allocation, and
`tc_a100_normal_short` carries priority 1500 with a 1-day cap -- the joint highest on the
A100 partition. See `docs/arc/README.md` for the full QOS table.

**Interactive sessions need a different QOS**, `tc_a100_normal_int` (same priority, 7-day
cap). For first contact with the GPU code, an interactive session beats a batch job: when
a moved vLLM API throws, you fix and rerun in seconds instead of re-queuing.

```bash
interact --account=ece-6524-spring2026 --partition=a100_normal_q \
         --qos=tc_a100_normal_int --gres=gpu:a100:1 \
         --cpus-per-task=8 --mem=96G --time=01:00:00
```

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
