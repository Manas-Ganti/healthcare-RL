# ARC reference

Operational notes for Virginia Tech's ARC cluster, written for a **sibling project**
(`Visual-reasoning-rlvr`) on the same hardware. They are kept here because almost
everything in them is about the cluster rather than that project, and because each entry
was paid for with a wasted allocation.

What this repo took from them, and where it landed:

| finding | applied in |
|---|---|
| vLLM pins transformers ~4.51, TRL needs >=4.56; crossing them fails at import ~20 min into a job | `pyproject.toml` — split into `[infer]` and `[train]`, with two venvs |
| `--qos` is the biggest scheduling lever: priority 1330 -> 2312, pend reason Priority -> Resources | every `.sbatch` header |
| `--mem=0` means all node memory and can never backfill on a small job | explicit `--mem` everywhere |
| `/projects` is per-allocation and not user-writable; home is 640GB | `DXENV_SCRATCH` defaults to `$HOME/dxenv` |
| `source activate` can report success without switching interpreters; a zero-byte python exits 0 silently | `env.sh` asserts `sys.prefix` before any work |
| Do not use `*_preemptable_q` — normal preempts preemptable | header comments |
| `sbatch` copies the script at submit time; pending jobs ignore later edits | header comments |

**Not carried over**, because this project differs: the conda layout (this repo uses
venvs, which sidesteps the activation failure entirely), `NCCL_SOCKET_IFNAME` (single-GPU
here), and everything about the visual-reasoning task itself.

**One discrepancy to resolve before the first submission.** `quickref.md` names the
account `ece-6474-spring2026`; the live `quota` output on 2026-09-03 shows
`ece-6554-fall2025` and `ece-6524-spring2026` and no `6474`. This repo's `.sbatch` files
use `ece-6524-spring2026`. Confirm with `sacctmgr show assoc user=$USER` before queueing.
