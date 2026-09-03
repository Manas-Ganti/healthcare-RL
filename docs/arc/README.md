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

## Verified on this cluster (2026-09-03)

`sacctmgr` confirms all three accounts exist -- `ece-6474-*`, `ece-6524-*`, `ece-6554-*` --
so the quickref and the `quota` output were never in conflict; the listing was just
truncated. This repo uses `ece-6524-spring2026`, the allocation with hours already drawn.

A100 QOS, confirmed:

| QOS | priority | max wall |
|---|---|---|
| `tc_a100_normal_short` | **1500** | 1-00:00:00 |
| `tc_a100_normal_int` | **1500** | 7-00:00:00 |
| `tc_a100_normal_base` | 1000 | 7-00:00:00 |
| `tc_a100_normal_long` | 500 | 14-00:00:00 |
| `tc_a100_preemptable_base` | 0 | 30-00:00:00 |

`short` and `int` share the top priority. Batch jobs use `short`; **interactive sessions
need `tc_a100_normal_int`**, which carries the same priority with a 7-day cap:

```bash
interact --account=ece-6524-spring2026 --partition=a100_normal_q \
         --qos=tc_a100_normal_int --gres=gpu:a100:1 \
         --cpus-per-task=8 --mem=96G --time=01:00:00
```

Nothing should ever use `tc_a100_preemptable_base`: priority 0, and normal preempts
preemptable.
