# dxenv — a diagnostic RLVR environment

A multi-turn reinforcement learning environment where an LLM agent plays diagnostician
over synthetic patient records. Each episode, the agent sees a filtered view of a patient,
decides which tests are worth their cost, and terminates by reporting a probability
distribution over conditions — or by abstaining. It is scored once, at the end, against
hidden ground truth.

**The research contribution is the environment, not the policy.** Three things make it
worth building:

- **Verifiable reward.** Every term is computed from ground truth by a pure function. No
  learned reward model, no LLM judge.
- **A computable Bayes-optimal ceiling**, which doubles as an automatic reward-hacking
  detector: the environment knows the best score achievable by perfect reasoning over the
  same evidence, so an agent that beats it has information it should not have.
- **A budget-conditioned cost–accuracy frontier.** Budget is sampled per episode and
  exposed to the agent, so one policy spans the whole frontier and evaluation sweeps it.

> **This is a synthetic research environment.** It is not a clinical decision tool, is not
> validated against real patients, and no artifact from this repo may be presented as
> clinical guidance. Severity weights, likelihoods and contraindication rules are
> simulation parameters chosen for RL dynamics, not clinical recommendations.

---

## Why it is hard to build correctly

The generator produces records from explicit disease modules, so the diagnosis is present
in the record in about six places — and the *sparsity pattern* of the record leaks it in a
seventh. Most of the engineering exists to close those channels. A leaky environment
produces excellent-looking numbers that mean nothing, and the failure is silent.

So the environment is built around twelve invariants, each with its own test file in
`tests/invariants/`. The load-bearing ones:

| | |
|---|---|
| **I1** | Ground truth never enters an observation — the observation type has no field that could hold it |
| **I3** | The action menu is global and identical for every patient. A per-patient menu *is* the diagnosis |
| **I4** | Every test returns a value for every patient. No "unavailable", so sparsity carries no signal |
| **I5** | Ordering a test never produces positive reward, under any shaping term. Tests only subtract |
| **I7** | Terminal scoring uses a strictly proper rule (Brier), which rules out hedging mathematically |
| **I9** | Episode reward never exceeds the Bayes ceiling. A violation halts training as a suspected leak |
| **I12** | The eval split is frozen and hash-verified. Training never reads it |

---

## What the environment provides

**Observations** are built by allowlist and fail closed. `Condition`, `MedicationRequest`,
`CarePlan`, `Procedure`, `reasonCode`, `DiagnosticReport.conclusion` and `CareTeam` are all
blocked — each is the label in disguise. Display strings are scrubbed too, because a lab
name reading "HbA1c — diabetes monitoring" leaks even when the field does not.

**Actions** are a fixed global menu: ~100 tests and panels, a treatment set, `diagnose`,
and `abstain`. Action ids are content-hashed, so adding a test does not renumber the
others and invalidate stored trajectories.

**Reward** is a pure function of `(trajectory, ground_truth, config)`:

```
R = brier(p, c_true) · severity_weight(c_true)
    − λ · Σ cost(t) − μ · n_turns
    + treatment_score + shaping + predict_then_verify
```

Brier rather than log-loss, because log-loss is unbounded below and one bad rollout wrecks
GRPO's advantage normalisation. Severity weights stop the agent maximising raw accuracy on
common benign conditions while eating the rare-severe tail. Treatments are scored twice —
against the declared diagnosis and against the truth — so a lucky-correct treatment under a
wrong diagnosis does not collect.

**Two ceilings.** A per-episode hard ceiling that is sound on every realisation and safe to
halt on, and a tighter expected ceiling checked on running means. Conflating them is the
trap: a single lucky rollout can legitimately beat the expected one.

**Everything is persisted.** One JSON line per episode under pinned config hashes. Reward is
pure, so rescoring a stored corpus under new weights is free — and you will change the
weights repeatedly.

**Training.** Constrained decoding makes invalid output impossible (so the format reward is
exactly zero and sampling entropy survives), a privileged teacher generates demonstrations
that are then stripped of their privilege, rejection sampling filters on full reward rather
than on correctness, and GRPO trains against monitors that halt on ceiling breach, entropy
collapse, and cost-distribution collapse.

---

## Install

```bash
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && pip install -e ".[dev]"
```

The `[dev]` install runs everything below. Training a model additionally needs
`pip install -e ".[gpu]"` (vLLM, TRL, PEFT) on a CUDA host; those imports are lazy, so the
full test suite and the whole training loop run on a laptop without them.

## Run

```bash
pytest                       # fast suite: unit + invariants, ~30s
pytest -m slow               # corpus-wide sweeps + toy-MDP policy invariance
ruff check dxenv tests scripts && mypy
```

```python
from dxenv.data.corpus import generate_corpus
from dxenv.env.episode import DiagnosticEpisode
from dxenv.policy.baselines import GreedyBayesPolicy, run_episode
from dxenv.reward.engine import GroundTruth, load_reward_config, score_trajectory

rec = generate_corpus(1, seed=0)[0]
traj = run_episode(DiagnosticEpisode(rec, seed=0, budget=100.0), GreedyBayesPolicy())
gt = GroundTruth(rec.condition, rec.analytes, rec.allergies)
print(score_trajectory(traj, gt, load_reward_config()).as_dict())
```

**Evaluation and probes**, no GPU required:

```bash
python scripts/phase0_feasibility.py --n 10000 --seed 7   # is there a learnable signal?
python scripts/check_gate_a.py                            # evaluate against the gate
python scripts/phase3_prompted_baseline.py --n 200 --k 8  # baselines + the model-free floor
python scripts/check_gate_b.py --results runs/phase3/prompted_baseline.json
python scripts/train_grpo.py --dry-run --steps 20         # real rollouts, no gradient
```

**Training**, on a CUDA host:

```bash
# 1. BEFORE any SFT: can the base model do this at all? Decides whether SFT is needed.
python scripts/phase3_prompted_baseline.py --n 200 --k 8 --model <hf-id>
python scripts/check_gate_b.py --results runs/phase3/prompted_baseline.json

# 2. Only if step 1 says SFT is needed:
python scripts/build_sft_data.py --n 2000 --frozen-split --out runs/phase3/sft.jsonl
#    ... train the LoRA (dxenv.policy.sft.train_lora) ...

# 3. Gate B proper: same measurement on the SFT'd policy. The go/no-go into GRPO.
python scripts/phase3_prompted_baseline.py --n 200 --k 8 --model <hf-id> --lora runs/sft/final
python scripts/check_gate_b.py --results runs/phase3/sft_baseline.json

# 4. Train, then rescore the stored rollouts under whatever weights you want.
python scripts/train_grpo.py --reference-adapter runs/sft/final --steps 2000
python scripts/rescore.py runs/grpo --corpus-n 20000 --corpus-seed 20260901
```

## Running on a cluster

`slurm/` has job scripts for a SLURM cluster (written against Virginia Tech ARC), plus
notes on the three things a scheduler changes: GPUs are allocated rather than present, so
the CPU install and the whole test suite run on the login node; jobs have a wall clock, so
a long GRPO run is a *chain* of jobs and `train_grpo.py --resume` restores the step index,
curriculum stage, RNG and monitor windows between them; and home directories have quotas,
so `HF_HOME` and the rollout stores point at scratch. See [`slurm/README.md`](slurm/README.md).

---

## Status

| Phase | State |
|---|---|
| **0 — feasibility** | Gate A passes on both substantive criteria |
| **1 — environment** | Complete |
| **2 — reward engine** | Complete |
| **3 — cold start** | Complete. Gate B measured on base and SFT |
| **4 — GRPO** | Complete. 99 steps trained on Qwen2.5-7B + LoRA |
| **5 — evaluation** | Complete: audit suite, Pareto sweep, calibration |

326 fast + 5 slow tests. `ruff` and `mypy --strict` clean. CI runs the fast suite, lint and
types on every commit, and the corpus-wide suite nightly.

**What has actually run.** Gate A passes. Gate B was measured three times — the prompted 7B
(FAIL, and *how* it fails is the finding), the SFT'd policy (PASS under the amended
gate_b2, the go/no-go into GRPO), and the GRPO adapter (FAIL, on pass@k and schema
validity). GRPO trained for 99 steps; the curve is committed at `runs/grpo/curve.png` and
regenerates from `runs/grpo/steps.jsonl`.

**The headline result is about the environment, not the policy.** GRPO took test-ordering
from 0.79 to 3.5 per episode with no reward term for testing — under I5 tests only ever
subtract — so the only route was that tests paid for themselves. The policy itself is still
poor: it sits below the blank-record floor, and the reward curve is a fast correction
followed by a plateau. See [`docs/design-notes.md`](docs/design-notes.md) for the reading.

---

## Layout

```
dxenv/
  data/      taxonomy (149 flat labels) · corpus · splits · store (JSONL) · eval_split.json
  env/       filter [I1,I2] · actions [I3] · obs_model [I4] · bayes [I9] · episode · schemas
  reward/    scoring [I7] · costs [I5] · treatment · verify · shaping [I6] · engine [I8]
  policy/    prompt · decoding (grammar) · llm · rollout · teacher · rejection · sft · baselines
  train/     grpo (loop + LoRA updater) · monitors · curriculum
  eval/      audit · pareto · calibration
  configs/   gate_a · gate_a2 · gate_b · severity · reward · costs · env · treatments
tests/       invariants (one file per I1–I12) · unit · property · golden
scripts/     gates, the Phase 3 pipeline, training, and rescoring
```

Import rules are enforced by a test that parses the source, so a violation is caught even
inside a function: `reward/` never imports `policy/` or `train/`; `env/` never imports
`reward/`; `data/` depends on nothing above it. `env.step()` returns **no reward** — the
environment produces trajectories, the reward engine scores them, which is what makes
offline rescoring free.

---

## Known gaps

- **Likelihood parameters are invented**, so the Bayes baselines are parameterised from the
  same generative model that produced the data. This environment measures whether a policy
  can learn a synthetic mapping, not whether it can diagnose. It is the caveat underneath
  every number here.
- **No arm has cleared the no-information floor on mean reward.** Base −0.689, SFT −0.664,
  GRPO −0.569, against a blank-record floor of −0.018. GRPO's best-of-8 does clear it
  (+0.034), the first arm to.
- **GRPO spent most of its exploration budget.** pass@8 fell 0.330 → 0.105 and within-group
  spread halved while the mean improved — the diversity a further RL round would sharpen is
  largely gone.
- **The curriculum never advanced.** All 99 GRPO steps ran in `single_condition_short`, so
  the full-horizon and comorbid stages are untested in practice — and the policy was trained
  at 8 turns while Gate B evaluates at 20.
- **`data/snomed_map.yaml` is empty**, so real Synthea output cannot be ingested yet.
- **Comorbidity is unimplemented.** The curriculum declares a `comorbid` stage; the
  generator emits one condition per patient.
- **Likelihood parameters are invented.** Consistent and leak-free, but not drawn from
  published likelihood ratios.

---

## Further reading

- [`docs/design-notes.md`](docs/design-notes.md) — the engineering record: measurements,
  the calibration of `λ`, the two places the spec did not survive contact, and the audit
  suite results.
- [`docs/interview-reference.md`](docs/interview-reference.md) — the complete reference:
  every feature, every design decision with its reasoning, every caveat, and the debugging
  record.
- [`CLAUDE.md`](CLAUDE.md) — the full specification: invariants, phase plan, and the
  testing philosophy the repo is built to.
