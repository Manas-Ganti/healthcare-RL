# dxenv — a diagnostic RLVR environment

A multi-turn RL environment in which an LLM agent plays diagnostician over synthetic
patient records. Each episode: the agent receives a filtered observation, orders tests,
prescribes treatments, and terminates by declaring a probability distribution over
conditions (or abstaining). The episode is scored once, at termination, against hidden
ground truth.

**The research contribution is the environment, not the policy.** Specifically: verifiable
per-step rewards, a computable Bayes-optimal ceiling that doubles as a reward-hacking
detector, and a budget-conditioned cost–accuracy frontier.

> **This is a synthetic research environment.** It is not a clinical decision tool, is not
> validated against real patients, and no artifact from this repo may be presented as
> clinical guidance. Severity weights, likelihoods and contraindication rules are
> simulation parameters chosen for RL dynamics, not clinical recommendations.

---

## Status

| Phase | State |
|---|---|
| **0 — feasibility** | Gate A **passes** on both substantive criteria (see below) |
| **1 — environment** | Complete: taxonomy, catalog, action menu, observation model, Bayes, episode |
| **2 — reward engine** | Complete: scoring, costs, treatment, verify, shaping, pure engine |
| **3 — cold start** | Complete: grammar, teacher, de-leaking, rejection sampling, SFT. Gate B **pending a GPU run** |
| **4 — GRPO** | Complete: loop, monitors, curriculum, LoRA updater. Runs end to end; no GPU run yet |
| **5 — evaluation** | Audit suite, Pareto sweep, calibration complete |

296 tests (291 fast + 5 slow). `ruff` and `mypy --strict` clean. CI runs the fast suite,
lint and types on every commit and the corpus-wide suite nightly.

**Nothing here has been trained.** Phases 3 and 4 are built, tested end to end against a
grammar-sampling backend, and verified on real rollouts — but no model has been fine-tuned
and no GRPO step has taken a gradient. Every number below comes from the environment and
from heuristic policies. The GPU-only paths (`VLLMBackend`, `train_lora`,
`TorchLoRAUpdater`) are lazily imported and unexercised; treat them as unproven until they
have run.

---

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && pip install -e ".[dev]"

pytest                       # fast suite: unit + invariants, ~20s
pytest -m slow               # corpus-wide sweeps + toy-MDP policy invariance
ruff check dxenv tests && mypy

python scripts/phase0_feasibility.py --n 10000 --seed 7   # Phase 0 probes
python scripts/check_gate_a.py                            # evaluate against the gate
python scripts/regenerate_golden.py                       # deliberately, then read the diff
```

On a CUDA host (`pip install -e ".[gpu]"`), the Phase 3 → 4 path:

```bash
# 8.1 -- BEFORE any SFT. Does the base model clear the bar with spread? This decides
# whether SFT is needed at all. Without --model it reports the model-free floor instead.
python scripts/phase3_prompted_baseline.py --n 200 --k 8 --model Qwen/Qwen2.5-7B-Instruct
python scripts/check_gate_b.py --results runs/phase3/prompted_baseline.json

# only if 8.1 says SFT is needed:
python scripts/build_sft_data.py --n 2000 --frozen-split --out runs/phase3/sft.jsonl
#   ... train the LoRA (policy.sft.train_lora) ...

# Gate B PROPER -- the same measurement, now on the SFT'd policy. This is the go/no-go
# into Phase 4, and it writes to sft_baseline.json so the pre-SFT run is not overwritten.
python scripts/phase3_prompted_baseline.py --n 200 --k 8 \
    --model Qwen/Qwen2.5-7B-Instruct --lora runs/sft/final
python scripts/check_gate_b.py --results runs/phase3/sft_baseline.json

python scripts/train_grpo.py --dry-run --steps 20        # real rollouts, no gradient
python scripts/train_grpo.py --reference-adapter runs/sft/final --steps 2000
python scripts/rescore.py runs/grpo --corpus-n 20000 --corpus-seed 20260901
```

The same sweep runs twice because it answers two different questions. Before SFT: *can
the base model do this at all?* After SFT: *did SFT help without destroying what GRPO
needs?* Two of Gate B's criteria — calibration survived, and within-group variance — are
essentially free passes on a base model and only bite on the second run, because SFT is
what destroys calibration (one-hot targets) and what collapses diversity (over-training).
They are there to catch SFT damage specifically.

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

---

## Phase 0 — Gate A

Thresholds were committed **before** the probe ran; a test enforces that ordering from git
history, because that enforcement is the only thing that makes a gate a gate.

| probe | bal. acc | top-1 | top-5 | weighted Brier |
|---|---|---|---|---|
| `F` blank record | 0.0072 | 0.0330 | 0.1605 | 0.0004 |
| `V` vitals + complaint | 0.2743 | 0.4625 | 0.8395 | 0.6025 |
| `T` vitals + all tests | 0.7099 | 0.7860 | 0.9640 | 1.5054 |
| `T_with_leak` (positive control) | 0.9286 | 0.9845 | 0.9950 | 1.9140 |

- **V − F = +0.267** (threshold ≥ 0.08) — **PASS**
- **T − V = +0.436** (threshold ≥ 0.20) — **PASS**. This is the size of the entire prize.

Two things worth knowing about how this number was arrived at:

1. **The first run reported T − V = −0.002**, i.e. chance. That was the *detector*, not the
   environment: it ordinal-encoded categorical analytes (an index into an unordered
   vocabulary is not a number) and gave 149 classes to gradient boosting on too few
   samples each. The Bayes posterior on the same data reaches top-1 0.50 from vitals and
   0.91 with tests, which is what identified the harness as the problem. The encoding was
   fixed; no threshold was touched.
2. **One criterion fails as literally written** and is superseded rather than edited.
   `gate_a.yaml` requires the blank probe to land near the *majority-class rate* while
   declaring *balanced* accuracy as the metric; those floors differ (0.0381 vs
   1/149 = 0.0067), so it could never pass on its own metric. `gate_a2.yaml` records the
   correction and its reasoning; `gate_a.yaml` is untouched, both verdicts are reported,
   and a test asserts the amendment moved no substantive threshold.

The leakage ablation passes *trivially* here, because blocked resources never reach a
feature matrix at all. A trivially-satisfied leak check is not evidence of no leak, so the
probe is also run **with** the label injected and is required to gain ≥ 0.10; it gains
0.219.

---

## Reward

```
R = brier(p, c_true) · severity_weight(c_true)
    − λ · Σ cost(t) − μ · n_turns
    + treatment_score + shaping + predict_then_verify
```

`λ = 0.004`, **calibrated from measurement**. A myopic-entropy policy gains +0.76 weighted
score over the first ~18 cost units, +0.23 over the next ~36, and effectively nothing past
6 tests — so marginal value per cost unit runs 0.041 → 0.0062 → 0.0010. λ sits between the
mid and late figures, putting the optimum at 3–6 tests. At the obvious-looking λ = 1.0 no
test on the menu is ever worth ordering and the environment collapses to "guess from
vitals".

Baselines on 60 patients at B = 200:

| policy | total | diagnosis | tests |
|---|---|---|---|
| prior (lazy) | −0.29 | −0.27 | 0.0 |
| vitals-only Bayes | +0.56 | +0.58 | 0.0 |
| greedy Bayes | +1.38 | +1.59 | 6.0 |

### Shaping ships disabled — a real tension in the spec

I5 requires that ordering a test never produces positive reward *"under any shaping term"*.
Potential-based shaping with Φ = −H(posterior) sums over an episode to
`scale · (H(s₀) − H(s_T))`, which **is** total information gain — the thing I5's commentary
prohibits paying for. They reconcile only if shaping is scaled below the cheapest order's
cost: at λ = 0.004 and a cheapest test of 1.0 cost unit, one transition can gain at most
`scale · log(149)`, so strict per-step non-positivity caps `scale ≤ 0.0008`. At that scale
the whole-episode shaping contribution is under 0.004 against a diagnosis term spanning ±6.

So the machinery is built and fully tested — telescoping, closed-loop-zero, policy
invariance on a brute-forced toy MDP — and it is **off**. Enabling it with a meaningful
scale makes `validate_reward_config` raise, by design. Ng et al. (1999) guarantees the
optimal policy set is unchanged for any Φ, so enabling it is defensible; that is a decision
to take deliberately, not to inherit from a default. **This is the open question most worth
your attention.**

`validate_reward_config` proves I5 arithmetically rather than trusting a test to notice.
The predict-then-verify reward is a *fraction of each order's own cost*, so `cost + verify`
is strictly negative for every test individually — not merely for the cheapest one.

---

## The two ceilings

`env/bayes.py` computes two, and conflating them is the trap:

- **`hard_ceiling`** — the score of a perfectly confident, correct report. Sound on every
  realisation, so it is safe to assert per episode and halt on. This is what
  `train/monitors.assert_below_ceiling` uses.
- **`expected_ceiling`** — the Bayes-optimal expected score given every analyte revealed
  for free. Tight, and the number worth reporting, but **a single lucky rollout can exceed
  it**: a proper scoring rule only guarantees truthful reporting wins *on average*.
  Asserting it per episode would fire on luck, and a detector that cries wolf gets switched
  off. It is checked on running means instead (`RunningCeilingMonitor`).

The bound direction is documented where it is relied upon: more evidence never lowers the
attainable expected score (Blackwell), and costs only subtract, so the full-information
value upper-bounds every policy. `env/bayes.py` takes the scoring rule as an **injected
callable**, so `env/` never imports `reward/` and there is still exactly one implementation
of the rule.

---

## Audit suite

`python -c "from dxenv.data.corpus import generate_corpus; from dxenv.eval.audit import run_audit; print(run_audit(generate_corpus(60, seed=777)).render())"`

All seven probes pass. Two are worth calling out, because both **failed on first run and
both failures were the probe, not the environment**:

- *counterfactual perturbation* ranked candidate conditions by the **mean** of an analyte,
  but Bayes moves mass by **likelihood at the observed value**. A condition with mean 1200
  and sd 900 has low density at 42, so the probe had the direction wrong and reported 61%.
  Corrected, it is 200/200 — and it now also carries six named clinical spot checks,
  because the generic property is guaranteed by Bayes and would pass even if every
  override in `obs_overrides.yaml` were attached to the wrong condition.
- *blank-record baseline* read −0.28 against an analytic floor of +0.001, because the
  baseline policy truncated its report to a top-8 — which costs an uninformed policy far
  more than an informed one and depresses the floor everything else is measured against.

Every probe that could pass trivially carries a positive control. An audit suite that would
not catch a real failure is worse than none, because it manufactures confidence.

---

## Phase 3 — cold start

### The prompted baseline (CLAUDE.md 8.1)

Run on 200 patients, k=8. `random_schema` is the grammar with no policy behind it —
format-valid and uninformed, which is a different and more useful floor than the prior.

| policy | mean R | best@8 | group std | tests |
|---|---|---|---|---|
| prior (blank-record floor) | −0.018 | −0.018 | 0.000 | 0.00 |
| vitals-only Bayes (**the Gate B bar**) | +0.675 | +0.675 | 0.000 | 0.00 |
| greedy Bayes | +1.183 | +1.183 | 0.000 | 5.76 |
| random_schema | −0.385 | −0.044 | 0.322 | 0.37 |

Gate B evaluates but does **not** pass, and the honest reading is that it cannot yet:
the subject row is a uniform sampler over the grammar, which clears the vitals-only bar on
1% of patients at both k=1 and k=8. Five of six criteria pass on it — spread, calibration,
headroom, schema validity — and pass@k does not, because there is no policy. The checker
says so explicitly rather than printing a bare FAIL. **Re-run with `--model` to evaluate
the gate as written.**

The two things that row does establish: the grammar produces 100% parseable output over
1,600 generations, and it produces real within-group spread (0.322), which is the
precondition GRPO needs and the thing that would be missing if the schema over-constrained
the model.

### The de-leaking check that CLAUDE.md asks for does not work as written

CLAUDE.md 8.2 wants a trace that never "references the answer before earning it", and the
literal reading — the reasoning must not contain the condition's name — fails on contact.
A de-leaked trace names the leading hypotheses from the *visible* posterior, and the
visible posterior is usually right. Enforced literally it rejected **19 of 20 clean
traces**, and it would forbid the model from ever writing a differential, which is the
entire content of diagnostic reasoning.

The distinction that matters is counterfactual: is the condition named *because the
teacher knew*, or because the evidence ranked it? Three checks, increasing in strength:

| check | privileged trace | de-leaked | what it is for |
|---|---|---|---|
| literal substring + similarity | 252 findings | 0 | the **positive control** — it must fire |
| grounding (rank + attached probability) | 56 findings | 0 | the filter the SFT set runs |
| rank-matched ablation | gap **+0.732** | **+0.036** | survives an LLM teacher |

The grounding filter requires every named condition to sit in the visible posterior's
top-*n* **with its posterior probability attached**. An assertion of fact carries no
probability and is rejected; a hypothesis the evidence does not support ranks too low and
is rejected. `deleak_is_label_blind` is the exact version: swap the true label, and this
module's deterministic de-leaker returns byte-identical reasoning.

**The obvious ablation null is wrong.** Comparing against a uniformly random condition
reports a gap of **+0.63** on a de-leaker that is label-blind by construction — because
the true condition really is usually near the top. Drawing the null *from the visible
posterior* holds rank fixed and the gap collapses to +0.036. A check that fails on correct
behaviour is a check that gets switched off.

Two related bugs, both the same shape and both caught by these tests: `leak_strings`
contains "mi" and "all", and substring matching fired them inside "com**mi**t" and "at
**all**", rejecting three-quarters of a clean SFT set. Prose matching is now word-boundary,
and *mention* detection uses the taxonomy's narrow forms while the observation scrubber
keeps the wide ones — the asymmetry runs opposite ways in the two places, and the taxonomy
already encoded that distinction.

### How wide a report may be

`DEFAULT_MAX_LABELS = 16`, chosen from measurement. Unnamed posterior mass at the initial
observation, over 300 patients:

| k | mean tail | p90 | max |
|---|---|---|---|
| 4 | 0.204 | 0.523 | 0.690 |
| 8 | 0.107 | 0.331 | 0.512 |
| **16** | **0.044** | **0.152** | **0.327** |
| 32 | 0.011 | 0.018 | 0.150 |

16 roughly halves the p90 tail against 8 for ~200 extra tokens on the one diagnose turn of
an episode; 32 halves it again but the token cost lands on every rollout of every group.

The residual is **not renormalised away**. A report naming sixteen conditions totalling
0.7 has said "0.3 of my belief is elsewhere", and renormalising would convert that into a
confidence the agent never claimed — which a proper scoring rule would duly reward when it
happened to be right. The unnamed mass is spread uniformly instead: the max-entropy
completion of what was actually said. So the SFT target is the posterior's top-16 *plus a
max-entropy tail*, not the posterior; measured mean total-variation distance from the true
posterior is 0.018, worst case 0.218, both bounded by the unnamed mass. Saying "the target
is the posterior" would overstate it.

---

## Phase 4 — GRPO

The loop runs end to end today, against a grammar-sampling backend, on the frozen split:

```
eval split verified against its committed hash; {"train": 14854, "eval": 3713, "holdout_modules": 1433}
step     0 [single_condition_short] R=-0.254 dx=-0.205 tests=0.25 group_std=0.071 degen=0.0% gap=+1.335 seqs=19
step     1 [single_condition_short] R=-0.531 dx=-0.468 tests=0.33 group_std=0.127 degen=0.0% gap=+2.316 seqs=28
```

**Orchestration is separated from the gradient step.** `GRPOTrainer` owns everything that
can be wrong without a GPU — which patients are sampled, whether the eval split was
touched, how advantages are formed, when a monitor halts, what gets persisted — and the
gradient arrives as an `Updater`. `NullUpdater` runs the whole loop with no model at all.

That is not a testing convenience. The failures this project is exposed to are leakage,
reward hacking, and a monitor that would not have fired; none of them live in the backward
pass, and all of them would otherwise be untestable without eight hours on an A100.
`test_ceiling_assertion_fires_on_synthetic_violation` and
`test_training_never_reads_eval_split` both run in the fast suite because of this split.

Monitors halt; none of them warn:

| monitor | fires on | why it is not per-episode |
|---|---|---|
| hard ceiling [I9] | reward above a perfectly confident correct answer | it *is* per-episode — sound on every realisation |
| running expected ceiling | mean reward above the mean Bayes value | a lucky rollout may beat it; a running mean may not |
| degenerate groups | >50% of a window with zero reward spread | one flat group is an easy patient, not a collapse |
| cost distribution | collapse to zero tests **or** to the budget cap | both ends are failures, and they look nothing alike |

Credit assignment is one episode-level advantage broadcast uniformly across every token
the episode generated. Standard for multi-turn GRPO, and stated as an assumption rather
than inherited as a default: it says a good episode makes each of its turns slightly more
likely, including the turns incidental to why it was good. The alternative — per-turn
credit from a learned value head — reintroduces a learned model into a reward pipeline
whose entire premise is that reward is verifiable.

KL uses the k3 estimator, `exp(r) − r − 1`. The naive difference is also unbiased but goes
negative on individual tokens, so the penalty occasionally *pays* the policy for leaving
the reference — a small effect and a very odd one to debug.

Three things in the GPU updater worth knowing before it runs, all of which would have
failed **silently**:

- **Rollout weights are synced every step.** `sync_rollout_weights` existed on the
  protocol and both implementations and nothing called it; the vLLM `LoRARequest` also
  pinned adapter id 1, which vLLM caches. Rollouts would have come from the frozen SFT
  reference all run while the trained adapter drifted away — no crash, just a run that
  quietly isn't GRPO.
- **There is no separate reference model.** `get_peft_model` injects LoRA into the base
  *in place*, so holding a reference to it aliases the modules the adapter now lives in —
  the reference forward pass would run with the trainable adapter active and KL would read
  0.000 forever. Reference logprobs come from `disable_adapter()`, which also means one
  copy of the weights rather than two.
- **The clipping is inert at one inner epoch.** `old_logp` is the batch's own detached
  logprobs, so the ratio is identically 1 and this reduces to a plain policy gradient.
  Correct single-epoch GRPO; `clip_eps` starts mattering the moment a second inner epoch
  is added.

`VLLMBackend.gpu_memory_utilization` defaults to 0.55, below vLLM's own default, because
in a GRPO run the engine shares a device with the trainer and vLLM preallocates its KV
cache at startup. Raise it for a standalone eval sweep.

---

## Persistence, and why it came first

`runs/{run_id}/episodes.jsonl`, one JSON line per episode, under pinned config hashes.
Reward is pure [I8], so rescoring a stored corpus under new weights is free; regenerating
rollouts is not, and on a 7B policy it is the dominant cost of the whole project.

Verified on a real 6,600-episode store: rescoring reproduces the stored totals to
`mean_delta 0.0`.

Ground truth lives on the line, so a run is self-contained. That has one sharp edge — **an
episodes.jsonl file is not safe to feed to a model** — and the rule is structural rather
than advisory: `stored_trajectory()` returns the trajectory alone and is the only accessor
the rollout and training paths use.

The store's config guard caught something real the first time the loop ran: a curriculum
stage changes `max_turns`, which changes the episode config hash, so one run legitimately
writes lines under several. The fix was to **declare** every stage's hash at run start
rather than loosen the check to a warning — an undeclared hash still fails, because it
means an episode was generated under a configuration nobody intended.

---

## Layout

```
dxenv/
  data/      taxonomy (149 flat labels) · corpus · splits · store (JSONL) · eval_split.json
  env/       filter [I1,I2] · actions [I3] · obs_model [I4] · bayes [I9] · episode · schemas
  reward/    scoring [I7] · costs [I5] · treatment · verify · shaping [I6] · engine [I8]
  policy/    prompt · decoding (grammar) · llm (vLLM + grammar sampler) · rollout ·
             teacher (privilege + de-leak) · rejection · sft · baselines
  train/     grpo (loop + LoRA updater) · monitors · curriculum
  eval/      audit · pareto · calibration
  configs/   gate_a · gate_a2 · gate_b · severity · reward · costs · env · treatments
tests/       invariants (one file per I1–I12) · unit · property · golden
scripts/     phase0_feasibility · check_gate_a · freeze_eval_split ·
             phase3_prompted_baseline · check_gate_b · build_sft_data · train_grpo ·
             rescore · regenerate_golden
```

Module rules enforced by `tests/unit/test_module_boundaries.py`, which parses the source
rather than importing it, so a violation is caught even inside a function: `reward/` never
imports `policy/` or `train/`; `env/` never imports `reward/`; `data/` depends on nothing
above it. `env.step()` deliberately returns **no reward** — the environment produces
trajectories, the reward engine scores them. The same test asserts that no module reads
config at import time.

Everything GPU-shaped is behind a lazy import and the `[gpu]` extra, so the full invariant
suite, both gates and the whole GRPO loop run on a laptop.

---

## Known gaps

- **Nothing has been trained.** The GPU paths — `VLLMBackend`, `sft.train_lora`,
  `TorchLoRAUpdater` — are lazily imported and have never executed. They are written
  against current vLLM/TRL/PEFT APIs and are unproven until they run.
- **Gate B is not evaluated.** It is pre-registered, committed before any measurement, and
  the checker works; the subject row is a grammar sampler, not a model.
- **`data/snomed_map.yaml` is empty.** The generator drives everything today, so nothing
  depends on it; ingesting real Synthea output requires populating it first, and
  `map_snomed` raises on any unmapped code rather than silently dropping the condition.
- **The teacher is heuristic, not an LLM.** It picks tests by likelihood ratio at the
  patient's actual value and reports the Bayes posterior. Its de-leaked reasoning is
  templated from the visible posterior, which is why `deleak_is_label_blind` can be an
  exact check. Swapping in an LLM teacher makes that check unavailable and leaves
  `deleak_ablation` as the backstop — which is why the ablation exists.
- **`obs_model` conditions on the condition alone** — no age or sex effects on
  likelihoods. Consistent and leak-free, but less realistic than it could be.
- **Comorbidity is unimplemented.** The curriculum declares a `comorbid` stage; the
  generator emits one condition per patient.
- **Likelihood parameters are invented.** They are consistent and produce the intended
  RL dynamics; they are not drawn from published likelihood ratios except incidentally.

## Open decisions (CLAUDE.md §12)

Resolved here, and flagged as reversible: label set = **149**; `optimal_stopping_value` is
**bounded, not exact** (full-information Bayes value, direction proved in the docstring);
severity = **4 tiers at 1.0 / 1.8 / 3.2 / 6.0**; `p(B)` = **discrete mixture** over
[10, 25, 50, 100, 200]; report width = **16 labels**, sized from the tail-mass measurement
above. Still open: whether the near-miss cost matrix ships in v1 and **whether to enable
shaping**. Whether Phase 3 SFT is needed is now answerable rather than open — run 8.1 with
`--model` and read Gate B.
