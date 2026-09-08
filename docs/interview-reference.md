# dxenv — complete reference

Everything this project contains, why each decision was made, and what is wrong with it.
Written to be **studied**, not skimmed: the "why" lines are the ones interviews probe, and
the caveats section is the one that decides whether you sound like you built this or
inherited it.

Companion documents: [`CLAUDE.md`](../CLAUDE.md) is the spec, [`design-notes.md`](design-notes.md)
is the engineering record, [`README.md`](../README.md) is the overview. This file is the
superset, organised for recall.

---

## Contents

1. [The 60-second version](#1-the-60-second-version)
2. [What the environment is](#2-what-the-environment-is)
3. [The twelve invariants](#3-the-twelve-invariants)
4. [Architecture and module boundaries](#4-architecture-and-module-boundaries)
5. [Component by component](#5-component-by-component)
6. [The reward engine](#6-the-reward-engine)
7. [Phase 0 and Gate A](#7-phase-0-and-gate-a)
8. [Phase 3 — cold start](#8-phase-3--cold-start)
9. [Gate B, measured three times](#9-gate-b-measured-three-times)
10. [Phase 4 — GRPO](#10-phase-4--grpo)
11. [Phase 5 — evaluation and the audit suite](#11-phase-5--evaluation-and-the-audit-suite)
12. [Infrastructure](#12-infrastructure)
13. [Every caveat](#13-every-caveat)
14. [War stories](#14-war-stories)
15. [Numbers worth memorising](#15-numbers-worth-memorising)
16. [Likely interview questions](#16-likely-interview-questions)

---

## 1. The 60-second version

A multi-turn RL environment where an LLM plays diagnostician over synthetic patient
records. Each episode the agent sees a filtered patient view, decides which tests are worth
their cost, and terminates by reporting a **probability distribution** over 149 conditions
— or abstaining. It is scored once, at the end, against hidden ground truth.

**The contribution is the environment, not the policy.** Three properties make it worth
building:

- **Verifiable reward.** Every term is a pure function of ground truth. No learned reward
  model, no LLM judge.
- **A computable Bayes-optimal ceiling** that doubles as an automatic reward-hacking
  detector. The environment knows the best score achievable by perfect reasoning over the
  same evidence, so an agent that beats it has information it should not have.
- **A budget-conditioned cost–accuracy frontier.** Budget is sampled per episode and
  exposed to the agent, so one policy spans the whole frontier and evaluation sweeps it.

The hard part is that the data generator (Synthea) writes records from explicit disease
modules, so the diagnosis appears in the record in about six places — and the *sparsity
pattern* leaks it in a seventh. Most of the engineering exists to close those channels,
because a leaky environment produces excellent-looking numbers that mean nothing, and the
failure is silent.

---

## 2. What the environment is

### The episode loop

1. Agent receives a filtered observation: demographics, vitals, presenting complaint,
   family history, allergies, remaining budget, turns remaining, menu fingerprint.
2. Each turn it takes one action from a **global** menu: order a test (with a mandatory
   prediction of the result), prescribe a treatment, `diagnose` with a distribution, or
   `abstain`.
3. Episode ends on `diagnose`, `abstain`, `max_turns`, `budget_exhausted`, or
   `decode_failure`.
4. It is scored **once**, at termination. `env.step()` returns no reward at all.

### Sizes

| | |
|---|---|
| Condition labels | **149**, flat |
| Menu actions | **147** |
| Orderable tests | **73** |
| Max turns | 20 (8 in curriculum stage 1) |
| Budget support | {10, 25, 50, 100, 200}, weights {.15, .25, .30, .20, .10} |
| Corpus split | train 14,854 · eval 3,713 · holdout-modules 1,433 |
| Tests in the suite | 326 fast + 5 slow = **331** |

### Why `env.step()` returns no reward

The environment produces trajectories; the reward engine scores them. Keeping these apart
is what makes **offline rescoring free** — and you will change reward weights repeatedly,
while regenerating rollouts on a 7B policy is the dominant cost of the whole project.
Verified on a real 6,600-episode store: rescoring reproduces stored totals to
`mean_delta 0.0`.

### Termination reason `decode_failure`

If the policy produces nothing the grammar and parser can turn into an action, that is
recorded as a non-decision — like running out of turns. **Not** repaired, retried, or
replaced with a default action. Inventing an action on the policy's behalf would put a
decision in the trajectory that the policy never made, converting a loud findable bug into
a quiet distributional shift in the training data.

---

## 3. The twelve invariants

Each has its own test file in `tests/invariants/`. These are correctness properties, not
style preferences.

| # | Invariant | Why it exists | How it is enforced |
|---|---|---|---|
| **I1** | Ground truth never enters an observation | the label is the whole game | *structural*: `Observation` has no field that could hold it — no `condition`, no `reason`, no free text |
| **I2** | Observation built by **allowlist**, fails closed | denylists miss the field nobody thought of | unknown resource type **raises** in strict mode |
| **I3** | Action menu global and identical for every patient | a per-patient menu *is* the diagnosis | set equality of action ids across 1000 patients |
| **I4** | Every test returns a value for every patient | otherwise the *sparsity pattern* leaks the label | no `None`, no "unavailable", no default-to-normal |
| **I5** | Ordering a test never produces positive reward | information-gain bonuses are farmable | property test over every action × reachable state, plus an arithmetic proof in `validate_reward_config` |
| **I6** | Shaping must be potential-based `γΦ(s′) − Φ(s)` | Ng et al. (1999): optimal policy set preserved | telescoping + closed-loop-zero + toy-MDP policy invariance |
| **I7** | Terminal scoring uses a strictly proper rule (Brier) | rules out hedging *mathematically*, not heuristically | properness property test — the single most important test in the repo |
| **I8** | Reward is a pure function of (trajectory, truth, config) | makes offline rescoring sound | same input twice → identical output; no RNG or clock reachable |
| **I9** | Episode reward never exceeds the Bayes ceiling | the automatic hacking detector | halts the run and dumps the trajectory |
| **I10** | Episodes deterministic given (patient, seed, config_hash) | reproducibility across machines | verified bit-for-bit across macOS/numpy 1.26 and Linux/numpy 2.4 |
| **I11** | Reward finite and bounded | NaN/inf silently clipped is the worst outcome | hard failure, never clipped away |
| **I12** | Eval split frozen and hash-verified; training never reads it | the split is the only honest number | monkeypatch the loader to raise on eval paths |

### The two that get violated by accident

**I5** is the one. Every reward-hacking story in this environment starts with someone
adding a plausible-looking "reward informative tests" term. The agent will then find tests
that maximise entropy reduction under the belief model without improving the answer. A test
pays for itself **only** by improving the terminal score enough to cover its cost.

**I4** is the subtle one. If some tests return "unavailable" for some patients, then *which
tests returned anything* is itself a feature correlated with the condition — the model
learns the sparsity pattern rather than the values. The direct test for this trains a
classifier on which tests returned values, **ignoring the values themselves**, and requires
chance performance.

---

## 4. Architecture and module boundaries

```
dxenv/
  data/      taxonomy (149 flat labels) · corpus · splits · store (JSONL) · eval_split.json
  env/       filter [I1,I2] · actions [I3] · obs_model [I4] · bayes [I9] · episode · schemas
  reward/    scoring [I7] · costs [I5] · treatment · verify · shaping [I6] · engine [I8]
  policy/    prompt · decoding · llm · rollout · teacher · rejection · sft · baselines
  train/     grpo (loop + LoRA updater) · monitors · curriculum
  eval/      audit · pareto · calibration
  configs/   gate_a · gate_a2 · gate_b · gate_b2 · severity · reward · costs · env · treatments
```

### The import rules, and why each one earns its keep

- `reward/` must not import `policy/` or `train/`.
- `env/` must not import `reward/` — **the environment produces trajectories, the reward
  engine scores them.** This is what makes offline rescoring possible.
- `data/` depends on nothing above it.
- Anything that reads config takes it as an argument. No module-level config reads.

These are enforced by a test that **parses the source**, so a violation is caught even
inside a function body.

### The one place the boundary was awkward, and how it was resolved

`env/bayes.py` needs a scoring rule to compute the ceiling, but `env/` may not import
`reward/`. Solution: bayes takes the scoring rule as an **injected callable**. So the
boundary holds *and* there is still exactly one implementation of the rule.

### `costs.yaml` is read by both sides

`env/episode.py` reads it for the budget ledger (what an order costs to place) and
`reward/costs.py` reads it for the reward term (what an order costs you). **One file**, so
the two cannot drift, without either importing the other.

---

## 5. Component by component

### 5.1 `data/taxonomy.py` — 149 flat labels

**Flat is load-bearing.** A hierarchy lets an agent hedge upward ("endocrine disorder") and
collect partial credit everywhere, which defeats the proper scoring rule.

The taxonomy encodes **narrow** and **wide** mention forms per label, and the asymmetry runs
opposite ways in two places: the observation scrubber uses the *wide* forms (scrub
aggressively), while leak *detection* in traces uses the *narrow* forms (a differential
naming a hypothesis is not a leak). Getting this backwards rejects clean data.

### 5.2 `env/filter.py` — the observation builder [I1, I2]

Allowlist of `(resource_type, field)` pairs. Permitted: demographics, vitals, presenting
complaint, prior lab *values*, family history.

Blocked, each because it is the label in disguise:

| Field | Why |
|---|---|
| `Condition` | The label. |
| `MedicationRequest` | Metformin *is* a diabetes diagnosis. The one people forget. |
| `Encounter.reasonCode` / `reasonReference` | The record's own explanation of the visit. |
| `CarePlan` | Named after the condition. |
| `Procedure` | Dialysis implies renal failure. |
| `DiagnosticReport.conclusion` | The answer, already written out. |
| `CareTeam` | "Oncology team" narrows things considerably. |

**Strings leak even when fields don't.** A lab value is fine; a lab `display` reading
"HbA1c — diabetes monitoring" is not. Display strings come from the **catalog**, never
lifted from the patient record.

`test_no_label_string_in_observation` runs over the **full corpus**, not a sample — a leak
appearing in 2% of patients is still a leak.

### 5.3 `env/actions.py` — the global menu [I3]

147 actions. Ids are **content-hashed, not positional**, so adding a test does not silently
renumber the others and invalidate every stored trajectory.

The observation carries a `menu_fingerprint` so a trajectory can be checked against the menu
it was generated under.

### 5.4 `env/obs_model.py` — generative `p(result | condition)` [I4]

Full cross product of tests × conditions, no fallbacks, no `KeyError` paths. Two table
types: `CatTable` (categorical) and `QuantTable` (quantitative, with `log_likelihood`).

This is what makes I4 achievable: every test returns something for every patient, so the
sparsity pattern carries no information.

### 5.5 `env/bayes.py` — posterior and **two** ceilings [I9]

This module has **four downstream consumers**: the ceiling (Phase 5), the shaping potential
(Phase 2), SFT soft labels (Phase 3), and the rejection-sampling process filter (Phase 3).
It is the critical path, not preprocessing.

**Conflating the two ceilings is the trap:**

| | what it is | assert on it? |
|---|---|---|
| `hard_ceiling` | score of a perfectly confident, correct report | **yes, per episode** — sound on every realisation, safe to halt on |
| `expected_ceiling` | Bayes-optimal expected score with every analyte revealed free | **no, per episode** — a single lucky rollout can legitimately beat it |

A proper scoring rule only guarantees truthful reporting wins **on average**. Asserting the
expected ceiling per episode would fire on luck, and *a detector that cries wolf gets
switched off*, which is worse than not having one. It is checked on running means instead.

**Bound direction is proved, not assumed:** more evidence never lowers the attainable
expected score (Blackwell), and costs only subtract, so the full-information value
upper-bounds every policy. `optimal_stopping_value` is **bounded, not exact** — documented,
because if the bound pointed the other way I9 would be unsound.

### 5.6 `env/episode.py`

Gym-style. Handles the turn loop, budget ledger, dedup, termination.

**Dedup matters more than it looks.** The second order of the same test costs nothing and
returns the cached result. Without it the agent finds the cheapest test and spams it.

---

## 6. The reward engine

```
R = brier(p, c_true) · severity_weight(c_true)
    − λ · Σ cost(t) − μ · n_turns
    + treatment_score
    + shaping                      # telescopes [I6] — SHIPS DISABLED
    + predict_then_verify
```

### 6.1 Brier, not log-loss

Log-loss is **unbounded below**. One rollout putting ≈0 on the truth produces a huge
negative, which wrecks GRPO's advantage normalisation (advantages are standardised within
the group, so one outlier flattens every other sample's advantage to ≈0). Brier is bounded
[I11].

Brier here is the **negative** of the usual sum-of-squares loss, shifted so a uniform report
over the flat label set scores exactly 0. Confident-correct is positive; confident-wrong is
negative and bounded.

**Properness** is the property that rules out hedging *mathematically* rather than
penalising it heuristically: the unique expected-score maximum is reporting your true
belief. `test_brier_is_proper` samples random true beliefs `q` and perturbations `p ≠ q` and
requires reporting `q` to strictly win.

### 6.2 Severity weights — an explicit value judgement

Four tiers: **1.0 / 1.8 / 3.2 / 6.0**.

Without them the score-maximising policy nails common benign conditions and eats the
rare-severe tail, because the tail is rare — the wrong incentive, and invisible in aggregate
accuracy.

The 6× ratio is the only real choice: too flat and the tail is ignored; too steep and the
agent guesses "emergency" on every ambiguous presentation, because a low-probability
high-weight condition beats a confident benign call. **6× keeps a tier-4 guess from paying
off below roughly a 1-in-6 posterior.**

The weight keys on the **true** condition, not the reported one, so the agent cannot inflate
its score by reporting high-urgency conditions.

### 6.3 λ = 0.004, calibrated from measurement

A myopic-entropy policy gains **+0.76** weighted score over the first ~18 cost units,
**+0.23** over the next ~36, and effectively nothing past 6 tests. So marginal value per
cost unit runs 0.041 → 0.0062 → 0.0010. **λ sits between the mid and late figures**, which
puts the optimum at an *interior* 3–6 tests.

At the obvious-looking λ = 1.0, no test on the menu is ever worth ordering and the whole
environment collapses to "guess from vitals" — the frontier becomes a point.

μ = 0.02 per turn, deliberately small relative to test costs: *the thing being priced is
investigation, not deliberation.*

### 6.4 Shaping ships **disabled** — a genuine tension in the spec

I5 requires that ordering a test never produces positive reward "under any shaping term".
But potential-based shaping with Φ = −H(posterior) sums over an episode to
`scale · (H(s₀) − H(s_T))`, which **is total information gain** — exactly what I5's
commentary prohibits paying for.

They reconcile only if shaping is scaled below the cheapest order's cost. At λ = 0.004 and a
cheapest test of 1.0 cost unit, one transition can gain at most `scale · log(149) ≈ 5·scale`,
so strict per-step non-positivity caps **scale ≤ 0.0008**. At that scale the whole-episode
contribution is under 0.004 against a diagnosis term spanning ±6. *It would be a no-op
dressed up as a mechanism.*

So the machinery is built and fully tested — telescoping, closed-loop-zero, policy
invariance on a brute-forced toy MDP — and it is **off**. Enabling it with a meaningful
scale makes `validate_reward_config` **raise**, by design.

> This is the open question most worth attention. Ng et al. guarantees the optimal policy
> set is unchanged for any Φ, so enabling it is defensible — but that is a decision to take
> deliberately, not to inherit from a default.

### 6.5 Predict-then-verify

The agent commits to a coarse prediction (`low`/`normal`/`high`/…) **before** the result is
revealed. Unhackable because ground truth is hidden until after commitment. The commitment
is **mandatory at the schema level** — `OrderTest` has a required `prediction` field — so
`test_commit_is_mandatory` cannot be satisfied by a runtime check someone later forgets to
call.

The reward is a **fraction of each order's own cost** (0.25), not a flat scale. A flat scale
would have to sit below the cheapest test's price to satisfy I5, making it negligible for
exactly the expensive tests where committing to a prediction matters. As a fraction below 1,
`cost + verify` is strictly negative **for every test individually** — not merely for the
cheapest one.

Side effect worth naming: an agent that can predict a result exactly has learned the test is
redundant.

### 6.6 Treatment — scored twice

Once for **coherence** with the *declared* diagnosis (scale 0.5), once for correctness
against the **true** condition (scale 1.0). A lucky-correct treatment under a wrong diagnosis
does not collect.

Contraindication penalty is **4.0**, asymmetric and large: a contraindicated prescription
must score strictly worse than a suboptimal-but-safe one **at every level of diagnostic
accuracy**. Allergy list, eGFR, pregnancy status and drug–drug interactions are all
verifiable from the record.

### 6.7 Abstain, priced deliberately

Value 0.0 with a 0.05 penalty on top, so abstaining is a real option under genuine
uncertainty but never a free ride — and a policy that abstains on everything does strictly
worse than one reporting the prior. Otherwise abstain becomes the lazy attractor.

### 6.8 Budget conditioning

`B ~ p(B)` per episode, exposed in the observation. **One policy spans the whole frontier**
and eval sweeps `B` to trace a Pareto curve.

Discrete mixture rather than continuous, so eval sweeps exactly the support and Pareto points
are directly comparable across runs. Weighted toward the middle because the extremes teach
degenerate policies: at B=0 the only move is to guess the prior, and at B=∞ ordering
everything is optimal.

`test_policy_behavior_varies_with_budget` — a policy whose test count doesn't respond to `B`
is ignoring the constraint and getting reward another way.

---

## 7. Phase 0 and Gate A

**Purpose: determine whether a learnable problem exists, before building anything.**

Thresholds committed **before** the probe ran, and a test enforces that ordering from git
history — *that enforcement is the only thing that makes a gate a gate.*

| probe | bal. acc | top-1 | top-5 | weighted Brier |
|---|---|---|---|---|
| `F` blank record | 0.0072 | 0.0330 | 0.1605 | 0.0004 |
| `V` vitals + complaint | 0.2743 | 0.4625 | 0.8395 | 0.6025 |
| `T` vitals + all tests | 0.7099 | 0.7860 | 0.9640 | 1.5054 |
| `T_with_leak` (positive control) | 0.9286 | 0.9845 | 0.9950 | 1.9140 |

- **V − F = +0.267** (threshold ≥ 0.08) — **PASS**
- **T − V = +0.436** (threshold ≥ 0.20) — **PASS**. *This is the size of the entire prize.*
- Leak positive control gains **+0.219** (threshold ≥ 0.10) — **PASS**

### Two things worth knowing

**1. The first run reported T − V = −0.002, i.e. chance.** That was the *detector*, not the
environment: it ordinal-encoded categorical analytes (an index into an unordered vocabulary
is not a number) and gave 149 classes to gradient boosting on too few samples each. The
Bayes posterior on the same data reaches top-1 0.50 from vitals and 0.91 with tests, which
is what identified the harness as the problem. **The encoding was fixed; no threshold was
touched.**

**2. One criterion fails as literally written, and was superseded rather than edited.**
`gate_a.yaml` requires the blank probe near the *majority-class rate* while declaring
*balanced* accuracy as the metric. Those floors differ (0.0381 vs 1/149 = 0.0067), so the
criterion could never pass on its own metric regardless of what the environment does.
`gate_a2.yaml` records the correction; `gate_a.yaml` is untouched; both verdicts are
reported; and a test asserts the amendment **moved no substantive threshold**.

### The trivially-satisfied leak check

The leakage ablation passes *trivially* here, because blocked resources never reach a feature
matrix at all. **A trivially-satisfied leak check is not evidence of no leak** — so the probe
is also run *with* the label injected and required to gain ≥ 0.10. An ablation that cannot
detect a leak it was handed proves nothing.

---

## 8. Phase 3 — cold start

### 8.1 Format is not an SFT problem

Constrained decoding (xgrammar via vLLM) with a JSON schema per action type. This makes
invalid output impossible, keeps the format reward at exactly zero, and — critically —
**preserves the entropy GRPO needs.** SFT on format burns diversity; GRPO computes
advantages from within-group variation, and identical rollouts give zero gradient.

Key facts about the grammar:

- **`WIRE_KEY_ORDER`** pins serialisation to schema declaration order. (`sort_keys=True`
  wrecked the first SFT — see War stories.)
- **`MAX_REASONING_CHARS = 700`**, bound as a compiled **`pattern`**, not `maxLength`.
  `maxLength` is advisory and grammar backends do not enforce it; `pattern` *is* compiled
  into the grammar, so the bound becomes structural.
- **`DEFAULT_MAX_LABELS = 16`**, sized from measured tail mass.
- `parse_action` uses `json.loads(text, strict=False)` — grammar backends vary in how much
  of a regex they honour, so parse tolerantly and record failures rather than crashing.

#### How wide a report may be

Unnamed posterior mass at the initial observation, over 300 patients:

| k | mean tail | p90 | max |
|---|---|---|---|
| 4 | 0.204 | 0.523 | 0.690 |
| 8 | 0.107 | 0.331 | 0.512 |
| **16** | **0.044** | **0.152** | **0.327** |
| 32 | 0.011 | 0.018 | 0.150 |

16 roughly halves the p90 tail against 8 for ~200 extra tokens on the *one* diagnose turn;
32 halves it again but the token cost lands on **every rollout of every group**.

#### The residual is **not renormalised away**

A report naming sixteen conditions totalling 0.7 has said "0.3 of my belief is elsewhere".
Renormalising would convert that into a confidence the agent never claimed — which a proper
scoring rule would duly reward when it happened to be right. The unnamed mass is spread
**uniformly** instead: the max-entropy completion of what was actually said.

Measured mean total-variation distance from the true posterior: **0.018**, worst case 0.218,
both bounded by the unnamed mass. Saying "the SFT target is the posterior" would overstate
it — it is *the posterior's top-16 plus a max-entropy tail*.

### 8.2 Privileged teacher and de-leaking

Teacher sees ground truth, produces expert test sequences, then the privilege is **stripped**:
reasoning is regenerated conditioned only on what was visible at that turn.

> Leaked reasoning in SFT data is worse than no SFT data — it trains the model to assert
> conclusions it has no evidence for, which is precisely the pathology the environment
> exists to prevent.

**The check CLAUDE.md asks for does not work as written.** The literal reading — the
reasoning must not contain the condition's name — fails on contact. A de-leaked trace names
the leading hypotheses from the *visible* posterior, and the visible posterior is usually
right. Enforced literally it rejected **19 of 20 clean traces**, and it would forbid the
model from ever writing a differential, *which is the entire content of diagnostic
reasoning*.

The distinction that matters is **counterfactual**: is the condition named *because the
teacher knew*, or because the evidence ranked it? Three checks of increasing strength:

| check | privileged trace | de-leaked | what it is for |
|---|---|---|---|
| literal substring + similarity | 252 findings | 0 | the **positive control** — it must fire |
| grounding (rank + attached probability) | 56 findings | 0 | the filter the SFT set runs |
| rank-matched ablation | gap **+0.732** | **+0.036** | survives an LLM teacher |

The **grounding filter** requires every named condition to sit in the visible posterior's
top-*n* **with its posterior probability attached**. An assertion of fact carries no
probability and is rejected; a hypothesis the evidence does not support ranks too low and is
rejected.

`deleak_is_label_blind` is the exact version: swap the true label, and the deterministic
de-leaker returns **byte-identical** reasoning.

#### The obvious ablation null is wrong

Comparing against a **uniformly random** condition reports a gap of **+0.63** on a de-leaker
that is label-blind by construction — because the true condition really is usually near the
top. Drawing the null **from the visible posterior** holds rank fixed and the gap collapses
to **+0.036**. *A check that fails on correct behaviour is a check that gets switched off.*

`min_gain = 0.15` in `PrivilegedTeacher` is why the teacher stops early — and why SFT left
the policy at 0.79 tests per episode.

### 8.3 Rejection sampling

Sample k=8 at high temperature. **Filtering on correct diagnosis alone selects for lucky
guesses and shotgun test-ordering** — a trajectory that got it right after 40 tests is a bad
demonstration, and that habit is stubborn once trained in.

Filter on: full reward including cost; process validity (does the Bayes posterior move the
right direction after each test?); reproducibility across the k samples; condition balance.

Condition balance matters because otherwise the SFT set is dominated by whatever the
generator produces most, and the model learns **the prior instead of the reasoning**.

### 8.4 SFT with Bayes-posterior soft labels

**Targets are the Bayes posterior, not one-hot.** SFT on winning trajectories otherwise
teaches the model to say 0.99 every time, destroying the calibration the Brier score exists
to reward — *before RL even starts.*

- `soft_label_wire` computes the posterior to 9 decimal places.
- `seed_abstentions` seeds abstention explicitly, or the action is never sampled and RL never
  discovers it.
- `SFTDataset.validate()` asserts `mean_target_entropy` above `DEFAULT_ENTROPY_FLOOR = 0.25`.
- 1–2 epochs, low LR, LoRA, stop early. **Deliberately undertrained** — the goal is a
  competent prior, not a finished policy.

`_trl_config` splits TRL config fields into **essential** (raise if unsupported) and
**optional** (drop, loudly). TRL's `SFTConfig` field names churn between versions; silently
dropping a field that defines the run is how you get a training run that isn't the one you
configured.

---

## 9. Gate B, measured three times

Pre-registered in `configs/gate_b.yaml` on 2026-09-03. Protocol: 200 patients from the
**train** split (never eval [I12]), k=8, temperature 1.0, budget sampled per patient and
held fixed across that patient's group *so within-group variance measures the policy and not
the draw*.

### The pass bar is a procedure, not a number

```yaml
pass_bar:
  rule: mean_total_reward_of_vitals_only_bayes_baseline_on_the_same_patients
  policy: dxenv.policy.baselines.VitalsOnlyPolicy
```

A number chosen today would either be arbitrary or reverse-engineered from a pilot run, and
the second is pre-registration in name only. The vitals-only Bayes baseline is what an agent
gets for using every free observation optimally and ordering nothing. **Clearing it means
the tests bought something.**

### The criteria

| criterion | threshold | why |
|---|---|---|
| pass@8 − pass@1 | ≥ 0.15 | RLVR sharpens behaviour the model already samples; it cannot manufacture behaviour that never appears |
| mean group reward std | ≥ 0.05 | zero spread → zero advantage → no gradient |
| fraction degenerate groups | ≤ 0.25 | a run where a third of groups are degenerate trains on two-thirds of its batch |
| calibration margin | ≥ 0.05 | the model's own distribution must Brier-score better than its argmax collapsed to one-hot |
| headroom below ceiling | ≥ 0.10 | no headroom means the task is saturated or the ceiling is wrong — the second is far more likely |
| schema valid fraction | 1.0 → **0.99** | a check that the grammar was applied, not a target to train toward |

### Measurement 1 — the base model. **FAIL**, and how it fails is the finding

Qwen2.5-7B-Instruct, 200 patients, k=8, temperature 1.0, constrained decoding.

| policy | mean R | best@8 | group std | tests |
|---|---|---|---|---|
| prior (blank-record floor) | −0.018 | −0.018 | 0.000 | 0.00 |
| random_schema | −0.385 | −0.044 | 0.322 | 0.37 |
| **prompted 7B** | **−0.689** | **−0.480** | 0.147 | 5.47 |
| vitals-only Bayes (**the bar**) | +0.675 | +0.675 | 0.000 | 0.00 |
| greedy Bayes | +1.183 | +1.183 | 0.000 | 5.76 |

`GATE B: FAIL` — pass@8 0.030 vs pass@1 0.000, a gap of **+0.030** against a required
**+0.15**. The other five criteria passed.

**The 7B is worse than guessing the base rate**, and worse than a grammar sampler with no
model behind it. Cost and turns account for only ≈ −0.335 of the −0.689, so the diagnosis
term is ≈ −0.354 against the prior's ≈ 0.00. It orders about as many tests as the greedy
baseline — **it is not shotgunning. It is confidently wrong.**

#### Why: the baselines have oracle access and the model cannot

Every patient whose best-of-8 cleared the bar:

```
chronic_kidney_disease ×2      type_2_diabetes ×2
major_depressive_disorder      streptococcal_pharyngitis
```

All urgency 2, all commoner than average, and **every one a condition where a single test is
close to definitional** — HbA1c, creatinine, rapid strep, a depression screen. Exactly the
cases where invented likelihood parameters cannot diverge much from real clinical knowledge,
because the mapping is definitional rather than statistical.

> `obs_model`'s parameters are invented, and the Bayes baselines are parameterised **from the
> very generative model that produced the data**. The LLM reasons from real clinical
> statistics that this environment does not share, so its knowledge *actively misleads*
> rather than merely failing to help.

**So this environment measures "can a policy learn this synthetic mapping", not "can it
diagnose".** That is legitimate for studying RL dynamics and it is the honest caveat on
every number here.

**The gate was not amended.** "The bar is unfair to a zero-shot model" is true, was knowable
in advance, and produces the same next step either way — SFT.

### Measurement 2 — the SFT'd policy. **FAIL under B, PASS under B2**

| metric | value | B | B2 |
|---|---|---|---|
| schema_valid_fraction | 0.997 (~10 failures / ~2,900 gens) | ✗ (needs 1.0) | ✓ (needs 0.99) |
| pass@8 | 0.330 | | |
| pass@1 | 0.080 | ✓ gap +0.25 ≥ 0.15 | ✓ |
| mean group reward std | 0.7085 | ✓ | ✓ |
| calibration margin | 1.0778 | ✓ | ✓ |
| **verdict** | | **FAIL** | **PASS** |

#### Why 1.0 was wrong *in principle*, not merely in practice

Constrained decoding guarantees any **completed** generation is schema-valid. It does not
guarantee that a generation completes. xgrammar terminates a request when the model attempts
to emit EOS mid-JSON (`grammar rejected tokens [151643]`), and a generation can exhaust its
token budget mid-string. Both yield a structurally valid **prefix**, which is not parseable
and never will be. A threshold of 1.0 therefore fails every sufficiently long run regardless
of policy quality — **it measures run length, not the property it was meant to check.**

#### The compounding error, and the one worth telling in an interview

**The metric was also incapable of failing.** Generations that did not parse were dropped
before being logged, so `schema_valid_fraction` was **1.0 by construction** and reported a
PASS on the one run that actually had failures to report.

Fixed separately (commit `1dda692`): failures are now recorded with `parsed=False`. *The
threshold question above is only live because that was fixed.* Had it not been, Gate B2
would never have been written and the amendment would have looked like moving the goalposts
rather than repairing a broken instrument.

`gate_b2.yaml` records both verdicts, and
`test_gate_b2_changes_no_substantive_threshold` enforces mechanically that the amendment
moved nothing else.

### Measurement 3 — the GRPO adapter. In progress

At last report ≈ **−0.62** mean R. Standing order: base −0.689 → SFT −0.664 → GRPO −0.620.
Monotone but marginal, and all far below the blank-record floor (−0.018).

**Known confound, recorded rather than fixed:** GRPO trained entirely under an 8-turn budget
(all 99 steps in curriculum stage 1) but Gate B evaluates at the standard 20 turns. The
policy keeps ordering tests past where it learned to stop, and every extra test only
subtracts. Training reward was −0.390 under 8 turns; the eval shows −0.620 under 20.

Not corrected, because base and SFT were both measured at 20 turns and changing it for one
arm only would break comparability. **The clean experiment is to evaluate the same adapter at
8 turns** — one variable, same patients.

---

## 10. Phase 4 — GRPO

### 10.1 Orchestration is separated from the gradient step

`GRPOTrainer` owns everything that can be wrong without a GPU — which patients are sampled,
whether the eval split was touched, how advantages are formed, when a monitor halts, what
gets persisted. The gradient arrives as an injected `Updater`. `NullUpdater` runs the whole
loop with **no model at all**.

> This is not a testing convenience. The failures this project is exposed to are leakage,
> reward hacking, and a monitor that would not have fired. **None of them live in the
> backward pass**, and all of them would otherwise be untestable without eight hours on an
> A100.

`test_ceiling_assertion_fires_on_synthetic_violation` and
`test_training_never_reads_eval_split` both run in the **fast suite** because of this split.

### 10.2 The algorithm

- **Group advantages**: standardised within the group, `(r − mean) / (std + eps)`.
- **KL** via Schulman's **k3 estimator**: `exp(r) − r − 1` where `r = logp_ref − logp_policy`.
  The naive difference `logp_policy − logp_ref` is also unbiased but **goes negative on
  individual tokens**, so the penalty occasionally *pays* the policy for leaving the
  reference. A small effect, and a very odd one to debug.
- **Clipped surrogate**, PPO-style — but **inert at one inner epoch**: `old_logp` is the
  batch's own detached logprobs, so the ratio is identically 1 and this reduces to a plain
  policy gradient. Correct single-epoch GRPO; `clip_eps` starts mattering the moment a second
  inner epoch is added.
- **Credit assignment**: one episode-level advantage broadcast uniformly across every token
  the episode generated. Stated as an assumption rather than inherited as a default — it says
  a good episode makes each of its turns slightly more likely, *including the turns
  incidental to why it was good*. The alternative (per-turn credit from a learned value head)
  reintroduces a learned model into a reward pipeline whose entire premise is that reward is
  verifiable.
- **Token-weighted gradient accumulation**, `max_grad_tokens = 6144` per micro-batch.

### 10.3 The monitors — all of them halt, none of them warn

| monitor | fires on | why it is not per-episode |
|---|---|---|
| hard ceiling [I9] | reward above a perfectly confident correct answer | it *is* per-episode — sound on every realisation |
| running expected ceiling | mean reward above the mean Bayes value | a lucky rollout may beat it; a running mean may not |
| degenerate groups | >50% of a window with zero reward spread | one flat group is an easy patient, not a collapse |
| cost distribution | collapse to zero tests **or** to the budget cap | both ends are failures, and they look nothing alike |

Cost collapse in each direction means something different:

- **to zero** — the agent learned tests never pay, which is I5 working *too well*. λ is too
  high and the frontier has collapsed to a point.
- **to the cap** — "order everything" became a survival strategy, which is what the
  curriculum exists to prevent. Appears early, while the policy is still confused and
  exhaustive testing genuinely *is* its best available move.

### 10.4 Three GPU-path bugs that would all have failed **silently**

1. **Rollout weights were never synced.** `sync_rollout_weights` existed on the protocol and
   on both implementations, and **nothing called it**. The vLLM `LoRARequest` also pinned
   adapter id 1, which vLLM caches. Rollouts would have come from the frozen SFT reference
   for the whole run while the trained adapter drifted away — no crash, just a run that
   quietly isn't GRPO.
2. **There is no separate reference model.** `get_peft_model` injects LoRA into the base
   **in place**, so holding a reference to it aliases the modules the adapter now lives in.
   The reference forward pass would have run with the trainable adapter active and **KL would
   have read 0.000 forever.** Reference logprobs come from `disable_adapter()` — which also
   means one copy of the weights rather than two.
3. **`gpu_memory_utilization` defaults to 0.55**, below vLLM's own default, because in a GRPO
   run the engine shares a device with the trainer and vLLM **preallocates its KV cache at
   startup**. Raise it for a standalone eval sweep.

### 10.5 The result — a fast correction, then a plateau

99 steps on Qwen2.5-7B + LoRA from the SFT checkpoint. 8 patients × 8 samples per step,
~8 min/step, one 12-hour job.

| | steps 0–19 | steps 20–98 | slope after step 20 |
|---|---|---|---|
| episode reward | −0.437 | −0.360 | **−0.00035 / step** |
| diagnosis score | −0.191 | −0.053 | −0.00019 / step |
| tests per episode | 2.96 | 3.57 | +0.00077 / step |
| within-group std | 0.300 | 0.145 | +0.00100 / step |
| KL from SFT reference | 0.001 | 0.044 | +0.00111 / step |

**The shape is a step change, not a trend.** Reward moves −1.1 → −0.35 and diagnosis
−1.0 → −0.05 inside the first ~20 steps, then both flatten for the remaining 78.

> Fitting a line to the whole run gives **+0.0007/step and reads as steady learning**. That
> is an artefact of the transient. **After step 20 the reward slope is negative.**

KL sits at zero until step 40 then climbs to 0.08 by step 70 — so the policy is demonstrably
*moving* in the back half and *not improving* while it does.

#### What did move: test-ordering

The clearest signal, and the one this environment was built to show. SFT left the policy at
**0.79 tests per episode**, faithfully imitating a teacher that stops early
(`min_gain = 0.15`). GRPO took it to **3.5** within twenty steps and held it there.

**No term rewards testing — I5 forbids that.** So the only route is that tests improved the
terminal score by more than they cost. *That is the cost–accuracy mechanism working, and it
is a result about the environment rather than about the policy.*

#### What this does not show

- **The policy is still poor.** −0.36 sits below the blank-record floor (−0.018), far below
  the vitals-only bar (+0.675), and 2.4 below the Bayes ceiling.
- **The curriculum never advanced.** All 99 steps in `single_condition_short`; criterion +0.70.
- **Not entropy collapse.** Within-group std settled at ~0.15, above the 0.05 floor, with
  `degenerate_fraction` at 0.0% throughout. There was always gradient available.

#### The open question

Movement without improvement, from a policy with spread to learn from and headroom to climb
into, **points at the reward signal rather than at the optimiser.** The likeliest candidate
is the one Gate B already identified: the likelihood parameters are invented, so the terminal
score rewards learning a synthetic mapping that 99 steps over ~6,300 episodes is far too
small a sample to fit across 149 conditions.

Distinguishing that from a reward-shape problem is the next experiment, **not a conclusion
this run supports.**

---

## 11. Phase 5 — evaluation and the audit suite

| Probe | Pass condition |
|---|---|
| Blank-record baseline | agent with empty observation ≈ prior; all results reported above this floor |
| Leakage ablation | strip conditions/meds/careplans/reasonCodes; accuracy barely moves |
| No-test ablation | zero-test accuracy meaningfully worse than with-test |
| Shuffled labels | reward drops to chance |
| Counterfactual perturbation | flip a lab normal→abnormal; posterior moves in the clinically correct direction |
| Bayes ceiling | agent ≤ ceiling |
| Held-out modules | split by generator module; report the generalisation gap |

All seven pass. **Every probe that could pass trivially carries a positive control.**

### Both interesting failures were the *probe*, not the environment

- **Counterfactual perturbation** ranked candidate conditions by the **mean** of an analyte,
  but Bayes moves mass by **likelihood at the observed value**. A condition with mean 1200 and
  sd 900 has low density at 42, so the probe had the direction wrong and reported 61%.
  Corrected: 200/200 — and it now also carries six named clinical spot checks, because the
  generic property is guaranteed by Bayes and would pass even if every override in
  `obs_overrides.yaml` were attached to the wrong condition.
- **Blank-record baseline** read −0.28 against an analytic floor of +0.001, because the
  baseline policy truncated its report to a top-8 — which costs an **uninformed** policy far
  more than an informed one and depresses the floor everything else is measured against.
  Hence `TOP_K_REPORT = None`.

> An audit suite that would not catch a real failure is worse than none, because it
> manufactures confidence.

### Persistence

`runs/{run_id}/episodes.jsonl`, one JSON line per episode, under pinned config hashes.

Ground truth lives on the line so a run is self-contained. That has one sharp edge — **an
`episodes.jsonl` file is not safe to feed to a model** — and the rule is *structural rather
than advisory*: `stored_trajectory()` returns the trajectory alone and is the only accessor
the rollout and training paths use.

The store's config guard caught something real on the first loop run: a curriculum stage
changes `max_turns`, which changes the episode config hash, so one run legitimately writes
lines under several hashes. **The fix was to declare every stage's hash at run start** rather
than loosen the check to a warning — an undeclared hash still fails, because it means an
episode was generated under a configuration nobody intended.

---

## 12. Infrastructure

Virginia Tech ARC, SLURM, allocation `ece-6524-spring2026`.

| script | partition | QoS | wall |
|---|---|---|---|
| `00_check_gpu` | `a100_normal_q` | `tc_a100_normal_short` | short |
| `01_fetch_model` | `normal_q` (CPU) | — | — |
| `02_gate_b` | `a100_normal_q` | `tc_a100_normal_short` | 24h |
| `03_sft` | `a100_normal_q` | `tc_a100_normal_short` | — |
| `04_grpo` | `h200_normal_q` | `tc_h200_normal_short` | 24h |

### Three things a scheduler changes

1. **GPUs are allocated, not present.** Nothing on a login node sees a GPU, so the install is
   split: CPU install and the whole test suite run on the login node.
2. **Jobs have a wall clock.** A long GRPO run is a *chain* of jobs. `--resume` restores the
   step index, curriculum stage, RNG **and monitor windows**. Without the last one, every job
   in the chain refills the detector windows from empty, which leaves the ceiling and collapse
   monitors **off** for the first stretch of each job.
3. **Home directories have quotas.** A 7B checkpoint is ~15GB; `HF_HOME` points at scratch.

### The two-venv split — do not try to unify it

vLLM pins transformers ~4.51; TRL needs ≥4.56. Crossing them fails **at import, ~20 minutes
into a job holding a full node**:

```
ImportError: cannot import name 'is_trackio_available' from 'transformers'
RuntimeError: Failed to import trl.trainer.sft_trainer
```

So: `.venv` (`[infer]`) for rollouts, Gate B and GRPO; `.venv-train` (`[train]`) for SFT
only. It works because the two paths need different things — rollouts and GRPO need vLLM plus
raw torch/peft for the gradient step, and only SFT needs TRL's `SFTTrainer`.

Both extras import **lazily**, so the entire invariant suite runs on a laptop with neither
installed.

### Cluster facts, each paid for with a wasted allocation

- **`--qos` is the biggest scheduling lever.** Adding it moved a sibling job from priority
  1330 to 2312. "short" caps at a full day and carries the highest priority.
- **Never `--mem=0`** on a small job — it means *all* node memory, satisfiable only on a
  wholly idle node, so it can never backfill.
- **`/projects` is per-allocation**, not per-user, and not writable by you.
- **Assert the interpreter.** `source activate` can report success without switching
  interpreters, and a zero-byte python exits 0 printing nothing — a GPU job then "succeeds"
  in two seconds with an empty log. `env.sh` checks `sys.prefix` before any work happens.
- **Submit from the repo root.** `#SBATCH` directives are parsed before any shell runs, so
  `--output` cannot contain a variable and is relative to the **submission** directory.

### vLLM API detection

vLLM renamed `guided_decoding=GuidedDecodingParams(...)` to
`structured_outputs=StructuredOutputsParams(...)`. `_structured_output_kwargs()` **detects
the API by signature** rather than pinning a name that has already moved once, and falls
back through a config ladder `[{"backend": "xgrammar"}, {}]` instead of losing a job to each
mismatch.

### Notifications

Jobs report to Telegram on start, success (with a log tail) and failure (with a longer tail).
Gate B sends its verdict; the GRPO chain reports step and curriculum stage on each requeue.

Two deliberate choices:

- The token lives in `~/.config/dxenv/telegram.env`, **outside the repo**, because this repo
  is public and a committed bot token is a live credential rather than a config value.
  `.gitignore` refuses `*.env` as a second line of defence.
- **A failed send is silent by design** everywhere inside a job. A notifier that can turn a
  successful twenty-hour run into a failed one because an HTTPS call timed out is worse than
  no notifier. `--require` makes it loud, for setup only.

The wall-clock case is the one worth having either way: SLURM sends SIGTERM before killing a
job, the trap catches it, and you get told. **Without it a run that hits its time limit
simply vanishes with no error message at all.**

---

## 13. Every caveat

Ordered by how much they matter. Volunteer the first three; do not wait to be asked.

### 13.1 The likelihood parameters are invented

`obs_model`'s `p(result | condition)` numbers are consistent and leak-free but **not drawn
from published likelihood ratios**. Consequence: the Bayes baselines are parameterised from
the very generative model that produced the data, so they have oracle access no LLM can have.

**This environment measures "can a policy learn this synthetic mapping", not "can it
diagnose".** Legitimate for studying RL dynamics; fatal to any clinical claim.

### 13.2 The final policy is below the no-information baseline

−0.36 (training) / −0.62 (Gate B) against a blank-record floor of −0.018 and a vitals-only
bar of +0.675. The direction of travel is right and every arm is monotone, but **no arm has
cleared the floor.**

### 13.3 The GRPO run is small

99 steps × 8 patients × 8 samples ≈ 6,300 episodes, across 149 conditions. That is a very
small sample for the mapping being fitted, and it is the most likely explanation for the
plateau.

### 13.4 Train/eval turn-budget mismatch

GRPO trained entirely at `max_turns = 8`; Gate B evaluates at 20. Recorded, not fixed,
because fixing it for one arm breaks comparability with base and SFT. The clean experiment
is a same-adapter 8-turn evaluation.

### 13.5 The curriculum never advanced

All 99 steps in `single_condition_short` (criterion +0.70, never approached). So stages 2 and
3 — full horizon, and comorbid — are **untested in practice**.

### 13.6 Comorbidity is unimplemented

The curriculum declares a `comorbid` stage; the generator emits **one condition per patient**.

### 13.7 `data/snomed_map.yaml` is empty

Real Synthea output cannot be ingested yet. Everything runs on the internal generator.

### 13.8 Shaping is built, tested, and disabled

See §6.4. Not a gap so much as an unresolved design tension, and the open question most worth
attention.

### 13.9 `optimal_stopping_value` is bounded, not exact

Full-information Bayes value, direction proved in the docstring. Sound for I9 (it must be an
*upper* bound), but looser than an exact DP would be.

### 13.10 The clipped surrogate is currently inert

One inner epoch → ratio identically 1 → plain policy gradient. Correct, but `clip_eps` is
doing nothing today.

### 13.11 The near-miss cost matrix does not ship

Deferred. If added it must be an explicit cost matrix over condition pairs **keyed on
consequence of the error**, never on semantic similarity of names.

### 13.12 Gate B measurement 3 is incomplete

The GRPO arm was still running at the time of writing. Numbers quoted for it are partial.

---

## 14. War stories

The debugging record. Each one is a transferable lesson, which is what makes them interview
material rather than trivia.

### "GPU stack OK" printed while everything failed

`set -e` lived in `env.sh`, so when `env.sh` itself failed to load, the script **lost its
error handling** and every subsequent failure was ignored. The job reported success in
seconds with an empty log.

*Fix:* `set -euo pipefail` as the **first line of each sbatch**, plus a bootstrap that walks
up from `$SLURM_SUBMIT_DIR` looking for `pyproject.toml`.
*Lesson:* error handling that lives in the thing that can fail is not error handling.

### Truncation misdiagnosed as a token budget

Generations were truncating mid-JSON. I raised `max_tokens` 512 → 1438. It failed again at
2876.

Real cause: **`maxLength` is advisory** — grammar backends do not enforce string length. The
model was writing a 3,000-character `reasoning` field because nothing stopped it.

*Fix:* a compiled `pattern`, which **is** compiled into the grammar, making the bound
structural.
*Lesson:* two failures at increasing budget is evidence the budget is not the variable.

### The SFT key-order mismatch — the costliest bug

`render_wire` used `sort_keys=True`, teaching the model alphabetical key order, while the
decoder forces **schema declaration order**. The SFT'd 7B came out rambling in Chinese and
Norwegian and scored *worse than base*.

Compounding it: I then advised testing the old adapter before retraining. **That was wrong —
the mismatch was baked into the weights**, so the test could only reconfirm the failure. It
cost ~70 minutes of GPU time.

*Fix:* `WIRE_KEY_ORDER` pins serialisation to schema order, and `03_sft.sbatch` now
**validates key order before reusing an SFT set** (commit `7a9b96f`: "Refuse to reuse an SFT
set whose targets predate the current grammar").
*Lesson:* train/inference format mismatches present as *capability* failures, not format
failures. And when the bug is in the weights, there is nothing to test — retrain.

### A gate criterion that could not fail

`schema_valid_fraction` was **1.0 by construction**: failed generations were dropped before
being logged. It reported PASS on the one run that had failures to report.

*Fix:* record failures with `parsed=False` (commit `1dda692`), *then* amend the threshold.
*Lesson:* **test the detector, not just the thing it detects.** And the order matters — fixing
the metric first is what made the amendment a repair rather than a goalpost move.

### Three OOMs, and two wrong guesses

1. Whole-batch forward → micro-batch it.
2. `logits.float().log_softmax()` over 4000 × 152064 → slice to completion positions and use
   `cross_entropy` directly.
3. Contention between vLLM's preallocated KV cache and the trainer → two GPUs (H200s).

I guessed wrong twice about which term dominated before adding memory diagnostics. The
fp32-weights hypothesis was disproved by measurement: 14.19 GiB, exactly bf16 as intended.

*Lesson:* instrument before hypothesising. The first successful run came within **20 MiB**.

### `trainer_state.json` said step 9 while 99 steps had run

`save_state` was in a `finally` block, which **SIGTERM bypasses**. I initially read this as
"180 min/step" and was wrong by 30×.

*Fix:* checkpoint on SIGTERM **and every step** (commit `ab7da55`: "a 12-hour run nearly lost
89 steps").
*Lesson:* a `finally` is not a signal handler.

### Misreading the GRPO curve from summary statistics

I reported "steady +0.00071/step" from a whole-run linear fit. The plot showed a **step change
then a plateau**, with the slope after step 20 actually *negative*. Self-corrected once the
figure was rendered.

*Lesson:* fit a line to a transient and you will report a trend that does not exist. Plot it.

### Sequential rollouts could not have finished

The original design generated episodes one at a time: ~356 GPU-hours for a run that had to fit
in 12.

*Fix:* `rollout_lockstep()` steps many episodes per backend call, with **per-conversation
seeds** so determinism [I10] survives batching. Plus an alignment guard that checks
`CASE {patient_ref}` appears in the returned prompt, because a batched backend that silently
reorders outputs would corrupt every episode without erroring.

### Two substring bugs of the same shape

`leak_strings` contains **"mi"** and **"all"**, and substring matching fired them inside
"com**mi**t" and "at **all**" — rejecting three-quarters of a clean SFT set.

*Fix:* word-boundary matching, and *mention* detection uses the taxonomy's **narrow** forms
while the observation scrubber keeps the **wide** ones. The asymmetry runs opposite ways in the
two places, and the taxonomy already encoded that distinction.

### Committed lint errors twice

My `&&` chain didn't extend to the commit line, so verification passed and the commit went
through regardless.

*Fix:* chain verification and commit together in one expression.

---

## 15. Numbers worth memorising

| | |
|---|---|
| Labels / menu actions / orderable tests | 149 / 147 / 73 |
| Tests in suite | 326 fast + 5 slow |
| λ (cost) / μ (turn) | 0.004 / 0.02 |
| Severity tiers | 1.0 / 1.8 / 3.2 / 6.0 |
| Budget support | 10, 25, 50, 100, 200 |
| Max turns (full / stage 1) | 20 / 8 |
| Report width | 16 labels |
| Reasoning cap | 700 chars, as a `pattern` |
| Split sizes | 14,854 / 3,713 / 1,433 |
| **Gate A** V−F / T−V | **+0.267** (≥0.08) / **+0.436** (≥0.20) |
| Gate A leak positive control | +0.219 (≥0.10) |
| Blank-record floor | **−0.018** |
| Vitals-only Bayes bar | **+0.675** |
| Greedy Bayes | +1.183 |
| Expected ceiling | 1.722 |
| Base 7B mean R | **−0.689** (pass@8−pass@1 = +0.030 vs +0.15 required) |
| SFT pass@8 / pass@1 | 0.330 / 0.080 → gap **+0.25** ✓ |
| SFT schema valid | 0.997 → FAIL under B (1.0), PASS under B2 (0.99) |
| GRPO steps / episodes | 99 / ~6,300 |
| GRPO tests per episode | **0.79 (SFT) → 3.5 (GRPO)** |
| GRPO reward slope after step 20 | **−0.00035 / step** |
| De-leak ablation gap | +0.732 privileged → **+0.036** de-leaked |
| Wrong ablation null would report | +0.63 |

---

## 16. Likely interview questions

**"Why Brier and not log-loss?"**
Log-loss is unbounded below. GRPO standardises advantages within the group, so one rollout
putting ≈0 on the truth produces an outlier that flattens every other sample's advantage to
≈0 — the batch teaches nothing. Brier is bounded [I11]. Both are proper; only one is safe
here.

**"How do you know the environment isn't leaking?"**
Four layers. Structural: the observation type has no field that could hold a label. Allowlist
that fails closed. A corpus-wide test that no observation string contains the patient's
condition or synonyms. And the Bayes ceiling as a runtime detector — an agent scoring above
what perfect reasoning could achieve on the same evidence has information it should not have,
and the run halts. Plus every ablation carries a **positive control**, because a leak check
that passes trivially proves nothing.

**"What's the hardest bug you hit?"**
The SFT key-order mismatch. Targets were rendered with `sort_keys=True` while the decoder
forces schema declaration order, so the model was trained on a format it could never emit at
inference. It presented as a *capability* failure — the 7B rambling in Chinese — not a format
one, which is why it took a while. Two lessons: train/inference format mismatches don't look
like format bugs, and when the defect is in the weights there's nothing to test, only to
retrain. I got that second part wrong first and burned ~70 minutes of GPU on it.

**"Did it work?"**
Partly, and the honest answer is more interesting than a yes. GRPO recovered test-ordering
from 0.79 to 3.5 per episode with **no reward term for testing** — tests only ever subtract —
which is the environment's cost–accuracy mechanism doing exactly what it was built to do. But
the final policy is still below the no-information baseline, and the reward curve is a fast
correction followed by a plateau with KL still rising. Movement without improvement, from a
policy that has both spread and headroom, points at the reward signal rather than the
optimiser. The likeliest cause is that the likelihood parameters are invented, so the task is
learning a synthetic mapping — and 99 steps over ~6,300 episodes is far too small for 149
conditions.

**"Why is the action menu global?"**
If the menu were derived from the patient's own record, **the menu is the diagnosis**. You'd
be handing the agent a shortlist and measuring how well it picks from it. It's the single
easiest way to build an environment that produces beautiful numbers and measures nothing.

**"Why doesn't `env.step()` return a reward?"**
Because the reward is a pure function of the stored trajectory [I8], keeping the environment
and the reward engine apart makes **offline rescoring free**. You will change reward weights
repeatedly; regenerating 7B rollouts to do it is the dominant cost of the project. Verified on
a 6,600-episode store — rescoring reproduces stored totals exactly.

**"What would you do next?"**
Three things, in order. First, the 8-turn evaluation of the current adapter, to settle the
turn-budget confound with one variable. Second, ground the likelihood parameters in published
likelihood ratios for at least the diagnostically important pairs — that's the caveat sitting
underneath every other number. Third, decide the shaping question deliberately rather than
inheriting the default; the machinery is built and tested and currently switched off.

**"What's the weakest part of the work?"**
The likelihood parameters, and I'd say so before being asked. Everything downstream inherits
it: the Bayes baselines have oracle access to the generative model, the LLM's real clinical
knowledge actively misleads rather than merely failing to help, and no zero-shot model was
ever going to clear a bar set that way. That was knowable in advance and I recorded it as the
reading of Gate B rather than discovering it twice.
