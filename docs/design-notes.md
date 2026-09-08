# Design notes

The engineering record: what was measured, what was decided, and the places where the
spec in `CLAUDE.md` turned out not to survive contact. Kept out of the README so that
stays an overview, kept in the repo because most of it is the reasoning behind a number
that would otherwise look arbitrary.

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

## Gate B — the prompted baseline FAILS, and how it fails is the finding

Qwen2.5-7B-Instruct, 200 patients, k=8, temperature 1.0, constrained decoding.

| policy | mean R | best@8 | group std | tests |
|---|---|---|---|---|
| prior (blank-record floor) | −0.018 | −0.018 | 0.000 | 0.00 |
| random_schema | −0.385 | −0.044 | 0.322 | 0.37 |
| **prompted 7B** | **−0.689** | **−0.480** | 0.147 | 5.47 |
| vitals-only Bayes (the bar) | +0.675 | +0.675 | 0.000 | 0.00 |
| greedy Bayes | +1.183 | +1.183 | 0.000 | 5.76 |

`GATE B: FAIL` — pass@8 0.030 vs pass@1 0.000, a gap of +0.030 against a required +0.15.
The other five criteria pass: schema validity 1.0000 over ~8,700 generations, group std
0.147 with 15.5% degenerate groups, calibration margin +1.0045, headroom 2.41.

**The 7B is worse than guessing the base rate**, and worse than a grammar sampler with no
model behind it. Cost and turns account for only ~−0.335 of the −0.689 (5.47 tests × λ ×
median price, plus μ·turns), so the diagnosis term is ≈ −0.354 against the prior's ≈ 0.00.
It orders about as many tests as the greedy baseline; it is not shotgunning. It is
confidently wrong — and the calibration margin says so from the other direction, since
collapsing its report onto its argmax would cost a full point.

### Why: the baselines have oracle access and the model cannot

Every patient whose best-of-8 cleared the bar:

    chronic_kidney_disease  ×2      type_2_diabetes  ×2
    major_depressive_disorder       streptococcal_pharyngitis

All urgency 2, all commoner than average (prior weight 2.83 vs 2.12), and every one a
condition where a single test is close to DEFINITIONAL — HbA1c, creatinine, rapid strep, a
depression screen. Those are exactly the cases where invented likelihood parameters cannot
diverge much from real clinical knowledge, because the mapping is definitional rather than
statistical.

That is the whole result. `obs_model`'s parameters are invented (see Known gaps), and the
Bayes baselines are parameterised from the very generative model that produced the data.
The LLM reasons from real clinical statistics that this environment does not share, so its
knowledge actively misleads rather than merely failing to help. Gate A established the
signal is learnable — but by a classifier TRAINED on this data.

**So this environment measures "can a policy learn this synthetic mapping", not "can it
diagnose".** That is legitimate for studying RL dynamics and it is the honest caveat on
every number here. It also means no zero-shot model was ever going to clear a bar set by
an oracle-parameterised baseline, which is worth stating plainly rather than discovering
twice.

### What it says about the next step

First sample beat the blank-record floor on 1 patient of 200; best-of-8 on 17. Thin, but
not nothing — and `corr(group_std, best) = +0.435`, with the 31 zero-spread patients
scoring worse, so the diversity RLVR needs tracks the outcomes it needs.

The gate's pre-registered action is SFT, and it is the right instrument for this specific
failure: the teacher HAS the true observation model, so it can teach the mapping the base
model has no way to know. The gate was not amended — "the bar is unfair to a zero-shot
model" is true, was knowable in advance, and produces the same next step either way.

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

## Phase 4 result — a fast correction, then a plateau

99 GRPO steps on Qwen2.5-7B + LoRA, starting from the SFT checkpoint. 8 patients x 8
samples per step, ~8 min/step, one 12-hour job. Curve: `runs/grpo/curve.png`, regenerated
from committed data by `scripts/plot_grpo.py`.

|  | steps 0-19 | steps 20-98 | slope after step 20 |
|---|---|---|---|
| episode reward | −0.437 | −0.360 | **−0.00035 / step** |
| diagnosis score | −0.191 | −0.053 | −0.00019 / step |
| tests per episode | 2.96 | 3.57 | +0.00077 / step |
| within-group std | 0.300 | 0.145 | +0.00100 / step |
| KL from SFT reference | 0.001 | 0.044 | +0.00111 / step |

**The shape is a step change, not a trend.** Reward moves −1.1 → −0.35 and diagnosis
−1.0 → −0.05 inside the first ~20 steps, then both flatten for the remaining 78. Fitting a
line to the whole run gives +0.0007/step and reads as steady learning; that is an artefact
of the transient. After step 20 the reward slope is NEGATIVE.

Meanwhile KL sits at zero until step 40 and then climbs to 0.08 by step 70. So the policy
is demonstrably moving in the back half of the run and not improving while it does. The
early gain is most plausibly GRPO correcting the SFT policy's degenerate habits -- chiefly
that it had stopped ordering tests -- rather than learning to diagnose.

### What did move: test-ordering

The clearest signal, and the one this environment was built to show. SFT left the policy at
**0.79 tests per episode**, faithfully imitating a teacher that stops early (`min_gain=0.15`
in `PrivilegedTeacher`). GRPO took it to **3.5** within twenty steps and held it there. No
term rewards testing -- I5 forbids that -- so the only route is that tests improved the
terminal score by more than they cost. That is the cost-accuracy mechanism working, and it
is a result about the ENVIRONMENT rather than about the policy.

### What this does not show

- **The policy is still poor.** −0.36 sits below the blank-record floor (−0.018), far below
  the vitals-only Bayes bar (+0.675), and 2.4 below the Bayes ceiling.
- **The curriculum never advanced.** All 99 steps in `single_condition_short`; the criterion
  is +0.70.
- **Within-group std settled at ~0.15**, above the 0.05 floor and with `degenerate_fraction`
  at 0.0% throughout, so there was always gradient available -- the plateau is not entropy
  collapse.

### The open question

Movement without improvement, from a policy with spread to learn from and headroom to
climb into, points at the reward signal rather than at the optimiser. The likeliest
candidate is the one Gate B already identified: the environment's likelihood parameters are
invented, so the terminal score rewards learning a synthetic mapping that 99 steps over
~6,300 episodes is simply too small a sample to fit across 149 conditions. Distinguishing
that from a reward-shape problem is the next experiment, not a conclusion this run
supports.

## Gate B on the GRPO adapter — the mean rose and the tail collapsed

Same instrument, same 200 patients, same bar. **FAIL under both gate_b and gate_b2**, on
pass@k and on schema validity.

| | base 7B | SFT | GRPO |
|---|---|---|---|
| mean R | −0.689 | −0.664 | **−0.569** |
| best@8 | −0.480 | — | **+0.034** |
| pass@1 | 0.000 | 0.080 | 0.010 |
| pass@8 | 0.030 | **0.330** | 0.105 |
| pass@8 − pass@1 | +0.030 | **+0.250** | +0.095 |
| within-group std | 0.147 | 0.709 | 0.371 |
| degenerate groups | 15.5% | — | 0.5% |
| calibration margin | +1.0045 | +1.0778 | +0.8489 |
| schema valid | 1.000 | 0.997 | **0.985** |
| tests / episode | 5.47 | — | 4.69 |

Two things moved in opposite directions, and the gate only sees one of them.

**The mean improved.** −0.664 → −0.569, and best-of-8 cleared the blank-record floor
(+0.034 against −0.018) for the first time in any arm. GRPO also fixed the SFT policy's
degenerate group rate: 0.5%, against 15.5% for the base model.

**The tail collapsed.** Within-group spread halved, 0.709 → 0.371, and with it the pass
rates: pass@8 0.330 → 0.105, pass@1 0.080 → 0.010. The pass bar (+0.675) sits far above the
mean (−0.569), so *clearing it is a tail event*. A policy that becomes more consistent
loses tail events even as its centre improves, and the gate's headline criterion is a tail
statistic.

That is the familiar RLVR trade -- RL sharpens the policy toward its own mean and spends
the diversity it was given -- with one wrinkle worth naming: **pass@1 fell too**. The usual
story is pass@1 up, pass@k down. Here both fell while the mean rose, which says the
compression was symmetric rather than a sharpening onto the good samples.

The consequence is practical rather than cosmetic. pass@k is the exploration budget the
*next* round of GRPO would have to sharpen, and this run spent two thirds of it to buy
+0.095 of mean reward. Continuing from this checkpoint has less to work with than starting
again from SFT would.

### The verdict had to be fixed before it could be read

The first run of the checker reported a different and meaningless FAIL. The subject chain
was `sft -> prompted -> random_schema` with no `grpo` entry, so a GRPO results file fell
through to the grammar sampler: five of six criteria were computed against a policy with no
model behind it, printed under a heading naming the adapter, and a verdict was issued
either way. Only `calibration_margin` and `schema_valid_fraction` read the subject, because
those are top-level fields -- so the output was a mix of two policies.

The tell was arithmetic: the reported headroom of 2.1067 is `1.7220 − (−0.3847)`, and
−0.3847 is `random_schema`'s mean, not the adapter's.

Fixed by honouring the `subject_policy` the results file already declares, and by refusing
rather than falling back when the declared row is absent. `--gate` was added at the same
time so the B2 amendment can be evaluated without editing the script, and an amendment now
resolves against the gate named in its `amends:` field rather than against its own
`unchanged_from_gate_b` list -- which omits `degenerate_group_std` and would have dropped
that criterion silently. `unchanged_from_gate_b` is now checked as the redundant assertion
it is: a value disagreeing with the amended gate refuses to run.

### Schema validity regressed, and the artefacts could not say why

0.997 → 0.9849, a fivefold rise in unparseable generations, and below even the amended 0.99
floor. `gate_b2.yaml`'s pre-registered reading of that is "a SYSTEMATIC decoding problem,
not the tail of long-horizon sampling".

The leading hypothesis is KL drift: the adapter moved away from the SFT reference over the
run (KL 0.001 → 0.044), and a drifted policy writes longer reasoning that runs into the
700-character `pattern` bound mid-string, yielding a structurally valid prefix that is not
parseable. The GRPO arm also evaluates at 20 turns rather than 8, so it emits more
generations per episode with more opportunity to hit it.

**Neither could be checked**, because the sweep dropped `generations` before persisting --
correctly, since they are enormous, but it left the metric detectable and not diagnosable.
Failed completions are now kept (capped at 40 per row) with their `finish_reason`, which is
what distinguishes a grammar-terminated request from one that ran out of tokens. The
diagnosis waits on the next sweep.

### What the gate does and does not say here

Gate B is the Phase 3 go/no-go: *may we start GRPO?* That question was answered by the SFT
row, which passed under gate_b2. Running the same instrument on the Phase 4 output is a
**diagnostic, not a gate decision** -- the pre-registered `on_failure` action for pass@k
("do NOT proceed to GRPO") is addressed to a decision already taken on different evidence.

It is reported because it is informative, and because reporting only the arm where the
instrument was designed to be used would be selective. The honest summary is that GRPO
improved the mean, cleared the floor on best-of-8, and paid for it in the diversity a
subsequent RL round would need.

---

### The turn budget differs between training and evaluation

Recorded before the Gate B numbers landed, because it changes how they read.

GRPO trained entirely inside curriculum stage `single_condition_short`, which sets
`max_turns = 8`. The stage never advanced -- its criterion is +0.70 and the run plateaued
near -0.36 -- so all 99 steps ran under an 8-turn budget. Gate B evaluates under the
standard environment, `max_turns = 20`, and the eval log shows episodes reaching turns 16,
17 and 18.

The policy therefore keeps ordering tests past the point it was ever trained to stop, and
under I5 every one of those only subtracts. Training reward was -0.390; the Gate B arm
reads -0.620 on the same adapter.

**Not corrected**, and the reason is comparability: base and SFT were both measured at 20
turns, so changing the horizon for the GRPO arm alone would make the three numbers
incommensurable and flatter the one arm that got the change. The mismatch is reported as a
limitation instead.

It is a hypothesis rather than a measured effect. The two numbers differ in more than the
turn budget -- different patients, and the training figure is a running mean over a policy
that was still changing. The clean experiment is the same adapter evaluated at
`max_turns = 8` against the same patients: one variable, ~10 GPU-hours, and the script does
not currently expose the flag.

If it holds it is a result about the environment rather than a defect: a policy that cannot
recognise when evidence has stopped paying is actively punished for being given more
budget, which is the cost-accuracy mechanism seen from the unflattering direction.

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

## Open decisions (CLAUDE.md §12)

See the README for the current list of known gaps.

Resolved here, and flagged as reversible: label set = **149**; `optimal_stopping_value` is
**bounded, not exact** (full-information Bayes value, direction proved in the docstring);
severity = **4 tiers at 1.0 / 1.8 / 3.2 / 6.0**; `p(B)` = **discrete mixture** over
[10, 25, 50, 100, 200]; report width = **16 labels**, sized from the tail-mass measurement
above. Still open: whether the near-miss cost matrix ships in v1 and **whether to enable
shaping**. Whether Phase 3 SFT is needed is now answerable rather than open — run 8.1 with
`--model` and read Gate B.
