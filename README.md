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
| **3 — cold start** | Not started (constrained decoding, teacher, rejection sampling, SFT) |
| **4 — GRPO** | Monitors and curriculum scaffolding only; no training loop |
| **5 — evaluation** | Audit suite, Pareto sweep, calibration complete |

188 tests (183 fast + 5 slow). `ruff` and `mypy --strict` clean.

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

## Layout

```
dxenv/
  data/      taxonomy (149 flat labels) · corpus (Synthea parser + generator) · splits
  env/       filter [I1,I2] · actions [I3] · obs_model [I4] · bayes [I9] · episode · schemas
  reward/    scoring [I7] · costs [I5] · treatment · verify · shaping [I6] · engine [I8]
  policy/    baselines (prior, vitals-only, greedy Bayes)
  train/     monitors (ceiling assertion, advantage variance) · curriculum
  eval/      audit · pareto · calibration
  configs/   gate_a · gate_a2 · severity · reward · costs · env · treatments
tests/       invariants (one file per I1–I12) · unit · property · golden
scripts/     phase0_feasibility · check_gate_a · regenerate_golden
```

Module rules that are enforced by tests, not convention: `reward/` never imports `policy/`
or `train/`; `env/` never imports `reward/` (which is what makes offline rescoring of a
stored trajectory corpus free). `env.step()` deliberately returns **no reward** — the
environment produces trajectories, the reward engine scores them.

---

## Known gaps

- **`data/snomed_map.yaml` is empty.** The generator drives everything today, so nothing
  depends on it; ingesting real Synthea output requires populating it first, and
  `map_snomed` raises on any unmapped code rather than silently dropping the condition.
- **`policy/` has only heuristic baselines.** No constrained decoding, teacher,
  rejection sampling, or SFT (Phase 3), and no GRPO loop (Phase 4) — monitors and
  curriculum scaffolding exist and are tested, but nothing trains yet.
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
[10, 25, 50, 100, 200]. Still open: whether the near-miss cost matrix ships in v1, whether
Phase 3 SFT is needed at all, and **whether to enable shaping** (see above).
