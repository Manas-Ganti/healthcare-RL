# CLAUDE.md — Diagnostic RLVR Environment

Context for any agent writing code in this repo. Read this fully before touching a file.

---

## 1. What this project is

A multi-turn RL environment where an LLM agent plays a diagnostician over synthetic patient
records from [Synthea](https://github.com/synthetichealth/synthea). Each episode:

1. Agent receives a filtered patient observation (demographics, vitals, complaint).
2. Agent takes turns: order a test, prescribe a treatment, or terminate by declaring a
   probability distribution over conditions (or abstaining).
3. Episode is scored once at termination against hidden ground truth.

The research contribution is **the environment**, not the policy. Specifically: a diagnostic
RLVR environment with (a) verifiable per-step rewards, (b) a computable Bayes-optimal ceiling
that doubles as a reward-hacking detector, and (c) a budget-conditioned cost–accuracy frontier.

**This is a synthetic research environment. It is not a clinical decision tool, is not
validated against real patients, and no artifact from this repo may be presented as clinical
guidance.** Severity weights and contraindication rules are simulation parameters chosen for
RL dynamics, not clinical recommendations.

### Why the environment is hard to build correctly

Synthea generates records from explicit rule-based disease modules. The diagnosis is therefore
present in the record in roughly six places, and the *sparsity pattern* of the record leaks it
in a seventh. Most of the engineering below exists to close those channels. A leaky environment
produces excellent-looking numbers that mean nothing, and the failure is silent.

---

## 2. Non-negotiable invariants

These are correctness properties, not style preferences. Every one has a test in
`tests/invariants/`. **Do not weaken a test to make code pass — fix the code, or raise the
conflict.** If a change appears to require violating one of these, stop and ask.

| # | Invariant |
|---|---|
| **I1** | Ground truth never enters an observation. The observation builder returns a dict that structurally cannot contain the label. |
| **I2** | The observation is built by **allowlist**. Unknown fields are dropped by default; an unrecognised resource type raises in strict mode rather than passing through. |
| **I3** | The action menu is **global and identical for every patient**. It is never derived from the patient's own record. |
| **I4** | Every orderable test returns a value for every patient. There is no "unavailable", no `None`, no default-to-normal. |
| **I5** | Ordering a test never produces positive reward, at any step, under any shaping term. Tests only ever subtract. |
| **I6** | Any shaping term must be potential-based: `F(s,a,s') = γΦ(s') − Φ(s)`. No other per-step bonus exists. |
| **I7** | Terminal diagnosis scoring uses a strictly proper scoring rule (Brier) over a fixed flat label set. No free-text scoring, no LLM judge, no hierarchy. |
| **I8** | Reward is a pure function of `(trajectory, ground_truth, reward_config)`. No RNG, no clock, no I/O, no global state. |
| **I9** | Episode reward never exceeds the Bayes-optimal ceiling for that patient. Violation = leak; the training loop asserts on this. |
| **I10** | Episodes are deterministic given `(patient_id, seed, config_hash)`. |
| **I11** | Reward is finite and bounded. `NaN`/`inf` is a hard failure, never clipped away silently. |
| **I12** | The eval split is frozen and hash-verified before any training run. Training never reads it. |

**I5 is the one that gets violated by accident.** Every reward-hacking story in this environment
starts with someone adding a plausible-looking "reward informative tests" term. Information gain
bonuses are prohibited: the agent will find tests that maximise entropy reduction under the
belief model without improving the answer. A test pays for itself only by improving the terminal
score enough to cover its cost.

---

## 3. Repo layout

```
dxenv/
  data/
    corpus.py              # Synthea generation + parsing to internal records
    taxonomy.py            # Fixed flat condition label set, SNOMED -> label mapping
    splits.py              # Frozen train/eval/held-out-module splits + hashing
  env/
    filter.py              # Allowlist observation builder            [I1, I2]
    actions.py             # Global action space definition           [I3]
    obs_model.py           # Generative p(result | condition)         [I4]
    bayes.py               # Posterior, optimal stopping, ceiling     [I9]
    episode.py             # Gym-style env, turn loop, budget accounting
    schemas.py             # Pydantic models for actions + observations
  reward/
    scoring.py             # Brier, severity weights                  [I7]
    costs.py               # Test costs, turn penalty, dedup
    treatment.py           # Treatment appropriateness + contraindications
    shaping.py             # Potential-based shaping                  [I6]
    verify.py              # Predict-then-verify per-test scoring
    engine.py              # Pure composition of the above            [I8]
  policy/
    decoding.py            # Constrained decoding grammars
    teacher.py             # Privileged trajectory generation + de-leaking
    rejection.py           # Rejection sampling filters
    sft.py                 # SFT with Bayes-posterior soft labels
  train/
    grpo.py                # GRPO loop
    curriculum.py          # Stage scheduling
    monitors.py            # Entropy, ceiling assertion, KL, cost dist
  eval/
    audit.py               # The audit suite (Section 9)
    pareto.py              # Budget sweep -> frontier
    calibration.py         # Reliability diagrams, ECE
  configs/                 # YAML; every run pins a config hash
tests/
  invariants/              # One file per invariant I1-I12
  unit/                    # Mirrors package layout
  property/                # Hypothesis-based
  golden/                  # Frozen fixtures + expected outputs
```

### Module rules

- Every module above is independently importable and independently testable. No circular imports.
- `reward/` must not import from `policy/` or `train/`.
- `env/` must not import from `reward/`. The environment produces trajectories; the reward
  engine scores them. Keeping these apart is what makes offline rescoring possible.
- Anything that reads config takes it as an argument. No module-level config reads.

---

## 4. Conventions

- Python 3.11+, `uv` for deps, full type hints, `pydantic` for all boundary schemas.
- `ruff` + `mypy --strict` on `dxenv/`. Both must pass before a phase is considered done.
- **Never use a bare `except:` or swallow an exception in `env/` or `reward/`.** A silent failure
  here produces plausible-looking wrong numbers, which is the worst outcome in this project.
- **No silent defaults.** If a lookup misses, raise. Defaults are how leaks get introduced —
  "return normal if the test isn't found" is exactly the bug I4 exists to prevent.
- Seeding: every stochastic call takes an explicit `np.random.Generator`. No global `np.random`.
- **Persist every trajectory ever generated**, as JSONL with the config hash. Reward is pure
  (I8), so rescoring a stored corpus under new weights is free; regenerating rollouts is not.
  You will change the reward weights repeatedly. Design for it from day one.
- Logging: structured, one JSON line per episode, to `runs/{run_id}/episodes.jsonl`.

---

## 5. Phase 0 — feasibility

**Purpose: determine whether a learnable problem exists, before building anything.** Runs on raw
Synthea output with a throwaway harness. This phase is allowed to be ugly. It is not allowed to
be skipped.

### Build

- Generate ~10k patients across all Synthea modules.
- Map conditions to the flat taxonomy (`data/taxonomy.py`).
- Three probe conditions:
  - `F` — blank record. No observation at all.
  - `V` — vitals only. Demographics, vitals, presenting complaint. No tests.
  - `T` — vitals + all tests present in the record.
- Score with a simple classifier (logistic regression / gradient boosting is fine — this is not
  the agent, it is a signal detector).

### Gate A — pre-register these thresholds in `configs/gate_a.yaml` BEFORE running

A threshold chosen after seeing the result is not a threshold. Write the numbers down first,
along with what you do if it fails (harder cohort / thinner observation / shelve).

- `V > F` by a clear margin.
- **`T − V` is large.** This is the size of the entire prize. If tests buy only a few points
  over vitals alone, the reward signal is that many points wide and any policy gradient is
  buried in noise.
- Stripping `Condition` / `MedicationRequest` / `CarePlan` / `reasonCode` barely moves accuracy.
  If it collapses, everything above was measuring leakage.

The expected failure mode is `T ≈ V`, because Synthea's modules generate labs with unrealistically
clean separation from the condition. If that happens: harder patient mix, comorbidities,
ambiguous presentations, thinner initial observation.

### Tests

- `test_taxonomy_mapping_total` — every condition in the corpus maps to exactly one label; no
  unmapped SNOMED codes.
- `test_probe_conditions_disjoint` — the `V` feature set contains no feature present only in `T`.
- `test_blank_baseline_is_prior` — `F` accuracy is within tolerance of majority-class rate.
- `test_gate_a_thresholds_preregistered` — the config file exists, is committed, and its git
  timestamp precedes the results file. Enforce this mechanically; it is the only thing that makes
  a gate a gate.

---

## 6. Phase 1 — environment

The bulk of the work, and the critical path. Four of the five components below feed the Bayes
model or are fed by it.

### 6.1 `data/taxonomy.py`

Fixed **flat** label set, ~100–200 conditions. Flat is load-bearing: hierarchy is what lets an
agent hedge upward ("endocrine disorder") and collect partial credit everywhere.

Tests:
- `test_labels_are_flat` — no label is an ancestor of another in SNOMED.
- `test_label_set_frozen` — hash of the sorted label list matches the committed value.
- `test_every_corpus_condition_mapped` — no silent drops.

### 6.2 `env/filter.py` — the observation builder [I1, I2]

Allowlist of `(resource_type, field)` pairs. Permitted: demographics, vitals, presenting
complaint (symptom-coded), prior lab *values*, family history.

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

**Strings leak even when fields don't.** A lab value is fine; a lab `display` string reading
"HbA1c — diabetes monitoring" is not. Sanitize display strings inside permitted fields, and
scrub any free-text note.

Tests:
- `test_no_blocked_resource_in_observation` — over the whole corpus, no blocked type appears.
- `test_unknown_field_raises_in_strict_mode` — the allowlist fails closed.
- `test_no_label_string_in_observation` — for every patient, no observation string contains that
  patient's condition name or any of its synonyms. Run over the full corpus, not a sample.
- `test_observation_is_json_serializable_and_stable` — key order deterministic.
- `test_filter_is_idempotent` — filtering twice equals filtering once.

### 6.3 `env/actions.py` — global action space [I3]

Fixed menu: 60–100 tests/panels, treatment set, `diagnose`, `abstain`. Identical for every
patient. If the menu is derived per-patient, the menu *is* the diagnosis.

Tests:
- `test_menu_identical_across_patients` — set equality of action IDs across 1000 random patients.
- `test_menu_independent_of_ground_truth` — mutating a patient's condition does not change
  the menu.
- `test_action_ids_stable` — IDs are content-hashed, not positional, so adding a test doesn't
  silently renumber others and invalidate stored trajectories.

### 6.4 `env/obs_model.py` — generative observation model [I4]

For each `(test, condition)` pair, a distribution over results. Sample conditional on the true
latent condition. This is what makes I4 achievable: every test returns something, nothing is ever
absent, so the sparsity pattern carries no information.

Sources for the numbers: reference ranges for normals, empirical distributions from the Synthea
corpus where data exists, published likelihood ratios for the diagnostically important pairs.
It does not need to be clinically publishable. It needs to be *consistent* and *uncorrelated with
the label except through the intended channel*.

Tests:
- `test_every_pair_has_a_distribution` — full cross product of tests × conditions covered.
  No fallbacks, no `KeyError` paths.
- `test_sampling_is_total` — sample every pair 100×; no `None`, no exception, all finite.
- `test_distributions_normalize` — discrete cases sum to 1 within tolerance.
- `test_sampled_values_in_plausible_range` — property test; values within physiological bounds.
- `test_no_side_channel` — a classifier trained on *which* tests returned values (ignoring the
  values themselves) performs at chance. This is the direct test that I4 worked.
- `test_deterministic_under_seed` — same `(patient, test, seed)` → same result.

### 6.5 `env/bayes.py` — posterior and ceiling [I9]

Given explicit `p(o | c)` and enumerable priors from Synthea's modules, the model is fully
specified. Implement:

- `posterior(evidence) -> np.ndarray` — exact Bayesian update.
- `optimal_stopping_value(state, budget) -> float` — best achievable score by an agent reasoning
  perfectly from the same information. Exact by DP where tractable; upper-bounded otherwise
  (document which, and make the bound direction explicit — it must be an *upper* bound or I9
  becomes unsound).

**This module has four downstream consumers**: the ceiling (Phase 5), the shaping potential
(Phase 2), SFT soft labels (Phase 3), and the rejection-sampling process filter (Phase 3).
Everything after Phase 1 degrades if it is weak. Treat it as the critical path, not as
preprocessing.

Tests:
- `test_posterior_normalizes` and `test_posterior_nonnegative`.
- `test_posterior_matches_hand_computed` — a 2-condition, 2-test toy case worked out by hand in
  the docstring. Golden test.
- `test_posterior_moves_correct_direction` — property test: informative abnormal evidence
  increases mass on conditions with higher likelihood for that result.
- `test_irrelevant_evidence_leaves_posterior_unchanged` — a test independent of the condition
  set does not move the posterior.
- `test_ceiling_is_upper_bound` — on a toy MDP small enough to brute-force, the computed ceiling
  is ≥ the true optimum.
- `test_order_invariance` — posterior depends on the evidence set, not the order it arrived in.

### 6.6 `env/episode.py`

Gym-style API. Handles turn loop, budget accounting, dedup of repeated orders, termination.

Tests:
- `test_episode_deterministic_under_seed` [I10].
- `test_budget_never_exceeded` — property test over random policies.
- `test_remaining_budget_in_observation_matches_ledger`.
- `test_repeat_order_is_deduped` — second order of the same test costs nothing and returns the
  cached result. Without this the agent finds the cheapest test and spams it.
- `test_terminates_on_diagnose_and_abstain`.
- `test_max_turns_enforced`.

---

## 7. Phase 2 — reward engine

Pure function of `(trajectory, ground_truth, reward_config)` [I8].

```
R = brier(p, c_true) * severity_weight(c_true)
    - lambda * sum(cost(t) for t in tests_ordered)
    - mu * n_turns
    + treatment_score
    + sum(potential_shaping_terms)      # telescopes, see I6
    + sum(predict_then_verify_terms)
```

### 7.1 `reward/scoring.py`

Brier, not log-loss: log-loss is unbounded below, and one rollout putting ~0 on the truth wrecks
GRPO advantage normalization [I11]. Severity weights scale by urgency tier — without them the
agent maximises raw accuracy by nailing common benign conditions and eating the rare-severe tail.
Tiers go in `configs/severity.yaml` with the rationale written down; it is a value judgment and
should be legible as one.

Near-miss softening, if used, is an explicit **cost matrix over condition pairs keyed on
consequence of the error**, never on semantic similarity of the names.

Tests:
- `test_brier_is_proper` — **the single most important test in the repo.** For random true
  beliefs `q`, sample perturbations `p ≠ q`; expected score under `q` must be strictly higher for
  reporting `q` than for reporting any `p`. This is what mathematically rules out hedging.
- `test_brier_bounded` — output in a known finite range, never NaN.
- `test_more_mass_on_truth_scores_higher` — monotonicity.
- `test_severity_weight_orders_correctly` — missing a high-urgency condition costs more than
  missing a benign one, holding the distribution fixed.
- `test_flat_distribution_scores_below_confident_correct`.

### 7.2 `reward/costs.py`

Tests:
- `test_test_step_reward_is_never_positive` [I5] — property test over every action in the menu
  and every reachable state. This test is the guard on the invariant most likely to be violated
  by a well-meaning future edit.
- `test_dedup_does_not_double_charge`.
- `test_cost_table_covers_menu` — every action has a cost; missing entries raise.

### 7.3 `reward/shaping.py` [I6]

`Φ` = negative entropy of the Bayes posterior (or Brier of a frozen probe). Ng et al. (1999)
guarantees the optimal policy set is unchanged for *any* Φ.

Tests:
- `test_shaping_telescopes` — sum over a trajectory equals `γ^T Φ(s_T) − Φ(s_0)`, path-independent.
- `test_closed_loop_shaping_is_zero` — return to a previously visited state, net shaping ≈ 0.
  This is the property that makes shaping unfarmable.
- `test_shaping_preserves_optimal_policy` — on a toy MDP small enough to solve exactly, the
  argmax policy is identical with and without shaping. Slow test; mark it, but keep it.

### 7.4 `reward/verify.py`

Predict-then-verify: agent commits to a coarse prediction of the result before it is revealed.
Verifiable, per-step, and unhackable because ground truth is hidden until after commitment. Side
effect: an agent that can predict a result exactly learns the test is redundant.

Tests:
- `test_result_not_visible_before_commit` — structural; the prediction call cannot access the
  result object.
- `test_verify_score_zero_for_chance_predictions`.
- `test_commit_is_mandatory` — a test action without a prediction is rejected by the schema.

### 7.5 `reward/treatment.py`

Score twice: appropriateness conditional on the *declared* diagnosis (rewards coherence) and
against the *true* condition. A lucky-correct treatment under a wrong diagnosis does not collect.
Contraindication penalties are asymmetric and large — allergy list, eGFR, pregnancy status, and
drug–drug interactions are all verifiable from the record.

Tests:
- `test_contraindication_penalty_dominates_suboptimal` — a contraindicated prescription scores
  strictly worse than a suboptimal-but-safe one.
- `test_lucky_treatment_with_wrong_dx_scores_low`.
- `test_coherent_treatment_scores_above_incoherent` at fixed diagnosis accuracy.

### 7.6 Budget

Budget-conditioned: sample `B ~ p(B)` per episode, expose remaining budget in the observation.
One policy spans the whole frontier; eval sweeps `B` to produce a Pareto curve.

Tests:
- `test_budget_in_observation_and_updates`.
- `test_policy_behavior_varies_with_budget` — integration test; a policy whose test count doesn't
  respond to `B` is ignoring the constraint and getting reward another way.

### 7.7 Engine-level

- `test_reward_is_pure` [I8] — same input twice → identical output; no RNG or clock reachable.
- `test_reward_finite_over_corpus` [I11] — no NaN/inf over the full trajectory store.
- `test_abstain_priced_between_correct_and_incorrect` — near the EV of a calibrated prior guess.
- `test_lazy_policy_scores_below_working_policy` — guess-the-prior-and-order-nothing must score
  clearly worse than doing the work, but not catastrophically worse (which produces reckless
  overconfidence early in training). Assert both directions of that gap.
- `test_rescoring_stored_trajectory_matches` — golden test against frozen fixtures.

---

## 8. Phase 3 — cold start

### 8.1 Format is not an SFT problem

Use constrained decoding (xgrammar / Outlines / vLLM structured output) with a JSON schema per
action type. This makes invalid output impossible, keeps the format reward at exactly zero, and
— critically — **preserves the entropy GRPO needs.** SFT on format burns diversity; GRPO computes
advantages from within-group variation, and identical rollouts give zero gradient.

Before building anything else in this phase: run the prompted base model with constrained
decoding on 200 patients. If it clears the blank-record floor with reasonable spread, you may not
need SFT at all.

Tests:
- `test_constrained_decoding_always_valid` — 1000 samples, 100% schema-valid.
- `test_probabilities_sum_to_one_post_decode`.
- `test_grammar_rejects_off_menu_actions`.

### 8.2 Privileged teacher

Teacher sees ground truth and produces expert test sequences. Then **strip the privilege**:
regenerate reasoning conditioned only on what was visible at that turn, and filter hard for
traces that reference the answer before earning it.

Leaked reasoning in SFT data is worse than no SFT data — it trains the model to assert
conclusions it has no evidence for, which is precisely the pathology the environment exists to
prevent.

Tests:
- `test_no_trace_mentions_condition_before_diagnosis_turn` — string and embedding-similarity
  check against the condition name and synonyms. Run over the whole SFT set, fail on any hit.
- `test_teacher_trajectories_respect_action_schema`.
- `test_deleaked_trace_still_justifies_action` — sampled manual review harness; at minimum,
  assert reasoning length didn't collapse to a stub after de-leaking.

### 8.3 Rejection sampling

Sample k=8 at high temperature. **Filtering on correct diagnosis alone selects for lucky guesses
and shotgun test-ordering** — a trajectory that got it right after 40 tests is a bad
demonstration, and that habit is stubborn once trained in.

Filter on: full reward including cost; process validity (does the Bayes posterior move the right
direction after each test?); reproducibility across the k samples; condition balance.

Tests:
- `test_filter_rejects_high_cost_correct`.
- `test_filter_rejects_lucky_single_sample` — correct once in 8 does not pass.
- `test_condition_balance_within_tolerance` — otherwise the SFT set is dominated by whatever
  Synthea generates most and the model learns the prior instead of the reasoning.
- `test_process_filter_uses_posterior_not_outcome`.

### 8.4 SFT

**Label targets with the Bayes posterior, not one-hot.** SFT on winning trajectories otherwise
teaches the model to say 0.99 every time, destroying the calibration the Brier score exists to
reward — before RL even starts. Seed abstention explicitly, or the action is never sampled and
RL never discovers it.

1–2 epochs, low LR, LoRA, stop early. Deliberately undertrain. A few thousand trajectories is
plenty; the goal is a competent prior, not a finished policy.

Tests:
- `test_soft_labels_match_posterior` and `test_soft_labels_normalize`.
- `test_abstain_present_in_sft_set` — above a minimum frequency.
- `test_sft_targets_not_onehot` — assert target entropy above a floor.

### Gate B — pre-register in `configs/gate_b.yaml`

- **pass@8 clearly above pass@1.** RLVR sharpens behavior the model already samples sometimes;
  it cannot manufacture behavior that never appears.
- **Non-zero within-group reward variance.** Zero spread → zero advantage → no gradient.
- **Calibration survived**: the model's own distribution Brier-scores better than its argmax
  collapsed to one-hot.
- **Headroom below the ceiling** remains.

---

## 9. Phase 4 — GRPO

7B + LoRA, vLLM for rollout generation. Curriculum: single-condition → comorbid, short horizon →
full budget. Both prevent "order everything" becoming a survival strategy while the model is
still confused.

Monitors, asserted every N steps:
- **Reward ≤ Bayes ceiling** [I9]. Violation halts the run and dumps the offending trajectory.
  This is the automatic hacking detector; treat a trip as a leak until proven otherwise.
- Group entropy / advantage variance above floor.
- Cost distribution — watch for collapse to zero tests or to the budget cap.
- KL from the SFT reference.

Tests:
- `test_advantage_zero_when_group_identical` — the degenerate case behaves as expected.
- `test_ceiling_assertion_fires_on_synthetic_violation` — inject a trajectory scoring above the
  ceiling, confirm the halt. **Test the detector, not just the thing it detects.**
- `test_curriculum_advances_on_criterion` and `test_curriculum_does_not_skip_stages`.
- `test_kl_matches_reference_implementation` on a fixed batch.
- `test_training_never_reads_eval_split` [I12] — monkeypatch the loader to raise on eval paths.

---

## 10. Phase 5 — evaluation

### Audit suite (`eval/audit.py`)

| Probe | Pass condition |
|---|---|
| Blank-record baseline | Agent with empty observation ≈ prior. All results reported above this floor. |
| Leakage ablation | Strip conditions/meds/careplans/reasonCodes; accuracy barely moves. |
| No-test ablation | Zero-test accuracy meaningfully worse than with-test. |
| Shuffled labels | Reward drops to chance. |
| Counterfactual perturbation | Flip a lab normal→abnormal; posterior moves in the clinically correct direction. |
| Bayes ceiling | Agent ≤ ceiling. |
| Held-out modules | Split by Synthea module; report the generalization gap. |

Every probe is a test, and every probe also runs as a reported result. Baselines to report
against: prompted, SFT-only, greedy heuristic, Bayes-optimal.

Tests:
- One test per probe asserting the pass condition.
- `test_audit_suite_runs_end_to_end_on_fixture` — 10-patient fixture, fast, runs in CI.
- `test_eval_split_hash_matches` [I12].
- `test_pareto_sweep_covers_budget_range` and `test_pareto_is_broadly_monotone` — accuracy should
  not *decrease* with more budget beyond noise.
- `test_calibration_metrics_match_reference` on a synthetic set with known calibration.

---

## 11. Testing philosophy

- **Invariant tests run over the full corpus, not a sample.** A leak that appears in 2% of
  patients is still a leak, and sampling will miss it.
- **Property-based tests (`hypothesis`) for anything with a mathematical guarantee** — properness,
  telescoping, normalization, boundedness. These are the properties the design rests on; assert
  them directly rather than testing examples.
- **Golden tests with frozen fixtures** for the Bayes solver and the reward engine. Both will be
  refactored and both are easy to break subtly.
- **Test the detectors.** `test_ceiling_assertion_fires_on_synthetic_violation` and
  `test_filter_rejects_lucky_single_sample` matter as much as the things they guard. An audit
  suite that would not catch a real failure is worse than none, because it manufactures
  confidence.
- Fast suite (< 60s) runs on every commit: unit + invariants on a 100-patient fixture. Full suite
  (corpus-wide, toy-MDP policy-invariance) runs nightly and before every gate.
- Mark slow tests with `@pytest.mark.slow`. Never delete one to speed up CI.

---

## 12. Open decisions

Not yet settled. Raise these rather than silently picking:

- Exact size of the flat label set (100 vs 200 changes the Bayes DP tractability).
- Whether `optimal_stopping_value` is exact or bounded, and where the tractability boundary sits.
- Severity tier count and the weights within them.
- Whether the near-miss cost matrix ships in v1 or after the first full training run.
- `p(B)` — the budget distribution shape.
- Whether Phase 3 SFT is needed at all, pending the prompted-baseline check in 8.1.

---

## 13. Glossary

- **RLVR** — RL with verifiable rewards; reward computed from ground truth, not a learned model.
- **Ceiling** — best achievable score by exact Bayesian reasoning over the same evidence.
- **Proper scoring rule** — scoring function whose unique expected-score maximum is reporting your
  true belief. Why hedging is ruled out mathematically rather than penalized heuristically.
- **Potential-based shaping** — per-step reward of the form `γΦ(s') − Φ(s)`; provably preserves the
  optimal policy set.
- **Predict-then-verify** — agent commits to a prediction of a test result before it is revealed.
- **Gate** — a pre-registered go/no-go with thresholds written down before measurement.