# Evaluation Guide
## Supplier-Selection Evaluation Project

**Project root expected by all current scripts**

```text
C:\Users\moham\OneDrive - Constructor University\llmsr
```

This guide records the evaluation design, experiment conventions, and validation requirements. Consult the current project context and audit for later status corrections.

---

# 1. Project purpose

The project evaluates an LLM-enhanced supplier-selection framework with a strict separation of responsibilities:

```text
Natural-language intent and evidence
                ↓
Governed LLM modules
                ↓
Structured scenario/evidence objects
                ↓
Deterministic analytical engine
                ↓
Supplier ranking and scenario comparison
                ↓
Grounded explanation and human review
```

The LLM does **not** directly calculate TOPSIS, WSM, or VIKOR scores. It does not autonomously produce final numerical weights or reorder suppliers.

The completed evaluation has two complementary parts:

1. **Real public procurement applicability study**
   - TED CN/CAN records;
   - frozen 2015–2016 supplier history;
   - 2017 procurement target cases;
   - deterministic ranking and scenario experiments.

2. **Controlled module-level benchmarks**
   - intent/scenario formulation;
   - advisory scenario refinement;
   - evidence extraction;
   - explanation and governance;
   - prompt/schema/reasoning ablations;
   - noisy and incomplete evidence tests.

---

# 2. Non-negotiable terminology

Use these terms consistently in code, reports, and manuscript text.

## Required terms

```text
historical reference outcome
historical reference supplier
historically active supplier comparison set
real public procurement applicability study
retrospective reference-supplier positioning
deterministic ranking engine
controlled author-created benchmark
author-adjudicated benchmark
```

## Terms that must not be used for the TED evaluation

```text
optimal supplier
correct supplier
true best supplier
actual bidder pool
candidate bidder pool
field validation
validated procurement outcome
```

The observed TED awardee is only a historical reference outcome. The comparison sets are reconstructed from prior public-procurement activity and are not actual tender bidder lists.

---

# 3. High-level architecture

The conceptual framework contains:

```text
F         structured supplier feature table
D_evid    qualitative evidence corpus
P         metric, scenario, policy, and governance catalogue
M_I       intent/scenario formulation LLM
M_E       evidence extraction LLM
M_X       explanation/governance LLM
M_W       optional bounded advisory weight-refinement LLM
tau_s     approved scenario
E_s       extracted evidence set
G_s       evidence-role mapping
R_s       deterministic ranking
X_s       grounded explanation
C1-C4     governance checkpoints
L         audit log
```

Evidence roles:

```text
ELIGIBILITY
FEATURE
SCENARIO
ADVISORY
```

The deterministic engine is the numerical authority.

---

# 4. Confirmed project structure

The current project already contains the following main directories:

```text
llmsr/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── analysis/
│   ├── e3/
│   ├── e3b/
│   ├── e4/
│   └── e5/
├── reports/
└── scripts/
```

## 4.1 Important raw data

```text
data/raw/
├── can_2015.csv
├── can_2016.csv
├── can_2017.csv
└── cn_2017.csv
```

## 4.2 Important processed data

```text
data/processed/can_awards/
data/processed/cn_targets_2017/
data/processed/supplier_history_2015_2016/
data/processed/winner_history_2017/
data/processed/candidate_pools_2017/
data/processed/supplier_feature_store/
```

Critical files include:

```text
data/processed/cn_targets_2017/procurement_cases_2017.parquet
data/processed/winner_history_2017/winner_history_map_2017.parquet
data/processed/candidate_pools_2017/target_pool_assignments_2017.parquet
data/processed/candidate_pools_2017/historical_supplier_context_eligibility_2015_2016.parquet
data/processed/supplier_feature_store/supplier_global_features_2015_2016.parquet
data/processed/supplier_feature_store/supplier_cpv2_features_2015_2016.parquet
data/processed/supplier_feature_store/target_case_features_2017.parquet
```

## 4.3 Important analysis outputs

```text
data/analysis/e1_rankings/e1_case_ranking_results.parquet
data/analysis/e1b_rankings/e1b_case_model_results.parquet
data/analysis/e1x_rankings/e1x_case_model_results.parquet
data/analysis/e2_scenarios/e2_case_scenario_results.parquet
data/analysis/e2x_scenarios/e2x_case_scenario_results.parquet
```

LLM outputs:

```text
data/analysis/e3_final/
data/analysis/e3b_final/
data/analysis/e3b_final_v2/
data/analysis/e4_final/
data/analysis/e5_final/
data/analysis/e5_final_v2/
```

Reviewer 2.2 additions:

```text
data/analysis/r22_e1c_ablation/
data/analysis/r22_e3c_ablation/
data/analysis/r22_e4b_robustness/
```

## 4.4 Important reports

```text
reports/e1_rankings/
reports/e1x_rankings/
reports/e2x_scenarios/
reports/e3_final/
reports/e3b_final_v2/
reports/e4_final/
reports/e5_final_v2/
reports/r22_preflight/
reports/r22_e1c_ablation/
reports/r22_e2b_robustness/
reports/r22_e3c_ablation/
reports/r22_e4b_robustness/
```

---

# 5. Data lineage

The intended reproducible data flow is:

```text
CAN 2015-2017
    ↓
build_can_award_tables.py
    ↓
normalized award facts
    ↓
build_supplier_dimension_2015_2016.py
    ↓
frozen supplier entities and aliases
    ↓
build_supplier_feature_store_2015_2016.py
    ↓
historical supplier features

CN 2017
    ↓
build_cn_targets_2017.py
    ↓
strict procurement target cases
    ↓
map_2017_winners_to_history.py
    ↓
historical reference-supplier mapping
    ↓
build_candidate_pools_2017.py
    ↓
historically active supplier comparison sets

historical features + target cases + comparison sets
    ↓
build_e1b_procurement_fit_indices.py
build_extended_procurement_fit_indices.py
    ↓
E1/E1x ranking experiments
    ↓
E2/E2x scenario experiments
    ↓
R2.2 ablations and robustness
```

Strict temporal rule:

```text
2015-2016 = supplier identity and feature construction
2017      = target procurements and historical reference outcomes
```

Never use 2017 award outcomes to construct historical supplier features.

---

# 6. Completed structured-data experiments

## 6.1 E1 deterministic baselines

Runner:

```text
scripts/run_e1_deterministic_rankings.py
```

Methods:

```text
FREQUENCY
WSM
TOPSIS
VIKOR
```

Primary comparison-set definition:

```text
CPV2_MIN1
```

Main result:

| Method | W@10 | W@50 | MRR |
|---|---:|---:|---:|
| Frequency | 3.338% | 11.134% | 0.018528 |
| WSM | 5.011% | 13.396% | 0.024259 |
| TOPSIS | 5.259% | 14.148% | 0.025639 |
| VIKOR | 5.083% | 13.761% | 0.024189 |

## 6.2 E1x procurement-fit feature models

Runner:

```text
scripts/run_e1x_extended_procurement_fit_rankings_v2.py
```

Frozen final backbone:

```text
M3_GEO_BUYER
```

Feature groups:

```text
CATEGORY_FIT
GEOGRAPHIC_FIT
BUYER_RELATIONSHIP
ACTIVITY_PROFILE
```

Default group weights:

```json
{
  "CATEGORY_FIT": 0.25,
  "GEOGRAPHIC_FIT": 0.25,
  "BUYER_RELATIONSHIP": 0.25,
  "ACTIVITY_PROFILE": 0.25
}
```

Main result:

```text
M0 W@10          = 5.259%
M3 W@10          = 31.642%
M0 median rank   = 914.2
M3 median rank   = 34.0
```

## 6.3 E1c feature-group ablation

Runner:

```text
scripts/run_r22_e1c_feature_ablation.py
```

Models:

```text
M0_ORIGINAL
M3_FULL
ABLATE_CATEGORY
ABLATE_GEOGRAPHY
ABLATE_BUYER
ABLATE_ACTIVITY
CATEGORY_ONLY
GEOGRAPHY_ONLY
BUYER_ONLY
ACTIVITY_ONLY
```

Completed result:

| Removed group | Change in W@10 |
|---|---:|
| Geography | -7.593 percentage points |
| Buyer relationship | -3.603 points |
| Category fit | -2.600 points |
| Activity profile | -0.121 points |

Interpretation:

```text
Geography is the strongest marginal group.
Buyer and category provide complementary signal.
Activity has little marginal top-k effect once other groups are present.
```

## 6.4 E2x scenario sensitivity

Runner:

```text
scripts/run_e2x_scenario_sensitivity.py
```

Scenarios:

```json
{
  "BALANCED": {
    "CATEGORY_FIT": 0.25,
    "GEOGRAPHIC_FIT": 0.25,
    "BUYER_RELATIONSHIP": 0.25,
    "ACTIVITY_PROFILE": 0.25
  },
  "CONTINUITY": {
    "CATEGORY_FIT": 0.25,
    "GEOGRAPHIC_FIT": 0.20,
    "BUYER_RELATIONSHIP": 0.40,
    "ACTIVITY_PROFILE": 0.15
  },
  "DIVERSIFICATION": {
    "CATEGORY_FIT": 0.30,
    "GEOGRAPHIC_FIT": 0.35,
    "BUYER_RELATIONSHIP": 0.10,
    "ACTIVITY_PROFILE": 0.25
  }
}
```

Main results:

| Scenario | W@10 | W@50 | MRR | Median rank |
|---|---:|---:|---:|---:|
| Balanced | 31.642% | 56.293% | 0.163086 | 34 |
| Continuity | 32.155% | 56.338% | 0.164561 | 34 |
| Diversification | 30.121% | 56.023% | 0.157701 | 35 |

Do not describe continuity as statistically superior. It is a policy profile.

## 6.5 E2b bootstrap and comparison-set robustness

Runner:

```text
scripts/run_r22_e2b_bootstrap_robustness.py
```

Completed with:

```text
300 bootstrap resamples
```

Outputs:

```text
reports/r22_e2b_robustness/r22_e2b_robustness_report.md
reports/r22_e2b_robustness/r22_e2b_e1_bootstrap.csv
reports/r22_e2b_robustness/r22_e2b_e2_bootstrap.csv
reports/r22_e2b_robustness/r22_e2b_scenario_rank_movement.csv
```

---

# 7. Controlled benchmark files and manual workflow

This section is critical. Benchmark files are research artifacts and must be treated as versioned, frozen inputs.

## 7.1 E3 intent benchmark

Confirmed file:

```text
data/e3/e3_benchmark_50x3_gold_reviewed_v1.csv
```

Exact columns:

```text
request_id
base_case_id
variant_id
category
request_text
gold_scenario_profile
gold_priority_groups_json
gold_constraints_json
gold_evidence_requests_json
gold_requires_clarification
```

The CSV may also contain an automatically written index column. Scripts should not depend on it.

Design:

```text
50 base cases
3 variants per base case: V0, V1, V2
150 total requests
```

Categories:

```text
direct_simple
ambiguous
multi_criteria
constraint_sensitive
scenario_comparison
```

Scenario values:

```text
BALANCED
CONTINUITY
DIVERSIFICATION
CUSTOM
CLARIFICATION_REQUIRED
```

Priority-group values:

```text
CATEGORY_FIT
GEOGRAPHIC_FIT
BUYER_RELATIONSHIP
ACTIVITY_PROFILE
```

Example constraint object:

```json
{
  "category": "COMPLIANCE",
  "hardness": "HARD"
}
```

Evidence-request examples:

```text
CERTIFICATION
SUSTAINABILITY
SOCIAL
DELIVERY_CAPACITY
TECHNICAL_CAPABILITY
PAST_PERFORMANCE
FINANCIAL_STABILITY
GOVERNANCE_COMPLIANCE
```

### E3 manual adjudication file

Confirmed file:

```text
data/e3/e3_benchmark_adjudication_v1.csv
```

Exact columns:

```text
base_case_id
category
adjudication_status
gold_scenario_profile
gold_priority_groups_json
gold_constraints_json
gold_evidence_requests_json
gold_requires_clarification
adjudication_note
recommended_change
```

Confirmed adjudication statuses include:

```text
ACCEPT
ACCEPT_WITH_NOTE
```

This file exists to make manual review easier. Gold labels should be changed only through an explicit new adjudication version.

### Intended E3 editing workflow

```text
Draft base cases
    ↓
Create V0/V1/V2 paraphrases
    ↓
Create adjudication CSV at base-case level
    ↓
Manually review scenario, priorities, constraints, evidence needs, clarification
    ↓
Freeze gold-reviewed benchmark
    ↓
Run API evaluator
    ↓
Store predictions and metrics outside the benchmark directory
```

Never edit `e3_benchmark_50x3_gold_reviewed_v1.csv` after looking at final test results. Create a new version instead.

## 7.2 E3c prompt/schema/reasoning ablation

Runner:

```text
scripts/run_r22_e3c_prompt_schema_ablation.py
```

E3c does not require a manually authored benchmark. It automatically selects:

```text
one V0 request from each of the first N base cases
```

Generated subset:

```text
data/analysis/r22_e3c_ablation/e3c_v0_20_base_cases.csv
```

Configurations evaluated:

```text
STRICT_ZERO_SHOT_LOW_REUSED
STRICT_FEW_SHOT_LOW
RELAXED_FEW_SHOT_LOW
STRICT_FEW_SHOT_NONE
```

Important implementation behavior:

- the strict zero-shot baseline is reused from `data/analysis/e3_final/e3_api_results.csv`;
- generated runner copies are created under `data/analysis/r22_e3c_ablation/_generated_runners/`;
- the relaxed schema keeps required typed fields but changes Pydantic `extra="forbid"` to `extra="ignore"` and removes cross-field governance validation;
- it is not a free-text condition;
- no API call occurs without `--run-api`.

Completed 20-case results:

| Configuration | Priority F1 | Mean latency |
|---|---:|---:|
| Strict zero-shot, low | 0.900 | 1.672 s |
| Strict few-shot, low | 1.000 | 2.523 s |
| Relaxed few-shot, low | 0.950 | 1.940 s |
| Strict few-shot, none | 0.950 | 1.977 s |

## 7.3 E3b advisory weight-refinement benchmark

Confirmed file:

```text
data/e3b/e3b_benchmark_20_gold.csv
```

Exact columns:

```text
case_id
category
user_request
initial_scenario_profile
base_weights_json
ranking_summary_json
gold_recommendation_type
gold_increase_json
gold_decrease_json
gold_preserve_json
gold_adjustment_strength
gold_requires_human_approval
gold_clarification_required
gold_risk_flags_json
```

Recommendation types include:

```text
KEEP
ADJUST
CLARIFY
```

Strength values:

```text
SMALL
MODERATE
LARGE
```

The LLM never writes final numerical weights. A deterministic controller applies bounded changes.

Controller constraints:

```text
minimum group weight = 0.10
maximum group weight = 0.45
weights sum to 1.0
```

## 7.4 E4 evidence benchmark

Confirmed template:

```text
data/e4/e4_benchmark_template_5packets.csv
```

Confirmed final benchmark:

```text
data/e4/e4_benchmark_40_gold.csv
```

Exact columns:

```text
packet_id
supplier_name
category
evidence_packet
gold_claims_json
```

Each element inside `gold_claims_json` has:

```text
claim_id
aspect
polarity
temporal_status
evidence_role
source_quote
normalized_claim
```

Allowed/observed aspect values include:

```text
CERTIFICATION
SUSTAINABILITY
SOCIAL
DELIVERY_CAPACITY
TECHNICAL_CAPABILITY
PAST_PERFORMANCE
FINANCIAL_STABILITY
GOVERNANCE_COMPLIANCE
```

Polarity values:

```text
POSITIVE
NEGATIVE
NEUTRAL
MIXED
UNKNOWN
```

Temporal-status values:

```text
CURRENT
HISTORICAL
FUTURE
EXPIRED
UNKNOWN
```

Evidence-role values:

```text
ELIGIBILITY
FEATURE
SCENARIO
ADVISORY
```

### E4 manual editing rule

Every `source_quote` must occur verbatim inside `evidence_packet`.

A benchmark validator should fail when:

```text
source_quote not in evidence_packet
duplicate packet_id
duplicate claim_id within a packet
invalid enum value
empty gold claim list
```

### E4 intended manual workflow

```text
Start from 5-packet template
    ↓
Write a short evidence packet
    ↓
Split packet into atomic gold claims
    ↓
Copy exact supporting source_quote
    ↓
Assign aspect, polarity, time, and evidence role
    ↓
Run validation
    ↓
Review manually
    ↓
Freeze final 40-packet benchmark
```

## 7.5 E4b robustness benchmark

Runner:

```text
scripts/run_r22_e4b_evidence_robustness.py
```

This benchmark is generated automatically from the frozen E4 benchmark.

Conditions:

```text
NOISY
INCOMPLETE
optional CONFLICTING
```

Rules:

### NOISY

Add irrelevant but plausible text. Keep all original gold claims.

### INCOMPLETE

Remove a sentence supporting one claim. Remove every gold claim whose source quote is no longer present.

The model must not be penalized for omitting evidence that was removed.

### CONFLICTING

Add an explicit later dispute as a separate claim with:

```json
{
  "polarity": "MIXED",
  "temporal_status": "CURRENT",
  "evidence_role": "ADVISORY"
}
```

Conflict handling should be manually inspected. The model should preserve or escalate conflict, not silently resolve it.

## 7.6 E5 explanation/governance benchmark

Confirmed file:

```text
data/e5/e5_benchmark_24_gold.csv
```

Exact columns:

```text
case_id
category
user_request
scenario_profile
priority_groups_json
group_weights_json
supplier_context_json
evidence_claims_json
gold_top_supplier_order_json
gold_required_priority_groups_json
gold_required_governance_flags_json
gold_requires_human_review
gold_forbidden_terms_json
```

Each supplier object inside `supplier_context_json` contains:

```text
supplier_id
supplier_name
deterministic_rank
final_score
margin_note
group_scores
structured_summary
```

Each claim object inside `evidence_claims_json` contains:

```text
claim_id
supplier_id
aspect
evidence_role
polarity
claim_text
source_quote
```

The explanation LLM must:

```text
preserve deterministic supplier order
preserve top-1 supplier
use only supplied supplier IDs
cite only supplied evidence claims
not invent evidence
not recalculate scores
not use forbidden certainty language
flag escalated review when required
```

Typical forbidden terms:

```text
optimal
winner
objectively best
guaranteed
will win
certain winner
```

`gold_requires_human_review` means additional or escalated review beyond ordinary procurement oversight.

---

# 8. API runner contract

Every LLM experiment should follow this pattern.

## Required CLI behavior

```text
--benchmark
--model
--output-dir
--report-dir
--reasoning-effort
--max-output-tokens
--resume
```

Reviewer-specific wrappers also use:

```text
--root
--run-api
```

## Non-negotiable API safety

```text
No API call unless --run-api is explicitly supplied.
Fail if OPENAI_API_KEY is missing.
Use checkpoint files for resume.
Never overwrite frozen benchmark files.
Write raw API outputs before metric aggregation.
Record model and reasoning settings in metadata JSON.
```

## Standard output files

```text
*_api_results.csv
*_api_results.parquet
*_checkpoint.jsonl
*_case_metrics.csv
*_packet_metrics.csv
*_claim_level_matches.csv
*_summary_metrics.csv
*_metrics_by_category.csv
*_run_metadata.json
*_report.md
```

Not every experiment uses every output type, but naming should remain consistent.

---

# 9. Preflight and validation rules

Run before any new experiment:

```powershell
python scripts/run_r22_preflight.py --root .
```

The confirmed preflight previously produced:

```text
40 passed
0 failed
```

Validate inputs before computation.

Minimum checks:

```text
required file exists
required columns exist
IDs are unique
JSON columns parse successfully
enum values are valid
source quotes are grounded
weights sum to one
weight bounds are satisfied
case IDs match across reused outputs
no outcome leakage from 2017 into 2015-2016 features
```

Every new experiment should support a no-cost mode:

```text
--check-only
or preparation without --run-api
or --self-test
```

---

# 10. Completed evaluation status

## Complete

```text
TED preprocessing
supplier dimension
target-case construction
historical reference mapping
comparison-set construction
feature store
E1 MCDM baselines
E1x procurement-fit models
E1c feature ablation
E2/E2x scenario sensitivity
E2b bootstrap robustness
E3 intent formulation
E3c prompt/schema/reasoning ablation on 20 cases
E3b advisory refinement
E4 evidence extraction
E4b noisy/incomplete robustness on 15 packets
E5 explanation/governance
```

## Prepared but not yet complete

```text
E1d historical-proxy subgroup sensitivity
E1e traditional ML baselines
E2c profit-lead-time-ESG cross-modal experiment
E4b conflicting-evidence extension
disruption-response demonstration
```

Do not report prepared code as completed evidence.

---

# 11. Remaining experiment priorities

Use this order.

## Priority 1: E1d historical-proxy subgroup sensitivity

Questions:

```text
Does the full model behave differently when the reference supplier has prior buyer history?
Are gains concentrated among highly visible historical suppliers?
Do geography features behave differently for domestic versus cross-border suppliers?
Does performance change strongly by comparison-set size?
```

Report full model and key ablations by subgroup. Do not call this a complete fairness audit.

## Priority 2: E1e traditional ML baseline

Compare:

```text
regularized logistic regression
histogram gradient boosting
frozen M3 TOPSIS
```

Rules:

```text
strict temporal split
negative sampling only for training
complete comparison-set scoring during validation/test
same test cases for all methods
historical reference outcome as label
```

The ML model predicts historical awardee positioning. It does not learn an objectively optimal supplier.

## Priority 3: E2c profit-lead-time-ESG

Controlled benchmark with known:

```text
profit_per_unit
lead_time_days
ESG evidence packet
gold ESG claims
```

Flow:

```text
ESG text
    ↓ M_E
structured ESG claims
    ↓ deterministic policy mapping
approved ESG feature or eligibility status
    ↓ deterministic TOPSIS
scenario-specific supplier ranking
```

Scenarios:

```text
ECONOMIC_PRIORITY
BALANCED
ESG_PRIORITY
ESG_CONSTRAINED
```

The LLM extracts evidence only.

## Priority 4: conflicting evidence

Run existing E4b with:

```text
--include-conflicting
```

Manually review whether the model preserves both claims or escalates review.

---

# 12. Coding standards

## 12.1 General

```text
Python 3.11+
type hints
argparse CLI
pathlib.Path
clear error messages
deterministic random seeds
UTF-8 or UTF-8-SIG for manually reviewed CSVs
Parquet for large case-level outputs
CSV and Markdown for summaries
JSON metadata for run configuration
```

## 12.2 Data-frame rules

```text
Validate required columns immediately after loading.
Do not silently rename unexpected columns.
Do not silently drop duplicate IDs.
Never depend on an unnamed CSV index column.
Parse nested JSON explicitly.
Use stable sorting before deterministic outputs.
```

## 12.3 Experiment rules

```text
Freeze benchmark before test execution.
Do not tune prompts after reading final test errors.
Separate pilot and final output directories.
Reuse frozen baselines only when model and settings match.
Record reused baseline provenance.
Use identical cases for method comparisons.
Use tie-aware ranks.
```

## 12.4 Reporting rules

Always include:

```text
N
metric definition
confidence interval where relevant
comparison setting
temporal scope
limitations
validation checks
```

Never describe a result as “accuracy” when it is actually historical reference positioning.

---

# 13. Output naming convention

Preferred pattern:

```text
data/analysis/<experiment_name>/
reports/<experiment_name>/
```

Examples:

```text
data/analysis/r22_e1d_proxy_sensitivity/
reports/r22_e1d_proxy_sensitivity/

data/analysis/r22_e1e_ml_baselines/
reports/r22_e1e_ml_baselines/

data/analysis/r22_e2c_cross_modal/
reports/r22_e2c_cross_modal/
```

Each experiment should produce:

```text
<experiment>_manifest.json
<experiment>_validation_checks.csv
<experiment>_case_results.parquet
<experiment>_summary.csv
<experiment>_report.md
```

API experiments additionally produce:

```text
<experiment>_api_results.csv
<experiment>_api_results.parquet
<experiment>_checkpoint.jsonl
<experiment>_run_metadata.json
```

---

# 14. Research integrity requirements

```text
Do not create new benchmarks from memory without inspecting existing schemas.
Do not overwrite frozen gold benchmarks.
Do not change historical results to improve metrics.
Do not add 2017 outcome-derived features.
Do not call comparison sets bidder pools.
Do not call the observed awardee optimal.
Do not let an LLM calculate or modify supplier ranks.
Do not silently call an API.
Do not claim a script was validated merely because it compiled.
Do not report pilot results as final.
Do not invent missing project files or directories.
Do not rebuild working ranking logic when an existing validated runner can be reused.
```

---

# 15. Evaluation preparation

When starting in the repository:

1. Read this handoff.
2. Print the current repository tree.
3. Confirm the exact project root.
4. Run:

```powershell
python scripts/run_r22_preflight.py --root .
```

5. Inspect these frozen benchmark schemas:

```text
data/e3/e3_benchmark_50x3_gold_reviewed_v1.csv
data/e3/e3_benchmark_adjudication_v1.csv
data/e3b/e3b_benchmark_20_gold.csv
data/e4/e4_benchmark_40_gold.csv
data/e5/e5_benchmark_24_gold.csv
```

6. Inspect completed reports before changing anything:

```text
reports/r22_e1c_ablation/r22_e1c_ablation_report.md
reports/r22_e2b_robustness/r22_e2b_robustness_report.md
reports/r22_e3c_ablation/r22_e3c_ablation_report.md
reports/r22_e4b_robustness/r22_e4b_robustness_report.md
```

7. Continue only with a clearly scoped pending experiment.
8. Add preflight and self-test behavior before full execution.
9. Run a pilot.
10. Inspect outputs manually.
11. Freeze the final run.
12. Update manuscript tables only from saved result files.

---

# 16. Final research framing

The final paper should present three complementary evaluation studies:

```text
Study A
Real public procurement applicability
E1, E1x, E1c, E2/E2x, E2b, pending E1d/E1e

Study B
Controlled cross-modal supplier selection
pending E2c profit-lead-time-ESG

Study C
Governed LLM module evaluation
E3, E3c, E3b, E4, E4b, E5
```

Central conclusion:

```text
Procurement-specific feature engineering improves deterministic ranking on real
public procurement records. Explicit scenarios create transparent and stable
policy-sensitive ranking changes. Bounded LLM modules can support intent
formulation, evidence extraction, advisory refinement, and explanation while
deterministic analytics and human governance retain decision authority.
```
