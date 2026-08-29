# Project Context
## Governed LLM-Enhanced Supplier-Selection Framework

**Repository root**

```text
C:\Users\moham\OneDrive - Constructor University\llmsr
```

**Purpose of this document:** Record the implementation, completed evaluations,
validation evidence, limitations, and pending work as of the repository audit.

---

# 1. Source-of-truth order

When documents conflict, use this order:

1. Current repository code, manifests, validation files, saved result tables, and benchmark hashes.
2. `reports/repository_audit/REPOSITORY_AUDIT.md` and its machine-readable appendices.
3. This master context and `PROJECT_FACTS.json`.
4. Older handoff or consolidated reports only when not contradicted by newer audit evidence.

The old handoff is stale on E1d, E1e, E2c, E3b terminology, CPV4, and buyer recency.

---

# 2. Research problem and honest framing

The original practical problem is supplier selection with quantitative and
qualitative information, policy priorities, and human governance. Complete
private tender-level data were unavailable, so the evaluation uses:

- real TED administrative records for large-scale applicability and retrospective positioning;
- controlled supplier cases for known profit, lead-time, ESG, evidence, and governance behavior;
- controlled LLM benchmarks for intent, evidence, advisory, and explanation modules.

TED was not designed to reveal all bidders, unsuccessful offers, complete prices,
delivery commitments, buyer scores, or internal deliberations. Therefore:

- the observed awardee is a **historical reference outcome**;
- the reconstructed sets are **historically active supplier comparison sets**;
- TED results measure **retrospective reference-supplier positioning**, not optimality;
- the study is a **real public procurement applicability study**, not field validation.

---

# 3. Framework mechanism

Conceptual objects:

```text
F       structured feature table
D_evid  supplier evidence corpus
P       metric, policy, scenario, and governance catalogue
M_I     intent/scenario formulation LLM
M_E     evidence extraction LLM
M_X     explanation/governance LLM
M_W     optional bounded advisory refinement LLM
tau_s   approved scenario
E_s     extracted evidence
G_s     evidence-role mapping
R_s     deterministic supplier ranking
X_s     grounded explanation
C1-C4   governance checkpoints
L       audit log
```

Operational flow:

```text
Procurement request
→ M_I: scenario, priorities, constraints, evidence needs, clarification
→ M_E: atomic source-grounded evidence claims
→ deterministic policy/feature mapping
→ deterministic WSM/TOPSIS/VIKOR/Frequency or grouped TOPSIS
→ fixed ranking R_s
→ M_X: explanation and governance flags without reranking
→ human approval
```

`M_W` is optional and can only recommend qualitative directions and strength.
A deterministic controller converts approved advice into bounded weights.

**Non-negotiable:** No LLM performs numerical ranking or autonomously changes the
supplier order.

---

# 4. Scientific evaluation structure

## Study A — Real public procurement applicability

TED CN/CAN pipeline, supplier history, reference mapping, comparison sets,
features, E1/E1x/E1c/E1d/E1e, E2/E2x/E2b.

Question answered: Can the deterministic framework operate at scale on real
procurement records, and how do features, scenarios, methods, comparison-set
choices, subgroups, uncertainty, and ML baselines affect historical reference
positioning?

## Study B — Controlled cross-modal supplier selection

E2c profit–lead-time–ESG experiment.

Question answered: Under controlled known inputs, can qualitative ESG evidence be
extracted, policy-mapped, and combined with structured criteria while ranking
remains deterministic?

## Study C — Governed LLM module evaluation

E3, E3c/E3d, E3b, E4/E4b, and E5.

Question answered: Can bounded LLM modules reliably formulate intent, extract
source-grounded evidence, advise within limits, and explain fixed rankings?

Do not collapse these studies into one vague “accuracy” claim.

---

# 5. TED data lineage and verified counts

## CAN normalization

- Raw CAN rows: **1,812,113**
- Valid award facts: **1,540,684**
- Historical 2015–2016 award facts: **977,201**
- Assigned to supplier entities: **880,906**
- Award key: `(ID_NOTICE_CAN, ID_AWARD)`
- No conflicting repeated core award keys after normalization.

## Frozen supplier dimension

- Supplier entities: **301,164**
- Reliable national-ID anchors: **10,739**
- One-to-one name-country bridges: **9,790**
- Ambiguous bridges: **239**
- No 2017 identity learning.
- No aggressive fuzzy matching.

## CN 2017 targets

- Raw CN rows: **721,676**
- Unique notices: **213,336**
- Strict target cohort: **46,942**

## Historical reference mapping

- Historical matches: **26,052 / 46,942 = 55.498%**
- Unique name-country: **22,374**
- Direct identifier: **3,284**
- Unique name-town: **394**
- No match: **19,405**
- Ambiguous: **533**
- Invalid: **952**

## Temporal rule

```text
2015–2016: identity, history, comparison-set eligibility, supplier features
2017: target procurements and historical reference outcomes
```

No inspected path placed 2017 winner/rank/outcome fields into historical features.

---

# 6. CPV comparison sets

CPV is the Common Procurement Vocabulary hierarchy:

- CPV2: first two digits / division / broad;
- CPV3: first three digits / group / narrower;
- CPV4: first four digits / class / narrower again;
- MIN1: at least one historical award in the matched context;
- MIN2: at least two historical awards.

| Set | Reference included | Conditional coverage | Median size | Mean size | Status |
|---|---:|---:|---:|---:|---|
| CPV2_MIN1 | 22,230 | 85.329% | 15,525 | 19,995.9 | Primary |
| CPV3_MIN1 | 18,839 | 72.313% | 3,753 | 6,855.2 | Sensitivity |
| CPV4_MIN1 | 16,232 | 62.306% | 1,485 | 2,569.8 | Preprocessed, not ranked in E1 |
| CPV2_MIN2 | 16,940 | 65.024% | 4,621 | 5,780.6 | Sensitivity |

Never call these actual bidders.

---

# 7. Supplier feature store and ranking mechanism

Feature-store rows:

- global suppliers: **301,164**
- supplier-CPV2: **333,835**
- supplier-CPV3: **364,027**
- supplier-CPV4: **392,242**
- target cases: **46,942**

Methods:

- Frequency;
- WSM;
- TOPSIS;
- VIKOR (`v=0.5`);
- grouped TOPSIS.

Ranking semantics:

- scores rounded to 12 decimals before tie assignment;
- best rank, worst rank, midrank, and tie-group size retained;
- W@k uses `rank_mid <= k`;
- MRR uses `1/rank_mid`;
- percentile uses `1-(rank_mid-1)/(pool_size-1)`.

The frozen final backbone is:

```text
M3_GEO_BUYER = grouped TOPSIS(
  CATEGORY_FIT,
  GEOGRAPHIC_FIT,
  BUYER_RELATIONSHIP,
  ACTIVITY_PROFILE
)
```

Balanced group weights are 0.25 each.

Known feature exception: `PRIOR_BUYER_RECENCY` was inactive because the extended
index builder read `DT_AWARD`, while normalized awards expose `DT_AWARD_CLEAN`.
The mask prevented corruption, but buyer recency must not be claimed as an active
contributor in saved E1x results.

---

# 8. Structured-data experiments and verified results

## E1 — Original deterministic methods

Primary CPV2_MIN1, N=22,230:

| Method | W@10 | W@50 | MRR |
|---|---:|---:|---:|
| Frequency | 3.338% | 11.134% | 0.018528 |
| WSM | 5.011% | 13.396% | 0.024259 |
| TOPSIS | 5.259% | 14.148% | 0.025639 |
| VIKOR | 5.083% | 13.761% | 0.024189 |

Validation: nine passed, one CPV4 scope check failed because CPV4 was not
configured/executed. The executed E1 pools remain usable.

## E1x — Procurement-fit models

- M0 W@10: **5.259%**, median rank **914.2**
- M1 category W@10: **15.272%**, median **226**
- M2 geography W@10: **28.039%**, median **40**
- M2 buyer W@10: **24.049%**, median **125**
- M3 final W@10: **31.642%**, W@50 **56.293%**, MRR **0.163086**, median **34**
- M4 W@10: **31.682%**
- M5 W@10: **31.462%**

Main conclusion: procurement-specific feature representation mattered much more
than the small differences among mainstream MCDM methods.

## E1c — Feature ablation

| Removed group | W@10 decline |
|---|---:|
| Geography | -7.593 pp |
| Buyer | -3.603 pp |
| Category | -2.600 pp |
| Activity | -0.121 pp |

Geography is the strongest marginal group. Buyer and category are complementary.
Activity adds little marginal top-k signal. Buyer-only results are heterogeneous
and represent continuity/incumbency context, not general quality.

## E2/E2x — Scenarios

```text
BALANCED        category .25, geography .25, buyer .25, activity .25
CONTINUITY      category .25, geography .20, buyer .40, activity .15
DIVERSIFICATION category .30, geography .35, buyer .10, activity .25
```

| Scenario | W@10 | W@50 | MRR | Median rank |
|---|---:|---:|---:|---:|
| Balanced | 31.642% | 56.293% | 0.163086 | 34 |
| Continuity | 32.155% | 56.338% | 0.164561 | 34 |
| Diversification | 30.121% | 56.023% | 0.157701 | 35 |

Continuity is not statistically superior; it is a policy profile. Diversification
causes the larger shortlist movement.

## E2b — Bootstrap robustness

- 300 case-level bootstrap resamples.
- Balanced W@10 95% CI: **[31.050, 32.230]**
- Continuity: **[31.556, 32.764]**
- Diversification: **[29.483, 30.675]**
- Continuity Spearman vs balanced: **0.9944**
- Diversification Spearman: **0.9701**

Comparison-set sensitivity was executed for CPV3_MIN1 and CPV2_MIN2, not CPV4.

## E1d — Historical-proxy subgroup sensitivity

Status: report-backed execution.

Verified highlight:

- prior-buyer subgroup M3 W@10: **72.747%**
- no-prior-buyer subgroup M3 W@10: **20.805%**

Interpret only as historical-proxy/subgroup sensitivity. It is not a fairness
audit and does not prove supplier quality.

## E1e — Traditional ML baselines

Saved chronological run:

- train: **6,000** cases;
- validation: **1,000**;
- test: **1,399**.

Test W@10:

- frozen TOPSIS M3: **31.451%**
- regularized logistic regression: **28.949%**
- histogram gradient boosting: **17.941%**

Status: executed, but its formal “pilot versus final” designation and standalone
validation are unresolved. Do not present it as an unqualified final baseline.

## E2c — Controlled profit–lead-time–ESG

- 72 suppliers;
- 12 cases per four scenarios;
- author-controlled benchmark.

Top-1 agreement:

- economic priority: **0.750**
- balanced: **0.667**
- ESG priority: **0.583**
- ESG constrained: **0.833**

This is Study B controlled evidence, not real TED outcome validation.

---

# 9. LLM experiments and verified results

## E3 — Intent/scenario formulation

- 50 author-adjudicated base cases;
- V0/V1/V2 variants;
- 150 requests.

Results:

- API success 1.000;
- scenario accuracy 0.973;
- clarification accuracy 0.993;
- priority exact 0.893;
- priority F1 0.916;
- constraint F1 0.990;
- evidence-request F1 0.996;
- paraphrase scenario consistency 0.940;
- priority Jaccard 0.909.

## E3c/E3d — Prompt, schema, reasoning ablation

20 V0 requests per configuration:

| Configuration | Priority F1 | Mean latency |
|---|---:|---:|
| Strict zero-shot, low | 0.900 | 1.672 s |
| Strict few-shot, low | 1.000 | 2.523 s |
| Relaxed few-shot, low | 0.950 | 1.940 s |
| Strict few-shot, no reasoning | 0.950 | 1.977 s |

This is a small controlled ablation, not a universal inference-setting claim.
E3d is contained inside the E3c output tree, not a separate experiment tree.

## E3b — Bounded advisory refinement

Authoritative enum: `NO_ADJUST`, `ADJUST`, `CLARIFY`.

- decision accuracy 1.000;
- increase F1 0.950;
- decrease F1 0.633;
- preserve F1 0.818;
- direction macro F1 0.801;
- strength accuracy 0.950;
- approval accuracy 1.000;
- clarification accuracy 1.000;
- risk F1 0.870;
- bound violations 0;
- mean L1 weight distance 0.059.

The LLM never writes unrestricted final weights.

## E4 — Evidence extraction

- 40 author-created packets;
- 160 atomic claims.

Results:

- precision 0.915;
- recall 0.944;
- F1 0.928;
- source grounding 0.995;
- unsupported-source hallucination 0.005;
- aspect accuracy 0.881;
- polarity 0.921;
- temporal status 0.834;
- evidence role 0.748.

Evidence-role classification is the main weakness. Source grounding is checked
after parsing rather than guaranteed by the Pydantic schema.

## E4b — Evidence robustness

15 clean reused packets plus 30 perturbed packets:

| Condition | F1 | Grounding | Unsupported hallucination |
|---|---:|---:|---:|
| Clean | 0.931 | 1.000 | 0.000 |
| Noisy | 0.928 | 1.000 | 0.000 |
| Incomplete | 0.922 | 1.000 | 0.000 |

Conflicting evidence is supported by code but was not executed.

## E5 — Explanation/governance

- 24 controlled cases.

Results:

- order preservation 1.000;
- top-1 preservation 1.000;
- valid supplier IDs 1.000;
- citation validity 1.000;
- unsupported citation 0;
- priority F1 0.789;
- governance flag F1 0.950;
- human-review accuracy 1.000;
- forbidden-term absence 1.000.

The final v2 definition treats `human_review_required` as escalated review beyond
ordinary procurement oversight.

---

# 10. Frozen benchmark schemas

### E3_INTENT_GOLD
- Path: `data/e3/e3_benchmark_50x3_gold_reviewed_v1.csv`
- Rows: 150
- Columns: `request_id, base_case_id, variant_id, category, request_text, gold_scenario_profile, gold_priority_groups_json, gold_constraints_json, gold_evidence_requests_json, gold_requires_clarification`
- JSON columns: `['gold_priority_groups_json', 'gold_constraints_json', 'gold_evidence_requests_json']`
- Enums: `{"variant_id": ["V0", "V1", "V2"], "category": ["ambiguous", "constraint_sensitive", "direct_simple", "multi_criteria", "scenario_comparison"], "gold_scenario_profile": ["BALANCED", "CLARIFICATION_REQUIRED", "CONTINUITY", "CUSTOM", "DIVERSIFICATION"]}`
- Immutable status: immutable; create a new version for corrections

### E3_ADJUDICATION
- Path: `data/e3/e3_benchmark_adjudication_v1.csv`
- Rows: 50
- Columns: `base_case_id, category, adjudication_status, gold_scenario_profile, gold_priority_groups_json, gold_constraints_json, gold_evidence_requests_json, gold_requires_clarification, adjudication_note, recommended_change`
- JSON columns: `['gold_priority_groups_json', 'gold_constraints_json', 'gold_evidence_requests_json']`
- Enums: `{"category": ["ambiguous", "constraint_sensitive", "direct_simple", "multi_criteria", "scenario_comparison"], "adjudication_status": ["ACCEPT", "ACCEPT_WITH_NOTE"], "gold_scenario_profile": ["BALANCED", "CLARIFICATION_REQUIRED", "CONTINUITY", "CUSTOM", "DIVERSIFICATION"]}`
- Immutable status: immutable; create a new version for corrections

### E3B_ADVISORY_GOLD
- Path: `data/e3b/e3b_benchmark_20_gold.csv`
- Rows: 20
- Columns: `case_id, category, user_request, initial_scenario_profile, base_weights_json, ranking_summary_json, gold_recommendation_type, gold_increase_json, gold_decrease_json, gold_preserve_json, gold_adjustment_strength, gold_requires_human_approval, gold_clarification_required, gold_risk_flags_json`
- JSON columns: `['base_weights_json', 'ranking_summary_json', 'gold_increase_json', 'gold_decrease_json', 'gold_preserve_json', 'gold_risk_flags_json']`
- Enums: `{"category": ["activity_recency_emphasis", "ambiguous_clarify", "category_emphasis", "continuity_increase", "diversification_increase", "no_adjust"], "initial_scenario_profile": ["BALANCED", "CONTINUITY", "DIVERSIFICATION"], "gold_recommendation_type": ["ADJUST", "CLARIFY", "NO_ADJUST"], "gold_adjustment_strength": ["LARGE", "MODERATE", "NONE", "SMALL"]}`
- Immutable status: immutable; create a new version for corrections

### E4_EVIDENCE_GOLD
- Path: `data/e4/e4_benchmark_40_gold.csv`
- Rows: 40
- Columns: `packet_id, supplier_name, category, evidence_packet, gold_claims_json`
- JSON columns: `['gold_claims_json']`
- Enums: `{"category": ["certification_compliance", "delivery_capacity", "financial_stability", "mixed_conflicting", "past_performance", "social_labor", "sustainability", "technical_capability"], "claim.aspect": ["CERTIFICATION", "DELIVERY_CAPACITY", "FINANCIAL_STABILITY", "GOVERNANCE_COMPLIANCE", "PAST_PERFORMANCE", "SOCIAL", "SUSTAINABILITY", "TECHNICAL_CAPABILITY"], "claim.polarity": ["MIXED", "NEGATIVE", "NEUTRAL", "POSITIVE", "UNKNOWN"], "claim.temporal_status": ["CURRENT", "EXPIRED", "FUTURE", "HISTORICAL", "UNKNOWN"], "claim.evidence_role": ["ADVISORY", "ELIGIBILITY", "FEATURE", "SCENARIO"]}`
- Immutable status: immutable; create a new version for corrections

### E5_EXPLANATION_GOLD
- Path: `data/e5/e5_benchmark_24_gold.csv`
- Rows: 24
- Columns: `case_id, category, user_request, scenario_profile, priority_groups_json, group_weights_json, supplier_context_json, evidence_claims_json, gold_top_supplier_order_json, gold_required_priority_groups_json, gold_required_governance_flags_json, gold_requires_human_review, gold_forbidden_terms_json`
- JSON columns: `['priority_groups_json', 'group_weights_json', 'supplier_context_json', 'evidence_claims_json', 'gold_top_supplier_order_json', 'gold_required_priority_groups_json', 'gold_required_governance_flags_json', 'gold_forbidden_terms_json']`
- Enums: `{"category": ["close_margin", "constraint_sensitive", "continuity", "diversification", "missing_evidence", "negative_evidence", "standard_balanced"], "scenario_profile": ["BALANCED", "CONTINUITY", "CUSTOM", "DIVERSIFICATION"]}`
- Immutable status: immutable; create a new version for corrections


---

# 11. Canonical code and output map

## Preprocessing

```text
scripts/build_can_award_tables.py
scripts/audit_supplier_resolution.py
scripts/build_supplier_dimension_2015_2016.py
scripts/build_cn_targets_2017.py
scripts/map_2017_winners_to_history.py
scripts/build_candidate_pools_2017.py
scripts/build_supplier_feature_store_2015_2016.py
scripts/build_e1b_procurement_fit_indices.py
scripts/build_extended_procurement_fit_indices.py
```

## Ranking/scenario

```text
scripts/run_e1_deterministic_rankings.py
scripts/run_e1b_nested_procurement_fit_rankings.py
scripts/run_e1x_extended_procurement_fit_rankings_v2.py
scripts/run_e2_scenario_sensitivity.py
scripts/run_e2x_scenario_sensitivity.py
scripts/run_r22_e1c_feature_ablation.py
scripts/run_r22_e1d_proxy_subgroup_sensitivity.py
scripts/run_r22_e1e_traditional_ml_baselines.py
scripts/run_r22_e2b_bootstrap_robustness.py
scripts/run_r22_e2c_profit_leadtime_esg_tradeoffs.py
```

## LLM modules

```text
scripts/run_e3_intent_scenario_openai_v5.py
scripts/run_e3b_advisory_reweighting_openai_v2.py
scripts/run_e4_evidence_extraction_openai_v2.py
scripts/run_e5_explanation_governance_openai_v2.py
scripts/run_r22_e3c_prompt_schema_ablation.py
scripts/run_r22_e3d_reasoning_effort_ablation.py
scripts/run_r22_e4b_evidence_robustness.py
```

Use `experiment_status.csv` and the full audit for canonical output/report
paths. Never assume the oldest versioned runner or report is authoritative.

---

# 12. Known repository and reproducibility exceptions

1. **CPV4 scope mismatch:** preprocessing coverage exists; E1 CPV4 ranking does not.
2. **Inactive buyer recency:** field mismatch `DT_AWARD` vs `DT_AWARD_CLEAN`.
3. **Core API safety:** core E3/E3b/E4/E5 runners do not all require explicit `--run-api`.
4. **Raw API retention:** parsed outputs and response IDs exist; full raw payloads do not.
5. **E1e designation:** executed paper-scale run but final/pilot wording is ambiguous.
6. **E4b conflict:** prepared only, not executed.
7. **Resilience:** conceptual/illustrative only, not experimental evidence.
8. **Dependencies:** unpinned; Python observed in audit was 3.13.9, while code should remain compatible with the intended environment.
9. **Repository provenance:** Git history unavailable during audit.
10. **Portability:** two profiling DuckDB catalogs contain stale absolute paths.
11. **Benchmark review provenance:** strongest for E3; separate adjudication artifacts are incomplete for E3b/E4/E5.
12. **Duplicates/superseded files:** do not delete or consolidate without a versioned provenance plan.

---

# 13. Reviewer 2.2 status

Substantially addressed:

- feature ablation;
- MCDM comparison;
- bootstrap uncertainty;
- comparison-set sensitivity for executed pools;
- scenario trade-offs;
- prompt/schema/reasoning ablation;
- noisy and incomplete evidence;
- subgroup/proxy sensitivity;
- controlled ML baseline execution;
- controlled profit–lead-time–ESG execution.

Still incomplete or qualified:

- E4b conflicting evidence;
- E1 CPV4 ranking sensitivity;
- E1e final designation/standalone validation;
- repaired buyer-recency feature and downstream revalidation;
- disruption/resilience experiment.

Use “substantially but not fully addressed.”

---

# 14. Scientific claim boundaries

Supported:

- large-scale real public procurement applicability;
- retrospective reference-supplier positioning;
- deterministic ranking and feature contributions;
- scenario responsiveness and stability;
- bootstrap uncertainty;
- bounded LLM intent, evidence, advisory, explanation, and governance behavior;
- controlled cross-modal trade-off behavior.

Not supported:

- optimal or correct supplier identification from TED;
- actual bidder-pool reconstruction;
- field validation or live organizational deployment;
- causal cost, delivery, ESG, resilience, or recovery improvement;
- CPV4 E1 ranking sensitivity;
- active buyer-recency contribution in saved E1x;
- completed conflicting-evidence robustness;
- complete fairness conclusions;
- exact local replay of raw API responses.

---

# 15. Immutable artifacts and versioning rules

Never overwrite:

```text
data/e3/e3_benchmark_50x3_gold_reviewed_v1.csv
data/e3/e3_benchmark_adjudication_v1.csv
data/e3b/e3b_benchmark_20_gold.csv
data/e4/e4_benchmark_40_gold.csv
data/e5/e5_benchmark_24_gold.csv
completed result directories
completed run metadata/manifests
```

A scientific change requires:

- new versioned script or clearly versioned configuration;
- new output directory;
- new report directory;
- input hashes;
- run metadata;
- validation checks;
- explicit comparison to frozen results;
- no silent manuscript-number replacement.

---

# 16. Coding and evaluation workflow

Before implementation, document:

1. task understanding;
2. Study A, B, or C;
3. files to inspect;
4. files to create;
5. files that remain untouched;
6. validation plan;
7. API requirement;
8. whether a frozen experiment changes;
9. expected scientific claim after the task;
10. residual limitation.

For API work:

- no API call without explicit researcher approval and `--run-api`;
- use a new output tree;
- verify benchmark hash;
- preserve complete raw payloads for future runs where policy permits;
- checkpoint and resume;
- inspect a pilot before final execution.

For deterministic work:

- use saved inputs where possible;
- add a no-cost preflight/self-test;
- preserve temporal rules and rank semantics;
- compare new outputs to frozen baselines;
- write validation CSV and Markdown report.

---

# 17. Priority fixes

1. Create a versioned status correction reflecting current audited experiment status.
2. Add validation-only E1 pool-scope logic that marks CPV4 `NOT_EXECUTED` rather than failed coverage.
3. Design and approve a versioned `DT_AWARD_CLEAN` buyer-recency correction; identify every downstream result that would change before rerunning.
4. Resolve E1e designation and add standalone validation.
5. Add explicit API gates and raw-response retention strategy before any future API execution.
6. Execute E4b conflicting only after approval in a new output tree.
7. Keep disruption/resilience conceptual unless a real controlled experiment is built.

