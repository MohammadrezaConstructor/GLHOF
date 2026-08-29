# Full Repository Audit

Generated: 2026-07-15

## 1. Executive verdict

**COMPLETE WITH DOCUMENTED EXCEPTIONS.**

The repository contains a coherent, mostly reproducible governed supplier-selection
research pipeline. The central scientific separation is implemented: frozen
2015-2016 public-procurement history supplies identity, eligibility, and supplier
features; 2017 supplies target procurements and historical reference outcomes;
deterministic code computes TED rankings; LLM modules perform bounded intent,
advisory, evidence, and explanation tasks.

No inspected historical feature path contains a 2017 winner/rank/outcome field.
Independent E1 rank/metric recomputation matched 12 of 12 method-case checks
exactly. E1x and E2x validations pass. Final E3/E3b/E4/E5 benchmark hashes match
their run metadata and saved metrics match reports.

The exceptions are material but bounded. E1's CPV4 validation fails because CPV4
was preprocessed but never configured as an E1 ranking pool. Four core LLM
runners lack an explicit `--run-api` gate, although the R2.2 wrappers have one.
Full raw API response payloads were not retained. Git history is inaccessible,
dependencies are unpinned, two profiling DuckDB catalogs contain broken absolute
paths, and manual-review provenance is incomplete for E3b/E4/E5. E1e's saved run
is paper-scale but its final designation is ambiguous. E4b conflicting evidence
was not executed.

## 2. Audit scope and read-only confirmation

The audit inspected the repository from actual code, schemas, manifests,
checkpoints, outputs, and reports. Handoff documents were treated as claims to
verify, not authoritative evidence.

- 541 files were inventoried, excluding `.git`, environment/transient paths, and
  the audit output directory itself.
- 48 Python files were AST-inspected; all parsed successfully.
- 437 structured files were attempted: 435 parsed/inspected and two DuckDB
  catalogs were recorded as not verifiable because their external views point to
  a missing former absolute path.
- Final/manual benchmarks were fully scanned; large source files were inspected
  by schema/metadata and bounded samples unless a targeted full check was needed.
- Three E1 cases across three executed pool definitions and four methods were
  recomputed in memory only.
- The saved R2.2 preflight was read but not rerun. No completed experiment was
  rerun.

Audit writes were limited to `reports/repository_audit/` and the permitted helper
`tools/repository_audit.py`. No API call was made, no API key was used,
and no production script, benchmark, data, result, or pre-existing report was
modified.

## 3. Complete repository structure

The helper's inventory records 541 files by category:

| Category | Files |
|---|---:|
| report | 334 |
| analysis output | 81 |
| code | 50 |
| processed data | 38 |
| benchmark | 15 |
| checkpoint | 15 |
| raw data | 4 |
| config | 2 |
| log | 1 |
| unknown | 1 |

Extensions are 315 CSV, 76 Parquet, 48 Python, 45 Markdown, 21 JSON, 15 JSONL,
10 DuckDB, 7 TXT, 2 joblib, and 2 PowerShell. Large assets include approximately
3.97 GB `_can_build.duckdb`, 1.17 GB `_supplier_dimension_build.duckdb`, and raw
CAN/CN CSVs between roughly 285 and 435 MB. Files over the configured 100 MB hash
limit are marked `HASH_SKIPPED_LARGE_FILE` with byte size.

There are 37 duplicate-hash groups. Important duplicates include E3 adjudication
under both `data/e3/` and `data/raw/`, the E4 template under both `data/e3/` and
`data/e4/`, E5 final gold copied into `scripts/`, a v3 E3 runner copied into
`data/raw/`, many profiling report generations, and duplicate E1c result/report
names. The inventory identifies seven orphan candidates, but that is a lexical
reference heuristic: documentation and serialized models may be valid even when
no source file names them directly. Low-context files `aa.py`, `aa2.py`,
`scripts/code.py`, and root scratch TXT files need owner classification.

Files/results referred to conceptually but absent are an E1 CPV4 ranking result,
E4b conflicting-condition output, a disruption-response experiment, and any
manuscript source/PDF. The static I/O graph contains many unresolved dynamic path
expressions; these are not automatically treated as missing files.

Evidence: `repository_file_inventory.csv`, `python_script_inventory.csv`,
`data_schema_inventory.csv`, and `repository_dependency_graph.md`.

## 4. Verified data lineage

The reconstructed code path is:

```text
raw CAN 2015-2017 -> build_can_award_tables.py -> normalized award facts
normalized award facts -> audit_supplier_resolution.py -> identity audit
2015-2016 award facts -> build_supplier_dimension_2015_2016.py -> frozen dimension
raw CN 2017 -> build_cn_targets_2017.py -> 2017 target cases
2017 targets + frozen dimension -> map_2017_winners_to_history.py -> reference mapping
frozen history + targets + reference mapping -> build_candidate_pools_2017.py
frozen history + separate targets -> build_supplier_feature_store_2015_2016.py
candidate pools/features -> E1b/E1x index builders -> deterministic ranking/scenarios
controlled gold CSVs -> E3/E3b/E4/E5 runners -> parsed outputs/metrics/reports
```

`build_supplier_dimension_2015_2016.py::validate_input` and its SQL require source
years 2015 and 2016. `map_2017_winners_to_history.py::validate_historical_inputs`
maps 2017 awardee strings into frozen lookups and does not learn 2017 aliases.
`build_candidate_pools_2017.py` uses historical context counts. The reference
supplier is used to decide evaluability after the pool is built; it is not used
to create candidate features.

The 2017 CAN enrichment in winner mapping provides target-winner town/descriptive
fields after the dimension is frozen. This is not identity learning. The output
is a retrospective historical reference map, not evidence that an observed
awardee was uniquely optimal.

## 5. Temporal leakage assessment

**Assessment: PASS with two warnings and one provenance limitation.**

Historical publication dates end 2016-12-30. Target dispatch dates run from
2017-01-01 through 2017-12-13, and saved strict history-before-target checks are
true. Historical supplier schemas have no target, winner, reference-rank, or
outcome columns. Comparison-set eligibility comes from frozen historical
activity.

E1e uses a chronological date split: train through 2017-07-11, validation
2017-07-12 to 2017-09-11, and test 2017-09-12 to 2017-11-24. Labels identify the
historical reference supplier only in training; frozen fit variables and target
attributes are the features. Hard negatives are deterministic TOPSIS candidates,
not future-outcome features.

Warnings:

1. Some award-date values inside 2015/2016 source-year records parse after 2016,
   including 2018/2020. The pipeline's temporal anchors use source year and
   publication date, so this does not create observed leakage, but it requires
   documentation.
2. `build_extended_procurement_fit_indices.py` reads `DT_AWARD` while normalized
   award facts expose `DT_AWARD_CLEAN`; `LAST_PRIOR_BUYER_AWARD_DATE` is null in
   all 488,734 extended-index rows. The feature mask makes
   PRIOR_BUYER_RECENCY inactive, avoiding leakage but defeating that intended
   signal.

Git history is unavailable, so historical freeze chronology beyond current
content hashes is not independently verifiable. Full findings are in
`temporal_leakage_audit.csv`.

## 6. Deterministic ranking mechanism

E1 defines four benefit features: log context experience, log observed buyer
breadth, reciprocal publication recency, and context specialization. Frequency
uses raw context count. WSM uses casewise min-max normalization and equal weights.
TOPSIS uses vector normalization, equal weights, benefit ideals, and closeness to
the worst/best distances. VIKOR uses equal-weight regret with `v=0.5`, lower Q
better.

Scores are rounded to 12 decimals before ties. Saved outputs include best rank,
worst rank, midrank, and tie-group size. W@1/5/10/50 uses midrank thresholds; MRR
is mean `1/midrank`; percentile is
`1-(midrank-1)/(pool_size-1)`; median rank is median midrank.

E1x vector-normalizes active features and aggregates equal within-group
contributions into grouped TOPSIS. M3 uses CATEGORY_FIT, GEOGRAPHIC_FIT,
BUYER_RELATIONSHIP, and ACTIVITY_PROFILE. E2x changes only group weights:
BALANCED .25/.25/.25/.25, CONTINUITY .25/.20/.40/.15, and DIVERSIFICATION
.30/.35/.10/.25. Weights are renormalized over active groups. BALANCED exactly
reproduces E1x M3.

E2b and E1d use case-level bootstrap sampling, 300 resamples, seed 42 plus a
deterministic group offset, and 2.5/97.5 percentile intervals. Scenario movement
includes correlations, absolute movement, direction percentages, and top-k
overlap against BALANCED.

Independent in-memory recomputation exactly matched saved rank/tie/metric fields
for 12 E1 method-case combinations. Candidate-level scores are not saved, so
score-file equality is not verifiable. See `deterministic_mechanism_audit.md` and
`metric_definition_inventory.csv`.

## 7. LLM module mechanism and governance

M_I (`run_e3_intent_scenario_openai_v5.py`) emits strict structured intent,
scenario, priorities, constraints, evidence requests, clarification, and
rationale. Its prompt forbids supplier ranking, TOPSIS, numerical weights, and
outcome inference.

M_W (`run_e3b_advisory_reweighting_openai_v2.py`) emits NO_ADJUST, ADJUST, or
CLARIFY; qualitative group directions; coarse strength; approval; risks; and
rationale. It cannot emit final numbers. A deterministic controller applies
0/.05/.10/.15 deltas, per-group bounds .10-.45, and a unit-sum constraint.

M_E (`run_e4_evidence_extraction_openai_v2.py`) emits atomic typed claims and a
required source quote. The prompt demands packet grounding. Grounding is checked
after parsing rather than hard-enforced by Pydantic; final measured grounding is
.995, not 1.0.

M_X (`run_e5_explanation_governance_openai_v2.py`) explains a supplied order. The
prompt forbids reranking, new scores/weights, unknown suppliers/claim IDs, and
certainty language. Evaluation checks order, top one, IDs, citations, flags,
review, and forbidden terms. Its outputs do not feed the ranking engine.

All current core runners use structured parsing, up to five total attempts, and
no semantic fallback. Exhausted failures become checkpoint ERROR rows. Existing
checkpoints require `--resume` or a new directory. R2.2 wrappers require
`--run-api`, but core E3/E3b/E4/E5 runners do not. Final local artifacts preserve
parsed JSON and response IDs, not full raw response payloads. See
`llm_module_audit.md` and `llm_runner_inventory.csv`.

## 8. Manual benchmark creation and review workflow

E3 final has 150 unique requests, 50 base cases, and V0/V1/V2. No base case has
variant-inconsistent gold. Its 50-row adjudication file joins exactly to final
gold fields (41 ACCEPT, 9 ACCEPT_WITH_NOTE), and the 15-row pilot is an exact
subset. This provides a visible manual-review chain.

E3b has 20 unique cases and valid controlled enums. E4 has 40 unique packets,
160 gold claims, no duplicate claim IDs within packets, and 160/160 grounded
gold quotes. E5 has 24 unique cases and 24/24 gold orders match deterministic
rank order. Each authoritative current benchmark hash exactly equals its final
run metadata hash.

E3b, E4, and E5 lack separate review/adjudication artifacts, so their manual
creation path is only partially verifiable. Benchmark/prediction directories are
properly separated, but duplicate benchmark copies create canonical-path risk.
See `benchmark_version_lineage.csv`, `benchmark_integrity_audit.csv`, and
`manual_review_workflow.md`.

## 9. Experiment provenance and completion status

The audit identifies 38 experiments/sub-experiments. Authoritative statuses are
in `experiment_provenance_matrix.csv`.

### Validated

Preprocessing/frozen construction, E1b, E1x v2, E1c, E2, E2x, E3 final, E3b v2,
E4 final, E4b clean reused baseline, and E5 v2 have saved outputs plus explicit
validation evidence or complete benchmark/metadata checks.

### Final executed, not uniformly validated

E1 refined, E1d, E2b, E2c, E3c, E3d (inside E3c), and E4b noisy/incomplete are
report-backed final executions but lack a uniform standalone validation manifest.

### Validation warnings

- E1 has one CPV4 validation failure; executed CPV2/CPV3 results remain intact.
- The extended index is valid but prior-buyer recency is inactive.
- E1e is a 6,000/1,000/1,399 chronological paper-scale execution, but final
  designation is ambiguous because its report retains pilot language.

### Pilot/superseded/prepared

E3 smoke and pilots, E4 smoke, E3b v1, E5 v1, and earlier runner versions are not
authoritative final results. E4b conflicting is PREPARED only. No dedicated
disruption/resilience result exists: NOT_STARTED.

## 10. Report/manuscript consistency

Saved CSV and Markdown numbers agree for primary E1, E1x, E1c, E2x, E2b, E1d,
E1e, E2c, E3, E3c, E3b, E4/E4b, and E5 metrics, allowing ordinary display
rounding. No hidden headline-number drift was detected.

Status/claim mismatches remain:

- `EVALUATION_GUIDE.md` says KEEP; authoritative E3b is NO_ADJUST.
- Handoff lines 1341-1343 call E1d/E1e/E2c pending. E1d and E2c are executed;
  E1e is executed with a designation warning.
- The saved 40/40 R2.2 preflight must not be generalized to all historical
  validations because E1 retains one failed check.
- E1e report prose does not cleanly describe its saved paper-scale settings.
- E4b report/code support for conflicting evidence is prospective, not an
  executed result.

No manuscript source or PDF was found, so manuscript-number and wording
consistency is NOT_VERIFIABLE. See `report_number_consistency.csv` and
`terminology_and_claim_audit.csv`.

## 11. Confirmed discrepancies

1. **E1 CPV4:** preprocessing expected 16,232 reference-included cases; E1 has no
   CPV4 pool specification or result. This is a real configuration/validation
   mismatch and absent sensitivity run.
2. **E3b enum:** NO_ADJUST is authoritative; handoff KEEP is stale.
3. **E1d:** report-backed FINAL_EXECUTED; handoff pending is stale.
4. **E1e:** paper-scale execution with VALIDATION_WARNING; handoff pending is
   stale and report final/pilot language is ambiguous.
5. **E2c:** default 12x6/72-packet controlled FINAL_EXECUTED; handoff pending is
   stale.
6. **E4b conflicting:** code support exists; manifest/results prove it was not
   executed.
7. **E3d:** reasoning-none was executed in shared E3c outputs, not a separate
   E3d tree.

Full evidence is in `targeted_discrepancy_investigation.md`.

## 12. Reproducibility risks

The risk register contains 0 critical, 2 high, 14 medium, and 9 low findings.

The high findings are core-runner API authorization and absent complete raw API
payloads. Medium findings cover CPV4, inactive buyer recency, E1e designation,
E2c stale-output reuse behavior, broken DuckDB absolute paths, unpinned
dependencies, inaccessible Git, partial benchmark review provenance, duplicate
gold copies, stale handoff claims, unexecuted E4b conflict, incomplete standalone
validation coverage, non-hard E4 grounding, and absolute manifest paths.

Python observed during the audit was 3.13.9. `scripts/requirements.txt` names
numpy, pandas, pyarrow, scikit-learn, joblib, openai, and pydantic without
versions. Deterministic scripts generally record seeds; API model metadata is
strong. Deterministic reports can be reconstructed from saved results, but some
LLM report generation is coupled to API-runner entry points and full raw outputs
cannot be reconstructed. See `reproducibility_risk_register.csv`.

## 13. Orphaned, duplicated, stale, and superseded files

- 37 duplicate-hash groups are listed through `repository_file_inventory.csv`.
- Canonical-risk duplicates: E3 adjudication, E4 template, E5 gold, and E3 v3
  runner copies outside their expected directories.
- E1c deliberately/accidentally stores byte-identical `e1x_*` and `r22_e1c_*`
  outputs.
- E3b final v1 and E5 final v1 are superseded by v2; old E3 pilot/smoke outputs
  should not be cited as final.
- Three generations of CAN profiling reports overlap heavily.
- Seven lexical orphan candidates and several low-context scratch files require
  owner classification, not automatic deletion.
- Two saved profiling DuckDB files are stale/nonportable because they reference
  `C:/Users/morezaei/Desktop/llmsr/...`.

No file should be deleted or moved without a separate approved provenance plan.

## 14. What is scientifically supported

- A real public-procurement applicability study using historically active
  supplier comparison sets and historical reference-supplier positioning.
- Deterministic Frequency, WSM, TOPSIS, VIKOR, grouped TOPSIS, ablation, scenario,
  subgroup, bootstrap, and chronological ML comparisons for their saved cohorts.
- The primary E1 CPV2_MIN1, CPV3_MIN1, and CPV2_MIN2 results; E1x M3 and E2x
  scenario results; E1c/E1d/E2b analyses under stated limitations.
- A controlled, author-created E2c cross-modal supplier-selection study.
- Controlled benchmark performance of M_I, M_W, M_E, and M_X using the exact
  saved model/settings/content hashes.
- E4b robustness for clean reused, noisy, and incomplete evidence on 15 source
  packets.

These studies answer complementary questions and should remain separated as
Study A real public procurement applicability, Study B controlled cross-modal
selection, and Study C governed LLM module evaluation.

## 15. What is not scientifically supported

- Identification of an optimal, correct, true-best, or causally superior
  supplier from the TED study.
- Claims that comparison sets are actual bidder pools.
- Field validation, validated procurement outcomes, or demonstrated real-world
  outcome improvement.
- An E1 CPV4 candidate-pool ranking sensitivity result.
- An active contribution from PRIOR_BUYER_RECENCY in the saved E1x run.
- A completed E4b conflicting-evidence result.
- A completed disruption/resilience demonstration.
- A fairness conclusion from E1d proxy subgroups.
- A manuscript claim match, because no manuscript artifact was present.
- Exact raw-response replay for final LLM runs.

## 16. Exact pending work

1. Resolve E1 CPV4 validation scope or execute a separately approved/versioned
   CPV4 sensitivity; until then, do not cite one.
2. Trace and version the prior-buyer award-date field correction; assess how it
   changes E1x before claiming recency.
3. Add explicit core-runner API authorization gates and tests before any future
   API execution.
4. Establish full raw-response retention/provenance for future API runs.
5. Issue a versioned E1e final/pilot designation and standalone validation.
6. Add hash-checked existing-output reuse to E2c.
7. Run E4b conflicting only after explicit approval and into a new output tree.
8. Create the disruption demonstration if it remains in scope.
9. Publish pinned dependencies, accessible release/version history, canonical
   benchmark paths, and review provenance.
10. Supply manuscript artifacts for a separate number/claim audit.

## 17. Recommended corrections, ordered by severity

1. **HIGH:** Gate direct E3/E3b/E4/E5 API execution with explicit authorization.
2. **HIGH:** Preserve complete raw API responses or document a compliant durable
   retrieval policy for future runs.
3. **MEDIUM:** Align E1 CPV4 validation and configured pool scope.
4. **MEDIUM:** Fix/version prior-buyer recency construction and revalidate.
5. **MEDIUM:** Resolve E1e final designation and add missing standalone
   validations.
6. **MEDIUM:** Hash-check E2c benchmark/extraction reuse before evaluation.
7. **MEDIUM:** Pin the environment and restore version-control/release evidence.
8. **MEDIUM:** Publish E3b/E4/E5 review provenance and canonical paths.
9. **MEDIUM:** Version the handoff corrections; preserve frozen result files.
10. **LOW:** Classify/archive superseded, duplicate, and scratch artifacts under
    an approved cleanup plan.

## 18. Smallest safe next coding task

Create a versioned, non-API validation-only test for E1 that derives expected
pool coverage from `POOL_SPECS`, explicitly records CPV4 as not executed, and
fails if reported sensitivities and configured pools diverge. This can be tested
against saved candidate summaries without recomputing rankings or touching
frozen outputs. Any correction to existing reports should remain a separate,
approved versioned action.

## 19. Final status

**COMPLETE WITH DOCUMENTED EXCEPTIONS**

Machine-readable appendices in this directory provide the complete inventory,
schemas, dependencies, integrity findings, temporal checks, metric definitions,
runner details, benchmark lineage, experiment provenance, consistency checks,
and risk register.

