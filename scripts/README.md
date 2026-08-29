# R2.2 Remaining-Concerns Experiment Suite

This package addresses the parts of Reviewer 2.2 that are not fully covered by
E1c, E2b, E3c, and E4b.

## What is already complete

- E1c: leave-one-feature-group-out and single-group ablations.
- E2b: bootstrap confidence intervals and comparison-set sensitivity.
- E3c: zero-shot/few-shot and strict/relaxed-schema ablation on 20 cases.
- E4b: clean/noisy/incomplete evidence robustness on 15 packets.
- E1: Frequency, WSM, TOPSIS, and VIKOR comparisons.
- E2: balanced, continuity, and diversification multi-criteria scenarios.

## What this package adds

### E1d: historical-proxy and observability sensitivity

`run_r22_e1d_proxy_subgroup_sensitivity.py`

No API calls. It evaluates the full model and selected ablations separately for:

- prior buyer relationship versus no prior buyer relationship;
- low versus high historical visibility;
- domestic versus cross-border reference suppliers;
- comparison-set size quartiles.

This is a historical-proxy sensitivity analysis, not a complete fairness audit.

### E1e: traditional-ML ranking baselines

`run_r22_e1e_traditional_ml_baselines.py`

No API calls. It compares:

- frozen M3 TOPSIS;
- regularized logistic regression;
- histogram gradient boosting.

The split is strictly temporal by target dispatch date. Models are trained using
bounded negative sampling, but validation and test ranking use complete
historically active CPV2_MIN1 comparison sets.

The supervised target is the historical reference outcome, not a normatively
optimal supplier.

### E3d: reasoning-effort ablation

`run_r22_e3d_reasoning_effort_ablation.py`

Adds strict-few-shot `reasoning_effort=none` to the existing E3c comparison. With
20 cases and checkpoint reuse, this should require about 20 new API calls.

### E2c: profit--lead-time--ESG cross-modal trade-offs

`run_r22_e2c_profit_leadtime_esg_tradeoffs.py`

Creates a controlled benchmark combining:

- profit per unit;
- lead time;
- source-linked ESG evidence;
- deterministic policy mapping;
- deterministic TOPSIS ranking.

The default 12 cases x 6 suppliers requires 72 extraction calls. A six-case
smoke test requires 36 calls.

The script can use:

1. seeded author-controlled structured metrics; or
2. an optional CSV containing real/precomputed structured supplier metrics with
   columns:
   `case_id,supplier_id,supplier_name,profit_per_unit,lead_time_days`.

In both modes, ESG packets are controlled so that extraction accuracy and
ranking preservation can be measured exactly.

### Conflicting evidence

The existing verified E4b script already supports a conflicting-evidence
condition. Running it on the same 15 packets with `--include-conflicting` and
`--resume` should add only the new conflict packets.

## Installation

```powershell
pip install numpy pandas pyarrow scikit-learn joblib openai pydantic
```

## Recommended time-conscious order

### 1. E1d proxy sensitivity

```powershell
python scripts/run_r22_e1d_proxy_subgroup_sensitivity.py `
  --root . `
  --bootstrap 300
```

### 2. E3d reasoning-effort ablation

Prepare/check without calls:

```powershell
python scripts/run_r22_e3d_reasoning_effort_ablation.py `
  --root . `
  --base-cases 20
```

Execute with checkpoint reuse:

```powershell
python scripts/run_r22_e3d_reasoning_effort_ablation.py `
  --root . `
  --base-cases 20 `
  --run-api `
  --resume
```

### 3. Add conflicting evidence to E4b

```powershell
python scripts/run_r22_e4b_evidence_robustness.py `
  --root . `
  --packets 15 `
  --include-conflicting `
  --run-api `
  --resume
```

### 4. Traditional-ML pilot

Preflight one real case:

```powershell
python scripts/run_r22_e1e_traditional_ml_baselines.py `
  --root . `
  --check-only
```

Pilot:

```powershell
python scripts/run_r22_e1e_traditional_ml_baselines.py `
  --root . `
  --max-train-cases 1500 `
  --max-validation-cases 300 `
  --max-test-cases 500 `
  --random-negatives 20 `
  --hard-negatives 5
```

Larger paper run after the pilot is validated:

```powershell
python scripts/run_r22_e1e_traditional_ml_baselines.py `
  --root . `
  --max-train-cases 6000 `
  --max-validation-cases 1000 `
  --max-test-cases 2000 `
  --random-negatives 25 `
  --hard-negatives 5
```

### 5. Cross-modal trade-off smoke test

Prepare only:

```powershell
python scripts/run_r22_e2c_profit_leadtime_esg_tradeoffs.py `
  --root . `
  --cases 6 `
  --suppliers-per-case 6
```

Execute 36 extraction calls:

```powershell
python scripts/run_r22_e2c_profit_leadtime_esg_tradeoffs.py `
  --root . `
  --cases 6 `
  --suppliers-per-case 6 `
  --run-api `
  --resume
```

Final controlled run, 72 calls:

```powershell
python scripts/run_r22_e2c_profit_leadtime_esg_tradeoffs.py `
  --root . `
  --cases 12 `
  --suppliers-per-case 6 `
  --run-api `
  --resume
```

To use a structured supplier-metric CSV:

```powershell
python scripts/run_r22_e2c_profit_leadtime_esg_tradeoffs.py `
  --root . `
  --structured-input data/e2c/my_structured_supplier_metrics.csv `
  --cases 12 `
  --suppliers-per-case 6 `
  --run-api `
  --resume
```

## Reporting rules

- Do not call observed awardees optimal suppliers.
- Use "historical reference outcome".
- Use "historically active supplier comparison set", not bidder pool.
- E1d is not a complete fairness audit.
- ML baselines predict historical awards and may reproduce historical patterns.
- E2c is a controlled cross-modal benchmark, not a real public procurement
  applicability result.
- Do not report pilot runs as final experiments.
- Freeze prompts, mappings, and model settings before final API runs.
