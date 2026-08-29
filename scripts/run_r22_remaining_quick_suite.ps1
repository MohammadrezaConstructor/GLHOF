# R2.2 remaining-concerns sequence.
# Run from the project root after copying these files into scripts/.

$Python = "python"

# 1. No API calls; usually the quickest substantive addition.
& $Python "scripts/run_r22_e1d_proxy_subgroup_sensitivity.py" `
  --root . `
  --bootstrap 300
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 2. Preflight the ML pipeline on one actual case; no model fitting.
& $Python "scripts/run_r22_e1e_traditional_ml_baselines.py" `
  --root . `
  --check-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3. Prepare a small cross-modal benchmark; no API calls.
& $Python "scripts/run_r22_e2c_profit_leadtime_esg_tradeoffs.py" `
  --root . `
  --cases 6 `
  --suppliers-per-case 6
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Deterministic analyses/preflights completed."
Write-Host "Review outputs before enabling API calls."

# API experiments (uncomment deliberately):

# About 20 new calls, reusing E3c checkpoints.
# & $Python "scripts/run_r22_e3d_reasoning_effort_ablation.py" `
#   --root . --base-cases 20 --run-api --resume

# About 15 new calls if noisy/incomplete checkpoints already exist.
# & $Python "scripts/run_r22_e4b_evidence_robustness.py" `
#   --root . --packets 15 --include-conflicting --run-api --resume

# 36-call cross-modal smoke test.
# & $Python "scripts/run_r22_e2c_profit_leadtime_esg_tradeoffs.py" `
#   --root . --cases 6 --suppliers-per-case 6 --run-api --resume

# ML pilot; no API calls but may take substantial local compute time.
# & $Python "scripts/run_r22_e1e_traditional_ml_baselines.py" `
#   --root . --max-train-cases 1500 --max-validation-cases 300 `
#   --max-test-cases 500 --random-negatives 20 --hard-negatives 5
