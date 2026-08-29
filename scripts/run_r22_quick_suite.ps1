# Reviewer 2.2 experiment sequence
# Run from the project root shown in the supplied folder structure.
# The scripts use exact project-relative defaults, so only --root . is required.

$Python = "python"

# 0. Verify every required path, schema, dependency, and runner marker.
& $Python "scripts/run_r22_preflight.py" --root .
if ($LASTEXITCODE -ne 0) { throw "R2.2 preflight failed." }

# 1. Deterministic feature-group ablation.
# This performs one E1x-style pass with all ablation models together.
& $Python "scripts/run_r22_e1c_feature_ablation.py" --root .
if ($LASTEXITCODE -ne 0) { throw "E1c ablation failed." }

# 2. No-API bootstrap and comparison-set robustness.
& $Python "scripts/run_r22_e2b_bootstrap_robustness.py" `
  --root . `
  --bootstrap 300
if ($LASTEXITCODE -ne 0) { throw "E2b robustness analysis failed." }

# 3. Prepare E3c pilot files only. No API calls yet.
& $Python "scripts/run_r22_e3c_prompt_schema_ablation.py" `
  --root . `
  --base-cases 20
if ($LASTEXITCODE -ne 0) { throw "E3c preparation failed." }

# After reviewing the generated manifest and subset, run the E3c pilot.
# The existing strict zero-shot baseline is reused.
# New API calls: 20 cases x 2 new configurations = 40 calls.
#
# & $Python "scripts/run_r22_e3c_prompt_schema_ablation.py" `
#   --root . `
#   --base-cases 20 `
#   --run-api `
#   --resume

# 4. Prepare E4b pilot perturbations only. No API calls yet.
& $Python "scripts/run_r22_e4b_evidence_robustness.py" `
  --root . `
  --packets 15
if ($LASTEXITCODE -ne 0) { throw "E4b preparation failed." }

# After reviewing the perturbation benchmark, run the E4b pilot.
# Existing clean metrics are reused.
# New API calls: 15 packets x 2 perturbations = 30 calls.
#
# & $Python "scripts/run_r22_e4b_evidence_robustness.py" `
#   --root . `
#   --packets 15 `
#   --run-api `
#   --resume

Write-Host "Deterministic R2.2 analyses completed; E3c/E4b pilot files prepared."
