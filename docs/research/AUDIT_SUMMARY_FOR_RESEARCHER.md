# Audit Summary for the Researcher

## Bottom line

The repository is scientifically coherent and the main saved results are real,
not empty placeholders. The deterministic ranking pipeline respects the intended
2015-2016 history versus 2017 target split. The controlled LLM studies have
complete parsed outputs, checkpoints, metadata, and matching benchmark hashes.

The overall verdict is **COMPLETE WITH DOCUMENTED EXCEPTIONS**. The exceptions
must be addressed or clearly disclosed before manuscript revision.

## Is the repository organized correctly?

Mostly. The main flow from raw TED files to frozen supplier history, 2017 targets,
comparison sets, features, rankings, and reports is understandable and supported
by code. LLM benchmarks are separated from prediction outputs.

Organization is weakened by duplicate benchmark copies, several superseded run
directories, duplicate E1c outputs, low-context scratch files, absolute paths in
manifests, and two DuckDB profiling files that still point to another user's old
desktop path. These are provenance and maintenance problems, not evidence that
the primary results are fabricated.

## Can the experiments be reproduced?

The deterministic results are substantially reproducible from saved data. A
small independent E1 recomputation matched all 12 saved method-case rank checks
exactly. E1x and E2x validations reproduce their baselines, and bootstrap seeds
are recorded.

Exact environment reproduction is weaker because dependencies are not pinned and
Git history is inaccessible. Exact LLM raw-response reproduction is impossible
from local files because only parsed predictions and response IDs were saved;
response storage was disabled. The model, reasoning effort, token limit,
benchmark hash, prompt, schema, usage, and timing are preserved.

No API calls are needed to inspect the saved results. Some report-generation
code is coupled to API runners, so a clean offline report-only workflow would
improve reproducibility.

## Which experiments are truly complete?

Validated or strongly validated:

- TED preprocessing and frozen temporal data construction;
- E1b, E1x v2, E1c, E2, and E2x;
- E3 final intent benchmark;
- E3b v2 advisory benchmark;
- E4 final evidence benchmark;
- E5 v2 explanation benchmark;
- E4b clean baseline as a reused validated E4 subset.

Report-backed final executions without a uniform standalone validation manifest:

- E1 refined, E1d, E2b, and E2c;
- E3c prompt/schema ablation;
- E3d reasoning effort, stored inside E3c outputs;
- E4b noisy and incomplete conditions.

E1 is complete for its three executed pools, but carries a validation warning
because the validator also expects an unexecuted CPV4 result.

## Which experiments are only pilots or ambiguous?

- E3 smoke and 15-request pilot outputs are pilots/superseded by E3 final.
- E4 smoke is a pilot.
- E3b v1 and E5 v1 are superseded by v2.
- E1e is not merely the script's small default pilot: saved metadata shows a
  6,000/1,000/1,399 paper-scale chronological run. However, the report still
  contains pilot/future-final language, so its formal citation status is
  ambiguous until the authors designate and validate it.
- E4b conflicting is prepared in code but not run.
- The disruption/resilience demonstration has not started.

## Which results should not yet be cited?

- Any E1 CPV4 candidate-pool sensitivity.
- E4b conflicting evidence.
- A disruption/resilience result.
- PRIOR_BUYER_RECENCY as an active E1x feature.
- E1e as an unqualified final baseline until its status is resolved.
- Superseded E3b v1 or E5 v1 metrics.
- Claims that the TED comparison sets are actual bidder pools.
- Claims that the observed awardee is optimal/correct/true best.
- Claims of field validation or improved real-world procurement outcomes.
- Fairness conclusions from E1d.

## What needs correction before manuscript revision?

1. Correct the handoff: E3b uses NO_ADJUST, E1d and E2c are executed, and E1e is
   executed with a designation warning.
2. State clearly that CPV4 preprocessing coverage exists but E1 CPV4 ranking was
   not run.
3. Resolve E1e's final/pilot wording and add a validation manifest.
4. Disclose that prior-buyer recency was inactive.
5. Keep E4b conflicting and disruption work listed as pending.
6. Use only historical reference supplier/outcome and historically active
   comparison-set terminology for TED results.
7. Add manuscript files to a future read-only audit so every cited number can be
   checked.
8. Before any new API run, add explicit authorization gates to the core runners
   and decide how full raw responses will be retained.

## Safest next step

Start with a non-API, validation-only E1 change in a new version: make the
validator compare reported pool sensitivities with the runner's configured
`POOL_SPECS` and explicitly label CPV4 as not executed. This is small, uses only
saved outputs, and resolves the clearest result-scope ambiguity without rerunning
an experiment or touching frozen artifacts.

The detailed evidence is in `REPOSITORY_AUDIT.md`; machine-readable
findings are in the CSV files beside it.

