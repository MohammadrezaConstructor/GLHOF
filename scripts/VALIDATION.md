# Validation status

Completed in an isolated validation environment:

- All four Python files pass syntax compilation.
- E2c benchmark preparation was executed on a temporary project fixture.
- E2c ranking evaluation was tested with perfect synthetic extraction outputs;
  top-1 agreement, top-3 overlap, rank correlation, and eligibility accuracy all
  equalled 1.0 as expected.
- E1e temporal splitting and negative-selection helpers were exercised on
  synthetic case data.
- E1d metric, subgroup, and bootstrap logic was inspected against the actual
  project column names found in the source scripts.

Not executed in an isolated validation environment:

- E1d against the local parquet files.
- E1e full feature loading, model fitting, or complete-set ranking.
- Any OpenAI API calls.
- E2c extraction using the live E4 runner.

Each script performs local path/schema checks before substantive execution.
