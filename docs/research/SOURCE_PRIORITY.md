# Source priority and conflict resolution

Use this order when facts conflict:

1. Actual current code, manifests, validation files, saved CSV/Parquet outputs,
   benchmark hashes, and run metadata.
2. `reports/repository_audit/REPOSITORY_AUDIT.md` and its appendices.
3. `PROJECT_CONTEXT.md` and `PROJECT_FACTS.json`.
4. Older `EVALUATION_GUIDE.md`, consolidated evaluation reports, and
   manuscript prose only when not contradicted by newer evidence.

Known stale statements in older documents:

- E1d is not merely pending; report-backed outputs exist.
- E2c is not merely pending; controlled results exist.
- E1e is executed but its final/pilot designation is unresolved.
- E3b uses `NO_ADJUST`, not `KEEP`.
- CPV4 preprocessing exists, but E1 CPV4 ranking was not executed.
- `PRIOR_BUYER_RECENCY` was inactive in the saved E1x run.
- E4b conflicting evidence and disruption/resilience remain incomplete.
