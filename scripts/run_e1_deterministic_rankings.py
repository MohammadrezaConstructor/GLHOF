#!/usr/bin/env python3
"""
E1: Deterministic ranking comparison for the supplier-selection study.

Main purpose
------------
Evaluate a focused set of deterministic ranking methods on the compact
historical supplier feature store produced by preprocessing.

Methods
-------
1. FREQUENCY
   Rank by N_CONTEXT_AWARDS descending.

2. WSM
   Equal-weight weighted-sum model over four normalized criteria.

3. TOPSIS
   Equal-weight TOPSIS over the same four criteria.

4. VIKOR
   Equal-weight VIKOR compromise ranking over the same four criteria,
   with v = 0.5.

Core criteria
-------------
1. CONTEXT_EXPERIENCE
       log(1 + N_CONTEXT_AWARDS)                    benefit

2. OBSERVED_BUYER_BREADTH
       log(1 + N_UNIQUE_BUYERS_IN_CONTEXT)         benefit

3. PUBLIC_ACTIVITY_RECENCY
       1 / (1 + DAYS_SINCE_LAST_CONTEXT_PUBLICATION)
                                                        benefit

4. CONTEXT_SPECIALIZATION
       SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT         benefit

Why publication-date recency?
-----------------------------
The main E1 recency criterion uses LAST_CONTEXT_PUBLICATION_DATE rather than
LAST_CONTEXT_AWARD_DATE. This makes the criterion explicitly tied to public
information availability and avoids dependence on anomalous award dates.
A later sensitivity analysis may compare publication-date and award-date
recency if desired.

Pool definitions
----------------
Primary:
    CPV2_MIN1

Sensitivity:
    CPV3_MIN1
    CPV2_MIN2

Evaluation cohort
-----------------
Ranking metrics are computed only for target cases whose observed winner is
present in the corresponding candidate pool.

This intentionally separates:
    historical observability
        -> candidate retention
        -> ranking performance

Tie handling
------------
Scores are ranked using midranks. The script also records best and worst
possible ranks within each tie group.

Outputs
-------
Processed:
- e1_case_ranking_results.parquet

Reports:
- e1_method_summary.csv
- e1_results_by_cpv2.csv
- e1_results_by_country.csv
- e1_pool_sensitivity_summary.csv
- e1_feature_specification.csv
- e1_validation_checks.csv
- e1_ranking_report.md

Requires
--------
pip install numpy pandas pyarrow duckdb
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

METHODS = ("FREQUENCY", "WSM", "TOPSIS", "VIKOR")

FEATURE_NAMES = (
    "CONTEXT_EXPERIENCE",
    "OBSERVED_BUYER_BREADTH",
    "PUBLIC_ACTIVITY_RECENCY",
    "CONTEXT_SPECIALIZATION",
)

TOP_K_VALUES = (1, 5, 10, 50)


@dataclass(frozen=True)
class PoolSpec:
    name: str
    level: str
    target_context_column: str
    min_context_awards: int
    feature_path_arg: str


POOL_SPECS = (
    PoolSpec(
        name="CPV2_MIN1",
        level="CPV2",
        target_context_column="TARGET_CPV2",
        min_context_awards=1,
        feature_path_arg="cpv2_features",
    ),
    PoolSpec(
        name="CPV3_MIN1",
        level="CPV3",
        target_context_column="TARGET_CPV3",
        min_context_awards=1,
        feature_path_arg="cpv3_features",
    ),
    PoolSpec(
        name="CPV2_MIN2",
        level="CPV2",
        target_context_column="TARGET_CPV2",
        min_context_awards=2,
        feature_path_arg="cpv2_features",
    ),
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def parquet_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{sql_path(path)}')
            """
        ).fetchall()
    }


def require_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    required: Iterable[str],
    label: str,
) -> None:
    cols = parquet_columns(con, path)
    missing = sorted(set(required) - cols)
    if missing:
        raise ValueError(
            f"{label} is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def minmax_benefit(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalize columns as benefit criteria.

    A constant criterion contributes the same information for all candidates.
    It is set to zero because adding an equal constant to all alternatives
    cannot change within-pool ranks.
    """
    x = np.asarray(x, dtype=np.float64)
    mins = np.nanmin(x, axis=0)
    maxs = np.nanmax(x, axis=0)
    spans = maxs - mins

    z = np.zeros_like(x, dtype=np.float64)
    varying = spans > 0
    z[:, varying] = (
        x[:, varying] - mins[varying]
    ) / spans[varying]
    return z


def wsm_scores(x: np.ndarray) -> np.ndarray:
    z = minmax_benefit(x)
    weights = np.full(z.shape[1], 1.0 / z.shape[1], dtype=np.float64)
    return z @ weights


def topsis_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.sqrt(np.sum(np.square(x), axis=0))

    normalized = np.zeros_like(x, dtype=np.float64)
    nonzero = norms > 0
    normalized[:, nonzero] = x[:, nonzero] / norms[nonzero]

    weights = np.full(
        normalized.shape[1],
        1.0 / normalized.shape[1],
        dtype=np.float64,
    )
    weighted = normalized * weights

    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    d_best = np.sqrt(np.sum(np.square(weighted - ideal_best), axis=1))
    d_worst = np.sqrt(np.sum(np.square(weighted - ideal_worst), axis=1))

    denom = d_best + d_worst
    scores = np.full(len(x), 0.5, dtype=np.float64)
    valid = denom > 0
    scores[valid] = d_worst[valid] / denom[valid]
    return scores


def vikor_q(
    x: np.ndarray,
    v: float = 0.5,
) -> np.ndarray:
    """
    Equal-weight VIKOR Q index for benefit criteria.

    Lower Q is better.
    """
    x = np.asarray(x, dtype=np.float64)
    best = np.max(x, axis=0)
    worst = np.min(x, axis=0)
    spans = best - worst

    regret = np.zeros_like(x, dtype=np.float64)
    varying = spans > 0
    regret[:, varying] = (
        best[varying] - x[:, varying]
    ) / spans[varying]

    weights = np.full(
        regret.shape[1],
        1.0 / regret.shape[1],
        dtype=np.float64,
    )
    weighted_regret = regret * weights

    s = np.sum(weighted_regret, axis=1)
    r = np.max(weighted_regret, axis=1)

    s_best = np.min(s)
    s_worst = np.max(s)
    r_best = np.min(r)
    r_worst = np.max(r)

    s_term = np.zeros_like(s)
    r_term = np.zeros_like(r)

    if s_worst > s_best:
        s_term = (s - s_best) / (s_worst - s_best)

    if r_worst > r_best:
        r_term = (r - r_best) / (r_worst - r_best)

    return v * s_term + (1.0 - v) * r_term


def rank_with_ties(
    scores: np.ndarray,
    higher_is_better: bool,
    decimals: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return midrank, best rank, worst rank, and tie-group size.

    Scores are rounded before tie grouping to avoid tiny floating-point
    differences creating artificial tie breaks.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = np.round(s, decimals=decimals)

    key = -s if higher_is_better else s
    order = np.argsort(key, kind="mergesort")
    sorted_key = key[order]

    n = len(s)
    midrank = np.empty(n, dtype=np.float64)
    best_rank = np.empty(n, dtype=np.int64)
    worst_rank = np.empty(n, dtype=np.int64)
    tie_size = np.empty(n, dtype=np.int64)

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_key[end] == sorted_key[start]:
            end += 1

        # 1-indexed rank interval [start+1, end]
        best = start + 1
        worst = end
        mid = (best + worst) / 2.0
        size = end - start

        idx = order[start:end]
        midrank[idx] = mid
        best_rank[idx] = best
        worst_rank[idx] = worst
        tie_size[idx] = size

        start = end

    return midrank, best_rank, worst_rank, tie_size


def make_feature_matrix(
    candidates: pd.DataFrame,
    target_date: pd.Timestamp,
) -> np.ndarray:
    last_pub = pd.to_datetime(
        candidates["LAST_CONTEXT_PUBLICATION_DATE"],
        errors="raise",
    )

    days = (
        target_date.normalize() - last_pub.dt.normalize()
    ).dt.days.to_numpy(dtype=np.int64)

    if np.any(days < 0):
        bad = int(np.sum(days < 0))
        raise ValueError(
            f"Found {bad} candidate rows with historical publication dates "
            "after the target date."
        )

    experience = np.log1p(
        candidates["N_CONTEXT_AWARDS"].to_numpy(dtype=np.float64)
    )

    buyer_breadth = np.log1p(
        candidates["N_UNIQUE_BUYERS_IN_CONTEXT"]
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )

    recency = 1.0 / (1.0 + days.astype(np.float64))

    specialization = (
        candidates["SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT"]
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )

    x = np.column_stack(
        [
            experience,
            buyer_breadth,
            recency,
            specialization,
        ]
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "Feature matrix contains non-finite values after transformation."
        )

    return x


def method_scores(
    candidates: pd.DataFrame,
    target_date: pd.Timestamp,
) -> dict[str, tuple[np.ndarray, bool]]:
    x = make_feature_matrix(candidates, target_date)

    return {
        "FREQUENCY": (
            candidates["N_CONTEXT_AWARDS"].to_numpy(dtype=np.float64),
            True,
        ),
        "WSM": (
            wsm_scores(x),
            True,
        ),
        "TOPSIS": (
            topsis_scores(x),
            True,
        ),
        "VIKOR": (
            vikor_q(x, v=0.5),
            False,
        ),
    }


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def validate_inputs(
    con: duckdb.DuckDBPyConnection,
    target_cases: Path,
    cpv2_features: Path,
    cpv3_features: Path,
    candidate_pool_summary: Path | None,
) -> None:
    require_columns(
        con,
        target_cases,
        {
            "TARGET_CASE_ID",
            "TARGET_DISPATCH_DATE",
            "TARGET_CPV2",
            "TARGET_CPV3",
            "TARGET_PROCUREMENT_COUNTRY",
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
            "OBSERVED_WINNER_HISTORICALLY_MATCHED",
            "HISTORY_MATCH_STATUS",
        },
        "target_case_features",
    )

    required_context_cols = {
        "CONTEXT_LEVEL",
        "CONTEXT_KEY",
        "SUPPLIER_ENTITY_ID",
        "N_CONTEXT_AWARDS",
        "N_UNIQUE_BUYERS_IN_CONTEXT",
        "LAST_CONTEXT_PUBLICATION_DATE",
        "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT",
    }

    require_columns(
        con,
        cpv2_features,
        required_context_cols,
        "supplier_cpv2_features",
    )

    require_columns(
        con,
        cpv3_features,
        required_context_cols,
        "supplier_cpv3_features",
    )

    if candidate_pool_summary is not None:
        require_columns(
            con,
            candidate_pool_summary,
            {
                "TARGET_CASE_ID",
                "POOL_DEFINITION",
                "OBSERVED_WINNER_INCLUDED",
                "CANDIDATE_POOL_SIZE",
            },
            "candidate_pool_case_summary",
        )


def load_targets(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        path,
        columns=[
            "TARGET_CASE_ID",
            "TARGET_DISPATCH_DATE",
            "TARGET_CPV2",
            "TARGET_CPV3",
            "TARGET_PROCUREMENT_COUNTRY",
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
            "OBSERVED_WINNER_HISTORICALLY_MATCHED",
            "HISTORY_MATCH_STATUS",
        ],
    )

    df["TARGET_DISPATCH_DATE"] = pd.to_datetime(
        df["TARGET_DISPATCH_DATE"],
        errors="raise",
    )
    df["OBSERVED_WINNER_HISTORICAL_ENTITY_ID"] = (
        df["OBSERVED_WINNER_HISTORICAL_ENTITY_ID"].astype("string")
    )
    return df


def load_context_features(
    path: Path,
    min_awards: int,
) -> pd.DataFrame:
    df = pd.read_parquet(
        path,
        columns=[
            "CONTEXT_KEY",
            "SUPPLIER_ENTITY_ID",
            "N_CONTEXT_AWARDS",
            "N_UNIQUE_BUYERS_IN_CONTEXT",
            "LAST_CONTEXT_PUBLICATION_DATE",
            "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT",
        ],
    )

    df = df.loc[
        df["N_CONTEXT_AWARDS"] >= min_awards
    ].copy()

    df["CONTEXT_KEY"] = df["CONTEXT_KEY"].astype("string")
    df["SUPPLIER_ENTITY_ID"] = df["SUPPLIER_ENTITY_ID"].astype("string")
    df["LAST_CONTEXT_PUBLICATION_DATE"] = pd.to_datetime(
        df["LAST_CONTEXT_PUBLICATION_DATE"],
        errors="raise",
    )

    dup = df.duplicated(
        subset=["CONTEXT_KEY", "SUPPLIER_ENTITY_ID"],
        keep=False,
    )
    if dup.any():
        raise ValueError(
            f"Context feature table contains {int(dup.sum())} duplicated "
            "(CONTEXT_KEY, SUPPLIER_ENTITY_ID) rows."
        )

    return df


# ---------------------------------------------------------------------
# Pool evaluation
# ---------------------------------------------------------------------

def eligible_target_cases(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    spec: PoolSpec,
) -> pd.DataFrame:
    t = targets.copy()
    t["CONTEXT_KEY"] = t[spec.target_context_column].astype("string")

    membership = candidates[
        ["CONTEXT_KEY", "SUPPLIER_ENTITY_ID"]
    ].rename(
        columns={
            "SUPPLIER_ENTITY_ID": "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"
        }
    )

    eligible = t.merge(
        membership,
        on=[
            "CONTEXT_KEY",
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        ],
        how="inner",
        validate="many_to_one",
    )

    eligible["POOL_DEFINITION"] = spec.name
    eligible["CONTEXT_LEVEL"] = spec.level
    return eligible


def evaluate_pool(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    spec: PoolSpec,
) -> pd.DataFrame:
    eligible = eligible_target_cases(targets, candidates, spec)

    if eligible.empty:
        raise ValueError(
            f"No winner-in-pool target cases found for {spec.name}."
        )

    candidate_groups = {
        str(context): group.reset_index(drop=True)
        for context, group in candidates.groupby(
            "CONTEXT_KEY",
            sort=False,
            observed=True,
        )
    }

    records: list[dict[str, object]] = []

    grouped_targets = eligible.groupby(
        ["CONTEXT_KEY", "TARGET_DISPATCH_DATE"],
        sort=False,
        observed=True,
    )

    total_groups = grouped_targets.ngroups
    print(
        f"  {spec.name}: {len(eligible):,} winner-in-pool cases "
        f"across {total_groups:,} context-date groups"
    )

    for group_no, ((context_key, target_date), cases) in enumerate(
        grouped_targets,
        start=1,
    ):
        context_key = str(context_key)
        cands = candidate_groups.get(context_key)

        if cands is None or cands.empty:
            raise ValueError(
                f"Missing candidate rows for context {context_key} "
                f"in {spec.name}."
            )

        supplier_ids = cands["SUPPLIER_ENTITY_ID"].astype("string").to_numpy()
        supplier_to_idx = {
            supplier_id: idx
            for idx, supplier_id in enumerate(supplier_ids)
        }

        score_sets = method_scores(
            cands,
            pd.Timestamp(target_date),
        )

        ranking_cache: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        for method, (scores, higher_is_better) in score_sets.items():
            ranking_cache[method] = rank_with_ties(
                scores,
                higher_is_better=higher_is_better,
            )

        pool_size = len(cands)

        for case in cases.itertuples(index=False):
            winner_id = str(
                case.OBSERVED_WINNER_HISTORICAL_ENTITY_ID
            )
            winner_idx = supplier_to_idx.get(winner_id)

            if winner_idx is None:
                raise ValueError(
                    f"Winner {winner_id} unexpectedly absent from "
                    f"{spec.name} context {context_key}."
                )

            for method in METHODS:
                midrank, best_rank, worst_rank, tie_size = ranking_cache[method]

                r_mid = float(midrank[winner_idx])
                r_best = int(best_rank[winner_idx])
                r_worst = int(worst_rank[winner_idx])
                t_size = int(tie_size[winner_idx])

                if pool_size <= 1:
                    percentile = 1.0
                else:
                    percentile = 1.0 - (
                        (r_mid - 1.0) / (pool_size - 1.0)
                    )

                records.append(
                    {
                        "TARGET_CASE_ID": case.TARGET_CASE_ID,
                        "TARGET_DISPATCH_DATE": case.TARGET_DISPATCH_DATE,
                        "TARGET_CPV2": case.TARGET_CPV2,
                        "TARGET_CPV3": case.TARGET_CPV3,
                        "TARGET_PROCUREMENT_COUNTRY":
                            case.TARGET_PROCUREMENT_COUNTRY,
                        "POOL_DEFINITION": spec.name,
                        "CONTEXT_LEVEL": spec.level,
                        "CONTEXT_KEY": context_key,
                        "METHOD": method,
                        "POOL_SIZE": pool_size,
                        "WINNER_RANK_MID": r_mid,
                        "WINNER_RANK_BEST": r_best,
                        "WINNER_RANK_WORST": r_worst,
                        "TIE_GROUP_SIZE": t_size,
                        "RECIPROCAL_RANK": 1.0 / r_mid,
                        "PERCENTILE_RANK": percentile,
                        "WINNER_AT_1": r_mid <= 1,
                        "WINNER_AT_5": r_mid <= 5,
                        "WINNER_AT_10": r_mid <= 10,
                        "WINNER_AT_50": r_mid <= 50,
                    }
                )

        if group_no % 500 == 0 or group_no == total_groups:
            print(
                f"    processed {group_no:,}/{total_groups:,} "
                "context-date groups"
            )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def summarize_methods(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (pool, method), g in results.groupby(
        ["POOL_DEFINITION", "METHOD"],
        sort=False,
    ):
        rows.append(
            {
                "POOL_DEFINITION": pool,
                "METHOD": method,
                "N_CASES": len(g),
                "WINNER_AT_1_PCT": 100.0 * g["WINNER_AT_1"].mean(),
                "WINNER_AT_5_PCT": 100.0 * g["WINNER_AT_5"].mean(),
                "WINNER_AT_10_PCT": 100.0 * g["WINNER_AT_10"].mean(),
                "WINNER_AT_50_PCT": 100.0 * g["WINNER_AT_50"].mean(),
                "MRR": g["RECIPROCAL_RANK"].mean(),
                "MEAN_PERCENTILE_RANK": g["PERCENTILE_RANK"].mean(),
                "MEDIAN_WINNER_RANK": g["WINNER_RANK_MID"].median(),
                "MEAN_TIE_GROUP_SIZE": g["TIE_GROUP_SIZE"].mean(),
                "MEDIAN_POOL_SIZE": g["POOL_SIZE"].median(),
            }
        )

    out = pd.DataFrame(rows)
    method_order = {
        "FREQUENCY": 1,
        "WSM": 2,
        "TOPSIS": 3,
        "VIKOR": 4,
    }
    pool_order = {
        "CPV2_MIN1": 1,
        "CPV3_MIN1": 2,
        "CPV2_MIN2": 3,
    }

    out["_pool_order"] = out["POOL_DEFINITION"].map(pool_order)
    out["_method_order"] = out["METHOD"].map(method_order)
    out = out.sort_values(
        ["_pool_order", "_method_order"]
    ).drop(columns=["_pool_order", "_method_order"])

    return out.reset_index(drop=True)


def summarize_by_dimension(
    results: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    rows = []

    for keys, g in results.groupby(
        ["POOL_DEFINITION", "METHOD", dimension],
        dropna=False,
        sort=False,
    ):
        pool, method, value = keys
        rows.append(
            {
                "POOL_DEFINITION": pool,
                "METHOD": method,
                dimension: value,
                "N_CASES": len(g),
                "WINNER_AT_1_PCT": 100.0 * g["WINNER_AT_1"].mean(),
                "WINNER_AT_5_PCT": 100.0 * g["WINNER_AT_5"].mean(),
                "WINNER_AT_10_PCT": 100.0 * g["WINNER_AT_10"].mean(),
                "MRR": g["RECIPROCAL_RANK"].mean(),
                "MEAN_PERCENTILE_RANK": g["PERCENTILE_RANK"].mean(),
                "MEDIAN_WINNER_RANK": g["WINNER_RANK_MID"].median(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["POOL_DEFINITION", "METHOD", "N_CASES"],
        ascending=[True, True, False],
    )


def feature_specification() -> pd.DataFrame:
    rows = [
        {
            "FEATURE": "CONTEXT_EXPERIENCE",
            "SOURCE_FIELD": "N_CONTEXT_AWARDS",
            "TRANSFORMATION": "log(1+x)",
            "DIRECTION": "benefit",
            "ROLE": "core",
            "INTERPRETATION":
                "Observed historical award frequency in the target CPV context.",
        },
        {
            "FEATURE": "OBSERVED_BUYER_BREADTH",
            "SOURCE_FIELD": "N_UNIQUE_BUYERS_IN_CONTEXT",
            "TRANSFORMATION": "log(1+x)",
            "DIRECTION": "benefit",
            "ROLE": "core",
            "INTERPRETATION":
                "Breadth of observed contracting-authority relationships in context.",
        },
        {
            "FEATURE": "PUBLIC_ACTIVITY_RECENCY",
            "SOURCE_FIELD": "LAST_CONTEXT_PUBLICATION_DATE",
            "TRANSFORMATION":
                "1/(1 + target_dispatch_date - last_context_publication_date)",
            "DIRECTION": "benefit",
            "ROLE": "core",
            "INTERPRETATION":
                "Recency of publicly available historical activity in context.",
        },
        {
            "FEATURE": "CONTEXT_SPECIALIZATION",
            "SOURCE_FIELD":
                "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT",
            "TRANSFORMATION": "identity",
            "DIRECTION": "benefit",
            "ROLE": "core",
            "INTERPRETATION":
                "Share of the supplier's observed historical awards occurring in context.",
        },
    ]
    return pd.DataFrame(rows)


def validation_checks(
    results: pd.DataFrame,
    candidate_pool_summary_path: Path | None,
) -> pd.DataFrame:
    rows = []

    for pool, g in results.groupby("POOL_DEFINITION"):
        # Each target case should have exactly one row per method.
        counts = g.groupby("TARGET_CASE_ID")["METHOD"].nunique()
        rows.append(
            {
                "CHECK": f"{pool}__four_methods_per_case",
                "VALUE": bool((counts == len(METHODS)).all()),
                "DETAIL": (
                    f"cases={counts.size}; "
                    f"min_methods={counts.min()}; max_methods={counts.max()}"
                ),
            }
        )

        rows.append(
            {
                "CHECK": f"{pool}__unique_case_method_rows",
                "VALUE": not g.duplicated(
                    ["TARGET_CASE_ID", "METHOD"]
                ).any(),
                "DETAIL": (
                    f"duplicate_rows="
                    f"{int(g.duplicated(['TARGET_CASE_ID', 'METHOD']).sum())}"
                ),
            }
        )

    if candidate_pool_summary_path is not None:
        cps = pd.read_parquet(
            candidate_pool_summary_path,
            columns=[
                "TARGET_CASE_ID",
                "POOL_DEFINITION",
                "OBSERVED_WINNER_INCLUDED",
            ],
        )

        expected = (
            cps.loc[cps["OBSERVED_WINNER_INCLUDED"]]
            .groupby("POOL_DEFINITION")["TARGET_CASE_ID"]
            .nunique()
            .to_dict()
        )

        observed = (
            results.groupby("POOL_DEFINITION")["TARGET_CASE_ID"]
            .nunique()
            .to_dict()
        )

        for pool in sorted(set(expected) | set(observed)):
            rows.append(
                {
                    "CHECK": f"{pool}__winner_in_pool_case_count_matches_preprocessing",
                    "VALUE": expected.get(pool) == observed.get(pool),
                    "DETAIL": (
                        f"expected={expected.get(pool)}; "
                        f"observed={observed.get(pool)}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def create_report(
    summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# E1 Deterministic Ranking Comparison Report",
        "",
        "## Scope",
        "",
        "E1 compares four deterministic ranking methods on cases whose observed "
        "winner is present in the corresponding historically active supplier "
        "comparison set.",
        "",
        "The primary analysis uses `CPV2_MIN1`. `CPV3_MIN1` and `CPV2_MIN2` "
        "are sensitivity analyses.",
        "",
        "## Methods",
        "",
        "- `FREQUENCY`: context award count.",
        "- `WSM`: equal-weight weighted sum over four core criteria.",
        "- `TOPSIS`: equal-weight TOPSIS over the same criteria.",
        "- `VIKOR`: equal-weight VIKOR with v = 0.5.",
        "",
        "## Core criteria",
        "",
        "1. log-transformed context award frequency;",
        "2. log-transformed observed buyer breadth in context;",
        "3. recency of last publicly available activity in context;",
        "4. share of supplier awards occurring in the target context.",
        "",
        "## Main results",
        "",
        "| Pool | Method | N | W@1 (%) | W@5 (%) | W@10 (%) | "
        "W@50 (%) | MRR | Mean percentile rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"| `{row['POOL_DEFINITION']}` "
            f"| `{row['METHOD']}` "
            f"| {int(row['N_CASES'])} "
            f"| {row['WINNER_AT_1_PCT']:.3f} "
            f"| {row['WINNER_AT_5_PCT']:.3f} "
            f"| {row['WINNER_AT_10_PCT']:.3f} "
            f"| {row['WINNER_AT_50_PCT']:.3f} "
            f"| {row['MRR']:.6f} "
            f"| {row['MEAN_PERCENTILE_RANK']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Tie handling",
            "",
            "Winner ranks use midranks. Best and worst possible ranks within "
            "each tie group are retained in the case-level result table.",
            "",
            "## Validation",
            "",
        ]
    )

    for _, row in validation_df.iterrows():
        status = "PASS" if bool(row["VALUE"]) else "FAIL"
        lines.append(
            f"- **{status}** `{row['CHECK']}`: {row['DETAIL']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation caution",
            "",
            "The observed procurement winner is a reference outcome, not proof of "
            "the uniquely optimal supplier. The ranking metrics therefore measure "
            "how strongly each historical ranking method aligns with observed award "
            "outcomes within the candidate sets, rather than causal or normative "
            "optimality.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run E1 deterministic ranking comparison on the compact "
            "supplier feature store."
        )
    )

    parser.add_argument(
        "--target-cases",
        required=True,
        type=Path,
        help="Path to target_case_features_2017.parquet",
    )
    parser.add_argument(
        "--cpv2-features",
        required=True,
        type=Path,
        help="Path to supplier_cpv2_features_2015_2016.parquet",
    )
    parser.add_argument(
        "--cpv3-features",
        required=True,
        type=Path,
        help="Path to supplier_cpv3_features_2015_2016.parquet",
    )
    parser.add_argument(
        "--candidate-pool-summary",
        type=Path,
        default=None,
        help=(
            "Optional path to candidate_pool_case_summary_2017.parquet "
            "for validation against preprocessing winner-in-pool counts."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/e1_rankings"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/e1_rankings"),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
    )

    args = parser.parse_args()

    for path in [
        args.target_cases,
        args.cpv2_features,
        args.cpv3_features,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

    if (
        args.candidate_pool_summary is not None
        and not args.candidate_pool_summary.exists()
    ):
        raise FileNotFoundError(
            f"Input not found: {args.candidate_pool_summary}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")

    try:
        print("1/7 Validating inputs...")
        validate_inputs(
            con,
            args.target_cases,
            args.cpv2_features,
            args.cpv3_features,
            args.candidate_pool_summary,
        )

        print("2/7 Loading target cases...")
        targets = load_targets(args.target_cases)

        print("3/7 Running CPV2_MIN1 primary analysis...")
        cpv2_min1 = load_context_features(
            args.cpv2_features,
            min_awards=1,
        )
        r_cpv2_min1 = evaluate_pool(
            targets,
            cpv2_min1,
            POOL_SPECS[0],
        )

        print("4/7 Running CPV3_MIN1 sensitivity analysis...")
        cpv3_min1 = load_context_features(
            args.cpv3_features,
            min_awards=1,
        )
        r_cpv3_min1 = evaluate_pool(
            targets,
            cpv3_min1,
            POOL_SPECS[1],
        )

        print("5/7 Running CPV2_MIN2 sensitivity analysis...")
        cpv2_min2 = cpv2_min1.loc[
            cpv2_min1["N_CONTEXT_AWARDS"] >= 2
        ].copy()
        r_cpv2_min2 = evaluate_pool(
            targets,
            cpv2_min2,
            POOL_SPECS[2],
        )

        print("6/7 Combining results and writing reports...")
        results = pd.concat(
            [
                r_cpv2_min1,
                r_cpv3_min1,
                r_cpv2_min2,
            ],
            ignore_index=True,
        )

        summary_df = summarize_methods(results)
        by_cpv2_df = summarize_by_dimension(
            results,
            "TARGET_CPV2",
        )
        by_country_df = summarize_by_dimension(
            results,
            "TARGET_PROCUREMENT_COUNTRY",
        )
        feature_df = feature_specification()
        validation_df = validation_checks(
            results,
            args.candidate_pool_summary,
        )

        write_csv(
            summary_df,
            args.report_dir / "e1_method_summary.csv",
        )
        write_csv(
            by_cpv2_df,
            args.report_dir / "e1_results_by_cpv2.csv",
        )
        write_csv(
            by_country_df,
            args.report_dir / "e1_results_by_country.csv",
        )
        write_csv(
            summary_df,
            args.report_dir / "e1_pool_sensitivity_summary.csv",
        )
        write_csv(
            feature_df,
            args.report_dir / "e1_feature_specification.csv",
        )
        write_csv(
            validation_df,
            args.report_dir / "e1_validation_checks.csv",
        )

        create_report(
            summary_df,
            validation_df,
            args.report_dir / "e1_ranking_report.md",
        )

        print("7/7 Writing compact case-level ranking results...")
        results.to_parquet(
            args.output_dir / "e1_case_ranking_results.parquet",
            index=False,
        )

        print()
        print("=" * 78)
        print("E1 deterministic ranking comparison complete")
        print("=" * 78)
        print(f"Processed results: {args.output_dir.resolve()}")
        print(f"Reports:           {args.report_dir.resolve()}")
        print()
        print("Review these first:")
        print("  e1_ranking_report.md")
        print("  e1_method_summary.csv")
        print("  e1_validation_checks.csv")
        print("  e1_results_by_cpv2.csv")
        print("  e1_results_by_country.csv")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
