#!/usr/bin/env python3
"""
E2: Scenario-sensitive TOPSIS analysis.

Purpose
-------
Evaluate how controlled changes in criterion weights alter supplier rankings
while keeping the candidate pool, features, and TOPSIS aggregation method fixed.

Primary pool
------------
CPV2_MIN1 only.

Scenarios
---------
BALANCED
    Experience            0.25
    Buyer breadth         0.25
    Public recency        0.25
    Context specialization 0.25

EXPERIENCE
    Experience            0.45
    Buyer breadth         0.25
    Public recency        0.15
    Context specialization 0.15

RECENCY
    Experience            0.20
    Buyer breadth         0.15
    Public recency        0.45
    Context specialization 0.20

These are controlled sensitivity scenarios, not estimated decision-maker
preferences.

Core features
-------------
1. log(1 + N_CONTEXT_AWARDS)                         benefit
2. log(1 + N_UNIQUE_BUYERS_IN_CONTEXT)              benefit
3. 1 / (1 + DAYS_SINCE_LAST_CONTEXT_PUBLICATION)   benefit
4. SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT              benefit

Evaluation
----------
For each target case whose observed winner is present in CPV2_MIN1:
- winner rank under each scenario,
- Winner@1, Winner@5, Winner@10, Winner@50,
- reciprocal rank,
- percentile rank.

Scenario sensitivity:
- mean Spearman rank correlation,
- mean Top-10 overlap,
- mean Top-50 overlap,
- mean and median absolute observed-winner rank movement,
- direction and magnitude of observed-winner rank movement.

Scenario responsiveness:
- EXPERIENCE entrants versus exits from Balanced Top-10,
- RECENCY entrants versus exits from Balanced Top-10.

Optional validation:
- compare BALANCED scenario winner ranks with E1 CPV2_MIN1 TOPSIS results.

Outputs
-------
Processed:
- e2_case_scenario_results.parquet

Reports:
- e2_scenario_specification.csv
- e2_scenario_summary.csv
- e2_pairwise_stability.csv
- e2_winner_rank_movement.csv
- e2_scenario_responsiveness.csv
- e2_validation_checks.csv
- e2_scenario_report.md

Requires
--------
pip install numpy pandas pyarrow
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FEATURE_NAMES = (
    "CONTEXT_EXPERIENCE",
    "OBSERVED_BUYER_BREADTH",
    "PUBLIC_ACTIVITY_RECENCY",
    "CONTEXT_SPECIALIZATION",
)

SCENARIOS = {
    "BALANCED": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
    "EXPERIENCE": np.array([0.45, 0.25, 0.15, 0.15], dtype=np.float64),
    "RECENCY": np.array([0.20, 0.15, 0.45, 0.20], dtype=np.float64),
}

TOP_K_VALUES = (1, 5, 10, 50)


# ---------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------

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
        raise ValueError(
            f"Found {int(np.sum(days < 0))} candidate rows with "
            "historical publication dates after the target date."
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
            "Feature matrix contains non-finite values."
        )

    return x


def topsis_scores(
    x: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    if len(weights) != x.shape[1]:
        raise ValueError("weights length does not match number of criteria.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("scenario weights must sum to 1.")

    norms = np.sqrt(np.sum(np.square(x), axis=0))

    normalized = np.zeros_like(x, dtype=np.float64)
    nonzero = norms > 0
    normalized[:, nonzero] = x[:, nonzero] / norms[nonzero]

    weighted = normalized * weights

    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    d_best = np.sqrt(
        np.sum(np.square(weighted - ideal_best), axis=1)
    )
    d_worst = np.sqrt(
        np.sum(np.square(weighted - ideal_worst), axis=1)
    )

    denom = d_best + d_worst

    scores = np.full(len(x), 0.5, dtype=np.float64)
    valid = denom > 0
    scores[valid] = d_worst[valid] / denom[valid]

    return scores


def rank_with_ties(
    scores: np.ndarray,
    decimals: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Higher score is better. Returns:
      midrank, best rank, worst rank, tie-group size.
    """
    s = np.round(
        np.asarray(scores, dtype=np.float64),
        decimals=decimals,
    )

    order = np.argsort(-s, kind="mergesort")
    sorted_s = s[order]

    n = len(s)
    midrank = np.empty(n, dtype=np.float64)
    best_rank = np.empty(n, dtype=np.int64)
    worst_rank = np.empty(n, dtype=np.int64)
    tie_size = np.empty(n, dtype=np.int64)

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_s[end] == sorted_s[start]:
            end += 1

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


def ordinal_rank_no_ties(scores: np.ndarray) -> np.ndarray:
    """
    Deterministic 1..n ordinal ranks for global rank-correlation calculations.
    Ties are secondarily ordered by stable original position.
    """
    order = np.argsort(
        -np.round(scores, 12),
        kind="mergesort",
    )
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return ranks


def spearman_corr(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
) -> float:
    a = np.asarray(ranks_a, dtype=np.float64)
    b = np.asarray(ranks_b, dtype=np.float64)

    if len(a) <= 1:
        return 1.0

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    denom = np.sqrt(
        np.sum(np.square(a_centered))
        * np.sum(np.square(b_centered))
    )

    if denom == 0:
        return 1.0

    return float(
        np.sum(a_centered * b_centered) / denom
    )


def topk_supplier_ids(
    scores: np.ndarray,
    supplier_ids: np.ndarray,
    k: int,
) -> set[str]:
    k_eff = min(k, len(scores))
    order = np.argsort(
        -np.round(scores, 12),
        kind="mergesort",
    )
    return set(
        supplier_ids[order[:k_eff]].astype(str).tolist()
    )


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def load_targets(path: Path) -> pd.DataFrame:
    cols = [
        "TARGET_CASE_ID",
        "TARGET_DISPATCH_DATE",
        "TARGET_CPV2",
        "TARGET_PROCUREMENT_COUNTRY",
        "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        "OBSERVED_WINNER_HISTORICALLY_MATCHED",
        "HISTORY_MATCH_STATUS",
    ]

    df = pd.read_parquet(path, columns=cols)

    df["TARGET_DISPATCH_DATE"] = pd.to_datetime(
        df["TARGET_DISPATCH_DATE"],
        errors="raise",
    )
    df["TARGET_CPV2"] = df["TARGET_CPV2"].astype("string")
    df["OBSERVED_WINNER_HISTORICAL_ENTITY_ID"] = (
        df["OBSERVED_WINNER_HISTORICAL_ENTITY_ID"].astype("string")
    )

    return df


def load_cpv2_features(path: Path) -> pd.DataFrame:
    cols = [
        "CONTEXT_KEY",
        "SUPPLIER_ENTITY_ID",
        "N_CONTEXT_AWARDS",
        "N_UNIQUE_BUYERS_IN_CONTEXT",
        "LAST_CONTEXT_PUBLICATION_DATE",
        "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT",
    ]

    df = pd.read_parquet(path, columns=cols)

    df["CONTEXT_KEY"] = df["CONTEXT_KEY"].astype("string")
    df["SUPPLIER_ENTITY_ID"] = df["SUPPLIER_ENTITY_ID"].astype("string")
    df["LAST_CONTEXT_PUBLICATION_DATE"] = pd.to_datetime(
        df["LAST_CONTEXT_PUBLICATION_DATE"],
        errors="raise",
    )

    dup = df.duplicated(
        ["CONTEXT_KEY", "SUPPLIER_ENTITY_ID"],
        keep=False,
    )
    if dup.any():
        raise ValueError(
            f"CPV2 feature table contains {int(dup.sum())} duplicated "
            "(CONTEXT_KEY, SUPPLIER_ENTITY_ID) rows."
        )

    return df


def eligible_cases(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    membership = candidates[
        ["CONTEXT_KEY", "SUPPLIER_ENTITY_ID"]
    ].rename(
        columns={
            "CONTEXT_KEY": "TARGET_CPV2",
            "SUPPLIER_ENTITY_ID":
                "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        }
    )

    out = targets.merge(
        membership,
        on=[
            "TARGET_CPV2",
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        ],
        how="inner",
        validate="many_to_one",
    )

    return out


# ---------------------------------------------------------------------
# E2 analysis
# ---------------------------------------------------------------------

def run_e2(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    eligible = eligible_cases(targets, candidates)

    if eligible.empty:
        raise ValueError("No CPV2 winner-in-pool cases found.")

    candidate_groups = {
        str(context): group.reset_index(drop=True)
        for context, group in candidates.groupby(
            "CONTEXT_KEY",
            sort=False,
            observed=True,
        )
    }

    case_records: list[dict[str, object]] = []
    stability_records: list[dict[str, object]] = []
    responsiveness_records: list[dict[str, object]] = []

    grouped_targets = eligible.groupby(
        ["TARGET_CPV2", "TARGET_DISPATCH_DATE"],
        sort=False,
        observed=True,
    )

    total_groups = grouped_targets.ngroups

    print(
        f"E2: {len(eligible):,} winner-in-pool cases across "
        f"{total_groups:,} context-date groups"
    )

    scenario_pairs = [
        ("BALANCED", "EXPERIENCE"),
        ("BALANCED", "RECENCY"),
        ("EXPERIENCE", "RECENCY"),
    ]

    for group_no, ((context_key, target_date), cases) in enumerate(
        grouped_targets,
        start=1,
    ):
        context_key = str(context_key)
        cands = candidate_groups.get(context_key)

        if cands is None or cands.empty:
            raise ValueError(
                f"Missing CPV2 candidates for context {context_key}."
            )

        supplier_ids = (
            cands["SUPPLIER_ENTITY_ID"]
            .astype("string")
            .to_numpy()
        )
        supplier_to_idx = {
            str(supplier_id): idx
            for idx, supplier_id in enumerate(supplier_ids)
        }

        x = make_feature_matrix(
            cands,
            pd.Timestamp(target_date),
        )

        scenario_scores: dict[str, np.ndarray] = {}
        scenario_midrank: dict[str, np.ndarray] = {}
        scenario_best: dict[str, np.ndarray] = {}
        scenario_worst: dict[str, np.ndarray] = {}
        scenario_tie: dict[str, np.ndarray] = {}
        scenario_ordinal: dict[str, np.ndarray] = {}

        for scenario, weights in SCENARIOS.items():
            scores = topsis_scores(x, weights)
            midrank, best, worst, tie = rank_with_ties(scores)

            scenario_scores[scenario] = scores
            scenario_midrank[scenario] = midrank
            scenario_best[scenario] = best
            scenario_worst[scenario] = worst
            scenario_tie[scenario] = tie
            scenario_ordinal[scenario] = ordinal_rank_no_ties(scores)

        pool_size = len(cands)
        case_count = len(cases)

        # Pairwise stability metrics are identical for all cases in this
        # context-date group; N_CASES_IN_GROUP allows case-weighted aggregation.
        for scenario_a, scenario_b in scenario_pairs:
            top10_a = topk_supplier_ids(
                scenario_scores[scenario_a],
                supplier_ids,
                10,
            )
            top10_b = topk_supplier_ids(
                scenario_scores[scenario_b],
                supplier_ids,
                10,
            )

            top50_a = topk_supplier_ids(
                scenario_scores[scenario_a],
                supplier_ids,
                50,
            )
            top50_b = topk_supplier_ids(
                scenario_scores[scenario_b],
                supplier_ids,
                50,
            )

            denom10 = max(
                1,
                min(10, pool_size),
            )
            denom50 = max(
                1,
                min(50, pool_size),
            )

            stability_records.append(
                {
                    "CONTEXT_KEY": context_key,
                    "TARGET_DISPATCH_DATE": target_date,
                    "SCENARIO_A": scenario_a,
                    "SCENARIO_B": scenario_b,
                    "POOL_SIZE": pool_size,
                    "N_CASES_IN_GROUP": case_count,
                    "SPEARMAN_RHO": spearman_corr(
                        scenario_ordinal[scenario_a],
                        scenario_ordinal[scenario_b],
                    ),
                    "TOP10_OVERLAP":
                        len(top10_a & top10_b) / denom10,
                    "TOP50_OVERLAP":
                        len(top50_a & top50_b) / denom50,
                }
            )

        # Scenario responsiveness: entrants/exits relative to BALANCED.
        balanced_top10 = topk_supplier_ids(
            scenario_scores["BALANCED"],
            supplier_ids,
            10,
        )

        for scenario, focus_feature_idx, focus_feature_name in [
            ("EXPERIENCE", 0, "CONTEXT_EXPERIENCE"),
            ("RECENCY", 2, "PUBLIC_ACTIVITY_RECENCY"),
        ]:
            scenario_top10 = topk_supplier_ids(
                scenario_scores[scenario],
                supplier_ids,
                10,
            )

            entrants = scenario_top10 - balanced_top10
            exits = balanced_top10 - scenario_top10

            supplier_index = {
                str(s): i for i, s in enumerate(supplier_ids)
            }

            entrant_vals = np.array(
                [
                    x[supplier_index[s], focus_feature_idx]
                    for s in entrants
                ],
                dtype=np.float64,
            )
            exit_vals = np.array(
                [
                    x[supplier_index[s], focus_feature_idx]
                    for s in exits
                ],
                dtype=np.float64,
            )

            responsiveness_records.append(
                {
                    "CONTEXT_KEY": context_key,
                    "TARGET_DISPATCH_DATE": target_date,
                    "SCENARIO": scenario,
                    "FOCUS_FEATURE": focus_feature_name,
                    "POOL_SIZE": pool_size,
                    "N_CASES_IN_GROUP": case_count,
                    "N_TOP10_ENTRANTS": len(entrants),
                    "N_TOP10_EXITS": len(exits),
                    "MEAN_ENTRANT_FEATURE":
                        float(np.mean(entrant_vals))
                        if len(entrant_vals) else np.nan,
                    "MEAN_EXIT_FEATURE":
                        float(np.mean(exit_vals))
                        if len(exit_vals) else np.nan,
                    "ENTRANT_MINUS_EXIT_FEATURE":
                        float(np.mean(entrant_vals) - np.mean(exit_vals))
                        if len(entrant_vals) and len(exit_vals)
                        else np.nan,
                }
            )

        # Case-level winner ranks.
        for case in cases.itertuples(index=False):
            winner_id = str(
                case.OBSERVED_WINNER_HISTORICAL_ENTITY_ID
            )
            winner_idx = supplier_to_idx.get(winner_id)

            if winner_idx is None:
                raise ValueError(
                    f"Winner {winner_id} unexpectedly absent from "
                    f"CPV2 context {context_key}."
                )

            for scenario in SCENARIOS:
                midrank = float(
                    scenario_midrank[scenario][winner_idx]
                )
                best = int(
                    scenario_best[scenario][winner_idx]
                )
                worst = int(
                    scenario_worst[scenario][winner_idx]
                )
                tie_size = int(
                    scenario_tie[scenario][winner_idx]
                )

                if pool_size <= 1:
                    percentile = 1.0
                else:
                    percentile = 1.0 - (
                        (midrank - 1.0) / (pool_size - 1.0)
                    )

                case_records.append(
                    {
                        "TARGET_CASE_ID": case.TARGET_CASE_ID,
                        "TARGET_DISPATCH_DATE":
                            case.TARGET_DISPATCH_DATE,
                        "TARGET_CPV2": case.TARGET_CPV2,
                        "TARGET_PROCUREMENT_COUNTRY":
                            case.TARGET_PROCUREMENT_COUNTRY,
                        "SCENARIO": scenario,
                        "POOL_DEFINITION": "CPV2_MIN1",
                        "POOL_SIZE": pool_size,
                        "WINNER_RANK_MID": midrank,
                        "WINNER_RANK_BEST": best,
                        "WINNER_RANK_WORST": worst,
                        "TIE_GROUP_SIZE": tie_size,
                        "RECIPROCAL_RANK": 1.0 / midrank,
                        "PERCENTILE_RANK": percentile,
                        "WINNER_AT_1": midrank <= 1,
                        "WINNER_AT_5": midrank <= 5,
                        "WINNER_AT_10": midrank <= 10,
                        "WINNER_AT_50": midrank <= 50,
                    }
                )

        if group_no % 500 == 0 or group_no == total_groups:
            print(
                f"  processed {group_no:,}/{total_groups:,} "
                "context-date groups"
            )

    return (
        pd.DataFrame(case_records),
        pd.DataFrame(stability_records),
        pd.DataFrame(responsiveness_records),
    )


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def scenario_summary(
    case_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for scenario, g in case_results.groupby(
        "SCENARIO",
        sort=False,
    ):
        rows.append(
            {
                "SCENARIO": scenario,
                "N_CASES": len(g),
                "WINNER_AT_1_PCT":
                    100.0 * g["WINNER_AT_1"].mean(),
                "WINNER_AT_5_PCT":
                    100.0 * g["WINNER_AT_5"].mean(),
                "WINNER_AT_10_PCT":
                    100.0 * g["WINNER_AT_10"].mean(),
                "WINNER_AT_50_PCT":
                    100.0 * g["WINNER_AT_50"].mean(),
                "MRR": g["RECIPROCAL_RANK"].mean(),
                "MEAN_PERCENTILE_RANK":
                    g["PERCENTILE_RANK"].mean(),
                "MEDIAN_WINNER_RANK":
                    g["WINNER_RANK_MID"].median(),
            }
        )

    order = {
        "BALANCED": 1,
        "EXPERIENCE": 2,
        "RECENCY": 3,
    }

    out = pd.DataFrame(rows)
    out["_order"] = out["SCENARIO"].map(order)

    return (
        out.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def pairwise_stability_summary(
    stability_group: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (a, b), g in stability_group.groupby(
        ["SCENARIO_A", "SCENARIO_B"],
        sort=False,
    ):
        w = g["N_CASES_IN_GROUP"].to_numpy(dtype=np.float64)
        w = w / w.sum()

        rows.append(
            {
                "SCENARIO_A": a,
                "SCENARIO_B": b,
                "N_CONTEXT_DATE_GROUPS": len(g),
                "N_CASES_WEIGHTED":
                    int(g["N_CASES_IN_GROUP"].sum()),
                "MEAN_SPEARMAN_RHO":
                    float(np.sum(w * g["SPEARMAN_RHO"])),
                "MEAN_TOP10_OVERLAP":
                    float(np.sum(w * g["TOP10_OVERLAP"])),
                "MEAN_TOP50_OVERLAP":
                    float(np.sum(w * g["TOP50_OVERLAP"])),
            }
        )

    return pd.DataFrame(rows)


def winner_rank_movement(
    case_results: pd.DataFrame,
) -> pd.DataFrame:
    pivot = case_results.pivot(
        index="TARGET_CASE_ID",
        columns="SCENARIO",
        values="WINNER_RANK_MID",
    )

    rows = []

    for scenario in ["EXPERIENCE", "RECENCY"]:
        delta = pivot[scenario] - pivot["BALANCED"]

        rows.append(
            {
                "REFERENCE_SCENARIO": "BALANCED",
                "COMPARISON_SCENARIO": scenario,
                "N_CASES": len(delta),
                "MEAN_RANK_CHANGE":
                    float(delta.mean()),
                "MEDIAN_RANK_CHANGE":
                    float(delta.median()),
                "MEAN_ABS_RANK_CHANGE":
                    float(delta.abs().mean()),
                "MEDIAN_ABS_RANK_CHANGE":
                    float(delta.abs().median()),
                "PCT_WINNER_MOVED_UP":
                    100.0 * float((delta < 0).mean()),
                "PCT_WINNER_UNCHANGED":
                    100.0 * float((delta == 0).mean()),
                "PCT_WINNER_MOVED_DOWN":
                    100.0 * float((delta > 0).mean()),
            }
        )

    return pd.DataFrame(rows)


def responsiveness_summary(
    responsiveness_group: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (scenario, feature), g in responsiveness_group.groupby(
        ["SCENARIO", "FOCUS_FEATURE"],
        sort=False,
    ):
        valid = g["ENTRANT_MINUS_EXIT_FEATURE"].notna()
        gv = g.loc[valid].copy()

        if gv.empty:
            rows.append(
                {
                    "SCENARIO": scenario,
                    "FOCUS_FEATURE": feature,
                    "N_VALID_CONTEXT_DATE_GROUPS": 0,
                    "N_CASES_WEIGHTED": 0,
                    "MEAN_ENTRANT_MINUS_EXIT_FEATURE": np.nan,
                    "PCT_GROUPS_WITH_EXPECTED_POSITIVE_DELTA": np.nan,
                }
            )
            continue

        w = gv["N_CASES_IN_GROUP"].to_numpy(dtype=np.float64)
        w = w / w.sum()

        delta = gv[
            "ENTRANT_MINUS_EXIT_FEATURE"
        ].to_numpy(dtype=np.float64)

        rows.append(
            {
                "SCENARIO": scenario,
                "FOCUS_FEATURE": feature,
                "N_VALID_CONTEXT_DATE_GROUPS": len(gv),
                "N_CASES_WEIGHTED":
                    int(gv["N_CASES_IN_GROUP"].sum()),
                "MEAN_ENTRANT_MINUS_EXIT_FEATURE":
                    float(np.sum(w * delta)),
                "PCT_GROUPS_WITH_EXPECTED_POSITIVE_DELTA":
                    100.0 * float(np.mean(delta > 0)),
            }
        )

    return pd.DataFrame(rows)


def scenario_specification() -> pd.DataFrame:
    rows = []

    for scenario, weights in SCENARIOS.items():
        for feature, weight in zip(
            FEATURE_NAMES,
            weights,
        ):
            rows.append(
                {
                    "SCENARIO": scenario,
                    "FEATURE": feature,
                    "WEIGHT": float(weight),
                }
            )

    return pd.DataFrame(rows)


def validation_checks(
    case_results: pd.DataFrame,
    e1_case_results_path: Path | None,
) -> pd.DataFrame:
    rows = []

    counts = (
        case_results.groupby("TARGET_CASE_ID")["SCENARIO"]
        .nunique()
    )

    rows.append(
        {
            "CHECK": "three_scenarios_per_case",
            "VALUE": bool((counts == 3).all()),
            "DETAIL":
                f"cases={counts.size}; min={counts.min()}; max={counts.max()}",
        }
    )

    dup_count = int(
        case_results.duplicated(
            ["TARGET_CASE_ID", "SCENARIO"]
        ).sum()
    )

    rows.append(
        {
            "CHECK": "unique_case_scenario_rows",
            "VALUE": dup_count == 0,
            "DETAIL": f"duplicate_rows={dup_count}",
        }
    )

    scenario_counts = (
        case_results.groupby("SCENARIO")["TARGET_CASE_ID"]
        .nunique()
        .to_dict()
    )

    rows.append(
        {
            "CHECK": "equal_case_count_across_scenarios",
            "VALUE":
                len(set(scenario_counts.values())) == 1,
            "DETAIL": str(scenario_counts),
        }
    )

    if e1_case_results_path is not None:
        e1 = pd.read_parquet(
            e1_case_results_path,
            columns=[
                "TARGET_CASE_ID",
                "POOL_DEFINITION",
                "METHOD",
                "WINNER_RANK_MID",
            ],
        )

        e1 = e1.loc[
            (e1["POOL_DEFINITION"] == "CPV2_MIN1")
            & (e1["METHOD"] == "TOPSIS"),
            [
                "TARGET_CASE_ID",
                "WINNER_RANK_MID",
            ],
        ].rename(
            columns={
                "WINNER_RANK_MID":
                    "E1_TOPSIS_WINNER_RANK_MID",
            }
        )

        e2 = case_results.loc[
            case_results["SCENARIO"] == "BALANCED",
            [
                "TARGET_CASE_ID",
                "WINNER_RANK_MID",
            ],
        ].rename(
            columns={
                "WINNER_RANK_MID":
                    "E2_BALANCED_WINNER_RANK_MID",
            }
        )

        merged = e1.merge(
            e2,
            on="TARGET_CASE_ID",
            how="outer",
            indicator=True,
            validate="one_to_one",
        )

        matched_ids = bool(
            (merged["_merge"] == "both").all()
        )

        rank_match = bool(
            np.allclose(
                merged.loc[
                    merged["_merge"] == "both",
                    "E1_TOPSIS_WINNER_RANK_MID",
                ],
                merged.loc[
                    merged["_merge"] == "both",
                    "E2_BALANCED_WINNER_RANK_MID",
                ],
                rtol=0,
                atol=1e-9,
            )
        )

        rows.append(
            {
                "CHECK": "balanced_case_ids_match_e1_topsis",
                "VALUE": matched_ids,
                "DETAIL":
                    str(merged["_merge"].value_counts().to_dict()),
            }
        )

        rows.append(
            {
                "CHECK": "balanced_winner_ranks_match_e1_topsis",
                "VALUE": rank_match,
                "DETAIL":
                    f"matched_cases={int((merged['_merge'] == 'both').sum())}",
            }
        )

    return pd.DataFrame(rows)


def write_report(
    scenario_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    responsiveness_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# E2 Scenario-Sensitive TOPSIS Report",
        "",
        "## Scope",
        "",
        "E2 holds the candidate pool, feature definitions, and TOPSIS method "
        "constant while changing only the criterion weights across three "
        "controlled scenarios: BALANCED, EXPERIENCE, and RECENCY.",
        "",
        "The scenarios are sensitivity profiles and are not claimed to be "
        "estimated decision-maker preferences.",
        "",
        "## Scenario ranking performance",
        "",
        "| Scenario | N | W@1 (%) | W@5 (%) | W@10 (%) | W@50 (%) | MRR | Mean percentile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in scenario_df.iterrows():
        lines.append(
            f"| `{row['SCENARIO']}` "
            f"| {int(row['N_CASES'])} "
            f"| {row['WINNER_AT_1_PCT']:.3f} "
            f"| {row['WINNER_AT_5_PCT']:.3f} "
            f"| {row['WINNER_AT_10_PCT']:.3f} "
            f"| {row['WINNER_AT_50_PCT']:.3f} "
            f"| {row['MRR']:.6f} "
            f"| {row['MEAN_PERCENTILE_RANK']:.6f} |"
        )

    lines.extend([
        "",
        "## Pairwise scenario stability",
        "",
        "| Scenario A | Scenario B | Mean Spearman rho | Top-10 overlap | Top-50 overlap |",
        "|---|---|---:|---:|---:|",
    ])

    for _, row in stability_df.iterrows():
        lines.append(
            f"| `{row['SCENARIO_A']}` "
            f"| `{row['SCENARIO_B']}` "
            f"| {row['MEAN_SPEARMAN_RHO']:.6f} "
            f"| {row['MEAN_TOP10_OVERLAP']:.6f} "
            f"| {row['MEAN_TOP50_OVERLAP']:.6f} |"
        )

    lines.extend([
        "",
        "## Observed-winner rank movement relative to BALANCED",
        "",
        "| Scenario | Mean change | Median change | Mean absolute change | % moved up | % unchanged | % moved down |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for _, row in movement_df.iterrows():
        lines.append(
            f"| `{row['COMPARISON_SCENARIO']}` "
            f"| {row['MEAN_RANK_CHANGE']:.3f} "
            f"| {row['MEDIAN_RANK_CHANGE']:.3f} "
            f"| {row['MEAN_ABS_RANK_CHANGE']:.3f} "
            f"| {row['PCT_WINNER_MOVED_UP']:.3f} "
            f"| {row['PCT_WINNER_UNCHANGED']:.3f} "
            f"| {row['PCT_WINNER_MOVED_DOWN']:.3f} |"
        )

    lines.extend([
        "",
        "## Scenario responsiveness",
        "",
        "| Scenario | Focus feature | Mean entrant-minus-exit feature | % groups positive |",
        "|---|---|---:|---:|",
    ])

    for _, row in responsiveness_df.iterrows():
        lines.append(
            f"| `{row['SCENARIO']}` "
            f"| `{row['FOCUS_FEATURE']}` "
            f"| {row['MEAN_ENTRANT_MINUS_EXIT_FEATURE']:.6f} "
            f"| {row['PCT_GROUPS_WITH_EXPECTED_POSITIVE_DELTA']:.3f} |"
        )

    lines.extend([
        "",
        "## Validation",
        "",
    ])

    for _, row in validation_df.iterrows():
        status = "PASS" if bool(row["VALUE"]) else "FAIL"
        lines.append(
            f"- **{status}** `{row['CHECK']}`: {row['DETAIL']}"
        )

    lines.extend([
        "",
        "## Interpretation caution",
        "",
        "E2 evaluates controlled ranking sensitivity to explicit weight changes. "
        "It does not claim that the EXPERIENCE or RECENCY weights represent "
        "empirically estimated procurement preferences. The purpose is to test "
        "whether transparent scenario changes lead to measurable and directionally "
        "consistent changes in supplier rankings.",
        "",
    ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-cases",
        required=True,
        type=Path,
        help="target_case_features_2017.parquet",
    )
    parser.add_argument(
        "--cpv2-features",
        required=True,
        type=Path,
        help="supplier_cpv2_features_2015_2016.parquet",
    )
    parser.add_argument(
        "--e1-case-results",
        type=Path,
        default=None,
        help=(
            "Optional e1_case_ranking_results.parquet for validation that "
            "BALANCED reproduces E1 TOPSIS CPV2_MIN1 winner ranks."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/e2_scenarios"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/e2_scenarios"),
    )

    args = parser.parse_args()

    for path in [
        args.target_cases,
        args.cpv2_features,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if (
        args.e1_case_results is not None
        and not args.e1_case_results.exists()
    ):
        raise FileNotFoundError(args.e1_case_results)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1/6 Loading inputs...")
    targets = load_targets(args.target_cases)
    candidates = load_cpv2_features(args.cpv2_features)

    print("2/6 Running scenario rankings...")
    (
        case_results,
        stability_group,
        responsiveness_group,
    ) = run_e2(
        targets,
        candidates,
    )

    print("3/6 Summarizing scenario performance...")
    scenario_df = scenario_summary(case_results)
    stability_df = pairwise_stability_summary(
        stability_group
    )
    movement_df = winner_rank_movement(
        case_results
    )
    responsiveness_df = responsiveness_summary(
        responsiveness_group
    )
    specification_df = scenario_specification()

    print("4/6 Running validation...")
    validation_df = validation_checks(
        case_results,
        args.e1_case_results,
    )

    print("5/6 Writing outputs...")
    case_results.to_parquet(
        args.output_dir
        / "e2_case_scenario_results.parquet",
        index=False,
    )

    specification_df.to_csv(
        args.report_dir
        / "e2_scenario_specification.csv",
        index=False,
    )
    scenario_df.to_csv(
        args.report_dir
        / "e2_scenario_summary.csv",
        index=False,
    )
    stability_df.to_csv(
        args.report_dir
        / "e2_pairwise_stability.csv",
        index=False,
    )
    movement_df.to_csv(
        args.report_dir
        / "e2_winner_rank_movement.csv",
        index=False,
    )
    responsiveness_df.to_csv(
        args.report_dir
        / "e2_scenario_responsiveness.csv",
        index=False,
    )
    validation_df.to_csv(
        args.report_dir
        / "e2_validation_checks.csv",
        index=False,
    )

    write_report(
        scenario_df,
        stability_df,
        movement_df,
        responsiveness_df,
        validation_df,
        args.report_dir
        / "e2_scenario_report.md",
    )

    print("6/6 Complete.")
    print()
    print("Review these first:")
    print("  e2_scenario_report.md")
    print("  e2_scenario_summary.csv")
    print("  e2_pairwise_stability.csv")
    print("  e2_winner_rank_movement.csv")
    print("  e2_scenario_responsiveness.csv")
    print("  e2_validation_checks.csv")


if __name__ == "__main__":
    main()
