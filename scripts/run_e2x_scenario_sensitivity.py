#!/usr/bin/env python3
"""
E2x scenario-sensitivity experiment on the frozen E1x M3 structured backbone.

This script reuses the validated feature-construction functions from:
    run_e1x_extended_procurement_fit_rankings_v2.py

Backbone groups
---------------
1. CATEGORY_FIT
2. GEOGRAPHIC_FIT
3. BUYER_RELATIONSHIP
4. ACTIVITY_PROFILE

Scenarios
---------
BALANCED
    Category 0.25, Geography 0.25, Buyer 0.25, Activity 0.25

CONTINUITY
    Category 0.25, Geography 0.20, Buyer 0.40, Activity 0.15

DIVERSIFICATION
    Category 0.30, Geography 0.35, Buyer 0.10, Activity 0.25

Purpose
-------
Evaluate whether explicit scenario weights produce:
- stable but meaningful ranking changes;
- interpretable top-k turnover;
- directional responsiveness in the emphasized conceptual group;
- observed-winner rank movement.

Important
---------
This is a sensitivity experiment, not a second model-selection search.
The BALANCED scenario must reproduce E1x M3_GEO_BUYER ranks exactly.

Outputs
-------
Processed:
- e2x_case_scenario_results.parquet

Reports:
- e2x_scenario_specification.csv
- e2x_scenario_summary.csv
- e2x_pairwise_stability.csv
- e2x_directional_responsiveness.csv
- e2x_validation_checks.csv
- e2x_scenario_report.md
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------

SCENARIOS = {
    "BALANCED": {
        "CATEGORY_FIT": 0.25,
        "GEOGRAPHIC_FIT": 0.25,
        "BUYER_RELATIONSHIP": 0.25,
        "ACTIVITY_PROFILE": 0.25,
    },
    "CONTINUITY": {
        "CATEGORY_FIT": 0.25,
        "GEOGRAPHIC_FIT": 0.20,
        "BUYER_RELATIONSHIP": 0.40,
        "ACTIVITY_PROFILE": 0.15,
    },
    "DIVERSIFICATION": {
        "CATEGORY_FIT": 0.30,
        "GEOGRAPHIC_FIT": 0.35,
        "BUYER_RELATIONSHIP": 0.10,
        "ACTIVITY_PROFILE": 0.25,
    },
}

FOCUS_GROUP = {
    "CONTINUITY": "BUYER_RELATIONSHIP",
    "DIVERSIFICATION": "GEOGRAPHIC_FIT",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_e1x_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "e1x_module",
        str(path.resolve()),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import E1x module from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_functions = [
        "load_targets",
        "load_token_map",
        "load_cpv2_candidates",
        "create_supplier_universe",
        "load_hierarchical_sparse_index",
        "load_simple_count_lookup",
        "load_buyer_lookup",
        "load_regime_lookup",
        "make_case_features",
        "normalize_feature_matrix",
        "group_distance_contributions",
        "direct_winner_rank",
        "safe_percentile",
    ]

    missing = [
        name
        for name in required_functions
        if not hasattr(module, name)
    ]

    if missing:
        raise ImportError(
            "E1x module is missing required functions:\n  - "
            + "\n  - ".join(missing)
        )

    return module


def scenario_scores(
    contributions: dict[
        str,
        tuple[np.ndarray, np.ndarray, int]
    ],
    scenario_weights: dict[str, float],
) -> np.ndarray:
    """
    Combine group distance contributions using arbitrary group weights.

    The E1x group contribution already contains equal weighting among active
    features within that group. Here we renormalize scenario weights over
    active groups only.
    """
    active_groups = [
        group_name
        for group_name, weight in scenario_weights.items()
        if (
            weight > 0
            and contributions[group_name][2] > 0
        )
    ]

    if not active_groups:
        first = next(iter(contributions.values()))
        return np.full(
            len(first[0]),
            0.5,
            dtype=np.float64,
        )

    weight_sum = sum(
        scenario_weights[g]
        for g in active_groups
    )

    normalized_weights = {
        g: scenario_weights[g] / weight_sum
        for g in active_groups
    }

    n = len(
        contributions[active_groups[0]][0]
    )

    d_best_sq = np.zeros(
        n,
        dtype=np.float64,
    )

    d_worst_sq = np.zeros(
        n,
        dtype=np.float64,
    )

    for group_name in active_groups:
        best_sq, worst_sq, _ = contributions[group_name]
        w = normalized_weights[group_name]

        d_best_sq += (w * w) * best_sq
        d_worst_sq += (w * w) * worst_sq

    d_best = np.sqrt(d_best_sq)
    d_worst = np.sqrt(d_worst_sq)

    denom = d_best + d_worst

    scores = np.full(
        n,
        0.5,
        dtype=np.float64,
    )

    valid = denom > 0

    scores[valid] = (
        d_worst[valid]
        / denom[valid]
    )

    return scores


def group_only_scores(
    contribution: tuple[
        np.ndarray,
        np.ndarray,
        int,
    ],
) -> np.ndarray:
    best_sq, worst_sq, n_features = contribution

    if n_features <= 0:
        return np.full(
            len(best_sq),
            np.nan,
            dtype=np.float64,
        )

    d_best = np.sqrt(best_sq)
    d_worst = np.sqrt(worst_sq)
    denom = d_best + d_worst

    scores = np.full(
        len(best_sq),
        0.5,
        dtype=np.float64,
    )

    valid = denom > 0

    scores[valid] = (
        d_worst[valid]
        / denom[valid]
    )

    return scores


def topk_indices(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Return an unordered exact-size top-k index set using argpartition.

    Top-k overlap is therefore a fixed-k composition measure. Very rare exact
    boundary ties may be resolved arbitrarily by NumPy; winner ranks retain the
    tie-aware midrank treatment from E1/E1x.
    """
    n = len(scores)

    if n == 0:
        return np.empty(
            0,
            dtype=np.int32,
        )

    k_eff = min(
        int(k),
        n,
    )

    if k_eff == n:
        return np.arange(
            n,
            dtype=np.int32,
        )

    idx = np.argpartition(
        -scores,
        k_eff - 1,
    )[:k_eff]

    return idx.astype(
        np.int32,
        copy=False,
    )


def set_overlap_ratio(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0

    denom = min(
        len(a),
        len(b),
    )

    if denom == 0:
        return 0.0

    return (
        len(
            set(a.tolist())
            & set(b.tolist())
        )
        / denom
    )


def pearson_score_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if len(a) <= 1:
        return 1.0

    a_std = float(np.std(a))
    b_std = float(np.std(b))

    if a_std == 0.0 and b_std == 0.0:
        return 1.0

    if a_std == 0.0 or b_std == 0.0:
        return 0.0

    return float(
        np.corrcoef(a, b)[0, 1]
    )


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_sensitivity(
    e1x,
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    cpv_token_map,
    nuts_token_map,
    supplier_to_idx: pd.Series,
    cpv_lookup,
    nuts_lookup,
    country_lookup,
    buyer_lookup,
    regime_match_lookup,
    regime_total_lookup,
    criterion_lookup,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Returns:
      case-scenario results
      pairwise stability case records
      directional responsiveness case records
    """
    candidate_groups = {
        str(context): g.reset_index(drop=True)
        for context, g in candidates.groupby(
            "CONTEXT_KEY",
            sort=False,
            observed=True,
        )
    }

    target_groups = list(
        targets.groupby(
            "TARGET_CPV2",
            sort=False,
            observed=True,
        )
    )

    global_to_local = np.full(
        len(supplier_to_idx),
        -1,
        dtype=np.int32,
    )

    result_records: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []
    responsiveness_records: list[dict[str, Any]] = []

    total_contexts = len(target_groups)

    for context_no, (
        context_key,
        target_context,
    ) in enumerate(
        target_groups,
        start=1,
    ):
        context_key = str(context_key)

        cands = candidate_groups.get(context_key)

        if cands is None or cands.empty:
            raise ValueError(
                f"Missing candidate context {context_key}."
            )

        candidate_ids = (
            cands[
                "SUPPLIER_ENTITY_ID"
            ]
            .astype(str)
            .to_numpy()
        )

        candidate_global_idx = (
            pd.Series(candidate_ids)
            .map(supplier_to_idx)
            .to_numpy(dtype=np.int32)
        )

        n_candidates = len(cands)

        global_to_local[
            candidate_global_idx
        ] = np.arange(
            n_candidates,
            dtype=np.int32,
        )

        print(
            f"[{context_no}/{total_contexts}] "
            f"CPV2={context_key}: "
            f"{len(target_context):,} cases, "
            f"{n_candidates:,} candidates"
        )

        for case_no, (_, target_row) in enumerate(
            target_context.iterrows(),
            start=1,
        ):
            case_id = str(
                target_row["TARGET_CASE_ID"]
            )

            winner_id = str(
                target_row[
                    "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"
                ]
            )

            winner_global = supplier_to_idx.get(
                winner_id,
                None,
            )

            if winner_global is None:
                raise ValueError(
                    f"Winner {winner_id} not in supplier universe."
                )

            winner_local = int(
                global_to_local[
                    int(winner_global)
                ]
            )

            if winner_local < 0:
                raise ValueError(
                    f"Winner {winner_id} absent from context {context_key}."
                )

            cpv_levels = cpv_token_map.get(
                case_id,
                {},
            )

            nuts_levels = nuts_token_map.get(
                case_id,
                {},
            )

            x, active_mask = e1x.make_case_features(
                candidates=cands,
                target_row=target_row,
                cpv_levels=cpv_levels,
                nuts_levels=nuts_levels,
                global_to_local=global_to_local,
                cpv_lookup=cpv_lookup,
                nuts_lookup=nuts_lookup,
                country_lookup=country_lookup,
                buyer_lookup=buyer_lookup,
                regime_match_lookup=regime_match_lookup,
                regime_total_lookup=regime_total_lookup,
                criterion_lookup=criterion_lookup,
            )

            normalized, nonzero_mask = (
                e1x.normalize_feature_matrix(x)
            )

            contributions = (
                e1x.group_distance_contributions(
                    normalized,
                    active_mask,
                    nonzero_mask,
                )
            )

            score_map = {
                scenario: scenario_scores(
                    contributions,
                    weights,
                )
                for scenario, weights in SCENARIOS.items()
            }

            top10_map = {
                scenario: topk_indices(
                    scores,
                    10,
                )
                for scenario, scores in score_map.items()
            }

            top50_map = {
                scenario: topk_indices(
                    scores,
                    50,
                )
                for scenario, scores in score_map.items()
            }

            # Case-scenario evaluation records.
            for scenario, scores in score_map.items():
                (
                    midrank,
                    best_rank,
                    worst_rank,
                    tie_size,
                ) = e1x.direct_winner_rank(
                    scores,
                    winner_local,
                )

                result_records.append(
                    {
                        "TARGET_CASE_ID":
                            case_id,
                        "TARGET_CPV2":
                            context_key,
                        "SCENARIO":
                            scenario,
                        "POOL_DEFINITION":
                            "CPV2_MIN1",
                        "POOL_SIZE":
                            n_candidates,
                        "WINNER_RANK_MID":
                            midrank,
                        "WINNER_RANK_BEST":
                            best_rank,
                        "WINNER_RANK_WORST":
                            worst_rank,
                        "TIE_GROUP_SIZE":
                            tie_size,
                        "RECIPROCAL_RANK":
                            1.0 / midrank,
                        "PERCENTILE_RANK":
                            e1x.safe_percentile(
                                midrank,
                                n_candidates,
                            ),
                        "WINNER_AT_1":
                            midrank <= 1,
                        "WINNER_AT_5":
                            midrank <= 5,
                        "WINNER_AT_10":
                            midrank <= 10,
                        "WINNER_AT_50":
                            midrank <= 50,
                    }
                )

            # Stability relative to Balanced.
            base_scores = score_map["BALANCED"]

            for scenario in [
                "CONTINUITY",
                "DIVERSIFICATION",
            ]:
                scenario_scores_arr = score_map[scenario]

                stability_records.append(
                    {
                        "TARGET_CASE_ID":
                            case_id,
                        "COMPARISON":
                            f"BALANCED__{scenario}",
                        "SCORE_PEARSON_CORRELATION":
                            pearson_score_correlation(
                                base_scores,
                                scenario_scores_arr,
                            ),
                        "TOP10_OVERLAP":
                            set_overlap_ratio(
                                top10_map["BALANCED"],
                                top10_map[scenario],
                            ),
                        "TOP50_OVERLAP":
                            set_overlap_ratio(
                                top50_map["BALANCED"],
                                top50_map[scenario],
                            ),
                    }
                )

                # Directional responsiveness at Top-10.
                focus_group = FOCUS_GROUP[scenario]

                group_scores = group_only_scores(
                    contributions[focus_group]
                )

                base_set = set(
                    top10_map["BALANCED"].tolist()
                )

                scenario_set = set(
                    top10_map[scenario].tolist()
                )

                entrants = list(
                    scenario_set - base_set
                )

                exits = list(
                    base_set - scenario_set
                )

                if (
                    entrants
                    and exits
                    and np.isfinite(
                        group_scores[
                            entrants + exits
                        ]
                    ).all()
                ):
                    delta = (
                        float(
                            np.mean(
                                group_scores[entrants]
                            )
                        )
                        - float(
                            np.mean(
                                group_scores[exits]
                            )
                        )
                    )

                    responsiveness_records.append(
                        {
                            "TARGET_CASE_ID":
                                case_id,
                            "SCENARIO":
                                scenario,
                            "FOCUS_GROUP":
                                focus_group,
                            "N_ENTRANTS":
                                len(entrants),
                            "N_EXITS":
                                len(exits),
                            "ENTRANT_MINUS_EXIT_FOCUS_SCORE":
                                delta,
                            "DIRECTIONALLY_POSITIVE":
                                delta > 0,
                        }
                    )

            if (
                case_no % 250 == 0
                or case_no
                == len(target_context)
            ):
                print(
                    f"    processed "
                    f"{case_no:,}/"
                    f"{len(target_context):,} cases"
                )

        global_to_local[
            candidate_global_idx
        ] = -1

        gc.collect()

    return (
        pd.DataFrame.from_records(
            result_records
        ),
        pd.DataFrame.from_records(
            stability_records
        ),
        pd.DataFrame.from_records(
            responsiveness_records
        ),
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def scenario_specification() -> pd.DataFrame:
    rows = []

    for scenario, weights in SCENARIOS.items():
        for group_name, weight in weights.items():
            rows.append(
                {
                    "SCENARIO":
                        scenario,
                    "CONCEPTUAL_GROUP":
                        group_name,
                    "GROUP_WEIGHT":
                        float(weight),
                    "FOCUS_GROUP":
                        FOCUS_GROUP.get(
                            scenario,
                            None,
                        ),
                }
            )

    return pd.DataFrame(rows)


def summarize_scenarios(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for scenario, g in results.groupby(
        "SCENARIO",
        sort=False,
    ):
        rows.append(
            {
                "SCENARIO":
                    scenario,
                "N_CASES":
                    len(g),
                "WINNER_AT_1_PCT":
                    100.0
                    * g["WINNER_AT_1"].mean(),
                "WINNER_AT_5_PCT":
                    100.0
                    * g["WINNER_AT_5"].mean(),
                "WINNER_AT_10_PCT":
                    100.0
                    * g["WINNER_AT_10"].mean(),
                "WINNER_AT_50_PCT":
                    100.0
                    * g["WINNER_AT_50"].mean(),
                "MRR":
                    g[
                        "RECIPROCAL_RANK"
                    ].mean(),
                "MEAN_PERCENTILE_RANK":
                    g[
                        "PERCENTILE_RANK"
                    ].mean(),
                "MEDIAN_WINNER_RANK":
                    g[
                        "WINNER_RANK_MID"
                    ].median(),
            }
        )

    order = {
        "BALANCED": 0,
        "CONTINUITY": 1,
        "DIVERSIFICATION": 2,
    }

    out = pd.DataFrame(rows)
    out["_order"] = out["SCENARIO"].map(order)

    return (
        out.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def summarize_stability(
    results: pd.DataFrame,
    stability_cases: pd.DataFrame,
) -> pd.DataFrame:
    rank_pivot = results.pivot(
        index="TARGET_CASE_ID",
        columns="SCENARIO",
        values="WINNER_RANK_MID",
    )

    rows = []

    for scenario in [
        "CONTINUITY",
        "DIVERSIFICATION",
    ]:
        delta = (
            rank_pivot[scenario]
            - rank_pivot["BALANCED"]
        )

        comparison_name = (
            f"BALANCED__{scenario}"
        )

        g = stability_cases.loc[
            stability_cases["COMPARISON"]
            == comparison_name
        ]

        rows.append(
            {
                "REFERENCE_SCENARIO":
                    "BALANCED",
                "COMPARISON_SCENARIO":
                    scenario,
                "N_CASES":
                    len(delta),
                "MEAN_SCORE_PEARSON_CORRELATION":
                    g[
                        "SCORE_PEARSON_CORRELATION"
                    ].mean(),
                "MEAN_TOP10_OVERLAP":
                    g[
                        "TOP10_OVERLAP"
                    ].mean(),
                "MEAN_TOP50_OVERLAP":
                    g[
                        "TOP50_OVERLAP"
                    ].mean(),
                "MEAN_WINNER_RANK_CHANGE":
                    delta.mean(),
                "MEDIAN_WINNER_RANK_CHANGE":
                    delta.median(),
                "MEAN_ABS_WINNER_RANK_CHANGE":
                    delta.abs().mean(),
                "PCT_WINNER_MOVED_UP":
                    100.0
                    * (delta < 0).mean(),
                "PCT_WINNER_UNCHANGED":
                    100.0
                    * (delta == 0).mean(),
                "PCT_WINNER_MOVED_DOWN":
                    100.0
                    * (delta > 0).mean(),
            }
        )

    return pd.DataFrame(rows)


def summarize_responsiveness(
    responsiveness_cases: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        scenario,
        focus_group,
    ), g in responsiveness_cases.groupby(
        [
            "SCENARIO",
            "FOCUS_GROUP",
        ],
        sort=False,
        observed=True,
    ):
        rows.append(
            {
                "SCENARIO":
                    scenario,
                "FOCUS_GROUP":
                    focus_group,
                "N_CASES_WITH_TOP10_TURNOVER":
                    len(g),
                "MEAN_ENTRANT_MINUS_EXIT_FOCUS_SCORE":
                    g[
                        "ENTRANT_MINUS_EXIT_FOCUS_SCORE"
                    ].mean(),
                "MEDIAN_ENTRANT_MINUS_EXIT_FOCUS_SCORE":
                    g[
                        "ENTRANT_MINUS_EXIT_FOCUS_SCORE"
                    ].median(),
                "PCT_CASES_DIRECTIONALLY_POSITIVE":
                    100.0
                    * g[
                        "DIRECTIONALLY_POSITIVE"
                    ].mean(),
            }
        )

    return pd.DataFrame(rows)


def validation_checks(
    results: pd.DataFrame,
    e1x_case_results_path: Path,
) -> pd.DataFrame:
    rows = []

    counts = (
        results.groupby(
            "TARGET_CASE_ID"
        )["SCENARIO"]
        .nunique()
    )

    rows.append(
        {
            "CHECK":
                "three_scenarios_per_case",
            "VALUE":
                bool(
                    (counts == 3).all()
                ),
            "DETAIL":
                (
                    f"cases={counts.size}; "
                    f"min={counts.min()}; "
                    f"max={counts.max()}"
                ),
        }
    )

    dup_count = int(
        results.duplicated(
            [
                "TARGET_CASE_ID",
                "SCENARIO",
            ]
        ).sum()
    )

    rows.append(
        {
            "CHECK":
                "unique_case_scenario_rows",
            "VALUE":
                dup_count == 0,
            "DETAIL":
                f"duplicate_rows={dup_count}",
        }
    )

    scenario_counts = (
        results.groupby(
            "SCENARIO"
        )["TARGET_CASE_ID"]
        .nunique()
        .to_dict()
    )

    rows.append(
        {
            "CHECK":
                "equal_case_count_across_scenarios",
            "VALUE":
                len(
                    set(
                        scenario_counts.values()
                    )
                ) == 1,
            "DETAIL":
                str(scenario_counts),
        }
    )

    # Balanced must exactly reproduce E1x M3_GEO_BUYER.
    e1x = pd.read_parquet(
        e1x_case_results_path,
        columns=[
            "TARGET_CASE_ID",
            "MODEL",
            "WINNER_RANK_MID",
        ],
    )

    e1x = e1x.loc[
        e1x["MODEL"]
        == "M3_GEO_BUYER",
        [
            "TARGET_CASE_ID",
            "WINNER_RANK_MID",
        ],
    ].rename(
        columns={
            "WINNER_RANK_MID":
                "E1X_M3_RANK",
        }
    )

    e1x["TARGET_CASE_ID"] = (
        e1x["TARGET_CASE_ID"]
        .astype("string")
    )

    balanced = results.loc[
        results["SCENARIO"]
        == "BALANCED",
        [
            "TARGET_CASE_ID",
            "WINNER_RANK_MID",
        ],
    ].rename(
        columns={
            "WINNER_RANK_MID":
                "E2X_BALANCED_RANK",
        }
    )

    balanced["TARGET_CASE_ID"] = (
        balanced["TARGET_CASE_ID"]
        .astype("string")
    )

    merged = e1x.merge(
        balanced,
        on="TARGET_CASE_ID",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    ids_match = bool(
        (
            merged["_merge"]
            == "both"
        ).all()
    )

    ranks_match = bool(
        np.allclose(
            merged.loc[
                merged["_merge"]
                == "both",
                "E1X_M3_RANK",
            ],
            merged.loc[
                merged["_merge"]
                == "both",
                "E2X_BALANCED_RANK",
            ],
            rtol=0,
            atol=1e-9,
        )
    )

    rows.append(
        {
            "CHECK":
                "balanced_case_ids_match_e1x_m3",
            "VALUE":
                ids_match,
            "DETAIL":
                str(
                    merged[
                        "_merge"
                    ]
                    .value_counts()
                    .to_dict()
                ),
        }
    )

    rows.append(
        {
            "CHECK":
                "balanced_ranks_match_e1x_m3",
            "VALUE":
                ranks_match,
            "DETAIL":
                (
                    f"matched_cases="
                    f"{int((merged['_merge'] == 'both').sum())}"
                ),
        }
    )

    # Scenario weights.
    for scenario, weights in SCENARIOS.items():
        weight_sum = sum(weights.values())

        rows.append(
            {
                "CHECK":
                    f"{scenario.lower()}_weights_sum_to_one",
                "VALUE":
                    abs(weight_sum - 1.0) < 1e-12,
                "DETAIL":
                    f"sum={weight_sum:.12f}",
            }
        )

    return pd.DataFrame(rows)


def write_report(
    scenario_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    responsiveness_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# E2x Scenario Sensitivity on the Frozen M3 Backbone",
        "",
        "## Scope",
        "",
        "E2x keeps the E1x M3 feature backbone and candidate sets fixed. "
        "Only conceptual-group weights change across Balanced, Continuity, "
        "and Diversification scenarios.",
        "",
        "## Scenario performance",
        "",
        "| Scenario | N | W@1 (%) | W@5 (%) | W@10 (%) | W@50 (%) | MRR | Mean percentile | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in scenario_summary.iterrows():
        lines.append(
            f"| `{row['SCENARIO']}` "
            f"| {int(row['N_CASES'])} "
            f"| {row['WINNER_AT_1_PCT']:.3f} "
            f"| {row['WINNER_AT_5_PCT']:.3f} "
            f"| {row['WINNER_AT_10_PCT']:.3f} "
            f"| {row['WINNER_AT_50_PCT']:.3f} "
            f"| {row['MRR']:.6f} "
            f"| {row['MEAN_PERCENTILE_RANK']:.6f} "
            f"| {row['MEDIAN_WINNER_RANK']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Stability relative to Balanced",
            "",
            "| Scenario | Score correlation | Top10 overlap | Top50 overlap | Mean abs winner-rank change | % winner up | % unchanged | % down |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in stability_summary.iterrows():
        lines.append(
            f"| `{row['COMPARISON_SCENARIO']}` "
            f"| {row['MEAN_SCORE_PEARSON_CORRELATION']:.6f} "
            f"| {row['MEAN_TOP10_OVERLAP']:.6f} "
            f"| {row['MEAN_TOP50_OVERLAP']:.6f} "
            f"| {row['MEAN_ABS_WINNER_RANK_CHANGE']:.3f} "
            f"| {row['PCT_WINNER_MOVED_UP']:.3f} "
            f"| {row['PCT_WINNER_UNCHANGED']:.3f} "
            f"| {row['PCT_WINNER_MOVED_DOWN']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Directional responsiveness",
            "",
            "| Scenario | Focus group | Cases with Top10 turnover | Mean entrant-minus-exit focus score | % directionally positive |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for _, row in responsiveness_summary.iterrows():
        lines.append(
            f"| `{row['SCENARIO']}` "
            f"| `{row['FOCUS_GROUP']}` "
            f"| {int(row['N_CASES_WITH_TOP10_TURNOVER'])} "
            f"| {row['MEAN_ENTRANT_MINUS_EXIT_FOCUS_SCORE']:.6f} "
            f"| {row['PCT_CASES_DIRECTIONALLY_POSITIVE']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
        ]
    )

    for _, row in validation_df.iterrows():
        status = (
            "PASS"
            if bool(row["VALUE"])
            else "FAIL"
        )

        lines.append(
            f"- **{status}** `{row['CHECK']}`: {row['DETAIL']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment evaluates scenario responsiveness rather than selecting "
            "a new predictive model. The structured feature backbone is fixed to "
            "E1x M3_GEO_BUYER. Meaningful evidence consists of ranking stability "
            "combined with interpretable top-k turnover and directional movement "
            "toward the scenario-emphasized conceptual group.",
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
        description="Run revised E2 scenario sensitivity on E1x M3."
    )

    parser.add_argument(
        "--e1x-module",
        required=True,
        type=Path,
        help=(
            "Path to run_e1x_extended_procurement_fit_rankings_v2.py"
        ),
    )

    parser.add_argument(
        "--extended-index-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--base-fit-index-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--target-case-features",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--cpv2-features",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--e1x-case-results",
        required=True,
        type=Path,
        help="e1x_case_model_results.parquet",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/e2x_scenarios"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "reports/e2x_scenarios"
        ),
    )

    args = parser.parse_args()

    require_file(args.e1x_module)
    require_file(args.target_case_features)
    require_file(args.cpv2_features)
    require_file(args.e1x_case_results)

    extended = args.extended_index_dir
    base_fit = args.base_fit_index_dir

    required_extended_files = [
        "e1x_target_profiles_2017.parquet",
        "e1x_target_cpv_tokens_2017.parquet",
        "e1x_target_nuts_tokens_2017.parquet",
        "supplier_cpv_fit_counts_2015_2016.parquet",
        "supplier_nuts_fit_counts_2015_2016.parquet",
        "supplier_buyer_relationships_2015_2016.parquet",
        "supplier_regime_fit_counts_2015_2016.parquet",
        "supplier_criterion_code_fit_counts_2015_2016.parquet",
        "supplier_cpv2_context_descriptors_2015_2016.parquet",
    ]

    for filename in required_extended_files:
        require_file(
            extended / filename
        )

    country_path = (
        base_fit
        / "supplier_country_fit_counts_2015_2016.parquet"
    )

    require_file(country_path)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1/10 Importing validated E1x feature code...")
    e1x = load_e1x_module(
        args.e1x_module
    )

    print("2/10 Loading target profiles and outcomes...")
    targets = e1x.load_targets(
        extended
        / "e1x_target_profiles_2017.parquet",
        args.target_case_features,
    )

    print(
        f"    cases: {len(targets):,}"
    )

    print("3/10 Loading target CPV/NUTS token maps...")
    cpv_token_map = e1x.load_token_map(
        extended
        / "e1x_target_cpv_tokens_2017.parquet",
        [
            "CPV_CODE",
            "CPV3",
            "CPV4",
        ],
    )

    nuts_token_map = e1x.load_token_map(
        extended
        / "e1x_target_nuts_tokens_2017.parquet",
        [
            "NUTS1",
            "NUTS2",
            "NUTS3",
        ],
    )

    print("4/10 Loading candidate features...")
    candidates = e1x.load_cpv2_candidates(
        args.cpv2_features,
        extended
        / "supplier_cpv2_context_descriptors_2015_2016.parquet",
    )

    print(
        f"    candidate-context rows: {len(candidates):,}"
    )

    print("5/10 Building supplier universe...")
    (
        supplier_array,
        supplier_to_idx,
    ) = e1x.create_supplier_universe(
        candidates
    )

    print(
        f"    suppliers: {len(supplier_array):,}"
    )

    print("6/10 Loading sparse fit indices...")

    cpv_lookup = (
        e1x.load_hierarchical_sparse_index(
            extended
            / "supplier_cpv_fit_counts_2015_2016.parquet",
            level_col="CPV_LEVEL",
            needed_levels={
                "EXACT",
                "CPV4",
                "CPV3",
            },
            supplier_to_idx=supplier_to_idx,
        )
    )

    nuts_lookup = (
        e1x.load_hierarchical_sparse_index(
            extended
            / "supplier_nuts_fit_counts_2015_2016.parquet",
            level_col="NUTS_LEVEL",
            needed_levels={
                "NUTS1",
                "NUTS2",
                "NUTS3",
            },
            supplier_to_idx=supplier_to_idx,
        )
    )

    country_lookup = (
        e1x.load_simple_count_lookup(
            country_path,
            count_col="N_COUNTRY_AWARDS",
            supplier_to_idx=supplier_to_idx,
        )
    )

    buyer_lookup = (
        e1x.load_buyer_lookup(
            extended
            / "supplier_buyer_relationships_2015_2016.parquet",
            supplier_to_idx,
        )
    )

    (
        regime_match_lookup,
        regime_total_lookup,
    ) = e1x.load_regime_lookup(
        extended
        / "supplier_regime_fit_counts_2015_2016.parquet",
        supplier_to_idx,
    )

    criterion_lookup = (
        e1x.load_simple_count_lookup(
            extended
            / "supplier_criterion_code_fit_counts_2015_2016.parquet",
            count_col="N_MATCHED_AWARDS",
            supplier_to_idx=supplier_to_idx,
        )
    )

    print("7/10 Running scenario sensitivity...")
    (
        results,
        stability_cases,
        responsiveness_cases,
    ) = run_sensitivity(
        e1x=e1x,
        targets=targets,
        candidates=candidates,
        cpv_token_map=cpv_token_map,
        nuts_token_map=nuts_token_map,
        supplier_to_idx=supplier_to_idx,
        cpv_lookup=cpv_lookup,
        nuts_lookup=nuts_lookup,
        country_lookup=country_lookup,
        buyer_lookup=buyer_lookup,
        regime_match_lookup=regime_match_lookup,
        regime_total_lookup=regime_total_lookup,
        criterion_lookup=criterion_lookup,
    )

    print("8/10 Summarizing scenario results...")
    specification_df = (
        scenario_specification()
    )

    summary_df = (
        summarize_scenarios(
            results
        )
    )

    stability_summary_df = (
        summarize_stability(
            results,
            stability_cases,
        )
    )

    responsiveness_summary_df = (
        summarize_responsiveness(
            responsiveness_cases
        )
    )

    print("9/10 Running validation...")
    validation_df = validation_checks(
        results,
        args.e1x_case_results,
    )

    print("10/10 Writing outputs...")
    results.to_parquet(
        args.output_dir
        / "e2x_case_scenario_results.parquet",
        index=False,
    )

    specification_df.to_csv(
        args.report_dir
        / "e2x_scenario_specification.csv",
        index=False,
    )

    summary_df.to_csv(
        args.report_dir
        / "e2x_scenario_summary.csv",
        index=False,
    )

    stability_summary_df.to_csv(
        args.report_dir
        / "e2x_pairwise_stability.csv",
        index=False,
    )

    responsiveness_summary_df.to_csv(
        args.report_dir
        / "e2x_directional_responsiveness.csv",
        index=False,
    )

    validation_df.to_csv(
        args.report_dir
        / "e2x_validation_checks.csv",
        index=False,
    )

    write_report(
        summary_df,
        stability_summary_df,
        responsiveness_summary_df,
        validation_df,
        args.report_dir
        / "e2x_scenario_report.md",
    )

    print()
    print("E2x scenario sensitivity complete.")
    print("Review first:")
    print("  e2x_scenario_report.md")
    print("  e2x_scenario_summary.csv")
    print("  e2x_pairwise_stability.csv")
    print("  e2x_directional_responsiveness.csv")
    print("  e2x_validation_checks.csv")


if __name__ == "__main__":
    main()
