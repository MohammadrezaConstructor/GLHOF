#!/usr/bin/env python3
"""
Refine an already completed E1 run without rerunning supplier rankings.

Adds:
- analytical random-ranking expectations by pool,
- lift versus random expectation,
- corrected validation restricted to evaluated pools,
- refined E1 report.

Inputs:
- e1_case_ranking_results.parquet
- candidate_pool_case_summary_2017.parquet

Outputs:
- e1_method_summary_refined.csv
- e1_random_expectation_summary.csv
- e1_method_lift_vs_random.csv
- e1_validation_checks_refined.csv
- e1_ranking_report_refined.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("FREQUENCY", "WSM", "TOPSIS", "VIKOR")


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

    pool_order = {
        "CPV2_MIN1": 1,
        "CPV3_MIN1": 2,
        "CPV2_MIN2": 3,
    }
    method_order = {
        "FREQUENCY": 1,
        "WSM": 2,
        "TOPSIS": 3,
        "VIKOR": 4,
    }

    out["_pool"] = out["POOL_DEFINITION"].map(pool_order)
    out["_method"] = out["METHOD"].map(method_order)

    return (
        out.sort_values(["_pool", "_method"])
        .drop(columns=["_pool", "_method"])
        .reset_index(drop=True)
    )


def random_expectation_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    case_pools = (
        results[
            [
                "POOL_DEFINITION",
                "TARGET_CASE_ID",
                "POOL_SIZE",
            ]
        ]
        .drop_duplicates(
            subset=[
                "POOL_DEFINITION",
                "TARGET_CASE_ID",
            ]
        )
        .copy()
    )

    max_n = int(case_pools["POOL_SIZE"].max())

    harmonic = np.zeros(max_n + 1, dtype=np.float64)
    harmonic[1:] = np.cumsum(
        1.0 / np.arange(1, max_n + 1, dtype=np.float64)
    )

    rows = []

    for pool, g in case_pools.groupby(
        "POOL_DEFINITION",
        sort=False,
    ):
        n = g["POOL_SIZE"].to_numpy(dtype=np.int64)

        if np.any(n <= 0):
            raise ValueError(
                f"Non-positive candidate-pool size found in {pool}."
            )

        rows.append(
            {
                "POOL_DEFINITION": pool,
                "METHOD": "RANDOM_EXPECTATION",
                "N_CASES": len(g),
                "WINNER_AT_1_PCT":
                    100.0 * np.mean(np.minimum(1.0, 1.0 / n)),
                "WINNER_AT_5_PCT":
                    100.0 * np.mean(np.minimum(1.0, 5.0 / n)),
                "WINNER_AT_10_PCT":
                    100.0 * np.mean(np.minimum(1.0, 10.0 / n)),
                "WINNER_AT_50_PCT":
                    100.0 * np.mean(np.minimum(1.0, 50.0 / n)),
                "MRR":
                    np.mean(harmonic[n] / n),
                "MEAN_PERCENTILE_RANK":
                    np.mean(np.where(n == 1, 1.0, 0.5)),
                "MEDIAN_POOL_SIZE":
                    np.median(n),
            }
        )

    return pd.DataFrame(rows)


def method_lift_vs_random(
    method_df: pd.DataFrame,
    random_df: pd.DataFrame,
) -> pd.DataFrame:
    r = random_df[
        [
            "POOL_DEFINITION",
            "WINNER_AT_1_PCT",
            "WINNER_AT_5_PCT",
            "WINNER_AT_10_PCT",
            "WINNER_AT_50_PCT",
            "MRR",
        ]
    ].rename(
        columns={
            "WINNER_AT_1_PCT": "RANDOM_WINNER_AT_1_PCT",
            "WINNER_AT_5_PCT": "RANDOM_WINNER_AT_5_PCT",
            "WINNER_AT_10_PCT": "RANDOM_WINNER_AT_10_PCT",
            "WINNER_AT_50_PCT": "RANDOM_WINNER_AT_50_PCT",
            "MRR": "RANDOM_MRR",
        }
    )

    out = method_df.merge(
        r,
        on="POOL_DEFINITION",
        how="left",
        validate="many_to_one",
    )

    out["LIFT_W1_VS_RANDOM"] = (
        out["WINNER_AT_1_PCT"]
        / out["RANDOM_WINNER_AT_1_PCT"]
    )
    out["LIFT_W5_VS_RANDOM"] = (
        out["WINNER_AT_5_PCT"]
        / out["RANDOM_WINNER_AT_5_PCT"]
    )
    out["LIFT_W10_VS_RANDOM"] = (
        out["WINNER_AT_10_PCT"]
        / out["RANDOM_WINNER_AT_10_PCT"]
    )
    out["LIFT_W50_VS_RANDOM"] = (
        out["WINNER_AT_50_PCT"]
        / out["RANDOM_WINNER_AT_50_PCT"]
    )
    out["LIFT_MRR_VS_RANDOM"] = (
        out["MRR"]
        / out["RANDOM_MRR"]
    )

    return out


def validation_checks(
    results: pd.DataFrame,
    candidate_pool_summary: Path,
) -> pd.DataFrame:
    rows = []

    evaluated_pools = set(
        results["POOL_DEFINITION"].dropna().unique().tolist()
    )

    for pool, g in results.groupby("POOL_DEFINITION"):
        counts = g.groupby("TARGET_CASE_ID")["METHOD"].nunique()

        rows.append(
            {
                "CHECK": f"{pool}__four_methods_per_case",
                "VALUE": bool((counts == len(METHODS)).all()),
                "DETAIL":
                    f"cases={counts.size}; min_methods={counts.min()}; "
                    f"max_methods={counts.max()}",
            }
        )

        duplicate_rows = int(
            g.duplicated(
                ["TARGET_CASE_ID", "METHOD"]
            ).sum()
        )

        rows.append(
            {
                "CHECK": f"{pool}__unique_case_method_rows",
                "VALUE": duplicate_rows == 0,
                "DETAIL": f"duplicate_rows={duplicate_rows}",
            }
        )

    cps = pd.read_parquet(
        candidate_pool_summary,
        columns=[
            "TARGET_CASE_ID",
            "POOL_DEFINITION",
            "OBSERVED_WINNER_INCLUDED",
        ],
    )

    cps = cps.loc[
        cps["POOL_DEFINITION"].isin(evaluated_pools)
    ].copy()

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

    for pool in sorted(evaluated_pools):
        rows.append(
            {
                "CHECK":
                    f"{pool}__winner_in_pool_case_count_matches_preprocessing",
                "VALUE":
                    expected.get(pool) == observed.get(pool),
                "DETAIL":
                    f"expected={expected.get(pool)}; observed={observed.get(pool)}",
            }
        )

    return pd.DataFrame(rows)


def create_report(
    method_df: pd.DataFrame,
    random_df: pd.DataFrame,
    lift_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# E1 Deterministic Ranking Comparison Report — Refined",
        "",
        "## Scope",
        "",
        "E1 compares Frequency, WSM, TOPSIS, and VIKOR on cases whose observed "
        "winner is present in the corresponding candidate pool. CPV2_MIN1 is "
        "the primary analysis; CPV3_MIN1 and CPV2_MIN2 are sensitivity analyses.",
        "",
        "## Deterministic method results",
        "",
        "| Pool | Method | N | W@1 (%) | W@5 (%) | W@10 (%) | W@50 (%) | MRR | Mean percentile |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in method_df.iterrows():
        lines.append(
            f"| `{row['POOL_DEFINITION']}` | `{row['METHOD']}` "
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
        "## Analytical random-ranking expectation",
        "",
        "The random reference is computed analytically from each case-specific "
        "candidate-pool size; no simulation is used.",
        "",
        "| Pool | N | E[W@1] (%) | E[W@5] (%) | E[W@10] (%) | E[W@50] (%) | E[MRR] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for _, row in random_df.iterrows():
        lines.append(
            f"| `{row['POOL_DEFINITION']}` "
            f"| {int(row['N_CASES'])} "
            f"| {row['WINNER_AT_1_PCT']:.6f} "
            f"| {row['WINNER_AT_5_PCT']:.6f} "
            f"| {row['WINNER_AT_10_PCT']:.6f} "
            f"| {row['WINNER_AT_50_PCT']:.6f} "
            f"| {row['MRR']:.8f} |"
        )

    lines.extend([
        "",
        "## Lift versus random expectation",
        "",
        "| Pool | Method | W@10 lift | W@50 lift | MRR lift |",
        "|---|---|---:|---:|---:|",
    ])

    for _, row in lift_df.iterrows():
        lines.append(
            f"| `{row['POOL_DEFINITION']}` | `{row['METHOD']}` "
            f"| {row['LIFT_W10_VS_RANDOM']:.2f}x "
            f"| {row['LIFT_W50_VS_RANDOM']:.2f}x "
            f"| {row['LIFT_MRR_VS_RANDOM']:.2f}x |"
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
        "The observed winner is a reference procurement outcome, not proof of the "
        "uniquely optimal supplier. E1 therefore measures alignment between "
        "historical ranking rules and observed award outcomes within the candidate "
        "sets. The random-ranking benchmark is an analytical interpretive reference.",
        "",
    ])

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case-results",
        required=True,
        type=Path,
        help="Path to e1_case_ranking_results.parquet",
    )
    parser.add_argument(
        "--candidate-pool-summary",
        required=True,
        type=Path,
        help="Path to candidate_pool_case_summary_2017.parquet",
    )
    parser.add_argument(
        "--report-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    for path in [
        args.case_results,
        args.candidate_pool_summary,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.report_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_parquet(args.case_results)

    method_df = summarize_methods(results)
    random_df = random_expectation_summary(results)
    lift_df = method_lift_vs_random(method_df, random_df)
    validation_df = validation_checks(
        results,
        args.candidate_pool_summary,
    )

    method_df.to_csv(
        args.report_dir / "e1_method_summary_refined.csv",
        index=False,
    )
    random_df.to_csv(
        args.report_dir / "e1_random_expectation_summary.csv",
        index=False,
    )
    lift_df.to_csv(
        args.report_dir / "e1_method_lift_vs_random.csv",
        index=False,
    )
    validation_df.to_csv(
        args.report_dir / "e1_validation_checks_refined.csv",
        index=False,
    )

    create_report(
        method_df,
        random_df,
        lift_df,
        validation_df,
        args.report_dir / "e1_ranking_report_refined.md",
    )

    print("Refined E1 reports written to:", args.report_dir.resolve())


if __name__ == "__main__":
    main()
