#!/usr/bin/env python3
"""
E2b bootstrap and comparison-set robustness for Reviewer 2.2.

This script uses existing case-level E1 and E2x outputs. It performs no API
calls and does not rebuild the supplier feature store.

To keep runtime bounded, it bootstraps:
- all E1 methods on the primary CPV2_MIN1 comparison set;
- TOPSIS across all available comparison-set definitions;
- all E2x scenarios.

Defaults match the supplied project tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_NAMES = [
    "W_AT_1_PCT",
    "W_AT_5_PCT",
    "W_AT_10_PCT",
    "W_AT_50_PCT",
    "MRR",
    "MEAN_PERCENTILE",
    "MEDIAN_RANK",
]


def read_parquet_checked(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def metric_values(rank: np.ndarray, pool: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(rank) & np.isfinite(pool) & (rank > 0) & (pool > 0)
    rank = rank[valid]
    pool = pool[valid]
    if rank.size == 0:
        return {name: float("nan") for name in METRIC_NAMES} | {"N_CASES": 0}
    percentile = 1.0 - ((rank - 1.0) / np.maximum(pool - 1.0, 1.0))
    return {
        "N_CASES": int(rank.size),
        "W_AT_1_PCT": 100.0 * float(np.mean(rank <= 1)),
        "W_AT_5_PCT": 100.0 * float(np.mean(rank <= 5)),
        "W_AT_10_PCT": 100.0 * float(np.mean(rank <= 10)),
        "W_AT_50_PCT": 100.0 * float(np.mean(rank <= 50)),
        "MRR": float(np.mean(1.0 / rank)),
        "MEAN_PERCENTILE": float(np.mean(percentile)),
        "MEDIAN_RANK": float(np.median(rank)),
    }


def bootstrap_ci(
    rank: np.ndarray,
    pool: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    valid = np.isfinite(rank) & np.isfinite(pool) & (rank > 0) & (pool > 0)
    rank = rank[valid]
    pool = pool[valid]
    if rank.size == 0:
        return {name: (float("nan"), float("nan")) for name in METRIC_NAMES}

    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_bootstrap, dtype=float) for name in METRIC_NAMES}
    n = rank.size
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values = metric_values(rank[idx], pool[idx])
        for name in METRIC_NAMES:
            draws[name][b] = values[name]

    return {
        name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for name, values in draws.items()
    }


def summarize_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for group_index, (key, group) in enumerate(
        df.groupby(grouper, sort=False, observed=True),
        start=1,
    ):
        key_tuple = (key,) if len(group_cols) == 1 else tuple(key)
        rank = pd.to_numeric(group["WINNER_RANK_MID"], errors="coerce").to_numpy(float)
        pool = pd.to_numeric(group["POOL_SIZE"], errors="coerce").to_numpy(float)
        row = dict(zip(group_cols, key_tuple))
        row.update(metric_values(rank, pool))
        intervals = bootstrap_ci(rank, pool, n_bootstrap, seed + group_index)
        for metric, (low, high) in intervals.items():
            row[f"{metric}_CI_LOW"] = low
            row[f"{metric}_CI_HIGH"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def select_e1_groups(e1: pd.DataFrame) -> pd.DataFrame:
    method = e1["METHOD"].astype(str).str.upper()
    pool = e1["POOL_DEFINITION"].astype(str)

    primary_methods = e1.loc[
        pool.eq("CPV2_MIN1")
        & method.isin({"FREQUENCY", "WSM", "TOPSIS", "VIKOR"})
    ]
    topsis_pools = e1.loc[method.eq("TOPSIS")]

    selected = pd.concat([primary_methods, topsis_pools], ignore_index=True)
    return selected.drop_duplicates(
        subset=["TARGET_CASE_ID", "POOL_DEFINITION", "METHOD"]
    )


def scenario_movements(e2: pd.DataFrame) -> pd.DataFrame:
    pivot = e2.pivot_table(
        index="TARGET_CASE_ID",
        columns="SCENARIO",
        values="WINNER_RANK_MID",
        aggfunc="first",
    )
    if "BALANCED" not in pivot.columns:
        raise ValueError("E2x results do not contain the BALANCED scenario.")

    rows = []
    for comparison in [c for c in pivot.columns if c != "BALANCED"]:
        pair = pivot[["BALANCED", comparison]].dropna()
        delta = pair[comparison] - pair["BALANCED"]
        rows.append({
            "REFERENCE_SCENARIO": "BALANCED",
            "COMPARISON_SCENARIO": comparison,
            "N_CASES": len(pair),
            "PEARSON_RANK_CORRELATION": pair.corr(method="pearson").iloc[0, 1],
            "SPEARMAN_RANK_CORRELATION": pair.corr(method="spearman").iloc[0, 1],
            "MEAN_RANK_CHANGE": float(delta.mean()),
            "MEAN_ABS_RANK_CHANGE": float(delta.abs().mean()),
            "MEDIAN_ABS_RANK_CHANGE": float(delta.abs().median()),
            "PCT_REFERENCE_MOVED_UP": 100.0 * float((delta < 0).mean()),
            "PCT_REFERENCE_UNCHANGED": 100.0 * float((delta == 0).mean()),
            "PCT_REFERENCE_MOVED_DOWN": 100.0 * float((delta > 0).mean()),
        })
    return pd.DataFrame(rows)


def write_report(
    e1_summary: pd.DataFrame,
    e2_summary: pd.DataFrame,
    movements: pd.DataFrame,
    output_path: Path,
    n_bootstrap: int,
) -> None:
    lines = [
        "# E2b Bootstrap and Comparison-Set Robustness",
        "",
        f"All 95% confidence intervals use {n_bootstrap} case-level bootstrap resamples.",
        "",
        "## E1 method and comparison-set sensitivity",
        "",
        "| Pool | Method | N | W@10 | 95% CI | W@50 | 95% CI | MRR | 95% CI |",
        "|---|---|---:|---:|---|---:|---|---:|---|",
    ]
    for _, row in e1_summary.iterrows():
        lines.append(
            f"| `{row['POOL_DEFINITION']}` | `{row['METHOD']}` | {int(row['N_CASES'])} "
            f"| {row['W_AT_10_PCT']:.3f} "
            f"| [{row['W_AT_10_PCT_CI_LOW']:.3f}, {row['W_AT_10_PCT_CI_HIGH']:.3f}] "
            f"| {row['W_AT_50_PCT']:.3f} "
            f"| [{row['W_AT_50_PCT_CI_LOW']:.3f}, {row['W_AT_50_PCT_CI_HIGH']:.3f}] "
            f"| {row['MRR']:.6f} "
            f"| [{row['MRR_CI_LOW']:.6f}, {row['MRR_CI_HIGH']:.6f}] |"
        )

    lines.extend([
        "",
        "## E2 scenario uncertainty",
        "",
        "| Scenario | N | W@10 | 95% CI | W@50 | 95% CI | MRR | 95% CI |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ])
    for _, row in e2_summary.iterrows():
        lines.append(
            f"| `{row['SCENARIO']}` | {int(row['N_CASES'])} "
            f"| {row['W_AT_10_PCT']:.3f} "
            f"| [{row['W_AT_10_PCT_CI_LOW']:.3f}, {row['W_AT_10_PCT_CI_HIGH']:.3f}] "
            f"| {row['W_AT_50_PCT']:.3f} "
            f"| [{row['W_AT_50_PCT_CI_LOW']:.3f}, {row['W_AT_50_PCT_CI_HIGH']:.3f}] "
            f"| {row['MRR']:.6f} "
            f"| [{row['MRR_CI_LOW']:.6f}, {row['MRR_CI_HIGH']:.6f}] |"
        )

    lines.extend([
        "",
        "## Reference-rank movement from BALANCED",
        "",
        "| Scenario | N | Spearman | Mean absolute rank change | % moved up | % unchanged | % moved down |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in movements.iterrows():
        lines.append(
            f"| `{row['COMPARISON_SCENARIO']}` | {int(row['N_CASES'])} "
            f"| {row['SPEARMAN_RANK_CORRELATION']:.4f} "
            f"| {row['MEAN_ABS_RANK_CHANGE']:.3f} "
            f"| {row['PCT_REFERENCE_MOVED_UP']:.3f} "
            f"| {row['PCT_REFERENCE_UNCHANGED']:.3f} "
            f"| {row['PCT_REFERENCE_MOVED_DOWN']:.3f} |"
        )

    lines.extend([
        "",
        "## Scope",
        "",
        "This analysis evaluates uncertainty and sensitivity of historical "
        "reference-outcome positioning. Historically active supplier comparison "
        "sets are not actual bidder pools, and the observed awardee is not treated "
        "as a verified optimal supplier.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.bootstrap < 50:
        raise ValueError("Use at least 50 bootstrap resamples.")

    root = args.root.resolve()
    output_dir = root / "reports/r22_e2b_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        rng = np.random.default_rng(7)
        ids = [f"T{i:04d}" for i in range(200)]
        e1_rows = []
        for pool_name, pool_size in [("CPV2_MIN1", 1000), ("CPV3_MIN1", 300)]:
            for method in ["FREQUENCY", "WSM", "TOPSIS", "VIKOR"]:
                for case_id in ids:
                    e1_rows.append({
                        "TARGET_CASE_ID": case_id,
                        "POOL_DEFINITION": pool_name,
                        "METHOD": method,
                        "POOL_SIZE": pool_size,
                        "WINNER_RANK_MID": int(rng.integers(1, pool_size + 1)),
                    })
        e2_rows = []
        for scenario in ["BALANCED", "CONTINUITY", "DIVERSIFICATION"]:
            for case_id in ids:
                e2_rows.append({
                    "TARGET_CASE_ID": case_id,
                    "SCENARIO": scenario,
                    "POOL_SIZE": 1000,
                    "WINNER_RANK_MID": int(rng.integers(1, 1001)),
                })
        e1 = pd.DataFrame(e1_rows)
        e2 = pd.DataFrame(e2_rows)
    else:
        e1 = read_parquet_checked(
            root / "data/analysis/e1_rankings/e1_case_ranking_results.parquet",
            {"TARGET_CASE_ID", "POOL_DEFINITION", "METHOD", "POOL_SIZE", "WINNER_RANK_MID"},
        )
        e2 = read_parquet_checked(
            root / "data/analysis/e2x_scenarios/e2x_case_scenario_results.parquet",
            {"TARGET_CASE_ID", "SCENARIO", "POOL_SIZE", "WINNER_RANK_MID"},
        )

    e1_selected = select_e1_groups(e1)
    e1_summary = summarize_groups(
        e1_selected,
        ["POOL_DEFINITION", "METHOD"],
        args.bootstrap,
        args.seed,
    )
    e2_summary = summarize_groups(
        e2,
        ["SCENARIO"],
        args.bootstrap,
        args.seed + 1000,
    )
    movements = scenario_movements(e2)

    e1_summary.to_csv(output_dir / "r22_e2b_e1_bootstrap.csv", index=False)
    e2_summary.to_csv(output_dir / "r22_e2b_e2_bootstrap.csv", index=False)
    movements.to_csv(output_dir / "r22_e2b_scenario_rank_movement.csv", index=False)
    write_report(
        e1_summary,
        e2_summary,
        movements,
        output_dir / "r22_e2b_robustness_report.md",
        args.bootstrap,
    )

    print(f"E2b complete. Report: {output_dir / 'r22_e2b_robustness_report.md'}")


if __name__ == "__main__":
    main()
