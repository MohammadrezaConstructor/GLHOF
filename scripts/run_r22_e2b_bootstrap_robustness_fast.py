#!/usr/bin/env python3
"""
R2.2 / E2b fast robustness analysis using existing case-level outputs.

This script requires no API calls and does not recompute supplier features.
It provides:
- bootstrap confidence intervals for E1 ranking metrics by method/pool;
- bootstrap confidence intervals for E2 scenario metrics;
- sensitivity across historically active supplier comparison-set definitions;
- pairwise scenario reference-rank movement.

It is intentionally lightweight for a revision deadline.

Requirements
------------
pip install numpy pandas pyarrow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def metric_vector(g: pd.DataFrame) -> dict[str, float]:
    rank = pd.to_numeric(g["WINNER_RANK_MID"], errors="coerce").to_numpy(float)
    pool = pd.to_numeric(g["POOL_SIZE"], errors="coerce").to_numpy(float)
    valid = np.isfinite(rank) & np.isfinite(pool) & (rank > 0) & (pool > 0)
    rank = rank[valid]
    pool = pool[valid]
    if rank.size == 0:
        return {
            "N_CASES": 0,
            "W_AT_1_PCT": np.nan,
            "W_AT_5_PCT": np.nan,
            "W_AT_10_PCT": np.nan,
            "W_AT_50_PCT": np.nan,
            "MRR": np.nan,
            "MEAN_PERCENTILE": np.nan,
            "MEDIAN_RANK": np.nan,
        }
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


def bootstrap_group(
    g: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    g = g.reset_index(drop=True)
    n = len(g)
    metrics = {
        "W_AT_10_PCT": [],
        "W_AT_50_PCT": [],
        "MRR": [],
        "MEAN_PERCENTILE": [],
        "MEDIAN_RANK": [],
    }
    for _ in range(n_bootstrap):
        sample = g.iloc[rng.integers(0, n, size=n)]
        values = metric_vector(sample)
        for key in metrics:
            metrics[key].append(values[key])
    return {
        key: (
            float(np.nanquantile(values, 0.025)),
            float(np.nanquantile(values, 0.975)),
        )
        for key, values in metrics.items()
    }


def summarize_with_bootstrap(
    df: pd.DataFrame,
    group_cols: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, g in df.groupby(grouper, sort=False, observed=True):
        key = (key,) if len(group_cols) == 1 else tuple(key)
        row = dict(zip(group_cols, key))
        row.update(metric_vector(g))
        intervals = bootstrap_group(g, n_bootstrap, rng)
        for metric, (low, high) in intervals.items():
            row[f"{metric}_CI_LOW"] = low
            row[f"{metric}_CI_HIGH"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_rank_movements(e2: pd.DataFrame) -> pd.DataFrame:
    pivot = e2.pivot_table(
        index="TARGET_CASE_ID",
        columns="SCENARIO",
        values="WINNER_RANK_MID",
        aggfunc="first",
    )
    if "BALANCED" not in pivot.columns:
        return pd.DataFrame()
    rows = []
    base = pivot["BALANCED"]
    for scenario in pivot.columns:
        if scenario == "BALANCED":
            continue
        pair = pd.concat([base, pivot[scenario]], axis=1).dropna()
        pair.columns = ["BALANCED", "COMPARISON"]
        delta = pair["COMPARISON"] - pair["BALANCED"]
        rows.append({
            "REFERENCE": "BALANCED",
            "COMPARISON": scenario,
            "N_CASES": len(pair),
            "RANK_PEARSON_CORRELATION": pair.corr(method="pearson").iloc[0, 1],
            "RANK_SPEARMAN_CORRELATION": pair.corr(method="spearman").iloc[0, 1],
            "MEAN_RANK_CHANGE": delta.mean(),
            "MEAN_ABS_RANK_CHANGE": delta.abs().mean(),
            "MEDIAN_ABS_RANK_CHANGE": delta.abs().median(),
            "PCT_REFERENCE_MOVED_UP": 100.0 * (delta < 0).mean(),
            "PCT_REFERENCE_UNCHANGED": 100.0 * (delta == 0).mean(),
            "PCT_REFERENCE_MOVED_DOWN": 100.0 * (delta > 0).mean(),
        })
    return pd.DataFrame(rows)


def write_report(
    e1_summary: pd.DataFrame,
    e2_summary: pd.DataFrame,
    movement: pd.DataFrame,
    output_path: Path,
    n_bootstrap: int,
) -> None:
    lines = [
        "# R2.2 E2b Bootstrap and Comparison-Set Robustness",
        "",
        f"Confidence intervals use {n_bootstrap} case-level bootstrap resamples.",
        "",
        "## E1 method and comparison-set sensitivity",
        "",
        "| Pool | Method | N | W@10 | 95% CI | W@50 | 95% CI | MRR | 95% CI |",
        "|---|---|---:|---:|---|---:|---|---:|---|",
    ]
    for _, r in e1_summary.iterrows():
        lines.append(
            f"| `{r['POOL_DEFINITION']}` | `{r['METHOD']}` | {int(r['N_CASES'])} "
            f"| {r['W_AT_10_PCT']:.3f} "
            f"| [{r['W_AT_10_PCT_CI_LOW']:.3f}, {r['W_AT_10_PCT_CI_HIGH']:.3f}] "
            f"| {r['W_AT_50_PCT']:.3f} "
            f"| [{r['W_AT_50_PCT_CI_LOW']:.3f}, {r['W_AT_50_PCT_CI_HIGH']:.3f}] "
            f"| {r['MRR']:.6f} "
            f"| [{r['MRR_CI_LOW']:.6f}, {r['MRR_CI_HIGH']:.6f}] |"
        )

    lines.extend([
        "",
        "## E2 scenario uncertainty",
        "",
        "| Scenario | N | W@10 | 95% CI | W@50 | 95% CI | MRR | 95% CI |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ])
    for _, r in e2_summary.iterrows():
        lines.append(
            f"| `{r['SCENARIO']}` | {int(r['N_CASES'])} "
            f"| {r['W_AT_10_PCT']:.3f} "
            f"| [{r['W_AT_10_PCT_CI_LOW']:.3f}, {r['W_AT_10_PCT_CI_HIGH']:.3f}] "
            f"| {r['W_AT_50_PCT']:.3f} "
            f"| [{r['W_AT_50_PCT_CI_LOW']:.3f}, {r['W_AT_50_PCT_CI_HIGH']:.3f}] "
            f"| {r['MRR']:.6f} "
            f"| [{r['MRR_CI_LOW']:.6f}, {r['MRR_CI_HIGH']:.6f}] |"
        )

    if not movement.empty:
        lines.extend([
            "",
            "## Pairwise reference-rank movement",
            "",
            "| Comparison | N | Spearman | Mean absolute rank change | % moved up | % moved down |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for _, r in movement.iterrows():
            lines.append(
                f"| `{r['REFERENCE']} -> {r['COMPARISON']}` | {int(r['N_CASES'])} "
                f"| {r['RANK_SPEARMAN_CORRELATION']:.4f} "
                f"| {r['MEAN_ABS_RANK_CHANGE']:.3f} "
                f"| {r['PCT_REFERENCE_MOVED_UP']:.3f} "
                f"| {r['PCT_REFERENCE_MOVED_DOWN']:.3f} |"
            )

    lines.extend([
        "",
        "## Interpretation caution",
        "",
        "The analysis measures robustness of reference-outcome positioning under "
        "alternative deterministic methods, comparison-set definitions, and "
        "scenario profiles. It does not establish that the observed awardee was "
        "the uniquely optimal supplier.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-case-results", required=True, type=Path)
    parser.add_argument("--e2x-case-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--e1-methods",
        nargs="+",
        default=["TOPSIS", "WSM", "VIKOR", "FREQUENCY"],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    e1 = read_table(args.e1_case_results)
    e2 = read_table(args.e2x_case_results)

    required_e1 = {"TARGET_CASE_ID", "POOL_DEFINITION", "METHOD", "POOL_SIZE", "WINNER_RANK_MID"}
    required_e2 = {"TARGET_CASE_ID", "SCENARIO", "POOL_SIZE", "WINNER_RANK_MID"}
    missing_e1 = required_e1 - set(e1.columns)
    missing_e2 = required_e2 - set(e2.columns)
    if missing_e1:
        raise ValueError(f"E1 file missing columns: {sorted(missing_e1)}")
    if missing_e2:
        raise ValueError(f"E2x file missing columns: {sorted(missing_e2)}")

    e1 = e1.loc[e1["METHOD"].astype(str).str.upper().isin(
        {x.upper() for x in args.e1_methods}
    )].copy()

    e1_summary = summarize_with_bootstrap(
        e1,
        ["POOL_DEFINITION", "METHOD"],
        args.bootstrap,
        args.seed,
    )
    e2_summary = summarize_with_bootstrap(
        e2,
        ["SCENARIO"],
        args.bootstrap,
        args.seed + 1,
    )
    movement = scenario_rank_movements(e2)

    e1_summary.to_csv(args.output_dir / "r22_e2b_e1_bootstrap_by_pool_method.csv", index=False)
    e2_summary.to_csv(args.output_dir / "r22_e2b_e2_bootstrap_by_scenario.csv", index=False)
    movement.to_csv(args.output_dir / "r22_e2b_scenario_rank_movement.csv", index=False)
    write_report(
        e1_summary,
        e2_summary,
        movement,
        args.output_dir / "r22_e2b_robustness_report.md",
        args.bootstrap,
    )
    print(f"Done. Report: {args.output_dir / 'r22_e2b_robustness_report.md'}")


if __name__ == "__main__":
    main()
