#!/usr/bin/env python3
"""
E1d: historical-proxy and observability subgroup sensitivity for Reviewer 2.2.

Purpose
-------
Evaluate whether the final deterministic model's reference-outcome positioning
is concentrated among:
- suppliers with a prior relationship to the target buyer;
- suppliers with extensive historical visibility;
- domestic versus cross-border reference suppliers;
- smaller versus larger historically active supplier comparison sets.

This is a sensitivity analysis of historical proxies and observability. It is
not a complete fairness audit and does not treat the observed awardee as a
verified optimal supplier.

No API calls are made.

Defaults match the supplied project tree.

Requirements
------------
pip install numpy pandas pyarrow

Example
-------
python scripts/run_r22_e1d_proxy_subgroup_sensitivity.py --root . --bootstrap 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = [
    "M3_FULL",
    "ABLATE_BUYER",
    "ABLATE_GEOGRAPHY",
    "ABLATE_CATEGORY",
]


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def read_parquet_columns(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise RuntimeError(f"Could not read {label} from {path}: {exc}") from exc


def safe_percentile(rank: np.ndarray, pool: np.ndarray) -> np.ndarray:
    denom = np.maximum(pool - 1.0, 1.0)
    return 1.0 - ((rank - 1.0) / denom)


def summarize(g: pd.DataFrame) -> dict[str, float]:
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
    pct = safe_percentile(rank, pool)
    return {
        "N_CASES": int(rank.size),
        "W_AT_1_PCT": 100.0 * float(np.mean(rank <= 1)),
        "W_AT_5_PCT": 100.0 * float(np.mean(rank <= 5)),
        "W_AT_10_PCT": 100.0 * float(np.mean(rank <= 10)),
        "W_AT_50_PCT": 100.0 * float(np.mean(rank <= 50)),
        "MRR": float(np.mean(1.0 / rank)),
        "MEAN_PERCENTILE": float(np.mean(pct)),
        "MEDIAN_RANK": float(np.median(rank)),
    }


def bootstrap_ci(
    g: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if len(g) == 0 or n_bootstrap <= 0:
        return {}
    metrics = {"W_AT_10_PCT": [], "MRR": [], "MEAN_PERCENTILE": []}
    n = len(g)
    for _ in range(n_bootstrap):
        sample = g.iloc[rng.integers(0, n, size=n)]
        row = summarize(sample)
        for metric in metrics:
            metrics[metric].append(row[metric])
    result: dict[str, float] = {}
    for metric, values in metrics.items():
        result[f"{metric}_CI_LOW"] = float(np.nanquantile(values, 0.025))
        result[f"{metric}_CI_HIGH"] = float(np.nanquantile(values, 0.975))
    return result


def normalize_country(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    return out.where(out.str.fullmatch(r"[A-Z]{2}", na=False))


def build_case_metadata(root: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    extended = root / "data/analysis/e1x_extended_fit_indices"
    target_profiles_path = extended / "e1x_target_profiles_2017.parquet"
    target_case_path = root / "data/processed/supplier_feature_store/target_case_features_2017.parquet"
    buyer_path = extended / "supplier_buyer_relationships_2015_2016.parquet"
    global_path = root / "data/processed/supplier_feature_store/supplier_global_features_2015_2016.parquet"

    profiles = read_parquet_columns(
        target_profiles_path,
        [
            "TARGET_CASE_ID",
            "TARGET_BUYER_KEY",
            "TARGET_PROCUREMENT_COUNTRY",
            "TARGET_DISPATCH_DATE",
        ],
        "target profiles",
    )
    outcomes = read_parquet_columns(
        target_case_path,
        [
            "TARGET_CASE_ID",
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        ],
        "target outcomes",
    )
    buyer = read_parquet_columns(
        buyer_path,
        [
            "BUYER_KEY",
            "SUPPLIER_ENTITY_ID",
            "N_PRIOR_BUYER_AWARDS",
            "BUYER_RELATIONSHIP_AWARD_SHARE",
        ],
        "buyer relationships",
    )
    global_features = read_parquet_columns(
        global_path,
        [
            "SUPPLIER_ENTITY_ID",
            "SUPPLIER_COUNTRY",
            "N_HISTORICAL_AWARDS",
        ],
        "supplier global features",
    )

    for df in (profiles, outcomes):
        df["TARGET_CASE_ID"] = df["TARGET_CASE_ID"].astype("string")
    profiles["TARGET_BUYER_KEY"] = profiles["TARGET_BUYER_KEY"].astype("string")
    outcomes["OBSERVED_WINNER_HISTORICAL_ENTITY_ID"] = outcomes[
        "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"
    ].astype("string")
    buyer["BUYER_KEY"] = buyer["BUYER_KEY"].astype("string")
    buyer["SUPPLIER_ENTITY_ID"] = buyer["SUPPLIER_ENTITY_ID"].astype("string")
    global_features["SUPPLIER_ENTITY_ID"] = global_features["SUPPLIER_ENTITY_ID"].astype("string")

    if profiles["TARGET_CASE_ID"].duplicated().any():
        raise ValueError("Target profiles contain duplicate TARGET_CASE_ID values.")
    if outcomes["TARGET_CASE_ID"].duplicated().any():
        raise ValueError("Target outcomes contain duplicate TARGET_CASE_ID values.")
    if global_features["SUPPLIER_ENTITY_ID"].duplicated().any():
        raise ValueError("Supplier global features contain duplicate supplier IDs.")

    buyer = buyer.rename(
        columns={
            "BUYER_KEY": "TARGET_BUYER_KEY",
            "SUPPLIER_ENTITY_ID": "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        }
    )
    buyer = (
        buyer.groupby(
            ["TARGET_BUYER_KEY", "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"],
            as_index=False,
            observed=True,
        )
        .agg(
            N_PRIOR_BUYER_AWARDS=("N_PRIOR_BUYER_AWARDS", "max"),
            BUYER_RELATIONSHIP_AWARD_SHARE=("BUYER_RELATIONSHIP_AWARD_SHARE", "max"),
        )
    )

    meta = profiles.merge(outcomes, on="TARGET_CASE_ID", how="inner", validate="one_to_one")
    meta = meta.merge(
        buyer,
        on=["TARGET_BUYER_KEY", "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"],
        how="left",
        validate="many_to_one",
    )
    meta = meta.merge(
        global_features.rename(
            columns={"SUPPLIER_ENTITY_ID": "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"}
        ),
        on="OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        how="left",
        validate="many_to_one",
    )

    meta["N_PRIOR_BUYER_AWARDS"] = pd.to_numeric(
        meta["N_PRIOR_BUYER_AWARDS"], errors="coerce"
    ).fillna(0.0)
    meta["BUYER_RELATIONSHIP_AWARD_SHARE"] = pd.to_numeric(
        meta["BUYER_RELATIONSHIP_AWARD_SHARE"], errors="coerce"
    ).fillna(0.0)
    meta["N_HISTORICAL_AWARDS"] = pd.to_numeric(
        meta["N_HISTORICAL_AWARDS"], errors="coerce"
    )

    meta["PRIOR_BUYER_GROUP"] = np.where(
        meta["N_PRIOR_BUYER_AWARDS"] > 0,
        "PRIOR_BUYER_RELATIONSHIP",
        "NO_PRIOR_BUYER_RELATIONSHIP",
    )

    valid_history = meta["N_HISTORICAL_AWARDS"].dropna()
    history_median = float(valid_history.median()) if not valid_history.empty else np.nan
    meta["HISTORY_VISIBILITY_GROUP"] = np.select(
        [
            meta["N_HISTORICAL_AWARDS"].isna(),
            meta["N_HISTORICAL_AWARDS"] <= history_median,
        ],
        ["UNKNOWN_HISTORY", "LOW_HISTORY_VISIBILITY"],
        default="HIGH_HISTORY_VISIBILITY",
    )

    target_country = normalize_country(meta["TARGET_PROCUREMENT_COUNTRY"])
    supplier_country = normalize_country(meta["SUPPLIER_COUNTRY"])
    comparable = target_country.notna() & supplier_country.notna()
    meta["BORDER_GROUP"] = np.select(
        [
            comparable & target_country.eq(supplier_country),
            comparable & target_country.ne(supplier_country),
        ],
        ["DOMESTIC_REFERENCE", "CROSS_BORDER_REFERENCE"],
        default="UNKNOWN_BORDER_STATUS",
    )

    metadata_info = {
        "history_visibility_median_awards": history_median,
        "n_metadata_cases": int(len(meta)),
        "n_prior_buyer_cases": int((meta["N_PRIOR_BUYER_AWARDS"] > 0).sum()),
        "n_cross_border_comparable": int(comparable.sum()),
    }
    return meta, metadata_info


def add_pool_size_group(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    case_pool = (
        df[["TARGET_CASE_ID", "POOL_SIZE"]]
        .drop_duplicates("TARGET_CASE_ID")
        .copy()
    )
    if case_pool["TARGET_CASE_ID"].duplicated().any():
        raise ValueError("POOL_SIZE is inconsistent within target case.")
    try:
        bins, edges = pd.qcut(
            pd.to_numeric(case_pool["POOL_SIZE"], errors="coerce"),
            q=4,
            labels=["POOL_Q1_SMALLEST", "POOL_Q2", "POOL_Q3", "POOL_Q4_LARGEST"],
            retbins=True,
            duplicates="drop",
        )
        case_pool["POOL_SIZE_GROUP"] = bins.astype("string")
    except ValueError:
        case_pool["POOL_SIZE_GROUP"] = "POOL_ALL"
        edges = np.array([], dtype=float)
    return df.merge(case_pool, on=["TARGET_CASE_ID", "POOL_SIZE"], how="left"), [float(x) for x in edges]


def summarize_subgroups(
    results: pd.DataFrame,
    subgroup_columns: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for subgroup_col in subgroup_columns:
        for (subgroup_value, model), g in results.groupby(
            [subgroup_col, "MODEL"], sort=False, observed=True
        ):
            row = {
                "SUBGROUP_DIMENSION": subgroup_col,
                "SUBGROUP": str(subgroup_value),
                "MODEL": str(model),
            }
            row.update(summarize(g))
            row.update(bootstrap_ci(g, n_bootstrap, rng))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_model_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["MODEL"] == "M3_FULL"].copy()
    base = base.rename(
        columns={
            "W_AT_10_PCT": "BASE_W_AT_10_PCT",
            "MRR": "BASE_MRR",
            "MEAN_PERCENTILE": "BASE_MEAN_PERCENTILE",
            "MEDIAN_RANK": "BASE_MEDIAN_RANK",
        }
    )[
        [
            "SUBGROUP_DIMENSION",
            "SUBGROUP",
            "BASE_W_AT_10_PCT",
            "BASE_MRR",
            "BASE_MEAN_PERCENTILE",
            "BASE_MEDIAN_RANK",
        ]
    ]
    comparison = summary.loc[summary["MODEL"] != "M3_FULL"].merge(
        base,
        on=["SUBGROUP_DIMENSION", "SUBGROUP"],
        how="left",
        validate="many_to_one",
    )
    comparison["DELTA_W_AT_10_PP_VS_FULL"] = (
        comparison["W_AT_10_PCT"] - comparison["BASE_W_AT_10_PCT"]
    )
    comparison["DELTA_MRR_VS_FULL"] = comparison["MRR"] - comparison["BASE_MRR"]
    comparison["DELTA_MEAN_PERCENTILE_VS_FULL"] = (
        comparison["MEAN_PERCENTILE"] - comparison["BASE_MEAN_PERCENTILE"]
    )
    comparison["DELTA_MEDIAN_RANK_VS_FULL"] = (
        comparison["MEDIAN_RANK"] - comparison["BASE_MEDIAN_RANK"]
    )
    return comparison


def write_report(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    metadata_info: dict[str, float],
    pool_edges: list[float],
    output_path: Path,
) -> None:
    lines = [
        "# E1d Historical-Proxy and Observability Sensitivity",
        "",
        "This analysis evaluates sensitivity to prior-buyer history, historical visibility, "
        "domestic/cross-border status, and comparison-set size. It is not a complete fairness "
        "audit. The observed awardee remains a historical reference outcome.",
        "",
        f"- Median historical awards used for the visibility split: "
        f"{metadata_info['history_visibility_median_awards']:.1f}",
        f"- Cases with a prior buyer relationship: {metadata_info['n_prior_buyer_cases']:,}",
        f"- Cases with comparable buyer/supplier countries: {metadata_info['n_cross_border_comparable']:,}",
    ]
    if pool_edges:
        lines.append(f"- Pool-size quartile edges: {pool_edges}")

    for dimension in [
        "PRIOR_BUYER_GROUP",
        "HISTORY_VISIBILITY_GROUP",
        "BORDER_GROUP",
        "POOL_SIZE_GROUP",
    ]:
        part = summary.loc[summary["SUBGROUP_DIMENSION"] == dimension]
        if part.empty:
            continue
        lines.extend([
            "",
            f"## {dimension}",
            "",
            "| Subgroup | Model | N | W@10 (%) | 95% CI | MRR | Mean percentile | Median rank |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ])
        for _, r in part.iterrows():
            ci = ""
            if pd.notna(r.get("W_AT_10_PCT_CI_LOW")):
                ci = f"[{r['W_AT_10_PCT_CI_LOW']:.3f}, {r['W_AT_10_PCT_CI_HIGH']:.3f}]"
            lines.append(
                f"| `{r['SUBGROUP']}` | `{r['MODEL']}` | {int(r['N_CASES'])} "
                f"| {r['W_AT_10_PCT']:.3f} | {ci} | {r['MRR']:.6f} "
                f"| {r['MEAN_PERCENTILE']:.6f} | {r['MEDIAN_RANK']:.1f} |"
            )

    lines.extend([
        "",
        "## Model-removal effects within subgroups",
        "",
        "| Dimension | Subgroup | Ablation | ΔW@10 vs full (pp) | ΔMRR | Δmedian rank |",
        "|---|---|---|---:|---:|---:|",
    ])
    for _, r in deltas.iterrows():
        lines.append(
            f"| `{r['SUBGROUP_DIMENSION']}` | `{r['SUBGROUP']}` | `{r['MODEL']}` "
            f"| {r['DELTA_W_AT_10_PP_VS_FULL']:.3f} "
            f"| {r['DELTA_MRR_VS_FULL']:.6f} "
            f"| {r['DELTA_MEDIAN_RANK_VS_FULL']:.1f} |"
        )

    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- A large performance gap between prior-buyer and no-prior-buyer cases indicates "
        "that historical continuity/visibility affects reference-outcome positioning.",
        "- If removing the buyer group narrows that gap, the buyer features are acting as a "
        "continuity/incumbency proxy. This does not by itself establish unfair treatment.",
        "- Domestic/cross-border differences must be interpreted jointly with data coverage "
        "and the geographic-fit ablation.",
        "- Subgroup estimates with small N should not support broad claims.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    case_results_path = (
        root / "data/analysis/r22_e1c_ablation/r22_e1c_case_model_results.parquet"
    )
    if not case_results_path.exists():
        raise FileNotFoundError(
            f"Missing E1c case results: {case_results_path}. Run E1c first."
        )

    results = pd.read_parquet(case_results_path)
    require_columns(
        results,
        {"TARGET_CASE_ID", "MODEL", "POOL_SIZE", "WINNER_RANK_MID"},
        "E1c case results",
    )
    results["TARGET_CASE_ID"] = results["TARGET_CASE_ID"].astype("string")
    results = results.loc[results["MODEL"].astype(str).isin(MODELS)].copy()
    missing_models = sorted(set(MODELS) - set(results["MODEL"].astype(str)))
    if missing_models:
        raise ValueError(f"E1c case results missing models: {missing_models}")

    meta, metadata_info = build_case_metadata(root)
    merged = results.merge(meta, on="TARGET_CASE_ID", how="left", validate="many_to_one")
    if merged["OBSERVED_WINNER_HISTORICAL_ENTITY_ID"].isna().any():
        n = int(merged["OBSERVED_WINNER_HISTORICAL_ENTITY_ID"].isna().sum())
        raise ValueError(f"Missing case metadata for {n} model-case rows.")

    merged, pool_edges = add_pool_size_group(merged)

    expected_per_case = len(MODELS)
    per_case = merged.groupby("TARGET_CASE_ID", observed=True)["MODEL"].nunique()
    if not per_case.eq(expected_per_case).all():
        raise ValueError("Not every case has all requested models after merging metadata.")

    if args.check_only:
        print(
            f"E1d preflight passed: {merged['TARGET_CASE_ID'].nunique():,} cases, "
            f"{len(merged):,} model-case rows."
        )
        return

    report_dir = root / "reports/r22_e1d_proxy_sensitivity"
    output_dir = root / "data/analysis/r22_e1d_proxy_sensitivity"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    subgroup_columns = [
        "PRIOR_BUYER_GROUP",
        "HISTORY_VISIBILITY_GROUP",
        "BORDER_GROUP",
        "POOL_SIZE_GROUP",
    ]
    summary = summarize_subgroups(
        merged,
        subgroup_columns=subgroup_columns,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    deltas = compute_model_deltas(summary)

    summary.to_csv(report_dir / "r22_e1d_subgroup_summary.csv", index=False)
    deltas.to_csv(report_dir / "r22_e1d_subgroup_ablation_deltas.csv", index=False)
    merged[
        [
            "TARGET_CASE_ID",
            "MODEL",
            "POOL_SIZE",
            "WINNER_RANK_MID",
            "PRIOR_BUYER_GROUP",
            "N_PRIOR_BUYER_AWARDS",
            "HISTORY_VISIBILITY_GROUP",
            "N_HISTORICAL_AWARDS",
            "BORDER_GROUP",
            "POOL_SIZE_GROUP",
        ]
    ].to_parquet(output_dir / "r22_e1d_case_subgroups.parquet", index=False)

    metadata = {
        **metadata_info,
        "pool_size_edges": pool_edges,
        "models": MODELS,
        "bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
    }
    (report_dir / "r22_e1d_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(
        summary,
        deltas,
        metadata_info,
        pool_edges,
        report_dir / "r22_e1d_proxy_sensitivity_report.md",
    )
    print(f"E1d complete. Report: {report_dir / 'r22_e1d_proxy_sensitivity_report.md'}")


if __name__ == "__main__":
    main()
