#!/usr/bin/env python3
"""
R2.2 / E1c fast feature-group ablation.

This wrapper reuses the validated E1x data-loading and TOPSIS implementation.
It computes the full four-group model, leave-one-group-out ablations, and
single-group models in one pass over the target cases.

The script deliberately retains M0_ORIGINAL because the underlying E1x runner
uses it for a validation check against the earlier TOPSIS baseline.

Requirements
------------
pip install numpy pandas pyarrow

Example
-------
python run_r22_e1c_feature_ablation_fast.py ^
  --e1x-script scripts/run_e1x_extended_procurement_fit_rankings_v2.py ^
  --extended-index-dir data/processed/e1x_indices ^
  --base-fit-index-dir data/processed/e1b_indices ^
  --target-case-features data/processed/supplier_features/target_case_features_2017.parquet ^
  --cpv2-features data/processed/supplier_features/supplier_features_cpv2_2015_2016.parquet ^
  --e1-case-results data/analysis/e1_rankings/e1_case_method_results.parquet ^
  --output-dir data/analysis/r22_e1c_ablation ^
  --report-dir reports/r22_e1c_ablation
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("e1x_ablation_base", str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import E1x script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_report(summary_df, incremental_df, validation_df, output_path: Path) -> None:
    lines = [
        "# R2.2 E1c Feature-Group Ablation",
        "",
        "The experiment reuses the frozen E1x feature construction, comparison set, "
        "tie-aware ranking, and TOPSIS implementation. Observed awardees are "
        "historical reference outcomes, not proof of uniquely optimal suppliers.",
        "",
        "## Model performance",
        "",
        "| Model | N | W@1 (%) | W@5 (%) | W@10 (%) | W@50 (%) | MRR | Mean percentile | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| `{row['MODEL']}` | {int(row['N_CASES'])} "
            f"| {row['WINNER_AT_1_PCT']:.3f} "
            f"| {row['WINNER_AT_5_PCT']:.3f} "
            f"| {row['WINNER_AT_10_PCT']:.3f} "
            f"| {row['WINNER_AT_50_PCT']:.3f} "
            f"| {row['MRR']:.6f} "
            f"| {row['MEAN_PERCENTILE_RANK']:.6f} "
            f"| {row['MEDIAN_WINNER_RANK']:.1f} |"
        )

    lines.extend([
        "",
        "## Change relative to the full four-group model",
        "",
        "| Comparison | ΔW@10 (pp) | ΔW@50 (pp) | ΔMRR | Mean rank change | % moved up | % moved down |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in incremental_df.iterrows():
        lines.append(
            f"| `{row['COMPARISON_MODEL']}` "
            f"| {row['DELTA_WINNER_AT_10_PP']:.3f} "
            f"| {row['DELTA_WINNER_AT_50_PP']:.3f} "
            f"| {row['DELTA_MRR']:.6f} "
            f"| {row['MEAN_RANK_CHANGE']:.3f} "
            f"| {row['PCT_WINNER_MOVED_UP']:.3f} "
            f"| {row['PCT_WINNER_MOVED_DOWN']:.3f} |"
        )

    lines.extend(["", "## Validation", ""])
    for _, row in validation_df.iterrows():
        status = "PASS" if bool(row["VALUE"]) else "FAIL"
        lines.append(f"- **{status}** `{row['CHECK']}`: {row['DETAIL']}")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The leave-one-group-out rows isolate the marginal contribution of each "
        "retained feature family. Single-group rows show the standalone signal "
        "available from category, geography, buyer relationship, and historical "
        "activity. Buyer-relationship variables should be interpreted as "
        "continuity or repeat-buyer signals rather than general supplier quality.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1x-script", required=True, type=Path)
    parser.add_argument("--extended-index-dir", required=True, type=Path)
    parser.add_argument("--base-fit-index-dir", required=True, type=Path)
    parser.add_argument("--target-case-features", required=True, type=Path)
    parser.add_argument("--cpv2-features", required=True, type=Path)
    parser.add_argument("--e1-case-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()

    module = load_module(args.e1x_script)

    module.MODEL_GROUPS = OrderedDict([
        (
            "M0_ORIGINAL",
            ["M0_ORIGINAL_FEATURES"],
        ),
        (
            "M3_FULL",
            [
                "CATEGORY_FIT",
                "GEOGRAPHIC_FIT",
                "BUYER_RELATIONSHIP",
                "ACTIVITY_PROFILE",
            ],
        ),
        (
            "ABLATE_CATEGORY",
            [
                "GEOGRAPHIC_FIT",
                "BUYER_RELATIONSHIP",
                "ACTIVITY_PROFILE",
            ],
        ),
        (
            "ABLATE_GEOGRAPHY",
            [
                "CATEGORY_FIT",
                "BUYER_RELATIONSHIP",
                "ACTIVITY_PROFILE",
            ],
        ),
        (
            "ABLATE_BUYER",
            [
                "CATEGORY_FIT",
                "GEOGRAPHIC_FIT",
                "ACTIVITY_PROFILE",
            ],
        ),
        (
            "ABLATE_ACTIVITY",
            [
                "CATEGORY_FIT",
                "GEOGRAPHIC_FIT",
                "BUYER_RELATIONSHIP",
            ],
        ),
        ("CATEGORY_ONLY", ["CATEGORY_FIT"]),
        ("GEOGRAPHY_ONLY", ["GEOGRAPHIC_FIT"]),
        ("BUYER_ONLY", ["BUYER_RELATIONSHIP"]),
        ("ACTIVITY_ONLY", ["ACTIVITY_PROFILE"]),
    ])
    module.MODEL_ORDER = {
        name: index for index, name in enumerate(module.MODEL_GROUPS)
    }

    comparison_models = [
        name
        for name in module.MODEL_GROUPS
        if name not in {"M0_ORIGINAL", "M3_FULL"}
    ]

    def incremental_comparisons(results, summary_df):
        return module.rank_movement_comparison(
            results,
            summary_df,
            "M3_FULL",
            comparison_models,
        )

    def decomposition(results, summary_df):
        return module.rank_movement_comparison(
            results,
            summary_df,
            "M3_FULL",
            [
                "CATEGORY_ONLY",
                "GEOGRAPHY_ONLY",
                "BUYER_ONLY",
                "ACTIVITY_ONLY",
            ],
        )

    def write_report(summary_df, incremental_df, buyer_geo_df, validation_df, output_path):
        make_report(summary_df, incremental_df, validation_df, output_path)

    module.incremental_comparisons = incremental_comparisons
    module.buyer_geo_decomposition = decomposition
    module.write_report = write_report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            str(args.e1x_script),
            "--extended-index-dir", str(args.extended_index_dir),
            "--base-fit-index-dir", str(args.base_fit_index_dir),
            "--target-case-features", str(args.target_case_features),
            "--cpv2-features", str(args.cpv2_features),
            "--e1-case-results", str(args.e1_case_results),
            "--output-dir", str(args.output_dir),
            "--report-dir", str(args.report_dir),
        ]
        module.main()
    finally:
        sys.argv = original_argv

    copies = {
        args.report_dir / "e1x_model_summary.csv":
            args.report_dir / "r22_e1c_ablation_summary.csv",
        args.report_dir / "e1x_incremental_comparison.csv":
            args.report_dir / "r22_e1c_ablation_deltas.csv",
        args.report_dir / "e1x_ranking_report.md":
            args.report_dir / "r22_e1c_ablation_report.md",
        args.output_dir / "e1x_case_model_results.parquet":
            args.output_dir / "r22_e1c_case_model_results.parquet",
    }
    for source, target in copies.items():
        if source.exists():
            shutil.copy2(source, target)

    print(f"Done. Main report: {args.report_dir / 'r22_e1c_ablation_report.md'}")


if __name__ == "__main__":
    main()
