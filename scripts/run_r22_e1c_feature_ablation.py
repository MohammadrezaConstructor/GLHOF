#!/usr/bin/env python3
"""
E1c feature-group ablation for Reviewer 2.2.

The script reuses the validated E1x data-loading, feature-construction, TOPSIS,
tie-aware ranking, and validation functions. It changes only the model list so
that all leave-one-group-out and single-group configurations are scored in one
pass over the primary CPV2_MIN1 cohort.

Defaults match the project tree supplied by the user.

Example
-------
python scripts/run_r22_e1c_feature_ablation.py --root .
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd


ABLATION_MODEL_GROUPS = OrderedDict([
    ("M0_ORIGINAL", ["M0_ORIGINAL_FEATURES"]),
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
        ["GEOGRAPHIC_FIT", "BUYER_RELATIONSHIP", "ACTIVITY_PROFILE"],
    ),
    (
        "ABLATE_GEOGRAPHY",
        ["CATEGORY_FIT", "BUYER_RELATIONSHIP", "ACTIVITY_PROFILE"],
    ),
    (
        "ABLATE_BUYER",
        ["CATEGORY_FIT", "GEOGRAPHIC_FIT", "ACTIVITY_PROFILE"],
    ),
    (
        "ABLATE_ACTIVITY",
        ["CATEGORY_FIT", "GEOGRAPHIC_FIT", "BUYER_RELATIONSHIP"],
    ),
    ("CATEGORY_ONLY", ["CATEGORY_FIT"]),
    ("GEOGRAPHY_ONLY", ["GEOGRAPHIC_FIT"]),
    ("BUYER_ONLY", ["BUYER_RELATIONSHIP"]),
    ("ACTIVITY_ONLY", ["ACTIVITY_PROFILE"]),
])


COMPARISON_MODELS = [
    "ABLATE_CATEGORY",
    "ABLATE_GEOGRAPHY",
    "ABLATE_BUYER",
    "ABLATE_ACTIVITY",
    "CATEGORY_ONLY",
    "GEOGRAPHY_ONLY",
    "BUYER_ONLY",
    "ACTIVITY_ONLY",
]


def load_e1x(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "r22_e1x_ablation_base",
        str(path.resolve()),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses/Pydantic-like forward references in dynamically
    # imported modules and harmless for this plain analytical module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "MODEL_GROUPS",
        "MODEL_ORDER",
        "run_experiment",
        "summarize_models",
        "rank_movement_comparison",
        "validation_checks",
        "main",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(f"E1x runner is missing required objects: {missing}")
    return module


def make_ablation_report(
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# E1c Feature-Group Ablation",
        "",
        "All models use the same CPV2_MIN1 historically active supplier comparison "
        "sets, the same temporally frozen features, and the same deterministic "
        "TOPSIS implementation. The observed awardee is treated as a historical "
        "reference outcome, not as proof of a uniquely optimal supplier.",
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
        "Positive rank change means a worse (numerically larger) reference rank.",
        "",
        "| Comparison | ΔW@10 (pp) | ΔW@50 (pp) | ΔMRR | Mean rank change | Mean absolute rank change | % moved up | % moved down |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in delta_df.iterrows():
        lines.append(
            f"| `{row['COMPARISON_MODEL']}` "
            f"| {row['DELTA_WINNER_AT_10_PP']:.3f} "
            f"| {row['DELTA_WINNER_AT_50_PP']:.3f} "
            f"| {row['DELTA_MRR']:.6f} "
            f"| {row['MEAN_RANK_CHANGE']:.3f} "
            f"| {row['MEAN_ABS_RANK_CHANGE']:.3f} "
            f"| {row['PCT_WINNER_MOVED_UP']:.3f} "
            f"| {row['PCT_WINNER_MOVED_DOWN']:.3f} |"
        )

    lines.extend(["", "## Validation", ""])
    for _, row in validation_df.iterrows():
        status = "PASS" if bool(row["VALUE"]) else "FAIL"
        lines.append(f"- **{status}** `{row['CHECK']}`: {row['DETAIL']}")

    lines.extend([
        "",
        "## Interpretation guidance",
        "",
        "The leave-one-group-out models estimate the marginal contribution of each "
        "retained feature family within this deterministic specification. The "
        "buyer-relationship group should be interpreted as continuity or repeat-buyer "
        "history, not as general supplier quality. Single-group models describe "
        "standalone historical signal and are not proposed as operational decision rules.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def patch_module(module) -> None:
    module.MODEL_GROUPS = ABLATION_MODEL_GROUPS
    module.MODEL_ORDER = {
        model: index for index, model in enumerate(ABLATION_MODEL_GROUPS)
    }

    def incremental_comparisons(results: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
        return module.rank_movement_comparison(
            results,
            summary_df,
            "M3_FULL",
            COMPARISON_MODELS,
        )

    def group_decomposition(results: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
        return module.rank_movement_comparison(
            results,
            summary_df,
            "M3_FULL",
            ["CATEGORY_ONLY", "GEOGRAPHY_ONLY", "BUYER_ONLY", "ACTIVITY_ONLY"],
        )

    def write_report(
        summary_df: pd.DataFrame,
        incremental_df: pd.DataFrame,
        buyer_geo_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        del buyer_geo_df
        make_ablation_report(summary_df, incremental_df, validation_df, output_path)

    module.incremental_comparisons = incremental_comparisons
    module.buyer_geo_decomposition = group_decomposition
    module.write_report = write_report


def required_paths(root: Path) -> dict[str, Path]:
    return {
        "e1x_script": root / "scripts/run_e1x_extended_procurement_fit_rankings_v2.py",
        "extended_index_dir": root / "data/analysis/e1x_extended_fit_indices",
        "base_fit_index_dir": root / "data/analysis/e1b_fit_indices",
        "target_case_features": root / "data/processed/supplier_feature_store/target_case_features_2017.parquet",
        "cpv2_features": root / "data/processed/supplier_feature_store/supplier_cpv2_features_2015_2016.parquet",
        "e1_case_results": root / "data/analysis/e1_rankings/e1_case_ranking_results.parquet",
        "output_dir": root / "data/analysis/r22_e1c_ablation",
        "report_dir": root / "reports/r22_e1c_ablation",
    }


def preflight(paths: dict[str, Path]) -> None:
    required = [
        "e1x_script",
        "extended_index_dir",
        "base_fit_index_dir",
        "target_case_features",
        "cpv2_features",
        "e1_case_results",
    ]
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  - " + "\n  - ".join(missing))

    # Cheap schema check on the baseline case-level result.
    expected = {
        "TARGET_CASE_ID",
        "POOL_DEFINITION",
        "METHOD",
        "WINNER_RANK_MID",
    }
    actual = set(pd.read_parquet(paths["e1_case_results"], columns=list(expected)).columns)
    if expected - actual:
        raise ValueError(f"E1 case results missing columns: {sorted(expected - actual)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    paths = required_paths(root)
    preflight(paths)

    module = load_e1x(paths["e1x_script"])
    patch_module(module)

    # Verify that every referenced group exists in the imported runner.
    unknown_groups = sorted({
        group
        for groups in ABLATION_MODEL_GROUPS.values()
        for group in groups
        if group not in module.GROUPS
    })
    if unknown_groups:
        raise ValueError(f"Unknown E1x groups in ablation specification: {unknown_groups}")

    if args.check_only:
        print("E1c preflight passed. No ranking experiment was executed.")
        return

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(paths["e1x_script"]),
            "--extended-index-dir", str(paths["extended_index_dir"]),
            "--base-fit-index-dir", str(paths["base_fit_index_dir"]),
            "--target-case-features", str(paths["target_case_features"]),
            "--cpv2-features", str(paths["cpv2_features"]),
            "--e1-case-results", str(paths["e1_case_results"]),
            "--output-dir", str(paths["output_dir"]),
            "--report-dir", str(paths["report_dir"]),
        ]
        module.main()
    finally:
        sys.argv = old_argv

    aliases = {
        paths["output_dir"] / "e1x_case_model_results.parquet":
            paths["output_dir"] / "r22_e1c_case_model_results.parquet",
        paths["report_dir"] / "e1x_model_summary.csv":
            paths["report_dir"] / "r22_e1c_ablation_summary.csv",
        paths["report_dir"] / "e1x_incremental_comparison.csv":
            paths["report_dir"] / "r22_e1c_ablation_deltas.csv",
        paths["report_dir"] / "e1x_model_specification.csv":
            paths["report_dir"] / "r22_e1c_model_specification.csv",
        paths["report_dir"] / "e1x_validation_checks.csv":
            paths["report_dir"] / "r22_e1c_validation_checks.csv",
        paths["report_dir"] / "e1x_ranking_report.md":
            paths["report_dir"] / "r22_e1c_ablation_report.md",
    }
    for source, target in aliases.items():
        if not source.exists():
            raise FileNotFoundError(f"Expected output was not produced: {source}")
        shutil.copy2(source, target)

    validation = pd.read_csv(paths["report_dir"] / "r22_e1c_validation_checks.csv")
    value_text = validation["VALUE"].astype(str).str.strip().str.lower()
    value_bool = value_text.map({"true": True, "false": False, "1": True, "0": False})
    if value_bool.isna().any():
        raise ValueError("Unrecognized boolean values in E1c validation output.")
    if not value_bool.all():
        failed = validation.loc[~value_bool, ["CHECK", "DETAIL"]]
        raise RuntimeError(f"E1c validation failed:\n{failed.to_string(index=False)}")

    print("E1c complete and validation passed.")
    print(f"Report: {paths['report_dir'] / 'r22_e1c_ablation_report.md'}")


if __name__ == "__main__":
    main()
