#!/usr/bin/env python3
"""
Preflight checks for the Reviewer 2.2 experiment suite.

The defaults match the project tree supplied by the user. The script performs
only local checks: file existence, expected columns, Python dependencies, and
runner-source compatibility. It does not call any API or alter experiment data.

Example
-------
python scripts/run_r22_preflight.py --root .
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def project_paths(root: Path) -> dict[str, Path]:
    return {
        "e1x_runner": root / "scripts/run_e1x_extended_procurement_fit_rankings_v2.py",
        "e3_runner": root / "scripts/run_e3_intent_scenario_openai_v5.py",
        "e4_runner": root / "scripts/run_e4_evidence_extraction_openai_v2.py",
        "extended_index_dir": root / "data/analysis/e1x_extended_fit_indices",
        "base_fit_index_dir": root / "data/analysis/e1b_fit_indices",
        "target_case_features": root / "data/processed/supplier_feature_store/target_case_features_2017.parquet",
        "cpv2_features": root / "data/processed/supplier_feature_store/supplier_cpv2_features_2015_2016.parquet",
        "e1_case_results": root / "data/analysis/e1_rankings/e1_case_ranking_results.parquet",
        "e1x_case_results": root / "data/analysis/e1x_rankings/e1x_case_model_results.parquet",
        "e2x_case_results": root / "data/analysis/e2x_scenarios/e2x_case_scenario_results.parquet",
        "e3_benchmark": root / "data/e3/e3_benchmark_50x3_gold_reviewed_v1.csv",
        "e3_existing_results": root / "data/analysis/e3_final/e3_api_results.csv",
        "e3_existing_metadata": root / "reports/e3_final/e3_run_metadata.json",
        "e4_benchmark": root / "data/e4/e4_benchmark_40_gold.csv",
        "e4_existing_packet_metrics": root / "data/analysis/e4_final/e4_packet_metrics.csv",
    }


EXPECTED_EXTENDED_FILES = [
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

EXPECTED_BASE_FIT_FILES = [
    "supplier_country_fit_counts_2015_2016.parquet",
]


CSV_COLUMNS = {
    "e3_benchmark": {
        "request_id",
        "base_case_id",
        "variant_id",
        "category",
        "request_text",
        "gold_scenario_profile",
        "gold_priority_groups_json",
        "gold_constraints_json",
        "gold_evidence_requests_json",
        "gold_requires_clarification",
    },
    "e3_existing_results": {
        "request_id",
        "status",
        "SCENARIO_CORRECT",
        "CLARIFICATION_CORRECT",
        "PRIORITY_GROUP_EXACT",
        "PRIORITY_GROUP_F1",
        "CONSTRAINT_F1",
        "EVIDENCE_REQUEST_F1",
    },
    "e4_benchmark": {
        "packet_id",
        "supplier_name",
        "category",
        "evidence_packet",
        "gold_claims_json",
    },
    "e4_existing_packet_metrics": {
        "packet_id",
        "claim_precision",
        "claim_recall",
        "claim_f1",
        "source_grounding_rate",
        "hallucination_rate",
        "over_extraction_rate",
        "missed_gold_rate",
    },
}

PARQUET_COLUMNS = {
    "e1_case_results": {
        "TARGET_CASE_ID",
        "POOL_DEFINITION",
        "METHOD",
        "POOL_SIZE",
        "WINNER_RANK_MID",
    },
    "e1x_case_results": {
        "TARGET_CASE_ID",
        "MODEL",
        "POOL_SIZE",
        "WINNER_RANK_MID",
    },
    "e2x_case_results": {
        "TARGET_CASE_ID",
        "SCENARIO",
        "POOL_SIZE",
        "WINNER_RANK_MID",
    },
}


def check_exists(name: str, path: Path) -> Check:
    return Check(name, path.exists(), str(path.resolve()))


def read_parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        # Fallback for environments where pandas has a different parquet engine.
        return set(pd.read_parquet(path).columns)


def check_columns(name: str, path: Path, expected: set[str], file_type: str) -> Check:
    if not path.exists():
        return Check(f"{name}_columns", False, f"missing file: {path}")
    try:
        if file_type == "csv":
            actual = set(pd.read_csv(path, nrows=2).columns)
        else:
            actual = read_parquet_columns(path)
    except Exception as exc:
        return Check(f"{name}_columns", False, f"{type(exc).__name__}: {exc}")

    missing = sorted(expected - actual)
    if missing:
        return Check(f"{name}_columns", False, f"missing columns: {missing}")
    return Check(f"{name}_columns", True, f"expected columns present ({len(expected)})")


def check_dependency(module_name: str, attribute: str | None = None) -> Check:
    try:
        module = importlib.import_module(module_name)
        if attribute is not None and not hasattr(module, attribute):
            return Check(
                f"dependency_{module_name}",
                False,
                f"module imported but attribute {attribute!r} is missing",
            )
        return Check(f"dependency_{module_name}", True, "available")
    except Exception as exc:
        return Check(f"dependency_{module_name}", False, f"{type(exc).__name__}: {exc}")


def check_runner_markers(name: str, path: Path, markers: Iterable[str]) -> Check:
    if not path.exists():
        return Check(f"{name}_markers", False, f"missing file: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        return Check(f"{name}_markers", False, f"missing source markers: {missing}")
    try:
        compile(text, str(path), "exec")
    except SyntaxError as exc:
        return Check(f"{name}_markers", False, f"syntax error: {exc}")
    return Check(f"{name}_markers", True, "source markers and syntax verified")


def write_reports(checks: list[Check], report_dir: Path, root: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_root": str(root.resolve()),
        "passed": all(item.passed for item in checks),
        "checks": [asdict(item) for item in checks],
    }
    (report_dir / "r22_preflight.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# R2.2 Experiment Preflight",
        "",
        f"Project root: `{root.resolve()}`",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        status = "PASS" if item.passed else "FAIL"
        detail = item.detail.replace("|", "\\|")
        lines.append(f"| `{item.name}` | **{status}** | {detail} |")
    lines.extend([
        "",
        f"Overall status: **{'PASS' if payload['passed'] else 'FAIL'}**",
        "",
    ])
    (report_dir / "r22_preflight.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Default: <root>/reports/r22_preflight",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    paths = project_paths(root)
    report_dir = args.report_dir or (root / "reports/r22_preflight")

    checks: list[Check] = []

    for name, path in paths.items():
        checks.append(check_exists(name, path))

    for filename in EXPECTED_EXTENDED_FILES:
        checks.append(check_exists(
            f"extended_index_{filename}",
            paths["extended_index_dir"] / filename,
        ))
    for filename in EXPECTED_BASE_FIT_FILES:
        checks.append(check_exists(
            f"base_fit_{filename}",
            paths["base_fit_index_dir"] / filename,
        ))

    for name, expected in CSV_COLUMNS.items():
        checks.append(check_columns(name, paths[name], expected, "csv"))
    for name, expected in PARQUET_COLUMNS.items():
        checks.append(check_columns(name, paths[name], expected, "parquet"))

    checks.extend([
        check_dependency("numpy"),
        check_dependency("pandas"),
        check_dependency("pyarrow"),
        check_dependency("pydantic"),
        check_dependency("openai", "OpenAI"),
    ])

    checks.append(check_runner_markers(
        "e1x_runner",
        paths["e1x_runner"],
        ["MODEL_GROUPS =", "def run_experiment(", "def validation_checks(", "def main() -> None:"],
    ))
    checks.append(check_runner_markers(
        "e3_runner",
        paths["e3_runner"],
        [
            'model_config = ConfigDict(extra="forbid")',
            '@model_validator(mode="after")',
            "SYSTEM_PROMPT =",
            "def main() -> None:",
        ],
    ))
    checks.append(check_runner_markers(
        "e4_runner",
        paths["e4_runner"],
        ["def validate_benchmark(", "def evaluate_packet(", "def main() -> None:"],
    ))

    write_reports(checks, report_dir, root)

    failed = [item for item in checks if not item.passed]
    print(f"Preflight checks: {len(checks) - len(failed)} passed, {len(failed)} failed.")
    print(f"Report: {report_dir / 'r22_preflight.md'}")
    if failed:
        for item in failed:
            print(f"FAIL {item.name}: {item.detail}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
