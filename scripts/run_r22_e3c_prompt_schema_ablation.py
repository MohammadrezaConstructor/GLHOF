#!/usr/bin/env python3
"""
E3c prompt/schema/inference ablation for Reviewer 2.2.

Time-conscious design
---------------------
- Uses one V0 request per selected base case.
- Reuses the existing strict zero-shot E3 results when available.
- Runs only the new few-shot and relaxed-schema configurations.
- Optionally adds a reasoning-effort ablation.

No API call is made unless --run-api is supplied.

Defaults match the supplied project tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


FEW_SHOT_APPENDIX = """
Examples:

User request:
Use the standard balanced approach and do not favour continuity or diversification.

Valid interpretation:
- intent_status: CLEAR
- scenario_profile: BALANCED
- priority_groups: []
- constraints: []
- evidence_requests: []
- clarification_question: null

User request:
Reduce dependence on established suppliers and broaden geographic market coverage.

Valid interpretation:
- intent_status: CLEAR
- scenario_profile: DIVERSIFICATION
- priority_groups: [GEOGRAPHIC_FIT]
- constraints: []
- evidence_requests: []
- clarification_question: null

These examples illustrate the schema and scenario catalogue. Apply the catalogue
and governance rules independently to the actual request.
""".strip()


SUMMARY_METRICS = [
    "API_SUCCESS_RATE",
    "SCENARIO_ACCURACY_END_TO_END",
    "CLARIFICATION_ACCURACY_END_TO_END",
    "PRIORITY_GROUP_EXACT_MATCH",
    "PRIORITY_GROUP_SET_F1",
    "CONSTRAINT_SET_F1",
    "EVIDENCE_REQUEST_SET_F1",
    "MEAN_LATENCY_SECONDS",
]


def select_v0_cases(benchmark_path: Path, n_base_cases: int, output_path: Path) -> pd.DataFrame:
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    benchmark = pd.read_csv(benchmark_path)
    required = {"request_id", "base_case_id", "variant_id"}
    missing = sorted(required - set(benchmark.columns))
    if missing:
        raise ValueError(f"Benchmark missing columns: {missing}")

    base_ids = list(dict.fromkeys(benchmark["base_case_id"].astype(str)))
    if n_base_cases > len(base_ids):
        raise ValueError(
            f"Requested {n_base_cases} base cases, but benchmark has {len(base_ids)}."
        )
    chosen = set(base_ids[:n_base_cases])
    subset = benchmark.loc[
        benchmark["base_case_id"].astype(str).isin(chosen)
        & benchmark["variant_id"].astype(str).eq("V0")
    ].copy()
    if len(subset) != n_base_cases:
        raise ValueError(
            f"Expected {n_base_cases} V0 requests, selected {len(subset)}."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False, encoding="utf-8-sig")
    return subset


def insert_few_shot(source: str) -> str:
    marker = "\ndef main() -> None:\n"
    if marker not in source:
        raise ValueError("Could not locate 'def main() -> None:' in E3 runner.")
    injection = (
        "\n\n# R2.2 generated few-shot appendix\n"
        f"FEW_SHOT_APPENDIX = {FEW_SHOT_APPENDIX!r}\n"
        "SYSTEM_PROMPT = SYSTEM_PROMPT + '\\n\\n' + FEW_SHOT_APPENDIX\n"
    )
    return source.replace(marker, injection + marker, 1)


def relax_schema(source: str) -> str:
    strict_marker = 'model_config = ConfigDict(extra="forbid")'
    if strict_marker not in source:
        raise ValueError("Could not locate strict Pydantic extra-field policy.")
    source = source.replace(
        strict_marker,
        'model_config = ConfigDict(extra="ignore")',
        1,
    )

    start_marker = '    @model_validator(mode="after")\n    def validate_governance_consistency(self):'
    start = source.find(start_marker)
    if start < 0:
        raise ValueError("Could not locate governance validator in E3 runner.")
    end_marker = "\n\nSYSTEM_PROMPT ="
    end = source.find(end_marker, start)
    if end < 0:
        raise ValueError("Could not locate end of governance validator.")

    relaxed_validator = (
        '    @model_validator(mode="after")\n'
        '    def validate_governance_consistency(self):\n'
        '        # R2.2 relaxed-schema condition: retain field types and required\n'
        '        # fields, but remove cross-field governance validation.\n'
        '        return self\n'
    )
    return source[:start] + relaxed_validator + source[end:]


def generate_runner(
    original_path: Path,
    output_path: Path,
    *,
    few_shot: bool,
    relaxed: bool,
) -> None:
    if not original_path.exists():
        raise FileNotFoundError(original_path)
    source = original_path.read_text(encoding="utf-8")
    if relaxed:
        source = relax_schema(source)
    if few_shot:
        source = insert_few_shot(source)
    compile(source, str(output_path), "exec")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")


def metric_map_from_existing(results_path: Path, selected_ids: list[str]) -> dict[str, float]:
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    results = pd.read_csv(results_path)
    required = {
        "request_id",
        "status",
        "SCENARIO_CORRECT",
        "CLARIFICATION_CORRECT",
        "PRIORITY_GROUP_EXACT",
        "PRIORITY_GROUP_F1",
        "CONSTRAINT_F1",
        "EVIDENCE_REQUEST_F1",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Existing E3 results missing columns: {missing}")

    selected = results.loc[results["request_id"].astype(str).isin(selected_ids)].copy()
    if set(selected["request_id"].astype(str)) != set(selected_ids):
        absent = sorted(set(selected_ids) - set(selected["request_id"].astype(str)))
        raise ValueError(f"Existing E3 results are missing selected requests: {absent[:10]}")

    ok = selected["status"].astype(str).eq("OK")
    return {
        "API_SUCCESS_RATE": float(ok.mean()),
        "SCENARIO_ACCURACY_END_TO_END": float(selected["SCENARIO_CORRECT"].astype(float).mean()),
        "CLARIFICATION_ACCURACY_END_TO_END": float(selected["CLARIFICATION_CORRECT"].astype(float).mean()),
        "PRIORITY_GROUP_EXACT_MATCH": float(selected["PRIORITY_GROUP_EXACT"].astype(float).mean()),
        "PRIORITY_GROUP_SET_F1": float(selected["PRIORITY_GROUP_F1"].astype(float).mean()),
        "CONSTRAINT_SET_F1": float(selected["CONSTRAINT_F1"].astype(float).mean()),
        "EVIDENCE_REQUEST_SET_F1": float(selected["EVIDENCE_REQUEST_F1"].astype(float).mean()),
        "MEAN_LATENCY_SECONDS": float(
            pd.to_numeric(selected.get("latency_seconds"), errors="coerce").mean()
        ) if "latency_seconds" in selected.columns else float("nan"),
    }


def metric_map_from_summary(summary_path: Path) -> dict[str, float]:
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    required = {"METRIC", "VALUE"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary missing columns: {missing}")
    raw = dict(zip(summary["METRIC"].astype(str), pd.to_numeric(summary["VALUE"], errors="coerce")))
    return {metric: float(raw.get(metric, float("nan"))) for metric in SUMMARY_METRICS}


def run_child(
    runner: Path,
    benchmark: Path,
    output_dir: Path,
    report_dir: Path,
    model: str,
    reasoning_effort: str,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(runner),
        "--benchmark", str(benchmark),
        "--model", model,
        "--output-dir", str(output_dir),
        "--report-dir", str(report_dir),
        "--reasoning-effort", reasoning_effort,
        "--max-output-tokens", "1200",
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True)


def aggregate(rows: list[dict], output_dir: Path, n_base_cases: int) -> None:
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "r22_e3c_configuration_comparison.csv", index=False)

    lines = [
        "# E3c Prompt, Schema, and Inference Ablation",
        "",
        f"The ablation uses one V0 request for each of {n_base_cases} base cases. "
        "The existing strict zero-shot baseline is reused; only new configurations "
        "require API calls.",
        "",
        "| Configuration | N | API success | Scenario accuracy | Clarification accuracy | Priority exact | Priority F1 | Constraint F1 | Evidence F1 | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| `{row['CONFIGURATION']}` | {int(row['N_REQUESTS'])} "
            f"| {row['API_SUCCESS_RATE']:.3f} "
            f"| {row['SCENARIO_ACCURACY_END_TO_END']:.3f} "
            f"| {row['CLARIFICATION_ACCURACY_END_TO_END']:.3f} "
            f"| {row['PRIORITY_GROUP_EXACT_MATCH']:.3f} "
            f"| {row['PRIORITY_GROUP_SET_F1']:.3f} "
            f"| {row['CONSTRAINT_SET_F1']:.3f} "
            f"| {row['EVIDENCE_REQUEST_SET_F1']:.3f} "
            f"| {row['MEAN_LATENCY_SECONDS']:.3f} |"
        )
    lines.extend([
        "",
        "The relaxed-schema condition retains required fields and controlled value "
        "types but removes extra-field rejection and cross-field governance validation. "
        "It is therefore a schema-strictness ablation, not an unconstrained free-text condition.",
        "",
        "Prompts and generated runner files should be frozen before the final run. "
        "Do not tune them after inspecting final test outcomes.",
        "",
    ])
    (output_dir / "r22_e3c_ablation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--base-cases", type=int, default=20)
    parser.add_argument("--include-reasoning-ablation", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-baseline", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    runner = root / "scripts/run_e3_intent_scenario_openai_v5.py"
    benchmark = root / "data/e3/e3_benchmark_50x3_gold_reviewed_v1.csv"
    existing_results = root / "data/analysis/e3_final/e3_api_results.csv"
    existing_metadata = root / "reports/e3_final/e3_run_metadata.json"
    data_root = root / "data/analysis/r22_e3c_ablation"
    report_root = root / "reports/r22_e3c_ablation"
    generated_root = data_root / "_generated_runners"
    subset_path = data_root / f"e3c_v0_{args.base_cases}_base_cases.csv"

    subset = select_v0_cases(benchmark, args.base_cases, subset_path)
    selected_ids = subset["request_id"].astype(str).tolist()

    strict_few = generated_root / "e3_runner_strict_few_shot.py"
    relaxed_few = generated_root / "e3_runner_relaxed_few_shot.py"
    generate_runner(runner, strict_few, few_shot=True, relaxed=False)
    generate_runner(runner, relaxed_few, few_shot=True, relaxed=True)

    generated = {
        "STRICT_FEW_SHOT_LOW": (strict_few, "low"),
        "RELAXED_FEW_SHOT_LOW": (relaxed_few, "low"),
    }
    if args.include_reasoning_ablation:
        generated["STRICT_FEW_SHOT_NONE"] = (strict_few, "none")

    manifest = {
        "source_runner": str(runner),
        "benchmark": str(benchmark),
        "subset": str(subset_path),
        "n_base_cases": args.base_cases,
        "model": args.model,
        "configurations": {
            name: {"runner": str(path), "reasoning_effort": effort}
            for name, (path, effort) in generated.items()
        },
        "baseline_reused": not args.rerun_baseline,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "r22_e3c_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if not args.run_api:
        print("E3c runners and benchmark subset prepared; no API calls were made.")
        print(f"Manifest: {report_root / 'r22_e3c_manifest.json'}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    rows: list[dict] = []

    if args.rerun_baseline:
        baseline_name = "STRICT_ZERO_SHOT_LOW_RERUN"
        run_child(
            runner,
            subset_path,
            data_root / baseline_name,
            report_root / baseline_name,
            args.model,
            "low",
            args.resume,
        )
        baseline_metrics = metric_map_from_summary(
            report_root / baseline_name / "e3_summary_metrics.csv"
        )
    else:
        baseline_name = "STRICT_ZERO_SHOT_LOW_REUSED"
        baseline_metrics = metric_map_from_existing(existing_results, selected_ids)
        if existing_metadata.exists():
            metadata = json.loads(existing_metadata.read_text(encoding="utf-8"))
            if metadata.get("model") != args.model:
                print(
                    "WARNING: existing E3 baseline model differs from requested model: "
                    f"{metadata.get('model')} vs {args.model}"
                )
            if metadata.get("reasoning_effort") != "low":
                print(
                    "WARNING: existing E3 baseline reasoning effort is not low: "
                    f"{metadata.get('reasoning_effort')}"
                )

    rows.append({
        "CONFIGURATION": baseline_name,
        "N_REQUESTS": args.base_cases,
        **baseline_metrics,
    })

    for name, (config_runner, effort) in generated.items():
        run_child(
            config_runner,
            subset_path,
            data_root / name,
            report_root / name,
            args.model,
            effort,
            args.resume,
        )
        metrics = metric_map_from_summary(
            report_root / name / "e3_summary_metrics.csv"
        )
        rows.append({
            "CONFIGURATION": name,
            "N_REQUESTS": args.base_cases,
            **metrics,
        })

    aggregate(rows, report_root, args.base_cases)
    print(f"E3c complete. Report: {report_root / 'r22_e3c_ablation_report.md'}")


if __name__ == "__main__":
    main()
