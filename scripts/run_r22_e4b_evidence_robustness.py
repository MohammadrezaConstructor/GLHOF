#!/usr/bin/env python3
"""
E4b robustness to noisy, incomplete, and conflicting evidence.

Time-conscious design
---------------------
- Reuses existing clean E4 packet metrics.
- Runs API calls only for perturbation variants.
- Pilot default: 15 source packets, NOISY and INCOMPLETE = 30 calls.
- Full run with --packets 40 --include-conflicting = 120 calls.

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


NOISE_TEXT = (
    "The supplier also updated its general website navigation and published an "
    "administrative office-hours notice. These statements do not concern the "
    "procurement criteria in this packet."
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def quote_in_packet(quote: str, packet: str) -> bool:
    return normalize_text(quote) in normalize_text(packet)


def remove_sentence_supporting_claim(packet: str, quote: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", packet.strip())
    kept = [sentence for sentence in sentences if normalize_text(quote) not in normalize_text(sentence)]
    if len(kept) == len(sentences):
        # Fallback: remove the exact quote while preserving the rest of the packet.
        return packet.replace(quote, "")
    return " ".join(kept).strip()


def validate_gold_claims(packet_id: str, packet: str, claims: list[dict]) -> None:
    for claim in claims:
        quote = str(claim.get("source_quote", ""))
        if not quote or not quote_in_packet(quote, packet):
            raise ValueError(
                f"{packet_id}: gold source quote is not present after perturbation: {quote!r}"
            )


def build_variants(
    source: pd.DataFrame,
    n_packets: int,
    include_conflicting: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if n_packets > len(source):
        raise ValueError(f"Requested {n_packets} packets, benchmark has {len(source)}.")

    selected = source.head(n_packets).copy()
    original_ids = selected["packet_id"].astype(str).tolist()
    rows: list[dict] = []

    for _, row in selected.iterrows():
        packet_id = str(row["packet_id"])
        supplier_name = str(row["supplier_name"])
        category = str(row["category"])
        packet = str(row["evidence_packet"])
        claims = json.loads(row["gold_claims_json"])
        if not isinstance(claims, list) or not claims:
            raise ValueError(f"{packet_id}: expected a non-empty gold claim list.")

        noisy_packet = packet.rstrip() + " " + NOISE_TEXT
        validate_gold_claims(packet_id, noisy_packet, claims)
        rows.append({
            "packet_id": f"{packet_id}__NOISY",
            "supplier_name": supplier_name,
            "category": category,
            "evidence_packet": noisy_packet,
            "gold_claims_json": json.dumps(claims, ensure_ascii=False),
        })

        removed_quote = str(claims[0]["source_quote"])
        incomplete_packet = remove_sentence_supporting_claim(packet, removed_quote)
        # Remove every gold claim whose supporting quote disappeared. This avoids
        # penalising the model for evidence that is no longer present.
        incomplete_claims = [
            claim for claim in claims
            if quote_in_packet(str(claim.get("source_quote", "")), incomplete_packet)
        ]
        validate_gold_claims(packet_id, incomplete_packet, incomplete_claims)
        rows.append({
            "packet_id": f"{packet_id}__INCOMPLETE",
            "supplier_name": supplier_name,
            "category": category,
            "evidence_packet": incomplete_packet,
            "gold_claims_json": json.dumps(incomplete_claims, ensure_ascii=False),
        })

        if include_conflicting:
            aspect = str(claims[0].get("aspect", "OTHER"))
            conflict_sentence = (
                f"A later review states that the earlier {aspect.lower().replace('_', ' ')} "
                "information is disputed and should not be treated as confirmed current evidence."
            )
            conflicting_packet = packet.rstrip() + " " + conflict_sentence
            conflict_claim = {
                "claim_id": f"{packet_id}_CONFLICT",
                "aspect": aspect,
                "polarity": "MIXED",
                "temporal_status": "CURRENT",
                "evidence_role": "ADVISORY",
                "source_quote": conflict_sentence,
                "normalized_claim": (
                    f"A later review disputes the current status of earlier {aspect.lower().replace('_', ' ')} evidence."
                ),
            }
            conflicting_claims = claims + [conflict_claim]
            validate_gold_claims(packet_id, conflicting_packet, conflicting_claims)
            rows.append({
                "packet_id": f"{packet_id}__CONFLICTING",
                "supplier_name": supplier_name,
                "category": category,
                "evidence_packet": conflicting_packet,
                "gold_claims_json": json.dumps(conflicting_claims, ensure_ascii=False),
            })

    variants = pd.DataFrame(rows)
    if variants["packet_id"].duplicated().any():
        raise ValueError("Generated duplicate packet identifiers.")
    return variants, original_ids


def run_e4(
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
        "--max-output-tokens", "2500",
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True)


def summarize(
    clean_metrics_path: Path,
    perturbed_metrics_path: Path,
    original_ids: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    if not clean_metrics_path.exists():
        raise FileNotFoundError(clean_metrics_path)
    if not perturbed_metrics_path.exists():
        raise FileNotFoundError(perturbed_metrics_path)

    clean = pd.read_csv(clean_metrics_path)
    perturbed = pd.read_csv(perturbed_metrics_path)

    required = {
        "packet_id",
        "claim_precision",
        "claim_recall",
        "claim_f1",
        "source_grounding_rate",
        "hallucination_rate",
        "over_extraction_rate",
        "missed_gold_rate",
    }
    for name, frame in [("clean", clean), ("perturbed", perturbed)]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} packet metrics missing columns: {missing}")

    clean = clean.loc[clean["packet_id"].astype(str).isin(original_ids)].copy()
    if set(clean["packet_id"].astype(str)) != set(original_ids):
        absent = sorted(set(original_ids) - set(clean["packet_id"].astype(str)))
        raise ValueError(f"Clean E4 metrics missing selected packets: {absent}")
    clean["PERTURBATION"] = "CLEAN"
    clean["SOURCE_PACKET_ID"] = clean["packet_id"].astype(str)

    split = perturbed["packet_id"].astype(str).str.rsplit("__", n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError("Perturbed packet identifiers do not contain '__<CONDITION>'.")
    perturbed = perturbed.copy()
    perturbed["SOURCE_PACKET_ID"] = split[0]
    perturbed["PERTURBATION"] = split[1]

    combined = pd.concat([clean, perturbed], ignore_index=True, sort=False)
    metrics = [
        "claim_precision",
        "claim_recall",
        "claim_f1",
        "source_grounding_rate",
        "hallucination_rate",
        "over_extraction_rate",
        "missed_gold_rate",
    ]
    rows = []
    for condition, group in combined.groupby("PERTURBATION", sort=False, observed=True):
        row = {"PERTURBATION": condition, "N_PACKETS": len(group)}
        for metric in metrics:
            row[f"MEAN_{metric.upper()}"] = pd.to_numeric(group[metric], errors="coerce").mean()
        rows.append(row)
    summary = pd.DataFrame(rows)

    clean_row = summary.loc[summary["PERTURBATION"].eq("CLEAN")]
    if clean_row.empty:
        raise RuntimeError("Clean condition is missing from summary.")
    clean_row = clean_row.iloc[0]
    for metric in metrics:
        column = f"MEAN_{metric.upper()}"
        summary[f"DELTA_{column}_VS_CLEAN"] = summary[column] - clean_row[column]

    combined.to_csv(output_dir / "r22_e4b_packet_metrics_combined.csv", index=False)
    summary.to_csv(output_dir / "r22_e4b_metrics_by_perturbation.csv", index=False)

    lines = [
        "# E4b Evidence-Input Robustness",
        "",
        "Clean metrics are reused from the frozen E4 run. API calls are made only "
        "for perturbed packets.",
        "",
        "| Condition | N | Precision | Recall | F1 | Grounding | Hallucination | Over-extraction | Missed gold | ΔF1 vs clean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['PERTURBATION']}` | {int(row['N_PACKETS'])} "
            f"| {row['MEAN_CLAIM_PRECISION']:.3f} "
            f"| {row['MEAN_CLAIM_RECALL']:.3f} "
            f"| {row['MEAN_CLAIM_F1']:.3f} "
            f"| {row['MEAN_SOURCE_GROUNDING_RATE']:.3f} "
            f"| {row['MEAN_HALLUCINATION_RATE']:.3f} "
            f"| {row['MEAN_OVER_EXTRACTION_RATE']:.3f} "
            f"| {row['MEAN_MISSED_GOLD_RATE']:.3f} "
            f"| {row['DELTA_MEAN_CLAIM_F1_VS_CLEAN']:.3f} |"
        )
    lines.extend([
        "",
        "For INCOMPLETE packets, removed claims are also removed from the gold set; "
        "the robust behaviour is to omit unsupported claims rather than reconstruct "
        "them. For CONFLICTING packets, the benchmark adds an explicit later dispute "
        "as a separate advisory claim; manual review of conflict handling remains necessary.",
        "",
    ])
    (output_dir / "r22_e4b_robustness_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--packets", type=int, default=15)
    parser.add_argument("--include-conflicting", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    runner = root / "scripts/run_e4_evidence_extraction_openai_v2.py"
    source_benchmark = root / "data/e4/e4_benchmark_40_gold.csv"
    clean_metrics = root / "data/analysis/e4_final/e4_packet_metrics.csv"
    data_dir = root / "data/analysis/r22_e4b_robustness"
    report_dir = root / "reports/r22_e4b_robustness"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not runner.exists():
        raise FileNotFoundError(runner)
    if not source_benchmark.exists():
        raise FileNotFoundError(source_benchmark)
    if not clean_metrics.exists():
        raise FileNotFoundError(clean_metrics)

    source = pd.read_csv(source_benchmark)
    required = {"packet_id", "supplier_name", "category", "evidence_packet", "gold_claims_json"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"E4 benchmark missing columns: {missing}")

    variants, original_ids = build_variants(
        source,
        args.packets,
        args.include_conflicting,
    )
    benchmark_out = data_dir / "r22_e4b_perturbation_benchmark.csv"
    variants.to_csv(benchmark_out, index=False, encoding="utf-8-sig")

    manifest = {
        "source_benchmark": str(source_benchmark),
        "clean_metrics_reused": str(clean_metrics),
        "perturbation_benchmark": str(benchmark_out),
        "source_packets": original_ids,
        "n_source_packets": len(original_ids),
        "n_api_packets": len(variants),
        "conditions": sorted(variants["packet_id"].str.rsplit("__", n=1).str[-1].unique()),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
    }
    (report_dir / "r22_e4b_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if not args.run_api:
        print(f"Prepared {len(variants)} perturbation packets; no API calls were made.")
        print(f"Benchmark: {benchmark_out}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    run_e4(
        runner,
        benchmark_out,
        data_dir,
        report_dir / "api_run",
        args.model,
        args.reasoning_effort,
        args.resume,
    )

    summarize(
        clean_metrics,
        data_dir / "e4_packet_metrics.csv",
        original_ids,
        report_dir,
    )
    print(f"E4b complete. Report: {report_dir / 'r22_e4b_robustness_report.md'}")


if __name__ == "__main__":
    main()
