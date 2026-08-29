#!/usr/bin/env python3
"""
E2c: controlled cross-modal profit, lead-time, and ESG trade-off experiment.

Purpose
-------
Directly address Reviewer 2.2's request for a multi-criteria experiment that
combines structured economic/delivery criteria with qualitative ESG evidence.

The experiment is intentionally controlled:
- profit per unit and lead time are deterministic structured features;
- ESG evidence is provided as short source-linked packets;
- M_E extracts ESG claims using the existing E4 runner;
- a deterministic policy maps extracted claim attributes to ESG evidence levels;
- TOPSIS computes all rankings;
- gold-evidence and extracted-evidence rankings are compared.

This is a controlled cross-modal benchmark, not a field validation and not a
claim that ESG can be reduced to a universally valid scalar.

Time-conscious default
----------------------
12 cases x 6 suppliers = 72 API packets.
Use --cases 6 for a 36-call smoke test.

No API call is made unless --run-api is supplied.

Requirements
------------
pip install numpy pandas pyarrow openai pydantic
Set OPENAI_API_KEY when using --run-api.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STATUS_POLICY = {
    "CERTIFIED_CURRENT": {"score": 1.00, "eligible": True},
    "MIXED_CURRENT": {"score": 0.50, "eligible": True},
    "CERTIFICATION_EXPIRED": {"score": 0.25, "eligible": True},
    "MATERIAL_VIOLATION": {"score": 0.00, "eligible": False},
    "NO_SUPPORTED_EVIDENCE": {"score": 0.00, "eligible": True},
}

SCENARIOS = {
    "ECONOMIC_PRIORITY": {
        "profit_weight": 0.55,
        "lead_weight": 0.35,
        "esg_weight": 0.10,
        "exclude_violation": False,
    },
    "BALANCED": {
        "profit_weight": 0.40,
        "lead_weight": 0.30,
        "esg_weight": 0.30,
        "exclude_violation": False,
    },
    "ESG_PRIORITY": {
        "profit_weight": 0.25,
        "lead_weight": 0.25,
        "esg_weight": 0.50,
        "exclude_violation": False,
    },
    "ESG_CONSTRAINED": {
        "profit_weight": 0.45,
        "lead_weight": 0.35,
        "esg_weight": 0.20,
        "exclude_violation": True,
    },
}

STATUS_CYCLE = [
    "CERTIFIED_CURRENT",
    "MIXED_CURRENT",
    "CERTIFICATION_EXPIRED",
    "MATERIAL_VIOLATION",
    "CERTIFIED_CURRENT",
    "MIXED_CURRENT",
]


def evidence_for_status(status: str, supplier_name: str, case_no: int) -> tuple[str, list[dict]]:
    claim_id = f"C{case_no:03d}_{supplier_name.replace(' ', '_')}_ESG"
    if status == "CERTIFIED_CURRENT":
        quote = "holds ISO 14001 certification valid until 31 December 2027"
        text = (
            f"{supplier_name} {quote}. The certification record was checked for "
            "the current sourcing review."
        )
        claim = {
            "claim_id": claim_id,
            "aspect": "CERTIFICATION",
            "polarity": "POSITIVE",
            "temporal_status": "CURRENT",
            "evidence_role": "FEATURE",
            "source_quote": quote,
            "normalized_claim": (
                f"{supplier_name} holds current ISO 14001 certification valid until 2027."
            ),
        }
    elif status == "MIXED_CURRENT":
        quote = (
            "has a documented emissions-reduction plan, but the latest report "
            "does not verify achieved reductions"
        )
        text = f"{supplier_name} {quote}. The plan remains active."
        claim = {
            "claim_id": claim_id,
            "aspect": "SUSTAINABILITY",
            "polarity": "MIXED",
            "temporal_status": "CURRENT",
            "evidence_role": "FEATURE",
            "source_quote": quote,
            "normalized_claim": (
                f"{supplier_name} has a current emissions plan, but achieved reductions are unverified."
            ),
        }
    elif status == "CERTIFICATION_EXPIRED":
        quote = (
            "ISO 14001 certification expired on 31 December 2024 and no renewal "
            "is documented"
        )
        text = f"For {supplier_name}, the {quote}."
        claim = {
            "claim_id": claim_id,
            "aspect": "CERTIFICATION",
            "polarity": "NEGATIVE",
            "temporal_status": "EXPIRED",
            "evidence_role": "FEATURE",
            "source_quote": quote,
            "normalized_claim": (
                f"{supplier_name}'s ISO 14001 certification is expired and renewal is undocumented."
            ),
        }
    elif status == "MATERIAL_VIOLATION":
        quote = (
            "a current regulatory notice records a material environmental "
            "compliance violation"
        )
        text = f"For {supplier_name}, {quote}. The notice remains unresolved."
        claim = {
            "claim_id": claim_id,
            "aspect": "GOVERNANCE_COMPLIANCE",
            "polarity": "NEGATIVE",
            "temporal_status": "CURRENT",
            "evidence_role": "ELIGIBILITY",
            "source_quote": quote,
            "normalized_claim": (
                f"{supplier_name} has a current unresolved material environmental compliance violation."
            ),
        }
    else:
        raise ValueError(f"Unknown ESG status: {status}")
    return text, [claim]


def build_controlled_benchmark(
    n_cases: int,
    suppliers_per_case: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if suppliers_per_case < 6:
        raise ValueError("Use at least six suppliers per case to represent all ESG conditions.")
    rng = np.random.default_rng(seed)
    structured_rows: list[dict] = []
    evidence_rows: list[dict] = []

    for case_no in range(1, n_cases + 1):
        # Independent permutations create explicit economic/delivery trade-offs.
        profit_base = np.linspace(35.0, 105.0, suppliers_per_case)
        lead_base = np.linspace(2.0, 18.0, suppliers_per_case)
        profit_values = rng.permutation(profit_base) + rng.normal(0.0, 2.0, suppliers_per_case)
        lead_values = rng.permutation(lead_base) + rng.normal(0.0, 0.35, suppliers_per_case)
        statuses = [STATUS_CYCLE[i % len(STATUS_CYCLE)] for i in range(suppliers_per_case)]
        statuses = list(rng.permutation(statuses))

        for supplier_no in range(1, suppliers_per_case + 1):
            case_id = f"XMOD_{case_no:03d}"
            supplier_id = f"{case_id}_S{supplier_no:02d}"
            supplier_name = f"Case {case_no:03d} Supplier {supplier_no:02d}"
            status = statuses[supplier_no - 1]
            policy = STATUS_POLICY[status]
            packet_id = f"{supplier_id}_ESG"
            evidence_text, gold_claims = evidence_for_status(status, supplier_name, case_no)

            structured_rows.append(
                {
                    "case_id": case_id,
                    "supplier_id": supplier_id,
                    "supplier_name": supplier_name,
                    "profit_per_unit": round(float(profit_values[supplier_no - 1]), 4),
                    "lead_time_days": round(float(max(0.25, lead_values[supplier_no - 1])), 4),
                    "gold_esg_status": status,
                    "gold_esg_score": policy["score"],
                    "gold_esg_eligible": policy["eligible"],
                    "packet_id": packet_id,
                }
            )
            evidence_rows.append(
                {
                    "packet_id": packet_id,
                    "supplier_name": supplier_name,
                    "category": "cross_modal_esg_tradeoff",
                    "evidence_packet": evidence_text,
                    "gold_claims_json": json.dumps(gold_claims, ensure_ascii=False),
                }
            )

    structured = pd.DataFrame(structured_rows)
    evidence = pd.DataFrame(evidence_rows)

    # Validation: every case must contain all four policy statuses.
    required_statuses = set(STATUS_POLICY) - {"NO_SUPPORTED_EVIDENCE"}
    for case_id, g in structured.groupby("case_id", observed=True):
        missing = required_statuses - set(g["gold_esg_status"])
        if missing:
            raise ValueError(f"{case_id} is missing ESG statuses: {sorted(missing)}")
    return structured, evidence



def build_from_structured_input(
    path: Path,
    n_cases: int,
    suppliers_per_case: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    source = pd.read_csv(path)
    required = {
        "case_id",
        "supplier_id",
        "supplier_name",
        "profit_per_unit",
        "lead_time_days",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Structured input missing columns: {missing}")
    if source.duplicated(["case_id", "supplier_id"]).any():
        raise ValueError("Structured input has duplicate case_id/supplier_id rows.")
    source["profit_per_unit"] = pd.to_numeric(source["profit_per_unit"], errors="raise")
    source["lead_time_days"] = pd.to_numeric(source["lead_time_days"], errors="raise")
    if (source["lead_time_days"] < 0).any():
        raise ValueError("lead_time_days must be non-negative.")

    rng = np.random.default_rng(seed)
    case_ids = list(dict.fromkeys(source["case_id"].astype(str)))[:n_cases]
    structured_rows: list[dict] = []
    evidence_rows: list[dict] = []
    for case_no, case_id in enumerate(case_ids, start=1):
        g = source.loc[source["case_id"].astype(str).eq(case_id)].copy()
        if len(g) < suppliers_per_case:
            raise ValueError(
                f"Case {case_id} has {len(g)} suppliers; need {suppliers_per_case}."
            )
        if len(g) > suppliers_per_case:
            g = g.sample(n=suppliers_per_case, random_state=seed + case_no)
        statuses = [STATUS_CYCLE[i % len(STATUS_CYCLE)] for i in range(len(g))]
        statuses = list(rng.permutation(statuses))
        for row_no, (_, row) in enumerate(g.reset_index(drop=True).iterrows()):
            status = statuses[row_no]
            policy = STATUS_POLICY[status]
            packet_id = f"{row['supplier_id']}_ESG_{case_no:03d}"
            evidence_text, gold_claims = evidence_for_status(
                status, str(row["supplier_name"]), case_no
            )
            structured_rows.append(
                {
                    "case_id": str(case_id),
                    "supplier_id": str(row["supplier_id"]),
                    "supplier_name": str(row["supplier_name"]),
                    "profit_per_unit": float(row["profit_per_unit"]),
                    "lead_time_days": float(row["lead_time_days"]),
                    "gold_esg_status": status,
                    "gold_esg_score": policy["score"],
                    "gold_esg_eligible": policy["eligible"],
                    "packet_id": packet_id,
                }
            )
            evidence_rows.append(
                {
                    "packet_id": packet_id,
                    "supplier_name": str(row["supplier_name"]),
                    "category": "cross_modal_esg_tradeoff",
                    "evidence_packet": evidence_text,
                    "gold_claims_json": json.dumps(gold_claims, ensure_ascii=False),
                }
            )
    if not structured_rows:
        raise ValueError("Structured input did not provide any usable cases.")
    return pd.DataFrame(structured_rows), pd.DataFrame(evidence_rows)

def run_e4(
    root: Path,
    benchmark_path: Path,
    model: str,
    reasoning_effort: str,
    resume: bool,
) -> Path:
    runner = root / "scripts/run_e4_evidence_extraction_openai_v2.py"
    if not runner.exists():
        raise FileNotFoundError(runner)
    output_dir = root / "data/analysis/r22_e2c_cross_modal/e4_extraction"
    report_dir = root / "reports/r22_e2c_cross_modal/e4_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(runner),
        "--benchmark",
        str(benchmark_path),
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
        "--max-output-tokens",
        "1400",
        "--output-dir",
        str(output_dir),
        "--report-dir",
        str(report_dir),
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True)
    return output_dir / "e4_api_results.csv"


def classify_prediction(prediction_json: str | float | None) -> tuple[str, float, bool]:
    if prediction_json is None or (isinstance(prediction_json, float) and np.isnan(prediction_json)):
        status = "NO_SUPPORTED_EVIDENCE"
        policy = STATUS_POLICY[status]
        return status, policy["score"], policy["eligible"]
    try:
        payload = json.loads(str(prediction_json))
    except json.JSONDecodeError:
        status = "NO_SUPPORTED_EVIDENCE"
        policy = STATUS_POLICY[status]
        return status, policy["score"], policy["eligible"]
    claims = payload.get("claims", []) if isinstance(payload, dict) else []
    if not claims:
        status = "NO_SUPPORTED_EVIDENCE"
    else:
        def has(predicate):
            return any(predicate(c) for c in claims if isinstance(c, dict))

        if has(
            lambda c: c.get("polarity") == "NEGATIVE"
            and c.get("temporal_status") == "CURRENT"
            and c.get("aspect") in {"GOVERNANCE_COMPLIANCE", "SUSTAINABILITY"}
        ):
            status = "MATERIAL_VIOLATION"
        elif has(lambda c: c.get("temporal_status") == "EXPIRED"):
            status = "CERTIFICATION_EXPIRED"
        elif has(
            lambda c: c.get("polarity") == "POSITIVE"
            and c.get("temporal_status") == "CURRENT"
            and c.get("aspect") in {"CERTIFICATION", "SUSTAINABILITY"}
        ):
            status = "CERTIFIED_CURRENT"
        else:
            status = "MIXED_CURRENT"
    policy = STATUS_POLICY[status]
    return status, policy["score"], policy["eligible"]


def topsis_scores(df: pd.DataFrame, scenario: dict, esg_prefix: str) -> pd.Series:
    eligible_col = f"{esg_prefix}_esg_eligible"
    score_col = f"{esg_prefix}_esg_score"
    work = df.copy()
    if scenario["exclude_violation"]:
        work = work.loc[work[eligible_col].astype(bool)].copy()
    if work.empty:
        return pd.Series(dtype=float)

    matrix = np.column_stack(
        [
            pd.to_numeric(work["profit_per_unit"], errors="raise").to_numpy(float),
            pd.to_numeric(work["lead_time_days"], errors="raise").to_numpy(float),
            pd.to_numeric(work[score_col], errors="raise").to_numpy(float),
        ]
    )
    norms = np.sqrt(np.square(matrix).sum(axis=0))
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    weights = np.array(
        [scenario["profit_weight"], scenario["lead_weight"], scenario["esg_weight"]],
        dtype=float,
    )
    weighted = normalized * weights
    ideal_best = np.array(
        [weighted[:, 0].max(), weighted[:, 1].min(), weighted[:, 2].max()]
    )
    ideal_worst = np.array(
        [weighted[:, 0].min(), weighted[:, 1].max(), weighted[:, 2].min()]
    )
    d_best = np.sqrt(np.square(weighted - ideal_best).sum(axis=1))
    d_worst = np.sqrt(np.square(weighted - ideal_worst).sum(axis=1))
    denom = d_best + d_worst
    scores = np.divide(d_worst, denom, out=np.full_like(d_worst, 0.5), where=denom > 0)
    return pd.Series(scores, index=work.index)


def rank_case(df: pd.DataFrame, scenario_name: str, scenario: dict, esg_prefix: str) -> pd.DataFrame:
    scores = topsis_scores(df, scenario, esg_prefix)
    result = df.copy()
    result["scenario"] = scenario_name
    result["evidence_source"] = esg_prefix.upper()
    result["eligible_for_scenario"] = True
    if scenario["exclude_violation"]:
        result["eligible_for_scenario"] = result[f"{esg_prefix}_esg_eligible"].astype(bool)
    result["topsis_score"] = np.nan
    result.loc[scores.index, "topsis_score"] = scores
    eligible = result.loc[result["eligible_for_scenario"]].copy()
    eligible = eligible.sort_values(
        ["topsis_score", "supplier_id"], ascending=[False, True], kind="stable"
    )
    eligible["rank"] = np.arange(1, len(eligible) + 1)
    result = result.merge(
        eligible[["supplier_id", "rank"]], on="supplier_id", how="left", validate="one_to_one"
    )
    return result


def spearman_rank(gold: pd.DataFrame, pred: pd.DataFrame) -> float:
    merged = gold[["supplier_id", "rank"]].rename(columns={"rank": "gold_rank"}).merge(
        pred[["supplier_id", "rank"]].rename(columns={"rank": "pred_rank"}),
        on="supplier_id",
        how="outer",
    )
    max_rank = len(merged) + 1
    merged["gold_rank"] = merged["gold_rank"].fillna(max_rank)
    merged["pred_rank"] = merged["pred_rank"].fillna(max_rank)
    if merged["gold_rank"].nunique() <= 1 or merged["pred_rank"].nunique() <= 1:
        return 1.0
    return float(merged[["gold_rank", "pred_rank"]].corr(method="spearman").iloc[0, 1])


def evaluate(structured: pd.DataFrame, api_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"packet_id", "status", "prediction_json"}
    missing = required - set(api_results.columns)
    if missing:
        raise ValueError(f"E4 API results missing columns: {sorted(missing)}")
    pred_rows = []
    for _, row in api_results.iterrows():
        status, score, eligible = classify_prediction(row["prediction_json"])
        pred_rows.append(
            {
                "packet_id": str(row["packet_id"]),
                "api_status": row["status"],
                "pred_esg_status": status,
                "pred_esg_score": score,
                "pred_esg_eligible": eligible,
            }
        )
    pred = pd.DataFrame(pred_rows)
    data = structured.merge(pred, on="packet_id", how="left", validate="one_to_one")
    if data["pred_esg_status"].isna().any():
        missing_n = int(data["pred_esg_status"].isna().sum())
        raise ValueError(f"Missing extracted ESG outputs for {missing_n} suppliers.")

    ranking_parts = []
    case_metrics = []
    for case_id, case_df in data.groupby("case_id", sort=False, observed=True):
        for scenario_name, scenario in SCENARIOS.items():
            gold = rank_case(case_df, scenario_name, scenario, "gold")
            pred_ranked = rank_case(case_df, scenario_name, scenario, "pred")
            ranking_parts.extend([gold, pred_ranked])

            gold_eligible = gold.loc[gold["eligible_for_scenario"]].sort_values("rank")
            pred_eligible = pred_ranked.loc[pred_ranked["eligible_for_scenario"]].sort_values("rank")
            gold_top1 = gold_eligible.iloc[0]["supplier_id"] if len(gold_eligible) else None
            pred_top1 = pred_eligible.iloc[0]["supplier_id"] if len(pred_eligible) else None
            gold_top3 = set(gold_eligible.head(3)["supplier_id"])
            pred_top3 = set(pred_eligible.head(3)["supplier_id"])
            case_metrics.append(
                {
                    "case_id": case_id,
                    "scenario": scenario_name,
                    "top1_agreement": gold_top1 == pred_top1,
                    "top3_overlap": len(gold_top3 & pred_top3) / max(len(gold_top3), 1),
                    "rank_spearman": spearman_rank(gold, pred_ranked),
                    "gold_top3_mean_esg": gold_eligible.head(3)["gold_esg_score"].mean(),
                    "pred_top3_mean_gold_esg": pred_eligible.head(3)["gold_esg_score"].mean(),
                }
            )

    rankings = pd.concat(ranking_parts, ignore_index=True)
    case_metrics_df = pd.DataFrame(case_metrics)
    status_accuracy = data["gold_esg_status"].eq(data["pred_esg_status"])
    eligibility_accuracy = data["gold_esg_eligible"].eq(data["pred_esg_eligible"])

    summary_rows = []
    for scenario, g in case_metrics_df.groupby("scenario", sort=False, observed=True):
        summary_rows.append(
            {
                "SCENARIO": scenario,
                "N_CASES": len(g),
                "TOP1_AGREEMENT": g["top1_agreement"].mean(),
                "MEAN_TOP3_OVERLAP": g["top3_overlap"].mean(),
                "MEAN_RANK_SPEARMAN": g["rank_spearman"].mean(),
                "MEAN_GOLD_TOP3_ESG": g["gold_top3_mean_esg"].mean(),
                "MEAN_PREDICTED_TOP3_GOLD_ESG": g["pred_top3_mean_gold_esg"].mean(),
                "STATUS_ACCURACY": status_accuracy.mean(),
                "ELIGIBILITY_ACCURACY": eligibility_accuracy.mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return data, rankings, summary


def write_report(summary: pd.DataFrame, n_suppliers: int, output_path: Path) -> None:
    lines = [
        "# E2c Cross-Modal Profit--Lead-Time--ESG Trade-Offs",
        "",
        "This controlled benchmark combines deterministic profitability and lead-time "
        "features with source-grounded ESG evidence. The LLM extracts claims only; "
        "a deterministic policy maps claim attributes to evidence levels, and TOPSIS "
        "computes the rankings.",
        "",
        f"Suppliers evaluated: {n_suppliers}",
        "",
        "| Scenario | N cases | Top-1 agreement | Mean top-3 overlap | Mean Spearman | Status accuracy | Eligibility accuracy | Gold top-3 ESG | Extracted top-3 gold ESG |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r['SCENARIO']}` | {int(r['N_CASES'])} "
            f"| {r['TOP1_AGREEMENT']:.3f} | {r['MEAN_TOP3_OVERLAP']:.3f} "
            f"| {r['MEAN_RANK_SPEARMAN']:.3f} | {r['STATUS_ACCURACY']:.3f} "
            f"| {r['ELIGIBILITY_ACCURACY']:.3f} | {r['MEAN_GOLD_TOP3_ESG']:.3f} "
            f"| {r['MEAN_PREDICTED_TOP3_GOLD_ESG']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- The benchmark tests cross-modal analytical behaviour, not realised ESG performance.",
            "- ESG evidence levels are policy mappings for this controlled experiment, not a "
            "universal ESG scoring standard.",
            "- Profit and lead-time values are author-controlled structured inputs. The experiment "
            "should be reported separately from the real TED applicability study.",
            "- The main validity question is whether extracted evidence preserves gold-evidence "
            "ranking and eligibility decisions across transparent scenarios.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--suppliers-per-case", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--structured-input",
        type=Path,
        default=None,
        help=(
            "Optional CSV with case_id, supplier_id, supplier_name, "
            "profit_per_unit, and lead_time_days. ESG packets remain controlled."
        ),
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    data_dir = root / "data/e2c"
    analysis_dir = root / "data/analysis/r22_e2c_cross_modal"
    report_dir = root / "reports/r22_e2c_cross_modal"
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.structured_input is not None:
        structured_input = args.structured_input
        if not structured_input.is_absolute():
            structured_input = root / structured_input
        structured, evidence = build_from_structured_input(
            structured_input, args.cases, args.suppliers_per_case, args.seed
        )
        structured_source = str(structured_input)
        controlled_structured_metrics = False
    else:
        structured, evidence = build_controlled_benchmark(
            args.cases, args.suppliers_per_case, args.seed
        )
        structured_source = "seeded_author_controlled"
        controlled_structured_metrics = True
    structured_path = data_dir / "r22_e2c_structured_supplier_metrics.csv"
    benchmark_path = data_dir / "r22_e2c_esg_evidence_benchmark.csv"
    structured.to_csv(structured_path, index=False, encoding="utf-8-sig")
    evidence.to_csv(benchmark_path, index=False, encoding="utf-8-sig")

    manifest = {
        "n_cases": args.cases,
        "suppliers_per_case": args.suppliers_per_case,
        "n_api_packets": int(len(evidence)),
        "seed": args.seed,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "scenarios": SCENARIOS,
        "status_policy": STATUS_POLICY,
        "controlled_benchmark": True,
        "structured_source": structured_source,
        "controlled_structured_metrics": controlled_structured_metrics,
    }
    (report_dir / "r22_e2c_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    api_path = analysis_dir / "e4_extraction/e4_api_results.csv"
    if args.run_api:
        api_path = run_e4(
            root,
            benchmark_path,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            resume=args.resume,
        )

    if not api_path.exists():
        print(f"Prepared {len(evidence)} ESG packets; no API calls were made.")
        print(f"Benchmark: {benchmark_path}")
        print("Run again with --run-api --resume to execute extraction and ranking evaluation.")
        return

    api_results = pd.read_csv(api_path)
    supplier_results, rankings, summary = evaluate(structured, api_results)
    supplier_results.to_csv(
        analysis_dir / "r22_e2c_supplier_esg_results.csv", index=False, encoding="utf-8-sig"
    )
    rankings.to_csv(
        analysis_dir / "r22_e2c_rankings.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        report_dir / "r22_e2c_cross_modal_summary.csv", index=False
    )
    write_report(
        summary,
        n_suppliers=len(structured),
        output_path=report_dir / "r22_e2c_cross_modal_report.md",
    )
    print(f"E2c complete. Report: {report_dir / 'r22_e2c_cross_modal_report.md'}")


if __name__ == "__main__":
    main()
