#!/usr/bin/env python3
"""
E3: OpenAI API intent/scenario-formulation benchmark runner.

Purpose
-------
Evaluate an LLM intent-formulation component M_I without allowing the model to:
- rank suppliers,
- calculate TOPSIS scores,
- invent arbitrary numerical weights,
- use award outcomes.

Recommended benchmark
---------------------
50 base requests:
- 10 direct/simple
- 10 ambiguous
- 10 multi-criteria
- 10 constraint-sensitive
- 10 scenario-comparison

Each base request has V0 original, V1 paraphrase, V2 paraphrase: 150 calls total.

Requirements
------------
pip install -U openai pydantic pandas pyarrow

Environment
-----------
Set OPENAI_API_KEY before running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator


ScenarioProfile = Literal[
    "BALANCED",
    "CONTINUITY",
    "DIVERSIFICATION",
    "CUSTOM",
    "CLARIFICATION_REQUIRED",
]

PriorityGroup = Literal[
    "CATEGORY_FIT",
    "GEOGRAPHIC_FIT",
    "BUYER_RELATIONSHIP",
    "ACTIVITY_PROFILE",
]

ConstraintCategory = Literal[
    "ELIGIBILITY",
    "BUDGET",
    "GEOGRAPHY",
    "TIMING",
    "CONTRACT_STRUCTURE",
    "COMPLIANCE",
    "SUSTAINABILITY",
    "SME_INCLUSION",
    "TECHNICAL_REQUIREMENT",
    "OTHER",
]

Hardness = Literal["HARD", "SOFT"]

EvidenceType = Literal[
    "CERTIFICATION",
    "SUSTAINABILITY",
    "SOCIAL",
    "GOVERNANCE_COMPLIANCE",
    "TECHNICAL_CAPABILITY",
    "DELIVERY_CAPACITY",
    "PAST_PERFORMANCE",
    "FINANCIAL_STABILITY",
    "OTHER",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Constraint(StrictModel):
    category: ConstraintCategory
    hardness: Hardness
    normalized_value: str = Field(
        description=(
            "Short normalized value or requirement taken from the request. "
            "Use an empty string only when the category is explicit but no value is stated."
        )
    )


class EvidenceRequest(StrictModel):
    evidence_type: EvidenceType
    reason: str = Field(
        description="Short explanation of why this evidence is needed."
    )


class IntentFormulation(StrictModel):
    intent_status: Literal[
        "CLEAR",
        "AMBIGUOUS",
        "INSUFFICIENT_INFORMATION",
    ]
    scenario_profile: ScenarioProfile
    priority_groups: list[PriorityGroup] = Field(
        description=(
            "Only explicitly or strongly implied priority groups. "
            "Do not list every group by default."
        )
    )
    constraints: list[Constraint]
    evidence_requests: list[EvidenceRequest]
    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "One targeted clarification question when clarification is required; "
            "otherwise null."
        ),
    )
    rationale: str = Field(
        min_length=1,
        max_length=800,
        description=(
            "Brief decision rationale grounded only in the request and scenario catalogue."
        )
    )

    @model_validator(mode="after")
    def validate_governance_consistency(self):
        self.priority_groups = list(dict.fromkeys(self.priority_groups))

        constraint_keys = [
            (item.category, item.hardness, item.normalized_value.strip())
            for item in self.constraints
        ]
        if len(constraint_keys) != len(set(constraint_keys)):
            raise ValueError("Duplicate constraints are not allowed.")

        evidence_keys = [
            item.evidence_type
            for item in self.evidence_requests
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("Duplicate evidence requests are not allowed.")

        if self.scenario_profile == "CLARIFICATION_REQUIRED":
            if self.intent_status == "CLEAR":
                raise ValueError(
                    "CLARIFICATION_REQUIRED cannot have intent_status=CLEAR."
                )
            if not self.clarification_question or not self.clarification_question.strip():
                raise ValueError(
                    "CLARIFICATION_REQUIRED requires one non-empty clarification question."
                )
        else:
            if self.clarification_question not in (None, ""):
                raise ValueError(
                    "clarification_question must be null unless clarification is required."
                )
            self.clarification_question = None

        return self


SYSTEM_PROMPT = """
You are the intent-formulation component M_I in a governed public-procurement
supplier-selection framework.

Translate the user's procurement request into the supplied structured schema.

You MUST NOT:
- rank suppliers;
- calculate TOPSIS or any ranking score;
- invent numerical criterion weights;
- infer award outcomes;
- add constraints not stated or strongly implied;
- treat prior buyer relationship as general supplier quality;
- treat geography as general supplier quality.

Scenario catalogue:

BALANCED
Use when there is no special emphasis on continuity, incumbent relationships,
or geographic diversification.

CONTINUITY
Use when the request explicitly emphasizes continuity, transition stability,
buyer familiarity, established working relationships, or reduced handover risk.

DIVERSIFICATION
Use when the request emphasizes broadening the supplier base, reducing
dependence on established relationships, geographic market diversification,
or broader regional experience.

CUSTOM
Use when clear priorities are not adequately represented by the three fixed
profiles. Identify relevant priority groups, but do not invent numerical weights.

CLARIFICATION_REQUIRED
Use when material ambiguity or missing information prevents a defensible
scenario formulation. Ask exactly one targeted clarification question.

Conceptual groups:
- CATEGORY_FIT: procurement-category and CPV-set experience.
- GEOGRAPHIC_FIT: country and NUTS-level delivery-market experience.
- BUYER_RELATIONSHIP: prior buyer awards, buyer-specific recency, and
  relationship share; this is continuity evidence, not general quality.
- ACTIVITY_PROFILE: buyer breadth, public-activity recency, and context
  specialization.

Extract hard and soft constraints separately. Request qualitative evidence only
when the request creates an evidence need not fully represented by the structured
ranking groups.

Constraint handling rules:
- The scenario_profile reflects decision-priority emphasis. It must not change
  merely because a request contains hard or soft constraints.
- A constraint is not a priority group. For example, a hard regional-service
  requirement should be extracted as a GEOGRAPHY constraint, not as a
  GEOGRAPHIC_FIT priority group, unless the request also says to emphasize
  geographic market experience, regional supplier-base breadth, or similar.
- A hard deadline should be extracted as a TIMING constraint, not as an
  ACTIVITY_PROFILE priority group, unless the request also says to emphasize
  recent public activity.
- Do not request clarification merely because a placeholder value is used, such
  as "specified region", "delivery deadline", "approved budget", or "required
  certification". Extract the placeholder as the normalized_value.
- Ask clarification only when the decision-priority scenario itself is unclear,
  when the constraint is materially contradictory, or when the request cannot be
  converted into the schema without guessing.

Governance consistency rules:
- If scenario_profile is CLARIFICATION_REQUIRED, intent_status must be
  AMBIGUOUS or INSUFFICIENT_INFORMATION and clarification_question must contain
  exactly one targeted question.
- Otherwise clarification_question must be null.
- Do not duplicate priority groups, constraints, or evidence requests.

Be conservative. For material ambiguity, prefer CLARIFICATION_REQUIRED over
guessing. Keep the rationale brief.
""".strip()


def parse_json_list(value):
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got: {text}")
    return parsed


def set_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def usage_value(usage, *names):
    if usage is None:
        return None
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def python_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_optional_parquet(df: pd.DataFrame, path: Path) -> bool:
    """
    Write Parquet when pyarrow/fastparquet is installed. If not, continue with
    CSV-only output rather than failing after paid API calls have completed.
    """
    try:
        df.to_parquet(path, index=False)
        return True
    except ImportError as exc:
        print(
            f"WARNING: Could not write {path.name} because no Parquet engine is installed. "
            "CSV output was still written. Install pyarrow to enable Parquet output."
        )
        print(f"         {type(exc).__name__}: {exc}")
        return False


def is_retryable_exception(exc: Exception) -> bool:
    """
    Avoid depending on SDK exception re-exports, which have varied across
    openai-python releases. Use class names and status_code when available.
    """
    class_name = type(exc).__name__

    if class_name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }:
        return True

    status_code = getattr(exc, "status_code", None)

    if status_code is not None:
        try:
            return int(status_code) >= 500 or int(status_code) == 429
        except (TypeError, ValueError):
            return False

    # Structured-output parsing can occasionally fail because of an incomplete
    # or malformed model response. Retry schema/parse failures, but not clear
    # local programming or benchmark errors.
    return class_name in {
        "ValidationError",
        "JSONDecodeError",
        "APIResponseValidationError",
    }


def validate_benchmark(benchmark: pd.DataFrame) -> None:
    if benchmark.empty:
        raise ValueError("Benchmark is empty.")

    if benchmark["request_id"].isna().any():
        raise ValueError("request_id contains missing values.")

    if benchmark["request_id"].astype(str).duplicated().any():
        duplicates = (
            benchmark.loc[
                benchmark["request_id"].astype(str).duplicated(keep=False),
                "request_id",
            ]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(f"Duplicate request_id values: {duplicates}")

    allowed_categories = {
        "direct_simple",
        "ambiguous",
        "multi_criteria",
        "constraint_sensitive",
        "scenario_comparison",
    }
    allowed_scenarios = {
        "BALANCED",
        "CONTINUITY",
        "DIVERSIFICATION",
        "CUSTOM",
        "CLARIFICATION_REQUIRED",
    }
    allowed_groups = {
        "CATEGORY_FIT",
        "GEOGRAPHIC_FIT",
        "BUYER_RELATIONSHIP",
        "ACTIVITY_PROFILE",
    }
    allowed_constraint_categories = {
        "ELIGIBILITY",
        "BUDGET",
        "GEOGRAPHY",
        "TIMING",
        "CONTRACT_STRUCTURE",
        "COMPLIANCE",
        "SUSTAINABILITY",
        "SME_INCLUSION",
        "TECHNICAL_REQUIREMENT",
        "OTHER",
    }
    allowed_hardness = {"HARD", "SOFT"}
    allowed_evidence = {
        "CERTIFICATION",
        "SUSTAINABILITY",
        "SOCIAL",
        "GOVERNANCE_COMPLIANCE",
        "TECHNICAL_CAPABILITY",
        "DELIVERY_CAPACITY",
        "PAST_PERFORMANCE",
        "FINANCIAL_STABILITY",
        "OTHER",
    }

    unknown_categories = sorted(
        set(benchmark["category"].astype(str)) - allowed_categories
    )
    if unknown_categories:
        raise ValueError(f"Unknown benchmark categories: {unknown_categories}")

    unknown_scenarios = sorted(
        set(benchmark["gold_scenario_profile"].astype(str)) - allowed_scenarios
    )
    if unknown_scenarios:
        raise ValueError(f"Unknown gold scenarios: {unknown_scenarios}")

    for row_number, row in benchmark.iterrows():
        priorities = parse_json_list(row["gold_priority_groups_json"])
        constraints = parse_json_list(row["gold_constraints_json"])
        evidence = parse_json_list(row["gold_evidence_requests_json"])

        bad_groups = sorted(set(map(str, priorities)) - allowed_groups)
        if bad_groups:
            raise ValueError(
                f"Row {row_number}: unknown priority groups {bad_groups}"
            )

        bad_evidence = sorted(set(map(str, evidence)) - allowed_evidence)
        if bad_evidence:
            raise ValueError(
                f"Row {row_number}: unknown evidence types {bad_evidence}"
            )

        for item in constraints:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Row {row_number}: every gold constraint must be an object."
                )
            if item.get("category") not in allowed_constraint_categories:
                raise ValueError(
                    f"Row {row_number}: invalid constraint category {item.get('category')!r}"
                )
            if item.get("hardness") not in allowed_hardness:
                raise ValueError(
                    f"Row {row_number}: invalid constraint hardness {item.get('hardness')!r}"
                )

    # All variants of one base case must share the same gold annotations.
    gold_cols = [
        "category",
        "gold_scenario_profile",
        "gold_priority_groups_json",
        "gold_constraints_json",
        "gold_evidence_requests_json",
        "gold_requires_clarification",
    ]
    inconsistent = []
    for base_case_id, group in benchmark.groupby("base_case_id", sort=False):
        if any(group[col].astype(str).nunique(dropna=False) != 1 for col in gold_cols):
            inconsistent.append(str(base_case_id))
    if inconsistent:
        raise ValueError(
            "Variants have inconsistent gold annotations for base cases: "
            + ", ".join(inconsistent[:10])
        )


def load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid checkpoint JSON on line {line_number}: {exc}"
                ) from exc
            records[str(record["request_id"])] = record
    return records


def append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def call_with_retry(
    client: OpenAI,
    model: str,
    request_text: str,
    max_retries: int,
    retry_base_seconds: float,
    reasoning_effort: str,
    max_output_tokens: int,
    store_responses: bool,
):
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()

            response = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": request_text},
                ],
                text_format=IntentFormulation,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_output_tokens,
                store=store_responses,
            )

            latency = time.perf_counter() - started
            parsed = response.output_parsed

            if parsed is None:
                raise RuntimeError("Response completed without output_parsed.")

            return response, parsed, latency, attempt

        except Exception as exc:
            last_error = exc

            if attempt >= max_retries or not is_retryable_exception(exc):
                break

            delay = (
                retry_base_seconds * (2 ** attempt)
                + random.uniform(0.0, 0.5)
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Request failed after {max_retries + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def gold_priority_set(row: pd.Series) -> set[str]:
    return set(
        str(x)
        for x in parse_json_list(row["gold_priority_groups_json"])
    )


def gold_constraint_set(row: pd.Series) -> set[str]:
    result = set()
    for item in parse_json_list(row["gold_constraints_json"]):
        category = str(item.get("category", "")).strip()
        hardness = str(item.get("hardness", "")).strip()
        if category and hardness:
            result.add(f"{category}::{hardness}")
    return result


def gold_evidence_set(row: pd.Series) -> set[str]:
    return set(
        str(x)
        for x in parse_json_list(row["gold_evidence_requests_json"])
    )


def pred_priority_set(pred: dict) -> set[str]:
    return set(str(x) for x in pred.get("priority_groups", []))


def pred_constraint_set(pred: dict) -> set[str]:
    return {
        f"{item['category']}::{item['hardness']}"
        for item in pred.get("constraints", [])
    }


def pred_evidence_set(pred: dict) -> set[str]:
    return {
        str(item["evidence_type"])
        for item in pred.get("evidence_requests", [])
    }


def evaluate_predictions(results: pd.DataFrame) -> pd.DataFrame:
    evaluated = results.copy()

    scenario_correct = []
    clarification_correct = []
    priority_exact = []
    priority_f1 = []
    constraint_f1 = []
    evidence_f1 = []

    for _, row in evaluated.iterrows():
        if row["status"] != "OK":
            scenario_correct.append(False)
            clarification_correct.append(False)
            priority_exact.append(False)
            priority_f1.append(0.0)
            constraint_f1.append(0.0)
            evidence_f1.append(0.0)
            continue

        pred = json.loads(row["prediction_json"])

        gold_scenario = str(row["gold_scenario_profile"])
        pred_scenario = str(pred["scenario_profile"])

        scenario_correct.append(pred_scenario == gold_scenario)

        gold_requires = bool(row["gold_requires_clarification"])
        pred_requires = pred_scenario == "CLARIFICATION_REQUIRED"

        clarification_correct.append(pred_requires == gold_requires)

        gp = gold_priority_set(row)
        pp = pred_priority_set(pred)
        priority_exact.append(gp == pp)
        priority_f1.append(set_f1(pp, gp))

        gc = gold_constraint_set(row)
        pc = pred_constraint_set(pred)
        constraint_f1.append(set_f1(pc, gc))

        ge = gold_evidence_set(row)
        pe = pred_evidence_set(pred)
        evidence_f1.append(set_f1(pe, ge))

    evaluated["SCENARIO_CORRECT"] = scenario_correct
    evaluated["CLARIFICATION_CORRECT"] = clarification_correct
    evaluated["PRIORITY_GROUP_EXACT"] = priority_exact
    evaluated["PRIORITY_GROUP_F1"] = priority_f1
    evaluated["CONSTRAINT_F1"] = constraint_f1
    evaluated["EVIDENCE_REQUEST_F1"] = evidence_f1

    return evaluated


def summarize_metrics(evaluated: pd.DataFrame) -> pd.DataFrame:
    successful = evaluated.loc[evaluated["status"].eq("OK")]

    rows = [
        ("API_SUCCESS_RATE", evaluated["status"].eq("OK").mean()),
        ("SCENARIO_ACCURACY_END_TO_END", evaluated["SCENARIO_CORRECT"].mean()),
        (
            "SCENARIO_ACCURACY_SUCCESSFUL_ONLY",
            successful["SCENARIO_CORRECT"].mean() if len(successful) else float("nan"),
        ),
        ("CLARIFICATION_ACCURACY_END_TO_END", evaluated["CLARIFICATION_CORRECT"].mean()),
        (
            "CLARIFICATION_ACCURACY_SUCCESSFUL_ONLY",
            successful["CLARIFICATION_CORRECT"].mean() if len(successful) else float("nan"),
        ),
        ("PRIORITY_GROUP_EXACT_MATCH", evaluated["PRIORITY_GROUP_EXACT"].mean()),
        ("PRIORITY_GROUP_SET_F1", evaluated["PRIORITY_GROUP_F1"].mean()),
        ("CONSTRAINT_SET_F1", evaluated["CONSTRAINT_F1"].mean()),
        ("EVIDENCE_REQUEST_SET_F1", evaluated["EVIDENCE_REQUEST_F1"].mean()),
        (
            "MEAN_LATENCY_SECONDS",
            evaluated.loc[
                evaluated["status"] == "OK",
                "latency_seconds",
            ].mean(),
        ),
    ]

    return pd.DataFrame(rows, columns=["METRIC", "VALUE"])


def metrics_by_category(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category, g in evaluated.groupby(
        "category",
        sort=False,
        observed=True,
    ):
        rows.append(
            {
                "CATEGORY": category,
                "N": len(g),
                "SCENARIO_ACCURACY": g["SCENARIO_CORRECT"].mean(),
                "CLARIFICATION_ACCURACY": g["CLARIFICATION_CORRECT"].mean(),
                "PRIORITY_GROUP_EXACT_MATCH": g["PRIORITY_GROUP_EXACT"].mean(),
                "PRIORITY_GROUP_SET_F1": g["PRIORITY_GROUP_F1"].mean(),
                "CONSTRAINT_SET_F1": g["CONSTRAINT_F1"].mean(),
                "EVIDENCE_REQUEST_SET_F1": g["EVIDENCE_REQUEST_F1"].mean(),
            }
        )

    return pd.DataFrame(rows)


def paraphrase_consistency(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for base_case_id, g in evaluated.groupby(
        "base_case_id",
        sort=False,
        observed=True,
    ):
        successful = g.loc[g["status"] == "OK"]

        if len(successful) < 2:
            rows.append(
                {
                    "base_case_id": base_case_id,
                    "N_VARIANTS": len(g),
                    "N_SUCCESSFUL": len(successful),
                    "SCENARIO_FULL_CONSISTENCY": False,
                    "MEAN_PAIRWISE_PRIORITY_JACCARD": 0.0,
                }
            )
            continue

        preds = [
            json.loads(x)
            for x in successful["prediction_json"]
        ]

        scenarios = [p["scenario_profile"] for p in preds]
        priority_sets = [pred_priority_set(p) for p in preds]

        pairwise = []
        for i in range(len(priority_sets)):
            for j in range(i + 1, len(priority_sets)):
                pairwise.append(
                    jaccard(priority_sets[i], priority_sets[j])
                )

        rows.append(
            {
                "base_case_id": base_case_id,
                "N_VARIANTS": len(g),
                "N_SUCCESSFUL": len(successful),
                "SCENARIO_FULL_CONSISTENCY": len(set(scenarios)) == 1,
                "MEAN_PAIRWISE_PRIORITY_JACCARD": (
                    sum(pairwise) / len(pairwise)
                    if pairwise
                    else 1.0
                ),
            }
        )

    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    category_metrics: pd.DataFrame,
    consistency: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_map = dict(zip(summary["METRIC"], summary["VALUE"]))

    scenario_consistency = (
        consistency["SCENARIO_FULL_CONSISTENCY"].mean()
        if not consistency.empty
        else float("nan")
    )

    priority_consistency = (
        consistency["MEAN_PAIRWISE_PRIORITY_JACCARD"].mean()
        if not consistency.empty
        else float("nan")
    )

    lines = [
        "# E3 Intent and Scenario Formulation Evaluation",
        "",
        "## Overall results",
        "",
        f"- API success rate: {metric_map.get('API_SUCCESS_RATE', float('nan')):.3f}",
        f"- Scenario accuracy, end-to-end: {metric_map.get('SCENARIO_ACCURACY_END_TO_END', float('nan')):.3f}",
        f"- Scenario accuracy, successful calls only: {metric_map.get('SCENARIO_ACCURACY_SUCCESSFUL_ONLY', float('nan')):.3f}",
        f"- Clarification accuracy, end-to-end: {metric_map.get('CLARIFICATION_ACCURACY_END_TO_END', float('nan')):.3f}",
        f"- Clarification accuracy, successful calls only: {metric_map.get('CLARIFICATION_ACCURACY_SUCCESSFUL_ONLY', float('nan')):.3f}",
        f"- Priority-group exact match: {metric_map.get('PRIORITY_GROUP_EXACT_MATCH', float('nan')):.3f}",
        f"- Priority-group set F1: {metric_map.get('PRIORITY_GROUP_SET_F1', float('nan')):.3f}",
        f"- Constraint set F1: {metric_map.get('CONSTRAINT_SET_F1', float('nan')):.3f}",
        f"- Evidence-request set F1: {metric_map.get('EVIDENCE_REQUEST_SET_F1', float('nan')):.3f}",
        f"- Base-case scenario consistency across paraphrases: {scenario_consistency:.3f}",
        f"- Mean pairwise priority-group Jaccard across paraphrases: {priority_consistency:.3f}",
        "",
        "## Results by request category",
        "",
        "| Category | N | Scenario accuracy | Clarification accuracy | Priority F1 | Constraint F1 | Evidence F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in category_metrics.iterrows():
        lines.append(
            f"| `{row['CATEGORY']}` "
            f"| {int(row['N'])} "
            f"| {row['SCENARIO_ACCURACY']:.3f} "
            f"| {row['CLARIFICATION_ACCURACY']:.3f} "
            f"| {row['PRIORITY_GROUP_SET_F1']:.3f} "
            f"| {row['CONSTRAINT_SET_F1']:.3f} "
            f"| {row['EVIDENCE_REQUEST_SET_F1']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "E3 evaluates intent formulation rather than supplier ranking. "
            "The LLM maps procurement requests into governed scenario labels, "
            "priority groups, constraints, and evidence needs. Numerical ranking "
            "weights remain deterministic and external to the LLM.",
            "",
            "Ambiguous requests are evaluated on whether the model requests "
            "clarification instead of silently inventing priorities.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/e3_openai"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/e3_openai"),
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="low",
        help="Frozen reasoning effort for the benchmark.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the JSONL checkpoint in the output directory.",
    )
    parser.add_argument(
        "--store-responses",
        action="store_true",
        help=(
            "Store Responses API objects server-side. Useful only for a tiny "
            "dashboard-visible smoke test. Keep disabled for the final run unless "
            "you intentionally want platform-side response storage."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pilot limit, e.g. --limit 10",
    )

    args = parser.parse_args()

    if not args.benchmark.exists():
        raise FileNotFoundError(args.benchmark)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(args.benchmark)

    required_cols = {
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
    }

    missing = sorted(required_cols - set(benchmark.columns))

    if missing:
        raise ValueError(
            "Benchmark missing columns:\n  - "
            + "\n  - ".join(missing)
        )

    benchmark["gold_requires_clarification"] = (
        benchmark["gold_requires_clarification"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )

    if benchmark["gold_requires_clarification"].isna().any():
        raise ValueError(
            "gold_requires_clarification contains unrecognized values."
        )

    validate_benchmark(benchmark)

    if args.limit is not None:
        benchmark = benchmark.head(args.limit).copy()

    checkpoint_path = args.output_dir / "e3_checkpoint.jsonl"

    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. "
            "Use --resume or choose a new output directory."
        )

    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    records = list(completed.values())

    client = OpenAI(
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    run_started = datetime.now(timezone.utc).isoformat()

    total = len(benchmark)
    api_requests_made = 0
    checkpointed_requests = 0

    for i, (_, row) in enumerate(
        benchmark.iterrows(),
        start=1,
    ):
        request_id = str(row["request_id"])

        if request_id in completed:
            checkpointed_requests += 1
            print(f"[{i}/{total}] {request_id} (checkpointed; no API call made)")
            continue

        print(f"[{i}/{total}] {request_id}")

        base_record = {
            key: python_scalar(value)
            for key, value in row.to_dict().items()
        }

        try:
            response, parsed, latency, retry_count = call_with_retry(
                client=client,
                model=args.model,
                request_text=str(row["request_text"]),
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                store_responses=args.store_responses,
            )
            api_requests_made += 1

            pred_dict = parsed.model_dump(mode="json")
            usage = getattr(response, "usage", None)

            records.append(
                {
                    **base_record,
                    "status": "OK",
                    "api_call_made": True,
                    "store_responses": bool(args.store_responses),
                    "model": args.model,
                    "response_id": getattr(response, "id", None),
                    "latency_seconds": latency,
                    "retry_count": retry_count,
                    "input_tokens": usage_value(
                        usage,
                        "input_tokens",
                        "prompt_tokens",
                    ),
                    "output_tokens": usage_value(
                        usage,
                        "output_tokens",
                        "completion_tokens",
                    ),
                    "prediction_json": json.dumps(
                        pred_dict,
                        ensure_ascii=False,
                    ),
                    "error": None,
                }
            )
            append_checkpoint(checkpoint_path, records[-1])

        except Exception as exc:
            records.append(
                {
                    **base_record,
                    "status": "ERROR",
                    "api_call_made": True,
                    "store_responses": bool(args.store_responses),
                    "model": args.model,
                    "response_id": None,
                    "latency_seconds": None,
                    "retry_count": args.max_retries,
                    "input_tokens": None,
                    "output_tokens": None,
                    "prediction_json": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            append_checkpoint(checkpoint_path, records[-1])

    # Keep only requests in the current benchmark selection and preserve its order.
    record_map = {
        str(record["request_id"]): record
        for record in records
    }
    selected_ids = benchmark["request_id"].astype(str).tolist()
    missing_results = [
        request_id
        for request_id in selected_ids
        if request_id not in record_map
    ]
    if missing_results:
        raise RuntimeError(
            f"Missing results for {len(missing_results)} requests: "
            f"{missing_results[:10]}"
        )

    raw_results = pd.DataFrame(
        [record_map[request_id] for request_id in selected_ids]
    )
    evaluated = evaluate_predictions(raw_results)
    summary = summarize_metrics(evaluated)
    category_metrics = metrics_by_category(evaluated)
    consistency = paraphrase_consistency(evaluated)

    parquet_written = write_optional_parquet(
        evaluated,
        args.output_dir / "e3_api_results.parquet",
    )
    evaluated.to_csv(
        args.output_dir / "e3_api_results.csv",
        index=False,
    )

    summary.to_csv(
        args.report_dir / "e3_summary_metrics.csv",
        index=False,
    )
    category_metrics.to_csv(
        args.report_dir / "e3_metrics_by_category.csv",
        index=False,
    )
    consistency.to_csv(
        args.report_dir / "e3_paraphrase_consistency.csv",
        index=False,
    )

    run_metadata = {
        "model": args.model,
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_requests": int(total),
        "n_completed_api_responses_this_run": int(api_requests_made),
        "n_checkpointed_requests_skipped_this_run": int(checkpointed_requests),
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "benchmark_sha256": sha256_file(args.benchmark),
        "store": False,
        "parquet_written": bool(parquet_written),
        "system_prompt": SYSTEM_PROMPT,
        "schema": IntentFormulation.model_json_schema(),
    }

    (
        args.report_dir / "e3_run_metadata.json"
    ).write_text(
        json.dumps(
            run_metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_report(
        summary,
        category_metrics,
        consistency,
        args.report_dir / "e3_report.md",
    )

    print()
    print(f"Completed API responses this run: {api_requests_made}")
    print(f"Checkpointed requests skipped this run: {checkpointed_requests}")
    if api_requests_made == 0:
        print("WARNING: no new API responses were created in this run.")
    if not args.store_responses:
        print("Note: --store-responses was not enabled, so responses may not appear in stored response logs/history.")
    print()
    print("E3 complete.")
    print("Review first:")
    print("  e3_report.md")
    print("  e3_summary_metrics.csv")
    print("  e3_metrics_by_category.csv")
    print("  e3_paraphrase_consistency.csv")


if __name__ == "__main__":
    main()
