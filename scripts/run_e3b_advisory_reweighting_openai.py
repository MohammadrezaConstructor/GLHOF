#!/usr/bin/env python3
"""
E3b: LLM-assisted advisory reweighting evaluation.

Purpose
-------
Evaluate a complementary advisory module M_W.

M_W receives:
- a procurement request;
- the initial scenario profile;
- deterministic base group weights;
- a neutral deterministic ranking summary.

M_W must NOT output final numerical weights. It only recommends whether the
weights should be adjusted, and if so, which feature groups should increase,
decrease, or be preserved, plus a coarse adjustment strength.

A deterministic policy controller maps the advisory direction into valid
bounded numerical weights.

Input CSV columns
-----------------
case_id
category
user_request
initial_scenario_profile
base_weights_json
ranking_summary_json
gold_recommendation_type
gold_increase_json
gold_decrease_json
gold_preserve_json
gold_adjustment_strength
gold_requires_human_approval
gold_clarification_required
gold_risk_flags_json

Outputs
-------
Processed:
- e3b_api_results.csv
- e3b_case_metrics.csv
- e3b_predicted_weights.csv

Reports:
- e3b_summary_metrics.csv
- e3b_metrics_by_category.csv
- e3b_error_rows.csv
- e3b_run_metadata.json
- e3b_report.md

Example
-------
python scripts/run_e3b_advisory_reweighting_openai.py `
  --benchmark data/e3b/e3b_benchmark_20_gold.csv `
  --model gpt-5.6-terra `
  --reasoning-effort low `
  --output-dir data/analysis/e3b_final `
  --report-dir reports/e3b_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator


Group = Literal[
    "CATEGORY_FIT",
    "GEOGRAPHIC_FIT",
    "BUYER_RELATIONSHIP",
    "ACTIVITY_PROFILE",
]

RecommendationType = Literal[
    "NO_ADJUST",
    "ADJUST",
    "CLARIFY",
]

AdjustmentStrength = Literal[
    "NONE",
    "SMALL",
    "MODERATE",
    "LARGE",
]

RiskFlag = Literal[
    "NONE",
    "AMBIGUOUS_REQUEST",
    "CONFLICTING_PRIORITIES",
    "CHANGES_BUYER_RELATIONSHIP",
    "CHANGES_GEOGRAPHIC_EMPHASIS",
    "CHANGES_CATEGORY_EMPHASIS",
    "CHANGES_ACTIVITY_EMPHASIS",
    "RANKING_STABILITY_RISK",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PriorityDirection(StrictModel):
    increase: list[Group]
    decrease: list[Group]
    preserve: list[Group]

    @model_validator(mode="after")
    def dedupe_and_validate(self):
        for attr in ["increase", "decrease", "preserve"]:
            seen = set()
            deduped = []
            for item in getattr(self, attr):
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            setattr(self, attr, deduped)

        inc = set(self.increase)
        dec = set(self.decrease)
        pre = set(self.preserve)

        # A group cannot be both increased and decreased.
        if inc & dec:
            raise ValueError(
                f"Groups cannot be both increased and decreased: {sorted(inc & dec)}"
            )

        # Preserve is allowed to be incomplete, but it must not conflict.
        if pre & inc or pre & dec:
            raise ValueError(
                "Preserve groups cannot also be listed as increase or decrease."
            )

        return self


class AdvisoryReweightingOutput(StrictModel):
    case_id: str
    recommendation_type: RecommendationType
    priority_direction: PriorityDirection
    adjustment_strength: AdjustmentStrength
    requires_human_approval: bool
    clarification_question: str | None = Field(default=None, max_length=500)
    risk_flags: list[RiskFlag]
    rationale: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_consistency(self):
        # Deduplicate flags while preserving order.
        seen = set()
        deduped = []
        for flag in self.risk_flags:
            if flag not in seen:
                seen.add(flag)
                deduped.append(flag)
        self.risk_flags = deduped or ["NONE"]

        if "NONE" in self.risk_flags and len(self.risk_flags) > 1:
            self.risk_flags = [f for f in self.risk_flags if f != "NONE"]

        if self.recommendation_type == "NO_ADJUST":
            if self.adjustment_strength != "NONE":
                raise ValueError("NO_ADJUST requires adjustment_strength = NONE.")
            if self.priority_direction.increase or self.priority_direction.decrease:
                raise ValueError("NO_ADJUST requires empty increase/decrease lists.")
            if self.clarification_question is not None:
                raise ValueError("NO_ADJUST requires clarification_question = null.")

        if self.recommendation_type == "CLARIFY":
            if not self.clarification_question:
                raise ValueError("CLARIFY requires a clarification question.")
            if self.adjustment_strength != "NONE":
                raise ValueError("CLARIFY requires adjustment_strength = NONE.")
            if self.priority_direction.increase or self.priority_direction.decrease:
                raise ValueError("CLARIFY requires empty increase/decrease lists.")

        if self.recommendation_type == "ADJUST":
            if self.adjustment_strength == "NONE":
                raise ValueError("ADJUST requires SMALL, MODERATE, or LARGE strength.")
            if not self.priority_direction.increase and not self.priority_direction.decrease:
                raise ValueError("ADJUST requires at least one increase or decrease group.")
            if self.clarification_question is not None:
                raise ValueError("ADJUST requires clarification_question = null.")

        return self


SYSTEM_PROMPT = """
You are the advisory scenario-weight refinement module M_W in a governed public
procurement supplier-selection framework.

Your task is NOT to produce final numerical MCDM weights.

You receive:
- a procurement request;
- an initial scenario profile;
- deterministic base group weights;
- a neutral summary of the deterministic ranking profile.

You must output only:
- whether a weight adjustment is advisable;
- which feature groups should increase, decrease, or be preserved;
- a coarse adjustment strength;
- whether additional/escalated human approval is required;
- any risk flags;
- a short rationale.

The four feature groups are:
- CATEGORY_FIT: procurement category and CPV/domain fit.
- GEOGRAPHIC_FIT: country, region, NUTS, local delivery coverage.
- BUYER_RELATIONSHIP: prior buyer relationship, continuity, incumbency/repeat-buyer evidence.
- ACTIVITY_PROFILE: recency, breadth, public-sector activity, specialization.

Governance rules:
- Do not output numerical weights.
- Do not rank suppliers.
- Do not mention observed winners or optimality.
- Do not recommend changing weights merely because one supplier is ranked first.
- Recommend ADJUST only when the request expresses a clear priority not already
  captured by the initial scenario.
- Recommend NO_ADJUST when the initial scenario already matches the user's stated
  priority or when the request asks for standard/balanced treatment.
- For NO_ADJUST, leave increase and decrease empty and list all four feature
  groups in preserve.
- Recommend CLARIFY when the request contains unresolved competing priorities or
  insufficient information to determine a direction.
- For CLARIFY, leave increase and decrease empty and list all four feature
  groups in preserve until the analyst provides clarification.
- requires_human_approval should be true for any ADJUST recommendation, because
  changing scenario weights is a consequential governance action.
- requires_human_approval should also be true for CLARIFY.
- If recommendation_type is NO_ADJUST, requires_human_approval should normally
  be false unless the request itself flags a governance issue.
- Use SMALL for mild emphasis, MODERATE for clear but bounded emphasis, and LARGE
  only when the user explicitly makes one priority dominant.
""".strip()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

GROUPS = [
    "CATEGORY_FIT",
    "GEOGRAPHIC_FIT",
    "BUYER_RELATIONSHIP",
    "ACTIVITY_PROFILE",
]

ALLOWED_RECOMMENDATION = {"NO_ADJUST", "ADJUST", "CLARIFY"}
ALLOWED_STRENGTH = {"NONE", "SMALL", "MODERATE", "LARGE"}
ALLOWED_FLAGS = {
    "NONE",
    "AMBIGUOUS_REQUEST",
    "CONFLICTING_PRIORITIES",
    "CHANGES_BUYER_RELATIONSHIP",
    "CHANGES_GEOGRAPHIC_EMPHASIS",
    "CHANGES_CATEGORY_EMPHASIS",
    "CHANGES_ACTIVITY_EMPHASIS",
    "RANKING_STABILITY_RISK",
}

DELTA_BY_STRENGTH = {
    "NONE": 0.0,
    "SMALL": 0.05,
    "MODERATE": 0.10,
    "LARGE": 0.15,
}

MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.45


def parse_json(value, label: str):
    if value is None or pd.isna(value):
        raise ValueError(f"{label} is missing.")
    return json.loads(str(value))


def parse_json_list(value, label: str) -> list:
    parsed = parse_json(value, label)
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must be a JSON list.")
    return parsed


def parse_json_dict(value, label: str) -> dict:
    parsed = parse_json(value, label)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def bool_from_value(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def set_f1(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def normalize_flags(flags: list[str]) -> set[str]:
    out = set(str(f) for f in flags)
    if "NONE" in out and len(out) > 1:
        out.remove("NONE")
    return out or {"NONE"}


def python_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def usage_value(usage, *names):
    if usage is None:
        return None
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_retryable_exception(exc: Exception) -> bool:
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
    return class_name in {
        "ValidationError",
        "JSONDecodeError",
        "APIResponseValidationError",
    }


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
            records[str(record["case_id"])] = record
    return records


def append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def try_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:
        print(f"WARNING: could not write {path.name}: {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------
# Deterministic policy controller
# ---------------------------------------------------------------------

def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(float(weights.get(g, 0.0)) for g in GROUPS)
    if total <= 0:
        raise ValueError("Weight total must be positive.")
    return {g: float(weights.get(g, 0.0)) / total for g in GROUPS}


def apply_bounds_and_renormalize(weights: dict[str, float]) -> dict[str, float]:
    w = {g: min(MAX_WEIGHT, max(MIN_WEIGHT, float(weights.get(g, 0.0)))) for g in GROUPS}

    # Iteratively distribute surplus/deficit to groups not at their bounds.
    for _ in range(20):
        total = sum(w.values())
        diff = 1.0 - total
        if abs(diff) < 1e-10:
            break

        if diff > 0:
            adjustable = [g for g in GROUPS if w[g] < MAX_WEIGHT - 1e-12]
            if not adjustable:
                break
            add = diff / len(adjustable)
            for g in adjustable:
                w[g] = min(MAX_WEIGHT, w[g] + add)
        else:
            adjustable = [g for g in GROUPS if w[g] > MIN_WEIGHT + 1e-12]
            if not adjustable:
                break
            sub = (-diff) / len(adjustable)
            for g in adjustable:
                w[g] = max(MIN_WEIGHT, w[g] - sub)

    # Final exact normalization with negligible correction to largest adjustable group.
    diff = 1.0 - sum(w.values())
    if abs(diff) > 1e-9:
        candidates = [g for g in GROUPS if MIN_WEIGHT <= w[g] + diff <= MAX_WEIGHT]
        if candidates:
            w[candidates[0]] += diff

    return {g: round(float(w[g]), 6) for g in GROUPS}


def deterministic_controller(
    base_weights: dict[str, float],
    recommendation_type: str,
    increase: list[str],
    decrease: list[str],
    strength: str,
) -> tuple[dict[str, float], bool]:
    base = normalize_weights(base_weights)

    if recommendation_type != "ADJUST" or strength == "NONE":
        bounded = apply_bounds_and_renormalize(base)
        return bounded, check_weight_bounds(bounded)

    delta = DELTA_BY_STRENGTH[strength]
    inc = [g for g in increase if g in GROUPS]
    dec = [g for g in decrease if g in GROUPS]

    if not inc and not dec:
        bounded = apply_bounds_and_renormalize(base)
        return bounded, check_weight_bounds(bounded)

    proposed = dict(base)

    if inc and dec:
        inc_add = delta / len(inc)
        dec_sub = delta / len(dec)
        for g in inc:
            proposed[g] += inc_add
        for g in dec:
            proposed[g] -= dec_sub
    elif inc:
        inc_add = delta / len(inc)
        dec_pool = [g for g in GROUPS if g not in inc]
        dec_sub = delta / len(dec_pool)
        for g in inc:
            proposed[g] += inc_add
        for g in dec_pool:
            proposed[g] -= dec_sub
    else:
        dec_sub = delta / len(dec)
        inc_pool = [g for g in GROUPS if g not in dec]
        inc_add = delta / len(inc_pool)
        for g in dec:
            proposed[g] -= dec_sub
        for g in inc_pool:
            proposed[g] += inc_add

    bounded = apply_bounds_and_renormalize(proposed)
    return bounded, check_weight_bounds(bounded)


def check_weight_bounds(weights: dict[str, float]) -> bool:
    if abs(sum(float(weights.get(g, 0.0)) for g in GROUPS) - 1.0) > 1e-6:
        return False
    for g in GROUPS:
        value = float(weights.get(g, 0.0))
        if value < MIN_WEIGHT - 1e-9 or value > MAX_WEIGHT + 1e-9:
            return False
    return True


# ---------------------------------------------------------------------
# Benchmark validation
# ---------------------------------------------------------------------

def validate_benchmark(df: pd.DataFrame) -> None:
    required = {
        "case_id",
        "category",
        "user_request",
        "initial_scenario_profile",
        "base_weights_json",
        "ranking_summary_json",
        "gold_recommendation_type",
        "gold_increase_json",
        "gold_decrease_json",
        "gold_preserve_json",
        "gold_adjustment_strength",
        "gold_requires_human_approval",
        "gold_clarification_required",
        "gold_risk_flags_json",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Benchmark missing columns:\n  - " + "\n  - ".join(missing))

    if df.empty:
        raise ValueError("Benchmark is empty.")

    if df["case_id"].astype(str).duplicated().any():
        dupes = (
            df.loc[df["case_id"].astype(str).duplicated(keep=False), "case_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(f"Duplicate case_id values: {dupes}")

    for _, row in df.iterrows():
        case_id = str(row["case_id"])

        weights = parse_json_dict(row["base_weights_json"], f"{case_id}.base_weights_json")
        if set(weights) != set(GROUPS):
            raise ValueError(f"{case_id}: base_weights_json must contain exactly {GROUPS}")
        if not check_weight_bounds(normalize_weights(weights)):
            raise ValueError(f"{case_id}: base weights fail bounds/sum check.")

        _ = parse_json_dict(row["ranking_summary_json"], f"{case_id}.ranking_summary_json")

        rec = str(row["gold_recommendation_type"])
        if rec not in ALLOWED_RECOMMENDATION:
            raise ValueError(f"{case_id}: invalid gold recommendation {rec!r}")

        inc = parse_json_list(row["gold_increase_json"], f"{case_id}.gold_increase_json")
        dec = parse_json_list(row["gold_decrease_json"], f"{case_id}.gold_decrease_json")
        pre = parse_json_list(row["gold_preserve_json"], f"{case_id}.gold_preserve_json")

        if not set(inc).issubset(GROUPS):
            raise ValueError(f"{case_id}: invalid gold increase groups.")
        if not set(dec).issubset(GROUPS):
            raise ValueError(f"{case_id}: invalid gold decrease groups.")
        if not set(pre).issubset(GROUPS):
            raise ValueError(f"{case_id}: invalid gold preserve groups.")

        strength = str(row["gold_adjustment_strength"])
        if strength not in ALLOWED_STRENGTH:
            raise ValueError(f"{case_id}: invalid strength {strength!r}")

        flags = parse_json_list(row["gold_risk_flags_json"], f"{case_id}.gold_risk_flags_json")
        if not set(flags).issubset(ALLOWED_FLAGS):
            raise ValueError(f"{case_id}: invalid flags {flags}")


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------

def build_user_input(row: pd.Series) -> str:
    payload = {
        "case_id": str(row["case_id"]),
        "user_request": str(row["user_request"]),
        "initial_scenario_profile": str(row["initial_scenario_profile"]),
        "base_group_weights": parse_json_dict(row["base_weights_json"], "base_weights_json"),
        "ranking_summary": parse_json_dict(row["ranking_summary_json"], "ranking_summary_json"),
        "instruction": (
            "Return an advisory reweighting recommendation. Do not output numerical weights. "
            "The deterministic controller will apply any approved direction."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_with_retry(
    client: OpenAI,
    *,
    model: str,
    row: pd.Series,
    max_retries: int,
    retry_base_seconds: float,
    reasoning_effort: str,
    max_output_tokens: int,
    store_responses: bool,
):
    last_error = None
    user_input = build_user_input(row)

    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()

            response = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                text_format=AdvisoryReweightingOutput,
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

            delay = retry_base_seconds * (2 ** attempt) + random.uniform(0.0, 0.5)
            time.sleep(delay)

    raise RuntimeError(
        f"Request failed after {max_retries + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_case(row: pd.Series) -> dict:
    case_id = str(row["case_id"])
    category = str(row["category"])
    status = str(row["status"])

    gold_rec = str(row["gold_recommendation_type"])
    gold_inc = set(parse_json_list(row["gold_increase_json"], f"{case_id}.gold_increase_json"))
    gold_dec = set(parse_json_list(row["gold_decrease_json"], f"{case_id}.gold_decrease_json"))
    gold_pre = set(parse_json_list(row["gold_preserve_json"], f"{case_id}.gold_preserve_json"))
    gold_strength = str(row["gold_adjustment_strength"])
    gold_approval = bool_from_value(row["gold_requires_human_approval"])
    gold_clarify = bool_from_value(row["gold_clarification_required"])
    gold_flags = normalize_flags(parse_json_list(row["gold_risk_flags_json"], f"{case_id}.gold_risk_flags_json"))

    base_weights = parse_json_dict(row["base_weights_json"], f"{case_id}.base_weights_json")

    if status != "OK" or not row.get("prediction_json"):
        pred_rec = "ERROR"
        pred_inc = set()
        pred_dec = set()
        pred_pre = set()
        pred_strength = "ERROR"
        pred_approval = False
        pred_clarify = False
        pred_flags = set()
        pred_weights, pred_bounds_ok = deterministic_controller(base_weights, "NO_ADJUST", [], [], "NONE")
    else:
        pred = json.loads(row["prediction_json"])
        pred_rec = str(pred.get("recommendation_type"))
        direction = pred.get("priority_direction", {})
        pred_inc = set(str(x) for x in direction.get("increase", []))
        pred_dec = set(str(x) for x in direction.get("decrease", []))
        pred_pre = set(str(x) for x in direction.get("preserve", []))
        pred_strength = str(pred.get("adjustment_strength"))
        pred_approval = bool(pred.get("requires_human_approval"))
        pred_clarify = pred_rec == "CLARIFY" or bool(pred.get("clarification_question"))
        pred_flags = normalize_flags(pred.get("risk_flags", []))
        pred_weights, pred_bounds_ok = deterministic_controller(
            base_weights,
            pred_rec,
            sorted(pred_inc),
            sorted(pred_dec),
            pred_strength,
        )

    decision_correct = pred_rec == gold_rec
    strength_correct = pred_strength == gold_strength
    approval_correct = pred_approval == gold_approval
    clarification_correct = pred_clarify == gold_clarify

    increase_f1 = set_f1(gold_inc, pred_inc)
    decrease_f1 = set_f1(gold_dec, pred_dec)
    preserve_f1 = set_f1(gold_pre, pred_pre)
    direction_macro_f1 = (increase_f1 + decrease_f1 + preserve_f1) / 3.0
    risk_flag_f1 = set_f1(gold_flags, pred_flags)

    # Controller output for the gold label, useful for stability/output inspection.
    gold_weights, gold_bounds_ok = deterministic_controller(
        base_weights,
        gold_rec,
        sorted(gold_inc),
        sorted(gold_dec),
        gold_strength,
    )

    weight_l1_distance_to_gold = sum(
        abs(float(pred_weights[g]) - float(gold_weights[g]))
        for g in GROUPS
    )

    return {
        "case_id": case_id,
        "category": category,
        "status": status,
        "decision_correct": decision_correct,
        "increase_f1": increase_f1,
        "decrease_f1": decrease_f1,
        "preserve_f1": preserve_f1,
        "direction_macro_f1": direction_macro_f1,
        "strength_correct": strength_correct,
        "approval_correct": approval_correct,
        "clarification_correct": clarification_correct,
        "risk_flag_f1": risk_flag_f1,
        "pred_bounds_ok": pred_bounds_ok,
        "gold_bounds_ok": gold_bounds_ok,
        "bound_violation": not pred_bounds_ok,
        "weight_l1_distance_to_gold": weight_l1_distance_to_gold,
        "gold_recommendation_type": gold_rec,
        "pred_recommendation_type": pred_rec,
        "gold_adjustment_strength": gold_strength,
        "pred_adjustment_strength": pred_strength,
        "gold_increase_json": json.dumps(sorted(gold_inc), ensure_ascii=False),
        "pred_increase_json": json.dumps(sorted(pred_inc), ensure_ascii=False),
        "gold_decrease_json": json.dumps(sorted(gold_dec), ensure_ascii=False),
        "pred_decrease_json": json.dumps(sorted(pred_dec), ensure_ascii=False),
        "gold_preserve_json": json.dumps(sorted(gold_pre), ensure_ascii=False),
        "pred_preserve_json": json.dumps(sorted(pred_pre), ensure_ascii=False),
        "gold_risk_flags_json": json.dumps(sorted(gold_flags), ensure_ascii=False),
        "pred_risk_flags_json": json.dumps(sorted(pred_flags), ensure_ascii=False),
        "pred_weights_json": json.dumps(pred_weights, ensure_ascii=False),
        "gold_controller_weights_json": json.dumps(gold_weights, ensure_ascii=False),
    }


def evaluate_all(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([evaluate_case(row) for _, row in results.iterrows()])


def summarize_metrics(results: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("API_SUCCESS_RATE", results["status"].eq("OK").mean()),
        ("ADJUSTMENT_DECISION_ACCURACY", metrics["decision_correct"].mean()),
        ("INCREASE_DIRECTION_F1", metrics["increase_f1"].mean()),
        ("DECREASE_DIRECTION_F1", metrics["decrease_f1"].mean()),
        ("PRESERVE_DIRECTION_F1", metrics["preserve_f1"].mean()),
        ("DIRECTION_MACRO_F1", metrics["direction_macro_f1"].mean()),
        ("STRENGTH_ACCURACY", metrics["strength_correct"].mean()),
        ("APPROVAL_FLAG_ACCURACY", metrics["approval_correct"].mean()),
        ("CLARIFICATION_ACCURACY", metrics["clarification_correct"].mean()),
        ("RISK_FLAG_F1", metrics["risk_flag_f1"].mean()),
        ("BOUND_VIOLATION_RATE", metrics["bound_violation"].mean()),
        ("MEAN_WEIGHT_L1_DISTANCE_TO_GOLD_CONTROLLER", metrics["weight_l1_distance_to_gold"].mean()),
        ("N_CASES", len(metrics)),
    ]
    return pd.DataFrame(rows, columns=["METRIC", "VALUE"])


def summarize_by_category(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, g in metrics.groupby("category", sort=False, observed=True):
        rows.append({
            "CATEGORY": category,
            "N": len(g),
            "ADJUSTMENT_DECISION_ACCURACY": g["decision_correct"].mean(),
            "DIRECTION_MACRO_F1": g["direction_macro_f1"].mean(),
            "STRENGTH_ACCURACY": g["strength_correct"].mean(),
            "APPROVAL_FLAG_ACCURACY": g["approval_correct"].mean(),
            "CLARIFICATION_ACCURACY": g["clarification_correct"].mean(),
            "RISK_FLAG_F1": g["risk_flag_f1"].mean(),
            "BOUND_VIOLATION_RATE": g["bound_violation"].mean(),
        })
    return pd.DataFrame(rows)


def error_rows(results: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    merged = metrics.merge(
        results[["case_id", "user_request", "initial_scenario_profile", "prediction_json", "error"]],
        on="case_id",
        how="left",
    )
    return merged.loc[
        (~merged["decision_correct"])
        | (merged["direction_macro_f1"] < 1.0)
        | (~merged["strength_correct"])
        | (~merged["approval_correct"])
        | (~merged["clarification_correct"])
        | (merged["risk_flag_f1"] < 1.0)
        | (merged["bound_violation"])
    ].copy()


def write_report(summary: pd.DataFrame, by_category: pd.DataFrame, path: Path) -> None:
    m = dict(zip(summary["METRIC"], summary["VALUE"]))

    lines = [
        "# E3b Advisory Scenario-Weight Refinement Evaluation",
        "",
        "## Overall results",
        "",
        f"- API success rate: {m.get('API_SUCCESS_RATE', float('nan')):.3f}",
        f"- Adjustment-decision accuracy: {m.get('ADJUSTMENT_DECISION_ACCURACY', float('nan')):.3f}",
        f"- Increase-direction F1: {m.get('INCREASE_DIRECTION_F1', float('nan')):.3f}",
        f"- Decrease-direction F1: {m.get('DECREASE_DIRECTION_F1', float('nan')):.3f}",
        f"- Preserve-direction F1: {m.get('PRESERVE_DIRECTION_F1', float('nan')):.3f}",
        f"- Direction macro F1: {m.get('DIRECTION_MACRO_F1', float('nan')):.3f}",
        f"- Strength accuracy: {m.get('STRENGTH_ACCURACY', float('nan')):.3f}",
        f"- Approval-flag accuracy: {m.get('APPROVAL_FLAG_ACCURACY', float('nan')):.3f}",
        f"- Clarification accuracy: {m.get('CLARIFICATION_ACCURACY', float('nan')):.3f}",
        f"- Risk-flag F1: {m.get('RISK_FLAG_F1', float('nan')):.3f}",
        f"- Bound-violation rate: {m.get('BOUND_VIOLATION_RATE', float('nan')):.3f}",
        f"- Mean L1 distance to gold controller weights: {m.get('MEAN_WEIGHT_L1_DISTANCE_TO_GOLD_CONTROLLER', float('nan')):.3f}",
        f"- Cases: {int(m.get('N_CASES', 0))}",
        "",
        "## Results by case category",
        "",
        "| Category | N | Decision accuracy | Direction F1 | Strength accuracy | Approval accuracy | Clarification accuracy | Risk F1 | Bound violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in by_category.iterrows():
        lines.append(
            f"| `{row['CATEGORY']}` "
            f"| {int(row['N'])} "
            f"| {row['ADJUSTMENT_DECISION_ACCURACY']:.3f} "
            f"| {row['DIRECTION_MACRO_F1']:.3f} "
            f"| {row['STRENGTH_ACCURACY']:.3f} "
            f"| {row['APPROVAL_FLAG_ACCURACY']:.3f} "
            f"| {row['CLARIFICATION_ACCURACY']:.3f} "
            f"| {row['RISK_FLAG_F1']:.3f} "
            f"| {row['BOUND_VIOLATION_RATE']:.3f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "E3b is a complementary governance test. It does not allow the LLM to output final numerical weights. "
        "The LLM only recommends a direction and strength of adjustment. A deterministic controller maps any "
        "approved recommendation into bounded valid weights. This preserves the main framework principle that "
        "numerical ranking remains deterministic and auditable.",
        "",
        "Error rows should be manually reviewed before final reporting, especially where the model selects the "
        "right adjustment decision but differs on preserve groups or risk flags.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/e3b_openai"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/e3b_openai"))
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="low",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--store-responses", action="store_true")

    args = parser.parse_args()

    if not args.benchmark.exists():
        raise FileNotFoundError(args.benchmark)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(args.benchmark)
    validate_benchmark(benchmark)

    if args.limit is not None:
        benchmark = benchmark.head(args.limit).copy()

    checkpoint_path = args.output_dir / "e3b_checkpoint.jsonl"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. "
            "Use --resume or choose a new output directory."
        )

    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    records = list(completed.values())

    client = OpenAI(timeout=args.timeout_seconds, max_retries=0)
    run_started = datetime.now(timezone.utc).isoformat()

    total = len(benchmark)
    api_requests_made = 0
    checkpointed_requests = 0

    for i, (_, row) in enumerate(benchmark.iterrows(), start=1):
        case_id = str(row["case_id"])

        if case_id in completed:
            checkpointed_requests += 1
            print(f"[{i}/{total}] {case_id} (checkpointed; no API call made)")
            continue

        print(f"[{i}/{total}] {case_id}")

        base_record = {
            key: python_scalar(value)
            for key, value in row.to_dict().items()
        }

        try:
            response, parsed, latency, retry_count = call_with_retry(
                client=client,
                model=args.model,
                row=row,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                store_responses=args.store_responses,
            )

            api_requests_made += 1
            pred_dict = parsed.model_dump(mode="json")
            usage = getattr(response, "usage", None)

            records.append({
                **base_record,
                "status": "OK",
                "api_call_made": True,
                "store_responses": bool(args.store_responses),
                "model": args.model,
                "response_id": getattr(response, "id", None),
                "latency_seconds": latency,
                "retry_count": retry_count,
                "input_tokens": usage_value(usage, "input_tokens", "prompt_tokens"),
                "output_tokens": usage_value(usage, "output_tokens", "completion_tokens"),
                "prediction_json": json.dumps(pred_dict, ensure_ascii=False),
                "error": None,
            })

            append_checkpoint(checkpoint_path, records[-1])

        except Exception as exc:
            records.append({
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
            })

            append_checkpoint(checkpoint_path, records[-1])

    record_map = {str(record["case_id"]): record for record in records}
    selected_ids = benchmark["case_id"].astype(str).tolist()
    missing = [case_id for case_id in selected_ids if case_id not in record_map]
    if missing:
        raise RuntimeError(f"Missing results for cases: {missing[:10]}")

    raw_results = pd.DataFrame([record_map[case_id] for case_id in selected_ids])
    metrics = evaluate_all(raw_results)
    summary = summarize_metrics(raw_results, metrics)
    by_category = summarize_by_category(metrics)
    errors = error_rows(raw_results, metrics)

    predicted_weights = metrics[[
        "case_id",
        "category",
        "gold_controller_weights_json",
        "pred_weights_json",
        "pred_bounds_ok",
        "weight_l1_distance_to_gold",
    ]].copy()

    raw_results.to_csv(args.output_dir / "e3b_api_results.csv", index=False, encoding="utf-8-sig")
    try_write_parquet(raw_results, args.output_dir / "e3b_api_results.parquet")

    metrics.to_csv(args.output_dir / "e3b_case_metrics.csv", index=False, encoding="utf-8-sig")
    predicted_weights.to_csv(args.output_dir / "e3b_predicted_weights.csv", index=False, encoding="utf-8-sig")

    summary.to_csv(args.report_dir / "e3b_summary_metrics.csv", index=False)
    by_category.to_csv(args.report_dir / "e3b_metrics_by_category.csv", index=False)
    errors.to_csv(args.report_dir / "e3b_error_rows.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "model": args.model,
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_cases": int(total),
        "n_completed_api_responses_this_run": int(api_requests_made),
        "n_checkpointed_requests_skipped_this_run": int(checkpointed_requests),
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "benchmark_sha256": sha256_file(args.benchmark),
        "store_responses": bool(args.store_responses),
        "controller_min_weight": MIN_WEIGHT,
        "controller_max_weight": MAX_WEIGHT,
        "controller_delta_by_strength": DELTA_BY_STRENGTH,
        "system_prompt": SYSTEM_PROMPT,
        "schema": AdvisoryReweightingOutput.model_json_schema(),
    }

    (args.report_dir / "e3b_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(summary, by_category, args.report_dir / "e3b_report.md")

    print()
    print(f"Completed API responses this run: {api_requests_made}")
    print(f"Checkpointed cases skipped this run: {checkpointed_requests}")
    if api_requests_made == 0:
        print("WARNING: no new API responses were created in this run.")
    if not args.store_responses:
        print("Note: --store-responses was not enabled, so responses may not appear in stored response logs/history.")
    print()
    print("E3b complete.")
    print("Review first:")
    print("  e3b_report.md")
    print("  e3b_summary_metrics.csv")
    print("  e3b_metrics_by_category.csv")
    print("  e3b_error_rows.csv")
    print("  e3b_predicted_weights.csv")


if __name__ == "__main__":
    main()
