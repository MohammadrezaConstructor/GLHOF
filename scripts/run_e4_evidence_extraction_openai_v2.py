#!/usr/bin/env python3
"""
E4: OpenAI API supplier-evidence extraction and grounding benchmark.

Purpose
-------
Evaluate M_E, the evidence-extraction LLM component.

The model receives one supplier evidence packet and must extract only claims
that are explicitly supported by the provided text. It must ground every claim
in a short source quote copied from the packet.

It must not:
- rank suppliers;
- calculate scores;
- infer missing evidence;
- turn absence of evidence into a negative finding unless the packet explicitly
  states an absence;
- extract unsupported claims.

Input CSV columns
-----------------
packet_id
supplier_name
category
evidence_packet
gold_claims_json

gold_claims_json is a JSON list. Each gold claim should contain:
[
  {
    "claim_id": "P001_C01",
    "aspect": "CERTIFICATION",
    "polarity": "POSITIVE",
    "temporal_status": "CURRENT",
    "evidence_role": "FEATURE",
    "source_quote": "holds ISO 14001 certification valid until 2026",
    "normalized_claim": "Supplier holds ISO 14001 certification valid until 2026."
  }
]

Allowed aspects:
- CERTIFICATION
- SUSTAINABILITY
- SOCIAL
- GOVERNANCE_COMPLIANCE
- TECHNICAL_CAPABILITY
- DELIVERY_CAPACITY
- PAST_PERFORMANCE
- FINANCIAL_STABILITY
- OTHER

Allowed polarity:
- POSITIVE
- NEGATIVE
- NEUTRAL
- MIXED
- UNKNOWN

Allowed temporal_status:
- CURRENT
- HISTORICAL
- FUTURE
- EXPIRED
- UNKNOWN

Allowed evidence_role:
- ELIGIBILITY
- FEATURE
- SCENARIO
- ADVISORY

Outputs
-------
Processed:
- e4_api_results.csv
- e4_api_results.parquet if pyarrow/fastparquet is available
- e4_claim_level_matches.csv
- e4_packet_metrics.csv

Reports:
- e4_summary_metrics.csv
- e4_metrics_by_category.csv
- e4_metrics_by_aspect.csv
- e4_error_rows.csv
- e4_run_metadata.json
- e4_report.md

Requirements
------------
pip install -U openai pydantic pandas pyarrow

Example
-------
python scripts/run_e4_evidence_extraction_openai.py `
  --benchmark data/e4/e4_benchmark.csv `
  --model gpt-5.6-terra `
  --reasoning-effort low `
  --output-dir data/analysis/e4_final `
  --report-dir reports/e4_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator


Aspect = Literal[
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

Polarity = Literal[
    "POSITIVE",
    "NEGATIVE",
    "NEUTRAL",
    "MIXED",
    "UNKNOWN",
]

TemporalStatus = Literal[
    "CURRENT",
    "HISTORICAL",
    "FUTURE",
    "EXPIRED",
    "UNKNOWN",
]

EvidenceRole = Literal[
    "ELIGIBILITY",
    "FEATURE",
    "SCENARIO",
    "ADVISORY",
]

Uncertainty = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedClaim(StrictModel):
    claim_text: str = Field(
        min_length=1,
        max_length=600,
        description="Concise atomic claim grounded only in the evidence packet.",
    )
    aspect: Aspect
    polarity: Polarity
    temporal_status: TemporalStatus
    evidence_role: EvidenceRole
    source_quote: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "A short exact quote copied from the evidence packet that supports the claim."
        ),
    )
    uncertainty: Uncertainty


class EvidenceExtraction(StrictModel):
    supplier_name: str
    claims: list[ExtractedClaim] = Field(
        description="Only explicitly supported atomic claims."
    )

    @model_validator(mode="after")
    def validate_output(self):
        seen = set()
        deduped = []

        for claim in self.claims:
            key = (
                claim.claim_text.strip().lower(),
                claim.source_quote.strip().lower(),
                claim.aspect,
                claim.evidence_role,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(claim)

        self.claims = deduped
        return self


SYSTEM_PROMPT = """
You are the evidence-extraction component M_E in a governed public-procurement
supplier-selection framework.

Your task is to extract structured, source-grounded supplier evidence from the
provided evidence packet.

You MUST:
- extract only atomic claims explicitly supported by the packet;
- include a short source_quote copied from the packet for every claim;
- keep source_quote short but sufficient to support the claim;
- preserve uncertainty when the packet is incomplete or qualified;
- classify each claim using the controlled schema.

You MUST NOT:
- rank suppliers;
- calculate scores or weights;
- infer missing facts;
- use outside knowledge;
- treat absence of information as negative evidence unless the packet explicitly
  states the absence;
- invent certifications, dates, quantities, or contract experience;
- extract generic marketing language unless it states a concrete evidence claim.

Evidence roles:
- ELIGIBILITY: mandatory qualification, exclusion, license, certification, or
  compliance evidence needed to pass a hard requirement.
- FEATURE: structured or qualitative feature that can inform supplier comparison,
  such as past performance, capacity, certification, or financial evidence.
- SCENARIO: evidence especially relevant to a stated scenario priority, such as
  sustainability for a sustainability-focused scenario or delivery capacity for
  urgent delivery.
- ADVISORY: contextual information that may inform human deliberation but should
  not be treated as a direct score.

Aspect guidance:
- CERTIFICATION: named certifications, accreditations, licenses, standards.
- SUSTAINABILITY: environmental performance, emissions, energy, waste, circularity.
- SOCIAL: labor, inclusion, health and safety, workforce training, SME/social value.
- GOVERNANCE_COMPLIANCE: legal/regulatory compliance, audits, sanctions, controls.
- TECHNICAL_CAPABILITY: technical systems, specialist expertise, equipment, methods.
- DELIVERY_CAPACITY: capacity, staffing, coverage, logistics, ability to deliver.
- PAST_PERFORMANCE: completed contracts, references, delivery history.
- FINANCIAL_STABILITY: revenue, solvency, credit, audited accounts, going concern.
- OTHER: supported claim that does not fit the above.

Temporal guidance:
- CURRENT: valid now or stated as current/ongoing.
- HISTORICAL: past experience or past performance.
- FUTURE: planned or committed future action.
- EXPIRED: expired or no longer valid.
- UNKNOWN: no clear time status.

If the packet contains no extractable supported claims, return an empty claims
list. Do not fill the schema just to produce output.
""".strip()


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_match(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def token_set(text: str) -> set[str]:
    return {
        tok
        for tok in normalize_for_match(text).split()
        if tok
    }


def token_jaccard(a: str, b: str) -> float:
    aa = token_set(a)
    bb = token_set(b)

    if not aa and not bb:
        return 1.0

    if not aa or not bb:
        return 0.0

    return len(aa & bb) / len(aa | bb)


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        normalize_for_match(a),
        normalize_for_match(b),
    ).ratio()


def quote_in_packet(quote: str, packet: str) -> bool:
    q = normalize_for_match(quote)
    p = normalize_for_match(packet)

    if not q:
        return False

    return q in p


def parse_json_list(value, label: str = "json") -> list:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(
            f"{label} must be a JSON list."
        )

    return parsed


def safe_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
    )


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


def usage_value(usage, *names):
    if usage is None:
        return None

    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)

    return None


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

            records[str(record["packet_id"])] = record

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
# Benchmark validation
# ---------------------------------------------------------------------

ALLOWED_ASPECTS = {
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

ALLOWED_POLARITIES = {
    "POSITIVE",
    "NEGATIVE",
    "NEUTRAL",
    "MIXED",
    "UNKNOWN",
}

ALLOWED_TEMPORAL = {
    "CURRENT",
    "HISTORICAL",
    "FUTURE",
    "EXPIRED",
    "UNKNOWN",
}

ALLOWED_ROLES = {
    "ELIGIBILITY",
    "FEATURE",
    "SCENARIO",
    "ADVISORY",
}


def validate_gold_claim(packet_id: str, claim: dict, packet: str) -> None:
    required = {
        "claim_id",
        "aspect",
        "polarity",
        "temporal_status",
        "evidence_role",
        "source_quote",
        "normalized_claim",
    }

    missing = required - set(claim)

    if missing:
        raise ValueError(
            f"{packet_id}: gold claim missing fields {sorted(missing)}"
        )

    if claim["aspect"] not in ALLOWED_ASPECTS:
        raise ValueError(
            f"{packet_id}: invalid aspect {claim['aspect']!r}"
        )

    if claim["polarity"] not in ALLOWED_POLARITIES:
        raise ValueError(
            f"{packet_id}: invalid polarity {claim['polarity']!r}"
        )

    if claim["temporal_status"] not in ALLOWED_TEMPORAL:
        raise ValueError(
            f"{packet_id}: invalid temporal_status {claim['temporal_status']!r}"
        )

    if claim["evidence_role"] not in ALLOWED_ROLES:
        raise ValueError(
            f"{packet_id}: invalid evidence_role {claim['evidence_role']!r}"
        )

    if not quote_in_packet(claim["source_quote"], packet):
        raise ValueError(
            f"{packet_id}: source_quote not found in evidence_packet for "
            f"claim {claim['claim_id']}: {claim['source_quote']!r}"
        )


def validate_benchmark(df: pd.DataFrame) -> None:
    required_cols = {
        "packet_id",
        "supplier_name",
        "category",
        "evidence_packet",
        "gold_claims_json",
    }

    missing = sorted(required_cols - set(df.columns))

    if missing:
        raise ValueError(
            "Benchmark missing columns:\n  - " + "\n  - ".join(missing)
        )

    if df.empty:
        raise ValueError("Benchmark is empty.")

    if df["packet_id"].astype(str).duplicated().any():
        duplicate_ids = (
            df.loc[df["packet_id"].astype(str).duplicated(keep=False), "packet_id"]
            .astype(str)
            .drop_duplicates()
            .head(20)
            .tolist()
        )
        raise ValueError(f"Duplicate packet_id values: {duplicate_ids}")

    for row_idx, row in df.iterrows():
        packet_id = str(row["packet_id"])
        packet = str(row["evidence_packet"])

        if not packet.strip():
            raise ValueError(f"{packet_id}: empty evidence_packet.")

        claims = parse_json_list(row["gold_claims_json"], f"{packet_id}.gold_claims_json")

        claim_ids = []

        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError(
                    f"{packet_id}: every gold claim must be an object."
                )

            validate_gold_claim(packet_id, claim, packet)
            claim_ids.append(str(claim["claim_id"]))

        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(
                f"{packet_id}: duplicate gold claim_id values."
            )


# ---------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------

def build_user_input(supplier_name: str, evidence_packet: str) -> str:
    return (
        f"Supplier name: {supplier_name}\n\n"
        "Evidence packet:\n"
        f"{evidence_packet}\n\n"
        "Extract only supported supplier evidence claims from this packet."
    )


def call_with_retry(
    client: OpenAI,
    *,
    model: str,
    supplier_name: str,
    evidence_packet: str,
    max_retries: int,
    retry_base_seconds: float,
    reasoning_effort: str,
    max_output_tokens: int,
    store_responses: bool,
):
    last_error = None
    user_input = build_user_input(supplier_name, evidence_packet)

    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()

            response = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                text_format=EvidenceExtraction,
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


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def claim_similarity(gold: dict, pred: dict) -> float:
    gold_quote = str(gold.get("source_quote", ""))
    pred_quote = str(pred.get("source_quote", ""))
    gold_claim = str(gold.get("normalized_claim", ""))
    pred_claim = str(pred.get("claim_text", ""))

    quote_score = max(
        token_jaccard(gold_quote, pred_quote),
        sequence_ratio(gold_quote, pred_quote),
    )

    claim_score = max(
        token_jaccard(gold_claim, pred_claim),
        sequence_ratio(gold_claim, pred_claim),
    )

    aspect_bonus = 0.05 if gold.get("aspect") == pred.get("aspect") else 0.0

    return min(
        1.0,
        0.65 * quote_score + 0.30 * claim_score + aspect_bonus,
    )


def greedy_match_claims(
    gold_claims: list[dict],
    pred_claims: list[dict],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    pairs = []

    for gi, gold in enumerate(gold_claims):
        for pi, pred in enumerate(pred_claims):
            sim = claim_similarity(gold, pred)
            if sim >= threshold:
                pairs.append((gi, pi, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)

    used_gold = set()
    used_pred = set()
    matches = []

    for gi, pi, sim in pairs:
        if gi in used_gold or pi in used_pred:
            continue

        used_gold.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, sim))

    unmatched_gold = set(range(len(gold_claims))) - used_gold
    unmatched_pred = set(range(len(pred_claims))) - used_pred

    return matches, unmatched_gold, unmatched_pred


def evaluate_packet(
    row: pd.Series,
    prediction_json: str | None,
    match_threshold: float,
) -> tuple[dict, list[dict]]:
    packet_id = str(row["packet_id"])
    supplier_name = str(row["supplier_name"])
    packet = str(row["evidence_packet"])
    category = str(row["category"])

    gold_claims = parse_json_list(row["gold_claims_json"], f"{packet_id}.gold_claims_json")

    if prediction_json is None or pd.isna(prediction_json):
        pred_claims = []
        pred_supplier = None
    else:
        pred_obj = json.loads(prediction_json)
        pred_supplier = pred_obj.get("supplier_name")
        pred_claims = pred_obj.get("claims", [])

    # Ensure all predicted source quotes can be grounded in the packet.
    pred_grounded = [
        quote_in_packet(pred.get("source_quote", ""), packet)
        for pred in pred_claims
    ]

    matches, unmatched_gold, unmatched_pred = greedy_match_claims(
        gold_claims,
        pred_claims,
        match_threshold,
    )

    n_gold = len(gold_claims)
    n_pred = len(pred_claims)
    n_match = len(matches)

    precision = n_match / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    recall = n_match / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)

    claim_f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    grounded_rate = (
        sum(pred_grounded) / n_pred
        if n_pred
        else 1.0
    )

    # Hallucination is reserved for unsupported predictions: the model provides
    # a source_quote that cannot be grounded in the packet. A grounded but
    # unmatched prediction is instead counted as over-extraction because it may
    # reflect claim-granularity differences rather than fabrication.
    unsupported_pred_count = sum(1 for grounded in pred_grounded if not grounded)
    over_extraction_count = len(unmatched_pred)
    missed_gold_count = len(unmatched_gold)

    hallucination_count = unsupported_pred_count

    hallucination_rate = (
        unsupported_pred_count / n_pred
        if n_pred
        else 0.0
    )

    over_extraction_rate = (
        over_extraction_count / n_pred
        if n_pred
        else 0.0
    )

    missed_gold_rate = (
        missed_gold_count / n_gold
        if n_gold
        else 0.0
    )

    attr_totals = {
        "aspect": [],
        "polarity": [],
        "temporal_status": [],
        "evidence_role": [],
    }

    match_rows = []

    for gi, pi, sim in matches:
        gold = gold_claims[gi]
        pred = pred_claims[pi]

        row_out = {
            "packet_id": packet_id,
            "category": category,
            "supplier_name": supplier_name,
            "gold_claim_id": gold.get("claim_id"),
            "match_status": "MATCHED",
            "match_similarity": sim,
            "gold_claim": gold.get("normalized_claim"),
            "pred_claim": pred.get("claim_text"),
            "gold_source_quote": gold.get("source_quote"),
            "pred_source_quote": pred.get("source_quote"),
            "pred_source_grounded": quote_in_packet(pred.get("source_quote", ""), packet),
        }

        for attr in attr_totals:
            correct = gold.get(attr) == pred.get(attr)
            attr_totals[attr].append(correct)
            row_out[f"gold_{attr}"] = gold.get(attr)
            row_out[f"pred_{attr}"] = pred.get(attr)
            row_out[f"{attr}_correct"] = correct

        match_rows.append(row_out)

    for gi in sorted(unmatched_gold):
        gold = gold_claims[gi]
        match_rows.append({
            "packet_id": packet_id,
            "category": category,
            "supplier_name": supplier_name,
            "gold_claim_id": gold.get("claim_id"),
            "match_status": "MISSED_GOLD",
            "match_similarity": 0.0,
            "gold_claim": gold.get("normalized_claim"),
            "pred_claim": None,
            "gold_source_quote": gold.get("source_quote"),
            "pred_source_quote": None,
            "pred_source_grounded": None,
            "gold_aspect": gold.get("aspect"),
            "pred_aspect": None,
            "aspect_correct": False,
            "gold_polarity": gold.get("polarity"),
            "pred_polarity": None,
            "polarity_correct": False,
            "gold_temporal_status": gold.get("temporal_status"),
            "pred_temporal_status": None,
            "temporal_status_correct": False,
            "gold_evidence_role": gold.get("evidence_role"),
            "pred_evidence_role": None,
            "evidence_role_correct": False,
        })

    for pi in sorted(unmatched_pred):
        pred = pred_claims[pi]
        match_rows.append({
            "packet_id": packet_id,
            "category": category,
            "supplier_name": supplier_name,
            "gold_claim_id": None,
            "match_status": "UNMATCHED_PREDICTION",
            "match_similarity": 0.0,
            "gold_claim": None,
            "pred_claim": pred.get("claim_text"),
            "gold_source_quote": None,
            "pred_source_quote": pred.get("source_quote"),
            "pred_source_grounded": quote_in_packet(pred.get("source_quote", ""), packet),
            "gold_aspect": None,
            "pred_aspect": pred.get("aspect"),
            "aspect_correct": False,
            "gold_polarity": None,
            "pred_polarity": pred.get("polarity"),
            "polarity_correct": False,
            "gold_temporal_status": None,
            "pred_temporal_status": pred.get("temporal_status"),
            "temporal_status_correct": False,
            "gold_evidence_role": None,
            "pred_evidence_role": pred.get("evidence_role"),
            "evidence_role_correct": False,
        })

    packet_metrics = {
        "packet_id": packet_id,
        "category": category,
        "supplier_name": supplier_name,
        "n_gold_claims": n_gold,
        "n_pred_claims": n_pred,
        "n_matched_claims": n_match,
        "claim_precision": precision,
        "claim_recall": recall,
        "claim_f1": claim_f1,
        "source_grounding_rate": grounded_rate,
        "hallucination_rate": hallucination_rate,
        "hallucination_count": hallucination_count,
        "unsupported_pred_count": unsupported_pred_count,
        "over_extraction_count": over_extraction_count,
        "over_extraction_rate": over_extraction_rate,
        "missed_gold_count": missed_gold_count,
        "missed_gold_rate": missed_gold_rate,
        "supplier_name_exact": normalize_for_match(pred_supplier or "") == normalize_for_match(supplier_name),
        "aspect_accuracy_matched": (
            sum(attr_totals["aspect"]) / len(attr_totals["aspect"])
            if attr_totals["aspect"]
            else None
        ),
        "polarity_accuracy_matched": (
            sum(attr_totals["polarity"]) / len(attr_totals["polarity"])
            if attr_totals["polarity"]
            else None
        ),
        "temporal_status_accuracy_matched": (
            sum(attr_totals["temporal_status"]) / len(attr_totals["temporal_status"])
            if attr_totals["temporal_status"]
            else None
        ),
        "evidence_role_accuracy_matched": (
            sum(attr_totals["evidence_role"]) / len(attr_totals["evidence_role"])
            if attr_totals["evidence_role"]
            else None
        ),
    }

    return packet_metrics, match_rows


def evaluate_all(
    results: pd.DataFrame,
    match_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    packet_rows = []
    claim_rows = []

    for _, row in results.iterrows():
        if row["status"] != "OK":
            # Evaluate as empty prediction.
            packet_metrics, matches = evaluate_packet(
                row,
                None,
                match_threshold,
            )
        else:
            packet_metrics, matches = evaluate_packet(
                row,
                row["prediction_json"],
                match_threshold,
            )

        packet_metrics["status"] = row["status"]
        packet_rows.append(packet_metrics)
        claim_rows.extend(matches)

    return (
        pd.DataFrame(packet_rows),
        pd.DataFrame(claim_rows),
    )


def summarize_metrics(
    results: pd.DataFrame,
    packet_metrics: pd.DataFrame,
    claim_matches: pd.DataFrame,
) -> pd.DataFrame:
    matched = claim_matches.loc[
        claim_matches["match_status"] == "MATCHED"
    ]

    rows = [
        ("API_SUCCESS_RATE", results["status"].eq("OK").mean()),
        ("MEAN_CLAIM_PRECISION", packet_metrics["claim_precision"].mean()),
        ("MEAN_CLAIM_RECALL", packet_metrics["claim_recall"].mean()),
        ("MEAN_CLAIM_F1", packet_metrics["claim_f1"].mean()),
        ("MEAN_SOURCE_GROUNDING_RATE", packet_metrics["source_grounding_rate"].mean()),
        ("MEAN_HALLUCINATION_RATE", packet_metrics["hallucination_rate"].mean()),
        ("MEAN_OVER_EXTRACTION_RATE", packet_metrics["over_extraction_rate"].mean()),
        ("MEAN_MISSED_GOLD_RATE", packet_metrics["missed_gold_rate"].mean()),
        ("SUPPLIER_NAME_EXACT_RATE", packet_metrics["supplier_name_exact"].mean()),
        (
            "ASPECT_ACCURACY_MATCHED",
            matched["aspect_correct"].mean() if len(matched) else float("nan"),
        ),
        (
            "POLARITY_ACCURACY_MATCHED",
            matched["polarity_correct"].mean() if len(matched) else float("nan"),
        ),
        (
            "TEMPORAL_STATUS_ACCURACY_MATCHED",
            matched["temporal_status_correct"].mean() if len(matched) else float("nan"),
        ),
        (
            "EVIDENCE_ROLE_ACCURACY_MATCHED",
            matched["evidence_role_correct"].mean() if len(matched) else float("nan"),
        ),
        (
            "N_GOLD_CLAIMS",
            packet_metrics["n_gold_claims"].sum(),
        ),
        (
            "N_PRED_CLAIMS",
            packet_metrics["n_pred_claims"].sum(),
        ),
        (
            "N_MATCHED_CLAIMS",
            packet_metrics["n_matched_claims"].sum(),
        ),
    ]

    return pd.DataFrame(rows, columns=["METRIC", "VALUE"])


def summarize_by_category(packet_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category, g in packet_metrics.groupby(
        "category",
        sort=False,
        observed=True,
    ):
        rows.append({
            "CATEGORY": category,
            "N_PACKETS": len(g),
            "MEAN_CLAIM_PRECISION": g["claim_precision"].mean(),
            "MEAN_CLAIM_RECALL": g["claim_recall"].mean(),
            "MEAN_CLAIM_F1": g["claim_f1"].mean(),
            "MEAN_SOURCE_GROUNDING_RATE": g["source_grounding_rate"].mean(),
            "MEAN_HALLUCINATION_RATE": g["hallucination_rate"].mean(),
            "MEAN_OVER_EXTRACTION_RATE": g["over_extraction_rate"].mean(),
            "MEAN_MISSED_GOLD_RATE": g["missed_gold_rate"].mean(),
            "N_GOLD_CLAIMS": g["n_gold_claims"].sum(),
            "N_PRED_CLAIMS": g["n_pred_claims"].sum(),
            "N_MATCHED_CLAIMS": g["n_matched_claims"].sum(),
        })

    return pd.DataFrame(rows)


def summarize_by_aspect(claim_matches: pd.DataFrame) -> pd.DataFrame:
    gold_rows = claim_matches.loc[
        claim_matches["gold_aspect"].notna()
    ]

    rows = []

    for aspect, g in gold_rows.groupby(
        "gold_aspect",
        sort=False,
        observed=True,
    ):
        n_gold = len(g)
        n_matched = int((g["match_status"] == "MATCHED").sum())

        matched = g.loc[g["match_status"] == "MATCHED"]

        rows.append({
            "ASPECT": aspect,
            "N_GOLD_CLAIMS": n_gold,
            "N_MATCHED_CLAIMS": n_matched,
            "RECALL": n_matched / n_gold if n_gold else 0.0,
            "ASPECT_ACCURACY_MATCHED": (
                matched["aspect_correct"].mean()
                if len(matched)
                else float("nan")
            ),
            "EVIDENCE_ROLE_ACCURACY_MATCHED": (
                matched["evidence_role_correct"].mean()
                if len(matched)
                else float("nan")
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "N_GOLD_CLAIMS",
        ascending=False,
    )


def error_rows(
    packet_metrics: pd.DataFrame,
    claim_matches: pd.DataFrame,
) -> pd.DataFrame:
    bad_packets = packet_metrics.loc[
        (packet_metrics["claim_f1"] < 1.0)
        | (packet_metrics["source_grounding_rate"] < 1.0)
        | (packet_metrics["hallucination_rate"] > 0.0)
        | (packet_metrics["over_extraction_rate"] > 0.0)
        | (packet_metrics["missed_gold_rate"] > 0.0)
        | (packet_metrics["aspect_accuracy_matched"].fillna(1.0) < 1.0)
        | (packet_metrics["polarity_accuracy_matched"].fillna(1.0) < 1.0)
        | (packet_metrics["temporal_status_accuracy_matched"].fillna(1.0) < 1.0)
        | (packet_metrics["evidence_role_accuracy_matched"].fillna(1.0) < 1.0)
    ]["packet_id"].tolist()

    return claim_matches.loc[
        claim_matches["packet_id"].isin(bad_packets)
    ].copy()


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame,
    by_category: pd.DataFrame,
    by_aspect: pd.DataFrame,
    output_path: Path,
) -> None:
    metric = dict(zip(summary["METRIC"], summary["VALUE"]))

    lines = [
        "# E4 Evidence Extraction and Grounding Evaluation",
        "",
        "## Overall results",
        "",
        f"- API success rate: {metric.get('API_SUCCESS_RATE', float('nan')):.3f}",
        f"- Mean claim precision: {metric.get('MEAN_CLAIM_PRECISION', float('nan')):.3f}",
        f"- Mean claim recall: {metric.get('MEAN_CLAIM_RECALL', float('nan')):.3f}",
        f"- Mean claim F1: {metric.get('MEAN_CLAIM_F1', float('nan')):.3f}",
        f"- Mean source-grounding rate: {metric.get('MEAN_SOURCE_GROUNDING_RATE', float('nan')):.3f}",
        f"- Mean hallucination rate, unsupported source quotes: {metric.get('MEAN_HALLUCINATION_RATE', float('nan')):.3f}",
        f"- Mean over-extraction rate, grounded but unmatched predictions: {metric.get('MEAN_OVER_EXTRACTION_RATE', float('nan')):.3f}",
        f"- Mean missed-gold rate: {metric.get('MEAN_MISSED_GOLD_RATE', float('nan')):.3f}",
        f"- Supplier-name exact rate: {metric.get('SUPPLIER_NAME_EXACT_RATE', float('nan')):.3f}",
        f"- Aspect accuracy on matched claims: {metric.get('ASPECT_ACCURACY_MATCHED', float('nan')):.3f}",
        f"- Polarity accuracy on matched claims: {metric.get('POLARITY_ACCURACY_MATCHED', float('nan')):.3f}",
        f"- Temporal-status accuracy on matched claims: {metric.get('TEMPORAL_STATUS_ACCURACY_MATCHED', float('nan')):.3f}",
        f"- Evidence-role accuracy on matched claims: {metric.get('EVIDENCE_ROLE_ACCURACY_MATCHED', float('nan')):.3f}",
        f"- Gold claims: {int(metric.get('N_GOLD_CLAIMS', 0))}",
        f"- Predicted claims: {int(metric.get('N_PRED_CLAIMS', 0))}",
        f"- Matched claims: {int(metric.get('N_MATCHED_CLAIMS', 0))}",
        "",
        "## Results by packet category",
        "",
        "| Category | Packets | Gold claims | Predicted claims | Matched claims | Claim F1 | Grounding | Hallucination | Over-extraction | Missed-gold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in by_category.iterrows():
        lines.append(
            f"| `{row['CATEGORY']}` "
            f"| {int(row['N_PACKETS'])} "
            f"| {int(row['N_GOLD_CLAIMS'])} "
            f"| {int(row['N_PRED_CLAIMS'])} "
            f"| {int(row['N_MATCHED_CLAIMS'])} "
            f"| {row['MEAN_CLAIM_F1']:.3f} "
            f"| {row['MEAN_SOURCE_GROUNDING_RATE']:.3f} "
            f"| {row['MEAN_HALLUCINATION_RATE']:.3f} "
            f"| {row['MEAN_OVER_EXTRACTION_RATE']:.3f} "
            f"| {row['MEAN_MISSED_GOLD_RATE']:.3f} |"
        )

    lines.extend([
        "",
        "## Results by gold aspect",
        "",
        "| Aspect | Gold claims | Matched claims | Recall | Aspect accuracy | Role accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for _, row in by_aspect.iterrows():
        lines.append(
            f"| `{row['ASPECT']}` "
            f"| {int(row['N_GOLD_CLAIMS'])} "
            f"| {int(row['N_MATCHED_CLAIMS'])} "
            f"| {row['RECALL']:.3f} "
            f"| {row['ASPECT_ACCURACY_MATCHED']:.3f} "
            f"| {row['EVIDENCE_ROLE_ACCURACY_MATCHED']:.3f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "E4 evaluates evidence extraction and grounding, not ranking. A claim is "
        "credited only when it matches a gold atomic claim sufficiently closely; "
        "attribute accuracy is computed over matched claims. Source grounding "
        "checks whether each predicted source_quote appears in the provided packet. "
        "Hallucination is defined narrowly as an unsupported source quote. Grounded "
        "but unmatched predictions are reported separately as over-extraction.",
        "",
        "Because claim matching uses fuzzy overlap, the error rows should be manually "
        "reviewed before reporting final qualitative conclusions.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/e4_openai"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/e4_openai"),
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="low",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.72,
        help="Greedy gold/predicted claim matching threshold.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pilot limit, e.g. --limit 5",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the JSONL checkpoint in the output directory.",
    )
    parser.add_argument(
        "--store-responses",
        action="store_true",
        help="Store Responses API objects server-side. Keep disabled for final runs unless intentional.",
    )

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

    checkpoint_path = args.output_dir / "e4_checkpoint.jsonl"

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

    for i, (_, row) in enumerate(benchmark.iterrows(), start=1):
        packet_id = str(row["packet_id"])

        if packet_id in completed:
            checkpointed_requests += 1
            print(f"[{i}/{total}] {packet_id} (checkpointed; no API call made)")
            continue

        print(f"[{i}/{total}] {packet_id}")

        base_record = {
            key: python_scalar(value)
            for key, value in row.to_dict().items()
        }

        try:
            response, parsed, latency, retry_count = call_with_retry(
                client=client,
                model=args.model,
                supplier_name=str(row["supplier_name"]),
                evidence_packet=str(row["evidence_packet"]),
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

    record_map = {
        str(record["packet_id"]): record
        for record in records
    }

    selected_ids = benchmark["packet_id"].astype(str).tolist()

    missing_results = [
        packet_id
        for packet_id in selected_ids
        if packet_id not in record_map
    ]

    if missing_results:
        raise RuntimeError(
            f"Missing results for {len(missing_results)} packets: "
            f"{missing_results[:10]}"
        )

    raw_results = pd.DataFrame(
        [record_map[packet_id] for packet_id in selected_ids]
    )

    packet_metrics, claim_matches = evaluate_all(
        raw_results,
        args.match_threshold,
    )

    summary = summarize_metrics(
        raw_results,
        packet_metrics,
        claim_matches,
    )

    by_category = summarize_by_category(
        packet_metrics,
    )

    by_aspect = summarize_by_aspect(
        claim_matches,
    )

    errors = error_rows(
        packet_metrics,
        claim_matches,
    )

    # Outputs
    raw_results.to_csv(
        args.output_dir / "e4_api_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    try_write_parquet(
        raw_results,
        args.output_dir / "e4_api_results.parquet",
    )

    packet_metrics.to_csv(
        args.output_dir / "e4_packet_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    claim_matches.to_csv(
        args.output_dir / "e4_claim_level_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        args.report_dir / "e4_summary_metrics.csv",
        index=False,
    )

    by_category.to_csv(
        args.report_dir / "e4_metrics_by_category.csv",
        index=False,
    )

    by_aspect.to_csv(
        args.report_dir / "e4_metrics_by_aspect.csv",
        index=False,
    )

    errors.to_csv(
        args.report_dir / "e4_error_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_metadata = {
        "model": args.model,
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_packets": int(total),
        "n_completed_api_responses_this_run": int(api_requests_made),
        "n_checkpointed_requests_skipped_this_run": int(checkpointed_requests),
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "match_threshold": args.match_threshold,
        "benchmark_sha256": sha256_file(args.benchmark),
        "store_responses": bool(args.store_responses),
        "system_prompt": SYSTEM_PROMPT,
        "schema": EvidenceExtraction.model_json_schema(),
    }

    (args.report_dir / "e4_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(
        summary,
        by_category,
        by_aspect,
        args.report_dir / "e4_report.md",
    )

    print()
    print(f"Completed API responses this run: {api_requests_made}")
    print(f"Checkpointed packets skipped this run: {checkpointed_requests}")
    if api_requests_made == 0:
        print("WARNING: no new API responses were created in this run.")
    if not args.store_responses:
        print("Note: --store-responses was not enabled, so responses may not appear in stored response logs/history.")
    print()
    print("E4 complete.")
    print("Review first:")
    print("  e4_report.md")
    print("  e4_summary_metrics.csv")
    print("  e4_metrics_by_category.csv")
    print("  e4_metrics_by_aspect.csv")
    print("  e4_error_rows.csv")


if __name__ == "__main__":
    main()
