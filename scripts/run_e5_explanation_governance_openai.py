
#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, random, re, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

ScenarioProfile = Literal["BALANCED", "CONTINUITY", "DIVERSIFICATION", "CUSTOM"]
PriorityGroup = Literal["CATEGORY_FIT", "GEOGRAPHIC_FIT", "BUYER_RELATIONSHIP", "ACTIVITY_PROFILE"]
ReasonType = Literal["STRUCTURED_FEATURE", "EVIDENCE_CLAIM", "MIXED"]
GovernanceFlag = Literal["NONE", "LOW_SCORE_MARGIN", "MISSING_EVIDENCE", "NEGATIVE_EVIDENCE", "CONFLICTING_EVIDENCE", "CONSTRAINT_SENSITIVE", "HUMAN_REVIEW_REQUIRED"]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class SupplierExplanation(StrictModel):
    supplier_id: str
    position: int = Field(ge=1, le=10)
    reason_type: ReasonType
    rationale: str = Field(min_length=1, max_length=1200)
    feature_groups_used: list[PriorityGroup]
    cited_claim_ids: list[str]

class Tradeoff(StrictModel):
    supplier_id: str
    tradeoff_text: str = Field(min_length=1, max_length=1000)
    cited_claim_ids: list[str]

class Caveat(StrictModel):
    caveat_text: str = Field(min_length=1, max_length=1000)
    cited_claim_ids: list[str]

class ExplanationOutput(StrictModel):
    case_id: str
    scenario_profile: ScenarioProfile
    supplier_order: list[str]
    overall_summary: str = Field(min_length=1, max_length=1400)
    supplier_explanations: list[SupplierExplanation]
    tradeoffs: list[Tradeoff]
    caveats: list[Caveat]
    governance_flags: list[GovernanceFlag]
    human_review_required: bool
    audit_note: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def clean_lists(self):
        seen = set(); order = []
        for sid in self.supplier_order:
            if sid not in seen:
                seen.add(sid); order.append(sid)
        self.supplier_order = order
        if "NONE" in self.governance_flags and len(self.governance_flags) > 1:
            self.governance_flags = [f for f in self.governance_flags if f != "NONE"]
        seen = set(); flags = []
        for f in self.governance_flags:
            if f not in seen:
                seen.add(f); flags.append(f)
        self.governance_flags = flags or ["NONE"]
        return self

SYSTEM_PROMPT = """
You are the explanation and governance-audit component M_X in a governed public-procurement supplier-selection framework.

The deterministic analytics has already produced the ranking. Your task is to explain it, not to change it.

You MUST:
- preserve the deterministic supplier order exactly;
- use only supplier IDs provided in supplier_context;
- cite only claim IDs provided in evidence_claims;
- explain how the scenario, weights, structured scores, and evidence support the ranking;
- identify tradeoffs, caveats, and governance flags.

You MUST NOT:
- rerank suppliers;
- change weights;
- calculate new scores;
- invent evidence;
- cite missing claim IDs;
- call a supplier "optimal", "winner", "objectively best", or "guaranteed";
- imply that a reference outcome proves optimality.

Use cautious language such as "ranked first by the deterministic model", "supports the ranking", and "requires human review".

Governance flags:
- LOW_SCORE_MARGIN: top suppliers are very close or the input says margin is close.
- MISSING_EVIDENCE: relevant evidence is absent or explicitly missing.
- NEGATIVE_EVIDENCE: adverse findings, late delivery, compliance concerns, financial risk, or similar.
- CONFLICTING_EVIDENCE: positive and negative evidence point in different directions.
- CONSTRAINT_SENSITIVE: hard constraints are mentioned and should not be treated as soft preferences.
- HUMAN_REVIEW_REQUIRED: material governance issue should be reviewed.
- NONE: only if no material governance issue is present.
""".strip()

ALLOWED_GROUPS = {"CATEGORY_FIT", "GEOGRAPHIC_FIT", "BUYER_RELATIONSHIP", "ACTIVITY_PROFILE"}
ALLOWED_FLAGS = {"NONE", "LOW_SCORE_MARGIN", "MISSING_EVIDENCE", "NEGATIVE_EVIDENCE", "CONFLICTING_EVIDENCE", "CONSTRAINT_SENSITIVE", "HUMAN_REVIEW_REQUIRED"}
REQ_COLS = ["case_id","category","user_request","scenario_profile","priority_groups_json","group_weights_json","supplier_context_json","evidence_claims_json","gold_top_supplier_order_json","gold_required_priority_groups_json","gold_required_governance_flags_json","gold_requires_human_review","gold_forbidden_terms_json"]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()

def parse_list(x, label):
    v = json.loads(str(x))
    if not isinstance(v, list): raise ValueError(f"{label} must be a list")
    return v

def parse_dict(x, label):
    v = json.loads(str(x))
    if not isinstance(v, dict): raise ValueError(f"{label} must be an object")
    return v

def f1(gold, pred):
    gold, pred = set(gold), set(pred)
    if not gold and not pred: return 1.0
    if not gold or not pred: return 0.0
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 0.0 if p + r == 0 else 2*p*r/(p+r)

def clean_flags(flags):
    out = set(str(x) for x in flags)
    if "NONE" in out and len(out) > 1: out.remove("NONE")
    return out or {"NONE"}

def as_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def py_scalar(v):
    if pd.isna(v): return None
    if hasattr(v, "item"):
        try: return v.item()
        except Exception: pass
    return v

def usage_value(usage, *names):
    if usage is None: return None
    for name in names:
        val = getattr(usage, name, None)
        if val is not None: return int(val)
    return None

def retryable(exc):
    name = type(exc).__name__
    if name in {"APIConnectionError","APITimeoutError","RateLimitError","InternalServerError"}: return True
    code = getattr(exc, "status_code", None)
    if code is not None:
        try: return int(code) >= 500 or int(code) == 429
        except Exception: return False
    return name in {"ValidationError", "JSONDecodeError", "APIResponseValidationError"}

def validate_benchmark(df):
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing: raise ValueError("Missing columns: " + ", ".join(missing))
    if df.empty: raise ValueError("Benchmark is empty")
    if df["case_id"].astype(str).duplicated().any(): raise ValueError("Duplicate case_id values")
    for _, row in df.iterrows():
        cid = str(row["case_id"])
        groups = parse_list(row["priority_groups_json"], cid+" priority_groups")
        if not set(groups).issubset(ALLOWED_GROUPS): raise ValueError(f"{cid}: invalid priority groups")
        weights = parse_dict(row["group_weights_json"], cid+" weights")
        if not set(weights).issubset(ALLOWED_GROUPS): raise ValueError(f"{cid}: invalid weight groups")
        if abs(sum(float(v) for v in weights.values()) - 1.0) > 1e-6: raise ValueError(f"{cid}: weights do not sum to one")
        suppliers = parse_list(row["supplier_context_json"], cid+" suppliers")
        sids = [str(s["supplier_id"]) for s in suppliers]
        if len(sids) != len(set(sids)): raise ValueError(f"{cid}: duplicate supplier IDs")
        gold_order = parse_list(row["gold_top_supplier_order_json"], cid+" gold_order")
        if not set(gold_order).issubset(set(sids)): raise ValueError(f"{cid}: gold order contains unknown suppliers")
        claims = parse_list(row["evidence_claims_json"], cid+" evidence_claims")
        claim_ids = [str(c["claim_id"]) for c in claims]
        if len(claim_ids) != len(set(claim_ids)): raise ValueError(f"{cid}: duplicate claim IDs")
        flags = parse_list(row["gold_required_governance_flags_json"], cid+" flags")
        if not set(flags).issubset(ALLOWED_FLAGS): raise ValueError(f"{cid}: invalid governance flags")
        parse_list(row["gold_forbidden_terms_json"], cid+" forbidden")

def checkpoint_load(path):
    if not path.exists(): return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[str(rec["case_id"])] = rec
    return out

def checkpoint_append(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        f.flush(); os.fsync(f.fileno())

def user_payload(row):
    payload = {
        "case_id": str(row["case_id"]),
        "user_request": str(row["user_request"]),
        "scenario_profile": str(row["scenario_profile"]),
        "priority_groups": parse_list(row["priority_groups_json"], "priority_groups"),
        "group_weights": parse_dict(row["group_weights_json"], "weights"),
        "supplier_context": parse_list(row["supplier_context_json"], "suppliers"),
        "evidence_claims": parse_list(row["evidence_claims_json"], "claims"),
        "instruction": "Explain the deterministic ranking while preserving supplier order. Use only provided supplier IDs and claim IDs."
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)

def call_api(client, args, row):
    last = None
    for attempt in range(args.max_retries + 1):
        try:
            start = time.perf_counter()
            resp = client.responses.parse(
                model=args.model,
                input=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_payload(row)}],
                text_format=ExplanationOutput,
                reasoning={"effort": args.reasoning_effort},
                max_output_tokens=args.max_output_tokens,
                store=args.store_responses,
            )
            latency = time.perf_counter() - start
            if resp.output_parsed is None: raise RuntimeError("No parsed output")
            return resp, resp.output_parsed, latency, attempt
        except Exception as exc:
            last = exc
            if attempt >= args.max_retries or not retryable(exc): break
            time.sleep(args.retry_base_seconds * (2 ** attempt) + random.uniform(0, 0.5))
    raise RuntimeError(f"Request failed after {args.max_retries + 1} attempts: {type(last).__name__}: {last}") from last

def collect_text(pred):
    parts = [pred.get("overall_summary", ""), pred.get("audit_note", "")]
    parts += [x.get("rationale", "") for x in pred.get("supplier_explanations", [])]
    parts += [x.get("tradeoff_text", "") for x in pred.get("tradeoffs", [])]
    parts += [x.get("caveat_text", "") for x in pred.get("caveats", [])]
    return "\n".join(str(x) for x in parts)

def cited_ids(pred):
    out = []
    for key, field in [("supplier_explanations","cited_claim_ids"),("tradeoffs","cited_claim_ids"),("caveats","cited_claim_ids")]:
        for obj in pred.get(key, []): out.extend(str(x) for x in obj.get(field, []))
    return out

def evaluate_row(row):
    cid = str(row["case_id"])
    suppliers = parse_list(row["supplier_context_json"], cid+" suppliers")
    supplier_ids = [str(s["supplier_id"]) for s in suppliers]
    supplier_set = set(supplier_ids)
    claims = parse_list(row["evidence_claims_json"], cid+" claims")
    valid_claims = {str(c["claim_id"]) for c in claims}
    gold_order = [str(x) for x in parse_list(row["gold_top_supplier_order_json"], cid+" gold order")]
    gold_groups = set(str(x) for x in parse_list(row["gold_required_priority_groups_json"], cid+" groups"))
    gold_flags = clean_flags(parse_list(row["gold_required_governance_flags_json"], cid+" flags"))
    gold_review = as_bool(row["gold_requires_human_review"])
    forbidden = [str(x) for x in parse_list(row["gold_forbidden_terms_json"], cid+" forbidden")]
    base = {"case_id": cid, "category": str(row["category"]), "status": str(row["status"])}
    if row["status"] != "OK" or not row.get("prediction_json"):
        return {**base, "order_exact": False, "top1_correct": False, "supplier_set_valid": False, "invalid_supplier_count": None, "citation_valid_rate": 0.0, "unsupported_citation_rate": 1.0, "priority_group_f1": 0.0 if gold_groups else 1.0, "governance_flag_f1": 0.0 if gold_flags != {"NONE"} else 1.0, "human_review_correct": False, "forbidden_terms_absent": False, "forbidden_terms_found_json": "[]", "pred_governance_flags_json": "[]", "pred_priority_groups_json": "[]", "pred_supplier_order_json": "[]"}
    pred = json.loads(row["prediction_json"])
    pred_order = [str(x) for x in pred.get("supplier_order", [])]
    invalid_suppliers = [x for x in pred_order if x not in supplier_set]
    for exp in pred.get("supplier_explanations", []):
        sid = str(exp.get("supplier_id", ""))
        if sid not in supplier_set: invalid_suppliers.append(sid)
    pred_groups = set()
    for exp in pred.get("supplier_explanations", []): pred_groups.update(str(x) for x in exp.get("feature_groups_used", []))
    pred_groups = pred_groups & ALLOWED_GROUPS
    cites = cited_ids(pred)
    invalid_cites = [x for x in cites if x not in valid_claims]
    citation_valid = 1.0 - len(invalid_cites) / len(cites) if cites else 1.0
    unsupported = len(invalid_cites) / len(cites) if cites else 0.0
    pred_flags = clean_flags(pred.get("governance_flags", []))
    text = norm(collect_text(pred))
    found = [term for term in forbidden if norm(term) and norm(term) in text]
    return {**base,
        "order_exact": pred_order[:len(gold_order)] == gold_order,
        "top1_correct": bool(pred_order) and bool(gold_order) and pred_order[0] == gold_order[0],
        "supplier_set_valid": len(invalid_suppliers) == 0,
        "invalid_supplier_count": len(invalid_suppliers),
        "citation_valid_rate": citation_valid,
        "unsupported_citation_rate": unsupported,
        "priority_group_f1": f1(gold_groups, pred_groups),
        "governance_flag_f1": f1(gold_flags, pred_flags),
        "human_review_correct": bool(pred.get("human_review_required")) == gold_review,
        "forbidden_terms_absent": len(found) == 0,
        "forbidden_terms_found_json": json.dumps(found, ensure_ascii=False),
        "pred_governance_flags_json": json.dumps(sorted(pred_flags), ensure_ascii=False),
        "pred_priority_groups_json": json.dumps(sorted(pred_groups), ensure_ascii=False),
        "pred_supplier_order_json": json.dumps(pred_order, ensure_ascii=False),
    }

def summarize(results, metrics):
    rows = [
        ("API_SUCCESS_RATE", results["status"].eq("OK").mean()),
        ("ORDER_EXACT_RATE", metrics["order_exact"].mean()),
        ("TOP1_CORRECT_RATE", metrics["top1_correct"].mean()),
        ("SUPPLIER_SET_VALID_RATE", metrics["supplier_set_valid"].mean()),
        ("MEAN_CITATION_VALID_RATE", metrics["citation_valid_rate"].mean()),
        ("MEAN_UNSUPPORTED_CITATION_RATE", metrics["unsupported_citation_rate"].mean()),
        ("MEAN_PRIORITY_GROUP_F1", metrics["priority_group_f1"].mean()),
        ("MEAN_GOVERNANCE_FLAG_F1", metrics["governance_flag_f1"].mean()),
        ("HUMAN_REVIEW_ACCURACY", metrics["human_review_correct"].mean()),
        ("FORBIDDEN_TERMS_ABSENT_RATE", metrics["forbidden_terms_absent"].mean()),
        ("N_CASES", len(metrics)),
    ]
    return pd.DataFrame(rows, columns=["METRIC", "VALUE"])

def by_category(metrics):
    rows = []
    for cat, g in metrics.groupby("category", sort=False):
        rows.append({"CATEGORY": cat, "N": len(g), "ORDER_EXACT_RATE": g["order_exact"].mean(), "TOP1_CORRECT_RATE": g["top1_correct"].mean(), "SUPPLIER_SET_VALID_RATE": g["supplier_set_valid"].mean(), "MEAN_CITATION_VALID_RATE": g["citation_valid_rate"].mean(), "MEAN_PRIORITY_GROUP_F1": g["priority_group_f1"].mean(), "MEAN_GOVERNANCE_FLAG_F1": g["governance_flag_f1"].mean(), "HUMAN_REVIEW_ACCURACY": g["human_review_correct"].mean(), "FORBIDDEN_TERMS_ABSENT_RATE": g["forbidden_terms_absent"].mean()})
    return pd.DataFrame(rows)

def make_errors(results, metrics):
    merged = metrics.merge(results[["case_id","user_request","scenario_profile","prediction_json","error"]], on="case_id", how="left")
    return merged.loc[(~merged["order_exact"])|(~merged["top1_correct"])|(~merged["supplier_set_valid"])|(merged["citation_valid_rate"]<1.0)|(merged["priority_group_f1"]<1.0)|(merged["governance_flag_f1"]<1.0)|(~merged["human_review_correct"])|(~merged["forbidden_terms_absent"])].copy()

def write_report(summary, cat, path):
    m = dict(zip(summary["METRIC"], summary["VALUE"]))
    lines = ["# E5 Explanation and Governance-Audit Evaluation", "", "## Overall results", "",
        f"- API success rate: {m.get('API_SUCCESS_RATE', float('nan')):.3f}",
        f"- Deterministic order preservation: {m.get('ORDER_EXACT_RATE', float('nan')):.3f}",
        f"- Top-1 preservation: {m.get('TOP1_CORRECT_RATE', float('nan')):.3f}",
        f"- Valid supplier-ID rate: {m.get('SUPPLIER_SET_VALID_RATE', float('nan')):.3f}",
        f"- Mean citation-validity rate: {m.get('MEAN_CITATION_VALID_RATE', float('nan')):.3f}",
        f"- Mean unsupported-citation rate: {m.get('MEAN_UNSUPPORTED_CITATION_RATE', float('nan')):.3f}",
        f"- Mean priority-group F1: {m.get('MEAN_PRIORITY_GROUP_F1', float('nan')):.3f}",
        f"- Mean governance-flag F1: {m.get('MEAN_GOVERNANCE_FLAG_F1', float('nan')):.3f}",
        f"- Human-review flag accuracy: {m.get('HUMAN_REVIEW_ACCURACY', float('nan')):.3f}",
        f"- Forbidden-term absence rate: {m.get('FORBIDDEN_TERMS_ABSENT_RATE', float('nan')):.3f}",
        f"- Cases: {int(m.get('N_CASES', 0))}", "", "## Results by case category", "",
        "| Category | N | Order exact | Top-1 | Supplier IDs valid | Citation validity | Priority F1 | Governance F1 | Human-review accuracy | Forbidden terms absent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for _, r in cat.iterrows():
        lines.append(f"| `{r['CATEGORY']}` | {int(r['N'])} | {r['ORDER_EXACT_RATE']:.3f} | {r['TOP1_CORRECT_RATE']:.3f} | {r['SUPPLIER_SET_VALID_RATE']:.3f} | {r['MEAN_CITATION_VALID_RATE']:.3f} | {r['MEAN_PRIORITY_GROUP_F1']:.3f} | {r['MEAN_GOVERNANCE_FLAG_F1']:.3f} | {r['HUMAN_REVIEW_ACCURACY']:.3f} | {r['FORBIDDEN_TERMS_ABSENT_RATE']:.3f} |")
    lines += ["", "## Interpretation", "", "E5 evaluates whether the explanation LLM preserves the deterministic ranking, uses only valid supplier and claim identifiers, avoids prohibited language, and flags governance issues. It does not evaluate whether the ranking itself is optimal. The observed CAN winner/reference outcome is not used in this explanation benchmark.", "", "Error rows should be manually inspected before final reporting, especially when governance-flag F1 is below one because some flags are judgment-sensitive."]
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--output-dir", type=Path, default=Path("data/analysis/e5_openai"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports/e5_openai"))
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--retry-base-seconds", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", choices=["none","low","medium","high","xhigh","max"], default="low")
    ap.add_argument("--max-output-tokens", type=int, default=3500)
    ap.add_argument("--timeout-seconds", type=float, default=90.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--store-responses", action="store_true")
    args = ap.parse_args()
    if not os.getenv("OPENAI_API_KEY"): raise EnvironmentError("OPENAI_API_KEY is not set")
    if not args.benchmark.exists(): raise FileNotFoundError(args.benchmark)
    args.output_dir.mkdir(parents=True, exist_ok=True); args.report_dir.mkdir(parents=True, exist_ok=True)
    bench = pd.read_csv(args.benchmark); validate_benchmark(bench)
    if args.limit is not None: bench = bench.head(args.limit).copy()
    ckpt = args.output_dir / "e5_checkpoint.jsonl"
    if ckpt.exists() and not args.resume: raise FileExistsError(f"Checkpoint exists: {ckpt}. Use --resume or a new output dir.")
    completed = checkpoint_load(ckpt) if args.resume else {}; records = list(completed.values())
    client = OpenAI(timeout=args.timeout_seconds, max_retries=0)
    started = datetime.now(timezone.utc).isoformat(); made = 0; skipped = 0
    total = len(bench)
    for i, (_, row) in enumerate(bench.iterrows(), 1):
        cid = str(row["case_id"])
        if cid in completed:
            skipped += 1; print(f"[{i}/{total}] {cid} (checkpointed; no API call made)"); continue
        print(f"[{i}/{total}] {cid}")
        base = {k: py_scalar(v) for k, v in row.to_dict().items()}
        try:
            resp, parsed, latency, retry_count = call_api(client, args, row); made += 1
            usage = getattr(resp, "usage", None)
            rec = {**base, "status":"OK", "api_call_made":True, "store_responses":bool(args.store_responses), "model":args.model, "response_id":getattr(resp,"id",None), "latency_seconds":latency, "retry_count":retry_count, "input_tokens":usage_value(usage,"input_tokens","prompt_tokens"), "output_tokens":usage_value(usage,"output_tokens","completion_tokens"), "prediction_json":json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False), "error":None}
        except Exception as exc:
            rec = {**base, "status":"ERROR", "api_call_made":True, "store_responses":bool(args.store_responses), "model":args.model, "response_id":None, "latency_seconds":None, "retry_count":args.max_retries, "input_tokens":None, "output_tokens":None, "prediction_json":None, "error":f"{type(exc).__name__}: {exc}"}
        records.append(rec); checkpoint_append(ckpt, rec)
    by_id = {str(r["case_id"]): r for r in records}; ids = bench["case_id"].astype(str).tolist()
    missing = [x for x in ids if x not in by_id]
    if missing: raise RuntimeError(f"Missing results: {missing[:10]}")
    results = pd.DataFrame([by_id[x] for x in ids])
    metrics = pd.DataFrame([evaluate_row(row) for _, row in results.iterrows()])
    summary = summarize(results, metrics); cat = by_category(metrics); errors = make_errors(results, metrics)
    results.to_csv(args.output_dir / "e5_api_results.csv", index=False, encoding="utf-8-sig")
    try:
        results.to_parquet(args.output_dir / "e5_api_results.parquet", index=False)
    except Exception as exc:
        print(f"WARNING: could not write parquet: {type(exc).__name__}: {exc}")
    metrics.to_csv(args.output_dir / "e5_case_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.report_dir / "e5_summary_metrics.csv", index=False)
    cat.to_csv(args.report_dir / "e5_metrics_by_category.csv", index=False)
    errors.to_csv(args.report_dir / "e5_error_rows.csv", index=False, encoding="utf-8-sig")
    metadata = {"model":args.model, "run_started_utc":started, "run_finished_utc":datetime.now(timezone.utc).isoformat(), "n_cases":int(total), "n_completed_api_responses_this_run":int(made), "n_checkpointed_requests_skipped_this_run":int(skipped), "reasoning_effort":args.reasoning_effort, "max_output_tokens":args.max_output_tokens, "benchmark_sha256":sha256_file(args.benchmark), "store_responses":bool(args.store_responses), "system_prompt":SYSTEM_PROMPT, "schema":ExplanationOutput.model_json_schema()}
    (args.report_dir / "e5_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, cat, args.report_dir / "e5_report.md")
    print(); print(f"Completed API responses this run: {made}"); print(f"Checkpointed cases skipped this run: {skipped}")
    if made == 0: print("WARNING: no new API responses were created in this run.")
    if not args.store_responses: print("Note: --store-responses was not enabled, so responses may not appear in stored response logs/history.")
    print(); print("E5 complete."); print("Review first:"); print("  e5_report.md"); print("  e5_summary_metrics.csv"); print("  e5_metrics_by_category.csv"); print("  e5_error_rows.csv")

if __name__ == "__main__":
    main()
