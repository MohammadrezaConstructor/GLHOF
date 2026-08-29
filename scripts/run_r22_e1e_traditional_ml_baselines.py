#!/usr/bin/env python3
"""
E1e: temporally separated traditional-ML ranking baselines for Reviewer 2.2.

Purpose
-------
Compare the frozen deterministic procurement-fit TOPSIS model with:
- regularized logistic regression;
- histogram gradient boosting.

The supervised models predict the historical reference outcome. They do not
learn a normatively optimal supplier-selection rule.

Design
------
- 2017 cases are sorted by target dispatch date.
- Earliest 60%: training region.
- Next 20%: validation region.
- Latest 20%: test region.
- Training uses the reference supplier plus bounded negative sampling per case.
- Validation/test rank the reference supplier against the complete historically
  active CPV2_MIN1 comparison set.
- TOPSIS is recomputed on the exact same validation/test cases.

Time controls
-------------
Defaults are a credible pilot:
- 1,500 training cases;
- 300 validation cases;
- 500 test cases;
- 20 random + 5 hard negatives per training case.

For the paper run, use approximately:
--max-train-cases 6000 --max-validation-cases 1000 --max-test-cases 2000

No API calls are made.

Requirements
------------
pip install numpy pandas pyarrow scikit-learn joblib
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - local dependency check
    raise SystemExit(
        "Missing scikit-learn/joblib. Install with: "
        "pip install scikit-learn joblib"
    ) from exc


FINAL_FEATURES = [
    "CPV2_EXPERIENCE",
    "CPV_EXACT_SET_FIT",
    "CPV4_SET_FIT",
    "CPV3_SET_FIT",
    "COUNTRY_FIT",
    "NUTS1_FIT",
    "NUTS2_FIT",
    "NUTS3_FIT",
    "PRIOR_BUYER_COUNT",
    "PRIOR_BUYER_RECENCY",
    "BUYER_RELATIONSHIP_SHARE",
    "BUYER_BREADTH",
    "PUBLIC_ACTIVITY_RECENCY",
    "CONTEXT_SPECIALIZATION",
]
FINAL_GROUPS = [
    "CATEGORY_FIT",
    "GEOGRAPHIC_FIT",
    "BUYER_RELATIONSHIP",
    "ACTIVITY_PROFILE",
]


@dataclass
class CaseData:
    case_id: str
    dispatch_date: pd.Timestamp
    cpv2: str
    pool_size: int
    winner_local: int
    features: np.ndarray
    topsis_scores: np.ndarray


def load_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("r22_e1e_e1x_base", str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import E1x runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bounded_subsample(df: pd.DataFrame, max_cases: int, seed: int) -> pd.DataFrame:
    if max_cases <= 0 or len(df) <= max_cases:
        return df.copy()
    return (
        df.sample(n=max_cases, random_state=seed, replace=False)
        .sort_values(["TARGET_DISPATCH_DATE", "TARGET_CASE_ID"])
        .reset_index(drop=True)
    )


def chronological_splits(
    targets: pd.DataFrame,
    max_train: int,
    max_validation: int,
    max_test: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = targets.copy()
    ordered["TARGET_DISPATCH_DATE"] = pd.to_datetime(
        ordered["TARGET_DISPATCH_DATE"], errors="raise"
    ).dt.normalize()
    ordered = ordered.sort_values(
        ["TARGET_DISPATCH_DATE", "TARGET_CASE_ID"]
    ).reset_index(drop=True)

    unique_dates = np.sort(ordered["TARGET_DISPATCH_DATE"].dropna().unique())
    if len(unique_dates) < 5:
        raise ValueError("Need at least five unique target dates for a temporal split.")
    train_cut_pos = max(1, int(np.floor(0.60 * len(unique_dates)))) - 1
    validation_cut_pos = max(
        train_cut_pos + 1,
        int(np.floor(0.80 * len(unique_dates))) - 1,
    )
    validation_cut_pos = min(validation_cut_pos, len(unique_dates) - 2)
    train_cut = pd.Timestamp(unique_dates[train_cut_pos])
    validation_cut = pd.Timestamp(unique_dates[validation_cut_pos])

    train_region = ordered.loc[ordered["TARGET_DISPATCH_DATE"] <= train_cut]
    validation_region = ordered.loc[
        (ordered["TARGET_DISPATCH_DATE"] > train_cut)
        & (ordered["TARGET_DISPATCH_DATE"] <= validation_cut)
    ]
    test_region = ordered.loc[ordered["TARGET_DISPATCH_DATE"] > validation_cut]

    train = bounded_subsample(train_region, max_train, seed)
    validation = bounded_subsample(validation_region, max_validation, seed + 1)
    test = bounded_subsample(test_region, max_test, seed + 2)

    sets = [set(x["TARGET_CASE_ID"].astype(str)) for x in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Temporal case splits overlap.")
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition.")
    if train["TARGET_DISPATCH_DATE"].max() >= validation["TARGET_DISPATCH_DATE"].min():
        raise ValueError("Training and validation dates are not strictly separated.")
    if validation["TARGET_DISPATCH_DATE"].max() >= test["TARGET_DISPATCH_DATE"].min():
        raise ValueError("Validation and test dates are not strictly separated.")
    return train, validation, test


def project_features(module, x: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    idx = np.array([module.FEATURE_INDEX[name] for name in FINAL_FEATURES], dtype=np.int64)
    raw = x[:, idx].astype(np.float64, copy=True)
    active = active_mask[idx].astype(np.float64, copy=False)
    raw[:, active == 0] = 0.0
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    masks = np.broadcast_to(active, raw.shape)
    return np.concatenate([raw, masks], axis=1).astype(np.float32, copy=False)


def safe_percentile(rank: float, pool_size: int) -> float:
    if pool_size <= 1:
        return 1.0
    return 1.0 - ((rank - 1.0) / (pool_size - 1.0))


def make_paths(root: Path) -> dict[str, Path]:
    return {
        "e1x_script": root / "scripts/run_e1x_extended_procurement_fit_rankings_v2.py",
        "extended": root / "data/analysis/e1x_extended_fit_indices",
        "base": root / "data/analysis/e1b_fit_indices",
        "target_case": root / "data/processed/supplier_feature_store/target_case_features_2017.parquet",
        "cpv2": root / "data/processed/supplier_feature_store/supplier_cpv2_features_2015_2016.parquet",
    }


def load_engine(root: Path):
    paths = make_paths(root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  - " + "\n  - ".join(missing))

    module = load_module(paths["e1x_script"])
    extended = paths["extended"]

    targets = module.load_targets(
        extended / "e1x_target_profiles_2017.parquet",
        paths["target_case"],
    )
    cpv_token_map = module.load_token_map(
        extended / "e1x_target_cpv_tokens_2017.parquet",
        ["CPV_CODE", "CPV3", "CPV4"],
    )
    nuts_token_map = module.load_token_map(
        extended / "e1x_target_nuts_tokens_2017.parquet",
        ["NUTS1", "NUTS2", "NUTS3"],
    )
    candidates = module.load_cpv2_candidates(
        paths["cpv2"],
        extended / "supplier_cpv2_context_descriptors_2015_2016.parquet",
    )
    supplier_array, supplier_to_idx = module.create_supplier_universe(candidates)
    cpv_lookup = module.load_hierarchical_sparse_index(
        extended / "supplier_cpv_fit_counts_2015_2016.parquet",
        level_col="CPV_LEVEL",
        needed_levels={"EXACT", "CPV4", "CPV3"},
        supplier_to_idx=supplier_to_idx,
    )
    nuts_lookup = module.load_hierarchical_sparse_index(
        extended / "supplier_nuts_fit_counts_2015_2016.parquet",
        level_col="NUTS_LEVEL",
        needed_levels={"NUTS1", "NUTS2", "NUTS3"},
        supplier_to_idx=supplier_to_idx,
    )
    country_lookup = module.load_simple_count_lookup(
        paths["base"] / "supplier_country_fit_counts_2015_2016.parquet",
        count_col="N_COUNTRY_AWARDS",
        supplier_to_idx=supplier_to_idx,
    )
    buyer_lookup = module.load_buyer_lookup(
        extended / "supplier_buyer_relationships_2015_2016.parquet",
        supplier_to_idx,
    )
    regime_match_lookup, regime_total_lookup = module.load_regime_lookup(
        extended / "supplier_regime_fit_counts_2015_2016.parquet",
        supplier_to_idx,
    )
    criterion_lookup = module.load_simple_count_lookup(
        extended / "supplier_criterion_code_fit_counts_2015_2016.parquet",
        count_col="N_MATCHED_AWARDS",
        supplier_to_idx=supplier_to_idx,
    )

    resources = {
        "module": module,
        "targets": targets,
        "candidates": candidates,
        "candidate_groups": {
            str(context): g.reset_index(drop=True)
            for context, g in candidates.groupby("CONTEXT_KEY", sort=False, observed=True)
        },
        "cpv_token_map": cpv_token_map,
        "nuts_token_map": nuts_token_map,
        "supplier_to_idx": supplier_to_idx,
        "cpv_lookup": cpv_lookup,
        "nuts_lookup": nuts_lookup,
        "country_lookup": country_lookup,
        "buyer_lookup": buyer_lookup,
        "regime_match_lookup": regime_match_lookup,
        "regime_total_lookup": regime_total_lookup,
        "criterion_lookup": criterion_lookup,
    }
    return resources


def iter_cases(resources, selected_targets: pd.DataFrame) -> Iterator[CaseData]:
    module = resources["module"]
    supplier_to_idx = resources["supplier_to_idx"]
    global_to_local = np.full(len(supplier_to_idx), -1, dtype=np.int32)

    for context_key, target_context in selected_targets.groupby(
        "TARGET_CPV2", sort=False, observed=True
    ):
        context_key = str(context_key)
        cands = resources["candidate_groups"].get(context_key)
        if cands is None or cands.empty:
            raise ValueError(f"Missing candidate context {context_key}.")

        candidate_ids = cands["SUPPLIER_ENTITY_ID"].astype(str).to_numpy()
        candidate_global_idx = (
            pd.Series(candidate_ids).map(supplier_to_idx).to_numpy(dtype=np.int32)
        )
        if np.any(candidate_global_idx < 0):
            raise ValueError(f"Unmapped candidates in context {context_key}.")
        global_to_local[candidate_global_idx] = np.arange(len(cands), dtype=np.int32)

        for _, target_row in target_context.iterrows():
            case_id = str(target_row["TARGET_CASE_ID"])
            winner_id = str(target_row["OBSERVED_WINNER_HISTORICAL_ENTITY_ID"])
            winner_global = supplier_to_idx.get(winner_id, None)
            if winner_global is None:
                raise ValueError(f"Reference supplier {winner_id} not in supplier universe.")
            winner_local = int(global_to_local[int(winner_global)])
            if winner_local < 0:
                raise ValueError(f"Reference supplier absent from context {context_key}.")

            x, active_mask = module.make_case_features(
                candidates=cands,
                target_row=target_row,
                cpv_levels=resources["cpv_token_map"].get(case_id, {}),
                nuts_levels=resources["nuts_token_map"].get(case_id, {}),
                global_to_local=global_to_local,
                cpv_lookup=resources["cpv_lookup"],
                nuts_lookup=resources["nuts_lookup"],
                country_lookup=resources["country_lookup"],
                buyer_lookup=resources["buyer_lookup"],
                regime_match_lookup=resources["regime_match_lookup"],
                regime_total_lookup=resources["regime_total_lookup"],
                criterion_lookup=resources["criterion_lookup"],
            )
            features = project_features(module, x, active_mask)
            normalized, nonzero = module.normalize_feature_matrix(x)
            contributions = module.group_distance_contributions(
                normalized, active_mask, nonzero
            )
            topsis_scores = module.topsis_scores_from_groups(contributions, FINAL_GROUPS)
            yield CaseData(
                case_id=case_id,
                dispatch_date=pd.Timestamp(target_row["TARGET_DISPATCH_DATE"]),
                cpv2=context_key,
                pool_size=len(cands),
                winner_local=winner_local,
                features=features,
                topsis_scores=topsis_scores,
            )

        global_to_local[candidate_global_idx] = -1


def choose_training_rows(
    case: CaseData,
    random_negatives: int,
    hard_negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = case.pool_size
    all_idx = np.arange(n, dtype=np.int64)
    nonwinner = all_idx[all_idx != case.winner_local]

    hard_idx = np.array([], dtype=np.int64)
    if hard_negatives > 0 and len(nonwinner) > 0:
        order = nonwinner[np.argsort(-case.topsis_scores[nonwinner], kind="stable")]
        hard_idx = order[: min(hard_negatives, len(order))]

    remaining = np.setdiff1d(nonwinner, hard_idx, assume_unique=False)
    random_n = min(random_negatives, len(remaining))
    random_idx = (
        rng.choice(remaining, size=random_n, replace=False)
        if random_n > 0
        else np.array([], dtype=np.int64)
    )
    selected = np.concatenate(
        [np.array([case.winner_local], dtype=np.int64), hard_idx, random_idx]
    )
    y = np.zeros(len(selected), dtype=np.int8)
    y[0] = 1
    return case.features[selected], y


def build_training_matrix(
    resources,
    train_targets: pd.DataFrame,
    random_negatives: int,
    hard_negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    case_rows: list[dict] = []
    started = time.perf_counter()

    for i, case in enumerate(iter_cases(resources, train_targets), start=1):
        x_case, y_case = choose_training_rows(
            case, random_negatives, hard_negatives, rng
        )
        x_parts.append(x_case)
        y_parts.append(y_case)
        case_rows.append(
            {
                "TARGET_CASE_ID": case.case_id,
                "TARGET_DISPATCH_DATE": case.dispatch_date,
                "TARGET_CPV2": case.cpv2,
                "POOL_SIZE": case.pool_size,
                "N_TRAIN_ROWS": len(y_case),
                "N_POSITIVE": int(y_case.sum()),
            }
        )
        if i % 250 == 0 or i == len(train_targets):
            print(f"    training features {i:,}/{len(train_targets):,} cases")

    x_train = np.vstack(x_parts).astype(np.float32, copy=False)
    y_train = np.concatenate(y_parts)
    print(
        f"    built {len(y_train):,} training rows in "
        f"{time.perf_counter() - started:.1f}s"
    )
    return x_train, y_train, pd.DataFrame(case_rows)


def fit_models(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    logistic.fit(x_train, y_train)

    positives = max(int(y_train.sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    sample_weight = np.where(y_train == 1, negatives / positives, 1.0)
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.10,
        random_state=seed,
    )
    hgb.fit(x_train, y_train, sample_weight=sample_weight)
    return {"LOGISTIC_REGRESSION": logistic, "HIST_GRADIENT_BOOSTING": hgb}


def evaluate_split(resources, targets: pd.DataFrame, models: dict, split: str) -> pd.DataFrame:
    module = resources["module"]
    records: list[dict] = []
    for i, case in enumerate(iter_cases(resources, targets), start=1):
        score_sets = {"TOPSIS_M3": case.topsis_scores}
        for name, model in models.items():
            score_sets[name] = model.predict_proba(case.features)[:, 1]

        for name, scores in score_sets.items():
            mid, best, worst, tie = module.direct_winner_rank(scores, case.winner_local)
            records.append(
                {
                    "SPLIT": split,
                    "TARGET_CASE_ID": case.case_id,
                    "TARGET_DISPATCH_DATE": case.dispatch_date,
                    "TARGET_CPV2": case.cpv2,
                    "MODEL": name,
                    "POOL_SIZE": case.pool_size,
                    "WINNER_RANK_MID": mid,
                    "WINNER_RANK_BEST": best,
                    "WINNER_RANK_WORST": worst,
                    "TIE_GROUP_SIZE": tie,
                    "RECIPROCAL_RANK": 1.0 / mid,
                    "PERCENTILE_RANK": safe_percentile(mid, case.pool_size),
                    "WINNER_AT_1": mid <= 1,
                    "WINNER_AT_5": mid <= 5,
                    "WINNER_AT_10": mid <= 10,
                    "WINNER_AT_50": mid <= 50,
                }
            )
        if i % 100 == 0 or i == len(targets):
            print(f"    {split.lower()} scoring {i:,}/{len(targets):,} cases")
    return pd.DataFrame.from_records(records)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, model), g in results.groupby(["SPLIT", "MODEL"], sort=False, observed=True):
        rows.append(
            {
                "SPLIT": split,
                "MODEL": model,
                "N_CASES": len(g),
                "W_AT_1_PCT": 100.0 * g["WINNER_AT_1"].mean(),
                "W_AT_5_PCT": 100.0 * g["WINNER_AT_5"].mean(),
                "W_AT_10_PCT": 100.0 * g["WINNER_AT_10"].mean(),
                "W_AT_50_PCT": 100.0 * g["WINNER_AT_50"].mean(),
                "MRR": g["RECIPROCAL_RANK"].mean(),
                "MEAN_PERCENTILE": g["PERCENTILE_RANK"].mean(),
                "MEDIAN_RANK": g["WINNER_RANK_MID"].median(),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, metadata: dict, output_path: Path) -> None:
    lines = [
        "# E1e Traditional-ML Ranking Baselines",
        "",
        "The supervised models predict the historical reference outcome; they do not "
        "identify a normatively optimal supplier. Validation and test rankings use the "
        "complete CPV2_MIN1 historically active supplier comparison set.",
        "",
        "## Temporal split",
        "",
        f"- Training cases: {metadata['n_train_cases']:,}",
        f"- Validation cases: {metadata['n_validation_cases']:,}",
        f"- Test cases: {metadata['n_test_cases']:,}",
        f"- Training rows after negative sampling: {metadata['n_training_rows']:,}",
        f"- Random negatives per training case: {metadata['random_negatives_per_case']}",
        f"- Hard negatives per training case: {metadata['hard_negatives_per_case']}",
        "",
        "## Ranking results",
        "",
        "| Split | Model | N | W@1 | W@5 | W@10 | W@50 | MRR | Mean percentile | Median rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r['SPLIT']}` | `{r['MODEL']}` | {int(r['N_CASES'])} "
            f"| {r['W_AT_1_PCT']:.3f} | {r['W_AT_5_PCT']:.3f} "
            f"| {r['W_AT_10_PCT']:.3f} | {r['W_AT_50_PCT']:.3f} "
            f"| {r['MRR']:.6f} | {r['MEAN_PERCENTILE']:.6f} "
            f"| {r['MEDIAN_RANK']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- ML performance reflects predictability of historical awards from frozen "
            "observable features, not supplier quality or optimality.",
            "- Negative sampling is used only for model fitting; complete comparison sets "
            "are used for validation and test ranking.",
            "- The temporal split prevents later target outcomes from entering training.",
            "- Results from the default pilot should be labelled as a bounded baseline "
            "until the larger final configuration is executed.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-train-cases", type=int, default=1500)
    parser.add_argument("--max-validation-cases", type=int, default=300)
    parser.add_argument("--max-test-cases", type=int, default=500)
    parser.add_argument("--random-negatives", type=int, default=20)
    parser.add_argument("--hard-negatives", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    resources = load_engine(root)
    module = resources["module"]

    unknown_features = sorted(set(FINAL_FEATURES) - set(module.FEATURE_INDEX))
    unknown_groups = sorted(set(FINAL_GROUPS) - set(module.GROUPS))
    if unknown_features or unknown_groups:
        raise ValueError(
            f"Unknown E1x features/groups. Features={unknown_features}; groups={unknown_groups}"
        )

    train_targets, validation_targets, test_targets = chronological_splits(
        resources["targets"],
        args.max_train_cases,
        args.max_validation_cases,
        args.max_test_cases,
        args.seed,
    )
    print(
        f"Temporal cases: train={len(train_targets):,}, "
        f"validation={len(validation_targets):,}, test={len(test_targets):,}"
    )

    if args.check_only:
        # Exercise one case end-to-end without fitting models.
        first = next(iter_cases(resources, train_targets.head(1)))
        expected_features = 2 * len(FINAL_FEATURES)
        if first.features.shape != (first.pool_size, expected_features):
            raise ValueError(
                f"Unexpected candidate feature shape: {first.features.shape}; "
                f"expected ({first.pool_size}, {expected_features})"
            )
        print(
            f"E1e preflight passed. First case={first.case_id}, "
            f"pool={first.pool_size:,}, features={expected_features}."
        )
        return

    output_dir = root / "data/analysis/r22_e1e_ml_baselines"
    report_dir = root / "reports/r22_e1e_ml_baselines"
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_manifest = build_training_matrix(
        resources,
        train_targets,
        random_negatives=args.random_negatives,
        hard_negatives=args.hard_negatives,
        seed=args.seed,
    )
    models = fit_models(x_train, y_train, args.seed)
    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name.lower()}.joblib")

    validation_results = evaluate_split(
        resources, validation_targets, models, "VALIDATION"
    )
    test_results = evaluate_split(resources, test_targets, models, "TEST")
    results = pd.concat([validation_results, test_results], ignore_index=True)
    summary = summarize(results)

    expected_models = {"TOPSIS_M3", *models.keys()}
    counts = results.groupby(["SPLIT", "TARGET_CASE_ID"], observed=True)["MODEL"].nunique()
    if not counts.eq(len(expected_models)).all():
        raise ValueError("Not every validation/test case has all baseline models.")
    if set(results["MODEL"].astype(str)) != expected_models:
        raise ValueError("Unexpected model set in evaluation results.")

    results.to_parquet(output_dir / "r22_e1e_ml_case_results.parquet", index=False)
    train_manifest.to_csv(output_dir / "r22_e1e_training_case_manifest.csv", index=False)
    summary.to_csv(report_dir / "r22_e1e_ml_summary.csv", index=False)

    metadata = {
        "n_total_target_cases": int(len(resources["targets"])),
        "n_train_cases": int(len(train_targets)),
        "n_validation_cases": int(len(validation_targets)),
        "n_test_cases": int(len(test_targets)),
        "n_training_rows": int(len(y_train)),
        "n_positive_training_rows": int(y_train.sum()),
        "random_negatives_per_case": args.random_negatives,
        "hard_negatives_per_case": args.hard_negatives,
        "feature_names": FINAL_FEATURES,
        "mask_features_included": True,
        "seed": args.seed,
        "train_date_min": str(train_targets["TARGET_DISPATCH_DATE"].min()),
        "train_date_max": str(train_targets["TARGET_DISPATCH_DATE"].max()),
        "validation_date_min": str(validation_targets["TARGET_DISPATCH_DATE"].min()),
        "validation_date_max": str(validation_targets["TARGET_DISPATCH_DATE"].max()),
        "test_date_min": str(test_targets["TARGET_DISPATCH_DATE"].min()),
        "test_date_max": str(test_targets["TARGET_DISPATCH_DATE"].max()),
    }
    (report_dir / "r22_e1e_ml_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(summary, metadata, report_dir / "r22_e1e_ml_report.md")
    print(f"E1e complete. Report: {report_dir / 'r22_e1e_ml_report.md'}")


if __name__ == "__main__":
    main()
