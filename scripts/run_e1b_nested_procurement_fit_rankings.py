#!/usr/bin/env python3
"""
E1b nested procurement-fit TOPSIS comparison.

Purpose
-------
Compare increasingly procurement-specific TOPSIS specifications on the exact
same CPV2_MIN1 winner-in-pool cohort used in E1.

Models
------
M0_ORIGINAL
    Exact E1 TOPSIS baseline:
    - CPV2 context experience
    - observed buyer breadth in CPV2 context
    - public-activity recency
    - CPV2 context specialization

M1_CATEGORY_FIT
    M0 conceptually reorganized into two equally weighted groups:
    - category fit: CPV2, CPV3, CPV4 experience
    - activity profile: buyer breadth, recency, specialization

M2_RELATIONSHIP_MARKET
    Three equally weighted groups:
    - category fit
    - relationship/market fit: prior buyer, procurement country
    - activity profile

M3_FULL_CONTEXT_FIT
    Four equally weighted groups:
    - category fit
    - relationship/market fit
    - procurement-context fit:
      contract type, buyer main activity, authority type, procedure type
    - activity profile

Important methodological point
------------------------------
Weights are specified a priori at the conceptual-group level. They are not
estimated from 2017 winner outcomes.

All count-based fit features use log(1 + count). All features are benefits.
Missing target attributes deactivate the corresponding criterion for that case;
remaining active weights are renormalized.

Primary evaluation
------------------
The same 22,230 CPV2_MIN1 winner-in-pool cases as E1.

Outputs
-------
Processed:
- e1b_case_model_results.parquet

Reports:
- e1b_model_specification.csv
- e1b_model_summary.csv
- e1b_model_comparison_vs_m0.csv
- e1b_results_by_cpv2.csv
- e1b_validation_checks.csv
- e1b_ranking_report.md

Requires
--------
pip install numpy pandas pyarrow
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Feature and model definitions
# ---------------------------------------------------------------------

FEATURES = [
    "CPV2_EXPERIENCE",
    "BUYER_BREADTH",
    "PUBLIC_ACTIVITY_RECENCY",
    "CONTEXT_SPECIALIZATION",
    "CPV3_FIT",
    "CPV4_FIT",
    "PRIOR_BUYER_FIT",
    "COUNTRY_FIT",
    "CONTRACT_TYPE_FIT",
    "MAIN_ACTIVITY_FIT",
    "AUTHORITY_TYPE_FIT",
    "PROCEDURE_TYPE_FIT",
]

FEATURE_INDEX = {
    name: i
    for i, name in enumerate(FEATURES)
}


MODEL_SPECS: dict[str, dict[str, float]] = {
    # Exact E1 TOPSIS baseline.
    "M0_ORIGINAL": {
        "CPV2_EXPERIENCE": 0.25,
        "BUYER_BREADTH": 0.25,
        "PUBLIC_ACTIVITY_RECENCY": 0.25,
        "CONTEXT_SPECIALIZATION": 0.25,
    },

    # Two groups, 0.50 each.
    "M1_CATEGORY_FIT": {
        "CPV2_EXPERIENCE": 1.0 / 6.0,
        "CPV3_FIT": 1.0 / 6.0,
        "CPV4_FIT": 1.0 / 6.0,
        "BUYER_BREADTH": 1.0 / 6.0,
        "PUBLIC_ACTIVITY_RECENCY": 1.0 / 6.0,
        "CONTEXT_SPECIALIZATION": 1.0 / 6.0,
    },

    # Three groups, 1/3 each.
    "M2_RELATIONSHIP_MARKET": {
        "CPV2_EXPERIENCE": 1.0 / 9.0,
        "CPV3_FIT": 1.0 / 9.0,
        "CPV4_FIT": 1.0 / 9.0,

        "PRIOR_BUYER_FIT": 1.0 / 6.0,
        "COUNTRY_FIT": 1.0 / 6.0,

        "BUYER_BREADTH": 1.0 / 9.0,
        "PUBLIC_ACTIVITY_RECENCY": 1.0 / 9.0,
        "CONTEXT_SPECIALIZATION": 1.0 / 9.0,
    },

    # Four groups, 0.25 each.
    "M3_FULL_CONTEXT_FIT": {
        "CPV2_EXPERIENCE": 1.0 / 12.0,
        "CPV3_FIT": 1.0 / 12.0,
        "CPV4_FIT": 1.0 / 12.0,

        "PRIOR_BUYER_FIT": 0.125,
        "COUNTRY_FIT": 0.125,

        "CONTRACT_TYPE_FIT": 0.0625,
        "MAIN_ACTIVITY_FIT": 0.0625,
        "AUTHORITY_TYPE_FIT": 0.0625,
        "PROCEDURE_TYPE_FIT": 0.0625,

        "BUYER_BREADTH": 1.0 / 12.0,
        "PUBLIC_ACTIVITY_RECENCY": 1.0 / 12.0,
        "CONTEXT_SPECIALIZATION": 1.0 / 12.0,
    },
}


FIT_INDEX_SPECS = {
    "CPV3_FIT": {
        "filename": "supplier_cpv3_fit_counts_2015_2016.parquet",
        "count_col": "N_CPV3_AWARDS",
        "target_col": "TARGET_CPV3",
    },
    "CPV4_FIT": {
        "filename": "supplier_cpv4_fit_counts_2015_2016.parquet",
        "count_col": "N_CPV4_AWARDS",
        "target_col": "TARGET_CPV4",
    },
    "PRIOR_BUYER_FIT": {
        "filename": "supplier_buyer_fit_counts_2015_2016.parquet",
        "count_col": "N_PRIOR_BUYER_AWARDS",
        "target_col": "TARGET_BUYER_KEY",
    },
    "COUNTRY_FIT": {
        "filename": "supplier_country_fit_counts_2015_2016.parquet",
        "count_col": "N_COUNTRY_AWARDS",
        "target_col": "TARGET_PROCUREMENT_COUNTRY",
    },
    "CONTRACT_TYPE_FIT": {
        "filename": "supplier_contract_type_fit_counts_2015_2016.parquet",
        "count_col": "N_CONTRACT_TYPE_AWARDS",
        "target_col": "TARGET_CONTRACT_TYPE",
    },
    "MAIN_ACTIVITY_FIT": {
        "filename": "supplier_main_activity_fit_counts_2015_2016.parquet",
        "count_col": "N_MAIN_ACTIVITY_AWARDS",
        "target_col": "TARGET_MAIN_ACTIVITY",
    },
    "AUTHORITY_TYPE_FIT": {
        "filename": "supplier_authority_type_fit_counts_2015_2016.parquet",
        "count_col": "N_AUTHORITY_TYPE_AWARDS",
        "target_col": "TARGET_AUTHORITY_TYPE",
    },
    "PROCEDURE_TYPE_FIT": {
        "filename": "supplier_procedure_type_fit_counts_2015_2016.parquet",
        "count_col": "N_PROCEDURE_TYPE_AWARDS",
        "target_col": "TARGET_PROCEDURE_TYPE",
    },
}


SIGNATURE_COLUMNS = [
    "TARGET_DISPATCH_DATE",
    "TARGET_CPV3",
    "TARGET_CPV4",
    "TARGET_BUYER_KEY",
    "TARGET_PROCUREMENT_COUNTRY",
    "TARGET_CONTRACT_TYPE",
    "TARGET_MAIN_ACTIVITY",
    "TARGET_AUTHORITY_TYPE",
    "TARGET_PROCEDURE_TYPE",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def normalize_target_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def rank_with_ties(
    scores: np.ndarray,
    decimals: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Higher scores are better.

    Returns:
      midrank, best rank, worst rank, tie-group size
    """
    s = np.round(
        np.asarray(scores, dtype=np.float64),
        decimals=decimals,
    )

    order = np.argsort(-s, kind="mergesort")
    sorted_s = s[order]

    n = len(s)

    midrank = np.empty(n, dtype=np.float64)
    best_rank = np.empty(n, dtype=np.int64)
    worst_rank = np.empty(n, dtype=np.int64)
    tie_size = np.empty(n, dtype=np.int64)

    start = 0

    while start < n:
        end = start + 1

        while end < n and sorted_s[end] == sorted_s[start]:
            end += 1

        best = start + 1
        worst = end
        mid = (best + worst) / 2.0
        size = end - start

        idx = order[start:end]

        midrank[idx] = mid
        best_rank[idx] = best
        worst_rank[idx] = worst
        tie_size[idx] = size

        start = end

    return midrank, best_rank, worst_rank, tie_size


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def load_target_profiles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = {
        "TARGET_CASE_ID",
        "TARGET_DISPATCH_DATE",
        "TARGET_CPV2",
        "TARGET_CPV3",
        "TARGET_CPV4",
        "TARGET_BUYER_KEY",
        "TARGET_PROCUREMENT_COUNTRY",
        "TARGET_CONTRACT_TYPE",
        "TARGET_MAIN_ACTIVITY",
        "TARGET_AUTHORITY_TYPE",
        "TARGET_PROCEDURE_TYPE",
        "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Target profile table missing columns:\n  - "
            + "\n  - ".join(missing)
        )

    df["TARGET_CASE_ID"] = df["TARGET_CASE_ID"].astype("string")
    df["TARGET_DISPATCH_DATE"] = pd.to_datetime(
        df["TARGET_DISPATCH_DATE"],
        errors="raise",
    )

    for col in [
        "TARGET_CPV2",
        "TARGET_CPV3",
        "TARGET_CPV4",
        "TARGET_BUYER_KEY",
        "TARGET_PROCUREMENT_COUNTRY",
        "TARGET_CONTRACT_TYPE",
        "TARGET_MAIN_ACTIVITY",
        "TARGET_AUTHORITY_TYPE",
        "TARGET_PROCEDURE_TYPE",
        "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
    ]:
        df[col] = df[col].astype("string")

    return df


def load_cpv2_features(path: Path) -> pd.DataFrame:
    cols = [
        "CONTEXT_KEY",
        "SUPPLIER_ENTITY_ID",
        "N_CONTEXT_AWARDS",
        "N_UNIQUE_BUYERS_IN_CONTEXT",
        "LAST_CONTEXT_PUBLICATION_DATE",
        "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT",
    ]

    df = pd.read_parquet(path, columns=cols)

    df = df.loc[
        df["N_CONTEXT_AWARDS"] >= 1
    ].copy()

    df["CONTEXT_KEY"] = df["CONTEXT_KEY"].astype("string")
    df["SUPPLIER_ENTITY_ID"] = df["SUPPLIER_ENTITY_ID"].astype("string")
    df["LAST_CONTEXT_PUBLICATION_DATE"] = pd.to_datetime(
        df["LAST_CONTEXT_PUBLICATION_DATE"],
        errors="raise",
    )

    dup = df.duplicated(
        ["CONTEXT_KEY", "SUPPLIER_ENTITY_ID"],
        keep=False,
    )

    if dup.any():
        raise ValueError(
            f"CPV2 feature table has {int(dup.sum())} duplicated "
            "(CONTEXT_KEY, SUPPLIER_ENTITY_ID) rows."
        )

    return df


def create_supplier_universe(
    cpv2: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series]:
    supplier_ids = pd.Index(
        cpv2["SUPPLIER_ENTITY_ID"]
        .dropna()
        .astype(str)
        .unique()
    )

    supplier_array = supplier_ids.to_numpy(dtype=object)

    supplier_to_idx = pd.Series(
        np.arange(
            len(supplier_array),
            dtype=np.int32,
        ),
        index=supplier_ids,
    )

    return supplier_array, supplier_to_idx


def load_sparse_fit_indices(
    fit_index_dir: Path,
    supplier_to_idx: pd.Series,
) -> dict[
    str,
    dict[str, tuple[np.ndarray, np.ndarray]]
]:
    all_indices = {}

    for feature_name, spec in FIT_INDEX_SPECS.items():
        path = fit_index_dir / spec["filename"]
        require_file(path)

        print(f"    loading {feature_name}...")

        df = pd.read_parquet(
            path,
            columns=[
                "FIT_VALUE",
                "SUPPLIER_ENTITY_ID",
                spec["count_col"],
            ],
        )

        df["FIT_VALUE"] = df["FIT_VALUE"].astype("string")
        supplier_ids = df["SUPPLIER_ENTITY_ID"].astype(str)

        mapped = supplier_ids.map(supplier_to_idx)

        keep = mapped.notna()

        df = df.loc[keep].copy()
        df["SUPPLIER_IDX"] = (
            mapped.loc[keep]
            .astype(np.int32)
            .to_numpy()
        )

        lookup: dict[
            str,
            tuple[np.ndarray, np.ndarray]
        ] = {}

        for fit_value, g in df.groupby(
            "FIT_VALUE",
            sort=False,
            observed=True,
        ):
            lookup[str(fit_value)] = (
                g["SUPPLIER_IDX"].to_numpy(
                    dtype=np.int32,
                    copy=True,
                ),
                g[spec["count_col"]].to_numpy(
                    dtype=np.float64,
                    copy=True,
                ),
            )

        all_indices[feature_name] = lookup

        del df

    return all_indices


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

def sparse_local_feature(
    feature_name: str,
    target_value: str | None,
    n_candidates: int,
    global_to_local: np.ndarray,
    sparse_indices: dict[
        str,
        dict[str, tuple[np.ndarray, np.ndarray]]
    ],
    local_cache: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[np.ndarray, bool]:
    """
    Return log1p historical count vector and target-attribute activity flag.
    """
    values = np.zeros(
        n_candidates,
        dtype=np.float64,
    )

    if target_value is None:
        return values, False

    cache_key = (feature_name, target_value)

    if cache_key not in local_cache:
        source = sparse_indices[
            feature_name
        ].get(target_value)

        if source is None:
            local_cache[cache_key] = (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            )
        else:
            global_idx, counts = source

            local_pos = global_to_local[global_idx]
            keep = local_pos >= 0

            local_cache[cache_key] = (
                local_pos[keep].astype(
                    np.int32,
                    copy=True,
                ),
                np.log1p(
                    counts[keep].astype(
                        np.float64,
                        copy=False,
                    )
                ),
            )

    local_pos, transformed_counts = local_cache[cache_key]

    if len(local_pos):
        values[local_pos] = transformed_counts

    return values, True


def make_all_features(
    candidates: pd.DataFrame,
    target_signature: dict[str, Any],
    global_to_local: np.ndarray,
    sparse_indices: dict[
        str,
        dict[str, tuple[np.ndarray, np.ndarray]]
    ],
    local_cache: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      feature matrix: n_candidates x 12
      target-active mask: length 12
    """
    target_date = pd.Timestamp(
        target_signature["TARGET_DISPATCH_DATE"]
    )

    last_pub = candidates[
        "LAST_CONTEXT_PUBLICATION_DATE"
    ]

    days = (
        target_date.normalize()
        - last_pub.dt.normalize()
    ).dt.days.to_numpy(dtype=np.int64)

    if np.any(days < 0):
        raise ValueError(
            f"Found {int(np.sum(days < 0))} historical publication "
            "dates after target date."
        )

    n = len(candidates)

    arrays: dict[str, np.ndarray] = {
        "CPV2_EXPERIENCE":
            np.log1p(
                candidates[
                    "N_CONTEXT_AWARDS"
                ].to_numpy(dtype=np.float64)
            ),

        "BUYER_BREADTH":
            np.log1p(
                candidates[
                    "N_UNIQUE_BUYERS_IN_CONTEXT"
                ]
                .fillna(0)
                .to_numpy(dtype=np.float64)
            ),

        "PUBLIC_ACTIVITY_RECENCY":
            1.0 / (
                1.0
                + days.astype(np.float64)
            ),

        "CONTEXT_SPECIALIZATION":
            candidates[
                "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT"
            ]
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=np.float64),
    }

    active = {
        name: True
        for name in arrays
    }

    for feature_name, spec in FIT_INDEX_SPECS.items():
        target_value = normalize_target_value(
            target_signature[
                spec["target_col"]
            ]
        )

        values, is_active = sparse_local_feature(
            feature_name=feature_name,
            target_value=target_value,
            n_candidates=n,
            global_to_local=global_to_local,
            sparse_indices=sparse_indices,
            local_cache=local_cache,
        )

        arrays[feature_name] = values
        active[feature_name] = is_active

    matrix = np.column_stack(
        [
            arrays[name]
            for name in FEATURES
        ]
    )

    active_mask = np.array(
        [
            active[name]
            for name in FEATURES
        ],
        dtype=bool,
    )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Feature matrix contains non-finite values."
        )

    return matrix, active_mask


# ---------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------

def vector_normalize_features(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norms = np.sqrt(
        np.sum(
            np.square(x),
            axis=0,
        )
    )

    normalized = np.zeros_like(
        x,
        dtype=np.float64,
    )

    nonzero = norms > 0

    normalized[:, nonzero] = (
        x[:, nonzero]
        / norms[nonzero]
    )

    return normalized, nonzero


def topsis_from_normalized(
    normalized: np.ndarray,
    nonzero_mask: np.ndarray,
    target_active_mask: np.ndarray,
    model_spec: dict[str, float],
) -> np.ndarray:
    feature_idx = np.array(
        [
            FEATURE_INDEX[name]
            for name in model_spec
        ],
        dtype=np.int64,
    )

    base_weights = np.array(
        [
            model_spec[name]
            for name in model_spec
        ],
        dtype=np.float64,
    )

    usable = (
        target_active_mask[feature_idx]
        & nonzero_mask[feature_idx]
    )

    if not usable.any():
        return np.full(
            normalized.shape[0],
            0.5,
            dtype=np.float64,
        )

    idx = feature_idx[usable]
    weights = base_weights[usable]

    weights = weights / weights.sum()

    weighted = (
        normalized[:, idx]
        * weights
    )

    ideal_best = np.max(
        weighted,
        axis=0,
    )
    ideal_worst = np.min(
        weighted,
        axis=0,
    )

    d_best = np.sqrt(
        np.sum(
            np.square(
                weighted - ideal_best
            ),
            axis=1,
        )
    )

    d_worst = np.sqrt(
        np.sum(
            np.square(
                weighted - ideal_worst
            ),
            axis=1,
        )
    )

    denom = d_best + d_worst

    scores = np.full(
        normalized.shape[0],
        0.5,
        dtype=np.float64,
    )

    valid = denom > 0

    scores[valid] = (
        d_worst[valid]
        / denom[valid]
    )

    return scores


# ---------------------------------------------------------------------
# Main ranking loop
# ---------------------------------------------------------------------

def run_nested_models(
    targets: pd.DataFrame,
    cpv2: pd.DataFrame,
    supplier_to_idx: pd.Series,
    sparse_indices: dict[
        str,
        dict[str, tuple[np.ndarray, np.ndarray]]
    ],
) -> pd.DataFrame:
    candidate_groups = {
        str(context): g.reset_index(drop=True)
        for context, g in cpv2.groupby(
            "CONTEXT_KEY",
            sort=False,
            observed=True,
        )
    }

    target_contexts = list(
        targets.groupby(
            "TARGET_CPV2",
            sort=False,
            observed=True,
        )
    )

    global_to_local = np.full(
        len(supplier_to_idx),
        -1,
        dtype=np.int32,
    )

    records: list[dict[str, Any]] = []

    total_contexts = len(target_contexts)

    for context_no, (context_key, target_context) in enumerate(
        target_contexts,
        start=1,
    ):
        context_key = str(context_key)

        candidates = candidate_groups.get(
            context_key
        )

        if candidates is None or candidates.empty:
            raise ValueError(
                f"Missing CPV2 candidate context {context_key}."
            )

        candidate_supplier_ids = (
            candidates["SUPPLIER_ENTITY_ID"]
            .astype(str)
            .to_numpy()
        )

        candidate_global_idx = (
            pd.Series(candidate_supplier_ids)
            .map(supplier_to_idx)
            .to_numpy(dtype=np.int32)
        )

        n_candidates = len(candidates)

        global_to_local[
            candidate_global_idx
        ] = np.arange(
            n_candidates,
            dtype=np.int32,
        )

        local_cache: dict[
            tuple[str, str],
            tuple[np.ndarray, np.ndarray]
        ] = {}

        signature_groups = target_context.groupby(
            SIGNATURE_COLUMNS,
            dropna=False,
            sort=False,
            observed=True,
        )

        n_signature_groups = signature_groups.ngroups

        print(
            f"[{context_no}/{total_contexts}] CPV2={context_key}: "
            f"{len(target_context):,} cases, "
            f"{n_candidates:,} candidates, "
            f"{n_signature_groups:,} signatures"
        )

        for signature_values, cases in signature_groups:
            if not isinstance(
                signature_values,
                tuple,
            ):
                signature_values = (
                    signature_values,
                )

            signature = dict(
                zip(
                    SIGNATURE_COLUMNS,
                    signature_values,
                )
            )

            x, target_active_mask = make_all_features(
                candidates=candidates,
                target_signature=signature,
                global_to_local=global_to_local,
                sparse_indices=sparse_indices,
                local_cache=local_cache,
            )

            normalized, nonzero_mask = (
                vector_normalize_features(x)
            )

            ranking_cache = {}

            for model_name, model_spec in MODEL_SPECS.items():
                scores = topsis_from_normalized(
                    normalized=normalized,
                    nonzero_mask=nonzero_mask,
                    target_active_mask=target_active_mask,
                    model_spec=model_spec,
                )

                ranking_cache[
                    model_name
                ] = rank_with_ties(scores)

            for case in cases.itertuples(index=False):
                winner_id = str(
                    case.OBSERVED_WINNER_HISTORICAL_ENTITY_ID
                )

                winner_global_idx = supplier_to_idx.get(
                    winner_id,
                    None,
                )

                if winner_global_idx is None:
                    raise ValueError(
                        f"Winner {winner_id} absent from supplier universe."
                    )

                winner_local_idx = int(
                    global_to_local[
                        int(winner_global_idx)
                    ]
                )

                if winner_local_idx < 0:
                    raise ValueError(
                        f"Winner {winner_id} absent from CPV2 context "
                        f"{context_key}."
                    )

                for model_name in MODEL_SPECS:
                    (
                        midrank,
                        best_rank,
                        worst_rank,
                        tie_size,
                    ) = ranking_cache[model_name]

                    r_mid = float(
                        midrank[winner_local_idx]
                    )

                    if n_candidates <= 1:
                        percentile = 1.0
                    else:
                        percentile = 1.0 - (
                            (r_mid - 1.0)
                            / (n_candidates - 1.0)
                        )

                    records.append(
                        {
                            "TARGET_CASE_ID":
                                case.TARGET_CASE_ID,
                            "TARGET_CPV2":
                                case.TARGET_CPV2,
                            "TARGET_PROCUREMENT_COUNTRY":
                                case.TARGET_PROCUREMENT_COUNTRY,
                            "MODEL":
                                model_name,
                            "POOL_DEFINITION":
                                "CPV2_MIN1",
                            "POOL_SIZE":
                                n_candidates,
                            "WINNER_RANK_MID":
                                r_mid,
                            "WINNER_RANK_BEST":
                                int(
                                    best_rank[
                                        winner_local_idx
                                    ]
                                ),
                            "WINNER_RANK_WORST":
                                int(
                                    worst_rank[
                                        winner_local_idx
                                    ]
                                ),
                            "TIE_GROUP_SIZE":
                                int(
                                    tie_size[
                                        winner_local_idx
                                    ]
                                ),
                            "RECIPROCAL_RANK":
                                1.0 / r_mid,
                            "PERCENTILE_RANK":
                                percentile,
                            "WINNER_AT_1":
                                r_mid <= 1,
                            "WINNER_AT_5":
                                r_mid <= 5,
                            "WINNER_AT_10":
                                r_mid <= 10,
                            "WINNER_AT_50":
                                r_mid <= 50,
                        }
                    )

        global_to_local[
            candidate_global_idx
        ] = -1

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def model_specification() -> pd.DataFrame:
    rows = []

    group_lookup = {
        "CPV2_EXPERIENCE": "CATEGORY_FIT",
        "CPV3_FIT": "CATEGORY_FIT",
        "CPV4_FIT": "CATEGORY_FIT",

        "PRIOR_BUYER_FIT":
            "RELATIONSHIP_MARKET_FIT",
        "COUNTRY_FIT":
            "RELATIONSHIP_MARKET_FIT",

        "CONTRACT_TYPE_FIT":
            "PROCUREMENT_CONTEXT_FIT",
        "MAIN_ACTIVITY_FIT":
            "PROCUREMENT_CONTEXT_FIT",
        "AUTHORITY_TYPE_FIT":
            "PROCUREMENT_CONTEXT_FIT",
        "PROCEDURE_TYPE_FIT":
            "PROCUREMENT_CONTEXT_FIT",

        "BUYER_BREADTH":
            "ACTIVITY_PROFILE",
        "PUBLIC_ACTIVITY_RECENCY":
            "ACTIVITY_PROFILE",
        "CONTEXT_SPECIALIZATION":
            "ACTIVITY_PROFILE",
    }

    for model_name, spec in MODEL_SPECS.items():
        for feature_name, weight in spec.items():
            rows.append(
                {
                    "MODEL": model_name,
                    "FEATURE": feature_name,
                    "CONCEPTUAL_GROUP":
                        group_lookup[feature_name],
                    "WEIGHT": float(weight),
                    "TRANSFORMATION":
                        (
                            "log(1+x)"
                            if feature_name
                            not in {
                                "PUBLIC_ACTIVITY_RECENCY",
                                "CONTEXT_SPECIALIZATION",
                            }
                            else (
                                "1/(1+days)"
                                if feature_name
                                == "PUBLIC_ACTIVITY_RECENCY"
                                else "identity"
                            )
                        ),
                    "DIRECTION": "benefit",
                }
            )

    return pd.DataFrame(rows)


def summarize_models(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, g in results.groupby(
        "MODEL",
        sort=False,
    ):
        rows.append(
            {
                "MODEL": model,
                "N_CASES": len(g),
                "WINNER_AT_1_PCT":
                    100.0
                    * g["WINNER_AT_1"].mean(),
                "WINNER_AT_5_PCT":
                    100.0
                    * g["WINNER_AT_5"].mean(),
                "WINNER_AT_10_PCT":
                    100.0
                    * g["WINNER_AT_10"].mean(),
                "WINNER_AT_50_PCT":
                    100.0
                    * g["WINNER_AT_50"].mean(),
                "MRR":
                    g["RECIPROCAL_RANK"].mean(),
                "MEAN_PERCENTILE_RANK":
                    g["PERCENTILE_RANK"].mean(),
                "MEDIAN_WINNER_RANK":
                    g["WINNER_RANK_MID"].median(),
                "MEAN_TIE_GROUP_SIZE":
                    g["TIE_GROUP_SIZE"].mean(),
            }
        )

    order = {
        "M0_ORIGINAL": 0,
        "M1_CATEGORY_FIT": 1,
        "M2_RELATIONSHIP_MARKET": 2,
        "M3_FULL_CONTEXT_FIT": 3,
    }

    out = pd.DataFrame(rows)
    out["_order"] = out["MODEL"].map(order)

    return (
        out.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def compare_vs_m0(
    results: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    pivot = results.pivot(
        index="TARGET_CASE_ID",
        columns="MODEL",
        values="WINNER_RANK_MID",
    )

    base_summary = (
        summary_df
        .set_index("MODEL")
        .loc["M0_ORIGINAL"]
    )

    rows = []

    for model in [
        "M1_CATEGORY_FIT",
        "M2_RELATIONSHIP_MARKET",
        "M3_FULL_CONTEXT_FIT",
    ]:
        delta = (
            pivot[model]
            - pivot["M0_ORIGINAL"]
        )

        model_summary = (
            summary_df
            .set_index("MODEL")
            .loc[model]
        )

        rows.append(
            {
                "REFERENCE_MODEL":
                    "M0_ORIGINAL",
                "COMPARISON_MODEL":
                    model,
                "N_CASES":
                    len(delta),
                "DELTA_WINNER_AT_1_PP":
                    model_summary[
                        "WINNER_AT_1_PCT"
                    ]
                    - base_summary[
                        "WINNER_AT_1_PCT"
                    ],
                "DELTA_WINNER_AT_5_PP":
                    model_summary[
                        "WINNER_AT_5_PCT"
                    ]
                    - base_summary[
                        "WINNER_AT_5_PCT"
                    ],
                "DELTA_WINNER_AT_10_PP":
                    model_summary[
                        "WINNER_AT_10_PCT"
                    ]
                    - base_summary[
                        "WINNER_AT_10_PCT"
                    ],
                "DELTA_WINNER_AT_50_PP":
                    model_summary[
                        "WINNER_AT_50_PCT"
                    ]
                    - base_summary[
                        "WINNER_AT_50_PCT"
                    ],
                "DELTA_MRR":
                    model_summary["MRR"]
                    - base_summary["MRR"],
                "MEAN_RANK_CHANGE":
                    float(delta.mean()),
                "MEDIAN_RANK_CHANGE":
                    float(delta.median()),
                "MEAN_ABS_RANK_CHANGE":
                    float(delta.abs().mean()),
                "PCT_WINNER_MOVED_UP":
                    100.0
                    * float(
                        (delta < 0).mean()
                    ),
                "PCT_WINNER_UNCHANGED":
                    100.0
                    * float(
                        (delta == 0).mean()
                    ),
                "PCT_WINNER_MOVED_DOWN":
                    100.0
                    * float(
                        (delta > 0).mean()
                    ),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_cpv2(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (model, cpv2), g in results.groupby(
        ["MODEL", "TARGET_CPV2"],
        dropna=False,
        sort=False,
    ):
        rows.append(
            {
                "MODEL": model,
                "TARGET_CPV2": cpv2,
                "N_CASES": len(g),
                "WINNER_AT_10_PCT":
                    100.0
                    * g["WINNER_AT_10"].mean(),
                "WINNER_AT_50_PCT":
                    100.0
                    * g["WINNER_AT_50"].mean(),
                "MRR":
                    g["RECIPROCAL_RANK"].mean(),
                "MEAN_PERCENTILE_RANK":
                    g["PERCENTILE_RANK"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["MODEL", "N_CASES"],
        ascending=[True, False],
    )


def validation_checks(
    results: pd.DataFrame,
    e1_case_results_path: Path,
) -> pd.DataFrame:
    rows = []

    counts = (
        results.groupby(
            "TARGET_CASE_ID"
        )["MODEL"]
        .nunique()
    )

    rows.append(
        {
            "CHECK":
                "four_models_per_case",
            "VALUE":
                bool(
                    (counts == len(MODEL_SPECS)).all()
                ),
            "DETAIL":
                (
                    f"cases={counts.size}; "
                    f"min={counts.min()}; "
                    f"max={counts.max()}"
                ),
        }
    )

    dup_count = int(
        results.duplicated(
            ["TARGET_CASE_ID", "MODEL"]
        ).sum()
    )

    rows.append(
        {
            "CHECK":
                "unique_case_model_rows",
            "VALUE":
                dup_count == 0,
            "DETAIL":
                f"duplicate_rows={dup_count}",
        }
    )

    model_counts = (
        results.groupby(
            "MODEL"
        )["TARGET_CASE_ID"]
        .nunique()
        .to_dict()
    )

    rows.append(
        {
            "CHECK":
                "equal_case_count_across_models",
            "VALUE":
                len(
                    set(
                        model_counts.values()
                    )
                ) == 1,
            "DETAIL":
                str(model_counts),
        }
    )

    e1 = pd.read_parquet(
        e1_case_results_path,
        columns=[
            "TARGET_CASE_ID",
            "POOL_DEFINITION",
            "METHOD",
            "WINNER_RANK_MID",
        ],
    )

    e1 = e1.loc[
        (e1["POOL_DEFINITION"] == "CPV2_MIN1")
        & (e1["METHOD"] == "TOPSIS"),
        [
            "TARGET_CASE_ID",
            "WINNER_RANK_MID",
        ],
    ].rename(
        columns={
            "WINNER_RANK_MID":
                "E1_TOPSIS_RANK",
        }
    )

    m0 = results.loc[
        results["MODEL"]
        == "M0_ORIGINAL",
        [
            "TARGET_CASE_ID",
            "WINNER_RANK_MID",
        ],
    ].rename(
        columns={
            "WINNER_RANK_MID":
                "E1B_M0_RANK",
        }
    )

    merged = e1.merge(
        m0,
        on="TARGET_CASE_ID",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    ids_match = bool(
        (merged["_merge"] == "both").all()
    )

    ranks_match = bool(
        np.allclose(
            merged.loc[
                merged["_merge"] == "both",
                "E1_TOPSIS_RANK",
            ],
            merged.loc[
                merged["_merge"] == "both",
                "E1B_M0_RANK",
            ],
            rtol=0,
            atol=1e-9,
        )
    )

    rows.append(
        {
            "CHECK":
                "m0_case_ids_match_e1_topsis",
            "VALUE":
                ids_match,
            "DETAIL":
                str(
                    merged[
                        "_merge"
                    ].value_counts().to_dict()
                ),
        }
    )

    rows.append(
        {
            "CHECK":
                "m0_winner_ranks_match_e1_topsis",
            "VALUE":
                ranks_match,
            "DETAIL":
                (
                    f"matched_cases="
                    f"{int((merged['_merge'] == 'both').sum())}"
                ),
        }
    )

    return pd.DataFrame(rows)


def write_report(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# E1b Nested Procurement-Fit TOPSIS Report",
        "",
        "## Scope",
        "",
        "E1b compares four nested TOPSIS specifications on the exact same "
        "CPV2_MIN1 winner-in-pool cohort used in E1. All weights are "
        "specified a priori at the conceptual-group level and are not "
        "estimated from 2017 outcomes.",
        "",
        "## Model performance",
        "",
        "| Model | N | W@1 (%) | W@5 (%) | W@10 (%) | W@50 (%) | MRR | Mean percentile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"| `{row['MODEL']}` "
            f"| {int(row['N_CASES'])} "
            f"| {row['WINNER_AT_1_PCT']:.3f} "
            f"| {row['WINNER_AT_5_PCT']:.3f} "
            f"| {row['WINNER_AT_10_PCT']:.3f} "
            f"| {row['WINNER_AT_50_PCT']:.3f} "
            f"| {row['MRR']:.6f} "
            f"| {row['MEAN_PERCENTILE_RANK']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Change relative to M0",
            "",
            "| Model | ΔW@10 (pp) | ΔW@50 (pp) | ΔMRR | % winner moved up | % unchanged | % moved down |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in comparison_df.iterrows():
        lines.append(
            f"| `{row['COMPARISON_MODEL']}` "
            f"| {row['DELTA_WINNER_AT_10_PP']:.3f} "
            f"| {row['DELTA_WINNER_AT_50_PP']:.3f} "
            f"| {row['DELTA_MRR']:.6f} "
            f"| {row['PCT_WINNER_MOVED_UP']:.3f} "
            f"| {row['PCT_WINNER_UNCHANGED']:.3f} "
            f"| {row['PCT_WINNER_MOVED_DOWN']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
        ]
    )

    for _, row in validation_df.iterrows():
        status = (
            "PASS"
            if bool(row["VALUE"])
            else "FAIL"
        )

        lines.append(
            f"- **{status}** `{row['CHECK']}`: "
            f"{row['DETAIL']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation caution",
            "",
            "The observed winner is used as a reference outcome rather than "
            "proof of a uniquely optimal supplier. Improvements over M0 indicate "
            "stronger alignment between procurement-specific historical fit and "
            "observed award outcomes within the same candidate cohort.",
            "",
            "Prior-buyer fit captures historical contracting relationships and "
            "must not be interpreted as supplier quality. If M2 or M3 gains are "
            "large, a follow-up no-prior-buyer ablation should be reported.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-profiles",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cpv2-features",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--fit-index-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--e1-case-results",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/e1b_rankings"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "reports/e1b_rankings"
        ),
    )

    args = parser.parse_args()

    for path in [
        args.target_profiles,
        args.cpv2_features,
        args.e1_case_results,
    ]:
        require_file(path)

    require_file(args.fit_index_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1/8 Loading target profiles...")
    targets = load_target_profiles(
        args.target_profiles
    )

    print(
        f"    cases: {len(targets):,}"
    )

    print("2/8 Loading CPV2 candidate features...")
    cpv2 = load_cpv2_features(
        args.cpv2_features
    )

    print(
        f"    candidate-context rows: {len(cpv2):,}"
    )

    print("3/8 Building supplier universe...")
    (
        supplier_array,
        supplier_to_idx,
    ) = create_supplier_universe(cpv2)

    print(
        f"    suppliers: {len(supplier_array):,}"
    )

    print("4/8 Loading compact fit indices...")
    sparse_indices = load_sparse_fit_indices(
        args.fit_index_dir,
        supplier_to_idx,
    )

    print("5/8 Running nested TOPSIS models...")
    results = run_nested_models(
        targets=targets,
        cpv2=cpv2,
        supplier_to_idx=supplier_to_idx,
        sparse_indices=sparse_indices,
    )

    print("6/8 Summarizing models...")
    spec_df = model_specification()
    summary_df = summarize_models(results)
    comparison_df = compare_vs_m0(
        results,
        summary_df,
    )
    by_cpv2_df = summarize_by_cpv2(
        results
    )

    print("7/8 Running validation...")
    validation_df = validation_checks(
        results,
        args.e1_case_results,
    )

    print("8/8 Writing outputs...")
    results.to_parquet(
        args.output_dir
        / "e1b_case_model_results.parquet",
        index=False,
    )

    spec_df.to_csv(
        args.report_dir
        / "e1b_model_specification.csv",
        index=False,
    )

    summary_df.to_csv(
        args.report_dir
        / "e1b_model_summary.csv",
        index=False,
    )

    comparison_df.to_csv(
        args.report_dir
        / "e1b_model_comparison_vs_m0.csv",
        index=False,
    )

    by_cpv2_df.to_csv(
        args.report_dir
        / "e1b_results_by_cpv2.csv",
        index=False,
    )

    validation_df.to_csv(
        args.report_dir
        / "e1b_validation_checks.csv",
        index=False,
    )

    write_report(
        summary_df,
        comparison_df,
        validation_df,
        args.report_dir
        / "e1b_ranking_report.md",
    )

    print()
    print("E1b nested ranking comparison complete.")
    print("Review first:")
    print("  e1b_ranking_report.md")
    print("  e1b_model_summary.csv")
    print("  e1b_model_comparison_vs_m0.csv")
    print("  e1b_validation_checks.csv")


if __name__ == "__main__":
    main()
