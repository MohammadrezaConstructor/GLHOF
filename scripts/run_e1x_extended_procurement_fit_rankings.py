#!/usr/bin/env python3
"""
E1x: extended nested procurement-fit TOPSIS ranking experiment.

Purpose
-------
Evaluate richer procurement-specific feature families on the exact same
CPV2_MIN1 winner-in-pool cohort as E1, while preserving deterministic
candidate generation and deterministic TOPSIS ranking.

Model sequence
--------------
M0_ORIGINAL
    Exact E1 TOPSIS baseline:
    - CPV2 context experience
    - buyer breadth
    - public-activity recency
    - context specialization

M1_CATEGORY_SET
    Group-balanced model:
    - category fit:
      CPV2 experience, exact CPV-set fit, CPV4-set fit, CPV3-set fit
    - activity profile:
      buyer breadth, public recency, context specialization

M2_GEO
    M1 plus geographic delivery-market fit:
    - procurement-country experience
    - NUTS1 fit
    - NUTS2 fit
    - NUTS3 fit

M2_BUYER
    M1 plus buyer-relationship dynamics:
    - prior-buyer count
    - prior-buyer recency
    - buyer relationship award share

M3_GEO_BUYER
    M1 plus both geographic fit and buyer-relationship dynamics.

M4_SELECTED_REGIME
    M3 plus selected regime-match share using:
    - B_FRA_AGREEMENT
    - B_GPA
    - B_EU_FUNDS
    - B_ELECTRONIC_AUCTION
    - B_ON_BEHALF

    Excluded from the main model:
    - B_DYN_PURCH_SYST: no positive target cases in the audited primary cohort
    - B_ACCELERATED: only 2.1% target coverage and only positive observed values
    - joint procurement / central body: sparse historical observation coverage

M5_STRUCTURED_CRITERIA
    M4 plus:
    - criterion-code historical fit
    - price-weight similarity where target and supplier historical values exist

Method
------
- Same candidate set for all models in a case.
- TOPSIS only.
- Positive/benefit criteria.
- Count features use log(1 + count).
- Target CPV/NUTS set features are mean log-count support over the target set.
- Conceptual groups receive equal weight.
- Active features within a group receive equal weight.
- Missing target attributes deactivate the corresponding feature.
- Missing historical evidence for a supplier gives zero support for count/share
  fit features; target-side missingness deactivates a criterion.

Efficiency
----------
The script never sorts full candidate rankings. It computes the observed
winner's best/mid/worst rank directly from rounded TOPSIS scores.

Outputs
-------
Processed:
- e1x_case_model_results.parquet

Reports:
- e1x_model_specification.csv
- e1x_model_summary.csv
- e1x_model_comparison_vs_m0.csv
- e1x_incremental_comparison.csv
- e1x_buyer_geo_decomposition.csv
- e1x_results_by_cpv2.csv
- e1x_validation_checks.csv
- e1x_ranking_report.md

Requires
--------
pip install numpy pandas pyarrow
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Feature and model definitions
# ---------------------------------------------------------------------

FEATURES = [
    # Original E1 features
    "CPV2_EXPERIENCE",
    "BUYER_BREADTH",
    "PUBLIC_ACTIVITY_RECENCY",
    "CONTEXT_SPECIALIZATION",

    # Extended category-set fit
    "CPV_EXACT_SET_FIT",
    "CPV4_SET_FIT",
    "CPV3_SET_FIT",

    # Geography
    "COUNTRY_FIT",
    "NUTS1_FIT",
    "NUTS2_FIT",
    "NUTS3_FIT",

    # Buyer relationship dynamics
    "PRIOR_BUYER_COUNT",
    "PRIOR_BUYER_RECENCY",
    "BUYER_RELATIONSHIP_SHARE",

    # Regime
    "REGIME_MATCH_SHARE",

    # Structured criteria
    "CRIT_CODE_FIT",
    "PRICE_WEIGHT_SIMILARITY",
]

FEATURE_INDEX = {
    name: i
    for i, name in enumerate(FEATURES)
}


GROUPS = {
    "M0_ORIGINAL_FEATURES": [
        "CPV2_EXPERIENCE",
        "BUYER_BREADTH",
        "PUBLIC_ACTIVITY_RECENCY",
        "CONTEXT_SPECIALIZATION",
    ],

    "CATEGORY_FIT": [
        "CPV2_EXPERIENCE",
        "CPV_EXACT_SET_FIT",
        "CPV4_SET_FIT",
        "CPV3_SET_FIT",
    ],

    "ACTIVITY_PROFILE": [
        "BUYER_BREADTH",
        "PUBLIC_ACTIVITY_RECENCY",
        "CONTEXT_SPECIALIZATION",
    ],

    "GEOGRAPHIC_FIT": [
        "COUNTRY_FIT",
        "NUTS1_FIT",
        "NUTS2_FIT",
        "NUTS3_FIT",
    ],

    "BUYER_RELATIONSHIP": [
        "PRIOR_BUYER_COUNT",
        "PRIOR_BUYER_RECENCY",
        "BUYER_RELATIONSHIP_SHARE",
    ],

    "REGIME_FIT": [
        "REGIME_MATCH_SHARE",
    ],

    "STRUCTURED_CRITERIA": [
        "CRIT_CODE_FIT",
        "PRICE_WEIGHT_SIMILARITY",
    ],
}


MODEL_GROUPS = {
    "M0_ORIGINAL": [
        "M0_ORIGINAL_FEATURES",
    ],

    "M1_CATEGORY_SET": [
        "CATEGORY_FIT",
        "ACTIVITY_PROFILE",
    ],

    "M2_GEO": [
        "CATEGORY_FIT",
        "GEOGRAPHIC_FIT",
        "ACTIVITY_PROFILE",
    ],

    "M2_BUYER": [
        "CATEGORY_FIT",
        "BUYER_RELATIONSHIP",
        "ACTIVITY_PROFILE",
    ],

    "M3_GEO_BUYER": [
        "CATEGORY_FIT",
        "GEOGRAPHIC_FIT",
        "BUYER_RELATIONSHIP",
        "ACTIVITY_PROFILE",
    ],

    "M4_SELECTED_REGIME": [
        "CATEGORY_FIT",
        "GEOGRAPHIC_FIT",
        "BUYER_RELATIONSHIP",
        "REGIME_FIT",
        "ACTIVITY_PROFILE",
    ],

    "M5_STRUCTURED_CRITERIA": [
        "CATEGORY_FIT",
        "GEOGRAPHIC_FIT",
        "BUYER_RELATIONSHIP",
        "REGIME_FIT",
        "STRUCTURED_CRITERIA",
        "ACTIVITY_PROFILE",
    ],
}


MODEL_ORDER = {
    model: idx
    for idx, model in enumerate(MODEL_GROUPS)
}


SELECTED_REGIME_FIELDS = [
    "B_FRA_AGREEMENT",
    "B_GPA",
    "B_EU_FUNDS",
    "B_ELECTRONIC_AUCTION",
    "B_ON_BEHALF",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def normalize_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def normalize_bool_to_index_value(
    value: Any,
) -> str | None:
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"

    text = str(value).strip().lower()

    if text in {"true", "t", "1", "y", "yes"}:
        return "true"

    if text in {"false", "f", "0", "n", "no"}:
        return "false"

    return None


def direct_winner_rank(
    scores: np.ndarray,
    winner_idx: int,
    decimals: int = 12,
) -> tuple[float, int, int, int]:
    """
    Higher score is better.

    Returns:
      midrank, best_rank, worst_rank, tie_size
    """
    rounded = np.round(
        np.asarray(scores, dtype=np.float64),
        decimals=decimals,
    )

    winner_score = rounded[winner_idx]

    n_better = int(
        np.sum(rounded > winner_score)
    )

    n_equal = int(
        np.sum(rounded == winner_score)
    )

    best = n_better + 1
    worst = n_better + n_equal
    mid = (best + worst) / 2.0

    return (
        float(mid),
        int(best),
        int(worst),
        int(n_equal),
    )


def safe_percentile(
    midrank: float,
    pool_size: int,
) -> float:
    if pool_size <= 1:
        return 1.0

    return 1.0 - (
        (midrank - 1.0)
        / (pool_size - 1.0)
    )


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def load_targets(
    extended_profile_path: Path,
    target_case_features_path: Path,
) -> pd.DataFrame:
    profile = pd.read_parquet(
        extended_profile_path
    )

    outcome = pd.read_parquet(
        target_case_features_path,
        columns=[
            "TARGET_CASE_ID",
            "TARGET_CPV2",
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        ],
    ).rename(
        columns={
            "TARGET_CPV2": "E1_TARGET_CPV2",
        }
    )

    profile["TARGET_CASE_ID"] = (
        profile["TARGET_CASE_ID"]
        .astype("string")
    )

    outcome["TARGET_CASE_ID"] = (
        outcome["TARGET_CASE_ID"]
        .astype("string")
    )

    outcome[
        "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"
    ] = (
        outcome[
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"
        ]
        .astype("string")
    )

    outcome["E1_TARGET_CPV2"] = (
        outcome["E1_TARGET_CPV2"]
        .astype("string")
    )

    df = profile.merge(
        outcome,
        on="TARGET_CASE_ID",
        how="inner",
        validate="one_to_one",
    )

    df["TARGET_DISPATCH_DATE"] = pd.to_datetime(
        df["TARGET_DISPATCH_DATE"],
        errors="raise",
    )

    profile_cpv2 = df["TARGET_CPV2"].astype("string")
    e1_cpv2 = df["E1_TARGET_CPV2"].astype("string")

    mismatch = (
        profile_cpv2.notna()
        & e1_cpv2.notna()
        & (profile_cpv2 != e1_cpv2)
    )

    if mismatch.any():
        raise ValueError(
            f"Extended target profile CPV2 disagrees with the frozen E1 "
            f"target CPV2 for {int(mismatch.sum())} cases."
        )

    # Use the frozen E1 target context as the candidate-pool key.
    df["TARGET_CPV2"] = e1_cpv2
    df = df.drop(columns=["E1_TARGET_CPV2"])

    for col in [
        "TARGET_CPV2",
        "TARGET_CPV3",
        "TARGET_CPV4",
        "TARGET_MAIN_CPV",
        "TARGET_BUYER_KEY",
        "TARGET_PROCUREMENT_COUNTRY",
        "TARGET_CRIT_CODE",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def load_token_map(
    path: Path,
    level_columns: list[str],
) -> dict[str, dict[str, tuple[str, ...]]]:
    df = pd.read_parquet(path)

    df["TARGET_CASE_ID"] = (
        df["TARGET_CASE_ID"]
        .astype("string")
    )

    result: dict[
        str,
        dict[str, tuple[str, ...]]
    ] = {}

    for case_id, g in df.groupby(
        "TARGET_CASE_ID",
        sort=False,
        observed=True,
    ):
        level_map = {}

        for col in level_columns:
            values = tuple(
                sorted(
                    {
                        str(v)
                        for v in g[col].dropna().tolist()
                        if str(v).strip()
                    }
                )
            )

            level_map[col] = values

        result[str(case_id)] = level_map

    return result


def load_cpv2_candidates(
    cpv2_features_path: Path,
    cpv2_descriptors_path: Path,
) -> pd.DataFrame:
    base_cols = [
        "CONTEXT_KEY",
        "SUPPLIER_ENTITY_ID",
        "N_CONTEXT_AWARDS",
        "N_UNIQUE_BUYERS_IN_CONTEXT",
        "LAST_CONTEXT_PUBLICATION_DATE",
        "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT",
    ]

    base = pd.read_parquet(
        cpv2_features_path,
        columns=base_cols,
    )

    base = base.loc[
        base["N_CONTEXT_AWARDS"] >= 1
    ].copy()

    descriptors = pd.read_parquet(
        cpv2_descriptors_path,
        columns=[
            "CONTEXT_KEY",
            "SUPPLIER_ENTITY_ID",
            "MEDIAN_CRIT_PRICE_WEIGHT",
            "N_CRIT_PRICE_WEIGHT_OBS",
        ],
    )

    for df in [base, descriptors]:
        df["CONTEXT_KEY"] = (
            df["CONTEXT_KEY"]
            .astype("string")
        )
        df["SUPPLIER_ENTITY_ID"] = (
            df["SUPPLIER_ENTITY_ID"]
            .astype("string")
        )

    base["LAST_CONTEXT_PUBLICATION_DATE"] = (
        pd.to_datetime(
            base["LAST_CONTEXT_PUBLICATION_DATE"],
            errors="raise",
        )
    )

    out = base.merge(
        descriptors,
        on=[
            "CONTEXT_KEY",
            "SUPPLIER_ENTITY_ID",
        ],
        how="left",
        validate="one_to_one",
    )

    dup = out.duplicated(
        [
            "CONTEXT_KEY",
            "SUPPLIER_ENTITY_ID",
        ],
        keep=False,
    )

    if dup.any():
        raise ValueError(
            f"Candidate table contains {int(dup.sum())} "
            "duplicated context-supplier rows."
        )

    return out


def create_supplier_universe(
    candidates: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series]:
    supplier_ids = pd.Index(
        candidates[
            "SUPPLIER_ENTITY_ID"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    supplier_array = supplier_ids.to_numpy(
        dtype=object
    )

    supplier_to_idx = pd.Series(
        np.arange(
            len(supplier_array),
            dtype=np.int32,
        ),
        index=supplier_ids,
    )

    return supplier_array, supplier_to_idx


# ---------------------------------------------------------------------
# Sparse index loaders
# ---------------------------------------------------------------------

SparseLookup = dict[
    Any,
    tuple[np.ndarray, np.ndarray]
]


def map_supplier_indices(
    supplier_ids: pd.Series,
    supplier_to_idx: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    mapped = (
        supplier_ids.astype(str)
        .map(supplier_to_idx)
    )

    keep = mapped.notna().to_numpy()

    indices = (
        mapped.loc[keep]
        .astype(np.int32)
        .to_numpy()
    )

    return keep, indices


def load_hierarchical_sparse_index(
    path: Path,
    level_col: str,
    needed_levels: set[str],
    supplier_to_idx: pd.Series,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    print(
        f"    loading {path.name}..."
    )

    df = pd.read_parquet(
        path,
        columns=[
            level_col,
            "FIT_VALUE",
            "SUPPLIER_ENTITY_ID",
            "N_MATCHED_AWARDS",
        ],
    )

    df = df.loc[
        df[level_col].isin(needed_levels)
    ].copy()

    keep, mapped = map_supplier_indices(
        df["SUPPLIER_ENTITY_ID"],
        supplier_to_idx,
    )

    df = df.loc[keep].copy()
    df["SUPPLIER_IDX"] = mapped

    lookup: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray]
    ] = {}

    for key, g in df.groupby(
        [level_col, "FIT_VALUE"],
        sort=False,
        observed=True,
    ):
        level, value = key

        lookup[
            (str(level), str(value))
        ] = (
            g["SUPPLIER_IDX"].to_numpy(
                dtype=np.int32,
                copy=True,
            ),
            g["N_MATCHED_AWARDS"].to_numpy(
                dtype=np.float64,
                copy=True,
            ),
        )

    del df
    gc.collect()

    return lookup


def load_simple_count_lookup(
    path: Path,
    count_col: str,
    supplier_to_idx: pd.Series,
) -> SparseLookup:
    print(
        f"    loading {path.name}..."
    )

    df = pd.read_parquet(
        path,
        columns=[
            "FIT_VALUE",
            "SUPPLIER_ENTITY_ID",
            count_col,
        ],
    )

    keep, mapped = map_supplier_indices(
        df["SUPPLIER_ENTITY_ID"],
        supplier_to_idx,
    )

    df = df.loc[keep].copy()
    df["SUPPLIER_IDX"] = mapped

    lookup: SparseLookup = {}

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
            g[count_col].to_numpy(
                dtype=np.float64,
                copy=True,
            ),
        )

    del df
    gc.collect()

    return lookup


def load_buyer_lookup(
    path: Path,
    supplier_to_idx: pd.Series,
) -> dict[
    str,
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:
    print(
        f"    loading {path.name}..."
    )

    df = pd.read_parquet(
        path,
        columns=[
            "BUYER_KEY",
            "SUPPLIER_ENTITY_ID",
            "N_PRIOR_BUYER_AWARDS",
            "LAST_PRIOR_BUYER_AWARD_DATE",
            "BUYER_RELATIONSHIP_AWARD_SHARE",
        ],
    )

    keep, mapped = map_supplier_indices(
        df["SUPPLIER_ENTITY_ID"],
        supplier_to_idx,
    )

    df = df.loc[keep].copy()
    df["SUPPLIER_IDX"] = mapped

    df["LAST_PRIOR_BUYER_AWARD_DATE"] = (
        pd.to_datetime(
            df["LAST_PRIOR_BUYER_AWARD_DATE"],
            errors="coerce",
        )
    )

    lookup = {}

    for buyer_key, g in df.groupby(
        "BUYER_KEY",
        sort=False,
        observed=True,
    ):
        lookup[str(buyer_key)] = (
            g["SUPPLIER_IDX"].to_numpy(
                dtype=np.int32,
                copy=True,
            ),
            g["N_PRIOR_BUYER_AWARDS"].to_numpy(
                dtype=np.float64,
                copy=True,
            ),
            g[
                "LAST_PRIOR_BUYER_AWARD_DATE"
            ].to_numpy(
                dtype="datetime64[ns]",
                copy=True,
            ),
            g[
                "BUYER_RELATIONSHIP_AWARD_SHARE"
            ].fillna(0.0).to_numpy(
                dtype=np.float64,
                copy=True,
            ),
        )

    del df
    gc.collect()

    return lookup


def load_regime_lookup(
    path: Path,
    supplier_to_idx: pd.Series,
) -> tuple[
    dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray]
    ],
    dict[
        str,
        tuple[np.ndarray, np.ndarray]
    ],
]:
    print(
        f"    loading {path.name}..."
    )

    df = pd.read_parquet(
        path,
        columns=[
            "DIMENSION",
            "FIT_VALUE",
            "SUPPLIER_ENTITY_ID",
            "N_MATCHED_AWARDS",
        ],
    )

    df = df.loc[
        df["DIMENSION"].isin(
            SELECTED_REGIME_FIELDS
        )
    ].copy()

    keep, mapped = map_supplier_indices(
        df["SUPPLIER_ENTITY_ID"],
        supplier_to_idx,
    )

    df = df.loc[keep].copy()
    df["SUPPLIER_IDX"] = mapped

    match_lookup = {}

    for key, g in df.groupby(
        ["DIMENSION", "FIT_VALUE"],
        sort=False,
        observed=True,
    ):
        dim, value = key

        match_lookup[
            (
                str(dim),
                str(value).lower(),
            )
        ] = (
            g["SUPPLIER_IDX"].to_numpy(
                dtype=np.int32,
                copy=True,
            ),
            g["N_MATCHED_AWARDS"].to_numpy(
                dtype=np.float64,
                copy=True,
            ),
        )

    totals = (
        df.groupby(
            [
                "DIMENSION",
                "SUPPLIER_IDX",
            ],
            observed=True,
            sort=False,
        )["N_MATCHED_AWARDS"]
        .sum()
        .reset_index()
    )

    total_lookup = {}

    for dim, g in totals.groupby(
        "DIMENSION",
        sort=False,
        observed=True,
    ):
        total_lookup[str(dim)] = (
            g["SUPPLIER_IDX"].to_numpy(
                dtype=np.int32,
                copy=True,
            ),
            g["N_MATCHED_AWARDS"].to_numpy(
                dtype=np.float64,
                copy=True,
            ),
        )

    del df, totals
    gc.collect()

    return match_lookup, total_lookup


# ---------------------------------------------------------------------
# Sparse-local vector helpers
# ---------------------------------------------------------------------

def sparse_to_local_dense(
    source: tuple[np.ndarray, np.ndarray] | None,
    n_candidates: int,
    global_to_local: np.ndarray,
    transform_log1p: bool = False,
) -> np.ndarray:
    values = np.zeros(
        n_candidates,
        dtype=np.float64,
    )

    if source is None:
        return values

    global_idx, source_values = source

    local_pos = global_to_local[
        global_idx
    ]

    keep = local_pos >= 0

    if not np.any(keep):
        return values

    local = local_pos[keep]
    vals = source_values[keep].astype(
        np.float64,
        copy=False,
    )

    if transform_log1p:
        vals = np.log1p(vals)

    values[local] = vals

    return values


def mean_log_support_over_target_values(
    lookup: SparseLookup,
    keys: list[Any],
    n_candidates: int,
    global_to_local: np.ndarray,
) -> tuple[np.ndarray, bool]:
    unique_keys = list(
        dict.fromkeys(keys)
    )

    if not unique_keys:
        return (
            np.zeros(
                n_candidates,
                dtype=np.float64,
            ),
            False,
        )

    total = np.zeros(
        n_candidates,
        dtype=np.float64,
    )

    for key in unique_keys:
        total += sparse_to_local_dense(
            lookup.get(key),
            n_candidates=n_candidates,
            global_to_local=global_to_local,
            transform_log1p=True,
        )

    total /= float(
        len(unique_keys)
    )

    return total, True


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

def make_case_features(
    candidates: pd.DataFrame,
    target_row: pd.Series,
    cpv_levels: dict[str, tuple[str, ...]],
    nuts_levels: dict[str, tuple[str, ...]],
    global_to_local: np.ndarray,

    cpv_lookup: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray]
    ],

    nuts_lookup: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray]
    ],

    country_lookup: SparseLookup,

    buyer_lookup: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],

    regime_match_lookup: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray]
    ],

    regime_total_lookup: dict[
        str,
        tuple[np.ndarray, np.ndarray]
    ],

    criterion_lookup: SparseLookup,

) -> tuple[np.ndarray, np.ndarray]:
    n = len(candidates)

    arrays: dict[str, np.ndarray] = {}
    active: dict[str, bool] = {}

    # --------------------------------------------------------------
    # Original E1 features
    # --------------------------------------------------------------
    target_date = pd.Timestamp(
        target_row["TARGET_DISPATCH_DATE"]
    )

    last_pub = candidates[
        "LAST_CONTEXT_PUBLICATION_DATE"
    ]

    days = (
        target_date.normalize()
        - last_pub.dt.normalize()
    ).dt.days.to_numpy(
        dtype=np.int64
    )

    if np.any(days < 0):
        raise ValueError(
            f"Found {int(np.sum(days < 0))} candidate rows "
            "with publication date after target date."
        )

    arrays["CPV2_EXPERIENCE"] = np.log1p(
        candidates[
            "N_CONTEXT_AWARDS"
        ].to_numpy(dtype=np.float64)
    )
    active["CPV2_EXPERIENCE"] = True

    arrays["BUYER_BREADTH"] = np.log1p(
        candidates[
            "N_UNIQUE_BUYERS_IN_CONTEXT"
        ]
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    active["BUYER_BREADTH"] = True

    arrays[
        "PUBLIC_ACTIVITY_RECENCY"
    ] = (
        1.0
        / (
            1.0
            + days.astype(np.float64)
        )
    )
    active[
        "PUBLIC_ACTIVITY_RECENCY"
    ] = True

    arrays[
        "CONTEXT_SPECIALIZATION"
    ] = (
        candidates[
            "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT"
        ]
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    active[
        "CONTEXT_SPECIALIZATION"
    ] = True

    # --------------------------------------------------------------
    # CPV set fit
    # --------------------------------------------------------------
    exact_keys = [
        ("EXACT", v)
        for v in cpv_levels.get(
            "CPV_CODE",
            (),
        )
    ]

    cpv4_keys = [
        ("CPV4", v)
        for v in cpv_levels.get(
            "CPV4",
            (),
        )
    ]

    cpv3_keys = [
        ("CPV3", v)
        for v in cpv_levels.get(
            "CPV3",
            (),
        )
    ]

    (
        arrays["CPV_EXACT_SET_FIT"],
        active["CPV_EXACT_SET_FIT"],
    ) = mean_log_support_over_target_values(
        cpv_lookup,
        exact_keys,
        n,
        global_to_local,
    )

    (
        arrays["CPV4_SET_FIT"],
        active["CPV4_SET_FIT"],
    ) = mean_log_support_over_target_values(
        cpv_lookup,
        cpv4_keys,
        n,
        global_to_local,
    )

    (
        arrays["CPV3_SET_FIT"],
        active["CPV3_SET_FIT"],
    ) = mean_log_support_over_target_values(
        cpv_lookup,
        cpv3_keys,
        n,
        global_to_local,
    )

    # --------------------------------------------------------------
    # Geography
    # --------------------------------------------------------------
    country = normalize_text(
        target_row.get(
            "TARGET_PROCUREMENT_COUNTRY"
        )
    )

    if country is None:
        arrays["COUNTRY_FIT"] = np.zeros(
            n,
            dtype=np.float64,
        )
        active["COUNTRY_FIT"] = False
    else:
        arrays["COUNTRY_FIT"] = (
            sparse_to_local_dense(
                country_lookup.get(country),
                n,
                global_to_local,
                transform_log1p=True,
            )
        )
        active["COUNTRY_FIT"] = True

    for feature_name, nuts_level in [
        ("NUTS1_FIT", "NUTS1"),
        ("NUTS2_FIT", "NUTS2"),
        ("NUTS3_FIT", "NUTS3"),
    ]:
        keys = [
            (nuts_level, v)
            for v in nuts_levels.get(
                nuts_level,
                (),
            )
        ]

        (
            arrays[feature_name],
            active[feature_name],
        ) = mean_log_support_over_target_values(
            nuts_lookup,
            keys,
            n,
            global_to_local,
        )

    # --------------------------------------------------------------
    # Buyer relationship dynamics
    # --------------------------------------------------------------
    buyer_key = normalize_text(
        target_row.get(
            "TARGET_BUYER_KEY"
        )
    )

    buyer_count = np.zeros(
        n,
        dtype=np.float64,
    )
    buyer_recency = np.zeros(
        n,
        dtype=np.float64,
    )
    buyer_share = np.zeros(
        n,
        dtype=np.float64,
    )

    buyer_is_active = (
        buyer_key is not None
    )

    if buyer_is_active:
        source = buyer_lookup.get(
            buyer_key
        )

        if source is not None:
            (
                global_idx,
                counts,
                last_dates,
                shares,
            ) = source

            local_pos = global_to_local[
                global_idx
            ]

            keep = local_pos >= 0

            local = local_pos[keep]

            buyer_count[local] = np.log1p(
                counts[keep]
            )

            buyer_share[local] = np.clip(
                shares[keep],
                0.0,
                1.0,
            )

            candidate_dates = pd.to_datetime(
                last_dates[keep],
                errors="coerce",
            )

            valid_dates = ~pd.isna(
                candidate_dates
            )

            if np.any(valid_dates):
                delta_days = (
                    target_date.normalize()
                    - candidate_dates[
                        valid_dates
                    ].normalize()
                ).days

                delta_days = np.asarray(
                    delta_days,
                    dtype=np.float64,
                )

                valid_nonnegative = (
                    delta_days >= 0
                )

                valid_local = local[
                    valid_dates
                ]

                buyer_recency[
                    valid_local[
                        valid_nonnegative
                    ]
                ] = (
                    1.0
                    / (
                        1.0
                        + delta_days[
                            valid_nonnegative
                        ]
                    )
                )

    arrays[
        "PRIOR_BUYER_COUNT"
    ] = buyer_count
    arrays[
        "PRIOR_BUYER_RECENCY"
    ] = buyer_recency
    arrays[
        "BUYER_RELATIONSHIP_SHARE"
    ] = buyer_share

    active[
        "PRIOR_BUYER_COUNT"
    ] = buyer_is_active
    active[
        "PRIOR_BUYER_RECENCY"
    ] = buyer_is_active
    active[
        "BUYER_RELATIONSHIP_SHARE"
    ] = buyer_is_active

    # --------------------------------------------------------------
    # Selected regime match share
    # --------------------------------------------------------------
    regime_sum = np.zeros(
        n,
        dtype=np.float64,
    )

    n_active_regime_dims = 0

    for dim in SELECTED_REGIME_FIELDS:
        target_col = f"TARGET_{dim}"

        target_value = normalize_bool_to_index_value(
            target_row.get(target_col)
        )

        if target_value is None:
            continue

        match = sparse_to_local_dense(
            regime_match_lookup.get(
                (
                    dim,
                    target_value,
                )
            ),
            n,
            global_to_local,
            transform_log1p=False,
        )

        total = sparse_to_local_dense(
            regime_total_lookup.get(dim),
            n,
            global_to_local,
            transform_log1p=False,
        )

        share = np.zeros(
            n,
            dtype=np.float64,
        )

        valid_total = total > 0

        share[valid_total] = (
            match[valid_total]
            / total[valid_total]
        )

        regime_sum += share
        n_active_regime_dims += 1

    if n_active_regime_dims > 0:
        regime_sum /= float(
            n_active_regime_dims
        )
        active[
            "REGIME_MATCH_SHARE"
        ] = True
    else:
        active[
            "REGIME_MATCH_SHARE"
        ] = False

    arrays[
        "REGIME_MATCH_SHARE"
    ] = regime_sum

    # --------------------------------------------------------------
    # Structured criteria
    # --------------------------------------------------------------
    crit_code = normalize_text(
        target_row.get(
            "TARGET_CRIT_CODE"
        )
    )

    if crit_code is None:
        arrays["CRIT_CODE_FIT"] = (
            np.zeros(
                n,
                dtype=np.float64,
            )
        )
        active["CRIT_CODE_FIT"] = False
    else:
        arrays["CRIT_CODE_FIT"] = (
            sparse_to_local_dense(
                criterion_lookup.get(
                    crit_code.upper()
                ),
                n,
                global_to_local,
                transform_log1p=True,
            )
        )
        active["CRIT_CODE_FIT"] = True

    target_pw = target_row.get(
        "TARGET_CRIT_PRICE_WEIGHT"
    )

    if pd.isna(target_pw):
        arrays[
            "PRICE_WEIGHT_SIMILARITY"
        ] = np.zeros(
            n,
            dtype=np.float64,
        )
        active[
            "PRICE_WEIGHT_SIMILARITY"
        ] = False
    else:
        target_pw = float(
            target_pw
        )

        historical_pw = (
            candidates[
                "MEDIAN_CRIT_PRICE_WEIGHT"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        similarity = np.zeros(
            n,
            dtype=np.float64,
        )

        valid = np.isfinite(
            historical_pw
        )

        similarity[valid] = (
            1.0
            - np.minimum(
                np.abs(
                    historical_pw[valid]
                    - target_pw
                )
                / 100.0,
                1.0,
            )
        )

        arrays[
            "PRICE_WEIGHT_SIMILARITY"
        ] = similarity
        active[
            "PRICE_WEIGHT_SIMILARITY"
        ] = True

    # --------------------------------------------------------------
    # Matrix
    # --------------------------------------------------------------
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

    if not np.isfinite(
        matrix
    ).all():
        raise ValueError(
            "Feature matrix contains non-finite values."
        )

    return (
        matrix,
        active_mask,
    )


# ---------------------------------------------------------------------
# Group-balanced TOPSIS
# ---------------------------------------------------------------------

def normalize_feature_matrix(
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

    normalized[
        :,
        nonzero,
    ] = (
        x[:, nonzero]
        / norms[nonzero]
    )

    return (
        normalized,
        nonzero,
    )


def group_distance_contributions(
    normalized: np.ndarray,
    target_active_mask: np.ndarray,
    nonzero_mask: np.ndarray,
) -> dict[
    str,
    tuple[
        np.ndarray,
        np.ndarray,
        int,
    ]
]:
    """
    For each conceptual group, precompute within-group equally weighted
    squared-distance contributions to ideal best and ideal worst.
    """
    contributions = {}

    for group_name, feature_names in GROUPS.items():
        idx = np.array(
            [
                FEATURE_INDEX[name]
                for name in feature_names
            ],
            dtype=np.int64,
        )

        usable = (
            target_active_mask[idx]
            & nonzero_mask[idx]
        )

        active_idx = idx[usable]

        if len(active_idx) == 0:
            contributions[group_name] = (
                np.zeros(
                    normalized.shape[0],
                    dtype=np.float64,
                ),
                np.zeros(
                    normalized.shape[0],
                    dtype=np.float64,
                ),
                0,
            )
            continue

        z = normalized[
            :,
            active_idx,
        ]

        ideal_best = np.max(
            z,
            axis=0,
        )
        ideal_worst = np.min(
            z,
            axis=0,
        )

        m = float(
            len(active_idx)
        )

        d_best_sq = (
            np.sum(
                np.square(
                    z - ideal_best
                ),
                axis=1,
            )
            / (m * m)
        )

        d_worst_sq = (
            np.sum(
                np.square(
                    z - ideal_worst
                ),
                axis=1,
            )
            / (m * m)
        )

        contributions[group_name] = (
            d_best_sq,
            d_worst_sq,
            len(active_idx),
        )

    return contributions


def topsis_scores_from_groups(
    contributions: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
            int,
        ]
    ],
    model_group_names: list[str],
) -> np.ndarray:
    active_groups = [
        group_name
        for group_name in model_group_names
        if contributions[
            group_name
        ][2] > 0
    ]

    if not active_groups:
        first = next(
            iter(contributions.values())
        )
        return np.full(
            len(first[0]),
            0.5,
            dtype=np.float64,
        )

    n_groups = float(
        len(active_groups)
    )

    group_weight_sq = (
        1.0 / n_groups
    ) ** 2

    d_best_sq = np.zeros_like(
        contributions[
            active_groups[0]
        ][0]
    )

    d_worst_sq = np.zeros_like(
        d_best_sq
    )

    for group_name in active_groups:
        best_sq, worst_sq, _ = (
            contributions[group_name]
        )

        d_best_sq += (
            group_weight_sq
            * best_sq
        )

        d_worst_sq += (
            group_weight_sq
            * worst_sq
        )

    d_best = np.sqrt(
        d_best_sq
    )

    d_worst = np.sqrt(
        d_worst_sq
    )

    denom = (
        d_best
        + d_worst
    )

    scores = np.full(
        len(d_best),
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
# Ranking experiment
# ---------------------------------------------------------------------

def run_experiment(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    cpv_token_map: dict[
        str,
        dict[str, tuple[str, ...]]
    ],
    nuts_token_map: dict[
        str,
        dict[str, tuple[str, ...]]
    ],
    supplier_to_idx: pd.Series,
    cpv_lookup,
    nuts_lookup,
    country_lookup,
    buyer_lookup,
    regime_match_lookup,
    regime_total_lookup,
    criterion_lookup,
) -> pd.DataFrame:
    candidate_groups = {
        str(context): g.reset_index(
            drop=True
        )
        for context, g in candidates.groupby(
            "CONTEXT_KEY",
            sort=False,
            observed=True,
        )
    }

    target_groups = list(
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

    records = []

    total_contexts = len(
        target_groups
    )

    for context_no, (
        context_key,
        target_context,
    ) in enumerate(
        target_groups,
        start=1,
    ):
        context_key = str(
            context_key
        )

        cands = candidate_groups.get(
            context_key
        )

        if cands is None or cands.empty:
            raise ValueError(
                f"Missing candidate context {context_key}."
            )

        candidate_ids = (
            cands[
                "SUPPLIER_ENTITY_ID"
            ]
            .astype(str)
            .to_numpy()
        )

        candidate_global_idx = (
            pd.Series(
                candidate_ids
            )
            .map(
                supplier_to_idx
            )
            .to_numpy(
                dtype=np.int32
            )
        )

        n_candidates = len(
            cands
        )

        global_to_local[
            candidate_global_idx
        ] = np.arange(
            n_candidates,
            dtype=np.int32,
        )

        print(
            f"[{context_no}/{total_contexts}] "
            f"CPV2={context_key}: "
            f"{len(target_context):,} cases, "
            f"{n_candidates:,} candidates"
        )

        for case_no, (_, target_row) in enumerate(
            target_context.iterrows(),
            start=1,
        ):
            case_id = str(
                target_row[
                    "TARGET_CASE_ID"
                ]
            )

            winner_id = str(
                target_row[
                    "OBSERVED_WINNER_HISTORICAL_ENTITY_ID"
                ]
            )

            winner_global = (
                supplier_to_idx.get(
                    winner_id,
                    None,
                )
            )

            if winner_global is None:
                raise ValueError(
                    f"Winner {winner_id} not in supplier universe."
                )

            winner_local = int(
                global_to_local[
                    int(winner_global)
                ]
            )

            if winner_local < 0:
                raise ValueError(
                    f"Winner {winner_id} absent from "
                    f"context {context_key}."
                )

            cpv_levels = cpv_token_map.get(
                case_id,
                {},
            )

            nuts_levels = nuts_token_map.get(
                case_id,
                {},
            )

            (
                x,
                active_mask,
            ) = make_case_features(
                candidates=cands,
                target_row=target_row,
                cpv_levels=cpv_levels,
                nuts_levels=nuts_levels,
                global_to_local=global_to_local,
                cpv_lookup=cpv_lookup,
                nuts_lookup=nuts_lookup,
                country_lookup=country_lookup,
                buyer_lookup=buyer_lookup,
                regime_match_lookup=regime_match_lookup,
                regime_total_lookup=regime_total_lookup,
                criterion_lookup=criterion_lookup,
            )

            (
                normalized,
                nonzero_mask,
            ) = normalize_feature_matrix(
                x
            )

            contributions = (
                group_distance_contributions(
                    normalized,
                    active_mask,
                    nonzero_mask,
                )
            )

            for model_name, model_groups in (
                MODEL_GROUPS.items()
            ):
                scores = topsis_scores_from_groups(
                    contributions,
                    model_groups,
                )

                (
                    midrank,
                    best_rank,
                    worst_rank,
                    tie_size,
                ) = direct_winner_rank(
                    scores,
                    winner_local,
                )

                records.append(
                    {
                        "TARGET_CASE_ID":
                            case_id,
                        "TARGET_CPV2":
                            context_key,
                        "MODEL":
                            model_name,
                        "POOL_DEFINITION":
                            "CPV2_MIN1",
                        "POOL_SIZE":
                            n_candidates,
                        "WINNER_RANK_MID":
                            midrank,
                        "WINNER_RANK_BEST":
                            best_rank,
                        "WINNER_RANK_WORST":
                            worst_rank,
                        "TIE_GROUP_SIZE":
                            tie_size,
                        "RECIPROCAL_RANK":
                            1.0
                            / midrank,
                        "PERCENTILE_RANK":
                            safe_percentile(
                                midrank,
                                n_candidates,
                            ),
                        "WINNER_AT_1":
                            midrank <= 1,
                        "WINNER_AT_5":
                            midrank <= 5,
                        "WINNER_AT_10":
                            midrank <= 10,
                        "WINNER_AT_50":
                            midrank <= 50,
                    }
                )

            if (
                case_no % 250 == 0
                or case_no
                == len(target_context)
            ):
                print(
                    f"    processed "
                    f"{case_no:,}/"
                    f"{len(target_context):,} cases"
                )

        global_to_local[
            candidate_global_idx
        ] = -1

    return pd.DataFrame.from_records(
        records
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def model_specification() -> pd.DataFrame:
    rows = []

    for model_name, group_names in (
        MODEL_GROUPS.items()
    ):
        for group_name in group_names:
            for feature_name in GROUPS[
                group_name
            ]:
                rows.append(
                    {
                        "MODEL":
                            model_name,
                        "CONCEPTUAL_GROUP":
                            group_name,
                        "FEATURE":
                            feature_name,
                        "GROUP_WEIGHT_RULE":
                            "equal_across_active_groups",
                        "WITHIN_GROUP_WEIGHT_RULE":
                            "equal_across_active_features",
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
                "N_CASES":
                    len(g),
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
                    g[
                        "RECIPROCAL_RANK"
                    ].mean(),
                "MEAN_PERCENTILE_RANK":
                    g[
                        "PERCENTILE_RANK"
                    ].mean(),
                "MEDIAN_WINNER_RANK":
                    g[
                        "WINNER_RANK_MID"
                    ].median(),
                "MEAN_TIE_GROUP_SIZE":
                    g[
                        "TIE_GROUP_SIZE"
                    ].mean(),
            }
        )

    out = pd.DataFrame(
        rows
    )

    out["_order"] = (
        out["MODEL"]
        .map(MODEL_ORDER)
    )

    return (
        out.sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )


def rank_movement_comparison(
    results: pd.DataFrame,
    summary_df: pd.DataFrame,
    reference_model: str,
    comparison_models: list[str],
) -> pd.DataFrame:
    pivot = results.pivot(
        index="TARGET_CASE_ID",
        columns="MODEL",
        values="WINNER_RANK_MID",
    )

    summary_index = (
        summary_df
        .set_index("MODEL")
    )

    base = summary_index.loc[
        reference_model
    ]

    rows = []

    for model in comparison_models:
        comp = summary_index.loc[
            model
        ]

        delta = (
            pivot[model]
            - pivot[reference_model]
        )

        rows.append(
            {
                "REFERENCE_MODEL":
                    reference_model,
                "COMPARISON_MODEL":
                    model,
                "N_CASES":
                    len(delta),
                "DELTA_WINNER_AT_1_PP":
                    comp["WINNER_AT_1_PCT"]
                    - base["WINNER_AT_1_PCT"],
                "DELTA_WINNER_AT_5_PP":
                    comp["WINNER_AT_5_PCT"]
                    - base["WINNER_AT_5_PCT"],
                "DELTA_WINNER_AT_10_PP":
                    comp["WINNER_AT_10_PCT"]
                    - base["WINNER_AT_10_PCT"],
                "DELTA_WINNER_AT_50_PP":
                    comp["WINNER_AT_50_PCT"]
                    - base["WINNER_AT_50_PCT"],
                "DELTA_MRR":
                    comp["MRR"]
                    - base["MRR"],
                "MEAN_RANK_CHANGE":
                    float(delta.mean()),
                "MEDIAN_RANK_CHANGE":
                    float(delta.median()),
                "MEAN_ABS_RANK_CHANGE":
                    float(
                        delta.abs().mean()
                    ),
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

    return pd.DataFrame(
        rows
    )


def incremental_comparisons(
    results: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    pairs = [
        (
            "M0_ORIGINAL",
            "M1_CATEGORY_SET",
        ),
        (
            "M1_CATEGORY_SET",
            "M2_GEO",
        ),
        (
            "M1_CATEGORY_SET",
            "M2_BUYER",
        ),
        (
            "M1_CATEGORY_SET",
            "M3_GEO_BUYER",
        ),
        (
            "M3_GEO_BUYER",
            "M4_SELECTED_REGIME",
        ),
        (
            "M4_SELECTED_REGIME",
            "M5_STRUCTURED_CRITERIA",
        ),
    ]

    frames = []

    for reference, comparison in pairs:
        frames.append(
            rank_movement_comparison(
                results,
                summary_df,
                reference,
                [comparison],
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def buyer_geo_decomposition(
    results: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    return rank_movement_comparison(
        results,
        summary_df,
        "M1_CATEGORY_SET",
        [
            "M2_GEO",
            "M2_BUYER",
            "M3_GEO_BUYER",
        ],
    )


def summarize_by_cpv2(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        model,
        cpv2,
    ), g in results.groupby(
        [
            "MODEL",
            "TARGET_CPV2",
        ],
        sort=False,
        observed=True,
    ):
        rows.append(
            {
                "MODEL":
                    model,
                "TARGET_CPV2":
                    cpv2,
                "N_CASES":
                    len(g),
                "WINNER_AT_10_PCT":
                    100.0
                    * g[
                        "WINNER_AT_10"
                    ].mean(),
                "WINNER_AT_50_PCT":
                    100.0
                    * g[
                        "WINNER_AT_50"
                    ].mean(),
                "MRR":
                    g[
                        "RECIPROCAL_RANK"
                    ].mean(),
                "MEAN_PERCENTILE_RANK":
                    g[
                        "PERCENTILE_RANK"
                    ].mean(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "MODEL",
                "N_CASES",
            ],
            ascending=[
                True,
                False,
            ],
        )
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
                "all_models_per_case",
            "VALUE":
                bool(
                    (
                        counts
                        == len(
                            MODEL_GROUPS
                        )
                    ).all()
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
            [
                "TARGET_CASE_ID",
                "MODEL",
            ]
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
        )[
            "TARGET_CASE_ID"
        ]
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
        (
            e1[
                "POOL_DEFINITION"
            ]
            == "CPV2_MIN1"
        )
        & (
            e1[
                "METHOD"
            ]
            == "TOPSIS"
        ),
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

    e1["TARGET_CASE_ID"] = (
        e1["TARGET_CASE_ID"]
        .astype("string")
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
                "E1X_M0_RANK",
        }
    )

    m0["TARGET_CASE_ID"] = (
        m0["TARGET_CASE_ID"]
        .astype("string")
    )

    merged = e1.merge(
        m0,
        on="TARGET_CASE_ID",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    ids_match = bool(
        (
            merged["_merge"]
            == "both"
        ).all()
    )

    ranks_match = bool(
        np.allclose(
            merged.loc[
                merged["_merge"]
                == "both",
                "E1_TOPSIS_RANK",
            ],
            merged.loc[
                merged["_merge"]
                == "both",
                "E1X_M0_RANK",
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
                    ]
                    .value_counts()
                    .to_dict()
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

    return pd.DataFrame(
        rows
    )


def write_report(
    summary_df: pd.DataFrame,
    incremental_df: pd.DataFrame,
    buyer_geo_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# E1x Extended Procurement-Fit TOPSIS Report",
        "",
        "## Scope",
        "",
        "E1x compares nested procurement-fit models on the exact same "
        "CPV2_MIN1 winner-in-pool cohort used in E1. The model family adds "
        "category-set fit, geography, buyer-relationship dynamics, selected "
        "procurement-regime fit, and structured award-criteria fit.",
        "",
        "## Model performance",
        "",
        "| Model | N | W@1 (%) | W@5 (%) | W@10 (%) | W@50 (%) | MRR | Mean percentile | Median rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"| {row['MEAN_PERCENTILE_RANK']:.6f} "
            f"| {row['MEDIAN_WINNER_RANK']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Incremental comparisons",
            "",
            "| Reference | Comparison | ΔW@10 (pp) | ΔW@50 (pp) | ΔMRR | % moved up | % unchanged | % moved down |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in incremental_df.iterrows():
        lines.append(
            f"| `{row['REFERENCE_MODEL']}` "
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
            "## Buyer versus geography decomposition relative to M1",
            "",
            "| Comparison | ΔW@10 (pp) | ΔW@50 (pp) | ΔMRR | % moved up | % moved down |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in buyer_geo_df.iterrows():
        lines.append(
            f"| `{row['COMPARISON_MODEL']}` "
            f"| {row['DELTA_WINNER_AT_10_PP']:.3f} "
            f"| {row['DELTA_WINNER_AT_50_PP']:.3f} "
            f"| {row['DELTA_MRR']:.6f} "
            f"| {row['PCT_WINNER_MOVED_UP']:.3f} "
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
            "Observed winners are reference outcomes, not proof of uniquely optimal "
            "suppliers. Buyer-relationship variables may capture incumbency or repeat "
            "contracting patterns and are therefore decomposed from geographic fit. "
            "Regime matching is restricted to selected fields with adequate target "
            "variation and historical observation support.",
            "",
            "Price-weight similarity is active only where the target procurement "
            "reports a usable price weight; missing target criteria deactivate the "
            "feature rather than being converted to zero performance.",
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
    parser = argparse.ArgumentParser(
        description="Run E1x extended procurement-fit TOPSIS models."
    )

    parser.add_argument(
        "--extended-index-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--base-fit-index-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing "
            "supplier_country_fit_counts_2015_2016.parquet"
        ),
    )

    parser.add_argument(
        "--target-case-features",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--cpv2-features",
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
            "data/analysis/e1x_rankings"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "reports/e1x_rankings"
        ),
    )

    args = parser.parse_args()

    extended = args.extended_index_dir

    required_extended_files = [
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

    for filename in required_extended_files:
        require_file(
            extended / filename
        )

    country_path = (
        args.base_fit_index_dir
        / "supplier_country_fit_counts_2015_2016.parquet"
    )

    require_file(country_path)
    require_file(
        args.target_case_features
    )
    require_file(
        args.cpv2_features
    )
    require_file(
        args.e1_case_results
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1/10 Loading target profiles and outcomes...")
    targets = load_targets(
        extended
        / "e1x_target_profiles_2017.parquet",
        args.target_case_features,
    )

    print(
        f"    cases: {len(targets):,}"
    )

    print("2/10 Loading target CPV/NUTS token maps...")
    cpv_token_map = load_token_map(
        extended
        / "e1x_target_cpv_tokens_2017.parquet",
        [
            "CPV_CODE",
            "CPV3",
            "CPV4",
        ],
    )

    nuts_token_map = load_token_map(
        extended
        / "e1x_target_nuts_tokens_2017.parquet",
        [
            "NUTS1",
            "NUTS2",
            "NUTS3",
        ],
    )

    print("3/10 Loading CPV2 candidates and context descriptors...")
    candidates = load_cpv2_candidates(
        args.cpv2_features,
        extended
        / "supplier_cpv2_context_descriptors_2015_2016.parquet",
    )

    print(
        f"    candidate-context rows: {len(candidates):,}"
    )

    print("4/10 Building supplier universe...")
    (
        supplier_array,
        supplier_to_idx,
    ) = create_supplier_universe(
        candidates
    )

    print(
        f"    suppliers: {len(supplier_array):,}"
    )

    print("5/10 Loading sparse fit indices...")

    cpv_lookup = (
        load_hierarchical_sparse_index(
            extended
            / "supplier_cpv_fit_counts_2015_2016.parquet",
            level_col="CPV_LEVEL",
            needed_levels={
                "EXACT",
                "CPV4",
                "CPV3",
            },
            supplier_to_idx=supplier_to_idx,
        )
    )

    nuts_lookup = (
        load_hierarchical_sparse_index(
            extended
            / "supplier_nuts_fit_counts_2015_2016.parquet",
            level_col="NUTS_LEVEL",
            needed_levels={
                "NUTS1",
                "NUTS2",
                "NUTS3",
            },
            supplier_to_idx=supplier_to_idx,
        )
    )

    country_lookup = (
        load_simple_count_lookup(
            country_path,
            count_col="N_COUNTRY_AWARDS",
            supplier_to_idx=supplier_to_idx,
        )
    )

    buyer_lookup = load_buyer_lookup(
        extended
        / "supplier_buyer_relationships_2015_2016.parquet",
        supplier_to_idx,
    )

    (
        regime_match_lookup,
        regime_total_lookup,
    ) = load_regime_lookup(
        extended
        / "supplier_regime_fit_counts_2015_2016.parquet",
        supplier_to_idx,
    )

    criterion_lookup = (
        load_simple_count_lookup(
            extended
            / "supplier_criterion_code_fit_counts_2015_2016.parquet",
            count_col="N_MATCHED_AWARDS",
            supplier_to_idx=supplier_to_idx,
        )
    )

    print("6/10 Running E1x nested ranking models...")
    results = run_experiment(
        targets=targets,
        candidates=candidates,
        cpv_token_map=cpv_token_map,
        nuts_token_map=nuts_token_map,
        supplier_to_idx=supplier_to_idx,
        cpv_lookup=cpv_lookup,
        nuts_lookup=nuts_lookup,
        country_lookup=country_lookup,
        buyer_lookup=buyer_lookup,
        regime_match_lookup=regime_match_lookup,
        regime_total_lookup=regime_total_lookup,
        criterion_lookup=criterion_lookup,
    )

    print("7/10 Summarizing performance...")
    specification_df = (
        model_specification()
    )

    summary_df = summarize_models(
        results
    )

    comparison_vs_m0_df = (
        rank_movement_comparison(
            results,
            summary_df,
            "M0_ORIGINAL",
            [
                model
                for model in MODEL_GROUPS
                if model != "M0_ORIGINAL"
            ],
        )
    )

    incremental_df = (
        incremental_comparisons(
            results,
            summary_df,
        )
    )

    buyer_geo_df = (
        buyer_geo_decomposition(
            results,
            summary_df,
        )
    )

    by_cpv2_df = (
        summarize_by_cpv2(
            results
        )
    )

    print("8/10 Running validation...")
    validation_df = validation_checks(
        results,
        args.e1_case_results,
    )

    print("9/10 Writing case-level and CSV outputs...")
    results.to_parquet(
        args.output_dir
        / "e1x_case_model_results.parquet",
        index=False,
    )

    specification_df.to_csv(
        args.report_dir
        / "e1x_model_specification.csv",
        index=False,
    )

    summary_df.to_csv(
        args.report_dir
        / "e1x_model_summary.csv",
        index=False,
    )

    comparison_vs_m0_df.to_csv(
        args.report_dir
        / "e1x_model_comparison_vs_m0.csv",
        index=False,
    )

    incremental_df.to_csv(
        args.report_dir
        / "e1x_incremental_comparison.csv",
        index=False,
    )

    buyer_geo_df.to_csv(
        args.report_dir
        / "e1x_buyer_geo_decomposition.csv",
        index=False,
    )

    by_cpv2_df.to_csv(
        args.report_dir
        / "e1x_results_by_cpv2.csv",
        index=False,
    )

    validation_df.to_csv(
        args.report_dir
        / "e1x_validation_checks.csv",
        index=False,
    )

    print("10/10 Writing report...")
    write_report(
        summary_df,
        incremental_df,
        buyer_geo_df,
        validation_df,
        args.report_dir
        / "e1x_ranking_report.md",
    )

    print()
    print("E1x ranking experiment complete.")
    print("Review first:")
    print("  e1x_ranking_report.md")
    print("  e1x_model_summary.csv")
    print("  e1x_incremental_comparison.csv")
    print("  e1x_buyer_geo_decomposition.csv")
    print("  e1x_validation_checks.csv")


if __name__ == "__main__":
    main()
