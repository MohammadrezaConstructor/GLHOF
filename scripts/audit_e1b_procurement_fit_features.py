#!/usr/bin/env python3
"""
E1b feature-feasibility audit.

Purpose
-------
Audit whether richer procurement-fit features are sufficiently available and
historically supported before building the E1b ranking model.

This script does NOT fit a ranking model. It measures:

1. target-field coverage on the primary E1 cohort (CPV2_MIN1 winner-in-pool cases);
2. whether target attribute values are represented in the 2015-2016 history;
3. compact historical supplier-by-attribute index sizes;
4. descriptive observed-winner support for each proposed fit dimension;
5. target-value and historical award-value feasibility for a scale-fit sensitivity.

Primary E1b candidate dimensions
--------------------------------
- CPV3 experience
- CPV4 experience
- prior buyer experience
- procurement-country experience
- contract-type experience
- buyer main-activity experience
- authority-type experience
- procedure-type experience
- framework-agreement experience
- dynamic-purchasing-system experience
- electronic-auction experience

Important caution
-----------------
Observed-winner support is descriptive only. It must not be used to learn
case-specific weights or leak 2017 outcomes into feature construction.

Outputs
-------
- e1b_target_field_coverage.csv
- e1b_historical_support_summary.csv
- e1b_winner_fit_support_summary.csv
- e1b_value_feasibility.csv
- e1b_feature_feasibility_report.md

Requires
--------
pip install duckdb pandas pyarrow
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def scalar(
    con: duckdb.DuckDBPyConnection,
    sql: str,
):
    return con.execute(sql).fetchone()[0]


def upper_clean(expr: str) -> str:
    return (
        f"UPPER(NULLIF(TRIM(CAST({expr} AS VARCHAR)), ''))"
    )


def alnum_norm(expr: str) -> str:
    return (
        "NULLIF("
        "REGEXP_REPLACE("
        f"UPPER(TRIM(CAST({expr} AS VARCHAR))), "
        "'[^A-Z0-9]', '', 'g'"
        "), '')"
    )


def valid_country(expr: str) -> str:
    clean = upper_clean(expr)
    return (
        f"CASE WHEN {clean} IS NOT NULL "
        f"AND LENGTH({clean}) = 2 "
        f"THEN {clean} ELSE NULL END"
    )


def buyer_key_sql(
    country_expr: str,
    national_id_expr: str,
    name_expr: str,
) -> str:
    country = valid_country(country_expr)
    nid = alnum_norm(national_id_expr)
    name = alnum_norm(name_expr)

    valid_nid = (
        f"{nid} IS NOT NULL "
        f"AND LENGTH({nid}) >= 4 "
        f"AND {nid} NOT IN "
        "('0000','000000','00000000','NA','NONE','UNKNOWN','NULL') "
        f"AND NOT REGEXP_MATCHES({nid}, '^0+$')"
    )

    return (
        "CASE "
        f"WHEN {country} IS NOT NULL AND ({valid_nid}) "
        f"THEN 'BID||' || {country} || '||' || {nid} "
        f"WHEN {country} IS NOT NULL AND {name} IS NOT NULL "
        f"THEN 'BN||' || {country} || '||' || {name} "
        "ELSE NULL END"
    )


def bool_text(expr: str) -> str:
    return (
        "CASE "
        f"WHEN TRY_CAST({expr} AS BOOLEAN) IS TRUE THEN 'TRUE' "
        f"WHEN TRY_CAST({expr} AS BOOLEAN) IS FALSE THEN 'FALSE' "
        "ELSE NULL END"
    )


def cpv_prefix(expr: str, n: int) -> str:
    digits = (
        "REGEXP_REPLACE("
        f"COALESCE(CAST({expr} AS VARCHAR), ''), "
        "'[^0-9]', '', 'g'"
        ")"
    )
    return (
        f"CASE WHEN LENGTH({digits}) >= {n} "
        f"THEN SUBSTR({digits}, 1, {n}) "
        "ELSE NULL END"
    )


def require_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    required: set[str],
    label: str,
) -> None:
    cols = {
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{sql_path(path)}')
            """
        ).fetchall()
    }

    missing = sorted(required - cols)
    if missing:
        raise ValueError(
            f"{label} is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_inputs(
    con: duckdb.DuckDBPyConnection,
    args,
) -> None:
    require_columns(
        con,
        args.target_cases,
        {
            "TARGET_CASE_ID",
            "TARGET_DISPATCH_DATE",
            "TARGET_CPV2",
            "TARGET_CPV3",
            "TARGET_CPV4",
            "TARGET_PROCUREMENT_COUNTRY",
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        },
        "target_case_features",
    )

    require_columns(
        con,
        args.procurement_cases,
        {
            "ID_NOTICE_CN_CLEAN",
            "CAE_NAME_CLEAN",
            "CAE_NATIONALID_CLEAN",
            "ISO_COUNTRY_CODE_CLEAN",
            "CAE_TYPE_CLEAN",
            "MAIN_ACTIVITY_CLEAN",
            "TYPE_OF_CONTRACT_CLEAN",
            "TOP_TYPE_CLEAN",
            "B_FRA_AGREEMENT_CLEAN",
            "B_DYN_PURCH_SYST_CLEAN",
            "B_ELECTRONIC_AUCTION_CLEAN",
            "VALUE_EURO_MAX",
            "VALUE_EURO_FIN_1_MAX",
            "VALUE_EURO_FIN_2_MAX",
        },
        "procurement_cases",
    )

    require_columns(
        con,
        args.award_supplier_map,
        {
            "AWARD_KEY",
            "SUPPLIER_ENTITY_ID",
        },
        "award_supplier_map",
    )

    require_columns(
        con,
        args.award_fact,
        {
            "AWARD_KEY",
            "CPV_CLEAN",
            "CAE_NAME_CLEAN",
            "CAE_NATIONALID_CLEAN",
            "ISO_COUNTRY_CODE_CLEAN",
            "CAE_TYPE_CLEAN",
            "MAIN_ACTIVITY_CLEAN",
            "TYPE_OF_CONTRACT_CLEAN",
            "TOP_TYPE_CLEAN",
            "B_FRA_AGREEMENT_CLEAN",
            "B_DYN_PURCH_SYST_CLEAN",
            "B_ELECTRONIC_AUCTION_CLEAN",
            "AWARD_VALUE_EURO_CLEAN",
            "AWARD_VALUE_EURO_FIN_1_CLEAN",
            "AWARD_EST_VALUE_EURO_CLEAN",
        },
        "award_fact",
    )

    require_columns(
        con,
        args.cpv2_features,
        {
            "CONTEXT_KEY",
            "SUPPLIER_ENTITY_ID",
            "MEDIAN_CONTEXT_AWARD_VALUE_EURO",
        },
        "supplier_cpv2_features",
    )

    require_columns(
        con,
        args.e1_case_results,
        {
            "TARGET_CASE_ID",
            "POOL_DEFINITION",
            "METHOD",
        },
        "E1 case results",
    )


# ---------------------------------------------------------------------
# Build audit views
# ---------------------------------------------------------------------

def create_views(
    con: duckdb.DuckDBPyConnection,
    args,
) -> None:
    target_buyer_key = buyer_key_sql(
        "p.ISO_COUNTRY_CODE_CLEAN",
        "p.CAE_NATIONALID_CLEAN",
        "p.CAE_NAME_CLEAN",
    )

    hist_buyer_key = buyer_key_sql(
        "f.ISO_COUNTRY_CODE_CLEAN",
        "f.CAE_NATIONALID_CLEAN",
        "f.CAE_NAME_CLEAN",
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW e1_primary_cases AS
        SELECT DISTINCT
            CAST(TARGET_CASE_ID AS VARCHAR) AS TARGET_CASE_ID
        FROM read_parquet('{sql_path(args.e1_case_results)}')
        WHERE POOL_DEFINITION = 'CPV2_MIN1'
          AND METHOD = 'TOPSIS'
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW target_fit_audit AS
        SELECT
            CAST(t.TARGET_CASE_ID AS VARCHAR)
                AS TARGET_CASE_ID,

            CAST(t.TARGET_DISPATCH_DATE AS DATE)
                AS TARGET_DISPATCH_DATE,

            CAST(t.TARGET_CPV2 AS VARCHAR)
                AS CPV2,

            CAST(t.TARGET_CPV3 AS VARCHAR)
                AS CPV3,

            CAST(t.TARGET_CPV4 AS VARCHAR)
                AS CPV4,

            {target_buyer_key}
                AS BUYER_KEY,

            {valid_country("p.ISO_COUNTRY_CODE_CLEAN")}
                AS PROCUREMENT_COUNTRY,

            {upper_clean("p.TYPE_OF_CONTRACT_CLEAN")}
                AS CONTRACT_TYPE,

            {upper_clean("p.MAIN_ACTIVITY_CLEAN")}
                AS MAIN_ACTIVITY,

            {upper_clean("p.CAE_TYPE_CLEAN")}
                AS AUTHORITY_TYPE,

            {upper_clean("p.TOP_TYPE_CLEAN")}
                AS PROCEDURE_TYPE,

            {bool_text("p.B_FRA_AGREEMENT_CLEAN")}
                AS FRAMEWORK_AGREEMENT,

            {bool_text("p.B_DYN_PURCH_SYST_CLEAN")}
                AS DYNAMIC_PURCHASING_SYSTEM,

            {bool_text("p.B_ELECTRONIC_AUCTION_CLEAN")}
                AS ELECTRONIC_AUCTION,

            COALESCE(
                TRY_CAST(p.VALUE_EURO_MAX AS DOUBLE),
                TRY_CAST(p.VALUE_EURO_FIN_1_MAX AS DOUBLE),
                TRY_CAST(p.VALUE_EURO_FIN_2_MAX AS DOUBLE)
            ) AS TARGET_VALUE_EURO,

            CAST(
                t.OBSERVED_WINNER_HISTORICAL_ENTITY_ID
                AS VARCHAR
            ) AS OBSERVED_WINNER_HISTORICAL_ENTITY_ID

        FROM read_parquet('{sql_path(args.target_cases)}') t
        JOIN e1_primary_cases e
          ON CAST(t.TARGET_CASE_ID AS VARCHAR)
           = e.TARGET_CASE_ID
        JOIN read_parquet('{sql_path(args.procurement_cases)}') p
          ON CAST(p.ID_NOTICE_CN_CLEAN AS VARCHAR)
           = CAST(t.TARGET_CASE_ID AS VARCHAR)
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW hist_fit_base AS
        SELECT
            CAST(m.SUPPLIER_ENTITY_ID AS VARCHAR)
                AS SUPPLIER_ENTITY_ID,

            {cpv_prefix("f.CPV_CLEAN", 3)}
                AS CPV3,

            {cpv_prefix("f.CPV_CLEAN", 4)}
                AS CPV4,

            {hist_buyer_key}
                AS BUYER_KEY,

            {valid_country("f.ISO_COUNTRY_CODE_CLEAN")}
                AS PROCUREMENT_COUNTRY,

            {upper_clean("f.TYPE_OF_CONTRACT_CLEAN")}
                AS CONTRACT_TYPE,

            {upper_clean("f.MAIN_ACTIVITY_CLEAN")}
                AS MAIN_ACTIVITY,

            {upper_clean("f.CAE_TYPE_CLEAN")}
                AS AUTHORITY_TYPE,

            {upper_clean("f.TOP_TYPE_CLEAN")}
                AS PROCEDURE_TYPE,

            {bool_text("f.B_FRA_AGREEMENT_CLEAN")}
                AS FRAMEWORK_AGREEMENT,

            {bool_text("f.B_DYN_PURCH_SYST_CLEAN")}
                AS DYNAMIC_PURCHASING_SYSTEM,

            {bool_text("f.B_ELECTRONIC_AUCTION_CLEAN")}
                AS ELECTRONIC_AUCTION,

            COALESCE(
                TRY_CAST(f.AWARD_VALUE_EURO_CLEAN AS DOUBLE),
                TRY_CAST(f.AWARD_VALUE_EURO_FIN_1_CLEAN AS DOUBLE),
                TRY_CAST(f.AWARD_EST_VALUE_EURO_CLEAN AS DOUBLE)
            ) AS HISTORICAL_AWARD_VALUE_EURO

        FROM read_parquet('{sql_path(args.award_supplier_map)}') m
        JOIN read_parquet('{sql_path(args.award_fact)}') f
          USING (AWARD_KEY)
        WHERE m.SUPPLIER_ENTITY_ID IS NOT NULL
        """
    )


# ---------------------------------------------------------------------
# Audit summaries
# ---------------------------------------------------------------------

DIMENSIONS = [
    ("CPV3", "Hierarchical category fit at 3-digit CPV level"),
    ("CPV4", "Hierarchical category fit at 4-digit CPV level"),
    ("BUYER_KEY", "Prior relationship with the target contracting authority"),
    ("PROCUREMENT_COUNTRY", "Historical experience in the target procurement country"),
    ("CONTRACT_TYPE", "Historical experience with the target contract type"),
    ("MAIN_ACTIVITY", "Historical experience with buyer main-activity context"),
    ("AUTHORITY_TYPE", "Historical experience with the target authority type"),
    ("PROCEDURE_TYPE", "Historical experience with the target procedure type"),
    ("FRAMEWORK_AGREEMENT", "Historical experience with framework-agreement context"),
    ("DYNAMIC_PURCHASING_SYSTEM", "Historical experience with dynamic purchasing systems"),
    ("ELECTRONIC_AUCTION", "Historical experience with electronic-auction context"),
]


def target_field_coverage(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []
    n_cases = int(
        scalar(
            con,
            "SELECT COUNT(*) FROM target_fit_audit",
        )
    )

    for dim, description in DIMENSIONS:
        row = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE {qident(dim)} IS NOT NULL
                ) AS n_non_null,
                COUNT(DISTINCT {qident(dim)}) FILTER (
                    WHERE {qident(dim)} IS NOT NULL
                ) AS n_unique
            FROM target_fit_audit
            """
        ).fetchone()

        n_non_null = int(row[0])
        n_unique = int(row[1])

        rows.append(
            {
                "DIMENSION": dim,
                "DESCRIPTION": description,
                "N_TARGET_CASES": n_cases,
                "N_NON_NULL_TARGET_CASES": n_non_null,
                "TARGET_COVERAGE_PCT":
                    100.0 * n_non_null / n_cases
                    if n_cases else 0.0,
                "N_UNIQUE_TARGET_VALUES": n_unique,
            }
        )

    return pd.DataFrame(rows)


def historical_support_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []

    for dim, description in DIMENSIONS:
        result = con.execute(
            f"""
            WITH
            hist_pairs AS (
                SELECT DISTINCT
                    SUPPLIER_ENTITY_ID,
                    {qident(dim)} AS DIM_VALUE
                FROM hist_fit_base
                WHERE {qident(dim)} IS NOT NULL
            ),
            hist_values AS (
                SELECT DISTINCT DIM_VALUE
                FROM hist_pairs
            )
            SELECT
                COUNT(*) FILTER (
                    WHERE t.{qident(dim)} IS NOT NULL
                ) AS n_target_non_null,

                COUNT(*) FILTER (
                    WHERE t.{qident(dim)} IS NOT NULL
                      AND hv.DIM_VALUE IS NOT NULL
                ) AS n_target_value_seen_in_history,

                (
                    SELECT COUNT(*)
                    FROM hist_pairs
                ) AS n_historical_supplier_value_pairs,

                (
                    SELECT COUNT(DISTINCT SUPPLIER_ENTITY_ID)
                    FROM hist_pairs
                ) AS n_historical_suppliers_with_dimension

            FROM target_fit_audit t
            LEFT JOIN hist_values hv
              ON t.{qident(dim)} = hv.DIM_VALUE
            """
        ).fetchone()

        n_non_null = int(result[0])
        n_seen = int(result[1])

        rows.append(
            {
                "DIMENSION": dim,
                "DESCRIPTION": description,
                "N_TARGET_NON_NULL": n_non_null,
                "N_TARGET_CASES_VALUE_SEEN_IN_HISTORY": n_seen,
                "VALUE_SEEN_IN_HISTORY_PCT_OF_NON_NULL":
                    100.0 * n_seen / n_non_null
                    if n_non_null else 0.0,
                "N_HISTORICAL_SUPPLIER_VALUE_PAIRS":
                    int(result[2]),
                "N_HISTORICAL_SUPPLIERS_WITH_DIMENSION":
                    int(result[3]),
            }
        )

    return pd.DataFrame(rows)


def winner_fit_support_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []

    for dim, description in DIMENSIONS:
        result = con.execute(
            f"""
            WITH hist_pairs AS (
                SELECT DISTINCT
                    SUPPLIER_ENTITY_ID,
                    {qident(dim)} AS DIM_VALUE
                FROM hist_fit_base
                WHERE {qident(dim)} IS NOT NULL
            )
            SELECT
                COUNT(*) FILTER (
                    WHERE t.{qident(dim)} IS NOT NULL
                ) AS n_target_non_null,

                COUNT(*) FILTER (
                    WHERE t.{qident(dim)} IS NOT NULL
                      AND hp.SUPPLIER_ENTITY_ID IS NOT NULL
                ) AS n_winner_positive_fit

            FROM target_fit_audit t
            LEFT JOIN hist_pairs hp
              ON hp.SUPPLIER_ENTITY_ID
               = t.OBSERVED_WINNER_HISTORICAL_ENTITY_ID
             AND hp.DIM_VALUE
               = t.{qident(dim)}
            """
        ).fetchone()

        n_non_null = int(result[0])
        n_positive = int(result[1])

        rows.append(
            {
                "DIMENSION": dim,
                "DESCRIPTION": description,
                "N_TARGET_NON_NULL": n_non_null,
                "N_OBSERVED_WINNERS_WITH_POSITIVE_HISTORICAL_FIT":
                    n_positive,
                "WINNER_POSITIVE_FIT_PCT_OF_NON_NULL":
                    100.0 * n_positive / n_non_null
                    if n_non_null else 0.0,
                "INTERPRETATION_CAUTION":
                    "Descriptive audit only; do not use 2017 winner outcomes to construct features or weights.",
            }
        )

    return pd.DataFrame(rows)


def value_feasibility(
    con: duckdb.DuckDBPyConnection,
    cpv2_features: Path,
) -> pd.DataFrame:
    target_stats = con.execute(
        """
        SELECT
            COUNT(*) AS n_cases,
            COUNT(TARGET_VALUE_EURO) AS n_target_value,
            MEDIAN(TARGET_VALUE_EURO)
                FILTER (WHERE TARGET_VALUE_EURO > 0)
                AS median_target_value
        FROM target_fit_audit
        """
    ).fetchone()

    cpv2_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(MEDIAN_CONTEXT_AWARD_VALUE_EURO)
                AS n_value_rows
        FROM read_parquet('{sql_path(cpv2_features)}')
        """
    ).fetchone()

    paired_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_cases,
            COUNT(*) FILTER (
                WHERE t.TARGET_VALUE_EURO IS NOT NULL
                  AND f.MEDIAN_CONTEXT_AWARD_VALUE_EURO IS NOT NULL
            ) AS n_target_and_winner_context_value
        FROM target_fit_audit t
        LEFT JOIN read_parquet('{sql_path(cpv2_features)}') f
          ON CAST(f.CONTEXT_KEY AS VARCHAR) = t.CPV2
         AND CAST(f.SUPPLIER_ENTITY_ID AS VARCHAR)
           = t.OBSERVED_WINNER_HISTORICAL_ENTITY_ID
        """
    ).fetchone()

    n_cases = int(target_stats[0])
    n_target_value = int(target_stats[1])
    n_cpv2_rows = int(cpv2_stats[0])
    n_cpv2_value = int(cpv2_stats[1])
    n_paired = int(paired_stats[1])

    rows = [
        {
            "METRIC": "target_value_coverage_pct",
            "VALUE":
                100.0 * n_target_value / n_cases
                if n_cases else 0.0,
            "N_DENOMINATOR": n_cases,
            "N_NUMERATOR": n_target_value,
        },
        {
            "METRIC": "cpv2_supplier_context_value_coverage_pct",
            "VALUE":
                100.0 * n_cpv2_value / n_cpv2_rows
                if n_cpv2_rows else 0.0,
            "N_DENOMINATOR": n_cpv2_rows,
            "N_NUMERATOR": n_cpv2_value,
        },
        {
            "METRIC":
                "target_and_observed_winner_context_value_pair_coverage_pct",
            "VALUE":
                100.0 * n_paired / n_cases
                if n_cases else 0.0,
            "N_DENOMINATOR": n_cases,
            "N_NUMERATOR": n_paired,
        },
        {
            "METRIC": "median_positive_target_value_euro",
            "VALUE":
                float(target_stats[2])
                if target_stats[2] is not None
                else None,
            "N_DENOMINATOR": n_target_value,
            "N_NUMERATOR": None,
        },
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_report(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    winner_df: pd.DataFrame,
    value_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# E1b Procurement-Fit Feature Feasibility Audit",
        "",
        "## Purpose",
        "",
        "This audit evaluates whether richer procurement-fit dimensions are sufficiently "
        "available and historically represented before constructing the E1b ranking model.",
        "",
        "The audit uses the primary E1 winner-in-pool cohort and frozen 2015-2016 history. "
        "Observed-winner support statistics are descriptive only and are not used to "
        "construct features or weights.",
        "",
        "## Target-field coverage",
        "",
        "| Dimension | Target coverage (%) | Unique target values |",
        "|---|---:|---:|",
    ]

    for _, row in target_df.iterrows():
        lines.append(
            f"| `{row['DIMENSION']}` "
            f"| {row['TARGET_COVERAGE_PCT']:.3f} "
            f"| {int(row['N_UNIQUE_TARGET_VALUES'])} |"
        )

    lines.extend([
        "",
        "## Historical support for target values",
        "",
        "| Dimension | Target values seen in history (%) | Historical supplier-value pairs |",
        "|---|---:|---:|",
    ])

    for _, row in history_df.iterrows():
        lines.append(
            f"| `{row['DIMENSION']}` "
            f"| {row['VALUE_SEEN_IN_HISTORY_PCT_OF_NON_NULL']:.3f} "
            f"| {int(row['N_HISTORICAL_SUPPLIER_VALUE_PAIRS'])} |"
        )

    lines.extend([
        "",
        "## Descriptive observed-winner fit support",
        "",
        "| Dimension | Winners with positive historical fit (%) |",
        "|---|---:|",
    ])

    for _, row in winner_df.iterrows():
        lines.append(
            f"| `{row['DIMENSION']}` "
            f"| {row['WINNER_POSITIVE_FIT_PCT_OF_NON_NULL']:.3f} |"
        )

    lines.extend([
        "",
        "## Value-fit feasibility",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ])

    for _, row in value_df.iterrows():
        value = row["VALUE"]
        value_text = (
            f"{value:.6f}"
            if pd.notna(value)
            else "NA"
        )
        lines.append(
            f"| `{row['METRIC']}` | {value_text} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "A field should not be included in the E1b ranking merely because it exists. "
        "The main selection criteria should combine conceptual relevance, target-side "
        "coverage, historical support, and manageable dimensionality. Sparse or "
        "special-purpose fields may be retained for sensitivity or scenario analysis.",
        "",
    ])

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
        "--target-cases",
        required=True,
        type=Path,
        help="target_case_features_2017.parquet",
    )
    parser.add_argument(
        "--procurement-cases",
        required=True,
        type=Path,
        help="procurement_cases_2017.parquet",
    )
    parser.add_argument(
        "--award-supplier-map",
        required=True,
        type=Path,
        help="award_supplier_map_2015_2016.parquet",
    )
    parser.add_argument(
        "--award-fact",
        required=True,
        type=Path,
        help="award_fact_2015_2017.parquet",
    )
    parser.add_argument(
        "--cpv2-features",
        required=True,
        type=Path,
        help="supplier_cpv2_features_2015_2016.parquet",
    )
    parser.add_argument(
        "--e1-case-results",
        required=True,
        type=Path,
        help="e1_case_ranking_results.parquet",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/e1b_feature_audit"),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=None,
        help="Optional DuckDB memory limit, e.g. 8GB",
    )
    parser.add_argument(
        "--temp-directory",
        type=Path,
        default=None,
        help="Optional DuckDB spill directory",
    )

    args = parser.parse_args()

    for path in [
        args.target_cases,
        args.procurement_cases,
        args.award_supplier_map,
        args.award_fact,
        args.cpv2_features,
        args.e1_case_results,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")

    if args.memory_limit:
        con.execute(
            f"SET memory_limit='{args.memory_limit}'"
        )

    if args.temp_directory:
        args.temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        con.execute(
            f"SET temp_directory='{sql_path(args.temp_directory)}'"
        )

    try:
        print("1/7 Validating inputs...")
        validate_inputs(con, args)

        print("2/7 Creating audit views...")
        create_views(con, args)

        n_cases = scalar(
            con,
            "SELECT COUNT(*) FROM target_fit_audit",
        )
        print(f"    primary E1 cohort cases: {int(n_cases):,}")

        print("3/7 Auditing target-field coverage...")
        target_df = target_field_coverage(con)

        print("4/7 Auditing historical support...")
        history_df = historical_support_summary(con)

        print("5/7 Auditing descriptive winner-fit support...")
        winner_df = winner_fit_support_summary(con)

        print("6/7 Auditing value-fit feasibility...")
        value_df = value_feasibility(
            con,
            args.cpv2_features,
        )

        print("7/7 Writing reports...")
        target_df.to_csv(
            args.report_dir
            / "e1b_target_field_coverage.csv",
            index=False,
        )
        history_df.to_csv(
            args.report_dir
            / "e1b_historical_support_summary.csv",
            index=False,
        )
        winner_df.to_csv(
            args.report_dir
            / "e1b_winner_fit_support_summary.csv",
            index=False,
        )
        value_df.to_csv(
            args.report_dir
            / "e1b_value_feasibility.csv",
            index=False,
        )

        write_report(
            target_df,
            history_df,
            winner_df,
            value_df,
            args.report_dir
            / "e1b_feature_feasibility_report.md",
        )

        print()
        print("E1b feature-feasibility audit complete.")
        print("Review:")
        print("  e1b_feature_feasibility_report.md")
        print("  e1b_target_field_coverage.csv")
        print("  e1b_historical_support_summary.csv")
        print("  e1b_winner_fit_support_summary.csv")
        print("  e1b_value_feasibility.csv")

    finally:
        con.close()


if __name__ == "__main__":
    main()
