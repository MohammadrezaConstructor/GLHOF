#!/usr/bin/env python3
"""
Extended CN/CAN feature-family audit for richer supplier-ranking design.

Purpose
-------
Re-audit the richer raw TED schemas before constructing the next ranking model.

The script uses:
- 2017 CN rows for target-side procurement information;
- 2015-2016 CAN rows for frozen historical information;
- the E1 primary CPV2_MIN1 TOPSIS cohort to define the target cases;
- the historical award-to-supplier map to restrict CAN analysis to resolved
  supplier entities.

The audit covers:
1. target-side CN field coverage and within-case consistency;
2. historical CAN award/supplier coverage;
3. common CN/CAN fields that can support target-supplier fit features;
4. list/text feasibility for CPVs, NUTS, and award criteria;
5. historical outcome-context descriptors such as competition, SME, group,
   and subcontracting variables;
6. target-only modifiers such as recurrence, duration, lots, options, renewals,
   variants, application deadlines, and operator-envelope fields.

Important
---------
This script audits feasibility. It does NOT construct a final ranking model and
does NOT use 2017 CAN outcomes to engineer target-case features.

Inputs
------
Raw delimited text or Parquet files are supported.

Required:
--cn-2017
--can-2015
--can-2016
--award-supplier-map
--e1-case-results

Outputs
-------
- extended_target_cn_field_audit.csv
- extended_historical_can_field_audit.csv
- extended_cross_side_fit_audit.csv
- extended_list_text_audit.csv
- extended_derived_target_metrics.csv
- extended_feature_family_recommendations.csv
- extended_validation_checks.csv
- extended_feature_audit_report.md

Requires
--------
pip install duckdb pandas pyarrow
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


# ---------------------------------------------------------------------
# Field catalogues
# ---------------------------------------------------------------------

TARGET_CN_FIELDS = {
    # Category / geography
    "ADDITIONAL_CPVS": "CATEGORY_SET",
    "TAL_LOCATION_NUTS": "GEOGRAPHY",

    # Procurement regime / process
    "B_FRA_AGREEMENT": "PROCUREMENT_REGIME",
    "B_DYN_PURCH_SYST": "PROCUREMENT_REGIME",
    "B_GPA": "PROCUREMENT_REGIME",
    "B_EU_FUNDS": "PROCUREMENT_REGIME",
    "B_ACCELERATED": "PROCUREMENT_REGIME",
    "B_ELECTRONIC_AUCTION": "PROCUREMENT_REGIME",
    "B_ON_BEHALF": "INSTITUTIONAL_COMPLEXITY",
    "B_INVOLVES_JOINT_PROCUREMENT": "INSTITUTIONAL_COMPLEXITY",
    "B_AWARDED_BY_CENTRAL_BODY": "INSTITUTIONAL_COMPLEXITY",

    # Lot / structure
    "LOTS_NUMBER": "LOT_STRUCTURE",
    "LOTS_SUBMISSION": "LOT_STRUCTURE",
    "B_VARIANTS": "CONTRACT_STRUCTURE",
    "B_OPTIONS": "CONTRACT_STRUCTURE",
    "B_RENEWALS": "CONTRACT_STRUCTURE",

    # Timing / complexity
    "DURATION": "TIMING",
    "CONTRACT_START": "TIMING",
    "CONTRACT_COMPLETION": "TIMING",
    "DT_APPLICATIONS": "TIMING",

    # Criteria
    "CRIT_CODE": "AWARD_CRITERIA",
    "CRIT_PRICE_WEIGHT": "AWARD_CRITERIA",
    "CRIT_CRITERIA": "AWARD_CRITERIA_TEXT",
    "CRIT_WEIGHTS": "AWARD_CRITERIA_TEXT",

    # Target-only modifiers
    "B_RECURRENT_PROCUREMENT": "RECURRENCE",
    "ENV_OPERATORS": "OPERATOR_ENVELOPE",
    "ENV_MIN_OPERATORS": "OPERATOR_ENVELOPE",
    "ENV_MAX_OPERATORS": "OPERATOR_ENVELOPE",

    # Value context
    "VALUE_EURO": "VALUE",
    "VALUE_EURO_FIN_1": "VALUE",
    "VALUE_EURO_FIN_2": "VALUE",
}

HIST_CAN_FIELDS = {
    # Category / geography
    "ADDITIONAL_CPVS": "CATEGORY_SET",
    "TAL_LOCATION_NUTS": "GEOGRAPHY",

    # Procurement regime / process
    "B_FRA_AGREEMENT": "PROCUREMENT_REGIME",
    "B_DYN_PURCH_SYST": "PROCUREMENT_REGIME",
    "B_GPA": "PROCUREMENT_REGIME",
    "B_EU_FUNDS": "PROCUREMENT_REGIME",
    "B_ACCELERATED": "PROCUREMENT_REGIME",
    "B_ELECTRONIC_AUCTION": "PROCUREMENT_REGIME",
    "B_ON_BEHALF": "INSTITUTIONAL_COMPLEXITY",
    "B_INVOLVES_JOINT_PROCUREMENT": "INSTITUTIONAL_COMPLEXITY",
    "B_AWARDED_BY_CENTRAL_BODY": "INSTITUTIONAL_COMPLEXITY",

    # Lot / criteria context
    "LOTS_NUMBER": "LOT_STRUCTURE",
    "CRIT_CODE": "AWARD_CRITERIA",
    "CRIT_PRICE_WEIGHT": "AWARD_CRITERIA",
    "CRIT_CRITERIA": "AWARD_CRITERIA_TEXT",
    "CRIT_WEIGHTS": "AWARD_CRITERIA_TEXT",

    # Historical competition context
    "NUMBER_OFFERS": "COMPETITION_HISTORY",
    "NUMBER_TENDERS_SME": "COMPETITION_HISTORY",
    "NUMBER_TENDERS_OTHER_EU": "COMPETITION_HISTORY",
    "NUMBER_TENDERS_NON_EU": "COMPETITION_HISTORY",
    "NUMBER_OFFERS_ELECTR": "COMPETITION_HISTORY",

    # Historical supplier / delivery descriptors
    "B_CONTRACTOR_SME": "SUPPLIER_POLICY_PROFILE",
    "B_AWARDED_TO_A_GROUP": "DELIVERY_STRUCTURE_HISTORY",
    "B_SUBCONTRACTED": "DELIVERY_STRUCTURE_HISTORY",

    # Historical award scale
    "AWARD_EST_VALUE_EURO": "AWARD_SCALE_HISTORY",
    "AWARD_VALUE_EURO": "AWARD_SCALE_HISTORY",
    "AWARD_VALUE_EURO_FIN_1": "AWARD_SCALE_HISTORY",
}

COMMON_FIT_FIELDS = [
    ("B_FRA_AGREEMENT", "Framework-agreement experience fit"),
    ("B_DYN_PURCH_SYST", "Dynamic-purchasing-system experience fit"),
    ("B_GPA", "GPA coverage/regime fit"),
    ("B_EU_FUNDS", "EU-funds procurement experience fit"),
    ("B_ACCELERATED", "Accelerated-procedure experience fit"),
    ("B_ELECTRONIC_AUCTION", "Electronic-auction experience fit"),
    ("B_ON_BEHALF", "On-behalf purchasing context fit"),
    ("B_INVOLVES_JOINT_PROCUREMENT", "Joint-procurement experience fit"),
    ("B_AWARDED_BY_CENTRAL_BODY", "Central-purchasing-body experience fit"),
    ("CRIT_CODE", "Award-criterion regime/code fit"),
    ("CRIT_PRICE_WEIGHT", "Price-weight context similarity"),
]

BOOLEAN_FIELDS = {
    "B_FRA_AGREEMENT",
    "B_DYN_PURCH_SYST",
    "B_GPA",
    "B_EU_FUNDS",
    "B_ACCELERATED",
    "B_ELECTRONIC_AUCTION",
    "B_ON_BEHALF",
    "B_INVOLVES_JOINT_PROCUREMENT",
    "B_AWARDED_BY_CENTRAL_BODY",
    "B_VARIANTS",
    "B_OPTIONS",
    "B_RENEWALS",
    "B_RECURRENT_PROCUREMENT",
    "B_CONTRACTOR_SME",
    "B_AWARDED_TO_A_GROUP",
    "B_SUBCONTRACTED",
}

NUMERIC_FIELDS = {
    "LOTS_NUMBER",
    "DURATION",
    "ENV_OPERATORS",
    "ENV_MIN_OPERATORS",
    "ENV_MAX_OPERATORS",
    "CRIT_PRICE_WEIGHT",
    "VALUE_EURO",
    "VALUE_EURO_FIN_1",
    "VALUE_EURO_FIN_2",
    "NUMBER_OFFERS",
    "NUMBER_TENDERS_SME",
    "NUMBER_TENDERS_OTHER_EU",
    "NUMBER_TENDERS_NON_EU",
    "NUMBER_OFFERS_ELECTR",
    "AWARD_EST_VALUE_EURO",
    "AWARD_VALUE_EURO",
    "AWARD_VALUE_EURO_FIN_1",
}

DATE_FIELDS = {
    "CONTRACT_START",
    "CONTRACT_COMPLETION",
    "DT_APPLICATIONS",
}

LIST_TEXT_FIELDS = [
    ("CN", "ADDITIONAL_CPVS", "triple_hyphen"),
    ("CAN", "ADDITIONAL_CPVS", "triple_hyphen"),
    ("CN", "TAL_LOCATION_NUTS", "comma"),
    ("CAN", "TAL_LOCATION_NUTS", "comma"),
    ("CN", "CRIT_CRITERIA", "triple_hyphen"),
    ("CAN", "CRIT_CRITERIA", "triple_hyphen"),
    ("CN", "CRIT_WEIGHTS", "triple_hyphen"),
    ("CAN", "CRIT_WEIGHTS", "triple_hyphen"),
]


# ---------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------

def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def clean_expr(expr: str) -> str:
    return f"NULLIF(TRIM(CAST({expr} AS VARCHAR)), '')"


def norm_expr(expr: str) -> str:
    return f"UPPER({clean_expr(expr)})"


def numeric_expr(expr: str) -> str:
    # Conservative parse: trim spaces, remove percent sign, convert comma decimal.
    cleaned = (
        f"REPLACE(REPLACE(REPLACE(TRIM(CAST({expr} AS VARCHAR)), "
        "' ', ''), '%', ''), ',', '.')"
    )
    return f"TRY_CAST({cleaned} AS DOUBLE)"


def boolean_expr(expr: str) -> str:
    u = f"UPPER(TRIM(CAST({expr} AS VARCHAR)))"
    return (
        "CASE "
        f"WHEN {u} IN ('Y','YES','TRUE','T','1') THEN TRUE "
        f"WHEN {u} IN ('N','NO','FALSE','F','0') THEN FALSE "
        "ELSE NULL END"
    )


def date_expr(expr: str) -> str:
    c = clean_expr(expr)
    return (
        "COALESCE("
        f"TRY_STRPTIME({c}, '%d/%m/%Y')::DATE, "
        f"TRY_STRPTIME({c}, '%d/%m/%y')::DATE, "
        f"TRY_STRPTIME({c}, '%Y-%m-%d')::DATE, "
        f"TRY_CAST({c} AS DATE)"
        ")"
    )


def key_part(expr: str) -> str:
    return clean_expr(expr)


def detect_delimiter(path: Path) -> str:
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        sample = f.read(131072)

    try:
        return csv.Sniffer().sniff(
            sample,
            delimiters="\t,;|",
        ).delimiter
    except csv.Error:
        return "\t"


def relation_sql(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".parquet"):
        return f"read_parquet('{sql_path(path)}')"

    delimiter = detect_delimiter(path)
    delim = delimiter.replace("'", "''")

    return (
        "read_csv("
        f"'{sql_path(path)}', "
        "header=true, "
        f"delim='{delim}', "
        "all_varchar=true, "
        "ignore_errors=false, "
        "null_padding=true"
        ")"
    )


def relation_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM {relation_sql(path)}"
        ).fetchall()
    }


def select_with_missing(
    relation: str,
    available_cols: set[str],
    desired_cols: Iterable[str],
    source_year: int | None = None,
) -> str:
    exprs = []

    if source_year is not None:
        exprs.append(
            f"{int(source_year)}::INTEGER AS SOURCE_YEAR"
        )

    for col in desired_cols:
        if col in available_cols:
            exprs.append(
                f"CAST({qident(col)} AS VARCHAR) AS {qident(col)}"
            )
        else:
            exprs.append(
                f"NULL::VARCHAR AS {qident(col)}"
            )

    return (
        "SELECT\n    "
        + ",\n    ".join(exprs)
        + f"\nFROM {relation}"
    )


def scalar(
    con: duckdb.DuckDBPyConnection,
    sql: str,
):
    return con.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------
# Base-table creation
# ---------------------------------------------------------------------

def validate_minimum_columns(
    con: duckdb.DuckDBPyConnection,
    args,
) -> tuple[set[str], set[str], set[str]]:
    cn_cols = relation_columns(
        con,
        args.cn_2017,
    )
    can15_cols = relation_columns(
        con,
        args.can_2015,
    )
    can16_cols = relation_columns(
        con,
        args.can_2016,
    )

    required_cn = {"ID_NOTICE_CN"}
    required_can = {
        "ID_NOTICE_CAN",
        "ID_AWARD",
    }

    missing_cn = sorted(required_cn - cn_cols)
    if missing_cn:
        raise ValueError(
            "CN input missing required columns: "
            + ", ".join(missing_cn)
        )

    for label, cols in [
        ("CAN 2015", can15_cols),
        ("CAN 2016", can16_cols),
    ]:
        missing = sorted(required_can - cols)
        if missing:
            raise ValueError(
                f"{label} missing required columns: "
                + ", ".join(missing)
            )

    # Validate Parquet helper inputs.
    for path, required, label in [
        (
            args.award_supplier_map,
            {"AWARD_KEY", "SUPPLIER_ENTITY_ID"},
            "award_supplier_map",
        ),
        (
            args.e1_case_results,
            {
                "TARGET_CASE_ID",
                "POOL_DEFINITION",
                "METHOD",
            },
            "e1_case_results",
        ),
    ]:
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
                f"{label} missing required columns: "
                + ", ".join(missing)
            )

    return cn_cols, can15_cols, can16_cols


def create_base_tables(
    con: duckdb.DuckDBPyConnection,
    args,
    cn_cols: set[str],
    can15_cols: set[str],
    can16_cols: set[str],
) -> None:
    print("    materializing primary E1 case IDs...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE e1_primary_cases AS
        SELECT DISTINCT
            CAST(TARGET_CASE_ID AS VARCHAR)
                AS TARGET_CASE_ID
        FROM read_parquet('{sql_path(args.e1_case_results)}')
        WHERE POOL_DEFINITION = 'CPV2_MIN1'
          AND METHOD = 'TOPSIS'
        """
    )

    desired_cn_cols = [
        "ID_NOTICE_CN",
        "DT_DISPATCH",
        *TARGET_CN_FIELDS.keys(),
    ]

    cn_select = select_with_missing(
        relation_sql(args.cn_2017),
        cn_cols,
        desired_cn_cols,
    )

    print("    materializing primary-cohort CN rows...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE cn_primary_raw AS
        SELECT r.*
        FROM (
            {cn_select}
        ) r
        JOIN e1_primary_cases e
          ON {clean_expr('r.ID_NOTICE_CN')}
           = e.TARGET_CASE_ID
        """
    )

    desired_can_cols = [
        "ID_NOTICE_CAN",
        "ID_AWARD",
        *HIST_CAN_FIELDS.keys(),
    ]

    can15_select = select_with_missing(
        relation_sql(args.can_2015),
        can15_cols,
        desired_can_cols,
        source_year=2015,
    )

    can16_select = select_with_missing(
        relation_sql(args.can_2016),
        can16_cols,
        desired_can_cols,
        source_year=2016,
    )

    print("    materializing 2015-2016 raw CAN rows...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE can_hist_raw AS
        {can15_select}
        UNION ALL
        {can16_select}
        """
    )

    print("    joining historical CAN rows to resolved supplier entities...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE can_hist_mapped_raw AS
        WITH keyed AS (
            SELECT
                *,
                CASE
                    WHEN {clean_expr('ID_NOTICE_CAN')} IS NOT NULL
                     AND {clean_expr('ID_AWARD')} IS NOT NULL
                    THEN
                        {clean_expr('ID_NOTICE_CAN')}
                        || '||' ||
                        {clean_expr('ID_AWARD')}
                    ELSE NULL
                END AS AWARD_KEY
            FROM can_hist_raw
        )
        SELECT
            k.*,
            CAST(m.SUPPLIER_ENTITY_ID AS VARCHAR)
                AS SUPPLIER_ENTITY_ID
        FROM keyed k
        JOIN read_parquet('{sql_path(args.award_supplier_map)}') m
          USING (AWARD_KEY)
        WHERE m.SUPPLIER_ENTITY_ID IS NOT NULL
        """
    )


# ---------------------------------------------------------------------
# Target-side field audit
# ---------------------------------------------------------------------

def target_field_audit(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    total_cases = int(
        scalar(
            con,
            "SELECT COUNT(*) FROM e1_primary_cases",
        )
    )

    rows = []

    for field, family in TARGET_CN_FIELDS.items():
        f = qident(field)
        c = clean_expr(f)

        stats = con.execute(
            f"""
            WITH per_case AS (
                SELECT
                    {clean_expr('ID_NOTICE_CN')} AS TARGET_CASE_ID,
                    COUNT(DISTINCT {c})
                        FILTER (WHERE {c} IS NOT NULL)
                        AS N_DISTINCT_NON_NULL,
                    MAX(CASE WHEN {c} IS NOT NULL THEN 1 ELSE 0 END)
                        AS HAS_VALUE
                FROM cn_primary_raw
                GROUP BY TARGET_CASE_ID
            )
            SELECT
                SUM(HAS_VALUE),
                SUM(
                    CASE
                        WHEN N_DISTINCT_NON_NULL > 1
                        THEN 1 ELSE 0
                    END
                ),
                MAX(N_DISTINCT_NON_NULL)
            FROM per_case
            """
        ).fetchone()

        n_covered = int(stats[0] or 0)
        n_conflict = int(stats[1] or 0)
        max_distinct = int(stats[2] or 0)

        n_unique_values = int(
            scalar(
                con,
                f"""
                SELECT COUNT(DISTINCT {c})
                FROM cn_primary_raw
                WHERE {c} IS NOT NULL
                """,
            )
            or 0
        )

        row = {
            "FIELD": field,
            "FAMILY": family,
            "N_TARGET_CASES": total_cases,
            "N_CASES_WITH_VALUE": n_covered,
            "CASE_COVERAGE_PCT":
                100.0 * n_covered / total_cases
                if total_cases else 0.0,
            "N_CASES_WITH_MULTIPLE_DISTINCT_VALUES":
                n_conflict,
            "MULTI_VALUE_CASE_PCT":
                100.0 * n_conflict / total_cases
                if total_cases else 0.0,
            "MAX_DISTINCT_VALUES_WITHIN_CASE":
                max_distinct,
            "N_UNIQUE_RAW_VALUES":
                n_unique_values,
            "NUMERIC_PARSE_PCT_OF_NON_NULL": None,
            "BOOLEAN_PARSE_PCT_OF_NON_NULL": None,
            "DATE_PARSE_PCT_OF_NON_NULL": None,
        }

        if field in NUMERIC_FIELDS:
            n_non_null, n_parsed = con.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE {c} IS NOT NULL),
                    COUNT(*) FILTER (
                        WHERE {c} IS NOT NULL
                          AND {numeric_expr(f)} IS NOT NULL
                    )
                FROM cn_primary_raw
                """
            ).fetchone()

            row["NUMERIC_PARSE_PCT_OF_NON_NULL"] = (
                100.0 * int(n_parsed) / int(n_non_null)
                if n_non_null else None
            )

        if field in BOOLEAN_FIELDS:
            n_non_null, n_parsed = con.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE {c} IS NOT NULL),
                    COUNT(*) FILTER (
                        WHERE {c} IS NOT NULL
                          AND {boolean_expr(f)} IS NOT NULL
                    )
                FROM cn_primary_raw
                """
            ).fetchone()

            row["BOOLEAN_PARSE_PCT_OF_NON_NULL"] = (
                100.0 * int(n_parsed) / int(n_non_null)
                if n_non_null else None
            )

        if field in DATE_FIELDS:
            n_non_null, n_parsed = con.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE {c} IS NOT NULL),
                    COUNT(*) FILTER (
                        WHERE {c} IS NOT NULL
                          AND {date_expr(f)} IS NOT NULL
                    )
                FROM cn_primary_raw
                """
            ).fetchone()

            row["DATE_PARSE_PCT_OF_NON_NULL"] = (
                100.0 * int(n_parsed) / int(n_non_null)
                if n_non_null else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Historical CAN field audit
# ---------------------------------------------------------------------

def historical_field_audit(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    total_awards = int(
        scalar(
            con,
            """
            SELECT COUNT(DISTINCT AWARD_KEY)
            FROM can_hist_mapped_raw
            """,
        )
    )

    total_suppliers = int(
        scalar(
            con,
            """
            SELECT COUNT(DISTINCT SUPPLIER_ENTITY_ID)
            FROM can_hist_mapped_raw
            """,
        )
    )

    rows = []

    for field, family in HIST_CAN_FIELDS.items():
        f = qident(field)
        c = clean_expr(f)

        stats = con.execute(
            f"""
            WITH award_field AS (
                SELECT
                    AWARD_KEY,
                    SUPPLIER_ENTITY_ID,
                    COUNT(DISTINCT {c})
                        FILTER (WHERE {c} IS NOT NULL)
                        AS N_DISTINCT_NON_NULL,
                    MAX(CASE WHEN {c} IS NOT NULL THEN 1 ELSE 0 END)
                        AS HAS_VALUE
                FROM can_hist_mapped_raw
                GROUP BY AWARD_KEY, SUPPLIER_ENTITY_ID
            ),
            supplier_field AS (
                SELECT
                    SUPPLIER_ENTITY_ID,
                    MAX(HAS_VALUE) AS HAS_VALUE
                FROM award_field
                GROUP BY SUPPLIER_ENTITY_ID
            )
            SELECT
                (SELECT SUM(HAS_VALUE) FROM award_field),
                (SELECT SUM(HAS_VALUE) FROM supplier_field),
                (
                    SELECT SUM(
                        CASE WHEN N_DISTINCT_NON_NULL > 1
                        THEN 1 ELSE 0 END
                    )
                    FROM award_field
                ),
                (
                    SELECT MAX(N_DISTINCT_NON_NULL)
                    FROM award_field
                )
            """
        ).fetchone()

        n_awards_covered = int(stats[0] or 0)
        n_suppliers_covered = int(stats[1] or 0)
        n_award_multivalue = int(stats[2] or 0)
        max_distinct = int(stats[3] or 0)

        n_unique_values = int(
            scalar(
                con,
                f"""
                SELECT COUNT(DISTINCT {c})
                FROM can_hist_mapped_raw
                WHERE {c} IS NOT NULL
                """,
            )
            or 0
        )

        row = {
            "FIELD": field,
            "FAMILY": family,
            "N_HISTORICAL_AWARDS": total_awards,
            "N_AWARDS_WITH_VALUE": n_awards_covered,
            "AWARD_COVERAGE_PCT":
                100.0 * n_awards_covered / total_awards
                if total_awards else 0.0,
            "N_HISTORICAL_SUPPLIERS": total_suppliers,
            "N_SUPPLIERS_WITH_VALUE": n_suppliers_covered,
            "SUPPLIER_COVERAGE_PCT":
                100.0 * n_suppliers_covered / total_suppliers
                if total_suppliers else 0.0,
            "N_AWARDS_WITH_MULTIPLE_DISTINCT_VALUES":
                n_award_multivalue,
            "MULTI_VALUE_AWARD_PCT":
                100.0 * n_award_multivalue / total_awards
                if total_awards else 0.0,
            "MAX_DISTINCT_VALUES_WITHIN_AWARD":
                max_distinct,
            "N_UNIQUE_RAW_VALUES":
                n_unique_values,
            "NUMERIC_PARSE_PCT_OF_NON_NULL": None,
            "BOOLEAN_PARSE_PCT_OF_NON_NULL": None,
        }

        if field in NUMERIC_FIELDS:
            n_non_null, n_parsed = con.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE {c} IS NOT NULL),
                    COUNT(*) FILTER (
                        WHERE {c} IS NOT NULL
                          AND {numeric_expr(f)} IS NOT NULL
                    )
                FROM can_hist_mapped_raw
                """
            ).fetchone()

            row["NUMERIC_PARSE_PCT_OF_NON_NULL"] = (
                100.0 * int(n_parsed) / int(n_non_null)
                if n_non_null else None
            )

        if field in BOOLEAN_FIELDS:
            n_non_null, n_parsed = con.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE {c} IS NOT NULL),
                    COUNT(*) FILTER (
                        WHERE {c} IS NOT NULL
                          AND {boolean_expr(f)} IS NOT NULL
                    )
                FROM can_hist_mapped_raw
                """
            ).fetchone()

            row["BOOLEAN_PARSE_PCT_OF_NON_NULL"] = (
                100.0 * int(n_parsed) / int(n_non_null)
                if n_non_null else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Cross-side fit audit
# ---------------------------------------------------------------------

def cross_side_fit_audit(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    total_cases = int(
        scalar(
            con,
            "SELECT COUNT(*) FROM e1_primary_cases",
        )
    )

    rows = []

    for field, interpretation in COMMON_FIT_FIELDS:
        f = qident(field)

        if field in BOOLEAN_FIELDS:
            target_value = (
                f"CAST({boolean_expr('c.' + f)} AS VARCHAR)"
            )
            hist_value = (
                f"CAST({boolean_expr('h.' + f)} AS VARCHAR)"
            )
        elif field in NUMERIC_FIELDS:
            target_value = (
                f"CAST({numeric_expr('c.' + f)} AS VARCHAR)"
            )
            hist_value = (
                f"CAST({numeric_expr('h.' + f)} AS VARCHAR)"
            )
        else:
            target_value = norm_expr("c." + f)
            hist_value = norm_expr("h." + f)

        stats = con.execute(
            f"""
            WITH target_values AS (
                SELECT DISTINCT
                    {clean_expr('c.ID_NOTICE_CN')}
                        AS TARGET_CASE_ID,
                    {target_value}
                        AS FIT_VALUE
                FROM cn_primary_raw c
                WHERE {target_value} IS NOT NULL
            ),
            hist_values AS (
                SELECT DISTINCT
                    {hist_value}
                        AS FIT_VALUE
                FROM can_hist_mapped_raw h
                WHERE {hist_value} IS NOT NULL
            ),
            target_case_status AS (
                SELECT
                    t.TARGET_CASE_ID,
                    MAX(
                        CASE
                            WHEN hv.FIT_VALUE IS NOT NULL
                            THEN 1 ELSE 0
                        END
                    ) AS VALUE_SEEN_IN_HISTORY
                FROM target_values t
                LEFT JOIN hist_values hv
                  USING (FIT_VALUE)
                GROUP BY t.TARGET_CASE_ID
            )
            SELECT
                (SELECT COUNT(DISTINCT TARGET_CASE_ID)
                 FROM target_values),
                (SELECT SUM(VALUE_SEEN_IN_HISTORY)
                 FROM target_case_status)
            """
        ).fetchone()

        n_target_non_null = int(stats[0] or 0)
        n_seen = int(stats[1] or 0)

        rows.append(
            {
                "FIELD": field,
                "PROPOSED_USE": interpretation,
                "N_TARGET_CASES": total_cases,
                "N_TARGET_CASES_WITH_VALUE":
                    n_target_non_null,
                "TARGET_COVERAGE_PCT":
                    100.0 * n_target_non_null / total_cases
                    if total_cases else 0.0,
                "N_TARGET_CASES_VALUE_SEEN_IN_HISTORY":
                    n_seen,
                "VALUE_SEEN_IN_HISTORY_PCT_OF_NON_NULL":
                    100.0 * n_seen / n_target_non_null
                    if n_target_non_null else 0.0,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# List/text audit
# ---------------------------------------------------------------------

def token_count_expr(expr: str, mode: str) -> str:
    c = clean_expr(expr)

    if mode == "triple_hyphen":
        return (
            f"CASE WHEN {c} IS NULL THEN NULL ELSE "
            f"1 + ((LENGTH({c}) - LENGTH(REPLACE({c}, '---', ''))) / 3) "
            "END"
        )

    if mode == "comma":
        return (
            f"CASE WHEN {c} IS NULL THEN NULL ELSE "
            f"1 + (LENGTH({c}) - LENGTH(REPLACE({c}, ',', ''))) "
            "END"
        )

    return f"CASE WHEN {c} IS NULL THEN NULL ELSE 1 END"


def list_text_audit(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []

    for side, field, mode in LIST_TEXT_FIELDS:
        table = (
            "cn_primary_raw"
            if side == "CN"
            else "can_hist_mapped_raw"
        )
        id_col = (
            "ID_NOTICE_CN"
            if side == "CN"
            else "AWARD_KEY"
        )

        f = qident(field)
        c = clean_expr(f)
        token_expr = token_count_expr(f, mode)

        stats = con.execute(
            f"""
            WITH per_unit AS (
                SELECT
                    {clean_expr(id_col)} AS UNIT_ID,
                    MAX(CASE WHEN {c} IS NOT NULL THEN 1 ELSE 0 END)
                        AS HAS_VALUE,
                    MAX({token_expr})
                        AS MAX_TOKEN_COUNT,
                    MAX(LENGTH({c}))
                        AS MAX_CHAR_LENGTH
                FROM {table}
                GROUP BY UNIT_ID
            )
            SELECT
                COUNT(*),
                SUM(HAS_VALUE),
                AVG(MAX_TOKEN_COUNT)
                    FILTER (WHERE HAS_VALUE = 1),
                QUANTILE_CONT(MAX_TOKEN_COUNT, 0.95)
                    FILTER (WHERE HAS_VALUE = 1),
                AVG(MAX_CHAR_LENGTH)
                    FILTER (WHERE HAS_VALUE = 1),
                QUANTILE_CONT(MAX_CHAR_LENGTH, 0.95)
                    FILTER (WHERE HAS_VALUE = 1)
            FROM per_unit
            """
        ).fetchone()

        n_units = int(stats[0] or 0)
        n_covered = int(stats[1] or 0)

        rows.append(
            {
                "SIDE": side,
                "FIELD": field,
                "N_UNITS": n_units,
                "N_UNITS_WITH_VALUE": n_covered,
                "COVERAGE_PCT":
                    100.0 * n_covered / n_units
                    if n_units else 0.0,
                "MEAN_MAX_TOKEN_COUNT":
                    float(stats[2]) if stats[2] is not None else None,
                "P95_MAX_TOKEN_COUNT":
                    float(stats[3]) if stats[3] is not None else None,
                "MEAN_MAX_CHAR_LENGTH":
                    float(stats[4]) if stats[4] is not None else None,
                "P95_MAX_CHAR_LENGTH":
                    float(stats[5]) if stats[5] is not None else None,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Derived target metrics
# ---------------------------------------------------------------------

def derived_target_metrics(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []

    # Application lead time.
    lead_stats = con.execute(
        f"""
        WITH per_case AS (
            SELECT
                {clean_expr('ID_NOTICE_CN')} AS TARGET_CASE_ID,
                MIN({date_expr('DT_APPLICATIONS')})
                    AS APPLICATION_DATE,
                MIN({date_expr('DT_DISPATCH')})
                    AS DISPATCH_DATE
            FROM cn_primary_raw
            GROUP BY TARGET_CASE_ID
        ),
        derived AS (
            SELECT
                DATE_DIFF(
                    'day',
                    DISPATCH_DATE,
                    APPLICATION_DATE
                ) AS VALUE
            FROM per_case
            WHERE APPLICATION_DATE IS NOT NULL
              AND DISPATCH_DATE IS NOT NULL
        )
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE VALUE >= 0),
            MEDIAN(VALUE) FILTER (WHERE VALUE >= 0),
            QUANTILE_CONT(VALUE, 0.90) FILTER (WHERE VALUE >= 0)
        FROM derived
        """
    ).fetchone()

    total_cases = int(
        scalar(
            con,
            "SELECT COUNT(*) FROM e1_primary_cases",
        )
    )

    n_pair = int(lead_stats[0] or 0)
    n_valid = int(lead_stats[1] or 0)

    rows.append(
        {
            "DERIVED_METRIC": "APPLICATION_LEAD_DAYS",
            "N_TARGET_CASES": total_cases,
            "N_CASES_COMPUTABLE": n_valid,
            "COVERAGE_PCT":
                100.0 * n_valid / total_cases
                if total_cases else 0.0,
            "MEDIAN_VALUE":
                float(lead_stats[2]) if lead_stats[2] is not None else None,
            "P90_VALUE":
                float(lead_stats[3]) if lead_stats[3] is not None else None,
            "NOTE":
                f"{n_pair} cases had parseable date pairs before nonnegative filtering.",
        }
    )

    # Coalesced target value.
    target_value = (
        "COALESCE("
        f"{numeric_expr('VALUE_EURO')}, "
        f"{numeric_expr('VALUE_EURO_FIN_1')}, "
        f"{numeric_expr('VALUE_EURO_FIN_2')}"
        ")"
    )

    value_stats = con.execute(
        f"""
        WITH per_case AS (
            SELECT
                {clean_expr('ID_NOTICE_CN')} AS TARGET_CASE_ID,
                MAX({target_value}) AS VALUE
            FROM cn_primary_raw
            GROUP BY TARGET_CASE_ID
        )
        SELECT
            COUNT(VALUE),
            MEDIAN(VALUE) FILTER (WHERE VALUE > 0),
            QUANTILE_CONT(VALUE, 0.90)
                FILTER (WHERE VALUE > 0)
        FROM per_case
        """
    ).fetchone()

    n_value = int(value_stats[0] or 0)

    rows.append(
        {
            "DERIVED_METRIC": "TARGET_VALUE_EURO_COALESCED",
            "N_TARGET_CASES": total_cases,
            "N_CASES_COMPUTABLE": n_value,
            "COVERAGE_PCT":
                100.0 * n_value / total_cases
                if total_cases else 0.0,
            "MEDIAN_VALUE":
                float(value_stats[1]) if value_stats[1] is not None else None,
            "P90_VALUE":
                float(value_stats[2]) if value_stats[2] is not None else None,
            "NOTE":
                "Coalesces VALUE_EURO, VALUE_EURO_FIN_1, VALUE_EURO_FIN_2.",
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------

def feature_family_recommendations() -> pd.DataFrame:
    rows = [
        {
            "FEATURE_FAMILY": "Hierarchical CPV + additional CPV set overlap",
            "SOURCE_FIELDS": "CPV; ADDITIONAL_CPVS",
            "PROPOSED_ROLE": "MAIN_RANKING_FIT",
            "NEXT_ENGINEERING_STEP":
                "Build exact/CPV4/CPV3/CPV2 overlap counts and target-set coverage.",
            "CAUTION":
                "Parse list delimiters conservatively and deduplicate codes.",
        },
        {
            "FEATURE_FAMILY": "NUTS geographic delivery fit",
            "SOURCE_FIELDS": "TAL_LOCATION_NUTS; ISO_COUNTRY_CODE",
            "PROPOSED_ROLE": "MAIN_RANKING_FIT",
            "NEXT_ENGINEERING_STEP":
                "Parse NUTS codes and build historical counts at NUTS1/NUTS2/NUTS3 plus country.",
            "CAUTION":
                "Do not treat geography as supplier quality; interpret as delivery-market experience.",
        },
        {
            "FEATURE_FAMILY": "Buyer relationship dynamics",
            "SOURCE_FIELDS": "CAE identity; DT_AWARD; B_RECURRENT_PROCUREMENT",
            "PROPOSED_ROLE": "MAIN_RANKING_FIT_WITH_ABLATION",
            "NEXT_ENGINEERING_STEP":
                "Build prior-buyer count, buyer-specific recency, relationship share, and recurrence interaction.",
            "CAUTION":
                "May capture incumbency/repeat contracting; report no-buyer ablation.",
        },
        {
            "FEATURE_FAMILY": "Procurement-regime fit",
            "SOURCE_FIELDS":
                "B_FRA_AGREEMENT; B_DYN_PURCH_SYST; B_GPA; B_EU_FUNDS; "
                "B_ACCELERATED; B_ELECTRONIC_AUCTION; joint/central/on-behalf flags",
            "PROPOSED_ROLE": "MAIN_OR_SENSITIVITY_AFTER_AUDIT",
            "NEXT_ENGINEERING_STEP":
                "Build historical supplier counts conditional on matching target regime values.",
            "CAUTION":
                "Sparse or nearly constant flags should not receive standalone weight.",
        },
        {
            "FEATURE_FAMILY": "Award-criteria alignment",
            "SOURCE_FIELDS":
                "CRIT_CODE; CRIT_PRICE_WEIGHT; CRIT_CRITERIA; CRIT_WEIGHTS",
            "PROPOSED_ROLE": "MAIN_FIT_PLUS_LLM_BRIDGE",
            "NEXT_ENGINEERING_STEP":
                "Start with structured code/price-weight fit; separately evaluate text semantic similarity.",
            "CAUTION":
                "Criteria text is multilingual and repeated; keep source provenance and do not mix unvalidated embeddings blindly.",
        },
        {
            "FEATURE_FAMILY": "Scale fit",
            "SOURCE_FIELDS":
                "CN VALUE_EURO*; CAN AWARD_VALUE_EURO*; AWARD_EST_VALUE_EURO",
            "PROPOSED_ROLE": "VALUE_OBSERVED_SENSITIVITY",
            "NEXT_ENGINEERING_STEP":
                "Build log-scale distance to supplier historical median/P75/P90 award values.",
            "CAUTION":
                "Use only cases with observed target value; do not impute a main-model value for missing cases.",
        },
        {
            "FEATURE_FAMILY": "Lot and contract complexity",
            "SOURCE_FIELDS":
                "LOTS_NUMBER; LOTS_SUBMISSION; DURATION; B_OPTIONS; B_RENEWALS; B_VARIANTS",
            "PROPOSED_ROLE": "TARGET_COMPLEXITY_MODIFIER",
            "NEXT_ENGINEERING_STEP":
                "Create target complexity indicators and interactions with relevant historical experience.",
            "CAUTION":
                "Avoid treating larger/more complex procurements as intrinsically better.",
        },
        {
            "FEATURE_FAMILY": "Competition history",
            "SOURCE_FIELDS":
                "NUMBER_OFFERS; NUMBER_TENDERS_SME; NUMBER_TENDERS_OTHER_EU; "
                "NUMBER_TENDERS_NON_EU; NUMBER_OFFERS_ELECTR",
            "PROPOSED_ROLE": "HISTORICAL_SUPPLIER_DESCRIPTOR",
            "NEXT_ENGINEERING_STEP":
                "Aggregate supplier median/P75 competition context and high-competition win share.",
            "CAUTION":
                "Measures context in which awards were won, not supplier quality.",
        },
        {
            "FEATURE_FAMILY": "SME / delivery-structure profile",
            "SOURCE_FIELDS":
                "B_CONTRACTOR_SME; B_AWARDED_TO_A_GROUP; B_SUBCONTRACTED",
            "PROPOSED_ROLE": "SCENARIO_OR_COMPLEXITY_INTERACTION",
            "NEXT_ENGINEERING_STEP":
                "Build conservative historical shares/flags after checking multi-supplier records.",
            "CAUTION":
                "Do not treat SME, consortium, or subcontracting status as universal benefit criteria.",
        },
        {
            "FEATURE_FAMILY": "Tender timing",
            "SOURCE_FIELDS": "DT_DISPATCH; DT_APPLICATIONS",
            "PROPOSED_ROLE": "TARGET_COMPLEXITY_MODIFIER",
            "NEXT_ENGINEERING_STEP":
                "Use application lead days as a target-context feature or interaction.",
            "CAUTION":
                "Not a supplier quality signal by itself.",
        },
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Validation and report
# ---------------------------------------------------------------------

def validation_checks(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []

    n_e1 = int(
        scalar(
            con,
            "SELECT COUNT(*) FROM e1_primary_cases",
        )
    )

    n_cn_ids = int(
        scalar(
            con,
            f"""
            SELECT COUNT(DISTINCT {clean_expr('ID_NOTICE_CN')})
            FROM cn_primary_raw
            """,
        )
    )

    rows.append(
        {
            "CHECK": "primary_e1_cases_found_in_raw_cn",
            "VALUE": n_e1 == n_cn_ids,
            "DETAIL":
                f"e1_cases={n_e1}; raw_cn_matched_cases={n_cn_ids}",
        }
    )

    n_mapped_awards = int(
        scalar(
            con,
            """
            SELECT COUNT(DISTINCT AWARD_KEY)
            FROM can_hist_mapped_raw
            """,
        )
    )

    n_mapped_suppliers = int(
        scalar(
            con,
            """
            SELECT COUNT(DISTINCT SUPPLIER_ENTITY_ID)
            FROM can_hist_mapped_raw
            """,
        )
    )

    rows.append(
        {
            "CHECK": "historical_mapped_awards_nonempty",
            "VALUE": n_mapped_awards > 0,
            "DETAIL":
                f"mapped_awards={n_mapped_awards}; mapped_suppliers={n_mapped_suppliers}",
        }
    )

    years = con.execute(
        """
        SELECT DISTINCT SOURCE_YEAR
        FROM can_hist_mapped_raw
        ORDER BY SOURCE_YEAR
        """
    ).fetchall()

    years_list = [int(r[0]) for r in years]

    rows.append(
        {
            "CHECK": "historical_can_years_are_2015_2016_only",
            "VALUE": years_list == [2015, 2016],
            "DETAIL": str(years_list),
        }
    )

    return pd.DataFrame(rows)


def write_report(
    target_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    list_df: pd.DataFrame,
    derived_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Extended CN/CAN Feature-Family Audit",
        "",
        "## Scope",
        "",
        "This audit re-examines the richer raw CN and CAN schemas before constructing "
        "the next supplier-ranking model. Target-side information comes from 2017 CN "
        "records in the primary E1 cohort; historical information comes only from "
        "2015-2016 CAN awards mapped to frozen supplier entities.",
        "",
        "## Target-side CN field coverage",
        "",
        "| Field | Family | Coverage (%) | Multi-value cases (%) | Unique values |",
        "|---|---|---:|---:|---:|",
    ]

    for _, row in target_df.iterrows():
        lines.append(
            f"| `{row['FIELD']}` "
            f"| `{row['FAMILY']}` "
            f"| {row['CASE_COVERAGE_PCT']:.3f} "
            f"| {row['MULTI_VALUE_CASE_PCT']:.3f} "
            f"| {int(row['N_UNIQUE_RAW_VALUES'])} |"
        )

    lines.extend([
        "",
        "## Historical CAN field coverage",
        "",
        "| Field | Family | Award coverage (%) | Supplier coverage (%) | Multi-value awards (%) |",
        "|---|---|---:|---:|---:|",
    ])

    for _, row in hist_df.iterrows():
        lines.append(
            f"| `{row['FIELD']}` "
            f"| `{row['FAMILY']}` "
            f"| {row['AWARD_COVERAGE_PCT']:.3f} "
            f"| {row['SUPPLIER_COVERAGE_PCT']:.3f} "
            f"| {row['MULTI_VALUE_AWARD_PCT']:.3f} |"
        )

    lines.extend([
        "",
        "## Cross-side fit feasibility",
        "",
        "| Field | Proposed use | Target coverage (%) | Target value seen in history (%) |",
        "|---|---|---:|---:|",
    ])

    for _, row in cross_df.iterrows():
        lines.append(
            f"| `{row['FIELD']}` "
            f"| {row['PROPOSED_USE']} "
            f"| {row['TARGET_COVERAGE_PCT']:.3f} "
            f"| {row['VALUE_SEEN_IN_HISTORY_PCT_OF_NON_NULL']:.3f} |"
        )

    lines.extend([
        "",
        "## List/text feasibility",
        "",
        "| Side | Field | Coverage (%) | Mean max tokens | P95 max tokens | Mean max chars |",
        "|---|---|---:|---:|---:|---:|",
    ])

    for _, row in list_df.iterrows():
        mean_tokens = (
            f"{row['MEAN_MAX_TOKEN_COUNT']:.3f}"
            if pd.notna(row["MEAN_MAX_TOKEN_COUNT"])
            else "NA"
        )
        p95_tokens = (
            f"{row['P95_MAX_TOKEN_COUNT']:.3f}"
            if pd.notna(row["P95_MAX_TOKEN_COUNT"])
            else "NA"
        )
        mean_chars = (
            f"{row['MEAN_MAX_CHAR_LENGTH']:.3f}"
            if pd.notna(row["MEAN_MAX_CHAR_LENGTH"])
            else "NA"
        )

        lines.append(
            f"| `{row['SIDE']}` "
            f"| `{row['FIELD']}` "
            f"| {row['COVERAGE_PCT']:.3f} "
            f"| {mean_tokens} "
            f"| {p95_tokens} "
            f"| {mean_chars} |"
        )

    lines.extend([
        "",
        "## Derived target metrics",
        "",
        "| Metric | Coverage (%) | Median | P90 |",
        "|---|---:|---:|---:|",
    ])

    for _, row in derived_df.iterrows():
        median = (
            f"{row['MEDIAN_VALUE']:.3f}"
            if pd.notna(row["MEDIAN_VALUE"])
            else "NA"
        )
        p90 = (
            f"{row['P90_VALUE']:.3f}"
            if pd.notna(row["P90_VALUE"])
            else "NA"
        )

        lines.append(
            f"| `{row['DERIVED_METRIC']}` "
            f"| {row['COVERAGE_PCT']:.3f} "
            f"| {median} "
            f"| {p90} |"
        )

    lines.extend([
        "",
        "## Recommended feature-family roles",
        "",
        "| Feature family | Proposed role | Next engineering step |",
        "|---|---|---|",
    ])

    for _, row in rec_df.iterrows():
        lines.append(
            f"| {row['FEATURE_FAMILY']} "
            f"| `{row['PROPOSED_ROLE']}` "
            f"| {row['NEXT_ENGINEERING_STEP']} |"
        )

    lines.extend([
        "",
        "## Validation",
        "",
    ])

    for _, row in validation_df.iterrows():
        status = "PASS" if bool(row["VALUE"]) else "FAIL"
        lines.append(
            f"- **{status}** `{row['CHECK']}`: {row['DETAIL']}"
        )

    lines.extend([
        "",
        "## Interpretation caution",
        "",
        "Coverage alone is not sufficient for inclusion in a ranking model. Fields "
        "should be retained only where the target-side meaning is available before "
        "award, historical supplier support is adequate, and the feature has a clear "
        "decision role. Historical competition, SME, consortium, and subcontracting "
        "variables describe award contexts or delivery structures and should not be "
        "treated as universal supplier-quality scores.",
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
    parser = argparse.ArgumentParser(
        description="Audit richer raw CN/CAN feature families."
    )

    parser.add_argument(
        "--cn-2017",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--can-2015",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--can-2016",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--award-supplier-map",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--e1-case-results",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "reports/extended_feature_audit"
        ),
    )
    parser.add_argument(
        "--work-db",
        type=Path,
        default=None,
        help="Optional DuckDB database file. Default uses in-memory DB.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(
            1,
            min(8, os.cpu_count() or 4),
        ),
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
        help="Optional spill directory",
    )

    args = parser.parse_args()

    for path in [
        args.cn_2017,
        args.can_2015,
        args.can_2016,
        args.award_supplier_map,
        args.e1_case_results,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.work_db:
        args.work_db.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        con = duckdb.connect(
            str(args.work_db)
        )
    else:
        con = duckdb.connect()

    con.execute(
        f"SET threads={int(args.threads)}"
    )

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
        print("1/9 Validating raw schemas...")
        (
            cn_cols,
            can15_cols,
            can16_cols,
        ) = validate_minimum_columns(
            con,
            args,
        )

        print("2/9 Building cohort-restricted audit tables...")
        create_base_tables(
            con,
            args,
            cn_cols,
            can15_cols,
            can16_cols,
        )

        print("3/9 Auditing target-side CN fields...")
        target_df = target_field_audit(con)

        print("4/9 Auditing historical CAN fields...")
        hist_df = historical_field_audit(con)

        print("5/9 Auditing cross-side fit feasibility...")
        cross_df = cross_side_fit_audit(con)

        print("6/9 Auditing list/text fields...")
        list_df = list_text_audit(con)

        print("7/9 Computing derived target metrics...")
        derived_df = derived_target_metrics(con)

        print("8/9 Running validation and preparing recommendations...")
        rec_df = feature_family_recommendations()
        validation_df = validation_checks(con)

        print("9/9 Writing reports...")
        target_df.to_csv(
            args.report_dir
            / "extended_target_cn_field_audit.csv",
            index=False,
        )
        hist_df.to_csv(
            args.report_dir
            / "extended_historical_can_field_audit.csv",
            index=False,
        )
        cross_df.to_csv(
            args.report_dir
            / "extended_cross_side_fit_audit.csv",
            index=False,
        )
        list_df.to_csv(
            args.report_dir
            / "extended_list_text_audit.csv",
            index=False,
        )
        derived_df.to_csv(
            args.report_dir
            / "extended_derived_target_metrics.csv",
            index=False,
        )
        rec_df.to_csv(
            args.report_dir
            / "extended_feature_family_recommendations.csv",
            index=False,
        )
        validation_df.to_csv(
            args.report_dir
            / "extended_validation_checks.csv",
            index=False,
        )

        write_report(
            target_df,
            hist_df,
            cross_df,
            list_df,
            derived_df,
            rec_df,
            validation_df,
            args.report_dir
            / "extended_feature_audit_report.md",
        )

        print()
        print("Extended feature-family audit complete.")
        print("Review first:")
        print("  extended_feature_audit_report.md")
        print("  extended_target_cn_field_audit.csv")
        print("  extended_historical_can_field_audit.csv")
        print("  extended_cross_side_fit_audit.csv")
        print("  extended_list_text_audit.csv")
        print("  extended_derived_target_metrics.csv")
        print("  extended_validation_checks.csv")

    finally:
        con.close()


if __name__ == "__main__":
    main()
