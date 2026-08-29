#!/usr/bin/env python3
"""
Build extended procurement-fit feature indices from raw TED CN/CAN schemas.

Design
------
Target side:
    2017 Contract Notices (CN), restricted to the exact primary E1 cohort.

Historical side:
    2015-2016 Contract Award Notices (CAN), joined to the frozen historical
    supplier entity map.

This script builds compact analysis-layer tables. It does NOT create a giant
case-by-supplier Cartesian product and does NOT use 2017 CAN outcomes to
construct target features.

Built outputs
-------------
Target tables:
- e1x_target_profiles_2017.parquet
- e1x_target_cpv_tokens_2017.parquet
- e1x_target_nuts_tokens_2017.parquet
- e1x_target_criteria_2017.parquet

Historical indices:
- supplier_cpv_fit_counts_2015_2016.parquet
- supplier_nuts_fit_counts_2015_2016.parquet
- supplier_buyer_relationships_2015_2016.parquet
- supplier_regime_fit_counts_2015_2016.parquet
- supplier_criterion_code_fit_counts_2015_2016.parquet
- supplier_cpv2_context_descriptors_2015_2016.parquet

Reports:
- e1x_target_boolean_prevalence.csv
- e1x_historical_boolean_prevalence.csv
- e1x_output_metrics.csv
- e1x_validation_checks.csv
- e1x_extended_feature_build_report.md

Feature families represented
----------------------------
1. Hierarchical CPV fit, including ADDITIONAL_CPVS.
2. Hierarchical NUTS delivery-location fit.
3. Buyer relationship dynamics:
   count, last award date, and supplier relationship share.
4. Procurement-regime fit:
   framework, DPS, GPA, EU funds, electronic auction, on-behalf,
   joint procurement, central body, accelerated.
5. Structured award-criteria fit:
   CRIT_CODE and historical CPV2-level price-weight context.
6. Historical CPV2-level descriptors:
   competition, lot structure, award value scale, and selected delivery/policy
   context shares.

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
# Field definitions
# ---------------------------------------------------------------------

REGIME_FIELDS = [
    "B_FRA_AGREEMENT",
    "B_DYN_PURCH_SYST",
    "B_GPA",
    "B_EU_FUNDS",
    "B_ACCELERATED",
    "B_ELECTRONIC_AUCTION",
    "B_ON_BEHALF",
    "B_INVOLVES_JOINT_PROCUREMENT",
    "B_AWARDED_BY_CENTRAL_BODY",
]

TARGET_BOOLEAN_FIELDS = REGIME_FIELDS + [
    "B_VARIANTS",
    "B_OPTIONS",
    "B_RENEWALS",
    "B_RECURRENT_PROCUREMENT",
]

HISTORICAL_BOOLEAN_FIELDS = REGIME_FIELDS + [
    "B_CONTRACTOR_SME",
    "B_AWARDED_TO_A_GROUP",
    "B_SUBCONTRACTED",
]

CN_REQUIRED = {
    "ID_NOTICE_CN",
    "DT_DISPATCH",
    "CPV",
}

CAN_REQUIRED = {
    "ID_NOTICE_CAN",
    "ID_AWARD",
    "CPV",
}

CN_DESIRED = [
    "ID_NOTICE_CN",
    "DT_DISPATCH",
    "CPV",
    "ADDITIONAL_CPVS",
    "TAL_LOCATION_NUTS",

    "CAE_NAME",
    "CAE_NATIONALID",
    "ISO_COUNTRY_CODE",

    *REGIME_FIELDS,

    "LOTS_NUMBER",
    "LOTS_SUBMISSION",
    "B_VARIANTS",
    "B_OPTIONS",
    "B_RENEWALS",

    "DURATION",
    "CONTRACT_START",
    "CONTRACT_COMPLETION",
    "DT_APPLICATIONS",

    "CRIT_CODE",
    "CRIT_PRICE_WEIGHT",
    "CRIT_CRITERIA",
    "CRIT_WEIGHTS",

    "B_RECURRENT_PROCUREMENT",

    "ENV_OPERATORS",
    "ENV_MIN_OPERATORS",
    "ENV_MAX_OPERATORS",

    "VALUE_EURO",
    "VALUE_EURO_FIN_1",
    "VALUE_EURO_FIN_2",
]

CAN_DESIRED = [
    "ID_NOTICE_CAN",
    "ID_AWARD",
    "DT_AWARD",
    "CPV",
    "ADDITIONAL_CPVS",
    "TAL_LOCATION_NUTS",

    "CAE_NAME",
    "CAE_NATIONALID",
    "ISO_COUNTRY_CODE",

    *REGIME_FIELDS,

    "LOTS_NUMBER",

    "CRIT_CODE",
    "CRIT_PRICE_WEIGHT",

    "NUMBER_OFFERS",
    "NUMBER_TENDERS_SME",
    "NUMBER_TENDERS_OTHER_EU",
    "NUMBER_TENDERS_NON_EU",
    "NUMBER_OFFERS_ELECTR",

    "B_CONTRACTOR_SME",
    "B_AWARDED_TO_A_GROUP",
    "B_SUBCONTRACTED",

    "AWARD_EST_VALUE_EURO",
    "AWARD_VALUE_EURO",
    "AWARD_VALUE_EURO_FIN_1",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def clean_expr(expr: str) -> str:
    return f"NULLIF(TRIM(CAST({expr} AS VARCHAR)), '')"


def upper_clean(expr: str) -> str:
    return f"UPPER({clean_expr(expr)})"


def numeric_expr(expr: str) -> str:
    cleaned = (
        f"REPLACE(REPLACE(REPLACE(TRIM(CAST({expr} AS VARCHAR)), "
        "' ', ''), '%', ''), ',', '.')"
    )
    return f"TRY_CAST({cleaned} AS DOUBLE)"


def bool_expr(expr: str) -> str:
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


def cpv_digits(expr: str) -> str:
    return (
        "REGEXP_REPLACE("
        f"COALESCE(CAST({expr} AS VARCHAR), ''), "
        "'[^0-9]', '', 'g'"
        ")"
    )


def valid_cpv_expr(expr: str) -> str:
    digits = cpv_digits(expr)
    return (
        f"CASE WHEN LENGTH({digits}) = 8 "
        f"THEN {digits} ELSE NULL END"
    )


def nuts_clean_expr(expr: str) -> str:
    return (
        "UPPER(REGEXP_REPLACE("
        f"TRIM(CAST({expr} AS VARCHAR)), "
        "'\\s+', '', 'g'"
        "))"
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

    delim = detect_delimiter(path).replace("'", "''")

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


def copy_query_to_parquet(
    con: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    con.execute(
        f"""
        COPY (
            {query}
        )
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def scalar(
    con: duckdb.DuckDBPyConnection,
    sql: str,
):
    return con.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------
# Input validation and materialization
# ---------------------------------------------------------------------

def validate_inputs(
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

    missing_cn = sorted(CN_REQUIRED - cn_cols)
    if missing_cn:
        raise ValueError(
            "CN input missing required columns: "
            + ", ".join(missing_cn)
        )

    for label, cols in [
        ("CAN 2015", can15_cols),
        ("CAN 2016", can16_cols),
    ]:
        missing = sorted(CAN_REQUIRED - cols)
        if missing:
            raise ValueError(
                f"{label} missing required columns: "
                + ", ".join(missing)
            )

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
    print("    materializing primary E1 cases...")

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

    cn_select = select_with_missing(
        relation_sql(args.cn_2017),
        cn_cols,
        CN_DESIRED,
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

    can15_select = select_with_missing(
        relation_sql(args.can_2015),
        can15_cols,
        CAN_DESIRED,
        source_year=2015,
    )

    can16_select = select_with_missing(
        relation_sql(args.can_2016),
        can16_cols,
        CAN_DESIRED,
        source_year=2016,
    )

    print("    materializing 2015-2016 CAN rows...")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE can_hist_raw AS
        {can15_select}
        UNION ALL
        {can16_select}
        """
    )

    print("    joining historical awards to frozen supplier entities...")

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
                        || '||'
                        || {clean_expr('ID_AWARD')}
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
# Target-side outputs
# ---------------------------------------------------------------------

def single_bool_case_expr(field: str) -> str:
    parsed = bool_expr(qident(field))
    return (
        "CASE "
        f"WHEN COUNT(DISTINCT {parsed}) "
        f"FILTER (WHERE {parsed} IS NOT NULL) = 1 "
        f"THEN MAX({parsed}) "
        "ELSE NULL END"
    )


def single_text_case_expr(field: str) -> str:
    cleaned = upper_clean(qident(field))
    return (
        "CASE "
        f"WHEN COUNT(DISTINCT {cleaned}) "
        f"FILTER (WHERE {cleaned} IS NOT NULL) = 1 "
        f"THEN MAX({cleaned}) "
        "ELSE NULL END"
    )


def build_target_profile(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    buyer_key = buyer_key_sql(
        "ISO_COUNTRY_CODE",
        "CAE_NATIONALID",
        "CAE_NAME",
    )

    target_value = (
        "COALESCE("
        f"{numeric_expr('VALUE_EURO')}, "
        f"{numeric_expr('VALUE_EURO_FIN_1')}, "
        f"{numeric_expr('VALUE_EURO_FIN_2')}"
        ")"
    )

    main_cpv = valid_cpv_expr("CPV")

    regime_selects = [
        f"{single_bool_case_expr(field)} AS TARGET_{field}"
        for field in REGIME_FIELDS
    ]

    query = f"""
        SELECT
            {clean_expr('ID_NOTICE_CN')}
                AS TARGET_CASE_ID,

            MIN({date_expr('DT_DISPATCH')})
                AS TARGET_DISPATCH_DATE,

            MAX({main_cpv})
                AS TARGET_MAIN_CPV,

            SUBSTR(MAX({main_cpv}), 1, 2)
                AS TARGET_CPV2,

            SUBSTR(MAX({main_cpv}), 1, 3)
                AS TARGET_CPV3,

            SUBSTR(MAX({main_cpv}), 1, 4)
                AS TARGET_CPV4,

            CASE
                WHEN COUNT(DISTINCT {buyer_key})
                     FILTER (WHERE {buyer_key} IS NOT NULL) = 1
                THEN MAX({buyer_key})
                ELSE NULL
            END AS TARGET_BUYER_KEY,

            CASE
                WHEN COUNT(DISTINCT {valid_country('ISO_COUNTRY_CODE')})
                     FILTER (
                        WHERE {valid_country('ISO_COUNTRY_CODE')} IS NOT NULL
                     ) = 1
                THEN MAX({valid_country('ISO_COUNTRY_CODE')})
                ELSE NULL
            END AS TARGET_PROCUREMENT_COUNTRY,

            {', '.join(regime_selects)},

            MEDIAN({numeric_expr('LOTS_NUMBER')})
                FILTER (
                    WHERE {numeric_expr('LOTS_NUMBER')} IS NOT NULL
                )
                AS TARGET_LOTS_NUMBER,

            {single_text_case_expr('LOTS_SUBMISSION')}
                AS TARGET_LOTS_SUBMISSION,

            {single_bool_case_expr('B_VARIANTS')}
                AS TARGET_B_VARIANTS,

            {single_bool_case_expr('B_OPTIONS')}
                AS TARGET_B_OPTIONS,

            {single_bool_case_expr('B_RENEWALS')}
                AS TARGET_B_RENEWALS,

            MEDIAN({numeric_expr('DURATION')})
                FILTER (
                    WHERE {numeric_expr('DURATION')} IS NOT NULL
                )
                AS TARGET_DURATION,

            MIN({date_expr('CONTRACT_START')})
                AS TARGET_CONTRACT_START,

            MAX({date_expr('CONTRACT_COMPLETION')})
                AS TARGET_CONTRACT_COMPLETION,

            MIN({date_expr('DT_APPLICATIONS')})
                AS TARGET_APPLICATION_DATE,

            DATE_DIFF(
                'day',
                MIN({date_expr('DT_DISPATCH')}),
                MIN({date_expr('DT_APPLICATIONS')})
            ) AS TARGET_APPLICATION_LEAD_DAYS,

            {single_text_case_expr('CRIT_CODE')}
                AS TARGET_CRIT_CODE,

            MEDIAN({numeric_expr('CRIT_PRICE_WEIGHT')})
                FILTER (
                    WHERE {numeric_expr('CRIT_PRICE_WEIGHT')} IS NOT NULL
                )
                AS TARGET_CRIT_PRICE_WEIGHT,

            {single_bool_case_expr('B_RECURRENT_PROCUREMENT')}
                AS TARGET_B_RECURRENT_PROCUREMENT,

            MEDIAN({numeric_expr('ENV_OPERATORS')})
                FILTER (
                    WHERE {numeric_expr('ENV_OPERATORS')} IS NOT NULL
                )
                AS TARGET_ENV_OPERATORS,

            MEDIAN({numeric_expr('ENV_MIN_OPERATORS')})
                FILTER (
                    WHERE {numeric_expr('ENV_MIN_OPERATORS')} IS NOT NULL
                )
                AS TARGET_ENV_MIN_OPERATORS,

            MEDIAN({numeric_expr('ENV_MAX_OPERATORS')})
                FILTER (
                    WHERE {numeric_expr('ENV_MAX_OPERATORS')} IS NOT NULL
                )
                AS TARGET_ENV_MAX_OPERATORS,

            MAX({target_value})
                AS TARGET_VALUE_EURO

        FROM cn_primary_raw
        GROUP BY TARGET_CASE_ID
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir / "e1x_target_profiles_2017.parquet",
    )


def build_target_cpv_tokens(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    query = f"""
        WITH raw_codes AS (
            SELECT
                {clean_expr('ID_NOTICE_CN')}
                    AS TARGET_CASE_ID,
                {valid_cpv_expr('CPV')}
                    AS CPV_CODE,
                'MAIN' AS CODE_SOURCE
            FROM cn_primary_raw

            UNION ALL

            SELECT
                {clean_expr('c.ID_NOTICE_CN')}
                    AS TARGET_CASE_ID,
                {valid_cpv_expr('tok.code')}
                    AS CPV_CODE,
                'ADDITIONAL' AS CODE_SOURCE
            FROM cn_primary_raw c,
            UNNEST(
                STR_SPLIT(
                    COALESCE(CAST(c.ADDITIONAL_CPVS AS VARCHAR), ''),
                    '---'
                )
            ) AS tok(code)
        ),
        valid AS (
            SELECT DISTINCT
                TARGET_CASE_ID,
                CPV_CODE,
                CODE_SOURCE
            FROM raw_codes
            WHERE CPV_CODE IS NOT NULL
        )
        SELECT
            TARGET_CASE_ID,
            CPV_CODE,
            CODE_SOURCE,
            SUBSTR(CPV_CODE, 1, 2) AS CPV2,
            SUBSTR(CPV_CODE, 1, 3) AS CPV3,
            SUBSTR(CPV_CODE, 1, 4) AS CPV4
        FROM valid
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir / "e1x_target_cpv_tokens_2017.parquet",
    )


def build_target_nuts_tokens(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    cleaned = nuts_clean_expr("tok.code")

    query = f"""
        WITH exploded AS (
            SELECT
                {clean_expr('c.ID_NOTICE_CN')}
                    AS TARGET_CASE_ID,
                {cleaned}
                    AS NUTS_CODE
            FROM cn_primary_raw c,
            UNNEST(
                REGEXP_SPLIT_TO_ARRAY(
                    COALESCE(CAST(c.TAL_LOCATION_NUTS AS VARCHAR), ''),
                    '[,;|]+'
                )
            ) AS tok(code)
        ),
        valid AS (
            SELECT DISTINCT
                TARGET_CASE_ID,
                NUTS_CODE
            FROM exploded
            WHERE REGEXP_MATCHES(
                NUTS_CODE,
                '^[A-Z]{{2}}[A-Z0-9]{{1,3}}$'
            )
        )
        SELECT
            TARGET_CASE_ID,
            NUTS_CODE,
            CASE
                WHEN LENGTH(NUTS_CODE) >= 3
                THEN SUBSTR(NUTS_CODE, 1, 3)
            END AS NUTS1,
            CASE
                WHEN LENGTH(NUTS_CODE) >= 4
                THEN SUBSTR(NUTS_CODE, 1, 4)
            END AS NUTS2,
            CASE
                WHEN LENGTH(NUTS_CODE) >= 5
                THEN SUBSTR(NUTS_CODE, 1, 5)
            END AS NUTS3
        FROM valid
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir / "e1x_target_nuts_tokens_2017.parquet",
    )


def build_target_criteria(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    query = f"""
        SELECT DISTINCT
            {clean_expr('ID_NOTICE_CN')}
                AS TARGET_CASE_ID,

            {upper_clean('CRIT_CODE')}
                AS CRIT_CODE,

            {numeric_expr('CRIT_PRICE_WEIGHT')}
                AS CRIT_PRICE_WEIGHT,

            {clean_expr('CRIT_CRITERIA')}
                AS CRIT_CRITERIA,

            {clean_expr('CRIT_WEIGHTS')}
                AS CRIT_WEIGHTS

        FROM cn_primary_raw
        WHERE
            {clean_expr('CRIT_CODE')} IS NOT NULL
            OR {clean_expr('CRIT_PRICE_WEIGHT')} IS NOT NULL
            OR {clean_expr('CRIT_CRITERIA')} IS NOT NULL
            OR {clean_expr('CRIT_WEIGHTS')} IS NOT NULL
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir / "e1x_target_criteria_2017.parquet",
    )


# ---------------------------------------------------------------------
# Historical CPV / NUTS indices
# ---------------------------------------------------------------------

def build_supplier_cpv_index(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    query = f"""
        WITH raw_codes AS (
            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                {valid_cpv_expr('CPV')}
                    AS CPV_CODE
            FROM can_hist_mapped_raw

            UNION ALL

            SELECT
                c.AWARD_KEY,
                c.SUPPLIER_ENTITY_ID,
                {valid_cpv_expr('tok.code')}
                    AS CPV_CODE
            FROM can_hist_mapped_raw c,
            UNNEST(
                STR_SPLIT(
                    COALESCE(CAST(c.ADDITIONAL_CPVS AS VARCHAR), ''),
                    '---'
                )
            ) AS tok(code)
        ),
        award_codes AS (
            SELECT DISTINCT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                CPV_CODE
            FROM raw_codes
            WHERE CPV_CODE IS NOT NULL
        ),
        levels AS (
            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                'EXACT' AS CPV_LEVEL,
                CPV_CODE AS FIT_VALUE
            FROM award_codes

            UNION ALL

            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                'CPV4' AS CPV_LEVEL,
                SUBSTR(CPV_CODE, 1, 4) AS FIT_VALUE
            FROM award_codes

            UNION ALL

            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                'CPV3' AS CPV_LEVEL,
                SUBSTR(CPV_CODE, 1, 3) AS FIT_VALUE
            FROM award_codes

            UNION ALL

            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                'CPV2' AS CPV_LEVEL,
                SUBSTR(CPV_CODE, 1, 2) AS FIT_VALUE
            FROM award_codes
        ),
        dedup AS (
            SELECT DISTINCT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                CPV_LEVEL,
                FIT_VALUE
            FROM levels
        )
        SELECT
            CPV_LEVEL,
            FIT_VALUE,
            SUPPLIER_ENTITY_ID,
            COUNT(*)::BIGINT AS N_MATCHED_AWARDS
        FROM dedup
        GROUP BY
            CPV_LEVEL,
            FIT_VALUE,
            SUPPLIER_ENTITY_ID
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir
        / "supplier_cpv_fit_counts_2015_2016.parquet",
    )


def build_supplier_nuts_index(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    cleaned = nuts_clean_expr("tok.code")

    query = f"""
        WITH exploded AS (
            SELECT
                c.AWARD_KEY,
                c.SUPPLIER_ENTITY_ID,
                {cleaned} AS NUTS_CODE
            FROM can_hist_mapped_raw c,
            UNNEST(
                REGEXP_SPLIT_TO_ARRAY(
                    COALESCE(CAST(c.TAL_LOCATION_NUTS AS VARCHAR), ''),
                    '[,;|]+'
                )
            ) AS tok(code)
        ),
        valid AS (
            SELECT DISTINCT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                NUTS_CODE
            FROM exploded
            WHERE REGEXP_MATCHES(
                NUTS_CODE,
                '^[A-Z]{{2}}[A-Z0-9]{{1,3}}$'
            )
        ),
        levels AS (
            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                'NUTS1' AS NUTS_LEVEL,
                SUBSTR(NUTS_CODE, 1, 3) AS FIT_VALUE
            FROM valid
            WHERE LENGTH(NUTS_CODE) >= 3

            UNION ALL

            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                'NUTS2' AS NUTS_LEVEL,
                SUBSTR(NUTS_CODE, 1, 4) AS FIT_VALUE
            FROM valid
            WHERE LENGTH(NUTS_CODE) >= 4

            UNION ALL

            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                'NUTS3' AS NUTS_LEVEL,
                SUBSTR(NUTS_CODE, 1, 5) AS FIT_VALUE
            FROM valid
            WHERE LENGTH(NUTS_CODE) >= 5
        ),
        dedup AS (
            SELECT DISTINCT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                NUTS_LEVEL,
                FIT_VALUE
            FROM levels
        )
        SELECT
            NUTS_LEVEL,
            FIT_VALUE,
            SUPPLIER_ENTITY_ID,
            COUNT(*)::BIGINT AS N_MATCHED_AWARDS
        FROM dedup
        GROUP BY
            NUTS_LEVEL,
            FIT_VALUE,
            SUPPLIER_ENTITY_ID
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir
        / "supplier_nuts_fit_counts_2015_2016.parquet",
    )


# ---------------------------------------------------------------------
# Buyer relationships
# ---------------------------------------------------------------------

def build_buyer_relationships(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    buyer_key = buyer_key_sql(
        "ISO_COUNTRY_CODE",
        "CAE_NATIONALID",
        "CAE_NAME",
    )

    query = f"""
        WITH award_buyer AS (
            SELECT DISTINCT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                {buyer_key} AS BUYER_KEY,
                {date_expr('DT_AWARD')} AS AWARD_DATE
            FROM can_hist_mapped_raw
            WHERE {buyer_key} IS NOT NULL
        ),
        supplier_totals AS (
            SELECT
                SUPPLIER_ENTITY_ID,
                COUNT(DISTINCT AWARD_KEY)::BIGINT
                    AS N_SUPPLIER_HISTORICAL_AWARDS
            FROM can_hist_mapped_raw
            GROUP BY SUPPLIER_ENTITY_ID
        ),
        rel AS (
            SELECT
                BUYER_KEY,
                SUPPLIER_ENTITY_ID,
                COUNT(DISTINCT AWARD_KEY)::BIGINT
                    AS N_PRIOR_BUYER_AWARDS,
                MAX(AWARD_DATE)
                    AS LAST_PRIOR_BUYER_AWARD_DATE
            FROM award_buyer
            GROUP BY
                BUYER_KEY,
                SUPPLIER_ENTITY_ID
        )
        SELECT
            r.BUYER_KEY,
            r.SUPPLIER_ENTITY_ID,
            r.N_PRIOR_BUYER_AWARDS,
            r.LAST_PRIOR_BUYER_AWARD_DATE,
            t.N_SUPPLIER_HISTORICAL_AWARDS,
            r.N_PRIOR_BUYER_AWARDS::DOUBLE
                / NULLIF(
                    t.N_SUPPLIER_HISTORICAL_AWARDS,
                    0
                )
                AS BUYER_RELATIONSHIP_AWARD_SHARE
        FROM rel r
        JOIN supplier_totals t
          USING (SUPPLIER_ENTITY_ID)
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir
        / "supplier_buyer_relationships_2015_2016.parquet",
    )


# ---------------------------------------------------------------------
# Regime and criteria indices
# ---------------------------------------------------------------------

def build_regime_fit_index(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    unions = []

    for field in REGIME_FIELDS:
        parsed = bool_expr(qident(field))
        unions.append(
            f"""
            SELECT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                '{field}' AS DIMENSION,
                CAST({parsed} AS VARCHAR) AS FIT_VALUE
            FROM can_hist_mapped_raw
            WHERE {parsed} IS NOT NULL
            """
        )

    union_sql = "\nUNION ALL\n".join(unions)

    query = f"""
        WITH long AS (
            {union_sql}
        ),
        dedup AS (
            SELECT DISTINCT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                DIMENSION,
                FIT_VALUE
            FROM long
        )
        SELECT
            DIMENSION,
            FIT_VALUE,
            SUPPLIER_ENTITY_ID,
            COUNT(*)::BIGINT AS N_MATCHED_AWARDS
        FROM dedup
        GROUP BY
            DIMENSION,
            FIT_VALUE,
            SUPPLIER_ENTITY_ID
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir
        / "supplier_regime_fit_counts_2015_2016.parquet",
    )


def build_criterion_code_index(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    query = f"""
        WITH award_codes AS (
            SELECT DISTINCT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,
                {upper_clean('CRIT_CODE')}
                    AS CRIT_CODE
            FROM can_hist_mapped_raw
            WHERE {upper_clean('CRIT_CODE')} IS NOT NULL
        )
        SELECT
            CRIT_CODE AS FIT_VALUE,
            SUPPLIER_ENTITY_ID,
            COUNT(*)::BIGINT AS N_MATCHED_AWARDS
        FROM award_codes
        GROUP BY
            CRIT_CODE,
            SUPPLIER_ENTITY_ID
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir
        / "supplier_criterion_code_fit_counts_2015_2016.parquet",
    )


# ---------------------------------------------------------------------
# Historical CPV2 context descriptors
# ---------------------------------------------------------------------

def build_cpv2_context_descriptors(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    main_cpv = valid_cpv_expr("CPV")

    award_value = (
        "COALESCE("
        f"{numeric_expr('AWARD_VALUE_EURO')}, "
        f"{numeric_expr('AWARD_VALUE_EURO_FIN_1')}, "
        f"{numeric_expr('AWARD_EST_VALUE_EURO')}"
        ")"
    )

    boolean_metric_exprs = []

    for field in HISTORICAL_BOOLEAN_FIELDS:
        parsed = bool_expr(qident(field))
        alias = field.removeprefix("B_")
        boolean_metric_exprs.extend(
            [
                (
                    f"AVG(CAST({parsed} AS DOUBLE)) "
                    f"FILTER (WHERE {parsed} IS NOT NULL) "
                    f"AS SHARE_{alias}_TRUE"
                ),
                (
                    f"COUNT(*) FILTER (WHERE {parsed} IS NOT NULL) "
                    f"AS N_{alias}_OBS"
                ),
            ]
        )

    bool_sql = ",\n            ".join(
        boolean_metric_exprs
    )

    query = f"""
        WITH award_rows AS (
            SELECT DISTINCT
                AWARD_KEY,
                SUPPLIER_ENTITY_ID,

                CASE
                    WHEN {main_cpv} IS NOT NULL
                    THEN SUBSTR({main_cpv}, 1, 2)
                END AS CPV2,

                {numeric_expr('NUMBER_OFFERS')}
                    AS NUMBER_OFFERS_NUM,

                {numeric_expr('LOTS_NUMBER')}
                    AS LOTS_NUMBER_NUM,

                {numeric_expr('CRIT_PRICE_WEIGHT')}
                    AS CRIT_PRICE_WEIGHT_NUM,

                {award_value}
                    AS AWARD_VALUE_NUM,

                {', '.join(
                    f'{bool_expr(qident(field))} AS {qident(field + "_BOOL")}'
                    for field in HISTORICAL_BOOLEAN_FIELDS
                )}

            FROM can_hist_mapped_raw
        ),
        normalized AS (
            SELECT
                *,
                {', '.join(
                    f'{qident(field + "_BOOL")} AS {qident(field)}'
                    for field in HISTORICAL_BOOLEAN_FIELDS
                )}
            FROM award_rows
            WHERE CPV2 IS NOT NULL
        )
        SELECT
            CPV2 AS CONTEXT_KEY,
            SUPPLIER_ENTITY_ID,

            COUNT(*)::BIGINT
                AS N_CONTEXT_AWARDS,

            MEDIAN(NUMBER_OFFERS_NUM)
                FILTER (WHERE NUMBER_OFFERS_NUM IS NOT NULL)
                AS MEDIAN_NUMBER_OFFERS,

            QUANTILE_CONT(NUMBER_OFFERS_NUM, 0.75)
                FILTER (WHERE NUMBER_OFFERS_NUM IS NOT NULL)
                AS P75_NUMBER_OFFERS,

            COUNT(NUMBER_OFFERS_NUM)
                AS N_NUMBER_OFFERS_OBS,

            MEDIAN(LOTS_NUMBER_NUM)
                FILTER (WHERE LOTS_NUMBER_NUM IS NOT NULL)
                AS MEDIAN_LOTS_NUMBER,

            COUNT(LOTS_NUMBER_NUM)
                AS N_LOTS_NUMBER_OBS,

            MEDIAN(CRIT_PRICE_WEIGHT_NUM)
                FILTER (WHERE CRIT_PRICE_WEIGHT_NUM IS NOT NULL)
                AS MEDIAN_CRIT_PRICE_WEIGHT,

            COUNT(CRIT_PRICE_WEIGHT_NUM)
                AS N_CRIT_PRICE_WEIGHT_OBS,

            MEDIAN(AWARD_VALUE_NUM)
                FILTER (WHERE AWARD_VALUE_NUM > 0)
                AS MEDIAN_AWARD_VALUE_EURO,

            QUANTILE_CONT(AWARD_VALUE_NUM, 0.75)
                FILTER (WHERE AWARD_VALUE_NUM > 0)
                AS P75_AWARD_VALUE_EURO,

            QUANTILE_CONT(AWARD_VALUE_NUM, 0.90)
                FILTER (WHERE AWARD_VALUE_NUM > 0)
                AS P90_AWARD_VALUE_EURO,

            MAX(AWARD_VALUE_NUM)
                FILTER (WHERE AWARD_VALUE_NUM > 0)
                AS MAX_AWARD_VALUE_EURO,

            COUNT(AWARD_VALUE_NUM)
                FILTER (WHERE AWARD_VALUE_NUM > 0)
                AS N_AWARD_VALUE_OBS,

            {bool_sql}

        FROM normalized
        GROUP BY
            CONTEXT_KEY,
            SUPPLIER_ENTITY_ID
    """

    copy_query_to_parquet(
        con,
        query,
        output_dir
        / "supplier_cpv2_context_descriptors_2015_2016.parquet",
    )


# ---------------------------------------------------------------------
# Prevalence reports
# ---------------------------------------------------------------------

def target_boolean_prevalence(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []
    total_cases = int(
        scalar(
            con,
            "SELECT COUNT(*) FROM e1_primary_cases",
        )
    )

    for field in TARGET_BOOLEAN_FIELDS:
        parsed = bool_expr(qident(field))

        n_non_null, n_true, n_false = con.execute(
            f"""
            WITH per_case AS (
                SELECT
                    {clean_expr('ID_NOTICE_CN')}
                        AS TARGET_CASE_ID,
                    CASE
                        WHEN COUNT(DISTINCT {parsed})
                             FILTER (WHERE {parsed} IS NOT NULL) = 1
                        THEN MAX({parsed})
                        ELSE NULL
                    END AS VALUE
                FROM cn_primary_raw
                GROUP BY TARGET_CASE_ID
            )
            SELECT
                COUNT(VALUE),
                COUNT(*) FILTER (WHERE VALUE IS TRUE),
                COUNT(*) FILTER (WHERE VALUE IS FALSE)
            FROM per_case
            """
        ).fetchone()

        n_non_null = int(n_non_null)
        n_true = int(n_true)
        n_false = int(n_false)

        rows.append(
            {
                "FIELD": field,
                "N_TARGET_CASES": total_cases,
                "N_NON_NULL_CASES": n_non_null,
                "COVERAGE_PCT":
                    100.0 * n_non_null / total_cases
                    if total_cases else 0.0,
                "N_TRUE": n_true,
                "N_FALSE": n_false,
                "TRUE_PCT_OF_NON_NULL":
                    100.0 * n_true / n_non_null
                    if n_non_null else None,
            }
        )

    return pd.DataFrame(rows)


def historical_boolean_prevalence(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []

    total_awards = int(
        scalar(
            con,
            """
            SELECT COUNT(DISTINCT AWARD_KEY)
            FROM can_hist_mapped_raw
            """,
        )
    )

    for field in HISTORICAL_BOOLEAN_FIELDS:
        parsed = bool_expr(qident(field))

        n_non_null, n_true, n_false = con.execute(
            f"""
            WITH per_award AS (
                SELECT
                    AWARD_KEY,
                    CASE
                        WHEN COUNT(DISTINCT {parsed})
                             FILTER (WHERE {parsed} IS NOT NULL) = 1
                        THEN MAX({parsed})
                        ELSE NULL
                    END AS VALUE
                FROM can_hist_mapped_raw
                GROUP BY AWARD_KEY
            )
            SELECT
                COUNT(VALUE),
                COUNT(*) FILTER (WHERE VALUE IS TRUE),
                COUNT(*) FILTER (WHERE VALUE IS FALSE)
            FROM per_award
            """
        ).fetchone()

        n_non_null = int(n_non_null)
        n_true = int(n_true)
        n_false = int(n_false)

        rows.append(
            {
                "FIELD": field,
                "N_HISTORICAL_AWARDS": total_awards,
                "N_NON_NULL_AWARDS": n_non_null,
                "COVERAGE_PCT":
                    100.0 * n_non_null / total_awards
                    if total_awards else 0.0,
                "N_TRUE": n_true,
                "N_FALSE": n_false,
                "TRUE_PCT_OF_NON_NULL":
                    100.0 * n_true / n_non_null
                    if n_non_null else None,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Metrics and validation
# ---------------------------------------------------------------------

OUTPUT_SPECS = [
    (
        "TARGET_PROFILE",
        "e1x_target_profiles_2017.parquet",
        ["TARGET_CASE_ID"],
    ),
    (
        "TARGET_CPV_TOKENS",
        "e1x_target_cpv_tokens_2017.parquet",
        ["TARGET_CASE_ID", "CPV_CODE", "CODE_SOURCE"],
    ),
    (
        "TARGET_NUTS_TOKENS",
        "e1x_target_nuts_tokens_2017.parquet",
        ["TARGET_CASE_ID", "NUTS_CODE"],
    ),
    (
        "TARGET_CRITERIA",
        "e1x_target_criteria_2017.parquet",
        [
            "TARGET_CASE_ID",
            "CRIT_CODE",
            "CRIT_PRICE_WEIGHT",
            "CRIT_CRITERIA",
            "CRIT_WEIGHTS",
        ],
    ),
    (
        "SUPPLIER_CPV_FIT",
        "supplier_cpv_fit_counts_2015_2016.parquet",
        ["CPV_LEVEL", "FIT_VALUE", "SUPPLIER_ENTITY_ID"],
    ),
    (
        "SUPPLIER_NUTS_FIT",
        "supplier_nuts_fit_counts_2015_2016.parquet",
        ["NUTS_LEVEL", "FIT_VALUE", "SUPPLIER_ENTITY_ID"],
    ),
    (
        "SUPPLIER_BUYER_RELATIONSHIPS",
        "supplier_buyer_relationships_2015_2016.parquet",
        ["BUYER_KEY", "SUPPLIER_ENTITY_ID"],
    ),
    (
        "SUPPLIER_REGIME_FIT",
        "supplier_regime_fit_counts_2015_2016.parquet",
        ["DIMENSION", "FIT_VALUE", "SUPPLIER_ENTITY_ID"],
    ),
    (
        "SUPPLIER_CRITERION_CODE_FIT",
        "supplier_criterion_code_fit_counts_2015_2016.parquet",
        ["FIT_VALUE", "SUPPLIER_ENTITY_ID"],
    ),
    (
        "SUPPLIER_CPV2_DESCRIPTORS",
        "supplier_cpv2_context_descriptors_2015_2016.parquet",
        ["CONTEXT_KEY", "SUPPLIER_ENTITY_ID"],
    ),
]


def output_metrics(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []

    for name, filename, _keys in OUTPUT_SPECS:
        path = output_dir / filename

        n_rows = int(
            scalar(
                con,
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_path(path)}')
                """,
            )
        )

        rows.append(
            {
                "OUTPUT_NAME": name,
                "FILE": filename,
                "N_ROWS": n_rows,
            }
        )

    return pd.DataFrame(rows)


def validation_checks(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []

    n_e1 = int(
        scalar(
            con,
            "SELECT COUNT(*) FROM e1_primary_cases",
        )
    )

    profile_path = (
        output_dir
        / "e1x_target_profiles_2017.parquet"
    )

    n_profile = int(
        scalar(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_path(profile_path)}')
            """,
        )
    )

    n_profile_unique = int(
        scalar(
            con,
            f"""
            SELECT COUNT(DISTINCT TARGET_CASE_ID)
            FROM read_parquet('{sql_path(profile_path)}')
            """,
        )
    )

    rows.append(
        {
            "CHECK": "target_profile_matches_primary_cohort",
            "VALUE":
                n_profile == n_e1
                and n_profile_unique == n_e1,
            "DETAIL":
                (
                    f"e1_cases={n_e1}; "
                    f"profile_rows={n_profile}; "
                    f"profile_unique={n_profile_unique}"
                ),
        }
    )

    for name, filename, keys in OUTPUT_SPECS:
        path = output_dir / filename

        key_sql = ", ".join(qident(k) for k in keys)

        dup_groups = int(
            scalar(
                con,
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT
                        {key_sql},
                        COUNT(*) AS N
                    FROM read_parquet('{sql_path(path)}')
                    GROUP BY {key_sql}
                    HAVING COUNT(*) > 1
                )
                """,
            )
        )

        rows.append(
            {
                "CHECK": f"{name}__unique_key",
                "VALUE": dup_groups == 0,
                "DETAIL":
                    f"duplicate_key_groups={dup_groups}",
            }
        )

    years = [
        int(r[0])
        for r in con.execute(
            """
            SELECT DISTINCT SOURCE_YEAR
            FROM can_hist_mapped_raw
            ORDER BY SOURCE_YEAR
            """
        ).fetchall()
    ]

    rows.append(
        {
            "CHECK": "historical_years_are_2015_2016_only",
            "VALUE": years == [2015, 2016],
            "DETAIL": str(years),
        }
    )

    return pd.DataFrame(rows)


def write_report(
    target_prev: pd.DataFrame,
    hist_prev: pd.DataFrame,
    metrics_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Extended Procurement-Fit Feature Build Report",
        "",
        "## Scope",
        "",
        "The build combines 2017 CN target information with frozen 2015-2016 "
        "CAN supplier history. It creates compact target tables and historical "
        "supplier indices without constructing a case-by-supplier Cartesian product.",
        "",
        "## Output sizes",
        "",
        "| Output | Rows |",
        "|---|---:|",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| `{row['OUTPUT_NAME']}` "
            f"| {int(row['N_ROWS'])} |"
        )

    lines.extend(
        [
            "",
            "## Target boolean prevalence",
            "",
            "| Field | Coverage (%) | True among observed (%) |",
            "|---|---:|---:|",
        ]
    )

    for _, row in target_prev.iterrows():
        true_pct = (
            f"{row['TRUE_PCT_OF_NON_NULL']:.3f}"
            if pd.notna(row["TRUE_PCT_OF_NON_NULL"])
            else "NA"
        )

        lines.append(
            f"| `{row['FIELD']}` "
            f"| {row['COVERAGE_PCT']:.3f} "
            f"| {true_pct} |"
        )

    lines.extend(
        [
            "",
            "## Historical boolean prevalence",
            "",
            "| Field | Award coverage (%) | True among observed (%) |",
            "|---|---:|---:|",
        ]
    )

    for _, row in hist_prev.iterrows():
        true_pct = (
            f"{row['TRUE_PCT_OF_NON_NULL']:.3f}"
            if pd.notna(row["TRUE_PCT_OF_NON_NULL"])
            else "NA"
        )

        lines.append(
            f"| `{row['FIELD']}` "
            f"| {row['COVERAGE_PCT']:.3f} "
            f"| {true_pct} |"
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
            f"- **{status}** `{row['CHECK']}`: {row['DETAIL']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The prevalence tables should be used before adding regime flags to a "
            "ranking model. High coverage does not imply useful variation. Near-constant "
            "flags should be excluded from the main model or used only in targeted "
            "scenario analyses.",
            "",
            "Historical competition, SME, consortium, and subcontracting statistics "
            "are descriptive context variables and must not be interpreted as universal "
            "supplier-quality scores.",
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
        description="Build extended procurement-fit feature indices."
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
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/e1x_extended_fit_indices"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "reports/e1x_extended_fit_indices"
        ),
    )
    parser.add_argument(
        "--work-db",
        type=Path,
        default=None,
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
    )
    parser.add_argument(
        "--temp-directory",
        type=Path,
        default=None,
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
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
        print("1/11 Validating inputs...")
        (
            cn_cols,
            can15_cols,
            can16_cols,
        ) = validate_inputs(
            con,
            args,
        )

        print("2/11 Materializing cohort and history...")
        create_base_tables(
            con,
            args,
            cn_cols,
            can15_cols,
            can16_cols,
        )

        print("3/11 Building target profile and target tokens...")
        build_target_profile(
            con,
            args.output_dir,
        )
        build_target_cpv_tokens(
            con,
            args.output_dir,
        )
        build_target_nuts_tokens(
            con,
            args.output_dir,
        )
        build_target_criteria(
            con,
            args.output_dir,
        )

        print("4/11 Building supplier CPV fit index...")
        build_supplier_cpv_index(
            con,
            args.output_dir,
        )

        print("5/11 Building supplier NUTS fit index...")
        build_supplier_nuts_index(
            con,
            args.output_dir,
        )

        print("6/11 Building buyer relationship dynamics...")
        build_buyer_relationships(
            con,
            args.output_dir,
        )

        print("7/11 Building procurement-regime fit index...")
        build_regime_fit_index(
            con,
            args.output_dir,
        )

        print("8/11 Building criterion-code fit index...")
        build_criterion_code_index(
            con,
            args.output_dir,
        )

        print("9/11 Building CPV2 historical context descriptors...")
        build_cpv2_context_descriptors(
            con,
            args.output_dir,
        )

        print("10/11 Computing prevalence and validation reports...")
        target_prev = target_boolean_prevalence(
            con
        )
        hist_prev = historical_boolean_prevalence(
            con
        )
        metrics_df = output_metrics(
            con,
            args.output_dir,
        )
        validation_df = validation_checks(
            con,
            args.output_dir,
        )

        print("11/11 Writing reports...")
        target_prev.to_csv(
            args.report_dir
            / "e1x_target_boolean_prevalence.csv",
            index=False,
        )

        hist_prev.to_csv(
            args.report_dir
            / "e1x_historical_boolean_prevalence.csv",
            index=False,
        )

        metrics_df.to_csv(
            args.report_dir
            / "e1x_output_metrics.csv",
            index=False,
        )

        validation_df.to_csv(
            args.report_dir
            / "e1x_validation_checks.csv",
            index=False,
        )

        write_report(
            target_prev,
            hist_prev,
            metrics_df,
            validation_df,
            args.report_dir
            / "e1x_extended_feature_build_report.md",
        )

        print()
        print("Extended feature-index build complete.")
        print("Review first:")
        print("  e1x_extended_feature_build_report.md")
        print("  e1x_target_boolean_prevalence.csv")
        print("  e1x_historical_boolean_prevalence.csv")
        print("  e1x_validation_checks.csv")

    finally:
        con.close()


if __name__ == "__main__":
    main()
