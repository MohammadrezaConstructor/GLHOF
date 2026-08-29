#!/usr/bin/env python3
"""
Build normalized TED CN 2017 tables, exact CN-CAN pair links, and 2017 target-case cohorts.

Inputs
------
1. Raw CN 2017 CSV.
2. award_fact_2015_2017.parquet produced by build_can_award_tables.py.

Outputs
-------
Processed:
- cn_notice_2017.parquet
- cn_criteria_2017.parquet
- cn_lot_context_2017.parquet
- cn_provenance_2017.parquet
- cn_can_links_2017.parquet
- procurement_cases_2017.parquet
- cn_notice_conflicts_2017.parquet
- cn_link_parse_review_2017.parquet

Reports:
- target_case_funnel.csv
- cohort_classification_summary.csv
- link_estimation_status_summary.csv
- cn_to_can_pair_cardinality.csv
- can_to_cn_pair_cardinality.csv
- linked_can_award_multiplicity.csv
- criteria_notice_coverage.csv
- notice_quality_summary.csv
- cn_target_build_metrics.csv
- cn_target_build_report.md

Key methodological rules
------------------------
- One target case is not a raw CN row.
- CN rows are normalized to one notice table plus criteria/lot child tables.
- CN-CAN links are built at exact pair grain: (ID_NOTICE_CN, ID_NOTICE_CAN).
- FUTURE_CAN_ID is never collapsed with MAX() before linkage.
- Triple-dash lists are split deterministically; unsupported delimiters are flagged.
- CN link cardinality is classified before target-case selection.
- The first structural cohort is:
      valid 2017 CN core
      + one future CAN ID
      + CAN present in 2017 award fact
      + one CAN award key
      + one observed winner
      + not group/list-valued winner structure.
- Estimated-link status is preserved separately. The script reports both:
      structural strict cohort,
      strict cohort excluding explicitly estimated links,
      strict cohort with explicitly confirmed non-estimated links.
- This script does not map 2017 winners to the 2015-2016 supplier dimension.

Requires
--------
pip install duckdb pandas pyarrow
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import duckdb
import pandas as pd


# -----------------------------------------------------------------------------
# Expected CN schema
# -----------------------------------------------------------------------------

EXPECTED_CN_COLUMNS = [
    "ID_NOTICE_CN", "TED_NOTICE_URL", "YEAR", "ID_TYPE", "DT_DISPATCH",
    "XSD_VERSION", "CANCELLED", "CORRECTIONS", "FUTURE_CAN_ID",
    "FUTURE_CAN_ID_ESTIMATED", "B_MULTIPLE_CAE", "CAE_NAME",
    "CAE_NATIONALID", "CAE_ADDRESS", "CAE_TOWN", "CAE_POSTAL_CODE",
    "ISO_COUNTRY_CODE", "B_MULTIPLE_COUNTRY", "ISO_COUNTRY_CODE_ALL",
    "CAE_TYPE", "EU_INST_CODE", "MAIN_ACTIVITY", "B_ON_BEHALF",
    "B_INVOLVES_JOINT_PROCUREMENT", "B_AWARDED_BY_CENTRAL_BODY",
    "TYPE_OF_CONTRACT", "TAL_LOCATION_NUTS", "B_FRA_AGREEMENT",
    "FRA_ESTIMATED", "B_FRA_SINGLE_OPERATOR", "FRA_NUMBER_OPERATORS",
    "FRA_NUMBER_MAX_OPERATORS", "B_DYN_PURCH_SYST", "CPV", "ID_LOT",
    "ADDITIONAL_CPVS", "B_GPA", "LOTS_NUMBER", "LOTS_SUBMISSION",
    "B_VARIANTS", "VALUE_EURO", "VALUE_EURO_FIN_1", "VALUE_EURO_FIN_2",
    "B_OPTIONS", "B_EU_FUNDS", "B_RENEWALS", "DURATION", "CONTRACT_START",
    "CONTRACT_COMPLETION", "TOP_TYPE", "B_ACCELERATED", "ENV_OPERATORS",
    "ENV_MIN_OPERATORS", "ENV_MAX_OPERATORS", "CRIT_CODE",
    "CRIT_PRICE_WEIGHT", "CRIT_CRITERIA", "CRIT_WEIGHTS",
    "B_ELECTRONIC_AUCTION", "DT_APPLICATIONS", "B_LANGUAGE_ANY_EC",
    "ADMIN_LANGUAGES_TENDER", "ADMIN_OTHER_LANGUAGES_TENDER",
    "B_RECURRENT_PROCUREMENT",
]

REQUIRED_CAN_COLUMNS = {
    "ID_NOTICE_CAN_CLEAN",
    "AWARD_KEY",
    "SOURCE_YEAR_MIN",
    "SOURCE_YEAR_MAX",
    "DT_DISPATCH_CLEAN",
    "DT_AWARD_CLEAN",
    "WIN_NAME_CLEAN",
    "WIN_NATIONALID_CLEAN",
    "WIN_COUNTRY_CODE_CLEAN",
    "B_AWARDED_TO_A_GROUP_CLEAN",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def clean_text_sql(column: str) -> str:
    c = qident(column)
    return f"NULLIF(TRIM(CAST({c} AS VARCHAR)), '')"


def numeric_sql(column: str) -> str:
    return f"TRY_CAST({clean_text_sql(column)} AS DOUBLE)"


def boolean_sql(column: str) -> str:
    c = f"UPPER({clean_text_sql(column)})"
    return f"""
    CASE
        WHEN {c} IN ('1', 'Y', 'YES', 'TRUE', 'T') THEN TRUE
        WHEN {c} IN ('0', 'N', 'NO', 'FALSE', 'F') THEN FALSE
        ELSE NULL
    END
    """


def date_sql(column: str) -> str:
    c = clean_text_sql(column)
    return f"""
    CAST(
        COALESCE(
            TRY_STRPTIME({c}, '%d-%b-%y'),
            TRY_STRPTIME({c}, '%d/%m/%y'),
            TRY_STRPTIME({c}, '%Y-%m-%d'),
            TRY_STRPTIME({c}, '%d/%m/%Y'),
            TRY_STRPTIME({c}, '%Y%m%d'),
            TRY_STRPTIME({c}, '%d-%m-%Y')
        ) AS DATE
    )
    """


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(131072)
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
    except csv.Error:
        return "\t"


def read_header(path: Path, delimiter: str) -> list[str]:
    with path.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as f:
        return next(csv.reader(f, delimiter=delimiter))


def csv_relation(path: Path, delimiter: str) -> str:
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


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def copy_table_to_parquet(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    con.execute(
        f"""
        COPY {qident(table_name)}
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------

def validate_cn_schema(path: Path, delimiter: str) -> None:
    actual = read_header(path, delimiter)
    if actual == EXPECTED_CN_COLUMNS:
        return

    expected_set = set(EXPECTED_CN_COLUMNS)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    same_set = expected_set == actual_set

    raise ValueError(
        "CN 2017 schema mismatch.\n"
        f"  Actual columns: {len(actual)}\n"
        f"  Expected columns: {len(EXPECTED_CN_COLUMNS)}\n"
        f"  Same set, different order: {same_set and actual != EXPECTED_CN_COLUMNS}\n"
        f"  Missing: {missing}\n"
        f"  Extra: {extra}"
    )


def validate_can_schema(
    con: duckdb.DuckDBPyConnection,
    can_award_fact: Path,
) -> None:
    cols = {
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE SELECT *
            FROM read_parquet('{sql_path(can_award_fact)}')
            """
        ).fetchall()
    }
    missing = sorted(REQUIRED_CAN_COLUMNS - cols)
    if missing:
        raise ValueError(
            "CAN award fact missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


# -----------------------------------------------------------------------------
# Raw load and cleaned CN rows
# -----------------------------------------------------------------------------

def build_cn_rows_clean(
    con: duckdb.DuckDBPyConnection,
    cn_path: Path,
    delimiter: str,
) -> None:
    relation = csv_relation(cn_path, delimiter)

    con.execute("DROP TABLE IF EXISTS cn_raw")
    con.execute(
        f"""
        CREATE TABLE cn_raw AS
        SELECT
            *,
            ROW_NUMBER() OVER ()::BIGINT AS SOURCE_ROW_ID,
            '{cn_path.name.replace("'", "''")}'::VARCHAR AS SOURCE_FILE
        FROM {relation}
        """
    )

    con.execute("DROP TABLE IF EXISTS cn_rows_clean")
    con.execute(
        f"""
        CREATE TABLE cn_rows_clean AS
        SELECT
            SOURCE_ROW_ID,
            SOURCE_FILE,

            {clean_text_sql('ID_NOTICE_CN')} AS ID_NOTICE_CN_CLEAN,
            {clean_text_sql('TED_NOTICE_URL')} AS TED_NOTICE_URL_CLEAN,
            {numeric_sql('YEAR')} AS YEAR_CLEAN,
            {clean_text_sql('ID_TYPE')} AS ID_TYPE_CLEAN,
            {date_sql('DT_DISPATCH')} AS DT_DISPATCH_CLEAN,
            {clean_text_sql('XSD_VERSION')} AS XSD_VERSION_CLEAN,

            {clean_text_sql('CANCELLED')} AS CANCELLED_RAW,
            {boolean_sql('CANCELLED')} AS CANCELLED_CLEAN,
            {clean_text_sql('CORRECTIONS')} AS CORRECTIONS_RAW,
            {boolean_sql('CORRECTIONS')} AS CORRECTIONS_CLEAN,

            {clean_text_sql('FUTURE_CAN_ID')} AS FUTURE_CAN_ID_CLEAN,
            {clean_text_sql('FUTURE_CAN_ID_ESTIMATED')}
                AS FUTURE_CAN_ID_ESTIMATED_RAW,
            {boolean_sql('FUTURE_CAN_ID_ESTIMATED')}
                AS FUTURE_CAN_ID_ESTIMATED_CLEAN,

            {clean_text_sql('CAE_NAME')} AS CAE_NAME_CLEAN,
            {clean_text_sql('CAE_NATIONALID')} AS CAE_NATIONALID_CLEAN,
            {clean_text_sql('CAE_TOWN')} AS CAE_TOWN_CLEAN,
            {clean_text_sql('ISO_COUNTRY_CODE')} AS ISO_COUNTRY_CODE_CLEAN,
            {clean_text_sql('ISO_COUNTRY_CODE_ALL')}
                AS ISO_COUNTRY_CODE_ALL_CLEAN,
            {clean_text_sql('CAE_TYPE')} AS CAE_TYPE_CLEAN,
            {clean_text_sql('MAIN_ACTIVITY')} AS MAIN_ACTIVITY_CLEAN,

            {clean_text_sql('TYPE_OF_CONTRACT')} AS TYPE_OF_CONTRACT_CLEAN,
            {clean_text_sql('TAL_LOCATION_NUTS')} AS TAL_LOCATION_NUTS_CLEAN,
            {clean_text_sql('CPV')} AS CPV_CLEAN,
            {clean_text_sql('ADDITIONAL_CPVS')} AS ADDITIONAL_CPVS_CLEAN,
            {clean_text_sql('ID_LOT')} AS ID_LOT_CLEAN,
            {numeric_sql('LOTS_NUMBER')} AS LOTS_NUMBER_CLEAN,
            {clean_text_sql('LOTS_SUBMISSION')} AS LOTS_SUBMISSION_CLEAN,

            {numeric_sql('VALUE_EURO')} AS VALUE_EURO_CLEAN,
            {numeric_sql('VALUE_EURO_FIN_1')} AS VALUE_EURO_FIN_1_CLEAN,
            {numeric_sql('VALUE_EURO_FIN_2')} AS VALUE_EURO_FIN_2_CLEAN,

            {clean_text_sql('CRIT_CODE')} AS CRIT_CODE_CLEAN,
            {numeric_sql('CRIT_PRICE_WEIGHT')} AS CRIT_PRICE_WEIGHT_CLEAN,
            {clean_text_sql('CRIT_CRITERIA')} AS CRIT_CRITERIA_CLEAN,
            {clean_text_sql('CRIT_WEIGHTS')} AS CRIT_WEIGHTS_CLEAN,

            {date_sql('CONTRACT_START')} AS CONTRACT_START_CLEAN,
            {date_sql('CONTRACT_COMPLETION')} AS CONTRACT_COMPLETION_CLEAN,
            {date_sql('DT_APPLICATIONS')} AS DT_APPLICATIONS_CLEAN,

            {clean_text_sql('TOP_TYPE')} AS TOP_TYPE_CLEAN,
            {clean_text_sql('B_FRA_AGREEMENT')} AS B_FRA_AGREEMENT_RAW,
            {boolean_sql('B_FRA_AGREEMENT')} AS B_FRA_AGREEMENT_CLEAN,
            {clean_text_sql('B_DYN_PURCH_SYST')} AS B_DYN_PURCH_SYST_RAW,
            {boolean_sql('B_DYN_PURCH_SYST')} AS B_DYN_PURCH_SYST_CLEAN,
            {clean_text_sql('B_ELECTRONIC_AUCTION')}
                AS B_ELECTRONIC_AUCTION_RAW,
            {boolean_sql('B_ELECTRONIC_AUCTION')}
                AS B_ELECTRONIC_AUCTION_CLEAN,

            CASE
                WHEN POSITION('---' IN COALESCE({clean_text_sql('FUTURE_CAN_ID')}, '')) > 0
                    THEN 'split_triple_dash'
                WHEN POSITION(',' IN COALESCE({clean_text_sql('FUTURE_CAN_ID')}, '')) > 0
                  OR POSITION(';' IN COALESCE({clean_text_sql('FUTURE_CAN_ID')}, '')) > 0
                  OR POSITION('|' IN COALESCE({clean_text_sql('FUTURE_CAN_ID')}, '')) > 0
                    THEN 'unsupported_delimiter_review'
                WHEN {clean_text_sql('FUTURE_CAN_ID')} IS NOT NULL
                    THEN 'single_token'
                ELSE 'missing'
            END AS FUTURE_CAN_PARSE_STATUS

        FROM cn_raw
        """
    )


# -----------------------------------------------------------------------------
# Notice table and child tables
# -----------------------------------------------------------------------------

def build_cn_notice_table(
    con: duckdb.DuckDBPyConnection,
    target_year: int,
) -> None:
    con.execute("DROP TABLE IF EXISTS cn_notice")
    con.execute(
        f"""
        CREATE TABLE cn_notice AS
        SELECT
            ID_NOTICE_CN_CLEAN,

            COUNT(*) AS SOURCE_ROW_COUNT,

            MAX(TED_NOTICE_URL_CLEAN) AS TED_NOTICE_URL_CLEAN,
            MAX(YEAR_CLEAN) AS SOURCE_YEAR_REPORTED,
            MAX(ID_TYPE_CLEAN) AS ID_TYPE_CLEAN,
            MAX(XSD_VERSION_CLEAN) AS XSD_VERSION_CLEAN,

            MAX(DT_DISPATCH_CLEAN) AS DT_DISPATCH_CLEAN,
            COUNT(DISTINCT DT_DISPATCH_CLEAN) FILTER (
                WHERE DT_DISPATCH_CLEAN IS NOT NULL
            ) AS N_DISTINCT_DT_DISPATCH,

            MAX(CAE_NAME_CLEAN) AS CAE_NAME_CLEAN,
            MAX(CAE_NATIONALID_CLEAN) AS CAE_NATIONALID_CLEAN,
            MAX(CAE_TOWN_CLEAN) AS CAE_TOWN_CLEAN,
            MAX(ISO_COUNTRY_CODE_CLEAN) AS ISO_COUNTRY_CODE_CLEAN,
            MAX(ISO_COUNTRY_CODE_ALL_CLEAN) AS ISO_COUNTRY_CODE_ALL_CLEAN,
            MAX(CAE_TYPE_CLEAN) AS CAE_TYPE_CLEAN,
            MAX(MAIN_ACTIVITY_CLEAN) AS MAIN_ACTIVITY_CLEAN,

            MAX(TYPE_OF_CONTRACT_CLEAN) AS TYPE_OF_CONTRACT_CLEAN,
            MAX(TAL_LOCATION_NUTS_CLEAN) AS TAL_LOCATION_NUTS_CLEAN,
            MAX(CPV_CLEAN) AS CPV_CLEAN,
            MAX(ADDITIONAL_CPVS_CLEAN) AS ADDITIONAL_CPVS_CLEAN,
            MAX(TOP_TYPE_CLEAN) AS TOP_TYPE_CLEAN,

            COUNT(DISTINCT ISO_COUNTRY_CODE_CLEAN) FILTER (
                WHERE ISO_COUNTRY_CODE_CLEAN IS NOT NULL
            ) AS N_DISTINCT_COUNTRIES,

            COUNT(DISTINCT TYPE_OF_CONTRACT_CLEAN) FILTER (
                WHERE TYPE_OF_CONTRACT_CLEAN IS NOT NULL
            ) AS N_DISTINCT_CONTRACT_TYPES,

            COUNT(DISTINCT CPV_CLEAN) FILTER (
                WHERE CPV_CLEAN IS NOT NULL
            ) AS N_DISTINCT_PRIMARY_CPV,

            MAX(VALUE_EURO_CLEAN) AS VALUE_EURO_MAX,
            MAX(VALUE_EURO_FIN_1_CLEAN) AS VALUE_EURO_FIN_1_MAX,
            MAX(VALUE_EURO_FIN_2_CLEAN) AS VALUE_EURO_FIN_2_MAX,

            COUNT(DISTINCT VALUE_EURO_CLEAN) FILTER (
                WHERE VALUE_EURO_CLEAN IS NOT NULL
            ) AS N_DISTINCT_VALUE_EURO,

            MAX(LOTS_NUMBER_CLEAN) AS LOTS_NUMBER_MAX,
            COUNT(DISTINCT ID_LOT_CLEAN) FILTER (
                WHERE ID_LOT_CLEAN IS NOT NULL
            ) AS N_DISTINCT_ID_LOT,

            CASE
                WHEN BOOL_OR(CANCELLED_CLEAN IS TRUE)
                    THEN 'cancelled'
                WHEN BOOL_OR(CANCELLED_CLEAN IS FALSE)
                    THEN 'not_cancelled'
                ELSE 'unknown'
            END AS CANCELLED_STATUS,

            CASE
                WHEN BOOL_OR(CORRECTIONS_CLEAN IS TRUE)
                    THEN 'has_correction_indicator'
                WHEN BOOL_OR(CORRECTIONS_CLEAN IS FALSE)
                    THEN 'no_correction_indicator'
                ELSE 'unknown'
            END AS CORRECTION_STATUS,

            CASE
                WHEN COUNT(DISTINCT DT_DISPATCH_CLEAN) FILTER (
                    WHERE DT_DISPATCH_CLEAN IS NOT NULL
                ) > 1
                  OR COUNT(DISTINCT ISO_COUNTRY_CODE_CLEAN) FILTER (
                    WHERE ISO_COUNTRY_CODE_CLEAN IS NOT NULL
                ) > 1
                  OR COUNT(DISTINCT TYPE_OF_CONTRACT_CLEAN) FILTER (
                    WHERE TYPE_OF_CONTRACT_CLEAN IS NOT NULL
                ) > 1
                  OR COUNT(DISTINCT CPV_CLEAN) FILTER (
                    WHERE CPV_CLEAN IS NOT NULL
                ) > 1
                    THEN TRUE
                ELSE FALSE
            END AS HAS_CORE_NOTICE_CONFLICT,

            CASE
                WHEN MAX(DT_DISPATCH_CLEAN) IS NOT NULL THEN TRUE
                ELSE FALSE
            END AS HAS_VALID_DISPATCH_DATE,

            CASE
                WHEN YEAR(MAX(DT_DISPATCH_CLEAN)) = {int(target_year)}
                    THEN TRUE
                ELSE FALSE
            END AS IS_TARGET_DISPATCH_YEAR,

            CASE
                WHEN MAX(CPV_CLEAN) IS NOT NULL
                 AND LENGTH(REGEXP_EXTRACT(MAX(CPV_CLEAN), '([0-9]{{2}})', 1)) = 2
                    THEN TRUE
                ELSE FALSE
            END AS HAS_VALID_PRIMARY_CPV,

            CASE
                WHEN MAX(DT_DISPATCH_CLEAN) IS NOT NULL
                 AND YEAR(MAX(DT_DISPATCH_CLEAN)) = {int(target_year)}
                 AND MAX(CPV_CLEAN) IS NOT NULL
                 AND LENGTH(REGEXP_EXTRACT(MAX(CPV_CLEAN), '([0-9]{{2}})', 1)) = 2
                 AND NOT (
                    COUNT(DISTINCT DT_DISPATCH_CLEAN) FILTER (
                        WHERE DT_DISPATCH_CLEAN IS NOT NULL
                    ) > 1
                    OR COUNT(DISTINCT ISO_COUNTRY_CODE_CLEAN) FILTER (
                        WHERE ISO_COUNTRY_CODE_CLEAN IS NOT NULL
                    ) > 1
                    OR COUNT(DISTINCT TYPE_OF_CONTRACT_CLEAN) FILTER (
                        WHERE TYPE_OF_CONTRACT_CLEAN IS NOT NULL
                    ) > 1
                    OR COUNT(DISTINCT CPV_CLEAN) FILTER (
                        WHERE CPV_CLEAN IS NOT NULL
                    ) > 1
                 )
                 AND CASE
                        WHEN BOOL_OR(CANCELLED_CLEAN IS TRUE)
                            THEN FALSE
                        WHEN BOOL_OR(CANCELLED_CLEAN IS FALSE)
                            THEN TRUE
                        ELSE FALSE
                     END
                    THEN TRUE
                ELSE FALSE
            END AS IS_CORE_ELIGIBLE_TARGET

        FROM cn_rows_clean
        WHERE ID_NOTICE_CN_CLEAN IS NOT NULL
        GROUP BY ID_NOTICE_CN_CLEAN
        """
    )

    con.execute("DROP TABLE IF EXISTS cn_notice_conflicts")
    con.execute(
        """
        CREATE TABLE cn_notice_conflicts AS
        SELECT *
        FROM cn_notice
        WHERE HAS_CORE_NOTICE_CONFLICT
        """
    )


def build_child_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS cn_criteria")
    con.execute(
        """
        CREATE TABLE cn_criteria AS
        SELECT DISTINCT
            ID_NOTICE_CN_CLEAN,
            XSD_VERSION_CLEAN,
            CRIT_CODE_CLEAN,
            CRIT_PRICE_WEIGHT_CLEAN,
            CRIT_CRITERIA_CLEAN,
            CRIT_WEIGHTS_CLEAN
        FROM cn_rows_clean
        WHERE
            ID_NOTICE_CN_CLEAN IS NOT NULL
            AND (
                CRIT_CODE_CLEAN IS NOT NULL
                OR CRIT_PRICE_WEIGHT_CLEAN IS NOT NULL
                OR CRIT_CRITERIA_CLEAN IS NOT NULL
                OR CRIT_WEIGHTS_CLEAN IS NOT NULL
            )
        """
    )

    con.execute("DROP TABLE IF EXISTS cn_lot_context")
    con.execute(
        """
        CREATE TABLE cn_lot_context AS
        SELECT DISTINCT
            ID_NOTICE_CN_CLEAN,
            ID_LOT_CLEAN,
            CPV_CLEAN,
            ADDITIONAL_CPVS_CLEAN,
            VALUE_EURO_CLEAN,
            VALUE_EURO_FIN_1_CLEAN,
            VALUE_EURO_FIN_2_CLEAN,
            LOTS_NUMBER_CLEAN,
            LOTS_SUBMISSION_CLEAN
        FROM cn_rows_clean
        WHERE
            ID_NOTICE_CN_CLEAN IS NOT NULL
            AND (
                ID_LOT_CLEAN IS NOT NULL
                OR CPV_CLEAN IS NOT NULL
                OR ADDITIONAL_CPVS_CLEAN IS NOT NULL
                OR VALUE_EURO_CLEAN IS NOT NULL
                OR VALUE_EURO_FIN_1_CLEAN IS NOT NULL
                OR VALUE_EURO_FIN_2_CLEAN IS NOT NULL
            )
        """
    )

    con.execute("DROP TABLE IF EXISTS cn_provenance")
    con.execute(
        """
        CREATE TABLE cn_provenance AS
        SELECT
            ID_NOTICE_CN_CLEAN,
            SOURCE_FILE,
            SOURCE_ROW_ID
        FROM cn_rows_clean
        WHERE ID_NOTICE_CN_CLEAN IS NOT NULL
        """
    )


# -----------------------------------------------------------------------------
# Future CAN pair extraction and CAN 2017 summary
# -----------------------------------------------------------------------------

def build_future_can_pairs(con: duckdb.DuckDBPyConnection) -> None:
    """
    Conservative tokenization:
    - split TED triple-dash lists;
    - do not split comma/semicolon/pipe strings automatically;
    - flag those for review.
    """
    con.execute("DROP TABLE IF EXISTS cn_future_can_pair_evidence")
    con.execute(
        """
        CREATE TABLE cn_future_can_pair_evidence AS
        SELECT
            r.ID_NOTICE_CN_CLEAN,
            r.SOURCE_ROW_ID,
            r.FUTURE_CAN_ID_CLEAN AS FUTURE_CAN_ID_RAW_STRING,
            NULLIF(TRIM(token), '') AS ID_NOTICE_CAN_CLEAN,
            r.FUTURE_CAN_ID_ESTIMATED_RAW,
            r.FUTURE_CAN_ID_ESTIMATED_CLEAN,
            r.FUTURE_CAN_PARSE_STATUS
        FROM cn_rows_clean r,
        UNNEST(
            STRING_SPLIT(
                COALESCE(r.FUTURE_CAN_ID_CLEAN, ''),
                '---'
            )
        ) AS t(token)
        WHERE
            r.ID_NOTICE_CN_CLEAN IS NOT NULL
            AND r.FUTURE_CAN_ID_CLEAN IS NOT NULL
            AND NULLIF(TRIM(token), '') IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS cn_link_parse_review")
    con.execute(
        """
        CREATE TABLE cn_link_parse_review AS
        SELECT *
        FROM cn_rows_clean
        WHERE
            ID_NOTICE_CN_CLEAN IS NOT NULL
            AND FUTURE_CAN_PARSE_STATUS = 'unsupported_delimiter_review'
        """
    )

    con.execute("DROP TABLE IF EXISTS cn_future_can_pairs")
    con.execute(
        """
        CREATE TABLE cn_future_can_pairs AS
        SELECT
            ID_NOTICE_CN_CLEAN,
            ID_NOTICE_CAN_CLEAN,
            COUNT(*) AS LINK_EVIDENCE_ROWS,

            COUNT(*) FILTER (
                WHERE FUTURE_CAN_ID_ESTIMATED_CLEAN IS TRUE
            ) AS ESTIMATED_TRUE_EVIDENCE_ROWS,

            COUNT(*) FILTER (
                WHERE FUTURE_CAN_ID_ESTIMATED_CLEAN IS FALSE
            ) AS ESTIMATED_FALSE_EVIDENCE_ROWS,

            COUNT(*) FILTER (
                WHERE FUTURE_CAN_ID_ESTIMATED_CLEAN IS NULL
            ) AS ESTIMATED_UNKNOWN_EVIDENCE_ROWS,

            CASE
                WHEN COUNT(*) FILTER (
                    WHERE FUTURE_CAN_ID_ESTIMATED_CLEAN IS TRUE
                ) > 0
                 AND COUNT(*) FILTER (
                    WHERE FUTURE_CAN_ID_ESTIMATED_CLEAN IS FALSE
                ) > 0
                    THEN 'mixed_estimation_flags'

                WHEN COUNT(*) FILTER (
                    WHERE FUTURE_CAN_ID_ESTIMATED_CLEAN IS TRUE
                ) > 0
                    THEN 'explicitly_estimated'

                WHEN COUNT(*) FILTER (
                    WHERE FUTURE_CAN_ID_ESTIMATED_CLEAN IS FALSE
                ) > 0
                    THEN 'explicitly_non_estimated'

                ELSE 'unknown_estimation_status'
            END AS LINK_ESTIMATION_STATUS,

            MAX(CASE
                WHEN FUTURE_CAN_PARSE_STATUS = 'unsupported_delimiter_review'
                    THEN 1 ELSE 0
            END)::BOOLEAN AS HAS_UNSUPPORTED_DELIMITER_EVIDENCE

        FROM cn_future_can_pair_evidence
        GROUP BY
            ID_NOTICE_CN_CLEAN,
            ID_NOTICE_CAN_CLEAN
        """
    )


def build_can_2017_notice_summary(
    con: duckdb.DuckDBPyConnection,
    can_award_fact: Path,
) -> None:
    validate_can_schema(con, can_award_fact)

    con.execute("DROP TABLE IF EXISTS can_2017_notice_summary")
    con.execute(
        f"""
        CREATE TABLE can_2017_notice_summary AS
        SELECT
            ID_NOTICE_CAN_CLEAN,
            COUNT(DISTINCT AWARD_KEY) AS N_AWARD_KEYS,
            COUNT(*) AS N_AWARD_FACT_ROWS,

            COUNT(DISTINCT WIN_NAME_CLEAN) FILTER (
                WHERE WIN_NAME_CLEAN IS NOT NULL
            ) AS N_DISTINCT_WINNER_NAMES,

            COUNT(DISTINCT WIN_COUNTRY_CODE_CLEAN) FILTER (
                WHERE WIN_COUNTRY_CODE_CLEAN IS NOT NULL
            ) AS N_DISTINCT_WINNER_COUNTRIES,

            MAX(WIN_NAME_CLEAN) AS ACTUAL_WINNER_NAME,
            MAX(WIN_NATIONALID_CLEAN) AS ACTUAL_WINNER_NATIONALID,
            MAX(WIN_COUNTRY_CODE_CLEAN) AS ACTUAL_WINNER_COUNTRY,

            MIN(DT_AWARD_CLEAN) AS CAN_MIN_AWARD_DATE,
            MAX(DT_AWARD_CLEAN) AS CAN_MAX_AWARD_DATE,
            MIN(DT_DISPATCH_CLEAN) AS CAN_MIN_DISPATCH_DATE,
            MAX(DT_DISPATCH_CLEAN) AS CAN_MAX_DISPATCH_DATE,

            CASE
                WHEN BOOL_OR(COALESCE(B_AWARDED_TO_A_GROUP_CLEAN, FALSE))
                    THEN TRUE
                WHEN BOOL_OR(
                    POSITION('---' IN COALESCE(WIN_NAME_CLEAN, '')) > 0
                )
                    THEN TRUE
                WHEN BOOL_OR(
                    POSITION('---' IN COALESCE(WIN_NATIONALID_CLEAN, '')) > 0
                )
                    THEN TRUE
                WHEN BOOL_OR(
                    POSITION('---' IN COALESCE(WIN_COUNTRY_CODE_CLEAN, '')) > 0
                )
                    THEN TRUE
                ELSE FALSE
            END AS HAS_GROUP_OR_LIST_WINNER_STRUCTURE

        FROM read_parquet('{sql_path(can_award_fact)}')
        WHERE
            SOURCE_YEAR_MIN = 2017
            AND SOURCE_YEAR_MAX = 2017
            AND ID_NOTICE_CAN_CLEAN IS NOT NULL
        GROUP BY ID_NOTICE_CAN_CLEAN
        """
    )


# -----------------------------------------------------------------------------
# Exact pair link table
# -----------------------------------------------------------------------------

def build_cn_can_links(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS cn_can_links")
    con.execute(
        """
        CREATE TABLE cn_can_links AS
        SELECT
            p.ID_NOTICE_CN_CLEAN,
            p.ID_NOTICE_CAN_CLEAN,
            p.LINK_EVIDENCE_ROWS,
            p.ESTIMATED_TRUE_EVIDENCE_ROWS,
            p.ESTIMATED_FALSE_EVIDENCE_ROWS,
            p.ESTIMATED_UNKNOWN_EVIDENCE_ROWS,
            p.LINK_ESTIMATION_STATUS,
            p.HAS_UNSUPPORTED_DELIMITER_EVIDENCE,

            CASE
                WHEN c.ID_NOTICE_CAN_CLEAN IS NOT NULL
                    THEN 'matched_to_2017_award_fact'
                ELSE 'not_in_2017_award_fact'
            END AS CAN_MATCH_STATUS,

            c.N_AWARD_KEYS,
            c.N_AWARD_FACT_ROWS,
            c.N_DISTINCT_WINNER_NAMES,
            c.N_DISTINCT_WINNER_COUNTRIES,
            c.ACTUAL_WINNER_NAME,
            c.ACTUAL_WINNER_NATIONALID,
            c.ACTUAL_WINNER_COUNTRY,
            c.CAN_MIN_AWARD_DATE,
            c.CAN_MAX_AWARD_DATE,
            c.CAN_MIN_DISPATCH_DATE,
            c.CAN_MAX_DISPATCH_DATE,
            c.HAS_GROUP_OR_LIST_WINNER_STRUCTURE

        FROM cn_future_can_pairs p
        LEFT JOIN can_2017_notice_summary c
          USING (ID_NOTICE_CAN_CLEAN)
        """
    )


# -----------------------------------------------------------------------------
# Procurement cases and cohort classification
# -----------------------------------------------------------------------------

def build_procurement_cases(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS cn_link_summary")
    con.execute(
        """
        CREATE TABLE cn_link_summary AS
        SELECT
            n.ID_NOTICE_CN_CLEAN,

            COUNT(l.ID_NOTICE_CAN_CLEAN) AS N_FUTURE_CAN_PAIRS,

            COUNT(l.ID_NOTICE_CAN_CLEAN) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS N_MATCHED_CAN_PAIRS,

            COUNT(l.ID_NOTICE_CAN_CLEAN) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'not_in_2017_award_fact'
            ) AS N_UNMATCHED_CAN_PAIRS,

            MAX(l.ID_NOTICE_CAN_CLEAN) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS SINGLE_MATCHED_CAN_ID,

            MAX(l.LINK_ESTIMATION_STATUS) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_LINK_ESTIMATION_STATUS,

            MAX(l.N_AWARD_KEYS) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_CAN_N_AWARD_KEYS,

            MAX(l.N_DISTINCT_WINNER_NAMES) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_CAN_N_DISTINCT_WINNER_NAMES,

            MAX(l.N_DISTINCT_WINNER_COUNTRIES) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_CAN_N_DISTINCT_WINNER_COUNTRIES,

            MAX(l.ACTUAL_WINNER_NAME) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS ACTUAL_WINNER_NAME,

            MAX(l.ACTUAL_WINNER_NATIONALID) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS ACTUAL_WINNER_NATIONALID,

            MAX(l.ACTUAL_WINNER_COUNTRY) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS ACTUAL_WINNER_COUNTRY,

            BOOL_OR(COALESCE(l.HAS_GROUP_OR_LIST_WINNER_STRUCTURE, FALSE)) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_CAN_HAS_GROUP_OR_LIST_WINNER_STRUCTURE,

            MIN(l.CAN_MIN_AWARD_DATE) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_CAN_MIN_AWARD_DATE,

            MAX(l.CAN_MAX_AWARD_DATE) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_CAN_MAX_AWARD_DATE,

            MIN(l.CAN_MIN_DISPATCH_DATE) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_CAN_MIN_DISPATCH_DATE,

            MAX(l.CAN_MAX_DISPATCH_DATE) FILTER (
                WHERE l.CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            ) AS MATCHED_CAN_MAX_DISPATCH_DATE

        FROM cn_notice n
        LEFT JOIN cn_can_links l
          USING (ID_NOTICE_CN_CLEAN)
        GROUP BY n.ID_NOTICE_CN_CLEAN
        """
    )

    con.execute("DROP TABLE IF EXISTS procurement_cases")
    con.execute(
        """
        CREATE TABLE procurement_cases AS
        SELECT
            n.*,
            s.N_FUTURE_CAN_PAIRS,
            s.N_MATCHED_CAN_PAIRS,
            s.N_UNMATCHED_CAN_PAIRS,
            s.SINGLE_MATCHED_CAN_ID,
            s.MATCHED_LINK_ESTIMATION_STATUS,
            s.MATCHED_CAN_N_AWARD_KEYS,
            s.MATCHED_CAN_N_DISTINCT_WINNER_NAMES,
            s.MATCHED_CAN_N_DISTINCT_WINNER_COUNTRIES,
            s.ACTUAL_WINNER_NAME,
            s.ACTUAL_WINNER_NATIONALID,
            s.ACTUAL_WINNER_COUNTRY,
            s.MATCHED_CAN_HAS_GROUP_OR_LIST_WINNER_STRUCTURE,
            s.MATCHED_CAN_MIN_AWARD_DATE,
            s.MATCHED_CAN_MAX_AWARD_DATE,
            s.MATCHED_CAN_MIN_DISPATCH_DATE,
            s.MATCHED_CAN_MAX_DISPATCH_DATE,

            CASE
                WHEN NOT n.IS_CORE_ELIGIBLE_TARGET
                    THEN 'invalid_target'

                WHEN COALESCE(s.N_FUTURE_CAN_PAIRS, 0) = 0
                    THEN 'no_future_can_id'

                WHEN s.N_FUTURE_CAN_PAIRS > 1
                    THEN 'multi_link'

                WHEN s.N_FUTURE_CAN_PAIRS = 1
                 AND s.N_MATCHED_CAN_PAIRS = 0
                    THEN 'single_link_not_in_2017_award_fact'

                WHEN s.N_FUTURE_CAN_PAIRS = 1
                 AND s.N_MATCHED_CAN_PAIRS = 1
                 AND s.MATCHED_CAN_N_AWARD_KEYS > 1
                    THEN 'single_link_multi_award'

                WHEN s.N_FUTURE_CAN_PAIRS = 1
                 AND s.N_MATCHED_CAN_PAIRS = 1
                 AND s.MATCHED_CAN_N_AWARD_KEYS = 1
                 AND COALESCE(s.MATCHED_CAN_N_DISTINCT_WINNER_NAMES, 0) <> 1
                    THEN 'single_link_single_award_nonunique_winner'

                WHEN s.N_FUTURE_CAN_PAIRS = 1
                 AND s.N_MATCHED_CAN_PAIRS = 1
                 AND s.MATCHED_CAN_N_AWARD_KEYS = 1
                 AND s.MATCHED_CAN_N_DISTINCT_WINNER_NAMES = 1
                 AND COALESCE(
                    s.MATCHED_CAN_HAS_GROUP_OR_LIST_WINNER_STRUCTURE,
                    FALSE
                 )
                    THEN 'single_link_single_award_group_or_list_winner'

                WHEN s.N_FUTURE_CAN_PAIRS = 1
                 AND s.N_MATCHED_CAN_PAIRS = 1
                 AND s.MATCHED_CAN_N_AWARD_KEYS = 1
                 AND s.MATCHED_CAN_N_DISTINCT_WINNER_NAMES = 1
                 AND NOT COALESCE(
                    s.MATCHED_CAN_HAS_GROUP_OR_LIST_WINNER_STRUCTURE,
                    FALSE
                 )
                    THEN 'strict_single_link_single_award'

                ELSE 'other_review'
            END AS COHORT_CLASSIFICATION,

            CASE
                WHEN n.IS_CORE_ELIGIBLE_TARGET
                 AND s.N_FUTURE_CAN_PAIRS = 1
                 AND s.N_MATCHED_CAN_PAIRS = 1
                 AND s.MATCHED_CAN_N_AWARD_KEYS = 1
                 AND s.MATCHED_CAN_N_DISTINCT_WINNER_NAMES = 1
                 AND NOT COALESCE(
                    s.MATCHED_CAN_HAS_GROUP_OR_LIST_WINNER_STRUCTURE,
                    FALSE
                 )
                    THEN TRUE
                ELSE FALSE
            END AS IS_STRICT_STRUCTURAL_COHORT,

            CASE
                WHEN n.IS_CORE_ELIGIBLE_TARGET
                 AND s.N_FUTURE_CAN_PAIRS = 1
                 AND s.N_MATCHED_CAN_PAIRS = 1
                 AND s.MATCHED_CAN_N_AWARD_KEYS = 1
                 AND s.MATCHED_CAN_N_DISTINCT_WINNER_NAMES = 1
                 AND NOT COALESCE(
                    s.MATCHED_CAN_HAS_GROUP_OR_LIST_WINNER_STRUCTURE,
                    FALSE
                 )
                 AND COALESCE(
                    s.MATCHED_LINK_ESTIMATION_STATUS,
                    'unknown_estimation_status'
                 ) NOT IN ('explicitly_estimated', 'mixed_estimation_flags')
                    THEN TRUE
                ELSE FALSE
            END AS IS_STRICT_EXCLUDING_EXPLICIT_ESTIMATED,

            CASE
                WHEN n.IS_CORE_ELIGIBLE_TARGET
                 AND s.N_FUTURE_CAN_PAIRS = 1
                 AND s.N_MATCHED_CAN_PAIRS = 1
                 AND s.MATCHED_CAN_N_AWARD_KEYS = 1
                 AND s.MATCHED_CAN_N_DISTINCT_WINNER_NAMES = 1
                 AND NOT COALESCE(
                    s.MATCHED_CAN_HAS_GROUP_OR_LIST_WINNER_STRUCTURE,
                    FALSE
                 )
                 AND s.MATCHED_LINK_ESTIMATION_STATUS = 'explicitly_non_estimated'
                    THEN TRUE
                ELSE FALSE
            END AS IS_STRICT_CONFIRMED_NON_ESTIMATED

        FROM cn_notice n
        LEFT JOIN cn_link_summary s
          USING (ID_NOTICE_CN_CLEAN)
        """
    )


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------

def target_case_funnel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    steps = [
        ("01_unique_cn_notices", "TRUE"),
        ("02_valid_dispatch_date", "HAS_VALID_DISPATCH_DATE"),
        ("03_dispatch_year_2017", "HAS_VALID_DISPATCH_DATE AND IS_TARGET_DISPATCH_YEAR"),
        ("04_valid_primary_cpv", "HAS_VALID_DISPATCH_DATE AND IS_TARGET_DISPATCH_YEAR AND HAS_VALID_PRIMARY_CPV"),
        ("05_not_cancelled", "HAS_VALID_DISPATCH_DATE AND IS_TARGET_DISPATCH_YEAR AND HAS_VALID_PRIMARY_CPV AND CANCELLED_STATUS = 'not_cancelled'"),
        ("06_no_core_conflict", "IS_CORE_ELIGIBLE_TARGET"),
        ("07_one_future_can_pair", "IS_CORE_ELIGIBLE_TARGET AND N_FUTURE_CAN_PAIRS = 1"),
        ("08_single_pair_matched_2017_can", "IS_CORE_ELIGIBLE_TARGET AND N_FUTURE_CAN_PAIRS = 1 AND N_MATCHED_CAN_PAIRS = 1"),
        ("09_single_can_award", "IS_CORE_ELIGIBLE_TARGET AND N_FUTURE_CAN_PAIRS = 1 AND N_MATCHED_CAN_PAIRS = 1 AND MATCHED_CAN_N_AWARD_KEYS = 1"),
        ("10_single_observed_winner", "IS_CORE_ELIGIBLE_TARGET AND N_FUTURE_CAN_PAIRS = 1 AND N_MATCHED_CAN_PAIRS = 1 AND MATCHED_CAN_N_AWARD_KEYS = 1 AND MATCHED_CAN_N_DISTINCT_WINNER_NAMES = 1"),
        ("11_non_group_or_list_winner", "IS_STRICT_STRUCTURAL_COHORT"),
        ("12_strict_excluding_explicit_estimated", "IS_STRICT_EXCLUDING_EXPLICIT_ESTIMATED"),
        ("13_strict_confirmed_non_estimated", "IS_STRICT_CONFIRMED_NON_ESTIMATED"),
    ]

    total = scalar(con, "SELECT COUNT(*) FROM procurement_cases")
    rows = []
    previous = total

    for step, condition in steps:
        n = scalar(
            con,
            f"SELECT COUNT(*) FROM procurement_cases WHERE {condition}",
        )
        rows.append(
            {
                "step": step,
                "n_notices": n,
                "share_of_all_cn_notices_pct": 100.0 * n / total if total else None,
                "retention_from_previous_step_pct": 100.0 * n / previous if previous else None,
            }
        )
        previous = n

    return pd.DataFrame(rows)


def cohort_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COHORT_CLASSIFICATION,
            COUNT(*) AS N_CN_NOTICES,
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS SHARE_PCT
        FROM procurement_cases
        GROUP BY COHORT_CLASSIFICATION
        ORDER BY N_CN_NOTICES DESC
        """,
    )


def link_estimation_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            LINK_ESTIMATION_STATUS,
            CAN_MATCH_STATUS,
            COUNT(*) AS N_CN_CAN_PAIRS
        FROM cn_can_links
        GROUP BY LINK_ESTIMATION_STATUS, CAN_MATCH_STATUS
        ORDER BY N_CN_CAN_PAIRS DESC
        """,
    )


def cn_to_can_cardinality(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COALESCE(N_FUTURE_CAN_PAIRS, 0) AS N_FUTURE_CAN_PAIRS,
            COUNT(*) AS N_CN_NOTICES
        FROM procurement_cases
        GROUP BY COALESCE(N_FUTURE_CAN_PAIRS, 0)
        ORDER BY N_FUTURE_CAN_PAIRS
        """,
    )


def can_to_cn_cardinality(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        WITH per_can AS (
            SELECT
                ID_NOTICE_CAN_CLEAN,
                COUNT(DISTINCT ID_NOTICE_CN_CLEAN) AS N_CN_NOTICES
            FROM cn_can_links
            GROUP BY ID_NOTICE_CAN_CLEAN
        )
        SELECT
            N_CN_NOTICES,
            COUNT(*) AS N_CAN_IDS
        FROM per_can
        GROUP BY N_CN_NOTICES
        ORDER BY N_CN_NOTICES
        """,
    )


def linked_award_multiplicity(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            MATCHED_CAN_N_AWARD_KEYS AS N_AWARD_KEYS,
            COUNT(*) AS N_CN_NOTICES
        FROM procurement_cases
        WHERE
            N_FUTURE_CAN_PAIRS = 1
            AND N_MATCHED_CAN_PAIRS = 1
            AND MATCHED_CAN_N_AWARD_KEYS IS NOT NULL
        GROUP BY MATCHED_CAN_N_AWARD_KEYS
        ORDER BY N_AWARD_KEYS
        """,
    )


def criteria_notice_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        WITH flags AS (
            SELECT
                n.ID_NOTICE_CN_CLEAN,
                MAX(CASE WHEN c.CRIT_CODE_CLEAN IS NOT NULL THEN 1 ELSE 0 END)
                    AS HAS_CRIT_CODE,
                MAX(CASE WHEN c.CRIT_PRICE_WEIGHT_CLEAN IS NOT NULL THEN 1 ELSE 0 END)
                    AS HAS_CRIT_PRICE_WEIGHT,
                MAX(CASE WHEN c.CRIT_CRITERIA_CLEAN IS NOT NULL THEN 1 ELSE 0 END)
                    AS HAS_CRIT_CRITERIA,
                MAX(CASE WHEN c.CRIT_WEIGHTS_CLEAN IS NOT NULL THEN 1 ELSE 0 END)
                    AS HAS_CRIT_WEIGHTS
            FROM cn_notice n
            LEFT JOIN cn_criteria c
              USING (ID_NOTICE_CN_CLEAN)
            GROUP BY n.ID_NOTICE_CN_CLEAN
        ),
        long AS (
            SELECT 'CRIT_CODE' AS FIELD, HAS_CRIT_CODE AS HAS_FIELD FROM flags
            UNION ALL
            SELECT 'CRIT_PRICE_WEIGHT', HAS_CRIT_PRICE_WEIGHT FROM flags
            UNION ALL
            SELECT 'CRIT_CRITERIA', HAS_CRIT_CRITERIA FROM flags
            UNION ALL
            SELECT 'CRIT_WEIGHTS', HAS_CRIT_WEIGHTS FROM flags
        )
        SELECT
            FIELD,
            COUNT(*) AS N_NOTICES,
            SUM(HAS_FIELD) AS NOTICES_WITH_FIELD,
            100.0 * SUM(HAS_FIELD) / COUNT(*) AS NOTICE_COVERAGE_PCT
        FROM long
        GROUP BY FIELD
        ORDER BY FIELD
        """,
    )


def notice_quality_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT 'cancelled_status' AS DIMENSION, CANCELLED_STATUS AS VALUE,
               COUNT(*) AS N_NOTICES
        FROM cn_notice
        GROUP BY CANCELLED_STATUS

        UNION ALL

        SELECT 'correction_status', CORRECTION_STATUS, COUNT(*)
        FROM cn_notice
        GROUP BY CORRECTION_STATUS

        UNION ALL

        SELECT 'core_conflict',
               CASE WHEN HAS_CORE_NOTICE_CONFLICT THEN 'true' ELSE 'false' END,
               COUNT(*)
        FROM cn_notice
        GROUP BY HAS_CORE_NOTICE_CONFLICT

        UNION ALL

        SELECT 'target_year',
               CASE WHEN IS_TARGET_DISPATCH_YEAR THEN 'true' ELSE 'false' END,
               COUNT(*)
        FROM cn_notice
        GROUP BY IS_TARGET_DISPATCH_YEAR

        ORDER BY DIMENSION, N_NOTICES DESC
        """,
    )


def build_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(metric: str, value, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    add("raw_cn_rows", scalar(con, "SELECT COUNT(*) FROM cn_rows_clean"))
    add("unique_cn_notices", scalar(con, "SELECT COUNT(*) FROM cn_notice"))
    add("cn_can_pairs", scalar(con, "SELECT COUNT(*) FROM cn_can_links"))
    add(
        "matched_cn_can_pairs",
        scalar(
            con,
            """
            SELECT COUNT(*) FROM cn_can_links
            WHERE CAN_MATCH_STATUS = 'matched_to_2017_award_fact'
            """,
        ),
    )
    add(
        "core_eligible_cn_notices",
        scalar(
            con,
            "SELECT COUNT(*) FROM procurement_cases WHERE IS_CORE_ELIGIBLE_TARGET",
        ),
    )
    add(
        "strict_structural_cohort",
        scalar(
            con,
            "SELECT COUNT(*) FROM procurement_cases WHERE IS_STRICT_STRUCTURAL_COHORT",
        ),
    )
    add(
        "strict_excluding_explicit_estimated",
        scalar(
            con,
            "SELECT COUNT(*) FROM procurement_cases WHERE IS_STRICT_EXCLUDING_EXPLICIT_ESTIMATED",
        ),
    )
    add(
        "strict_confirmed_non_estimated",
        scalar(
            con,
            "SELECT COUNT(*) FROM procurement_cases WHERE IS_STRICT_CONFIRMED_NON_ESTIMATED",
        ),
    )
    add(
        "unsupported_future_can_delimiter_rows",
        scalar(con, "SELECT COUNT(*) FROM cn_link_parse_review"),
    )
    add(
        "core_conflict_notices",
        scalar(con, "SELECT COUNT(*) FROM cn_notice_conflicts"),
    )

    return pd.DataFrame(rows)


def create_markdown_report(
    metrics: pd.DataFrame,
    funnel: pd.DataFrame,
    cohort: pd.DataFrame,
    output_path: Path,
    target_year: int,
) -> None:
    metric_map = dict(zip(metrics["metric"], metrics["value"]))

    def mv(name: str, default="N/A"):
        return metric_map.get(name, default)

    lines = [
        f"# TED CN {target_year} Target-Case Build Report",
        "",
        "## Build summary",
        "",
        f"- Raw CN rows: {mv('raw_cn_rows')}",
        f"- Unique CN notices: {mv('unique_cn_notices')}",
        f"- Distinct CN-CAN pairs: {mv('cn_can_pairs')}",
        f"- Matched CN-CAN pairs in filtered CAN {target_year} award fact: {mv('matched_cn_can_pairs')}",
        f"- Core-eligible CN notices: {mv('core_eligible_cn_notices')}",
        f"- Strict structural cohort: {mv('strict_structural_cohort')}",
        f"- Strict cohort excluding explicitly estimated/mixed links: {mv('strict_excluding_explicit_estimated')}",
        f"- Strict cohort with explicitly confirmed non-estimated links: {mv('strict_confirmed_non_estimated')}",
        "",
        "## Linkage policy",
        "",
        "CN-CAN linkage is constructed at distinct pair grain `(ID_NOTICE_CN, ID_NOTICE_CAN)`. "
        "The build never chooses a future CAN ID with `MAX()` before cardinality classification. "
        "Triple-dash FUTURE_CAN_ID lists are split deterministically; other delimiter patterns are flagged for review.",
        "",
        "## Strict structural cohort",
        "",
        "A notice enters the strict structural cohort only when it has a valid target-year dispatch date, "
        "valid primary CPV, explicit non-cancelled status, no core notice conflict, one future CAN pair, "
        "a matched CAN notice in the filtered target-year award fact, one CAN award key, one observed winner, "
        "and no group/list-valued winner structure.",
        "",
        "Estimated-link status is kept separate so the paper can report both a broader structural cohort and "
        "a stricter confirmed-non-estimated sensitivity cohort.",
        "",
        "## Funnel",
        "",
        "| Step | Notices | Share of all CN notices (%) | Retention from previous step (%) |",
        "|---|---:|---:|---:|",
    ]

    for _, row in funnel.iterrows():
        lines.append(
            f"| {row['step']} | {int(row['n_notices'])} | "
            f"{float(row['share_of_all_cn_notices_pct']):.3f} | "
            f"{float(row['retention_from_previous_step_pct']):.3f} |"
        )

    lines.extend([
        "",
        "## Cohort classifications",
        "",
        "| Classification | CN notices | Share (%) |",
        "|---|---:|---:|",
    ])

    for _, row in cohort.iterrows():
        lines.append(
            f"| `{row['COHORT_CLASSIFICATION']}` | {int(row['N_CN_NOTICES'])} | "
            f"{float(row['SHARE_PCT']):.3f} |"
        )

    lines.extend([
        "",
        "## Interpretation limitation",
        "",
        "A matched 2017 CAN outcome is an observed procurement outcome, not proof that its winner is the uniquely best supplier. "
        "The later candidate pool will represent historically active comparison suppliers, not the actual bidder set, because unsuccessful bidder identities are not available in the CAN award-fact data.",
        "",
        "## Next step",
        "",
        "Map strict 2017 winners to the frozen 2015-2016 supplier dimension without relearning identities from 2017, "
        "then build CPV-contextual candidate pools and measure historical winner coverage before any ranking experiment.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized CN 2017 target tables and exact CN-CAN links."
    )

    parser.add_argument("--cn-2017", required=True, type=Path)
    parser.add_argument("--can-award-fact", required=True, type=Path)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cn_targets_2017"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/cn_targets_2017"),
    )
    parser.add_argument("--work-db", type=Path, default=None)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="Optional DuckDB memory limit, e.g. 8GB.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2017,
    )

    args = parser.parse_args()

    if not args.cn_2017.exists():
        raise FileNotFoundError(f"CN 2017 CSV not found: {args.cn_2017}")
    if not args.can_award_fact.exists():
        raise FileNotFoundError(
            f"CAN award fact not found: {args.can_award_fact}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    delimiter = detect_delimiter(args.cn_2017)
    validate_cn_schema(args.cn_2017, delimiter)

    work_db = (
        args.work_db
        if args.work_db is not None
        else args.output_dir / "_cn_target_build.duckdb"
    )
    work_db.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(work_db))
    con.execute(f"SET threads={int(args.threads)}")

    if args.memory_limit:
        mem = args.memory_limit.replace("'", "''")
        con.execute(f"SET memory_limit='{mem}'")

    temp_dir = args.output_dir / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")

    try:
        print("1/10 Validating CAN award fact...")
        validate_can_schema(con, args.can_award_fact)

        print("2/10 Loading and normalizing CN rows...")
        build_cn_rows_clean(con, args.cn_2017, delimiter)

        print("3/10 Building normalized notice and child tables...")
        build_cn_notice_table(con, target_year=args.target_year)
        build_child_tables(con)

        print("4/10 Extracting exact CN-CAN pair evidence...")
        build_future_can_pairs(con)

        print("5/10 Building 2017 CAN notice summary...")
        build_can_2017_notice_summary(con, args.can_award_fact)

        print("6/10 Building exact CN-CAN link table...")
        build_cn_can_links(con)

        print("7/10 Classifying procurement cases...")
        build_procurement_cases(con)

        print("8/10 Building diagnostics and funnel...")
        funnel_df = target_case_funnel(con)
        cohort_df = cohort_summary(con)
        estimation_df = link_estimation_summary(con)
        cn_to_can_df = cn_to_can_cardinality(con)
        can_to_cn_df = can_to_cn_cardinality(con)
        multiplicity_df = linked_award_multiplicity(con)
        criteria_df = criteria_notice_coverage(con)
        quality_df = notice_quality_summary(con)
        metrics_df = build_metrics(con)

        print("9/10 Writing Parquet outputs...")
        outputs = {
            "cn_notice": "cn_notice_2017.parquet",
            "cn_criteria": "cn_criteria_2017.parquet",
            "cn_lot_context": "cn_lot_context_2017.parquet",
            "cn_provenance": "cn_provenance_2017.parquet",
            "cn_can_links": "cn_can_links_2017.parquet",
            "procurement_cases": "procurement_cases_2017.parquet",
            "cn_notice_conflicts": "cn_notice_conflicts_2017.parquet",
            "cn_link_parse_review": "cn_link_parse_review_2017.parquet",
        }

        for table_name, filename in outputs.items():
            copy_table_to_parquet(
                con,
                table_name,
                args.output_dir / filename,
            )

        print("10/10 Writing reports...")
        reports = {
            "target_case_funnel.csv": funnel_df,
            "cohort_classification_summary.csv": cohort_df,
            "link_estimation_status_summary.csv": estimation_df,
            "cn_to_can_pair_cardinality.csv": cn_to_can_df,
            "can_to_cn_pair_cardinality.csv": can_to_cn_df,
            "linked_can_award_multiplicity.csv": multiplicity_df,
            "criteria_notice_coverage.csv": criteria_df,
            "notice_quality_summary.csv": quality_df,
            "cn_target_build_metrics.csv": metrics_df,
        }

        for filename, df in reports.items():
            write_csv(df, args.report_dir / filename)

        create_markdown_report(
            metrics=metrics_df,
            funnel=funnel_df,
            cohort=cohort_df,
            output_path=args.report_dir / "cn_target_build_report.md",
            target_year=args.target_year,
        )

        print()
        print("=" * 78)
        print("CN 2017 target-case build complete")
        print("=" * 78)
        print(f"Processed outputs: {args.output_dir.resolve()}")
        print(f"Reports:           {args.report_dir.resolve()}")
        print(f"Work DB:           {work_db.resolve()}")
        print()
        print("Review these first:")
        print("  cn_target_build_report.md")
        print("  target_case_funnel.csv")
        print("  cohort_classification_summary.csv")
        print("  link_estimation_status_summary.csv")
        print("  cn_to_can_pair_cardinality.csv")
        print("  can_to_cn_pair_cardinality.csv")
        print("  linked_can_award_multiplicity.csv")
        print("  criteria_notice_coverage.csv")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
