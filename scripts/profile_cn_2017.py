#!/usr/bin/env python3
"""
Profile TED Contract Notice (CN) 2017 data for target-case construction.

Purpose
-------
This profiler is designed for the next stage of the study:

    CN 2017
        -> profile and clean
        -> link to CAN 2017 outcomes
        -> construct target procurement cases

It profiles:
- schema compatibility and row count;
- date parsing;
- missingness;
- notice grain and rows per notice;
- FUTURE_CAN_ID coverage and structure;
- FUTURE_CAN_ID_ESTIMATED values;
- CN -> CAN and CAN -> CN cardinality;
- CPV coverage and multiplicity;
- country and contract-type coverage;
- criteria and weight coverage;
- value coverage;
- lot structure;
- cancellation and correction encodings;
- XSD-version distribution;
- list-like delimiter signals;
- optional linkability to the existing CAN 2017 award-fact table.

The script does NOT build procurement cases. It only diagnoses the CN 2017
structure before cleaning and linking.

Requires:
    pip install duckdb pandas pyarrow
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------
# Expected CN schema
# ---------------------------------------------------------------------

EXPECTED_CN_COLUMNS = [
    "ID_NOTICE_CN",
    "TED_NOTICE_URL",
    "YEAR",
    "ID_TYPE",
    "DT_DISPATCH",
    "XSD_VERSION",
    "CANCELLED",
    "CORRECTIONS",
    "FUTURE_CAN_ID",
    "FUTURE_CAN_ID_ESTIMATED",
    "B_MULTIPLE_CAE",
    "CAE_NAME",
    "CAE_NATIONALID",
    "CAE_ADDRESS",
    "CAE_TOWN",
    "CAE_POSTAL_CODE",
    "ISO_COUNTRY_CODE",
    "B_MULTIPLE_COUNTRY",
    "ISO_COUNTRY_CODE_ALL",
    "CAE_TYPE",
    "EU_INST_CODE",
    "MAIN_ACTIVITY",
    "B_ON_BEHALF",
    "B_INVOLVES_JOINT_PROCUREMENT",
    "B_AWARDED_BY_CENTRAL_BODY",
    "TYPE_OF_CONTRACT",
    "TAL_LOCATION_NUTS",
    "B_FRA_AGREEMENT",
    "FRA_ESTIMATED",
    "B_FRA_SINGLE_OPERATOR",
    "FRA_NUMBER_OPERATORS",
    "FRA_NUMBER_MAX_OPERATORS",
    "B_DYN_PURCH_SYST",
    "CPV",
    "ID_LOT",
    "ADDITIONAL_CPVS",
    "B_GPA",
    "LOTS_NUMBER",
    "LOTS_SUBMISSION",
    "B_VARIANTS",
    "VALUE_EURO",
    "VALUE_EURO_FIN_1",
    "VALUE_EURO_FIN_2",
    "B_OPTIONS",
    "B_EU_FUNDS",
    "B_RENEWALS",
    "DURATION",
    "CONTRACT_START",
    "CONTRACT_COMPLETION",
    "TOP_TYPE",
    "B_ACCELERATED",
    "ENV_OPERATORS",
    "ENV_MIN_OPERATORS",
    "ENV_MAX_OPERATORS",
    "CRIT_CODE",
    "CRIT_PRICE_WEIGHT",
    "CRIT_CRITERIA",
    "CRIT_WEIGHTS",
    "B_ELECTRONIC_AUCTION",
    "DT_APPLICATIONS",
    "B_LANGUAGE_ANY_EC",
    "ADMIN_LANGUAGES_TENDER",
    "ADMIN_OTHER_LANGUAGES_TENDER",
    "B_RECURRENT_PROCUREMENT",
]


DATE_COLUMNS = [
    "DT_DISPATCH",
    "CONTRACT_START",
    "CONTRACT_COMPLETION",
    "DT_APPLICATIONS",
]

LINK_FIELDS = [
    "FUTURE_CAN_ID",
    "FUTURE_CAN_ID_ESTIMATED",
]

CRITERIA_FIELDS = [
    "CRIT_CODE",
    "CRIT_PRICE_WEIGHT",
    "CRIT_CRITERIA",
    "CRIT_WEIGHTS",
]

VALUE_FIELDS = [
    "VALUE_EURO",
    "VALUE_EURO_FIN_1",
    "VALUE_EURO_FIN_2",
]

LOT_FIELDS = [
    "ID_LOT",
    "LOTS_NUMBER",
    "LOTS_SUBMISSION",
]

BOOLEAN_LIKE_FIELDS = [
    "CANCELLED",
    "CORRECTIONS",
    "FUTURE_CAN_ID_ESTIMATED",
    "B_MULTIPLE_CAE",
    "B_MULTIPLE_COUNTRY",
    "B_ON_BEHALF",
    "B_INVOLVES_JOINT_PROCUREMENT",
    "B_AWARDED_BY_CENTRAL_BODY",
    "B_FRA_AGREEMENT",
    "B_FRA_SINGLE_OPERATOR",
    "B_DYN_PURCH_SYST",
    "B_GPA",
    "B_VARIANTS",
    "B_OPTIONS",
    "B_EU_FUNDS",
    "B_RENEWALS",
    "B_ACCELERATED",
    "B_ELECTRONIC_AUCTION",
    "B_LANGUAGE_ANY_EC",
    "B_RECURRENT_PROCUREMENT",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def clean_text_sql(column: str) -> str:
    c = qident(column)
    return f"NULLIF(TRIM(CAST({c} AS VARCHAR)), '')"


def date_parse_sql(column: str) -> str:
    """
    Supports formats already observed in TED annual files plus fallbacks.
    """
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
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------
# Validation and raw table
# ---------------------------------------------------------------------

def validate_schema(path: Path, delimiter: str) -> pd.DataFrame:
    actual = read_header(path, delimiter)

    expected_set = set(EXPECTED_CN_COLUMNS)
    actual_set = set(actual)

    return pd.DataFrame(
        [{
            "file": path.name,
            "actual_column_count": len(actual),
            "expected_column_count": len(EXPECTED_CN_COLUMNS),
            "same_column_set": actual_set == expected_set,
            "same_column_order": actual == EXPECTED_CN_COLUMNS,
            "missing_columns": " | ".join(
                sorted(expected_set - actual_set)
            ),
            "extra_columns": " | ".join(
                sorted(actual_set - expected_set)
            ),
        }]
    )


def create_raw_table(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    delimiter: str,
) -> None:
    relation = csv_relation(path, delimiter)

    con.execute("DROP TABLE IF EXISTS cn_raw")
    con.execute(
        f"""
        CREATE TABLE cn_raw AS
        SELECT
            *,
            ROW_NUMBER() OVER ()::BIGINT AS SOURCE_ROW_ID,
            '{path.name.replace("'", "''")}'::VARCHAR AS SOURCE_FILE
        FROM {relation}
        """
    )


# ---------------------------------------------------------------------
# Core profile reports
# ---------------------------------------------------------------------

def profile_overview(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COUNT(*) AS raw_rows,
            COUNT(DISTINCT NULLIF(TRIM(ID_NOTICE_CN), ''))
                AS unique_notice_ids,
            COUNT(*) - COUNT(DISTINCT NULLIF(TRIM(ID_NOTICE_CN), ''))
                AS rows_above_unique_notice_count,
            COUNT(DISTINCT NULLIF(TRIM(FUTURE_CAN_ID), ''))
                AS unique_nonblank_future_can_ids,
            COUNT(*) FILTER (
                WHERE NULLIF(TRIM(FUTURE_CAN_ID), '') IS NOT NULL
            ) AS rows_with_future_can_id
        FROM cn_raw
        """
    )


def profile_date_parse(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frames = []

    for col in DATE_COLUMNS:
        parsed = date_parse_sql(col)
        c = qident(col)

        df = fetch_df(
            con,
            f"""
            SELECT
                '{col}' AS column_name,
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                ) AS nonmissing_rows,
                COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                      AND ({parsed}) IS NOT NULL
                ) AS parsed_nonmissing_rows,
                COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                      AND ({parsed}) IS NULL
                ) AS failed_nonmissing_rows,
                100.0 * COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                      AND ({parsed}) IS NOT NULL
                )
                / NULLIF(
                    COUNT(*) FILTER (
                        WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                    ),
                    0
                ) AS parse_success_pct_nonmissing,
                MIN({parsed}) AS min_parsed_date,
                MAX({parsed}) AS max_parsed_date
            FROM cn_raw
            """
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def profile_failed_date_examples(
    con: duckdb.DuckDBPyConnection,
    top_n: int,
) -> pd.DataFrame:
    frames = []

    for col in DATE_COLUMNS:
        parsed = date_parse_sql(col)
        c = qident(col)

        df = fetch_df(
            con,
            f"""
            SELECT
                '{col}' AS column_name,
                CAST({c} AS VARCHAR) AS raw_value,
                COUNT(*) AS n
            FROM cn_raw
            WHERE
                NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                AND ({parsed}) IS NULL
            GROUP BY CAST({c} AS VARCHAR)
            ORDER BY n DESC
            LIMIT {int(top_n)}
            """
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def profile_missingness(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pieces = []

    for col in EXPECTED_CN_COLUMNS:
        c = qident(col)
        pieces.append(
            f"""
            SELECT
                '{col}' AS column_name,
                COUNT(*) AS n_rows,
                COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NULL
                ) AS missing_rows,
                100.0 * COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NULL
                ) / COUNT(*) AS missing_pct,
                COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                ) AS nonmissing_rows,
                100.0 * COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                ) / COUNT(*) AS coverage_pct
            FROM cn_raw
            """
        )

    return fetch_df(
        con,
        "\nUNION ALL\n".join(pieces)
        + "\nORDER BY missing_pct DESC, column_name"
    )


def profile_notice_grain(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        WITH per_notice AS (
            SELECT
                NULLIF(TRIM(ID_NOTICE_CN), '') AS ID_NOTICE_CN_CLEAN,
                COUNT(*) AS rows_per_notice,
                COUNT(DISTINCT NULLIF(TRIM(ID_LOT), ''))
                    AS distinct_id_lot,
                COUNT(DISTINCT NULLIF(TRIM(CPV), ''))
                    AS distinct_cpv,
                COUNT(DISTINCT NULLIF(TRIM(FUTURE_CAN_ID), ''))
                    AS distinct_future_can_ids,
                COUNT(DISTINCT NULLIF(TRIM(CRIT_CODE), ''))
                    AS distinct_crit_codes,
                COUNT(DISTINCT NULLIF(TRIM(CRIT_CRITERIA), ''))
                    AS distinct_crit_criteria
            FROM cn_raw
            WHERE NULLIF(TRIM(ID_NOTICE_CN), '') IS NOT NULL
            GROUP BY ID_NOTICE_CN_CLEAN
        )
        SELECT
            COUNT(*) AS unique_notices,
            MIN(rows_per_notice) AS min_rows_per_notice,
            AVG(rows_per_notice) AS mean_rows_per_notice,
            MEDIAN(rows_per_notice) AS median_rows_per_notice,
            QUANTILE_CONT(rows_per_notice, 0.95) AS p95_rows_per_notice,
            QUANTILE_CONT(rows_per_notice, 0.99) AS p99_rows_per_notice,
            MAX(rows_per_notice) AS max_rows_per_notice,

            COUNT(*) FILTER (
                WHERE distinct_id_lot > 1
            ) AS notices_with_multiple_id_lot,

            COUNT(*) FILTER (
                WHERE distinct_cpv > 1
            ) AS notices_with_multiple_cpv,

            COUNT(*) FILTER (
                WHERE distinct_future_can_ids > 1
            ) AS notices_with_multiple_future_can_ids,

            COUNT(*) FILTER (
                WHERE distinct_crit_codes > 1
                   OR distinct_crit_criteria > 1
            ) AS notices_with_multiple_criteria_records
        FROM per_notice
        """
    )


def profile_rows_per_notice(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        WITH counts AS (
            SELECT
                NULLIF(TRIM(ID_NOTICE_CN), '') AS ID_NOTICE_CN_CLEAN,
                COUNT(*) AS rows_per_notice
            FROM cn_raw
            WHERE NULLIF(TRIM(ID_NOTICE_CN), '') IS NOT NULL
            GROUP BY ID_NOTICE_CN_CLEAN
        )
        SELECT
            rows_per_notice,
            COUNT(*) AS n_notices
        FROM counts
        GROUP BY rows_per_notice
        ORDER BY rows_per_notice
        """
    )


# ---------------------------------------------------------------------
# Link-field structure
# ---------------------------------------------------------------------

def profile_future_can_fields(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COUNT(*) AS raw_rows,

            COUNT(*) FILTER (
                WHERE NULLIF(TRIM(FUTURE_CAN_ID), '') IS NOT NULL
            ) AS rows_with_future_can_id,

            100.0 * COUNT(*) FILTER (
                WHERE NULLIF(TRIM(FUTURE_CAN_ID), '') IS NOT NULL
            ) / COUNT(*) AS future_can_id_coverage_pct,

            COUNT(DISTINCT NULLIF(TRIM(FUTURE_CAN_ID), ''))
                AS distinct_future_can_id_strings,

            COUNT(*) FILTER (
                WHERE NULLIF(TRIM(FUTURE_CAN_ID_ESTIMATED), '') IS NOT NULL
            ) AS rows_with_estimated_flag,

            100.0 * COUNT(*) FILTER (
                WHERE NULLIF(TRIM(FUTURE_CAN_ID_ESTIMATED), '') IS NOT NULL
            ) / COUNT(*) AS estimated_flag_coverage_pct,

            COUNT(*) FILTER (
                WHERE POSITION('---' IN COALESCE(FUTURE_CAN_ID, '')) > 0
            ) AS rows_with_triple_dash_future_can_id,

            COUNT(*) FILTER (
                WHERE POSITION(',' IN COALESCE(FUTURE_CAN_ID, '')) > 0
            ) AS rows_with_comma_future_can_id,

            COUNT(*) FILTER (
                WHERE POSITION(';' IN COALESCE(FUTURE_CAN_ID, '')) > 0
            ) AS rows_with_semicolon_future_can_id,

            COUNT(*) FILTER (
                WHERE POSITION('|' IN COALESCE(FUTURE_CAN_ID, '')) > 0
            ) AS rows_with_pipe_future_can_id

        FROM cn_raw
        """
    )


def profile_estimated_values(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COALESCE(
                NULLIF(TRIM(FUTURE_CAN_ID_ESTIMATED), ''),
                '<BLANK>'
            ) AS future_can_id_estimated_value,
            COUNT(*) AS n_rows,
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                AS share_pct
        FROM cn_raw
        GROUP BY future_can_id_estimated_value
        ORDER BY n_rows DESC
        """
    )


def profile_cn_to_can_cardinality(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        WITH per_cn AS (
            SELECT
                NULLIF(TRIM(ID_NOTICE_CN), '') AS ID_NOTICE_CN_CLEAN,
                COUNT(DISTINCT NULLIF(TRIM(FUTURE_CAN_ID), ''))
                    AS n_distinct_future_can_id_strings
            FROM cn_raw
            WHERE NULLIF(TRIM(ID_NOTICE_CN), '') IS NOT NULL
            GROUP BY ID_NOTICE_CN_CLEAN
        )
        SELECT
            n_distinct_future_can_id_strings,
            COUNT(*) AS n_cn_notices
        FROM per_cn
        GROUP BY n_distinct_future_can_id_strings
        ORDER BY n_distinct_future_can_id_strings
        """
    )


def profile_can_to_cn_cardinality(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        WITH distinct_pairs AS (
            SELECT DISTINCT
                NULLIF(TRIM(ID_NOTICE_CN), '') AS ID_NOTICE_CN_CLEAN,
                NULLIF(TRIM(FUTURE_CAN_ID), '') AS FUTURE_CAN_ID_CLEAN
            FROM cn_raw
            WHERE
                NULLIF(TRIM(ID_NOTICE_CN), '') IS NOT NULL
                AND NULLIF(TRIM(FUTURE_CAN_ID), '') IS NOT NULL
        ),
        per_can AS (
            SELECT
                FUTURE_CAN_ID_CLEAN,
                COUNT(DISTINCT ID_NOTICE_CN_CLEAN)
                    AS n_distinct_cn_notices
            FROM distinct_pairs
            GROUP BY FUTURE_CAN_ID_CLEAN
        )
        SELECT
            n_distinct_cn_notices,
            COUNT(*) AS n_future_can_id_strings
        FROM per_can
        GROUP BY n_distinct_cn_notices
        ORDER BY n_distinct_cn_notices
        """
    )


# ---------------------------------------------------------------------
# Domain coverage reports
# ---------------------------------------------------------------------

def profile_simple_distribution(
    con: duckdb.DuckDBPyConnection,
    column: str,
    top_n: int,
) -> pd.DataFrame:
    c = qident(column)

    return fetch_df(
        con,
        f"""
        SELECT
            COALESCE(
                NULLIF(TRIM(CAST({c} AS VARCHAR)), ''),
                '<BLANK>'
            ) AS value,
            COUNT(*) AS n_rows,
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                AS share_pct
        FROM cn_raw
        GROUP BY value
        ORDER BY n_rows DESC
        LIMIT {int(top_n)}
        """
    )


def profile_cpv(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COUNT(*) AS raw_rows,
            COUNT(*) FILTER (
                WHERE NULLIF(TRIM(CPV), '') IS NOT NULL
            ) AS rows_with_cpv,
            100.0 * COUNT(*) FILTER (
                WHERE NULLIF(TRIM(CPV), '') IS NOT NULL
            ) / COUNT(*) AS cpv_coverage_pct,
            COUNT(DISTINCT NULLIF(TRIM(CPV), ''))
                AS distinct_cpv_strings,
            COUNT(*) FILTER (
                WHERE POSITION('---' IN COALESCE(CPV, '')) > 0
            ) AS rows_with_triple_dash_cpv,
            COUNT(*) FILTER (
                WHERE NULLIF(TRIM(ADDITIONAL_CPVS), '') IS NOT NULL
            ) AS rows_with_additional_cpvs,
            100.0 * COUNT(*) FILTER (
                WHERE NULLIF(TRIM(ADDITIONAL_CPVS), '') IS NOT NULL
            ) / COUNT(*) AS additional_cpvs_coverage_pct
        FROM cn_raw
        """
    )


def profile_criteria_coverage(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    pieces = []

    for col in CRITERIA_FIELDS:
        c = qident(col)
        pieces.append(
            f"""
            SELECT
                '{col}' AS field,
                COUNT(*) AS n_rows,
                COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                ) AS nonmissing_rows,
                100.0 * COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                ) / COUNT(*) AS coverage_pct
            FROM cn_raw
            """
        )

    return fetch_df(
        con,
        "\nUNION ALL\n".join(pieces)
        + "\nORDER BY field"
    )


def profile_value_coverage(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    pieces = []

    for col in VALUE_FIELDS:
        c = qident(col)
        pieces.append(
            f"""
            SELECT
                '{col}' AS field,
                COUNT(*) AS n_rows,
                COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                ) AS nonmissing_rows,
                100.0 * COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                ) / COUNT(*) AS coverage_pct,
                COUNT(*) FILTER (
                    WHERE NULLIF(TRIM(CAST({c} AS VARCHAR)), '') IS NOT NULL
                      AND TRY_CAST(
                          NULLIF(TRIM(CAST({c} AS VARCHAR)), '')
                          AS DOUBLE
                      ) IS NULL
                ) AS nonmissing_non_numeric_rows
            FROM cn_raw
            """
        )

    return fetch_df(
        con,
        "\nUNION ALL\n".join(pieces)
        + "\nORDER BY field"
    )


def profile_lot_structure(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        WITH per_notice AS (
            SELECT
                NULLIF(TRIM(ID_NOTICE_CN), '') AS ID_NOTICE_CN_CLEAN,
                COUNT(*) AS rows_per_notice,
                COUNT(DISTINCT NULLIF(TRIM(ID_LOT), ''))
                    AS n_distinct_id_lot,
                MAX(
                    TRY_CAST(
                        NULLIF(TRIM(LOTS_NUMBER), '')
                        AS DOUBLE
                    )
                ) AS max_reported_lots_number
            FROM cn_raw
            WHERE NULLIF(TRIM(ID_NOTICE_CN), '') IS NOT NULL
            GROUP BY ID_NOTICE_CN_CLEAN
        )
        SELECT
            COUNT(*) AS unique_notices,
            COUNT(*) FILTER (
                WHERE n_distinct_id_lot = 0
            ) AS notices_without_id_lot,
            COUNT(*) FILTER (
                WHERE n_distinct_id_lot = 1
            ) AS notices_with_one_distinct_id_lot,
            COUNT(*) FILTER (
                WHERE n_distinct_id_lot > 1
            ) AS notices_with_multiple_distinct_id_lot,
            MAX(n_distinct_id_lot) AS max_distinct_id_lot_per_notice,
            COUNT(*) FILTER (
                WHERE max_reported_lots_number > 1
            ) AS notices_reporting_multiple_lots
        FROM per_notice
        """
    )


def profile_boolean_encodings(
    con: duckdb.DuckDBPyConnection,
    top_n_per_field: int,
) -> pd.DataFrame:
    frames = []

    for col in BOOLEAN_LIKE_FIELDS:
        c = qident(col)

        df = fetch_df(
            con,
            f"""
            SELECT
                '{col}' AS field,
                COALESCE(
                    NULLIF(TRIM(CAST({c} AS VARCHAR)), ''),
                    '<BLANK>'
                ) AS value,
                COUNT(*) AS n_rows
            FROM cn_raw
            GROUP BY value
            ORDER BY n_rows DESC
            LIMIT {int(top_n_per_field)}
            """
        )

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def profile_list_delimiters(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    fields = [
        "FUTURE_CAN_ID",
        "CAE_NAME",
        "ISO_COUNTRY_CODE_ALL",
        "CPV",
        "ADDITIONAL_CPVS",
        "ID_LOT",
        "CRIT_CRITERIA",
        "CRIT_WEIGHTS",
        "ADMIN_LANGUAGES_TENDER",
    ]

    pieces = []

    for col in fields:
        c = qident(col)

        pieces.append(
            f"""
            SELECT
                '{col}' AS field,
                COUNT(*) FILTER (
                    WHERE POSITION('---' IN COALESCE(CAST({c} AS VARCHAR), '')) > 0
                ) AS triple_dash_rows,
                COUNT(*) FILTER (
                    WHERE POSITION(',' IN COALESCE(CAST({c} AS VARCHAR), '')) > 0
                ) AS comma_rows,
                COUNT(*) FILTER (
                    WHERE POSITION(';' IN COALESCE(CAST({c} AS VARCHAR), '')) > 0
                ) AS semicolon_rows,
                COUNT(*) FILTER (
                    WHERE POSITION('|' IN COALESCE(CAST({c} AS VARCHAR), '')) > 0
                ) AS pipe_rows
            FROM cn_raw
            """
        )

    return fetch_df(
        con,
        "\nUNION ALL\n".join(pieces)
        + "\nORDER BY field"
    )


# ---------------------------------------------------------------------
# Optional linkability to CAN 2017 award fact
# ---------------------------------------------------------------------

def validate_can_award_fact(
    con: duckdb.DuckDBPyConnection,
    can_path: Path,
) -> None:
    cols = {
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{sql_path(can_path)}')
            """
        ).fetchall()
    }

    required = {
        "ID_NOTICE_CAN_CLEAN",
        "SOURCE_YEAR_MIN",
        "SOURCE_YEAR_MAX",
        "AWARD_KEY",
    }

    missing = sorted(required - cols)

    if missing:
        raise ValueError(
            "CAN award fact is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def build_optional_can_linkability(
    con: duckdb.DuckDBPyConnection,
    can_path: Path,
) -> dict[str, pd.DataFrame]:
    validate_can_award_fact(con, can_path)

    con.execute(
        f"""
        CREATE OR REPLACE VIEW can_2017_notice_summary AS
        SELECT
            ID_NOTICE_CAN_CLEAN,
            COUNT(DISTINCT AWARD_KEY) AS n_award_keys,
            COUNT(*) AS n_award_fact_rows
        FROM read_parquet('{sql_path(can_path)}')
        WHERE
            SOURCE_YEAR_MIN = 2017
            AND SOURCE_YEAR_MAX = 2017
            AND ID_NOTICE_CAN_CLEAN IS NOT NULL
        GROUP BY ID_NOTICE_CAN_CLEAN
        """
    )

    # Direct exact-string link only. Multi-valued FUTURE_CAN_ID strings are
    # reported separately and are not split automatically here.
    overview = fetch_df(
        con,
        """
        WITH cn_notices AS (
            SELECT
                NULLIF(TRIM(ID_NOTICE_CN), '') AS ID_NOTICE_CN_CLEAN,
                NULLIF(TRIM(FUTURE_CAN_ID), '') AS FUTURE_CAN_ID_CLEAN,
                NULLIF(TRIM(FUTURE_CAN_ID_ESTIMATED), '')
                    AS FUTURE_CAN_ID_ESTIMATED_CLEAN
            FROM cn_raw
            WHERE NULLIF(TRIM(ID_NOTICE_CN), '') IS NOT NULL
        ),
        notice_level AS (
            SELECT
                ID_NOTICE_CN_CLEAN,
                MAX(FUTURE_CAN_ID_CLEAN) AS FUTURE_CAN_ID_CLEAN,
                MAX(FUTURE_CAN_ID_ESTIMATED_CLEAN)
                    AS FUTURE_CAN_ID_ESTIMATED_CLEAN,
                COUNT(DISTINCT FUTURE_CAN_ID_CLEAN) FILTER (
                    WHERE FUTURE_CAN_ID_CLEAN IS NOT NULL
                ) AS n_distinct_future_can_id_strings
            FROM cn_notices
            GROUP BY ID_NOTICE_CN_CLEAN
        )
        SELECT
            COUNT(*) AS cn_notices,
            COUNT(*) FILTER (
                WHERE FUTURE_CAN_ID_CLEAN IS NOT NULL
            ) AS cn_notices_with_future_can_id,

            COUNT(*) FILTER (
                WHERE FUTURE_CAN_ID_CLEAN IS NOT NULL
                  AND n_distinct_future_can_id_strings = 1
            ) AS cn_notices_with_single_future_can_id_string,

            COUNT(*) FILTER (
                WHERE c.ID_NOTICE_CAN_CLEAN IS NOT NULL
            ) AS cn_notices_exactly_linkable_to_can_2017_notice,

            100.0 * COUNT(*) FILTER (
                WHERE c.ID_NOTICE_CAN_CLEAN IS NOT NULL
            ) / COUNT(*) AS exact_linkable_share_of_all_cn_notices_pct,

            100.0 * COUNT(*) FILTER (
                WHERE c.ID_NOTICE_CAN_CLEAN IS NOT NULL
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE FUTURE_CAN_ID_CLEAN IS NOT NULL
                ),
                0
            ) AS exact_linkable_share_of_future_can_populated_cn_pct

        FROM notice_level n
        LEFT JOIN can_2017_notice_summary c
          ON n.FUTURE_CAN_ID_CLEAN = c.ID_NOTICE_CAN_CLEAN
        """
    )

    by_estimated = fetch_df(
        con,
        """
        WITH notice_level AS (
            SELECT
                NULLIF(TRIM(ID_NOTICE_CN), '') AS ID_NOTICE_CN_CLEAN,
                MAX(NULLIF(TRIM(FUTURE_CAN_ID), ''))
                    AS FUTURE_CAN_ID_CLEAN,
                MAX(NULLIF(TRIM(FUTURE_CAN_ID_ESTIMATED), ''))
                    AS FUTURE_CAN_ID_ESTIMATED_CLEAN,
                COUNT(DISTINCT NULLIF(TRIM(FUTURE_CAN_ID), '')) FILTER (
                    WHERE NULLIF(TRIM(FUTURE_CAN_ID), '') IS NOT NULL
                ) AS n_distinct_future_can_id_strings
            FROM cn_raw
            WHERE NULLIF(TRIM(ID_NOTICE_CN), '') IS NOT NULL
            GROUP BY ID_NOTICE_CN_CLEAN
        )
        SELECT
            COALESCE(
                n.FUTURE_CAN_ID_ESTIMATED_CLEAN,
                '<BLANK>'
            ) AS estimated_status,
            COUNT(*) AS n_cn_notices,
            COUNT(*) FILTER (
                WHERE n.FUTURE_CAN_ID_CLEAN IS NOT NULL
            ) AS n_with_future_can_id,
            COUNT(*) FILTER (
                WHERE c.ID_NOTICE_CAN_CLEAN IS NOT NULL
            ) AS n_exactly_linkable,
            100.0 * COUNT(*) FILTER (
                WHERE c.ID_NOTICE_CAN_CLEAN IS NOT NULL
            ) / COUNT(*) AS exactly_linkable_pct
        FROM notice_level n
        LEFT JOIN can_2017_notice_summary c
          ON n.FUTURE_CAN_ID_CLEAN = c.ID_NOTICE_CAN_CLEAN
        GROUP BY estimated_status
        ORDER BY n_cn_notices DESC
        """
    )

    award_multiplicity = fetch_df(
        con,
        """
        WITH notice_level AS (
            SELECT
                NULLIF(TRIM(ID_NOTICE_CN), '') AS ID_NOTICE_CN_CLEAN,
                MAX(NULLIF(TRIM(FUTURE_CAN_ID), ''))
                    AS FUTURE_CAN_ID_CLEAN
            FROM cn_raw
            WHERE NULLIF(TRIM(ID_NOTICE_CN), '') IS NOT NULL
            GROUP BY ID_NOTICE_CN_CLEAN
        )
        SELECT
            c.n_award_keys,
            COUNT(*) AS n_linked_cn_notices
        FROM notice_level n
        JOIN can_2017_notice_summary c
          ON n.FUTURE_CAN_ID_CLEAN = c.ID_NOTICE_CAN_CLEAN
        GROUP BY c.n_award_keys
        ORDER BY c.n_award_keys
        """
    )

    return {
        "17_linkability_to_can_2017.csv": overview,
        "18_linkability_by_estimated_status.csv": by_estimated,
        "19_linked_can_award_multiplicity.csv": award_multiplicity,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile TED CN 2017 for target-case construction."
    )

    parser.add_argument(
        "--cn-2017",
        required=True,
        type=Path,
        help="Path to the raw CN 2017 CSV file.",
    )

    parser.add_argument(
        "--can-award-fact",
        type=Path,
        default=None,
        help=(
            "Optional path to award_fact_2015_2017.parquet. "
            "When provided, the profiler also reports direct exact-string "
            "CN FUTURE_CAN_ID linkability to 2017 CAN notice IDs."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/cn_2017_profile"),
    )

    parser.add_argument(
        "--work-db",
        type=Path,
        default=None,
    )

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
        "--top-n-distribution",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--failed-date-example-limit",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    if not args.cn_2017.exists():
        raise FileNotFoundError(
            f"CN 2017 file not found: {args.cn_2017}"
        )

    if (
        args.can_award_fact is not None
        and not args.can_award_fact.exists()
    ):
        raise FileNotFoundError(
            f"CAN award fact not found: {args.can_award_fact}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    delimiter = detect_delimiter(args.cn_2017)

    schema_df = validate_schema(args.cn_2017, delimiter)

    if not bool(schema_df.iloc[0]["same_column_set"]):
        raise ValueError(
            "CN 2017 schema does not match the expected column set. "
            "See schema diagnostics."
        )

    work_db = (
        args.work_db
        if args.work_db is not None
        else args.output_dir / "_cn_2017_profile.duckdb"
    )
    work_db.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(work_db))
    con.execute(f"SET threads={int(args.threads)}")

    if args.memory_limit:
        memory_limit = args.memory_limit.replace("'", "''")
        con.execute(f"SET memory_limit='{memory_limit}'")

    temp_dir = args.output_dir / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"SET temp_directory='{sql_path(temp_dir)}'"
    )

    try:
        print("1/12 Loading CN 2017...")
        create_raw_table(
            con,
            args.cn_2017,
            delimiter,
        )

        print("2/12 Profiling overview and dates...")
        reports: dict[str, pd.DataFrame] = {
            "01_schema_overview.csv": schema_df,
            "02_dataset_overview.csv": profile_overview(con),
            "03_date_parse_report.csv": profile_date_parse(con),
            "03b_failed_date_examples.csv":
                profile_failed_date_examples(
                    con,
                    args.failed_date_example_limit,
                ),
        }

        print("3/12 Profiling missingness...")
        reports["04_missingness.csv"] = profile_missingness(con)

        print("4/12 Profiling notice grain...")
        reports["05_notice_grain_summary.csv"] = (
            profile_notice_grain(con)
        )
        reports["06_rows_per_notice_distribution.csv"] = (
            profile_rows_per_notice(con)
        )

        print("5/12 Profiling future-CAN link fields...")
        reports["07_future_can_field_summary.csv"] = (
            profile_future_can_fields(con)
        )
        reports["08_future_can_estimated_values.csv"] = (
            profile_estimated_values(con)
        )
        reports["09_cn_to_can_cardinality.csv"] = (
            profile_cn_to_can_cardinality(con)
        )
        reports["10_can_to_cn_cardinality.csv"] = (
            profile_can_to_cn_cardinality(con)
        )

        print("6/12 Profiling CPV and procurement context...")
        reports["11_cpv_summary.csv"] = profile_cpv(con)
        reports["11b_top_cpv_values.csv"] = (
            profile_simple_distribution(
                con,
                "CPV",
                args.top_n_distribution,
            )
        )
        reports["12_contract_type_distribution.csv"] = (
            profile_simple_distribution(
                con,
                "TYPE_OF_CONTRACT",
                args.top_n_distribution,
            )
        )
        reports["13_country_distribution.csv"] = (
            profile_simple_distribution(
                con,
                "ISO_COUNTRY_CODE",
                args.top_n_distribution,
            )
        )

        print("7/12 Profiling criteria and value fields...")
        reports["14_criteria_coverage.csv"] = (
            profile_criteria_coverage(con)
        )
        reports["15_value_coverage.csv"] = (
            profile_value_coverage(con)
        )

        print("8/12 Profiling lot structure...")
        reports["16_lot_structure.csv"] = (
            profile_lot_structure(con)
        )

        print("9/12 Profiling encodings and XSD versions...")
        reports["16b_boolean_value_encodings.csv"] = (
            profile_boolean_encodings(
                con,
                top_n_per_field=20,
            )
        )
        reports["16c_xsd_distribution.csv"] = (
            profile_simple_distribution(
                con,
                "XSD_VERSION",
                args.top_n_distribution,
            )
        )
        reports["16d_list_delimiter_signals.csv"] = (
            profile_list_delimiters(con)
        )

        if args.can_award_fact is not None:
            print("10/12 Profiling exact-string linkability to CAN 2017...")
            reports.update(
                build_optional_can_linkability(
                    con,
                    args.can_award_fact,
                )
            )
        else:
            print(
                "10/12 Skipping CAN linkability "
                "(--can-award-fact not supplied)..."
            )

        print("11/12 Writing reports...")
        for filename, df in reports.items():
            write_csv(
                df,
                args.output_dir / filename,
            )

        print("12/12 Complete.")
        print()
        print("=" * 78)
        print("CN 2017 profiling complete")
        print("=" * 78)
        print(f"Reports: {args.output_dir.resolve()}")
        print(f"Work DB: {work_db.resolve()}")
        print()
        print("Review these first:")
        print("  02_dataset_overview.csv")
        print("  03_date_parse_report.csv")
        print("  05_notice_grain_summary.csv")
        print("  07_future_can_field_summary.csv")
        print("  08_future_can_estimated_values.csv")
        print("  09_cn_to_can_cardinality.csv")
        print("  10_can_to_cn_cardinality.csv")
        print("  14_criteria_coverage.csv")
        print("  15_value_coverage.csv")
        print("  16_lot_structure.csv")
        if args.can_award_fact is not None:
            print("  17_linkability_to_can_2017.csv")
            print("  18_linkability_by_estimated_status.csv")
            print("  19_linked_can_award_multiplicity.csv")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
