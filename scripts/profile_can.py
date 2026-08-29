from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATE_COLUMNS = [
    "DT_DISPATCH",
    "DT_AWARD",
]

LIST_LIKE_COLUMNS = [
    "ADDITIONAL_CPVS",
    "CRIT_CRITERIA",
    "CRIT_WEIGHTS",
    "ISO_COUNTRY_CODE_ALL",
    "B_CONTRACTOR_SME",
]

SUPPLIER_ID_COLUMNS = [
    "WIN_NAME",
    "WIN_NATIONALID",
    "WIN_COUNTRY_CODE",
    "WIN_TOWN",
]

AWARD_VALUE_COLUMNS = [
    "AWARD_EST_VALUE_EURO",
    "AWARD_VALUE_EURO",
    "AWARD_VALUE_EURO_FIN_1",
]

CRITERIA_COLUMNS = [
    "CRIT_CODE",
    "CRIT_PRICE_WEIGHT",
    "CRIT_CRITERIA",
    "CRIT_WEIGHTS",
]

COMPETITION_COLUMNS = [
    "NUMBER_OFFERS",
    "NUMBER_TENDERS_SME",
    "NUMBER_TENDERS_OTHER_EU",
    "NUMBER_TENDERS_NON_EU",
    "NUMBER_OFFERS_ELECTR",
]

IMPORTANT_ID_COLUMNS = [
    "ID_NOTICE_CAN",
    "ID_AWARD",
    "ID_LOT",
    "ID_LOT_AWARDED",
    "CONTRACT_NUMBER",
]

XSD_COVERAGE_COLUMNS = [
    "WIN_NAME",
    "WIN_NATIONALID",
    "WIN_COUNTRY_CODE",
    "B_CONTRACTOR_SME",
    "ID_LOT",
    "ID_LOT_AWARDED",
    "B_AWARDED_TO_A_GROUP",
    "AWARD_EST_VALUE_EURO",
    "AWARD_VALUE_EURO",
    "AWARD_VALUE_EURO_FIN_1",
    "NUMBER_OFFERS",
    "CRIT_CODE",
    "CRIT_PRICE_WEIGHT",
    "CRIT_CRITERIA",
    "CRIT_WEIGHTS",
]

REPEATED_AWARD_VARIABILITY_COLUMNS = [
    "ID_LOT",
    "ID_LOT_AWARDED",
    "WIN_NAME",
    "WIN_NATIONALID",
    "WIN_COUNTRY_CODE",
    "B_CONTRACTOR_SME",
    "B_AWARDED_TO_A_GROUP",
    "CONTRACT_NUMBER",
    "TITLE",
    "CPV",
    "CRIT_CODE",
    "CRIT_PRICE_WEIGHT",
    "CRIT_CRITERIA",
    "CRIT_WEIGHTS",
    "NUMBER_OFFERS",
    "AWARD_EST_VALUE_EURO",
    "AWARD_VALUE_EURO",
    "AWARD_VALUE_EURO_FIN_1",
    "DT_AWARD",
]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def quote_sql_string(value: str) -> str:
    """Escape a Python string for embedding in a DuckDB SQL string literal."""
    return value.replace("'", "''")


def qcol(column: str) -> str:
    """Quote a SQL identifier."""
    return '"' + column.replace('"', '""') + '"'


def detect_delimiter(path: Path) -> str:
    """Detect CSV delimiter from the first 64 KB, with tab as fallback."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(65536)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
        return dialect.delimiter
    except csv.Error:
        return "\t"


def read_header(path: Path, delimiter: str) -> list[str]:
    """Read only the header row."""
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return next(reader)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def sql_nonmissing(column: str) -> str:
    """SQL predicate for a meaningful non-empty value."""
    c = qcol(column)
    return f"{c} IS NOT NULL AND TRIM(CAST({c} AS VARCHAR)) <> ''"


def csv_relation(path: Path, delimiter: str) -> str:
    """
    DuckDB relation expression for robust profiling.

    all_varchar=true is deliberate: identifiers preserve leading zeros and
    cross-year type inconsistencies do not break the profiling pass.
    """
    path_sql = quote_sql_string(str(path.resolve()))
    delim_sql = quote_sql_string(delimiter)

    return (
        "read_csv("
        f"'{path_sql}', "
        "header=true, "
        f"delim='{delim_sql}', "
        "all_varchar=true, "
        "ignore_errors=false, "
        "null_padding=true"
        ")"
    )


def inferred_csv_relation(path: Path, delimiter: str) -> str:
    """Relation used only to observe DuckDB's inferred physical types."""
    path_sql = quote_sql_string(str(path.resolve()))
    delim_sql = quote_sql_string(delimiter)

    return (
        "read_csv_auto("
        f"'{path_sql}', "
        "header=true, "
        f"delim='{delim_sql}', "
        "sample_size=-1"
        ")"
    )


# -----------------------------------------------------------------------------
# Date parsing helpers
# -----------------------------------------------------------------------------

def date_parse_expression(column: str) -> str:
    """
    Parse TED date formats observed across CAN 2015-2017.

    Observed:
      23-DEC-14 -> %d-%b-%y
      22/12/16  -> %d/%m/%y

    Additional fallbacks are retained for robustness.
    """
    c = f"TRIM(CAST({qcol(column)} AS VARCHAR))"

    return f"""
    COALESCE(
        TRY_STRPTIME({c}, '%d-%b-%y'),
        TRY_STRPTIME({c}, '%d/%m/%y'),
        TRY_STRPTIME({c}, '%Y-%m-%d'),
        TRY_STRPTIME({c}, '%d/%m/%Y'),
        TRY_STRPTIME({c}, '%Y%m%d'),
        TRY_STRPTIME({c}, '%d-%m-%Y')
    )
    """


# -----------------------------------------------------------------------------
# Schema analysis
# -----------------------------------------------------------------------------

def check_schema_compatibility(
    files: dict[int, Path],
    delimiters: dict[int, str],
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    headers = {
        year: read_header(path, delimiters[year])
        for year, path in files.items()
    }

    reference_year = min(headers)
    reference = headers[reference_year]
    rows: list[dict[str, object]] = []

    for year, columns in headers.items():
        rows.append(
            {
                "year": year,
                "n_columns": len(columns),
                "same_column_set_as_reference": set(columns) == set(reference),
                "same_column_order_as_reference": columns == reference,
                "missing_columns": json.dumps(sorted(set(reference) - set(columns))),
                "extra_columns": json.dumps(sorted(set(columns) - set(reference))),
            }
        )

    return pd.DataFrame(rows), headers


def profile_inferred_types(
    con: duckdb.DuckDBPyConnection,
    files: dict[int, Path],
    delimiters: dict[int, str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for year, path in files.items():
        relation = inferred_csv_relation(path, delimiters[year])
        try:
            df = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchdf()
            df["year"] = year
            frames.append(df)
        except Exception as exc:  # pragma: no cover - diagnostic path
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "column_name": "__ERROR__",
                            "column_type": str(exc),
                            "year": year,
                        }
                    ]
                )
            )

    return concat_or_empty(frames)


def compare_inferred_types(type_df: pd.DataFrame) -> pd.DataFrame:
    if type_df.empty or "column_name" not in type_df.columns:
        return pd.DataFrame()

    clean = type_df[type_df["column_name"] != "__ERROR__"].copy()
    if clean.empty:
        return pd.DataFrame()

    pivot = clean.pivot_table(
        index="column_name",
        columns="year",
        values="column_type",
        aggfunc="first",
    ).reset_index()

    year_cols = [column for column in pivot.columns if column != "column_name"]

    def consistent(row: pd.Series) -> bool:
        values = [row[column] for column in year_cols if pd.notna(row[column])]
        return len(set(values)) <= 1

    pivot["types_consistent"] = pivot.apply(consistent, axis=1)
    return pivot


# -----------------------------------------------------------------------------
# Missingness and field coverage
# -----------------------------------------------------------------------------

def profile_missingness(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    columns: list[str],
) -> pd.DataFrame:
    total = con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
    rows: list[dict[str, object]] = []

    for column in columns:
        condition = sql_nonmissing(column)
        nonmissing = con.execute(
            f"SELECT COUNT(*) FROM {relation} WHERE {condition}"
        ).fetchone()[0]

        rows.append(
            {
                "year": year,
                "column": column,
                "total_rows": total,
                "nonmissing_rows": nonmissing,
                "missing_rows": total - nonmissing,
                "coverage_pct": (100.0 * nonmissing / total if total else None),
            }
        )

    return pd.DataFrame(rows)


def profile_field_group_coverage(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    group_name: str,
    columns: list[str],
) -> pd.DataFrame:
    total = con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
    rows: list[dict[str, object]] = []

    for column in columns:
        count = con.execute(
            f"SELECT COUNT(*) FROM {relation} WHERE {sql_nonmissing(column)}"
        ).fetchone()[0]

        rows.append(
            {
                "year": year,
                "field_group": group_name,
                "column": column,
                "nonmissing_rows": count,
                "total_rows": total,
                "coverage_pct": (100.0 * count / total if total else None),
            }
        )

    return pd.DataFrame(rows)


def profile_xsd_field_coverage(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    columns: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for column in columns:
        if not column:
            continue

        df = con.execute(
            f"""
            SELECT
                {year} AS year,
                XSD_VERSION,
                '{quote_sql_string(column)}' AS column_name,
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (WHERE {sql_nonmissing(column)}) AS nonmissing_rows,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE {sql_nonmissing(column)})
                    / NULLIF(COUNT(*), 0),
                    3
                ) AS coverage_pct
            FROM {relation}
            GROUP BY XSD_VERSION
            ORDER BY XSD_VERSION
            """
        ).fetchdf()

        frames.append(df)

    return concat_or_empty(frames)


# -----------------------------------------------------------------------------
# Boolean and categorical encodings
# -----------------------------------------------------------------------------

def identify_boolean_like_columns(columns: Iterable[str]) -> list[str]:
    result: list[str] = []

    for column in columns:
        if (
            column.startswith("B_")
            or column
            in {
                "CANCELLED",
                "CORRECTIONS",
                "OUT_OF_DIRECTIVES",
                "INFO_UNPUBLISHED",
            }
        ):
            result.append(column)

    return sorted(result)


def profile_distinct_values(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    columns: list[str],
    top_n: int = 25,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for column in columns:
        c = qcol(column)
        df = con.execute(
            f"""
            SELECT
                {year} AS year,
                '{quote_sql_string(column)}' AS column_name,
                CAST({c} AS VARCHAR) AS value,
                COUNT(*) AS n
            FROM {relation}
            GROUP BY {c}
            ORDER BY n DESC
            LIMIT {top_n}
            """
        ).fetchdf()
        frames.append(df)

    return concat_or_empty(frames)


# -----------------------------------------------------------------------------
# Date profiling
# -----------------------------------------------------------------------------

def profile_date_parsing(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in columns:
        nonmissing = sql_nonmissing(column)
        parsed = date_parse_expression(column)

        nonmissing_n, parsed_n, failed_n = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE {nonmissing}) AS nonmissing,
                COUNT(*) FILTER (
                    WHERE {nonmissing} AND ({parsed}) IS NOT NULL
                ) AS parsed,
                COUNT(*) FILTER (
                    WHERE {nonmissing} AND ({parsed}) IS NULL
                ) AS failed
            FROM {relation}
            """
        ).fetchone()

        rows.append(
            {
                "year": year,
                "column": column,
                "nonmissing": nonmissing_n,
                "parsed": parsed_n,
                "failed": failed_n,
                "parse_success_pct": (
                    100.0 * parsed_n / nonmissing_n if nonmissing_n else None
                ),
            }
        )

    return pd.DataFrame(rows)


def profile_failed_date_examples(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    columns: list[str],
    top_n: int = 30,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for column in columns:
        c = qcol(column)
        parsed = date_parse_expression(column)

        df = con.execute(
            f"""
            SELECT
                {year} AS year,
                '{quote_sql_string(column)}' AS column_name,
                CAST({c} AS VARCHAR) AS raw_value,
                COUNT(*) AS n
            FROM {relation}
            WHERE {sql_nonmissing(column)}
              AND ({parsed}) IS NULL
            GROUP BY {c}
            ORDER BY n DESC
            LIMIT {top_n}
            """
        ).fetchdf()
        frames.append(df)

    return concat_or_empty(frames)


def profile_temporal_sanity(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    dispatch = date_parse_expression("DT_DISPATCH")
    award = date_parse_expression("DT_AWARD")

    return con.execute(
        f"""
        WITH dates AS (
            SELECT
                {dispatch} AS dispatch_date,
                {award} AS award_date
            FROM {relation}
        ),
        valid_pairs AS (
            SELECT
                dispatch_date,
                award_date,
                DATE_DIFF('day', dispatch_date, award_date) AS lag_days
            FROM dates
            WHERE dispatch_date IS NOT NULL
              AND award_date IS NOT NULL
        )
        SELECT
            {year} AS year,
            COUNT(*) AS valid_date_pairs,
            COUNT(*) FILTER (WHERE lag_days < 0) AS negative_lag_count,
            MIN(lag_days) AS min_lag_days,
            MEDIAN(lag_days) AS median_lag_days,
            AVG(lag_days) AS mean_lag_days,
            QUANTILE_CONT(lag_days, 0.95) AS p95_lag_days,
            QUANTILE_CONT(lag_days, 0.99) AS p99_lag_days,
            MAX(lag_days) AS max_lag_days
        FROM valid_pairs
        """
    ).fetchdf()


def profile_file_year_offsets(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    dispatch = date_parse_expression("DT_DISPATCH")
    award = date_parse_expression("DT_AWARD")

    return con.execute(
        f"""
        WITH parsed AS (
            SELECT
                {dispatch} AS dispatch_date,
                {award} AS award_date
            FROM {relation}
        ),
        offsets AS (
            SELECT
                {year} - EXTRACT(YEAR FROM dispatch_date) AS dispatch_year_offset,
                {year} - EXTRACT(YEAR FROM award_date) AS award_year_offset
            FROM parsed
        )
        SELECT
            {year} AS year,
            'DT_DISPATCH' AS date_column,
            CAST(dispatch_year_offset AS BIGINT) AS year_offset,
            COUNT(*) AS n
        FROM offsets
        WHERE dispatch_year_offset IS NOT NULL
        GROUP BY dispatch_year_offset

        UNION ALL

        SELECT
            {year} AS year,
            'DT_AWARD' AS date_column,
            CAST(award_year_offset AS BIGINT) AS year_offset,
            COUNT(*) AS n
        FROM offsets
        WHERE award_year_offset IS NOT NULL
        GROUP BY award_year_offset

        ORDER BY date_column, year_offset
        """
    ).fetchdf()


# -----------------------------------------------------------------------------
# List-like field inspection
# -----------------------------------------------------------------------------

def profile_list_like_examples(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    columns: list[str],
    top_n: int = 30,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for column in columns:
        c = qcol(column)
        df = con.execute(
            f"""
            SELECT
                {year} AS year,
                '{quote_sql_string(column)}' AS column_name,
                CAST({c} AS VARCHAR) AS example_value,
                COUNT(*) AS n
            FROM {relation}
            WHERE {sql_nonmissing(column)}
            GROUP BY {c}
            ORDER BY n DESC
            LIMIT {top_n}
            """
        ).fetchdf()
        frames.append(df)

    return concat_or_empty(frames)


def profile_delimiter_signals(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    columns: list[str],
) -> pd.DataFrame:
    delimiters = {
        "semicolon": ";",
        "pipe": "|",
        "comma": ",",
        "double_hyphen": "--",
        "triple_hyphen": "---",
        "at_sign": "@",
        "hash": "#",
    }

    rows: list[dict[str, object]] = []

    for column in columns:
        c = f"CAST({qcol(column)} AS VARCHAR)"
        for delimiter_name, delimiter in delimiters.items():
            delimiter_sql = quote_sql_string(delimiter)
            count = con.execute(
                f"""
                SELECT COUNT(*)
                FROM {relation}
                WHERE {sql_nonmissing(column)}
                  AND CONTAINS({c}, '{delimiter_sql}')
                """
            ).fetchone()[0]

            rows.append(
                {
                    "year": year,
                    "column": column,
                    "delimiter": delimiter_name,
                    "rows_containing_delimiter": count,
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# XSD version distribution
# -----------------------------------------------------------------------------

def profile_xsd_versions(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT
            {year} AS year,
            XSD_VERSION,
            COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct
        FROM {relation}
        GROUP BY XSD_VERSION
        ORDER BY n DESC
        """
    ).fetchdf()


# -----------------------------------------------------------------------------
# Grain and cardinality
# -----------------------------------------------------------------------------

def profile_grain(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    queries = {
        "total_rows": f"SELECT COUNT(*) FROM {relation}",
        "unique_notice_ids": f"""
            SELECT COUNT(DISTINCT ID_NOTICE_CAN)
            FROM {relation}
        """,
        "unique_notice_award_pairs": f"""
            SELECT COUNT(DISTINCT CONCAT_WS(
                '||',
                COALESCE(ID_NOTICE_CAN, ''),
                COALESCE(ID_AWARD, '')
            ))
            FROM {relation}
            WHERE {sql_nonmissing('ID_AWARD')}
        """,
        "unique_notice_award_lot_pairs": f"""
            SELECT COUNT(DISTINCT CONCAT_WS(
                '||',
                COALESCE(ID_NOTICE_CAN, ''),
                COALESCE(ID_AWARD, ''),
                COALESCE(ID_LOT_AWARDED, '')
            ))
            FROM {relation}
            WHERE {sql_nonmissing('ID_AWARD')}
        """,
        "unique_winner_names": f"""
            SELECT COUNT(DISTINCT WIN_NAME)
            FROM {relation}
            WHERE {sql_nonmissing('WIN_NAME')}
        """,
    }

    rows: list[dict[str, object]] = []
    for metric, query in queries.items():
        value = con.execute(query).fetchone()[0]
        rows.append({"year": year, "metric": metric, "value": value})

    return pd.DataFrame(rows)


def profile_rows_per_notice(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH counts AS (
            SELECT
                ID_NOTICE_CAN,
                COUNT(*) AS rows_per_notice,
                COUNT(DISTINCT ID_AWARD) AS awards_per_notice,
                COUNT(DISTINCT ID_LOT_AWARDED) AS awarded_lots_per_notice
            FROM {relation}
            GROUP BY ID_NOTICE_CAN
        )
        SELECT
            {year} AS year,
            MIN(rows_per_notice) AS min_rows_per_notice,
            MEDIAN(rows_per_notice) AS median_rows_per_notice,
            AVG(rows_per_notice) AS mean_rows_per_notice,
            MAX(rows_per_notice) AS max_rows_per_notice,
            MIN(awards_per_notice) AS min_awards_per_notice,
            MEDIAN(awards_per_notice) AS median_awards_per_notice,
            AVG(awards_per_notice) AS mean_awards_per_notice,
            MAX(awards_per_notice) AS max_awards_per_notice,
            MIN(awarded_lots_per_notice) AS min_lots_per_notice,
            MEDIAN(awarded_lots_per_notice) AS median_lots_per_notice,
            AVG(awarded_lots_per_notice) AS mean_lots_per_notice,
            MAX(awarded_lots_per_notice) AS max_lots_per_notice
        FROM counts
        """
    ).fetchdf()


def profile_rows_per_award(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH counts AS (
            SELECT
                ID_NOTICE_CAN,
                ID_AWARD,
                COUNT(*) AS rows_per_award,
                COUNT(DISTINCT ID_LOT_AWARDED) AS lots_per_award,
                COUNT(DISTINCT WIN_NAME) AS winner_names_per_award
            FROM {relation}
            WHERE {sql_nonmissing('ID_AWARD')}
            GROUP BY ID_NOTICE_CAN, ID_AWARD
        )
        SELECT
            {year} AS year,
            MIN(rows_per_award) AS min_rows_per_award,
            MEDIAN(rows_per_award) AS median_rows_per_award,
            AVG(rows_per_award) AS mean_rows_per_award,
            MAX(rows_per_award) AS max_rows_per_award,
            MIN(lots_per_award) AS min_lots_per_award,
            MEDIAN(lots_per_award) AS median_lots_per_award,
            AVG(lots_per_award) AS mean_lots_per_award,
            MAX(lots_per_award) AS max_lots_per_award,
            MIN(winner_names_per_award) AS min_winners_per_award,
            MEDIAN(winner_names_per_award) AS median_winners_per_award,
            AVG(winner_names_per_award) AS mean_winners_per_award,
            MAX(winner_names_per_award) AS max_winners_per_award
        FROM counts
        """
    ).fetchdf()


def classify_notice_cardinality(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH summary AS (
            SELECT
                ID_NOTICE_CAN,
                COUNT(*) AS n_rows,
                COUNT(DISTINCT ID_AWARD) AS n_awards,
                COUNT(DISTINCT ID_LOT_AWARDED) AS n_awarded_lots
            FROM {relation}
            GROUP BY ID_NOTICE_CAN
        ),
        classified AS (
            SELECT
                CASE
                    WHEN n_rows = 1 AND n_awards <= 1 AND n_awarded_lots <= 1
                        THEN 'simple'
                    WHEN n_awards > 1 AND n_awarded_lots <= 1
                        THEN 'multiple_awards'
                    WHEN n_awards <= 1 AND n_awarded_lots > 1
                        THEN 'multiple_lots'
                    WHEN n_awards > 1 AND n_awarded_lots > 1
                        THEN 'multiple_awards_and_lots'
                    ELSE 'other'
                END AS structure_type
            FROM summary
        )
        SELECT
            {year} AS year,
            structure_type,
            COUNT(*) AS n_notices
        FROM classified
        GROUP BY structure_type
        ORDER BY n_notices DESC
        """
    ).fetchdf()


# -----------------------------------------------------------------------------
# Duplicate and repeated-award diagnostics
# -----------------------------------------------------------------------------

def profile_business_key_duplicates(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH grouped AS (
            SELECT
                ID_NOTICE_CAN,
                ID_AWARD,
                ID_LOT_AWARDED,
                WIN_NAME,
                COUNT(*) AS n
            FROM {relation}
            GROUP BY ID_NOTICE_CAN, ID_AWARD, ID_LOT_AWARDED, WIN_NAME
        )
        SELECT
            {year} AS year,
            COUNT(*) FILTER (WHERE n > 1) AS duplicate_key_groups,
            COALESCE(SUM(n - 1) FILTER (WHERE n > 1), 0) AS excess_rows
        FROM grouped
        """
    ).fetchdf()


def profile_exact_duplicates(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH total AS (
            SELECT COUNT(*) AS n FROM {relation}
        ),
        unique_rows AS (
            SELECT COUNT(*) AS n
            FROM (SELECT DISTINCT * FROM {relation})
        )
        SELECT
            {year} AS year,
            total.n AS total_rows,
            unique_rows.n AS unique_rows,
            total.n - unique_rows.n AS exact_duplicate_rows
        FROM total, unique_rows
        """
    ).fetchdf()


def profile_award_like_repetition(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH award_like AS (
            SELECT *
            FROM {relation}
            WHERE {sql_nonmissing('ID_AWARD')}
              AND {sql_nonmissing('WIN_NAME')}
        ),
        grouped AS (
            SELECT
                ID_NOTICE_CAN,
                ID_AWARD,
                COUNT(*) AS n_rows,
                COUNT(DISTINCT WIN_NAME) AS n_winner_names,
                COUNT(DISTINCT ID_LOT_AWARDED) AS n_lots
            FROM award_like
            GROUP BY ID_NOTICE_CAN, ID_AWARD
        )
        SELECT
            {year} AS year,
            COUNT(*) AS award_like_key_count,
            COUNT(*) FILTER (WHERE n_rows > 1) AS repeated_award_key_count,
            SUM(n_rows) AS award_like_rows,
            SUM(n_rows - 1) FILTER (WHERE n_rows > 1) AS excess_rows_in_repeated_keys,
            MAX(n_rows) AS max_rows_per_award_key,
            COUNT(*) FILTER (WHERE n_winner_names > 1) AS repeated_keys_with_multiple_winner_names,
            COUNT(*) FILTER (WHERE n_lots > 1) AS repeated_keys_with_multiple_lots
        FROM grouped
        """
    ).fetchdf()


def profile_repeated_award_variability(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in columns:
        c = qcol(column)
        result = con.execute(
            f"""
            WITH repeated_keys AS (
                SELECT ID_NOTICE_CAN, ID_AWARD
                FROM {relation}
                WHERE {sql_nonmissing('ID_AWARD')}
                  AND {sql_nonmissing('WIN_NAME')}
                GROUP BY ID_NOTICE_CAN, ID_AWARD
                HAVING COUNT(*) > 1
            ),
            group_variation AS (
                SELECT
                    r.ID_NOTICE_CAN,
                    r.ID_AWARD,
                    COUNT(DISTINCT CAST(t.{c} AS VARCHAR))
                        FILTER (WHERE t.{c} IS NOT NULL) AS distinct_nonnull_values
                FROM repeated_keys r
                JOIN {relation} t
                  ON t.ID_NOTICE_CAN = r.ID_NOTICE_CAN
                 AND t.ID_AWARD = r.ID_AWARD
                GROUP BY r.ID_NOTICE_CAN, r.ID_AWARD
            )
            SELECT
                COUNT(*) AS repeated_key_count,
                COUNT(*) FILTER (WHERE distinct_nonnull_values > 1)
                    AS keys_with_multiple_distinct_values
            FROM group_variation
            """
        ).fetchone()

        repeated_key_count, keys_with_variation = result
        rows.append(
            {
                "year": year,
                "column": column,
                "repeated_key_count": repeated_key_count,
                "keys_with_multiple_distinct_values": keys_with_variation,
                "variation_pct": (
                    100.0 * keys_with_variation / repeated_key_count
                    if repeated_key_count
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Supplier identity coverage
# -----------------------------------------------------------------------------

def profile_supplier_identity(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT
            {year} AS year,
            COUNT(*) AS total_rows,
            COUNT(*) FILTER (WHERE {sql_nonmissing('WIN_NAME')}) AS rows_with_name,
            COUNT(*) FILTER (WHERE {sql_nonmissing('WIN_NATIONALID')}) AS rows_with_national_id,
            COUNT(*) FILTER (WHERE {sql_nonmissing('WIN_COUNTRY_CODE')}) AS rows_with_country,
            COUNT(*) FILTER (
                WHERE {sql_nonmissing('WIN_NATIONALID')}
                  AND {sql_nonmissing('WIN_COUNTRY_CODE')}
            ) AS strong_id_rows,
            COUNT(*) FILTER (
                WHERE {sql_nonmissing('WIN_NAME')}
                  AND {sql_nonmissing('WIN_COUNTRY_CODE')}
            ) AS fallback_name_country_rows,
            COUNT(*) FILTER (
                WHERE {sql_nonmissing('WIN_NAME')}
                  AND {sql_nonmissing('WIN_COUNTRY_CODE')}
                  AND {sql_nonmissing('WIN_TOWN')}
            ) AS fallback_name_country_town_rows
        FROM {relation}
        """
    ).fetchdf()


# -----------------------------------------------------------------------------
# Award-status diagnostics
# -----------------------------------------------------------------------------

def profile_award_status_values(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT
            {year} AS year,
            INFO_ON_NON_AWARD,
            COUNT(*) AS n
        FROM {relation}
        GROUP BY INFO_ON_NON_AWARD
        ORDER BY n DESC
        """
    ).fetchdf()


def profile_possible_actual_awards(
    con: duckdb.DuckDBPyConnection,
    year: int,
    relation: str,
) -> pd.DataFrame:
    """
    Diagnostic only. This is not a final semantic definition of an actual award.
    """
    return con.execute(
        f"""
        SELECT
            {year} AS year,
            COUNT(*) FILTER (
                WHERE {sql_nonmissing('ID_AWARD')}
                  AND {sql_nonmissing('WIN_NAME')}
            ) AS award_like_rows,
            COUNT(DISTINCT CONCAT_WS(
                '||',
                COALESCE(ID_NOTICE_CAN, ''),
                COALESCE(ID_AWARD, '')
            )) FILTER (
                WHERE {sql_nonmissing('ID_AWARD')}
                  AND {sql_nonmissing('WIN_NAME')}
            ) AS unique_award_like_pairs,
            COUNT(*) FILTER (
                WHERE {sql_nonmissing('ID_AWARD')}
                  AND {sql_nonmissing('WIN_NAME')}
                  AND (
                      INFO_ON_NON_AWARD IS NULL
                      OR TRIM(CAST(INFO_ON_NON_AWARD AS VARCHAR)) = ''
                  )
            ) AS award_like_rows_without_non_award_flag
        FROM {relation}
        """
    ).fetchdf()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile TED CAN CSV files before harmonization and aggregation."
    )

    parser.add_argument("--can-2015", required=True, type=Path)
    parser.add_argument("--can-2016", required=True, type=Path)
    parser.add_argument("--can-2017", required=True, type=Path)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/can_profile"),
    )

    parser.add_argument(
        "--skip-exact-duplicates",
        action="store_true",
        help="Skip the expensive full-row exact duplicate check.",
    )

    parser.add_argument(
        "--skip-repeated-award-variability",
        action="store_true",
        help="Skip per-column variability checks within repeated award keys.",
    )

    args = parser.parse_args()

    files = {
        2015: args.can_2015,
        2016: args.can_2016,
        2017: args.can_2017,
    }

    for year, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"CAN {year} file does not exist: {path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    temp_dir = output_dir / "duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)
    con.execute(
        f"SET temp_directory='{quote_sql_string(str(temp_dir.resolve()))}'"
    )

    delimiters = {
        year: detect_delimiter(path)
        for year, path in files.items()
    }

    write_csv(
        pd.DataFrame(
            [
                {
                    "year": year,
                    "file": str(files[year]),
                    "detected_delimiter": repr(delimiters[year]),
                }
                for year in files
            ]
        ),
        output_dir / "00_detected_delimiters.csv",
    )

    schema_df, headers = check_schema_compatibility(files, delimiters)
    write_csv(schema_df, output_dir / "01_schema_compatibility.csv")

    reference_year = min(headers)
    reference_columns = headers[reference_year]

    incompatible = schema_df[
        ~(schema_df["same_column_set_as_reference"])
    ]
    if not incompatible.empty:
        raise ValueError(
            "Schemas are not column-set compatible. Review "
            f"{output_dir / '01_schema_compatibility.csv'} before continuing."
        )

    inferred_types = profile_inferred_types(con, files, delimiters)
    write_csv(inferred_types, output_dir / "02_inferred_types_long.csv")

    type_comparison = compare_inferred_types(inferred_types)
    write_csv(type_comparison, output_dir / "03_inferred_type_comparison.csv")

    all_missingness: list[pd.DataFrame] = []
    all_boolean_values: list[pd.DataFrame] = []
    all_date_profiles: list[pd.DataFrame] = []
    all_failed_date_examples: list[pd.DataFrame] = []
    all_temporal_sanity: list[pd.DataFrame] = []
    all_file_year_offsets: list[pd.DataFrame] = []
    all_list_examples: list[pd.DataFrame] = []
    all_delimiter_signals: list[pd.DataFrame] = []
    all_xsd: list[pd.DataFrame] = []
    all_xsd_field_coverage: list[pd.DataFrame] = []
    all_grain: list[pd.DataFrame] = []
    all_notice_grain: list[pd.DataFrame] = []
    all_award_grain: list[pd.DataFrame] = []
    all_structures: list[pd.DataFrame] = []
    all_business_duplicates: list[pd.DataFrame] = []
    all_exact_duplicates: list[pd.DataFrame] = []
    all_award_like_repetition: list[pd.DataFrame] = []
    all_repeated_award_variability: list[pd.DataFrame] = []
    all_supplier_identity: list[pd.DataFrame] = []
    all_group_coverage: list[pd.DataFrame] = []
    all_award_status: list[pd.DataFrame] = []
    all_possible_awards: list[pd.DataFrame] = []

    boolean_columns = identify_boolean_like_columns(reference_columns)

    for year, path in files.items():
        print(f"Profiling CAN {year}: {path}")
        relation = csv_relation(path, delimiters[year])

        all_missingness.append(
            profile_missingness(con, year, relation, reference_columns)
        )

        all_boolean_values.append(
            profile_distinct_values(
                con,
                year,
                relation,
                boolean_columns,
                top_n=30,
            )
        )

        existing_dates = [
            column for column in DATE_COLUMNS if column in reference_columns
        ]

        all_date_profiles.append(
            profile_date_parsing(con, year, relation, existing_dates)
        )

        all_failed_date_examples.append(
            profile_failed_date_examples(con, year, relation, existing_dates)
        )

        all_temporal_sanity.append(
            profile_temporal_sanity(con, year, relation)
        )

        all_file_year_offsets.append(
            profile_file_year_offsets(con, year, relation)
        )

        existing_list_columns = [
            column for column in LIST_LIKE_COLUMNS if column in reference_columns
        ]

        all_list_examples.append(
            profile_list_like_examples(
                con,
                year,
                relation,
                existing_list_columns,
            )
        )

        all_delimiter_signals.append(
            profile_delimiter_signals(
                con,
                year,
                relation,
                existing_list_columns,
            )
        )

        all_xsd.append(profile_xsd_versions(con, year, relation))

        existing_xsd_columns = [
            column for column in XSD_COVERAGE_COLUMNS
            if column in reference_columns
        ]
        all_xsd_field_coverage.append(
            profile_xsd_field_coverage(
                con,
                year,
                relation,
                existing_xsd_columns,
            )
        )

        all_grain.append(profile_grain(con, year, relation))
        all_notice_grain.append(profile_rows_per_notice(con, year, relation))
        all_award_grain.append(profile_rows_per_award(con, year, relation))
        all_structures.append(classify_notice_cardinality(con, year, relation))

        all_business_duplicates.append(
            profile_business_key_duplicates(con, year, relation)
        )

        if not args.skip_exact_duplicates:
            print(
                f"  Running exact duplicate check for {year}; this may take time."
            )
            all_exact_duplicates.append(
                profile_exact_duplicates(con, year, relation)
            )

        all_award_like_repetition.append(
            profile_award_like_repetition(con, year, relation)
        )

        if not args.skip_repeated_award_variability:
            print(
                f"  Profiling repeated award-key variability for {year}; "
                "this may take time."
            )
            existing_repeat_columns = [
                column for column in REPEATED_AWARD_VARIABILITY_COLUMNS
                if column in reference_columns
            ]
            all_repeated_award_variability.append(
                profile_repeated_award_variability(
                    con,
                    year,
                    relation,
                    existing_repeat_columns,
                )
            )

        all_supplier_identity.append(
            profile_supplier_identity(con, year, relation)
        )

        field_groups = {
            "supplier_identity": SUPPLIER_ID_COLUMNS,
            "award_value": AWARD_VALUE_COLUMNS,
            "criteria": CRITERIA_COLUMNS,
            "competition": COMPETITION_COLUMNS,
            "important_ids": IMPORTANT_ID_COLUMNS,
        }

        for group_name, columns in field_groups.items():
            existing = [
                column for column in columns if column in reference_columns
            ]
            all_group_coverage.append(
                profile_field_group_coverage(
                    con,
                    year,
                    relation,
                    group_name,
                    existing,
                )
            )

        all_award_status.append(
            profile_award_status_values(con, year, relation)
        )

        all_possible_awards.append(
            profile_possible_actual_awards(con, year, relation)
        )

    report_map: list[tuple[str, pd.DataFrame]] = [
        ("04_missingness_by_year.csv", concat_or_empty(all_missingness)),
        ("05_boolean_value_encodings.csv", concat_or_empty(all_boolean_values)),
        ("06_date_parse_report.csv", concat_or_empty(all_date_profiles)),
        ("06b_failed_date_examples.csv", concat_or_empty(all_failed_date_examples)),
        ("06c_temporal_sanity.csv", concat_or_empty(all_temporal_sanity)),
        ("06d_file_year_offsets.csv", concat_or_empty(all_file_year_offsets)),
        ("07_list_field_examples.csv", concat_or_empty(all_list_examples)),
        ("08_list_delimiter_signals.csv", concat_or_empty(all_delimiter_signals)),
        ("09_xsd_distribution.csv", concat_or_empty(all_xsd)),
        ("09b_xsd_field_coverage.csv", concat_or_empty(all_xsd_field_coverage)),
        ("10_grain_metrics.csv", concat_or_empty(all_grain)),
        ("11_notice_grain_summary.csv", concat_or_empty(all_notice_grain)),
        ("12_award_grain_summary.csv", concat_or_empty(all_award_grain)),
        ("13_notice_structure_types.csv", concat_or_empty(all_structures)),
        ("14_business_key_duplicates.csv", concat_or_empty(all_business_duplicates)),
        ("16_supplier_identity_coverage.csv", concat_or_empty(all_supplier_identity)),
        ("17_field_group_coverage.csv", concat_or_empty(all_group_coverage)),
        ("18_info_on_non_award_values.csv", concat_or_empty(all_award_status)),
        ("19_possible_actual_award_counts.csv", concat_or_empty(all_possible_awards)),
        ("20_award_like_repetition_summary.csv", concat_or_empty(all_award_like_repetition)),
    ]

    if all_exact_duplicates:
        report_map.append(
            ("15_exact_duplicates.csv", concat_or_empty(all_exact_duplicates))
        )

    if all_repeated_award_variability:
        report_map.append(
            (
                "21_repeated_award_key_variability.csv",
                concat_or_empty(all_repeated_award_variability),
            )
        )

    for filename, dataframe in report_map:
        if not dataframe.empty:
            write_csv(dataframe, output_dir / filename)

    print()
    print("=" * 78)
    print("CAN profiling complete")
    print("=" * 78)
    print(f"Reports written to: {output_dir.resolve()}")
    print()
    print("Review these first:")
    print("  01_schema_compatibility.csv")
    print("  03_inferred_type_comparison.csv")
    print("  04_missingness_by_year.csv")
    print("  05_boolean_value_encodings.csv")
    print("  06_date_parse_report.csv")
    print("  06b_failed_date_examples.csv")
    print("  06c_temporal_sanity.csv")
    print("  06d_file_year_offsets.csv")
    print("  09_xsd_distribution.csv")
    print("  09b_xsd_field_coverage.csv")
    print("  10_grain_metrics.csv")
    print("  12_award_grain_summary.csv")
    print("  14_business_key_duplicates.csv")
    print("  16_supplier_identity_coverage.csv")
    print("  17_field_group_coverage.csv")
    print("  18_info_on_non_award_values.csv")
    print("  20_award_like_repetition_summary.csv")
    if not args.skip_repeated_award_variability:
        print("  21_repeated_award_key_variability.csv")


if __name__ == "__main__":
    main()
