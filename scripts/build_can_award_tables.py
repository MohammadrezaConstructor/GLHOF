#!/usr/bin/env python3
"""
Build normalized TED Contract Award Notice (CAN) analytical tables
from annual CAN CSV extracts.

Designed for CAN 2015-2017 after profiling showed:
- identical 75-column schemas,
- mixed date formats across years,
- repeated (ID_NOTICE_CAN, ID_AWARD) keys,
- core award facts largely invariant within repeated keys,
- lot/criteria structure that must be preserved separately,
- XSD-related missingness differences,
- limited historical WIN_NATIONALID coverage.

Outputs:
- can_combined_clean.parquet
- award_fact_2015_2017.parquet
- award_criteria_2015_2017.parquet
- award_lot_context_2015_2017.parquet
- award_provenance_2015_2017.parquet
- award_quarantine_2015_2017.parquet
- row_exclusion_audit_2015_2017.parquet
- repeated_key_exact_duplicate_summary.csv
- award_table_build_metrics.csv
- award_table_build_report.md

Important methodological choices:
1. Raw annual CAN files are appended vertically with provenance.
2. Dates are parsed from both DD-MON-YY and DD/MM/YY formats.
3. Publication lag is defined correctly as:
       DT_DISPATCH - DT_AWARD
4. Award candidates require:
       ID_AWARD present,
       WIN_NAME present,
       INFO_ON_NON_AWARD blank,
       explicit non-cancelled status,
       valid dispatch and award dates.
5. Core award conflicts are quarantined, never silently collapsed.
6. Criteria and notice-lot context are preserved in child tables.
7. Supplier identifiers are provisional and conservative:
       national_id + country
       else normalized_name + country
       else normalized_name + town
       else unresolved.
8. SME and group-award fields are preserved as raw values; no unsupported
   single-Boolean interpretation is imposed on multi-value SME strings.
9. YEAR is preserved as source metadata and is not required to equal
   the year embedded in event dates.
10. CORRECTIONS is preserved and reported, but not automatically filtered,
    because correction semantics require a separate notice-version policy.

Requires:
    pip install duckdb pandas pyarrow
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPECTED_CAN_COLUMNS = [
    "ID_NOTICE_CAN", "TED_NOTICE_URL", "YEAR", "ID_TYPE", "DT_DISPATCH",
    "XSD_VERSION", "CANCELLED", "CORRECTIONS", "B_MULTIPLE_CAE",
    "CAE_NAME", "CAE_NATIONALID", "CAE_ADDRESS", "CAE_TOWN",
    "CAE_POSTAL_CODE", "CAE_GPA_ANNEX", "ISO_COUNTRY_CODE",
    "ISO_COUNTRY_CODE_GPA", "B_MULTIPLE_COUNTRY", "ISO_COUNTRY_CODE_ALL",
    "CAE_TYPE", "EU_INST_CODE", "MAIN_ACTIVITY", "B_ON_BEHALF",
    "B_INVOLVES_JOINT_PROCUREMENT", "B_AWARDED_BY_CENTRAL_BODY",
    "TYPE_OF_CONTRACT", "TAL_LOCATION_NUTS", "B_FRA_AGREEMENT",
    "FRA_ESTIMATED", "B_FRA_CONTRACT", "B_DYN_PURCH_SYST", "CPV",
    "MAIN_CPV_CODE_GPA", "ID_LOT", "ADDITIONAL_CPVS", "B_GPA",
    "GPA_COVERAGE", "LOTS_NUMBER", "VALUE_EURO", "VALUE_EURO_FIN_1",
    "VALUE_EURO_FIN_2", "B_EU_FUNDS", "TOP_TYPE", "B_ACCELERATED",
    "OUT_OF_DIRECTIVES", "CRIT_CODE", "CRIT_PRICE_WEIGHT",
    "CRIT_CRITERIA", "CRIT_WEIGHTS", "B_ELECTRONIC_AUCTION",
    "NUMBER_AWARDS", "ID_AWARD", "ID_LOT_AWARDED", "INFO_ON_NON_AWARD",
    "INFO_UNPUBLISHED", "B_AWARDED_TO_A_GROUP", "WIN_NAME",
    "WIN_NATIONALID", "WIN_ADDRESS", "WIN_TOWN", "WIN_POSTAL_CODE",
    "WIN_COUNTRY_CODE", "B_CONTRACTOR_SME", "CONTRACT_NUMBER", "TITLE",
    "NUMBER_OFFERS", "NUMBER_TENDERS_SME", "NUMBER_TENDERS_OTHER_EU",
    "NUMBER_TENDERS_NON_EU", "NUMBER_OFFERS_ELECTR",
    "AWARD_EST_VALUE_EURO", "AWARD_VALUE_EURO",
    "AWARD_VALUE_EURO_FIN_1", "B_SUBCONTRACTED", "DT_AWARD",
]

# Fields treated as core award facts. If more than one distinct non-null
# value occurs within an award key, the award key is quarantined.
CORE_CONFLICT_FIELDS = [
    "WIN_NAME_CLEAN",
    "WIN_COUNTRY_CODE_CLEAN",
    "ID_LOT_AWARDED_CLEAN",
    "CONTRACT_NUMBER_CLEAN",
    "CPV_CLEAN",
    "NUMBER_OFFERS_CLEAN",
    "AWARD_EST_VALUE_EURO_CLEAN",
    "AWARD_VALUE_EURO_CLEAN",
    "AWARD_VALUE_EURO_FIN_1_CLEAN",
    "DT_AWARD_CLEAN",
    "DT_DISPATCH_CLEAN",
]

NUMERIC_COLUMNS = [
    "YEAR",
    "LOTS_NUMBER",
    "VALUE_EURO",
    "VALUE_EURO_FIN_1",
    "VALUE_EURO_FIN_2",
    "CRIT_PRICE_WEIGHT",
    "NUMBER_AWARDS",
    "NUMBER_OFFERS",
    "NUMBER_TENDERS_SME",
    "NUMBER_TENDERS_OTHER_EU",
    "NUMBER_TENDERS_NON_EU",
    "NUMBER_OFFERS_ELECTR",
    "AWARD_EST_VALUE_EURO",
    "AWARD_VALUE_EURO",
    "AWARD_VALUE_EURO_FIN_1",
]

BOOLEAN_LIKE_COLUMNS = [
    "CANCELLED",
    "CORRECTIONS",
    "B_MULTIPLE_CAE",
    "B_MULTIPLE_COUNTRY",
    "B_ON_BEHALF",
    "B_INVOLVES_JOINT_PROCUREMENT",
    "B_AWARDED_BY_CENTRAL_BODY",
    "B_FRA_AGREEMENT",
    "B_FRA_CONTRACT",
    "B_DYN_PURCH_SYST",
    "B_GPA",
    "B_EU_FUNDS",
    "B_ACCELERATED",
    "OUT_OF_DIRECTIVES",
    "B_ELECTRONIC_AUCTION",
    "INFO_UNPUBLISHED",
    "B_AWARDED_TO_A_GROUP",
    "B_SUBCONTRACTED",
]

# Fields placed in the core award fact output.
AWARD_FACT_FIELDS = [
    "ID_NOTICE_CAN_CLEAN",
    "ID_AWARD_CLEAN",
    "ID_LOT_AWARDED_CLEAN",
    "DT_DISPATCH_CLEAN",
    "DT_AWARD_CLEAN",
    "PUBLICATION_LAG_DAYS",
    "XSD_VERSION_CLEAN",
    "CORRECTIONS_RAW",
    "CORRECTIONS_CLEAN",
    "CAE_NAME_CLEAN",
    "CAE_NATIONALID_CLEAN",
    "ISO_COUNTRY_CODE_CLEAN",
    "CAE_TYPE_CLEAN",
    "MAIN_ACTIVITY_CLEAN",
    "TYPE_OF_CONTRACT_CLEAN",
    "TAL_LOCATION_NUTS_CLEAN",
    "CPV_CLEAN",
    "ADDITIONAL_CPVS_CLEAN",
    "TOP_TYPE_CLEAN",
    "B_FRA_AGREEMENT_RAW",
    "B_FRA_AGREEMENT_CLEAN",
    "B_DYN_PURCH_SYST_RAW",
    "B_DYN_PURCH_SYST_CLEAN",
    "B_ELECTRONIC_AUCTION_RAW",
    "B_ELECTRONIC_AUCTION_CLEAN",
    "WIN_NAME_CLEAN",
    "WIN_NATIONALID_CLEAN",
    "WIN_TOWN_CLEAN",
    "WIN_COUNTRY_CODE_CLEAN",
    "B_CONTRACTOR_SME_RAW",
    "B_AWARDED_TO_A_GROUP_RAW",
    "B_AWARDED_TO_A_GROUP_CLEAN",
    "CONTRACT_NUMBER_CLEAN",
    "TITLE_CLEAN",
    "NUMBER_OFFERS_CLEAN",
    "NUMBER_TENDERS_SME_CLEAN",
    "NUMBER_TENDERS_OTHER_EU_CLEAN",
    "NUMBER_TENDERS_NON_EU_CLEAN",
    "NUMBER_OFFERS_ELECTR_CLEAN",
    "AWARD_EST_VALUE_EURO_CLEAN",
    "AWARD_VALUE_EURO_CLEAN",
    "AWARD_VALUE_EURO_FIN_1_CLEAN",
    "B_SUBCONTRACTED_RAW",
    "B_SUBCONTRACTED_CLEAN",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_sql_string(value: str) -> str:
    return value.replace("'", "''")


def sql_path(path: Path) -> str:
    # DuckDB handles forward slashes reliably on Windows.
    return quote_sql_string(path.resolve().as_posix())


def clean_text_sql(column: str) -> str:
    c = quote_ident(column)
    return f"NULLIF(TRIM(CAST({c} AS VARCHAR)), '')"


def normalize_entity_text_sql(column: str) -> str:
    c = clean_text_sql(column)
    return (
        "NULLIF(TRIM(REGEXP_REPLACE("
        f"LOWER(COALESCE({c}, '')), "
        "'[^[:alnum:]]+', ' ', 'g')), '')"
    )


def normalize_identifier_sql(column: str) -> str:
    c = clean_text_sql(column)
    return (
        "NULLIF(REGEXP_REPLACE("
        f"UPPER(COALESCE({c}, '')), "
        "'[^[:alnum:]]+', '', 'g'), '')"
    )


def boolean_clean_sql(column: str) -> str:
    c = f"UPPER({clean_text_sql(column)})"
    return f"""
    CASE
        WHEN {c} IN ('1', 'Y', 'YES', 'TRUE', 'T') THEN TRUE
        WHEN {c} IN ('0', 'N', 'NO', 'FALSE', 'F') THEN FALSE
        ELSE NULL
    END
    """


def numeric_clean_sql(column: str) -> str:
    """
    Conservative numeric parser.
    It does not silently reinterpret thousands/decimal separators.
    """
    c = clean_text_sql(column)
    return f"TRY_CAST({c} AS DOUBLE)"


def date_clean_sql(column: str) -> str:
    """
    TED date formats observed in the supplied CAN files:
      2015-2016: 23-DEC-14
      2017:      22/12/16

    Additional fallback formats are included.
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
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return next(csv.reader(f, delimiter=delimiter))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def csv_relation(path: Path, delimiter: str) -> str:
    delim = quote_sql_string(delimiter)
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
        COPY {quote_ident(table_name)}
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def write_df_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def scalar(con: duckdb.DuckDBPyConnection, query: str):
    return con.execute(query).fetchone()[0]


# ---------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------

def validate_headers(files: dict[int, Path], delimiters: dict[int, str]) -> None:
    errors: list[str] = []

    for year, path in files.items():
        actual = read_header(path, delimiters[year])

        if actual != EXPECTED_CAN_COLUMNS:
            missing = sorted(set(EXPECTED_CAN_COLUMNS) - set(actual))
            extra = sorted(set(actual) - set(EXPECTED_CAN_COLUMNS))
            order_ok = set(actual) == set(EXPECTED_CAN_COLUMNS)

            errors.append(
                f"CAN {year} schema mismatch.\n"
                f"  Column count: {len(actual)} (expected {len(EXPECTED_CAN_COLUMNS)})\n"
                f"  Same set but different order: {order_ok and actual != EXPECTED_CAN_COLUMNS}\n"
                f"  Missing: {missing}\n"
                f"  Extra: {extra}"
            )

    if errors:
        raise ValueError("\n\n".join(errors))


# ---------------------------------------------------------------------
# Build raw and cleaned tables
# ---------------------------------------------------------------------

def build_raw_table(
    con: duckdb.DuckDBPyConnection,
    files: dict[int, Path],
    delimiters: dict[int, str],
) -> None:
    selects = []

    for year, path in files.items():
        relation = csv_relation(path, delimiters[year])
        source_file = quote_sql_string(path.name)

        selects.append(
            f"""
            SELECT
                *,
                {year}::INTEGER AS SOURCE_YEAR,
                '{source_file}'::VARCHAR AS SOURCE_FILE,
                ROW_NUMBER() OVER ()::BIGINT AS SOURCE_ROW_ID
            FROM {relation}
            """
        )

    union_sql = "\nUNION ALL BY NAME\n".join(selects)

    con.execute("DROP TABLE IF EXISTS can_raw")
    con.execute(f"CREATE TABLE can_raw AS {union_sql}")


def build_clean_table(con: duckdb.DuckDBPyConnection) -> None:
    raw_cols = EXPECTED_CAN_COLUMNS

    select_exprs: list[str] = [
        "SOURCE_YEAR",
        "SOURCE_FILE",
        "SOURCE_ROW_ID",
    ]

    # Preserve all original raw columns.
    select_exprs.extend(quote_ident(c) for c in raw_cols)

    # Common cleaned text identifiers/context.
    text_clean_columns = [
        "ID_NOTICE_CAN",
        "TED_NOTICE_URL",
        "ID_TYPE",
        "XSD_VERSION",
        "CAE_NAME",
        "CAE_NATIONALID",
        "CAE_TOWN",
        "ISO_COUNTRY_CODE",
        "CAE_TYPE",
        "MAIN_ACTIVITY",
        "TYPE_OF_CONTRACT",
        "TAL_LOCATION_NUTS",
        "CPV",
        "ID_LOT",
        "ADDITIONAL_CPVS",
        "TOP_TYPE",
        "CRIT_CODE",
        "CRIT_CRITERIA",
        "CRIT_WEIGHTS",
        "ID_AWARD",
        "ID_LOT_AWARDED",
        "INFO_ON_NON_AWARD",
        "WIN_NAME",
        "WIN_NATIONALID",
        "WIN_TOWN",
        "WIN_COUNTRY_CODE",
        "CONTRACT_NUMBER",
        "TITLE",
    ]

    for col in text_clean_columns:
        select_exprs.append(
            f"{clean_text_sql(col)} AS {quote_ident(col + '_CLEAN')}"
        )

    # Dates.
    select_exprs.append(
        f"{date_clean_sql('DT_DISPATCH')} AS DT_DISPATCH_CLEAN"
    )
    select_exprs.append(
        f"{date_clean_sql('DT_AWARD')} AS DT_AWARD_CLEAN"
    )

    # Correct publication lag: notice dispatch minus award date.
    select_exprs.append(
        """
        DATE_DIFF(
            'day',
            CAST(COALESCE(
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_AWARD AS VARCHAR)), ''), '%d-%b-%y'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_AWARD AS VARCHAR)), ''), '%d/%m/%y'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_AWARD AS VARCHAR)), ''), '%Y-%m-%d'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_AWARD AS VARCHAR)), ''), '%d/%m/%Y'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_AWARD AS VARCHAR)), ''), '%Y%m%d'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_AWARD AS VARCHAR)), ''), '%d-%m-%Y')
            ) AS DATE),
            CAST(COALESCE(
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_DISPATCH AS VARCHAR)), ''), '%d-%b-%y'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_DISPATCH AS VARCHAR)), ''), '%d/%m/%y'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_DISPATCH AS VARCHAR)), ''), '%Y-%m-%d'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_DISPATCH AS VARCHAR)), ''), '%d/%m/%Y'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_DISPATCH AS VARCHAR)), ''), '%Y%m%d'),
                TRY_STRPTIME(NULLIF(TRIM(CAST(DT_DISPATCH AS VARCHAR)), ''), '%d-%m-%Y')
            ) AS DATE)
        ) AS PUBLICATION_LAG_DAYS
        """
    )

    # Numerics.
    for col in NUMERIC_COLUMNS:
        select_exprs.append(
            f"{numeric_clean_sql(col)} AS {quote_ident(col + '_CLEAN')}"
        )

    # Raw + clean boolean-like fields. Raw is retained explicitly because
    # some TED fields are multi-value or schema-dependent.
    for col in BOOLEAN_LIKE_COLUMNS:
        select_exprs.append(
            f"{clean_text_sql(col)} AS {quote_ident(col + '_RAW')}"
        )
        select_exprs.append(
            f"{boolean_clean_sql(col)} AS {quote_ident(col + '_CLEAN')}"
        )

    # SME status is intentionally raw-only for now.
    select_exprs.append(
        f"{clean_text_sql('B_CONTRACTOR_SME')} AS B_CONTRACTOR_SME_RAW"
    )

    # Conservative normalized supplier fields.
    select_exprs.append(
        f"{normalize_entity_text_sql('WIN_NAME')} AS WIN_NAME_NORMALIZED"
    )
    select_exprs.append(
        f"{normalize_identifier_sql('WIN_NATIONALID')} AS WIN_NATIONALID_NORMALIZED"
    )
    select_exprs.append(
        f"UPPER({clean_text_sql('WIN_COUNTRY_CODE')}) AS WIN_COUNTRY_CODE_NORMALIZED"
    )
    select_exprs.append(
        f"{normalize_entity_text_sql('WIN_TOWN')} AS WIN_TOWN_NORMALIZED"
    )

    select_sql = ",\n        ".join(select_exprs)

    con.execute("DROP TABLE IF EXISTS can_clean")
    con.execute(
        f"""
        CREATE TABLE can_clean AS
        SELECT
            {select_sql}
        FROM can_raw
        """
    )

    # Award key and conservative provisional supplier identity.
    con.execute("DROP TABLE IF EXISTS can_clean_keyed")
    con.execute(
        """
        CREATE TABLE can_clean_keyed AS
        SELECT
            *,
            CASE
                WHEN ID_NOTICE_CAN_CLEAN IS NOT NULL
                 AND ID_AWARD_CLEAN IS NOT NULL
                THEN ID_NOTICE_CAN_CLEAN || '||' || ID_AWARD_CLEAN
                ELSE NULL
            END AS AWARD_KEY,

            CASE
                WHEN WIN_NATIONALID_NORMALIZED IS NOT NULL
                 AND WIN_COUNTRY_CODE_NORMALIZED IS NOT NULL
                THEN
                    'nid_country||' ||
                    WIN_COUNTRY_CODE_NORMALIZED || '||' ||
                    WIN_NATIONALID_NORMALIZED

                WHEN WIN_NAME_NORMALIZED IS NOT NULL
                 AND WIN_COUNTRY_CODE_NORMALIZED IS NOT NULL
                THEN
                    'name_country||' ||
                    WIN_COUNTRY_CODE_NORMALIZED || '||' ||
                    WIN_NAME_NORMALIZED

                WHEN WIN_NAME_NORMALIZED IS NOT NULL
                 AND WIN_TOWN_NORMALIZED IS NOT NULL
                THEN
                    'name_town||' ||
                    WIN_TOWN_NORMALIZED || '||' ||
                    WIN_NAME_NORMALIZED

                ELSE NULL
            END AS SUPPLIER_KEY_PROVISIONAL,

            CASE
                WHEN WIN_NATIONALID_NORMALIZED IS NOT NULL
                 AND WIN_COUNTRY_CODE_NORMALIZED IS NOT NULL
                THEN 'national_id_country'

                WHEN WIN_NAME_NORMALIZED IS NOT NULL
                 AND WIN_COUNTRY_CODE_NORMALIZED IS NOT NULL
                THEN 'normalized_name_country'

                WHEN WIN_NAME_NORMALIZED IS NOT NULL
                 AND WIN_TOWN_NORMALIZED IS NOT NULL
                THEN 'normalized_name_town'

                ELSE 'unresolved'
            END AS SUPPLIER_RESOLUTION_METHOD

        FROM can_clean
        """
    )


# ---------------------------------------------------------------------
# Candidate cohort and exclusion audit
# ---------------------------------------------------------------------

def build_candidate_and_exclusion_tables(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Apply conservative award-candidate rules.

    CORRECTIONS is NOT used as an exclusion criterion here. Correction/version
    handling is preserved for later notice-version analysis.
    """

    con.execute("DROP TABLE IF EXISTS row_exclusion_audit")
    con.execute(
        """
        CREATE TABLE row_exclusion_audit AS
        SELECT
            SOURCE_YEAR,
            SOURCE_FILE,
            SOURCE_ROW_ID,
            ID_NOTICE_CAN_CLEAN,
            ID_AWARD_CLEAN,
            WIN_NAME_CLEAN,
            CASE
                WHEN ID_AWARD_CLEAN IS NULL
                    THEN 'missing_award_id'
                WHEN WIN_NAME_CLEAN IS NULL
                    THEN 'missing_winner_name'
                WHEN INFO_ON_NON_AWARD_CLEAN IS NOT NULL
                    THEN 'non_award_status'
                WHEN CANCELLED_CLEAN IS TRUE
                    THEN 'cancelled'
                WHEN CANCELLED_CLEAN IS NULL
                    THEN 'unknown_cancelled_status'
                WHEN DT_DISPATCH_CLEAN IS NULL
                    THEN 'invalid_dispatch_date'
                WHEN DT_AWARD_CLEAN IS NULL
                    THEN 'invalid_award_date'
                ELSE NULL
            END AS EXCLUSION_REASON
        FROM can_clean_keyed
        WHERE
            ID_AWARD_CLEAN IS NULL
            OR WIN_NAME_CLEAN IS NULL
            OR INFO_ON_NON_AWARD_CLEAN IS NOT NULL
            OR CANCELLED_CLEAN IS TRUE
            OR CANCELLED_CLEAN IS NULL
            OR DT_DISPATCH_CLEAN IS NULL
            OR DT_AWARD_CLEAN IS NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS award_candidates")
    con.execute(
        """
        CREATE TABLE award_candidates AS
        SELECT *
        FROM can_clean_keyed
        WHERE
            ID_AWARD_CLEAN IS NOT NULL
            AND WIN_NAME_CLEAN IS NOT NULL
            AND INFO_ON_NON_AWARD_CLEAN IS NULL
            AND CANCELLED_CLEAN IS FALSE
            AND DT_DISPATCH_CLEAN IS NOT NULL
            AND DT_AWARD_CLEAN IS NOT NULL
        """
    )


# ---------------------------------------------------------------------
# Conflict detection and quarantine
# ---------------------------------------------------------------------

def build_conflict_tables(con: duckdb.DuckDBPyConnection) -> None:
    conflict_exprs = []

    for field in CORE_CONFLICT_FIELDS:
        q = quote_ident(field)
        alias = f"N_DISTINCT__{field}"
        conflict_exprs.append(
            f"COUNT(DISTINCT {q}) FILTER (WHERE {q} IS NOT NULL) AS {quote_ident(alias)}"
        )

    conflict_sql = ",\n            ".join(conflict_exprs)

    con.execute("DROP TABLE IF EXISTS award_key_conflict_profile")
    con.execute(
        f"""
        CREATE TABLE award_key_conflict_profile AS
        SELECT
            AWARD_KEY,
            COUNT(*) AS RAW_ROW_COUNT,
            {conflict_sql}
        FROM award_candidates
        GROUP BY AWARD_KEY
        """
    )

    conflict_conditions = [
        f"{quote_ident('N_DISTINCT__' + field)} > 1"
        for field in CORE_CONFLICT_FIELDS
    ]
    any_conflict = " OR ".join(conflict_conditions)

    con.execute("DROP TABLE IF EXISTS award_conflict_keys")
    con.execute(
        f"""
        CREATE TABLE award_conflict_keys AS
        SELECT *
        FROM award_key_conflict_profile
        WHERE {any_conflict}
        """
    )

    con.execute("DROP TABLE IF EXISTS award_quarantine")
    con.execute(
        """
        CREATE TABLE award_quarantine AS
        SELECT
            c.*,
            p.*
            EXCLUDE (AWARD_KEY)
        FROM award_candidates c
        JOIN award_conflict_keys p
          USING (AWARD_KEY)
        """
    )

    con.execute("DROP TABLE IF EXISTS valid_award_candidates")
    con.execute(
        """
        CREATE TABLE valid_award_candidates AS
        SELECT c.*
        FROM award_candidates c
        LEFT JOIN award_conflict_keys q
          USING (AWARD_KEY)
        WHERE q.AWARD_KEY IS NULL
        """
    )


# ---------------------------------------------------------------------
# Build normalized analytical tables
# ---------------------------------------------------------------------

def build_award_fact(con: duckdb.DuckDBPyConnection) -> None:
    """
    One row per valid AWARD_KEY. Child-table fields such as criteria and
    notice-lot context are intentionally not collapsed here.
    """
    aggregate_exprs = []

    for field in AWARD_FACT_FIELDS:
        q = quote_ident(field)
        aggregate_exprs.append(f"MAX({q}) AS {q}")

    aggregate_exprs.extend([
        "MAX(SUPPLIER_KEY_PROVISIONAL) AS SUPPLIER_KEY_PROVISIONAL",
        "MAX(SUPPLIER_RESOLUTION_METHOD) AS SUPPLIER_RESOLUTION_METHOD",
        "MIN(SOURCE_YEAR) AS SOURCE_YEAR_MIN",
        "MAX(SOURCE_YEAR) AS SOURCE_YEAR_MAX",
        "COUNT(*) AS SOURCE_ROW_COUNT",
    ])

    agg_sql = ",\n            ".join(aggregate_exprs)

    con.execute("DROP TABLE IF EXISTS award_fact")
    con.execute(
        f"""
        CREATE TABLE award_fact AS
        SELECT
            AWARD_KEY,
            {agg_sql}
        FROM valid_award_candidates
        GROUP BY AWARD_KEY
        """
    )


def build_award_criteria(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS award_criteria")
    con.execute(
        """
        CREATE TABLE award_criteria AS
        SELECT DISTINCT
            AWARD_KEY,
            XSD_VERSION_CLEAN,
            CRIT_CODE_CLEAN,
            CRIT_PRICE_WEIGHT_CLEAN,
            CRIT_CRITERIA_CLEAN,
            CRIT_WEIGHTS_CLEAN
        FROM valid_award_candidates
        WHERE
            CRIT_CODE_CLEAN IS NOT NULL
            OR CRIT_PRICE_WEIGHT_CLEAN IS NOT NULL
            OR CRIT_CRITERIA_CLEAN IS NOT NULL
            OR CRIT_WEIGHTS_CLEAN IS NOT NULL
        """
    )


def build_award_lot_context(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS award_lot_context")
    con.execute(
        """
        CREATE TABLE award_lot_context AS
        SELECT DISTINCT
            AWARD_KEY,
            ID_LOT_CLEAN,
            ID_LOT_AWARDED_CLEAN
        FROM valid_award_candidates
        WHERE
            ID_LOT_CLEAN IS NOT NULL
            OR ID_LOT_AWARDED_CLEAN IS NOT NULL
        """
    )


def build_award_provenance(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS award_provenance")
    con.execute(
        """
        CREATE TABLE award_provenance AS
        SELECT
            AWARD_KEY,
            SOURCE_YEAR,
            SOURCE_FILE,
            SOURCE_ROW_ID
        FROM valid_award_candidates
        """
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def repeated_key_exact_duplicate_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Exact-duplicate check restricted to candidate rows belonging to repeated
    award keys. Provenance columns are excluded from duplicate identity.
    """
    repeated_rows = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM award_candidates
        WHERE AWARD_KEY IN (
            SELECT AWARD_KEY
            FROM award_candidates
            GROUP BY AWARD_KEY
            HAVING COUNT(*) > 1
        )
        """,
    )

    if repeated_rows == 0:
        return pd.DataFrame([{
            "repeated_candidate_rows": 0,
            "distinct_repeated_rows_excluding_provenance": 0,
            "exact_duplicate_excess_rows": 0,
        }])

    # Use the original 75 TED columns plus AWARD_KEY, excluding provenance.
    raw_select = ", ".join(quote_ident(c) for c in EXPECTED_CAN_COLUMNS)

    distinct_rows = scalar(
        con,
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                AWARD_KEY,
                {raw_select}
            FROM award_candidates
            WHERE AWARD_KEY IN (
                SELECT AWARD_KEY
                FROM award_candidates
                GROUP BY AWARD_KEY
                HAVING COUNT(*) > 1
            )
        )
        """,
    )

    return pd.DataFrame([{
        "repeated_candidate_rows": repeated_rows,
        "distinct_repeated_rows_excluding_provenance": distinct_rows,
        "exact_duplicate_excess_rows": repeated_rows - distinct_rows,
    }])


def build_metrics(
    con: duckdb.DuckDBPyConnection,
    files: dict[int, Path],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(metric: str, value, note: str = ""):
        rows.append({"metric": metric, "value": value, "note": note})

    add("raw_rows", scalar(con, "SELECT COUNT(*) FROM can_raw"))
    add("clean_rows", scalar(con, "SELECT COUNT(*) FROM can_clean_keyed"))
    add("excluded_rows", scalar(con, "SELECT COUNT(*) FROM row_exclusion_audit"))
    add("award_candidate_rows", scalar(con, "SELECT COUNT(*) FROM award_candidates"))
    add("award_candidate_unique_keys", scalar(con, "SELECT COUNT(DISTINCT AWARD_KEY) FROM award_candidates"))
    add(
        "repeated_award_keys",
        scalar(
            con,
            """
            SELECT COUNT(*)
            FROM (
                SELECT AWARD_KEY
                FROM award_candidates
                GROUP BY AWARD_KEY
                HAVING COUNT(*) > 1
            )
            """,
        ),
    )
    add("conflict_award_keys", scalar(con, "SELECT COUNT(*) FROM award_conflict_keys"))
    add("quarantine_rows", scalar(con, "SELECT COUNT(*) FROM award_quarantine"))
    add("valid_award_keys", scalar(con, "SELECT COUNT(*) FROM award_fact"))
    add("criteria_rows", scalar(con, "SELECT COUNT(*) FROM award_criteria"))
    add("lot_context_rows", scalar(con, "SELECT COUNT(*) FROM award_lot_context"))
    add("provenance_rows", scalar(con, "SELECT COUNT(*) FROM award_provenance"))

    # Correct publication-lag diagnostics.
    lag = con.execute(
        """
        SELECT
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE PUBLICATION_LAG_DAYS < 0) AS negative_n,
            COUNT(*) FILTER (WHERE PUBLICATION_LAG_DAYS = 0) AS zero_n,
            COUNT(*) FILTER (WHERE PUBLICATION_LAG_DAYS > 0) AS positive_n,
            MIN(PUBLICATION_LAG_DAYS) AS min_days,
            MEDIAN(PUBLICATION_LAG_DAYS) AS median_days,
            AVG(PUBLICATION_LAG_DAYS) AS mean_days,
            QUANTILE_CONT(PUBLICATION_LAG_DAYS, 0.95) AS p95_days,
            QUANTILE_CONT(PUBLICATION_LAG_DAYS, 0.99) AS p99_days,
            MAX(PUBLICATION_LAG_DAYS) AS max_days
        FROM award_fact
        WHERE PUBLICATION_LAG_DAYS IS NOT NULL
        """
    ).fetchdf().iloc[0]

    for key, value in lag.items():
        add(f"publication_lag_{key}", value)

    # Provisional supplier identity coverage.
    supplier = con.execute(
        """
        SELECT
            COUNT(*) AS total_awards,
            COUNT(*) FILTER (
                WHERE SUPPLIER_RESOLUTION_METHOD = 'national_id_country'
            ) AS national_id_country,
            COUNT(*) FILTER (
                WHERE SUPPLIER_RESOLUTION_METHOD = 'normalized_name_country'
            ) AS normalized_name_country,
            COUNT(*) FILTER (
                WHERE SUPPLIER_RESOLUTION_METHOD = 'normalized_name_town'
            ) AS normalized_name_town,
            COUNT(*) FILTER (
                WHERE SUPPLIER_RESOLUTION_METHOD = 'unresolved'
            ) AS unresolved
        FROM award_fact
        """
    ).fetchdf().iloc[0]

    for key, value in supplier.items():
        add(f"supplier_resolution_{key}", value)

    # Exclusion reason counts.
    exclusions = con.execute(
        """
        SELECT EXCLUSION_REASON, COUNT(*) AS n
        FROM row_exclusion_audit
        GROUP BY EXCLUSION_REASON
        ORDER BY n DESC
        """
    ).fetchdf()

    for _, row in exclusions.iterrows():
        add(
            f"excluded__{row['EXCLUSION_REASON']}",
            int(row["n"]),
        )

    # File checksums.
    for year, path in files.items():
        add(f"sha256_can_{year}", file_sha256(path), str(path.resolve()))

    return pd.DataFrame(rows)


def xsd_coverage_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    fields = [
        "WIN_NATIONALID_CLEAN",
        "B_CONTRACTOR_SME_RAW",
        "ID_LOT_CLEAN",
        "CRIT_CODE_CLEAN",
        "CRIT_PRICE_WEIGHT_CLEAN",
        "CRIT_CRITERIA_CLEAN",
        "CRIT_WEIGHTS_CLEAN",
        "NUMBER_OFFERS_CLEAN",
        "AWARD_VALUE_EURO_CLEAN",
        "AWARD_VALUE_EURO_FIN_1_CLEAN",
    ]

    pieces = []

    for field in fields:
        q = quote_ident(field)
        field_label = quote_sql_string(field)
        pieces.append(
            f"""
            SELECT
                SOURCE_YEAR,
                XSD_VERSION_CLEAN,
                '{field_label}' AS field,
                COUNT(*) AS n_rows,
                COUNT(*) FILTER (WHERE {q} IS NOT NULL) AS nonmissing,
                100.0 * COUNT(*) FILTER (WHERE {q} IS NOT NULL)
                    / NULLIF(COUNT(*), 0) AS coverage_pct
            FROM award_candidates
            GROUP BY SOURCE_YEAR, XSD_VERSION_CLEAN
            """
        )

    return con.execute(
        "\nUNION ALL\n".join(pieces)
        + "\nORDER BY SOURCE_YEAR, XSD_VERSION_CLEAN, field"
    ).fetchdf()


def conflict_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pieces = []

    for field in CORE_CONFLICT_FIELDS:
        col = quote_ident("N_DISTINCT__" + field)
        field_label = quote_sql_string(field)
        pieces.append(
            f"""
            SELECT
                '{field_label}' AS field,
                COUNT(*) FILTER (WHERE {col} > 1) AS conflicting_award_keys
            FROM award_key_conflict_profile
            """
        )

    return con.execute("\nUNION ALL\n".join(pieces)).fetchdf()


def year_date_offset_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        WITH base AS (
            SELECT
                SOURCE_YEAR,
                YEAR(DT_DISPATCH_CLEAN) AS dispatch_year,
                YEAR(DT_AWARD_CLEAN) AS award_year
            FROM award_candidates
        )
        SELECT
            SOURCE_YEAR,
            'dispatch' AS date_type,
            SOURCE_YEAR - dispatch_year AS year_offset,
            COUNT(*) AS n
        FROM base
        WHERE dispatch_year IS NOT NULL
        GROUP BY SOURCE_YEAR, year_offset

        UNION ALL

        SELECT
            SOURCE_YEAR,
            'award' AS date_type,
            SOURCE_YEAR - award_year AS year_offset,
            COUNT(*) AS n
        FROM base
        WHERE award_year IS NOT NULL
        GROUP BY SOURCE_YEAR, year_offset

        ORDER BY SOURCE_YEAR, date_type, year_offset
        """
    ).fetchdf()


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def create_markdown_report(
    metrics: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    conflict_df: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_map = dict(zip(metrics["metric"], metrics["value"]))

    def mv(name: str, default="N/A"):
        return metric_map.get(name, default)

    lines = [
        "# TED CAN Award Table Build Report",
        "",
        "## Build summary",
        "",
        f"- Raw rows: {mv('raw_rows')}",
        f"- Award-candidate rows: {mv('award_candidate_rows')}",
        f"- Unique award-candidate keys: {mv('award_candidate_unique_keys')}",
        f"- Repeated award keys: {mv('repeated_award_keys')}",
        f"- Conflict award keys quarantined: {mv('conflict_award_keys')}",
        f"- Valid award facts: {mv('valid_award_keys')}",
        f"- Criteria child rows: {mv('criteria_rows')}",
        f"- Lot-context child rows: {mv('lot_context_rows')}",
        "",
        "## Methodological rules applied",
        "",
        "1. Annual CAN files were appended vertically after exact header validation.",
        "2. Source year, source file, and source row ID were preserved.",
        "3. Mixed TED date formats were parsed explicitly.",
        "4. Publication lag was defined as `DT_DISPATCH - DT_AWARD`.",
        "5. `YEAR` was retained as source metadata; event dates were not forced to match it.",
        "6. Award candidates required a non-null award ID, winner name, blank non-award status, explicit non-cancelled status, and valid dates.",
        "7. Correction indicators were preserved but not automatically filtered.",
        "8. Core award-key conflicts were quarantined rather than silently collapsed.",
        "9. Criteria and lot context were preserved in separate child tables.",
        "10. SME status was preserved as raw data because multi-value strings occur.",
        "11. Supplier identity keys are provisional and use conservative hierarchical rules.",
        "12. Group-award status is retained; this script does not allocate group-award value to a supplier history.",
        "",
        "## Publication lag",
        "",
        f"- Negative publication lag records: {mv('publication_lag_negative_n')}",
        f"- Zero-day publication lag records: {mv('publication_lag_zero_n')}",
        f"- Positive publication lag records: {mv('publication_lag_positive_n')}",
        f"- Median publication lag: {mv('publication_lag_median_days')} days",
        f"- 95th percentile: {mv('publication_lag_p95_days')} days",
        f"- 99th percentile: {mv('publication_lag_p99_days')} days",
        "",
        "A negative publication lag means the dispatch date precedes the award date and should be inspected. "
        "For later leakage-safe public-information evaluation, historical availability should be determined by CAN dispatch/publication date relative to the target CN dispatch date; award date should be used for recency only after the record is admitted to the information set.",
        "",
        "## Exact duplicates inside repeated award keys",
        "",
    ]

    if not duplicate_summary.empty:
        r = duplicate_summary.iloc[0]
        lines.extend([
            f"- Repeated candidate rows: {r['repeated_candidate_rows']}",
            f"- Distinct repeated rows excluding provenance: {r['distinct_repeated_rows_excluding_provenance']}",
            f"- Exact duplicate excess rows: {r['exact_duplicate_excess_rows']}",
        ])

    lines.extend([
        "",
        "## Core conflict counts",
        "",
        "| Core field | Conflicting award keys |",
        "|---|---:|",
    ])

    for _, row in conflict_df.iterrows():
        lines.append(f"| `{row['field']}` | {int(row['conflicting_award_keys'])} |")

    lines.extend([
        "",
        "## Important limitation",
        "",
        "The provisional supplier key is not a final pan-European entity-resolution solution. "
        "Historical national-ID coverage is incomplete, so name/geography fallbacks are used conservatively. "
        "Do not build final supplier-history aggregates until the supplier-resolution diagnostics are reviewed.",
        "",
        "## Recommended next step",
        "",
        "Inspect the quarantine, provisional supplier-resolution coverage, XSD-stratified field coverage, "
        "and publication-lag outliers. Then construct a supplier dimension and time-valid historical feature snapshots.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized TED CAN award analytical tables."
    )

    parser.add_argument("--can-2015", required=True, type=Path)
    parser.add_argument("--can-2016", required=True, type=Path)
    parser.add_argument("--can-2017", required=True, type=Path)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/can_awards"),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/can_build"),
    )

    parser.add_argument(
        "--work-db",
        type=Path,
        default=None,
        help=(
            "DuckDB work database path. Defaults to "
            "<output-dir>/_can_build.duckdb."
        ),
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

    args = parser.parse_args()

    files = {
        2015: args.can_2015,
        2016: args.can_2016,
        2017: args.can_2017,
    }

    for year, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"CAN {year} file not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    work_db = (
        args.work_db
        if args.work_db is not None
        else args.output_dir / "_can_build.duckdb"
    )
    work_db.parent.mkdir(parents=True, exist_ok=True)

    delimiters = {
        year: detect_delimiter(path)
        for year, path in files.items()
    }

    print("Validating schemas...")
    validate_headers(files, delimiters)

    print(f"Opening DuckDB work database: {work_db}")
    con = duckdb.connect(str(work_db))
    con.execute(f"SET threads={int(args.threads)}")

    if args.memory_limit:
        mem = quote_sql_string(args.memory_limit)
        con.execute(f"SET memory_limit='{mem}'")

    temp_dir = args.output_dir / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")

    try:
        print("1/10 Building raw vertically appended table...")
        build_raw_table(con, files, delimiters)

        print("2/10 Normalizing fields and dates...")
        build_clean_table(con)

        print("3/10 Building candidate cohort and exclusion audit...")
        build_candidate_and_exclusion_tables(con)

        print("4/10 Detecting core award-key conflicts...")
        build_conflict_tables(con)

        print("5/10 Building award fact table...")
        build_award_fact(con)

        print("6/10 Building criteria child table...")
        build_award_criteria(con)

        print("7/10 Building lot-context child table...")
        build_award_lot_context(con)

        print("8/10 Building provenance table...")
        build_award_provenance(con)

        print("9/10 Running targeted diagnostics...")
        duplicate_df = repeated_key_exact_duplicate_summary(con)
        metrics_df = build_metrics(con, files)
        xsd_df = xsd_coverage_report(con)
        conflicts_df = conflict_summary(con)
        offsets_df = year_date_offset_report(con)

        print("10/10 Writing Parquet outputs and reports...")

        # Full cleaned combined table: useful for traceability and later audits.
        copy_table_to_parquet(
            con,
            "can_clean_keyed",
            args.output_dir / "can_combined_clean.parquet",
        )
        copy_table_to_parquet(
            con,
            "award_fact",
            args.output_dir / "award_fact_2015_2017.parquet",
        )
        copy_table_to_parquet(
            con,
            "award_criteria",
            args.output_dir / "award_criteria_2015_2017.parquet",
        )
        copy_table_to_parquet(
            con,
            "award_lot_context",
            args.output_dir / "award_lot_context_2015_2017.parquet",
        )
        copy_table_to_parquet(
            con,
            "award_provenance",
            args.output_dir / "award_provenance_2015_2017.parquet",
        )
        copy_table_to_parquet(
            con,
            "award_quarantine",
            args.output_dir / "award_quarantine_2015_2017.parquet",
        )
        copy_table_to_parquet(
            con,
            "row_exclusion_audit",
            args.output_dir / "row_exclusion_audit_2015_2017.parquet",
        )

        write_df_csv(
            duplicate_df,
            args.report_dir / "repeated_key_exact_duplicate_summary.csv",
        )
        write_df_csv(
            metrics_df,
            args.report_dir / "award_table_build_metrics.csv",
        )
        write_df_csv(
            xsd_df,
            args.report_dir / "xsd_field_coverage_award_candidates.csv",
        )
        write_df_csv(
            conflicts_df,
            args.report_dir / "core_conflict_summary.csv",
        )
        write_df_csv(
            offsets_df,
            args.report_dir / "source_year_date_offsets.csv",
        )

        create_markdown_report(
            metrics_df,
            duplicate_df,
            conflicts_df,
            args.report_dir / "award_table_build_report.md",
        )

        print()
        print("=" * 78)
        print("TED CAN award-table build complete")
        print("=" * 78)
        print(f"Processed data: {args.output_dir.resolve()}")
        print(f"Reports:        {args.report_dir.resolve()}")
        print(f"Work database:  {work_db.resolve()}")
        print()
        print("Review these first:")
        print("  award_table_build_report.md")
        print("  award_table_build_metrics.csv")
        print("  core_conflict_summary.csv")
        print("  repeated_key_exact_duplicate_summary.csv")
        print("  xsd_field_coverage_award_candidates.csv")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
