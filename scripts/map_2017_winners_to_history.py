#!/usr/bin/env python3
"""
Map strict 2017 procurement-case winners to the frozen 2015-2016 supplier history.

Inputs
------
1. procurement_cases_2017.parquet
2. supplier_dim_2015_2016.parquet
3. award_supplier_map_2015_2016.parquet
4. supplier_ambiguities_2015_2016.parquet

Core rule
---------
No identity learning is performed from 2017 records. All anchor, name-country,
name-town, and ambiguity lookups are learned exclusively from the frozen
2015-2016 historical supplier outputs.

Mapping precedence
------------------
1. direct_id_anchor_match
2. historically_ambiguous
3. unique_name_country_match
4. unique_name_town_match
5. no_historical_match
6. invalid_or_insufficient_identity

A valid 2017 ID match may resolve a winner even when its name-country key is
historically ambiguous. No fuzzy matching is performed.

Outputs
-------
Processed:
- winner_history_map_2017.parquet
- historical_name_country_lookup_2015_2016.parquet
- historical_name_town_lookup_2015_2016.parquet
- historical_id_anchor_lookup_2015_2016.parquet

Reports:
- winner_history_match_summary.csv
- winner_history_match_by_method.csv
- winner_history_match_by_country.csv
- winner_history_match_by_cpv2.csv
- winner_history_match_by_entity_tier.csv
- winner_history_ambiguity_summary.csv
- winner_history_mapping_metrics.csv
- winner_history_mapping_report.md

Requires:
    pip install duckdb pandas pyarrow
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


# ---------------------------------------------------------------------
# Candidate target-case column names
# ---------------------------------------------------------------------

COLUMN_CANDIDATES = {
    "case_id": [
        "ID_NOTICE_CN_CLEAN",
        "TARGET_CASE_ID",
        "CASE_ID",
        "ID_NOTICE_CN",
    ],
    "case_classification": [
        "CASE_CLASSIFICATION",
        "COHORT_CLASSIFICATION",
        "TARGET_CLASSIFICATION",
        "CLASSIFICATION",
    ],
    "strict_flag": [
        "IS_STRICT_STRUCTURAL_COHORT",
        "IS_STRICT_COHORT",
    ],
    "target_date": [
        "DT_DISPATCH_CLEAN",
        "CN_DT_DISPATCH_CLEAN",
        "TARGET_DISPATCH_DATE",
        "CN_DISPATCH_DATE",
    ],
    "cpv": [
        "CPV_CLEAN",
        "PRIMARY_CPV_CLEAN",
        "CN_CPV_CLEAN",
        "CPV",
    ],
    "country": [
        "ISO_COUNTRY_CODE_CLEAN",
        "CN_COUNTRY_CODE_CLEAN",
        "PROCUREMENT_COUNTRY",
        "ISO_COUNTRY_CODE",
    ],
    "can_notice": [
        "ID_NOTICE_CAN_CLEAN",
        "SINGLE_MATCHED_CAN_ID",
        "MATCHED_CAN_NOTICE_ID",
        "FUTURE_CAN_ID_CLEAN",
    ],
    "winner_name": [
        "ACTUAL_WINNER_NAME",
        "ACTUAL_WINNER_NAME_CLEAN",
        "WIN_NAME_CLEAN",
        "CAN_WIN_NAME_CLEAN",
        "WINNER_NAME",
    ],
    "winner_national_id": [
        "ACTUAL_WINNER_NATIONALID",
        "ACTUAL_WINNER_NATIONAL_ID",
        "ACTUAL_WINNER_NATIONALID_CLEAN",
        "WIN_NATIONALID_CLEAN",
        "CAN_WIN_NATIONALID_CLEAN",
        "WINNER_NATIONAL_ID",
    ],
    "winner_country": [
        "ACTUAL_WINNER_COUNTRY",
        "ACTUAL_WINNER_COUNTRY_CODE",
        "WIN_COUNTRY_CODE_CLEAN",
        "CAN_WIN_COUNTRY_CODE_CLEAN",
        "WINNER_COUNTRY",
    ],
    "winner_town": [
        "ACTUAL_WINNER_TOWN",
        "ACTUAL_WINNER_TOWN_CLEAN",
        "WIN_TOWN_CLEAN",
        "CAN_WIN_TOWN_CLEAN",
        "WINNER_TOWN",
    ],
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def clean_text_sql(column: str) -> str:
    return f"NULLIF(TRIM(CAST({qident(column)} AS VARCHAR)), '')"


def normalize_entity_sql(column: str) -> str:
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


def parquet_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{sql_path(path)}')
            """
        ).fetchall()
    ]


def resolve_column(
    available: Iterable[str],
    candidates: list[str],
    logical_name: str,
    required: bool = True,
) -> str | None:
    available_set = set(available)

    for candidate in candidates:
        if candidate in available_set:
            return candidate

    if required:
        raise ValueError(
            f"Could not resolve required target-case column '{logical_name}'. "
            f"Tried: {candidates}"
        )

    return None


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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def validate_historical_inputs(
    con: duckdb.DuckDBPyConnection,
    supplier_dim: Path,
    award_supplier_map: Path,
    supplier_ambiguities: Path,
) -> None:
    supplier_dim_required = {
        "SUPPLIER_ENTITY_ID",
        "CANONICAL_NAME",
        "SUPPLIER_COUNTRY",
        "ENTITY_RESOLUTION_TIER",
        "ENTITY_RESOLUTION_CONFIDENCE",
        "VALIDATED_NATIONAL_ID",
    }

    map_required = {
        "AWARD_KEY",
        "DT_AWARD_CLEAN",
        "DT_DISPATCH_CLEAN",
        "NAME_NORM",
        "NATIONAL_ID_NORM",
        "SUPPLIER_COUNTRY_NORM",
        "TOWN_NORM",
        "NAME_COUNTRY_KEY",
        "NAME_TOWN_KEY",
        "SUPPLIER_ENTITY_ID",
        "AWARD_RESOLUTION_STATUS",
    }

    ambiguity_required = {
        "NAME_COUNTRY_KEY",
        "CANDIDATE_SUPPLIER_ENTITY_ID",
        "SUPPORTING_AWARD_COUNT",
        "AMBIGUITY_REASON",
    }

    checks = [
        ("supplier_dim", supplier_dim, supplier_dim_required),
        ("award_supplier_map", award_supplier_map, map_required),
        ("supplier_ambiguities", supplier_ambiguities, ambiguity_required),
    ]

    for label, path, required in checks:
        cols = set(parquet_columns(con, path))
        missing = sorted(required - cols)
        if missing:
            raise ValueError(
                f"{label} is missing required columns:\n  - "
                + "\n  - ".join(missing)
            )


# ---------------------------------------------------------------------
# Resolve target-case schema
# ---------------------------------------------------------------------

def resolve_target_schema(
    con: duckdb.DuckDBPyConnection,
    cases_path: Path,
) -> dict[str, str | None]:
    cols = parquet_columns(con, cases_path)

    resolved = {
        "case_id": resolve_column(
            cols, COLUMN_CANDIDATES["case_id"], "case_id", True
        ),
        "case_classification": resolve_column(
            cols,
            COLUMN_CANDIDATES["case_classification"],
            "case_classification",
            False,
        ),
        "strict_flag": resolve_column(
            cols,
            COLUMN_CANDIDATES["strict_flag"],
            "strict_flag",
            False,
        ),
        "target_date": resolve_column(
            cols, COLUMN_CANDIDATES["target_date"], "target_date", True
        ),
        "cpv": resolve_column(
            cols, COLUMN_CANDIDATES["cpv"], "cpv", True
        ),
        "country": resolve_column(
            cols, COLUMN_CANDIDATES["country"], "country", False
        ),
        "can_notice": resolve_column(
            cols, COLUMN_CANDIDATES["can_notice"], "can_notice", False
        ),
        "winner_name": resolve_column(
            cols,
            COLUMN_CANDIDATES["winner_name"],
            "winner_name",
            True,
        ),
        "winner_national_id": resolve_column(
            cols,
            COLUMN_CANDIDATES["winner_national_id"],
            "winner_national_id",
            False,
        ),
        "winner_country": resolve_column(
            cols,
            COLUMN_CANDIDATES["winner_country"],
            "winner_country",
            True,
        ),
        "winner_town": resolve_column(
            cols,
            COLUMN_CANDIDATES["winner_town"],
            "winner_town",
            False,
        ),
    }

    if resolved["strict_flag"] is None and resolved["case_classification"] is None:
        raise ValueError(
            "Could not identify either a strict-cohort Boolean flag or a "
            "case-classification column in procurement_cases_2017.parquet."
        )

    return resolved


# ---------------------------------------------------------------------
# Strict target cohort
# ---------------------------------------------------------------------

def build_strict_target_base(
    con: duckdb.DuckDBPyConnection,
    cases_path: Path,
    cols: dict[str, str | None],
) -> None:
    case_id = qident(cols["case_id"])
    target_date = qident(cols["target_date"])
    cpv = qident(cols["cpv"])
    winner_name = qident(cols["winner_name"])
    winner_country = qident(cols["winner_country"])

    country_expr = (
        f"CAST({qident(cols['country'])} AS VARCHAR)"
        if cols["country"] is not None
        else "NULL::VARCHAR"
    )
    can_notice_expr = (
        f"CAST({qident(cols['can_notice'])} AS VARCHAR)"
        if cols["can_notice"] is not None
        else "NULL::VARCHAR"
    )
    winner_nid_expr = (
        f"CAST({qident(cols['winner_national_id'])} AS VARCHAR)"
        if cols["winner_national_id"] is not None
        else "NULL::VARCHAR"
    )
    winner_town_expr = (
        f"CAST({qident(cols['winner_town'])} AS VARCHAR)"
        if cols["winner_town"] is not None
        else "NULL::VARCHAR"
    )

    if cols["strict_flag"] is not None:
        strict_condition = (
            f"COALESCE(TRY_CAST({qident(cols['strict_flag'])} AS BOOLEAN), FALSE)"
        )
    else:
        strict_condition = (
            f"LOWER(TRIM(CAST({qident(cols['case_classification'])} AS VARCHAR))) "
            "= 'strict_single_link_single_award'"
        )

    name_norm = normalize_entity_sql(cols["winner_name"])
    winner_country_norm = (
        f"UPPER(NULLIF(TRIM(CAST({winner_country} AS VARCHAR)), ''))"
    )

    if cols["winner_national_id"] is not None:
        nid_norm = normalize_identifier_sql(cols["winner_national_id"])
    else:
        nid_norm = "NULL::VARCHAR"

    if cols["winner_town"] is not None:
        town_norm = normalize_entity_sql(cols["winner_town"])
    else:
        town_norm = "NULL::VARCHAR"

    con.execute("DROP TABLE IF EXISTS target_strict_2017")
    con.execute(
        f"""
        CREATE TABLE target_strict_2017 AS
        SELECT
            CAST({case_id} AS VARCHAR) AS TARGET_CASE_ID,
            CAST({target_date} AS DATE) AS TARGET_DISPATCH_DATE,
            CAST({cpv} AS VARCHAR) AS TARGET_CPV,
            REGEXP_EXTRACT(
                CAST({cpv} AS VARCHAR),
                '([0-9]{{2}})',
                1
            ) AS TARGET_CPV2,

            {country_expr} AS TARGET_PROCUREMENT_COUNTRY,
            {can_notice_expr} AS ID_NOTICE_CAN_CLEAN,

            CAST({winner_name} AS VARCHAR) AS ACTUAL_WINNER_NAME,
            {winner_nid_expr} AS ACTUAL_WINNER_NATIONAL_ID,
            CAST({winner_country} AS VARCHAR) AS ACTUAL_WINNER_COUNTRY,
            {winner_town_expr} AS ACTUAL_WINNER_TOWN,

            {name_norm} AS TARGET_WINNER_NAME_NORM,
            {nid_norm} AS TARGET_WINNER_NATIONAL_ID_NORM,
            {winner_country_norm} AS TARGET_WINNER_COUNTRY_NORM,
            {town_norm} AS TARGET_WINNER_TOWN_NORM,

            CASE
                WHEN {name_norm} IS NOT NULL
                 AND {winner_country_norm} IS NOT NULL
                 AND LENGTH({winner_country_norm}) = 2
                THEN {winner_country_norm} || '||' || {name_norm}
                ELSE NULL
            END AS TARGET_NAME_COUNTRY_KEY,

            CASE
                WHEN {name_norm} IS NOT NULL
                 AND {town_norm} IS NOT NULL
                THEN {town_norm} || '||' || {name_norm}
                ELSE NULL
            END AS TARGET_NAME_TOWN_KEY,

            CASE
                WHEN POSITION('---' IN COALESCE(CAST({winner_name} AS VARCHAR), '')) > 0
                    THEN TRUE
                WHEN POSITION('---' IN COALESCE({winner_nid_expr}, '')) > 0
                    THEN TRUE
                WHEN POSITION('---' IN COALESCE(CAST({winner_country} AS VARCHAR), '')) > 0
                    THEN TRUE
                WHEN POSITION('---' IN COALESCE({winner_town_expr}, '')) > 0
                    THEN TRUE
                ELSE FALSE
            END AS TARGET_HAS_LIST_VALUED_IDENTITY

        FROM read_parquet('{sql_path(cases_path)}')
        WHERE {strict_condition}
        """
    )

    duplicate_cases = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT TARGET_CASE_ID
            FROM target_strict_2017
            GROUP BY TARGET_CASE_ID
            HAVING COUNT(*) > 1
        )
        """,
    )

    if duplicate_cases:
        raise ValueError(
            f"Strict target table contains {duplicate_cases} duplicate TARGET_CASE_ID values. "
            "Expected one row per strict procurement case."
        )


# ---------------------------------------------------------------------
# Optional target winner-town enrichment from CAN 2017 award fact
# ---------------------------------------------------------------------

def enrich_target_town_from_can(
    con: duckdb.DuckDBPyConnection,
    can_award_fact_path: Path,
) -> None:
    """
    Enrich target winner town when procurement_cases_2017 does not contain it.

    The lookup is descriptive only. It does not learn supplier identities from 2017.
    It retrieves the observed winner-town field from the already constructed CAN
    award fact for the same matched CAN notice.
    """
    required = {
        "ID_NOTICE_CAN_CLEAN",
        "SOURCE_YEAR_MIN",
        "SOURCE_YEAR_MAX",
        "WIN_TOWN_CLEAN",
    }
    cols = set(parquet_columns(con, can_award_fact_path))
    missing = sorted(required - cols)
    if missing:
        raise ValueError(
            "Optional CAN award fact is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )

    con.execute("DROP TABLE IF EXISTS target_can_winner_town_lookup")
    con.execute(
        f"""
        CREATE TABLE target_can_winner_town_lookup AS
        SELECT
            ID_NOTICE_CAN_CLEAN,
            CASE
                WHEN COUNT(DISTINCT WIN_TOWN_CLEAN) FILTER (
                    WHERE WIN_TOWN_CLEAN IS NOT NULL
                ) = 1
                THEN MAX(WIN_TOWN_CLEAN)
                ELSE NULL
            END AS WIN_TOWN_CLEAN,
            COUNT(DISTINCT WIN_TOWN_CLEAN) FILTER (
                WHERE WIN_TOWN_CLEAN IS NOT NULL
            ) AS N_DISTINCT_WINNER_TOWNS
        FROM read_parquet('{sql_path(can_award_fact_path)}')
        WHERE
            SOURCE_YEAR_MIN = 2017
            AND SOURCE_YEAR_MAX = 2017
            AND ID_NOTICE_CAN_CLEAN IS NOT NULL
        GROUP BY ID_NOTICE_CAN_CLEAN
        """
    )

    con.execute("DROP TABLE IF EXISTS target_strict_2017_enriched")
    con.execute(
        """
        CREATE TABLE target_strict_2017_enriched AS
        SELECT
            t.* EXCLUDE (
                ACTUAL_WINNER_TOWN,
                TARGET_WINNER_TOWN_NORM,
                TARGET_NAME_TOWN_KEY,
                TARGET_HAS_LIST_VALUED_IDENTITY
            ),

            c.WIN_TOWN_CLEAN AS ACTUAL_WINNER_TOWN,

            NULLIF(
                TRIM(
                    REGEXP_REPLACE(
                        LOWER(COALESCE(c.WIN_TOWN_CLEAN, '')),
                        '[^[:alnum:]]+',
                        ' ',
                        'g'
                    )
                ),
                ''
            ) AS TARGET_WINNER_TOWN_NORM,

            CASE
                WHEN t.TARGET_WINNER_NAME_NORM IS NOT NULL
                 AND NULLIF(
                        TRIM(
                            REGEXP_REPLACE(
                                LOWER(COALESCE(c.WIN_TOWN_CLEAN, '')),
                                '[^[:alnum:]]+',
                                ' ',
                                'g'
                            )
                        ),
                        ''
                     ) IS NOT NULL
                THEN
                    NULLIF(
                        TRIM(
                            REGEXP_REPLACE(
                                LOWER(COALESCE(c.WIN_TOWN_CLEAN, '')),
                                '[^[:alnum:]]+',
                                ' ',
                                'g'
                            )
                        ),
                        ''
                    ) || '||' || t.TARGET_WINNER_NAME_NORM
                ELSE NULL
            END AS TARGET_NAME_TOWN_KEY,

            CASE
                WHEN t.TARGET_HAS_LIST_VALUED_IDENTITY THEN TRUE
                WHEN POSITION('---' IN COALESCE(c.WIN_TOWN_CLEAN, '')) > 0
                    THEN TRUE
                ELSE FALSE
            END AS TARGET_HAS_LIST_VALUED_IDENTITY

        FROM target_strict_2017 t
        LEFT JOIN target_can_winner_town_lookup c
          USING (ID_NOTICE_CAN_CLEAN)
        """
    )

    con.execute("DROP TABLE target_strict_2017")
    con.execute(
        "ALTER TABLE target_strict_2017_enriched RENAME TO target_strict_2017"
    )

# ---------------------------------------------------------------------
# Frozen historical lookups
# ---------------------------------------------------------------------

def build_historical_lookups(
    con: duckdb.DuckDBPyConnection,
    supplier_dim_path: Path,
    award_supplier_map_path: Path,
    ambiguities_path: Path,
) -> None:
    con.execute("DROP TABLE IF EXISTS historical_id_anchor_lookup")
    con.execute(
        f"""
        CREATE TABLE historical_id_anchor_lookup AS
        SELECT
            SUPPLIER_ENTITY_ID,
            UPPER(NULLIF(TRIM(SUPPLIER_COUNTRY), ''))
                AS SUPPLIER_COUNTRY_NORM,
            UPPER(
                REGEXP_REPLACE(
                    COALESCE(VALIDATED_NATIONAL_ID, ''),
                    '[^[:alnum:]]+',
                    '',
                    'g'
                )
            ) AS NATIONAL_ID_NORM,
            CANONICAL_NAME,
            ENTITY_RESOLUTION_TIER,
            ENTITY_RESOLUTION_CONFIDENCE
        FROM read_parquet('{sql_path(supplier_dim_path)}')
        WHERE
            VALIDATED_NATIONAL_ID IS NOT NULL
            AND SUPPLIER_COUNTRY IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS historical_name_country_lookup")
    con.execute(
        f"""
        CREATE TABLE historical_name_country_lookup AS
        WITH support AS (
            SELECT
                NAME_COUNTRY_KEY,
                SUPPLIER_ENTITY_ID,
                COUNT(*) AS N_HISTORY_AWARDS,
                MIN(DT_AWARD_CLEAN) AS FIRST_HISTORY_AWARD_DATE,
                MAX(DT_AWARD_CLEAN) AS LAST_HISTORY_AWARD_DATE,
                MIN(DT_DISPATCH_CLEAN) AS FIRST_HISTORY_PUBLICATION_DATE,
                MAX(DT_DISPATCH_CLEAN) AS LAST_HISTORY_PUBLICATION_DATE
            FROM read_parquet('{sql_path(award_supplier_map_path)}')
            WHERE
                NAME_COUNTRY_KEY IS NOT NULL
                AND SUPPLIER_ENTITY_ID IS NOT NULL
            GROUP BY
                NAME_COUNTRY_KEY,
                SUPPLIER_ENTITY_ID
        ),
        profile AS (
            SELECT
                NAME_COUNTRY_KEY,
                COUNT(DISTINCT SUPPLIER_ENTITY_ID)
                    AS N_HISTORICAL_ENTITIES,
                SUM(N_HISTORY_AWARDS)
                    AS TOTAL_HISTORY_AWARDS
            FROM support
            GROUP BY NAME_COUNTRY_KEY
        )
        SELECT
            s.*,
            p.N_HISTORICAL_ENTITIES,
            p.TOTAL_HISTORY_AWARDS,
            CASE
                WHEN p.N_HISTORICAL_ENTITIES = 1
                    THEN 'unique'
                ELSE 'multiple_resolved_entities'
            END AS LOOKUP_STATUS
        FROM support s
        JOIN profile p
          USING (NAME_COUNTRY_KEY)
        """
    )

    con.execute("DROP TABLE IF EXISTS historical_name_town_lookup")
    con.execute(
        f"""
        CREATE TABLE historical_name_town_lookup AS
        WITH support AS (
            SELECT
                NAME_TOWN_KEY,
                SUPPLIER_ENTITY_ID,
                COUNT(*) AS N_HISTORY_AWARDS,
                MIN(DT_AWARD_CLEAN) AS FIRST_HISTORY_AWARD_DATE,
                MAX(DT_AWARD_CLEAN) AS LAST_HISTORY_AWARD_DATE,
                MIN(DT_DISPATCH_CLEAN) AS FIRST_HISTORY_PUBLICATION_DATE,
                MAX(DT_DISPATCH_CLEAN) AS LAST_HISTORY_PUBLICATION_DATE
            FROM read_parquet('{sql_path(award_supplier_map_path)}')
            WHERE
                NAME_TOWN_KEY IS NOT NULL
                AND SUPPLIER_ENTITY_ID IS NOT NULL
            GROUP BY
                NAME_TOWN_KEY,
                SUPPLIER_ENTITY_ID
        ),
        profile AS (
            SELECT
                NAME_TOWN_KEY,
                COUNT(DISTINCT SUPPLIER_ENTITY_ID)
                    AS N_HISTORICAL_ENTITIES,
                SUM(N_HISTORY_AWARDS)
                    AS TOTAL_HISTORY_AWARDS
            FROM support
            GROUP BY NAME_TOWN_KEY
        )
        SELECT
            s.*,
            p.N_HISTORICAL_ENTITIES,
            p.TOTAL_HISTORY_AWARDS,
            CASE
                WHEN p.N_HISTORICAL_ENTITIES = 1
                    THEN 'unique'
                ELSE 'multiple_resolved_entities'
            END AS LOOKUP_STATUS
        FROM support s
        JOIN profile p
          USING (NAME_TOWN_KEY)
        """
    )

    con.execute("DROP TABLE IF EXISTS historical_ambiguous_name_country_keys")
    con.execute(
        f"""
        CREATE TABLE historical_ambiguous_name_country_keys AS
        SELECT
            NAME_COUNTRY_KEY,
            COUNT(DISTINCT CANDIDATE_SUPPLIER_ENTITY_ID)
                AS N_CANDIDATE_HISTORICAL_ENTITIES,
            SUM(SUPPORTING_AWARD_COUNT)
                AS TOTAL_SUPPORTING_AWARDS
        FROM read_parquet('{sql_path(ambiguities_path)}')
        GROUP BY NAME_COUNTRY_KEY
        """
    )

    # Enrich unique historical lookup rows with supplier-dimension metadata.
    con.execute("DROP TABLE IF EXISTS historical_name_country_unique")
    con.execute(
        f"""
        CREATE TABLE historical_name_country_unique AS
        SELECT
            l.NAME_COUNTRY_KEY,
            l.SUPPLIER_ENTITY_ID,
            l.N_HISTORY_AWARDS,
            l.FIRST_HISTORY_AWARD_DATE,
            l.LAST_HISTORY_AWARD_DATE,
            l.FIRST_HISTORY_PUBLICATION_DATE,
            l.LAST_HISTORY_PUBLICATION_DATE,
            d.CANONICAL_NAME,
            d.SUPPLIER_COUNTRY,
            d.ENTITY_RESOLUTION_TIER,
            d.ENTITY_RESOLUTION_CONFIDENCE
        FROM historical_name_country_lookup l
        LEFT JOIN read_parquet('{sql_path(supplier_dim_path)}') d
          USING (SUPPLIER_ENTITY_ID)
        WHERE l.LOOKUP_STATUS = 'unique'
        """
    )

    con.execute("DROP TABLE IF EXISTS historical_name_town_unique")
    con.execute(
        f"""
        CREATE TABLE historical_name_town_unique AS
        SELECT
            l.NAME_TOWN_KEY,
            l.SUPPLIER_ENTITY_ID,
            l.N_HISTORY_AWARDS,
            l.FIRST_HISTORY_AWARD_DATE,
            l.LAST_HISTORY_AWARD_DATE,
            l.FIRST_HISTORY_PUBLICATION_DATE,
            l.LAST_HISTORY_PUBLICATION_DATE,
            d.CANONICAL_NAME,
            d.SUPPLIER_COUNTRY,
            d.ENTITY_RESOLUTION_TIER,
            d.ENTITY_RESOLUTION_CONFIDENCE
        FROM historical_name_town_lookup l
        LEFT JOIN read_parquet('{sql_path(supplier_dim_path)}') d
          USING (SUPPLIER_ENTITY_ID)
        WHERE l.LOOKUP_STATUS = 'unique'
        """
    )


# ---------------------------------------------------------------------
# Winner-to-history mapping
# ---------------------------------------------------------------------

def build_winner_history_map(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Precedence:
      direct ID anchor
      historical name-country ambiguity
      unique name-country
      unique name-town
      no historical match
      invalid/insufficient identity

    A direct ID match overrides a historically ambiguous name-country key.
    """
    con.execute("DROP TABLE IF EXISTS winner_history_map")
    con.execute(
        """
        CREATE TABLE winner_history_map AS
        SELECT
            t.*,

            CASE
                WHEN t.TARGET_HAS_LIST_VALUED_IDENTITY
                    THEN NULL

                WHEN idm.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN idm.SUPPLIER_ENTITY_ID

                WHEN amb.NAME_COUNTRY_KEY IS NOT NULL
                    THEN NULL

                WHEN nc.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN nc.SUPPLIER_ENTITY_ID

                WHEN nt.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN nt.SUPPLIER_ENTITY_ID

                ELSE NULL
            END AS HISTORICAL_SUPPLIER_ENTITY_ID,

            CASE
                WHEN t.TARGET_HAS_LIST_VALUED_IDENTITY
                    THEN 'invalid_or_insufficient_identity'

                WHEN idm.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN 'direct_id_anchor_match'

                WHEN amb.NAME_COUNTRY_KEY IS NOT NULL
                    THEN 'historically_ambiguous'

                WHEN nc.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN 'unique_name_country_match'

                WHEN nt.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN 'unique_name_town_match'

                WHEN t.TARGET_WINNER_NAME_NORM IS NULL
                    THEN 'invalid_or_insufficient_identity'

                WHEN t.TARGET_WINNER_COUNTRY_NORM IS NULL
                 AND t.TARGET_WINNER_TOWN_NORM IS NULL
                    THEN 'invalid_or_insufficient_identity'

                ELSE 'no_historical_match'
            END AS HISTORY_MATCH_STATUS,

            CASE
                WHEN t.TARGET_HAS_LIST_VALUED_IDENTITY
                    THEN FALSE

                WHEN idm.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN TRUE
                WHEN amb.NAME_COUNTRY_KEY IS NOT NULL
                    THEN FALSE
                WHEN nc.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN TRUE
                WHEN nt.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN TRUE
                ELSE FALSE
            END AS HISTORICAL_MATCHED,

            CASE
                WHEN idm.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN 'high'
                WHEN nc.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN 'medium_high'
                WHEN nt.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN 'low'
                ELSE NULL
            END AS HISTORY_MATCH_CONFIDENCE,

            COALESCE(
                idm.CANONICAL_NAME,
                nc.CANONICAL_NAME,
                nt.CANONICAL_NAME
            ) AS HISTORICAL_CANONICAL_NAME,

            COALESCE(
                idm.ENTITY_RESOLUTION_TIER,
                nc.ENTITY_RESOLUTION_TIER,
                nt.ENTITY_RESOLUTION_TIER
            ) AS HISTORICAL_ENTITY_TIER,

            COALESCE(
                idm.ENTITY_RESOLUTION_CONFIDENCE,
                nc.ENTITY_RESOLUTION_CONFIDENCE,
                nt.ENTITY_RESOLUTION_CONFIDENCE
            ) AS HISTORICAL_ENTITY_CONFIDENCE,

            CASE
                WHEN idm.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN NULL
                WHEN amb.NAME_COUNTRY_KEY IS NOT NULL
                    THEN amb.N_CANDIDATE_HISTORICAL_ENTITIES
                ELSE NULL
            END AS N_AMBIGUOUS_HISTORICAL_CANDIDATES,

            COALESCE(
                nc.N_HISTORY_AWARDS,
                nt.N_HISTORY_AWARDS
            ) AS LOOKUP_SUPPORT_HISTORY_AWARDS,

            COALESCE(
                nc.FIRST_HISTORY_AWARD_DATE,
                nt.FIRST_HISTORY_AWARD_DATE
            ) AS LOOKUP_FIRST_HISTORY_AWARD_DATE,

            COALESCE(
                nc.LAST_HISTORY_AWARD_DATE,
                nt.LAST_HISTORY_AWARD_DATE
            ) AS LOOKUP_LAST_HISTORY_AWARD_DATE,

            COALESCE(
                nc.FIRST_HISTORY_PUBLICATION_DATE,
                nt.FIRST_HISTORY_PUBLICATION_DATE
            ) AS LOOKUP_FIRST_HISTORY_PUBLICATION_DATE,

            COALESCE(
                nc.LAST_HISTORY_PUBLICATION_DATE,
                nt.LAST_HISTORY_PUBLICATION_DATE
            ) AS LOOKUP_LAST_HISTORY_PUBLICATION_DATE

        FROM target_strict_2017 t

        LEFT JOIN historical_id_anchor_lookup idm
          ON t.TARGET_WINNER_COUNTRY_NORM = idm.SUPPLIER_COUNTRY_NORM
         AND t.TARGET_WINNER_NATIONAL_ID_NORM = idm.NATIONAL_ID_NORM

        LEFT JOIN historical_ambiguous_name_country_keys amb
          ON t.TARGET_NAME_COUNTRY_KEY = amb.NAME_COUNTRY_KEY

        LEFT JOIN historical_name_country_unique nc
          ON t.TARGET_NAME_COUNTRY_KEY = nc.NAME_COUNTRY_KEY

        LEFT JOIN historical_name_town_unique nt
          ON t.TARGET_NAME_TOWN_KEY = nt.NAME_TOWN_KEY
        """
    )

    # Defensive uniqueness validation.
    duplicate_cases = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT TARGET_CASE_ID
            FROM winner_history_map
            GROUP BY TARGET_CASE_ID
            HAVING COUNT(*) > 1
        )
        """,
    )

    if duplicate_cases:
        raise ValueError(
            f"Winner mapping created {duplicate_cases} duplicated target-case IDs. "
            "Historical lookup uniqueness assumptions were violated."
        )


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def summary_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        WITH base AS (
            SELECT
                COUNT(*) AS N_STRICT_CASES,
                COUNT(*) FILTER (
                    WHERE HISTORICAL_MATCHED
                ) AS N_HISTORICAL_MATCHED,
                COUNT(*) FILTER (
                    WHERE NOT HISTORICAL_MATCHED
                ) AS N_NOT_MATCHED
            FROM winner_history_map
        )
        SELECT
            N_STRICT_CASES,
            N_HISTORICAL_MATCHED,
            N_NOT_MATCHED,
            100.0 * N_HISTORICAL_MATCHED
                / NULLIF(N_STRICT_CASES, 0)
                AS HISTORICAL_WINNER_MATCH_RATE_PCT
        FROM base
        """
    )


def by_method_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            HISTORY_MATCH_STATUS,
            COUNT(*) AS N_CASES,
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                AS SHARE_OF_STRICT_CASES_PCT,
            COUNT(*) FILTER (
                WHERE HISTORICAL_MATCHED
            ) AS N_MATCHED
        FROM winner_history_map
        GROUP BY HISTORY_MATCH_STATUS
        ORDER BY N_CASES DESC
        """
    )


def by_country_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COALESCE(TARGET_WINNER_COUNTRY_NORM, '<MISSING>')
                AS WINNER_COUNTRY,
            COUNT(*) AS N_CASES,
            COUNT(*) FILTER (
                WHERE HISTORICAL_MATCHED
            ) AS N_MATCHED,
            100.0 * COUNT(*) FILTER (
                WHERE HISTORICAL_MATCHED
            ) / COUNT(*) AS MATCH_RATE_PCT
        FROM winner_history_map
        GROUP BY WINNER_COUNTRY
        ORDER BY N_CASES DESC
        """
    )


def by_cpv2_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COALESCE(TARGET_CPV2, '<MISSING>') AS TARGET_CPV2,
            COUNT(*) AS N_CASES,
            COUNT(*) FILTER (
                WHERE HISTORICAL_MATCHED
            ) AS N_MATCHED,
            100.0 * COUNT(*) FILTER (
                WHERE HISTORICAL_MATCHED
            ) / COUNT(*) AS MATCH_RATE_PCT
        FROM winner_history_map
        GROUP BY TARGET_CPV2
        ORDER BY N_CASES DESC
        """
    )


def by_entity_tier_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COALESCE(HISTORICAL_ENTITY_TIER, '<NO_MATCH>')
                AS HISTORICAL_ENTITY_TIER,
            HISTORY_MATCH_STATUS,
            COUNT(*) AS N_CASES,
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                AS SHARE_OF_STRICT_CASES_PCT
        FROM winner_history_map
        GROUP BY
            HISTORICAL_ENTITY_TIER,
            HISTORY_MATCH_STATUS
        ORDER BY N_CASES DESC
        """
    )


def ambiguity_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            COALESCE(
                N_AMBIGUOUS_HISTORICAL_CANDIDATES,
                0
            ) AS N_AMBIGUOUS_HISTORICAL_CANDIDATES,
            COUNT(*) AS N_CASES
        FROM winner_history_map
        WHERE HISTORY_MATCH_STATUS = 'historically_ambiguous'
        GROUP BY N_AMBIGUOUS_HISTORICAL_CANDIDATES
        ORDER BY N_AMBIGUOUS_HISTORICAL_CANDIDATES
        """
    )


def build_metrics(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(metric: str, value, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    n = scalar(con, "SELECT COUNT(*) FROM winner_history_map")
    matched = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM winner_history_map
        WHERE HISTORICAL_MATCHED
        """,
    )

    add("strict_target_cases", n)
    add("historically_matched_winners", matched)
    add("historically_unmatched_or_ambiguous_winners", n - matched)
    add(
        "historical_winner_match_rate_pct",
        100.0 * matched / n if n else None,
    )

    status_df = by_method_report(con)
    for _, row in status_df.iterrows():
        add(
            f"match_status__{row['HISTORY_MATCH_STATUS']}",
            int(row["N_CASES"]),
        )

    add(
        "distinct_matched_historical_entities",
        scalar(
            con,
            """
            SELECT COUNT(DISTINCT HISTORICAL_SUPPLIER_ENTITY_ID)
            FROM winner_history_map
            WHERE HISTORICAL_SUPPLIER_ENTITY_ID IS NOT NULL
            """,
        ),
    )

    add(
        "historically_ambiguous_cases",
        scalar(
            con,
            """
            SELECT COUNT(*)
            FROM winner_history_map
            WHERE HISTORY_MATCH_STATUS = 'historically_ambiguous'
            """,
        ),
    )

    add(
        "no_historical_match_cases",
        scalar(
            con,
            """
            SELECT COUNT(*)
            FROM winner_history_map
            WHERE HISTORY_MATCH_STATUS = 'no_historical_match'
            """,
        ),
    )

    add(
        "invalid_or_insufficient_identity_cases",
        scalar(
            con,
            """
            SELECT COUNT(*)
            FROM winner_history_map
            WHERE HISTORY_MATCH_STATUS = 'invalid_or_insufficient_identity'
            """,
        ),
    )

    return pd.DataFrame(rows)


def create_markdown_report(
    summary_df: pd.DataFrame,
    method_df: pd.DataFrame,
    output_path: Path,
) -> None:
    s = summary_df.iloc[0]

    lines = [
        "# TED 2017 Winner-to-History Mapping Report",
        "",
        "## Scope",
        "",
        "This analysis maps observed winners from the strict 2017 procurement-case cohort "
        "to the frozen 2015-2016 historical supplier dimension. No 2017 record is used "
        "to learn new anchors, aliases, bridges, or ambiguity rules.",
        "",
        "## Headline coverage",
        "",
        f"- Strict 2017 target cases: {int(s['N_STRICT_CASES'])}",
        f"- Winners matched to historical supplier entities: {int(s['N_HISTORICAL_MATCHED'])}",
        f"- Winners not historically matched or preserved as ambiguous: {int(s['N_NOT_MATCHED'])}",
        f"- Historical winner match rate: {float(s['HISTORICAL_WINNER_MATCH_RATE_PCT']):.3f}%",
        "",
        "## Mapping methods",
        "",
        "| Mapping status | Cases | Share of strict cohort (%) |",
        "|---|---:|---:|",
    ]

    for _, row in method_df.iterrows():
        lines.append(
            f"| `{row['HISTORY_MATCH_STATUS']}` "
            f"| {int(row['N_CASES'])} "
            f"| {float(row['SHARE_OF_STRICT_CASES_PCT']):.3f} |"
        )

    lines.extend([
        "",
        "## Mapping policy",
        "",
        "1. Direct validated historical ID-anchor matches have highest precedence.",
        "2. A valid direct ID match may resolve a winner even when its name-country key is historically ambiguous.",
        "3. Historically ambiguous name-country keys are not forced to one supplier entity.",
        "4. Unique frozen 2015-2016 name-country matches are used next.",
        "5. Unique frozen name-town matches are used only as a lower-confidence fallback.",
        "6. No fuzzy matching is performed.",
        "7. A valid 2017 supplier with no 2015-2016 match is classified as `no_historical_match`, not as an invalid supplier.",
        "",
        "## Interpretation",
        "",
        "Historical winner match rate measures whether the observed 2017 winner is represented "
        "in the frozen 2015-2016 supplier universe. It is not a ranking-accuracy metric. "
        "The next step is to construct case-specific CPV-contextual historically active supplier "
        "comparison sets and then measure winner candidate coverage.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Map strict 2017 observed winners to the frozen 2015-2016 "
            "historical supplier dimension."
        )
    )

    parser.add_argument(
        "--procurement-cases",
        required=True,
        type=Path,
        help="Path to procurement_cases_2017.parquet",
    )
    parser.add_argument(
        "--supplier-dim",
        required=True,
        type=Path,
        help="Path to supplier_dim_2015_2016.parquet",
    )
    parser.add_argument(
        "--award-supplier-map",
        required=True,
        type=Path,
        help="Path to award_supplier_map_2015_2016.parquet",
    )
    parser.add_argument(
        "--supplier-ambiguities",
        required=True,
        type=Path,
        help="Path to supplier_ambiguities_2015_2016.parquet",
    )
    parser.add_argument(
        "--can-award-fact",
        type=Path,
        default=None,
        help=(
            "Optional award_fact_2015_2017.parquet. Recommended when the "
            "procurement-case table does not contain winner town; used only to "
            "retrieve observed 2017 winner town for the low-confidence name-town "
            "fallback, not to learn identity links."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/winner_history_2017"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/winner_history_2017"),
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

    args = parser.parse_args()

    inputs = [
        args.procurement_cases,
        args.supplier_dim,
        args.award_supplier_map,
        args.supplier_ambiguities,
    ]
    if args.can_award_fact is not None:
        inputs.append(args.can_award_fact)
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    work_db = (
        args.work_db
        if args.work_db is not None
        else args.output_dir / "_winner_history_mapping.duckdb"
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
        print("1/8 Validating frozen historical inputs...")
        validate_historical_inputs(
            con,
            args.supplier_dim,
            args.award_supplier_map,
            args.supplier_ambiguities,
        )

        print("2/8 Resolving target-case schema...")
        target_cols = resolve_target_schema(
            con,
            args.procurement_cases,
        )

        print("Resolved target columns:")
        for logical, physical in target_cols.items():
            print(f"  {logical}: {physical}")

        print("3/8 Building strict 2017 target winner base...")
        build_strict_target_base(
            con,
            args.procurement_cases,
            target_cols,
        )

        if target_cols["winner_town"] is None and args.can_award_fact is not None:
            print("   Enriching observed winner town from CAN 2017 award fact...")
            enrich_target_town_from_can(con, args.can_award_fact)
        elif target_cols["winner_town"] is None:
            print(
                "   Winner town unavailable; name-town fallback will be inactive. "
                "Provide --can-award-fact to enable observational town enrichment."
            )

        print("4/8 Building frozen 2015-2016 historical lookups...")
        build_historical_lookups(
            con,
            args.supplier_dim,
            args.award_supplier_map,
            args.supplier_ambiguities,
        )

        print("5/8 Mapping 2017 winners to historical entities...")
        build_winner_history_map(con)

        print("6/8 Building reports...")
        summary_df = summary_report(con)
        method_df = by_method_report(con)
        country_df = by_country_report(con)
        cpv2_df = by_cpv2_report(con)
        tier_df = by_entity_tier_report(con)
        ambiguity_df = ambiguity_report(con)
        metrics_df = build_metrics(con)

        write_csv(
            summary_df,
            args.report_dir / "winner_history_match_summary.csv",
        )
        write_csv(
            method_df,
            args.report_dir / "winner_history_match_by_method.csv",
        )
        write_csv(
            country_df,
            args.report_dir / "winner_history_match_by_country.csv",
        )
        write_csv(
            cpv2_df,
            args.report_dir / "winner_history_match_by_cpv2.csv",
        )
        write_csv(
            tier_df,
            args.report_dir / "winner_history_match_by_entity_tier.csv",
        )
        write_csv(
            ambiguity_df,
            args.report_dir / "winner_history_ambiguity_summary.csv",
        )
        write_csv(
            metrics_df,
            args.report_dir / "winner_history_mapping_metrics.csv",
        )

        create_markdown_report(
            summary_df,
            method_df,
            args.report_dir / "winner_history_mapping_report.md",
        )

        print("7/8 Writing Parquet outputs...")
        copy_table_to_parquet(
            con,
            "winner_history_map",
            args.output_dir / "winner_history_map_2017.parquet",
        )
        copy_table_to_parquet(
            con,
            "historical_id_anchor_lookup",
            args.output_dir / "historical_id_anchor_lookup_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "historical_name_country_lookup",
            args.output_dir / "historical_name_country_lookup_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "historical_name_town_lookup",
            args.output_dir / "historical_name_town_lookup_2015_2016.parquet",
        )

        print("8/8 Complete.")
        print()
        print("=" * 78)
        print("2017 winner-to-history mapping complete")
        print("=" * 78)
        print(f"Processed outputs: {args.output_dir.resolve()}")
        print(f"Reports:           {args.report_dir.resolve()}")
        print(f"Work database:     {work_db.resolve()}")
        print()
        print("Review these first:")
        print("  winner_history_mapping_report.md")
        print("  winner_history_match_summary.csv")
        print("  winner_history_match_by_method.csv")
        print("  winner_history_match_by_country.csv")
        print("  winner_history_match_by_cpv2.csv")
        print("  winner_history_ambiguity_summary.csv")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
