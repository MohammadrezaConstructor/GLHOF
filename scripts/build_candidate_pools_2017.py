#!/usr/bin/env python3
"""
Build context-based historical supplier candidate pools for strict 2017 cases.

Purpose
-------
This is the first of the final two main data-construction steps.

The script combines:
    winner_history_map_2017.parquet
    award_supplier_map_2015_2016.parquet
    award_fact_2015_2017.parquet
    supplier_dim_2015_2016.parquet

It constructs historically active supplier comparison sets using frozen
2015-2016 history only.

Pool definitions
----------------
Primary:
    CPV2_MIN1
        Same CPV 2-digit division, at least 1 historical award.

Sensitivity:
    CPV3_MIN1
        Same CPV 3-digit group prefix, at least 1 historical award.

    CPV4_MIN1
        Same CPV 4-digit class prefix, at least 1 historical award.

    CPV2_MIN2
        Same CPV 2-digit division, at least 2 historical awards in that
        division.

Important interpretation
------------------------
These are historically active supplier comparison sets. They are NOT actual
bidder sets because unsuccessful bidder identities are not available in the
CAN award-fact data.

Scalable representation
-----------------------
By default, the script does NOT materialize the full
(target_case_id, supplier_entity_id) cross-product. That table can be very
large.

Instead it writes:
1. historical_supplier_context_eligibility_2015_2016.parquet
   Grain: pool_definition, context_key, supplier_entity_id

2. target_pool_assignments_2017.parquet
   Grain: target_case_id, pool_definition

3. candidate_pool_case_summary_2017.parquet
   Grain: target_case_id, pool_definition

These three tables fully define the candidate pools and are enough to compute
coverage and pool-size diagnostics.

Use --materialize-case-supplier-membership only if you explicitly need the
full long membership table.

Coverage metrics
----------------
ALL_CASE_COVERAGE:
    winner included / all strict target cases

CONDITIONAL_HISTORICAL_COVERAGE:
    winner included / target cases whose observed winner was already matched
    somewhere in the frozen historical supplier universe

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

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def parquet_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{sql_path(path)}')
            """
        ).fetchall()
    }


def require_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    required: set[str],
    label: str,
) -> None:
    cols = parquet_columns(con, path)
    missing = sorted(required - cols)
    if missing:
        raise ValueError(
            f"{label} is missing required columns:\n  - "
            + "\n  - ".join(missing)
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
        COPY {qident(table_name)}
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fetch_df(
    con: duckdb.DuckDBPyConnection,
    sql: str,
) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def scalar(
    con: duckdb.DuckDBPyConnection,
    sql: str,
):
    return con.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_inputs(
    con: duckdb.DuckDBPyConnection,
    winner_history_map: Path,
    award_supplier_map: Path,
    award_fact: Path,
    supplier_dim: Path,
) -> None:
    require_columns(
        con,
        winner_history_map,
        {
            "TARGET_CASE_ID",
            "TARGET_DISPATCH_DATE",
            "TARGET_CPV",
            "TARGET_CPV2",
            "TARGET_PROCUREMENT_COUNTRY",
            "HISTORICAL_SUPPLIER_ENTITY_ID",
            "HISTORICAL_MATCHED",
            "HISTORY_MATCH_STATUS",
        },
        "winner_history_map",
    )

    require_columns(
        con,
        award_supplier_map,
        {
            "AWARD_KEY",
            "SOURCE_YEAR_MIN",
            "SOURCE_YEAR_MAX",
            "DT_DISPATCH_CLEAN",
            "SUPPLIER_ENTITY_ID",
            "AWARD_RESOLUTION_STATUS",
        },
        "award_supplier_map",
    )

    require_columns(
        con,
        award_fact,
        {
            "AWARD_KEY",
            "SOURCE_YEAR_MIN",
            "SOURCE_YEAR_MAX",
            "CPV_CLEAN",
        },
        "award_fact",
    )

    require_columns(
        con,
        supplier_dim,
        {
            "SUPPLIER_ENTITY_ID",
            "CANONICAL_NAME",
            "SUPPLIER_COUNTRY",
            "ENTITY_RESOLUTION_TIER",
            "ENTITY_RESOLUTION_CONFIDENCE",
        },
        "supplier_dim",
    )


# ---------------------------------------------------------------------
# Frozen historical activity
# ---------------------------------------------------------------------

def build_history_context_base(
    con: duckdb.DuckDBPyConnection,
    award_supplier_map: Path,
    award_fact: Path,
    supplier_dim: Path,
) -> None:
    """
    Build one row per resolved historical award with CPV hierarchy fields.

    The award-supplier map supplies the frozen historical supplier entity.
    The award-fact table supplies the award CPV.
    """
    con.execute("DROP TABLE IF EXISTS history_award_fact_unique")
    con.execute(
        f"""
        CREATE TABLE history_award_fact_unique AS
        SELECT
            AWARD_KEY,
            MIN(CAST(CPV_CLEAN AS VARCHAR))
                AS CPV_CLEAN
        FROM read_parquet('{sql_path(award_fact)}')
        WHERE
            SOURCE_YEAR_MIN IN (2015, 2016)
            AND SOURCE_YEAR_MAX IN (2015, 2016)
        GROUP BY AWARD_KEY
        """
    )

    con.execute("DROP TABLE IF EXISTS history_context_awards")
    con.execute(
        f"""
        CREATE TABLE history_context_awards AS
        WITH joined AS (
            SELECT
                m.AWARD_KEY,
                m.SUPPLIER_ENTITY_ID,
                m.DT_DISPATCH_CLEAN AS HISTORY_PUBLICATION_DATE,
                f.CPV_CLEAN,
                REGEXP_REPLACE(
                    COALESCE(CAST(f.CPV_CLEAN AS VARCHAR), ''),
                    '[^0-9]',
                    '',
                    'g'
                ) AS CPV_DIGITS
            FROM read_parquet('{sql_path(award_supplier_map)}') m
            JOIN history_award_fact_unique f
              USING (AWARD_KEY)
            JOIN read_parquet('{sql_path(supplier_dim)}') d
              ON m.SUPPLIER_ENTITY_ID = d.SUPPLIER_ENTITY_ID
            WHERE
                m.SOURCE_YEAR_MIN IN (2015, 2016)
                AND m.SOURCE_YEAR_MAX IN (2015, 2016)
                AND m.SUPPLIER_ENTITY_ID IS NOT NULL
                AND m.DT_DISPATCH_CLEAN IS NOT NULL
        )
        SELECT
            AWARD_KEY,
            SUPPLIER_ENTITY_ID,
            HISTORY_PUBLICATION_DATE,
            CPV_CLEAN,
            CPV_DIGITS,

            CASE
                WHEN LENGTH(CPV_DIGITS) >= 2
                THEN SUBSTR(CPV_DIGITS, 1, 2)
                ELSE NULL
            END AS CPV2,

            CASE
                WHEN LENGTH(CPV_DIGITS) >= 3
                THEN SUBSTR(CPV_DIGITS, 1, 3)
                ELSE NULL
            END AS CPV3,

            CASE
                WHEN LENGTH(CPV_DIGITS) >= 4
                THEN SUBSTR(CPV_DIGITS, 1, 4)
                ELSE NULL
            END AS CPV4

        FROM joined
        """
    )


def build_temporal_audit(
    con: duckdb.DuckDBPyConnection,
    winner_history_map: Path,
) -> pd.DataFrame:
    con.execute("DROP TABLE IF EXISTS target_dates_2017")
    con.execute(
        f"""
        CREATE TABLE target_dates_2017 AS
        SELECT
            TARGET_CASE_ID,
            CAST(TARGET_DISPATCH_DATE AS DATE)
                AS TARGET_DISPATCH_DATE
        FROM read_parquet('{sql_path(winner_history_map)}')
        """
    )

    return fetch_df(
        con,
        """
        SELECT
            MIN(t.TARGET_DISPATCH_DATE)
                AS MIN_TARGET_DISPATCH_DATE,
            MAX(t.TARGET_DISPATCH_DATE)
                AS MAX_TARGET_DISPATCH_DATE,
            MIN(h.HISTORY_PUBLICATION_DATE)
                AS MIN_HISTORY_PUBLICATION_DATE,
            MAX(h.HISTORY_PUBLICATION_DATE)
                AS MAX_HISTORY_PUBLICATION_DATE,
            CASE
                WHEN MAX(h.HISTORY_PUBLICATION_DATE)
                     < MIN(t.TARGET_DISPATCH_DATE)
                THEN TRUE
                ELSE FALSE
            END AS HISTORY_WINDOW_STRICTLY_PRECEDES_TARGET_WINDOW
        FROM target_dates_2017 t
        CROSS JOIN history_context_awards h
        """
    )


# ---------------------------------------------------------------------
# Historical context eligibility
# ---------------------------------------------------------------------

def build_context_eligibility(
    con: duckdb.DuckDBPyConnection,
    supplier_dim: Path,
) -> None:
    """
    Build compact context-level supplier eligibility.

    The same historical supplier-context aggregate is reused for target cases
    sharing a CPV context, avoiding an unnecessary target-supplier explosion.
    """
    con.execute("DROP TABLE IF EXISTS cpv2_activity")
    con.execute(
        """
        CREATE TABLE cpv2_activity AS
        SELECT
            CPV2 AS CONTEXT_KEY,
            SUPPLIER_ENTITY_ID,
            COUNT(DISTINCT AWARD_KEY)
                AS N_HISTORY_AWARDS_IN_CONTEXT,
            MIN(HISTORY_PUBLICATION_DATE)
                AS FIRST_HISTORY_PUBLICATION_DATE,
            MAX(HISTORY_PUBLICATION_DATE)
                AS LAST_HISTORY_PUBLICATION_DATE
        FROM history_context_awards
        WHERE CPV2 IS NOT NULL
        GROUP BY
            CPV2,
            SUPPLIER_ENTITY_ID
        """
    )

    con.execute("DROP TABLE IF EXISTS cpv3_activity")
    con.execute(
        """
        CREATE TABLE cpv3_activity AS
        SELECT
            CPV3 AS CONTEXT_KEY,
            SUPPLIER_ENTITY_ID,
            COUNT(DISTINCT AWARD_KEY)
                AS N_HISTORY_AWARDS_IN_CONTEXT,
            MIN(HISTORY_PUBLICATION_DATE)
                AS FIRST_HISTORY_PUBLICATION_DATE,
            MAX(HISTORY_PUBLICATION_DATE)
                AS LAST_HISTORY_PUBLICATION_DATE
        FROM history_context_awards
        WHERE CPV3 IS NOT NULL
        GROUP BY
            CPV3,
            SUPPLIER_ENTITY_ID
        """
    )

    con.execute("DROP TABLE IF EXISTS cpv4_activity")
    con.execute(
        """
        CREATE TABLE cpv4_activity AS
        SELECT
            CPV4 AS CONTEXT_KEY,
            SUPPLIER_ENTITY_ID,
            COUNT(DISTINCT AWARD_KEY)
                AS N_HISTORY_AWARDS_IN_CONTEXT,
            MIN(HISTORY_PUBLICATION_DATE)
                AS FIRST_HISTORY_PUBLICATION_DATE,
            MAX(HISTORY_PUBLICATION_DATE)
                AS LAST_HISTORY_PUBLICATION_DATE
        FROM history_context_awards
        WHERE CPV4 IS NOT NULL
        GROUP BY
            CPV4,
            SUPPLIER_ENTITY_ID
        """
    )

    con.execute("DROP TABLE IF EXISTS historical_supplier_context_eligibility")
    con.execute(
        f"""
        CREATE TABLE historical_supplier_context_eligibility AS

        SELECT
            'CPV2_MIN1'::VARCHAR AS POOL_DEFINITION,
            'CPV2'::VARCHAR AS CONTEXT_LEVEL,
            a.CONTEXT_KEY,
            a.SUPPLIER_ENTITY_ID,
            a.N_HISTORY_AWARDS_IN_CONTEXT,
            a.FIRST_HISTORY_PUBLICATION_DATE,
            a.LAST_HISTORY_PUBLICATION_DATE,
            d.CANONICAL_NAME,
            d.SUPPLIER_COUNTRY,
            d.ENTITY_RESOLUTION_TIER,
            d.ENTITY_RESOLUTION_CONFIDENCE
        FROM cpv2_activity a
        JOIN read_parquet('{sql_path(supplier_dim)}') d
          USING (SUPPLIER_ENTITY_ID)
        WHERE a.N_HISTORY_AWARDS_IN_CONTEXT >= 1

        UNION ALL

        SELECT
            'CPV3_MIN1',
            'CPV3',
            a.CONTEXT_KEY,
            a.SUPPLIER_ENTITY_ID,
            a.N_HISTORY_AWARDS_IN_CONTEXT,
            a.FIRST_HISTORY_PUBLICATION_DATE,
            a.LAST_HISTORY_PUBLICATION_DATE,
            d.CANONICAL_NAME,
            d.SUPPLIER_COUNTRY,
            d.ENTITY_RESOLUTION_TIER,
            d.ENTITY_RESOLUTION_CONFIDENCE
        FROM cpv3_activity a
        JOIN read_parquet('{sql_path(supplier_dim)}') d
          USING (SUPPLIER_ENTITY_ID)
        WHERE a.N_HISTORY_AWARDS_IN_CONTEXT >= 1

        UNION ALL

        SELECT
            'CPV4_MIN1',
            'CPV4',
            a.CONTEXT_KEY,
            a.SUPPLIER_ENTITY_ID,
            a.N_HISTORY_AWARDS_IN_CONTEXT,
            a.FIRST_HISTORY_PUBLICATION_DATE,
            a.LAST_HISTORY_PUBLICATION_DATE,
            d.CANONICAL_NAME,
            d.SUPPLIER_COUNTRY,
            d.ENTITY_RESOLUTION_TIER,
            d.ENTITY_RESOLUTION_CONFIDENCE
        FROM cpv4_activity a
        JOIN read_parquet('{sql_path(supplier_dim)}') d
          USING (SUPPLIER_ENTITY_ID)
        WHERE a.N_HISTORY_AWARDS_IN_CONTEXT >= 1

        UNION ALL

        SELECT
            'CPV2_MIN2',
            'CPV2',
            a.CONTEXT_KEY,
            a.SUPPLIER_ENTITY_ID,
            a.N_HISTORY_AWARDS_IN_CONTEXT,
            a.FIRST_HISTORY_PUBLICATION_DATE,
            a.LAST_HISTORY_PUBLICATION_DATE,
            d.CANONICAL_NAME,
            d.SUPPLIER_COUNTRY,
            d.ENTITY_RESOLUTION_TIER,
            d.ENTITY_RESOLUTION_CONFIDENCE
        FROM cpv2_activity a
        JOIN read_parquet('{sql_path(supplier_dim)}') d
          USING (SUPPLIER_ENTITY_ID)
        WHERE a.N_HISTORY_AWARDS_IN_CONTEXT >= 2
        """
    )

    duplicate_count = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                POOL_DEFINITION,
                CONTEXT_KEY,
                SUPPLIER_ENTITY_ID
            FROM historical_supplier_context_eligibility
            GROUP BY
                POOL_DEFINITION,
                CONTEXT_KEY,
                SUPPLIER_ENTITY_ID
            HAVING COUNT(*) > 1
        )
        """,
    )

    if duplicate_count:
        raise ValueError(
            "Historical context eligibility contains duplicate "
            "(pool, context, supplier) rows."
        )


# ---------------------------------------------------------------------
# Target-to-context assignments
# ---------------------------------------------------------------------

def build_target_pool_assignments(
    con: duckdb.DuckDBPyConnection,
    winner_history_map: Path,
) -> None:
    con.execute("DROP TABLE IF EXISTS target_pool_base")
    con.execute(
        f"""
        CREATE TABLE target_pool_base AS
        WITH base AS (
            SELECT
                TARGET_CASE_ID,
                CAST(TARGET_DISPATCH_DATE AS DATE)
                    AS TARGET_DISPATCH_DATE,
                CAST(TARGET_CPV AS VARCHAR)
                    AS TARGET_CPV,
                TARGET_PROCUREMENT_COUNTRY,
                HISTORICAL_SUPPLIER_ENTITY_ID,
                COALESCE(
                    TRY_CAST(HISTORICAL_MATCHED AS BOOLEAN),
                    FALSE
                ) AS HISTORICAL_MATCHED,
                HISTORY_MATCH_STATUS,
                REGEXP_REPLACE(
                    COALESCE(CAST(TARGET_CPV AS VARCHAR), ''),
                    '[^0-9]',
                    '',
                    'g'
                ) AS TARGET_CPV_DIGITS
            FROM read_parquet('{sql_path(winner_history_map)}')
        )
        SELECT
            *,
            CASE
                WHEN LENGTH(TARGET_CPV_DIGITS) >= 2
                THEN SUBSTR(TARGET_CPV_DIGITS, 1, 2)
                ELSE NULL
            END AS TARGET_CPV2_DERIVED,

            CASE
                WHEN LENGTH(TARGET_CPV_DIGITS) >= 3
                THEN SUBSTR(TARGET_CPV_DIGITS, 1, 3)
                ELSE NULL
            END AS TARGET_CPV3_DERIVED,

            CASE
                WHEN LENGTH(TARGET_CPV_DIGITS) >= 4
                THEN SUBSTR(TARGET_CPV_DIGITS, 1, 4)
                ELSE NULL
            END AS TARGET_CPV4_DERIVED

        FROM base
        """
    )

    con.execute("DROP TABLE IF EXISTS target_pool_assignments")
    con.execute(
        """
        CREATE TABLE target_pool_assignments AS

        SELECT
            TARGET_CASE_ID,
            TARGET_DISPATCH_DATE,
            TARGET_CPV,
            TARGET_CPV2_DERIVED AS TARGET_CPV2,
            TARGET_PROCUREMENT_COUNTRY,
            HISTORICAL_SUPPLIER_ENTITY_ID,
            HISTORICAL_MATCHED,
            HISTORY_MATCH_STATUS,
            'CPV2_MIN1'::VARCHAR AS POOL_DEFINITION,
            'CPV2'::VARCHAR AS CONTEXT_LEVEL,
            TARGET_CPV2_DERIVED AS CONTEXT_KEY
        FROM target_pool_base

        UNION ALL

        SELECT
            TARGET_CASE_ID,
            TARGET_DISPATCH_DATE,
            TARGET_CPV,
            TARGET_CPV2_DERIVED,
            TARGET_PROCUREMENT_COUNTRY,
            HISTORICAL_SUPPLIER_ENTITY_ID,
            HISTORICAL_MATCHED,
            HISTORY_MATCH_STATUS,
            'CPV3_MIN1',
            'CPV3',
            TARGET_CPV3_DERIVED
        FROM target_pool_base

        UNION ALL

        SELECT
            TARGET_CASE_ID,
            TARGET_DISPATCH_DATE,
            TARGET_CPV,
            TARGET_CPV2_DERIVED,
            TARGET_PROCUREMENT_COUNTRY,
            HISTORICAL_SUPPLIER_ENTITY_ID,
            HISTORICAL_MATCHED,
            HISTORY_MATCH_STATUS,
            'CPV4_MIN1',
            'CPV4',
            TARGET_CPV4_DERIVED
        FROM target_pool_base

        UNION ALL

        SELECT
            TARGET_CASE_ID,
            TARGET_DISPATCH_DATE,
            TARGET_CPV,
            TARGET_CPV2_DERIVED,
            TARGET_PROCUREMENT_COUNTRY,
            HISTORICAL_SUPPLIER_ENTITY_ID,
            HISTORICAL_MATCHED,
            HISTORY_MATCH_STATUS,
            'CPV2_MIN2',
            'CPV2',
            TARGET_CPV2_DERIVED
        FROM target_pool_base
        """
    )

    duplicate_count = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                TARGET_CASE_ID,
                POOL_DEFINITION
            FROM target_pool_assignments
            GROUP BY
                TARGET_CASE_ID,
                POOL_DEFINITION
            HAVING COUNT(*) > 1
        )
        """,
    )

    if duplicate_count:
        raise ValueError(
            "Target pool assignments contain duplicate "
            "(target_case_id, pool_definition) rows."
        )


# ---------------------------------------------------------------------
# Case-level candidate pool coverage
# ---------------------------------------------------------------------

def build_case_pool_summary(
    con: duckdb.DuckDBPyConnection,
) -> None:
    con.execute("DROP TABLE IF EXISTS pool_size_by_context")
    con.execute(
        """
        CREATE TABLE pool_size_by_context AS
        SELECT
            POOL_DEFINITION,
            CONTEXT_LEVEL,
            CONTEXT_KEY,
            COUNT(DISTINCT SUPPLIER_ENTITY_ID)
                AS CANDIDATE_POOL_SIZE
        FROM historical_supplier_context_eligibility
        GROUP BY
            POOL_DEFINITION,
            CONTEXT_LEVEL,
            CONTEXT_KEY
        """
    )

    con.execute("DROP TABLE IF EXISTS candidate_pool_case_summary")
    con.execute(
        """
        CREATE TABLE candidate_pool_case_summary AS
        SELECT
            t.TARGET_CASE_ID,
            t.TARGET_DISPATCH_DATE,
            t.TARGET_CPV,
            t.TARGET_CPV2,
            t.TARGET_PROCUREMENT_COUNTRY,

            t.POOL_DEFINITION,
            t.CONTEXT_LEVEL,
            t.CONTEXT_KEY,

            COALESCE(s.CANDIDATE_POOL_SIZE, 0)
                AS CANDIDATE_POOL_SIZE,

            t.HISTORICAL_SUPPLIER_ENTITY_ID,
            t.HISTORICAL_MATCHED,
            t.HISTORY_MATCH_STATUS,

            CASE
                WHEN t.HISTORICAL_SUPPLIER_ENTITY_ID IS NULL
                    THEN FALSE
                WHEN e.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN TRUE
                ELSE FALSE
            END AS OBSERVED_WINNER_INCLUDED,

            CASE
                WHEN COALESCE(s.CANDIDATE_POOL_SIZE, 0) = 0
                    THEN TRUE
                ELSE FALSE
            END AS EMPTY_POOL,

            e.N_HISTORY_AWARDS_IN_CONTEXT
                AS WINNER_HISTORY_AWARDS_IN_POOL_CONTEXT

        FROM target_pool_assignments t

        LEFT JOIN pool_size_by_context s
          ON t.POOL_DEFINITION = s.POOL_DEFINITION
         AND t.CONTEXT_LEVEL = s.CONTEXT_LEVEL
         AND t.CONTEXT_KEY = s.CONTEXT_KEY

        LEFT JOIN historical_supplier_context_eligibility e
          ON t.POOL_DEFINITION = e.POOL_DEFINITION
         AND t.CONTEXT_LEVEL = e.CONTEXT_LEVEL
         AND t.CONTEXT_KEY = e.CONTEXT_KEY
         AND t.HISTORICAL_SUPPLIER_ENTITY_ID = e.SUPPLIER_ENTITY_ID
        """
    )


# ---------------------------------------------------------------------
# Optional full membership materialization
# ---------------------------------------------------------------------

def materialize_case_supplier_membership(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Potentially very large. Run only when explicitly requested.
    """
    con.execute("DROP TABLE IF EXISTS candidate_pools_2017")
    con.execute(
        """
        CREATE TABLE candidate_pools_2017 AS
        SELECT
            t.TARGET_CASE_ID,
            t.TARGET_DISPATCH_DATE,
            t.TARGET_CPV,
            t.TARGET_CPV2,
            t.TARGET_PROCUREMENT_COUNTRY,

            t.POOL_DEFINITION,
            t.CONTEXT_LEVEL,
            t.CONTEXT_KEY,

            e.SUPPLIER_ENTITY_ID,
            e.CANONICAL_NAME,
            e.SUPPLIER_COUNTRY,
            e.ENTITY_RESOLUTION_TIER,
            e.ENTITY_RESOLUTION_CONFIDENCE,

            e.N_HISTORY_AWARDS_IN_CONTEXT,
            e.FIRST_HISTORY_PUBLICATION_DATE,
            e.LAST_HISTORY_PUBLICATION_DATE,

            CASE
                WHEN t.HISTORICAL_SUPPLIER_ENTITY_ID
                     = e.SUPPLIER_ENTITY_ID
                THEN TRUE
                ELSE FALSE
            END AS IS_OBSERVED_WINNER

        FROM target_pool_assignments t
        JOIN historical_supplier_context_eligibility e
          ON t.POOL_DEFINITION = e.POOL_DEFINITION
         AND t.CONTEXT_LEVEL = e.CONTEXT_LEVEL
         AND t.CONTEXT_KEY = e.CONTEXT_KEY
        """
    )


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def coverage_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            POOL_DEFINITION,

            COUNT(*) AS N_TARGET_CASES,

            COUNT(*) FILTER (
                WHERE HISTORICAL_MATCHED
            ) AS N_HISTORICALLY_MATCHED_WINNERS,

            COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            ) AS N_CASES_WINNER_INCLUDED,

            100.0 * COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            ) / COUNT(*)
                AS ALL_CASE_COVERAGE_PCT,

            100.0 * COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE HISTORICAL_MATCHED
                ),
                0
            ) AS CONDITIONAL_HISTORICAL_COVERAGE_PCT,

            COUNT(*) FILTER (
                WHERE EMPTY_POOL
            ) AS N_EMPTY_POOLS,

            100.0 * COUNT(*) FILTER (
                WHERE EMPTY_POOL
            ) / COUNT(*)
                AS EMPTY_POOL_RATE_PCT

        FROM candidate_pool_case_summary
        GROUP BY POOL_DEFINITION
        ORDER BY
            CASE POOL_DEFINITION
                WHEN 'CPV2_MIN1' THEN 1
                WHEN 'CPV3_MIN1' THEN 2
                WHEN 'CPV4_MIN1' THEN 3
                WHEN 'CPV2_MIN2' THEN 4
                ELSE 99
            END
        """
    )


def pool_size_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            POOL_DEFINITION,
            COUNT(*) AS N_TARGET_CASES,
            MIN(CANDIDATE_POOL_SIZE)
                AS MIN_POOL_SIZE,
            AVG(CANDIDATE_POOL_SIZE)
                AS MEAN_POOL_SIZE,
            MEDIAN(CANDIDATE_POOL_SIZE)
                AS MEDIAN_POOL_SIZE,
            QUANTILE_CONT(CANDIDATE_POOL_SIZE, 0.95)
                AS P95_POOL_SIZE,
            QUANTILE_CONT(CANDIDATE_POOL_SIZE, 0.99)
                AS P99_POOL_SIZE,
            MAX(CANDIDATE_POOL_SIZE)
                AS MAX_POOL_SIZE
        FROM candidate_pool_case_summary
        GROUP BY POOL_DEFINITION
        ORDER BY
            CASE POOL_DEFINITION
                WHEN 'CPV2_MIN1' THEN 1
                WHEN 'CPV3_MIN1' THEN 2
                WHEN 'CPV4_MIN1' THEN 3
                WHEN 'CPV2_MIN2' THEN 4
                ELSE 99
            END
        """
    )


def coverage_by_country(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            POOL_DEFINITION,
            COALESCE(
                TARGET_PROCUREMENT_COUNTRY,
                '<MISSING>'
            ) AS TARGET_PROCUREMENT_COUNTRY,

            COUNT(*) AS N_TARGET_CASES,

            COUNT(*) FILTER (
                WHERE HISTORICAL_MATCHED
            ) AS N_HISTORICALLY_MATCHED_WINNERS,

            COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            ) AS N_CASES_WINNER_INCLUDED,

            100.0 * COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            ) / COUNT(*)
                AS ALL_CASE_COVERAGE_PCT,

            100.0 * COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE HISTORICAL_MATCHED
                ),
                0
            ) AS CONDITIONAL_HISTORICAL_COVERAGE_PCT,

            MEDIAN(CANDIDATE_POOL_SIZE)
                AS MEDIAN_POOL_SIZE

        FROM candidate_pool_case_summary
        GROUP BY
            POOL_DEFINITION,
            TARGET_PROCUREMENT_COUNTRY
        ORDER BY
            POOL_DEFINITION,
            N_TARGET_CASES DESC
        """
    )


def coverage_by_cpv2(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            POOL_DEFINITION,
            COALESCE(TARGET_CPV2, '<MISSING>')
                AS TARGET_CPV2,

            COUNT(*) AS N_TARGET_CASES,

            COUNT(*) FILTER (
                WHERE HISTORICAL_MATCHED
            ) AS N_HISTORICALLY_MATCHED_WINNERS,

            COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            ) AS N_CASES_WINNER_INCLUDED,

            100.0 * COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            ) / COUNT(*)
                AS ALL_CASE_COVERAGE_PCT,

            100.0 * COUNT(*) FILTER (
                WHERE OBSERVED_WINNER_INCLUDED
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE HISTORICAL_MATCHED
                ),
                0
            ) AS CONDITIONAL_HISTORICAL_COVERAGE_PCT,

            MEDIAN(CANDIDATE_POOL_SIZE)
                AS MEDIAN_POOL_SIZE

        FROM candidate_pool_case_summary
        GROUP BY
            POOL_DEFINITION,
            TARGET_CPV2
        ORDER BY
            POOL_DEFINITION,
            N_TARGET_CASES DESC
        """
    )


def eligibility_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            POOL_DEFINITION,
            COUNT(DISTINCT CONTEXT_KEY)
                AS N_CONTEXTS,
            COUNT(DISTINCT SUPPLIER_ENTITY_ID)
                AS N_DISTINCT_ELIGIBLE_SUPPLIERS,
            COUNT(*) AS N_SUPPLIER_CONTEXT_MEMBERSHIPS,
            MEDIAN(N_HISTORY_AWARDS_IN_CONTEXT)
                AS MEDIAN_HISTORY_AWARDS_PER_MEMBERSHIP,
            QUANTILE_CONT(
                N_HISTORY_AWARDS_IN_CONTEXT,
                0.95
            ) AS P95_HISTORY_AWARDS_PER_MEMBERSHIP
        FROM historical_supplier_context_eligibility
        GROUP BY POOL_DEFINITION
        ORDER BY
            CASE POOL_DEFINITION
                WHEN 'CPV2_MIN1' THEN 1
                WHEN 'CPV3_MIN1' THEN 2
                WHEN 'CPV4_MIN1' THEN 3
                WHEN 'CPV2_MIN2' THEN 4
                ELSE 99
            END
        """
    )


def build_metrics(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(metric: str, value, note: str = "") -> None:
        rows.append(
            {
                "metric": metric,
                "value": value,
                "note": note,
            }
        )

    n_targets = scalar(
        con,
        """
        SELECT COUNT(DISTINCT TARGET_CASE_ID)
        FROM target_pool_assignments
        """,
    )
    n_history_awards = scalar(
        con,
        "SELECT COUNT(*) FROM history_context_awards",
    )
    n_history_suppliers = scalar(
        con,
        """
        SELECT COUNT(DISTINCT SUPPLIER_ENTITY_ID)
        FROM history_context_awards
        """,
    )

    add("strict_target_cases", n_targets)
    add("history_context_awards", n_history_awards)
    add("historical_suppliers_with_valid_cpv_activity", n_history_suppliers)

    coverage_df = coverage_summary(con)

    for _, row in coverage_df.iterrows():
        pool = row["POOL_DEFINITION"]
        add(
            f"{pool}__winner_included_cases",
            int(row["N_CASES_WINNER_INCLUDED"]),
        )
        add(
            f"{pool}__all_case_coverage_pct",
            float(row["ALL_CASE_COVERAGE_PCT"]),
        )
        add(
            f"{pool}__conditional_historical_coverage_pct",
            (
                float(row["CONDITIONAL_HISTORICAL_COVERAGE_PCT"])
                if pd.notna(
                    row["CONDITIONAL_HISTORICAL_COVERAGE_PCT"]
                )
                else None
            ),
        )
        add(
            f"{pool}__empty_pool_rate_pct",
            float(row["EMPTY_POOL_RATE_PCT"]),
        )

    return pd.DataFrame(rows)


def create_report(
    coverage_df: pd.DataFrame,
    pool_size_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    output_path: Path,
) -> None:
    temporal = temporal_df.iloc[0]

    lines = [
        "# TED 2017 Historical Candidate-Pool Build Report",
        "",
        "## Scope",
        "",
        "Candidate pools are constructed from the frozen 2015-2016 historical "
        "supplier universe and applied to strict 2017 target procurement cases.",
        "",
        "The pools are historically active supplier comparison sets, not actual "
        "bidder sets.",
        "",
        "## Temporal audit",
        "",
        f"- Earliest target dispatch date: {temporal['MIN_TARGET_DISPATCH_DATE']}",
        f"- Latest target dispatch date: {temporal['MAX_TARGET_DISPATCH_DATE']}",
        f"- Earliest historical publication date: {temporal['MIN_HISTORY_PUBLICATION_DATE']}",
        f"- Latest historical publication date: {temporal['MAX_HISTORY_PUBLICATION_DATE']}",
        f"- Historical window strictly precedes target window: "
        f"{bool(temporal['HISTORY_WINDOW_STRICTLY_PRECEDES_TARGET_WINDOW'])}",
        "",
        "## Candidate-pool coverage",
        "",
        "| Pool | Cases | Winner included | All-case coverage (%) | "
        "Conditional historical coverage (%) | Empty-pool rate (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in coverage_df.iterrows():
        conditional = row["CONDITIONAL_HISTORICAL_COVERAGE_PCT"]
        conditional_text = (
            f"{float(conditional):.3f}"
            if pd.notna(conditional)
            else "NA"
        )
        lines.append(
            f"| `{row['POOL_DEFINITION']}` "
            f"| {int(row['N_TARGET_CASES'])} "
            f"| {int(row['N_CASES_WINNER_INCLUDED'])} "
            f"| {float(row['ALL_CASE_COVERAGE_PCT']):.3f} "
            f"| {conditional_text} "
            f"| {float(row['EMPTY_POOL_RATE_PCT']):.3f} |"
        )

    lines.extend([
        "",
        "## Candidate-pool size",
        "",
        "| Pool | Mean | Median | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for _, row in pool_size_df.iterrows():
        lines.append(
            f"| `{row['POOL_DEFINITION']}` "
            f"| {float(row['MEAN_POOL_SIZE']):.1f} "
            f"| {float(row['MEDIAN_POOL_SIZE']):.1f} "
            f"| {float(row['P95_POOL_SIZE']):.1f} "
            f"| {float(row['P99_POOL_SIZE']):.1f} "
            f"| {int(row['MAX_POOL_SIZE'])} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "`ALL_CASE_COVERAGE_PCT` uses all strict target cases as the denominator.",
        "",
        "`CONDITIONAL_HISTORICAL_COVERAGE_PCT` uses only target cases whose observed "
        "winner was already matched somewhere in the frozen 2015-2016 historical "
        "supplier universe.",
        "",
        "The primary pool is `CPV2_MIN1`. The other three definitions are "
        "sensitivity analyses.",
        "",
        "## Next step",
        "",
        "Construct the target-supplier historical feature matrix for the selected "
        "primary and sensitivity pool definitions, then begin deterministic "
        "baseline ranking and scenario experiments.",
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
        description=(
            "Build CPV-contextual historical supplier candidate pools "
            "for strict 2017 target cases."
        )
    )

    parser.add_argument(
        "--winner-history-map",
        required=True,
        type=Path,
        help="Path to winner_history_map_2017.parquet",
    )
    parser.add_argument(
        "--award-supplier-map",
        required=True,
        type=Path,
        help="Path to award_supplier_map_2015_2016.parquet",
    )
    parser.add_argument(
        "--award-fact",
        required=True,
        type=Path,
        help="Path to award_fact_2015_2017.parquet",
    )
    parser.add_argument(
        "--supplier-dim",
        required=True,
        type=Path,
        help="Path to supplier_dim_2015_2016.parquet",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/candidate_pools_2017"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/candidate_pools_2017"),
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
        "--allow-overlap-history-target",
        action="store_true",
        help=(
            "Allow the historical publication window to overlap the target "
            "dispatch window. Not recommended for the current design."
        ),
    )
    parser.add_argument(
        "--materialize-case-supplier-membership",
        action="store_true",
        help=(
            "Materialize the full long target-case x supplier candidate table. "
            "This can be very large and is disabled by default."
        ),
    )

    args = parser.parse_args()

    for path in [
        args.winner_history_map,
        args.award_supplier_map,
        args.award_fact,
        args.supplier_dim,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    work_db = (
        args.work_db
        if args.work_db is not None
        else args.output_dir / "_candidate_pool_build.duckdb"
    )
    work_db.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(work_db))
    con.execute(f"SET threads={int(args.threads)}")

    if args.memory_limit:
        mem = args.memory_limit.replace("'", "''")
        con.execute(f"SET memory_limit='{mem}'")

    temp_dir = args.output_dir / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"SET temp_directory='{sql_path(temp_dir)}'"
    )

    try:
        print("1/9 Validating inputs...")
        validate_inputs(
            con,
            args.winner_history_map,
            args.award_supplier_map,
            args.award_fact,
            args.supplier_dim,
        )

        print("2/9 Building frozen historical CPV activity...")
        build_history_context_base(
            con,
            args.award_supplier_map,
            args.award_fact,
            args.supplier_dim,
        )

        print("3/9 Auditing temporal separation...")
        temporal_df = build_temporal_audit(
            con,
            args.winner_history_map,
        )

        temporal_safe = bool(
            temporal_df.iloc[0][
                "HISTORY_WINDOW_STRICTLY_PRECEDES_TARGET_WINDOW"
            ]
        )

        if not temporal_safe and not args.allow_overlap_history_target:
            raise ValueError(
                "Historical publication window overlaps the target dispatch "
                "window. The compact context-pool representation would not be "
                "temporally safe. Re-run only after implementing target-date "
                "specific eligibility, or pass "
                "--allow-overlap-history-target after explicit review."
            )

        print("4/9 Building historical supplier-context eligibility...")
        build_context_eligibility(
            con,
            args.supplier_dim,
        )

        print("5/9 Assigning target cases to pool contexts...")
        build_target_pool_assignments(
            con,
            args.winner_history_map,
        )

        print("6/9 Computing candidate coverage and pool sizes...")
        build_case_pool_summary(con)

        if args.materialize_case_supplier_membership:
            print(
                "7/9 Materializing full case-supplier membership "
                "(this may be large)..."
            )
            materialize_case_supplier_membership(con)
        else:
            print(
                "7/9 Skipping full case-supplier materialization "
                "(compact representation retained)..."
            )

        print("8/9 Writing reports...")
        coverage_df = coverage_summary(con)
        pool_size_df = pool_size_summary(con)
        country_df = coverage_by_country(con)
        cpv2_df = coverage_by_cpv2(con)
        eligibility_df = eligibility_summary(con)
        metrics_df = build_metrics(con)

        write_csv(
            coverage_df,
            args.report_dir / "candidate_pool_coverage_summary.csv",
        )
        write_csv(
            pool_size_df,
            args.report_dir / "candidate_pool_size_summary.csv",
        )
        write_csv(
            country_df,
            args.report_dir / "candidate_pool_coverage_by_country.csv",
        )
        write_csv(
            cpv2_df,
            args.report_dir / "candidate_pool_coverage_by_cpv2.csv",
        )
        write_csv(
            eligibility_df,
            args.report_dir / "historical_context_eligibility_summary.csv",
        )
        write_csv(
            temporal_df,
            args.report_dir / "candidate_pool_temporal_audit.csv",
        )
        write_csv(
            metrics_df,
            args.report_dir / "candidate_pool_build_metrics.csv",
        )

        create_report(
            coverage_df,
            pool_size_df,
            temporal_df,
            args.report_dir / "candidate_pool_build_report.md",
        )

        print("9/9 Writing processed Parquet outputs...")
        copy_table_to_parquet(
            con,
            "historical_supplier_context_eligibility",
            args.output_dir
            / "historical_supplier_context_eligibility_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "target_pool_assignments",
            args.output_dir
            / "target_pool_assignments_2017.parquet",
        )
        copy_table_to_parquet(
            con,
            "candidate_pool_case_summary",
            args.output_dir
            / "candidate_pool_case_summary_2017.parquet",
        )
        copy_table_to_parquet(
            con,
            "pool_size_by_context",
            args.output_dir
            / "candidate_pool_context_sizes_2017.parquet",
        )

        if args.materialize_case_supplier_membership:
            copy_table_to_parquet(
                con,
                "candidate_pools_2017",
                args.output_dir
                / "candidate_pools_2017.parquet",
            )

        print()
        print("=" * 78)
        print("Candidate-pool build complete")
        print("=" * 78)
        print(f"Processed outputs: {args.output_dir.resolve()}")
        print(f"Reports:           {args.report_dir.resolve()}")
        print(f"Work database:     {work_db.resolve()}")
        print()
        print("Primary reports to review:")
        print("  candidate_pool_build_report.md")
        print("  candidate_pool_coverage_summary.csv")
        print("  candidate_pool_size_summary.csv")
        print("  candidate_pool_temporal_audit.csv")
        print("  candidate_pool_coverage_by_country.csv")
        print("  candidate_pool_coverage_by_cpv2.csv")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
