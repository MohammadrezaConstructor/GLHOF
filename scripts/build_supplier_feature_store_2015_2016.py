#!/usr/bin/env python3
"""
Build the compact historical supplier feature store for the 2015-2016 history
window and the 2017 target-case index.

This is the second and final main preprocessing step.

Inputs
------
- award_supplier_map_2015_2016.parquet
- award_fact_2015_2017.parquet
- supplier_dim_2015_2016.parquet
- winner_history_map_2017.parquet

Outputs
-------
Processed:
- supplier_global_features_2015_2016.parquet
- supplier_cpv2_features_2015_2016.parquet
- supplier_cpv3_features_2015_2016.parquet
- supplier_cpv4_features_2015_2016.parquet
- target_case_features_2017.parquet

Reports:
- feature_dictionary.csv
- feature_build_metrics.csv
- feature_coverage_summary.csv
- context_feature_summary.csv
- feature_temporal_audit.csv
- feature_build_report.md

Design principles
-----------------
1. Historical features use only frozen 2015-2016 records.
2. No 2017 outcome is used to construct supplier features.
3. Features are stored compactly at supplier-global and supplier-context grain.
4. Target-relative recency can later be computed as:
       TARGET_DISPATCH_DATE - LAST_HISTORY_AWARD_DATE
   without materializing hundreds of millions of case-supplier rows.
5. Buyer HHI is an observed buyer-concentration measure, not a resilience score.
6. NUMBER_OFFERS is treated as procurement competition context, not supplier quality.
7. Public procurement award history is observational; these features do not imply
   private performance, unsuccessful bidder participation, or causal supplier quality.

Requires
--------
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


def first_existing(
    available: Iterable[str],
    candidates: list[str],
) -> str | None:
    available_set = set(available)
    for candidate in candidates:
        if candidate in available_set:
            return candidate
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
# Input validation and dynamic column resolution
# ---------------------------------------------------------------------

def validate_inputs(
    con: duckdb.DuckDBPyConnection,
    award_supplier_map: Path,
    award_fact: Path,
    supplier_dim: Path,
    winner_history_map: Path,
) -> dict[str, str | None]:
    require_columns(
        con,
        award_supplier_map,
        {
            "AWARD_KEY",
            "SOURCE_YEAR_MIN",
            "SOURCE_YEAR_MAX",
            "DT_AWARD_CLEAN",
            "DT_DISPATCH_CLEAN",
            "SUPPLIER_ENTITY_ID",
            "SUPPLIER_COUNTRY_NORM",
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
            "CAE_NAME_CLEAN",
            "ISO_COUNTRY_CODE_CLEAN",
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
            "FIRST_OBSERVED_AWARD_DATE",
            "LAST_OBSERVED_AWARD_DATE",
            "FIRST_PUBLICATION_DATE",
            "LAST_PUBLICATION_DATE",
            "SOURCE_AWARD_COUNT",
        },
        "supplier_dim",
    )

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

    award_fact_cols = parquet_columns(con, award_fact)

    value_primary = first_existing(
        award_fact_cols,
        [
            "AWARD_VALUE_EURO_CLEAN",
            "AWARD_VALUE_EURO_FIN_1_CLEAN",
            "AWARD_EST_VALUE_EURO_CLEAN",
        ],
    )

    value_fallback = None
    if value_primary == "AWARD_VALUE_EURO_CLEAN":
        value_fallback = first_existing(
            award_fact_cols,
            [
                "AWARD_VALUE_EURO_FIN_1_CLEAN",
                "AWARD_EST_VALUE_EURO_CLEAN",
            ],
        )
    elif value_primary == "AWARD_VALUE_EURO_FIN_1_CLEAN":
        value_fallback = first_existing(
            award_fact_cols,
            ["AWARD_EST_VALUE_EURO_CLEAN"],
        )

    number_offers = first_existing(
        award_fact_cols,
        [
            "NUMBER_OFFERS_CLEAN",
            "NUMBER_OFFERS",
        ],
    )

    return {
        "value_primary": value_primary,
        "value_fallback": value_fallback,
        "number_offers": number_offers,
    }


# ---------------------------------------------------------------------
# Historical analytic award base
# ---------------------------------------------------------------------

def build_historical_award_base(
    con: duckdb.DuckDBPyConnection,
    award_supplier_map: Path,
    award_fact: Path,
    supplier_dim: Path,
    resolved_cols: dict[str, str | None],
) -> None:
    """
    Build one analytical row per resolved historical award.

    Value selection:
      preferred field first, then fallback if available.
    """

    value_primary = resolved_cols["value_primary"]
    value_fallback = resolved_cols["value_fallback"]
    number_offers = resolved_cols["number_offers"]

    if value_primary is None:
        value_expr = "NULL::DOUBLE"
        value_source_expr = "'unavailable'::VARCHAR"
    elif value_fallback is None:
        value_expr = (
            f"TRY_CAST(f.{qident(value_primary)} AS DOUBLE)"
        )
        value_source_expr = (
            f"CASE WHEN TRY_CAST(f.{qident(value_primary)} AS DOUBLE) IS NOT NULL "
            f"THEN '{value_primary}' ELSE 'missing' END"
        )
    else:
        value_expr = (
            f"COALESCE("
            f"TRY_CAST(f.{qident(value_primary)} AS DOUBLE), "
            f"TRY_CAST(f.{qident(value_fallback)} AS DOUBLE)"
            f")"
        )
        value_source_expr = (
            f"CASE "
            f"WHEN TRY_CAST(f.{qident(value_primary)} AS DOUBLE) IS NOT NULL "
            f"THEN '{value_primary}' "
            f"WHEN TRY_CAST(f.{qident(value_fallback)} AS DOUBLE) IS NOT NULL "
            f"THEN '{value_fallback}' "
            f"ELSE 'missing' END"
        )

    if number_offers is None:
        number_offers_expr = "NULL::DOUBLE"
    else:
        number_offers_expr = (
            f"TRY_CAST(f.{qident(number_offers)} AS DOUBLE)"
        )

    con.execute("DROP TABLE IF EXISTS historical_award_fact_unique")
    con.execute(
        f"""
        CREATE TABLE historical_award_fact_unique AS
        SELECT
            AWARD_KEY,
            MIN(CAST(CPV_CLEAN AS VARCHAR))
                AS CPV_CLEAN,
            MIN(CAST(CAE_NAME_CLEAN AS VARCHAR))
                AS CAE_NAME_CLEAN,
            MIN(CAST(ISO_COUNTRY_CODE_CLEAN AS VARCHAR))
                AS ISO_COUNTRY_CODE_CLEAN,

            MIN({value_expr})
                AS AWARD_VALUE_EURO_PREFERRED,

            MIN({value_source_expr})
                AS AWARD_VALUE_SOURCE,

            MIN({number_offers_expr})
                AS NUMBER_OFFERS_ANALYTIC

        FROM read_parquet('{sql_path(award_fact)}') f
        WHERE
            SOURCE_YEAR_MIN IN (2015, 2016)
            AND SOURCE_YEAR_MAX IN (2015, 2016)
        GROUP BY AWARD_KEY
        """
    )

    con.execute("DROP TABLE IF EXISTS historical_award_analytic")
    con.execute(
        f"""
        CREATE TABLE historical_award_analytic AS
        WITH joined AS (
            SELECT
                m.AWARD_KEY,
                m.SUPPLIER_ENTITY_ID,
                m.SUPPLIER_COUNTRY_NORM,
                CAST(m.DT_AWARD_CLEAN AS DATE) AS HISTORY_AWARD_DATE,
                CAST(m.DT_DISPATCH_CLEAN AS DATE) AS HISTORY_PUBLICATION_DATE,
                m.AWARD_RESOLUTION_STATUS,

                f.CPV_CLEAN,
                f.CAE_NAME_CLEAN,
                f.ISO_COUNTRY_CODE_CLEAN,
                f.AWARD_VALUE_EURO_PREFERRED,
                f.AWARD_VALUE_SOURCE,
                f.NUMBER_OFFERS_ANALYTIC,

                REGEXP_REPLACE(
                    COALESCE(CAST(f.CPV_CLEAN AS VARCHAR), ''),
                    '[^0-9]',
                    '',
                    'g'
                ) AS CPV_DIGITS,

                NULLIF(
                    TRIM(
                        REGEXP_REPLACE(
                            LOWER(COALESCE(f.CAE_NAME_CLEAN, '')),
                            '[^[:alnum:]]+',
                            ' ',
                            'g'
                        )
                    ),
                    ''
                ) AS BUYER_NAME_NORM,

                UPPER(
                    NULLIF(
                        TRIM(CAST(f.ISO_COUNTRY_CODE_CLEAN AS VARCHAR)),
                        ''
                    )
                ) AS BUYER_COUNTRY_NORM

            FROM read_parquet('{sql_path(award_supplier_map)}') m

            JOIN historical_award_fact_unique f
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
            *,

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
            END AS CPV4,

            CASE
                WHEN BUYER_NAME_NORM IS NOT NULL
                 AND BUYER_COUNTRY_NORM IS NOT NULL
                THEN BUYER_COUNTRY_NORM || '||' || BUYER_NAME_NORM
                WHEN BUYER_NAME_NORM IS NOT NULL
                THEN 'UNKNOWN_COUNTRY||' || BUYER_NAME_NORM
                ELSE NULL
            END AS BUYER_KEY,

            CASE
                WHEN LENGTH(COALESCE(SUPPLIER_COUNTRY_NORM, '')) = 2
                 AND LENGTH(COALESCE(BUYER_COUNTRY_NORM, '')) = 2
                THEN TRUE
                ELSE FALSE
            END AS IS_CROSS_BORDER_COMPARABLE,

            CASE
                WHEN LENGTH(COALESCE(SUPPLIER_COUNTRY_NORM, '')) = 2
                 AND LENGTH(COALESCE(BUYER_COUNTRY_NORM, '')) = 2
                 AND SUPPLIER_COUNTRY_NORM <> BUYER_COUNTRY_NORM
                THEN 1
                WHEN LENGTH(COALESCE(SUPPLIER_COUNTRY_NORM, '')) = 2
                 AND LENGTH(COALESCE(BUYER_COUNTRY_NORM, '')) = 2
                THEN 0
                ELSE NULL
            END AS CROSS_BORDER_FLAG

        FROM joined
        """
    )

    duplicate_awards = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT AWARD_KEY
            FROM historical_award_analytic
            GROUP BY AWARD_KEY
            HAVING COUNT(*) > 1
        )
        """,
    )

    if duplicate_awards:
        raise ValueError(
            f"Historical analytic base has {duplicate_awards} duplicated AWARD_KEY values."
        )


# ---------------------------------------------------------------------
# Global supplier features
# ---------------------------------------------------------------------

def build_global_supplier_features(
    con: duckdb.DuckDBPyConnection,
    supplier_dim: Path,
) -> None:
    con.execute("DROP TABLE IF EXISTS global_buyer_counts")
    con.execute(
        """
        CREATE TABLE global_buyer_counts AS
        SELECT
            SUPPLIER_ENTITY_ID,
            BUYER_KEY,
            COUNT(DISTINCT AWARD_KEY) AS BUYER_AWARD_COUNT
        FROM historical_award_analytic
        WHERE BUYER_KEY IS NOT NULL
        GROUP BY
            SUPPLIER_ENTITY_ID,
            BUYER_KEY
        """
    )

    con.execute("DROP TABLE IF EXISTS global_buyer_hhi")
    con.execute(
        """
        CREATE TABLE global_buyer_hhi AS
        WITH totals AS (
            SELECT
                SUPPLIER_ENTITY_ID,
                SUM(BUYER_AWARD_COUNT) AS TOTAL_IDENTIFIED_BUYER_AWARDS
            FROM global_buyer_counts
            GROUP BY SUPPLIER_ENTITY_ID
        )
        SELECT
            b.SUPPLIER_ENTITY_ID,
            COUNT(*) AS N_UNIQUE_BUYERS,
            t.TOTAL_IDENTIFIED_BUYER_AWARDS
                AS N_AWARDS_WITH_IDENTIFIED_BUYER,
            SUM(
                POWER(
                    CAST(b.BUYER_AWARD_COUNT AS DOUBLE)
                    / NULLIF(t.TOTAL_IDENTIFIED_BUYER_AWARDS, 0),
                    2
                )
            ) AS BUYER_HHI
        FROM global_buyer_counts b
        JOIN totals t
          USING (SUPPLIER_ENTITY_ID)
        GROUP BY
            b.SUPPLIER_ENTITY_ID,
            t.TOTAL_IDENTIFIED_BUYER_AWARDS
        """
    )

    con.execute("DROP TABLE IF EXISTS global_supplier_aggregate")
    con.execute(
        """
        CREATE TABLE global_supplier_aggregate AS
        SELECT
            SUPPLIER_ENTITY_ID,

            COUNT(DISTINCT AWARD_KEY)
                AS N_HISTORICAL_AWARDS,

            COUNT(DISTINCT AWARD_KEY) FILTER (
                WHERE AWARD_VALUE_EURO_PREFERRED IS NOT NULL
            ) AS N_AWARDS_WITH_VALUE,

            SUM(AWARD_VALUE_EURO_PREFERRED)
                AS TOTAL_AWARD_VALUE_EURO,

            MEDIAN(AWARD_VALUE_EURO_PREFERRED)
                AS MEDIAN_AWARD_VALUE_EURO,

            AVG(AWARD_VALUE_EURO_PREFERRED)
                AS MEAN_AWARD_VALUE_EURO,

            MIN(HISTORY_AWARD_DATE)
                AS FIRST_HISTORY_AWARD_DATE,

            MAX(HISTORY_AWARD_DATE)
                AS LAST_HISTORY_AWARD_DATE,

            MIN(HISTORY_PUBLICATION_DATE)
                AS FIRST_HISTORY_PUBLICATION_DATE,

            MAX(HISTORY_PUBLICATION_DATE)
                AS LAST_HISTORY_PUBLICATION_DATE,

            COUNT(DISTINCT CPV2) FILTER (
                WHERE CPV2 IS NOT NULL
            ) AS CPV2_BREADTH,

            COUNT(DISTINCT CPV3) FILTER (
                WHERE CPV3 IS NOT NULL
            ) AS CPV3_BREADTH,

            COUNT(DISTINCT CPV4) FILTER (
                WHERE CPV4 IS NOT NULL
            ) AS CPV4_BREADTH,

            COUNT(DISTINCT AWARD_KEY) FILTER (
                WHERE NUMBER_OFFERS_ANALYTIC IS NOT NULL
            ) AS N_AWARDS_WITH_NUMBER_OFFERS,

            MEDIAN(NUMBER_OFFERS_ANALYTIC)
                AS MEDIAN_NUMBER_OFFERS,

            COUNT(DISTINCT AWARD_KEY) FILTER (
                WHERE IS_CROSS_BORDER_COMPARABLE
            ) AS N_CROSS_BORDER_COMPARABLE_AWARDS,

            AVG(CAST(CROSS_BORDER_FLAG AS DOUBLE))
                AS CROSS_BORDER_AWARD_SHARE

        FROM historical_award_analytic
        GROUP BY SUPPLIER_ENTITY_ID
        """
    )

    con.execute("DROP TABLE IF EXISTS supplier_global_features")
    con.execute(
        f"""
        CREATE TABLE supplier_global_features AS
        SELECT
            d.SUPPLIER_ENTITY_ID,
            d.CANONICAL_NAME,
            d.SUPPLIER_COUNTRY,
            d.ENTITY_RESOLUTION_TIER,
            d.ENTITY_RESOLUTION_STATUS,
            d.ENTITY_RESOLUTION_CONFIDENCE,

            g.N_HISTORICAL_AWARDS,
            g.N_AWARDS_WITH_VALUE,
            g.TOTAL_AWARD_VALUE_EURO,
            g.MEDIAN_AWARD_VALUE_EURO,
            g.MEAN_AWARD_VALUE_EURO,

            COALESCE(h.N_UNIQUE_BUYERS, 0)
                AS N_UNIQUE_BUYERS,
            h.N_AWARDS_WITH_IDENTIFIED_BUYER,
            h.BUYER_HHI,

            g.CPV2_BREADTH,
            g.CPV3_BREADTH,
            g.CPV4_BREADTH,

            g.N_AWARDS_WITH_NUMBER_OFFERS,
            g.MEDIAN_NUMBER_OFFERS,

            g.N_CROSS_BORDER_COMPARABLE_AWARDS,
            g.CROSS_BORDER_AWARD_SHARE,

            g.FIRST_HISTORY_AWARD_DATE,
            g.LAST_HISTORY_AWARD_DATE,
            g.FIRST_HISTORY_PUBLICATION_DATE,
            g.LAST_HISTORY_PUBLICATION_DATE

        FROM read_parquet('{sql_path(supplier_dim)}') d
        JOIN global_supplier_aggregate g
          USING (SUPPLIER_ENTITY_ID)
        LEFT JOIN global_buyer_hhi h
          USING (SUPPLIER_ENTITY_ID)
        """
    )


# ---------------------------------------------------------------------
# Context feature builder
# ---------------------------------------------------------------------

def build_context_features(
    con: duckdb.DuckDBPyConnection,
    level: str,
) -> str:
    """
    Build supplier-context features for CPV2, CPV3, or CPV4.
    """
    if level not in {"CPV2", "CPV3", "CPV4"}:
        raise ValueError(f"Unsupported context level: {level}")

    lower = level.lower()
    context_col = qident(level)

    buyer_counts = f"{lower}_buyer_counts"
    buyer_hhi = f"{lower}_buyer_hhi"
    context_agg = f"{lower}_supplier_aggregate"
    output_table = f"supplier_{lower}_features"

    con.execute(f"DROP TABLE IF EXISTS {buyer_counts}")
    con.execute(
        f"""
        CREATE TABLE {buyer_counts} AS
        SELECT
            {context_col} AS CONTEXT_KEY,
            SUPPLIER_ENTITY_ID,
            BUYER_KEY,
            COUNT(DISTINCT AWARD_KEY) AS BUYER_AWARD_COUNT
        FROM historical_award_analytic
        WHERE
            {context_col} IS NOT NULL
            AND BUYER_KEY IS NOT NULL
        GROUP BY
            {context_col},
            SUPPLIER_ENTITY_ID,
            BUYER_KEY
        """
    )

    con.execute(f"DROP TABLE IF EXISTS {buyer_hhi}")
    con.execute(
        f"""
        CREATE TABLE {buyer_hhi} AS
        WITH totals AS (
            SELECT
                CONTEXT_KEY,
                SUPPLIER_ENTITY_ID,
                SUM(BUYER_AWARD_COUNT)
                    AS TOTAL_IDENTIFIED_BUYER_AWARDS
            FROM {buyer_counts}
            GROUP BY
                CONTEXT_KEY,
                SUPPLIER_ENTITY_ID
        )
        SELECT
            b.CONTEXT_KEY,
            b.SUPPLIER_ENTITY_ID,
            COUNT(*) AS N_UNIQUE_BUYERS_IN_CONTEXT,
            t.TOTAL_IDENTIFIED_BUYER_AWARDS
                AS N_CONTEXT_AWARDS_WITH_IDENTIFIED_BUYER,
            SUM(
                POWER(
                    CAST(b.BUYER_AWARD_COUNT AS DOUBLE)
                    / NULLIF(t.TOTAL_IDENTIFIED_BUYER_AWARDS, 0),
                    2
                )
            ) AS CONTEXT_BUYER_HHI
        FROM {buyer_counts} b
        JOIN totals t
          USING (CONTEXT_KEY, SUPPLIER_ENTITY_ID)
        GROUP BY
            b.CONTEXT_KEY,
            b.SUPPLIER_ENTITY_ID,
            t.TOTAL_IDENTIFIED_BUYER_AWARDS
        """
    )

    con.execute(f"DROP TABLE IF EXISTS {context_agg}")
    con.execute(
        f"""
        CREATE TABLE {context_agg} AS
        SELECT
            {context_col} AS CONTEXT_KEY,
            SUPPLIER_ENTITY_ID,

            COUNT(DISTINCT AWARD_KEY)
                AS N_CONTEXT_AWARDS,

            COUNT(DISTINCT AWARD_KEY) FILTER (
                WHERE AWARD_VALUE_EURO_PREFERRED IS NOT NULL
            ) AS N_CONTEXT_AWARDS_WITH_VALUE,

            SUM(AWARD_VALUE_EURO_PREFERRED)
                AS TOTAL_CONTEXT_AWARD_VALUE_EURO,

            MEDIAN(AWARD_VALUE_EURO_PREFERRED)
                AS MEDIAN_CONTEXT_AWARD_VALUE_EURO,

            AVG(AWARD_VALUE_EURO_PREFERRED)
                AS MEAN_CONTEXT_AWARD_VALUE_EURO,

            MIN(HISTORY_AWARD_DATE)
                AS FIRST_CONTEXT_AWARD_DATE,

            MAX(HISTORY_AWARD_DATE)
                AS LAST_CONTEXT_AWARD_DATE,

            MIN(HISTORY_PUBLICATION_DATE)
                AS FIRST_CONTEXT_PUBLICATION_DATE,

            MAX(HISTORY_PUBLICATION_DATE)
                AS LAST_CONTEXT_PUBLICATION_DATE,

            COUNT(DISTINCT AWARD_KEY) FILTER (
                WHERE NUMBER_OFFERS_ANALYTIC IS NOT NULL
            ) AS N_CONTEXT_AWARDS_WITH_NUMBER_OFFERS,

            MEDIAN(NUMBER_OFFERS_ANALYTIC)
                AS MEDIAN_CONTEXT_NUMBER_OFFERS,

            COUNT(DISTINCT AWARD_KEY) FILTER (
                WHERE IS_CROSS_BORDER_COMPARABLE
            ) AS N_CONTEXT_CROSS_BORDER_COMPARABLE_AWARDS,

            AVG(CAST(CROSS_BORDER_FLAG AS DOUBLE))
                AS CONTEXT_CROSS_BORDER_AWARD_SHARE

        FROM historical_award_analytic
        WHERE {context_col} IS NOT NULL
        GROUP BY
            {context_col},
            SUPPLIER_ENTITY_ID
        """
    )

    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(
        f"""
        CREATE TABLE {output_table} AS
        SELECT
            '{level}'::VARCHAR AS CONTEXT_LEVEL,
            c.CONTEXT_KEY,
            c.SUPPLIER_ENTITY_ID,

            c.N_CONTEXT_AWARDS,
            c.N_CONTEXT_AWARDS_WITH_VALUE,
            c.TOTAL_CONTEXT_AWARD_VALUE_EURO,
            c.MEDIAN_CONTEXT_AWARD_VALUE_EURO,
            c.MEAN_CONTEXT_AWARD_VALUE_EURO,

            COALESCE(h.N_UNIQUE_BUYERS_IN_CONTEXT, 0)
                AS N_UNIQUE_BUYERS_IN_CONTEXT,
            h.N_CONTEXT_AWARDS_WITH_IDENTIFIED_BUYER,
            h.CONTEXT_BUYER_HHI,

            c.N_CONTEXT_AWARDS_WITH_NUMBER_OFFERS,
            c.MEDIAN_CONTEXT_NUMBER_OFFERS,

            c.N_CONTEXT_CROSS_BORDER_COMPARABLE_AWARDS,
            c.CONTEXT_CROSS_BORDER_AWARD_SHARE,

            c.FIRST_CONTEXT_AWARD_DATE,
            c.LAST_CONTEXT_AWARD_DATE,
            c.FIRST_CONTEXT_PUBLICATION_DATE,
            c.LAST_CONTEXT_PUBLICATION_DATE,

            CAST(c.N_CONTEXT_AWARDS AS DOUBLE)
            / NULLIF(g.N_HISTORICAL_AWARDS, 0)
                AS SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT,

            c.TOTAL_CONTEXT_AWARD_VALUE_EURO
            / NULLIF(g.TOTAL_AWARD_VALUE_EURO, 0)
                AS SHARE_OF_SUPPLIER_VALUE_IN_CONTEXT

        FROM {context_agg} c
        JOIN supplier_global_features g
          USING (SUPPLIER_ENTITY_ID)
        LEFT JOIN {buyer_hhi} h
          USING (CONTEXT_KEY, SUPPLIER_ENTITY_ID)
        """
    )

    duplicate_rows = scalar(
        con,
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                CONTEXT_KEY,
                SUPPLIER_ENTITY_ID
            FROM {output_table}
            GROUP BY
                CONTEXT_KEY,
                SUPPLIER_ENTITY_ID
            HAVING COUNT(*) > 1
        )
        """,
    )

    if duplicate_rows:
        raise ValueError(
            f"{output_table} contains duplicate context-supplier rows."
        )

    return output_table


# ---------------------------------------------------------------------
# Target-case feature index
# ---------------------------------------------------------------------

def build_target_case_features(
    con: duckdb.DuckDBPyConnection,
    winner_history_map: Path,
) -> None:
    con.execute("DROP TABLE IF EXISTS target_case_features")
    con.execute(
        f"""
        CREATE TABLE target_case_features AS
        WITH base AS (
            SELECT
                TARGET_CASE_ID,
                CAST(TARGET_DISPATCH_DATE AS DATE)
                    AS TARGET_DISPATCH_DATE,
                CAST(TARGET_CPV AS VARCHAR)
                    AS TARGET_CPV,
                TARGET_PROCUREMENT_COUNTRY,
                HISTORICAL_SUPPLIER_ENTITY_ID
                    AS OBSERVED_WINNER_HISTORICAL_ENTITY_ID,
                COALESCE(
                    TRY_CAST(HISTORICAL_MATCHED AS BOOLEAN),
                    FALSE
                ) AS OBSERVED_WINNER_HISTORICALLY_MATCHED,
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
            TARGET_CASE_ID,
            TARGET_DISPATCH_DATE,
            TARGET_CPV,

            CASE
                WHEN LENGTH(TARGET_CPV_DIGITS) >= 2
                THEN SUBSTR(TARGET_CPV_DIGITS, 1, 2)
                ELSE NULL
            END AS TARGET_CPV2,

            CASE
                WHEN LENGTH(TARGET_CPV_DIGITS) >= 3
                THEN SUBSTR(TARGET_CPV_DIGITS, 1, 3)
                ELSE NULL
            END AS TARGET_CPV3,

            CASE
                WHEN LENGTH(TARGET_CPV_DIGITS) >= 4
                THEN SUBSTR(TARGET_CPV_DIGITS, 1, 4)
                ELSE NULL
            END AS TARGET_CPV4,

            TARGET_PROCUREMENT_COUNTRY,

            OBSERVED_WINNER_HISTORICAL_ENTITY_ID,
            OBSERVED_WINNER_HISTORICALLY_MATCHED,
            HISTORY_MATCH_STATUS

        FROM base
        """
    )

    duplicate_cases = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT TARGET_CASE_ID
            FROM target_case_features
            GROUP BY TARGET_CASE_ID
            HAVING COUNT(*) > 1
        )
        """,
    )

    if duplicate_cases:
        raise ValueError(
            f"target_case_features contains {duplicate_cases} duplicate target cases."
        )


# ---------------------------------------------------------------------
# Diagnostics and reports
# ---------------------------------------------------------------------

def feature_dictionary() -> pd.DataFrame:
    rows = [
        # Global
        ("supplier_global_features", "N_HISTORICAL_AWARDS", "count", "benefit", "Observed number of resolved historical awards in 2015-2016."),
        ("supplier_global_features", "TOTAL_AWARD_VALUE_EURO", "EUR", "benefit_or_context", "Sum of preferred observed award values; use cautiously because value scales differ by sector."),
        ("supplier_global_features", "MEDIAN_AWARD_VALUE_EURO", "EUR", "context", "Median preferred observed award value."),
        ("supplier_global_features", "MEAN_AWARD_VALUE_EURO", "EUR", "context", "Mean preferred observed award value."),
        ("supplier_global_features", "N_UNIQUE_BUYERS", "count", "benefit_or_context", "Number of distinct observed contracting-authority keys."),
        ("supplier_global_features", "BUYER_HHI", "0_to_1", "cost_if_diversification_desired", "Observed buyer concentration based on award-count shares; not a resilience score."),
        ("supplier_global_features", "CPV2_BREADTH", "count", "benefit_or_context", "Number of distinct observed CPV2 divisions."),
        ("supplier_global_features", "CPV3_BREADTH", "count", "benefit_or_context", "Number of distinct observed CPV3 prefixes."),
        ("supplier_global_features", "CPV4_BREADTH", "count", "benefit_or_context", "Number of distinct observed CPV4 prefixes."),
        ("supplier_global_features", "MEDIAN_NUMBER_OFFERS", "count", "context", "Median observed procurement competition context; not supplier quality."),
        ("supplier_global_features", "CROSS_BORDER_AWARD_SHARE", "0_to_1", "benefit_or_context", "Share of comparable awards where supplier and buyer countries differ."),
        ("supplier_global_features", "LAST_HISTORY_AWARD_DATE", "date", "recency_input", "Used with target date to derive target-relative award recency."),
        ("supplier_global_features", "LAST_HISTORY_PUBLICATION_DATE", "date", "availability_input", "Last historical publication date in the frozen feature window."),

        # Context
        ("supplier_context_features", "N_CONTEXT_AWARDS", "count", "benefit", "Observed awards in the matching CPV context."),
        ("supplier_context_features", "TOTAL_CONTEXT_AWARD_VALUE_EURO", "EUR", "benefit_or_context", "Total preferred award value in the matching CPV context."),
        ("supplier_context_features", "MEDIAN_CONTEXT_AWARD_VALUE_EURO", "EUR", "context", "Median preferred award value in the matching CPV context."),
        ("supplier_context_features", "N_UNIQUE_BUYERS_IN_CONTEXT", "count", "benefit_or_context", "Distinct observed buyers in the matching CPV context."),
        ("supplier_context_features", "CONTEXT_BUYER_HHI", "0_to_1", "cost_if_diversification_desired", "Buyer concentration within the matching CPV context."),
        ("supplier_context_features", "MEDIAN_CONTEXT_NUMBER_OFFERS", "count", "context", "Median observed competition context within the CPV context."),
        ("supplier_context_features", "CONTEXT_CROSS_BORDER_AWARD_SHARE", "0_to_1", "benefit_or_context", "Cross-border observed-award share within the CPV context."),
        ("supplier_context_features", "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT", "0_to_1", "benefit_or_context", "Share of the supplier's historical awards occurring in the context."),
        ("supplier_context_features", "SHARE_OF_SUPPLIER_VALUE_IN_CONTEXT", "ratio", "benefit_or_context", "Share of observed supplier award value occurring in the context."),
        ("supplier_context_features", "LAST_CONTEXT_AWARD_DATE", "date", "recency_input", "Used with target date to derive context-specific recency."),

        # Target
        ("target_case_features", "TARGET_DISPATCH_DATE", "date", "target_input", "Target procurement publication date."),
        ("target_case_features", "TARGET_CPV2", "code", "target_context", "Two-digit CPV context key."),
        ("target_case_features", "TARGET_CPV3", "code", "target_context", "Three-digit CPV context key."),
        ("target_case_features", "TARGET_CPV4", "code", "target_context", "Four-digit CPV context key."),
        ("target_case_features", "OBSERVED_WINNER_HISTORICALLY_MATCHED", "boolean", "evaluation_label", "Whether the observed winner maps to the frozen historical universe."),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "table",
            "feature",
            "unit",
            "suggested_role",
            "definition",
        ],
    )


def feature_coverage_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    rows = []

    global_features = [
        "TOTAL_AWARD_VALUE_EURO",
        "MEDIAN_AWARD_VALUE_EURO",
        "N_UNIQUE_BUYERS",
        "BUYER_HHI",
        "CPV2_BREADTH",
        "CPV3_BREADTH",
        "CPV4_BREADTH",
        "MEDIAN_NUMBER_OFFERS",
        "CROSS_BORDER_AWARD_SHARE",
    ]

    for feature in global_features:
        rows.append(
            fetch_df(
                con,
                f"""
                SELECT
                    'supplier_global_features' AS table_name,
                    '{feature}' AS feature_name,
                    COUNT(*) AS n_rows,
                    COUNT(*) FILTER (
                        WHERE {qident(feature)} IS NOT NULL
                    ) AS nonmissing_rows,
                    100.0 * COUNT(*) FILTER (
                        WHERE {qident(feature)} IS NOT NULL
                    ) / COUNT(*) AS coverage_pct
                FROM supplier_global_features
                """
            )
        )

    context_tables = [
        ("supplier_cpv2_features", "CPV2"),
        ("supplier_cpv3_features", "CPV3"),
        ("supplier_cpv4_features", "CPV4"),
    ]

    context_features = [
        "TOTAL_CONTEXT_AWARD_VALUE_EURO",
        "MEDIAN_CONTEXT_AWARD_VALUE_EURO",
        "N_UNIQUE_BUYERS_IN_CONTEXT",
        "CONTEXT_BUYER_HHI",
        "MEDIAN_CONTEXT_NUMBER_OFFERS",
        "CONTEXT_CROSS_BORDER_AWARD_SHARE",
        "SHARE_OF_SUPPLIER_AWARDS_IN_CONTEXT",
        "SHARE_OF_SUPPLIER_VALUE_IN_CONTEXT",
    ]

    for table, _level in context_tables:
        for feature in context_features:
            rows.append(
                fetch_df(
                    con,
                    f"""
                    SELECT
                        '{table}' AS table_name,
                        '{feature}' AS feature_name,
                        COUNT(*) AS n_rows,
                        COUNT(*) FILTER (
                            WHERE {qident(feature)} IS NOT NULL
                        ) AS nonmissing_rows,
                        100.0 * COUNT(*) FILTER (
                            WHERE {qident(feature)} IS NOT NULL
                        ) / COUNT(*) AS coverage_pct
                    FROM {table}
                    """
                )
            )

    return pd.concat(rows, ignore_index=True)


def context_feature_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    pieces = []

    for table, level in [
        ("supplier_cpv2_features", "CPV2"),
        ("supplier_cpv3_features", "CPV3"),
        ("supplier_cpv4_features", "CPV4"),
    ]:
        pieces.append(
            fetch_df(
                con,
                f"""
                SELECT
                    '{level}' AS CONTEXT_LEVEL,
                    COUNT(DISTINCT CONTEXT_KEY)
                        AS N_CONTEXTS,
                    COUNT(DISTINCT SUPPLIER_ENTITY_ID)
                        AS N_DISTINCT_SUPPLIERS,
                    COUNT(*) AS N_SUPPLIER_CONTEXT_ROWS,
                    MEDIAN(N_CONTEXT_AWARDS)
                        AS MEDIAN_CONTEXT_AWARDS,
                    QUANTILE_CONT(N_CONTEXT_AWARDS, 0.95)
                        AS P95_CONTEXT_AWARDS,
                    MAX(N_CONTEXT_AWARDS)
                        AS MAX_CONTEXT_AWARDS
                FROM {table}
                """
            )
        )

    return pd.concat(pieces, ignore_index=True)


def temporal_audit(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return fetch_df(
        con,
        """
        SELECT
            MIN(t.TARGET_DISPATCH_DATE)
                AS MIN_TARGET_DISPATCH_DATE,
            MAX(t.TARGET_DISPATCH_DATE)
                AS MAX_TARGET_DISPATCH_DATE,

            MIN(g.FIRST_HISTORY_PUBLICATION_DATE)
                AS MIN_HISTORY_PUBLICATION_DATE,
            MAX(g.LAST_HISTORY_PUBLICATION_DATE)
                AS MAX_HISTORY_PUBLICATION_DATE,

            CASE
                WHEN MAX(g.LAST_HISTORY_PUBLICATION_DATE)
                     < MIN(t.TARGET_DISPATCH_DATE)
                THEN TRUE
                ELSE FALSE
            END AS HISTORY_WINDOW_STRICTLY_PRECEDES_TARGET_WINDOW

        FROM target_case_features t
        CROSS JOIN supplier_global_features g
        """
    )


def build_metrics(
    con: duckdb.DuckDBPyConnection,
    resolved_cols: dict[str, str | None],
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

    add(
        "historical_analytic_awards",
        scalar(con, "SELECT COUNT(*) FROM historical_award_analytic"),
    )
    add(
        "supplier_global_feature_rows",
        scalar(con, "SELECT COUNT(*) FROM supplier_global_features"),
    )
    add(
        "supplier_cpv2_feature_rows",
        scalar(con, "SELECT COUNT(*) FROM supplier_cpv2_features"),
    )
    add(
        "supplier_cpv3_feature_rows",
        scalar(con, "SELECT COUNT(*) FROM supplier_cpv3_features"),
    )
    add(
        "supplier_cpv4_feature_rows",
        scalar(con, "SELECT COUNT(*) FROM supplier_cpv4_features"),
    )
    add(
        "target_case_rows",
        scalar(con, "SELECT COUNT(*) FROM target_case_features"),
    )
    add(
        "award_value_primary_field",
        resolved_cols["value_primary"],
    )
    add(
        "award_value_fallback_field",
        resolved_cols["value_fallback"],
    )
    add(
        "number_offers_field",
        resolved_cols["number_offers"],
    )

    return pd.DataFrame(rows)


def create_report(
    con: duckdb.DuckDBPyConnection,
    metrics_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_map = dict(zip(metrics_df["metric"], metrics_df["value"]))
    t = temporal_df.iloc[0]

    lines = [
        "# TED 2015-2016 Historical Supplier Feature Store Build Report",
        "",
        "## Purpose",
        "",
        "This feature store is the final main preprocessing layer before ranking, "
        "scenario, ablation, and robustness experiments.",
        "",
        "It uses only frozen 2015-2016 historical award information. No 2017 "
        "outcome is used to construct supplier features.",
        "",
        "## Output size",
        "",
        f"- Historical analytical awards: {metric_map.get('historical_analytic_awards')}",
        f"- Supplier-global rows: {metric_map.get('supplier_global_feature_rows')}",
        f"- Supplier-CPV2 rows: {metric_map.get('supplier_cpv2_feature_rows')}",
        f"- Supplier-CPV3 rows: {metric_map.get('supplier_cpv3_feature_rows')}",
        f"- Supplier-CPV4 rows: {metric_map.get('supplier_cpv4_feature_rows')}",
        f"- Target-case rows: {metric_map.get('target_case_rows')}",
        "",
        "## Source-field choices",
        "",
        f"- Preferred award-value field: {metric_map.get('award_value_primary_field')}",
        f"- Award-value fallback field: {metric_map.get('award_value_fallback_field')}",
        f"- Number-offers field: {metric_map.get('number_offers_field')}",
        "",
        "## Temporal audit",
        "",
        f"- Earliest target dispatch date: {t['MIN_TARGET_DISPATCH_DATE']}",
        f"- Latest target dispatch date: {t['MAX_TARGET_DISPATCH_DATE']}",
        f"- Earliest historical publication date: {t['MIN_HISTORY_PUBLICATION_DATE']}",
        f"- Latest historical publication date: {t['MAX_HISTORY_PUBLICATION_DATE']}",
        f"- History strictly precedes target window: "
        f"{bool(t['HISTORY_WINDOW_STRICTLY_PRECEDES_TARGET_WINDOW'])}",
        "",
        "## Feature interpretation",
        "",
        "- BUYER_HHI and CONTEXT_BUYER_HHI measure observed buyer concentration.",
        "- MEDIAN_NUMBER_OFFERS and MEDIAN_CONTEXT_NUMBER_OFFERS describe observed "
        "competition context, not supplier quality.",
        "- Cross-border shares are computed only where supplier and buyer countries "
        "are both usable two-letter country codes.",
        "- Award-value features describe observed public-award history and should be "
        "normalized or transformed before multi-criteria ranking.",
        "- Target-relative recency should be derived during analysis from target date "
        "minus the relevant last historical date.",
        "",
        "## End of preprocessing",
        "",
        "After this feature store passes coverage review, the main data construction "
        "pipeline is complete. Subsequent scripts should be analysis scripts for "
        "baseline ranking, scenario ranking, LLM evaluation, ablation, and robustness.",
        "",
    ]

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
            "Build compact 2015-2016 supplier-global and supplier-context "
            "feature tables plus the 2017 target-case feature index."
        )
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
        "--winner-history-map",
        required=True,
        type=Path,
        help="Path to winner_history_map_2017.parquet",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/supplier_feature_store"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/supplier_feature_store"),
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
            "Allow the historical feature window to overlap the target window. "
            "Not recommended for the current 2015-2016 -> 2017 design."
        ),
    )

    args = parser.parse_args()

    for path in [
        args.award_supplier_map,
        args.award_fact,
        args.supplier_dim,
        args.winner_history_map,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    work_db = (
        args.work_db
        if args.work_db is not None
        else args.output_dir / "_supplier_feature_store_build.duckdb"
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
        print("1/8 Validating inputs and resolving source fields...")
        resolved_cols = validate_inputs(
            con,
            args.award_supplier_map,
            args.award_fact,
            args.supplier_dim,
            args.winner_history_map,
        )

        print("Resolved source fields:")
        for key, value in resolved_cols.items():
            print(f"  {key}: {value}")

        print("2/8 Building historical analytical award base...")
        build_historical_award_base(
            con,
            args.award_supplier_map,
            args.award_fact,
            args.supplier_dim,
            resolved_cols,
        )

        print("3/8 Building supplier-global features...")
        build_global_supplier_features(
            con,
            args.supplier_dim,
        )

        print("4/8 Building supplier-context feature tables...")
        build_context_features(con, "CPV2")
        build_context_features(con, "CPV3")
        build_context_features(con, "CPV4")

        print("5/8 Building 2017 target-case feature index...")
        build_target_case_features(
            con,
            args.winner_history_map,
        )

        print("6/8 Running temporal audit...")
        temporal_df = temporal_audit(con)
        temporal_safe = bool(
            temporal_df.iloc[0][
                "HISTORY_WINDOW_STRICTLY_PRECEDES_TARGET_WINDOW"
            ]
        )

        if not temporal_safe and not args.allow_overlap_history_target:
            raise ValueError(
                "Historical feature window overlaps the target window. "
                "Review temporal availability before proceeding."
            )

        print("7/8 Writing reports...")
        dictionary_df = feature_dictionary()
        coverage_df = feature_coverage_summary(con)
        context_df = context_feature_summary(con)
        metrics_df = build_metrics(con, resolved_cols)

        write_csv(
            dictionary_df,
            args.report_dir / "feature_dictionary.csv",
        )
        write_csv(
            metrics_df,
            args.report_dir / "feature_build_metrics.csv",
        )
        write_csv(
            coverage_df,
            args.report_dir / "feature_coverage_summary.csv",
        )
        write_csv(
            context_df,
            args.report_dir / "context_feature_summary.csv",
        )
        write_csv(
            temporal_df,
            args.report_dir / "feature_temporal_audit.csv",
        )

        create_report(
            con,
            metrics_df,
            temporal_df,
            args.report_dir / "feature_build_report.md",
        )

        print("8/8 Writing compact Parquet feature-store outputs...")
        copy_table_to_parquet(
            con,
            "supplier_global_features",
            args.output_dir
            / "supplier_global_features_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "supplier_cpv2_features",
            args.output_dir
            / "supplier_cpv2_features_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "supplier_cpv3_features",
            args.output_dir
            / "supplier_cpv3_features_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "supplier_cpv4_features",
            args.output_dir
            / "supplier_cpv4_features_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "target_case_features",
            args.output_dir
            / "target_case_features_2017.parquet",
        )

        print()
        print("=" * 78)
        print("Supplier feature-store build complete")
        print("=" * 78)
        print(f"Processed outputs: {args.output_dir.resolve()}")
        print(f"Reports:           {args.report_dir.resolve()}")
        print(f"Work database:     {work_db.resolve()}")
        print()
        print("Review these first:")
        print("  feature_build_report.md")
        print("  feature_build_metrics.csv")
        print("  feature_coverage_summary.csv")
        print("  context_feature_summary.csv")
        print("  feature_temporal_audit.csv")
        print("  feature_dictionary.csv")
        print()
        print("After coverage review, main preprocessing is complete.")
        print()

    finally:
        con.close()


if __name__ == "__main__":
    main()
