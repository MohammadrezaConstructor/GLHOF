#!/usr/bin/env python3
"""
Build a conservative historical supplier dimension from TED CAN 2015-2016.

Input
-----
award_fact_2015_2017.parquet produced by build_can_award_tables.py.

Core design
-----------
- Use only source-year 2015-2016 award facts.
- Do not use 2017 data to learn supplier identities or name bridges.
- Separate explicit group awards and structurally list-valued winner records.
- Apply conservative national-ID hygiene before creating ID anchors.
- Learn name-country -> ID-anchor bridges from 2015-2016 only.
- Preserve one-to-many bridges as ambiguous instead of forcing a match.
- Create standalone name-country entities only when no reliable anchor evidence exists.
- Use name-town only as a low-confidence fallback.
- Do not fuzzy-match suppliers.

Outputs
-------
Processed:
  supplier_dim_2015_2016.parquet
  supplier_aliases_2015_2016.parquet
  supplier_ambiguities_2015_2016.parquet
  award_supplier_map_2015_2016.parquet
  national_id_anchor_profile_2015_2016.parquet

Reports:
  supplier_dimension_build_metrics.csv
  national_id_quality_summary.csv
  resolution_method_summary.csv
  bridge_status_summary.csv
  temporal_quality_summary.csv
  supplier_dimension_build_report.md

Requires:
  pip install duckdb pandas pyarrow
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd


REQUIRED_COLUMNS = {
    "AWARD_KEY",
    "SOURCE_YEAR_MIN",
    "SOURCE_YEAR_MAX",
    "XSD_VERSION_CLEAN",
    "DT_AWARD_CLEAN",
    "DT_DISPATCH_CLEAN",
    "PUBLICATION_LAG_DAYS",
    "WIN_NAME_CLEAN",
    "WIN_NATIONALID_CLEAN",
    "WIN_TOWN_CLEAN",
    "WIN_COUNTRY_CODE_CLEAN",
    "B_AWARDED_TO_A_GROUP_RAW",
    "B_AWARDED_TO_A_GROUP_CLEAN",
    "CAE_NAME_CLEAN",
    "ISO_COUNTRY_CODE_CLEAN",
    "CPV_CLEAN",
}

GENERIC_ID_PLACEHOLDERS = {
    "0", "1", "00", "000", "0000", "NA", "NAN", "NONE", "NULL",
    "UNKNOWN", "NOTAVAILABLE", "NOTAPPLICABLE", "VAT", "VATNO", "NIL",
    "NOID", "NATIONALID", "REGISTRATIONNUMBER",
}


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def clean_text_expr(column: str) -> str:
    return f"NULLIF(TRIM(CAST({qident(column)} AS VARCHAR)), '')"


def normalize_entity_expr(column: str) -> str:
    c = clean_text_expr(column)
    return (
        "NULLIF(TRIM(REGEXP_REPLACE("
        f"LOWER(COALESCE({c}, '')), '[^[:alnum:]]+', ' ', 'g')), '')"
    )


def normalize_identifier_expr(column: str) -> str:
    c = clean_text_expr(column)
    return (
        "NULLIF(REGEXP_REPLACE("
        f"UPPER(COALESCE({c}, '')), '[^[:alnum:]]+', '', 'g'), '')"
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


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def placeholder_sql_list() -> str:
    vals = sorted(v.replace("'", "''") for v in GENERIC_ID_PLACEHOLDERS)
    return ", ".join(f"'{v}'" for v in vals)


def validate_input(con: duckdb.DuckDBPyConnection, award_fact_path: Path) -> None:
    cols = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_path(award_fact_path)}')"
        ).fetchall()
    }
    missing = sorted(REQUIRED_COLUMNS - cols)
    if missing:
        raise ValueError(
            "Input award fact is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def build_history_base(
    con: duckdb.DuckDBPyConnection,
    award_fact_path: Path,
) -> None:
    con.execute("DROP TABLE IF EXISTS history_awards_raw")
    con.execute(
        f"""
        CREATE TABLE history_awards_raw AS
        SELECT *
        FROM read_parquet('{sql_path(award_fact_path)}')
        WHERE SOURCE_YEAR_MIN IN (2015, 2016)
          AND SOURCE_YEAR_MAX IN (2015, 2016)
        """
    )

    name_norm = normalize_entity_expr("WIN_NAME_CLEAN")
    nid_norm = normalize_identifier_expr("WIN_NATIONALID_CLEAN")
    town_norm = normalize_entity_expr("WIN_TOWN_CLEAN")
    buyer_norm = normalize_entity_expr("CAE_NAME_CLEAN")
    supplier_country = f"UPPER({clean_text_expr('WIN_COUNTRY_CODE_CLEAN')})"
    buyer_country = f"UPPER({clean_text_expr('ISO_COUNTRY_CODE_CLEAN')})"

    con.execute("DROP TABLE IF EXISTS history_awards_base")
    con.execute(
        f"""
        CREATE TABLE history_awards_base AS
        SELECT
            *,
            {name_norm} AS NAME_NORM,
            {nid_norm} AS NATIONAL_ID_NORM,
            {supplier_country} AS SUPPLIER_COUNTRY_NORM,
            {town_norm} AS TOWN_NORM,
            {buyer_norm} AS BUYER_NAME_NORM,
            {buyer_country} AS BUYER_COUNTRY_NORM,

            CASE
                WHEN {name_norm} IS NOT NULL
                 AND {supplier_country} IS NOT NULL
                 AND LENGTH({supplier_country}) = 2
                THEN {supplier_country} || '||' || {name_norm}
                ELSE NULL
            END AS NAME_COUNTRY_KEY,

            CASE
                WHEN {name_norm} IS NOT NULL
                 AND {town_norm} IS NOT NULL
                THEN {town_norm} || '||' || {name_norm}
                ELSE NULL
            END AS NAME_TOWN_KEY,

            CASE
                WHEN {buyer_norm} IS NOT NULL AND {buyer_country} IS NOT NULL
                THEN {buyer_country} || '||' || {buyer_norm}
                WHEN {buyer_norm} IS NOT NULL
                THEN 'UNKNOWN_COUNTRY||' || {buyer_norm}
                ELSE NULL
            END AS BUYER_KEY_AUDIT,

            CASE
                WHEN CPV_CLEAN IS NOT NULL
                THEN REGEXP_EXTRACT(CAST(CPV_CLEAN AS VARCHAR), '([0-9]{{2}})', 1)
                ELSE NULL
            END AS CPV2,

            CASE
                WHEN COALESCE(B_AWARDED_TO_A_GROUP_CLEAN, FALSE) IS TRUE THEN TRUE
                WHEN POSITION('---' IN COALESCE(WIN_NAME_CLEAN, '')) > 0 THEN TRUE
                WHEN POSITION('---' IN COALESCE(WIN_NATIONALID_CLEAN, '')) > 0 THEN TRUE
                WHEN POSITION('---' IN COALESCE(WIN_COUNTRY_CODE_CLEAN, '')) > 0 THEN TRUE
                WHEN POSITION('---' IN COALESCE(WIN_TOWN_CLEAN, '')) > 0 THEN TRUE
                ELSE FALSE
            END AS IS_GROUP_OR_LIST_RECORD,

            CASE
                WHEN COALESCE(B_AWARDED_TO_A_GROUP_CLEAN, FALSE) IS TRUE
                    THEN 'explicit_group_award'
                WHEN POSITION('---' IN COALESCE(WIN_NAME_CLEAN, '')) > 0
                    THEN 'list_valued_winner_name'
                WHEN POSITION('---' IN COALESCE(WIN_NATIONALID_CLEAN, '')) > 0
                    THEN 'list_valued_national_id'
                WHEN POSITION('---' IN COALESCE(WIN_COUNTRY_CODE_CLEAN, '')) > 0
                    THEN 'list_valued_supplier_country'
                WHEN POSITION('---' IN COALESCE(WIN_TOWN_CLEAN, '')) > 0
                    THEN 'list_valued_supplier_town'
                ELSE NULL
            END AS GROUP_OR_LIST_REASON
        FROM history_awards_raw
        """
    )


def classify_id_hygiene(
    con: duckdb.DuckDBPyConnection,
    min_id_length: int,
) -> None:
    placeholders = placeholder_sql_list()

    con.execute("DROP TABLE IF EXISTS history_awards_id_quality")
    con.execute(
        f"""
        CREATE TABLE history_awards_id_quality AS
        SELECT
            *,
            CASE
                WHEN NATIONAL_ID_NORM IS NULL
                    THEN 'missing'
                WHEN POSITION('---' IN COALESCE(WIN_NATIONALID_CLEAN, '')) > 0
                    THEN 'multi_value_identifier'
                WHEN IS_GROUP_OR_LIST_RECORD
                    THEN 'group_or_list_record'
                WHEN SUPPLIER_COUNTRY_NORM IS NULL
                    THEN 'missing_country_context'
                WHEN LENGTH(SUPPLIER_COUNTRY_NORM) <> 2
                    THEN 'invalid_country_context'
                WHEN NATIONAL_ID_NORM = SUPPLIER_COUNTRY_NORM
                    THEN 'country_code_placeholder'
                WHEN NATIONAL_ID_NORM IN ({placeholders})
                    THEN 'generic_placeholder'
                WHEN LENGTH(NATIONAL_ID_NORM) < {int(min_id_length)}
                    THEN 'suspicious_short'
                ELSE 'candidate_valid_anchor'
            END AS NATIONAL_ID_VALIDITY_INITIAL
        FROM history_awards_base
        """
    )


def build_anchor_profile_and_reliable_anchors(
    con: duckdb.DuckDBPyConnection,
    max_names_per_id_anchor: int,
) -> None:
    con.execute("DROP TABLE IF EXISTS national_id_anchor_profile")
    con.execute(
        f"""
        CREATE TABLE national_id_anchor_profile AS
        SELECT
            SUPPLIER_COUNTRY_NORM,
            NATIONAL_ID_NORM,
            SUPPLIER_COUNTRY_NORM || '||' || NATIONAL_ID_NORM
                AS NATIONAL_ID_COUNTRY_KEY,
            COUNT(*) AS N_AWARDS,
            COUNT(DISTINCT NAME_NORM) FILTER (WHERE NAME_NORM IS NOT NULL)
                AS N_DISTINCT_NAMES,
            COUNT(DISTINCT NAME_COUNTRY_KEY) FILTER (WHERE NAME_COUNTRY_KEY IS NOT NULL)
                AS N_DISTINCT_NAME_COUNTRY_KEYS,
            COUNT(DISTINCT TOWN_NORM) FILTER (WHERE TOWN_NORM IS NOT NULL)
                AS N_DISTINCT_TOWNS,
            MIN(DT_AWARD_CLEAN) AS FIRST_AWARD_DATE,
            MAX(DT_AWARD_CLEAN) AS LAST_AWARD_DATE,
            MIN(DT_DISPATCH_CLEAN) AS FIRST_DISPATCH_DATE,
            MAX(DT_DISPATCH_CLEAN) AS LAST_DISPATCH_DATE,
            CASE
                WHEN COUNT(DISTINCT NAME_NORM) FILTER (WHERE NAME_NORM IS NOT NULL)
                     > {int(max_names_per_id_anchor)}
                THEN 'excessive_name_alias_dispersion'
                ELSE 'reliable_anchor'
            END AS ANCHOR_RELIABILITY
        FROM history_awards_id_quality
        WHERE NATIONAL_ID_VALIDITY_INITIAL = 'candidate_valid_anchor'
        GROUP BY SUPPLIER_COUNTRY_NORM, NATIONAL_ID_NORM
        """
    )

    con.execute("DROP TABLE IF EXISTS reliable_id_anchors")
    con.execute(
        """
        CREATE TABLE reliable_id_anchors AS
        SELECT
            *,
            'ID||' || NATIONAL_ID_COUNTRY_KEY AS SUPPLIER_ENTITY_ID
        FROM national_id_anchor_profile
        WHERE ANCHOR_RELIABILITY = 'reliable_anchor'
        """
    )

    con.execute("DROP TABLE IF EXISTS history_awards_id_final")
    con.execute(
        """
        CREATE TABLE history_awards_id_final AS
        SELECT
            h.*,
            CASE
                WHEN h.NATIONAL_ID_VALIDITY_INITIAL <> 'candidate_valid_anchor'
                    THEN h.NATIONAL_ID_VALIDITY_INITIAL
                WHEN a.ANCHOR_RELIABILITY = 'reliable_anchor'
                    THEN 'valid_anchor'
                WHEN p.ANCHOR_RELIABILITY = 'excessive_name_alias_dispersion'
                    THEN 'excessive_name_alias_dispersion'
                ELSE 'candidate_anchor_not_classified'
            END AS NATIONAL_ID_VALIDITY,
            a.SUPPLIER_ENTITY_ID AS DIRECT_ANCHOR_ENTITY_ID
        FROM history_awards_id_quality h
        LEFT JOIN reliable_id_anchors a
          ON h.SUPPLIER_COUNTRY_NORM = a.SUPPLIER_COUNTRY_NORM
         AND h.NATIONAL_ID_NORM = a.NATIONAL_ID_NORM
        LEFT JOIN national_id_anchor_profile p
          ON h.SUPPLIER_COUNTRY_NORM = p.SUPPLIER_COUNTRY_NORM
         AND h.NATIONAL_ID_NORM = p.NATIONAL_ID_NORM
        """
    )


def build_name_country_bridges(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS name_country_anchor_support")
    con.execute(
        """
        CREATE TABLE name_country_anchor_support AS
        SELECT
            NAME_COUNTRY_KEY,
            DIRECT_ANCHOR_ENTITY_ID AS CANDIDATE_SUPPLIER_ENTITY_ID,
            COUNT(*) AS SUPPORTING_AWARD_COUNT,
            MIN(DT_AWARD_CLEAN) AS FIRST_SUPPORT_AWARD_DATE,
            MAX(DT_AWARD_CLEAN) AS LAST_SUPPORT_AWARD_DATE,
            MIN(WIN_NAME_CLEAN) AS EXAMPLE_WINNER_NAME
        FROM history_awards_id_final
        WHERE NAME_COUNTRY_KEY IS NOT NULL
          AND DIRECT_ANCHOR_ENTITY_ID IS NOT NULL
        GROUP BY NAME_COUNTRY_KEY, DIRECT_ANCHOR_ENTITY_ID
        """
    )

    con.execute("DROP TABLE IF EXISTS name_country_bridge_profile")
    con.execute(
        """
        CREATE TABLE name_country_bridge_profile AS
        SELECT
            NAME_COUNTRY_KEY,
            COUNT(DISTINCT CANDIDATE_SUPPLIER_ENTITY_ID) AS N_CANDIDATE_ANCHORS,
            SUM(SUPPORTING_AWARD_COUNT) AS TOTAL_SUPPORTING_AWARDS,
            CASE
                WHEN COUNT(DISTINCT CANDIDATE_SUPPLIER_ENTITY_ID) = 1
                    THEN 'one_to_one_anchor_bridge'
                WHEN COUNT(DISTINCT CANDIDATE_SUPPLIER_ENTITY_ID) > 1
                    THEN 'ambiguous_multi_anchor_bridge'
                ELSE 'no_anchor_bridge'
            END AS BRIDGE_STATUS
        FROM name_country_anchor_support
        GROUP BY NAME_COUNTRY_KEY
        """
    )

    con.execute("DROP TABLE IF EXISTS one_to_one_name_country_bridges")
    con.execute(
        """
        CREATE TABLE one_to_one_name_country_bridges AS
        SELECT
            s.NAME_COUNTRY_KEY,
            MIN(s.CANDIDATE_SUPPLIER_ENTITY_ID) AS SUPPLIER_ENTITY_ID,
            SUM(s.SUPPORTING_AWARD_COUNT) AS SUPPORTING_AWARD_COUNT
        FROM name_country_anchor_support s
        JOIN name_country_bridge_profile p USING (NAME_COUNTRY_KEY)
        WHERE p.BRIDGE_STATUS = 'one_to_one_anchor_bridge'
        GROUP BY s.NAME_COUNTRY_KEY
        """
    )

    con.execute("DROP TABLE IF EXISTS supplier_ambiguities")
    con.execute(
        """
        CREATE TABLE supplier_ambiguities AS
        SELECT
            s.NAME_COUNTRY_KEY,
            s.CANDIDATE_SUPPLIER_ENTITY_ID,
            s.SUPPORTING_AWARD_COUNT,
            s.FIRST_SUPPORT_AWARD_DATE,
            s.LAST_SUPPORT_AWARD_DATE,
            s.EXAMPLE_WINNER_NAME,
            'name_country_maps_to_multiple_reliable_id_anchors'
                AS AMBIGUITY_REASON
        FROM name_country_anchor_support s
        JOIN name_country_bridge_profile p USING (NAME_COUNTRY_KEY)
        WHERE p.BRIDGE_STATUS = 'ambiguous_multi_anchor_bridge'
        """
    )


def build_award_supplier_map(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS award_supplier_map")
    con.execute(
        """
        CREATE TABLE award_supplier_map AS
        SELECT
            h.AWARD_KEY,
            h.SOURCE_YEAR_MIN,
            h.SOURCE_YEAR_MAX,
            h.XSD_VERSION_CLEAN,
            h.DT_AWARD_CLEAN,
            h.DT_DISPATCH_CLEAN,
            h.PUBLICATION_LAG_DAYS,
            h.WIN_NAME_CLEAN,
            h.NAME_NORM,
            h.WIN_NATIONALID_CLEAN,
            h.NATIONAL_ID_NORM,
            h.WIN_COUNTRY_CODE_CLEAN,
            h.SUPPLIER_COUNTRY_NORM,
            h.WIN_TOWN_CLEAN,
            h.TOWN_NORM,
            h.NAME_COUNTRY_KEY,
            h.NAME_TOWN_KEY,
            h.IS_GROUP_OR_LIST_RECORD,
            h.GROUP_OR_LIST_REASON,
            h.NATIONAL_ID_VALIDITY,

            CASE
                WHEN h.IS_GROUP_OR_LIST_RECORD THEN NULL
                WHEN h.DIRECT_ANCHOR_ENTITY_ID IS NOT NULL
                    THEN h.DIRECT_ANCHOR_ENTITY_ID
                WHEN p.BRIDGE_STATUS = 'ambiguous_multi_anchor_bridge'
                    THEN NULL
                WHEN b.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN b.SUPPLIER_ENTITY_ID
                WHEN h.NAME_COUNTRY_KEY IS NOT NULL
                    THEN 'NC||' || h.NAME_COUNTRY_KEY
                WHEN h.NAME_TOWN_KEY IS NOT NULL
                    THEN 'NT||' || h.NAME_TOWN_KEY
                ELSE NULL
            END AS SUPPLIER_ENTITY_ID,

            CASE
                WHEN h.IS_GROUP_OR_LIST_RECORD THEN 'group_or_multi_supplier'
                WHEN h.DIRECT_ANCHOR_ENTITY_ID IS NOT NULL THEN 'validated_id_anchor'
                WHEN p.BRIDGE_STATUS = 'ambiguous_multi_anchor_bridge'
                    THEN 'ambiguous_name_country'
                WHEN b.SUPPLIER_ENTITY_ID IS NOT NULL THEN 'bridged_name_country'
                WHEN h.NAME_COUNTRY_KEY IS NOT NULL THEN 'standalone_name_country'
                WHEN h.NAME_TOWN_KEY IS NOT NULL THEN 'standalone_name_town'
                ELSE 'unresolved'
            END AS AWARD_RESOLUTION_STATUS,

            CASE
                WHEN h.IS_GROUP_OR_LIST_RECORD
                    THEN 'excluded_from_ordinary_supplier_entity'
                WHEN h.DIRECT_ANCHOR_ENTITY_ID IS NOT NULL
                    THEN 'direct_validated_national_id'
                WHEN p.BRIDGE_STATUS = 'ambiguous_multi_anchor_bridge'
                    THEN 'preserved_ambiguous_name_country'
                WHEN b.SUPPLIER_ENTITY_ID IS NOT NULL
                    THEN 'one_to_one_name_country_bridge'
                WHEN h.NAME_COUNTRY_KEY IS NOT NULL
                    THEN 'standalone_name_country_key'
                WHEN h.NAME_TOWN_KEY IS NOT NULL
                    THEN 'standalone_name_town_key'
                ELSE 'insufficient_identity_information'
            END AS AWARD_RESOLUTION_METHOD,

            CASE
                WHEN h.IS_GROUP_OR_LIST_RECORD THEN 'unresolved'
                WHEN h.DIRECT_ANCHOR_ENTITY_ID IS NOT NULL THEN 'high'
                WHEN p.BRIDGE_STATUS = 'ambiguous_multi_anchor_bridge' THEN 'unresolved'
                WHEN b.SUPPLIER_ENTITY_ID IS NOT NULL THEN 'medium_high'
                WHEN h.NAME_COUNTRY_KEY IS NOT NULL THEN 'medium'
                WHEN h.NAME_TOWN_KEY IS NOT NULL THEN 'low'
                ELSE 'unresolved'
            END AS AWARD_RESOLUTION_CONFIDENCE

        FROM history_awards_id_final h
        LEFT JOIN name_country_bridge_profile p USING (NAME_COUNTRY_KEY)
        LEFT JOIN one_to_one_name_country_bridges b USING (NAME_COUNTRY_KEY)
        """
    )


def build_aliases_and_dimension(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS supplier_aliases")
    con.execute(
        """
        CREATE TABLE supplier_aliases AS
        SELECT
            SUPPLIER_ENTITY_ID,
            WIN_NAME_CLEAN AS ALIAS_NAME_RAW,
            NAME_NORM AS ALIAS_NAME_NORMALIZED,
            MIN(DT_AWARD_CLEAN) AS FIRST_SEEN_AWARD_DATE,
            MAX(DT_AWARD_CLEAN) AS LAST_SEEN_AWARD_DATE,
            MIN(DT_DISPATCH_CLEAN) AS FIRST_SEEN_DISPATCH_DATE,
            MAX(DT_DISPATCH_CLEAN) AS LAST_SEEN_DISPATCH_DATE,
            COUNT(*) AS N_AWARDS,
            COUNT(*) FILTER (WHERE AWARD_RESOLUTION_STATUS = 'validated_id_anchor')
                AS N_DIRECT_ID_AWARDS,
            COUNT(*) FILTER (WHERE AWARD_RESOLUTION_STATUS = 'bridged_name_country')
                AS N_BRIDGED_AWARDS
        FROM award_supplier_map
        WHERE SUPPLIER_ENTITY_ID IS NOT NULL
          AND WIN_NAME_CLEAN IS NOT NULL
        GROUP BY SUPPLIER_ENTITY_ID, WIN_NAME_CLEAN, NAME_NORM
        """
    )

    con.execute("DROP TABLE IF EXISTS supplier_canonical_names")
    con.execute(
        """
        CREATE TABLE supplier_canonical_names AS
        SELECT
            SUPPLIER_ENTITY_ID,
            ALIAS_NAME_RAW AS CANONICAL_NAME,
            ALIAS_NAME_NORMALIZED AS CANONICAL_NAME_NORMALIZED
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY SUPPLIER_ENTITY_ID
                    ORDER BY N_AWARDS DESC,
                             LAST_SEEN_AWARD_DATE DESC NULLS LAST,
                             LENGTH(ALIAS_NAME_RAW) DESC,
                             ALIAS_NAME_RAW
                ) AS RN
            FROM supplier_aliases
        )
        WHERE RN = 1
        """
    )

    con.execute("DROP TABLE IF EXISTS supplier_dim")
    con.execute(
        """
        CREATE TABLE supplier_dim AS
        WITH entity_stats AS (
            SELECT
                SUPPLIER_ENTITY_ID,
                MAX(SUPPLIER_COUNTRY_NORM) AS SUPPLIER_COUNTRY,
                MIN(DT_AWARD_CLEAN) AS FIRST_OBSERVED_AWARD_DATE,
                MAX(DT_AWARD_CLEAN) AS LAST_OBSERVED_AWARD_DATE,
                MIN(DT_DISPATCH_CLEAN) AS FIRST_PUBLICATION_DATE,
                MAX(DT_DISPATCH_CLEAN) AS LAST_PUBLICATION_DATE,
                COUNT(*) AS SOURCE_AWARD_COUNT,
                COUNT(*) FILTER (WHERE AWARD_RESOLUTION_STATUS = 'validated_id_anchor')
                    AS DIRECT_ID_AWARD_COUNT,
                COUNT(*) FILTER (WHERE AWARD_RESOLUTION_STATUS = 'bridged_name_country')
                    AS BRIDGED_NAME_COUNTRY_AWARD_COUNT,
                COUNT(DISTINCT NAME_NORM) FILTER (WHERE NAME_NORM IS NOT NULL)
                    AS N_DISTINCT_NORMALIZED_NAMES,
                COUNT(DISTINCT TOWN_NORM) FILTER (WHERE TOWN_NORM IS NOT NULL)
                    AS N_DISTINCT_TOWNS
            FROM award_supplier_map
            WHERE SUPPLIER_ENTITY_ID IS NOT NULL
            GROUP BY SUPPLIER_ENTITY_ID
        ),
        anchor_info AS (
            SELECT
                SUPPLIER_ENTITY_ID,
                SUPPLIER_COUNTRY_NORM AS ANCHOR_COUNTRY,
                NATIONAL_ID_NORM AS VALIDATED_NATIONAL_ID
            FROM reliable_id_anchors
        )
        SELECT
            e.SUPPLIER_ENTITY_ID,
            c.CANONICAL_NAME,
            c.CANONICAL_NAME_NORMALIZED,
            COALESCE(a.ANCHOR_COUNTRY, e.SUPPLIER_COUNTRY) AS SUPPLIER_COUNTRY,
            CASE
                WHEN a.SUPPLIER_ENTITY_ID IS NOT NULL THEN 'validated_id_anchor'
                WHEN e.SUPPLIER_ENTITY_ID LIKE 'NC||%' THEN 'standalone_name_country'
                WHEN e.SUPPLIER_ENTITY_ID LIKE 'NT||%' THEN 'standalone_name_town'
                ELSE 'other_resolved'
            END AS ENTITY_RESOLUTION_TIER,
            CASE
                WHEN a.SUPPLIER_ENTITY_ID IS NOT NULL THEN 'resolved_anchor'
                WHEN e.SUPPLIER_ENTITY_ID LIKE 'NC||%' THEN 'resolved_observational'
                WHEN e.SUPPLIER_ENTITY_ID LIKE 'NT||%'
                    THEN 'resolved_observational_low_confidence'
                ELSE 'resolved'
            END AS ENTITY_RESOLUTION_STATUS,
            CASE
                WHEN a.SUPPLIER_ENTITY_ID IS NOT NULL THEN 'high'
                WHEN e.SUPPLIER_ENTITY_ID LIKE 'NC||%' THEN 'medium'
                WHEN e.SUPPLIER_ENTITY_ID LIKE 'NT||%' THEN 'low'
                ELSE 'unknown'
            END AS ENTITY_RESOLUTION_CONFIDENCE,
            a.VALIDATED_NATIONAL_ID,
            e.FIRST_OBSERVED_AWARD_DATE,
            e.LAST_OBSERVED_AWARD_DATE,
            e.FIRST_PUBLICATION_DATE,
            e.LAST_PUBLICATION_DATE,
            e.SOURCE_AWARD_COUNT,
            e.DIRECT_ID_AWARD_COUNT,
            e.BRIDGED_NAME_COUNTRY_AWARD_COUNT,
            e.N_DISTINCT_NORMALIZED_NAMES,
            e.N_DISTINCT_TOWNS
        FROM entity_stats e
        LEFT JOIN supplier_canonical_names c USING (SUPPLIER_ENTITY_ID)
        LEFT JOIN anchor_info a USING (SUPPLIER_ENTITY_ID)
        """
    )


def build_temporal_quality(
    con: duckdb.DuckDBPyConnection,
    extreme_publication_lag_days: int,
) -> None:
    con.execute("DROP TABLE IF EXISTS temporal_quality_profile")
    con.execute(
        f"""
        CREATE TABLE temporal_quality_profile AS
        SELECT
            AWARD_KEY,
            CASE
                WHEN DT_AWARD_CLEAN IS NULL THEN 'missing_award_date'
                WHEN DT_DISPATCH_CLEAN IS NULL THEN 'missing_dispatch_date'
                WHEN TRY_CAST(DT_AWARD_CLEAN AS DATE) > DATE '2020-12-31'
                    THEN 'implausible_future_award_date'
                WHEN TRY_CAST(DT_AWARD_CLEAN AS DATE) < DATE '1990-01-01'
                    THEN 'extreme_historical_award_date'
                WHEN PUBLICATION_LAG_DAYS < 0
                    THEN 'negative_publication_lag_review'
                WHEN PUBLICATION_LAG_DAYS = 0 THEN 'same_day'
                WHEN PUBLICATION_LAG_DAYS > {int(extreme_publication_lag_days)}
                    THEN 'extreme_long_publication_lag_review'
                ELSE 'normal_positive'
            END AS TEMPORAL_QUALITY
        FROM history_awards_base
        """
    )


def build_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(metric: str, value, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    add("history_awards_2015_2016", scalar(con, "SELECT COUNT(*) FROM history_awards_base"))
    add(
        "resolved_awards",
        scalar(con, "SELECT COUNT(*) FROM award_supplier_map WHERE SUPPLIER_ENTITY_ID IS NOT NULL"),
    )
    add(
        "unresolved_or_ambiguous_awards",
        scalar(con, "SELECT COUNT(*) FROM award_supplier_map WHERE SUPPLIER_ENTITY_ID IS NULL"),
    )
    add("supplier_entities", scalar(con, "SELECT COUNT(*) FROM supplier_dim"))
    add("reliable_id_anchor_entities", scalar(con, "SELECT COUNT(*) FROM reliable_id_anchors"))
    add(
        "ambiguous_name_country_keys",
        scalar(
            con,
            "SELECT COUNT(*) FROM name_country_bridge_profile "
            "WHERE BRIDGE_STATUS = 'ambiguous_multi_anchor_bridge'",
        ),
    )
    add(
        "one_to_one_name_country_bridge_keys",
        scalar(
            con,
            "SELECT COUNT(*) FROM name_country_bridge_profile "
            "WHERE BRIDGE_STATUS = 'one_to_one_anchor_bridge'",
        ),
    )
    add(
        "group_or_list_awards",
        scalar(
            con,
            "SELECT COUNT(*) FROM award_supplier_map "
            "WHERE AWARD_RESOLUTION_STATUS = 'group_or_multi_supplier'",
        ),
    )

    status_df = fetch_df(
        con,
        """
        SELECT AWARD_RESOLUTION_STATUS, COUNT(*) AS N
        FROM award_supplier_map
        GROUP BY AWARD_RESOLUTION_STATUS
        ORDER BY N DESC
        """,
    )
    for _, row in status_df.iterrows():
        add(f"award_resolution_status__{row['AWARD_RESOLUTION_STATUS']}", int(row["N"]))

    tier_df = fetch_df(
        con,
        """
        SELECT ENTITY_RESOLUTION_TIER, COUNT(*) AS N
        FROM supplier_dim
        GROUP BY ENTITY_RESOLUTION_TIER
        ORDER BY N DESC
        """,
    )
    for _, row in tier_df.iterrows():
        add(f"entity_tier__{row['ENTITY_RESOLUTION_TIER']}", int(row["N"]))

    return pd.DataFrame(rows)


def build_report_tables(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    return {
        "national_id_quality_summary.csv": fetch_df(
            con,
            """
            SELECT NATIONAL_ID_VALIDITY,
                   COUNT(*) AS N_AWARDS,
                   100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS AWARD_SHARE_PCT
            FROM history_awards_id_final
            GROUP BY NATIONAL_ID_VALIDITY
            ORDER BY N_AWARDS DESC
            """,
        ),
        "resolution_method_summary.csv": fetch_df(
            con,
            """
            SELECT AWARD_RESOLUTION_STATUS,
                   AWARD_RESOLUTION_METHOD,
                   AWARD_RESOLUTION_CONFIDENCE,
                   COUNT(*) AS N_AWARDS,
                   100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS AWARD_SHARE_PCT
            FROM award_supplier_map
            GROUP BY AWARD_RESOLUTION_STATUS,
                     AWARD_RESOLUTION_METHOD,
                     AWARD_RESOLUTION_CONFIDENCE
            ORDER BY N_AWARDS DESC
            """,
        ),
        "bridge_status_summary.csv": fetch_df(
            con,
            """
            SELECT BRIDGE_STATUS,
                   COUNT(*) AS N_NAME_COUNTRY_KEYS,
                   SUM(TOTAL_SUPPORTING_AWARDS) AS SUPPORTING_AWARDS
            FROM name_country_bridge_profile
            GROUP BY BRIDGE_STATUS
            ORDER BY N_NAME_COUNTRY_KEYS DESC
            """,
        ),
        "temporal_quality_summary.csv": fetch_df(
            con,
            """
            SELECT TEMPORAL_QUALITY,
                   COUNT(*) AS N_AWARDS,
                   100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS AWARD_SHARE_PCT
            FROM temporal_quality_profile
            GROUP BY TEMPORAL_QUALITY
            ORDER BY N_AWARDS DESC
            """,
        ),
    }


def create_markdown_report(
    metrics: pd.DataFrame,
    output_path: Path,
    min_id_length: int,
    max_names_per_id_anchor: int,
    extreme_publication_lag_days: int,
) -> None:
    metric_map = dict(zip(metrics["metric"], metrics["value"]))

    def mv(name: str, default="N/A"):
        return metric_map.get(name, default)

    lines = [
        "# TED CAN 2015-2016 Historical Supplier Dimension Build Report",
        "",
        "## Scope",
        "",
        "This build uses only CAN award facts whose source-year range is entirely within 2015-2016. "
        "No 2017 award record is used to learn supplier anchors, aliases, or bridges.",
        "",
        "## Build summary",
        "",
        f"- Historical award facts: {mv('history_awards_2015_2016')}",
        f"- Awards assigned to a supplier entity: {mv('resolved_awards')}",
        f"- Awards left ambiguous or unresolved: {mv('unresolved_or_ambiguous_awards')}",
        f"- Supplier entities created: {mv('supplier_entities')}",
        f"- Reliable ID-anchor entities: {mv('reliable_id_anchor_entities')}",
        f"- One-to-one name-country bridge keys: {mv('one_to_one_name_country_bridge_keys')}",
        f"- Ambiguous name-country keys: {mv('ambiguous_name_country_keys')}",
        f"- Group/list-valued awards excluded from ordinary supplier entities: {mv('group_or_list_awards')}",
        "",
        "## Deterministic resolution policy",
        "",
        "1. Explicit group awards and list-valued winner records are separated from ordinary supplier entities.",
        "2. National IDs are normalized and subjected to conservative hygiene checks.",
        f"3. IDs shorter than {min_id_length} normalized characters are not used as automatic anchors.",
        "4. Country-code placeholders and generic placeholder IDs are rejected.",
        f"5. Candidate ID anchors linked to more than {max_names_per_id_anchor} distinct normalized names are excluded from automatic anchoring.",
        "6. Reliable country+ID pairs define high-confidence anchors.",
        "7. Name-country keys mapping to exactly one reliable anchor are bridged to that anchor.",
        "8. Name-country keys mapping to multiple reliable anchors remain ambiguous.",
        "9. Unanchored, non-ambiguous name-country keys become medium-confidence observational entities.",
        "10. Name-town fallback entities are retained at low confidence without cross-town merging.",
        "11. No fuzzy matching is performed.",
        "",
        "## Temporal quality",
        "",
        f"Publication lags greater than {extreme_publication_lag_days} days are flagged for review. "
        "Temporal flags are diagnostic; this script does not yet construct time-valid historical features.",
        "",
        "## Next methodological step",
        "",
        "Profile and clean CN 2017, link target CN records to CAN 2017 outcomes, and then map 2017 winners "
        "to this historical supplier dimension without using 2017 records to relearn identities.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conservative TED CAN 2015-2016 supplier dimension."
    )
    parser.add_argument("--award-fact", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/supplier_history_2015_2016"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/supplier_dimension_2015_2016"),
    )
    parser.add_argument("--work-db", type=Path, default=None)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
    )
    parser.add_argument("--memory-limit", default=None)
    parser.add_argument("--min-id-length", type=int, default=4)
    parser.add_argument("--max-names-per-id-anchor", type=int, default=20)
    parser.add_argument("--extreme-publication-lag-days", type=int, default=730)
    args = parser.parse_args()

    if not args.award_fact.exists():
        raise FileNotFoundError(f"Award fact not found: {args.award_fact}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    work_db = args.work_db or (args.output_dir / "_supplier_dimension_build.duckdb")
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
        print("1/10 Validating award-fact input...")
        validate_input(con, args.award_fact)

        print("2/10 Building strict 2015-2016 history base...")
        build_history_base(con, args.award_fact)

        print("3/10 Applying national-ID hygiene rules...")
        classify_id_hygiene(con, args.min_id_length)

        print("4/10 Profiling and filtering ID anchors...")
        build_anchor_profile_and_reliable_anchors(con, args.max_names_per_id_anchor)

        print("5/10 Learning name-country bridges from history only...")
        build_name_country_bridges(con)

        print("6/10 Resolving historical award-to-supplier assignments...")
        build_award_supplier_map(con)

        print("7/10 Building aliases and supplier dimension...")
        build_aliases_and_dimension(con)

        print("8/10 Building temporal-quality diagnostics...")
        build_temporal_quality(con, args.extreme_publication_lag_days)

        print("9/10 Building metrics and reports...")
        metrics = build_metrics(con)
        write_csv(metrics, args.report_dir / "supplier_dimension_build_metrics.csv")
        for filename, df in build_report_tables(con).items():
            write_csv(df, args.report_dir / filename)

        create_markdown_report(
            metrics,
            args.report_dir / "supplier_dimension_build_report.md",
            args.min_id_length,
            args.max_names_per_id_anchor,
            args.extreme_publication_lag_days,
        )

        print("10/10 Writing Parquet outputs...")
        copy_table_to_parquet(
            con,
            "supplier_dim",
            args.output_dir / "supplier_dim_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "supplier_aliases",
            args.output_dir / "supplier_aliases_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "supplier_ambiguities",
            args.output_dir / "supplier_ambiguities_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "award_supplier_map",
            args.output_dir / "award_supplier_map_2015_2016.parquet",
        )
        copy_table_to_parquet(
            con,
            "national_id_anchor_profile",
            args.output_dir / "national_id_anchor_profile_2015_2016.parquet",
        )

        print()
        print("=" * 78)
        print("Historical supplier dimension build complete")
        print("=" * 78)
        print(f"Processed outputs: {args.output_dir.resolve()}")
        print(f"Reports:           {args.report_dir.resolve()}")
        print(f"Work database:     {work_db.resolve()}")
        print()
        print("Review these first:")
        print("  supplier_dimension_build_report.md")
        print("  supplier_dimension_build_metrics.csv")
        print("  national_id_quality_summary.csv")
        print("  resolution_method_summary.csv")
        print("  bridge_status_summary.csv")
        print("  temporal_quality_summary.csv")
    finally:
        con.close()


if __name__ == "__main__":
    main()
