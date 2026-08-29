#!/usr/bin/env python3
"""
Audit provisional supplier identity resolution for TED CAN award facts.

Input
-----
award_fact_2015_2017.parquet produced by build_can_award_tables.py

Purpose
-------
This script does NOT finalize supplier entity resolution. It audits whether the
provisional supplier-key strategy is sufficiently reliable to support a later
supplier dimension.

It measures:
1. coverage of each provisional resolution method;
2. name-country keys linked to multiple observed national IDs (collision risk);
3. national-ID-country keys linked to multiple normalized names (alias/fragmentation);
4. name-country town dispersion;
5. buyer and CPV activity dispersion;
6. name-country <-> national-ID bridge edges for later controlled linkage;
7. unresolved-key reasons;
8. group-award resolution coverage;
9. year/XSD resolution coverage;
10. conservative review flags without automatically merging or splitting entities.

Outputs
-------
- supplier_resolution_summary.csv
- supplier_resolution_method_coverage.csv
- supplier_resolution_method_by_year.csv
- supplier_resolution_method_by_xsd.csv
- name_country_national_id_edges.csv
- name_country_national_id_collisions.csv
- national_id_name_aliases.csv
- name_country_town_dispersion.csv
- name_country_activity_dispersion.csv
- supplier_resolution_review_flags.csv
- unresolved_reason_summary.csv
- group_award_resolution_coverage.csv
- supplier_resolution_audit_report.md

Methodological cautions
-----------------------
- A name-country key linked to >1 national ID is a strong observed collision signal.
- A national ID linked to >1 name is NOT automatically an error; subsidiaries,
  abbreviations, legal-name changes, and spelling variants may explain it.
- High town/buyer/CPV dispersion is only a review signal, not proof of collision.
- Group awards are audited separately and should not automatically contribute full
  award value to a single supplier history.
- No fuzzy matching is performed here.
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
    "WIN_NAME_CLEAN",
    "WIN_NATIONALID_CLEAN",
    "WIN_TOWN_CLEAN",
    "WIN_COUNTRY_CODE_CLEAN",
    "SUPPLIER_KEY_PROVISIONAL",
    "SUPPLIER_RESOLUTION_METHOD",
    "CAE_NAME_CLEAN",
    "ISO_COUNTRY_CODE_CLEAN",
    "CPV_CLEAN",
    "B_AWARDED_TO_A_GROUP_RAW",
    "B_AWARDED_TO_A_GROUP_CLEAN",
}


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def clean_text_expr(column: str) -> str:
    c = qident(column)
    return f"NULLIF(TRIM(CAST({c} AS VARCHAR)), '')"


def normalize_entity_expr(column: str) -> str:
    """Keep normalization consistent with the upstream CAN build script."""
    c = clean_text_expr(column)
    return (
        "NULLIF(TRIM(REGEXP_REPLACE("
        f"LOWER(COALESCE({c}, '')), "
        "'[^[:alnum:]]+', ' ', 'g')), '')"
    )


def normalize_identifier_expr(column: str) -> str:
    c = clean_text_expr(column)
    return (
        "NULLIF(REGEXP_REPLACE("
        f"UPPER(COALESCE({c}, '')), "
        "'[^[:alnum:]]+', '', 'g'), '')"
    )


def copy_query_to_csv(
    con: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY ({query})
        TO '{sql_path(output_path)}'
        (HEADER, DELIMITER ',')
        """
    )


def fetch_df(con: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    return con.execute(query).fetchdf()


def scalar(con: duckdb.DuckDBPyConnection, query: str):
    return con.execute(query).fetchone()[0]


def validate_input_columns(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
) -> None:
    cols = {
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE SELECT *
            FROM read_parquet('{sql_path(parquet_path)}')
            """
        ).fetchall()
    }
    missing = sorted(REQUIRED_COLUMNS - cols)
    if missing:
        raise ValueError(
            "Input award fact table is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def create_base_views(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW award_fact_input AS
        SELECT *
        FROM read_parquet('{sql_path(parquet_path)}')
        """
    )

    name_norm = normalize_entity_expr("WIN_NAME_CLEAN")
    national_id_norm = normalize_identifier_expr("WIN_NATIONALID_CLEAN")
    town_norm = normalize_entity_expr("WIN_TOWN_CLEAN")
    buyer_norm = normalize_entity_expr("CAE_NAME_CLEAN")
    supplier_country = f"UPPER({clean_text_expr('WIN_COUNTRY_CODE_CLEAN')})"
    buyer_country = f"UPPER({clean_text_expr('ISO_COUNTRY_CODE_CLEAN')})"

    con.execute(
        f"""
        CREATE OR REPLACE VIEW supplier_audit_base AS
        SELECT
            *,
            {name_norm} AS NAME_NORM,
            {national_id_norm} AS NATIONAL_ID_NORM,
            {supplier_country} AS SUPPLIER_COUNTRY_NORM,
            {town_norm} AS TOWN_NORM,
            {buyer_norm} AS BUYER_NAME_NORM,
            {buyer_country} AS BUYER_COUNTRY_NORM,

            CASE
                WHEN {name_norm} IS NOT NULL
                 AND {supplier_country} IS NOT NULL
                THEN {supplier_country} || '||' || {name_norm}
                ELSE NULL
            END AS NAME_COUNTRY_KEY,

            CASE
                WHEN {national_id_norm} IS NOT NULL
                 AND {supplier_country} IS NOT NULL
                THEN {supplier_country} || '||' || {national_id_norm}
                ELSE NULL
            END AS NATIONAL_ID_COUNTRY_KEY,

            CASE
                WHEN {buyer_norm} IS NOT NULL
                 AND {buyer_country} IS NOT NULL
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
                WHEN B_AWARDED_TO_A_GROUP_CLEAN IS TRUE THEN 'group_award'
                WHEN B_AWARDED_TO_A_GROUP_CLEAN IS FALSE THEN 'single_or_not_group'
                ELSE 'unknown_group_status'
            END AS GROUP_AWARD_STATUS
        FROM award_fact_input
        """
    )


def method_coverage_query() -> str:
    return """
    SELECT
        SUPPLIER_RESOLUTION_METHOD,
        COUNT(*) AS n_awards,
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS award_share_pct,
        COUNT(DISTINCT SUPPLIER_KEY_PROVISIONAL) AS n_distinct_provisional_keys
    FROM supplier_audit_base
    GROUP BY SUPPLIER_RESOLUTION_METHOD
    ORDER BY n_awards DESC
    """


def method_by_year_query() -> str:
    return """
    SELECT
        SOURCE_YEAR_MIN AS source_year,
        SUPPLIER_RESOLUTION_METHOD,
        COUNT(*) AS n_awards,
        100.0 * COUNT(*)
            / SUM(COUNT(*)) OVER (PARTITION BY SOURCE_YEAR_MIN)
            AS award_share_pct
    FROM supplier_audit_base
    GROUP BY SOURCE_YEAR_MIN, SUPPLIER_RESOLUTION_METHOD
    ORDER BY source_year, n_awards DESC
    """


def method_by_xsd_query() -> str:
    return """
    SELECT
        SOURCE_YEAR_MIN AS source_year,
        XSD_VERSION_CLEAN,
        SUPPLIER_RESOLUTION_METHOD,
        COUNT(*) AS n_awards,
        100.0 * COUNT(*)
            / SUM(COUNT(*)) OVER (
                PARTITION BY SOURCE_YEAR_MIN, XSD_VERSION_CLEAN
            ) AS award_share_pct
    FROM supplier_audit_base
    GROUP BY
        SOURCE_YEAR_MIN,
        XSD_VERSION_CLEAN,
        SUPPLIER_RESOLUTION_METHOD
    ORDER BY source_year, XSD_VERSION_CLEAN, n_awards DESC
    """


def bridge_edges_query() -> str:
    return """
    SELECT
        NAME_COUNTRY_KEY,
        NATIONAL_ID_COUNTRY_KEY,
        SUPPLIER_COUNTRY_NORM AS supplier_country,
        MIN(WIN_NAME_CLEAN) AS example_winner_name,
        MIN(WIN_NATIONALID_CLEAN) AS example_national_id,
        COUNT(*) AS n_awards,
        MIN(DT_AWARD_CLEAN) AS first_award_date,
        MAX(DT_AWARD_CLEAN) AS last_award_date
    FROM supplier_audit_base
    WHERE
        NAME_COUNTRY_KEY IS NOT NULL
        AND NATIONAL_ID_COUNTRY_KEY IS NOT NULL
    GROUP BY
        NAME_COUNTRY_KEY,
        NATIONAL_ID_COUNTRY_KEY,
        SUPPLIER_COUNTRY_NORM
    ORDER BY n_awards DESC
    """


def name_country_collision_query() -> str:
    return """
    WITH base AS (
        SELECT
            NAME_COUNTRY_KEY,
            SUPPLIER_COUNTRY_NORM AS supplier_country,
            MIN(WIN_NAME_CLEAN) AS example_winner_name,
            COUNT(*) AS n_awards_with_observed_id,
            COUNT(DISTINCT NATIONAL_ID_COUNTRY_KEY) AS n_distinct_national_ids,
            STRING_AGG(DISTINCT NATIONAL_ID_COUNTRY_KEY, ' | ')
                AS observed_national_id_keys,
            MIN(DT_AWARD_CLEAN) AS first_award_date,
            MAX(DT_AWARD_CLEAN) AS last_award_date
        FROM supplier_audit_base
        WHERE
            NAME_COUNTRY_KEY IS NOT NULL
            AND NATIONAL_ID_COUNTRY_KEY IS NOT NULL
        GROUP BY NAME_COUNTRY_KEY, SUPPLIER_COUNTRY_NORM
    )
    SELECT *
    FROM base
    WHERE n_distinct_national_ids > 1
    ORDER BY n_distinct_national_ids DESC, n_awards_with_observed_id DESC
    """


def national_id_alias_query() -> str:
    return """
    WITH base AS (
        SELECT
            NATIONAL_ID_COUNTRY_KEY,
            SUPPLIER_COUNTRY_NORM AS supplier_country,
            MIN(WIN_NATIONALID_CLEAN) AS example_national_id,
            COUNT(*) AS n_awards,
            COUNT(DISTINCT NAME_NORM) AS n_distinct_normalized_names,
            STRING_AGG(DISTINCT NAME_NORM, ' | ')
                AS observed_normalized_names,
            COUNT(DISTINCT TOWN_NORM) FILTER (
                WHERE TOWN_NORM IS NOT NULL
            ) AS n_distinct_towns,
            MIN(DT_AWARD_CLEAN) AS first_award_date,
            MAX(DT_AWARD_CLEAN) AS last_award_date
        FROM supplier_audit_base
        WHERE
            NATIONAL_ID_COUNTRY_KEY IS NOT NULL
            AND NAME_NORM IS NOT NULL
        GROUP BY NATIONAL_ID_COUNTRY_KEY, SUPPLIER_COUNTRY_NORM
    )
    SELECT *
    FROM base
    WHERE n_distinct_normalized_names > 1
    ORDER BY n_distinct_normalized_names DESC, n_awards DESC
    """


def town_dispersion_query() -> str:
    return """
    WITH base AS (
        SELECT
            NAME_COUNTRY_KEY,
            SUPPLIER_COUNTRY_NORM AS supplier_country,
            MIN(WIN_NAME_CLEAN) AS example_winner_name,
            COUNT(*) AS n_awards,
            COUNT(DISTINCT TOWN_NORM) FILTER (
                WHERE TOWN_NORM IS NOT NULL
            ) AS n_distinct_towns,
            STRING_AGG(DISTINCT TOWN_NORM, ' | ') FILTER (
                WHERE TOWN_NORM IS NOT NULL
            ) AS observed_towns,
            COUNT(DISTINCT NATIONAL_ID_COUNTRY_KEY) FILTER (
                WHERE NATIONAL_ID_COUNTRY_KEY IS NOT NULL
            ) AS n_distinct_observed_national_ids,
            MIN(DT_AWARD_CLEAN) AS first_award_date,
            MAX(DT_AWARD_CLEAN) AS last_award_date
        FROM supplier_audit_base
        WHERE NAME_COUNTRY_KEY IS NOT NULL
        GROUP BY NAME_COUNTRY_KEY, SUPPLIER_COUNTRY_NORM
    )
    SELECT *
    FROM base
    WHERE n_distinct_towns > 1
    ORDER BY n_distinct_towns DESC, n_awards DESC
    """


def activity_dispersion_query() -> str:
    return """
    SELECT
        NAME_COUNTRY_KEY,
        SUPPLIER_COUNTRY_NORM AS supplier_country,
        MIN(WIN_NAME_CLEAN) AS example_winner_name,
        COUNT(*) AS n_awards,
        COUNT(DISTINCT BUYER_KEY_AUDIT) FILTER (
            WHERE BUYER_KEY_AUDIT IS NOT NULL
        ) AS n_distinct_buyers,
        COUNT(DISTINCT BUYER_COUNTRY_NORM) FILTER (
            WHERE BUYER_COUNTRY_NORM IS NOT NULL
        ) AS n_distinct_buyer_countries,
        COUNT(DISTINCT CPV2) FILTER (
            WHERE CPV2 IS NOT NULL AND CPV2 <> ''
        ) AS n_distinct_cpv2_groups,
        COUNT(DISTINCT TOWN_NORM) FILTER (
            WHERE TOWN_NORM IS NOT NULL
        ) AS n_distinct_towns,
        COUNT(DISTINCT NATIONAL_ID_COUNTRY_KEY) FILTER (
            WHERE NATIONAL_ID_COUNTRY_KEY IS NOT NULL
        ) AS n_distinct_observed_national_ids,
        MIN(DT_AWARD_CLEAN) AS first_award_date,
        MAX(DT_AWARD_CLEAN) AS last_award_date
    FROM supplier_audit_base
    WHERE NAME_COUNTRY_KEY IS NOT NULL
    GROUP BY NAME_COUNTRY_KEY, SUPPLIER_COUNTRY_NORM
    ORDER BY n_awards DESC
    """


def review_flags_query(
    min_awards: int,
    high_towns: int,
    high_buyers: int,
    high_cpv2: int,
) -> str:
    activity = activity_dispersion_query().strip().rstrip(';')
    return f"""
    WITH activity AS (
        {activity}
    )
    SELECT
        *,
        CASE
            WHEN n_distinct_observed_national_ids > 1
                THEN 'observed_id_collision'
            WHEN n_awards >= {int(min_awards)}
             AND (
                    n_distinct_towns >= {int(high_towns)}
                 OR n_distinct_buyers >= {int(high_buyers)}
                 OR n_distinct_cpv2_groups >= {int(high_cpv2)}
             )
                THEN 'high_dispersion_review'
            WHEN n_awards = 1
             AND n_distinct_observed_national_ids = 0
                THEN 'low_information_key'
            ELSE 'no_automatic_flag'
        END AS review_flag
    FROM activity
    ORDER BY
        CASE review_flag
            WHEN 'observed_id_collision' THEN 1
            WHEN 'high_dispersion_review' THEN 2
            WHEN 'low_information_key' THEN 3
            ELSE 4
        END,
        n_awards DESC
    """


def unresolved_reason_query() -> str:
    return """
    SELECT
        CASE
            WHEN NAME_NORM IS NULL
                THEN 'missing_name_after_normalization'
            WHEN SUPPLIER_COUNTRY_NORM IS NULL AND TOWN_NORM IS NULL
                THEN 'missing_country_and_town'
            WHEN SUPPLIER_COUNTRY_NORM IS NULL
                THEN 'missing_country'
            ELSE 'other'
        END AS unresolved_reason,
        COUNT(*) AS n_awards
    FROM supplier_audit_base
    WHERE SUPPLIER_RESOLUTION_METHOD = 'unresolved'
    GROUP BY unresolved_reason
    ORDER BY n_awards DESC
    """


def group_award_coverage_query() -> str:
    return """
    SELECT
        GROUP_AWARD_STATUS,
        SUPPLIER_RESOLUTION_METHOD,
        COUNT(*) AS n_awards,
        100.0 * COUNT(*)
            / SUM(COUNT(*)) OVER (PARTITION BY GROUP_AWARD_STATUS)
            AS share_within_group_status_pct
    FROM supplier_audit_base
    GROUP BY GROUP_AWARD_STATUS, SUPPLIER_RESOLUTION_METHOD
    ORDER BY GROUP_AWARD_STATUS, n_awards DESC
    """


def build_summary(
    con: duckdb.DuckDBPyConnection,
    min_awards: int,
    high_towns: int,
    high_buyers: int,
    high_cpv2: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(metric: str, value, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    total_awards = scalar(con, "SELECT COUNT(*) FROM supplier_audit_base")
    total_keys = scalar(
        con,
        """
        SELECT COUNT(DISTINCT SUPPLIER_KEY_PROVISIONAL)
        FROM supplier_audit_base
        WHERE SUPPLIER_KEY_PROVISIONAL IS NOT NULL
        """,
    )
    add("total_awards", total_awards)
    add("distinct_provisional_supplier_keys", total_keys)

    method_df = fetch_df(con, method_coverage_query())
    for _, row in method_df.iterrows():
        method = row["SUPPLIER_RESOLUTION_METHOD"]
        add(f"method_awards__{method}", int(row["n_awards"]))
        add(f"method_award_share_pct__{method}", float(row["award_share_pct"]))

    eligible_name_country_keys = scalar(
        con,
        """
        SELECT COUNT(DISTINCT NAME_COUNTRY_KEY)
        FROM supplier_audit_base
        WHERE NAME_COUNTRY_KEY IS NOT NULL
          AND NATIONAL_ID_COUNTRY_KEY IS NOT NULL
        """,
    )
    collision_keys = scalar(
        con,
        f"SELECT COUNT(*) FROM ({name_country_collision_query()})",
    )
    collision_awards = scalar(
        con,
        f"""
        SELECT COALESCE(SUM(n_awards_with_observed_id), 0)
        FROM ({name_country_collision_query()})
        """,
    )
    add("eligible_name_country_keys_with_observed_id", eligible_name_country_keys)
    add("name_country_collision_keys", collision_keys)
    add(
        "name_country_collision_key_rate_pct",
        100.0 * collision_keys / eligible_name_country_keys
        if eligible_name_country_keys else None,
    )
    add("awards_on_collision_name_country_keys_with_observed_id", collision_awards)

    national_id_keys = scalar(
        con,
        """
        SELECT COUNT(DISTINCT NATIONAL_ID_COUNTRY_KEY)
        FROM supplier_audit_base
        WHERE NATIONAL_ID_COUNTRY_KEY IS NOT NULL
        """,
    )
    alias_id_keys = scalar(
        con,
        f"SELECT COUNT(*) FROM ({national_id_alias_query()})",
    )
    add("observed_national_id_country_keys", national_id_keys)
    add("national_id_keys_with_multiple_names", alias_id_keys)
    add(
        "national_id_alias_key_rate_pct",
        100.0 * alias_id_keys / national_id_keys if national_id_keys else None,
    )

    flags_q = review_flags_query(min_awards, high_towns, high_buyers, high_cpv2)
    flag_counts = fetch_df(
        con,
        f"""
        SELECT review_flag, COUNT(*) AS n_keys
        FROM ({flags_q})
        GROUP BY review_flag
        ORDER BY n_keys DESC
        """,
    )
    for _, row in flag_counts.iterrows():
        add(f"review_flag_keys__{row['review_flag']}", int(row["n_keys"]))

    group_awards = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM supplier_audit_base
        WHERE B_AWARDED_TO_A_GROUP_CLEAN IS TRUE
        """,
    )
    add("group_awards_explicit_true", group_awards)
    add(
        "group_award_share_pct",
        100.0 * group_awards / total_awards if total_awards else None,
    )

    date_row = con.execute(
        """
        SELECT
            MIN(DT_AWARD_CLEAN), MAX(DT_AWARD_CLEAN),
            MIN(DT_DISPATCH_CLEAN), MAX(DT_DISPATCH_CLEAN)
        FROM supplier_audit_base
        """
    ).fetchone()
    add("min_award_date", date_row[0])
    add("max_award_date", date_row[1])
    add("min_dispatch_date", date_row[2])
    add("max_dispatch_date", date_row[3])

    return pd.DataFrame(rows)


def create_markdown_report(
    summary: pd.DataFrame,
    method_df: pd.DataFrame,
    output_path: Path,
    min_awards: int,
    high_towns: int,
    high_buyers: int,
    high_cpv2: int,
) -> None:
    metric_map = dict(zip(summary["metric"], summary["value"]))

    def mv(name: str, default="N/A"):
        return metric_map.get(name, default)

    lines = [
        "# TED CAN Supplier Resolution Audit Report",
        "",
        "## Scope",
        "",
        "This report audits the provisional supplier identity strategy. It does not declare the provisional key to be final ground truth.",
        "",
        "## Dataset",
        "",
        f"- Award facts audited: {mv('total_awards')}",
        f"- Distinct non-null provisional supplier keys: {mv('distinct_provisional_supplier_keys')}",
        f"- Award-date range: {mv('min_award_date')} to {mv('max_award_date')}",
        "",
        "## Resolution-method coverage",
        "",
        "| Method | Awards | Share (%) | Distinct provisional keys |",
        "|---|---:|---:|---:|",
    ]

    for _, row in method_df.iterrows():
        lines.append(
            f"| `{row['SUPPLIER_RESOLUTION_METHOD']}` "
            f"| {int(row['n_awards'])} "
            f"| {float(row['award_share_pct']):.3f} "
            f"| {int(row['n_distinct_provisional_keys'])} |"
        )

    lines.extend([
        "",
        "## Observed name-country collision risk",
        "",
        f"- Name-country keys with an observed national ID: {mv('eligible_name_country_keys_with_observed_id')}",
        f"- Name-country keys linked to more than one observed national ID: {mv('name_country_collision_keys')}",
        f"- Collision-key rate among ID-observed name-country keys: {mv('name_country_collision_key_rate_pct')}%",
        "",
        "A multi-ID name-country key is a strong observed collision signal. Those keys should not be merged automatically in the supplier dimension.",
        "",
        "## National-ID alias fragmentation",
        "",
        f"- Observed national-ID-country keys: {mv('observed_national_id_country_keys')}",
        f"- National-ID-country keys linked to multiple normalized names: {mv('national_id_keys_with_multiple_names')}",
        f"- Alias-key rate: {mv('national_id_alias_key_rate_pct')}%",
        "",
        "Multiple names per national ID indicate aliasing or name variation. They are not automatically errors and should be used as controlled bridge evidence.",
        "",
        "## Review-flag thresholds",
        "",
        f"- High-dispersion review requires at least {min_awards} awards and at least one of:",
        f"  - {high_towns} distinct towns;",
        f"  - {high_buyers} distinct buyers;",
        f"  - {high_cpv2} distinct CPV 2-digit groups.",
        "",
        "High dispersion is only a review signal. It is not proof that a key represents multiple legal entities.",
        "",
        "## Group awards",
        "",
        f"- Explicit group-award records: {mv('group_awards_explicit_true')}",
        f"- Share of all audited award facts: {mv('group_award_share_pct')}%",
        "",
        "Do not allocate the full award value of a group award to one supplier history unless member-level allocation is known.",
        "",
        "## Recommended interpretation",
        "",
        "1. Treat observed name-country -> multiple national-ID cases as collision exclusions from automatic merge rules.",
        "2. Use national-ID -> multiple name links to construct alias candidate groups, but keep provenance.",
        "3. Review high-dispersion keys manually or with conservative rule-based checks before final supplier aggregation.",
        "4. Exclude or separately handle group awards in supplier-level value histories.",
        "5. Build a supplier dimension only after reviewing the collision and alias outputs.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit provisional supplier identity resolution for TED CAN awards."
    )
    parser.add_argument(
        "--award-fact", required=True, type=Path,
        help="Path to award_fact_2015_2017.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/supplier_resolution_audit"),
    )
    parser.add_argument(
        "--work-db", type=Path, default=None,
        help="Optional DuckDB work database path.",
    )
    parser.add_argument(
        "--threads", type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
    )
    parser.add_argument(
        "--memory-limit", default=None,
        help="Optional DuckDB memory limit, e.g. 8GB.",
    )
    parser.add_argument("--min-awards-for-dispersion-review", type=int, default=20)
    parser.add_argument("--high-town-threshold", type=int, default=5)
    parser.add_argument("--high-buyer-threshold", type=int, default=50)
    parser.add_argument("--high-cpv2-threshold", type=int, default=10)
    args = parser.parse_args()

    if not args.award_fact.exists():
        raise FileNotFoundError(f"Award fact file not found: {args.award_fact}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_db = args.work_db or args.output_dir / "_supplier_resolution_audit.duckdb"
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
        print("1/8 Validating award-fact input...")
        validate_input_columns(con, args.award_fact)

        print("2/8 Creating normalized audit views...")
        create_base_views(con, args.award_fact)

        print("3/8 Writing resolution-method coverage...")
        method_df = fetch_df(con, method_coverage_query())
        method_df.to_csv(args.output_dir / "supplier_resolution_method_coverage.csv", index=False)
        fetch_df(con, method_by_year_query()).to_csv(
            args.output_dir / "supplier_resolution_method_by_year.csv", index=False
        )
        fetch_df(con, method_by_xsd_query()).to_csv(
            args.output_dir / "supplier_resolution_method_by_xsd.csv", index=False
        )

        print("4/8 Writing name-ID bridge and collision diagnostics...")
        copy_query_to_csv(con, bridge_edges_query(), args.output_dir / "name_country_national_id_edges.csv")
        copy_query_to_csv(con, name_country_collision_query(), args.output_dir / "name_country_national_id_collisions.csv")
        copy_query_to_csv(con, national_id_alias_query(), args.output_dir / "national_id_name_aliases.csv")

        print("5/8 Writing town and activity dispersion diagnostics...")
        copy_query_to_csv(con, town_dispersion_query(), args.output_dir / "name_country_town_dispersion.csv")
        copy_query_to_csv(con, activity_dispersion_query(), args.output_dir / "name_country_activity_dispersion.csv")

        print("6/8 Writing conservative review flags...")
        flags_q = review_flags_query(
            args.min_awards_for_dispersion_review,
            args.high_town_threshold,
            args.high_buyer_threshold,
            args.high_cpv2_threshold,
        )
        copy_query_to_csv(con, flags_q, args.output_dir / "supplier_resolution_review_flags.csv")

        print("7/8 Writing unresolved and group-award diagnostics...")
        fetch_df(con, unresolved_reason_query()).to_csv(
            args.output_dir / "unresolved_reason_summary.csv", index=False
        )
        fetch_df(con, group_award_coverage_query()).to_csv(
            args.output_dir / "group_award_resolution_coverage.csv", index=False
        )

        print("8/8 Building summary and report...")
        summary_df = build_summary(
            con,
            args.min_awards_for_dispersion_review,
            args.high_town_threshold,
            args.high_buyer_threshold,
            args.high_cpv2_threshold,
        )
        summary_df.to_csv(args.output_dir / "supplier_resolution_summary.csv", index=False)
        create_markdown_report(
            summary_df,
            method_df,
            args.output_dir / "supplier_resolution_audit_report.md",
            args.min_awards_for_dispersion_review,
            args.high_town_threshold,
            args.high_buyer_threshold,
            args.high_cpv2_threshold,
        )

        print()
        print("=" * 78)
        print("Supplier resolution audit complete")
        print("=" * 78)
        print(f"Reports: {args.output_dir.resolve()}")
        print()
        print("Review these first:")
        print("  supplier_resolution_audit_report.md")
        print("  supplier_resolution_summary.csv")
        print("  name_country_national_id_collisions.csv")
        print("  national_id_name_aliases.csv")
        print("  supplier_resolution_review_flags.csv")
        print("  group_award_resolution_coverage.csv")
        print()
    finally:
        con.close()


if __name__ == "__main__":
    main()
