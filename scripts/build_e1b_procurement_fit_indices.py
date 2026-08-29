#!/usr/bin/env python3
"""
Build compact E1b procurement-fit indices.

Purpose
-------
Construct analysis-layer historical fit indices after the E1b feasibility
audit. The outputs are compact supplier-by-attribute count tables used by the
subsequent E1b ranking experiment.

Primary cohort
--------------
Only target cases appearing in the primary E1 CPV2_MIN1 TOPSIS cohort are
written to the target-profile table. Historical indices remain based only on
the frozen 2015-2016 award-supplier map.

Built fit dimensions
--------------------
- CPV3
- CPV4
- BUYER_KEY
- PROCUREMENT_COUNTRY
- CONTRACT_TYPE
- MAIN_ACTIVITY
- AUTHORITY_TYPE
- PROCEDURE_TYPE

No 2017 outcome is used to construct historical fit counts.

Outputs
-------
Processed:
- e1b_target_profiles_2017.parquet
- supplier_cpv3_fit_counts_2015_2016.parquet
- supplier_cpv4_fit_counts_2015_2016.parquet
- supplier_buyer_fit_counts_2015_2016.parquet
- supplier_country_fit_counts_2015_2016.parquet
- supplier_contract_type_fit_counts_2015_2016.parquet
- supplier_main_activity_fit_counts_2015_2016.parquet
- supplier_authority_type_fit_counts_2015_2016.parquet
- supplier_procedure_type_fit_counts_2015_2016.parquet

Reports:
- e1b_fit_index_metrics.csv
- e1b_fit_index_validation.csv
- e1b_fit_index_report.md

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


def upper_clean(expr: str) -> str:
    return (
        f"UPPER(NULLIF(TRIM(CAST({expr} AS VARCHAR)), ''))"
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


def cpv_prefix(expr: str, n: int) -> str:
    digits = (
        "REGEXP_REPLACE("
        f"COALESCE(CAST({expr} AS VARCHAR), ''), "
        "'[^0-9]', '', 'g'"
        ")"
    )
    return (
        f"CASE WHEN LENGTH({digits}) >= {n} "
        f"THEN SUBSTR({digits}, 1, {n}) "
        "ELSE NULL END"
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


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def validate_inputs(
    con: duckdb.DuckDBPyConnection,
    args,
) -> None:
    require_columns(
        con,
        args.target_cases,
        {
            "TARGET_CASE_ID",
            "TARGET_DISPATCH_DATE",
            "TARGET_CPV2",
            "TARGET_CPV3",
            "TARGET_CPV4",
            "TARGET_PROCUREMENT_COUNTRY",
            "OBSERVED_WINNER_HISTORICAL_ENTITY_ID",
        },
        "target_case_features",
    )

    require_columns(
        con,
        args.procurement_cases,
        {
            "ID_NOTICE_CN_CLEAN",
            "CAE_NAME_CLEAN",
            "CAE_NATIONALID_CLEAN",
            "ISO_COUNTRY_CODE_CLEAN",
            "TYPE_OF_CONTRACT_CLEAN",
            "MAIN_ACTIVITY_CLEAN",
            "CAE_TYPE_CLEAN",
            "TOP_TYPE_CLEAN",
        },
        "procurement_cases",
    )

    require_columns(
        con,
        args.award_supplier_map,
        {
            "AWARD_KEY",
            "SUPPLIER_ENTITY_ID",
        },
        "award_supplier_map",
    )

    require_columns(
        con,
        args.award_fact,
        {
            "AWARD_KEY",
            "CPV_CLEAN",
            "CAE_NAME_CLEAN",
            "CAE_NATIONALID_CLEAN",
            "ISO_COUNTRY_CODE_CLEAN",
            "TYPE_OF_CONTRACT_CLEAN",
            "MAIN_ACTIVITY_CLEAN",
            "CAE_TYPE_CLEAN",
            "TOP_TYPE_CLEAN",
        },
        "award_fact",
    )

    require_columns(
        con,
        args.e1_case_results,
        {
            "TARGET_CASE_ID",
            "POOL_DEFINITION",
            "METHOD",
        },
        "E1 case results",
    )


# ---------------------------------------------------------------------
# Build base views
# ---------------------------------------------------------------------

def create_base_views(
    con: duckdb.DuckDBPyConnection,
    args,
) -> None:
    target_buyer_key = buyer_key_sql(
        "p.ISO_COUNTRY_CODE_CLEAN",
        "p.CAE_NATIONALID_CLEAN",
        "p.CAE_NAME_CLEAN",
    )

    hist_buyer_key = buyer_key_sql(
        "f.ISO_COUNTRY_CODE_CLEAN",
        "f.CAE_NATIONALID_CLEAN",
        "f.CAE_NAME_CLEAN",
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW e1_primary_cases AS
        SELECT DISTINCT
            CAST(TARGET_CASE_ID AS VARCHAR)
                AS TARGET_CASE_ID
        FROM read_parquet('{sql_path(args.e1_case_results)}')
        WHERE POOL_DEFINITION = 'CPV2_MIN1'
          AND METHOD = 'TOPSIS'
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW target_profile_base AS
        SELECT
            CAST(t.TARGET_CASE_ID AS VARCHAR)
                AS TARGET_CASE_ID,

            CAST(t.TARGET_DISPATCH_DATE AS DATE)
                AS TARGET_DISPATCH_DATE,

            CAST(t.TARGET_CPV2 AS VARCHAR)
                AS TARGET_CPV2,

            CAST(t.TARGET_CPV3 AS VARCHAR)
                AS TARGET_CPV3,

            CAST(t.TARGET_CPV4 AS VARCHAR)
                AS TARGET_CPV4,

            {target_buyer_key}
                AS TARGET_BUYER_KEY,

            {valid_country("p.ISO_COUNTRY_CODE_CLEAN")}
                AS TARGET_PROCUREMENT_COUNTRY,

            {upper_clean("p.TYPE_OF_CONTRACT_CLEAN")}
                AS TARGET_CONTRACT_TYPE,

            {upper_clean("p.MAIN_ACTIVITY_CLEAN")}
                AS TARGET_MAIN_ACTIVITY,

            {upper_clean("p.CAE_TYPE_CLEAN")}
                AS TARGET_AUTHORITY_TYPE,

            {upper_clean("p.TOP_TYPE_CLEAN")}
                AS TARGET_PROCEDURE_TYPE,

            CAST(
                t.OBSERVED_WINNER_HISTORICAL_ENTITY_ID
                AS VARCHAR
            ) AS OBSERVED_WINNER_HISTORICAL_ENTITY_ID

        FROM read_parquet('{sql_path(args.target_cases)}') t
        JOIN e1_primary_cases e
          ON CAST(t.TARGET_CASE_ID AS VARCHAR)
           = e.TARGET_CASE_ID
        JOIN read_parquet('{sql_path(args.procurement_cases)}') p
          ON CAST(p.ID_NOTICE_CN_CLEAN AS VARCHAR)
           = CAST(t.TARGET_CASE_ID AS VARCHAR)
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW hist_fit_base AS
        SELECT
            CAST(m.SUPPLIER_ENTITY_ID AS VARCHAR)
                AS SUPPLIER_ENTITY_ID,

            {cpv_prefix("f.CPV_CLEAN", 3)}
                AS CPV3,

            {cpv_prefix("f.CPV_CLEAN", 4)}
                AS CPV4,

            {hist_buyer_key}
                AS BUYER_KEY,

            {valid_country("f.ISO_COUNTRY_CODE_CLEAN")}
                AS PROCUREMENT_COUNTRY,

            {upper_clean("f.TYPE_OF_CONTRACT_CLEAN")}
                AS CONTRACT_TYPE,

            {upper_clean("f.MAIN_ACTIVITY_CLEAN")}
                AS MAIN_ACTIVITY,

            {upper_clean("f.CAE_TYPE_CLEAN")}
                AS AUTHORITY_TYPE,

            {upper_clean("f.TOP_TYPE_CLEAN")}
                AS PROCEDURE_TYPE

        FROM read_parquet('{sql_path(args.award_supplier_map)}') m
        JOIN read_parquet('{sql_path(args.award_fact)}') f
          USING (AWARD_KEY)
        WHERE m.SUPPLIER_ENTITY_ID IS NOT NULL
        """
    )


# ---------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------

INDEX_SPECS = [
    {
        "name": "CPV3",
        "value_col": "CPV3",
        "count_col": "N_CPV3_AWARDS",
        "filename": "supplier_cpv3_fit_counts_2015_2016.parquet",
    },
    {
        "name": "CPV4",
        "value_col": "CPV4",
        "count_col": "N_CPV4_AWARDS",
        "filename": "supplier_cpv4_fit_counts_2015_2016.parquet",
    },
    {
        "name": "BUYER_KEY",
        "value_col": "BUYER_KEY",
        "count_col": "N_PRIOR_BUYER_AWARDS",
        "filename": "supplier_buyer_fit_counts_2015_2016.parquet",
    },
    {
        "name": "PROCUREMENT_COUNTRY",
        "value_col": "PROCUREMENT_COUNTRY",
        "count_col": "N_COUNTRY_AWARDS",
        "filename": "supplier_country_fit_counts_2015_2016.parquet",
    },
    {
        "name": "CONTRACT_TYPE",
        "value_col": "CONTRACT_TYPE",
        "count_col": "N_CONTRACT_TYPE_AWARDS",
        "filename": "supplier_contract_type_fit_counts_2015_2016.parquet",
    },
    {
        "name": "MAIN_ACTIVITY",
        "value_col": "MAIN_ACTIVITY",
        "count_col": "N_MAIN_ACTIVITY_AWARDS",
        "filename": "supplier_main_activity_fit_counts_2015_2016.parquet",
    },
    {
        "name": "AUTHORITY_TYPE",
        "value_col": "AUTHORITY_TYPE",
        "count_col": "N_AUTHORITY_TYPE_AWARDS",
        "filename": "supplier_authority_type_fit_counts_2015_2016.parquet",
    },
    {
        "name": "PROCEDURE_TYPE",
        "value_col": "PROCEDURE_TYPE",
        "count_col": "N_PROCEDURE_TYPE_AWARDS",
        "filename": "supplier_procedure_type_fit_counts_2015_2016.parquet",
    },
]


def build_indices(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> pd.DataFrame:
    metrics = []

    for spec in INDEX_SPECS:
        value_col = spec["value_col"]
        count_col = spec["count_col"]
        output_path = output_dir / spec["filename"]

        print(f"    building {spec['name']} index...")

        query = f"""
            SELECT
                CAST({value_col} AS VARCHAR)
                    AS FIT_VALUE,
                SUPPLIER_ENTITY_ID,
                COUNT(*)::BIGINT
                    AS {count_col}
            FROM hist_fit_base
            WHERE {value_col} IS NOT NULL
            GROUP BY
                FIT_VALUE,
                SUPPLIER_ENTITY_ID
        """

        copy_query_to_parquet(
            con,
            query,
            output_path,
        )

        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT FIT_VALUE) AS n_values,
                COUNT(DISTINCT SUPPLIER_ENTITY_ID) AS n_suppliers,
                SUM({count_col}) AS n_awards_represented
            FROM read_parquet('{sql_path(output_path)}')
            """
        ).fetchone()

        metrics.append(
            {
                "INDEX_NAME": spec["name"],
                "FILE": spec["filename"],
                "N_ROWS": int(row[0]),
                "N_DISTINCT_VALUES": int(row[1]),
                "N_DISTINCT_SUPPLIERS": int(row[2]),
                "N_HISTORICAL_AWARDS_REPRESENTED":
                    int(row[3]) if row[3] is not None else 0,
            }
        )

    return pd.DataFrame(metrics)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validation_checks(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []

    target_path = output_dir / "e1b_target_profiles_2017.parquet"

    n_target_rows = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_path(target_path)}')
            """
        ).fetchone()[0]
    )

    n_target_ids = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT TARGET_CASE_ID)
            FROM read_parquet('{sql_path(target_path)}')
            """
        ).fetchone()[0]
    )

    rows.append(
        {
            "CHECK": "target_profile_unique_case_rows",
            "VALUE": n_target_rows == n_target_ids,
            "DETAIL":
                f"rows={n_target_rows}; unique_cases={n_target_ids}",
        }
    )

    for spec in INDEX_SPECS:
        path = output_dir / spec["filename"]
        count_col = spec["count_col"]

        dup_count = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT
                        FIT_VALUE,
                        SUPPLIER_ENTITY_ID,
                        COUNT(*) AS n
                    FROM read_parquet('{sql_path(path)}')
                    GROUP BY FIT_VALUE, SUPPLIER_ENTITY_ID
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )

        non_positive = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_path(path)}')
                WHERE {count_col} <= 0
                """
            ).fetchone()[0]
        )

        rows.append(
            {
                "CHECK":
                    f"{spec['name']}__unique_value_supplier_rows",
                "VALUE": dup_count == 0,
                "DETAIL": f"duplicate_groups={dup_count}",
            }
        )

        rows.append(
            {
                "CHECK":
                    f"{spec['name']}__positive_counts",
                "VALUE": non_positive == 0,
                "DETAIL": f"non_positive_rows={non_positive}",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_report(
    metrics_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# E1b Procurement-Fit Index Build Report",
        "",
        "## Purpose",
        "",
        "This build creates compact 2015-2016 historical supplier-by-attribute "
        "count indices for the E1b procurement-fit ranking experiment.",
        "",
        "No 2017 outcome information is used to construct historical counts.",
        "",
        "## Index sizes",
        "",
        "| Index | Rows | Distinct values | Suppliers | Historical awards represented |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| `{row['INDEX_NAME']}` "
            f"| {int(row['N_ROWS'])} "
            f"| {int(row['N_DISTINCT_VALUES'])} "
            f"| {int(row['N_DISTINCT_SUPPLIERS'])} "
            f"| {int(row['N_HISTORICAL_AWARDS_REPRESENTED'])} |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
        ]
    )

    for _, row in validation_df.iterrows():
        status = "PASS" if bool(row["VALUE"]) else "FAIL"
        lines.append(
            f"- **{status}** `{row['CHECK']}`: {row['DETAIL']}"
        )

    lines.extend(
        [
            "",
            "## Next analytical step",
            "",
            "The next script should compare nested TOPSIS specifications on the "
            "same CPV2_MIN1 winner-in-pool cohort: original E1 features, "
            "hierarchical CPV fit, relationship/geographic fit, and full "
            "procurement-context fit.",
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-cases",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--procurement-cases",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--award-supplier-map",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--award-fact",
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
        default=Path("data/analysis/e1b_fit_indices"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/e1b_fit_indices"),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
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
        args.target_cases,
        args.procurement_cases,
        args.award_supplier_map,
        args.award_fact,
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

    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")

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
        print("1/6 Validating inputs...")
        validate_inputs(con, args)

        print("2/6 Creating base views...")
        create_base_views(con, args)

        print("3/6 Writing target profile table...")
        copy_query_to_parquet(
            con,
            """
            SELECT *
            FROM target_profile_base
            """,
            args.output_dir
            / "e1b_target_profiles_2017.parquet",
        )

        print("4/6 Building historical fit indices...")
        metrics_df = build_indices(
            con,
            args.output_dir,
        )

        print("5/6 Running validation...")
        validation_df = validation_checks(
            con,
            args.output_dir,
        )

        print("6/6 Writing reports...")
        metrics_df.to_csv(
            args.report_dir
            / "e1b_fit_index_metrics.csv",
            index=False,
        )
        validation_df.to_csv(
            args.report_dir
            / "e1b_fit_index_validation.csv",
            index=False,
        )

        write_report(
            metrics_df,
            validation_df,
            args.report_dir
            / "e1b_fit_index_report.md",
        )

        print()
        print("E1b fit-index build complete.")
        print("Review:")
        print("  e1b_fit_index_report.md")
        print("  e1b_fit_index_metrics.csv")
        print("  e1b_fit_index_validation.csv")

    finally:
        con.close()


if __name__ == "__main__":
    main()
