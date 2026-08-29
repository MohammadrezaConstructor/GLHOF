#!/usr/bin/env python3
"""
E3d: reasoning-effort ablation wrapper for Reviewer 2.2.

This wrapper invokes the already validated E3c experiment with one additional
configuration: strict few-shot prompting with reasoning_effort='none'. The
strict few-shot low-effort results are reused through checkpoints, so the default
20-case run should require only about 20 new API calls.

No API call is made unless --run-api is supplied.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base-cases", type=int, default=20)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    runner = root / "scripts/run_r22_e3c_prompt_schema_ablation.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"Missing validated E3c runner: {runner}. Copy the verified R2.2 suite "
            "into the scripts directory first."
        )

    command = [
        sys.executable,
        str(runner),
        "--root",
        str(root),
        "--base-cases",
        str(args.base_cases),
        "--model",
        args.model,
        "--include-reasoning-ablation",
    ]
    if args.run_api:
        command.append("--run-api")
    if args.resume:
        command.append("--resume")

    subprocess.run(command, check=True)

    comparison_path = (
        root / "reports/r22_e3c_ablation/r22_e3c_configuration_comparison.csv"
    )
    if not comparison_path.exists():
        if args.run_api:
            raise FileNotFoundError(
                f"Expected E3c comparison output was not produced: {comparison_path}"
            )
        print("Reasoning-ablation runner prepared; no API calls were made.")
        return

    comparison = pd.read_csv(comparison_path)
    if args.run_api and "STRICT_FEW_SHOT_NONE" not in set(
        comparison["CONFIGURATION"].astype(str)
    ):
        raise ValueError(
            "Reasoning-ablation output does not contain STRICT_FEW_SHOT_NONE."
        )
    print(f"E3d complete. Comparison: {comparison_path}")


if __name__ == "__main__":
    main()
