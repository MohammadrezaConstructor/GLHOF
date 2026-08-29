"""Read-only repository inventory and machine-checkable audit extraction.

The helper reads repository content and writes only beneath the requested audit
output directory. Research findings are recorded as data; they do not cause a
nonzero exit code unless the helper itself cannot complete.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}
TRANSIENT_FILE_PATTERNS = (
    re.compile(r"^~\$"),
    re.compile(r"\.sw[op]$", re.IGNORECASE),
    re.compile(r"\.tmp$", re.IGNORECASE),
)
TEXT_REFERENCE_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".ps1",
    ".bat",
    ".sh",
}
STRUCTURED_EXTENSIONS = {
    ".csv",
    ".parquet",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xlsx",
    ".xls",
    ".duckdb",
}
HASH_PRIORITY_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".ps1",
    ".bat",
    ".sh",
    ".csv",
}
DATA_FILE_EXTENSIONS = {
    ".csv",
    ".parquet",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xlsx",
    ".xls",
    ".duckdb",
    ".joblib",
}
PATH_SUFFIXES = tuple(DATA_FILE_EXTENSIONS | {".md", ".txt", ".log", ".py"})
PATH_TOKEN_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\() \-]+\.(?:csv|parquet|jsonl?|ya?ml|xlsx?|duckdb|joblib|md|txt|log|py))",
    re.IGNORECASE,
)
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
JSON_LIKE_RE = re.compile(r"^\s*[\[{]")
DATE_NAME_RE = re.compile(r"date|time|dispatch|award|start|completion", re.IGNORECASE)
YEAR_NAME_RE = re.compile(r"year", re.IGNORECASE)
ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|case_id|packet_id|request_id|supplier_id|notice_id)", re.IGNORECASE)
OUTCOME_NAME_RE = re.compile(r"winner|reference|outcome|target|rank|awardee|label", re.IGNORECASE)


@dataclass(frozen=True)
class AuditOptions:
    root: Path
    output_dir: Path
    max_hash_bytes: int
    sample_rows: int
    full_benchmark_scan: bool
    check_python_ast: bool
    check_parquet: bool
    check_json_columns: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-hash-size-mb", type=float, default=100.0)
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--full-benchmark-scan", action="store_true")
    parser.add_argument("--check-python-ast", action="store_true")
    parser.add_argument("--check-parquet", action="store_true")
    parser.add_argument("--check-json-columns", action="store_true")
    return parser.parse_args(argv)


def resolve_options(args: argparse.Namespace) -> AuditOptions:
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("--output-dir must be inside --root")
    if args.max_hash_size_mb <= 0:
        raise ValueError("--max-hash-size-mb must be positive")
    if args.sample_rows <= 0:
        raise ValueError("--sample-rows must be positive")
    return AuditOptions(
        root=root,
        output_dir=output_dir,
        max_hash_bytes=int(args.max_hash_size_mb * 1024 * 1024),
        sample_rows=int(args.sample_rows),
        full_benchmark_scan=bool(args.full_benchmark_scan),
        check_python_ast=bool(args.check_python_ast),
        check_parquet=bool(args.check_parquet),
        check_json_columns=bool(args.check_json_columns),
    )


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("repository_audit")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "audit_helper.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(stream_handler)
    return logger


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_excluded(path: Path, options: AuditOptions) -> bool:
    if path == options.output_dir or options.output_dir in path.parents:
        return True
    try:
        relative_parts = path.relative_to(options.root).parts
    except ValueError:
        return True
    if any(part in EXCLUDED_DIRS for part in relative_parts):
        return True
    return any(pattern.search(path.name) for pattern in TRANSIENT_FILE_PATTERNS)


def iter_repository_files(options: AuditOptions, logger: logging.Logger) -> Iterator[Path]:
    for directory, dirnames, filenames in os.walk(options.root, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in EXCLUDED_DIRS
            and not is_excluded(directory_path / name, options)
        ]
        for filename in filenames:
            path = directory_path / filename
            if is_excluded(path, options):
                continue
            try:
                if path.is_file():
                    yield path
            except OSError as exc:
                logger.warning("Cannot inspect file kind for %s: %s", path, exc)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key, "")) for key in fieldnames})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        if isinstance(value, set):
            value = sorted(value)
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if value is None:
        return ""
    return value


def hash_file(path: Path, max_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        return "HASH_SKIPPED_LARGE_FILE"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_for(relative_path: str) -> str:
    lower = relative_path.lower()
    suffix = Path(relative_path).suffix.lower()
    name = Path(relative_path).name.lower()
    if suffix in {".docx", ".pdf", ".tex", ".bib", ".odt", ".rtf"} or "manuscript" in lower:
        return "manuscript"
    if suffix in {".py", ".ps1", ".bat", ".sh", ".sql", ".r"}:
        return "code"
    if "checkpoint" in name or suffix == ".jsonl" and "checkpoint" in lower:
        return "checkpoint"
    if suffix == ".log" or "log" in Path(relative_path).parts:
        return "log"
    if suffix in {".zip", ".7z", ".tar", ".gz", ".bz2"}:
        return "archive"
    if any(marker in name for marker in ("benchmark", "gold", "template", "adjudication", "review")):
        return "benchmark"
    if lower.startswith("data/raw/"):
        return "raw data"
    if lower.startswith("data/processed/") or lower.startswith("data/merged/"):
        return "processed data"
    if lower.startswith("data/analysis/") or lower.startswith("data/e2c/"):
        return "analysis output"
    if lower.startswith("reports/"):
        return "report"
    if suffix == ".md" or name in {"report.txt", "structure.txt", "file_report.txt"}:
        return "report"
    if name == "_history.txt":
        return "log"
    if "prompt" in name:
        return "config"
    if name.startswith("requirements") or suffix in {".toml", ".ini", ".cfg", ".yaml", ".yml"}:
        return "config"
    return "unknown"


def read_text_bounded(path: Path, max_bytes: int = 20 * 1024 * 1024) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def build_reference_index(
    files: Sequence[Path], options: AuditOptions, logger: logging.Logger
) -> dict[str, set[str]]:
    by_basename: dict[str, list[str]] = defaultdict(list)
    known_relative: dict[str, str] = {}
    for path in files:
        relative = relpath(path, options.root)
        by_basename[path.name.lower()].append(relative)
        known_relative[relative.lower()] = relative

    references: dict[str, set[str]] = defaultdict(set)
    for source in files:
        if source.suffix.lower() not in TEXT_REFERENCE_EXTENSIONS:
            continue
        text = read_text_bounded(source)
        if text is None:
            logger.warning("Reference scan skipped unreadable/large text file: %s", relpath(source, options.root))
            continue
        source_relative = relpath(source, options.root)
        normalized = text.replace("\\", "/").lower()
        for known_lower, known_original in known_relative.items():
            if known_original == source_relative:
                continue
            if "/" in known_lower and known_lower in normalized:
                references[known_original].add(source_relative)
        for match in PATH_TOKEN_RE.finditer(text):
            basename = Path(match.group("path").replace("\\", "/")).name.lower()
            for target in by_basename.get(basename, []):
                if target != source_relative:
                    references[target].add(source_relative)
    return references


def version_family(name: str) -> tuple[str, int] | None:
    match = re.search(r"(?:_v|\bv)(\d+)(?=\.[^.]+$)", name, flags=re.IGNORECASE)
    if not match:
        return None
    base = name[: match.start()] + name[match.end() :]
    return base.lower(), int(match.group(1))


def lifecycle_for(
    path: Path,
    relative: str,
    category: str,
    sha256: str,
    hash_groups: dict[str, list[str]],
    references: dict[str, set[str]],
    version_maxima: dict[tuple[str, str], int],
) -> tuple[str, str]:
    statuses: list[str] = []
    reasons: list[str] = []
    lower = relative.lower()
    name = path.name.lower()
    family = version_family(path.name)
    if category == "benchmark" and any(marker in name for marker in ("gold", "reviewed", "adjudication")):
        statuses.append("frozen")
        reasons.append("authoritative benchmark naming")
    if lower.startswith(("data/processed/", "data/analysis/", "reports/")):
        statuses.append("generated")
        reasons.append("processed/analysis/report location")
    if sha256 not in {"", "HASH_SKIPPED_LARGE_FILE", "HASH_ERROR"} and len(hash_groups.get(sha256, [])) > 1:
        statuses.append("duplicated")
        reasons.append("identical SHA-256")
    if family:
        base, version = family
        maximum = version_maxima.get((path.parent.as_posix().lower(), base), version)
        if version < maximum:
            statuses.append("superseded")
            reasons.append(f"newer v{maximum} exists in same directory")
    if any(marker in lower for marker in ("smoke_test", "pilot_", "_pilot", "draft", "prior", "old")):
        statuses.append("superseded_candidate")
        reasons.append("pilot/smoke/draft/prior naming")
    if relative in references:
        statuses.append("active")
        reasons.append("referenced by inspected code/report")
    elif category not in {"raw data", "config", "manuscript"} and path.suffix.lower() not in {".py", ".ps1"}:
        statuses.append("orphaned_candidate")
        reasons.append("no explicit filename/path reference found")
    if not statuses:
        statuses.append("active")
        reasons.append("source/config/raw input or no contrary evidence")
    return ";".join(dict.fromkeys(statuses)), "; ".join(dict.fromkeys(reasons))


def inventory_files(
    files: Sequence[Path], options: AuditOptions, logger: logging.Logger
) -> list[dict[str, Any]]:
    references = build_reference_index(files, options, logger)
    hash_values: dict[str, str] = {}
    hash_groups: dict[str, list[str]] = defaultdict(list)
    version_maxima: dict[tuple[str, str], int] = {}
    for path in files:
        family = version_family(path.name)
        if family:
            base, version = family
            key = (path.parent.as_posix().lower(), base)
            version_maxima[key] = max(version, version_maxima.get(key, version))
        relative = relpath(path, options.root)
        try:
            value = hash_file(path, options.max_hash_bytes)
        except OSError as exc:
            logger.warning("Hash failed for %s: %s", relative, exc)
            value = "HASH_ERROR"
        hash_values[relative] = value
        if value not in {"", "HASH_SKIPPED_LARGE_FILE", "HASH_ERROR"}:
            hash_groups[value].append(relative)

    rows: list[dict[str, Any]] = []
    for path in files:
        relative = relpath(path, options.root)
        category = category_for(relative)
        sha256 = hash_values[relative]
        lifecycle, basis = lifecycle_for(
            path, relative, category, sha256, hash_groups, references, version_maxima
        )
        try:
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            size = stat.st_size
        except OSError as exc:
            logger.warning("Stat failed for %s: %s", relative, exc)
            modified = ""
            size = ""
        referenced_by = sorted(references.get(relative, set()))
        rows.append(
            {
                "relative_path": relative,
                "extension": path.suffix.lower() or "[none]",
                "byte_size": size,
                "modified_time_utc": modified,
                "sha256": sha256,
                "category": category,
                "referenced": bool(referenced_by),
                "referenced_by_count": len(referenced_by),
                "referenced_by": referenced_by,
                "lifecycle_status": lifecycle,
                "status_basis": basis,
            }
        )
    return rows


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def literal_value(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        if isinstance(node, ast.Name):
            return f"<{node.id}>"
        return ast.unparse(node) if hasattr(ast, "unparse") else "<expression>"


def path_expression(node: ast.AST, assignments: dict[str, str], depth: int = 0) -> str | None:
    if depth > 8:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return assignments.get(node.id, f"<{node.id}>")
    if isinstance(node, ast.Attribute):
        return f"<{call_name(node)}>"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = path_expression(node.left, assignments, depth + 1)
        right = path_expression(node.right, assignments, depth + 1)
        if left and right:
            return f"{left.rstrip('/\\')}/{right.lstrip('/\\')}"
    if isinstance(node, ast.Call) and call_name(node.func).split(".")[-1] in {"Path", "resolve"} and node.args:
        return path_expression(node.args[0], assignments, depth + 1)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            else:
                parts.append("<expression>")
        return "".join(parts)
    return None


def looks_like_path(value: str) -> bool:
    lower = value.lower().replace("\\", "/")
    return (
        any(lower.endswith(suffix) for suffix in PATH_SUFFIXES)
        or lower.startswith(("data/", "reports/", "scripts/", "tools/"))
        or "/data/" in lower
        or "/reports/" in lower
    )


def classify_io(call: str, variable_name: str = "") -> str:
    lower = call.lower()
    variable_lower = variable_name.lower()
    if any(marker in lower for marker in ("read_", "readtext", "read_text", "load", "scan_", "from_csv", "parquet_file")):
        return "input"
    if any(marker in lower for marker in ("to_csv", "to_parquet", "write_", "write_text", "dump", "save", "export")):
        return "output"
    if any(marker in variable_lower for marker in ("output", "report", "checkpoint", "manifest", "metadata", "result")):
        return "declared_output"
    if any(marker in variable_lower for marker in ("input", "benchmark", "source")):
        return "declared_input"
    return "declared_path"


def infer_purpose(path: Path, docstring: str) -> str:
    if docstring:
        return docstring.strip().splitlines()[0][:500]
    name = path.stem.replace("_", " ")
    return name[:1].upper() + name[1:]


def inspect_python_file(
    path: Path,
    options: AuditOptions,
    local_modules: dict[str, list[str]],
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    relative = relpath(path, options.root)
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=relative)
    except SyntaxError as exc:
        logger.warning("AST parse failed for %s: %s", relative, exc)
        return (
            {
                "file_path": relative,
                "ast_status": f"PARSE_ERROR: {exc}",
                "module_docstring": "",
                "main_purpose": "unparsed Python file",
            },
            [],
            [],
        )

    docstring = ast.get_docstring(tree) or ""
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    top_imports = sorted({name.split(".")[0] for name in imports})
    local_imports = sorted({name for name in top_imports if name in local_modules})
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    external_imports = sorted({name for name in top_imports if name not in local_modules and name not in stdlib})

    functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    validation_functions = sorted(
        name for name in functions if re.search(r"validate|validation|check|audit|assert|verify", name, re.IGNORECASE)
    )

    cli_arguments: list[dict[str, Any]] = []
    env_vars: set[str] = set()
    seeds: set[str] = set()
    assignments: dict[str, str] = {}
    io_paths: list[dict[str, str]] = []
    destructive_calls: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value_node = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = path_expression(value_node, assignments) if value_node else None
            for target in targets:
                if isinstance(target, ast.Name) and value:
                    assignments[target.id] = value
                    if looks_like_path(value):
                        io_paths.append(
                            {
                                "path": value,
                                "relationship": classify_io("", target.id),
                                "evidence": f"assignment:{target.id}",
                            }
                        )
                if isinstance(target, ast.Name) and "seed" in target.id.lower():
                    seeds.add(f"{target.id}={literal_value(value_node)}")
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            short = name.split(".")[-1]
            if short == "add_argument":
                flags = [literal_value(arg) for arg in node.args if isinstance(arg, ast.Constant)]
                kwargs = {kw.arg: literal_value(kw.value) for kw in node.keywords if kw.arg}
                cli_arguments.append({"flags": flags, **kwargs})
                if any("seed" in str(flag).lower() for flag in flags):
                    seeds.add(f"CLI seed default={kwargs.get('default')}")
            if name.endswith(("os.getenv", "getenv")) and node.args:
                value = literal_value(node.args[0])
                if isinstance(value, str):
                    env_vars.add(value)
            if short in {"seed", "default_rng"} and node.args:
                seeds.add(f"{name}({literal_value(node.args[0])})")
            if (
                name in {"os.remove", "os.unlink", "os.rmdir", "shutil.rmtree"}
                or short in {"unlink", "rmdir"}
                or name.endswith((".unlink", ".rmdir"))
            ):
                destructive_calls.add(name)
            relationship = classify_io(name)
            if relationship in {"input", "output"}:
                candidates = list(node.args) + [kw.value for kw in node.keywords if kw.arg in {"path", "path_or_buf", "fname", "file"}]
                for candidate in candidates[:2]:
                    value = path_expression(candidate, assignments)
                    if value and looks_like_path(value):
                        io_paths.append(
                            {
                                "path": value,
                                "relationship": relationship,
                                "evidence": f"call:{name}",
                            }
                        )
        if isinstance(node, ast.Subscript) and call_name(node.value).endswith("os.environ"):
            value = literal_value(node.slice)
            if isinstance(value, str):
                env_vars.add(value)

    string_constants = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    for value in string_constants:
        if looks_like_path(value):
            io_paths.append({"path": value, "relationship": "literal_path", "evidence": "string_literal"})
    unique_io = {
        (item["path"], item["relationship"], item["evidence"]): item for item in io_paths
    }
    io_paths = sorted(unique_io.values(), key=lambda item: (item["path"], item["relationship"]))

    local_edges: list[dict[str, Any]] = []
    for module in local_imports:
        for target in local_modules[module]:
            if target != relative:
                local_edges.append(
                    {
                        "source_script": relative,
                        "target_module": target,
                        "import_name": module,
                        "edge_type": "local_import",
                    }
                )

    io_edges: list[dict[str, Any]] = []
    for item in io_paths:
        raw_path = item["path"].replace("\\", "/")
        normalized = raw_path
        normalized = re.sub(r"^<[^>]+>/", "", normalized)
        normalized = normalized.lstrip("./")
        target = options.root / normalized
        exists = target.exists() if "<" not in normalized else False
        io_edges.append(
            {
                "script_path": relative,
                "relationship": item["relationship"],
                "referenced_path_expression": raw_path,
                "resolved_relative_path": normalized if "<" not in normalized else "",
                "target_exists": exists,
                "evidence": item["evidence"],
            }
        )

    absolute_literals = sorted({value for value in string_constants if WINDOWS_ABSOLUTE_RE.match(value)})
    has_main_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and "__name__" in ast.unparse(node.test)
        and "__main__" in ast.unparse(node.test)
        for node in tree.body
    )
    api_markers = sorted(
        marker
        for marker in (
            "OpenAI(",
            "client.responses.create",
            "client.chat.completions.create",
            "OPENAI_API_KEY",
        )
        if marker in text
    )
    checkpoint_paths = sorted({item["path"] for item in io_paths if "checkpoint" in item["path"].lower()})
    report_paths = sorted({item["path"] for item in io_paths if "report" in item["path"].lower()})
    output_paths = sorted(
        {
            item["path"]
            for item in io_paths
            if item["relationship"] in {"output", "declared_output"}
        }
    )
    input_paths = sorted(
        {
            item["path"]
            for item in io_paths
            if item["relationship"] in {"input", "declared_input"}
        }
    )
    row = {
        "file_path": relative,
        "ast_status": "OK",
        "module_docstring": docstring,
        "main_purpose": infer_purpose(path, docstring),
        "imported_local_modules": local_imports,
        "imported_external_packages": external_imports,
        "defined_functions": functions,
        "defined_classes": classes,
        "cli_arguments_and_defaults": cli_arguments,
        "environment_variables": sorted(env_vars),
        "random_seeds": sorted(seeds),
        "input_paths": input_paths,
        "output_paths": output_paths,
        "report_paths": report_paths,
        "checkpoint_paths": checkpoint_paths,
        "api_usage_markers": api_markers,
        "resume_behavior": "checkpoint/resume markers present" if "resume" in text.lower() else "no explicit resume marker",
        "validation_functions": validation_functions,
        "destructive_operations": sorted(destructive_calls),
        "has_main_guard": has_main_guard,
        "hard_coded_absolute_paths": absolute_literals,
        "path_strategy": "hard-coded absolute path present" if absolute_literals else (
            "root-relative markers present" if "--root" in text or "Path(__file__)" in text else "mixed/implicit"
        ),
    }
    return row, local_edges, io_edges


def inspect_python(
    files: Sequence[Path], options: AuditOptions, logger: logging.Logger
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    python_files = [path for path in files if path.suffix.lower() == ".py"]
    local_modules: dict[str, list[str]] = defaultdict(list)
    for path in python_files:
        local_modules[path.stem].append(relpath(path, options.root))
    rows: list[dict[str, Any]] = []
    local_edges: list[dict[str, Any]] = []
    io_edges: list[dict[str, Any]] = []
    for path in python_files:
        try:
            row, script_edges, script_io = inspect_python_file(
                path, options, local_modules, logger
            )
            rows.append(row)
            local_edges.extend(script_edges)
            io_edges.extend(script_io)
        except Exception as exc:  # audit continues and logs the per-file failure
            logger.warning("Python inspection failed for %s: %s", relpath(path, options.root), exc)
            rows.append(
                {
                    "file_path": relpath(path, options.root),
                    "ast_status": f"INSPECTION_ERROR: {type(exc).__name__}: {exc}",
                }
            )
    return rows, local_edges, io_edges


def benchmark_file(path: Path, root: Path) -> bool:
    relative = relpath(path, root).lower()
    return category_for(relative) == "benchmark" or any(
        marker in path.name.lower() for marker in ("benchmark", "gold", "adjudication")
    )


def count_binary_newlines(path: Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    if path.stat().st_size and last != b"\n":
        count += 1
    return max(0, count - 1)


def identifier_columns(columns: Sequence[str]) -> list[str]:
    return [column for column in columns if ID_NAME_RE.search(str(column))]


def enum_values(frame: Any, max_values: int = 30) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for column in frame.columns:
        if ID_NAME_RE.search(str(column)):
            continue
        try:
            values = frame[column].dropna().unique().tolist()
        except Exception:
            continue
        if 0 < len(values) <= max_values:
            result[str(column)] = sorted((str(value) for value in values))
    return result


def null_rates(frame: Any) -> dict[str, float]:
    if len(frame) == 0:
        return {str(column): 0.0 for column in frame.columns}
    return {str(column): round(float(frame[column].isna().mean()), 8) for column in frame.columns}


def duplicate_id_count(frame: Any, ids: Sequence[str]) -> int | str:
    if not ids or len(frame) == 0:
        return 0
    try:
        return int(frame.duplicated(subset=list(ids), keep=False).sum())
    except Exception:
        return "NOT_COMPUTED"


def json_column_audit(
    frame: Any, enabled: bool, full_scan: bool
) -> tuple[list[str], dict[str, int]]:
    if not enabled or len(frame) == 0:
        return [], {}
    columns: list[str] = []
    failures: dict[str, int] = {}
    for column in frame.columns:
        series = frame[column].dropna()
        if len(series) == 0:
            continue
        name_suggests = "json" in str(column).lower()
        sample_values = series.astype(str).head(20)
        value_suggests = bool(sample_values.map(lambda value: bool(JSON_LIKE_RE.match(value))).mean() >= 0.8)
        if not name_suggests and not value_suggests:
            continue
        columns.append(str(column))
        target = series if full_scan else series.head(5000)
        failure_count = 0
        for value in target:
            try:
                json.loads(str(value))
            except (TypeError, ValueError, json.JSONDecodeError):
                failure_count += 1
        failures[str(column)] = failure_count
    return columns, failures


def date_and_year_bounds(frame: Any) -> tuple[str, str, str, str]:
    try:
        import pandas as pd
    except ImportError:
        return "", "", "", ""
    dates: list[Any] = []
    years: list[float] = []
    for column in frame.columns:
        name = str(column)
        series = frame[column].dropna()
        if len(series) == 0:
            continue
        if YEAR_NAME_RE.search(name):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            years.extend(float(value) for value in numeric if 1900 <= float(value) <= 2100)
        elif DATE_NAME_RE.search(name):
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=True).dropna()
            if len(parsed):
                dates.extend(parsed.tolist())
    min_date = min(dates).isoformat() if dates else ""
    max_date = max(dates).isoformat() if dates else ""
    min_year = str(int(min(years))) if years else ""
    max_year = str(int(max(years))) if years else ""
    return min_date, max_date, min_year, max_year


def temporal_feature_flag(relative: str, columns: Sequence[str], max_year: str) -> str:
    lower = relative.lower()
    historical = "2015_2016" in lower or "histor" in lower
    suspicious_columns = [column for column in columns if OUTCOME_NAME_RE.search(str(column))]
    if not historical:
        return "NO"
    if max_year and int(float(max_year)) >= 2017:
        return "YES_MAX_YEAR_2017_OR_LATER"
    if suspicious_columns:
        return "REVIEW_OUTCOME_NAMED_COLUMNS:" + "|".join(suspicious_columns)
    return "NO"


def inspect_csv(
    path: Path, relative: str, options: AuditOptions
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pandas as pd

    size = path.stat().st_size
    full_scan = benchmark_file(path, options.root) and options.full_benchmark_scan
    if not full_scan and size <= 50 * 1024 * 1024:
        full_scan = True
    warnings: list[str] = []
    try:
        frame = pd.read_csv(path, low_memory=False, encoding="utf-8") if full_scan else pd.read_csv(
            path, nrows=options.sample_rows, low_memory=False, encoding="utf-8"
        )
    except UnicodeDecodeError:
        frame = pd.read_csv(path, nrows=None if full_scan else options.sample_rows, low_memory=False, encoding="latin-1")
        warnings.append("decoded_as_latin_1")
    row_count = len(frame) if full_scan else count_binary_newlines(path)
    scope = "full" if full_scan else f"bounded_sample_{len(frame)};row_count_by_newlines"
    columns = [str(column) for column in frame.columns]
    ids = identifier_columns(columns)
    json_columns, json_failures = json_column_audit(
        frame, options.check_json_columns, full_scan or benchmark_file(path, options.root)
    )
    min_date, max_date, min_year, max_year = date_and_year_bounds(frame)
    duplicate_count: int | str = duplicate_id_count(frame, ids)
    if not full_scan and ids:
        duplicate_count = f"SAMPLE_ONLY:{duplicate_count}"
    findings: list[dict[str, Any]] = []
    if any(value for value in json_failures.values()):
        findings.append(
            {
                "finding_id": "AUTO_JSON_PARSE_FAILURE",
                "path": relative,
                "severity": "HIGH" if benchmark_file(path, options.root) else "MEDIUM",
                "status": "FAIL",
                "evidence": json_failures,
                "recommended_action": "Inspect malformed JSON-encoded values.",
            }
        )
    if isinstance(duplicate_count, int) and duplicate_count > 0 and benchmark_file(path, options.root):
        findings.append(
            {
                "finding_id": "AUTO_DUPLICATE_IDENTIFIER",
                "path": relative,
                "severity": "HIGH" if benchmark_file(path, options.root) else "MEDIUM",
                "status": "FAIL",
                "evidence": {"identifier_columns": ids, "duplicate_rows": duplicate_count},
                "recommended_action": "Verify identifier grain and duplicate policy.",
            }
        )
    flag = temporal_feature_flag(relative, columns, max_year)
    if flag.startswith("YES"):
        findings.append(
            {
                "finding_id": "AUTO_POTENTIAL_TEMPORAL_LEAKAGE",
                "path": relative,
                "severity": "HIGH",
                "status": "WARNING",
                "evidence": flag,
                "recommended_action": "Inspect historical feature construction and source years.",
            }
        )
    row = {
        "path": relative,
        "format": "csv",
        "inspection_scope": scope,
        "row_count": row_count,
        "column_count": len(columns),
        "exact_column_names": columns,
        "data_types": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "identifier_columns": ids,
        "duplicate_identifier_count": duplicate_count,
        "null_rates": null_rates(frame),
        "json_encoded_columns": json_columns,
        "json_parse_failures": json_failures,
        "enum_values": enum_values(frame),
        "minimum_date": min_date,
        "maximum_date": max_date,
        "minimum_historical_year": min_year,
        "maximum_historical_year": max_year,
        "potential_2017_outcome_in_historical_feature": flag,
        "source_scripts": [],
        "downstream_scripts": [],
        "warnings": warnings,
    }
    return row, findings


def inspect_parquet(
    path: Path, relative: str, options: AuditOptions
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pandas as pd
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    columns = schema.names
    row_count = parquet.metadata.num_rows
    sample = parquet.read_row_groups(
        list(range(min(parquet.num_row_groups, 2))),
    ).slice(0, options.sample_rows).to_pandas()
    ids = identifier_columns(columns)
    min_date, max_date, min_year, max_year = date_and_year_bounds(sample)

    year_columns = [name for name in columns if YEAR_NAME_RE.search(name)]
    if year_columns:
        year_table = parquet.read(columns=year_columns)
        year_frame = year_table.to_pandas()
        _, _, exact_min_year, exact_max_year = date_and_year_bounds(year_frame)
        min_year = exact_min_year or min_year
        max_year = exact_max_year or max_year

    exact_nulls: dict[str, int] = defaultdict(int)
    null_stats_available = True
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        for column_index, name in enumerate(columns):
            statistics = row_group.column(column_index).statistics
            if statistics is None or statistics.null_count is None:
                null_stats_available = False
                continue
            exact_nulls[name] += int(statistics.null_count)
    null_summary = (
        {name: round(exact_nulls[name] / row_count, 8) if row_count else 0.0 for name in columns}
        if null_stats_available
        else null_rates(sample)
    )
    duplicate_count: int | str = "NOT_COMPUTED_LARGE_PARQUET"
    if ids and row_count <= 1_000_000:
        id_frame = parquet.read(columns=ids).to_pandas()
        duplicate_count = duplicate_id_count(id_frame, ids)
    json_columns, json_failures = json_column_audit(sample, options.check_json_columns, False)
    flag = temporal_feature_flag(relative, columns, max_year)
    findings: list[dict[str, Any]] = []
    if flag.startswith("YES"):
        findings.append(
            {
                "finding_id": "AUTO_POTENTIAL_TEMPORAL_LEAKAGE",
                "path": relative,
                "severity": "HIGH",
                "status": "WARNING",
                "evidence": flag,
                "recommended_action": "Inspect historical feature construction and source years.",
            }
        )
    row = {
        "path": relative,
        "format": "parquet",
        "inspection_scope": f"metadata_full;sample_{len(sample)};year_columns_full",
        "row_count": row_count,
        "column_count": len(columns),
        "exact_column_names": columns,
        "data_types": {field.name: str(field.type) for field in schema},
        "identifier_columns": ids,
        "duplicate_identifier_count": duplicate_count,
        "null_rates": null_summary,
        "json_encoded_columns": json_columns,
        "json_parse_failures": json_failures,
        "enum_values": enum_values(sample),
        "minimum_date": min_date,
        "maximum_date": max_date,
        "minimum_historical_year": min_year,
        "maximum_historical_year": max_year,
        "potential_2017_outcome_in_historical_feature": flag,
        "source_scripts": [],
        "downstream_scripts": [],
        "warnings": [] if null_stats_available else ["null_rates_from_sample_due_to_missing_parquet_stats"],
    }
    return row, findings


def normalize_json_records(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(value, list):
        records = [item for item in value if isinstance(item, dict)]
        columns = sorted({key for item in records for key in item})
        return records, columns
    if isinstance(value, dict):
        return [value], list(value.keys())
    return [], []


def inspect_json_like(
    path: Path, relative: str, options: AuditOptions
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    values: list[Any] = []
    warnings: list[str] = []
    parse_failures = 0
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError:
                    parse_failures += 1
                    warnings.append(f"invalid_json_line:{line_number}")
    elif suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            value = yaml.safe_load(handle)
        values = value if isinstance(value, list) else [value]
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            value = json.load(handle)
        values = value if isinstance(value, list) else [value]
    records = [item for item in values if isinstance(item, dict)]
    columns = sorted({str(key) for record in records for key in record})
    findings: list[dict[str, Any]] = []
    if parse_failures:
        findings.append(
            {
                "finding_id": "AUTO_JSONL_PARSE_FAILURE",
                "path": relative,
                "severity": "MEDIUM",
                "status": "FAIL",
                "evidence": {"parse_failures": parse_failures},
                "recommended_action": "Inspect invalid JSONL records.",
            }
        )
    row = {
        "path": relative,
        "format": suffix.lstrip("."),
        "inspection_scope": "full",
        "row_count": len(values),
        "column_count": len(columns),
        "exact_column_names": columns,
        "data_types": {
            column: sorted({type(record.get(column)).__name__ for record in records if column in record})
            for column in columns
        },
        "identifier_columns": identifier_columns(columns),
        "duplicate_identifier_count": "NOT_COMPUTED_NESTED_JSON",
        "null_rates": {},
        "json_encoded_columns": [],
        "json_parse_failures": {"file": parse_failures},
        "enum_values": {},
        "minimum_date": "",
        "maximum_date": "",
        "minimum_historical_year": "",
        "maximum_historical_year": "",
        "potential_2017_outcome_in_historical_feature": temporal_feature_flag(relative, columns, ""),
        "source_scripts": [],
        "downstream_scripts": [],
        "warnings": warnings,
    }
    return row, findings


def inspect_excel(
    path: Path, relative: str, options: AuditOptions
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pandas as pd

    workbook = pd.ExcelFile(path)
    sheet_summary: dict[str, Any] = {}
    total_rows = 0
    all_columns: list[str] = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet, nrows=options.sample_rows)
        sheet_summary[sheet] = {
            "sample_rows": len(frame),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        }
        total_rows += len(frame)
        all_columns.extend(f"{sheet}:{column}" for column in frame.columns)
    return (
        {
            "path": relative,
            "format": path.suffix.lower().lstrip("."),
            "inspection_scope": f"bounded_sample_{options.sample_rows}_per_sheet",
            "row_count": total_rows,
            "column_count": len(all_columns),
            "exact_column_names": all_columns,
            "data_types": sheet_summary,
            "identifier_columns": [],
            "duplicate_identifier_count": "NOT_COMPUTED_MULTI_SHEET",
            "null_rates": {},
            "json_encoded_columns": [],
            "json_parse_failures": {},
            "enum_values": {},
            "minimum_date": "",
            "maximum_date": "",
            "minimum_historical_year": "",
            "maximum_historical_year": "",
            "potential_2017_outcome_in_historical_feature": "NOT_VERIFIABLE",
            "source_scripts": [],
            "downstream_scripts": [],
            "warnings": ["row_count_is_sample_total"],
        },
        [],
    )


def inspect_duckdb(
    path: Path, relative: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import duckdb

    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = [row[0] for row in connection.execute("SHOW TABLES").fetchall()]
        estimated_rows = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT table_name, estimated_size FROM duckdb_tables() WHERE NOT internal"
            ).fetchall()
        }
        schemas: dict[str, list[dict[str, str]]] = {}
        row_counts: dict[str, int | str] = {}
        exact_counts_allowed = path.stat().st_size <= 100 * 1024 * 1024
        for table in tables:
            escaped = table.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            schemas[table] = [
                {"name": str(column[1]), "type": str(column[2])} for column in columns
            ]
            if exact_counts_allowed:
                try:
                    row_counts[table] = int(connection.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0])
                except Exception as exc:
                    row_counts[table] = f"COUNT_ERROR:{type(exc).__name__}"
            else:
                row_counts[table] = f"CATALOG_ESTIMATE:{estimated_rows.get(table, 'UNAVAILABLE')}"
        flattened_columns = [f"{table}.{column['name']}" for table, columns in schemas.items() for column in columns]
        total = sum(value for value in row_counts.values() if isinstance(value, int))
        return (
            {
                "path": relative,
                "format": "duckdb",
                "inspection_scope": (
                    "read_only_catalog_and_exact_counts"
                    if exact_counts_allowed
                    else "read_only_catalog_and_estimated_counts_large_database"
                ),
                "row_count": total if exact_counts_allowed else "SEE_TABLE_CATALOG_ESTIMATES",
                "column_count": len(flattened_columns),
                "exact_column_names": flattened_columns,
                "data_types": {"schemas": schemas, "table_row_counts": row_counts},
                "identifier_columns": identifier_columns(flattened_columns),
                "duplicate_identifier_count": "NOT_COMPUTED_DATABASE",
                "null_rates": {},
                "json_encoded_columns": [],
                "json_parse_failures": {},
                "enum_values": {},
                "minimum_date": "",
                "maximum_date": "",
                "minimum_historical_year": "",
                "maximum_historical_year": "",
                "potential_2017_outcome_in_historical_feature": temporal_feature_flag(relative, flattened_columns, ""),
                "source_scripts": [],
                "downstream_scripts": [],
                "warnings": [],
            },
            [],
        )
    finally:
        connection.close()


def index_io_sources(
    io_edges: Sequence[dict[str, Any]], root: Path
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    sources: dict[str, set[str]] = defaultdict(set)
    downstream: dict[str, set[str]] = defaultdict(set)
    for edge in io_edges:
        resolved = str(edge.get("resolved_relative_path", "")).replace("\\", "/")
        if not resolved:
            continue
        if edge.get("relationship") in {"output", "declared_output"}:
            sources[resolved].add(str(edge.get("script_path", "")))
        elif edge.get("relationship") in {"input", "declared_input"}:
            downstream[resolved].add(str(edge.get("script_path", "")))
    return sources, downstream


def inspect_structured_files(
    files: Sequence[Path],
    options: AuditOptions,
    io_edges: Sequence[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources, downstream = index_io_sources(io_edges, options.root)
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    structured = [path for path in files if path.suffix.lower() in STRUCTURED_EXTENSIONS]
    for index, path in enumerate(structured, start=1):
        relative = relpath(path, options.root)
        logger.info("Structured inspection %s/%s: %s", index, len(structured), relative)
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                row, file_findings = inspect_csv(path, relative, options)
            elif suffix == ".parquet":
                if options.check_parquet:
                    row, file_findings = inspect_parquet(path, relative, options)
                else:
                    row = {"path": relative, "format": "parquet", "inspection_scope": "SKIPPED_BY_FLAG"}
                    file_findings = []
            elif suffix in {".json", ".jsonl", ".yaml", ".yml"}:
                row, file_findings = inspect_json_like(path, relative, options)
            elif suffix in {".xlsx", ".xls"}:
                row, file_findings = inspect_excel(path, relative, options)
            elif suffix == ".duckdb":
                row, file_findings = inspect_duckdb(path, relative)
            else:
                continue
            row["source_scripts"] = sorted(sources.get(relative, set()))
            row["downstream_scripts"] = sorted(downstream.get(relative, set()))
            rows.append(row)
            findings.extend(file_findings)
        except Exception as exc:  # audit continues and records the failure
            logger.warning("Structured inspection failed for %s: %s", relative, exc)
            rows.append(
                {
                    "path": relative,
                    "format": path.suffix.lower().lstrip("."),
                    "inspection_scope": "INSPECTION_ERROR",
                    "warnings": [f"{type(exc).__name__}: {exc}"],
                }
            )
            findings.append(
                {
                    "finding_id": "AUTO_STRUCTURED_INSPECTION_ERROR",
                    "path": relative,
                    "severity": "MEDIUM",
                    "status": "NOT_VERIFIABLE",
                    "evidence": f"{type(exc).__name__}: {exc}",
                    "recommended_action": "Review format compatibility or file integrity.",
                }
            )
    return rows, findings


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        options = resolve_options(args)
        logger = configure_logging(options.output_dir)
        logger.info("Audit root: %s", options.root)
        logger.info("Audit output: %s", options.output_dir)
        files = sorted(iter_repository_files(options, logger), key=lambda path: relpath(path, options.root).lower())
        logger.info("Repository files discovered: %s", len(files))

        inventory_rows = inventory_files(files, options, logger)
        write_csv(
            options.output_dir / "repository_file_inventory.csv",
            [
                "relative_path",
                "extension",
                "byte_size",
                "modified_time_utc",
                "sha256",
                "category",
                "referenced",
                "referenced_by_count",
                "referenced_by",
                "lifecycle_status",
                "status_basis",
            ],
            inventory_rows,
        )

        python_rows: list[dict[str, Any]] = []
        local_edges: list[dict[str, Any]] = []
        io_edges: list[dict[str, Any]] = []
        if options.check_python_ast:
            python_rows, local_edges, io_edges = inspect_python(files, options, logger)
        write_csv(
            options.output_dir / "python_script_inventory.csv",
            [
                "file_path",
                "ast_status",
                "module_docstring",
                "main_purpose",
                "imported_local_modules",
                "imported_external_packages",
                "defined_functions",
                "defined_classes",
                "cli_arguments_and_defaults",
                "environment_variables",
                "random_seeds",
                "input_paths",
                "output_paths",
                "report_paths",
                "checkpoint_paths",
                "api_usage_markers",
                "resume_behavior",
                "validation_functions",
                "destructive_operations",
                "has_main_guard",
                "hard_coded_absolute_paths",
                "path_strategy",
            ],
            python_rows,
        )
        write_csv(
            options.output_dir / "local_dependency_edges.csv",
            ["source_script", "target_module", "import_name", "edge_type"],
            local_edges,
        )
        write_csv(
            options.output_dir / "io_dependency_edges.csv",
            [
                "script_path",
                "relationship",
                "referenced_path_expression",
                "resolved_relative_path",
                "target_exists",
                "evidence",
            ],
            io_edges,
        )

        schema_rows, integrity_findings = inspect_structured_files(
            files, options, io_edges, logger
        )
        write_csv(
            options.output_dir / "data_schema_inventory.csv",
            [
                "path",
                "format",
                "inspection_scope",
                "row_count",
                "column_count",
                "exact_column_names",
                "data_types",
                "identifier_columns",
                "duplicate_identifier_count",
                "null_rates",
                "json_encoded_columns",
                "json_parse_failures",
                "enum_values",
                "minimum_date",
                "maximum_date",
                "minimum_historical_year",
                "maximum_historical_year",
                "potential_2017_outcome_in_historical_feature",
                "source_scripts",
                "downstream_scripts",
                "warnings",
            ],
            schema_rows,
        )
        write_csv(
            options.output_dir / "data_integrity_findings.csv",
            ["finding_id", "path", "severity", "status", "evidence", "recommended_action"],
            integrity_findings,
        )

        category_counts = Counter(row["category"] for row in inventory_rows)
        extension_counts = Counter(row["extension"] for row in inventory_rows)
        duplicate_groups = len(
            {
                row["sha256"]
                for row in inventory_rows
                if row["sha256"] not in {"HASH_SKIPPED_LARGE_FILE", "HASH_ERROR", ""}
                and sum(other["sha256"] == row["sha256"] for other in inventory_rows) > 1
            }
        )
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(options.root),
            "audit_output_excluded_from_inventory": relpath(options.output_dir, options.root),
            "file_count": len(inventory_rows),
            "python_script_count": len(python_rows),
            "structured_file_count": len(schema_rows),
            "category_counts": dict(sorted(category_counts.items())),
            "extension_counts": dict(sorted(extension_counts.items())),
            "duplicate_hash_groups": duplicate_groups,
            "integrity_finding_count": len(integrity_findings),
        }
        with (options.output_dir / "audit_helper_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        logger.info("Audit helper completed: %s", json.dumps(summary, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR audit helper failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
