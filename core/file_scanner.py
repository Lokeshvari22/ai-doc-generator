"""
file_scanner.py — Discover source files to document.

FIX: Python files containing zero function/class definitions (pure data
modules like Courses.py) are now detected and skipped before the pipeline,
saving API quota on files that produce only low-quality chunk docs.
"""

import ast
from pathlib import Path
from config import Config


def _has_documentable_units(path: Path) -> bool:
    """
    Return True if a Python file contains at least one function or class.
    Data-only files (variable assignments, lists, dicts) return False.

    Non-Python files always return True — we can't inspect them cheaply.
    """
    if path.suffix != ".py":
        return True

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree   = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef,
                                  ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                return True
        return False   # only variable assignments / data literals
    except Exception:
        return True    # parse error → let the pipeline handle it


def scan_files(extract_dir: str) -> list:
    """
    Return sorted list of Path objects for all supported source files.

    FIX: Python files with no functions or classes are excluded with a
    warning printed to stdout so the user knows they were skipped.
    """
    root  = Path(extract_dir)
    files = []

    for path in root.rglob("*"):

        # Skip directories
        if not path.is_file():
            continue

        # Skip hidden files
        if path.name.startswith("."):
            continue

        # Skip ignored directories
        if any(part in Config.IGNORED_DIRS for part in path.parts):
            continue

        # Check extension
        if path.suffix not in Config.SUPPORTED_EXTENSIONS:
            continue

        # FIX: skip data-only Python files
        if not _has_documentable_units(path):
            print(f"    ⚠ Skipping {path.name} — no functions or classes "
                  f"(data-only module, would produce low-quality chunk docs)")
            continue

        files.append(path)

    return sorted(files)