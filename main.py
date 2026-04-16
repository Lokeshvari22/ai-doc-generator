"""
main.py — AI Code Documentation Generator
Standard: IBM 4-Tier + IJIRT Paper

UPDATED:
  - Pipeline label updated: Groq → GPT-4o → gpt-4o-mini + Phi-4
  - setup_context_cache() call removed (Gemini gone)
  - generate_architecture() now returns (img, analyses) tuple — P5 fix
  - add_architecture() receives analyses for text fallback
  - Rate limit display updated
"""

import os
import sys
import subprocess
from pathlib import Path

from config import Config
from core.zip_handler import extract_zip
from core.file_scanner import scan_files
from core.parser import parse_file, detect_language
from core.complexity import analyze, Complexity
from llm.pipeline import process_file, resolve_all_dependencies
from llm.github_client import print_usage
from llm.groq_client import generate
import llm.cache as cache_mod

from output.doc_builder import (
    create_doc, add_file_heading,
    add_unit_doc, add_complexity_table,
    add_architecture, add_summary, save_doc,
)
from output.architecture import generate_architecture


import re as _re


def _simple_complexity(units: list) -> dict:
    """
    Regex-based complexity for non-Python files (Java/JS/C++/C/TS).
    Reuses the same Complexity dataclass as core.complexity so
    add_complexity_table() works identically for all languages.
    """
    branch_pat = _re.compile(
        r'\b(if|else\s*if|for|while|switch|case|catch)\b|(\&\&|\|\|)'
    )
    result = {}
    for unit in units:
        code  = unit.code
        lines = code.count("\n") + 1
        score = 1 + len(branch_pat.findall(code))

        # Count params — commas inside first () after function name
        params = 0
        sig_match = _re.search(_re.escape(unit.name) + r'\s*\(([^)]*)\)', code)
        if sig_match:
            args = sig_match.group(1).strip()
            params = len(args.split(",")) if args else 0

        # Recursion — name called again after its own definition line
        first_line_end = code.find("\n")
        rest = code[first_line_end:] if first_line_end != -1 else ""
        has_recursion = bool(_re.search(r'\b' + _re.escape(unit.name) + r'\s*\(', rest))

        if score <= 5:
            level = "🟢 Low"
        elif score <= 10:
            level = "🟡 Medium"
        else:
            level = "🔴 High"

        result[unit.name] = Complexity(
            name=unit.name, score=score, lines=lines,
            params=params, has_recursion=has_recursion, level=level,
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Git change detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_changed_files(extract_dir: str) -> set | None:
    if not Config.INCREMENTAL_MODE:
        return None
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True,
            cwd=extract_dir, timeout=5,
        )
        if result.returncode == 0:
            changed = {Path(f).name for f in result.stdout.strip().splitlines() if f}
            if changed:
                print(f"  🔄 Git: {len(changed)} changed file(s) since last commit")
            else:
                print("  ✓ Git: no changed files detected")
            return changed
    except Exception:
        pass
    return None


def _should_process(file_path: Path, changed_files: set | None) -> bool:
    if changed_files is None:
        return True
    return file_path.name in changed_files


# ─────────────────────────────────────────────────────────────────────────────
#  Dependency name → PyPI package normaliser
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_pypi_names(raw_deps: set) -> list:
    resolved = set()
    for dep in raw_deps:
        if not dep or not isinstance(dep, str):
            continue
        mapped = Config.DEP_ALIASES.get(dep.lower(), dep)
        if mapped is None:
            continue
        resolved.add(mapped)
    return sorted(resolved)


# ─────────────────────────────────────────────────────────────────────────────
#  README generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_readme(project_name: str, all_results: dict,
                      all_deps: set, all_security: list) -> str:
    func_rows = []
    for filename, results in all_results.items():
        for r in results:
            facts   = r.get("facts", {})
            purpose = facts.get("purpose", "")[:80]
            func_rows.append(f"| `{r['name']}` | `{filename}` | {purpose} |")

    func_table = (
        "| Function | File | Purpose |\n"
        "|----------|------|---------|\n"
        + "\n".join(func_rows[:20])
    )

    pypi_deps  = _resolve_pypi_names(all_deps)
    deps_str   = " ".join(pypi_deps[:15]) or "# no external packages identified"
    deps_label = ", ".join(pypi_deps[:15]) or "Standard library only"

    security_section = ""
    if all_security:
        unique_sec = list(set(s for s in all_security if s))[:8]
        security_section = (
            "\n\n## Security Notes\n"
            + "\n".join(f"- {s}" for s in unique_sec)
        )

    prompt = f"""
Write a professional README.md for the project "{project_name}".
Use ONLY the information below — do not invent any features or usage.

PROJECT FACTS:
- Source files: {', '.join(all_results.keys())}
- Dependencies: {deps_label}
- Documented functions:
{func_table}

Write exactly these sections in order:

# {project_name}
(2-sentence description of what this project does, from the function purposes above)

## Features
(bullet list of 4-6 key capabilities, derived from the function purpose column above)

## Installation
```bash
pip install {deps_str}
```

## Usage
(realistic code example using actual function names from the table above)

## Project Structure
(one line per file: filename — its role)

## API Reference
(paste the function table exactly as provided)
{security_section}

Keep everything concise and accurate. Do not add sections not listed here.
"""
    return generate(prompt, "text", "readme")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("   AI Code Documentation Generator")
    print("   Standard  : IBM 4-Tier + IJIRT Paper")
    print("   Pipeline  : Groq → GPT-4o → gpt-4o-mini + Phi-4")
    print("   Features  : Regex Parser · Incremental · Security Scan")
    print("=" * 60)

    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        zip_path = input("\nEnter ZIP file path: ").strip().strip('"')

    if not os.path.exists(zip_path):
        print(f"❌ File not found: {zip_path}")
        sys.exit(1)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR,  exist_ok=True)
    print(f"\n  Cache entries: {cache_mod.count()}")

    # ── Step 1: Extract ZIP ───────────────────────────────────────────────
    print("\n[1/7] Extracting ZIP...")
    ok, count, err = extract_zip(zip_path)
    if not ok:
        print(f"  ❌ {err}")
        sys.exit(1)
    print(f"  ✅ Extracted {count} source files")

    # ── Step 2: Scan + git change detection ───────────────────────────────
    print("\n[2/7] Scanning files...")
    files         = scan_files(Config.EXTRACT_DIR)
    changed_files = _get_changed_files(Config.EXTRACT_DIR)

    print(f"  Found {len(files)} source file(s):")
    for f in files:
        status = ""
        if changed_files is not None:
            status = " ← CHANGED" if f.name in changed_files else " (unchanged)"
        print(f"    → {f.name}{status}")

    # ── Step 3: Init document ─────────────────────────────────────────────
    print("\n[3/7] Initialising document...")
    project_name = Path(zip_path).stem
    doc          = create_doc(project_name)
    # Note: Gemini context cache setup removed (Gemini no longer used)

    # ── Step 4: Generate documentation ────────────────────────────────────
    print("\n[4/7] Generating IBM documentation...")
    all_results    : dict = {}
    all_summaries  : list = []
    skipped_files  = 0
    trivial_count  = 0
    pipeline_count = 0

    for i, file_path in enumerate(files, 1):
        lang = detect_language(file_path)
        print(f"\n  ── [{i}/{len(files)}] {file_path.name} ({lang}) ──")

        if not _should_process(file_path, changed_files):
            print("  ⏭ Skipping — unchanged since last commit")
            skipped_files += 1
            continue

        units = parse_file(file_path)
        print(f"  Found {len(units)} unit(s) to document")

        if not units:
            print("  Skipping — no parseable units found")
            continue

        add_file_heading(doc, file_path.name, lang)

        cx_map: dict = {}
        if lang == "python":
            cx_list = analyze(file_path)
            cx_map  = {c.name: c for c in cx_list}
        else:
            cx_map = _simple_complexity(units)

        results = process_file(units)
        all_results[file_path.name] = results

        for r in results:
            if r.get("_trivial"):
                trivial_count  += 1
            else:
                pipeline_count += 1

            add_unit_doc(
                doc,
                name          = r["name"],
                unit_type     = r["type"],
                documentation = r["final"],
                score         = r["score"],
            )

        if cx_map:
            add_complexity_table(doc, list(cx_map.values()))

        lines = []
        for r in results:
            facts  = r.get("facts", {})
            p      = facts.get("purpose", "")
            intent = facts.get("intent", "")
            sec    = "; ".join(facts.get("security_notes", [])[:2])
            mark   = "⚡" if r.get("_trivial") else "◆"
            line   = f"{mark} **{r['name']}:** {p}"
            if intent:
                line += f" | {intent}"
            if sec:
                line += f" | 🔒 {sec}"
            lines.append(line)

        all_summaries.append(f"## {file_path.name} ({lang})\n" + "\n".join(lines))

    # ── IJIRT: Cross-file dependency resolution ────────────────────────────
    print("\n[4b] Resolving cross-file dependencies (IJIRT)...")
    resolve_all_dependencies(all_results)
    print("  ✅ Dependencies resolved")

    # ── Step 5: Architecture diagram ──────────────────────────────────────
    # P5 FIX: generate_architecture returns (img, analyses) tuple
    print("\n[5/7] Generating architecture diagram...")
    try:
        img, file_analyses = generate_architecture(
            files, f"{Config.OUTPUT_DIR}architecture.png"
        )
        # Pass file_analyses so add_architecture can use text fallback if needed
        add_architecture(doc, img, file_analyses)
        if img:
            print("  ✅ Diagram added")
        else:
            print("  ⚠️  Diagram image unavailable — text fallback used")
    except Exception as e:
        print(f"  ⚠️  Architecture skipped entirely: {e}")

    # ── Step 6: Aggregate findings ────────────────────────────────────────
    print("\n[6/7] Aggregating security findings and generating README...")

    all_security    : list = []
    all_decisions   : list = []
    all_deps        : set  = set()
    security_by_func: dict = {}

    for filename, results in all_results.items():
        for r in results:
            facts     = r.get("facts", {})
            sec_notes = facts.get("security_notes", [])
            if sec_notes:
                all_security.extend(sec_notes)
                security_by_func[r["name"]] = {
                    "file"     : filename,
                    "notes"    : sec_notes,
                    "deep_scan": r.get("security", {}),
                }
            all_decisions.extend(facts.get("design_decisions", []))
            all_deps.update(
                d for d in facts.get("dependencies", []) if isinstance(d, str)
            )

    readme_content = _generate_readme(
        project_name, all_results, all_deps, all_security
    )
    readme_path = f"{Config.OUTPUT_DIR}{project_name}_README.md"
    Path(readme_path).write_text(readme_content, encoding="utf-8")
    print(f"  ✅ README.md → {readme_path}")

    # ── Step 7: IBM project summary ────────────────────────────────────────
    print("\n[7/7] Writing IBM project summary + saving...")

    security_block  = "\n".join(f"- {s}" for s in set(all_security)  if s) or "None identified."
    decisions_block = "\n".join(f"- {d}" for d in set(all_decisions) if d) or "None documented."
    pypi_deps       = _resolve_pypi_names(all_deps)
    deps_block      = ", ".join(f"`{d}`" for d in pypi_deps) or "None."

    summary_prompt = f"""\
Write a professional IBM-standard project overview with exactly four sections.
Use markdown formatting (### headings, **bold**, - bullets).

### PROJECT PURPOSE
(2 paragraphs: what this project does, who it serves, the business problem it solves)

### ARCHITECTURE & COMPONENTS
(1 paragraph: how the files and functions work together, key data flows)

### TECHNOLOGY STACK
Installed dependencies: {deps_block}
(For each major dependency, explain specifically why it is used in this project.)

### SECURITY & DESIGN NOTES
Security findings identified across the codebase:
{security_block}

Key design decisions:
{decisions_block}

Source material (function summaries by file):
{chr(10).join(all_summaries[:6])}
"""
    summary = generate(summary_prompt, "text", "summary")
    # P6 FIX: add_summary now routes through _add_markdown() — see doc_builder.py
    add_summary(doc, summary)

    out_path = f"{Config.OUTPUT_DIR}{project_name}_docs.docx"

    if not save_doc(doc, out_path):
        print(f"\n  ❌ Documentation file could not be saved to {out_path}")
        print("     Check that the file is not open in Word and retry.")
        sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ✅ Documentation → {out_path}")
    print(f"  ✅ README        → {readme_path}")
    print(f"\n  📊 Run statistics:")
    print(f"     Files processed : {len(all_results)}")
    print(f"     Files skipped   : {skipped_files}  (git: unchanged)")
    print(f"     Trivial funcs   : {trivial_count}  (template docs — no API cost)")
    print(f"     Pipeline funcs  : {pipeline_count}  (full 3-stage)")
    print(f"     Cache entries   : {cache_mod.count()}")

    if security_by_func:
        print(f"     🔒 Security flags: {len(security_by_func)} function(s)")
        limit = Config.MAX_SECURITY_DISPLAY
        items = list(security_by_func.items())
        if limit is not None:
            items = items[:limit]
        for fname, info in items:
            risk     = info.get("deep_scan", {}).get("overall_risk", "")
            risk_str = f"  [{risk.upper()}]" if risk else ""
            print(f"        {fname} ({info['file']}){risk_str}")

    print_usage()
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()