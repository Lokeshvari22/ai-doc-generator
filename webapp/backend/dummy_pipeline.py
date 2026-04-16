"""
dummy_pipeline.py — Zero-API-cost test mode for the web app.

When DUMMY_MODE=true in .env (or Groq quota is exhausted), this module
replaces the real pipeline and produces output with the EXACT same structure
that the real pipeline returns.

Why this matters:
  - The frontend, database, doc_builder, and SSE streaming all work
    identically in dummy mode — no special-casing needed.
  - You can test the full web flow (upload → stream logs → download .docx)
    without spending any API tokens.
  - The dummy facts are extracted from the real code via AST — so param
    names, function names, and structure are accurate, just no LLM prose.

To enable: set DUMMY_MODE=true in your .env file.
To disable: set DUMMY_MODE=false (or remove the line entirely).
"""

import ast
import time
import random
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  AST-based fact extraction (no API call)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_facts_from_ast(code: str, name: str) -> dict:
    """
    Extract real facts using only Python's AST — zero tokens spent.
    Returns the same 14-field dict the real Groq Stage 1 returns.
    """
    params     = []
    ret_ann    = "unknown"
    logic_steps = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == name:
                # Params
                for arg in node.args.args:
                    if arg.arg == "self":
                        continue
                    ann = ""
                    if arg.annotation:
                        try:
                            ann = ast.unparse(arg.annotation)
                        except Exception:
                            ann = "unknown"
                    params.append({
                        "name"        : arg.arg,
                        "type"        : ann or "unknown",
                        "description" : f"Parameter {arg.arg}",
                        "optional"    : False,
                        "default"     : "",
                    })
                # Return annotation
                if node.returns:
                    try:
                        ret_ann = ast.unparse(node.returns)
                    except Exception:
                        pass
                # Logic steps from statement types
                stmt_names = {
                    ast.If:     "Conditional branch",
                    ast.For:    "Loop iteration",
                    ast.While:  "While loop",
                    ast.Try:    "Exception handling",
                    ast.With:   "Context manager",
                    ast.Return: "Returns a value",
                    ast.Assign: "Variable assignment",
                    ast.Expr:   "Expression / function call",
                }
                for stmt in node.body[:8]:
                    for stype, label in stmt_names.items():
                        if isinstance(stmt, stype):
                            logic_steps.append(label)
                            break
                break
    except Exception:
        pass

    clean_name = name.replace("_", " ").strip("_")
    purpose    = f"Implements {clean_name} functionality."

    return {
        "purpose"         : purpose,
        "intent"          : f"Provides {clean_name} capability to the application.",
        "params"          : params,
        "returns"         : {"type": ret_ann, "description": "Return value"},
        "raises"          : [],
        "logic_steps"     : logic_steps or ["See source implementation."],
        "control_flow"    : "Sequential execution" if not logic_steps else
                            "Conditional branches and loops as described in logic steps.",
        "edge_cases"      : ["Empty input", "None value passed"],
        "security_notes"  : [],
        "design_decisions": ["Standard implementation pattern"],
        "naming_analysis" : f"Name '{name}' describes its primary purpose.",
        "dependencies"    : [],
        "cross_references": [],
        "example"         : {
            "code"        : f"result = {name}({', '.join(p['name'] for p in params[:3])})",
            "description" : f"Basic usage of {name}",
        },
        "docstring"       : f'"""\n{purpose}\n"""',
    }


def _build_dummy_doc(name: str, facts: dict) -> str:
    """
    Build an IBM-formatted documentation string from AST facts.
    Same format as Gemini Stage 2 would produce — just without prose quality.
    """
    # Parameter table
    if facts["params"]:
        rows = "\n".join(
            f"| `{p['name']}` | `{p['type']}` | Yes | — | {p['description']} |"
            for p in facts["params"]
        )
    else:
        rows = "| — | — | — | — | No parameters. |"

    params_table = (
        "| Parameter | Type | Required | Default | Description |\n"
        "|-----------|------|----------|---------|-------------|\n"
        + rows
    )

    # Logic walkthrough
    steps = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(facts["logic_steps"])
    )

    ret = facts["returns"]
    ret_str = f"`{ret['type']}` — {ret['description']}"

    return f"""\
---
### `{name}`

**What it does:**
{facts['purpose']}

**Why it exists:**
{facts['intent']}

**Inline Docstring:**
```python
{facts['docstring']}
```

**Parameters:**
{params_table}

**Returns:**
{ret_str}

**Raises:**
None documented.

**Control Flow:**
{facts['control_flow']}

**Logic Walkthrough:**
{steps}

**Edge Cases:**
- {chr(10).join('- ' + e for e in facts['edge_cases'])}

**Security Notes:**
None identified (dummy mode — run real pipeline for security analysis).

**Design Decisions:**
- {facts['design_decisions'][0]}

**Naming Insights:**
{facts['naming_analysis']}

**Dependencies:**
None detected.

**Cross-References:**
None.

**Usage Example:**
```python
{facts['example']['code']}
# {facts['example']['description']}
```

> ⚠ **Dummy Mode** — This documentation was generated using AST analysis only
> (no Groq/Gemini/GitHub calls). Re-run with real API keys when quota resets.
---"""


# ─────────────────────────────────────────────────────────────────────────────
#  Public: process a list of CodeUnit objects (same API as pipeline.process_file)
# ─────────────────────────────────────────────────────────────────────────────

def process_file_dummy(units: list, log_fn=None) -> list:
    """
    Drop-in replacement for pipeline.process_file().
    Returns the EXACT same list-of-dict structure. Zero API calls.

    log_fn: optional callable(msg, level) for SSE streaming
    """
    def _log(msg: str, level: str = "info"):
        print(f"[DUMMY] {msg}")
        if log_fn:
            log_fn(msg, level)

    results = []

    for unit in units:
        _log(f"  [{unit.unit_type}] {unit.name} — AST extraction (dummy mode)", "info")
        time.sleep(0.3)   # small delay so the log stream looks realistic

        facts = _extract_facts_from_ast(unit.code, unit.name)
        doc   = _build_dummy_doc(unit.name, facts)
        score = round(random.uniform(2.8, 3.5), 1)   # realistic dummy score

        _log(f"    ✓ {facts['purpose'][:60]}", "success")

        results.append({
            "name"         : unit.name,
            "type"         : unit.unit_type,
            "facts"        : facts,
            "documentation": doc,
            "final"        : doc,
            "score"        : score,
            "security"     : {},
            "_trivial"     : False,
            "_dummy"       : True,
        })

    return results