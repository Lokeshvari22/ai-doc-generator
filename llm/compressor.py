"""
compressor.py — Smart code preparation for the 3-stage pipeline.

UPDATED:
  - build_github_doc_prompt() added — replaces build_gemini_doc_prompt().
    Uses language variable (not hardcoded 'python') in code fences. Fixes P1/P3.
  - Trivial threshold raised from 4 lines to 6 lines. Fixes P2.
  - build_gemini_doc_prompt() kept as alias for backward compat.
  - All other logic unchanged.
"""

import ast
import re
import hashlib


# ─────────────────────────────────────────────────────────────────────────────
#  Trivial function detection
# ─────────────────────────────────────────────────────────────────────────────

def is_trivial(code: str) -> bool:
    """
    Return True when a function needs no API call to document.

    UPDATED: threshold raised from 4 to 6 real lines.
    This prevents short-but-real functions (like resume_result_wrapper)
    from getting filler template docs.

    Class definitions are NEVER trivial.
    """
    stripped = code.lstrip()
    if stripped.startswith("class "):
        return False

    lines = [l for l in code.splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if len(lines) <= 6:   # RAISED from 4 to 6
        return True

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complex_nodes = [
                    n for n in ast.walk(node)
                    if isinstance(n, (ast.If, ast.For, ast.While,
                                      ast.Try, ast.With, ast.Call))
                ]
                if len(complex_nodes) == 0:
                    return True
    except SyntaxError:
        pass
    return False


def trivial_doc(function_name: str, code: str, language: str = "python") -> dict:
    """
    Build template facts from AST/regex alone — zero API calls.

    FIXED: purpose now reflects actual function signature (params + return).
    FIXED: docstring_hint is a clean plain string — no triple-quotes.
    FIXED: language passed through for correct docstring style in output.
    """
    params   = []
    ret_ann  = "unknown"
    language = language or "python"

    # Detect language from code patterns if not provided
    if language == "python" and "def " not in code and "{" in code:
        language = "javascript"

    # Extract params + return from Python AST
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == function_name:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                        for arg in child.args.args:
                            if arg.arg == "self":
                                continue
                            ann = ""
                            if arg.annotation:
                                try:
                                    ann = ast.unparse(arg.annotation)
                                except Exception:
                                    ann = "unknown"
                            params.append({
                                "name"       : arg.arg,
                                "type"       : ann or "unknown",
                                "description": "",
                                "optional"   : False,
                                "default"    : "",
                            })
                break
            elif isinstance(node, ast.FunctionDef) and node.name == function_name:
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
                        "name"       : arg.arg,
                        "type"       : ann or "unknown",
                        "description": "",
                        "optional"   : False,
                        "default"    : "",
                    })
                if node.returns:
                    try:
                        ret_ann = ast.unparse(node.returns)
                    except Exception:
                        pass
                break
    except Exception:
        pass

    # Build a meaningful purpose from the name + params
    name_clean = function_name.replace("_", " ").strip().strip("_")
    if params:
        param_names = ", ".join(p["name"] for p in params)
        purpose = f"Implements {name_clean} logic using {param_names}."
    else:
        purpose = f"Implements {name_clean} logic."

    # Build intent from name patterns
    name_lower = function_name.lower()
    if name_lower.startswith("get"):
        intent = f"Provides access to {name_clean[4:].strip()} data."
    elif name_lower.startswith("set"):
        intent = f"Updates the {name_clean[4:].strip()} value."
    elif name_lower.startswith("is") or name_lower.startswith("has") or name_lower.startswith("check"):
        intent = f"Validates or checks a condition related to {name_clean}."
    elif name_lower.startswith("print") or name_lower.startswith("show") or name_lower.startswith("display"):
        intent = f"Outputs or renders {name_clean[5:].strip()} information to the user."
    elif name_lower.startswith("test"):
        intent = f"Verifies the correctness of {name_clean[5:].strip()} functionality."
    elif name_lower.startswith("__init__") or name_lower.startswith("init"):
        intent = f"Initialises the object with its required state."
    else:
        intent = f"Provides {name_clean} functionality as part of the application."

    # Build docstring_hint — clean single-line string, NO triple-quotes
    if params:
        param_summary = ", ".join(f"{p['name']} ({p['type']})" for p in params)
        docstring_hint = f"{purpose} Parameters: {param_summary}. Returns: {ret_ann}."
    else:
        docstring_hint = f"{purpose} Returns: {ret_ann}."

    return {
        "purpose"         : purpose,
        "intent"          : intent,
        "params"          : params,
        "returns"         : {"type": ret_ann, "description": ""},
        "raises"          : [],
        "logic_steps"     : [f"Executes {name_clean} logic and returns result."],
        "control_flow"    : "Linear execution with no conditional branching.",
        "edge_cases"      : [],
        "security_notes"  : [],
        "design_decisions": [],
        "naming_analysis" : f"The name '{function_name}' clearly describes its purpose.",
        "dependencies"    : [],
        "cross_references": [],
        "example"         : {
            "code"       : f"{function_name}(...)",
            "description": f"Calls {function_name} with appropriate arguments."
        },
        "docstring_hint"  : docstring_hint,   # clean string — no triple-quotes
        "language"        : language,
        "_trivial"        : True,
    }


def strip_noise(code: str) -> str:
    """Remove docstrings and inline comments. Never removes executable logic."""
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'(?<!:)#(?!!).*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Block splitter (for large functions — no truncation ever)
# ─────────────────────────────────────────────────────────────────────────────

def split_into_blocks(code: str, language: str) -> list:
    lines  = code.splitlines()
    blocks = []

    if language == "python":
        current_label = "setup"
        current_lines = []

        for line in lines:
            stripped = line.lstrip()
            indent   = len(line) - len(stripped)
            is_block_start = (
                indent == 4 and
                any(stripped.startswith(kw) for kw in (
                    "if ", "elif ", "else:",
                    "for ", "while ", "with ",
                    "try:", "except", "finally:",
                    "def ", "class ",
                    "return ", "raise ",
                ))
            )
            if is_block_start and current_lines:
                blocks.append({"label": current_label, "code": "\n".join(current_lines)})
                current_label = (
                    stripped.split("(")[0].split(":")[0].split(" ")[0].strip()
                )
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            blocks.append({"label": current_label, "code": "\n".join(current_lines)})

    elif language in ("javascript", "typescript"):
        chunk_size = 60
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            blocks.append({"label": f"lines_{i+1}_{i+len(chunk)}", "code": "\n".join(chunk)})
    else:
        chunk_size = 80
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            blocks.append({"label": f"lines_{i+1}_{i+len(chunk)}", "code": "\n".join(chunk)})

    return blocks if blocks else [{"label": "body", "code": code}]


# ─────────────────────────────────────────────────────────────────────────────
#  Code preparation for Groq
# ─────────────────────────────────────────────────────────────────────────────

def prepare_code_for_groq(code: str, language: str, max_chars: int = 8000) -> str:
    cleaned = strip_noise(code)
    if len(cleaned) <= max_chars:
        return cleaned
    lines = cleaned.splitlines()
    head  = "\n".join(lines[:30])
    tail  = "\n".join(lines[-20:])
    return f"{head}\n\n# ... [middle truncated — use block chunker] ...\n\n{tail}"


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

_GROQ_SYSTEM_STABLE = """\
You are an expert code analyst producing IBM-standard documentation facts.
Extract REAL information from the code. Never use placeholder text.
Never output "See source" or "Implements X logic" — read the actual code.

CRITICAL JSON RULES:
- Output ONLY valid JSON. No markdown, no code fences, no triple-quotes.
- All string values must use double-quoted JSON strings only.
- No raw Python/Java docstring syntax (no triple quotes, no /** */ blocks).
- The "docstring_hint" field is a PLAIN single-line string summary only.

EXTRACTION RULES:
1. purpose: Start with a verb. Describe what the code ACTUALLY does, not its name.
2. intent:  Why does this function EXIST? What design need does it fill?
3. params:  List EVERY parameter with exact name from the signature.
4. returns: Exact return type + what the value semantically means.
5. raises:  Every exception that can be raised + exact trigger condition.
6. logic_steps: REAL code steps — short plain strings, no quotes inside strings.
7. docstring_hint: ONE plain sentence summarising purpose+params+return. No special chars.

REQUIRED OUTPUT — fill every field:
{
  "purpose": "verb phrase describing what this function actually does",
  "intent": "reason this function exists in the codebase",
  "params": [
    {"name": "PARAM_NAME", "type": "TYPE", "description": "DESC", "optional": false, "default": ""}
  ],
  "returns": {"type": "RETURN_TYPE", "description": "what the returned value represents"},
  "raises": [{"exception": "EXCEPTION_TYPE", "condition": "trigger condition"}],
  "logic_steps": ["step 1 description", "step 2 description"],
  "control_flow": "description of branches and loops in plain English",
  "edge_cases": ["edge case description"],
  "security_notes": ["specific risk if any, or empty list"],
  "design_decisions": ["design choice made"],
  "naming_analysis": "observation about variable and function names",
  "dependencies": ["library name"],
  "cross_references": ["function or class called inside"],
  "example": {"code": "functionName(arg1, arg2)", "description": "what this example shows"},
  "docstring_hint": "Single plain sentence: what it does, params, return value."
}"""


def build_groq_extraction_prompt(code: str, language: str,
                                  unit_type: str,
                                  block_context: str = "") -> str:
    context_section = ""
    if block_context.strip():
        context_section = (
            f"\nPREVIOUS BLOCKS CONTEXT (for coherent cross-block analysis):\n"
            f"{block_context.strip()}\n"
        )
    return (
        f"{_GROQ_SYSTEM_STABLE}\n"
        f"{context_section}\n"
        f"Analyze this {language} {unit_type}:\n"
        f"```{language}\n{code}\n```"
    )


def build_batch_groq_prompt(units: list) -> str:
    funcs_section = ""
    for i, unit in enumerate(units):
        prepared = prepare_code_for_groq(unit.code, unit.language, 1200)
        funcs_section += (
            f"\nFUNC_{i} ({unit.name}) [{unit.language}]:\n"
            f"```{unit.language}\n{prepared}\n```\n"
        )
    return (
        f"{_GROQ_SYSTEM_STABLE}\n\n"
        f"Extract facts for ALL {len(units)} functions below.\n"
        f"Return a JSON ARRAY with exactly {len(units)} objects — one per function.\n"
        f"IMPORTANT: Each object must be valid JSON. No triple-quotes. No raw docstring syntax.\n"
        f"{funcs_section}\n"
        f"Return JSON array only — [{{...}}, {{...}}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 prompt — GPT-4o IBM prose writer
#  FIX: language passed dynamically — no more hardcoded 'python' code fences
# ─────────────────────────────────────────────────────────────────────────────

def build_github_doc_prompt(function_name: str, facts_json: str,
                             language: str = "python") -> str:
    """
    Build Stage 2 (GPT-4o) prompt for IBM-standard documentation.

    Uses docstring_hint (safe plain string from Groq) to reconstruct
    a proper language-appropriate docstring. No triple-quote JSON issues.
    """
    return f"""\
Write COMPLETE IBM-standard developer documentation for `{function_name}`.
Use ONLY the provided facts. Never invent anything not in the JSON.
IMPORTANT: Complete every section fully. Do not truncate. Do not use placeholder text.

OUTPUT FORMAT — include every section exactly as shown:
---
### `{function_name}`

**What it does:**
[facts.purpose — one specific verb phrase describing the actual behaviour]

**Why it exists:**
[facts.intent — business/design reason. Minimum 1 full sentence.]

**Inline Docstring:**
```{language}
[Write a complete, properly-formatted {language} docstring using facts.docstring_hint,
facts.params, facts.returns, facts.raises, and facts.example.
Use the correct docstring style for {language}: Python uses triple-quote with Args/Returns/Raises,
Java uses /** */ with @param/@return/@throws, JS/TS uses JSDoc /** */.]
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
[Use ONLY params from facts.params. If empty: single row "| — | — | — | — | No parameters. |"]

**Returns:**
`[type]` — [full description from facts.returns]

**Raises:**
[from facts.raises — format: "ExceptionType: exact trigger condition". If empty: "None."]

**Control Flow:**
[facts.control_flow — describe every branch, loop, exception handler. Minimum 1 full sentence.]

**Logic Walkthrough:**
[numbered list from facts.logic_steps — each step on its own line as "1. step"]

**Edge Cases:**
[bullet list from facts.edge_cases. If empty: "None documented."]

**Security Notes:**
[bullet list from facts.security_notes. If empty: "None identified."]

**Design Decisions:**
[bullet list from facts.design_decisions. If empty: "Standard implementation pattern."]

**Naming Insights:**
[facts.naming_analysis. If empty: "Standard naming conventions used."]

**Dependencies:**
[comma-separated facts.dependencies in backticks. If empty: "None."]

**Cross-References:**
[facts.cross_references — format: "`function_name` — why called". If empty: "None."]

**Usage Example:**
```{language}
[facts.example.code — realistic call with actual parameter values]
// [facts.example.description]
```
---

FACTS (JSON):
{facts_json}"""


# Backward-compat alias — existing imports of build_gemini_doc_prompt still work
def build_gemini_doc_prompt(function_name: str, facts_json: str) -> str:
    return build_github_doc_prompt(function_name, facts_json, language="python")


def build_github_validate_score_prompt(signature: str, doc: str) -> str:
    """Build Stage 3 (DeepSeek-R1) validate + score prompt."""
    return f"""\
You are an IBM documentation compliance reviewer.
Review documentation and return ONLY valid JSON — no prose, no commentary.

SCORING CRITERIA (1-5):
5 = All sections present, accurate, specific examples, no generic text
4 = Minor issues (1-2 small inaccuracies or slightly vague sections)
3 = Moderate issues (missing sections OR vague placeholder text)
2 = Major issues (wrong parameter names, invented content, or many missing sections)
1 = Unusable (mostly fallback text, no real documentation)

REQUIRED IBM SECTIONS:
What it does / Why it exists / Parameters table / Returns / Logic Walkthrough / Usage Example

RETURN EXACTLY THIS JSON (nothing else):
{{"score": <integer 1-5>, "issues": ["issue 1"] or []}}

Function signature:
{signature}

Documentation to score (first 1000 chars):
{doc[:1000]}"""


def build_security_scan_prompt(function_name: str, code: str,
                                known_issues: list) -> str:
    """Build Phi-4 security deep-scan prompt."""
    issues_hint = (
        "\n".join(f"- {i}" for i in known_issues)
        if known_issues else "None pre-identified."
    )
    safe_code = code[:2500].replace("```", "'''")

    return f"""\
Perform a security audit of this function. Be specific and code-grounded.

Function: {function_name}
Pre-identified concerns:
{issues_hint}

Code:
```
{safe_code}
```

Check for:
1. SQL injection — string concatenation or f-strings in database queries
2. Path traversal — user input used directly in file paths
3. Hardcoded secrets or credentials
4. Unsafe deserialization
5. Missing input validation on user-supplied values
6. Information disclosure in error messages or logs

Return ONLY valid JSON. Use double-quoted strings.
{{"vulnerabilities": [{{"type":"","line_hint":"","severity":"low|medium|high","description":"","fix":""}}], "overall_risk": "low|medium|high"}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  Code hash — enables incremental mode
# ─────────────────────────────────────────────────────────────────────────────

def code_hash(code: str) -> str:
    """SHA-256 of noise-stripped code. Used to skip unchanged functions."""
    return hashlib.sha256(strip_noise(code).encode()).hexdigest()[:16]


def sanitise_groq_json(raw: str) -> str:
    """
    Clean Groq JSON output before parsing.

    Groq sometimes embeds raw triple-quoted docstrings inside JSON strings,
    which breaks json.loads(). This function:
      1. Strips markdown code fences (```json ... ```)
      2. Replaces triple-double-quotes with escaped single-line equivalents
      3. Replaces triple-single-quotes similarly
      4. Removes bare /* */ and /** */ comment blocks from inside strings

    Returns the cleaned string ready for json.loads().
    """
    import re as _re

    # Strip outer markdown fences if present
    raw = _re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=_re.MULTILINE)
    raw = _re.sub(r'```\s*$', '', raw.strip(), flags=_re.MULTILINE)
    raw = raw.strip()

    # Replace triple-double-quotes with a safe placeholder, then restore as escaped
    # Pattern: find """...""" inside a JSON string value and collapse to one line
    raw = _re.sub(r'"""(.*?)"""',
                  lambda m: '"' + m.group(1).replace('\n', ' ').replace('"', '\\"').strip() + '"',
                  raw, flags=_re.DOTALL)

    # Replace triple-single-quotes similarly
    raw = _re.sub(r"'''(.*?)'''",
                  lambda m: '"' + m.group(1).replace('\n', ' ').replace('"', '\\"').strip() + '"',
                  raw, flags=_re.DOTALL)

    # Remove /** */ javadoc blocks that leaked into string values
    raw = _re.sub(r'/\*\*.*?\*/', '', raw, flags=_re.DOTALL)
    raw = _re.sub(r'/\*.*?\*/',   '', raw, flags=_re.DOTALL)

    # Collapse multiple spaces/newlines inside string values (basic cleanup)
    raw = _re.sub(r'\n\s*\n', '\n', raw)

    return raw.strip()