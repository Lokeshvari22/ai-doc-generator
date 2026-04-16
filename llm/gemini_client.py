"""
gemini_client.py — Stage 2: Write IBM-standard prose from structured facts.

FIX: Context cache failure is now persisted to a flag file so subsequent
     runs skip the doomed caches.create() call on the free tier.
FIX: Flat imports (was llm.rate_limiter etc).
"""

import ast
import json
import os
from pathlib import Path
from google import genai
from google.genai import types
from config import Config
from llm.rate_limiter import RateLimiter, with_backoff
from llm.compressor import build_gemini_doc_prompt     
import llm.cache as cache

client  = genai.Client(api_key=Config.GEMINI_API_KEY)
limiter = RateLimiter(rpm=Config.GEMINI_RPM, name="Gemini", rpd=Config.GEMINI_RPD)

_cache_name: str | None = None

# FIX: flag file — if Gemini context cache is unsupported, we skip it forever
_CACHE_UNSUPPORTED_FLAG = os.path.join(Config.CACHE_DIR, "gemini_cache_unsupported.flag")
os.makedirs(Config.CACHE_DIR, exist_ok=True)


def _cache_is_unsupported() -> bool:
    """Return True if a previous run confirmed context cache is unavailable."""
    return os.path.exists(_CACHE_UNSUPPORTED_FLAG)


def _mark_cache_unsupported():
    """Persist the 'context cache not available' state for future runs."""
    try:
        Path(_CACHE_UNSUPPORTED_FLAG).write_text(
            "Gemini context cache unavailable on this account tier.\n"
            "Delete this file to retry on the next run.\n"
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  AST signature extractor (zero API cost)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_signatures(files: list) -> str:
    """
    Extract all function/class signatures from source files via AST.
    Used to pad the Gemini context cache past the 1024-token minimum.
    Zero API cost — pure local parsing.
    """
    sigs = []
    for f in files:
        try:
            path   = Path(f)
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
            sigs.append(f"\n--- {path.name} ---")
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    sigs.append(f"class {node.name}:")
                elif isinstance(node, ast.FunctionDef):
                    args = []
                    for arg in node.args.args:
                        ann = ""
                        if arg.annotation:
                            try:
                                ann = f": {ast.unparse(arg.annotation)}"
                            except Exception:
                                pass
                        args.append(f"{arg.arg}{ann}")
                    ret = ""
                    if node.returns:
                        try:
                            ret = f" -> {ast.unparse(node.returns)}"
                        except Exception:
                            pass
                    sigs.append(f"  def {node.name}({', '.join(args)}){ret}")
        except Exception:
            continue
    return "\n".join(sigs)


# ─────────────────────────────────────────────────────────────────────────────
#  Context cache setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_context_cache(project_name: str, files: list):
    """
    Upload project context once to Gemini servers.
    All write_documentation() calls reuse it — saves ~80% input tokens.

    FIX: If a prior run already confirmed the cache is unavailable (free tier
         returns limit=0), skip immediately without making an API call.
    """
    global _cache_name

    # FIX: skip the doomed API call if we already know it will fail
    if _cache_is_unsupported():
        print("  ℹ Gemini context cache skipped (unsupported on this tier)")
        print("  → All Gemini calls will pay full token cost")
        _cache_name = None
        return

    style_guide = f"""You are a senior technical writer producing IBM-standard developer \
documentation for the software project "{project_name}".

IBM DOCUMENTATION STANDARD — 4 TIERS:
TIER 1 — LOW-LEVEL: Inline docstrings — purpose, parameters with types, returns, examples.
  Every parameter: name, type hint, required/optional, default value, description.
  Returns: type and what the value represents (not just the type name).
  Raises: exception type and exact condition that triggers it.
  Usage Example: realistic call using actual parameter names, not placeholder strings.

TIER 2 — HIGH-LEVEL: Control flow description, module dependency map, cross-file references.
TIER 3 — EXTERNAL: README-style overview, installation, CLI/API usage guide.
TIER 4 — INTERNAL: Design decisions (why this approach), security findings, naming conventions.

IBM QUALITY RULES — every doc must obey all of these:
- Document the WHY: explain design decisions, not just what the code does.
- Active voice: "Returns the sum" not "The sum is returned by this function".
- No platitudes: "use secure methods" is not a security note. Be code-specific.
- Every security note must reference specific code: line context, variable, or pattern.
- No invented information — base everything only on the provided JSON facts.
- No truncated sentences — every section must be complete.
- Specific over generic: advice that applies to all code is not useful documentation.

REQUIRED SECTIONS — all must be present and non-empty in every function doc:
What it does | Why it exists | Inline Docstring | Parameters | Returns |
Raises | Control Flow | Logic Walkthrough | Edge Cases | Security Notes |
Design Decisions | Naming Insights | Dependencies | Cross-References | Usage Example

DETAILED PARAMETER RULES:
- Parameter names must EXACTLY match the function signature — no paraphrasing.
- Types: use Python type hints (str, int, list[str], dict[str, Any], etc.).
- Optional parameters must show default value in the Default column.
- Description must explain WHAT the parameter controls, not restate its name.

DETAILED SECURITY RULES:
- SQL injection: flag string concatenation or f-strings in any database query.
- Path traversal: flag user input used in file open(), os.path.join(), etc.
- Hardcoded secrets: flag any string literal that looks like a password or key.
- Missing validation: flag user-supplied values used without bounds checking.
- Error disclosure: flag exception handlers that expose system paths or internals.

DETAILED EXAMPLE RULES:
- Use realistic values, not "your_value" or "example_string" placeholders.
- Show the function being called, not being defined.
- Add a comment explaining what the example demonstrates."""

    signatures  = _extract_signatures(files)
    sig_section = f"\nPROJECT FUNCTION INDEX (all signatures in this project):\n{signatures}"
    full_context = style_guide.strip() + "\n" + sig_section

    TARGET_CHARS = 5500

    padding_blocks = [
        """
ADDITIONAL IBM CROSS-REFERENCE RULES:
- Cross-References must list every function called inside this function by name.
- For each cross-reference: state what the called function does and why it is called here.
- If a function is imported from another module, note the source module name.
- If a function calls database operations, flag the exact table name and SQL operation type.
- If a function calls an external API, note the endpoint and the data it sends or receives.""",
        """
ADDITIONAL CONTROL FLOW RULES:
- Describe every conditional branch: what condition triggers it and what it does.
- For loops: describe what is being iterated, what accumulates, and the termination condition.
- For exception handlers: describe what exception is caught and the recovery action taken.
- Do not use generic phrases like "handles the logic" — be specific about each branch outcome.
- For nested conditions, describe the outer condition first then each inner branch separately.""",
        """
ADDITIONAL NAMING ANALYSIS RULES:
- Explain what each non-obvious variable name reveals about its purpose.
- Flag any names that are misleading or inconsistent with the code's actual behaviour.
- Note naming patterns: prefixes like is_, has_, get_, set_ reveal intent.
- Identify abbreviations and explain what they stand for if not immediately obvious.""",
        """
ADDITIONAL DOCSTRING FORMAT RULES:
- Args section: list every parameter with its type hint and a one-line description.
- Returns section: state the type and what the returned value represents semantically.
- Raises section: list every exception the function can raise with the exact trigger condition.
- Example section: show a complete realistic function call with real parameter values.
- The docstring must be complete enough to use the function without reading the source code.""",
        """
ADDITIONAL SECURITY AUDIT RULES:
- SQL injection: check every database query for string concatenation or f-string interpolation.
- Path traversal: check every file open or path join that uses user-supplied data.
- Hardcoded credentials: check for string literals containing passwords, keys, or tokens.
- Input validation: check every function that accepts external input for bounds or type checks.
- Sensitive data in logs: check for print() or log statements that output user data.
- Error disclosure: check exception handlers that expose stack traces or file paths to users.""",
    ]

    for block in padding_blocks:
        if len(full_context) >= TARGET_CHARS:
            break
        full_context += "\n" + block.strip()

    estimated_tokens = len(full_context) // 4
    print(f"  Context size: ~{estimated_tokens} estimated tokens ({len(full_context)} chars)")

    try:
        response = client.caches.create(
            model  = Config.GEMINI_MODEL,
            config = types.CreateCachedContentConfig(
                contents = [full_context],
                ttl      = Config.GEMINI_CACHE_TTL,
            ),
        )
        _cache_name = response.name
        print(f"  ✅ Gemini cache: {_cache_name[:50]}...")
    except Exception as e:
        print(f"  ⚠ Context cache failed: {e}")
        print("  → All Gemini calls will pay full token cost")
        _cache_name = None

        # FIX: persist the failure so future runs skip this immediately
        if "limit=0" in str(e) or "TotalCachedContent" in str(e):
            _mark_cache_unsupported()
            print("  ℹ Marked as unsupported — future runs will skip cache setup")


# ─────────────────────────────────────────────────────────────────────────────
#  Documentation writer
# ─────────────────────────────────────────────────────────────────────────────

def write_documentation(function_name: str, facts: dict) -> str:
    """
    Stage 2: Write full IBM-standard prose documentation from Groq's JSON facts.

    Uses context cache if available (saves ~80% input tokens).
    Falls back to _facts_to_doc_fallback() when Gemini is rate-limited.
    """
    cache_key = f"{function_name}:{json.dumps(facts, sort_keys=True)[:300]}"
    cached    = cache.get(cache_key, "gemini")
    if cached:
        print("      ✓ Cache hit (Gemini)")
        return cached

    facts_json = json.dumps(facts, indent=2)
    prompt     = build_gemini_doc_prompt(function_name, facts_json)

    def _call():
        limiter.wait()
        cfg = types.GenerateContentConfig(
            temperature       = 0.1,
            max_output_tokens = Config.GEMINI_MAX_OUT_TOKENS,
        )
        if _cache_name:
            cfg.cached_content = _cache_name
        resp = client.models.generate_content(
            model    = Config.GEMINI_MODEL,
            contents = prompt,
            config   = cfg,
        )
        return resp.text.strip()

    try:
        result = with_backoff(_call)
        cache.set(cache_key, "gemini", result)
        return result
    except Exception as e:
        print(f"      ✗ Gemini error: {e} — building from facts")
        return _facts_to_doc_fallback(function_name, facts)


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback: accurate IBM doc built directly from JSON facts
# ─────────────────────────────────────────────────────────────────────────────

def _facts_to_doc_fallback(function_name: str, facts: dict) -> str:
    """
    Build complete IBM-standard documentation without Gemini.
    Grounded entirely in Groq's extracted facts — never invents.
    """
    param_rows = []
    for p in facts.get("params", []):
        if isinstance(p, dict):
            opt     = "No" if not p.get("optional") else "Yes"
            default = p.get("default", "") or "—"
            param_rows.append(
                f"| `{p.get('name','?')}` | `{p.get('type','?')}` "
                f"| {opt} | `{default}` | {p.get('description','')} |"
            )
        elif isinstance(p, str):
            param_rows.append(f"| `{p}` | unknown | ? | — | — |")

    params_md = (
        "| Parameter | Type | Required | Default | Description |\n"
        "|-----------|------|----------|---------|-------------|\n"
        + ("\n".join(param_rows) if param_rows
           else "| — | — | — | — | No parameters. |")
    )

    ret      = facts.get("returns", {})
    ret_type = ret.get("type", "unknown") if isinstance(ret, dict) else str(ret)
    ret_desc = ret.get("description", "")  if isinstance(ret, dict) else ""

    raises_md = "\n".join(
        f"- `{r.get('exception','?')}` — {r.get('condition','')}"
        for r in facts.get("raises", []) if isinstance(r, dict)
    ) or "None."

    steps_md = "\n".join(
        f"{i+1}. {s}"
        for i, s in enumerate(facts.get("logic_steps", []))
        if isinstance(s, str)
    ) or "See source."

    def _bullets(key: str, empty: str = "None documented.") -> str:
        items = [x for x in facts.get(key, []) if isinstance(x, str) and x]
        return "\n".join(f"- {x}" for x in items) if items else empty

    edges    = _bullets("edge_cases",        "None documented.")
    security = _bullets("security_notes",    "None identified.")
    design   = _bullets("design_decisions",  "Standard implementation.")
    refs     = _bullets("cross_references",  "None.")
    deps     = (
        ", ".join(f"`{d}`" for d in facts.get("dependencies", []) if isinstance(d, str))
        or "None."
    )

    ex      = facts.get("example", {})
    ex_code = ex.get("code", f"{function_name}(...)") if isinstance(ex, dict) else f"{function_name}(...)"
    ex_desc = ex.get("description", "")               if isinstance(ex, dict) else ""

    docstring = (
        facts.get("docstring", "")
        or f'"""\n{facts.get("purpose", function_name)}.\n"""'
    )

    return f"""\
---
### `{function_name}`

**What it does:**
{facts.get('purpose', 'See source.')}

**Why it exists:**
{facts.get('intent', 'Core utility function.')}

**Inline Docstring:**
```python
{docstring}
```

**Parameters:**
{params_md}

**Returns:**
`{ret_type}` — {ret_desc}

**Raises:**
{raises_md}

**Control Flow:**
{facts.get('control_flow', 'See logic walkthrough below.')}

**Logic Walkthrough:**
{steps_md}

**Edge Cases:**
{edges}

**Security Notes:**
{security}

**Design Decisions:**
{design}

**Naming Insights:**
{facts.get('naming_analysis', 'Standard naming conventions.')}

**Dependencies:**
{deps}

**Cross-References:**
{refs}

**Usage Example:**
```python
{ex_code}
# {ex_desc}
```
---""""""
gemini_client.py — Stage 2: Write IBM-standard prose from structured facts.

FIX: Context cache failure is now persisted to a flag file so subsequent
     runs skip the doomed caches.create() call on the free tier.
FIX: Flat imports (was llm.rate_limiter etc).
"""

import ast
import json
import os
from pathlib import Path
from google import genai
from google.genai import types
from config import Config
from llm.rate_limiter import RateLimiter, with_backoff, DailyQuotaExhausted
from llm.compressor import build_gemini_doc_prompt
import llm.cache as cache

client  = genai.Client(api_key=Config.GEMINI_API_KEY)
limiter = RateLimiter(rpm=Config.GEMINI_RPM, name="Gemini", rpd=Config.GEMINI_RPD)

_cache_name: str | None = None

# Set to True the first time a DailyQuotaExhausted is raised.
# All subsequent write_documentation() calls skip the API entirely and go
# straight to _facts_to_doc_fallback() — no more 4-minute retry storms.
_daily_quota_exhausted: bool = False

# Flag file — if Gemini context cache is unsupported, we skip it forever
_CACHE_UNSUPPORTED_FLAG = os.path.join(Config.CACHE_DIR, "gemini_cache_unsupported.flag")
os.makedirs(Config.CACHE_DIR, exist_ok=True)


def _cache_is_unsupported() -> bool:
    """Return True if a previous run confirmed context cache is unavailable."""
    return os.path.exists(_CACHE_UNSUPPORTED_FLAG)


def _mark_cache_unsupported():
    """Persist the 'context cache not available' state for future runs."""
    try:
        Path(_CACHE_UNSUPPORTED_FLAG).write_text(
            "Gemini context cache unavailable on this account tier.\n"
            "Delete this file to retry on the next run.\n"
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  AST signature extractor (zero API cost)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_signatures(files: list) -> str:
    """
    Extract all function/class signatures from source files via AST.
    Used to pad the Gemini context cache past the 1024-token minimum.
    Zero API cost — pure local parsing.
    """
    sigs = []
    for f in files:
        try:
            path   = Path(f)
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
            sigs.append(f"\n--- {path.name} ---")
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    sigs.append(f"class {node.name}:")
                elif isinstance(node, ast.FunctionDef):
                    args = []
                    for arg in node.args.args:
                        ann = ""
                        if arg.annotation:
                            try:
                                ann = f": {ast.unparse(arg.annotation)}"
                            except Exception:
                                pass
                        args.append(f"{arg.arg}{ann}")
                    ret = ""
                    if node.returns:
                        try:
                            ret = f" -> {ast.unparse(node.returns)}"
                        except Exception:
                            pass
                    sigs.append(f"  def {node.name}({', '.join(args)}){ret}")
        except Exception:
            continue
    return "\n".join(sigs)


# ─────────────────────────────────────────────────────────────────────────────
#  Context cache setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_context_cache(project_name: str, files: list):
    """
    Upload project context once to Gemini servers.
    All write_documentation() calls reuse it — saves ~80% input tokens.

    FIX: If a prior run already confirmed the cache is unavailable (free tier
         returns limit=0), skip immediately without making an API call.
    """
    global _cache_name

    # FIX: skip the doomed API call if we already know it will fail
    if _cache_is_unsupported():
        print("  ℹ Gemini context cache skipped (unsupported on this tier)")
        print("  → All Gemini calls will pay full token cost")
        _cache_name = None
        return

    style_guide = f"""You are a senior technical writer producing IBM-standard developer \
documentation for the software project "{project_name}".

IBM DOCUMENTATION STANDARD — 4 TIERS:
TIER 1 — LOW-LEVEL: Inline docstrings — purpose, parameters with types, returns, examples.
  Every parameter: name, type hint, required/optional, default value, description.
  Returns: type and what the value represents (not just the type name).
  Raises: exception type and exact condition that triggers it.
  Usage Example: realistic call using actual parameter names, not placeholder strings.

TIER 2 — HIGH-LEVEL: Control flow description, module dependency map, cross-file references.
TIER 3 — EXTERNAL: README-style overview, installation, CLI/API usage guide.
TIER 4 — INTERNAL: Design decisions (why this approach), security findings, naming conventions.

IBM QUALITY RULES — every doc must obey all of these:
- Document the WHY: explain design decisions, not just what the code does.
- Active voice: "Returns the sum" not "The sum is returned by this function".
- No platitudes: "use secure methods" is not a security note. Be code-specific.
- Every security note must reference specific code: line context, variable, or pattern.
- No invented information — base everything only on the provided JSON facts.
- No truncated sentences — every section must be complete.
- Specific over generic: advice that applies to all code is not useful documentation.

REQUIRED SECTIONS — all must be present and non-empty in every function doc:
What it does | Why it exists | Inline Docstring | Parameters | Returns |
Raises | Control Flow | Logic Walkthrough | Edge Cases | Security Notes |
Design Decisions | Naming Insights | Dependencies | Cross-References | Usage Example

DETAILED PARAMETER RULES:
- Parameter names must EXACTLY match the function signature — no paraphrasing.
- Types: use Python type hints (str, int, list[str], dict[str, Any], etc.).
- Optional parameters must show default value in the Default column.
- Description must explain WHAT the parameter controls, not restate its name.

DETAILED SECURITY RULES:
- SQL injection: flag string concatenation or f-strings in any database query.
- Path traversal: flag user input used in file open(), os.path.join(), etc.
- Hardcoded secrets: flag any string literal that looks like a password or key.
- Missing validation: flag user-supplied values used without bounds checking.
- Error disclosure: flag exception handlers that expose system paths or internals.

DETAILED EXAMPLE RULES:
- Use realistic values, not "your_value" or "example_string" placeholders.
- Show the function being called, not being defined.
- Add a comment explaining what the example demonstrates."""

    signatures  = _extract_signatures(files)
    sig_section = f"\nPROJECT FUNCTION INDEX (all signatures in this project):\n{signatures}"
    full_context = style_guide.strip() + "\n" + sig_section

    TARGET_CHARS = 5500

    padding_blocks = [
        """
ADDITIONAL IBM CROSS-REFERENCE RULES:
- Cross-References must list every function called inside this function by name.
- For each cross-reference: state what the called function does and why it is called here.
- If a function is imported from another module, note the source module name.
- If a function calls database operations, flag the exact table name and SQL operation type.
- If a function calls an external API, note the endpoint and the data it sends or receives.""",
        """
ADDITIONAL CONTROL FLOW RULES:
- Describe every conditional branch: what condition triggers it and what it does.
- For loops: describe what is being iterated, what accumulates, and the termination condition.
- For exception handlers: describe what exception is caught and the recovery action taken.
- Do not use generic phrases like "handles the logic" — be specific about each branch outcome.
- For nested conditions, describe the outer condition first then each inner branch separately.""",
        """
ADDITIONAL NAMING ANALYSIS RULES:
- Explain what each non-obvious variable name reveals about its purpose.
- Flag any names that are misleading or inconsistent with the code's actual behaviour.
- Note naming patterns: prefixes like is_, has_, get_, set_ reveal intent.
- Identify abbreviations and explain what they stand for if not immediately obvious.""",
        """
ADDITIONAL DOCSTRING FORMAT RULES:
- Args section: list every parameter with its type hint and a one-line description.
- Returns section: state the type and what the returned value represents semantically.
- Raises section: list every exception the function can raise with the exact trigger condition.
- Example section: show a complete realistic function call with real parameter values.
- The docstring must be complete enough to use the function without reading the source code.""",
        """
ADDITIONAL SECURITY AUDIT RULES:
- SQL injection: check every database query for string concatenation or f-string interpolation.
- Path traversal: check every file open or path join that uses user-supplied data.
- Hardcoded credentials: check for string literals containing passwords, keys, or tokens.
- Input validation: check every function that accepts external input for bounds or type checks.
- Sensitive data in logs: check for print() or log statements that output user data.
- Error disclosure: check exception handlers that expose stack traces or file paths to users.""",
    ]

    for block in padding_blocks:
        if len(full_context) >= TARGET_CHARS:
            break
        full_context += "\n" + block.strip()

    estimated_tokens = len(full_context) // 4
    print(f"  Context size: ~{estimated_tokens} estimated tokens ({len(full_context)} chars)")

    try:
        response = client.caches.create(
            model  = Config.GEMINI_MODEL,
            config = types.CreateCachedContentConfig(
                contents = [full_context],
                ttl      = Config.GEMINI_CACHE_TTL,
            ),
        )
        _cache_name = response.name
        print(f"  ✅ Gemini cache: {_cache_name[:50]}...")
    except Exception as e:
        print(f"  ⚠ Context cache failed: {e}")
        print("  → All Gemini calls will pay full token cost")
        _cache_name = None

        # FIX: persist the failure so future runs skip this immediately
        if "limit=0" in str(e) or "TotalCachedContent" in str(e):
            _mark_cache_unsupported()
            print("  ℹ Marked as unsupported — future runs will skip cache setup")


# ─────────────────────────────────────────────────────────────────────────────
#  Documentation writer
# ─────────────────────────────────────────────────────────────────────────────

def write_documentation(function_name: str, facts: dict) -> str:
    """
    Stage 2: Write full IBM-standard prose documentation from Groq's JSON facts.

    Uses context cache if available (saves ~80% input tokens).
    Falls back to _facts_to_doc_fallback() when Gemini is rate-limited.

    FIX: Once a DailyQuotaExhausted error is seen, all subsequent calls skip
    the API immediately — no retry storms wasting 4+ minutes per function.
    The free tier limit is 20 requests/day; after that, the fallback builder
    (which uses Groq's Stage 1 facts directly) still produces complete docs.
    """
    global _daily_quota_exhausted

    # Cache check — works even after quota is exhausted
    cache_key = f"{function_name}:{json.dumps(facts, sort_keys=True)[:300]}"
    cached    = cache.get(cache_key, "gemini")
    if cached:
        print("      ✓ Cache hit (Gemini)")
        return cached

    # FIX: skip API entirely if daily quota already confirmed exhausted
    if _daily_quota_exhausted:
        print("      ⚡ Gemini daily quota exhausted — using facts fallback")
        return _facts_to_doc_fallback(function_name, facts)

    facts_json = json.dumps(facts, indent=2)
    prompt     = build_gemini_doc_prompt(function_name, facts_json)

    def _call():
        limiter.wait()
        cfg = types.GenerateContentConfig(
            temperature       = 0.1,
            max_output_tokens = Config.GEMINI_MAX_OUT_TOKENS,
        )
        if _cache_name:
            cfg.cached_content = _cache_name
        resp = client.models.generate_content(
            model    = Config.GEMINI_MODEL,
            contents = prompt,
            config   = cfg,
        )
        return resp.text.strip()

    try:
        result = with_backoff(_call)
        cache.set(cache_key, "gemini", result)
        return result

    except DailyQuotaExhausted as e:
        # FIX: set flag so every remaining unit skips the API immediately
        _daily_quota_exhausted = True
        print(
            f"      ✗ Gemini daily quota reached (20 req/day free tier limit).\n"
            f"        Remaining units will use facts fallback — docs still complete.\n"
            f"        Quota resets at midnight Pacific. Run again tomorrow for Gemini prose."
        )
        return _facts_to_doc_fallback(function_name, facts)

    except Exception as e:
        print(f"      ✗ Gemini error: {e} — building from facts")
        return _facts_to_doc_fallback(function_name, facts)


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback: accurate IBM doc built directly from JSON facts
# ─────────────────────────────────────────────────────────────────────────────

def _facts_to_doc_fallback(function_name: str, facts: dict) -> str:
    """
    Build complete IBM-standard documentation without Gemini.
    Grounded entirely in Groq's extracted facts — never invents.
    """
    param_rows = []
    for p in facts.get("params", []):
        if isinstance(p, dict):
            opt     = "No" if not p.get("optional") else "Yes"
            default = p.get("default", "") or "—"
            param_rows.append(
                f"| `{p.get('name','?')}` | `{p.get('type','?')}` "
                f"| {opt} | `{default}` | {p.get('description','')} |"
            )
        elif isinstance(p, str):
            param_rows.append(f"| `{p}` | unknown | ? | — | — |")

    params_md = (
        "| Parameter | Type | Required | Default | Description |\n"
        "|-----------|------|----------|---------|-------------|\n"
        + ("\n".join(param_rows) if param_rows
           else "| — | — | — | — | No parameters. |")
    )

    ret      = facts.get("returns", {})
    ret_type = ret.get("type", "unknown") if isinstance(ret, dict) else str(ret)
    ret_desc = ret.get("description", "")  if isinstance(ret, dict) else ""

    raises_md = "\n".join(
        f"- `{r.get('exception','?')}` — {r.get('condition','')}"
        for r in facts.get("raises", []) if isinstance(r, dict)
    ) or "None."

    steps_md = "\n".join(
        f"{i+1}. {s}"
        for i, s in enumerate(facts.get("logic_steps", []))
        if isinstance(s, str)
    ) or "See source."

    def _bullets(key: str, empty: str = "None documented.") -> str:
        items = [x for x in facts.get(key, []) if isinstance(x, str) and x]
        return "\n".join(f"- {x}" for x in items) if items else empty

    edges    = _bullets("edge_cases",        "None documented.")
    security = _bullets("security_notes",    "None identified.")
    design   = _bullets("design_decisions",  "Standard implementation.")
    refs     = _bullets("cross_references",  "None.")
    deps     = (
        ", ".join(f"`{d}`" for d in facts.get("dependencies", []) if isinstance(d, str))
        or "None."
    )

    ex      = facts.get("example", {})
    ex_code = ex.get("code", f"{function_name}(...)") if isinstance(ex, dict) else f"{function_name}(...)"
    ex_desc = ex.get("description", "")               if isinstance(ex, dict) else ""

    docstring = (
        facts.get("docstring", "")
        or f'"""\n{facts.get("purpose", function_name)}.\n"""'
    )

    return f"""\
---
### `{function_name}`

**What it does:**
{facts.get('purpose', 'See source.')}

**Why it exists:**
{facts.get('intent', 'Core utility function.')}

**Inline Docstring:**
```python
{docstring}
```

**Parameters:**
{params_md}

**Returns:**
`{ret_type}` — {ret_desc}

**Raises:**
{raises_md}

**Control Flow:**
{facts.get('control_flow', 'See logic walkthrough below.')}

**Logic Walkthrough:**
{steps_md}

**Edge Cases:**
{edges}

**Security Notes:**
{security}

**Design Decisions:**
{design}

**Naming Insights:**
{facts.get('naming_analysis', 'Standard naming conventions.')}

**Dependencies:**
{deps}

**Cross-References:**
{refs}

**Usage Example:**
```python
{ex_code}
# {ex_desc}
```
---"""