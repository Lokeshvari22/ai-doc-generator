"""
github_client.py — Stages 2 and 3 via GitHub Models (Student Developer Pack).

UPDATED:
  Stage 2 (NEW)  — GPT-4o writes IBM-standard prose documentation.
                   Replaces Gemini. 10M TPM / 60K RPM — no daily limit issues.
  Stage 3a       — gpt-4o-mini validates IBM compliance + scores quality. (DeepSeek-R1 removed — rate limits on every call)
                   Replaces gpt-4o-mini (better chain-of-thought reasoning).
  Stage 3b       — Phi-4 security deep-scan (unchanged).

Rate limiter updated to reflect Student Pack actual limits.
"""

import os
import re
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor

from config import Config
from llm.rate_limiter import RateLimiter, with_backoff
from llm.compressor import (
    build_github_doc_prompt,
    build_github_validate_score_prompt,
    build_security_scan_prompt,
)
import llm.cache as cache

URL     = "https://models.github.ai/inference/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {Config.GITHUB_TOKEN}",
    "Content-Type" : "application/json",
}

# Separate rate limiters for Stage 2 (writing) and Stage 3 (validation)
_limiter_write    = RateLimiter(Config.GITHUB_WRITE_RPM, "GitHub/GPT-4o-write",   rpd=Config.GITHUB_WRITE_RPD)
_limiter_validate = RateLimiter(Config.GITHUB_RPM,       "GitHub/gpt4omini-valid", rpd=Config.GITHUB_RPD)
_limiter_security = RateLimiter(Config.GITHUB_RPM,       "GitHub/Phi4-security",  rpd=Config.GITHUB_RPD)

_failed_models: set = set()

_executor = ThreadPoolExecutor(max_workers=4)


# ─────────────────────────────────────────────────────────────────────────────
#  Raw API call
# ─────────────────────────────────────────────────────────────────────────────

def _call_api(model: str, system_msg: str, user_msg: str,
              max_tokens: int = 1000, json_mode: bool = False,
              temperature: float = 0.1) -> str:
    payload: dict = {
        "model"      : model,
        "messages"   : [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens" : max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=60)

    if resp.status_code == 404:
        raise Exception(f"404: model {model} not found")
    if resp.status_code == 429:
        raise Exception(f"429: rate limited on {model}")
    if resp.status_code != 200:
        raise Exception(f"{resp.status_code}: {resp.text[:300]}")

    return resp.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 — IBM prose documentation via GPT-4o
# ─────────────────────────────────────────────────────────────────────────────

_daily_quota_exhausted_write = False


def write_documentation(function_name: str, facts: dict,
                         language: str = "python") -> str:
    """
    Stage 2: GPT-4o writes full IBM-standard prose documentation from Groq facts.
    language parameter ensures correct docstring syntax (Java /** */, Python triple-quote).
    """
    global _daily_quota_exhausted_write

    # Cache check
    cache_key = f"gpt4o_doc:{function_name}:{language}:{json.dumps(facts, sort_keys=True)[:300]}"
    cached = cache.get(cache_key, "github_doc")
    if cached:
        print("      ✓ Cache hit (GPT-4o doc)")
        return cached

    if _daily_quota_exhausted_write:
        print("      ⚡ GitHub write quota exhausted — using facts fallback")
        return _facts_to_doc_fallback(function_name, facts, language)

    facts_json = json.dumps(facts, indent=2)
    user_msg   = build_github_doc_prompt(function_name, facts_json, language)
    system_msg = (
        "You are a senior IBM technical writer. "
        "Write complete, professional documentation. "
        "Never truncate. Complete every section fully. "
        "Never use placeholder text like 'See source' or 'Standard implementation'."
    )

    def _call():
        _limiter_write.wait()
        return _call_api(
            Config.GITHUB_WRITE_MODEL,
            system_msg,
            user_msg,
            max_tokens  = Config.GITHUB_WRITE_MAX_TOKENS,
            json_mode   = False,
            temperature = 0.15,
        )

    try:
        # FIX: retries=2 means max 1 retry — wait 65s once then give up.
        # Old retries=4: 66s+81s+96s = 4 min wasted per unit.
        # New retries=2: 65s wait once = max 70s per unit.
        result = with_backoff(_call, retries=2)
        cache.set(cache_key, "github_doc", result)
        return result
    except Exception as e:
        err = str(e)
        # FIX: 429 after retries = RPM burst, NOT daily quota exhausted.
        # Student Pack has no hard daily cap on GPT-4o.
        # Only fall back for THIS unit — do NOT block remaining units.
        # _daily_quota_exhausted_write stays False so next unit tries normally.
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            print(f"      ⚠ GPT-4o RPM hit (retries exhausted) — facts fallback for {function_name} only")
        else:
            print(f"      ✗ GPT-4o error: {e} — facts fallback for {function_name} only")
        return _facts_to_doc_fallback(function_name, facts, language)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3a — Validate + Score via gpt-4o-mini
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_score(function_signature: str, documentation: str) -> tuple:
    """
    Stage 3a: gpt-4o-mini validates IBM compliance and scores quality (1-5).
    Returns: (documentation, score_float)
    The documentation is returned unchanged — we only score, not rewrite.
    """
    cache_key = f"ds_vs:{function_signature[:80]}:{documentation[:300]}"
    cached    = cache.get(cache_key, "github_vs")
    if cached:
        try:
            data = json.loads(cached)
            print("      ✓ Cache hit (gpt-4o-mini validate+score)")
            return documentation, float(data.get("score", 3.0))
        except Exception:
            pass

    system_msg = (
        "You are an IBM documentation compliance reviewer. "
        "Return valid JSON only. No prose, no markdown."
    )
    user_msg = build_github_validate_score_prompt(function_signature, documentation)

    def _call():
        _limiter_validate.wait()
        return _call_api(
            Config.GITHUB_VALIDATE_MODEL,
            system_msg,
            user_msg,
            max_tokens = Config.GITHUB_MAX_OUT_TOKENS,
            json_mode  = True,
        )

    try:
        raw   = with_backoff(_call)
        data  = json.loads(raw)
        score = max(1.0, min(5.0, float(data.get("score", 3.0))))
        cache.set(cache_key, "github_vs", json.dumps({"score": score}))
        return documentation, score
    except json.JSONDecodeError:
        return documentation, 3.0
    except Exception as e:
        print(f"      ⚠ Validate error: {e} — score defaulting to 3.0")
        return documentation, 3.0


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3b — Security deep-scan via Phi-4
# ─────────────────────────────────────────────────────────────────────────────

def security_deep_scan(function_name: str, code: str,
                        known_issues: list) -> dict:
    """
    Deep security analysis using Phi-4.
    Only called when Groq Stage 1 flagged security concerns.
    """
    if not Config.SECURITY_SCAN_ENABLED:
        return {}

    cache_key = f"sec:{function_name}:{code[:200]}"
    cached    = cache.get(cache_key, "github_sec")
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    user_msg = build_security_scan_prompt(function_name, code, known_issues)

    def _call():
        _limiter_security.wait()
        return _call_api(
            Config.GITHUB_SECURITY_MODEL,
            "You are a security code auditor. Return valid JSON only.",
            user_msg,
            max_tokens = 500,
            json_mode  = True,
        )

    try:
        raw    = with_backoff(_call)
        result = json.loads(raw)
        cache.set(cache_key, "github_sec", raw)
        return result
    except Exception as e:
        print(f"      ⚠ Security scan failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Backward-compat wrappers (pipeline.py calls these names)
# ─────────────────────────────────────────────────────────────────────────────

def validate(function_signature: str, documentation: str) -> str:
    doc, _ = validate_and_score(function_signature, documentation)
    return doc


def evaluate(function_signature: str, doc: str) -> float:
    _, score = validate_and_score(function_signature, doc)
    return score


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback doc builder (used when API quota hit)
# ─────────────────────────────────────────────────────────────────────────────

def _facts_to_doc_fallback(function_name: str, facts: dict,
                            language: str = "python") -> str:
    """
    Build complete IBM doc from facts JSON without any API call.

    FIXED: Returns field no longer shows bare backtick when type is empty.
    FIXED: Docstring rendered as clean code block — no triple-quote artifacts.
    FIXED: Cross-references stripped of dangling backticks.
    FIXED: Language-aware docstring style (Java uses /** */, Python uses triple-quote).
    FIXED: Logic steps restart numbering from 1 (plain text, not List Number style).
    """
    # Language from facts if not passed directly
    language = facts.get("language", language) or "python"

    param_rows = []
    for p in facts.get("params", []):
        if isinstance(p, dict):
            opt     = "No" if not p.get("optional") else "Yes"
            default = p.get("default", "") or "—"
            param_rows.append(
                f"| {p.get('name','?')} | {p.get('type','?')} "
                f"| {opt} | {default} | {p.get('description','')} |"
            )
        elif isinstance(p, str):
            param_rows.append(f"| {p} | unknown | ? | — | — |")

    params_md = (
        "| Parameter | Type | Required | Default | Description |\n"
        "|-----------|------|----------|---------|-------------|\n"
        + ("\n".join(param_rows) if param_rows
           else "| — | — | — | — | No parameters. |")
    )

    ret      = facts.get("returns", {})
    ret_type = ret.get("type", "") if isinstance(ret, dict) else str(ret)
    ret_desc = ret.get("description", "") if isinstance(ret, dict) else ""

    # FIX 3: Returns line — only show type backtick if type is real
    if ret_type and ret_type not in ("unknown", ""):
        returns_line = f"`{ret_type}` — {ret_desc}" if ret_desc else f"`{ret_type}`"
    else:
        returns_line = ret_desc if ret_desc else "See source code."

    raises_md = "\n".join(
        f"- {r.get('exception','?')} — {r.get('condition','')}"
        for r in facts.get("raises", []) if isinstance(r, dict)
    ) or "None."

    # FIX 5: Logic steps as plain numbered text — NOT List Number style
    # (List Number style auto-increments across the whole document)
    logic_steps_raw = facts.get("logic_steps", [])
    if logic_steps_raw and isinstance(logic_steps_raw, list):
        steps_lines = []
        for i, s in enumerate(logic_steps_raw):
            if isinstance(s, str) and s.strip():
                steps_lines.append(f"{i+1}. {s.strip()}")
        steps_md = "\n".join(steps_lines) if steps_lines else "See source code."
    else:
        steps_md = "See source code."

    def _bullets(key: str, empty: str = "None documented.") -> str:
        items = [x for x in facts.get(key, []) if isinstance(x, str) and x.strip()]
        return "\n".join(f"- {x}" for x in items) if items else empty

    edges    = _bullets("edge_cases",       "None documented.")
    security = _bullets("security_notes",   "None identified.")
    design   = _bullets("design_decisions", "Standard implementation.")

    # FIX 4: Strip dangling backticks from cross-references
    raw_refs = facts.get("cross_references", [])
    clean_refs = []
    for ref in raw_refs:
        if isinstance(ref, str) and ref.strip():
            # Remove leading/trailing backticks and class/function prefixes
            cleaned = ref.strip().lstrip("`").rstrip("`").strip()
            cleaned = re.sub(r'^(class|function)\s*`?\s*', '', cleaned)
            if cleaned:
                clean_refs.append(f"- `{cleaned}`")
    refs = "\n".join(clean_refs) if clean_refs else "None."

    deps = (
        ", ".join(f"`{d}`" for d in facts.get("dependencies", [])
                  if isinstance(d, str) and d.strip())
        or "None."
    )

    ex      = facts.get("example", {})
    ex_code = ex.get("code", f"{function_name}(...)") if isinstance(ex, dict) else f"{function_name}(...)"
    ex_desc = ex.get("description", "") if isinstance(ex, dict) else ""

    # FIX 2: Build docstring cleanly — no triple-quotes as raw text
    # Use docstring_hint (clean plain string) as content
    hint = (facts.get("docstring_hint", "")
            or facts.get("purpose", f"Implements {function_name} logic."))
    # Strip any accidental triple-quotes from hint
    hint = hint.replace('"""', '').replace("'''", "").strip()

    # Build language-appropriate docstring
    param_doc_lines = "\n".join(
        f"    @param {p.get('name','?')} {p.get('type','?')} - {p.get('description','')}"
        if language == "java"
        else f"    {p.get('name','?')} ({p.get('type','?')}): {p.get('description','')}"
        for p in facts.get("params", []) if isinstance(p, dict)
    )

    if language == "java":
        docstring_content = (
            f"/**\n"
            f" * {hint}\n"
            + (f" *\n{chr(10).join(' * ' + l for l in param_doc_lines.splitlines())}\n" if param_doc_lines else "")
            + (f" * @return {returns_line}\n" if returns_line != "See source code." else "")
            + f" */"
        )
    elif language in ("javascript", "typescript"):
        docstring_content = (
            f"/**\n"
            f" * {hint}\n"
            + (f" *\n{chr(10).join(' * ' + l for l in param_doc_lines.splitlines())}\n" if param_doc_lines else "")
            + (f" * @returns {returns_line}\n" if returns_line != "See source code." else "")
            + f" */"
        )
    else:
        # Python triple-quote — but the fence markers go in the markdown, content is clean
        param_section = f"\nArgs:\n{param_doc_lines}\n" if param_doc_lines else ""
        docstring_content = (
            f"{hint}"
            f"{param_section}"
            f"\nReturns:\n    {returns_line}"
        )

    # Choose the fence language tag
    fence_lang = {
        "java": "java", "javascript": "javascript",
        "typescript": "typescript", "cpp": "cpp", "c": "c",
    }.get(language, "python")

    return f"""\
---
### `{function_name}`

**What it does:**
{facts.get('purpose', 'See source.')}

**Why it exists:**
{facts.get('intent', '') or 'Core utility function providing essential support to the application.'}

**Inline Docstring:**
```{fence_lang}
{docstring_content}
```

**Parameters:**
{params_md}

**Returns:**
{returns_line}

**Raises:**
{raises_md}

**Control Flow:**
{facts.get('control_flow', 'Linear execution.')}

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
```{fence_lang}
{ex_code}
// {ex_desc}
```
---"""


# ─────────────────────────────────────────────────────────────────────────────
#  Usage report
# ─────────────────────────────────────────────────────────────────────────────

def print_usage():
    print("\n── GitHub Models Usage ───────────────────────────────")
    print(f"  Stage 2 (GPT-4o write)   : {_limiter_write.day_count} calls")
    print(f"  Stage 3a (gpt-4o-mini valid): {_limiter_validate.day_count} calls")
    print(f"  Stage 3b (Phi-4 security): {_limiter_security.day_count} calls")
    total = _limiter_write.day_count + _limiter_validate.day_count + _limiter_security.day_count
    print(f"  Total GitHub calls       : {total}")
    print("─────────────────────────────────────────────────────\n")