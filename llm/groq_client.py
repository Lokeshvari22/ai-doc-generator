"""
groq_client.py — Stage 1: Fast JSON fact extraction via Groq llama-3.3-70b.

UPDATED:
  - sanitise_groq_json() applied before every json.loads() call.
    Fixes: json_validate_failed when Groq outputs triple-quoted docstrings.
  - _safe_parse() wraps all JSON parsing with sanitisation + fallback.
  - batch_extract_facts() retries with individual calls if batch JSON is invalid.
  - GROQ_BATCH_DELAY respected between batches (read from Config).
  - Groq no longer asked for 'docstring' field — uses 'docstring_hint' instead.
"""

import os
import re
import json
import time
import random
from groq import Groq

from config import Config
from llm.rate_limiter import RateLimiter, with_backoff
from llm.compressor import (
    build_groq_extraction_prompt,
    build_batch_groq_prompt,
    prepare_code_for_groq,
    sanitise_groq_json,
    split_into_blocks,
    code_hash,
)
import llm.cache as cache

client  = Groq(api_key=Config.GROQ_API_KEY)
limiter = RateLimiter(rpm=Config.GROQ_RPM, name="Groq", rpd=Config.GROQ_RPD)

os.makedirs(Config.CACHE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  JSON parsing with sanitisation
# ─────────────────────────────────────────────────────────────────────────────

def _safe_parse(raw: str, context: str = "") -> dict | None:
    """
    Parse Groq JSON response safely.
    Applies sanitise_groq_json() before json.loads() to handle
    triple-quoted docstrings and JavaDoc blocks that break strict JSON.
    Returns None on unrecoverable parse failure.
    """
    if not raw or not raw.strip():
        return None

    # Try raw first (fastest path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Sanitise then retry
    try:
        cleaned = sanitise_groq_json(raw)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: extract outermost { ... } block
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        cleaned = sanitise_groq_json(raw[start:end])
        return json.loads(cleaned)
    except (ValueError, json.JSONDecodeError) as e:
        if context:
            print(f"      ⚠ JSON parse failed for {context}: {e}")
        return None


def _safe_parse_array(raw: str) -> list | None:
    """Parse a JSON array response from Groq, with sanitisation."""
    if not raw or not raw.strip():
        return None

    # Try raw first
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Sanitise and retry
    try:
        cleaned = sanitise_groq_json(raw)
        result  = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Extract [...] block
    try:
        start = raw.index("[")
        end   = raw.rindex("]") + 1
        cleaned = sanitise_groq_json(raw[start:end])
        result  = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except (ValueError, json.JSONDecodeError):
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Empty/fallback facts dict
# ─────────────────────────────────────────────────────────────────────────────

def _empty_facts(name: str = "unknown") -> dict:
    return {
        "purpose"         : f"Implements {name} functionality.",
        "intent"          : "",
        "params"          : [],
        "returns"         : {"type": "unknown", "description": ""},
        "raises"          : [],
        "logic_steps"     : ["See source code."],
        "control_flow"    : "See source code.",
        "edge_cases"      : [],
        "security_notes"  : [],
        "design_decisions": [],
        "naming_analysis" : "",
        "dependencies"    : [],
        "cross_references": [],
        "example"         : {"code": f"{name}(...)", "description": ""},
        "docstring_hint"  : f"Implements {name} functionality.",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Core API call
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str, max_tokens: int = None) -> str:
    """Single Groq API call with rate limiting."""
    max_tokens = max_tokens or Config.GROQ_MAX_OUT_TOKENS

    def _call():
        limiter.wait()
        response = client.chat.completions.create(
            model       = Config.GROQ_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = max_tokens,
            response_format = {"type": "json_object"},
        )
        return response.choices[0].message.content.strip()

    return with_backoff(_call)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — Single function extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_facts(code: str, language: str, unit_type: str) -> dict:
    """Extract 14-field JSON facts from a single code unit via Groq."""
    h = code_hash(code)
    if Config.INCREMENTAL_MODE:
        cached = cache.get(h, "groq_facts")
        if cached:
            data = _safe_parse(cached, "cache")
            if data:
                print("      ✓ Incremental hit (code unchanged)")
                return data

    prepared = prepare_code_for_groq(code, language, Config.MAX_FUNC_CHARS)
    prompt   = build_groq_extraction_prompt(prepared, language, unit_type)

    try:
        raw  = _call_groq(prompt)
        data = _safe_parse(raw, f"{language} {unit_type}")
        if data:
            cache.set(h, "groq_facts", json.dumps(data))
            return data
        print("      ⚠ Groq JSON unparseable — using empty facts")
        return _empty_facts()
    except Exception as e:
        print(f"      ✗ Groq error: {e} — using empty facts")
        return _empty_facts()


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — Batch extraction (small functions)
# ─────────────────────────────────────────────────────────────────────────────

def batch_extract_facts(units: list) -> list:
    """
    Extract facts for multiple small functions in one Groq call.
    Returns list of facts dicts in the same order as units.

    If batch JSON parse fails, falls back to individual calls per function
    so a single bad function doesn't poison the whole batch.
    """
    if not units:
        return []

    if len(units) == 1:
        facts = extract_facts(units[0].code, units[0].language, units[0].unit_type)
        return [facts]

    # Check incremental cache for all units first
    results    = [None] * len(units)
    need_api   = []
    for i, unit in enumerate(units):
        h = code_hash(unit.code)
        if Config.INCREMENTAL_MODE:
            cached = cache.get(h, "groq_facts")
            if cached:
                data = _safe_parse(cached, unit.name)
                if data:
                    results[i] = data
                    print(f"      ✓ Incremental hit (code unchanged)")
                    continue
        need_api.append((i, unit))

    if not need_api:
        return results

    # Build batch prompt for uncached units
    api_units = [u for _, u in need_api]
    prompt    = build_batch_groq_prompt(api_units)

    try:
        def _call():
            limiter.wait()
            # Batch calls use text mode — JSON array response
            response = client.chat.completions.create(
                model       = Config.GROQ_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.0,
                max_tokens  = Config.GROQ_MAX_OUT_TOKENS * len(api_units),
            )
            return response.choices[0].message.content.strip()

        raw    = with_backoff(_call)
        parsed = _safe_parse_array(raw)

        if parsed and len(parsed) == len(api_units):
            for (orig_i, unit), facts_data in zip(need_api, parsed):
                if isinstance(facts_data, dict):
                    h = code_hash(unit.code)
                    cache.set(h, "groq_facts", json.dumps(facts_data))
                    results[orig_i] = facts_data
                else:
                    results[orig_i] = _empty_facts(unit.name)
            return results

        # Batch JSON invalid — fall back to individual calls
        print(f"      ⚠ Batch JSON invalid — falling back to individual calls")
        for orig_i, unit in need_api:
            results[orig_i] = extract_facts(unit.code, unit.language, unit.unit_type)
            time.sleep(1.5)   # small delay between individual fallback calls
        return results

    except Exception as e:
        print(f"      ✗ Groq batch error: {e} — individual fallback")
        for orig_i, unit in need_api:
            try:
                results[orig_i] = extract_facts(unit.code, unit.language, unit.unit_type)
                time.sleep(1.5)
            except Exception as e2:
                print(f"        ✗ Individual fallback failed for {unit.name}: {e2}")
                results[orig_i] = _empty_facts(unit.name)
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — Large function (block-chunked)
# ─────────────────────────────────────────────────────────────────────────────

def extract_facts_for_large_function(code: str, language: str,
                                      unit_type: str, name: str) -> dict:
    """
    Extract facts from a large function by splitting into logical blocks.
    Results accumulated across blocks — no context lost.
    """
    h = code_hash(code)
    if Config.INCREMENTAL_MODE:
        cached = cache.get(h, "groq_facts")
        if cached:
            data = _safe_parse(cached, name)
            if data:
                print("      ✓ Incremental hit (code unchanged)")
                return data

    print(f"      → Block-chunking {name} ({len(code)} chars)...")
    blocks = split_into_blocks(code, language)
    print(f"      → {len(blocks)} blocks")

    accumulated: dict = _empty_facts(name)
    context_so_far    = ""

    for bi, block in enumerate(blocks):
        print(f"        Block {bi+1}/{len(blocks)}: [{block['label']}]")
        prepared = prepare_code_for_groq(block["code"], language, 3000)
        prompt   = build_groq_extraction_prompt(
            prepared, language, unit_type,
            block_context=context_so_far
        )

        try:
            raw  = _call_groq(prompt)
            data = _safe_parse(raw, f"{name} block {bi+1}")
            if data:
                _merge_facts(accumulated, data)
                context_so_far = (
                    f"Block {bi+1} purpose: {data.get('purpose','')}\n"
                    f"Steps so far: {'; '.join(data.get('logic_steps',[])[:3])}\n"
                )
        except Exception as e:
            print(f"        ✗ Block {bi+1} failed: {e}")

        if bi < len(blocks) - 1:
            time.sleep(2.0)   # brief pause between block calls

    cache.set(h, "groq_facts", json.dumps(accumulated))
    return accumulated


def _merge_facts(base: dict, update: dict):
    """Merge facts from a block update into the accumulated base dict."""
    # String fields: take non-empty update value
    for key in ("purpose", "intent", "control_flow", "naming_analysis"):
        if update.get(key) and not base.get(key):
            base[key] = update[key]

    # List fields: extend without duplicates
    for key in ("params", "raises", "logic_steps", "edge_cases",
                "security_notes", "design_decisions",
                "dependencies", "cross_references"):
        existing = base.get(key, [])
        new_items = update.get(key, [])
        if isinstance(existing, list) and isinstance(new_items, list):
            seen = {json.dumps(x, sort_keys=True) for x in existing}
            for item in new_items:
                k = json.dumps(item, sort_keys=True)
                if k not in seen:
                    existing.append(item)
                    seen.add(k)
            base[key] = existing

    # Returns: take update if base is empty
    if update.get("returns") and not base.get("returns", {}).get("description"):
        base["returns"] = update["returns"]

    # Example: take update if base has placeholder
    ex = update.get("example", {})
    if ex and ex.get("code") and "..." not in ex.get("code", ""):
        base["example"] = ex

    # docstring_hint: take first non-empty
    if update.get("docstring_hint") and not base.get("docstring_hint"):
        base["docstring_hint"] = update["docstring_hint"]


# ─────────────────────────────────────────────────────────────────────────────
#  IJIRT: cross-file dependency resolution
# ─────────────────────────────────────────────────────────────────────────────

def analyze_project_dependencies(all_results: dict) -> dict:
    """
    IJIRT: resolve cross-file function references.
    Adds 'resolved_dependencies' to each unit result in-place.
    """
    # Build global name → file map
    name_map: dict = {}
    for filename, results in all_results.items():
        for r in results:
            name_map[r["name"]] = filename

    # Resolve cross-references
    for filename, results in all_results.items():
        for r in results:
            facts = r.get("facts", {})
            refs  = facts.get("cross_references", [])
            resolved = []
            for ref in refs:
                ref_name = ref.split("(")[0].strip().strip("`") if ref else ""
                target   = name_map.get(ref_name)
                if target and target != filename:
                    resolved.append(f"{ref} → {target}")
                else:
                    resolved.append(ref)
            r["resolved_dependencies"] = resolved

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  General text generation (README, summary — non-JSON)
# ─────────────────────────────────────────────────────────────────────────────

def generate(prompt: str, output_type: str = "text",
             cache_key_hint: str = "") -> str:
    """General-purpose Groq text generation (not JSON mode)."""
    h = code_hash(prompt[:500]) if cache_key_hint else ""
    if h and Config.INCREMENTAL_MODE:
        cached = cache.get(h, f"groq_{cache_key_hint}")
        if cached:
            return cached

    def _call():
        limiter.wait()
        response = client.chat.completions.create(
            model       = Config.GROQ_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.3,
            max_tokens  = 1200,
        )
        return response.choices[0].message.content.strip()

    try:
        result = with_backoff(_call)
        if h:
            cache.set(h, f"groq_{cache_key_hint}", result)
        return result
    except Exception as e:
        print(f"  ✗ Groq generate error: {e}")
        return f"[Generation failed: {e}]"