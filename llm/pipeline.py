"""
pipeline.py — Orchestrates 3-stage IBM documentation pipeline.

UPDATED: Gemini removed from Stage 2.
         GPT-4o (github_client.write_documentation) replaces it.
         Stage 3a: gpt-4o-mini validate+score.
         Stage 3b: Phi-4 security scan.
         Stage 2 + Stage 3 now SEQUENTIAL — parallel was causing GitHub 429s.
         GitHub real RPM limit is ~10/min per model, not 60K (that's token budget).

Stage 1 — Groq:          full code → 14-field JSON facts
Stage 2 — GitHub GPT-4o: JSON facts → IBM-standard prose (sequential, 7s delay)
Stage 3 — GitHub:        validate IBM completeness + score (sequential, 7s delay)
"""

import time
import json
import llm.cache as llm_cache

from core.parser import CodeUnit
from config import Config
from llm.groq_client import (
    extract_facts,
    extract_facts_for_large_function,
    batch_extract_facts,
    analyze_project_dependencies,
)
from llm.github_client import (
    write_documentation,
    _facts_to_doc_fallback,
    validate_and_score,
    security_deep_scan,
)
from llm.compressor import is_trivial, trivial_doc


# ─────────────────────────────────────────────────────────────────────────────
#  Single-unit processing
# ─────────────────────────────────────────────────────────────────────────────

def process_unit(unit: CodeUnit) -> dict:
    """Full 3-stage pipeline for a single code unit."""
    print(f"\n    [{unit.unit_type}] {unit.name}")

    # Pre-filter: trivial functions get template docs at zero API cost
    if is_trivial(unit.code) and Config.TRIVIAL_SKIP_ALWAYS:
        print("      ⚡ Trivial — template doc (no API calls)")
        facts = trivial_doc(unit.name, unit.code, unit.language)
        doc   = _facts_to_doc_fallback(unit.name, facts, unit.language)
        return {
            "name"         : unit.name,
            "type"         : unit.unit_type,
            "facts"        : facts,
            "documentation": doc,
            "final"        : doc,
            "score"        : 3.0,
            "security"     : {},
            "_trivial"     : True,
        }

    # ── Stage 1: Groq — JSON fact extraction ─────────────────────────────
    print("      → Stage 1: Groq extracting facts...")
    if len(unit.code) > Config.LARGE_FUNC_CHARS:
        facts = extract_facts_for_large_function(
            unit.code, unit.language, unit.unit_type, unit.name
        )
    else:
        facts = extract_facts(unit.code, unit.language, unit.unit_type)
    print(f"        ✓ {facts.get('purpose','')[:70]}")

    # Security deep-scan (only if Groq flagged issues)
    security_result: dict = {}
    if facts.get("security_notes") and Config.SECURITY_SCAN_ENABLED:
        print("      → Security scan (Phi-4)...")
        security_result = security_deep_scan(
            unit.name, unit.code, facts["security_notes"]
        )
        if security_result.get("vulnerabilities"):
            extra = [
                v["description"] for v in security_result["vulnerabilities"]
                if isinstance(v, dict) and v.get("description")
            ]
            facts["security_notes"] = list(set(facts["security_notes"] + extra))

    # ── Stage 2: GPT-4o — IBM prose writing ───────────────────────────────
    print("      → Stage 2: GPT-4o writing documentation...")
    documentation = write_documentation(unit.name, facts)

    # ── Stage 3: DeepSeek-R1 — validate + score ───────────────────────────
    print("      → Stage 3: gpt-4o-mini validate+score...")
    sig          = unit.code.splitlines()[0] if unit.code else unit.name
    final, score = validate_and_score(sig, documentation)
    print(f"      → Quality: {score:.1f}/5 ⭐")

    return {
        "name"         : unit.name,
        "type"         : unit.unit_type,
        "facts"        : facts,
        "documentation": documentation,
        "final"        : final or documentation,
        "score"        : score,
        "security"     : security_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-unit processing
# ─────────────────────────────────────────────────────────────────────────────

def process_file(units: list) -> list:
    """
    Process all units in a file.

    Execution plan:
      1. Pre-filter trivial functions → template docs (zero API cost)
      2. Stage 1 (Groq): small units batched, large units block-chunked
      3. Security scan on flagged functions (Phi-4, parallel)
      4. Stage 2 (GPT-4o): can run in parallel — 60K RPM, no bottleneck
      5. Stage 3 (gpt-4o-mini): sequential validate+score
    """
    if not units:
        return []

    print(f"\n    Processing {len(units)} units...")

    # ── Pre-filter ────────────────────────────────────────────────────────
    trivial_units  = [u for u in units if is_trivial(u.code) and Config.TRIVIAL_SKIP_ALWAYS]
    pipeline_units = [u for u in units if not (is_trivial(u.code) and Config.TRIVIAL_SKIP_ALWAYS)]

    if trivial_units:
        print(f"      ⚡ {len(trivial_units)} trivial — template docs, no API calls")

    trivial_results = []
    for u in trivial_units:
        facts = trivial_doc(u.name, u.code, u.language)
        doc   = _facts_to_doc_fallback(u.name, facts, u.language)
        trivial_results.append({
            "unit"         : u,
            "name"         : u.name,
            "type"         : u.unit_type,
            "facts"        : facts,
            "documentation": doc,
            "final"        : doc,
            "score"        : 3.0,
            "security"     : {},
            "_trivial"     : True,
        })

    if not pipeline_units:
        return trivial_results

    # ── Stage 1: Groq fact extraction ────────────────────────────────────
    small = [u for u in pipeline_units
             if u.code.count("\n") < Config.SMALL_LINES
             and len(u.code) <= Config.LARGE_FUNC_CHARS]
    large = [u for u in pipeline_units
             if u.code.count("\n") >= Config.SMALL_LINES
             or len(u.code) > Config.LARGE_FUNC_CHARS]

    print(f"    ── Stage 1 (Groq): {len(small)} small (batched), {len(large)} large (chunked)")
    facts_map: dict = {}

    for i in range(0, len(small), Config.BATCH_SIZE):
        batch = small[i:i + Config.BATCH_SIZE]
        print(f"      Batch {i // Config.BATCH_SIZE + 1}: {[u.name for u in batch]}")
        for unit, facts in zip(batch, batch_extract_facts(batch)):
            facts_map[unit.name] = facts
        # TPM cooldown between batches — prevents Groq 12K TPM overflow.
        # Each batch ≈ 3 functions × ~2,500 tokens = ~7,500 tokens.
        # Waiting 8s between batches keeps us safely under 12K TPM/min.
        if i + Config.BATCH_SIZE < len(small):
            import time as _time
            delay = getattr(Config, 'GROQ_BATCH_DELAY', 8.0)
            print(f"      ⏳ TPM cooldown {delay:.0f}s before next batch...")
            _time.sleep(delay)

    for unit in large:
        print(f"      Large: {unit.name} ({len(unit.code)} chars)")
        facts_map[unit.name] = extract_facts_for_large_function(
            unit.code, unit.language, unit.unit_type, unit.name
        )

    # ── Security scan ─────────────────────────────────────────────────────
    security_map: dict = {}
    if Config.SECURITY_SCAN_ENABLED:
        flagged = [
            u for u in pipeline_units
            if facts_map.get(u.name, {}).get("security_notes")
        ]
        if flagged:
            print(f"    🔒 Security scanning {len(flagged)} flagged function(s)...")
            for u in flagged:
                result = security_deep_scan(
                    u.name, u.code, facts_map[u.name].get("security_notes", [])
                )
                security_map[u.name] = result
                if result.get("vulnerabilities"):
                    extra = [
                        v["description"] for v in result["vulnerabilities"]
                        if isinstance(v, dict) and v.get("description")
                    ]
                    facts_map[u.name]["security_notes"] = list(set(
                        facts_map[u.name].get("security_notes", []) + extra
                    ))

    # ── Stage 2: GPT-4o — IBM prose (SEQUENTIAL — ~10 RPM per model) ────
    # Running in parallel fires all requests simultaneously → instant 429.
    # Sequential with GITHUB_CALL_DELAY keeps us under the per-model RPM.
    # FIX: delay skipped when write_documentation returns a cache hit —
    #      no API call was made so no rate-limit slot was consumed.
    print(f"\n    ── Stage 2 (GPT-4o): {len(pipeline_units)} units sequentially...")
    docs_map: dict = {}
    _stage2_delay = getattr(Config, 'GITHUB_CALL_DELAY', 10.0)

    for ui, unit in enumerate(pipeline_units):
        # Check cache before calling — if hit, skip the inter-call delay
        _cache_key = (
            f"gpt4o_doc:{unit.name}:{unit.language}:"
            + json.dumps(facts_map.get(unit.name, {}), sort_keys=True)[:300]
        )
        import llm.cache as _cache
        _was_cached = _cache.get(_cache_key, "github_doc") is not None

        try:
            docs_map[unit.name] = write_documentation(
                unit.name, facts_map.get(unit.name, {}), unit.language
            )
        except Exception as e:
            print(f"      ✗ GPT-4o {unit.name}: {e} — facts fallback")
            docs_map[unit.name] = _facts_to_doc_fallback(
                unit.name, facts_map.get(unit.name, {}), unit.language
            )
            _was_cached = True   # fallback counts as no real API call

        # Only delay when a real API call was made — skip on cache hits
        if ui < len(pipeline_units) - 1 and not _was_cached:
            import time as _t
            print(f"      ⏳ {_stage2_delay:.0f}s cooldown (GPT-4o RPM)...")
            _t.sleep(_stage2_delay)

    # ── Stage 3: gpt-4o-mini validate+score (SEQUENTIAL — 3s delay) ─────
    # gpt-4o-mini: 2M TPM on Student Pack — much faster than DeepSeek-R1.
    # 3s delay is enough to stay safely under per-model RPM.
    print(f"    ── Stage 3 (gpt-4o-mini): {len(pipeline_units)} units sequentially...")
    _stage3_delay = 3.0   # gpt-4o-mini is reliable — shorter delay than GPT-4o
    validations = []

    for ui, unit in enumerate(pipeline_units):
        sig = unit.code.splitlines()[0] if unit.code else unit.name
        doc_for_unit = docs_map.get(unit.name, "")

        # Check Stage 3 cache before calling — skip delay if cache hit
        _s3_key    = f"ds_vs:{sig[:80]}:{doc_for_unit[:300]}"
        _s3_cached = llm_cache.get(_s3_key, "github_vs") is not None

        try:
            validations.append(validate_and_score(sig, doc_for_unit))
        except Exception as e:
            print(f"      ⚠ Validate error {unit.name}: {e} — score 3.0")
            validations.append((doc_for_unit, 3.0))
            _s3_cached = True   # fallback = no real API call

        # Only delay when a real API call was made
        if ui < len(pipeline_units) - 1 and not _s3_cached:
            import time as _t
            print(f"      ⏳ {_stage3_delay:.0f}s cooldown (gpt-4o-mini RPM)...")
            _t.sleep(_stage3_delay)

    # ── Assemble results ──────────────────────────────────────────────────
    pipeline_results = []
    for unit, validation in zip(pipeline_units, validations):
        if isinstance(validation, Exception):
            final = docs_map.get(unit.name, "No documentation generated.")
            score = 3.0
        else:
            final, score = validation

        print(f"      {unit.name}: {score:.1f}/5 ⭐")
        pipeline_results.append({
            "unit"         : unit,
            "name"         : unit.name,
            "type"         : unit.unit_type,
            "facts"        : facts_map.get(unit.name, {}),
            "documentation": docs_map.get(unit.name, ""),
            "final"        : final or docs_map.get(unit.name, "No documentation generated."),
            "score"        : score,
            "security"     : security_map.get(unit.name, {}),
        })

    # Reconstruct in original parse order
    all_by_name = {r["name"]: r for r in trivial_results + pipeline_results}
    return [all_by_name[u.name] for u in units if u.name in all_by_name]


# ─────────────────────────────────────────────────────────────────────────────
#  IJIRT: project-level dependency resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_all_dependencies(all_results: dict) -> dict:
    """IJIRT: resolve cross-file function references after all files processed."""
    return analyze_project_dependencies(all_results)