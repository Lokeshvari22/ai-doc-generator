"""
app.py — FastAPI backend for AI Code Documentation Generator Web App.

UPDATED:
  - Pipeline label: Groq → GPT-4o → gpt-4o-mini + Phi-4
  - Gemini context cache setup removed
  - generate_architecture() tuple return handled (img, analyses)
  - add_architecture() receives analyses for text fallback (P5 fix)
  - LIMITS dict updated to reflect new GitHub Student Pack quotas
  - Gemini quota tracking removed from log_quota calls
"""

import os
import sys
import re
import uuid
import json
import shutil
import asyncio
import tempfile
import threading
import traceback
from pathlib import Path
from datetime import datetime, date
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

THIS_FILE    = Path(__file__).resolve()
BACKEND_DIR  = THIS_FILE.parent
WEBAPP_DIR   = BACKEND_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

UPLOAD_DIR  = BACKEND_DIR / "uploads"
OUTPUT_DIR  = BACKEND_DIR / "outputs"
EXTRACT_DIR = BACKEND_DIR / "extracted"
CACHE_DIR   = PROJECT_ROOT / "cache"

for d in (UPLOAD_DIR, OUTPUT_DIR, EXTRACT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

from database       import init_db, create_job, update_job, get_job, list_jobs, log_quota, get_today_quota
from dummy_pipeline import process_file_dummy


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()
DUMMY_MODE = os.environ.get("DUMMY_MODE", "false").lower() in ("true", "1", "yes")


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy import cache — pipeline modules loaded once on first job, not on startup
#  This keeps server startup fast (~1s) and avoids re-importing heavy libs.
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_loaded = False

def _ensure_pipeline():
    """Import pipeline modules once and cache them as module-level names."""
    global _pipeline_loaded
    if _pipeline_loaded:
        return
    global process_file, parse_file, parse_file_lang, detect_language, analyze
    global create_doc, add_file_heading, add_unit_doc, add_complexity_table
    global add_architecture, add_summary, save_doc, generate_architecture
    global resolve_all_dependencies

    from llm.pipeline      import process_file, resolve_all_dependencies
    from core.parser       import parse_file, detect_language
    from core.complexity   import analyze
    from output.doc_builder import (
        create_doc, add_file_heading, add_unit_doc,
        add_complexity_table, add_architecture, add_summary, save_doc,
    )
    from output.architecture import generate_architecture
    _pipeline_loaded = True


# ─────────────────────────────────────────────────────────────────────────────
#  Token budget constants for UI display — updated for Student Pack
# ─────────────────────────────────────────────────────────────────────────────

LIMITS = {
    "snippet"      : {"max_lines": 150,  "max_kb": None},
    "file"         : {"max_lines": None, "max_kb": 100},
    "zip"          : {"max_mb": 5,       "rec_files": 5, "rec_mb": 2},
    "groq_tpm"     : 12_000,        # Groq: tokens per minute
    "groq_rpd"     : 14_400,        # Groq: requests per day
    "github_rpm"   : 60_000,        # GitHub GPT-4o Student Pack
    "github_rpd"   : 999_999,       # GitHub: no hard daily cap
    "tokens_per_fn": 2_500,         # GPT-4o output budget per function
}


# ─────────────────────────────────────────────────────────────────────────────
#  SSE queue registry
# ─────────────────────────────────────────────────────────────────────────────

_sse_queues: dict = {}
_sse_lock   = threading.Lock()


def _get_queue(job_id: str) -> asyncio.Queue:
    with _sse_lock:
        if job_id not in _sse_queues:
            _sse_queues[job_id] = asyncio.Queue(maxsize=1000)
        return _sse_queues[job_id]


def _push(job_id: str, obj: dict):
    q = _get_queue(job_id)
    try:
        q.put_nowait(json.dumps(obj))
    except asyncio.QueueFull:
        pass


def _push_log(job_id: str, msg: str, level: str = "info"):
    _push(job_id, {
        "type" : "log",
        "level": level,
        "msg"  : msg,
        "ts"   : datetime.utcnow().isoformat(),
    })


def _push_done(job_id: str, payload: dict):
    _push(job_id, {"type": "done", **payload})
    _push(job_id, {"type": "__close__"})


def _push_error(job_id: str, msg: str):
    _push(job_id, {"type": "error", "msg": msg})
    _push(job_id, {"type": "__close__"})


class _PrintCapture:
    """Redirect stdout to SSE queue during a job run."""
    def __init__(self, job_id: str, orig):
        self.job_id = job_id
        self.orig   = orig

    def write(self, text: str):
        self.orig.write(text)
        msg = text.strip()
        if not msg:
            return
        level = (
            "error"   if any(x in msg for x in ("❌", "✗", "Error", "error")) else
            "success" if any(x in msg for x in ("✅", "Done", "done", "Saved")) else
            "warn"    if any(x in msg for x in ("⚠", "warn", "Warn"))          else
            "info"
        )
        _push_log(self.job_id, msg, level)

    def flush(self):
        self.orig.flush()


def _results_to_text(results: list) -> str:
    lines = []
    for r in results:
        lines.append("=" * 60)
        lines.append(f"  {r['type'].upper()}: {r['name']}   "
                     f"[Score: {r['score']:.1f}/5]"
                     + ("  [DUMMY MODE]" if r.get("_dummy") else ""))
        lines.append("=" * 60)
        lines.append(r.get("final", "No documentation generated."))
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Job runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_snippet(job_id: str, code: str, language: str):
    orig = sys.stdout
    sys.stdout = _PrintCapture(job_id, orig)
    try:
        update_job(job_id, status="running")
        mode_label = "[DUMMY MODE] " if DUMMY_MODE else ""
        _push_log(job_id, f"🚀 {mode_label}Starting snippet documentation ({language})...", "info")

        suffix = {"python": ".py", "javascript": ".js", "typescript": ".ts",
                  "java": ".java", "cpp": ".cpp", "c": ".c"}.get(language, ".py")
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, mode="w", encoding="utf-8", delete=False
        )
        tmp.write(code)
        tmp.flush()
        tmp.close()
        tmp_path = Path(tmp.name)

        from core.parser import CodeUnit
        _ensure_pipeline()
        _push_log(job_id, "Parsing code units...", "info")
        units = parse_file(tmp_path)

        if not units:
            _push_error(job_id, "No parseable code units found in snippet.")
            return

        # FIX: If parser fell back to chunk (section_1 etc.), try to extract a
        # real name directly from the snippet so the doc title is meaningful.
        import re as _re
        _name_patterns = {
            "c"         : _re.compile(r'\b(\w+)\s*\([^)]*\)\s*\{', _re.MULTILINE),
            "cpp"       : _re.compile(r'\b(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{', _re.MULTILINE),
            "java"      : _re.compile(r'(?:public|private|protected|static)[^(]+\s(\w+)\s*\(', _re.MULTILINE),
            "javascript": _re.compile(r'function\s+(\w+)\s*\(|(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(', _re.MULTILINE),
            "typescript": _re.compile(r'function\s+(\w+)\s*\(|(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(', _re.MULTILINE),
        }
        for unit in units:
            if unit.name.startswith("section_"):
                pat = _name_patterns.get(language)
                if pat:
                    m = pat.search(code)
                    if m:
                        real_name = next((g for g in m.groups() if g), None)
                        if real_name and real_name not in ("if","for","while","return","void","int","char","bool"):
                            unit.name = real_name
                            _push_log(job_id, f"  ↳ Resolved name: {real_name}", "info")

        _push_log(job_id, f"Found {len(units)} unit(s). Running pipeline...", "info")

        if DUMMY_MODE:
            results = process_file_dummy(units)
        else:
            results = process_file(units)

        output_text = _results_to_text(results)

        # Build a .docx for the snippet too
        doc = create_doc("Snippet")
        for r in results:
            add_unit_doc(doc, r["name"], r["type"], r["final"], r["score"])

        out_path = OUTPUT_DIR / f"{job_id}.docx"
        save_doc(doc, str(out_path))

        avg_score  = sum(r["score"] for r in results) / len(results) if results else 0
        total_funcs = len(results)

        update_job(job_id,
                   status      = "done",
                   func_count  = total_funcs,
                   score_avg   = round(avg_score, 2),
                   output_path = str(out_path),
                   is_dummy    = DUMMY_MODE,
                   finished_at = datetime.utcnow().isoformat())

        if not DUMMY_MODE:
            log_quota(groq_tokens=total_funcs * 700, github_calls=total_funcs * 2)

        _push_log(job_id,
                  f"✅ Done! {total_funcs} unit(s) documented. "
                  f"Avg score: {avg_score:.1f}/5",
                  "success")

        _push_done(job_id, {
            "text"        : output_text,
            "download_url": f"/api/download/{job_id}",
            "filename"    : "snippet_docs.docx",
            "score_avg"   : round(avg_score, 2),
            "func_count"  : total_funcs,
            "is_dummy"    : DUMMY_MODE,
            "job_id"      : job_id,
        })

    except Exception as e:
        _push_log(job_id, f"❌ {e}", "error")
        update_job(job_id, status="error", error_msg=str(e),
                   finished_at=datetime.utcnow().isoformat())
        _push_error(job_id, str(e))
    finally:
        sys.stdout = orig
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _run_file(job_id: str, file_path: Path, original_name: str):
    orig = sys.stdout
    sys.stdout = _PrintCapture(job_id, orig)
    try:
        update_job(job_id, status="running")
        mode_label = "[DUMMY MODE] " if DUMMY_MODE else ""
        _push_log(job_id, f"🚀 {mode_label}Starting file documentation: {original_name}", "info")

        _ensure_pipeline()
        lang  = detect_language(file_path)
        units = parse_file(file_path)
        _push_log(job_id, f"Found {len(units)} unit(s) in {original_name} ({lang})", "info")

        if not units:
            _push_error(job_id, "No parseable code units found.")
            return

        if DUMMY_MODE:
            results = process_file_dummy(units)
        else:
            results = process_file(units)

        doc = create_doc(Path(original_name).stem)
        add_file_heading(doc, original_name, lang)

        for r in results:
            add_unit_doc(doc, r["name"], r["type"], r["final"], r["score"])

        if lang == "python":
            cx_list = analyze(file_path)
            if cx_list:
                add_complexity_table(doc, cx_list)

        out_path = OUTPUT_DIR / f"{job_id}.docx"
        save_doc(doc, str(out_path))

        avg_score   = sum(r["score"] for r in results) / len(results) if results else 0
        total_funcs = len(results)

        update_job(job_id,
                   status      = "done",
                   func_count  = total_funcs,
                   score_avg   = round(avg_score, 2),
                   output_path = str(out_path),
                   is_dummy    = DUMMY_MODE,
                   finished_at = datetime.utcnow().isoformat())

        if not DUMMY_MODE:
            log_quota(groq_tokens=total_funcs * 700, github_calls=total_funcs * 2)

        _push_log(job_id,
                  f"✅ Done! {total_funcs} functions documented. "
                  f"Avg score: {avg_score:.1f}/5",
                  "success")

        _push_done(job_id, {
            "download_url": f"/api/download/{job_id}",
            "filename"    : f"{Path(original_name).stem}_docs.docx",
            "score_avg"   : round(avg_score, 2),
            "func_count"  : total_funcs,
            "is_dummy"    : DUMMY_MODE,
            "job_id"      : job_id,
        })

    except Exception as e:
        _push_log(job_id, f"❌ {e}", "error")
        update_job(job_id, status="error", error_msg=str(e),
                   finished_at=datetime.utcnow().isoformat())
        _push_error(job_id, str(e))
    finally:
        sys.stdout = orig
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


def _run_zip(job_id: str, zip_path: Path, project_name: str):
    orig = sys.stdout
    sys.stdout = _PrintCapture(job_id, orig)
    extract_dir = EXTRACT_DIR / job_id

    try:
        update_job(job_id, status="running")
        mode_label = "[DUMMY MODE] " if DUMMY_MODE else ""
        _push_log(job_id, f"🚀 {mode_label}Starting ZIP documentation: {project_name}", "info")

        import zipfile
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        _push_log(job_id, f"✅ ZIP extracted", "success")

        from core.file_scanner import scan_files
        _ensure_pipeline()

        files = scan_files(str(extract_dir))
        _push_log(job_id, f"Found {len(files)} source file(s)", "info")

        if not files:
            _push_error(job_id, "No supported source files found in ZIP.")
            return

        doc         = create_doc(project_name)
        all_results : dict = {}
        all_summaries: list = []
        total_funcs  = 0

        for i, file_path in enumerate(files, 1):
            lang  = detect_language(file_path)
            _push_log(job_id, f"[{i}/{len(files)}] Processing {file_path.name} ({lang})...", "info")

            units = parse_file(file_path)
            if not units:
                _push_log(job_id, f"  Skipping {file_path.name} — no parseable units", "warn")
                continue

            add_file_heading(doc, file_path.name, lang)

            if DUMMY_MODE:
                results = process_file_dummy(units)
            else:
                results = process_file(units)

            all_results[file_path.name] = results
            total_funcs += len(results)

            for r in results:
                add_unit_doc(doc, r["name"], r["type"], r["final"], r["score"])

            if lang == "python":
                cx_list = analyze(file_path)
                if cx_list:
                    add_complexity_table(doc, cx_list)

            lines = []
            for r in results:
                facts  = r.get("facts", {})
                p      = facts.get("purpose", "")
                mark   = "⚡" if r.get("_trivial") else "◆"
                lines.append(f"{mark} **{r['name']}:** {p}")
            all_summaries.append(f"## {file_path.name}\n" + "\n".join(lines))

        # P5 FIX: architecture returns (img, analyses) tuple
        _push_log(job_id, "Generating architecture diagram...", "info")
        try:
            arch_out = str(OUTPUT_DIR / f"{job_id}_arch.png")
            img, file_analyses = generate_architecture(files, arch_out)
            add_architecture(doc, img, file_analyses)
        except Exception as e:
            _push_log(job_id, f"⚠ Architecture skipped: {e}", "warn")

        # Dependency resolution
        if not DUMMY_MODE:
            resolve_all_dependencies(all_results)

        # Summary
        if all_summaries and not DUMMY_MODE:
            from llm.groq_client import generate as _generate
            summary_prompt = (
                f"Write a professional IBM project summary for '{project_name}' "
                f"using markdown (### headings, **bold**, - bullets).\n\n"
                + "\n".join(all_summaries[:5])
            )
            summary = _generate(summary_prompt, "text", "summary")
            add_summary(doc, summary)

        out_path = OUTPUT_DIR / f"{job_id}.docx"
        save_doc(doc, str(out_path))

        avg_score = (
            sum(r["score"] for res in all_results.values() for r in res) / total_funcs
            if total_funcs else 0
        )

        update_job(job_id,
                   status      = "done",
                   func_count  = total_funcs,
                   score_avg   = round(avg_score, 2),
                   output_path = str(out_path),
                   is_dummy    = DUMMY_MODE,
                   finished_at = datetime.utcnow().isoformat())

        if not DUMMY_MODE:
            log_quota(groq_tokens=total_funcs * 700, github_calls=total_funcs * 2)

        _push_log(job_id,
                  f"✅ Done! {total_funcs} functions, {len(all_results)} files. "
                  f"Avg score: {avg_score:.1f}/5",
                  "success")

        _push_done(job_id, {
            "download_url": f"/api/download/{job_id}",
            "filename"    : f"{project_name}_docs.docx",
            "score_avg"   : round(avg_score, 2),
            "func_count"  : total_funcs,
            "file_count"  : len(all_results),
            "is_dummy"    : DUMMY_MODE,
            "job_id"      : job_id,
        })

    except Exception as e:
        _push_log(job_id, f"❌ {e}", "error")
        update_job(job_id, status="error", error_msg=str(e),
                   finished_at=datetime.utcnow().isoformat())
        _push_error(job_id, str(e))
    finally:
        sys.stdout = orig
        try:
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app_inst):
    init_db()
    mode = "DUMMY (no API calls)" if DUMMY_MODE else "REAL (Groq + GPT-4o + gpt-4o-mini)"
    print(f"\n{'='*58}")
    print(f"  CodeDoc AI Web App")
    print(f"  Mode        : {mode}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Outputs     : {OUTPUT_DIR}")
    print(f"  DB          : {BACKEND_DIR / 'jobs.db'}")
    print(f"{'='*58}\n")
    yield

app = FastAPI(title="CodeDoc AI", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status"    : "ok",
        "dummy_mode": DUMMY_MODE,
        "pipeline"  : "Groq → GPT-4o → gpt-4o-mini + Phi-4",
        "timestamp" : datetime.utcnow().isoformat(),
    }


@app.get("/api/limits")
async def get_limits():
    quota = get_today_quota()
    return {
        "limits"     : LIMITS,
        "quota_today": quota,
        "dummy_mode" : DUMMY_MODE,
    }


@app.post("/api/snippet")
async def submit_snippet(
    background_tasks: BackgroundTasks,
    code    : str = Form(...),
    language: str = Form("python"),
):
    code = code.strip()
    if len(code) < 5:
        raise HTTPException(400, "Code snippet is too short (min 5 chars)")
    if code.count("\n") > 200:
        raise HTTPException(400, "Snippet too long — max 200 lines")

    job_id = str(uuid.uuid4())
    create_job(job_id, "snippet", f"snippet.{language[:3]}")
    _get_queue(job_id)
    background_tasks.add_task(_run_snippet, job_id, code, language)
    return {"job_id": job_id, "dummy_mode": DUMMY_MODE}


@app.post("/api/file")
async def submit_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    content = await file.read()
    if len(content) > 100 * 1024:
        raise HTTPException(400, "File too large — max 100 KB")

    allowed = {".py", ".js", ".ts", ".java", ".cpp", ".c"}
    suffix  = Path(file.filename or "file.py").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. "
                                  f"Allowed: {', '.join(sorted(allowed))}")

    job_id    = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}{suffix}"
    save_path.write_bytes(content)

    create_job(job_id, "file", file.filename or "upload")
    _get_queue(job_id)
    background_tasks.add_task(_run_file, job_id, save_path,
                               file.filename or f"upload{suffix}")
    return {"job_id": job_id, "dummy_mode": DUMMY_MODE}


@app.post("/api/zip")
async def submit_zip(
    background_tasks: BackgroundTasks,
    file        : UploadFile = File(...),
    project_name: str        = Form("my_project"),
):
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(400, "ZIP too large — max 5 MB")
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Must be a .zip file")

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", project_name.strip())[:40] or "project"
    job_id    = str(uuid.uuid4())
    zip_path  = UPLOAD_DIR / f"{job_id}.zip"
    zip_path.write_bytes(content)

    create_job(job_id, "zip", file.filename or "project.zip")
    _get_queue(job_id)
    background_tasks.add_task(_run_zip, job_id, zip_path, safe_name)
    return {"job_id": job_id, "dummy_mode": DUMMY_MODE}


@app.get("/api/stream/{job_id}")
async def stream_logs(job_id: str):
    q = _get_queue(job_id)

    async def generator():
        while True:
            try:
                raw = await asyncio.wait_for(q.get(), timeout=30)
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"
                continue
            msg = json.loads(raw)
            if msg.get("type") == "__close__":
                yield "event: close\ndata: {}\n\n"
                break
            yield f"data: {raw}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/job/{job_id}")
async def get_single_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs")
async def get_jobs(limit: int = 30):
    return list_jobs(limit)


@app.get("/api/quota")
async def quota():
    return get_today_quota()


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job status is '{job['status']}', not done yet")

    out = Path(job["output_path"])
    if not out.exists():
        raise HTTPException(404, "Output file not found on disk")

    media = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if out.suffix == ".docx" else "text/plain; charset=utf-8"
    )
    base = Path(job["input_name"]).stem
    return FileResponse(str(out), media_type=media,
                        filename=f"{base}_docs{out.suffix}")


_DIST = WEBAPP_DIR / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")
else:
    @app.get("/")
    async def frontend_missing():
        return {"error": "Frontend not found",
                "fix"  : f"Copy index.html to {_DIST}/index.html"}


if __name__ == "__main__":
    import sys as _sys
    # Use reload=False in production for fast startup.
    # Pass --dev flag to enable file-watching reload during development:
    #   python app.py --dev
    dev_mode = "--dev" in _sys.argv

    uvicorn.run(
        "app:app",
        host        = "127.0.0.1",
        port        = 8000,
        reload      = dev_mode,
        # Only watch the backend folder when reload is on —
        # prevents restarts when llm/ or core/ files change
        reload_dirs = [str(BACKEND_DIR)] if dev_mode else None,
        log_level   = "warning",   # suppress per-request INFO logs — less noise
    )