"""
database.py — SQLite job-tracking database for the web app.

PURPOSE: This is SEPARATE from the existing cache.py (which caches raw
         API responses in cache/doc_cache.json). This tracks web app jobs:
         who submitted what, current status, scores, download links.

Tables:
  jobs       — one row per documentation request (snippet / file / zip)
  quota_log  — daily API usage totals for the quota display
"""

import sqlite3
import threading
from pathlib import Path
from datetime import datetime, date
from typing import Optional

# DB lives alongside app.py in webapp/backend/
DB_PATH = Path(__file__).parent / "jobs.db"
_lock   = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


# ─────────────────────────────────────────────────────────────────────────────
#  Schema init (called once at startup)
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                mode         TEXT NOT NULL,   -- snippet | file | zip
                input_name   TEXT NOT NULL,   -- filename or "snippet"
                status       TEXT NOT NULL,   -- queued | running | done | error
                created_at   TEXT NOT NULL,
                finished_at  TEXT,
                error_msg    TEXT,
                output_path  TEXT,            -- absolute path to generated file
                token_est    INTEGER DEFAULT 0,
                score_avg    REAL    DEFAULT 0,
                func_count   INTEGER DEFAULT 0,
                is_dummy     INTEGER DEFAULT 0  -- 1 = generated with dummy mode
            );

            CREATE TABLE IF NOT EXISTS quota_log (
                log_date     TEXT PRIMARY KEY,
                groq_tokens  INTEGER DEFAULT 0,
                gemini_calls INTEGER DEFAULT 0,
                github_calls INTEGER DEFAULT 0,
                updated_at   TEXT NOT NULL
            );
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  Jobs CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_job(job_id: str, mode: str, input_name: str) -> dict:
    now = datetime.utcnow().isoformat()
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO jobs(id,mode,input_name,status,created_at) VALUES(?,?,?,?,?)",
            (job_id, mode, input_name, "queued", now)
        )
    return {"id": job_id, "mode": mode, "input_name": input_name,
            "status": "queued", "created_at": now}


def update_job(job_id: str, **kwargs):
    if not kwargs:
        return
    sets   = ", ".join(f"{k}=?" for k in kwargs)
    values = list(kwargs.values()) + [job_id]
    with _lock, _conn() as c:
        c.execute(f"UPDATE jobs SET {sets} WHERE id=?", values)


def get_job(job_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
#  Quota tracking
# ─────────────────────────────────────────────────────────────────────────────

def log_quota(groq_tokens: int = 0, gemini_calls: int = 0, github_calls: int = 0):
    today = date.today().isoformat()
    now   = datetime.utcnow().isoformat()
    with _lock, _conn() as c:
        exists = c.execute(
            "SELECT 1 FROM quota_log WHERE log_date=?", (today,)
        ).fetchone()
        if exists:
            c.execute("""
                UPDATE quota_log
                   SET groq_tokens  = groq_tokens  + ?,
                       gemini_calls = gemini_calls + ?,
                       github_calls = github_calls + ?,
                       updated_at   = ?
                 WHERE log_date = ?
            """, (groq_tokens, gemini_calls, github_calls, now, today))
        else:
            c.execute(
                "INSERT INTO quota_log VALUES(?,?,?,?,?)",
                (today, groq_tokens, gemini_calls, github_calls, now)
            )


def get_today_quota() -> dict:
    today = date.today().isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM quota_log WHERE log_date=?", (today,)
        ).fetchone()
    if row:
        return dict(row)
    return {"log_date": today, "groq_tokens": 0,
            "gemini_calls": 0, "github_calls": 0}