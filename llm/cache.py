"""
cache.py — Persistent JSON key-value cache for all 3 pipeline stages.

Keys: MD5(code + stage)
Values: raw API response strings (JSON facts, prose docs, scores)

FIX: threading.Lock added to set() — Stage 3 runs GitHub calls in parallel
     (ThreadPoolExecutor, 5 workers). Without a lock, concurrent writes all
     read the same stale file state, then each overwrites the previous write.
     Result: only 1 entry survived per parallel batch instead of 7.
     With the lock, writes are serialised and all entries are preserved.
"""

import json
import hashlib
import os
import threading

CACHE_FILE = "cache/doc_cache.json"
os.makedirs("cache", exist_ok=True)

# One lock for the entire process — all threads share it
_lock = threading.Lock()


def _key(code: str, stage: str) -> str:
    return hashlib.md5(
        f"{code.strip()}{stage}".encode()
    ).hexdigest()


def get(code: str, stage: str):
    """Return cached value or None. Never raises."""
    try:
        with open(CACHE_FILE) as f:
            return json.load(f).get(_key(code, stage))
    except Exception:
        return None


def set(code: str, stage: str, value: str):
    """
    Write a value to cache.

    FIX: Entire read-modify-write is inside _lock so parallel Stage 3 threads
    cannot overwrite each other's entries.
    """
    try:
        with _lock:
            try:
                with open(CACHE_FILE) as f:
                    data = json.load(f)
            except Exception:
                data = {}
            data[_key(code, stage)] = value
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  Cache write error: {e}")


def count() -> int:
    """Return number of cached entries."""
    try:
        with open(CACHE_FILE) as f:
            return len(json.load(f))
    except Exception:
        return 0