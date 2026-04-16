"""
config.py — Central configuration for AI Code Documentation Generator.

UPDATED: Gemini removed. GPT-4o (GitHub Student Pack) replaces it as Stage 2.
         Rate limits updated to match actual GitHub Student Developer Pack quotas.
         Groq stays as Stage 1 (fast JSON extraction — unchanged).
         gpt-4o-mini used for validation (2M TPM Student Pack — reliable). DeepSeek-R1 removed.
"""

import os

# ── Load .env ──────────────────────────────────────────────────────────────────
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()


class Config:

    # ── API Keys ───────────────────────────────────────────────────────────────
    GROQ_API_KEY  = os.environ.get("GROQ_API_KEY",  "")
    GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN",  "")

    # Gemini key kept for backward compat but no longer used in pipeline
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

    # ── Models ─────────────────────────────────────────────────────────────────
    # Stage 1 — Groq: fast JSON fact extraction (unchanged)
    GROQ_MODEL = "llama-3.3-70b-versatile"

    # Stage 2 — GitHub GPT-4o: IBM prose writing (replaces Gemini)
    # GitHub Student Pack gives 10M TPM / 60K RPM — no daily cap issues
    GITHUB_WRITE_MODEL = "openai/gpt-4o"

    # Stage 3 — Validation + Security
    GITHUB_VALIDATE_MODEL = "openai/gpt-4o-mini"  # DeepSeek-R1 removed — hits RPM on every call
    GITHUB_SECURITY_MODEL = "microsoft/Phi-4"             # security scan (unchanged)

    # Fallback chain for Stage 3 if primary model fails
    GITHUB_MODELS = [
        "openai/gpt-4o",
        "microsoft/Phi-4",
        "meta/Llama-3.3-70B-Instruct",
        "openai/gpt-4o-mini",
    ]

    # ── Rate limits ────────────────────────────────────────────────────────────
    # Groq free tier — 12K TPM is the real bottleneck, not RPM
    GROQ_RPM = 30
    GROQ_RPD = 14_400
    GROQ_TPM = 12_000      # tokens per minute — true bottleneck

    # Groq batch: REDUCED from 5 to 3 to stay under 12K TPM per batch call.
    # Each function extraction ≈ 1,500-2,500 tokens input+output.
    # 3 functions × 2,500 = 7,500 tokens — safe under 12K TPM.
    # Was 5 functions × 2,500 = 12,500 — over limit → rate errors.
    GROQ_BATCH_DELAY = 5.0  # seconds to wait between Groq batch calls (TPM cooldown)

    # GitHub Student Developer Pack — actual per-model RPM is ~10-15
    # The 10M TPM / 60K RPM figures are account-wide token budgets,
    # NOT per-model per-minute request rates. Real observed limit: ~10 RPM.
    GITHUB_RPM = 10         # per-model requests per minute (real observed limit)
    GITHUB_RPD = 999_999    # no hard daily cap on Student Pack

    # GitHub Stage 2 (GPT-4o prose writing)
    GITHUB_WRITE_RPM = 10
    GITHUB_WRITE_RPD = 999_999

    # Delay between sequential GitHub calls (seconds)
    # 10 RPM = 1 request per 6s. We use 7s for safety margin.
    GITHUB_CALL_DELAY = 7.0

    # ── Token budgets per stage ────────────────────────────────────────────────
    GROQ_MAX_OUT_TOKENS    = 700    # Stage 1: JSON facts
    GITHUB_WRITE_MAX_TOKENS = 2500  # Stage 2: Full IBM prose (was 1200 in Gemini — INCREASED)
    GITHUB_MAX_OUT_TOKENS  = 300    # Stage 3: validate+score JSON

    # Legacy alias kept for any code that references it
    GEMINI_MAX_OUT_TOKENS  = 2500
    MAX_OUT_TOKENS         = 2500

    # ── Chunking ───────────────────────────────────────────────────────────────
    MAX_FUNC_CHARS   = 8_000
    LARGE_FUNC_CHARS = 5_000
    BATCH_SIZE       = 3    # REDUCED from 5 — prevents Groq 12K TPM overflow
    SMALL_LINES      = 50

    # ── Trivial function pre-filter ────────────────────────────────────────────
    # Functions under this threshold get template docs, no API call.
    # Set to False to force every function through the full pipeline.
    TRIVIAL_SKIP_ALWAYS = True

    # ── Prefix caching ─────────────────────────────────────────────────────────
    ENABLE_PREFIX_CACHE = True

    # ── Block-chunk context accumulator ───────────────────────────────────────
    BLOCK_CONTEXT_ENABLED = True

    # ── Incremental / change-detection mode ───────────────────────────────────
    INCREMENTAL_MODE = True

    # ── Security deep-scan ────────────────────────────────────────────────────
    SECURITY_SCAN_ENABLED = True

    # ── Display limits ─────────────────────────────────────────────────────────
    MAX_SECURITY_DISPLAY = None   # None = show all

    # ── Paths ──────────────────────────────────────────────────────────────────
    EXTRACT_DIR = "extracted/"
    OUTPUT_DIR  = "output/"
    CACHE_DIR   = "cache/"
    MIN_FILES   = 1

    # ── Supported languages ────────────────────────────────────────────────────
    SUPPORTED_EXTENSIONS = {
        ".py"  : "python",
        ".java": "java",
        ".cpp" : "cpp",
        ".js"  : "javascript",
        ".c"   : "c",
        ".ts"  : "typescript",
    }

    IGNORED_DIRS = {
        "venv", ".git", "__pycache__",
        "node_modules", ".idea",
        "dist", "build", ".env",
    }

    # ── Dependency name → PyPI package mapping ─────────────────────────────────
    DEP_ALIASES: dict = {
        "st"                : "streamlit",
        "streamlit"         : "streamlit",
        "streamlit library" : "streamlit",
        "sql"               : "mysql-connector-python",
        "sqlite3"           : None,
        "PIL"               : "Pillow",
        "pillow"            : "Pillow",
        "cv2"               : "opencv-python",
        "sklearn"           : "scikit-learn",
        "bs4"               : "beautifulsoup4",
        "yaml"              : "pyyaml",
        "dotenv"            : "python-dotenv",
        "nominatim"         : "geopy",
        "geocoder"          : "geocoder",
        "geopy"             : "geopy",
        "docx"              : "python-docx",
        "pd"                : "pandas",
        "np"                : "numpy",
        "plt"               : "matplotlib",
        "px"                : "plotly",
        "tf"                : "tensorflow",
        "torch"             : "torch",
        "spacy"             : "spacy",
        "nltk"              : "nltk",
        "flask"             : "flask",
        "fastapi"           : "fastapi",
        "django"            : "django",
        "requests"          : "requests",
        "groq"              : "groq",
        "google"            : "google-genai",
        "genai"             : "google-genai",
        "anthropic"         : "anthropic",
        "openai"            : "openai",
        "pymysql"           : "pymysql",
        "psycopg2"          : "psycopg2-binary",
        "redis"             : "redis",
        "celery"            : "celery",
        "pydantic"          : "pydantic",
        "aiohttp"           : "aiohttp",
        "httpx"             : "httpx",
        "pytest"            : "pytest",
        "boto3"             : "boto3",
        "plotly"            : "plotly",
        "plotly (px)"       : "plotly",
        "llmlingua"         : "llmlingua",
        "matplotlib"        : "matplotlib",
        "pypdf2"            : "PyPDF2",
        "pdfminer"          : "pdfminer.six",
        "pdfplumber"        : "pdfplumber",
        "docstring"         : None,
        "cross_references"  : None,
        "example"           : None,
        "utils"             : None,
        "config"            : None,
        "socket"            : None,
        "os"                : None,
        "sys"               : None,
        "re"                : None,
        "json"              : None,
        "ast"               : None,
        "time"              : None,
        "datetime"          : None,
        "pathlib"           : None,
        "platform"          : None,
        "hashlib"           : None,
        "base64"            : None,
        "io"                : None,
        "secrets"           : None,
        "random"            : None,
        "math"              : None,
        "collections"       : None,
        "itertools"         : None,
        "functools"         : None,
        "typing"            : None,
        "abc"               : None,
        "copy"              : None,
        "subprocess"        : None,
        "threading"         : None,
        "multiprocessing"   : None,
        "logging"           : None,
        "unittest"          : None,
        "csv"               : None,
        "string"            : None,
        "struct"            : None,
        "enum"              : None,
        "dataclasses"       : None,
        "contextlib"        : None,
        "shutil"            : None,
        "tempfile"          : None,
        "glob"              : None,
        "fnmatch"           : None,
        "pickle"            : None,
        "shelve"            : None,
        "urllib"            : None,
        "http"              : None,
        "email"             : None,
        "html"              : None,
        "xml"               : None,
        "zipfile"           : None,
        "tarfile"           : None,
        "gzip"              : None,
        "warnings"          : None,
        "traceback"         : None,
        "inspect"           : None,
        "importlib"         : None,
        "textwrap"          : None,
        "pprint"            : None,
    }