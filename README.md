# AI Code Documentation Generator

An AI-powered tool that automatically generates professional, IBM 4-Tier standard documentation for multi-language codebases. Upload a ZIP of your source code and receive a fully formatted `.docx` documentation file, architecture diagram, security findings, and a `README.md` — all in one run.

---

## Features

- **Multi-language support** — Parses Python, JavaScript, TypeScript, Java, C, and C++ source files using Tree-sitter and regex-based fallback parsers.
- **Two-stage AI pipeline** — Groq handles fast fact extraction per function; GPT-4o generates polished IBM-standard documentation; `gpt-4o-mini` + Phi-4 handle secondary summarisation.
- **Cyclomatic complexity analysis** — Computes complexity scores, line counts, parameter counts, and recursion detection per function; outputs colour-coded complexity tables.
- **Architecture diagram generation** — Automatically produces a visual component diagram (`architecture.png`) showing how files and modules relate.
- **Security scanning** — Flags security-relevant patterns in function code (e.g. injection risks, unsafe calls) and aggregates findings across the whole project.
- **Incremental mode (Git-aware)** — Detects which files changed since the last commit and skips unchanged files, reducing API cost and runtime.
- **Response caching** — LLM responses are cached locally so repeated runs on the same code don't incur duplicate API calls.
- **Auto-generated README** — Produces a `_README.md` for the uploaded project based on extracted facts, real function names, and resolved dependencies.

---

## Project Structure

```
ai-doc-generator/
│
├── main.py               — Orchestration: ZIP → parse → document → save
├── config.py             — Central configuration (paths, model names, flags, aliases)
├── requirements.txt      — All Python dependencies
├── check_key.py          — Utility to validate API keys
├── check_limit.py        — Utility to check API rate limits
├── clear_cache.py        — Utility to clear the local LLM response cache
│
├── core/
│   ├── zip_handler.py    — Extracts uploaded ZIP files to a temp directory
│   ├── file_scanner.py   — Discovers supported source files recursively
│   ├── parser.py         — Language-specific function/class unit extraction
│   └── complexity.py     — Cyclomatic complexity analysis (Python-native + regex fallback)
│
├── llm/
│   ├── pipeline.py       — Orchestrates Groq → GPT-4o documentation pipeline per file
│   ├── groq_client.py    — Groq API client (fact extraction + README generation)
│   ├── github_client.py  — GPT-4o API client via GitHub Models endpoint
│   └── cache.py          — Disk-based LLM response cache
│
├── output/
│   ├── doc_builder.py    — Builds the .docx document (headings, tables, summaries)
│   └── architecture.py   — Generates the architecture PNG diagram
│
└── webapp/               — (FastAPI web interface)
```

---

## Installation

### Prerequisites

- Python 3.10+
- API keys for **Groq** and **OpenAI / GitHub Models** (GPT-4o)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Lokeshvari22/ai-doc-generator.git
cd ai-doc-generator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys — create a .env file
GROQ_API_KEY=your_groq_key_here
GITHUB_TOKEN=your_github_models_token_here   # used for GPT-4o access
OPENAI_API_KEY=your_openai_key_here          # optional fallback
```

---

## Usage

### Command-line

```bash
python main.py path/to/your_project.zip
```

Or run without arguments and enter the path when prompted:

```bash
python main.py
# Enter ZIP file path: /home/user/myproject.zip
```

### Output

After a successful run, the `output/` directory will contain:

| File | Description |
|------|-------------|
| `<project>_docs.docx` | Full IBM 4-Tier documentation with per-function docs, complexity tables, architecture diagram, and project summary |
| `<project>_README.md` | Auto-generated README for the uploaded project |
| `architecture.png` | Visual component/module diagram |

### Web Interface

A FastAPI web app is available in the `webapp/` directory:

```bash
uvicorn webapp.app:app --reload
# Visit http://localhost:8000
```

---

## Pipeline Overview

```
ZIP Upload
   │
   ▼
[1] Extract ZIP          ← core/zip_handler.py
[2] Scan + Git diff      ← core/file_scanner.py
[3] Parse functions      ← core/parser.py  (Tree-sitter + regex fallback)
[4] Groq: extract facts  ← llm/groq_client.py  (purpose, params, security notes, deps)
    └─► GPT-4o: write docs ← llm/github_client.py  (IBM 4-tier formatted documentation)
[4b] Resolve cross-file dependencies (IJIRT standard)
[5] Generate architecture diagram  ← output/architecture.py
[6] Aggregate security findings + generate README
[7] Build + save .docx   ← output/doc_builder.py
```

Trivial functions (getters, one-liners) are detected early and handled with template docs — no API call is made, keeping costs low.

---

## Configuration

Key settings in `config.py`:

| Setting | Description |
|---------|-------------|
| `INCREMENTAL_MODE` | `True` to skip git-unchanged files |
| `OUTPUT_DIR` | Where docs and diagrams are saved |
| `CACHE_DIR` | Where LLM response cache is stored |
| `EXTRACT_DIR` | Where ZIPs are extracted |
| `MAX_SECURITY_DISPLAY` | Max security findings shown in terminal |
| `DEP_ALIASES` | Maps internal import names to PyPI package names |

---

## Supported Languages

| Language | Parser |
|----------|--------|
| Python | `ast` module + `radon` for complexity |
| JavaScript / TypeScript | Tree-sitter + regex fallback |
| Java | Tree-sitter + regex fallback |
| C / C++ | Tree-sitter + regex fallback |

---

## Security Notes

The tool scans for common security patterns in each function and reports them in both the `.docx` output and the terminal summary. Findings are labelled by function name and risk level (`LOW` / `MEDIUM` / `HIGH`).

---

## Key Dependencies

| Package | Role |
|---------|------|
| `groq` | Fast LLM inference for fact extraction |
| `openai` / GitHub Models | GPT-4o documentation generation |
| `tree-sitter`, `tree-sitter-languages` | Source code parsing |
| `python-docx` | `.docx` document generation |
| `radon` | Python cyclomatic complexity |
| `matplotlib`, `graphviz` | Architecture diagram rendering |
| `fastapi`, `uvicorn` | Web interface |
| `transformers`, `torch` | Local model support (Phi-4) |
| `llmlingua` | Prompt compression |
| `tiktoken` | Token counting |
| `python-dotenv` | Environment variable management |

---

## License

This project is currently unlicensed. Contact the repository owner for usage terms.
