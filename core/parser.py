"""
parser.py — Code unit extraction for all supported languages.

UPDATED:
  - Added regex_fallback() between tree-sitter and chunk_fallback.
    Fixes "section_1" naming for C/C++/Java/JS when tree-sitter is unavailable.
  - parse_treesitter() now extracts real function names (was always "func_N").
  - language stored in CodeUnit and used downstream for correct code fences.
"""

import re
import ast
from pathlib import Path
from dataclasses import dataclass
from config import Config


@dataclass
class CodeUnit:
    name      : str
    code      : str
    unit_type : str   # "function" / "class" / "chunk"
    language  : str
    start_line: int
    end_line  : int


def detect_language(file_path: Path) -> str:
    return Config.SUPPORTED_EXTENSIONS.get(file_path.suffix, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
#  Python parser — AST (most accurate)
# ─────────────────────────────────────────────────────────────────────────────

def parse_python(file_path: Path) -> list:
    units = []
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree   = ast.parse(source)
        lines  = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s    = node.lineno - 1
                e    = node.end_lineno
                code = "\n".join(lines[s:e])
                units.append(CodeUnit(
                    name       = node.name,
                    code       = code,
                    unit_type  = "function",
                    language   = "python",
                    start_line = s,
                    end_line   = e,
                ))
            elif isinstance(node, ast.ClassDef):
                s    = node.lineno - 1
                e    = node.end_lineno
                code = "\n".join(lines[s:e])
                units.append(CodeUnit(
                    name       = node.name,
                    code       = code,
                    unit_type  = "class",
                    language   = "python",
                    start_line = s,
                    end_line   = e,
                ))
    except Exception as e:
        print(f"    Parse error: {e}, using fallback")
        return chunk_fallback(file_path, "python")

    return units if units else chunk_fallback(file_path, "python")


# ─────────────────────────────────────────────────────────────────────────────
#  Regex fallback — extracts real names for C/C++/Java/JS/TS
#  FIX: replaces chunk_fallback as middle tier, so C code gets real names
# ─────────────────────────────────────────────────────────────────────────────

# Per-language regex patterns — each captures group(1) as function name
# C pattern does NOT require start-of-line anchor so it matches bare snippets.
_REGEX_PATTERNS = {
    "c": re.compile(
        r'(?:static\s+|extern\s+|inline\s+|volatile\s+)?'
        r'(?:unsigned\s+|signed\s+|long\s+|short\s+|const\s+)*'
        r'(?:int|void|char|float|double|long|short|unsigned|signed|bool|size_t|\w+_t|\w+)\s*\**\s+'
        r'(\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE,
    ),
    "cpp": re.compile(
        r'^\s*(?:static\s+|virtual\s+|inline\s+|explicit\s+|constexpr\s+)?'
        r'(?:const\s+)?(?:[\w:<>]+\s+)+?'
        r'(\w+)\s*\([^)]*\)(?:\s*const)?\s*(?:noexcept)?\s*(?:override)?\s*\{',
        re.MULTILINE,
    ),
    "java": re.compile(
        r'^\s*(?:public|private|protected|static|final|synchronized|native|abstract)?'
        r'(?:\s+(?:public|private|protected|static|final|synchronized))*'
        r'\s+[\w<>\[\]]+\s+'
        r'(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
        re.MULTILINE,
    ),
    "javascript": re.compile(
        r'(?:'
        r'function\s+(\w+)\s*\('          # named function declaration
        r'|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>'  # arrow fn
        r'|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?function'        # fn expression
        r')',
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r'(?:'
        r'function\s+(\w+)\s*(?:<[^>]*>)?\s*\('
        r'|(?:const|let|var)\s+(\w+)\s*(?::\s*[\w<>\[\]|&]+)?\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[\w<>\[\]|&]+)?\s*=>'
        r'|(?:public|private|protected|static|async)?\s+(\w+)\s*\([^)]*\)\s*(?::\s*[\w<>\[\]|&]+)?\s*\{'
        r')',
        re.MULTILINE,
    ),
}


def _extract_ts_context(source: str) -> str:
    """
    Extract TypeScript interface and type alias declarations from source.
    These are prepended to each function's code so Groq/GPT-4o understands
    parameter shapes (e.g. what fields Student has) without re-analysing
    the whole file.

    Matches:
      interface Foo { ... }
      type Bar = { ... }
      type Bar = string | number;
    """
    blocks = []

    # interface Name { ... }  — possibly multiline
    for m in re.finditer(r'\binterface\s+\w+\s*(?:extends\s+[\w,\s<>]+)?\{[^}]*\}', source, re.DOTALL):
        blocks.append(m.group(0).strip())

    # type Name = { ... }  or  type Name = Foo | Bar;
    for m in re.finditer(r'\btype\s+\w+\s*=\s*(?:\{[^}]*\}|[^;]+);', source, re.DOTALL):
        blocks.append(m.group(0).strip())

    return "\n\n".join(blocks)


def regex_fallback(file_path: Path, lang: str) -> list:
    """
    Extract functions by regex — middle tier between tree-sitter and chunk_fallback.
    Gives real function names for C/C++/Java/JS/TS.

    For short snippets where regex finds nothing, treats the whole file as
    one unit and extracts the name from the first function-like line.
    Falls back to chunk_fallback only if content is very long.

    FIX (TypeScript/JavaScript): interface and type declarations are prepended
    to each function's code so the AI pipeline understands parameter types.
    """
    pattern = _REGEX_PATTERNS.get(lang)
    if not pattern:
        return chunk_fallback(file_path, lang)

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        lines  = source.splitlines()
        units  = []
        seen   : set = set()

        # Collect TS/JS type context once for the whole file
        ts_context = ""
        if lang in ("typescript", "javascript"):
            ts_context = _extract_ts_context(source)

        for m in pattern.finditer(source):
            fn_name = next((g for g in m.groups() if g), None)
            if not fn_name or fn_name in seen:
                continue
            # Skip keywords that look like function names.
            # NOTE: 'main' is only skipped for C/C++ — in Java/JS/TS it is
            # a real documented entry point and should be kept.
            c_only_keywords = {"main"} if lang in ("c", "cpp") else set()
            skip_keywords   = {"if","else","for","while","switch","return","sizeof",
                                "typedef","struct","enum","union"} | c_only_keywords
            if fn_name in skip_keywords:
                continue
            seen.add(fn_name)

            start_line = source[:m.start()].count("\n")
            end_line   = _find_closing_brace(lines, start_line)

            # Safety: ensure end_line is always past start_line
            if end_line <= start_line:
                end_line = min(start_line + 80, len(lines))

            code = "\n".join(lines[start_line:end_line])
            if not code.strip():
                continue

            # FIX: prepend interface/type context for TS/JS so the AI
            # pipeline knows what each parameter type looks like.
            if ts_context and ts_context not in code:
                code = ts_context + "\n\n" + code

            units.append(CodeUnit(
                name       = fn_name,
                code       = code,
                unit_type  = "function",
                language   = lang,
                start_line = start_line,
                end_line   = end_line,
            ))

        if units:
            print(f"    ✓ Regex extracted {len(units)} function(s) from {file_path.name}")
            return units

        # No regex matches — short snippet fallback:
        # treat whole file as one unit, name from first identifier before '('
        if len(lines) <= 120:
            name = _extract_name_from_first_line(source, lang)
            if name:
                print(f"    ✓ Single-unit fallback: '{name}' from {file_path.name}")
                # FIX: for TS/JS, the full source already contains the interfaces,
                # so no need to prepend — just use source directly.
                return [CodeUnit(
                    name       = name,
                    code       = source,   # full source = interfaces + function
                    unit_type  = "function",
                    language   = lang,
                    start_line = 0,
                    end_line   = len(lines),
                )]

    except Exception as e:
        print(f"    Regex fallback error: {e}")

    return chunk_fallback(file_path, lang)


def _extract_name_from_first_line(source: str, lang: str) -> str | None:
    """
    Last-resort name extractor: scan source for the first identifier
    immediately before '(' that isn't a keyword or type name.
    Works for all C-family languages.
    """
    _keywords = {
        "if","else","for","while","do","switch","return","sizeof","typedef",
        "struct","class","enum","union","namespace","void","int","char",
        "float","double","bool","long","short","unsigned","signed","static",
        "const","inline","extern","volatile","public","private","protected",
        "function","async","await","new","delete","try","catch","throw",
        "import","export","default","var","let","const",
    }
    # Find identifiers before '('
    matches = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', source)
    for name in matches:
        if name.lower() not in _keywords and len(name) > 1:
            return name
    return None


def _find_closing_brace(lines: list, start: int, max_lines: int = 200) -> int:
    """
    Find the line number of the closing brace matching the opening brace
    of the function that begins at `start`.

    FIX: depth starts at 0 and we count every { and } from the start line
    onwards. This correctly handles cases where the opening { is on the
    same line as the function signature (the common case for TS/JS/Java).
    Previously, starting depth at 0 and skipping the start line caused
    early exit when { and } counts happened to balance before the real end.
    """
    depth = 0
    for i, line in enumerate(lines[start:start + max_lines], start=start):
        depth += line.count("{") - line.count("}")
        # Only close once we've seen at least one opening brace (depth was >0)
        if depth <= 0 and i > start:
            return i + 1
        if depth <= 0 and i == start and "{" not in line:
            # Signature-only line with no brace yet — keep scanning
            continue
    # Fallback: return up to 80 lines from start
    return min(start + 80, len(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Tree-sitter parser — uses real names now
# ─────────────────────────────────────────────────────────────────────────────

def parse_treesitter(file_path: Path, lang: str) -> list:
    from tree_sitter_languages import get_parser
    parser = get_parser(lang)
    source  = file_path.read_bytes()
    tree    = parser.parse(source)
    src_str = source.decode("utf-8", errors="ignore")
    lines   = src_str.splitlines()
    units   = []

    def _get_name(node) -> str:
        """Extract real function name from tree-sitter node."""
        for child in node.children:
            if child.type in ("identifier", "name"):
                return src_str[child.start_byte:child.end_byte]
        return f"func_{len(units)+1}"

    def walk(node):
        if node.type in ("function_definition", "method_declaration",
                         "function_declaration", "method_definition"):
            s    = node.start_point[0]
            e    = node.end_point[0] + 1
            code = "\n".join(lines[s:e])
            name = _get_name(node)
            units.append(CodeUnit(
                name       = name,
                code       = code,
                unit_type  = "function",
                language   = lang,
                start_line = s,
                end_line   = e,
            ))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return units if units else regex_fallback(file_path, lang)


# ─────────────────────────────────────────────────────────────────────────────
#  Chunk fallback — last resort, generic names
# ─────────────────────────────────────────────────────────────────────────────

def chunk_fallback(file_path: Path, lang: str) -> list:
    """Split file into fixed-size chunks if all other parsing fails."""
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    lines  = source.splitlines()
    units  = []
    size   = 40

    for i in range(0, len(lines), size):
        chunk = "\n".join(lines[i:i+size])
        if chunk.strip():
            units.append(CodeUnit(
                name       = f"section_{i//size + 1}",
                code       = chunk,
                unit_type  = "chunk",
                language   = lang,
                start_line = i,
                end_line   = min(i+size, len(lines)),
            ))
    return units


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_file(file_path: Path) -> list:
    """
    Parse a source file into CodeUnit list.

    Priority order:
      1. Python AST (Python files — perfect accuracy)
      2. tree-sitter (if installed — best for other languages)
      3. Regex fallback (NEW — real names for C/C++/Java/JS)
      4. chunk_fallback (last resort — generic section_N names)
    """
    lang = detect_language(file_path)

    if lang == "python":
        return parse_python(file_path)

    # Try tree-sitter first
    try:
        return parse_treesitter(file_path, lang)
    except Exception:
        pass

    # FIX: regex fallback gives real names — avoids chunk_fallback's section_N
    return regex_fallback(file_path, lang)