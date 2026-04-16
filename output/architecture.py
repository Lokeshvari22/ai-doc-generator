"""
architecture.py — Professional architecture diagram generator.

P5 FIX: generate_architecture() now returns (img_path, file_analyses).
        file_analyses is passed to add_architecture() as fallback data
        so the Word doc always has SOMETHING even if matplotlib fails.
        All matplotlib operations wrapped in try/except with clear messages.
"""

import ast
import re
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    _MATPLOTLIB_OK = True
except Exception:
    _MATPLOTLIB_OK = False

PALETTE = {
    "bg"            : "#0d0d1a",
    "grid"          : "#1e1e35",
    "file_header"   : "#1a1a2e",
    "class_box"     : "#0f3460",
    "class_border"  : "#7fc8f8",
    "func_public"   : "#1a1a35",
    "func_private"  : "#121225",
    "external_box"  : "#2d6a4f",
    "external_border": "#3a9e6e",
    "cc_low"        : "#2d6a4f",
    "cc_med"        : "#e9b44c",
    "cc_high"       : "#c03221",
    "edge_import"   : "#7fc8f8",
    "edge_call"     : "#f5a623",
    "text_light"    : "#f0f0f0",
    "text_dim"      : "#aaaaaa",
    "text_blue"     : "#7fc8f8",
    "text_green"    : "#b8e994",
    "accent_border" : "#4a9eff",
}

KNOWN_EXTERNAL = {
    "streamlit": "Streamlit",   "pandas": "pandas",
    "numpy": "NumPy",           "flask": "Flask",
    "fastapi": "FastAPI",       "django": "Django",
    "sqlalchemy": "SQLAlchemy", "pymysql": "PyMySQL",
    "requests": "requests",     "aiohttp": "aiohttp",
    "spacy": "spaCy",           "nltk": "NLTK",
    "sklearn": "scikit-learn",  "torch": "PyTorch",
    "tensorflow": "TensorFlow", "groq": "Groq",
    "google": "Gemini / Google","openai": "OpenAI",
    "anthropic": "Anthropic",   "pdfminer": "PDFMiner",
    "pypdf2": "PyPDF2",         "geocoder": "geocoder",
    "plotly": "Plotly",         "PIL": "Pillow",
    "cv2": "OpenCV",            "pytest": "pytest",
    "pydantic": "Pydantic",     "sqlmodel": "SQLModel",
    "boto3": "AWS boto3",       "redis": "Redis",
    "celery": "Celery",         "asyncpg": "asyncpg",
}


def _cyclomatic(node: ast.FunctionDef) -> int:
    cc = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For,
                               ast.ExceptHandler, ast.With,
                               ast.Assert, ast.comprehension)):
            cc += 1
        elif isinstance(child, ast.BoolOp):
            cc += len(child.values) - 1
    return cc


def _cc_color(cc: int) -> str:
    if cc <= 3: return PALETTE["cc_low"]
    if cc <= 7: return PALETTE["cc_med"]
    return PALETTE["cc_high"]


def analyse_file(path: Path) -> dict:
    result = {
        "imports"      : [],
        "classes"      : [],
        "functions"    : [],
        "calls"        : [],
        "external_libs": set(),
    }
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree   = ast.parse(source)
    except Exception:
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0].lower()
                result["imports"].append(base)
                if base in KNOWN_EXTERNAL:
                    result["external_libs"].add(KNOWN_EXTERNAL[base])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0].lower()
                result["imports"].append(base)
                if base in KNOWN_EXTERNAL:
                    result["external_libs"].add(KNOWN_EXTERNAL[base])

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            result["classes"].append({"name": node.name, "methods": methods, "lineno": node.lineno})

    func_names: set = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            cc = _cyclomatic(node)
            result["functions"].append({
                "name": node.name, "cc": cc,
                "is_private": node.name.startswith("_"),
                "lineno": node.lineno,
            })
            func_names.add(node.name)

    for fn_node in tree.body:
        if not isinstance(fn_node, ast.FunctionDef):
            continue
        for child in ast.walk(fn_node):
            if isinstance(child, ast.Call):
                callee = None
                if isinstance(child.func, ast.Name):
                    callee = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    callee = child.func.attr
                if callee and callee in func_names and callee != fn_node.name:
                    result["calls"].append((fn_node.name, callee))

    return result


def _rounded_box(ax, x, y, w, h, facecolor, edgecolor,
                  linewidth=1.0, alpha=1.0, radius=0.06):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle  = f"round,pad={radius}",
        linewidth = linewidth,
        edgecolor = edgecolor,
        facecolor = facecolor,
        alpha     = alpha,
    )
    ax.add_patch(box)


def _arrow(ax, x1, y1, x2, y2, color, lw=1.0, rad=0.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def generate_architecture(files: list, out_path: str) -> tuple:
    """
    Generate architecture diagram PNG.

    P5 FIX: Returns (img_path_or_None, file_analyses_dict).
    file_analyses is always populated even if matplotlib fails,
    so doc_builder.add_architecture() can show a text table fallback.

    Previously returned just img_path (str). Now returns tuple.
    Callers updated to: img, analyses = generate_architecture(...)
    """
    # Always analyse files — this works even without matplotlib
    analyses: dict = {}
    for f in files:
        try:
            analyses[f.name] = analyse_file(f)
        except Exception:
            analyses[f.name] = {"imports": [], "classes": [], "functions": [], "calls": [], "external_libs": set()}

    if not _MATPLOTLIB_OK:
        print("  ⚠ matplotlib not available — diagram skipped, text fallback used")
        return None, analyses

    try:
        return _draw_diagram(files, analyses, out_path), analyses
    except Exception as e:
        print(f"  ⚠ Diagram render failed: {e} — text fallback used")
        return None, analyses


def _draw_diagram(files: list, analyses: dict, out_path: str) -> str:
    """Internal: draw and save the matplotlib diagram."""
    n_files   = max(len(files), 1)
    COL_W     = 3.2
    EXT_W     = 2.2
    NODE_H    = 0.32
    NODE_GAP  = 0.08
    CLASS_PAD = 0.26
    HDR_H     = 0.55
    TITLE_H   = 0.55

    all_external = set()
    for a in analyses.values():
        all_external.update(a["external_libs"])

    max_items = max(
        (len(a["classes"]) * 4 + len(a["functions"]) for a in analyses.values()),
        default=5,
    )
    fig_h = TITLE_H + HDR_H + max_items * (NODE_H + NODE_GAP) + 1.5

    fig_w = n_files * COL_W + (EXT_W if all_external else 0.3) + 0.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.text(
        fig_w / 2, fig_h - TITLE_H / 2,
        "Project Architecture",
        color=PALETTE["text_light"], fontsize=12, fontweight="bold",
        ha="center", va="center",
    )

    node_pos: dict = {}

    for fi, f in enumerate(files):
        a      = analyses.get(f.name, {})
        col_x0 = fi * COL_W + 0.1
        col_w  = COL_W - 0.2
        cx     = col_x0 + col_w / 2
        y      = fig_h - TITLE_H

        _rounded_box(ax, col_x0, 0.3, col_w, y - 0.3,
                     facecolor=PALETTE["grid"], edgecolor=PALETTE["accent_border"],
                     linewidth=1.2, radius=0.05)
        _rounded_box(ax, col_x0, y - HDR_H, col_w, HDR_H,
                     facecolor=PALETTE["file_header"], edgecolor=PALETTE["accent_border"],
                     linewidth=1.5)

        ext_map = {"py": "Python", "js": "JavaScript", "ts": "TypeScript",
                   "java": "Java", "cpp": "C++", "c": "C"}
        lang = ext_map.get(f.suffix.lstrip("."), f.suffix.upper())

        ax.text(cx, y - HDR_H / 2 + 0.08, f.name,
                color=PALETTE["text_light"], fontsize=9.5, fontweight="bold",
                ha="center", va="center", fontfamily="monospace")
        ax.text(cx, y - HDR_H / 2 - 0.10, f"[ {lang} ]",
                color=PALETTE["text_blue"], fontsize=7, ha="center", va="center")

        y -= HDR_H + 0.18

        for cls in a.get("classes", []):
            n_methods = len(cls["methods"])
            box_h     = 0.40 + n_methods * CLASS_PAD
            _rounded_box(ax, col_x0, y - box_h, col_w, box_h,
                         facecolor=PALETTE["class_box"], edgecolor=PALETTE["class_border"],
                         linewidth=1.2)
            ax.text(cx, y - 0.20, f"\u25c6 {cls['name']}",
                    color=PALETTE["text_light"], fontsize=9, fontweight="bold",
                    ha="center", va="center")
            for mi, method in enumerate(cls["methods"][:7]):
                vis   = "+" if not method.startswith("_") else "-"
                label = method[:24] + ("..." if len(method) > 24 else "")
                ax.text(col_x0 + 0.18, y - 0.38 - mi * CLASS_PAD,
                        f"  {vis} {label}()",
                        color="#b8d4f0", fontsize=6.8,
                        va="center", fontfamily="monospace")
            if n_methods > 7:
                ax.text(cx, y - 0.38 - 7 * CLASS_PAD, f"  ... +{n_methods - 7} more",
                        color="#666688", fontsize=6.5, ha="center")
            node_pos[(f.name, cls["name"])] = (cx, y - box_h / 2)
            y -= box_h + NODE_GAP

        for fn in a.get("functions", []):
            if y - NODE_H < 0.6:
                remaining = len(a["functions"]) - a["functions"].index(fn)
                ax.text(cx, y - 0.15, f"  ... +{remaining} more",
                        color="#666688", fontsize=7, ha="center")
                break
            cc         = fn["cc"]
            cc_col     = _cc_color(cc)
            is_private = fn["is_private"]
            label      = fn["name"][:24] + (".." if len(fn["name"]) > 24 else "")
            _rounded_box(ax, col_x0, y - NODE_H, col_w, NODE_H,
                         facecolor=PALETTE["func_private"] if is_private else PALETTE["func_public"],
                         edgecolor=cc_col, linewidth=0.9, alpha=0.88)
            sym = "\u25c7 " if is_private else "\u25c6 "
            ax.text(col_x0 + 0.16, y - NODE_H / 2, sym + label + "()",
                    color=PALETTE["text_dim"] if is_private else PALETTE["text_light"],
                    fontsize=8, va="center", fontfamily="monospace")
            ax.text(col_x0 + col_w - 0.10, y - NODE_H / 2, f"cc:{cc}",
                    color=cc_col, fontsize=6.5, ha="right", va="center", fontweight="bold")
            node_pos[(f.name, fn["name"])] = (cx, y - NODE_H / 2)
            y -= NODE_H + NODE_GAP

    if all_external:
        px      = n_files * COL_W + 0.2
        py      = fig_h - TITLE_H - 0.3
        panel_w = EXT_W - 0.25
        ax.text(px + panel_w / 2, py - 0.20, "External\nDependencies",
                color=PALETTE["text_light"], fontsize=8.5, fontweight="bold",
                ha="center", va="center")
        for ei, lib in enumerate(sorted(all_external)):
            ey = py - 0.60 - ei * 0.40
            if ey < 0.8:
                ax.text(px + panel_w / 2, ey, f"  ... +{len(all_external) - ei} more",
                        color="#666688", fontsize=7, ha="center")
                break
            _rounded_box(ax, px + 0.08, ey - 0.27, panel_w - 0.16, 0.30,
                         facecolor=PALETTE["external_box"], edgecolor=PALETTE["external_border"],
                         linewidth=0.8, alpha=0.88)
            ax.text(px + panel_w / 2, ey - 0.12, f"[pkg] {lib}",
                    color=PALETTE["text_green"], fontsize=7.5, ha="center", va="center")

    for fname, a in analyses.items():
        for (caller, callee) in set(a.get("calls", [])):
            p1 = node_pos.get((fname, caller))
            p2 = node_pos.get((fname, callee))
            if p1 and p2 and p1 != p2:
                _arrow(ax, p1[0], p1[1], p2[0], p2[1],
                       color=PALETTE["edge_call"], lw=0.85, rad=0.35)

    stem_map = {f.stem: f.name for f in files}
    for fi, f in enumerate(files):
        for imp in analyses[f.name].get("imports", []):
            target = stem_map.get(imp)
            if target and target != f.name:
                ti = next((i for i, ff in enumerate(files) if ff.name == target), None)
                if ti is None:
                    continue
                x1     = fi * COL_W + COL_W / 2 + 0.1
                x2     = ti * COL_W + COL_W / 2 + 0.1
                y_edge = fig_h - TITLE_H - 0.55
                _arrow(ax, x1, y_edge, x2, y_edge,
                       color=PALETTE["edge_import"], lw=1.1, rad=-0.18)

    legend_items = [
        mpatches.Patch(color=PALETTE["class_box"],    label="Class"),
        mpatches.Patch(color=PALETTE["func_public"],  label="Public function"),
        mpatches.Patch(color=PALETTE["func_private"], label="Private function"),
        mpatches.Patch(color=PALETTE["external_box"], label="External lib"),
        mpatches.Patch(color=PALETTE["cc_low"],       label="Low complexity"),
        mpatches.Patch(color=PALETTE["cc_med"],       label="Medium complexity"),
        mpatches.Patch(color=PALETTE["cc_high"],      label="High complexity"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=4,
              bbox_to_anchor=(0.44, 0.0), framealpha=0.18,
              facecolor="#1a1a2e", edgecolor="#333355",
              labelcolor=PALETTE["text_light"], fontsize=7.5)

    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys
    test_files = [Path(p) for p in sys.argv[1:]]
    if not test_files:
        test_files = sorted(Path(".").glob("*.py"))[:5]
    if test_files:
        img, _ = generate_architecture(test_files, "test_architecture.png")
        print(f"Saved: {img}")
    else:
        print("Usage: python architecture.py file1.py file2.py ...")