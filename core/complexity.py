import ast
from dataclasses import dataclass


@dataclass
class Complexity:
    name         : str
    score        : int
    lines        : int
    params       : int
    has_recursion: bool
    level        : str


def analyze(file_path) -> list:
    results = []
    try:
        source = open(file_path,
                      encoding="utf-8",
                      errors="ignore").read()
        tree   = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                score = 1
                for n in ast.walk(node):
                    if isinstance(n, (ast.If, ast.For,
                                      ast.While,
                                      ast.ExceptHandler,
                                      ast.With)):
                        score += 1
                    elif isinstance(n, ast.BoolOp):
                        score += len(n.values) - 1

                recursive = any(
                    isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Name) and
                    n.func.id == node.name
                    for n in ast.walk(node)
                )

                if score <= 5:
                    level = "🟢 Low"
                elif score <= 10:
                    level = "🟡 Medium"
                else:
                    level = "🔴 High"

                results.append(Complexity(
                    name          = node.name,
                    score         = score,
                    lines         = (node.end_lineno
                                     - node.lineno),
                    params        = len(node.args.args),
                    has_recursion = recursive,
                    level         = level
                ))
    except Exception:
        pass
    return results