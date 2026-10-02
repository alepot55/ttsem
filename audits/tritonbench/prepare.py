"""Write the files TritonBench runs for its published answers: the answer, the separator, the test.

    python audits/tritonbench/prepare.py TRITONBENCH_ROOT PRED_DIR

TRITONBENCH_ROOT is a clone of github.com/thunlp/TritonBench. Every answer file it publishes
(`LLM_generated/Bench_{G,T}_*/<model>.jsonl`) becomes one folder PRED_DIR/{G,T}/<model> with one
file per answer, named after its task. The mapping from an answer to its task and the extraction
of the code are the benchmark's own (`EVAL/eval_G/0_call_acc.py`, `EVAL/eval_T/0_call_acc.py`),
reimplemented so that nothing of theirs is executed: for G the answer is reduced to its imports and
function definitions, for T it is the last fenced block. An answer the benchmark cannot map to a
task is left out, as the benchmark leaves it out. Only the standard library is needed.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

SEP = "#" * 146
STOP = ("<|im_end|>", "<|EOT|>")


def clear_code(code: str) -> str:
    """The benchmark's own extraction of a T answer: what follows its last ```python fence."""
    if "```python" in code:
        code = code.split("```python")[-1]
        for token in STOP:
            code = code.replace(token, "")
    return code


def imports_and_functions(code: str) -> str:
    """The benchmark's own extraction of a G answer: its imports, then its function definitions,
    found anywhere in the module and written back with `ast.unparse`."""
    code = clear_code(code)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    imports, functions = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef):
            functions.append(ast.unparse(node))
    return "\n".join(imports) + "\n\n" + "\n".join(functions)


def test_of(folder: Path, name: str) -> str:
    """The test of the task `name`: what follows the separator in the benchmark's own file."""
    return (folder / name).read_text(encoding="utf-8").split(SEP)[-1]


def files_g(items: list[dict[str, str]], stats: list[dict[str, str]]) -> list[str | None]:
    """The task file of each G answer, matched as the benchmark matches it: by the reference
    code the answer's label holds, else by the instruction it was given; each task once."""
    left = list(stats)
    files: list[str | None] = []
    for item in items:
        label = item["label"]
        for token in STOP:
            label = label.replace(token, "")
        found = next((s for s in left if label in s["output"]), None)
        if found is None:
            text = item.get("instruction") or item.get("prompt") or ""
            found = next(
                (s for s in left if s["comp_instru"] in text or s["simp_instru"] in text), None
            )
        if found is not None:
            left.remove(found)
        files.append(found["file"] if found else None)
    return files


def files_t(items: list[dict[str, str]], stats: list[dict[str, str]]) -> list[str | None]:
    """The task file of each T answer: the one task whose description holds the functional
    description of the answer's prompt."""
    files: list[str | None] = []
    for item in items:
        text = next(iter(item.values()))
        if "Functional Description: " not in text:
            files.append(None)
            continue
        func = text.split("Functional Description: ")[-1].split("Wrapper Entry Information:")[0]
        func = func.replace("\n", "")
        hits = [s["file"] for s in stats if func in s["description"].replace("\n", "")]
        files.append(hits[0] if len(hits) == 1 else None)
    return files


def write_model(root: Path, path: Path, out: Path) -> tuple[str, int, int]:
    """Write the runnable files of one answer file (`Bench_{G,T}_*/<model>.jsonl`) under
    OUT/{G,T}/<model>. Returns the benchmark and how many answers of how many were mapped."""
    bench = "G" if "Bench_G" in path.parent.name else "T"
    stats_name = "TritonBench_G_v1.json" if bench == "G" else "TritonBench_T_v1.jsonl"
    # the T file is named .jsonl but holds one JSON array, as the G file does
    stats = json.loads((root / "data" / stats_name).read_text(encoding="utf-8"))
    lines = path.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    gold = root / "data" / f"TritonBench_{bench}_v1"
    files = files_g(items, stats) if bench == "G" else files_t(items, stats)
    extract = imports_and_functions if bench == "G" else clear_code
    folder = out / bench / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for item, name in zip(items, files, strict=True):
        if name is None:
            continue
        body = extract(item["predict"]) + "\n" + SEP + "\n" + test_of(gold, name)
        target = folder / name
        # a file already there is left alone when it is the same: a judge may be running it
        if not target.exists() or target.read_text(encoding="utf-8") != body:
            target.write_text(body, encoding="utf-8")
        written += 1
    return bench, written, len(items)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.stderr.write(__doc__ or "")
        return 2
    root, out = Path(args[0]), Path(args[1])
    for path in sorted((root / "LLM_generated").glob("*/*.jsonl")):
        bench, written, total = write_model(root, path, out)
        print(f"{bench} {path.stem}: {written} of {total} answers mapped to a task")
    return 0


if __name__ == "__main__":
    sys.exit(main())
