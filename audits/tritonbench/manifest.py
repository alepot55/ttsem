"""The manifests of the TritonBench audit, for `python -m ttsem judge --tritonbench --manifest`.

    python audits/tritonbench/manifest.py PRED_DIR TRITONBENCH_ROOT OUT_DIR

PRED_DIR is what `prepare.py` wrote (PRED_DIR/{G,T}/<model>/<task>.py: the answer, the separator,
the task's test). Each file becomes a line `{"id", "task", "answer", "bench", "model"}` of
OUT_DIR/G.jsonl or OUT_DIR/T.jsonl: its task is the benchmark's own file of the same name, its id
`<bench>.<model>.<task>`. OUT_DIR/all.jsonl holds both, the whole audit in one command. The judge
reads an answer file only up to its separator and puts the task's test after it, as `prepare.py`
did. Paths are absolute, so the manifests can live anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def lines_of(pred: Path, root: Path, bench: str) -> list[str]:
    data = root / "data" / f"TritonBench_{bench}_v1"
    lines = []
    for answer in sorted((pred / bench).glob("*/*.py")):
        model = answer.parent.name
        item = {
            "id": f"{bench}.{model}.{answer.stem}",
            "task": str(data / answer.name),
            "answer": str(answer),
            "bench": bench,
            "model": model,
        }
        lines.append(json.dumps(item))
    return lines


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        sys.stderr.write(__doc__ or "")
        return 2
    pred, root, out = (Path(a).resolve() for a in args)
    out.mkdir(parents=True, exist_ok=True)
    both: list[str] = []
    for bench in ("G", "T"):
        lines = lines_of(pred, root, bench)
        (out / f"{bench}.jsonl").write_text("".join(f"{line}\n" for line in lines))
        print(f"{bench}: {len(lines)} answers")
        both += lines
    (out / "all.jsonl").write_text("".join(f"{line}\n" for line in both))
    return 0


if __name__ == "__main__":
    sys.exit(main())
