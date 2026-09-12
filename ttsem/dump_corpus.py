"""Write the generic-form IR of the first launch of every program in a directory.

    python -m ttsem.dump_corpus --programs DIR --out DIR --stage ttgir
        --triton-opt PATH [--jobs N]

One file `<program>.<stage>.generic` per program, produced in a subprocess per program so that
a program that crashes the compiler does not take the others with it. The output directory is
what `python -m ttsem.check_fixtures --dir` walks, so level-2 checks run over a whole corpus.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ONE = """
from ttsem import harness
rec = harness.capture_launches(__import__("pathlib").Path({program!r}), "cuda")[0]
import torch
major, minor = torch.cuda.get_device_capability()
target = harness.GPUTarget("cuda", major * 10 + minor, 32)
text = harness.ir_for_launch(rec, {stage!r}, target, triton_opt={topt!r})
open({out!r}, "w").write(text)
"""


def _one(program: Path, out_dir: Path, stage: str, topt: str) -> str:
    out = out_dir / f"{program.stem}.{stage}.generic"
    code = _ONE.format(program=str(program), stage=stage, topt=topt, out=str(out))
    try:
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    return "ok" if r.returncode == 0 and out.exists() else "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", default="ttgir")
    ap.add_argument("--triton-opt", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    programs = sorted(a.programs.glob("*.py"))
    with ThreadPoolExecutor(a.jobs) as pool:
        verdicts = list(pool.map(lambda p: _one(p, a.out, a.stage, a.triton_opt), programs))
    from collections import Counter

    print(f"{len(programs)} programs: {dict(Counter(verdicts))}")


if __name__ == "__main__":
    main()
