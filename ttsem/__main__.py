"""``python -m ttsem``: what the package can do, and which module does it."""

from __future__ import annotations

import sys

from ttsem import __version__

USAGE = f"""ttsem {__version__} - an executable semantics of Triton's TTIR and TTGIR.

The package is a library first; every entry point below is a module you run with `-m`.

  python -m ttsem demo [--live]
      The whole idea on one real bug (triton#11519): replay the committed recording of a
      per-pass validation, or, with --live and the Triton wheel, redo it here. No GPU.

  python -m ttsem.validate PROGRAM.py [--device cpu|cuda] [--json OUT]
      Compile PROGRAM.py's first launch with the dump on and run every pass's input
      through the semantics. Names the first pass whose output the semantics reads
      differently from the device. Needs Triton.

  python -m ttsem.minimize PROGRAM.py [--stage PASS] [--out small.mlir]
      Shrink that module to the smallest one on which the pass still changes the meaning.

  python -m ttsem.races MODULE.generic.mlir [--strip-barriers]
      Shared-memory races between the warps of a CTA, from one execution of the module.
      Needs no Triton and no GPU.

  python -m ttsem.check_fixtures [--dir DIR] [--pattern GLOB]
      Level-2 layout checks over a directory of generic-form TTGIR dumps. No Triton.

  python -m ttsem.bot summary RUN/ | diff PREVIOUS/ CURRENT/
      Bookkeeping over the JSONL a validated test run writes.

  pytest -p ttsem.pytest_ttsem ...            validate every launch of a test run
  python -m ttsem.inductor_trace --out DIR SCRIPT.py
                                              the same for torch.compile's kernels
  python -m ttsem.validate_corpus --programs DIR --results OUT
  python -m ttsem.pass_stress --programs DIR  which passes a corpus actually wakes
  python -m ttsem.dump_corpus --programs DIR --out DIR

In Python:

  from ttsem import parse, Interp, Memory        # level 1, numpy only
  from ttsem import check_module                 # level 2, layouts
  from ttsem import detect                       # level 3, races
  from ttsem import validate_stages, culprit_of  # the per-pass validator, needs Triton
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "demo":
        from ttsem.demo import main as demo_main

        return demo_main(argv[1:])
    if argv:
        sys.stderr.write(f"unknown command {argv[0]!r}\n\n")
        sys.stderr.write(USAGE)
        return 2
    sys.stdout.write(USAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
