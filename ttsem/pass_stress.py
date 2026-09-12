"""Which passes a corpus actually wakes: for every program, the stages whose module differs
from the one before it, aggregated per pass.

A validator that never sees a pass change anything has not tested that pass. The same
question asked of Triton's pipeline: the dump gives every stage, so a pass "fires" on a
program when the module it hands on is not the one it received (locations stripped).

    python -m ttsem.pass_stress --programs DIR [--device cpu|cuda] [--limit N] [--jobs J]
        [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ttsem import minimize
from ttsem import validate


def changed_stages(program: Path, device: str, cc: int) -> list[tuple[str, bool]] | None:
    """``(pass_name, fired)`` per stage: `fired` when the *next* stage's module differs from
    this one's, i.e. the pass named by this stage changed the program. The last stage has
    nothing after it and is not counted."""
    try:
        stages = validate.split_dump(validate.capture_dump(program, device, cc))
    except Exception:
        return None
    if len(stages) < 2:
        return None
    # a pass fires when the module it hands to the next pass differs from the one it received;
    # the intermediate dumps some passes print are not pass boundaries (and come in another
    # printing form), so they are skipped rather than compared
    boundaries = [(name, text) for name, text in stages if "(step " not in name]
    texts = [minimize.strip_locs(text) for _, text in boundaries]
    out = []
    for i in range(len(boundaries) - 1):
        out.append((boundaries[i][0], texts[i] != texts[i + 1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--programs", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--cc", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    programs = sorted(args.programs.glob("*.py"))
    if args.limit:
        programs = programs[: args.limit]
    fired: collections.Counter[str] = collections.Counter()
    seen: collections.Counter[str] = collections.Counter()
    ran = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for result in pool.map(lambda p: changed_stages(p, args.device, args.cc), programs):
            if result is None:
                continue
            ran += 1
            for name, did in result:
                seen[name] += 1
                if did:
                    fired[name] += 1
    rows = sorted(seen, key=lambda n: (-fired[n] / seen[n], n))
    print(f"{ran} of {len(programs)} programs compiled; {len(seen)} stages")
    for name in rows:
        rate = fired[name] / seen[name]
        flag = "asleep" if fired[name] == 0 else ""
        print(f"{rate:6.1%} {fired[name]:>5}/{seen[name]:<5} {name[:70]} {flag}")
    if args.json:
        args.json.write_text(
            json.dumps({n: {"fired": fired[n], "seen": seen[n]} for n in rows}, indent=1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
