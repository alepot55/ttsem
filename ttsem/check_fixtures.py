"""Run the level-2 layout checks over generic-form TTGIR dumps.

    python -m ttsem.check_fixtures [--dir DIR] [--pattern GLOB] [--verbose]

Every check implemented today reads types, not values, so a whole dump can be
checked without running it: the script parses each file, reads `ttg.num-warps`
and `ttg.threads-per-warp` off the module, and walks every op through
`layout_checks.check`. It prints, per file, how many ops carried a check, the
violations, and the encodings the port could not build. A gap is a hole in
coverage and is reported as one, never counted as a pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ttsem import mlir
from ttsem.interp2 import check_module, launch_config
from ttsem.layout_checks import LayoutViolation

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FileReport:
    path: Path
    num_warps: int | None
    threads_per_warp: int | None
    violations: list[LayoutViolation]
    gaps: dict[str, str]
    checked: Counter[str]


def check_file(path: Path) -> FileReport:
    """Parse one dump and run the layout checks over it."""
    text = path.read_text()
    num_warps, threads_per_warp = launch_config(text)
    violations, gaps, checked = check_module(mlir.parse(text), num_warps, threads_per_warp)
    return FileReport(path, num_warps, threads_per_warp, violations, gaps, checked)


def report(result: FileReport, verbose: bool) -> None:
    print(
        f"{result.path.name}: num-warps={result.num_warps} "
        f"threads-per-warp={result.threads_per_warp} "
        f"checked={sum(result.checked.values())} ops "
        f"({len(result.checked)} kinds), {len(result.violations)} violations, "
        f"{len(result.gaps)} gaps"
    )
    if verbose and result.checked:
        counts = ", ".join(f"{name} x{n}" for name, n in sorted(result.checked.items()))
        print(f"    {counts}")
    for violation in result.violations:
        print(f"    VIOLATION {violation}")
    for encoding, reason in result.gaps.items():
        print(f"    GAP {encoding[:90]}")
        print(f"        {reason.splitlines()[0][:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=HERE / "fixtures")
    parser.add_argument("--pattern", default="*.ttgir.generic")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no file matches {args.pattern!r} under {args.dir}")

    violations = gaps = 0
    covered: Counter[str] = Counter()
    for path in paths:
        result = check_file(path)
        report(result, args.verbose)
        violations += len(result.violations)
        gaps += len(result.gaps)
        covered.update(result.checked)
    exercised = ", ".join(f"{name} x{n}" for name, n in sorted(covered.items()))
    print(
        f"{len(paths)} files, {sum(covered.values())} ops checked, "
        f"{violations} violations, {gaps} gaps"
    )
    print(f"checks exercised on: {exercised or 'nothing'}")


if __name__ == "__main__":
    main()
