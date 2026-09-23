"""``python -m ttsem demo``: the per-pass validator on one real Triton bug, end to end.

The point of an executable semantics is that a compiler bug stops being "the answer is wrong"
and becomes "this pass changed the meaning of the program". The demo shows that on
triton#11519: ``tritongpu-fuse-nested-loops`` trusted an ``llvm.assume`` sitting on the branch
the loop nest cannot reach, so a nest whose inner loop runs zero times executed its body
anyway. peterbell10 fixed it upstream in triton#11521.

Two ways to see it, and they differ only in where the stages come from.

``python -m ttsem demo`` replays the recording committed in ``examples/``: the verdict the
semantics reached on every stage of one launch, stage by stage, as recorded on Triton main
``e1f944a7a``. Nothing but ``numpy`` is needed, because nothing is compiled: this is
bookkeeping over verdicts somebody else's machine already produced, and the demo says so.
``culprit_of`` below is the library's own function, run here on the recorded verdicts, so the
pass the demo names is derived and not a string in this file.

``python -m ttsem demo --live`` does the whole thing here and now: it compiles
``examples/e15_repro.py`` through the fake driver with the IR dump on, runs the module that
every pass in the pipeline receives through :class:`ttsem.interp.Interp` on the launch's own
inputs, and compares each one with the launch's reference output. That needs the Triton wheel
(``pip install 'ttsem[triton]'``) but still no GPU, and takes tens of seconds.

Why the recording carries verdicts and not IR: a stage dump of this pipeline is some megabytes
of MLIR per launch, and the recording is meant to be read in a diff. Reproducing the verdicts
from scratch is what ``--live`` is for.

Only this one bug is demonstrated, on purpose. ``--live --program`` takes any of the other
reproducers in ``examples/``, but most of them are not this shape: triton#11601 shows up as a
stage that *reads* memory the source never read (``PassResult.extra_reads``, which the report
prints and this summary does not), and the witnesses of the interpreter defects are not
per-pass at all. A demo that showed those under a heading about output elements would be
lying about what it measured.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = _ROOT / "examples"

DEFAULT_RECORDING = EXAMPLES / "e15_11519_main_e1f944a7a.json"
LIVE_PROGRAM = EXAMPLES / "e15_repro.py"

# What is publicly known about the pass the verdicts point at. Keyed by the pipeline name the
# recording yields, so that a recording naming some other pass gets no citation rather than
# this one: the demo must not be able to print a bug number the evidence does not support.
CITATIONS = {
    "tritongpu-fuse-nested-loops": "triton#11519, fixed upstream in #11521",
}

VERDICT_WIDTH = 12


@dataclass(frozen=True)
class Stage:
    """One pass's dump and the verdict the semantics reached on it."""

    pass_name: str
    verdict: str
    n_diff: int
    unsupported: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class Group:
    """A run of consecutive stages that all reached the same verdict."""

    first: int  # 1-based index of the first stage in the run
    last: int
    verdict: str
    n_diff: int
    first_name: str
    last_name: str
    note: str


def short_name(pass_name: str) -> str:
    """The pipeline name of a pass: what ``triton-opt`` would be asked for.

    ``"Before TritonGPUFuseNestedLoops: tritongpu-fuse-nested-loops"`` is
    ``"tritongpu-fuse-nested-loops"``; the option braces MLIR prints go, the ``(step 1)`` an
    intermediate dump carries stays, because it distinguishes two stages of one pass.
    """
    name = re.sub(r"\{[^}]*\}", "", pass_name)
    head, colon, tail = name.partition(": ")
    name = tail if colon else head
    return " ".join(name.split())


def group_stages(stages: Iterable[Stage]) -> list[Group]:
    """Consecutive stages with the same verdict and the same number of differing elements,
    collapsed into one run. A pipeline is 80-odd stages of which four or five say anything."""
    groups: list[Group] = []
    index = 1
    for (verdict, n_diff), run in itertools.groupby(stages, key=lambda s: (s.verdict, s.n_diff)):
        members = list(run)
        note = ""
        if verdict == "unsupported":
            note = ", ".join(members[0].unsupported) or members[0].message
        groups.append(
            Group(
                first=index,
                last=index + len(members) - 1,
                verdict=verdict,
                n_diff=n_diff,
                first_name=short_name(members[0].pass_name),
                last_name=short_name(members[-1].pass_name),
                note=note.splitlines()[0][:60] if note else "",
            )
        )
        index += len(members)
    return groups


def format_group(group: Group) -> str:
    """One line: the stages it covers, the verdict, the differing elements, the passes."""
    span = f"{group.first}-{group.last}" if group.last > group.first else f"{group.first}"
    # `n_diff` is how many elements of the launch's output buffers the semantics read
    # differently from the reference; a stage that was never compared has none.
    elements = "n_diff=-" if group.n_diff < 0 else f"n_diff={group.n_diff}"
    passes = group.first_name
    if group.last_name != group.first_name:
        passes += f" .. {group.last_name}"
    if group.note:
        passes += f"  ({group.note})"
    return f"  stages {span:>7}  {group.verdict:<{VERDICT_WIDTH}} {elements:<12} {passes}"


def load_recording(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """``(program name, launches)`` from a report ``python -m ttsem.validate --json`` wrote."""
    data = json.loads(path.read_text())
    launches = data.get("launches") or []
    if not launches:
        raise ValueError(f"{path} records no launch")
    return str(data.get("program", path.stem)), launches


def stages_of(launch: dict[str, Any]) -> list[Stage]:
    """The recorded per-pass verdicts of one launch, in pipeline order.

    This is the adapter the recording needs. ``validate_stages`` builds a ``Report`` out of
    ``PassResult`` objects by running each stage's module; the recording is the far side of
    that, the ``PassResult`` list already serialised by ``validate.main``, so the demo turns
    the rows back into the same shape and hands them to the same ``culprit_of``.
    """
    return [
        Stage(
            pass_name=row["pass_name"],
            verdict=row["verdict"],
            n_diff=int(row.get("n_diff", -1)),
            unsupported=tuple(row.get("unsupported") or ()),
            message=str(row.get("message", "")),
        )
        for row in launch["passes"]
    ]


def culprit(stages: list[Stage]) -> str | None:
    """The pass the verdicts point at, from the library's own two readings of a stage list.

    ``culprit_of`` is the reading when the reference is a device that computed the wrong
    answer: the IR disagrees with it up to some pass and agrees after it.
    ``changed_by_index`` is the reading when the reference is the source program: the IR agrees
    up to some pass and disagrees after it. A recording made against a GPU gives the first, the
    ``--live`` run on this machine gives the second, and on this bug they name the same pass.
    """
    from ttsem.validate import PassResult, changed_by_index, culprit_of

    results = [PassResult(s.pass_name, s.verdict, s.n_diff, list(s.unsupported)) for s in stages]
    found = culprit_of(results)
    if found is None:
        index = changed_by_index(results)
        found = results[index].pass_name if index is not None else None
    return None if found is None else short_name(found)


def verdict_line(name: str | None) -> str:
    """The demo's last line: the pass, and what is publicly known about it."""
    if name is None:
        return "no pass changes the meaning: every stage agrees with the reference"
    citation = CITATIONS.get(name)
    suffix = f" ({citation})" if citation else ""
    return f"first pass that changes the meaning: {name}{suffix}"


def _report_stages(report: Any) -> Iterator[Stage]:
    for r in report.passes:
        yield Stage(r.pass_name, r.verdict, r.n_diff, tuple(r.unsupported), r.message)


def run_recorded(path: Path, out: Any) -> int:
    """Replay a committed recording. Needs numpy and nothing else."""
    if not path.exists():
        print(f"no recording at {path}", file=out)
        return 1
    program, launches = load_recording(path)
    print(f"replaying a recorded per-pass validation of {program}", file=out)
    print(f"  recording   {path}", file=out)
    print("  reference   a real launch, recorded on Triton main e1f944a7a", file=out)
    print(
        "\n  The verdicts below are the ones the semantics reached when it ran each stage's\n"
        "  module on that launch's inputs. The recording keeps the verdicts, not the modules,\n"
        "  so nothing is re-executed here; the pass named at the end is computed from them by\n"
        "  ttsem.validate. Run `python -m ttsem demo --live` to redo the whole thing locally.\n",
        file=out,
    )
    named: str | None = None
    for launch in launches:
        stages = stages_of(launch)
        print(
            f"  launch {launch.get('launch_id', 0)} ({launch.get('fn_name', '?')}), "
            f"{len(stages)} stages",
            file=out,
        )
        for group in group_stages(stages):
            print(format_group(group), file=out)
        named = culprit(stages)
        print("", file=out)
    print(verdict_line(named), file=out)
    return 0


def run_live(program: Path, out: Any) -> int:
    """Produce the stages here: compile the kernel with the dump on, run every one of them."""
    try:
        import triton  # noqa: F401
    except ImportError:
        print(
            "--live needs the Triton wheel, which is not installed here.\n"
            "It compiles the kernel (through the fake driver, so no GPU is needed) to get the\n"
            "module every pass receives. Install it with:\n"
            "    pip install 'ttsem[triton]'\n"
            "The default `python -m ttsem demo` replays a committed recording instead.",
            file=out,
        )
        return 1

    from ttsem import harness
    from ttsem.validate import validate

    if not program.exists():
        print(f"no program at {program}", file=out)
        return 1

    print(f"validating {program.name} pass by pass, on this machine", file=out)
    print("  device      cpu (Triton's own interpreter is the reference, no GPU needed)", file=out)
    print(
        "  the whole pipeline is compiled with the IR dump on, and every module it prints is\n"
        "  run on this launch's inputs: tens of seconds\n",
        file=out,
    )

    records = harness.capture_launches(program, "cpu")
    if not records:
        print(f"{program.name} made no kernel launch", file=out)
        return 1

    named: str | None = None
    for record in records:
        report = validate(program, record, device="cpu")
        stages = list(_report_stages(report))
        if not stages:
            print(f"  launch {record.launch_id}: no stage was dumped", file=out)
            continue
        print(f"  launch {record.launch_id} ({record.fn_name}), {len(stages)} stages", file=out)
        for group in group_stages(stages):
            print(format_group(group), file=out)
        named = culprit(stages)
        print("", file=out)
    print(verdict_line(named), file=out)
    return 0


def main(argv: list[str] | None = None, out: Any = None) -> int:
    out = out if out is not None else sys.stdout
    parser = argparse.ArgumentParser(
        prog="python -m ttsem demo",
        description="The per-pass validator on triton#11519, recorded or live.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="compile the kernel here and validate every stage (needs the Triton wheel)",
    )
    parser.add_argument(
        "--recording",
        type=Path,
        default=DEFAULT_RECORDING,
        help="the report to replay, as `python -m ttsem.validate --json` writes one",
    )
    parser.add_argument(
        "--program",
        type=Path,
        default=LIVE_PROGRAM,
        help="the program --live validates: any script with a main() taking --device and --out",
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="races in the kernels torch.compile generates, found without a GPU (needs torch)",
    )
    args, rest = parser.parse_known_args(argv)
    if args.torch:
        from ttsem.torch_demo import main as torch_main

        return torch_main(rest, out)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.live:
        return run_live(args.program, out)
    return run_recorded(args.recording, out)


if __name__ == "__main__":
    raise SystemExit(main())
