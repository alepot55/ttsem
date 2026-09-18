"""The smallest module on which one pass still changes the meaning.

Given the IR a pass receives, a way to run that pass, and a launch record for the inputs, the
minimiser deletes operations, whole regions first and single lines last, while the module the
pass produces still means something different from the module it was given, under the level-1
semantics. No device is needed: the property is a disagreement between two interpretations of
the same inputs, and the semantics judges both sides (its calibration is in the results notes).

    python -m ttsem.minimize program.py            # the pass is the report's `changed_by`
    python -m ttsem.minimize program.py --stage FuseNestedLoops --out small.mlir
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ttsem import harness
from ttsem import mlir
import numpy as np
from ttsem import validate
from ttsem.defuse import definer, defs

RunPass = Callable[[str], str]  # generic module text -> generic module text after the pass
Interesting = Callable[[str, str], bool]  # (module, pass(module)) -> the property still holds

# Never deleted: the ops that hold the module together, and the terminators of its blocks.
_STRUCTURE = {"builtin.module", "tt.func", "func.func", "llvm.func"}
_TERMINATORS = {
    "tt.return",
    "func.return",
    "scf.yield",
    "scf.condition",
    "cf.br",
    "cf.cond_br",
    "tt.reduce.return",
    "tt.scan.return",
    "ttg.warp_yield",
    "ttg.warp_return",
    "ttg.partition_return",
    "llvm.return",
}
_OP_LINE = re.compile(r'^\s*(?:%[\w#:]+(?:,\s*%[\w#:]+)*\s*=\s*)?"([\w.]+)"\(')


def op_spans(lines: list[str]) -> list[tuple[int, int, str]]:
    """``(first, last, op_name)`` for every operation of a generic-form module, one per op.

    The generic printer with local scope puts one op per line, and an op with regions closes
    where its braces balance; the braces inside an attribute balance on their own line."""
    spans = []
    for i, line in enumerate(lines):
        m = _OP_LINE.match(line)
        if m is None:
            continue
        depth = 0
        end = i
        for j in range(i, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth <= 0:
                end = j
                break
        spans.append((i, end, m.group(1)))
    return spans


_RESULT = re.compile(r"\)\s*->\s*(.+?)(?:\s+loc\(.*)?$")
_SCALAR = re.compile(r"^(?:[su]?i(\d+)|(f16|bf16|f32|f64))$")


def _zero_of(ty: str) -> str | None:
    """The generic ``arith.constant`` attribute for the zero of a numeric type, or ``None`` for
    a type that has no constant (pointers, tokens, descriptors, tuples)."""
    ty = ty.strip()
    if ty.startswith("tensor<"):
        inner = ty[len("tensor<") : -1]
        elem = inner.split(",")[0].rsplit("x", 1)[-1].strip()
        lit = _zero_of(elem)
        return None if lit is None else f"dense<{lit.split(' : ')[0]}> : {ty}"
    m = _SCALAR.match(ty)
    if m is None:
        return None
    if m.group(1) == "1":
        return f"false : {ty}"
    if m.group(1):
        return f"0 : {ty}"
    return f"0.000000e+00 : {ty}"


def constant_for(lines: list[str], first: int, last: int) -> str | None:
    """The line that defines the single result of the op at ``lines[first:last+1]`` as a zero
    constant of its type, or ``None`` when the op has no single numeric result."""
    head = lines[first].lstrip()
    if not head.startswith("%") or " = " not in head:
        return None
    name = head.split(" = ", 1)[0].strip()
    if ":" in name or "," in name:  # several results
        return None
    m = _RESULT.search(lines[last].rstrip())
    if m is None:
        return None
    ty = m.group(1).strip()
    if ty.startswith("("):
        return None
    lit = _zero_of(ty)
    if lit is None:
        return None
    indent = lines[first][: len(lines[first]) - len(head)]
    return f'{indent}{name} = "arith.constant"() <{{value = {lit}}}> : () -> {ty}'


def well_formed(text: str) -> bool:
    """Parses, and every operand of every op has a definition in scope."""
    try:
        module = mlir.parse(text)
    except Exception:
        return False
    table = defs(module)

    def walk(op_list: list[Any]) -> bool:
        for op in op_list:
            if any(definer(table, op, name) is None for name in op.operands):
                return False
            for region in op.regions:
                for block in region.blocks:
                    if not walk(block.ops):
                        return False
        return True

    return all(
        walk(block.ops)
        for fn in module.funcs.values()
        for region in fn.regions
        for block in region.blocks
    )


def meaning_changed(record: harness.LaunchRecord, before: str, after: str) -> bool:
    """True when ``before`` and ``after`` leave different memory behind on the record's inputs,
    beyond the float policy; False as well when either does not run to the end."""
    a = harness.final_state(record, before)
    if isinstance(a, harness.Comparison):
        return False
    b = harness.final_state(record, after)
    if isinstance(b, harness.Comparison):
        return False
    scan = harness.scan_inexact(mlir.parse(before))
    for name, x in a.items():
        y = b.get(name)
        if y is None:
            return True
        policy = harness.float_policy(harness._elem_spelling(record, name), scan)
        cmp = harness.compare(x.reshape(-1), y.reshape(-1), policy=policy)
        if not cmp["equal"] and not cmp.get("approx"):
            return True
    return False


def reads_changed(record: harness.LaunchRecord, before: str, after: str) -> bool:
    """True when ``after`` loads a byte address that ``before`` never loads on the record's
    inputs: the observable of a load hoisted above a loop that does not run (#11601), where the
    outputs are identical and only the read-set moves. False when either does not run."""
    a = harness.execute(record, before, trace_reads=True)
    b = harness.execute(record, after, trace_reads=True)
    if a.failure is not None or b.failure is not None:
        return False
    extra = np.setdiff1d(b.memory.read_bytes(), a.memory.read_bytes())
    return bool(extra.size)


@dataclasses.dataclass
class Reduction:
    before: str  # the smallest module found that the pass still changes
    after: str  # what the pass makes of it
    trials: int
    lines_from: int
    lines_to: int


def failure_signature(message: str) -> str:
    """What a failure is, without where it is: the first line of the message with positions
    and numbers blanked, so that the same diagnostic on a smaller module compares equal and a
    different one (the verifier on a candidate the reduction broke) does not."""
    first = next((ln for ln in message.splitlines() if ln.strip()), "")
    return re.sub(r"\d+", "N", first).strip()


def reduce(
    record: harness.LaunchRecord | None,
    module_text: str,
    run_pass: RunPass,
    interesting: Interesting | None = None,
    max_trials: int = 20_000,
    crash: bool = False,
    verify: Callable[[str], Any] | None = None,
) -> Reduction:
    """Greedy reduction to a fixpoint, largest spans first, two transforms per op: delete it,
    or replace it by the zero constant of its result type (which frees everything it used).

    A candidate survives when it still parses with every operand defined, the pass accepts it,
    and ``interesting(candidate, pass(candidate))`` holds. In crash mode the property is that
    the pass fails *the way it failed on the module as given* (`failure_signature`), on a
    candidate that ``verify`` accepts: a module the reduction made invalid fails too, and is
    not the crash."""
    if interesting is None and not crash:
        if record is None:
            raise ValueError("a record is needed to compare meanings; use crash=True without one")

        def interesting(before: str, after: str) -> bool:
            return meaning_changed(record, before, after)

    signature: list[str] = []  # of the failure on the module as given (crash mode)

    def holds(text: str) -> tuple[bool, str]:
        """(the property holds on `text`, the pass's output or '')."""
        try:
            out = run_pass(text)
        except Exception as exc:
            if not crash:
                return False, ""
            # in crash mode a failing pass is the property itself, if it is the same failure
            if not signature:
                signature.append(failure_signature(str(exc)))
            elif failure_signature(str(exc)) != signature[0]:
                return False, ""
            if verify is not None:
                try:
                    verify(text)
                except Exception:
                    return False, ""
            return True, ""
        if crash:
            return False, out
        assert interesting is not None
        return interesting(text, out), out

    ok, after = holds(module_text)
    if not ok:
        raise ValueError(
            "the pass does not fail on the module as given"
            if crash
            else "the pass does not change the meaning of the module as given"
        )
    text = module_text
    trials = 0
    progress = True
    while progress and trials < max_trials:
        progress = False
        lines = text.split("\n")
        spans = [s for s in op_spans(lines) if s[2] not in _TERMINATORS | _STRUCTURE]
        spans.sort(key=lambda s: s[1] - s[0], reverse=True)
        candidates: list[str] = []
        for first, last, _ in spans:
            candidates.append("\n".join(lines[:first] + lines[last + 1 :]))
            if last > first or '"arith.constant"' not in lines[first]:
                const = constant_for(lines, first, last)
                if const is not None:
                    candidates.append("\n".join(lines[:first] + [const] + lines[last + 1 :]))
        for candidate in candidates:
            trials += 1
            if not well_formed(candidate):
                continue
            ok, candidate_after = holds(candidate)
            if ok:
                text, after = candidate, candidate_after
                progress = True
                break
    return Reduction(text, after, trials, module_text.count("\n") + 1, text.count("\n") + 1)


def strip_locs(text: str) -> str:
    """The module without its ``loc(...)`` trailers: the same program, and a witness that no
    longer carries the paths of the machine it was found on. Locations nest (``callsite``,
    ``fused``), so the parenthesis is matched by hand, skipping over quoted strings."""
    # alias definitions (`#loc3 = loc("file":1:2)`) go with the trailers that reference them
    text = "\n".join(ln for ln in text.split("\n") if not ln.startswith("#loc"))
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find(" loc(", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len(" loc(")
        depth = 1
        while k < n and depth:
            c = text[k]
            if c == '"':
                k = text.find('"', k + 1)
                if k < 0:
                    k = n
                    break
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            k += 1
        i = k
    return "".join(out)


# --- running one pass by the name the dump gives it -------------------------------------

_PREFIXES = (
    "tritongpu-",
    "triton-nvidia-gpu-",
    "triton-nvidia-",
    "triton-amdgpu-",
    "triton-",
    "nvgpu-",
    "convert-",
)
_ALIASES = {
    "canonicalize": "canonicalizer",
    "inline": "inliner",
    "loop-invariant-code-motion": "licm",
}


def pass_argument(stage_name: str) -> str:
    """The argument name in parentheses: ``tritongpu-fuse-nested-loops`` out of
    ``Before TritonGPUFuseNestedLoops (tritongpu-fuse-nested-loops)``."""
    m = re.search(r"\(([\w-]+)", stage_name)
    if m is None:
        if re.fullmatch(r"[\w-]+", stage_name):
            return stage_name  # a bare pass argument, as `triton-opt --<this>` takes it
        raise ValueError(f"no pass argument in stage name {stage_name!r}")
    return m.group(1)


def binding_for(arg: str) -> Callable[..., None]:
    """The wheel's ``add_<pass>`` binding for a pass argument name."""
    from triton._C.libtriton import passes

    modules: list[Any] = [passes.ttgpuir, passes.ttir, passes.common, passes.convert, passes.llvmir]
    for extra in ("gluon",):
        if hasattr(passes, extra):
            modules.append(getattr(passes, extra))
    try:
        from triton._C.libtriton import nvidia

        modules.append(nvidia.passes.ttnvgpuir)
    except Exception:
        pass
    names = [arg] + [arg[len(p) :] for p in _PREFIXES if arg.startswith(p)]
    names += [_ALIASES[n] for n in names if n in _ALIASES]
    for n in names:
        attr = "add_" + n.replace("-", "_")
        for m in modules:
            if hasattr(m, attr):
                return getattr(m, attr)
    raise LookupError(f"no pass binding for {arg!r} (tried {names})")


def pass_runner(
    stage_name: str,
    triton_opt: str | None = None,
    num_stages: int = 3,
    pre: tuple[str, ...] = (),
) -> RunPass:
    """Run the one pass a dump stage names, through ``triton-opt`` when given and otherwise
    through the wheel's pass bindings, printing the result in the generic form.

    ``pre`` names passes to run first (``canonicalize``, ``cse``): a crash minimiser without
    them converges on modules with dead loops and unused values that the pipeline would have
    cleaned before the pass, which no source program can reach."""
    arg = pass_argument(stage_name)
    if triton_opt:
        flags = tuple(f"--{p}" for p in pre) + (f"--{arg}",)

        def run_opt(text: str) -> str:
            return mlir._run_triton_opt(text, triton_opt, flags)

        return run_opt

    from triton._C.libtriton import ir

    add = binding_for(arg)
    mlir.enable_generic_printing()
    ctx = ir.context()
    ir.load_dialects(ctx)
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import make_backend

        make_backend(GPUTarget("cuda", harness.DEFAULT_CC, 32)).load_dialects(ctx)
    except Exception:
        pass

    def run(text: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            mod = ir.parse_mlir_module(path, ctx)
            mod.context = ctx
            pm = ir.pass_manager(ctx)
            pm.enable_debug()
            for name in pre:
                binding_for(name)(pm)
            try:
                add(pm)
            except TypeError:
                add(pm, num_stages, False)  # the pipeliner takes its options positionally
            try:
                pm.run(mod, "ttsem")  # newer wheels name the pipeline for their timing report
            except TypeError:
                pm.run(mod)
            return str(mod)
        finally:
            os.unlink(path)

    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("program", type=Path, nargs="?", default=None)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    ap.add_argument("--cc", type=int, default=harness.DEFAULT_CC)
    ap.add_argument("--triton-opt", default=None)
    ap.add_argument("--launch", type=int, default=0, help="which launch of the program")
    ap.add_argument("--stage", default=None, help="a substring of the stage to minimise for")
    ap.add_argument("--out", type=Path, default=None, help="write the reduced module here")
    ap.add_argument(
        "--reads",
        action="store_true",
        help="the property is a new read address, not a different output (hoisted loads)",
    )
    ap.add_argument(
        "--crash",
        action="store_true",
        help="the property is that the pass fails (asserts, verifier error); needs --triton-opt "
        "so that the failure is a subprocess's and not ours, and --module or --stage",
    )
    ap.add_argument(
        "--module",
        type=Path,
        default=None,
        help="with --crash: minimise this module text directly, no program or launch needed",
    )
    ap.add_argument(
        "--pre",
        default="canonicalize,cse",
        help="passes run before the one under test on every trial (crash mode); '' for none",
    )
    args = ap.parse_args()
    pre = tuple(p for p in args.pre.split(",") if p)

    if args.crash and args.module is not None:
        if not args.triton_opt or not args.stage:
            print(
                "--crash --module needs --triton-opt and --stage (the pass name)", file=sys.stderr
            )
            return 2
        before = mlir.to_generic(args.module.read_text(), args.triton_opt)
        runner = pass_runner(args.stage, args.triton_opt, pre=pre)

        def verifies(text: str) -> None:
            mlir._run_triton_opt(text, args.triton_opt)

        reduction = reduce(None, before, runner, crash=True, verify=verifies)
        lines = f"{reduction.lines_from} -> {reduction.lines_to} lines"
        print(f"{args.stage}: {lines}, {reduction.trials} trials")
        witness = strip_locs(reduction.before)
        if args.out:
            args.out.write_text(witness)
            print(f"wrote {args.out}")
        else:
            print(witness)
        return 0

    records = harness.capture_launches(args.program, args.device, args.cc)
    if not records:
        print("no launches recorded", file=sys.stderr)
        return 2
    record = records[args.launch]
    stages = validate.split_dump(validate.capture_dump(args.program, args.device, args.cc))
    if args.stage:
        idx = next((i for i, (n, _) in enumerate(stages) if args.stage in n), None)
        if idx is None:
            print(f"no stage named like {args.stage!r}", file=sys.stderr)
            return 2
    else:
        report = validate.validate_stages(record, stages, args.triton_opt, args.program.name)
        idx = validate.changed_by_index(report.passes)
        if idx is None and args.reads:
            idx = next((i for i, r in enumerate(report.passes) if r.extra_reads), None)
            idx = None if idx is None or idx == 0 else idx - 1  # the stage that receives the IR
        if idx is None:
            print(f"no stage changes the meaning (culprit={report.culprit!r})", file=sys.stderr)
            return 1
    name, text = stages[idx]
    before = mlir.to_generic(text, args.triton_opt)
    interesting: Interesting | None = None
    if args.reads:

        def interesting(b: str, a: str) -> bool:
            return reads_changed(record, b, a)

    reduction = reduce(record, before, pass_runner(name, args.triton_opt), interesting)
    print(
        f"{name}: {reduction.lines_from} -> {reduction.lines_to} lines, {reduction.trials} trials"
    )
    witness = strip_locs(reduction.before)
    if args.out:
        args.out.write_text(witness)
        print(f"wrote {args.out}")
    else:
        print(witness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
