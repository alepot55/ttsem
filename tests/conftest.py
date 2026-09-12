"""Builders for hand-made IR, so the ops can be tested without going through the parser.

Everything here constructs the dataclasses of `ttsem.ir_types` directly. When a test says
`op("arith.addi", ["i32", "i32"], "i32")` it is writing the same thing the generic form would
print, minus the syntax.
"""

from __future__ import annotations

import numpy as np

from ttsem.interp import Interp
from ttsem.ir_types import Block, Module, Op, Region, Type
from ttsem.memory import Memory

FLOAT_WIDTH = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}


def ty(spelling: str) -> Type:
    """A scalar type from its printed spelling: `i32`, `f32`, `bf16`, `f8E4M3FN`, `index`."""
    if spelling == "index":
        return Type("index", width=64, name=spelling)
    if spelling.startswith("i"):
        return Type("int", width=int(spelling[1:]), name=spelling)
    width = FLOAT_WIDTH.get(spelling, 8)
    return Type("float", width=width, name=spelling)


def tensor(shape: tuple[int, ...], elem: str | Type) -> Type:
    inner = ty(elem) if isinstance(elem, str) else elem
    return Type("tensor", shape=tuple(shape), elem=inner, name="tensor")


def ptr(elem: str | Type) -> Type:
    inner = ty(elem) if isinstance(elem, str) else elem
    return Type("ptr", elem=inner, width=64, name="ptr")


def memdesc(shape: tuple[int, ...], elem: str | Type) -> Type:
    inner = ty(elem) if isinstance(elem, str) else elem
    return Type("memdesc", shape=tuple(shape), elem=inner, name="memdesc")


def tensordesc(shape: tuple[int, ...], elem: str | Type) -> Type:
    inner = ty(elem) if isinstance(elem, str) else elem
    return Type("tensordesc", shape=tuple(shape), elem=inner, name="tensordesc")


def op(
    name: str,
    operand_types: list[Type] | None = None,
    result_types: list[Type] | Type | None = None,
    *,
    attrs: dict[str, object] | None = None,
    regions: list[Region] | None = None,
    successors: list[str] | None = None,
    results: list[str] | None = None,
    operands: list[str] | None = None,
) -> Op:
    """An op whose operands default to `%a0, %a1, ...` and results to `%r0, %r1, ...`."""
    rts = [result_types] if isinstance(result_types, Type) else list(result_types or [])
    ots = list(operand_types or [])
    return Op(
        name=name,
        results=results if results is not None else [f"%r{i}" for i in range(len(rts))],
        result_types=rts,
        operands=operands if operands is not None else [f"%a{i}" for i in range(len(ots))],
        operand_types=ots,
        attrs=dict(attrs or {}),
        regions=list(regions or []),
        successors=list(successors or []),
    )


def block(args: list[tuple[str, Type]], ops: list[Op], label: str | None = None) -> Block:
    return Block(label=label, args=args, ops=ops)


def region(*blocks: Block) -> Region:
    return Region(blocks=list(blocks))


def func(name: str, params: list[tuple[str, Type]], ops: list[Op]) -> Module:
    entry = Op(name="tt.func", attrs={"sym_name": name}, regions=[region(block(params, ops))])
    return Module(funcs={name: entry}, ops=[entry])


def interp(module: Module | None = None, memory: Memory | None = None, **kw: object) -> Interp:
    grid = kw.get("num_programs", (1, 1, 1))
    assert isinstance(grid, tuple)
    return Interp(module or Module(), memory or Memory(), grid)


def evaluate(
    target: Op,
    args: list[object] | None = None,
    memory: Memory | None = None,
    **kw: object,
) -> list[object]:
    """Evaluate one op on the given operand values and return its results."""
    machine = interp(memory=memory, **kw)
    pid = kw.get("program_id", (0, 0, 0))
    assert isinstance(pid, tuple)
    machine.program_id = pid
    machine.scopes = [dict(zip(target.operands, args or [], strict=True))]
    machine.eval_op(target)
    return [machine.value(n) for n in target.results]


def one(target: Op, args: list[object] | None = None, **kw: object) -> np.ndarray:
    result = evaluate(target, args, **kw)[0]
    assert isinstance(result, np.ndarray)
    return result


def buffer(memory: Memory, base: int, array: np.ndarray) -> np.ndarray:
    memory.register(base, array)
    return array
