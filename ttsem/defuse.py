"""Def-use over a parsed module: which op (or block argument) defines each SSA name, and
which ops use it. Names are resolved lexically, innermost region first, since printed SSA
names repeat across disjoint regions."""

from __future__ import annotations

from dataclasses import dataclass

from ttsem.ir_types import Block, Module, Op, Region


@dataclass(frozen=True)
class Def:
    op: Op | None  # None for a block argument
    block: Block | None  # the block that declares a block argument
    index: int  # result index of `op`, or argument index of `block`


def _walk(region: Region, scope: dict[str, Def], defs: dict[int, dict[str, Def]]) -> None:
    # A region with several blocks is a CFG: a block may use a value another block of the same
    # region defines (dominance, not lexical order, decides validity, and the printer does not
    # sort blocks by it). So every definition of the region is in scope in every block, and
    # within a block the definitions precede their uses as usual.
    region_defs: dict[str, Def] = {}
    for block in region.blocks:
        for i, (name, _) in enumerate(block.args):
            region_defs[name] = Def(None, block, i)
        for op in block.ops:
            for i, name in enumerate(op.results):
                region_defs[name] = Def(op, None, i)
    for block in region.blocks:
        inner = dict(scope)
        if len(region.blocks) > 1:
            inner.update(region_defs)
        for i, (name, _) in enumerate(block.args):
            inner[name] = Def(None, block, i)
        for op in block.ops:
            defs[id(op)] = dict(inner)
            for r in op.regions:
                _walk(r, inner, defs)
            for i, name in enumerate(op.results):
                inner[name] = Def(op, None, i)


def defs(module: Module) -> dict[int, dict[str, Def]]:
    """For every op (keyed by ``id(op)``), the names in scope at that op and their definers."""
    table: dict[int, dict[str, Def]] = {}
    for op in module.ops:
        table[id(op)] = {}
        for r in op.regions:
            _walk(r, {}, table)
    return table


def definer(table: dict[int, dict[str, Def]], op: Op, operand: str) -> Def | None:
    """The definition of ``operand`` as seen from ``op``, or None for an unbound name."""
    return table.get(id(op), {}).get(operand)


def users(module: Module) -> dict[int, list[tuple[Op, int]]]:
    """For every defining op (keyed by ``id(op)``), the ops that use one of its results and the
    operand position they use it at."""
    table = defs(module)
    out: dict[int, list[tuple[Op, int]]] = {}

    def visit(region: Region) -> None:
        for block in region.blocks:
            for op in block.ops:
                for i, name in enumerate(op.operands):
                    d = table.get(id(op), {}).get(name)
                    if d is not None and d.op is not None:
                        out.setdefault(id(d.op), []).append((op, i))
                for r in op.regions:
                    visit(r)

    for op in module.ops:
        for r in op.regions:
            visit(r)
    return out


def parents(module: Module) -> dict[int, tuple[Op | None, Block | None]]:
    """For every op (keyed by ``id(op)``), the op that owns the region it sits in (None at the
    top level) and the block it sits in."""
    out: dict[int, tuple[Op | None, Block | None]] = {}

    def visit(region: Region, owner: Op) -> None:
        for block in region.blocks:
            for op in block.ops:
                out[id(op)] = (owner, block)
                for r in op.regions:
                    visit(r, op)

    for op in module.ops:
        out[id(op)] = (None, None)
        for r in op.regions:
            visit(r, op)
    return out


def ancestors(table: dict[int, tuple[Op | None, Block | None]], op: Op) -> list[Op]:
    """The enclosing ops of ``op``, innermost first (its ``scf.if``, then its ``scf.for``, then
    the function), read off ``parents(module)``."""
    chain: list[Op] = []
    cur = table.get(id(op), (None, None))[0]
    while cur is not None:
        chain.append(cur)
        cur = table.get(id(cur), (None, None))[0]
    return chain
