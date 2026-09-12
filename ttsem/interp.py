"""The interpreter: walk the IR, bind SSA names, run the ops.

One program instance at a time. `run_grid` runs every program id in the grid in order, which
is a legal schedule of a Triton launch as long as the kernel does not depend on the order in
which programs reach the same address; the only ops that could tell the difference are the
atomics, and they see the same set of updates.

Names are bound in a stack of scopes, one per entered region, so that a combine region's block
arguments do not leak out of `tt.reduce` and a loop body rebinds its induction variable each
iteration. A value defined in a dominating block stays visible to the blocks it dominates.
"""

from __future__ import annotations

import itertools

from ttsem.ir_types import Module, Op, Region
from ttsem.memory import Memory
from ttsem.ops import OPS, Branch, Condition, Yield
from ttsem.values import Unsupported, Value


class Interp:
    def __init__(
        self,
        module: Module,
        memory: Memory,
        num_programs: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        self.module = module
        self.memory = memory
        self.num_programs = tuple(num_programs)
        self.program_id: tuple[int, int, int] = (0, 0, 0)
        self.unsupported: set[str] = set()
        self.output: list[str] = []
        self.scopes: list[dict[str, Value]] = []

    # ---------------------------------------------------------------- entry points

    def run(
        self,
        fn: str,
        args: list[Value],
        program_id: tuple[int, int, int] = (0, 0, 0),
    ) -> list[Value]:
        if fn not in self.module.funcs:
            raise KeyError(f"no function {fn!r}; module has {sorted(self.module.funcs)}")
        func = self.module.funcs[fn]
        self.program_id = tuple(program_id)
        self.scopes = []
        return self.run_region(func.regions[0], list(args))

    def run_grid(self, fn: str, args: list[Value]) -> None:
        nx, ny, nz = self.num_programs
        for z, y, x in itertools.product(range(nz), range(ny), range(nx)):
            self.run(fn, args, (x, y, z))

    # ---------------------------------------------------------------- regions

    def run_region(self, region: Region, args: list[Value]) -> list[Value]:
        return self._execute(region, args)[1]

    def run_condition(self, region: Region, args: list[Value]) -> tuple[bool, list[Value]]:
        """Run a `scf.while` "before" region, which ends in `scf.condition`."""
        return self._execute(region, args)

    def _execute(self, region: Region, args: list[Value]) -> tuple[bool, list[Value]]:
        if not region.blocks:
            return True, []
        labelled = {b.label: b for b in region.blocks if b.label is not None}
        block, incoming = region.blocks[0], list(args)
        self.scopes.append({})
        try:
            while True:
                self._bind_block_args(block.args, incoming)
                try:
                    for op in block.ops:
                        self.eval_op(op)
                except Yield as signal:
                    return True, signal.values
                except Condition as signal:
                    return signal.cond, signal.values
                except Branch as signal:
                    if signal.label not in labelled:
                        raise KeyError(f"no block {signal.label} in this region") from None
                    block, incoming = labelled[signal.label], signal.args
                    continue
                return True, []
        finally:
            self.scopes.pop()

    def _bind_block_args(self, params: list[tuple[str, object]], values: list[Value]) -> None:
        if len(params) != len(values):
            raise ValueError(f"block takes {len(params)} arguments, got {len(values)}")
        for (name, _ty), value in zip(params, values, strict=True):
            self.scopes[-1][name] = value

    # ---------------------------------------------------------------- ops

    def eval_op(self, op: Op) -> None:
        handler = OPS.get(op.name)
        if handler is None:
            self.unsupported.add(op.name)
            raise Unsupported(op.name, op.text or "")
        try:
            results = handler(self, op, [self.value(n) for n in op.operands])
        except Exception as e:
            if not isinstance(e, Unsupported) and type(e).__module__ != "ops":
                e.add_note(f"while evaluating {(op.text or op.name)[:300]}")
            raise
        self.bind_results(op, results)

    def value(self, name: str) -> Value:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise KeyError(f"{name} is not bound")

    def bind_results(self, op: Op, results: list[Value]) -> None:
        names = op.results
        if not names:
            return
        if len(names) == len(results):
            for name, value in zip(names, results, strict=True):
                self.scopes[-1][name] = value
            return
        if len(names) == 1 and len(results) > 1:
            for i, value in enumerate(results):
                self.scopes[-1][f"{names[0]}#{i}"] = value
            return
        raise ValueError(f"{op.name} defines {names} but produced {len(results)} values")
