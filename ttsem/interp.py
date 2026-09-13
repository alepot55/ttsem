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
import threading
from collections.abc import Callable

from ttsem.ir_types import Module, Op, Region
from ttsem.memory import Memory
from ttsem.ops import OPS, Branch, Condition, Yield
from ttsem.values import Barrier, Unsupported, Value


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
        self._tls = threading.local()
        self.scopes = []
        self.scheduler: Scheduler | None = None
        self.mbarriers: dict[int, Barrier] = {}  # by the address of the memdesc

    # The scope stack is per thread: the partitions of a `ttg.warp_specialize` each run in a
    # thread of their own (see `Scheduler`), over a copy of the enclosing stack.
    @property
    def scopes(self) -> list[dict[str, Value]]:
        stack = getattr(self._tls, "scopes", None)
        if stack is None:
            stack = self._tls.scopes = []
        return stack

    @scopes.setter
    def scopes(self, stack: list[dict[str, Value]]) -> None:
        self._tls.scopes = stack

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

    # ---------------------------------------------------------------- partitions

    def run_scheduled(
        self,
        default: Region,
        default_args: list[Value],
        tasks: list[tuple[str, Region, list[Value]]],
    ) -> list[Value]:
        """Run `default` here and every task region in a thread of its own, one at a time: a
        thread runs until it blocks on a barrier phase or an aref slot, then hands over."""
        if not tasks:
            return self.run_region(default, default_args)
        if self.scheduler is not None:
            raise Unsupported("ttg.warp_specialize", "nested inside a partition")
        sched = Scheduler(self)
        self.scheduler = sched
        try:
            sched.start(
                [(name, self._region_task(region, args)) for name, region, args in tasks],
                list(self.scopes),
            )
            results = self.run_region(default, default_args)
            sched.join()
        finally:
            sched.abort()
            self.scheduler = None
        return results

    def _region_task(self, region: Region, args: list[Value]) -> Callable[[], None]:
        def task() -> None:
            self.run_region(region, list(args))

        return task

    def block_until(self, ready: Callable[[], bool], what: str) -> None:
        """Wait for `ready`: hand the turn to another partition until it holds. Outside a
        `warp_specialize` nothing else can make it hold, so a wait that fails is a deadlock."""
        if self.scheduler is not None:
            self.scheduler.block_until(ready, what)
        elif not ready():
            raise Unsupported(what, "waits on a phase that nothing before it completes")

    def progress(self) -> None:
        """Something a wait may depend on changed: an arrival, a copy, a slot released."""
        if self.scheduler is not None:
            self.scheduler.version += 1

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


class _Aborted(Exception):
    """Unwinds a partition thread once another one has failed."""


class Scheduler:
    """The partitions of one `ttg.warp_specialize`, run cooperatively.

    Exactly one member runs at a time (the baton is `current`); a member gives the baton up only
    when a wait of its fails or when it finishes, so the interleaving is deterministic. A
    deadlock is every live member blocked with nothing having changed (`version`) since it
    last looked: the program itself would hang on the device, or the model lacks an arrival.
    """

    def __init__(self, interp: Interp) -> None:
        self.interp = interp
        self.cv = threading.Condition()
        self.current = 0  # index of the member holding the baton; 0 is the caller
        self.version = 0
        self.names: list[str] = ["default partition"]
        self.threads: list[threading.Thread | None] = [None]
        self.done: list[bool] = [False]
        self.blocked_at: list[tuple[int, str] | None] = [None]
        self.error: BaseException | None = None
        self._index: dict[int, int] = {threading.get_ident(): 0}

    # -- lifecycle
    def start(self, tasks: list[tuple[str, Callable[[], None]]], scopes: list[dict]) -> None:
        for name, task in tasks:
            index = len(self.names)
            self.names.append(name)
            self.done.append(False)
            self.blocked_at.append(None)
            thread = threading.Thread(
                target=self._run_member, args=(index, task, scopes), name=name, daemon=True
            )
            self.threads.append(thread)
            with self.cv:
                self._index[-index] = index  # placeholder until the thread reports its ident
            thread.start()

    def _run_member(self, index: int, task: Callable[[], None], scopes: list[dict]) -> None:
        with self.cv:
            self._index[threading.get_ident()] = index
            self.cv.wait_for(lambda: self.current == index or self.error is not None)
        try:
            if self.error is None:
                self.interp.scopes = list(scopes)
                task()
        except _Aborted:
            pass
        except BaseException as exc:  # noqa: BLE001 - reported to the caller by `join`
            with self.cv:
                if self.error is None:
                    self.error = exc
        finally:
            with self.cv:
                self.done[index] = True
                self.version += 1
                self._pass_baton(index)

    def join(self) -> None:
        self.block_until(lambda: all(self.done[1:]), "warp_specialize")
        if self.error is not None:
            raise self.error

    def abort(self) -> None:
        """Wake every member so its thread can end (after an error, or a caller unwinding)."""
        with self.cv:
            if self.error is None and not all(self.done[1:]):
                self.error = _Aborted("the caller left the warp_specialize")
            self.cv.notify_all()
        for thread in self.threads[1:]:
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)

    # -- waiting
    def _me(self) -> int:
        return self._index[threading.get_ident()]

    def block_until(self, ready: Callable[[], bool], what: str) -> None:
        me = self._me()
        while True:
            with self.cv:
                if self.error is not None and me != 0:
                    raise _Aborted()
                if ready():
                    self.blocked_at[me] = None
                    return
                self.blocked_at[me] = (self.version, what)
                live = [i for i in range(len(self.names)) if not self.done[i]]
                stuck = [
                    i for i in live if self.blocked_at[i] and self.blocked_at[i][0] == self.version
                ]
                if len(stuck) == len(live):
                    waits = "; ".join(f"{self.names[i]}: {self.blocked_at[i][1]}" for i in live)
                    raise Unsupported(
                        "warp_specialize", f"deadlock, every partition waits ({waits})"
                    )
                self._pass_baton(me)
                self.cv.wait_for(lambda: self.current == me or self.error is not None)

    def _pass_baton(self, me: int) -> None:
        """Hand the turn to the next member that is not done (the caller holds `cv`)."""
        n = len(self.names)
        for step in range(1, n + 1):
            candidate = (me + step) % n
            if not self.done[candidate]:
                self.current = candidate
                break
        self.cv.notify_all()
