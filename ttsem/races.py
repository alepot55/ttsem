"""Level 3: shared-memory races between the warps of a CTA, found by executing the program.

One execution (level 1's) decides values and control flow. This interpreter replays it and, at
every shared-memory op, computes the byte set each warp touches from the layouts (level 2's
linear layouts, the view's origin inside its allocation, the allocation's offset) and logs it
with the warp's epoch, the number of `ttg.barrier` ops executed so far in its group. Two
accesses from different warps to overlapping bytes, one of them a write, in the same epoch,
are a race: no barrier separates them. Asynchronous copies are pending from issue to
completion, and anything that touches their bytes in between races with them. The design notes
for "Level 3" have the model; the calibration corpus is Membar's own lit file.

    python -m ttsem.races membar_after.generic.mlir      # every function, races per function
    python -m ttsem.races membar_after.generic.mlir --strip-barriers  # what the barriers prevent
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ttsem import mlir
from ttsem import ops as ops_mod
from ttsem.interp2 import LayoutInterp, launch_config
from ttsem.ir_types import Module, Op, Type
from ttsem.layouts import EncodingParseError, drop_pipelining_dims, parse_encoding, to_linear_layout
from ttsem.linear_layout import LayoutError, LinearLayout
from ttsem.memory import Memory
from ttsem.values import Descriptor, MemDesc, Unsupported, Value, to_numpy

ASYNC = "async"  # the copy engine and the tensor core: an agent that is no warp
BARRIERS = ("ttg.barrier", "gpu.barrier")


@dataclasses.dataclass
class Access:
    group: int
    agent: str  # "w3" or ASYNC
    kind: str  # "r" | "w"
    addrs: np.ndarray  # sorted unique byte addresses in the shared address space
    epoch: int  # -1 once an asynchronous access has completed and is ordered with everyone
    op: str
    pending: str | None = None  # the completion key while in flight
    instance: int = 0  # which execution of which op produced this access
    exact: bool = True  # False when no layout was known and the bytes are the whole view
    time: int = 0  # the group's event counter when the access happened (level 3.2)

    def overlaps(self, other: Access) -> int:
        return int(np.intersect1d(self.addrs, other.addrs, assume_unique=True).size)


@dataclasses.dataclass
class Race:
    first: Access
    second: Access
    overlap: int  # bytes in common


@dataclasses.dataclass
class Report3:
    function: str
    accesses: int
    barriers: int
    races: list[Race]
    pairs: dict[tuple[str, str], int]  # (op text a, op text b) -> number of racing warp pairs
    status: str  # "ok" | "unsupported: ..." | "error: ..."
    gaps: dict[str, str]  # op name -> why its byte set was the whole view


# ----------------------------------------------------------------------------- layouts


def apply_many(layout: LinearLayout, ins: dict[str, Any]) -> dict[str, np.ndarray]:
    """`layout.apply` over arrays: every input dim as an int or an int array, broadcast."""
    names = list(layout.out_dims)
    shape = np.broadcast(*[np.asarray(v) for v in ins.values()]).shape
    out = {n: np.zeros(shape, np.int64) for n in names}
    for in_dim, value in ins.items():
        value = np.asarray(value, np.int64)
        for bit, basis in enumerate(layout.bases.get(in_dim, [])):
            mask = (value >> bit) & 1
            for j, n in enumerate(names):
                if basis[j]:
                    out[n] ^= mask * basis[j]
    return out


def _whole_view_coords(shape: tuple[int, ...]) -> list[np.ndarray]:
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij") if shape else []
    return [g.reshape(-1) for g in grids]


# ----------------------------------------------------------------------------- interpreter


class RaceInterp(LayoutInterp):
    """`LayoutInterp` plus agents, epochs, byte sets and the race log."""

    def __init__(
        self,
        module: Module,
        memory: Memory,
        num_programs: tuple[int, int, int] = (1, 1, 1),
        num_warps: int | None = None,
        threads_per_warp: int | None = None,
    ) -> None:
        super().__init__(module, memory, num_programs, num_warps or 4, threads_per_warp or 32)
        self.group = 0
        self.group_warps: dict[int, int] = {0: self.num_warps or 4}
        self.epoch: dict[int, int] = {0: 0}
        self.barriers = 0
        self.implicit_barriers = 0  # scratch-using ops, whose lowering carries a bar.sync
        self.accesses: list[Access] = []
        self.by_epoch: dict[tuple[int, int], list[Access]] = {}  # (group, epoch) -> accesses
        self.seen: set[tuple[int, int, str, str, str, int]] = set()  # a loop repeats its accesses
        self.pending: list[Access] = []
        self.races: list[Race] = []
        self.commit_open: list[Access] = []
        self.commit_groups: list[list[Access]] = []
        self.gaps: dict[str, str] = {}
        self.frame = 0  # the callee's allocations sit at the call site's `allocation.offset`
        # level 3.2: events that order groups. `clock[g]` counts a group's events; arrivals and
        # waits name a barrier and carry (group, time); `barrier_count` is init_barrier's count.
        self.clock: dict[int, int] = {0: 0}
        self.arrivals: dict[str, list[tuple[int, int]]] = {}
        self.waits: dict[str, list[tuple[int, int]]] = {}
        self.barrier_count: dict[str, int] = {}
        self.cross_races: list[Race] = []
        self.steps = 0
        self.step_budget = 50_000
        self.instance = 0  # one per hooked op execution; every warp's access of it shares it
        self._next_alloc = 0
        self._shared_inverse: dict[
            tuple[str, tuple[int, ...]], tuple[LinearLayout, int] | None
        ] = {}

    # ------------------------------------------------------------------ dispatch

    def block_until(self, ready, what: str) -> None:  # type: ignore[override]
        """Level 3 orders the partitions with its own epochs and vector clocks and runs them
        one after the other; the level-1 wait must not block or object here."""
        return None

    def eval_op(self, op: Op) -> None:
        self.steps += 1
        if self.steps > self.step_budget:
            raise Unsupported(
                "steps", f"more than {self.step_budget} ops: a loop this input never leaves"
            )
        if op.name in BARRIERS:
            self.epoch[self.group] += 1
            self.barriers += 1
            self.tick()
            return
        if op.name == "ttg.warp_specialize":
            self._warp_specialize(op)
            return
        if (
            "allocation.offset" in op.attrs
            and op.name not in ("ttg.local_alloc", "tt.call")
            and not op.name.startswith("ttg.warp_specialize")
        ):
            # An op that lowers through a scratch buffer of its own (convert_layout, reduce,
            # scan, histogram, gather) emits a bar.sync between its scratch store and load: a
            # CTA-wide barrier for everyone, which Membar relies on and so does this model.
            self.epoch[self.group] += 1
            self.implicit_barriers += 1
        if op.name == "tt.call":
            outer = self.frame
            self.frame += _attr_int(op.attrs, "allocation.offset") or 0
            try:
                super().eval_op(op)
            finally:
                self.frame = outer
            return
        args = [self.value(name) for name in op.operands]
        super().eval_op(op)
        hook = _HOOKS.get(op.name)
        if hook is not None:
            self.instance += 1
            hook(self, op, args, list(self._results))

    def _warp_specialize(self, op: Op) -> None:
        sizes = op.attrs.get("partitionNumWarps")
        sizes_list = _ints(sizes) if sizes is not None else []
        partitions = op.regions[1].blocks[0].ops[0] if op.regions[1].blocks else None
        has_partitions = (
            partitions is not None and partitions.name == "ttg.warp_specialize.partitions"
        )
        outer = self.group
        ws = f"ws:{self.instance}:{id(op)}"
        if has_partitions:
            # the outer group's past, up to here, is visible to every partition: an arrival
            # before the default region runs, which every partition waits on at its start
            self.arrive(f"{ws}:entry")
            self.barrier_count[f"{ws}:entry"] = 1
            self.barrier_count[f"{ws}:exit"] = len(partitions.regions)
        results = self.run_region(op.regions[0], [])
        self.bind_results(op, results)
        if not has_partitions:
            return
        args = [self.value(name) for name in partitions.operands] if partitions.operands else []
        for k, region in enumerate(partitions.regions):
            self.group = len(self.group_warps)
            self.group_warps[self.group] = sizes_list[k] if k < len(sizes_list) else 4
            self.epoch[self.group] = 0
            self.clock[self.group] = 0
            self.wait(f"{ws}:entry")
            self.run_region(region, list(args))
            self.arrive(f"{ws}:exit")
        self.group = outer
        self.epoch[outer] += 1  # every warp of the CTA leaves the op together (#11324)
        self.wait(f"{ws}:exit")
        # A partition's outstanding copies are waited by their consumers before the op ends;
        # what is still pending here is ordered with everything after by the exit sync.
        for acc in [p for p in self.pending if p.group != outer]:
            self.complete(acc.pending or "", None)

    # ------------------------------------------------------------------ allocations

    def alloc_attrs(self, op: Op | None, ty: Type) -> dict[str, Any]:
        """The bookkeeping a `MemDesc` carries: where its allocation starts, its shape and
        layout, and where this view sits inside it."""
        shape = tuple(ty.shape or ())
        elem = ty.elem if ty.elem is not None else ty
        elem_bytes = np.dtype(to_numpy(ty)).itemsize
        base = None
        if op is not None:
            raw = op.attrs.get("allocation.offset")
            if raw is not None:  # a partitioned allocation lists one offset per partition
                offsets = _ints(raw)
                base = offsets[0] if offsets else None
        if base is None:  # no allocation pass ran: give every allocation its own range
            base = self._next_alloc
            self._next_alloc += int(np.prod(shape or (1,))) * elem_bytes + 1024
        return {
            "alloc_base": base + (self.frame if op is not None else 0),
            "alloc_shape": shape,
            "enc": ty.encoding or "",
            "elem_bytes": elem_bytes,
            "elem": elem,
            "origin": [0] * len(shape),
            "drop": 0,
            "opaque": False,
        }

    def shared_inverse(self, attrs: dict[str, Any]) -> tuple[LinearLayout, int] | None:
        """`(dims -> offset, lead)`: the inverse of the allocation's shared layout over its
        layout-ranked suffix, and the number of leading pipelining dims it does not describe;
        stage `s` of a multi-buffered allocation starts `s * prod(suffix)` elements in."""
        key = (attrs["enc"], attrs["alloc_shape"])
        if key in self._shared_inverse:
            return self._shared_inverse[key]
        found: tuple[LinearLayout, int] | None
        try:
            enc = parse_encoding(attrs["enc"])
            shape = list(attrs["alloc_shape"])
            suffix = drop_pipelining_dims(shape, enc)
            layout = to_linear_layout(enc, suffix, self.num_warps, self.threads_per_warp)
            # A CGA-split allocation (`CGALayout` with a non-zero basis) covers the tensor
            # with the offset *and* the CTA index: keep both, so that the inverse is total
            # and every CTA's bytes land in an address space of their own (`view_bytes`).
            ins = ["offset"]
            if layout.has_in_dim(K_BLOCK) and not layout.sublayout_is_zero(
                [K_BLOCK], layout.out_dim_names()
            ):
                ins.append(K_BLOCK)
            layout = layout.sublayout(ins, layout.out_dim_names())
            inv = layout.invert() if layout.is_invertible() else layout.pseudoinvert()
            found = (inv, len(shape) - len(suffix))
        except (EncodingParseError, LayoutError, KeyError, ValueError) as exc:
            self.gaps.setdefault(attrs["enc"][:80], f"shared layout: {exc}")
            found = None
        self._shared_inverse[key] = found
        return found

    def view_bytes(self, md: MemDesc, coords: list[np.ndarray] | None) -> np.ndarray:
        """Byte addresses of the view's elements at `coords` (arrays over the view's dims), or
        of the whole view when `coords` is None; the whole allocation when the view is opaque."""
        a = md.attrs
        elem_bytes, base = a["elem_bytes"], a["alloc_base"]
        if a.get("opaque") or not a.get("alloc_shape"):
            n = int(np.prod(a.get("alloc_shape") or md.data.shape or (1,)))
            return np.arange(base, base + n * elem_bytes, dtype=np.int64)
        if coords is None:
            coords = _whole_view_coords(tuple(md.data.shape))
        found = self.shared_inverse(a)
        rank = len(a["alloc_shape"])
        drop, origin = a["drop"], a["origin"]
        if found is None or len(coords) != rank - drop:
            n = int(np.prod(md.data.shape or (1,)))
            inside = all(o < s for o, s in zip(origin, a["alloc_shape"], strict=False))
            first = base
            if inside:
                first += int(np.ravel_multi_index(tuple(origin), a["alloc_shape"])) * elem_bytes
            return np.arange(first, first + n * elem_bytes, dtype=np.int64)
        inv, lead = found
        size = coords[0].shape if coords else ()
        full = [np.full(size, origin[i], np.int64) for i in range(drop)]
        full += [np.asarray(c, np.int64) + origin[drop + i] for i, c in enumerate(coords)]
        suffix_shape = a["alloc_shape"][lead:]
        outs = apply_many(inv, {f"dim{i}": full[lead + i] for i in range(rank - lead)})
        offset = outs["offset"]
        stage = np.zeros_like(offset)
        for i in range(lead):  # row-major over the pipelining dims
            stage = stage * a["alloc_shape"][i] + full[i]
        offset = offset + stage * int(np.prod(suffix_shape or (1,)))
        addrs = base + offset * elem_bytes
        if K_BLOCK in outs:  # another CTA's shared memory is another address space
            addrs = addrs + np.asarray(outs[K_BLOCK], np.int64) * CTA_SPACE
        elems = np.unique(addrs.reshape(-1))
        return (elems[:, None] + np.arange(elem_bytes)[None, :]).reshape(-1)

    def warp_coords(self, ty: Type | None, warp: int) -> list[np.ndarray] | None:
        """The coordinates of the elements warp `warp` holds of a distributed tensor."""
        layout = self.layout_of(ty)
        if layout is None or ty is None or ty.shape is None:
            return None
        regs = (
            np.arange(layout.get_in_dim_size("register"))
            if layout.has_in_dim("register")
            else np.zeros(1, int)
        )
        lanes = (
            np.arange(layout.get_in_dim_size("lane"))
            if layout.has_in_dim("lane")
            else np.zeros(1, int)
        )
        nwarps = layout.get_in_dim_size("warp") if layout.has_in_dim("warp") else 1
        ins: dict[str, Any] = {
            "register": regs[:, None],
            "lane": lanes[None, :],
            "warp": warp % nwarps,
            "block": 0,
        }
        ins = {k: v for k, v in ins.items() if layout.has_in_dim(k)}
        outs = apply_many(layout, ins)
        return [outs[f"dim{i}"].reshape(-1) % ty.shape[i] for i in range(len(ty.shape))]

    # ------------------------------------------------------------------ the log

    def tick(self, group: int | None = None) -> int:
        g = self.group if group is None else group
        self.clock[g] = self.clock.get(g, 0) + 1
        return self.clock[g]

    def arrive(self, key: str) -> None:
        self.arrivals.setdefault(key, []).append((self.group, self.tick()))

    def wait(self, key: str) -> None:
        self.waits.setdefault(key, []).append((self.group, self.tick()))

    def log(self, acc: Access) -> None:
        acc.instance = self.instance
        acc.time = self.tick(acc.group)
        for prev in self.pending:  # cross-group ordering goes through mbarriers: level 3.2
            if (
                prev is not acc
                and prev.group == acc.group
                and prev.instance != acc.instance
                and prev.overlaps(acc)
            ):
                self.races.append(Race(prev, acc, prev.overlaps(acc)))
        # A loop body without a barrier repeats the same access every iteration: the second
        # copy can race with nothing the first did not, so it is neither checked nor kept.
        key = (acc.group, acc.epoch, acc.agent, acc.kind, acc.op[:200], _digest(acc.addrs))
        if acc.pending is None and key in self.seen:
            return
        self.seen.add(key)
        if acc.epoch >= 0:
            for prev in self.by_epoch.get((acc.group, acc.epoch), []):
                if (
                    prev.agent != acc.agent
                    and prev.pending is None
                    and prev.instance != acc.instance
                    and ("w" in (prev.kind, acc.kind))
                ):
                    n = prev.overlaps(acc)
                    if n and not self._own_copy(prev, acc):
                        self.races.append(Race(prev, acc, n))
        self.accesses.append(acc)
        self.by_epoch.setdefault((acc.group, acc.epoch), []).append(acc)
        if acc.pending is not None:
            self.pending.append(acc)

    def _own_copy(self, prev: Access, acc: Access) -> bool:
        """True when `acc`'s warp performed the very access `prev` is, on the same bytes: a
        replicated element is held by several warps, and each of them writes or reads its own
        copy, so the other warps' copies of the same op carry the same value and are ordered
        with the warp's own by program order."""
        if not (prev.exact and acc.exact):
            return False  # an unknown layout is not a replicated one
        overlap = np.intersect1d(prev.addrs, acc.addrs, assume_unique=True)
        for other in self.by_epoch.get((prev.group, prev.epoch), []):
            if other.instance == prev.instance and other.agent == acc.agent:
                if np.isin(overlap, other.addrs, assume_unique=True).all():
                    return True
        return False

    def complete(self, key: str, final_epoch: int | None) -> None:
        """Retire pending accesses with `key`: at `final_epoch` as ordinary accesses of their
        agent, or inert (-1) when the completion orders them with every warp."""
        for acc in [p for p in self.pending if p.pending == key]:
            self.pending.remove(acc)
            acc.pending = None
            self.by_epoch.get((acc.group, acc.epoch), []).remove(acc)
            acc.epoch = -1 if final_epoch is None else final_epoch
            self.by_epoch.setdefault((acc.group, acc.epoch), []).append(acc)

    def per_warp(
        self, md: MemDesc, ty: Type | None, kind: str, op: Op, coords_of=None, **more
    ) -> None:
        for w in range(self.group_warps[self.group]):
            coords = self.warp_coords(ty, w)
            if coords is None:
                self.gaps.setdefault(op.name, f"no register layout for {ty.name if ty else ty}")
            if coords is not None and coords_of is not None:
                coords = coords_of(coords, w)
            addrs = self.view_bytes(md, coords)
            acc = Access(
                self.group,
                f"w{w}",
                kind,
                addrs,
                self.epoch[self.group],
                op.text or op.name,
                **more,
            )
            acc.exact = coords is not None
            self.log(acc)

    def whole(self, md: MemDesc, kind: str, op: Op, agent: str = ASYNC, **more) -> None:
        addrs = self.view_bytes(md, None)
        self.log(
            Access(
                self.group, agent, kind, addrs, self.epoch[self.group], op.text or op.name, **more
            )
        )


def _digest(addrs: np.ndarray) -> int:
    return hash(
        (
            int(addrs.size),
            int(addrs[0]) if addrs.size else -1,
            int(addrs[-1]) if addrs.size else -1,
            int(addrs.sum()),
        )
    )


def _ints(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value)
    return [
        int(x) for x in re.findall(r"-?\d+", text.split(":", 1)[-1] if "array<" in text else text)
    ]


def _md_arg(args: list[Value], i: int) -> MemDesc | None:
    return args[i] if i < len(args) and isinstance(args[i], MemDesc) else None


# ----------------------------------------------------------------------------- hooks


def _h_local_alloc(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    md = results[0]
    assert isinstance(md, MemDesc)
    md.attrs.update(ri.alloc_attrs(op, op.result_types[0]))
    if args:
        ri.per_warp(md, op.operand_types[0], "w", op)


def _derive(
    ri: RaceInterp, op: Op, args: list[Value], results: list[Value]
) -> tuple[MemDesc, MemDesc] | None:
    src, dst = _md_arg(args, 0), results[0] if results else None
    if src is None or not isinstance(dst, MemDesc):
        return None
    if "alloc_base" not in src.attrs:
        src.attrs.update(ri.alloc_attrs(None, op.operand_types[0]))
    dst.attrs = dict(src.attrs)
    dst.attrs["origin"] = list(src.attrs["origin"])
    return src, dst


def _h_index(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    pair = _derive(ri, op, args, results)
    if pair is None:
        return
    _, dst = pair
    drop = dst.attrs["drop"]
    if drop < len(dst.attrs["origin"]):
        dst.attrs["origin"][drop] = int(np.asarray(args[1]).reshape(-1)[0])
    dst.attrs["drop"] = drop + 1


def _h_subslice(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    pair = _derive(ri, op, args, results)
    if pair is None:
        return
    _, dst = pair
    offsets = (
        _ints(op.attrs.get("offsets", []))
        if op.name == "ttg.memdesc_subslice"
        else [int(np.asarray(a).reshape(-1)[0]) for a in args[1:]]
    )
    drop, origin = dst.attrs["drop"], dst.attrs["origin"]
    pad = len(offsets) - (len(origin) - drop)
    for i, off in enumerate(offsets):
        if i < pad:
            if drop < len(origin):
                origin[drop] = off
                drop += 1
        elif drop + i - pad < len(origin):
            origin[drop + i - pad] += off
    dst.attrs["drop"] = drop


def _h_regeometry(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    """`memdesc_reinterpret`, `memdesc_trans`, `memdesc_reshape`: the same bytes under a new
    shape, element type or layout. The result's own encoding over its own shape addresses those
    bytes, so the view becomes an allocation of its own that starts where the source view did."""
    pair = _derive(ri, op, args, results)
    if pair is None:
        return
    src, dst = pair
    a = src.attrs
    first = a["alloc_base"]
    if a.get("alloc_shape") and all(
        o < n for o, n in zip(a["origin"], a["alloc_shape"], strict=False)
    ):
        found = ri.shared_inverse(a)
        if found is not None and not a.get("opaque"):
            inv, lead = found
            rank = len(a["alloc_shape"])
            coords = {f"dim{i}": np.array([a["origin"][lead + i]]) for i in range(rank - lead)}
            outs = apply_many(inv, coords)
            off = int(outs["offset"][0])
            if K_BLOCK in outs:
                first += int(outs[K_BLOCK][0]) * CTA_SPACE
            stage = 0
            for i in range(lead):
                stage = stage * a["alloc_shape"][i] + a["origin"][i]
            off += stage * int(np.prod(a["alloc_shape"][lead:] or (1,)))
            first += off * a["elem_bytes"]
    fresh = ri.alloc_attrs(None, op.result_types[0])
    fresh["alloc_base"] = first
    dst.attrs = fresh


def _h_local_load(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    md = _md_arg(args, 0)
    if md is not None:
        ri.per_warp(md, op.result_types[0], "r", op)


def _h_local_store(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    md = _md_arg(args, 1)
    if md is not None:
        ri.per_warp(md, op.operand_types[0], "w", op)


def _h_local_gather(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    md = _md_arg(args, 0)
    if md is None:
        return
    axis = int(str(op.attrs.get("axis", 0)).split(":")[0])
    idx = np.asarray(args[1]).astype(np.int64)

    def gathered(coords: list[np.ndarray], w: int) -> list[np.ndarray]:
        out = list(coords)
        out[axis] = idx[tuple(coords)]
        return out

    ri.per_warp(md, op.result_types[0], "r", op, coords_of=gathered)


def _h_local_scatter(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    md = _md_arg(args, 0)
    if md is None:
        return
    axis = int(str(op.attrs.get("axis", 0)).split(":")[0])
    idx = np.asarray(args[1]).astype(np.int64)

    def scattered(coords: list[np.ndarray], w: int) -> list[np.ndarray]:
        out = list(coords)
        out[axis] = idx[tuple(coords)]
        return out

    ri.per_warp(md, op.operand_types[2], "w", op, coords_of=scattered)


def _h_async_copy(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    md = _md_arg(args, 1)
    if md is None:
        return
    if len(args) > 2 and not isinstance(args[2], MemDesc) and not np.asarray(args[2]).any():
        return  # a mask that is false everywhere: the copy touches nothing
    before = len(ri.accesses)
    ri.per_warp(md, op.operand_types[0], "w", op, pending="cp.async")
    ri.commit_open.extend(ri.accesses[before:])


def _h_commit(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    ri.commit_groups.append(ri.commit_open)
    ri.commit_open = []


def _h_async_wait(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    num = int(str(op.attrs.get("num", 0)).split(":")[0])
    if ri.commit_open:  # an uncommitted copy is committed by the wait
        ri.commit_groups.append(ri.commit_open)
        ri.commit_open = []
    done, ri.commit_groups = (
        ri.commit_groups[: max(len(ri.commit_groups) - num, 0)],
        ri.commit_groups[max(len(ri.commit_groups) - num, 0) :],
    )
    for group in done:
        for acc in group:
            if acc in ri.pending:
                ri.pending.remove(acc)
                acc.pending = None
                ri.by_epoch.get((acc.group, acc.epoch), []).remove(acc)
                acc.epoch = ri.epoch[ri.group]
                ri.by_epoch.setdefault((acc.group, acc.epoch), []).append(acc)


def _barrier_key(md: MemDesc | None) -> str:
    if md is None or "alloc_base" not in md.attrs:
        return "mbarrier"
    return f"mbarrier@{md.attrs['alloc_base']}+{md.attrs['origin']}"


def _predicated_off(op: Op, args: list[Value], n_segments: int) -> bool:
    """True when the op's trailing `pred` operand is present and false: the pipeliner issues a
    copy for every stage and predicates the ones past the trip count off."""
    try:
        segments = ops_mod._segments(op, args, n_segments)
    except Exception:
        return False
    pred = segments[-1]
    return bool(pred) and not bool(np.asarray(pred[0]).reshape(-1)[0])


def _h_tma_load(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    if _predicated_off(op, args, 6):
        return
    mds = [a for a in args if isinstance(a, MemDesc)]
    if not mds:
        return
    barrier = mds[0] if len(mds) > 1 else None
    dest = mds[-1]
    ri.whole(dest, "w", op, pending=_barrier_key(barrier))
    ri.arrive(_barrier_key(barrier))  # the copy's completion is the phase's arrival


def _h_init_barrier(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    ri.barrier_count[_barrier_key(_md_arg(args, 0))] = _attr_int(op.attrs, "count") or 1


def _h_arrive_barrier(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    if _predicated_off(op, args, 2):
        return
    ri.arrive(_barrier_key(_md_arg(args, 0)))


def _h_wait_barrier(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    key = _barrier_key(_md_arg(args, 0))
    ri.complete(key, None)
    ri.wait(key)


def _h_tma_store(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    mds = [a for a in args if isinstance(a, MemDesc)]
    if mds:
        ri.whole(mds[-1], "r", op, pending="tma-store")


def _h_tma_store_wait(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    ri.complete("tma-store", None)


def _h_wgmma(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    is_async = "isAsync = true" in (op.text or "")
    for a in args[:2]:
        if isinstance(a, MemDesc):
            if is_async:
                ri.whole(a, "r", op, pending="wgmma")
            else:
                for w in range(ri.group_warps[ri.group]):
                    ri.whole(a, "r", op, agent=f"w{w}")


def _h_wgmma_wait(ri: RaceInterp, op: Op, args: list[Value], results: list[Value]) -> None:
    ri.complete("wgmma", None)


_HOOKS = {
    "ttg.local_alloc": _h_local_alloc,
    "ttg.memdesc_index": _h_index,
    "ttg.memdesc_subslice": _h_subslice,
    "ttg.memdesc_subview": _h_subslice,
    "ttg.memdesc_reinterpret": _h_regeometry,
    "ttg.memdesc_trans": _h_regeometry,
    "ttg.memdesc_reshape": _h_regeometry,
    "ttg.local_load": _h_local_load,
    "ttg.local_store": _h_local_store,
    "ttg.local_gather": _h_local_gather,
    "ttg.local_scatter": _h_local_scatter,
    "ttg.async_copy_global_to_local": _h_async_copy,
    "ttg.async_commit_group": _h_commit,
    "ttg.async_wait": _h_async_wait,
    "ttng.async_tma_copy_global_to_local": _h_tma_load,
    "ttng.wait_barrier": _h_wait_barrier,
    "ttng.init_barrier": _h_init_barrier,
    "ttng.arrive_barrier": _h_arrive_barrier,
    "ttng.async_tma_copy_local_to_global": _h_tma_store,
    "ttng.async_tma_store_wait": _h_tma_store_wait,
    "ttng.warp_group_dot": _h_wgmma,
    "ttng.warp_group_dot_wait": _h_wgmma_wait,
}


# ----------------------------------------------------------------------------- driving


def synthetic_args(ri: RaceInterp, fn: Op, memory: Memory) -> list[Value]:
    """Arguments for a function nobody launched: buffers for pointers, small integers for
    scalars, zero tensors, and shared views with their own allocation ranges."""
    values: list[Value] = []
    base = 1 << 20
    for _name, ty in fn.regions[0].blocks[0].args:
        t = ty if isinstance(ty, Type) else None
        if t is None:
            values.append(np.zeros((), np.int8))
            continue
        if t.kind == "ptr":
            memory.register(base, np.zeros(1 << 20, np.uint8))  # a megabyte per pointer
            values.append(np.array(base, np.int64))
            base += 1 << 21
        elif t.kind == "memdesc":
            md = MemDesc(np.zeros(t.shape or (), to_numpy(t)), t.elem if t.elem is not None else t)
            md.attrs.update(ri.alloc_attrs(None, t))
            values.append(md)
        elif t.kind == "tensor" and t.elem is not None and t.elem.kind == "ptr":
            # a tensor of pointers: one buffer, the elements laid out in row-major order
            shape = tuple(t.shape or ())
            width = np.dtype(to_numpy(t.elem.elem)).itemsize if t.elem.elem is not None else 4
            memory.register(base, np.zeros(max(int(np.prod(shape or (1,))) * width, 1), np.uint8))
            values.append(
                base + np.arange(int(np.prod(shape or (1,))), dtype=np.int64).reshape(shape) * width
            )
            base += 1 << 20
        elif t.kind == "tensor":
            values.append(np.zeros(tuple(t.shape or ()), to_numpy(t)))
        elif t.kind == "tensordesc":
            block = tuple(t.shape or ())
            elem = t.elem if t.elem is not None else t
            width = np.dtype(to_numpy(t)).itemsize
            shape = tuple(2 * d for d in block)
            strides = tuple(int(np.prod(shape[i + 1 :])) for i in range(len(shape)))
            memory.register(base, np.zeros(int(np.prod(shape or (1,))) * width, np.uint8))
            values.append(Descriptor(base, shape, strides, block, elem))
            base += 1 << 21
        elif t.kind == "int":
            values.append(np.array(2, to_numpy(t)))
        elif t.kind == "index":
            values.append(np.array(2, np.int64))
        elif t.kind == "float":
            values.append(np.array(1.0, to_numpy(t)))
        else:
            values.append(np.zeros((), np.int8))
    return values


def _phases(ri: RaceInterp, key: str) -> list[list[tuple[int, int]]]:
    """Arrivals on `key` cut into phases of the barrier's count, in replay order."""
    count = max(ri.barrier_count.get(key, 1), 1)
    arrivals = ri.arrivals.get(key, [])
    return [arrivals[i : i + count] for i in range(0, len(arrivals), count)]


def vector_clocks(ri: RaceInterp) -> dict[tuple[int, int], dict[int, int]]:
    """For every sync event (group, time) the vector clock after it: what that group knows of
    every group's time. A wait joins the clocks of the arrivals of its phase; the k-th wait of a
    group on a barrier pairs with the k-th phase. Replay order is not real order (the default
    region runs before the partitions), so the joins iterate to a fixpoint."""
    events: dict[int, list[tuple[int, str, str]]] = {}  # group -> [(time, kind, key)]
    for key, lst in ri.arrivals.items():
        for g, t in lst:
            events.setdefault(g, []).append((t, "arrive", key))
    for key, lst in ri.waits.items():
        for g, t in lst:
            events.setdefault(g, []).append((t, "wait", key))
    for g in events:
        events[g].sort()
    vc: dict[tuple[int, int], dict[int, int]] = {}
    for _round in range(8):
        changed = False
        for g, lst in events.items():
            known: dict[int, int] = {}
            waits_seen: dict[str, int] = {}
            for t, kind, key in lst:
                known[g] = t
                if kind == "wait":
                    k = waits_seen.get(key, 0)
                    waits_seen[key] = k + 1
                    phases = _phases(ri, key)
                    if k < len(phases):
                        for ag, at in phases[k]:
                            for og, ot in vc.get((ag, at), {ag: at}).items():
                                if known.get(og, -1) < ot:
                                    known[og] = ot
                if vc.get((g, t)) != known:
                    vc[(g, t)] = dict(known)
                    changed = True
        if not changed:
            break
    return vc


def cross_group_races(ri: RaceInterp) -> list[Race]:
    """Accesses of different groups on overlapping bytes, one a write, that no chain of
    arrivals and waits orders either way."""
    vc = vector_clocks(ri)
    # the clock a group holds at time t: the last sync event at or before t
    by_group: dict[int, list[tuple[int, dict[int, int]]]] = {}
    for (g, t), known in vc.items():
        by_group.setdefault(g, []).append((t, known))
    for g in by_group:
        by_group[g].sort(key=lambda x: x[0])

    def knows(g: int, t: int) -> dict[int, int]:
        best: dict[int, int] = {}
        for st, known in by_group.get(g, []):
            if st <= t:
                best = known
            else:
                break
        return best

    races: list[Race] = []
    groups = sorted({a.group for a in ri.accesses})
    for i, g in enumerate(groups):
        for h in groups[i + 1 :]:
            for a in ri.accesses:
                if a.group != g or a.pending is not None:
                    continue
                for b in ri.accesses:
                    if b.group != h or b.pending is not None:
                        continue
                    if "w" not in (a.kind, b.kind):
                        continue
                    if a.addrs[-1] < b.addrs[0] or b.addrs[-1] < a.addrs[0]:
                        continue
                    n = a.overlaps(b)
                    if not n:
                        continue
                    a_before_b = knows(h, b.time).get(g, -1) >= a.time
                    b_before_a = knows(g, a.time).get(h, -1) >= b.time
                    if not a_before_b and not b_before_a:
                        races.append(Race(a, b, n))
    return races


def _summarise(fn_name: str, ri: RaceInterp, status: str) -> Report3:
    pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    for race in ri.races:
        pairs[(race.first.op[:120], race.second.op[:120])] += 1
    return Report3(
        fn_name, len(ri.accesses), ri.barriers, ri.races, dict(pairs), status, dict(ri.gaps)
    )


def _attr_int(attrs: dict[str, Any], key: str) -> int | None:
    raw = attrs.get(key)
    if raw is None:
        return None
    try:
        return int(str(raw).split(":")[0].strip())
    except ValueError:
        return None


def _units(module: Module, text: str) -> list[tuple[Module, int | None, int | None]]:
    """The modules to run, innermost first: a `-split-input-file` dump in generic form nests
    every input module inside one outer module, and each carries its own launch config."""
    nested = [op for op in module.ops if op.name == "builtin.module"]
    if not nested:
        nw, tpw = launch_config(text)
        return [
            (
                module,
                _attr_int(module.attrs, "ttg.num-warps") or nw,
                _attr_int(module.attrs, "ttg.threads-per-warp") or tpw,
            )
        ]
    units = []
    for op in nested:
        body = op.regions[0].blocks[0].ops if op.regions and op.regions[0].blocks else []
        funcs = {
            str(o.attrs["sym_name"]): o
            for o in body
            if o.name.endswith(".func") and "sym_name" in o.attrs
        }
        sub = Module(funcs=funcs, ops=body, attrs=dict(op.attrs))
        units.append(
            (
                sub,
                _attr_int(sub.attrs, "ttg.num-warps"),
                _attr_int(sub.attrs, "ttg.threads-per-warp"),
            )
        )
    return units


def detect(module_text: str, fn_name: str | None = None, verbose: bool = False) -> list[Report3]:
    """Run every function of a module (or `fn_name`) on synthetic inputs and report its races."""
    reports = []
    for module, num_warps, threads_per_warp in _units(mlir.parse(module_text), module_text):
        reports += _detect_module(module, num_warps, threads_per_warp, fn_name, verbose)
    return reports


def _detect_module(
    module: Module,
    num_warps: int | None,
    threads_per_warp: int | None,
    fn_name: str | None,
    verbose: bool,
) -> list[Report3]:
    reports = []
    for name, fn in module.funcs.items():
        if fn_name is not None and name != fn_name:
            continue
        memory = Memory()
        ri = RaceInterp(module, memory, (1, 1, 1), num_warps, threads_per_warp)
        status = "ok"
        t0 = time.time()
        try:
            args = synthetic_args(ri, fn, memory)
            ri.run(name, args)
        except Unsupported as exc:
            status = f"unsupported: {exc}"[:160]
        except Exception as exc:  # the level-1 run itself failed
            status = f"error: {type(exc).__name__}: {exc}"[:160]
        if status == "ok" and len(ri.group_warps) > 1:
            ri.cross_races = cross_group_races(ri)
            ri.races.extend(ri.cross_races)
        if verbose:
            stats = f"{len(ri.accesses)} accesses, {len(ri.races)} races, {ri.steps} steps"
            print(f"# {name}: {time.time() - t0:.1f}s, {stats}", file=sys.stderr, flush=True)
        reports.append(_summarise(name, ri, status))
    return reports


def split_modules(text: str) -> list[str]:
    """The modules of a `-split-input-file` output, in generic form."""
    parts = re.split(r'(?m)^(?="builtin\.module"\(\))', text)
    return [p for p in parts if p.strip().startswith('"builtin.module"')]


def strip_barriers(text: str) -> str:
    return "\n".join(ln for ln in text.split("\n") if not any(f'"{b}"' in ln for b in BARRIERS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("corpus", type=Path, help="generic-form modules (split-input-file output)")
    ap.add_argument("--strip-barriers", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--only", default=None, help="a function name")
    args = ap.parse_args()
    text = args.corpus.read_text()
    if args.strip_barriers:
        text = strip_barriers(text)
    rows = []
    for module_text in split_modules(text):
        try:
            t0 = time.time()
            reports = detect(module_text, args.only, verbose=True)
            print(
                f"# module of {len(reports)} functions in {time.time() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            rows.append({"function": "<module>", "status": f"parse: {exc}"[:160], "races": 0})
            continue
        for r in reports:
            rows.append(
                {
                    "function": r.function,
                    "status": r.status,
                    "accesses": r.accesses,
                    "barriers": r.barriers,
                    "races": len(r.races),
                    "pairs": [{"a": a, "b": b, "n": n} for (a, b), n in r.pairs.items()],
                    "gaps": r.gaps,
                }
            )
    for row in rows:
        flag = "RACE" if row.get("races") else "    "
        counts = f"races={row.get('races', 0):<4} acc={row.get('accesses', 0):<4}"
        print(
            f"{flag} {row['function']:<48} {counts} bar={row.get('barriers', 0):<3} {row['status']}"
        )
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_race = sum(1 for r in rows if r.get("races"))
    print(f"{len(rows)} functions, {n_ok} ran, {n_race} with a race")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
