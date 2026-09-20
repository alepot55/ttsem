"""A flat, byte-addressable memory: the only thing a pointer points into.

Buffers are host copies of the kernel arguments, registered at the device address they had at
launch. Every access goes through an `int64` address, so pointer arithmetic done by the IR is
what decides which bytes are touched; nothing in the semantics knows about tensor identity.

A masked-off lane never touches memory, on any of load, store or atomic. An out-of-range
address whose lane is live raises `MemoryFault`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from ttsem.values import Poison

# Decodes a stored value into the value it denotes, or encodes it back (`values.to_float` and
# `values.from_float` for the types whose storage is a bit pattern).
Codec = Callable[[np.ndarray], np.ndarray]

RMW = {
    "add": np.add,
    "fadd": np.add,
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "xor": np.bitwise_xor,
    "max": np.maximum,
    "min": np.minimum,
    "umax": np.maximum,
    "umin": np.minimum,
    "xchg": lambda _old, new: new,
}
UNSIGNED_RMW = ("umax", "umin")


class MemoryFault(Exception):
    """An access outside every registered buffer. `addrs` are all the unmapped addresses of the
    access and `kind` says whether it was a read or a write, for a caller that wants to tell
    the author of the kernel which buffer was overrun and by how much (`sanitize.py`)."""

    def __init__(self, addr: int, addrs: np.ndarray | None = None, itemsize: int = 1) -> None:
        super().__init__(f"unmapped address 0x{addr:x}")
        self.addr = addr
        self.addrs = np.array([addr], dtype=np.int64) if addrs is None else addrs
        self.itemsize = itemsize
        self.kind = ""


def _reject_conflicting_lanes(addrs: np.ndarray, raw: np.ndarray) -> None:
    """Two lanes of one store on the same address with different bytes: the IR does not say
    which one lands, and the hardware serialises them in an order nobody chose. Replicated
    elements (a broadcast layout) write the same bytes from every lane and pass; a warp-uniform
    address written from lanes whose values differ in the last bit does not (a class seen in a
    reduction epilogue at the LLVM level)."""
    if addrs.size < 2:
        return
    order = np.argsort(addrs, kind="stable")
    sorted_addrs = addrs[order]
    same = sorted_addrs[1:] == sorted_addrs[:-1]
    if not same.any():
        return
    sorted_raw = raw[order]
    differ = same & (sorted_raw[1:] != sorted_raw[:-1]).any(axis=1)
    if differ.any():
        k = int(np.flatnonzero(differ)[0])
        a, b = sorted_raw[k].tobytes().hex(), sorted_raw[k + 1].tobytes().hex()
        raise Poison(
            f"two lanes of one store wrote address 0x{int(sorted_addrs[k]):x} "
            f"with different values (0x{a} and 0x{b})"
        )


class Memory:
    """Buffers registered at their base addresses; loads and stores by address."""

    def __init__(self) -> None:
        self._bases: list[int] = []
        self.exchanged: list[np.ndarray] = []  # addresses hit by atomic xchg / cas
        # byte address -> the bytes a descriptor store would put there if it wrote whole
        # 16-byte granules past the inner extent, as the TMA unit does (triton#11583)
        self.pad_writes: dict[int, bytes] = {}
        self._arrays: list[np.ndarray] = []
        self._views: list[np.ndarray] = []
        self._sizes: list[int] = []
        self._sorted = np.zeros(0, dtype=np.int64)
        self._order = np.zeros(0, dtype=np.int64)
        self.trace_reads = False  # when set, every load's byte addresses are recorded
        self._reads: list[np.ndarray] = []
        # When a list, every access is recorded as (agent, kind, op, element addresses, itemsize,
        # bytes written or None): who touched what, for the race detector between program
        # instances (`pidraces.py`). `agent` and `current_op` are set by the interpreter.
        self.access_log: list[tuple[Any, str, str, np.ndarray, int, np.ndarray | None]] | None = (
            None
        )
        self.agent: Any = (0, 0, 0)
        self.current_op = ""
        # The log holds every address of every access: past this many elements it is dropped
        # and `access_overflow` says the launch was too large to be checked for races.
        self.access_budget = 4_000_000
        self.access_overflow = False
        self._logged = 0

    def _log(self, kind: str, addrs: np.ndarray, itemsize: int, raw: np.ndarray | None) -> None:
        if self.access_log is None or not addrs.size:
            return
        self._logged += int(addrs.size)
        if self._logged > self.access_budget:
            self.access_log, self.access_overflow = None, True
            return
        self.access_log.append((self.agent, kind, self.current_op, addrs, itemsize, raw))

    def register(self, base: int, array: np.ndarray) -> None:
        if not array.flags["C_CONTIGUOUS"]:
            raise ValueError("a registered buffer must be C-contiguous")
        self._bases.append(int(base))
        self._arrays.append(array)
        self._views.append(array.reshape(-1).view(np.uint8))
        self._sizes.append(array.nbytes)
        order = np.argsort(np.array(self._bases, dtype=np.int64), kind="stable")
        self._order = order
        self._sorted = np.array(self._bases, dtype=np.int64)[order]

    def read_bytes(self) -> np.ndarray:
        """The distinct byte addresses loaded so far (empty unless `trace_reads`)."""
        if not self._reads:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(self._reads))

    def buffers(self) -> dict[int, np.ndarray]:
        return dict(zip(self._bases, self._arrays, strict=True))

    def _resolve(self, addrs: np.ndarray, itemsize: int) -> tuple[np.ndarray, np.ndarray]:
        """Map addresses to (buffer index, byte offset), faulting on anything unmapped."""
        if len(self._bases) == 0:
            if addrs.size:
                raise MemoryFault(int(addrs[0]))
            return addrs.astype(np.int64), addrs.astype(np.int64)
        slot = np.searchsorted(self._sorted, addrs, side="right") - 1
        bad = slot < 0
        idx = self._order[np.maximum(slot, 0)]
        sizes = np.array(self._sizes, dtype=np.int64)[idx]
        off = addrs - np.array(self._bases, dtype=np.int64)[idx]
        bad |= (off < 0) | (off + itemsize > sizes)
        if bad.any():
            raise MemoryFault(int(addrs[bad][0]), np.asarray(addrs[bad], dtype=np.int64), itemsize)
        return idx, off

    def _read(self, addrs: np.ndarray, dtype: np.dtype) -> np.ndarray:
        try:
            idx, off = self._resolve(addrs, dtype.itemsize)
        except MemoryFault as fault:
            fault.kind = "read"
            raise
        span = np.arange(dtype.itemsize, dtype=np.int64)
        if idx.size and idx.min() == idx.max():  # one buffer: the usual access, no grouping
            raw = self._views[int(idx[0])][off[:, None] + span]
            return np.ascontiguousarray(raw).view(dtype).reshape(-1)
        raw = np.zeros((addrs.size, dtype.itemsize), dtype=np.uint8)
        for b in np.unique(idx):
            sel = idx == b
            raw[sel] = self._views[int(b)][off[sel][:, None] + span]
        return raw.view(dtype).reshape(-1)

    def _write(self, addrs: np.ndarray, vals: np.ndarray) -> None:
        if addrs.size == 0:  # a store whose mask is false everywhere touches nothing
            return
        try:
            idx, off = self._resolve(addrs, vals.dtype.itemsize)
        except MemoryFault as fault:
            fault.kind = "write"
            raise
        raw = np.ascontiguousarray(vals).view(np.uint8).reshape(addrs.size, -1)
        _reject_conflicting_lanes(addrs, raw)
        span = np.arange(vals.dtype.itemsize, dtype=np.int64)
        if idx.min() == idx.max():  # one buffer: the usual access, no grouping
            self._views[int(idx[0])][off[:, None] + span] = raw
            return
        for b in np.unique(idx):
            sel = idx == b
            self._views[int(b)][off[sel][:, None] + span] = raw[sel]

    def load(
        self,
        addrs: np.ndarray,
        mask: np.ndarray | None,
        other: object,
        dtype: np.dtype,
    ) -> np.ndarray:
        dtype = np.dtype(dtype)
        shape = np.shape(addrs)
        if other is None:
            out = np.zeros(shape, dtype)
        else:
            out = np.array(np.broadcast_to(other, shape), dtype=dtype)
        live = _live(mask, shape).reshape(-1)
        flat = np.asarray(addrs, dtype=np.int64).reshape(-1)[live]
        out.reshape(-1)[live] = self._read(flat, dtype)
        if self.trace_reads and flat.size:
            self._reads.append(_byte_span(flat, dtype.itemsize))
        self._log("r", flat, dtype.itemsize, None)
        return out

    def store(self, addrs: np.ndarray, values: np.ndarray, mask: np.ndarray | None) -> None:
        shape = np.shape(addrs)
        live = _live(mask, shape).reshape(-1)
        flat = np.asarray(addrs, dtype=np.int64).reshape(-1)[live]
        vals = np.ascontiguousarray(np.broadcast_to(values, shape).reshape(-1)[live])
        silent = None
        if self.access_log is not None and flat.size:
            try:  # a store of the bytes already there changes nothing for whoever loads them
                held = np.ascontiguousarray(self._read(flat, vals.dtype)).view(np.uint8)
                silent = held.reshape(flat.size, -1) == vals.view(np.uint8).reshape(flat.size, -1)
                silent = silent.all(axis=1)
            except MemoryFault:
                silent = None  # out of bounds: the write below says so
        self._write(flat, vals)
        if self.access_log is not None and flat.size:
            raw = vals.view(np.uint8).reshape(flat.size, -1)
            if silent is None or not silent.any():
                self._log("w", flat, vals.dtype.itemsize, raw)
            else:
                self._log("w", flat[~silent], vals.dtype.itemsize, raw[~silent])
                self._log("s", flat[silent], vals.dtype.itemsize, raw[silent])

    def atomic(
        self,
        kind: str,
        addrs: np.ndarray,
        values: object,
        mask: np.ndarray | None,
        sem: str = "acq_rel",
        codec: tuple[Codec, Codec] | None = None,
    ) -> np.ndarray:
        """Read-modify-write, one live lane at a time, returning the old values.

        `kind` is a key of `RMW` or `"cas"`; for `"cas"`, `values` is the pair `(cmp, val)`.
        Lanes are applied in row-major order, so two lanes at the same address compose in that
        order (`sem` does not change a single-threaded execution and is only recorded).

        The addresses an exchange or a compare-and-swap touches are kept in `exchanged`: what
        such a buffer holds at the end depends on the order the programs arrived in, which a
        sequential replay fixes one way and the device another.

        `codec` is an optional `(decode, encode)` pair for an element type whose storage is a
        bit pattern rather than the value (bf16 and the fp8 kinds): without it a `fadd` would
        add the *bit patterns*, which is what made a bf16 `tl.atomic_add` return 1.4e38. It
        does not apply to `cas`, which compares bits by definition.
        """
        if kind in ("xchg", "cas"):
            live = np.asarray(addrs, dtype=np.int64).reshape(-1)
            if mask is not None:
                live = live[np.broadcast_to(np.asarray(mask, bool), np.shape(addrs)).reshape(-1)]
            self.exchanged.append(live)
        del sem
        cmp_, val = values if kind == "cas" else (None, values)
        val = np.asarray(val)
        shape = np.shape(addrs)
        val_b = np.broadcast_to(val, shape).reshape(-1)
        cmp_b = np.broadcast_to(np.asarray(cmp_), shape).reshape(-1) if kind == "cas" else None
        live = _live(mask, shape).reshape(-1)
        flat = np.asarray(addrs, dtype=np.int64).reshape(-1)
        old = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=val.dtype)
        if self.access_log is not None and live.any():
            self._log("a", flat[live], val.dtype.itemsize, None)
        for i in np.nonzero(live)[0]:
            addr = flat[i : i + 1]
            cur = self._read(addr, val.dtype)
            old[i] = cur[0]
            if kind == "cas":
                new = val_b[i] if cur[0] == cmp_b[i] else cur[0]
            else:
                new = _rmw(kind, cur[0], val_b[i], codec)
            self._write(addr, np.array([new], dtype=val.dtype))
        return old.reshape(shape)


def _rmw(
    kind: str,
    old: np.generic,
    new: np.generic,
    codec: tuple[Codec, Codec] | None = None,
) -> np.generic:
    if kind not in RMW:
        raise ValueError(f"unknown atomic kind {kind}")
    if codec is not None:
        decode, encode = codec
        combined = RMW[kind](decode(old), decode(new))
        return np.asarray(encode(combined)).astype(old.dtype).reshape(())
    if kind in UNSIGNED_RMW:
        bits = old.dtype.itemsize * 8
        udtype = np.dtype(f"uint{bits}")
        return RMW[kind](old.astype(udtype), new.astype(udtype)).astype(old.dtype)
    with np.errstate(over="ignore"):
        return np.asarray(RMW[kind](old, new)).astype(old.dtype)


def _live(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    return np.broadcast_to(np.asarray(mask, dtype=bool), shape)


def _byte_span(addrs: np.ndarray, size: int) -> np.ndarray:
    """Every byte address covered by elements of `size` bytes at `addrs`."""
    return (addrs[:, None] + np.arange(size, dtype=np.int64)[None, :]).reshape(-1)
