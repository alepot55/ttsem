"""Races between the program instances of one launch, from the accesses the semantics made.

A launch is a grid of program instances that run in any order and at the same time. The
level-1 semantics runs them one after the other, which is *one* of the schedules: a kernel whose
instances touch the same bytes gets the answer of that schedule, and a test that compares
outputs passes. With `Memory.access_log` on, every load, store and atomic update is kept with
the instance that made it, and this module looks for the pairs that make the result depend on
the schedule:

- the last stores of two instances to the same address hold different bytes;
- one instance loads an address another one stores to or updates atomically;
- one instance stores to an address another one updates atomically.

Two atomic updates of one address are not a race (that is what they are for), nor are two loads.
A store of the bytes the address already holds (kind `s`, a silent store) is no conflict for a
load: whoever loads sees the same bytes before and after it. Compilers write whole tensors back
after an update of a slice (`x[::2] += ...` stores the untouched rows of `x` again), and every
instance that reads those rows would otherwise be a race with the one that rewrites them. A silent
store still counts among the last stores of its address: if another instance leaves other bytes
there, the end state depends on the order.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

Entry = tuple[Any, str, str, np.ndarray, int, "np.ndarray | None"]


@dataclasses.dataclass
class PidRace:
    kind: str  # "write-write" | "read-write" | "write-atomic"
    address: int  # first byte the two accesses share
    first: tuple[Any, str, str]  # (program instance, access kind r|w|a, op text)
    second: tuple[Any, str, str]
    shared_bytes: int  # how many byte addresses are in a conflict of this kind


def _expand(log: list[Entry]) -> tuple[np.ndarray, ...]:
    """Every access as rows: address, instance index, kind code, entry index, value written.
    When every access has the same element size (the usual case) a row is an element and its
    value the element's bytes as one integer; otherwise a row is a byte."""
    sizes = {itemsize for _, _, _, _, itemsize, _ in log}
    if len(sizes) == 1 and next(iter(sizes)) <= 8:
        return _expand_elements(log, next(iter(sizes)))
    agents: dict[Any, int] = {}
    addr, who, kind, entry, val = [], [], [], [], []
    codes = {"r": 0, "w": 1, "a": 2, "s": 3}
    for index, (agent, k, _op, addrs, itemsize, raw) in enumerate(log):
        ident = agents.setdefault(agent, len(agents))
        span = (addrs[:, None] + np.arange(itemsize, dtype=np.int64)[None, :]).reshape(-1)
        addr.append(span)
        who.append(np.full(span.size, ident, dtype=np.int64))
        kind.append(np.full(span.size, codes[k], dtype=np.int8))
        entry.append(np.full(span.size, index, dtype=np.int64))
        val.append(
            raw.reshape(-1).astype(np.int16) if raw is not None else np.zeros(span.size, np.int16)
        )
    if not addr:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty.astype(np.int8), empty, empty.astype(np.int16)
    return tuple(np.concatenate(x) for x in (addr, who, kind, entry, val))


def _expand_elements(log: list[Entry], itemsize: int) -> tuple[np.ndarray, ...]:
    agents: dict[Any, int] = {}
    addr, who, kind, entry, val = [], [], [], [], []
    codes = {"r": 0, "w": 1, "a": 2, "s": 3}
    weights = (1 << (8 * np.arange(itemsize, dtype=np.uint64))).astype(np.uint64)
    for index, (agent, k, _op, addrs, _size, raw) in enumerate(log):
        ident = agents.setdefault(agent, len(agents))
        addr.append(addrs)
        who.append(np.full(addrs.size, ident, dtype=np.int64))
        kind.append(np.full(addrs.size, codes[k], dtype=np.int8))
        entry.append(np.full(addrs.size, index, dtype=np.int64))
        if raw is None:
            val.append(np.zeros(addrs.size, dtype=np.uint64))
        else:
            val.append((raw.reshape(addrs.size, itemsize).astype(np.uint64) * weights).sum(axis=1))
    return tuple(np.concatenate(x) for x in (addr, who, kind, entry, val))


def _spread(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unique keys, min of `values` per key, max of `values` per key)."""
    order = np.argsort(keys, kind="stable")
    k, v = keys[order], values[order]
    starts = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
    return k[starts], np.minimum.reduceat(v, starts), np.maximum.reduceat(v, starts)


def _may_overlap(log: list[Entry]) -> bool:
    """False when no two instances can share an address a race needs: every instance's stores
    and atomic updates stay inside an address range no other instance touches at all, except
    by loads of ranges nobody writes. Ranges are [min, max] per access, so True is only
    "look closer"; the usual kernel (each instance its own slice of the output, everyone
    loading the same inputs) is settled here without sorting a single address."""
    writes: list[tuple[int, int, Any]] = []
    reads: list[tuple[int, int, Any]] = []
    for agent, kind, _op, addrs, itemsize, _raw in log:
        span = (int(addrs.min()), int(addrs.max()) + itemsize - 1, agent)
        (reads if kind == "r" else writes).append(span)
    if not writes:
        return False
    writes.sort(key=lambda w: (w[0], w[1]))
    reach, owner = writes[0][1], writes[0][2]
    for lo, hi, agent in writes[1:]:
        if lo <= reach and agent != owner:
            return True
        if hi > reach:
            reach, owner = hi, agent
    starts = [w[0] for w in writes]
    ends_running: list[int] = []
    top = -1
    for w in writes:
        top = max(top, w[1])
        ends_running.append(top)
    import bisect

    for lo, hi, agent in reads:
        k = bisect.bisect_right(starts, hi)  # writes starting at or before the read's end
        if k and ends_running[k - 1] >= lo:
            # some write range reaches into the read: a race only if another instance's
            j = k - 1
            while j >= 0 and ends_running[j] >= lo:
                wlo, whi, wagent = writes[j]
                if whi >= lo and wlo <= hi and wagent != agent:
                    return True
                j -= 1
    return False


def find_race(log: list[Entry] | None) -> PidRace | None:
    if not log:
        return None
    if not _may_overlap(log):
        return None
    addr, who, kind, entry, val = _expand(log)
    if addr.size == 0 or np.unique(who).size < 2:
        return None
    # (kind, conflict addrs, the accesses of one side, the accesses of the other)
    found: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    w = kind == 1
    stores = w | (kind == 3)
    if stores.any():
        # what an address holds in the end is the last store of some instance, so only the last
        # store of each instance counts: instances that each write v1 then v2 always leave v2
        order = np.lexsort((entry[stores], who[stores], addr[stores]))
        a, o, v = addr[stores][order], who[stores][order], val[stores][order]
        last = np.concatenate(((a[1:] != a[:-1]) | (o[1:] != o[:-1]), [True]))
        keys, lo, hi = _spread(a[last], o[last])
        _, vlo, vhi = _spread(a[last], v[last])
        clash = keys[(lo != hi) & (vlo != vhi)]
        if clash.size:
            found.append(("write-write", clash, stores, stores))

    def cross(a_mask: np.ndarray, b_mask: np.ndarray) -> np.ndarray:
        """Addresses touched under `a_mask` by one instance and under `b_mask` by another."""
        if not a_mask.any() or not b_mask.any():
            return np.zeros(0, dtype=np.int64)
        ka, alo, ahi = _spread(addr[a_mask], who[a_mask])
        kb, blo, bhi = _spread(addr[b_mask], who[b_mask])
        common, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
        same = (alo[ia] == ahi[ia]) & (blo[ib] == bhi[ib]) & (alo[ia] == blo[ib])
        return common[~same]

    reads, atomics = kind == 0, kind == 2
    rw = cross(reads, w | atomics)
    if rw.size:
        found.append(("read-write", rw, reads, w | atomics))
    wa = cross(w, atomics)
    if wa.size:
        found.append(("write-atomic", wa, w, atomics))
    if not found:
        return None

    name, clash, side_a, side_b = found[0]
    at = int(clash.min())
    here = addr == at
    first_rows = np.flatnonzero(here & side_a)
    a = int(first_rows[0])
    second_rows = np.flatnonzero(here & side_b & (who != who[a]))
    if second_rows.size == 0:  # the first row was not one of the clashing instances
        for cand in first_rows[1:]:
            second_rows = np.flatnonzero(here & side_b & (who != who[cand]))
            if second_rows.size:
                a = int(cand)
                break
    b = int(second_rows[0])
    ea, eb = log[int(entry[a])], log[int(entry[b])]
    unit = log[0][4] if len({e[4] for e in log}) == 1 and log[0][4] <= 8 else 1
    shared = int(clash.size) * unit
    return PidRace(name, at, (ea[0], ea[1], ea[2]), (eb[0], eb[1], eb[2]), shared)
