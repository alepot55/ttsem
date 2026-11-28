"""The MAP-Elites archive of the search: the best correct program of each cell, and every record.

A cell is :meth:`Verdict.cell`: passes over memory, state fields, float32 class. Its elite is the
correct candidate with the lowest fitness, the time measured on the GPU when one is attached to
the record and the cost model's estimate otherwise. Two fitnesses within ``tolerance`` of each
other (2% by default) are a tie, which goes to the smaller program, then to the earlier one; so
are two measured times whose ranges (the spread of the repeated timings) overlap, since the GPU
cannot tell them apart. An archive built ``measured_only`` (the search with
``--gpu``) chooses its elites among the records with a measured time only, so an estimate never
competes with a measurement; a record whose kernel failed the GPU check (status ``gpu-failed``)
is never an elite, hence never a parent nor an inspiration. Every evaluated candidate stays as a
record, wrong ones and duplicates (a program whose digest was seen before) included, so the
archive is also the run's history. Parents are drawn from the occupied cells, uniformly but for
a bias towards fewer passes; inspirations are the elites of other cells.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from discovery.evaluate import Verdict

Cell = tuple[int, int, str]
TOLERANCE = 0.02  # relative: two fitnesses this close are a tie, which the smaller program wins


def verdict_from_json(d: dict) -> Verdict:
    """The inverse of :meth:`Verdict.to_json`."""
    d = dict(d)
    d["exact_err"] = math.inf if d.get("exact_err") is None else d["exact_err"]
    d["fp"] = {
        fam: {k: math.inf if e is None else e for k, e in errs.items()}
        for fam, errs in d.get("fp", {}).items()
    }
    return Verdict(**d)


@dataclasses.dataclass
class Record:
    verdict: Verdict
    iteration: int = 0
    parent: str = ""  # digest of the program the proposer started from; empty for the seed
    # what inserting it did (for a time attached afterwards, what the measurement did):
    # new-cell | improved | kept | rejected | duplicate | gpu-failed | unmeasured
    status: str = ""
    measured_us: float | None = None
    gpu_failure: str = ""  # the GPU check's message when the program's kernel failed it
    # the range of the repeated timings behind measured_us (lowest, highest), when known
    measured_range: tuple[float, float] | None = None

    @property
    def fitness(self) -> float:
        if self.measured_us is not None:
            return self.measured_us
        return float(self.verdict.cost.get("time_us", math.inf))

    def to_json(self) -> dict:
        return {
            "iteration": self.iteration,
            "parent": self.parent,
            "status": self.status,
            "measured_us": self.measured_us,
            "gpu_failure": self.gpu_failure,
            "measured_range": None if self.measured_range is None else list(self.measured_range),
            "verdict": self.verdict.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> Record:
        return cls(
            verdict_from_json(d["verdict"]),
            d["iteration"],
            d["parent"],
            d["status"],
            d.get("measured_us"),
            d.get("gpu_failure", ""),
            None if d.get("measured_range") is None else tuple(d["measured_range"]),
        )


def _overlap(a: Record, b: Record) -> bool:
    """Whether both records were timed and their ranges of repeated timings overlap."""
    ra, rb = a.measured_range, b.measured_range
    if a.measured_us is None or b.measured_us is None or ra is None or rb is None:
        return False
    return ra[0] <= rb[1] and rb[0] <= ra[1]


def _better(a: Record, b: Record, tolerance: float = TOLERANCE) -> bool:
    """Whether ``a`` beats ``b``: faster by more than ``tolerance`` and, when both were timed,
    with a range of timings apart from ``b``'s; otherwise (a tie) smaller. With a tolerance of 0
    and no ranges, the smaller program wins exact ties only."""
    fa, fb = a.fitness, b.fitness
    if abs(fa - fb) <= tolerance * min(fa, fb) or _overlap(a, b):
        return a.verdict.nodes < b.verdict.nodes
    return fa < fb


class Archive:
    def __init__(
        self, pass_bias: float = 1.0, tolerance: float = TOLERANCE, measured_only: bool = False
    ) -> None:
        # a cell's chance to give the parent goes as exp(-pass_bias * passes); 0 is uniform
        self.pass_bias = pass_bias
        self.tolerance = tolerance
        self.measured_only = measured_only
        self.records: list[Record] = []
        self.elites: dict[Cell, Record] = {}
        self.by_digest: dict[str, Record] = {}  # the first record of each program

    def eligible(self, rec: Record) -> bool:
        """Whether a record may be its cell's elite: correct, not a duplicate, its kernel not
        failed on the GPU, and in a ``measured_only`` archive timed."""
        return (
            rec.verdict.correct
            and rec.status not in ("duplicate", "gpu-failed")
            and (rec.measured_us is not None or not self.measured_only)
        )

    def insert(
        self,
        verdict: Verdict,
        iteration: int = 0,
        parent: str = "",
        measured_us: float | None = None,
        gpu_failure: str = "",
        measured_range: tuple[float, float] | None = None,
    ) -> Record:
        """Record one evaluated candidate; a correct one may become its cell's elite.
        ``measured_us`` is the time its kernel took on the GPU and ``measured_range`` the range of
        its repeated timings, ``gpu_failure`` the message of a kernel that failed the GPU check. A
        correct candidate with no time, in a ``measured_only`` archive, is ``unmeasured`` until
        :meth:`attach_time`."""
        rec = Record(
            verdict,
            iteration,
            parent,
            measured_us=measured_us,
            gpu_failure=gpu_failure,
            measured_range=measured_range,
        )
        cell = verdict.cell()
        if verdict.digest and verdict.digest in self.by_digest:
            rec.status = "duplicate"
        elif not verdict.correct:
            rec.status = "rejected"
        elif gpu_failure:
            rec.status = "gpu-failed"
        elif not self.eligible(rec):
            rec.status = "unmeasured"
        elif cell not in self.elites:
            rec.status = "new-cell"
        elif _better(rec, self.elites[cell], self.tolerance):
            rec.status = "improved"
        else:
            rec.status = "kept"
        if rec.status in ("new-cell", "improved"):
            self.elites[cell] = rec
        if verdict.digest and rec.status != "duplicate":
            self.by_digest[verdict.digest] = rec
        self.records.append(rec)
        return rec

    def attach_time(
        self, digest: str, us: float, measured_range: tuple[float, float] | None = None
    ) -> None:
        """Make a measured time the fitness of a program and choose its cell's elite again. The
        record's status becomes what the measurement did: ``new-cell`` or ``improved`` when it
        made the record its cell's elite, ``kept`` when the record is not (or no longer) it."""
        rec = self.by_digest[digest]
        rec.measured_us = us
        rec.measured_range = measured_range
        if rec.status == "gpu-failed" or not rec.verdict.correct:
            return
        cell = rec.verdict.cell()
        before = self.elites.get(cell)
        self._choose(cell)
        if self.elites.get(cell) is not rec:
            rec.status = "kept"
        elif before is None:
            rec.status = "new-cell"
        elif before is not rec:
            rec.status = "improved"

    def _choose(self, cell: Cell) -> None:
        """Choose a cell's elite again among its eligible records, in the order they came."""
        self.elites.pop(cell, None)
        for r in self.records:
            if r.verdict.cell() == cell and self.eligible(r):
                if cell not in self.elites or _better(r, self.elites[cell], self.tolerance):
                    self.elites[cell] = r

    def select_parent(self, rng: np.random.Generator) -> Record | None:
        """The elite of a cell drawn among the occupied ones, fewer passes weighing more."""
        if not self.elites:
            return None
        cells = sorted(self.elites)
        passes = np.array([c[0] for c in cells], dtype=float)
        weight = np.exp(-self.pass_bias * (passes - passes.min()))
        return self.elites[cells[rng.choice(len(cells), p=weight / weight.sum())]]

    def inspirations(
        self, parent: Record | None, rng: np.random.Generator, k: int = 2
    ) -> list[Record]:
        """Up to ``k`` elites of cells other than the parent's, drawn uniformly."""
        own = parent.verdict.cell() if parent is not None else None
        others = [c for c in sorted(self.elites) if c != own]
        picked = rng.choice(len(others), size=min(k, len(others)), replace=False)
        return [self.elites[others[i]] for i in sorted(picked.tolist())]

    def save(self, path: str | Path) -> None:
        data = {
            "pass_bias": self.pass_bias,
            "tolerance": self.tolerance,
            "measured_only": self.measured_only,
            "records": [r.to_json() for r in self.records],
        }
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Archive:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # a file with no tolerance was written when only exact ties went to the smaller program
        archive = cls(
            data["pass_bias"], data.get("tolerance", 0.0), data.get("measured_only", False)
        )
        for d in data["records"]:
            rec = Record.from_json(d)
            archive.records.append(rec)
            if rec.status != "duplicate" and rec.verdict.digest:
                archive.by_digest.setdefault(rec.verdict.digest, rec)
        for cell in sorted({r.verdict.cell() for r in archive.records if archive.eligible(r)}):
            archive._choose(cell)
        return archive
