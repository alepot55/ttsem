"""Tier 3's memory: what earlier candidates taught, and the lemmas the prover has proved.

Per problem, a JSON file under ``discovery/runs/knowledge/`` keeps every rejected candidate with
its stage and reason (a kernel that failed the GPU check included, at the stage ``gpu``), and
every accepted one with its cell and estimated time; the proposer of a later iteration, or of a
later run, reads the last rejections. Blind runs keep a file of their own, so a blind proposer
never reads a name an open run wrote.

The lemmas come from a library index, by default the proof tier's
``lean/Discovery/Library/index.json``, which this module only reads, once, when the knowledge is
created: every prompt of a run shows the same lemmas, whatever the prover adds meanwhile. A
lemma's statement spells out the merges of the program it proves, so a lemma about one of a
problem's known forms puts that form in the prompt; ``library=None`` shows no lemma at all (the
rediscovery control's default). Whether feeding any of this back helps is to be measured at
equal compute: here it is recorded and shown.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from discovery.evaluate import Verdict

HERE = Path(__file__).resolve().parent
KNOWLEDGE_DIR = HERE / "runs" / "knowledge"
LIBRARY_INDEX = HERE / "lean" / "Discovery" / "Library" / "index.json"
LEMMAS_SHOWN = 12  # a prompt shows the last lemmas of the index, at most this many


class Knowledge:
    def __init__(
        self,
        problem: str,
        blind: bool = False,
        path: str | Path | None = None,
        library: str | Path | None = LIBRARY_INDEX,
    ) -> None:
        self.problem = problem
        default = KNOWLEDGE_DIR / f"{problem}{'-blind' if blind else ''}.json"
        self.path = Path(path) if path is not None else default
        self.library = Path(library) if library is not None else None
        self.rejected: list[dict] = []
        self.accepted: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.rejected, self.accepted = data["rejected"], data["accepted"]
        self._seen = {_key(e) for e in self.rejected + self.accepted}
        self._lemmas = self._read_library()

    def record(self, verdict: Verdict, gpu_failure: str = "") -> None:
        """Remember one evaluated candidate; a program already remembered is not stored again.
        ``gpu_failure`` is the message of a kernel that failed the GPU check: the candidate is
        then remembered as rejected at the stage ``gpu``, even if it was accepted before."""
        key = verdict.digest or verdict.text
        if gpu_failure:
            self.accepted = [e for e in self.accepted if _key(e) != key]
            if any(_key(e) == key for e in self.rejected):
                return
        elif key in self._seen:
            return
        self._seen.add(key)
        if verdict.correct and not gpu_failure:
            self.accepted.append(
                {
                    "digest": verdict.digest,
                    "text": verdict.text,
                    "cell": list(verdict.cell()),
                    "time_us": verdict.cost["time_us"],
                }
            )
        else:
            self.rejected.append(
                {
                    "digest": verdict.digest,
                    "text": verdict.text,
                    "stage": "gpu" if gpu_failure else verdict.stage,
                    "reason": gpu_failure or verdict.message,
                }
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"problem": self.problem, "rejected": self.rejected, "accepted": self.accepted}
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def _read_library(self) -> list[dict]:
        if self.library is None:
            return []
        try:
            data = json.loads(self.library.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        entries = data.get("lemmas", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            return []
        return [
            e
            for e in entries
            if isinstance(e, dict)
            and e.get("statement")
            and e.get("problem", self.problem) == self.problem
        ]

    def lemmas(self) -> list[dict]:
        """The proved lemmas of the library index that concern this problem or no problem in
        particular, as the index held them when this knowledge was created (none without a
        library). The index is a JSON list of objects (or ``{"lemmas": [...]}``), each with a
        ``statement`` and, optionally, a ``name``, a ``problem`` and the ``program`` proved."""
        return list(self._lemmas)

    def shown_lemmas(self, lemmas: int = LEMMAS_SHOWN) -> list[dict]:
        """The lemmas a prompt shows: the last ``lemmas`` of :meth:`lemmas`."""
        return self._lemmas[-lemmas:] if lemmas > 0 else []

    def prompt_section(
        self, show: Callable[[str], str] = str, last: int = 4, lemmas: int = LEMMAS_SHOWN
    ) -> str:
        """The proposer's view: the last rejected programs with their reasons, then the proved
        lemmas. ``show`` rewrites every text first (the blind mode's renaming)."""
        parts = []
        if self.rejected and last > 0:
            lines = [
                "# Recently rejected programs",
                "",
                "Each failed for the reason given after it; do not propose them again.",
            ]
            for e in self.rejected[-last:]:
                lines += [
                    "",
                    "```lisp",
                    show(e["text"]).strip(),
                    "```",
                    f"Rejected: {show(e['reason'])}",
                ]
            parts.append("\n".join(lines))
        proved = self.shown_lemmas(lemmas)
        if proved:
            lines = [
                "# Proved lemmas",
                "",
                "Proved in Lean 4 about earlier programs; a merge of the same shape is correct for "
                "the same reason.",
                "",
            ]
            lines += [f"- {show(e.get('name', 'lemma'))}: {show(e['statement'])}" for e in proved]
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


def _key(entry: dict) -> str:
    return entry["digest"] or entry["text"]
