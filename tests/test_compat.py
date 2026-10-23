"""ttsem/_compat.py: what stands in for Python 3.11+ on 3.10."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ttsem._compat import add_note, python_script


def test_add_note_keeps_the_notes_where_their_readers_look() -> None:
    exc = ValueError("x")
    add_note(exc, "first")
    add_note(exc, "second")
    assert exc.__notes__ == ["first", "second"]


def test_a_script_runs_with_neither_its_directory_nor_the_cwd_on_the_path(tmp_path: Path) -> None:
    here, there = tmp_path / "cwd", tmp_path / "scripts"
    for d in (here, there):
        d.mkdir()
        (d / "planted.py").write_text("raise SystemExit('planted module imported')\n")
    script = there / "probe.py"
    script.write_text(
        "import importlib.util, sys\n"
        "print(importlib.util.find_spec('planted') is None, sys.argv[1:])\n"
    )
    done = subprocess.run(
        [*python_script(sys.executable, str(script)), "a", "b"],
        cwd=here,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == "True ['a', 'b']"
