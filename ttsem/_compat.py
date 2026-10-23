"""What Python 3.10 lacks of the newer standard library, in one place."""

from __future__ import annotations


def add_note(exc: BaseException, note: str) -> None:
    """`BaseException.add_note` (Python 3.11+); on 3.10 the same `__notes__` list, which the
    readers of a note take with `getattr(exc, "__notes__", None)`."""
    if hasattr(exc, "add_note"):
        exc.add_note(note)
        return
    notes = getattr(exc, "__notes__", None)
    if notes is None:
        notes = []
        exc.__notes__ = notes  # type: ignore[attr-defined]
    notes.append(note)


# `python -P -m RUNNER` (3.11+) keeps the working directory, where an answer's files are, off
# `sys.path`. Python 3.10 has no -P: the same effect from -c, which puts "" first on the path
# and is told to take it off before the runner is imported.
_NO_CWD = (
    "import runpy, sys; sys.path[:] = [p for p in sys.path if p not in ('', '.')]; "
    "runpy.run_module({runner!r}, run_name='__main__', alter_sys=True)"
)


def python_module(executable: str, runner: str) -> list[str]:
    """The command that runs module `runner` with the working directory kept off `sys.path`;
    its arguments follow."""
    import sys

    if sys.version_info >= (3, 11):
        return [executable, "-P", "-m", runner]
    return [executable, "-c", _NO_CWD.format(runner=runner)]


_NO_DIR = (
    "import runpy, sys; sys.path[:] = [p for p in sys.path if p not in ('', '.')]; "
    "sys.argv[0] = {path!r}; runpy.run_path({path!r}, run_name='__main__')"
)


def python_script(executable: str, path: str) -> list[str]:
    """`python -P PATH` (3.11+): the script with neither its directory nor the working directory
    on `sys.path` (`runpy.run_path` does not add a file's directory); its arguments follow."""
    import sys

    if sys.version_info >= (3, 11):
        return [executable, "-P", path]
    return [executable, "-c", _NO_DIR.format(path=path)]
