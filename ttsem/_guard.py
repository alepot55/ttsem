"""Two stop-gaps that make an answer's own process a little less able to fool the judge, short of
the full separation of the answer from the checker planned for the next release.

The answer runs in the same Python process as the semantics, so in principle it can reach into the
checker and rewrite it. These two measures catch the direct forms of that seen so far; they are not
a sandbox for the checker, and the module docstring of :mod:`ttsem.judge_io` and ``docs/JUDGE.md``
say what they do and do not promise.

- :func:`tamper_scan` reads the answer's source (never runs it) and refuses, before the answer is
  launched, code that reaches into the judge: importing the package or one of its modules, walking
  ``sys.modules``, the garbage collector or the interpreter's frames, ``ctypes``, assigning an
  object's ``__code__`` / ``__class__`` / ``__dict__``, reading ``/proc/self/fd``, calling
  ``os.write`` / ``os.dup2`` / ``os._exit`` and the like, or naming the judge's own events flag.
  The rules are kept narrow on purpose: ordinary Triton kernels and PyTorch code use none of these.
- :class:`Integrity` snapshots, before the answer is imported, the identity and the code object of
  the checker's critical callables (``Memory`` and its methods, the semantics' ``execute`` and IR
  entry points, the race finder, the launch hook on ``JITFunction.run``, and the ``os`` bindings
  the channel uses), and reports a mismatch so the run can stop with ``not_judged`` rather than
  trust an account produced after the answer swapped the checker's own code.

Imports only the standard library at module level, so a runner can build the snapshot before any
code under test is imported.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from typing import Any

# --- the static scan ------------------------------------------------------------------------------

# The os functions an answer has no business calling: writing or duplicating a file descriptor it
# did not open (the judge's events pipe), or leaving the process without the runner's last word.
_OS_CALLS = {"write", "dup", "dup2", "_exit", "fork", "forkpty", "kill", "pipe", "openpty"}
# Attributes whose assignment rewrites an object's behaviour, wherever the object comes from.
_DUNDER_WRITES = {"__code__", "__class__", "__dict__", "__globals__", "__func__", "__get__"}
# The interpreter frame fields a walker reads to climb out of its own scope.
_FRAME_FIELDS = {"f_back", "f_globals", "f_locals", "f_code", "f_builtins", "tb_frame"}
# Names that, called, hand out other objects' frames or the live object graph.
_INTROSPECTION = {
    ("sys", "_getframe"),
    ("sys", "settrace"),
    ("inspect", "currentframe"),
    ("inspect", "stack"),
    ("inspect", "trace"),
    ("inspect", "getouterframes"),
    ("inspect", "getinnerframes"),
    ("gc", "get_objects"),
    ("gc", "get_referrers"),
    ("gc", "get_referents"),
}


def _module_roots(node: ast.AST) -> Iterable[str]:
    """The top-level module names an ``import`` or ``from ... import`` node brings in."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name.split(".")[0]
    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        yield node.module.split(".")[0]


def _attr_chain(node: ast.AST) -> tuple[str, str] | None:
    """``(value_name, attr)`` for ``name.attr`` where the value is a bare name, else None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id, node.attr
    return None


def tamper_scan(source: str, forbidden_modules: frozenset[str]) -> str | None:
    """A reason the answer must not be run, read off its source without running it, or None.

    ``forbidden_modules`` is the set of top-level module names that are the judge's own (the
    package, or the flat modules of the private copy): importing any of them, by ``import`` or by
    a string handed to ``importlib.import_module`` / ``__import__``, is refused. A source that does
    not parse is left to the run (the answer's own error, reported there), not refused here."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for root in _module_roots(node):
                if root in forbidden_modules:
                    return f"it imports the judge's own module `{root}`"
                if root == "ctypes":
                    return "it imports `ctypes`"
        elif isinstance(node, ast.Call):
            reason = _call_reason(node, forbidden_modules)
            if reason is not None:
                return reason
        elif isinstance(node, ast.Attribute):
            pair = _attr_chain(node)
            if pair == ("sys", "modules"):
                return "it reaches into `sys.modules`"
            if node.attr in _FRAME_FIELDS:
                return f"it walks interpreter frames (`.{node.attr}`)"
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            reason = _assign_reason(node)
            if reason is not None:
                return reason
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "/proc/self/fd" in node.value:
                return "it reads `/proc/self/fd`"
            if node.value == "--events":
                return "it names the judge's events flag (`--events`)"
    return None


def _call_reason(node: ast.Call, forbidden_modules: frozenset[str]) -> str | None:
    pair = _attr_chain(node.func)
    if pair is not None:
        value, attr = pair
        if value == "os" and attr in _OS_CALLS:
            return f"it calls `os.{attr}`"
        if (value, attr) in _INTROSPECTION:
            return f"it reaches into the interpreter (`{value}.{attr}`)"
    name = node.func.id if isinstance(node.func, ast.Name) else ""
    if name == "__import__" or pair == ("importlib", "import_module"):
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value.split(".")[0] in forbidden_modules:
                return f"it imports the judge's own module `{first.value.split('.')[0]}`"
    if name == "setattr" and len(node.args) >= 2:  # setattr(obj, "__code__", ...)
        key = node.args[1]
        if isinstance(key, ast.Constant) and key.value in _DUNDER_WRITES:
            return f"it rewrites `{key.value}` of an object"
    return None


def _assign_reason(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> str | None:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if isinstance(target, ast.Attribute) and target.attr in _DUNDER_WRITES:
            return f"it rewrites `{target.attr}` of an object"
    return None


# --- the runtime integrity check ------------------------------------------------------------------


class Integrity:
    """A snapshot of the checker's critical callables, taken before the answer is imported, and a
    check that each one still is the object it was, with the code it had.

    ``watch`` records how to fetch one callable or class (a getter, so a later rebinding of the
    attribute is seen); ``arm`` takes the snapshot once every legitimate install is in place;
    ``breach`` returns the label of the first one that changed, or None. The original objects are
    held so that a swapped class or function is caught by identity, not only by a code object that
    a later allocation might reuse."""

    def __init__(self) -> None:
        self._watch: list[tuple[str, Callable[[], Any]]] = []
        self._snap: list[tuple[str, Callable[[], Any], Any, Any]] = []

    def watch(self, label: str, getter: Callable[[], Any]) -> None:
        self._watch.append((label, getter))

    def arm(self) -> None:
        self._snap = [
            (label, getter, getter(), _code_of(getter())) for label, getter in self._watch
        ]

    def breach(self, skip: frozenset[str] = frozenset()) -> str | None:
        """The label of the first watched object that is no longer the one snapshotted (a swapped
        attribute), or whose code object changed (``f.__code__ = g.__code__``), or None. ``skip``
        leaves out labels whose being reached already proves they hold (the launch hook, checked
        from inside itself)."""
        for label, getter, obj, code in self._snap:
            if label in skip:
                continue
            try:
                now = getter()
            except Exception:  # noqa: BLE001  the attribute itself was removed
                return label
            if now is not obj or _code_of(now) is not code:
                return label
        return None


def _code_of(obj: Any) -> Any:
    """The code object of a function, or None for a class or a builtin (identity carries those)."""
    return getattr(obj, "__code__", None)


def checker_integrity(
    *, memory: Any, harness: Any, pidraces: Any, jitfunction: Any, os_module: Any
) -> Integrity:
    """The standard watch list for a judge's runner: the out-of-bounds check and the race log
    (``Memory`` and its methods), the semantics' execute and IR entry points, the race finder, the
    launch hook on ``JITFunction.run``, and the ``os`` bindings the channel leaves the process by.
    Built from the modules the runner already imported, so it does not care how they are named
    (the package's ``ttsem.memory`` or the private copy's flat ``memory``). Call ``arm`` once every
    legitimate install is in place (after the session hook and any wrapper on it) and before the
    answer is imported."""
    g = Integrity()
    g.watch("Memory", lambda: memory.Memory)
    g.watch("Memory._resolve", lambda: memory.Memory._resolve)
    g.watch("Memory._log", lambda: memory.Memory._log)
    g.watch("Memory.load", lambda: memory.Memory.load)
    g.watch("Memory.store", lambda: memory.Memory.store)
    g.watch("harness.execute", lambda: harness.execute)
    g.watch("harness.ir_for_launch", lambda: harness.ir_for_launch)
    g.watch("harness.record_launch", lambda: harness.record_launch)
    g.watch("pidraces.find_race", lambda: pidraces.find_race)
    g.watch("os.write", lambda: os_module.write)
    g.watch("os._exit", lambda: os_module._exit)
    g.watch("JITFunction.run", lambda: jitfunction.run)
    return g


# the label left out of the in-launch check: being inside the launch hook already proves it holds
HOOK = frozenset({"JITFunction.run"})
