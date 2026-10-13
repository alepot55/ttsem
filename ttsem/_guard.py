"""Two stop-gaps against an answer that tampers with the judge from inside its own run, until the
answer and the checker run in separate processes (planned for 0.2).

The answer runs in the same Python process as the checker. What the judge compares outside that
process (an answer's values against the reference's) holds against any answer; the `unsafe` and
`no_kernel` verdicts are decided inside it, and rest on these two measures:

- :func:`tamper_scan` reads the answer's source (never runs it) and refuses, before the answer is
  launched, any way to reach the checker without naming it: importing the judge, the import
  machinery, the interpreter's frames, object graph or module table, code built from strings or
  bytes, attribute names computed at run time, dunder attributes outside a short list, threads,
  signals, child processes, native code, and rebinding a module's attributes.
- :class:`Integrity` snapshots, before the answer is imported, the identity and code of every
  function, class and module-level binding of the checker, of the libraries it calls (numpy,
  Triton, the parts of torch it uses, the standard modules it imports, the builtins), and of the
  launch hook, plus the process state a patch could hide behind (threads, signal handlers, trace
  hooks, the import system); the check runs before and after every launch and before the run's
  account is written, and names the first thing that changed.

Neither is a proof. They block every attack we know and test (``tests/test_judge_integrity.py``),
but an adversary with the source can likely find another way until the checker leaves the
answer's process.

Imports only the standard library at module level, so a runner can build the snapshot before any
code under test is imported.
"""

from __future__ import annotations

import ast
import functools
import gc
import importlib
import importlib.util
import io
import operator
import signal
import sys
import tokenize
import types
from collections.abc import Callable, Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

# --- the static scan ------------------------------------------------------------------------------

# Modules an answer may not import in any form: the import machinery, code from strings or bytes,
# the interpreter's frames and object graph, native code, other threads and processes, signals,
# and the libraries that import a module or run code named by a string (mock.patch, pydoc.locate,
# timeit, doctest, the profilers, the debuggers).
_FORBIDDEN_IMPORTS = frozenset({
    "builtins", "__builtin__", "importlib", "imp", "pkgutil", "runpy", "zipimport", "code",
    "codeop", "site", "ctypes", "_ctypes", "cffi", "gc", "inspect", "marshal", "pickle", "_pickle",
    "cPickle", "dill", "cloudpickle", "joblib", "shelve", "copyreg", "_thread", "concurrent",
    "multiprocessing", "asyncio", "signal", "_signal", "faulthandler", "subprocess",
    "_posixsubprocess", "pty", "posix", "nt", "resource", "mmap", "unittest", "pydoc", "timeit",
    "doctest", "trace", "profile", "cProfile", "pdb", "bdb", "pandas", "numexpr", "sympy",
})  # fmt: skip
# Attribute names refused on any object: the forbidden modules reached as another module's
# attribute (`torch.serialization.os`), and the names that hand out frames, the module table,
# the import machinery, threads, signals, child processes or native code, whatever holds them.
_FORBIDDEN_ATTRIBUTES = frozenset({
    "builtins", "importlib", "pkgutil", "runpy", "zipimport", "ctypes", "_ctypes", "gc",
    "inspect", "marshal", "pickle", "_pickle", "dill", "cloudpickle", "shelve", "copyreg",
    "_thread", "concurrent", "multiprocessing", "asyncio", "subprocess", "_posixsubprocess",
    "pty", "posix", "faulthandler", "mmap", "unittest", "pydoc", "doctest", "cProfile", "pdb",
    "bdb", "os", "sys",
    # frames, tracebacks, code
    "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "f_trace", "tb_frame", "tb_next",
    "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "co_code",
    "currentframe", "getouterframes", "getinnerframes", "getframeinfo", "_getframe",
    "_getframemodulename", "_current_frames", "_current_exceptions",
    # the object graph, the module table, the import machinery
    "get_objects", "get_referrers", "get_referents", "getmodule", "getmembers", "getclosurevars",
    "getattr_static", "meta_path", "path_hooks", "path_importer_cache", "import_module",
    "reload", "resolve_name", "find_loader", "get_loader", "load_module", "exec_module",
    "module_from_spec", "spec_from_file_location", "spec_from_loader", "run_module", "run_path",
    "get_type_hints", "ForwardRef", "PackageImporter", "get_field", "attrgetter",
    "methodcaller",
    # trace and audit hooks
    "settrace", "setprofile", "settrace_all_threads", "setprofile_all_threads", "monitoring",
    "addaudithook", "set_asyncgen_hooks", "setswitchinterval",
    # threads, signals, processes, limits
    "Thread", "Timer", "ThreadPool", "ThreadPoolExecutor", "ProcessPoolExecutor",
    "start_new_thread", "start_joinable_thread", "_start_new_thread", "_start_joinable_thread",
    "interrupt_main", "setitimer", "alarm", "raise_signal", "pthread_kill", "set_wakeup_fd",
    "siginterrupt", "Popen", "check_output", "check_call", "setrlimit", "prlimit",
    # leaving the process, or writing a descriptor the answer did not open
    "_exit", "dup", "dup2", "fdopen", "forkpty", "openpty", "killpg", "posix_spawn",
    "posix_spawnp", "register_at_fork", "memfd_create", "system", "popen", "execl", "execle",
    "execlp", "execlpe", "execv", "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp",
    "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    # native code and unpickling
    "load_inline", "cpp_extension", "load_library", "compile_module_from_src", "ctypeslib",
    "pythonapi", "CDLL", "PyDLL", "cdll", "pydll", "read_pickle", "hub",
})  # fmt: skip
# Functions of `os` refused under their own names (where `write` or `open` on any other object
# is an ordinary method): a write to a descriptor the answer did not open, leaving the process,
# starting another.
_OS_FUNCTIONS = frozenset({
    "write", "writev", "pwrite", "open", "fork", "kill", "pipe", "pipe2", "sendfile", "abort",
})  # fmt: skip
# `types` hands out the constructors of code, functions, frames and modules
_TYPES = frozenset({
    "CodeType", "FunctionType", "LambdaType", "CellType", "FrameType", "TracebackType",
    "ModuleType", "MethodType", "new_class",
})  # fmt: skip
# Builtins refused by name: code from strings, the namespaces as dictionaries, the debugger.
_BUILTINS = frozenset({"exec", "eval", "compile", "globals", "locals", "vars", "breakpoint"})
# The attribute builtins, allowed only as a direct call with a string constant for the name.
_ATTRIBUTE_BUILTINS = frozenset({"getattr", "setattr", "delattr", "hasattr"})

# Dunder attributes an answer may read (decided from 7,752 TritonBench answers and the
# KernelBench-style corpora: `super().__init__()`, `self.__class__.__name__` in a `__repr__`,
# `torch.__version__`, `super().__setstate__(state)` in an optimizer), and the few it may set (a
# decorator copying a name and a docstring). Every other dunder is refused: `__globals__`,
# `__code__`, `__closure__`, `__dict__`, `__subclasses__`, `__builtins__`, `__import__`,
# `__traceback__` and the rest are how an answer climbs from an object it holds to the checker.
_DUNDER_READ = frozenset({
    "__init__", "__name__", "__qualname__", "__doc__", "__module__", "__version__", "__class__",
    "__call__", "__repr__", "__str__", "__len__", "__getitem__", "__setitem__", "__iter__",
    "__next__", "__enter__", "__exit__", "__file__", "__setstate__", "__getstate__",
})  # fmt: skip
_DUNDER_WRITE = frozenset({"__name__", "__qualname__", "__doc__", "__module__"})
# dunder names an answer may read as plain names, and bind at module or class level
_DUNDER_NAMES_READ = frozenset({"__name__", "__file__", "__doc__", "__class__", "__debug__"})
_DUNDER_NAMES_BOUND = frozenset({
    "__all__", "__constants__", "__doc__", "__version__", "__slots__", "__author__",
    "__annotations__",
})  # fmt: skip
# dunder methods an answer's class may define: those of the data model the corpora use, and the
# operators. Refused: the hooks through which the checker's own operations would call back into
# the answer (`__del__`, `__getattr__`, `__torch_function__`, `__index__`, `__init_subclass__`,
# descriptors, metaclass hooks, pickling).
_DUNDER_DEFS = frozenset({
    "__init__", "__repr__", "__str__", "__call__", "__getitem__", "__setitem__", "__delitem__",
    "__len__", "__iter__", "__next__", "__contains__", "__enter__", "__exit__", "__setstate__",
    "__getstate__", "__post_init__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__hash__", "__bool__", "__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__",
    "__mul__", "__rmul__", "__imul__", "__truediv__", "__rtruediv__", "__itruediv__",
    "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__", "__pow__", "__rpow__", "__matmul__",
    "__rmatmul__", "__neg__", "__pos__", "__abs__", "__invert__", "__and__", "__rand__", "__or__",
    "__ror__", "__xor__", "__rxor__", "__lshift__", "__rshift__",
})  # fmt: skip
# dunder strings an answer may hold as constants (`if __name__ == "__main__"`, `hasattr(x,
# "__len__")`)
_DUNDER_STRINGS = _DUNDER_READ | {"__main__"}
# library attributes an answer may set: configuration flags
_SETTABLE = ("torch.backends.", "torch._dynamo.config.", "torch._inductor.config.")


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def tamper_scan(
    source: str | bytes, forbidden_modules: frozenset[str], *, within: str | bytes | None = None
) -> str | None:
    """A reason the answer must not be run, read off its source without running it, or None.

    ``forbidden_modules`` is the set of top-level module names that are the judge's own (the
    package, or the flat modules of the private copy): importing any of them is refused, as is
    every way to reach a module without an import statement the scan can read.

    The source is read as the interpreter will read it: as bytes, so a coding declaration applies
    (one naming another encoding than UTF-8 is refused). A source that does not parse is left to
    the run, which fails the same way, unless ``within`` is given: the whole file the run
    executes, the answer's code first (a TritonBench answer runs with the task's test after it).
    Then the answer's code is read inside that file, and checked as far as its own lines go."""
    data = source.encode("utf-8") if isinstance(source, str) else source
    reason = _encoding_reason(data)
    if reason is not None:
        return reason
    last: int | None = None
    try:
        tree = ast.parse(data)
    except (SyntaxError, ValueError):
        if within is None:
            return None
        whole = within.encode("utf-8") if isinstance(within, str) else within
        try:
            tree = ast.parse(whole)
        except (SyntaxError, ValueError):
            return None
        last = data.count(b"\n") + 1
    return _Scan(forbidden_modules, last).run(tree)


def _encoding_reason(data: bytes) -> str | None:
    """A coding declaration other than UTF-8: the interpreter would decode the answer otherwise
    than the scan reads it (UTF-7 hides a line in a comment)."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    except SyntaxError:  # an encoding Python does not know: the run fails on it as well
        return None
    if encoding not in ("utf-8", "utf-8-sig"):
        return f"it declares the source encoding `{encoding}`"
    return None


class _Scan:
    """One pass over an answer's AST: the first rule it breaks, in source order."""

    def __init__(self, forbidden_modules: frozenset[str], last: int | None) -> None:
        self.forbidden = forbidden_modules | _FORBIDDEN_IMPORTS
        self.judge = forbidden_modules
        self.last = last
        self.aliases: dict[str, str] = {}  # a local name bound by an import -> the dotted name
        # a plain name bound to what an import bound, by a simple binding (`w = os`) -> the
        # dotted names it may stand for (`bind`)
        self.bound: dict[str, set[str]] = {}
        self.parents: dict[int, ast.AST] = {}

    def run(self, tree: ast.AST) -> str | None:
        nodes = []
        bindings: list[ast.AST] = []
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[id(child)] = node
            if self.last is not None and getattr(node, "lineno", 0) > self.last:
                continue  # the task's own test, after the answer's code
            nodes.append(node)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    self.aliases[bound] = alias.name if alias.asname else bound
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.arguments)):
                bindings.append(node)
        self.bind(bindings)
        nodes.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
        for node in nodes:
            reason = self.check(node)
            if reason is not None:
                line = getattr(node, "lineno", 0)
                return f"{reason} (line {line})" if line else reason
        return None

    # -- the rules, one per kind of node --

    def check(self, node: ast.AST) -> str | None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return self.imported(node)
        if isinstance(node, ast.Attribute):
            return self.attribute(node)
        if isinstance(node, ast.Name):
            return self.name(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            return self.assigned(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self.defined(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self.string(node.value)
        return None

    def imported(self, node: ast.Import | ast.ImportFrom) -> str | None:
        if isinstance(node, ast.ImportFrom) and node.level:
            # Allowed: an answer is a file of its own, with no package, so a relative import
            # fails when it runs (103 TritonBench answers write one, and get `error`), and the
            # package it would resolve against (`__package__`, `__spec__`, `__name__`) can only
            # be set through a dunder name the scan refuses.
            return None
        names = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [str(node.module)] + [f"{node.module}.{a.name}" for a in node.names]
        )
        for dotted in names:
            parts = dotted.split(".")
            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if prefix in self.judge or parts[0] in self.judge:
                    return f"it imports the judge's own module `{parts[0]}`"
                if prefix in self.forbidden:
                    return f"it imports `{prefix}`"
            reason = self.dotted_reason(dotted, imported=isinstance(node, ast.ImportFrom))
            if reason is not None:
                return reason
        return None

    def dotted_reason(self, dotted: str, imported: bool = False) -> str | None:
        """A name the answer reaches by a dotted path (through its imports): `os.write`,
        `sys.modules`, `types.CodeType`, `torch.load` and the like."""
        head, _, attr = dotted.rpartition(".")
        if not head:
            return None
        if head in ("os", "posix", "nt") and (
            attr in _OS_FUNCTIONS or attr in _FORBIDDEN_ATTRIBUTES
        ):
            return f"it reaches `os.{attr}`"
        if head == "sys" and attr in ("modules", "path", "excepthook", "unraisablehook",
                                      "displayhook", "breakpointhook", "audit"):  # fmt: skip
            return f"it reaches `sys.{attr}`"
        if head == "types" and attr in _TYPES:
            return f"it reaches `types.{attr}`"
        if attr in _FORBIDDEN_ATTRIBUTES and imported:
            return f"it imports `{attr}`"
        return None

    def bind(self, nodes: list[ast.AST]) -> None:
        """Follow the simple bindings of a plain name to a name an import bound, or to an
        attribute chain on one: `w = os`, `w: Any = os`, `(w := os)`, `w, s = os, sys`, a default
        `def f(w=os)`, and chains of them (`v = w` after `w = os`, `p = os.path`). Each such name
        then resolves to every dotted name it may stand for, wherever it is used (`w.write` and
        `getattr(w, "write")` are `os.write`): bindings are taken as a set per name, regardless of
        scope and order, so a name rebound to two modules is checked as both.

        Only these forms are followed. A module reached through any other expression (returned by
        a function or a lambda, held in a container or an object's attribute, picked by a
        conditional) is not resolved here: the run-time integrity check and the channel's token
        are what stand against those (`Integrity`, `ttsem.judge_io`)."""
        pairs: list[tuple[str, ast.AST]] = []
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    pairs += _bound_pairs(target, node.value)
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                pairs += _bound_pairs(node.target, node.value)
            elif isinstance(node, ast.arguments):
                positional = [*node.posonlyargs, *node.args]
                given = positional[len(positional) - len(node.defaults) :]
                pairs += [(a.arg, d) for a, d in zip(given, node.defaults, strict=True)]
                pairs += [
                    (a.arg, d)
                    for a, d in zip(node.kwonlyargs, node.kw_defaults, strict=True)
                    if d is not None
                ]
        for _ in range(8):  # chains of bindings, never deep; a name bound to itself grows no more
            grew = False
            for name, value in pairs:
                found = set(self.resolve(value)) - self.bound.get(name, set())
                if found and len(self.bound.get(name, ())) < 16:
                    self.bound.setdefault(name, set()).update(found)
                    grew = True
            if not grew:
                break

    def resolve(self, node: ast.AST) -> list[str]:
        """Every dotted name an attribute chain may stand for, when its root is a name an import
        bound (`np.searchsorted` -> `numpy.searchsorted`) or a name a simple binding gave one of
        those (`w = os; w.write` -> `os.write`, `bind`); empty for anything else."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return []
        roots = set(self.bound.get(node.id, ()))
        if node.id in self.aliases:
            roots.add(self.aliases[node.id])
        return sorted(".".join([root, *reversed(parts)]) for root in roots)

    def attribute(self, node: ast.Attribute) -> str | None:
        attr = node.attr
        if _is_dunder(attr):
            allowed = _DUNDER_WRITE if isinstance(node.ctx, (ast.Store, ast.Del)) else _DUNDER_READ
            if attr not in allowed:
                return f"it uses the dunder attribute `{attr}`"
            return None
        parent = self.parents.get(id(node))
        if attr == "modules":  # `self.modules()` of an nn.Module is fine, `sys.modules` is not
            if not (isinstance(parent, ast.Call) and parent.func is node):
                return "it reaches a module table (`.modules`)"
        if attr == "ctypes":  # `array.ctypes.data` (an address) is fine, the module is not
            if not (isinstance(parent, ast.Attribute) and parent.attr == "data"):
                return "it reaches `ctypes`"
            return None
        if attr in _FORBIDDEN_ATTRIBUTES:
            return f"it uses `.{attr}`"
        for dotted in self.resolve(node):
            reason = self.dotted_reason(dotted)
            if reason is not None:
                return reason
        return None

    def name(self, node: ast.Name) -> str | None:
        ident = node.id
        if _is_dunder(ident):
            allowed = _DUNDER_NAMES_READ if isinstance(node.ctx, ast.Load) else _DUNDER_NAMES_BOUND
            if ident not in allowed:
                return f"it uses the name `{ident}`"
            return None
        if not isinstance(node.ctx, ast.Load) or ident in self.aliases:
            return None
        if ident in _BUILTINS:
            return f"it uses `{ident}`"
        if ident in _ATTRIBUTE_BUILTINS:
            parent = self.parents.get(id(node))
            if not (isinstance(parent, ast.Call) and parent.func is node):
                return f"it passes `{ident}` around instead of calling it"
        return None

    def call(self, node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _ATTRIBUTE_BUILTINS:
            if func.id in self.aliases:
                return None
            key = node.args[1] if len(node.args) > 1 else None
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return f"it calls `{func.id}` with a name computed at run time"
            name = key.value
            write = func.id in ("setattr", "delattr")
            if _is_dunder(name) and name not in (_DUNDER_WRITE if write else _DUNDER_READ):
                return f"it uses the dunder attribute `{name}` through `{func.id}`"
            if name in _FORBIDDEN_ATTRIBUTES or name == "modules":
                return f"it uses `.{name}` through `{func.id}`"
            for target in self.resolve(node.args[0]) if node.args else []:
                if write and not self.settable(f"{target}.{name}"):
                    return f"it rebinds `{target}.{name}`, a library's own attribute"
                # what `os.write` or `types.CodeType` is refused as, it is refused as through
                # `getattr(os, "write")` too
                reason = self.dotted_reason(f"{target}.{name}")
                if reason is not None:
                    return f"{reason} through `{func.id}`"
            return None
        if isinstance(func, ast.Name) and func.id == "open" and "open" not in self.aliases:
            path = node.args[0] if node.args else None
            if not (isinstance(path, ast.Constant) and isinstance(path.value, (str, bytes))):
                return "it opens a path computed at run time"
        called = self.resolve(func)
        if {"torch.load", "torch.serialization.load"} & set(called):
            safe = any(
                k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value
                for k in node.keywords
            )
            if not safe:
                return "it calls `torch.load` without `weights_only=True` (unpickling)"
        if {"numpy.load", "numpy.lib.npyio.load"} & set(called):
            for k in node.keywords:
                if k.arg == "allow_pickle" and not (
                    isinstance(k.value, ast.Constant) and k.value.value is False
                ):
                    return "it calls `numpy.load` with `allow_pickle` (unpickling)"
        return None

    def settable(self, dotted: str) -> bool:
        return dotted.startswith(_SETTABLE)

    def assigned(self, node: ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Delete) -> str | None:
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        for target in targets:
            for leaf in ast.walk(target):
                if isinstance(leaf, ast.Attribute) and isinstance(leaf.ctx, (ast.Store, ast.Del)):
                    for dotted in self.resolve(leaf):
                        if not self.settable(dotted):
                            return f"it rebinds `{dotted}`, a library's own attribute"
        return None

    def defined(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
        if _is_dunder(node.name) and node.name not in _DUNDER_DEFS:
            return f"it defines `{node.name}`"
        return None

    def string(self, value: str) -> str | None:
        if _is_dunder(value) and value not in _DUNDER_STRINGS:
            return f"it names the dunder `{value}` in a string"
        if "/proc/" in value or value.rstrip("/") == "/proc" or "/dev/mem" in value:
            return "it names a path under `/proc`"
        if value == "--events":
            return "it names the judge's events flag (`--events`)"
        return None


def _bound_pairs(target: ast.AST, value: ast.AST) -> list[tuple[str, ast.AST]]:
    """The (plain name, value) pairs of one binding `target = value`: a name, or a tuple or list
    of them matched element by element against a tuple or list of the same length."""
    if isinstance(target, ast.Name):
        return [(target.id, value)]
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(e, ast.Starred) for e in [*target.elts, *value.elts])
    ):
        return [p for t, v in zip(target.elts, value.elts, strict=True) for p in _bound_pairs(t, v)]
    return []


# --- the runtime integrity check ------------------------------------------------------------------

# the standard modules the check watches whether or not the checker names them: the builtins,
# os.path's implementation, and those every Python program runs through
_ALWAYS = (
    "builtins", "posixpath", "genericpath", "io", "abc", "linecache", "tokenize", "copy",
    "weakref", "enum", "functools", "operator", "itertools", "collections", "collections.abc",
    "types", "typing", "warnings", "codecs", "struct", "numbers", "json", "json.encoder",
    "json.decoder", "json.scanner", "os", "sys", "math", "signal", "threading", "gc",
)  # fmt: skip
# the parts of torch the checker calls: the namespace, the tensor class, the function modes, the
# CUDA stubs the session installs
_TORCH = (
    "torch", "torch._C", "torch._tensor", "torch.storage", "torch.overrides", "torch.cuda",
    "torch.nn.parameter", "torch.utils._python_dispatch", "torch.types",
)  # fmt: skip
# libraries watched whole, every submodule loaded at arm time
_WHOLE = ("numpy", "triton")
# keys a namespace legitimately gains at run time: the warnings registry (put in place at arm)
_GAINED = "__warningregistry__"


class Integrity:
    """A snapshot of the checker, taken before the answer is imported, and a check that nothing in
    it changed since.

    ``watch_package`` names the checker's own code (the directories its modules live in) and
    ``arm`` takes the snapshot: every module-level binding of those modules (all of them, and no
    new one), every class they define (its whole dictionary, its bases), every function (its code,
    defaults and the callables its closure holds); the namespaces of the libraries they import
    (callables, classes and modules: identity), the classes defined there (whole dictionary) and
    their functions' code; the functions ``watch_function`` names, closure by closure (the launch
    hook); and the process state a patch could hide behind (the Python threads, the signal
    handlers, the trace, profile and monitoring hooks, the garbage collector's callbacks, the import
    system's finders and the module table's entries).

    ``arm`` returns the check, a closure that holds everything it reads, so that rebinding a
    method of this class or a name of this module does not change what it does: it returns the
    name of the first thing that changed, or None. Rebinding an attribute, swapping a function's
    code (``f.__code__ = g.__code__``), adding a method that shadows a base class's, replacing a
    module in ``sys.modules`` or starting a thread are all seen; a mutation of a container's
    contents (a cache, a list) is not, unless the container holds callables."""

    def __init__(self) -> None:
        self._explicit: list[tuple[str, Callable[[], Any]]] = []
        self._functions: list[tuple[str, Any]] = []
        self._roots: list[Path] = []
        self._check: Callable[[], str | None] | None = None

    def watch(self, label: str, getter: Callable[[], Any]) -> None:
        """One object by a getter (read again at each check, so a rebinding is seen)."""
        self._explicit.append((label, getter))

    def watch_function(self, label: str, fn: Any) -> None:
        """A function and, closure by closure, the functions it holds (the launch hook)."""
        self._functions.append((label, fn))

    def watch_package(self, *roots: Path) -> None:
        """The checker's own modules: every module whose file lives under one of `roots`."""
        self._roots.extend(Path(r).resolve() for r in roots)

    def arm(self) -> Callable[[], str | None]:
        """Take the snapshot (once every legitimate install is in place, before the answer is
        imported) and return the check."""
        self._check = _Snapshot(self._roots, self._explicit, self._functions).check
        return self._check

    def breach(self) -> str | None:
        """The name of the first watched thing that changed since `arm`, or None."""
        if self._check is None:
            raise RuntimeError("Integrity.breach before arm")
        return self._check()


def _module_file(module: types.ModuleType) -> Path | None:
    name = getattr(module, "__file__", None)
    if not isinstance(name, str):
        return None
    try:
        return Path(name).resolve()
    except (OSError, RuntimeError):
        return None


def _own_modules(roots: list[Path]) -> dict[str, types.ModuleType]:
    """The modules whose file lives under one of `roots`, by name, after importing every module
    of theirs their sources import (a lazy import inside a function included), so the checker
    has no module left to load after the snapshot."""
    seen: set[str] = set()
    while True:
        found = {
            name: module
            for name, module in list(sys.modules.items())
            if isinstance(module, types.ModuleType)
            and (path := _module_file(module)) is not None
            and any(path.is_relative_to(root) for root in roots)
        }
        pending = set()
        for name, module in found.items():
            if name in seen:
                continue
            seen.add(name)
            for target in _imports_of(module):
                if target not in sys.modules and _resolves_under(target, roots):
                    pending.add(target)
        if not pending:
            return found
        for target in sorted(pending):
            try:
                importlib.import_module(target)
            except Exception:  # noqa: BLE001  a module the checker cannot load here is not used
                seen.add(target)


def _imports_of(module: types.ModuleType) -> Iterator[str]:
    """The absolute module names a module's source imports, anywhere in it."""
    path = _module_file(module)
    if path is None or path.suffix != ".py":
        return
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module
            for alias in node.names:
                yield f"{node.module}.{alias.name}"


def _resolves_under(name: str, roots: list[Path]) -> bool:
    """Whether module `name` would load from a file under one of `roots`. Its top-level package
    is looked up first, without importing anything (`find_spec` of a dotted name imports its
    parents: `torch._inductor....` would pull in half of torch)."""

    def under(spec: Any) -> bool:
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not isinstance(origin, str):
            return False
        try:
            path = Path(origin).resolve()
        except (OSError, RuntimeError):
            return False
        return any(path.is_relative_to(root) for root in roots)

    top = name.split(".")[0]
    try:
        loaded = sys.modules.get(top)
        if not (under(loaded.__spec__) if loaded is not None else under(_find(top))):
            return False
        return name == top or under(_find(name))
    except (ImportError, ValueError, AttributeError):
        return False


def _find(name: str) -> Any:
    return importlib.util.find_spec(name)


def _library_modules(own: dict[str, types.ModuleType]) -> dict[str, types.ModuleType]:
    """The library modules the check watches: those the checker's sources import (and the
    standard ones among them loaded now), numpy and Triton whole, the parts of torch it calls,
    and `_ALWAYS`."""
    wanted: set[str] = set(_ALWAYS) | set(_TORCH)
    for module in own.values():
        for target in _imports_of(module):
            parts = target.split(".")
            wanted.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    stdlib = sys.stdlib_module_names
    for name in sorted(wanted):
        top = name.split(".")[0]
        if name not in sys.modules and top in stdlib and top not in ("antigravity", "this"):
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001  not every name an import names is a module
                pass
    found: dict[str, types.ModuleType] = {}
    for name, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType) or name in own or name == "__main__":
            continue
        if name in wanted or name.split(".")[0] in _WHOLE:
            found[name] = module
    return found


def _functions_of(value: Any) -> Iterator[Any]:
    """The plain functions a namespace or class member stands for: itself, a staticmethod's or
    classmethod's function, a property's accessors, a partial's or a cached property's function,
    and what a decorator wrapped (`__wrapped__`, as `functools.wraps` and `lru_cache` keep it)."""
    for _ in range(4):
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        elif isinstance(value, property):
            for accessor in (value.fget, value.fset, value.fdel):
                if isinstance(accessor, types.FunctionType):
                    yield accessor
            return
        elif isinstance(value, (functools.partial, functools.cached_property)):
            value = value.func
        elif isinstance(value, types.FunctionType):
            yield value
            value = value.__dict__.get("__wrapped__")
        elif isinstance(value, _LRU):
            value = value.__wrapped__
        else:
            return


def _is_watched_value(value: Any) -> bool:
    """A library binding worth its identity: a callable, a class or a module (not a counter or a
    cache that changes with use)."""
    return isinstance(value, (type, types.ModuleType)) or callable(value)


class _Snapshot:
    """What `Integrity.arm` records, and the check over it. The check reads only what this object
    captured when it was built, and compares identities with C-level iteration (`itemgetter`,
    `attrgetter`, `map`, `operator.is_`): about 3 ms for some 40,000 watched references."""

    def __init__(
        self,
        roots: list[Path],
        explicit: list[tuple[str, Callable[[], Any]]],
        functions: list[tuple[str, Any]],
    ) -> None:
        own = _own_modules(roots) if roots else {}
        libraries = _library_modules(own) if roots else {}
        # (label, mapping, keys, exact): a module's namespace or a class's dictionary
        self.spaces: list[tuple[str, Any, list[str], bool]] = []
        # id -> (label, function, own): the checker's own functions with their code,
        # defaults and keyword defaults, the libraries' with their code
        self.funcs: dict[int, tuple[str, Any, bool]] = {}
        self.cells: list[tuple[str, Any, Any]] = []
        self.classes: set[int] = set()
        for name, module in sorted(own.items()):
            space = module.__dict__
            space.setdefault(_GAINED, {})  # a warning raised later reuses it instead of adding it
            self.spaces.append((name, space, list(space), True))
            for key, value in list(space.items()):
                self.member(f"{name}.{key}", value, own=True, home=name)
        for name, module in sorted(libraries.items()):
            space = module.__dict__
            keys = [k for k, v in list(space.items()) if _is_watched_value(v)]
            self.spaces.append((name, space, keys, False))
            for key in keys:
                self.member(f"{name}.{key}", space[key], own=False, home=name)
        for label, fn in functions:
            self.closure(label, fn, depth=4)
        own_names = set(own)
        for mode in _function_modes():
            cls = type(mode)
            if getattr(cls, "__module__", None) in own_names:
                self.cls(f"{cls.__module__}.{cls.__qualname__}", cls, own=True)
        modules = {**own, **libraries}
        # `__main__` is the runner, and `runpy` lends its entry to the file it runs
        self.module_names = sorted(n for n in modules if n != "__main__")
        self.modules = [modules[n] for n in self.module_names]
        self.explicit = [(label, get, get(), _code_of(get())) for label, get in explicit]
        self.env = _environment()
        self.check = self._bind()

    # -- building the snapshot --

    def member(self, label: str, value: Any, own: bool, home: str) -> None:
        if isinstance(value, type):
            module = str(getattr(value, "__module__", ""))
            if module == home or (not own and module.split(".")[0] == home.split(".")[0]):
                self.cls(label, value, own)
            return
        for fn in _functions_of(value):
            self.function(label, fn, own)
        if own and not isinstance(value, (types.ModuleType, types.FunctionType)):
            instance_dict = getattr(value, "__dict__", None)
            if isinstance(instance_dict, dict) and type(value).__module__ == home:
                self.spaces.append((label, instance_dict, list(instance_dict), True))

    def cls(self, label: str, cls: type, own: bool) -> None:
        if id(cls) in self.classes or not (cls.__flags__ & _HEAPTYPE):
            return  # seen already, or a built-in type (immutable)
        self.classes.add(id(cls))
        proxy = cls.__dict__
        self.spaces.append((label, proxy, list(proxy), True))
        for key in list(proxy):
            value = proxy[key]
            for fn in _functions_of(value):
                self.function(f"{label}.{key}", fn, own)
            if isinstance(value, type):
                self.cls(f"{label}.{key}", value, own)
        self.cells.append((f"{label}.__bases__", _BasesCell(cls), cls.__bases__))

    def function(self, label: str, fn: Any, own: bool) -> None:
        seen = self.funcs.get(id(fn))
        if seen is None or (own and not seen[2]):
            self.funcs[id(fn)] = (label, fn, own)
        if own:
            self.cells_of(label, fn, depth=1)

    def closure(self, label: str, fn: Any, depth: int) -> None:
        """A function watched with the callables its closure holds, `depth` closures deep."""
        if isinstance(fn, types.FunctionType):
            self.funcs[id(fn)] = (label, fn, True)
            self.cells_of(label, fn, depth)

    def cells_of(self, label: str, fn: Any, depth: int) -> None:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=False):
            try:
                content = cell.cell_contents
            except ValueError:
                continue
            if not _is_watched_value(content):
                continue
            self.cells.append((f"{label}.<{name}>", cell, content))
            if depth > 1:
                self.closure(f"{label}.<{name}>", content, depth - 1)

    # -- the check --

    def _bind(self) -> Callable[[], str | None]:
        """The check as a closure over this snapshot's own lists, with every builtin it calls
        bound as a default argument: nothing it does looks a name up when it runs. The fast path
        compares everything at C speed; only a difference walks the lists to name it."""
        getters: list[Any] = []
        maps: list[Any] = []
        want: list[Any] = []
        for _label, mapping, keys, _exact in self.spaces:
            if not keys:
                continue
            picked = keys if len(keys) > 1 else keys * 2  # itemgetter of one key is no tuple
            getters.append(operator.itemgetter(*picked))
            maps.append(mapping)
            want.extend(mapping[k] for k in picked)
        exact_maps = [mapping for _, mapping, _, exact in self.spaces if exact]
        exact_labels = [label for label, _, _, exact in self.spaces if exact]
        exact_lens = [len(m) for m in exact_maps]
        mine = [(label, fn) for label, fn, own in self.funcs.values() if own]
        theirs = [(label, fn) for label, fn, own in self.funcs.values() if not own]
        funcs = [fn for _, fn in mine]
        func_want = list(chain.from_iterable(map(_FUNCTION, funcs)))
        lib_funcs = [fn for _, fn in theirs]
        lib_want = list(map(_CODE, lib_funcs))
        cell_list = [cell for _, cell, _ in self.cells]
        cell_want = [content for _, _, content in self.cells]
        cell_labels = [label for label, _, _ in self.cells]
        module_names, module_want = self.module_names, self.modules
        explicit, env, spaces = self.explicit, self.env, self.spaces
        modules = sys.modules

        def named() -> str:
            """The first difference, found the slow way (the fast path saw one)."""
            index = 0
            for label, mapping, keys, _exact in spaces:
                for key in keys if len(keys) != 1 else keys * 2:
                    if mapping.get(key, _MISSING) is not want[index]:
                        return f"{label}.{key}"
                    index += 1
            for label, mapping, length in zip(exact_labels, exact_maps, exact_lens, strict=True):
                if len(mapping) != length:
                    return f"{label} (a name added or removed)"
            for index, (label, fn) in enumerate(mine):
                got = _FUNCTION(fn)
                if any(got[k] is not func_want[3 * index + k] for k in range(3)):
                    return label
            for (label, fn), code in zip(theirs, lib_want, strict=True):
                if _CODE(fn) is not code:
                    return label
            for label, cell, content in zip(cell_labels, cell_list, cell_want, strict=True):
                try:
                    if cell.cell_contents is not content:
                        return label
                except ValueError:
                    return label
            for name, module in zip(module_names, module_want, strict=True):
                if modules.get(name) is not module:
                    return f"sys.modules[{name!r}]"
            return "the checker"

        def check(
            _all: Any = all,
            _map: Any = map,
            _is: Any = operator.is_,
            _call: Any = _apply,
            _flat: Any = chain.from_iterable,
            _len: Any = len,
            _list: Any = list,
            _function: Any = _FUNCTION,
            _code: Any = _CODE,
            _contents: Any = _CONTENTS,
            _mget: Any = modules.get,
            _code_of: Any = _code_of,
            _named: Any = named,
        ) -> str | None:
            try:
                same = (
                    _all(_map(_is, _flat(_map(_call, getters, maps)), want))
                    and _list(_map(_len, exact_maps)) == exact_lens
                    and _all(_map(_is, _flat(_map(_function, funcs)), func_want))
                    and _all(_map(_is, _map(_code, lib_funcs), lib_want))
                    and _all(_map(_is, _map(_contents, cell_list), cell_want))
                    and _all(_map(_is, _map(_mget, module_names), module_want))
                )
            except Exception:  # noqa: BLE001  a name removed, a cell emptied
                same = False
            if not same:
                return _named()
            for label, get, obj, code in explicit:
                try:
                    value = get()
                except Exception:  # noqa: BLE001  the attribute itself was removed
                    return label
                if value is not obj or _code_of(value) is not code:
                    return label
            for label, read, now in env:
                if read() != now:
                    return label
            return None

        return check


def _apply(getter: Any, mapping: Any) -> Any:
    """`getter(mapping)`, as a named function (`operator.call` is 3.11+; the private copy floors
    at 3.10). One trampoline per namespace; the itemgetter inside it runs at C speed."""
    return getter(mapping)


_MISSING = object()
_LRU = type(functools.lru_cache(maxsize=None)(lambda: None))
_HEAPTYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE: a class whose dictionary Python code can change
_FUNCTION = operator.attrgetter("__code__", "__defaults__", "__kwdefaults__")
_CODE = operator.attrgetter("__code__")
_CONTENTS = operator.attrgetter("cell_contents")


class _BasesCell:
    """A class's bases, read like a closure cell, so the check compares them with the rest."""

    __slots__ = ("cls",)

    def __init__(self, cls: type) -> None:
        self.cls = cls

    @property
    def cell_contents(self) -> Any:
        return self.cls.__bases__


def _code_of(obj: Any) -> Any:
    """The code object of a function, or None for a class or a builtin (identity carries those)."""
    return getattr(obj, "__code__", None)


def _function_modes() -> list[Any]:
    """The torch function modes in force now (the session's own), if torch is loaded."""
    overrides = sys.modules.get("torch.overrides")
    stack = getattr(overrides, "_get_current_function_mode_stack", None)
    try:
        return list(stack()) if callable(stack) else []
    except Exception:  # noqa: BLE001  an older torch
        return []


def _environment() -> list[tuple[str, Callable[[], Any], Any]]:
    """The process state a patch could hide behind, each as (label, read it now, its value at
    arm time): the Python threads, the signal handlers, the trace, profile and monitoring hooks,
    the garbage collector's callbacks, the import system's finders and path hooks."""
    current_frames = sys._current_frames
    getsignal = signal.getsignal
    numbers = sorted(int(s) for s in signal.valid_signals())

    def threads(_frozenset: Any = frozenset) -> frozenset[int]:
        return _frozenset(current_frames())

    def handlers(_getsignal: Any = getsignal, _id: Any = id) -> list[int]:
        found = []
        for number in numbers:
            try:
                found.append(_id(_getsignal(number)))
            except (OSError, ValueError):
                found.append(0)
        return found

    def ids_of(items: Any) -> Callable[[], list[int]]:
        def read(_list: Any = list, _map: Any = map, _id: Any = id) -> list[int]:
            return _list(_map(_id, items))

        return read

    def meta_path(_sys: Any = sys, _id: Any = id) -> int:
        return _id(_sys.meta_path)

    def module_table(_sys: Any = sys, _id: Any = id) -> int:
        return _id(_sys.modules)

    reads: list[tuple[str, Callable[[], Any]]] = [
        ("a thread started after the snapshot", threads),
        ("a signal handler", handlers),
        ("sys.settrace", sys.gettrace),
        ("sys.setprofile", sys.getprofile),
        ("gc.callbacks", ids_of(gc.callbacks)),
        ("sys.meta_path", ids_of(sys.meta_path)),
        ("sys.path_hooks", ids_of(sys.path_hooks)),
        ("the list sys.meta_path", meta_path),
        ("the dict sys.modules", module_table),
    ]
    monitoring = getattr(sys, "monitoring", None)  # 3.12+: the low-level monitoring tools
    if monitoring is not None:
        get_tool, get_events, tools = monitoring.get_tool, monitoring.get_events, range(6)

        def monitors(_get_tool: Any = get_tool, _get_events: Any = get_events) -> list[Any]:
            return [(_get_tool(t), _get_events(t)) for t in tools]

        reads.append(("sys.monitoring", monitors))
    return [(label, read, read()) for label, read in reads]


def enforcer(
    check: Callable[[], str | None], stop: Callable[..., Any], event: dict[str, Any]
) -> Callable[[], None]:
    """`check` bound to the channel's own `stop`: on a breach, `event` with the name of what
    changed goes to the judge at once and the process ends, through nothing the answer could
    have rebound since the snapshot (no exception is raised into code that could catch it)."""

    frozen = dict(event)

    def enforce(_check: Any = check, _stop: Any = stop) -> None:
        bad = _check()
        if bad is not None:
            _stop({**frozen, "message": "the answer changed the judge's own code (" + bad + ")"})

    return enforce


def sealer(enforce: Callable[[], None]) -> Callable[[], None]:
    """The last check of a run, before its account is written and its `done` sent: the garbage
    collected now (a finalizer of the answer's runs here, before the check, or never), the
    collector left off, then `enforce`. Nothing of the answer's may run after it: the runner
    writes only plain data it built before."""

    def seal(
        _collect: Any = gc.collect, _disable: Any = gc.disable, _enforce: Any = enforce
    ) -> None:
        _collect()
        _disable()
        _enforce()

    return seal


def checker_integrity(*, roots: Iterable[Path], hooks: dict[str, Any] | None = None) -> Integrity:
    """The standard snapshot for a judge's runner: the checker's own modules (every module under
    `roots`: the package, or the private copy's directories), the libraries they use, and the
    launch hook (`hooks`: label -> the function installed on `JITFunction.run` and the like, each
    watched with the callables its closures hold). Call ``arm`` once every legitimate install is
    in place (the session hook and any wrapper on it) and before the answer is imported."""
    guard = Integrity()
    guard.watch_package(*roots)
    for label, fn in (hooks or {}).items():
        guard.watch_function(label, fn)
    return guard
