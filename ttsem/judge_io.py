"""What a judge's sandboxed run hands back to the judge, and how the judge reads it.

The process that runs an answer belongs to the answer: its code can catch any exception, register
exit handlers, and write any file its sandbox lets it write. So:

- a kernel fault, or a launch the semantics cannot judge, is sent at once on a pipe the runner was
  given before the answer was imported, and the process ends there (`Channel.stop`): the answer
  cannot catch it, and cannot take back what was sent;
- a run that ends normally sends `done` and ends with `os._exit` too, so no exit handler of the
  answer's runs after the runner's last word;
- a run writes raw outputs (tensors, counts), never a verdict: the judge compares them with a
  reference computed in another sandbox (`compare`), and reads every file a run wrote as
  untrusted (`read_untrusted`, `load_npz`: a regular file of bounded size, no link, no pickle,
  sizes checked before anything is allocated).

What this does not stop: code that patches the runner inside its own process (the launch counts
it reports, the channel itself) can still claim launches it never made, or send a `stop` of its
own choosing. The values it returns are still compared with a reference it cannot touch.

Imports only the standard library and numpy at module level: the judge's process reads its runs
with it, the runners write with it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, NoReturn

import numpy as np

# the exit status of a run that a fault or an unjudged launch stopped
EXIT_STOPPED = 3
# how much of a run the judge reads back: its events, a JSON report, its arrays
EVENTS_LIMIT = 1 << 20
REPORT_LIMIT = 64 << 20
ARRAYS_LIMIT = 4 << 30

# --- the runner's side ----------------------------------------------------------------------------


class Channel:
    """The runner's end of the pipe to the judge (`--events FD`), and the way out of the process.
    Both are taken when the runner starts, before any code under test is imported, so that a
    later `os.write = ...` or `os._exit = ...` of that code does not reach them."""

    def __init__(self, fd: int | None) -> None:
        self.fd = fd
        self._write, self._exit = os.write, os._exit

    def send(self, event: Mapping[str, Any]) -> None:
        """One event, as a line of JSON. The newline before it ends whatever partial line the
        code under test may have written to the same pipe."""
        if self.fd is None:
            return
        data = b"\n" + json.dumps(dict(event), default=str).encode() + b"\n"
        with contextlib.suppress(OSError):
            while data:
                data = data[self._write(self.fd, data) :]

    def finish(self, code: int = 0) -> NoReturn:
        """`done`, then out of the process without running anything else (no exit handler, no
        finalizer, no thread of the code under test)."""
        self.send({"event": "done"})
        self.end(code)

    def stop(self, event: Mapping[str, Any]) -> NoReturn:
        """A `stop` event (a fault, a launch that cannot be judged), then out of the process."""
        self.send({**event, "event": "stop"})
        self.end(EXIT_STOPPED)

    def end(self, code: int) -> NoReturn:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        self._exit(code)


def tensor_bytes(value: Any) -> tuple[dict[str, Any], np.ndarray | None]:
    """How one output travels: a tensor as its dtype, its shape and its bytes; anything else as
    the name of its type (`compare` takes it for "not a tensor")."""
    import torch

    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}, None
    name = str(value.dtype).removeprefix("torch.")
    try:
        flat = value.detach().cpu().contiguous().reshape(-1)
        data = flat.view(torch.uint8).numpy().copy()
    except (RuntimeError, TypeError, ValueError) as exc:  # a sparse or quantized tensor, a meta one
        return {"type": f"Tensor ({name}: {type(exc).__name__})"}, None
    if name not in DTYPES:
        return {"type": f"Tensor ({name})"}, None
    return {"dtype": name, "shape": [int(n) for n in value.shape]}, data


# the dtypes an output may come back with (a name from the run is looked up only among these)
DTYPES = (
    "float64", "float32", "float16", "bfloat16", "complex128", "complex64", "complex32",
    "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8", "bool",
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu",
)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class NotATensor:
    """An output that was not a tensor, by the name of its type."""

    name: str


def tensor_of(spec: Mapping[str, Any], arrays: Mapping[str, np.ndarray], key: str) -> Any:
    """An output back from `tensor_bytes`'s description and the bytes stored under `key`."""
    import torch

    if "dtype" not in spec:
        return NotATensor(str(spec.get("type", "?")))
    name, shape = str(spec["dtype"]), [int(n) for n in spec["shape"]]
    if name not in DTYPES or any(n < 0 for n in shape):
        raise ValueError(f"{key}: dtype {name[:40]!r}, shape {shape[:8]}")
    dtype = getattr(torch, name)
    data = arrays[key]
    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if data.dtype != np.uint8 or data.size != size:
        raise ValueError(f"{key}: {data.size} bytes for {size}")
    if size == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(data.tobytes()), dtype=dtype).reshape(shape)


# --- the judge's side -----------------------------------------------------------------------------


@dataclasses.dataclass
class Ran:
    """A sandboxed run as the judge sees it: its exit status (None: killed at the timeout), the
    last line of its stderr, and the events it sent."""

    code: int | None
    tail: str
    events: list[dict[str, Any]]

    def stop(self) -> dict[str, Any] | None:
        """The first `stop` the run sent: final, whatever came after it."""
        return next((e for e in self.events if e.get("event") == "stop"), None)

    def done(self) -> bool:
        return any(e.get("event") == "done" for e in self.events)


def run_sandboxed(cmd: Sequence[str], env: Mapping[str, str], cwd: Path, timeout: int) -> Ran:
    """`cmd` (a runner that takes `--events FD`) in its own process with a pipe for its events:
    stdin closed (the run reads nothing of the terminal), stdout dropped, stderr kept in a file of
    the judge's own (only its last line is read), killed at `timeout`."""
    read_end, write_end = os.pipe()
    try:
        with tempfile.TemporaryFile() as err:
            try:
                done = subprocess.run(
                    [*cmd, "--events", str(write_end)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=err,
                    env=dict(env),
                    cwd=cwd,
                    timeout=timeout,
                    pass_fds=(write_end,),
                    check=False,
                )
                code: int | None = done.returncode
            except subprocess.TimeoutExpired:
                code = None
            finally:
                os.close(write_end)
            tail = last_line(err)
        return Ran(code, tail, events_of(drain(read_end)))
    finally:
        os.close(read_end)


def last_line(file: IO[bytes]) -> str:
    """The last non-empty line of `file`, from its last 8 KiB."""
    size = file.seek(0, os.SEEK_END)
    file.seek(max(0, size - 8192))
    lines = file.read().decode("utf-8", "replace").strip().splitlines()
    return lines[-1][:300] if lines else ""


def drain(fd: int) -> bytes:
    """What is in the pipe, without waiting for a writer that may still hold it open (a process
    the code under test left behind, outside the sandbox's pid namespace)."""
    os.set_blocking(fd, False)
    data = bytearray()
    with contextlib.suppress(BlockingIOError):
        while len(data) < EVENTS_LIMIT:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            data += chunk
    return bytes(data[:EVENTS_LIMIT])


def events_of(data: bytes) -> list[dict[str, Any]]:
    """The events in what a run sent: one JSON object per line, anything else skipped."""
    found = []
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(event, dict) and isinstance(event.get("event"), str):
            found.append(event)
    return found


def read_untrusted(path: Path, limit: int = REPORT_LIMIT) -> bytes | None:
    """The bytes of `path`, a file a sandboxed run may have written, or None where it is not a
    regular file of at most `limit` bytes (missing, a link, a fifo, a device, too big)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            return None
        return file.read(limit)


def read_json(path: Path, limit: int = REPORT_LIMIT) -> Any:
    """A JSON file a sandboxed run may have written, or None where it does not read as JSON."""
    data = read_untrusted(path, limit)
    if data is None:
        return None
    try:
        return json.loads(data)
    except (ValueError, RecursionError):
        return None


def load_npz(path: Path, limit: int = ARRAYS_LIMIT) -> dict[str, np.ndarray]:
    """The arrays of an .npz a sandboxed run may have written, read as untrusted: a regular file
    (not a link) of at most `limit` bytes, every member stored uncompressed (as `np.savez` stores
    it), each member's header checked against its size before anything is allocated, and no
    object arrays. ValueError, OSError or `zipfile.BadZipFile` otherwise."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"{path.name}: not a regular file of at most {limit} bytes")
        arrays: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(file) as archive:
            for member in archive.infolist():
                stored = member.compress_type == zipfile.ZIP_STORED
                if not stored or max(member.file_size, member.compress_size) > info.st_size:
                    raise ValueError(f"{member.filename[:40]}: not stored as np.savez stores it")
                with archive.open(member) as stream:
                    arrays[member.filename.removesuffix(".npy")] = read_npy(stream, member)
    return arrays


def read_npy(stream: IO[bytes], member: zipfile.ZipInfo) -> np.ndarray:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"{member.filename[:40]}: .npy version {version}")
    if dtype.hasobject or dtype.itemsize == 0:
        raise ValueError(f"{member.filename[:40]}: dtype {dtype}")
    size = math.prod(shape) * dtype.itemsize
    if size > member.file_size:
        raise ValueError(f"{member.filename[:40]}: {size} bytes declared in {member.file_size}")
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"{member.filename[:40]}: cut short")
    return np.frombuffer(data, dtype=dtype).reshape(shape, order="F" if fortran else "C")


_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def printable(text: str) -> str:
    """`text` with its control characters written as escapes: what a run says (an exception's
    message, a kernel's source line) is the answer's to choose, and a terminal takes some
    control sequences as commands."""
    return _CONTROL.sub(lambda m: ascii(m.group())[1:-1], text)


# --- the comparison -------------------------------------------------------------------------------

V3_TOLERANCE = {  # KernelBench-v3 `src/eval/benchmark.py`, by the dtype of the first input
    "fp4": (0.5, 0.1),
    "fp8": (0.1, 0.05),
    "fp16": (0.01, 0.01),
    "bf16": (0.01, 0.01),
    "fp32": (0.001, 0.001),
}


def v3_precision(inputs: Sequence[Any]) -> str:
    """The KernelBench-v3 harness's precision of a trial: the dtype of its first tensor input."""
    import torch

    for value in inputs:
        if isinstance(value, torch.Tensor):
            text = str(value.dtype)
            for key, name in (("float8", "fp8"), ("bfloat16", "bf16"), ("float16", "fp16")):
                if key in text:
                    return name
            return "fp32" if "float32" in text else text.replace("torch.", "")
    return "fp32"


def compare(
    want: Any, got: Any, precision: str, rule: str, atol: float = 1e-2, rtol: float = 1e-2
) -> dict[str, Any]:
    """One trial under the chosen rule (`pass`), with the other rules' answers alongside:
    KernelBench's elementwise `torch.allclose` at 1e-2 (v0.1) and 1e-4 (its fp32 default now), and
    KernelBench-v3's `max|diff| < atol + rtol * max|ref|` with finite outputs, at the tolerance of
    `precision` (`v3_precision` of the trial's inputs)."""
    import torch

    row: dict[str, Any] = {}
    if not isinstance(want, torch.Tensor) or not isinstance(got, torch.Tensor):
        row.update(pass_=False, why="not a tensor")
    elif got.shape != want.shape:
        row.update(pass_=False, shape=[list(want.shape), list(got.shape)])
    else:
        w, g = want.float(), got.float()
        diff = (w - g).abs()
        max_diff = float(diff.max()) if diff.numel() else 0.0
        max_ref = float(w.abs().max()) if w.numel() else 0.0
        v3_atol, v3_rtol = V3_TOLERANCE.get(precision, (0.05, 0.02))
        finite = bool(torch.isfinite(w).all()) and bool(torch.isfinite(g).all())
        row.update(
            max_abs_diff=max_diff,
            max_ref=max_ref,
            ref_nan=bool(torch.isnan(w).any()),
            allclose=bool(torch.allclose(want, got, atol=atol, rtol=rtol)),
            allclose_1e4=bool(torch.allclose(want, got, atol=1e-4, rtol=1e-4)),
            v3=finite and max_diff < v3_atol + v3_rtol * max_ref,
        )
        row["pass_"] = row["v3"] if rule == "v3" else row["allclose"]
    row["pass"] = row.pop("pass_")
    return row
