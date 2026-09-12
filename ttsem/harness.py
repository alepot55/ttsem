"""Capture kernel launches and check the level-1 semantics against them.

For every ``kernel[grid](...)`` call in a fuzzer-style program (see ``tests/fixtures/``),
:func:`capture_launches` records the JIT function, the grid, the host copies of every tensor
argument before and after the launch, and the device address of each: the pair (pre, post) is
the reference this package's own :class:`ttsem.interp.Interp` is checked against.

Two details make this work without a GPU:

* on a machine without a GPU (``--device cpu``), a real Triton compile still needs an active
  driver to ask for a target; :mod:`ttsem._fakedriver`'s ``_FakeDriver`` supplies exactly
  enough of one, and its ``_stub_torch_cuda`` answers the ``torch.cuda`` queries some backend
  code makes. The launch itself cannot run on real hardware, so it runs through Triton's own
  interpreter instead (``TRITON_INTERPRET=1`` semantics) and *that* result stands in for "the
  device". This needs **two** fresh imports of the same program (``_capture_interpreted`` then
  ``_capture_compilable``): a kernel that calls another ``@triton.jit``'d function inline (a
  ``tl.associative_scan`` combine_fn, say) only interprets correctly when the whole module was
  decorated under ``TRITON_INTERPRET=1`` from the start (every nested jit'd function has to be
  an ``InterpretedFunction`` too, not patched in after the fact), which rules out compiling and
  interpreting the very same ``JITFunction`` object a one-pass "unified mode" would rely on; a
  plain, undecorated second import gets a real, compilable ``JITFunction`` for the same
  launches instead.
* compiling a launch exactly as it happened means reusing the specialization the concrete
  arguments imply; :func:`_compile_asm` does that and keeps the whole ``asm`` dict, because
  ``ir_for_launch`` needs ttir as well as ttgir/llir/ptx.

``_bits``/``compare`` below are deliberately simple: comparison is bitwise, and floats are
compared by bit pattern.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

# The wheel is needed to *record* a launch and to compile it, not to run the semantics over a
# module that was recorded earlier. Guarding the import the same way the level-1 modules below
# are guarded keeps `compare`, `split_dump`, `culprit_of` and the rest of the pure bookkeeping
# importable with nothing but numpy installed; `_require_triton` fails loudly in the few
# functions that really do need a compiler.
try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend
    from triton.runtime import driver as _driver
    from triton.runtime.jit import JITFunction, create_function_from_signature

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - the triton-free path is a whole CI job, not a branch
    triton = None  # type: ignore[assignment]
    GPUTarget = None  # type: ignore[assignment, misc]
    ASTSource = None  # type: ignore[assignment, misc]
    make_backend = None  # type: ignore[assignment]
    _driver = None  # type: ignore[assignment]
    JITFunction = None  # type: ignore[assignment, misc]
    create_function_from_signature = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


def _require_triton() -> None:
    """Raise where a Triton install is the actual missing piece, not ten frames later."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "this needs a Triton install: recording or compiling a launch goes through the "
            "wheel. Install the extra with `pip install 'ttsem[triton]'`. Running an already "
            "recorded module through the semantics needs numpy only."
        )


try:
    from ttsem import mlir  # type: ignore[import-not-found]
except ImportError:
    mlir = None  # type: ignore[assignment]

try:
    from ttsem.memory import Memory  # type: ignore[import-not-found]
except ImportError:
    Memory = None  # type: ignore[assignment, misc]

try:
    from ttsem.interp import Interp, Unsupported  # type: ignore[import-not-found]
except ImportError:
    Interp = None  # type: ignore[assignment, misc]

    class Unsupported(Exception):  # type: ignore[no-redef]
        """Placeholder used only while ``interp.py`` does not exist yet."""


try:
    from ttsem import values  # type: ignore[import-not-found]
except ImportError:
    values = None  # type: ignore[assignment]

SEMANTICS_AVAILABLE = mlir is not None and Memory is not None and Interp is not None

DEFAULT_CC = 90
LAUNCH_KWARG_KEYS = ("grid", "warmup")


@dataclasses.dataclass
class LaunchRecord:
    """One ``kernel[grid](...)`` call, with enough state to recompile and re-run it."""

    launch_id: int
    fn: JITFunction
    fn_name: str
    fn_file: str
    grid: tuple[Any, ...]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    bound: dict[str, Any]  # every kernel parameter, by name, as passed at the call site
    pre: dict[str, np.ndarray]  # tensor arguments, host copy, before the launch
    post: dict[str, np.ndarray]  # tensor arguments, host copy, after the launch
    ptrs: dict[str, int]  # tensor arguments, device address at launch time
    bases: dict[str, int] = dataclasses.field(default_factory=dict)  # storage base per tensor
    elems: dict[str, str] = dataclasses.field(default_factory=dict)  # torch dtype per tensor


@dataclasses.dataclass
class Comparison:
    """The verdict of running one launch's IR against its recorded (pre, post) state."""

    verdict: str  # "match" | "mismatch" | "unsupported" | "error"
    n_diff: int
    diffs: list[dict[str, Any]]
    unsupported: list[str]
    message: str = ""
    reads: Any = dataclasses.field(default=None, repr=False)  # byte addresses loaded, if traced

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["reads"] = None if self.reads is None else int(len(self.reads))
        return d


def _bits(x: np.ndarray) -> np.ndarray:
    """Floats compared by their bit pattern: a signed zero or a NaN payload counts too.

    (See the module docstring for why this is
    copied rather than imported).
    """
    return x.view(f"u{x.dtype.itemsize}") if np.issubdtype(x.dtype, np.floating) else x


def compare(
    ref: np.ndarray,
    dev: np.ndarray,
    max_diffs: int = 32,
    policy: FloatPolicy | None = None,
) -> dict[str, Any]:
    """Compare two flat buffers (no padding layout here:
    level-1 buffers are compared whole).

    The comparison is always bitwise first, so ``equal`` keeps meaning *bit for bit*. When it
    fails and the buffer holds floats, ``policy`` says which of the differences the device is
    allowed to have: the result then carries ``approx`` (every difference is explained),
    ``max_rel_err`` and, for a buffer of a native numpy float dtype, ``max_ulp``.
    """
    if ref.shape != dev.shape or ref.dtype != dev.dtype:
        return {"equal": False, "n_diff": -1, "diffs": [], "note": "shape or dtype differs"}
    where = np.nonzero(_bits(ref) != _bits(dev))[0]
    diffs = [[int(p), ref[p].item(), dev[p].item()] for p in where[:max_diffs]]
    out: dict[str, Any] = {"equal": len(where) == 0, "n_diff": int(len(where)), "diffs": diffs}
    if not len(where):
        return out
    if np.issubdtype(ref.dtype, np.floating):
        out["max_ulp"] = int(_ulp_distance(ref[where], dev[where]).max())
    whole = _decode_floats(ref, policy)
    fref = _decode_floats(ref[where], policy)
    fdev = _decode_floats(dev[where], policy)
    if whole is None or fref is None or fdev is None:
        return out  # integer and boolean buffers stay bitwise, with no way to be approximate
    finite = whole[np.isfinite(whole)]
    scale = float(np.abs(finite).max()) if finite.size else 1.0
    explained, max_rel, max_abs = _explain_float_diffs(fref, fdev, max(scale, 1.0), policy)
    out["max_rel_err"] = max_rel
    out["max_abs_err"] = max_abs
    out["approx"] = explained
    return out


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Units in the last place between same-dtype floats; huge for a NaN on either side.

    The bit patterns are mapped to a monotone *unsigned* key rather than to a signed one:
    ``1 << 63`` does not fit in an ``int64``, so the signed form silently gave nonsense for
    ``f64``. Both zeros are one key apart, which is IEEE's total order.
    """
    bits = a.dtype.itemsize * 8
    sign = np.uint64(1) << np.uint64(bits - 1)
    ua = _bits(a).astype(np.uint64)
    ub = _bits(b).astype(np.uint64)
    mask = np.uint64((1 << bits) - 1)
    ka = np.where(ua & sign, (~ua) & mask, ua | sign)
    kb = np.where(ub & sign, (~ub) & mask, ub | sign)
    d = (np.maximum(ka, kb) - np.minimum(ka, kb)).astype(np.uint64)
    nan = np.isnan(a) | np.isnan(b)
    return np.where(nan, np.uint64(1) << np.uint64(40), d).astype(np.int64)


APPROX_ULP = 2  # `arith.divf` on NVIDIA is `div.full.f32`, within 2 ulp of the rounded quotient


# --------------------------------------------------------- the float comparison policy

# Every op whose device result the level-1 semantics cannot be expected to reproduce bit for
# bit, and the class of licence it takes. `"div"` is a bounded number of ulp; `"wide"` is a
# relative tolerance, because a reassociated sum or an approximate instruction is off by an
# amount no ulp count bounds. An op absent from this table is exact on both sides, and a
# buffer written by a launch of only such ops is compared bitwise: that is what keeps the
# exactness picture visible instead of dissolving it into tolerances.
INEXACT_OPS: dict[str, str] = {
    # A tensor-core dot sums the products in a hardware reduction tree whose shape is the
    # lowering's choice, and f16/bf16/tf32 operands round before the multiply: hundreds of
    # ulp in an f16 result are normal, not a defect.
    "tt.dot": "wide",
    # The same tree, plus per-block scales.
    "tt.dot_scaled": "wide",
    # A reduction lowers to a butterfly over lanes, not to the left fold this semantics runs,
    # and float addition is not associative.
    "tt.reduce": "wide",
    # A scan lowers to a Kogge-Stone tree, for the same reason.
    "tt.scan": "wide",
    # Float atomics compose in the order the lanes happen to reach the address.
    "tt.atomic_rmw": "wide",
    # libdevice is specified to a few ulp, not to correct rounding.
    "tt.extern_elementwise": "wide",
    # `math.fma` is one fused rounding on the device and two roundings here.
    "math.fma": "wide",
    # These lower to the approximate SASS instructions (`ex2.approx`, `lg2.approx`,
    # `sin.approx`, `rsqrt.approx`, ...), whose error is a few ulp of the result.
    "math.exp": "wide",
    "math.exp2": "wide",
    "math.log": "wide",
    "math.log2": "wide",
    "math.sin": "wide",
    "math.cos": "wide",
    "math.tanh": "wide",
    "math.erf": "wide",
    "math.sqrt": "wide",
    "math.rsqrt": "wide",
    # `arith.divf` lowers to `div.full.f32`, which NVIDIA documents as within 2 ulp of the
    # correctly rounded quotient; `tt.precise_divf` is the rounded one on both sides.
    "arith.divf": "div",
}

# rtol and atol per element type, for a buffer of a launch that contains a `"wide"` op. The
# narrow types get a band a hundred times looser because a single rounding of theirs already
# is: one ulp of bf16 is 0.8% of the value.
WIDE_TOLERANCE: dict[str, tuple[float, float]] = {
    "f16": (1e-2, 1e-2),
    "bf16": (1e-2, 1e-2),
    "f8E4M3FN": (1e-2, 1e-2),
    "f8E5M2": (1e-2, 1e-2),
    # a reassociated sum of thousands of terms (a split scan's carry, a 4096-wide row) leaves an
    # absolute error of a few hundred ulp of the terms it cancelled, which may dwarf the result
    "f32": (1e-4, 1e-4),
    "f64": (1e-4, 1e-4),
}

# An upper bound on the relative spacing of each float type: one ulp is at most this fraction
# of the value. It turns an ulp count into a relative tolerance, which is what an ulp count
# means and, unlike a bit-pattern distance, does not change when the result is widened on the
# way to memory (`test_bin_op` with `/` stores an f32 quotient into an f64 buffer, where one
# f32 ulp is 2**29 f64 ulp).
ULP_AS_RTOL: dict[str, float] = {
    "f8E4M3FN": 2.0**-3,
    "f8E5M2": 2.0**-2,
    "bf16": 2.0**-7,
    "f16": 2.0**-10,
    "f32": 2.0**-23,
    "f64": 2.0**-52,
}


@dataclasses.dataclass(frozen=True)
class FloatPolicy:
    """How much one buffer of a launch is allowed to differ from the device.

    `elem` is the Triton spelling of the element type, which is the only thing that says a
    `uint16` buffer is really `bf16` (bf16 and fp8 travel as bit patterns). A difference is
    allowed when it is within `rtol` relatively or within `atol` times the buffer's own
    largest magnitude. Both zero means "bitwise", and then only two NaNs still compare equal.
    """

    elem: str = ""
    rtol: float = 0.0
    atol: float = 0.0


def _float_type(elem: str) -> Any:
    return values.Type(kind="float", shape=None, elem=None, width=None, name=elem, encoding=None)


def _decode_floats(x: np.ndarray, policy: FloatPolicy | None) -> np.ndarray | None:
    """A buffer's elements as numpy floats, or None when the buffer does not hold floats."""
    elem = policy.elem if policy is not None else ""
    if values is not None and elem in values.FLOAT_FORMATS and elem not in ("f16", "f32", "f64"):
        return np.asarray(values.to_float(x, _float_type(elem)), dtype=np.float64)
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float64)
    return None


def _explain_float_diffs(
    fref: np.ndarray,
    fdev: np.ndarray,
    scale: float,
    policy: FloatPolicy | None,
) -> tuple[bool, float, float]:
    """Whether every listed difference is one the policy allows, and how big the worst is.

    Two NaNs are equal whatever their payloads and signs, always and independently of the
    policy: IEEE leaves the payload of a produced NaN unspecified and the device canonicalises
    it in min/max and in conversions (`test_propagate_nan`). An infinity has to match exactly,
    which it does by falling out of every tolerance below.

    The absolute tolerance is multiplied by `scale`, the largest finite magnitude in the whole
    buffer. A reassociated sum is wrong by an amount proportional to the size of its *partial
    sums*, not to the size of the element that survives a cancellation, so an element that
    cancelled to near zero has an unbounded relative error and a bounded absolute one; the
    buffer's own dynamic range is the only proxy for the partial sums a comparison of outputs
    can see (`test_dot3d` in f16, `test_scan2d` with `cumsum`).
    """
    both_nan = np.isnan(fref) & np.isnan(fdev)
    any_nan = np.isnan(fref) | np.isnan(fdev)
    diff = np.abs(fref - fdev)
    magnitude = np.maximum(np.abs(fref), np.abs(fdev))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.divide(diff, magnitude, out=np.zeros_like(diff), where=magnitude > 0)
    rel = np.where(both_nan, 0.0, np.where(any_nan | ~np.isfinite(rel), np.inf, rel))
    abs_err = np.where(both_nan, 0.0, np.where(any_nan | ~np.isfinite(diff), np.inf, diff))
    within = both_nan
    if policy is not None and (policy.rtol or policy.atol):
        within = within | (abs_err <= policy.atol * scale) | (rel <= policy.rtol)
    worst_rel = float(np.max(rel)) if rel.size else 0.0
    worst_abs = float(np.max(abs_err)) if abs_err.size else 0.0
    return bool(np.all(within)), worst_rel, worst_abs


@dataclasses.dataclass(frozen=True)
class InexactScan:
    """What a module's inexact float ops are, and in which float types they compute."""

    classes: frozenset[str] = frozenset()
    elems: frozenset[str] = frozenset()


def _walk_ops(ops: list[Any]) -> Any:
    for op in ops:
        yield op
        for region in op.regions:
            for block in region.blocks:
                yield from _walk_ops(block.ops)


def _float_elems(op: Any) -> set[str]:
    if values is None:
        return set()
    types = [*op.result_types, *op.operand_types]
    return {t.scalar.name for t in types if values.is_float(t)}


def scan_inexact(module: Any) -> InexactScan:
    """The classes of `INEXACT_OPS` a module contains, and the float types they compute in.

    One class is a pattern and not an op: a float add or subtract fed by a float multiply.
    Triton compiles with `enable_fp_fusion` on, so LLVM contracts `a * b + c` into one fused
    multiply-add with a single rounding where this semantics rounds twice; the difference is an
    ulp of the compute type, and every fused pointwise kernel Inductor writes has the pattern.
    """
    classes: set[str] = set()
    elems: set[str] = set()
    table = _definers(module)
    for op in _walk_ops(module.ops):
        cls = INEXACT_OPS.get(op.name)
        if cls is None and op.name in ("arith.addf", "arith.subf") and table is not None:
            feeders = [table.get(id(op), {}).get(name) for name in op.operands]
            if any(
                d is not None and d.op is not None and d.op.name == "arith.mulf" for d in feeders
            ):
                cls = "contract"
        if cls is None:
            continue
        found = _float_elems(op)
        if found or values is None:
            classes.add(cls)
            elems |= found
    return InexactScan(frozenset(classes), frozenset(elems))


def _definers(module: Any) -> Any:
    try:
        from ttsem.defuse import defs

        return defs(module)
    except Exception:
        return None


def float_policy(elem: str, scan: InexactScan) -> FloatPolicy:
    """The policy for a buffer of element type `elem` in a launch `scan` describes.

    A launch whose only inexact op is a division is held to `APPROX_ULP` ulp, which is what
    NVIDIA promises for `div.full.f32`, taken in the narrowest type any division computes in
    rather than in the buffer's; anything wider moves the whole launch to the relative band,
    because no ulp count bounds a reassociated sum.
    """
    if values is None or elem not in values.FLOAT_FORMATS or not scan.classes:
        return FloatPolicy(elem)
    if scan.classes <= {"div", "contract"}:  # a few ulp: `div.full.f32`, or one fused rounding
        # in the widest of the types the launch computes in and the buffer's own: one rounding
        # of an f32 product still flips the last bit of the bf16 it is stored as
        widths = ({e for e in scan.elems if e in ULP_AS_RTOL} | {elem}) & set(ULP_AS_RTOL)
        band = APPROX_ULP * max(ULP_AS_RTOL[e] for e in widths)
        # relative to the value, or to the buffer's largest magnitude: a fused `a * b + c` whose
        # terms cancel leaves a result far smaller than the rounding of the product it absorbed
        return FloatPolicy(elem, rtol=band, atol=band)
    rtol, atol = WIDE_TOLERANCE.get(elem, WIDE_TOLERANCE["f32"])
    return FloatPolicy(elem, rtol=rtol, atol=atol)


def _is_tensor_like(value: object) -> bool:
    return hasattr(value, "data_ptr") and hasattr(value, "shape")


_BIT_PATTERN_DTYPES = {
    "torch.bfloat16": "int16",
    "torch.float8_e4m3fn": "int8",
    "torch.float8_e5m2": "int8",
}


def _to_numpy(value: Any) -> np.ndarray:
    """Host copy of a tensor. bf16 and fp8 have no numpy dtype: they travel as the unsigned
    bit patterns :mod:`values` uses (uint16 and uint8)."""
    value = _unwrap(value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    narrow = _BIT_PATTERN_DTYPES.get(str(getattr(value, "dtype", "")))
    if narrow is not None:
        import torch

        bits = value.contiguous().view(getattr(torch, narrow))
        return np.array(bits, copy=True).view(np.uint16 if narrow == "int16" else np.uint8)
    return np.array(value, copy=True)


def storage_copy(tensor: Any) -> tuple[int, np.ndarray]:
    """The whole storage behind a tensor, as a host array, with the device address of its
    first byte. A view (a slice, a padded allocation seen through a descriptor) addresses
    memory outside its own elements, so the memory model must hold the storage, not the view."""
    import torch

    tensor = _unwrap(tensor)
    flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    flat.set_(tensor.untyped_storage())
    return int(flat.data_ptr()), _to_numpy(flat)


def _unwrap(value: Any) -> Any:
    """The torch tensor behind a `triton.reinterpret` wrapper (whose `dtype` is a Triton
    dtype): the storage and its bytes are the wrapper's base tensor's."""
    import torch

    while not isinstance(getattr(value, "dtype", None), torch.dtype) and hasattr(value, "base"):
        value = value.base
    return value


def _is_descriptor(value: object) -> bool:
    """A host ``TensorDescriptor``: a base tensor plus shape, strides and block shape."""
    base = getattr(value, "base", None)
    return _is_tensor_like(base) and hasattr(value, "block_shape") and hasattr(value, "strides")


_TORCH_ELEM = {
    "torch.float32": ("float", 32, "f32"),
    "torch.float64": ("float", 64, "f64"),
    "torch.float16": ("float", 16, "f16"),
    "torch.bfloat16": ("float", 16, "bf16"),
    "torch.float8_e4m3fn": ("float", 8, "f8E4M3FN"),
    "torch.float8_e5m2": ("float", 8, "f8E5M2"),
    "torch.int8": ("int", 8, "i8"),
    "torch.uint8": ("int", 8, "i8"),
    "torch.int16": ("int", 16, "i16"),
    "torch.int32": ("int", 32, "i32"),
    "torch.int64": ("int", 64, "i64"),
    "torch.bool": ("int", 1, "i1"),
}


def _elem_type(torch_dtype: Any) -> Any:
    kind, width, name = _TORCH_ELEM[str(torch_dtype)]
    return values.Type(kind=kind, shape=None, elem=None, width=width, name=name, encoding=None)


def descriptor_values(desc: Any, base_ptr: int, emulated: bool = False) -> list[Any]:
    """The block arguments a host descriptor becomes in ``tt.func``.

    With TMA (sm_90 and later) that is the descriptor itself, then its shape as i32 and its
    strides as i64. Without TMA the frontend flattens the descriptor into its fields, in the
    order of the ``TensorDescriptor`` dataclass: the base pointer, the shape and the strides as
    i64, ``padding == "nan"`` and ``round_f32_to_tf32`` as i1, and only then the same trailing
    shape and strides (an RTX 4070, sm_89, on ``test_cat_nd``: 4r+3 block arguments per
    descriptor of rank r instead of 2r+1). ``execute`` picks the layout by the argument count.
    """
    shape = tuple(int(s) for s in desc.shape)
    strides = tuple(int(s) for s in desc.strides)
    block_shape = tuple(int(b) for b in desc.block_shape)
    padding = str(getattr(desc, "padding", "zero"))
    d = values.Descriptor(
        base=int(base_ptr),
        shape=shape,
        strides=strides,
        block_shape=block_shape,
        elem=_elem_type(desc.base.dtype),
        pad=padding,
    )
    ints: list[Any] = [np.array(s, dtype=np.int32) for s in shape]
    ints += [np.array(s, dtype=np.int64) for s in strides]
    if not emulated:
        return [d, *ints]
    head: list[Any] = [np.array(int(base_ptr), dtype=np.int64)]
    head += [np.array(s, dtype=np.int64) for s in shape]
    head += [np.array(s, dtype=np.int64) for s in strides]
    head += [np.array(padding == "nan"), np.array(bool(getattr(desc, "round_f32_to_tf32", False)))]
    return [*head, *ints]


def _arg_values(
    record: LaunchRecord, runtime_names: list[str], block: Any, emulated: bool
) -> list[Any]:
    """The block-argument values of a launch, in signature order."""
    arg_values: list[Any] = []
    for name in runtime_names:
        # The cpu path keeps only the scalars in `bound`: tensors live in pre/post/ptrs.
        bound_value = record.bound.get(name)
        if _is_descriptor(bound_value):
            arg_values.extend(descriptor_values(bound_value, record.ptrs[name], emulated))
        elif name in record.ptrs:
            arg_values.append(np.array(record.ptrs[name], dtype=np.int64))
        else:
            ty = block.args[len(arg_values)][1] if len(arg_values) < len(block.args) else None
            arg_values.append(_scalar_value(bound_value, ty))
    return arg_values


def _bind_names(
    arg_names: list[str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every kernel parameter, by name, exactly as ``fn[grid](*args, **kwargs)`` bound it.

    A parameter the call site left out keeps the default from the ``@triton.jit``'d function's
    signature: the compiled kernel has a block argument for it either way, and without the
    default there would be no value to pass (``test_default``, ``test_keyword_only_arguments``).
    """
    bound = dict(defaults or {})
    bound.update(zip(arg_names, args, strict=False))
    for name in arg_names[len(args) :]:
        if name in kwargs:
            bound[name] = kwargs[name]
    return bound


def _param_defaults(fn: Any) -> dict[str, Any]:
    params = getattr(fn, "params", None) or []
    return {p.name: p.default for p in params if getattr(p, "has_default", False)}


def _install_fake_driver(cc: int) -> None:
    """Enough of a CUDA driver to compile without a GPU."""
    _require_triton()
    from ttsem._fakedriver import _FakeDriver, _stub_torch_cuda

    _driver.set_active(_FakeDriver(cc))
    _stub_torch_cuda(cc)


def target_for(device: str, cc: int) -> GPUTarget:
    _require_triton()
    if device == "cpu":
        return GPUTarget("cuda", cc, 32)
    if device == "cuda":
        return _driver.active.get_current_target()
    raise ValueError(f"unsupported device {device!r}")


def _run_program(program_path: Path, device: str) -> None:
    """Import ``program_path`` fresh and call its ``main()`` as if run as a script."""
    module_name = f"_ttsem_prog_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    old_argv = sys.argv
    try:
        spec.loader.exec_module(module)
        if not hasattr(module, "main"):
            raise RuntimeError(f"{program_path} has no main()")
        with tempfile.TemporaryDirectory(prefix="ttsem-out-") as tmp:
            out_path = str(Path(tmp) / "out.npy")
            sys.argv = [str(program_path), "--device", device, "--out", out_path]
            module.main()
    finally:
        sys.argv = old_argv
        sys.modules.pop(module_name, None)


@dataclasses.dataclass
class _InterpretedLaunch:
    """One launch as genuine ``TRITON_INTERPRET=1`` semantics see it (no compile involved)."""

    fn_name: str
    grid: tuple[Any, ...]
    bound: dict[str, Any]
    pre: dict[str, np.ndarray]
    post: dict[str, np.ndarray]
    ptrs: dict[str, int]
    elems: dict[str, str] = dataclasses.field(default_factory=dict)


_INTERP_RUNNER = "ttsem._interp_runner"


def _capture_interpreted(program_path: Path) -> list[_InterpretedLaunch]:
    """Run ``program_path`` under genuine ``TRITON_INTERPRET=1`` semantics, in its own
    subprocess, and record every launch's (pre, post) state.

    This has to be a fresh *process*, not just a flipped env var: ``triton.language``'s own
    ``@triton.jit``-decorated builtins (``tl.zeros``, ``tl.sort``, ``tl.associative_scan``, the
    combine_fn machinery, ...) are wrapped as plain, uninterpretable ``JITFunction`` the first
    time ``triton.language`` is imported into a process, and stay that way for its whole
    lifetime no matter what ``TRITON_INTERPRET`` is set to afterwards. This module's own
    top-level ``import triton`` (needed for the compile side, below) has already done that by
    the time any function here could set the env var, so any kernel using one of those
    builtins would raise ``Cannot call @triton.jit'd outside of the scope of a kernel``. A
    subprocess started with the env var already set, before it ever imports triton, does not
    have this problem. See ``_interp_runner.py``.
    """
    with tempfile.TemporaryDirectory(prefix="ttsem-interp-") as tmp:
        out_path = Path(tmp) / "launches.pkl"
        env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
        result = subprocess.run(
            [sys.executable, "-m", _INTERP_RUNNER, str(program_path), str(out_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"interpreted reference pass failed for {program_path}: {result.stderr[-4000:]}"
            )
        with out_path.open("rb") as f:
            raw: list[dict[str, Any]] = pickle.load(f)
    return [
        _InterpretedLaunch(
            fn_name=item["fn_name"],
            grid=tuple(item["grid"]),
            bound=item["bound"],
            pre=item["pre"],
            post=item["post"],
            ptrs=item["ptrs"],
            elems=item.get("elems", {}),
        )
        for item in raw
    ]


def _capture_compilable(
    program_path: Path, cc: int
) -> list[tuple[JITFunction, tuple[Any, ...], dict[str, Any]]]:
    """Run ``program_path`` a second time, undecorated, purely to get a real ``JITFunction``
    (with a genuine ``.signature``/``.params``) per launch: ``.run`` is replaced so nothing
    ever tries to actually launch on the (absent) GPU."""
    _install_fake_driver(cc)
    calls: list[tuple[JITFunction, tuple[Any, ...], dict[str, Any]]] = []
    orig_run = JITFunction.run

    def run(self: JITFunction, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        if not warmup:
            calls.append((self, args, dict(kwargs)))
        return None

    JITFunction.run = run
    try:
        _run_program(program_path, "cpu")
    finally:
        JITFunction.run = orig_run
    return calls


def _capture_cpu(program_path: Path, cc: int) -> list[LaunchRecord]:
    launches = _capture_interpreted(program_path)
    calls = _capture_compilable(program_path, cc)
    if len(launches) != len(calls):
        raise RuntimeError(
            f"launch count mismatch between the interpreted ({len(launches)}) and the "
            f"compile ({len(calls)}) pass over {program_path}"
        )
    records: list[LaunchRecord] = []
    for i, (launch, (fn, args, kwargs)) in enumerate(zip(launches, calls, strict=True)):
        records.append(
            LaunchRecord(
                launch_id=i,
                fn=fn,
                fn_name=launch.fn_name,
                fn_file=str(program_path),
                grid=launch.grid,
                args=args,
                kwargs=kwargs,
                bound=launch.bound,
                pre=launch.pre,
                post=launch.post,
                ptrs=launch.ptrs,
                elems=launch.elems,
            )
        )
    return records


@dataclasses.dataclass
class Recorded:
    record: LaunchRecord
    result: Any


def record_launch(
    fn: JITFunction,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    grid: Any,
    orig_run: Any,
    fn_file: str,
    launch_id: int = 0,
) -> Recorded:
    """Run one launch through ``orig_run`` with the storages of its tensor arguments copied
    before and after, and return the record next to the launch's own result."""
    bound = _bind_names(fn.arg_names, args, kwargs, _param_defaults(fn))
    if callable(grid):
        grid = grid(bound)
    tensors = {name: v for name, v in bound.items() if _is_tensor_like(v)}
    tensors.update({name: v.base for name, v in bound.items() if _is_descriptor(v)})
    copies = {name: storage_copy(v) for name, v in tensors.items()}
    pre = {name: arr for name, (_, arr) in copies.items()}
    ptrs = {name: int(v.data_ptr()) for name, v in tensors.items()}
    bases = {name: base for name, (base, _) in copies.items()}
    elems = {name: str(_unwrap(v).dtype) for name, v in tensors.items()}
    result = orig_run(fn, *args, grid=grid, warmup=False, **kwargs)
    post = {name: storage_copy(v)[1] for name, v in tensors.items()}
    record = LaunchRecord(
        launch_id=launch_id,
        fn=fn,
        fn_name=fn.__name__,
        fn_file=fn_file,
        grid=tuple(grid) if isinstance(grid, (tuple, list)) else (grid,),
        args=args,
        kwargs=dict(kwargs),
        bound=bound,
        pre=pre,
        post=post,
        ptrs=ptrs,
        bases=bases,
        elems=elems,
    )
    return Recorded(record, result)


def _capture_cuda(program_path: Path) -> list[LaunchRecord]:
    records: list[LaunchRecord] = []
    orig_run = JITFunction.run

    def run(self: JITFunction, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
        if warmup:
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        rec = record_launch(self, args, kwargs, grid, orig_run, str(program_path), len(records))
        records.append(rec.record)
        return rec.result

    JITFunction.run = run
    try:
        _run_program(program_path, "cuda")
    finally:
        JITFunction.run = orig_run
    return records


def capture_launches(program_path: Path, device: str, cc: int = DEFAULT_CC) -> list[LaunchRecord]:
    """Run ``program_path`` and record every kernel launch it makes.

    On ``device == "cpu"`` the launch runs through Triton's own interpreter (there is no GPU
    here), which is treated as the reference; on ``device == "cuda"`` the real launch runs and
    its device buffers are the reference. Either way, every :class:`LaunchRecord` carries a
    real, compilable ``JITFunction`` (see the module docstring for why ``cpu`` needs two passes
    to get both a correct interpreted result and a compilable function out of the same launch).
    """
    if device == "cpu":
        return _capture_cpu(program_path, cc)
    if device == "cuda":
        return _capture_cuda(program_path)
    raise ValueError(f"unsupported device {device!r}")


def _specialize(
    fn: JITFunction, args: tuple[Any, ...], kwargs: dict[str, Any], target: GPUTarget
) -> tuple[Any, dict[str, str], dict[Any, Any], dict[Any, Any]]:
    """``(options, signature, constexprs, attrs)`` for a launch on ``args``/``kwargs``.

    ``signature`` maps every kernel parameter name to either its compiled type (``"i32"``,
    ``"*fp32"``, ...) or the string ``"constexpr"``: a plain ``int`` argument whose concrete
    *value* specializes to a constant (e.g. a stride of 1) gets folded away exactly like a
    ``tl.constexpr`` one, dropped from the compiled function's arguments, even though nothing
    in the Python signature marked it that way. This is the one place that decides which
    parameter names still correspond to a ``tt.func`` block argument.
    """
    backend = make_backend(target)
    kwargs = {k: v for k, v in kwargs.items() if k not in LAUNCH_KWARG_KEYS}
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound_args, specialization, options = binder(*args, **kwargs)
    return fn._pack_args(backend, kwargs, bound_args, specialization, options)


def _compile_asm(
    fn: JITFunction, args: tuple[Any, ...], kwargs: dict[str, Any], target: GPUTarget
) -> dict[str, Any]:
    """Full ``asm`` dict of the compile a launch on ``args``/``kwargs`` implies.

    Mirrors ``compile_side.compile_like_launch`` (same binder / ``_pack_args`` steps, so the
    compile assumes exactly what the launch assumed), but keeps every IR stage: ``CompiledSide``
    only surfaces ttgir/llir/ptx, and :func:`ir_for_launch` also needs ttir.
    """
    options, signature, constexprs, attrs = _specialize(fn, args, kwargs, target)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    compiled = triton.compile(src, target=target, options=options.__dict__)
    return dict(compiled.asm)


def dump_for_launch(record: LaunchRecord, target: GPUTarget) -> str:
    """The ``MLIR_ENABLE_DUMP`` trace of compiling ``record``'s launch, in the generic form:
    every pass's module, which :func:`validate.split_dump` cuts into stages.

    The compile runs in this process with the cache bypassed, its stderr caught at the file
    descriptor (the dumps come from C++), and the wheel's printer switched to the generic form
    for the rest of the process (see :func:`mlir.enable_generic_printing`)."""
    if mlir is None:
        raise RuntimeError("mlir.py is not available yet")
    try:
        mlir.enable_generic_printing()
    except RuntimeError:
        pass  # a source build hides the symbol: the dump comes pretty, `to_generic` converts it
    saved = {k: os.environ.get(k) for k in ("MLIR_ENABLE_DUMP", "TRITON_ALWAYS_COMPILE")}
    os.environ["MLIR_ENABLE_DUMP"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    fd_err = 2  # the C++ side writes to the process's fd 2, whatever sys.stderr is wrapped in
    try:
        sys.stderr.flush()
    except Exception:
        pass
    keep = os.dup(fd_err)
    with tempfile.TemporaryFile(mode="w+b") as sink:
        os.dup2(sink.fileno(), fd_err)
        try:
            _compile_asm(record.fn, record.args, record.kwargs, target)
        finally:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(keep, fd_err)
            os.close(keep)
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        sink.seek(0)
        return sink.read().decode("utf-8", errors="replace")


def ir_for_launch(
    record: LaunchRecord,
    stage: str,
    target: GPUTarget,
    triton_opt: str | None = None,
) -> str:
    """The generic-form IR of ``record``'s launch, ``stage`` in ``{"ttir", "ttgir"}``."""
    if stage not in ("ttir", "ttgir"):
        raise ValueError(f"unsupported stage {stage!r}")
    if mlir is None:
        raise RuntimeError("mlir.py is not available yet")
    asm = _compile_asm(record.fn, record.args, record.kwargs, target)
    pretty = asm[stage]
    assert isinstance(pretty, str)
    return mlir.to_generic(pretty, triton_opt)


def _runtime_param_names(record: LaunchRecord) -> list[str]:
    """Kernel parameter names, in ``tt.func``'s block-argument order.

    Not just the ones without a ``tl.constexpr`` annotation: :func:`_specialize` on this
    launch's concrete arguments is the only authority on which parameters the *compiled*
    signature actually kept (see its docstring -- a plain integer argument can specialize away
    too). Any valid target works here since this is a frontend decision, independent of the
    compute capability the launch happened to compile for.
    """
    _require_triton()
    target = GPUTarget("cuda", DEFAULT_CC, 32)
    _options, signature, _constexprs, _attrs = _specialize(
        record.fn, record.args, record.kwargs, target
    )
    return [name for name, ty in signature.items() if ty != "constexpr"]


def _grid3(grid: tuple[Any, ...]) -> tuple[int, int, int]:
    dims = [int(g) for g in grid[:3]]
    dims += [1] * (3 - len(dims))
    return dims[0], dims[1], dims[2]


def _describe(e: BaseException) -> str:
    """``repr`` of an exception, plus the last frames of its traceback when
    ``TTSEM_TRACEBACK`` is set."""
    text = repr(e)
    notes = getattr(e, "__notes__", None)
    if notes:
        text += " " + "; ".join(notes)
    if os.environ.get("TTSEM_TRACEBACK"):
        import traceback

        tail = traceback.format_exception(type(e), e, e.__traceback__)[-4:]
        text += "\n" + "".join(tail).strip()
    return text


def _scalar_value(value: Any, ty: Any) -> np.ndarray:
    """A non-pointer runtime argument (e.g. a stride) as the numpy value ``ty`` expects."""
    if values is not None:
        try:
            dtype = values.to_numpy(ty)
            return np.array(value, dtype=dtype)
        except Exception:
            pass
    return np.array(value)


@dataclasses.dataclass
class Execution:
    """One run of a module on a record's pre-launch state, before any comparison."""

    failure: Comparison | None
    module: Any = None
    memory: Any = None
    runtime_names: list[str] = dataclasses.field(default_factory=list)
    unsupported: list[str] = dataclasses.field(default_factory=list)

    def buffer(self, record: LaunchRecord, name: str) -> np.ndarray | None:
        """The storage of tensor argument ``name`` after the run, or ``None``."""
        if self.memory is None:
            return None
        got = self.memory.buffers().get(record.bases.get(name, record.ptrs[name]))
        return None if got is None else np.asarray(got)


def execute(record: LaunchRecord, ir_text: str, trace_reads: bool = False) -> Execution:
    """Run ``ir_text`` (generic-form IR) on ``record``'s pre-launch state. The memory the run
    leaves behind is in the result, or ``failure`` says why it did not run to the end."""
    if not SEMANTICS_AVAILABLE:
        return Execution(Comparison("error", -1, [], [], "mlir/memory/interp not available yet"))
    try:
        module = mlir.parse(ir_text)
    except Exception as e:  # the parser is someone else's code in progress; never crash on it
        return Execution(Comparison("error", -1, [], [], f"parse failed: {e!r}"))

    fn_name = record.fn_name
    if fn_name not in module.funcs:
        candidates = list(module.funcs)
        if len(candidates) != 1:
            return Execution(
                Comparison("error", -1, [], [], f"no unique tt.func (have {candidates})")
            )
        fn_name = candidates[0]

    block = module.funcs[fn_name].regions[0].blocks[0]
    runtime_names = _runtime_param_names(record)

    memory = Memory()
    memory.trace_reads = trace_reads
    registered: set[int] = set()
    for name in runtime_names:
        if name in record.ptrs:
            base = record.bases.get(name, record.ptrs[name])
            if base not in registered:  # two arguments may share one storage
                # A *copy*: `Memory.register` keeps the array it is handed and the stores of
                # the run write through it, so registering `record.pre` itself would leave the
                # record holding post-launch state. The per-pass validator runs the same record
                # once per pass dump, and every stage after the first would then start from
                # already-accumulated inputs (`validate_corpus.py` on the `loops` family: only
                # the programs whose stores are idempotent survived it).
                memory.register(base, np.array(record.pre[name], copy=True))
                registered.add(base)

    arg_values = _arg_values(record, runtime_names, block, emulated=False)
    if len(arg_values) != len(block.args) and any(
        _is_descriptor(record.bound.get(name)) for name in runtime_names
    ):
        arg_values = _arg_values(record, runtime_names, block, emulated=True)
    if len(arg_values) != len(block.args):
        message = (
            f"arg count mismatch: {len(runtime_names)} runtime params "
            f"({len(arg_values)} values) vs {len(block.args)} block args"
        )
        return Execution(Comparison("error", -1, [], [], message))

    interp = Interp(module, memory, _grid3(record.grid))
    try:
        interp.run_grid(fn_name, arg_values)
    except values.Poison as e:
        return Execution(Comparison("poison", -1, [], [], f"undefined by the semantics: {e}"))
    except Unsupported as e:
        unsupported = sorted(set(getattr(interp, "unsupported", None) or [str(e)]))
        return Execution(Comparison("unsupported", -1, [], unsupported, str(e)))
    except Exception as e:
        return Execution(Comparison("error", -1, [], [], _describe(e)))

    unsupported = sorted(set(getattr(interp, "unsupported", None) or []))
    return Execution(None, module, memory, runtime_names, unsupported)


def run_launch(record: LaunchRecord, ir_text: str, trace_reads: bool = False) -> Comparison:
    """Run ``ir_text`` (generic-form IR) on ``record``'s pre-launch state and compare with its
    recorded post-launch state, buffer by buffer."""
    ex = execute(record, ir_text, trace_reads)
    if ex.failure is not None:
        return ex.failure
    reads = ex.memory.read_bytes() if trace_reads else None
    scan = scan_inexact(ex.module)
    diffs: list[dict[str, Any]] = []
    n_diff = 0
    for name in ex.runtime_names:
        if name not in record.ptrs:
            continue
        got = ex.buffer(record, name)
        want = record.post[name]
        if got is None:
            return Comparison("error", -1, [], [], f"buffer {name!r} missing after run")
        policy = float_policy(_elem_spelling(record, name), scan)
        cmp = compare(want.reshape(-1), got.reshape(-1), policy=policy)
        n_diff += max(cmp["n_diff"], 0)
        if not cmp["equal"]:
            if not cmp.get("approx") and _only_exchanged_differ(ex.memory, record, name, want, got):
                cmp["approx"] = True
                cmp["note"] = (
                    "a buffer of atomic exchanges: the device's arrival order, not a result"
                )
            diffs.append({"name": name, **cmp})

    if ex.unsupported:
        return Comparison("unsupported", n_diff, diffs, ex.unsupported, reads=reads)
    if n_diff == 0:
        return Comparison("match", 0, diffs, [], reads=reads)
    if all(d.get("approx") for d in diffs):
        return Comparison("approx", n_diff, diffs, [], _approx_note(scan.classes), reads=reads)
    return Comparison("mismatch", n_diff, diffs, [], reads=reads)


def _only_exchanged_differ(
    memory: Any, record: LaunchRecord, name: str, want: np.ndarray, got: np.ndarray
) -> bool:
    """True when the buffer is one that atomic exchanges or compare-and-swaps write: the
    protocol words of a split scan or a decoupled lookback. Which slot a program claims there
    is the order the programs arrived in (a dynamic block id from an atomic counter), so the
    replay and the device fill different words with different values, and none of it is a
    result of the kernel: the buffer as a whole is the device's schedule, not a mismatch."""
    hits = getattr(memory, "exchanged", None)
    if not hits:
        return False
    addrs = np.concatenate(hits)
    base = record.bases.get(name, record.ptrs[name])
    nbytes = int(np.asarray(want).nbytes)
    return bool(((addrs >= base) & (addrs < base + nbytes)).any())


def final_state(record: LaunchRecord, ir_text: str) -> dict[str, np.ndarray] | Comparison:
    """The tensor arguments after running ``ir_text`` on ``record``'s inputs, by name, or the
    :class:`Comparison` that says why the run did not complete.

    This is the semantics as a function, with no device output in sight: two modules can be
    held against each other on the same inputs, which is what a minimiser needs."""
    ex = execute(record, ir_text)
    if ex.failure is not None:
        return ex.failure
    if ex.unsupported:
        return Comparison("unsupported", -1, [], ex.unsupported)
    out: dict[str, np.ndarray] = {}
    for name in ex.runtime_names:
        if name not in record.ptrs:
            continue
        got = ex.buffer(record, name)
        if got is None:
            return Comparison("error", -1, [], [], f"buffer {name!r} missing after run")
        out[name] = np.array(got, copy=True)
    return out


def _elem_spelling(record: LaunchRecord, name: str) -> str:
    """The Triton spelling of a buffer's element type, from the torch dtype recorded for it."""
    torch_dtype = record.elems.get(name)
    if torch_dtype is None or torch_dtype not in _TORCH_ELEM:
        return ""
    return _TORCH_ELEM[torch_dtype][2]


def _approx_note(classes: frozenset[str]) -> str:
    if classes == {"div"}:
        return f"floats within {APPROX_ULP} ulp (approximate device division)"
    if classes:
        return "floats within tolerance (the launch reassociates or approximates), or NaN"
    return "the only differences are NaN payloads"


# A program's verdict is the worst of its launches', in this order. `approx` and `poison` used
# to be missing, so a program with one poisoned launch and one exact one was summarised as
# `match`, which is the one word it certainly is not.
VERDICT_ORDER = ("match", "approx", "poison", "unsupported", "mismatch", "error")


def _aggregate(verdicts: list[str]) -> str:
    if not verdicts:
        return "error"
    return max(
        verdicts, key=lambda v: VERDICT_ORDER.index(v) if v in VERDICT_ORDER else len(VERDICT_ORDER)
    )


def _process_program(
    py: Path, stage: str, device: str, triton_opt: str | None, cc: int
) -> dict[str, Any]:
    """Capture every launch in ``py`` and check it at ``stage``; picklable for a process pool."""
    result: dict[str, Any] = {"program": py.name, "stage": stage, "device": device, "launches": []}
    try:
        records = capture_launches(py, device, cc)
    except Exception as e:
        result["verdict"] = "error"
        result["message"] = f"capture failed: {e!r}"
        return result
    if not records:
        result["verdict"] = "error"
        result["message"] = "no launches captured"
        return result

    target = target_for(device, cc)
    verdicts: list[str] = []
    for record in records:
        entry: dict[str, Any] = {"launch_id": record.launch_id, "fn": record.fn_name}
        try:
            ir_text = ir_for_launch(record, stage, target, triton_opt)
        except Exception as e:
            entry["verdict"] = "error"
            entry["message"] = f"compile failed: {e!r}"
            result["launches"].append(entry)
            verdicts.append("error")
            continue
        cmp = run_launch(record, ir_text)
        entry.update(cmp.to_dict())
        result["launches"].append(entry)
        verdicts.append(cmp.verdict)
    result["verdict"] = _aggregate(verdicts)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--programs", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--stage", choices=("ttir", "ttgir"), default="ttir")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    ap.add_argument("--triton-opt", default=None)
    ap.add_argument("--cc", type=int, default=DEFAULT_CC)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    programs = sorted(args.programs.glob("*.py"))
    if args.limit is not None:
        programs = programs[: args.limit]
    if not programs:
        print("no programs")
        return 0

    args.results.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter[str] = collections.Counter()

    def _handle(res: dict[str, Any]) -> None:
        (args.results / f"{Path(res['program']).stem}.json").write_text(json.dumps(res, indent=1))
        counts[res["verdict"]] += 1
        print(f"{res['program']:>24} {res['verdict']:<12} {res.get('message', '')}")

    if args.jobs <= 1:
        for py in programs:
            _handle(_process_program(py, args.stage, args.device, args.triton_opt, args.cc))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(_process_program, py, args.stage, args.device, args.triton_opt, args.cc)
                for py in programs
            ]
            for future in futures:
                _handle(future.result())

    print(f"{len(programs)} programs: {dict(counts)}")
    return 1 if counts["mismatch"] or counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
