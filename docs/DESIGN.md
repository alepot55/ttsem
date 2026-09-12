# ttsem: an executable semantics of Triton IR, and a validator for every pass

## Why

Triton has 200k tests and no definition of what a pass must preserve. When the differential
fuzzer (a companion generator of layout, dtype and descriptor programs) finds a wrong output, the guilty pass is found by hand. This
package gives the IR a reference execution, so that

1. any compiled kernel can be run on the CPU **from its IR**, not from Python source (the
   shipped interpreter reruns the Python AST, so it never sees what a pass did);
2. the IR after every pass can be run on the same inputs, and the first pass whose output
   differs is the culprit (`ttsem/validate.py`);
3. the semantics is a document a maintainer can read, and later a Lean model can be checked
   against it.

Alive2 did this for LLVM. Nothing does it for tile IRs, whose values are *distributed* tensors:
here layouts are erased (level 1), then modelled (level 2, later).

## Levels

| level | IR | what is modelled | validated against |
|---|---|---|---|
| 1 | TTIR and TTGIR with layouts erased | tensors as numpy arrays, pointers as int64 addresses, a flat byte-addressable memory, control flow, program ids; async ops sequentialised | device outputs of recorded launches (oracle harness), the fuzzer corpus |
| 2 | TTGIR with layouts | per-thread and per-warp values through linear layouts; `convert_layout`, reduce/scan/gather lowering choices | level 1 on the same module, device |
| 3 | pipelined and warp-specialised TTGIR | shared memory buffers, barriers, partitions in lock-step | level 1 |

## Interfaces (fixed; every module programs against these)

### `ttsem/mlir.py` (parser)

Parses the **generic** form printed by `triton-opt --mlir-print-op-generic` (or by the Python
bindings). The generic form is regular: every op is
`%r0, %r1 = "dialect.op"(%a, %b) <{attr = value}> ({region}, {region}) : (T, T) -> (T, T) loc(...)`.

```python
@dataclass
class Type:        # parsed MLIR type
    kind: str      # "tensor" | "ptr" | "int" | "float" | "index" | "memdesc" | "other"
    shape: tuple[int, ...] | None   # tensors and memdescs
    elem: "Type | None"             # tensor element, pointer pointee
    width: int | None               # ints and floats (bf16 = 16 with name "bf16")
    name: str                       # the printed spelling, e.g. "bf16", "f8E4M3FN", "i1"
    encoding: str | None            # the raw layout text, kept but never interpreted at level 1

@dataclass
class Op:
    name: str                       # "arith.addi", "tt.load", ...
    results: list[str]              # SSA names as printed ("%3", "%3#1")
    result_types: list[Type]
    operands: list[str]
    operand_types: list[Type]
    attrs: dict[str, object]        # ints, floats, bools, str, lists, dict for nested, DenseAttr for dense<>
    regions: list["Region"]
    successors: list[str]           # cf ops
    loc: str | None

@dataclass
class Block:
    label: str | None               # "^bb0"
    args: list[tuple[str, Type]]
    ops: list[Op]

@dataclass
class Region:
    blocks: list[Block]

@dataclass
class Module:
    funcs: dict[str, Op]            # tt.func ops by symbol name
    ops: list[Op]

def parse(text: str) -> Module
def to_generic(text: str, triton_opt: str | None = None) -> str   # pretty -> generic
```

`DenseAttr(value, elem_type, shape)` holds `dense<...>` constants: splat or list. Attribute values
`3 : i32` become `int`; `1.5 : f32` become `float`; `true` bool; strings unquoted; arrays lists;
dictionaries dicts; type-valued attributes `Type`; anything unparsed is kept as the raw string.

### `ttsem/memory.py`

```python
class Memory:
    """Flat address space. Buffers are registered with a base address and a numpy array
    (host copies of the kernel arguments); loads and stores go through int64 addresses."""
    def register(self, base: int, array: np.ndarray) -> None
    def load(self, addrs: np.ndarray, mask: np.ndarray | None, other, dtype) -> np.ndarray
    def store(self, addrs: np.ndarray, values: np.ndarray, mask: np.ndarray | None) -> None
    def atomic(self, kind: str, addrs, values, mask, sem: str) -> np.ndarray   # returns old values
    def buffers(self) -> dict[int, np.ndarray]                                   # for output comparison
```

Out-of-range addresses with a true mask raise `MemoryFault(addr)`; a masked-off lane never
touches memory (a masked load yields `other`, or 0 when absent, exactly as the docs promise).

### `ttsem/values.py`

Tensor values are `np.ndarray` with the numpy dtype of the element type; `i1` is `np.bool_`;
`bf16` is `np.uint16` bit patterns with helper conversions (as `triton.runtime.interpreter` does);
fp8 kinds are `np.uint8` with conversions; pointers are `np.int64` addresses (a pointer tensor
is an int64 array); scalars are 0-d arrays. `int_dtype(Type)`, `to_numpy(Type)`,
`from_bits(...)`, `to_bits(...)`.

### `ttsem/ops.py`

One function per op name, registered in `OPS: dict[str, Callable]`, signature
`fn(interp: "Interp", op: Op, args: list[Value]) -> list[Value]`. Region-carrying ops
(`tt.reduce`, `tt.scan`, `scf.*`, `tt.elementwise_inline_asm` unsupported) call
`interp.run_region(region, args)`. Integer arithmetic wraps at the width; `divsi`/`remsi`
follow C (truncation toward zero); shifts by >= width give 0 (or sign fill for `shrsi`);
`cmpi` predicates by attr; float ops are done in float32/float64 per the type, results rounded
to the element type. `tt.dot` follows the interpreter's precision rules
(the fuzzer's `mma.py` has the exactness argument).

### `ttsem/interp.py`

```python
class Interp:
    def __init__(self, module: Module, memory: Memory, num_programs: tuple[int,int,int])
    def run(self, fn: str, args: list[Value], program_id: tuple[int,int,int]) -> list[Value]
    def run_grid(self, fn: str, args: list[Value]) -> None       # every program id, in order
    def run_region(self, region: Region, args: list[Value]) -> list[Value]
```

Blocks execute in order; `cf.br`/`cf.cond_br` jump; `scf.for`/`scf.if`/`scf.while`/`scf.yield`
are structured. Unknown op names raise `Unsupported(op.name)` with the op text, so coverage is
measurable: `interp.unsupported` collects them.

### `ttsem/harness.py`

Records launches (reusing `ttsem/oracle/launch_hook.py`), gets the IR of each
launch (`compile_like_launch` from the oracle, or `TRITON_KERNEL_DUMP`), copies the argument
tensors to host, registers them in a `Memory` at their device addresses, runs `Interp.run_grid`,
and compares every output buffer with the device result: bitwise for integers, booleans and
every float buffer of a launch whose IR has no inexact float op; with a relative and a scaled
absolute bound per element type (verdict `approx`) when the IR divides, reduces, scans, dots or
calls a transcendental on floats (the table `INEXACT_OPS` in `ttsem/harness.py`, one comment per
entry). `match` stays strictly bitwise; two NaNs are equal. CLI: `ttsem/harness.py --programs DIR` over the fuzzer corpus, `--pytest` via the
oracle plugin.

### `ttsem/validate.py`

`MLIR_ENABLE_DUMP=1` prints the IR before every pass. `split_dump(text)` yields
`(pass_name, ir_text)`; each is converted to generic form and run by `Interp` on the same
inputs; the report names the first pass after which the output differs from the device (or
from the previous pass), and which ops were unsupported at each stage.

## Rules

- numpy only in the semantics; torch only in the harness.
- Every op implemented gets a test in `tests/` that runs a two-line generic-form module.
- Semantics choices that the docs leave open are written down in `SEMANTICS.md` with the
  program that decided them (device behaviour is the tie-breaker, and the note says so).
- No layout is interpreted at level 1: `ttg.convert_layout` is the identity, `ttg.local_alloc`
  / `local_load` / `local_store` / `async_copy_*` / `async_wait` / `mbarrier` ops are a plain
  buffer and no-ops, in program order.

## Level 2: distributed values

Level 1 erases layouts, so it answers "does this kernel compute the right array". Level 2 adds
"and does each thread hold the right part of it". The two modules below are the whole of it as
of today: the algebra of linear layouts and the port of Triton's encoding-to-layout map. The
interpreter is wired to them by a flag, described at the end.

### `ttsem/linear_layout.py`

A port of `include/triton/Tools/LinearLayout.h` and `lib/Tools/LinearLayout.cpp`.

```python
class LinearLayout:
    """A GF(2)-linear map from named input dims to named output dims."""
    def __init__(self, bases, out_dims, require_surjective=True)
    # bases: {in_dim: [basis, ...]}, basis[i] is the out-coordinate of in_dim = 2**i
    # out_dims: [name, ...] (sizes inferred, surjective) or [(name, size), ...]

    @staticmethod
    def empty() / identity1D(size, in_dim, out_dim) / strided1D(...) / zeros1D(...)

    def apply(self, ins: Mapping[str, int]) -> dict[str, int]
    def __mul__(self, outer) -> LinearLayout          # direct sum, C++ operator*
    def compose(self, outer) -> LinearLayout          # outer o self
    def invert_and_compose(self, outer) -> LinearLayout   # C(x) with outer(C(x)) == self(x)
    def invert(self) / pseudoinvert(self) -> LinearLayout
    def sublayout(self, in_dims, out_dims) / sublayout_is_zero(self, in_dims, out_dims)
    def transpose_ins / transpose_outs / reshape_ins / reshape_outs / rename_outs
    def resize_out_dim(self, out_dim, new_size) / remove_zero_bases_along_dim(self, dim)
    def free_variable_masks(self) -> dict[str, int]   # input bits that do not change the output
    def num_consecutive_in_out(self) -> int           # C++ getNumConsecutiveInOut

def lstsq(a, b) / divide_left(a, b) / supremum(x, y) / rref_gf2(rows, num_cols)
def output_basis_mask(layout, in_dims, out_dim) / input_basis_mask(layout, in_dim, out_dims)
```

Conventions are the C++ ones and the tests pin them: dimensions are ordered minor-to-major and
that order only matters for reshape; bit order inside a dimension is little-endian; out-dim
sizes are powers of two; a layout that infers its own sizes must be surjective. `LayoutError`
replaces the C++ asserts.

### `ttsem/layouts.py`

A port of `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` plus the parts of
`LayoutUtils.cpp` and `Dialect.cpp` it needs, restricted to the NVIDIA encodings.

```python
def parse_encoding(text: str) -> Encoding      # the raw text of Type.encoding
def to_linear_layout(encoding, shape, num_warps=None, threads_per_warp=None) -> LinearLayout
def drop_pipelining_dims(shape, encoding) -> list[int]   # C++ dropPipeliningDim
```

`Encoding` is one of `Blocked`, `NvidiaMma` (v2 and v3), `DotOperand` (over a blocked or an mma
parent), `Slice`, `Linear`, `SwizzledShared`, `NvmmaShared`. `num_warps` and
`threads_per_warp` come from the module attributes `ttg.num-warps` and
`ttg.threads-per-warp`; when passed they are checked against the layout, which catches an
encoding read against the wrong module. A memdesc keeps its memory space and its `mutable`
flag inside `Type.encoding`, and its leading dimensions may be pipelining stages that the
shared encoding does not describe, so a memdesc shape goes through `drop_pipelining_dims`
first.

### The value model

At level 1 a tensor value is a numpy array in logical index order. At level 2 a *distributed*
value is a pair `(array, layout)`. The array is the same array level 1 computes; the layout
says where each of its elements lives. Register `r` of lane `l` in warp `w` of block `b` holds

```
array[layout.apply({"register": r, "lane": l, "warp": w, "block": b})]
```

Two consequences. First, the logical array does not change, so every op implementation of
`ttsem/ops.py` keeps working unchanged: level 2 adds an assertion about the layout, not a different
value. Second, broadcast is explicit, because a zero basis means several hardware locations
hold the same element, and `layout.total_in_dim_size()` may exceed `prod(shape)`.

### `ttg.convert_layout`

A conversion does not change the logical array. It changes which (register, lane, warp) holds
which element, and the change of basis is `src.invert_and_compose(dst)`. Level 2 checks three
things:

1. src and dst have the same out-dim sizes, that is the same tensor shape;
2. the conversion is a bijection on the non-broadcast part: after
   `remove_zero_bases_along_dim("register")` on both sides, the composed layout has no free
   variables outside the dims that are broadcast in both;
3. no element is lost: every logical index in the image of src is in the image of dst.

A pass that picks a layout in which two lanes claim the same element while a third element has
no owner violates (2). That is the wrong-lane class.

### Reductions, scans, gathers, histograms

The result arrays are the level-1 ones. What level 2 adds is where the result is.

- `tt.reduce` over `axis` gives a value laid out by `#ttg.slice<{dim = axis, parent = src}>`,
  which the port builds by removing the standard dim and then dropping the register bases that
  became zero. Every lane whose src coordinates differ only along `axis` holds the same
  result, and the slice layout records that broadcast as zero bases. A lowering that leaves
  the result in one lane and forgets the broadcast is a level-2 failure and a level-1 pass.
- `tt.scan` keeps the src layout, and level 2 states that the value at register `r` of lane
  `l` is the prefix in *logical* order, not in register order, so a lowering that scans inside
  a thread without the cross-lane step is caught.
- `tt.gather` along `axis` is warp local exactly when
  `src_layout.sublayout(["warp"], [f"dim{axis}"])` is zero; when it is not, the lowering must
  go through shared memory.
- `tt.histogram` produces a value whose out dim is the bin index, and level 2 states which
  lane accumulates which bins.

### Shared memory

A shared layout maps `offset`, an element index inside the allocation and not a byte, to a
logical index. `ttg.local_alloc` gives a buffer and a layout; `local_store` writes each element
at the offset that the inverse of the shared layout assigns to its logical index, and
`local_load` reads it back. The byte address is `offset * elem_bytes`, so with an
`nvmma_shared` swizzle the address map is byte exact and comparable with the device; stage `s`
of a pipelined buffer starts at `s * prod(trailing_dims) * elem_bytes`.

The round trip `local_load(local_store(v))` must be the identity. It is not when the swizzle
that wrote differs from the swizzle that read, which is the wrong-swizzle class. Level 1
cannot see it: it stores into and loads from a plain buffer in program order, so the swizzle
cancels.

### What level 2 catches that level 1 cannot

Level 1's oracle is the output buffer, so it only sees a bug that survives to memory.

| class | example | why level 1 is blind |
|---|---|---|
| wrong lane (tile mapping, 6.3% of the ISSTA 2026 tile-bug corpus) | a `convert_layout` or an mma layout choice that gives an element to the wrong lane | the element still reaches memory once, through some other lane |
| wrong swizzle (the indexing part of memory, 11.6% of that corpus) | a shared buffer written and read under different swizzles | the plain buffer of level 1 makes the two swizzles cancel |
| warp locality | [triton#11600](https://github.com/triton-lang/triton/issues/11600): `setOptimizedGatherLayout` lays the leftover warps along the gather axis and then asserts `isWarpLocal` | level 1 never builds the gather's layout, so the eight programs of `RESULTS.md` that never compiled are simply dropped |

The third row is the concrete one: those eight `layout` and `narrow` fuzzer programs are
counted as "never compiled on the device" today. A level-2 validator builds the same layout
the pass builds and names the violated property instead of watching an assert fire.

### Wiring into `ttsem/interp.py`

Level 1 does not move. One flag:

```python
class Interp:
    def __init__(self, module, memory, num_programs, layouts: bool = False)
```

- `layouts=False` is today's behaviour byte for byte: no encoding is parsed, a `Value` stays a
  numpy array, and the level-1 tests are untouched.
- `layouts=True` makes the interpreter keep a side table `layout_of: dict[str, LinearLayout]`,
  filled lazily from `Type.encoding` through `parse_encoding` and `to_linear_layout`, with
  `ttg.num-warps` and `ttg.threads-per-warp` read from the module attributes. Values stay
  numpy arrays, so `ttsem/ops.py` is not touched.
- the checks live in a new `ttsem/layout_checks.py` registered as a post-op hook: after an op, the
  hook looks up the result layouts and asserts the property for that op name. A failure raises
  `LayoutViolation(op, reason)`, which `ttsem/validate.py` reports beside the value mismatch it
  already reports.
- an op with no layout meaning is skipped, so coverage grows one op at a time, and
  `interp.unsupported` keeps measuring level-1 coverage only.

An encoding the port does not know raises `EncodingParseError`, which the hook records the way
`interp.unsupported` records an unknown op: a layout that cannot be built is a gap in coverage,
never a silent pass.

## Level 3: agents, shared memory, and the race detector

Level 1 runs a launch as one sequential program and level 2 knows which lane holds which
element. Level 3 adds the one fact both ignore: the warps of a CTA are concurrent, and shared
memory is where they meet. The question it answers is Membar's question, asked dynamically:
**between two accesses to overlapping shared bytes from different agents, one of them a write,
is there a synchronisation?** Membar answers it statically with an alias model of buffers and
offsets; the two Membar defects found by hand this summer (#11324, #11325) were both errors of
that model. Level 3 executes the program and computes the byte sets, so it has no alias model
to get wrong.

### Agents and time

- **Agents** are warps. The default region of a `tt.func` has warps `0..num_warps-1`; each
  `ttg.warp_specialize` partition has its own warps and is its own *group*. Two accesses in
  the same group are ordered by program order when they come from the same warp, and by
  `ttg.barrier` (or `gpu.barrier`) when they do not: every warp of the group passes the barrier
  before any warp continues.
- **Time** in a group is the count of barriers executed so far, the *epoch*. One execution
  (level 1's) decides values and control flow, which is uniform across the warps of a group at
  the TTGIR level; the detector replays that execution and logs, for each shared-memory op and
  each warp, the byte set the warp touches, with the epoch.
- **Race** (intra-group): two logged accesses from different warps, overlapping bytes, at least
  one write, same epoch. Cross-group ordering goes through mbarriers only and is level 3.2.

### Byte sets

The bytes a warp touches are a function of layouts, not of values, which is why level 2 is
the prerequisite: `local_store` and `local_load` take the register layout of the tensor
(`register, lane, warp -> dim_i`) restricted to the warp, compose the view's offsets into the
allocation's coordinates, and apply the inverse of the allocation's shared layout
(`dim_i -> offset`) to get element offsets; the byte is `allocation.offset + offset *
elem_bytes` after `allocate-shared-memory`. `memdesc_index` and `memdesc_subslice` add
coordinates; `memdesc_reinterpret`, `memdesc_trans` and `memdesc_reshape` are followed only
when the byte image is the identity, otherwise the access is logged as the whole view.

### Asynchrony

An asynchronous write is *pending* from the op that issues it to the op that completes it, and
an access to its bytes by anyone in between is a race with it, the issuing warp included:

| issue | bytes | completes at |
|---|---|---|
| `ttg.async_copy_global_to_local` | the issuing warp's elements of the pointer tensor, mapped as a store | the warp's `ttg.async_wait` that covers its commit group |
| `ttng.async_tma_copy_global_to_local` | the whole destination view | `ttng.wait_barrier` on the barrier the copy names |
| `ttng.warp_group_dot` (reads) | the whole operand views | `ttng.warp_group_dot_wait` |

After completion the access counts as made by the completing warp at the epoch of completion.

### Interfaces

```python
class RaceInterp(LayoutInterp):
    accesses: list[Access]      # (group, warp, kind, lo, hi, epoch, op)
    races: list[Race]           # (a, b) pairs, first overlapping byte, count

def detect(module_text, record=None, num_warps=None) -> Report3  # record None: synthetic args
```

The calibration corpus is Membar's own: `test/Analysis/test-membar.mlir` after
`--allocate-shared-memory -test-print-membar` (120 functions, 156 barriers). With the
barriers in place the detector must report no intra-group race. With the barriers stripped it
reports one only where bytes actually cross warps, which on the lit corpus is 16 functions of
110: the lit cases mostly reload their own bytes, and Membar inserts a barrier regardless. The
sensitivity is measured on real kernels instead (`ttsem/races_corpus.py`, the fuzzer's pipelined
families), where every barrier can be removed one at a time and counted as necessary or not.
A function that shows a race *with* the barriers is either a detector defect or a Membar
defect, and #11325 is the template of the second kind. Numbers: `RESULTS.md`, "Level 3".

### Level 3.2: ordering across groups through mbarriers (first version in `ttsem/races.py`, 6 Sep)

Within a group, `ttg.barrier` counts are a total order and the epoch test is exact. Across
groups (the default region and each `warp_specialize` partition, the TMA engine, the tensor
core) the only orderings are mbarrier arrivals and waits, and the replay is sequential (the
default region runs before the partitions), so a wait can be replayed before the arrival it
pairs with. The model that fits is a trace one:

1. **Collect.** One replay produces, per group, the list of events in program order: shared
   accesses, barriers, `arrive_barrier` / `barrier_expect` / `async_tma_copy` (arrivals on a
   named barrier, the copy's arrival standing for its write), `wait_barrier` (a wait on a
   named barrier), and the exit of `warp_specialize` (an arrival of every group on one
   implicit barrier that the outer group waits on).
2. **Pair.** For every barrier, arrivals accumulate into phases of `count` arrivals
   (`init_barrier`'s count, or one per `barrier_expect` for a TMA phase); the k-th wait of a
   group on that barrier pairs with the k-th completed phase.
3. **Clock.** Vector clocks per group, one component per group; every event increments its
   own component; a wait joins the clocks of the arrivals of its phase. A fixpoint over the
   lists resolves waits whose arrivals come later in replay order.
4. **Race.** Two accesses from different groups, overlapping bytes, one a write, neither
   happens-before the other by the clocks. Same-group pairs keep the epoch test.

The first version does exactly this, with two simplifications: arrivals are cut into phases in
replay order (exact for the count-1 barriers of the pipelined producer/consumer pattern), and the
entry and exit of `warp_specialize` are one implicit barrier each (the outer group arrives at
the entry before the default region runs; every partition arrives at the exit; the k-th wait
pairs with the k-th phase). The cross-group check runs after the replay, on every pair of
accesses of different groups with overlapping bytes and a write among them.

What this catches that 3.1 cannot: a consumer partition reading a stage the producer has not
yet filled because the wait names the wrong barrier or phase, a producer overwriting a stage
the consumer is still reading because the `empty` arrival is missing, and a TMA store issued
before the `wait_barrier` that covers its source. The four hand-found barrier bugs of the
summer live in 3.1's territory except #11328 and #11404, which are cluster barriers: those
need a second CTA as a group, which the same clocks accommodate.
