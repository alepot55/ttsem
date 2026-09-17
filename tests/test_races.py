"""Level 3: byte sets from layouts, epochs from barriers, pending asynchronous copies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ttsem import races
from ttsem.layouts import parse_encoding, to_linear_layout

FIXTURES = Path(__file__).resolve().parent / "fixtures"

BLOCKED = (
    "#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>"
)
SHARED = "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>"
TENSOR = f"tensor<128xi32, {BLOCKED}>"
# two elements per thread: warp 0 reads 0..63, which warp 1 wrote under BLOCKED (32..63)
WIDE = "#ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>"
TENSOR2 = f"tensor<128xi32, {WIDE}>"
MEMDESC = f"!ttg.memdesc<128xi32, {SHARED}, #ttg.shared_memory, mutable>"
PTRS = f"tensor<128x!tt.ptr<i32>, {BLOCKED}>"


def module(body: str) -> str:
    return f"""
"builtin.module"() ({{
  "tt.func"() <{{sym_name = "k", function_type = (!tt.ptr<i32>) -> ()}}> ({{
  ^bb0(%p: !tt.ptr<i32>):
    %r = "tt.make_range"() <{{end = 128 : i32, start = 0 : i32}}> : () -> {TENSOR}
    %buf = "ttg.local_alloc"() {{allocation.offset = 0 : i32}} : () -> {MEMDESC}
{body}
    "tt.return"() : () -> ()
  }}) : () -> ()
}}) {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32}} : () -> ()
"""


STORE = f'    "ttg.local_store"(%r, %buf) : ({TENSOR}, {MEMDESC}) -> ()'
LOAD = f'    %v = "ttg.local_load"(%buf) : ({MEMDESC}) -> {TENSOR2}'
BARRIER = '    "ttg.barrier"() : () -> ()'


def test_apply_many_agrees_with_apply() -> None:
    layout = to_linear_layout(parse_encoding(BLOCKED), [128], 4, 32)
    regs = np.arange(layout.get_in_dim_size("register"))[:, None]
    lanes = np.arange(32)[None, :]
    many = races.apply_many(layout, {"register": regs, "lane": lanes, "warp": 2, "block": 0})
    for r in range(regs.size):
        for lane in (0, 7, 31):
            one = layout.apply({"register": r, "lane": lane, "warp": 2, "block": 0})
            assert many["dim0"][r, lane] == one["dim0"]


def test_store_then_load_by_other_warps_races_without_a_barrier() -> None:
    (report,) = races.detect(module("\n".join([STORE, LOAD])))
    assert report.status == "ok", report.status
    assert report.races, "a load right after a store by other warps must race"
    a, b = next(iter(report.pairs))
    assert "local_store" in a and "local_load" in b
    (fenced,) = races.detect(module("\n".join([STORE, BARRIER, LOAD])))
    assert fenced.status == "ok" and not fenced.races
    assert fenced.barriers == 1


# Two CTAs, the tensor split between them: each CTA holds 128 of the 256 elements in a shared
# memory of its own. `CGALayout = [[1]]` is the printed form of that split on both sides.
CGA_BLOCKED = (
    "#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], "
    "CGALayout = [[1]]}>"
)
CGA_WIDE = (
    "#ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], "
    "CGALayout = [[1]]}>"
)
CGA_SHARED = (
    "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>"
)
CGA_MEMDESC = f"!ttg.memdesc<256xi32, {CGA_SHARED}, #ttg.shared_memory, mutable>"
CGA_TENSOR = f"tensor<256xi32, {CGA_BLOCKED}>"
CGA_ATTRS = '"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32'


def cga_module(body: str) -> str:
    return f"""
"builtin.module"() ({{
  "tt.func"() <{{sym_name = "k", function_type = (!tt.ptr<i32>) -> ()}}> ({{
  ^bb0(%p: !tt.ptr<i32>):
    %r = "tt.make_range"() <{{end = 256 : i32, start = 0 : i32}}> : () -> {CGA_TENSOR}
    %buf = "ttg.local_alloc"() {{allocation.offset = 0 : i32}} : () -> {CGA_MEMDESC}
{body}
    "tt.return"() : () -> ()
  }}) : () -> ()
}}) {{{CGA_ATTRS}}} : () -> ()
"""


def test_a_cga_split_allocation_is_modelled_per_cta() -> None:
    store = (
        f'    "ttg.local_store"(%r, %buf) : (tensor<256xi32, {CGA_BLOCKED}>, {CGA_MEMDESC}) -> ()'
    )
    same = f'    %v = "ttg.local_load"(%buf) : ({CGA_MEMDESC}) -> tensor<256xi32, {CGA_BLOCKED}>'
    wide = f'    %w = "ttg.local_load"(%buf) : ({CGA_MEMDESC}) -> tensor<256xi32, {CGA_WIDE}>'
    # the split layout is understood: no gap, and a warp reading its own bytes does not race
    (report,) = races.detect(cga_module("\n".join([store, same])))
    assert report.status == "ok", report.status
    assert not report.gaps, report.gaps
    assert not report.races
    # a load in another layout still exchanges bytes between the warps of one CTA
    (report,) = races.detect(cga_module("\n".join([store, wide])))
    assert report.races and all(a.exact for r in report.races for a in (r.first, r.second))
    (fenced,) = races.detect(cga_module("\n".join([store, BARRIER, wide])))
    assert not fenced.races and fenced.barriers == 1


CLUSTER = '    "ttng.cluster_barrier"() : () -> ()'
RELAXED = '    "ttng.cluster_barrier"() {relaxed = true} : () -> ()'


def test_a_cluster_barrier_orders_the_warps_unless_it_is_relaxed() -> None:
    (fenced,) = races.detect(module("\n".join([STORE, CLUSTER, LOAD])))
    assert fenced.status == "ok" and not fenced.races and fenced.barriers == 1
    (relaxed,) = races.detect(module("\n".join([STORE, RELAXED, LOAD])))
    assert relaxed.races and relaxed.barriers == 0
    # the ablation strips the cluster barrier like any other
    (stripped,) = races.detect(races.strip_barriers(module("\n".join([STORE, CLUSTER, LOAD]))))
    assert stripped.races


def test_a_warp_reading_its_own_elements_does_not_race() -> None:
    # the same layout on both sides: every warp loads exactly the bytes it stored
    same = f'    %v = "ttg.local_load"(%buf) : ({MEMDESC}) -> {TENSOR}'
    (report,) = races.detect(module("\n".join([STORE, same])))
    assert report.status == "ok" and not report.races


def test_async_copy_read_before_its_wait_races() -> None:
    ptrs = f'    %q = "tt.splat"(%p) : (!tt.ptr<i32>) -> {PTRS}'
    addr = f'    %a = "tt.addptr"(%q, %r) : ({PTRS}, {TENSOR}) -> {PTRS}'
    copy = (
        '    %t = "ttg.async_copy_global_to_local"(%a, %buf)'
        f" : ({PTRS}, {MEMDESC}) -> !ttg.async.token"
    )
    commit = '    %c = "ttg.async_commit_group"(%t) : (!ttg.async.token) -> !ttg.async.token'
    wait = (
        '    %w = "ttg.async_wait"(%c) <{num = 0 : i32}> : (!ttg.async.token) -> !ttg.async.token'
    )
    early = "\n".join([ptrs, addr, copy, commit, LOAD, wait])
    (report,) = races.detect(module(early))
    assert report.status == "ok", report.status
    assert report.races, "reading the destination before async_wait must race"
    proper = "\n".join([ptrs, addr, copy, commit, wait, BARRIER, LOAD])
    (clean,) = races.detect(module(proper))
    assert clean.status == "ok" and not clean.races
    no_barrier = "\n".join([ptrs, addr, copy, commit, wait, LOAD])
    (cross,) = races.detect(module(no_barrier))
    assert cross.races, "other warps' copies are not visible without a barrier after the wait"


def test_the_11325_class_is_found_when_its_barrier_is_removed() -> None:
    """The four Membar tests of PR #11325 after main's `--allocate-shared-memory
    -test-print-membar`: main inserts the barrier the PR asked for; without it the two
    reinterpret cases race and the two controls stay silent."""
    text = (FIXTURES / "membar_11325.generic").read_text()
    with_barriers = {r.function: r for m in races.split_modules(text) for r in races.detect(m)}
    assert len(with_barriers) == 4 and all(r.status == "ok" for r in with_barriers.values())
    assert not any(r.races for r in with_barriers.values())
    stripped = races.strip_barriers(text)
    without = {r.function: r for m in races.split_modules(stripped) for r in races.detect(m)}
    racing = sorted(name for name, r in without.items() if r.races)
    assert racing == [
        "subslice_offsets_after_reinterpret",
        "subslice_offsets_after_reinterpret_element_type",
    ]
    a, b = next(iter(without["subslice_offsets_after_reinterpret"].pairs))
    assert "local_store" in a and "local_load" in b


def test_an_op_with_scratch_of_its_own_orders_the_warps() -> None:
    """`convert_layout` through shared memory lowers with a `bar.sync` between its scratch
    store and load; Membar relies on that barrier and so does the detector."""
    scratch = "{allocation.offset = 4096 : i32, allocation.size = 512 : i32}"
    convert = f'    %c = "ttg.convert_layout"(%r) {scratch} : ({TENSOR}) -> {TENSOR2}'
    (report,) = races.detect(module("\n".join([STORE, convert, LOAD])))
    assert report.status == "ok" and not report.races
    plain = f'    %c = "ttg.convert_layout"(%r) : ({TENSOR}) -> {TENSOR2}'
    (unordered,) = races.detect(module("\n".join([STORE, plain, LOAD])))
    assert unordered.races, "a convert_layout without scratch orders nothing"


BARDESC = f"!ttg.memdesc<1xi64, {SHARED}, #ttg.shared_memory, mutable>"


def ws_module(default_body: str, partition_body: str) -> str:
    return f"""
"builtin.module"() ({{
  "tt.func"() <{{sym_name = "k", function_type = (!tt.ptr<i32>) -> ()}}> ({{
  ^bb0(%p: !tt.ptr<i32>):
    %r = "tt.make_range"() <{{end = 128 : i32, start = 0 : i32}}> : () -> {TENSOR}
    %buf = "ttg.local_alloc"() {{allocation.offset = 0 : i32}} : () -> {MEMDESC}
    %bar = "ttg.local_alloc"() {{allocation.offset = 1024 : i32}} : () -> {BARDESC}
    %bar2 = "ttg.local_alloc"() {{allocation.offset = 1032 : i32}} : () -> {BARDESC}
    "ttng.init_barrier"(%bar) <{{count = 1 : i32}}> : ({BARDESC}) -> ()
    "ttng.init_barrier"(%bar2) <{{count = 1 : i32}}> : ({BARDESC}) -> ()
    %ph = "arith.constant"() <{{value = 0 : i32}}> : () -> i32
    "ttg.warp_specialize"(%buf, %bar, %r) <{{partitionNumWarps = array<i32: 4>}}> ({{
{default_body}
      "ttg.warp_yield"() : () -> ()
    }}, {{
      "ttg.warp_specialize.partitions"(%buf, %bar, %r) ({{
      ^bb0(%b: {MEMDESC}, %ba: {BARDESC}, %rr: {TENSOR}):
{partition_body}
        "ttg.warp_return"() : () -> ()
      }}) : ({MEMDESC}, {BARDESC}, {TENSOR}) -> ()
    }}) : ({MEMDESC}, {BARDESC}, {TENSOR}) -> ()
    "tt.return"() : () -> ()
  }}) : () -> ()
}}) {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32}} : () -> ()
"""


WAIT = (
    '      "ttng.wait_barrier"(%bar, %ph) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>'
    f" : ({BARDESC}, i32) -> ()"
)
WAIT_OTHER = (
    '      "ttng.wait_barrier"(%bar2, %ph) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>'
    f" : ({BARDESC}, i32) -> ()"
)
LOAD_DEFAULT = f'      %v = "ttg.local_load"(%buf) : ({MEMDESC}) -> {TENSOR2}'
STORE_PART = f'        "ttg.local_store"(%rr, %b) : ({TENSOR}, {MEMDESC}) -> ()'
ARRIVE_PART = f'        "ttng.arrive_barrier"(%ba) <{{count = 1 : i32}}> : ({BARDESC}) -> ()'


def test_a_producer_partition_and_a_consumer_default_are_ordered_by_the_mbarrier() -> None:
    (report,) = races.detect(
        ws_module("\n".join([WAIT, LOAD_DEFAULT]), "\n".join([STORE_PART, ARRIVE_PART]))
    )
    assert report.status == "ok", report.status
    assert not report.races, [(r.first.op[:40], r.second.op[:40]) for r in report.races]


def test_a_consumer_that_does_not_wait_races_with_the_producer() -> None:
    (report,) = races.detect(ws_module(LOAD_DEFAULT, "\n".join([STORE_PART, ARRIVE_PART])))
    assert report.status == "ok", report.status
    assert report.races
    a, b = next(iter(report.pairs))
    assert "local_store" in a + b and "local_load" in a + b


def test_waiting_on_the_wrong_barrier_orders_nothing() -> None:
    (report,) = races.detect(
        ws_module("\n".join([WAIT_OTHER, LOAD_DEFAULT]), "\n".join([STORE_PART, ARRIVE_PART]))
    )
    assert report.status == "ok", report.status
    assert report.races


def test_an_inline_ptx_fragment_is_opaque_and_does_not_hide_the_race() -> None:
    asm = (
        f'    %a = "tt.elementwise_inline_asm"(%r) <{{asm_string = "mov.b32 $0, $1;", '
        f'constraints = "=r,r", packed_element = 1 : i32, pure = true}}> : ({TENSOR}) -> {TENSOR}'
    )
    store = f'    "ttg.local_store"(%a, %buf) : ({TENSOR}, {MEMDESC}) -> ()'
    (report,) = races.detect(module("\n".join([asm, store, LOAD])))
    assert report.status == "ok"
    assert report.opaque == 1
    assert report.races
    (fenced,) = races.detect(module("\n".join([asm, store, BARRIER, LOAD])))
    assert fenced.status == "ok" and not fenced.races
