# The Triton code of torch.compile, under ttsem

TorchInductor turns a PyTorch program into Triton kernels. It emits Triton for CPU tensors too
(`torch._inductor.config.cpu_backend = "triton"`, the same lowering and the same passes), so the
kernels `torch.compile` generates can run under ttsem on a machine with no GPU. When a pass gets
aliasing wrong, a kernel reads memory that another program instance of the same launch writes;
ttsem stops at the kernel line where that happens.

## The demo

```console
$ python -m ttsem demo --torch
torch 2.14.0+cpu, no GPU: Inductor's Triton code run under ttsem

pytorch#197829: x[1:] = x[:-1].clone()
  eager     [0.0, 0.0, 1.0, 2.0, 3.0, 4.0] ...
  ttsem     race: kernel `triton_poi_fused_slice_0`, launch 0: `cba7hipldmr4zbbjqbevoicrvi5chj4lmfqmc7i3fvct4pu2irp3.py:25: tl.store(out_ptr0 + (1 + x0), tmp0, xmask)`: program instances (0, 0, 0) and (1, 0, 0) both touch element 16 of `out_ptr0`: (0, 0, 0) stores to it, (1, 0, 0) loads it (`cba7hipldmr4zbbjqbevoicrvi5chj4lmfqmc7i3fvct4pu2irp3.py:24: tmp0 = tl.load(in_ptr0 + (x0), xmask)`). ...
  upstream  fixed on main by pytorch#198010 (22 Sep 2026)

pytorch#198033: torch._foreach_add_([x], [x.flip(0)])
  eager     [63.0, 63.0, 63.0, 63.0, 63.0, 63.0] ...
  ttsem     race: kernel `triton_poi_fused_0`, launch 0: `cmk55yrd6qkwbkwa4zwmxwf7z2qbxctma7ideim6wbd4keifib5a.py:27: tl.store(out_ptr0 + (x0), tmp2, xmask)`: ...
  ...

2 of 2 flagged on this torch.
```

(Shortened where `...` stands: each race message goes on to say how many bytes the two instances
share and what to change.) This run is torch 2.14.0+cpu on a 4-core VM, 18 s. On a torch that
carries the fix, a case says `ok`.

## Code you already have

The module `torch._inductor.utils.run_and_get_code` returns, saved as `output_code.py`, runs under
ttsem on any CPU:

```python
import runpy, torch._inductor.utils as inductor_utils
from ttsem import sanitize
inductor_utils.print_performance = lambda fn, *a, **k: fn()  # call the graph once
sanitize.inductor_names(block=16)  # cap the block: one instance per tensor hides every race
with sanitize.session():  # every Triton launch runs under ttsem; a fault raises KernelFault
    code = runpy.run_path("output_code.py", run_name="output_code")
    code["benchmark_compiled_module"](code["get_args"](), times=1, repeat=1)  # torch >= 2.10
```

On the code torch 2.14 emits for `x[1:] = x[:-1].clone()` on CPU tensors (with
`cpu_backend = "triton"`), this stops with the same `KernelFault` as the demo, at line 25 of the
kernel. Code emitted for CUDA tensors goes through the same names (`inductor_names` answers
`_empty_strided_cuda` and the kernel's switch to the GPU driver), but it was not tried here.

Producing `output_code.py` is the part that needs care on a machine with no GPU: Inductor compiles
the Triton it emits, and with no active Triton driver that compile fails (`0 active drivers`). The
demo gets around it by compiling inside a ttsem session, which supplies a driver; see
`run_case` in `ttsem/torch_demo.py`. Code saved on a machine with a GPU needs none of this.

## The block cap

A race needs two program instances on the same data. At a test's small shape Inductor's default
block covers the whole tensor with one program instance, and nothing can race.
`sanitize.inductor_names(block=16)` caps `XBLOCK`, `YBLOCK` and `ZBLOCK` of the config Inductor
launches with; a generated kernel is valid under every block size its autotuner may pick, so the
cap changes the schedule, not the meaning.

Older Inductor output launches its kernels directly, as `kernel[grid](..., XBLOCK=64)` with a grid
computed from the block. For that code, set `TTSEM_BLOCK_CAP=16` in the environment: inside a
session, a launch whose grid is a function of the block gets its `XBLOCK`, `YBLOCK` and `ZBLOCK`
capped the same way, and the grid grows to cover the same elements. A grid given as numbers is
left alone. The judge passes the variable into its sandbox.

## What is not here

The numbers on `torch.compile` in the README (3,292 generated in-place programs, 856 wrong on an RTX
4070, all 856 flagged with no GPU) come from a generator of PyTorch programs that is not in this
repository; the bugs it found upstream are in
[LEDGER.md](https://github.com/alepot55/ttsem/blob/main/LEDGER.md).
