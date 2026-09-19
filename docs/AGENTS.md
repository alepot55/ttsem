# For coding agents: check a Triton kernel without a GPU

You are writing or changing a Triton kernel and the machine you run on has no GPU. You can still
run it, and you get more than a GPU run gives.

```
pip install -e .           # from a clone of this repository; needs torch (CPU is enough) and triton
python -m ttsem.sanitize pytest tests/ -q        # a test suite, as it is written for a GPU
python -m ttsem.sanitize script.py               # any script that launches kernels
```

Nothing in your code changes: `device="cuda"`, `.cuda()` and `.to("cuda")` get the CPU, every
kernel is compiled by Triton's real frontend and executed by ttsem's semantics of the IR, and the
result is written back into your tensors, so your own assertions run on what the kernel computed.
Benchmarks are skipped; an autotuned kernel has **every** configuration executed once.

A launch stops with a `KernelFault`, the failure of the test that launched it, when the kernel

| verdict | what it means | why a GPU run does not tell you |
|---|---|---|
| `oob_write`, `oob_read` | an access outside the tensors it was given: which line, which argument, how many elements past the end | the allocator rounds sizes up, the access lands in padding, every test passes |
| `race` | two program instances touch the same element and the result depends on which runs first | run one after the other the output is right; on a GPU it is wrong now and then |
| `poison` | a value the IR leaves undefined reaches a store | it is whatever the hardware happens to do |
| a wrong result full of NaN | the kernel read memory nobody wrote: `torch.empty` and its kin are poisoned here | fresh pages are zero more often than not, so accumulating into `torch.empty` passes until the allocator reuses memory |
| compile errors | code that `TRITON_INTERPRET=1` accepts and the compiler rejects (`triton.cdiv` on a runtime value inside `@triton.jit`, ...) | the interpreter is Python; this runs the real frontend |

The message names the line of **your** source:

```
KernelFault: kernel `add_kernel`, launch 1: `kernel.py:15: tl.store(out_ptr + offs, x + y)` writes
24 element(s) past the end of the tensor passed as `out_ptr`. A GPU run usually hides this
(allocations are rounded up). Check the mask of this access against the tensor's real size, for
every program id and for sizes that are not a multiple of the block.
```

What to test once it runs: sizes that are **not** a multiple of the block (a test on n = 1024
cannot see a missing mask; n = 1000 can), n = 0 and n = 1, non-contiguous inputs, and more than
one program instance writing to the same output.

Limits, said plainly: it is an interpreter, so millions of elements take seconds to minutes; a
launch that uses an op the semantics does not model stops with `unsupported` rather than a
guess; timing and performance are out of scope; the semantics is calibrated against real devices
on Triton's own test suite (see `docs/RESULTS.md`), not proved.
