---
name: triton-verify
description: Use when writing, changing or reviewing a Triton GPU kernel and no GPU is available (or before trusting a GPU run) - runs the kernel's tests on the CPU under an executable semantics and reports out-of-bounds accesses, races between program instances and uninitialised reads with the kernel's own source line
---

# Verify a Triton kernel without a GPU

Run the project's tests, unchanged, under ttsem:

```
python -m ttsem.sanitize pytest <paths or pytest args>      # a suite written for device="cuda"
python -m ttsem.sanitize script.py                           # any script that launches kernels
```

Install once with `pip install -e ".[triton]"` from a clone of https://github.com/alepot55/ttsem
plus `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

What you get that a GPU run does not give you:

- `oob_write` / `oob_read`: the line, the tensor argument, how many elements past the end. A GPU
  hides these because allocations are rounded up.
- `race`: two program instances touch the same element and the result depends on the schedule.
  The output is right when instances run one after the other, so no assertion on outputs fails.
- NaN everywhere in an output: the kernel read memory from `torch.empty` before writing it.
- Compile errors that `TRITON_INTERPRET=1` does not raise: this runs the real frontend.

Do not stop at the sizes that divide the block. Add, for every kernel: a size that is not a
multiple of the block (1000, not 1024), 0 and 1, a non-contiguous input, and a case where several
program instances write the same output.

When a run stops with `KernelFault`, fix the named line first and rerun; do not widen tolerances
or delete the failing case. `unsupported` means the semantics does not model an op: say so rather
than claiming the kernel was checked. Timing is out of scope: benchmarks are skipped.
