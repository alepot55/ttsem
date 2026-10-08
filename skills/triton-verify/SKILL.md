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

Install once (Linux; Triton 3.8.0 and the CPU build of torch come with it):

```
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    'ttsem[triton] @ git+https://github.com/alepot55/ttsem'
```

Once ttsem is on PyPI, `'ttsem[triton]'` replaces the git requirement.

What you get that a GPU run does not give you:

- `oob_write` / `oob_read`: the line, the tensor argument, how many elements past the end. A GPU
  hides these because allocations are rounded up.
- `race`: two program instances touch the same element and the result depends on the schedule.
  One order of the instances may give the right values and another not, so an assertion on
  outputs may not fail.
- NaN everywhere in an output: the kernel read memory from `torch.empty` before writing it.
- Compile errors that `TRITON_INTERPRET=1` does not raise: this runs the real frontend.

Do not stop at the sizes that divide the block. Add, for every kernel: a size that is not a
multiple of the block (1000, not 1024), 0 and 1, a non-contiguous input, and a case where several
program instances write the same output.

When a run stops with `KernelFault`, fix the named line first and rerun; do not widen tolerances
or delete the failing case. `unsupported` means the semantics does not model an op: say so rather
than claiming the kernel was checked. Timing is out of scope: benchmarks are skipped.
