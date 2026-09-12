# Contributing

1. **Run the tests**: `python -m pytest -q tests` (554 tests, 538 of them without the Triton wheel, no GPU and no `triton-opt`
   needed). Anything marked `gpu` needs a card and a Triton build; run it with `-m gpu`.
2. **Add an op** in `ttsem/ops.py`: one function per op name, registered in `OPS`, with the
   signature `fn(interp, op, args) -> list[Value]`. Region-carrying ops call
   `interp.run_region(region, args)`; numpy only, no torch.
3. **Every op gets a test** in `tests/`, a two-line module in generic MLIR form that the parser and
   the interpreter run end to end. Prefer the smallest module that pins the choice you made.
4. **If the answer is not obvious from the docs**, it is a semantic choice: add one line to
   `SEMANTICS.md` saying what was picked and why, with the tie-breakers in order (the device, then
   the shipped interpreter, then LLVM's equivalent scalar op).
5. **Never guess**: an op you do not model raises `Unsupported(name)`, which is counted, rather
   than a value that happens to look right.
6. **A device deviation is a finding, not a defect to hide.** When the device disagrees and the
   IR does not determine the answer, add it to "Device deviations" in `SEMANTICS.md` with the
   launch that shows it, and leave it a `mismatch`. The semantics is never bent toward the hardware
   to make a number come out.
7. **Reporting a mismatch**: open an issue with the IR dump (`MLIR_ENABLE_DUMP=1`, or the module
   the stage received), the input buffers or the seed that generates them, the Triton commit or
   wheel version, and the device. Without the IR and the inputs a mismatch cannot be reproduced,
   and an unreproducible mismatch cannot be triaged.
8. **A smaller witness is a better one**: `python -m ttsem minimize <program>` reduces a module
   while the pass still changes its meaning, with no GPU involved.
