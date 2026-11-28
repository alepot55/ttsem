"""Rescaling policies of the online softmax (M4 of the floating-point session).

The online softmax rescales its running sum at every merge, to the larger of the two maxima.
FlashAttention-4 rescales only when the maximum has grown by more than a threshold τ, keeping a
stale reference otherwise: the sum then holds terms up to e^τ relative to the reference, and a
rescale is skipped. Over the reals both are exact, but only up to the value the state encodes:
the conditional state's reference is not the maximum, so its merge is right up to the relation
"same normalizer d·e^m", not equal to the summary of the merged block (`Tree.eval_rel_block`).

In the kernel's loop (a left fold) the reference lags the maximum by at most τ; along a merge
tree it can lag by τ for every right step (the tree's right depth ρ), so the sum is bounded by
n·e^(τρ), not n·e^τ: `results/m4/` holds a right comb of depth 15 where τ = 8 overflows float32.
"""

from __future__ import annotations

from discovery import ir, problems

POLICIES: dict[str, str] = {
    # rescale at every merge: the library's online softmax
    "always": problems.SOFTMAX.known["online"],
    # rescale only when the right summary's maximum exceeds the reference by more than tau
    "conditional": """(program softmax-conditional
  (input x)
  (param tau 8)
  (pass a
    (state m d)
    (block (m (rmax x))
           (d (sum (exp (- x m)))))
    (merge (m (ifgt (- R.m L.m) tau R.m L.m))
           (d (+ (* L.d (exp (- L.m m))) (* R.d (exp (- R.m m)))))))
  (output elementwise (/ (exp (- x a.m)) a.d)))""",
}


def program(name: str) -> ir.Program:
    return problems.SOFTMAX.parse(POLICIES[name])
