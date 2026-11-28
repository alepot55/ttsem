import FPStatements

namespace Discovery.FP.SoftmaxLse

open Discovery Discovery.Library.SoftmaxLse

-- No helper lemmas: `fp_bound` as stated is false (see the counterexample above:
-- the final rounding `r7 = (v_m + r6)(1+δ)` compounds as (1+u)^depth, while the bound is linear in depth).

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List ℝ), X ≠ [] →
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ ((2 : ℝ) * ((t.depth : ℝ) + (1 : ℝ)) * ((F.u * (hi - lo)) + ((2 : ℝ) * F.u * (X.length : ℝ)) + (F.u * (max |lo| |hi|)) + ((2 : ℝ) * F.u) + F.ue + (F.ul * (X.length : ℝ)) + ((X.length : ℝ) * F.ηe) + ((D : ℝ) * F.u))) := by
  -- I cannot provide a valid proof: the statement is refuted by the counterexample above
  -- (u = 1/256, ue = ul = η = ηe = 0, D = 0, X = 2001 copies of 10^6, caterpillar tree of depth 2000).
  -- A provable variant would need a bound of the form (1+u)^depth − 1 rather than a linear one in depth,
  -- or a hypothesis such as depth·u ≤ 1 so that the compounding is dominated by the linear term.

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxLse
