import FPStatements

namespace Discovery.FP.SoftmaxOnline

open Discovery Discovery.Library.SoftmaxOnline

-- No helper lemmas: fp_bound is false as stated (counterexample above), so no proof exists.

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List ℝ), X ≠ [] →
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((2 : ℝ) * ((F.u * (hi - lo) * ((t.depth : ℝ) + (1 : ℝ))) + (F.u * ((D : ℝ) + ((2 : ℝ) * (t.depth : ℝ)))) + (F.ue * ((t.depth : ℝ) + (1 : ℝ))))) + ((4 : ℝ) * (X.length : ℝ) * ((t.depth : ℝ) + (1 : ℝ)) * F.ηe) + ((4 : ℝ) * (X.length : ℝ) * F.η)) := by
  -- No proof: the statement is refuted by the counterexample above
  -- (u = ue = ul = 0, η = ηe = 2^-100, left comb tree with about 2^100 leaves, st.d = 0).

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxOnline
