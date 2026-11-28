import FPStatements

namespace Discovery.FP.SoftmaxConditionalExact

open Discovery

theorem znorm_eq (st : State_a) : znorm st = st.d * Real.exp st.m := by
  unfold znorm
  rw [zero_sub, Real.exp_neg]
  have := Real.exp_pos st.m
  field_simp

theorem sum_exp_shift (A : List ℝ) (m : ℝ) :
    (A.map fun x => Real.exp (x - m)).sum * Real.exp m = (A.map Real.exp).sum := by
  induction A with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, add_mul]
    rw [ih, Real.exp_sub]
    have := Real.exp_pos m
    field_simp

theorem eval_inv (tau : ℝ) : ∀ t : Discovery.Tree ℝ,
    znorm (t.eval (block_a tau) (merge_a tau)) = (t.flat.map Real.exp).sum
  | .leaf b => by
    simp only [Tree.eval, Tree.flat, znorm_eq, block_a]
    exact sum_exp_shift b _
  | .node l r => by
    have hl := eval_inv tau l
    have hr := eval_inv tau r
    rw [znorm_eq] at hl hr
    simp only [Tree.eval, Tree.flat, List.map_append, List.sum_append, ← hl, ← hr, znorm_eq, merge_a]
    generalize (l.eval (block_a tau) (merge_a tau)) = L
    generalize (r.eval (block_a tau) (merge_a tau)) = R
    have key : ∀ m : ℝ, (L.d * Real.exp (L.m - m) + R.d * Real.exp (R.m - m)) * Real.exp m
        = L.d * Real.exp L.m + R.d * Real.exp R.m := by
      intro m
      simp only [Real.exp_sub]
      have := Real.exp_pos m
      field_simp
    split_ifs <;> exact key _

theorem fp_bound : ∀ (tau : ℝ), 
  ∀ (X : List ℝ), X ≠ [] → ∀ (t : Discovery.Tree ℝ), t.flat = X →
    (∀ b ∈ t.leaves, b ≠ []) →
      znorm (t.eval (block_a tau) (merge_a tau)) = (X.map Real.exp).sum := by
  intro tau X _ t ht _
  subst ht
  exact eval_inv tau t

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxConditionalExact
