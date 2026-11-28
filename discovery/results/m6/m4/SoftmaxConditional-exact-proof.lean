import FPStatements

namespace Discovery.FP.SoftmaxConditionalExact

open Discovery

theorem znorm_eq' (st : State_a) : znorm st = st.d * Real.exp st.m := by
  unfold znorm
  rw [zero_sub, Real.exp_neg]
  field_simp

theorem znorm_block' (tau : ℝ) (A : List ℝ) :
    znorm (block_a tau A) = (A.map Real.exp).sum := by
  rw [znorm_eq']
  simp only [block_a]
  rw [← List.sum_map_mul_right]
  congr 1
  apply List.map_congr_left
  intro x _
  simp [← Real.exp_add]

theorem znorm_merge' (tau : ℝ) (L R : State_a) :
    znorm (merge_a tau L R) = znorm L + znorm R := by
  rw [znorm_eq', znorm_eq', znorm_eq']
  simp only [merge_a]
  split_ifs
  · simp only [add_mul, mul_assoc, ← Real.exp_add]; simp
  · simp only [add_mul, mul_assoc, ← Real.exp_add]; simp

theorem znorm_tree' (tau : ℝ) : ∀ t : Discovery.Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) →
    znorm (t.eval (block_a tau) (merge_a tau)) = (t.flat.map Real.exp).sum
  | .leaf b, _ => by simpa [Tree.eval, Tree.flat] using znorm_block' tau b
  | .node l r, ht => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [Tree.leaves, hb])
    have hr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => ht b (by simp [Tree.leaves, hb])
    simp only [Tree.eval, Tree.flat]
    rw [znorm_merge', znorm_tree' tau l hl, znorm_tree' tau r hr]
    simp

theorem fp_bound : ∀ (tau : ℝ), 
  ∀ (X : List ℝ), X ≠ [] → ∀ (t : Discovery.Tree ℝ), t.flat = X →
    (∀ b ∈ t.leaves, b ≠ []) →
      znorm (t.eval (block_a tau) (merge_a tau)) = (X.map Real.exp).sum := by
  intro tau X _ t ht hl
  subst ht
  exact znorm_tree' tau t hl

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxConditionalExact
