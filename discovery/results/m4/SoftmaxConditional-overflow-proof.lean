import FPStatements

namespace Discovery.FP.SoftmaxConditionalOverflow

open Discovery

theorem sm_aux (tau : ℝ) (htau : 0 ≤ tau) : ∀ t : Discovery.Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) →
    0 ≤ (t.eval (block_a tau) (merge_a tau)).d ∧
    (t.eval (block_a tau) (merge_a tau)).d ≤ (t.flat.length : ℝ) * Real.exp (tau * (rdepth t : ℝ))
  | .leaf b, hb => by
    have hne : b ≠ [] := hb b (by simp [Discovery.Tree.leaves])
    simp only [Discovery.Tree.eval, Discovery.Tree.flat, rdepth, block_a]
    simp only [Nat.cast_zero, mul_zero, Real.exp_zero, mul_one]
    constructor
    · apply List.sum_nonneg
      intro x hx
      simp only [List.mem_map] at hx
      obtain ⟨y, _, rfl⟩ := hx
      exact (Real.exp_pos _).le
    · have := List.sum_le_length_nsmul (b.map fun x => Real.exp (x - lmax b)) 1 (by
        intro x hx
        simp only [List.mem_map] at hx
        obtain ⟨y, hy, rfl⟩ := hx
        rw [Real.exp_le_one_iff]
        linarith [le_lmax hy])
      simpa using this
  | .node l r, hb => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b h => hb b (by simp [Discovery.Tree.leaves, h])
    have hr : ∀ b ∈ r.leaves, b ≠ [] := fun b h => hb b (by simp [Discovery.Tree.leaves, h])
    obtain ⟨l0, l1⟩ := sm_aux tau htau l hl
    obtain ⟨r0, r1⟩ := sm_aux tau htau r hr
    simp only [Discovery.Tree.eval, Discovery.Tree.flat, rdepth, List.length_append, Nat.cast_add]
    have nl0 : (0:ℝ) ≤ l.flat.length := Nat.cast_nonneg _
    have nr0 : (0:ℝ) ≤ r.flat.length := Nat.cast_nonneg _
    generalize (l.eval (block_a tau) (merge_a tau)) = L at *
    generalize (r.eval (block_a tau) (merge_a tau)) = R at *
    generalize (l.flat.length : ℝ) = nl at *
    generalize (r.flat.length : ℝ) = nr at *
    have e1 : Real.exp (tau * (rdepth l : ℝ)) ≤ Real.exp (tau * ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ)) := by
      apply Real.exp_le_exp.mpr
      apply mul_le_mul_of_nonneg_left _ htau
      exact_mod_cast le_max_left _ _
    have e2 : Real.exp (tau * (rdepth r : ℝ)) * Real.exp tau ≤ Real.exp (tau * ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ)) := by
      rw [← Real.exp_add]
      apply Real.exp_le_exp.mpr
      have : (rdepth r : ℝ) + 1 ≤ ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ) := by
        exact_mod_cast le_max_right _ _
      nlinarith [mul_le_mul_of_nonneg_left this htau]
    have e3 : Real.exp (tau * (rdepth r : ℝ)) ≤ Real.exp (tau * ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ)) := by
      have h1 : 1 ≤ Real.exp tau := Real.one_le_exp htau
      have h2 := Real.exp_pos (tau * (rdepth r : ℝ))
      nlinarith
    generalize Real.exp (tau * ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ)) = E at *
    generalize Real.exp (tau * (rdepth l : ℝ)) = El at *
    generalize Real.exp (tau * (rdepth r : ℝ)) = Er at *
    have P := Real.exp_pos tau
    unfold merge_a
    simp only
    split_ifs with h
    · simp only [sub_self, Real.exp_zero, mul_one]
      have h3 : Real.exp (L.m - R.m) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
      have h2 := Real.exp_pos (L.m - R.m)
      constructor
      · positivity
      · nlinarith [mul_le_mul_of_nonneg_left h3 l0, mul_le_mul_of_nonneg_left e1 nl0, mul_le_mul_of_nonneg_left e3 nr0]
    · simp only [sub_self, Real.exp_zero, mul_one]
      have h3 : Real.exp (R.m - L.m) ≤ Real.exp tau := Real.exp_le_exp.mpr (by linarith)
      have h2 := Real.exp_pos (R.m - L.m)
      constructor
      · positivity
      · nlinarith [mul_le_mul_of_nonneg_left h3 r0, mul_le_mul_of_nonneg_left e1 nl0, mul_le_mul_of_nonneg_left e2 nr0]

theorem fp_bound : ∀ (tau : ℝ), 0 ≤ tau → 
  ∀ (X : List ℝ), X ≠ [] → ∀ (t : Discovery.Tree ℝ), t.flat = X →
    (∀ b ∈ t.leaves, b ≠ []) →
      (t.eval (block_a tau) (merge_a tau)).d ≤ (X.length : ℝ) * Real.exp (tau * (rdepth t : ℝ)) := by
  intro tau htau X _ t hX hb
  subst hX
  exact (sm_aux tau htau t hb).2

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxConditionalOverflow
