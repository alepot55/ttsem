import FPStatements

namespace Discovery.FP.SoftmaxConditionalOverflow

open Discovery

theorem blk_aux (A : List ℝ) : 0 ≤ (A.map fun x => Real.exp (x - lmax A)).sum ∧
    (A.map fun x => Real.exp (x - lmax A)).sum ≤ A.length := by
  have h1 : ∀ l : List ℝ, (∀ x ∈ l, x ≤ lmax A) →
      0 ≤ (l.map fun x => Real.exp (x - lmax A)).sum ∧
      (l.map fun x => Real.exp (x - lmax A)).sum ≤ l.length := by
    intro l
    induction l with
    | nil => intro _; simp
    | cons a l ih =>
      intro h
      have := ih (fun x hx => h x (by simp [hx]))
      have ha : Real.exp (a - lmax A) ≤ 1 := by
        rw [Real.exp_le_one_iff]; linarith [h a (by simp)]
      have hp := Real.exp_pos (a - lmax A)
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      push_cast
      constructor <;> linarith [this.1, this.2]
  exact h1 A (fun x hx => le_lmax hx)

theorem tree_aux (tau : ℝ) (htau : 0 ≤ tau) : ∀ t : Discovery.Tree ℝ,
    (∀ b ∈ t.leaves, b ≠ []) →
    0 ≤ (t.eval (block_a tau) (merge_a tau)).d ∧
    (t.eval (block_a tau) (merge_a tau)).d ≤ (t.flat.length : ℝ) * Real.exp (tau * (rdepth t : ℝ))
  | .leaf b, _ => by
    have := blk_aux b
    simp only [Discovery.Tree.eval, Discovery.Tree.flat, rdepth, block_a]
    simpa using this
  | .node l r, ht => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [Discovery.Tree.leaves, hb])
    have hr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => ht b (by simp [Discovery.Tree.leaves, hb])
    obtain ⟨l0, l1⟩ := tree_aux tau htau l hl
    obtain ⟨r0, r1⟩ := tree_aux tau htau r hr
    set L := Discovery.Tree.eval (block_a tau) (merge_a tau) l
    set R := Discovery.Tree.eval (block_a tau) (merge_a tau) r
    simp only [Discovery.Tree.eval, Discovery.Tree.flat, rdepth, merge_a, List.length_append]
    simp only [Nat.cast_add]
    have e1 : Real.exp (tau * (rdepth l : ℝ)) ≤ Real.exp (tau * ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ)) := by
      apply Real.exp_le_exp.mpr
      apply mul_le_mul_of_nonneg_left _ htau
      exact_mod_cast le_max_left _ _
    have e2 : Real.exp (tau * ((rdepth r : ℝ) + 1)) ≤ Real.exp (tau * ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ)) := by
      apply Real.exp_le_exp.mpr
      apply mul_le_mul_of_nonneg_left _ htau
      have : rdepth r + 1 ≤ max (rdepth l) (rdepth r + 1) := le_max_right _ _
      exact_mod_cast this
    set E := Real.exp (tau * ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ))
    have nl : (0:ℝ) ≤ (l.flat.length : ℝ) := Nat.cast_nonneg _
    have nr : (0:ℝ) ≤ (r.flat.length : ℝ) := Nat.cast_nonneg _
    have e3 : Real.exp (tau * ((rdepth r : ℝ) + 1)) = Real.exp (tau * (rdepth r : ℝ)) * Real.exp tau := by
      rw [← Real.exp_add]; ring_nf
    split_ifs with h
    · simp only [sub_self, Real.exp_zero, mul_one]
      have : Real.exp (L.m - R.m) ≤ 1 := by
        rw [Real.exp_le_one_iff]; linarith
      have hp := Real.exp_pos (L.m - R.m)
      have a1 : L.d * Real.exp (L.m - R.m) ≤ L.d := by nlinarith
      have a2 : L.d ≤ l.flat.length * E := l1.trans (mul_le_mul_of_nonneg_left e1 nl)
      have a3 : R.d ≤ r.flat.length * E := by
        have : Real.exp (tau * (rdepth r : ℝ)) ≤ E := by
          have := Real.exp_pos tau
          have h1 := Real.one_le_exp htau
          nlinarith [Real.exp_pos (tau * (rdepth r : ℝ))]
        exact r1.trans (mul_le_mul_of_nonneg_left this nr)
      have hE := Real.exp_pos (tau * ((max (rdepth l) (rdepth r + 1) : ℕ) : ℝ))
      constructor
      · have := mul_nonneg l0 hp.le; positivity
      · nlinarith
    · simp only [sub_self, Real.exp_zero, mul_one]
      have : Real.exp (R.m - L.m) ≤ Real.exp tau := Real.exp_le_exp.mpr (by linarith)
      have hp := Real.exp_pos (R.m - L.m)
      have a1 : R.d * Real.exp (R.m - L.m) ≤ R.d * Real.exp tau := mul_le_mul_of_nonneg_left this r0
      have a2 : L.d ≤ l.flat.length * E := l1.trans (mul_le_mul_of_nonneg_left e1 nl)
      have a3 : R.d * Real.exp tau ≤ r.flat.length * E := by
        calc R.d * Real.exp tau ≤ (r.flat.length * Real.exp (tau * (rdepth r : ℝ))) * Real.exp tau :=
              mul_le_mul_of_nonneg_right r1 (Real.exp_pos tau).le
          _ = r.flat.length * Real.exp (tau * ((rdepth r : ℝ) + 1)) := by rw [e3]; ring
          _ ≤ _ := mul_le_mul_of_nonneg_left e2 nr
      constructor
      · have := mul_nonneg r0 hp.le; positivity
      · nlinarith

theorem fp_bound : ∀ (tau : ℝ), 0 ≤ tau → 
  ∀ (X : List ℝ), X ≠ [] → ∀ (t : Discovery.Tree ℝ), t.flat = X →
    (∀ b ∈ t.leaves, b ≠ []) →
      (t.eval (block_a tau) (merge_a tau)).d ≤ (X.length : ℝ) * Real.exp (tau * (rdepth t : ℝ)) := by
  intro tau htau X _ t ht hl
  subst ht
  exact (tree_aux tau htau t hl).2

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxConditionalOverflow
