import FPStatements

namespace Discovery.FP.Cand4f04f56628

open Discovery

theorem abs_log_sandwich {y z γ : ℝ} (hy : 0 < y) (hz : 0 < z) (hγ0 : 0 ≤ γ) (hγ : γ ≤ 1/11)
    (h1 : z ≤ y * (1 + γ)) (h2 : y * (1 - γ) ≤ z) : |Real.log z - Real.log y| ≤ 11/10 * γ := by
  rw [abs_le]; constructor
  · have h3 : Real.log (y/z) ≤ y/z - 1 := Real.log_le_sub_one_of_pos (div_pos hy hz)
    rw [Real.log_div hy.ne' hz.ne'] at h3
    have hy11 : y ≤ 11/10 * z := by nlinarith
    have : y/z - 1 ≤ 11/10*γ := by
      rw [div_sub_one hz.ne', div_le_iff₀ hz]
      nlinarith [mul_le_mul_of_nonneg_left hy11 hγ0]
    linarith
  · have h3 := Real.log_le_sub_one_of_pos (div_pos hz hy)
    rw [Real.log_div hz.ne' hy.ne'] at h3
    have : z/y - 1 ≤ γ := by rw [div_sub_one hy.ne', div_le_iff₀ hy]; nlinarith
    linarith

theorem x_exp_neg_le {x : ℝ} (hx : 0 ≤ x) : x * Real.exp (-x) ≤ 1 := by
  have := Real.add_one_le_exp x
  rw [Real.exp_neg, mul_inv_le_iff₀ (Real.exp_pos x)]
  linarith

theorem exp_round_arg {d δ u : ℝ} (hd : 0 ≤ d) (hu0 : 0 ≤ u) (hu : u ≤ 1/256) (hδ : |δ| ≤ u) :
    Real.exp (-d) - u ≤ Real.exp (-d * (1+δ)) ∧ Real.exp (-d * (1+δ)) ≤ Real.exp (-d) + 101/100 * u := by
  have hδ' := abs_le.mp hδ
  have hp := Real.exp_pos (-d)
  have hrw : Real.exp (-d * (1+δ)) = Real.exp (-d) * Real.exp (-(d*δ)) := by
    rw [← Real.exp_add]; congr 1; ring
  constructor
  · have h1 := Real.add_one_le_exp (-(d*δ))
    have h2 : d * u ≥ d * δ := mul_le_mul_of_nonneg_left hδ'.2 hd
    have h3 := x_exp_neg_le hd
    rw [hrw]
    nlinarith [mul_le_mul_of_nonneg_left h1 hp.le, mul_nonneg hu0 (sub_nonneg.2 h3)]
  · have h2 : -(d * δ) ≤ u * d := by nlinarith [mul_le_mul_of_nonneg_left hδ'.1 hd]
    have h4 : Real.exp (-(d*δ)) ≤ Real.exp (u*d) := Real.exp_le_exp.2 h2
    have h5 := Real.add_one_le_exp (-(u*d))
    have h6 : Real.exp (u*d) * (1 - u*d) ≤ 1 := by
      have : Real.exp (u*d) * Real.exp (-(u*d)) = 1 := by rw [← Real.exp_add]; simp
      nlinarith [Real.exp_pos (u*d)]
    -- exp(-d) * exp(u d) = exp(-(1-u) d)
    have h7 := x_exp_neg_le (mul_nonneg (by linarith : (0:ℝ) ≤ 1 - u) hd)
    have h8 : Real.exp (-d) * Real.exp (u*d) = Real.exp (-((1-u)*d)) := by
      rw [← Real.exp_add]; congr 1; ring
    have h9 : Real.exp (-d) * (Real.exp (u*d) - 1) ≤ Real.exp (-d) * (u*d*Real.exp (u*d)) := by
      apply mul_le_mul_of_nonneg_left _ hp.le; nlinarith
    have h10 : Real.exp (-d) * (u*d*Real.exp (u*d)) = u * d * Real.exp (-((1-u)*d)) := by
      rw [← h8]; ring
    rw [hrw]
    have h11 : Real.exp (-d) * Real.exp (-(d*δ)) ≤ Real.exp (-d) * Real.exp (u*d) :=
      mul_le_mul_of_nonneg_left h4 hp.le
    have h12 : d * Real.exp (-((1-u)*d)) ≤ 256/255 := by
      have : (1-u) * (d * Real.exp (-((1-u)*d))) ≤ 1 := by nlinarith
      nlinarith [mul_nonneg hd (Real.exp_pos (-((1-u)*d))).le]
    have h13 : u * (d * Real.exp (-((1-u)*d))) ≤ u * (256/255) := mul_le_mul_of_nonneg_left h12 hu0
    nlinarith
theorem small_side {F : Rounding} {d r1 rs : ℝ} (hd : 0 ≤ d) (hu : F.u ≤ 1/256)
    (h1 : F.add (-d) r1) (h2 : F.exp r1 rs) :
    (1 - F.ue) * (Real.exp (-d) - F.u) - F.ηe ≤ rs ∧
      rs ≤ (1 + F.ue) * (Real.exp (-d) + 101/100 * F.u) + F.ηe := by
  obtain ⟨δ, hδ, rfl⟩ := h1
  obtain ⟨δ2, e2, hδ2, he2, rfl, h0⟩ := h2
  have hX := exp_round_arg hd F.hu hu hδ
  have hX0 := Real.exp_pos (-d * (1+δ))
  have a2 := abs_le.mp hδ2
  have a3 := abs_le.mp he2
  have := F.hue1
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left a2.1 hX0.le, mul_le_mul_of_nonneg_left hX.1 (by linarith : (0:ℝ) ≤ 1 - F.ue)]
  · nlinarith [mul_le_mul_of_nonneg_left a2.2 hX0.le, mul_le_mul_of_nonneg_left hX.2 (by linarith : (0:ℝ) ≤ 1 + F.ue)]

theorem big_side {F : Rounding} {r3 r4 : ℝ} (h1 : F.add 0 r3) (h2 : F.exp r3 r4) :
    (1 - F.ue) - F.ηe ≤ r4 ∧ r4 ≤ (1 + F.ue) + F.ηe := by
  obtain ⟨δ, hδ, rfl⟩ := h1
  obtain ⟨δ2, e2, hδ2, he2, rfl, h0⟩ := h2
  have a2 := abs_le.mp hδ2
  have a3 := abs_le.mp he2
  simp only [zero_mul, Real.exp_zero, one_mul]
  constructor <;> linarith

theorem merge_sum {F : Rounding} {d r1 r2 r3 r4 r5 : ℝ} (hd : 0 ≤ d) (hu : F.u ≤ 1/256)
    (hue : F.ue ≤ 1/64) (hηu : F.ηe ≤ F.u)
    (h1 : F.add (-d) r1) (h2 : F.exp r1 r2) (h3 : F.add 0 r3) (h4 : F.exp r3 r4)
    (h5 : F.add (r2 + r4) r5) :
    (Real.exp (-d) + 1) * (1 - (F.ue + 21/10 * F.u + 2 * F.ηe)) ≤ r5 ∧
    r5 ≤ (Real.exp (-d) + 1) * (1 + (F.ue + 21/10 * F.u + 2 * F.ηe)) := by
  have s1 := small_side hd hu h1 h2
  have s2 := big_side h3 h4
  obtain ⟨δ, hδ, rfl⟩ := h5
  have a := abs_le.mp hδ
  have hp := Real.exp_pos (-d)
  have hp1 : Real.exp (-d) ≤ 1 := by rw [Real.exp_le_one_iff]; linarith
  have := F.hu; have := F.hue; have := F.hη; have := F.hηe; have := F.hue1
  set p := Real.exp (-d)
  set S := r2 + r4 with hS
  -- S bounds
  have hSl : (p+1) * (1 - (F.ue + F.u + 2*F.ηe)) ≤ S := by
    nlinarith [mul_nonneg F.hue F.hu]
  have hSu : S ≤ (p+1) * (1 + (F.ue + 103/100*F.u + 2*F.ηe)) := by
    nlinarith [mul_nonneg F.hue F.hu, mul_nonneg hp.le F.hu, mul_nonneg hp.le F.hηe, mul_nonneg hp.le F.hue]
  have hS0 : 0 ≤ S := by nlinarith [mul_nonneg F.hue F.hu]
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left a.1 hS0, mul_nonneg F.hue F.hu, mul_nonneg F.hu F.hηe]
  · nlinarith [mul_le_mul_of_nonneg_left a.2 hS0, mul_nonneg F.hue F.hu, mul_nonneg F.hu F.hηe]
theorem merge_tail {F : Rounding} (hu : F.u ≤ 1/256) (hul : F.ul ≤ 1/64)
    {mm Mm p r5 r6 r7 γ : ℝ} (hp0 : 0 < p) (hp1 : p ≤ 1) (hγ0 : 0 ≤ γ) (hγ : γ ≤ 1/30)
    (h5 : (p+1)*(1-γ) ≤ r5 ∧ r5 ≤ (p+1)*(1+γ)) (hL : F.log r5 r6) (h7 : F.add (mm + r6) r7)
    (hM : |mm| ≤ Mm) :
    |r7 - (mm + Real.log (p+1))| ≤ F.u * Mm + 106/100 * F.u + 11/10 * γ + 104/100 * F.ul := by
  have hr5 : 0 < r5 := lt_of_lt_of_le (by nlinarith) h5.1
  have hs := abs_log_sandwich (y := p+1) (z := r5) (by linarith) hr5 hγ0 (by linarith) h5.2 h5.1
  have hlog1 : Real.log (p+1) ≤ 1 := by
    have := Real.log_le_sub_one_of_pos (show 0 < p + 1 by linarith); linarith
  have hlog0 : 0 ≤ Real.log (p+1) := Real.log_nonneg (by linarith)
  have hs' := abs_le.mp hs
  have hL1 : |Real.log r5| ≤ 104/100 := by
    rw [abs_le]; constructor <;> linarith
  obtain ⟨δl, hδl, rfl⟩ := hL
  obtain ⟨δ7, hδ7, rfl⟩ := h7
  have hu0 := F.hu
  have hul0 := F.hul
  set L := Real.log r5 with hLdef
  have e1 : (mm + L * (1+δl)) * (1+δ7) - (mm + Real.log (p+1))
      = δ7 * (mm + L*(1+δl)) + (L - Real.log (p+1)) + δl * L := by ring
  have b1 : |L * (1+δl)| ≤ 106/100 := by
    rw [abs_mul]
    have : |1+δl| ≤ 1 + 1/64 := by
      calc |1+δl| ≤ |1| + |δl| := abs_add_le _ _
        _ ≤ 1 + 1/64 := by simp; linarith
    have h0 : 0 ≤ |1+δl| := abs_nonneg _
    nlinarith [abs_nonneg L]
  have b2 : |mm + L*(1+δl)| ≤ Mm + 106/100 := by
    calc |mm + L*(1+δl)| ≤ |mm| + |L*(1+δl)| := abs_add_le _ _
      _ ≤ _ := by linarith
  have b3 : |δ7 * (mm + L*(1+δl))| ≤ F.u * (Mm + 106/100) := by
    rw [abs_mul]
    exact mul_le_mul hδ7 b2 (abs_nonneg _) hu0
  have b4 : |δl * L| ≤ F.ul * (104/100) := by
    rw [abs_mul]
    exact mul_le_mul hδl hL1 (abs_nonneg _) hul0
  rw [e1]
  calc _ ≤ |δ7 * (mm + L*(1+δl)) + (L - Real.log (p+1))| + |δl * L| := abs_add_le _ _
    _ ≤ |δ7 * (mm + L*(1+δl))| + |L - Real.log (p+1)| + |δl * L| := by
        gcongr; exact abs_add_le _ _
    _ ≤ _ := by nlinarith
theorem lse_eq {a b : ℝ} :
    a ≤ b → Real.log (Real.exp a + Real.exp b) = b + Real.log (Real.exp (-(b - a)) + 1) := by
  intro _
  have : Real.exp a + Real.exp b = Real.exp b * (Real.exp (-(b - a)) + 1) := by
    have : Real.exp a = Real.exp b * Real.exp (-(b-a)) := by
      rw [← Real.exp_add]; congr 1; ring
    rw [this]; ring
  rw [this, Real.log_mul (Real.exp_pos _).ne' (by positivity), Real.log_exp]

theorem merge_core {F : Rounding} (hu : F.u ≤ 1/256) (hue : F.ue ≤ 1/64) (hul : F.ul ≤ 1/64)
    (hηu : F.ηe ≤ F.u) {a b Mm : ℝ} {st : State_a} (hM : |max a b| ≤ Mm)
    (h : fmerge_a F ⟨a⟩ ⟨b⟩ st) :
    |st.s - Real.log (Real.exp a + Real.exp b)| ≤
      F.u * Mm + 106/100 * F.u + 11/10 * (F.ue + 21/10 * F.u + 2 * F.ηe) + 104/100 * F.ul := by
  obtain ⟨mm, s, r1, r2, r3, r4, r5, r6, r7, hmm, hA1, hE1, hA3, hE3, hA5, hL, hA7, hs7, hst⟩ := h
  dsimp only at hmm hA1 hA3 hst
  subst hst hs7
  have hγ0 : 0 ≤ F.ue + 21/10 * F.u + 2 * F.ηe := by
    have := F.hu; have := F.hue; have := F.hηe; positivity
  have hγ1 : F.ue + 21/10 * F.u + 2 * F.ηe ≤ 1/30 := by linarith
  have hM' : |mm| ≤ Mm := by rw [hmm]; exact hM
  rcases le_total a b with hab | hba
  · have hm : mm = b := by rw [hmm]; exact max_eq_right hab
    have e1 : a - mm = -(b - a) := by rw [hm]; ring
    have e2 : b - mm = 0 := by rw [hm]; ring
    rw [e1] at hA1
    rw [e2] at hA3
    have h5 := merge_sum (sub_nonneg.2 hab) hu hue hηu hA1 hE1 hA3 hE3 hA5
    have hp1 : Real.exp (-(b - a)) ≤ 1 := by rw [Real.exp_le_one_iff]; linarith
    have := merge_tail hu hul (Real.exp_pos _) hp1 hγ0 hγ1 h5 hL hA7 hM'
    rw [lse_eq hab]
    rw [hm] at this
    exact this
  · have hm : mm = a := by rw [hmm]; exact max_eq_left hba
    have e1 : b - mm = -(a - b) := by rw [hm]; ring
    have e2 : a - mm = 0 := by rw [hm]; ring
    rw [e1] at hA3
    rw [e2] at hA1
    rw [add_comm] at hA5
    have h5 := merge_sum (sub_nonneg.2 hba) hu hue hηu hA3 hE3 hA1 hE1 hA5
    have hp1 : Real.exp (-(a - b)) ≤ 1 := by rw [Real.exp_le_one_iff]; linarith
    have := merge_tail hu hul (Real.exp_pos _) hp1 hγ0 hγ1 h5 hL hA7 hM'
    rw [add_comm (Real.exp a), lse_eq hba]
    rw [hm] at this
    exact this
theorem term_bounds {F : Rounding} {mx x y : ℝ}
    (h : ∃ q1 q2, F.add (x - mx) q1 ∧ F.exp q1 q2 ∧ y = q2) :
    max 0 ((1 - F.ue) * Real.exp ((x - mx) - F.u * |x - mx|) - F.ηe) ≤ y ∧
    y ≤ (1 + F.ue) * Real.exp ((x - mx) + F.u * |x - mx|) + F.ηe := by
  obtain ⟨q1, q2, ⟨δ, hδ, hq1⟩, ⟨δ2, e2, hδ2, he2, hq2, h0⟩, hy⟩ := h
  subst hy hq2 hq1
  have hd : |(x - mx) * δ| ≤ F.u * |x - mx| := by
    rw [abs_mul, mul_comm]; exact mul_le_mul_of_nonneg_right hδ (abs_nonneg _) |>.trans (le_of_eq rfl) 
  have hd' := abs_le.mp hd
  have a2 := abs_le.mp hδ2
  have a3 := abs_le.mp he2
  have hX0 := Real.exp_pos ((x - mx) * (1 + δ))
  have hX1 : Real.exp ((x - mx) - F.u * |x - mx|) ≤ Real.exp ((x - mx) * (1 + δ)) :=
    Real.exp_le_exp.2 (by nlinarith)
  have hX2 : Real.exp ((x - mx) * (1 + δ)) ≤ Real.exp ((x - mx) + F.u * |x - mx|) :=
    Real.exp_le_exp.2 (by nlinarith)
  have := F.hue1
  have := F.hue
  constructor
  · apply max_le h0
    nlinarith [mul_le_mul_of_nonneg_left a2.1 hX0.le, mul_le_mul_of_nonneg_left hX1 (by linarith : (0:ℝ) ≤ 1 - F.ue)]
  · nlinarith [mul_le_mul_of_nonneg_left a2.2 hX0.le, mul_le_mul_of_nonneg_left hX2 (by linarith : (0:ℝ) ≤ 1 + F.ue)]

theorem sum_affine {α : Type*} (f : α → ℝ) (k c : ℝ) (l : List α) :
    (l.map fun x => k * f x + c).sum = k * (l.map f).sum + l.length * c := by
  induction l with
  | nil => simp
  | cons a l ih => simp [ih]; ring

theorem log_one_sub_ge {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1/11) : -(11/10 * x) ≤ Real.log (1 - x) := by
  have := Real.one_sub_inv_le_log_of_pos (show 0 < 1 - x by linarith)
  have h2 : (1 - x)⁻¹ ≤ 1 + 11/10 * x := by
    rw [inv_eq_one_div, div_le_iff₀ (by linarith)]
    nlinarith
  linarith
theorem leaf_r2 {F : Rounding} {D : ℕ} {lo hi : ℝ} {A : List ℝ} {r2 : ℝ}
    (hA : A ≠ []) (hr : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi)
    (hκ : F.ue + A.length * F.ηe ≤ 1/32)
    (hfs : fsum F D (fun x y => ∃ q1 q2, F.add (x - lmax A) q1 ∧ F.exp q1 q2 ∧ y = q2) A r2) :
    (1 - F.u)^D * ((1 - F.ue - A.length * F.ηe) *
        (Real.exp (-(F.u*(hi-lo))) * (A.map fun x => Real.exp (x - lmax A)).sum)) ≤ r2 ∧
    r2 ≤ (1+F.u)^D * (Real.exp (F.u*(hi-lo)) * (1 + F.ue + A.length*F.ηe) *
        (A.map fun x => Real.exp (x - lmax A)).sum) := by
  have hmx : lmax A ∈ A := lmax_mem hA
  have hle : ∀ x ∈ A, x ≤ lmax A := fun x hx => le_lmax hx
  set mx := lmax A with hmxdef
  have hb := fsum_bounds F D (fun x y => ∃ q1 q2, F.add (x - mx) q1 ∧ F.exp q1 q2 ∧ y = q2)
    (fun x => max 0 ((1 - F.ue) * Real.exp ((x - mx) - F.u * |x - mx|) - F.ηe))
    (fun x => (1 + F.ue) * Real.exp ((x - mx) + F.u * |x - mx|) + F.ηe)
    (fun x => le_max_left _ _) (fun x y h => term_bounds h) A r2 hfs
  set T := (A.map fun x => Real.exp (x - mx)).sum with hT
  have hW : 0 ≤ hi - lo := by
    have := hr mx hmx; have := hr mx hmx; linarith [(hr mx hmx).1, (hr mx hmx).2]
  have hu0 := F.hu
  have hmxr := hr mx hmx
  -- T ≥ 1
  have hT1 : 1 ≤ T := by
    have := List.single_le_sum (l := A.map fun x => Real.exp (x - mx))
      (by intro y hy; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hy; exact (Real.exp_pos _).le)
      (Real.exp (mx - mx)) (List.mem_map.2 ⟨mx, hmx, rfl⟩)
    simpa using this
  -- upper
  have hup : (A.map fun x => (1 + F.ue) * Real.exp ((x - mx) + F.u * |x - mx|) + F.ηe).sum ≤
      ((1+F.ue) * Real.exp (F.u*(hi-lo))) * T + A.length * F.ηe := by
    have : (A.map fun x => (1 + F.ue) * Real.exp ((x - mx) + F.u * |x - mx|) + F.ηe).sum ≤
        (A.map fun x => ((1+F.ue) * Real.exp (F.u*(hi-lo))) * Real.exp (x - mx) + F.ηe).sum := by
      apply List.sum_le_sum
      intro x hx
      have h1 : |x - mx| = mx - x := by rw [abs_of_nonpos (by linarith [hle x hx])]; ring
      have h2 : Real.exp ((x - mx) + F.u * |x - mx|) ≤ Real.exp (F.u*(hi-lo)) * Real.exp (x - mx) := by
        rw [← Real.exp_add]; apply Real.exp_le_exp.2
        rw [h1]; nlinarith [(hr x hx).1, hmxr.2]
      have := F.hue
      nlinarith [mul_le_mul_of_nonneg_left h2 (by linarith : (0:ℝ) ≤ 1 + F.ue)]
    rw [sum_affine (fun x => Real.exp (x - mx))] at this
    exact this
  -- lower
  have hv : ∀ x ∈ A, 0 ≤ Real.exp ((x - mx) - F.u * |x - mx|) := fun x _ => (Real.exp_pos _).le
  have hlow : (1 - F.ue) * (A.map fun x => Real.exp ((x - mx) - F.u * |x - mx|)).sum
        + A.length * (-F.ηe) ≤ (A.map fun x => max 0 ((1 - F.ue) * Real.exp ((x - mx) - F.u * |x - mx|) - F.ηe)).sum := by
    rw [← sum_affine (fun x => Real.exp ((x - mx) - F.u * |x - mx|))]
    apply List.sum_le_sum
    intro x hx
    simp only [← sub_eq_add_neg]
    exact le_max_right _ _
  have hV1 : 1 ≤ (A.map fun x => Real.exp ((x - mx) - F.u * |x - mx|)).sum := by
    have := List.single_le_sum (l := A.map fun x => Real.exp ((x - mx) - F.u * |x - mx|))
      (by intro y hy; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hy; exact (Real.exp_pos _).le)
      (Real.exp ((mx - mx) - F.u * |mx - mx|)) (List.mem_map.2 ⟨mx, hmx, rfl⟩)
    simpa using this
  have hV2 : Real.exp (-(F.u*(hi-lo))) * T ≤ (A.map fun x => Real.exp ((x - mx) - F.u * |x - mx|)).sum := by
    rw [hT, ← List.sum_map_mul_left]
    apply List.sum_le_sum
    intro x hx
    have h1 : |x - mx| = mx - x := by rw [abs_of_nonpos (by linarith [hle x hx])]; ring
    rw [← Real.exp_add]; apply Real.exp_le_exp.2
    rw [h1]; nlinarith [(hr x hx).1, hmxr.2]
  have hη := F.hηe
  have hue := F.hue
  have hmη : 0 ≤ (A.length : ℝ) * F.ηe := by positivity
  set V := (A.map fun x => Real.exp ((x - mx) - F.u * |x - mx|)).sum
  have hEu : 1 ≤ Real.exp (F.u*(hi-lo)) := Real.one_le_exp (by positivity)
  have hEl0 := Real.exp_pos (-(F.u*(hi-lo)))
  have hpos : 0 ≤ 1 - F.ue - A.length * F.ηe := by linarith
  have hpu : 0 ≤ (1 - F.u)^D := pow_nonneg (by linarith [F.hu1]) _
  have hpu' : 0 ≤ (1 + F.u)^D := pow_nonneg (by linarith) _
  constructor
  · have h1 : (1 - F.ue - A.length * F.ηe) * V ≤ (A.map fun x => max 0 ((1 - F.ue) * Real.exp ((x - mx) - F.u * |x - mx|) - F.ηe)).sum := by
      nlinarith [mul_le_mul_of_nonneg_left hV1 hmη]
    have h2 : (1 - F.ue - A.length * F.ηe) * (Real.exp (-(F.u*(hi-lo))) * T) ≤ (1 - F.ue - A.length * F.ηe) * V :=
      mul_le_mul_of_nonneg_left hV2 hpos
    have := mul_le_mul_of_nonneg_left (h2.trans h1) hpu
    exact this.trans hb.1
  · have hET : 1 ≤ Real.exp (F.u*(hi-lo)) * T := by nlinarith
    have h1 : (A.map fun x => (1 + F.ue) * Real.exp ((x - mx) + F.u * |x - mx|) + F.ηe).sum ≤
        Real.exp (F.u*(hi-lo)) * (1 + F.ue + A.length*F.ηe) * T := by
      nlinarith [mul_le_mul_of_nonneg_left hET hmη]
    exact hb.2.trans (mul_le_mul_of_nonneg_left (hup.trans (by nlinarith [mul_le_mul_of_nonneg_left hET hmη])) hpu')
theorem T_bounds {A : List ℝ} (hA : A ≠ []) :
    1 ≤ (A.map fun x => Real.exp (x - lmax A)).sum ∧
    (A.map fun x => Real.exp (x - lmax A)).sum ≤ A.length := by
  have hmx : lmax A ∈ A := lmax_mem hA
  constructor
  · have := List.single_le_sum (l := A.map fun x => Real.exp (x - lmax A))
      (by intro y hy; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hy; exact (Real.exp_pos _).le)
      (Real.exp (lmax A - lmax A)) (List.mem_map.2 ⟨lmax A, hmx, rfl⟩)
    simpa using this
  · have := List.sum_le_length_nsmul (A.map fun x => Real.exp (x - lmax A)) 1
      (by intro y hy; obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hy
          rw [Real.exp_le_one_iff]; linarith [le_lmax hx])
    simpa using this

theorem sum_exp_eq (A : List ℝ) :
    (A.map Real.exp).sum = Real.exp (lmax A) * (A.map fun x => Real.exp (x - lmax A)).sum := by
  rw [← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro x _
  rw [← Real.exp_add]; congr 1; ring

theorem leaf_log {F : Rounding} {D : ℕ} {lo hi : ℝ} {A : List ℝ} {r2 : ℝ}
    (hA : A ≠ []) (hr : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi) (hu : F.u ≤ 1/256)
    (hκ : F.ue + A.length * F.ηe ≤ 1/32)
    (hfs : fsum F D (fun x y => ∃ q1 q2, F.add (x - lmax A) q1 ∧ F.exp q1 q2 ∧ y = q2) A r2) :
    0 < r2 ∧ |Real.log r2 - Real.log (A.map fun x => Real.exp (x - lmax A)).sum| ≤
      11/10 * (D * F.u + (F.ue + A.length * F.ηe)) + F.u * (hi - lo) := by
  have hb := leaf_r2 hA hr hκ hfs
  have hT := T_bounds hA
  set T := (A.map fun x => Real.exp (x - lmax A)).sum
  have hmxr := hr _ (lmax_mem hA)
  have hW : 0 ≤ hi - lo := by linarith [hmxr.1, hmxr.2]
  have hu0 := F.hu
  have hη := F.hηe
  have hm0 : (0:ℝ) ≤ A.length * F.ηe := by positivity
  have hue0 := F.hue
  set κ := F.ue + A.length * F.ηe with hκdef
  have hκ0 : 0 ≤ κ := by positivity
  have hEl := Real.exp_pos (-(F.u*(hi-lo)))
  have hT0 : 0 < T := by linarith [hT.1]
  have hlow : 0 < (1 - F.u)^D * ((1 - F.ue - A.length * F.ηe) * (Real.exp (-(F.u*(hi-lo))) * T)) := by
    have : 0 < 1 - F.u := by linarith
    have : 0 < 1 - F.ue - A.length * F.ηe := by linarith
    positivity
  have hr2 : 0 < r2 := lt_of_lt_of_le hlow hb.1
  refine ⟨hr2, ?_⟩
  have e1 : 1 - F.ue - A.length * F.ηe = 1 - κ := by rw [hκdef]; ring
  have e2 : 1 + F.ue + A.length * F.ηe = 1 + κ := by rw [hκdef]; ring
  rw [e1] at hb hlow
  rw [e2] at hb
  have h1k : 0 < 1 - κ := by linarith
  have hD0 : (0:ℝ) ≤ D := by positivity
  rw [abs_le]; constructor
  · -- lower
    have := Real.log_le_log hlow hb.1
    have h1u : 0 < 1 - F.u := by linarith
    rw [Real.log_mul (pow_pos h1u _).ne' (mul_pos h1k (mul_pos hEl hT0)).ne',
      Real.log_mul h1k.ne' (mul_pos hEl hT0).ne',
      Real.log_mul hEl.ne' hT0.ne', Real.log_exp, Real.log_pow] at this
    have h2 := log_one_sub_ge hu0 (by linarith : F.u ≤ 1/11)
    have h3 := log_one_sub_ge hκ0 (by linarith : κ ≤ 1/11)
    nlinarith [mul_le_mul_of_nonneg_left h2 hD0]
  · have hup : 0 < (1+F.u)^D * (Real.exp (F.u*(hi-lo)) * (1 + κ) * T) := by positivity
    have := Real.log_le_log hr2 hb.2
    rw [Real.log_mul (by positivity) (by positivity), Real.log_mul (by positivity) hT0.ne',
      Real.log_mul (by positivity) (by positivity), Real.log_exp, Real.log_pow] at this
    have h2 := Real.log_le_sub_one_of_pos (show 0 < 1 + F.u by linarith)
    have h3 := Real.log_le_sub_one_of_pos (show 0 < 1 + κ by linarith)
    nlinarith [mul_le_mul_of_nonneg_left h2 hD0]
set_option maxHeartbeats 1000000 in
theorem leaf_core {F : Rounding} {D : ℕ} {lo hi n : ℝ} {A : List ℝ} {st : State_a}
    (hu : F.u ≤ 1/256) (hul : F.ul ≤ 1/64) (hue : F.ue ≤ 1/64)
    (hηn : n * (F.η + F.ηe) ≤ F.u) (hA : A ≠ []) (hlen : (A.length:ℝ) ≤ n)
    (hr : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi) (h : fblock_a F D A st) :
    |st.s - Real.log ((A.map Real.exp).sum)| ≤
      3/2 * (F.u*(hi-lo) + F.ue + D*F.u + F.ul*n + F.u * max |lo| |hi| + F.u*n + n*F.ηe) := by
  obtain ⟨v_mx, v_s, r1, r2, r3, r4, hr1, hv, hfs, hL, hA4, hvs, hst⟩ := h
  subst hv hr1 hvs hst
  have hA1 : (1:ℝ) ≤ A.length := by
    have : 0 < A.length := List.length_pos_iff.2 hA
    exact_mod_cast this
  have hn1 : 1 ≤ n := hA1.trans hlen
  have hu0 := F.hu
  have hη0 := F.hη
  have hηe0 := F.hηe
  have hul0 := F.hul
  have hue0 := F.hue
  have hmη : (A.length : ℝ) * F.ηe ≤ n * F.ηe := mul_le_mul_of_nonneg_right hlen hηe0
  have hnη : n * F.ηe ≤ F.u := by nlinarith [mul_nonneg (by linarith : (0:ℝ) ≤ n) hη0]
  have hκ : F.ue + A.length * F.ηe ≤ 1/32 := by linarith
  obtain ⟨hr2, hlog⟩ := leaf_log hA hr hu hκ hfs
  have hT := T_bounds hA
  set T := (A.map fun x => Real.exp (x - lmax A)).sum with hTdef
  have hT0 : 0 < T := by linarith [hT.1]
  have hmxr := hr _ (lmax_mem hA)
  have hmx : |lmax A| ≤ max |lo| |hi| := abs_le_max_abs_abs hmxr.1 hmxr.2
  have hW : 0 ≤ hi - lo := by linarith [hmxr.1, hmxr.2]
  have hLt0 : 0 ≤ Real.log T := Real.log_nonneg hT.1
  have hLt1 : Real.log T ≤ n - 1 := by
    have := Real.log_le_sub_one_of_pos hT0; linarith [hT.2]
  have hG : Real.log ((A.map Real.exp).sum) = lmax A + Real.log T := by
    rw [sum_exp_eq, Real.log_mul (Real.exp_pos _).ne' hT0.ne', Real.log_exp]
  rw [hG]
  obtain ⟨δl, hδl, rfl⟩ := hL
  obtain ⟨δ, hδ, rfl⟩ := hA4
  dsimp only
  set lam := 11/10 * ((D:ℝ) * F.u + (F.ue + A.length * F.ηe)) + F.u * (hi - lo) with hlam
  set L2 := Real.log r2 with hL2
  set Lt := Real.log T with hLt
  set mx := lmax A with hmxdef
  set M := max |lo| |hi| with hM
  have hDu0 : (0:ℝ) ≤ (D:ℝ) * F.u := by positivity
  have hlam0 : 0 ≤ lam := by
    have : 0 ≤ F.u * (hi - lo) := by positivity
    have : 0 ≤ F.ue + A.length * F.ηe := by positivity
    rw [hlam]; positivity
  have hR : |L2 - Lt| ≤ lam := hlog
  have hL2b : |L2| ≤ Lt + lam := by
    have := abs_le.mp hR
    rw [abs_le]; constructor <;> linarith
  -- r3 - Lt
  have e1 : (mx + L2 * (1+δl)) * (1+δ) - (mx + Lt) =
      δ * (mx + L2 * (1+δl)) + ((L2 - Lt) + δl * L2) := by ring
  have b1 : |δl * L2| ≤ F.ul * (Lt + lam) := by
    rw [abs_mul]; exact mul_le_mul hδl hL2b (abs_nonneg _) hul0
  have hmu : |(L2 - Lt) + δl * L2| ≤ lam + F.ul * (Lt + lam) :=
    (abs_add_le _ _).trans (add_le_add hR b1)
  have b2 : |mx + L2 * (1+δl)| ≤ M + Lt + (lam + F.ul * (Lt + lam)) := by
    have : mx + L2 * (1+δl) = mx + Lt + ((L2 - Lt) + δl * L2) := by ring
    rw [this]
    calc _ ≤ |mx + Lt| + |(L2 - Lt) + δl * L2| := abs_add_le _ _
      _ ≤ (|mx| + |Lt|) + _ := add_le_add (abs_add_le _ _) hmu
      _ ≤ _ := by rw [abs_of_nonneg hLt0]; linarith
  have b3 : |δ * (mx + L2 * (1+δl))| ≤ F.u * (M + Lt + (lam + F.ul * (Lt + lam))) := by
    rw [abs_mul]; exact mul_le_mul hδ b2 (abs_nonneg _) hu0
  rw [e1]
  have tot : |δ * (mx + L2 * (1+δl)) + ((L2 - Lt) + δl * L2)| ≤
      F.u * (M + Lt + (lam + F.ul * (Lt + lam))) + (lam + F.ul * (Lt + lam)) :=
    (abs_add_le _ _).trans (add_le_add b3 hmu)
  refine tot.trans ?_
  have hM0 : 0 ≤ M := le_trans (abs_nonneg _) hmx
  -- arithmetic
  have p1 : F.u * Lt ≤ F.u * (n - 1) := mul_le_mul_of_nonneg_left hLt1 hu0
  have p2 : F.ul * Lt ≤ F.ul * (n - 1) := mul_le_mul_of_nonneg_left hLt1 hul0
  have p3 : F.u * (F.ul * (Lt + lam)) ≤ F.u * (F.ul * (n - 1 + lam)) := by
    apply mul_le_mul_of_nonneg_left _ hu0
    apply mul_le_mul_of_nonneg_left _ hul0; linarith
  have p4 : F.u * lam ≤ (1/256) * lam := mul_le_mul_of_nonneg_right hu hlam0
  have p5 : F.ul * lam ≤ (1/64) * lam := mul_le_mul_of_nonneg_right hul hlam0
  have p6 : F.u * (F.ul * lam) ≤ (1/256) * ((1/64) * lam) :=
    mul_le_mul hu p5 (by positivity) (by norm_num)
  have hn1' : 0 ≤ n - 1 := by linarith
  have p7 : F.u * (F.ul * (n - 1)) ≤ (1/256) * (F.ul * (n - 1)) :=
    mul_le_mul_of_nonneg_right hu (by positivity)
  have p8 : lam ≤ 11/10 * ((D:ℝ) * F.u + (F.ue + n * F.ηe)) + F.u * (hi - lo) := by
    rw [hlam]; linarith
  have p9 : F.ul * (n - 1) ≤ F.ul * n := mul_le_mul_of_nonneg_left (by linarith) hul0
  have q : F.u * (M + Lt + (lam + F.ul * (Lt + lam))) + (lam + F.ul * (Lt + lam)) =
      F.u * M + F.u * Lt + F.u * lam + F.u * (F.ul * (Lt + lam)) + lam + F.ul * Lt + F.ul * lam := by ring
  rw [q]
  have q2 : F.u * (F.ul * (n - 1 + lam)) = F.u * (F.ul * (n - 1)) + F.u * (F.ul * lam) := by ring
  have n1 : 0 ≤ F.u * (hi - lo) := mul_nonneg hu0 hW
  have n2 : 0 ≤ F.u * M := mul_nonneg hu0 hM0
  have n3 : 0 ≤ F.ul * n := mul_nonneg hul0 (by linarith)
  have n4 : 0 ≤ F.u * n := mul_nonneg hu0 (by linarith)
  have n5 : 0 ≤ n * F.ηe := mul_nonneg (by linarith) hηe0
  linarith
theorem lse_le {a b A B ε : ℝ} (ha : a ≤ A + ε) (hb : b ≤ B + ε) :
    Real.log (Real.exp a + Real.exp b) ≤ Real.log (Real.exp A + Real.exp B) + ε := by
  have h1 : Real.exp a + Real.exp b ≤ Real.exp ε * (Real.exp A + Real.exp B) := by
    have h2 : Real.exp a ≤ Real.exp ε * Real.exp A := by
      rw [← Real.exp_add]; apply Real.exp_le_exp.2; linarith
    have h3 : Real.exp b ≤ Real.exp ε * Real.exp B := by
      rw [← Real.exp_add]; apply Real.exp_le_exp.2; linarith
    linarith
  have h4 := Real.log_le_log (by positivity) h1
  rw [Real.log_mul (Real.exp_pos _).ne' (by positivity), Real.log_exp] at h4
  linarith

theorem lse_lip {a b A B εa εb : ℝ} (ha : |a - A| ≤ εa) (hb : |b - B| ≤ εb) :
    |Real.log (Real.exp a + Real.exp b) - Real.log (Real.exp A + Real.exp B)| ≤ max εa εb := by
  have ha' := abs_le.mp ha
  have hb' := abs_le.mp hb
  have m1 := le_max_left εa εb
  have m2 := le_max_right εa εb
  rw [abs_le]; constructor
  · have := lse_le (a := A) (b := B) (A := a) (B := b) (ε := max εa εb) (by linarith) (by linarith)
    linarith
  · have := lse_le (a := a) (b := b) (A := A) (B := B) (ε := max εa εb) (by linarith) (by linarith)
    linarith

set_option maxHeartbeats 1000000 in
theorem node_step {F : Rounding} {lo hi n E a b A B εa εb : ℝ} {st : State_a}
    (hu : F.u ≤ 1/256) (hue : F.ue ≤ 1/64) (hul : F.ul ≤ 1/64) (hηu : F.ηe ≤ F.u)
    (hn : 1 ≤ n) (hE : E ≤ (max |lo| |hi| + n) / 2) (hεa : εa ≤ E) (hεb : εb ≤ E)
    (hA : lo ≤ A ∧ A ≤ hi + (n - 1)) (hB : lo ≤ B ∧ B ≤ hi + (n - 1))
    (ha : |a - A| ≤ εa) (hb : |b - B| ≤ εb) (h : fmerge_a F ⟨a⟩ ⟨b⟩ st) :
    |st.s - Real.log (Real.exp A + Real.exp B)| ≤ max εa εb +
      3/2 * (3 * F.u + F.ue + F.ul + F.ηe + F.u * max |lo| |hi| + F.u * n) := by
  have ha' := abs_le.mp ha
  have hb' := abs_le.mp hb
  have hlo := neg_abs_le lo
  have hhi := le_abs_self hi
  have hM1 := le_max_left |lo| |hi|
  have hM2 := le_max_right |lo| |hi|
  set M := max |lo| |hi|
  have hMm : |max a b| ≤ M + (n - 1) + E := by
    rw [abs_le]; constructor
    · have : -(M + (n-1) + E) ≤ a := by linarith [abs_nonneg hi, abs_nonneg lo]
      linarith [le_max_left a b]
    · apply max_le <;> linarith [abs_nonneg hi, abs_nonneg lo]
  have h1 := merge_core hu hue hul hηu hMm h
  have h2 := lse_lip ha hb
  have h3 : |st.s - Real.log (Real.exp A + Real.exp B)| ≤
      |st.s - Real.log (Real.exp a + Real.exp b)| +
      |Real.log (Real.exp a + Real.exp b) - Real.log (Real.exp A + Real.exp B)| := by
    have := abs_add_le (st.s - Real.log (Real.exp a + Real.exp b))
      (Real.log (Real.exp a + Real.exp b) - Real.log (Real.exp A + Real.exp B))
    simpa using this
  have hu0 := F.hu
  have hηe0 := F.hηe
  have hue0 := F.hue
  have hul0 := F.hul
  have p1 : F.u * E ≤ F.u * ((M + n) / 2) := mul_le_mul_of_nonneg_left hE hu0
  have : F.u * (M + (n - 1) + E) = F.u * M + F.u * n - F.u + F.u * E := by ring
  have : F.u * ((M + n)/2) = (F.u * M + F.u * n) / 2 := by ring
  have : 0 ≤ F.u * M := mul_nonneg hu0 (le_trans (abs_nonneg _) hM1)
  have : 0 ≤ F.u * n := mul_nonneg hu0 (by linarith)
  linarith
theorem flat_ne : ∀ t : Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → t.flat ≠ []
  | .leaf b, h => by simpa [Tree.flat, Tree.leaves] using h b (by simp [Tree.leaves])
  | .node l r, h => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => h b (by simp [Tree.leaves, hb])
    have := flat_ne l hl
    simp [Tree.flat, this]

theorem depth_le_length : ∀ t : Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → t.depth + 1 ≤ t.flat.length
  | .leaf b, h => by
    have := h b (by simp [Tree.leaves])
    simp [Tree.depth, Tree.flat]
    exact List.length_pos_iff.2 this
  | .node l r, h => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => h b (by simp [Tree.leaves, hb])
    have hr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => h b (by simp [Tree.leaves, hb])
    have h1 := depth_le_length l hl
    have h2 := depth_le_length r hr
    simp only [Tree.depth, Tree.flat, List.length_append]
    omega

theorem sum_exp_pos {A : List ℝ} (hA : A ≠ []) : 0 < (A.map Real.exp).sum := by
  rw [sum_exp_eq]
  have := (T_bounds hA).1
  have := Real.exp_pos (lmax A)
  positivity

theorem G_bounds {lo hi : ℝ} {A : List ℝ} (hA : A ≠ []) (hr : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi) :
    lo ≤ Real.log ((A.map Real.exp).sum) ∧ Real.log ((A.map Real.exp).sum) ≤ hi + (A.length - 1) := by
  have hT := T_bounds hA
  have hT0 : 0 < (A.map fun x => Real.exp (x - lmax A)).sum := by linarith [hT.1]
  have h1 : Real.log ((A.map Real.exp).sum) = lmax A + Real.log (A.map fun x => Real.exp (x - lmax A)).sum := by
    rw [sum_exp_eq, Real.log_mul (Real.exp_pos _).ne' hT0.ne', Real.log_exp]
  have hm := hr _ (lmax_mem hA)
  have h2 : 0 ≤ Real.log (A.map fun x => Real.exp (x - lmax A)).sum := Real.log_nonneg hT.1
  have h3 := Real.log_le_sub_one_of_pos hT0
  rw [h1]; constructor <;> linarith [hT.2]

set_option maxHeartbeats 1000000 in
theorem tree_bound {F : Rounding} {D : ℕ} {lo hi n e0 c E : ℝ}
    (hu : F.u ≤ 1/256) (hue : F.ue ≤ 1/64) (hul : F.ul ≤ 1/64) (hηu : F.ηe ≤ F.u) (hn : 1 ≤ n)
    (hleaf : ∀ (A : List ℝ) (st : State_a), A ≠ [] → (A.length:ℝ) ≤ n → (∀ x ∈ A, lo ≤ x ∧ x ≤ hi) →
        fblock_a F D A st → |st.s - Real.log ((A.map Real.exp).sum)| ≤ e0)
    (hE : E ≤ (max |lo| |hi| + n)/2) (hc : 0 ≤ c)
    (hcdef : c = 3/2 * (3 * F.u + F.ue + F.ul + F.ηe + F.u * max |lo| |hi| + F.u * n)) (dd : ℕ)
    (hdd : e0 + c * dd ≤ E) :
    ∀ t : Tree ℝ, t.depth ≤ dd → (∀ x ∈ t.flat, lo ≤ x ∧ x ≤ hi) → (∀ b ∈ t.leaves, b ≠ []) →
      (t.flat.length : ℝ) ≤ n → ∀ st : State_a,
      Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |st.s - Real.log ((t.flat.map Real.exp).sum)| ≤ e0 + c * t.depth := by
  intro t
  induction t with
  | leaf b =>
    intro hd hr hl hlen st hev
    cases hev with
    | leaf hb =>
      have := hleaf b st (by simpa [Tree.leaves] using hl b (by simp [Tree.leaves])) (by simpa [Tree.flat] using hlen)
        (by simpa [Tree.flat] using hr) hb
      simpa [Tree.depth, Tree.flat] using this
  | node l r ihl ihr =>
    intro hd hr hl hlen st hev
    cases hev with
    | node hel her hm =>
      rename_i La Ra
      have hll : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => hl b (by simp [Tree.leaves, hb])
      have hrl : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => hl b (by simp [Tree.leaves, hb])
      have hlr : ∀ x ∈ l.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hr x (by simp [Tree.flat, hx])
      have hrr : ∀ x ∈ r.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hr x (by simp [Tree.flat, hx])
      have hlen' : (l.flat.length : ℝ) + r.flat.length ≤ n := by
        simpa [Tree.flat] using hlen
      have hl0 : (0:ℝ) ≤ l.flat.length := by positivity
      have hr0 : (0:ℝ) ≤ r.flat.length := by positivity
      have hdl : l.depth ≤ dd := by simp only [Tree.depth] at hd; omega
      have hdr : r.depth ≤ dd := by simp only [Tree.depth] at hd; omega
      have ih1 := ihl hdl hlr hll (by linarith) _ hel
      have ih2 := ihr hdr hrr hrl (by linarith) _ her
      have hfl := flat_ne l hll
      have hfr := flat_ne r hrl
      have hGl := G_bounds hfl hlr
      have hGr := G_bounds hfr hrr
      have hdl' : (l.depth : ℝ) ≤ dd := by exact_mod_cast hdl
      have hdr' : (r.depth : ℝ) ≤ dd := by exact_mod_cast hdr
      have e1 : e0 + c * l.depth ≤ E := by nlinarith
      have e2 : e0 + c * r.depth ≤ E := by nlinarith
      have hstep := node_step (E := E) (hu := hu) (hue := hue) (hul := hul) (hηu := hηu) hn hE e1 e2
        ⟨hGl.1, by linarith⟩ ⟨hGr.1, by linarith⟩ ih1 ih2 hm
      have hpl := sum_exp_pos hfl
      have hpr := sum_exp_pos hfr
      have hsum : Real.exp (Real.log ((l.flat.map Real.exp).sum)) + Real.exp (Real.log ((r.flat.map Real.exp).sum))
          = ((l.flat ++ r.flat).map Real.exp).sum := by
        rw [Real.exp_log hpl, Real.exp_log hpr, List.map_append, List.sum_append]
      rw [hsum] at hstep
      have hmax : max (e0 + c * l.depth) (e0 + c * r.depth) + c = e0 + c * ((max l.depth r.depth + 1 : ℕ) : ℝ) := by
        rcases le_total l.depth r.depth with h | h
        · have : (l.depth : ℝ) ≤ r.depth := by exact_mod_cast h
          rw [max_eq_right (by nlinarith), max_eq_right h]; push_cast; ring
        · have : (r.depth : ℝ) ≤ l.depth := by exact_mod_cast h
          rw [max_eq_left (by nlinarith), max_eq_left h]; push_cast; ring
      simp only [Tree.depth, Tree.flat]
      rw [← hmax]
      rw [← hcdef] at hstep
      exact hstep
set_option maxHeartbeats 1000000 in
theorem E_bound {u ue ul ηe d Dr n lo hi : ℝ} (hu0 : 0 ≤ u) (hue0 : 0 ≤ ue) (hul0 : 0 ≤ ul)
    (hηe0 : 0 ≤ ηe) (hu : u ≤ 1/256) (hul : ul ≤ 1/64) (hd0 : 0 ≤ d) (hDr : 0 ≤ Dr)
    (hdn : d + 1 ≤ n) (hnη : n * ηe ≤ u) (hs : (d + Dr + 1) * (u + ue + ul) ≤ 1/16)
    (hlh : lo ≤ hi) :
    3/2 * (u * (hi - lo) + ue + Dr * u + ul * n + u * max |lo| |hi| + u * n + n * ηe) +
      (3/2 * (3 * u + ue + ul + ηe + u * max |lo| |hi| + u * n)) * d ≤ (max |lo| |hi| + n) / 2 := by
  have hM1 := le_max_left |lo| |hi|
  have hM2 := le_max_right |lo| |hi|
  have hlo := neg_abs_le lo
  have hhi := le_abs_self hi
  set M := max |lo| |hi| with hM
  have hM0 : 0 ≤ M := le_trans (abs_nonneg _) hM1
  have hn0 : 0 ≤ n := by linarith
  have hW : hi - lo ≤ 2 * M := by linarith
  have a1 : u * (hi - lo) ≤ u * (2 * M) := mul_le_mul_of_nonneg_left hW hu0
  have a2 : u * M ≤ (1/256) * M := mul_le_mul_of_nonneg_right hu hM0
  have a3 : u * n ≤ (1/256) * n := mul_le_mul_of_nonneg_right hu hn0
  have a4 : ul * n ≤ (1/64) * n := mul_le_mul_of_nonneg_right hul hn0
  have b1 : 0 ≤ d * ue := mul_nonneg hd0 hue0
  have b2 : 0 ≤ d * ul := mul_nonneg hd0 hul0
  have b3 : 0 ≤ Dr * ue := mul_nonneg hDr hue0
  have b4 : 0 ≤ Dr * ul := mul_nonneg hDr hul0
  have b5 : 0 ≤ d * u := mul_nonneg hd0 hu0
  have b6 : 0 ≤ Dr * u := mul_nonneg hDr hu0
  have b7 : 0 ≤ ue := hue0
  have b8 : 0 ≤ ul := hul0
  have hdu : d * u ≤ 1/16 := by nlinarith
  have c2 : d * u * M ≤ (1/16) * M := mul_le_mul_of_nonneg_right hdu hM0
  have c3 : d * u * n ≤ (1/16) * n := mul_le_mul_of_nonneg_right hdu hn0
  have c4 : d * ηe ≤ n * ηe := mul_le_mul_of_nonneg_right (by linarith) hηe0
  have c5 : 0 ≤ u * M := mul_nonneg hu0 hM0
  have c6 : 0 ≤ u * n := mul_nonneg hu0 hn0
  have c7 : 0 ≤ ul * n := mul_nonneg hul0 hn0
  have c8 : 0 ≤ n * ηe := mul_nonneg hn0 hηe0
  have c9 : 0 ≤ d * ηe := mul_nonneg hd0 hηe0
  have hs' : d * u + d * ue + d * ul + Dr * u + Dr * ue + Dr * ul + u + ue + ul ≤ 1/16 := by nlinarith
  nlinarith

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ), ∀ (lo hi : ℝ) (X : List ℝ),
    X ≠ [] → (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ ((3 : ℝ) * ((F.u * (hi - lo)) + F.ue + ((D : ℝ) * F.u) + (F.ul * (X.length : ℝ)) + (F.u * (max |lo| |hi|)) + (F.u * (X.length : ℝ)) + ((X.length : ℝ) * F.ηe) + ((t.depth : ℝ) * (((3 : ℝ) * F.u) + F.ue + F.ul + F.ηe + (F.u * (max |lo| |hi|)) + (F.u * (X.length : ℝ))))) * ((1 : ℝ) / 2)) := by
  intro F D lo hi X hX hr hsm hnη t hflat hleaves hdep st hev
  obtain ⟨hu, hue, hul, hη, hηe⟩ := hsm
  have hlen : (1:ℝ) ≤ (X.length : ℝ) := by exact_mod_cast List.length_pos_iff.2 hX
  have hu0 := F.hu
  have hη0 := F.hη
  have hηe0 := F.hηe
  have hue0 := F.hue
  have hul0 := F.hul
  have hnηe : (X.length : ℝ) * F.ηe ≤ F.u := by
    nlinarith [mul_nonneg (by linarith : (0:ℝ) ≤ (X.length : ℝ)) hη0]
  have hηu : F.ηe ≤ F.u := by nlinarith [mul_le_mul_of_nonneg_right hlen hηe0]
  have hdl : (t.depth : ℝ) + 1 ≤ (X.length : ℝ) := by
    have := depth_le_length t hleaves
    rw [hflat] at this
    exact_mod_cast this
  have hlh : lo ≤ hi := by
    obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil X hX
    have := hr x hx; linarith [this.1, this.2]
  have hEb := E_bound (u := F.u) (ue := F.ue) (ul := F.ul) (ηe := F.ηe) (d := (t.depth : ℝ))
    (Dr := (D:ℝ)) (n := (X.length : ℝ)) (lo := lo) (hi := hi) hu0 hue0 hul0 hηe0 hu hul
    (by positivity) (by positivity) hdl hnηe hdep hlh
  have hc : 0 ≤ 3/2 * (3 * F.u + F.ue + F.ul + F.ηe + F.u * max |lo| |hi| + F.u * (X.length : ℝ)) := by
    positivity
  have key := tree_bound (F := F) (D := D) (lo := lo) (hi := hi) (n := (X.length : ℝ))
    (e0 := 3/2 * (F.u*(hi-lo) + F.ue + D*F.u + F.ul*(X.length : ℝ) + F.u * max |lo| |hi| + F.u*(X.length : ℝ) + (X.length : ℝ)*F.ηe))
    (c := 3/2 * (3 * F.u + F.ue + F.ul + F.ηe + F.u * max |lo| |hi| + F.u * (X.length : ℝ)))
    (E := 3/2 * (F.u*(hi-lo) + F.ue + D*F.u + F.ul*(X.length : ℝ) + F.u * max |lo| |hi| + F.u*(X.length : ℝ) + (X.length : ℝ)*F.ηe) + (3/2 * (3 * F.u + F.ue + F.ul + F.ηe + F.u * max |lo| |hi| + F.u * (X.length : ℝ))) * (t.depth : ℝ))
    hu hue hul hηu hlen
    (fun A st hA hlA hrA h => leaf_core hu hul hue hnη hA hlA hrA h)
    hEb hc rfl t.depth le_rfl t le_rfl (by rw [hflat]; exact hr) hleaves (by rw [hflat]) st hev
  rw [hflat] at key
  have hz : Real.log (znorm st) = st.s := by
    have : znorm st = Real.exp st.s := by
      unfold znorm; rw [one_div, ← Real.exp_neg]; congr 1; ring
    rw [this, Real.log_exp]
  rw [hz]
  refine key.trans (le_of_eq ?_)
  ring

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.Cand4f04f56628
