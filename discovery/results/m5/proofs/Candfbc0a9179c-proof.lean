import FPStatements

namespace Discovery.FP.Candfbc0a9179c

open Discovery

theorem r_add_up (F : Rounding) {s v T : ℝ} (h : F.add s v) (hs : s ≤ T) (hT : 0 ≤ T) :
    v ≤ (1 + F.u) * T := by
  obtain ⟨δ, hδ, rfl⟩ := h
  have h1 := abs_le.mp hδ
  have hu := F.hu
  rcases le_total 0 s with h0 | h0
  · nlinarith [mul_nonneg h0 (sub_nonneg.2 h1.2), mul_nonneg hu (sub_nonneg.2 hs)]
  · have := F.hu1
    nlinarith [mul_nonneg (neg_nonneg.2 h0) (by linarith : 0 ≤ 1 + δ), mul_nonneg hu hT]

theorem r_add_lo (F : Rounding) {s v z e : ℝ} (h : F.add s v) (hs : z - e ≤ s) (hz : 0 ≤ z)
    (he : 0 ≤ e) : (1 - F.u) * z - (1 + F.u) * e ≤ v := by
  obtain ⟨δ, hδ, rfl⟩ := h
  have h1 := abs_le.mp hδ
  have hu := F.hu
  rcases le_total 0 s with h0 | h0
  · have := F.hu1
    nlinarith [mul_nonneg h0 (by linarith : 0 ≤ δ + F.u),
      mul_nonneg (by linarith : 0 ≤ 1 - F.u) (sub_nonneg.2 hs),
      mul_nonneg hu he, mul_nonneg hu hz]
  · have := F.hu1
    nlinarith [mul_nonneg (neg_nonneg.2 h0) (by linarith : 0 ≤ F.u - δ),
      mul_nonneg (by linarith : 0 ≤ 1 + F.u) (sub_nonneg.2 hs),
      mul_nonneg hu he, mul_nonneg hu hz]

theorem r_mul_up (F : Rounding) {x v X : ℝ} (h : F.mul x v) (hx : x ≤ X) (hX : 0 ≤ X) :
    v ≤ (1 + F.u) * X + F.η := by
  obtain ⟨δ, e, hδ, he, hv, hs⟩ := h
  have h1 := abs_le.mp hδ
  have h2 := abs_le.mp he
  have hu := F.hu
  rcases le_total 0 x with h0 | h0
  · nlinarith [mul_nonneg h0 (by linarith : 0 ≤ F.u - δ), mul_nonneg hu (sub_nonneg.2 hx)]
  · rcases lt_or_eq_of_le h0 with h0 | h0
    · have : v ≤ 0 := by
        by_contra hc
        push_neg at hc
        nlinarith
      nlinarith [F.hη, mul_nonneg hu hX]
    · subst h0
      nlinarith [F.hη, mul_nonneg hu hX]

theorem r_mul_lo (F : Rounding) {x v Z : ℝ} (h : F.mul x v) (hx : 0 ≤ x) (hZ : Z ≤ x) :
    (1 - F.u) * Z - F.η ≤ v := by
  obtain ⟨δ, e, hδ, he, hv, hs⟩ := h
  have h1 := abs_le.mp hδ
  have h2 := abs_le.mp he
  have hu := F.hu
  have := F.hu1
  nlinarith [mul_nonneg hx (by linarith : 0 ≤ δ + F.u), mul_nonneg (by linarith : 0 ≤ 1 - F.u) (sub_nonneg.2 hZ)]

-- END1

set_option maxHeartbeats 1000000 in
theorem half_step(F : Rounding) (hs : small F) {mc m d r1 r2 r3 n Sp Sm P B a A : ℝ}
    (hmc : mc ≤ m) (h1 : F.add (mc - m) r1) (h2 : F.exp r1 r2) (h3 : F.mul (d * r2) r3)
    (hd : 0 ≤ d) (hdU : d ≤ P * Sp + n * B) (hdL : a * Sm - n * B ≤ d) (hdA : A ≤ d)
    (hSp0 : 0 ≤ Sp) (hSp : Sp ≤ n) (hSm : 0 ≤ Sm) (hP : 0 ≤ P) (hB : 0 ≤ B) (ha : 0 ≤ a)
    (hA : 0 ≤ A) (hn : 0 ≤ n) :
    r3 ≤ (1 + F.u) * ((1 + F.ue) * P * Real.exp (-((1 - F.u) * (m - mc))) * Sp
        + n * ((1 + F.ue) * B + (P + B) * F.ηe)) + F.η ∧
    (1 - F.u) * ((1 - F.ue) * a * Real.exp (-((1 + F.u) * (m - mc))) * Sm
        - n * (B + (P + B) * F.ηe)) - F.η ≤ r3 ∧
    -F.η ≤ r3 ∧
    (mc = m → (1 - F.u) * (A * (1 - F.ue - F.ηe)) - F.η ≤ r3) := by
  obtain ⟨hu256, hue64, -, hη, hηe⟩ := hs
  have hu := F.hu
  have hue := F.hue
  have hηp := F.hη
  have hηep := F.hηe
  have hη' : F.η ≤ 1/2 := by
    have : (1:ℝ) / 2 ^ 100 ≤ 1/2 := by norm_num
    linarith
  have hηe' : F.ηe ≤ 1/2 := by
    have : (1:ℝ) / 2 ^ 100 ≤ 1/2 := by norm_num
    linarith
  obtain ⟨δ, hδ, hr1⟩ := h1
  have hδ' := abs_le.mp hδ
  have ht : 0 ≤ m - mc := by linarith
  have hr1u : r1 ≤ -((1 - F.u) * (m - mc)) := by
    rw [hr1]; nlinarith [mul_nonneg ht (by linarith : 0 ≤ δ + F.u)]
  have hr1l : -((1 + F.u) * (m - mc)) ≤ r1 := by
    rw [hr1]; nlinarith [mul_nonneg ht (by linarith : 0 ≤ F.u - δ)]
  obtain ⟨hE1, hE2, hE3⟩ := F.exp_bounds h2
  have he1 : Real.exp r1 ≤ Real.exp (-((1 - F.u) * (m - mc))) := Real.exp_le_exp.2 hr1u
  have he2 : Real.exp (-((1 + F.u) * (m - mc))) ≤ Real.exp r1 := Real.exp_le_exp.2 hr1l
  have he1_1 : Real.exp (-((1 - F.u) * (m - mc))) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    nlinarith [mul_nonneg ht (by linarith : 0 ≤ 1 - F.u)]
  have he1_0 := Real.exp_pos (-((1 - F.u) * (m - mc)))
  have he2_0 := Real.exp_pos (-((1 + F.u) * (m - mc)))
  have he0 := Real.exp_pos r1
  have hexp1 : Real.exp r1 ≤ 1 := he1.trans he1_1
  set e1 := Real.exp (-((1 - F.u) * (m - mc))) with he1def
  set e2 := Real.exp (-((1 + F.u) * (m - mc))) with he2def
  have hr2u : r2 ≤ e1 * (1 + F.ue) + F.ηe := by
    nlinarith [mul_le_mul_of_nonneg_right he1 (by linarith : 0 ≤ 1 + F.ue)]
  have hx0 : 0 ≤ d * r2 := mul_nonneg hd hE3
  refine ⟨?_, ?_, ?_, ?_⟩
  · have k1 : d * e1 ≤ P * Sp * e1 + n * B := by
      nlinarith [mul_le_mul_of_nonneg_right hdU he1_0.le, mul_nonneg hn hB,
        mul_nonneg (mul_nonneg hn hB) (sub_nonneg.2 he1_1)]
    have k2 : d ≤ (P + B) * n := by nlinarith [mul_le_mul_of_nonneg_left hSp hP]
    have k3 : d * r2 ≤ d * (e1 * (1 + F.ue) + F.ηe) := mul_le_mul_of_nonneg_left hr2u hd
    have hX : d * r2 ≤ (1 + F.ue) * P * e1 * Sp + n * ((1 + F.ue) * B + (P + B) * F.ηe) := by
      linarith [mul_le_mul_of_nonneg_right k1 (by linarith : 0 ≤ 1 + F.ue),
        mul_le_mul_of_nonneg_right k2 hηep]
    refine r_mul_up F h3 hX ?_
    exact add_nonneg (mul_nonneg (mul_nonneg (mul_nonneg (by linarith) hP) he1_0.le) hSp0)
      (mul_nonneg hn (add_nonneg (mul_nonneg (by linarith) hB)
        (mul_nonneg (add_nonneg hP hB) hηep)))
  · have hv0 : 0 ≤ Real.exp r1 * (1 - F.ue) := mul_nonneg he0.le (by linarith)
    have k1 : (a * Sm - n * B) * (Real.exp r1 * (1 - F.ue)) ≤ d * (Real.exp r1 * (1 - F.ue)) :=
      mul_le_mul_of_nonneg_right hdL hv0
    have k2 : d ≤ (P + B) * n := by nlinarith [mul_le_mul_of_nonneg_left hSp hP]
    have k3 : d * (Real.exp r1 * (1 - F.ue) - F.ηe) ≤ d * r2 :=
      mul_le_mul_of_nonneg_left (by linarith) hd
    have k4 : e2 * (1 - F.ue) ≤ Real.exp r1 * (1 - F.ue) :=
      mul_le_mul_of_nonneg_right he2 (by linarith)
    have k5 : Real.exp r1 * (1 - F.ue) ≤ 1 := by nlinarith
    have k6 : a * Sm * (e2 * (1 - F.ue)) ≤ a * Sm * (Real.exp r1 * (1 - F.ue)) :=
      mul_le_mul_of_nonneg_left k4 (mul_nonneg ha hSm)
    have k7 : n * B * (Real.exp r1 * (1 - F.ue)) ≤ n * B * 1 :=
      mul_le_mul_of_nonneg_left k5 (mul_nonneg hn hB)
    have k8 := mul_le_mul_of_nonneg_right k2 hηep
    have hZ : (1 - F.ue) * a * e2 * Sm - n * (B + (P + B) * F.ηe) ≤ d * r2 := by
      linarith
    exact r_mul_lo F h3 hx0 hZ
  · have := r_mul_lo F h3 hx0 (Z := 0) (by linarith)
    nlinarith
  · intro hc
    subst hc
    have hr1z : r1 = 0 := by rw [hr1]; simp
    rw [hr1z, Real.exp_zero] at hE1
    have hr2l : 1 - F.ue - F.ηe ≤ r2 := by linarith
    have hpos : 0 ≤ 1 - F.ue - F.ηe := by linarith
    have hZ : A * (1 - F.ue - F.ηe) ≤ d * r2 := by
      nlinarith [mul_le_mul_of_nonneg_right hdA hE3, mul_le_mul_of_nonneg_left hr2l hA]
    exact r_mul_lo F h3 hx0 hZ

set_option maxHeartbeats 2000000 in
theorem merge_step (F : Rounding) (hs : small F) {L R st : State_a} (hm : fmerge_a F L R st)
    {nL nR SpL SpR SmL SmR Sp Sm P B a A P' B' a' A' : ℝ}
    (hnL : 1 ≤ nL) (hnR : 1 ≤ nR)
    (hLU : L.d ≤ P * SpL + nL * B) (hLL : a * SmL - nL * B ≤ L.d) (hLA : A ≤ L.d)
    (hRU : R.d ≤ P * SpR + nR * B) (hRL : a * SmR - nR * B ≤ R.d) (hRA : A ≤ R.d)
    (hSpL0 : 0 ≤ SpL) (hSpL : SpL ≤ nL) (hSmL : 0 ≤ SmL)
    (hSpR0 : 0 ≤ SpR) (hSpR : SpR ≤ nR) (hSmR : 0 ≤ SmR)
    (hP : 0 ≤ P) (hB : 0 ≤ B) (ha : 0 ≤ a) (hA : 0 ≤ A)
    (hSp : Sp = Real.exp (-((1 - F.u) * (st.m - L.m))) * SpL
      + Real.exp (-((1 - F.u) * (st.m - R.m))) * SpR)
    (hSm : Sm = Real.exp (-((1 + F.u) * (st.m - L.m))) * SmL
      + Real.exp (-((1 + F.u) * (st.m - R.m))) * SmR)
    (hP' : (1 + F.u) ^ 2 * (1 + F.ue) * P ≤ P')
    (hB' : (1 + F.u) ^ 2 * ((1 + F.ue) * B + (P + B) * F.ηe) + (1 + F.u) * F.η ≤ B')
    (ha' : a' ≤ (1 - F.u) ^ 2 * (1 - F.ue) * a)
    (hA' : A' ≤ (1 - F.u) ^ 2 * (1 - F.ue - F.ηe) * A - 2 * (1 + F.u) * F.η) :
    st.m = max L.m R.m ∧ st.d ≤ P' * Sp + (nL + nR) * B' ∧
      a' * Sm - (nL + nR) * B' ≤ st.d ∧ A' ≤ st.d := by
  obtain ⟨v_m, v_d, r1, r2, r3, r4, r5, r6, r7, hvm, h1, h2, h3, h4, h5, h6, h7, hvd, hst⟩ := hm
  have hstm : st.m = v_m := by rw [hst]
  have hstd : st.d = r7 := by rw [hst, hvd]
  rw [hstm] at hSp hSm
  rw [hstm, hstd]
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hηp := F.hη
  have hηep := F.hηe
  have hs' : small F := hs
  obtain ⟨hu256, hue64, -, hη, hηe⟩ := hs
  have hLd0 : 0 ≤ L.d := hA.trans hLA
  have hRd0 : 0 ≤ R.d := hA.trans hRA
  have hLm : L.m ≤ v_m := by rw [hvm]; exact le_max_left _ _
  have hRm : R.m ≤ v_m := by rw [hvm]; exact le_max_right _ _
  obtain ⟨hL1, hL2, hL3, hL4⟩ := half_step F hs' hLm h1 h2 h3 hLd0 hLU hLL hLA hSpL0 hSpL hSmL hP hB ha hA
    (by linarith)
  obtain ⟨hR1, hR2, hR3, hR4⟩ := half_step F hs' hRm h4 h5 h6 hRd0 hRU hRL hRA hSpR0 hSpR hSmR hP hB ha hA
    (by linarith)
  refine ⟨hvm, ?_, ?_, ?_⟩
  · -- upper
    set e1L := Real.exp (-((1 - F.u) * (v_m - L.m))) with he1L
    set e1R := Real.exp (-((1 - F.u) * (v_m - R.m))) with he1R
    have e1L0 : 0 < e1L := Real.exp_pos _
    have e1R0 : 0 < e1R := Real.exp_pos _
    set cc := (1 + F.ue) * B + (P + B) * F.ηe with hcc
    have hcc0 : 0 ≤ cc := by positivity
    have hT0 : 0 ≤ (1 + F.u) * ((1 + F.ue) * P * e1L * SpL + nL * cc)
        + (1 + F.u) * ((1 + F.ue) * P * e1R * SpR + nR * cc) + 2 * F.η := by positivity
    have hs7 := r_add_up F h7 (T := (1 + F.u) * ((1 + F.ue) * P * e1L * SpL + nL * cc)
        + (1 + F.u) * ((1 + F.ue) * P * e1R * SpR + nR * cc) + 2 * F.η) (by linarith) hT0
    have q1 : (1 + F.u) ^ 2 * (1 + F.ue) * P * Sp ≤ P' * Sp := mul_le_mul_of_nonneg_right hP'
      (by rw [hSp]; positivity)
    have q2 : (nL + nR) * ((1 + F.u) ^ 2 * cc + (1 + F.u) * F.η) ≤ (nL + nR) * B' :=
      mul_le_mul_of_nonneg_left hB' (by linarith)
    have q3 : 0 ≤ (nL + nR - 2) * ((1 + F.u) * F.η) := mul_nonneg (by linarith) (by positivity)
    have q4 : 0 ≤ F.η * F.u := by positivity
    have q5 : (1 + F.u) * ((1 + F.u) * ((1 + F.ue) * P * e1L * SpL + nL * cc)
        + (1 + F.u) * ((1 + F.ue) * P * e1R * SpR + nR * cc) + 2 * F.η)
        = (1 + F.u) ^ 2 * (1 + F.ue) * P * Sp + (1 + F.u) ^ 2 * ((nL + nR) * cc) + 2 * (1 + F.u) * F.η := by
      rw [hSp]; ring
    linarith
  · -- lower
    set e2L := Real.exp (-((1 + F.u) * (v_m - L.m))) with he2L
    set e2R := Real.exp (-((1 + F.u) * (v_m - R.m))) with he2R
    have e2L0 : 0 < e2L := Real.exp_pos _
    have e2R0 : 0 < e2R := Real.exp_pos _
    set bb := B + (P + B) * F.ηe with hbb
    have hbb0 : 0 ≤ bb := by positivity
    have hZL : 0 ≤ (1 - F.ue) * a * e2L * SmL := by
      have : 0 ≤ 1 - F.ue := by linarith
      positivity
    have hZR : 0 ≤ (1 - F.ue) * a * e2R * SmR := by
      have : 0 ≤ 1 - F.ue := by linarith
      positivity
    have hz0 : 0 ≤ (1 - F.u) * ((1 - F.ue) * a * e2L * SmL + (1 - F.ue) * a * e2R * SmR) := by
      have : 0 ≤ 1 - F.u := by linarith
      positivity
    have he0 : 0 ≤ (1 - F.u) * ((nL + nR) * bb) + 2 * F.η := by
      have : 0 ≤ 1 - F.u := by linarith
      have : 0 ≤ nL + nR := by linarith
      positivity
    have hs7 := r_add_lo F h7 (z := (1 - F.u) * ((1 - F.ue) * a * e2L * SmL + (1 - F.ue) * a * e2R * SmR))
      (e := (1 - F.u) * ((nL + nR) * bb) + 2 * F.η) (by linarith) hz0 he0
    have q1 : a' * Sm ≤ (1 - F.u) ^ 2 * (1 - F.ue) * a * Sm := mul_le_mul_of_nonneg_right ha'
      (by rw [hSm]; positivity)
    have hcc : bb ≤ (1 + F.u) ^ 2 * ((1 + F.ue) * B + (P + B) * F.ηe) := by
      have h0 : 0 ≤ (1 + F.ue) * B + (P + B) * F.ηe := by positivity
      have h1 : bb ≤ (1 + F.ue) * B + (P + B) * F.ηe := by
        rw [hbb]; linarith [mul_nonneg hue hB]
      have h2 : (1 + F.ue) * B + (P + B) * F.ηe ≤ (1 + F.u) ^ 2 * ((1 + F.ue) * B + (P + B) * F.ηe) :=
        le_mul_of_one_le_left h0 (one_le_pow₀ (by linarith))
      linarith
    have q2 : (nL + nR) * (bb + (1 + F.u) * F.η) ≤ (nL + nR) * B' :=
      mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    have q3 : 0 ≤ (nL + nR - 2) * ((1 + F.u) * F.η) := mul_nonneg (by linarith) (by positivity)
    have q4 : 0 ≤ (nL + nR) * bb * (F.u * F.u) := by
      have : 0 ≤ nL + nR := by linarith
      positivity
    have q5 : (1 - F.u) * ((1 - F.u) * ((1 - F.ue) * a * e2L * SmL + (1 - F.ue) * a * e2R * SmR))
        = (1 - F.u) ^ 2 * (1 - F.ue) * a * Sm := by
      rw [hSm]; ring
    linarith
  · -- the max-holding child
    rcases le_total L.m R.m with hlr | hlr
    · have hvmR : v_m = R.m := by rw [hvm]; exact max_eq_right hlr
      have hc := hR4 hvmR.symm
      have hs7 := r_add_lo F h7 (z := (1 - F.u) * (A * (1 - F.ue - F.ηe))) (e := 2 * F.η)
        (by linarith) (by
          have : 0 ≤ 1 - F.u := by linarith
          have : 0 ≤ 1 - F.ue - F.ηe := by linarith
          positivity) (by linarith)
      linarith
    · have hvmL : v_m = L.m := by rw [hvm]; exact max_eq_left hlr
      have hc := hL4 hvmL.symm
      have hs7 := r_add_lo F h7 (z := (1 - F.u) * (A * (1 - F.ue - F.ηe))) (e := 2 * F.η)
        (by linarith) (by
          have : 0 ≤ 1 - F.u := by linarith
          have : 0 ≤ 1 - F.ue - F.ηe := by linarith
          positivity) (by linarith)
      linarith

theorem sum_affine (f : ℝ → ℝ) (c d : ℝ) (l : List ℝ) :
    (l.map fun x => c * f x + d).sum = c * (l.map f).sum + l.length * d := by
  induction l with
  | nil => simp
  | cons a l ih => simp [List.sum_cons, ih]; ring

theorem sum_exp_nonneg (θ m : ℝ) (l : List ℝ) :
    0 ≤ (l.map fun x => Real.exp (-(θ * (m - x)))).sum := by
  apply List.sum_nonneg
  intro y hy
  obtain ⟨x, -, rfl⟩ := List.mem_map.mp hy
  exact (Real.exp_pos _).le

theorem term_bounds (F : Rounding) (hs : small F) (m x y : ℝ)
    (h : ∃ q1 q2, F.add (x - m) q1 ∧ F.exp q1 q2 ∧ y = q2) :
    max 0 ((1 - F.ue) * Real.exp (min ((1 - F.u) * (x - m)) ((1 + F.u) * (x - m))) - F.ηe) ≤ y ∧
    y ≤ (1 + F.ue) * Real.exp (max ((1 - F.u) * (x - m)) ((1 + F.u) * (x - m))) + F.ηe ∧
    (x = m → 1 - F.ue - F.ηe ≤ y) := by
  obtain ⟨q1, q2, ⟨δ, hδ, hq⟩, hq2, rfl⟩ := h
  obtain ⟨hE1, hE2, hE3⟩ := F.exp_bounds hq2
  have hδ' := abs_le.mp hδ
  have hue := F.hue
  have hue1 := F.hue1
  have hq1 : min ((1 - F.u) * (x - m)) ((1 + F.u) * (x - m)) ≤ q1 := by
    rcases le_total 0 (x - m) with h0 | h0
    · refine (min_le_left _ _).trans ?_
      rw [hq]; nlinarith [mul_nonneg h0 (by linarith : 0 ≤ δ + F.u)]
    · refine (min_le_right _ _).trans ?_
      rw [hq]; nlinarith [mul_nonneg (neg_nonneg.2 h0) (by linarith : 0 ≤ F.u - δ)]
  have hq1' : q1 ≤ max ((1 - F.u) * (x - m)) ((1 + F.u) * (x - m)) := by
    rcases le_total 0 (x - m) with h0 | h0
    · refine le_trans ?_ (le_max_right _ _)
      rw [hq]; nlinarith [mul_nonneg h0 (by linarith : 0 ≤ F.u - δ)]
    · refine le_trans ?_ (le_max_left _ _)
      rw [hq]; nlinarith [mul_nonneg (neg_nonneg.2 h0) (by linarith : 0 ≤ δ + F.u)]
  refine ⟨max_le hE3 ?_, ?_, ?_⟩
  · have := mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 hq1) (by linarith : 0 ≤ 1 - F.ue)
    linarith
  · have := mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 hq1') (by linarith : 0 ≤ 1 + F.ue)
    linarith
  · intro hx
    subst hx
    have : q1 = 0 := by rw [hq]; simp
    rw [this, Real.exp_zero] at hE1
    linarith

theorem single_le_sum' (l : List ℝ) (hl : ∀ x ∈ l, 0 ≤ x) {a : ℝ} (ha : a ∈ l) : a ≤ l.sum := by
  exact List.single_le_sum hl a ha

set_option maxHeartbeats 1000000 in
theorem leaf_step (F : Rounding) (hs : small F) (D : ℕ) {A : List ℝ} {st : State_a} (hA : A ≠ [])
    (h : fblock_a F D A st) {P0 B0 a0 A0 : ℝ}
    (hP0 : (1 + F.u) ^ D * (1 + F.ue) ≤ P0) (hB0 : (1 + F.u) ^ D * F.ηe ≤ B0)
    (ha0 : a0 ≤ (1 - F.u) ^ D * (1 - F.ue))
    (hA0 : A0 ≤ (1 - F.u) ^ D * (1 - F.ue - F.ηe)) :
    st.m = lmax A ∧
    st.d ≤ P0 * (A.map fun x => Real.exp (-((1 - F.u) * (lmax A - x)))).sum + A.length * B0 ∧
    a0 * (A.map fun x => Real.exp (-((1 + F.u) * (lmax A - x)))).sum - A.length * B0 ≤ st.d ∧
    A0 ≤ st.d := by
  obtain ⟨v_m, v_d, r1, r2, hr1, hvm, hsum, hvd, hst⟩ := h
  have hvm' : v_m = lmax A := by rw [hvm, hr1]
  have hstm : st.m = lmax A := by rw [hst]; exact hvm'
  have hstd : st.d = r2 := by rw [hst, hvd]
  rw [hstd]
  rw [hvm'] at hsum
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hue1 := F.hue1
  have hηp := F.hη
  have hηep := F.hηe
  have hs' := hs
  obtain ⟨hu256, hue64, -, hη, hηe⟩ := hs
  have hηe' : F.ηe ≤ 1/2 := by
    have : (1:ℝ) / 2 ^ 100 ≤ 1/2 := by norm_num
    linarith
  have hmem : ∀ x ∈ A, x ≤ lmax A := fun x hx => le_lmax hx
  have hlen : (0:ℝ) ≤ A.length := Nat.cast_nonneg _
  have hSp0 := sum_exp_nonneg (1 - F.u) (lmax A) A
  have hSm0 := sum_exp_nonneg (1 + F.u) (lmax A) A
  have hpu : (1 + F.u) ^ D ≥ 1 := one_le_pow₀ (by linarith)
  have hpl : (1 - F.u) ^ D ≤ 1 := pow_le_one₀ (by linarith) (by linarith)
  have hpl0 : 0 ≤ (1 - F.u) ^ D := pow_nonneg (by linarith) _
  have hterm := term_bounds F hs' (lmax A)
  refine ⟨hstm, ?_, ?_, ?_⟩
  · have key := (fsum_bounds F D (fun x y => ∃ q1 q2, F.add (x - lmax A) q1 ∧ F.exp q1 q2 ∧ y = q2)
      (fun x => max 0 ((1 - F.ue) * Real.exp (min ((1 - F.u) * (x - lmax A)) ((1 + F.u) * (x - lmax A)))
        - F.ηe))
      (fun x => (1 + F.ue) * Real.exp (max ((1 - F.u) * (x - lmax A)) ((1 + F.u) * (x - lmax A))) + F.ηe)
      (fun x => le_max_left _ _) (fun x y hxy => ⟨(hterm x y hxy).1, (hterm x y hxy).2.1⟩) A r2 hsum).2
    have e1 : (A.map fun x => (1 + F.ue) * Real.exp (max ((1 - F.u) * (x - lmax A))
        ((1 + F.u) * (x - lmax A))) + F.ηe) =
        A.map fun x => (1 + F.ue) * Real.exp (-((1 - F.u) * (lmax A - x))) + F.ηe := by
      apply List.map_congr_left
      intro x hx
      have := hmem x hx
      rw [max_eq_left (by nlinarith)]
      congr 3; ring
    rw [e1, sum_affine (fun x => Real.exp (-((1 - F.u) * (lmax A - x)))) (1 + F.ue) F.ηe A] at key
    have k2 : (1 + F.u) ^ D * ((1 + F.ue) * (A.map fun x => Real.exp (-((1 - F.u) * (lmax A - x)))).sum
        + A.length * F.ηe) ≤ P0 * (A.map fun x => Real.exp (-((1 - F.u) * (lmax A - x)))).sum
        + A.length * B0 := by
      have a1 := mul_le_mul_of_nonneg_right hP0 hSp0
      have a2 := mul_le_mul_of_nonneg_left hB0 hlen
      nlinarith
    exact key.trans k2
  · have key := (fsum_bounds F D (fun x y => ∃ q1 q2, F.add (x - lmax A) q1 ∧ F.exp q1 q2 ∧ y = q2)
      (fun x => max 0 ((1 - F.ue) * Real.exp (min ((1 - F.u) * (x - lmax A)) ((1 + F.u) * (x - lmax A)))
        - F.ηe))
      (fun x => (1 + F.ue) * Real.exp (max ((1 - F.u) * (x - lmax A)) ((1 + F.u) * (x - lmax A))) + F.ηe)
      (fun x => le_max_left _ _) (fun x y hxy => ⟨(hterm x y hxy).1, (hterm x y hxy).2.1⟩) A r2 hsum).1
    have e1 : (A.map fun x => (1 - F.ue) * Real.exp (-((1 + F.u) * (lmax A - x))) - F.ηe) =
        A.map fun x => (1 - F.ue) * Real.exp (-((1 + F.u) * (lmax A - x))) + (-F.ηe) := by
      simp [sub_eq_add_neg]
    have e2 : (A.map fun x => (1 - F.ue) * Real.exp (-((1 + F.u) * (lmax A - x))) + (-F.ηe)).sum ≤
        (A.map fun x => max 0 ((1 - F.ue) * Real.exp (min ((1 - F.u) * (x - lmax A))
          ((1 + F.u) * (x - lmax A))) - F.ηe)).sum := by
      apply List.sum_le_sum
      intro x hx
      have := hmem x hx
      refine le_trans ?_ (le_max_right _ _)
      rw [min_eq_right (by nlinarith)]
      have : (1 + F.u) * (x - lmax A) = -((1 + F.u) * (lmax A - x)) := by ring
      rw [this]; linarith
    rw [sum_affine (fun x => Real.exp (-((1 + F.u) * (lmax A - x)))) (1 - F.ue) (-F.ηe) A] at e2
    have a1 := mul_le_mul_of_nonneg_left e2 hpl0
    have a2 := mul_le_mul_of_nonneg_right ha0 hSm0
    have a3 := mul_le_mul_of_nonneg_right hpl (mul_nonneg hlen hηep)
    have a4 : A.length * F.ηe ≤ A.length * B0 := by
      apply mul_le_mul_of_nonneg_left _ hlen
      nlinarith
    nlinarith
  · have key := (fsum_bounds F D (fun x y => ∃ q1 q2, F.add (x - lmax A) q1 ∧ F.exp q1 q2 ∧ y = q2)
      (fun x => if x = lmax A then 1 - F.ue - F.ηe else 0)
      (fun x => (1 + F.ue) * Real.exp (max ((1 - F.u) * (x - lmax A)) ((1 + F.u) * (x - lmax A))) + F.ηe)
      (fun x => by beta_reduce; split_ifs <;> linarith)
      (fun x y hxy => ⟨by
        beta_reduce
        split_ifs with hx
        · exact (hterm x y hxy).2.2 hx
        · exact (hterm x y hxy).1.trans' (le_max_left _ _), (hterm x y hxy).2.1⟩) A r2 hsum).1
    have hmem2 : (1 - F.ue - F.ηe) ∈ A.map (fun x => if x = lmax A then 1 - F.ue - F.ηe else 0) := by
      refine List.mem_map.mpr ⟨lmax A, lmax_mem hA, by simp⟩
    have := single_le_sum' _ (by
      intro x hx
      obtain ⟨y, -, rfl⟩ := List.mem_map.mp hx
      beta_reduce
      split_ifs <;> linarith) hmem2
    have a1 := mul_le_mul_of_nonneg_left this hpl0
    have a2 := mul_le_mul_of_nonneg_right hpl (by linarith : 0 ≤ 1 - F.ue - F.ηe)
    nlinarith

theorem fmerge_m (F : Rounding) {L R st : State_a} (h : fmerge_a F L R st) :
    st.m = max L.m R.m := by
  obtain ⟨v_m, v_d, r1, r2, r3, r4, r5, r6, r7, hvm, -, -, -, -, -, -, -, -, hst⟩ := h
  rw [hst]; exact hvm

theorem flat_ne: ∀ t : Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → t.flat ≠ []
  | Tree.leaf b, ht => by
    simpa [Tree.flat] using ht b (by simp [Tree.leaves])
  | Tree.node l r, ht => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [Tree.leaves, hb])
    have := flat_ne l hl
    simp [Tree.flat, this]

theorem sum_split (θ M mL mR : ℝ) (L R : List ℝ) :
    ((L ++ R).map fun x => Real.exp (-(θ * (M - x)))).sum
      = Real.exp (-(θ * (M - mL))) * (L.map fun x => Real.exp (-(θ * (mL - x)))).sum
        + Real.exp (-(θ * (M - mR))) * (R.map fun x => Real.exp (-(θ * (mR - x)))).sum := by
  have key : ∀ (m : ℝ) (l : List ℝ), (l.map fun x => Real.exp (-(θ * (M - x)))).sum
      = Real.exp (-(θ * (M - m))) * (l.map fun x => Real.exp (-(θ * (m - x)))).sum := by
    intro m l
    rw [← List.sum_map_mul_left]
    congr 1
    apply List.map_congr_left
    intro x _
    rw [← Real.exp_add]; congr 1; ring
  rw [List.map_append, List.sum_append, key mL L, key mR R]

theorem sum_exp_le (θ m : ℝ) (hθ : 0 ≤ θ) (l : List ℝ) (h : ∀ x ∈ l, x ≤ m) :
    (l.map fun x => Real.exp (-(θ * (m - x)))).sum ≤ l.length := by
  have := List.sum_le_card_nsmul (l.map fun x => Real.exp (-(θ * (m - x)))) 1 (by
    intro y hy
    obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hy
    rw [Real.exp_le_one_iff]
    nlinarith [mul_nonneg hθ (sub_nonneg.2 (h x hx))])
  simpa using this

set_option maxHeartbeats 1000000 in
theorem tree_inv (F : Rounding) (hs : small F) (D K : ℕ) (P B a A : ℕ → ℝ)
    (hP0 : (1 + F.u) ^ D * (1 + F.ue) ≤ P 0) (hB0 : (1 + F.u) ^ D * F.ηe ≤ B 0)
    (ha0 : a 0 ≤ (1 - F.u) ^ D * (1 - F.ue)) (hA0 : A 0 ≤ (1 - F.u) ^ D * (1 - F.ue - F.ηe))
    (hnn : ∀ j, j ≤ K → 0 ≤ P j ∧ 0 ≤ B j ∧ 0 ≤ a j ∧ 0 ≤ A j)
    (hstep : ∀ j, j < K → (1 + F.u) ^ 2 * (1 + F.ue) * P j ≤ P (j + 1) ∧
      (1 + F.u) ^ 2 * ((1 + F.ue) * B j + (P j + B j) * F.ηe) + (1 + F.u) * F.η ≤ B (j + 1) ∧
      a (j + 1) ≤ (1 - F.u) ^ 2 * (1 - F.ue) * a j ∧
      A (j + 1) ≤ (1 - F.u) ^ 2 * (1 - F.ue - F.ηe) * A j - 2 * (1 + F.u) * F.η) :
    ∀ (t : Tree ℝ), (∀ b ∈ t.leaves, b ≠ []) → ∀ k, t.depth ≤ k → k ≤ K →
      ∀ st, Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      st.m = lmax t.flat ∧
      st.d ≤ P k * (t.flat.map fun x => Real.exp (-((1 - F.u) * (lmax t.flat - x)))).sum
        + t.flat.length * B k ∧
      a k * (t.flat.map fun x => Real.exp (-((1 + F.u) * (lmax t.flat - x)))).sum
        - t.flat.length * B k ≤ st.d ∧
      A k ≤ st.d := by
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hηp := F.hη
  have hηep := F.hηe
  have hPm : ∀ k, k ≤ K → P 0 ≤ P k := by
    intro k
    induction k with
    | zero => intro _; exact le_rfl
    | succ k ih =>
      intro hk
      have h1 := ih (by omega)
      obtain ⟨h2, -⟩ := hstep k (by omega)
      have h3 := (hnn k (by omega)).1
      nlinarith [mul_nonneg h3 hu, mul_nonneg h3 hue, mul_nonneg (mul_nonneg h3 hu) hu,
        mul_nonneg (mul_nonneg h3 hu) hue, mul_nonneg (mul_nonneg (mul_nonneg h3 hu) hu) hue]
  have hBm : ∀ k, k ≤ K → B 0 ≤ B k := by
    intro k
    induction k with
    | zero => intro _; exact le_rfl
    | succ k ih =>
      intro hk
      have h1 := ih (by omega)
      obtain ⟨-, h2, -⟩ := hstep k (by omega)
      obtain ⟨h3, h4, -⟩ := hnn k (by omega)
      nlinarith [mul_nonneg h4 hu, mul_nonneg h4 hue, mul_nonneg (mul_nonneg h4 hu) hu,
        mul_nonneg (mul_nonneg h4 hu) hue, mul_nonneg (mul_nonneg (mul_nonneg h4 hu) hu) hue,
        mul_nonneg (add_nonneg h3 h4) hηep, mul_nonneg (mul_nonneg (add_nonneg h3 h4) hηep) hu,
        mul_nonneg (mul_nonneg (add_nonneg h3 h4) hηep) (mul_nonneg hu hu), mul_nonneg hu hηp]
  have ham : ∀ k, k ≤ K → a k ≤ a 0 := by
    intro k
    induction k with
    | zero => intro _; exact le_rfl
    | succ k ih =>
      intro hk
      have h1 := ih (by omega)
      obtain ⟨-, -, h2, -⟩ := hstep k (by omega)
      obtain ⟨-, -, h4, -⟩ := hnn k (by omega)
      have : (1 - F.u) ^ 2 * (1 - F.ue) * a k ≤ a k := by
        have hh : (1 - F.u) ^ 2 * (1 - F.ue) ≤ 1 := by
          have : (1 - F.u) ^ 2 ≤ 1 := by nlinarith
          nlinarith
        nlinarith
      linarith
  have hAm : ∀ k, k ≤ K → A k ≤ A 0 := by
    intro k
    induction k with
    | zero => intro _; exact le_rfl
    | succ k ih =>
      intro hk
      have h1 := ih (by omega)
      obtain ⟨-, -, -, h2⟩ := hstep k (by omega)
      obtain ⟨-, -, -, h4⟩ := hnn k (by omega)
      have : (1 - F.u) ^ 2 * (1 - F.ue - F.ηe) * A k ≤ A k := by
        have hh : (1 - F.u) ^ 2 * (1 - F.ue - F.ηe) ≤ 1 := by
          have : (1 - F.u) ^ 2 ≤ 1 := by nlinarith
          nlinarith
        nlinarith
      nlinarith [mul_nonneg hu hηp]
  intro t
  induction t with
  | leaf b =>
    intro ht k _ hk st hst
    cases hst with
    | leaf hb =>
      have hb0 : b ≠ [] := ht b (by simp [Tree.leaves])
      simp only [Tree.flat]
      exact leaf_step F hs D hb0 hb ((hP0.trans (hPm k hk))) (hB0.trans (hBm k hk))
        ((ham k hk).trans ha0) ((hAm k hk).trans hA0)
  | node l r ihl ihr =>
    intro ht k hdk hk st hst
    cases hst with
    | node hl hr hm =>
      rename_i x y
      have htl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [Tree.leaves, hb])
      have htr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => ht b (by simp [Tree.leaves, hb])
      have hdk' : max l.depth r.depth + 1 ≤ k := hdk
      obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
      have hjK : j < K := by omega
      obtain ⟨hxm, hxU, hxL, hxA⟩ := ihl htl j (by omega) (by omega) x hl
      obtain ⟨hym, hyU, hyL, hyA⟩ := ihr htr j (by omega) (by omega) y hr
      obtain ⟨hP1, hB1, ha1, hA1⟩ := hstep j hjK
      obtain ⟨hPj, hBj, haj, hAj⟩ := hnn j (by omega)
      have hfl := flat_ne l htl
      have hfr := flat_ne r htr
      have hnL : (1 : ℝ) ≤ l.flat.length := by
        have := List.length_pos_of_ne_nil hfl
        exact_mod_cast this
      have hnR : (1 : ℝ) ≤ r.flat.length := by
        have := List.length_pos_of_ne_nil hfr
        exact_mod_cast this
      have hmax : lmax (l.flat ++ r.flat) = max (lmax l.flat) (lmax r.flat) := lmax_append hfl hfr
      have hstm0 := fmerge_m F hm
      have hstm : st.m = lmax (l.flat ++ r.flat) := by rw [hstm0, hmax, hxm, hym]
      have hsp := sum_split (1 - F.u) (lmax (l.flat ++ r.flat)) (lmax l.flat) (lmax r.flat)
        l.flat r.flat
      have hsm := sum_split (1 + F.u) (lmax (l.flat ++ r.flat)) (lmax l.flat) (lmax r.flat)
        l.flat r.flat
      have hlen : (((l.flat ++ r.flat).length : ℕ) : ℝ) = (l.flat.length : ℝ) + (r.flat.length : ℝ) := by
        simp
      have key := merge_step F hs hm (nL := l.flat.length) (nR := r.flat.length)
        (SpL := (l.flat.map fun z => Real.exp (-((1 - F.u) * (lmax l.flat - z)))).sum)
        (SpR := (r.flat.map fun z => Real.exp (-((1 - F.u) * (lmax r.flat - z)))).sum)
        (SmL := (l.flat.map fun z => Real.exp (-((1 + F.u) * (lmax l.flat - z)))).sum)
        (SmR := (r.flat.map fun z => Real.exp (-((1 + F.u) * (lmax r.flat - z)))).sum)
        (Sp := ((l.flat ++ r.flat).map fun z =>
          Real.exp (-((1 - F.u) * (lmax (l.flat ++ r.flat) - z)))).sum)
        (Sm := ((l.flat ++ r.flat).map fun z =>
          Real.exp (-((1 + F.u) * (lmax (l.flat ++ r.flat) - z)))).sum)
        (P := P j) (B := B j) (a := a j) (A := A j) (P' := P (j + 1)) (B' := B (j + 1))
        (a' := a (j + 1)) (A' := A (j + 1)) hnL hnR hxU hxL hxA hyU hyL hyA
        (sum_exp_nonneg _ _ _) (sum_exp_le _ _ (by linarith) _ (fun z hz => le_lmax hz))
        (sum_exp_nonneg _ _ _)
        (sum_exp_nonneg _ _ _) (sum_exp_le _ _ (by linarith) _ (fun z hz => le_lmax hz))
        (sum_exp_nonneg _ _ _) hPj hBj haj hAj
        (by rw [hstm, hxm, hym]; exact hsp) (by rw [hstm, hxm, hym]; exact hsm)
        hP1 hB1 ha1 hA1
      simp only [Tree.flat]
      rw [hlen]
      exact ⟨hstm0.trans (by rw [hmax, hxm, hym]), key.2⟩

theorem exp_le_inv (y : ℝ) (hy : y ≤ 1 / 6) : Real.exp y ≤ 6 / 5 := by
  have h1 := Real.add_one_le_exp (-y)
  have h2 : Real.exp y * Real.exp (-y) = 1 := by rw [← Real.exp_add]; simp
  have h3 := Real.exp_pos y
  nlinarith [mul_le_mul_of_nonneg_left h1 h3.le]

theorem rho_le (u ue : ℝ) (hu : 0 ≤ u) (hue : 0 ≤ ue) (hu1 : u ≤ 1 / 256) (hue1 : ue ≤ 1 / 64) :
    (1 + u) ^ 2 * (1 + ue) ≤ 1 + (51 / 25) * (u + ue) := by
  nlinarith [mul_nonneg hu hu, mul_nonneg hu hue, mul_nonneg (mul_nonneg hu hu) hue]

theorem num_P (u ue : ℝ) (hu : 0 ≤ u) (hue : 0 ≤ ue) (hu1 : u ≤ 1 / 256) (hue1 : ue ≤ 1 / 64)
    (D j : ℕ) (h : ((D : ℝ) + j) * (u + ue) ≤ 1 / 16) :
    (1 + u) ^ D * (1 + ue) * ((1 + u) ^ 2 * (1 + ue)) ^ j ≤ 6 / 5 := by
  have h1 : (1 + u) ^ D ≤ Real.exp (D * u) := by
    rw [Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp u]) D
  have h2 : 1 + ue ≤ Real.exp ue := by linarith [Real.add_one_le_exp ue]
  have h3 : ((1 + u) ^ 2 * (1 + ue)) ^ j ≤ Real.exp (j * ((51 / 25) * (u + ue))) := by
    rw [Real.exp_nat_mul]
    refine pow_le_pow_left₀ (by positivity) ?_ j
    exact (rho_le u ue hu hue hu1 hue1).trans (by linarith [Real.add_one_le_exp ((51 / 25) * (u + ue))])
  have h4 : (1 + u) ^ D * (1 + ue) * ((1 + u) ^ 2 * (1 + ue)) ^ j ≤
      Real.exp (D * u) * Real.exp ue * Real.exp (j * ((51 / 25) * (u + ue))) := by
    gcongr
  rw [← Real.exp_add, ← Real.exp_add] at h4
  refine h4.trans (exp_le_inv _ ?_)
  have hD : (0 : ℝ) ≤ D := Nat.cast_nonneg _
  have hj : (0 : ℝ) ≤ j := Nat.cast_nonneg _
  nlinarith [mul_nonneg hD hue, mul_nonneg hj hu, mul_nonneg hj hue]

theorem seq_step_B (u ue η ηe Pj x : ℝ) (hu : 0 ≤ u) (hue : 0 ≤ ue) (hη : 0 ≤ η) (hηe : 0 ≤ ηe)
    (hu1 : u ≤ 1 / 256) (hue1 : ue ≤ 1 / 64) (hx : 0 ≤ x) (hxs : x * (u + ue) ≤ 1 / 16)
    (hPj : Pj ≤ 6 / 5) (hxb : 2 * x * (η + ηe) ≤ 1 / 100) :
    (1 + u) ^ 2 * ((1 + ue) * (2 * x * (η + ηe)) + (Pj + 2 * x * (η + ηe)) * ηe) + (1 + u) * η
      ≤ 2 * (x + 1) * (η + ηe) := by
  have hρ := rho_le u ue hu hue hu1 hue1
  have hη' : 0 ≤ η + ηe := by linarith
  have hb : 0 ≤ 2 * x * (η + ηe) := by positivity
  have e1 : (1 + u) ^ 2 * (1 + ue) * (2 * x * (η + ηe)) ≤
      (1 + (51 / 25) * (u + ue)) * (2 * x * (η + ηe)) := mul_le_mul_of_nonneg_right hρ hb
  have e2 : x * (u + ue) * (η + ηe) ≤ 1 / 16 * (η + ηe) := mul_le_mul_of_nonneg_right hxs hη'
  have e3 : (1 + u) ^ 2 ≤ 51 / 50 := by nlinarith
  have e4 : Pj + 2 * x * (η + ηe) ≤ 13 / 10 := by linarith
  have e5 : (1 + u) ^ 2 * ((Pj + 2 * x * (η + ηe)) * ηe) ≤ 51 / 50 * (13 / 10 * ηe) := by
    have h1 : (Pj + 2 * x * (η + ηe)) * ηe ≤ 13 / 10 * ηe := mul_le_mul_of_nonneg_right e4 hηe
    have h2 : 0 ≤ (1 + u) ^ 2 := by positivity
    calc (1 + u) ^ 2 * ((Pj + 2 * x * (η + ηe)) * ηe) ≤ (1 + u) ^ 2 * (13 / 10 * ηe) :=
          mul_le_mul_of_nonneg_left h1 h2
      _ ≤ 51 / 50 * (13 / 10 * ηe) := mul_le_mul_of_nonneg_right e3 (by positivity)
  have e6 : (1 + u) * η ≤ 51 / 50 * η := by nlinarith
  nlinarith

theorem num_Q (u v : ℝ) (hu : 0 ≤ u) (hv : 0 ≤ v) (hu1 : u ≤ 1) (hv1 : v ≤ 1) (N M : ℕ)
    (hp : N * u ≤ 1) :
    1 - N * u - M * v ≤ (1 - u) ^ N * (1 - v) ^ M := by
  have h1 : 1 - N * u ≤ (1 - u) ^ N := by
    have := one_add_mul_le_pow (a := -u) (by linarith) N
    simpa [sub_eq_add_neg, neg_mul, mul_neg] using this
  have h2 : 1 - M * v ≤ (1 - v) ^ M := by
    have := one_add_mul_le_pow (a := -v) (by linarith) M
    simpa [sub_eq_add_neg, neg_mul, mul_neg] using this
  have h3 : 0 ≤ (1 - v) ^ M := pow_nonneg (by linarith) _
  have hN : (0 : ℝ) ≤ N := Nat.cast_nonneg _
  have hM : (0 : ℝ) ≤ M := Nat.cast_nonneg _
  nlinarith [mul_nonneg (sub_nonneg.2 h1) h3, mul_nonneg (by linarith : 0 ≤ 1 - (N:ℝ) * u) (sub_nonneg.2 h2),
    mul_nonneg (mul_nonneg hN hu) (mul_nonneg hM hv)]

theorem Q_lower (u v : ℝ) (hu : 0 ≤ u) (hv : 0 ≤ v) (hu1 : u ≤ 1) (hv1 : v ≤ 1) (D j : ℕ)
    (hp : ((D : ℝ) + 2 * j) * u ≤ 1) :
    1 - ((D : ℝ) + 2 * j) * u - ((j : ℝ) + 1) * v ≤
      (1 - u) ^ D * (1 - v) * ((1 - u) ^ 2 * (1 - v)) ^ j := by
  have := num_Q u v hu hv hu1 hv1 (D + 2 * j) (j + 1) (by push_cast; exact hp)
  have e : (1 - u) ^ D * (1 - v) * ((1 - u) ^ 2 * (1 - v)) ^ j
      = (1 - u) ^ (D + 2 * j) * (1 - v) ^ (j + 1) := by
    rw [mul_pow, ← pow_mul, pow_add, pow_succ]
    generalize 1 - u = p
    generalize 1 - v = q
    ring
  rw [e]
  push_cast at this
  exact this

theorem depth_len : ∀ t : Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → t.depth + 1 ≤ t.flat.length
  | Tree.leaf b, ht => by
    have : b ≠ [] := ht b (by simp [Tree.leaves])
    simp [Tree.depth, Tree.flat, List.length_pos_iff.mpr this]
    exact List.length_pos_iff.mpr this
  | Tree.node l r, ht => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [Tree.leaves, hb])
    have hr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => ht b (by simp [Tree.leaves, hb])
    have h1 := depth_len l hl
    have h2 := depth_len r hr
    simp only [Tree.depth, Tree.flat, List.length_append]
    omega

theorem seq_nn (F : Rounding) (hs : small F) (D K : ℕ)
    (hT1 : ((K : ℝ) + D + 1) * (F.u + F.ue) ≤ 1 / 16)
    (hKη : ((K : ℝ) + 1) * (F.η + F.ηe) ≤ F.u) (j : ℕ) (hj : j ≤ K) :
    0 ≤ (1 + F.u) ^ D * (1 + F.ue) * ((1 + F.u) ^ 2 * (1 + F.ue)) ^ j ∧
    0 ≤ 2 * ((j : ℝ) + 1) * (F.η + F.ηe) ∧
    0 ≤ (1 - F.u) ^ D * (1 - F.ue) * ((1 - F.u) ^ 2 * (1 - F.ue)) ^ j ∧
    0 ≤ (1 - F.u) ^ D * (1 - F.ue - F.ηe) * ((1 - F.u) ^ 2 * (1 - F.ue - F.ηe)) ^ j
      - 5 / 2 * (j : ℝ) * F.η := by
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hηp := F.hη
  have hηep := F.hηe
  obtain ⟨hu256, hue64, -, hη, hηe⟩ := hs
  have hηe' : F.ηe ≤ 1 / 100 := by
    have : (1:ℝ) / 2 ^ 100 ≤ 1 / 100 := by norm_num
    linarith
  have hjK : (j : ℝ) ≤ K := by exact_mod_cast hj
  have hD : (0 : ℝ) ≤ D := Nat.cast_nonneg _
  have hj0 : (0 : ℝ) ≤ j := Nat.cast_nonneg _
  refine ⟨by positivity, by positivity, ?_, ?_⟩
  · have : 0 ≤ 1 - F.ue := by linarith
    have : 0 ≤ 1 - F.u := by linarith
    positivity
  · have h1 : ((D : ℝ) + j + 1) * (F.u + F.ue) ≤ 1 / 16 :=
      (mul_le_mul_of_nonneg_right (by linarith) (by linarith)).trans hT1
    have h2 : ((j : ℝ) + 1) * (F.η + F.ηe) ≤ F.u :=
      (mul_le_mul_of_nonneg_right (by linarith) (by linarith)).trans hKη
    have hp : ((D : ℝ) + 2 * j) * F.u ≤ 1 := by
      nlinarith [mul_nonneg hD hu, mul_nonneg hj0 hu, mul_nonneg hD hue, mul_nonneg hj0 hue]
    have hq := Q_lower F.u (F.ue + F.ηe) hu (by linarith) (by linarith) (by linarith) D j hp
    have e : 1 - (F.ue + F.ηe) = 1 - F.ue - F.ηe := by ring
    rw [e] at hq
    nlinarith [mul_nonneg hD hu, mul_nonneg hj0 hu, mul_nonneg hD hue, mul_nonneg hj0 hue,
      mul_nonneg hj0 hηp, mul_nonneg hj0 hηep]

theorem seq_step (F : Rounding) (hs : small F) (D K : ℕ)
    (hT1 : ((K : ℝ) + D + 1) * (F.u + F.ue) ≤ 1 / 16)
    (hKη : ((K : ℝ) + 1) * (F.η + F.ηe) ≤ F.u) (j : ℕ) (hj : j < K) :
    (1 + F.u) ^ 2 * (1 + F.ue) * ((1 + F.u) ^ D * (1 + F.ue) * ((1 + F.u) ^ 2 * (1 + F.ue)) ^ j)
      ≤ (1 + F.u) ^ D * (1 + F.ue) * ((1 + F.u) ^ 2 * (1 + F.ue)) ^ (j + 1) ∧
    (1 + F.u) ^ 2 * ((1 + F.ue) * (2 * ((j : ℝ) + 1) * (F.η + F.ηe))
      + ((1 + F.u) ^ D * (1 + F.ue) * ((1 + F.u) ^ 2 * (1 + F.ue)) ^ j
        + 2 * ((j : ℝ) + 1) * (F.η + F.ηe)) * F.ηe) + (1 + F.u) * F.η
      ≤ 2 * (((j + 1 : ℕ) : ℝ) + 1) * (F.η + F.ηe) ∧
    (1 - F.u) ^ D * (1 - F.ue) * ((1 - F.u) ^ 2 * (1 - F.ue)) ^ (j + 1)
      ≤ (1 - F.u) ^ 2 * (1 - F.ue) * ((1 - F.u) ^ D * (1 - F.ue) * ((1 - F.u) ^ 2 * (1 - F.ue)) ^ j) ∧
    (1 - F.u) ^ D * (1 - F.ue - F.ηe) * ((1 - F.u) ^ 2 * (1 - F.ue - F.ηe)) ^ (j + 1)
      - 5 / 2 * ((j + 1 : ℕ) : ℝ) * F.η ≤
    (1 - F.u) ^ 2 * (1 - F.ue - F.ηe) * ((1 - F.u) ^ D * (1 - F.ue - F.ηe) *
      ((1 - F.u) ^ 2 * (1 - F.ue - F.ηe)) ^ j - 5 / 2 * (j : ℝ) * F.η) - 2 * (1 + F.u) * F.η := by
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hηp := F.hη
  have hηep := F.hηe
  have hs' : small F := hs
  obtain ⟨hu256, hue64, -, hη, hηe⟩ := hs
  have hjK : (j : ℝ) + 1 ≤ K := by exact_mod_cast hj
  have hD : (0 : ℝ) ≤ D := Nat.cast_nonneg _
  have hj0 : (0 : ℝ) ≤ j := Nat.cast_nonneg _
  refine ⟨le_of_eq (by ring), ?_, le_of_eq (by ring), ?_⟩
  · have hP : (1 + F.u) ^ D * (1 + F.ue) * ((1 + F.u) ^ 2 * (1 + F.ue)) ^ j ≤ 6 / 5 :=
      num_P F.u F.ue hu hue hu256 hue64 D j (by
        have : ((D : ℝ) + j) * (F.u + F.ue) ≤ ((K : ℝ) + D + 1) * (F.u + F.ue) :=
          mul_le_mul_of_nonneg_right (by linarith) (by linarith)
        linarith)
    have := seq_step_B F.u F.ue F.η F.ηe _ ((j : ℝ) + 1) hu hue hηp hηep hu256 hue64 (by linarith)
      ((mul_le_mul_of_nonneg_right (by linarith) (by linarith)).trans hT1) hP (by
        have : ((j : ℝ) + 1) * (F.η + F.ηe) ≤ F.u :=
          (mul_le_mul_of_nonneg_right (by linarith) (by linarith)).trans hKη
        linarith)
    push_cast
    linarith
  · have hρ : (1 - F.u) ^ 2 * (1 - F.ue - F.ηe) ≤ 1 := by
      have hηe' : F.ηe ≤ 1 / 100 := by
        have : (1:ℝ) / 2 ^ 100 ≤ 1 / 100 := by norm_num
        linarith
      have : (1 - F.u) ^ 2 ≤ 1 := by nlinarith
      nlinarith
    have e : (1 - F.u) ^ D * (1 - F.ue - F.ηe) * ((1 - F.u) ^ 2 * (1 - F.ue - F.ηe)) ^ (j + 1)
        = (1 - F.u) ^ 2 * (1 - F.ue - F.ηe) * ((1 - F.u) ^ D * (1 - F.ue - F.ηe) *
          ((1 - F.u) ^ 2 * (1 - F.ue - F.ηe)) ^ j) := by ring
    rw [e]
    push_cast
    nlinarith [mul_nonneg (mul_nonneg hj0 hηp) (sub_nonneg.2 hρ)]

theorem upper_core (c d S PK nB Sp : ℝ) (hS : 1 ≤ S) (hSp : Sp ≤ Real.exp c * S) (hP1 : 1 ≤ PK)
    (hc : 0 ≤ c) (hnB : 0 ≤ nB) (hdU : d ≤ PK * Sp + nB) (hd0 : 0 < d) :
    Real.log d - Real.log S ≤ c + Real.log PK + nB := by
  have hec : 1 ≤ Real.exp c := Real.one_le_exp hc
  set y := PK * Real.exp c with hy
  have hy1 : 1 ≤ y := by rw [hy]; nlinarith
  have hy0 : 0 < y := by linarith
  have h1 : d ≤ (y + nB) * S := by
    have := mul_le_mul_of_nonneg_left hSp (by linarith : 0 ≤ PK)
    nlinarith
  have hx0 : 0 < y + nB := by linarith
  have hS0 : 0 < S := by linarith
  have h2 : Real.log d ≤ Real.log (y + nB) + Real.log S := by
    rw [← Real.log_mul hx0.ne' hS0.ne']
    exact Real.log_le_log hd0 h1
  have h3 : Real.log (y + nB) ≤ Real.log y + nB := by
    have := Real.log_le_sub_one_of_pos (by positivity : 0 < (y + nB) / y)
    rw [Real.log_div hx0.ne' hy0.ne'] at this
    have h4 : (y + nB) / y - 1 = nB / y := by field_simp; ring
    have h5 : nB / y ≤ nB := by
      rw [div_le_iff₀ hy0]; nlinarith
    linarith
  have h6 : Real.log y = Real.log PK + c := by
    rw [hy, Real.log_mul (by linarith) (Real.exp_pos c).ne', Real.log_exp]
  linarith

theorem lower_core (c d S Sm aK AK nB W : ℝ) (hS : 1 ≤ S) (hSm : Real.exp (-c) * S ≤ Sm)
    (hd1 : aK * Sm - nB ≤ d) (hd2 : AK ≤ d) (haK : 0 ≤ aK) (hnB : 0 ≤ nB)
    (hW1 : W ≤ aK - nB) (hW2 : W ≤ AK) (hW : 0 < W) :
    Real.log W - c ≤ Real.log d - Real.log S := by
  have hS0 : 0 < S := by linarith
  rcases le_or_gt S (Real.exp c) with h | h
  · have hd0 : 0 < d := by linarith
    have h1 : Real.log W ≤ Real.log d := Real.log_le_log hW (by linarith)
    have h2 : Real.log S ≤ c := (Real.log_le_iff_le_exp hS0).2 h
    linarith
  · have hE : Real.exp (-c) * Real.exp c = 1 := by rw [← Real.exp_add]; simp
    have hE0 := Real.exp_pos (-c)
    have h3 : 1 ≤ Real.exp (-c) * S := by
      have := mul_le_mul_of_nonneg_left h.le hE0.le
      linarith
    have h4 : aK * (Real.exp (-c) * S) ≤ aK * Sm := mul_le_mul_of_nonneg_left hSm haK
    have h5 : nB ≤ nB * (Real.exp (-c) * S) := by nlinarith
    have h6 : W * (Real.exp (-c) * S) ≤ d := by
      have : (aK - nB) * (Real.exp (-c) * S) ≤ d := by nlinarith
      have h7 := mul_le_mul_of_nonneg_right hW1 (by linarith : 0 ≤ Real.exp (-c) * S)
      linarith
    have hd0 : 0 < d := lt_of_lt_of_le (by positivity) h6
    have := Real.log_le_log (by positivity) h6
    rw [Real.log_mul hW.ne' (by positivity), Real.log_mul hE0.ne' hS0.ne', Real.log_exp] at this
    linarith

theorem log_one_add_le (x : ℝ) (hx : 0 ≤ x) : Real.log (1 + x) ≤ x := by
  have := Real.log_le_sub_one_of_pos (by linarith : 0 < 1 + x)
  linarith

theorem log_one_sub_ge (x : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1 / 60) :
    -(51 / 50 * x) ≤ Real.log (1 - x) := by
  have h0 : 0 < 1 - x := by linarith
  have := Real.one_sub_inv_le_log_of_pos h0
  have h1 : (1 - x)⁻¹ ≤ 1 + 51 / 50 * x := by
    rw [inv_eq_one_div, div_le_iff₀ h0]
    nlinarith
  linarith

theorem logP (u ue : ℝ) (hu : 0 ≤ u) (hue : 0 ≤ ue) (D K : ℕ) :
    1 ≤ (1 + u) ^ D * (1 + ue) * ((1 + u) ^ 2 * (1 + ue)) ^ K ∧
    Real.log ((1 + u) ^ D * (1 + ue) * ((1 + u) ^ 2 * (1 + ue)) ^ K) ≤
      D * u + ue + K * (2 * u + ue) := by
  have h1 : 0 < 1 + u := by linarith
  have h2 : 0 < 1 + ue := by linarith
  refine ⟨?_, ?_⟩
  · have a1 : 1 ≤ (1 + u) ^ D := one_le_pow₀ (by linarith)
    have a2 : 1 ≤ ((1 + u) ^ 2 * (1 + ue)) ^ K :=
      one_le_pow₀ (by nlinarith [mul_nonneg hu hue, mul_nonneg hu hu])
    have a3 : 1 ≤ (1 + u) ^ D * (1 + ue) := by nlinarith
    nlinarith
  · rw [Real.log_mul (by positivity) (by positivity), Real.log_mul (by positivity) (by positivity),
      Real.log_pow, Real.log_pow, Real.log_mul (by positivity) (by positivity), Real.log_pow]
    have := log_one_add_le u hu
    have := log_one_add_le ue hue
    push_cast
    nlinarith [Nat.cast_nonneg (α := ℝ) D, Nat.cast_nonneg (α := ℝ) K,
      mul_nonneg (Nat.cast_nonneg (α := ℝ) D) (sub_nonneg.2 ‹Real.log (1 + u) ≤ u›),
      mul_nonneg (Nat.cast_nonneg (α := ℝ) K) (sub_nonneg.2 ‹Real.log (1 + u) ≤ u›),
      mul_nonneg (Nat.cast_nonneg (α := ℝ) K) (sub_nonneg.2 ‹Real.log (1 + ue) ≤ ue›)]

theorem logQ (u v : ℝ) (hu : 0 ≤ u) (hv : 0 ≤ v) (hu1 : u ≤ 1 / 60) (hv1 : v ≤ 1 / 60) (D K : ℕ) :
    -(51 / 50 * (((D : ℝ) + 2 * K) * u + ((K : ℝ) + 1) * v)) ≤
      Real.log ((1 - u) ^ D * (1 - v) * ((1 - u) ^ 2 * (1 - v)) ^ K) := by
  have h1 : 0 < 1 - u := by linarith
  have h2 : 0 < 1 - v := by linarith
  rw [Real.log_mul (by positivity) (by positivity), Real.log_mul (by positivity) (by positivity),
      Real.log_pow, Real.log_pow, Real.log_mul (by positivity) (by positivity), Real.log_pow]
  have a1 := log_one_sub_ge u hu hu1
  have a2 := log_one_sub_ge v hv hv1
  push_cast
  have hD := Nat.cast_nonneg (α := ℝ) D
  have hK := Nat.cast_nonneg (α := ℝ) K
  nlinarith [mul_nonneg hD (by linarith : 0 ≤ Real.log (1 - u) + 51 / 50 * u),
    mul_nonneg hK (by linarith : 0 ≤ Real.log (1 - u) + 51 / 50 * u),
    mul_nonneg hK (by linarith : 0 ≤ Real.log (1 - v) + 51 / 50 * v)]

theorem log_sub_ge (Q z : ℝ) (hQ : 0 < Q) (hz : 0 ≤ z) (hQz : 0 < Q - z) :
    Real.log Q - z / (Q - z) ≤ Real.log (Q - z) := by
  have h0 : 0 < (Q - z) / Q := by positivity
  have := Real.one_sub_inv_le_log_of_pos h0
  rw [Real.log_div hQz.ne' hQ.ne'] at this
  have e : 1 - ((Q - z) / Q)⁻¹ = -(z / (Q - z)) := by
    field_simp
    ring
  linarith

theorem Q_le_a (u ue ηe : ℝ) (hu : 0 ≤ u) (hu1 : u ≤ 1) (hηe : 0 ≤ ηe) (hue1 : ue + ηe ≤ 1)
    (hue : 0 ≤ ue) (D K : ℕ) :
    (1 - u) ^ D * (1 - ue - ηe) * ((1 - u) ^ 2 * (1 - ue - ηe)) ^ K ≤
      (1 - u) ^ D * (1 - ue) * ((1 - u) ^ 2 * (1 - ue)) ^ K := by
  have h1 : 0 ≤ 1 - u := by linarith
  have h2 : 0 ≤ 1 - ue - ηe := by linarith
  have h3 : 0 ≤ 1 - ue := by linarith
  have h4 : 0 ≤ (1 - u) ^ D := pow_nonneg h1 _
  have h5 : 1 - ue - ηe ≤ 1 - ue := by linarith
  have h6 : (1 - u) ^ 2 * (1 - ue - ηe) ≤ (1 - u) ^ 2 * (1 - ue) :=
    mul_le_mul_of_nonneg_left h5 (by positivity)
  have h7 : ((1 - u) ^ 2 * (1 - ue - ηe)) ^ K ≤ ((1 - u) ^ 2 * (1 - ue)) ^ K :=
    pow_le_pow_left₀ (by positivity) h6 K
  have h8 : (1 - u) ^ D * (1 - ue - ηe) ≤ (1 - u) ^ D * (1 - ue) :=
    mul_le_mul_of_nonneg_left h5 h4
  exact mul_le_mul h8 h7 (by positivity) (by positivity)

theorem sums_cmp (X : List ℝ) (lo hi u : ℝ) (hu : 0 ≤ u) (hX : X ≠ [])
    (hb : ∀ x ∈ X, lo ≤ x ∧ x ≤ hi) :
    (X.map fun x => Real.exp (-((1 - u) * (lmax X - x)))).sum ≤
      Real.exp (u * (hi - lo)) * (X.map fun x => Real.exp (x - lmax X)).sum ∧
    Real.exp (-(u * (hi - lo))) * (X.map fun x => Real.exp (x - lmax X)).sum ≤
      (X.map fun x => Real.exp (-((1 + u) * (lmax X - x)))).sum ∧
    1 ≤ (X.map fun x => Real.exp (x - lmax X)).sum ∧
    (X.map Real.exp).sum = Real.exp (lmax X) * (X.map fun x => Real.exp (x - lmax X)).sum := by
  have hm : lmax X ≤ hi := (hb _ (lmax_mem hX)).2
  have hmx : ∀ x ∈ X, u * (lmax X - x) ≤ u * (hi - lo) := by
    intro x hx
    have := (hb x hx).1
    exact mul_le_mul_of_nonneg_left (by linarith) hu
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [← List.sum_map_mul_left]
    apply List.sum_le_sum
    intro x hx
    rw [← Real.exp_add]
    apply Real.exp_le_exp.2
    have := hmx x hx
    nlinarith
  · rw [← List.sum_map_mul_left]
    apply List.sum_le_sum
    intro x hx
    rw [← Real.exp_add]
    apply Real.exp_le_exp.2
    have := hmx x hx
    nlinarith
  · have h1 : Real.exp (lmax X - lmax X) ∈ X.map fun x => Real.exp (x - lmax X) :=
      List.mem_map.mpr ⟨lmax X, lmax_mem hX, rfl⟩
    have := List.single_le_sum (l := X.map fun x => Real.exp (x - lmax X))
      (by
        intro y hy
        obtain ⟨x, -, rfl⟩ := List.mem_map.mp hy
        exact (Real.exp_pos _).le) _ h1
    simpa using this
  · rw [← List.sum_map_mul_left]
    congr 1
    apply List.map_congr_left
    intro x _
    rw [← Real.exp_add]; congr 1; ring

theorem log_connect (m d S : ℝ) (hd : 0 < d) (hS : 0 < S) :
    Real.log (1 / (Real.exp ((0 : ℝ) - m) / d)) - Real.log (Real.exp m * S)
      = Real.log d - Real.log S := by
  have h1 : 1 / (Real.exp ((0 : ℝ) - m) / d) = d * Real.exp m := by
    rw [zero_sub, Real.exp_neg]
    field_simp
  rw [h1, Real.log_mul hd.ne' (Real.exp_pos m).ne', Real.log_mul (Real.exp_pos m).ne' hS.ne',
    Real.log_exp]
  ring

set_option maxHeartbeats 4000000 in
theorem fp_bound_main : ∀ (F : Discovery.Rounding) (D : ℕ), ∀ (lo hi : ℝ) (X : List ℝ),
    X ≠ [] → (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((2 : ℝ) * F.u * (hi - lo)) + (((5 : ℝ) * ((1 : ℝ) / 4)) * (((D : ℝ) * F.u) + ((2 : ℝ) * (t.depth : ℝ) * F.u) + F.u + (((t.depth : ℝ) + (1 : ℝ)) * F.ue))) + ((4 : ℝ) * (X.length : ℝ) * ((t.depth : ℝ) + (2 : ℝ)) * (F.η + F.ηe))) := by
  intro F D lo hi X hX hbd hs hnη t hflat hleaves hT st hst
  subst hflat
  obtain ⟨K, hK⟩ : ∃ K, t.depth = K := ⟨_, rfl⟩
  rw [hK] at hT ⊢
  have hs' := hs
  obtain ⟨hu256, hue64, hul64, hη, hηe⟩ := hs
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hul := F.hul
  have hηp := F.hη
  have hηep := F.hηe
  have hηe' : F.ηe ≤ 1 / 100 := by
    have : (1:ℝ) / 2 ^ 100 ≤ 1 / 100 := by norm_num
    linarith
  have hη'' : F.η ≤ 1 / 100 := by
    have : (1:ℝ) / 2 ^ 100 ≤ 1 / 100 := by norm_num
    linarith
  have hD : (0 : ℝ) ≤ D := Nat.cast_nonneg _
  have hK0 : (0 : ℝ) ≤ K := Nat.cast_nonneg _
  have hKn : (K : ℝ) + 1 ≤ (t.flat.length : ℝ) := by
    have := depth_len t hleaves
    rw [hK] at this
    exact_mod_cast this
  have hη'0 : 0 ≤ F.η + F.ηe := by linarith
  have hKη : ((K : ℝ) + 1) * (F.η + F.ηe) ≤ F.u :=
    (mul_le_mul_of_nonneg_right hKn hη'0).trans hnη
  have hT1 : ((K : ℝ) + D + 1) * (F.u + F.ue) ≤ 1 / 16 := by
    have : ((K : ℝ) + D + 1) * (F.u + F.ue) ≤ ((K : ℝ) + D + 1) * (F.u + F.ue + F.ul) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    linarith
  have hP00 : (1 + F.u) ^ D * (1 + F.ue) * ((1 + F.u) ^ 2 * (1 + F.ue)) ^ 0 ≤ 6 / 5 :=
    num_P F.u F.ue hu hue hu256 hue64 D 0 (by
      simp only [Nat.cast_zero, add_zero]
      have : (D : ℝ) * (F.u + F.ue) ≤ ((K : ℝ) + D + 1) * (F.u + F.ue) :=
        mul_le_mul_of_nonneg_right (by linarith) (by linarith)
      linarith)
  have hpowD : (1 + F.u) ^ D ≤ 6 / 5 := by
    have h1 : 1 ≤ 1 + F.ue := by linarith
    have h2 : 0 ≤ (1 + F.u) ^ D := by positivity
    simp only [pow_zero, mul_one] at hP00
    nlinarith
  have inv := tree_inv F hs' D K
    (fun j => (1 + F.u) ^ D * (1 + F.ue) * ((1 + F.u) ^ 2 * (1 + F.ue)) ^ j)
    (fun j => 2 * ((j : ℝ) + 1) * (F.η + F.ηe))
    (fun j => (1 - F.u) ^ D * (1 - F.ue) * ((1 - F.u) ^ 2 * (1 - F.ue)) ^ j)
    (fun j => (1 - F.u) ^ D * (1 - F.ue - F.ηe) * ((1 - F.u) ^ 2 * (1 - F.ue - F.ηe)) ^ j
      - 5 / 2 * (j : ℝ) * F.η)
    (by simp)
    (by
      simp only [Nat.cast_zero, zero_add, mul_one]
      nlinarith [mul_le_mul_of_nonneg_right hpowD hηep])
    (by simp)
    (by simp)
    (fun j hj => seq_nn F hs' D K hT1 hKη j hj)
    (fun j hj => seq_step F hs' D K hT1 hKη j hj)
    t hleaves K hK.le le_rfl st hst
  obtain ⟨hm, hU, hL, hA⟩ := inv
  have hX := flat_ne t hleaves
  obtain ⟨hc1, hc2, hS1, hsumexp⟩ := sums_cmp t.flat lo hi F.u hu hX hbd
  beta_reduce at hU hL hA
  have hlohi : lo ≤ hi := by
    have := hbd _ (lmax_mem hX)
    linarith [this.1, this.2]
  have hn0 : (0 : ℝ) ≤ (t.flat.length : ℝ) := Nat.cast_nonneg _
  have hn1 : (1 : ℝ) ≤ (t.flat.length : ℝ) := by linarith
  generalize (t.flat.length : ℝ) = n at *
  generalize (t.flat.map fun x => Real.exp (x - lmax t.flat)).sum = S at *
  generalize (t.flat.map fun x => Real.exp (-((1 - F.u) * (lmax t.flat - x)))).sum = Sp at *
  generalize (t.flat.map fun x => Real.exp (-((1 + F.u) * (lmax t.flat - x)))).sum = Sm at *
  have hK1 : (0 : ℝ) ≤ (K : ℝ) + 1 := by linarith
  -- numerical facts
  have hEu : ((K : ℝ) + 1) * F.u ≤ 1 / 16 := by
    linarith [mul_nonneg hD hu, mul_nonneg hK0 hue, mul_nonneg hD hue]
  have hE1 : ((D : ℝ) + 2 * K) * F.u + ((K : ℝ) + 1) * F.ue ≤ 1 / 8 := by
    linarith [mul_nonneg hD hu, mul_nonneg hK0 hue, mul_nonneg hD hue, mul_nonneg hK0 hu]
  have hE2 : ((K : ℝ) + 1) * F.ηe ≤ F.u := by
    linarith [mul_nonneg hK1 hηp]
  have hp : ((D : ℝ) + 2 * K) * F.u ≤ 1 := by
    linarith [mul_nonneg hK1 hue]
  set Q := (1 - F.u) ^ D * (1 - F.ue - F.ηe) * ((1 - F.u) ^ 2 * (1 - F.ue - F.ηe)) ^ K with hQdef
  have hQ1 : 1 - ((D : ℝ) + 2 * K) * F.u - ((K : ℝ) + 1) * (F.ue + F.ηe) ≤ Q := by
    have := Q_lower F.u (F.ue + F.ηe) hu (by linarith) (by linarith) (by linarith) D K hp
    rw [show (1:ℝ) - (F.ue + F.ηe) = 1 - F.ue - F.ηe by ring] at this
    exact this
  have hQ86 : 86 / 100 ≤ Q := by linarith
  set z := 5 / 2 * ((K : ℝ) + 1) * n * (F.η + F.ηe) with hzdef
  have hz0 : 0 ≤ z := by positivity
  have hz : z ≤ 5 / 32 := by
    have : ((K : ℝ) + 1) * (n * (F.η + F.ηe)) ≤ ((K : ℝ) + 1) * F.u :=
      mul_le_mul_of_nonneg_left hnη hK1
    rw [hzdef]; linarith
  have hQz : 7 / 10 ≤ Q - z := by linarith
  have hQpos : 0 < Q := by linarith
  have hnBK : 0 ≤ n * (2 * ((K : ℝ) + 1) * (F.η + F.ηe)) := by positivity
  have hnBz : n * (2 * ((K : ℝ) + 1) * (F.η + F.ηe)) ≤ z := by
    rw [hzdef]; linarith [mul_nonneg (mul_nonneg hK1 hn0) hη'0]
  have haQ : Q ≤ (1 - F.u) ^ D * (1 - F.ue) * ((1 - F.u) ^ 2 * (1 - F.ue)) ^ K :=
    Q_le_a F.u F.ue F.ηe hu hu1.le hηep (by linarith) hue D K
  have haK0 : 0 ≤ (1 - F.u) ^ D * (1 - F.ue) * ((1 - F.u) ^ 2 * (1 - F.ue)) ^ K := by linarith
  have hKηz : 5 / 2 * (K : ℝ) * F.η ≤ z := by
    have h1 : ((K : ℝ) + 1) * 1 * (F.η + F.ηe) ≤ ((K : ℝ) + 1) * n * (F.η + F.ηe) := by
      linarith [mul_nonneg (mul_nonneg hK1 hη'0) (by linarith : (0:ℝ) ≤ n - 1)]
    rw [hzdef]; linarith [mul_nonneg hK0 hηep]
  have hAK : Q - z ≤ Q - 5 / 2 * (K : ℝ) * F.η := by linarith
  have hd0 : 0 < st.d := by linarith
  have hS0 : 0 < S := by linarith
  have hc0 : 0 ≤ F.u * (hi - lo) := mul_nonneg hu (by linarith)
  have hlow := lower_core (F.u * (hi - lo)) st.d S Sm _ _ (n * (2 * ((K : ℝ) + 1) * (F.η + F.ηe)))
    (Q - z) hS1 hc2 hL hA haK0 hnBK (by linarith) hAK (by linarith)
  have hup := upper_core (F.u * (hi - lo)) st.d S _ (n * (2 * ((K : ℝ) + 1) * (F.η + F.ηe))) Sp hS1 hc1
    (logP F.u F.ue hu hue D K).1 hc0 hnBK hU hd0
  have hlogP := (logP F.u F.ue hu hue D K).2
  have hlogQ := logQ F.u (F.ue + F.ηe) hu (by linarith) (by linarith) (by linarith) D K
  rw [show (1:ℝ) - (F.ue + F.ηe) = 1 - F.ue - F.ηe by ring] at hlogQ
  have hlogW := log_sub_ge Q z hQpos hz0 (by linarith)
  have hzdiv : z / (Q - z) ≤ 10 / 7 * z := by
    rw [div_le_iff₀ (by linarith)]
    have := mul_le_mul_of_nonneg_left hQz hz0
    linarith
  unfold znorm
  rw [hm, hsumexp, log_connect _ _ _ hd0 hS0, abs_le]
  have hDu := mul_nonneg hD hu
  have hKu := mul_nonneg hK0 hu
  have hKue := mul_nonneg hK0 hue
  have hnK := mul_nonneg (mul_nonneg hK0 hn0) hη'0
  have hnη0 := mul_nonneg hn0 hη'0
  have hΔ : F.u * (hi - lo) ≤ 2 * F.u * (hi - lo) := by linarith
  constructor
  · linarith
  · linarith

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ), ∀ (lo hi : ℝ) (X : List ℝ),
    X ≠ [] → (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((2 : ℝ) * F.u * (hi - lo)) + (((5 : ℝ) * ((1 : ℝ) / 4)) * (((D : ℝ) * F.u) + ((2 : ℝ) * (t.depth : ℝ) * F.u) + F.u + (((t.depth : ℝ) + (1 : ℝ)) * F.ue))) + ((4 : ℝ) * (X.length : ℝ) * ((t.depth : ℝ) + (2 : ℝ)) * (F.η + F.ηe))) := by
  exact fp_bound_main

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.Candfbc0a9179c
