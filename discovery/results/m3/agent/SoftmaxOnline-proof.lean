import FPStatements

namespace Discovery.FP.SoftmaxOnline

open Discovery Discovery.Library.SoftmaxOnline

theorem exp_neg_le_one_sub (x : ℝ) (hx0 : 0 ≤ x) (hx : x ≤ 1/64) :
    Real.exp (-(51/50*x)) ≤ 1 - x := by
  have h := Real.add_one_le_exp (51/50*x)
  have h2 : Real.exp (-(51/50*x)) * (51/50*x + 1) ≤ 1 := by
    have := mul_le_mul_of_nonneg_left h (Real.exp_pos (-(51/50*x))).le
    rw [← Real.exp_add] at this
    simpa using this
  have h3 : (1 - x) * (51/50*x + 1) ≥ 1 := by nlinarith
  by_contra hc
  push Not at hc
  nlinarith [Real.exp_pos (-(51/50*x))]

theorem rel_exp (F : Discovery.Rounding) (R z q1 q2 : ℝ) (hz0 : z ≤ 0) (hzR : -z ≤ R)
    (h1 : F.add z q1) (h2 : F.exp q1 q2) :
    q2 ≤ Real.exp z * Real.exp (F.u*R) * (1+F.ue) + F.ηe ∧ q2 ≤ 1 + F.ue + F.ηe ∧
    Real.exp z * Real.exp (-(F.u*R)) * (1-F.ue) - F.ηe ≤ q2 ∧ 0 ≤ q2 ∧
    (z = 0 → 1 - F.ue - F.ηe ≤ q2) := by
  obtain ⟨δ, hδ, rfl⟩ := h1
  have hb := F.exp_bounds h2
  have hu := F.hu
  have hue := F.hue
  have hue1 := F.hue1
  have hδ' := abs_le.mp hδ
  have hq : z * (1+δ) = z + z*δ := by ring
  have hzd1 : z*δ ≤ F.u*R := by nlinarith [hδ'.1, hδ'.2]
  have hzd2 : -(F.u*R) ≤ z*δ := by nlinarith [hδ'.1, hδ'.2]
  have hexp : Real.exp (z*(1+δ)) = Real.exp z * Real.exp (z*δ) := by
    rw [hq, Real.exp_add]
  have hA : Real.exp (z*δ) ≤ Real.exp (F.u*R) := Real.exp_le_exp.mpr hzd1
  have hB : Real.exp (-(F.u*R)) ≤ Real.exp (z*δ) := Real.exp_le_exp.mpr hzd2
  have hu1 := F.hu1
  have hq0 : z*(1+δ) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hz0 (by linarith [hδ'.1])
  have hq1 : Real.exp (z*(1+δ)) ≤ 1 := Real.exp_le_one_iff.mpr hq0
  have hez := Real.exp_pos z
  refine ⟨?_, ?_, ?_, hb.2.2, ?_⟩
  · have : Real.exp (z*(1+δ)) ≤ Real.exp z * Real.exp (F.u*R) := by
      rw [hexp]; exact mul_le_mul_of_nonneg_left hA hez.le
    nlinarith [hb.2.1]
  · nlinarith [hb.2.1]
  · have : Real.exp z * Real.exp (-(F.u*R)) ≤ Real.exp (z*(1+δ)) := by
      rw [hexp]; exact mul_le_mul_of_nonneg_left hB hez.le
    nlinarith [hb.1]
  · intro hz
    subst hz
    simp at hb
    simp
    linarith [hb.1]

theorem term_gen (F : Discovery.Rounding) (z q1 q2 : ℝ)
    (h1 : F.add z q1) (h2 : F.exp q1 q2) :
    0 ≤ q2 ∧ q2 ≤ Real.exp (2*|z|) * 2 + F.ηe := by
  obtain ⟨δ, hδ, rfl⟩ := h1
  have hb := F.exp_bounds h2
  have hu1 := F.hu1
  have hue1 := F.hue1
  have hδ' := abs_le.mp hδ
  have habs : |z*(1+δ)| ≤ 2*|z| := by
    rw [abs_mul]
    have : |1+δ| ≤ 2 := by rw [abs_le]; constructor <;> linarith [hδ'.1, hδ'.2]
    nlinarith [abs_nonneg z, abs_nonneg (1+δ)]
  have hq : z*(1+δ) ≤ 2*|z| := le_trans (le_abs_self _) habs
  have he : Real.exp (z*(1+δ)) ≤ Real.exp (2*|z|) := Real.exp_le_exp.mpr hq
  have hpos := Real.exp_pos (z*(1+δ))
  refine ⟨hb.2.2, ?_⟩
  nlinarith [hb.2.1, F.hue]

theorem fsum_term_bounds (F : Discovery.Rounding) (D : ℕ) (m : ℝ) (A : List ℝ) (w : ℝ)
    (hsum : Discovery.fsum F D (fun x y => ∃ q1 q2, F.add (x - m) q1 ∧ F.exp q1 q2 ∧ y = q2) A w)
    (lo hi : ℝ → ℝ) (hlo : ∀ x ∈ A, 0 ≤ lo x)
    (hterm : ∀ x ∈ A, ∀ y, (∃ q1 q2, F.add (x - m) q1 ∧ F.exp q1 q2 ∧ y = q2) →
       lo x ≤ y ∧ y ≤ hi x) :
    (1 - F.u) ^ D * (A.map lo).sum ≤ w ∧ w ≤ (1 + F.u) ^ D * (A.map hi).sum := by
  classical
  have := Discovery.fsum_bounds F D (fun x y => ∃ q1 q2, F.add (x - m) q1 ∧ F.exp q1 q2 ∧ y = q2)
    (fun x => if x ∈ A then lo x else 0)
    (fun x => if x ∈ A then hi x else Real.exp (2*|x - m|) * 2 + F.ηe)
    (by intro x; by_cases hx : x ∈ A <;> simp [hx, hlo]) 
    (by
      intro x y hxy
      by_cases hx : x ∈ A
      · simp only [hx, if_true]; exact hterm x hx y hxy
      · simp only [hx, if_false]
        obtain ⟨q1, q2, h1, h2, rfl⟩ := hxy
        have := term_gen F _ _ _ h1 h2
        exact ⟨this.1, this.2⟩) A w hsum
  have e1 : A.map (fun x => if x ∈ A then lo x else 0) = A.map lo :=
    List.map_congr_left (fun x hx => by simp [hx])
  have e2 : A.map (fun x => if x ∈ A then hi x else Real.exp (2*|x - m|) * 2 + F.ηe) = A.map hi :=
    List.map_congr_left (fun x hx => by simp [hx])
  rw [e1, e2] at this
  exact this

theorem sum_affine (A : List ℝ) (f : ℝ → ℝ) (a b : ℝ) :
    (A.map fun x => a * f x + b).sum = a * (A.map f).sum + A.length * b := by
  induction A with
  | nil => simp
  | cons x l ih =>
    simp only [List.map_cons, List.sum_cons, ih, List.length_cons]
    push_cast; ring

theorem S_ge_one (A : List ℝ) (hA : A ≠ []) :
    1 ≤ (A.map fun x => Real.exp (x - lmax A)).sum := by
  have hm := lmax_mem hA
  have := List.single_le_sum (l := A.map fun x => Real.exp (x - lmax A))
    (by intro x hx; obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx; exact (Real.exp_pos _).le)
    (Real.exp (lmax A - lmax A)) (List.mem_map_of_mem (f := fun x => Real.exp (x - lmax A)) hm)
  simpa using this

theorem S_le_n (A : List ℝ) :
    (A.map fun x => Real.exp (x - lmax A)).sum ≤ A.length := by
  by_cases hA : A = []
  · subst hA; simp
  have h := List.sum_le_length_nsmul (l := A.map fun x => Real.exp (x - lmax A)) 1
    (by
      intro x hx
      obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
      have := le_lmax hy
      exact Real.exp_le_one_iff.mpr (by linarith))
  simpa using h

set_option maxHeartbeats 1000000 in
theorem leaf_inv (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (A : List ℝ) (hA : A ≠ [])
    (hr : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi) (hs : small F) (st : State_a)
    (h : fblock_a F D A st) :
    st.m = lmax A ∧
    1 - (D*F.u + F.ue + F.ηe) ≤ st.d ∧
    st.d ≤ A.length * Real.exp (D*F.u + F.ue + F.ηe) ∧
    st.d ≤ Real.exp (F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u)) *
      ((A.map fun x => Real.exp (x - lmax A)).sum + A.length * F.ηe) ∧
    Real.exp (-(F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u))) *
      (A.map fun x => Real.exp (x - lmax A)).sum - A.length * F.ηe ≤ st.d := by
  obtain ⟨v_m, v_d, r1, r2, hr1, hvm, hsum, hvd, hst⟩ := h
  subst hst
  have hm := lmax_mem hA
  have hmA : v_m = lmax A := by rw [hvm, hr1]
  subst hvd
  refine ⟨hmA, ?_⟩
  obtain ⟨hs1, hs2, hs3, hs4, hs5⟩ := hs
  have hu := F.hu
  have hue := F.hue
  have hη := F.hη
  have hηe := F.hηe
  have hu1 := F.hu1
  have hue1 := F.hue1
  have hηe' : F.ηe ≤ 1/64 := by
    have : (1:ℝ)/2^100 ≤ 1/64 := by norm_num
    linarith
  have hlh : lo ≤ hi := le_trans (hr _ hm).1 (hr _ hm).2
  set m := lmax A with hmdef
  have hR : 0 ≤ hi - lo := by linarith
  -- per-element facts
  have key : ∀ x ∈ A, ∀ y, (∃ q1 q2, F.add (x - v_m) q1 ∧ F.exp q1 q2 ∧ y = q2) →
      (y ≤ Real.exp (x - m) * Real.exp (F.u*(hi-lo)) * (1+F.ue) + F.ηe ∧ y ≤ 1 + F.ue + F.ηe ∧
      Real.exp (x - m) * Real.exp (-(F.u*(hi-lo))) * (1-F.ue) - F.ηe ≤ y ∧ 0 ≤ y ∧
      (x - m = 0 → 1 - F.ue - F.ηe ≤ y)) := by
    intro x hx y ⟨q1, q2, h1, h2, hy⟩
    subst hy
    rw [hmA] at h1
    have hxm : x ≤ m := le_lmax hx
    have := rel_exp F (hi - lo) (x - m) q1 y (by linarith) (by linarith [(hr x hx).1, (hr m hm).2]) h1 h2
    exact this
  have hn : (0:ℝ) ≤ A.length := Nat.cast_nonneg _
  have hD : (0:ℝ) ≤ D := Nat.cast_nonneg _
  have hc0 : 0 ≤ 1 - F.ue - F.ηe := by linarith
  -- call 1
  have c1 := fsum_term_bounds F D v_m A v_d hsum
    (fun x => if x = m then 1 - F.ue - F.ηe else 0) (fun _ => 1 + F.ue + F.ηe)
    (by intro x hx; by_cases h : x = m <;> simp [h, hc0])
    (by
      intro x hx y hy
      obtain ⟨k1, k2, k3, k4, k5⟩ := key x hx y hy
      refine ⟨?_, k2⟩
      by_cases h : x = m
      · simp only [h, if_true]; exact k5 (by simp [h])
      · simp only [h, if_false]; exact k4)
  have hsl : 1 - F.ue - F.ηe ≤ (A.map (fun x => if x = m then 1 - F.ue - F.ηe else 0)).sum := by
    have := List.single_le_sum (l := A.map (fun x => if x = m then 1 - F.ue - F.ηe else 0))
      (by intro x hx; obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx
          by_cases h : y = m <;> simp [h, hc0])
      (1 - F.ue - F.ηe)
      (List.mem_map.mpr ⟨m, hm, by simp⟩)
    exact this
  have hsh : (A.map (fun _ => 1 + F.ue + F.ηe)).sum = A.length * (1 + F.ue + F.ηe) := by
    have := sum_affine A (fun _ => 0) 0 (1 + F.ue + F.ηe)
    simpa using this
  rw [hsh] at c1
  have hpw : 1 - D * F.u ≤ (1 - F.u)^D := by
    have := one_add_mul_le_pow (a := -F.u) (by linarith) D
    simpa [sub_eq_add_neg] using this
  have hpw1 : 0 ≤ (1 - F.u)^D := pow_nonneg (by linarith) D
  have hpw2 : (1 - F.u)^D ≤ 1 := pow_le_one₀ (by linarith) (by linarith)
  have hpwu : (1 + F.u)^D ≤ Real.exp (D * F.u) := by
    rw [Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp F.u]) D
  -- call 2
  have c2 := fsum_term_bounds F D v_m A v_d hsum
    (fun x => max 0 (Real.exp (x - m) * Real.exp (-(F.u*(hi-lo))) * (1-F.ue) - F.ηe))
    (fun x => Real.exp (x - m) * Real.exp (F.u*(hi-lo)) * (1+F.ue) + F.ηe)
    (fun x _ => le_max_left _ _)
    (by
      intro x hx y hy
      obtain ⟨k1, k2, k3, k4, k5⟩ := key x hx y hy
      exact ⟨max_le k4 k3, k1⟩)
  set S := (A.map fun x => Real.exp (x - m)).sum with hS
  have hlo2 : Real.exp (-(F.u*(hi-lo))) * (1-F.ue) * S - A.length * F.ηe ≤
      (A.map (fun x => max 0 (Real.exp (x - m) * Real.exp (-(F.u*(hi-lo))) * (1-F.ue) - F.ηe))).sum := by
    have e := sum_affine A (fun x => Real.exp (x - m)) (Real.exp (-(F.u*(hi-lo))) * (1-F.ue)) (-F.ηe)
    have : (A.map fun x => Real.exp (x - m) * Real.exp (-(F.u*(hi-lo))) * (1-F.ue) - F.ηe).sum ≤
        (A.map (fun x => max 0 (Real.exp (x - m) * Real.exp (-(F.u*(hi-lo))) * (1-F.ue) - F.ηe))).sum := by
      apply List.sum_le_sum
      intro x _
      exact le_max_right _ _
    have e2 : (A.map fun x => Real.exp (x - m) * Real.exp (-(F.u*(hi-lo))) * (1-F.ue) - F.ηe) =
        A.map fun x => Real.exp (-(F.u*(hi-lo))) * (1-F.ue) * Real.exp (x - m) + -F.ηe :=
      List.map_congr_left (fun x _ => by ring)
    rw [e2, e] at this
    linarith
  have hhi2 : (A.map (fun x => Real.exp (x - m) * Real.exp (F.u*(hi-lo)) * (1+F.ue) + F.ηe)).sum =
      Real.exp (F.u*(hi-lo)) * (1+F.ue) * S + A.length * F.ηe := by
    have e := sum_affine A (fun x => Real.exp (x - m)) (Real.exp (F.u*(hi-lo)) * (1+F.ue)) (F.ηe)
    have e2 : (A.map fun x => Real.exp (x - m) * Real.exp (F.u*(hi-lo)) * (1+F.ue) + F.ηe) =
        A.map fun x => Real.exp (F.u*(hi-lo)) * (1+F.ue) * Real.exp (x - m) + F.ηe :=
      List.map_congr_left (fun x _ => by ring)
    rw [e2, e]
  rw [hhi2] at c2
  have hSn : 1 ≤ S := S_ge_one A hA
  dsimp only
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h1 : (1 - D*F.u) * (1 - F.ue - F.ηe) ≤ (1 - F.u)^D * (1 - F.ue - F.ηe) :=
      mul_le_mul_of_nonneg_right hpw hc0
    have h2 : (1 - F.u)^D * (1 - F.ue - F.ηe) ≤ (1 - F.u)^D *
        (A.map (fun x => if x = m then 1 - F.ue - F.ηe else 0)).sum :=
      mul_le_mul_of_nonneg_left hsl hpw1
    have e : (1 - D*F.u) * (1 - F.ue - F.ηe) = (1 - F.ue - F.ηe) - D*F.u + D*F.u*(F.ue+F.ηe) := by ring
    have : 0 ≤ D*F.u*(F.ue+F.ηe) := by positivity
    linarith [c1.1]
  · have h1 : Real.exp (F.ue + F.ηe) ≥ 1 + F.ue + F.ηe := by
      have := Real.add_one_le_exp (F.ue + F.ηe); linarith
    have : (1 + F.u)^D * (A.length * (1 + F.ue + F.ηe)) ≤
        Real.exp (D*F.u) * (A.length * Real.exp (F.ue + F.ηe)) := by
      apply mul_le_mul hpwu _ (by positivity) (Real.exp_pos _).le
      exact mul_le_mul_of_nonneg_left h1 hn
    have e : Real.exp (D*F.u + F.ue + F.ηe) = Real.exp (D*F.u) * Real.exp (F.ue + F.ηe) := by
      rw [← Real.exp_add]; ring_nf
    have e3 : Real.exp (D*F.u) * (A.length * Real.exp (F.ue + F.ηe)) =
        A.length * (Real.exp (D*F.u) * Real.exp (F.ue + F.ηe)) := by ring
    rw [e]; linarith [c1.2]
  · -- upper, relative
    have hρ : 0 ≤ F.u * (hi - lo) := mul_nonneg hu hR
    have ea : Real.exp (F.u*(hi-lo)) * (1+F.ue) ≤ Real.exp (F.u*(hi-lo) + F.ue) := by
      rw [Real.exp_add]
      exact mul_le_mul_of_nonneg_left (by linarith [Real.add_one_le_exp F.ue]) (Real.exp_pos _).le
    set a0 := F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u) with ha0
    have hDu : 0 ≤ D * F.u := mul_nonneg hD hu
    have h1 : Real.exp (D*F.u + (F.u*(hi-lo) + F.ue)) ≤ Real.exp a0 :=
      Real.exp_le_exp.mpr (by rw [ha0]; linarith)
    have h2 : Real.exp (D*F.u) ≤ Real.exp a0 := Real.exp_le_exp.mpr (by rw [ha0]; linarith)
    have hpos : 0 ≤ Real.exp (F.u*(hi-lo)) * (1+F.ue) * S := by positivity
    have step1 : (1 + F.u)^D * (Real.exp (F.u*(hi-lo)) * (1+F.ue) * S + A.length * F.ηe) ≤
        Real.exp (D*F.u) * (Real.exp (F.u*(hi-lo) + F.ue) * S + A.length * F.ηe) := by
      apply mul_le_mul hpwu _ (by positivity) (Real.exp_pos _).le
      have := mul_le_mul_of_nonneg_right ea (by linarith : (0:ℝ) ≤ S)
      linarith
    have step2 : Real.exp (D*F.u) * (Real.exp (F.u*(hi-lo) + F.ue) * S + A.length * F.ηe) =
        Real.exp (D*F.u + (F.u*(hi-lo) + F.ue)) * S + Real.exp (D*F.u) * (A.length * F.ηe) := by
      simp only [Real.exp_add]; ring
    have step3 : Real.exp (D*F.u + (F.u*(hi-lo) + F.ue)) * S + Real.exp (D*F.u) * (A.length * F.ηe) ≤
        Real.exp a0 * (S + A.length * F.ηe) := by
      have := mul_le_mul_of_nonneg_right h1 (by linarith : (0:ℝ) ≤ S)
      have h3 := mul_le_mul_of_nonneg_right h2 (by positivity : (0:ℝ) ≤ A.length * F.ηe)
      rw [mul_add]; linarith
    linarith [c2.2]
  · -- lower, relative
    set a0 := F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u) with ha0
    have hq := exp_neg_le_one_sub F.ue hue (by linarith)
    have hp := exp_neg_le_one_sub F.u hu (by linarith)
    have hpD : Real.exp (-(51/50*(D*F.u))) ≤ (1 - F.u)^D := by
      have : Real.exp (-(51/50*(D*F.u))) = (Real.exp (-(51/50*F.u)))^D := by
        rw [← Real.exp_nat_mul]; congr 1; ring
      rw [this]
      exact pow_le_pow_left₀ (Real.exp_pos _).le hp D
    have e : Real.exp (-a0) = Real.exp (-(F.u*(hi-lo))) * Real.exp (-(51/50*F.ue)) *
        Real.exp (-(51/50*(D*F.u))) := by
      rw [← Real.exp_add, ← Real.exp_add]; congr 1; rw [ha0]; ring
    have hc1 : 0 ≤ Real.exp (-(F.u*(hi-lo))) * (1-F.ue) * S := by
      have : 0 ≤ 1 - F.ue := by linarith
      positivity
    have h1 : Real.exp (-a0) * S ≤ (1 - F.u)^D * (Real.exp (-(F.u*(hi-lo))) * (1-F.ue) * S) := by
      rw [e]
      have h5 : Real.exp (-(F.u*(hi-lo))) * Real.exp (-(51/50*F.ue)) * Real.exp (-(51/50*(D*F.u))) ≤
          (1 - F.u)^D * (Real.exp (-(F.u*(hi-lo))) * (1-F.ue)) := by
        have := mul_le_mul hpD (mul_le_mul_of_nonneg_left hq (Real.exp_pos (-(F.u*(hi-lo)))).le)
          (by positivity) hpw1
        have e5 : Real.exp (-(F.u*(hi-lo))) * Real.exp (-(51/50*F.ue)) * Real.exp (-(51/50*(D*F.u))) =
            Real.exp (-(51/50*(D*F.u))) * (Real.exp (-(F.u*(hi-lo))) * Real.exp (-(51/50*F.ue))) := by ring
        rw [e5]; exact this
      have := mul_le_mul_of_nonneg_right h5 (by linarith : (0:ℝ) ≤ S)
      linarith
    have h2 : (1 - F.u)^D * (Real.exp (-(F.u*(hi-lo))) * (1-F.ue) * S - A.length * F.ηe) ≤
        (1 - F.u)^D * (A.map (fun x => max 0 (Real.exp (x - m) * Real.exp (-(F.u*(hi-lo))) * (1-F.ue) - F.ηe))).sum :=
      mul_le_mul_of_nonneg_left hlo2 hpw1
    have h3 : (1 - F.u)^D * (A.length * F.ηe) ≤ A.length * F.ηe := by
      have := mul_le_mul_of_nonneg_right hpw2 (by positivity : (0:ℝ) ≤ A.length * F.ηe)
      linarith
    rw [mul_sub] at h2
    linarith [c2.1]


set_option maxHeartbeats 1000000 in
theorem side_bounds (F : Discovery.Rounding) (R z q1 q2 r d SS nn α β θ ζ : ℝ)
    (hz0 : z ≤ 0) (hzR : -z ≤ R)
    (h1 : F.add z q1) (h2 : F.exp q1 q2) (h3 : F.mul (d*q2) r)
    (hd0 : 0 ≤ d) (hSS : 0 ≤ SS) (hnn : 0 ≤ nn) (hθ : 0 ≤ θ) (hζ : ζ ≤ 1)
    (hsm : F.ue + F.ηe ≤ 1/2)
    (hc : d ≤ nn * Real.exp β) (hd : d ≤ Real.exp α * (SS + nn*θ))
    (he : Real.exp (-α) * SS - nn*θ ≤ d) :
    0 ≤ d*q2 ∧
    r ≤ (1+F.u)*(Real.exp (α + F.u*R)*(1+F.ue)*(Real.exp z*(SS + nn*θ)) + nn*Real.exp β*F.ηe) + F.η ∧
    (1-F.u)*(Real.exp (-(α + F.u*R))*(1-F.ue)*(Real.exp z*SS) - nn*(θ + Real.exp β*F.ηe)) - F.η ≤ r ∧
    r ≤ (1+F.u)*(nn*Real.exp β*(1+F.ue+F.ηe)) + F.η ∧
    (z = 0 → 1 - ζ ≤ d → (1-F.u)*((1-ζ)*(1-F.ue-F.ηe)) - F.η ≤ r) ∧ -F.η ≤ r := by
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hue1 := F.hue1
  have hη := F.hη
  have hηe := F.hηe
  obtain ⟨hq1, hq2, hq3, hq4, hq5⟩ := rel_exp F R z q1 q2 hz0 hzR h1 h2
  have hx0 : 0 ≤ d*q2 := mul_nonneg hd0 hq4
  obtain ⟨δ, e, hδ, he', hr, -⟩ := h3
  have hδ' := abs_le.mp hδ
  have he'' := abs_le.mp he'
  have hrU : r ≤ (1+F.u)*(d*q2) + F.η := by nlinarith [mul_nonneg hx0 (sub_nonneg.mpr hδ'.2)]
  have hrL : (1-F.u)*(d*q2) - F.η ≤ r := by nlinarith [mul_nonneg hx0 (by linarith [hδ'.1] : 0 ≤ δ + F.u)]
  have ha0 : 0 < Real.exp z := Real.exp_pos z
  have ha1 : Real.exp z ≤ 1 := Real.exp_le_one_iff.mpr hz0
  have hα1 : 0 < Real.exp α := Real.exp_pos α
  have hβ : 0 < Real.exp β := Real.exp_pos β
  have hρ1 : 0 < Real.exp (F.u*R) := Real.exp_pos _
  have hρ2 : 0 < Real.exp (-(F.u*R)) := Real.exp_pos _
  have hρ3 : Real.exp (-(F.u*R)) ≤ 1 := by
    apply Real.exp_le_one_iff.mpr
    have : 0 ≤ F.u * R := mul_nonneg hu (by linarith)
    linarith
  -- upper bound of x
  have hxU : d*q2 ≤ Real.exp (α + F.u*R)*(1+F.ue)*(Real.exp z*(SS + nn*θ)) + nn*Real.exp β*F.ηe := by
    have s1 : d*q2 ≤ d*(Real.exp z * Real.exp (F.u*R) * (1+F.ue) + F.ηe) :=
      mul_le_mul_of_nonneg_left hq1 hd0
    have s2 : d*(Real.exp z * Real.exp (F.u*R) * (1+F.ue)) ≤
        (Real.exp α * (SS + nn*θ)) * (Real.exp z * Real.exp (F.u*R) * (1+F.ue)) :=
      mul_le_mul_of_nonneg_right hd (by positivity)
    have s3 : d*F.ηe ≤ nn * Real.exp β * F.ηe := mul_le_mul_of_nonneg_right hc hηe
    have e1 : Real.exp (α + F.u*R) = Real.exp α * Real.exp (F.u*R) := Real.exp_add _ _
    rw [e1]
    nlinarith
  have hw0 : 0 ≤ Real.exp z * Real.exp (-(F.u*R)) * (1-F.ue) := by
    have : 0 ≤ 1 - F.ue := by linarith
    positivity
  have hw1 : Real.exp z * Real.exp (-(F.u*R)) * (1-F.ue) ≤ 1 := by
    have : Real.exp z * Real.exp (-(F.u*R)) ≤ 1 := by nlinarith
    nlinarith
  have hxL : Real.exp (-(α + F.u*R))*(1-F.ue)*(Real.exp z*SS) - nn*(θ + Real.exp β*F.ηe) ≤ d*q2 := by
    have s1 : d*(Real.exp z * Real.exp (-(F.u*R)) * (1-F.ue) - F.ηe) ≤ d*q2 :=
      mul_le_mul_of_nonneg_left hq3 hd0
    have s2 : (Real.exp (-α) * SS - nn*θ) * (Real.exp z * Real.exp (-(F.u*R)) * (1-F.ue)) ≤
        d * (Real.exp z * Real.exp (-(F.u*R)) * (1-F.ue)) :=
      mul_le_mul_of_nonneg_right he hw0
    have s3 : d*F.ηe ≤ nn * Real.exp β * F.ηe := mul_le_mul_of_nonneg_right hc hηe
    have s4 : nn*θ*(Real.exp z * Real.exp (-(F.u*R)) * (1-F.ue)) ≤ nn*θ :=
      by have := mul_le_mul_of_nonneg_left hw1 (mul_nonneg hnn hθ); linarith
    have e1 : Real.exp (-(α + F.u*R)) = Real.exp (-α) * Real.exp (-(F.u*R)) := by
      rw [← Real.exp_add]; congr 1; ring
    rw [e1]
    nlinarith
  have hxU2 : d*q2 ≤ nn*Real.exp β*(1+F.ue+F.ηe) := by
    have s1 : d*q2 ≤ d*(1+F.ue+F.ηe) := mul_le_mul_of_nonneg_left hq2 hd0
    have s2 : d*(1+F.ue+F.ηe) ≤ nn*Real.exp β*(1+F.ue+F.ηe) :=
      mul_le_mul_of_nonneg_right hc (by linarith)
    linarith
  refine ⟨hx0, ?_, ?_, ?_, ?_, ?_⟩
  · have := mul_le_mul_of_nonneg_left hxU (by linarith : (0:ℝ) ≤ 1 + F.u)
    linarith
  · have := mul_le_mul_of_nonneg_left hxL (by linarith : (0:ℝ) ≤ 1 - F.u)
    linarith
  · have := mul_le_mul_of_nonneg_left hxU2 (by linarith : (0:ℝ) ≤ 1 + F.u)
    linarith
  · intro hz hdz
    subst hz
    have hq5' := hq5 rfl
    have hz1 : 0 ≤ 1 - ζ := by linarith
    have hc0 : 0 ≤ 1 - F.ue - F.ηe := by linarith
    have : (1-ζ)*(1-F.ue-F.ηe) ≤ d*q2 := mul_le_mul hdz hq5' hc0 hd0
    have := mul_le_mul_of_nonneg_left this (by linarith : (0:ℝ) ≤ 1 - F.u)
    linarith
  · nlinarith [mul_nonneg hx0 (by linarith [hδ'.1] : 0 ≤ 1 + δ)]


theorem num1 (u x ζ : ℝ) (hu : 0 ≤ u) (hu1 : u ≤ 1/2) (hx : 0 ≤ x) (hx1 : x ≤ 1/2)
    (hζ : 0 ≤ ζ) (hζ1 : ζ ≤ 1/2) :
    1 - ζ - 2*u - x ≤ (1-u)^2 * ((1-ζ)*(1-x)) := by
  have h1 : 1 - 2*u ≤ (1-u)^2 := by nlinarith
  have h2 : 0 ≤ (1-ζ)*(1-x) := by nlinarith
  have h3 : (1-2*u)*((1-ζ)*(1-x)) ≤ (1-u)^2 * ((1-ζ)*(1-x)) := mul_le_mul_of_nonneg_right h1 h2
  nlinarith [mul_nonneg hu hζ, mul_nonneg hx hζ, mul_nonneg hu hx, mul_nonneg (mul_nonneg hu hx) hζ]

theorem num2 (u ue : ℝ) (hu : 0 ≤ u) (hu1 : u ≤ 1/256) (hue : 0 ≤ ue) (hue1 : ue ≤ 1/64) :
    (1+u)^2*(1+ue) ≤ Real.exp (2*(51/50)*u + 51/50*ue) := by
  have h1 : 1 + u ≤ Real.exp u := by linarith [Real.add_one_le_exp u]
  have h2 : 1 + ue ≤ Real.exp ue := by linarith [Real.add_one_le_exp ue]
  have h3 : (1+u)^2 ≤ Real.exp u ^ 2 := pow_le_pow_left₀ (by linarith) h1 2
  calc (1+u)^2*(1+ue) ≤ Real.exp u ^ 2 * Real.exp ue :=
        mul_le_mul h3 h2 (by linarith) (by positivity)
    _ = Real.exp (2*u + ue) := by rw [Real.exp_add, ← Real.exp_nat_mul]; norm_num
    _ ≤ _ := Real.exp_le_exp.mpr (by linarith)

theorem num3 (u ue : ℝ) (hu : 0 ≤ u) (hu1 : u ≤ 1/256) (hue : 0 ≤ ue) (hue1 : ue ≤ 1/64) :
    Real.exp (-(2*(51/50)*u + 51/50*ue)) ≤ (1-u)^2*(1-ue) := by
  have h1 := exp_neg_le_one_sub u hu (by linarith)
  have h2 := exp_neg_le_one_sub ue hue hue1
  have e : Real.exp (-(2*(51/50)*u + 51/50*ue)) =
      Real.exp (-(51/50*u))^2 * Real.exp (-(51/50*ue)) := by
    rw [← Real.exp_nat_mul, ← Real.exp_add]; congr 1; push_cast; ring
  rw [e]
  have h3 : Real.exp (-(51/50*u))^2 ≤ (1-u)^2 := pow_le_pow_left₀ (Real.exp_pos _).le h1 2
  exact mul_le_mul h3 h2 (Real.exp_pos _).le (by positivity)

theorem comb_c (u ue η ηe β β' nL nR r3 r6 v_d : ℝ)
    (hu : 0 ≤ u) (hη : 0 ≤ η) (hηe : 0 ≤ ηe) (hue : 0 ≤ ue)
    (hnL : 1 ≤ nL) (hnR : 1 ≤ nR) (hβ0 : 0 ≤ β)
    (hβ' : β' = β + (2*u + ue + ηe + η))
    (cL : r3 ≤ (1+u)*(nL*Real.exp β*(1+ue+ηe)) + η)
    (cR : r6 ≤ (1+u)*(nR*Real.exp β*(1+ue+ηe)) + η)
    (hr7U : v_d ≤ (1+u)*(r3+r6)) : v_d ≤ (nL+nR) * Real.exp β' := by
  have heb : 1 ≤ Real.exp β := Real.one_le_exp hβ0
  have hn2 : 2 ≤ (nL+nR) * Real.exp β * (1+u) := by
    have h1 : nL + nR ≤ (nL+nR) * Real.exp β := by nlinarith
    have h2 : (nL+nR) * Real.exp β ≤ (nL+nR) * Real.exp β * (1+u) := by
      have : 0 ≤ (nL+nR) * Real.exp β * u := by positivity
      nlinarith
    linarith
  have hQ : (1+u)^2 * (1 + ue + ηe + η) ≤ Real.exp (2*u + ue + ηe + η) := by
    have h1 : 1 + u ≤ Real.exp u := by linarith [Real.add_one_le_exp u]
    have h2 : 1 + ue + ηe + η ≤ Real.exp (ue + ηe + η) := by
      linarith [Real.add_one_le_exp (ue + ηe + η)]
    have h3 : (1+u)^2 ≤ Real.exp u ^ 2 := pow_le_pow_left₀ (by linarith) h1 2
    calc (1+u)^2 * (1 + ue + ηe + η) ≤ Real.exp u ^ 2 * Real.exp (ue + ηe + η) :=
          mul_le_mul h3 h2 (by linarith) (by positivity)
      _ = Real.exp (2*u + ue + ηe + η) := by
          rw [← Real.exp_nat_mul, ← Real.exp_add]; congr 1; push_cast; ring
  have e : Real.exp β' = Real.exp β * Real.exp (2*u + ue + ηe + η) := by
    rw [hβ', Real.exp_add]
  have hsum : r3 + r6 ≤ (1+u)*((nL+nR) * Real.exp β * (1+ue+ηe)) + 2*η := by
    nlinarith
  have h1 : v_d ≤ (1+u)*((1+u)*((nL+nR) * Real.exp β * (1+ue+ηe)) + 2*η) :=
    le_trans hr7U (mul_le_mul_of_nonneg_left hsum (by linarith))
  have h2 : (1+u)*((1+u)*((nL+nR) * Real.exp β * (1+ue+ηe)) + 2*η) ≤
      (nL+nR) * Real.exp β * ((1+u)^2 * (1 + ue + ηe + η)) := by
    have : (1+u) * (2*η) ≤ ((nL+nR) * Real.exp β * (1+u)) * ((1+u) * η) := by
      have := mul_le_mul_of_nonneg_right hn2 (by positivity : (0:ℝ) ≤ (1+u) * η)
      nlinarith
    nlinarith
  have h3 : (nL+nR) * Real.exp β * ((1+u)^2 * (1 + ue + ηe + η)) ≤
      (nL+nR) * Real.exp β * Real.exp (2*u + ue + ηe + η) :=
    mul_le_mul_of_nonneg_left hQ (by positivity)
  rw [e]
  linarith


theorem comb_d (u ue η ηe R α β θ K α' θ' SL SR nL nR aL aR r3 r6 v_d : ℝ)
    (hu : 0 ≤ u) (hu1 : u ≤ 1/256) (hue : 0 ≤ ue) (hue1 : ue ≤ 1/64) (hη : 0 ≤ η) (hηe : 0 ≤ ηe)
    (hR0 : 0 ≤ R) (hα0 : 0 ≤ α) (hβ0 : 0 ≤ β) (hθ0 : 0 ≤ θ)
    (hnL : 1 ≤ nL) (hnR : 1 ≤ nR) (hSL : 0 ≤ SL) (hSR : 0 ≤ SR)
    (haL0 : 0 ≤ aL) (haL1 : aL ≤ 1) (haR0 : 0 ≤ aR) (haR1 : aR ≤ 1)
    (hK : (1+u)^2 * Real.exp β ≤ K)
    (hα' : α' = α + u*R + 2*(51/50)*u + 51/50*ue) (hθ' : θ' = θ + K*(ηe + η))
    (uL : r3 ≤ (1+u)*(Real.exp (α + u*R)*(1+ue)*(aL*(SL + nL*θ)) + nL*Real.exp β*ηe) + η)
    (uR : r6 ≤ (1+u)*(Real.exp (α + u*R)*(1+ue)*(aR*(SR + nR*θ)) + nR*Real.exp β*ηe) + η)
    (hr7U : v_d ≤ (1+u)*(r3+r6)) :
    v_d ≤ Real.exp α' * (aL*SL + aR*SR + (nL+nR)*θ') := by
  have heb : 1 ≤ Real.exp β := Real.one_le_exp hβ0
  have hT : 1 ≤ Real.exp (α + u*R) := Real.one_le_exp (by positivity)
  have hW1 : 1 ≤ Real.exp (2*(51/50)*u + 51/50*ue) := Real.one_le_exp (by positivity)
  have hW := num2 u ue hu hu1 hue hue1
  have e' : Real.exp α' = Real.exp (α + u*R) * Real.exp (2*(51/50)*u + 51/50*ue) := by
    rw [hα', ← Real.exp_add]; congr 1; ring
  rw [e']
  generalize Real.exp (α + u*R) = T at hT uL uR ⊢
  generalize Real.exp (2*(51/50)*u + 51/50*ue) = W at hW1 hW ⊢
  generalize Real.exp β = eb at heb hK uL uR
  have hK1 : eb ≤ K := by
    have h1 : 1 ≤ (1+u)^2 := by nlinarith
    have : eb ≤ (1+u)^2 * eb := le_mul_of_one_le_left (by linarith) h1
    linarith
  have hKb : 1 + u ≤ K := by
    have h1 : 1 + u ≤ (1+u)^2 := by nlinarith
    have : (1+u)^2 ≤ (1+u)^2 * eb := le_mul_of_one_le_right (by positivity) heb
    linarith
  have hP : r3 + r6 ≤ (1+u)*(T*(1+ue)*(aL*(SL + nL*θ) + aR*(SR + nR*θ)) + (nL+nR)*eb*ηe) + 2*η := by
    linarith
  have hY : aL*(SL + nL*θ) + aR*(SR + nR*θ) ≤ aL*SL + aR*SR + (nL+nR)*θ := by
    have h1 : aL*(nL*θ) ≤ nL*θ := by
      have := mul_le_mul_of_nonneg_right haL1 (by positivity : (0:ℝ) ≤ nL*θ); linarith
    have h2 : aR*(nR*θ) ≤ nR*θ := by
      have := mul_le_mul_of_nonneg_right haR1 (by positivity : (0:ℝ) ≤ nR*θ); linarith
    linarith
  -- relative part
  have hrel : (1+u)*((1+u)*(T*(1+ue)*(aL*(SL + nL*θ) + aR*(SR + nR*θ)))) ≤
      T*W*(aL*SL + aR*SR + (nL+nR)*θ) := by
    have h1 : (1+u)*((1+u)*(T*(1+ue)*(aL*(SL + nL*θ) + aR*(SR + nR*θ)))) =
        T*((1+u)^2*(1+ue))*(aL*(SL + nL*θ) + aR*(SR + nR*θ)) := by ring
    rw [h1]
    have h2 : T*((1+u)^2*(1+ue)) ≤ T*W := mul_le_mul_of_nonneg_left hW (by positivity)
    exact mul_le_mul h2 hY (by positivity) (by positivity)
  -- absolute part
  have habs : (1+u)*((1+u)*((nL+nR)*eb*ηe) + 2*η) ≤ (nL+nR)*K*(ηe+η) := by
    have n2 : 2 ≤ nL + nR := by linarith
    have a1 : (1+u)^2*eb*((nL+nR)*ηe) ≤ K*((nL+nR)*ηe) :=
      mul_le_mul_of_nonneg_right hK (by positivity)
    have a2 : (1+u)*(2*η) ≤ (nL+nR)*(K*η) := by
      have h1 : (1+u) * 2 ≤ (nL+nR) * K := by
        have := mul_nonneg (sub_nonneg.2 n2) (by linarith : (0:ℝ) ≤ K)
        linarith
      have := mul_le_mul_of_nonneg_right h1 hη
      linarith
    have e : (1+u)*((1+u)*((nL+nR)*eb*ηe) + 2*η) =
        (1+u)^2*eb*((nL+nR)*ηe) + (1+u)*(2*η) := by ring
    rw [e]
    linarith
  have hTW : 1 ≤ T*W := one_le_mul_of_one_le_of_one_le hT hW1
  have hn0 : 0 ≤ (nL+nR)*K*(ηe+η) := by
    have : 0 ≤ K := by linarith
    positivity
  have h1 : v_d ≤ (1+u)*((1+u)*(T*(1+ue)*(aL*(SL + nL*θ) + aR*(SR + nR*θ)) + (nL+nR)*eb*ηe) + 2*η) :=
    le_trans hr7U (mul_le_mul_of_nonneg_left (by linarith) (by linarith))
  have e2 : (1+u)*((1+u)*(T*(1+ue)*(aL*(SL + nL*θ) + aR*(SR + nR*θ)) + (nL+nR)*eb*ηe) + 2*η) =
      (1+u)*((1+u)*(T*(1+ue)*(aL*(SL + nL*θ) + aR*(SR + nR*θ)))) +
      (1+u)*((1+u)*((nL+nR)*eb*ηe) + 2*η) := by ring
  have hfin : v_d ≤ T*W*(aL*SL + aR*SR + (nL+nR)*θ) + (nL+nR)*K*(ηe+η) := by
    linarith
  have h5 : (nL+nR)*K*(ηe+η) ≤ T*W*((nL+nR)*K*(ηe+η)) := le_mul_of_one_le_left hn0 hTW
  rw [hθ']
  have e3 : T * W * (aL * SL + aR * SR + (nL + nR) * (θ + K * (ηe + η))) =
      T*W*(aL*SL + aR*SR + (nL+nR)*θ) + T*W*((nL+nR)*K*(ηe+η)) := by ring
  rw [e3]
  linarith

theorem comb_e (u ue η ηe R α β θ K α' θ' SL SR nL nR aL aR r3 r6 v_d : ℝ)
    (hu : 0 ≤ u) (hu1 : u ≤ 1/256) (hue : 0 ≤ ue) (hue1 : ue ≤ 1/64) (hη : 0 ≤ η) (hηe : 0 ≤ ηe)
    (hR0 : 0 ≤ R) (hα0 : 0 ≤ α) (hβ0 : 0 ≤ β) (hθ0 : 0 ≤ θ)
    (hnL : 1 ≤ nL) (hnR : 1 ≤ nR) (hSL : 0 ≤ SL) (hSR : 0 ≤ SR)
    (haL0 : 0 ≤ aL) (haR0 : 0 ≤ aR)
    (hK : (1+u)^2 * Real.exp β ≤ K)
    (hα' : α' = α + u*R + 2*(51/50)*u + 51/50*ue) (hθ' : θ' = θ + K*(ηe + η))
    (lL : (1-u)*(Real.exp (-(α + u*R))*(1-ue)*(aL*SL) - nL*(θ + Real.exp β*ηe)) - η ≤ r3)
    (lR : (1-u)*(Real.exp (-(α + u*R))*(1-ue)*(aR*SR) - nR*(θ + Real.exp β*ηe)) - η ≤ r6)
    (hr7L : (1-u)*(r3+r6) ≤ v_d) :
    Real.exp (-α') * (aL*SL + aR*SR) - (nL+nR)*θ' ≤ v_d := by
  have heb : 1 ≤ Real.exp β := Real.one_le_exp hβ0
  have e' : Real.exp (-α') = Real.exp (-(α + u*R)) * Real.exp (-(2*(51/50)*u + 51/50*ue)) := by
    rw [hα', ← Real.exp_add]; congr 1; ring
  have hn3 := num3 u ue hu hu1 hue hue1
  have hV0 : 0 < Real.exp (-(α + u*R)) := Real.exp_pos _
  rw [e']
  generalize Real.exp (-(α + u*R)) = V at hV0 lL lR ⊢
  generalize Real.exp (-(2*(51/50)*u + 51/50*ue)) = W at hn3 ⊢
  generalize Real.exp β = eb at heb hK lL lR
  have hK1 : eb ≤ K := by
    have h1 : 1 ≤ (1+u)^2 := by nlinarith
    have : eb ≤ (1+u)^2 * eb := le_mul_of_one_le_left (by linarith) h1
    linarith
  have hK0 : 1 ≤ K := by linarith
  have hP : (1-u)*(V*(1-ue)*(aL*SL + aR*SR) - (nL+nR)*(θ + eb*ηe)) - 2*η ≤ r3 + r6 := by
    linarith
  have hSx0 : 0 ≤ aL*SL + aR*SR := by positivity
  have hXn0 : 0 ≤ (nL+nR)*(θ + eb*ηe) := by positivity
  have h1 : (1-u)*((1-u)*(V*(1-ue)*(aL*SL + aR*SR) - (nL+nR)*(θ + eb*ηe)) - 2*η) ≤ (1-u)*(r3+r6) :=
    mul_le_mul_of_nonneg_left hP (by linarith)
  have h2 : V * W * (aL*SL + aR*SR) ≤ (1-u)^2*(V*(1-ue))*(aL*SL + aR*SR) := by
    have : V * W ≤ V * ((1-u)^2*(1-ue)) := mul_le_mul_of_nonneg_left hn3 hV0.le
    have := mul_le_mul_of_nonneg_right this hSx0
    linarith
  have b1 : (1-u)*(1-u) ≤ 1 := by nlinarith
  have a1 : (1-u)*(1-u)*((nL+nR)*(θ + eb*ηe)) ≤ (nL+nR)*(θ + eb*ηe) :=
    mul_le_of_le_one_left hXn0 b1
  have a2 : (1-u)*(2*η) ≤ 2*η := mul_le_of_le_one_left (by positivity) (by linarith)
  have e : (1-u)*((1-u)*(V*(1-ue)*(aL*SL + aR*SR) - (nL+nR)*(θ + eb*ηe)) - 2*η) =
      (1-u)^2*(V*(1-ue))*(aL*SL + aR*SR) - (1-u)*(1-u)*((nL+nR)*(θ + eb*ηe)) - (1-u)*(2*η) := by ring
  have h4 : (nL+nR)*(θ + eb*ηe) + 2*η ≤ (nL+nR)*θ' := by
    rw [hθ']
    have a1 : eb*ηe ≤ K*ηe := mul_le_mul_of_nonneg_right hK1 hηe
    have a2 : 2*η ≤ (nL+nR)*(K*η) := by
      have h2 : 2*1 ≤ (nL+nR)*K := mul_le_mul (by linarith) hK0 (by norm_num) (by linarith)
      have := mul_le_mul_of_nonneg_right h2 hη
      linarith
    have a3 : (nL+nR)*(eb*ηe) ≤ (nL+nR)*(K*ηe) := mul_le_mul_of_nonneg_left a1 (by linarith)
    linarith
  linarith


set_option maxHeartbeats 1000000 in
theorem merge_inv (F : Discovery.Rounding) (hs : small F)
    (R α β ζ θ α' β' ζ' θ' K SL SR nL nR dL dR mL mR : ℝ) (st : State_a)
    (hmerge : fmerge_a F ⟨mL, dL⟩ ⟨mR, dR⟩ st)
    (hLR : |mL - mR| ≤ R)
    (hα' : α' = α + F.u*R + 2*(51/50)*F.u + 51/50*F.ue)
    (hβ' : β' = β + (2*F.u + F.ue + F.ηe + F.η))
    (hζ' : ζ' = ζ + (2*F.u + F.ue + F.ηe + 2*F.η))
    (hθ' : θ' = θ + K*(F.ηe + F.η))
    (hα0 : 0 ≤ α) (hβ0 : 0 ≤ β) (hθ0 : 0 ≤ θ) (hζ0 : 0 ≤ ζ) (hζ1 : ζ ≤ 1/2)
    (hK : (1+F.u)^2 * Real.exp β ≤ K)
    (hSL : 0 ≤ SL) (hSR : 0 ≤ SR) (hnL : 1 ≤ nL) (hnR : 1 ≤ nR)
    (hL1 : 1 - ζ ≤ dL) (hL2 : dL ≤ nL * Real.exp β)
    (hL3 : dL ≤ Real.exp α * (SL + nL*θ)) (hL4 : Real.exp (-α) * SL - nL*θ ≤ dL)
    (hR1 : 1 - ζ ≤ dR) (hR2 : dR ≤ nR * Real.exp β)
    (hR3 : dR ≤ Real.exp α * (SR + nR*θ)) (hR4 : Real.exp (-α) * SR - nR*θ ≤ dR) :
    st.m = max mL mR ∧ 1 - ζ' ≤ st.d ∧ st.d ≤ (nL+nR) * Real.exp β' ∧
    st.d ≤ Real.exp α' * (Real.exp (mL - max mL mR) * SL + Real.exp (mR - max mL mR) * SR
      + (nL+nR)*θ') ∧
    Real.exp (-α') * (Real.exp (mL - max mL mR) * SL + Real.exp (mR - max mL mR) * SR)
      - (nL+nR)*θ' ≤ st.d := by
  obtain ⟨v_m, v_d, r1, r2, r3, r4, r5, r6, r7, hvm, a1, a2, a3, a4, a5, a6, a7, hvd, hst⟩ := hmerge
  subst hst
  have hvm' : v_m = max mL mR := hvm
  subst hvm'
  subst hvd
  obtain ⟨hs1, hs2, hs3, hs4, hs5⟩ := hs
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hηe := F.hηe
  have hη := F.hη
  have hsm : F.ue + F.ηe ≤ 1/2 := by
    have : (1:ℝ)/2^100 ≤ 1/64 := by norm_num
    linarith
  have hR0 : 0 ≤ R := le_trans (abs_nonneg _) hLR
  have hLRb := abs_le.mp hLR
  have hzL0 : mL - max mL mR ≤ 0 := by have := le_max_left mL mR; linarith
  have hzR0 : mR - max mL mR ≤ 0 := by have := le_max_right mL mR; linarith
  have hzLR : -(mL - max mL mR) ≤ R := by
    rcases max_cases mL mR with ⟨h, _⟩ | ⟨h, _⟩ <;> rw [h] <;> linarith [hLRb.1, hLRb.2]
  have hzRR : -(mR - max mL mR) ≤ R := by
    rcases max_cases mL mR with ⟨h, _⟩ | ⟨h, _⟩ <;> rw [h] <;> linarith [hLRb.1, hLRb.2]
  have hnL0 : 0 ≤ nL := by linarith
  have hnR0 : 0 ≤ nR := by linarith
  have hdL0 : 0 ≤ dL := by linarith
  have hdR0 : 0 ≤ dR := by linarith
  obtain ⟨xL0, uL, lL, cL, mxL, nL'⟩ := side_bounds F R _ _ _ _ dL SL nL α β θ ζ hzL0 hzLR a1 a2 a3
    hdL0 hSL hnL0 hθ0 (by linarith) hsm hL2 hL3 hL4
  obtain ⟨xR0, uR, lR, cR, mxR, nR'⟩ := side_bounds F R _ _ _ _ dR SR nR α β θ ζ hzR0 hzRR a4 a5 a6
    hdR0 hSR hnR0 hθ0 (by linarith) hsm hR2 hR3 hR4
  obtain ⟨δ7, hδ7, hr7⟩ := a7
  have hδ7' := abs_le.mp hδ7
  have hP : (1-F.u)*((1-ζ)*(1-F.ue-F.ηe)) - 2*F.η ≤ r3 + r6 := by
    rcases le_total mR mL with h | h
    · have hz : mL - max mL mR = 0 := by rw [max_eq_left h]; ring
      have := mxL hz hL1
      linarith
    · have hz : mR - max mL mR = 0 := by rw [max_eq_right h]; ring
      have := mxR hz hR1
      linarith
  have hM : (255/256)*((1/2)*(49/50)) ≤ (1-F.u)*((1-ζ)*(1-F.ue-F.ηe)) := by
    have e1 : (49/50:ℝ) ≤ 1 - F.ue - F.ηe := by
      have : (1:ℝ)/64 + 1/2^100 ≤ 1/50 := by norm_num
      linarith
    have h23 : (1/2)*(49/50) ≤ (1-ζ)*(1-F.ue-F.ηe) :=
      mul_le_mul (by linarith) e1 (by norm_num) (by linarith)
    exact mul_le_mul (by linarith) h23 (by norm_num) (by linarith)
  have hη2 : 2*F.η ≤ 1/2^99 := by
    have : F.η ≤ 1/2^100 := hs4
    have e : (1:ℝ)/2^99 = 2*(1/2^100) := by norm_num
    linarith
  have hP4 : 1/4 ≤ r3 + r6 := by
    have : (1:ℝ)/2^99 ≤ 1/100 := by norm_num
    linarith
  have hPnn : 0 ≤ r3 + r6 := by linarith
  have hr7L : (1-F.u)*(r3+r6) ≤ v_d := by
    rw [hr7]; linarith [mul_nonneg hPnn (by linarith [hδ7'.1] : 0 ≤ δ7 + F.u)]
  have hr7U : v_d ≤ (1+F.u)*(r3+r6) := by
    rw [hr7]; linarith [mul_nonneg hPnn (by linarith [hδ7'.2] : 0 ≤ F.u - δ7)]
  have haL0 := Real.exp_pos (mL - max mL mR)
  have haR0 := Real.exp_pos (mR - max mL mR)
  have haL1 : Real.exp (mL - max mL mR) ≤ 1 := Real.exp_le_one_iff.mpr hzL0
  have haR1 : Real.exp (mR - max mL mR) ≤ 1 := Real.exp_le_one_iff.mpr hzR0
  refine ⟨rfl, ?_, ?_, ?_, ?_⟩
  · -- d ≥ 1 - ζ'
    show 1 - ζ' ≤ v_d
    have n1 := num1 F.u (F.ue + F.ηe) ζ hu (by linarith) (by linarith) (by linarith) hζ0 hζ1
    have h1 : (1-F.u)*((1-F.u)*((1-ζ)*(1-F.ue-F.ηe)) - 2*F.η) ≤ (1-F.u)*(r3+r6) :=
      mul_le_mul_of_nonneg_left hP (by linarith)
    have h3 : (1-F.u)*(2*F.η) ≤ 2*F.η := mul_le_of_le_one_left (by positivity) (by linarith)
    have h4 : (1-F.u)*((1-F.u)*((1-ζ)*(1-F.ue-F.ηe)) - 2*F.η) =
        (1-F.u)*((1-F.u)*((1-ζ)*(1-F.ue-F.ηe))) - (1-F.u)*(2*F.η) := by ring
    have h5 : (1-F.u)^2*((1-ζ)*(1-(F.ue+F.ηe))) = (1-F.u)*((1-F.u)*((1-ζ)*(1-F.ue-F.ηe))) := by ring
    rw [hζ']
    linarith
  · show v_d ≤ (nL+nR) * Real.exp β'
    exact comb_c F.u F.ue F.η F.ηe β β' nL nR r3 r6 v_d hu hη hηe hue hnL hnR hβ0 hβ' cL cR hr7U
  · show v_d ≤ _
    exact comb_d F.u F.ue F.η F.ηe R α β θ K α' θ' SL SR nL nR _ _ r3 r6 v_d hu (by linarith) hue
      (by linarith) hη hηe hR0 hα0 hβ0 hθ0 hnL hnR hSL hSR haL0.le haL1 haR0.le haR1 hK hα' hθ'
      uL uR hr7U
  · show _ ≤ v_d
    exact comb_e F.u F.ue F.η F.ηe R α β θ K α' θ' SL SR nL nR _ _ r3 r6 v_d hu (by linarith) hue
      (by linarith) hη hηe hR0 hα0 hβ0 hθ0 hnL hnR hSL hSR haL0.le haR0.le hK hα' hθ'
      lL lR hr7L


theorem flat_ne (t : Discovery.Tree ℝ) (h : ∀ b ∈ t.leaves, b ≠ []) : t.flat ≠ [] := by
  induction t with
  | leaf b => simpa [Discovery.Tree.leaves, Discovery.Tree.flat] using h b (by simp [Discovery.Tree.leaves])
  | node l r ihl ihr =>
    have hl : l.flat ≠ [] := ihl (fun b hb => h b (by simp [Discovery.Tree.leaves, hb]))
    simp [Discovery.Tree.flat, hl]

theorem depth_le_len (t : Discovery.Tree ℝ) (h : ∀ b ∈ t.leaves, b ≠ []) :
    t.depth + 1 ≤ t.flat.length := by
  induction t with
  | leaf b =>
    have := h b (by simp [Discovery.Tree.leaves])
    simp only [Discovery.Tree.depth, Discovery.Tree.flat]
    exact List.length_pos_iff.mpr this
  | node l r ihl ihr =>
    have hl := ihl (fun b hb => h b (by simp [Discovery.Tree.leaves, hb]))
    have hr := ihr (fun b hb => h b (by simp [Discovery.Tree.leaves, hb]))
    simp only [Discovery.Tree.depth, Discovery.Tree.flat, List.length_append]
    omega

theorem S_append (A B : List ℝ) (hA : A ≠ []) (hB : B ≠ []) :
    ((A ++ B).map fun x => Real.exp (x - lmax (A ++ B))).sum =
    Real.exp (lmax A - lmax (A ++ B)) * (A.map fun x => Real.exp (x - lmax A)).sum +
    Real.exp (lmax B - lmax (A ++ B)) * (B.map fun x => Real.exp (x - lmax B)).sum := by
  rw [List.map_append, List.sum_append]
  have key : ∀ (C : List ℝ) (c M : ℝ), (C.map fun x => Real.exp (x - M)).sum =
      Real.exp (c - M) * (C.map fun x => Real.exp (x - c)).sum := by
    intro C c M
    rw [← List.sum_map_mul_left]
    congr 1
    apply List.map_congr_left
    intro x _
    rw [← Real.exp_add]; congr 1; ring
  rw [key A (lmax A) (lmax (A ++ B)), key B (lmax B) (lmax (A ++ B))]


set_option maxHeartbeats 1000000 in
theorem tree_inv (F : Discovery.Rounding) (D : ℕ) (lo hi K : ℝ) (hs : small F) (hlh : lo ≤ hi)
    (hK0 : 0 ≤ K) (Kd : ℕ) (a0 a1 b0 b1 z0 z1 t0 t1 : ℝ)
    (ha0 : a0 = F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u))
    (ha1 : a1 = F.u*(hi-lo) + 2*(51/50)*F.u + 51/50*F.ue)
    (hb0 : b0 = D*F.u + F.ue + F.ηe) (hb1 : b1 = 2*F.u + F.ue + F.ηe + F.η)
    (hz0 : z0 = D*F.u + F.ue + F.ηe) (hz1 : z1 = 2*F.u + F.ue + F.ηe + 2*F.η)
    (ht0 : t0 = F.ηe) (ht1 : t1 = K*(F.ηe + F.η))
    (hKk : ∀ k : ℕ, k ≤ Kd → (1+F.u)^2 * Real.exp (b0 + k*b1) ≤ K)
    (hZk : ∀ k : ℕ, k ≤ Kd → z0 + k*z1 ≤ 1/2) :
    ∀ t : Discovery.Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → (∀ x ∈ t.flat, lo ≤ x ∧ x ≤ hi) →
    ∀ k : ℕ, t.depth ≤ k → k ≤ Kd → ∀ st : State_a,
    Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
    st.m = lmax t.flat ∧ 1 - (z0 + k*z1) ≤ st.d ∧ st.d ≤ t.flat.length * Real.exp (b0 + k*b1) ∧
    st.d ≤ Real.exp (a0 + k*a1) * ((t.flat.map fun x => Real.exp (x - lmax t.flat)).sum
      + t.flat.length * (t0 + k*t1)) ∧
    Real.exp (-(a0 + k*a1)) * (t.flat.map fun x => Real.exp (x - lmax t.flat)).sum
      - t.flat.length * (t0 + k*t1) ≤ st.d := by
  have hu := F.hu
  have hue := F.hue
  have hη := F.hη
  have hηe := F.hηe
  have hR : 0 ≤ hi - lo := by linarith
  have ha0' : 0 ≤ a0 := by rw [ha0]; positivity
  have ha1' : 0 ≤ a1 := by rw [ha1]; positivity
  have hb1' : 0 ≤ b1 := by rw [hb1]; positivity
  have hz1' : 0 ≤ z1 := by rw [hz1]; positivity
  have hz0' : 0 ≤ z0 := by rw [hz0]; positivity
  have hb0' : 0 ≤ b0 := by rw [hb0]; positivity
  have ht0' : 0 ≤ t0 := by rw [ht0]; positivity
  have ht1' : 0 ≤ t1 := by rw [ht1]; positivity
  intro t
  induction t with
  | leaf b =>
    subst ha0 ha1 hb0 hb1 hz0 hz1 ht0 ht1
    intro hne hr k hk hkd st hev
    cases hev with
    | leaf hb =>
      simp only [Discovery.Tree.flat] at hr ⊢
      have hbne : b ≠ [] := hne b (by simp [Discovery.Tree.leaves])
      obtain ⟨h1, h2, h3, h4, h5⟩ := leaf_inv F D lo hi b hbne hr hs st hb
      have hk0 : (0:ℝ) ≤ k := Nat.cast_nonneg _
      have hS0 : 0 ≤ (b.map fun x => Real.exp (x - lmax b)).sum := by
        have := S_ge_one b hbne; linarith
      have hn0 : (0:ℝ) ≤ b.length := Nat.cast_nonneg _
      have hkz : 0 ≤ (k:ℝ)*(2*F.u + F.ue + F.ηe + 2*F.η) := by positivity
      have hkb : 0 ≤ (k:ℝ)*(2*F.u + F.ue + F.ηe + F.η) := by positivity
      have hka : 0 ≤ (k:ℝ)*(F.u*(hi-lo) + 2*(51/50)*F.u + 51/50*F.ue) := by positivity
      have hkt : 0 ≤ (k:ℝ)*(K*(F.ηe + F.η)) := by positivity
      refine ⟨h1, ?_, ?_, ?_, ?_⟩
      · linarith
      · exact le_trans h3 (mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) hn0)
      · refine le_trans h4 (mul_le_mul (Real.exp_le_exp.mpr ?_) ?_ (by positivity) (Real.exp_pos _).le)
        · linarith
        · have := mul_nonneg hn0 hkt
          nlinarith
      · have e1 : Real.exp (-(F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u) +
            k*(F.u*(hi-lo) + 2*(51/50)*F.u + 51/50*F.ue))) ≤
            Real.exp (-(F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u))) :=
          Real.exp_le_exp.mpr (by linarith)
        have e2 := mul_le_mul_of_nonneg_right e1 hS0
        have := mul_nonneg hn0 hkt
        nlinarith
  | node l r ihl ihr =>
    intro hne hr k hk hkd st hev
    cases hev with
    | node ha hc hm =>
      rename_i a c
      simp only [Discovery.Tree.depth] at hk
      obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
      have ld : l.depth ≤ k' := by omega
      have rd : r.depth ≤ k' := by omega
      have hnel : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => hne b (by simp [Discovery.Tree.leaves, hb])
      have hner : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => hne b (by simp [Discovery.Tree.leaves, hb])
      have hrl : ∀ x ∈ l.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hr x (by simp [Discovery.Tree.flat, hx])
      have hrr : ∀ x ∈ r.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hr x (by simp [Discovery.Tree.flat, hx])
      have hLne := flat_ne l hnel
      have hRne := flat_ne r hner
      obtain ⟨m1, p1, c1, d1, e1⟩ := ihl hnel hrl k' ld (by omega) a ha
      obtain ⟨m2, p2, c2, d2, e2⟩ := ihr hner hrr k' rd (by omega) c hc
      have hSn := S_append l.flat r.flat hLne hRne
      have hMx : lmax (l.flat ++ r.flat) = max a.m c.m := by
        rw [lmax_append hLne hRne, m1, m2]
      have hS1 := S_ge_one l.flat hLne
      have hS2 := S_ge_one r.flat hRne
      have hnL : (1:ℝ) ≤ l.flat.length := Nat.one_le_cast.mpr (List.length_pos_iff.mpr hLne)
      have hnR : (1:ℝ) ≤ r.flat.length := Nat.one_le_cast.mpr (List.length_pos_iff.mpr hRne)
      have hbl := hrl _ (lmax_mem hLne)
      have hbr := hrr _ (lmax_mem hRne)
      have hLR : |a.m - c.m| ≤ hi - lo := by
        rw [m1, m2, abs_le]; constructor <;> linarith [hbl.1, hbl.2, hbr.1, hbr.2]
      have hn : (((l.flat ++ r.flat).length : ℕ) : ℝ) = l.flat.length + r.flat.length := by simp
      have hfl : (Discovery.Tree.node l r).flat = l.flat ++ r.flat := rfl
      rw [hfl, hSn, hn, hMx]
      generalize (l.flat.map fun x => Real.exp (x - lmax l.flat)).sum = SL at *
      generalize (r.flat.map fun x => Real.exp (x - lmax r.flat)).sum = SR at *
      rw [← m1, ← m2]
      have hk0 : (0:ℝ) ≤ k' := Nat.cast_nonneg _
      have := merge_inv F hs (hi - lo) (a0 + k'*a1) (b0 + k'*b1) (z0 + k'*z1) (t0 + k'*t1)
        (a0 + ((k'+1 : ℕ):ℝ)*a1) (b0 + ((k'+1 : ℕ):ℝ)*b1) (z0 + ((k'+1 : ℕ):ℝ)*z1)
        (t0 + ((k'+1 : ℕ):ℝ)*t1) K SL SR l.flat.length r.flat.length a.d c.d a.m c.m _ hm hLR
        (by push_cast; rw [ha1]; ring) (by push_cast; rw [hb1]; ring) (by push_cast; rw [hz1]; ring)
        (by push_cast; rw [ht1]; ring)
        (by positivity) (by positivity) (by positivity) (by positivity) (hZk k' (by omega))
        (hKk k' (by omega)) (by linarith) (by linarith) hnL hnR
        p1 c1 d1 e1 p2 c2 d2 e2
      exact this


theorem final_ana (S d a E : ℝ) (hS : 1 ≤ S) (ha : 0 ≤ a) (hE0 : 0 ≤ E) (hE : E ≤ 2/25)
    (hd : 4/5 ≤ d) (hup : d ≤ Real.exp a * (S + E)) (hlo : Real.exp (-a) * S - E ≤ d) :
    |Real.log d - Real.log S| ≤ 10/9*(a+E) := by
  have hSpos : 0 < S := by linarith
  have hdpos : 0 < d := by linarith
  have hea : 0 < Real.exp a := Real.exp_pos a
  rw [abs_le]
  constructor
  · by_cases hc : Real.exp a * E * 10 ≤ S
    · have hP : 0 < Real.exp (-a) * S := by positivity
      have hexp : Real.exp (-a) * Real.exp a = 1 := by rw [← Real.exp_add]; simp
      have hE10 : 10 * E ≤ Real.exp (-a) * S := by
        have := mul_le_mul_of_nonneg_left hc (Real.exp_pos (-a)).le
        nlinarith
      set P := Real.exp (-a) * S with hPdef
      set r := E / P with hr
      have hr0 : 0 ≤ r := by positivity
      have hrE : r * P = E := by rw [hr]; field_simp
      have hr1 : r ≤ 1/10 := by
        rw [hr, div_le_iff₀ hP]; linarith
      have hd9 : P * (1 - r) ≤ d := by
        have : P * (1 - r) = P - E := by rw [mul_sub, mul_one, mul_comm, hrE]
        linarith
      have h1r : 0 < 1 - r := by linarith
      have hlogd : Real.log (P * (1 - r)) ≤ Real.log d :=
        Real.log_le_log (mul_pos hP h1r) hd9
      have hlog1 : 1 - 1/(1 - r) ≤ Real.log (1 - r) := by
        have := Real.one_sub_inv_le_log_of_pos h1r
        simpa [one_div] using this
      have hlogP : Real.log P = -a + Real.log S := by
        rw [hPdef, Real.log_mul (Real.exp_pos _).ne' hSpos.ne', Real.log_exp]
      rw [Real.log_mul hP.ne' h1r.ne'] at hlogd
      have hfrac : 1 - 1/(1 - r) ≥ -(10/9) * r := by
        have : 1/(1 - r) ≤ 1 + 10/9 * r := by
          rw [div_le_iff₀ h1r]; nlinarith
        linarith
      -- r ≤ E + a/10
      have hexp_le : Real.exp a ≤ 1 + a * Real.exp a := by
        have := Real.add_one_le_exp (-a)
        have h2 : Real.exp a * (-a + 1) ≤ Real.exp a * Real.exp (-a) :=
          mul_le_mul_of_nonneg_left this hea.le
        rw [mul_comm (Real.exp a) (Real.exp (-a)), hexp] at h2
        nlinarith
      have hrS : r * S = E * Real.exp a := by
        rw [hr, hPdef, Real.exp_neg]; field_simp
      have h3 : r * S ≤ E + a * (r * S) := by
        rw [hrS]; nlinarith [mul_le_mul_of_nonneg_left hexp_le hE0]
      have h4 : r ≤ E / S + a * r := by
        have : r ≤ (E + a * (r * S)) / S := by
          rw [le_div_iff₀ hSpos]; linarith
        have e : (E + a * (r * S)) / S = E / S + a * r := by field_simp
        linarith
      have h5 : E / S ≤ E := div_le_self hE0 hS
      have h6 : a * r ≤ a * (1/10) := mul_le_mul_of_nonneg_left hr1 ha
      have h7 : r ≤ E + a/10 := by linarith
      nlinarith
    · push Not at hc
      have hEpos : 0 < E := by
        by_contra h
        have : E = 0 := le_antisymm (not_lt.mp h) hE0
        subst this
        linarith
      have h1 : S ≤ d * Real.exp a := by
        have : S < 10 * (2/25) * Real.exp a := by nlinarith
        nlinarith
      have h2 := Real.log_le_log hSpos h1
      rw [Real.log_mul hdpos.ne' hea.ne', Real.log_exp] at h2
      nlinarith
  · have h1 : Real.log d ≤ a + Real.log (S + E) := by
      have := Real.log_le_log hdpos hup
      rw [Real.log_mul hea.ne' (by linarith), Real.log_exp] at this
      exact this
    have h2 : Real.log (S + E) ≤ Real.log S + E := by
      have h3 : Real.log ((S + E) / S) ≤ (S + E) / S - 1 :=
        Real.log_le_sub_one_of_pos (by positivity)
      rw [Real.log_div (by linarith) hSpos.ne'] at h3
      have : (S + E) / S - 1 = E / S := by field_simp; ring
      have h5 : E / S ≤ E := div_le_self hE0 hS
      linarith
    nlinarith


theorem sum_exp_eq (A : List ℝ) (c : ℝ) :
    (A.map Real.exp).sum = Real.exp c * (A.map fun x => Real.exp (x - c)).sum := by
  rw [← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro x _
  rw [← Real.exp_add]; congr 1; ring

theorem exp_small (B : ℝ) (h0 : 0 ≤ B) (h1 : B ≤ 13/100) : Real.exp B ≤ 23/20 := by
  have h := Real.add_one_le_exp (-B)
  have e : Real.exp B * Real.exp (-B) = 1 := by rw [← Real.exp_add]; simp
  have h2 : Real.exp B * (-B + 1) ≤ 1 := by
    have := mul_le_mul_of_nonneg_left h (Real.exp_pos B).le
    linarith
  nlinarith [Real.exp_pos B]

theorem goal_ineq (u ue η ηe hl D k n : ℝ) (hu0 : 0 ≤ u) (hue0 : 0 ≤ ue) (hη0 : 0 ≤ η)
    (hηe0 : 0 ≤ ηe) (hl0 : 0 ≤ hl) (hD0 : 0 ≤ D) (hk0 : 0 ≤ k) (hn0 : 0 ≤ n)
    (hg : n*(η+ηe) ≤ u) :
    10/9*((u*hl + 51/50*ue + 51/50*(D*u) + k*(u*hl + 2*(51/50)*u + 51/50*ue)) +
      n*(ηe + k*(6/5*(ηe+η)))) ≤
    2*(k+1)*hl*u + 2*(k+1)*ue + 2*(2*k+D)*u + 4*n*(η+ηe) := by
  have h1 : 0 ≤ u*hl := mul_nonneg hu0 hl0
  have h2 : 0 ≤ k*(u*hl) := mul_nonneg hk0 h1
  have h3 : 0 ≤ k*ue := mul_nonneg hk0 hue0
  have h4 : 0 ≤ D*u := mul_nonneg hD0 hu0
  have h5 : 0 ≤ k*u := mul_nonneg hk0 hu0
  have h6 : k*(n*(η+ηe)) ≤ k*u := mul_le_mul_of_nonneg_left hg hk0
  have h7 : 0 ≤ n*η := mul_nonneg hn0 hη0
  have h8 : 0 ≤ n*ηe := mul_nonneg hn0 hηe0
  nlinarith


set_option maxHeartbeats 2000000 in
theorem fp_bound_aux : ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List (ℝ)), X ≠ [] →
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree (ℝ)), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((2 : ℝ) * ((t.depth : ℝ) + (1 : ℝ)) * (hi - lo) * F.u) + ((2 : ℝ) * ((t.depth : ℝ) + (1 : ℝ)) * F.ue) + ((2 : ℝ) * (((2 : ℝ) * (t.depth : ℝ)) + (D : ℝ)) * F.u) + ((4 : ℝ) * (X.length : ℝ) * (F.η + F.ηe))) := by
  intro F D lo hi X hX hr hs hnX t hflat hleaves hdep st hev
  subst hflat
  have hs' := hs
  obtain ⟨hs1, hs2, hs3, hs4, hs5⟩ := hs
  have hu := F.hu
  have hue := F.hue
  have hul := F.hul
  have hη := F.hη
  have hηe := F.hηe
  have hne := flat_ne t hleaves
  have hdl := depth_le_len t hleaves
  have hm0 := lmax_mem hne
  have hlh : lo ≤ hi := le_trans (hr _ hm0).1 (hr _ hm0).2
  have hk0 : (0:ℝ) ≤ t.depth := Nat.cast_nonneg _
  have hD0 : (0:ℝ) ≤ D := Nat.cast_nonneg _
  have hn0 : (0:ℝ) ≤ t.flat.length := Nat.cast_nonneg _
  have hnd : ((t.depth : ℕ) : ℝ) + 1 ≤ t.flat.length := by exact_mod_cast hdl
  -- base estimates
  have hbase : ∀ j : ℕ, j ≤ t.depth →
      (D + 2*(j:ℝ))*F.u + ((j:ℝ)+1)*F.ue ≤ 1/8 ∧ ((j:ℝ)+1)*(F.η+F.ηe) ≤ F.u := by
    intro j hj
    have hjd : (j:ℝ) ≤ t.depth := by exact_mod_cast hj
    have hj0 : (0:ℝ) ≤ j := Nat.cast_nonneg _
    constructor
    · have a1 : (j:ℝ)*F.u ≤ t.depth*F.u := mul_le_mul_of_nonneg_right hjd hu
      have a2 : (j:ℝ)*F.ue ≤ t.depth*F.ue := mul_le_mul_of_nonneg_right hjd hue
      have := mul_nonneg hk0 hul
      have := mul_nonneg hD0 hul
      have := mul_nonneg hD0 hue
      have := mul_nonneg hD0 hu
      nlinarith
    · have : ((j:ℝ)+1) ≤ t.flat.length := by linarith
      calc ((j:ℝ)+1)*(F.η+F.ηe) ≤ t.flat.length * (F.η+F.ηe) :=
            mul_le_mul_of_nonneg_right this (by linarith)
        _ ≤ F.u := hnX
  have hKk : ∀ j : ℕ, j ≤ t.depth →
      (1+F.u)^2 * Real.exp ((D*F.u + F.ue + F.ηe) + j*(2*F.u + F.ue + F.ηe + F.η)) ≤ 6/5 := by
    intro j hj
    obtain ⟨b1, b2⟩ := hbase j hj
    have hj0 : (0:ℝ) ≤ j := Nat.cast_nonneg _
    have hB : (D*F.u + F.ue + F.ηe) + j*(2*F.u + F.ue + F.ηe + F.η) ≤ 13/100 := by nlinarith
    have hB0 : 0 ≤ (D*F.u + F.ue + F.ηe) + j*(2*F.u + F.ue + F.ηe + F.η) := by positivity
    have h1 := exp_small _ hB0 hB
    have h2 : (1+F.u)^2 ≤ (257/256)^2 := pow_le_pow_left₀ (by linarith) (by linarith) 2
    calc (1+F.u)^2 * Real.exp _ ≤ (257/256)^2 * (23/20) :=
          mul_le_mul h2 h1 (Real.exp_pos _).le (by norm_num)
      _ ≤ 6/5 := by norm_num
  have hZk : ∀ j : ℕ, j ≤ t.depth →
      (D*F.u + F.ue + F.ηe) + j*(2*F.u + F.ue + F.ηe + 2*F.η) ≤ 1/5 := by
    intro j hj
    obtain ⟨b1, b2⟩ := hbase j hj
    have hj0 : (0:ℝ) ≤ j := Nat.cast_nonneg _
    nlinarith
  have hinv := tree_inv F D lo hi (6/5) hs' hlh (by norm_num) t.depth
    (F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u))
    (F.u*(hi-lo) + 2*(51/50)*F.u + 51/50*F.ue)
    (D*F.u + F.ue + F.ηe) (2*F.u + F.ue + F.ηe + F.η)
    (D*F.u + F.ue + F.ηe) (2*F.u + F.ue + F.ηe + 2*F.η)
    F.ηe ((6/5)*(F.ηe + F.η)) rfl rfl rfl rfl rfl rfl rfl rfl hKk
    (fun j hj => le_trans (hZk j hj) (by norm_num)) t hleaves hr t.depth le_rfl le_rfl st hev
  obtain ⟨hm, hlb, hc, hd, he⟩ := hinv
  have hZd := hZk t.depth le_rfl
  have hd45 : 4/5 ≤ st.d := by linarith
  have hdpos : 0 < st.d := by linarith
  have hS1 := S_ge_one t.flat hne
  -- the true quantity
  have hsum := sum_exp_eq t.flat (lmax t.flat)
  have hSpos : 0 < (t.flat.map fun x => Real.exp (x - lmax t.flat)).sum := by linarith
  have hz : znorm st = st.d * Real.exp (lmax t.flat) := by
    simp only [znorm, hm, zero_sub, Real.exp_neg]
    field_simp
  have hlog1 : Real.log (znorm st) = Real.log st.d + lmax t.flat := by
    rw [hz, Real.log_mul hdpos.ne' (Real.exp_pos _).ne', Real.log_exp]
  have hlog2 : Real.log ((t.flat.map Real.exp).sum) =
      lmax t.flat + Real.log ((t.flat.map fun x => Real.exp (x - lmax t.flat)).sum) := by
    rw [hsum, Real.log_mul (Real.exp_pos _).ne' hSpos.ne', Real.log_exp]
  have e : Real.log (znorm st) - Real.log ((t.flat.map Real.exp).sum) =
      Real.log st.d - Real.log ((t.flat.map fun x => Real.exp (x - lmax t.flat)).sum) := by
    rw [hlog1, hlog2]; ring
  rw [e]
  -- numeric parameters
  have hku : (t.depth:ℝ) * F.u ≤ 1/16 := by
    have := mul_nonneg hk0 hue
    have := mul_nonneg hk0 hul
    have := mul_nonneg hD0 hu
    have := mul_nonneg hD0 hue
    have := mul_nonneg hD0 hul
    nlinarith
  have hg : (t.flat.length:ℝ) * (F.η + F.ηe) ≤ F.u := hnX
  have hEnn : 0 ≤ (t.flat.length:ℝ) * (F.ηe + (t.depth:ℝ) * ((6/5)*(F.ηe + F.η))) := by positivity
  have hE : (t.flat.length:ℝ) * (F.ηe + (t.depth:ℝ) * ((6/5)*(F.ηe + F.η))) ≤ 2/25 := by
    have h1 : (t.depth:ℝ) * ((t.flat.length:ℝ) * (F.η + F.ηe)) ≤ (t.depth:ℝ) * F.u :=
      mul_le_mul_of_nonneg_left hg hk0
    have h2 : 0 ≤ (t.flat.length:ℝ) * F.η := mul_nonneg hn0 hη
    nlinarith
  have hapos : 0 ≤ (F.u*(hi-lo) + 51/50*F.ue + 51/50*(D*F.u)) +
      (t.depth:ℝ) * (F.u*(hi-lo) + 2*(51/50)*F.u + 51/50*F.ue) := by
    have : 0 ≤ hi - lo := by linarith
    positivity
  have fin := final_ana _ _ _ _ hS1 hapos hEnn hE hd45 hd (by simpa using he)
  refine le_trans fin ?_
  have := goal_ineq F.u F.ue F.η F.ηe (hi - lo) D t.depth t.flat.length hu hue hη hηe
    (by linarith) hD0 hk0 hn0 hg
  nlinarith

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List (ℝ)), X ≠ [] →
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree (ℝ)), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((2 : ℝ) * ((t.depth : ℝ) + (1 : ℝ)) * (hi - lo) * F.u) + ((2 : ℝ) * ((t.depth : ℝ) + (1 : ℝ)) * F.ue) + ((2 : ℝ) * (((2 : ℝ) * (t.depth : ℝ)) + (D : ℝ)) * F.u) + ((4 : ℝ) * (X.length : ℝ) * (F.η + F.ηe))) := by
  exact fp_bound_aux

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxOnline
