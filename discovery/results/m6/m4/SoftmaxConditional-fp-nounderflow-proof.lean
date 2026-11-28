import FPStatements

namespace Discovery.FP.SoftmaxConditionalFpNounderflow

open Discovery

-- helper lemmas here
theorem fac_bounds {θ s : ℝ} (h : |θ| ≤ s) (hs : s ≤ 1/2) :
    Real.exp (-(2*s)) ≤ 1+θ ∧ 1+θ ≤ Real.exp (2*s) := by
  have h1 := abs_le.1 h
  have hs0 : 0 ≤ s := (abs_nonneg _).trans h
  have hg : 1 + 2*s ≤ Real.exp (2*s) := by linarith [Real.add_one_le_exp (2*s)]
  have hE := Real.exp_pos (-(2*s))
  have e : Real.exp (-(2*s)) * Real.exp (2*s) = 1 := by rw [← Real.exp_add]; simp
  constructor
  · by_contra hc
    rw [not_le] at hc
    nlinarith [mul_le_mul_of_nonneg_left hg hE.le]
  · linarith

theorem mulb {a b p q : ℝ} (ha : Real.exp (-p) ≤ a ∧ a ≤ Real.exp p)
    (hb : Real.exp (-q) ≤ b ∧ b ≤ Real.exp q) :
    Real.exp (-(p+q)) ≤ a*b ∧ a*b ≤ Real.exp (p+q) := by
  have h1 := Real.exp_pos (-p)
  have h2 := Real.exp_pos (-q)
  have h3 := Real.exp_pos p
  have h4 := Real.exp_pos q
  rw [neg_add, Real.exp_add, Real.exp_add]
  exact ⟨mul_le_mul ha.1 hb.1 h2.le (h1.le.trans ha.1),
    mul_le_mul ha.2 hb.2 (h2.le.trans hb.1) h3.le⟩

theorem comb {Z a r ε εa κ : ℝ} (hZ : 0 < Z) (hε : εa ≤ ε)
    (h1 : Z * Real.exp (-εa) ≤ a) (h2 : a ≤ Z * Real.exp εa)
    (hr1 : Real.exp (-κ) * a ≤ r) (hr2 : r ≤ Real.exp κ * a) :
    Real.exp (-(ε+κ)) * Z ≤ r ∧ r ≤ Real.exp (ε+κ) * Z := by
  have e1 : Real.exp (-ε) ≤ Real.exp (-εa) := Real.exp_le_exp.2 (by linarith)
  have e2 : Real.exp εa ≤ Real.exp ε := Real.exp_le_exp.2 hε
  have k1 := Real.exp_pos (-κ)
  have k2 := Real.exp_pos κ
  have hz1 : Real.exp (-ε) * Z ≤ a := by nlinarith [mul_le_mul_of_nonneg_left e1 hZ.le]
  have hz2 : a ≤ Real.exp ε * Z := by nlinarith [mul_le_mul_of_nonneg_left e2 hZ.le]
  rw [neg_add, Real.exp_add, Real.exp_add]
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left hz1 k1.le]
  · nlinarith [mul_le_mul_of_nonneg_left hz2 k2.le]

theorem term_bound (F : Discovery.Rounding) (he0 : F.ηe = 0) (m x y : ℝ)
    (h : ∃ q1 q2, F.add (x - m) q1 ∧ F.exp q1 q2 ∧ y = q2) :
    Real.exp ((x-m) - |x-m| * F.u) * (1 - F.ue) ≤ y ∧
      y ≤ Real.exp ((x-m) + |x-m| * F.u) * (1 + F.ue) := by
  obtain ⟨q1, q2, ⟨δ, hδ, hq1⟩, ⟨δe, e, hδe, he, hq2, _⟩, rfl⟩ := h
  rw [he0] at he
  have e0 : e = 0 := abs_nonpos_iff.1 he
  have hθ : |(x-m)*δ| ≤ |x-m| * F.u := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hδ (abs_nonneg _)
  have hq : q1 = (x-m) + (x-m)*δ := by rw [hq1]; ring
  have h1 := abs_le.1 hθ
  have hE1 : Real.exp ((x-m) - |x-m| * F.u) ≤ Real.exp q1 :=
    Real.exp_le_exp.2 (by linarith)
  have hE2 : Real.exp q1 ≤ Real.exp ((x-m) + |x-m| * F.u) :=
    Real.exp_le_exp.2 (by linarith)
  have h2 := abs_le.1 hδe
  have hue := F.hue1
  rw [hq2, e0, add_zero]
  constructor
  · exact mul_le_mul hE1 (by linarith) (by linarith) (Real.exp_pos _).le
  · exact mul_le_mul hE2 (by linarith) (by linarith) (Real.exp_pos _).le

theorem term_step (F : Discovery.Rounding) (hs : small F) (h0 : F.η = 0) (he0 : F.ηe = 0)
    (d mL m' Δ r2 r3 r4 : ℝ) (hd : 0 < d) (hΔ : |mL - m'| ≤ Δ)
    (h2 : F.add (mL - m') r2) (h3 : F.exp r2 r3) (h4 : F.mul (d * r3) r4) :
    0 < r4 ∧ Real.exp (-(Δ*F.u + 2*F.ue + 2*F.u)) * (d * Real.exp mL) ≤ r4 * Real.exp m' ∧
      r4 * Real.exp m' ≤ Real.exp (Δ*F.u + 2*F.ue + 2*F.u) * (d * Real.exp mL) := by
  obtain ⟨δ1, hδ1, hr2⟩ := h2
  obtain ⟨δe, e, hδe, he, hr3, _⟩ := h3
  obtain ⟨δm, em, hδm, hem, hr4, _⟩ := h4
  rw [he0] at he
  rw [h0] at hem
  have e0 : e = 0 := abs_nonpos_iff.1 he
  have em0 : em = 0 := abs_nonpos_iff.1 hem
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hs' := hs
  obtain ⟨su, sue, -, -, -⟩ := hs'
  set θ := (mL - m') * δ1 with hθdef
  have hθ : |θ| ≤ Δ * F.u := by
    rw [hθdef, abs_mul]
    exact mul_le_mul hΔ hδ1 (abs_nonneg _) ((abs_nonneg _).trans hΔ)
  have key : r4 * Real.exp m' = (d * Real.exp mL) * (Real.exp θ * (1+δe) * (1+δm)) := by
    have : Real.exp r2 * Real.exp m' = Real.exp mL * Real.exp θ := by
      rw [← Real.exp_add, ← Real.exp_add, hr2, hθdef]; congr 1; ring
    rw [hr4, hr3, e0, em0]
    linear_combination (d * (1+δe) * (1+δm)) * this
  have fa := fac_bounds hδe (s := F.ue) (by linarith)
  have fb := fac_bounds hδm (s := F.u) (by linarith)
  have fc : Real.exp (-(Δ*F.u)) ≤ Real.exp θ ∧ Real.exp θ ≤ Real.exp (Δ*F.u) :=
    ⟨Real.exp_le_exp.2 (abs_le.1 hθ).1 |> fun h => by simpa using h, Real.exp_le_exp.2 (abs_le.1 hθ).2⟩
  have m1 := mulb (mulb fc fa) fb
  have hpos : 0 < Real.exp θ * (1+δe) * (1+δm) :=
    lt_of_lt_of_le (Real.exp_pos _) m1.1
  have hdd : 0 < d * Real.exp mL := by positivity
  have hr4pos : 0 < r4 * Real.exp m' := by rw [key]; positivity
  refine ⟨?_, ?_, ?_⟩
  · by_contra hc
    rw [not_lt] at hc
    nlinarith [Real.exp_pos m']
  · rw [key]
    have := mul_le_mul_of_nonneg_left m1.1 hdd.le
    have e : Δ*F.u + 2*F.ue + 2*F.u = Δ*F.u + 2*F.ue + 2*F.u := rfl
    have e' : (Δ*F.u + 2*F.ue) + 2*F.u = Δ*F.u + 2*F.ue + 2*F.u := rfl
    nlinarith
  · rw [key]
    have := mul_le_mul_of_nonneg_left m1.2 hdd.le
    nlinarith

theorem merge_step (F : Discovery.Rounding) (hs : small F) (h0 : F.η = 0) (he0 : F.ηe = 0)
    (tau lo hi : ℝ) (L R st : State_a) (ZL ZR εa εb ε : ℝ) (hZL : 0 < ZL) (hZR : 0 < ZR)
    (hdL : 0 < L.d) (hdR : 0 < R.d) (hmL : lo ≤ L.m ∧ L.m ≤ hi) (hmR : lo ≤ R.m ∧ R.m ≤ hi)
    (hεa : εa ≤ ε) (hεb : εb ≤ ε)
    (hL : ZL * Real.exp (-εa) ≤ L.d * Real.exp L.m ∧ L.d * Real.exp L.m ≤ ZL * Real.exp εa)
    (hR : ZR * Real.exp (-εb) ≤ R.d * Real.exp R.m ∧ R.d * Real.exp R.m ≤ ZR * Real.exp εb)
    (h : fmerge_a F tau L R st) :
    0 < st.d ∧ (st.m = L.m ∨ st.m = R.m) ∧
      Real.exp (-(ε + (2*(hi-lo)*F.u + 4*F.u + 2*F.ue))) * (ZL+ZR) ≤ st.d * Real.exp st.m ∧
      st.d * Real.exp st.m ≤ Real.exp (ε + (2*(hi-lo)*F.u + 4*F.u + 2*F.ue)) * (ZL+ZR) := by
  obtain ⟨v_m, v_d, r1, r2, r3, r4, r5, r6, r7, r8, h1, hvm, h2, h3, h4, h5, h6, h7, h8, hvd, hst⟩ := h
  subst hst
  subst hvd
  simp only
  have hm' : v_m = L.m ∨ v_m = R.m := by
    rw [hvm]; split_ifs
    · exact Or.inr rfl
    · exact Or.inl rfl
  have hΔ : 0 ≤ hi - lo := by linarith [hmL.1, hmL.2]
  have hd1 : |L.m - v_m| ≤ hi - lo := by
    rw [abs_le]; rcases hm' with h | h <;> rw [h] <;> constructor <;> linarith [hmL.1, hmL.2, hmR.1, hmR.2]
  have hd2 : |R.m - v_m| ≤ hi - lo := by
    rw [abs_le]; rcases hm' with h | h <;> rw [h] <;> constructor <;> linarith [hmL.1, hmL.2, hmR.1, hmR.2]
  obtain ⟨p4, a4, b4⟩ := term_step F hs h0 he0 L.d L.m v_m (hi-lo) r2 r3 r4 hdL hd1 h2 h3 h4
  obtain ⟨p7, a7, b7⟩ := term_step F hs h0 he0 R.d R.m v_m (hi-lo) r5 r6 r7 hdR hd2 h5 h6 h7
  have cL := comb hZL hεa hL.1 hL.2 a4 b4
  have cR := comb hZR hεb hR.1 hR.2 a7 b7
  obtain ⟨δ, hδ, hr8⟩ := h8
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  obtain ⟨su, -, -, -, -⟩ := hs
  have fa := fac_bounds hδ (s := F.u) (by linarith)
  have hsum : Real.exp (-(ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u))) * (ZL+ZR) ≤ (r4 + r7) * Real.exp v_m ∧
      (r4 + r7) * Real.exp v_m ≤ Real.exp (ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u)) * (ZL+ZR) := by
    rw [mul_add]
    constructor <;> linarith [cL.1, cL.2, cR.1, cR.2, add_mul r4 r7 (Real.exp v_m)]
  have hr8e : v_d * Real.exp v_m = ((r4 + r7) * Real.exp v_m) * (1 + δ) := by
    rw [hr8]; ring
  have hq : 0 < (r4 + r7) * Real.exp v_m := by positivity
  have hq0 : 0 < ZL + ZR := by linarith
  have ex1 := Real.exp_pos (-(ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u)))
  have ex2 := Real.exp_pos (ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u))
  have ex3 := Real.exp_pos (-(2*F.u))
  have ex4 := Real.exp_pos (2*F.u)
  have lo1 : Real.exp (-(ε + (2*(hi-lo)*F.u + 4*F.u + 2*F.ue))) ≤
      Real.exp (-(ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u) + 2*F.u)) :=
    Real.exp_le_exp.2 (by linarith [mul_nonneg hΔ hu])
  have hi1 : Real.exp (ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u) + 2*F.u) ≤
      Real.exp (ε + (2*(hi-lo)*F.u + 4*F.u + 2*F.ue)) :=
    Real.exp_le_exp.2 (by linarith [mul_nonneg hΔ hu])
  have lowv : Real.exp (-(ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u))) * (ZL+ZR) * Real.exp (-(2*F.u))
      ≤ v_d * Real.exp v_m := by
    rw [hr8e]
    exact mul_le_mul hsum.1 fa.1 ex3.le hq.le
  have upv : v_d * Real.exp v_m ≤ Real.exp (ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u)) * (ZL+ZR) * Real.exp (2*F.u) := by
    rw [hr8e]
    exact mul_le_mul hsum.2 fa.2 (by linarith [fa.1]) (by positivity)
  have e1 : Real.exp (-(ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u) + 2*F.u)) =
      Real.exp (-(ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u))) * Real.exp (-(2*F.u)) := by
    rw [← Real.exp_add]; congr 1; ring
  have e2 : Real.exp (ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u) + 2*F.u) =
      Real.exp (ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u)) * Real.exp (2*F.u) := by
    rw [← Real.exp_add]
  have hvpos : 0 < v_d * Real.exp v_m := by
    refine lt_of_lt_of_le ?_ lowv
    positivity
  refine ⟨?_, hm', ?_, ?_⟩
  · exact (mul_pos_iff_of_pos_right (Real.exp_pos _)).1 hvpos
  · calc _ ≤ Real.exp (-(ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u) + 2*F.u)) * (ZL+ZR) :=
          mul_le_mul_of_nonneg_right lo1 hq0.le
      _ = Real.exp (-(ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u))) * (ZL+ZR) * Real.exp (-(2*F.u)) := by
          rw [e1]; ring
      _ ≤ _ := lowv
  · calc _ ≤ Real.exp (ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u)) * (ZL+ZR) * Real.exp (2*F.u) := upv
      _ = Real.exp (ε + ((hi-lo)*F.u + 2*F.ue + 2*F.u) + 2*F.u) * (ZL+ZR) := by
          rw [e2]; ring
      _ ≤ _ := mul_le_mul_of_nonneg_right hi1 hq0.le

theorem sum_pos_of_ne (g : ℝ → ℝ) (hg : ∀ x, 0 < g x) :
    ∀ {l : List ℝ}, l ≠ [] → 0 < (l.map g).sum := by
  intro l h
  cases l with
  | nil => exact absurd rfl h
  | cons a l =>
    have h1 := hg a
    have h2 : 0 ≤ (l.map g).sum := by
      apply List.sum_nonneg
      intro y hy
      obtain ⟨z, _, rfl⟩ := List.mem_map.1 hy
      exact (hg z).le
    simp only [List.map_cons, List.sum_cons]
    linarith

theorem leaf_step (F : Discovery.Rounding) (hs : small F) (he0 : F.ηe = 0) (D : ℕ)
    (tau lo hi : ℝ) (A : List ℝ) (hA : A ≠ []) (hr : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi)
    (st : State_a) (h : fblock_a F D tau A st) :
    0 < st.d ∧ st.m = lmax A ∧
      Real.exp (-(2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue)) * (A.map Real.exp).sum ≤ st.d * Real.exp st.m ∧
      st.d * Real.exp st.m ≤ Real.exp (2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) * (A.map Real.exp).sum := by
  obtain ⟨v_m, v_d, r1, r2, hr1, hvm, hsum, hvd, hst⟩ := h
  subst hvm
  subst hr1
  subst v_d
  subst hst
  simp only
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  obtain ⟨su, sue, -, -, -⟩ := hs
  have hmem := lmax_mem hA
  have hmr := hr _ hmem
  have hΔ : 0 ≤ hi - lo := by linarith [hmr.1, hmr.2]
  have hΔx : ∀ x ∈ A, |x - lmax A| ≤ hi - lo := by
    intro x hx
    have h1 := hr x hx
    have h2 := le_lmax hx
    rw [abs_le]; constructor <;> linarith [hmr.1, hmr.2]
  have hb := fsum_bounds F D (fun x y => ∃ q1 q2, F.add (x - lmax A) q1 ∧ F.exp q1 q2 ∧ y = q2)
    (fun x => Real.exp ((x - lmax A) - |x - lmax A| * F.u) * (1 - F.ue))
    (fun x => Real.exp ((x - lmax A) + |x - lmax A| * F.u) * (1 + F.ue))
    (fun x => mul_nonneg (Real.exp_pos _).le (by linarith))
    (fun x y h => term_bound F he0 (lmax A) x y h) A r2 hsum
  set m := lmax A with hm
  set S := (A.map fun x => Real.exp (x - m)).sum with hS
  have hSpos : 0 < S := sum_pos_of_ne (fun x => Real.exp (x - m)) (fun x => Real.exp_pos _) hA
  have hZ : (A.map Real.exp).sum = Real.exp m * S := by
    have : A.map Real.exp = A.map (fun x => Real.exp m * Real.exp (x - m)) :=
      List.map_congr_left (fun x _ => by rw [← Real.exp_add]; congr 1; ring)
    rw [this, List.sum_map_mul_left]
  have f1 := fac_bounds (θ := -F.ue) (s := F.ue) (by rw [abs_neg, abs_of_nonneg hue]) (by linarith)
  have f2 := fac_bounds (θ := F.ue) (s := F.ue) (by rw [abs_of_nonneg hue]) (by linarith)
  have f3 := fac_bounds (θ := -F.u) (s := F.u) (by rw [abs_neg, abs_of_nonneg hu]) (by linarith)
  have hlo : Real.exp (-((hi-lo)*F.u + 2*F.ue)) * S ≤
      (A.map fun x => Real.exp ((x - m) - |x - m| * F.u) * (1 - F.ue)).sum := by
    rw [hS, ← List.sum_map_mul_left]
    apply List.sum_le_sum
    intro x hx
    have a1 : Real.exp (-((hi-lo)*F.u)) * Real.exp (x - m) ≤ Real.exp ((x - m) - |x - m| * F.u) := by
      rw [← Real.exp_add]; apply Real.exp_le_exp.2
      nlinarith [mul_le_mul_of_nonneg_right (hΔx x hx) hu]
    have e : Real.exp (-((hi-lo)*F.u + 2*F.ue)) * Real.exp (x - m) =
        (Real.exp (-((hi-lo)*F.u)) * Real.exp (x - m)) * Real.exp (-(2*F.ue)) := by
      rw [neg_add, Real.exp_add]; ring
    rw [e]
    exact mul_le_mul a1 (by linarith [f1.1]) (Real.exp_pos _).le (Real.exp_pos _).le
  have hhi : (A.map fun x => Real.exp ((x - m) + |x - m| * F.u) * (1 + F.ue)).sum ≤
      Real.exp ((hi-lo)*F.u + 2*F.ue) * S := by
    rw [hS, ← List.sum_map_mul_left]
    apply List.sum_le_sum
    intro x hx
    have a1 : Real.exp ((x - m) + |x - m| * F.u) ≤ Real.exp ((hi-lo)*F.u) * Real.exp (x - m) := by
      rw [← Real.exp_add]; apply Real.exp_le_exp.2
      nlinarith [mul_le_mul_of_nonneg_right (hΔx x hx) hu]
    have e : Real.exp ((hi-lo)*F.u + 2*F.ue) * Real.exp (x - m) =
        (Real.exp ((hi-lo)*F.u) * Real.exp (x - m)) * Real.exp (2*F.ue) := by
      rw [Real.exp_add]; ring
    rw [e]
    exact mul_le_mul a1 f2.2 (by linarith) (by positivity)
  have hpl : Real.exp (-(2*F.u)) ^ D ≤ (1 - F.u) ^ D :=
    pow_le_pow_left₀ (Real.exp_pos _).le (by linarith [f3.1]) D
  have hpu : (1 + F.u) ^ D ≤ Real.exp (F.u) ^ D :=
    pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp F.u]) D
  rw [← Real.exp_nat_mul] at hpl hpu
  have hD0 : (0:ℝ) ≤ D := Nat.cast_nonneg D
  have hlow : Real.exp (-(2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue)) * S ≤ r2 := by
    have h1 : Real.exp ((D:ℝ) * -(2*F.u)) * (Real.exp (-((hi-lo)*F.u + 2*F.ue)) * S) ≤ r2 := by
      refine le_trans ?_ hb.1
      exact mul_le_mul hpl hlo (mul_pos (Real.exp_pos _) hSpos).le
        (pow_nonneg (by linarith) _)
    have h2 : Real.exp (-(2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue)) ≤
        Real.exp ((D:ℝ) * -(2*F.u)) * Real.exp (-((hi-lo)*F.u + 2*F.ue)) := by
      rw [← Real.exp_add]; apply Real.exp_le_exp.2
      nlinarith [mul_nonneg hΔ hu]
    nlinarith [mul_le_mul_of_nonneg_right h2 hSpos.le]
  have hup : r2 ≤ Real.exp (2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) * S := by
    have h1 : r2 ≤ Real.exp ((D:ℝ) * F.u) * (Real.exp ((hi-lo)*F.u + 2*F.ue) * S) := by
      refine le_trans hb.2 ?_
      exact mul_le_mul hpu hhi (List.sum_nonneg (fun y hy => by
          obtain ⟨z, _, rfl⟩ := List.mem_map.1 hy
          exact mul_nonneg (Real.exp_pos _).le (by linarith))) (Real.exp_pos _).le
    have h2 : Real.exp ((D:ℝ) * F.u) * Real.exp ((hi-lo)*F.u + 2*F.ue) ≤
        Real.exp (2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) := by
      rw [← Real.exp_add]; apply Real.exp_le_exp.2
      nlinarith [mul_nonneg hΔ hu, mul_nonneg hD0 hu]
    nlinarith [mul_le_mul_of_nonneg_right h2 hSpos.le]
  have hexp := Real.exp_pos (-(2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue))
  have hr2pos : 0 < r2 := lt_of_lt_of_le (mul_pos hexp hSpos) hlow
  refine ⟨hr2pos, trivial, ?_, ?_⟩
  · rw [hZ]; nlinarith [mul_le_mul_of_nonneg_right hlow (Real.exp_pos m).le]
  · rw [hZ]; nlinarith [mul_le_mul_of_nonneg_right hup (Real.exp_pos m).le]

theorem flat_ne : ∀ t : Discovery.Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → t.flat ≠ []
  | .leaf b, h => by simpa [Tree.flat] using h b (by simp [Tree.leaves])
  | .node l r, h => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => h b (by simp [Tree.leaves, hb])
    simp only [Tree.flat]
    exact List.append_ne_nil_of_left_ne_nil (flat_ne l hl) _

theorem main_ind (F : Discovery.Rounding) (hs : small F) (h0 : F.η = 0) (he0 : F.ηe = 0)
    (D : ℕ) (tau lo hi : ℝ) :
    ∀ t : Discovery.Tree ℝ, (∀ x ∈ t.flat, lo ≤ x ∧ x ≤ hi) → (∀ b ∈ t.leaves, b ≠ []) →
    ∀ st, Tree.FEval (fblock_a F D tau) (fmerge_a F tau) t st →
      0 < st.d ∧ (lo ≤ st.m ∧ st.m ≤ hi) ∧
      Real.exp (-((2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) + (t.depth:ℝ) * (2*(hi-lo)*F.u + 4*F.u + 2*F.ue)))
        * (t.flat.map Real.exp).sum ≤ st.d * Real.exp st.m ∧
      st.d * Real.exp st.m ≤
        Real.exp ((2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) + (t.depth:ℝ) * (2*(hi-lo)*F.u + 4*F.u + 2*F.ue))
        * (t.flat.map Real.exp).sum := by
  intro t
  induction t with
  | leaf b =>
    intro hr hl st hev
    cases hev with
    | leaf hf =>
      have hne : b ≠ [] := hl b (by simp [Tree.leaves])
      have hr' : ∀ x ∈ b, lo ≤ x ∧ x ≤ hi := hr
      obtain ⟨p, hm, a, c⟩ := leaf_step F hs he0 D tau lo hi b hne hr' st hf
      have hmr := hr' _ (lmax_mem hne)
      rw [hm]
      simp only [Tree.depth, Tree.flat, Nat.cast_zero, zero_mul, add_zero]
      exact ⟨p, hmr, by rw [← hm]; exact a, by rw [← hm]; exact c⟩
  | node l r ihl ihr =>
    intro hr hl st hev
    cases hev with
    | node e1 e2 hm =>
      rename_i a c
      have hll : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => hl b (by simp [Tree.leaves, hb])
      have hlr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => hl b (by simp [Tree.leaves, hb])
      have hrl : ∀ x ∈ l.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hr x (by simp [Tree.flat, hx])
      have hrr : ∀ x ∈ r.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hr x (by simp [Tree.flat, hx])
      obtain ⟨pL, mL, aL, bL⟩ := ihl hrl hll a e1
      obtain ⟨pR, mR, aR, bR⟩ := ihr hrr hlr c e2
      have zL := sum_pos_of_ne Real.exp (fun x => Real.exp_pos x) (flat_ne l hll)
      have zR := sum_pos_of_ne Real.exp (fun x => Real.exp_pos x) (flat_ne r hlr)
      have hu := F.hu
      have hue := F.hue
      have hΔ : 0 ≤ hi - lo := by linarith [mL.1, mL.2]
      have hc : 0 ≤ 2*(hi-lo)*F.u + 4*F.u + 2*F.ue := by
        have := mul_nonneg hΔ hu
        linarith
      have hdl : ((l.depth : ℕ) : ℝ) ≤ max (l.depth : ℝ) (r.depth : ℝ) := le_max_left _ _
      have hdr : ((r.depth : ℕ) : ℝ) ≤ max (l.depth : ℝ) (r.depth : ℝ) := le_max_right _ _
      have := merge_step F hs h0 he0 tau lo hi a c st _ _
        ((2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) + (l.depth:ℝ) * (2*(hi-lo)*F.u + 4*F.u + 2*F.ue))
        ((2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) + (r.depth:ℝ) * (2*(hi-lo)*F.u + 4*F.u + 2*F.ue))
        ((2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) + (max (l.depth : ℝ) (r.depth : ℝ)) * (2*(hi-lo)*F.u + 4*F.u + 2*F.ue))
        zL zR pL pR mL mR
        (by nlinarith [mul_le_mul_of_nonneg_right hdl hc])
        (by nlinarith [mul_le_mul_of_nonneg_right hdr hc])
        ⟨by linarith [aL], by linarith [bL]⟩
        ⟨by linarith [aR], by linarith [bR]⟩ hm
      obtain ⟨p, mm, A1, A2⟩ := this
      have hmr : lo ≤ st.m ∧ st.m ≤ hi := by
        rcases mm with h | h <;> rw [h]
        · exact mL
        · exact mR
      have hd : ((Tree.node l r).depth : ℝ) = max (l.depth : ℝ) (r.depth : ℝ) + 1 := by
        simp only [Tree.depth]; push_cast; rfl
      have hfl : (Tree.node l r).flat = l.flat ++ r.flat := rfl
      rw [hd, hfl, List.map_append, List.sum_append]
      refine ⟨p, hmr, ?_, ?_⟩
      · refine le_trans (le_of_eq ?_) A1; congr 2; ring
      · refine le_trans A2 (le_of_eq ?_); congr 2; ring

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (tau : ℝ), 0 ≤ tau → ∀ (lo hi : ℝ) (X : List ℝ),
    X ≠ [] → (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    (X.length : ℝ) * Real.exp (tau * (rdepth t : ℝ)) * (F.η + F.ηe) ≤ F.u →
    F.η = 0 → F.ηe = 0 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D tau) (fmerge_a F tau) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((2 : ℝ) * (t.depth : ℝ) * (hi - lo) * F.u) + ((2 : ℝ) * (hi - lo) * F.u) + ((2 : ℝ) * (D : ℝ) * F.u) + ((2 : ℝ) * F.ue) + ((4 : ℝ) * (t.depth : ℝ) * F.u) + ((2 : ℝ) * (t.depth : ℝ) * F.ue)) := by
  intro F D tau htau lo hi X hX hrange hsmall hlen t hflat hl _ _ h0 he0 st hev
  subst hflat
  obtain ⟨p, _, A1, A2⟩ := main_ind F hsmall h0 he0 D tau lo hi t hrange hl st hev
  have hzn : znorm st = st.d * Real.exp st.m := by
    unfold znorm
    rw [zero_sub, Real.exp_neg]
    have := Real.exp_pos st.m
    field_simp
  have hZ := sum_pos_of_ne Real.exp (fun x => Real.exp_pos x) hX
  have hw : 0 < st.d * Real.exp st.m := mul_pos p (Real.exp_pos _)
  rw [hzn]
  set E := (2*(hi-lo)*F.u + 2*(D:ℝ)*F.u + 2*F.ue) + (t.depth:ℝ) * (2*(hi-lo)*F.u + 4*F.u + 2*F.ue)
    with hE
  have l1 := Real.log_le_log (mul_pos (Real.exp_pos _) hZ) A1
  have l2 := Real.log_le_log hw A2
  rw [Real.log_mul (Real.exp_pos _).ne' hZ.ne', Real.log_exp] at l1
  rw [Real.log_mul (Real.exp_pos _).ne' hZ.ne', Real.log_exp] at l2
  have : E = ((2 : ℝ) * (t.depth : ℝ) * (hi - lo) * F.u) + ((2 : ℝ) * (hi - lo) * F.u) + ((2 : ℝ) * (D : ℝ) * F.u) + ((2 : ℝ) * F.ue) + ((4 : ℝ) * (t.depth : ℝ) * F.u) + ((2 : ℝ) * (t.depth : ℝ) * F.ue) := by
    rw [hE]; ring
  rw [← this, abs_le]
  constructor <;> linarith

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxConditionalFpNounderflow
