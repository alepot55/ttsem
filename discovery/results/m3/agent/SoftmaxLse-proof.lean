import FPStatements

namespace Discovery.FP.SoftmaxLse

open Discovery Discovery.Library.SoftmaxLse

theorem h_log_ratio {P Q θ : ℝ} (hP : 0 < P) (hQ : 0 < Q) (hθ : 0 ≤ θ) (hθ1 : θ < 1)
    (h1 : (1 - θ) * P ≤ Q) (h2 : Q ≤ (1 + θ) * P) :
    |Real.log Q - Real.log P| ≤ θ / (1 - θ) := by
  have h1θ : 0 < 1 - θ := by linarith
  have hθd : θ ≤ θ / (1 - θ) := by
    rw [le_div_iff₀ h1θ]; nlinarith
  rw [abs_le]
  constructor
  · have : Real.log (P / Q) ≤ P / Q - 1 := Real.log_le_sub_one_of_pos (div_pos hP hQ)
    rw [Real.log_div hP.ne' hQ.ne'] at this
    have h3 : P / Q - 1 ≤ θ / (1 - θ) := by
      rw [div_sub_one hQ.ne', div_le_div_iff₀ hQ h1θ]
      nlinarith
    linarith
  · have : Real.log (Q / P) ≤ Q / P - 1 := Real.log_le_sub_one_of_pos (div_pos hQ hP)
    rw [Real.log_div hQ.ne' hP.ne'] at this
    have h3 : Q / P - 1 ≤ θ := by
      rw [div_sub_one hP.ne', div_le_iff₀ hP]
      nlinarith
    linarith

theorem h_ye_le {y : ℝ} (hy : 0 ≤ y) : y * Real.exp (-y) ≤ 1 / 2 := by
  have h := Real.quadratic_le_exp_of_nonneg hy
  rw [Real.exp_neg, ← div_eq_mul_inv, div_le_iff₀ (Real.exp_pos y)]
  nlinarith [sq_nonneg (y - 1)]

theorem h_abs_exp_sub_one (w : ℝ) : |Real.exp w - 1| ≤ |w| * Real.exp |w| := by
  have hw := Real.add_one_le_exp (-w)
  have hw' := Real.add_one_le_exp w
  have hpos := Real.exp_pos w
  have hpos' := Real.exp_pos (-w)
  have hinv : Real.exp w * Real.exp (-w) = 1 := by rw [← Real.exp_add]; simp
  rcases le_total 0 w with h | h
  · rw [abs_of_nonneg h, abs_le]
    have : Real.exp w * (1 - w) ≤ 1 := by nlinarith
    constructor
    · nlinarith [Real.add_one_le_exp w, Real.exp_pos w]
    · nlinarith
  · rw [abs_of_nonpos h, abs_le]
    have h1 : Real.exp (-w) ≥ 1 := by
      have := Real.one_le_exp (neg_nonneg.mpr h); exact this
    constructor
    · nlinarith [Real.exp_pos w]
    · nlinarith [Real.exp_pos w]

theorem h_exp_diff {ℓ δ u : ℝ} (hℓ : ℓ ≤ 0) (hδ : |δ| ≤ u) (hu : u ≤ 1 / 256) :
    |Real.exp (ℓ * (1 + δ)) - Real.exp ℓ| ≤ u := by
  have hu0 : 0 ≤ u := le_trans (abs_nonneg _) hδ
  set z := -ℓ with hz
  have hz0 : 0 ≤ z := by linarith
  have hw : |ℓ * δ| ≤ z * u := by
    rw [abs_mul, abs_of_nonpos hℓ]
    nlinarith [abs_nonneg δ]
  have hexp : Real.exp (ℓ * (1 + δ)) = Real.exp ℓ * Real.exp (ℓ * δ) := by
    rw [← Real.exp_add]; ring_nf
  have key : Real.exp (ℓ * (1 + δ)) - Real.exp ℓ = Real.exp ℓ * (Real.exp (ℓ * δ) - 1) := by
    rw [hexp]; ring
  rw [key, abs_mul, abs_of_pos (Real.exp_pos ℓ)]
  have h1 := h_abs_exp_sub_one (ℓ * δ)
  have h2 : |ℓ * δ| * Real.exp |ℓ * δ| ≤ z * u * Real.exp (z * u) := by
    have := Real.exp_le_exp.mpr hw
    exact mul_le_mul hw this (Real.exp_pos _).le (by positivity)
  have h3 : Real.exp ℓ * (z * u * Real.exp (z * u)) = u * (z * Real.exp (-((1 - u) * z))) := by
    have : Real.exp ℓ * Real.exp (z * u) = Real.exp (-((1 - u) * z)) := by
      rw [← Real.exp_add]; congr 1; rw [hz]; ring
    calc Real.exp ℓ * (z * u * Real.exp (z * u)) = u * z * (Real.exp ℓ * Real.exp (z * u)) := by ring
      _ = _ := by rw [this]; ring
  have h4 : z * Real.exp (-((1 - u) * z)) ≤ 1 := by
    have hy := h_ye_le (y := (1 - u) * z) (by nlinarith)
    have : z ≤ 256 / 255 * ((1 - u) * z) := by nlinarith
    calc z * Real.exp (-((1 - u) * z)) ≤ 256 / 255 * ((1 - u) * z) * Real.exp (-((1 - u) * z)) := by
          gcongr
      _ = 256 / 255 * (((1 - u) * z) * Real.exp (-((1 - u) * z))) := by ring
      _ ≤ 1 := by linarith
  calc Real.exp ℓ * |Real.exp (ℓ * δ) - 1| ≤ Real.exp ℓ * (z * u * Real.exp (z * u)) := by
        gcongr; exact h1.trans h2
    _ = _ := h3
    _ ≤ u * 1 := by gcongr
    _ = u := by ring


theorem h_lse_lip {L R α β E : ℝ} (hL : |L - α| ≤ E) (hR : |R - β| ≤ E) :
    |Real.log (Real.exp L + Real.exp R) - Real.log (Real.exp α + Real.exp β)| ≤ E := by
  have hp : 0 < Real.exp α + Real.exp β := by positivity
  have hq : 0 < Real.exp L + Real.exp R := by positivity
  rw [abs_le] at *
  have e1 : Real.exp L ≤ Real.exp E * Real.exp α := by
    rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith [hL.2])
  have e2 : Real.exp R ≤ Real.exp E * Real.exp β := by
    rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith [hR.2])
  have e3 : Real.exp α ≤ Real.exp E * Real.exp L := by
    rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith [hL.1])
  have e4 : Real.exp β ≤ Real.exp E * Real.exp R := by
    rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith [hR.1])
  constructor
  · have : Real.exp α + Real.exp β ≤ Real.exp E * (Real.exp L + Real.exp R) := by nlinarith
    have h := Real.log_le_log hp this
    rw [Real.log_mul (Real.exp_pos E).ne' hq.ne', Real.log_exp] at h
    linarith
  · have : Real.exp L + Real.exp R ≤ Real.exp E * (Real.exp α + Real.exp β) := by nlinarith
    have h := Real.log_le_log hq this
    rw [Real.log_mul (Real.exp_pos E).ne' hp.ne', Real.log_exp] at h
    linarith


theorem h_merge_core (F : Discovery.Rounding) (hu : F.u ≤ 1 / 256) (hue : F.ue ≤ 1 / 64)
    (hul : F.ul ≤ 1 / 64) (hηe : 2 * F.ηe ≤ F.u)
    (μ ℓ r1 r2 r4 r5 r6 r7 : ℝ) (hℓ : ℓ ≤ 0)
    (h1 : F.add ℓ r1) (h2 : F.exp r1 r2) (h4 : F.exp 0 r4) (h5 : F.add (r2 + r4) r5)
    (h6 : F.log r5 r6) (h7 : F.add (μ + r6) r7) :
    |r7 - (μ + Real.log (Real.exp ℓ + 1))| ≤ F.u * (|μ| + 5) + 6 / 5 * F.ue + 11 / 10 * F.ul := by
  have hu0 := F.hu
  have hue0 := F.hue
  have hul0 := F.hul
  have hηe0 := F.hηe
  obtain ⟨δ1, hδ1, e1⟩ := h1
  obtain ⟨δ2, e2, hδ2, he2, hr2, hr2n⟩ := h2
  obtain ⟨δ4, e4, hδ4, he4, hr4, hr4n⟩ := h4
  obtain ⟨δ5, hδ5, hr5⟩ := h5
  obtain ⟨δ6, hδ6, hr6⟩ := h6
  obtain ⟨δ7, hδ7, hr7⟩ := h7
  have hr1n : r1 ≤ 0 := by
    rw [e1]
    have : |δ1| ≤ 1/256 := hδ1.trans hu
    rw [abs_le] at this
    nlinarith
  have hd := h_exp_diff hℓ hδ1 hu
  rw [← e1] at hd
  rw [abs_le] at hd hδ2 he2 hδ4 he4
  have hexpℓ0 : Real.exp ℓ ≤ 1 := Real.exp_le_one_iff.mpr hℓ
  have hexpℓp := Real.exp_pos ℓ
  have hexpr1 : Real.exp r1 ≤ 1 := Real.exp_le_one_iff.mpr hr1n
  have hexpr1p := Real.exp_pos r1
  rw [Real.exp_zero] at hr4
  have hP1 : 1 ≤ Real.exp ℓ + 1 := by linarith
  have hP2 : Real.exp ℓ + 1 ≤ 2 := by linarith
  have a1 : Real.exp r1 * δ2 ≤ F.ue * Real.exp r1 := by nlinarith
  have a2 : -(F.ue * Real.exp r1) ≤ Real.exp r1 * δ2 := by nlinarith
  have a3 : F.ue * Real.exp r1 ≤ F.ue * (Real.exp ℓ + F.u) := by
    apply mul_le_mul_of_nonneg_left _ hue0; linarith
  have c1 : 0 ≤ F.u * Real.exp ℓ := by positivity
  have c2 : 0 ≤ F.ue * F.u * Real.exp ℓ := by positivity
  have c3 : 0 ≤ F.ue * F.u := by positivity
  have hSP : |(r2 + r4) - (Real.exp ℓ + 1)| ≤ (Real.exp ℓ + 1) * (2 * F.u + F.ue + F.ue * F.u) := by
    have eq : r2 + r4 - (Real.exp ℓ + 1) = (Real.exp r1 - Real.exp ℓ) + Real.exp r1 * δ2 + e2 + δ4 + e4 := by
      rw [hr2, hr4]; ring
    rw [abs_le, eq]
    obtain ⟨hd1, hd2⟩ := hd
    constructor <;> linarith
  set P := Real.exp ℓ + 1 with hP
  set S := r2 + r4 with hS
  have hP0 : 0 < P := by linarith
  have huu : F.u * F.u ≤ F.u / 256 := by
    have := mul_le_mul_of_nonneg_left hu hu0; linarith
  have hueu : F.ue * F.u ≤ F.ue / 256 := by
    have := mul_le_mul_of_nonneg_left hu hue0; linarith
  have hueuu : F.ue * F.u * F.u ≤ F.ue / 65536 := by
    have := mul_le_mul_of_nonneg_left hu c3; linarith
  set θ : ℝ := 31 / 10 * F.u + 11 / 10 * F.ue with hθ
  have hθ0 : 0 ≤ θ := by positivity
  have hθ1 : θ ≤ 3 / 100 := by rw [hθ]; linarith
  have hkey : (2 * F.u + F.ue + F.ue * F.u) * (1 + F.u) + F.u ≤ θ := by
    rw [hθ]; linarith
  have h5' : |r5 - P| ≤ θ * P := by
    have eq : r5 - P = (S - P) * (1 + δ5) + δ5 * P := by rw [hr5]; ring
    rw [eq]
    have b1 : |1 + δ5| ≤ 1 + F.u := by
      refine (abs_add_le _ _).trans ?_; rw [abs_one]; linarith
    calc |(S - P) * (1 + δ5) + δ5 * P| ≤ |S - P| * |1 + δ5| + |δ5| * |P| := by
          refine (abs_add_le _ _).trans ?_; rw [abs_mul, abs_mul]
      _ ≤ (P * (2 * F.u + F.ue + F.ue * F.u)) * (1 + F.u) + F.u * P := by
          rw [abs_of_pos hP0]
          gcongr
      _ = P * ((2 * F.u + F.ue + F.ue * F.u) * (1 + F.u) + F.u) := by ring
      _ ≤ P * θ := by gcongr
      _ = θ * P := by ring
  have h5'' := abs_le.mp h5'
  have hθP : 0 < (1 - θ) * P := mul_pos (by linarith) hP0
  have hr5p : 0 < r5 := by linarith [h5''.1]
  have hlog := h_log_ratio hP0 hr5p hθ0 (by linarith) (by linarith [h5''.1]) (by linarith [h5''.2])
  have hΛ : |Real.log r5 - Real.log P| ≤ 33 / 10 * F.u + 23 / 20 * F.ue := by
    refine hlog.trans ?_
    rw [div_le_iff₀ (by linarith)]
    have : θ * (1 - 3 / 100) ≤ θ * (1 - θ) := by
      apply mul_le_mul_of_nonneg_left _ hθ0; linarith
    have h2 : (33 / 10 * F.u + 23 / 20 * F.ue) * (1 - θ) ≥ (33 / 10 * F.u + 23 / 20 * F.ue) * (97/100) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity); linarith
    rw [hθ] at this ⊢
    linarith
  have hΛ' := abs_le.mp hΛ
  have hlP0 : 0 ≤ Real.log P := Real.log_nonneg hP1
  have hlP1 : Real.log P ≤ 1 := by
    have := Real.log_le_sub_one_of_pos hP0
    linarith
  have hlr5 : |Real.log r5| ≤ 21 / 20 := by
    rw [abs_le]; constructor <;> linarith
  have hδ6l : |δ6 * Real.log r5| ≤ F.ul * (21 / 20) := by
    rw [abs_mul]; exact mul_le_mul hδ6 hlr5 (abs_nonneg _) hul0
  have hr6' : |r6 - Real.log P| ≤ 33 / 10 * F.u + 23 / 20 * F.ue + 21 / 20 * F.ul := by
    have eq : r6 - Real.log P = (Real.log r5 - Real.log P) + δ6 * Real.log r5 := by
      rw [hr6]; ring
    rw [eq]
    refine (abs_add_le _ _).trans ?_
    linarith
  have hr6'' := abs_le.mp hr6'
  have hμ : |μ + r6| ≤ |μ| + 21 / 20 := by
    have eq : μ + r6 = μ + (r6 - Real.log P) + Real.log P := by ring
    rw [eq]
    have b1 : |μ + (r6 - Real.log P)| ≤ |μ| + |r6 - Real.log P| := abs_add_le _ _
    have b2 : |μ + (r6 - Real.log P) + Real.log P| ≤ |μ + (r6 - Real.log P)| + |Real.log P| :=
      abs_add_le _ _
    rw [abs_of_nonneg hlP0] at b2
    have : |r6 - Real.log P| ≤ 1 / 20 := by
      refine hr6'.trans ?_; linarith
    linarith
  have hr7' : |r7 - (μ + Real.log P)| ≤ F.u * (|μ| + 5) + 6 / 5 * F.ue + 11 / 10 * F.ul := by
    have eq : r7 - (μ + Real.log P) = (r6 - Real.log P) + δ7 * (μ + r6) := by
      rw [hr7]; ring
    rw [eq]
    have b1 : |δ7 * (μ + r6)| ≤ F.u * (|μ| + 21 / 20) := by
      rw [abs_mul]; exact mul_le_mul hδ7 hμ (abs_nonneg _) hu0
    have b2 := abs_add_le (r6 - Real.log P) (δ7 * (μ + r6))
    linarith
  exact hr7'


theorem h_lse_shift (a b : ℝ) :
    Real.log (Real.exp a + Real.exp b) = a + Real.log (Real.exp (b - a) + 1) := by
  have : Real.exp a + Real.exp b = Real.exp a * (Real.exp (b - a) + 1) := by
    rw [mul_add, ← Real.exp_add]; ring_nf
  rw [this, Real.log_mul (Real.exp_pos a).ne' (by positivity), Real.log_exp]

theorem h_add_zero {F : Discovery.Rounding} {r : ℝ} (h : F.add 0 r) : r = 0 := by
  obtain ⟨δ, _, hr⟩ := h
  rw [hr]; ring

theorem h_merge_step (F : Discovery.Rounding) (hu : F.u ≤ 1 / 256) (hue : F.ue ≤ 1 / 64)
    (hul : F.ul ≤ 1 / 64) (hηe : 2 * F.ηe ≤ F.u) (K E α β : ℝ) (hK : 2 ≤ K) (hE0 : 0 ≤ E)
    (hα : |α| ≤ K) (hβ : |β| ≤ K)
    (hE : F.u * E ≤ (F.u * (K + 2) + F.ue + F.ul) / 4) (x y w : State_a)
    (hx : |x.s - α| ≤ E) (hy : |y.s - β| ≤ E) (hm : fmerge_a F x y w) :
    |w.s - Real.log (Real.exp α + Real.exp β)| ≤ E + 2 * (F.u * (K + 2) + F.ue + F.ul) := by
  have hu0 := F.hu
  have hue0 := F.hue
  have hul0 := F.hul
  obtain ⟨vm, vs, r1, r2, r3, r4, r5, r6, r7, hvm, h1, h2, h3, h4, h5, h6, h7, hvs, hst⟩ := hm
  have hws : w.s = r7 := by rw [hst, hvs]
  rw [hws]
  have hL : |x.s| ≤ K + E := by
    have := abs_sub_abs_le_abs_sub x.s α; linarith
  have hR : |y.s| ≤ K + E := by
    have := abs_sub_abs_le_abs_sub y.s β; linarith
  have hlip := h_lse_lip hx hy
  have hfinal : ∀ ρ μ : ℝ, |μ| ≤ K + E → |ρ - Real.log (Real.exp α + Real.exp β)| ≤ E →
      |r7 - ρ| ≤ F.u * (|μ| + 5) + 6 / 5 * F.ue + 11 / 10 * F.ul →
      |r7 - Real.log (Real.exp α + Real.exp β)| ≤ E + 2 * (F.u * (K + 2) + F.ue + F.ul) := by
    intro ρ μ hμ h1 h2
    have b1 := abs_add_le (r7 - ρ) (ρ - Real.log (Real.exp α + Real.exp β))
    have b2 : F.u * |μ| ≤ F.u * (K + E) := mul_le_mul_of_nonneg_left hμ hu0
    have b3 : F.u * 2 ≤ F.u * K := mul_le_mul_of_nonneg_left hK hu0
    have : r7 - ρ + (ρ - Real.log (Real.exp α + Real.exp β)) = r7 - Real.log (Real.exp α + Real.exp β) := by ring
    rw [this] at b1
    linarith
  rcases le_total y.s x.s with hle | hle
  · rw [max_eq_left hle] at hvm
    subst hvm
    rw [sub_self] at h1
    have hr1 := h_add_zero h1
    subst hr1
    have hcore := h_merge_core F hu hue hul hηe x.s (y.s - x.s) r3 r4 r2 r5 r6 r7 (by linarith)
      h3 h4 h2 (by rw [add_comm]; exact h5) h6 h7
    have hρ := h_lse_shift x.s y.s
    refine hfinal (Real.log (Real.exp x.s + Real.exp y.s)) x.s hL ?_ ?_
    · exact hlip
    · rw [hρ]; exact hcore
  · rw [max_eq_right hle] at hvm
    subst hvm
    rw [sub_self] at h3
    have hr3 := h_add_zero h3
    subst hr3
    have hcore := h_merge_core F hu hue hul hηe y.s (x.s - y.s) r1 r2 r4 r5 r6 r7 (by linarith)
      h1 h2 h4 h5 h6 h7
    have hρ := h_lse_shift y.s x.s
    refine hfinal (Real.log (Real.exp x.s + Real.exp y.s)) y.s hR ?_ ?_
    · exact hlip
    · rw [add_comm (Real.exp x.s), hρ]; exact hcore


theorem h_fsum_two_sided (F : Discovery.Rounding) (hu : F.u ≤ 1 / 256) (hue : F.ue ≤ 1 / 64)
    (D : ℕ) (lo hi : ℝ) (A : List ℝ) (m : ℝ) (hmA : m ∈ A)
    (hxm : ∀ x ∈ A, x ≤ m ∧ lo ≤ x) (hmhi : m ≤ hi)
    (hk : (A.length : ℝ) * F.ηe ≤ F.u) (r2 : ℝ)
    (hs : Discovery.fsum F D (fun x y => ∃ q1 q2, F.add (x - m) q1 ∧ F.exp q1 q2 ∧ y = q2) A r2) :
    (1 - F.u) ^ D * (1 - F.ue - F.u) * Real.exp (-(F.u * (hi - lo))) *
        (A.map fun x => Real.exp (x - m)).sum ≤ r2 ∧
    r2 ≤ (1 + F.u) ^ D * Real.exp (F.u * (hi - lo)) * (1 + F.ue + F.u) *
        (A.map fun x => Real.exp (x - m)).sum := by
  have hu0 := F.hu
  have hue0 := F.hue
  have hη0 := F.hηe
  have hmlo : lo ≤ m := (hxm m hmA).2
  have hab : ∀ x ∈ A, |x - m| ≤ hi - lo := by
    intro x hx
    obtain ⟨h1, h2⟩ := hxm x hx
    rw [abs_of_nonpos (by linarith)]; linarith
  have hua : ∀ x ∈ A, F.u * |x - m| ≤ F.u * (hi - lo) := fun x hx =>
    mul_le_mul_of_nonneg_left (hab x hx) hu0
  set a := F.u * (hi - lo) with ha
  set T := (A.map fun x => Real.exp (x - m)).sum with hT
  have hT1 : 1 ≤ T := by
    have := List.single_le_sum (l := A.map fun x => Real.exp (x - m))
      (by intro x hx; simp only [List.mem_map] at hx; obtain ⟨y, _, rfl⟩ := hx; positivity)
      (Real.exp (m - m)) (List.mem_map.mpr ⟨m, hmA, rfl⟩)
    simpa using this
  have hterm : ∀ x y, (∃ q1 q2, F.add (x - m) q1 ∧ F.exp q1 q2 ∧ y = q2) →
      max 0 ((1 - F.ue) * Real.exp (x - m - F.u * |x - m|) - F.ηe) ≤ y ∧
      y ≤ (1 + F.ue) * Real.exp (x - m + F.u * |x - m|) + F.ηe := by
    intro x y ⟨q1, q2, h1, h2, hy⟩
    subst hy
    obtain ⟨δ, hδ, hq1⟩ := h1
    obtain ⟨b1, b2, b3⟩ := F.exp_bounds h2
    have hd : |(x - m) * δ| ≤ F.u * |x - m| := by
      rw [abs_mul, mul_comm]; exact mul_le_mul_of_nonneg_right hδ (abs_nonneg _)
    rw [abs_le] at hd
    have hq1' : q1 = x - m + (x - m) * δ := by rw [hq1]; ring
    constructor
    · refine max_le b3 ?_
      have : Real.exp (x - m - F.u * |x - m|) ≤ Real.exp q1 :=
        Real.exp_le_exp.mpr (by linarith)
      have := mul_le_mul_of_nonneg_left this (by linarith : 0 ≤ 1 - F.ue)
      linarith
    · have : Real.exp q1 ≤ Real.exp (x - m + F.u * |x - m|) :=
        Real.exp_le_exp.mpr (by linarith)
      have := mul_le_mul_of_nonneg_left this (by linarith : 0 ≤ 1 + F.ue)
      linarith
  obtain ⟨h1, h2⟩ := Discovery.fsum_bounds F D _
    (fun x => max 0 ((1 - F.ue) * Real.exp (x - m - F.u * |x - m|) - F.ηe))
    (fun x => (1 + F.ue) * Real.exp (x - m + F.u * |x - m|) + F.ηe)
    (fun x => le_max_left _ _) hterm A r2 hs
  have hpow1 : 0 ≤ (1 - F.u) ^ D := pow_nonneg (by linarith) D
  have hpow2 : 0 ≤ (1 + F.u) ^ D := pow_nonneg (by linarith) D
  constructor
  · -- lower
    have hsumlo : (1 - F.ue) * (A.map fun x => Real.exp (x - m - F.u * |x - m|)).sum - A.length * F.ηe
        ≤ (A.map fun x => max 0 ((1 - F.ue) * Real.exp (x - m - F.u * |x - m|) - F.ηe)).sum := by
      have := List.sum_le_sum (l := A)
        (f := fun x => (1 - F.ue) * Real.exp (x - m - F.u * |x - m|) + (-F.ηe))
        (g := fun x => max 0 ((1 - F.ue) * Real.exp (x - m - F.u * |x - m|) - F.ηe))
        (fun x _ => by
          have := le_max_right 0 ((1 - F.ue) * Real.exp (x - m - F.u * |x - m|) - F.ηe)
          simp only [← sub_eq_add_neg]; exact this)
      refine le_trans (le_of_eq ?_) this
      rw [List.sum_map_add, List.sum_map_mul_left]
      simp
      ring
    set G := (A.map fun x => Real.exp (x - m - F.u * |x - m|)).sum with hG
    have hG1 : 1 ≤ G := by
      have := List.single_le_sum (l := A.map fun x => Real.exp (x - m - F.u * |x - m|))
        (by intro x hx; simp only [List.mem_map] at hx; obtain ⟨y, _, rfl⟩ := hx; positivity)
        (Real.exp (m - m - F.u * |m - m|)) (List.mem_map.mpr ⟨m, hmA, rfl⟩)
      simpa using this
    have hG2 : Real.exp (-a) * T ≤ G := by
      rw [hT, ← List.sum_map_mul_left]
      refine List.sum_le_sum (fun x hx => ?_)
      rw [← Real.exp_add]
      apply Real.exp_le_exp.mpr
      have := hua x hx
      linarith
    have hR : (1 - F.ue - F.u) * Real.exp (-a) * T ≤
        (A.map fun x => max 0 ((1 - F.ue) * Real.exp (x - m - F.u * |x - m|) - F.ηe)).sum := by
      refine le_trans ?_ hsumlo
      have hpos : 0 < 1 - F.ue - F.u := by linarith
      rcases le_total 1 (Real.exp (-a) * T) with hc | hc
      · have := mul_le_mul_of_nonneg_left hG2 (by linarith : 0 ≤ 1 - F.ue)
        have h3 : F.u * 1 ≤ F.u * (Real.exp (-a) * T) := mul_le_mul_of_nonneg_left hc hu0
        nlinarith
      · have := mul_le_mul_of_nonneg_left hG1 (by linarith : 0 ≤ 1 - F.ue)
        have h3 : (1 - F.ue - F.u) * (Real.exp (-a) * T) ≤ (1 - F.ue - F.u) * 1 :=
          mul_le_mul_of_nonneg_left hc hpos.le
        nlinarith
    calc (1 - F.u) ^ D * (1 - F.ue - F.u) * Real.exp (-a) * T
        = (1 - F.u) ^ D * ((1 - F.ue - F.u) * Real.exp (-a) * T) := by ring
      _ ≤ _ := le_trans (mul_le_mul_of_nonneg_left hR hpow1) h1
  · -- upper
    have hsumhi : (A.map fun x => (1 + F.ue) * Real.exp (x - m + F.u * |x - m|) + F.ηe).sum
        ≤ (1 + F.ue) * Real.exp a * T + A.length * F.ηe := by
      have := List.sum_le_sum (l := A)
        (f := fun x => (1 + F.ue) * Real.exp (x - m + F.u * |x - m|) + F.ηe)
        (g := fun x => (1 + F.ue) * Real.exp a * Real.exp (x - m) + F.ηe)
        (fun x hx => by
          have : Real.exp (x - m + F.u * |x - m|) ≤ Real.exp a * Real.exp (x - m) := by
            rw [← Real.exp_add]; apply Real.exp_le_exp.mpr; have := hua x hx; linarith
          have := mul_le_mul_of_nonneg_left this (by linarith : 0 ≤ 1 + F.ue)
          linarith)
      refine le_trans this (le_of_eq ?_)
      rw [List.sum_map_add, List.sum_map_mul_left]
      simp [hT]
    have hea : 1 ≤ Real.exp a := Real.one_le_exp (by rw [ha]; nlinarith [hab m hmA])
    have hR : (A.map fun x => (1 + F.ue) * Real.exp (x - m + F.u * |x - m|) + F.ηe).sum
        ≤ (1 + F.ue + F.u) * Real.exp a * T := by
      refine le_trans hsumhi ?_
      have h3 : F.u * 1 ≤ F.u * (Real.exp a * T) := by
        apply mul_le_mul_of_nonneg_left _ hu0; nlinarith
      nlinarith
    calc r2 ≤ _ := h2
      _ ≤ (1 + F.u) ^ D * ((1 + F.ue + F.u) * Real.exp a * T) :=
          mul_le_mul_of_nonneg_left hR hpow2
      _ = _ := by ring


theorem h_log_one_sub {y : ℝ} (h0 : 0 ≤ y) (h1 : y ≤ 5 / 256) :
    -(51 / 50 * y) ≤ Real.log (1 - y) := by
  have hp : 0 < 1 - y := by linarith
  have h := Real.log_le_sub_one_of_pos (inv_pos.mpr hp)
  rw [Real.log_inv] at h
  have h2 : (1 - y)⁻¹ ≤ 1 + 51 / 50 * y := by
    rw [← one_div, div_le_iff₀ hp]; nlinarith
  linarith

theorem h_log_f (u ue a T r2 : ℝ) (D : ℕ) (hu0 : 0 ≤ u) (hu : u ≤ 1 / 256) (hue0 : 0 ≤ ue)
    (hue : ue ≤ 1 / 64) (ha : 0 ≤ a) (hT : 1 ≤ T)
    (h1 : (1 - u) ^ D * (1 - ue - u) * Real.exp (-a) * T ≤ r2)
    (h2 : r2 ≤ (1 + u) ^ D * Real.exp a * (1 + ue + u) * T) :
    0 < r2 ∧ |Real.log r2 - Real.log T| ≤ 51 / 50 * (D * u + a + ue + u) := by
  have hTp : 0 < T := by linarith
  have hV : 0 < (1 - u) ^ D * (1 - ue - u) * Real.exp (-a) := by
    have : 0 < 1 - ue - u := by linarith
    have : 0 < 1 - u := by linarith
    positivity
  have hr : 0 < r2 := lt_of_lt_of_le (mul_pos hV hTp) h1
  refine ⟨hr, ?_⟩
  have hW : 0 < (1 + u) ^ D * Real.exp a * (1 + ue + u) := by positivity
  rw [abs_le]
  constructor
  · have := Real.log_le_log (mul_pos hV hTp) h1
    have h1u : 0 < 1 - u := by linarith
    have h1p : 0 < 1 - ue - u := by linarith
    rw [Real.log_mul hV.ne' hTp.ne', Real.log_mul (mul_pos (pow_pos h1u D) h1p).ne'
      (Real.exp_pos _).ne',
      Real.log_mul (pow_pos h1u D).ne' h1p.ne', Real.log_pow, Real.log_exp] at this
    have e1 := h_log_one_sub hu0 (by linarith : u ≤ 5 / 256)
    have e2 := h_log_one_sub (by linarith : 0 ≤ ue + u) (by linarith : ue + u ≤ 5 / 256)
    have e3 : 1 - ue - u = 1 - (ue + u) := by ring
    rw [e3] at this
    have e4 : (D : ℝ) * (-(51 / 50 * u)) ≤ D * Real.log (1 - u) :=
      mul_le_mul_of_nonneg_left e1 (Nat.cast_nonneg D)
    nlinarith
  · have := Real.log_le_log hr h2
    rw [Real.log_mul hW.ne' hTp.ne', Real.log_mul (by positivity) (by linarith),
      Real.log_mul (by positivity) (Real.exp_pos _).ne',
      Real.log_pow, Real.log_exp] at this
    have e1 := Real.log_le_sub_one_of_pos (by linarith : 0 < 1 + u)
    have e2 := Real.log_le_sub_one_of_pos (by linarith : 0 < 1 + ue + u)
    have e4 : (D : ℝ) * Real.log (1 + u) ≤ D * u :=
      mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg D)
    have : 0 ≤ (D : ℝ) * u := by positivity
    nlinarith


theorem h_block (F : Discovery.Rounding) (hu : F.u ≤ 1 / 256) (hue : F.ue ≤ 1 / 64)
    (hul : F.ul ≤ 1 / 64) (D : ℕ) (lo hi : ℝ) (A : List ℝ) (hA : A ≠ [])
    (hb : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi) (N : ℝ) (hN : (A.length : ℝ) ≤ N)
    (hkη : N * F.ηe ≤ F.u) (st : State_a) (h : fblock_a F D A st) :
    |st.s - Real.log (A.map Real.exp).sum| ≤
      2 * (F.u * ((D : ℝ) + (hi - lo) + max |lo| |hi| + N + 2) + F.ue + F.ul * (N + 1)) := by
  have hu0 := F.hu
  have hue0 := F.hue
  have hul0 := F.hul
  have hη0 := F.hηe
  obtain ⟨v_m, v_s, r1, r2, r3, r4, hr1, hvm, hfs, hlog, hadd, hvs, hst⟩ := h
  have hmA : lmax A ∈ A := lmax_mem hA
  have hxm : ∀ x ∈ A, x ≤ lmax A ∧ lo ≤ x := fun x hx => ⟨le_lmax hx, (hb x hx).1⟩
  have hmhi : lmax A ≤ hi := (hb _ hmA).2
  have hmlo : lo ≤ lmax A := (hb _ hmA).1
  have hvm' : v_m = lmax A := by rw [hvm, hr1]
  rw [hvm'] at hfs hadd
  have hk : (A.length : ℝ) * F.ηe ≤ F.u := (mul_le_mul_of_nonneg_right hN hη0).trans hkη
  obtain ⟨hl1, hl2⟩ := h_fsum_two_sided F hu hue D lo hi A (lmax A) hmA hxm hmhi hk r2 hfs
  have hlohi : lo ≤ hi := hmlo.trans hmhi
  have ha : 0 ≤ F.u * (hi - lo) := mul_nonneg hu0 (by linarith)
  set m := lmax A with hm
  set T := (A.map fun x => Real.exp (x - m)).sum with hT
  have hT1 : 1 ≤ T := by
    have := List.single_le_sum (l := A.map fun x => Real.exp (x - m))
      (by intro x hx; simp only [List.mem_map] at hx; obtain ⟨y, _, rfl⟩ := hx; positivity)
      (Real.exp (m - m)) (List.mem_map.mpr ⟨m, hmA, rfl⟩)
    simpa using this
  have hTle : T ≤ A.length := by
    have := List.sum_le_length_nsmul (A.map fun x => Real.exp (x - m)) 1
      (by intro x hx; simp only [List.mem_map] at hx; obtain ⟨y, hy, rfl⟩ := hx
          rw [Real.exp_le_one_iff]; linarith [(hxm y hy).1])
    simpa using this
  obtain ⟨hr2p, hφ⟩ := h_log_f F.u F.ue (F.u * (hi - lo)) T r2 D hu0 hu hue0 hue ha hT1 hl1 hl2
  have hTpos : 0 < T := by linarith
  have hLT0 : 0 ≤ Real.log T := Real.log_nonneg hT1
  have hLT1 : Real.log T ≤ N := by
    have := Real.log_le_sub_one_of_pos hTpos
    linarith
  have hsum : (A.map Real.exp).sum = Real.exp m * T := by
    rw [hT, ← List.sum_map_mul_left]
    congr 1
    apply List.map_congr_left
    intro x _
    rw [← Real.exp_add]; congr 1; ring
  have htrue : Real.log (A.map Real.exp).sum = m + Real.log T := by
    rw [hsum, Real.log_mul (Real.exp_pos _).ne' hTpos.ne', Real.log_exp]
  rw [htrue]
  obtain ⟨δl, hδl, hr3⟩ := hlog
  obtain ⟨δ, hδ, hr4⟩ := hadd
  have hs : st.s = r4 := by rw [hst, hvs]
  rw [hs]
  set g := 51 / 50 * ((D : ℝ) * F.u + F.u * (hi - lo) + F.ue + F.u) with hg
  have hg0 : 0 ≤ g := by rw [hg]; positivity
  set φ := Real.log r2 - Real.log T with hφdef
  have hφ' : |φ| ≤ g := hφ
  have hlogr2 : |Real.log r2| ≤ N + g := by
    have : Real.log r2 = Real.log T + φ := by rw [hφdef]; ring
    rw [this]
    refine (abs_add_le _ _).trans ?_
    rw [abs_of_nonneg hLT0]; linarith
  have hr3b : |r3| ≤ (N + g) * (1 + F.ul) := by
    rw [hr3, abs_mul]
    have : |1 + δl| ≤ 1 + F.ul := (abs_add_le _ _).trans (by rw [abs_one]; linarith)
    exact mul_le_mul hlogr2 this (abs_nonneg _) (by linarith)
  have hmb : |m| ≤ max |lo| |hi| := by
    rw [abs_le]
    constructor
    · have := le_max_left |lo| |hi|
      have := neg_abs_le lo
      linarith
    · have := le_max_right |lo| |hi|
      have := le_abs_self hi
      linarith
  have hmr3 : |m + r3| ≤ max |lo| |hi| + (N + g) * (1 + F.ul) :=
    (abs_add_le _ _).trans (add_le_add hmb hr3b)
  have heq : r4 - (m + Real.log T) = (φ + δl * Real.log r2) + δ * (m + r3) := by
    rw [hr4, hr3, hφdef]; ring
  rw [heq]
  have b1 : |δl * Real.log r2| ≤ F.ul * (N + g) := by
    rw [abs_mul]; exact mul_le_mul hδl hlogr2 (abs_nonneg _) hul0
  have b2 : |δ * (m + r3)| ≤ F.u * (max |lo| |hi| + (N + g) * (1 + F.ul)) := by
    rw [abs_mul]; exact mul_le_mul hδ hmr3 (abs_nonneg _) hu0
  have b3 := abs_add_le (φ + δl * Real.log r2) (δ * (m + r3))
  have b4 := abs_add_le φ (δl * Real.log r2)
  have hN0 : 0 ≤ N := by linarith [hLT0, hLT1]
  have c1 : F.ul * g ≤ g / 64 := by
    have := mul_le_mul_of_nonneg_right hul hg0; linarith
  have c2 : F.u * g ≤ g / 256 := by
    have := mul_le_mul_of_nonneg_right hu hg0; linarith
  have c3 : F.u * F.ul * g ≤ g / 16384 := by
    have h1 := mul_le_mul_of_nonneg_right hul (mul_nonneg hu0 hg0)
    have h2 := mul_le_mul_of_nonneg_right hu hg0
    linarith
  have c4 : F.u * (F.ul * N) ≤ F.u * N / 64 := by
    have := mul_le_mul_of_nonneg_left hul (mul_nonneg hu0 hN0); linarith
  have hMn : 0 ≤ F.u * max |lo| |hi| := mul_nonneg hu0 (le_trans (abs_nonneg _) (le_max_left _ _))
  have d1 : 0 ≤ F.ul * N := mul_nonneg hul0 hN0
  have d2 : 0 ≤ F.u * N := mul_nonneg hu0 hN0
  have d3 : 0 ≤ (D : ℝ) * F.u := by positivity
  linarith


theorem h_uE (u ue ul D d lo hi M N : ℝ) (hu0 : 0 ≤ u) (hue0 : 0 ≤ ue) (hul0 : 0 ≤ ul)
    (hu : u ≤ 1 / 256) (hue : ue ≤ 1 / 64) (hul : ul ≤ 1 / 64)
    (hhl : hi - lo ≤ 2 * M) (hM : 0 ≤ M) (hN : 0 ≤ N) (hD : 0 ≤ D) (hd : 0 ≤ d)
    (hsum : (d + D + 1) * (u + ue + ul) ≤ 1 / 16) :
    u * (2 * (u * (D + (hi - lo) + M + N + 2) + ue + ul * (N + 1)) +
        2 * (u * (M + N + 2) + ue + ul) * d) ≤ (u * (M + N + 2) + ue + ul) / 4 := by
  have hs0 : 0 ≤ u + ue + ul := by positivity
  have hud : u * d ≤ 1 / 16 := by
    have h1 : u * d ≤ (u + ue + ul) * d := by
      apply mul_le_mul_of_nonneg_right _ hd; linarith
    nlinarith
  have huD : u * D ≤ 1 / 16 := by
    have h1 : u * D ≤ (u + ue + ul) * D := by
      apply mul_le_mul_of_nonneg_right _ hD; linarith
    nlinarith
  have hc : 0 ≤ u * (M + N + 2) + ue + ul := by positivity
  have f1 : u * (u * D) ≤ u / 16 := by
    have := mul_le_mul_of_nonneg_left huD hu0; linarith
  have g1 : u * u ≤ u / 256 := by
    have := mul_le_mul_of_nonneg_left hu hu0; linarith
  have f2 : u * u * (hi - lo) ≤ u * u * (2 * M) :=
    mul_le_mul_of_nonneg_left hhl (mul_nonneg hu0 hu0)
  have f3 : u * u * M ≤ u / 256 * M := mul_le_mul_of_nonneg_right g1 hM
  have f4 : u * u * N ≤ u / 256 * N := mul_le_mul_of_nonneg_right g1 hN
  have f5 : u * ue ≤ ue / 256 := by
    have := mul_le_mul_of_nonneg_right hu hue0; linarith
  have f6 : ul * (u * (N + 1)) ≤ 1 / 64 * (u * (N + 1)) :=
    mul_le_mul_of_nonneg_right hul (by positivity)
  have f7 : (u * d) * (2 * (u * (M + N + 2) + ue + ul)) ≤ 1 / 16 * (2 * (u * (M + N + 2) + ue + ul)) :=
    mul_le_mul_of_nonneg_right hud (by positivity)
  have f8 : 0 ≤ u * M := by positivity
  have f9 : 0 ≤ u * N := by positivity
  linarith


theorem h_flat_ne : ∀ t : Discovery.Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → t.flat ≠ []
  | .leaf b, h => by
    simp only [Discovery.Tree.leaves, List.mem_singleton, forall_eq] at h
    simpa [Discovery.Tree.flat] using h
  | .node l r, h => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => h b (by simp [Discovery.Tree.leaves, hb])
    have := h_flat_ne l hl
    simp [Discovery.Tree.flat, this]

theorem h_logsum (A : List ℝ) (hA : A ≠ []) (lo hi : ℝ) (hb : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi) :
    0 < (A.map Real.exp).sum ∧ lo ≤ Real.log (A.map Real.exp).sum ∧
      Real.log (A.map Real.exp).sum ≤ hi + A.length := by
  obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil A hA
  have hnn : ∀ y ∈ A.map Real.exp, 0 ≤ y := by
    intro y hy; simp only [List.mem_map] at hy; obtain ⟨z, _, rfl⟩ := hy; positivity
  have h1 := List.single_le_sum hnn (Real.exp x) (List.mem_map.mpr ⟨x, hx, rfl⟩)
  have hpos : 0 < (A.map Real.exp).sum := lt_of_lt_of_le (Real.exp_pos x) h1
  refine ⟨hpos, ?_, ?_⟩
  · have := Real.log_le_log (Real.exp_pos x) h1
    rw [Real.log_exp] at this
    linarith [(hb x hx).1]
  · have h2 := List.sum_le_length_nsmul (A.map Real.exp) (Real.exp hi)
      (by intro y hy; simp only [List.mem_map] at hy; obtain ⟨z, hz, rfl⟩ := hy
          exact Real.exp_le_exp.mpr (hb z hz).2)
    simp only [List.length_map, nsmul_eq_mul] at h2
    have h3 : (A.map Real.exp).sum * Real.exp (-hi) ≤ A.length := by
      have := mul_le_mul_of_nonneg_right h2 (Real.exp_pos (-hi)).le
      calc _ ≤ A.length * Real.exp hi * Real.exp (-hi) := this
        _ = A.length := by rw [mul_assoc, ← Real.exp_add]; simp
    have h4 := Real.log_le_sub_one_of_pos (mul_pos hpos (Real.exp_pos (-hi)))
    rw [Real.log_mul hpos.ne' (Real.exp_pos _).ne', Real.log_exp] at h4
    linarith

theorem h_tree (F : Discovery.Rounding) (hu : F.u ≤ 1 / 256) (hue : F.ue ≤ 1 / 64)
    (hul : F.ul ≤ 1 / 64) (D : ℕ) (lo hi : ℝ) (hlohi : lo ≤ hi) (N : ℝ)
    (hη : N * (F.η + F.ηe) ≤ F.u) :
    ∀ t : Discovery.Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → (∀ x ∈ t.flat, lo ≤ x ∧ x ≤ hi) →
      (t.flat.length : ℝ) ≤ N → ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
      ∀ st : State_a, Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
        |st.s - Real.log (t.flat.map Real.exp).sum| ≤
          2 * (F.u * ((D : ℝ) + (hi - lo) + max |lo| |hi| + N + 2) + F.ue + F.ul * (N + 1)) +
          2 * (F.u * (max |lo| |hi| + N + 2) + F.ue + F.ul) * (t.depth : ℝ) := by
  have hu0 := F.hu
  have hue0 := F.hue
  have hul0 := F.hul
  have hη0 := F.hη
  have hηe0 := F.hηe
  intro t
  induction t with
  | leaf b =>
    intro hne hb hN _ st hf
    cases hf with
    | leaf hfb =>
      have hbne : b ≠ [] := hne b (by simp [Discovery.Tree.leaves])
      have hN0 : 0 ≤ N := le_trans (Nat.cast_nonneg _) hN
      have hkη : N * F.ηe ≤ F.u :=
        le_trans (mul_le_mul_of_nonneg_left (by linarith) hN0) hη
      have := h_block F hu hue hul D lo hi b hbne hb N hN hkη st hfb
      simp only [Discovery.Tree.depth, Nat.cast_zero, mul_zero, add_zero]
      exact this
  | node l r ihl ihr =>
    intro hne hb hN hdep st hf
    cases hf with
    | node hfl hfr hfm =>
      rename_i a c
      have hl_ne : ∀ b ∈ l.leaves, b ≠ [] := fun b h => hne b (by simp [Discovery.Tree.leaves, h])
      have hr_ne : ∀ b ∈ r.leaves, b ≠ [] := fun b h => hne b (by simp [Discovery.Tree.leaves, h])
      have hlf := h_flat_ne l hl_ne
      have hrf := h_flat_ne r hr_ne
      have hflat : (Discovery.Tree.node l r).flat = l.flat ++ r.flat := rfl
      rw [hflat] at hb hN ⊢
      have hbl : ∀ x ∈ l.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hb x (List.mem_append_left _ hx)
      have hbr : ∀ x ∈ r.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hb x (List.mem_append_right _ hx)
      have hlen : (l.flat.length : ℝ) + r.flat.length ≤ N := by
        have := hN; rw [List.length_append] at this; push_cast at this; exact this
      have hl1 : (1 : ℝ) ≤ l.flat.length := by
        exact_mod_cast List.length_pos_iff.mpr hlf
      have hr1 : (1 : ℝ) ≤ r.flat.length := by
        exact_mod_cast List.length_pos_iff.mpr hrf
      have hNl : (l.flat.length : ℝ) ≤ N := by linarith
      have hNr : (r.flat.length : ℝ) ≤ N := by linarith
      have hN2 : 2 ≤ N := by linarith
      have hηe : 2 * F.ηe ≤ F.u := by
        have h1 : 2 * F.ηe ≤ N * F.ηe := mul_le_mul_of_nonneg_right hN2 hηe0
        have h2 : N * F.ηe ≤ N * (F.η + F.ηe) :=
          mul_le_mul_of_nonneg_left (by linarith) (by linarith)
        linarith
      have hdepth : ((Discovery.Tree.node l r).depth : ℝ) = ((max l.depth r.depth : ℕ) : ℝ) + 1 := by
        simp [Discovery.Tree.depth]
      rw [hdepth] at hdep ⊢
      have hdl : (l.depth : ℝ) ≤ ((max l.depth r.depth : ℕ) : ℝ) := Nat.cast_le.mpr (le_max_left _ _)
      have hdr : (r.depth : ℝ) ≤ ((max l.depth r.depth : ℕ) : ℝ) := Nat.cast_le.mpr (le_max_right _ _)
      have hss : 0 ≤ F.u + F.ue + F.ul := by positivity
      have hdd0 : (0 : ℝ) ≤ ((max l.depth r.depth : ℕ) : ℝ) := Nat.cast_nonneg _
      have hD0 : (0 : ℝ) ≤ D := Nat.cast_nonneg _
      have hdepl : ((l.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 := by
        refine le_trans (mul_le_mul_of_nonneg_right (by linarith) hss) hdep
      have hdepr : ((r.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 := by
        refine le_trans (mul_le_mul_of_nonneg_right (by linarith) hss) hdep
      have hdepd : (((max l.depth r.depth : ℕ) : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 := by
        refine le_trans (mul_le_mul_of_nonneg_right (by linarith) hss) hdep
      have h1 := ihl hl_ne hbl hNl hdepl a hfl
      have h2 := ihr hr_ne hbr hNr hdepr c hfr
      have hM0 : 0 ≤ max |lo| |hi| := le_trans (abs_nonneg _) (le_max_left _ _)
      have hhl : hi - lo ≤ 2 * max |lo| |hi| := by
        have := le_max_left |lo| |hi|
        have := le_max_right |lo| |hi|
        have := neg_abs_le lo
        have := le_abs_self hi
        linarith
      have hN0 : 0 ≤ N := by linarith
      have hE := h_uE F.u F.ue F.ul D ((max l.depth r.depth : ℕ) : ℝ) lo hi (max |lo| |hi|) N
        hu0 hue0 hul0 hu hue hul hhl hM0 hN0 hD0 hdd0 hdepd
      obtain ⟨ε0, hε0⟩ : ∃ e : ℝ, e = 2 * (F.u * ((D : ℝ) + (hi - lo) + max |lo| |hi| + N + 2) + F.ue + F.ul * (N + 1)) := ⟨_, rfl⟩
      obtain ⟨cc, hcc⟩ : ∃ e : ℝ, e = 2 * (F.u * (max |lo| |hi| + N + 2) + F.ue + F.ul) := ⟨_, rfl⟩
      rw [← hε0, ← hcc] at h1 h2 hE ⊢
      have hcc0 : 0 ≤ cc := by rw [hcc]; positivity
      have hε00 : 0 ≤ ε0 := by
        rw [hε0]
        have h := mul_nonneg hu0 (show (0 : ℝ) ≤ (D : ℝ) + (hi - lo) + max |lo| |hi| + N + 2 by
          have : (0:ℝ) ≤ hi - lo := by linarith
          positivity)
        have := mul_nonneg hul0 (show (0 : ℝ) ≤ N + 1 by linarith)
        linarith
      have hE0 : 0 ≤ ε0 + cc * ((max l.depth r.depth : ℕ) : ℝ) := by positivity
      have hx : |a.s - Real.log (l.flat.map Real.exp).sum| ≤ ε0 + cc * ((max l.depth r.depth : ℕ) : ℝ) :=
        h1.trans (by have := mul_le_mul_of_nonneg_left hdl hcc0; linarith)
      have hy : |c.s - Real.log (r.flat.map Real.exp).sum| ≤ ε0 + cc * ((max l.depth r.depth : ℕ) : ℝ) :=
        h2.trans (by have := mul_le_mul_of_nonneg_left hdr hcc0; linarith)
      obtain ⟨hlp, hllo, hlhi⟩ := h_logsum l.flat hlf lo hi hbl
      obtain ⟨hrp, hrlo, hrhi⟩ := h_logsum r.flat hrf lo hi hbr
      have hα : |Real.log (l.flat.map Real.exp).sum| ≤ max |lo| |hi| + N := by
        rw [abs_le]
        have := le_max_left |lo| |hi|
        have := le_max_right |lo| |hi|
        have := neg_abs_le lo
        have := le_abs_self hi
        constructor <;> linarith
      have hβ : |Real.log (r.flat.map Real.exp).sum| ≤ max |lo| |hi| + N := by
        rw [abs_le]
        have := le_max_left |lo| |hi|
        have := le_max_right |lo| |hi|
        have := neg_abs_le lo
        have := le_abs_self hi
        constructor <;> linarith
      have hms := h_merge_step F hu hue hul hηe (max |lo| |hi| + N)
        (ε0 + cc * ((max l.depth r.depth : ℕ) : ℝ)) _ _ (by linarith) hE0 hα hβ
        hE a c st hx hy hfm
      rw [Real.exp_log hlp, Real.exp_log hrp, ← List.sum_append, ← List.map_append] at hms
      rw [← hcc] at hms
      have : ε0 + cc * (((max l.depth r.depth : ℕ) : ℝ) + 1) =
          ε0 + cc * ((max l.depth r.depth : ℕ) : ℝ) + cc := by ring
      rw [this]
      exact hms

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List (ℝ)), X ≠ [] →
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree (ℝ)), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((2 : ℝ) * ((F.u * ((D : ℝ) + (hi - lo) + (max |lo| |hi|) + (X.length : ℝ) + (2 : ℝ))) + F.ue + (F.ul * ((X.length : ℝ) + (1 : ℝ))))) + ((2 : ℝ) * (t.depth : ℝ) * ((F.u * ((max |lo| |hi|) + (X.length : ℝ) + (2 : ℝ))) + F.ue + F.ul))) := by
  intro F D lo hi X hX hb hsm hn t hflat hne hdep st hf
  obtain ⟨hu, hue, hul, _, _⟩ := hsm
  obtain ⟨x0, hx0⟩ := List.exists_mem_of_ne_nil X hX
  have hlohi : lo ≤ hi := (hb x0 hx0).1.trans (hb x0 hx0).2
  have hb' : ∀ x ∈ t.flat, lo ≤ x ∧ x ≤ hi := by rw [hflat]; exact hb
  have hlen : (t.flat.length : ℝ) ≤ (X.length : ℝ) := by rw [hflat]
  have h := h_tree F hu hue hul D lo hi hlohi (X.length : ℝ) hn t hne hb' hlen hdep st hf
  rw [hflat] at h
  have hz : Real.log (znorm st) = st.s := by
    unfold znorm
    rw [one_div, ← Real.exp_neg, Real.log_exp]; ring
  rw [hz]
  linarith

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxLse
