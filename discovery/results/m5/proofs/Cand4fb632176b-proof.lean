import FPStatements

namespace Discovery.FP.Cand4fb632176b

open Discovery

-- helper lemmas here

theorem fpx_exp_sub_le {x y : ℝ} (h : y ≤ x) :
    Real.exp x - Real.exp y ≤ (x - y) * Real.exp x := by
  have h1 := Real.add_one_le_exp (y - x)
  have e : Real.exp y = Real.exp x * Real.exp (y - x) := by
    rw [← Real.exp_add]; ring_nf
  have := Real.exp_pos x
  rw [e]
  nlinarith

theorem fpx_ze (z : ℝ) (hz : 0 ≤ z) : z * Real.exp (-z) ≤ 1 / 2 := by
  have h1 := Real.add_one_le_exp (z - 1)
  have h2 : (2 : ℝ) ≤ Real.exp 1 := by
    have := Real.add_one_le_exp (1 : ℝ); linarith
  have e : Real.exp z = Real.exp (z - 1) * Real.exp 1 := by
    rw [← Real.exp_add]; ring_nf
  have h3 : 2 * z ≤ Real.exp z := by
    rw [e]; nlinarith [Real.exp_pos (z - 1)]
  have h4 : Real.exp (-z) * Real.exp z = 1 := by
    rw [← Real.exp_add]; simp
  have := Real.exp_pos (-z)
  nlinarith

theorem fpx_exp_pert {a δ u : ℝ} (ha : a ≤ 0) (hδ : |δ| ≤ u) (hu0 : 0 ≤ u) (hu : u ≤ 1 / 256) :
    |Real.exp (a * (1 + δ)) - Real.exp a| ≤ u * (51 / 100) := by
  have hδ' := abs_le.mp hδ
  have hr : a * (1 + δ) ≤ a * (1 - u) := by nlinarith
  have hr2 : a ≤ a * (1 - u) := by nlinarith
  have hm1 : Real.exp (a * (1 + δ)) ≤ Real.exp (a * (1 - u)) := Real.exp_le_exp.mpr hr
  have hm2 : Real.exp a ≤ Real.exp (a * (1 - u)) := Real.exp_le_exp.mpr hr2
  have key : Real.exp (a * (1 - u)) * (-a) ≤ 51 / 100 := by
    have h := fpx_ze ((1 - u) * (-a)) (by nlinarith)
    have e : -((1 - u) * (-a)) = a * (1 - u) := by ring
    rw [e] at h
    have hc : 0 < 1 - u := by linarith
    nlinarith [Real.exp_pos (a * (1 - u))]
  have habs : |a * (1 + δ) - a| ≤ u * (-a) := by
    have : a * (1 + δ) - a = a * δ := by ring
    rw [this, abs_mul, abs_of_nonpos ha]
    nlinarith [abs_nonneg δ]
  have hE := Real.exp_pos (a * (1 - u))
  rw [abs_le]
  constructor
  · rcases le_total a (a * (1 + δ)) with h | h
    · have := Real.exp_le_exp.mpr h
      nlinarith
    · have := fpx_exp_sub_le h
      have h5 := (abs_le.mp habs).2
      have h6 : (a - a * (1 + δ)) ≤ u * (-a) := by linarith [(abs_le.mp habs).1]
      nlinarith
  · rcases le_total a (a * (1 + δ)) with h | h
    · have := fpx_exp_sub_le h
      have h6 : (a * (1 + δ) - a) ≤ u * (-a) := by linarith [(abs_le.mp habs).2]
      nlinarith
    · have := Real.exp_le_exp.mpr h
      nlinarith

theorem fpx_lipschitz {a b a' b' ε : ℝ} (ha : |a' - a| ≤ ε) (hb : |b' - b| ≤ ε) :
    |Real.log (Real.exp a' + Real.exp b') - Real.log (Real.exp a + Real.exp b)| ≤ ε := by
  have key : ∀ a b a' b' : ℝ, |a' - a| ≤ ε → |b' - b| ≤ ε →
      Real.log (Real.exp a' + Real.exp b') - Real.log (Real.exp a + Real.exp b) ≤ ε := by
    intro a b a' b' ha hb
    have h1 : Real.exp a' ≤ Real.exp a * Real.exp ε := by
      rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp ha).2])
    have h2 : Real.exp b' ≤ Real.exp b * Real.exp ε := by
      rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp hb).2])
    have hp : 0 < Real.exp a + Real.exp b := by positivity
    have h3 : Real.log (Real.exp a' + Real.exp b') ≤
        Real.log ((Real.exp a + Real.exp b) * Real.exp ε) :=
      Real.log_le_log (by positivity) (by nlinarith)
    rw [Real.log_mul hp.ne' (Real.exp_pos ε).ne', Real.log_exp] at h3
    linarith
  rw [abs_le]
  constructor
  · have := key a' b' a b (by rwa [abs_sub_comm]) (by rwa [abs_sub_comm])
    linarith
  · exact key a b a' b' ha hb

theorem fpx_log_close {r T θ : ℝ} (hT : 1 ≤ T) (hθ0 : 0 ≤ θ) (hθ : θ ≤ 1 / 30)
    (h : |r - T| ≤ T * θ) : 0 < r ∧ |Real.log r - Real.log T| ≤ 21 / 20 * θ := by
  have h' := abs_le.mp h
  have hr : 0 < r := by nlinarith
  refine ⟨hr, ?_⟩
  have hTp : 0 < T := by linarith
  rw [abs_le]
  constructor
  · -- log T - log r ≤ T / r - 1
    have h1 := Real.log_le_sub_one_of_pos (div_pos hTp hr)
    rw [Real.log_div hTp.ne' hr.ne'] at h1
    have h2 : T / r ≤ 1 + 21 / 20 * θ := by
      rw [div_le_iff₀ hr]; nlinarith
    linarith
  · have h1 := Real.log_le_sub_one_of_pos (div_pos hr hTp)
    rw [Real.log_div hr.ne' hTp.ne'] at h1
    have h2 : r / T ≤ 1 + θ := by
      rw [div_le_iff₀ hTp]; nlinarith
    linarith

theorem fpx_arith {u ue ηe d2 d4 d5 f2 f4 ea eb Ta Tb r2 r4 r5 : ℝ} (hu0 : 0 ≤ u)
    (hu : u ≤ 1 / 256) (hue0 : 0 ≤ ue) (hue : ue ≤ 1 / 64) (hη0 : 0 ≤ ηe) (hηu : ηe ≤ u)
    (hd2 : |d2| ≤ ue) (hd4 : |d4| ≤ ue) (hd5 : |d5| ≤ u) (hf2 : |f2| ≤ ηe) (hf4 : |f4| ≤ ηe)
    (hr2 : 0 ≤ r2) (hr4 : 0 ≤ r4) (hea0 : 0 ≤ ea) (hea1 : ea ≤ 1) (heb0 : 0 ≤ eb) (heb1 : eb ≤ 1)
    (hTa1 : Ta ≤ 1) (hTb1 : Tb ≤ 1) (hT : 1 ≤ Ta + Tb)
    (hA : |ea - Ta| ≤ u * (51 / 100)) (hB : |eb - Tb| ≤ u * (51 / 100))
    (e2 : r2 = ea * (1 + d2) + f2) (e4 : r4 = eb * (1 + d4) + f4) (e5 : r5 = (r2 + r4) * (1 + d5)) :
    |r5 - (Ta + Tb)| ≤ (Ta + Tb) * (ue + 41 / 10 * u) := by
  have hA' := abs_le.mp hA
  have hB' := abs_le.mp hB
  have h2 : |r2 - ea| ≤ ue * ea + ηe := by
    have : r2 - ea = ea * d2 + f2 := by rw [e2]; ring
    rw [this]
    calc |ea * d2 + f2| ≤ |ea * d2| + |f2| := abs_add_le _ _
      _ ≤ ue * ea + ηe := by
        rw [abs_mul, abs_of_nonneg hea0]; nlinarith
  have h4 : |r4 - eb| ≤ ue * eb + ηe := by
    have : r4 - eb = eb * d4 + f4 := by rw [e4]; ring
    rw [this]
    calc |eb * d4 + f4| ≤ |eb * d4| + |f4| := abs_add_le _ _
      _ ≤ ue * eb + ηe := by
        rw [abs_mul, abs_of_nonneg heb0]; nlinarith
  have h2' := abs_le.mp h2
  have h4' := abs_le.mp h4
  have hT2 : Ta + Tb ≤ 2 := by linarith
  have p1 : ue * (ea + eb) ≤ ue * ((Ta + Tb) + 102 / 100 * u) :=
    mul_le_mul_of_nonneg_left (by linarith) hue0
  have p2 : ue * u ≤ u / 64 := by nlinarith
  have p3 : u ≤ u * (Ta + Tb) := by nlinarith
  have p4 : ue * (ea + eb) ≤ ue * (Ta + Tb) + u / 50 := by nlinarith
  have hS : |r2 + r4 - (Ta + Tb)| ≤ (Ta + Tb) * (ue + 304 / 100 * u) := by
    rw [abs_le]
    constructor <;> linarith
  have hS' := abs_le.mp hS
  have hθ : ue + 304 / 100 * u ≤ 1 / 30 := by linarith
  have q1 : (Ta + Tb) * (ue + 304 / 100 * u) ≤ (Ta + Tb) * (1 / 30) :=
    mul_le_mul_of_nonneg_left hθ (by linarith)
  have hS2 : r2 + r4 ≤ (Ta + Tb) * (106 / 100) := by linarith
  have hS0 : 0 ≤ r2 + r4 := by linarith
  have : r5 - (Ta + Tb) = (r2 + r4 - (Ta + Tb)) + d5 * (r2 + r4) := by rw [e5]; ring
  rw [this]
  have q2 : |d5| * (r2 + r4) ≤ u * ((Ta + Tb) * (106 / 100)) :=
    mul_le_mul hd5 hS2 hS0 hu0
  calc _ ≤ |r2 + r4 - (Ta + Tb)| + |d5 * (r2 + r4)| := abs_add_le _ _
    _ ≤ _ := by
      rw [abs_mul, abs_of_nonneg hS0]
      linarith [hS, q2]

theorem fpx_merge (F : Rounding) (hs : small F) (hηe : F.ηe ≤ F.u) (lo hi : ℝ) (D : ℕ) (N : ℝ)
    (hN : 1 ≤ N) (hlh : lo ≤ hi) (L R L' R' ε K : ℝ) (hε : 0 ≤ ε)
    (hL : |L' - L| ≤ ε) (hR : |R' - R| ≤ ε) (hL1 : lo ≤ L) (hL2 : L ≤ hi + (N - 1))
    (hR1 : lo ≤ R) (hR2 : R ≤ hi + (N - 1))
    (hK : K = F.u * ((hi - lo) + (D : ℝ) + (max |lo| |hi|) + N + 3) + F.ue + F.ul * (N + 1))
    (hKε : F.u * ε ≤ K / 8) (st : State_a) (h : fmerge_a F ⟨L'⟩ ⟨R'⟩ st) :
    |st.s - Real.log (Real.exp L + Real.exp R)| ≤ ε + 2 * K := by
  obtain ⟨hu1, hue1, hul1, -, -⟩ := hs
  have hu0 := F.hu
  have hue0 := F.hue
  have hul0 := F.hul
  obtain ⟨vmm, vs, r1, r2, r3, r4, r5, r6, r7, hmm, ⟨d1, hd1, e1⟩, ⟨d2, f2, hd2, hf2, e2, n2⟩,
    ⟨d3, hd3, e3⟩, ⟨d4, f4, hd4, hf4, e4, n4⟩, ⟨d5, hd5, e5⟩, ⟨d6, hd6, e6⟩, ⟨d7, hd7, e7⟩,
    hvs, hst⟩ := h
  have hmm' : vmm = max L' R' := hmm
  have e1' : r1 = (L' - vmm) * (1 + d1) := e1
  have e3' : r3 = (R' - vmm) * (1 + d3) := e3
  have hs7 : st.s = r7 := by rw [hst]; exact hvs
  have ha : L' - vmm ≤ 0 := by rw [hmm']; linarith [le_max_left L' R']
  have hb : R' - vmm ≤ 0 := by rw [hmm']; linarith [le_max_right L' R']
  have hab : L' - vmm = 0 ∨ R' - vmm = 0 := by
    rcases max_choice L' R' with h | h
    · left; rw [hmm', h]; ring
    · right; rw [hmm', h]; ring
  have hd1' := abs_le.mp hd1
  have hd3' := abs_le.mp hd3
  have hr1 : r1 ≤ 0 := by
    have := mul_nonneg (neg_nonneg.mpr ha) (show 0 ≤ 1 + d1 by linarith)
    rw [e1']; linarith
  have hr3 : r3 ≤ 0 := by
    have := mul_nonneg (neg_nonneg.mpr hb) (show 0 ≤ 1 + d3 by linarith)
    rw [e3']; linarith
  have hA := fpx_exp_pert ha hd1 hu0 hu1
  have hB := fpx_exp_pert hb hd3 hu0 hu1
  rw [← e1'] at hA
  rw [← e3'] at hB
  have hTa1 : Real.exp (L' - vmm) ≤ 1 := Real.exp_le_one_iff.mpr ha
  have hTb1 : Real.exp (R' - vmm) ≤ 1 := Real.exp_le_one_iff.mpr hb
  have hTa0 := Real.exp_pos (L' - vmm)
  have hTb0 := Real.exp_pos (R' - vmm)
  have hT : 1 ≤ Real.exp (L' - vmm) + Real.exp (R' - vmm) := by
    rcases hab with h | h
    · rw [h]; simp; linarith
    · rw [h]; simp; linarith
  have hea1 : Real.exp r1 ≤ 1 := Real.exp_le_one_iff.mpr hr1
  have heb1 : Real.exp r3 ≤ 1 := Real.exp_le_one_iff.mpr hr3
  have hAR := fpx_arith (u := F.u) (ue := F.ue) (ηe := F.ηe) hu0 hu1 hue0 hue1 F.hηe hηe hd2 hd4
    hd5 hf2 hf4 n2 n4 (Real.exp_pos r1).le hea1 (Real.exp_pos r3).le heb1 hTa1 hTb1 hT hA hB e2 e4 e5
  set T := Real.exp (L' - vmm) + Real.exp (R' - vmm) with hTdef
  have hT2 : T ≤ 2 := by linarith
  have hθ0 : 0 ≤ F.ue + 41 / 10 * F.u := by linarith
  have hθ : F.ue + 41 / 10 * F.u ≤ 1 / 30 := by linarith
  obtain ⟨hr5, hlog⟩ := fpx_log_close hT hθ0 hθ hAR
  have hlogT0 : 0 ≤ Real.log T := Real.log_nonneg hT
  have hlogT1 : Real.log T ≤ 1 := by linarith [Real.log_le_sub_one_of_pos (by linarith : 0 < T)]
  have hlr := abs_le.mp hlog
  have hlr5 : |Real.log r5| ≤ 104 / 100 := by
    rw [abs_le]; constructor <;> linarith
  have hr6 : |r6 - Real.log r5| ≤ F.ul * (104 / 100) := by
    have : r6 - Real.log r5 = Real.log r5 * d6 := by rw [e6]; ring
    rw [this, abs_mul, mul_comm]
    exact mul_le_mul hd6 hlr5 (abs_nonneg _) hul0
  have hr6b : |r6| ≤ 106 / 100 := by
    have : r6 = Real.log r5 * (1 + d6) := e6
    rw [this, abs_mul]
    have h1 : |1 + d6| ≤ 1 + F.ul := by
      calc |1 + d6| ≤ |(1 : ℝ)| + |d6| := abs_add_le _ _
        _ ≤ 1 + F.ul := by simp; exact hd6
    have := mul_le_mul hlr5 h1 (abs_nonneg _) (by norm_num : (0:ℝ) ≤ 104 / 100)
    linarith only [this, hul1]
  -- the maximum
  have hm1 : lo - ε ≤ vmm := by
    rw [hmm']; have := (abs_le.mp hL).1; have := le_max_left L' R'; linarith
  have hm2 : vmm ≤ hi + (N - 1) + ε := by
    rw [hmm']; apply max_le <;> linarith [(abs_le.mp hL).2, (abs_le.mp hR).2]
  have hM1 : |lo| ≤ max |lo| |hi| := le_max_left _ _
  have hM2 : |hi| ≤ max |lo| |hi| := le_max_right _ _
  have hM0 : 0 ≤ max |lo| |hi| := le_trans (abs_nonneg _) hM1
  have hmabs : |vmm| ≤ max |lo| |hi| + (N - 1) + ε := by
    rw [abs_le]; constructor
    · linarith [neg_abs_le lo]
    · linarith [le_abs_self hi]
  have hr7 : |r7 - (vmm + r6)| ≤ F.u * (|vmm| + 106 / 100) := by
    have : r7 - (vmm + r6) = (vmm + r6) * d7 := by rw [e7]; ring
    rw [this, abs_mul, mul_comm]
    have h1 : |vmm + r6| ≤ |vmm| + 106 / 100 := by
      calc |vmm + r6| ≤ |vmm| + |r6| := abs_add_le _ _
        _ ≤ _ := by linarith
    exact mul_le_mul hd7 h1 (abs_nonneg _) hu0
  -- the exact value
  have hexact : vmm + Real.log T = Real.log (Real.exp L' + Real.exp R') := by
    have : Real.exp L' + Real.exp R' = Real.exp vmm * T := by
      rw [hTdef, mul_add, ← Real.exp_add, ← Real.exp_add]; ring_nf
    rw [this, Real.log_mul (Real.exp_pos _).ne' (by linarith), Real.log_exp]
  have hlip := fpx_lipschitz hL hR
  rw [hexact.symm] at hlip
  have hmain : |r7 - (vmm + Real.log T)| ≤
      F.u * (max |lo| |hi| + (N - 1) + ε + 106 / 100) + F.ul * (104 / 100) + 21 / 20 * (F.ue + 41 / 10 * F.u) := by
    have h1 : r7 - (vmm + Real.log T) = (r7 - (vmm + r6)) + (r6 - Real.log r5) + (Real.log r5 - Real.log T) := by ring
    have h2 : F.u * (|vmm| + 106 / 100) ≤ F.u * (max |lo| |hi| + (N - 1) + ε + 106 / 100) :=
      mul_le_mul_of_nonneg_left (by linarith) hu0
    rw [h1]
    calc _ ≤ |r7 - (vmm + r6)| + |r6 - Real.log r5| + |Real.log r5 - Real.log T| := abs_add_three _ _ _
      _ ≤ _ := by linarith
  rw [hs7]
  have htri : |r7 - Real.log (Real.exp L + Real.exp R)| ≤
      |r7 - (vmm + Real.log T)| + |(vmm + Real.log T) - Real.log (Real.exp L + Real.exp R)| := by
    have := abs_add_le (r7 - (vmm + Real.log T)) ((vmm + Real.log T) - Real.log (Real.exp L + Real.exp R))
    simpa using this
  have hlip' : |(vmm + Real.log T) - Real.log (Real.exp L + Real.exp R)| ≤ ε := by
    exact hlip
  have hKu : F.u * (max |lo| |hi| + (N - 1) + ε + 106 / 100) = F.u * max |lo| |hi| + F.u * N + F.u * ε + F.u * (6/100) := by ring
  have hKexp : K = F.u * (hi - lo) + F.u * D + F.u * max |lo| |hi| + F.u * N + 3 * F.u + F.ue + F.ul * N + F.ul := by
    rw [hK]; ring
  have p1 : 0 ≤ F.u * (hi - lo) := mul_nonneg hu0 (by linarith)
  have p2 : 0 ≤ F.u * (D : ℝ) := mul_nonneg hu0 (Nat.cast_nonneg _)
  have p3 : 0 ≤ F.u * max |lo| |hi| := mul_nonneg hu0 hM0
  have p4 : F.u ≤ F.u * N := by
    have := mul_le_mul_of_nonneg_left hN hu0; linarith only [this]
  have p5 : 0 ≤ F.ul * N := mul_nonneg hul0 (by linarith)
  linarith

theorem fpx_exp_low {a δ u : ℝ} (ha : a ≤ 0) (hδ : |δ| ≤ u) :
    Real.exp a * (1 + u * a) ≤ Real.exp (a * (1 + δ)) := by
  have hδ' := abs_le.mp hδ
  have h1 : a * (1 + u) ≤ a * (1 + δ) := by nlinarith
  have h2 := Real.exp_le_exp.mpr h1
  have h3 := Real.add_one_le_exp (u * a)
  have e : Real.exp (a * (1 + u)) = Real.exp a * Real.exp (u * a) := by
    rw [← Real.exp_add]; ring_nf
  have := Real.exp_pos a
  rw [e] at h2
  nlinarith

theorem fpx_sum_lin (f g : ℝ → ℝ) (c1 c2 c3 : ℝ) (A : List ℝ) :
    (A.map fun x => c1 * f x + c2 * g x + c3).sum =
      c1 * (A.map f).sum + c2 * (A.map g).sum + c3 * A.length := by
  induction A with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.sum_cons, ih, List.length_cons]; push_cast; ring

theorem fpx_sums (A : List ℝ) (mx : ℝ) (hmx : mx ∈ A) (hle : ∀ x ∈ A, x ≤ mx) (κ : ℝ)
    (hκ : 0 < κ) :
    1 ≤ (A.map fun x => Real.exp (x - mx)).sum ∧
    (A.map fun x => Real.exp (x - mx)).sum ≤ A.length ∧
    (A.map fun x => Real.exp (x - mx) * (mx - x)).sum ≤
      A.length * κ - (A.map fun x => Real.exp (x - mx)).sum -
        (A.map fun x => Real.exp (x - mx)).sum * Real.log κ := by
  refine ⟨?_, ?_, ?_⟩
  · have h := List.single_le_sum (l := A.map fun x => Real.exp (x - mx)) (x := Real.exp (mx - mx))
      (fun x hx => by
        obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx
        exact (Real.exp_pos _).le) (List.mem_map.mpr ⟨mx, hmx, rfl⟩)
    simpa using h
  · have : (A.map fun x => Real.exp (x - mx)).sum ≤ (A.map fun _ => (1 : ℝ)).sum :=
      List.sum_le_sum (fun x hx => by
        have := hle x hx
        exact Real.exp_le_one_iff.mpr (by linarith))
    simpa using this
  · have h1 : (A.map fun x => Real.exp (x - mx) * (mx - x)).sum ≤
        (A.map fun x => (-1) * Real.exp (x - mx) + (-Real.log κ) * Real.exp (x - mx) + κ).sum :=
      List.sum_le_sum (fun x hx => by
        have hp := Real.exp_pos (x - mx)
        have h2 := Real.log_le_sub_one_of_pos (div_pos hκ hp)
        rw [Real.log_div hκ.ne' hp.ne', Real.log_exp] at h2
        have h3 : (κ / Real.exp (x - mx)) * Real.exp (x - mx) = κ := by field_simp
        nlinarith)
    rw [fpx_sum_lin (fun x => Real.exp (x - mx)) (fun x => Real.exp (x - mx)) (-1) (-Real.log κ) κ A] at h1
    linarith

theorem fpx_len_pow (T : Tree ℝ) (h : ∀ b ∈ T.leaves, ∃ x, b = [x]) :
    T.flat.length ≤ 2 ^ T.depth := by
  induction T with
  | leaf b =>
    obtain ⟨x, hx⟩ := h b (by simp [Tree.leaves])
    simp [Tree.flat, Tree.depth, hx]
  | node l r ihl ihr =>
    have hl := ihl (fun b hb => h b (by simp [Tree.leaves, hb]))
    have hr := ihr (fun b hb => h b (by simp [Tree.leaves, hb]))
    simp only [Tree.flat, Tree.depth, List.length_append]
    have h1 : 2 ^ l.depth ≤ 2 ^ (max l.depth r.depth) := Nat.pow_le_pow_right (by norm_num) (le_max_left _ _)
    have h2 : 2 ^ r.depth ≤ 2 ^ (max l.depth r.depth) := Nat.pow_le_pow_right (by norm_num) (le_max_right _ _)
    rw [pow_succ]; omega

theorem fpx_log_sandwich {u ue N μ t w : ℝ} {Dn : ℕ} (hu0 : 0 ≤ u) (hu1 : u ≤ 1 / 256)
    (hue0 : 0 ≤ ue) (hue1 : ue ≤ 1 / 64) (hN : 1 ≤ N) (hμ0 : 0 ≤ μ) (hμu : u * μ ≤ 1 / 16)
    (ht : 1 ≤ t)
    (hlow : (1 - u) ^ Dn * (t * ((1 - ue) * (1 - u * μ) - u)) ≤ w)
    (hup : w ≤ (1 + u) ^ Dn * (t * ((1 + ue) * (1 + 51 / 100 * u * N) * (1 + u)))) :
    |Real.log w - Real.log t| ≤ 1004 / 1000 * Dn * u + 11 / 10 * (ue + u * μ + u) + 51 / 100 * u * N := by
  have hp1 : 0 < 1 - u := by linarith
  have hP1 : 0 ≤ ue * (u * μ) := mul_nonneg hue0 (mul_nonneg hu0 hμ0)
  have hP : 0.9 ≤ (1 - ue) * (1 - u * μ) - u := by nlinarith
  have hPpos : 0 < (1 - ue) * (1 - u * μ) - u := by linarith
  have htp : 0 < t := by linarith
  have hpow : 0 < (1 - u) ^ Dn := pow_pos hp1 _
  have hwpos : 0 < w := lt_of_lt_of_le (by positivity) hlow
  have hQ : 0 < (1 + ue) * (1 + 51 / 100 * u * N) * (1 + u) := by
    have : 0 ≤ u * N := mul_nonneg hu0 (by linarith)
    have : 0 < 1 + 51 / 100 * u * N := by nlinarith
    positivity
  rw [abs_le]
  constructor
  · have h1 := Real.log_le_log (by positivity) hlow
    rw [Real.log_mul hpow.ne' (by positivity), Real.log_mul htp.ne' hPpos.ne', Real.log_pow] at h1
    have h2 := Real.one_sub_inv_le_log_of_pos hp1
    have h3 := Real.one_sub_inv_le_log_of_pos hPpos
    have h4 : (1 - u)⁻¹ ≤ 1 + 1004 / 1000 * u := by
      rw [inv_le_iff_one_le_mul₀ hp1]; nlinarith
    have h5 : ((1 - ue) * (1 - u * μ) - u)⁻¹ ≤ 1 + 11 / 10 * (ue + u * μ + u) := by
      rw [inv_le_iff_one_le_mul₀ hPpos]
      generalize hPv : (1 - ue) * (1 - u * μ) - u = Pv at hPpos hP ⊢
      have hq : 1 - Pv ≤ ue + u * μ + u := by rw [← hPv]; nlinarith
      have hq0 : 0 ≤ 1 - Pv := by rw [← hPv]; nlinarith
      have hq9 : 1 - Pv ≤ 0.09 := by linarith
      have e1 : 1 + 11 / 10 * (1 - Pv) ≤ 1 + 11 / 10 * (ue + u * μ + u) := by linarith
      have e2 : (1 + 11 / 10 * (1 - Pv)) * Pv ≤ (1 + 11 / 10 * (ue + u * μ + u)) * Pv :=
        mul_le_mul_of_nonneg_right e1 hPpos.le
      nlinarith [mul_nonneg hq0 (by linarith : (0:ℝ) ≤ 0.09 - (1 - Pv))]
    have h6 : (Dn : ℝ) * (1 - (1 - u)⁻¹) ≤ (Dn : ℝ) * Real.log (1 - u) :=
      mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg _)
    have h7 : (Dn : ℝ) * (1 - (1 - u)⁻¹) ≥ (Dn : ℝ) * (-(1004 / 1000 * u)) :=
      mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg _)
    have : 0 ≤ u * N := mul_nonneg hu0 (by linarith)
    nlinarith
  · have h1 := Real.log_le_log hwpos hup
    rw [Real.log_mul (pow_pos (by linarith : (0:ℝ) < 1 + u) _).ne' (by positivity),
      Real.log_mul htp.ne' hQ.ne', Real.log_mul (by positivity) (by positivity),
      Real.log_mul (by positivity) (by positivity), Real.log_pow] at h1
    have a1 := Real.log_le_sub_one_of_pos (by linarith : (0:ℝ) < 1 + u)
    have a2 := Real.log_le_sub_one_of_pos (by linarith : (0:ℝ) < 1 + ue)
    have hNu : 0 ≤ u * N := mul_nonneg hu0 (by linarith)
    have a3 := Real.log_le_sub_one_of_pos (by nlinarith : (0:ℝ) < 1 + 51 / 100 * u * N)
    have h6 : (Dn : ℝ) * Real.log (1 + u) ≤ (Dn : ℝ) * u :=
      mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg _)
    have : (0:ℝ) ≤ Dn * u := mul_nonneg (Nat.cast_nonneg _) hu0
    nlinarith

theorem fpx_leaf_arith {u ue ul N μ t w mx M d6 d7 r3 r4 K Λ : ℝ} {Dn : ℕ}
    (hu0 : 0 ≤ u) (hu1 : u ≤ 1 / 256) (hue0 : 0 ≤ ue) (hue1 : ue ≤ 1 / 64)
    (hul0 : 0 ≤ ul) (hul1 : ul ≤ 1 / 64) (hN : 1 ≤ N) (hμ0 : 0 ≤ μ) (hμD : μ ≤ Dn)
    (hμN : μ ≤ N - 1) (hDu : (Dn : ℝ) * u ≤ 1 / 16) (ht : 1 ≤ t) (htμ : Real.log t ≤ μ)
    (hlow : (1 - u) ^ Dn * (t * ((1 - ue) * (1 - u * μ) - u)) ≤ w)
    (hup : w ≤ (1 + u) ^ Dn * (t * ((1 + ue) * (1 + 51 / 100 * u * N) * (1 + u))))
    (hd6 : |d6| ≤ ul) (hd7 : |d7| ≤ u) (hw : r3 = Real.log w * (1 + d6))
    (hr4 : r4 = (mx + r3) * (1 + d7)) (hmx : |mx| ≤ M) (hΛ : 0 ≤ Λ)
    (hK : K = u * (Λ + (Dn : ℝ) + M + N + 3) + ue + ul * (N + 1)) :
    |r4 - (mx + Real.log t)| ≤ 2 * K := by
  have hμu : u * μ ≤ 1 / 16 := by
    have : u * μ ≤ u * Dn := mul_le_mul_of_nonneg_left hμD hu0
    linarith
  have hν := fpx_log_sandwich hu0 hu1 hue0 hue1 hN hμ0 hμu ht hlow hup
  set E := 1004 / 1000 * (Dn : ℝ) * u + 11 / 10 * (ue + u * μ + u) + 51 / 100 * u * N with hE
  have hDn0 : (0 : ℝ) ≤ Dn := Nat.cast_nonneg _
  have hDuu : 0 ≤ (Dn : ℝ) * u := mul_nonneg hDn0 hu0
  have huμ0 : 0 ≤ u * μ := mul_nonneg hu0 hμ0
  have huN0 : 0 ≤ u * N := mul_nonneg hu0 (by linarith)
  have hE0 : 0 ≤ E := by rw [hE]; nlinarith
  have hν' := abs_le.mp hν
  have hlt0 : 0 ≤ Real.log t := Real.log_nonneg ht
  have hlw : |Real.log w| ≤ μ + E := by
    rw [abs_le]; constructor <;> linarith
  have h3 : |r3 - Real.log w| ≤ ul * (μ + E) := by
    have : r3 - Real.log w = Real.log w * d6 := by rw [hw]; ring
    rw [this, abs_mul, mul_comm]
    exact mul_le_mul hd6 hlw (abs_nonneg _) hul0
  have h4 : |r3| ≤ 65 / 64 * (μ + E) := by
    rw [hw, abs_mul]
    have h1 : |1 + d6| ≤ 65 / 64 := by
      calc |1 + d6| ≤ |(1 : ℝ)| + |d6| := abs_add_le _ _
        _ ≤ 65 / 64 := by simp; linarith
    have := mul_le_mul hlw h1 (abs_nonneg _) (by linarith)
    linarith only [this]
  have h5 : |r4 - (mx + r3)| ≤ u * (M + 65 / 64 * (μ + E)) := by
    have : r4 - (mx + r3) = (mx + r3) * d7 := by rw [hr4]; ring
    rw [this, abs_mul, mul_comm]
    have h1 : |mx + r3| ≤ M + 65 / 64 * (μ + E) := by
      calc |mx + r3| ≤ |mx| + |r3| := abs_add_le _ _
        _ ≤ _ := by linarith
    exact mul_le_mul hd7 h1 (abs_nonneg _) hu0
  have htot : |r4 - (mx + Real.log t)| ≤
      |r4 - (mx + r3)| + |r3 - Real.log w| + |Real.log w - Real.log t| := by
    have : r4 - (mx + Real.log t) = (r4 - (mx + r3)) + (r3 - Real.log w) + (Real.log w - Real.log t) := by ring
    rw [this]; exact abs_add_three _ _ _
  have q1 : u * E ≤ 1 / 256 * E := mul_le_mul_of_nonneg_right hu1 hE0
  have q2 : ul * E ≤ 1 / 64 * E := mul_le_mul_of_nonneg_right hul1 hE0
  have q3 : u * μ ≤ u * (2 / 5 * Dn + 3 / 5 * (N - 1)) := by
    apply mul_le_mul_of_nonneg_left _ hu0
    have : (2 / 5 : ℝ) * μ ≤ 2 / 5 * Dn := by linarith
    linarith
  have q4 : ul * μ ≤ ul * (N - 1) := mul_le_mul_of_nonneg_left hμN hul0
  have q5 : 0 ≤ u * Λ := mul_nonneg hu0 hΛ
  have q6 : u ≤ u * N := by
    have := mul_le_mul_of_nonneg_left hN hu0; linarith only [this]
  have q7 : 0 ≤ ul * N := mul_nonneg hul0 (by linarith)
  have q8 : 0 ≤ u * M := mul_nonneg hu0 (le_trans (abs_nonneg _) hmx)
  have hKe : K = u * Λ + u * Dn + u * M + u * N + 3 * u + ue + ul * N + ul := by rw [hK]; ring
  have hEe : E = 1004 / 1000 * ((Dn : ℝ) * u) + 11 / 10 * ue + 11 / 10 * (u * μ) + 11 / 10 * u
      + 51 / 100 * (u * N) := by rw [hE]; ring
  have hsum : u * (M + 65 / 64 * (μ + E)) = u * M + 65 / 64 * (u * μ) + 65 / 64 * (u * E) := by ring
  have hsum2 : ul * (μ + E) = ul * μ + ul * E := by ring
  have hs1 : u * (2 / 5 * (Dn : ℝ) + 3 / 5 * (N - 1)) = 2 / 5 * (u * Dn) + 3 / 5 * (u * N) - 3 / 5 * u := by ring
  have hs2 : ul * (N - 1) = ul * N - ul := by ring
  have hDuc : (Dn : ℝ) * u = u * Dn := by ring
  have : |r4 - (mx + Real.log t)| ≤ 2 * K := by
    linarith [hν, h3, h5, htot]
  exact this

theorem fpx_term_le (F : Rounding) (hu1 : F.u ≤ 1 / 256) {x mx q1 q2 : ℝ} (hx : x ≤ mx)
    (h1 : F.add (x - mx) q1) (h2 : F.exp q1 q2) :
    0 ≤ q2 ∧
    (1 - F.ue) * (Real.exp (x - mx) - F.u * (Real.exp (x - mx) * (mx - x))) - F.ηe ≤ q2 ∧
    q2 ≤ (Real.exp (x - mx) + 51 / 100 * F.u) * (1 + F.ue) + F.ηe := by
  obtain ⟨δ, hδ, e1⟩ := h1
  obtain ⟨δe, e, hδe, he, e2, n2⟩ := h2
  have ha : x - mx ≤ 0 := by linarith
  have hp := fpx_exp_pert ha hδ F.hu hu1
  have hl := fpx_exp_low ha hδ
  rw [← e1] at hp hl
  have hp' := abs_le.mp hp
  have hδe' := abs_le.mp hδe
  have he' := abs_le.mp he
  have hq := Real.exp_pos q1
  refine ⟨n2, ?_, ?_⟩
  · have h3 : (1 - F.ue) * (Real.exp (x - mx) * (1 + F.u * (x - mx))) ≤ (1 - F.ue) * Real.exp q1 :=
      mul_le_mul_of_nonneg_left hl (by linarith [F.hue, F.hue1])
    have h4 : 0 ≤ Real.exp q1 * (δe + F.ue) := mul_nonneg hq.le (by linarith)
    have h5 : (1 - F.ue) * (Real.exp (x - mx) - F.u * (Real.exp (x - mx) * (mx - x))) =
        (1 - F.ue) * (Real.exp (x - mx) * (1 + F.u * (x - mx))) := by ring
    rw [h5, e2]
    nlinarith
  · have h4 : Real.exp q1 * (1 + δe) ≤ Real.exp q1 * (1 + F.ue) :=
      mul_le_mul_of_nonneg_left (by linarith) hq.le
    have h5 : Real.exp q1 * (1 + F.ue) ≤ (Real.exp (x - mx) + 51 / 100 * F.u) * (1 + F.ue) :=
      mul_le_mul_of_nonneg_right (by linarith) (by linarith [F.hue])
    rw [e2]
    linarith

theorem fpx_term_gt (F : Rounding) (hu1 : F.u ≤ 1) (hue1 : F.ue ≤ 1) (hηe1 : F.ηe ≤ 1)
    {x mx q1 q2 : ℝ} (hx : mx < x)
    (h1 : F.add (x - mx) q1) (h2 : F.exp q1 q2) :
    0 ≤ q2 ∧ q2 ≤ 2 * Real.exp (2 * (x - mx)) + 1 := by
  obtain ⟨δ, hδ, e1⟩ := h1
  obtain ⟨δe, e, hδe, he, e2, n2⟩ := h2
  refine ⟨n2, ?_⟩
  have hδ' := abs_le.mp hδ
  have hδe' := abs_le.mp hδe
  have he' := abs_le.mp he
  have h1 : q1 ≤ 2 * (x - mx) := by rw [e1]; nlinarith
  have h2 := Real.exp_le_exp.mpr h1
  have h3 := Real.exp_pos q1
  have h4 : Real.exp q1 * (1 + δe) ≤ Real.exp q1 * 2 :=
    mul_le_mul_of_nonneg_left (by linarith) h3.le
  rw [e2]
  nlinarith

theorem fpx_exp_sum (A : List ℝ) (mx : ℝ) :
    (A.map Real.exp).sum = Real.exp mx * (A.map fun x => Real.exp (x - mx)).sum := by
  rw [← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro x _
  rw [← Real.exp_add]; ring_nf

theorem fpx_leaf (F : Rounding) (hs : small F) (lo hi : ℝ) (D : ℕ) (N : ℝ) (hN : 1 ≤ N)
    (hlh : lo ≤ hi) (hNη : N * (F.η + F.ηe) ≤ F.u) (hDu : (D : ℝ) * F.u ≤ 1 / 16) (K : ℝ)
    (hK : K = F.u * ((hi - lo) + (D : ℝ) + (max |lo| |hi|) + N + 3) + F.ue + F.ul * (N + 1))
    (A : List ℝ) (hA : A ≠ []) (hlen : (A.length : ℝ) ≤ N) (hb : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi)
    (st : State_a) (h : fblock_a F D A st) :
    |st.s - Real.log ((A.map Real.exp).sum)| ≤ 2 * K := by
  obtain ⟨hu1, hue1, hul1, hη1, hηe1⟩ := hs
  have hu0 := F.hu
  have hue0 := F.hue
  have hul0 := F.hul
  have hη0 := F.hη
  have hηe0 := F.hηe
  have hηeN : F.ηe ≤ F.u := by
    have : F.ηe ≤ N * (F.η + F.ηe) := by nlinarith
    linarith
  have hnηe : (A.length : ℝ) * F.ηe ≤ F.u := by
    have h1 : (A.length : ℝ) * F.ηe ≤ (A.length : ℝ) * (F.η + F.ηe) :=
      mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg _)
    have h2 : (A.length : ℝ) * (F.η + F.ηe) ≤ N * (F.η + F.ηe) :=
      mul_le_mul_of_nonneg_right hlen (by linarith)
    linarith
  obtain ⟨v_mx, v_t, v_s, r1, r2, r3, r4, hr1, hvmx, hfs, hvt, ⟨d6, hd6, e6⟩, ⟨d7, hd7, e7⟩,
    hvs, hst⟩ := h
  have hvmx' : v_mx = lmax A := by rw [hvmx, hr1]
  have hmxA : lmax A ∈ A := lmax_mem hA
  have hle : ∀ x ∈ A, x ≤ lmax A := fun x hx => le_lmax hx
  obtain ⟨mx, hmxdef⟩ : ∃ mx, mx = lmax A := ⟨_, rfl⟩
  rw [← hmxdef] at hvmx' hmxA hle
  have hvmx'' : mx = v_mx := hvmx'.symm
  subst hvmx''
  have hst' : st.s = r4 := by rw [hst]; exact hvs
  -- the length of the block
  have hlen2 : A.length ≤ 2 ^ D := by
    obtain ⟨T, hTflat, hTleaves, hTdepth, -⟩ := hfs
    have := fpx_len_pow T hTleaves
    rw [hTflat] at this
    exact le_trans this (Nat.pow_le_pow_right (by norm_num) hTdepth)
  have hn1 : 1 ≤ A.length := List.length_pos_iff.mpr hA
  have hn1r : (1 : ℝ) ≤ A.length := by exact_mod_cast hn1
  have hnpos : (0 : ℝ) < A.length := by linarith
  obtain ⟨μ, hμ⟩ : ∃ μ, μ = Real.log (A.length : ℝ) := ⟨_, rfl⟩
  have hμ0 : 0 ≤ μ := by rw [hμ]; exact Real.log_nonneg hn1r
  have hμN : μ ≤ N - 1 := by
    have := Real.log_le_sub_one_of_pos hnpos; rw [hμ]; linarith
  have hμD : μ ≤ D := by
    have h2 : (A.length : ℝ) ≤ (2 : ℝ) ^ D := by exact_mod_cast hlen2
    have h3 : Real.log (A.length : ℝ) ≤ Real.log ((2 : ℝ) ^ D) := Real.log_le_log hnpos h2
    rw [← hμ] at h3
    rw [Real.log_pow] at h3
    have h4 := Real.log_le_sub_one_of_pos (by norm_num : (0:ℝ) < 2)
    have : (D : ℝ) * Real.log 2 ≤ D * 1 :=
      mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg _)
    linarith
  -- sums
  obtain ⟨ht1, htn, hP1⟩ := fpx_sums A mx hmxA hle (A.length : ℝ)⁻¹ (inv_pos.mpr hnpos)
  set t := (A.map fun x => Real.exp (x - mx)).sum with ht
  set P1 := (A.map fun x => Real.exp (x - mx) * (mx - x)).sum with hP1def
  have hP1' : P1 ≤ t * μ := by
    rw [Real.log_inv] at hP1
    have h2 : (A.length : ℝ) * (A.length : ℝ)⁻¹ = 1 := mul_inv_cancel₀ hnpos.ne'
    rw [← hμ] at hP1
    linarith [hP1, h2, ht1]
  -- fsum bounds
  have hfs' := hfs
  have hbounds := fsum_bounds F D _
    (fun x => if x ≤ mx then max 0 ((1 - F.ue) * (Real.exp (x - mx) - F.u * (Real.exp (x - mx) * (mx - x))) - F.ηe) else 0)
    (fun x => if x ≤ mx then (Real.exp (x - mx) + 51 / 100 * F.u) * (1 + F.ue) + F.ηe else 2 * Real.exp (2 * (x - mx)) + 1)
    (fun x => by beta_reduce; split_ifs
                 · exact le_max_left _ _
                 · exact le_refl _)
    (fun x y hxy => by
      obtain ⟨q1, q2, h1, h2, rfl⟩ := hxy
      by_cases hx : x ≤ mx
      · obtain ⟨a1, a2, a3⟩ := fpx_term_le F hu1 hx h1 h2
        simp only [if_pos hx]
        exact ⟨max_le a1 a2, a3⟩
      · have hx' : mx < x := lt_of_not_ge hx
        obtain ⟨a1, a2⟩ := fpx_term_gt F (by linarith) (by linarith) (by norm_num at hηe1 ⊢; linarith) hx' h1 h2
        simp only [if_neg hx]
        exact ⟨a1, a2⟩)
    A r2 hfs'
  obtain ⟨hb1, hb2⟩ := hbounds
  -- lower sum
  have hsl : (1 - F.ue) * t + (-((1 - F.ue) * F.u)) * P1 + (-F.ηe) * A.length ≤
      (A.map fun x => if x ≤ mx then max 0 ((1 - F.ue) * (Real.exp (x - mx) - F.u * (Real.exp (x - mx) * (mx - x))) - F.ηe) else 0).sum := by
    rw [← fpx_sum_lin (fun x => Real.exp (x - mx)) (fun x => Real.exp (x - mx) * (mx - x))]
    apply List.sum_le_sum
    intro x hx
    simp only [if_pos (hle x hx)]
    refine le_trans (le_of_eq ?_) (le_max_right _ _)
    ring
  have hsh : (A.map fun x => if x ≤ mx then (Real.exp (x - mx) + 51 / 100 * F.u) * (1 + F.ue) + F.ηe else 2 * Real.exp (2 * (x - mx)) + 1).sum ≤
      (1 + F.ue) * t + 0 * P1 + (51 / 100 * F.u * (1 + F.ue) + F.ηe) * A.length := by
    rw [← fpx_sum_lin (fun x => Real.exp (x - mx)) (fun x => Real.exp (x - mx) * (mx - x))]
    apply List.sum_le_sum
    intro x hx
    simp only [if_pos (hle x hx)]
    apply le_of_eq
    ring
  have hu1' : 0 ≤ 1 - F.ue := by linarith
  have hlow : (1 - F.u) ^ D * (t * ((1 - F.ue) * (1 - F.u * μ) - F.u)) ≤ r2 := by
    refine le_trans (mul_le_mul_of_nonneg_left ?_ (pow_nonneg (by linarith) _)) hb1
    have e1 : F.u * P1 ≤ F.u * (t * μ) := mul_le_mul_of_nonneg_left hP1' hu0
    have e2 : (1 - F.ue) * (F.u * P1) ≤ (1 - F.ue) * (F.u * (t * μ)) :=
      mul_le_mul_of_nonneg_left e1 hu1'
    have e3 : F.u ≤ F.u * t := by
      have := mul_le_mul_of_nonneg_left ht1 hu0; linarith only [this]
    refine le_trans ?_ hsl
    linarith [e2, e3, hnηe]
  have hup : r2 ≤ (1 + F.u) ^ D * (t * ((1 + F.ue) * (1 + 51 / 100 * F.u * N) * (1 + F.u))) := by
    refine le_trans hb2 (mul_le_mul_of_nonneg_left ?_ (pow_nonneg (by linarith) _))
    refine le_trans hsh ?_
    have e1 : 51 / 100 * F.u * (1 + F.ue) * A.length ≤ 51 / 100 * F.u * (1 + F.ue) * N :=
      mul_le_mul_of_nonneg_left hlen (by positivity)
    have e2 : 51 / 100 * F.u * (1 + F.ue) * N ≤ t * (51 / 100 * F.u * (1 + F.ue) * N) := by
      have : 0 ≤ 51 / 100 * F.u * (1 + F.ue) * N := by
        have : 0 ≤ N := by linarith
        positivity
      have := mul_le_mul_of_nonneg_right ht1 this
      linarith only [this]
    have e3 : F.u ≤ F.u * t := by
      have := mul_le_mul_of_nonneg_left ht1 hu0; linarith only [this]
    have hX : 1 ≤ (1 + F.ue) * (1 + 51 / 100 * F.u * N) := by
      have : 0 ≤ F.ue * (51 / 100 * F.u * N) := by
        have : 0 ≤ N := by linarith
        positivity
      have h0 : 0 ≤ 51 / 100 * F.u * N := by
        have : 0 ≤ N := by linarith
        positivity
      linarith [this, h0]
    have e4 : t * F.u ≤ t * (F.u * ((1 + F.ue) * (1 + 51 / 100 * F.u * N))) := by
      apply mul_le_mul_of_nonneg_left _ (by linarith)
      have := mul_le_mul_of_nonneg_left hX hu0
      linarith only [this]
    linarith [e1, e2, e3, e4, hnηe]
  -- the true value
  have htrue : Real.log ((A.map Real.exp).sum) = mx + Real.log t := by
    rw [fpx_exp_sum A mx, Real.log_mul (Real.exp_pos _).ne' (by linarith), Real.log_exp]
  have htμ : Real.log t ≤ μ := by rw [hμ]; exact Real.log_le_log (by linarith) htn
  have hmxb := hb mx hmxA
  have hMb : |mx| ≤ max |lo| |hi| := by
    rw [abs_le]; constructor
    · linarith [neg_abs_le lo, le_max_left |lo| |hi|]
    · linarith [le_abs_self hi, le_max_right |lo| |hi|]
  have hr2eq : r3 = Real.log r2 * (1 + d6) := by rw [e6, hvt]
  have := fpx_leaf_arith (u := F.u) (ue := F.ue) (ul := F.ul) (N := N) (μ := μ) (t := t) (w := r2)
    (mx := mx) (M := max |lo| |hi|) (d6 := d6) (d7 := d7) (r3 := r3) (r4 := r4) (K := K)
    (Λ := hi - lo) (Dn := D) hu0 hu1 hue0 hue1 hul0 hul1 hN hμ0 hμD hμN hDu ht1 htμ hlow hup hd6
    hd7 hr2eq e7 hMb (by linarith) hK
  rw [hst', htrue]
  exact this

theorem fpx_flat_ne (T : Tree ℝ) (h : ∀ b ∈ T.leaves, b ≠ []) : T.flat ≠ [] := by
  induction T with
  | leaf b => simpa [Tree.flat, Tree.leaves] using h b (by simp [Tree.leaves])
  | node l r ihl ihr =>
    have hl := ihl (fun b hb => h b (by simp [Tree.leaves, hb]))
    simp only [Tree.flat]
    intro h0
    exact hl (List.append_eq_nil_iff.mp h0).1

theorem fpx_sum_le (A : List ℝ) (hi : ℝ) (h : ∀ x ∈ A, x ≤ hi) :
    (A.map Real.exp).sum ≤ A.length * Real.exp hi := by
  induction A with
  | nil => simp
  | cons a l ih =>
    have h1 := ih (fun x hx => h x (by simp [hx]))
    have h2 := Real.exp_le_exp.mpr (h a (by simp))
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    linarith

theorem fpx_true_bounds (A : List ℝ) (hA : A ≠ []) (lo hi : ℝ) (hb : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi) :
    0 < (A.map Real.exp).sum ∧ lo ≤ Real.log ((A.map Real.exp).sum) ∧
      Real.log ((A.map Real.exp).sum) ≤ hi + (A.length - 1) := by
  obtain ⟨x0, hx0⟩ := List.exists_mem_of_ne_nil A hA
  have h1 : Real.exp x0 ≤ (A.map Real.exp).sum :=
    List.single_le_sum (l := A.map Real.exp) (x := Real.exp x0)
      (fun x hx => by
        obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx
        exact (Real.exp_pos _).le) (List.mem_map.mpr ⟨x0, hx0, rfl⟩)
  have hpos : 0 < (A.map Real.exp).sum := lt_of_lt_of_le (Real.exp_pos _) h1
  refine ⟨hpos, ?_, ?_⟩
  · have h2 : Real.exp lo ≤ (A.map Real.exp).sum :=
      le_trans (Real.exp_le_exp.mpr (hb x0 hx0).1) h1
    have := Real.log_le_log (Real.exp_pos _) h2
    rwa [Real.log_exp] at this
  · have h2 := fpx_sum_le A hi (fun x hx => (hb x hx).2)
    have hn : (0 : ℝ) < A.length := by
      have := List.length_pos_iff.mpr hA; exact_mod_cast this
    have h3 := Real.log_le_log hpos h2
    rw [Real.log_mul hn.ne' (Real.exp_pos _).ne', Real.log_exp] at h3
    have := Real.log_le_sub_one_of_pos hn
    linarith

theorem fpx_main (F : Rounding) (hs : small F) (lo hi : ℝ) (D : ℕ) (N : ℝ) (hN : 1 ≤ N)
    (hlh : lo ≤ hi) (hNη : N * (F.η + F.ηe) ≤ F.u) (K : ℝ)
    (hK : K = F.u * ((hi - lo) + (D : ℝ) + (max |lo| |hi|) + N + 3) + F.ue + F.ul * (N + 1))
    (hK0 : 0 ≤ K) :
    ∀ (t : Tree ℝ) (st : State_a), Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      (∀ b ∈ t.leaves, b ≠ []) → (∀ x ∈ t.flat, lo ≤ x ∧ x ≤ hi) → (t.flat.length : ℝ) ≤ N →
      ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
      |st.s - Real.log ((t.flat.map Real.exp).sum)| ≤ 2 * K * ((t.depth : ℝ) + 1) := by
  intro t st hev
  induction hev with
  | leaf hb =>
    intro hl hx hlen hdep
    rename_i b w
    have hDu : (D : ℝ) * F.u ≤ 1 / 16 := by
      have h1 : (D : ℝ) * F.u ≤ ((0 : ℝ) + D + 1) * (F.u + F.ue + F.ul) := by
        have := F.hu; have := F.hue; have := F.hul
        nlinarith [Nat.cast_nonneg (α := ℝ) D]
      simp only [Tree.depth, Nat.cast_zero] at hdep
      linarith
    have hne : b ≠ [] := hl b (by simp [Tree.leaves])
    have := fpx_leaf F hs lo hi D N hN hlh hNη hDu K hK b hne hlen hx w hb
    simp only [Tree.flat, Tree.depth, Nat.cast_zero] at *
    linarith
  | node hl hr hm ihl ihr =>
    rename_i l r a c w
    intro hleaves hx hlen hdep
    have hleaves_l : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => hleaves b (by simp [Tree.leaves, hb])
    have hleaves_r : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => hleaves b (by simp [Tree.leaves, hb])
    have hx_l : ∀ x ∈ l.flat, lo ≤ x ∧ x ≤ hi := fun x h => hx x (by simp [Tree.flat, h])
    have hx_r : ∀ x ∈ r.flat, lo ≤ x ∧ x ≤ hi := fun x h => hx x (by simp [Tree.flat, h])
    have hlen' : (l.flat.length : ℝ) + r.flat.length ≤ N := by
      simp only [Tree.flat, List.length_append] at hlen; push_cast at hlen; exact hlen
    have hl0 : (0 : ℝ) ≤ l.flat.length := Nat.cast_nonneg _
    have hr0 : (0 : ℝ) ≤ r.flat.length := Nat.cast_nonneg _
    have hne_l := fpx_flat_ne l hleaves_l
    have hne_r := fpx_flat_ne r hleaves_r
    have hn_l : (1 : ℝ) ≤ l.flat.length := by
      have := List.length_pos_iff.mpr hne_l; exact_mod_cast this
    have hn_r : (1 : ℝ) ≤ r.flat.length := by
      have := List.length_pos_iff.mpr hne_r; exact_mod_cast this
    have hs0 : 0 ≤ F.u + F.ue + F.ul := by linarith [F.hu, F.hue, F.hul]
    have hdl : ((l.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 := by
      refine le_trans (mul_le_mul_of_nonneg_right ?_ hs0) hdep
      have := le_max_left (l.depth : ℝ) (r.depth : ℝ)
      simp only [Tree.depth]; push_cast; linarith
    have hdr : ((r.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 := by
      refine le_trans (mul_le_mul_of_nonneg_right ?_ hs0) hdep
      have := le_max_right (l.depth : ℝ) (r.depth : ℝ)
      simp only [Tree.depth]; push_cast; linarith
    have IL := ihl hleaves_l hx_l (by linarith) hdl
    have IR := ihr hleaves_r hx_r (by linarith) hdr
    obtain ⟨pl, ll, ul⟩ := fpx_true_bounds l.flat hne_l lo hi hx_l
    obtain ⟨pr, lr, ur⟩ := fpx_true_bounds r.flat hne_r lo hi hx_r
    set Sl := Real.log ((l.flat.map Real.exp).sum) with hSl
    set Sr := Real.log ((r.flat.map Real.exp).sum) with hSr
    set m : ℝ := max (l.depth : ℝ) (r.depth : ℝ) with hm_def
    have hdt : ((Tree.node l r).depth : ℝ) = m + 1 := by
      show ((max l.depth r.depth + 1 : ℕ) : ℝ) = _
      push_cast; rfl
    have hm0 : (0 : ℝ) ≤ m := le_trans (Nat.cast_nonneg _) (le_max_left _ _)
    have hεl : 2 * K * ((l.depth : ℝ) + 1) ≤ 2 * K * (m + 1) :=
      mul_le_mul_of_nonneg_left (by linarith [le_max_left (l.depth : ℝ) (r.depth : ℝ)]) (by linarith)
    have hεr : 2 * K * ((r.depth : ℝ) + 1) ≤ 2 * K * (m + 1) :=
      mul_le_mul_of_nonneg_left (by linarith [le_max_right (l.depth : ℝ) (r.depth : ℝ)]) (by linarith)
    have hηeN : F.ηe ≤ F.u := by
      have : F.ηe ≤ N * (F.η + F.ηe) := by nlinarith [F.hη, F.hηe]
      linarith
    have hKε : F.u * (2 * K * (m + 1)) ≤ K / 8 := by
      have h1 : F.u * (m + 1) ≤ 1 / 16 := by
        have : F.u * ((Tree.node l r).depth : ℝ) ≤ ((Tree.node l r).depth + D + 1 : ℝ) * (F.u + F.ue + F.ul) := by
          have := F.hu; have := F.hue; have := F.hul
          nlinarith [Nat.cast_nonneg (α := ℝ) D, Nat.cast_nonneg (α := ℝ) (Tree.node l r).depth]
        rw [hdt] at this hdep
        linarith
      nlinarith
    have hmerge := fpx_merge F hs hηeN lo hi D N hN hlh (Sl) (Sr) a.s c.s (2 * K * (m + 1)) K
      (mul_nonneg (mul_nonneg (by norm_num) hK0) (by linarith)) (le_trans IL hεl) (le_trans IR hεr)
      ll (by linarith) lr (by linarith) hK hKε w hm
    have e1 : Real.exp Sl = (l.flat.map Real.exp).sum := Real.exp_log pl
    have e2 : Real.exp Sr = (r.flat.map Real.exp).sum := Real.exp_log pr
    have hsum : ((Tree.node l r).flat.map Real.exp).sum = Real.exp Sl + Real.exp Sr := by
      simp only [Tree.flat, List.map_append, List.sum_append]; rw [e1, e2]
    rw [hsum, hdt]
    linarith

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ), ∀ (lo hi : ℝ) (X : List ℝ),
    X ≠ [] → (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ ((2 : ℝ) * ((t.depth : ℝ) + (1 : ℝ)) * ((F.u * ((hi - lo) + (D : ℝ) + (max |lo| |hi|) + (X.length : ℝ) + (3 : ℝ))) + F.ue + (F.ul * ((X.length : ℝ) + (1 : ℝ))))) := by
  intro F D lo hi X hX hb hsm hNη t hflat hleaves hdep st hev
  subst hflat
  have hN : (1 : ℝ) ≤ t.flat.length := by
    have := List.length_pos_iff.mpr hX; exact_mod_cast this
  have hlh : lo ≤ hi := by
    obtain ⟨x0, h0⟩ := List.exists_mem_of_ne_nil _ hX
    exact le_trans (hb x0 h0).1 (hb x0 h0).2
  obtain ⟨K, hK⟩ : ∃ K : ℝ, K = F.u * ((hi - lo) + (D : ℝ) + (max |lo| |hi|) + (t.flat.length : ℝ) + 3)
      + F.ue + F.ul * ((t.flat.length : ℝ) + 1) := ⟨_, rfl⟩
  have hK0 : 0 ≤ K := by
    have h1 : 0 ≤ max |lo| |hi| := le_trans (abs_nonneg _) (le_max_left _ _)
    have h2 : 0 ≤ F.u * ((hi - lo) + (D : ℝ) + (max |lo| |hi|) + (t.flat.length : ℝ) + 3) :=
      mul_nonneg F.hu (by have := Nat.cast_nonneg (α := ℝ) D; linarith)
    have h3 : 0 ≤ F.ul * ((t.flat.length : ℝ) + 1) := mul_nonneg F.hul (by linarith)
    rw [hK]; linarith [F.hue]
  have hmain := fpx_main F hsm lo hi D (t.flat.length : ℝ) hN hlh hNη K hK hK0 t st hev hleaves hb
    le_rfl hdep
  have hz : Real.log (znorm st) = st.s := by
    unfold znorm
    rw [zero_sub, Real.exp_neg, one_div, inv_inv, Real.log_exp]
  rw [hz]
  rw [hK] at hmain
  linarith [hmain]

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.Cand4fb632176b
