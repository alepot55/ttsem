import FPStatements

namespace Discovery.FP.SoftmaxConditionalFpNounderflow

open Discovery

theorem sm_sum_le {l : List ℝ} {f h : ℝ → ℝ} (c : ℝ) (H : ∀ x ∈ l, f x + c ≤ h x) :
    (l.map fun x => Real.exp (f x)).sum * Real.exp c ≤ (l.map fun x => Real.exp (h x)).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, add_mul]
    have : Real.exp (f a) * Real.exp c ≤ Real.exp (h a) := by
      rw [← Real.exp_add]; exact Real.exp_le_exp.2 (H a (by simp))
    have := ih (fun x hx => H x (List.mem_cons_of_mem _ hx))
    linarith

theorem sm_sum_ge {l : List ℝ} {f h : ℝ → ℝ} (c : ℝ) (H : ∀ x ∈ l, h x ≤ f x + c) :
    (l.map fun x => Real.exp (h x)).sum ≤ (l.map fun x => Real.exp (f x)).sum * Real.exp c := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, add_mul]
    have : Real.exp (h a) ≤ Real.exp (f a) * Real.exp c := by
      rw [← Real.exp_add]; exact Real.exp_le_exp.2 (H a (by simp))
    have := ih (fun x hx => H x (List.mem_cons_of_mem _ hx))
    linarith

theorem sm_sum_nonneg (l : List ℝ) (f : ℝ → ℝ) : 0 ≤ (l.map fun x => Real.exp (f x)).sum := by
  apply List.sum_nonneg
  intro y hy
  simp only [List.mem_map] at hy
  obtain ⟨x, _, rfl⟩ := hy
  exact (Real.exp_pos _).le

theorem sm_exp_up {v : ℝ} (h0 : 0 ≤ v) (h1 : v < 1) : 1 + v ≤ Real.exp (v / (1 - v)) := by
  have h : v ≤ v/(1-v) := by
    rw [le_div_iff₀ (by linarith)]; nlinarith
  linarith [Real.add_one_le_exp (v/(1-v))]

theorem sm_exp_lo {v : ℝ} (h0 : 0 ≤ v) (h1 : v < 1) : Real.exp (-(v / (1 - v))) ≤ 1 - v := by
  have hp : 0 < 1 - v := by linarith
  have h : 1/(1-v) ≤ Real.exp (v/(1-v)) := by
    have := Real.add_one_le_exp (v/(1-v))
    have e : 1/(1-v) = v/(1-v) + 1 := by field_simp; ring
    linarith
  rw [Real.exp_neg]
  calc (Real.exp (v/(1-v)))⁻¹ ≤ (1/(1-v))⁻¹ := inv_anti₀ (by positivity) h
    _ = 1 - v := by simp

theorem sm_side (F : Rounding) (hη : F.η = 0) (hηe : F.ηe = 0)
    (mS dS m' : ℝ) (l : List ℝ) (Al c1 A' : ℝ)
    (h1 : (l.map fun x => Real.exp (x - mS - (F.u*(mS - x) + Al))).sum ≤ dS)
    (h2 : dS ≤ (l.map fun x => Real.exp (x - mS + (F.u*(mS - x) + Al))).sum)
    (hcu : (1+F.u)*(1+F.ue) ≤ Real.exp c1) (hcl : Real.exp (-c1) ≤ (1-F.u)*(1-F.ue))
    (hA : Al + c1 + F.u*|mS - m'| ≤ F.u*(m' - mS) + A')
    (r2 r3 r4 : ℝ) (a2 : F.add (mS - m') r2) (a3 : F.exp r2 r3) (a4 : F.mul (dS * r3) r4) :
    (l.map fun x => Real.exp (x - m' - (F.u*(m' - x) + A'))).sum ≤ r4 ∧
    r4 ≤ (l.map fun x => Real.exp (x - m' + (F.u*(m' - x) + A'))).sum := by
  have hu0 := F.hu
  have hue0 := F.hue
  have hue1 := F.hue1
  have hu1 := F.hu1
  have hd0 : 0 ≤ dS := le_trans (sm_sum_nonneg _ _) h1
  have e2 := F.add_err a2
  obtain ⟨δ, e, hδ, he, hr4, -⟩ := a4
  have he0 : e = 0 := by rw [hη] at he; exact abs_nonpos_iff.1 he
  subst he0
  obtain ⟨b3l, b3u, b30⟩ := F.exp_bounds a3
  rw [hηe] at b3l b3u
  obtain ⟨e2l, e2u⟩ := abs_le.1 e2
  obtain ⟨hδl, hδu⟩ := abs_le.1 hδ
  have hε0 : 0 ≤ F.u * |mS - m'| := by positivity
  have x1 : Real.exp r2 ≤ Real.exp (mS - m' + F.u * |mS - m'|) :=
    Real.exp_le_exp.2 (by linarith)
  have x2 : Real.exp (mS - m' - F.u * |mS - m'|) ≤ Real.exp r2 :=
    Real.exp_le_exp.2 (by linarith)
  have hdr : 0 ≤ dS * r3 := mul_nonneg hd0 b30
  constructor
  · have A1 : dS * r3 * (1 - F.u) ≤ r4 := by
      have : 0 ≤ dS * r3 * (δ + F.u) := mul_nonneg hdr (by linarith)
      rw [hr4]; nlinarith
    have A2 : dS * (Real.exp r2 * (1 - F.ue)) * (1 - F.u) ≤ dS * r3 * (1 - F.u) :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (by linarith) hd0) (by linarith)
    have key : Real.exp (mS - m' - F.u * |mS - m'|) * Real.exp (-c1)
        ≤ Real.exp r2 * ((1 - F.ue) * (1 - F.u)) :=
      mul_le_mul x2 (by nlinarith [hcl]) (Real.exp_pos _).le (Real.exp_pos _).le
    have A3 : dS * (Real.exp (mS - m' - F.u * |mS - m'|) * Real.exp (-c1)) ≤
        dS * (Real.exp r2 * ((1 - F.ue) * (1 - F.u))) := mul_le_mul_of_nonneg_left key hd0
    have B := sm_sum_ge (l := l)
      (f := fun x => x - mS - (F.u*(mS - x) + Al)) (h := fun x => x - m' - (F.u*(m' - x) + A'))
      ((mS - m' - F.u * |mS - m'|) + (-c1)) (by intro x _; nlinarith)
    rw [Real.exp_add] at B
    have C := mul_le_mul_of_nonneg_right h1 (mul_nonneg (Real.exp_pos (mS - m' - F.u * |mS - m'|)).le (Real.exp_pos (-c1)).le)
    have A2' : dS * (Real.exp r2 * (1 - F.ue)) * (1 - F.u) = dS * (Real.exp r2 * ((1 - F.ue) * (1 - F.u))) := by ring
    linarith
  · have A1 : r4 ≤ dS * r3 * (1 + F.u) := by
      have : 0 ≤ dS * r3 * (F.u - δ) := mul_nonneg hdr (by linarith)
      rw [hr4]; nlinarith
    have A2 : dS * r3 * (1 + F.u) ≤ dS * (Real.exp r2 * (1 + F.ue)) * (1 + F.u) :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (by linarith) hd0) (by linarith)
    have key : Real.exp r2 * ((1 + F.ue) * (1 + F.u)) ≤
        Real.exp (mS - m' + F.u * |mS - m'|) * Real.exp c1 :=
      mul_le_mul x1 (by nlinarith [hcu]) (by positivity) (Real.exp_pos _).le
    have A3 : dS * (Real.exp r2 * ((1 + F.ue) * (1 + F.u))) ≤
        dS * (Real.exp (mS - m' + F.u * |mS - m'|) * Real.exp c1) :=
      mul_le_mul_of_nonneg_left key hd0
    have B := sm_sum_le (l := l)
      (f := fun x => x - mS + (F.u*(mS - x) + Al)) (h := fun x => x - m' + (F.u*(m' - x) + A'))
      ((mS - m' + F.u * |mS - m'|) + c1) (by intro x _; nlinarith)
    rw [Real.exp_add] at B
    have C := mul_le_mul_of_nonneg_right h2 (mul_nonneg (Real.exp_pos (mS - m' + F.u * |mS - m'|)).le (Real.exp_pos c1).le)
    have A2' : dS * (Real.exp r2 * (1 + F.ue)) * (1 + F.u) = dS * (Real.exp r2 * ((1 + F.ue) * (1 + F.u))) := by ring
    linarith

theorem sm_merge (F : Rounding) (hη : F.η = 0) (hηe : F.ηe = 0) (tau : ℝ) (htau : 0 ≤ tau)
    (L R st : State_a) (lL lR : List ℝ) (Al Ar A' c1 c2 g : ℝ)
    (hL1 : (lL.map fun x => Real.exp (x - L.m - (F.u*(L.m - x) + Al))).sum ≤ L.d)
    (hL2 : L.d ≤ (lL.map fun x => Real.exp (x - L.m + (F.u*(L.m - x) + Al))).sum)
    (hR1 : (lR.map fun x => Real.exp (x - R.m - (F.u*(R.m - x) + Ar))).sum ≤ R.d)
    (hR2 : R.d ≤ (lR.map fun x => Real.exp (x - R.m + (F.u*(R.m - x) + Ar))).sum)
    (hcu : (1+F.u)*(1+F.ue) ≤ Real.exp c1) (hcl : Real.exp (-c1) ≤ (1-F.u)*(1-F.ue))
    (hc2u : 1 + F.u ≤ Real.exp c2) (hc2l : Real.exp (-c2) ≤ 1 - F.u)
    (hg0 : 0 ≤ g) (hg : 2*F.u*(tau/(1-F.u)) ≤ g)
    (hAl : Al + c1 + c2 ≤ A') (hAr : Ar + g + c1 + c2 ≤ A')
    (hm : fmerge_a F tau L R st) :
    (st.m = L.m ∨ st.m = R.m) ∧
    (((lL ++ lR).map fun x => Real.exp (x - st.m - (F.u*(st.m - x) + A'))).sum ≤ st.d ∧
     st.d ≤ ((lL ++ lR).map fun x => Real.exp (x - st.m + (F.u*(st.m - x) + A'))).sum) := by
  have hu0 := F.hu
  have hu1 := F.hu1
  obtain ⟨v_m, v_d, r1, r2, r3, r4, r5, r6, r7, r8, a1, hvm, a2, a3, a4, a5, a6, a7, a8, hvd, hst⟩ := hm
  have hcase : (v_m = R.m ∧ L.m ≤ R.m) ∨ (v_m = L.m ∧ R.m - L.m ≤ tau/(1-F.u)) := by
    obtain ⟨δ, hδ, e1⟩ := a1
    obtain ⟨hδl, hδu⟩ := abs_le.1 hδ
    by_cases hc : r1 > tau
    · left
      rw [if_pos hc] at hvm
      refine ⟨hvm, ?_⟩
      by_contra hlt
      push_neg at hlt
      have : (R.m - L.m) * (1 + δ) ≤ 0 :=
        mul_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)
      linarith
    · right
      rw [if_neg hc] at hvm
      refine ⟨hvm, ?_⟩
      push_neg at hc
      by_cases hp : R.m - L.m ≤ 0
      · exact le_trans hp (div_nonneg htau (by linarith))
      · push_neg at hp
        rw [le_div_iff₀ (by linarith)]
        nlinarith
  have hA'' : ∃ A'', A'' = A' - c2 := ⟨_, rfl⟩
  obtain ⟨A'', hA''⟩ := hA''
  have hAL : Al + c1 + F.u*|L.m - v_m| ≤ F.u*(v_m - L.m) + A'' := by
    rcases hcase with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · rw [h1, abs_of_nonpos (by linarith)]; linarith
    · rw [h1]; simp; linarith
  have hAR : Ar + c1 + F.u*|R.m - v_m| ≤ F.u*(v_m - R.m) + A'' := by
    rcases hcase with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · rw [h1]; simp; linarith
    · rw [h1]
      by_cases hp : R.m ≤ L.m
      · rw [abs_of_nonpos (by linarith)]; linarith
      · push_neg at hp
        rw [abs_of_nonneg (by linarith)]
        have : 2 * F.u * (R.m - L.m) ≤ 2 * F.u * (tau / (1 - F.u)) :=
          mul_le_mul_of_nonneg_left h2 (by linarith)
        nlinarith
  obtain ⟨sL1, sL2⟩ := sm_side F hη hηe L.m L.d v_m lL Al c1 A'' hL1 hL2 hcu hcl hAL r2 r3 r4 a2 a3 a4
  obtain ⟨sR1, sR2⟩ := sm_side F hη hηe R.m R.d v_m lR Ar c1 A'' hR1 hR2 hcu hcl hAR r5 r6 r7 a5 a6 a7
  obtain ⟨δ, hδ, e8⟩ := a8
  obtain ⟨hδl, hδu⟩ := abs_le.1 hδ
  subst hst
  dsimp only
  rw [hvd]
  refine ⟨?_, ?_⟩
  · rcases hcase with ⟨h1, _⟩ | ⟨h1, _⟩
    · right; exact h1
    · left; exact h1
  simp only [List.map_append, List.sum_append]
  have n4 := le_trans (sm_sum_nonneg lL (fun x => x - v_m - (F.u*(v_m - x) + A''))) sL1
  have n7 := le_trans (sm_sum_nonneg lR (fun x => x - v_m - (F.u*(v_m - x) + A''))) sR1
  constructor
  · have U1 := sm_sum_ge (l := lL) (f := fun x => x - v_m - (F.u*(v_m - x) + A''))
      (h := fun x => x - v_m - (F.u*(v_m - x) + A')) (-c2) (by intro x _; linarith)
    have U2 := sm_sum_ge (l := lR) (f := fun x => x - v_m - (F.u*(v_m - x) + A''))
      (h := fun x => x - v_m - (F.u*(v_m - x) + A')) (-c2) (by intro x _; linarith)
    have P1 : (r4 + r7) * Real.exp (-c2) ≤ (r4 + r7) * (1 - F.u) :=
      mul_le_mul_of_nonneg_left hc2l (by linarith)
    have P2 : (r4 + r7) * (1 - F.u) ≤ r8 := by
      rw [e8]; have : 0 ≤ (r4 + r7) * (δ + F.u) := mul_nonneg (by linarith) (by linarith)
      nlinarith
    have P3 : ((lL.map fun x => Real.exp (x - v_m - (F.u*(v_m - x) + A''))).sum +
        (lR.map fun x => Real.exp (x - v_m - (F.u*(v_m - x) + A''))).sum) * Real.exp (-c2) ≤
        (r4 + r7) * Real.exp (-c2) := mul_le_mul_of_nonneg_right (by linarith) (Real.exp_pos _).le
    linarith
  · have U1 := sm_sum_le (l := lL) (f := fun x => x - v_m + (F.u*(v_m - x) + A''))
      (h := fun x => x - v_m + (F.u*(v_m - x) + A')) c2 (by intro x _; linarith)
    have U2 := sm_sum_le (l := lR) (f := fun x => x - v_m + (F.u*(v_m - x) + A''))
      (h := fun x => x - v_m + (F.u*(v_m - x) + A')) c2 (by intro x _; linarith)
    have P1 : (r4 + r7) * (1 + F.u) ≤ (r4 + r7) * Real.exp c2 :=
      mul_le_mul_of_nonneg_left hc2u (by linarith)
    have P2 : r8 ≤ (r4 + r7) * (1 + F.u) := by
      rw [e8]; have : 0 ≤ (r4 + r7) * (F.u - δ) := mul_nonneg (by linarith) (by linarith)
      nlinarith
    have P3 : (r4 + r7) * Real.exp c2 ≤
        ((lL.map fun x => Real.exp (x - v_m + (F.u*(v_m - x) + A''))).sum +
        (lR.map fun x => Real.exp (x - v_m + (F.u*(v_m - x) + A''))).sum) * Real.exp c2 :=
      mul_le_mul_of_nonneg_right (by linarith) (Real.exp_pos _).le
    rw [add_mul] at P3
    linarith

theorem sm_leaf (F : Rounding) (hη : F.η = 0) (hηe : F.ηe = 0) (D : ℕ) (tau A0 : ℝ)
    (hA0u : (1+F.u)^D*(1+F.ue) ≤ Real.exp A0) (hA0l : Real.exp (-A0) ≤ (1-F.u)^D*(1-F.ue))
    (A : List ℝ) (hA : A ≠ []) (st : State_a) (h : fblock_a F D tau A st) :
    st.m ∈ A ∧
    ((A.map fun x => Real.exp (x - st.m - (F.u*(st.m - x) + A0))).sum ≤ st.d ∧
     st.d ≤ (A.map fun x => Real.exp (x - st.m + (F.u*(st.m - x) + A0))).sum) := by
  have hu0 := F.hu
  have hue0 := F.hue
  have hue1 := F.hue1
  have hu1 := F.hu1
  obtain ⟨v_m, v_d, r1, r2, hr1, hvm, hfs, hvd, hst⟩ := h
  subst hst
  dsimp only
  rw [hvd]
  have hmax : ∀ x ∈ A, x ≤ v_m := by
    intro x hx; rw [hvm, hr1]; exact le_lmax hx
  have hmem : v_m ∈ A := by rw [hvm, hr1]; exact lmax_mem hA
  have hb := fsum_bounds F D _
    (fun x => Real.exp (x - v_m - F.u * |v_m - x|) * (1 - F.ue))
    (fun x => Real.exp (x - v_m + F.u * |v_m - x|) * (1 + F.ue))
    (fun x => mul_nonneg (Real.exp_pos _).le (by linarith)) ?_ A r2 hfs
  · rw [List.sum_map_mul_right, List.sum_map_mul_right] at hb
    obtain ⟨hb1, hb2⟩ := hb
    refine ⟨hmem, ?_, ?_⟩
    · have B := sm_sum_ge (l := A)
        (f := fun x => x - v_m - F.u * |v_m - x|)
        (h := fun x => x - v_m - (F.u * (v_m - x) + A0)) (-A0)
        (by intro x hx; rw [abs_of_nonneg (by linarith [hmax x hx])]; linarith)
      have S0 := sm_sum_nonneg A (fun x => x - v_m - F.u * |v_m - x|)
      have h1 : (1 - F.u) ^ D * 1 ≤ 1 := by
        have : (1 - F.u) ^ D ≤ 1 := pow_le_one₀ (by linarith) (by linarith)
        linarith
      have C : (A.map fun x => Real.exp (x - v_m - F.u * |v_m - x|)).sum * Real.exp (-A0) ≤
          (A.map fun x => Real.exp (x - v_m - F.u * |v_m - x|)).sum * ((1 - F.u)^D * (1 - F.ue)) :=
        mul_le_mul_of_nonneg_left hA0l S0
      have e : (1 - F.u) ^ D * ((A.map fun x => Real.exp (x - v_m - F.u * |v_m - x|)).sum * (1 - F.ue))
          = (A.map fun x => Real.exp (x - v_m - F.u * |v_m - x|)).sum * ((1 - F.u)^D * (1 - F.ue)) := by ring
      linarith
    · have B := sm_sum_le (l := A)
        (f := fun x => x - v_m + F.u * |v_m - x|)
        (h := fun x => x - v_m + (F.u * (v_m - x) + A0)) (A0)
        (by intro x hx; rw [abs_of_nonneg (by linarith [hmax x hx])]; linarith)
      have S0 := sm_sum_nonneg A (fun x => x - v_m + F.u * |v_m - x|)
      have C : (A.map fun x => Real.exp (x - v_m + F.u * |v_m - x|)).sum * ((1 + F.u)^D * (1 + F.ue)) ≤
          (A.map fun x => Real.exp (x - v_m + F.u * |v_m - x|)).sum * Real.exp A0 :=
        mul_le_mul_of_nonneg_left hA0u S0
      have e : (1 + F.u) ^ D * ((A.map fun x => Real.exp (x - v_m + F.u * |v_m - x|)).sum * (1 + F.ue))
          = (A.map fun x => Real.exp (x - v_m + F.u * |v_m - x|)).sum * ((1 + F.u)^D * (1 + F.ue)) := by ring
      linarith
  · intro x y ⟨q1, q2, h1, h2, h3⟩
    have e1 := F.add_err h1
    obtain ⟨b1, b2, b3⟩ := F.exp_bounds h2
    rw [hηe] at b1 b2
    have hab : |x - v_m| = |v_m - x| := abs_sub_comm _ _
    rw [hab] at e1
    obtain ⟨e1l, e1u⟩ := abs_le.1 e1
    have x1 : Real.exp (x - v_m - F.u * |v_m - x|) ≤ Real.exp q1 := Real.exp_le_exp.2 (by linarith)
    have x2 : Real.exp q1 ≤ Real.exp (x - v_m + F.u * |v_m - x|) := Real.exp_le_exp.2 (by linarith)
    subst h3
    constructor
    · have := mul_le_mul_of_nonneg_right x1 (show 0 ≤ 1 - F.ue by linarith)
      linarith
    · have := mul_le_mul_of_nonneg_right x2 (show 0 ≤ 1 + F.ue by linarith)
      linarith

theorem sm_tree (F : Rounding) (hη : F.η = 0) (hηe : F.ηe = 0) (D : ℕ) (tau : ℝ) (htau : 0 ≤ tau)
    (A0 c1 c2 g : ℝ) (hc1 : 0 ≤ c1) (hc2 : 0 ≤ c2) (hg0 : 0 ≤ g)
    (hA0u : (1+F.u)^D*(1+F.ue) ≤ Real.exp A0) (hA0l : Real.exp (-A0) ≤ (1-F.u)^D*(1-F.ue))
    (hcu : (1+F.u)*(1+F.ue) ≤ Real.exp c1) (hcl : Real.exp (-c1) ≤ (1-F.u)*(1-F.ue))
    (hc2u : 1 + F.u ≤ Real.exp c2) (hc2l : Real.exp (-c2) ≤ 1 - F.u)
    (hg : 2*F.u*(tau/(1-F.u)) ≤ g) (t : Tree ℝ) :
    (∀ b ∈ t.leaves, b ≠ []) → ∀ st, Tree.FEval (fblock_a F D tau) (fmerge_a F tau) t st →
    st.m ∈ t.flat ∧
    ((t.flat.map fun x => Real.exp (x - st.m - (F.u*(st.m - x) +
        (A0 + (c1+c2)*(t.depth:ℝ) + g*(rdepth t:ℝ))))).sum ≤ st.d ∧
     st.d ≤ (t.flat.map fun x => Real.exp (x - st.m + (F.u*(st.m - x) +
        (A0 + (c1+c2)*(t.depth:ℝ) + g*(rdepth t:ℝ))))).sum) := by
  induction t with
  | leaf b =>
    intro hb st h
    cases h with
    | leaf h =>
      have e : (A0 + (c1+c2)*((Tree.leaf b).depth:ℝ) + g*(rdepth (Tree.leaf b):ℝ)) = A0 := by
        simp [Tree.depth, rdepth]
      rw [e]
      exact sm_leaf F hη hηe D tau A0 hA0u hA0l b (hb b (by simp [Tree.leaves])) st h
  | node l r ihl ihr =>
    intro hb st h
    cases h with
    | node hl hr hm =>
      rename_i a c
      have hbl : ∀ b ∈ l.leaves, b ≠ [] := fun b h => hb b (by simp [Tree.leaves, h])
      have hbr : ∀ b ∈ r.leaves, b ≠ [] := fun b h => hb b (by simp [Tree.leaves, h])
      obtain ⟨ml, pl1, pl2⟩ := ihl hbl a hl
      obtain ⟨mr, pr1, pr2⟩ := ihr hbr c hr
      have dl : ((l.depth : ℕ) : ℝ) + 1 ≤ ((Tree.node l r).depth : ℕ) := by
        simp only [Tree.depth]
        have : l.depth + 1 ≤ max l.depth r.depth + 1 := by omega
        exact_mod_cast this
      have dr : ((r.depth : ℕ) : ℝ) + 1 ≤ ((Tree.node l r).depth : ℕ) := by
        simp only [Tree.depth]
        have : r.depth + 1 ≤ max l.depth r.depth + 1 := by omega
        exact_mod_cast this
      have rl : ((rdepth l : ℕ) : ℝ) ≤ ((rdepth (Tree.node l r) : ℕ) : ℝ) := by
        simp only [rdepth]
        exact_mod_cast le_max_left _ _
      have rr : ((rdepth r : ℕ) : ℝ) + 1 ≤ ((rdepth (Tree.node l r) : ℕ) : ℝ) := by
        simp only [rdepth]
        have : rdepth r + 1 ≤ max (rdepth l) (rdepth r + 1) := le_max_right _ _
        exact_mod_cast this
      have m1 := mul_le_mul_of_nonneg_left dl (show 0 ≤ c1 + c2 by linarith)
      have m2 := mul_le_mul_of_nonneg_left dr (show 0 ≤ c1 + c2 by linarith)
      have m3 := mul_le_mul_of_nonneg_left rl hg0
      have m4 := mul_le_mul_of_nonneg_left rr hg0
      obtain ⟨hmem, hP⟩ := sm_merge F hη hηe tau htau a c st l.flat r.flat
        (A0 + (c1+c2)*(l.depth:ℝ) + g*(rdepth l:ℝ))
        (A0 + (c1+c2)*(r.depth:ℝ) + g*(rdepth r:ℝ))
        (A0 + (c1+c2)*((Tree.node l r).depth:ℝ) + g*(rdepth (Tree.node l r):ℝ))
        c1 c2 g pl1 pl2 pr1 pr2 hcu hcl hc2u hc2l hg0 hg (by linarith) (by linarith) hm
      refine ⟨?_, hP⟩
      simp only [Tree.flat, List.mem_append]
      rcases hmem with h | h
      · left; rw [h]; exact ml
      · right; rw [h]; exact mr

theorem sm_sum_sub (X : List ℝ) (m : ℝ) :
    (X.map fun x => Real.exp (x - m)).sum = (X.map Real.exp).sum * Real.exp (-m) := by
  induction X with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, add_mul] at ih ⊢
    rw [ih, Real.exp_sub, Real.exp_neg, div_eq_mul_inv]

theorem sm_final (u : ℝ) (hu : 0 ≤ u) (X : List ℝ) (hX : X ≠ []) (lo hi : ℝ)
    (hb : ∀ x ∈ X, lo ≤ x ∧ x ≤ hi) (m d A : ℝ) (hm : m ∈ X)
    (h1 : (X.map fun x => Real.exp (x - m - (u*(m - x) + A))).sum ≤ d)
    (h2 : d ≤ (X.map fun x => Real.exp (x - m + (u*(m - x) + A))).sum) :
    |Real.log (d * Real.exp m) - Real.log ((X.map Real.exp).sum)| ≤ u * (hi - lo) + A := by
  have hS : 0 < (X.map Real.exp).sum := by
    apply List.sum_pos
    · intro y hy
      simp only [List.mem_map] at hy
      obtain ⟨x, _, rfl⟩ := hy
      exact Real.exp_pos _
    · simpa using hX
  have hmb := hb m hm
  have hsum := sm_sum_sub X m
  have hH : ∀ x ∈ X, u * (m - x) ≤ u * (hi - lo) := by
    intro x hx
    exact mul_le_mul_of_nonneg_left (by linarith [(hb x hx).1]) hu
  have B1 := sm_sum_ge (l := X) (f := fun x => x - m) (h := fun x => x - m + (u*(m - x) + A))
    (u * (hi - lo) + A) (by intro x hx; linarith [hH x hx])
  have B2 := sm_sum_le (l := X) (f := fun x => x - m) (h := fun x => x - m - (u*(m - x) + A))
    (-(u * (hi - lo) + A)) (by intro x hx; linarith [hH x hx])
  rw [hsum] at B1 B2
  have hE := Real.exp_pos m
  have hP : 0 < (X.map Real.exp).sum * Real.exp (-m) * Real.exp (-(u * (hi - lo) + A)) := by positivity
  have hd : 0 < d := by linarith
  have l1 : Real.log d ≤ Real.log ((X.map Real.exp).sum * Real.exp (-m) * Real.exp (u * (hi - lo) + A)) :=
    Real.log_le_log hd (by linarith)
  have l2 : Real.log ((X.map Real.exp).sum * Real.exp (-m) * Real.exp (-(u * (hi - lo) + A))) ≤ Real.log d :=
    Real.log_le_log hP (by linarith)
  have hp1 : 0 < (X.map Real.exp).sum * Real.exp (-m) := mul_pos hS (Real.exp_pos _)
  have e1 : Real.log ((X.map Real.exp).sum * Real.exp (-m) * Real.exp (u * (hi - lo) + A))
      = Real.log (X.map Real.exp).sum - m + (u * (hi - lo) + A) := by
    rw [Real.log_mul hp1.ne' (Real.exp_pos _).ne', Real.log_mul hS.ne' (Real.exp_pos _).ne',
      Real.log_exp, Real.log_exp]; ring
  have e2 : Real.log ((X.map Real.exp).sum * Real.exp (-m) * Real.exp (-(u * (hi - lo) + A)))
      = Real.log (X.map Real.exp).sum - m - (u * (hi - lo) + A) := by
    rw [Real.log_mul hp1.ne' (Real.exp_pos _).ne', Real.log_mul hS.ne' (Real.exp_pos _).ne',
      Real.log_exp, Real.log_exp]; ring
  rw [e1] at l1
  rw [e2] at l2
  rw [Real.log_mul hd.ne' hE.ne', Real.log_exp]
  rw [abs_le]
  constructor <;> linarith
theorem sm_znorm (st : State_a) : znorm st = st.d * Real.exp st.m := by
  unfold znorm
  rw [zero_sub, Real.exp_neg, one_div, inv_div, div_inv_eq_mul]

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (tau : ℝ), 0 ≤ tau → ∀ (lo hi : ℝ) (X : List ℝ),
    X ≠ [] → (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree ℝ), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    (X.length : ℝ) * Real.exp (tau * (rdepth t : ℝ)) * (F.η + F.ηe) ≤ F.u →
    F.η = 0 → F.ηe = 0 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D tau) (fmerge_a F tau) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((11 : ℝ) * ((1 : ℝ) / 10)) * ((F.u * (D : ℝ)) + ((2 : ℝ) * F.u * (t.depth : ℝ)) + F.ue + (F.ue * (t.depth : ℝ)) + ((2 : ℝ) * F.u * (hi - lo)) + ((2 : ℝ) * F.u * tau * (rdepth t : ℝ)) + ((3 : ℝ) * F.u))) := by
  intro F D tau htau lo hi X hX hbd hsm hlen t htX hleaves hdep hlen2 hη hηe st hev
  obtain ⟨hu256, hue64, -, -, -⟩ := hsm
  subst htX
  have hu0 := F.hu
  have hu1 := F.hu1
  have hue0 := F.hue
  have hue1 := F.hue1
  have eu := sm_exp_up hu0 hu1
  have el := sm_exp_lo hu0 hu1
  have eue := sm_exp_up hue0 hue1
  have elue := sm_exp_lo hue0 hue1
  have hcu : (1+F.u)*(1+F.ue) ≤ Real.exp (F.u/(1-F.u) + F.ue/(1-F.ue)) := by
    rw [Real.exp_add]; exact mul_le_mul eu eue (by linarith) (Real.exp_pos _).le
  have hcl : Real.exp (-(F.u/(1-F.u) + F.ue/(1-F.ue))) ≤ (1-F.u)*(1-F.ue) := by
    rw [neg_add, Real.exp_add]; exact mul_le_mul el elue (Real.exp_pos _).le (by linarith)
  have hA0u : (1+F.u)^D*(1+F.ue) ≤ Real.exp ((D:ℝ) * (F.u/(1-F.u)) + F.ue/(1-F.ue)) := by
    have : (1+F.u)^D ≤ Real.exp ((D:ℝ) * (F.u/(1-F.u))) := by
      rw [Real.exp_nat_mul]; exact pow_le_pow_left₀ (by linarith) eu D
    rw [Real.exp_add]; exact mul_le_mul this eue (by linarith) (Real.exp_pos _).le
  have hA0l : Real.exp (-((D:ℝ) * (F.u/(1-F.u)) + F.ue/(1-F.ue))) ≤ (1-F.u)^D*(1-F.ue) := by
    have h1 : Real.exp (-((D:ℝ) * (F.u/(1-F.u)))) ≤ (1-F.u)^D := by
      have : Real.exp (-((D:ℝ) * (F.u/(1-F.u)))) = Real.exp (-(F.u/(1-F.u)))^D := by
        rw [← Real.exp_nat_mul]; congr 1; ring
      rw [this]; exact pow_le_pow_left₀ (Real.exp_pos _).le el D
    rw [neg_add, Real.exp_add]; exact mul_le_mul h1 elue (Real.exp_pos _).le (by positivity)
  have hg : 2*F.u*(tau/(1-F.u)) ≤ 2*(F.u/(1-F.u))*tau := le_of_eq (by ring)
  have hu' : F.u/(1-F.u) ≤ 11/10 * F.u := by
    rw [div_le_iff₀ (by linarith)]; nlinarith
  have hue' : F.ue/(1-F.ue) ≤ 11/10 * F.ue := by
    rw [div_le_iff₀ (by linarith)]; nlinarith
  have hu'0 : 0 ≤ F.u/(1-F.u) := div_nonneg hu0 (by linarith)
  have hue'0 : 0 ≤ F.ue/(1-F.ue) := div_nonneg hue0 (by linarith)
  obtain ⟨⟨hmem, hP1, hP2⟩⟩ : Nonempty (_ ∧ _ ∧ _) := ⟨sm_tree F hη hηe D tau htau
    ((D:ℝ) * (F.u/(1-F.u)) + F.ue/(1-F.ue)) (F.u/(1-F.u) + F.ue/(1-F.ue)) (F.u/(1-F.u))
    (2*(F.u/(1-F.u))*tau) (by linarith) hu'0 (by positivity) hA0u hA0l hcu hcl eu el hg t
    hleaves st hev⟩
  have hfin := sm_final F.u hu0 t.flat hX lo hi hbd st.m st.d _ hmem hP1 hP2
  rw [sm_znorm]
  obtain ⟨x0, hx0⟩ := List.exists_mem_of_ne_nil _ hX
  have hlh : 0 ≤ hi - lo := by linarith [hbd x0 hx0]
  have hdn : (0:ℝ) ≤ (t.depth : ℝ) := Nat.cast_nonneg _
  have hrn : (0:ℝ) ≤ (rdepth t : ℝ) := Nat.cast_nonneg _
  have hDn : (0:ℝ) ≤ (D : ℝ) := Nat.cast_nonneg _
  have k1 := mul_le_mul_of_nonneg_left hu' hDn
  have k2 := mul_le_mul_of_nonneg_right (show 2 * (F.u/(1-F.u)) + F.ue/(1-F.ue) + F.u/(1-F.u)
      ≤ 3 * (11/10 * F.u) + 11/10 * F.ue by linarith) hdn
  have k3 := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right
      (show 2 * (F.u/(1-F.u)) ≤ 2 * (11/10 * F.u) by linarith) htau) hrn
  have k4 := mul_nonneg hu0 hlh
  have k5 := mul_nonneg hue0 hdn
  refine le_trans hfin ?_
  nlinarith

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxConditionalFpNounderflow
