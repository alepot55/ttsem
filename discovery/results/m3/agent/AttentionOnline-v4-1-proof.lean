import FPStatements

namespace Discovery.FP.AttentionOnline

open Discovery Discovery.Library.AttentionOnline

lemma theta_bd {y δ κ ue : ℝ} (hy : |y| ≤ κ) (hδ : |δ| ≤ ue) (hue : ue < 1) :
    |Real.exp y * (1 + δ) - 1| ≤ Real.exp (κ + ue) - 1 := by
  have hy' := abs_le.mp hy
  have hδ' := abs_le.mp hδ
  have h1 : Real.exp y ≤ Real.exp κ := Real.exp_le_exp.mpr hy'.2
  have h2 : Real.exp (-κ) ≤ Real.exp y := Real.exp_le_exp.mpr (by linarith [hy'.1])
  have h3 : 1 + ue ≤ Real.exp ue := by linarith [Real.add_one_le_exp ue]
  have h4 : -κ + 1 ≤ Real.exp (-κ) := Real.add_one_le_exp (-κ)
  have h5 : κ + ue + 1 ≤ Real.exp (κ + ue) := Real.add_one_le_exp _
  have hpos : 0 < Real.exp y := Real.exp_pos y
  have hk : 0 ≤ κ := le_trans (abs_nonneg _) hy
  have hue0 : 0 ≤ ue := le_trans (abs_nonneg _) hδ
  rw [abs_le]
  constructor
  · have : (1 - κ) * (1 - ue) ≤ Real.exp y * (1 + δ) := by
      calc (1 - κ) * (1 - ue) ≤ Real.exp (-κ) * (1 - ue) := by
            apply mul_le_mul_of_nonneg_right (by linarith) (by linarith)
        _ ≤ Real.exp y * (1 - ue) := by
            apply mul_le_mul_of_nonneg_right h2 (by linarith)
        _ ≤ Real.exp y * (1 + δ) := by
            apply mul_le_mul_of_nonneg_left (by linarith) hpos.le
    nlinarith [mul_nonneg hk hue0]
  · have : Real.exp y * (1 + δ) ≤ Real.exp κ * Real.exp ue := by
      calc Real.exp y * (1 + δ) ≤ Real.exp κ * (1 + ue) := by
            apply mul_le_mul h1 (by linarith) (by linarith) (Real.exp_pos _).le
        _ ≤ Real.exp κ * Real.exp ue := by
            apply mul_le_mul_of_nonneg_left h3 (Real.exp_pos _).le
    rw [← Real.exp_add] at this
    linarith

lemma side_core (F : Rounding) {mL m aL ZL BL eL γL γ0 r1 r2 r3 : ℝ}
    (hm : mL ≤ m) (hB : 0 ≤ BL) (hZ : |ZL| ≤ BL) (he : 0 ≤ eL)
    (ha : |aL - ZL| ≤ (Real.exp γL - 1) * BL + eL)
    (hγ : γL + F.u * (m - mL) ≤ γ0)
    (h1 : F.add (mL - m) r1) (h2 : F.exp r1 r2) (h3 : F.mul (aL * r2) r3) :
    |r3 - ZL * Real.exp (mL - m)| ≤
      BL * Real.exp (mL - m) * ((1 + F.u) * Real.exp (γ0 + F.ue) - 1)
        + (1 + F.u) * ((Real.exp F.ue + F.ηe) * eL + Real.exp γ0 * BL * F.ηe) + F.η := by
  obtain ⟨δ0, hδ0, hr1⟩ := h1
  obtain ⟨δ1, e1, hδ1, he1, hr2, hr2nn⟩ := h2
  obtain ⟨δ2, e2, hδ2, he2, hr3, -⟩ := h3
  have hu := F.hu
  have hu1 := F.hu1
  have hue := F.hue
  have hue1 := F.hue1
  have hηe := F.hηe
  have hη := F.hη
  have hE : 0 < Real.exp (mL - m) := Real.exp_pos _
  have hE1 : Real.exp (mL - m) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hκ0 : 0 ≤ F.u * (m - mL) := mul_nonneg hu (by linarith)
  have hexp : Real.exp r1 = Real.exp (mL - m) * Real.exp ((mL - m) * δ0) := by
    rw [hr1, ← Real.exp_add]; congr 1; ring
  have hy : |(mL - m) * δ0| ≤ F.u * (m - mL) := by
    rw [abs_mul, abs_of_nonpos (by linarith : mL - m ≤ 0)]
    nlinarith [abs_nonneg δ0]
  have hθ := theta_bd hy hδ1 hue1
  have hδ1' := abs_le.mp hδ1
  have hθpos : 0 < Real.exp ((mL - m) * δ0) * (1 + δ1) :=
    mul_pos (Real.exp_pos _) (by linarith)
  have hr2' : r2 = Real.exp (mL - m) * (Real.exp ((mL - m) * δ0) * (1 + δ1)) + e1 := by
    rw [hr2, hexp]; ring
  generalize hθdef : Real.exp ((mL - m) * δ0) * (1 + δ1) = θ at hθ hθpos hr2'
  generalize hXdef : Real.exp (F.u * (m - mL) + F.ue) = X at hθ
  have hθ1 := abs_le.mp hθ
  have hX : Real.exp (F.u * (m - mL) + F.ue) = X := hXdef
  have hEX : Real.exp (mL - m) * X ≤ Real.exp F.ue := by
    rw [← hX, ← Real.exp_add]
    apply Real.exp_le_exp.mpr
    nlinarith
  have hγ0 : γL ≤ γ0 := by linarith
  have hexpγ : Real.exp γL * X ≤ Real.exp (γ0 + F.ue) := by
    rw [← hX, ← Real.exp_add]
    apply Real.exp_le_exp.mpr
    linarith
  have hexpγ' : Real.exp γL ≤ Real.exp γ0 := Real.exp_le_exp.mpr hγ0
  have hXpos : 0 < X := by rw [← hX]; exact Real.exp_pos _
  have hP : 0 ≤ (Real.exp γL - 1) * BL + eL := le_trans (abs_nonneg _) ha
  have haL : |aL| ≤ Real.exp γL * BL + eL := by
    calc |aL| = |ZL + (aL - ZL)| := by ring_nf
      _ ≤ |ZL| + |aL - ZL| := abs_add_le _ _
      _ ≤ BL + ((Real.exp γL - 1) * BL + eL) := by linarith
      _ = Real.exp γL * BL + eL := by ring
  -- x := aL * r2
  have hx : |aL * r2 - ZL * Real.exp (mL - m)| ≤
      BL * Real.exp (mL - m) * (Real.exp (γ0 + F.ue) - 1)
        + ((Real.exp F.ue + F.ηe) * eL + Real.exp γ0 * BL * F.ηe) := by
    have eq : aL * r2 - ZL * Real.exp (mL - m) =
        (aL - ZL) * (Real.exp (mL - m) * θ) + ZL * Real.exp (mL - m) * (θ - 1) + aL * e1 := by
      rw [hr2']; ring
    rw [eq]
    have t1 : |(aL - ZL) * (Real.exp (mL - m) * θ)| ≤
        ((Real.exp γL - 1) * BL + eL) * (Real.exp (mL - m) * X) := by
      rw [abs_mul, abs_of_pos (mul_pos hE hθpos)]
      apply mul_le_mul ha _ (mul_pos hE hθpos).le hP
      apply mul_le_mul_of_nonneg_left _ hE.le
      linarith [hθ1.2]
    have t2 : |ZL * Real.exp (mL - m) * (θ - 1)| ≤ BL * Real.exp (mL - m) * (X - 1) := by
      rw [abs_mul, abs_mul, abs_of_pos hE]
      have : |θ - 1| ≤ X - 1 := hθ
      calc |ZL| * Real.exp (mL - m) * |θ - 1| ≤ BL * Real.exp (mL - m) * (X - 1) := by
            apply mul_le_mul (mul_le_mul_of_nonneg_right hZ hE.le) this (abs_nonneg _)
              (mul_nonneg hB hE.le)
        _ = _ := rfl
    have t3 : |aL * e1| ≤ (Real.exp γL * BL + eL) * F.ηe := by
      rw [abs_mul]
      apply mul_le_mul haL (le_trans he1 le_rfl) (abs_nonneg _) (by positivity)
    calc _ ≤ |(aL - ZL) * (Real.exp (mL - m) * θ)| + |ZL * Real.exp (mL - m) * (θ - 1)| + |aL * e1| :=
          abs_add_three _ _ _
      _ ≤ ((Real.exp γL - 1) * BL + eL) * (Real.exp (mL - m) * X) +
            BL * Real.exp (mL - m) * (X - 1) + (Real.exp γL * BL + eL) * F.ηe := by linarith
      _ ≤ _ := by
        have h1 : (Real.exp γL * X - 1) * (BL * Real.exp (mL - m)) ≤
            (Real.exp (γ0 + F.ue) - 1) * (BL * Real.exp (mL - m)) :=
          mul_le_mul_of_nonneg_right (by linarith) (mul_nonneg hB hE.le)
        have h2 : eL * (Real.exp (mL - m) * X) ≤ eL * Real.exp F.ue :=
          mul_le_mul_of_nonneg_left hEX he
        have h3 : Real.exp γL * BL * F.ηe ≤ Real.exp γ0 * BL * F.ηe := by
          apply mul_le_mul_of_nonneg_right _ hηe
          exact mul_le_mul_of_nonneg_right hexpγ' hB
        linarith
  -- rounding of the product
  have hr3' : |r3 - aL * r2| ≤ F.u * |aL * r2| + F.η := by
    have : r3 - aL * r2 = aL * r2 * δ2 + e2 := by rw [hr3]; ring
    rw [this]
    calc |aL * r2 * δ2 + e2| ≤ |aL * r2 * δ2| + |e2| := abs_add_le _ _
      _ ≤ F.u * |aL * r2| + F.η := by
        rw [abs_mul]
        have : |aL * r2| * |δ2| ≤ |aL * r2| * F.u := mul_le_mul_of_nonneg_left hδ2 (abs_nonneg _)
        linarith
  have hxabs : |aL * r2| ≤ BL * Real.exp (mL - m) + |aL * r2 - ZL * Real.exp (mL - m)| := by
    calc |aL * r2| = |ZL * Real.exp (mL - m) + (aL * r2 - ZL * Real.exp (mL - m))| := by ring_nf
      _ ≤ |ZL * Real.exp (mL - m)| + _ := abs_add_le _ _
      _ ≤ _ := by
        rw [abs_mul, abs_of_pos hE]
        have := mul_le_mul_of_nonneg_right hZ hE.le
        linarith
  have : |r3 - ZL * Real.exp (mL - m)| ≤
      |aL * r2 - ZL * Real.exp (mL - m)| + |r3 - aL * r2| := by
    calc _ = |(aL * r2 - ZL * Real.exp (mL - m)) + (r3 - aL * r2)| := by ring_nf
      _ ≤ _ := abs_add_le _ _
  have hBE : 0 ≤ BL * Real.exp (mL - m) := mul_nonneg hB hE.le
  have k1 := mul_le_mul_of_nonneg_left hxabs hu
  have k2 := mul_le_mul_of_nonneg_left hx hu
  linarith


lemma merge_core (F : Rounding) {mL mR m aL aR ZL ZR BL BR eL eR γL γR γ0 r1 r2 r3 r4 r5 r6 r7 : ℝ}
    (hmL : mL ≤ m) (hmR : mR ≤ m) (hBL : 0 ≤ BL) (hBR : 0 ≤ BR)
    (hZL : |ZL| ≤ BL) (hZR : |ZR| ≤ BR) (heL : 0 ≤ eL) (heR : 0 ≤ eR)
    (haL : |aL - ZL| ≤ (Real.exp γL - 1) * BL + eL)
    (haR : |aR - ZR| ≤ (Real.exp γR - 1) * BR + eR)
    (hγL : γL + F.u * (m - mL) ≤ γ0) (hγR : γR + F.u * (m - mR) ≤ γ0)
    (h1 : F.add (mL - m) r1) (h2 : F.exp r1 r2) (h3 : F.mul (aL * r2) r3)
    (h4 : F.add (mR - m) r4) (h5 : F.exp r4 r5) (h6 : F.mul (aR * r5) r6)
    (h7 : F.add (r3 + r6) r7) :
    |r7 - (ZL * Real.exp (mL - m) + ZR * Real.exp (mR - m))| ≤
      (Real.exp (γ0 + F.ue + 2 * F.u) - 1) * (BL * Real.exp (mL - m) + BR * Real.exp (mR - m))
        + ((1 + F.u) ^ 2 * ((Real.exp F.ue + F.ηe) * (eL + eR) + Real.exp γ0 * (BL + BR) * F.ηe)
            + 2 * (1 + F.u) * F.η) := by
  have sL := side_core F hmL hBL hZL heL haL hγL h1 h2 h3
  have sR := side_core F hmR hBR hZR heR haR hγR h4 h5 h6
  obtain ⟨δ3, hδ3, hr7⟩ := h7
  have hu := F.hu
  have hEL : 0 < Real.exp (mL - m) := Real.exp_pos _
  have hER : 0 < Real.exp (mR - m) := Real.exp_pos _
  have hBn : 0 ≤ BL * Real.exp (mL - m) + BR * Real.exp (mR - m) :=
    add_nonneg (mul_nonneg hBL hEL.le) (mul_nonneg hBR hER.le)
  have hZn : |ZL * Real.exp (mL - m) + ZR * Real.exp (mR - m)| ≤
      BL * Real.exp (mL - m) + BR * Real.exp (mR - m) := by
    calc _ ≤ |ZL * Real.exp (mL - m)| + |ZR * Real.exp (mR - m)| := abs_add_le _ _
      _ ≤ _ := by
        rw [abs_mul, abs_mul, abs_of_pos hEL, abs_of_pos hER]
        have := mul_le_mul_of_nonneg_right hZL hEL.le
        have := mul_le_mul_of_nonneg_right hZR hER.le
        linarith
  set Zn := ZL * Real.exp (mL - m) + ZR * Real.exp (mR - m) with hZn_def
  set Bn := BL * Real.exp (mL - m) + BR * Real.exp (mR - m) with hBn_def
  have hs : |r3 + r6 - Zn| ≤ |r3 - ZL * Real.exp (mL - m)| + |r6 - ZR * Real.exp (mR - m)| := by
    calc _ = |(r3 - ZL * Real.exp (mL - m)) + (r6 - ZR * Real.exp (mR - m))| := by
          rw [hZn_def]; ring_nf
      _ ≤ _ := abs_add_le _ _
  have hr7' : |r7 - Zn| ≤ (1 + F.u) * |r3 + r6 - Zn| + F.u * Bn := by
    have : r7 - Zn = (r3 + r6 - Zn) + (r3 + r6) * δ3 := by rw [hr7]; ring
    rw [this]
    have hab : |(r3 + r6) * δ3| ≤ F.u * |r3 + r6| := by
      rw [abs_mul]
      have := mul_le_mul_of_nonneg_left hδ3 (abs_nonneg (r3 + r6))
      linarith
    have : |r3 + r6| ≤ Bn + |r3 + r6 - Zn| := by
      calc |r3 + r6| = |Zn + (r3 + r6 - Zn)| := by ring_nf
        _ ≤ |Zn| + |r3 + r6 - Zn| := abs_add_le _ _
        _ ≤ _ := by linarith
    have := mul_le_mul_of_nonneg_left this hu
    calc _ ≤ |r3 + r6 - Zn| + |(r3 + r6) * δ3| := abs_add_le _ _
      _ ≤ _ := by linarith
  have hc : (1 + F.u) ^ 2 * Real.exp (γ0 + F.ue) ≤ Real.exp (γ0 + F.ue + 2 * F.u) := by
    have h1 : 1 + F.u ≤ Real.exp F.u := by linarith [Real.add_one_le_exp F.u]
    have h2 : (1 + F.u) ^ 2 ≤ Real.exp (2 * F.u) := by
      rw [show 2 * F.u = F.u + F.u by ring, Real.exp_add]
      nlinarith
    rw [Real.exp_add (γ0 + F.ue) (2 * F.u), mul_comm]
    exact mul_le_mul_of_nonneg_left h2 (Real.exp_pos _).le
  have hc' := mul_le_mul_of_nonneg_right hc hBn
  have hu1 := mul_le_mul_of_nonneg_left (add_le_add sL sR) (by linarith : 0 ≤ 1 + F.u)
  have hfin := mul_le_mul_of_nonneg_left hs (by linarith : 0 ≤ 1 + F.u)
  have e1 : BL * Real.exp (mL - m) * ((1 + F.u) * Real.exp (γ0 + F.ue) - 1) +
      BR * Real.exp (mR - m) * ((1 + F.u) * Real.exp (γ0 + F.ue) - 1) =
      Bn * ((1 + F.u) * Real.exp (γ0 + F.ue) - 1) := by rw [hBn_def]; ring
  nlinarith [hc', hu1, hfin, hr7', e1]


lemma abs_sum_le_aux {E : Type*} (ζ β : E → ℝ) (hζ : ∀ x, |ζ x| ≤ β x) (L : List E) :
    |(L.map ζ).sum| ≤ (L.map β).sum := by
  induction L with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    calc _ ≤ |ζ a| + |(l.map ζ).sum| := abs_add_le _ _
      _ ≤ _ := add_le_add (hζ a) ih

lemma sum_nonneg_aux {E : Type*} (β : E → ℝ) (hβ : ∀ x, 0 ≤ β x) (L : List E) :
    0 ≤ (L.map β).sum := by
  induction L with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    exact add_nonneg (hβ a) ih

lemma tree_sum (F : Rounding) {E : Type*} (term : E → ℝ → Prop) (ζ β b : E → ℝ) (c : ℝ)
    (hc : 0 ≤ c) (hβ : ∀ x, 0 ≤ β x) (hb : ∀ x, 0 ≤ b x) (hζ : ∀ x, |ζ x| ≤ β x)
    (P : E → Prop) (hterm : ∀ x, P x → ∀ y, term x y → |y - ζ x| ≤ (c - 1) * β x + b x) :
    ∀ (t : Discovery.Tree E), (∀ x ∈ t.flat, P x) → (∀ blk ∈ t.leaves, ∃ x, blk = [x]) →
      ∀ k : ℕ, t.depth ≤ k → ∀ w,
      Discovery.Tree.FEval (fun blk y => ∃ x, blk = [x] ∧ term x y)
        (fun a c y => F.add (a + c) y) t w →
      |w - (t.flat.map ζ).sum| ≤
        ((1 + F.u) ^ k * c - 1) * (t.flat.map β).sum + (1 + F.u) ^ k * (t.flat.map b).sum := by
  intro t
  induction t with
  | leaf blk =>
    intro hP hl k hk w hw
    cases hw with
    | leaf h =>
      obtain ⟨x, rfl, hx⟩ := h
      have h1 := hterm x (hP x (by simp [Discovery.Tree.flat])) w hx
      have hp : 1 ≤ (1 + F.u) ^ k := one_le_pow₀ (by linarith [F.hu])
      simp only [Discovery.Tree.flat, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
        add_zero]
      have := hβ x
      have := hb x
      nlinarith [mul_nonneg (sub_nonneg.mpr hp) hc, mul_nonneg (sub_nonneg.mpr hp) (hb x),
        mul_nonneg (mul_nonneg (sub_nonneg.mpr hp) hc) (hβ x)]
  | node l r ihl ihr =>
    intro hP hl k hk w hw
    have hPl : ∀ x ∈ l.flat, P x := fun x h => hP x (by simp [Discovery.Tree.flat, h])
    have hPr : ∀ x ∈ r.flat, P x := fun x h => hP x (by simp [Discovery.Tree.flat, h])
    have hl' : ∀ blk ∈ l.leaves, ∃ x, blk = [x] := fun blk h =>
      hl blk (by simp [Discovery.Tree.leaves, h])
    have hr' : ∀ blk ∈ r.leaves, ∃ x, blk = [x] := fun blk h =>
      hl blk (by simp [Discovery.Tree.leaves, h])
    simp only [Discovery.Tree.depth] at hk
    obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    cases hw with
    | node hwl hwr hm =>
      rename_i a c' 
      have A1 := ihl hPl hl' k' (by omega) a hwl
      have A2 := ihr hPr hr' k' (by omega) c' hwr
      obtain ⟨δ, hδ, hw⟩ := hm
      simp only [Discovery.Tree.flat, List.map_append, List.sum_append]
      have hu := F.hu
      have hZ := abs_sum_le_aux ζ β hζ (l.flat ++ r.flat)
      simp only [List.map_append, List.sum_append] at hZ
      have hP : 1 ≤ (1 + F.u) ^ k' := one_le_pow₀ (by linarith)
      set Zl := (l.flat.map ζ).sum
      set Zr := (r.flat.map ζ).sum
      set Bl := (l.flat.map β).sum
      set Br := (r.flat.map β).sum
      set bl := (l.flat.map b).sum
      set br := (r.flat.map b).sum
      have hBl : 0 ≤ Bl := sum_nonneg_aux β hβ _
      have hBr : 0 ≤ Br := sum_nonneg_aux β hβ _
      have hbl : 0 ≤ bl := sum_nonneg_aux b hb _
      have hbr : 0 ≤ br := sum_nonneg_aux b hb _
      have e1 : |a + c' - (Zl + Zr)| ≤ ((1 + F.u) ^ k' * c - 1) * (Bl + Br) +
          (1 + F.u) ^ k' * (bl + br) := by
        calc _ = |(a - Zl) + (c' - Zr)| := by ring_nf
          _ ≤ |a - Zl| + |c' - Zr| := abs_add_le _ _
          _ ≤ _ := by linarith
      have e2 : |w - (Zl + Zr)| ≤ (1 + F.u) * |a + c' - (Zl + Zr)| + F.u * (Bl + Br) := by
        have : w - (Zl + Zr) = (a + c' - (Zl + Zr)) + (a + c') * δ := by rw [hw]; ring
        rw [this]
        have hab : |(a + c') * δ| ≤ F.u * |a + c'| := by
          rw [abs_mul]
          have := mul_le_mul_of_nonneg_left hδ (abs_nonneg (a + c'))
          linarith
        have h3 : |a + c'| ≤ (Bl + Br) + |a + c' - (Zl + Zr)| := by
          calc |a + c'| = |(Zl + Zr) + (a + c' - (Zl + Zr))| := by ring_nf
            _ ≤ |Zl + Zr| + |a + c' - (Zl + Zr)| := abs_add_le _ _
            _ ≤ _ := by linarith
        have := mul_le_mul_of_nonneg_left h3 hu
        calc _ ≤ |a + c' - (Zl + Zr)| + |(a + c') * δ| := abs_add_le _ _
          _ ≤ _ := by linarith
      have e3 := mul_le_mul_of_nonneg_left e1 (by linarith : 0 ≤ 1 + F.u)
      rw [pow_succ]
      nlinarith [e2, e3]

lemma fsum_dev (F : Rounding) {E : Type*} (D : ℕ) (term : E → ℝ → Prop) (ζ β b : E → ℝ) (c : ℝ)
    (hc : 0 ≤ c) (hβ : ∀ x, 0 ≤ β x) (hb : ∀ x, 0 ≤ b x) (hζ : ∀ x, |ζ x| ≤ β x)
    (A : List E) (hterm : ∀ x ∈ A, ∀ y, term x y → |y - ζ x| ≤ (c - 1) * β x + b x) (w : ℝ)
    (h : fsum F D term A w) :
    |w - (A.map ζ).sum| ≤
      ((1 + F.u) ^ D * c - 1) * (A.map β).sum + (1 + F.u) ^ D * (A.map b).sum := by
  obtain ⟨t, rfl, hl, hd, hev⟩ := h
  exact tree_sum F term ζ β b c hc hβ hb hζ (fun x => x ∈ t.flat) hterm t (fun x h => h) hl D hd w hev

lemma term_d (F : Rounding) {lo M s q1 q2 : ℝ} (hlo : lo ≤ s) (hs : s ≤ M)
    (h1 : F.add (s - M) q1) (h2 : F.exp q1 q2) :
    |q2 - Real.exp (s - M) * 1| ≤
      (Real.exp (F.u * (M - lo) + F.ue) - 1) * (Real.exp (s - M) * 1) + F.ηe := by
  obtain ⟨δ0, hδ0, hr1⟩ := h1
  obtain ⟨δ1, e1, hδ1, he1, hr2, -⟩ := h2
  have hu := F.hu
  have hE : 0 < Real.exp (s - M) := Real.exp_pos _
  have hexp : Real.exp q1 = Real.exp (s - M) * Real.exp ((s - M) * δ0) := by
    rw [hr1, ← Real.exp_add]; congr 1; ring
  have hy : |(s - M) * δ0| ≤ F.u * (M - lo) := by
    rw [abs_mul, abs_of_nonpos (by linarith : s - M ≤ 0)]
    have := mul_le_mul_of_nonneg_left hδ0 (by linarith : 0 ≤ -(s - M))
    nlinarith [abs_nonneg δ0]
  have hθ := theta_bd hy hδ1 F.hue1
  have eq : q2 - Real.exp (s - M) * 1 =
      Real.exp (s - M) * (Real.exp ((s - M) * δ0) * (1 + δ1) - 1) + e1 := by
    rw [hr2, hexp]; ring
  rw [eq]
  calc _ ≤ |Real.exp (s - M) * (Real.exp ((s - M) * δ0) * (1 + δ1) - 1)| + |e1| := abs_add_le _ _
    _ ≤ _ := by
      rw [abs_mul, abs_of_pos hE]
      have := mul_le_mul_of_nonneg_left hθ hE.le
      linarith

lemma term_o (F : Rounding) {lo M s v V q1 q2 q3 : ℝ} (hlo : lo ≤ s) (hs : s ≤ M) (hv : |v| ≤ V)
    (h1 : F.add (s - M) q1) (h2 : F.exp q1 q2) (h3 : F.mul (q2 * v) q3) :
    |q3 - Real.exp (s - M) * v| ≤
      ((1 + F.u) * Real.exp (F.u * (M - lo) + F.ue) - 1) * (Real.exp (s - M) * |v|)
        + (F.η + (1 + F.u) * F.ηe * V) := by
  obtain ⟨δ0, hδ0, hr1⟩ := h1
  obtain ⟨δ1, e1, hδ1, he1, hr2, -⟩ := h2
  obtain ⟨δ2, e2, hδ2, he2, hr3, -⟩ := h3
  have hu := F.hu
  have hE : 0 < Real.exp (s - M) := Real.exp_pos _
  have hexp : Real.exp q1 = Real.exp (s - M) * Real.exp ((s - M) * δ0) := by
    rw [hr1, ← Real.exp_add]; congr 1; ring
  have hy : |(s - M) * δ0| ≤ F.u * (M - lo) := by
    rw [abs_mul, abs_of_nonpos (by linarith : s - M ≤ 0)]
    have := mul_le_mul_of_nonneg_left hδ0 (by linarith : 0 ≤ -(s - M))
    nlinarith [abs_nonneg δ0]
  have hθ := theta_bd hy hδ1 F.hue1
  generalize hτ : Real.exp (F.u * (M - lo) + F.ue) = X at hθ
  generalize hθdef : Real.exp ((s - M) * δ0) * (1 + δ1) = θ at hθ
  have hq2 : q2 = Real.exp (s - M) * θ + e1 := by rw [hr2, hexp, ← hθdef]; ring
  have eq : q3 - Real.exp (s - M) * v =
      (Real.exp (s - M) * v * (θ - 1)) * (1 + δ2) + e1 * v * (1 + δ2)
        + Real.exp (s - M) * v * δ2 + e2 := by
    rw [hr3, hq2]; ring
  have hδ2' := abs_le.mp hδ2
  have h12 : |1 + δ2| ≤ 1 + F.u := by rw [abs_le]; constructor <;> linarith
  have hV : 0 ≤ V := le_trans (abs_nonneg _) hv
  rw [eq]
  have t1 : |(Real.exp (s - M) * v * (θ - 1)) * (1 + δ2)| ≤
      (Real.exp (s - M) * |v|) * (X - 1) * (1 + F.u) := by
    rw [abs_mul, abs_mul, abs_mul, abs_of_pos hE]
    apply mul_le_mul _ h12 (abs_nonneg _) (by
      have := le_trans (abs_nonneg _) hθ
      positivity)
    apply mul_le_mul_of_nonneg_left hθ (by positivity)
  have t2 : |e1 * v * (1 + δ2)| ≤ F.ηe * V * (1 + F.u) := by
    rw [abs_mul, abs_mul]
    apply mul_le_mul _ h12 (abs_nonneg _) (mul_nonneg F.hηe hV)
    exact mul_le_mul he1 hv (abs_nonneg _) F.hηe
  have t3 : |Real.exp (s - M) * v * δ2| ≤ Real.exp (s - M) * |v| * F.u := by
    rw [abs_mul, abs_mul, abs_of_pos hE]
    exact mul_le_mul_of_nonneg_left hδ2 (by positivity)
  calc _ ≤ |(Real.exp (s - M) * v * (θ - 1)) * (1 + δ2)| + |e1 * v * (1 + δ2)|
        + |Real.exp (s - M) * v * δ2| + |e2| := by
        have := abs_add_le ((Real.exp (s - M) * v * (θ - 1)) * (1 + δ2) + e1 * v * (1 + δ2)
          + Real.exp (s - M) * v * δ2) e2
        have := abs_add_le ((Real.exp (s - M) * v * (θ - 1)) * (1 + δ2) + e1 * v * (1 + δ2))
          (Real.exp (s - M) * v * δ2)
        have := abs_add_le ((Real.exp (s - M) * v * (θ - 1)) * (1 + δ2)) (e1 * v * (1 + δ2))
        linarith
    _ ≤ _ := by nlinarith



lemma pow_le_exp_aux {u : ℝ} (hu : 0 ≤ u) (k : ℕ) : (1 + u) ^ k ≤ Real.exp (k * u) := by
  rw [Real.exp_nat_mul]
  exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp u]) k

lemma sum_const_aux {E : Type*} (A : List E) (c : ℝ) : (A.map fun _ => c).sum = A.length * c := by
  induction A with
  | nil => simp
  | cons a l ih => simp [List.sum_cons]; ring

lemma leaf_d (F : Rounding) (D : ℕ) (lo hi M : ℝ) (A : List (ℝ × ℝ)) (hA : A ≠ [])
    (hrange : ∀ e ∈ A, lo ≤ e.1 ∧ e.1 ≤ M) (hMhi : lo ≤ hi) (g : ℝ) (hg : D * F.u + F.ue ≤ g) (j : ℕ) (r2 : ℝ)
    (h : fsum F D (fun x y => ∃ q1 q2, F.add (x.1 - M) q1 ∧ F.exp q1 q2 ∧ y = q2) A r2) :
    |r2 - (A.map fun e => Real.exp (e.1 - M) * 1).sum| ≤
      (Real.exp (g + F.u * (M - lo)) - 1) * (A.map fun e => Real.exp (e.1 - M) * 1).sum
        + ((3 * (A.length : ℝ) - 2) * F.η * Real.exp g
          + (A.length : ℝ) * ((j : ℝ) + 1) * F.ηe * 1 * Real.exp (g + F.u * (hi - lo))) := by
  have hu := F.hu
  have hηe := F.hηe
  have hη := F.hη
  have hn : (1 : ℝ) ≤ A.length := by
    have : 0 < A.length := List.length_pos_iff.mpr hA
    exact_mod_cast this
  have hMlo : lo ≤ M := by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil A hA
    exact le_trans (hrange e he).1 (hrange e he).2
  have key := fsum_dev F D _ (fun e : ℝ × ℝ => Real.exp (e.1 - M) * 1)
    (fun e : ℝ × ℝ => Real.exp (e.1 - M) * 1) (fun _ => F.ηe)
    (Real.exp (F.u * (M - lo) + F.ue)) (Real.exp_pos _).le
    (fun x => by positivity) (fun _ => hηe)
    (fun x => by rw [abs_of_pos (by positivity)]) A
    (by
      intro x hx y ⟨q1, q2, h1, h2, hy⟩
      subst hy
      exact term_d F (hrange x hx).1 (hrange x hx).2 h1 h2) r2 h
  rw [sum_const_aux] at key
  have hB : 0 ≤ (A.map fun e => Real.exp (e.1 - M) * 1).sum :=
    sum_nonneg_aux _ (fun x => by positivity) A
  have hx0 : 0 ≤ F.u * (hi - lo) := mul_nonneg hu (by linarith)
  have hc : (1 + F.u) ^ D * Real.exp (F.u * (M - lo) + F.ue) ≤ Real.exp (g + F.u * (M - lo)) := by
    calc _ ≤ Real.exp (D * F.u) * Real.exp (F.u * (M - lo) + F.ue) :=
          mul_le_mul_of_nonneg_right (pow_le_exp_aux hu D) (Real.exp_pos _).le
      _ = Real.exp (D * F.u + F.ue + F.u * (M - lo)) := by rw [← Real.exp_add]; congr 1; ring
      _ ≤ _ := Real.exp_le_exp.mpr (by linarith)
  have hp : (1 + F.u) ^ D ≤ Real.exp (g + F.u * (hi - lo)) :=
    le_trans (pow_le_exp_aux hu D) (Real.exp_le_exp.mpr (by
      have : 0 ≤ (D : ℝ) * F.u := by positivity
      nlinarith [F.hue]))
  have e1 := mul_le_mul_of_nonneg_right (sub_le_sub_right hc 1) hB
  have e2 : (1 + F.u) ^ D * ((A.length : ℝ) * F.ηe) ≤
      (A.length : ℝ) * ((j : ℝ) + 1) * F.ηe * 1 * Real.exp (g + F.u * (hi - lo)) := by
    have hj : (1 : ℝ) ≤ (j : ℝ) + 1 := by have : (0 : ℝ) ≤ j := Nat.cast_nonneg j; linarith
    have h0 : 0 ≤ (A.length : ℝ) * F.ηe := by positivity
    calc (1 + F.u) ^ D * ((A.length : ℝ) * F.ηe) ≤ Real.exp (g + F.u * (hi - lo)) * ((A.length : ℝ) * F.ηe) :=
          mul_le_mul_of_nonneg_right hp h0
      _ ≤ Real.exp (g + F.u * (hi - lo)) * ((A.length : ℝ) * F.ηe * ((j : ℝ) + 1)) :=
          mul_le_mul_of_nonneg_left (by nlinarith) (Real.exp_pos _).le
      _ = _ := by ring
  have e3 : 0 ≤ (3 * (A.length : ℝ) - 2) * F.η * Real.exp g := by
    have : 0 ≤ 3 * (A.length : ℝ) - 2 := by linarith
    positivity
  linarith

lemma leaf_o (F : Rounding) (D : ℕ) (lo hi M V : ℝ) (A : List (ℝ × ℝ)) (hA : A ≠ [])
    (hrange : ∀ e ∈ A, lo ≤ e.1 ∧ e.1 ≤ M) (hv : ∀ e ∈ A, |e.2| ≤ V) (hMhi : lo ≤ hi)
    (g : ℝ) (hg : ((D : ℝ) + 1) * F.u + F.ue ≤ g) (j : ℕ) (r3 : ℝ)
    (h : fsum F D (fun x y => ∃ q1 q2 q3, F.add (x.1 - M) q1 ∧ F.exp q1 q2 ∧
      F.mul (q2 * x.2) q3 ∧ y = q3) A r3) :
    |r3 - (A.map fun e => Real.exp (e.1 - M) * e.2).sum| ≤
      (Real.exp (g + F.u * (M - lo)) - 1) * (A.map fun e => Real.exp (e.1 - M) * |e.2|).sum
        + ((3 * (A.length : ℝ) - 2) * F.η * Real.exp g
          + (A.length : ℝ) * ((j : ℝ) + 1) * F.ηe * V * Real.exp (g + F.u * (hi - lo))) := by
  have hu := F.hu
  have hηe := F.hηe
  have hη := F.hη
  have hn : (1 : ℝ) ≤ A.length := by
    have : 0 < A.length := List.length_pos_iff.mpr hA
    exact_mod_cast this
  have hV : 0 ≤ V := by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil A hA
    exact le_trans (abs_nonneg _) (hv e he)
  have hMlo : lo ≤ M := by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil A hA
    exact le_trans (hrange e he).1 (hrange e he).2
  have key := fsum_dev F D _ (fun e : ℝ × ℝ => Real.exp (e.1 - M) * e.2)
    (fun e : ℝ × ℝ => Real.exp (e.1 - M) * |e.2|)
    (fun _ => F.η + (1 + F.u) * F.ηe * V)
    ((1 + F.u) * Real.exp (F.u * (M - lo) + F.ue)) (by positivity)
    (fun x => by positivity) (fun _ => by positivity)
    (fun x => by
      rw [abs_mul, abs_of_pos (Real.exp_pos _)])
    A
    (by
      intro x hx y ⟨q1, q2, q3, h1, h2, h3, hy⟩
      subst hy
      exact term_o F (hrange x hx).1 (hrange x hx).2 (hv x hx) h1 h2 h3) r3 h
  rw [sum_const_aux] at key
  have hB : 0 ≤ (A.map fun e => Real.exp (e.1 - M) * |e.2|).sum :=
    sum_nonneg_aux _ (fun x => by positivity) A
  have hx0 : 0 ≤ F.u * (hi - lo) := mul_nonneg hu (by linarith)
  have hc : (1 + F.u) ^ D * ((1 + F.u) * Real.exp (F.u * (M - lo) + F.ue)) ≤
      Real.exp (g + F.u * (M - lo)) := by
    calc _ = (1 + F.u) ^ (D + 1) * Real.exp (F.u * (M - lo) + F.ue) := by ring
      _ ≤ Real.exp (((D + 1 : ℕ) : ℝ) * F.u) * Real.exp (F.u * (M - lo) + F.ue) :=
          mul_le_mul_of_nonneg_right (pow_le_exp_aux hu (D + 1)) (Real.exp_pos _).le
      _ = Real.exp (((D : ℝ) + 1) * F.u + F.ue + F.u * (M - lo)) := by
          rw [← Real.exp_add]; congr 1; push_cast; ring
      _ ≤ _ := Real.exp_le_exp.mpr (by linarith)
  have hp : (1 + F.u) ^ D ≤ Real.exp g :=
    le_trans (pow_le_exp_aux hu D) (Real.exp_le_exp.mpr (by
      have : 0 ≤ (D : ℝ) * F.u := by positivity
      nlinarith [F.hue]))
  have hp1 : (1 + F.u) ^ D * (1 + F.u) ≤ Real.exp (g + F.u * (hi - lo)) := by
    calc _ = (1 + F.u) ^ (D + 1) := by ring
      _ ≤ Real.exp (((D + 1 : ℕ) : ℝ) * F.u) := pow_le_exp_aux hu (D + 1)
      _ ≤ _ := Real.exp_le_exp.mpr (by push_cast; nlinarith [F.hue])
  have e1 := mul_le_mul_of_nonneg_right (sub_le_sub_right hc 1) hB
  have hj : (1 : ℝ) ≤ (j : ℝ) + 1 := by have : (0 : ℝ) ≤ j := Nat.cast_nonneg j; linarith
  have e2 : (1 + F.u) ^ D * ((A.length : ℝ) * F.η) ≤
      (3 * (A.length : ℝ) - 2) * F.η * Real.exp g := by
    have h0 : 0 ≤ F.η := hη
    calc (1 + F.u) ^ D * ((A.length : ℝ) * F.η) ≤ Real.exp g * ((A.length : ℝ) * F.η) :=
          mul_le_mul_of_nonneg_right hp (by positivity)
      _ ≤ Real.exp g * ((3 * (A.length : ℝ) - 2) * F.η) :=
          mul_le_mul_of_nonneg_left (by nlinarith) (Real.exp_pos _).le
      _ = _ := by ring
  have e3 : (1 + F.u) ^ D * ((1 + F.u) * ((A.length : ℝ) * F.ηe * V)) ≤
      (A.length : ℝ) * ((j : ℝ) + 1) * F.ηe * V * Real.exp (g + F.u * (hi - lo)) := by
    have h0 : 0 ≤ (A.length : ℝ) * F.ηe * V := by positivity
    calc (1 + F.u) ^ D * ((1 + F.u) * ((A.length : ℝ) * F.ηe * V))
        = ((1 + F.u) ^ D * (1 + F.u)) * ((A.length : ℝ) * F.ηe * V) := by ring
      _ ≤ Real.exp (g + F.u * (hi - lo)) * ((A.length : ℝ) * F.ηe * V) :=
          mul_le_mul_of_nonneg_right hp1 h0
      _ ≤ Real.exp (g + F.u * (hi - lo)) * ((A.length : ℝ) * F.ηe * V * ((j : ℝ) + 1)) :=
          mul_le_mul_of_nonneg_left (by nlinarith) (Real.exp_pos _).le
      _ = _ := by ring
  nlinarith

lemma shift_sum (A : List (ℝ × ℝ)) (φ : ℝ × ℝ → ℝ) (m m' : ℝ) :
    (A.map fun e => Real.exp (e.1 - m') * φ e).sum =
      (A.map fun e => Real.exp (e.1 - m) * φ e).sum * Real.exp (m - m') := by
  rw [← List.sum_map_mul_right]
  congr 1
  apply List.map_congr_left
  intro e _
  rw [show e.1 - m' = (e.1 - m) + (m - m') by ring, Real.exp_add]; ring

lemma B_le_aux (A : List (ℝ × ℝ)) (ψ : ℝ × ℝ → ℝ) (Bm m : ℝ)
    (hψ : ∀ e ∈ A, 0 ≤ ψ e ∧ ψ e ≤ Bm)
    (hm : ∀ e ∈ A, e.1 ≤ m) :
    (A.map fun e => Real.exp (e.1 - m) * ψ e).sum ≤ A.length * Bm := by
  induction A with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    have h1 := ih (fun e he => hψ e (List.mem_cons_of_mem _ he))
      (fun e he => hm e (List.mem_cons_of_mem _ he))
    have h2 : Real.exp (a.1 - m) * ψ a ≤ Bm := by
      have : Real.exp (a.1 - m) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith [hm a (by simp)])
      calc _ ≤ 1 * ψ a := mul_le_mul_of_nonneg_right this (hψ a (by simp)).1
        _ ≤ Bm := by linarith [(hψ a (by simp)).2]
    push_cast
    linarith

lemma Z_le_aux (A : List (ℝ × ℝ)) (φ ψ : ℝ × ℝ → ℝ) (hφ : ∀ e ∈ A, |φ e| ≤ ψ e) (m : ℝ) :
    |(A.map fun e => Real.exp (e.1 - m) * φ e).sum| ≤
      (A.map fun e => Real.exp (e.1 - m) * ψ e).sum := by
  induction A with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    have h1 := ih (fun e he => hφ e (List.mem_cons_of_mem _ he))
    have h2 : |Real.exp (a.1 - m) * φ a| ≤ Real.exp (a.1 - m) * ψ a := by
      rw [abs_mul, abs_of_pos (Real.exp_pos _)]
      exact mul_le_mul_of_nonneg_left (hφ a (by simp)) (Real.exp_pos _).le
    calc _ ≤ |Real.exp (a.1 - m) * φ a| + |(l.map fun e => Real.exp (e.1 - m) * φ e).sum| :=
          abs_add_le _ _
      _ ≤ _ := add_le_add h2 h1

lemma sum_nonneg_mem (A : List (ℝ × ℝ)) (ψ : ℝ × ℝ → ℝ) (m : ℝ) (hψ : ∀ e ∈ A, 0 ≤ ψ e) :
    0 ≤ (A.map fun e => Real.exp (e.1 - m) * ψ e).sum := by
  induction A with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    exact add_nonneg (mul_nonneg (Real.exp_pos _).le (hψ a (by simp)))
      (ih (fun e he => hψ e (List.mem_cons_of_mem _ he)))

lemma abs_alg (u η ηe Bm nL nR j BL BR Eue S G Gx Gγ : ℝ) (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hηe : 0 ≤ ηe) (hBm : 0 ≤ Bm) (hnL : 1 ≤ nL) (hnR : 1 ≤ nR) (hj : 0 ≤ j)
    (hBL0 : 0 ≤ BL) (hBR0 : 0 ≤ BR) (hBL : BL ≤ nL * Bm) (hBR : BR ≤ nR * Bm)
    (hG : 0 < G) (hGx : 0 ≤ Gx) (hGγ : Gγ ≤ Gx) (hGγ0 : 0 ≤ Gγ) (hS : 0 ≤ S)
    (hK : (1 + u) ^ 2 * (Eue + ηe) ≤ S) (h2 : (1 + u) ^ 2 ≤ S) (h3 : 1 + u ≤ S * G) :
    (1 + u) ^ 2 * ((Eue + ηe) * (((3 * nL - 2) * η * G + nL * (j + 1) * ηe * Bm * Gx) +
        ((3 * nR - 2) * η * G + nR * (j + 1) * ηe * Bm * Gx)) + Gγ * (BL + BR) * ηe)
      + 2 * (1 + u) * η ≤
    (3 * (nL + nR) - 2) * η * (S * G) + (nL + nR) * ((j + 1) + 1) * ηe * Bm * (S * Gx) := by
  have he : 0 ≤ ((3 * nL - 2) * η * G + nL * (j + 1) * ηe * Bm * Gx) +
        ((3 * nR - 2) * η * G + nR * (j + 1) * ηe * Bm * Gx) := by
    have a1 : 0 ≤ 3 * nL - 2 := by linarith
    have a2 : 0 ≤ 3 * nR - 2 := by linarith
    have : 0 ≤ nL := by linarith
    have : 0 ≤ nR := by linarith
    positivity
  have t1 : (1 + u) ^ 2 * ((Eue + ηe) * (((3 * nL - 2) * η * G + nL * (j + 1) * ηe * Bm * Gx) +
        ((3 * nR - 2) * η * G + nR * (j + 1) * ηe * Bm * Gx))) ≤
      S * (((3 * nL - 2) * η * G + nL * (j + 1) * ηe * Bm * Gx) +
        ((3 * nR - 2) * η * G + nR * (j + 1) * ηe * Bm * Gx)) := by
    rw [← mul_assoc]
    exact mul_le_mul_of_nonneg_right hK he
  have t2 : (1 + u) ^ 2 * (Gγ * (BL + BR) * ηe) ≤ S * (Gx * ((nL + nR) * Bm) * ηe) := by
    have a : Gγ * (BL + BR) ≤ Gx * ((nL + nR) * Bm) :=
      mul_le_mul hGγ (by linarith) (by linarith) hGx
    have a' : Gγ * (BL + BR) * ηe ≤ Gx * ((nL + nR) * Bm) * ηe := mul_le_mul_of_nonneg_right a hηe
    have hnn : 0 ≤ Gγ * (BL + BR) * ηe := by positivity
    exact mul_le_mul h2 a' hnn hS
  have t3 : 2 * (1 + u) * η ≤ 2 * η * (S * G) := by nlinarith
  nlinarith [t1, t2, t3]

set_option maxHeartbeats 1600000 in
lemma chan_merge (F : Rounding) (lo hi Bm g : ℝ) (j : ℕ) (hg : 0 ≤ g) (hBm : 0 ≤ Bm)
    (A B : List (ℝ × ℝ)) (φ ψ : ℝ × ℝ → ℝ)
    (hψA : ∀ e ∈ A, 0 ≤ ψ e ∧ ψ e ≤ Bm) (hψB : ∀ e ∈ B, 0 ≤ ψ e ∧ ψ e ≤ Bm)
    (hφA : ∀ e ∈ A, |φ e| ≤ ψ e) (hφB : ∀ e ∈ B, |φ e| ≤ ψ e)
    (hA : A ≠ []) (hB : B ≠ []) (mL mR m : ℝ)
    (hAm : ∀ e ∈ A, e.1 ≤ mL) (hBmax : ∀ e ∈ B, e.1 ≤ mR) (hmL : mL ≤ m) (hmR : mR ≤ m) (hlom : lo ≤ m) (hmhi : m ≤ hi)
    {aL aR r1 r2 r3 r4 r5 r6 r7 : ℝ}
    (haL : |aL - (A.map fun e => Real.exp (e.1 - mL) * φ e).sum| ≤
      (Real.exp (g + F.u * (mL - lo)) - 1) * (A.map fun e => Real.exp (e.1 - mL) * ψ e).sum
        + ((3 * (A.length : ℝ) - 2) * F.η * Real.exp g
          + (A.length : ℝ) * ((j : ℝ) + 1) * F.ηe * Bm * Real.exp (g + F.u * (hi - lo))))
    (haR : |aR - (B.map fun e => Real.exp (e.1 - mR) * φ e).sum| ≤
      (Real.exp (g + F.u * (mR - lo)) - 1) * (B.map fun e => Real.exp (e.1 - mR) * ψ e).sum
        + ((3 * (B.length : ℝ) - 2) * F.η * Real.exp g
          + (B.length : ℝ) * ((j : ℝ) + 1) * F.ηe * Bm * Real.exp (g + F.u * (hi - lo))))
    (h1 : F.add (mL - m) r1) (h2 : F.exp r1 r2) (h3 : F.mul (aL * r2) r3)
    (h4 : F.add (mR - m) r4) (h5 : F.exp r4 r5) (h6 : F.mul (aR * r5) r6)
    (h7 : F.add (r3 + r6) r7) :
    |r7 - ((A ++ B).map fun e => Real.exp (e.1 - m) * φ e).sum| ≤
      (Real.exp ((g + (2 * F.u + F.ue + F.ηe)) + F.u * (m - lo)) - 1) *
          ((A ++ B).map fun e => Real.exp (e.1 - m) * ψ e).sum
        + ((3 * ((A ++ B).length : ℝ) - 2) * F.η * Real.exp (g + (2 * F.u + F.ue + F.ηe))
          + ((A ++ B).length : ℝ) * (((j + 1 : ℕ) : ℝ) + 1) * F.ηe * Bm *
            Real.exp ((g + (2 * F.u + F.ue + F.ηe)) + F.u * (hi - lo))) := by
  have hu := F.hu
  have hηe := F.hηe
  have hη := F.hη
  have hue := F.hue
  have hnA : (1 : ℝ) ≤ A.length := by
    have : 0 < A.length := List.length_pos_iff.mpr hA
    exact_mod_cast this
  have hnB : (1 : ℝ) ≤ B.length := by
    have : 0 < B.length := List.length_pos_iff.mpr hB
    exact_mod_cast this
  have hj : (0 : ℝ) ≤ j := Nat.cast_nonneg j
  have hx0 : 0 ≤ F.u * (hi - lo) := mul_nonneg hu (by linarith)
  set BL := (A.map fun e => Real.exp (e.1 - mL) * ψ e).sum with hBL
  set BR := (B.map fun e => Real.exp (e.1 - mR) * ψ e).sum with hBR
  have hBL0 : 0 ≤ BL := sum_nonneg_mem A ψ mL (fun e he => (hψA e he).1)
  have hBR0 : 0 ≤ BR := sum_nonneg_mem B ψ mR (fun e he => (hψB e he).1)
  have hBLn : BL ≤ A.length * Bm := B_le_aux A ψ Bm mL hψA hAm
  have hBRn : BR ≤ B.length * Bm := B_le_aux B ψ Bm mR hψB hBmax
  have hZL := Z_le_aux A φ ψ hφA mL
  have hZR := Z_le_aux B φ ψ hφB mR
  have heL : 0 ≤ (3 * (A.length : ℝ) - 2) * F.η * Real.exp g
          + (A.length : ℝ) * ((j : ℝ) + 1) * F.ηe * Bm * Real.exp (g + F.u * (hi - lo)) := by
    have : 0 ≤ 3 * (A.length : ℝ) - 2 := by linarith
    positivity
  have heR : 0 ≤ (3 * (B.length : ℝ) - 2) * F.η * Real.exp g
          + (B.length : ℝ) * ((j : ℝ) + 1) * F.ηe * Bm * Real.exp (g + F.u * (hi - lo)) := by
    have : 0 ≤ 3 * (B.length : ℝ) - 2 := by linarith
    positivity
  have core := merge_core F (γL := g + F.u * (mL - lo)) (γR := g + F.u * (mR - lo))
    (γ0 := g + F.u * (m - lo)) hmL hmR hBL0 hBR0 hZL hZR heL heR haL haR
    (by linarith) (by linarith) h1 h2 h3 h4 h5 h6 h7
  have eZ : ((A ++ B).map fun e => Real.exp (e.1 - m) * φ e).sum =
      (A.map fun e => Real.exp (e.1 - mL) * φ e).sum * Real.exp (mL - m) +
      (B.map fun e => Real.exp (e.1 - mR) * φ e).sum * Real.exp (mR - m) := by
    rw [List.map_append, List.sum_append, shift_sum A φ mL m, shift_sum B φ mR m]
  have eB : ((A ++ B).map fun e => Real.exp (e.1 - m) * ψ e).sum =
      BL * Real.exp (mL - m) + BR * Real.exp (mR - m) := by
    rw [List.map_append, List.sum_append, shift_sum A ψ mL m, shift_sum B ψ mR m]
  rw [eZ, eB]
  have hlen : (((A ++ B).length : ℕ) : ℝ) = (A.length : ℝ) + B.length := by
    rw [List.length_append]; push_cast; ring
  rw [hlen]
  have hBn : 0 ≤ BL * Real.exp (mL - m) + BR * Real.exp (mR - m) :=
    add_nonneg (mul_nonneg hBL0 (Real.exp_pos _).le) (mul_nonneg hBR0 (Real.exp_pos _).le)
  have hrel : (Real.exp (g + F.u * (m - lo) + F.ue + 2 * F.u) - 1) *
        (BL * Real.exp (mL - m) + BR * Real.exp (mR - m)) ≤
      (Real.exp ((g + (2 * F.u + F.ue + F.ηe)) + F.u * (m - lo)) - 1) *
        (BL * Real.exp (mL - m) + BR * Real.exp (mR - m)) := by
    apply mul_le_mul_of_nonneg_right _ hBn
    have := Real.exp_le_exp.mpr (show g + F.u * (m - lo) + F.ue + 2 * F.u ≤
      (g + (2 * F.u + F.ue + F.ηe)) + F.u * (m - lo) by linarith)
    linarith
  -- absolute part
  have hs0 : 0 ≤ 2 * F.u + F.ue + F.ηe := by linarith
  have hK : (1 + F.u) ^ 2 * (Real.exp F.ue + F.ηe) ≤ Real.exp (2 * F.u + F.ue + F.ηe) := by
    have h1 : (1 + F.u) ^ 2 ≤ Real.exp (2 * F.u) := by
      have h : 1 + F.u ≤ Real.exp F.u := by linarith [Real.add_one_le_exp F.u]
      rw [show 2 * F.u = F.u + F.u by ring, Real.exp_add, sq]
      exact mul_le_mul h h (by linarith) (Real.exp_pos _).le
    have h2 : Real.exp F.ue + F.ηe ≤ Real.exp F.ue * Real.exp F.ηe := by
      have a1 : 1 + F.ηe ≤ Real.exp F.ηe := by linarith [Real.add_one_le_exp F.ηe]
      have a2 : 1 ≤ Real.exp F.ue := Real.one_le_exp hue
      have b1 := mul_le_mul_of_nonneg_right a2 hηe
      have b2 := mul_le_mul_of_nonneg_left a1 (Real.exp_pos F.ue).le
      linarith
    calc _ ≤ Real.exp (2 * F.u) * (Real.exp F.ue * Real.exp F.ηe) :=
          mul_le_mul h1 h2 (by positivity) (Real.exp_pos _).le
      _ = _ := by rw [← Real.exp_add, ← Real.exp_add]; congr 1; ring
  have h2u : (1 + F.u) ^ 2 ≤ Real.exp (2 * F.u + F.ue + F.ηe) := by
    have h : 1 + F.u ≤ Real.exp F.u := by linarith [Real.add_one_le_exp F.u]
    calc (1 + F.u) ^ 2 ≤ Real.exp F.u ^ 2 := pow_le_pow_left₀ (by linarith) h 2
      _ = Real.exp (2 * F.u) := by rw [← Real.exp_nat_mul]; norm_num
      _ ≤ _ := Real.exp_le_exp.mpr (by linarith)
  have h3 : 1 + F.u ≤ Real.exp (2 * F.u + F.ue + F.ηe) * Real.exp g := by
    rw [← Real.exp_add]
    have : 1 + F.u ≤ Real.exp F.u := by linarith [Real.add_one_le_exp F.u]
    exact le_trans this (Real.exp_le_exp.mpr (by linarith))
  have hGγ : Real.exp (g + F.u * (m - lo)) ≤ Real.exp (g + F.u * (hi - lo)) :=
    Real.exp_le_exp.mpr (by nlinarith)
  have halg := abs_alg F.u F.η F.ηe Bm (A.length : ℝ) (B.length : ℝ) (j : ℝ) BL BR (Real.exp F.ue)
    (Real.exp (2 * F.u + F.ue + F.ηe)) (Real.exp g) (Real.exp (g + F.u * (hi - lo)))
    (Real.exp (g + F.u * (m - lo))) hu hη hηe hBm hnA hnB hj hBL0 hBR0 hBLn hBRn
    (Real.exp_pos _) (Real.exp_pos _).le hGγ (Real.exp_pos _).le (Real.exp_pos _).le hK h2u h3
  have e1 : Real.exp (g + (2 * F.u + F.ue + F.ηe)) =
      Real.exp (2 * F.u + F.ue + F.ηe) * Real.exp g := by
    rw [add_comm, Real.exp_add]
  have e2 : Real.exp ((g + (2 * F.u + F.ue + F.ηe)) + F.u * (hi - lo)) =
      Real.exp (2 * F.u + F.ue + F.ηe) * Real.exp (g + F.u * (hi - lo)) := by
    rw [show (g + (2 * F.u + F.ue + F.ηe)) + F.u * (hi - lo) =
      (2 * F.u + F.ue + F.ηe) + (g + F.u * (hi - lo)) by ring, Real.exp_add]
  rw [e1, e2]
  push_cast
  linarith [core, hrel, halg]

lemma flat_ne_aux : ∀ t : Discovery.Tree (ℝ × ℝ), (∀ b ∈ t.leaves, b ≠ []) → t.flat ≠ [] := by
  intro t
  induction t with
  | leaf b =>
    intro h
    exact h b (by simp [Discovery.Tree.leaves])
  | node l r ihl ihr =>
    intro h
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => h b (by simp [Discovery.Tree.leaves, hb])
    have := ihl hl
    simp only [Discovery.Tree.flat]
    intro heq
    exact this (List.append_eq_nil_iff.mp heq).1

lemma lmax_bounds_aux (A : List (ℝ × ℝ)) (hA : A ≠ []) (lo hi : ℝ)
    (hrange : ∀ e ∈ A, lo ≤ e.1 ∧ e.1 ≤ hi) :
    lo ≤ lmax (A.map fun e => e.1) ∧ lmax (A.map fun e => e.1) ≤ hi := by
  have hne : (A.map fun e => e.1) ≠ [] := by simpa using hA
  have hm := lmax_mem hne
  obtain ⟨e, he, heq⟩ := List.mem_map.mp hm
  rw [← heq]
  exact hrange e he

lemma tree_inv (F : Rounding) (D : ℕ) (lo hi V : ℝ) :
    ∀ t : Discovery.Tree (ℝ × ℝ), (∀ e ∈ t.flat, lo ≤ e.1 ∧ e.1 ≤ hi) → (∀ e ∈ t.flat, |e.2| ≤ V) →
    (∀ b ∈ t.leaves, b ≠ []) → ∀ j : ℕ, t.depth ≤ j → ∀ st : State_a ℝ,
    Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
    st.m = lmax (t.flat.map fun e => e.1) ∧
    |st.d - (t.flat.map fun e => Real.exp (e.1 - st.m) * 1).sum| ≤
      (Real.exp (((D : ℝ) * F.u + F.ue + (j : ℝ) * (2 * F.u + F.ue + F.ηe)) + F.u * (st.m - lo)) - 1) *
        (t.flat.map fun e => Real.exp (e.1 - st.m) * 1).sum
      + ((3 * (t.flat.length : ℝ) - 2) * F.η *
          Real.exp ((D : ℝ) * F.u + F.ue + (j : ℝ) * (2 * F.u + F.ue + F.ηe))
        + (t.flat.length : ℝ) * ((j : ℝ) + 1) * F.ηe * 1 *
          Real.exp (((D : ℝ) * F.u + F.ue + (j : ℝ) * (2 * F.u + F.ue + F.ηe)) + F.u * (hi - lo))) ∧
    |st.o - (t.flat.map fun e => Real.exp (e.1 - st.m) * e.2).sum| ≤
      (Real.exp ((((D : ℝ) * F.u + F.ue + (j : ℝ) * (2 * F.u + F.ue + F.ηe)) + F.u) + F.u * (st.m - lo)) - 1) *
        (t.flat.map fun e => Real.exp (e.1 - st.m) * |e.2|).sum
      + ((3 * (t.flat.length : ℝ) - 2) * F.η *
          Real.exp (((D : ℝ) * F.u + F.ue + (j : ℝ) * (2 * F.u + F.ue + F.ηe)) + F.u)
        + (t.flat.length : ℝ) * ((j : ℝ) + 1) * F.ηe * V *
          Real.exp ((((D : ℝ) * F.u + F.ue + (j : ℝ) * (2 * F.u + F.ue + F.ηe)) + F.u) + F.u * (hi - lo))) := by
  intro t
  induction t with
  | leaf blk =>
    intro hrange hv hne j hj st hw
    cases hw with
    | leaf h =>
      obtain ⟨v_m, v_d, v_o, r1, r2, r3, hr1, hvm, hs2, hvd, hs3, hvo, hst⟩ := h
      subst hvm hr1 hvd hvo hst
      simp only [Discovery.Tree.flat] at hrange hv ⊢
      have hbne : blk ≠ [] := hne blk (by simp [Discovery.Tree.leaves])
      have hb := lmax_bounds_aux blk hbne lo hi hrange
      have hrange' : ∀ e ∈ blk, lo ≤ e.1 ∧ e.1 ≤ lmax (blk.map fun e => e.1) := by
        intro e he
        exact ⟨(hrange e he).1, le_lmax (List.mem_map.mpr ⟨e, he, rfl⟩)⟩
      have hlohi : lo ≤ hi := le_trans hb.1 hb.2
      have hs := F.hu
      have hse := F.hue
      have hsη := F.hηe
      have hj0 : (0 : ℝ) ≤ j := Nat.cast_nonneg j
      refine ⟨trivial, ?_, ?_⟩
      · exact leaf_d F D lo hi _ blk hbne hrange' hlohi
          ((D : ℝ) * F.u + F.ue + (j : ℝ) * (2 * F.u + F.ue + F.ηe))
          (by nlinarith [mul_nonneg hj0 (by linarith : 0 ≤ 2 * F.u + F.ue + F.ηe)]) j v_d hs2
      · exact leaf_o F D lo hi _ V blk hbne hrange' hv hlohi
          (((D : ℝ) * F.u + F.ue + (j : ℝ) * (2 * F.u + F.ue + F.ηe)) + F.u)
          (by nlinarith [mul_nonneg hj0 (by linarith : 0 ≤ 2 * F.u + F.ue + F.ηe)]) j v_o hs3
  | node l r ihl ihr =>
    intro hrange hv hne j hj st hw
    have hrl : ∀ e ∈ l.flat, lo ≤ e.1 ∧ e.1 ≤ hi := fun e he =>
      hrange e (by simp [Discovery.Tree.flat, he])
    have hrr : ∀ e ∈ r.flat, lo ≤ e.1 ∧ e.1 ≤ hi := fun e he =>
      hrange e (by simp [Discovery.Tree.flat, he])
    have hvl : ∀ e ∈ l.flat, |e.2| ≤ V := fun e he => hv e (by simp [Discovery.Tree.flat, he])
    have hvr : ∀ e ∈ r.flat, |e.2| ≤ V := fun e he => hv e (by simp [Discovery.Tree.flat, he])
    have hnl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => hne b (by simp [Discovery.Tree.leaves, hb])
    have hnr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => hne b (by simp [Discovery.Tree.leaves, hb])
    have hAne := flat_ne_aux l hnl
    have hBne := flat_ne_aux r hnr
    simp only [Discovery.Tree.depth] at hj
    obtain ⟨j', rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
    cases hw with
    | node hwl hwr hm =>
      rename_i a c
      obtain ⟨hLm, hLd, hLo⟩ := ihl hrl hvl hnl j' (by omega) a hwl
      obtain ⟨hRm, hRd, hRo⟩ := ihr hrr hvr hnr j' (by omega) c hwr
      obtain ⟨v_m, v_d, v_o, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14,
        hvm, h1, h2, h3, h4, h5, h6, h7, hvd, h8, h9, h10, h11, h12, h13, h14, hvo, hst⟩ := hm
      subst hvd hvo hst
      have hbl := lmax_bounds_aux l.flat hAne lo hi hrl
      have hbr := lmax_bounds_aux r.flat hBne lo hi hrr
      rw [← hLm] at hbl
      rw [← hRm] at hbr
      have hmL : a.m ≤ v_m := by rw [hvm]; exact le_max_left _ _
      have hmR : c.m ≤ v_m := by rw [hvm]; exact le_max_right _ _
      have hlom : lo ≤ v_m := le_trans hbl.1 hmL
      have hmhi : v_m ≤ hi := by rw [hvm]; exact max_le hbl.2 hbr.2
      have hAm : ∀ e ∈ l.flat, e.1 ≤ a.m := fun e he => by
        rw [hLm]; exact le_lmax (List.mem_map.mpr ⟨e, he, rfl⟩)
      have hBmax : ∀ e ∈ r.flat, e.1 ≤ c.m := fun e he => by
        rw [hRm]; exact le_lmax (List.mem_map.mpr ⟨e, he, rfl⟩)
      have hu := F.hu
      have hue := F.hue
      have hηe := F.hηe
      have hj0 : (0 : ℝ) ≤ j' := Nat.cast_nonneg j'
      have hg0 : (0 : ℝ) ≤ (D : ℝ) * F.u + F.ue + (j' : ℝ) * (2 * F.u + F.ue + F.ηe) := by
        have : (0 : ℝ) ≤ D := Nat.cast_nonneg D
        have := mul_nonneg hj0 (by linarith : 0 ≤ 2 * F.u + F.ue + F.ηe)
        nlinarith
      have hV : 0 ≤ V := by
        obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil l.flat hAne
        exact le_trans (abs_nonneg _) (hvl e he)
      have Cd := chan_merge F lo hi 1 ((D : ℝ) * F.u + F.ue + (j' : ℝ) * (2 * F.u + F.ue + F.ηe)) j'
        hg0 zero_le_one l.flat r.flat (fun _ => 1) (fun _ => 1)
        (fun e _ => ⟨zero_le_one, le_rfl⟩) (fun e _ => ⟨zero_le_one, le_rfl⟩)
        (fun e _ => by simp) (fun e _ => by simp) hAne hBne a.m c.m v_m hAm hBmax hmL hmR hlom hmhi
        hLd hRd h1 h2 h3 h4 h5 h6 h7
      have Co := chan_merge F lo hi V (((D : ℝ) * F.u + F.ue + (j' : ℝ) * (2 * F.u + F.ue + F.ηe)) + F.u) j'
        (by linarith) hV l.flat r.flat (fun e => e.2) (fun e => |e.2|)
        (fun e he => ⟨abs_nonneg _, hvl e he⟩) (fun e he => ⟨abs_nonneg _, hvr e he⟩)
        (fun e _ => le_rfl) (fun e _ => le_rfl) hAne hBne a.m c.m v_m hAm hBmax hmL hmR hlom hmhi
        hLo hRo h8 h9 h10 h11 h12 h13 h14
      have eg : ((D : ℝ) * F.u + F.ue + ((j' + 1 : ℕ) : ℝ) * (2 * F.u + F.ue + F.ηe)) =
          ((D : ℝ) * F.u + F.ue + (j' : ℝ) * (2 * F.u + F.ue + F.ηe)) + (2 * F.u + F.ue + F.ηe) := by
        push_cast; ring
      simp only [Discovery.Tree.flat]
      rw [eg]
      refine ⟨?_, ?_, ?_⟩
      · show v_m = _
        rw [hvm, hLm, hRm]
        exact (lmax_map_append (fun e : ℝ × ℝ => e.1) hAne hBne).symm
      · exact Cd
      · have eg2 : (((D : ℝ) * F.u + F.ue + (j' : ℝ) * (2 * F.u + F.ue + F.ηe)) + F.u) +
            (2 * F.u + F.ue + F.ηe) = (((D : ℝ) * F.u + F.ue + (j' : ℝ) * (2 * F.u + F.ue + F.ηe)) +
            (2 * F.u + F.ue + F.ηe)) + F.u := by ring
        rw [eg2] at Co
        exact Co

lemma exp_le_quad {y : ℝ} (h0 : 0 ≤ y) (h1 : y ≤ 3 / 16) :
    Real.exp y ≤ 1 + y + 13 / 24 * y ^ 2 := by
  have hb := Real.exp_bound (x := y) (by rw [abs_of_nonneg h0]; linarith) (n := 3) (by norm_num)
  rw [abs_of_nonneg h0] at hb
  have := (abs_le.mp hb).2
  simp [Finset.sum_range_succ, Nat.factorial] at this
  have h2 : y ^ 3 ≤ 3 / 16 * y ^ 2 := by
    have : y ^ 3 = y * y ^ 2 := by ring
    rw [this]; exact mul_le_mul_of_nonneg_right h1 (by positivity)
  norm_num at this
  nlinarith

lemma ratio_lemma (Zd Zo S d o w δ e' u η ρd ρo ed eo : ℝ) (hZd : 1 ≤ Zd) (hS : 0 ≤ S)
    (hZo : |Zo| ≤ S * Zd) (hu : 0 ≤ u) (hρd : 0 ≤ ρd) (hρo : 0 ≤ ρo) (hed : 0 ≤ ed) (heo : 0 ≤ eo)
    (hΔ : 0 < 1 - ρd - ed)
    (hd : |d - Zd| ≤ ρd * Zd + ed) (ho : |o - Zo| ≤ ρo * (S * Zd) + eo)
    (hw : w = (o / d) * (1 + δ) + e') (hδ : |δ| ≤ u) (he' : |e'| ≤ η) :
    |w - Zo / Zd| ≤ S * (ρo + ρd + ed + u * (1 + ρo)) / (1 - ρd - ed) + (1 + u) * eo / (1 - ρd - ed) + η := by
  set Δ := 1 - ρd - ed with hΔdef
  have hZd0 : 0 < Zd := by linarith
  have hd1 : Zd * Δ ≤ d := by
    have := (abs_le.mp hd).1
    nlinarith
  have hdpos : 0 < d := lt_of_lt_of_le (mul_pos hZd0 hΔ) hd1
  have step1 : |o / d - Zo / Zd| ≤ (S * (ρo + ρd + ed) + eo) / Δ := by
    have e1 : o / d - Zo / Zd = ((o - Zo) * Zd - Zo * (d - Zd)) / (d * Zd) := by
      field_simp; ring
    rw [e1, abs_div, abs_of_pos (mul_pos hdpos hZd0), div_le_div_iff₀ (mul_pos hdpos hZd0) hΔ]
    have hnum : |(o - Zo) * Zd - Zo * (d - Zd)| ≤ (ρo * (S * Zd) + eo) * Zd + S * Zd * (ρd * Zd + ed) := by
      calc _ ≤ |(o - Zo) * Zd| + |Zo * (d - Zd)| := abs_sub _ _
        _ ≤ _ := by
          rw [abs_mul, abs_mul, abs_of_pos hZd0]
          have h1 := mul_le_mul_of_nonneg_right ho hZd0.le
          have h2 := mul_le_mul hZo hd (abs_nonneg _) (by positivity)
          linarith
    have h3 : (S * (ρo + ρd + ed) + eo) * (Zd * Δ * Zd) ≤ (S * (ρo + ρd + ed) + eo) * (d * Zd) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      exact mul_le_mul_of_nonneg_right hd1 hZd0.le
    have h4 : ((ρo * (S * Zd) + eo) * Zd + S * Zd * (ρd * Zd + ed)) * Δ ≤
        (S * (ρo + ρd + ed) + eo) * (Zd * Δ * Zd) := by
      have : 0 ≤ Zd * Δ * (Zd - 1) * (S * ed + eo) := by
        have : 0 ≤ Zd - 1 := by linarith
        positivity
      nlinarith
    calc _ ≤ ((ρo * (S * Zd) + eo) * Zd + S * Zd * (ρd * Zd + ed)) * Δ :=
          mul_le_mul_of_nonneg_right hnum hΔ.le
      _ ≤ _ := le_trans h4 h3
  have step2 : |o / d| ≤ (S * (1 + ρo) + eo) / Δ := by
    rw [abs_div, abs_of_pos hdpos, div_le_div_iff₀ hdpos hΔ]
    have h1 : |o| ≤ (1 + ρo) * (S * Zd) + eo := by
      calc |o| = |Zo + (o - Zo)| := by ring_nf
        _ ≤ |Zo| + |o - Zo| := abs_add_le _ _
        _ ≤ _ := by linarith
    have h2 : (S * (1 + ρo) + eo) * (Zd * Δ) ≤ (S * (1 + ρo) + eo) * d :=
      mul_le_mul_of_nonneg_left hd1 (by positivity)
    have h3 : |o| * Δ ≤ (S * (1 + ρo) + eo) * (Zd * Δ) := by
      have : 0 ≤ (Zd - 1) * eo * Δ := by
        have : 0 ≤ Zd - 1 := by linarith
        positivity
      nlinarith [mul_le_mul_of_nonneg_right h1 hΔ.le]
    linarith
  have hfin : |w - Zo / Zd| ≤ |o / d - Zo / Zd| + u * |o / d| + η := by
    have : w - Zo / Zd = (o / d - Zo / Zd) + (o / d) * δ + e' := by rw [hw]; ring
    rw [this]
    calc _ ≤ |(o / d - Zo / Zd) + (o / d) * δ| + |e'| := abs_add_le _ _
      _ ≤ |o / d - Zo / Zd| + |(o / d) * δ| + |e'| := by
          have := abs_add_le (o / d - Zo / Zd) ((o / d) * δ); linarith
      _ ≤ _ := by
          rw [abs_mul]
          have := mul_le_mul_of_nonneg_left hδ (abs_nonneg (o / d))
          linarith
  have hu2 := mul_le_mul_of_nonneg_left step2 hu
  have e3 : (S * (ρo + ρd + ed) + eo) / Δ + u * ((S * (1 + ρo) + eo) / Δ) =
      S * (ρo + ρd + ed + u * (1 + ρo)) / Δ + (1 + u) * eo / Δ := by
    field_simp; ring
  linarith

lemma poly1 (z u Q : ℝ) (hz0 : 0 ≤ z) (hz : z + u ≤ 3/16) (hu0 : 0 ≤ u) (hu : u ≤ 1/256)
    (hQ0 : 0 ≤ Q) (hQ : Q ≤ 1/16) :
    (1 + z + 13/24*z^2) * (1 + (1+u+u^2) + u*(1+u+u^2) + (Q+2*u) + 3*(z+2*u+Q)*(1+Q+2*u))
      ≤ 2 + 6*(z+2*u+Q) := by
  nlinarith [mul_nonneg hz0 hQ0, mul_nonneg hz0 hu0, mul_nonneg hu0 hQ0,
    mul_nonneg (sub_nonneg.2 hz) hz0, mul_nonneg (sub_nonneg.2 hQ) hQ0,
    mul_nonneg (sub_nonneg.2 hu) hu0, mul_nonneg (mul_nonneg hz0 hz0) hQ0,
    mul_nonneg (mul_nonneg hz0 hz0) hu0, mul_nonneg (mul_nonneg hz0 hz0) hz0,
    mul_nonneg (mul_nonneg hz0 hQ0) hQ0]

lemma poly_A1 (W z u Q q : ℝ) (hz0 : 0 ≤ z) (hz : z + u ≤ 3/16) (hu0 : 0 ≤ u) (hu : u ≤ 1/256)
    (hQ0 : 0 ≤ Q) (hQ : Q ≤ 1/16) (hq0 : 0 ≤ q) (hq : q ≤ Q + 2*u) (hW0 : 0 ≤ W)
    (hW : W ≤ 1 + z + 13/24*z^2) :
    (W - 1) + (W*(1+u+u^2) - 1) + W*q + u*(W*(1+u+u^2)) ≤ 3*(z+2*u+Q)*(2 - W*(1+q)) := by
  have key := poly1 z u Q hz0 hz hu0 hu hQ0 hQ
  have hP : 0 ≤ z + 2*u + Q := by linarith
  have hc2 : 1 + (1+u+u^2) + u*(1+u+u^2) + q + 3*(z+2*u+Q)*(1+q) ≤
      1 + (1+u+u^2) + u*(1+u+u^2) + (Q+2*u) + 3*(z+2*u+Q)*(1+Q+2*u) := by
    nlinarith [mul_nonneg hP (sub_nonneg.2 hq)]
  have hc2pos : 0 ≤ 1 + (1+u+u^2) + u*(1+u+u^2) + (Q+2*u) + 3*(z+2*u+Q)*(1+Q+2*u) := by
    positivity
  have h1 := mul_le_mul_of_nonneg_left hc2 hW0
  have h2 := mul_le_mul_of_nonneg_right hW hc2pos
  nlinarith

lemma poly_A2a (u n η W q : ℝ) (hu0 : 0 ≤ u) (hu : u ≤ 1/256) (hn : 1 ≤ n) (hη : 0 ≤ η)
    (hW : W ≤ 1207/1000) (hW1 : 0 ≤ W) (hq0 : 0 ≤ q) (hq : q ≤ 71/1000) :
    (1 + u) * ((3*n - 2) * η * (1134/1000)) ≤ (5*n*η - η) * (2 - W*(1+q)) := by
  have hD : 707/1000 ≤ 2 - W*(1+q) := by
    have : W*(1+q) ≤ 1207/1000 * (1071/1000) :=
      mul_le_mul hW (by linarith) (by linarith) (by norm_num)
    linarith
  have h1 : 0 ≤ (5*n*η - η) := by nlinarith
  have h2 := mul_le_mul_of_nonneg_left hD h1
  have h3 : 0 ≤ (3*n-2)*η := by
    have : 0 ≤ 3*n - 2 := by linarith
    positivity
  nlinarith [mul_nonneg hu0 h3]

lemma poly_A2b (u W a₁ q : ℝ) (hu0 : 0 ≤ u) (hu : u ≤ 1/256)
    (hW : W ≤ 1207/1000) (hW1 : 0 ≤ W) (ha : a₁ ≤ 1004/1000) (ha0 : 0 ≤ a₁) (hq0 : 0 ≤ q) (hq : q ≤ 71/1000) :
    (1 + u) * (W * a₁) ≤ 3 * (2 - W*(1+q)) := by
  have : (1 + u) * (W * a₁) ≤ (1 + 1/256) * (1207/1000 * (1004/1000)) := by
    apply mul_le_mul (by linarith) _ (by positivity) (by norm_num)
    exact mul_le_mul hW ha ha0 (by norm_num)
  nlinarith

lemma exp_aux1 {y : ℝ} (h0 : 0 ≤ y) (h1 : y ≤ 1/8) : Real.exp y ≤ 1134/1000 := by
  have := exp_le_quad h0 (by linarith)
  nlinarith

lemma exp_aux2 {u : ℝ} (h0 : 0 ≤ u) (h1 : u ≤ 1/256) : Real.exp u ≤ 1 + u + u^2 := by
  have := exp_le_quad h0 (by linarith)
  nlinarith

lemma W_aux {W z : ℝ} (h0 : 0 ≤ z) (h1 : z ≤ 3/16) (hW : W ≤ 1 + z + 13/24*z^2) :
    W ≤ 1207/1000 := by nlinarith

lemma u_aux {u : ℝ} (h0 : 0 ≤ u) (h1 : u ≤ 1/256) : 1 + u + u^2 ≤ 1004/1000 := by nlinarith

lemma final_core2 (u η ηe T n V S Zd Zo d o w δ e' W z E : ℝ)
    (hu : 0 ≤ u) (hu' : u ≤ 1/256) (hη : 0 ≤ η) (hηe : 0 ≤ ηe)
    (hnη : n * (η + ηe) ≤ u) (hn1 : 1 ≤ n) (hT0 : 0 ≤ T) (hTn : T ≤ n)
    (hT1u : (T + 1) * u ≤ 1/16)
    (hV : 0 ≤ V) (hS : 0 ≤ S) (hZd : 1 ≤ Zd) (hZo : |Zo| ≤ S * Zd)
    (hz0 : 0 ≤ z) (hz3 : z + u ≤ 3/16) (hEz : E = z + 2*u - T*ηe)
    (hW1 : 1 ≤ W) (hWq : W ≤ 1 + z + 13/24*z^2)
    (hd : |d - Zd| ≤ (W - 1) * Zd + W * (3 * n * η + n * (T + 1) * ηe))
    (ho : |o - Zo| ≤ (W * (1 + u + u^2) - 1) * (S * Zd) +
      ((3*n-2)*η*(1134/1000) + n*(T+1)*ηe*V*(W * (1 + u + u^2))))
    (hw : w = (o/d)*(1+δ)+e') (hδ : |δ| ≤ u) (he' : |e'| ≤ η) :
    |w - Zo/Zd| ≤ 3*S*E + 5*n*η + 3*(T+1)*n*ηe*V + 3*S*n*(η+(T+1)*ηe) := by
  have hn0 : 0 ≤ n := by linarith
  have hnη0 : 0 ≤ n * η := mul_nonneg hn0 hη
  have hnηe0 : 0 ≤ n * ηe := mul_nonneg hn0 hηe
  have hTη : T * ηe ≤ n * ηe := mul_le_mul_of_nonneg_right hTn hηe
  have hQ0 : 0 ≤ n * η + T * (n * ηe) := by positivity
  have hQ1 : n * η + T * (n * ηe) ≤ 1/16 := by
    have h1 := mul_le_mul_of_nonneg_left hnη (by linarith : 0 ≤ T + 1)
    have : 0 ≤ T * (n * η) := by positivity
    linarith
  have hq0 : 0 ≤ 3 * n * η + n * (T + 1) * ηe := by positivity
  have hqQ : 3 * n * η + n * (T + 1) * ηe ≤ (n * η + T * (n * ηe)) + 2 * u := by linarith
  have hq71 : 3 * n * η + n * (T + 1) * ηe ≤ 71/1000 := by linarith
  have hWnum : W ≤ 1207/1000 := W_aux hz0 (by linarith) hWq
  have ha₁ : 1 + u + u^2 ≤ 1004/1000 := u_aux hu hu'
  have hWpos : 0 ≤ W := by linarith
  have ha₁1 : 1 ≤ 1 + u + u^2 := by nlinarith
  generalize hqdef : 3 * n * η + n * (T + 1) * ηe = q at hd hq0 hqQ hq71
  generalize hQdef : n * η + T * (n * ηe) = Q at hQ0 hQ1 hqQ
  have hΔ' : 707/1000 ≤ 2 - W * (1 + q) := by
    have : W*(1+q) ≤ 1207/1000 * (1071/1000) :=
      mul_le_mul hWnum (by linarith) (by linarith) (by norm_num)
    linarith
  have hΔ : 0 < 1 - (W - 1) - W * q := by nlinarith
  have hρo : 0 ≤ W * (1 + u + u^2) - 1 := by nlinarith
  have hed : 0 ≤ W * q := by positivity
  have heo : 0 ≤ (3*n-2)*η*(1134/1000) + n*(T+1)*ηe*V*(W * (1 + u + u^2)) := by
    have : 0 ≤ 3*n - 2 := by linarith
    positivity
  have R := ratio_lemma Zd Zo S d o w δ e' u η (W - 1) (W * (1 + u + u^2) - 1) (W * q)
    ((3*n-2)*η*(1134/1000) + n*(T+1)*ηe*V*(W * (1 + u + u^2))) hZd hS hZo hu (by linarith) hρo hed heo
    hΔ hd ho hw hδ he'
  have A1 := poly_A1 W z u Q q hz0 hz3 hu hu' hQ0 hQ1 hq0 hqQ hWpos hWq
  have hP : z + 2 * u + Q ≤ E + (n * η + n * (T + 1) * ηe) := by
    rw [hEz]; linarith
  have hΔe : 1 - (W - 1) - W * q = 2 - W * (1 + q) := by ring
  rw [hΔe] at R
  have hΔ2 : 0 ≤ 2 - W * (1 + q) := by linarith
  have s1 : S * ((W * (1 + u + u^2) - 1) + (W - 1) + W * q + u * (1 + (W * (1 + u + u^2) - 1))) /
      (2 - W * (1 + q)) ≤ 3 * S * (E + (n * η + n * (T + 1) * ηe)) := by
    rw [div_le_iff₀ (by linarith)]
    have h1 : (W * (1 + u + u^2) - 1) + (W - 1) + W * q + u * (1 + (W * (1 + u + u^2) - 1)) ≤
        3 * (E + (n * η + n * (T + 1) * ηe)) * (2 - W * (1 + q)) := by
      have := mul_le_mul_of_nonneg_right hP hΔ2
      linarith [A1]
    have := mul_le_mul_of_nonneg_left h1 hS
    linarith
  have s2 : (1 + u) * ((3*n-2)*η*(1134/1000) + n*(T+1)*ηe*V*(W * (1 + u + u^2))) /
      (2 - W * (1 + q)) ≤ 5 * n * η - η + 3 * (n * (T + 1) * ηe * V) := by
    rw [div_le_iff₀ (by linarith)]
    have a := poly_A2a u n η W q hu hu' hn1 hη hWnum hWpos hq0 hq71
    have b := poly_A2b u W (1 + u + u^2) q hu hu' hWnum hWpos ha₁ (by linarith) hq0 hq71
    have hc : 0 ≤ n * (T + 1) * ηe * V := by positivity
    have := mul_le_mul_of_nonneg_left b hc
    linarith
  linarith [R, s1, s2]

lemma weak_d (n η ηe T Zd E1 Eg W : ℝ) (hn : 1 ≤ n) (hη : 0 ≤ η) (hηe : 0 ≤ ηe) (hT0 : 0 ≤ T)
    (hZd : 0 ≤ Zd) (hW : 0 ≤ W) (h1 : E1 ≤ W) (h2 : Eg ≤ W) :
    (E1 - 1) * Zd + ((3*n-2)*η*Eg + n*(T+1)*ηe*1*W) ≤
      (W - 1) * Zd + W * (3 * n * η + n * (T + 1) * ηe) := by
  have a1 : (E1 - 1) * Zd ≤ (W - 1) * Zd := mul_le_mul_of_nonneg_right (sub_le_sub_right h1 1) hZd
  have h3 : 0 ≤ 3 * n - 2 := by linarith
  have a2 : (3*n-2)*η*Eg ≤ (3*n-2)*η*W := mul_le_mul_of_nonneg_left h2 (mul_nonneg h3 hη)
  have a3 : 0 ≤ η * W := mul_nonneg hη hW
  nlinarith

lemma weak_o (n η ηe T V S Zd E3 E4 E5 W eu a₁ : ℝ) (hn : 1 ≤ n) (hη : 0 ≤ η) (hηe : 0 ≤ ηe)
    (hT0 : 0 ≤ T) (hV : 0 ≤ V) (hS : 0 ≤ S) (hZd : 0 ≤ Zd) (hW : 0 ≤ W)
    (h3 : E3 ≤ W * eu) (h4 : E4 ≤ 1134/1000) (h5 : E5 = W * eu) (hu : eu ≤ a₁) :
    (E3 - 1) * (S * Zd) + ((3*n-2)*η*E4 + n*(T+1)*ηe*V*E5) ≤
      (W * a₁ - 1) * (S * Zd) + ((3*n-2)*η*(1134/1000) + n*(T+1)*ηe*V*(W * a₁)) := by
  have hSZ : 0 ≤ S * Zd := mul_nonneg hS hZd
  have a1 : (E3 - 1) * (S * Zd) ≤ (W * a₁ - 1) * (S * Zd) := by
    apply mul_le_mul_of_nonneg_right _ hSZ
    have := mul_le_mul_of_nonneg_left hu hW
    linarith
  have h3' : 0 ≤ 3 * n - 2 := by linarith
  have a2 : (3*n-2)*η*E4 ≤ (3*n-2)*η*(1134/1000) := mul_le_mul_of_nonneg_left h4 (mul_nonneg h3' hη)
  have a3 : n*(T+1)*ηe*V*E5 ≤ n*(T+1)*ηe*V*(W * a₁) := by
    apply mul_le_mul_of_nonneg_left _ (by
      have : 0 ≤ n := by linarith
      positivity)
    rw [h5]; exact mul_le_mul_of_nonneg_left hu hW
  linarith

set_option maxHeartbeats 2000000 in
lemma final_core (u ue η ηe ul X T D n V S Zd Zo d o w δ e' M lo : ℝ)
    (hu : 0 ≤ u) (hu' : u ≤ 1/256) (hue : 0 ≤ ue) (hul : 0 ≤ ul) (hη : 0 ≤ η) (hηe : 0 ≤ ηe)
    (hnη : n * (η + ηe) ≤ u) (hn1 : 1 ≤ n) (hT0 : 0 ≤ T) (hTn : T ≤ n) (hD0 : 0 ≤ D)
    (hbud : (T + D + 1) * (u + ue + ul) ≤ 1/16)
    (hX0 : 0 ≤ X) (hX : X ≤ 1/16) (hMlo : 0 ≤ u * (M - lo)) (hMX : u * (M - lo) ≤ X)
    (hV : 0 ≤ V) (hS : 0 ≤ S) (hZd : 1 ≤ Zd) (hZo : |Zo| ≤ S * Zd)
    (hd : |d - Zd| ≤ (Real.exp ((D*u + ue + T*(2*u+ue+ηe)) + u*(M-lo)) - 1) * Zd +
      ((3*n-2)*η*Real.exp (D*u+ue+T*(2*u+ue+ηe)) +
        n*(T+1)*ηe*1*Real.exp ((D*u+ue+T*(2*u+ue+ηe)) + X)))
    (ho : |o - Zo| ≤ (Real.exp (((D*u+ue+T*(2*u+ue+ηe)) + u) + u*(M-lo)) - 1) * (S*Zd) +
      ((3*n-2)*η*Real.exp ((D*u+ue+T*(2*u+ue+ηe)) + u) +
        n*(T+1)*ηe*V*Real.exp (((D*u+ue+T*(2*u+ue+ηe)) + u) + X)))
    (hw : w = (o/d)*(1+δ)+e') (hδ : |δ| ≤ u) (he' : |e'| ≤ η) :
    |w - Zo/Zd| ≤ 3*S*(X + (T+1)*ue + (2*T+D+2)*u) + 5*n*η + 3*(T+1)*n*ηe*V
      + 3*S*n*(η+(T+1)*ηe) := by
  generalize hg : D*u + ue + T*(2*u+ue+ηe) = g at hd ho
  have hn0 : 0 ≤ n := by linarith
  have hTη : T * ηe ≤ u := by
    have h1 := mul_le_mul_of_nonneg_right hTn hηe
    nlinarith [mul_nonneg hn0 hη]
  have hg0 : 0 ≤ g := by
    rw [← hg]
    have := mul_nonneg hT0 (by linarith : 0 ≤ 2*u+ue+ηe)
    nlinarith [mul_nonneg hD0 hu]
  have hbud' : (T + D + 1) * (u + ue) ≤ 1/16 := by
    nlinarith [mul_nonneg (by linarith : 0 ≤ T + D + 1) hul]
  have hT1u : (T + 1) * u ≤ 1/16 := by
    nlinarith [mul_nonneg hD0 hu, mul_nonneg (by linarith : 0 ≤ T + D + 1) hue]
  have hE : X + (T+1)*ue + (2*T+D+2)*u ≤ 3/16 := by
    nlinarith [mul_nonneg hD0 hu, mul_nonneg hD0 hue, mul_nonneg hT0 hu, mul_nonneg hT0 hue]
  have hEz : X + (T+1)*ue + (2*T+D+2)*u = (g + X) + 2*u - T*ηe := by
    rw [← hg]; ring
  have hgR : g + u ≤ 1/8 := by
    have : g + u ≤ (T+1)*ue + (2*T+D+2)*u := by rw [← hg]; linarith
    linarith [mul_nonneg hD0 hu, mul_nonneg hD0 hue, mul_nonneg hT0 hu, mul_nonneg hT0 hue]
  have hz0 : 0 ≤ g + X := by linarith
  have hz3 : g + X + u ≤ 3/16 := by linarith
  have hW1 : 1 ≤ Real.exp (g + X) := Real.one_le_exp hz0
  have hWq := exp_le_quad hz0 (by linarith : g + X ≤ 3/16)
  have hexp1 : Real.exp (g + u*(M-lo)) ≤ Real.exp (g+X) := Real.exp_le_exp.mpr (by linarith)
  have hexp2 : Real.exp g ≤ Real.exp (g+X) := Real.exp_le_exp.mpr (by linarith)
  have hexp3 : Real.exp ((g+u)+u*(M-lo)) ≤ Real.exp (g+X) * Real.exp u := by
    rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith)
  have hexp4 : Real.exp ((g+u)+X) = Real.exp (g+X) * Real.exp u := by
    rw [← Real.exp_add]; congr 1; ring
  have hexp5 : Real.exp (g+u) ≤ 1134/1000 := exp_aux1 (by linarith) hgR
  have hexpu : Real.exp u ≤ 1 + u + u^2 := exp_aux2 hu hu'
  rw [hexp4] at ho
  have hW0 : 0 ≤ Real.exp (g + X) := (Real.exp_pos _).le
  have hd' := le_trans hd (weak_d n η ηe T Zd _ _ _ hn1 hη hηe hT0 (by linarith) hW0 hexp1 hexp2)
  have ho' := le_trans ho (weak_o n η ηe T V S Zd _ _ _ (Real.exp (g+X)) (Real.exp u) (1 + u + u^2)
    hn1 hη hηe hT0 hV hS (by linarith) hW0 hexp3 hexp5 rfl hexpu)
  clear hd ho
  exact final_core2 u η ηe T n V S Zd Zo d o w δ e' (Real.exp (g+X)) (g + X)
    (X + (T+1)*ue + (2*T+D+2)*u) hu hu' hη hηe hnη hn1 hT0 hTn hT1u hV hS hZd hZo hz0 hz3 hEz
    hW1 hWq hd' ho' hw hδ he'

lemma depth_le_length_aux : ∀ t : Discovery.Tree (ℝ × ℝ), (∀ b ∈ t.leaves, b ≠ []) →
    t.depth ≤ t.flat.length := by
  intro t
  induction t with
  | leaf b => intro _; simp [Discovery.Tree.depth]
  | node l r ihl ihr =>
    intro h
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => h b (by simp [Discovery.Tree.leaves, hb])
    have hr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => h b (by simp [Discovery.Tree.leaves, hb])
    have h1 := ihl hl
    have h2 := ihr hr
    have h3 : 0 < l.flat.length := List.length_pos_iff.mpr (flat_ne_aux l hl)
    have h4 : 0 < r.flat.length := List.length_pos_iff.mpr (flat_ne_aux r hr)
    simp only [Discovery.Tree.depth, Discovery.Tree.flat, List.length_append]
    omega

lemma one_le_sum_aux (A : List (ℝ × ℝ)) (M : ℝ) (hle : ∀ e ∈ A, e.1 ≤ M)
    (hex : ∃ e ∈ A, e.1 = M) : 1 ≤ (A.map fun e => Real.exp (e.1 - M) * 1).sum := by
  induction A with
  | nil => obtain ⟨e, he, _⟩ := hex; simp at he
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    have hnn : 0 ≤ (l.map fun e => Real.exp (e.1 - M) * 1).sum :=
      sum_nonneg_aux (fun e : ℝ × ℝ => Real.exp (e.1 - M) * 1) (fun x => by positivity) l
    by_cases ha : a.1 = M
    · have h1 : Real.exp (a.1 - M) * 1 = 1 := by rw [ha]; simp
      linarith
    · have : ∃ e ∈ l, e.1 = M := by
        obtain ⟨e, he, hem⟩ := hex
        rcases List.mem_cons.mp he with h | h
        · exact absurd (h ▸ hem) ha
        · exact ⟨e, h, hem⟩
      have := ih (fun e he => hle e (List.mem_cons_of_mem _ he)) this
      have : 0 ≤ Real.exp (a.1 - M) * 1 := by positivity
      linarith

lemma attn_aux (X : List (ℝ × ℝ)) (M : ℝ) :
    attn X = (X.map fun e => Real.exp (e.1 - M) * e.2).sum /
      (X.map fun e => Real.exp (e.1 - M) * 1).sum ∧
    attnScale X = (X.map fun e => Real.exp (e.1 - M) * |e.2|).sum /
      (X.map fun e => Real.exp (e.1 - M) * 1).sum := by
  have h1 := shift_sum X (fun e => e.2) M 0
  have h2 := shift_sum X (fun _ => 1) M 0
  have h3 := shift_sum X (fun e => |e.2|) M 0
  simp only [sub_zero] at h1 h2 h3
  unfold attn attnScale
  have hM : Real.exp M ≠ 0 := (Real.exp_pos M).ne'
  constructor
  · rw [h1, show (X.map fun e => Real.exp e.1) = (X.map fun e => Real.exp e.1 * 1) by simp, h2]
    rw [mul_div_mul_right _ _ hM]
  · rw [h3, show (X.map fun e => Real.exp e.1) = (X.map fun e => Real.exp e.1 * 1) by simp, h2]
    rw [mul_div_mul_right _ _ hM]

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List (ℝ × ℝ)), X ≠ [] →
    (∀ e ∈ X, lo ≤ e.1 ∧ e.1 ≤ hi) → ∀ (Vmax : ℝ), (∀ e ∈ X, |e.2| ≤ Vmax) →
    small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u → (hi - lo) * F.u ≤ 1 / 16 →
    ∀ (t : Discovery.Tree (ℝ × ℝ)), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : (State_a ℝ)), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
    ∀ (w : ℝ), fout_o F st w →
      |w - attn X| ≤ (((3 : ℝ) * (attnScale X) * (((hi - lo) * F.u) + (((t.depth : ℝ) + (1 : ℝ)) * F.ue) + ((((2 : ℝ) * (t.depth : ℝ)) + (D : ℝ) + (2 : ℝ)) * F.u))) + ((5 : ℝ) * (X.length : ℝ) * F.η) + ((3 : ℝ) * ((t.depth : ℝ) + (1 : ℝ)) * (X.length : ℝ) * F.ηe * Vmax) + ((3 : ℝ) * (attnScale X) * (X.length : ℝ) * (F.η + (((t.depth : ℝ) + (1 : ℝ)) * F.ηe)))) := by
  intro F D lo hi X hX hrange Vmax hV hsmall hnη hhi t htflat hleaves hbud st hev w hw
  subst htflat
  obtain ⟨hu', hue', hul', hη', hηe'⟩ := hsmall
  obtain ⟨hm, hd, ho⟩ := tree_inv F D lo hi Vmax t hrange hV hleaves t.depth le_rfl st hev
  obtain ⟨r1, hmul, hwr⟩ := hw
  obtain ⟨δ, e', hδ, he', hr1, -⟩ := hmul
  have hMb := lmax_bounds_aux t.flat hX lo hi hrange
  rw [← hm] at hMb
  have hVn : 0 ≤ Vmax := by
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil t.flat hX
    exact le_trans (abs_nonneg _) (hV e he)
  have hle : ∀ e ∈ t.flat, e.1 ≤ st.m := by
    intro e he
    rw [hm]; exact le_lmax (List.mem_map.mpr ⟨e, he, rfl⟩)
  have hex : ∃ e ∈ t.flat, e.1 = st.m := by
    have hne : (t.flat.map fun e => e.1) ≠ [] := by simpa using hX
    obtain ⟨e, he, heq⟩ := List.mem_map.mp (lmax_mem hne)
    exact ⟨e, he, by rw [hm]; exact heq⟩
  have hZd := one_le_sum_aux t.flat st.m hle hex
  obtain ⟨hA1, hA2⟩ := attn_aux t.flat st.m
  have hBo0 : 0 ≤ (t.flat.map fun e => Real.exp (e.1 - st.m) * |e.2|).sum :=
    sum_nonneg_mem t.flat (fun e => |e.2|) st.m (fun e _ => abs_nonneg _)
  have hZd0 : 0 < (t.flat.map fun e => Real.exp (e.1 - st.m) * 1).sum :=
    lt_of_lt_of_le one_pos hZd
  have hS0 : 0 ≤ attnScale t.flat := by rw [hA2]; exact div_nonneg hBo0 hZd0.le
  have hBo : (t.flat.map fun e => Real.exp (e.1 - st.m) * |e.2|).sum =
      attnScale t.flat * (t.flat.map fun e => Real.exp (e.1 - st.m) * 1).sum := by
    rw [hA2]; exact (div_mul_cancel₀ _ hZd0.ne').symm
  have hZo : |(t.flat.map fun e => Real.exp (e.1 - st.m) * e.2).sum| ≤
      attnScale t.flat * (t.flat.map fun e => Real.exp (e.1 - st.m) * 1).sum := by
    rw [← hBo]
    exact Z_le_aux t.flat (fun e => e.2) (fun e => |e.2|) (fun e _ => le_rfl) st.m
  rw [hBo] at ho
  have hnn : (1 : ℝ) ≤ (t.flat.length : ℝ) := by
    have : 0 < t.flat.length := List.length_pos_iff.mpr hX
    exact_mod_cast this
  have hTn : (t.depth : ℝ) ≤ (t.flat.length : ℝ) := by
    exact_mod_cast depth_le_length_aux t hleaves
  have hD0 : (0 : ℝ) ≤ D := Nat.cast_nonneg D
  have hT0 : (0 : ℝ) ≤ t.depth := Nat.cast_nonneg _
  have hX0 : 0 ≤ F.u * (hi - lo) := mul_nonneg F.hu (sub_nonneg.mpr (le_trans hMb.1 hMb.2))
  have hXb : F.u * (hi - lo) ≤ 1 / 16 := by rw [mul_comm]; exact hhi
  have hMlo : 0 ≤ F.u * (st.m - lo) := mul_nonneg F.hu (sub_nonneg.mpr hMb.1)
  have hMX : F.u * (st.m - lo) ≤ F.u * (hi - lo) :=
    mul_le_mul_of_nonneg_left (sub_le_sub_right hMb.2 lo) F.hu
  have hwe : w = (st.o / st.d) * (1 + δ) + e' := by rw [hwr, hr1]
  rw [hA1]
  generalize (t.flat.map fun e => Real.exp (e.1 - st.m) * 1).sum = Zd at hd ho hZd hZo ⊢
  generalize (t.flat.map fun e => Real.exp (e.1 - st.m) * e.2).sum = Zo at ho hZo ⊢
  generalize attnScale t.flat = S at ho hZo hS0 ⊢
  generalize (t.flat.length : ℝ) = n at hd ho hnn hTn hnη ⊢
  have key := final_core F.u F.ue F.η F.ηe F.ul (F.u * (hi - lo)) (t.depth : ℝ) (D : ℝ)
    n Vmax S Zd Zo st.d st.o w δ e' st.m lo
    F.hu hu' F.hue F.hul F.hη F.hηe hnη hnn hT0 hTn hD0 hbud hX0 hXb hMlo hMX hVn hS0 hZd hZo hd ho
    hwe hδ he'
  refine le_trans key (le_of_eq ?_)
  ring

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.AttentionOnline
