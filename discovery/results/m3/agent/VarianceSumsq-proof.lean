import FPStatements

namespace Discovery.FP.VarianceSumsq

open Discovery Discovery.Library.VarianceSumsq

theorem add_step (F : Rounding) (k : ℕ) (a b y G1 G2 C1 C2 H1 H2 : ℝ)
    (hH1 : 0 ≤ H1) (hH2 : 0 ≤ H2)
    (h1 : |a - G1| ≤ C1 + ((1+F.u)^k - 1) * H1) (h1' : |a| ≤ (1+F.u)^k * H1)
    (h2 : |b - G2| ≤ C2 + ((1+F.u)^k - 1) * H2) (h2' : |b| ≤ (1+F.u)^k * H2)
    (hy : F.add (a+b) y) :
    |y - (G1+G2)| ≤ (C1+C2) + ((1+F.u)^(k+1) - 1) * (H1+H2) ∧
      |y| ≤ (1+F.u)^(k+1) * (H1+H2) := by
  obtain ⟨δ, hδ, rfl⟩ := hy
  have hu := F.hu
  have hP0 : 0 ≤ (1+F.u)^k := pow_nonneg (by linarith) k
  have hab : |a+b| ≤ (1+F.u)^k*(H1+H2) := (abs_add_le a b).trans (by linarith)
  have hd : |(a+b)*δ| ≤ (1+F.u)^k*(H1+H2)*F.u := by
    rw [abs_mul]; exact mul_le_mul hab hδ (abs_nonneg _) (by positivity)
  have e : (a+b)*(1+δ) - (G1+G2) = (a-G1)+(b-G2) + (a+b)*δ := by ring
  rw [pow_succ]
  constructor
  · rw [e]
    rw [abs_le] at h1 h2 hd ⊢
    constructor <;> nlinarith [hd.1, hd.2, h1.1, h1.2, h2.1, h2.2]
  · rw [abs_mul]
    have : |1+δ| ≤ 1 + F.u := by
      rw [abs_le] at hδ ⊢; constructor <;> linarith [hδ.1, hδ.2]
    calc |a+b| * |1+δ| ≤ ((1+F.u)^k*(H1+H2)) * (1+F.u) :=
          mul_le_mul hab this (abs_nonneg _) (by positivity)
      _ = _ := by ring

theorem tree_fsum_err (F : Rounding) {E : Type _} (term : E → ℝ → Prop) (g c h : E → ℝ)
    (hh : ∀ x, 0 ≤ h x)
    (hterm : ∀ x y, term x y → |y - g x| ≤ c x ∧ |y| ≤ h x) :
    ∀ (t : Tree E) (w : ℝ),
      Tree.FEval (fun b y => ∃ x, b = [x] ∧ term x y) (fun a c y => F.add (a + c) y) t w →
      ∀ k, t.depth ≤ k →
      |w - (t.flat.map g).sum| ≤ (t.flat.map c).sum + ((1+F.u)^k - 1) * (t.flat.map h).sum ∧
      |w| ≤ (1+F.u)^k * (t.flat.map h).sum := by
  intro t w hev
  induction hev with
  | leaf hb =>
    rename_i b w
    intro k hk
    obtain ⟨x, rfl, hx⟩ := hb
    obtain ⟨h1, h2⟩ := hterm x w hx
    have hu := F.hu
    have hP : 1 ≤ (1+F.u)^k := one_le_pow₀ (by linarith)
    have := hh x
    simp only [Tree.flat, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    constructor
    · nlinarith
    · nlinarith
  | node hl hr hm ihl ihr =>
    rename_i l r a c' w
    intro k hk
    simp only [Tree.depth] at hk
    obtain ⟨j, rfl⟩ : ∃ j, k = j+1 := ⟨k-1, by omega⟩
    have a1 := ihl j (by omega)
    have a2 := ihr j (by omega)
    have hH : ∀ t : Tree E, 0 ≤ (t.flat.map h).sum := fun t =>
      List.sum_nonneg (by intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz; exact hh x)
    have := add_step F j a c' w _ _ _ _ _ _ (hH l) (hH r) a1.1 a1.2 a2.1 a2.2 hm
    simpa [Tree.flat, List.map_append, List.sum_append] using this

theorem fsum_err (F : Rounding) {E : Type _} (D : ℕ) (term : E → ℝ → Prop) (g c h : E → ℝ)
    (hh : ∀ x, 0 ≤ h x)
    (hterm : ∀ x y, term x y → |y - g x| ≤ c x ∧ |y| ≤ h x)
    (A : List E) (w : ℝ) (hw : fsum F D term A w) :
    |w - (A.map g).sum| ≤ (A.map c).sum + ((1+F.u)^D - 1) * (A.map h).sum ∧
      |w| ≤ (1+F.u)^D * (A.map h).sum := by
  obtain ⟨t, hf, _, hd, hev⟩ := hw
  have := tree_fsum_err F term g c h hh hterm t w hev D hd
  rwa [hf] at this

theorem tree_outer (F : Rounding) (D : ℕ) :
    ∀ (t : Tree ℝ) (st : State_a), Tree.FEval (fblock_a F D) (fmerge_a F) t st →
    ∀ k, D + t.depth ≤ k →
    st.n = (t.flat.length : ℝ) ∧
    (|st.t - t.flat.sum| ≤ ((1+F.u)^k - 1) * (t.flat.map fun x => |x|).sum ∧
      |st.t| ≤ (1+F.u)^k * (t.flat.map fun x => |x|).sum) ∧
    (|st.u - (t.flat.map fun x => x*x).sum| ≤ (t.flat.map fun x => F.u*(x*x)+F.η).sum
        + ((1+F.u)^k - 1) * (t.flat.map fun x => (1+F.u)*(x*x)+F.η).sum ∧
      |st.u| ≤ (1+F.u)^k * (t.flat.map fun x => (1+F.u)*(x*x)+F.η).sum) := by
  intro t st hev
  induction hev with
  | leaf hb =>
    rename_i b st
    intro k hk
    obtain ⟨v_n, v_t, v_u, r1, r2, r3, h1, h2, hs, h3, hs2, h4, h5⟩ := hb
    subst h5
    have hu := F.hu
    have hP : 1 ≤ (1+F.u)^k := one_le_pow₀ (by linarith)
    have hPD : (1+F.u)^D ≤ (1+F.u)^k :=
      pow_le_pow_right₀ (by linarith) (by simp [Tree.depth] at hk; omega)
    have hH0 : ∀ L : List ℝ, 0 ≤ (L.map fun x => |x|).sum := fun L =>
      List.sum_nonneg (by intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz; exact abs_nonneg x)
    have hH1 : ∀ L : List ℝ, 0 ≤ (L.map fun x => (1+F.u)*(x*x)+F.η).sum := fun L =>
      List.sum_nonneg (by
        intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz
        exact add_nonneg (mul_nonneg (by linarith) (mul_self_nonneg x)) F.hη)
    have A1 := fsum_err F D (fun x y => y = x) (fun x => x) (fun _ => 0) (fun x => |x|)
      (fun x => abs_nonneg x)
      (by intro x y hxy; subst hxy; simp) b r2 hs
    have A2 := fsum_err F D (fun x y => ∃ q1, F.mul (x * x) q1 ∧ y = q1) (fun x => x*x)
      (fun x => F.u*(x*x)+F.η) (fun x => (1+F.u)*(x*x)+F.η)
      (fun x => add_nonneg (mul_nonneg (by linarith) (mul_self_nonneg x)) F.hη)
      (by
        intro x y hxy
        obtain ⟨q1, hq, hy⟩ := hxy
        subst hy
        have e := F.mul_err hq
        have e2 : |y| ≤ |x*x| + |y - x*x| := by
          have := abs_add_le (x*x) (y - x*x); simpa using this
        rw [abs_of_nonneg (mul_self_nonneg x)] at e2 e
        constructor <;> nlinarith [mul_self_nonneg x]) b r3 hs2
    simp only [Tree.flat, List.map_id'] at *
    refine ⟨by simp [h1, h2], ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
    · have hD : (1+F.u)^D - 1 ≤ (1+F.u)^k - 1 := by linarith
      have := hH0 b
      have e1 := A1.1
      simp only [List.map_const', List.sum_replicate, smul_zero, add_zero] at e1
      rw [h3]
      nlinarith [mul_le_mul_of_nonneg_right hD this]
    · rw [h3]
      have := hH0 b
      nlinarith [mul_le_mul_of_nonneg_right hPD this, A1.2]
    · have hD : (1+F.u)^D - 1 ≤ (1+F.u)^k - 1 := by linarith
      have := hH1 b
      rw [h4]
      nlinarith [mul_le_mul_of_nonneg_right hD this, A2.1]
    · rw [h4]
      have := hH1 b
      nlinarith [mul_le_mul_of_nonneg_right hPD this, A2.2]
  | node hl hr hm ihl ihr =>
    rename_i l r a c' w
    intro k hk
    simp only [Tree.depth] at hk
    obtain ⟨j, rfl⟩ : ∃ j, k = j+1 := ⟨k-1, by omega⟩
    have a1 := ihl j (by omega)
    have a2 := ihr j (by omega)
    obtain ⟨v_n, v_t, v_u, r1, r2, h1, ha1, h2, ha2, h3, h4⟩ := hm
    subst h4
    have hH0 : ∀ L : List ℝ, 0 ≤ (L.map fun x => |x|).sum := fun L =>
      List.sum_nonneg (by intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz; exact abs_nonneg x)
    have hH1 : ∀ L : List ℝ, 0 ≤ (L.map fun x => (1+F.u)*(x*x)+F.η).sum := fun L =>
      List.sum_nonneg (by
        intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz
        exact add_nonneg (mul_nonneg (by linarith [F.hu]) (mul_self_nonneg x)) F.hη)
    have s1 := add_step F j a.t c'.t r1 _ _ 0 0 _ _ (hH0 l.flat) (hH0 r.flat)
      (by linarith [a1.2.1.1]) a1.2.1.2 (by linarith [a2.2.1.1]) a2.2.1.2 ha1
    have s2 := add_step F j a.u c'.u r2 _ _ _ _ _ _ (hH1 l.flat) (hH1 r.flat)
      a1.2.2.1 a1.2.2.2 a2.2.2.1 a2.2.2.2 ha2
    refine ⟨by simp [h1, a1.1, a2.1, Tree.flat], ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩
    · simp only [Tree.flat, List.map_append, List.sum_append, h2]
      linarith [s1.1]
    · simp only [Tree.flat, List.map_append, List.sum_append, h2]
      linarith [s1.2]
    · simp only [Tree.flat, List.map_append, List.sum_append, h3]
      linarith [s2.1]
    · simp only [Tree.flat, List.map_append, List.sum_append, h3]
      linarith [s2.2]
theorem core_real (u η e e0 s μ q p r1 r2 r3 r4 r5 K : ℝ)
    (hu0 : 0 ≤ u) (hu : u ≤ 1/256) (hη0 : 0 ≤ η) (hηu : η ≤ u) (hs0 : 0 ≤ s)
    (hμ : |μ| ≤ s) (he0 : 0 ≤ e0) (he : u + e0*(1+u) ≤ e) (heK : e ≤ 16/15*K*u)
    (hK1 : 1 ≤ K) (hKu : K*u ≤ 1/16)
    (hq : |q - s^2| ≤ (u+e0*(1+u))*s^2 + (1+e0)*η)
    (hq' : |q| ≤ (1+e0)*(1+u)*s^2 + (1+e0)*η)
    (hp : |p - μ| ≤ e0*s) (hp' : |p| ≤ (1+e0)*s)
    (hr1 : |r1 - q| ≤ u*|q| + η) (hr2 : |r2 - p| ≤ u*|p| + η) (hr3 : |r3 - p| ≤ u*|p| + η)
    (hr4 : |r4 - r2*r3| ≤ u*|r2*r3| + η) (hr5 : |r5 - (r1 - r4)| ≤ u*|r1 - r4|) :
    |r5 - (s^2 - μ^2)| ≤ 4*(K+2)*u*s^2 + 6*η := by
  have hm0 : 0 ≤ s^2 := by positivity
  have he1 : e ≤ 1/15 := by linarith
  have he_0 : 0 ≤ e := by
    have := mul_nonneg he0 (by linarith : (0:ℝ) ≤ 1+u); linarith
  have hee0 : e0 ≤ e := by
    have := mul_nonneg he0 hu0; linarith
  have hηm : η*s^2 ≤ u*s^2 := mul_le_mul_of_nonneg_right hηu hm0
  have hem : e*s^2 ≤ 16/15*K*u*s^2 := mul_le_mul_of_nonneg_right heK hm0
  have hηη : η*η ≤ η*(1/256) := mul_le_mul_of_nonneg_left (by linarith) hη0
  have hs2 : 2*s ≤ 1 + s^2 := by linarith [sq_nonneg (s-1)]
  have hηs : 2*(η*s) ≤ η*(1+s^2) := by
    have := mul_le_mul_of_nonneg_left hs2 hη0; linarith
  have hμ2 : μ^2 ≤ s^2 := sq_le_sq' (abs_le.1 hμ).1 (abs_le.1 hμ).2
  have hμ0 : 0 ≤ μ^2 := sq_nonneg μ
  have hc : (1+e0)*(1+u) ≤ 1 + e := by linarith
  have hue : u*(1+e0) ≤ 1 := by have := mul_nonneg he0 hu0; linarith
  -- r1
  have E1 : |r1 - s^2| ≤ (e + u + u*e)*s^2 + 3*η := by
    have h1 := abs_sub_le r1 q (s^2)
    have A : u*|q| ≤ u*((1+e0)*(1+u)*s^2 + (1+e0)*η) := mul_le_mul_of_nonneg_left hq' hu0
    have B : u*((1+e0)*(1+u)*s^2) ≤ u*((1+e)*s^2) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hc hm0) hu0
    have D : (1+e0)*(1+u)*η ≤ 16/15*η :=
      mul_le_mul_of_nonneg_right (by linarith) hη0
    have h4 : (u+e0*(1+u))*s^2 ≤ e*s^2 := mul_le_mul_of_nonneg_right he hm0
    linarith
  -- r2, r3
  have hpp : u*|p| ≤ u*((1+e0)*s) := mul_le_mul_of_nonneg_left hp' hu0
  have h6 : (u + e0*(1+u))*s ≤ e*s := mul_le_mul_of_nonneg_right he hs0
  have h7 : (1+e0)*(1+u)*s ≤ (1+e)*s := mul_le_mul_of_nonneg_right hc hs0
  have h8 : (1+u)*|p| ≤ (1+u)*((1+e0)*s) := mul_le_mul_of_nonneg_left hp' (by linarith)
  have E2 : |r2 - μ| ≤ e*s + η := by
    have := abs_sub_le r2 p μ
    linarith
  have E3 : |r3 - μ| ≤ e*s + η := by
    have := abs_sub_le r3 p μ
    linarith
  have E2' : |r2| ≤ (1+e)*s + η := by
    have := abs_add_le (r2 - p) p
    rw [sub_add_cancel] at this
    linarith
  have E3' : |r3| ≤ (1+e)*s + η := by
    have := abs_add_le (r3 - p) p
    rw [sub_add_cancel] at this
    linarith
  have hpos1 : 0 ≤ e*s + η := add_nonneg (mul_nonneg he_0 hs0) hη0
  have hpos2 : 0 ≤ (1+e)*s + η := add_nonneg (mul_nonneg (by linarith) hs0) hη0
  have hW : (1+e)*η*(1+s^2) ≤ 16/15*η + 16/15*u*s^2 := by
    have : (1+e)*η ≤ 16/15*η := mul_le_mul_of_nonneg_right (by linarith) hη0
    have := mul_le_mul_of_nonneg_right this (by positivity : (0:ℝ) ≤ 1 + s^2)
    linarith
  have hee : e*e*s^2 ≤ e*s^2*(1/15) := by
    have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left he1 he_0) hm0
    linarith
  have hh := mul_le_mul_of_nonneg_left hηs (by linarith : (0:ℝ) ≤ 1 + e)
  have P1 : |r2*r3 - μ^2| ≤ e*(2+e)*s^2 + (1+e)*η*(1+s^2) + η*η := by
    have eq : r2*r3 - μ^2 = (r2 - μ)*r3 + μ*(r3 - μ) := by ring
    rw [eq]
    have a1 : |(r2 - μ)*r3| ≤ (e*s+η)*((1+e)*s+η) := by
      rw [abs_mul]; exact mul_le_mul E2 E3' (abs_nonneg _) hpos1
    have a2 : |μ*(r3 - μ)| ≤ s*(e*s+η) := by
      rw [abs_mul]; exact mul_le_mul hμ E3 (abs_nonneg _) hs0
    have := abs_add_le ((r2 - μ)*r3) (μ*(r3 - μ))
    linarith
  have Q1 : |r2*r3| ≤ (1+e)^2*s^2 + (1+e)*η*(1+s^2) + η*η := by
    rw [abs_mul]
    have := mul_le_mul E2' E3' (abs_nonneg _) hpos2
    linarith
  have hu2 : u*s^2 ≤ s^2*(1/256) := by
    have := mul_le_mul_of_nonneg_right hu hm0; linarith
  have Q2 : |r2*r3| ≤ 6/5*s^2 + 11/10*η := by
    have : (1+e)^2 ≤ 57/50 := by
      have := mul_le_mul_of_nonneg_left he1 he_0; linarith
    have := mul_le_mul_of_nonneg_right this hm0
    linarith
  have R4 : |r4 - μ^2| ≤ u*|r2*r3| + η + |r2*r3 - μ^2| := by
    have := abs_sub_le r4 (r2*r3) (μ^2)
    linarith
  have uQ : u*|r2*r3| ≤ u*(6/5*s^2 + 11/10*η) := mul_le_mul_of_nonneg_left Q2 hu0
  have uη : u*η ≤ η*(1/256) := by
    have := mul_le_mul_of_nonneg_right hu hη0; linarith
  have hue2 : u*e*s^2 ≤ u*s^2*(1/15) := by
    have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left he1 hu0) hm0
    linarith
  have hK0 : 0 ≤ K*u*s^2 := mul_nonneg (mul_nonneg (by linarith) hu0) hm0
  have hus : 0 ≤ u*s^2 := mul_nonneg hu0 hm0
  have T4 : |r4 - μ^2| ≤ 9/4*K*u*s^2 + 7/3*u*s^2 + 21/10*η := by
    linarith
  have hBB : ∀ B : ℝ, |r1 - s^2| + |r4 - μ^2| ≤ B → B ≤ B → |r5 - (s^2 - μ^2)| ≤ u*(s^2 + B) + B := by
    intro B hB _
    have h1 := abs_sub_le r5 (r1 - r4) (s^2 - μ^2)
    have h2 : |r1 - r4| ≤ |r1 - s^2| + |s^2 - μ^2| + |μ^2 - r4| := by
      have := abs_add_le (r1 - s^2 + (s^2 - μ^2)) (μ^2 - r4)
      have := abs_add_le (r1 - s^2) (s^2 - μ^2)
      have e : r1 - r4 = r1 - s^2 + (s^2 - μ^2) + (μ^2 - r4) := by ring
      rw [e]; linarith
    have h3 : |s^2 - μ^2| ≤ s^2 := by rw [abs_of_nonneg (by linarith)]; linarith
    have h4 : |μ^2 - r4| = |r4 - μ^2| := abs_sub_comm _ _
    have h5 : |(r1 - r4) - (s^2 - μ^2)| ≤ |r1 - s^2| + |r4 - μ^2| := by
      have := abs_add_le (r1 - s^2) (μ^2 - r4)
      have e : (r1 - r4) - (s^2 - μ^2) = (r1 - s^2) + (μ^2 - r4) := by ring
      rw [e]; linarith
    have := mul_le_mul_of_nonneg_left (by linarith : |r1 - r4| ≤ s^2 + B) hu0
    linarith
  have hBb : |r1 - s^2| + |r4 - μ^2| ≤ 10/3*K*u*s^2 + 7/2*u*s^2 + 26/5*η := by
    have := mul_le_mul_of_nonneg_right hKu hm0
    linarith
  have hfin := hBB _ hBb le_rfl
  have hBnn : 0 ≤ 10/3*K*u*s^2 + 7/2*u*s^2 + 26/5*η := by linarith
  have := mul_le_mul_of_nonneg_right hu hBnn
  have := mul_le_mul_of_nonneg_left hBb hu0
  linarith

theorem list_shift (L : List ℝ) (g : ℝ → ℝ) (c : ℝ) :
    (L.map fun x => (g x - c)^2).sum =
      (L.map fun x => (g x)^2).sum - 2*c*(L.map g).sum + L.length*c^2 := by
  induction L with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons] at *
    rw [ih]; push_cast; ring

theorem list_aff (L : List ℝ) (a b : ℝ) :
    (L.map fun x => a*(x*x)+b).sum = a*(L.map fun x => x*x).sum + L.length*b := by
  induction L with
  | nil => simp
  | cons x l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons] at *
    rw [ih]; push_cast; ring

theorem list_sq (L : List ℝ) : (L.map fun x => x*x).sum = (L.map fun x => x^2).sum := by
  congr 1; apply List.map_congr_left; intro x _; ring

theorem pow_bound (u : ℝ) (n : ℕ) (hu : 0 ≤ u) (hK : (n:ℝ)*u ≤ 1/16) :
    (1+u)^n - 1 ≤ 16/15*n*u := by
  have h1 : (1+u)^n ≤ Real.exp (n*u) := by
    rw [Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp u]) n
  have hx : 0 ≤ (n:ℝ)*u := mul_nonneg (Nat.cast_nonneg n) hu
  have h2 := Real.exp_bound_div_one_sub_of_interval hx (by linarith)
  have h3 : 1/(1 - (n:ℝ)*u) ≤ 1 + 16/15*((n:ℝ)*u) := by
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  linarith

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List (ℝ)), X ≠ [] →
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u → (X.length : ℝ) * F.u ≤ 1 →
    ∀ (t : Discovery.Tree (ℝ)), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
    ∀ (w : ℝ), fout_var F st w →
      |w - vr X| ≤ (((4 : ℝ) * ((D : ℝ) + (t.depth : ℝ) + (3 : ℝ)) * F.u * (msq X)) + ((6 : ℝ) * F.η)) := by
  intro F D lo hi X hX _ hsm hnu hnu1 t hflat hleaves hdep st hev w hw
  subst hflat
  obtain ⟨hn, ⟨ht1, ht2⟩, ⟨hu1, hu2⟩⟩ := tree_outer F D t st hev (D + t.depth) le_rfl
  obtain ⟨r1, r2, r3, r4, r5, m1, m2, m3, m4, m5, rfl⟩ := hw
  obtain ⟨hsu, -, -, -, -⟩ := hsm
  have hu0 := F.hu
  have hη0 := F.hη
  have hηe0 := F.hηe
  generalize t.flat = L at *
  have hlen : (1:ℝ) ≤ L.length := by
    have : 0 < L.length := List.length_pos_iff.2 hX
    exact_mod_cast this
  generalize hnn : (L.length : ℝ) = n at *
  have hn0 : 0 < n := by linarith
  have hηu : F.η ≤ F.u := by
    have := mul_le_mul_of_nonneg_right hlen hη0
    have := mul_nonneg hn0.le hηe0
    linarith
  simp only [list_aff, list_sq] at hu1 hu2
  rw [hnn] at hu1 hu2
  generalize hSd : (L.map fun x => x^2).sum = S at *
  generalize hTd : L.sum = T at *
  generalize hAd : (L.map fun x => |x|).sum = A1 at *
  have hSnn : 0 ≤ S := by
    rw [← hSd]; exact List.sum_nonneg (by
      intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz; positivity)
  have hmsq : msq L = S / n := by
    unfold msq; rw [hSd, hnn]
  have hs2 : (Real.sqrt (S/n))^2 = S/n := Real.sq_sqrt (div_nonneg hSnn hn0.le)
  generalize hsd : Real.sqrt (S/n) = s at *
  have hs0 : 0 ≤ s := by rw [← hsd]; exact Real.sqrt_nonneg _
  have hS : S = s^2 * n := by rw [hs2]; field_simp
  -- mean
  have hμsq : (T/n)^2 ≤ s^2 := by
    have h := list_shift L (fun x => x) (T/n)
    simp only [List.map_id'] at h
    rw [hSd, hTd, hnn] at h
    have hnn0 : 0 ≤ (L.map fun x => (x - T/n)^2).sum :=
      List.sum_nonneg (by
        intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz; positivity)
    have : n * (T/n)^2 ≤ S := by
      have e : 2*(T/n)*T = 2*n*(T/n)^2 := by field_simp
      linarith
    rw [hS] at this
    nlinarith
  have hμ : |T/n| ≤ s := by
    have := Real.abs_le_sqrt (x := T/n) (y := s^2) hμsq
    rwa [Real.sqrt_sq hs0] at this
  have hvr : vr L = s^2 - (T/n)^2 := by
    have h := list_shift L (fun x => x) (T/n)
    simp only [List.map_id'] at h
    unfold vr
    rw [hSd, hTd, hnn] at h
    rw [hTd, hnn, h]
    rw [hS]; field_simp; ring
  have hA : A1 ≤ s * n := by
    have h := list_shift L (fun x => |x|) (A1/n)
    simp only [sq_abs] at h
    rw [hSd, hAd, hnn] at h
    have hnn0 : 0 ≤ (L.map fun x => (|x| - A1/n)^2).sum :=
      List.sum_nonneg (by
        intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz; positivity)
    have : n * (A1/n)^2 ≤ S := by
      have e : 2*(A1/n)*A1 = 2*n*(A1/n)^2 := by field_simp
      linarith
    rw [hS] at this
    have h2 : (A1/n)^2 ≤ s^2 := by nlinarith
    have := Real.abs_le_sqrt (x := A1/n) (y := s^2) h2
    rw [Real.sqrt_sq hs0] at this
    have h3 := (le_abs_self (A1/n)).trans this
    rwa [div_le_iff₀ hn0] at h3
  have hA0 : 0 ≤ A1 := by
    rw [← hAd]; exact List.sum_nonneg (by
      intro z hz; obtain ⟨x, _, rfl⟩ := List.mem_map.1 hz; exact abs_nonneg x)
  -- the exponent
  set k : ℕ := D + t.depth with hk
  have hKr : ((D:ℝ) + t.depth + 1) = ((k+1 : ℕ) : ℝ) := by rw [hk]; push_cast; ring
  have hKu : ((k+1 : ℕ) : ℝ) * F.u ≤ 1/16 := by
    have h0 : 0 ≤ ((k+1 : ℕ) : ℝ) := Nat.cast_nonneg _
    have := mul_le_mul_of_nonneg_left (by linarith [F.hue, F.hul] : F.u ≤ F.u + F.ue + F.ul) h0
    rw [← hKr] at this ⊢
    linarith
  have hpk : (1+F.u)^k = 1 + ((1+F.u)^k - 1) := by ring
  generalize he0d : (1+F.u)^k - 1 = e0 at hpk
  have he0 : 0 ≤ e0 := by
    have : 1 ≤ (1+F.u)^k := one_le_pow₀ (by linarith)
    linarith
  rw [hpk] at hu1 hu2 ht1 ht2
  have he : F.u + e0*(1+F.u) ≤ (1+F.u)^(k+1) - 1 := by
    rw [pow_succ, hpk]; linarith
  have heK := pow_bound F.u (k+1) hu0 hKu
  have hK1 : (1:ℝ) ≤ ((k+1 : ℕ) : ℝ) := Nat.one_le_cast.2 (Nat.succ_pos k)
  rw [hn] at m1 m2 m3
  have hr1 := F.mul_err m1
  have hr2 := F.mul_err m2
  have hr3 := F.mul_err m3
  have hr4 := F.mul_err m4
  have hr5 := F.add_err m5
  have hq : |st.u/n - s^2| ≤ (F.u+e0*(1+F.u))*s^2 + (1+e0)*F.η := by
    have e1 : st.u/n - s^2 = (st.u - S)/n := by rw [hS]; field_simp
    rw [e1, abs_div, abs_of_pos hn0, div_le_iff₀ hn0]
    rw [hS] at hu1 ⊢
    linarith
  have hq' : |st.u/n| ≤ (1+e0)*(1+F.u)*s^2 + (1+e0)*F.η := by
    rw [abs_div, abs_of_pos hn0, div_le_iff₀ hn0]
    rw [hS] at hu2
    linarith
  have hp : |st.t/n - T/n| ≤ e0*s := by
    have e1 : st.t/n - T/n = (st.t - T)/n := by field_simp
    rw [e1, abs_div, abs_of_pos hn0, div_le_iff₀ hn0]
    have := mul_le_mul_of_nonneg_left hA he0
    have ht1' : |st.t - T| ≤ e0 * A1 := by linarith
    linarith
  have hp' : |st.t/n| ≤ (1+e0)*s := by
    rw [abs_div, abs_of_pos hn0, div_le_iff₀ hn0]
    have := mul_le_mul_of_nonneg_left hA (by linarith : (0:ℝ) ≤ 1 + e0)
    have ht2' : |st.t| ≤ (1+e0) * A1 := by linarith
    linarith
  have key := core_real F.u F.η ((1+F.u)^(k+1) - 1) e0 s (T/n) (st.u/n) (st.t/n) r1 r2 r3 r4 w
    ((k+1 : ℕ) : ℝ) hu0 hsu hη0 hηu hs0 hμ he0 he heK hK1 hKu hq hq' hp hp' hr1 hr2 hr3 hr4 hr5
  rw [hvr, hmsq, ← hs2]
  have hD : (D:ℝ) + (t.depth:ℝ) + 3 = ((k+1 : ℕ) : ℝ) + 2 := by rw [hk]; push_cast; ring
  rw [hD]
  exact key

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.VarianceSumsq
