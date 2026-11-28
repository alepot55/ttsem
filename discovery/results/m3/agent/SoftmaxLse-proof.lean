import FPStatements

namespace Discovery.FP.SoftmaxLse

open Discovery Discovery.Library.SoftmaxLse

lemma lsum_mono {f g : ℝ → ℝ} : ∀ (A : List ℝ), (∀ x ∈ A, f x ≤ g x) →
    (A.map f).sum ≤ (A.map g).sum
  | [], _ => by simp
  | a :: as, h => by
    simp only [List.map_cons, List.sum_cons]
    exact add_le_add (h a (by simp)) (lsum_mono as (fun x hx => h x (by simp [hx])))

lemma lsum_affine (c e : ℝ) (f : ℝ → ℝ) : ∀ A : List ℝ,
    (A.map fun x => c * f x + e).sum = c * (A.map f).sum + A.length * e
  | [] => by simp
  | a :: as => by
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    rw [lsum_affine c e f as]; push_cast; ring

lemma abs_exp_sub_le {a b c : ℝ} (ha : a ≤ c) (hb : b ≤ c) :
    |Real.exp a - Real.exp b| ≤ Real.exp c * |a - b| := by
  have key : ∀ a b : ℝ, a ≤ b → b ≤ c → Real.exp b - Real.exp a ≤ Real.exp c * (b - a) := by
    intro a b hab hbc
    have h1 : Real.exp a = Real.exp b * Real.exp (a - b) := by rw [← Real.exp_add]; ring_nf
    have h2 := Real.add_one_le_exp (a - b)
    have h3 := Real.exp_pos b
    have h4 : Real.exp b ≤ Real.exp c := Real.exp_le_exp.mpr hbc
    rw [h1]; nlinarith
  rcases le_total a b with hab | hab
  · rw [abs_sub_comm (Real.exp a), abs_sub_comm a b, abs_of_nonneg (by linarith [Real.exp_le_exp.mpr hab]),
      abs_of_nonneg (by linarith)]
    exact key a b hab hb
  · rw [abs_of_nonneg (by linarith [Real.exp_le_exp.mpr hab]), abs_of_nonneg (by linarith)]
    exact key b a hab ha

lemma exp_close {d δ u : ℝ} (hd : d ≤ 0) (hδ : |δ| ≤ u) (hu : u ≤ 1/2) :
    |Real.exp (d * (1 + δ)) - Real.exp d| ≤ u := by
  have hδ' := abs_le.mp hδ
  have h1 : d * (1 + δ) ≤ d / 2 := by nlinarith
  have h2 : d ≤ d / 2 := by linarith
  have h3 := abs_exp_sub_le h1 h2
  have h4 : |d * (1 + δ) - d| ≤ -d * u := by
    rw [show d * (1 + δ) - d = d * δ by ring, abs_mul, abs_of_nonpos hd]
    exact mul_le_mul_of_nonneg_left hδ (by linarith)
  have h5 : -d ≤ Real.exp (-d / 2) := by
    have := Real.quadratic_le_exp_of_nonneg (x := -d/2) (by linarith)
    nlinarith
  have h6 : Real.exp (d / 2) * Real.exp (-d / 2) = 1 := by rw [← Real.exp_add]; ring_nf; simp
  have h7 := Real.exp_pos (d/2)
  have hu0 : 0 ≤ u := le_trans (abs_nonneg _) hδ
  calc _ ≤ Real.exp (d / 2) * (-d * u) := le_trans h3 (mul_le_mul_of_nonneg_left h4 h7.le)
    _ ≤ Real.exp (d / 2) * (Real.exp (-d / 2) * u) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right h5 hu0) h7.le
    _ = u := by rw [← mul_assoc, h6, one_mul]

lemma log_ratio {p q r S : ℝ} (hp : 0 < p) (hS : 0 < S) (h1 : p * S ≤ r) (h2 : r ≤ q * S) :
    Real.log p ≤ Real.log r - Real.log S ∧ Real.log r - Real.log S ≤ Real.log q := by
  have hr : 0 < r := lt_of_lt_of_le (mul_pos hp hS) h1
  have hq : 0 < q := by
    by_contra hq; push Not at hq; nlinarith
  constructor
  · have := Real.log_le_log (mul_pos hp hS) h1
    rw [Real.log_mul hp.ne' hS.ne'] at this; linarith
  · have := Real.log_le_log hr h2
    rw [Real.log_mul hq.ne' hS.ne'] at this; linarith

lemma log_one_sub_ge {x c : ℝ} (hx0 : 0 ≤ x) (hxc : x ≤ c) (hc : c < 1) :
    -(x / (1 - c)) ≤ Real.log (1 - x) := by
  have h := Real.one_sub_inv_le_log_of_pos (x := 1 - x) (by linarith)
  have hx1 : 1 - x ≠ 0 := by linarith
  have : 1 - (1 - x)⁻¹ = -(x / (1 - x)) := by field_simp; ring
  rw [this] at h
  have : x / (1 - x) ≤ x / (1 - c) := div_le_div_of_nonneg_left hx0 (by linarith) (by linarith)
  linarith

lemma log_one_add_le {x : ℝ} (hx : 0 ≤ x) : Real.log (1 + x) ≤ x := by
  have := Real.log_le_sub_one_of_pos (x := 1 + x) (by linarith); linarith

lemma lse2 (x y : ℝ) : max x y + Real.log (Real.exp (x - max x y) + Real.exp (y - max x y))
    = Real.log (Real.exp x + Real.exp y) := by
  have h : Real.exp (x - max x y) + Real.exp (y - max x y)
      = (Real.exp x + Real.exp y) * Real.exp (-max x y) := by
    rw [add_mul, ← Real.exp_add, ← Real.exp_add]; ring_nf
  rw [h, Real.log_mul (by positivity) (by positivity), Real.log_exp]; ring

lemma lse2_lip {x y a b ε : ℝ} (ha : |x - a| ≤ ε) (hb : |y - b| ≤ ε) :
    |Real.log (Real.exp x + Real.exp y) - Real.log (Real.exp a + Real.exp b)| ≤ ε := by
  have ha' := abs_le.mp ha; have hb' := abs_le.mp hb
  have key : ∀ x y a b : ℝ, x ≤ a + ε → y ≤ b + ε →
      Real.log (Real.exp x + Real.exp y) ≤ Real.log (Real.exp a + Real.exp b) + ε := by
    intro x y a b h1 h2
    have : Real.exp x + Real.exp y ≤ (Real.exp a + Real.exp b) * Real.exp ε := by
      rw [add_mul, ← Real.exp_add, ← Real.exp_add]
      exact add_le_add (Real.exp_le_exp.mpr h1) (Real.exp_le_exp.mpr h2)
    have := Real.log_le_log (by positivity) this
    rwa [Real.log_mul (by positivity) (by positivity), Real.log_exp] at this
  rw [abs_le]; constructor
  · have := key a b x y (by linarith) (by linarith); linarith
  · have := key x y a b (by linarith) (by linarith); linarith

lemma merge_s (a b : State_a) : (merge_a a b).s = Real.log (Real.exp a.s + Real.exp b.s) := by
  simp only [merge_a]; exact lse2 a.s b.s

lemma S_bounds {A : List ℝ} (hA : A ≠ []) :
    1 ≤ (A.map fun x => Real.exp (x - lmax A)).sum ∧
    (A.map fun x => Real.exp (x - lmax A)).sum ≤ A.length := by
  constructor
  · have := List.single_le_sum (l := A.map fun x => Real.exp (x - lmax A))
      (by intro x hx; obtain ⟨y, -, rfl⟩ := List.mem_map.mp hx; positivity) 1
      (List.mem_map.mpr ⟨lmax A, lmax_mem hA, by simp⟩)
    exact this
  · have := List.sum_le_length_nsmul (A.map fun x => Real.exp (x - lmax A)) 1
      (by intro x hx; obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
          exact Real.exp_le_one_iff.mpr (by linarith [le_lmax hy]))
    simpa using this

lemma block_s_eq {A : List ℝ} (hA : A ≠ []) :
    (block_a A).s = Real.log ((A.map Real.exp).sum) := by
  have h : (A.map fun x => Real.exp (x - lmax A)).sum = (A.map Real.exp).sum * Real.exp (- lmax A) := by
    rw [← List.sum_map_mul_right]; congr 1; apply List.map_congr_left; intro x _
    rw [← Real.exp_add]; ring_nf
  have h1 := (S_bounds hA).1
  have hT : (A.map Real.exp).sum ≠ 0 := by
    intro h0; rw [h, h0] at h1; simp at h1; linarith
  simp only [block_a]; rw [h, Real.log_mul hT (by positivity), Real.log_exp]; ring

lemma block_abs {A : List ℝ} {lo hi n : ℝ} (hA : A ≠ []) (hAb : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi)
    (hk : (A.length : ℝ) ≤ n) : |(block_a A).s| ≤ max |lo| |hi| + n - 1 := by
  obtain ⟨h1, h2⟩ := S_bounds hA
  have hm := hAb _ (lmax_mem hA)
  have hl0 := Real.log_nonneg h1
  have hl1 := Real.log_le_sub_one_of_pos (lt_of_lt_of_le one_pos h1)
  have : |lmax A| ≤ max |lo| |hi| := by
    rw [abs_le]; constructor
    · have := neg_abs_le lo; have := le_max_left |lo| |hi|; linarith
    · have := le_abs_self hi; have := le_max_right |lo| |hi|; linarith
  simp only [block_a]
  calc _ ≤ |lmax A| + |Real.log (A.map fun x => Real.exp (x - lmax A)).sum| := abs_add_le _ _
    _ ≤ _ := by rw [abs_of_nonneg hl0]; linarith

set_option maxHeartbeats 2000000 in
lemma block_step (F : Rounding) (hs : small F) (D : ℕ) (lo hi n : ℝ) (A : List ℝ) (hA : A ≠ [])
    (hAb : ∀ x ∈ A, lo ≤ x ∧ x ≤ hi) (hk : (A.length : ℝ) ≤ n) (hη : n * F.ηe ≤ F.u)
    (st : State_a) (h : fblock_a F D A st) :
    |st.s - (block_a A).s| ≤
      2 * (F.u * (D + (hi - lo) + max |lo| |hi| + n + 2) + F.ue + F.ul * (n + 1)) := by
  obtain ⟨hu1, hue1, hul1, -, -⟩ := hs
  have hu0 := F.hu; have hue0 := F.hue; have hul0 := F.hul; have hηe0 := F.hηe
  obtain ⟨v_m, v_s, r1, r2, r3, r4, h1, h2, hsum, hlog, hadd, h6, rfl⟩ := h
  subst h1 h2 h6
  have hmA : lmax A ∈ A := lmax_mem hA
  have hmb := hAb _ hmA
  have hxm : ∀ x ∈ A, x ≤ lmax A := fun x hx => le_lmax hx
  obtain ⟨hS1, hSk⟩ := S_bounds hA
  set m := lmax A with hm
  set S := (A.map fun x => Real.exp (x - m)).sum with hS
  show |v_s - (m + Real.log S)| ≤ _
  set K := hi - lo with hK
  have hK0 : 0 ≤ K := by linarith
  set T := (A.map fun x => Real.exp ((x - m) * (1 + F.u))).sum with hT
  set T' := (A.map fun x => Real.exp ((x - m) * (1 - F.u))).sum with hT'
  have hnn : ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) → ∀ z ∈ A.map f, 0 ≤ z := by
    intro f hf z hz; obtain ⟨y, -, rfl⟩ := List.mem_map.mp hz; exact hf y
  have hT1 : 1 ≤ T := by
    have := List.single_le_sum (hnn (fun x => Real.exp ((x - m) * (1 + F.u)))
      (fun x => (Real.exp_pos _).le)) 1 (List.mem_map.mpr ⟨m, hmA, by simp⟩)
    exact this
  have hT'1 : 1 ≤ T' := by
    have := List.single_le_sum (hnn (fun x => Real.exp ((x - m) * (1 - F.u)))
      (fun x => (Real.exp_pos _).le)) 1 (List.mem_map.mpr ⟨m, hmA, by simp⟩)
    exact this
  have hTS : S * Real.exp (-(F.u * K)) ≤ T := by
    rw [hS, ← List.sum_map_mul_right]
    apply lsum_mono; intro x hx
    rw [← Real.exp_add]; apply Real.exp_le_exp.mpr
    have := hAb x hx; nlinarith
  have hT'S : T' ≤ S * Real.exp (F.u * K) := by
    rw [hS, ← List.sum_map_mul_right]
    apply lsum_mono; intro x hx
    rw [← Real.exp_add]; apply Real.exp_le_exp.mpr
    have := hAb x hx; nlinarith
  have hb := fsum_bounds F D _
    (fun x => max 0 (Real.exp (min ((x - m) * (1 + F.u)) ((x - m) * (1 - F.u))) * (1 - F.ue) - F.ηe))
    (fun x => Real.exp (max ((x - m) * (1 + F.u)) ((x - m) * (1 - F.u))) * (1 + F.ue) + F.ηe)
    (fun x => le_max_left _ _) ?_ A r2 hsum
  swap
  · rintro x y ⟨q1, q2, hq1, hq2, rfl⟩
    obtain ⟨δ, hδ, rfl⟩ := hq1
    obtain ⟨e1, e2, e3⟩ := F.exp_bounds hq2
    have hδ' := abs_le.mp hδ
    have hmin : min ((x - m) * (1 + F.u)) ((x - m) * (1 - F.u)) ≤ (x - m) * (1 + δ) := by
      rcases le_total (x - m) 0 with h | h
      · exact min_le_of_left_le (by nlinarith)
      · exact min_le_of_right_le (by nlinarith)
    have hmax : (x - m) * (1 + δ) ≤ max ((x - m) * (1 + F.u)) ((x - m) * (1 - F.u)) := by
      rcases le_total (x - m) 0 with h | h
      · exact le_max_of_le_right (by nlinarith)
      · exact le_max_of_le_left (by nlinarith)
    have m1 := Real.exp_le_exp.mpr hmin
    have m2 := Real.exp_le_exp.mpr hmax
    constructor
    · apply max_le e3; nlinarith
    · nlinarith
  obtain ⟨hlo, hhi⟩ := hb
  have hL : (1 - F.ue) * T + (A.length : ℝ) * (-F.ηe) ≤
      (A.map fun x => max 0 (Real.exp (min ((x - m) * (1 + F.u)) ((x - m) * (1 - F.u))) * (1 - F.ue) - F.ηe)).sum := by
    rw [hT, ← lsum_affine]; apply lsum_mono; intro x hx
    have := hxm x hx
    rw [min_eq_left (by nlinarith)]
    exact le_trans (le_of_eq (by ring)) (le_max_right _ _)
  have hH : (A.map fun x => Real.exp (max ((x - m) * (1 + F.u)) ((x - m) * (1 - F.u))) * (1 + F.ue) + F.ηe).sum
      ≤ (1 + F.ue) * T' + (A.length : ℝ) * F.ηe := by
    rw [hT', ← lsum_affine]; apply lsum_mono; intro x hx
    have := hxm x hx
    rw [max_eq_right (by nlinarith)]
    exact le_of_eq (by ring)
  have hkη : (A.length : ℝ) * F.ηe ≤ F.u := le_trans (mul_le_mul_of_nonneg_right hk hηe0) hη
  -- r2 bounds
  have hP : 0 < (1 - F.u) ^ D * (1 - F.ue - F.u) * Real.exp (-(F.u * K)) := by
    have : 0 < 1 - F.u := by linarith
    have : 0 < 1 - F.ue - F.u := by linarith
    positivity
  have hS0 : 0 < S := by linarith
  have hr2lo : (1 - F.u) ^ D * (1 - F.ue - F.u) * Real.exp (-(F.u * K)) * S ≤ r2 := by
    have hpow : 0 ≤ (1 - F.u) ^ D := pow_nonneg (by linarith) _
    have : (1 - F.ue - F.u) * Real.exp (-(F.u * K)) * S ≤ (1 - F.ue) * T + (A.length : ℝ) * (-F.ηe) := by
      have h1 : (1 - F.ue - F.u) * (S * Real.exp (-(F.u * K))) ≤ (1 - F.ue - F.u) * T :=
        mul_le_mul_of_nonneg_left hTS (by linarith)
      have h2 : F.u * 1 ≤ F.u * T := mul_le_mul_of_nonneg_left hT1 hu0
      linarith
    calc _ = (1 - F.u) ^ D * ((1 - F.ue - F.u) * Real.exp (-(F.u * K)) * S) := by ring
      _ ≤ (1 - F.u) ^ D * ((1 - F.ue) * T + (A.length : ℝ) * (-F.ηe)) :=
          mul_le_mul_of_nonneg_left this hpow
      _ ≤ r2 := le_trans (mul_le_mul_of_nonneg_left hL hpow) hlo
  have hr2hi : r2 ≤ (1 + F.u) ^ D * (1 + F.ue + F.u) * Real.exp (F.u * K) * S := by
    have hpow : 0 ≤ (1 + F.u) ^ D := pow_nonneg (by linarith) _
    have : (1 + F.ue) * T' + (A.length : ℝ) * F.ηe ≤ (1 + F.ue + F.u) * Real.exp (F.u * K) * S := by
      have h1 : (1 + F.ue + F.u) * T' ≤ (1 + F.ue + F.u) * (S * Real.exp (F.u * K)) :=
        mul_le_mul_of_nonneg_left hT'S (by linarith)
      have h2 : F.u * 1 ≤ F.u * T' := mul_le_mul_of_nonneg_left hT'1 hu0
      linarith
    calc r2 ≤ _ := hhi
      _ ≤ (1 + F.u) ^ D * ((1 + F.ue) * T' + (A.length : ℝ) * F.ηe) :=
          mul_le_mul_of_nonneg_left hH hpow
      _ ≤ (1 + F.u) ^ D * ((1 + F.ue + F.u) * Real.exp (F.u * K) * S) :=
          mul_le_mul_of_nonneg_left this hpow
      _ = _ := by ring
  obtain ⟨lr1, lr2⟩ := log_ratio hP hS0 hr2lo hr2hi
  have hr2pos : 0 < r2 := lt_of_lt_of_le (mul_pos hP hS0) hr2lo
  have hp1 : 0 < 1 - F.u := by linarith
  have hp2 : 0 < 1 - F.ue - F.u := by linarith
  have hp3 : 0 < 1 + F.u := by linarith
  have hp4 : 0 < 1 + F.ue + F.u := by linarith
  rw [Real.log_mul (mul_pos (pow_pos hp1 D) hp2).ne' (by positivity),
    Real.log_mul (pow_pos hp1 D).ne' hp2.ne', Real.log_pow, Real.log_exp] at lr1
  rw [Real.log_mul (mul_pos (pow_pos hp3 D) hp4).ne' (by positivity),
    Real.log_mul (pow_pos hp3 D).ne' hp4.ne', Real.log_pow, Real.log_exp] at lr2
  clear hlo hhi hL hH hr2lo hr2hi hTS hT'S hT1 hT'1 hsum hP hnn
  clear_value T T' K S m
  have g1 := log_one_sub_ge hu0 hu1 (by norm_num)
  have g2 := log_one_sub_ge (x := F.ue + F.u) (c := 1/64 + 1/256) (by linarith) (by linarith) (by norm_num)
  have g3 := log_one_add_le hu0
  have g4 := log_one_add_le (x := F.ue + F.u) (by linarith)
  have hD0 : (0 : ℝ) ≤ D := Nat.cast_nonneg D
  obtain ⟨E1, hE1⟩ : ∃ E1 : ℝ, E1 = (D : ℝ) * F.u * (256 / 255) + (F.ue + F.u) * (256 / 251) + F.u * K := ⟨_, rfl⟩
  have hE1 : |Real.log r2 - Real.log S| ≤ E1 := by
    rw [abs_le]; constructor
    · have : (D : ℝ) * Real.log (1 - F.u) ≥ (D : ℝ) * (-(F.u / (1 - 1 / 256))) :=
        mul_le_mul_of_nonneg_left g1 hD0
      rw [show 1 - F.ue - F.u = 1 - (F.ue + F.u) by ring] at lr1
      have e1 : F.u / (1 - 1 / 256) = F.u * (256 / 255) := by ring
      have e2 : (F.ue + F.u) / (1 - (1 / 64 + 1 / 256)) = (F.ue + F.u) * (256 / 251) := by ring
      rw [e1] at this; rw [e2] at g2
      linarith
    · have : (D : ℝ) * Real.log (1 + F.u) ≤ (D : ℝ) * F.u := mul_le_mul_of_nonneg_left g3 hD0
      rw [show 1 + F.ue + F.u = 1 + (F.ue + F.u) by ring] at lr2
      have : 0 ≤ (D : ℝ) * F.u := by positivity
      linarith
  have hE1nn : 0 ≤ E1 := le_trans (abs_nonneg _) hE1
  have hLS0 := Real.log_nonneg hS1
  have hLSk : Real.log S ≤ n - 1 := by
    have := Real.log_le_sub_one_of_pos hS0; linarith
  -- r3
  have h3 := F.log_err hlog
  have habs2 : |Real.log r2| ≤ n - 1 + E1 := by
    have : |Real.log r2| ≤ |Real.log S| + |Real.log r2 - Real.log S| := by
      have := abs_sub_abs_le_abs_sub (Real.log r2) (Real.log S); linarith
    rw [abs_of_nonneg hLS0] at this; linarith
  have e3 : |r3 - Real.log S| ≤ E1 + F.ul * (n - 1 + E1) := by
    have : |r3 - Real.log S| ≤ |r3 - Real.log r2| + |Real.log r2 - Real.log S| := abs_sub_le _ _ _
    have := mul_le_mul_of_nonneg_left habs2 hul0
    linarith
  obtain ⟨e3v, he3v⟩ : ∃ e : ℝ, e = E1 + F.ul * (n - 1 + E1) := ⟨_, rfl⟩
  rw [← he3v] at e3
  have h4 := F.add_err hadd
  have hmM : |m| ≤ max |lo| |hi| := by
    rw [abs_le]; constructor
    · have := neg_abs_le lo; have := le_max_left |lo| |hi|; linarith
    · have := le_abs_self hi; have := le_max_right |lo| |hi|; linarith
  have habs4 : |m + r3| ≤ max |lo| |hi| + (n - 1) + e3v := by
    have : |m + r3| ≤ |m| + |r3| := abs_add_le _ _
    have : |r3| ≤ |Real.log S| + |r3 - Real.log S| := by
      have := abs_sub_abs_le_abs_sub r3 (Real.log S); linarith
    rw [abs_of_nonneg hLS0] at this; linarith
  have hfin : |v_s - (m + Real.log S)| ≤ F.u * (max |lo| |hi| + (n - 1) + e3v) + e3v := by
    have : |v_s - (m + Real.log S)| ≤ |v_s - (m + r3)| + |(m + r3) - (m + Real.log S)| :=
      abs_sub_le _ _ _
    have h5 : |(m + r3) - (m + Real.log S)| = |r3 - Real.log S| := by ring_nf
    have := mul_le_mul_of_nonneg_left habs4 hu0
    linarith
  refine le_trans hfin ?_
  -- arithmetic
  have hn1 : 1 ≤ n := by
    have : (1 : ℝ) ≤ A.length := by
      have := List.length_pos_of_ne_nil hA; exact_mod_cast this
    linarith
  have hKM : K ≤ 2 * max |lo| |hi| := by
    have := neg_abs_le lo; have := le_abs_self hi
    have := le_max_left |lo| |hi|; have := le_max_right |lo| |hi|; linarith
  have hM0 : 0 ≤ max |lo| |hi| := le_trans (abs_nonneg _) (le_max_left _ _)
  have q1 : F.ul * E1 ≤ (1 / 64) * E1 := mul_le_mul_of_nonneg_right hul1 hE1nn
  have he3nn : 0 ≤ e3v := by
    have := mul_nonneg hul0 (show 0 ≤ n - 1 + E1 by linarith); linarith
  have q2 : F.u * e3v ≤ (1 / 256) * e3v := mul_le_mul_of_nonneg_right hu1 he3nn
  have q4 : F.ul * (n - 1) ≥ 0 := mul_nonneg hul0 (by linarith)
  have q5 : F.u * K ≤ F.u * (2 * max |lo| |hi|) := mul_le_mul_of_nonneg_left hKM hu0
  have q6 : 0 ≤ F.u * max |lo| |hi| := mul_nonneg hu0 hM0
  have q7 : 0 ≤ (D : ℝ) * F.u := mul_nonneg hD0 hu0
  have q8 : 0 ≤ F.u * K := mul_nonneg hu0 hK0
  have q9 : 0 ≤ F.u * n := mul_nonneg hu0 (by linarith)
  linarith

lemma merge_term (F : Rounding) (hu : F.u ≤ 1 / 2) {z r v : ℝ} (hz : z ≤ 0) (h1 : F.add z r)
    (h2 : F.exp r v) :
    |v - Real.exp z| ≤ F.ue * (Real.exp z + F.u) + F.ηe + F.u ∧ 0 ≤ v := by
  obtain ⟨δ, hδ, rfl⟩ := h1
  have hc := exp_close hz hδ hu
  obtain ⟨δe, e, hδe, he, rfl, hv⟩ := h2
  refine ⟨?_, hv⟩
  have hc' := abs_le.mp hc
  have hpos := Real.exp_pos (z * (1 + δ))
  have hδe' := abs_le.mp hδe
  have he' := abs_le.mp he
  have hue0 := F.hue
  have k1 : |Real.exp (z * (1 + δ)) * δe| ≤ F.ue * (Real.exp z + F.u) := by
    rw [abs_mul, abs_of_pos hpos]
    calc _ ≤ (Real.exp z + F.u) * F.ue :=
          mul_le_mul (by linarith) hδe (abs_nonneg _) (by linarith)
      _ = _ := by ring
  have k2 : Real.exp (z * (1 + δ)) * (1 + δe) + e - Real.exp z
      = Real.exp (z * (1 + δ)) * δe + e + (Real.exp (z * (1 + δ)) - Real.exp z) := by ring
  rw [k2]
  calc _ ≤ |Real.exp (z * (1 + δ)) * δe + e| + |Real.exp (z * (1 + δ)) - Real.exp z| := abs_add_le _ _
    _ ≤ |Real.exp (z * (1 + δ)) * δe| + |e| + |Real.exp (z * (1 + δ)) - Real.exp z| := by
        linarith [abs_add_le (Real.exp (z * (1 + δ)) * δe) e]
    _ ≤ _ := by linarith

set_option maxHeartbeats 2000000 in
lemma merge_step (F : Rounding) (hs : small F) (hηe : F.ηe ≤ F.u) (M n ε a b : ℝ)
    (L R st : State_a) (hn : 2 ≤ n) (hM : 0 ≤ M) (hε : ε ≤ M / 5 + n / 5 + 1)
    (ha : |L.s - a| ≤ ε) (hb : |R.s - b| ≤ ε)
    (hl : |Real.log (Real.exp a + Real.exp b)| ≤ M + n - 1) (h : fmerge_a F L R st) :
    |st.s - Real.log (Real.exp a + Real.exp b)| ≤ ε + 2 * (F.u * (M + n + 2) + F.ue + F.ul) := by
  obtain ⟨hu1, hue1, hul1, -, -⟩ := hs
  have hu0 := F.hu; have hue0 := F.hue; have hul0 := F.hul; have hηe0 := F.hηe
  obtain ⟨v_m, v_s, r1, r2, r3, r4, r5, r6, r7, hm, h1, h2, h3, h4, h5, h6, h7, h8, rfl⟩ := h
  subst h8
  show |v_s - _| ≤ _
  obtain ⟨t1, t1p⟩ := merge_term F (by linarith) (by rw [hm]; exact sub_nonpos.mpr (le_max_left _ _)) h1 h2
  obtain ⟨t2, t2p⟩ := merge_term F (by linarith) (by rw [hm]; exact sub_nonpos.mpr (le_max_right _ _)) h3 h4
  set E := Real.exp (L.s - v_m) + Real.exp (R.s - v_m) with hE
  have hE1 : 1 ≤ E := by
    rcases le_total L.s R.s with hh | hh
    · have : v_m = R.s := by rw [hm]; exact max_eq_right hh
      rw [hE, this, sub_self, Real.exp_zero]; linarith [Real.exp_pos (L.s - R.s)]
    · have : v_m = L.s := by rw [hm]; exact max_eq_left hh
      rw [hE, this, sub_self, Real.exp_zero]; linarith [Real.exp_pos (R.s - L.s)]
  have hlE : Real.log (Real.exp L.s + Real.exp R.s) = v_m + Real.log E := by
    rw [hE, hm]; exact (lse2 L.s R.s).symm
  have hlogE0 := Real.log_nonneg hE1
  have hlogE1 : Real.log E ≤ 1 := by
    have h2E : E ≤ 2 := by
      have := Real.exp_le_one_iff.mpr (show L.s - v_m ≤ 0 by rw [hm]; exact sub_nonpos.mpr (le_max_left _ _))
      have := Real.exp_le_one_iff.mpr (show R.s - v_m ≤ 0 by rw [hm]; exact sub_nonpos.mpr (le_max_right _ _))
      linarith
    have := Real.log_le_sub_one_of_pos (show 0 < E by linarith); linarith
  clear_value E
  obtain ⟨g, hg⟩ : ∃ g : ℝ, g = F.ue + (129 / 32) * F.u := ⟨_, rfl⟩
  have hsum : |r2 + r4 - E| ≤ g * E := by
    have : r2 + r4 - E = (r2 - Real.exp (L.s - v_m)) + (r4 - Real.exp (R.s - v_m)) := by rw [hE]; ring
    rw [this]
    have := abs_add_le (r2 - Real.exp (L.s - v_m)) (r4 - Real.exp (R.s - v_m))
    have k : F.ue * F.u ≤ F.u / 64 := by nlinarith
    have k2 : g * 1 ≤ g * E := mul_le_mul_of_nonneg_left hE1 (by rw [hg]; positivity)
    rw [hg] at k2 ⊢
    nlinarith
  have hsum' := abs_le.mp hsum
  obtain ⟨b5l, b5h⟩ := F.add_bounds (by linarith) h5
  have hg1 : g ≤ 1 / 30 := by rw [hg]; linarith
  have hg0 : 0 ≤ g := by rw [hg]; positivity
  have hp1 : 0 < 1 - F.u := by linarith
  have hp2 : 0 < 1 - g := by linarith
  have r5lo : (1 - F.u) * (1 - g) * E ≤ r5 := by
    have : (1 - g) * E ≤ r2 + r4 := by linarith
    calc _ = (1 - F.u) * ((1 - g) * E) := by ring
      _ ≤ (1 - F.u) * (r2 + r4) := mul_le_mul_of_nonneg_left this hp1.le
      _ ≤ r5 := b5l
  have r5hi : r5 ≤ (1 + F.u) * (1 + g) * E := by
    have : r2 + r4 ≤ (1 + g) * E := by linarith
    calc r5 ≤ (1 + F.u) * (r2 + r4) := b5h
      _ ≤ (1 + F.u) * ((1 + g) * E) := mul_le_mul_of_nonneg_left this (by linarith)
      _ = _ := by ring
  obtain ⟨lr1, lr2⟩ := log_ratio (mul_pos hp1 hp2) (by linarith) r5lo r5hi
  rw [Real.log_mul hp1.ne' hp2.ne'] at lr1
  rw [Real.log_mul (by linarith) (by linarith)] at lr2
  have g1 := log_one_sub_ge hu0 hu1 (by norm_num)
  have g2 := log_one_sub_ge hg0 hg1 (by norm_num)
  have g3 := log_one_add_le hu0
  have g4 := log_one_add_le hg0
  obtain ⟨e5, he5⟩ : ∃ e : ℝ, e = F.u * (256 / 255) + g * (30 / 29) := ⟨_, rfl⟩
  have hl5 : |Real.log r5 - Real.log E| ≤ e5 := by
    rw [abs_le]; constructor
    · have e1 : F.u / (1 - 1 / 256) = F.u * (256 / 255) := by ring
      have e2 : g / (1 - 1 / 30) = g * (30 / 29) := by ring
      rw [e1] at g1; rw [e2] at g2; linarith
    · linarith
  have he50 : 0 ≤ e5 := le_trans (abs_nonneg _) hl5
  have hl6 := F.log_err h6
  have habs5 : |Real.log r5| ≤ 1 + e5 := by
    have := abs_sub_abs_le_abs_sub (Real.log r5) (Real.log E)
    rw [abs_of_nonneg hlogE0] at this; linarith
  obtain ⟨e6, he6⟩ : ∃ e : ℝ, e = e5 + F.ul * (1 + e5) := ⟨_, rfl⟩
  have hr6 : |r6 - Real.log E| ≤ e6 := by
    have := abs_sub_le r6 (Real.log r5) (Real.log E)
    have := mul_le_mul_of_nonneg_left habs5 hul0
    linarith
  have he60 : 0 ≤ e6 := le_trans (abs_nonneg _) hr6
  have hlip := lse2_lip ha hb
  have hl7 := F.add_err h7
  have habs7 : |v_m + r6| ≤ M + n - 1 + ε + e6 := by
    have k1 : |v_m + r6| ≤ |v_m + Real.log E| + |r6 - Real.log E| := by
      have := abs_sub_abs_le_abs_sub (v_m + r6) (v_m + Real.log E)
      have h' : v_m + r6 - (v_m + Real.log E) = r6 - Real.log E := by ring
      rw [h'] at this; linarith
    have k2 : |v_m + Real.log E| ≤ |Real.log (Real.exp a + Real.exp b)| + ε := by
      have := abs_sub_abs_le_abs_sub (v_m + Real.log E) (Real.log (Real.exp a + Real.exp b))
      rw [← hlE] at this ⊢; linarith
    linarith
  have hfin : |v_s - Real.log (Real.exp a + Real.exp b)| ≤
      F.u * (M + n - 1 + ε + e6) + e6 + ε := by
    have k : v_s - Real.log (Real.exp a + Real.exp b) = (v_s - (v_m + r6)) + (r6 - Real.log E)
        + (Real.log (Real.exp L.s + Real.exp R.s) - Real.log (Real.exp a + Real.exp b)) := by
      rw [hlE]; ring
    rw [k]
    have := abs_add_le ((v_s - (v_m + r6)) + (r6 - Real.log E))
        (Real.log (Real.exp L.s + Real.exp R.s) - Real.log (Real.exp a + Real.exp b))
    have := abs_add_le (v_s - (v_m + r6)) (r6 - Real.log E)
    have := mul_le_mul_of_nonneg_left habs7 hu0
    linarith
  refine le_trans hfin ?_
  have q1 : F.ul * e5 ≤ (1 / 64) * e5 := mul_le_mul_of_nonneg_right hul1 he50
  have q2 : F.u * e6 ≤ (1 / 256) * e6 := mul_le_mul_of_nonneg_right hu1 he60
  have q3 : F.u * ε ≤ F.u * (M / 5 + n / 5 + 1) := mul_le_mul_of_nonneg_left hε hu0
  have q4 : 0 ≤ F.u * M := mul_nonneg hu0 hM
  have q5 : 0 ≤ F.u * (n - 2) := mul_nonneg hu0 (by linarith)
  linarith

lemma flat_ne : ∀ t : Tree ℝ, (∀ b ∈ t.leaves, b ≠ []) → t.flat ≠ []
  | .leaf b, h => by simpa [Tree.flat, Tree.leaves] using h
  | .node l r, h => by
    simp only [Tree.flat]
    have := flat_ne l (fun b hb => h b (by simp [Tree.leaves, hb]))
    simp [this]

lemma eps_bound {u ue ul D K M n d T : ℝ} (hu0 : 0 ≤ u) (hue0 : 0 ≤ ue) (hul0 : 0 ≤ ul)
    (hu1 : u ≤ 1 / 256) (hul1 : ul ≤ 1 / 64) (hD0 : 0 ≤ D) (hM : 0 ≤ M) (hn : 0 ≤ n)
    (hd0 : 0 ≤ d) (hK : K ≤ 2 * M) (hd : d + 1 ≤ T) (hT : (T + D + 1) * (u + ue + ul) ≤ 1 / 16) :
    2 * (u * (D + K + M + n + 2) + ue + ul * (n + 1)) + 2 * d * (u * (M + n + 2) + ue + ul)
      ≤ M / 5 + n / 5 + 1 := by
  have hP : (d + 1 + D) * (u + ue + ul) ≤ 1 / 16 :=
    le_trans (mul_le_mul_of_nonneg_right (by linarith) (by positivity)) hT
  have k1 : (d + 1) * u ≤ 1 / 16 := by nlinarith
  have k2 : D * u ≤ 1 / 16 := by nlinarith
  have k3 : (d + 1) * ue ≤ 1 / 16 := by nlinarith
  have k4 : d * ul ≤ 1 / 16 := by nlinarith
  have k5 : ((d + 1) * u) * (M + n + 2) ≤ (1 / 16) * (M + n + 2) :=
    mul_le_mul_of_nonneg_right k1 (by linarith)
  have k6 : u * K ≤ u * (2 * M) := mul_le_mul_of_nonneg_left hK hu0
  have k7 : u * M ≤ (1 / 256) * M := mul_le_mul_of_nonneg_right hu1 hM
  have k8 : ul * (n + 1) ≤ (1 / 64) * (n + 1) := mul_le_mul_of_nonneg_right hul1 (by linarith)
  linarith

set_option maxHeartbeats 1000000 in
lemma main_ind (F : Rounding) (hs : small F) (D : ℕ) (lo hi n : ℝ) (T : ℕ)
    (hη : n * F.ηe ≤ F.u) (hηu : F.ηe ≤ F.u)
    (hT : ((T : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16) :
    ∀ (t : Tree ℝ) (w : State_a), Tree.FEval (fblock_a F D) (fmerge_a F) t w →
      (∀ b ∈ t.leaves, b ≠ []) → (∀ x ∈ t.flat, lo ≤ x ∧ x ≤ hi) →
      (t.flat.length : ℝ) ≤ n → t.depth ≤ T →
      |w.s - (block_a t.flat).s| ≤
        2 * (F.u * (D + (hi - lo) + max |lo| |hi| + n + 2) + F.ue + F.ul * (n + 1))
        + 2 * (t.depth : ℝ) * (F.u * (max |lo| |hi| + n + 2) + F.ue + F.ul) := by
  intro t w h
  induction h with
  | @leaf b w hb =>
    intro hl hf hn _
    have hb0 : b ≠ [] := hl b (by simp [Tree.leaves])
    have := block_step F hs D lo hi n b hb0 hf hn hη w hb
    simp only [Tree.flat, Tree.depth, Nat.cast_zero, mul_zero, zero_mul, add_zero]
    exact this
  | @node l r a b w hl hr hm ihl ihr =>
    intro hlv hf hn hd
    have hlv1 : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => hlv b (by simp [Tree.leaves, hb])
    have hlv2 : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => hlv b (by simp [Tree.leaves, hb])
    have hf1 : ∀ x ∈ l.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hf x (by simp [Tree.flat, hx])
    have hf2 : ∀ x ∈ r.flat, lo ≤ x ∧ x ≤ hi := fun x hx => hf x (by simp [Tree.flat, hx])
    have hne1 := flat_ne l hlv1
    have hne2 := flat_ne r hlv2
    have hlen : ((l.flat ++ r.flat).length : ℝ) = l.flat.length + r.flat.length := by
      simp
    have hn' : ((l.flat ++ r.flat).length : ℝ) ≤ n := hn
    have hl1 : (1 : ℝ) ≤ l.flat.length := by
      have := List.length_pos_of_ne_nil hne1; exact_mod_cast this
    have hl2 : (1 : ℝ) ≤ r.flat.length := by
      have := List.length_pos_of_ne_nil hne2; exact_mod_cast this
    have hd' : max l.depth r.depth + 1 ≤ T := hd
    have hdl : l.depth ≤ T := by omega
    have hdr : r.depth ≤ T := by omega
    have il := ihl hlv1 hf1 (by linarith) hdl
    have ir := ihr hlv2 hf2 (by linarith) hdr
    have hs' := hs
    obtain ⟨hu1, hue1, hul1, -, -⟩ := hs'
    have hu0 := F.hu; have hue0 := F.hue; have hul0 := F.hul
    set M := max |lo| |hi| with hMdef
    have hM0 : 0 ≤ M := le_trans (abs_nonneg _) (le_max_left _ _)
    have hn2 : 2 ≤ n := by linarith
    set c := F.u * (M + n + 2) + F.ue + F.ul with hc
    have hc0 : 0 ≤ c := by
      have := mul_nonneg hu0 (show 0 ≤ M + n + 2 by linarith); linarith
    set e0 := 2 * (F.u * (D + (hi - lo) + M + n + 2) + F.ue + F.ul * (n + 1)) with he0
    set d : ℝ := ((max l.depth r.depth : ℕ) : ℝ) with hd
    have hdl' : (l.depth : ℝ) ≤ d := by rw [hd]; exact_mod_cast le_max_left _ _
    have hdr' : (r.depth : ℝ) ≤ d := by rw [hd]; exact_mod_cast le_max_right _ _
    have il' : |a.s - (block_a l.flat).s| ≤ e0 + 2 * d * c := by
      have := mul_le_mul_of_nonneg_right hdl' hc0; linarith
    have ir' : |b.s - (block_a r.flat).s| ≤ e0 + 2 * d * c := by
      have := mul_le_mul_of_nonneg_right hdr' hc0; linarith
    have hd0 : 0 ≤ d := by rw [hd]; positivity
    have hdT : d + 1 ≤ (T : ℝ) := by rw [hd]; exact_mod_cast hd'
    have hK : hi - lo ≤ 2 * M := by
      have := neg_abs_le lo; have := le_abs_self hi
      have := le_max_left |lo| |hi|; have := le_max_right |lo| |hi|; linarith
    have heps := eps_bound hu0 hue0 hul0 hu1 hul1 (Nat.cast_nonneg D) hM0 (show (0:ℝ) ≤ n by linarith) hd0 hK hdT hT
    have hab := block_abs (List.append_ne_nil_of_left_ne_nil hne1 _) hf hn'
    rw [← merge_block_a _ _ hne1 hne2, merge_s] at hab
    have key := merge_step F hs hηu M n (e0 + 2 * d * c) _ _ a b w hn2 hM0
      (by rw [he0, hc]; linarith) il' ir' hab hm
    show |w.s - (block_a (l.flat ++ r.flat)).s| ≤ _
    rw [← merge_block_a _ _ hne1 hne2, merge_s]
    have hdepth : ((Tree.node l r).depth : ℝ) = d + 1 := by
      rw [hd]; simp [Tree.depth]
    rw [hdepth]
    have : e0 + 2 * d * c + 2 * c = e0 + 2 * (d + 1) * c := by ring
    linarith

theorem fp_bound : ∀ (F : Discovery.Rounding) (D : ℕ) (lo hi : ℝ) (X : List (ℝ)), X ≠ [] →
    (∀ x ∈ X, lo ≤ x ∧ x ≤ hi) → small F → (X.length : ℝ) * (F.η + F.ηe) ≤ F.u →
    ∀ (t : Discovery.Tree (ℝ)), t.flat = X → (∀ b ∈ t.leaves, b ≠ []) →
    ((t.depth : ℝ) + D + 1) * (F.u + F.ue + F.ul) ≤ 1 / 16 →
    ∀ (st : State_a), Discovery.Tree.FEval (fblock_a F D) (fmerge_a F) t st →
      |Real.log (znorm st) - Real.log ((X.map Real.exp).sum)| ≤ (((2 : ℝ) * ((F.u * ((D : ℝ) + (hi - lo) + (max |lo| |hi|) + (X.length : ℝ) + (2 : ℝ))) + F.ue + (F.ul * ((X.length : ℝ) + (1 : ℝ))))) + ((2 : ℝ) * (t.depth : ℝ) * ((F.u * ((max |lo| |hi|) + (X.length : ℝ) + (2 : ℝ))) + F.ue + F.ul))) := by
  intro F D lo hi X hX hb hs hn t ht hl hT st h
  have hn1 : (1 : ℝ) ≤ X.length := by
    have := List.length_pos_of_ne_nil hX; exact_mod_cast this
  have hη0 := F.hη
  have hηe0 := F.hηe
  have hnη : 0 ≤ (X.length : ℝ) * F.η := mul_nonneg (by linarith) hη0
  have hηe : (X.length : ℝ) * F.ηe ≤ F.u := by nlinarith
  have hηu : F.ηe ≤ F.u := by nlinarith
  have key := main_ind F hs D lo hi X.length t.depth hηe hηu hT t st h hl
    (by rw [ht]; exact hb) (by rw [ht]) le_rfl
  rw [ht, block_s_eq hX] at key
  have hz : Real.log (znorm st) = st.s := by
    rw [znorm, one_div, ← Real.exp_neg, Real.log_exp]; ring
  rw [hz]
  exact key

theorem fp_bound_holds : fp_bound_statement := fp_bound

#print axioms fp_bound_holds

end Discovery.FP.SoftmaxLse
