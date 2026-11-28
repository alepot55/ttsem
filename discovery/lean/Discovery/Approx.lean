module

public import Discovery.Blocked
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Positivity
public import Mathlib.Tactic.Ring
public import Mathlib.Tactic.GCongr

/-!
# Floating point over blocked folds: the rounding model and the depth law

Written and proved once, like `Blocked.lean`.

**The rounding model.** A floating-point operation whose exact result is `x` returns some
`x * (1 + δ) + η` with `|δ| ≤ u` and `|η| ≤ ηmin`: `δ` is the relative rounding error of a normal
result, `η` the absolute error of a subnormal one. An addition or a subtraction has no `η` (a
subnormal sum is exact); `exp` and `log` have their own relative errors `ue` and `ul`, since a
library's `exp` is not correctly rounded; `max`, `min`, `neg` and `abs` are exact. Overflow is not
modelled: a theorem over this model speaks of the runs that stay finite. Everything is a
parameter (`Rounding`), and an evaluation is a relation, not a function: `FEval ... t v` holds for
every `v` that some admissible choice of the errors produces.

**The depth law.** For a closeness relation `C ε ŝ s` between a float summary `ŝ` and an exact
summary `s`: if every block's float summary is `ε₀`-close to the exact one and a merge takes an
`εa`-close and an `εb`-close pair to a `max εa εb + c`-close result, then any blocking and any
merge tree of depth `d` gives a summary `ε₀ + c * d`-close to the exact summary of the whole row
(`Tree.depth_law`, and `Tree.depth_law_nat` for counts of roundings). The sum of positive numbers
along any tree, the classical bound by the tree's height, is an instance (`sum_tree_bound`).

**Signs.** A rounded product keeps the sign of the exact one and a rounded `exp` is never
negative, as with round to nearest and any `exp` that returns a float of the right sign; without
these facts the model would allow a negative normalizer that no run produces.

**A block's sum.** `fsum` adds a block's terms along any tree of depth at most `D`;
`fsum_bounds` brackets a sum of nonnegative terms.

**Up to an equivalence.** `Tree.eval_rel_block`: a merge that respects a relation `R` and is right
up to `R` on two blocks is right up to `R` on any merge tree. A state with a stale reference (a
conditional rescaling) is right only up to the value it encodes.
-/

@[expose] public section

namespace Discovery

/-! ## The rounding model -/

/-- The parameters of a floating-point format and of its `exp` and `log`. In float32 with
round to nearest: `u = 2^-24`, `η = 2^-150`; `ue`, `ηe` and `ul` are those of the library in
use. -/
structure Rounding where
  /-- relative error of `+ - * / sqrt` -/
  u : ℝ
  /-- absolute error of `*` and `/` with a subnormal result -/
  η : ℝ
  /-- relative error of `exp` -/
  ue : ℝ
  /-- absolute error of `exp` with a subnormal result -/
  ηe : ℝ
  /-- relative error of `log` -/
  ul : ℝ
  hu : 0 ≤ u
  hu1 : u < 1
  hη : 0 ≤ η
  hue : 0 ≤ ue
  hue1 : ue < 1
  hηe : 0 ≤ ηe
  hul : 0 ≤ ul

namespace Rounding

variable (F : Rounding)

/-- `v` is a possible result of an addition or subtraction whose exact result is `x`. -/
def add (x v : ℝ) : Prop := ∃ δ, |δ| ≤ F.u ∧ v = x * (1 + δ)

/-- `v` is a possible result of a multiplication or division whose exact result is `x`; the
rounding keeps the sign (`0 ≤ x * v`), as round-to-nearest does. -/
def mul (x v : ℝ) : Prop := ∃ δ e, |δ| ≤ F.u ∧ |e| ≤ F.η ∧ v = x * (1 + δ) + e ∧ 0 ≤ x * v

/-- `v` is a possible result of `sqrt` at `x`. -/
def sqrt (x v : ℝ) : Prop := ∃ δ, |δ| ≤ F.u ∧ v = Real.sqrt x * (1 + δ)

/-- `v` is a possible result of `exp` at `x`, never negative. -/
def exp (x v : ℝ) : Prop :=
  ∃ δ e, |δ| ≤ F.ue ∧ |e| ≤ F.ηe ∧ v = Real.exp x * (1 + δ) + e ∧ 0 ≤ v

/-- `v` is a possible result of `log` at `x`. -/
def log (x v : ℝ) : Prop := ∃ δ, |δ| ≤ F.ul ∧ v = Real.log x * (1 + δ)

theorem add_bounds {x v : ℝ} (hx : 0 ≤ x) (h : F.add x v) :
    (1 - F.u) * x ≤ v ∧ v ≤ (1 + F.u) * x := by
  obtain ⟨δ, hδ, rfl⟩ := h
  have := abs_le.mp hδ
  constructor <;> nlinarith [this.1, this.2]

theorem add_err {x v : ℝ} (h : F.add x v) : |v - x| ≤ F.u * |x| := by
  obtain ⟨δ, hδ, rfl⟩ := h
  calc |x * (1 + δ) - x| = |δ| * |x| := by
        rw [show x * (1 + δ) - x = δ * x by ring, abs_mul]
    _ ≤ F.u * |x| := mul_le_mul_of_nonneg_right hδ (abs_nonneg x)

theorem mul_err {x v : ℝ} (h : F.mul x v) : |v - x| ≤ F.u * |x| + F.η := by
  obtain ⟨δ, e, hδ, he, rfl, -⟩ := h
  have h1 : |δ * x| ≤ F.u * |x| := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_right hδ (abs_nonneg x)
  have h2 := abs_le.mp h1
  have h3 := abs_le.mp he
  rw [show x * (1 + δ) + e - x = δ * x + e by ring, abs_le]
  constructor <;> linarith [h2.1, h2.2, h3.1, h3.2]

theorem mul_sign {x v : ℝ} (h : F.mul x v) : 0 ≤ x * v := by
  obtain ⟨_, _, _, _, _, h⟩ := h
  exact h

theorem exp_bounds {x v : ℝ} (h : F.exp x v) :
    Real.exp x * (1 - F.ue) - F.ηe ≤ v ∧ v ≤ Real.exp x * (1 + F.ue) + F.ηe ∧ 0 ≤ v := by
  obtain ⟨δ, e, hδ, he, rfl, hv⟩ := h
  have h1 := abs_le.mp hδ
  have h2 := abs_le.mp he
  have hx := Real.exp_pos x
  exact ⟨by nlinarith [h1.1, h2.1], by nlinarith [h1.2, h2.2], hv⟩

theorem log_err {x v : ℝ} (h : F.log x v) : |v - Real.log x| ≤ F.ul * |Real.log x| := by
  obtain ⟨δ, hδ, rfl⟩ := h
  calc |Real.log x * (1 + δ) - Real.log x| = |δ| * |Real.log x| := by
        rw [show Real.log x * (1 + δ) - Real.log x = δ * Real.log x by ring, abs_mul]
    _ ≤ F.ul * |Real.log x| := mul_le_mul_of_nonneg_right hδ (abs_nonneg _)

end Rounding

namespace Tree

variable {E S T : Type*}

/-- The depth of a merge tree: `0` for one block. -/
def depth : Tree E → ℕ
  | leaf _ => 0
  | node l r => max l.depth r.depth + 1

/-- Floating-point evaluation of a merge tree, as a relation: `FEval fblock fmerge t v` when `v`
is a possible result, `fblock A w` when `w` is a possible float summary of the block `A`, and
`fmerge a b w` when `w` is a possible float merge of `a` then `b`. -/
inductive FEval (fblock : List E → T → Prop) (fmerge : T → T → T → Prop) : Tree E → T → Prop
  | leaf {b : List E} {w : T} : fblock b w → FEval fblock fmerge (leaf b) w
  | node {l r : Tree E} {a c w : T} :
      FEval fblock fmerge l a → FEval fblock fmerge r c → fmerge a c w →
        FEval fblock fmerge (node l r) w

/-- The depth law, with a real closeness index. -/
theorem depth_law {block : List E → S} {merge : S → S → S}
    {fblock : List E → T → Prop} {fmerge : T → T → T → Prop}
    (C : ℝ → T → S → Prop) (ε₀ c : ℝ) (hc : 0 ≤ c)
    (hhom : ∀ A B, A ≠ [] → B ≠ [] → merge (block A) (block B) = block (A ++ B))
    (hblock : ∀ A w, A ≠ [] → fblock A w → C ε₀ w (block A))
    (hmerge : ∀ εa εb a b x y w, C εa x a → C εb y b → fmerge x y w →
      C (max εa εb + c) w (merge a b)) :
    ∀ (t : Tree E) (w : T), (∀ b ∈ t.leaves, b ≠ []) → FEval fblock fmerge t w →
      C (ε₀ + c * t.depth) w (block t.flat) := by
  intro t
  induction t with
  | leaf b =>
    intro w ht h
    cases h with
    | leaf hb =>
      simpa [depth, flat] using hblock b w (ht b (by simp [leaves])) hb
  | node l r ihl ihr =>
    intro w ht h
    cases h with
    | node hl hr hm =>
      have hl' : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [leaves, hb])
      have hr' : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => ht b (by simp [leaves, hb])
      have key := hmerge _ _ _ _ _ _ _ (ihl _ hl' hl) (ihr _ hr' hr) hm
      rw [hhom _ _ (flat_ne_nil hl') (flat_ne_nil hr')] at key
      have e : max (ε₀ + c * l.depth) (ε₀ + c * r.depth) + c =
          ε₀ + c * ((max l.depth r.depth + 1 : ℕ) : ℝ) := by
        rw [max_add_add_left, ← mul_max_of_nonneg _ _ hc]
        push_cast
        ring
      simpa [depth, flat, e] using key

/-- The depth law, with the closeness index a count of roundings. -/
theorem depth_law_nat {block : List E → S} {merge : S → S → S}
    {fblock : List E → T → Prop} {fmerge : T → T → T → Prop}
    (C : ℕ → T → S → Prop) (k₀ c : ℕ)
    (hhom : ∀ A B, A ≠ [] → B ≠ [] → merge (block A) (block B) = block (A ++ B))
    (hblock : ∀ A w, A ≠ [] → fblock A w → C k₀ w (block A))
    (hmerge : ∀ ka kb a b x y w, C ka x a → C kb y b → fmerge x y w →
      C (max ka kb + c) w (merge a b)) :
    ∀ (t : Tree E) (w : T), (∀ b ∈ t.leaves, b ≠ []) → FEval fblock fmerge t w →
      C (k₀ + c * t.depth) w (block t.flat) := by
  intro t
  induction t with
  | leaf b =>
    intro w ht h
    cases h with
    | leaf hb =>
      simpa [depth, flat] using hblock b w (ht b (by simp [leaves])) hb
  | node l r ihl ihr =>
    intro w ht h
    cases h with
    | node hl hr hm =>
      have hl' : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [leaves, hb])
      have hr' : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => ht b (by simp [leaves, hb])
      have key := hmerge _ _ _ _ _ _ _ (ihl _ hl' hl) (ihr _ hr' hr) hm
      rw [hhom _ _ (flat_ne_nil hl') (flat_ne_nil hr')] at key
      have e : max (k₀ + c * l.depth) (k₀ + c * r.depth) + c =
          k₀ + c * (max l.depth r.depth + 1) := by
        rw [max_add_add_left, ← mul_max_of_nonneg _ _ (Nat.zero_le c)]
        ring
      show C (k₀ + c * (max l.depth r.depth + 1)) w (block (l.flat ++ r.flat))
      rw [← e]
      exact key

/-- Up to an equivalence: a merge that respects `R` and is right up to `R` on two nonempty
blocks is right up to `R` on any merge tree. -/
theorem eval_rel_block {block : List E → S} {merge : S → S → S} (R : S → S → Prop)
    (htrans : ∀ a b c, R a b → R b c → R a c)
    (hcongr : ∀ a a' b b', R a a' → R b b' → R (merge a b) (merge a' b'))
    (hhom : ∀ A B, A ≠ [] → B ≠ [] → R (merge (block A) (block B)) (block (A ++ B)))
    (hrefl : ∀ a, R a a) :
    ∀ t : Tree E, (∀ b ∈ t.leaves, b ≠ []) → R (t.eval block merge) (block t.flat)
  | leaf b, _ => hrefl (block b)
  | node l r, ht => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [leaves, hb])
    have hr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => ht b (by simp [leaves, hb])
    exact htrans _ _ _ (hcongr _ _ _ _ (eval_rel_block R htrans hcongr hhom hrefl l hl)
      (eval_rel_block R htrans hcongr hhom hrefl r hr)) (hhom _ _ (flat_ne_nil hl) (flat_ne_nil hr))

/-- An output that respects `R` reads the same from any merge tree. -/
theorem output_eq_of_rel {block : List E → S} {merge : S → S → S} {O : Type*} (R : S → S → Prop)
    (out : S → O) (hout : ∀ a b, R a b → out a = out b)
    (htrans : ∀ a b c, R a b → R b c → R a c)
    (hcongr : ∀ a a' b b', R a a' → R b b' → R (merge a b) (merge a' b'))
    (hhom : ∀ A B, A ≠ [] → B ≠ [] → R (merge (block A) (block B)) (block (A ++ B)))
    (hrefl : ∀ a, R a a) (t : Tree E) (ht : ∀ b ∈ t.leaves, b ≠ []) :
    out (t.eval block merge) = out (block t.flat) :=
  hout _ _ (eval_rel_block R htrans hcongr hhom hrefl t ht)

end Tree

/-! ## The sum of positive numbers along any tree -/

/-- `w` is the sum of a block of positive numbers, a leaf of the summation tree. -/
def posBlock (A : List ℝ) (w : ℝ) : Prop := (∀ x ∈ A, 0 < x) ∧ w = A.sum

/-- The classical bound by the height of the tree, as an instance of the depth law: summing
positive numbers along any tree of depth `d` (one rounded addition per node, exact leaves) gives
a result within a factor `(1 ± u)^d` of the exact sum. -/
theorem sum_tree_bound (F : Rounding) :
    ∀ (t : Tree ℝ) (w : ℝ), (∀ b ∈ t.leaves, b ≠ []) →
      Tree.FEval posBlock (fun a c w => F.add (a + c) w) t w →
      0 < t.flat.sum ∧ (1 - F.u) ^ t.depth * t.flat.sum ≤ w ∧
        w ≤ (1 + F.u) ^ t.depth * t.flat.sum := by
  intro t w ht h
  have hu := F.hu
  have hu1 := F.hu1
  have law := Tree.depth_law_nat (block := List.sum) (merge := (· + ·))
    (fblock := posBlock) (fmerge := fun a c w => F.add (a + c) w)
    (fun k w s => 0 < s ∧ (1 - F.u) ^ k * s ≤ w ∧ w ≤ (1 + F.u) ^ k * s) 0 1
    (fun A B _ _ => (List.sum_append).symm)
    (by
      intro A w hA hw
      obtain ⟨hpos, rfl⟩ := hw
      obtain ⟨a, A', rfl⟩ := List.exists_cons_of_ne_nil hA
      have : 0 < (a :: A').sum := by
        rw [List.sum_cons]
        have ha : 0 < a := hpos a List.mem_cons_self
        have : 0 ≤ A'.sum := List.sum_nonneg fun x hx => (hpos x (List.mem_cons_of_mem a hx)).le
        linarith
      simpa using this)
    (by
      intro ka kb a b x y w ⟨ha, hxa1, hxa2⟩ ⟨hb, hyb1, hyb2⟩ hw
      refine ⟨by linarith, ?_⟩
      set k := max ka kb
      have l1 : (1 - F.u) ^ k ≤ (1 - F.u) ^ ka :=
        pow_le_pow_of_le_one (by linarith) (by linarith) (le_max_left _ _)
      have l2 : (1 - F.u) ^ k ≤ (1 - F.u) ^ kb :=
        pow_le_pow_of_le_one (by linarith) (by linarith) (le_max_right _ _)
      have u1 : (1 + F.u) ^ ka ≤ (1 + F.u) ^ k := pow_le_pow_right₀ (by linarith) (le_max_left _ _)
      have u2 : (1 + F.u) ^ kb ≤ (1 + F.u) ^ k := pow_le_pow_right₀ (by linarith) (le_max_right _ _)
      have lo : (1 - F.u) ^ k * (a + b) ≤ x + y := by nlinarith
      have hi : x + y ≤ (1 + F.u) ^ k * (a + b) := by nlinarith
      have hxy : 0 ≤ x + y := by
        have := pow_nonneg (show (0 : ℝ) ≤ 1 - F.u by linarith) k
        nlinarith
      obtain ⟨hw1, hw2⟩ := F.add_bounds hxy hw
      have p1 : 0 ≤ 1 - F.u := by linarith
      constructor
      · calc (1 - F.u) ^ (k + 1) * (a + b) = (1 - F.u) * ((1 - F.u) ^ k * (a + b)) := by ring
          _ ≤ (1 - F.u) * (x + y) := mul_le_mul_of_nonneg_left lo p1
          _ ≤ w := hw1
      · calc w ≤ (1 + F.u) * (x + y) := hw2
          _ ≤ (1 + F.u) * ((1 + F.u) ^ k * (a + b)) := mul_le_mul_of_nonneg_left hi (by linarith)
          _ = (1 + F.u) ^ (k + 1) * (a + b) := by ring)
    t w ht h
  simpa using law

/-- The same, as a relative error: `|w - Σ| ≤ ((1 + u)^d - 1) Σ`. -/
theorem sum_tree_error (F : Rounding) (t : Tree ℝ) (w : ℝ) (ht : ∀ b ∈ t.leaves, b ≠ [])
    (h : Tree.FEval posBlock (fun a c w => F.add (a + c) w) t w) :
    |w - t.flat.sum| ≤ ((1 + F.u) ^ t.depth - 1) * t.flat.sum := by
  obtain ⟨hs, lo, hi⟩ := sum_tree_bound F t w ht h
  have hu := F.hu
  have hu1 := F.hu1
  have key : ∀ n : ℕ, 2 ≤ (1 + F.u) ^ n + (1 - F.u) ^ n := by
    intro n
    induction n with
    | zero => norm_num
    | succ n ih =>
      have hab : (1 - F.u) ^ n ≤ (1 + F.u) ^ n := pow_le_pow_left₀ (by linarith) (by linarith) n
      rw [pow_succ, pow_succ]
      nlinarith
  have k2 := key t.depth
  rw [abs_le]
  constructor <;> nlinarith

/-! ## A block's sum in floating point -/

/-- `w` is a possible float sum over the block `A`: the elements of `A`, in order, are the leaves
of a binary tree of depth at most `D`, each leaf is any `y` with `term x y` for its element `x`,
and each node is one rounded addition. -/
def fsum {E : Type*} (F : Rounding) (D : ℕ) (term : E → ℝ → Prop) (A : List E) (w : ℝ) : Prop :=
  ∃ t : Tree E, t.flat = A ∧ (∀ b ∈ t.leaves, ∃ x, b = [x]) ∧ t.depth ≤ D ∧
    Tree.FEval (fun b y => ∃ x, b = [x] ∧ term x y) (fun a c y => F.add (a + c) y) t w

/-- A float sum of nonnegative terms, each between `lo x` and `hi x`, lies between
`(1 - u)^D Σ lo` and `(1 + u)^D Σ hi`. -/
theorem fsum_bounds {E : Type*} (F : Rounding) (D : ℕ) (term : E → ℝ → Prop) (lo hi : E → ℝ)
    (hlo : ∀ x, 0 ≤ lo x) (hterm : ∀ x y, term x y → lo x ≤ y ∧ y ≤ hi x)
    (A : List E) (w : ℝ) (h : fsum F D term A w) :
    (1 - F.u) ^ D * (A.map lo).sum ≤ w ∧ w ≤ (1 + F.u) ^ D * (A.map hi).sum := by
  obtain ⟨t, hflat, hleaves, hdepth, hev⟩ := h
  have hu := F.hu
  have hu1 := F.hu1
  have law := Tree.depth_law_nat (E := E) (S := ℝ × ℝ) (T := ℝ)
    (block := fun A => ((A.map lo).sum, (A.map hi).sum))
    (merge := fun a b => (a.1 + b.1, a.2 + b.2))
    (fblock := fun b y => ∃ x, b = [x] ∧ term x y)
    (fmerge := fun a c y => F.add (a + c) y)
    (fun k w s => 0 ≤ s.1 ∧ 0 ≤ s.2 ∧ (1 - F.u) ^ k * s.1 ≤ w ∧ w ≤ (1 + F.u) ^ k * s.2) 0 1
    (fun A B _ _ => by simp [List.map_append, List.sum_append])
    (by
      rintro A y - ⟨x, rfl, hxy⟩
      obtain ⟨h1, h2⟩ := hterm x y hxy
      have := hlo x
      show 0 ≤ lo x + 0 ∧ 0 ≤ hi x + 0 ∧ (1 - F.u) ^ 0 * (lo x + 0) ≤ y ∧
        y ≤ (1 + F.u) ^ 0 * (hi x + 0)
      simp only [add_zero, pow_zero, one_mul]
      exact ⟨this, by linarith, h1, h2⟩)
    (by
      rintro ka kb a b x y w ⟨ha1, ha2, hxa1, hxa2⟩ ⟨hb1, hb2, hyb1, hyb2⟩ hw
      show 0 ≤ a.1 + b.1 ∧ 0 ≤ a.2 + b.2 ∧ (1 - F.u) ^ (max ka kb + 1) * (a.1 + b.1) ≤ w ∧
        w ≤ (1 + F.u) ^ (max ka kb + 1) * (a.2 + b.2)
      refine ⟨by linarith, by linarith, ?_⟩
      set k := max ka kb
      have l1 : (1 - F.u) ^ k ≤ (1 - F.u) ^ ka :=
        pow_le_pow_of_le_one (by linarith) (by linarith) (le_max_left _ _)
      have l2 : (1 - F.u) ^ k ≤ (1 - F.u) ^ kb :=
        pow_le_pow_of_le_one (by linarith) (by linarith) (le_max_right _ _)
      have u1 : (1 + F.u) ^ ka ≤ (1 + F.u) ^ k := pow_le_pow_right₀ (by linarith) (le_max_left _ _)
      have u2 : (1 + F.u) ^ kb ≤ (1 + F.u) ^ k := pow_le_pow_right₀ (by linarith) (le_max_right _ _)
      have p0 : 0 ≤ (1 - F.u) ^ k := pow_nonneg (by linarith) k
      have lo' : (1 - F.u) ^ k * (a.1 + b.1) ≤ x + y := by nlinarith
      have hi' : x + y ≤ (1 + F.u) ^ k * (a.2 + b.2) := by nlinarith
      have hxy : 0 ≤ x + y := le_trans (by positivity) lo'
      obtain ⟨hw1, hw2⟩ := F.add_bounds hxy hw
      have p1 : 0 ≤ 1 - F.u := by linarith
      constructor
      · calc (1 - F.u) ^ (k + 1) * (a.1 + b.1) = (1 - F.u) * ((1 - F.u) ^ k * (a.1 + b.1)) := by
              ring
          _ ≤ (1 - F.u) * (x + y) := mul_le_mul_of_nonneg_left lo' p1
          _ ≤ w := hw1
      · calc w ≤ (1 + F.u) * (x + y) := hw2
          _ ≤ (1 + F.u) * ((1 + F.u) ^ k * (a.2 + b.2)) :=
            mul_le_mul_of_nonneg_left hi' (by linarith)
          _ = (1 + F.u) ^ (k + 1) * (a.2 + b.2) := by ring)
    t w (fun b hb => by obtain ⟨x, rfl⟩ := hleaves b hb; simp) hev
  rw [hflat] at law
  obtain ⟨hs1, hs2, lo', hi'⟩ := law
  have e1 : (1 - F.u) ^ D ≤ (1 - F.u) ^ t.depth :=
    pow_le_pow_of_le_one (by linarith) (by linarith) hdepth
  have e2 : (1 + F.u) ^ t.depth ≤ (1 + F.u) ^ D := pow_le_pow_right₀ (by linarith) hdepth
  have lo'' : (1 - F.u) ^ t.depth * (A.map lo).sum ≤ w := by simpa using lo'
  have hi'' : w ≤ (1 + F.u) ^ t.depth * (A.map hi).sum := by simpa using hi'
  have hs1' : 0 ≤ (A.map lo).sum := hs1
  have hs2' : 0 ≤ (A.map hi).sum := hs2
  exact ⟨le_trans (mul_le_mul_of_nonneg_right e1 hs1') lo'',
    le_trans hi'' (mul_le_mul_of_nonneg_right e2 hs2')⟩

end Discovery

end
