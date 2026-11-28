module

public import Mathlib.Basic.Real.Basic

/-!
# Blocked folds

The framework every candidate's proof stands on, written and proved once.

A pass of a candidate reduces the elements of a row block by block: `block` summarises one
nonempty block and `merge` combines the summaries of two adjacent blocks, left then right. If the
merge is right on any two nonempty blocks, `merge (block A) (block B) = block (A ++ B)`, then
combining the summaries of any blocking of `X` along any merge tree gives `block X`
(`Tree.eval_eq_block`), and so does the left-to-right fold a kernel runs (`foldl_eq_block`).
The proof of a candidate is then one such merge lemma per pass, and an equation between its
output and the specification's when every pass reads all of `X` as one block.

`lmax` and `lmin` are the IR's reductions `rmax` and `rmin`.
-/

@[expose] public section

namespace Discovery

/-- The largest element of a list; `0` on the empty list, which no block is. -/
noncomputable def lmax : List ℝ → ℝ
  | [] => 0
  | a :: as => as.foldl max a

/-- The smallest element of a list; `0` on the empty list, which no block is. -/
noncomputable def lmin : List ℝ → ℝ
  | [] => 0
  | a :: as => as.foldl min a

/-- A left fold of an associative operation can take its first argument out. -/
theorem foldl_assoc {f : ℝ → ℝ → ℝ} (hf : ∀ a b c, f (f a b) c = f a (f b c)) (a b : ℝ)
    (l : List ℝ) : l.foldl f (f a b) = f a (l.foldl f b) := by
  induction l generalizing b with
  | nil => rfl
  | cons c l ih => simp only [List.foldl_cons, hf, ih]

@[simp] theorem lmax_singleton (a : ℝ) : lmax [a] = a := rfl

@[simp] theorem lmin_singleton (a : ℝ) : lmin [a] = a := rfl

theorem lmax_cons (a : ℝ) {l : List ℝ} (hl : l ≠ []) : lmax (a :: l) = max a (lmax l) := by
  obtain ⟨b, l, rfl⟩ := List.exists_cons_of_ne_nil hl
  simp only [lmax, List.foldl_cons, foldl_assoc max_assoc]

theorem lmin_cons (a : ℝ) {l : List ℝ} (hl : l ≠ []) : lmin (a :: l) = min a (lmin l) := by
  obtain ⟨b, l, rfl⟩ := List.exists_cons_of_ne_nil hl
  simp only [lmin, List.foldl_cons, foldl_assoc min_assoc]

theorem lmax_append {A B : List ℝ} (hA : A ≠ []) (hB : B ≠ []) :
    lmax (A ++ B) = max (lmax A) (lmax B) := by
  obtain ⟨a, A, rfl⟩ := List.exists_cons_of_ne_nil hA
  obtain ⟨b, B, rfl⟩ := List.exists_cons_of_ne_nil hB
  simp only [lmax, List.cons_append, List.foldl_append, List.foldl_cons, foldl_assoc max_assoc]

theorem lmin_append {A B : List ℝ} (hA : A ≠ []) (hB : B ≠ []) :
    lmin (A ++ B) = min (lmin A) (lmin B) := by
  obtain ⟨a, A, rfl⟩ := List.exists_cons_of_ne_nil hA
  obtain ⟨b, B, rfl⟩ := List.exists_cons_of_ne_nil hB
  simp only [lmin, List.cons_append, List.foldl_append, List.foldl_cons, foldl_assoc min_assoc]

/-- `lmax_append` for the reduction of a function of the elements, the form `rmax` exports to. -/
theorem lmax_map_append {α : Type*} (f : α → ℝ) {A B : List α} (hA : A ≠ []) (hB : B ≠ []) :
    lmax ((A ++ B).map f) = max (lmax (A.map f)) (lmax (B.map f)) := by
  rw [List.map_append]
  exact lmax_append (by simpa using hA) (by simpa using hB)

/-- `lmin_append` for the reduction of a function of the elements, the form `rmin` exports to. -/
theorem lmin_map_append {α : Type*} (f : α → ℝ) {A B : List α} (hA : A ≠ []) (hB : B ≠ []) :
    lmin ((A ++ B).map f) = min (lmin (A.map f)) (lmin (B.map f)) := by
  rw [List.map_append]
  exact lmin_append (by simpa using hA) (by simpa using hB)

theorem le_lmax {a : ℝ} {l : List ℝ} (h : a ∈ l) : a ≤ lmax l := by
  induction l with
  | nil => simp at h
  | cons b l ih =>
    rcases eq_or_ne l [] with rfl | hl
    · simp_all
    rw [lmax_cons b hl]
    rcases List.mem_cons.mp h with rfl | h
    · exact le_max_left _ _
    · exact (ih h).trans (le_max_right _ _)

theorem lmin_le {a : ℝ} {l : List ℝ} (h : a ∈ l) : lmin l ≤ a := by
  induction l with
  | nil => simp at h
  | cons b l ih =>
    rcases eq_or_ne l [] with rfl | hl
    · simp_all
    rw [lmin_cons b hl]
    rcases List.mem_cons.mp h with rfl | h
    · exact min_le_left _ _
    · exact (min_le_right _ _).trans (ih h)

theorem lmax_mem {l : List ℝ} (hl : l ≠ []) : lmax l ∈ l := by
  induction l with
  | nil => exact absurd rfl hl
  | cons b l ih =>
    rcases eq_or_ne l [] with rfl | hl'
    · simp
    rw [lmax_cons b hl']
    rcases max_choice b (lmax l) with h | h <;> rw [h]
    · exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ (ih hl')

theorem lmin_mem {l : List ℝ} (hl : l ≠ []) : lmin l ∈ l := by
  induction l with
  | nil => exact absurd rfl hl
  | cons b l ih =>
    rcases eq_or_ne l [] with rfl | hl'
    · simp
    rw [lmin_cons b hl']
    rcases min_choice b (lmin l) with h | h <;> rw [h]
    · exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ (ih hl')

/-- A merge tree over a row: its leaves are the blocks, left to right; a node merges the
summaries of its two subtrees. -/
inductive Tree (E : Type*) where
  | leaf : List E → Tree E
  | node : Tree E → Tree E → Tree E

namespace Tree

variable {E S : Type*}

/-- The blocks, left to right. -/
def leaves : Tree E → List (List E)
  | leaf b => [b]
  | node l r => l.leaves ++ r.leaves

/-- The elements the tree covers, in order. -/
def flat : Tree E → List E
  | leaf b => b
  | node l r => l.flat ++ r.flat

/-- Summarise every leaf with `block` and combine the summaries along the tree with `merge`. -/
def eval (block : List E → S) (merge : S → S → S) : Tree E → S
  | leaf b => block b
  | node l r => merge (l.eval block merge) (r.eval block merge)

theorem flat_ne_nil : ∀ {t : Tree E}, (∀ b ∈ t.leaves, b ≠ []) → t.flat ≠ []
  | leaf b, h => h b (by simp [leaves])
  | node l _, h => by
    simp only [flat]
    exact List.append_ne_nil_of_left_ne_nil (flat_ne_nil fun b hb => h b (by simp [leaves, hb])) _

/-- The framework theorem: a merge that is right on two nonempty blocks is right on any blocking
and any merge tree. -/
theorem eval_eq_block {block : List E → S} {merge : S → S → S}
    (h : ∀ A B, A ≠ [] → B ≠ [] → merge (block A) (block B) = block (A ++ B)) :
    ∀ t : Tree E, (∀ b ∈ t.leaves, b ≠ []) → t.eval block merge = block t.flat
  | leaf _, _ => rfl
  | node l r, ht => by
    have hl : ∀ b ∈ l.leaves, b ≠ [] := fun b hb => ht b (by simp [leaves, hb])
    have hr : ∀ b ∈ r.leaves, b ≠ [] := fun b hb => ht b (by simp [leaves, hb])
    simp only [eval, flat, eval_eq_block h l hl, eval_eq_block h r hr]
    exact h _ _ (flat_ne_nil hl) (flat_ne_nil hr)

end Tree

/-- The left-to-right fold a kernel runs over its blocks `b :: bs`, as a corollary. -/
theorem foldl_eq_block {E S : Type*} {block : List E → S} {merge : S → S → S}
    (h : ∀ A B, A ≠ [] → B ≠ [] → merge (block A) (block B) = block (A ++ B)) :
    ∀ (b : List E) (bs : List (List E)), b ≠ [] → (∀ c ∈ bs, c ≠ []) →
      (bs.map block).foldl merge (block b) = block (b ++ bs.flatten)
  | b, [], _, _ => by simp
  | b, c :: cs, hb, hcs => by
    simp only [List.map_cons, List.foldl_cons, List.flatten_cons]
    rw [h b c hb (hcs c List.mem_cons_self),
      foldl_eq_block h (b ++ c) cs (List.append_ne_nil_of_left_ne_nil hb c)
        (fun d hd => hcs d (List.mem_cons_of_mem c hd)),
      List.append_assoc]

end Discovery

end
