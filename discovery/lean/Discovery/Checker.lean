import Lean.Replay
import Lean.Util.CollectAxioms
import Lean.Util.Path
import Lean.Data.Json.Parser
import Lean.Data.Json.Printer

/-!
# The judge's checker

`discovery/prove.py` runs this file with `lean --run Checker.lean REQUEST.json` once a candidate's
proofs compile. Nothing here is elaborated next to the model's text: the request names two
compiled modules, which are read at run time with `importModules`'s machinery, extensions and
initializers off, so no attribute, elaborator or option of theirs takes effect here.

* `statements`: the generated definitions and the statement of every obligation, as a `Prop`
  (`X_statement`), compiled from the generated text alone, before any model text exists;
* `proofs`: the model's lemmas and proofs, the generated `blocked_eq_spec` and the checks
  `theorem X_holds : X_statement := X`, importing `statements`.

The checker

1. replays every declaration of `proofs` in the kernel, on top of its imports (what `leanchecker`
   does), so a declaration added without the kernel's check (`debug.skipKernelTC`, an environment
   hack) is found;
2. for every obligation `X`, checks that `X_holds` is a theorem of `proofs` whose type is exactly
   the proposition `X_statement` of `statements`, and collects the axioms of `X_holds` from the
   declarations' bodies (with the extensions off, no axiom list stored in a module is read).

In Lean 4.36 the kernel runs no compiled code: `native_decide` adds an axiom of its own
(`_native.native_decide.ax_N`), which the axioms show.

It prints one JSON object: `{"replay": null or the kernel's error,
"checks": {X: {"found", "statement", "type", "axioms"}}}`.
-/

open Lean

namespace Discovery.Checker

/-- A state monad over an environment, for `collectAxioms`. -/
abbrev EnvM := StateT Environment IO

instance : MonadEnv EnvM where
  getEnv := get
  modifyEnv := modify

/-- `∀ xs, s xs`: the type of a proof of the proposition `s : ∀ xs, Prop`, given the type of `s`
(its universe parameters already replaced by `us`). -/
def expectedType (s : Name) (us : List Level) : Expr → Nat → Expr
  | .forallE n d b bi, k => .forallE n d (expectedType s us b (k + 1)) bi
  | _, k => mkAppN (mkConst s us) ((List.range k).reverse.map mkBVar).toArray

/-- A field of the request. -/
def field (j : Json) (key : String) : Except String Json := j.getObjVal? key

def strings (j : Json) : Except String (List String) := do
  (← j.getArr?).toList.mapM Json.getStr?

/-- The verdict on one obligation `x` of the namespace `ns`. -/
def judge (base : Environment) (env : Environment) (kenv : Kernel.Environment)
    (statementsIdx : Option ModuleIdx) (own : Std.HashMap Name ConstantInfo) (ns : Name)
    (x : String) : IO Json := do
  let holds := Name.str ns (x ++ "_holds")
  let stmt := Name.str ns (x ++ "_statement")
  let found := own.contains holds
  let statementOk := statementsIdx.isSome && base.getModuleIdxFor? stmt == statementsIdx
  let typeOk := match kenv.find? holds, kenv.find? stmt with
    | some (.thmInfo h), some (.defnInfo s) =>
      let us := h.levelParams.map mkLevelParam
      h.levelParams.length == s.levelParams.length &&
        h.type == expectedType stmt us (s.type.instantiateLevelParams s.levelParams us) 0
    | _, _ => false
  let axioms ← if found then (collectAxioms holds : EnvM _).run' env else pure #[]
  return Json.mkObj [
    ("found", found), ("statement", statementOk), ("type", typeOk),
    ("axioms", Json.arr (axioms.map fun a => Json.str a.toString))]

unsafe def run (request : Json) : IO Json := do
  let statements := (← IO.ofExcept (field request "statements" >>= Json.getStr?)).toName
  let proofs := (← IO.ofExcept (field request "proofs" >>= Json.getStr?)).toName
  let ns := (← IO.ofExcept (field request "namespace" >>= strings)).foldl Name.str .anonymous
  let obligations ← IO.ofExcept (field request "obligations" >>= strings)
  initSearchPath (← findSysroot)
  -- the proofs module as compiled, and the environment of its imports
  let parts ← readModuleDataParts #[← findOLean proofs]
  let some (mod, _) := parts[parts.size - 1]? | throw <| IO.userError "no module data"
  let (_, s) ← importModulesCore mod.imports |>.run
  let base ← finalizeImport s mod.imports {} 0 false false (isModule := true)
  let mut own : Std.HashMap Name ConstantInfo := {}
  for name in mod.constNames, ci in mod.constants do
    own := own.insert name ci
  let kenv ← try base.toKernelEnv.replay own
    catch e => return Json.mkObj [("replay", Json.str e.toString), ("checks", Json.mkObj [])]
  let env := Environment.ofKernelEnv kenv
  let statementsIdx := base.getModuleIdx? statements
  let mut checks := #[]
  for x in obligations do
    checks := checks.push (x, ← judge base env kenv statementsIdx own ns x)
  return Json.mkObj [("replay", Json.null), ("checks", Json.mkObj checks.toList)]

end Discovery.Checker

unsafe def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: lean --run Checker.lean REQUEST.json"; return 2
  let request ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  IO.println (← Discovery.Checker.run request).compress
  return 0
