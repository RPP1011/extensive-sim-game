# Task 5.5d Patch Specification

**Repo HEAD:** `b50a6744`
**Scope (IN):** Bare `self`/`target` resolution as standalone expressions (call-arg shape), Let-bound local reads.
**Scope (DEFERRED):** `IrExpr::Match`, `IrExpr::Quantifier`, `IrExpr::Fold`, `IrExpr::PerUnit`, `IrStmt::For`. Surfaced as typed errors with documented rationale.

This spec is intended for an **apply** agent. Each section below names a file, the exact insertion / replacement points (with line numbers from HEAD `b50a6744`), and the new code. Sections are ordered so that earlier edits compile before later edits land (the new `CgExpr` variants appear first, then walkers ripple, then lowering arms route into them).

---

## Cross-patch interactions (read first)

1. **Task 5.5b already added** `LoweringCtx::local_ids: HashMap<LocalRef, LocalId>` and `LoweringCtx::allocate_local()` (see `crates/dsl_compiler/src/cg/lower/expr.rs:106` and `:224`). This spec **only reads** from `local_ids` — it does not extend the `LoweringCtx` struct. The Let-bound local read path looks up `ast_ref` in the existing map.
2. **Task 5.5b already added** `CgStmt::Let { local, value, ty }` (see `crates/dsl_compiler/src/cg/stmt.rs:262-266`) and the WGSL emit for it (`crates/dsl_compiler/src/cg/emit/wgsl_body.rs:589-602`). This spec adds the **read side** (`CgExpr::ReadLocal`); it does not change `CgStmt::Let`.
3. **Task 5.5a already added** `LoweringCtx::target_local: bool` (see `crates/dsl_compiler/src/cg/lower/expr.rs:139`) and the `IrExpr::Local(_, "target")` resolution **inside `lower_field`** (only as `target.<field>`, not bare). This spec extends the bare-form to leverage the same flag.
4. **Task 5.5c (in flight)** may extend `LoweringCtx` with additional binding flags (per-pair scoring rows / fold-body event binders). This spec only **reads** `target_local` from the existing field. If 5.5c lands first, no merge action is required from this spec. If 5.5d lands first, 5.5c only needs to set `target_local = true` (and possibly any new flags it adds) before lowering scoring / fold bodies — the resolution arm here will pick up the per-pair candidate emission via the existing flag.
5. **Future task** will close the Match/Quantifier/Fold/PerUnit/For shapes. The typed-error funnels added by this spec (some already exist) make those follow-ups additive.

---

## Section A — Add `CgExpr::AgentSelfId` and `CgExpr::PerPairCandidateId` variants

**File:** `crates/dsl_compiler/src/cg/expr.rs`

### A.1 — Extend `CgExpr` (line 552–587)

Append two new variants to the `CgExpr` enum **after** the existing `Select` variant. Result: the enum gains two zero-payload variants whose result type is pinned (`AgentId` for both), so they need no `ty:` field.

Doc text for both variants:

- `AgentSelfId` — "The current dispatch's actor as an `AgentId` value. Surfaces in the surface DSL as a bare `self` reference (e.g., `agents.alive(self)`, `target != self`). Resolved by lowering `IrExpr::Local(_, \"self\")` when no `.field` access is present (the `self.<field>` path goes through `CgExpr::Read(AgentField { target: AgentRef::Self_, ... })`). Always typed `CgTy::AgentId`. Distinct from `Read(AgentField { target: AgentRef::Self_, ... })`: this variant carries the actor's *id*, not the read of any field."
- `PerPairCandidateId` — "The per-pair candidate's `AgentId` value. Surfaces in pair-bound dispatch contexts (today only `mask <Name>(target) from query.nearby_agents(...)` predicates) where bare `target` appears (`agents.alive(target)`, `target != self`). Lowering only constructs this variant when `LoweringCtx::target_local` is `true`; outside pair-bound contexts the bare `target` reference surfaces as `LoweringError::UnsupportedLocalBinding`. Mirrors `AgentRef::PerPairCandidate`'s contract: the candidate's slot id is implicit in the surrounding `DispatchShape::PerPair { source }`; the IR layer just tags the read."

Both variants are unit-shape `=>` `CgTy::AgentId`. They are **not** `Read` variants (no `DataHandle` payload) — they are leaf scalar values that the emit layer resolves to a kernel-local identifier (`agent_id` for `AgentSelfId`; `per_pair_candidate` for `PerPairCandidateId`).

### A.2 — Extend `CgExpr::ty()` (line 696–706)

Add two arms:

```
CgExpr::AgentSelfId => CgTy::AgentId,
CgExpr::PerPairCandidateId => CgTy::AgentId,
```

### A.3 — Extend `pretty_into` (line 764–805)

Add two arms emitting deterministic s-expression tokens:

```
CgExpr::AgentSelfId => write!(out, "(agent self_id)"),
CgExpr::PerPairCandidateId => write!(out, "(agent per_pair_candidate_id)"),
```

### A.4 — Extend `type_check` (line 1007–1166)

Add two arms before the closing `}`:

```
CgExpr::AgentSelfId => Ok(CgTy::AgentId),
CgExpr::PerPairCandidateId => Ok(CgTy::AgentId),
```

(Neither carries a `ty:` field, so there's no claimed-type validation; nothing else to check.)

### A.5 — Tests (append to `mod tests` at end of file)

Add three small unit tests:

1. `agent_self_id_ty_is_agent_id` — construct `CgExpr::AgentSelfId`, assert `.ty() == CgTy::AgentId`, assert `pretty(...)` returns `"(agent self_id)"`, assert serde round-trip.
2. `per_pair_candidate_id_ty_is_agent_id` — same shape for `CgExpr::PerPairCandidateId`, expected pretty `"(agent per_pair_candidate_id)"`.
3. `type_check_passes_for_self_id_and_pair_candidate_id` — push each into a fresh `Vec<CgExpr>`, run `type_check` against `TypeCheckCtx::new(&arena)`, assert `Ok(CgTy::AgentId)`.

---

## Section B — Add `CgExpr::ReadLocal { local, ty }` variant

**File:** `crates/dsl_compiler/src/cg/expr.rs`

### B.1 — Extend `CgExpr` (right after the variants added in §A)

Append a new variant:

```
/// Read a let-bound local. `local` resolves through the surrounding op's
/// body — either an `IrStmt::Let` lowered to `CgStmt::Let { local, value,
/// ty }` (Task 5.5b), or a future match-arm binding once those wire in.
/// `ty` mirrors the binding's declared CG type so the type checker has
/// the result type without needing to walk the binder.
///
/// Lowering only constructs this variant when `IrExpr::Local(_, name)`
/// resolves through `LoweringCtx::local_ids` (the typed `LocalRef →
/// LocalId` registry). The read carries no `LocalRef` — once the
/// lowering binds, only the typed `LocalId` flows in the IR.
ReadLocal {
    local: LocalId,
    ty: CgTy,
},
```

Add `use crate::cg::stmt::LocalId;` to the file's import block (currently `expr.rs` does not import `LocalId`; check line ~33). The import line becomes:

```
use super::data_handle::{CgExprId, DataHandle, RngPurpose, ViewId};
use super::stmt::LocalId;
```

Note: `super::stmt` is `crate::cg::stmt`. If a circular-import problem surfaces (`stmt.rs` imports from `expr.rs`), make `LocalId` re-exported from a shared location or import via the long path. **Verify no circularity** — `stmt.rs` already imports `CgExpr`-adjacent types: see `cg/stmt.rs` `use super::expr::CgTy;` etc. The reverse import (`expr.rs` → `stmt::LocalId`) introduces a cycle. **Mitigation:** declare `LocalId` in `expr.rs` and re-export from `stmt.rs`, OR keep `LocalId` in `stmt.rs` and use `crate::cg::stmt::LocalId` qualified at every read site in `expr.rs` (no `use` line). **Pick the qualified-path approach** — it avoids touching `stmt.rs` and surfaces every `LocalId` reference site as load-bearing.

So: do NOT add an import. Use `crate::cg::stmt::LocalId` in the variant definition itself:

```
ReadLocal {
    local: crate::cg::stmt::LocalId,
    ty: CgTy,
},
```

### B.2 — Extend `CgExpr::ty()` (line 696–706)

Add one arm:

```
CgExpr::ReadLocal { ty, .. } => *ty,
```

### B.3 — Extend `pretty_into` (line 764–805)

Add one arm:

```
CgExpr::ReadLocal { local, ty } => write!(out, "(read_local {} {})", local, ty),
```

(Renders e.g. `(read_local local#3 f32)`. `LocalId`'s `Display` produces `local#N` already, see `cg/stmt.rs:115`.)

### B.4 — Extend `type_check` (line 1007–1166)

Add one arm:

```
CgExpr::ReadLocal { ty, .. } => Ok(*ty),
```

The type checker takes the binding's declared type at face value. There is no per-arena cross-check that the matching `CgStmt::Let` exists — that's the well-formed pass's job (a future task; today the well-formed pass does not validate `ReadLocal` against `Let` either, see §F).

### B.5 — Tests (append to `mod tests`)

Add two tests:

1. `read_local_ty_pinned_from_field` — construct `CgExpr::ReadLocal { local: LocalId(3), ty: CgTy::F32 }`, assert `.ty() == CgTy::F32`, pretty == `"(read_local local#3 f32)"`, serde round-trip.
2. `type_check_passes_for_read_local` — push into arena, `type_check` returns `Ok(CgTy::F32)`.

---

## Section C — Phase-1 ripples for the three new variants

Every site that walks `CgExpr` exhaustively must gain matching arms. The three new variants are leaves (`AgentSelfId`, `PerPairCandidateId`, `ReadLocal`) — none has child `CgExprId`s, so all walker arms are no-ops or fall through identically to `Lit` / `Rng`.

### C.1 — `crates/dsl_compiler/src/cg/program.rs` — `validate_expr_refs` (line 757–781)

Extend the leaf-arm:

```
CgExpr::Read(_)
    | CgExpr::Lit(_)
    | CgExpr::Rng { .. }
    | CgExpr::AgentSelfId
    | CgExpr::PerPairCandidateId
    | CgExpr::ReadLocal { .. } => Ok(()),
```

(All three new variants carry no embedded `CgExprId`, so no validation is needed.)

### C.2 — `crates/dsl_compiler/src/cg/well_formed.rs` — `collect_subexpr_ids` (line 480–530)

Extend the no-op arm at line 527:

```
CgExpr::Lit(_)
    | CgExpr::Rng { .. }
    | CgExpr::AgentSelfId
    | CgExpr::PerPairCandidateId
    | CgExpr::ReadLocal { .. } => {}
```

### C.3 — `crates/dsl_compiler/src/cg/stmt.rs` — `collect_expr_reads` (line 406–442)

Extend with three new arms (each a no-op — none reads a `DataHandle`; the `ReadLocal` variant references a binding scoped to the body, not a persisted handle):

```
CgExpr::AgentSelfId | CgExpr::PerPairCandidateId | CgExpr::ReadLocal { .. } => {
    // Bare actor / candidate id reads + let-bound local reads do not
    // contribute structural reads of any persisted DataHandle. The
    // actor / candidate slot ids are implicit in the dispatch shape;
    // the let-bound local lives in the surrounding body's scope.
}
```

Place the arm after the `CgExpr::Select { .. }` arm and before the closing `}` of the `match`.

### C.4 — `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` — `lower_cg_expr_to_wgsl` (line 461–507)

Extend the match with three new arms, mapping each to its WGSL identifier:

```
CgExpr::AgentSelfId => Ok("agent_id".to_string()),
CgExpr::PerPairCandidateId => Ok("per_pair_candidate".to_string()),
CgExpr::ReadLocal { local, ty: _ } => Ok(format!("local_{}", local.0)),
```

Rationale:

- `AgentSelfId` emits `agent_id` — matches the kernel-local name the existing `CgExpr::Rng` arm uses (line 489–491 references `agent_id` already), so dependency on a separate naming strategy is unnecessary.
- `PerPairCandidateId` emits `per_pair_candidate` — matches `agent_ref_token(AgentRef::PerPairCandidate)` (line 191).
- `ReadLocal { local }` emits `local_<N>` — matches the `let local_<N>: <ty> = ...;` form already produced by `CgStmt::Let` emission (line 597).

The `ty` field on `ReadLocal` is unused at this WGSL site (the binding's declaration carries it); we still match `ty: _` to make the destructure exhaustive for future use.

Add a comment above each arm noting the placeholder shape — the kernel local naming will be made consistent at Task 4.x consolidation.

---

## Section D — Lower bare `IrExpr::Local(_, "self" | "target")`

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`

### D.1 — Replace the bare `IrExpr::Local` arm (line 332–335)

Today this arm rejects every bare `Local`:

```
IrExpr::Local(_, name) => Err(LoweringError::UnsupportedLocalBinding {
    name: name.clone(),
    span,
}),
```

Replace with a typed dispatch on the local name (and, for `target`, the `target_local` flag), with **let-bound locals taking priority** when the AST `LocalRef` resolves through `ctx.local_ids`:

```
IrExpr::Local(local_ref, name) => lower_bare_local(*local_ref, name, span, ctx),
```

Then add a new helper `fn lower_bare_local(...)` immediately after `lower_field` (insertion site: between current line 617 and the `lower_binary` `fn` at line 622).

### D.2 — Add `fn lower_bare_local`

```
/// Lower a bare `IrExpr::Local(local_ref, name)` (no `.field` access) to
/// a CG expression.
///
/// Resolution order:
///
/// 1. **Let-bound local.** If `ctx.local_ids` has an entry for
///    `local_ref`, the local was introduced by an enclosing `IrStmt::Let`
///    (lowered to `CgStmt::Let { local, value, ty }` in Task 5.5b). The
///    expression resolves to `CgExpr::ReadLocal { local, ty }` where
///    `ty` is the binding's declared CG type. **The map's value is the
///    `LocalId` only; the binding's `ty` is not stored on the context.**
///    To recover `ty`, walk the surrounding op's body for the
///    `CgStmt::Let { local, ty, .. }` whose `local` matches. Today the
///    builder does not expose that walk; pragmatic resolution: surface
///    `LoweringError::UnsupportedAstNode { ast_label: "Local(let-bound,
///    type-unresolved)", span }` if the type cannot be recovered, and
///    extend `LoweringCtx` in a follow-up task to thread a parallel
///    `local_tys: HashMap<LocalId, CgTy>` map.
///
///    **However:** since pattern-bound locals (Task 5.5b's match arms)
///    do not record their CgTy on the context either, the cleanest
///    expedient is to add a sibling field `local_tys: HashMap<LocalId,
///    CgTy>` to `LoweringCtx` in this spec — see §D.3 below.
///
/// 2. **Bare `self`.** Resolves to `CgExpr::AgentSelfId` (typed
///    `AgentId`). Used in surface DSL like `agents.alive(self)` and
///    `target != self`.
///
/// 3. **Bare `target` in a pair-bound context.** Resolves to
///    `CgExpr::PerPairCandidateId` (typed `AgentId`) when
///    `ctx.target_local` is `true`. Outside pair-bound contexts, falls
///    through to the default error below — same shape as the existing
///    `target.<field>` rejection.
///
/// 4. **Anything else** — surfaces as the existing
///    `LoweringError::UnsupportedLocalBinding`.
fn lower_bare_local(
    local_ref: LocalRef,
    name: &str,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    // Step 1: let-bound local.
    if let Some(&local_id) = ctx.local_ids.get(&local_ref) {
        let ty = ctx.local_tys.get(&local_id).copied().ok_or_else(|| {
            LoweringError::UnknownLocalType {
                local: local_id,
                span,
            }
        })?;
        return add(
            ctx,
            CgExpr::ReadLocal { local: local_id, ty },
            span,
        );
    }

    // Step 2-3: structural locals.
    match name {
        "self" => add(ctx, CgExpr::AgentSelfId, span),
        "target" if ctx.target_local => add(ctx, CgExpr::PerPairCandidateId, span),
        _ => Err(LoweringError::UnsupportedLocalBinding {
            name: name.to_string(),
            span,
        }),
    }
}
```

### D.3 — Extend `LoweringCtx` with `local_tys`

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`

The Let lowering (`crates/dsl_compiler/src/cg/lower/physics.rs:553–598`) computes the bound expression's `CgTy` to populate `CgStmt::Let { ty, .. }`. To make the read side resolve, the lowering must record `LocalId → CgTy` on the context.

**Add a field to `LoweringCtx`** at line 139 (after `target_local`):

```
/// Typed `LocalId → CgTy` map. Populated by `IrStmt::Let` lowering
/// (Task 5.5b) and match-arm pattern binding lowering at the moment
/// each binding's CG type becomes known. Used by `IrExpr::Local`
/// resolution (Task 5.5d) to reconstruct `CgExpr::ReadLocal { local,
/// ty }` for bare-local reads.
///
/// Distinct from [`Self::local_ids`]: that map carries `LocalRef →
/// LocalId` (binder identity), this one carries `LocalId → CgTy`
/// (binder type). They could be merged but the two-step lookup keeps
/// the binding-id allocation independent of the type.
pub local_tys: HashMap<LocalId, CgTy>,
```

**Initialize it in `LoweringCtx::new`** (line 150–162) — add `local_tys: HashMap::new(),` to the literal.

**Extend `register_local` (line 213) and `allocate_local` (line 224)** — these only populate `local_ids`. Leave them unchanged; the `local_tys` map is populated by **a new helper** (§D.4).

**Restore-on-fail invariant.** For tests + future use, the map is additive — physics-Let lowering inserts on success. No restore is needed (a let-binding that fails never reaches this point; and re-entry of the same `LocalId` with a different type would be a driver-side defect, not a runtime concern).

### D.4 — Add `LoweringCtx::record_local_ty`

Add a helper after `allocate_local` (line 238):

```
/// Record `local_id → ty` for later `IrExpr::Local` resolution.
/// Called by physics-Let lowering after the bound expression's CG
/// type is known. Returns the prior `CgTy` if one was registered for
/// the same id (driver-side duplicate — surfacing it lets tests
/// assert exclusive allocation).
pub fn record_local_ty(&mut self, local_id: LocalId, ty: CgTy) -> Option<CgTy> {
    self.local_tys.insert(local_id, ty)
}
```

### D.5 — Wire `record_local_ty` into physics-Let lowering

**File:** `crates/dsl_compiler/src/cg/lower/physics.rs`

In the existing Let-lowering helper `lower_let` (around line 553), after the `CgStmt::Let { local, value, ty }` is constructed but before it is pushed (or after — the order doesn't affect correctness as long as the type is recorded before any read reaches `lower_bare_local`), add:

```
ctx.record_local_ty(local, ty);
```

Concretely, at the line where `let stmt = CgStmt::Let { local, value, ty };` is built (around line 592), insert `ctx.record_local_ty(local, ty);` immediately after `let local = ...` resolution completes (line 590) and before the `let stmt = ...` line. Pin the insertion point on the surrounding `match` shape — the apply agent reads the existing helper and inserts the call where `local` and `ty` are both in scope.

### D.6 — Add `LoweringError::UnknownLocalType`

**File:** `crates/dsl_compiler/src/cg/lower/error.rs`

Add a new variant after `UnsupportedLocalBinding` (line 213–216):

```
/// A bare `IrExpr::Local(local_ref, name)` resolved through
/// `LoweringCtx::local_ids` to a typed `LocalId`, but the matching
/// `LocalId → CgTy` entry in `LoweringCtx::local_tys` was missing.
/// The driver populates `local_tys` as part of `IrStmt::Let` lowering
/// (`record_local_ty`); a missing entry is either a stale registry or
/// a hand-built AST whose let-binding was lowered without recording
/// the type. Distinct from [`Self::UnsupportedLocalBinding`] — that
/// variant fires when the *name* is unknown; this one fires when the
/// *type* of a bound name is unknown.
UnknownLocalType {
    local: crate::cg::stmt::LocalId,
    span: Span,
},
```

Add a `Display` arm at the appropriate site (the file has a per-variant Display match; mirror the shape of `UnsupportedLocalBinding`'s Display, around line 853):

```
LoweringError::UnknownLocalType { local, span } => write!(
    f,
    "{}: bare local {} resolved to a binding with no recorded CgTy",
    span, local,
),
```

### D.7 — Tests

Append to the `mod tests` in `crates/dsl_compiler/src/cg/lower/expr.rs` (the existing test scaffold — after the field-access tests, around line 1450):

1. `bare_self_lowers_to_agent_self_id` — construct `IrExpr::Local(LocalRef(0), "self")` directly, lower with default ctx, assert `pretty(...)` equals `"(agent self_id)"`.
2. `bare_target_in_pair_bound_lowers_to_per_pair_candidate_id` — same shape but `ctx.target_local = true`, name `"target"`, expect `"(agent per_pair_candidate_id)"`.
3. `bare_target_outside_pair_bound_rejected` — `ctx.target_local = false`, name `"target"`, assert `Err(LoweringError::UnsupportedLocalBinding { name: "target", .. })`.
4. `bare_unknown_local_rejected` — name `"foo"`, assert `Err(LoweringError::UnsupportedLocalBinding { name: "foo", .. })`.
5. `agents_alive_self_lowers_through_namespace_call` — construct
   ```
   IrExpr::NamespaceCall {
     ns: NamespaceId::Agents,
     method: "alive",
     args: vec![arg(node(IrExpr::Local(LocalRef(0), "self")))],
   }
   ```
   lower it, assert it produces a `Read(AgentField { field: Alive, target: AgentRef::Target(_) })` whose target expr resolves to `(agent self_id)`. (The existing namespace-call lowering at line 1206 builds an `AgentRef::Target(target_id)` from the lowered argument — the bare `self` flows in as the target id.)
6. `target_neq_self_lowers_in_pair_bound` — `target_local = true`, AST: `Binary(NotEq, Local(_,"target"), Local(_,"self"))`, lower, assert pretty equals `"(ne.agent_id (agent per_pair_candidate_id) (agent self_id))"`.
7. `let_bound_local_read_lowers_to_read_local` — manually populate `ctx.local_ids` with `LocalRef(7) → LocalId(3)`, populate `ctx.local_tys` with `LocalId(3) → CgTy::F32`, lower `IrExpr::Local(LocalRef(7), "x")`, assert pretty equals `"(read_local local#3 f32)"`.
8. `let_bound_local_without_recorded_ty_rejected` — populate `local_ids` only (no `local_tys` entry), assert `Err(LoweringError::UnknownLocalType { local: LocalId(3), .. })`.

---

## Section E — Defer `Match` / `Quantifier` / `Fold` / `PerUnit` (expression-position) with documented typed errors

These already surface as `LoweringError::UnsupportedAstNode { ast_label: "..." }` (see `crates/dsl_compiler/src/cg/lower/expr.rs:420-451`). The current arms are correct. **Add a doc note** to each arm tagging the deferral rationale so the apply agent (and follow-up tasks) see the structural reason inline.

### E.1 — Replace the four arms (lines 420–451) with annotated forms

For each of the four — `IrExpr::Quantifier { .. }`, `IrExpr::Fold { .. }`, `IrExpr::Match { .. }`, `IrExpr::PerUnit { .. }` — keep the `Err(LoweringError::UnsupportedAstNode { ast_label: "...", span })` shape unchanged, but **prepend a one-line `//`** explanation:

```
// Quantifier (forall/exists) reduces over a collection. Closing this
// shape requires a CG-IR aggregation primitive (no abstraction today);
// deferred. See plan task 5.5d limitations.
IrExpr::Quantifier { .. } => Err(LoweringError::UnsupportedAstNode {
    ast_label: "Quantifier",
    span,
}),

// Fold reduces a collection with an accumulator. Same blocker as
// Quantifier — needs a CG-IR aggregation primitive; deferred.
IrExpr::Fold { .. } => Err(LoweringError::UnsupportedAstNode {
    ast_label: "Fold",
    span,
}),

// Match in expression position carries arm bodies returning values;
// statement-position Match is wired (cg/lower/physics.rs). Closing the
// expression form needs an aggregation/select primitive; deferred.
IrExpr::Match { .. } => Err(LoweringError::UnsupportedAstNode {
    ast_label: "Match",
    span,
}),

// PerUnit modifier (`+per_unit_distance(target.hp_pct)`) is a scoring-
// row score modifier shaped like a fold over the same source the row
// iterates. Same blocker as Fold; deferred.
IrExpr::PerUnit { .. } => Err(LoweringError::UnsupportedAstNode {
    ast_label: "PerUnit",
    span,
}),
```

### E.2 — Defer `IrStmt::For` in physics body

**File:** `crates/dsl_compiler/src/cg/lower/physics.rs`, line 408–412.

The arm already returns `LoweringError::UnsupportedPhysicsStmt { ast_label: "For", .. }`. Keep this unchanged; **add an inline comment** above it noting that even unrolling requires reading the ability registry, which the CG IR doesn't carry today:

```
// `For` in physics body iterates over `abilities.effects(ab)`
// (typically). Lowering needs either compile-time unrolling (requires
// the ability registry, which the CG IR doesn't carry today) or a
// `CgStmt::For` primitive (no abstraction today); deferred to a future
// task.
IrStmt::For { span, .. } => Err(LoweringError::UnsupportedPhysicsStmt {
    rule: rule_id,
    ast_label: "For",
    span,
}),
```

---

## Section F — Limitations doc block

**File:** `crates/dsl_compiler/src/cg/lower/expr.rs`

At the module-level docstring (currently at lines 1–22), after the existing "Diagnostics vs hard errors" section (line 12), append a new section **"# Limitations (Task 5.5d)"** with the following bullets:

```
//! # Limitations (Task 5.5d)
//!
//! - **`IrExpr::Match`** (expression position): deferred. Needs a
//!   CG-IR aggregation/select primitive that arms can return values
//!   into. The statement-position `Match` is wired in
//!   [`super::physics::lower_stmt`].
//! - **`IrExpr::Quantifier`** (`forall`/`exists`): deferred. Same
//!   blocker — no CG-IR aggregation primitive today.
//! - **`IrExpr::Fold`**: deferred. Same blocker.
//! - **`IrExpr::PerUnit`** (scoring-row modifier): deferred. Shaped
//!   like a fold; same blocker.
//! - **`IrStmt::For`** (physics body): deferred. Either a `CgStmt::For`
//!   primitive or compile-time unrolling against the ability registry;
//!   neither exists today.
//!
//! All five surface as typed `LoweringError::UnsupportedAstNode` /
//! `LoweringError::UnsupportedPhysicsStmt` deferrals — closing them is
//! a future task.
//!
//! - **Bare `self` / `target` resolution.** Wired here as
//!   `CgExpr::AgentSelfId` / `CgExpr::PerPairCandidateId`. Bare
//!   `target` requires `LoweringCtx::target_local = true` — set by the
//!   pair-bound mask driver (Task 5.5a) and, in 5.5c, by per-pair
//!   scoring + fold-body drivers.
//! - **Let-bound local reads.** Wired here as `CgExpr::ReadLocal {
//!   local, ty }`. Resolution requires both `local_ids` (from Task
//!   5.5b) and the new `local_tys` map (this task) to be populated;
//!   `IrStmt::Let` lowering calls `ctx.record_local_ty(...)` to
//!   populate the latter.
```

---

## Section G — Verification

After applying all sections:

1. `cargo build` should succeed.
2. `cargo test -p dsl_compiler` should run the new tests added in §A.5, §B.5, §D.7. Existing 755 lib + 10 xtask tests must still pass.
3. Optional smoke: `cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml` should still pass (no behavioral change is expected — bare-local + Let-read are additive paths that were not previously reachable on real DSL coverage; the path now lights up but produces the same output as the legacy emit because the WGSL identifiers (`agent_id`, `per_pair_candidate`, `local_<N>`) match the legacy hand-written emit's names).

---

## Summary of files modified

| File | Sections | Lines touched (approx) |
|---|---|---|
| `crates/dsl_compiler/src/cg/expr.rs` | A, B | enum (+3 variants), `ty()` (+3 arms), `pretty_into` (+3 arms), `type_check` (+3 arms), tests |
| `crates/dsl_compiler/src/cg/lower/expr.rs` | D, F | `LoweringCtx` (+1 field), `LoweringCtx::new` (+1 init), `record_local_ty` (new helper), `lower_expr` Local arm (replaced), `lower_bare_local` (new helper), module doc, tests |
| `crates/dsl_compiler/src/cg/lower/physics.rs` | D.5, E.2 | `lower_let` (+1 line), `IrStmt::For` arm (+1 comment) |
| `crates/dsl_compiler/src/cg/lower/error.rs` | D.6 | enum (+1 variant), Display match (+1 arm) |
| `crates/dsl_compiler/src/cg/program.rs` | C.1 | `validate_expr_refs` (extend leaf-arm) |
| `crates/dsl_compiler/src/cg/well_formed.rs` | C.2 | `collect_subexpr_ids` (extend no-op arm) |
| `crates/dsl_compiler/src/cg/stmt.rs` | C.3 | `collect_expr_reads` (+1 no-op arm group) |
| `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` | C.4 | `lower_cg_expr_to_wgsl` (+3 arms) |

**No edits to:** `cg/data_handle.rs`, `cg/op.rs`, `cg/dispatch.rs`, `cg/schedule/*`, `cg/emit/program.rs`, `cg/emit/kernel.rs`, `cg/emit/cross_cutting.rs`, `cg/lower/mask.rs`, `cg/lower/scoring.rs`, `cg/lower/driver.rs`, `cg/lower/plumbing.rs`, `cg/lower/spatial.rs`, `cg/lower/view.rs`. (Verified: no walker arms in those files would gain coverage from the new variants.)

---

## Out-of-scope (call out for follow-up tasks)

- Match / Quantifier / Fold / PerUnit / For — covered by typed deferrals; future task.
- Well-formed cross-validation that every `CgExpr::ReadLocal { local, .. }` has a matching `CgStmt::Let { local, .. }` somewhere in the surrounding op's body — future task. Today the type checker accepts the binding's claimed type at face value.
- Locale-name binding registry growing beyond `local_ids` + `local_tys` (e.g., source-name → LocalId for diagnostics) — future task.
- Match-arm pattern binding `record_local_ty` wiring — Task 5.5b's match lowering at `physics.rs:600+` records `local_ids` but not `local_tys` for pattern-bound locals (since match-arm bindings come from event-variant fields whose CG types are already statically known via the event registry). A read of a pattern-bound local would surface `UnknownLocalType` until that wiring lands. Out-of-scope here.
