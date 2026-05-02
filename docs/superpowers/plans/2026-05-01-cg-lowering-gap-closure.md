# CG Lowering Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the AST → CG IR lowering gap so the compiler emits real WGSL bodies for every kernel the runtime needs, allowing `parity_with_cpu --features gpu` to turn green for the smoke fixture.

**Architecture:** The Phase 5 switchover (commits 494760b8 → d32ed08d) wired the CG-emitted KernelId set, made the dispatch match itself a CG-emitted artifact, and replaced placeholder identifiers with B1 typed-default fallbacks so the pipeline runs end-to-end. Today the lowering produces 39 diagnostics across four categories: local-binding gaps, namespace-call gaps, view-key typing mismatches, and event-field schema mismatches. Each diagnostic class blocks specific rule bodies from lowering to real CG IR, which in turn keeps the corresponding kernels from emitting real bodies. The fixes are extensions to existing lowering modules (no new IR shapes); the design here is the precise inventory of which extension fixes which diagnostic and how to verify it.

**Tech Stack:** Rust workspace; `dsl_compiler` (lowering), `engine_gpu_rules` (CG-emitted output), `engine_gpu` (runtime consumer). `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical` regenerates; `cargo test -p engine_gpu --features gpu --test parity_with_cpu` is the runtime gate.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `LoweringError::UnsupportedLocalBinding` at `crates/dsl_compiler/src/cg/lower/error.rs:213`
  - `lower_bare_local` at `crates/dsl_compiler/src/cg/lower/expr.rs:735`
  - `LoweringCtx::register_local` / `local_ids` / `local_tys` at `crates/dsl_compiler/src/cg/lower/expr.rs:262, 106, 169`
  - `IrStmt::Let` lowering at `crates/dsl_compiler/src/cg/lower/physics.rs::lower_let` (Task 5.5b)
  - `view_storage_primary` / `view_key<#N>` typing at `crates/dsl_compiler/src/cg/well_formed.rs` (assign type-mismatch gate)
  - `lower_stmt` (fold body) at `crates/dsl_compiler/src/cg/lower/view.rs:464` (handles `IrStmt::SelfUpdate { op: "+=" }` only)
  - Namespace-call expression at `crates/resolver/src/...` (call expr → not yet routed through `lower_expr`)
  - `event-field schema` at `crates/dsl_compiler/src/cg/lower/driver.rs::populate_event_kinds` (event#37 = `template_id`-bearing event missing field registration)

  Search method: `rg`, `grep -n`, direct `Read` on the cited files.

- **Decision:** extend each existing lowering site. No new IR shapes. The four diagnostic categories all surface from missing match arms / un-registered ids in lowerings that are otherwise structurally complete.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: NONE (the DSL surface is already correct; the lowering is what's incomplete).
  - Generated outputs re-emitted: `crates/engine_gpu_rules/src/*.{rs,wgsl}` (every regen-on-DSL-change cycle).

- **Hand-written downstream code:** NONE.
  Task 7 (`engine_gpu/src/lib.rs`) is purely a delete-and-reduce: remove the legacy emit_*.rs files in `dsl_compiler/src/` after CG fully covers them (Task 5.8 from the parent plan). No new hand-written engine code.

- **Constitution check:**
  - P1 (Compiler-First): PASS — every change is in the DSL compiler. No `impl Rule` outside generated dirs.
  - P2 (Schema-Hash on Layout): N/A — no SoA layout change. The `template_id` event-field registration may bump `engine_data` event-schema hash; if so, regen is in scope.
  - P3 (Cross-Backend Parity): PASS — the entire goal is to make GPU match CPU for the smoke fixture. Parity gate is the validation.
  - P4 (`EffectOp` Size Budget): N/A — no `EffectOp` changes.
  - P5 (Determinism via Keyed PCG): PASS — RNG paths unchanged; lowerings preserve `CgExpr::Rng { purpose }` shape.
  - P6 (Events Are the Mutation Channel): PASS — `Emit` lowering is already in place; this plan extends `template_id` field registration so existing emits route through the schema.
  - P7 (Replayability Flagged): N/A — no event variant added.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with `git commit` + `closes_commit` SHA.
  - P10 (No Runtime Panic): PASS — lowerings emit `Result` and surface typed errors; no `unwrap()` introduced on hot paths.
  - P11 (Reduction Determinism): PASS — view-key typing fix preserves the existing atomic reduction shape; no new reductions introduced.

- **Runtime gate:**
  - `parity_with_cpu_n4_t1` at `crates/engine_gpu/tests/parity_with_cpu.rs` — CPU and GPU agent fingerprints byte-equal after 1 tick on the smoke fixture (4 idle agents). Run under `--features gpu`. Today RED (diverges on `pos_x_bits` for moving agents); GREEN at end of Task 12.
  - `parity_with_cpu_n4_t10`, `parity_with_cpu_n4_t100` — same, longer ticks. GREEN once Task 12 is complete (the chosen kernel set produces stable per-tick state).
  - `cargo test -p dsl_compiler --lib` — 801+ unit tests. Each task must keep this green; tasks that touch lowering shape update test fixtures rather than disabling them.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

---

## Diagnostic Inventory (drives task decomposition)

The 39 lowering diagnostics from `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical` cluster as:

| Category | Count | Cause | Tasks |
|---|---|---|---|
| Local binding not bound | 10 | `IrStmt::Let` not threaded into the rule body context (`ctx.local_ids` empty when `lower_bare_local` fires) | 1, 2, 3 |
| Namespace call no expr-lowering | 3 | `agents.is_hostile_to()`, `abilities.is_known()`, `agents.engaged_with_or()` not routed in `lower_expr` | 4 |
| View-fold type mismatch | 9 | `view[#N].primary expects view_key<#N>, got f32/u32` — fold value-expression type not matching view-key | 5, 6 |
| Emit-field schema mismatch | 6 | `template_id` not registered for event#37 in event-field schema | 7 |
| Virtual-field reads | 1 | `self.hp_pct` synthesized from `hp / max_hp` not lowered | 8 |
| (B3 cleanup) | — | Re-evaluate remaining gaps; emit Movement / ApplyActions / etc. via existing lowering | 9, 10, 11 |
| (Runtime gate) | — | Parity green | 12 |

Task 1-8 close the diagnostic surface. Task 9-11 verify that the emitted SCHEDULE contains the kernels engine_gpu expects (Movement, ApplyActions, AppendEvents, AlivePack, FusedAgentUnpack, MaskUnpack, ScoringUnpack, FoldStanding, Physics, SpatialHash split, SpatialKinQuery, SpatialEngagementQuery). Task 12 is the parity gate.

Where this plan says "missing kernel" it means "the SCHEDULE today doesn't contain it"; whether that's because lowering errors silently drop the op or because no IR construct produces it is what each task investigates.

---

## File Structure

This plan touches existing files only. No new modules.

- `crates/dsl_compiler/src/cg/lower/expr.rs` — `lower_bare_local`, `lower_namespace_call` (NEW match arm in `lower_expr`)
- `crates/dsl_compiler/src/cg/lower/physics.rs` — `lower_let`, `lower_for` (NEW for cast-rule for-loops)
- `crates/dsl_compiler/src/cg/lower/view.rs` — `lower_stmt` view-key typing
- `crates/dsl_compiler/src/cg/lower/scoring.rs` — virtual-field synthesis (`hp_pct`)
- `crates/dsl_compiler/src/cg/lower/driver.rs` — `populate_event_kinds` schema registration extension
- `crates/dsl_compiler/src/cg/well_formed.rs` — view-key type compatibility rule
- `crates/dsl_compiler/src/cg/op.rs` — possibly extend `PlumbingKind` if missing kernels need new variants (Task 9-11)
- `crates/engine_gpu_rules/src/*` — CG-emitted; regenerated each task
- `crates/engine_gpu/tests/parity_with_cpu.rs` — runtime gate (no edit; test must turn green)

---

## Task 1: Local-binding lowering for physics-rule body locals (top of body)

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/physics.rs:lower_let` — already exists (Task 5.5b); ensure every physics-rule body's `IrStmt::Let` reaches it.
- Modify: `crates/dsl_compiler/src/cg/lower/physics.rs::lower_physics` — verify the body walk visits every `IrStmt`, not just the `for body` head.
- Test: `crates/dsl_compiler/src/cg/lower/physics.rs::tests` — add fixture exercising `let t = ...; let delta = ...;` in a physics body.

- [ ] **Step 1: Write the failing test**

Add to `lower/physics.rs::tests`:
```rust
#[test]
fn lower_physics_handles_top_of_body_let_binding() {
    let mut prog = CgProgram::default();
    // physics rule with `on Damage { let t = self.tick; emit X { stamp: t } }`
    let comp = compile_str(
        r#"
        on event Damage(target: AgentId, amount: f32) {
            let t = world.tick;
            emit DamageRecorded { target: target, stamp: t };
        }
        "#,
    ).expect("compile parses");
    let mut ctx = LoweringCtx::new(&mut /* builder */);
    let mut diags = Vec::new();
    lower_all_physics(&comp, &[/*ring=0*/], &mut ctx, &mut diags);
    assert!(diags.iter().all(|d| !matches!(d, LoweringError::UnsupportedLocalBinding { name, .. } if name == "t")),
        "let-bound `t` must lower without UnsupportedLocalBinding diagnostic, got: {diags:?}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --lib lower_physics_handles_top_of_body_let_binding -- --nocapture`
Expected: FAIL with "UnsupportedLocalBinding { name: \"t\", ... }".

- [ ] **Step 3: Investigate `lower_physics` body walk**

Read `crates/dsl_compiler/src/cg/lower/physics.rs::lower_physics`. Identify whether `IrStmt::Let` arms are reached for top-of-body lets vs for-loop lets. The Task 5.5b lowering routes `Let` → `CgStmt::Let` + `ctx.register_local(...)`, so the failure means the walk doesn't reach the `Let` arm. Fix the walk. If `Let` IS reached but `register_local` isn't called for some reason, fix that.

- [ ] **Step 4: Apply the smallest fix that makes the test pass**

Likely edit: extend the body-iteration in `lower_physics::lower_rule_body` to dispatch each `IrStmt` through `lower_stmt`, including `Let`. Today it may special-case `For` and skip `Let` at the top level.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --lib lower_physics_handles_top_of_body_let_binding`
Expected: PASS.

- [ ] **Step 6: Run the full diagnostic count**

Run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | grep "lowering: local binding" | wc -l`
Expected: down from 10 (current) to ≤6 (the for-loop-bound and target-aliased ones survive; the top-of-body ones go away).

- [ ] **Step 7: Commit**

```bash
git add crates/dsl_compiler/src/cg/lower/physics.rs crates/engine_gpu_rules/
git commit -m "fix(dsl_compiler): physics-rule top-of-body Let binding now lowers to CgStmt::Let"
```

---

## Task 2: Local-binding lowering for cast-rule `for` body locals

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/physics.rs::lower_for` (likely NEW or extend existing for-handling)
- Test: `crates/dsl_compiler/src/cg/lower/physics.rs::tests`

- [ ] **Step 1: Write the failing test**

Add a test exercising `for target in spatial_query(...) { let delta = ...; ... }`:
```rust
#[test]
fn lower_physics_handles_for_loop_body_let() {
    // physics rule with `for target in agents.engaged() { let delta = self.dmg; emit Hit { target, amount: delta } }`
    // Compile and assert no UnsupportedLocalBinding diagnostics for `delta`.
    // (Concrete fixture text — see Task 1 pattern.)
}
```

- [ ] **Step 2: Verify it fails before the fix**

`cargo test -p dsl_compiler --lib lower_physics_handles_for_loop_body_let`. Expected: FAIL.

- [ ] **Step 3: Extend `lower_for` to thread body lets**

The for-loop body walk needs the same `IrStmt::Let` arm as Task 1, plus the loop-binder (`for target in ...`) registers as a structural local pointing at the per-iteration target id (similar to `target` in mask predicates — likely a new `CgExpr::PerPairCandidateId` shape or a new `CgExpr::ForLoopTarget(id)`).

- [ ] **Step 4: Verify pass + diagnostic count drop**

`cargo test -p dsl_compiler --lib`. All 801+ pass. Diagnostic count for `local binding` drops by the for-bound count (likely 4-6 of the remaining).

- [ ] **Step 5: Commit**

```bash
git commit -m "fix(dsl_compiler): cast-rule for-loop body Let bindings + loop-binder lower"
```

---

## Task 3: Diagnostic inventory mid-point checkpoint

**Files:** None (verification step).

- [ ] **Step 1: Re-run compile-dsl + count remaining diagnostics by category**

Run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | grep "lowering:" | sort | uniq -c | sort -rn`

Expected: All 10 `local binding` diagnostics gone; remaining categories untouched (3 namespace-call, 9 view-fold-type, 6 emit-field-schema, 1 virtual-field).

- [ ] **Step 2: If any local-binding diagnostics survive, write a focused test**

Pick one surviving diagnostic, write a minimal test reproducing it (Task 1 / Task 2 pattern), fix, commit.

- [ ] **Step 3: Document the post-Task-1+2 diagnostic state**

Update this plan's Diagnostic Inventory table with the live count (PR comment or amended commit). Don't commit a stale doc.

---

## Task 4: Namespace-call expression lowering

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/expr.rs::lower_expr` — add a `IrExpr::NamespaceCall { ns, method, args }` arm.
- Modify: `crates/dsl_compiler/src/cg/expr.rs` — possibly extend `BuiltinId` to add `IsHostileTo`, `IsKnown`, `EngagedWithOr` if they're routed as builtins.
- Test: `crates/dsl_compiler/src/cg/lower/expr.rs::tests`

- [ ] **Step 1: Write the failing test for `agents.is_hostile_to(other)`**

```rust
#[test]
fn lower_expr_handles_agents_is_hostile_to_call() {
    // Compile a rule using `agents.is_hostile_to(target)`, assert the lowering
    // produces `CgExpr::Builtin { fn_id: BuiltinId::IsHostileTo, args: [target_id] }`
    // (or whichever shape Task 4 chooses).
}
```

- [ ] **Step 2: Run test, expect FAIL**

Expected: `LoweringError::NamespaceCallNoExprLowering` or panic if there's no arm at all.

- [ ] **Step 3: Decide on the lowering shape**

Choose ONE:
- (a) Treat `agents.X()` calls as `CgExpr::Builtin { fn_id: BuiltinId::X, args }`. Pros: reuses existing builtin routing in `wgsl_body.rs::builtin_name`. Cons: pollutes BuiltinId namespace.
- (b) Add `CgExpr::NamespaceCall { ns: NsId, method: MethodId, args }`. Pros: cleaner separation. Cons: new variant + downstream emitter expansion.

Recommend (a) — three call sites, tightly bounded — and add a doc comment noting that (b) is the right move once the 4-method count grows past ~10.

- [ ] **Step 4: Implement + WGSL prelude**

Add prelude functions in the WGSL output for each builtin:
```wgsl
fn is_hostile_to(a: u32, b: u32) -> bool { /* faction lookup */ return false; }  // B1 stub
fn is_known(ability: u32) -> bool { return true; }
fn engaged_with_or(target: u32, fallback: u32) -> u32 { return fallback; }
```
The B1 stubs are placeholders; Task 12's parity gate will reveal whether the stubs need real bodies before parity is green for the smoke fixture (likely yes for `is_hostile_to`).

- [ ] **Step 5: Verify diagnostic count drops by 3, parity unchanged**

`cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | grep "namespace call" | wc -l` → 0.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(dsl_compiler): namespace-call expression lowering for agents.* / abilities.*"
```

---

## Task 5: View-key typing — accept scalar `+=` against view_key views

**Files:**
- Modify: `crates/dsl_compiler/src/cg/well_formed.rs` — assign type-mismatch rule for `ViewStorage{slot: Primary}` targets.
- Modify: `crates/dsl_compiler/src/cg/lower/view.rs::lower_stmt` (or upstream `validate_storage_slot`) — coerce `f32`/`u32` to `view_key<#N>` when the view's primary is the scalar accumulator slot.
- Test: `crates/dsl_compiler/src/cg/well_formed.rs::tests` + `lower/view.rs::tests`

- [ ] **Step 1: Write a test exercising `f32 += 1.0` against a view with `view_key<f32>` primary**

```rust
#[test]
fn fold_body_f32_self_update_accepts_view_key_f32_primary() {
    // ThreatLevel view: primary = view_key<f32>. Body: `self += 1.0`.
    // Lower should accept; well_formed should not flag a type mismatch.
}
```

- [ ] **Step 2: Verify FAIL**

Run: today the well_formed gate flags 9 such mismatches. Expected: panic from `lower_view::validate_storage_slot` or diagnostic from `well_formed::check_assign_type_compatibility`.

- [ ] **Step 3: Fix**

Two options:
- (a) The view_key<T> shape *is* T at the primitive level for f32 / u32 / vec3 keys; well_formed's strict equality is wrong. Relax the rule to accept `T <: view_key<T>` substitution.
- (b) The lowering should wrap the value in `CgExpr::ViewKeyCoerce { from, to }` before assigning. New IR shape.

Recommend (a): the view-key system is structurally a phantom type for "this value goes into view storage at key K". The runtime treats `view_key<f32>` as `f32`; well_formed's strict equality is more conservative than necessary. Relax with a typed-acceptance rule that documents the substitution explicitly.

- [ ] **Step 4: Verify diagnostic count drops by 9, all unit tests pass**

`cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | grep "view_key" | wc -l` → 0.

- [ ] **Step 5: Commit**

```bash
git commit -m "fix(dsl_compiler): view-key strict-equality relaxed to scalar substitution"
```

---

## Task 6: Slot-aware view-fold body emission (replaces B1 phony-discard)

**Files:**
- Modify: `crates/dsl_compiler/src/cg/emit/wgsl_body.rs::lower_cg_stmt_to_wgsl` — replace the B1 `_ = (rhs);` fallback for `Assign(ViewStorage, _)` with a real `view_storage_<slot>[event_target_id] += rhs;` emission.
- Modify: `crates/dsl_compiler/src/cg/emit/kernel.rs` — fold body composer threads `event_target_id` into scope.
- Test: `crates/dsl_compiler/src/cg/emit/wgsl_body.rs::tests::view_storage_assign_lowers_to_indexed_atomic_add`

- [ ] **Step 1: Write the test asserting the new emit shape**

```rust
#[test]
fn view_storage_assign_lowers_to_indexed_atomic_add() {
    // Build a CG IR with one Assign(ViewStorage{view: 3, slot: Primary}, Lit(1.0)).
    // Lower with EmitCtx::structural; assert body contains
    // "atomicAdd(&view_storage_primary[event_target_id], bitcast<u32>(1.0));"
    // (or the equivalent fp32 atomic shape that wgpu supports — may be a CAS loop).
}
```

- [ ] **Step 2: Run, expect FAIL — current emit is `_ = (1.0);`**

- [ ] **Step 3: Implement the slot-aware emission**

Inside `lower_cg_stmt_to_wgsl`'s Assign arm, when the target is `ViewStorage{slot, ...}`:
- Use the BGL-bound `view_storage_<slot>` name (NOT the structural `view_<id>_<slot>`).
- Index by `event_target_id` (the per-iteration event's target field — bound by the fold-kernel preamble).
- Emit `atomicAdd` if the kernel's reduction strategy is per-event accumulation; emit a CAS loop if WGSL's atomicAdd doesn't support the value type (f32 atomicAdd requires WGSL 1.x with the `atomic-fp` extension; fall back to bitcast<u32>+CAS).

- [ ] **Step 4: Update fold-body kernel preamble in `kernel.rs` to bind `event_target_id`**

Already mostly there (`let event_idx = gid.x; if (event_idx >= cfg.event_count) { return; }`). Add: `let event_target_id = event_ring[event_idx * RECORD_U32_STRIDE + TARGET_FIELD_OFFSET];` (real shape — port from legacy fold WGSL).

- [ ] **Step 5: Verify all 7 fold WGSL files emit the indexed-atomic shape**

`grep -c "atomicAdd(&view_storage_primary" crates/engine_gpu_rules/src/fold_*.wgsl` → 7 (matching the fold count).

- [ ] **Step 6: Update the `view_fold_wgsl_body_has_event_count_gate` test**

The B1 assertion `body.contains("_ = (1.0);")` (committed in d32ed08d) flips back to `body.contains("atomicAdd(&view_storage_primary")`.

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(dsl_compiler): ViewStorage Assign lowers to slot-indexed atomic accumulator"
```

---

## Task 7: Event-field schema — register `template_id`

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/driver.rs::populate_event_kinds` (or the event-field schema source)
- Test: `crates/dsl_compiler/src/cg/lower/driver.rs::tests`

- [ ] **Step 1: Identify event#37 + its DSL source**

Run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | grep "template_id at"` and read each span. Map to `assets/sim/events.sim` to find the event whose declaration includes `template_id`.

- [ ] **Step 2: Identify what's missing — DSL declaration vs schema parse**

If the DSL says `event Spawn(template_id: TemplateId, ...)` but the field-schema collector doesn't record `template_id`, fix the collector. If the DSL is missing the field declaration, that's a DSL bug — flag and fix the DSL.

Most likely outcome (based on the diagnostic shape "not registered for event#37"): the schema collector misses fields with non-primitive types. Extend the collector to handle `TemplateId` and any other non-primitive event-field types.

- [ ] **Step 3: Write a focused regression test**

```rust
#[test]
fn event_field_schema_registers_template_id_for_event_37() {
    let comp = compile_str(/* events.sim with template_id field */).unwrap();
    let mut ctx = LoweringCtx::new(/* ... */);
    let mut diags = Vec::new();
    let _ = populate_event_kinds(&comp, &mut ctx, &mut diags);
    assert!(diags.iter().all(|d| !matches!(d, LoweringError::EmitFieldSchema { field, .. } if field == "template_id")));
}
```

- [ ] **Step 4: Fix + verify**

`cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | grep "template_id" | wc -l` → 0.

- [ ] **Step 5: Commit**

```bash
git commit -m "fix(dsl_compiler): event-field schema registers template_id (TemplateId-typed field)"
```

---

## Task 8: Virtual-field synthesis (`hp_pct` from `hp / max_hp`)

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/expr.rs` — `lower_field_access` adds a `hp_pct` arm that synthesizes `hp / max_hp`.
- Test: `crates/dsl_compiler/src/cg/lower/expr.rs::tests`

- [ ] **Step 1: Write a test for `self.hp_pct` lowering**

```rust
#[test]
fn lower_self_hp_pct_synthesizes_hp_over_max_hp() {
    // Compile a fixture using `self.hp_pct < 0.5`. Assert the lowered CgExpr
    // shape is Binary { op: Lt, lhs: Binary { op: Div, lhs: Read(Hp), rhs: Read(MaxHp) }, rhs: Lit(0.5) }.
}
```

- [ ] **Step 2: Run, expect `does not name an agent field` error**

- [ ] **Step 3: Add `hp_pct` recognition in `lower_field_access`**

Match `hp_pct` against a virtual-field table; emit the synthesized expression. Add a doc comment listing every virtual field (today: `hp_pct` only; future: `mana_pct`, `cooldown_progress`, etc.).

- [ ] **Step 4: Verify**

`cargo test -p dsl_compiler --lib`. All pass. Diagnostic count for `does not name` drops by 1.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(dsl_compiler): virtual field hp_pct lowers to hp / max_hp"
```

---

## Task 9: Inventory the post-fix kernel set vs engine_gpu's expectations

**Files:** None (investigation step). Output: amended plan or new task entries.

- [ ] **Step 1: Regen + dump the SCHEDULE**

```bash
cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical
grep "DispatchOp::Kernel" crates/engine_gpu_rules/src/schedule.rs | sort -u
```

- [ ] **Step 2: List the kernels engine_gpu expects but CG still doesn't emit**

Compare against the legacy hand-written set from before Path A. Likely still missing:
- Movement, ApplyActions — physics-rule body lowerings that should now succeed (post Task 1+2).
- AppendEvents, AlivePack, FusedAgentUnpack, MaskUnpack, ScoringUnpack — plumbing kinds that may need new `PlumbingKind` variants.
- Physics — fixed-point dispatcher; depends on plumbing for the iteration tail-zero gate.
- SpatialHash split, SpatialKinQuery, SpatialEngagementQuery — the fused kernel today consolidates these; engine_gpu may expect them split, or the runtime can adapt to fused.
- FoldStanding — additional view fold; likely the `standing` view needs a `@materialized` annotation and a fold-body declaration in `assets/sim/views.sim`.

- [ ] **Step 2: Write a focused investigation per missing kernel**

For each kernel name on the list:
- Is the AST construct present in DSL? (grep `assets/sim/`)
- Does the existing lowering produce a `ComputeOp` for it?
- Does the schedule synthesizer fuse / drop the op?
- What's the smallest fix?

- [ ] **Step 3: Convert the investigation into Tasks 10, 11**

Add concrete tasks for the top-3 most-impactful gaps. Defer the rest if parity passes earlier (it might — if the smoke fixture's CPU step doesn't need every legacy kernel).

- [ ] **Step 4: Commit the amended plan**

```bash
git commit -m "docs: amend cg-lowering-gap-closure plan with post-Task-1-8 kernel inventory"
```

---

## Task 10: Emit Movement + ApplyActions kernels (highest-impact pair)

> Concrete steps depend on Task 9's investigation. Skeleton:

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/physics.rs` — extend lowering coverage so the Movement physics rule produces a `ComputeOpKind::PhysicsRule` op with a real body (not a stub).
- Modify: `crates/dsl_compiler/src/cg/op.rs::PlumbingKind` — possibly add `ApplyActions` if it's a sink kernel rather than a physics rule.
- Modify: `crates/dsl_compiler/src/cg/emit/kernel.rs::plumbing_body_for_kind` — body template for the new variant.
- Test: parity move-step micro-test (1 agent, 1 tick, expect non-zero pos delta).

- [ ] **Steps 1-N: To be filled in per Task 9 findings.**

- [ ] **Final: Commit**

```bash
git commit -m "feat(dsl_compiler): Movement physics rule + ApplyActions sink kernels"
```

---

## Task 11: Emit remaining missing kernels (catch-up pass)

> Concrete steps depend on Task 9's investigation. Lower-priority kernels (AppendEvents, AlivePack, etc.) get bundled here once the high-impact pair is in.

**Files:** Same files as Task 10. Add `PlumbingKind` variants and matching body templates.

- [ ] **Steps 1-N: Per-kernel sub-steps from Task 9 investigation.**

- [ ] **Final: Commit**

```bash
git commit -m "feat(dsl_compiler): emit remaining lowering-gap kernels (AppendEvents, AlivePack, ...)"
```

---

## Task 12: Runtime gate — parity_with_cpu green

**Files:**
- No source edit. Verification only.

- [ ] **Step 1: Regen + run parity**

```bash
cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical
cargo test -p engine_gpu --features gpu --test parity_with_cpu -- --nocapture
```

- [ ] **Step 2: Three test names must pass**

`parity_with_cpu_n4_t1`, `parity_with_cpu_n4_t10`, `parity_with_cpu_n4_t100`. All on the `smoke_fixture_n4` (4 agents, mixed creature types).

- [ ] **Step 3: If parity diverges**

For each diverging field in the fingerprint (CPU vs GPU after N ticks):
1. Bisect by ticks: which tick produces the first divergence?
2. Bisect by kernel: temporarily skip-dispatch each kernel one at a time (replace dispatch arm with no-op) and rerun. The kernel whose absence flips parity is the culprit.
3. Inspect the kernel's WGSL body vs its CPU equivalent (in `engine_rules/src/`).
4. Add a focused micro-test for the divergence; fix; re-run.

- [ ] **Step 4: Once green, run `cargo test --workspace --features gpu` for breadth**

Ensure no other tests broke. Pre-existing `per_entity_ring_emits_wgsl_fold_kernel` is unrelated and stays RED until the legacy emit_view_wgsl path is retired (Task 5.8 in the parent plan, out of scope here).

- [ ] **Step 5: Commit + close**

```bash
git commit --allow-empty -m "test: parity_with_cpu --features gpu green for smoke_fixture_n4 (3/3)"
```

Mark Task 145 in the project's task list as completed with `closes_commit` set to this commit's SHA (P9 invariant).

---

## Out-of-scope for this plan

- **`per_unit` modifier lowering** — the K-bounded-view-storage iteration path (deferred from Task 5.5d). Not required for the smoke-fixture parity gate; tracked as a separate plan.
- **Megakernel-strategy emission** — multi-strategy benchmarking flags (`--cg-strategy=...`). Tracked separately.
- **Legacy emitter retirement** — Task 5.8 of the parent plan retires `dsl_compiler/src/emit_*.rs` once CG covers their kernels. Out of scope here; CG-canonical mode already supersedes legacy at the engine_gpu_rules write step.
- **Pre-existing `per_entity_ring_emits_wgsl_fold_kernel` test failure** — unrelated to this plan (legacy emit_view_wgsl path).

## References

- Constitution: `docs/constitution.md` — P1, P2, P3, P5, P6, P11.
- Parent plan: `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md` — Phase 5 switchover context.
- Path A commit: `494760b8` — engine_gpu dispatch alignment.
- Path C commit: `a3fdfe3f` — emit-driven dispatch.
- B1 commit: `d32ed08d` — pipeline runs end-to-end on real GPU.
