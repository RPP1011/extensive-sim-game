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

The lowering diagnostics from `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical` cluster as:

| Category | Original | Post-Task-1 | Cause | Tasks |
|---|---|---|---|---|
| Local binding not bound | 15 | **0** | Event-pattern binders not synthesized into `CgStmt::Let`. Closed by Task 1 (commit `f9068cb7`). | ✓ Done |
| Emit-field schema mismatch | 6 | **0** | `template_id` not registered for event#37. Closed by Task 1 as side effect — `populate_event_kinds` now mirrors `pack_event` per-variant. | ✓ Done |
| View-fold type mismatch | 9 | **10** | `view[#N].primary expects view_key<#N>, got f32/u32/i32` — fold value-expression type not matching view-key. +1 unmasked (op#14, view_key<#9> got i32) after Task 1. | 5, 6 |
| Namespace call no expr-lowering | 3 | **3** | `agents.is_hostile_to()`, `agents.engaged_with_or()`, `query.nearest_hostile_to_or()` not routed in `lower_expr`. (`abilities.is_known()` from original count is gone — closed by Task 1; `query.nearest_*` newly visible.) | 4 |
| **Namespace field no expr-lowering** | 0 | **3** | `world.tick` reads have no expression-level lowering — newly visible after Task 1 unmasked physics handlers. NEW category. | 4 (expanded) |
| Virtual-field reads | 1 | **1** | `self.hp_pct` synthesized from `hp / max_hp` not lowered. | 8 |
| **Binary type mismatch (i32 vs u32)** | 0 | **3** | `delta != 0`, `f > 0`, `a != 0` — signed event field compared to default-u32 literal. Predicted in Task 1's subagent report; was masked by binding errors before. NEW category. | 5 (expanded) |
| (B3 cleanup) | — | — | Re-evaluate remaining gaps; emit Movement / ApplyActions / etc. via existing lowering | 9, 10, 11 |
| (Runtime gate) | — | — | Parity green | 12 |
| **Total** | **39** | **20** | — | — |

> **Re-counted 2026-05-01 post-Task-1 from `compile-dsl --cg-canonical 2>&1 | grep "lowering:"`**: 20 diagnostics across 5 categories. Task 1 closed 21 diagnostics in two categories (15 local-binding + 6 emit-field-schema, the latter as a side effect of populating layouts comprehensively). Two new categories surface (namespace-field + binary-type-mismatch) that were masked by the binding errors before. Task 7 is moot — preserved as a stub in the doc below but no implementation work needed.

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

## Task 1: Event-pattern binding lowering via `CgExpr::EventField` + schema-driven layout

> **AIS amendment (2026-05-01)**: The original Task 1 hypothesis — that top-of-body `Let` statements skipped `lower_let` — was *wrong*. A read-only investigation by the first dispatch confirmed `lower_stmt` already routes `IrStmt::Let` to `lower_let` (`physics.rs:402-407`), `lower_let` populates `ctx.local_ids` + `local_tys`, and `lower_bare_local` resolves to `CgExpr::ReadLocal`. The infrastructure works. **The actual cause of all 15 `local binding ... is not bound` diagnostics is event-pattern bindings**: every diagnostic traced (spans 1190, 5847, 14084, 18756, 26654, 28526, 29131, 29993) resolves to a `t` / `actor` / `target` / `amount` / `delta` / `dead` introduced by an `on EffectXxx { actor: c, target: t, amount: a, ... }` event pattern, then read inside the handler body. `lower_one_handler` (`physics.rs:255`) jumps from intern-name straight into `lower_stmt_list` *without* walking `handler.pattern.bindings()`. Task 1 is rewritten below to address this real cause; Task 2 (originally for-loop body Let) is dropped — for-loop body Let lowering already works, and any `delta`-style diagnostic the original AIS ascribed to it is actually a pattern binding.

**The real bug.** Event patterns introduce binders that map to typed values extracted from the event payload (`target: AgentId`, `actor: AgentId`, `amount: f32`, etc.). The IR has no shape today that represents "value extracted from the current event's payload" — so `lower_one_handler` can't synthesize the necessary `CgStmt::Let`s for each pattern binder. Once the value-side shape exists, the synthesis is mechanical: walk `handler.pattern.bindings()`, look up `(word_offset_in_payload, ty)` from the schema per binder, push `CgStmt::Let`, and the existing `local_ids` + `lower_bare_local` chain handles every read inside the body.

**Schema-driven layout (forward-compat with per-kind ring fanout).** The runtime today uses one shared event ring with fixed stride = 10 u32 (2 header + 8 payload, sized for `AgentMoved`/`AgentFled` at 7 payload words with one word headroom — see `crates/engine_gpu/src/event_ring.rs:79-110`). Future events (e.g. `AgentSpawned` with template + equipment list, `EffectAreaApplied` with hit_targets[N], `MemoryRecorded` with embedding[16]) will blow past 8 payload words; padding everything to max-payload becomes prohibitively wasteful. The runtime is expected to move to **per-kind ring fanout** — one ring per event kind, each sized to its actual payload + frequency. The IR design here doesn't force the runtime change, but it makes it a local edit when the time comes: the schema is the layout authority, and the WGSL emit reads `(record_stride_u32, header_word_count, buffer_name, field_offsets)` per kind from the schema. Today every kind returns the uniform values (stride=10, header=2, buffer="event_ring"); future kinds return per-kind values without any IR shape change.

**Files:**
- Modify: `crates/dsl_compiler/src/cg/expr.rs` — add `CgExpr::EventField { event_kind, word_offset_in_payload, ty }` variant; extend exhaustive matches across the codebase (~10 sites).
- Modify: `crates/dsl_compiler/src/cg/lower/ctx.rs` (or wherever `LoweringCtx` lives) — add `event_layouts: HashMap<EventKindId, EventLayout>` field; populate in `populate_event_kinds`.
- Modify: `crates/dsl_compiler/src/cg/lower/physics.rs::lower_one_handler` — walk `handler.pattern.bindings()`, synthesize `CgStmt::Let` per binder.
- Modify: `crates/dsl_compiler/src/cg/emit/wgsl_body.rs::lower_cg_expr_to_wgsl` — add `EventField` arm, schema-driven access form.
- Modify: `crates/dsl_compiler/src/cg/well_formed.rs` — add scope check that `EventField` only appears in PerEvent-shaped op bodies.
- Test: `crates/dsl_compiler/src/cg/lower/physics.rs::tests`, `wgsl_body.rs::tests`, `well_formed.rs::tests`.

- [ ] **Step 1: Add `CgExpr::EventField` variant + propagate exhaustive matches**

In `crates/dsl_compiler/src/cg/expr.rs`, extend the `CgExpr` enum:
```rust
pub enum CgExpr {
    // ... existing variants ...
    /// Read a typed field from the current event's payload. Schema-
    /// driven: `event_kind` keys into `LoweringCtx::event_layouts` to
    /// resolve `(record_stride_u32, header_word_count, buffer_name,
    /// field_offset)`. Per-kind ring fanout is forward-compatible —
    /// today's emit produces `event_ring[event_idx * 10u + 2u + offset]`
    /// for every kind; future schema returns per-kind buffer + stride
    /// and the same emit produces `event_ring_<kind>[...]` naturally.
    EventField {
        event_kind: EventKindId,
        word_offset_in_payload: u32,
        ty: CgTy,
    },
}
```

Run `cargo build -p dsl_compiler` and follow the exhaustiveness errors to update every match (likely `wgsl_body.rs::lower_cg_expr_to_wgsl`, `well_formed::type_check`, `op.rs::compute_dependencies`, possibly more — let the compiler enumerate them).

- [ ] **Step 2: Add `EventLayout` struct + `LoweringCtx::event_layouts`**

In whatever file holds `LoweringCtx`:
```rust
#[derive(Debug, Clone)]
pub struct EventLayout {
    /// u32 words per record. Today: 10 for every kind (= 2 header + 8
    /// payload). Future per-kind ring fanout: per-kind value.
    pub record_stride_u32: u32,
    /// Header word count (kind + tick). Constant across runtime
    /// strategies — kept on the layout struct for symmetry.
    pub header_word_count: u32,
    /// WGSL identifier for the storage buffer this kind reads from.
    /// Today: "event_ring" for every kind. Future: per-kind names like
    /// "event_ring_AgentMoved".
    pub buffer_name: String,
    /// Field name → offset within payload (u32 words from start of
    /// payload, NOT including header).
    pub fields: BTreeMap<String, FieldLayout>,
}

#[derive(Debug, Clone, Copy)]
pub struct FieldLayout {
    pub word_offset_in_payload: u32,
    pub word_count: u32,
    pub ty: CgTy,
}
```

Add `event_layouts: HashMap<EventKindId, EventLayout>` to `LoweringCtx`, populate in `populate_event_kinds` (driver.rs) — for now every kind gets uniform `record_stride_u32: 10, header_word_count: 2, buffer_name: "event_ring".to_string()`, with `fields` derived from the AST event variant's field list.

Mirror the existing `pack_event` field layouts from `crates/engine_gpu/src/event_ring.rs:256+` — that's the runtime's source of truth for which payload words hold which fields per variant.

- [ ] **Step 3: Extend `lower_one_handler` to synthesize `CgStmt::Let` per pattern binder**

In `lower_one_handler` (`physics.rs:255`), before `lower_stmt_list`:
```rust
for binding in handler.pattern.bindings() {
    let layout = ctx.event_layouts.get(&event_kind_id)
        .ok_or(LoweringError::UnregisteredEventKind { kind: event_kind_id })?;
    let field = layout.fields.get(&binding.name)
        .ok_or(LoweringError::UnregisteredEventField {
            event: event_kind_id, field: binding.name.clone(),
        })?;
    let local_id = ctx.allocate_local();
    let value_expr_id = add(ctx, CgExpr::EventField {
        event_kind: event_kind_id,
        word_offset_in_payload: field.word_offset_in_payload,
        ty: field.ty,
    }, binding.span)?;
    let stmt = CgStmt::Let { local: local_id, value: value_expr_id, ty: field.ty };
    builder.push_stmt(stmt);
    ctx.register_local(binding.local_ref, local_id);
    ctx.local_tys.insert(local_id, field.ty);
}
```

(Adapt to the actual API shapes — `handler.pattern.bindings()` may be named differently in the AST. Verify the API name with `rg "bindings\b" crates/dsl_ast/src/`.)

- [ ] **Step 4: Add WGSL emit arm**

In `wgsl_body.rs::lower_cg_expr_to_wgsl`:
```rust
CgExpr::EventField { event_kind, word_offset_in_payload, ty } => {
    let layout = ctx.event_layouts.get(event_kind)
        .ok_or(EmitError::UnregisteredEventKind { kind: *event_kind })?;
    let total_offset = layout.header_word_count + word_offset_in_payload;
    let base = format!("event_idx * {}u + {}u", layout.record_stride_u32, total_offset);
    Ok(match ty {
        CgTy::AgentId | CgTy::U32 => format!("{}[{}]", layout.buffer_name, base),
        CgTy::F32 => format!("bitcast<f32>({}[{}])", layout.buffer_name, base),
        CgTy::Vec3 => format!(
            "vec3<f32>(bitcast<f32>({buf}[event_idx * {s}u + {o}u]), bitcast<f32>({buf}[event_idx * {s}u + {o2}u]), bitcast<f32>({buf}[event_idx * {s}u + {o3}u]))",
            buf = layout.buffer_name, s = layout.record_stride_u32,
            o = total_offset, o2 = total_offset + 1, o3 = total_offset + 2,
        ),
        // ... other CgTy arms ...
    })
}
```

Note: `ctx.event_layouts` needs to be reachable from `EmitCtx`. If `EmitCtx` only carries `prog`, extend it to carry the layouts (or thread them through `prog` if natural).

- [ ] **Step 5: Add well_formed scope check**

In `well_formed.rs`, when walking an op body, track whether the surrounding op's `DispatchShape` is `PerEvent { .. }`. If `EventField` appears in a non-PerEvent body, surface as `CgError::EventFieldInNonPerEventBody { op_index }`.

- [ ] **Step 6: Write tests**

```rust
#[test]
fn event_pattern_binding_lowers_to_event_field_let() {
    // Build a physics rule via test fixture:
    //   on Damage { actor: c, target: t, amount: a } {
    //     emit DamageRecorded { target: t, stamp: a };
    //   }
    // Lower; assert the synthesized CgStmt::Let sequence at the head of
    // the body, each value being a CgExpr::EventField with the right
    // (event_kind, word_offset_in_payload, ty).
}

#[test]
fn event_field_emits_schema_driven_wgsl_access() {
    // Lower a CgExpr::EventField{event_kind: 0, word_offset_in_payload: 1, ty: AgentId}
    // with a uniform layout (stride=10, header=2). Assert WGSL is
    // "event_ring[event_idx * 10u + 3u]".
}

#[test]
fn event_field_in_per_agent_body_flagged_by_well_formed() {
    // Construct an op with DispatchShape::PerAgent whose body reads
    // EventField. Assert CgError::EventFieldInNonPerEventBody.
}
```

- [ ] **Step 7: Run tests to verify**

`cargo test -p dsl_compiler --lib`. The three new tests pass. All 801+ existing tests still pass.

- [ ] **Step 8: Verify diagnostic count drops to 0**

```bash
cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | grep "lowering: local binding" | wc -l
```
Expected: `0` (down from 15).

- [ ] **Step 9: Commit**

```bash
git add crates/dsl_compiler/src/cg/expr.rs \
        crates/dsl_compiler/src/cg/lower/ctx.rs \
        crates/dsl_compiler/src/cg/lower/physics.rs \
        crates/dsl_compiler/src/cg/lower/driver.rs \
        crates/dsl_compiler/src/cg/emit/wgsl_body.rs \
        crates/dsl_compiler/src/cg/well_formed.rs \
        crates/engine_gpu_rules/
git commit -m "feat(dsl_compiler): event-pattern binding lowering via CgExpr::EventField (schema-driven layout)"
```

---

## Task 2: (REMOVED — superseded by amended Task 1)

> The original Task 2 (cast-rule for-loop body Let lowering) was based on the misdiagnosis that drove Task 1. The first-dispatch investigation confirmed `lower_let` and `lower_for` body walks already work; every `local binding` diagnostic is an event-pattern binding, addressed by Task 1's synthesis hook in `lower_one_handler`.
>
> If Task 3's checkpoint reveals any surviving local-binding diagnostic that isn't an event-pattern binding, this slot will be re-populated with a focused fix.

---

## Task 3: Diagnostic inventory checkpoint (post-Task-1)

**Files:** None (verification step).

- [ ] **Step 1: Re-run compile-dsl + count remaining diagnostics by category**

Run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | grep "lowering:" | sort | uniq -c | sort -rn`

Expected after Task 1: All 15 `local binding` diagnostics gone; remaining categories untouched (3 namespace-call, 9 view-fold-type, 6 emit-field-schema, 1 virtual-field).

- [ ] **Step 2: If any local-binding diagnostics survive, write a focused test**

Pick one surviving diagnostic, write a minimal test reproducing it (Task 1 pattern: build the IR by hand or compile from a fixture, assert no `UnsupportedLocalBinding` for the named binder). Fix and commit. If the surviving diagnostic is structurally distinct from event-pattern bindings (e.g., a true for-loop body local that Task 1 didn't touch), this is where Task 2's slot gets re-populated with the focused fix — otherwise it stays empty.

- [ ] **Step 3: Document the post-Task-1 diagnostic state**

Update this plan's Diagnostic Inventory table with the live count (commit the doc edit alongside any focused fix). Don't commit a stale doc.

---

## Task 4: Namespace-call AND namespace-field expression lowering

> **Scope expanded post-Task-3 (2026-05-01)**: original Task 4 covered 3 namespace-call sites (`agents.is_hostile_to`, `abilities.is_known`, `agents.engaged_with_or`). Post-Task-1 the surface is different: `abilities.is_known` is gone (closed by Task 1), `query.nearest_hostile_to_or()` is newly visible, and a sibling shape — namespace-FIELD reads like `world.tick` (3 sites) — also surfaced. Task 4's scope now includes both.

**Sites to cover** (concrete from compile-dsl output, post-Task-1):
- Calls (3): `agents.is_hostile_to(target)`, `agents.engaged_with_or(target, fallback)`, `query.nearest_hostile_to_or(...)`.
- Fields (3 instances of 1 unique site): `world.tick` — global tick read.

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/expr.rs::lower_expr` — add `IrExpr::NamespaceCall { ns, method, args }` and `IrExpr::NamespaceField { ns, field }` arms.
- Modify: `crates/dsl_compiler/src/cg/expr.rs` — possibly extend `BuiltinId` to add `IsHostileTo`, `EngagedWithOr`, `NearestHostileToOr` if routed as builtins; add `WorldTick` (or treat as a Read of a synthesized DataHandle).
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

## Task 5: View-key typing AND signed/unsigned binary-operand coercion

> **Scope expanded post-Task-3 (2026-05-01)**: original Task 5 was scoped to view-key strict-equality (9 diagnostics, +1 unmasked → now 10 covering f32/u32/i32). A sibling type-system gap surfaced post-Task-1: 3 binary `NotEq`/`Gt` operations between i32 (signed event fields like `delta`/`f`/`a`) and u32 default literals fail with "mismatched operands". Both are typing-rule relaxations on closely related code, so they fold into Task 5.

**Diagnostic counts after Task 5:**
- View-fold-type: 10 → 0
- Binary i32-vs-u32 mismatch: 3 → 0

**Files:**
- Modify: `crates/dsl_compiler/src/cg/well_formed.rs` — assign type-mismatch rule for `ViewStorage{slot: Primary}` targets; add scalar-substitution acceptance for `view_key<T>`. Also: relax binary-operand equality so signed/unsigned literals coerce to the lhs type when the rhs is a literal (not when both are non-literal — that genuinely is a programming bug).
- Modify: `crates/dsl_compiler/src/cg/lower/view.rs::lower_stmt` (or upstream `validate_storage_slot`) — coerce `f32`/`u32`/`i32` to `view_key<#N>` when the view's primary is the scalar accumulator slot.
- Modify: `crates/dsl_compiler/src/cg/lower/expr.rs::lower_binary` — when one operand is a u32 literal `0` and the other is i32, lower the literal as i32 instead of u32 (DSL-side default-literal coercion).
- Test: `crates/dsl_compiler/src/cg/well_formed.rs::tests` + `lower/view.rs::tests` + `lower/expr.rs::tests`

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

## Task 7: (MOOT — closed by Task 1)

> Task 1's `populate_event_kinds` extension comprehensively mirrored `pack_event` for every event variant the runtime knows about, including `template_id`-bearing variants. The 6 emit-field-schema diagnostics that Task 7 was meant to address closed as a side effect. No implementation work needed; this slot is preserved for audit-trail continuity.
>
> If a future event variant adds a non-primitive field type that the schema collector doesn't recognize, this slot will be repopulated with a focused fix.

## Task 7 (legacy spec, kept for audit): Event-field schema — register `template_id`

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

## Task 9: Inventory checkpoint — POST-TASK-1/4/5/8 STATE (2026-05-01)

> **Inventory complete (commit `cf471517`).** SCHEDULE has 26 kernels. 33 of 39 original lowering diagnostics closed (the remaining 1 is a pre-existing cycle-detector warning, structurally separate from this plan).
>
> **SCHEDULE today (26 kernels):**
> - Masks (4): MaskAttack, MaskHold, MaskMoveToward, FusedMaskFlee
> - Folds (7): FoldEngagedWith, FoldKinFear, FoldMyEnemies, FoldPackFocus, FoldRallyBoost, FoldStanding, FoldThreatLevel
> - Physics chronicles (7, all Indirect): PhysicsChronicle{Attack, Break, Engagement, Flee, Rally, Rout, Wound}
> - Spatial (2): FusedSpatialBuildHash, SpatialEngagementQuery
> - Plumbing (5): UploadSimCfg, PackAgents, SeedIndirect0, UnpackAgents, KickSnapshot
> - Indirect-fold (1): FusedFoldMemoryRecordMemory
>
> **Genuinely missing (the decision/action chain that drives parity divergence):**
> - **Scoring / PickAbility** — per-agent action picker. The `ScoringArgmax` lowering exists in `cg/lower/scoring.rs` but no kernel named "scoring" emits in SCHEDULE; the scoring rules either (a) silently fuse into another kernel or (b) don't lower to `ComputeOps` yet.
> - **ApplyActions** — write decided actions to agent SoA (HP changes, position deltas, status effects).
> - **Movement** — physics rule applying movement deltas. Today's PhysicsChronicle* are chronicle-only (narrative event emit); the actual Movement physics rule isn't being emitted as a Movement kernel.
> - **AppendEvents** — append derived events to the cascade ring beyond what cascade Indirect already does.
> - **AlivePack / FusedAgentUnpack / MaskUnpack** — pack/unpack helpers between dispatch phases.
>
> **Why parity diverges (verified via `parity_with_cpu --features gpu`):** smoke fixture has 4 agents at ~10-unit distance. CPU's AI picks `MoveToward` (close-to-engage), CPU's Movement physics rule applies pos delta, agents close distance over the n_ticks. GPU's SCHEDULE has masks + folds + spatial-hash + chronicle-emit but no decision/movement chain → GPU agents stay at spawn position. Hence `pos_x_bits` divergence.
>
> **Scope of remaining work (Tasks 10-12):** multi-day work. Each missing kernel needs (a) a `ComputeOpKind` lowering (some exist, some need to be wired through), (b) a WGSL body template, (c) bind-source synthesis for new transient buffers (action_buf, scoring_output, etc.). Tasks 10 and 11's "skeleton" structure now has concrete content per the bullets above.

## Task 9 (legacy spec, kept for audit): Inventory the post-fix kernel set vs engine_gpu's expectations

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
