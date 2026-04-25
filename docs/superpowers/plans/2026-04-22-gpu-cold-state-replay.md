# GPU Cold-State Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the engine's cold-state rule handlers (chronicle, gold, standing, memory) to the GPU batch path. Driven by two new DSL `@materialized` view annotations and a `@cpu_only` escape hatch for narrative-text rules.

**Architecture:** Phase 1 extends the DSL compiler with `@symmetric_pair_topk(K)` and `@per_entity_ring(K)` view annotations (CPU + WGSL emit paths). Phases 2-4 use those primitives to port sub-item 4a (chronicle), 4b (gold + standing), and 4c (record_memory). `SparseStandings` and per-agent `memory` smallvecs become compiler-emitted views, replacing hand-written state. All rule bodies are DSL-lowered; engine-core infrastructure (side buffers, dispatch tables, snapshot handshakes) is hand-written.

**Tech Stack:** Rust, WGSL, `crates/dsl_compiler` (parser, IR, emit_physics, emit_physics_wgsl, emit_view, emit_view_wgsl), `crates/engine` (state, schema_hash), `crates/engine_gpu` (cascade_resident, lib), existing `@materialized` view infrastructure.

**Spec reference:** `docs/spec/gpu.md` §4 (cold-state replay).

---

## Scope decomposition

Spec has four phases. This plan details **Phase 1 in full**; Phases 2-4 are sketched and get their own detailed plans when Phase 1 lands. Rationale: Phase 1 is foundational (new DSL annotations block all other work), and landing + validating it first means Phases 2-4's plans can reference the real emit API surface instead of speculating.

### Phase 1 (this plan, detailed)
DSL compiler extensions: two new `@materialized` view annotations + their CPU and WGSL emit paths.

### Phase 2 (sketched — future plan)
Chronicle-stubs port (4a): 8 CPU rule handlers → structured GPU emissions + `@cpu_only` for narrative text.

### Phase 3 (sketched — future plan)
Gold (dedicated atomic side buffer) + standing (`@symmetric_pair_topk`-based view) port (4b).

### Phase 4 (sketched — future plan)
`record_memory` port using `@per_entity_ring` (4c).

---

# Phase 1 — DSL Compiler Extensions

**Goal:** Add two new view-storage shape annotations the subsequent phases consume. Each annotation needs parser/IR/CPU-emit/WGSL-emit support plus unit tests.

## File structure

### New files

- `crates/dsl_compiler/src/emit_view/symmetric_pair_topk.rs` — CPU fold emit + storage layout for symmetric-pair topk views.
- `crates/dsl_compiler/src/emit_view/per_entity_ring.rs` — CPU fold emit + storage layout for per-entity ring views.
- `crates/dsl_compiler/src/emit_view_wgsl/symmetric_pair_topk.rs` — WGSL fold kernel emit for symmetric-pair topk.
- `crates/dsl_compiler/src/emit_view_wgsl/per_entity_ring.rs` — WGSL fold kernel emit for per-entity ring.
- `crates/dsl_compiler/tests/annotation_symmetric_pair_topk.rs` — integration test lowering a minimal view.
- `crates/dsl_compiler/tests/annotation_per_entity_ring.rs` — integration test lowering a minimal view.

### Modified files

- `crates/dsl_compiler/src/parser.rs` — recognise `@symmetric_pair_topk(K)` + `@per_entity_ring(K)`.
- `crates/dsl_compiler/src/ast.rs` — add annotation variants.
- `crates/dsl_compiler/src/ir.rs` — add view-shape variants to materialized-view IR.
- `crates/dsl_compiler/src/resolve.rs` — thread annotations through lowering.
- `crates/dsl_compiler/src/emit_view.rs` / `emit_view_wgsl.rs` (or their mod.rs files) — dispatch to the new per-shape emitters.

### Untouched

- All engine crates (`engine`, `engine_gpu`).
- `assets/sim/` (views won't switch over until Phase 3/4).

---

## Task 1.1: Parser + AST for `@symmetric_pair_topk(K)`

**Files:**
- Modify: `crates/dsl_compiler/src/parser.rs`
- Modify: `crates/dsl_compiler/src/ast.rs`
- Test: `crates/dsl_compiler/tests/annotation_symmetric_pair_topk.rs`

### Step 1: Write the failing parser test

Create `crates/dsl_compiler/tests/annotation_symmetric_pair_topk.rs`:

```rust
//! Verify the `@symmetric_pair_topk(K)` annotation parses and lowers
//! into the materialized-view IR with the expected K value.

use dsl_compiler::{parse_program, Decl};

#[test]
fn symmetric_pair_topk_annotation_parses_with_k_argument() {
    let src = r#"
event StandingDelta { a: AgentId, b: AgentId, delta: i16 }

view standing
    @materialized
    @symmetric_pair_topk(K = 8)
{
    on StandingDelta { a, b, delta } {
        // fold body (stub for this parser test)
    }
}
"#;
    let program = parse_program(src).expect("parse OK");
    let view = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::View(v) if v.name == "standing" => Some(v),
            _ => None,
        })
        .expect("view 'standing' should exist");
    assert!(
        view.annotations.iter().any(|a| a.name == "symmetric_pair_topk"),
        "symmetric_pair_topk annotation should be present"
    );
    let ann = view
        .annotations
        .iter()
        .find(|a| a.name == "symmetric_pair_topk")
        .unwrap();
    // K=8 argument should parse as a single named int.
    assert_eq!(ann.args.len(), 1);
    match &ann.args[0] {
        dsl_compiler::ast::AnnotationArg::Named { name, value } => {
            assert_eq!(name, "K");
            // Value is an int expression; depends on AnnotationArg's variants.
            // Adjust this match arm to whatever the parser produces.
            let _ = value;
        }
        _ => panic!("K=8 should parse as a named-int arg"),
    }
}
```

Adjust `Decl::View` / `AnnotationArg` variants to the actual grammar — grep `pub enum Decl\|pub enum AnnotationArg` in `ast.rs`.

### Step 2: Run the test — expect FAIL

Run: `cargo test -p dsl_compiler --test annotation_symmetric_pair_topk`
Expected: FAIL because the annotation currently isn't a known annotation kind (it parses as a generic annotation with name `"symmetric_pair_topk"`, which may actually succeed at the parser level — this test might pass trivially. If so, strengthen by asserting the IR-level lowering, which happens in Task 1.3.)

### Step 3: Verify generic annotation handling

Many DSL compilers already parse arbitrary `@<name>(<args>)` via a generic Annotation struct (Phase 1 of the GPU sim state subsystem confirmed this for `@cpu_only`). If that's the case, the parser step is trivial — the annotation name + args flow through by convention. The meaningful work is in IR lowering (Task 1.3).

Run: `cargo test -p dsl_compiler` to confirm all existing tests still pass.

### Step 4: Commit

```bash
git add crates/dsl_compiler/tests/annotation_symmetric_pair_topk.rs
git commit -m "test(dsl_compiler): symmetric_pair_topk annotation parser test"
```

---

## Task 1.2: Parser + AST for `@per_entity_ring(K)`

**Files:**
- Test: `crates/dsl_compiler/tests/annotation_per_entity_ring.rs`

Mirror of Task 1.1 applied to the ring annotation.

### Step 1: Write the failing parser test

```rust
//! Verify the `@per_entity_ring(K)` annotation parses and surfaces
//! on the view's annotation list.

use dsl_compiler::{parse_program, Decl};

#[test]
fn per_entity_ring_annotation_parses_with_k_argument() {
    let src = r#"
event RecordMemory { observer: AgentId, source: AgentId, fact: u64, confidence: f32 }

view memory
    @materialized
    @per_entity_ring(K = 64)
{
    on RecordMemory { observer, source, fact, confidence } {
        // fold body (stub)
    }
}
"#;
    let program = parse_program(src).expect("parse OK");
    let view = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::View(v) if v.name == "memory" => Some(v),
            _ => None,
        })
        .expect("view 'memory' should exist");
    assert!(
        view.annotations.iter().any(|a| a.name == "per_entity_ring"),
        "per_entity_ring annotation should be present"
    );
}
```

### Step 2-4: same pattern as Task 1.1.

Commit: `test(dsl_compiler): per_entity_ring annotation parser test`

---

## Task 1.3: IR lowering — `SymmetricPairTopK` view-shape variant

**Files:**
- Modify: `crates/dsl_compiler/src/ir.rs`
- Modify: `crates/dsl_compiler/src/resolve.rs`

### Goal

The materialized-view IR carries a shape enum (`PerEntityTopK`, `PerEntityScalar`, etc. — the existing variants used by `kin_fear`, `my_enemies`, `threat_level`). Add a `SymmetricPairTopK { k: u32 }` variant. Resolver reads the annotation vec on the view AST and produces the IR variant.

### Step 1: Locate the existing view-shape enum

```
grep -n "pub enum.*ViewShape\|PerEntityTopK\|ViewStorage\|MaterializedView" crates/dsl_compiler/src/ir.rs | head
```

Find the existing shape enum (or tagged union). Variants likely include something for the `kin_fear` topk shape. Example structure (adjust to reality):

```rust
pub enum ViewShape {
    PerEntityScalar,
    PerEntityTopK { k: u32 },
    // ...
}
```

### Step 2: Add the new variant

```rust
pub enum ViewShape {
    // ... existing variants ...

    /// Task 167/169 adjacent — symmetric pair-keyed per-entity
    /// storage. Each agent keeps up to `k` edges; reads dedupe by
    /// min/max ordering so (a, b) and (b, a) resolve to the same
    /// entry. No decay.
    SymmetricPairTopK { k: u32 },
}
```

### Step 3: Resolver annotation lookup

In `resolve.rs`, find where view annotations are parsed into the IR `ViewShape`. Extend the dispatch:

```rust
if view_ast.annotations.iter().any(|a| a.name == "symmetric_pair_topk") {
    let k = extract_k_arg(&view_ast.annotations, "symmetric_pair_topk")?;
    ViewShape::SymmetricPairTopK { k }
}
```

`extract_k_arg` is a small helper (add if not present) that reads `K = <int>` from the annotation's args list.

### Step 4: Test lowering

Extend `annotation_symmetric_pair_topk.rs`:

```rust
#[test]
fn symmetric_pair_topk_lowers_to_ir_variant() {
    let src = /* same src as the parser test above */;
    let comp = dsl_compiler::compile(src).expect("compile OK");
    let view_ir = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR present");
    match view_ir.shape {
        ViewShape::SymmetricPairTopK { k } => assert_eq!(k, 8),
        other => panic!("expected SymmetricPairTopK, got {other:?}"),
    }
}
```

### Step 5: Run + commit

```
cargo test -p dsl_compiler --test annotation_symmetric_pair_topk
cargo test -p dsl_compiler
```

```bash
git add crates/dsl_compiler/src/ir.rs crates/dsl_compiler/src/resolve.rs crates/dsl_compiler/tests/annotation_symmetric_pair_topk.rs
git commit -m "feat(dsl_compiler): SymmetricPairTopK view-shape IR variant"
```

---

## Task 1.4: IR lowering — `PerEntityRing` view-shape variant

**Files:**
- Modify: `crates/dsl_compiler/src/ir.rs`
- Modify: `crates/dsl_compiler/src/resolve.rs`

Mirror of Task 1.3. Add `ViewShape::PerEntityRing { k: u32 }`. Same resolver pattern.

### Variant definition

```rust
/// Per-entity FIFO ring of fixed size K. Each entity has a
/// write cursor; pushes atomically increment it mod K. Reads
/// return all K slots in cursor-relative order.
PerEntityRing { k: u32 },
```

Commit message: `feat(dsl_compiler): PerEntityRing view-shape IR variant`

---

## Task 1.5: CPU emit — `SymmetricPairTopK` fold

**Files:**
- Create: `crates/dsl_compiler/src/emit_view/symmetric_pair_topk.rs`
- Modify: `crates/dsl_compiler/src/emit_view.rs` (or the emit_view `mod.rs`) — dispatch to the new emitter

### Goal

Emit Rust code for the CPU fold handler. The fold handler:
1. Takes an event (e.g. `StandingDelta { a, b, delta }`).
2. Computes canonical pair key: `(min(a, b), max(a, b))`.
3. Locates or creates an entry in one of the two endpoints' per-entity slot arrays (pick the lower-id endpoint as the "owner"; the higher-id reads symmetrically).
4. Updates the value (additive delta, clamped if the view declares clamp bounds).
5. If the owner's slot array is full (K entries) and this is a new pair, evicts the entry with the lowest absolute value.

### Skeleton

```rust
//! CPU emit for `@symmetric_pair_topk(K)` views. Mirrors the
//! existing `per_entity_topk` emitter's shape; differs in
//! pair-canonicalisation and symmetric read access.

use crate::ir::{ViewIR, ViewShape};

pub fn emit_cpu_storage(view: &ViewIR, k: u32) -> String {
    // Emits a struct like:
    //   pub struct Standing {
    //       pub slots: Vec<[PairEdge; K]>,
    //       pub counts: Vec<u8>,
    //   }
    //   impl Standing {
    //       pub fn get(&self, a: AgentId, b: AgentId) -> i16 { ... }
    //       pub fn adjust(&mut self, a: AgentId, b: AgentId, delta: i16) { ... }
    //   }
    //
    // `PairEdge { other: AgentId, value: i16 }` — per-slot record.
    // Exact field names inferred from the view's fold body.
    ...
}

pub fn emit_cpu_fold(view: &ViewIR, event_name: &str) -> String {
    // Emits the fold handler body that dispatches on the view's
    // `on <Event> { ... }` block(s).
    ...
}
```

### Step 1: Read the existing per-entity-topk emitter

Use it as the template. Grep: `grep -rn "per_entity_topk\|PerEntityTopK" crates/dsl_compiler/src/emit_view*`. The symmetric-pair case shares most of the structure; only the key canonicalization and storage layout differ.

### Step 2: Implement the emitter

Add `symmetric_pair_topk.rs` with `emit_cpu_storage` + `emit_cpu_fold` functions. Dispatch to them from the shape-dispatch in `emit_view.rs`.

### Step 3: Test via a minimal view fixture

Extend `annotation_symmetric_pair_topk.rs`:

```rust
#[test]
fn symmetric_pair_topk_emits_cpu_storage() {
    let src = /* ... */;
    let out = dsl_compiler::compile(src).expect("compile OK");
    let rust = out.views_rust_module(); // or equivalent accessor for CPU view Rust
    // Assert the generated struct exists and has slots + counts.
    assert!(rust.contains("pub struct Standing"));
    assert!(rust.contains("slots: Vec<["));
    assert!(rust.contains("fn adjust"));
    assert!(rust.contains("fn get"));
}
```

Adjust output accessor to reality.

### Step 4: Build + run

```
cargo build -p dsl_compiler
cargo test -p dsl_compiler
```

### Step 5: Commit

```bash
git add crates/dsl_compiler/src/emit_view/symmetric_pair_topk.rs \
        crates/dsl_compiler/src/emit_view.rs \
        crates/dsl_compiler/tests/annotation_symmetric_pair_topk.rs
git commit -m "feat(dsl_compiler): CPU emit for SymmetricPairTopK views"
```

---

## Task 1.6: CPU emit — `PerEntityRing` fold

**Files:**
- Create: `crates/dsl_compiler/src/emit_view/per_entity_ring.rs`
- Modify: `crates/dsl_compiler/src/emit_view.rs`

Mirror of Task 1.5 applied to rings. The CPU fold:
1. Each entity has a fixed-size `[Record; K]` array + a `cursor: u32`.
2. On push, write `ring[cursor % K] = record`, increment `cursor`.
3. Reads return the K slots in cursor-relative order (most recent first).

### Skeleton

```rust
//! CPU emit for `@per_entity_ring(K)` views. FIFO ring per entity
//! with fixed capacity K; oldest record evicted on overflow.

pub fn emit_cpu_storage(view: &ViewIR, k: u32) -> String {
    // Emits:
    //   pub struct Memory {
    //       pub rings: Vec<[MemoryEntry; K]>,
    //       pub cursors: Vec<u32>,
    //   }
    //   impl Memory {
    //       pub fn push(&mut self, observer: AgentId, entry: MemoryEntry);
    //       pub fn recent(&self, observer: AgentId) -> impl Iterator<Item=&MemoryEntry>;
    //   }
    ...
}
```

### Test

Inside `annotation_per_entity_ring.rs`:

```rust
#[test]
fn per_entity_ring_emits_cpu_storage() {
    let src = /* ... */;
    let out = dsl_compiler::compile(src).expect("compile OK");
    let rust = out.views_rust_module();
    assert!(rust.contains("pub struct Memory"));
    assert!(rust.contains("rings: Vec<["));
    assert!(rust.contains("cursors: Vec<u32>"));
    assert!(rust.contains("fn push"));
}
```

### Commit

```bash
git commit -m "feat(dsl_compiler): CPU emit for PerEntityRing views"
```

---

## Task 1.7: WGSL emit — `SymmetricPairTopK` fold kernel

**Files:**
- Create: `crates/dsl_compiler/src/emit_view_wgsl/symmetric_pair_topk.rs`
- Modify: `crates/dsl_compiler/src/emit_view_wgsl.rs`

### Goal

Emit a WGSL compute kernel that folds events into the GPU-resident symmetric-pair-topk view storage. The kernel's inputs:
- Event slice (typed records for this tick)
- Per-entity slot buffers (`slots_buf: array<PairEdge>`, length `agent_cap * K`)
- Per-entity counts (`counts_buf: array<u32>`, length `agent_cap`)

Kernel logic per event:
1. Compute canonical key: `let owner = min(a, b); let other = max(a, b);`
2. For `owner`'s slot array, linearly scan for an existing entry with `other == <other>`:
   - If found: atomically update the value.
   - If not found AND count < K: atomically append (atomic CAS on count).
   - If not found AND count == K: evict weakest (linear scan for lowest |value|, CAS to replace).

Use WGSL `atomicAdd` / `atomicCompareExchangeWeak` for the updates. Storage binding layout must match the Rust-side struct byte-for-byte (check with a `sim_cfg_layout`-style regression test).

### Skeleton

```rust
//! WGSL emit for @symmetric_pair_topk(K) views. Mirrors the existing
//! emit_view_wgsl per-entity-topk kernel's shape; differs in pair
//! canonicalisation.

pub fn emit_wgsl_struct(view: &ViewIR, k: u32) -> String {
    // struct PairEdge { other: u32, value: i32 };
    // @group(G) @binding(B) var<storage, read_write> slots: array<PairEdge>;
    // @group(G) @binding(B+1) var<storage, read_write> counts: array<atomic<u32>>;
    ...
}

pub fn emit_wgsl_fold(view: &ViewIR) -> String {
    // @compute @workgroup_size(N)
    // fn fold_<view_name>(@builtin(global_invocation_id) gid: vec3<u32>) {
    //     let event_idx = gid.x;
    //     if (event_idx >= num_events) { return; }
    //     let e = events[event_idx];
    //     let owner = min(e.a, e.b);
    //     let other = max(e.a, e.b);
    //     ... scan + update/insert/evict ...
    // }
    ...
}
```

### Step 1: Read the existing per-entity-topk WGSL emitter

```
grep -rn "per_entity_topk\|PerEntityTopK" crates/dsl_compiler/src/emit_view_wgsl*
```

Use as template.

### Step 2: Implement

Add `emit_view_wgsl/symmetric_pair_topk.rs`. Dispatch from `emit_view_wgsl.rs` based on `ViewShape::SymmetricPairTopK`.

### Step 3: Test WGSL emits + parses

Add an integration test that compiles a minimal view and asserts the emitted WGSL contains the expected `fold_<view_name>` function and binds the expected buffers. Use `naga` to validate the WGSL if that's the crate's convention.

### Step 4: Commit

```bash
git add crates/dsl_compiler/src/emit_view_wgsl/symmetric_pair_topk.rs \
        crates/dsl_compiler/src/emit_view_wgsl.rs
git commit -m "feat(dsl_compiler): WGSL emit for SymmetricPairTopK views"
```

---

## Task 1.8: WGSL emit — `PerEntityRing` fold kernel

**Files:**
- Create: `crates/dsl_compiler/src/emit_view_wgsl/per_entity_ring.rs`
- Modify: `crates/dsl_compiler/src/emit_view_wgsl.rs`

Mirror of Task 1.7 applied to rings. The WGSL fold:
1. Each entity has a write `cursor: atomic<u32>` and a `ring: array<MemoryEntry, K>`.
2. On push: `let slot = atomicAdd(&cursor[observer], 1u) % K; ring[observer * K + slot] = entry;`
3. Single-writer-per-entity if the event stream has no duplicates; multi-writer if it does, which atomicAdd handles correctly.

### Test

Same as Task 1.7 pattern — integration test emits WGSL, naga-validates, asserts expected kernel presence.

### Commit

```bash
git commit -m "feat(dsl_compiler): WGSL emit for PerEntityRing views"
```

---

## Task 1.9: Phase 1 regression sweep

- [ ] **Step 1:** `cargo test -p dsl_compiler` — all new + existing tests pass.

- [ ] **Step 2:** `cargo test --release --features gpu -p engine_gpu` — engine_gpu unaffected (annotations are DSL-only; no engine_gpu code consumes them yet). 113 + 3 ignored baseline.

- [ ] **Step 3:** Verify no DSL hash drift unless intentional:
   ```
   cargo test -p engine schema_hash
   ```
   Expected: pass. The new annotation variants don't alter any existing schema strings; only future views using them would.

- [ ] **Step 4:** Document any deviations or Phase-1 follow-ups.

- [ ] **Step 5:** Push the branch.

No commit — verification only.

---

# Phase 2 — 4a Chronicle Stubs (sketch)

**Prerequisite:** Phase 1 (new annotations available; note `@cpu_only` from subsystem 1 is already landed).

**Scope outline:**
- Per-rule audit of the 8 chronicle rules: which port to structured GPU emission (use existing chronicle ring), which stay `@cpu_only` (narrative text with string formatting).
- Dispatch framework in `step_batch` for cold-state handlers (per-event-kind kernel dispatch, amortised across batch).
- Integration tests: `cold_state_4a_structured_chronicle.rs`, `cold_state_4a_narrative_is_cpu_only.rs`.

**Estimated:** 5-7 tasks; ~1 week. Detailed plan after Phase 1 lands.

---

# Phase 3 — 4b Gold + Standing (sketch)

**Prerequisite:** Phase 1 (`@symmetric_pair_topk` needed for standing).

**Scope outline:**
- `gold_buf: array<atomic<i32>>` dedicated side buffer on `ResidentPathContext`. Snapshot handshake copies gold back to `SimState.cold_inventory` on `snapshot()`.
- Port `SparseStandings` hand-written state to a DSL `@materialized @symmetric_pair_topk(K = 8)` view. Delete the hand-written BTreeMap-backed struct.
- CPU consumers migrate to `state.views.standing.*`.
- `transfer_gold` physics rule: CPU-emitted handler writes `SimState.cold_inventory.gold`; WGSL-emitted handler does `atomicAdd(&gold_buf[slot], delta)`.
- Schema hash bump.

**Estimated:** 6-8 tasks; ~1.5 weeks.

---

# Phase 4 — 4c Record Memory (sketch)

**Prerequisite:** Phase 1 (`@per_entity_ring` needed for memory).

**Scope outline:**
- Port hand-written `memory: smallvec64<MemoryEvent>` on agents to DSL `@materialized @per_entity_ring(K = 64)` view.
- Delete hand-written smallvec handling.
- CPU consumers (GOAP, scoring if applicable) migrate to `state.views.memory.*`.
- `RecordMemory` event's CPU + GPU handlers via DSL view fold.
- Snapshot handshake for memory ring (if needed — may be purely cascade-internal).
- Schema hash bump.

**Estimated:** 4-6 tasks; ~1 week.

---

## Notes for the implementing engineer

- **`@materialized` infrastructure already exists** — used by `kin_fear`, `my_enemies`, `threat_level`, `pack_focus`, `rally_boost`, `engaged_with`. New annotations plug into the same pipeline.
- **Storage layout regression test**: for each new WGSL struct, add a Rust↔WGSL layout test mirroring `sim_cfg_layout.rs` — the byte offsets MUST match or GPU reads/writes silently corrupt.
- **Phase 2/3/4 will each add their own plan doc** after Phase 1 lands and the real emit API is known. Do NOT pre-plan them in detail here.
- **Subsystem (3) depends on this subsystem's Phase 3** for gold/standing GPU handlers. Without Phase 3, gold-transfer and standing-shift abilities silently no-op on the batch path. Prioritise Phase 3 after Phase 2 if ability-evaluation landing is time-sensitive.
- **Current pre-existing engine test failures** (not to debug): `rng::tests::per_agent_golden_value`, `dragon_attacks_all`, `parity_log_is_byte_identical_to_baseline`. These predate this plan and are out of scope.

## Open questions for Phase 2/3/4 (to resolve when their plans are drafted)

- **Dispatch-table emission**: where does the per-event-kind "which handlers fire" manifest live on the GPU path? Probably extends the existing GPU dispatch manifest emitted by `emit_physics_wgsl::emit_physics_dispatcher_wgsl`. Confirm during Phase 2.
- **Snapshot handshake scope**: `snapshot()` today reads agents + events + SimCfg.tick. Phase 3 adds gold readback; Phase 4 possibly adds memory readback. Evaluate memory cost at N=100k for memory ring (64 × 24B × 100k ≈ 150 MB — may push back on K=64).
- **Chronicle narrative-text audit**: the 8 chronicle rules haven't been individually audited for structured-vs-narrative. Phase 2 opens with that audit.
- **Gold snapshot API**: should `GpuSnapshot` grow a `gold: Vec<i32>` field, or should gold be accessed via a separate `read_gold()` method on `GpuBackend`? Influences Phase 3's spec.
