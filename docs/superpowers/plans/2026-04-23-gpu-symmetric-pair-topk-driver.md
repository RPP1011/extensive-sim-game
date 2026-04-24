# GPU View-Storage Driver for `@symmetric_pair_topk`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Land the engine_gpu infrastructure to actually store + fold + read-back `@symmetric_pair_topk` views on the resident path. Today Phase 1 of the cold-state plan emitted the WGSL fold kernel source but nothing in engine_gpu dispatches it or allocates storage. The `standing` view (Task 3.1) is the first production consumer; this driver unblocks it.

**Architecture:** New resident-path side storage owned by `ResidentPathContext`: per-agent `[StandingEdge; K=8]` arrays + counts + "dirty since last snapshot" watermark. Bound into the resident physics pipeline at new binding slots; the WGSL fold kernel (Phase 1 emitter output) is invoked from `state_adjust_standing` stub. Snapshot reads storage back and merges into `state.views.standing` on CPU.

**Tech Stack:** `crates/engine_gpu/src/` primarily — new `view_storage_symmetric_pair.rs` module, plus minimal touches to `backend/resident_ctx.rs`, `physics.rs`, `lib.rs`, `snapshot.rs`, `cascade_resident.rs`.

**Binding slot budget for this plan:** 18 (records), 19 (counts). Slot 17 = gold_buf. Slot 20+ reserved for subsystem 3.

**Shared-context reference:** `docs/GPU_WORKGRAPH.md` — read before starting.

---

## Prior state (inventory — do not re-do this work)

- `Standing` struct exists at `crates/engine/src/generated/views/standing.rs` (Task 3.1, commit `31c46e96`). Public API: `K=8`, `get(a,b) -> i32`, `adjust(a,b,delta,tick) -> i32`, `fold_event(&event, tick)`. `StandingEdge { other: u32, value: i32, anchor_tick: u32 }`.
- WGSL fold kernel source emitted by `crates/dsl_compiler/src/emit_view_wgsl.rs:900:emit_symmetric_pair_topk_fold_wgsl` for the `standing` view. Grep the generated source if you need to see its shape — it computes canonical `(min,max)` keys, checks 8 slots per owner, find-or-evict-lowest-|value|.
- GPU WGSL stub `state_adjust_standing(a, b, delta: i32)` at `crates/engine_gpu/src/physics.rs:1013` — currently a no-op. DSL `agents.adjust_standing(a, b, delta)` lowers here (via `crates/dsl_compiler/src/emit_physics_wgsl.rs:1369-1374`).
- CPU `cold_state_replay`'s `EffectStandingDelta` arm at `crates/engine_gpu/src/cascade.rs:647` still routes standing via the hand-written fold — kept load-bearing for sync. This plan's final task removes the arm.

## Prior work to mirror

Task 3.3-3.5 (`gold_buf`) is the closest parallel: side buffer owned by ResidentPathContext, bound at binding 17 on resident BGL, real WGSL stub via `has_gold_buf` flag, readback into `state.cold_inventory`. Differences:
- Standing storage is per-agent-array, not per-agent-scalar. Layout + readback are richer.
- The "mutation" isn't a raw `atomicAdd` — it calls the Phase-1-emitted fold WGSL (find-or-evict), which itself writes multiple fields per slot.

---

## File structure

### New
- `crates/engine_gpu/src/view_storage_symmetric_pair.rs` — owns the resident storage buffers, allocation logic, readback parser.

### Modified
- `crates/engine_gpu/src/backend/resident_ctx.rs` — new `standing_storage: Option<ViewStorageSymmetricPair>` field.
- `crates/engine_gpu/src/physics.rs` — new BGL slots 18/19; new `has_standing_storage: bool` flag on `build_physics_shader_with_chronicle` similar to `has_gold_buf`; real `state_adjust_standing` body when flag is true; BG entries + cache-key extension.
- `crates/engine_gpu/src/cascade_resident.rs` — pass `standing_storage` buffers through `run_batch_resident` signature.
- `crates/engine_gpu/src/lib.rs:ensure_resident_init` — allocate + upload from `state.views.standing`. `lib.rs:step_batch` — pass standing buffers into the cascade call.
- `crates/engine_gpu/src/lib.rs:snapshot` — read back standing storage, rebuild `state.views.standing`.
- `crates/engine_gpu/src/snapshot.rs` — optional: add standing readback helper if the logic is nontrivial.
- `crates/engine_gpu/src/cascade.rs:cold_state_replay` — remove `Event::EffectStandingDelta` arm (final task).
- `crates/engine_gpu/tests/cold_state_standing.rs` — remove `#[ignore]`, fill in the real test body.

### Reference (do not modify)
- `crates/engine/src/generated/views/standing.rs` — canonical CPU layout. Buffer layout must match byte-for-byte.
- `crates/dsl_compiler/src/emit_view_wgsl.rs:900-1050` — canonical WGSL fold kernel (Phase 1 emitter output).

---

## Task SP-1: Design the GPU storage layout

**Files:** Read-only audit. Output: a short doc block (~30-50 lines) added to `view_storage_symmetric_pair.rs` as its module docstring.

### Goal

Decide the exact GPU buffer layout so both the fold kernel (Phase-1-emitted WGSL) and the snapshot readback know the format.

### Steps

- [ ] **Step 1:** Read `crates/engine/src/generated/views/standing.rs` — note:
  - `StandingEdge` byte layout: `{ other: u32, value: i32, anchor_tick: u32 }` = 12 bytes.
  - `K = 8` slots per owner.
  - Per-owner count field (u32 or implicit via sentinel slots).

- [ ] **Step 2:** Read the Phase-1-emitted WGSL at `crates/dsl_compiler/src/emit_view_wgsl.rs:900-1050`. Note how the kernel references storage — bind slot names, atomic vs non-atomic, per-owner indexing.

- [ ] **Step 3:** Decide:
  - **Records buffer**: `array<StandingEdge>` sized to `agent_cap * K`. Flat, indexed by `owner * K + slot`. Total at N=2048: 2048 × 8 × 12 = 192 KB. Well under any GPU limit.
  - **Counts buffer**: `array<atomic<u32>>` sized to `agent_cap`. Each entry is the current population (0..=K). Needs atomics because multiple threads in the fold kernel may touch the same owner on the same tick.
  - **Canonical pair rule**: fold kernel writes into BOTH `a`'s and `b`'s slots (since `symmetric_pair_topk` is symmetric). Confirm by reading the emitted WGSL — if it only writes to canonical owner, the readback parser handles the symmetry; if it writes both, no extra work.

### Deliverable

A doc block at the top of `view_storage_symmetric_pair.rs` spelling out the binding layout + byte offsets. Sized for readability, not exhaustive.

---

## Task SP-2: `ViewStorageSymmetricPair` module scaffold

**Files:**
- Create: `crates/engine_gpu/src/view_storage_symmetric_pair.rs`

### Goal

New module with the storage struct, allocation, upload, readback. No pipeline wiring yet — that's SP-3 / SP-4.

### Steps

- [ ] **Step 1:** Create the file. Module shape:
  ```rust
  pub struct ViewStorageSymmetricPair {
      pub records_buf: wgpu::Buffer,   // array<StandingEdge>
      pub counts_buf:  wgpu::Buffer,   // array<atomic<u32>>
      pub agent_cap:   u32,
      pub k:           u32,            // K = 8 for standing
  }
  ```

- [ ] **Step 2:** Impl `new(device, agent_cap, k) -> Self` that creates the two buffers with `STORAGE | COPY_SRC | COPY_DST` usage, sized to `agent_cap * k * 12` and `agent_cap * 4`.

- [ ] **Step 3:** Impl `upload_from_cpu(&self, queue, cpu: &Standing)` that serialises the CPU `Standing` struct into the flat GPU arrays. Walk every owner slot 0..agent_cap, pack each owner's edges into contiguous slots, fill unused with sentinel (zero = unused since `other: 0` = AgentId sentinel).

- [ ] **Step 4:** Impl `readback_into_cpu(&self, device, queue, out: &mut Standing)` that copies records + counts back to CPU, rebuilds the CPU struct's internal representation. Use `readback_typed`.

- [ ] **Step 5:** Add unit tests at the bottom of the file — `new` allocates correctly, `upload_from_cpu` + `readback_into_cpu` round-trip with a canned fixture.

- [ ] **Step 6:** Wire the module into `crates/engine_gpu/src/lib.rs` via `mod view_storage_symmetric_pair;`.

- [ ] **Step 7:** Build + run unit tests. Commit.

```
feat(engine_gpu): ViewStorageSymmetricPair module — upload/readback, no pipeline wiring
```

---

## Task SP-3: Allocate + upload at ensure_resident_init

**Files:**
- Modify: `crates/engine_gpu/src/backend/resident_ctx.rs` — new field.
- Modify: `crates/engine_gpu/src/lib.rs:ensure_resident_init` — allocation block (mirror gold_buf pattern around the block we added in Task 3.3).

### Goal

`ensure_resident_init` now creates the standing storage from `state.views.standing` on first call or agent_cap grow.

### Steps

- [ ] **Step 1:** Add `standing_storage: Option<ViewStorageSymmetricPair>` + `standing_storage_cap: u32` to `ResidentPathContext`. Init to None/0.

- [ ] **Step 2:** In `ensure_resident_init`, mirror the `gold_buf` alloc/grow block. On (re)allocate, call `ViewStorageSymmetricPair::new` + `upload_from_cpu(&state.views.standing)`.

- [ ] **Step 3:** Build. No tests yet — storage is unwired into physics.

- [ ] **Step 4:** Commit.

```
feat(engine_gpu): allocate standing view storage (unwired)
```

---

## Task SP-4: Wire standing storage into resident physics

**Files:**
- Modify: `crates/engine_gpu/src/physics.rs` — BGL entries + BG entries + `has_standing_storage` flag + real `state_adjust_standing` WGSL body.
- Modify: `crates/engine_gpu/src/cascade_resident.rs` — thread buffers through `run_batch_resident` callsite.
- Modify: `crates/engine_gpu/src/lib.rs:step_batch` — pass standing buffers into the cascade call.

### Goal

`agents.adjust_standing(a, b, delta)` in physics DSL now lowers to a real WGSL body that calls the Phase-1-emitted fold logic against `standing_records_buf` + `standing_counts_buf`.

### Steps

- [ ] **Step 1:** Add BGL slots 18 (storage_rw) + 19 (storage_rw) to `bgl_entries_resident` at `crates/engine_gpu/src/physics.rs:1359-1380`.

- [ ] **Step 2:** Add `has_standing_storage: bool` to `build_physics_shader_with_chronicle` signature. Thread through to call sites. Resident builder passes `true`; sync builder passes `false`.

- [ ] **Step 3:** When `has_standing_storage = true`, emit:
  - Binding declarations for slots 18/19.
  - Real `state_adjust_standing(a: u32, b: u32, delta: i32)` body that implements the find-or-evict-else-drop logic. Mirror what `emit_symmetric_pair_topk_fold_wgsl` produces — or import it. The body canonicalises `(min(a,b), max(a,b))`, looks up the owner's slot row, finds a matching pair or an empty slot or evicts the lowest-|value| slot if `|delta|` is stronger, applies clamp `[-1000, 1000]`, writes back.

- [ ] **Step 4:** Extend `ResidentBgKey` with `standing_records: wgpu::Buffer` + `standing_counts: wgpu::Buffer`. Extend BG entries for slots 18/19.

- [ ] **Step 5:** Add `standing_records: &wgpu::Buffer` + `standing_counts: &wgpu::Buffer` params to `run_batch_resident`. Thread through `cascade_resident.rs::run_cascade_resident_with_iter_cap` and `run_cascade_resident`. Pass from `lib.rs:step_batch`.

- [ ] **Step 6:** Update the physics_run_batch_resident_smoke test to pass test-local standing buffers (mirror how gold_buf was added).

- [ ] **Step 7:** Build. Run regression:
  ```
  cargo build --release --features gpu -p engine_gpu 2>&1 | tail -10
  cargo test --release --features gpu -p engine_gpu --lib physics::tests::physics_resident_shader_parses_through_naga 2>&1 | tail -5
  cargo test --release --features gpu -p engine_gpu 2>&1 | tail -15
  ```

- [ ] **Step 8:** Commit.

```
feat(engine_gpu): wire standing view storage into resident physics; DSL adjust_standing is now GPU-native on batch path
```

---

## Task SP-5: Snapshot readback + `state.views.standing` merge

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs:snapshot` — add readback block after the gold_buf readback.

### Goal

Observers calling `snapshot(&mut state)` now see post-batch standing via `state.views.standing`. Mirrors Task 3.5's pattern for gold.

### Steps

- [ ] **Step 1:** In `snapshot()`, after the gold readback, add:
  ```rust
  if let Some(storage) = self.resident.standing_storage.as_ref() {
      storage.readback_into_cpu(&self.device, &self.queue, &mut state.views.standing)
          .map_err(crate::snapshot::SnapshotError::Ring)?;
  }
  ```
  Details depend on the Task SP-2 API — adapt the error conversion accordingly.

- [ ] **Step 2:** Extend the snapshot_double_buffer test with a standing round-trip: seed `state.views.standing.adjust(a, b, 50, 0)` on CPU, run `step_batch(3)` with no standing-delta events, snapshot, assert `state.views.standing.get(a, b) == 50` (round-trip).

- [ ] **Step 3:** Build + test. Commit.

```
feat(engine_gpu): snapshot() reads standing_storage into state.views.standing
```

---

## Task SP-6: End-to-end integration test — un-ignore cold_state_standing

**Files:**
- Modify: `crates/engine_gpu/tests/cold_state_standing.rs` — remove `#[ignore]`, fill in the real test body.

### Goal

The standing integration test from the Phase 3 skeleton becomes a real passing test.

### Steps

- [ ] **Step 1:** Mirror the shape of `cold_state_gold_transfer.rs` — seed an `Event::EffectStandingDelta { a, b, delta, tick }` event into the resident physics kernel's input ring, dispatch `run_batch_resident` with standing buffers bound, read back storage via `readback_typed`, assert `get(a,b) == delta`, assert symmetry `get(b,a) == get(a,b)`, assert clamp `[-1000, 1000]` by over-saturating with multiple deltas.

- [ ] **Step 2:** Remove the `#[ignore]` attributes on both tests in the file.

- [ ] **Step 3:** Run + commit.

```
test(engine_gpu): modify_standing fires on resident physics kernel (re-enabled)
```

---

## Task SP-7: Retire the `EffectStandingDelta` arm in `cold_state_replay`

**Files:**
- Modify: `crates/engine_gpu/src/cascade.rs:647` — delete the arm.

### Caveat

Same concern as Task 3.8's gold arm: if `cold_state_replay` is called from the SYNC path, and sync standing stubs are no-op, this removal breaks sync.

Verify: is sync's `state_adjust_standing` WGSL stub still no-op after this plan? Yes — this plan only flags `has_standing_storage=true` on the resident builder. Sync stays no-op.

Conclusion: **do NOT remove the arm if sync's stub is no-op.** Only remove if sync also binds standing storage (a separate expansion).

### Decision

Either:
- (A) Plan ends at SP-6. Arm stays, with an expanded doc comment explaining the sync gap.
- (B) Expand scope to wire sync too: extra BGL entry on sync, `has_standing_storage=true` on sync builder, sync bind group carries standing buffers. Then delete the arm.

**Recommend (A).** Sync parity is a separate concern; this plan focuses on the batch path. An "SP-7-follow-up: wire standing on sync BGL" task captures the deferred work.

### Steps (option A)

- [ ] **Step 1:** Update the `EffectStandingDelta` arm doc comment at `crates/engine_gpu/src/cascade.rs:647` to mirror the gold arm's load-bearing note: "stays because sync `state_adjust_standing` is no-op; batch path uses standing_storage via Task SP-4."

- [ ] **Step 2:** Commit.

```
docs(engine_gpu): document standing arm in cold_state_replay is load-bearing for sync
```

---

## Closeout

- [ ] Run full regression: `cargo test --release --features gpu -p engine_gpu` + `-p dsl_compiler` + `-p engine`. Expected: no new failures beyond the 3 pre-existing engine failures.
- [ ] Perf sweep: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 100`. Record the N=20k batch µs/tick as the new reference — extrapolated from Phase 3's N=2048=6093 µs/tick, expect roughly 60-100 ms/tick at N=20k depending on how well the spatial hash scales. Call out the actual number; use it as the Phase-3/SP baseline going forward. Fold-kernel invocations per physics dispatch add per-tick overhead — flag a regression if N=20k is more than 15% slower than a pre-SP baseline at the same N.

## Test-scale guideline for this plan

- **New integration tests**: default to `N=20000` agents unless the test genuinely needs smaller (e.g. hand-seeded fixture pair). Use `SimState::new(20_000, seed)` + dense spawn.
- **GPU-only assertions**: don't compare against sync-path state unless strictly necessary; test the batch path's contract directly.
- **Existing smaller-N tests** (gold transfer at N=8 etc.) stay as-is — they're kernel-level fixtures, not workload benchmarks.

## Notes

- **Byte-layout fragility**: If `StandingEdge` ever changes (another field, different width), the GPU side needs matching updates. Keep the layout documented at the top of `view_storage_symmetric_pair.rs`.
- **K change-resistance**: If someone bumps K in the DSL from 8 to 16, storage sizes + per-owner indexing need updating too. The kernel WGSL uses `const K: u32 = 8u;` today; make sure the resident shader's K matches the view's K.
- **Concurrent fold correctness**: If two physics threads hit the same owner's row in the same tick, atomics on the count + pointwise `atomicCompareExchangeWeak` on slot reads are necessary to avoid torn writes. The Phase 1 emitter's TODO(phase-3) markers flag this. Audit before SP-4 ships.
