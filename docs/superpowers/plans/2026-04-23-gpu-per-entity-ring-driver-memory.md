# GPU View-Storage Driver for `@per_entity_ring` — Subsystem 2 Phase 4 (Memory)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Follow the task-numbered checklists.

**Goal:** Port the per-agent `MemoryEvent` smallvec to a DSL `@materialized @per_entity_ring(K=64)` view and land the GPU driver that backs it. Parallels Task #79's `@symmetric_pair_topk` driver (commits `39921c09 → 945a3189`); same 7-task SP-1..SP-7 structure.

**Architecture:** Per-agent FIFO ring of fixed K=64 `MemoryEvent` records, with a per-agent u32 write cursor. On push, atomically increment cursor mod K and write at `ring[agent][cursor % K]`. Reads return K slots in cursor-relative order. No decay, no clamp, no canonicalisation — just FIFO with eviction.

**Tech Stack:** `crates/engine_gpu/src/` primarily — new `view_storage_per_entity_ring.rs`, minimal touches to `backend/resident_ctx.rs`, `physics.rs`, `lib.rs`, `snapshot.rs`, `cascade_resident.rs`. DSL view decl in `assets/sim/views.sim`.

**Binding slot budget:** 20 (records), 21 (cursors). Slots 17 = gold_buf, 18/19 = standing storage. Slot 22+ reserved for Subsystem 3.

**Shared-context reference:** `docs/GPU_WORKGRAPH.md`.

## Prior state inventory

- `MemoryEvent` at `crates/engine/src/state/agent_types.rs:84` — `{ source: AgentId, kind: u8, payload: u64, confidence_q8: u8, tick: u32 }`. Pod, 20 bytes (padded to 24).
- Hand-written CPU storage: `SimState.cold_memory: Vec<SmallVec<[MemoryEvent; 64]>>` at `crates/engine/src/state/mod.rs:117`. Consumers at lines 245, 343, 785, 790, 795, 1135.
- DSL compiler `@per_entity_ring` support already landed in Phase 1: CPU emit (task 1.6, commit `edc4c73c`), WGSL emit (task 1.8, commit `6176d21d`).
- GPU no-op stub for `state_record_memory` in `crates/engine_gpu/src/physics.rs` (grep to verify; roughly mirrors the pre-Task-#79 `state_adjust_standing` stub).
- Current `record_memory` physics rule in `assets/sim/physics.sim` — lowers to `state.record_memory(...)` on CPU; GPU stub is no-op today.
- CPU `cold_state_replay` at `crates/engine_gpu/src/cascade.rs` has an `Event::RecordMemory` arm — this stays for sync path (same load-bearing reason gold + standing arms stay: sync physics doesn't bind the GPU storage).

## Prior work to mirror

Task #79 (`@symmetric_pair_topk` driver). Differences from #79:
- **No clamp, no canonical key**: ring pushes go to `owner`'s ring only (no "store on both sides").
- **K=64 not K=8**: storage per agent is 64×24 = 1536 bytes + 4 byte cursor. At N=2048: ~3 MB. At N=100k: ~150 MB (non-trivial; accept per spec §4c).
- **Richer record shape**: 24-byte `MemoryEventGpu` with 5 fields (vs StandingEdge's 3). Byte layout must match CPU `MemoryEvent` exactly.
- **Cursor semantics**: per-agent u32 monotonic counter; slot = `cursor % K`; cursor never wraps to 0.

## File structure

### New
- `crates/engine_gpu/src/view_storage_per_entity_ring.rs` — the module.

### Modified
- `assets/sim/views.sim` — append `memory` view decl.
- `crates/engine/src/state/agent_types.rs` — `MemoryEvent` stays (used by consumers); the smallvec goes away.
- `crates/engine/src/state/mod.rs` — delete `cold_memory` field + accessors + methods; consumers migrate to `state.views.memory.*`.
- `crates/engine/src/schema_hash.rs` — remove `cold_memory` reference.
- `crates/engine_gpu/src/physics.rs` — new BGL slots 20/21; `has_memory_storage` flag on `build_physics_shader_with_chronicle`; real `state_record_memory` WGSL body; BG entries + cache key.
- `crates/engine_gpu/src/backend/resident_ctx.rs` — new `memory_storage` field.
- `crates/engine_gpu/src/cascade_resident.rs` — thread buffers through `run_batch_resident`.
- `crates/engine_gpu/src/lib.rs` — ensure_resident_init alloc + snapshot readback.

### Tests
- `crates/engine_gpu/tests/cold_state_memory.rs` (new) — kernel-level: seed `RecordMemory` event, dispatch `run_batch_resident`, read back ring, assert FIFO semantics + K-eviction.

---

## Task PR-1: Design GPU storage layout

Read-only audit — output a doc block at top of `view_storage_per_entity_ring.rs` (similar to Task #79's SP-1).

Decide:
- **Records buffer**: `array<MemoryEventGpu>` sized to `agent_cap * K = agent_cap * 64`. Flat, indexed by `owner * K + slot`.
- **Cursors buffer**: `array<atomic<u32>>` sized to `agent_cap`. Monotonic write cursor per owner; slot = `cursor % K`.
- **MemoryEventGpu byte layout**: 24 bytes (u32 source + u32 kind_pad + u64 payload + u32 conf_pad + u32 tick = 24 bytes after alignment). Must match CPU `MemoryEvent` byte-for-byte.

Commit (doc change inside the new module):
```
docs(engine_gpu): document @per_entity_ring GPU layout (Task PR-1)
```

---

## Task PR-2: `ViewStoragePerEntityRing` module scaffold

Mirror Task #79's SP-2:

```rust
pub struct ViewStoragePerEntityRing {
    pub records_buf: wgpu::Buffer,   // array<MemoryEventGpu>
    pub cursors_buf: wgpu::Buffer,   // array<atomic<u32>>
    pub agent_cap:   u32,
    pub k:           u32,            // K = 64 for memory
}
```

Impl:
- `new(device, agent_cap, k)` — create two storage buffers (records + cursors) with `STORAGE | COPY_SRC | COPY_DST`.
- `upload_from_cpu(&self, queue, cpu: &Memory)` — walk `state.cold_memory` OR the new `state.views.memory` (whichever the generated CPU struct is named post-regeneration), serialise each agent's ring + cursor into the flat GPU buffers.
- `readback_into_cpu(&self, device, queue, out: &mut Memory)` — copy records + cursors back, rebuild the CPU struct. FIFO order preserved: slot order in GPU buffer = raw; CPU reader iterates `(cursor - K + i) % K` for i=0..K to get chronological.
- Unit tests: alloc, upload, round-trip.

Commit:
```
feat(engine_gpu): ViewStoragePerEntityRing module — upload/readback, unwired
```

---

## Task PR-3: Allocate at ensure_resident_init

Mirror #79's SP-3. Add `memory_storage: Option<ViewStoragePerEntityRing>` + `memory_storage_cap: u32` to `ResidentPathContext`. Alloc/grow in `ensure_resident_init` from `state.views.memory`.

Commit:
```
feat(engine_gpu): allocate memory view storage (unwired)
```

---

## Task PR-4: Wire into resident physics

- BGL slots 20 (records) + 21 (cursors) on `bgl_entries_resident`.
- `has_memory_storage: bool` flag on `build_physics_shader_with_chronicle`. Resident passes `true`; sync passes `false`.
- When `true`, emit binding decls + real `state_record_memory(observer, source, kind, payload, confidence, tick)` body:

```wgsl
fn state_record_memory(
    observer: u32, source: u32,
    kind: u32, payload_lo: u32, payload_hi: u32,
    confidence: u32, tick: u32,
) {
    let s = slot_of(observer);
    if (s == 0xFFFFFFFFu) { return; }
    let idx = atomicAdd(&memory_cursors[s], 1u);
    let ring_slot = s * MEMORY_K + (idx % MEMORY_K);
    var r: MemoryEventGpu;
    r.source     = source;
    r.kind       = kind;
    r.payload_lo = payload_lo;
    r.payload_hi = payload_hi;
    r.confidence = confidence;
    r.tick       = tick;
    memory_records[ring_slot] = r;
}
```

- Extend `ResidentBgKey` with `memory_records` + `memory_cursors`. BG entries at slots 20/21. Add `memory_records_buf` / `memory_cursors_buf` params to `run_batch_resident`. Thread through `cascade_resident.rs` + `lib.rs:step_batch`.
- Update the physics_run_batch_resident_smoke test + chronicle_isolated_smoke test to pass test-local memory buffers.

Verify: full regression sweep, naga parse test.

Commit:
```
feat(engine_gpu): wire memory view storage into resident physics; DSL record_memory is GPU-native on batch
```

---

## Task PR-5: Declare `memory` view + migrate consumers

Parallel to Task 3.1 + 3.2 for standing:
- Append `@materialized @per_entity_ring(K = 64) view memory(observer) -> MemoryEventGpu { ... }` to `assets/sim/views.sim`.
- Run `cargo run --release --bin xtask -- compile-dsl`.
- Delete `cold_memory` field + `Vec<SmallVec<[MemoryEvent; 64]>>` from SimState.
- Migrate consumers (6 call sites from the initial grep) to `state.views.memory.*`.
- Update schema hash.

Commit:
```
refactor(engine): retire cold_memory SmallVec; migrate consumers to @materialized view memory
```

---

## Task PR-6: Snapshot readback

Add `memory_storage.readback_into_cpu(...)` call in `snapshot()` after gold + standing readbacks. Round-trip test in snapshot_double_buffer.rs.

Commit:
```
feat(engine_gpu): snapshot() reads memory_storage into state.views.memory
```

---

## Task PR-7: Integration test — kernel-level

Create `crates/engine_gpu/tests/cold_state_memory.rs`:
- Seed `Event::RecordMemory { observer: 1, source: 2, kind: 5, payload: 0xDEADBEEF, confidence_q8: 200, tick: 3 }`.
- Dispatch `run_batch_resident`.
- Read back memory_records for owner_id=1; assert one record at cursor-derived slot, payload + tick + source match.
- FIFO test: seed 65 RecordMemory events (K+1), assert oldest evicted.
- Cursor test: assert cursor_buf[owner] == number of events pushed.

GPU-only assertions; no CPU-parity comparison.

Commit:
```
test(engine_gpu): record_memory fires on resident physics kernel (Task PR-7)
```

---

## Closeout

- Full regression: `engine_gpu` + `dsl_compiler` + `engine`. Expected: no new failures beyond the 3 pre-existing.
- Perf sanity: run `chronicle_batch_perf_n100k` (our N=100k batch canary). Target: within 10% of the 309 ms/tick baseline. The memory fold adds per-event atomic + ring write, which at 100k × 50 ticks may nudge the number up — if >15%, audit.

## Test-scale guideline

- Integration tests default to N=20000 agents.
- Kernel-level microfixtures keep small N.
- GPU-only assertions throughout. No sync-path parity comparisons.

## Notes

- **Byte-layout fragility**: `MemoryEventGpu` struct must match `MemoryEvent` byte-for-byte. Keep the layout documented at the top of `view_storage_per_entity_ring.rs`.
- **Eviction semantics**: FIFO — oldest evicted when K reached. Since cursor is monotonic u32 (never wraps to 0 within a session), oldest is always slot `cursor % K` at push time. Readers handle the wrap by iterating `(cursor - K + i) % K` for chronological order.
- **Sync-path gap**: cold_state_replay's `EffectRecordMemory` arm (or equivalent) stays in place for sync, same load-bearing pattern as gold + standing. Removal requires sync BGL wiring (separate follow-up task).
