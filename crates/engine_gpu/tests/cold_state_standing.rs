//! Task 3.7 — modify_standing on the batch path (GPU view-storage driver missing).
//!
//! This test is **`#[ignore]`d** because the GPU driver for
//! `@symmetric_pair_topk` view storage does not yet exist. Phase 1
//! (commits `e828bcfb`, `6fb3caef`, `b7e75326`) landed the CPU emit
//! plus the WGSL fold kernel source, but no caller in `engine_gpu`
//! allocates per-agent StandingEdge storage, dispatches the fold
//! kernel during `step_batch`, or reads the storage back into
//! `state.views.standing` on `snapshot()`.
//!
//! Currently on the batch path, the DSL lowering for
//! `agents.adjust_standing(a, b, delta)` at
//! `crates/dsl_compiler/src/emit_physics_wgsl.rs:1369-1374` emits
//! `state_adjust_standing(a, b, delta)` — a no-op stub at
//! `crates/engine_gpu/src/physics.rs:1013`. Standing mutations flow
//! through the CPU `cold_state_replay` path
//! (`crates/engine_gpu/src/cascade.rs` `Event::EffectStandingDelta`
//! arm) because Task 3.8 deliberately left that arm in place until
//! the GPU view-storage driver lands.
//!
//! What this test WILL do once unblocked:
//!
//!   1. Seed `Event::EffectStandingDelta { a: 1, b: 2, delta: 42,
//!      tick: 0 }` into the resident physics kernel's input ring
//!      (same harness shape as `cold_state_gold_transfer.rs`).
//!   2. Dispatch `run_batch_resident` with the GPU standing storage
//!      bound (Task 3.7's prerequisite: allocate + bind storage).
//!   3. Read back the per-slot StandingEdge array via
//!      `readback_typed`.
//!   4. Assert the pair-symmetric delta is applied exactly once
//!      (`standing(a,b) == standing(b,a) == 42`) and clamped to
//!      `[-1000, 1000]` when the test saturates.
//!
//! What the Task 3.7 *re-enablement* requires:
//!
//!   * Allocate GPU storage for the standing view in
//!     `ResidentPathContext` (one `StandingEdge` array + count per
//!     agent slot, matching the CPU-side struct layout in
//!     `crates/engine/src/generated/views/standing.rs`).
//!   * Bind those buffers into the resident physics pipeline at new
//!     binding slots (18 / 19 / …) parallel to how `gold_buf`
//!     landed at 17 in Task 3.4.
//!   * Replace the no-op `state_adjust_standing` WGSL stub at
//!     `crates/engine_gpu/src/physics.rs:1013` with a body that
//!     calls the fold kernel (or inlines the find-or-evict-else-drop
//!     logic the Phase 1 emitter produced).
//!   * Read back the GPU standing storage in `snapshot()` and
//!     replace the CPU view's contents — or merge, depending on
//!     whether the sync path is still writing.
//!   * Remove the `Event::EffectStandingDelta` arm from
//!     `cold_state_replay` at that point (Task 3.8's deferred
//!     standing scope).
//!
//! This is a non-trivial chunk of engine-core work that the Phase 3
//! plan underestimated; tracking it as a separate follow-up avoids
//! wedging the whole cold-state subsystem on one big lift.

#![cfg(feature = "gpu")]

#[test]
#[ignore = "pending GPU view-storage driver for @symmetric_pair_topk — see file-level doc"]
fn modify_standing_applies_via_batch_path() {
    // Intentionally empty. When the GPU view-storage driver lands,
    // replace this body with a cold_state_gold_transfer.rs-shaped
    // harness that seeds EffectStandingDelta, dispatches the
    // resident physics kernel with standing storage bound, reads
    // back the StandingEdge array, and asserts the symmetric pair
    // delta + clamp invariants.
    unimplemented!("re-enable after GPU view-storage driver for symmetric_pair_topk lands");
}
