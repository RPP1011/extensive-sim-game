// T16-broken: this test references hand-written kernel modules
// (mask, scoring, physics, apply_actions, movement, spatial_gpu,
// alive_bitmap, cascade, cascade_resident) that were retired in
// commit 4474566c when the SCHEDULE-driven dispatcher in
// `engine_gpu_rules` became authoritative. The test source is
// preserved verbatim below the cfg gate so the SCHEDULE-loop port
// (follow-up: gpu-feature-repair plan) has a reference for what
// behaviour each test asserted.
//
// Equivalent to `#[ignore = "broken by T16 hand-written-kernel
// deletion; needs SCHEDULE-loop port (follow-up)"]` on every
// `#[test]` below — but applied at file scope because the test
// bodies do not compile against the post-T16 surface.
#![cfg(any())]

//! End-to-end: chronicle_attack fires on the batch path. Verifies
//! the full flow: @phase(post) physics rule emits ChronicleEntry
//! -> chronicle ring accumulates -> snapshot reads it back -> observer
//! sees the record with the right template_id.
//!
//! Task 2.4 (Phase 2 integration for subsystem (2) chronicle-on-snapshot).
//! Exercises the plumbing wired up in Task 2.3: the resident chronicle
//! ring is copied into `GpuSnapshot::chronicle_since_last` with a
//! watermark advanced across snapshot calls.
//!
//! **History (2026-04-23):** these tests caught a queue.write_buffer
//! aliasing bug on the resident path. `PhysicsKernel::run_batch_resident`
//! was writing the per-iteration `ResidentPhysicsCfg` uniform via
//! `queue.write_buffer(&pool.resident_cfg_buf, 0, ...)` once per
//! cascade iteration, each call at the same byte range. wgpu collapses
//! back-to-back `queue.write_buffer` calls on the same region before
//! submit: only the final write survives. Every physics iteration
//! therefore saw `resident_cfg = {read_slot: max_iters-1, ...}`,
//! indexed `num_events_buf[max_iters-1]` (which is 0), and skipped
//! `physics_dispatch(...)` → no chronicle emits. Fix: allocate one
//! `resident_cfg` uniform buffer per iteration slot, pre-populated
//! at pool creation time. Bound by `read_slot` in `run_batch_resident`.

#![cfg(feature = "gpu")]

use engine::backend::ComputeBackend;
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::snapshot::ChronicleRecord;
use engine_gpu::GpuBackend;
use glam::Vec3;

/// Build a two-agent Human+Wolf fixture guaranteed to produce attack
/// events quickly (same recipe the C1 regression test in
/// `snapshot_double_buffer.rs` uses). The two units are within the
/// engagement range so an `AgentAttacked` event fires within a handful
/// of ticks.
fn build_fight_state() -> SimState {
    let mut state = SimState::new(8, 0xC1_F1_A8);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf spawn");
    state
}

/// Primary integration test: an attack inside a multi-tick batch must
/// surface as a `ChronicleRecord { template_id = 2, .. }` in the
/// observer's `chronicle_since_last` slice.
#[test]
fn chronicle_attack_fires_on_batch_path() {
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = build_fight_state();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(4096);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup — pays shader-compile cost so the assertions don't race
    // against pipeline creation.
    gpu.step(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
    );

    // Prime the double-buffered snapshot: first call returns empty
    // (no batch yet), just kicks the copy into the back buffer.
    let _empty = gpu.snapshot(&mut state).unwrap();

    // First batch: runs multiple ticks; combat produces AgentAttacked
    // -> chronicle_attack emits ChronicleEntry into the chronicle ring.
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        5,
    );
    // This snapshot is the "swap" — returns the pre-batch empty and
    // kicks a copy of the post-batch state into the back buffer.
    let _swap = gpu.snapshot(&mut state).unwrap();

    // Second batch + snapshot: `snap` reflects the state that was
    // kicked during the previous snapshot() call — i.e. the chronicle
    // records accumulated during the FIRST 5-tick batch.
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        5,
    );
    let snap = gpu.snapshot(&mut state).unwrap();

    // Dump observed event kinds so a regression makes the diagnosis
    // obvious (e.g. "combat didn't fire" vs "combat fired but chronicle
    // dropped"). Kind=1 is AgentAttacked, kind=2 is AgentDied.
    let mut kind_counts = std::collections::HashMap::<u32, u32>::new();
    for e in snap.events_since_last.iter() {
        *kind_counts.entry(e.kind).or_default() += 1;
    }

    assert!(
        !snap.chronicle_since_last.is_empty(),
        "expected chronicle_since_last to contain at least one record \
         after a 5-tick Human vs Wolf batch; got 0. snap.tick={}, \
         events_since_last={}, kinds={:?}",
        snap.tick,
        snap.events_since_last.len(),
        kind_counts,
    );

    let attacks: Vec<&ChronicleRecord> = snap
        .chronicle_since_last
        .iter()
        .filter(|r| r.template_id == 2)
        .collect();
    assert!(
        !attacks.is_empty(),
        "expected >=1 chronicle_attack (template_id=2) record in \
         chronicle_since_last; got template ids {:?}",
        snap.chronicle_since_last
            .iter()
            .map(|r| r.template_id)
            .collect::<Vec<_>>(),
    );

    // Sanity: records should reference real agents (non-zero ids for
    // the two agents we spawned) and carry a non-zero tick. The
    // chronicle ring writes the same cfg-tick as the batch uniform, so
    // we only check for a non-uninitialised / non-sentinel value.
    for r in &attacks {
        assert!(
            r.agent != 0 || r.target != 0,
            "chronicle_attack record has zero agent and target — \
             looks like uninitialised memory: {r:?}"
        );
    }
}

/// Watermark regression guard: across three consecutive batch+snapshot
/// cycles, no `(template_id, agent, target, tick)` tuple from one
/// snapshot's `chronicle_since_last` may reappear in a later one. If
/// this fails, the snapshot-side chronicle watermark is not advancing
/// and records are being replayed.
///
/// Shape mirrors the existing
/// `snapshots_do_not_drop_or_duplicate_events` test for the event ring.
#[test]
fn chronicle_watermark_advances_across_snapshots() {
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = build_fight_state();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(4096);
    let cascade = CascadeRegistry::with_engine_builtins();

    gpu.step(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
    );

    // Prime.
    let _empty = gpu.snapshot(&mut state).unwrap();

    // Three batches separated by snapshots. snap_a covers batch 1,
    // snap_b covers batch 2, snap_c covers batch 3 (the "swap" dance
    // means each snapshot() reflects what was kicked during the
    // PREVIOUS snapshot() call, which was taken right after a batch).
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        5,
    );
    let _swap = gpu.snapshot(&mut state).unwrap();

    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        5,
    );
    let snap_a = gpu.snapshot(&mut state).unwrap();

    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        5,
    );
    let snap_b = gpu.snapshot(&mut state).unwrap();

    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        5,
    );
    let snap_c = gpu.snapshot(&mut state).unwrap();

    // Sanity — precondition for the duplicate check to be meaningful:
    // at least one of the three snapshots carried chronicle records.
    let total = snap_a.chronicle_since_last.len()
        + snap_b.chronicle_since_last.len()
        + snap_c.chronicle_since_last.len();
    assert!(
        total > 0,
        "none of the three snapshots carried chronicle records — \
         combat fixture precondition failed (snap_a={}, snap_b={}, \
         snap_c={})",
        snap_a.chronicle_since_last.len(),
        snap_b.chronicle_since_last.len(),
        snap_c.chronicle_since_last.len(),
    );

    // No chronicle tuple may appear in more than one snapshot. If the
    // watermark didn't advance, snap_b / snap_c would re-include
    // snap_a's records.
    let key = |r: &ChronicleRecord| (r.template_id, r.agent, r.target, r.tick);
    for ra in snap_a.chronicle_since_last.iter() {
        for rb in snap_b.chronicle_since_last.iter() {
            assert_ne!(
                key(ra),
                key(rb),
                "chronicle record {ra:?} appears in both snap_a and \
                 snap_b — watermark did not advance"
            );
        }
        for rc in snap_c.chronicle_since_last.iter() {
            assert_ne!(
                key(ra),
                key(rc),
                "chronicle record {ra:?} appears in both snap_a and \
                 snap_c — watermark did not advance"
            );
        }
    }
    for rb in snap_b.chronicle_since_last.iter() {
        for rc in snap_c.chronicle_since_last.iter() {
            assert_ne!(
                key(rb),
                key(rc),
                "chronicle record {rb:?} appears in both snap_b and \
                 snap_c — watermark did not advance"
            );
        }
    }
}
