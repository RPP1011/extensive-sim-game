#![cfg(feature = "gpu")]

use engine::backend::SimBackend;
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

#[test]
fn snapshots_do_not_drop_or_duplicate_events() {
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(64, 0xDEAD_BEEF);
    for i in 0..8 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: if i % 2 == 0 {
                    CreatureType::Human
                } else {
                    CreatureType::Wolf
                },
                pos: Vec3::new((i as f32) * 2.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup to pay shader-compile cost.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Empty snapshot (first call, no batch yet).
    let _empty = gpu.snapshot(&mut state).unwrap();

    // Run a batch, snapshot. This snapshot is the "swap" — front is
    // still empty (nothing was kicked into it before). The returned
    // snap is the previous-call's empty.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let _swap = gpu.snapshot(&mut state).unwrap();

    // Run another batch. This call takes the snapshot that was
    // kicked during the previous snapshot() call — so snap_a
    // reflects tick state right after the FIRST batch.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let snap_a = gpu.snapshot(&mut state).unwrap();

    // Run a third batch + snapshot — snap_b reflects state right
    // after the SECOND batch.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let snap_b = gpu.snapshot(&mut state).unwrap();

    // Ticks must be monotonic.
    assert!(
        snap_b.tick > snap_a.tick,
        "snap_b.tick ({}) must be > snap_a.tick ({})",
        snap_b.tick,
        snap_a.tick
    );

    // Events from snap_a and snap_b should be disjoint — snap_a
    // reflects events from the first 10-tick batch, snap_b reflects
    // events from the second 10-tick batch. The batch events ring is
    // reset at the top of each `step_batch`, so their tick ranges
    // don't overlap. Compare by (tick, kind, payload) tuple.
    for ea in snap_a.events_since_last.iter() {
        for eb in snap_b.events_since_last.iter() {
            assert_ne!(
                (ea.tick, ea.kind, ea.payload),
                (eb.tick, eb.kind, eb.payload),
                "same event (tick={}, kind={}, payload={:?}) appears in two consecutive snapshots",
                ea.tick,
                ea.kind,
                ea.payload,
            );
        }
    }
}

/// C1 regression test: the snapshot's `events_since_last` must cover
/// events from MULTIPLE ticks of a multi-tick batch — not just the
/// last tick. Before the C1 fix `step_batch` cleared
/// `apply_event_ring.tail` at the top of every tick (needed for
/// cascade-seed correctness) and `snapshot()` read from that same
/// ring, so N-1 of N ticks' events were dropped on a batch snapshot.
///
/// This test spawns a combat-guaranteed pair and compares the record
/// counts from a 1-tick batch vs. a 5-tick batch. After the fix the
/// 5-tick batch's snapshot must contain materially more records than
/// the 1-tick batch (both would produce ~last-tick's records pre-fix).
///
/// Note: asserting on `HashSet<tick>.len() > 1` doesn't work here
/// because the resident apply_actions kernel's cfg uniform is uploaded
/// once per batch (plan Open Question #1 — GPU-side tick/rng advance
/// deferred), so every tick's emitted records carry the same tick tag
/// inside a batch. The record-count comparison is a sharper test of
/// the C1 fix anyway: it directly measures "events accumulate across
/// all ticks" vs. "events are overwritten each tick".
#[test]
fn snapshot_events_cover_multiple_ticks() {
    // Baseline: one-tick batch snapshot. Before the C1 fix a 5-tick
    // batch would produce the SAME count as a 1-tick batch (both
    // dominated by the last tick's contribution). After the fix the
    // 5-tick batch should carry ~5x more records.
    let baseline = run_fight_and_snapshot(1);
    let multi = run_fight_and_snapshot(5);

    assert!(
        baseline > 0,
        "baseline (1-tick) snapshot produced 0 events — combat pair \
         precondition failed; check unit stats / engagement range"
    );
    assert!(
        multi > baseline,
        "C1 regression: 5-tick batch snapshot produced {multi} records \
         but 1-tick baseline produced {baseline}. Expected strictly more \
         from a 5-tick batch (events accumulate across all ticks). Before \
         the C1 fix `step_batch` cleared `apply_event_ring.tail` per tick \
         and `snapshot()` read from that ring, so N-1 of N ticks' events \
         were dropped and the count would match the baseline."
    );
    // Stronger: expect at least 2x the baseline from 5x the ticks
    // (conservative — real combat typically produces 3-5x).
    assert!(
        multi >= baseline * 2,
        "5-tick batch snapshot produced only {multi} records vs. baseline \
         {baseline} — expected at least 2x as the batch ring should \
         accumulate across all 5 ticks."
    );
}

/// Phase 3 Task 3.5 regression: `snapshot()` reads `gold_buf` back into
/// `SimState.cold_inventory`. With no gold-transfer events firing during
/// the batch, the round-trip (upload on resident init → atomic kernel
/// writes that never fire → readback → merge) must preserve the initial
/// gold value exactly.
///
/// Task 3.6 covers the mutation case (EffectGoldTransfer end-to-end). If
/// this test fails, the readback wire-up or the `cold_inventory_mut`
/// merge is broken, not the atomic kernels.
#[test]
fn snapshot_merges_gold_into_state() {
    use engine::ids::AgentId;

    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(4, 0xD0_CA);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();

    // Seed CPU-side gold BEFORE ensure_resident_init uploads
    // cold_inventory into gold_buf.
    let id = AgentId::new(1).unwrap();
    let mut inv = state.agent_inventory(id).unwrap_or_default();
    inv.gold = 100;
    state.set_agent_inventory(id, inv);

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup — cold_inventory uploaded to gold_buf on first
    // ensure_resident_init.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    // Batch with no gold-transfer events — gold_buf should round-trip
    // the initial 100 unchanged.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 3);

    // Triple-snapshot dance (matches the existing tests in this file:
    // first call returns empty, second kicks, third returns the batch).
    let _ = gpu.snapshot(&mut state).unwrap();
    let _ = gpu.snapshot(&mut state).unwrap();
    let _ = gpu.snapshot(&mut state).unwrap();

    // gold_buf is read back into state.cold_inventory — observer sees
    // unchanged 100.
    let cur = state.agent_inventory(id).unwrap();
    assert_eq!(
        cur.gold, 100,
        "gold should round-trip through gold_buf unchanged when no \
         EffectGoldTransfer event fires during the batch"
    );
}

/// Task #79 SP-5 regression: `snapshot()` reads standing_storage back
/// into `state.views.standing`. With no EffectStandingDelta events
/// firing during the batch, the round-trip (upload on resident init →
/// real fold body that never fires → readback → merge) must preserve
/// the initial standing value exactly.
///
/// SP-6 covers the mutation case end-to-end. If this test fails, the
/// snapshot readback wire-up (SP-5) or upload_from_cpu / readback_into_cpu
/// (SP-2) is broken, not the find-or-evict fold body (SP-4).
#[test]
fn snapshot_merges_standing_into_state() {
    use engine::ids::AgentId;

    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(8, 0xD0_CA);
    for i in 0..4 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }

    // Seed CPU-side standing BEFORE ensure_resident_init uploads
    // state.views.standing into standing_records_buf / counts.
    let a = AgentId::new(1).unwrap();
    let b = AgentId::new(2).unwrap();
    state.views.standing.adjust(a, b, 50, 0);

    // Sanity: CPU adjust landed.
    assert_eq!(state.views.standing.get(a, b), 50);

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup — state.views.standing uploaded to GPU on first
    // ensure_resident_init.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Batch with no standing-delta events — round-trip should
    // preserve the initial 50 unchanged. The humans at the seeded
    // positions may produce incidental combat events, but nothing
    // emits EffectStandingDelta in base physics.sim (modify_standing
    // is emitted only by gameplay code; no wolves in this fixture
    // so no death events to trigger it either).
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 3);

    // Triple-snapshot dance.
    let _ = gpu.snapshot(&mut state).unwrap();
    let _ = gpu.snapshot(&mut state).unwrap();
    let _ = gpu.snapshot(&mut state).unwrap();

    // state.views.standing is rehydrated from the GPU — observer sees
    // unchanged 50.
    assert_eq!(
        state.views.standing.get(a, b),
        50,
        "standing should round-trip through standing_storage unchanged \
         when no EffectStandingDelta event fires during the batch",
    );
    // Symmetry preserved after round-trip.
    assert_eq!(
        state.views.standing.get(b, a),
        50,
        "symmetry: get(b,a) == get(a,b) after readback",
    );
}

/// Subsystem 2 Phase 4 PR-6 regression: `snapshot()` reads
/// memory_storage back into `state.views.memory`. With no
/// `RecordMemory` events firing during the batch, the round-trip
/// (upload on resident init → real ring-push body that never fires
/// → readback → merge) must preserve the initial memory ring state
/// exactly.
#[test]
fn snapshot_merges_memory_into_state() {
    use engine::generated::views::memory::MemoryEntry;
    use engine::ids::AgentId;

    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(8, 0xD0_CA);
    for i in 0..4 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }

    // Seed CPU-side memory BEFORE ensure_resident_init uploads
    // state.views.memory into memory_records_buf / cursors.
    let observer = AgentId::new(1).unwrap();
    let src = AgentId::new(3).unwrap();
    state.views.memory.push(
        observer.raw(),
        MemoryEntry {
            source: src.raw(),
            value: 1.0,
            anchor_tick: 7,
        },
    );

    // Sanity: CPU push landed.
    assert_eq!(state.views.memory.cursor(observer), 1);

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup — state.views.memory uploaded to GPU on first
    // ensure_resident_init.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Batch with no RecordMemory events — round-trip should preserve
    // the seeded entry. Nothing in base physics.sim emits
    // `RecordMemory` (it's gameplay-level — `Announce` macro's
    // fan-out, etc.); no humans announce in this fixture.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 3);

    // Triple-snapshot dance (the double-buffer gives observers the
    // previous tick's data; three calls guarantee we see the post-
    // batch state).
    let _ = gpu.snapshot(&mut state).unwrap();
    let _ = gpu.snapshot(&mut state).unwrap();
    let _ = gpu.snapshot(&mut state).unwrap();

    // state.views.memory is rehydrated from the GPU — observer sees
    // the unchanged seeded entry.
    assert_eq!(
        state.views.memory.cursor(observer),
        1,
        "cursor should round-trip through memory_storage unchanged",
    );
    let row = state
        .views
        .memory
        .entries(observer)
        .expect("owner row present after readback");
    assert_eq!(row[0].source, src.raw(), "source preserved");
    assert_eq!(row[0].anchor_tick, 7, "anchor_tick preserved");
}

fn run_fight_and_snapshot(n_ticks: u32) -> usize {
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(8, 0xC1_F1_A8);
    // Spawn a pair that will definitely fight: human at (0,0,0),
    // wolf at (1,0,0) — within engagement_range. Both full HP so
    // they exchange attacks for several ticks.
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .unwrap();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup so shader compile doesn't mask a timing-dependent bug.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Prime the double-buffered snapshot: the first `snapshot()` call
    // after `step_batch(n)` returns the front buffer (still empty)
    // and kicks the copy into the back buffer.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, n_ticks);
    let _prime = gpu.snapshot(&mut state).unwrap();

    // Second batch + snapshot — this snapshot's `events_since_last`
    // reflects the batch ring tail captured during the previous
    // snapshot call, which was taken after the first n_ticks batch.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, n_ticks);
    let snap = gpu.snapshot(&mut state).unwrap();
    snap.events_since_last.len()
}
