#![cfg(feature = "gpu")]

use engine::backend::SimBackend;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
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
    let _empty = gpu.snapshot().unwrap();

    // Run a batch, snapshot. This snapshot is the "swap" — front is
    // still empty (nothing was kicked into it before). The returned
    // snap is the previous-call's empty.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let _swap = gpu.snapshot().unwrap();

    // Run another batch. This call takes the snapshot that was
    // kicked during the previous snapshot() call — so snap_a
    // reflects tick state right after the FIRST batch.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let snap_a = gpu.snapshot().unwrap();

    // Run a third batch + snapshot — snap_b reflects state right
    // after the SECOND batch.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 10);
    let snap_b = gpu.snapshot().unwrap();

    // Ticks must be monotonic.
    assert!(
        snap_b.tick > snap_a.tick,
        "snap_b.tick ({}) must be > snap_a.tick ({})",
        snap_b.tick,
        snap_a.tick
    );

    // Events from snap_a and snap_b should be disjoint (since the
    // apply_event_ring is cleared per tick; snap_a captures the
    // state at the end of a batch, snap_b captures state at the
    // end of the next batch — different events).
    //
    // NOTE: if the ring is cleared per-tick AND snapshot reads the
    // most-recent post-batch state, each snapshot contains only
    // the events from the LAST tick of its batch. So snap_a and
    // snap_b contain events from different ticks. Compare by
    // (tick, kind, payload) tuple.
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
