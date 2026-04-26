//! step_batch must NOT advance state.tick on CPU. GPU SimCfg.tick
//! advances each tick via the seed-indirect kernel's atomicAdd;
//! snapshot() reads it back to populate GpuSnapshot.tick (Task 2.11).
//!
//! This test has TWO assertions, split across two #[test] functions:
//!   - After step_batch(n), state.tick is unchanged from pre-batch
//!     (CPU-side stays stale; GPU is source of truth mid-batch).
//!   - After snapshot(), snap.tick reflects the advanced GPU tick.
//!     (Task 2.11 wires this; pre-2.11 this assertion fails — the
//!     test is `#[ignore]`'d until 2.11 lands.)

#![cfg(feature = "gpu")]

use engine::backend::ComputeBackend;
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

fn spawn_fixture() -> (
    GpuBackend,
    SimState,
    SimScratch,
    EventRing,
    CascadeRegistry,
) {
    let gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(16, 0xDEAD);
    for i in 0..4 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: if i % 2 == 0 {
                    CreatureType::Human
                } else {
                    CreatureType::Wolf
                },
                pos: Vec3::new(i as f32 * 2.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }
    let scratch = SimScratch::new(state.agent_cap() as usize);
    let events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::with_engine_builtins();
    (gpu, state, scratch, events, cascade)
}

#[test]
fn step_batch_does_not_advance_cpu_tick() {
    let (mut gpu, mut state, mut scratch, mut events, cascade) = spawn_fixture();

    // Warmup via sync step so ensure_resident_init is primed on first
    // step_batch. The sync step path does advance state.tick (that's
    // untouched by this task); we record the post-warmup value as our
    // baseline.
    gpu.step(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
    );
    let tick_after_sync_warmup = state.tick;

    // Run 10 ticks through the batch path.
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        10,
    );

    // CPU state.tick must NOT have advanced during step_batch. GPU
    // SimCfg.tick is the source of truth during a batch; CPU tick is
    // re-synced from the GPU at snapshot / end-of-batch boundaries.
    assert_eq!(
        state.tick, tick_after_sync_warmup,
        "state.tick should stay stale during batch; moved from {} to {}",
        tick_after_sync_warmup, state.tick,
    );
}

#[test]
fn snapshot_tick_reflects_gpu_advance() {
    let (mut gpu, mut state, mut scratch, mut events, cascade) = spawn_fixture();

    gpu.step(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
    );
    let tick_after_sync_warmup = state.tick;

    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        10,
    );

    // Snapshot double-buffering: first call returns empty (no kicked
    // copy yet), second call kicks the copy, third call returns the
    // batch state. The watermark advances between calls.
    let _empty = gpu.snapshot(&mut state).expect("first snapshot (empty)");
    let _kick = gpu.snapshot(&mut state).expect("kick snapshot");
    let snap = gpu.snapshot(&mut state).expect("third snapshot");
    assert_eq!(
        snap.tick,
        tick_after_sync_warmup + 10,
        "snapshot.tick should reflect 10 ticks of GPU-side advance \
         (currently reads stale CPU-tracked value; Task 2.11 fixes)",
    );
}
