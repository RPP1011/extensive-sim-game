//! Stage B.1 — verify the batch path's end-of-batch convergence
//! readback updates `batch_observed_max_iters` and that subsequent
//! batches cap iterations at `observed + 2`.
//!
//! Fixture: the same 1 human + 1 wolf scenario as
//! `indirect_cascade_converges.rs` — known to converge in <= 2
//! iterations, so after a batch `batch_observed_max_iters` should sit
//! at 1 or 2.

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

const SEED: u64 = 0xB1_0B_5E_EE_BEEF_C0DE;

#[test]
fn batch_observes_cascade_convergence_after_submit() {
    let Ok(mut gpu) = GpuBackend::new() else {
        eprintln!("GPU init unavailable — skipping Stage B.1 observation test");
        return;
    };
    let mut state = SimState::new(8, SEED);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    }).unwrap();
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(1.0, 0.0, 0.0),
        hp: 80.0,
        ..Default::default()
    }).unwrap();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup (sync) — compiles pipelines. The resident cascade ctx
    // (where `batch_observed_max_iters` lives) is only allocated once
    // `step_batch` runs, so the pre-batch accessor returns None.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    assert!(
        gpu.batch_observed_max_iters_for_test().is_none(),
        "resident cascade ctx should not exist before any step_batch call"
    );

    // First batch: 2 ticks. Readback should fire at end-of-batch and
    // update `batch_observed_max_iters` to reflect the final tick's
    // convergence.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 2);

    let post = gpu
        .batch_observed_max_iters_for_test()
        .expect("resident ctx present after batch");

    // The wolf-vs-human fixture converges in <=2 iters (usually 1, as
    // the first hit kills the human). We tolerate 0..=3 to absorb
    // kernel variance (0 = everyone dead, no new events on last tick).
    assert!(
        post <= 3,
        "observed max iters after batch should be <= 3 for a trivial \
         converging fixture; got {post}"
    );

    // Second batch: iter_cap is now derived from `post + 2`. Re-running
    // must not panic nor leave the observation pegged at MAX.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 2);
    let post2 = gpu
        .batch_observed_max_iters_for_test()
        .expect("resident ctx present after second batch");
    assert!(
        post2 <= 3,
        "second batch should still converge tightly; got {post2}"
    );
}
