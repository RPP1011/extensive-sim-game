//! Task D4 — smoke test for `GpuBackend::step_batch` rewrite.
//!
//! This test does NOT exercise parity against the CPU backend — that's
//! Task D5 (`async_smoke`) and downstream integration tests. What we
//! verify here:
//!
//!   * The resident init path builds successfully (buffer allocation,
//!     cascade DSL load, resident cascade ctx construction).
//!   * A 1-tick `step_batch` records + submits + polls without panic.
//!   * The CPU-side `state.tick` advances by the expected amount.
//!
//! A warmup `step()` runs first so `cascade_ctx` + `view_storage` are
//! initialised on the sync path too, exercising `ensure_resident_init`'s
//! idempotency path on the subsequent `step_batch` call.

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
fn step_batch_one_tick_does_not_panic() {
    let mut gpu = match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("step_batch_smoke: GPU init failed — skipping ({e})");
            return;
        }
    };
    let mut state = SimState::new(16, 0xBEEF);
    for i in 0..4 {
        let ct = if i % 2 == 0 {
            CreatureType::Human
        } else {
            CreatureType::Wolf
        };
        state
            .spawn_agent(AgentSpawn {
                creature_type: ct,
                pos: Vec3::new(i as f32 * 2.0, 0.0, 0.0),
                hp: 100.0,
                max_hp: 100.0,
            })
            .expect("spawn_agent");
    }
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup: sync step populates cascade_ctx + view_storage, so the
    // first `step_batch` call exercises the already-initialised branch
    // of `ensure_resident_init`.
    gpu.step(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
    );
    let tick_before = state.tick;

    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        1,
    );

    assert_eq!(
        state.tick,
        tick_before.wrapping_add(1),
        "step_batch must advance state.tick by n_ticks (=1 here); before={tick_before} after={}",
        state.tick,
    );
}

#[test]
fn step_batch_three_ticks_advances_tick_counter() {
    let mut gpu = match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("step_batch_smoke: GPU init failed — skipping ({e})");
            return;
        }
    };
    let mut state = SimState::new(8, 0xC0FFEE);
    for i in 0..2 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32 * 3.0, 0.0, 0.0),
                hp: 100.0,
                max_hp: 100.0,
            })
            .expect("spawn_agent");
    }
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::with_engine_builtins();

    let tick_before = state.tick;
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        3,
    );

    assert_eq!(
        state.tick,
        tick_before.wrapping_add(3),
        "step_batch(n=3) must advance state.tick by 3; before={tick_before} after={}",
        state.tick,
    );
}
