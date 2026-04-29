//! Shared test helpers for `engine_gpu` integration tests.
//!
//! Tests `mod common;` to opt in. Helpers focus on the SCHEDULE-loop
//! dispatcher surface; pre-T16 helpers (mask_kernel, scoring_kernel,
//! per-kernel direct dispatch) are gone.

#![allow(dead_code)]

use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;
use glam::Vec3;

/// Per-agent fingerprint: the fields that must match byte-for-byte
/// across CPU and GPU step paths after the same input. Floats are
/// bit-cast to u32 so NaN comparisons are stable (and the harness
/// would deliberately surface a NaN diff).
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AgentFingerprint {
    pub id:         u32,
    pub alive:      bool,
    pub hp_bits:    u32,
    pub max_hp_bits: u32,
    pub shield_bits: u32,
    pub pos_x_bits: u32,
    pub pos_y_bits: u32,
    pub pos_z_bits: u32,
    pub creature:   u32,
    pub spawn_tick: u32,
}

pub fn fingerprint_agent(state: &SimState, id: AgentId) -> AgentFingerprint {
    let alive = state.agent_alive(id);
    let pos = state.agent_pos(id).unwrap_or(Vec3::ZERO);
    AgentFingerprint {
        id:         id.raw(),
        alive,
        hp_bits:    state.agent_hp(id).unwrap_or(0.0).to_bits(),
        max_hp_bits: state.agent_max_hp(id).unwrap_or(0.0).to_bits(),
        shield_bits: state.agent_shield_hp(id).unwrap_or(0.0).to_bits(),
        pos_x_bits: pos.x.to_bits(),
        pos_y_bits: pos.y.to_bits(),
        pos_z_bits: pos.z.to_bits(),
        creature:   state.agent_creature_type(id).map(|c| c as u32).unwrap_or(u32::MAX),
        spawn_tick: state.agent_spawn_tick(id).unwrap_or(u32::MAX),
    }
}

/// Whole-state fingerprint: every agent slot 1..=cap. Stable order by
/// AgentId, so the resulting Vec is deterministic.
pub fn fingerprint_state(state: &SimState) -> Vec<AgentFingerprint> {
    let cap = state.agent_cap();
    (1..=cap)
        .filter_map(|raw| AgentId::new(raw).map(|id| fingerprint_agent(state, id)))
        .collect()
}

/// Runs `n_ticks` of stepping on two fresh states (one CPU via
/// `engine_rules::step::step` + `SerialBackend`, one GPU via
/// `step_batch`) and asserts the post-step agent fingerprints are
/// byte-equal. Also asserts tick counter advanced by `n_ticks` on
/// both sides.
///
/// `fixture` is called twice — once per side — so each backend
/// gets its own owned `SimState` (SimState has no `Clone`).
///
/// On parity failure, prints the first diverging agent slot before
/// panicking — far more informative than `assert_eq!` on a long Vec.
pub fn assert_cpu_gpu_parity<P, F>(
    fixture: F,
    policy: &P,
    cascade: &CascadeRegistry<Event, ViewRegistry>,
    n_ticks: u32,
) where
    P: engine::policy::PolicyBackend,
    F: Fn() -> SimState,
{
    // CPU side — via engine_rules::step::step + SerialBackend.
    let mut cpu_state = fixture();
    let mut cpu_scratch = SimScratch::new(cpu_state.agent_cap() as usize);
    let mut cpu_events = EventRing::<Event>::with_cap(4096);
    let mut cpu_views = ViewRegistry::default();
    let tick0_cpu = cpu_state.tick;
    for _ in 0..n_ticks {
        engine_rules::step::step(
            &mut engine_rules::backend::SerialBackend,
            &mut cpu_state,
            &mut cpu_scratch,
            &mut cpu_events,
            &mut cpu_views,
            policy,
            cascade,
            &engine::debug::DebugConfig::default(),
        );
    }
    assert_eq!(
        cpu_state.tick,
        tick0_cpu + n_ticks,
        "CPU side: tick didn't advance by n_ticks ({n_ticks})",
    );

    // GPU side — via step_batch which internally does the same
    // engine_rules::step::step under default features (no-gpu stub) or
    // with the SCHEDULE-loop dispatch + CPU forward under --features gpu.
    let mut gpu = engine_gpu::GpuBackend::new();
    let mut gpu_state = fixture();
    let mut gpu_scratch = SimScratch::new(gpu_state.agent_cap() as usize);
    let mut gpu_events = EventRing::<Event>::with_cap(4096);
    let mut gpu_views = ViewRegistry::default();
    let tick0_gpu = gpu_state.tick;
    gpu.step_batch(
        &mut gpu_state,
        &mut gpu_scratch,
        &mut gpu_events,
        &mut gpu_views,
        policy,
        cascade,
        n_ticks,
    );
    assert_eq!(
        gpu_state.tick,
        tick0_gpu + n_ticks,
        "GPU side: tick didn't advance by n_ticks ({n_ticks})",
    );

    // Per-agent byte equality.
    let cpu_fp = fingerprint_state(&cpu_state);
    let gpu_fp = fingerprint_state(&gpu_state);
    if cpu_fp != gpu_fp {
        for (i, (c, g)) in cpu_fp.iter().zip(gpu_fp.iter()).enumerate() {
            if c != g {
                panic!(
                    "parity diverged at agent slot {i} (id={}) after {n_ticks} ticks:\n\
                     CPU = {c:#?}\n\
                     GPU = {g:#?}",
                    c.id,
                );
            }
        }
        panic!(
            "parity fingerprint length differs: CPU has {} agents, GPU has {}",
            cpu_fp.len(),
            gpu_fp.len(),
        );
    }
}

/// Build a small fixture: 4 agents in a square, mixed creature types,
/// all alive at full HP. Stable seed for reproducibility.
pub fn smoke_fixture_n4() -> SimState {
    let mut state = SimState::new(8, 42);
    let positions = [
        (CreatureType::Wolf,  Vec3::new(-5.0, 0.0, -5.0)),
        (CreatureType::Wolf,  Vec3::new( 5.0, 0.0, -5.0)),
        (CreatureType::Human, Vec3::new(-5.0, 0.0,  5.0)),
        (CreatureType::Human, Vec3::new( 5.0, 0.0,  5.0)),
    ];
    for (ct, pos) in positions {
        state.spawn_agent(AgentSpawn {
            creature_type: ct,
            pos,
            hp: 100.0,
            max_hp: 100.0,
        });
    }
    state
}
