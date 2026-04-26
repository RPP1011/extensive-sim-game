//! Cross-backend parity for mask-fill (Plan 5a Task 8).
// The `gpu` feature is declared in the `engine_gpu` crate, not in `engine`.
// The check below is intentional — silence the expected cfg-unknown warning.
#![allow(unexpected_cfgs)]
//!
//!
//! Phase 1 of Plan 5a establishes the ComputeBackend trait routing for
//! mask-fill. This test asserts:
//! 1. SerialBackend produces deterministic (bit-identical) output when run
//!    twice on identical fixtures.
//! 2. (under --features gpu) GpuBackend stub produces byte-identical
//!    MaskBuffer state to SerialBackend on the same fixture.

use engine::mask::{MaskBuffer, TargetMask};
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_rules::backend::SerialBackend;
use glam::Vec3;

const AGENT_CAP: u32 = 8;
const SEED: u64 = 0xBEEF_CAFE_0001_0002;

/// Minimal fixture: 2 wolves and 1 human at distinct positions.
/// Mirrors the pattern used in wolves_and_humans_parity.rs.
fn make_fixture() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf 1 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf 2 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.5, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human spawn");
    state
}

#[test]
fn serial_backend_mask_fill_deterministic() {
    let state_a = make_fixture();
    let state_b = make_fixture();

    let n_a = state_a.agent_cap() as usize;
    let n_b = state_b.agent_cap() as usize;

    let mut buf_a = MaskBuffer::new(n_a);
    let mut buf_b = MaskBuffer::new(n_b);
    let mut tgt_a = TargetMask::new(n_a);
    let mut tgt_b = TargetMask::new(n_b);

    let mut serial_a = SerialBackend;
    let mut serial_b = SerialBackend;

    engine_rules::mask_fill::fill_all(&mut serial_a, &mut buf_a, &mut tgt_a, &state_a);
    engine_rules::mask_fill::fill_all(&mut serial_b, &mut buf_b, &mut tgt_b, &state_b);

    assert_eq!(
        buf_a.bits(),
        buf_b.bits(),
        "SerialBackend mask-fill must be deterministic across identical fixtures"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_stub_matches_serial_mask_fill() {
    let state = make_fixture();

    let n = state.agent_cap() as usize;

    let mut buf_serial = MaskBuffer::new(n);
    let mut buf_gpu = MaskBuffer::new(n);
    let mut tgt_serial = TargetMask::new(n);
    let mut tgt_gpu = TargetMask::new(n);

    let mut serial = SerialBackend;
    let mut gpu = engine_gpu::GpuBackend::default();

    engine_rules::mask_fill::fill_all(&mut serial, &mut buf_serial, &mut tgt_serial, &state);
    engine_rules::mask_fill::fill_all(&mut gpu, &mut buf_gpu, &mut tgt_gpu, &state);

    assert_eq!(
        buf_serial.bits(),
        buf_gpu.bits(),
        "GpuBackend stub must produce byte-identical MaskBuffer to SerialBackend"
    );
}
