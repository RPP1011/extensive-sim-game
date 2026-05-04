//! Cross-fixture determinism regression. Runs each per-fixture
//! runtime through the `CompiledSim` trait twice from the same seed
//! and asserts every position is bit-identical after a small number
//! of ticks. Catches RNG drift, kernel-level non-determinism, and
//! schedule-ordering regressions across the four fixture skeletons
//! that ship today.
//!
//! Each fixture gets a shorter run (50 ticks at 32 agents) than the
//! per-fixture smoke tests since the goal here is bit-stability, not
//! convergence. The harness lives in sim_app rather than each
//! fixture crate so adding a fifth fixture only requires importing
//! its `make_sim` and adding one row to the table.

use sim_runtime::BoidsState; // boids_runtime, aliased in Cargo.toml
use crowd_navigation_runtime::CrowdNavigationState;
use engine::sim_trait::CompiledSim;
use glam::Vec3;
use particle_collision_runtime::ParticleCollisionState;
use predator_prey_runtime::PredatorPreyState;

const SEED: u64 = 0xD17_5_BADD_CAFE_42;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 50;

fn run_to_positions(mut sim: Box<dyn CompiledSim>) -> Vec<Vec3> {
    for _ in 0..TICKS {
        sim.step();
    }
    sim.positions().to_vec()
}

fn assert_bit_identical(label: &str, a: Vec<Vec3>, b: Vec<Vec3>) {
    assert_eq!(
        a.len(),
        b.len(),
        "[{label}] agent_count diverged: {} vs {}",
        a.len(),
        b.len()
    );
    for (slot, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        let delta = (*pa - *pb).length();
        assert!(
            delta == 0.0,
            "[{label}] slot {slot} diverged: {pa:?} vs {pb:?} (Δ={delta:e})"
        );
    }
}

#[test]
fn boids_two_runs_bit_identical() {
    let a = run_to_positions(Box::new(BoidsState::new(SEED, AGENT_COUNT)));
    let b = run_to_positions(Box::new(BoidsState::new(SEED, AGENT_COUNT)));
    assert_bit_identical("boids", a, b);
}

#[test]
fn predator_prey_two_runs_bit_identical() {
    let a = run_to_positions(Box::new(PredatorPreyState::new(SEED, AGENT_COUNT)));
    let b = run_to_positions(Box::new(PredatorPreyState::new(SEED, AGENT_COUNT)));
    assert_bit_identical("predator_prey", a, b);
}

#[test]
fn particle_collision_two_runs_bit_identical() {
    let a = run_to_positions(Box::new(ParticleCollisionState::new(SEED, AGENT_COUNT)));
    let b = run_to_positions(Box::new(ParticleCollisionState::new(SEED, AGENT_COUNT)));
    assert_bit_identical("particle_collision", a, b);
}

#[test]
fn crowd_navigation_two_runs_bit_identical() {
    let a = run_to_positions(Box::new(CrowdNavigationState::new(SEED, AGENT_COUNT)));
    let b = run_to_positions(Box::new(CrowdNavigationState::new(SEED, AGENT_COUNT)));
    assert_bit_identical("crowd_navigation", a, b);
}
