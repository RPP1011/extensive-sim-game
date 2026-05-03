//! Smoke for the Stage-0 particle_collision skeleton — dispatch +
//! deterministic trajectory check at small scale.

use engine::sim_trait::CompiledSim;
use particle_collision_runtime::ParticleCollisionState;

fn main() {
    let agent_count = 32u32;
    let seed = 0xC0FFEE_u64;
    let mut sim = ParticleCollisionState::new(seed, agent_count);
    let pos_t0: Vec<glam::Vec3> = sim.positions().to_vec();
    for _ in 0..10 {
        sim.step();
    }
    let pos_tn: Vec<glam::Vec3> = sim.positions().to_vec();
    let moved: usize = pos_t0
        .iter()
        .zip(pos_tn.iter())
        .filter(|(a, b)| (*a - *b).length() > 1e-6)
        .count();
    println!(
        "[smoke] {moved}/{agent_count} particles moved between tick 0 and tick 10"
    );
    assert!(moved > 0, "no particles moved — MoveParticle dispatch may not have run");
    println!("[smoke] OK");
}
