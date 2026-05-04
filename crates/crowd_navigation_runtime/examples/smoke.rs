//! Smoke for the Stage-0 crowd_navigation skeleton — dispatch +
//! deterministic trajectory at small scale.

use engine::sim_trait::CompiledSim;
use crowd_navigation_runtime::CrowdNavigationState;

fn main() {
    let agent_count = 32u32;
    let seed = 0xC0FFEE_u64;
    let mut sim = CrowdNavigationState::new(seed, agent_count);
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
        "[smoke] {moved}/{agent_count} walkers moved between tick 0 and tick 10"
    );
    assert!(moved > 0, "no walkers moved — MoveWalker dispatch may not have run");
    println!("[smoke] OK");
}
