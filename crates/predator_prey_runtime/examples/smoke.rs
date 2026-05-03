//! Smoke test for the Stage 0 predator_prey_runtime skeleton — boot
//! a small sim, step a handful of ticks, confirm positions advanced.

use engine::sim_trait::CompiledSim;

fn main() {
    let agent_count = 32u32;
    let seed = 0xC0FFEE_u64;
    let mut sim = predator_prey_runtime::PredatorPreyState::new(seed, agent_count);

    let pos_t0: Vec<glam::Vec3> = sim.positions().to_vec();
    println!(
        "[smoke] tick 0: agent[0] = {:?}",
        pos_t0.first().copied().unwrap_or_default()
    );

    for _ in 0..10 {
        sim.step();
    }

    let pos_t10: Vec<glam::Vec3> = sim.positions().to_vec();
    println!(
        "[smoke] tick 10: agent[0] = {:?}",
        pos_t10.first().copied().unwrap_or_default()
    );

    let moved: usize = pos_t0
        .iter()
        .zip(pos_t10.iter())
        .filter(|(a, b)| (*a - *b).length() > 1e-6)
        .count();
    println!(
        "[smoke] {moved}/{agent_count} agents moved between tick 0 and tick 10"
    );
    assert!(
        moved > 0,
        "no agents moved — MoveAll dispatch may not have run"
    );
    println!("[smoke] OK");
}
