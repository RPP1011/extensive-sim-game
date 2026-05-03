//! Particle-collision Stage-0 demo. Drives the
//! [`particle_collision_runtime`] integrator through the same
//! generic harness shape as `sim_app` / `pp_app`. Stage 0 just
//! integrates pos += vel × damping per tick; later stages copy the
//! per-pair Collision detection + view-fold impulse accumulator
//! from the design target.

use engine::CompiledSim;
use glam::Vec3;
use particle_collision_runtime::ParticleCollisionState;

const SEED: u64 = 0xB01D_5_C0FF_EE_42;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 100;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = ParticleCollisionState::new(SEED, AGENT_COUNT);
    println!(
        "pc_app: starting run — seed=0x{:016X} particles={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );
    log_sample(&mut sim);
    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(&mut sim);
        }
    }
    println!(
        "pc_app: finished — final tick={} particles={}",
        sim.tick(),
        sim.agent_count(),
    );

    // Read back per-particle collision_count accumulator. The
    // placeholder MoveParticle emit fires Collision { a: self,
    // b: self, impulse: pos } each tick, so each Particle slot
    // matches BOTH `on Collision { a: agent }` and `on Collision
    // { b: agent }` view handlers — every slot increments by 2
    // per tick. Expected: 2 * TICKS per slot, total = 2 * TICKS *
    // AGENT_COUNT.
    let cc = sim.collision_counts();
    let mut total = 0.0_f32;
    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    for &v in cc {
        total += v;
        min = min.min(v);
        max = max.max(v);
    }
    let mean = total / AGENT_COUNT as f32;
    let expected_per_slot = (TICKS as f32) * 2.0;
    println!(
        "pc_app: collision_count — total={:.2} (expected {}); \
         per-slot min={:.2} mean={:.2} max={:.2} (expected ~{:.0} each)",
        total,
        AGENT_COUNT as f32 * expected_per_slot,
        min,
        mean,
        max,
        expected_per_slot,
    );
}

fn log_sample(sim: &mut ParticleCollisionState) {
    let tick = sim.tick();
    let positions = sim.positions();
    if positions.is_empty() {
        println!("  tick {:>4}: (no particles)", tick);
        return;
    }
    let mut min = positions[0];
    let mut max = positions[0];
    let mut sum = Vec3::ZERO;
    for p in positions {
        min = min.min(*p);
        max = max.max(*p);
        sum += *p;
    }
    let centroid = sum / positions.len() as f32;
    let span = max - min;
    let max_diameter = span.x.max(span.y).max(span.z);
    println!(
        "  tick {:>4}: centroid=({:>+7.3}, {:>+7.3}, {:>+7.3}) max_diameter={:>+6.2}",
        tick, centroid.x, centroid.y, centroid.z, max_diameter,
    );
}
