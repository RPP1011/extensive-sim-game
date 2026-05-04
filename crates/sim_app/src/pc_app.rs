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

    // Read back per-particle collision_count accumulator.
    //
    // Stage 1 (slice 2b body-form spatial query): MoveParticle's
    // `for other in spatial.nearby_particles(self) { emit Collision
    // { a: self, b: other, … } }` body-form walk emits ONE
    // Collision event per (self, other) candidate slot in the
    // 27-cell neighbourhood. Each event hits BOTH the
    // `on Collision { a: agent }` and `{ b: agent }` view handlers,
    // incrementing `collision_count[a]` AND `collision_count[b]`
    // by 1.0.
    //
    // # Analytical observable
    //
    // Per tick, the per-pair body fires ONCE per (self, candidate)
    // slot the spatial-grid walk surfaces. Define `K_self` as the
    // number of candidate slots in `self`'s 27-cell neighbourhood
    // (including `self` itself, since the body-form spatial walk
    // does NOT yet apply the `spatial_query` filter — that's a
    // follow-up). Then per tick:
    //
    //   total_emits  = Σ_self K_self
    //   total_count  = 2 * total_emits   (each emit increments two
    //                                     view slots)
    //
    // For 64 particles at AUTO_SPREAD = cbrt(64) = 4.0 (smaller
    // than CELL_SIZE = 6.0), every particle sits in the SAME
    // single grid cell. K_self = 64 for every particle. Per tick
    // the per-pair body emits 64 * 64 = 4096 Collision events;
    // each event increments BOTH `view[a]` and `view[b]` so the
    // sum-of-counts increases by 8192 per tick. Over TICKS=100
    // ticks: ~819 200 cumulative increments (modulo any
    // edge-cell drift).
    //
    // The previous Stage-0 placeholder emit fired exactly ONE
    // event per particle (a self-pair), so total was
    // `2 * TICKS * AGENT_COUNT = 12 800` — ~64x smaller. The
    // body-form walk's larger count is the observable proving the
    // new shape lit up end-to-end.
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
    println!(
        "pc_app: collision_count — total={:.2}; \
         per-slot min={:.2} mean={:.2} max={:.2}",
        total, min, mean, max,
    );
    // Sanity check: stage-0 (placeholder) total was
    //   2 * TICKS * AGENT_COUNT = 12_800.
    // Stage-1 (body-form spatial walk) must exceed that.
    let stage0_count = 2.0 * (TICKS as f32) * (AGENT_COUNT as f32);
    if total <= stage0_count {
        println!(
            "pc_app: WARNING — total ({:.0}) at-or-below stage-0 \
             baseline ({:.0}); body-form walk may not have fired",
            total, stage0_count,
        );
    } else {
        println!(
            "pc_app: stage-1 total is {:.1}x stage-0 baseline ({:.0}); \
             body-form walk lit up — emits scale with neighbourhood density",
            total / stage0_count,
            stage0_count,
        );
    }
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
