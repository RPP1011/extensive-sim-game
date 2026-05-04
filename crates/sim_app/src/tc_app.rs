//! Target-chaser stress demo. Drives the
//! [`target_chaser_runtime`] integrator through the same generic
//! harness shape as `pp_app` / `pc_app` / `cn_app`. Each agent
//! steers toward the position of the agent it points at via
//! `engaged_with`. The runtime initializes the engaged_with ring
//! to `(slot+1) mod cap`, so chasers form a cycle: 0 → 1 → … →
//! N-1 → 0.
//!
//! The observable: the inter-agent spread (`max_diameter`) should
//! shrink monotonically as the ring tightens. If slice 1's
//! cross-agent read regressed (silently returned `vec3(0.0)`), the
//! pull-toward-target term would be (0 - self.pos) — collapsing
//! every agent toward the origin instead of toward its target.
//! The "diameter shrinks but doesn't collapse to a point" pattern
//! is the slice 1 fingerprint.

use engine::CompiledSim;
use glam::Vec3;
use target_chaser_runtime::TargetChaserState;

const SEED: u64 = 0xC0DE_CAFE_F00D_42;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 200;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = TargetChaserState::new(SEED, AGENT_COUNT);
    println!(
        "tc_app: starting run — seed=0x{:016X} chasers={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );
    log_sample(&mut sim, "init");
    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(&mut sim, "");
        }
    }
    println!(
        "tc_app: finished — final tick={} chasers={}",
        sim.tick(),
        sim.agent_count(),
    );
}

fn log_sample(sim: &mut TargetChaserState, label: &str) {
    let tick = sim.tick();
    let positions = sim.positions();
    if positions.is_empty() {
        println!("  tick {:>4}: (no chasers)", tick);
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
    // Mean inter-agent distance in the ring — slot i ↔ slot
    // (i+1) mod cap. A useful "is the chase actually working?"
    // signal: this should shrink monotonically as chasers close
    // on their targets.
    let n = positions.len();
    let mut ring_sum = 0.0_f32;
    for i in 0..n {
        let j = (i + 1) % n;
        ring_sum += (positions[i] - positions[j]).length();
    }
    let mean_ring_dist = ring_sum / n as f32;
    println!(
        "  tick {:>4} {:>5}: centroid=({:>+7.3}, {:>+7.3}, {:>+7.3}) max_diameter={:>+6.2} mean_ring_dist={:>+6.3}",
        tick, label, centroid.x, centroid.y, centroid.z, max_diameter, mean_ring_dist,
    );
}
