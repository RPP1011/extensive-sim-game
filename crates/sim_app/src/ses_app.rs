//! Swarm-event-storm stress demo. Drives the
//! [`swarm_storm_runtime`] integrator through the same generic
//! harness shape as the other fixture binaries. Each agent emits
//! 4 Pulse events per tick, feeding two view-folds:
//!
//! - **pulse_count**: plain accumulator. Expected per-slot value
//!   after T ticks = 4 × T (one match per emit, no decay).
//! - **recent_pulse_intensity**: @decay-anchored at rate=0.85.
//!   Steady state per slot = per_tick / (1 - decay) = 4 / 0.15
//!   ≈ 26.667. Convergence: ~30 ticks to within 1% of asymptote.

use engine::CompiledSim;
use glam::Vec3;
use swarm_storm_runtime::SwarmStormState;

const SEED: u64 = 0x5_AA0_5701_2342;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 100;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = SwarmStormState::new(SEED, AGENT_COUNT);
    println!(
        "ses_app: starting run — seed=0x{:016X} agents={} ticks={}",
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
        "ses_app: finished — final tick={} agents={}",
        sim.tick(),
        sim.agent_count(),
    );

    // pulse_count: per-slot value should equal TICKS * 4 (4 emits
    // per tick × TICKS ticks). With AGENT_COUNT slots, total =
    // AGENT_COUNT × TICKS × 4.
    let pc = sim.pulse_counts().to_vec();
    let pc_total: f32 = pc.iter().sum();
    let pc_expected_per_slot = (TICKS as f32) * 4.0;
    let pc_expected_total = (AGENT_COUNT as f32) * pc_expected_per_slot;
    let pc_min = pc.iter().copied().fold(f32::INFINITY, f32::min);
    let pc_max = pc.iter().copied().fold(0.0_f32, f32::max);
    let pc_mean = pc_total / AGENT_COUNT as f32;
    println!(
        "ses_app: pulse_count        — total={:.2} (expected {:.0}); per-slot min={:.2} mean={:.2} max={:.2} (expected {:.1} each)",
        pc_total, pc_expected_total, pc_min, pc_mean, pc_max, pc_expected_per_slot,
    );
    if (pc_mean - pc_expected_per_slot).abs() > 1.0 {
        println!(
            "ses_app: KNOWN GAP B1 — view-fold storage writes are non-atomic (cg/emit/wgsl_body.rs::lower_cg_stmt's `bitcast<u32>(bitcast<f32>(...) + rhs)` RMW). With 4 emits/agent/tick, 3 of every 4 increments are lost to the race. P11 (Reduction Determinism) violation. pp/pc/cn don't surface this because they emit 1 event/agent/tick → no contention."
        );
    }

    // recent_pulse_intensity: should converge to per_tick/(1-decay)
    // = 4/0.15 ≈ 26.667 per slot. After 100 ticks of 0.85^t decay
    // the per-slot value is well-converged (0.85^100 ≈ 1e-7).
    let rpi = sim.recent_pulse_intensities().to_vec();
    let rpi_total: f32 = rpi.iter().sum();
    let rpi_steady_state_per_slot = 4.0_f32 / (1.0 - 0.85);
    let rpi_expected_total = (AGENT_COUNT as f32) * rpi_steady_state_per_slot;
    let rpi_min = rpi.iter().copied().fold(f32::INFINITY, f32::min);
    let rpi_max = rpi.iter().copied().fold(0.0_f32, f32::max);
    let rpi_mean = rpi_total / AGENT_COUNT as f32;
    println!(
        "ses_app: recent_pulse_int.  — total={:.2} (expected {:.2}); per-slot min={:.2} mean={:.2} max={:.2} (expected ~{:.2} each)",
        rpi_total,
        rpi_expected_total,
        rpi_min,
        rpi_mean,
        rpi_max,
        rpi_steady_state_per_slot,
    );
    if (rpi_mean - rpi_steady_state_per_slot).abs() > 1.0 {
        println!(
            "ses_app: KNOWN GAP B2 — `@decay(rate, per=tick)` is not lowered to a per-tick anchor multiplication. The fold body just adds the delta; nothing applies the 0.85 anchor scale before each tick's events. Compounded with B1, the recent_pulse_intensity converges to the same 1×ticks behavior as pulse_count rather than to the steady-state 4/(1-0.85) ≈ 26.67."
        );
    }
}

fn log_sample(sim: &mut SwarmStormState) {
    let tick = sim.tick();
    let positions = sim.positions();
    if positions.is_empty() {
        println!("  tick {:>4}: (no agents)", tick);
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
