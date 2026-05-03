//! Per-creature predator/prey harness — runs the
//! [`predator_prey_runtime`] fixture and logs per-creature spread/
//! centroid summaries each tick.
//!
//! The pp fixture splits agent slots between Hare (creature_type = 0,
//! even slots) and Wolf (creature_type = 1, odd slots) at init time
//! and dispatches both `physics_MoveHare` + `physics_MoveWolf` per
//! tick. Logging splits the populations on the same parity so the
//! per-creature behavior (different speeds, future chase/flee) is
//! visible per tick line.

use engine::CompiledSim;
use glam::Vec3;
use predator_prey_runtime::PredatorPreyState;

const SEED: u64 = 0xB01D_5_C0FF_EE_42;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 200;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = PredatorPreyState::new(SEED, AGENT_COUNT);
    let hare_count = sim.hare_count();
    let wolf_count = sim.wolf_count();
    println!(
        "pp_app: starting run — seed=0x{:016X} agents={} hares={} wolves={} ticks={}",
        SEED, AGENT_COUNT, hare_count, wolf_count, TICKS,
    );
    log_sample(&mut sim);

    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(&mut sim);
        }
    }

    println!(
        "pp_app: finished — final tick={} agents={} ({} hares + {} wolves)",
        sim.tick(),
        sim.agent_count(),
        hare_count,
        wolf_count,
    );
}

/// Print one summary line: tick + per-creature centroid + axis-aligned
/// span. Splitting the populations on slot parity matches the runtime
/// init pattern (even = Hare, odd = Wolf). The per-creature MoveHare
/// / MoveWolf dispatches use different speeds, so the centroids drift
/// at different rates.
fn log_sample(sim: &mut PredatorPreyState) {
    let tick = sim.tick();
    let positions = sim.positions().to_vec();
    if positions.is_empty() {
        println!("  tick {:>4}: (no agents)", tick);
        return;
    }
    let (hare_centroid, hare_span) = group_stats(&positions, |slot| slot % 2 == 0);
    let (wolf_centroid, wolf_span) = group_stats(&positions, |slot| slot % 2 == 1);
    println!(
        "  tick {:>4}: hare centroid=({:>+6.2}, {:>+6.2}, {:>+6.2}) span={:>+5.2} | \
         wolf centroid=({:>+6.2}, {:>+6.2}, {:>+6.2}) span={:>+5.2}",
        tick,
        hare_centroid.x,
        hare_centroid.y,
        hare_centroid.z,
        hare_span,
        wolf_centroid.x,
        wolf_centroid.y,
        wolf_centroid.z,
        wolf_span,
    );
}

/// Centroid + max axis-aligned span across the slot positions whose
/// indices match `pred`. Returns (`Vec3::ZERO`, 0.0) when no slots
/// match (defensive — shouldn't happen with the 50/50 init split).
fn group_stats(positions: &[Vec3], pred: impl Fn(usize) -> bool) -> (Vec3, f32) {
    let mut min = None::<Vec3>;
    let mut max = None::<Vec3>;
    let mut sum = Vec3::ZERO;
    let mut count = 0u32;
    for (slot, p) in positions.iter().enumerate() {
        if !pred(slot) {
            continue;
        }
        min = Some(min.map_or(*p, |m| m.min(*p)));
        max = Some(max.map_or(*p, |m| m.max(*p)));
        sum += *p;
        count += 1;
    }
    if count == 0 {
        return (Vec3::ZERO, 0.0);
    }
    let centroid = sum / count as f32;
    let span_v = max.unwrap() - min.unwrap();
    let span = span_v.x.max(span_v.y).max(span_v.z);
    (centroid, span)
}
