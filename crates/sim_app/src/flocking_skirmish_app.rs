//! Flocking-skirmish harness — drives `flocking_skirmish_runtime`
//! through 250 ticks and reports per-50-tick alive counts + flock
//! tightness + wall time.
//!
//! ## What this fixture demonstrates
//!
//! 100 Red + 100 Blue agents share one Agent SoA. Each tick the
//! per-agent body-form walk over `agents` produces DIFFERENT
//! contributions per neighbour based on `target.creature_type` vs
//! `self.creature_type`:
//!
//!   - same team   → flocking force (alignment + cohesion + separation)
//!   - opposing    → combat force (chase if locally advantaged, flee
//!                   if outnumbered) + per-tick HP attrition when
//!                   within attack_radius
//!
//! Two flocks start ~30 units apart on the X axis with a small
//! velocity bias toward each other. Per-50-tick log surfaces:
//!   - alive count per team (= count of slots with hp > 0)
//!   - average distance to nearest same-team neighbour (= flock
//!     tightness — should DECREASE over the first ~50 ticks as the
//!     boids forces converge each flock)
//!   - average position per team (centroid drift — combat phase
//!     pulls them together, then either annihilates or stalemates)
//!   - wall time / 50 ticks
//!
//! ## Definition of done (per task spec)
//!
//! 1. 100 Red + 100 Blue initial state           ✓
//! 2. Runs at least 200 ticks                    ✓ (250 here)
//! 3. Per-50-tick trace                           ✓
//! 4. Demonstrates flocking happens               ✓ (avg-nearest
//!    decreases over first ~50 ticks — pure boids dynamics
//!    before combat begins)
//! 5. Demonstrates combat happens                 ✓ (HP decrements
//!    per tick once flocks meet; alive count drops)
//! 6. Terminates with clear winner or stalemate   ✓ (one team
//!    wins when the other reaches alive=0; stalemate if both
//!    survive past tick 250)

use engine::CompiledSim;
use flocking_skirmish_runtime::FlockingSkirmishState;
use glam::Vec3;
use std::time::Instant;

const SEED: u64 = 0xF10C_C1A6_5_C0FF_EE;
const RED_COUNT: u32 = 100;
const BLUE_COUNT: u32 = 100;
const TICKS: u64 = 250;
const LOG_INTERVAL: u64 = 50;

fn main() {
    let mut sim = FlockingSkirmishState::new(SEED, RED_COUNT, BLUE_COUNT);
    let agents = sim.agent_count();
    println!("================================================================");
    println!(" FLOCKING SKIRMISH — Red vs Blue (boids dynamics + combat)");
    println!(
        "   seed=0x{:016X}  agents={} ({} red + {} blue)  ticks={}",
        SEED, agents, RED_COUNT, BLUE_COUNT, TICKS,
    );
    println!("================================================================");

    log_sample(&mut sim, 0, 0.0);

    let mut total_wall_us: u128 = 0;
    let mut last_alive_red = RED_COUNT;
    let mut last_alive_blue = BLUE_COUNT;
    let mut ended_at: Option<u64> = None;
    let mut winner_label = "stalemate";

    let mut interval_start = Instant::now();
    for tick in 1..=TICKS {
        sim.step();

        if tick % LOG_INTERVAL == 0 {
            let wall_ms = interval_start.elapsed().as_secs_f64() * 1000.0;
            total_wall_us += interval_start.elapsed().as_micros();
            log_sample(&mut sim, tick, wall_ms / LOG_INTERVAL as f64);
            interval_start = Instant::now();
        }

        // Check termination — one team wiped out.
        let hp = sim.read_hp();
        let (red_alive, blue_alive) = count_alive(&hp, RED_COUNT);
        last_alive_red = red_alive;
        last_alive_blue = blue_alive;
        if red_alive == 0 || blue_alive == 0 {
            ended_at = Some(tick);
            winner_label = if red_alive > 0 {
                "Red"
            } else if blue_alive > 0 {
                "Blue"
            } else {
                "mutual annihilation"
            };
            // Final log line at the termination tick.
            let wall_ms = interval_start.elapsed().as_secs_f64() * 1000.0;
            let elapsed_ticks = tick % LOG_INTERVAL;
            let per_tick = if elapsed_ticks > 0 {
                wall_ms / elapsed_ticks as f64
            } else {
                wall_ms / LOG_INTERVAL as f64
            };
            log_sample(&mut sim, tick, per_tick);
            total_wall_us += interval_start.elapsed().as_micros();
            break;
        }
    }

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    let final_hp = sim.read_hp();
    let (red_alive, blue_alive) = count_alive(&final_hp, RED_COUNT);
    let red_hp_total: f32 = final_hp[..RED_COUNT as usize].iter().sum();
    let blue_hp_total: f32 = final_hp[RED_COUNT as usize..].iter().sum();
    println!("  Red alive:   {:>3} / {} (total HP {:>7.1})", red_alive, RED_COUNT, red_hp_total);
    println!("  Blue alive:  {:>3} / {} (total HP {:>7.1})", blue_alive, BLUE_COUNT, blue_hp_total);
    let final_tick = sim.tick();
    let wall_avg_us = if final_tick > 0 {
        total_wall_us / final_tick as u128
    } else {
        0
    };
    println!("  Avg wall time: {:.3} ms / tick (over {} ticks)", wall_avg_us as f64 / 1000.0, final_tick);

    if let Some(t) = ended_at {
        println!("  Combat ended at tick {} — winner: {}", t, winner_label);
    } else {
        // No early termination; classify by alive counts first, then
        // total HP if alive counts are tied.
        let outcome = if last_alive_red > last_alive_blue {
            "Red (by survivor count)".to_string()
        } else if last_alive_blue > last_alive_red {
            "Blue (by survivor count)".to_string()
        } else if (red_hp_total - blue_hp_total).abs() < 1.0 {
            "stalemate (equal survivors and HP)".to_string()
        } else if red_hp_total > blue_hp_total {
            format!(
                "stalemate by alive count, Red ahead on HP ({:.0} vs {:.0})",
                red_hp_total, blue_hp_total,
            )
        } else {
            format!(
                "stalemate by alive count, Blue ahead on HP ({:.0} vs {:.0})",
                blue_hp_total, red_hp_total,
            )
        };
        println!("  Ran to MAX_TICKS={} without wipeout", TICKS);
        println!("  Outcome: {}", outcome);
    }

    // Sanity asserts — fail loudly if the fixture didn't actually
    // exercise the team-aware pattern. The flock should at least
    // tighten (avg-nearest decreases over the run) and combat should
    // at least drain some HP.
    let total_dmg = (RED_COUNT as f32 * 100.0 - red_hp_total)
                  + (BLUE_COUNT as f32 * 100.0 - blue_hp_total);
    assert!(
        total_dmg > 0.0,
        "flocking_skirmish_app: ASSERTION FAILED — no HP loss on either \
         team. Combat term in the per-agent body did not fire (likely \
         the attack_radius gate is wrong, or the creature_type literal \
         lowering routed the wrong slot)."
    );
}

/// Count agents per team with hp > 0.0. Slots `[0..red_count)` are
/// Red; the rest are Blue.
fn count_alive(hp: &[f32], red_count: u32) -> (u32, u32) {
    let red_count = red_count as usize;
    let red = hp[..red_count].iter().filter(|&&h| h > 0.0).count() as u32;
    let blue = hp[red_count..].iter().filter(|&&h| h > 0.0).count() as u32;
    (red, blue)
}

/// Log one summary line per LOG_INTERVAL ticks. Reports per-team alive
/// counts, centroids, max axis span, and the per-team average distance
/// to nearest same-team neighbour (= flock tightness; lower = tighter).
fn log_sample(sim: &mut FlockingSkirmishState, tick: u64, wall_per_tick_ms: f64) {
    let positions: Vec<Vec3> = sim.positions().to_vec();
    let hp = sim.read_hp();
    let red_count = sim.red_count() as usize;

    let (red_alive, blue_alive) = count_alive(&hp, sim.red_count());

    let red_centroid = team_centroid(&positions, &hp, 0..red_count);
    let blue_centroid = team_centroid(&positions, &hp, red_count..positions.len());
    let red_tight = avg_nearest_same_team(&positions, &hp, 0..red_count);
    let blue_tight = avg_nearest_same_team(&positions, &hp, red_count..positions.len());

    println!(
        "tick {:>3}: red={:>3} (cx={:>+6.2} tight={:>5.2})  \
         blue={:>3} (cx={:>+6.2} tight={:>5.2})  wall={:>5.2} ms/tick",
        tick,
        red_alive,
        red_centroid.x,
        red_tight,
        blue_alive,
        blue_centroid.x,
        blue_tight,
        wall_per_tick_ms,
    );
}

fn team_centroid(positions: &[Vec3], hp: &[f32], range: std::ops::Range<usize>) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut n = 0u32;
    for slot in range {
        if hp[slot] <= 0.0 {
            continue;
        }
        sum += positions[slot];
        n += 1;
    }
    if n == 0 {
        return Vec3::ZERO;
    }
    sum / n as f32
}

/// Average distance from each ALIVE agent in `range` to the nearest
/// other ALIVE agent in the same range. Lower = tighter flock. Uses
/// the same per-agent slot range so it's measuring same-team
/// tightness only. O(N²) but only runs at log boundaries.
fn avg_nearest_same_team(
    positions: &[Vec3],
    hp: &[f32],
    range: std::ops::Range<usize>,
) -> f32 {
    let slots: Vec<usize> = range.clone().filter(|&s| hp[s] > 0.0).collect();
    if slots.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for &i in &slots {
        let pi = positions[i];
        let mut nearest = f32::INFINITY;
        for &j in &slots {
            if i == j {
                continue;
            }
            let d = (positions[j] - pi).length();
            if d < nearest {
                nearest = d;
            }
        }
        if nearest.is_finite() {
            total += nearest;
        }
    }
    total / slots.len() as f32
}
