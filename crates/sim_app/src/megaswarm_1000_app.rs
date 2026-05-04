//! megaswarm_1000 harness — drives `megaswarm_1000_runtime` for up
//! to MAX_TICKS or until one team is wiped, whichever comes first.
//! Reports per-50-tick wall-time + per-team alive count + per-team
//! total HP + cumulative damage. Final block reports avg + peak
//! ms/tick — the load-bearing observable for "does pair-field
//! scoring scale to 1000²?".
//!
//! ## Sim shape
//!
//! 1000 agents (500 Red + 500 Blue). TWO team-mirrored verbs
//! (RedStrike + BlueStrike) — see `assets/sim/megaswarm_1000.sim`
//! header for why a single verb cannot route per-actor (inner
//! argmax has no per-candidate predicate, score expr can't see
//! `self.*`). Per tick:
//!   - 1 fused mask kernel: 1M-thread dispatch (agent_cap²) writes
//!     both mask_0 (RedStrike) + mask_1 (BlueStrike).
//!   - 1 scoring kernel: per-actor argmax over 2 rows × 1000
//!     candidates each (= 2M comparisons per tick).
//!   - 2 chronicle kernels (one per verb), 1000 threads each.
//!   - 1 apply kernel (PerEvent).
//!   - 1 fold (damage_dealt).
//!
//! ## Combat economy
//!
//! 1000 dmg / strike at HP=100 means a single landed strike kills.
//! The apply kernel uses non-atomic RMW on agent_hp, so when N
//! attackers pile on the same target only one strike's value
//! actually applies (last write wins). With 1000 dmg vs HP=100
//! that one write definitely crosses the death threshold → 1
//! kill/team/tick under symmetric pile-on. 500 kills/team → mutual
//! KO at ~tick 500. MAX_TICKS=1500 leaves headroom for asymmetric
//! cascades.
//!
//! ## Performance prediction
//!
//! At agent_cap=1000 the fused mask kernel dispatches 1M threads
//! (= 15625 workgroups of 64). On a discrete GPU this should land
//! below ~10ms/tick. If wall time per 50 ticks exceeds 5s (=
//! 100ms/tick), the gap is the mask + scoring kernel inner loop
//! cost — the brief's policy is to switch to spatial body-form
//! like duel_25v25.

use megaswarm_1000_runtime::{Megaswarm1000State, COMBATANT_HP, PER_TEAM, TOTAL_AGENTS};
use engine::CompiledSim;
use std::time::{Duration, Instant};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const MAX_TICKS: u64 = 1500;
const TRACE_INTERVAL: u64 = 50;

fn team_of(level: u32) -> &'static str {
    match level {
        1 => "Red",
        2 => "Blue",
        _ => "??",
    }
}

fn main() {
    let total_run_start = Instant::now();
    let mut sim = Megaswarm1000State::new(SEED);
    println!("================================================================");
    println!(" MEGASWARM 1000 — pair-field scoring at 1M-cell scale");
    println!("   seed=0x{:016X} agents={} max_ticks={}",
        SEED, TOTAL_AGENTS, MAX_TICKS);
    println!("   per team: {} Combatants (HP={:.0})", PER_TEAM, COMBATANT_HP);
    println!("================================================================");

    let levels = sim.read_level();
    debug_assert_eq!(levels.len(), TOTAL_AGENTS as usize);
    let mut team_counts = [0u32, 0u32];
    for &lvl in &levels {
        if (1..=2).contains(&lvl) {
            team_counts[(lvl - 1) as usize] += 1;
        }
    }
    println!("Team distribution (level → count):");
    for (i, &count) in team_counts.iter().enumerate() {
        let lvl = (i as u32) + 1;
        println!("  level={} ({}): count={}", lvl, team_of(lvl), count);
    }
    println!();

    let mut window_start = Instant::now();
    let mut ended_at: Option<u64> = None;
    let mut winner = "stalemate";
    let mut peak_window: Duration = Duration::ZERO;
    let mut total_window_time: Duration = Duration::ZERO;
    let mut total_window_ticks: u64 = 0;

    println!("{:>6} | {:>5}/{:<5} | {:>5}/{:<5} | {:>9} | {:>9} | {:>10} | {:>9}",
        "tick", "RAlv", PER_TEAM, "BAlv", PER_TEAM, "RHP", "BHP", "totalDmg",
        format!("ms/tick(N={})", TRACE_INTERVAL),
    );
    println!("{}", "-".repeat(95));

    for tick in 1..=MAX_TICKS {
        sim.step();
        // GAP workaround: compiler-emitted apply_damage uses non-
        // atomic RMW on agent_hp. When N>>1 strikes hit the same
        // dead agent in one tick, a death-branch sentinel-HP write
        // can be clobbered by an else-branch stale-value write,
        // leaving "dead but HP=10" agents that still win argmax
        // and freeze the sim. CPU-side sweep restores sentinel HP
        // for any alive=0 agent whose HP isn't at sentinel.
        sim.sweep_dead_to_sentinel();

        if tick % TRACE_INTERVAL == 0 || tick == 1 {
            let alive = sim.read_alive();
            let hp = sim.read_hp();
            let damage = sim.damage_dealt().to_vec();

            let mut red_alive = 0u32;
            let mut blue_alive = 0u32;
            let mut red_hp = 0.0_f32;
            let mut blue_hp = 0.0_f32;
            for i in 0..TOTAL_AGENTS as usize {
                if alive[i] == 1 {
                    if levels[i] == 1 {
                        red_alive += 1;
                        red_hp += hp[i].max(0.0);
                    } else {
                        blue_alive += 1;
                        blue_hp += hp[i].max(0.0);
                    }
                }
            }
            let total_dmg: f32 = damage.iter().sum();
            let elapsed = window_start.elapsed();

            // Track perf only for the full TRACE_INTERVAL windows
            // (skip the tick==1 bootstrap window which is just 1
            // tick and dominated by first-call kernel compile time).
            let window_ticks = if tick == 1 { 1 } else { TRACE_INTERVAL };
            if tick != 1 {
                if elapsed > peak_window {
                    peak_window = elapsed;
                }
                total_window_time += elapsed;
                total_window_ticks += window_ticks;
            }
            let ms_per_tick = elapsed.as_secs_f64() * 1000.0 / window_ticks as f64;

            window_start = Instant::now();

            println!(
                "{:>6} | {:>5}/{:<5} | {:>5}/{:<5} | {:>9.0} | {:>9.0} | {:>10.0} | {:>9.2}",
                tick, red_alive, PER_TEAM, blue_alive, PER_TEAM,
                red_hp, blue_hp, total_dmg, ms_per_tick,
            );

            if red_alive == 0 && blue_alive == 0 {
                ended_at = Some(tick);
                winner = "mutual KO";
                break;
            } else if red_alive == 0 {
                ended_at = Some(tick);
                winner = "Blue";
                break;
            } else if blue_alive == 0 {
                ended_at = Some(tick);
                winner = "Red";
                break;
            }
        }
    }

    let total_runtime = total_run_start.elapsed();

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    let final_alive = sim.read_alive();
    let final_hp = sim.read_hp();
    let damage = sim.damage_dealt().to_vec();

    let mut red_alive = 0u32;
    let mut blue_alive = 0u32;
    let mut red_hp = 0.0_f32;
    let mut blue_hp = 0.0_f32;
    let mut team_dmg = [0.0_f32; 2];
    for i in 0..TOTAL_AGENTS as usize {
        let lvl = levels[i];
        if (1..=2).contains(&lvl) {
            let team_idx = (lvl - 1) as usize;
            if final_alive[i] == 1 {
                if lvl == 1 {
                    red_alive += 1;
                    red_hp += final_hp[i].max(0.0);
                } else {
                    blue_alive += 1;
                    blue_hp += final_hp[i].max(0.0);
                }
            }
            team_dmg[team_idx] += damage[i];
        }
    }
    let total_dmg: f32 = damage.iter().sum();
    println!(
        "  Red:  alive={}/{}  hp_total={:.0}  damage_dealt={:.0}",
        red_alive, PER_TEAM, red_hp, team_dmg[0],
    );
    println!(
        "  Blue: alive={}/{}  hp_total={:.0}  damage_dealt={:.0}",
        blue_alive, PER_TEAM, blue_hp, team_dmg[1],
    );
    println!("  Total damage dealt: {:.0}", total_dmg);
    println!();
    if let Some(t) = ended_at {
        println!("  Combat ended at tick {} — winner: {}", t, winner);
    } else {
        println!("  Combat ran to MAX_TICKS={} — outcome: stalemate", MAX_TICKS);
    }

    println!();
    println!("================================================================");
    println!(" PERFORMANCE TRACE — pair-field scoring at agent_cap=1000");
    println!("================================================================");
    println!("  Total run time:       {:>8.3?}", total_runtime);
    if total_window_ticks > 0 {
        let avg_ms = total_window_time.as_secs_f64() * 1000.0 / total_window_ticks as f64;
        let peak_ms = peak_window.as_secs_f64() * 1000.0 / TRACE_INTERVAL as f64;
        println!(
            "  Avg ms/tick (excl. bootstrap, {} ticks): {:>8.3} ms",
            total_window_ticks, avg_ms,
        );
        println!(
            "  Peak ms/tick (max window /{}):           {:>8.3} ms",
            TRACE_INTERVAL, peak_ms,
        );
    }

    // Hard assertions: pair-field scoring at 1M-cell scale must
    // produce SOME damage. If a side dealt zero damage the mask
    // gate failed for that team — likely the per-pair grid did
    // not cover the full agent_cap × agent_cap space.
    assert!(
        total_dmg > 0.0,
        "megaswarm_1000: ASSERTION FAILED — zero total damage. \
         Pair-field scoring did not fire at 1M-cell scale.",
    );
    assert!(
        red_alive < PER_TEAM || blue_alive < PER_TEAM,
        "megaswarm_1000: ASSERTION FAILED — both teams full HP after {} ticks. \
         No combat occurred.",
        MAX_TICKS,
    );
}
