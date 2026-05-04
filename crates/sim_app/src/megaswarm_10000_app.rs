//! megaswarm_10000 harness — drives `megaswarm_10000_runtime` for up
//! to MAX_TICKS or until one team is wiped, whichever comes first.
//! Reports per-50-tick wall-time + per-team alive count + per-team
//! total HP + cumulative damage. Final block reports avg + peak
//! ms/tick — the load-bearing observable for "does pair-field
//! scoring scale to 10000² (and beyond `max_compute_workgroups_per_
//! dimension`)?".
//!
//! ## Sim shape
//!
//! 10000 agents (5000 Red + 5000 Blue). TWO team-mirrored verbs
//! (RedStrike + BlueStrike). Per tick:
//!   - ~32 chunked PerPair mask dispatches (50000 workgroups each,
//!     pair_offset advances per chunk via cfg._pad0 — see
//!     `mask_predicate_per_pair_body_with_prefix` in the dsl_compiler).
//!     Total threads: agent_cap² = 100 000 000 per verb.
//!   - 1 scoring kernel: per-actor argmax over 2 rows × 10000
//!     candidates each (= 200M comparisons per tick).
//!   - 2 chronicle kernels (one per verb), 10000 threads each.
//!   - 1 apply kernel (PerEvent).
//!   - 1 fold (damage_dealt).
//!
//! ## Combat economy
//!
//! 1000 dmg / strike at HP=100 means a single landed strike kills.
//! Symmetric pile-on (every alive agent strikes the same lowest-HP
//! target each tick) → 1 kill/team/tick under the apply kernel's
//! non-atomic last-write-wins RMW. 5000 kills/team → mutual KO at
//! ~tick 5000. We run for far fewer ticks than that and report the
//! per-tick perf.
//!
//! ## Performance prediction
//!
//! megaswarm_1000 ran 0.27 ms/tick avg with linear scaling at
//! agent_cap=1000. Linear extrapolation → ~2.7 ms/tick at 10x scale.
//! The actual cost is likely SUPERLINEAR because:
//!   - scoring inner loop is 100x larger (10x candidates × 10x actors)
//!   - mask kernel is dispatched in ~32 chunks (pipeline-state cost)
//!   - event ring is 10x larger
//! Either way we're vastly under the 100 ms/tick budget — the point
//! is to discover where the curve actually goes.

use megaswarm_10000_runtime::{Megaswarm10000State, COMBATANT_HP, PER_TEAM, TOTAL_AGENTS};
use engine::CompiledSim;
use std::time::{Duration, Instant};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
/// Run length. With 1 kill/team/tick under symmetric pile-on, 200
/// ticks gives a ~4% kill ratio per team (200 / 5000) — comfortably
/// asymmetric enough to surface any chunked-dispatch perf cliff but
/// not so long that a stalled run wastes a lot of wall time.
const MAX_TICKS: u64 = 200;
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
    let mut sim = Megaswarm10000State::new(SEED);
    println!("================================================================");
    println!(" MEGASWARM 10000 — pair-field scoring at 100M-cell scale");
    println!("   seed=0x{:016X} agents={} max_ticks={}",
        SEED, TOTAL_AGENTS, MAX_TICKS);
    println!("   per team: {} Combatants (HP={:.0})", PER_TEAM, COMBATANT_HP);
    println!("   PerPair mask kernel chunks per tick: {}", sim.num_mask_chunks());
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

    println!("{:>6} | {:>5}/{:<5} | {:>5}/{:<5} | {:>11} | {:>11} | {:>12} | {:>9}",
        "tick", "RAlv", PER_TEAM, "BAlv", PER_TEAM, "RHP", "BHP", "totalDmg",
        format!("ms/tick(N={})", TRACE_INTERVAL),
    );
    println!("{}", "-".repeat(105));

    for tick in 1..=MAX_TICKS {
        sim.step();
        // GAP workaround: same as megaswarm_1000 — apply_damage uses
        // non-atomic RMW so dead agents can be left with stale HP.
        // CPU-side sweep restores sentinel HP every tick.
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
            // (skip the tick==1 bootstrap window which is dominated
            // by first-call kernel compile time).
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
                "{:>6} | {:>5}/{:<5} | {:>5}/{:<5} | {:>11.0} | {:>11.0} | {:>12.0} | {:>9.2}",
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
        println!("  Combat ran to MAX_TICKS={} — outcome: {}", MAX_TICKS, winner);
    }

    println!();
    println!("================================================================");
    println!(" PERFORMANCE TRACE — pair-field scoring at agent_cap={}", TOTAL_AGENTS);
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
        // megaswarm_1000 baseline: 0.272 ms/tick. Linear scaling to
        // 10x agents predicts 2.72 ms/tick. Print actual ratio so the
        // experiment's payload is always visible at the bottom.
        let baseline = 0.272;
        let scaling_ratio = avg_ms / baseline;
        println!();
        println!("  Comparison vs megaswarm_1000 (0.272 ms/tick baseline):");
        println!("    actual scaling ratio: {:>5.1}x  (linear prediction: 10.0x)", scaling_ratio);
        if scaling_ratio < 12.0 {
            println!("    verdict: pair-field scoring scales LINEARLY past 10000² with chunked PerPair dispatch.");
        } else if scaling_ratio < 100.0 {
            println!("    verdict: pair-field scoring scales SUPERLINEARLY (likely scoring inner loop dominates).");
        } else {
            println!("    verdict: pair-field scoring blew past linear prediction by {:.0}x — chunked dispatch overhead dominates.", scaling_ratio / 10.0);
        }
    }

    // Hard assertions: pair-field scoring at 100M-cell scale must
    // produce SOME damage. If a side dealt zero damage the mask
    // gate failed for that team — likely the chunked-dispatch
    // pair_offset wiring is wrong.
    assert!(
        total_dmg > 0.0,
        "megaswarm_10000: ASSERTION FAILED — zero total damage. \
         Pair-field scoring did not fire at 100M-cell scale.",
    );
    assert!(
        red_alive < PER_TEAM || blue_alive < PER_TEAM,
        "megaswarm_10000: ASSERTION FAILED — both teams full HP after {} ticks. \
         No combat occurred.",
        MAX_TICKS,
    );
}
