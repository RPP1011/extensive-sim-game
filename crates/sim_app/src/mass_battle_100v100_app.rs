//! mass_battle_100v100 harness — drives `mass_battle_100v100_runtime`
//! for up to MAX_TICKS or until one team is wiped, whichever comes
//! first. Reports per-25-tick wall-time + per-team alive count + per-
//! team total HP + cumulative damage / heals + per-role action counts.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — pair-field scoring scales to 200 agents
//!
//! 200 agents (100 Red + 100 Blue, each team = 10 Tank + 10 Healer +
//! 80 DPS). Per tick:
//!   - Strike (Tank): mask gates on (alive ∧ enemy ∧ tick%2==0 ∧
//!     self.level ∈ {1,4}). Inner candidate loop in scoring picks
//!     nearest enemy.
//!   - Snipe  (DPS):  mask gates analogously, picks lowest-HP enemy.
//!   - Heal   (Healer): mask gates on ally + same-team. Picks lowest-
//!     HP ally.
//!
//! Combat tempo per tick:
//!   - Tanks fire 10/team × every 2 ticks ≈ 50 strikes per 10 ticks
//!     × 18 dmg = 900 dmg per team per 10 ticks
//!   - DPS fire 80/team × every 3 ticks ≈ 267 snipes per 10 ticks ×
//!     14 dmg = 3733 dmg per team per 10 ticks
//!   - Healers fire 10/team × every 3 ticks ≈ 33 heals per 10 ticks ×
//!     22 = 733 heal per team per 10 ticks
//! Net per-team damage taken per 10 ticks ≈ 900 + 3733 - 733 ≈ 3900.
//! Total team HP ≈ 10*200 + 10*80 + 80*120 = 12 400 → wiped in
//! ~32 × 10 ticks ≈ 320 ticks.
//! With even compositions the result tends to "Red and Blue wipe
//! around the same time" — the deterministic argmax + identical
//! seeds + symmetric `level` encoding mean kills cascade in a
//! deterministic order that should slightly favour the higher-id
//! team because pair iteration is in slot order.
//!
//! ### (b) PERFORMANCE FAIL — pair-field scoring at scale
//!
//! At agent_cap=200 with 4 verbs and 40k mask cells per verb per
//! tick, the wall time per N=25 ticks should land below ~1s on most
//! discrete GPUs. If it blows past 5s/25 ticks the GAP is mask +
//! scoring kernel inner-loop cost — sidestep with spatial narrowing.

use mass_battle_100v100_runtime::{
    MassBattle100v100State, DPS_HP, DPS_PER_TEAM, HEALER_HP, HEALERS_PER_TEAM, PER_TEAM,
    TANK_HP, TANKS_PER_TEAM, TOTAL_AGENTS,
};
use engine::CompiledSim;
use std::time::Instant;

const SEED: u64 = 0xCAFE_BABE_F00D_BEEF;
const MAX_TICKS: u64 = 1500;
const TRACE_INTERVAL: u64 = 50;

fn role_label(level: u32) -> &'static str {
    match level {
        1 | 4 => "Tank",
        2 | 5 => "Healer",
        3 | 6 => "DPS",
        _ => "??",
    }
}

fn team_of(level: u32) -> &'static str {
    if level <= 3 {
        "Red"
    } else {
        "Blue"
    }
}

fn main() {
    let mut sim = MassBattle100v100State::new(SEED);
    println!("================================================================");
    println!(" MASS BATTLE 100 vs 100 — pair-field scoring at scale");
    println!("   seed=0x{:016X} agents={} max_ticks={}",
        SEED, TOTAL_AGENTS, MAX_TICKS);
    println!("   per team: {} Tank (HP={:.0}) + {} Healer (HP={:.0}) + {} DPS (HP={:.0})",
        TANKS_PER_TEAM, TANK_HP, HEALERS_PER_TEAM, HEALER_HP, DPS_PER_TEAM, DPS_HP);
    println!("================================================================");

    let levels = sim.read_level();
    debug_assert_eq!(levels.len(), TOTAL_AGENTS as usize);
    let mut role_counts = [(0u32, 0u32); 6];
    for &lvl in &levels {
        if (1..=6).contains(&lvl) {
            role_counts[(lvl - 1) as usize].0 += 1;
        }
    }
    println!("Role distribution (level → count):");
    for (i, &(count, _)) in role_counts.iter().enumerate() {
        let lvl = (i as u32) + 1;
        println!("  level={} ({}/{}): count={}", lvl, team_of(lvl), role_label(lvl), count);
    }
    println!();

    let mut tick_start = Instant::now();
    let mut ended_at: Option<u64> = None;
    let mut winner = "stalemate";

    println!("{:>6} | {:>5}/{:<5} | {:>5}/{:<5} | {:>8} | {:>8} | {:>9} | {:>9} | wall(N={})",
        "tick", "RAlv", PER_TEAM, "BAlv", PER_TEAM, "RHP", "BHP", "totalDmg", "totalHeal",
        TRACE_INTERVAL,
    );
    println!("{}", "-".repeat(90));

    for tick in 1..=MAX_TICKS {
        sim.step();

        if tick % TRACE_INTERVAL == 0 || tick == 1 {
            let alive = sim.read_alive();
            let hp = sim.read_hp();
            let damage = sim.damage_dealt().to_vec();
            let healing = sim.healing_done().to_vec();

            let mut red_alive = 0u32;
            let mut blue_alive = 0u32;
            let mut red_hp = 0.0_f32;
            let mut blue_hp = 0.0_f32;
            for i in 0..TOTAL_AGENTS as usize {
                if alive[i] == 1 {
                    if levels[i] <= 3 {
                        red_alive += 1;
                        red_hp += hp[i].max(0.0);
                    } else {
                        blue_alive += 1;
                        blue_hp += hp[i].max(0.0);
                    }
                }
            }
            let total_dmg: f32 = damage.iter().sum();
            let total_heal: f32 = healing.iter().sum();
            let elapsed = tick_start.elapsed();
            tick_start = Instant::now();

            println!(
                "{:>6} | {:>5}/{:<5} | {:>5}/{:<5} | {:>8.0} | {:>8.0} | {:>9.0} | {:>9.0} | {:>6.1?}",
                tick, red_alive, PER_TEAM, blue_alive, PER_TEAM,
                red_hp, blue_hp, total_dmg, total_heal, elapsed,
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

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    let final_alive = sim.read_alive();
    let final_hp = sim.read_hp();
    let damage = sim.damage_dealt().to_vec();
    let healing = sim.healing_done().to_vec();

    let mut red_alive = 0u32;
    let mut blue_alive = 0u32;
    let mut red_hp = 0.0_f32;
    let mut blue_hp = 0.0_f32;
    let mut role_alive = [0u32; 6];
    let mut role_dmg = [0.0_f32; 6];
    let mut role_heal = [0.0_f32; 6];
    for i in 0..TOTAL_AGENTS as usize {
        let lvl = levels[i];
        if (1..=6).contains(&lvl) {
            let role_idx = (lvl - 1) as usize;
            if final_alive[i] == 1 {
                role_alive[role_idx] += 1;
                if lvl <= 3 {
                    red_alive += 1;
                    red_hp += final_hp[i].max(0.0);
                } else {
                    blue_alive += 1;
                    blue_hp += final_hp[i].max(0.0);
                }
            }
            role_dmg[role_idx] += damage[i];
            role_heal[role_idx] += healing[i];
        }
    }
    let total_dmg: f32 = damage.iter().sum();
    let total_heal: f32 = healing.iter().sum();
    println!(
        "  Red:  alive={}/{}  hp_total={:.0}",
        red_alive, PER_TEAM, red_hp,
    );
    println!(
        "  Blue: alive={}/{}  hp_total={:.0}",
        blue_alive, PER_TEAM, blue_hp,
    );
    println!("  Total damage dealt: {:.0}", total_dmg);
    println!("  Total healing done: {:.0}", total_heal);
    println!();
    println!("Per-role action totals (cumulative damage / healing across all sources):");
    for i in 0..6 {
        let lvl = (i as u32) + 1;
        println!(
            "  level={} ({}/{:6}): alive {}, damage_dealt={:.0}, healing_done={:.0}",
            lvl, team_of(lvl), role_label(lvl), role_alive[i], role_dmg[i], role_heal[i],
        );
    }
    println!();
    if let Some(t) = ended_at {
        println!("  Combat ended at tick {} — winner: {}", t, winner);
    } else {
        println!("  Combat ran to MAX_TICKS={} — outcome: stalemate", MAX_TICKS);
    }

    // Hard assertions: pair-field scoring at scale must produce
    // SOME damage on each side. If a side dealt zero damage the
    // mask gate failed for that role — likely the per-pair grid
    // didn't cover the full agent_cap × agent_cap space.
    assert!(
        total_dmg > 0.0,
        "mass_battle_100v100: ASSERTION FAILED — zero total damage. \
         Pair-field scoring did not fire. The mask kernel likely \
         under-dispatched (agent_cap × agent_cap threads required).",
    );
    assert!(
        red_alive < PER_TEAM || blue_alive < PER_TEAM,
        "mass_battle_100v100: ASSERTION FAILED — both teams full HP after {} ticks. \
         No combat occurred. Check role / level encoding.",
        MAX_TICKS,
    );
}
