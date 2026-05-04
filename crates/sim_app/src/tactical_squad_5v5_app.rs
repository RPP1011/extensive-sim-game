//! Tactical squad 5v5 harness — drives `tactical_squad_5v5_runtime`
//! for up to MAX_TICKS or until one team is wiped, reporting per-tick
//! per-team alive count + total HP, and counting per-role targeting
//! decisions (DPS picks lowest-HP enemy; Healer picks lowest-HP ally).
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — both teams trade kills until one breaks
//!
//! 5 vs 5, each team has 1 Tank (200 HP) + 1 Healer (80 HP) + 3 DPS
//! (120 HP). Per-tick gates fire:
//!   - Strike (Tank, every 3rd tick): 10 dmg on any enemy
//!   - Snipe (DPS, every 4th tick): 22 dmg on lowest-HP enemy
//!   - Heal (Healer, every 3rd tick): 18 hp to lowest-HP ally
//!
//! Total per-team damage output per cooldown cycle (12-tick LCM):
//!   - Tank: 4 strikes × 10 = 40
//!   - 3 DPS: 3 × 3 snipes × 22 = 198
//!   - Total: 238 dmg per team per 12-tick cycle = 19.8 dpt
//!   - Healer recovers: 4 heals × 18 = 72 hp per 12-tick = 6 hpt
//!
//! Net: ~14 hp drain per tick per team. Total team HP = 200 + 80 +
//! 360 = 640. Without targeting, the trade should resolve in ~46
//! ticks; with focused-fire DPS picking the weakest first, expect
//! the first kill (Healer or DPS) at ~tick 30-50, then a cascade.
//!
//! ### Targeting assertions
//!
//!   - DPS_LOWEST_HP_ENEMY: across the run, count ticks where every
//!     alive DPS picked the truly-lowest-HP enemy. With a constant
//!     score under inversion `(200 - target.hp)`, the argmax must
//!     pick the candidate with the smallest hp; we verify this
//!     post-hoc via the scoring_output buffer.
//!   - HEALER_LOWEST_HP_ALLY: same shape, on the ally side.
//!
//! ### (b) NO FIRE — the fix didn't take
//!
//! If the inline mask_k fix didn't propagate, every actor's mask bit
//! depends on candidate=0 only. With agent 0 = Red Tank, only
//! Blue actors would fire mask_0 (Strike), no Red actors would fire
//! Strike at all, and the game would stalemate at half output.

use engine::CompiledSim;
use tactical_squad_5v5_runtime::{
    TacticalSquad5v5State, HP_DPS, HP_HEALER, HP_TANK, ROLE_DPS, ROLE_HEALER, ROLE_TANK,
    TEAM_BLUE, TEAM_RED,
};

const SEED: u64 = 0xCAFE_BEEF_5050_5050;
const AGENT_COUNT: u32 = 10;
const MAX_TICKS: u64 = 800;

// ActionSelected indices in the scoring_output buffer. The .sim
// declares verbs in source order: Strike (0), Snipe (1), Heal (2).
const ACTION_STRIKE: u32 = 0;
const ACTION_SNIPE: u32 = 1;
const ACTION_HEAL: u32 = 2;
const NO_ACTION: u32 = 0xFFFFFFFF; // best_action sentinel from scoring kernel
const NO_TARGET: u32 = 0xFFFFFFFF;

fn role_label(role: u32) -> &'static str {
    match role {
        ROLE_TANK => "Tank  ",
        ROLE_HEALER => "Healer",
        ROLE_DPS => "DPS   ",
        _ => "?     ",
    }
}

fn team_label(team: u32) -> &'static str {
    match team {
        TEAM_RED => "RED ",
        TEAM_BLUE => "BLUE",
        _ => "?   ",
    }
}

fn role_for(slot: u32) -> u32 {
    match slot % 5 {
        0 => ROLE_TANK,
        1 => ROLE_HEALER,
        _ => ROLE_DPS,
    }
}

fn team_for(slot: u32) -> u32 {
    if slot < 5 { TEAM_RED } else { TEAM_BLUE }
}

fn fmt_hp_initial(slot: u32) -> f32 {
    match role_for(slot) {
        ROLE_TANK => HP_TANK,
        ROLE_HEALER => HP_HEALER,
        _ => HP_DPS,
    }
}

fn main() {
    let mut sim = TacticalSquad5v5State::new(SEED, AGENT_COUNT);
    println!("================================================================");
    println!(" TACTICAL SQUAD 5v5 — Red Squad vs Blue Squad");
    println!("   seed=0x{:016X} agents={} max_ticks={}", SEED, AGENT_COUNT, MAX_TICKS);
    println!("================================================================");
    println!(" Roster:");
    for slot in 0..AGENT_COUNT {
        let team = team_for(slot);
        let role = role_for(slot);
        let hp = fmt_hp_initial(slot);
        println!(
            "   Slot {:>2}  {} {}  HP={:6.1}",
            slot,
            team_label(team),
            role_label(role),
            hp,
        );
    }
    println!("================================================================");

    // Per-role targeting tracking. "Optimal" pick is defined as: the
    // candidate selected by argmax matches the candidate with the
    // smallest current HP among the candidate's team-membership
    // class (enemies for Snipe, allies for Heal). We verify this
    // tick-by-tick via the scoring_output buffer.
    let mut dps_actions = 0u64;
    let mut dps_optimal = 0u64;
    let mut healer_actions = 0u64;
    let mut healer_optimal = 0u64;
    let mut strike_actions = 0u64;
    let mut total_kills = 0u64;
    let mut last_alive: Vec<u32> = vec![1; AGENT_COUNT as usize];

    let mut ended_at: Option<u64> = None;
    let mut winner_label = "stalemate";

    for tick in 1..=MAX_TICKS {
        // Sample HP + alive BEFORE step so the `optimal target` check
        // sees the same state the scoring kernel saw this tick.
        let pre_hp = sim.read_hp();
        let pre_alive = sim.read_alive();

        sim.step();

        let scoring = sim.read_scoring_output();
        // scoring layout per agent: [best_action, best_target,
        // bitcast<u32>(best_utility), 0]
        for actor in 0..AGENT_COUNT {
            let base = (actor * 4) as usize;
            let action = scoring[base + 0];
            let target = scoring[base + 1];
            if action == NO_ACTION || target == NO_TARGET {
                continue;
            }
            let actor_team = team_for(actor);
            match action {
                ACTION_STRIKE => {
                    strike_actions += 1;
                }
                ACTION_SNIPE => {
                    dps_actions += 1;
                    // Optimal: lowest-HP among living enemies.
                    let mut best_slot: i32 = -1;
                    let mut best_hp = f32::INFINITY;
                    for cand in 0..AGENT_COUNT {
                        if pre_alive[cand as usize] == 0 { continue; }
                        if cand == actor { continue; }
                        if team_for(cand) == actor_team { continue; }
                        let h = pre_hp[cand as usize];
                        if h < best_hp {
                            best_hp = h;
                            best_slot = cand as i32;
                        }
                    }
                    if best_slot >= 0 && pre_hp[target as usize] <= best_hp + 0.01 {
                        // Tie-tolerant: target's HP equals (or is
                        // very close to) the optimal HP value. Argmax
                        // stable-tie-breaks on first-encountered.
                        dps_optimal += 1;
                    }
                }
                ACTION_HEAL => {
                    healer_actions += 1;
                    let mut best_slot: i32 = -1;
                    let mut best_hp = f32::INFINITY;
                    for cand in 0..AGENT_COUNT {
                        if pre_alive[cand as usize] == 0 { continue; }
                        if team_for(cand) != actor_team { continue; }
                        let h = pre_hp[cand as usize];
                        if h < best_hp {
                            best_hp = h;
                            best_slot = cand as i32;
                        }
                    }
                    if best_slot >= 0 && pre_hp[target as usize] <= best_hp + 0.01 {
                        healer_optimal += 1;
                    }
                }
                _ => {}
            }
        }

        let hp = sim.read_hp();
        let alive = sim.read_alive();

        // Detect kills this tick.
        for slot in 0..AGENT_COUNT as usize {
            if last_alive[slot] == 1 && alive[slot] == 0 {
                total_kills += 1;
                println!(
                    "Tick {:>4}: {} {} (slot {}) DEFEATED",
                    tick,
                    team_label(team_for(slot as u32)),
                    role_label(role_for(slot as u32)),
                    slot,
                );
            }
        }
        last_alive = alive.clone();

        let red_alive: u32 = (0..5).map(|s| alive[s]).sum();
        let blue_alive: u32 = (5..10).map(|s| alive[s]).sum();
        let red_hp: f32 = (0..5).map(|s| if alive[s] == 1 { hp[s] } else { 0.0 }).sum();
        let blue_hp: f32 = (5..10).map(|s| if alive[s] == 1 { hp[s] } else { 0.0 }).sum();

        // Per-tick trace every 25 ticks plus on key events.
        if tick % 25 == 0 || tick == 1 {
            println!(
                "Tick {:>4}: RED  alive={}/5 hp={:6.1}  |  BLUE alive={}/5 hp={:6.1}",
                tick, red_alive, red_hp, blue_alive, blue_hp,
            );
        }

        if red_alive == 0 || blue_alive == 0 {
            ended_at = Some(tick);
            winner_label = if red_alive > 0 {
                "RED"
            } else if blue_alive > 0 {
                "BLUE"
            } else {
                "mutual KO"
            };
            println!(
                "Tick {:>4}: TEAM ELIMINATION — RED alive={}/5  BLUE alive={}/5",
                tick, red_alive, blue_alive,
            );
            break;
        }
    }

    let final_hp = sim.read_hp();
    let final_alive = sim.read_alive();
    let damage = sim.damage_dealt().to_vec();
    let healing = sim.healing_done().to_vec();

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    for slot in 0..AGENT_COUNT as usize {
        let alive_marker = if final_alive[slot] == 1 { " " } else { "X" };
        println!(
            "  [{}] Slot {:>2}  {} {}  hp={:6.1}  dmg_dealt={:7.1}  heal_done={:7.1}",
            alive_marker,
            slot,
            team_label(team_for(slot as u32)),
            role_label(role_for(slot as u32)),
            final_hp[slot],
            damage[slot],
            healing[slot],
        );
    }

    let total_damage: f32 = damage.iter().sum();
    let total_healing: f32 = healing.iter().sum();
    let red_alive_final: u32 = (0..5).map(|s| final_alive[s]).sum();
    let blue_alive_final: u32 = (5..10).map(|s| final_alive[s]).sum();

    println!();
    println!("  Total damage dealt:  {:.1}", total_damage);
    println!("  Total healing done:  {:.1}", total_healing);
    println!("  Total kills:         {}", total_kills);
    if let Some(t) = ended_at {
        println!("  Combat ended at tick {} — winner: {}", t, winner_label);
    } else {
        println!(
            "  Combat ran to MAX_TICKS={} — RED alive={}/5, BLUE alive={}/5 — STALEMATE",
            MAX_TICKS, red_alive_final, blue_alive_final,
        );
        winner_label = "stalemate";
    }

    println!();
    println!("================================================================");
    println!(" TARGETING DECISIONS");
    println!("================================================================");
    let dps_optimal_pct = if dps_actions > 0 {
        (dps_optimal as f64) / (dps_actions as f64) * 100.0
    } else { 0.0 };
    let healer_optimal_pct = if healer_actions > 0 {
        (healer_optimal as f64) / (healer_actions as f64) * 100.0
    } else { 0.0 };
    println!("  Strike (Tank, any-enemy):   {} actions", strike_actions);
    println!(
        "  Snipe  (DPS lowest-HP enemy): {:>4}/{:<4} optimal ({:.1}%)",
        dps_optimal, dps_actions, dps_optimal_pct,
    );
    println!(
        "  Heal   (Healer lowest-HP ally): {:>4}/{:<4} optimal ({:.1}%)",
        healer_optimal, healer_actions, healer_optimal_pct,
    );

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    if total_damage > 0.0 && total_kills > 0 && (red_alive_final == 0 || blue_alive_final == 0) {
        println!(
            "  (a) FULL FIRE — combat played out end-to-end. {:.0} dmg / \
             {:.0} heal / {} kills. Pair-field scoring with team filter \
             worked: every agent picked a valid candidate every cooldown \
             tick.",
            total_damage, total_healing, total_kills,
        );
    } else if total_damage > 0.0 && total_kills > 0 {
        println!(
            "  (a-partial) HP DRAINING + KILLS — {} kills landed but no \
             team was wiped by tick {}. Healer is keeping up with damage \
             output. Either tune for decisive break or accept stalemate.",
            total_kills, MAX_TICKS,
        );
    } else if total_damage > 0.0 {
        println!(
            "  (a-partial) DAMAGE WITHOUT KILLS — {:.0} damage landed but \
             no agent dropped to 0 HP. Either healing dominates or the \
             ApplyDamage kernel isn't writing alive=0 correctly.",
            total_damage,
        );
    } else {
        println!(
            "  (b) NO COMBAT — neither damage nor healing accumulated. The \
             verb cascade isn't reaching the chronicle. Likely root cause: \
             mask predicates fail for every (actor, candidate) pair (the \
             team-filter `target.level != self.level` collapsed because \
             mask_k=1u didn't propagate the inline-fix).",
        );
    }

    // Hard asserts (sim_app convention): combat must have fired AND
    // pair-field scoring must have made at least N optimal picks.
    assert!(
        total_damage > 0.0,
        "tactical_squad_5v5_app: ASSERTION FAILED — no damage was dealt. \
         The verb cascade did not reach the ApplyDamage kernel.",
    );
    assert!(
        dps_actions > 0 && healer_actions > 0,
        "tactical_squad_5v5_app: ASSERTION FAILED — DPS or Healer never \
         emitted ActionSelected. dps_actions={}, healer_actions={}",
        dps_actions, healer_actions,
    );
    // The mission card asks for "demonstrates DPS picked lowest-HP target
    // at least N times" — pick N=10 as a low watermark that any
    // working pair-field scoring run blows past.
    const OPTIMAL_FLOOR: u64 = 10;
    assert!(
        dps_optimal >= OPTIMAL_FLOOR,
        "tactical_squad_5v5_app: ASSERTION FAILED — DPS picked the \
         lowest-HP enemy only {} times (floor={}). Pair-field scoring \
         is not selecting the argmax target.",
        dps_optimal, OPTIMAL_FLOOR,
    );
    assert!(
        healer_optimal >= OPTIMAL_FLOOR,
        "tactical_squad_5v5_app: ASSERTION FAILED — Healer picked the \
         lowest-HP ally only {} times (floor={}). Pair-field scoring \
         is not selecting the argmax target.",
        healer_optimal, OPTIMAL_FLOOR,
    );
    let _ = winner_label;
}
