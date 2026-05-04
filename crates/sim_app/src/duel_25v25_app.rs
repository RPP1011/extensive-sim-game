//! Duel 25v25 harness — drives `duel_25v25_runtime` for up to 500
//! ticks (or until one squad is fully eliminated, whichever comes
//! first) and reports per-100-tick squad survivor counts + a final
//! winner declaration.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — squad battle plays out
//!
//! 25 Red + 25 Blue start at hp=50, alive=1. Per tick:
//!   - Spatial-hash builds; ScanAndStrike walks 27-cell neighbourhood
//!     per agent; emits Damaged for any opposing-creature_type
//!     neighbour in range (gated every other tick by `world.tick % 2`).
//!   - ApplyDamage chronicle subtracts strike_damage (5.0) per Damaged
//!     event, sets alive=false on HP<=0, emits Defeated.
//!
//! Each Combatant in the contested zone takes ~5-15 dmg per active
//! tick (depending on neighbour density). With 50 hp, agents typically
//! fall in ~5-15 active ticks (~10-30 wall ticks). Battle resolves
//! within ~50-200 ticks, with one squad fully dead.
//!
//! ### (b) NO FIRE — gap surfaces
//!
//! Likely shapes:
//!   - HP unchanged at 50 → ScanAndStrike emitted 0 Damaged events
//!     (spatial walk filter or creature_type comparison didn't lower
//!     correctly; check ScanAndStrike WGSL).
//!   - All Red HP drops but Blue stays at 50 → asymmetric self-cell
//!     bug (only one team's neighbours fire emits).
//!   - Both teams take damage but no one dies → ApplyDamage's HP<=0
//!     gate didn't lower (check `set_alive` on non-self target —
//!     duel_1v1 surfaced this same gap).
//!
//! ## Discovery doc
//!
//! `docs/superpowers/notes/2026-05-04-duel_25v25.md` — same gap
//! list as duel_1v1 (P6, cycle, mask_k placeholder) plus 25v25-
//! specific notes.

use duel_25v25_runtime::Duel25v25State;
use engine::CompiledSim;

const SEED: u64 = 0xDEADBEEF_CAFE_F00D;
const AGENT_COUNT: u32 = 50;
const MAX_TICKS: u64 = 500;
const STRIKE_DAMAGE: f32 = 5.0;

fn count_team(alive: &[u32], creature_type: &[u32], team_disc: u32) -> u32 {
    alive
        .iter()
        .zip(creature_type.iter())
        .filter(|(a, c)| **a == 1 && **c == team_disc)
        .count() as u32
}

fn main() {
    let mut sim = Duel25v25State::new(SEED, AGENT_COUNT);
    println!("================================================================");
    println!(" DUEL 25v25 — Red Squad vs Blue Squad");
    println!(
        "   seed=0x{:016X} agents={} max_ticks={}",
        SEED, AGENT_COUNT, MAX_TICKS,
    );
    println!("   strike={:.1} dmg per emit, hp=50.0 each", STRIKE_DAMAGE);
    println!("================================================================");

    let creature_type = sim.read_creature_type();
    let initial_alive = sim.read_alive();
    let initial_hp = sim.read_hp();
    let initial_red = count_team(&initial_alive, &creature_type, 0);
    let initial_blue = count_team(&initial_alive, &creature_type, 1);
    println!(
        "Tick   0: Red={:2} Blue={:2}  total_hp(Red)={:.0} total_hp(Blue)={:.0}",
        initial_red,
        initial_blue,
        team_total_hp(&initial_hp, &creature_type, 0),
        team_total_hp(&initial_hp, &creature_type, 1),
    );

    let mut ended_at: Option<u64> = None;
    let mut winner_label: &'static str = "draw";

    for tick in 1..=MAX_TICKS {
        sim.step();

        // Per-100-tick log (cheap — readback only every 100 ticks).
        if tick % 100 == 0 || tick == 1 {
            let alive = sim.read_alive();
            let hp = sim.read_hp();
            let red = count_team(&alive, &creature_type, 0);
            let blue = count_team(&alive, &creature_type, 1);
            println!(
                "Tick {:>3}: Red={:2} Blue={:2}  total_hp(Red)={:.0} total_hp(Blue)={:.0}",
                tick,
                red,
                blue,
                team_total_hp(&hp, &creature_type, 0),
                team_total_hp(&hp, &creature_type, 1),
            );
        }

        // Stop on full-team eliminated. Readback every 25 ticks so we
        // don't miss the resolution boundary.
        if tick % 25 == 0 {
            let alive = sim.read_alive();
            let red = count_team(&alive, &creature_type, 0);
            let blue = count_team(&alive, &creature_type, 1);
            if red == 0 || blue == 0 {
                ended_at = Some(tick);
                winner_label = if red == 0 && blue == 0 {
                    "mutual annihilation"
                } else if red == 0 {
                    "Blue Squad"
                } else {
                    "Red Squad"
                };
                let hp = sim.read_hp();
                println!(
                    "Tick {:>3}: Red={:2} Blue={:2}  total_hp(Red)={:.0} total_hp(Blue)={:.0}  ← BATTLE ENDS",
                    tick,
                    red,
                    blue,
                    team_total_hp(&hp, &creature_type, 0),
                    team_total_hp(&hp, &creature_type, 1),
                );
                break;
            }
        }
    }

    let final_alive = sim.read_alive();
    let final_hp = sim.read_hp();
    let final_red = count_team(&final_alive, &creature_type, 0);
    let final_blue = count_team(&final_alive, &creature_type, 1);
    let damage = sim.damage_dealt().to_vec();
    let defeats = sim.defeats_received().to_vec();

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  Red survivors:    {:2} / 25", final_red);
    println!("  Blue survivors:   {:2} / 25", final_blue);
    println!(
        "  Red total HP:     {:.0}",
        team_total_hp(&final_hp, &creature_type, 0)
    );
    println!(
        "  Blue total HP:    {:.0}",
        team_total_hp(&final_hp, &creature_type, 1)
    );
    let total_damage: f32 = damage.iter().sum();
    let total_defeats: f32 = defeats.iter().sum();
    println!("  total damage:     {:.1}", total_damage);
    println!("  total defeats:    {:.0}", total_defeats);

    if let Some(t) = ended_at {
        println!("  Battle ended at tick {} — winner: {}", t, winner_label);
    } else {
        println!(
            "  Battle ran to MAX_TICKS={} without full elimination.",
            MAX_TICKS,
        );
        winner_label = if final_red > final_blue {
            "Red Squad (by survivor count)"
        } else if final_blue > final_red {
            "Blue Squad (by survivor count)"
        } else {
            "draw (equal survivors)"
        };
        println!("  Outcome: {}", winner_label);
    }

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    let any_damage = total_damage > 0.0;
    let any_defeat = total_defeats > 0.0;
    if any_damage && any_defeat && (final_red == 0 || final_blue == 0) {
        println!(
            "  (a) FULL FIRE — squad battle resolved end-to-end. \
             {:.0} total dmg dealt, {:.0} agents defeated, one squad fully \
             eliminated. Spatial body-form physics + chronicle damage \
             pipeline all wired at scale.",
            total_damage, total_defeats,
        );
    } else if any_damage && any_defeat {
        println!(
            "  (a-partial) BATTLE FIRED, NO RESOLUTION — combat is \
             happening ({:.0} dmg, {:.0} defeats) but neither squad fully \
             eliminated by tick {}. Either the engagement zone resolved \
             into a draw or one side's losses didn't propagate to a kill.",
            total_damage, total_defeats, MAX_TICKS,
        );
    } else if any_damage {
        println!(
            "  (b) DAMAGE BUT NO DEFEATS — {:.0} dmg dealt but no Defeated \
             events fired. ApplyDamage's `if (new_hp <= 0.0) {{ set_alive \
             … emit Defeated }}` cascade may not lower (check the WGSL \
             for the IfStmt + emit fan-out).",
            total_damage,
        );
    } else {
        println!(
            "  (b) NO COMBAT — neither damage nor defeats accumulated. \
             ScanAndStrike emitted no Damaged events; spatial walk likely \
             not finding cross-team neighbours. Check the body-side \
             `if (other.creature_type != self.creature_type)` lowering."
        );
    }

    // Hard assert: at least one Combatant takes damage.
    assert!(
        any_damage,
        "duel_25v25_app: ASSERTION FAILED — no damage dealt. Spatial \
         body-form ScanAndStrike never fired a Damaged event.",
    );
}

fn team_total_hp(hp: &[f32], creature_type: &[u32], team_disc: u32) -> f32 {
    hp.iter()
        .zip(creature_type.iter())
        .filter(|(_, c)| **c == team_disc)
        .map(|(h, _)| h.max(0.0))
        .sum()
}
