//! Duel abilities harness — drives `duel_abilities_runtime` for up to
//! 200 ticks (or until a Defeated event fires) and reports the per-tick
//! combat log + final winner.
//!
//! Mirrors `duel_1v1_app.rs` shape but with the Strike/ShieldUp/Mend
//! constants drawn from the .ability files (which the runtime asserts
//! at startup via `binding_check`). The fixture's reason for existing
//! is the binding check at construction time — actually rendering combat
//! is downstream proof the .sim emit pipeline still works for the
//! mirrored constants.

use duel_abilities_runtime::DuelAbilitiesState;
use engine::CompiledSim;

const SEED: u64 = 0xDEADBEEF_CAFE_F00D;
const AGENT_COUNT: u32 = 2;
const MAX_TICKS: u64 = 200;

const STRIKE_DAMAGE: f32 = 15.0;    // Strike.ability damage 15
const SHIELDUP_AMOUNT: f32 = 50.0;  // ShieldUp.ability shield 50 (modelled as +HP)
const MEND_AMOUNT: f32 = 25.0;      // Mend.ability heal 25

fn main() {
    // Construction runs the binding check — if any .ability file's
    // lowered AbilityProgram constants drift from the .sim's
    // hand-mirrored verb constants, this panics with a descriptive
    // message before any GPU work happens.
    let mut sim = DuelAbilitiesState::new(SEED, AGENT_COUNT);
    println!("================================================================");
    println!(" DUEL ABILITIES — Hero A vs Hero B");
    println!("   seed=0x{:016X} agents={} max_ticks={}", SEED, AGENT_COUNT, MAX_TICKS);
    println!("   (constants from .ability files: strike={:.1} shield={:.1} mend={:.1})",
        STRIKE_DAMAGE, SHIELDUP_AMOUNT, MEND_AMOUNT);
    println!("================================================================");

    let initial_hp = sim.read_hp();
    let initial_alive = sim.read_alive();
    println!("Tick   0: HP A={:6.2} B={:6.2} alive=[{}, {}]",
        initial_hp[0], initial_hp[1], initial_alive[0], initial_alive[1]);
    let mut last_a = initial_hp[0];
    let mut last_b = initial_hp[1];

    let mut ended_at: Option<u64> = None;
    let mut winner_label = "draw";

    for tick in 1..=MAX_TICKS {
        sim.step();

        let hp = sim.read_hp();
        let alive = sim.read_alive();
        let a_hp = hp[0];
        let b_hp = hp[1];
        let dmg_a = last_a - a_hp;
        let dmg_b = last_b - b_hp;

        let interesting = dmg_a.abs() > 0.01 || dmg_b.abs() > 0.01;
        if tick % 10 == 0 || interesting && tick % 2 == 0 {
            println!(
                "Tick {:>3}: HP A={:6.2} B={:6.2} (\u{0394}A={:+.1} \u{0394}B={:+.1}) alive=[{}, {}]",
                tick, a_hp, b_hp, -dmg_a, -dmg_b, alive[0], alive[1],
            );
        }

        if alive[0] == 0 || alive[1] == 0 {
            ended_at = Some(tick);
            winner_label = if alive[0] == 1 && alive[1] == 0 {
                "Hero A"
            } else if alive[1] == 1 && alive[0] == 0 {
                "Hero B"
            } else {
                "mutual KO"
            };
            println!(
                "Tick {:>3}: HP A={:6.2} B={:6.2} alive=[{}, {}]  <- DEFEATED",
                tick, a_hp, b_hp, alive[0], alive[1],
            );
            break;
        }

        last_a = a_hp;
        last_b = b_hp;
    }

    let final_hp = sim.read_hp();
    let final_alive = sim.read_alive();
    let damage = sim.damage_dealt().to_vec();
    let healing = sim.healing_done().to_vec();

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  Final HP:        A={:.2}, B={:.2}", final_hp[0], final_hp[1]);
    println!("  Final alive:     A={}, B={}", final_alive[0], final_alive[1]);
    println!("  damage_dealt:    A={:.2}, B={:.2}", damage[0], damage[1]);
    println!("  healing_done:    A={:.2}, B={:.2}", healing[0], healing[1]);
    if let Some(t) = ended_at {
        println!("  Combat ended at tick {} - winner: {}", t, winner_label);
    } else {
        println!("  Combat ran to MAX_TICKS={} without resolution", MAX_TICKS);
        if final_alive[0] == 1 && final_alive[1] == 1 {
            winner_label = "draw (both alive)";
        }
        println!("  Outcome: {}", winner_label);
    }

    let any_damage_taken = (final_hp[0] - 100.0).abs() > 0.01 || (final_hp[1] - 100.0).abs() > 0.01;

    assert!(
        any_damage_taken,
        "duel_abilities_app: ASSERTION FAILED - neither hero took damage \
         (HP A={:.2}, B={:.2}). The verb cascade did not reach the \
         ApplyDamage kernel.",
        final_hp[0], final_hp[1],
    );
}
