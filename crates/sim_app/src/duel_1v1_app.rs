//! Duel 1v1 harness — drives `duel_1v1_runtime` for up to 200 ticks
//! (or until a Defeated event fires) and reports the per-tick combat
//! log + final winner.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL FIRE — full cascade plays out
//!
//! Two Combatants start at hp=100, mana=100, alive=1. Per tick:
//!   - Strike scores 1.0, fires every 2nd tick → 12 dmg
//!   - Spell scores `(25*2 - target.hp)`, fires every 5th tick → 25 dmg
//!     (only when self.mana >= 30 — but mana never decrements today)
//!   - Heal scores 100.0, fires every 4th tick when self.hp < 30
//!
//! Strike alone deals ~12 dmg every 2 ticks. With both heroes alive and
//! exchanging blows, the duel converges in ~17 ticks (100 hp / 12 dmg
//! per 2-tick window ≈ 16 turns). When Spell wins argmax (every 5th
//! tick when target HP <50), one big swing deals 25.
//!
//! Final state: one survives, one is defeated. damage_dealt > 0 for
//! both. damage_dealt[winner] > damage_dealt[loser] (winner got the
//! killing blow).
//!
//! ### (b) NO FIRE — gap surfaces
//!
//! Documented in `docs/superpowers/notes/2026-05-04-duel_1v1.md`.
//! Likely surfaces:
//!   - HP not draining (apply kernel didn't actually write hp)
//!   - Both heroes survive at hp=100 (no Damaged events folded)
//!   - Single hero loses HP unilaterally (mask asymmetry / pair indexing)

use duel_1v1_runtime::Duel1v1State;
use engine::CompiledSim;

const SEED: u64 = 0xDEADBEEF_CAFE_F00D;
const AGENT_COUNT: u32 = 2;
const MAX_TICKS: u64 = 200;
const STRIKE_DAMAGE: f32 = 12.0;
const SPELL_DAMAGE: f32 = 25.0;
const HEAL_AMOUNT: f32 = 18.0;

fn main() {
    let mut sim = Duel1v1State::new(SEED, AGENT_COUNT);
    println!("================================================================");
    println!(" DUEL 1v1 — Hero A vs Hero B");
    println!("   seed=0x{:016X} agents={} max_ticks={}", SEED, AGENT_COUNT, MAX_TICKS);
    println!("   strike={:.1} dmg, spell={:.1} dmg, heal={:.1}", STRIKE_DAMAGE, SPELL_DAMAGE, HEAL_AMOUNT);
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

        // Log every 10 ticks OR when something interesting happens.
        let interesting = dmg_a.abs() > 0.01 || dmg_b.abs() > 0.01;
        if tick % 10 == 0 || interesting && tick % 2 == 0 {
            println!(
                "Tick {:>3}: HP A={:6.2} B={:6.2} (ΔA={:+.1} ΔB={:+.1}) \
                 alive=[{}, {}]",
                tick, a_hp, b_hp, -dmg_a, -dmg_b, alive[0], alive[1],
            );
        }

        // Stop on first defeated.
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
                "Tick {:>3}: HP A={:6.2} B={:6.2} alive=[{}, {}]  ← DEFEATED",
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
        println!("  Combat ended at tick {} — winner: {}", t, winner_label);
    } else {
        println!("  Combat ran to MAX_TICKS={} without resolution", MAX_TICKS);
        if final_alive[0] == 1 && final_alive[1] == 1 {
            winner_label = "draw (both alive)";
        }
        println!("  Outcome: {}", winner_label);
    }

    let total_damage: f32 = damage.iter().sum();
    let total_healing: f32 = healing.iter().sum();
    let any_damage_taken = (final_hp[0] - 100.0).abs() > 0.01 || (final_hp[1] - 100.0).abs() > 0.01;
    let any_alive_zero = final_alive[0] == 0 || final_alive[1] == 0;

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    if total_damage > 0.0 && any_damage_taken && any_alive_zero {
        println!(
            "  (a) FULL FIRE — combat played out end-to-end. {:.0} total dmg, \
            {:.0} total healing, one hero defeated. The verb cascade + \
            scoring + chronicle + ApplyDamage + ApplyHeal pipeline all wired.",
            total_damage, total_healing,
        );
    } else if total_damage > 0.0 && any_damage_taken {
        println!(
            "  (a-partial) HP DRAINING — combat is firing ({:.0} dmg dealt, \
            HP changed) but no one defeated yet by tick {}. Either combat \
            stalemates or the threshold guard isn't triggered.",
            total_damage, MAX_TICKS,
        );
    } else if total_damage > 0.0 {
        println!(
            "  (b) DAMAGE FOLDED BUT HP NOT WRITTEN — damage_dealt view \
            accumulates ({:.0} total) but per-agent HP never changes. \
            Suggests ApplyDamage kernel runs but the agents.set_hp write \
            isn't reaching the GPU buffer (binding shape mismatch?).",
            total_damage,
        );
    } else if total_healing > 0.0 {
        println!(
            "  (b) ONLY HEALING FIRED — Heal won every argmax over Strike/\
            Spell. The per-tick mask predicates for Strike/Spell aren't \
            firing on the duel pair (alive check, target!=self, or mod \
            cooldown gate).",
        );
    } else {
        println!(
            "  (b) NO COMBAT — neither damage_dealt nor healing_done \
            accumulated. The verb cascade isn't reaching the chronicle \
            for either Strike, Spell, or Heal. Likely root cause: \
            scoring kernel emits no ActionSelected events (mask predicates \
            fail for both heroes). See discovery doc for gap chain.",
        );
    }

    // Hard assert (sim_app convention): at least one combatant takes
    // damage. Fail loudly otherwise.
    assert!(
        any_damage_taken,
        "duel_1v1_app: ASSERTION FAILED — neither hero took damage \
         (HP A={:.2}, B={:.2}). The verb cascade did not reach the \
         ApplyDamage kernel. See OUTCOME above and discovery doc.",
        final_hp[0], final_hp[1],
    );
    assert!(
        total_damage > 0.0,
        "duel_1v1_app: ASSERTION FAILED — damage_dealt view is empty. \
         The fold_damage_dealt kernel did not consume any Damaged events.",
    );
}
