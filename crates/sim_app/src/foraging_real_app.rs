//! Foraging-real harness — drives `foraging_real_runtime` and reports
//! per-tick lifecycle: alive ant count, total stockpile credit, food
//! remaining, plus births / deaths / eats deltas.
//!
//! This is the FIRST fixture where the alive-bitmap actually flips
//! BOTH directions during a single run:
//!   - alive=true → false: per-ant `EnergyDecay` writes `set_alive
//!     (self, false)` when hunger reaches 0. ApplyEat writes
//!     `set_alive(food, false)` when food.hp depletes.
//!   - alive=false → true: CPU-side `process_births()` reads the
//!     `eat_count` view, computes the colony stockpile, and flips
//!     dead Ant slots back to alive=1 with reset state.
//!
//! Termination: stops when (a) no live ants remain (colony collapse),
//! (b) no live food remains AND no births can fire (starvation
//! winding down), or (c) MAX_TICKS reached.

use engine::CompiledSim;
use foraging_real_runtime::{
    ForagingRealState, INITIAL_ANTS, INITIAL_FOOD, SLOT_CAP,
};

const SEED: u64 = 0xF02A_61CA_7C01_01_F1;
const MAX_TICKS: u64 = 800;
const TRACE_EVERY: u64 = 25;

fn main() {
    let mut sim = ForagingRealState::new(SEED, SLOT_CAP);
    println!("================================================================");
    println!(" FORAGING REAL — Ant Colony with Lifecycle (births + deaths)");
    println!(
        "   seed=0x{:016X} slot_cap={} initial_ants={} initial_food={}",
        SEED, SLOT_CAP, INITIAL_ANTS, INITIAL_FOOD,
    );
    println!("   max_ticks={} trace_every={}", MAX_TICKS, TRACE_EVERY);
    println!("================================================================");

    let initial_ants = sim.count_alive_ants();
    let initial_food_count = sim.count_alive_food();
    let initial_food_qty = sim.total_food_remaining();
    println!(
        "Tick    0: ants={:>3} food_piles={:>2} food_qty={:>5.0} \
         births=0 deaths=0 stockpile=0 (initial)",
        initial_ants, initial_food_count, initial_food_qty,
    );

    let mut peak_ants: u32 = initial_ants;
    let mut last_trace_tick: u64 = 0;
    let mut ended_at: Option<u64> = None;
    let mut end_reason = "ran to MAX_TICKS";

    for tick in 1..=MAX_TICKS {
        sim.step();
        let lc = sim.last_lifecycle();
        let alive_ants = sim.count_alive_ants();
        let alive_food = sim.count_alive_food();
        let food_qty = sim.total_food_remaining();
        let stockpile_credit: f32 = sim.eat_count().iter().sum();

        if alive_ants > peak_ants {
            peak_ants = alive_ants;
        }

        let interesting =
            lc.births_this_tick > 0 || lc.deaths_this_tick > 0;
        if tick % TRACE_EVERY == 0 || tick == 1 || interesting {
            // Throttle the per-event traces — collapse runs of single-
            // event ticks into "B/D ticks" reports.
            if tick - last_trace_tick > 0 {
                println!(
                    "Tick {:>4}: ants={:>3} food_piles={:>2} food_qty={:>5.0} \
                     +births={:>2} +deaths={:>2} +eats={:>3} stockpile={:>5.0}",
                    tick,
                    alive_ants,
                    alive_food,
                    food_qty,
                    lc.births_this_tick,
                    lc.deaths_this_tick,
                    lc.eats_this_tick,
                    stockpile_credit,
                );
                last_trace_tick = tick;
            }
        }

        // Termination conditions.
        if alive_ants == 0 {
            ended_at = Some(tick);
            end_reason = "colony collapsed (zero live ants)";
            println!(
                "Tick {:>4}: ants=0  ← COLONY COLLAPSE (no surviving ants)",
                tick,
            );
            break;
        }
        if alive_food == 0 && food_qty <= 0.0 {
            // Food fully exhausted — let the colony starve down briefly
            // so we observe the death cascade, then bail.
            if alive_ants <= 1 {
                ended_at = Some(tick);
                end_reason = "food exhausted, colony winding down";
                println!(
                    "Tick {:>4}: food_piles=0  ← FOOD EXHAUSTED (last ants starving)",
                    tick,
                );
                break;
            }
        }
    }

    let final_ants = sim.count_alive_ants();
    let final_food_piles = sim.count_alive_food();
    let final_food_qty = sim.total_food_remaining();
    let total_eats: f32 = sim.eat_count().iter().sum();
    let total_starved: f32 = sim.starved_count().iter().sum();
    let total_depleted: f32 = sim.depleted_count().iter().sum();
    let births_total = sim.births_so_far();
    let deaths_total = sim.deaths_so_far();
    let final_tick = ended_at.unwrap_or(MAX_TICKS);

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  final tick:         {}", final_tick);
    println!("  end reason:         {}", end_reason);
    println!("  final ants:         {} / cap {}", final_ants, SLOT_CAP);
    println!("  peak ants:          {}", peak_ants);
    println!("  final food piles:   {} / initial {}", final_food_piles, INITIAL_FOOD);
    println!("  final food qty:     {:.0}", final_food_qty);
    println!("  total eats:         {:.0}", total_eats);
    println!("  total births:       {} (alive=false → true CPU-flips)", births_total);
    println!("  total starvations:  {:.0} (alive=true → false from EnergyDecay)", total_starved);
    println!("  total food deplete: {:.0} (alive=true → false from ApplyEat)", total_depleted);
    println!("  ant deaths counter: {}", deaths_total);

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    let any_births = births_total > 0;
    let any_deaths = total_starved > 0.0 || total_depleted > 0.0;
    let any_eats = total_eats > 0.0;

    if any_eats && any_births && any_deaths {
        println!(
            "  (a) FULL FIRE — full lifecycle exercised. {:.0} eats fed \
             {} births and {:.0} ant starvations + {:.0} food depletions. \
             alive bitmap flipped BOTH directions ({} → and {} ←).",
            total_eats,
            births_total,
            total_starved,
            total_depleted,
            births_total,
            total_starved as u32 + total_depleted as u32,
        );
    } else if any_eats && any_deaths && !any_births {
        println!(
            "  (a-partial) EATS + DEATHS BUT NO BIRTHS — {:.0} eats / \
             {:.0} starvations / {:.0} depletions registered, but no \
             birth ever flipped a dead slot to alive=1. Birth threshold \
             may be too high, or `process_births()` isn't running, or \
             eat_count view isn't accumulating.",
            total_eats, total_starved, total_depleted,
        );
    } else if any_eats && any_births && !any_deaths {
        println!(
            "  (a-partial) EATS + BIRTHS BUT NO DEATHS — {:.0} eats / \
             {} births registered, but no ant ever starved and no food \
             ever depleted. EnergyDecay's alive-flip gate or ApplyEat's \
             food-deplete gate may not be lowering correctly.",
            total_eats, births_total,
        );
    } else if any_eats {
        println!(
            "  (b) EATS ONLY — {:.0} eats fired but neither births nor \
             deaths. Both lifecycle paths failed to surface.",
            total_eats,
        );
    } else {
        println!(
            "  (b) NO EATS — neither lifecycle direction exercised. \
             AntFeed emitted no Eat events; spatial walk likely not \
             finding food piles. Check the body-side `(other.creature_type \
             == FoodPile && other.alive)` lowering.",
        );
    }

    // Hard asserts on the killer-feature claims of this fixture.
    assert!(any_eats, "foraging_real_app: no Eat events fired — spatial body-form path is broken.");
    assert!(
        any_deaths,
        "foraging_real_app: no deaths occurred — DSL set_alive(self, false) path is broken.",
    );
    assert!(
        any_births,
        "foraging_real_app: no births occurred — CPU-side alive=false → true flip path is broken.",
    );
}
