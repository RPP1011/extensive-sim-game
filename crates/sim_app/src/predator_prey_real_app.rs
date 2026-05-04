//! Predator-prey real harness — drives `predator_prey_real_runtime`
//! and reports per-tick population dynamics: alive wolf count, alive
//! sheep count, kills + births + starvations across both species.
//!
//! This is the FOURTH REAL fixture and the FIRST to compose:
//!   - COMBAT (wolf strikes sheep via per-tick spatial walk emitting
//!     Damaged events, processed by ApplyKill chronicle), AND
//!   - LIFECYCLE (energy decay → death; eats / well-fed-window
//!     threshold → birth)
//! ON TWO SIMULTANEOUSLY-LIVE creature types.
//!
//! Termination rules (in order of precedence):
//!   - Both extinct: alive_wolves == 0 AND alive_sheep == 0 → "both extinct"
//!   - Wolves crashed: alive_wolves == 0 (sheep persist) → "wolves crashed"
//!   - Sheep crashed: alive_sheep == 0 (wolves persist) → "sheep crashed"
//!   - MAX_TICKS reached with both species alive → "stable oscillation"

use engine::CompiledSim;
use predator_prey_real_runtime::{
    PredatorPreyRealState, INITIAL_SHEEP, INITIAL_WOLVES, SHEEP_CAP, SLOT_CAP, WOLF_CAP,
};

const SEED: u64 = 0x707E_DA70_47A0_BEEF;
const MAX_TICKS: u64 = 400;
const TRACE_EVERY: u64 = 20;

fn main() {
    let mut sim = PredatorPreyRealState::new(SEED, SLOT_CAP);
    println!("================================================================");
    println!(" PREDATOR PREY REAL — Combat + Lifecycle on TWO live species");
    println!(
        "   seed=0x{:016X} slot_cap={} (wolves [0..{}) sheep [{}..{}))",
        SEED, SLOT_CAP, WOLF_CAP, WOLF_CAP, WOLF_CAP + SHEEP_CAP,
    );
    println!(
        "   initial wolves={} initial sheep={} max_ticks={} trace_every={}",
        INITIAL_WOLVES, INITIAL_SHEEP, MAX_TICKS, TRACE_EVERY,
    );
    println!("================================================================");

    let initial_wolves = sim.count_alive_wolves();
    let initial_sheep = sim.count_alive_sheep();
    println!(
        "Tick    0: wolves={:>3} sheep={:>3} kills=0 wolf_births=0 sheep_births=0 \
         wolf_starv=0 sheep_starv=0 (initial)",
        initial_wolves, initial_sheep,
    );

    let mut peak_wolves: u32 = initial_wolves;
    let mut peak_sheep: u32 = initial_sheep;
    let mut ended_at: Option<u64> = None;
    let mut end_reason = "ran to MAX_TICKS";

    for tick in 1..=MAX_TICKS {
        sim.step();
        let lc = sim.last_lifecycle();
        let alive_wolves = sim.count_alive_wolves();
        let alive_sheep = sim.count_alive_sheep();

        if alive_wolves > peak_wolves {
            peak_wolves = alive_wolves;
        }
        if alive_sheep > peak_sheep {
            peak_sheep = alive_sheep;
        }

        let interesting = lc.wolf_births > 0
            || lc.sheep_births > 0
            || lc.wolf_starvations > 0
            || lc.sheep_starvations > 0
            || lc.sheep_kills > 0;
        if tick % TRACE_EVERY == 0 || tick == 1 || (tick < 10 && interesting) {
            println!(
                "Tick {:>4}: wolves={:>3} sheep={:>3} +kills={:>2} +wb={:>2} +sb={:>2} \
                 +ws={:>2} +ss={:>2} | tot_kills={:>4} tot_wb={:>3} tot_sb={:>3}",
                tick,
                alive_wolves,
                alive_sheep,
                lc.sheep_kills,
                lc.wolf_births,
                lc.sheep_births,
                lc.wolf_starvations,
                lc.sheep_starvations,
                sim.sheep_kills_so_far(),
                sim.wolf_births_so_far(),
                sim.sheep_births_so_far(),
            );
        }

        // Termination conditions.
        if alive_wolves == 0 && alive_sheep == 0 {
            ended_at = Some(tick);
            end_reason = "both extinct";
            println!(
                "Tick {:>4}: BOTH EXTINCT (no wolves, no sheep)",
                tick,
            );
            break;
        }
        if alive_wolves == 0 {
            ended_at = Some(tick);
            end_reason = "wolves crashed (sheep took over)";
            println!(
                "Tick {:>4}: WOLVES CRASHED (sheep={} surviving)",
                tick, alive_sheep,
            );
            break;
        }
        if alive_sheep == 0 {
            ended_at = Some(tick);
            end_reason = "sheep crashed (wolves to follow)";
            println!(
                "Tick {:>4}: SHEEP CRASHED (wolves={} surviving — will starve)",
                tick, alive_wolves,
            );
            break;
        }
    }

    let final_wolves = sim.count_alive_wolves();
    let final_sheep = sim.count_alive_sheep();
    let total_kills = sim.sheep_kills_so_far();
    let total_wolf_births = sim.wolf_births_so_far();
    let total_sheep_births = sim.sheep_births_so_far();
    let total_wolf_starv = sim.wolf_starvations_so_far();
    let total_sheep_starv = sim.sheep_starvations_so_far();
    let final_tick = ended_at.unwrap_or(MAX_TICKS);

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  final tick:            {}", final_tick);
    println!("  end reason:            {}", end_reason);
    println!("  final wolves:          {} / wolf_cap {}", final_wolves, WOLF_CAP);
    println!("  final sheep:           {} / sheep_cap {}", final_sheep, SHEEP_CAP);
    println!("  peak wolves:           {}", peak_wolves);
    println!("  peak sheep:            {}", peak_sheep);
    println!("  total sheep kills:     {} (combat deaths via ApplyKill set_alive)", total_kills);
    println!("  total wolf births:     {} (CPU-side alive=false → true on Wolf slots)", total_wolf_births);
    println!("  total sheep births:    {} (CPU-side alive=false → true on Sheep slots)", total_sheep_births);
    println!("  total wolf starv:      {} (alive=true → false from EnergyDecay; wolf-CT)", total_wolf_starv);
    println!("  total sheep starv:     {} (alive=true → false from EnergyDecay; sheep-CT)", total_sheep_starv);

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    let any_kills = total_kills > 0;
    let any_wolf_births = total_wolf_births > 0;
    let any_sheep_births = total_sheep_births > 0;
    let any_births = any_wolf_births || any_sheep_births;
    let any_wolf_deaths = total_wolf_starv > 0;
    let any_sheep_deaths = total_kills > 0 || total_sheep_starv > 0;
    let any_deaths = any_wolf_deaths || any_sheep_deaths;

    if any_kills && any_wolf_births && any_sheep_births && any_wolf_deaths && any_sheep_deaths {
        println!(
            "  (a) FULL FIRE — composed COMBAT + LIFECYCLE on BOTH species. \
             {} kills drove {} wolf births; {} sheep births fired from \
             well-fed-pair check; {} wolves starved; {} sheep starved + {} \
             killed in combat. Both species' alive bitmaps flipped BOTH \
             directions during the run.",
            total_kills,
            total_wolf_births,
            total_sheep_births,
            total_wolf_starv,
            total_sheep_starv,
            total_kills,
        );
    } else if any_kills && any_births && any_deaths {
        println!(
            "  (a-partial) PARTIAL FIRE — composed combat + lifecycle, but \
             not all 5 surfaces lit. kills={} wolf_b={} sheep_b={} \
             wolf_starv={} sheep_starv={}. Investigate which surface \
             didn't fire.",
            total_kills, total_wolf_births, total_sheep_births,
            total_wolf_starv, total_sheep_starv,
        );
    } else if any_kills && !any_births {
        println!(
            "  (b-partial) COMBAT ONLY — {} kills fired but no births. \
             KILLS_PER_WOLF_BIRTH may be too high or process_lifecycle's \
             wolf-birth path isn't running.",
            total_kills,
        );
    } else if !any_kills {
        println!(
            "  (b) NO KILLS — WolfHunt emitted no Damaged events that \
             dropped a sheep. Spatial walk likely not finding sheep, OR \
             the ApplyKill set_alive gate isn't firing.",
        );
    } else {
        println!(
            "  (b) UNEXPECTED — neither combat nor lifecycle paths surfaced. \
             kills={} wb={} sb={} ws={} ss={}",
            total_kills, total_wolf_births, total_sheep_births,
            total_wolf_starv, total_sheep_starv,
        );
    }

    // Hard asserts on the killer-feature claims of this fixture.
    assert!(
        any_kills,
        "predator_prey_real_app: no kills fired — WolfHunt + ApplyKill combat \
         path is broken.",
    );
    assert!(
        any_wolf_births,
        "predator_prey_real_app: no wolf births — CPU-side alive=false → true \
         path on Wolf slots is broken.",
    );
    assert!(
        any_sheep_births,
        "predator_prey_real_app: no sheep births — CPU-side alive=false → true \
         path on Sheep slots is broken.",
    );
    assert!(
        any_sheep_deaths,
        "predator_prey_real_app: no sheep deaths — combat death path is broken.",
    );
}
