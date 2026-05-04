//! Trophic-3-tier real harness — drives `trophic_3tier_runtime` and
//! reports per-100-tick population dynamics for a three-tier food web
//! (Grass → Herbivore → Carnivore) over 1000 ticks.
//!
//! This is the TWELFTH REAL fixture. Architectural escalation from
//! `predator_prey_real_app` (FOURTH real fixture):
//!   - predator_prey_real_app : 2 LIVE tiers (Wolves + Sheep), grazing
//!     proxied via uniform per-tick hunger gain.
//!   - trophic_3tier_app      : 3 LIVE tiers. Grass is a real entity
//!     with per-tile energy that herbivores actually deplete via
//!     ApplyEat chronicle. Full ecological cascade per tick.
//!
//! Termination rules (in order of precedence):
//!   - All upper tiers extinct: alive_herb=0 AND alive_carn=0 →
//!     "all crashed" (grass continues regrowing harmlessly).
//!   - Carnivores crashed: alive_carn=0 (herbivores persist) →
//!     "carnivores crashed (sheep took over)".
//!   - Herbivores crashed: alive_herb=0 (carnivores persist) →
//!     "herbivores crashed (wolves to starve)".
//!   - MAX_TICKS reached with all 3 tiers alive →
//!     "stable 3-tier ecosystem".

use engine::CompiledSim;
use trophic_3tier_runtime::{
    Trophic3TierState, CARN_CAP, GRASS_CAP, HERB_CAP, INITIAL_CARNIVORES, INITIAL_GRASS,
    INITIAL_HERBIVORES, SLOT_CAP,
};

const SEED: u64 = 0x7307_71C3_71E2_BEEF;
const MAX_TICKS: u64 = 1000;
const TRACE_EVERY: u64 = 100;

fn main() {
    let mut sim = Trophic3TierState::new(SEED, SLOT_CAP);
    println!("================================================================");
    println!(" TROPHIC 3-TIER — Three-tier food web (Grass → Herb → Carn)");
    println!(
        "   seed=0x{:016X} slot_cap={} (grass [0..{}) herb [{}..{}) carn [{}..{}))",
        SEED,
        SLOT_CAP,
        GRASS_CAP,
        GRASS_CAP,
        GRASS_CAP + HERB_CAP,
        GRASS_CAP + HERB_CAP,
        GRASS_CAP + HERB_CAP + CARN_CAP,
    );
    println!(
        "   initial grass={} herbivores={} carnivores={} max_ticks={} trace_every={}",
        INITIAL_GRASS, INITIAL_HERBIVORES, INITIAL_CARNIVORES, MAX_TICKS, TRACE_EVERY,
    );
    println!("================================================================");

    // Per-100-tick window deltas — accumulate per-tick lifecycle into
    // window-rolling counters that get reset each trace boundary.
    let mut window_grass_births = 0u32;
    let mut window_herb_births = 0u32;
    let mut window_carn_births = 0u32;
    let mut window_grass_deaths = 0u32;
    let mut window_herb_starv = 0u32;
    let mut window_carn_starv = 0u32;
    let mut window_herb_kills = 0u32;

    let initial_grass = sim.count_alive_grass();
    let initial_herb = sim.count_alive_herbivores();
    let initial_carn = sim.count_alive_carnivores();
    let (initial_eg, initial_eh, initial_ec) = sim.tier_energy_totals();
    println!(
        "Tick    0: grass={:>3} herb={:>3} carn={:>3} | energy g={:>5.0} h={:>5.0} c={:>5.0} (initial)",
        initial_grass, initial_herb, initial_carn, initial_eg, initial_eh, initial_ec,
    );

    let mut peak_grass = initial_grass;
    let mut peak_herb = initial_herb;
    let mut peak_carn = initial_carn;
    let mut ended_at: Option<u64> = None;
    let mut end_reason = "ran to MAX_TICKS";

    for tick in 1..=MAX_TICKS {
        sim.step();
        let lc = sim.last_lifecycle();
        window_grass_births += lc.grass_births;
        window_herb_births += lc.herb_births;
        window_carn_births += lc.carn_births;
        window_grass_deaths += lc.grass_deaths;
        window_herb_starv += lc.herb_starvations;
        window_carn_starv += lc.carn_starvations;
        window_herb_kills += lc.herb_kills;

        let alive_grass = sim.count_alive_grass();
        let alive_herb = sim.count_alive_herbivores();
        let alive_carn = sim.count_alive_carnivores();
        if alive_grass > peak_grass {
            peak_grass = alive_grass;
        }
        if alive_herb > peak_herb {
            peak_herb = alive_herb;
        }
        if alive_carn > peak_carn {
            peak_carn = alive_carn;
        }

        if tick % TRACE_EVERY == 0 {
            let (eg, eh, ec) = sim.tier_energy_totals();
            println!(
                "Tick {:>4}: grass={:>3} herb={:>3} carn={:>3} | energy g={:>5.0} h={:>5.0} c={:>5.0}",
                tick, alive_grass, alive_herb, alive_carn, eg, eh, ec,
            );
            println!(
                "           [last 100t]  +grass_births={:>3} +herb_births={:>3} +carn_births={:>3} \
                 | +grass_deaths={:>3} +herb_kills={:>3} +herb_starv={:>3} +carn_starv={:>3}",
                window_grass_births, window_herb_births, window_carn_births,
                window_grass_deaths, window_herb_kills,
                window_herb_starv, window_carn_starv,
            );
            window_grass_births = 0;
            window_herb_births = 0;
            window_carn_births = 0;
            window_grass_deaths = 0;
            window_herb_starv = 0;
            window_carn_starv = 0;
            window_herb_kills = 0;
        }

        // Termination conditions.
        if alive_herb == 0 && alive_carn == 0 {
            ended_at = Some(tick);
            end_reason = "all crashed (both upper tiers extinct, grass continues)";
            println!(
                "Tick {:>4}: ALL CRASHED — herb=0 carn=0 (grass={} surviving)",
                tick, alive_grass,
            );
            break;
        }
        if alive_carn == 0 {
            ended_at = Some(tick);
            end_reason = "carnivores crashed (herbivores took over)";
            println!(
                "Tick {:>4}: CARNIVORES CRASHED (herb={} surviving)",
                tick, alive_herb,
            );
            break;
        }
        if alive_herb == 0 {
            ended_at = Some(tick);
            end_reason = "herbivores crashed (carnivores to starve)";
            println!(
                "Tick {:>4}: HERBIVORES CRASHED (carn={} surviving — will starve)",
                tick, alive_carn,
            );
            break;
        }
    }

    let final_grass = sim.count_alive_grass();
    let final_herb = sim.count_alive_herbivores();
    let final_carn = sim.count_alive_carnivores();
    let total_kills = sim.herb_kills_so_far();
    let total_carn_births = sim.carn_births_so_far();
    let total_herb_births = sim.herb_births_so_far();
    let total_grass_births = sim.grass_births_so_far();
    let total_herb_starv = sim.herb_starvations_so_far();
    let total_carn_starv = sim.carn_starvations_so_far();
    let total_grass_deaths = sim.grass_deaths_so_far();
    let final_tick = ended_at.unwrap_or(MAX_TICKS);
    // Refine end-reason for the MAX_TICKS path: distinguish "stable
    // 3-tier ecosystem" (all 3 tiers >0 at termination) from any
    // late-tick edge case.
    let end_reason = if ended_at.is_none() && final_grass > 0 && final_herb > 0 && final_carn > 0 {
        "stable 3-tier ecosystem (ran to MAX_TICKS, all 3 tiers persist)"
    } else {
        end_reason
    };

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  final tick:               {}", final_tick);
    println!("  end reason:               {}", end_reason);
    println!("  final grass:              {} / cap {}", final_grass, GRASS_CAP);
    println!("  final herbivores:         {} / cap {}", final_herb, HERB_CAP);
    println!("  final carnivores:         {} / cap {}", final_carn, CARN_CAP);
    println!("  peak grass:               {}", peak_grass);
    println!("  peak herbivores:          {}", peak_herb);
    println!("  peak carnivores:          {}", peak_carn);
    println!("  total herbivore kills:    {} (combat deaths via ApplyStrike)", total_kills);
    println!("  total grass deaths:       {} (depleted via ApplyEat)", total_grass_deaths);
    println!("  total carnivore births:   {} (every {} kills)", total_carn_births, 6);
    println!("  total herbivore births:   {} (well-fed-pair check)", total_herb_births);
    println!("  total grass respawns:     {} (every 5 ticks, up to 4)", total_grass_births);
    println!("  total herbivore starv:    {}", total_herb_starv);
    println!("  total carnivore starv:    {}", total_carn_starv);

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    let any_kills = total_kills > 0;
    let any_grass_eaten = total_grass_deaths > 0;
    let any_carn_births = total_carn_births > 0;
    let any_herb_births = total_herb_births > 0;
    let any_grass_births = total_grass_births > 0;
    let any_starv = total_herb_starv > 0 || total_carn_starv > 0;

    if any_kills && any_grass_eaten && any_carn_births && any_herb_births
        && any_grass_births && any_starv
    {
        println!(
            "  (a) FULL FIRE — three-tier food web composed end-to-end. \
             Grass→Herbivore: {} grass tiles depleted by herbivore Eat events, \
             {} grass tiles respawned. Herbivore→Carnivore: {} kills drove \
             {} carnivore births. Lifecycle: {} herb starv + {} carn starv. \
             {} herbivore births fired from well-fed-pair check. \
             ALL THREE alive bitmaps flipped BOTH directions during the run.",
            total_grass_deaths, total_grass_births, total_kills, total_carn_births,
            total_herb_starv, total_carn_starv, total_herb_births,
        );
    } else if any_kills && any_grass_eaten {
        println!(
            "  (a-partial) PARTIAL FIRE — feeding cascade lit but not all \
             lifecycle surfaces. grass_eaten={} kills={} carn_b={} herb_b={} \
             grass_b={} starv={}+{}.",
            total_grass_deaths, total_kills, total_carn_births, total_herb_births,
            total_grass_births, total_herb_starv, total_carn_starv,
        );
    } else if any_grass_eaten && !any_kills {
        println!(
            "  (b-partial) HERBIVORES EAT BUT CARNIVORES DON'T HUNT — \
             {} grass deaths, 0 herbivore kills. CarnivoreHunt or \
             ApplyStrike chronicle is broken.",
            total_grass_deaths,
        );
    } else if !any_grass_eaten {
        println!(
            "  (b) NO GRASS EATEN — HerbivoreEat→ApplyEat chain didn't deplete \
             any grass tile. Spatial walk likely not finding grass.",
        );
    } else {
        println!(
            "  (b) UNEXPECTED — grass_eaten={} kills={} carn_b={} herb_b={} grass_b={}",
            total_grass_deaths, total_kills, total_carn_births, total_herb_births, total_grass_births,
        );
    }

    // Hard asserts on the killer-feature claims of this fixture.
    assert!(
        any_grass_eaten,
        "trophic_3tier_app: no grass tiles eaten — HerbivoreEat + ApplyEat \
         chain is broken.",
    );
    assert!(
        any_kills,
        "trophic_3tier_app: no herbivore kills — CarnivoreHunt + ApplyStrike \
         combat path is broken.",
    );
    assert!(
        any_grass_births,
        "trophic_3tier_app: no grass respawns — CPU-side grass alive=false → \
         true path is broken.",
    );
    assert!(
        any_herb_births,
        "trophic_3tier_app: no herbivore births — CPU-side alive=false → true \
         on Herbivore slots is broken.",
    );
}
