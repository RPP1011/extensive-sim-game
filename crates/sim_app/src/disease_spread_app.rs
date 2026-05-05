//! Disease spread harness — drives `disease_spread_runtime` for up to
//! 1000 ticks (or until the epidemic dies out, whichever comes first)
//! and reports the per-50-tick SIR trace + final outcome.
//!
//! ## Predicted observable shapes
//!
//! ### (a) FULL CURVE — classic SIR S→I→R/D
//!
//! 199 Susceptible + 1 Infected patient zero at tick 0. The single
//! patient zero takes ~50–150 ticks to find another agent in
//! infection range; once it does, the epidemic accelerates roughly
//! exponentially until the population of Susceptibles thins out and
//! growth slows. Peak infection should land in the 200–400 tick
//! range, then decline as agents recover or die. By tick ~700–1000
//! infected count drops to zero and the simulation terminates.
//!
//! ### (b) FAILURE TO IGNITE
//!
//! With movement step = 1.0 and infection radius = 4.0 in a 100×100
//! world, patient zero may wander forever without making contact
//! (200 agents in 10000 area = density 0.02). If this happens we
//! see Infected stuck at 1 for the entire run, no recoveries, no
//! deaths.

use disease_spread_runtime::{DiseaseSpreadState, INFECTION_DURATION, SirCounts, INFECTION_RADIUS};
use engine::CompiledSim;

const SEED: u64 = 0x5191_5191_C0FF_EE42;
const AGENT_COUNT: u32 = 200;
const MAX_TICKS: u64 = 1000;
const TRACE_INTERVAL: u64 = 50;

fn main() {
    let mut sim = DiseaseSpreadState::new(SEED, AGENT_COUNT);

    println!("================================================================");
    println!(" DISEASE SPREAD — SIR epidemic on {} agents", AGENT_COUNT);
    println!("   seed=0x{:016X} max_ticks={} trace_interval={}",
             SEED, MAX_TICKS, TRACE_INTERVAL);
    println!("   infection_radius={:.1} infection_duration={} ticks",
             INFECTION_RADIUS, INFECTION_DURATION);
    println!("================================================================");
    println!("   Legend: S=Susceptible, I=Infected, R=Recovered, D=Dead");
    println!();
    println!(" tick |    S    I    R    D | peak_I (tick) ");
    println!("------+----------------------+----------------");

    let initial = sim.sir_counts();
    print_trace(0, &initial, sim.peak_infected(), sim.peak_tick());

    let mut ended_at: Option<u64> = None;
    let mut last_traced_counts = initial;

    for tick in 1..=MAX_TICKS {
        sim.step();

        if tick % TRACE_INTERVAL == 0 {
            let c = sim.sir_counts();
            print_trace(tick, &c, sim.peak_infected(), sim.peak_tick());
            last_traced_counts = c;
        }

        // Termination — epidemic burned out.
        if sim.sir_counts().infected == 0 && tick >= 50 {
            // Print the terminal tick if we didn't just trace it.
            if tick % TRACE_INTERVAL != 0 {
                let c = sim.sir_counts();
                print_trace(tick, &c, sim.peak_infected(), sim.peak_tick());
                last_traced_counts = c;
            }
            ended_at = Some(tick);
            break;
        }
    }

    let final_counts = sim.sir_counts();
    let _ = last_traced_counts; // last_traced_counts kept for future diff dumps

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  Final tick:       {}", sim.tick());
    println!("  Susceptible:      {} (never infected)", final_counts.susceptible);
    println!("  Infected (alive): {} (still sick)", final_counts.infected);
    println!("  Recovered:        {} (immune survivors)", final_counts.recovered);
    println!("  Dead:             {} (mortality)", final_counts.dead);
    let total = final_counts.susceptible + final_counts.infected
        + final_counts.recovered + final_counts.dead;
    println!("  Total population: {} (conserved={})", total, total == AGENT_COUNT);
    println!("  Peak infected:    {} at tick {}",
             sim.peak_infected(), sim.peak_tick());
    let case_count = final_counts.recovered + final_counts.dead + final_counts.infected;
    if case_count > 0 {
        let case_fatality = (final_counts.dead as f32) / (case_count as f32);
        println!("  Total cases:      {} ({}% of population)",
                 case_count,
                 (case_count * 100) / AGENT_COUNT.max(1));
        println!("  Case fatality:    {:.1}%", case_fatality * 100.0);
    }
    if let Some(t) = ended_at {
        println!("  Epidemic over:    tick {} (Infected reached 0)", t);
    } else {
        println!("  Epidemic active:  ran to MAX_TICKS={} without burning out", MAX_TICKS);
    }

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    if sim.peak_infected() <= 1 {
        println!("  (b) FAILURE TO IGNITE — patient zero never spread the");
        println!("      disease. Density too low or contact rate too sparse.");
    } else if final_counts.infected > 0 {
        println!("  (a-partial) EPIDEMIC ONGOING — peak={} at tick {}, but",
                 sim.peak_infected(), sim.peak_tick());
        println!("      MAX_TICKS={} reached with {} still infected.",
                 MAX_TICKS, final_counts.infected);
    } else {
        println!("  (a) FULL SIR CURVE — epidemic ignited (peak={} at tick {}),",
                 sim.peak_infected(), sim.peak_tick());
        println!("      ran its course, then extinguished. {} recovered, {} dead.",
                 final_counts.recovered, final_counts.dead);
    }

    // Hard asserts (sim_app convention): the disease must spread beyond
    // patient zero, and by termination there must be more cases than
    // just the original infection.
    assert!(
        sim.peak_infected() > 1,
        "disease_spread_app: ASSERTION FAILED — epidemic never ignited \
         (peak infected = {}). Patient zero failed to make contact \
         within {} ticks. Try a larger seed sweep or shrink the world.",
        sim.peak_infected(), sim.tick(),
    );
    assert!(
        final_counts.recovered + final_counts.dead > 0,
        "disease_spread_app: ASSERTION FAILED — no recoveries or deaths. \
         The lifecycle pass did not transition any infected agent.",
    );
}

fn print_trace(tick: u64, c: &SirCounts, peak: u32, peak_tick: u64) {
    println!(
        " {:>4} | {:>4} {:>4} {:>4} {:>4} | {:>4} ({:>4})",
        tick, c.susceptible, c.infected, c.recovered, c.dead, peak, peak_tick,
    );
}
