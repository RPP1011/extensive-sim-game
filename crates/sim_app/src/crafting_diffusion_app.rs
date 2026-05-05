//! Crafting diffusion harness — drives `crafting_diffusion_runtime`
//! to convergence (or 1000-tick timeout) and reports per-50-tick
//! convergence stats.
//!
//! ## What this prints
//!
//! Per 50 ticks:
//!   - tick number
//!   - count of agents at FULL knowledge (mask = 1023)
//!   - mean bits-set across population
//!   - histogram of agents per knowledge level (0..10 bits)
//!   - cumulative trade events fired
//!
//! On termination:
//!   - whether full convergence was reached, and at which tick
//!   - reason for stalling (if not converged)
//!
//! ## Predicted shape
//!
//! Initial: each agent has 2 bits (group's pair). Trades between
//! cross-group neighbours OR bits both ways. The diffusion graph
//! has 5 group anchors evenly spaced on a circle of radius 24 with
//! per-agent jitter ±6, so adjacent groups overlap heavily and
//! cross-group trades fire from tick 1.
//!
//! Convergence speed depends on:
//!   - trade_radius (larger → faster)
//!   - cooldown (smaller → faster)
//!   - wander speed (faster → more mixing → faster)
//!
//! With the chosen defaults (radius=6, cooldown=5, wander=0.4) we
//! expect ~70% of agents at full knowledge by tick 200, full
//! convergence between tick 300-700.

use crafting_diffusion_runtime::{
    CraftingDiffusionState, FULL_MASK, NUM_GROUPS, NUM_RECIPES,
};
use engine::CompiledSim;

const SEED: u64 = 0xC0FFEE_BEEF_CAFE_42;
const AGENT_COUNT: u32 = 50;
const MAX_TICKS: u64 = 1000;
const TRACE_INTERVAL: u64 = 50;

fn print_trace(sim: &CraftingDiffusionState) {
    let tick = sim.tick();
    let full = sim.full_knowledge_count();
    let pct = (full as f32 / AGENT_COUNT as f32) * 100.0;
    let mean = sim.mean_bits();
    let hist = sim.knowledge_histogram();
    let trades = sim.total_trades();
    println!(
        "tick={:>4}  full={:>2}/{} ({:>5.1}%)  mean_bits={:>4.2}  trades_total={:>5}",
        tick, full, AGENT_COUNT, pct, mean, trades,
    );
    // Histogram on the next line, indented.
    let mut hist_str = String::from("           hist[bits→#agents]: ");
    for (b, &c) in hist.iter().enumerate() {
        if c > 0 {
            hist_str.push_str(&format!("{}={} ", b, c));
        }
    }
    println!("{hist_str}");
}

fn main() {
    let mut sim = CraftingDiffusionState::new(SEED, AGENT_COUNT);
    println!("================================================================");
    println!(" CRAFTING DIFFUSION — 21st REAL SIM (information sharing)");
    println!("   seed=0x{:016X}  agents={}  groups={}  recipes={}  full_mask=0b{:010b}",
        SEED, AGENT_COUNT, NUM_GROUPS, NUM_RECIPES, FULL_MASK);
    println!("   max_ticks={}  trace_interval={}", MAX_TICKS, TRACE_INTERVAL);
    println!("================================================================");

    // Sanity: print initial group/knowledge breakdown.
    {
        let mut group_init = vec![0u32; NUM_GROUPS as usize];
        for slot in 0..AGENT_COUNT as usize {
            group_init[sim.group(slot) as usize] += 1;
        }
        println!("Initial group sizes: {group_init:?}");
        for g in 0..NUM_GROUPS {
            // Find the first agent in this group to inspect its initial mask.
            for slot in 0..AGENT_COUNT as usize {
                if sim.group(slot) == g {
                    println!(
                        "   group {} initial mask: 0b{:010b} ({} bits)",
                        g, sim.knowledge(slot), sim.knowledge(slot).count_ones(),
                    );
                    break;
                }
            }
        }
    }

    println!();
    print_trace(&sim);

    let mut converged_at: Option<u64> = None;
    for tick in 1..=MAX_TICKS {
        sim.step();
        if tick % TRACE_INTERVAL == 0 {
            print_trace(&sim);
        }
        if sim.fully_converged() {
            converged_at = Some(tick);
            // Trace at the convergence tick if we didn't just print.
            if tick % TRACE_INTERVAL != 0 {
                print_trace(&sim);
            }
            break;
        }
    }

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    let final_tick = sim.tick();
    let final_full = sim.full_knowledge_count();
    let final_pct = (final_full as f32 / AGENT_COUNT as f32) * 100.0;
    let final_mean = sim.mean_bits();
    let final_trades = sim.total_trades();
    println!(
        "  Final tick:      {}\n  Full-knowledge:  {}/{} ({:.1}%)\n  Mean bits-set:   {:.2}\n  Total trades:    {}",
        final_tick, final_full, AGENT_COUNT, final_pct, final_mean, final_trades,
    );
    println!("  Final histogram: {:?}", sim.knowledge_histogram());

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    if let Some(t) = converged_at {
        println!(
            "  (a) FULL CONVERGENCE — every Trader knows every recipe by tick {t}.\n      Information propagated through {final_trades} pairwise knowledge merges.\n      The associative+commutative `mana := mana | other.mana` operator\n      reaches its asymptote (mask=0b{:010b}=1023) deterministically.",
            FULL_MASK,
        );
    } else if final_pct >= 80.0 {
        println!(
            "  (a-partial) NEAR CONVERGENCE — {:.1}% of Traders fully informed by\n      tick {} ({} short of full set). Probably a small number of agents\n      are stuck in sparsely-mixed regions; running longer would close them.",
            final_pct, final_tick, AGENT_COUNT - final_full,
        );
    } else if final_mean > 4.0 {
        println!(
            "  (a-grow) DIFFUSION ACTIVE — mean bits grew from 2.0 to {:.2}\n      over {} ticks but full convergence stalled. Likely cause: groups\n      cluster too tightly, so cross-group bridges are rare. Tunable:\n      raise wander_speed or trade_radius.",
            final_mean, final_tick,
        );
    } else {
        println!(
            "  (b) STALL — mean bits stayed near 2.0 over {} ticks. The trade\n      gate (`(target.knowledge & ~self.knowledge) != 0`) is firing rarely.\n      Likely cause: groups never physically meet (spawn anchors too far\n      apart, or wander too slow to bridge them).",
            final_tick,
        );
    }

    // Hard assertions for the harness — fail loudly if information
    // does not actually propagate.
    assert!(
        final_mean > 2.0,
        "crafting_diffusion_app: ASSERTION FAILED — mean bits-set is {:.2}; \
         no information propagated. Initial mean was 2.0; trade pipeline did \
         not fire.",
        final_mean,
    );
    assert!(
        final_trades > 0,
        "crafting_diffusion_app: ASSERTION FAILED — total_trades is 0. \
         The per-tick trade scan found no eligible pairs.",
    );
}
