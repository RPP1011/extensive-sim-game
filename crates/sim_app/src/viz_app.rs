//! Universal sim visualizer — picks any sim by name, renders it.
//!
//! Usage:
//!   viz_app                        # lists available sims
//!   viz_app <name>                 # runs <name> with default seed
//!   viz_app <name> <seed>          # runs <name> with explicit seed
//!   viz_app <name> <seed> <count>  # runs <name> with explicit agent count
//!
//! Every sim that implements `CompiledSim` is reachable here. Sims that
//! haven't opted into the viz trait methods (`snapshot`, `glyph_table`,
//! `default_viewport`) get auto-fallbacks: alphabet glyphs by
//! creature_type id, viewport auto-fit from observed agent positions.

mod viz;

use engine::CompiledSim;
use std::io::Write;
use std::thread::sleep;
use std::time::Duration;
use viz::render_sim_auto;

const FRAME_MS: u64 = 80;
const VIEW_W: u32 = 80;
const VIEW_H: u32 = 24;
const MAX_TICKS: u64 = 2000;

/// Registry of every sim in the workspace. Each entry maps a CLI name
/// to a factory + the runtime's preferred default agent count. The
/// factory takes (seed, agent_count) and returns a boxed CompiledSim.
type Factory = fn(u64, u32) -> Box<dyn CompiledSim>;

const SIMS: &[(&str, Factory, u32, &str)] = &[
    // (name, factory, default_count, one-line description)
    ("boids",                 sim_runtime::make_sim,                       512, "boids flocking"),
    ("predator_prey",         predator_prey_runtime::make_sim,             64,  "predator/prey 2-species"),
    ("particle_collision",    particle_collision_runtime::make_sim,        256, "elastic collisions"),
    ("crowd_navigation",      crowd_navigation_runtime::make_sim,          256, "crowd navigation"),
    ("target_chaser",         target_chaser_runtime::make_sim,             32,  "target-chaser stress"),
    ("swarm_storm",           swarm_storm_runtime::make_sim,               64,  "multi-emit swarm"),
    ("bartering",             bartering_runtime::make_sim,                 32,  "bartering probe"),
    ("ecosystem",             ecosystem_runtime::make_sim,                 96,  "3-tier ecosystem cascade"),
    ("foraging",              foraging_runtime::make_sim,                  32,  "ant foraging"),
    ("auction",               auction_runtime::make_sim,                   32,  "auction probe"),
    ("verb_probe",            verb_probe_runtime::make_sim,                16,  "verb cascade probe"),
    ("tom_probe",             tom_probe_runtime::make_sim,                 32,  "ToM probe"),
    ("trade_market",          trade_market_runtime::make_sim,              32,  "trade market probe"),
    ("abilities",             abilities_runtime::make_sim,                 16,  "abilities probe"),
    ("cooldown_probe",        cooldown_probe_runtime::make_sim,            16,  "cooldown probe"),
    ("diplomacy_probe",       diplomacy_probe_runtime::make_sim,           16,  "diplomacy probe"),
    ("stochastic_probe",      stochastic_probe_runtime::make_sim,          64,  "RNG probe"),
    ("pair_scoring_probe",    pair_scoring_probe_runtime::make_sim,        16,  "pair-scoring probe"),
    ("stdlib_math_probe",     stdlib_math_probe_runtime::make_sim,         32,  "stdlib math probe"),
    ("quest_probe",           quest_probe_runtime::make_sim,               16,  "quest probe"),
    ("duel_1v1",              duel_1v1_runtime::make_sim,                  2,   "1v1 combat duel"),
    ("duel_25v25",            duel_25v25_runtime::make_sim,                50,  "25v25 squad combat"),
    ("foraging_real",         foraging_real_runtime::make_sim,             50,  "REAL ant colony lifecycle"),
    ("predator_prey_real",    predator_prey_real_runtime::make_sim,        300, "REAL wolves vs sheep ecology"),
    ("tactical_squad_5v5",    tactical_squad_5v5_runtime::make_sim,        10,  "5v5 with Tank/Healer/DPS roles"),
    ("mass_battle_100v100",   mass_battle_100v100_runtime::make_sim,       200, "200-agent multi-verb cascade"),
    ("trade_market_real",     trade_market_real_runtime::make_sim,         60,  "REAL economy: 50 traders + 10 goods"),
    ("quest_arc_real",        quest_arc_real_runtime::make_sim,            30,  "5-stage quest state machine"),
    ("village_day_cycle",     village_day_cycle_runtime::make_sim,         30,  "villagers w/ time-of-day routines"),
    ("megaswarm_1000",        megaswarm_1000_runtime::make_sim,            1000,"1000-agent pair-field stress"),
    ("tactical_horde_500",    tactical_horde_500_runtime::make_sim,        1000,"500v500 multi-verb cascade"),
    ("trophic_3tier",         trophic_3tier_runtime::make_sim,             400, "3-tier food web (grass/sheep/wolf)"),
    ("flocking_skirmish",     flocking_skirmish_runtime::make_sim,         200, "team-aware boids combat"),
    ("tower_defense",         tower_defense_runtime::make_sim,             111, "wave-based TD"),
    ("boss_fight",            boss_fight_runtime::make_sim,                6,   "1 boss + 5 heroes"),
    ("megaswarm_10000",       megaswarm_10000_runtime::make_sim,           10000,"10k-agent GPU saturation"),
    ("objective_capture_10v10", objective_capture_10v10_runtime::make_sim, 21,  "CTF hold-time race"),
    ("scripted_battle",       scripted_battle_runtime::make_sim,           75,  "3-phase scripted narrative arc"),
    ("multi_zone_world",      multi_zone_world_runtime::make_sim,          35,  "Forest/Town/Dungeon zone migration"),
];

fn print_help() {
    eprintln!("usage: viz_app <sim_name> [seed] [agent_count]");
    eprintln!();
    eprintln!("Available sims ({} total):", SIMS.len());
    let mut max_name = 0;
    for (n, _, _, _) in SIMS {
        max_name = max_name.max(n.len());
    }
    for (name, _, default_count, desc) in SIMS {
        eprintln!(
            "  {:width$}  (n={:5})  {}",
            name,
            default_count,
            desc,
            width = max_name,
        );
    }
    eprintln!();
    eprintln!("Glyphs/colors come from the sim's `glyph_table()` (or fall back to");
    eprintln!("alphabet by creature_type). Viewport from `default_viewport()` (or");
    eprintln!("auto-fits from agent positions).");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_help();
        std::process::exit(1);
    }
    let name = args[1].as_str();
    let seed: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xC0FFEE_DEC1DE_42u64);

    let entry = SIMS.iter().find(|(n, _, _, _)| *n == name);
    let Some((_, factory, default_count, desc)) = entry else {
        eprintln!("unknown sim: {}", name);
        eprintln!();
        print_help();
        std::process::exit(2);
    };
    let agent_count: u32 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(*default_count);

    eprintln!(
        "Starting '{}' — {} (seed=0x{:016X}, n={})",
        name, desc, seed, agent_count,
    );
    let mut sim = factory(seed, agent_count);

    for tick in 0..=MAX_TICKS {
        if tick > 0 {
            sim.step();
        }
        let title = format!("\x1b[1m{}\x1b[0m  —  {}", name, desc);
        let extra = vec![
            format!(" seed: 0x{:016X}   n: {}", seed, agent_count),
            String::from(" Ctrl-C to quit"),
        ];
        let frame = render_sim_auto(&mut *sim, &title, VIEW_W, VIEW_H, &extra);
        if frame.is_empty() {
            // Sim doesn't expose snapshot — bail with hint.
            eprintln!(
                "\nsim '{}' returned empty snapshot — runtime hasn't \
                 implemented CompiledSim::snapshot() yet.",
                name,
            );
            std::process::exit(3);
        }
        print!("{}", frame);
        std::io::stdout().flush().ok();
        sleep(Duration::from_millis(FRAME_MS));
    }

    println!("\nReached MAX_TICKS={}. Done.", MAX_TICKS);
}
