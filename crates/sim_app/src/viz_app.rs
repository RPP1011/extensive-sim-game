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
    // boids retired here — the .sim now lives in the sims mega-crate
    // (`sims::boids::GeneratedRuntime`), but the mega-crate doesn't
    // yet expose `make_sim()` returning `Box<dyn CompiledSim>` or a
    // populated `positions()`. Re-add when those surfaces land.
    ("tom_probe",             tom_probe_runtime::make_sim,                 32,  "ToM probe"),
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
