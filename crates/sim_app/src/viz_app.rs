//! Universal ASCII visualizer — drives any per-fixture runtime that
//! implements [`engine::CompiledSim`] AND opts into the viz surface
//! by overriding `snapshot()`, `glyph_table()`, `default_viewport()`.
//!
//! ## Usage
//!
//! ```text
//! cargo run --bin viz_app --release -- <sim_name>
//! ```
//!
//! `<sim_name>` is one entry in the [`SIMS`] table below. Each entry
//! pairs a string label with a `make_sim` factory so a single binary
//! can serve every viz-enabled fixture.
//!
//! ## What it draws
//!
//! Per N ticks (N = `FRAME_INTERVAL`):
//!   1. Step the sim N times.
//!   2. Take a snapshot via `CompiledSim::snapshot()`.
//!   3. Project each agent's (x, z) onto a `WIDTH × HEIGHT` ASCII
//!      grid using the sim's `default_viewport()` bounding box.
//!   4. Render each cell with the sim's `glyph_table()[creature_type]`
//!      glyph + ANSI 256-colour foreground.
//!
//! The renderer is intentionally minimal — no double-buffering, no
//! sub-cell sampling, no z-ordering. Multiple agents in the same cell
//! show the LAST one visited (deterministic in agent order).

use engine::sim_trait::{CompiledSim, VizGlyph};
use glam::Vec3;
use std::collections::HashMap;

/// Render grid width (columns).
const WIDTH: usize = 80;
/// Render grid height (rows).
const HEIGHT: usize = 30;
/// Number of ticks between rendered frames. Smaller = smoother but
/// noisier; larger = chunkier but easier to read trajectory.
const FRAME_INTERVAL: u64 = 25;
/// Total frames to render before exit. Total ticks = FRAMES *
/// FRAME_INTERVAL.
const FRAMES: u32 = 40;

/// Registry of viz-enabled sims. Each entry binds a CLI label to a
/// boxed factory matching `engine::CompiledSim`. Adding a new sim is
/// a one-line change here once the runtime crate is depped in
/// `Cargo.toml`.
type Factory = fn() -> Box<dyn CompiledSim>;
const SIMS: &[(&str, Factory)] = &[
    ("crafting_diffusion", make_crafting_diffusion),
];

fn make_crafting_diffusion() -> Box<dyn CompiledSim> {
    crafting_diffusion_runtime::make_sim(0xC0FFEE_BEEF_CAFE_42, 50)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: viz_app <sim_name>");
        eprintln!("available sims:");
        for (name, _) in SIMS {
            eprintln!("  {name}");
        }
        std::process::exit(2);
    }
    let sim_name = &args[1];
    let factory = SIMS.iter().find(|(n, _)| n == sim_name).map(|(_, f)| *f);
    let make = match factory {
        Some(f) => f,
        None => {
            eprintln!("unknown sim '{sim_name}'. available:");
            for (n, _) in SIMS {
                eprintln!("  {n}");
            }
            std::process::exit(2);
        }
    };
    let mut sim = make();

    let glyphs = sim.glyph_table();
    let viewport = sim.default_viewport().unwrap_or((
        Vec3::new(-30.0, 0.0, -30.0),
        Vec3::new(30.0, 0.0, 30.0),
    ));

    println!("================================================================");
    println!(" viz_app — sim={sim_name}  agents={}  viewport=({:.1},{:.1})..({:.1},{:.1})",
        sim.agent_count(), viewport.0.x, viewport.0.z, viewport.1.x, viewport.1.z);
    println!(" frames={FRAMES}  ticks/frame={FRAME_INTERVAL}  total_ticks={}",
        FRAMES as u64 * FRAME_INTERVAL);
    println!("================================================================");

    for frame in 0..FRAMES {
        for _ in 0..FRAME_INTERVAL {
            sim.step();
        }
        let snap = sim.snapshot();
        render_frame(frame, sim.tick(), &snap.positions, &snap.creature_types,
                     &snap.alive, &glyphs, viewport);
    }
}

fn render_frame(
    frame: u32,
    tick: u64,
    positions: &[Vec3],
    creature_types: &[u32],
    alive: &[u32],
    glyphs: &[VizGlyph],
    viewport: (Vec3, Vec3),
) {
    // grid[y][x] → (glyph, color). None = empty.
    let mut grid: Vec<Vec<Option<(char, u8)>>> = vec![vec![None; WIDTH]; HEIGHT];

    let (lo, hi) = viewport;
    let dx = (hi.x - lo.x).max(1.0);
    let dz = (hi.z - lo.z).max(1.0);

    // Track per-cell creature_type count for a brief group-stats
    // breakdown printed under the frame.
    let mut group_counts: HashMap<u32, u32> = HashMap::new();

    for (i, &p) in positions.iter().enumerate() {
        if alive.get(i).copied().unwrap_or(0) == 0 {
            continue;
        }
        let ct = creature_types.get(i).copied().unwrap_or(0);
        *group_counts.entry(ct).or_insert(0) += 1;

        // Project (x, z) to grid coords.
        let nx = ((p.x - lo.x) / dx).clamp(0.0, 1.0);
        let nz = ((p.z - lo.z) / dz).clamp(0.0, 1.0);
        let gx = (nx * (WIDTH as f32 - 1.0)).round() as usize;
        let gy = (nz * (HEIGHT as f32 - 1.0)).round() as usize;

        let (glyph, color) = match glyphs.get(ct as usize) {
            Some(vg) => (vg.glyph, vg.fg_color),
            None => ('?', 7),
        };
        grid[gy][gx] = Some((glyph, color));
    }

    // Frame header.
    println!();
    println!("──── frame {:>3} / {} ── tick={} ────────────────────────────",
             frame + 1, FRAMES, tick);

    // Top border.
    print!("┌");
    for _ in 0..WIDTH { print!("─"); }
    println!("┐");

    // Body.
    for row in &grid {
        print!("│");
        for cell in row {
            match cell {
                Some((g, c)) => {
                    // ANSI 256-colour foreground escape: ESC[38;5;<n>m
                    print!("\x1b[38;5;{c}m{g}\x1b[0m");
                }
                None => print!(" "),
            }
        }
        println!("│");
    }

    // Bottom border.
    print!("└");
    for _ in 0..WIDTH { print!("─"); }
    println!("┘");
}
