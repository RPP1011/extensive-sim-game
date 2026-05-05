//! Generic ASCII renderer for `engine::CompiledSim` — Qud / CogMind
//! style. Any sim that implements the trait's viz methods (`snapshot`,
//! `glyph_table`, `default_viewport`) renders without per-sim glue.
//!
//! Intentionally tiny: no input, no double-buffering, no curses. Just
//! enough to *watch* a sim play out instead of reading scrollback.

use engine::{CompiledSim, VizGlyph};
use glam::Vec3;

/// Default glyph table when a sim doesn't expose one. Indexed by
/// creature_type discriminant. Picks single letters + ANSI 256 colors
/// that read well on dark terminals. Wraps after 26 types.
#[allow(dead_code)]
fn default_glyphs(max_ct: u32) -> Vec<VizGlyph> {
    const PALETTE: [u8; 8] = [196, 47, 33, 226, 201, 51, 208, 129];
    let n = (max_ct as usize + 1).max(1);
    (0..n)
        .map(|i| {
            let ch = (b'a' + (i % 26) as u8) as char;
            let color = PALETTE[i % PALETTE.len()];
            VizGlyph::new(ch, color)
        })
        .collect()
}

/// Compute a viewport from observed agent positions. Used when the sim
/// doesn't expose `default_viewport()`. Pads by 10% of the spread on
/// each axis so movement stays on-screen, with a minimum span of 2.0
/// to avoid div-by-zero when all agents share a coordinate.
#[allow(dead_code)]
fn auto_viewport(positions: &[Vec3], width: u32, height: u32) -> Viewport {
    let alive_pos: Vec<Vec3> = positions
        .iter()
        .copied()
        .filter(|p| p.is_finite())
        .collect();
    if alive_pos.is_empty() {
        return Viewport {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(1.0, 1.0, 0.0),
            width,
            height,
        };
    }
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in &alive_pos {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let span_x = (max_x - min_x).max(2.0);
    let span_y = (max_y - min_y).max(2.0);
    let pad_x = span_x * 0.10;
    let pad_y = span_y * 0.10;
    Viewport {
        min: Vec3::new(min_x - pad_x, min_y - pad_y, 0.0),
        max: Vec3::new(max_x + pad_x, max_y + pad_y, 0.0),
        width,
        height,
    }
}

/// Viewport bounds in world units. World (x, y) inside [min, max] maps
/// onto the terminal grid; positions outside are clipped. (Z is ignored
/// because sim runtimes use Y as the second 2D axis.)
#[derive(Clone, Copy)]
pub struct Viewport {
    pub min: Vec3,
    pub max: Vec3,
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    /// Build a viewport with the sim's default bounds + the requested
    /// terminal dimensions. Falls back to a unit square if the sim
    /// doesn't expose a default. Prefer [`Viewport::for_sim_auto`] for
    /// the generic viewer — it auto-fits from observed positions when
    /// the sim doesn't opt in.
    #[allow(dead_code)]
    pub fn for_sim(sim: &dyn CompiledSim, width: u32, height: u32) -> Self {
        let (min, max) = sim
            .default_viewport()
            .unwrap_or((Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)));
        Self {
            min,
            max,
            width,
            height,
        }
    }

    /// Build a viewport from the sim's `default_viewport()` if exposed,
    /// otherwise auto-fit from observed agent positions. Requires a
    /// snapshot — the renderer's primary entry point already calls
    /// snapshot, so use this when you have one in hand.
    #[allow(dead_code)]
    pub fn for_sim_auto(
        sim: &dyn CompiledSim,
        positions: &[Vec3],
        width: u32,
        height: u32,
    ) -> Self {
        match sim.default_viewport() {
            Some((min, max)) => Self {
                min,
                max,
                width,
                height,
            },
            None => auto_viewport(positions, width, height),
        }
    }

    fn project(&self, p: Vec3) -> Option<(u32, u32)> {
        let dx = self.max.x - self.min.x;
        let dy = self.max.y - self.min.y;
        if dx <= 0.0 || dy <= 0.0 {
            return None;
        }
        let nx = (p.x - self.min.x) / dx;
        let ny = (p.y - self.min.y) / dy;
        if !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
            return None;
        }
        let col = (nx * (self.width as f32 - 1.0)).round() as u32;
        // Flip y so increasing world-y goes UP the screen (north).
        let row = ((1.0 - ny) * (self.height as f32 - 1.0)).round() as u32;
        Some((col, row))
    }
}

/// One-call OOTB renderer for ANY sim. Auto-fills:
///   - viewport: sim's `default_viewport()` if exposed, else fit-to-positions
///   - glyphs: sim's `glyph_table()` if exposed, else alphabet+palette
///   - status: `tick`, total agents alive, per-creature_type alive counts
///
/// Adds the caller's optional `extra_status` lines below the auto status.
/// Returns an empty string if the sim's `snapshot()` is empty (sim doesn't
/// expose viz — caller knows nothing's renderable).
#[allow(dead_code)]
pub fn render_sim_auto(
    sim: &mut dyn CompiledSim,
    title: &str,
    width: u32,
    height: u32,
    extra_status: &[String],
) -> String {
    let mut snap = sim.snapshot();
    if snap.positions.is_empty() {
        // Sim hasn't opted into snapshot — synthesize from positions()
        // (which most runtimes implement). Treat every agent as alive,
        // creature_type=0. This is enough to render single-creature-type
        // sims like boids out of the box.
        let positions: Vec<Vec3> = sim.positions().to_vec();
        if positions.is_empty() {
            return String::new();
        }
        let n = positions.len();
        snap = engine::AgentSnapshot {
            positions,
            creature_types: vec![0u32; n],
            alive: vec![1u32; n],
        };
    }
    let viewport = Viewport::for_sim_auto(sim, &snap.positions, width, height);
    let mut glyphs = sim.glyph_table();
    if glyphs.is_empty() {
        let max_ct = snap.creature_types.iter().copied().max().unwrap_or(0);
        glyphs = default_glyphs(max_ct);
    }
    let alive_total: u32 = snap.alive.iter().filter(|a| **a > 0).count() as u32;
    // Per-creature_type alive counts, ordered by creature_type id.
    let max_ct = snap.creature_types.iter().copied().max().unwrap_or(0);
    let mut per_ct = vec![0u32; (max_ct as usize) + 1];
    for i in 0..snap.alive.len().min(snap.creature_types.len()) {
        if snap.alive[i] != 0 {
            let ct = snap.creature_types[i] as usize;
            if ct < per_ct.len() {
                per_ct[ct] += 1;
            }
        }
    }
    let mut auto_status = format!(
        " tick {:5}   alive {:5}/{:5}   ",
        sim.tick(),
        alive_total,
        snap.alive.len(),
    );
    for (ct, count) in per_ct.iter().enumerate() {
        let g = glyphs.get(ct);
        let (ch, color) = g.map(|g| (g.glyph, g.fg_color)).unwrap_or(('?', 248));
        auto_status.push_str(&format!(
            "\x1b[38;5;{}m{}\x1b[0m={:<4} ",
            color, ch, count,
        ));
    }
    let mut status_lines: Vec<String> = vec![auto_status];
    status_lines.extend(extra_status.iter().cloned());
    render_with_grid(&snap, &glyphs, viewport, title, &status_lines)
}

/// Render one frame of a sim to a String. Pulls positions, creature
/// types, alive bits, and the glyph table directly from the sim.
///
/// `title` is printed above the playfield; `status_lines` below.
/// Multiple agents in the same cell pick the highest-creature_type
/// glyph (predators dominate prey visually).
#[allow(dead_code)]
pub fn render_sim_frame(
    sim: &mut dyn CompiledSim,
    viewport: Viewport,
    title: &str,
    status_lines: &[String],
) -> String {
    let snap = sim.snapshot();
    let glyphs = sim.glyph_table();
    render_with_grid(&snap, &glyphs, viewport, title, status_lines)
}

fn render_with_grid(
    snap: &engine::AgentSnapshot,
    glyphs: &[VizGlyph],
    viewport: Viewport,
    title: &str,
    status_lines: &[String],
) -> String {
    let w = viewport.width as usize;
    let h = viewport.height as usize;
    let mut grid: Vec<Option<(char, u8, u32)>> = vec![None; w * h];

    let n = snap
        .positions
        .len()
        .min(snap.creature_types.len())
        .min(snap.alive.len());
    for i in 0..n {
        if snap.alive[i] == 0 {
            continue;
        }
        let ct_u = snap.creature_types[i];
        let Some(spec) = glyphs.get(ct_u as usize) else {
            continue;
        };
        let Some((col, row)) = viewport.project(snap.positions[i]) else {
            continue;
        };
        let idx = (row as usize) * w + (col as usize);
        match grid[idx] {
            None => grid[idx] = Some((spec.glyph, spec.fg_color, ct_u)),
            Some((_, _, prev_ct)) if ct_u > prev_ct => {
                grid[idx] = Some((spec.glyph, spec.fg_color, ct_u));
            }
            _ => {}
        }
    }

    let mut out = String::new();
    // Clear screen + cursor home so successive frames overdraw.
    out.push_str("\x1b[2J\x1b[H");
    out.push_str(title);
    out.push('\n');
    out.push('+');
    for _ in 0..w {
        out.push('-');
    }
    out.push_str("+\n");
    for row in 0..h {
        out.push('|');
        for col in 0..w {
            match grid[row * w + col] {
                None => out.push(' '),
                Some((ch, fg, _)) => {
                    out.push_str(&format!("\x1b[38;5;{}m{}\x1b[0m", fg, ch));
                }
            }
        }
        out.push_str("|\n");
    }
    out.push('+');
    for _ in 0..w {
        out.push('-');
    }
    out.push_str("+\n");
    for line in status_lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}
