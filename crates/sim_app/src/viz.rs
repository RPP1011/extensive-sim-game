//! Generic ASCII renderer for `engine::CompiledSim` — Qud / CogMind
//! style. Any sim that implements the trait's viz methods (`snapshot`,
//! `glyph_table`, `default_viewport`) renders without per-sim glue.
//!
//! Intentionally tiny: no input, no double-buffering, no curses. Just
//! enough to *watch* a sim play out instead of reading scrollback.

use engine::CompiledSim;
use glam::Vec3;

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
    /// doesn't expose a default.
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

/// Render one frame of a sim to a String. Pulls positions, creature
/// types, alive bits, and the glyph table directly from the sim.
///
/// `title` is printed above the playfield; `status_lines` below.
/// Multiple agents in the same cell pick the highest-creature_type
/// glyph (predators dominate prey visually).
pub fn render_sim_frame(
    sim: &mut dyn CompiledSim,
    viewport: Viewport,
    title: &str,
    status_lines: &[String],
) -> String {
    let snap = sim.snapshot();
    let glyphs = sim.glyph_table();
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
