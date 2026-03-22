//! Procedural ASCII art templates for common asset types.
//!
//! Used as fallback when no model backend is available.

use super::grid::{AsciiCell, AsciiGrid};
use super::palette;

/// LCG for deterministic template generation (same as aot_pipeline).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        let s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut lcg = Self(s);
        for _ in 0..8 {
            lcg.next_u64();
        }
        lcg
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (u32::MAX as f32)
    }

    fn next_usize_range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo { return lo; }
        let range = (hi - lo + 1) as u64;
        lo + (self.next_u64() % range) as usize
    }
}

/// Generate a procedural environment scene.
pub fn generate_environment(width: usize, height: usize, seed: u64) -> AsciiGrid {
    let mut grid = AsciiGrid::new(width, height);
    let mut rng = Lcg::new(seed);

    // Box-drawing border
    let border_fg = palette::COLOR_DIM;
    grid.set(0, 0, AsciiCell { ch: '┌', fg: border_fg, bg: None });
    grid.set(width - 1, 0, AsciiCell { ch: '┐', fg: border_fg, bg: None });
    grid.set(0, height - 1, AsciiCell { ch: '└', fg: border_fg, bg: None });
    grid.set(width - 1, height - 1, AsciiCell { ch: '┘', fg: border_fg, bg: None });

    for col in 1..width - 1 {
        grid.set(col, 0, AsciiCell { ch: '─', fg: border_fg, bg: None });
        grid.set(col, height - 1, AsciiCell { ch: '─', fg: border_fg, bg: None });
    }
    for row in 1..height - 1 {
        grid.set(0, row, AsciiCell { ch: '│', fg: border_fg, bg: None });
        grid.set(width - 1, row, AsciiCell { ch: '│', fg: border_fg, bg: None });
    }

    // Fill interior with terrain
    let floor_chars = ['.', '.', '.', ',', '`', ' '];
    let feature_chars = ['░', '▒', '█', '#', '*', '~'];

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let roll = rng.next_f32();
            let (ch, fg) = if roll < 0.65 {
                // Floor
                let ch = floor_chars[rng.next_usize_range(0, floor_chars.len() - 1)];
                (ch, palette::COLOR_FLOOR)
            } else if roll < 0.80 {
                // Vegetation
                let ch = ['♣', '♠', '*', '+'][rng.next_usize_range(0, 3)];
                (ch, palette::COLOR_FOREST)
            } else if roll < 0.92 {
                // Rock/wall
                let ch = feature_chars[rng.next_usize_range(0, feature_chars.len() - 1)];
                (ch, palette::COLOR_WALL)
            } else {
                // Water
                (
                    ['~', '≈'][rng.next_usize_range(0, 1)],
                    palette::COLOR_WATER,
                )
            };
            grid.set(col, row, AsciiCell { ch, fg, bg: Some(palette::COLOR_FLOOR) });
        }
    }

    grid
}

/// Generate a procedural character portrait.
pub fn generate_character_portrait(width: usize, height: usize, seed: u64) -> AsciiGrid {
    let mut grid = AsciiGrid::new(width, height);
    let mut rng = Lcg::new(seed);

    let fg = palette::COLOR_HERO;
    let bg = Some(palette::COLOR_HP_BG);

    // Simple centered figure
    let cx = width / 2;
    let cy = height / 2;

    // Head
    if cy >= 2 && cx >= 1 && cx + 1 < width {
        grid.set(cx - 1, cy - 2, AsciiCell { ch: '(', fg, bg });
        grid.set(cx, cy - 2, AsciiCell { ch: '^', fg, bg });
        grid.set(cx + 1, cy - 2, AsciiCell { ch: ')', fg, bg });
    }

    // Eyes
    if cy >= 1 && cx >= 1 && cx + 1 < width {
        grid.set(cx - 1, cy - 1, AsciiCell { ch: '|', fg, bg });
        grid.set(cx, cy - 1, AsciiCell { ch: 'o', fg: palette::COLOR_ALLY, bg });
        grid.set(cx + 1, cy - 1, AsciiCell { ch: '|', fg, bg });
    }

    // Body
    if cx >= 1 && cx + 1 < width && cy < height {
        grid.set(cx - 1, cy, AsciiCell { ch: '/', fg, bg });
        grid.set(cx, cy, AsciiCell { ch: '█', fg, bg });
        grid.set(cx + 1, cy, AsciiCell { ch: '\\', fg, bg });
    }

    // Legs
    if cy + 1 < height && cx >= 1 && cx + 1 < width {
        grid.set(cx - 1, cy + 1, AsciiCell { ch: '/', fg, bg });
        grid.set(cx + 1, cy + 1, AsciiCell { ch: '\\', fg, bg });
    }

    // Random embellishments
    for _ in 0..5 {
        let col = rng.next_usize_range(0, width - 1);
        let row = rng.next_usize_range(0, height - 1);
        if grid.get(col, row).map_or(true, |c| c.ch == ' ') {
            let ch = ['·', '°', '•', '∙'][rng.next_usize_range(0, 3)];
            grid.set(col, row, AsciiCell { ch, fg: palette::COLOR_DIM, bg: None });
        }
    }

    grid
}

/// Generate a procedural item icon.
pub fn generate_item_icon(width: usize, height: usize, seed: u64) -> AsciiGrid {
    let mut grid = AsciiGrid::new(width, height);
    let mut rng = Lcg::new(seed);

    let fg = palette::COLOR_CC; // Gold tint for items
    let cx = width / 2;
    let cy = height / 2;

    // Simple diamond shape
    for row in 0..height {
        for col in 0..width {
            let dx = (col as i32 - cx as i32).unsigned_abs();
            let dy = (row as i32 - cy as i32).unsigned_abs();
            let manhattan = dx + dy;
            let radius = (width.min(height) / 2) as u32;

            if manhattan == radius {
                grid.set(col, row, AsciiCell { ch: '◇', fg, bg: None });
            } else if manhattan < radius && manhattan > radius.saturating_sub(2) {
                let ch = ['·', ':', '∙'][rng.next_usize_range(0, 2)];
                grid.set(col, row, AsciiCell { ch, fg: palette::COLOR_DIM, bg: None });
            }
        }
    }

    // Center gem
    grid.set(cx, cy, AsciiCell { ch: '◆', fg: palette::COLOR_HEAL, bg: None });

    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_has_border() {
        let grid = generate_environment(10, 8, 42);
        assert_eq!(grid.get(0, 0).unwrap().ch, '┌');
        assert_eq!(grid.get(9, 0).unwrap().ch, '┐');
        assert_eq!(grid.get(0, 7).unwrap().ch, '└');
        assert_eq!(grid.get(9, 7).unwrap().ch, '┘');
    }

    #[test]
    fn portrait_has_body() {
        let grid = generate_character_portrait(7, 7, 42);
        let cx = 3;
        let cy = 3;
        assert_eq!(grid.get(cx, cy).unwrap().ch, '█');
    }

    #[test]
    fn deterministic_output() {
        let a = generate_environment(10, 8, 99);
        let b = generate_environment(10, 8, 99);
        assert_eq!(a.to_plain_text(), b.to_plain_text());
    }
}
