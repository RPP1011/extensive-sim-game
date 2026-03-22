/// Game color palette for ASCII art generation.
///
/// Colors are extracted from `src/ascii_viewport/mod.rs` and
/// `src/hub_ui_draw/overworld_map_strategic.rs` so that generated
/// ASCII art uses consistent colors.

/// An RGB color triple.
pub type Rgb = [u8; 3];

// Terrain colors
pub const COLOR_WALL: Rgb = [0x3C, 0x37, 0x32];
pub const COLOR_FLOOR: Rgb = [0x26, 0x28, 0x23];
pub const COLOR_ELEVATED: Rgb = [0x50, 0x4B, 0x37];
pub const COLOR_HALF_COVER: Rgb = [0x5A, 0x50, 0x3C];

// Team colors
pub const COLOR_HERO: Rgb = [0x50, 0xC8, 0x78];
pub const COLOR_ALLY: Rgb = [0x50, 0xA0, 0xFF];
pub const COLOR_ENEMY: Rgb = [0xFF, 0x5A, 0x50];
pub const COLOR_DEAD: Rgb = [0x50, 0x50, 0x50];

// Status colors
pub const COLOR_CC: Rgb = [0xE6, 0xC8, 0x50];
pub const COLOR_ZONE: Rgb = [0xA0, 0x78, 0xDC];
pub const COLOR_DMG: Rgb = [0xFF, 0xB4, 0x3C];
pub const COLOR_HEAL: Rgb = [0x50, 0xDC, 0x50];
pub const COLOR_DEATH: Rgb = [0xFF, 0x3C, 0x3C];

// UI chrome
pub const COLOR_HP_HIGH: Rgb = [0x50, 0xC8, 0x50];
pub const COLOR_HP_MID: Rgb = [0xDC, 0xC8, 0x32];
pub const COLOR_HP_LOW: Rgb = [0xDC, 0x3C, 0x28];
pub const COLOR_HP_BG: Rgb = [0x32, 0x32, 0x32];
pub const COLOR_HEADER: Rgb = [0xA0, 0xAA, 0xAB];
pub const COLOR_DIM: Rgb = [0x37, 0x3C, 0x44];
pub const COLOR_SECTION: Rgb = [0x64, 0x6E, 0x7D];

// Overworld strategic map
pub const COLOR_PLAINS: Rgb = [0x8A, 0x9A, 0x7A];
pub const COLOR_FOREST: Rgb = [0x5A, 0x7A, 0x4A];
pub const COLOR_MOUNTAIN: Rgb = [0x8A, 0x7A, 0x5A];
pub const COLOR_WATER: Rgb = [0x7A, 0x9A, 0xBB];
pub const COLOR_SETTLEMENT: Rgb = [0xDD, 0xDD, 0xCC];

/// Full game palette for color snapping.
pub const GAME_PALETTE: &[Rgb] = &[
    COLOR_WALL, COLOR_FLOOR, COLOR_ELEVATED, COLOR_HALF_COVER,
    COLOR_HERO, COLOR_ALLY, COLOR_ENEMY, COLOR_DEAD,
    COLOR_CC, COLOR_ZONE, COLOR_DMG, COLOR_HEAL, COLOR_DEATH,
    COLOR_HP_HIGH, COLOR_HP_MID, COLOR_HP_LOW, COLOR_HP_BG,
    COLOR_HEADER, COLOR_DIM, COLOR_SECTION,
    COLOR_PLAINS, COLOR_FOREST, COLOR_MOUNTAIN, COLOR_WATER, COLOR_SETTLEMENT,
    // Pure black/white for high contrast.
    [0x00, 0x00, 0x00],
    [0xFF, 0xFF, 0xFF],
];

/// Snap an arbitrary RGB color to the nearest color in the game palette.
pub fn snap_to_palette(color: Rgb) -> Rgb {
    let mut best = GAME_PALETTE[0];
    let mut best_dist = u32::MAX;
    for &p in GAME_PALETTE {
        let dr = (color[0] as i32 - p[0] as i32).unsigned_abs();
        let dg = (color[1] as i32 - p[1] as i32).unsigned_abs();
        let db = (color[2] as i32 - p[2] as i32).unsigned_abs();
        let dist = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best = p;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_exact_color_returns_same() {
        assert_eq!(snap_to_palette(COLOR_HERO), COLOR_HERO);
    }

    #[test]
    fn snap_near_white_returns_white() {
        let near_white = [0xFE, 0xFE, 0xFE];
        assert_eq!(snap_to_palette(near_white), [0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn snap_near_black_returns_black() {
        let near_black = [0x02, 0x01, 0x03];
        assert_eq!(snap_to_palette(near_black), [0x00, 0x00, 0x00]);
    }
}
