//! Faction border computation using 4-bit adjacency masks → box-drawing characters.

use super::OverworldGrid;

/// Compute faction borders for the grid. A cell is a border cell if any cardinal
/// neighbor has a different `faction_id`. The 4-bit adjacency mask (N=1, E=2, S=4, W=8)
/// maps to box-drawing characters.
pub fn compute_borders(grid: &mut OverworldGrid) {
    let w = grid.width as i32;
    let h = grid.height as i32;

    // Snapshot faction IDs
    let faction_ids: Vec<u8> = grid.cells.iter().map(|c| c.faction_id).collect();

    for y in 0..h {
        for x in 0..w {
            let idx = y as usize * w as usize + x as usize;
            let my_faction = faction_ids[idx];

            let mut mask: u8 = 0;
            // N
            if y > 0 {
                let ni = (y - 1) as usize * w as usize + x as usize;
                if faction_ids[ni] != my_faction {
                    mask |= 1;
                }
            }
            // E
            if x < w - 1 {
                let ni = y as usize * w as usize + (x + 1) as usize;
                if faction_ids[ni] != my_faction {
                    mask |= 2;
                }
            }
            // S
            if y < h - 1 {
                let ni = (y + 1) as usize * w as usize + x as usize;
                if faction_ids[ni] != my_faction {
                    mask |= 4;
                }
            }
            // W
            if x > 0 {
                let ni = y as usize * w as usize + (x - 1) as usize;
                if faction_ids[ni] != my_faction {
                    mask |= 8;
                }
            }

            if mask != 0 {
                grid.cells[idx].is_border = true;
                grid.cells[idx].border_glyph = border_glyph(mask);
            }
        }
    }
}

/// Map a 4-bit adjacency mask to a box-drawing character.
fn border_glyph(mask: u8) -> char {
    match mask {
        // Single directions
        0b0001 | 0b0100 | 0b0101 => '─', // N, S, or N+S → horizontal
        0b0010 | 0b1000 | 0b1010 => '│', // E, W, or E+W → vertical
        // Corners
        0b0011 => '╭', // N+E
        0b0110 => '╮', // E+S
        0b1100 => '╰', // S+W (note: mask bits S=4,W=8 → 0b1100)
        0b1001 => '╯', // W+N
        // 3+ edges → intersection
        _ if mask.count_ones() >= 3 => '┼',
        // Fallback
        _ => '─',
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overworld_grid::{MapCell, OverworldGrid, TerrainType};

    fn make_small_grid() -> OverworldGrid {
        // 4x4 grid with faction 0 on left half, faction 1 on right half
        let mut grid = OverworldGrid::new(4, 4);
        for y in 0..4u16 {
            for x in 0..4u16 {
                let cell = grid.cell_mut(x, y);
                cell.faction_id = if x < 2 { 0 } else { 1 };
                cell.terrain = TerrainType::Plains;
            }
        }
        grid
    }

    #[test]
    fn test_border_detection() {
        let mut grid = make_small_grid();
        compute_borders(&mut grid);

        // Column 1 (x=1) should have borders (eastern neighbor is faction 1)
        assert!(grid.cell(1, 0).is_border);
        assert!(grid.cell(1, 1).is_border);

        // Column 2 (x=2) should have borders (western neighbor is faction 0)
        assert!(grid.cell(2, 0).is_border);

        // Column 0 (x=0) should NOT have borders (all neighbors are faction 0)
        // except it borders the edge... let's check the interior
        // Actually x=0,y=1: N=faction0, E=faction0, S=faction0, W=edge(skip) → no border
        assert!(!grid.cell(0, 1).is_border);
    }

    #[test]
    fn test_border_glyph_mapping() {
        assert_eq!(border_glyph(0b0001), '─'); // N
        assert_eq!(border_glyph(0b0010), '│'); // E
        assert_eq!(border_glyph(0b0011), '╭'); // N+E
        assert_eq!(border_glyph(0b0110), '╮'); // E+S
        assert_eq!(border_glyph(0b1111), '┼'); // All
    }
}
