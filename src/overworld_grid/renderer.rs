//! 5-layer batched renderer for the overworld grid.
//!
//! Layer 0: Faction background tint (low-alpha rect per cell)
//! Layer 1: Terrain glyphs (textured quads from glyph atlas)
//! Layer 2: Faction borders (box-drawing chars)
//! Layer 3: Settlements + parties (bright faction-colored glyphs)
//! Layer 4: UI overlays (egui native — tooltips, path preview, selection)
//!
//! The renderer uses viewport culling to only emit quads for visible cells,
//! and caches terrain layers (L0+L1+L2), rebuilding only on camera/faction changes.

use super::camera::OverworldCamera;
use super::OverworldGrid;

// ---------------------------------------------------------------------------
// Faction colors
// ---------------------------------------------------------------------------

/// Standard faction color palette.
pub fn faction_color(faction_id: u8) -> [u8; 3] {
    match faction_id {
        0 => [0x40, 0xA0, 0xFF], // Blue
        1 => [0xFF, 0x60, 0x40], // Red
        2 => [0x40, 0xFF, 0x80], // Green
        3 => [0xFF, 0xCC, 0x40], // Gold
        4 => [0xCC, 0x60, 0xFF], // Purple
        _ => [0xAA, 0xAA, 0xAA], // Gray
    }
}

/// Blend two colors with alpha factor t (0.0 = all a, 1.0 = all b).
pub fn blend_colors(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let inv = 1.0 - t;
    [
        (a[0] as f32 * inv + b[0] as f32 * t) as u8,
        (a[1] as f32 * inv + b[1] as f32 * t) as u8,
        (a[2] as f32 * inv + b[2] as f32 * t) as u8,
    ]
}

// ---------------------------------------------------------------------------
// Render data (intermediate representation before egui output)
// ---------------------------------------------------------------------------

/// A single character to render at a specific position with color.
#[derive(Debug, Clone)]
pub struct RenderGlyph {
    pub x: u16,
    pub y: u16,
    pub glyph: char,
    pub fg: [u8; 3],
    pub bg: Option<[u8; 3]>,
}

/// Rendered frame data — the output of the rendering pipeline.
#[derive(Debug, Clone, Default)]
pub struct OverworldRenderFrame {
    pub glyphs: Vec<RenderGlyph>,
    pub viewport_x0: i32,
    pub viewport_y0: i32,
    pub viewport_x1: i32,
    pub viewport_y1: i32,
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// Cached terrain render data. Invalidated on camera movement or faction change.
pub struct TerrainRenderCache {
    pub frame: OverworldRenderFrame,
    pub last_camera_x: f32,
    pub last_camera_y: f32,
    pub last_camera_zoom: f32,
    pub last_faction_version: u64,
}

impl Default for TerrainRenderCache {
    fn default() -> Self {
        Self {
            frame: OverworldRenderFrame::default(),
            last_camera_x: -1.0,
            last_camera_y: -1.0,
            last_camera_zoom: -1.0,
            last_faction_version: u64::MAX,
        }
    }
}

impl TerrainRenderCache {
    pub fn is_valid(&self, camera: &OverworldCamera, faction_version: u64) -> bool {
        (self.last_camera_x - camera.pos_x).abs() < 0.01
            && (self.last_camera_y - camera.pos_y).abs() < 0.01
            && (self.last_camera_zoom - camera.zoom).abs() < 0.01
            && self.last_faction_version == faction_version
    }

    pub fn update(&mut self, camera: &OverworldCamera, faction_version: u64, frame: OverworldRenderFrame) {
        self.last_camera_x = camera.pos_x;
        self.last_camera_y = camera.pos_y;
        self.last_camera_zoom = camera.zoom;
        self.last_faction_version = faction_version;
        self.frame = frame;
    }
}

// ---------------------------------------------------------------------------
// Render pipeline
// ---------------------------------------------------------------------------

/// Render the overworld grid into a frame of glyphs.
/// Performs viewport culling based on camera position.
pub fn render_terrain_layers(
    grid: &OverworldGrid,
    camera: &OverworldCamera,
    viewport_w: f32,
    viewport_h: f32,
) -> OverworldRenderFrame {
    let (x0, y0, x1, y1) = camera.visible_rect(viewport_w, viewport_h);
    let mut frame = OverworldRenderFrame {
        viewport_x0: x0,
        viewport_y0: y0,
        viewport_x1: x1,
        viewport_y1: y1,
        glyphs: Vec::with_capacity(((x1 - x0) * (y1 - y0)) as usize),
    };

    for y in y0..y1 {
        for x in x0..x1 {
            if !grid.in_bounds(x, y) {
                continue;
            }
            let cell = grid.cell(x as u16, y as u16);

            // L0: Faction background tint (10-15% alpha blend)
            let fc = faction_color(cell.faction_id);
            let tinted_fg = blend_colors(cell.fg_color, fc, 0.15);

            // L2: Border override
            let (glyph, fg) = if cell.is_border {
                (cell.border_glyph, [0x88, 0x88, 0x88])
            } else {
                (cell.glyph, tinted_fg)
            };

            frame.glyphs.push(RenderGlyph {
                x: x as u16,
                y: y as u16,
                glyph,
                fg,
                bg: Some(blend_colors([13, 16, 22], fc, 0.10)),
            });
        }
    }

    frame
}

/// Render entity layer (L3): settlements and parties as bright glyphs.
pub fn render_entity_layer(
    grid: &OverworldGrid,
    camera: &OverworldCamera,
    viewport_w: f32,
    viewport_h: f32,
    player_region: usize,
    selected_region: usize,
) -> Vec<RenderGlyph> {
    let (x0, y0, x1, y1) = camera.visible_rect(viewport_w, viewport_h);
    let mut entities = Vec::new();

    // Settlement glyphs
    for settlement in &grid.settlements {
        let sx = settlement.x as i32;
        let sy = settlement.y as i32;
        if sx >= x0 && sx < x1 && sy >= y0 && sy < y1 {
            let cell = grid.cell(settlement.x, settlement.y);
            let fc = faction_color(cell.faction_id);
            entities.push(RenderGlyph {
                x: settlement.x,
                y: settlement.y,
                glyph: settlement.kind.glyph(),
                fg: fc,
                bg: None,
            });
        }
    }

    // Player marker (@) — bright green
    for settlement in &grid.settlements {
        if settlement.region_id as usize == player_region {
            let sy = settlement.y.saturating_sub(1);
            if (settlement.x as i32) >= x0 && (settlement.x as i32) < x1
                && (sy as i32) >= y0 && (sy as i32) < y1
            {
                entities.push(RenderGlyph {
                    x: settlement.x,
                    y: sy,
                    glyph: '@',
                    fg: [0x40, 0xFF, 0x40],
                    bg: None,
                });
            }
        }
    }

    // Selected region marker (>)
    if selected_region != player_region {
        for settlement in &grid.settlements {
            if settlement.region_id as usize == selected_region {
                let sy = settlement.y.saturating_sub(1);
                if (settlement.x as i32) >= x0 && (settlement.x as i32) < x1
                    && (sy as i32) >= y0 && (sy as i32) < y1
                {
                    entities.push(RenderGlyph {
                        x: settlement.x,
                        y: sy,
                        glyph: '>',
                        fg: [0x60, 0xDD, 0xFF],
                        bg: None,
                    });
                }
            }
        }
    }

    entities
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overworld_grid::{OverworldGrid, GRID_W, GRID_H, TerrainType, Settlement, SettlementKind};

    #[test]
    fn test_viewport_culling_reduces_glyphs() {
        let mut grid = OverworldGrid::new(GRID_W, GRID_H);
        // Fill with plains
        for cell in &mut grid.cells {
            cell.terrain = TerrainType::Plains;
            cell.glyph = '.';
            cell.fg_color = [0x8A, 0x9A, 0x7A];
        }

        let mut camera = OverworldCamera::new(GRID_W as f32, GRID_H as f32);
        camera.pos_x = 0.0;
        camera.pos_y = 0.0;

        // Full viewport
        let frame_full = render_terrain_layers(&grid, &camera, 150.0, 80.0);

        // Smaller viewport
        let frame_small = render_terrain_layers(&grid, &camera, 40.0, 20.0);

        assert!(
            frame_small.glyphs.len() < frame_full.glyphs.len(),
            "Smaller viewport should produce fewer glyphs: {} vs {}",
            frame_small.glyphs.len(),
            frame_full.glyphs.len()
        );
    }

    #[test]
    fn test_faction_color_tinting() {
        let base = [0x8A, 0x9A, 0x7A];
        let fc = [0x40, 0xA0, 0xFF];
        let blended = blend_colors(base, fc, 0.15);

        // Should shift slightly toward faction color
        assert!(blended[2] > base[2], "Blue channel should increase with blue faction tint");
    }

    #[test]
    fn test_cache_invalidation() {
        let camera = OverworldCamera::new(150.0, 80.0);
        let cache = TerrainRenderCache::default();

        // Default cache should be invalid
        assert!(!cache.is_valid(&camera, 0));
    }
}
