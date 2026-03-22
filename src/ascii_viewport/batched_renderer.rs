//! Batched mesh-based ASCII grid renderer.
//!
//! Converts a grid of `CellData` into a single `egui::Mesh` using
//! the glyph atlas texture, with viewport culling and per-vertex coloring.

use bevy_egui::egui;
use super::glyph_atlas::GlyphAtlas;

/// Rendering data for a single grid cell.
#[derive(Clone)]
pub struct CellData {
    /// Character to display in this cell.
    pub ch: char,
    /// Foreground color for the character.
    pub fg: egui::Color32,
    /// Optional background color for the cell.
    pub bg: Option<egui::Color32>,
}

impl Default for CellData {
    fn default() -> Self {
        Self {
            ch: '.',
            fg: egui::Color32::from_rgb(38, 40, 35),
            bg: None,
        }
    }
}

/// Render a grid of cells as a single batched egui::Mesh.
///
/// Only cells visible within the viewport rectangle are emitted.
/// Each cell produces up to two quads: an optional background rect
/// and a foreground textured glyph quad.
///
/// # Arguments
/// * `atlas` - The glyph texture atlas for UV lookups
/// * `grid` - 2D grid of cell data (rows x cols)
/// * `viewport` - The visible rectangle in screen coordinates
/// * `origin` - Top-left position where the grid starts rendering
///
/// Returns a tuple of (background mesh using WHITE texture, foreground mesh using atlas texture).
pub fn render_grid_mesh(
    atlas: &GlyphAtlas,
    grid: &[Vec<CellData>],
    viewport: egui::Rect,
    origin: egui::Pos2,
) -> (egui::Mesh, egui::Mesh) {
    let cell_w = atlas.cell_width;
    let cell_h = atlas.cell_height;
    let rows = grid.len();
    let cols = if rows > 0 { grid[0].len() } else { 0 };

    // Compute visible row/col range from viewport
    let min_row = ((viewport.min.y - origin.y) / cell_h).floor().max(0.0) as usize;
    let max_row = ((viewport.max.y - origin.y) / cell_h).ceil().max(0.0) as usize;
    let min_col = ((viewport.min.x - origin.x) / cell_w).floor().max(0.0) as usize;
    let max_col = ((viewport.max.x - origin.x) / cell_w).ceil().max(0.0) as usize;

    let min_row = min_row.min(rows);
    let max_row = max_row.min(rows);
    let min_col = min_col.min(cols);
    let max_col = max_col.min(cols);

    // Pre-allocate meshes
    let visible_cells = (max_row - min_row) * (max_col - min_col);
    let mut bg_mesh = egui::Mesh::default();
    bg_mesh.vertices.reserve(visible_cells * 4);
    bg_mesh.indices.reserve(visible_cells * 6);

    let mut fg_mesh = egui::Mesh::with_texture(atlas.texture_id);
    fg_mesh.vertices.reserve(visible_cells * 4);
    fg_mesh.indices.reserve(visible_cells * 6);

    let white_uv = egui::pos2(0.0, 0.0); // Dummy UV for solid color quads

    for row in min_row..max_row {
        for col in min_col..max_col {
            let cell = &grid[row][col];
            let x = origin.x + col as f32 * cell_w;
            let y = origin.y + row as f32 * cell_h;
            let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell_w, cell_h));

            // Background quad (solid color, no texture)
            if let Some(bg) = cell.bg {
                let base_idx = bg_mesh.vertices.len() as u32;
                bg_mesh.vertices.push(egui::epaint::Vertex { pos: rect.left_top(), uv: white_uv, color: bg });
                bg_mesh.vertices.push(egui::epaint::Vertex { pos: rect.right_top(), uv: white_uv, color: bg });
                bg_mesh.vertices.push(egui::epaint::Vertex { pos: rect.right_bottom(), uv: white_uv, color: bg });
                bg_mesh.vertices.push(egui::epaint::Vertex { pos: rect.left_bottom(), uv: white_uv, color: bg });
                bg_mesh.indices.extend_from_slice(&[
                    base_idx, base_idx + 1, base_idx + 2,
                    base_idx, base_idx + 2, base_idx + 3,
                ]);
            }

            // Foreground glyph quad (textured)
            let uv = atlas.uv_for(cell.ch);
            let base_idx = fg_mesh.vertices.len() as u32;
            fg_mesh.vertices.push(egui::epaint::Vertex { pos: rect.left_top(), uv: uv.left_top(), color: cell.fg });
            fg_mesh.vertices.push(egui::epaint::Vertex { pos: rect.right_top(), uv: uv.right_top(), color: cell.fg });
            fg_mesh.vertices.push(egui::epaint::Vertex { pos: rect.right_bottom(), uv: uv.right_bottom(), color: cell.fg });
            fg_mesh.vertices.push(egui::epaint::Vertex { pos: rect.left_bottom(), uv: uv.left_bottom(), color: cell.fg });
            fg_mesh.indices.extend_from_slice(&[
                base_idx, base_idx + 1, base_idx + 2,
                base_idx, base_idx + 2, base_idx + 3,
            ]);
        }
    }

    (bg_mesh, fg_mesh)
}

/// Convenience function to build a grid from the existing LayoutJob-based cell data.
/// Converts sim state grid information into a `Vec<Vec<CellData>>` for the batched renderer.
pub fn build_grid_from_nav(
    nav: &crate::ai::pathing::GridNav,
    unit_cells: &std::collections::HashMap<(i32, i32), (String, egui::Color32)>,
    cols: i32,
    rows: i32,
) -> Vec<Vec<CellData>> {
    let color_wall = egui::Color32::from_rgb(60, 55, 50);
    let color_floor = egui::Color32::from_rgb(38, 40, 35);
    let color_elevated = egui::Color32::from_rgb(80, 75, 55);
    let color_half_cover = egui::Color32::from_rgb(90, 80, 60);

    let mut grid = Vec::with_capacity(rows as usize);

    for row in 0..rows {
        let mut row_cells = Vec::with_capacity(cols as usize * 2); // 2 chars per cell
        for col in 0..cols {
            let cell_key = (col, row);
            let (ch1, ch2, fg) = if let Some((label, color)) = unit_cells.get(&cell_key) {
                let chars: Vec<char> = label.chars().collect();
                let c1 = chars.first().copied().unwrap_or(' ');
                let c2 = chars.get(1).copied().unwrap_or(' ');
                (c1, c2, *color)
            } else if nav.blocked.contains(&cell_key) {
                ('\u{2588}', '\u{2588}', color_wall)
            } else if nav.elevation_by_cell.get(&cell_key).copied().unwrap_or(0.0) > 1.0 {
                ('\u{25B2}', '\u{25B2}', color_elevated)
            } else if nav.elevation_by_cell.get(&cell_key).copied().unwrap_or(0.0) > 0.3 {
                ('\u{2591}', '\u{2591}', color_half_cover)
            } else {
                ('.', ' ', color_floor)
            };
            row_cells.push(CellData { ch: ch1, fg, bg: None });
            row_cells.push(CellData { ch: ch2, fg, bg: None });
        }
        grid.push(row_cells);
    }

    grid
}
