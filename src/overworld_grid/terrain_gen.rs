//! Terrain generation pipeline: Voronoi rasterization, noise-based terrain,
//! post-processing passes (mountain ridges, forest clumping, rivers, roads).

use super::{
    OverworldGrid, RoadSegment, Settlement, SettlementKind, TerrainType,
    TerrainVisualRegistry, GRID_H, GRID_W,
};

// ---------------------------------------------------------------------------
// Deterministic hash / noise
// ---------------------------------------------------------------------------

fn terrain_hash(seed: u64, x: i32, y: i32) -> u32 {
    let mut h = seed
        .wrapping_add(x as u64)
        .wrapping_mul(374761393)
        .wrapping_add(y as u64)
        .wrapping_mul(668265263);
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h = h ^ (h >> 16);
    h as u32
}

fn hash_f32(seed: u64, x: i32, y: i32) -> f32 {
    (terrain_hash(seed, x, y) & 0xFFFF) as f32 / 65535.0
}

/// Simple value noise with linear interpolation for smoother terrain.
fn value_noise(seed: u64, x: f32, y: f32, scale: f32) -> f32 {
    let sx = x / scale;
    let sy = y / scale;
    let ix = sx.floor() as i32;
    let iy = sy.floor() as i32;
    let fx = sx - sx.floor();
    let fy = sy - sy.floor();

    let v00 = hash_f32(seed, ix, iy);
    let v10 = hash_f32(seed, ix + 1, iy);
    let v01 = hash_f32(seed, ix, iy + 1);
    let v11 = hash_f32(seed, ix + 1, iy + 1);

    let top = v00 + (v10 - v00) * fx;
    let bot = v01 + (v11 - v01) * fx;
    top + (bot - top) * fy
}

/// Multi-octave value noise.
fn fbm_noise(seed: u64, x: f32, y: f32, scale: f32, octaves: u32) -> f32 {
    let mut total = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut max_val = 0.0f32;

    for i in 0..octaves {
        total += value_noise(seed.wrapping_add(i as u64 * 7919), x * frequency, y * frequency, scale)
            * amplitude;
        max_val += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    total / max_val
}

// ---------------------------------------------------------------------------
// Region seed positions (for Voronoi)
// ---------------------------------------------------------------------------

pub struct RegionSeed {
    pub x: f32,
    pub y: f32,
    pub region_id: u16,
    pub faction_id: u8,
}

/// Build region seed positions from hex coordinates.
pub fn build_region_seeds(
    hex_coords: &[(i32, i32)],
    faction_ids: &[usize],
    w: u16,
    h: u16,
) -> Vec<RegionSeed> {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    hex_coords
        .iter()
        .enumerate()
        .map(|(i, (q, r))| {
            let sx = cx + (*q as f32 + *r as f32 * 0.5) * (w as f32 / 9.0);
            let sy = cy + *r as f32 * (h as f32 / 6.0);
            let faction_id = faction_ids.get(i).copied().unwrap_or(0) as u8;
            RegionSeed {
                x: sx,
                y: sy,
                region_id: i as u16,
                faction_id,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Generation pipeline
// ---------------------------------------------------------------------------

/// Generate a full overworld grid from a seed and region data.
pub fn generate_overworld_grid(
    seed: u64,
    region_seeds: &[RegionSeed],
    region_names: &[String],
    visuals: &TerrainVisualRegistry,
) -> OverworldGrid {
    let w = GRID_W;
    let h = GRID_H;
    let mut grid = OverworldGrid::new(w, h);

    // Pass 1: Voronoi rasterization + noise-based terrain
    rasterize_voronoi(&mut grid, seed, region_seeds);

    // Pass 2: Mountain ridge tracing
    trace_mountain_ridges(&mut grid);

    // Pass 3: Forest clumping (cellular automata, 2 iterations)
    for _ in 0..2 {
        forest_cellular_automata(&mut grid);
    }

    // Pass 4: River carving
    carve_rivers(&mut grid, seed);

    // Pass 5: Road pathfinding (simplified Bresenham between settlements)
    let settlements = place_settlements(&mut grid, seed, region_seeds, region_names);
    grid.settlements = settlements;

    // Pass 6: Border computation
    super::border::compute_borders(&mut grid);

    // Apply visual properties
    grid.apply_visuals(visuals);

    grid
}

// ---------------------------------------------------------------------------
// Pass 1: Voronoi + terrain noise
// ---------------------------------------------------------------------------

fn rasterize_voronoi(grid: &mut OverworldGrid, seed: u64, seeds: &[RegionSeed]) {
    let w = grid.width as i32;
    let h = grid.height as i32;

    for y in 0..h {
        for x in 0..w {
            // Find nearest region seed
            let mut best_dist = f32::INFINITY;
            let mut best = 0usize;
            for (i, s) in seeds.iter().enumerate() {
                let dx = x as f32 - s.x;
                let dy = y as f32 - s.y;
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best = i;
                }
            }

            let height = fbm_noise(seed, x as f32, y as f32, 20.0, 3);
            let moisture = fbm_noise(seed.wrapping_add(12345), x as f32, y as f32, 25.0, 2);

            // Determine terrain from height + moisture
            let terrain = classify_terrain(height, moisture, best_dist.sqrt());

            let cell = grid.cell_mut(x as u16, y as u16);
            cell.region_id = seeds[best].region_id;
            cell.faction_id = seeds[best].faction_id;
            cell.terrain = terrain;
            cell.height = height;
            cell.moisture = moisture;
        }
    }
}

fn classify_terrain(height: f32, moisture: f32, dist_to_center: f32) -> TerrainType {
    if height > 0.82 {
        TerrainType::Peak
    } else if height > 0.70 {
        TerrainType::Mountain
    } else if height > 0.60 {
        if moisture > 0.5 { TerrainType::Hills } else { TerrainType::Foothills }
    } else if height < 0.15 && dist_to_center > 25.0 {
        TerrainType::DeepWater
    } else if height < 0.22 && dist_to_center > 20.0 {
        TerrainType::ShallowWater
    } else if height < 0.28 && dist_to_center > 18.0 {
        TerrainType::Coast
    } else if moisture > 0.70 {
        if height < 0.35 { TerrainType::Marsh } else { TerrainType::DenseForest }
    } else if moisture > 0.50 {
        TerrainType::Forest
    } else if moisture > 0.35 {
        TerrainType::Grassland
    } else {
        TerrainType::Plains
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Mountain ridge tracing
// ---------------------------------------------------------------------------

fn trace_mountain_ridges(grid: &mut OverworldGrid) {
    let w = grid.width as i32;
    let h = grid.height as i32;

    // Find high cells and extend ridges horizontally
    let heights: Vec<f32> = grid.cells.iter().map(|c| c.height).collect();

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y as usize * w as usize + x as usize;
            if heights[idx] > 0.72 {
                // Check if this is a local ridge (higher than N/S neighbors)
                let n = heights[(y - 1) as usize * w as usize + x as usize];
                let s = heights[(y + 1) as usize * w as usize + x as usize];
                if heights[idx] > n && heights[idx] > s {
                    grid.cells[idx].terrain = TerrainType::Peak;
                    // Extend ridge E/W
                    for dx in [-1i32, 1] {
                        let nx = x + dx;
                        if nx >= 0 && nx < w {
                            let ni = y as usize * w as usize + nx as usize;
                            if heights[ni] > 0.65 {
                                grid.cells[ni].terrain = TerrainType::Mountain;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 3: Forest cellular automata
// ---------------------------------------------------------------------------

fn forest_cellular_automata(grid: &mut OverworldGrid) {
    let w = grid.width as i32;
    let h = grid.height as i32;
    let snapshot: Vec<TerrainType> = grid.cells.iter().map(|c| c.terrain).collect();

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y as usize * w as usize + x as usize;
            let mut forest_count = 0u8;
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    if dx == 0 && dy == 0 { continue; }
                    let ni = (y + dy) as usize * w as usize + (x + dx) as usize;
                    if matches!(snapshot[ni], TerrainType::Forest | TerrainType::DenseForest) {
                        forest_count += 1;
                    }
                }
            }
            match snapshot[idx] {
                TerrainType::Forest if forest_count < 3 => {
                    grid.cells[idx].terrain = TerrainType::Plains;
                }
                TerrainType::Plains | TerrainType::Grassland if forest_count >= 5 => {
                    grid.cells[idx].terrain = TerrainType::Forest;
                }
                TerrainType::Forest if forest_count >= 7 => {
                    grid.cells[idx].terrain = TerrainType::DenseForest;
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 4: River carving
// ---------------------------------------------------------------------------

fn carve_rivers(grid: &mut OverworldGrid, seed: u64) {
    let w = grid.width as i32;
    let h = grid.height as i32;
    let n_rivers = 3 + (terrain_hash(seed, 999, 999) % 3) as i32;

    for river_idx in 0..n_rivers {
        let start_x = ((terrain_hash(seed, river_idx * 7, 0) % (w as u32 - 10)) + 5) as i32;
        let mut rx = start_x;
        let mut ry = 0i32;

        for _ in 0..(h as usize) {
            if rx < 0 || rx >= w || ry < 0 || ry >= h {
                break;
            }
            let cell = grid.cell_mut(rx as u16, ry as u16);
            if !matches!(cell.terrain, TerrainType::Road | TerrainType::Settlement) {
                cell.terrain = TerrainType::ShallowWater;
            }
            ry += 1;
            let drift = hash_f32(seed.wrapping_add(river_idx as u64 * 100), rx, ry);
            if drift < 0.3 {
                rx -= 1;
            } else if drift > 0.7 {
                rx += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 5: Settlement placement + roads
// ---------------------------------------------------------------------------

fn place_settlements(
    grid: &mut OverworldGrid,
    seed: u64,
    region_seeds: &[RegionSeed],
    region_names: &[String],
) -> Vec<Settlement> {
    let mut settlements = Vec::new();

    for (i, rs) in region_seeds.iter().enumerate() {
        let sx = rs.x.round() as u16;
        let sy = rs.y.round() as u16;
        if sx >= grid.width || sy >= grid.height {
            continue;
        }

        let kind = match terrain_hash(seed, i as i32, 999) % 4 {
            0 => SettlementKind::Town,
            1 => SettlementKind::Castle,
            2 => SettlementKind::Camp,
            _ => SettlementKind::Ruin,
        };

        let name = region_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("Region {}", i));

        // Mark settlement cell
        let cell = grid.cell_mut(sx, sy);
        cell.terrain = TerrainType::Settlement;
        cell.glyph = kind.glyph();

        settlements.push(Settlement {
            name,
            region_id: rs.region_id,
            x: sx,
            y: sy,
            kind,
        });
    }

    // Build roads between neighboring settlements (Bresenham)
    let mut roads = Vec::new();
    for i in 0..settlements.len() {
        for j in (i + 1)..settlements.len() {
            let a = &settlements[i];
            let b = &settlements[j];
            // Only connect if within reasonable distance
            let dx = (a.x as i32 - b.x as i32).abs();
            let dy = (a.y as i32 - b.y as i32).abs();
            if dx + dy < 50 {
                draw_road(grid, a.x, a.y, b.x, b.y);
                roads.push(RoadSegment {
                    from: (a.x, a.y),
                    to: (b.x, b.y),
                });
            }
        }
    }
    grid.roads = roads;

    settlements
}

fn draw_road(grid: &mut OverworldGrid, x0: u16, y0: u16, x1: u16, y1: u16) {
    let steps = ((x1 as i32 - x0 as i32).abs().max((y1 as i32 - y0 as i32).abs())).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let rx = (x0 as f32 + (x1 as f32 - x0 as f32) * t).round() as u16;
        let ry = (y0 as f32 + (y1 as f32 - y0 as f32) * t).round() as u16;
        if rx < grid.width && ry < grid.height {
            let cell = grid.cell_mut(rx, ry);
            if !cell.terrain.is_water() && cell.terrain != TerrainType::Settlement {
                cell.terrain = TerrainType::Road;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_seeds() -> Vec<RegionSeed> {
        vec![
            RegionSeed { x: 30.0, y: 20.0, region_id: 0, faction_id: 0 },
            RegionSeed { x: 75.0, y: 40.0, region_id: 1, faction_id: 1 },
            RegionSeed { x: 120.0, y: 60.0, region_id: 2, faction_id: 2 },
        ]
    }

    #[test]
    fn test_generation_deterministic() {
        let seeds = test_seeds();
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let visuals = TerrainVisualRegistry::default();

        let grid1 = generate_overworld_grid(42, &seeds, &names, &visuals);
        let grid2 = generate_overworld_grid(42, &seeds, &names, &visuals);

        for (a, b) in grid1.cells.iter().zip(grid2.cells.iter()) {
            assert_eq!(a.terrain, b.terrain);
            assert_eq!(a.region_id, b.region_id);
        }
    }

    #[test]
    fn test_terrain_type_distribution() {
        let seeds = test_seeds();
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let visuals = TerrainVisualRegistry::default();

        let grid = generate_overworld_grid(42, &seeds, &names, &visuals);

        // Count terrain types
        let mut counts = std::collections::HashMap::new();
        for cell in &grid.cells {
            *counts.entry(cell.terrain).or_insert(0u32) += 1;
        }

        // Should have reasonable variety
        assert!(counts.len() >= 5, "Should have at least 5 terrain types, got {}", counts.len());

        // Plains/Grassland should be present (most common)
        let plains = counts.get(&TerrainType::Plains).copied().unwrap_or(0)
            + counts.get(&TerrainType::Grassland).copied().unwrap_or(0);
        assert!(plains > 0, "Should have some plains/grassland");
    }

    #[test]
    fn test_settlements_placed() {
        let seeds = test_seeds();
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let visuals = TerrainVisualRegistry::default();

        let grid = generate_overworld_grid(42, &seeds, &names, &visuals);
        assert_eq!(grid.settlements.len(), 3);
    }
}
