use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use super::types::*;

fn axial_radius2_sorted() -> Vec<(i32, i32)> {
    let mut coords = Vec::new();
    let radius = 2;
    for q in -radius..=radius {
        let r_min = (-radius).max(-q - radius);
        let r_max = radius.min(-q + radius);
        for r in r_min..=r_max {
            coords.push((q, r));
        }
    }
    coords.sort_by_key(|(q, r)| (*q, *r));
    coords
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub(crate) fn rand01(seed: u64) -> f64 {
    ((splitmix64(seed) >> 11) & ((1_u64 << 53) - 1)) as f64 / (1_u64 << 53) as f64
}

pub fn build_site_positions(overworld: &OverworldData) -> Vec<(f64, f64)> {
    let n = overworld.regions.len();
    if n == 19 {
        return axial_radius2_sorted()
            .into_iter()
            .map(|(q, r)| (q as f64 + r as f64 * 0.5, r as f64 * 0.8660254))
            .collect();
    }

    (0..n)
        .map(|i| {
            let theta = (i as f64 / n.max(1) as f64) * PI * 2.0;
            (theta.cos(), theta.sin())
        })
        .collect()
}

pub fn normalize_points(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let dx = (max_x - min_x).max(1e-6);
    let dy = (max_y - min_y).max(1e-6);

    points.iter().map(|(x, y)| ((x - min_x) / dx, (y - min_y) / dy)).collect()
}

pub fn compute_region_weights(overworld: &OverworldData, strength_scale: f64) -> Vec<f64> {
    let strengths: Vec<f64> = overworld.factions.iter().map(|f| f.strength).collect();
    let s_min = strengths.iter().copied().fold(f64::INFINITY, f64::min);
    let s_max = strengths.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let denom = (s_max - s_min).max(1e-6);

    let mut faction_norm: HashMap<usize, f64> = HashMap::new();
    for faction in &overworld.factions {
        let v = if strengths.is_empty() { 0.0 } else { (faction.strength - s_min) / denom };
        faction_norm.insert(faction.id, v);
    }

    overworld
        .regions
        .iter()
        .map(|r| {
            let base = faction_norm.get(&r.owner_faction_id).copied().unwrap_or(0.0) * strength_scale;
            let local_mod = (r.control - r.unrest) / 100.0 * 0.05;
            base + local_mod
        })
        .collect()
}

pub fn weighted_partition(
    points: &[(f64, f64)],
    weights: &[f64],
    width: usize,
    height: usize,
) -> Vec<Vec<usize>> {
    let mut grid = vec![vec![0; width]; height];
    for (y, row) in grid.iter_mut().enumerate() {
        let py = y as f64 / (height.saturating_sub(1).max(1) as f64);
        for (x, cell) in row.iter_mut().enumerate() {
            let px = x as f64 / (width.saturating_sub(1).max(1) as f64);
            let mut best = 0;
            let mut best_score = f64::INFINITY;
            for (i, (sx, sy)) in points.iter().copied().enumerate() {
                let dx = px - sx;
                let dy = py - sy;
                let score = dx * dx + dy * dy - weights[i];
                if score < best_score {
                    best_score = score;
                    best = i;
                }
            }
            *cell = best;
        }
    }
    grid
}

pub fn boundary_complexity(grid: &[Vec<usize>], idx: usize) -> f64 {
    let h = grid.len();
    let w = grid.first().map_or(0, Vec::len);
    let mut edge: f64 = 0.0;
    let mut area: f64 = 0.0;

    for y in 0..h {
        for x in 0..w {
            if grid[y][x] != idx { continue; }
            area += 1.0;
            for (nx, ny) in [
                (x as isize + 1, y as isize),
                (x as isize - 1, y as isize),
                (x as isize, y as isize + 1),
                (x as isize, y as isize - 1),
            ] {
                if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h || grid[ny as usize][nx as usize] != idx {
                    edge += 1.0;
                }
            }
        }
    }

    if area == 0.0 { 0.0 } else { edge / area.sqrt().max(1.0) }
}

pub fn faction_area_ratio(grid: &[Vec<usize>], overworld: &OverworldData) -> HashMap<usize, f64> {
    let h = grid.len();
    let w = grid.first().map_or(0, Vec::len);
    let total = (h * w).max(1) as f64;

    let mut area_px: HashMap<usize, usize> = HashMap::new();
    for row in grid {
        for idx in row {
            let owner = overworld.regions.get(*idx).map(|r| r.owner_faction_id).unwrap_or(0);
            *area_px.entry(owner).or_insert(0) += 1;
        }
    }

    area_px.into_iter().map(|(fid, px)| (fid, px as f64 / total)).collect()
}

pub fn settlement_candidates(regions: &[RegionInput], per_faction: usize) -> Vec<Settlement> {
    let mut grouped: HashMap<usize, Vec<&RegionInput>> = HashMap::new();
    for region in regions {
        grouped.entry(region.owner_faction_id).or_default().push(region);
    }

    let mut picks = Vec::new();
    for (fid, mut arr) in grouped {
        arr.sort_by(|a, b| {
            let ka = (a.control - a.unrest, a.intel_level);
            let kb = (b.control - b.unrest, b.intel_level);
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
        });

        for region in arr.into_iter().take(per_faction) {
            picks.push(Settlement {
                faction_id: fid,
                region_id: region.id,
                name: region.name.clone(),
            });
        }
    }

    picks
}

pub fn roads_from_neighbors(regions: &[RegionInput]) -> Vec<Road> {
    let mut roads: HashSet<(usize, usize)> = HashSet::new();
    for region in regions {
        let a = region.id;
        for &b in &region.neighbors {
            if a == b { continue; }
            if a < b { roads.insert((a, b)); } else { roads.insert((b, a)); }
        }
    }

    let mut out = roads.into_iter().map(|(a, b)| Road { a, b }).collect::<Vec<_>>();
    out.sort_by_key(|r| (r.a, r.b));
    out
}
