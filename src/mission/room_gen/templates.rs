use super::lcg::{Lcg, ObstacleRegion};
use super::nav::NavGrid;
use super::primitives::*;

// ---------------------------------------------------------------------------
// Room-Type Template Generators
// ---------------------------------------------------------------------------

pub(crate) fn generate_entry_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let template = rng.next_usize_range(0, 4);
    let mut obs = Vec::new();
    let mid_c = cols / 2;
    let mid_r = rows / 2;
    match template {
        0 => {
            // "Pillared Hall": symmetric pillar grid with wide spacing — clean lanes
            obs.extend(place_pillar_grid(
                nav, rng, cols / 4, rows / 4, 3 * cols / 4, 3 * rows / 4, 5, 1, 1.5,
            ));
        }
        1 => {
            // "Divided Room": central horizontal wall with gap, mirrored cover blocks
            obs.extend(place_wall_segment(
                nav, rng, cols / 4, mid_r, cols / 2, true, 3, 1.5,
            ));
            // Symmetric cover blocks in each third
            obs.extend(place_barricade_line(nav, rng, cols / 5, cols / 5 + 3, rows / 3, 2, 1, 1.0));
            obs.extend(place_barricade_line(nav, rng, cols / 5, cols / 5 + 3, 2 * rows / 3, 2, 1, 1.0));
            obs.extend(place_barricade_line(nav, rng, 3 * cols / 5, 3 * cols / 5 + 3, rows / 3, 2, 1, 1.0));
            obs.extend(place_barricade_line(nav, rng, 3 * cols / 5, 3 * cols / 5 + 3, 2 * rows / 3, 2, 1, 1.0));
        }
        2 => {
            // "Fortified Positions": elevated platform at centre with sandbag arcs on flanks
            obs.extend(place_elevated_platform(nav, rng, mid_c - 1, mid_r - 1, 3, 3, 0.8));
            obs.extend(place_sandbag_arc(nav, rng, cols / 4, mid_r, 2, 4, 0.7));
            obs.extend(place_sandbag_arc(nav, rng, 3 * cols / 4, mid_r, 2, 4, 0.7));
        }
        _ => {
            // "Mirrored Ruins": symmetric L-shapes creating flanking corridors
            obs.extend(place_l_shape(nav, rng, cols / 4, rows / 3, 3, 1, 0, 1.5));
            obs.extend(place_l_shape(nav, rng, 3 * cols / 4, 2 * rows / 3, 3, 1, 2, 1.5));
            // Short horizontal barricades near each side for cover
            obs.extend(place_barricade_line(nav, rng, cols / 4 - 1, cols / 4 + 2, 2 * rows / 3, 2, 1, 1.0));
            obs.extend(place_barricade_line(nav, rng, 3 * cols / 4 - 2, 3 * cols / 4 + 1, rows / 3, 2, 1, 1.0));
        }
    }
    obs
}

pub(crate) fn generate_pressure_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let template = rng.next_usize_range(0, 4);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_corridor_walls(nav, rng, cols / 4, 3 * cols / 4, rows / 2, 4, 1.5));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 2, 3, 1.0));
        }
        1 => {
            obs.extend(place_barricade_line(nav, rng, cols / 4, 3 * cols / 4, rows / 3, 3, 2, 1.2));
            obs.extend(place_barricade_line(nav, rng, cols / 4 + 1, 3 * cols / 4 + 1, 2 * rows / 3, 3, 2, 1.2));
            obs.extend(place_pillar_grid(nav, rng, cols / 3, rows / 4, 2 * cols / 3, 3 * rows / 4, 5, 1, 1.5));
        }
        2 => {
            let gap_col = cols / 2;
            let wall_len = rows / 2 - 2;
            obs.extend(place_wall_segment(nav, rng, gap_col - 1, 1, wall_len, false, 0, 1.8));
            obs.extend(place_wall_segment(nav, rng, gap_col + 1, 1, wall_len, false, 0, 1.8));
            obs.extend(place_wall_segment(nav, rng, gap_col - 1, rows / 2 + 2, wall_len, false, 0, 1.8));
            obs.extend(place_wall_segment(nav, rng, gap_col + 1, rows / 2 + 2, wall_len, false, 0, 1.8));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 2, 2, 2, 1.0));
        }
        3 => {
            // "Elevated Chokepoint": raised centre with sandbag defenses
            obs.extend(place_elevated_platform(nav, rng, cols / 3, rows / 4, cols / 3, rows / 2, 1.0));
            obs.extend(place_sandbag_arc(nav, rng, cols / 3, rows / 2, 2, 4, 0.7));
            obs.extend(place_sandbag_arc(nav, rng, 2 * cols / 3, rows / 2, 2, 4, 0.7));
            obs.extend(place_barricade_line(nav, rng, cols / 4, 3 * cols / 4, rows / 3, 2, 2, 0.8));
        }
        _ => {
            obs.extend(place_cover_cluster(nav, rng, cols / 4, rows / 2, 3, 3, 1.0));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 3, 3, 3, 1.0));
            obs.extend(place_cover_cluster(nav, rng, 3 * cols / 4, rows / 2, 3, 3, 1.0));
            obs.extend(place_wall_segment(nav, rng, cols / 2 - 2, rows / 2, 5, true, 0, 1.2));
        }
    }
    obs
}

pub(crate) fn generate_pivot_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let template = rng.next_usize_range(0, 3);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_wall_segment(nav, rng, cols / 4, rows / 2, cols / 2, true, 2, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 2, rows / 4, rows / 2, false, 2, 1.5));
        }
        1 => {
            obs.extend(place_l_shape(nav, rng, cols / 4, rows / 4, 3, 1, 0, 1.5));
            obs.extend(place_l_shape(nav, rng, 3 * cols / 4, 3 * rows / 4, 3, 1, 2, 1.5));
        }
        2 => {
            // "High Ground Pivot": two elevated platforms with cover lips
            obs.extend(place_elevated_platform(nav, rng, cols / 4, rows / 4, 3, 3, 1.0));
            obs.extend(place_elevated_platform(nav, rng, 3 * cols / 4 - 2, 3 * rows / 4 - 2, 3, 3, 1.0));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 2, 3, 1.0));
        }
        _ => {
            obs.extend(place_pillar_grid(nav, rng, cols / 4, rows / 4, 3 * cols / 4, 3 * rows / 4, 3, 1, 1.5));
        }
    }
    obs
}

pub(crate) fn generate_setpiece_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let template = rng.next_usize_range(0, 3);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_corridor_walls(nav, rng, cols / 3, 2 * cols / 3, rows / 2, 6, 1.8));
            obs.extend(place_l_shape(nav, rng, cols / 3, rows / 3, 4, 1, 0, 1.5));
            obs.extend(place_l_shape(nav, rng, 2 * cols / 3, 2 * rows / 3, 4, 1, 2, 1.5));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 3, 3, 1.0));
        }
        1 => {
            obs.extend(place_pillar_grid(nav, rng, cols / 4, rows / 4, 3 * cols / 4, 3 * rows / 4, 5, 2, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 3, rows / 3, cols / 4, true, 3, 1.2));
            obs.extend(place_wall_segment(nav, rng, cols / 2, 2 * rows / 3, cols / 4, true, 2, 1.2));
            obs.extend(place_barricade_line(nav, rng, cols / 4, 3 * cols / 4, rows / 2, 4, 3, 1.0));
        }
        2 => {
            let wall_len_v = rows / 2;
            let wall_len_h = cols / 2;
            obs.extend(place_wall_segment(nav, rng, cols / 4, rows / 4, wall_len_v, false, 3, 1.5));
            obs.extend(place_wall_segment(nav, rng, 3 * cols / 4, rows / 4, wall_len_v, false, 3, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 4, rows / 4, wall_len_h, true, 3, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 4, 3 * rows / 4, wall_len_h, true, 3, 1.5));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 3, 3, 3, 1.0));
            obs.extend(place_cover_cluster(nav, rng, 2 * cols / 3, rows / 2, 3, 3, 1.0));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, 2 * rows / 3, 3, 3, 1.0));
        }
        _ => {
            let wall_len = rows / 3;
            obs.extend(place_wall_segment(nav, rng, cols / 4, rows / 4, wall_len, false, 2, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 2, rows / 3, wall_len, false, 2, 1.5));
            obs.extend(place_wall_segment(nav, rng, 3 * cols / 4, rows / 4, wall_len, false, 2, 1.5));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 2, 3, 2, 1.0));
            obs.extend(place_cover_cluster(nav, rng, 2 * cols / 3, rows / 2, 3, 2, 1.0));
        }
    }
    obs
}

pub(crate) fn generate_recovery_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let template = rng.next_usize_range(0, 2);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_pillar_grid(nav, rng, cols / 3, rows / 3, 2 * cols / 3, 2 * rows / 3, 3, 1, 1.2));
        }
        1 => {
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 2, 4, 1.0));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, 2 * rows / 3, 2, 2, 1.0));
        }
        _ => {
            obs.extend(place_wall_segment(nav, rng, cols / 3, rows / 3, 4, false, 0, 1.2));
            obs.extend(place_wall_segment(nav, rng, 2 * cols / 3, rows / 3, 4, false, 0, 1.2));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 2, 2, 1.0));
        }
    }
    obs
}

pub(crate) fn generate_climax_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let template = rng.next_usize_range(0, 3);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_pillar_grid(nav, rng, cols / 4, rows / 4, 3 * cols / 4, 3 * rows / 4, 4, 1, 1.8));
            obs.extend(place_barricade_line(nav, rng, cols / 3, 2 * cols / 3, rows / 3, 3, 2, 1.2));
            obs.extend(place_barricade_line(nav, rng, cols / 3, 2 * cols / 3, 2 * rows / 3, 3, 2, 1.2));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 2, 3, 2, 1.0));
            obs.extend(place_cover_cluster(nav, rng, 2 * cols / 3, rows / 2, 3, 2, 1.0));
        }
        1 => {
            obs.extend(place_corridor_walls(nav, rng, cols / 4, 3 * cols / 4, rows / 2, 8, 1.8));
            obs.extend(place_l_shape(nav, rng, cols / 4, rows / 4, 4, 1, 0, 1.5));
            obs.extend(place_l_shape(nav, rng, 3 * cols / 4, rows / 4, 4, 1, 1, 1.5));
            obs.extend(place_l_shape(nav, rng, cols / 4, 3 * rows / 4, 4, 1, 3, 1.5));
            obs.extend(place_l_shape(nav, rng, 3 * cols / 4, 3 * rows / 4, 4, 1, 2, 1.5));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 4, 3, 1.0));
        }
        2 => {
            let wall_len = cols / 3;
            obs.extend(place_wall_segment(nav, rng, cols / 4, rows / 3, wall_len, true, 3, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 3, rows / 2, wall_len, true, 3, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 4, 2 * rows / 3, wall_len, true, 3, 1.5));
            obs.extend(place_cover_cluster(nav, rng, cols / 4, rows / 4, 3, 2, 1.0));
            obs.extend(place_cover_cluster(nav, rng, 3 * cols / 4, rows / 4, 3, 2, 1.0));
            obs.extend(place_cover_cluster(nav, rng, cols / 4, 3 * rows / 4, 3, 2, 1.0));
            obs.extend(place_cover_cluster(nav, rng, 3 * cols / 4, 3 * rows / 4, 3, 2, 1.0));
        }
        _ => {
            let wall_len_h = cols / 2;
            let wall_len_v = rows / 2;
            obs.extend(place_wall_segment(nav, rng, cols / 4, rows / 4, wall_len_h, true, 3, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 4, 3 * rows / 4, wall_len_h, true, 3, 1.5));
            obs.extend(place_wall_segment(nav, rng, cols / 4, rows / 4, wall_len_v, false, 3, 1.5));
            obs.extend(place_wall_segment(nav, rng, 3 * cols / 4, rows / 4, wall_len_v, false, 3, 1.5));
            obs.extend(place_pillar_grid(nav, rng, cols / 3, rows / 3, 2 * cols / 3, 2 * rows / 3, 4, 1, 1.5));
        }
    }
    obs
}

/// Fallback: deterministic grid of 1x1 blocks in the centre (always valid).
pub(crate) fn generate_fallback_obstacles(nav: &mut NavGrid, _rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let mut obs = Vec::new();
    let c_lo = cols / 4;
    let c_hi = 3 * cols / 4;
    let r_lo = rows / 4;
    let r_hi = 3 * rows / 4;
    for r in (r_lo..=r_hi).step_by(2) {
        for c in (c_lo..=c_hi).step_by(2) {
            if c > 0 && c < cols - 1 && r > 0 && r < rows - 1 {
                nav.set_walkable_rect(c, r, c, r, false);
                obs.push(ObstacleRegion {
                    col0: c,
                    col1: c,
                    row0: r,
                    row1: r,
                    height: 1.0,
                });
            }
        }
    }
    obs
}
