use super::lcg::{Lcg, ObstacleRegion, OBS_PILLAR};
use super::nav::NavGrid;
use super::primitives::*;

// ---------------------------------------------------------------------------
// Size-independent tactical layout generator
// ---------------------------------------------------------------------------

/// Generate obstacles for any room size using structural layout strategies.
/// All strategies produce coherent tactical geometry with:
/// - Clear spawn margins on left/right edges (handled by caller)
/// - Obstacles concentrated in the central 60% of the room
/// - Connected structures (walls, corridors) rather than scattered dots
/// - Symmetric or near-symmetric layouts for balanced play
pub(crate) fn generate_tactical_obstacles(
    nav: &mut NavGrid,
    rng: &mut Lcg,
) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;

    // Play zone: the central area where obstacles go (between spawn margins)
    let margin = spawn_margin(cols);
    let play_lo = margin;
    let play_hi = cols - margin;
    let play_w = play_hi - play_lo;

    if play_w < 4 || rows < 6 {
        // Room too small for meaningful structure
        return generate_fallback_obstacles(nav, rng);
    }

    // Small/medium rooms only use strategies that work at compact scale
    if play_w < 14 || rows < 16 {
        let strategy = rng.next_usize_range(0, 3);
        return match strategy {
            0 => strategy_barricade_rows(nav, rng, play_lo, play_hi),
            1 => strategy_pillared_hall(nav, rng, play_lo, play_hi),
            2 => strategy_cross_walls(nav, rng, play_lo, play_hi),
            _ => strategy_l_cover(nav, rng, play_lo, play_hi),
        };
    }

    // Pick a layout strategy — weighted toward structural layouts,
    // elevated center only on largest rooms
    let strategy = rng.next_usize_range(0, 9);
    match strategy {
        0 | 1 => strategy_corridor(nav, rng, play_lo, play_hi),
        2 => strategy_pillared_hall(nav, rng, play_lo, play_hi),
        3 | 4 => strategy_cross_walls(nav, rng, play_lo, play_hi),
        5 => strategy_arena(nav, rng, play_lo, play_hi),
        6 => strategy_barricade_rows(nav, rng, play_lo, play_hi),
        7 => strategy_l_cover(nav, rng, play_lo, play_hi),
        8 => strategy_compound(nav, rng, play_lo, play_hi),
        _ => strategy_corridor(nav, rng, play_lo, play_hi),
    }
}

/// Compute spawn margin (columns reserved for spawns on each side).
pub(super) fn spawn_margin(cols: usize) -> usize {
    // ~20% of width on each side, minimum 2, maximum 6
    (cols / 5).clamp(2, 6)
}

// ---------------------------------------------------------------------------
// Layout strategies
// ---------------------------------------------------------------------------

/// Central corridor with parallel walls and a gap — creates a kill zone
/// with flanking routes above and below the corridor.
fn strategy_corridor(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    play_lo: usize,
    play_hi: usize,
) -> Vec<ObstacleRegion> {
    let rows = nav.rows;
    let mid_r = rows / 2;
    let play_mid = (play_lo + play_hi) / 2;
    let play_w = play_hi - play_lo;

    // Gap width scales with room height
    let gap = (rows / 4).clamp(3, 8);
    let wall_len = play_w;

    let mut obs = Vec::new();

    // Top corridor wall
    let top_r = mid_r.saturating_sub(gap / 2 + 1);
    if top_r > 1 {
        obs.extend(place_wall_segment(nav, rng, play_lo, top_r, wall_len, true, 0, 1.8));
    }

    // Bottom corridor wall
    let bot_r = mid_r + gap / 2 + 1;
    if bot_r < rows - 2 {
        obs.extend(place_wall_segment(nav, rng, play_lo, bot_r, wall_len, true, 0, 1.8));
    }

    // Cover blocks inside the corridor for firefight positions
    let cover_count = (play_w / 6).clamp(1, 4);
    for i in 0..cover_count {
        let c = play_lo + (i + 1) * play_w / (cover_count + 1);
        obs.extend(place_cover_cluster(nav, rng, c, mid_r, 1, 2, 1.0));
    }

    // Optional: flanking cover outside the corridor
    if rng.next_usize_range(0, 1) == 0 && top_r > 3 {
        let flank_r = top_r / 2;
        obs.extend(place_barricade_line(nav, rng, play_lo + 2, play_hi - 2, flank_r, 3, 2, 0.8));
    }
    if rng.next_usize_range(0, 1) == 0 && bot_r < rows - 4 {
        let flank_r = (bot_r + rows - 2) / 2;
        obs.extend(place_barricade_line(nav, rng, play_lo + 2, play_hi - 2, flank_r, 3, 2, 0.8));
    }

    obs
}

/// Evenly-spaced pillars in the play zone — breaks sightlines while
/// allowing movement in all directions. Classic arena feel.
fn strategy_pillared_hall(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    play_lo: usize,
    play_hi: usize,
) -> Vec<ObstacleRegion> {
    let rows = nav.rows;
    let play_w = play_hi - play_lo;

    // Spacing scales with room size: larger rooms get wider spacing
    let spacing = (play_w / 3).clamp(4, 8);
    let pillar_size = 2; // primitives enforce 2×2 minimum

    let r_lo = rows / 5;
    let r_hi = 4 * rows / 5;

    place_pillar_grid(nav, rng, play_lo + 1, r_lo, play_hi - 1, r_hi, spacing, pillar_size, 1.5)
}

/// Cross-shaped walls that divide the room into quadrants with gaps —
/// creates 4 sectors connected by doorways.
fn strategy_cross_walls(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    play_lo: usize,
    play_hi: usize,
) -> Vec<ObstacleRegion> {
    let rows = nav.rows;
    let play_w = play_hi - play_lo;
    let play_mid = (play_lo + play_hi) / 2;
    let mid_r = rows / 2;
    let r_lo = rows / 5;
    let r_hi = 4 * rows / 5;

    let mut obs = Vec::new();

    // Horizontal wall spanning play zone with doorway
    let h_gap = rng.next_usize_range(2, 3);
    obs.extend(place_wall_segment(nav, rng, play_lo, mid_r, play_w, true, h_gap, 1.5));

    // Vertical wall spanning most of room height with doorway — must cross the horizontal
    let v_len = r_hi - r_lo;
    let v_gap = rng.next_usize_range(2, 3);
    obs.extend(place_wall_segment(nav, rng, play_mid, r_lo, v_len, false, v_gap, 1.5));

    // Single cover wall in each quadrant (only if room is large enough)
    if play_w >= 12 && rows >= 14 {
        let ql = play_lo + play_w / 4;
        let qr = play_lo + 3 * play_w / 4;
        let qt = r_lo + (mid_r - r_lo) / 2;
        let qb = mid_r + (r_hi - mid_r) / 2;
        obs.extend(place_cover_wall(nav, rng, ql, qt, true, 1.0));
        obs.extend(place_cover_wall(nav, rng, qr, qb, true, 1.0));
    }

    obs
}

/// Circular/rectangular arena with walls enclosing a central fighting area
/// and cover positions inside. Spawns are outside the arena walls.
fn strategy_arena(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    play_lo: usize,
    play_hi: usize,
) -> Vec<ObstacleRegion> {
    let rows = nav.rows;
    let play_w = play_hi - play_lo;
    let arena_margin = (play_w / 6).max(2);

    let a_lo = play_lo + arena_margin;
    let a_hi = play_hi - arena_margin;
    let a_top = (rows / 5).max(2);
    let a_bot = rows - a_top;
    let a_w = a_hi - a_lo;
    let a_h = a_bot - a_top;

    if a_w < 4 || a_h < 4 {
        return strategy_pillared_hall(nav, rng, play_lo, play_hi);
    }

    let mut obs = Vec::new();
    let gap = rng.next_usize_range(2, 4);

    // Top wall with entrance
    obs.extend(place_wall_segment(nav, rng, a_lo, a_top, a_w, true, gap, 1.5));
    // Bottom wall with entrance
    obs.extend(place_wall_segment(nav, rng, a_lo, a_bot, a_w, true, gap, 1.5));
    // Left wall with entrance
    obs.extend(place_wall_segment(nav, rng, a_lo, a_top, a_h, false, gap, 1.5));
    // Right wall with entrance
    obs.extend(place_wall_segment(nav, rng, a_hi, a_top, a_h, false, gap, 1.5));

    // Single cover wall inside the arena for firefight positions
    let mid_c = (a_lo + a_hi) / 2;
    let mid_r = (a_top + a_bot) / 2;
    obs.extend(place_cover_wall(nav, rng, mid_c - 1, mid_r, true, 1.0));

    obs
}

/// Parallel horizontal barricade lines across the room — creates
/// layered defense positions that both teams must advance through.
fn strategy_barricade_rows(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    play_lo: usize,
    play_hi: usize,
) -> Vec<ObstacleRegion> {
    let rows = nav.rows;
    let play_w = play_hi - play_lo;

    // Number of barricade rows scales with room depth
    let n_rows = (rows / 6).clamp(2, 5);
    let seg_len = rng.next_usize_range(2, 4);
    let gap_len = rng.next_usize_range(2, 3);

    let mut obs = Vec::new();

    for i in 0..n_rows {
        let r = (i + 1) * rows / (n_rows + 1);
        if r > 1 && r < rows - 2 {
            // Offset alternate rows for a staggered pattern
            let offset = if i % 2 == 0 { 0 } else { seg_len / 2 };
            let c_start = play_lo + offset;
            let c_end = play_hi.saturating_sub(1);
            if c_start < c_end {
                obs.extend(place_barricade_line(
                    nav, rng, c_start, c_end, r, seg_len, gap_len, 1.2,
                ));
            }
        }
    }

    obs
}

/// Symmetric L-shaped cover blocks creating flanking corridors.
/// One L in each quadrant, mirrored for balanced play.
fn strategy_l_cover(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    play_lo: usize,
    play_hi: usize,
) -> Vec<ObstacleRegion> {
    let rows = nav.rows;
    let play_mid = (play_lo + play_hi) / 2;
    let play_w = play_hi - play_lo;

    let arm = (play_w / 6).clamp(2, 4);

    let mut obs = Vec::new();

    // Top-left L (opens right-down)
    obs.extend(place_l_shape(nav, rng, play_lo + play_w / 4, rows / 3, arm, 1, 0, 1.5));
    // Bottom-right L (opens left-up) — mirror
    obs.extend(place_l_shape(nav, rng, play_lo + 3 * play_w / 4, 2 * rows / 3, arm, 1, 2, 1.5));

    // Optional: secondary pair at different positions
    if play_w > 12 && rows > 12 {
        obs.extend(place_l_shape(nav, rng, play_lo + 3 * play_w / 4, rows / 3, arm, 1, 1, 1.5));
        obs.extend(place_l_shape(nav, rng, play_lo + play_w / 4, 2 * rows / 3, arm, 1, 3, 1.5));
    }

    // Barricade line between the L's for mid-field cover
    obs.extend(place_barricade_line(
        nav, rng, play_lo + 2, play_hi - 2, rows / 2, 2, 2, 1.0,
    ));

    obs
}

/// Elevated central platform with sandbag cover — creates a
/// king-of-the-hill dynamic with height advantage in the center.
fn strategy_elevated_center(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    play_lo: usize,
    play_hi: usize,
) -> Vec<ObstacleRegion> {
    let rows = nav.rows;
    let play_w = play_hi - play_lo;
    let play_mid = (play_lo + play_hi) / 2;
    let mid_r = rows / 2;

    // Platform size scales with room — keep small to leave fight space
    let plat_w = (play_w / 4).clamp(2, 5);
    let plat_h = (rows / 5).clamp(2, 4);
    let plat_c = play_mid - plat_w / 2;
    let plat_r = mid_r - plat_h / 2;

    let mut obs = Vec::new();

    // Elevated platform (walkable with elevation)
    obs.extend(place_elevated_platform(
        nav, rng, plat_c, plat_r, plat_w, plat_h, 1.0,
    ));

    // Sandbag arcs on each side of the platform for approach cover
    let arc_radius = (plat_w / 2).clamp(2, 4);
    let arc_count = (rows / 5).clamp(3, 6);
    obs.extend(place_sandbag_arc(nav, rng, play_lo + play_w / 4, mid_r, arc_radius, arc_count, 0.7));
    obs.extend(place_sandbag_arc(nav, rng, play_lo + 3 * play_w / 4, mid_r, arc_radius, arc_count, 0.7));

    // Small cover blocks near spawn approaches
    if play_w > 10 {
        obs.extend(place_cover_cluster(nav, rng, play_lo + 2, rows / 3, 2, 2, 1.0));
        obs.extend(place_cover_cluster(nav, rng, play_hi - 3, 2 * rows / 3, 2, 2, 1.0));
    }

    obs
}

/// Compound layout: combines 2 compatible smaller strategies for variety.
fn strategy_compound(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    play_lo: usize,
    play_hi: usize,
) -> Vec<ObstacleRegion> {
    let rows = nav.rows;
    let play_w = play_hi - play_lo;
    let mut obs = Vec::new();

    // Corridor walls in center + cover clusters on flanks
    let mid_r = rows / 2;
    let gap = (rows / 4).clamp(3, 6);

    let top_r = mid_r.saturating_sub(gap / 2 + 1);
    let bot_r = mid_r + gap / 2 + 1;

    if top_r > 2 && bot_r < rows - 2 {
        // Short corridor walls (not full width)
        let wall_len = play_w * 2 / 3;
        let wall_start = play_lo + play_w / 6;
        obs.extend(place_wall_segment(nav, rng, wall_start, top_r, wall_len, true, 0, 1.5));
        obs.extend(place_wall_segment(nav, rng, wall_start, bot_r, wall_len, true, 0, 1.5));
    }

    // L-shaped cover at approach angles
    let arm = (play_w / 8).clamp(2, 3);
    obs.extend(place_l_shape(nav, rng, play_lo + play_w / 4, rows / 4, arm, 1, 0, 1.2));
    obs.extend(place_l_shape(nav, rng, play_lo + 3 * play_w / 4, 3 * rows / 4, arm, 1, 2, 1.2));

    obs
}

/// Fallback: deterministic grid of 1x1 blocks in the centre (always valid).
pub(crate) fn generate_fallback_obstacles(nav: &mut NavGrid, _rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let margin = spawn_margin(cols);
    let mut obs = Vec::new();
    let c_lo = margin;
    let c_hi = cols - margin;
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
                    obs_type: OBS_PILLAR,
                });
            }
        }
    }
    obs
}
