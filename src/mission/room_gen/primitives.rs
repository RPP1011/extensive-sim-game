use super::lcg::{Lcg, ObstacleRegion, OBS_WALL, OBS_PILLAR, OBS_L_SHAPE, OBS_COVER_CLUSTER, OBS_BARRICADE, OBS_SANDBAG, OBS_PLATFORM_EDGE};
use super::nav::NavGrid;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Place a solid rectangular block, respecting interior bounds (avoids perimeter).
fn place_block(
    nav: &mut NavGrid,
    col0: usize,
    row0: usize,
    col1: usize,
    row1: usize,
    height: f32,
    obs_type: u8,
) -> Option<ObstacleRegion> {
    let c0 = col0.max(1);
    let r0 = row0.max(1);
    let c1 = col1.min(nav.cols.saturating_sub(2));
    let r1 = row1.min(nav.rows.saturating_sub(2));
    if c0 > c1 || r0 > r1 {
        return None;
    }
    nav.set_walkable_rect(c0, r0, c1, r1, false);
    Some(ObstacleRegion { col0: c0, col1: c1, row0: r0, row1: r1, height, obs_type })
}

// ---------------------------------------------------------------------------
// Obstacle Primitives
// ---------------------------------------------------------------------------

/// Place a wall segment: 2 cells thick for visual weight.
/// When `gap` > 0, leaves a doorway in the wall.
pub(crate) fn place_wall_segment(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    col: usize,
    row: usize,
    length: usize,
    horizontal: bool,
    gap: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    if length < 2 {
        return regions;
    }

    let effective_gap = if gap > 0 { gap } else { rng.next_usize_range(2, 3) };
    let gap_start = if length > effective_gap + 3 {
        rng.next_usize_range(2, length - effective_gap - 1)
    } else {
        length // skip gap if wall too short
    };

    // Place wall as connected 2-thick segments (not individual cells)
    let mut seg_start: Option<usize> = None;
    for i in 0..=length {
        let in_gap = i >= gap_start && i < gap_start + effective_gap;
        let past_end = i == length;

        if (in_gap || past_end) && seg_start.is_some() {
            // Close current segment
            let start = seg_start.unwrap();
            let end = i.saturating_sub(1);
            if end >= start {
                let (c0, r0, c1, r1) = if horizontal {
                    (col + start, row, col + end, row + 1)
                } else {
                    (col, row + start, col + 1, row + end)
                };
                if let Some(obs) = place_block(nav, c0, r0, c1, r1, height, OBS_WALL) {
                    regions.push(obs);
                }
            }
            seg_start = None;
        } else if !in_gap && !past_end && seg_start.is_none() {
            seg_start = Some(i);
        }
    }

    regions
}

/// Place thick pillars (2×2) in a grid pattern — breaks sightlines.
pub(crate) fn place_pillar_grid(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    col0: usize,
    row0: usize,
    col1: usize,
    row1: usize,
    spacing: usize,
    pillar_size: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let ps = pillar_size.max(2); // minimum 2×2 for visual weight
    let sp = spacing.max(ps + 2); // ensure gaps between pillars

    let mut c = col0;
    while c + ps - 1 <= col1 {
        let mut r = row0;
        while r + ps - 1 <= row1 {
            if let Some(obs) = place_block(nav, c, r, c + ps - 1, r + ps - 1, height, OBS_PILLAR) {
                regions.push(obs);
            }
            r += sp;
        }
        c += sp;
    }
    regions
}

/// Place an L-shaped cover block (2 cells thick for visual weight).
pub(crate) fn place_l_shape(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    anchor_col: usize,
    anchor_row: usize,
    arm_len: usize,
    thickness: usize,
    orientation: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let t = thickness.max(2); // minimum 2 thick
    let a = arm_len.clamp(3, 6);

    // orientation: 0=right-down, 1=left-down, 2=left-up, 3=right-up
    let (hc0, hc1, hr0, hr1) = match orientation {
        0 => (anchor_col, anchor_col + a - 1, anchor_row, anchor_row + t - 1),
        1 => (anchor_col.saturating_sub(a - 1), anchor_col, anchor_row, anchor_row + t - 1),
        2 => (anchor_col.saturating_sub(a - 1), anchor_col, anchor_row.saturating_sub(t - 1), anchor_row),
        _ => (anchor_col, anchor_col + a - 1, anchor_row.saturating_sub(t - 1), anchor_row),
    };
    let (vc0, vc1, vr0, vr1) = match orientation {
        0 => (anchor_col, anchor_col + t - 1, anchor_row, anchor_row + a - 1),
        1 => (anchor_col.saturating_sub(t - 1), anchor_col, anchor_row, anchor_row + a - 1),
        2 => (anchor_col.saturating_sub(t - 1), anchor_col, anchor_row.saturating_sub(a - 1), anchor_row),
        _ => (anchor_col, anchor_col + t - 1, anchor_row.saturating_sub(a - 1), anchor_row),
    };

    if let Some(obs) = place_block(nav, hc0, hr0, hc1, hr1, height, OBS_L_SHAPE) {
        regions.push(obs);
    }
    if let Some(obs) = place_block(nav, vc0, vr0, vc1, vr1, height, OBS_L_SHAPE) {
        regions.push(obs);
    }
    regions
}

/// Place a cover wall: a short, thick wall segment meant for peeking.
/// 3-5 cells long, 2 cells thick. The fundamental cover unit.
pub(crate) fn place_cover_wall(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    col: usize,
    row: usize,
    horizontal: bool,
    height: f32,
) -> Vec<ObstacleRegion> {
    let len = rng.next_usize_range(3, 5);
    let (c1, r1) = if horizontal {
        (col + len - 1, row + 1)
    } else {
        (col + 1, row + len - 1)
    };
    let mut regions = Vec::new();
    if let Some(obs) = place_block(nav, col, row, c1, r1, height, OBS_COVER_CLUSTER) {
        regions.push(obs);
    }
    regions
}

/// Place a cluster of 2-3 short cover walls arranged near a center point.
pub(crate) fn place_cover_cluster(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    centre_col: usize,
    centre_row: usize,
    spread: usize,
    count: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let n = count.clamp(2, 3);
    for _ in 0..n {
        let dc = rng.next_usize_range(0, spread) as isize;
        let dr = rng.next_usize_range(0, spread) as isize;
        let sign_c: isize = if rng.next_u64() % 2 == 0 { 1 } else { -1 };
        let sign_r: isize = if rng.next_u64() % 2 == 0 { 1 } else { -1 };
        let c = (centre_col as isize + sign_c * dc).clamp(1, nav.cols as isize - 3) as usize;
        let r = (centre_row as isize + sign_r * dr).clamp(1, nav.rows as isize - 3) as usize;
        let horizontal = rng.next_u64() % 2 == 0;
        regions.extend(place_cover_wall(nav, rng, c, r, horizontal, height));
    }
    regions
}

/// Place two parallel thick walls forming a corridor lane.
pub(crate) fn place_corridor_walls(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    col_lo: usize,
    col_hi: usize,
    centre_row: usize,
    gap_width: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let half_gap = gap_width / 2;
    let top_row = centre_row.saturating_sub(half_gap + 2);
    let bot_row = centre_row + half_gap;
    let len = col_hi.saturating_sub(col_lo);

    if top_row > 1 && top_row < nav.rows - 3 {
        regions.extend(place_wall_segment(nav, rng, col_lo, top_row, len, true, 0, height));
    }
    if bot_row > 1 && bot_row < nav.rows - 3 {
        regions.extend(place_wall_segment(nav, rng, col_lo, bot_row, len, true, 0, height));
    }
    regions
}

/// Place elevated cover positions: walkable cells with elevation.
pub(crate) fn place_elevated_platform(
    nav: &mut NavGrid,
    _rng: &mut Lcg,
    col0: usize,
    row0: usize,
    width: usize,
    depth: usize,
    elevation: f32,
) -> Vec<ObstacleRegion> {
    let c0 = col0.max(1);
    let r0 = row0.max(1);
    let c1 = (col0 + width - 1).min(nav.cols.saturating_sub(2));
    let r1 = (row0 + depth - 1).min(nav.rows.saturating_sub(2));
    nav.set_elevation_rect(c0, r0, c1, r1, elevation);

    let mut regions = Vec::new();
    // Lip on the right edge (1 cell thick, shorter height)
    if c1 + 1 < nav.cols - 1 {
        if let Some(obs) = place_block(nav, c1 + 1, r0, c1 + 1, r1, 0.6, OBS_PLATFORM_EDGE) {
            regions.push(obs);
        }
    }
    regions
}

/// Place a sandbag arc: short cover walls in a semicircle pattern.
pub(crate) fn place_sandbag_arc(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    centre_col: usize,
    centre_row: usize,
    radius: usize,
    count: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let n = count.clamp(3, 6);
    for i in 0..n {
        let angle = -std::f32::consts::FRAC_PI_2
            + std::f32::consts::PI * (i as f32 / (n - 1).max(1) as f32);
        let dc = (angle.cos() * radius as f32).round() as isize;
        let dr = (angle.sin() * radius as f32).round() as isize;
        let c = (centre_col as isize + dc).clamp(1, nav.cols as isize - 3) as usize;
        let r = (centre_row as isize + dr).clamp(1, nav.rows as isize - 3) as usize;
        // Place a short 2×2 sandbag block instead of a single cell
        if let Some(obs) = place_block(nav, c, r, c + 1, r + 1, height, OBS_SANDBAG) {
            regions.push(obs);
        }
    }
    regions
}

/// Place alternating block/gap segments along a row.
/// Segments are 2 cells thick for visual weight.
pub(crate) fn place_barricade_line(
    nav: &mut NavGrid,
    _rng: &mut Lcg,
    col_lo: usize,
    col_hi: usize,
    row: usize,
    segment_len: usize,
    gap_len: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    if row < 1 || row >= nav.rows.saturating_sub(2) {
        return regions;
    }
    let seg = segment_len.max(3); // minimum 3 cells long for visual weight
    let gap = gap_len.max(2);     // minimum 2 cells gap for movement
    let limit = col_hi.min(nav.cols.saturating_sub(2));
    let mut c = col_lo.max(1);
    let mut placing = true;
    let mut run = 0usize;

    while c <= limit {
        if placing {
            // Start of a segment — place as one block
            let end = (c + seg - 1).min(limit);
            if let Some(obs) = place_block(nav, c, row, end, row + 1, height, OBS_BARRICADE) {
                regions.push(obs);
            }
            c = end + 1;
            placing = false;
            run = 0;
        } else {
            run += 1;
            if run >= gap {
                placing = true;
                run = 0;
            }
            c += 1;
        }
    }
    regions
}
