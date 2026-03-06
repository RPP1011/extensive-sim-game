use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::ai::core::{sim_vec2, SimVec2};

use super::GridNav;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Node {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapEntry {
    f: f32,
    g: f32,
    node: Node,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .partial_cmp(&self.f)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.g.partial_cmp(&self.g).unwrap_or(Ordering::Equal))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn heuristic(a: Node, b: Node) -> f32 {
    let dx = (a.x - b.x) as f32;
    let dy = (a.y - b.y) as f32;
    (dx * dx + dy * dy).sqrt()
}

fn neighbors(n: Node) -> [Node; 8] {
    [
        Node { x: n.x + 1, y: n.y },
        Node { x: n.x - 1, y: n.y },
        Node { x: n.x, y: n.y + 1 },
        Node { x: n.x, y: n.y - 1 },
        Node {
            x: n.x + 1,
            y: n.y + 1,
        },
        Node {
            x: n.x + 1,
            y: n.y - 1,
        },
        Node {
            x: n.x - 1,
            y: n.y + 1,
        },
        Node {
            x: n.x - 1,
            y: n.y - 1,
        },
    ]
}

/// Check if a straight line between two cells is clear of blocked cells.
fn line_clear(nav: &GridNav, from: (i32, i32), to: (i32, i32)) -> bool {
    let dx = (to.0 - from.0).abs();
    let dy = (to.1 - from.1).abs();
    let sx = if from.0 < to.0 { 1 } else { -1 };
    let sy = if from.1 < to.1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = from.0;
    let mut y = from.1;
    loop {
        if nav.blocked.contains(&(x, y)) {
            return false;
        }
        if x == to.0 && y == to.1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
    true
}

pub fn next_waypoint(nav: &GridNav, from: SimVec2, goal: SimVec2) -> SimVec2 {
    let start_cell = nav.cell_of(from);
    let goal_cell = nav.cell_of(goal);
    if start_cell == goal_cell {
        return goal;
    }

    // Fast path: if no blocked cells exist, skip A* entirely
    if nav.blocked.is_empty() {
        return goal;
    }

    // Fast path: check if straight line is clear (Bresenham-style)
    if line_clear(nav, start_cell, goal_cell) {
        return goal;
    }

    let start = Node {
        x: start_cell.0,
        y: start_cell.1,
    };
    let target = Node {
        x: goal_cell.0,
        y: goal_cell.1,
    };

    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<Node, Node> = HashMap::new();
    let mut g_score: HashMap<Node, f32> = HashMap::new();
    g_score.insert(start, 0.0);
    open.push(HeapEntry {
        f: heuristic(start, target),
        g: 0.0,
        node: start,
    });

    let mut found = None;
    let mut iterations = 0u32;
    const MAX_ITERATIONS: u32 = 500;
    while let Some(current) = open.pop() {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            break;
        }
        if current.node == target {
            found = Some(target);
            break;
        }
        let current_g = *g_score.get(&current.node).unwrap_or(&f32::INFINITY);
        for next in neighbors(current.node) {
            if !nav.walkable(next.x, next.y) {
                continue;
            }
            // Prevent diagonal corner-cutting through blocked cells.
            let dx = next.x - current.node.x;
            let dy = next.y - current.node.y;
            if dx != 0 && dy != 0 {
                let side_a_x = current.node.x + dx;
                let side_a_y = current.node.y;
                let side_b_x = current.node.x;
                let side_b_y = current.node.y + dy;
                if !nav.walkable(side_a_x, side_a_y) || !nav.walkable(side_b_x, side_b_y) {
                    continue;
                }
            }
            let base_step_cost = heuristic(current.node, next);
            let slope_mult = nav.slope_cost_by_cell
                .get(&(next.x, next.y))
                .copied()
                .unwrap_or(1.0);
            let step_cost = base_step_cost * slope_mult;
            let tentative = current_g + step_cost;
            if tentative < *g_score.get(&next).unwrap_or(&f32::INFINITY) {
                came_from.insert(next, current.node);
                g_score.insert(next, tentative);
                open.push(HeapEntry {
                    f: tentative + heuristic(next, target),
                    g: tentative,
                    node: next,
                });
            }
        }
    }

    let Some(mut cur) = found else {
        return from;
    };

    let mut rev_path = vec![cur];
    while let Some(prev) = came_from.get(&cur).copied() {
        cur = prev;
        rev_path.push(cur);
        if cur == start {
            break;
        }
    }
    rev_path.reverse();
    if rev_path.len() >= 2 {
        nav.center_of((rev_path[1].x, rev_path[1].y))
    } else {
        goal
    }
}

pub fn clamp_step_to_walkable(nav: &GridNav, from: SimVec2, to: SimVec2) -> SimVec2 {
    if nav.is_walkable_pos(to) {
        return to;
    }
    // Back off along segment until landing in a walkable cell.
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    for i in (1..=9).rev() {
        let t = i as f32 / 10.0;
        let cand = sim_vec2(from.x + dx * t, from.y + dy * t);
        if nav.is_walkable_pos(cand) {
            return cand;
        }
    }
    from
}

/// Returns a cover factor (0.0 = no cover, up to 1.0 = full cover) for a
/// defender at `defender_pos` against an attacker at `attacker_pos`.
/// A cell counts as cover if there is a blocked cell adjacent to the defender
/// that lies roughly between the defender and attacker.
pub fn cover_factor(nav: &GridNav, defender_pos: SimVec2, attacker_pos: SimVec2) -> f32 {
    let def_cell = nav.cell_of(defender_pos);
    let dx = attacker_pos.x - defender_pos.x;
    let dy = attacker_pos.y - defender_pos.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= f32::EPSILON {
        return 0.0;
    }
    let nx = dx / len;
    let ny = dy / len;

    // Check 8 neighbors for blocking cells that provide cover
    let mut best = 0.0_f32;
    for &(ox, oy) in &[
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ] {
        let neighbor = (def_cell.0 + ox, def_cell.1 + oy);
        if !nav.blocked.contains(&neighbor) {
            continue;
        }
        // Does this blocked cell lie in the direction of the attacker?
        let ox_f = ox as f32;
        let oy_f = oy as f32;
        let neighbor_len = (ox_f * ox_f + oy_f * oy_f).sqrt();
        let dot = (ox_f * nx + oy_f * ny) / neighbor_len;
        // dot > 0 means the cover is between us and the attacker
        if dot > 0.3 {
            // Cardinal neighbors provide better cover than diagonal
            let quality = if ox == 0 || oy == 0 { 0.5 } else { 0.3 };
            best = best.max(quality);
        }
    }
    best
}

/// Returns the elevation difference (defender - attacker). Positive means
/// the defender has high ground.
pub fn elevation_advantage(nav: &GridNav, defender_pos: SimVec2, attacker_pos: SimVec2) -> f32 {
    nav.elevation_at_pos(defender_pos) - nav.elevation_at_pos(attacker_pos)
}

/// Given a desired move target, evaluate nearby positions and pick one that
/// balances reaching the target with terrain advantage (cover + elevation).
/// `enemy_centroid` is the average position of visible enemies — used for
/// cover direction.  Returns the best candidate within `max_step` of `from`.
pub fn terrain_biased_step(
    nav: &GridNav,
    from: SimVec2,
    desired: SimVec2,
    max_step: f32,
    enemy_centroid: Option<SimVec2>,
) -> SimVec2 {
    let base = clamp_step_to_walkable(nav, from, desired);

    // Generate lateral candidates perpendicular to movement direction
    let dx = desired.x - from.x;
    let dy = desired.y - from.y;
    let len = (dx * dx + dy * dy).sqrt();

    let mut candidates = vec![base];
    if len > f32::EPSILON {
        let nx = dx / len;
        let ny = dy / len;
        let lateral = max_step * 0.6;
        // Left and right of movement direction
        let left = sim_vec2(base.x - ny * lateral, base.y + nx * lateral);
        let right = sim_vec2(base.x + ny * lateral, base.y - nx * lateral);
        candidates.push(clamp_step_to_walkable(nav, from, left));
        candidates.push(clamp_step_to_walkable(nav, from, right));
        // Wider lateral options
        let wide = max_step * 0.9;
        let wl = sim_vec2(base.x - ny * wide, base.y + nx * wide);
        let wr = sim_vec2(base.x + ny * wide, base.y - nx * wide);
        candidates.push(clamp_step_to_walkable(nav, from, wl));
        candidates.push(clamp_step_to_walkable(nav, from, wr));
    }

    let mut best = base;
    let mut best_score = f32::NEG_INFINITY;

    for cand in &candidates {
        // Proximity to desired target (want to still make progress)
        let cdx = cand.x - desired.x;
        let cdy = cand.y - desired.y;
        let dist_penalty = (cdx * cdx + cdy * cdy).sqrt() * 80.0;

        // Cover bonus from nearest enemy direction
        let cover_bonus = match enemy_centroid {
            Some(ep) => cover_factor(nav, *cand, ep) * 400.0,
            None => 0.0,
        };

        // Elevation bonus (high ground = damage bonus + harder to hit)
        let elev_bonus = nav.elevation_at_pos(*cand) * 300.0;

        // Chokepoint bonus: count blocked neighbors (more = narrower passage = harder to surround)
        let cell = nav.cell_of(*cand);
        let mut blocked_neighbors = 0i32;
        for &(ox, oy) in &[(1,0),(-1,0),(0,1),(0,-1)] {
            let n = (cell.0 + ox, cell.1 + oy);
            if nav.blocked.contains(&n) {
                blocked_neighbors += 1;
            }
        }
        // 1-2 blocked neighbors = good (chokepoint cover), 3+ = cornered (bad)
        let choke_bonus = match blocked_neighbors {
            1 => 100.0,
            2 => 150.0,
            _ => 0.0,
        };

        let score = -dist_penalty + cover_bonus + elev_bonus + choke_bonus;
        if score > best_score {
            best_score = score;
            best = *cand;
        }
    }
    best
}

pub fn has_line_of_sight(nav: &GridNav, from: SimVec2, to: SimVec2) -> bool {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist <= f32::EPSILON {
        return true;
    }

    let sample_step = (nav.cell_size * 0.45).max(0.05);
    let steps = (dist / sample_step).ceil() as i32;
    if steps <= 1 {
        return true;
    }

    for i in 1..steps {
        let t = i as f32 / steps as f32;
        let probe = sim_vec2(from.x + dx * t, from.y + dy * t);
        let cell = nav.cell_of(probe);
        if nav.blocked.contains(&cell) {
            return false;
        }
    }
    true
}

/// Cast `n_rays` evenly-spaced rays from `origin` and return the distance to
/// the nearest blocked cell in each direction, capped at `max_dist`.
/// Returns a Vec of length `n_rays`.  Ray 0 points in the +X direction;
/// subsequent rays are spaced counter-clockwise.
pub fn raycast_distances(nav: &GridNav, origin: SimVec2, n_rays: usize, max_dist: f32) -> Vec<f32> {
    let step = nav.cell_size * 0.45;
    let max_steps = (max_dist / step).ceil() as i32;

    (0..n_rays)
        .map(|i| {
            let angle = (i as f32) * std::f32::consts::TAU / (n_rays as f32);
            let dx = angle.cos();
            let dy = angle.sin();

            for s in 1..=max_steps {
                let d = s as f32 * step;
                let probe = sim_vec2(origin.x + dx * d, origin.y + dy * d);
                let cell = nav.cell_of(probe);
                if nav.blocked.contains(&cell) {
                    return d;
                }
            }
            max_dist
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elevation_and_slope_metadata_default_and_setters() {
        let mut nav = GridNav::new(-10.0, 10.0, -10.0, 10.0, 1.0);
        let probe = sim_vec2(0.2, -0.4);
        assert_eq!(nav.elevation_at_pos(probe), 0.0);
        assert_eq!(nav.slope_cost_at_pos(probe), 1.0);

        nav.set_elevation_rect(-1.0, 1.0, -1.0, 1.0, 3.5);
        nav.set_slope_cost_rect(-1.0, 1.0, -1.0, 1.0, 1.25);
        assert_eq!(nav.elevation_at_pos(probe), 3.5);
        assert_eq!(nav.slope_cost_at_pos(probe), 1.25);
    }

    #[test]
    fn line_of_sight_detects_blocked_cells() {
        let mut nav = GridNav::new(-10.0, 10.0, -10.0, 10.0, 1.0);
        let a = sim_vec2(-4.0, 0.0);
        let b = sim_vec2(4.0, 0.0);
        assert!(has_line_of_sight(&nav, a, b));
        nav.add_block_rect(-0.5, 0.5, -0.5, 0.5);
        assert!(!has_line_of_sight(&nav, a, b));
        let c = sim_vec2(-4.0, 2.5);
        let d = sim_vec2(4.0, 2.5);
        assert!(has_line_of_sight(&nav, c, d));
    }
}
