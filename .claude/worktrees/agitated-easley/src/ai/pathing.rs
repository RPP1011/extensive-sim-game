use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::ai::core::{sim_vec2, SimVec2};

#[derive(Debug, Clone)]
pub struct GridNav {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub cell_size: f32,
    pub blocked: HashSet<(i32, i32)>,
    pub elevation_by_cell: HashMap<(i32, i32), f32>,
    pub slope_cost_by_cell: HashMap<(i32, i32), f32>,
}

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

impl GridNav {
    pub fn new(min_x: f32, max_x: f32, min_y: f32, max_y: f32, cell_size: f32) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            cell_size,
            blocked: HashSet::new(),
            elevation_by_cell: HashMap::new(),
            slope_cost_by_cell: HashMap::new(),
        }
    }

    pub fn add_block_rect(&mut self, min_x: f32, max_x: f32, min_y: f32, max_y: f32) {
        let (cx0, cy0) = self.cell_of(sim_vec2(min_x, min_y));
        let (cx1, cy1) = self.cell_of(sim_vec2(max_x, max_y));
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.blocked.insert((x, y));
            }
        }
    }

    pub fn carve_rect(&mut self, min_x: f32, max_x: f32, min_y: f32, max_y: f32) {
        let (cx0, cy0) = self.cell_of(sim_vec2(min_x, min_y));
        let (cx1, cy1) = self.cell_of(sim_vec2(max_x, max_y));
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.blocked.remove(&(x, y));
            }
        }
    }

    pub fn set_elevation_rect(
        &mut self,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
        elevation: f32,
    ) {
        let (cx0, cy0) = self.cell_of(sim_vec2(min_x, min_y));
        let (cx1, cy1) = self.cell_of(sim_vec2(max_x, max_y));
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.elevation_by_cell.insert((x, y), elevation);
            }
        }
    }

    #[allow(dead_code)]
    pub fn elevation_at_cell(&self, cell: (i32, i32)) -> f32 {
        self.elevation_by_cell.get(&cell).copied().unwrap_or(0.0)
    }

    #[allow(dead_code)]
    pub fn elevation_at_pos(&self, pos: SimVec2) -> f32 {
        self.elevation_at_cell(self.cell_of(pos))
    }

    pub fn set_slope_cost_rect(
        &mut self,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
        slope_cost_multiplier: f32,
    ) {
        let (cx0, cy0) = self.cell_of(sim_vec2(min_x, min_y));
        let (cx1, cy1) = self.cell_of(sim_vec2(max_x, max_y));
        let x0 = cx0.min(cx1);
        let x1 = cx0.max(cx1);
        let y0 = cy0.min(cy1);
        let y1 = cy0.max(cy1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.slope_cost_by_cell
                    .insert((x, y), slope_cost_multiplier.max(0.1));
            }
        }
    }

    #[allow(dead_code)]
    pub fn slope_cost_at_cell(&self, cell: (i32, i32)) -> f32 {
        self.slope_cost_by_cell.get(&cell).copied().unwrap_or(1.0)
    }

    pub fn slope_cost_at_pos(&self, pos: SimVec2) -> f32 {
        self.slope_cost_at_cell(self.cell_of(pos))
    }

    pub fn cell_of(&self, pos: SimVec2) -> (i32, i32) {
        let cx = ((pos.x - self.min_x) / self.cell_size).floor() as i32;
        let cy = ((pos.y - self.min_y) / self.cell_size).floor() as i32;
        (cx, cy)
    }

    pub fn center_of(&self, cell: (i32, i32)) -> SimVec2 {
        let x = self.min_x + (cell.0 as f32 + 0.5) * self.cell_size;
        let y = self.min_y + (cell.1 as f32 + 0.5) * self.cell_size;
        sim_vec2(x, y)
    }

    fn in_bounds(&self, n: Node) -> bool {
        let p = self.center_of((n.x, n.y));
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }

    fn walkable(&self, n: Node) -> bool {
        self.in_bounds(n) && !self.blocked.contains(&(n.x, n.y))
    }

    pub fn is_walkable_pos(&self, pos: SimVec2) -> bool {
        let c = self.cell_of(pos);
        self.walkable(Node { x: c.0, y: c.1 })
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

pub fn next_waypoint(nav: &GridNav, from: SimVec2, goal: SimVec2) -> SimVec2 {
    let start_cell = nav.cell_of(from);
    let goal_cell = nav.cell_of(goal);
    if start_cell == goal_cell {
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
    while let Some(current) = open.pop() {
        if current.node == target {
            found = Some(target);
            break;
        }
        let current_g = *g_score.get(&current.node).unwrap_or(&f32::INFINITY);
        for next in neighbors(current.node) {
            if !nav.walkable(next) {
                continue;
            }
            // Prevent diagonal corner-cutting through blocked cells.
            let dx = next.x - current.node.x;
            let dy = next.y - current.node.y;
            if dx != 0 && dy != 0 {
                let side_a = Node {
                    x: current.node.x + dx,
                    y: current.node.y,
                };
                let side_b = Node {
                    x: current.node.x,
                    y: current.node.y + dy,
                };
                if !nav.walkable(side_a) || !nav.walkable(side_b) {
                    continue;
                }
            }
            let step_cost = heuristic(current.node, next);
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
