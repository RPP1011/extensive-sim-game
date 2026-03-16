//! Layout validation: blocked-percentage and connectivity checks.

use super::nav::NavGrid;

/// Check that a layout has 2%-35% blocked interior and player<->enemy connectivity.
pub(crate) fn validate_layout(nav: &NavGrid) -> bool {
    let cols = nav.cols;
    let rows = nav.rows;

    let mut total_interior = 0usize;
    let mut blocked = 0usize;
    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            total_interior += 1;
            if !nav.walkable[r * cols + c] {
                blocked += 1;
            }
        }
    }

    if total_interior == 0 {
        return false;
    }

    let pct = (blocked as f32) / (total_interior as f32);
    if pct < 0.02 || pct > 0.35 {
        return false;
    }

    let player_col = cols / 6;
    let enemy_col = cols - cols / 6;
    let mid_row = rows / 2;

    let start = match find_nearest_walkable(nav, player_col, mid_row) {
        Some(pos) => pos,
        None => return false,
    };
    let goal = match find_nearest_walkable(nav, enemy_col, mid_row) {
        Some(pos) => pos,
        None => return false,
    };

    cells_connected(nav, start.0, start.1, goal.0, goal.1)
}

/// 4-directional BFS flood-fill; returns true if goal is reachable from start.
pub(super) fn cells_connected(
    nav: &NavGrid,
    start_col: usize,
    start_row: usize,
    goal_col: usize,
    goal_row: usize,
) -> bool {
    let cols = nav.cols;
    let rows = nav.rows;
    let mut visited = vec![false; cols * rows];
    let mut queue = std::collections::VecDeque::new();

    let start_idx = start_row * cols + start_col;
    visited[start_idx] = true;
    queue.push_back((start_col, start_row));

    while let Some((c, r)) = queue.pop_front() {
        if c == goal_col && r == goal_row {
            return true;
        }
        for &(dc, dr) in &[(0isize, -1isize), (0, 1), (-1, 0), (1, 0)] {
            let nc = c as isize + dc;
            let nr = r as isize + dr;
            if nc < 0 || nr < 0 || nc >= cols as isize || nr >= rows as isize {
                continue;
            }
            let nc = nc as usize;
            let nr = nr as usize;
            let idx = nr * cols + nc;
            if !visited[idx] && nav.walkable[idx] {
                visited[idx] = true;
                queue.push_back((nc, nr));
            }
        }
    }
    false
}

/// Expanding-square search for the nearest walkable cell from a target.
pub(super) fn find_nearest_walkable(nav: &NavGrid, col: usize, row: usize) -> Option<(usize, usize)> {
    let cols = nav.cols;
    let rows = nav.rows;

    if col < cols && row < rows && nav.walkable[row * cols + col] {
        return Some((col, row));
    }

    for radius in 1..cols.max(rows) {
        let r_min = row.saturating_sub(radius);
        let r_max = (row + radius).min(rows - 1);
        let c_min = col.saturating_sub(radius);
        let c_max = (col + radius).min(cols - 1);

        for r in r_min..=r_max {
            for c in c_min..=c_max {
                if r != r_min && r != r_max && c != c_min && c != c_max {
                    continue;
                }
                if nav.walkable[r * cols + c] {
                    return Some((c, r));
                }
            }
        }
    }
    None
}
