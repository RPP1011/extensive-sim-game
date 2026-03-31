//! WFC (Wave Function Collapse) solver for 2D tile grids.
//!
//! Generic over tile rules — usable for building interiors, dungeons, etc.

use super::tiles::{TileRule, Tile, compatible, opposite, NORTH, EAST, SOUTH, WEST};
#[cfg(test)]
use super::tiles::Socket;

/// Deterministic LCG random number generator.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        let s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut lcg = Self(s);
        lcg.next(); lcg.next(); // warm up
        lcg
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    /// Returns a value in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Returns a value in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        (self.next() & 0x7fff_ffff) as f32 / 0x8000_0000u32 as f32
    }
}

/// A cell in the WFC grid.
#[derive(Clone)]
struct WfcCell {
    /// Bitset of allowed rule indices. Bit `i` set = rule `i` still possible.
    possible: u64,
    /// Number of set bits (cached).
    count: u8,
    /// Chosen rule index (valid only when count == 1).
    chosen: u8,
}

impl WfcCell {
    fn new(num_rules: usize) -> Self {
        assert!(num_rules <= 64, "WFC supports max 64 tile rules");
        let mask = if num_rules == 64 { u64::MAX } else { (1u64 << num_rules) - 1 };
        WfcCell {
            possible: mask,
            count: num_rules as u8,
            chosen: 0,
        }
    }

    fn is_collapsed(&self) -> bool {
        self.count == 1
    }

    fn collapse_to(&mut self, rule_idx: usize) {
        self.possible = 1u64 << rule_idx;
        self.count = 1;
        self.chosen = rule_idx as u8;
    }

    /// Remove a rule from the possibility set. Returns true if the set changed.
    fn remove(&mut self, rule_idx: usize) -> bool {
        let bit = 1u64 << rule_idx;
        if self.possible & bit != 0 {
            self.possible &= !bit;
            self.count = self.possible.count_ones() as u8;
            if self.count == 1 {
                self.chosen = self.possible.trailing_zeros() as u8;
            }
            true
        } else {
            false
        }
    }
}

/// WFC solver result.
pub struct WfcResult {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Tile>,
}

/// Solve a WFC grid with the given rules and dimensions.
///
/// `boundary_rule`: if Some, pin all edge cells to this rule index.
/// `pins`: list of (col, row, rule_index) to pre-collapse.
///
/// Returns the solved tile grid, or None if all retry attempts fail.
pub fn solve(
    width: usize,
    height: usize,
    rules: &[TileRule],
    seed: u64,
    pins: &[(usize, usize, usize)],
) -> Option<WfcResult> {
    assert!(rules.len() <= 64);

    // Precompute compatibility table: for each (rule_idx, direction),
    // which rule indices are compatible on the neighbor's opposing edge?
    let num_rules = rules.len();
    let mut compat = vec![[0u64; 4]; num_rules];
    for (i, ri) in rules.iter().enumerate() {
        for dir in 0..4 {
            let my_socket = ri.sockets[dir];
            let opp = opposite(dir);
            let mut mask = 0u64;
            for (j, rj) in rules.iter().enumerate() {
                if compatible(my_socket, rj.sockets[opp]) {
                    mask |= 1u64 << j;
                }
            }
            compat[i][dir] = mask;
        }
    }

    // Try up to 5 times with different sub-seeds on contradiction.
    for attempt in 0..5 {
        let attempt_seed = seed.wrapping_add(attempt as u64 * 0xDEAD_BEEF);
        if let Some(result) = solve_attempt(width, height, rules, &compat, attempt_seed, pins) {
            return Some(result);
        }
    }
    None
}

fn solve_attempt(
    width: usize,
    height: usize,
    rules: &[TileRule],
    compat: &[[u64; 4]],
    seed: u64,
    pins: &[(usize, usize, usize)],
) -> Option<WfcResult> {
    let num_rules = rules.len();
    let n = width * height;
    let mut cells: Vec<WfcCell> = vec![WfcCell::new(num_rules); n];
    let mut rng = Lcg::new(seed);

    // Apply pins
    for &(col, row, rule_idx) in pins {
        let idx = row * width + col;
        cells[idx].collapse_to(rule_idx);
        if !propagate(&mut cells, width, height, rules, compat, idx) {
            return None; // contradiction from pins
        }
    }

    // Main loop: collapse minimum-entropy cell, propagate
    loop {
        // Find uncollapsed cell with minimum count (> 1)
        let mut min_count = u8::MAX;
        let mut min_idx = usize::MAX;
        let mut min_entropy = f32::MAX;

        for (i, cell) in cells.iter().enumerate() {
            if cell.is_collapsed() || cell.count == 0 { continue; }
            if cell.count < min_count || (cell.count == min_count && {
                // Tie-break with weighted entropy + noise
                let entropy = weighted_entropy(cell, rules) - rng.next_f32() * 0.001;
                entropy < min_entropy
            }) {
                min_count = cell.count;
                min_entropy = weighted_entropy(&cells[i], rules);
                min_idx = i;
            }
        }

        if min_idx == usize::MAX {
            break; // all collapsed
        }
        if min_count == 0 {
            return None; // contradiction
        }

        // Collapse: pick a rule weighted by weight
        let cell = &cells[min_idx];
        let chosen = pick_weighted(cell.possible, rules, &mut rng);
        cells[min_idx].collapse_to(chosen);

        if !propagate(&mut cells, width, height, rules, compat, min_idx) {
            return None; // contradiction
        }
    }

    // Extract result
    let tiles: Vec<Tile> = cells.iter().map(|c| {
        if c.count == 0 { Tile::Floor } // fallback
        else { rules[c.chosen as usize].tile }
    }).collect();

    Some(WfcResult { width, height, tiles })
}

/// Propagate constraints from a collapsed cell outward.
/// Returns false if a contradiction is found.
fn propagate(
    cells: &mut [WfcCell],
    width: usize,
    height: usize,
    rules: &[TileRule],
    compat: &[[u64; 4]],
    start: usize,
) -> bool {
    let mut stack = vec![start];

    while let Some(idx) = stack.pop() {
        let col = idx % width;
        let row = idx / width;
        let cell_possible = cells[idx].possible;

        // Compute the union of all compatible rules for each direction
        let mut allowed = [u64::MAX; 4];
        let num_rules = rules.len();
        for ri in 0..num_rules {
            if cell_possible & (1u64 << ri) == 0 { continue; }
            // This rule is possible — its neighbors must be compatible
        }
        // Actually: for each direction, the neighbor's possibilities must intersect
        // with the union of compatible rules from all our remaining possibilities.
        for dir in 0..4 {
            let mut union = 0u64;
            for ri in 0..num_rules {
                if cell_possible & (1u64 << ri) != 0 {
                    union |= compat[ri][dir];
                }
            }
            allowed[dir] = union;
        }

        // Apply to neighbors
        let neighbors: [(usize, i32, i32); 4] = [
            (NORTH, 0, -1),
            (EAST, 1, 0),
            (SOUTH, 0, 1),
            (WEST, -1, 0),
        ];

        for &(dir, dc, dr) in &neighbors {
            let nc = col as i32 + dc;
            let nr = row as i32 + dr;
            if nc < 0 || nc >= width as i32 || nr < 0 || nr >= height as i32 { continue; }
            let ni = nr as usize * width + nc as usize;

            let before = cells[ni].possible;
            let constrained = before & allowed[dir];
            if constrained == before { continue; } // no change

            cells[ni].possible = constrained;
            cells[ni].count = constrained.count_ones() as u8;
            if cells[ni].count == 0 { return false; } // contradiction
            if cells[ni].count == 1 {
                cells[ni].chosen = constrained.trailing_zeros() as u8;
            }

            stack.push(ni);
        }
    }

    true
}

/// Pick a rule index from the possibility bitset, weighted by rule weights.
fn pick_weighted(possible: u64, rules: &[TileRule], rng: &mut Lcg) -> usize {
    let mut total_weight = 0.0f32;
    let mut bits = possible;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        total_weight += rules[i].weight;
        bits &= bits - 1; // clear lowest set bit
    }

    let mut target = rng.next_f32() * total_weight;
    bits = possible;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        target -= rules[i].weight;
        if target <= 0.0 { return i; }
        bits &= bits - 1;
    }

    // Fallback: return last possible
    63 - possible.leading_zeros() as usize
}

/// Weighted entropy of a cell (sum of weights of remaining possibilities).
fn weighted_entropy(cell: &WfcCell, rules: &[TileRule]) -> f32 {
    let mut sum = 0.0f32;
    let mut bits = cell.possible;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        sum += rules[i].weight;
        bits &= bits - 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tiles::build_rules;

    #[test]
    fn solve_small_grid() {
        let rules = build_rules();
        // Pin boundaries to wall, solve a 6x6 interior
        let mut pins = Vec::new();
        let w = 6;
        let h = 6;

        // Find wall rule index (all WallFace sockets)
        let wall_idx = rules.iter().position(|r| {
            r.tile == Tile::Wall && r.sockets == [Socket::WallFace; 4]
        }).expect("no all-wall rule");

        // Pin boundary cells to wall
        for x in 0..w {
            pins.push((x, 0, wall_idx));
            pins.push((x, h - 1, wall_idx));
        }
        for y in 1..h-1 {
            pins.push((0, y, wall_idx));
            pins.push((w - 1, y, wall_idx));
        }

        let result = solve(w, h, &rules, 42, &pins);
        assert!(result.is_some(), "WFC failed to solve 6x6 grid");
        let result = result.unwrap();
        assert_eq!(result.tiles.len(), w * h);

        // Verify boundary is all wall
        for x in 0..w {
            assert!(result.tiles[x].is_wall(), "Top boundary not wall at {}", x);
            assert!(result.tiles[(h-1)*w + x].is_wall(), "Bottom boundary not wall at {}", x);
        }
        for y in 0..h {
            assert!(result.tiles[y*w].is_wall(), "Left boundary not wall at {}", y);
            assert!(result.tiles[y*w + w - 1].is_wall(), "Right boundary not wall at {}", y);
        }

        // Interior should have some floor
        let floor_count = result.tiles.iter().filter(|t| t.is_floor()).count();
        assert!(floor_count > 0, "No floor tiles generated");
    }

    #[test]
    fn solve_is_deterministic() {
        let rules = build_rules();
        let mut pins = Vec::new();
        let w = 8;
        let h = 8;
        let wall_idx = rules.iter().position(|r| {
            r.tile == Tile::Wall && r.sockets == [Socket::WallFace; 4]
        }).unwrap();
        for x in 0..w { pins.push((x, 0, wall_idx)); pins.push((x, h-1, wall_idx)); }
        for y in 1..h-1 { pins.push((0, y, wall_idx)); pins.push((w-1, y, wall_idx)); }

        let r1 = solve(w, h, &rules, 12345, &pins).unwrap();
        let r2 = solve(w, h, &rules, 12345, &pins).unwrap();
        assert_eq!(r1.tiles, r2.tiles, "Same seed must produce same result");
    }

    #[test]
    fn different_seeds_differ() {
        let rules = build_rules();
        let mut pins = Vec::new();
        let w = 8;
        let h = 8;
        let wall_idx = rules.iter().position(|r| {
            r.tile == Tile::Wall && r.sockets == [Socket::WallFace; 4]
        }).unwrap();
        for x in 0..w { pins.push((x, 0, wall_idx)); pins.push((x, h-1, wall_idx)); }
        for y in 1..h-1 { pins.push((0, y, wall_idx)); pins.push((w-1, y, wall_idx)); }

        let r1 = solve(w, h, &rules, 111, &pins).unwrap();
        let r2 = solve(w, h, &rules, 222, &pins).unwrap();
        assert_ne!(r1.tiles, r2.tiles, "Different seeds should produce different results");
    }
}
