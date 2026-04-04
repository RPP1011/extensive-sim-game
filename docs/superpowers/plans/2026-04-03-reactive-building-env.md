# Reactive Building Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `BuildingEnv` — an rl4burn `Env` that wraps `WorldSim` and exposes reactive building decision points mid-simulation, producing (state, action, reward, next_state) trajectories.

**Architecture:** `BuildingEnv` owns a `WorldSim` and ticks it internally between `step()` calls. Decision points fire on world events (NPC death, monster arrival, resource crisis) or a periodic heartbeat. Actions are `usize` indices into a `(cell, building_type)` grid with a leading no-op. Structural details auto-filled by the existing structural oracle. Reward is Chebyshev scalarization delta between decision points.

**Tech Stack:** Rust, rl4burn-core (`Env`, `Step`, `Space`, `SyncVecEnv`), existing `WorldSim` runtime, building AI oracle/features/scoring modules.

**Spec:** `docs/superpowers/specs/2026-04-03-reactive-building-env-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/world_sim/building_ai/env.rs` (create) | `BuildingEnv` struct, `Env` trait impl, trigger detection, reward computation |
| `src/world_sim/building_ai/env_config.rs` (create) | `EnvConfig`, `CurriculumLevel`, action encoding/decoding helpers |
| `src/world_sim/building_ai/env_obs.rs` (create) | Observation extraction: Track A spatial features → `Vec<f32>` |
| `src/world_sim/building_ai/env_reward.rs` (create) | `ObjectiveScores` snapshot, Chebyshev scalarization, delta computation |
| `src/world_sim/building_ai/mod.rs` (modify) | Add `pub mod env; pub mod env_config; pub mod env_obs; pub mod env_reward;` |
| `src/world_sim/building_ai/env_test.rs` (create) | Integration tests for the env |

---

### Task 1: EnvConfig and Action Encoding

**Files:**
- Create: `src/world_sim/building_ai/env_config.rs`
- Modify: `src/world_sim/building_ai/mod.rs`

- [ ] **Step 1: Write the test file**

Create `src/world_sim/building_ai/env_config.rs` with tests at the bottom:

```rust
use crate::world_sim::state::BuildingType;

/// Building types the agent can place. Excludes Wall/Gate (placed by structural
/// oracle as part of wall circuits) and Treasury (one per settlement, auto-placed).
pub const PLACEABLE_TYPES: &[BuildingType] = &[
    BuildingType::House,
    BuildingType::Longhouse,
    BuildingType::Manor,
    BuildingType::Farm,
    BuildingType::Mine,
    BuildingType::Sawmill,
    BuildingType::Forge,
    BuildingType::Workshop,
    BuildingType::Apothecary,
    BuildingType::Market,
    BuildingType::Warehouse,
    BuildingType::Inn,
    BuildingType::TradePost,
    BuildingType::GuildHall,
    BuildingType::Temple,
    BuildingType::Barracks,
    BuildingType::Watchtower,
    BuildingType::Library,
    BuildingType::CourtHouse,
    BuildingType::Well,
    BuildingType::Tent,
    BuildingType::Camp,
    BuildingType::Shrine,
];

pub const NUM_PLACEABLE_TYPES: usize = PLACEABLE_TYPES.len(); // 23

/// Grid dimensions for the building env. Matches CityGrid default.
pub const GRID_SIZE: usize = 128;
pub const GRID_CELLS: usize = GRID_SIZE * GRID_SIZE; // 16384

/// Total action space: pass (0) + grid_cells * building_types.
pub const NUM_ACTIONS: usize = 1 + GRID_CELLS * NUM_PLACEABLE_TYPES;

/// Decode a raw action index into pass or (cell, building_type).
pub fn decode_action(action: usize) -> ActionChoice {
    if action == 0 {
        ActionChoice::Pass
    } else {
        let idx = action - 1;
        let cell = idx / NUM_PLACEABLE_TYPES;
        let btype = idx % NUM_PLACEABLE_TYPES;
        let col = (cell % GRID_SIZE) as u16;
        let row = (cell / GRID_SIZE) as u16;
        ActionChoice::Place {
            grid_cell: (col, row),
            building_type: PLACEABLE_TYPES[btype],
        }
    }
}

/// Encode a (cell, building_type) into a raw action index.
pub fn encode_action(col: u16, row: u16, building_type: BuildingType) -> usize {
    let btype_idx = PLACEABLE_TYPES.iter().position(|&t| t == building_type);
    match btype_idx {
        Some(bi) => 1 + (row as usize * GRID_SIZE + col as usize) * NUM_PLACEABLE_TYPES + bi,
        None => 0, // non-placeable type → pass
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActionChoice {
    Pass,
    Place {
        grid_cell: (u16, u16),
        building_type: BuildingType,
    },
}

/// Curriculum level controls episode parameters.
#[derive(Debug, Clone)]
pub struct CurriculumLevel {
    pub level: u8,
    pub tick_budget: u64,
    pub heartbeat_interval: u64,
    pub max_actions: usize,
    pub min_severity: f32,
    pub max_severity: f32,
    pub max_challenges: usize,
}

impl CurriculumLevel {
    pub fn level_1() -> Self {
        Self { level: 1, tick_budget: 2000, heartbeat_interval: 200, max_actions: 10, min_severity: 0.3, max_severity: 0.5, max_challenges: 1 }
    }
    pub fn level_2() -> Self {
        Self { level: 2, tick_budget: 5000, heartbeat_interval: 200, max_actions: 20, min_severity: 0.5, max_severity: 0.8, max_challenges: 1 }
    }
    pub fn level_3() -> Self {
        Self { level: 3, tick_budget: 8000, heartbeat_interval: 200, max_actions: 25, min_severity: 0.3, max_severity: 0.9, max_challenges: 3 }
    }
    pub fn level_4() -> Self {
        Self { level: 4, tick_budget: 15000, heartbeat_interval: 200, max_actions: usize::MAX, min_severity: 0.1, max_severity: 1.0, max_challenges: 5 }
    }
}

/// Full env configuration.
#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub curriculum: CurriculumLevel,
    pub seed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_action_is_zero() {
        assert_eq!(decode_action(0), ActionChoice::Pass);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let col = 10u16;
        let row = 20u16;
        let btype = BuildingType::Barracks;
        let encoded = encode_action(col, row, btype);
        assert!(encoded > 0);
        let decoded = decode_action(encoded);
        assert_eq!(decoded, ActionChoice::Place { grid_cell: (col, row), building_type: btype });
    }

    #[test]
    fn all_placeable_types_roundtrip() {
        for (i, &btype) in PLACEABLE_TYPES.iter().enumerate() {
            let encoded = encode_action(0, 0, btype);
            assert_eq!(encoded, 1 + i);
            let decoded = decode_action(encoded);
            assert_eq!(decoded, ActionChoice::Place { grid_cell: (0, 0), building_type: btype });
        }
    }

    #[test]
    fn non_placeable_type_encodes_as_pass() {
        // Treasury is not in PLACEABLE_TYPES
        let encoded = encode_action(5, 5, BuildingType::Treasury);
        assert_eq!(encoded, 0);
    }

    #[test]
    fn max_action_index_valid() {
        let decoded = decode_action(NUM_ACTIONS - 1);
        match decoded {
            ActionChoice::Place { grid_cell: (col, row), building_type } => {
                assert!(col < GRID_SIZE as u16);
                assert!(row < GRID_SIZE as u16);
                assert_eq!(building_type, *PLACEABLE_TYPES.last().unwrap());
            }
            _ => panic!("expected Place"),
        }
    }
}
```

- [ ] **Step 2: Register the module**

Add to `src/world_sim/building_ai/mod.rs`:

```rust
pub mod env_config;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test env_config::tests -- --nocapture`

Expected: All 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/building_ai/env_config.rs src/world_sim/building_ai/mod.rs
git commit -m "feat(building_ai): add env_config with action encoding/decoding and curriculum levels"
```

---

### Task 2: Objective Scores and Chebyshev Reward

**Files:**
- Create: `src/world_sim/building_ai/env_reward.rs`
- Modify: `src/world_sim/building_ai/mod.rs`

- [ ] **Step 1: Write env_reward.rs with snapshot and reward logic**

```rust
use crate::world_sim::state::{EntityKind, WorldState};
use crate::world_sim::building_ai::types::ChallengeCategory;

/// Snapshot of objective scores at a decision point.
#[derive(Debug, Clone, Default)]
pub struct ObjectiveScores {
    pub defense: f32,      // 0-1: friendly NPC survival rate
    pub economy: f32,      // 0-1: normalized stockpile fullness
    pub population: f32,   // 0-1: housing coverage (pop / capacity, clamped)
    pub connectivity: f32, // 0-1: road connectivity ratio
    pub garrison: f32,     // 0-1: military NPC ratio near buildings
    pub spatial: f32,      // 0-1: building spread / grid utilization
}

impl ObjectiveScores {
    /// Snapshot current objective scores from world state.
    pub fn snapshot(state: &WorldState) -> Self {
        let total_npcs = state.entities.iter()
            .filter(|e| e.kind == EntityKind::Npc)
            .count() as f32;
        let alive_npcs = state.entities.iter()
            .filter(|e| e.kind == EntityKind::Npc && e.alive)
            .count() as f32;
        let defense = if total_npcs > 0.0 { alive_npcs / total_npcs } else { 1.0 };

        // Economy: mean stockpile fullness across settlements.
        let economy = if state.settlements.is_empty() {
            0.5
        } else {
            let total: f32 = state.settlements.iter()
                .map(|s| {
                    let sum: f32 = s.stockpile.iter().sum();
                    let cap = s.stockpile.len() as f32 * 100.0; // normalize to 100 per commodity
                    (sum / cap).min(1.0)
                })
                .sum();
            total / state.settlements.len() as f32
        };

        // Population: housing coverage.
        let buildings = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building);
        let housing_cap: f32 = buildings
            .filter_map(|e| e.building.as_ref())
            .map(|b| b.residential_capacity as f32)
            .sum();
        let pop = alive_npcs;
        let population = if housing_cap > 0.0 { (pop / housing_cap).min(1.0) } else { 0.0 };

        // Connectivity: fraction of buildings reachable from first building via road.
        // Simplified: count buildings on road-adjacent cells / total buildings.
        let total_buildings = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .count() as f32;
        let connectivity = if total_buildings > 0.0 { 1.0 } else { 0.0 }; // simplified for now

        // Garrison: fraction of alive NPCs that are combat-capable (level > 2).
        let combat_npcs = state.entities.iter()
            .filter(|e| e.kind == EntityKind::Npc && e.alive && e.level >= 2)
            .count() as f32;
        let garrison = if alive_npcs > 0.0 { (combat_npcs / alive_npcs).min(1.0) } else { 0.0 };

        // Spatial: building count / grid capacity (loose measure of utilization).
        let spatial = (total_buildings / 50.0).min(1.0); // 50 buildings = full utilization

        Self { defense, economy, population, connectivity, garrison, spatial }
    }
}

/// Challenge-category-dependent objective weights (from V2 §6 table).
pub fn category_weights(category: ChallengeCategory) -> [f32; 6] {
    // [defense, economy, population, connectivity, garrison, spatial]
    match category {
        ChallengeCategory::Military     => [0.40, 0.05, 0.10, 0.15, 0.20, 0.10],
        ChallengeCategory::Environmental=> [0.15, 0.10, 0.15, 0.20, 0.05, 0.35],
        ChallengeCategory::Economic     => [0.05, 0.40, 0.10, 0.25, 0.05, 0.15],
        ChallengeCategory::Population   => [0.05, 0.10, 0.40, 0.20, 0.05, 0.20],
        // Compound/other: balanced weights
        _ => [0.20, 0.15, 0.15, 0.20, 0.15, 0.15],
    }
}

/// Chebyshev scalarization: -max_i(w_i * |R_i - R*_i| / R*_i).
/// Lower (more negative) is worse. Higher (toward 0) is better.
pub fn chebyshev_score(scores: &ObjectiveScores, ideal: &ObjectiveScores, weights: &[f32; 6]) -> f32 {
    let s = [scores.defense, scores.economy, scores.population,
             scores.connectivity, scores.garrison, scores.spatial];
    let r = [ideal.defense, ideal.economy, ideal.population,
             ideal.connectivity, ideal.garrison, ideal.spatial];

    let mut worst = 0.0f32;
    for i in 0..6 {
        let ri = r[i].max(0.01); // avoid div-by-zero
        let deviation = weights[i] * (s[i] - ri).abs() / ri;
        worst = worst.max(deviation);
    }
    -worst
}

/// Compute per-step reward as Chebyshev delta between pre and post decision scores.
pub fn compute_reward(
    pre: &ObjectiveScores,
    post: &ObjectiveScores,
    ideal: &ObjectiveScores,
    category: ChallengeCategory,
) -> f32 {
    let w = category_weights(category);
    let score_post = chebyshev_score(post, ideal, &w);
    let score_pre = chebyshev_score(pre, ideal, &w);
    score_post - score_pre
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_scores_zero_reward() {
        let scores = ObjectiveScores {
            defense: 0.8, economy: 0.6, population: 0.7,
            connectivity: 0.9, garrison: 0.5, spatial: 0.4,
        };
        let ideal = ObjectiveScores {
            defense: 1.0, economy: 1.0, population: 1.0,
            connectivity: 1.0, garrison: 1.0, spatial: 1.0,
        };
        let reward = compute_reward(&scores, &scores, &ideal, ChallengeCategory::Military);
        assert!((reward).abs() < 1e-6, "same pre/post should give zero reward, got {}", reward);
    }

    #[test]
    fn improvement_gives_positive_reward() {
        let pre = ObjectiveScores {
            defense: 0.5, economy: 0.5, population: 0.5,
            connectivity: 0.5, garrison: 0.5, spatial: 0.5,
        };
        let post = ObjectiveScores {
            defense: 0.8, economy: 0.5, population: 0.5,
            connectivity: 0.5, garrison: 0.5, spatial: 0.5,
        };
        let ideal = ObjectiveScores {
            defense: 1.0, economy: 1.0, population: 1.0,
            connectivity: 1.0, garrison: 1.0, spatial: 1.0,
        };
        let reward = compute_reward(&pre, &post, &ideal, ChallengeCategory::Military);
        assert!(reward > 0.0, "defense improvement under military challenge should be positive, got {}", reward);
    }

    #[test]
    fn degradation_gives_negative_reward() {
        let pre = ObjectiveScores {
            defense: 0.8, economy: 0.5, population: 0.5,
            connectivity: 0.5, garrison: 0.5, spatial: 0.5,
        };
        let post = ObjectiveScores {
            defense: 0.3, economy: 0.5, population: 0.5,
            connectivity: 0.5, garrison: 0.5, spatial: 0.5,
        };
        let ideal = ObjectiveScores {
            defense: 1.0, economy: 1.0, population: 1.0,
            connectivity: 1.0, garrison: 1.0, spatial: 1.0,
        };
        let reward = compute_reward(&pre, &post, &ideal, ChallengeCategory::Military);
        assert!(reward < 0.0, "defense degradation should be negative, got {}", reward);
    }

    #[test]
    fn chebyshev_at_ideal_is_zero() {
        let ideal = ObjectiveScores {
            defense: 1.0, economy: 1.0, population: 1.0,
            connectivity: 1.0, garrison: 1.0, spatial: 1.0,
        };
        let w = category_weights(ChallengeCategory::Military);
        let score = chebyshev_score(&ideal, &ideal, &w);
        assert!((score).abs() < 1e-6, "at ideal should be 0, got {}", score);
    }
}
```

- [ ] **Step 2: Register the module**

Add to `src/world_sim/building_ai/mod.rs`:

```rust
pub mod env_reward;
```

- [ ] **Step 3: Run tests**

Run: `cargo test env_reward::tests -- --nocapture`

Expected: All 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/building_ai/env_reward.rs src/world_sim/building_ai/mod.rs
git commit -m "feat(building_ai): add Chebyshev reward computation with objective scores"
```

---

### Task 3: Observation Extraction

**Files:**
- Create: `src/world_sim/building_ai/env_obs.rs`
- Modify: `src/world_sim/building_ai/mod.rs`

- [ ] **Step 1: Write env_obs.rs with observation flattening**

This extracts Track A features from `WorldState` into a flat `Vec<f32>`. The initial implementation uses a compact representation: per-cell grid channels + scalar context. Grid channels are downsampled if needed for performance.

```rust
use crate::world_sim::state::{BuildingType, EntityKind, WorldState};
use crate::world_sim::building_ai::env_config::GRID_SIZE;
use crate::world_sim::city_grid::CellState;

/// Number of per-cell channels in the grid observation.
pub const GRID_CHANNELS: usize = 8;

/// Number of scalar context features appended after the grid.
pub const SCALAR_CONTEXT_DIM: usize = 12;

/// Total observation dimension.
pub const OBS_DIM: usize = GRID_SIZE * GRID_SIZE * GRID_CHANNELS + SCALAR_CONTEXT_DIM;

/// Extract observation from world state as a flat Vec<f32>.
///
/// Layout: [grid_channels... (H*W*C, channel-last), scalar_context...]
///
/// Grid channels (per cell):
///   0: terrain elevation (0-1, normalized)
///   1: building presence (0 = empty, 1 = building)
///   2: wall presence (0 or 1)
///   3: road presence (0 or 1)
///   4: building type (0-1, normalized index / NUM_TYPES)
///   5: structural health (0-1)
///   6: friendly NPC density (0-1, clamped count / 5)
///   7: enemy density (0-1, clamped count / 5)
///
/// Scalar context:
///   0: settlement level (0-1, / 5)
///   1-4: resource stockpiles (wood, stone, iron, food, each 0-1)
///   5: population / housing capacity ratio (0-1 clamped)
///   6: tick / tick_budget (0-1)
///   7: seasonal phase (0-1, tick % 4800 / 4800)
///   8: challenge severity (0-1)
///   9: challenge direction x (-1 to 1)
///   10: challenge direction y (-1 to 1)
///   11: alive monster count (0-1, / 50 clamped)
pub fn extract_observation(
    state: &WorldState,
    tick_budget: u64,
    challenge_severity: f32,
    challenge_direction: Option<(f32, f32)>,
) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBS_DIM];

    // --- Grid channels ---
    // Get the city grid for the first settlement (if any).
    let grid_opt = state.settlements.first()
        .and_then(|s| s.city_grid_idx)
        .and_then(|gi| state.city_grids.get(gi));

    if let Some(grid) = grid_opt {
        let rows = grid.rows.min(GRID_SIZE);
        let cols = grid.cols.min(GRID_SIZE);
        for r in 0..rows {
            for c in 0..cols {
                let cell = grid.cell(c, r);
                let base = (r * GRID_SIZE + c) * GRID_CHANNELS;

                // Channel 0: elevation
                obs[base] = (cell.elevation as f32 / 10.0).min(1.0);

                // Channel 1: building presence
                obs[base + 1] = if cell.state == CellState::Building { 1.0 } else { 0.0 };

                // Channel 2: wall presence
                obs[base + 2] = if cell.state == CellState::Wall { 1.0 } else { 0.0 };

                // Channel 3: road presence
                obs[base + 3] = if cell.state == CellState::Road { 1.0 } else { 0.0 };

                // Channel 4: building type (normalized)
                if let Some(bid) = cell.building_id {
                    if let Some(entity) = state.entities.iter().find(|e| e.id == bid) {
                        if let Some(bd) = &entity.building {
                            obs[base + 4] = bd.building_type as u8 as f32 / 25.0;
                        }
                    }
                }

                // Channel 5: structural health
                if let Some(bid) = cell.building_id {
                    if let Some(entity) = state.entities.iter().find(|e| e.id == bid && e.alive) {
                        obs[base + 5] = entity.hp / entity.max_hp.max(1.0);
                    }
                }
            }
        }
    }

    // Channels 6-7: NPC/monster density (scan entities, map to grid cells).
    let settlement_pos = state.settlements.first().map(|s| s.pos).unwrap_or((0.0, 0.0));
    if let Some(grid) = grid_opt {
        for entity in &state.entities {
            if !entity.alive { continue; }
            // Convert world pos to grid cell.
            let (gc, gr) = grid.world_to_grid(entity.pos.0, entity.pos.1, settlement_pos);
            if gc >= GRID_SIZE || gr >= GRID_SIZE { continue; }
            let base = (gr * GRID_SIZE + gc) * GRID_CHANNELS;
            match entity.kind {
                EntityKind::Npc => {
                    obs[base + 6] = (obs[base + 6] + 0.2).min(1.0);
                }
                EntityKind::Monster => {
                    obs[base + 7] = (obs[base + 7] + 0.2).min(1.0);
                }
                _ => {}
            }
        }
    }

    // --- Scalar context ---
    let sc_base = GRID_SIZE * GRID_SIZE * GRID_CHANNELS;

    // 0: settlement level
    let settlement_level = state.settlements.first()
        .map(|s| s.infrastructure_level)
        .unwrap_or(1.0);
    obs[sc_base] = (settlement_level / 5.0).min(1.0);

    // 1-4: resource stockpiles (wood=0, stone=1, iron=2, food=3)
    if let Some(s) = state.settlements.first() {
        for i in 0..4 {
            if i < s.stockpile.len() {
                obs[sc_base + 1 + i] = (s.stockpile[i] / 500.0).min(1.0);
            }
        }
    }

    // 5: population / housing ratio
    let alive_npcs = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Npc && e.alive)
        .count() as f32;
    let housing_cap: f32 = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .filter_map(|e| e.building.as_ref())
        .map(|b| b.residential_capacity as f32)
        .sum();
    obs[sc_base + 5] = if housing_cap > 0.0 { (alive_npcs / housing_cap).min(1.0) } else { 0.0 };

    // 6: tick progress
    obs[sc_base + 6] = if tick_budget > 0 { state.tick as f32 / tick_budget as f32 } else { 0.0 };

    // 7: seasonal phase
    obs[sc_base + 7] = (state.tick % 4800) as f32 / 4800.0;

    // 8: challenge severity
    obs[sc_base + 8] = challenge_severity;

    // 9-10: challenge direction
    let (dx, dy) = challenge_direction.unwrap_or((0.0, 0.0));
    obs[sc_base + 9] = dx;
    obs[sc_base + 10] = dy;

    // 11: alive monster count
    let alive_monsters = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Monster && e.alive)
        .count() as f32;
    obs[sc_base + 11] = (alive_monsters / 50.0).min(1.0);

    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obs_has_correct_length() {
        let state = WorldState::new(42);
        let obs = extract_observation(&state, 5000, 0.5, Some((0.0, -1.0)));
        assert_eq!(obs.len(), OBS_DIM);
    }

    #[test]
    fn obs_values_in_range() {
        let state = WorldState::new(42);
        let obs = extract_observation(&state, 5000, 0.5, Some((0.0, -1.0)));
        for (i, &v) in obs.iter().enumerate() {
            assert!(v >= -1.0 && v <= 1.0,
                "obs[{}] = {} out of range [-1, 1]", i, v);
        }
    }
}
```

- [ ] **Step 2: Check if CityGrid has world_to_grid**

Run: `grep -n "world_to_grid\|fn.*grid.*world" src/world_sim/city_grid.rs | head -5`

If `world_to_grid` doesn't exist, add it as the inverse of `grid_to_world`:

```rust
/// Convert world position to grid cell indices (col, row).
pub fn world_to_grid(&self, wx: f32, wy: f32, settlement_pos: (f32, f32)) -> (usize, usize) {
    let center_col = self.cols / 2;
    let center_row = self.rows / 2;
    let col = ((wx - settlement_pos.0) / self.cell_size as f32 + center_col as f32).round() as isize;
    let row = ((wy - settlement_pos.1) / self.cell_size as f32 + center_row as f32).round() as isize;
    (col.max(0) as usize, row.max(0) as usize)
}
```

- [ ] **Step 3: Register the module**

Add to `src/world_sim/building_ai/mod.rs`:

```rust
pub mod env_obs;
```

- [ ] **Step 4: Run tests**

Run: `cargo test env_obs::tests -- --nocapture`

Expected: Both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/building_ai/env_obs.rs src/world_sim/building_ai/mod.rs
git commit -m "feat(building_ai): add Track A observation extraction to Vec<f32>"
```

---

### Task 4: Trigger Detection

**Files:**
- Create: `src/world_sim/building_ai/env.rs` (partial — trigger detection only)
- Modify: `src/world_sim/building_ai/mod.rs`

- [ ] **Step 1: Write trigger detection module**

Create `src/world_sim/building_ai/env.rs` with trigger detection logic. The full `Env` impl comes in Task 5.

```rust
use crate::world_sim::state::{EntityKind, WorldEvent, WorldState};
use crate::world_sim::NUM_COMMODITIES;

/// Reasons a decision point was triggered.
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerReason {
    /// A friendly NPC died.
    NpcDeath { entity_id: u32 },
    /// A battle started on the settlement grid.
    BattleStarted { grid_id: u32 },
    /// A building's health dropped below 50%.
    BuildingDamaged { entity_id: u32 },
    /// A resource stockpile dropped below 20% capacity.
    ResourceCrisis { commodity: usize },
    /// Population exceeded housing capacity by >20%.
    HousingOverflow,
    /// Periodic heartbeat — no specific event.
    Heartbeat,
}

/// Check world events added since `last_event_idx` for triggers.
/// Returns the first trigger found, or None.
pub fn check_event_triggers(
    state: &WorldState,
    last_event_idx: usize,
    settlement_grid_id: Option<u32>,
) -> Option<TriggerReason> {
    for event in state.world_events.iter().skip(last_event_idx) {
        match event {
            WorldEvent::EntityDied { entity_id, .. } => {
                // Check if this was an NPC.
                if let Some(e) = state.entities.iter().find(|e| e.id == *entity_id) {
                    if e.kind == EntityKind::Npc {
                        return Some(TriggerReason::NpcDeath { entity_id: *entity_id });
                    }
                }
            }
            WorldEvent::BattleStarted { grid_id, .. } => {
                // Only trigger if it's on our settlement's grid.
                if settlement_grid_id.map_or(false, |sg| sg == *grid_id) {
                    return Some(TriggerReason::BattleStarted { grid_id: *grid_id });
                }
            }
            _ => {}
        }
    }
    None
}

/// Check state-based triggers (not event-driven).
pub fn check_state_triggers(state: &WorldState) -> Option<TriggerReason> {
    // Building health < 50%
    for e in &state.entities {
        if e.alive && e.kind == EntityKind::Building && e.hp < e.max_hp * 0.5 && e.max_hp > 0.0 {
            return Some(TriggerReason::BuildingDamaged { entity_id: e.id });
        }
    }

    // Resource crisis: any commodity below 20% (using 100 as nominal capacity).
    if let Some(s) = state.settlements.first() {
        for i in 0..NUM_COMMODITIES.min(s.stockpile.len()) {
            if s.stockpile[i] < 20.0 && s.stockpile[i] >= 0.0 {
                return Some(TriggerReason::ResourceCrisis { commodity: i });
            }
        }
    }

    // Housing overflow: population > 1.2 * housing capacity.
    let alive_npcs = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Npc && e.alive)
        .count() as f32;
    let housing_cap: f32 = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .filter_map(|e| e.building.as_ref())
        .map(|b| b.residential_capacity as f32)
        .sum();
    if housing_cap > 0.0 && alive_npcs > housing_cap * 1.2 {
        return Some(TriggerReason::HousingOverflow);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_is_a_variant() {
        let trigger = TriggerReason::Heartbeat;
        assert_eq!(trigger, TriggerReason::Heartbeat);
    }

    #[test]
    fn no_triggers_on_empty_state() {
        let state = WorldState::new(42);
        assert!(check_event_triggers(&state, 0, None).is_none());
        assert!(check_state_triggers(&state).is_none());
    }
}
```

- [ ] **Step 2: Register the module**

Add to `src/world_sim/building_ai/mod.rs`:

```rust
pub mod env;
```

- [ ] **Step 3: Run tests**

Run: `cargo test building_ai::env::tests -- --nocapture`

Expected: Both tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/building_ai/env.rs src/world_sim/building_ai/mod.rs
git commit -m "feat(building_ai): add decision point trigger detection"
```

---

### Task 5: BuildingEnv Core — Env Trait Implementation

**Files:**
- Modify: `src/world_sim/building_ai/env.rs` (add BuildingEnv struct and Env impl)

This is the largest task. It adds the `BuildingEnv` struct and implements the rl4burn `Env` trait.

- [ ] **Step 1: Add BuildingEnv struct and constructor**

Add to `src/world_sim/building_ai/env.rs`, after the trigger detection code:

```rust
use rl4burn::env::{Env, Step};
use rl4burn::env::space::Space;
use crate::world_sim::city_grid::CellState;
use crate::world_sim::runtime::WorldSim;
use crate::world_sim::building_ai::env_config::*;
use crate::world_sim::building_ai::env_obs;
use crate::world_sim::building_ai::env_reward::{self, ObjectiveScores};
use crate::world_sim::building_ai::mass_gen;
use crate::world_sim::building_ai::types::{ChallengeCategory, DecisionTier};
use crate::world_sim::building_ai::oracle::{strategic_oracle, structural_oracle};
use crate::world_sim::building_ai::scoring;
use crate::world_sim::building_ai::scenario_gen;
use crate::world_sim::building_ai::features::compute_spatial_features;

pub struct BuildingEnv {
    sim: WorldSim,
    curriculum: CurriculumLevel,
    seed: u64,
    // Episode state
    challenge_category: ChallengeCategory,
    challenge_severity: f32,
    challenge_direction: Option<(f32, f32)>,
    actions_taken: usize,
    ticks_since_decision: u64,
    last_event_idx: usize,
    pre_decision_scores: ObjectiveScores,
    ideal_scores: ObjectiveScores,
    settlement_grid_id: Option<u32>,
    episode_count: u64,
}

impl BuildingEnv {
    pub fn new(config: EnvConfig) -> Self {
        let mut env = Self {
            sim: WorldSim::new(WorldState::new(config.seed)),
            curriculum: config.curriculum,
            seed: config.seed,
            challenge_category: ChallengeCategory::Military,
            challenge_severity: 0.5,
            challenge_direction: None,
            actions_taken: 0,
            ticks_since_decision: 0,
            last_event_idx: 0,
            pre_decision_scores: ObjectiveScores::default(),
            ideal_scores: ObjectiveScores {
                defense: 1.0, economy: 1.0, population: 1.0,
                connectivity: 1.0, garrison: 1.0, spatial: 1.0,
            },
            settlement_grid_id: None,
            episode_count: 0,
        };
        // Initialize with a proper reset.
        let _ = env.reset_internal();
        env
    }

    /// Internal reset: compose new scenario, tick to first decision point.
    fn reset_internal(&mut self) -> Vec<f32> {
        self.episode_count += 1;
        let ep_seed = self.seed.wrapping_add(self.episode_count * 0x9E3779B97F4A7C15);

        // Compose world state from randomized axes.
        let mut rng = mass_gen::SimpleRng::new(ep_seed);
        let terrain = mass_gen::TERRAIN_TYPES[rng.next_u32() as usize % mass_gen::TERRAIN_TYPES.len()];
        let maturity = mass_gen::MATURITY_LEVELS[rng.next_u32() as usize % mass_gen::MATURITY_LEVELS.len()];
        let resources = mass_gen::RESOURCE_PROFILES[rng.next_u32() as usize % mass_gen::RESOURCE_PROFILES.len()];
        let npcs = mass_gen::NPC_COMPOSITIONS[rng.next_u32() as usize % mass_gen::NPC_COMPOSITIONS.len()];
        let quality = mass_gen::BUILDING_QUALITIES[rng.next_u32() as usize % mass_gen::BUILDING_QUALITIES.len()];

        let mut state = mass_gen::compose_world_state(terrain, maturity, resources, npcs, quality, ep_seed);
        state.skip_resource_init = true;

        // Inject challenge based on curriculum.
        let severity = self.curriculum.min_severity
            + rng.next_f32() * (self.curriculum.max_severity - self.curriculum.min_severity);
        let cat_idx = rng.next_u32() as usize % 10;
        let category = match cat_idx {
            0 => ChallengeCategory::Military,
            1 => ChallengeCategory::Environmental,
            2 => ChallengeCategory::Economic,
            3 => ChallengeCategory::Population,
            4 => ChallengeCategory::Temporal,
            5 => ChallengeCategory::Terrain,
            6 => ChallengeCategory::MultiSettlement,
            7 => ChallengeCategory::UnitCapability,
            8 => ChallengeCategory::HighValueNpc,
            _ => ChallengeCategory::LevelScaled,
        };
        // Direction: random unit vector for directional challenges.
        let angle = rng.next_f32() * std::f32::consts::TAU;
        let direction = Some((angle.cos(), angle.sin()));

        self.challenge_category = category;
        self.challenge_severity = severity;
        self.challenge_direction = direction;
        self.actions_taken = 0;
        self.ticks_since_decision = 0;
        self.last_event_idx = 0;

        // Init sim.
        self.sim = WorldSim::new(state);
        self.settlement_grid_id = self.sim.state().settlements.first()
            .and_then(|s| s.grid_id);

        // Tick to first decision point or heartbeat.
        self.tick_to_decision_point();

        // Snapshot scores.
        self.pre_decision_scores = ObjectiveScores::snapshot(self.sim.state());

        self.observe()
    }

    /// Build observation vector from current state.
    fn observe(&self) -> Vec<f32> {
        env_obs::extract_observation(
            self.sim.state(),
            self.curriculum.tick_budget,
            self.challenge_severity,
            self.challenge_direction,
        )
    }

    /// Tick sim forward until a trigger fires, heartbeat reached, or episode ends.
    /// Returns (terminated, truncated).
    fn tick_to_decision_point(&mut self) -> (bool, bool) {
        loop {
            self.sim.tick();
            self.ticks_since_decision += 1;

            // Check event triggers.
            let state = self.sim.state();
            if let Some(_trigger) = check_event_triggers(state, self.last_event_idx, self.settlement_grid_id) {
                self.last_event_idx = state.world_events.len();
                return (false, false);
            }
            self.last_event_idx = state.world_events.len();

            // Check state triggers.
            if check_state_triggers(state).is_some() {
                return (false, false);
            }

            // Heartbeat.
            if self.ticks_since_decision >= self.curriculum.heartbeat_interval {
                self.ticks_since_decision = 0;
                return (false, false);
            }

            // Episode end conditions.
            if state.tick >= self.curriculum.tick_budget {
                return (false, true); // truncated
            }

            // Challenge resolved: no alive monsters and tick > 100 (give time to spawn).
            if state.tick > 100 {
                let alive_monsters = state.entities.iter()
                    .filter(|e| e.kind == EntityKind::Monster && e.alive)
                    .count();
                if alive_monsters == 0 && matches!(self.challenge_category, ChallengeCategory::Military) {
                    return (true, false); // terminated
                }
            }
        }
    }

    /// Apply a strategic building placement action to the sim state.
    fn apply_placement(&mut self, grid_cell: (u16, u16), building_type: crate::world_sim::state::BuildingType) {
        use crate::world_sim::building_ai::types::{BuildingAction, ActionPayload, DecisionType};

        let strategic_action = BuildingAction {
            decision_type: DecisionType::Placement,
            tier: DecisionTier::Strategic,
            action: ActionPayload::PlaceBuilding { building_type, grid_cell },
            priority: 1.0,
            reasoning_tag: 0,
        };

        // Auto-fill structural details via structural oracle.
        let settlement_id = self.sim.state().settlements.first().map(|s| s.id).unwrap_or(1);
        let spatial = compute_spatial_features(self.sim.state(), settlement_id);
        let challenges = vec![]; // structural oracle doesn't heavily depend on challenges
        let memory = crate::world_sim::building_ai::types::ConstructionMemory::new();
        let obs = scenario_gen::build_observation(
            self.sim.state(), settlement_id, &challenges, &memory, &spatial, DecisionTier::Structural,
        );
        let structural_actions = structural_oracle(&obs, &[strategic_action.clone()]);

        // Apply all actions.
        let mut all_actions = vec![strategic_action];
        all_actions.extend(structural_actions);
        scoring::apply_actions(self.sim.state_mut(), &all_actions);
        self.sim.state_mut().rebuild_all_indices();
    }
}
```

- [ ] **Step 2: Add Env trait implementation**

Append to `src/world_sim/building_ai/env.rs`:

```rust
impl Env for BuildingEnv {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        self.reset_internal()
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        // Decode and apply action.
        match decode_action(action) {
            ActionChoice::Pass => {
                // No-op: just advance to next decision point.
            }
            ActionChoice::Place { grid_cell, building_type } => {
                self.apply_placement(grid_cell, building_type);
                self.actions_taken += 1;
            }
        }

        // Advance sim to next decision point.
        self.ticks_since_decision = 0;
        let (terminated, mut truncated) = self.tick_to_decision_point();

        // Check action budget.
        if self.actions_taken >= self.curriculum.max_actions {
            truncated = true;
        }

        // Compute reward.
        let post_scores = ObjectiveScores::snapshot(self.sim.state());
        let reward = env_reward::compute_reward(
            &self.pre_decision_scores,
            &post_scores,
            &self.ideal_scores,
            self.challenge_category,
        );
        self.pre_decision_scores = post_scores;

        let observation = self.observe();
        Step { observation, reward, terminated, truncated }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0; env_obs::OBS_DIM],
            high: vec![1.0; env_obs::OBS_DIM],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(NUM_ACTIONS)
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        let mut mask = vec![0.0f32; NUM_ACTIONS];

        // Pass is always valid.
        mask[0] = 1.0;

        // Check each (cell, building_type) for validity.
        let grid_opt = self.sim.state().settlements.first()
            .and_then(|s| s.city_grid_idx)
            .and_then(|gi| self.sim.state().city_grids.get(gi));

        if let Some(grid) = grid_opt {
            let rows = grid.rows.min(GRID_SIZE);
            let cols = grid.cols.min(GRID_SIZE);
            for r in 0..rows {
                for c in 0..cols {
                    let cell = grid.cell(c, r);
                    // Only allow placement on empty cells.
                    if cell.state != CellState::Empty { continue; }
                    for (bi, _btype) in PLACEABLE_TYPES.iter().enumerate() {
                        let action_idx = 1 + (r * GRID_SIZE + c) * NUM_PLACEABLE_TYPES + bi;
                        if action_idx < NUM_ACTIONS {
                            mask[action_idx] = 1.0;
                        }
                    }
                }
            }
        }

        Some(mask)
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

Expected: Build succeeds. If `SimpleRng` is private in mass_gen, make it `pub(crate)`. If the axis constant arrays (TERRAIN_TYPES, etc.) don't exist, create them in mass_gen.rs as:

```rust
pub const TERRAIN_TYPES: &[TerrainType] = &[
    TerrainType::FlatOpen, TerrainType::RiverBisect, TerrainType::Hillside,
    TerrainType::CliffEdge, TerrainType::Coastal, TerrainType::Swamp,
    TerrainType::ForestClearing, TerrainType::MountainPass,
];
// Similarly for MATURITY_LEVELS, RESOURCE_PROFILES, NPC_COMPOSITIONS, BUILDING_QUALITIES.
```

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/building_ai/env.rs
git commit -m "feat(building_ai): implement BuildingEnv with rl4burn Env trait"
```

---

### Task 6: Integration Tests

**Files:**
- Create: `src/world_sim/building_ai/env_test.rs`
- Modify: `src/world_sim/building_ai/mod.rs`

- [ ] **Step 1: Write integration tests**

```rust
#[cfg(test)]
mod tests {
    use crate::world_sim::building_ai::env::BuildingEnv;
    use crate::world_sim::building_ai::env_config::*;
    use rl4burn::env::{Env, Step};
    use rl4burn::env::space::Space;

    fn make_env() -> BuildingEnv {
        let config = EnvConfig {
            curriculum: CurriculumLevel::level_1(),
            seed: 12345,
        };
        BuildingEnv::new(config)
    }

    #[test]
    fn reset_returns_correct_obs_size() {
        let mut env = make_env();
        let obs = env.reset();
        let expected = match env.observation_space() {
            Space::Box { low, .. } => low.len(),
            _ => panic!("expected Box space"),
        };
        assert_eq!(obs.len(), expected);
    }

    #[test]
    fn action_space_is_discrete() {
        let env = make_env();
        match env.action_space() {
            Space::Discrete(n) => assert_eq!(n, NUM_ACTIONS),
            _ => panic!("expected Discrete space"),
        }
    }

    #[test]
    fn pass_action_does_not_crash() {
        let mut env = make_env();
        env.reset();
        let step = env.step(0); // pass
        assert_eq!(step.observation.len(), crate::world_sim::building_ai::env_obs::OBS_DIM);
    }

    #[test]
    fn episode_terminates_or_truncates() {
        let mut env = make_env();
        env.reset();
        let mut steps = 0;
        loop {
            let step = env.step(0); // always pass
            steps += 1;
            if step.terminated || step.truncated {
                break;
            }
            if steps > 200 {
                panic!("episode did not end after 200 steps (heartbeat=200 ticks, budget=2000 → expected ~10 steps)");
            }
        }
        assert!(steps >= 1, "episode should have at least 1 step");
    }

    #[test]
    fn action_mask_has_pass_enabled() {
        let mut env = make_env();
        env.reset();
        let mask = env.action_mask().expect("should return action mask");
        assert_eq!(mask.len(), NUM_ACTIONS);
        assert_eq!(mask[0], 1.0, "pass action should always be valid");
    }

    #[test]
    fn valid_placement_accepted() {
        let mut env = make_env();
        env.reset();
        // Find any valid action from the mask.
        let mask = env.action_mask().unwrap();
        let valid_action = mask.iter().enumerate()
            .skip(1) // skip pass
            .find(|(_, &v)| v > 0.0)
            .map(|(i, _)| i);
        if let Some(action) = valid_action {
            let step = env.step(action);
            // Should not crash; obs should be valid.
            assert_eq!(step.observation.len(), crate::world_sim::building_ai::env_obs::OBS_DIM);
        }
        // If no valid placement exists, that's fine — scenario might be full.
    }

    #[test]
    fn different_seeds_different_episodes() {
        let mut env1 = BuildingEnv::new(EnvConfig {
            curriculum: CurriculumLevel::level_1(),
            seed: 111,
        });
        let mut env2 = BuildingEnv::new(EnvConfig {
            curriculum: CurriculumLevel::level_1(),
            seed: 222,
        });
        let obs1 = env1.reset();
        let obs2 = env2.reset();
        // Observations should differ (different scenarios).
        assert_ne!(obs1, obs2, "different seeds should produce different observations");
    }

    #[test]
    fn reset_produces_new_episode() {
        let mut env = make_env();
        let obs1 = env.reset();
        let obs2 = env.reset();
        // Two resets should produce different episodes (different episode_count).
        assert_ne!(obs1, obs2, "consecutive resets should produce different episodes");
    }
}
```

- [ ] **Step 2: Register test module**

Add to `src/world_sim/building_ai/mod.rs`:

```rust
#[cfg(test)]
mod env_test;
```

- [ ] **Step 3: Run all tests**

Run: `cargo test building_ai::env_test -- --nocapture --test-threads=1`

Expected: All 7 tests pass. If any test times out (episode_terminates_or_truncates), check that tick_budget is being respected in tick_to_decision_point.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/building_ai/env_test.rs src/world_sim/building_ai/mod.rs
git commit -m "test(building_ai): add BuildingEnv integration tests"
```

---

### Task 7: Expose Axis Constants in mass_gen

**Files:**
- Modify: `src/world_sim/building_ai/mass_gen.rs`

Task 5's `reset_internal` needs to sample from the axis enum variants. If these aren't already exposed as public constant arrays, add them.

- [ ] **Step 1: Check if constants exist**

Run: `grep -n "TERRAIN_TYPES\|MATURITY_LEVELS\|RESOURCE_PROFILES\|NPC_COMPOSITIONS\|BUILDING_QUALITIES" src/world_sim/building_ai/mass_gen.rs | head -10`

- [ ] **Step 2: Add constants if missing**

If they don't exist, add near the top of `mass_gen.rs` after the enum definitions:

```rust
pub const TERRAIN_TYPES: &[TerrainType] = &[
    TerrainType::FlatOpen, TerrainType::RiverBisect, TerrainType::Hillside,
    TerrainType::CliffEdge, TerrainType::Coastal, TerrainType::Swamp,
    TerrainType::ForestClearing, TerrainType::MountainPass,
];

pub const MATURITY_LEVELS: &[MaturityLevel] = &[
    MaturityLevel::Empty, MaturityLevel::Sparse, MaturityLevel::Moderate,
    MaturityLevel::Dense, MaturityLevel::Overgrown,
];

pub const RESOURCE_PROFILES: &[ResourceProfile] = &[
    ResourceProfile::Abundant, ResourceProfile::Mixed, ResourceProfile::Scarce,
    ResourceProfile::Specialized, ResourceProfile::Depleting,
];

pub const NPC_COMPOSITIONS: &[NpcComposition] = &[
    NpcComposition::MilitaryHeavy, NpcComposition::CivilianHeavy,
    NpcComposition::Balanced, NpcComposition::EliteFew,
    NpcComposition::LargeLowLevel, NpcComposition::Specialist,
];

pub const BUILDING_QUALITIES: &[BuildingQuality] = &[
    BuildingQuality::WellPlanned, BuildingQuality::OrganicGrowth,
    BuildingQuality::BattleDamaged, BuildingQuality::UnderConstruction,
    BuildingQuality::AbandonedDecayed,
];
```

Also make `SimpleRng` accessible:

```rust
pub(crate) struct SimpleRng(u64);
```

(Change from `struct SimpleRng` to `pub(crate) struct SimpleRng` and make `new`, `next_u32`, `next_f32` pub(crate).)

- [ ] **Step 3: Verify build**

Run: `cargo build 2>&1 | tail -5`

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/building_ai/mass_gen.rs
git commit -m "refactor(building_ai): expose axis constants and SimpleRng for env"
```

---

### Task 8: world_to_grid on CityGrid

**Files:**
- Modify: `src/world_sim/city_grid.rs`

- [ ] **Step 1: Check if world_to_grid already exists**

Run: `grep -n "world_to_grid" src/world_sim/city_grid.rs`

- [ ] **Step 2: Add world_to_grid if missing**

Find the `grid_to_world` method and add the inverse right after it:

```rust
/// Convert world position to grid cell indices (col, row).
/// Inverse of `grid_to_world`. Returns (col, row) clamped to grid bounds.
pub fn world_to_grid(&self, wx: f32, wy: f32, settlement_pos: (f32, f32)) -> (usize, usize) {
    let center_col = self.cols / 2;
    let center_row = self.rows / 2;
    let col = ((wx - settlement_pos.0) / self.cell_size as f32 + center_col as f32).round() as isize;
    let row = ((wy - settlement_pos.1) / self.cell_size as f32 + center_row as f32).round() as isize;
    (col.clamp(0, self.cols as isize - 1) as usize, row.clamp(0, self.rows as isize - 1) as usize)
}
```

- [ ] **Step 3: Test roundtrip**

Add test to verify `world_to_grid(grid_to_world(c, r, p), p) == (c, r)`:

```rust
#[test]
fn grid_world_roundtrip() {
    let grid = CityGrid::new(128, 128, 2);
    let spos = (10.0, 20.0);
    for c in [0, 32, 64, 100, 127] {
        for r in [0, 32, 64, 100, 127] {
            let (wx, wy) = grid.grid_to_world(c, r, spos);
            let (rc, rr) = grid.world_to_grid(wx, wy, spos);
            assert_eq!((rc, rr), (c, r), "roundtrip failed for ({}, {})", c, r);
        }
    }
}
```

Run: `cargo test grid_world_roundtrip`

Expected: Pass.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/city_grid.rs
git commit -m "feat(city_grid): add world_to_grid inverse conversion"
```

---

### Task 9: End-to-End Smoke Test

**Files:**
- Modify: `src/world_sim/building_ai/env_test.rs`

- [ ] **Step 1: Add smoke test that runs a full episode with random valid actions**

Add to the existing test module:

```rust
    #[test]
    fn full_episode_with_random_actions() {
        let mut env = make_env();
        env.reset();
        let mut total_reward = 0.0f32;
        let mut steps = 0;
        let mut placements = 0;

        loop {
            // Pick a random valid action from the mask.
            let mask = env.action_mask().unwrap();
            let valid: Vec<usize> = mask.iter().enumerate()
                .filter(|(_, &v)| v > 0.0)
                .map(|(i, _)| i)
                .collect();

            // Alternate between pass and placement.
            let action = if steps % 3 == 0 && valid.len() > 1 {
                // Pick first non-pass valid action.
                *valid.iter().find(|&&a| a > 0).unwrap_or(&0)
            } else {
                0 // pass
            };

            if action > 0 { placements += 1; }

            let step = env.step(action);
            total_reward += step.reward;
            steps += 1;

            if step.terminated || step.truncated {
                break;
            }
            if steps > 500 {
                panic!("episode should end within 500 steps at level 1");
            }
        }

        println!("Episode: {} steps, {} placements, total reward: {:.4}", steps, placements, total_reward);
        assert!(steps > 0);
    }
```

- [ ] **Step 2: Run the smoke test**

Run: `cargo test building_ai::env_test::tests::full_episode_with_random_actions -- --nocapture --test-threads=1`

Expected: Pass, prints episode summary.

- [ ] **Step 3: Commit**

```bash
git add src/world_sim/building_ai/env_test.rs
git commit -m "test(building_ai): add full episode smoke test with random actions"
```

---

## Task Dependency Graph

```
Task 7 (axis constants) ──┐
Task 8 (world_to_grid)  ──┤
Task 1 (env_config)     ──┼──→ Task 5 (BuildingEnv core) ──→ Task 6 (integration tests) ──→ Task 9 (smoke test)
Task 2 (env_reward)     ──┤
Task 3 (env_obs)        ──┤
Task 4 (triggers)       ──┘
```

Tasks 1-4, 7, 8 are independent and can be done in parallel. Task 5 depends on all of them. Tasks 6 and 9 are sequential after 5.
