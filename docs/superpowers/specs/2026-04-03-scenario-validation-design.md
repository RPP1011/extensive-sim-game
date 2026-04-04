# Scenario Validation System — Building AI Training Data

**Date:** 2026-04-03
**Parent specs:**
- `2026-04-03-npc-building-intelligence-design.md` (Section 12: Simulation Gym Hardening)
- `2026-04-03-mass-scenario-generation-design.md` (compositional generator)

**Goal:** Catch every class of data corruption bug BEFORE training. No bad (observation, action) pair should ever reach the model. Validation runs both inline during generation (fast checks, fail-stop) and as a post-hoc batch audit (exhaustive checks, statistical analysis).

---

## 1. Module Structure

```
src/world_sim/building_ai/
    validation/
        mod.rs              — public API: validate_world_state, validate_action, validate_pair, etc.
        world_state.rs      — WorldState internal consistency checks
        action.rs           — action applicability and post-condition checks
        determinism.rs      — replay-based determinism verification
        features.rs         — spatial feature sanity (NaN, range, parity)
        coverage.rs         — coverage tracker integrity
        memory.rs           — construction memory buffer invariants
        probes.rs           — 5 probe environment definitions + expected-outcome assertions
```

All validation functions return `Vec<ValidationError>` (empty = pass). No panics, no silent skips. Every check has a unique error code for automated triage.

```rust
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub code: &'static str,       // e.g. "WS-OVERLAP-001"
    pub severity: Severity,       // Fatal, Warning, Info
    pub message: String,          // human-readable
    pub context: ErrorContext,    // scenario_id, entity_id, cell, etc.
}

#[derive(Debug, Clone)]
pub enum Severity {
    /// Pair must be rejected. Data is provably corrupt.
    Fatal,
    /// Pair is suspicious. Log and flag for manual review. Keep in dataset
    /// only if under a configurable warning threshold per batch.
    Warning,
    /// Informational. No action needed.
    Info,
}

#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub scenario_id: Option<u64>,
    pub entity_id: Option<u32>,
    pub grid_cell: Option<(u16, u16)>,
    pub field: Option<String>,
}
```

---

## 2. WorldState Consistency Checks (`world_state.rs`)

These validate that a generated `WorldState` is internally coherent. Run on every state produced by `generate_from_seed` and on every post-action state.

### 2.1 Grid-Entity Cross-Reference (WS-XREF-*)

| Code | Check | Catches |
|---|---|---|
| WS-XREF-001 | Every `Cell` with `building_id = Some(id)` has a corresponding alive `Entity` with that id and `entity.kind == EntityKind::Building` | Dangling building IDs from failed deletions |
| WS-XREF-002 | Every alive building `Entity` with `settlement_id = Some(sid)` has its `(grid_col, grid_row)` pointing to a cell whose `building_id == Some(entity.id)` | Entity thinks it's placed but grid disagrees |
| WS-XREF-003 | No two alive building entities share the same `(settlement_id, grid_col, grid_row)` | Overlapping buildings from double-placement |
| WS-XREF-004 | For multi-cell footprints (`footprint_w > 1` or `footprint_h > 1`), ALL covered cells `[grid_col..grid_col+footprint_w, grid_row..grid_row+footprint_h]` are in-bounds AND marked with this entity's building_id | Partial footprint stamp, out-of-bounds footprint |
| WS-XREF-005 | `Cell.state` matches entity type: walls have `CellState::Wall`, other buildings have `CellState::Building` | State/type mismatch after mutation |

### 2.2 Grid Bounds and Terrain (WS-GRID-*)

| Code | Check | Catches |
|---|---|---|
| WS-GRID-001 | CityGrid `cols * rows == cells.len()` | Buffer size mismatch |
| WS-GRID-002 | No building placed on `CellTerrain::Water` or `CellTerrain::Cliff` cells (unless building type explicitly allows it, currently none do) | Generator placing on unbuildable terrain |
| WS-GRID-003 | `settlement.city_grid_idx` is `Some(i)` where `i < state.city_grids.len()` and `state.city_grids[i].settlement_id == settlement.id` | Stale or cross-linked grid index |
| WS-GRID-004 | No cell has `CellState::Building` or `CellState::Wall` with `building_id == None` | Ghost buildings (cell marked but no entity) |
| WS-GRID-005 | No cell has `CellState::Empty` with `building_id == Some(_)` | Invisible building (entity present but cell says empty) |

### 2.3 Entity Invariants (WS-ENT-*)

| Code | Check | Catches |
|---|---|---|
| WS-ENT-001 | All entity IDs are unique across `state.entities` | ID collision from `next_entity_id` bug |
| WS-ENT-002 | Building entities have `entity.building == Some(_)` | Missing building data |
| WS-ENT-003 | NPC entities have `entity.npc == Some(_)` | Missing NPC data |
| WS-ENT-004 | `entity.hp <= entity.max_hp` and both are non-negative and finite | HP corruption |
| WS-ENT-005 | Dead entities (`alive == false`) are not referenced by any `Cell.building_id` | Grid still points to dead entity |
| WS-ENT-006 | `BuildingData.resident_ids` and `worker_ids` reference entities that exist and are alive NPCs | Dangling NPC references |
| WS-ENT-007 | `BuildingData.construction_progress` is in `[0.0, 1.0]` | Out-of-range progress |
| WS-ENT-008 | `BuildingData.storage` values are non-negative and finite | Negative or NaN storage |

### 2.4 Settlement Invariants (WS-SET-*)

| Code | Check | Catches |
|---|---|---|
| WS-SET-001 | `settlement.stockpile` values are all non-negative and finite | Negative or NaN stockpiles |
| WS-SET-002 | `settlement.population >= 0` (type is u32 so this is always true, but check against NPC count for consistency) | Population/entity mismatch |
| WS-SET-003 | Settlement has at least one city grid (required for building AI) | Missing grid |

### 2.5 Compound Pressure Coherence (WS-PRESS-*)

These check that pressure injection did not create contradictory states.

| Code | Check | Catches |
|---|---|---|
| WS-PRESS-001 | After flood injection, no building entities are placed on newly-created `CellState::Water` cells | Flood overwrote occupied cell |
| WS-PRESS-002 | After military injection, enemy entities have `move_target` pointing within grid bounds | Enemies targeting out-of-world positions |
| WS-PRESS-003 | When 2+ military pressures are injected, their `direction` vectors differ by at least 45 degrees (configurable) | Near-identical attack vectors wasting a pressure slot |
| WS-PRESS-004 | After environmental + military compound, grid still has at least 10% buildable cells | Pressure combo consumed all buildable space |

**Implementation:** `pub fn validate_world_state(state: &WorldState) -> Vec<ValidationError>`. Iterates entities once, grid cells once, settlements once. O(E + G + S) where E = entities, G = grid cells, S = settlements.

**Estimated effort:** 2-3 days. Most checks are straightforward field comparisons and cross-lookups.

---

## 3. Action Validity Checks (`action.rs`)

Validates that oracle-produced actions are applicable to the current WorldState.

### 3.1 Pre-Application Checks (ACT-PRE-*)

Run BEFORE `apply_actions`. If any Fatal error fires, skip this pair entirely.

| Code | Check | Catches |
|---|---|---|
| ACT-PRE-001 | `PlaceBuilding`: `grid_cell` is in-bounds on the settlement's CityGrid | Out-of-bounds placement |
| ACT-PRE-002 | `PlaceBuilding`: target cell state is `CellState::Empty` or `CellState::Road` (not `Building`, `Wall`, `Water`) | Placing on occupied or unbuildable cell |
| ACT-PRE-003 | `PlaceBuilding`: target cell terrain is not `CellTerrain::Water` or `CellTerrain::Cliff` | Placing on unbuildable terrain |
| ACT-PRE-004 | `PlaceBuilding`: settlement has sufficient resources for the building type at the current tech tier. Resource costs are defined per `BuildingType` (not yet codified -- see Section 8 on resource cost table). Until the cost table exists, this check emits `Warning` not `Fatal` | Impossible builds the model would learn as valid |
| ACT-PRE-005 | `PlaceBuilding`: tech tier allows the building type (e.g., Iron walls require tech_tier >= 3). Same caveat as ACT-PRE-004 | Tech-gated build treated as available |
| ACT-PRE-006 | `Demolish`: `building_id` references an alive building entity | Demolishing nonexistent building |
| ACT-PRE-007 | `SetBuildPriority`, `SetFootprint`, `SetVertical`, `SetWallSpec`, `SetFoundation`, `SetOpenings`, `SetInteriorLayout`, `SetMaterial`, `Renovate`: target entity ID exists and is an alive building | Operating on nonexistent entity |
| ACT-PRE-008 | `RouteRoad`: all waypoints are in-bounds | Out-of-bounds road segment |
| ACT-PRE-009 | `SetZone`: grid_cell is in-bounds | Out-of-bounds zone |
| ACT-PRE-010 | `SetWallSpec`: height > 0, thickness > 0 | Zero-dimension wall |
| ACT-PRE-011 | `SetVertical`: stories > 0 | Zero-story building |
| ACT-PRE-012 | `SetFootprint`: both dimensions > 0, footprint fits within grid bounds from the building's grid position | Zero-size or overflowing footprint |

### 3.2 Post-Application Checks (ACT-POST-*)

Run AFTER `apply_actions` on the mutated WorldState. These confirm the action produced a valid successor state.

| Code | Check | Catches |
|---|---|---|
| ACT-POST-001 | Re-run `validate_world_state` on the post-action state. Any new errors not present in the pre-action validation are attributed to the action application | `apply_actions` introduced corruption |
| ACT-POST-002 | Entity count did not decrease (no accidental entity deletion, except via `Demolish` which marks `alive = false`) | apply_actions accidentally removed entities from the vec |
| ACT-POST-003 | For `PlaceBuilding` actions: a new entity with the specified `building_type` exists at the specified `grid_cell` | `apply_actions` silently dropped the placement |
| ACT-POST-004 | For `Demolish` actions: the target entity has `alive == false` and its grid cell has `CellState::Empty` and `building_id == None` | Partial demolish left ghost state |

**Implementation:** `pub fn validate_action_batch(state: &WorldState, actions: &[BuildingAction]) -> Vec<ValidationError>` for pre-checks, `pub fn validate_post_action(pre_state: &WorldState, post_state: &WorldState, actions: &[BuildingAction]) -> Vec<ValidationError>` for post-checks.

**Estimated effort:** 2 days. The checks are field lookups against the grid and entity list. ACT-PRE-004 and ACT-PRE-005 depend on a resource cost table that may need to be stubbed initially.

---

## 4. Determinism Verification (`determinism.rs`)

### 4.1 Seed Determinism (DET-SEED-*)

| Code | Check | Catches |
|---|---|---|
| DET-SEED-001 | `generate_from_seed(seed, rng_seed)` called N times (default N=10) produces bitwise-identical `WorldState` (compare serialized bytes via `bincode` or field-by-field) | Non-deterministic generation (thread_rng leak, HashMap iteration order, etc.) |
| DET-SEED-002 | `inject_challenge(state, challenge)` called N times on cloned states produces identical results | Challenge injection non-determinism |
| DET-SEED-003 | `populate_memory(state, challenge, sid)` called N times on cloned states produces identical memory buffers | Memory population non-determinism |

### 4.2 Action Determinism (DET-ACT-*)

| Code | Check | Catches |
|---|---|---|
| DET-ACT-001 | `apply_actions(state.clone(), actions)` called N times (default N=10) produces bitwise-identical post-states | Non-deterministic action application |
| DET-ACT-002 | `strategic_oracle(obs)` called N times on the same observation produces identical action lists (same order, same priorities) | Oracle non-determinism (float comparison instability, HashMap iteration) |
| DET-ACT-003 | `structural_oracle(obs, strategic_actions)` called N times produces identical results | Same as above for structural layer |

### 4.3 Feature Determinism (DET-FEAT-*)

| Code | Check | Catches |
|---|---|---|
| DET-FEAT-001 | `compute_spatial_features(state, settlement_id)` called N times produces identical feature structs | BFS tie-breaking, float accumulation order |

### 4.4 Reward Determinism (DET-REW-*)

| Code | Check | Catches |
|---|---|---|
| DET-REW-001 | `compute_composite(...)` called 1000 times on the same score components produces identical results | Stochastic reward (the spec calls for 1000 repetitions) |

**Implementation approach:** All determinism checks follow the same pattern: clone inputs, run N times, assert byte-equality on outputs. Extract a generic helper:

```rust
fn assert_deterministic<T: Serialize + PartialEq + Debug>(
    name: &str,
    n: usize,
    f: impl Fn() -> T,
) -> Vec<ValidationError> {
    let baseline = f();
    let mut errors = Vec::new();
    for i in 1..n {
        let result = f();
        if result != baseline {
            errors.push(ValidationError {
                code: "DET-GENERIC",
                severity: Severity::Fatal,
                message: format!("{}: run {} differs from run 0", name, i),
                context: ErrorContext::default(),
            });
        }
    }
    errors
}
```

**Estimated effort:** 1-2 days. The checks are mechanical. The main work is ensuring the comparison function is complete (PartialEq on all relevant types, or serde-based byte comparison).

---

## 5. Spatial Feature Sanity (`features.rs`)

Validates the `SpatialFeatures` struct produced by `compute_spatial_features`.

### 5.1 Numeric Sanity (FEAT-NUM-*)

| Code | Check | Catches |
|---|---|---|
| FEAT-NUM-001 | No field in `SpatialFeatures` (recursive across all sub-structs) contains `NaN` or `Inf` | Division by zero, uninitialized float |
| FEAT-NUM-002 | All fraction fields (wall_coverage, evacuation_reachability, condition, housing_pressure denominator, storage_utilization, etc.) are in `[0.0, reasonable_max]` where reasonable_max is type-specific (most are `[0.0, 1.0]`, housing_pressure can exceed 1.0) | Unbounded feature blowing up model inputs |
| FEAT-NUM-003 | `stockpiles[i]` matches `state.settlements[0].stockpile[i]` for all i (features reflect actual state) | Stale or wrongly-indexed feature extraction |
| FEAT-NUM-004 | `worker_counts.total >= worker_counts.construction + worker_counts.masonry + worker_counts.labor` | Double-counted workers |
| FEAT-NUM-005 | `connected_components >= 1` when there is at least one developed cell | Zero components with buildings present |

### 5.2 Structural Sanity (FEAT-STRUCT-*)

| Code | Check | Catches |
|---|---|---|
| FEAT-STRUCT-001 | `key_building_paths` entries reference entity IDs that exist in the world state | Stale building path entries |
| FEAT-STRUCT-002 | `wall_segments` start/end coordinates are within grid bounds | Out-of-bounds wall segment |
| FEAT-STRUCT-003 | `garrison.coverage_map.len()` matches the number of perimeter cells (or is a consistent grid-derived size) | Buffer size mismatch |
| FEAT-STRUCT-004 | `garrison.response_time_map` values are non-negative (no negative BFS distances) | BFS bug |
| FEAT-STRUCT-005 | `garrison.synergy_hotspots` reference valid unit and structure IDs | Stale synergy entries |

### 5.3 Exhaustive Float Walk

Rather than enumerating every float field, implement a generic walker:

```rust
pub fn walk_floats(features: &SpatialFeatures) -> Vec<(String, f32)> {
    // Serialize to JSON, walk all number nodes, collect (path, value) pairs.
    // Check: is_finite(), within expected range per path.
}
```

This is defensive against new fields being added to `SpatialFeatures` without corresponding validation.

**Estimated effort:** 1-2 days. The float walker is ~50 lines using serde_json introspection. The specific checks are field lookups.

---

## 6. Observation-Action Timing (`action.rs`, integrated with post-action checks)

The V2 spec (Section 12.3) warns that off-by-one timing errors silently degrade training.

### 6.1 Timing Protocol

The generation pipeline MUST follow this sequence for each pair:

```
1. Generate WorldState (pre-action state)
2. Compute SpatialFeatures from pre-action state
3. Build BuildingObservation from pre-action state + features
4. Run oracle on observation -> actions
5. apply_actions(state, actions) -> post-action state
6. The EMITTED pair is: (observation_from_step_3, actions_from_step_4)
```

The observation reflects the state the oracle SAW when making its decision (pre-action), NOT the post-action state.

### 6.2 Timing Validation (TIME-*)

| Code | Check | Catches |
|---|---|---|
| TIME-001 | The observation's `tick` field matches the pre-action `state.tick`, not the post-action tick | Observation contains post-action time |
| TIME-002 | The observation's `spatial.economic.stockpiles` matches the pre-action state's stockpiles, not the post-action (which would be depleted by building costs if resource tracking is implemented) | Observation leaks post-action resource state |
| TIME-003 | The observation's `spatial.defensive.wall_coverage` does not account for walls placed by the oracle's own actions in this step | Observation includes effects of its own decision |
| TIME-004 | Building IDs referenced in oracle actions do not appear in the observation's feature structures (they don't exist yet for new placements) | Time-traveling entity references |

**Implementation:** The timing checks compare the observation's embedded data against a snapshot of the pre-action state. The generator must retain the pre-action state (or a hash of key fields) alongside each pair.

**Estimated effort:** 1 day. These are comparison checks between two snapshots.

---

## 7. Coverage Tracker Integrity (`coverage.rs`)

### 7.1 Matrix Checks (COV-*)

| Code | Check | Catches |
|---|---|---|
| COV-001 | `matrix` dimensions are exactly `[10][18]` (10 ChallengeCategory variants x 18 DecisionType variants) | Wrong matrix size |
| COV-002 | `sum(matrix[i][j] for all i,j) + dead_cell_adjustments == total_pairs` | Counter drift |
| COV-003 | Every `(category, decision_type)` pair in oracle output maps to a valid matrix cell index. Category index = `ChallengeCategory as u8`, decision index = `DecisionType as u8` | Enum variant not matching matrix position |
| COV-004 | Dead cells (identified empirically) are never counted (their matrix value stays 0) | Dead cell accidentally incremented |
| COV-005 | After recomputing sampling weights, all weights are non-negative and sum to > 0 for each axis | Zero-weight axis (generation stalls) |

**Estimated effort:** 0.5 days. Pure arithmetic checks.

---

## 8. Memory Buffer Integrity (`memory.rs`)

### 8.1 Ring Buffer Checks (MEM-*)

| Code | Check | Catches |
|---|---|---|
| MEM-001 | `ring_buffer.items.len() <= ring_buffer.capacity` | Buffer overflow |
| MEM-002 | `ring_buffer.head < ring_buffer.capacity` | Head pointer out of bounds |
| MEM-003 | Short-term buffer capacity is 64, medium-term is 256, long-term is 64 (matches `ConstructionMemory::new()`) | Misconfigured capacity |
| MEM-004 | All `ConstructionEvent.severity` values are in `[0.0, 1.0]` | Out-of-range severity |
| MEM-005 | All `AggregatedPattern.importance` values are in `[0.0, max_reasonable]` (uncapped, but should be finite) | NaN/Inf importance |
| MEM-006 | All `StructuralLesson.confidence` values are in `[0.0, 1.0]` | Out-of-range confidence |
| MEM-007 | `StructuralLesson.lesson_tag` is one of the known `bi_tags::*` constants | Unknown lesson tag (typo or corruption) |
| MEM-008 | Medium-term `first_tick <= last_tick` | Inverted tick range |
| MEM-009 | No event has a `tick` value greater than the current world state tick | Future-dated event |
| MEM-010 | Event locations `(col, row)` are within grid bounds | Out-of-bounds event location |

**Estimated effort:** 0.5 days. All field-level range checks.

---

## 9. Probe Environments (`probes.rs`)

Five minimal environments that must be solved before any full training begins. Each probe has a deterministic WorldState constructor, a single correct action (or ordered action set), and an assertion on expected outcome.

### Probe 1: Single Empty Cell

**Purpose:** Can the agent take any action at all?

```
WorldState:
- 1 settlement, level 1, tech_tier 1
- CityGrid: 3x3, all Empty except center cell is Road
- Resources: wood=1000, stone=1000 (abundant, no constraint)
- No NPCs, no enemies, no challenges
- One valid placement cell: (1, 0) -- the cell adjacent to center road

Expected oracle action:
- PlaceBuilding { building_type: House, grid_cell: (1, 0) }
  (any building type is acceptable; the point is non-empty action list)

Validation:
- Oracle produces >= 1 action
- Post-action state has exactly 1 new building entity
- Post-action grid cell (1, 0) has CellState::Building
```

### Probe 2: Two Cells, One Good One Bad

**Purpose:** Can the agent distinguish cell quality?

```
WorldState:
- 1 settlement, level 1, tech_tier 1
- CityGrid: 5x5, two empty cells: (1,2) is CellTerrain::Flat,
  (3,2) is CellTerrain::Slope with an adjacent Water cell making it flood-prone
- Center (2,2) is Road
- Resources: abundant
- Challenge: flood, severity 0.8, affecting row 3+

Expected oracle action:
- PlaceBuilding at (1,2), NOT at (3,2)
  (any building type; the point is correct cell selection)

Validation:
- Oracle places at (1,2), not (3,2)
- If oracle places at (3,2), probe FAILS -- spatial reasoning broken
```

### Probe 3: Resource Constraint

**Purpose:** Can the agent make conditional decisions under scarcity?

```
WorldState:
- 1 settlement, level 2, tech_tier 2
- CityGrid: 5x5, two empty cells near center
- Resources: wood=10, stone=500 (enough for one stone building, not wood+stone)
- No enemies
- Challenge: population pressure (housing_pressure = 2.0)
- Two candidate building types: House (costs wood, worse for stone-only scenario)
  and Warehouse (costs stone, but doesn't help housing)
- Correct choice depends on what the oracle prioritizes under constraint:
  a stone-compatible House placement

Expected oracle behavior:
- Oracle produces exactly 1 PlaceBuilding action (cannot afford two)
- The chosen building type addresses the highest-priority challenge (housing)

Validation:
- Exactly 1 PlaceBuilding action emitted
- Building type is House (or equivalent residential building)
- If 2+ placements emitted, probe FAILS -- ignoring resource constraint
```

### Probe 4: Two-Step Sequence (Wall Then Gate)

**Purpose:** Can the system handle credit assignment over horizon > 1?

```
WorldState:
- 1 settlement, level 2, tech_tier 2
- CityGrid: 7x7, north row (row 0) entirely empty
- Existing buildings in center (row 3)
- Resources: abundant
- Challenge: military, infantry raid from north, severity 0.9

Expected oracle behavior:
- Strategic oracle emits: PlaceBuilding(Wall, (3,0)) AND PlaceBuilding(Gate, (3,1))
  (or similar wall-then-gate adjacency)
- Wall and gate are co-located to form a defensive segment

Validation:
- Actions contain at least one Wall placement on the north edge (row 0 or 1)
- Actions contain at least one Gate placement adjacent to a Wall placement
- Post-action state: wall and gate entities exist, grid cells marked correctly
- If wall placed but no gate: probe WARNS (functional but suboptimal)
- If neither wall nor gate on threatened edge: probe FAILS
```

### Probe 5: Threat-Response (Directional Spatial Reasoning)

**Purpose:** Can the agent respond to a directional threat?

```
WorldState:
- 1 settlement, level 3, tech_tier 2
- CityGrid: 9x9, some existing buildings in center cluster (rows 3-5, cols 3-5)
- Partial wall coverage on south and east (rows 7-8, cols 7-8)
- NO wall coverage on north (rows 0-1)
- Resources: adequate
- Challenge: military, enemy from NORTH, severity 0.9, 10 infantry

Expected oracle behavior:
- Wall placement on the north edge (row 0 or 1, any column in the developed range)
- NOT redundant wall on south/east (already covered)

Validation:
- At least one Wall or defensive building placed in rows 0-2
- No defensive building placed in rows 7-8 (already covered, would be redundant)
- If all placements are on south/east: probe FAILS -- no directional reasoning
- If north wall placed: probe PASSES
```

### Probe Implementation

Each probe is a function returning `ProbeResult`:

```rust
pub struct ProbeResult {
    pub name: &'static str,
    pub passed: bool,
    pub warnings: Vec<String>,
    pub details: String,      // human-readable explanation of what happened
}

pub fn run_all_probes() -> Vec<ProbeResult> {
    vec![
        probe_single_cell(),
        probe_two_cells(),
        probe_resource_constraint(),
        probe_wall_then_gate(),
        probe_threat_response(),
    ]
}
```

Each probe function:
1. Constructs a deterministic WorldState (hardcoded seed)
2. Computes spatial features
3. Builds observation
4. Runs oracle
5. Applies actions
6. Asserts expected outcomes

Probes are also exposed as `#[test]` functions so they run in `cargo test`.

**Estimated effort:** 2 days. The state construction is manual but small (3x3 to 9x9 grids). The assertions are straightforward.

---

## 10. Pipeline Integration

### 10.1 Inline Validation (During Generation)

The mass generator (`generate_scenario` in the composition engine) integrates validation at three points:

```
generate_scenario():
    1. Generate WorldState from 5-axis composition
    2. >>> validate_world_state(state)          [INLINE, FAIL-STOP on Fatal]
    3. Inject pressures
    4. >>> validate_world_state(state)          [INLINE, FAIL-STOP on Fatal]
    5. Populate memory
    6. >>> validate_memory(memory)              [INLINE, FAIL-STOP on Fatal]
    7. Compute spatial features
    8. >>> validate_features(features)          [INLINE, FAIL-STOP on Fatal]
    9. Build observation
    10. Run oracle -> actions
    11. >>> validate_action_batch(state, actions) [INLINE, FAIL-STOP on Fatal]
    12. apply_actions(state, actions) -> post_state
    13. >>> validate_post_action(state, post_state, actions) [INLINE, FAIL-STOP]
    14. >>> validate_world_state(post_state)     [INLINE, FAIL-STOP on Fatal]
    15. Emit (observation, actions) pair to JSONL

    On any Fatal: skip this scenario, log error, increment rejection counter.
    On Warning: emit pair but tag it in metadata, increment warning counter.
```

**Performance budget:** Inline validation must add < 5% overhead to per-scenario generation time. The checks are O(E + G) per state (entity scan + grid scan). With 128x128 grids (~16K cells) and ~100 entities, each validation pass is ~microseconds. Three passes per scenario = negligible.

### 10.2 Post-Hoc Batch Validation

A separate validation pass over the entire JSONL dataset, run after generation completes. This catches statistical anomalies that inline checks miss.

```
validate_dataset(path):
    1. Load all (observation, action, meta) triples
    2. Per-pair:
       - Deserialize observation, validate all feature ranges
       - Deserialize actions, validate all payloads
       - Check observation-action timing (TIME-* checks against meta)
    3. Aggregate:
       - Distribution of actions per DecisionType (flag if any type is 0%)
       - Distribution of challenges per category (flag underrepresented)
       - Feature value distributions (flag outliers beyond 5 sigma)
       - Duplicate detection (identical observation+action pairs)
       - Coverage matrix reconstruction and comparison with generator's tracked matrix
    4. Determinism spot-check:
       - Pick 10 random scenario seeds from the dataset
       - Re-run generation with same seed
       - Assert identical output pairs
    5. Report: pass/fail, warning count, rejection count, coverage gaps
```

### 10.3 Determinism Stress Test

A dedicated pass, not run inline (too expensive), but run before any training campaign begins.

```
run_determinism_suite():
    1. Pick 100 random seeds from the dataset (or generate fresh)
    2. For each seed:
       a. Run full pipeline 10 times
       b. Assert bitwise-identical (observation, action) pairs
    3. Report any non-deterministic seeds
```

This covers DET-SEED-*, DET-ACT-*, DET-FEAT-*, DET-REW-* checks.

---

## 11. CLI Interface

### 11.1 Xtask Subcommand

```bash
# Run all probes (fast, ~seconds)
cargo run --bin xtask -- building-ai validate probes

# Validate a generated dataset (post-hoc)
cargo run --bin xtask -- building-ai validate dataset generated/building_bc.jsonl

# Run determinism stress test on 100 seeds
cargo run --bin xtask -- building-ai validate determinism --seeds 100 --reps 10

# Run inline validation stats from a generation run (reads rejection log)
cargo run --bin xtask -- building-ai validate report generated/building_validation.log

# Full pre-training validation suite (probes + determinism + dataset)
cargo run --bin xtask -- building-ai validate all \
    --dataset generated/building_bc.jsonl \
    --seeds 100
```

### 11.2 Test Suite

```bash
# Probes as unit tests (run in CI)
cargo test building_ai::validation::probes

# WorldState consistency as property tests
cargo test building_ai::validation::world_state

# Determinism tests (slower, run in CI with --release)
cargo test building_ai::validation::determinism -- --test-threads=1
```

Probes and WorldState checks are `#[test]` functions. Determinism tests use `#[test]` with `--test-threads=1` for reproducibility (same pattern as `src/ai/core/tests/determinism.rs`).

---

## 12. Error Reporting and Thresholds

### 12.1 Generation-Time Thresholds

| Metric | Threshold | Action |
|---|---|---|
| Fatal rejection rate | > 1% of generated scenarios | Stop generation, investigate. Generator has a systematic bug |
| Warning rate | > 5% of generated pairs | Flag for review. Likely a soft issue (borderline placements, edge-case features) |
| Determinism failure | Any | Stop everything. Fix the non-determinism source before proceeding |

### 12.2 Dataset-Level Thresholds

| Metric | Threshold | Action |
|---|---|---|
| Any coverage matrix cell < `min_cell` target | N/A (handled by fill-gaps command) | Run fill-gaps |
| Any DecisionType with 0 examples | Report as gap | May indicate dead oracle rule or missing pressure type |
| Duplicate pair rate | > 0.1% | Investigate RNG seeding |
| Feature outliers (> 5 sigma) | > 0.01% of pairs | Inspect manually. May be valid extreme scenarios or a bug |

### 12.3 Validation Log Format

```jsonl
{"timestamp": "...", "scenario_id": 1234, "code": "WS-XREF-003", "severity": "fatal", "message": "Overlapping buildings at (14, 7): entities 42 and 58", "context": {"entity_id": 42, "grid_cell": [14, 7]}}
{"timestamp": "...", "scenario_id": 1235, "code": "FEAT-NUM-001", "severity": "fatal", "message": "NaN in spatial.garrison.coverage_map[12]", "context": {"field": "garrison.coverage_map[12]"}}
```

---

## 13. Implementation Priority

Ordered by impact on training data quality:

| Priority | Component | Effort | Catches |
|---|---|---|---|
| **P0** | WorldState consistency (Section 2) | 2-3 days | The most common and dangerous class of bugs: overlapping buildings, dangling references, invalid grids |
| **P0** | Action pre-checks (Section 3.1) | 1 day | Oracle producing impossible actions that the model would learn as valid |
| **P0** | Feature NaN/Inf check (Section 5.1, FEAT-NUM-001 only) | 0.5 days | NaN propagation silently destroys model weights |
| **P1** | Probes (Section 9) | 2 days | Validates the oracle itself works correctly on minimal cases before trusting it on 50K scenarios |
| **P1** | Determinism (Section 4) | 1-2 days | Non-deterministic data means the model sees inconsistent labels for identical inputs |
| **P1** | Action post-checks (Section 3.2) | 1 day | apply_actions corrupting state |
| **P2** | Full feature validation (Section 5) | 1 day | Feature extraction bugs that produce learnable-but-wrong signals |
| **P2** | Timing checks (Section 6) | 1 day | Off-by-one temporal leakage |
| **P2** | Memory checks (Section 8) | 0.5 days | Memory buffer corruption |
| **P3** | Coverage tracker (Section 7) | 0.5 days | Coverage drift (less critical, caught by post-hoc audit) |
| **P3** | Post-hoc batch validation (Section 10.2) | 1-2 days | Statistical anomalies, distribution issues |
| **P3** | Compound pressure coherence (Section 2.5) | 0.5 days | Generator-specific edge cases |

**Total estimated effort:** 12-16 days for full implementation. P0 items (5-6 days) should be done before any generation run. P1 items (4-5 days) before the first training run. P2/P3 items can be added incrementally.

---

## 14. Placement Validity Exhaustion (Section 12.1 of V2 Spec)

The V2 spec calls for attempting every possible `(cell, building_type)` combination on 100 random settlement states. This is a distinct validation mode from the per-scenario inline checks.

### 14.1 Procedure

```
For each of 100 random seeds:
    1. Generate a WorldState at random maturity (empty through dense)
    2. For each cell (c, r) in the CityGrid:
        For each BuildingType variant (26 types):
            a. Clone the state
            b. Construct a PlaceBuilding action for (type, (c, r))
            c. Run validate_action_batch (pre-checks):
               - If REJECTED (Fatal): record as "invalid placement"
               - If ACCEPTED: apply the action, then:
                 i.  Run validate_world_state on post-state
                 ii. Run validate_post_action
                 iii. Check for pathological outcomes:
                      - Does this placement block all paths from any building
                        to any gate? (connectivity check)
                      - Does this placement create overlapping influence
                        with an adjacent building of the same type?
                      - Are resources double-counted?
                 iv. If post-checks FAIL: log as "valid-but-pathological"
    3. Aggregate:
       - Total valid placements per building type per maturity level
       - Total invalid placements and their rejection reasons
       - Total pathological placements (valid pre-check but bad post-state)
```

### 14.2 Scale

128x128 grid x 26 building types = ~425K combinations per seed. 100 seeds = ~42.5M checks. At ~1us per pre-check (no apply needed for rejected), this is ~42 seconds for rejections + apply cost for valid ones. Parallelizable across seeds. Budget: ~10 minutes on 8 cores.

### 14.3 CLI

```bash
cargo run --release --bin xtask -- building-ai validate exhaustive \
    --seeds 100 --threads 8 \
    --output generated/placement_exhaustion.json
```

**Estimated effort:** 1-2 days (separate from the per-component estimates above, as this is an integration-level test).

---

## 15. Resource Cost Table Dependency

Several action pre-checks (ACT-PRE-004, ACT-PRE-005) and the resource constraint probe (Probe 3) depend on a resource cost table that maps `(BuildingType, BuildMaterial, tech_tier)` to `(wood_cost, stone_cost, iron_cost, food_cost)`. This table does not yet exist in code.

### Options

1. **Stub with Warning:** Until the table exists, ACT-PRE-004 and ACT-PRE-005 emit `Warning` instead of `Fatal`. The probe hardcodes expected costs. This is the recommended path -- it unblocks all other validation work.

2. **Define the table first:** Add a `fn building_cost(btype: BuildingType, material: BuildMaterial, tier: u8) -> ResourceCost` to `src/world_sim/building_ai/types.rs`. This is independently useful for the generator and oracle.

The table should be added as a separate task before P1 validation is complete.
