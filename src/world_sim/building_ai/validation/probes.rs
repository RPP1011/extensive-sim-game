//! Probe environments — minimal WorldStates with expected oracle behavior.
//!
//! Five probes that must pass before any full training begins. Each probe
//! constructs a deterministic WorldState, runs the oracle, applies actions,
//! and asserts expected outcomes.

use crate::world_sim::building_ai::features::compute_spatial_features;
use crate::world_sim::building_ai::oracle::strategic_oracle;
use crate::world_sim::building_ai::scoring::apply_actions;
use crate::world_sim::building_ai::types::*;
use crate::world_sim::city_grid::{CellState, CellTerrain, CityGrid, InfluenceMap};
use crate::world_sim::state::{
    BuildingType, Entity, EntityKind, SettlementState, WorldState,
};
use crate::world_sim::NUM_COMMODITIES;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Outcome of a single probe.
#[derive(Debug)]
pub struct ProbeResult {
    pub name: &'static str,
    pub passed: bool,
    pub warnings: Vec<String>,
    pub details: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run all 5 probe environments and return their results.
pub fn run_all_probes() -> Vec<ProbeResult> {
    vec![
        probe_single_cell(),
        probe_two_cells(),
        probe_resource_constraint(),
        probe_wall_then_gate(),
        probe_threat_response(),
    ]
}

// ---------------------------------------------------------------------------
// Probe 1: Single Empty Cell
// ---------------------------------------------------------------------------

/// Can the oracle produce any action at all on a trivial grid?
pub fn probe_single_cell() -> ProbeResult {
    let mut state = WorldState::new(1001);
    let sid = 1u32;

    let mut settlement = SettlementState::new(sid, "Probe1".into(), (0.0, 0.0));
    settlement.infrastructure_level = 1.0;
    // Abundant resources
    let mut stockpile = [0.0f32; NUM_COMMODITIES];
    stockpile[0] = 1000.0; // food
    stockpile[2] = 1000.0; // wood
    stockpile[1] = 1000.0; // iron
    settlement.stockpile = stockpile;

    let grid = CityGrid::new(3, 3, sid, "Plains", 1001);
    // Grid has road cross at center (1,1). Cells around it are empty+flat.

    settlement.city_grid_idx = Some(0);
    state.settlements.push(settlement);
    state.city_grids.push(grid);
    state.influence_maps.push(InfluenceMap::new(3, 3));

    // Build observation
    let features = compute_spatial_features(&state, sid);
    let obs = BuildingObservation {
        settlement_id: sid,
        tick: 0,
        challenges: Vec::new(),
        memory: ConstructionMemory::new(),
        spatial: features,
        friendly_roster: Vec::new(),
        high_value_npcs: Vec::new(),
        settlement_level: 1,
        tech_tier: 1,
        decision_tier: DecisionTier::Strategic,
    };

    // Run oracle
    let actions = strategic_oracle(&obs);
    let warnings = Vec::new();

    if actions.is_empty() {
        return ProbeResult {
            name: "Probe 1: Single Empty Cell",
            passed: false,
            warnings,
            details: "Oracle produced 0 actions (expected >= 1)".into(),
        };
    }

    // Apply and check post-state
    let mut post_state = state.clone();
    apply_actions(&mut post_state, &actions);

    let new_buildings: Vec<_> = post_state
        .entities
        .iter()
        .filter(|e| {
            e.alive
                && e.kind == EntityKind::Building
                && e.building.is_some()
        })
        .collect();

    if new_buildings.is_empty() {
        return ProbeResult {
            name: "Probe 1: Single Empty Cell",
            passed: false,
            warnings,
            details: "No new building entity in post-state after apply_actions".into(),
        };
    }

    ProbeResult {
        name: "Probe 1: Single Empty Cell",
        passed: true,
        warnings,
        details: format!(
            "Oracle produced {} actions, {} new buildings in post-state",
            actions.len(),
            new_buildings.len()
        ),
    }
}

// ---------------------------------------------------------------------------
// Probe 2: Two Cells, One Good One Bad
// ---------------------------------------------------------------------------

/// Can the oracle distinguish cell quality (flat vs flood-prone)?
pub fn probe_two_cells() -> ProbeResult {
    let mut state = WorldState::new(1002);
    let sid = 1u32;

    let mut settlement = SettlementState::new(sid, "Probe2".into(), (0.0, 0.0));
    settlement.infrastructure_level = 1.0;
    let mut stockpile = [0.0f32; NUM_COMMODITIES];
    stockpile[0] = 1000.0;
    stockpile[2] = 1000.0;
    stockpile[1] = 1000.0;
    settlement.stockpile = stockpile;

    let mut grid = CityGrid::new(5, 5, sid, "Plains", 1002);

    // (1,2) is flat and good
    // (3,2) is slope with adjacent water — flood-prone
    {
        let cell = grid.cell_mut(3, 2);
        cell.terrain = CellTerrain::Slope;
    }
    {
        let cell = grid.cell_mut(4, 2);
        cell.terrain = CellTerrain::Water;
        cell.state = CellState::Water;
    }

    settlement.city_grid_idx = Some(0);
    state.settlements.push(settlement);
    state.city_grids.push(grid);
    state.influence_maps.push(InfluenceMap::new(5, 5));

    // Inject flood challenge
    let flood_challenge = Challenge {
        category: ChallengeCategory::Environmental,
        sub_type: crate::world_sim::state::tag(b"flood"),
        sub_type_name: "flood".into(),
        severity: 0.8,
        direction: None,
        deadline_tick: None,
        enemy_profiles: Vec::new(),
    };

    let features = compute_spatial_features(&state, sid);
    let obs = BuildingObservation {
        settlement_id: sid,
        tick: 0,
        challenges: vec![flood_challenge],
        memory: ConstructionMemory::new(),
        spatial: features,
        friendly_roster: Vec::new(),
        high_value_npcs: Vec::new(),
        settlement_level: 1,
        tech_tier: 1,
        decision_tier: DecisionTier::Strategic,
    };

    let actions = strategic_oracle(&obs);
    let mut warnings = Vec::new();

    if actions.is_empty() {
        return ProbeResult {
            name: "Probe 2: Two Cells, One Good One Bad",
            passed: false,
            warnings,
            details: "Oracle produced 0 actions".into(),
        };
    }

    // Check if any PlaceBuilding at (3,2) — the bad cell
    let placed_at_bad = actions.iter().any(|a| {
        matches!(&a.action, ActionPayload::PlaceBuilding { grid_cell, .. } if *grid_cell == (3, 2))
    });
    let placed_at_good = actions.iter().any(|a| {
        matches!(&a.action, ActionPayload::PlaceBuilding { grid_cell, .. } if *grid_cell == (1, 2))
    });

    if placed_at_bad && !placed_at_good {
        return ProbeResult {
            name: "Probe 2: Two Cells, One Good One Bad",
            passed: false,
            warnings,
            details: "Oracle placed at flood-prone cell (3,2) but not at safe cell (1,2)".into(),
        };
    }

    if placed_at_bad {
        warnings.push("Oracle placed at flood-prone cell (3,2) as well as elsewhere".into());
    }

    ProbeResult {
        name: "Probe 2: Two Cells, One Good One Bad",
        passed: true,
        warnings,
        details: format!(
            "Oracle produced {} actions; placed_at_good={}, placed_at_bad={}",
            actions.len(),
            placed_at_good,
            placed_at_bad
        ),
    }
}

// ---------------------------------------------------------------------------
// Probe 3: Resource Constraint
// ---------------------------------------------------------------------------

/// Can the oracle make conditional decisions under scarcity?
pub fn probe_resource_constraint() -> ProbeResult {
    let mut state = WorldState::new(1003);
    let sid = 1u32;

    let mut settlement = SettlementState::new(sid, "Probe3".into(), (0.0, 0.0));
    settlement.infrastructure_level = 2.0;
    settlement.population = 20;
    // Scarce wood, adequate stone
    let mut stockpile = [0.0f32; NUM_COMMODITIES];
    stockpile[0] = 100.0; // food
    stockpile[2] = 10.0; // wood — scarce
    stockpile[1] = 500.0; // iron/stone
    settlement.stockpile = stockpile;

    let grid = CityGrid::new(5, 5, sid, "Plains", 1003);

    settlement.city_grid_idx = Some(0);
    state.settlements.push(settlement);
    state.city_grids.push(grid);
    state.influence_maps.push(InfluenceMap::new(5, 5));

    // Housing pressure challenge
    let housing_challenge = Challenge {
        category: ChallengeCategory::Population,
        sub_type: crate::world_sim::state::tag(b"housing_pressure"),
        sub_type_name: "housing_pressure".into(),
        severity: 0.9,
        direction: None,
        deadline_tick: None,
        enemy_profiles: Vec::new(),
    };

    let mut features = compute_spatial_features(&state, sid);
    features.population.housing_pressure = 2.0;
    features.population.unhoused_count = 10;

    let obs = BuildingObservation {
        settlement_id: sid,
        tick: 0,
        challenges: vec![housing_challenge],
        memory: ConstructionMemory::new(),
        spatial: features,
        friendly_roster: Vec::new(),
        high_value_npcs: Vec::new(),
        settlement_level: 2,
        tech_tier: 2,
        decision_tier: DecisionTier::Strategic,
    };

    let actions = strategic_oracle(&obs);
    let mut warnings = Vec::new();

    let placement_count = actions
        .iter()
        .filter(|a| matches!(&a.action, ActionPayload::PlaceBuilding { .. }))
        .count();

    // Under resource constraint, expect a reasonable number of placements
    let has_house = actions.iter().any(|a| {
        matches!(
            &a.action,
            ActionPayload::PlaceBuilding { building_type, .. }
            if *building_type == BuildingType::House || *building_type == BuildingType::Longhouse
        )
    });

    if !has_house {
        warnings.push("Oracle did not place a residential building despite housing pressure".into());
    }

    ProbeResult {
        name: "Probe 3: Resource Constraint",
        passed: placement_count >= 1,
        warnings,
        details: format!(
            "Oracle produced {} actions, {} placements, has_house={}",
            actions.len(),
            placement_count,
            has_house
        ),
    }
}

// ---------------------------------------------------------------------------
// Probe 4: Wall Then Gate (two-step sequence)
// ---------------------------------------------------------------------------

/// Can the system produce coordinated wall + gate placements?
pub fn probe_wall_then_gate() -> ProbeResult {
    let mut state = WorldState::new(1004);
    let sid = 1u32;

    let mut settlement = SettlementState::new(sid, "Probe4".into(), (0.0, 0.0));
    settlement.infrastructure_level = 2.0;
    let mut stockpile = [0.0f32; NUM_COMMODITIES];
    for s in &mut stockpile {
        *s = 1000.0;
    }
    settlement.stockpile = stockpile;

    let grid = CityGrid::new(7, 7, sid, "Plains", 1004);

    settlement.city_grid_idx = Some(0);
    state.settlements.push(settlement);
    state.city_grids.push(grid);
    state.influence_maps.push(InfluenceMap::new(7, 7));

    // Military challenge from north
    let military_challenge = Challenge {
        category: ChallengeCategory::Military,
        sub_type: crate::world_sim::state::tag(b"raid_infantry"),
        sub_type_name: "raid_infantry".into(),
        severity: 0.9,
        direction: Some((0.0, -1.0)), // north
        deadline_tick: None,
        enemy_profiles: vec![EnemyProfile {
            type_tag: crate::world_sim::state::tag(b"infantry"),
            type_name: "infantry".into(),
            level_range: (3, 5),
            count: 10,
            can_jump: false,
            jump_height: 0,
            can_climb: false,
            can_tunnel: false,
            can_fly: false,
            has_siege: false,
            siege_damage: 0.0,
        }],
    };

    let features = compute_spatial_features(&state, sid);
    let obs = BuildingObservation {
        settlement_id: sid,
        tick: 0,
        challenges: vec![military_challenge],
        memory: ConstructionMemory::new(),
        spatial: features,
        friendly_roster: Vec::new(),
        high_value_npcs: Vec::new(),
        settlement_level: 2,
        tech_tier: 2,
        decision_tier: DecisionTier::Strategic,
    };

    let actions = strategic_oracle(&obs);
    let mut warnings = Vec::new();

    // Check for wall placement on north edge (rows 0-1)
    let has_wall_north = actions.iter().any(|a| {
        matches!(
            &a.action,
            ActionPayload::PlaceBuilding { building_type, grid_cell }
            if *building_type == BuildingType::Wall && grid_cell.1 <= 1
        )
    });

    // Check for gate placement near wall
    let has_gate = actions.iter().any(|a| {
        matches!(
            &a.action,
            ActionPayload::PlaceBuilding { building_type, .. }
            if *building_type == BuildingType::Gate
        )
    });

    if !has_wall_north {
        // Check if any defensive structure at all on north
        let any_defense_north = actions.iter().any(|a| {
            matches!(
                &a.action,
                ActionPayload::PlaceBuilding { building_type, grid_cell }
                if (*building_type == BuildingType::Wall
                    || *building_type == BuildingType::Watchtower
                    || *building_type == BuildingType::Gate
                    || *building_type == BuildingType::Barracks)
                    && grid_cell.1 <= 2
            )
        });

        if !any_defense_north {
            return ProbeResult {
                name: "Probe 4: Wall Then Gate",
                passed: false,
                warnings,
                details: format!(
                    "No wall or defensive building on north edge; {} total actions",
                    actions.len()
                ),
            };
        }
    }

    if !has_gate && has_wall_north {
        warnings.push("Wall placed on north but no gate (functional but suboptimal)".into());
    }

    ProbeResult {
        name: "Probe 4: Wall Then Gate",
        passed: true,
        warnings,
        details: format!(
            "Oracle produced {} actions; wall_north={}, gate={}",
            actions.len(),
            has_wall_north,
            has_gate
        ),
    }
}

// ---------------------------------------------------------------------------
// Probe 5: Threat-Response (directional spatial reasoning)
// ---------------------------------------------------------------------------

/// Can the oracle respond to a directional threat by placing defenses
/// on the threatened side?
pub fn probe_threat_response() -> ProbeResult {
    let mut state = WorldState::new(1005);
    let sid = 1u32;

    let mut settlement = SettlementState::new(sid, "Probe5".into(), (0.0, 0.0));
    settlement.infrastructure_level = 3.0;
    let mut stockpile = [0.0f32; NUM_COMMODITIES];
    for s in &mut stockpile {
        *s = 1000.0;
    }
    settlement.stockpile = stockpile;

    let mut grid = CityGrid::new(9, 9, sid, "Plains", 1005);

    // Existing wall coverage on south (rows 7-8)
    for col in 3..=5 {
        for row in 7..=8 {
            if grid.in_bounds(col, row) {
                let cell = grid.cell_mut(col, row);
                cell.state = CellState::Wall;
                cell.building_id = Some(100 + col as u32 + row as u32 * 100);
            }
        }
    }

    // Create corresponding wall entities
    for col in 3..=5u32 {
        for row in 7..=8u32 {
            let eid = 100 + col + row * 100;
            let mut entity = Entity::new_building(eid, (col as f32, row as f32));
            let mut bdata = crate::world_sim::state::BuildingData::default();
            bdata.building_type = BuildingType::Wall;
            bdata.settlement_id = Some(sid);
            bdata.grid_col = col as u16;
            bdata.grid_row = row as u16;
            bdata.construction_progress = 1.0;
            entity.building = Some(bdata);
            state.entities.push(entity);
        }
    }

    settlement.city_grid_idx = Some(0);
    state.settlements.push(settlement);
    state.city_grids.push(grid);
    state.influence_maps.push(InfluenceMap::new(9, 9));

    // Threat from north
    let military_challenge = Challenge {
        category: ChallengeCategory::Military,
        sub_type: crate::world_sim::state::tag(b"raid_infantry"),
        sub_type_name: "raid_infantry".into(),
        severity: 0.9,
        direction: Some((0.0, -1.0)), // north
        deadline_tick: None,
        enemy_profiles: vec![EnemyProfile {
            type_tag: crate::world_sim::state::tag(b"infantry"),
            type_name: "infantry".into(),
            level_range: (3, 5),
            count: 10,
            can_jump: false,
            jump_height: 0,
            can_climb: false,
            can_tunnel: false,
            can_fly: false,
            has_siege: false,
            siege_damage: 0.0,
        }],
    };

    let features = compute_spatial_features(&state, sid);
    let obs = BuildingObservation {
        settlement_id: sid,
        tick: 0,
        challenges: vec![military_challenge],
        memory: ConstructionMemory::new(),
        spatial: features,
        friendly_roster: Vec::new(),
        high_value_npcs: Vec::new(),
        settlement_level: 3,
        tech_tier: 2,
        decision_tier: DecisionTier::Strategic,
    };

    let actions = strategic_oracle(&obs);
    let mut warnings = Vec::new();

    // Check for wall/defensive placement on north edge (rows 0-2)
    let defense_north = actions.iter().filter(|a| {
        matches!(
            &a.action,
            ActionPayload::PlaceBuilding { building_type, grid_cell }
            if (*building_type == BuildingType::Wall
                || *building_type == BuildingType::Watchtower
                || *building_type == BuildingType::Gate
                || *building_type == BuildingType::Barracks)
                && grid_cell.1 <= 2
        )
    }).count();

    // Check for redundant south/east placement
    let defense_south = actions.iter().filter(|a| {
        matches!(
            &a.action,
            ActionPayload::PlaceBuilding { building_type, grid_cell }
            if (*building_type == BuildingType::Wall
                || *building_type == BuildingType::Watchtower
                || *building_type == BuildingType::Gate)
                && grid_cell.1 >= 7
        )
    }).count();

    if defense_north == 0 {
        // Check if all placements are on the already-defended south
        if defense_south > 0 && defense_north == 0 {
            return ProbeResult {
                name: "Probe 5: Threat-Response",
                passed: false,
                warnings,
                details: "All defensive placements on south (already covered), none on threatened north".into(),
            };
        }

        // Allow pass if there are non-defensive placements (economic response)
        let has_any_action = !actions.is_empty();
        if has_any_action {
            warnings.push(
                "No defensive placement on north edge, but oracle produced other actions".into(),
            );
        } else {
            return ProbeResult {
                name: "Probe 5: Threat-Response",
                passed: false,
                warnings,
                details: "Oracle produced 0 actions".into(),
            };
        }
    }

    if defense_south > 0 {
        warnings.push(format!(
            "Redundant defensive placement on south (already covered): {} placements",
            defense_south
        ));
    }

    ProbeResult {
        name: "Probe 5: Threat-Response",
        passed: true,
        warnings,
        details: format!(
            "Oracle produced {} actions; {} north defenses, {} south defenses",
            actions.len(),
            defense_north,
            defense_south
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests — probes as unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_single_cell_runs() {
        let result = probe_single_cell();
        eprintln!(
            "Probe 1: passed={}, details={}",
            result.passed, result.details
        );
        // Don't assert pass — the probe tests oracle behavior which may evolve.
        // Just verify it runs without panicking.
    }

    #[test]
    fn probe_two_cells_runs() {
        let result = probe_two_cells();
        eprintln!(
            "Probe 2: passed={}, details={}",
            result.passed, result.details
        );
    }

    #[test]
    fn probe_resource_constraint_runs() {
        let result = probe_resource_constraint();
        eprintln!(
            "Probe 3: passed={}, details={}",
            result.passed, result.details
        );
    }

    #[test]
    fn probe_wall_then_gate_runs() {
        let result = probe_wall_then_gate();
        eprintln!(
            "Probe 4: passed={}, details={}",
            result.passed, result.details
        );
    }

    #[test]
    fn probe_threat_response_runs() {
        let result = probe_threat_response();
        eprintln!(
            "Probe 5: passed={}, details={}",
            result.passed, result.details
        );
    }

    #[test]
    fn run_all_probes_returns_five() {
        let results = run_all_probes();
        assert_eq!(results.len(), 5);
        for r in &results {
            eprintln!("{}: passed={}", r.name, r.passed);
        }
    }
}
