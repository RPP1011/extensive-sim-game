//! Oracle heuristics — rule + utility layers that generate BC labels.
//!
//! The oracle reads a `BuildingObservation` and emits labeled `BuildingAction`s.
//! Two tiers: strategic (grid-level) and structural (tile-level).

use super::types::{
    bi_tags, ActionPayload, BuildMaterial, BuildingAction, BuildingObservation,
    BuildingComponent, ChallengeCategory, ConstructionEventKind, DecisionTier,
    DecisionType, Direction, FootprintShape, FoundationType, OpeningSpec, OpeningType,
    RoofType, RoomKindBi, RoomPlacement, WallFeatures,
};
use crate::world_sim::state::BuildingType;

/// Virtual grid is 128×128; max valid coordinate is 127.
const VGRID_MAX: u16 = 127;

/// Clamp a grid cell to valid virtual-grid bounds.
fn clamp_cell(col: u16, row: u16) -> (u16, u16) {
    (col.min(VGRID_MAX), row.min(VGRID_MAX))
}

// ---------------------------------------------------------------------------
// Oracle entry points
// ---------------------------------------------------------------------------

/// Strategic oracle: grid-level decisions (placement, routing, zone, priority, demolish).
/// Returns actions sorted by priority descending.
pub fn strategic_oracle(obs: &BuildingObservation) -> Vec<BuildingAction> {
    let weights = strategic_weights_for_obs(obs);

    // Phase 1: Rule layer — reactive rules that fire on obvious conditions.
    let mut rule_actions = strategic_rule_layer(obs);

    // Phase 2: Utility layer — score all candidate placements via Chebyshev.
    let mut utility_actions = strategic_utility_layer(obs, &weights, &rule_actions);

    // Merge: rule actions get a priority boost (they are reactive/urgent).
    for a in &mut rule_actions {
        a.priority += 0.5;
    }

    let mut all = rule_actions;
    all.append(&mut utility_actions);

    // Sort descending by priority. Tie-break: lower grid cell index.
    all.sort_by(|a, b| {
        let cmp = b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal);
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
        let cell_a = action_grid_cell(&a.action);
        let cell_b = action_grid_cell(&b.action);
        cell_a.cmp(&cell_b)
    });

    // Dynamic action limit: base 4 + 2 per challenge + 1 per 10 existing buildings, capped at 16.
    let n_challenges = obs.challenges.len();
    let n_buildings = obs.spatial.occupied_cells.len();
    let max_actions = (4 + n_challenges * 2 + n_buildings / 10).clamp(4, 16);
    all.truncate(max_actions);
    all
}

/// Structural oracle: tile-level decisions for each strategic action.
/// Takes the strategic actions as context (e.g. "we're placing a barracks at (14,3),
/// now decide its wall composition, openings, interior layout").
pub fn structural_oracle(
    obs: &BuildingObservation,
    strategic_actions: &[BuildingAction],
) -> Vec<BuildingAction> {
    let weights = structural_weights_for_obs(obs);
    let mut results = Vec::new();

    for (sa_idx, sa) in strategic_actions.iter().enumerate() {
        let building_id = sa_idx as u32 + 10000; // synthetic IDs for new buildings
        let segment_id = building_id; // wall specs keyed by same ID
        let grid_cell = action_grid_cell(&sa.action);

        match &sa.action {
            ActionPayload::PlaceBuilding { building_type, .. } => {
                // Generate structural specs for this building.
                let mut specs =
                    structural_for_building(obs, &weights, *building_type, grid_cell, building_id, segment_id);
                // Clamp footprint dimensions so they don't overflow the virtual grid.
                for spec in &mut specs {
                    if let ActionPayload::SetFootprint { dimensions, .. } = &mut spec.action {
                        let max_w = ((VGRID_MAX + 1).saturating_sub(grid_cell.0)).max(1) as u8;
                        let max_h = ((VGRID_MAX + 1).saturating_sub(grid_cell.1)).max(1) as u8;
                        dimensions.0 = dimensions.0.min(max_w);
                        dimensions.1 = dimensions.1.min(max_h);
                    }
                }
                results.append(&mut specs);
            }
            ActionPayload::SetZone { .. } | ActionPayload::Demolish { .. } => {
                // Zone/demolish actions don't need structural follow-up.
            }
            _ => {}
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Oracle weight context — shifts utility weights by challenge & constraints
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};

/// Utility weights for strategic scoring. Loaded from TOML or computed from challenge context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicWeights {
    pub threat_mitigation: f32,
    pub economic_value: f32,
    pub population_relief: f32,
    pub connectivity_improve: f32,
    pub resource_feasibility: f32,
    pub workforce_match: f32,
    pub level_appropriateness: f32,
    pub garrison_synergy: f32,
}

impl Default for StrategicWeights {
    fn default() -> Self {
        Self {
            threat_mitigation: 1.0,
            economic_value: 1.0,
            population_relief: 1.0,
            connectivity_improve: 1.0,
            resource_feasibility: 1.0,
            workforce_match: 1.0,
            level_appropriateness: 1.0,
            garrison_synergy: 1.0,
        }
    }
}

/// Utility weights for structural scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralWeights {
    pub defensive_value: f32,
    pub cost_efficiency: f32,
    pub construction_time: f32,
    pub material_availability: f32,
    pub terrain_fit: f32,
    pub expansion_potential: f32,
    pub unit_synergy: f32,
    pub garrison_amplification: f32,
}

impl Default for StructuralWeights {
    fn default() -> Self {
        Self {
            defensive_value: 1.0,
            cost_efficiency: 1.0,
            construction_time: 1.0,
            material_availability: 1.0,
            terrain_fit: 1.0,
            expansion_potential: 1.0,
            unit_synergy: 1.0,
            garrison_amplification: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight derivation from observation context
// ---------------------------------------------------------------------------

fn strategic_weights_for_obs(obs: &BuildingObservation) -> StrategicWeights {
    let mut w = StrategicWeights::default();
    for ch in &obs.challenges {
        match ch.category {
            ChallengeCategory::Military => {
                w.threat_mitigation *= 2.0;
                w.garrison_synergy *= 1.5;
            }
            ChallengeCategory::Environmental => {
                w.resource_feasibility *= 1.5;
                w.connectivity_improve *= 1.3;
            }
            ChallengeCategory::Economic => {
                w.economic_value *= 2.0;
                w.resource_feasibility *= 1.5;
            }
            ChallengeCategory::Population => {
                w.population_relief *= 2.0;
                w.workforce_match *= 1.3;
            }
            ChallengeCategory::Temporal => {
                w.resource_feasibility *= 1.5;
                w.workforce_match *= 1.5;
            }
            ChallengeCategory::UnitCapability => {
                w.garrison_synergy *= 2.0;
            }
            ChallengeCategory::HighValueNpc => {
                w.threat_mitigation *= 1.5;
                w.garrison_synergy *= 1.3;
            }
            _ => {}
        }
    }
    w
}

fn structural_weights_for_obs(obs: &BuildingObservation) -> StructuralWeights {
    let mut w = StructuralWeights::default();
    for ch in &obs.challenges {
        match ch.category {
            ChallengeCategory::Military => {
                w.defensive_value *= 2.0;
                w.garrison_amplification *= 1.5;
            }
            ChallengeCategory::Environmental => {
                w.terrain_fit *= 2.0;
            }
            ChallengeCategory::Economic => {
                w.cost_efficiency *= 2.0;
            }
            ChallengeCategory::Temporal => {
                w.construction_time *= 2.0;
                w.cost_efficiency *= 1.5;
            }
            _ => {}
        }
    }
    // Resource scarcity shifts toward cost efficiency.
    let scarcity = 1.0
        - (obs.spatial.economic.stockpiles.iter().sum::<f32>().min(1000.0) / 1000.0);
    if scarcity > 0.5 {
        w.cost_efficiency *= 1.5;
        w.material_availability *= 1.5;
    }
    w
}

// ---------------------------------------------------------------------------
// Ideal point estimation for Chebyshev scalarization
// ---------------------------------------------------------------------------

/// Compute per-objective ideal point R* from the current observation.
/// Each R*_i estimates the best achievable score for objective i given
/// current resources, terrain, settlement level, and active threats.
/// Values are in (0, 1] — never zero to avoid division by zero in Chebyshev.
fn compute_ideal_point(obs: &BuildingObservation) -> [f32; 8] {
    let has_threat = obs.challenges.iter().any(|c| c.severity > 0.3);
    // Resource abundance factor (0 = empty, 1 = well-stocked).
    let resource_factor = (obs.spatial.economic.stockpiles.iter().sum::<f32>() / 400.0)
        .clamp(0.1, 1.0);

    // Workforce factor.
    let total_workers = obs.spatial.economic.worker_counts.total as f32;
    let workforce_factor = (total_workers / 20.0).clamp(0.1, 1.0);

    // Settlement scale factor (larger settlements can achieve more).
    let scale = (obs.settlement_level as f32 / 5.0).clamp(0.2, 1.0);

    [
        // threat_mitigation: high if threats exist, limited by resources for walls.
        if has_threat { (0.5 + resource_factor * 0.5).min(1.0) } else { 0.3 },
        // economic_value: limited by settlement level and existing infrastructure.
        (0.3 + scale * 0.5 + resource_factor * 0.2).min(1.0),
        // population_relief: high if housing pressure exists.
        if obs.spatial.population.housing_pressure > 1.0 {
            (0.4 + resource_factor * 0.4 + workforce_factor * 0.2).min(1.0)
        } else {
            0.3
        },
        // connectivity_improve: based on how disconnected things are.
        if obs.spatial.connectivity.connected_components > 1 { 0.8 } else { 0.5 },
        // resource_feasibility: direct function of what's available.
        resource_factor,
        // workforce_match: based on available workers.
        workforce_factor,
        // level_appropriateness: based on settlement level.
        scale,
        // garrison_synergy: high if we have combat NPCs.
        {
            let garrison_count = obs.friendly_roster.iter().filter(|u| u.is_garrison).count();
            ((garrison_count as f32) / 5.0).clamp(0.2, 1.0)
        },
    ]
}

// ---------------------------------------------------------------------------
// Strategic rule layer
// ---------------------------------------------------------------------------

fn strategic_rule_layer(obs: &BuildingObservation) -> Vec<BuildingAction> {
    let mut actions = Vec::new();

    // Rule 1: Threat from direction D + no defensive coverage on D → place watchtower.
    for ch in &obs.challenges {
        if let Some(dir) = ch.direction {
            let threat_severity = ch.severity;
            if threat_severity < 0.1 {
                continue;
            }
            // Check if there's wall coverage in that direction.
            let has_wall = wall_covers_direction(obs, dir);
            if !has_wall {
                // Pick a grid cell along the threat direction from settlement center.
                let cell = direction_to_perimeter_cell(obs, dir);
                actions.push(BuildingAction {
                    decision_type: DecisionType::Placement,
                    tier: DecisionTier::Strategic,
                    action: ActionPayload::PlaceBuilding {
                        building_type: BuildingType::Watchtower,
                        grid_cell: cell,
                    },
                    priority: threat_severity,
                    reasoning_tag: bi_tags::THREAT_PROXIMITY,
                });
            }
        }
    }

    // Rule 2: Housing pressure > 1.2 → place residential building.
    if obs.spatial.population.housing_pressure > 1.2 {
        let cell = find_residential_cell(obs);
        let priority = (obs.spatial.population.housing_pressure - 1.0).min(2.0);
        actions.push(BuildingAction {
            decision_type: DecisionType::Placement,
            tier: DecisionTier::Strategic,
            action: ActionPayload::PlaceBuilding {
                building_type: BuildingType::House,
                grid_cell: cell,
            },
            priority,
            reasoning_tag: bi_tags::HOUSING_PRESSURE,
        });
    }

    // Rule 3: Fire destroyed wood cluster (check memory for fire events) → zone for stone rebuild.
    let fire_events: Vec<_> = obs
        .memory
        .short_term
        .iter()
        .filter(|e| {
            e.kind == ConstructionEventKind::FireSpread
                || e.kind == ConstructionEventKind::BuildingDestroyed
        })
        .collect();
    if !fire_events.is_empty() {
        // Find the most severe fire location.
        if let Some(worst) = fire_events.iter().max_by(|a, b| {
            a.severity.partial_cmp(&b.severity).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            actions.push(BuildingAction {
                decision_type: DecisionType::ZoneComposition,
                tier: DecisionTier::Strategic,
                action: ActionPayload::SetZone {
                    grid_cell: worst.location,
                    zone: "residential".to_string(), // stone rebuild zone
                },
                priority: worst.severity,
                reasoning_tag: bi_tags::FIRE_RECOVERY,
            });
        }
    }

    // Rule 4: High-value NPC unprotected + threat > 0.5 → defensive structure near them.
    for npc in &obs.high_value_npcs {
        if npc.protection_priority < 0.5 {
            continue;
        }
        let max_threat = obs
            .challenges
            .iter()
            .map(|c| c.severity)
            .fold(0.0f32, f32::max);
        if max_threat > 0.5 {
            let cell = (npc.position.0 as u16, npc.position.1 as u16);
            // Place a watchtower near them for protection.
            actions.push(BuildingAction {
                decision_type: DecisionType::Placement,
                tier: DecisionTier::Strategic,
                action: ActionPayload::PlaceBuilding {
                    building_type: BuildingType::Watchtower,
                    grid_cell: offset_cell(cell, 1, 0),
                },
                priority: npc.protection_priority * max_threat,
                reasoning_tag: bi_tags::LEADER_PROTECTION,
            });
        }
    }

    // Rule 5: High-level combat NPCs + active military threat + low garrison coverage
    //         → place barracks toward weakest defensive segment.
    let has_military_threat = obs.challenges.iter().any(|c| {
        c.category == ChallengeCategory::Military && c.severity > 0.3
    });
    let garrison_coverage = obs.spatial.garrison.coverage_map.iter()
        .copied().fold(0.0f32, f32::max);
    if has_military_threat && garrison_coverage < 0.5 {
        let combat_npcs: Vec<_> = obs
            .friendly_roster
            .iter()
            .filter(|u| u.combat_effectiveness > 5.0 && u.level >= 3)
            .collect();
        if combat_npcs.len() >= 2 {
            if let Some(weak_cell) = weakest_defensive_cell(obs) {
                actions.push(BuildingAction {
                    decision_type: DecisionType::Placement,
                    tier: DecisionTier::Strategic,
                    action: ActionPayload::PlaceBuilding {
                        building_type: BuildingType::Barracks,
                        grid_cell: weak_cell,
                    },
                    priority: 0.6,
                    reasoning_tag: bi_tags::GARRISON_SYNERGY,
                });
            }
        }
    }

    // Rule 6: Seasonal deadline approaching + insufficient storage → prioritize storage.
    for ch in &obs.challenges {
        if let Some(deadline) = ch.deadline_tick {
            let ticks_remaining = deadline.saturating_sub(obs.tick);
            if ticks_remaining < 500 && obs.spatial.economic.storage_utilization > 0.8 {
                let cell = find_empty_cell_near_center(obs);
                actions.push(BuildingAction {
                    decision_type: DecisionType::Placement,
                    tier: DecisionTier::Strategic,
                    action: ActionPayload::PlaceBuilding {
                        building_type: BuildingType::Warehouse,
                        grid_cell: cell,
                    },
                    priority: 0.8,
                    reasoning_tag: bi_tags::SEASONAL_PREP,
                });
            }
        }
    }

    // Rule 7: Disconnected components → routing action to connect them.
    if obs.spatial.connectivity.connected_components > 1 {
        // Connect the two nearest components via a path between chokepoints.
        let path_endpoints = if obs.spatial.connectivity.chokepoints.len() >= 2 {
            (
                obs.spatial.connectivity.chokepoints[0],
                obs.spatial.connectivity.chokepoints[1],
            )
        } else {
            // Fallback: connect center to nearest disconnected key building.
            let center = settlement_center(obs);
            let center_cell = (center.0 as u16, center.1 as u16);
            // Use the far endpoint of the first disconnected path as target.
            let target = obs
                .spatial
                .connectivity
                .key_building_paths
                .iter()
                .filter(|p| !p.path_exists || p.distance > 50.0) // unreachable or far
                .next()
                .map(|p| (p.id_b as u16, 0u16)) // approximate from building ID
                .unwrap_or(offset_cell(center_cell, 5, 0));
            (center_cell, target)
        };
        actions.push(BuildingAction {
            decision_type: DecisionType::Routing,
            tier: DecisionTier::Strategic,
            action: ActionPayload::RouteRoad {
                waypoints: vec![path_endpoints.0, path_endpoints.1],
            },
            priority: 0.7,
            reasoning_tag: bi_tags::TERRAIN_ADAPT,
        });
    }

    // Rule 8: Deadline pressure with multiple pending builds → prioritization action.
    {
        let has_deadline = obs
            .challenges
            .iter()
            .any(|c| c.deadline_tick.map_or(false, |d| d.saturating_sub(obs.tick) < 300));
        let has_housing_need = obs.spatial.population.housing_pressure > 1.0;
        let has_defense_need = obs.challenges.iter().any(|c| c.severity > 0.5);
        // If both housing and defense are needed under time pressure, emit prioritization.
        if has_deadline && has_housing_need && has_defense_need {
            // Defense before housing when under active threat.
            let max_severity = obs.challenges.iter().map(|c| c.severity).fold(0.0f32, f32::max);
            let defense_first = max_severity > obs.spatial.population.housing_pressure - 1.0;
            // Emit prioritization as a high-priority build order.
            // Use SetBuildPriority for the most urgent building category.
            actions.push(BuildingAction {
                decision_type: DecisionType::Prioritization,
                tier: DecisionTier::Strategic,
                action: ActionPayload::SetBuildPriority {
                    building_id: 0, // settlement-wide priority
                    priority: if defense_first { 1.0 } else { 0.8 },
                },
                priority: 0.9,
                reasoning_tag: bi_tags::SEASONAL_PREP,
            });
        }
    }

    // Rule 9: Severely damaged buildings → demolish and rebuild.
    for event in obs.memory.short_term.iter() {
        if event.kind == ConstructionEventKind::BuildingDestroyed && event.severity > 0.7 {
            actions.push(BuildingAction {
                decision_type: DecisionType::Demolition,
                tier: DecisionTier::Strategic,
                action: ActionPayload::Demolish {
                    building_id: event.source_entity.unwrap_or(0),
                },
                priority: event.severity * 0.6,
                reasoning_tag: bi_tags::FIRE_RECOVERY,
            });
        }
    }

    // Deduplicate placement actions by (building_type, grid_cell), keeping highest priority.
    dedup_placement_actions(&mut actions);

    actions
}

/// Remove duplicate PlaceBuilding actions at the same (building_type, grid_cell),
/// keeping only the one with the highest priority.
fn dedup_placement_actions(actions: &mut Vec<BuildingAction>) {
    let mut seen = std::collections::HashMap::<(u16, u16, u16), usize>::new(); // (col, row, btype_disc) → index
    let mut to_remove = Vec::new();
    for (i, a) in actions.iter().enumerate() {
        if let ActionPayload::PlaceBuilding { building_type, grid_cell } = &a.action {
            let key = (grid_cell.0, grid_cell.1, *building_type as u16);
            if let Some(&prev_idx) = seen.get(&key) {
                // Keep the one with higher priority.
                if a.priority > actions[prev_idx].priority {
                    to_remove.push(prev_idx);
                    seen.insert(key, i);
                } else {
                    to_remove.push(i);
                }
            } else {
                seen.insert(key, i);
            }
        }
    }
    to_remove.sort_unstable();
    for i in to_remove.into_iter().rev() {
        actions.remove(i);
    }
}

// ---------------------------------------------------------------------------
// Strategic utility layer (Chebyshev scalarization)
// ---------------------------------------------------------------------------

fn strategic_utility_layer(
    obs: &BuildingObservation,
    weights: &StrategicWeights,
    rule_actions: &[BuildingAction],
) -> Vec<BuildingAction> {
    // Reference point: ideal per-objective outcomes estimated from observation context.
    // R*_i represents the best achievable score for each objective given current state.
    let r_star = compute_ideal_point(obs);

    // Candidate building types to consider at each cell.
    // Must be a subset of PLACEABLE_TYPES from env_config.
    let candidate_types = [
        BuildingType::Watchtower,
        BuildingType::Barracks,
        BuildingType::House,
        BuildingType::Warehouse,
        BuildingType::Market,
        BuildingType::GuildHall,
        BuildingType::Forge,
        BuildingType::Farm,
        BuildingType::Inn,
    ];

    // Collect cells already used by rule actions so we don't double-place.
    let rule_cells: Vec<(u16, u16)> = rule_actions
        .iter()
        .filter_map(|a| match &a.action {
            ActionPayload::PlaceBuilding { grid_cell, .. } => Some(*grid_cell),
            _ => None,
        })
        .collect();

    // Generate candidate cells: empty cells from the grid.
    // We scan the spatial features for available positions.
    let candidate_cells = generate_candidate_cells(obs, &rule_cells);

    let weight_arr = [
        weights.threat_mitigation,
        weights.economic_value,
        weights.population_relief,
        weights.connectivity_improve,
        weights.resource_feasibility,
        weights.workforce_match,
        weights.level_appropriateness,
        weights.garrison_synergy,
    ];

    struct Candidate {
        building_type: BuildingType,
        cell: (u16, u16),
        score: f32,
        reasoning_tag: u32,
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    for &cell in &candidate_cells {
        for &btype in &candidate_types {
            // Estimate impact on each objective dimension.
            let impact = estimate_strategic_impact(obs, btype, cell);

            // Chebyshev scalarization: score = -max_i(w_i * |predicted_R_i - R*_i| / R*_i)
            let mut max_deviation = 0.0f32;
            for i in 0..8 {
                let deviation = weight_arr[i] * (impact[i] - r_star[i]).abs() / r_star[i];
                if deviation > max_deviation {
                    max_deviation = deviation;
                }
            }
            let score = -max_deviation;

            let reasoning_tag = reasoning_for_building(btype);

            candidates.push(Candidate {
                building_type: btype,
                cell,
                score,
                reasoning_tag,
            });
        }
    }

    // Sort by score descending (higher = better). Tie-break: lower grid-cell index.
    candidates.sort_by(|a, b| {
        let cmp = b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal);
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
        a.cell.cmp(&b.cell)
    });

    // Deduplicate: only one action per cell.
    let mut used_cells = std::collections::HashSet::new();
    let mut results = Vec::new();
    for c in &candidates {
        if used_cells.contains(&c.cell) {
            continue;
        }
        used_cells.insert(c.cell);

        // Confidence = gap between this score and second-best at this cell.
        let second_best = candidates
            .iter()
            .filter(|o| o.cell == c.cell && o.building_type != c.building_type)
            .map(|o| o.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let confidence = if second_best.is_finite() {
            (c.score - second_best).max(0.0)
        } else {
            0.5
        };

        results.push(BuildingAction {
            decision_type: DecisionType::Placement,
            tier: DecisionTier::Strategic,
            action: ActionPayload::PlaceBuilding {
                building_type: c.building_type,
                grid_cell: c.cell,
            },
            priority: (c.score + 1.0).max(0.0) * confidence.max(0.1),
            reasoning_tag: c.reasoning_tag,
        });

        if results.len() >= 8 {
            break;
        }
    }

    results
}

/// Estimate impact of placing `building_type` at `cell` on each strategic objective.
/// Returns [threat_mitigation, economic_value, population_relief, connectivity_improve,
///          resource_feasibility, workforce_match, level_appropriateness, garrison_synergy].
fn estimate_strategic_impact(
    obs: &BuildingObservation,
    building_type: BuildingType,
    cell: (u16, u16),
) -> [f32; 8] {
    let mut impact = [0.5f32; 8]; // baseline neutral impact

    let has_threat = obs.challenges.iter().any(|c| c.severity > 0.3);

    match building_type {
        BuildingType::Watchtower => {
            impact[0] = if has_threat { 0.9 } else { 0.6 }; // threat_mitigation
            impact[1] = 0.3;
            impact[4] = 0.3;
            impact[7] = if has_ranged_garrison(obs) { 0.95 } else { 0.6 };
        }
        BuildingType::Barracks => {
            impact[0] = 0.7;
            impact[1] = 0.3;
            impact[5] = 0.7; // workforce_match (military)
            impact[7] = 0.85; // garrison_synergy
        }
        BuildingType::House => {
            impact[0] = 0.3;
            impact[1] = 0.4;
            impact[2] = if obs.spatial.population.housing_pressure > 1.0 {
                0.95
            } else {
                0.5
            }; // population_relief
            impact[3] = 0.6; // connectivity
            impact[4] = 0.6;
        }
        BuildingType::Warehouse => {
            impact[1] = 0.7;
            impact[3] = 0.6;
            impact[4] = if obs.spatial.economic.storage_utilization > 0.7 {
                0.9
            } else {
                0.4
            };
        }
        BuildingType::Market => {
            impact[1] = 0.85; // economic_value
            impact[3] = 0.7; // connectivity
            impact[6] = level_appropriateness(obs, building_type);
        }
        BuildingType::GuildHall => {
            impact[1] = 0.7;
            impact[5] = 0.8; // workforce_match
            impact[6] = level_appropriateness(obs, building_type);
        }
        BuildingType::Forge => {
            impact[1] = 0.7;
            impact[4] = 0.5;
            impact[5] = 0.7;
            impact[6] = level_appropriateness(obs, building_type);
        }
        BuildingType::Farm => {
            impact[1] = 0.6; // economic (food production)
            impact[2] = 0.6; // population_relief (food capacity)
            impact[4] = 0.7; // resource_feasibility (low cost)
            impact[5] = 0.6; // workforce_match
        }
        BuildingType::Inn => {
            impact[1] = 0.6; // economic (trade income)
            impact[2] = 0.5; // population_relief (temporary housing)
            impact[3] = 0.7; // connectivity (social hub)
        }
        _ => {}
    }

    // Adjust for terrain: flood risk cells penalize non-defensive buildings.
    let in_flood = obs
        .spatial
        .environmental
        .flood_risk_cells
        .iter()
        .any(|&(c, r)| c == cell.0 && r == cell.1);
    if in_flood && !matches!(building_type, BuildingType::Watchtower | BuildingType::Barracks) {
        impact[4] *= 0.5; // resource_feasibility tanks in flood zone
    }

    // Adjust for fire risk: placing wood buildings in fire clusters is bad.
    let in_fire_cluster = obs
        .spatial
        .environmental
        .fire_risk_clusters
        .iter()
        .any(|fc| fc.cells.iter().any(|&(c, r)| c == cell.0 && r == cell.1));
    if in_fire_cluster && matches!(building_type, BuildingType::House) {
        impact[4] *= 0.6;
    }

    // Position-dependent adjustment: distance from settlement center affects value.
    let center = settlement_center(obs);
    let dx = cell.0 as f32 - center.0;
    let dy = cell.1 as f32 - center.1;
    let dist = (dx * dx + dy * dy).sqrt();
    let near_center = dist < 8.0;
    let near_perimeter = is_near_perimeter(obs, cell);

    match building_type {
        // Economic buildings benefit from central placement.
        BuildingType::Market | BuildingType::GuildHall | BuildingType::Forge
        | BuildingType::Warehouse | BuildingType::Inn => {
            if near_center {
                impact[3] += 0.1; // connectivity boost
            } else {
                impact[3] -= 0.1;
            }
        }
        // Defensive buildings benefit from perimeter placement.
        BuildingType::Watchtower | BuildingType::Barracks => {
            if near_perimeter {
                impact[0] += 0.1; // threat mitigation boost
                impact[7] += 0.1; // garrison synergy boost
            } else {
                impact[0] -= 0.1;
            }
        }
        // Residential buildings: moderate distance (not on perimeter, not crammed at center).
        BuildingType::House | BuildingType::Longhouse | BuildingType::Manor => {
            if dist > 3.0 && dist < 15.0 {
                impact[2] += 0.1; // population relief
            }
        }
        _ => {}
    }

    impact
}

// ---------------------------------------------------------------------------
// Structural layer implementation
// ---------------------------------------------------------------------------

fn structural_for_building(
    obs: &BuildingObservation,
    weights: &StructuralWeights,
    building_type: BuildingType,
    cell: (u16, u16),
    building_id: u32,
    segment_id: u32,
) -> Vec<BuildingAction> {
    let mut actions = Vec::new();

    // Gather threat context.
    let max_jump_height = max_enemy_jump_height(obs);
    let has_siege = any_enemy_has_siege(obs);
    let in_flood = obs
        .spatial
        .environmental
        .flood_risk_cells
        .iter()
        .any(|&(c, r)| c == cell.0 && r == cell.1);
    let adjacent_wood = obs
        .spatial
        .environmental
        .fire_risk_clusters
        .iter()
        .any(|fc| {
            fc.cells.iter().any(|&(c, r)| {
                (c as i32 - cell.0 as i32).unsigned_abs() <= 1
                    && (r as i32 - cell.1 as i32).unsigned_abs() <= 1
            })
        });
    let has_archer = has_ranged_garrison(obs);
    let is_leader_building = obs
        .high_value_npcs
        .iter()
        .any(|npc| {
            let dx = (npc.position.0 - cell.0 as f32).abs();
            let dy = (npc.position.1 - cell.1 as f32).abs();
            dx <= 2.0 && dy <= 2.0
        });
    let threat_dir = primary_threat_direction(obs);

    // -----------------------------------------------------------------------
    // Rule layer: counter enemy capabilities
    // -----------------------------------------------------------------------

    match building_type {
        BuildingType::Wall | BuildingType::Gate => {
            // Wall height >= jump_height + 2
            let min_height = if max_jump_height > 0 {
                (max_jump_height + 2).min(10)
            } else {
                3
            };
            // Wall thickness >= 2 if siege
            let min_thickness = if has_siege { 2 } else { 1 };

            let mut features = WallFeatures::default();
            if has_siege {
                features.buttressed = true;
            }
            if has_archer {
                features.crenellations = true;
            }

            // Material: stone if siege or high threat, wood otherwise.
            let material = choose_wall_material(obs, has_siege);

            let reasoning = if max_jump_height > 0 {
                bi_tags::JUMP_COUNTER
            } else if has_siege {
                bi_tags::SIEGE_COUNTER
            } else {
                bi_tags::THREAT_PROXIMITY
            };

            actions.push(BuildingAction {
                decision_type: DecisionType::WallComposition,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetWallSpec {
                    segment_id,
                    height: min_height,
                    thickness: min_thickness,
                    material,
                    features,
                },
                priority: 0.8,
                reasoning_tag: reasoning,
            });

            // Arrow slits facing threat direction.
            if has_archer {
                if let Some(dir) = threat_dir {
                    actions.push(BuildingAction {
                        decision_type: DecisionType::Openings,
                        tier: DecisionTier::Structural,
                        action: ActionPayload::SetOpenings {
                            building_id,
                            openings: vec![OpeningSpec {
                                opening_type: OpeningType::ArrowSlit,
                                wall_facing: dir,
                                count: 3,
                            }],
                        },
                        priority: 0.6,
                        reasoning_tag: bi_tags::GARRISON_SYNERGY,
                    });
                }
            }
        }

        BuildingType::House | BuildingType::Longhouse | BuildingType::Manor => {
            // Foundation.
            let (foundation_type, depth) = if in_flood {
                (FoundationType::Raised, 2)
            } else {
                (FoundationType::Slab, 1)
            };
            actions.push(BuildingAction {
                decision_type: DecisionType::Foundation,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFoundation {
                    building_id,
                    foundation_type,
                    depth,
                },
                priority: if in_flood { 0.7 } else { 0.3 },
                reasoning_tag: if in_flood {
                    bi_tags::FLOOD_PREVENTION
                } else {
                    bi_tags::TERRAIN_ADAPT
                },
            });

            // Material upgrade if adjacent to wood cluster (fire break).
            if adjacent_wood {
                actions.push(BuildingAction {
                    decision_type: DecisionType::MaterialSelection,
                    tier: DecisionTier::Structural,
                    action: ActionPayload::SetMaterial {
                        building_id,
                        component: BuildingComponent::NorthWall,
                        material: BuildMaterial::Stone,
                    },
                    priority: 0.6,
                    reasoning_tag: bi_tags::WOOD_BURNS,
                });
            }

            // Roof.
            let roof_type = if in_flood {
                RoofType::Pitched // steep for drainage
            } else {
                RoofType::Pitched
            };
            actions.push(BuildingAction {
                decision_type: DecisionType::RoofDesign,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetRoofSpec {
                    building_id,
                    roof_type,
                    material: if adjacent_wood {
                        BuildMaterial::Stone
                    } else {
                        BuildMaterial::Wood
                    },
                },
                priority: 0.3,
                reasoning_tag: bi_tags::TERRAIN_ADAPT,
            });

            // Footprint.
            actions.push(BuildingAction {
                decision_type: DecisionType::FootprintGeometry,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFootprint {
                    building_id,
                    shape: FootprintShape::Rectangular,
                    dimensions: (3, 3),
                },
                priority: 0.2,
                reasoning_tag: bi_tags::TERRAIN_ADAPT,
            });
        }

        BuildingType::Barracks => {
            // Barracks: training yard, armory, defensible.
            actions.push(BuildingAction {
                decision_type: DecisionType::FootprintGeometry,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFootprint {
                    building_id,
                    shape: FootprintShape::UShape,
                    dimensions: (5, 4),
                },
                priority: 0.4,
                reasoning_tag: bi_tags::GARRISON_SYNERGY,
            });

            actions.push(BuildingAction {
                decision_type: DecisionType::InteriorFlow,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetInteriorLayout {
                    building_id,
                    rooms: vec![
                        RoomPlacement {
                            kind: RoomKindBi::TrainingYard,
                            offset: (0.0, 0.0),
                            size: (3.0, 3.0),
                        },
                        RoomPlacement {
                            kind: RoomKindBi::Armory,
                            offset: (3.0, 0.0),
                            size: (2.0, 2.0),
                        },
                        RoomPlacement {
                            kind: RoomKindBi::Bedroom,
                            offset: (3.0, 2.0),
                            size: (2.0, 2.0),
                        },
                    ],
                },
                priority: 0.4,
                reasoning_tag: bi_tags::GARRISON_SYNERGY,
            });

            // Reinforced foundation.
            actions.push(BuildingAction {
                decision_type: DecisionType::Foundation,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFoundation {
                    building_id,
                    foundation_type: if in_flood {
                        FoundationType::Raised
                    } else {
                        FoundationType::Deep
                    },
                    depth: 2,
                },
                priority: 0.3,
                reasoning_tag: bi_tags::TERRAIN_ADAPT,
            });

            // Walkable roof for archers.
            if has_archer {
                actions.push(BuildingAction {
                    decision_type: DecisionType::RoofDesign,
                    tier: DecisionTier::Structural,
                    action: ActionPayload::SetRoofSpec {
                        building_id,
                        roof_type: RoofType::Walkable,
                        material: BuildMaterial::Stone,
                    },
                    priority: 0.5,
                    reasoning_tag: bi_tags::GARRISON_SYNERGY,
                });
            }
        }

        BuildingType::Watchtower => {
            // Tall, narrow, arrow slits on all sides.
            actions.push(BuildingAction {
                decision_type: DecisionType::VerticalDesign,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetVertical {
                    building_id,
                    stories: 3,
                    has_basement: false,
                    elevation: 0,
                },
                priority: 0.5,
                reasoning_tag: bi_tags::THREAT_PROXIMITY,
            });

            actions.push(BuildingAction {
                decision_type: DecisionType::FootprintGeometry,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFootprint {
                    building_id,
                    shape: FootprintShape::Circular,
                    dimensions: (2, 2),
                },
                priority: 0.3,
                reasoning_tag: bi_tags::THREAT_PROXIMITY,
            });

            // Arrow slits concentrated toward threat direction, fewer on safe sides.
            let mut openings = Vec::new();
            let threat_dirs = if let Some(td) = threat_dir {
                // 3 slits on threat-facing walls, 1 on others.
                vec![
                    (td, 3u8),
                    (opposite_direction(td), 1),
                    (rotate_cw(td), 2),
                    (rotate_ccw(td), 2),
                ]
            } else {
                // No threat — uniform 2 per face.
                vec![
                    (Direction::North, 2),
                    (Direction::East, 2),
                    (Direction::South, 2),
                    (Direction::West, 2),
                ]
            };
            for (dir, count) in threat_dirs {
                openings.push(OpeningSpec {
                    opening_type: OpeningType::ArrowSlit,
                    wall_facing: dir,
                    count,
                });
            }
            actions.push(BuildingAction {
                decision_type: DecisionType::Openings,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetOpenings {
                    building_id,
                    openings,
                },
                priority: 0.5,
                reasoning_tag: bi_tags::GARRISON_SYNERGY,
            });

            // Archer platform interior.
            actions.push(BuildingAction {
                decision_type: DecisionType::InteriorFlow,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetInteriorLayout {
                    building_id,
                    rooms: vec![RoomPlacement {
                        kind: RoomKindBi::ArcherPlatform,
                        offset: (0.0, 0.0),
                        size: (2.0, 2.0),
                    }],
                },
                priority: 0.4,
                reasoning_tag: bi_tags::GARRISON_SYNERGY,
            });
        }

        BuildingType::Warehouse => {
            // Large footprint, reinforced.
            actions.push(BuildingAction {
                decision_type: DecisionType::FootprintGeometry,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFootprint {
                    building_id,
                    shape: FootprintShape::Rectangular,
                    dimensions: (5, 4),
                },
                priority: 0.3,
                reasoning_tag: bi_tags::RESOURCE_SCARCITY,
            });

            // Foundation (raised if flood risk).
            actions.push(BuildingAction {
                decision_type: DecisionType::Foundation,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFoundation {
                    building_id,
                    foundation_type: if in_flood {
                        FoundationType::Raised
                    } else {
                        FoundationType::Slab
                    },
                    depth: if in_flood { 2 } else { 1 },
                },
                priority: if in_flood { 0.6 } else { 0.2 },
                reasoning_tag: if in_flood {
                    bi_tags::FLOOD_PREVENTION
                } else {
                    bi_tags::TERRAIN_ADAPT
                },
            });

            // Storeroom interior.
            actions.push(BuildingAction {
                decision_type: DecisionType::InteriorFlow,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetInteriorLayout {
                    building_id,
                    rooms: vec![
                        RoomPlacement {
                            kind: RoomKindBi::Storeroom,
                            offset: (0.0, 0.0),
                            size: (4.0, 3.0),
                        },
                        RoomPlacement {
                            kind: RoomKindBi::Entrance,
                            offset: (4.0, 0.0),
                            size: (1.0, 2.0),
                        },
                    ],
                },
                priority: 0.3,
                reasoning_tag: bi_tags::RESOURCE_SCARCITY,
            });
        }

        _ => {
            // Generic building: foundation + footprint + material.
            actions.push(BuildingAction {
                decision_type: DecisionType::Foundation,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFoundation {
                    building_id,
                    foundation_type: if in_flood {
                        FoundationType::Raised
                    } else {
                        FoundationType::Slab
                    },
                    depth: 1,
                },
                priority: 0.2,
                reasoning_tag: bi_tags::TERRAIN_ADAPT,
            });

            actions.push(BuildingAction {
                decision_type: DecisionType::FootprintGeometry,
                tier: DecisionTier::Structural,
                action: ActionPayload::SetFootprint {
                    building_id,
                    shape: FootprintShape::Rectangular,
                    dimensions: (3, 3),
                },
                priority: 0.2,
                reasoning_tag: bi_tags::TERRAIN_ADAPT,
            });
        }
    }

    // Leader residence special: escape route + reinforced door.
    if is_leader_building {
        actions.push(BuildingAction {
            decision_type: DecisionType::InteriorFlow,
            tier: DecisionTier::Structural,
            action: ActionPayload::SetInteriorLayout {
                building_id,
                rooms: vec![
                    RoomPlacement {
                        kind: RoomKindBi::SafeRoom,
                        offset: (0.0, 0.0),
                        size: (2.0, 2.0),
                    },
                    RoomPlacement {
                        kind: RoomKindBi::EscapeRoute,
                        offset: (2.0, 0.0),
                        size: (1.0, 2.0),
                    },
                ],
            },
            priority: 0.7,
            reasoning_tag: bi_tags::LEADER_PROTECTION,
        });

        // Reinforced door facing away from threat.
        let safe_dir = threat_dir
            .map(|d| opposite_direction(d))
            .unwrap_or(Direction::South);
        actions.push(BuildingAction {
            decision_type: DecisionType::Openings,
            tier: DecisionTier::Structural,
            action: ActionPayload::SetOpenings {
                building_id,
                openings: vec![OpeningSpec {
                    opening_type: OpeningType::Door,
                    wall_facing: safe_dir,
                    count: 1,
                }],
            },
            priority: 0.6,
            reasoning_tag: bi_tags::LEADER_PROTECTION,
        });
    }

    // -----------------------------------------------------------------------
    // Utility layer: Chebyshev over structural options
    // -----------------------------------------------------------------------

    // Score the existing actions via structural utility and adjust priorities.
    let weight_arr = [
        weights.defensive_value,
        weights.cost_efficiency,
        weights.construction_time,
        weights.material_availability,
        weights.terrain_fit,
        weights.expansion_potential,
        weights.unit_synergy,
        weights.garrison_amplification,
    ];

    for action in &mut actions {
        let impact = estimate_structural_impact(obs, action, cell, building_type);

        // Chebyshev: -max(w_i * |impact_i - 1.0|)
        let mut max_dev = 0.0f32;
        for i in 0..8 {
            let dev = weight_arr[i] * (impact[i] - 1.0).abs();
            if dev > max_dev {
                max_dev = dev;
            }
        }
        let utility_score = -max_dev;
        // Blend utility into priority: 70% rule, 30% utility.
        action.priority = action.priority * 0.7 + (utility_score + 1.0).max(0.0) * 0.3;
    }

    actions
}

/// Estimate structural impact on each dimension.
/// Returns [defensive_value, cost_efficiency, construction_time, material_availability,
///          terrain_fit, expansion_potential, unit_synergy, garrison_amplification].
fn estimate_structural_impact(
    obs: &BuildingObservation,
    action: &BuildingAction,
    cell: (u16, u16),
    building_type: BuildingType,
) -> [f32; 8] {
    let mut impact = [0.7f32; 8];

    let in_flood = obs
        .spatial
        .environmental
        .flood_risk_cells
        .iter()
        .any(|&(c, r)| c == cell.0 && r == cell.1);

    match &action.action {
        ActionPayload::SetWallSpec {
            height,
            thickness,
            material,
            features,
            ..
        } => {
            let material_factor = match material {
                BuildMaterial::Thatch => 0.2,
                BuildMaterial::Wood => 0.5,
                BuildMaterial::Brick => 0.7,
                BuildMaterial::Stone => 0.9,
                BuildMaterial::Iron => 1.0,
            };
            impact[0] = (*height as f32 / 5.0).min(1.0) * material_factor; // defensive
            impact[1] = 1.0 - material_factor * 0.5; // cost (expensive = low)
            impact[2] = 1.0 - (*thickness as f32 * 0.15); // construction time
            impact[3] = material_availability(obs, *material);
            impact[4] = if in_flood && *height < 3 { 0.4 } else { 0.8 };
            impact[6] = if features.crenellations { 0.9 } else { 0.6 }; // unit synergy
            impact[7] = if features.buttressed { 0.9 } else { 0.7 }; // garrison amp
        }
        ActionPayload::SetFoundation {
            foundation_type, depth, ..
        } => {
            impact[0] = 0.6 + *depth as f32 * 0.1;
            impact[1] = 1.0 - *depth as f32 * 0.15;
            impact[2] = 1.0 - *depth as f32 * 0.1;
            impact[4] = match foundation_type {
                FoundationType::Raised if in_flood => 0.95,
                FoundationType::Pilings if in_flood => 0.9,
                _ if in_flood => 0.4,
                _ => 0.8,
            };
            impact[5] = 0.7; // expansion potential
        }
        ActionPayload::SetRoofSpec { roof_type, material, .. } => {
            impact[0] = match roof_type {
                RoofType::Reinforced => 0.9,
                RoofType::Walkable => 0.85,
                _ => 0.6,
            };
            impact[1] = match material {
                BuildMaterial::Wood => 0.8,
                BuildMaterial::Thatch => 0.9,
                _ => 0.5,
            };
            impact[6] = if *roof_type == RoofType::Walkable { 0.9 } else { 0.5 };
        }
        ActionPayload::SetOpenings { openings, .. } => {
            let has_slits = openings
                .iter()
                .any(|o| o.opening_type == OpeningType::ArrowSlit);
            impact[0] = if has_slits { 0.8 } else { 0.6 };
            impact[6] = if has_slits { 0.95 } else { 0.5 };
            impact[7] = if has_slits { 0.9 } else { 0.5 };
        }
        ActionPayload::SetFootprint { dimensions, .. } => {
            let area = dimensions.0 as f32 * dimensions.1 as f32;
            impact[1] = (1.0 - area / 30.0).max(0.3);
            impact[2] = (1.0 - area / 40.0).max(0.3);
            impact[5] = if area < 12.0 { 0.8 } else { 0.5 }; // expansion
        }
        ActionPayload::SetMaterial { material, .. } => {
            impact[0] = match material {
                BuildMaterial::Stone => 0.9,
                BuildMaterial::Iron => 1.0,
                BuildMaterial::Brick => 0.7,
                BuildMaterial::Wood => 0.5,
                BuildMaterial::Thatch => 0.3,
            };
            impact[1] = match material {
                BuildMaterial::Thatch => 0.95,
                BuildMaterial::Wood => 0.8,
                BuildMaterial::Brick => 0.6,
                BuildMaterial::Stone => 0.4,
                BuildMaterial::Iron => 0.2,
            };
            impact[3] = material_availability(obs, *material);
        }
        _ => {}
    }

    // Garrison amplification: military buildings get bonus.
    if matches!(
        building_type,
        BuildingType::Barracks | BuildingType::Watchtower
    ) {
        impact[7] = impact[7].max(0.8);
    }

    impact
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extract grid cell from an action payload. Returns (0,0) for non-spatial actions.
fn action_grid_cell(action: &ActionPayload) -> (u16, u16) {
    match action {
        ActionPayload::PlaceBuilding { grid_cell, .. } => *grid_cell,
        ActionPayload::SetZone { grid_cell, .. } => *grid_cell,
        _ => (0, 0),
    }
}

/// Check if existing wall segments cover a given threat direction.
fn wall_covers_direction(obs: &BuildingObservation, dir: (f32, f32)) -> bool {
    // Compute settlement center from wall segments.
    let center = settlement_center(obs);

    let dir_angle = dir.1.atan2(dir.0);

    for seg in &obs.spatial.defensive.wall_segments {
        let seg_cx = (seg.start.0 as f32 + seg.end.0 as f32) / 2.0;
        let seg_cy = (seg.start.1 as f32 + seg.end.1 as f32) / 2.0;

        // Compute segment direction relative to settlement center.
        let rel_x = seg_cx - center.0;
        let rel_y = seg_cy - center.1;
        let seg_angle = rel_y.atan2(rel_x);

        let angle_diff = (seg_angle - dir_angle).abs();
        let angle_diff = if angle_diff > std::f32::consts::PI {
            2.0 * std::f32::consts::PI - angle_diff
        } else {
            angle_diff
        };

        if angle_diff < std::f32::consts::FRAC_PI_4 && seg.condition > 0.3 {
            return true;
        }
    }
    false
}

/// Convert a threat direction to a perimeter grid cell.
fn direction_to_perimeter_cell(obs: &BuildingObservation, dir: (f32, f32)) -> (u16, u16) {
    // Use wall segments to estimate grid bounds, or fall back to defaults.
    let max_col = obs
        .spatial
        .defensive
        .wall_segments
        .iter()
        .map(|s| s.end.0.max(s.start.0))
        .max()
        .unwrap_or(20);
    let max_row = obs
        .spatial
        .defensive
        .wall_segments
        .iter()
        .map(|s| s.end.1.max(s.start.1))
        .max()
        .unwrap_or(20);

    // Clamp to virtual grid bounds — wall segments can be in world coords > 63.
    let max_col = max_col.min(VGRID_MAX);
    let max_row = max_row.min(VGRID_MAX);

    let center_col = max_col / 2;
    let center_row = max_row / 2;

    // Project direction onto grid perimeter.
    let scale = (max_col as f32 / 2.0).max(max_row as f32 / 2.0);
    let col = (center_col as f32 + dir.0 * scale).clamp(0.0, max_col as f32) as u16;
    let row = (center_row as f32 + dir.1 * scale).clamp(0.0, max_row as f32) as u16;

    clamp_cell(col, row)
}

/// Find a good cell for residential placement, weighted away from industrial zone.
fn find_residential_cell(obs: &BuildingObservation) -> (u16, u16) {
    // If there are crowding hotspots, place nearby but not on top.
    if let Some(&hotspot) = obs.spatial.population.crowding_hotspots.first() {
        return offset_cell(hotspot, 2, 1);
    }
    find_empty_cell_near_center(obs)
}

/// Compute approximate settlement center from occupied cells, wall segments,
/// and elevation data. Occupied cells (buildings) are the strongest signal.
fn settlement_center(obs: &BuildingObservation) -> (f32, f32) {
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut count = 0u32;

    // Occupied cells (building positions) — strongest signal, weight 2x.
    for &(cx, cy) in &obs.spatial.occupied_cells {
        sum_x += cx as f32 * 2.0;
        sum_y += cy as f32 * 2.0;
        count += 2;
    }

    // NPC positions from friendly roster.
    for unit in &obs.friendly_roster {
        sum_x += unit.position.0;
        sum_y += unit.position.1;
        count += 1;
    }

    // Wall segment endpoints.
    for seg in &obs.spatial.defensive.wall_segments {
        sum_x += seg.start.0 as f32 + seg.end.0 as f32;
        sum_y += seg.start.1 as f32 + seg.end.1 as f32;
        count += 2;
    }

    for elev in &obs.spatial.environmental.elevation_map {
        sum_x += elev.cell.0 as f32;
        sum_y += elev.cell.1 as f32;
        count += 1;
    }
    if count > 0 {
        (sum_x / count as f32, sum_y / count as f32)
    } else {
        (32.0, 32.0)
    }
}

/// Find an empty cell near the settlement center.
/// Spirals outward from center to find a cell not already occupied by existing buildings.
fn find_empty_cell_near_center(obs: &BuildingObservation) -> (u16, u16) {
    let center = settlement_center(obs);
    let cx = center.0 as u16;
    let cy = center.1 as u16;

    // Use full occupied cell set from spatial features.
    let occupied: std::collections::HashSet<(u16, u16)> = obs
        .spatial
        .occupied_cells
        .iter()
        .copied()
        .collect();

    // Spiral outward from center.
    for radius in 0..20u16 {
        for dr in -(radius as i16)..=(radius as i16) {
            for dc in -(radius as i16)..=(radius as i16) {
                if dr.unsigned_abs() != radius && dc.unsigned_abs() != radius {
                    continue; // only check perimeter of this radius
                }
                let cell = clamp_cell(
                    (cx as i32 + dc as i32).max(0) as u16,
                    (cy as i32 + dr as i32).max(0) as u16,
                );
                if !occupied.contains(&cell) {
                    return cell;
                }
            }
        }
    }
    (cx, cy)
}

/// Offset a cell by (dc, dr), clamping to virtual grid bounds.
fn offset_cell(cell: (u16, u16), dc: i16, dr: i16) -> (u16, u16) {
    clamp_cell(
        (cell.0 as i32 + dc as i32).max(0) as u16,
        (cell.1 as i32 + dr as i32).max(0) as u16,
    )
}

/// Find the weakest segment of the defensive perimeter.
fn weakest_defensive_cell(obs: &BuildingObservation) -> Option<(u16, u16)> {
    if obs.spatial.garrison.coverage_map.is_empty() {
        return None;
    }
    // Find the perimeter index with lowest coverage.
    let mut min_val = f32::MAX;
    let mut min_idx = 0;
    for (i, &val) in obs.spatial.garrison.coverage_map.iter().enumerate() {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    // We don't have perimeter cell positions directly, so approximate from index.
    // Wall segments give us orientation.
    if let Some(seg) = obs.spatial.defensive.wall_segments.get(min_idx % obs.spatial.defensive.wall_segments.len().max(1)) {
        Some(seg.start)
    } else {
        Some(find_empty_cell_near_center(obs))
    }
}

/// Check if a cell is near the perimeter (within ~3 cells of a wall or edge).
fn is_near_perimeter(obs: &BuildingObservation, cell: (u16, u16)) -> bool {
    for seg in &obs.spatial.defensive.wall_segments {
        let dx = (cell.0 as i32 - seg.start.0 as i32).abs().min(
            (cell.0 as i32 - seg.end.0 as i32).abs(),
        );
        let dy = (cell.1 as i32 - seg.start.1 as i32).abs().min(
            (cell.1 as i32 - seg.end.1 as i32).abs(),
        );
        if dx <= 3 && dy <= 3 {
            return true;
        }
    }
    // Also consider cells near the grid edge.
    cell.0 <= 2 || cell.1 <= 2
}

/// Check if any garrison unit likely has ranged capability.
/// Heuristic: class_tag matches known ranged classes (archer, ranger, mage, crossbowman).
fn has_ranged_garrison(obs: &BuildingObservation) -> bool {
    use crate::world_sim::state::tag;
    const RANGED_CLASSES: &[u32] = &[
        tag(b"archer"),
        tag(b"ranger"),
        tag(b"mage"),
        tag(b"crossbowman"),
        tag(b"wizard"),
        tag(b"warlock"),
        tag(b"hunter"),
    ];
    obs.friendly_roster
        .iter()
        .any(|u| u.is_garrison && RANGED_CLASSES.contains(&u.class_tag))
}

/// Generate candidate cells for strategic placement.
fn generate_candidate_cells(
    obs: &BuildingObservation,
    exclude: &[(u16, u16)],
) -> Vec<(u16, u16)> {
    let mut cells = Vec::new();

    // Build occupied set from spatial features + rule exclusions.
    let occupied: std::collections::HashSet<(u16, u16)> = obs
        .spatial
        .occupied_cells
        .iter()
        .copied()
        .chain(exclude.iter().copied())
        .collect();

    // Estimate grid bounds from settlement extents, clamped to virtual grid.
    let mut min_col: u16 = u16::MAX;
    let mut min_row: u16 = u16::MAX;
    let mut max_col: u16 = 20;
    let mut max_row: u16 = 20;
    for &(cx, cy) in &obs.spatial.occupied_cells {
        min_col = min_col.min(cx);
        min_row = min_row.min(cy);
        max_col = max_col.max(cx + 3);
        max_row = max_row.max(cy + 3);
    }
    for seg in &obs.spatial.defensive.wall_segments {
        min_col = min_col.min(seg.start.0.min(seg.end.0));
        min_row = min_row.min(seg.start.1.min(seg.end.1));
        max_col = max_col.max(seg.end.0 + 2).max(seg.start.0 + 2);
        max_row = max_row.max(seg.end.1 + 2).max(seg.start.1 + 2);
    }
    // Clamp to virtual grid bounds.
    max_col = max_col.min(VGRID_MAX);
    max_row = max_row.min(VGRID_MAX);
    if min_col == u16::MAX { min_col = 0; }
    if min_row == u16::MAX { min_row = 0; }

    // Scan candidates in spiral order from settlement center outward,
    // stepping by 2 for denser coverage near the settlement.
    let center = settlement_center(obs);
    let cx = center.0 as i16;
    let cy = center.1 as i16;
    let stride = 2i16;

    for radius in 0..30i16 {
        for dr in -radius..=radius {
            for dc in -radius..=radius {
                if dr.abs() != radius && dc.abs() != radius {
                    continue; // only scan perimeter of this ring
                }
                let c = cx + dc * stride;
                let r = cy + dr * stride;
                if c < min_col as i16 || r < min_row as i16
                    || c > max_col as i16 || r > max_row as i16
                {
                    continue;
                }
                let cell = (c as u16, r as u16);
                if occupied.contains(&cell) {
                    continue;
                }
                cells.push(cell);
                if cells.len() >= 40 {
                    return cells;
                }
            }
        }
    }

    cells
}

/// Get maximum enemy jump height from all challenges.
fn max_enemy_jump_height(obs: &BuildingObservation) -> u8 {
    obs.challenges
        .iter()
        .flat_map(|c| &c.enemy_profiles)
        .filter(|p| p.can_jump)
        .map(|p| p.jump_height)
        .max()
        .unwrap_or(0)
}

/// Check if any enemy has siege capability.
fn any_enemy_has_siege(obs: &BuildingObservation) -> bool {
    obs.challenges
        .iter()
        .flat_map(|c| &c.enemy_profiles)
        .any(|p| p.has_siege)
}

/// Get the primary threat direction as a Direction enum.
fn primary_threat_direction(obs: &BuildingObservation) -> Option<Direction> {
    let dir = obs
        .challenges
        .iter()
        .filter_map(|c| c.direction)
        .max_by(|a, b| {
            let mag_a = a.0 * a.0 + a.1 * a.1;
            let mag_b = b.0 * b.0 + b.1 * b.1;
            mag_a.partial_cmp(&mag_b).unwrap_or(std::cmp::Ordering::Equal)
        })?;

    // Convert (f32, f32) direction to cardinal.
    Some(if dir.0.abs() > dir.1.abs() {
        if dir.0 > 0.0 {
            Direction::East
        } else {
            Direction::West
        }
    } else if dir.1 > 0.0 {
        Direction::South
    } else {
        Direction::North
    })
}

fn opposite_direction(dir: Direction) -> Direction {
    match dir {
        Direction::North => Direction::South,
        Direction::South => Direction::North,
        Direction::East => Direction::West,
        Direction::West => Direction::East,
    }
}

fn rotate_cw(dir: Direction) -> Direction {
    match dir {
        Direction::North => Direction::East,
        Direction::East => Direction::South,
        Direction::South => Direction::West,
        Direction::West => Direction::North,
    }
}

fn rotate_ccw(dir: Direction) -> Direction {
    match dir {
        Direction::North => Direction::West,
        Direction::West => Direction::South,
        Direction::South => Direction::East,
        Direction::East => Direction::North,
    }
}

/// Choose wall material based on threat context and resource availability.
fn choose_wall_material(obs: &BuildingObservation, has_siege: bool) -> BuildMaterial {
    if has_siege {
        // Need stone or better for siege resistance.
        if material_availability(obs, BuildMaterial::Stone) > 0.3 {
            return BuildMaterial::Stone;
        }
        return BuildMaterial::Brick;
    }
    // Default: best available that's reasonably stocked.
    if material_availability(obs, BuildMaterial::Stone) > 0.5 {
        BuildMaterial::Stone
    } else if material_availability(obs, BuildMaterial::Brick) > 0.5 {
        BuildMaterial::Brick
    } else {
        BuildMaterial::Wood
    }
}

/// Estimate material availability from stockpile levels (0.0 = none, 1.0 = abundant).
fn material_availability(obs: &BuildingObservation, material: BuildMaterial) -> f32 {
    // Map materials to stockpile indices (commodity module: FOOD=0, IRON=1, WOOD=2, ...).
    let idx = match material {
        BuildMaterial::Wood => 2,     // commodity::WOOD
        BuildMaterial::Thatch => 2,   // wood-derived
        BuildMaterial::Stone => 1,    // no stone commodity; approximate via IRON (masonry resource)
        BuildMaterial::Brick => 1,    // stone-adjacent
        BuildMaterial::Iron => 1,     // commodity::IRON
    };
    let stock = obs.spatial.economic.stockpiles.get(idx).copied().unwrap_or(0.0);
    (stock / 100.0).clamp(0.0, 1.0)
}

/// How appropriate is this building type for the settlement's current level?
fn level_appropriateness(obs: &BuildingObservation, building_type: BuildingType) -> f32 {
    let level = obs.settlement_level;
    let min_level = match building_type {
        BuildingType::GuildHall => 3,
        BuildingType::Market => 2,
        BuildingType::Forge => 2,
        BuildingType::Temple => 3,
        BuildingType::CourtHouse => 4,
        BuildingType::Library => 3,
        _ => 1,
    };
    if level >= min_level {
        1.0
    } else {
        0.3
    }
}

/// Map building type to a reasoning tag.
fn reasoning_for_building(building_type: BuildingType) -> u32 {
    match building_type {
        BuildingType::Wall | BuildingType::Gate => bi_tags::THREAT_PROXIMITY,
        BuildingType::Watchtower => bi_tags::THREAT_PROXIMITY,
        BuildingType::Barracks => bi_tags::GARRISON_SYNERGY,
        BuildingType::House | BuildingType::Longhouse | BuildingType::Manor => {
            bi_tags::HOUSING_PRESSURE
        }
        BuildingType::Warehouse => bi_tags::RESOURCE_SCARCITY,
        BuildingType::Market | BuildingType::TradePost => bi_tags::RESOURCE_SCARCITY,
        BuildingType::GuildHall => bi_tags::SPECIALIST_ACCESS,
        BuildingType::Forge | BuildingType::Workshop => bi_tags::UPGRADE_PATH,
        _ => bi_tags::TERRAIN_ADAPT,
    }
}
