//! Outcome scoring — evaluate how well building decisions performed.
//!
//! Scores are used to:
//! 1. Filter BC dataset quality (high-delta pairs are more valuable)
//! 2. Serve as future RL reward signal
//! 3. Compare oracle vs. no-build vs. random baselines

use serde::{Deserialize, Serialize};

use super::types::{
    ActionPayload, BuildingAction, BuildingObservation, Challenge, ChallengeCategory, DecisionType,
};
use crate::world_sim::state::{BuildingType, Entity, EntityKind, WorldState, WorldTeam};

// ---------------------------------------------------------------------------
// Score components
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutcomeScore {
    pub defensive: DefensiveScore,
    pub environmental: EnvironmentalScore,
    pub economic: EconomicScore,
    pub population: PopulationScore,
    pub spatial_quality: SpatialQualityScore,
    pub composite: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DefensiveScore {
    pub breach_count: u16,
    /// How far enemies penetrated (0.0 = perimeter only, 1.0 = reached center).
    pub breach_depth: f32,
    /// friendly_kills / enemy_kills (higher = better).
    pub casualty_ratio: f32,
    /// Fraction of high-value NPCs that survived (0.0–1.0).
    pub high_value_npc_survival: f32,
    /// Mean ticks between breach and first defender engagement.
    pub response_time_mean: f32,
    /// Ticks before defenses failed (u64::MAX if they held).
    pub hold_duration: u64,
    /// defensive_outcome / material_cost.
    pub resource_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentalScore {
    /// Buildings saved vs. counterfactual (no intervention).
    pub damage_prevented: f32,
    /// Ticks to return to pre-disaster functionality.
    pub recovery_time: u64,
    /// Did cascading failure occur? (0.0 = none, 1.0 = total).
    pub cascading_severity: f32,
    /// Water accumulation in occupied cells after rain.
    pub drainage_performance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EconomicScore {
    /// Trade volume / storage capacity.
    pub throughput: f32,
    /// Mean NPC travel time to workplace (lower = better).
    pub commute_efficiency: f32,
    /// Mean ticks from resource source to stockpile.
    pub resource_access_time: f32,
    /// Economic output / material invested.
    pub roi: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PopulationScore {
    /// unhoused_after / unhoused_before (lower = better, 0.0 = solved).
    pub housing_coverage_delta: f32,
    /// Mean shelter need across population (higher = better).
    pub satisfaction: f32,
    /// Per-role accessibility score (0.0–1.0).
    pub class_accessibility: f32,
    /// Remaining capacity / projected growth.
    pub growth_headroom: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpatialQualityScore {
    /// Graph robustness under single-structure removal (0.0–1.0).
    /// Weight: near-zero for hamlets, moderate for towns, full for cities.
    /// Halved under resource scarcity.
    pub connectivity_resilience: f32,
    /// Mean distinct routes between key building pairs.
    pub redundant_pathing: f32,
    /// Fraction of grid cells serving no function (context-dependent interpretation).
    pub dead_space: f32,
    /// Intentional chokepoints at threat vectors, no unintentional ones on civilian routes.
    pub chokepoint_quality: f32,
    /// Perimeter fraction within garrison response range, weighted by NPC effectiveness.
    pub garrison_coverage: f32,
}

// ---------------------------------------------------------------------------
// Scoring context — scales metrics by settlement level & resources
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringContext {
    pub settlement_level: u8,
    pub resource_scarcity: f32,
    pub active_challenges: Vec<ChallengeCategory>,
    /// Per-category weights for composite score.
    pub category_weights: CategoryWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryWeights {
    pub defensive: f32,
    pub environmental: f32,
    pub economic: f32,
    pub population: f32,
    pub spatial_quality: f32,
}

impl Default for CategoryWeights {
    fn default() -> Self {
        Self {
            defensive: 1.0,
            environmental: 1.0,
            economic: 1.0,
            population: 1.0,
            spatial_quality: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Full scenario result (one row in BC dataset metadata)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub observation: BuildingObservation,
    pub actions: Vec<BuildingAction>,
    pub oracle_score: OutcomeScore,
    pub no_build_score: OutcomeScore,
    /// N random baseline scores for difficulty estimation.
    pub random_scores: Vec<OutcomeScore>,
    /// oracle_composite - no_build_composite.
    pub delta: f32,
    /// oracle_composite - mean(random_composites).
    pub difficulty: f32,
    pub quality_tags: QualityTags,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTags {
    pub challenge_categories: Vec<ChallengeCategory>,
    pub decision_types: Vec<super::types::DecisionType>,
    pub compound_depth: u8,
    /// How constrained the oracle was (0.0 = abundant, 1.0 = starving).
    pub resource_constraint: f32,
    /// How much the solution relied on NPC placement vs. structure (0.0 = all structure, 1.0 = all garrison).
    pub garrison_factor: f32,
    /// Utility gap between chosen action and second-best (low = nuanced = valuable).
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Scoring entry points (implemented by testing workstream)
// ---------------------------------------------------------------------------

/// Apply oracle actions to a cloned world state.
///
/// Maps each `ActionPayload` variant to direct state mutations. Where possible
/// we create real entities so that forward simulation (tick) can pick them up.
pub fn apply_actions(state: &mut WorldState, actions: &[BuildingAction]) {
    // Compute building centroid — same reference point used by world_to_virtual
    // when building the spatial features / observation that the oracle consumed.
    let building_centroid = {
        let positions: Vec<(f32, f32)> = state
            .entities
            .iter()
            .filter(|e| e.alive && e.building.is_some() && e.pos != (0.0, 0.0))
            .map(|e| e.pos)
            .collect();
        if positions.is_empty() {
            state.settlements.first().map(|s| s.pos).unwrap_or((0.0, 0.0))
        } else {
            let cx = positions.iter().map(|p| p.0).sum::<f32>() / positions.len() as f32;
            let cy = positions.iter().map(|p| p.1).sum::<f32>() / positions.len() as f32;
            (cx, cy)
        }
    };

    for action in actions {
        match &action.action {
            ActionPayload::PlaceBuilding {
                building_type,
                grid_cell,
            } => {
                let id = state.next_entity_id();
                // Convert VIRT grid cell back to world space using building centroid.
                let pos = super::features::virtual_to_world(
                    grid_cell.0, grid_cell.1,
                    building_centroid.0, building_centroid.1,
                );
                let mut entity = Entity::new_building(id, pos);
                let mut bdata = crate::world_sim::state::BuildingData::default();
                bdata.building_type = *building_type;
                bdata.grid_col = grid_cell.0;
                bdata.grid_row = grid_cell.1;
                bdata.construction_progress = 1.0; // oracle actions are instant
                bdata.built_tick = state.tick;
                entity.building = Some(bdata);
                state.entities.push(entity);
            }
            ActionPayload::SetBuildPriority {
                building_id,
                priority,
            } => {
                // Store priority on the building's tier field (repurposed as priority hint).
                if let Some(e) = state.entities.iter_mut().find(|e| e.id == *building_id) {
                    if let Some(ref mut bd) = e.building {
                        // Clamp to u8 range — we use tier as a coarse priority bucket.
                        bd.tier = (*priority * 3.0).round().clamp(0.0, 3.0) as u8;
                    }
                }
            }
            ActionPayload::RouteRoad { .. } => {
                // Road routing in VoxelWorld not implemented in scoring simulation.
            }
            ActionPayload::SetZone { .. } => {
                // Zone assignment tracked on entities; VoxelWorld zones not modified here.
            }
            ActionPayload::Demolish { building_id } => {
                if let Some(e) = state.entities.iter_mut().find(|e| e.id == *building_id) {
                    e.alive = false;
                }
            }
            ActionPayload::SetFootprint {
                building_id,
                shape: _,
                dimensions,
            } => {
                if let Some(e) = state.entities.iter_mut().find(|e| e.id == *building_id) {
                    if let Some(ref mut bd) = e.building {
                        bd.footprint_w = dimensions.0;
                        bd.footprint_h = dimensions.1;
                    }
                }
            }
            ActionPayload::SetVertical {
                building_id,
                stories,
                has_basement: _,
                elevation: _,
            } => {
                // Map stories to HP scaling — taller buildings are more robust.
                if let Some(e) = state.entities.iter_mut().find(|e| e.id == *building_id) {
                    e.max_hp = 500.0 * (*stories).max(1) as f32;
                    e.hp = e.max_hp;
                }
            }
            ActionPayload::SetWallSpec {
                segment_id,
                height,
                thickness,
                material,
                features: _,
            } => {
                if let Some(e) = state.entities.iter_mut().find(|e| e.id == *segment_id) {
                    // Scale HP by wall dimensions and material.
                    let material_mult = match material {
                        super::types::BuildMaterial::Thatch => 0.5,
                        super::types::BuildMaterial::Wood => 1.0,
                        super::types::BuildMaterial::Brick => 1.5,
                        super::types::BuildMaterial::Stone => 2.0,
                        super::types::BuildMaterial::Iron => 3.0,
                    };
                    let base = *height as f32 * *thickness as f32 * 50.0;
                    e.max_hp = base * material_mult;
                    e.hp = e.max_hp;
                    e.armor = *thickness as f32 * 5.0 * material_mult;
                }
            }
            ActionPayload::SetRoofSpec {
                building_id,
                roof_type: _,
                material: _,
            } => {
                // Roof affects environmental resilience, but we don't have a direct
                // field — treat as a no-op for now (spatial quality captures this).
                let _ = building_id;
            }
            ActionPayload::SetFoundation {
                building_id,
                foundation_type: _,
                depth,
            } => {
                // Deeper foundations resist environmental damage — boost max HP slightly.
                if let Some(e) = state.entities.iter_mut().find(|e| e.id == *building_id) {
                    let bonus = *depth as f32 * 25.0;
                    e.max_hp += bonus;
                    e.hp += bonus;
                }
            }
            ActionPayload::SetOpenings {
                building_id,
                openings: _,
            } => {
                // Openings affect defensive quality through spatial scoring, not HP.
                let _ = building_id;
            }
            ActionPayload::SetInteriorLayout {
                building_id,
                rooms: _,
            } => {
                // Interior layout is captured by spatial quality metrics.
                let _ = building_id;
            }
            ActionPayload::SetMaterial {
                building_id,
                component: _,
                material,
            } => {
                if let Some(e) = state.entities.iter_mut().find(|e| e.id == *building_id) {
                    let mult = match material {
                        super::types::BuildMaterial::Thatch => 0.5,
                        super::types::BuildMaterial::Wood => 1.0,
                        super::types::BuildMaterial::Brick => 1.5,
                        super::types::BuildMaterial::Stone => 2.0,
                        super::types::BuildMaterial::Iron => 3.0,
                    };
                    e.armor = 10.0 * mult;
                }
            }
            ActionPayload::Renovate {
                building_id,
                upgrades,
            } => {
                if let Some(e) = state.entities.iter_mut().find(|e| e.id == *building_id) {
                    if let Some(ref mut bd) = e.building {
                        bd.tier = (bd.tier + 1).min(3);
                    }
                    // Each upgrade gives a small HP boost.
                    let bonus = upgrades.len() as f32 * 50.0;
                    e.max_hp += bonus;
                    e.hp += bonus;
                }
            }
        }
    }
    // Rebuild indices so tick() picks up new entities.
    state.rebuild_all_indices();
}

// ---------------------------------------------------------------------------
// Challenge injection helpers
// ---------------------------------------------------------------------------

/// Spawn enemy entities into the world state based on challenge enemy profiles.
fn inject_challenge_enemies(state: &mut WorldState, challenge: &Challenge) {
    // Determine spawn position from challenge direction, defaulting to edge.
    let spawn_base: (f32, f32) = if let Some(dir) = challenge.direction {
        // Place enemies 30 units along the threat direction from world origin.
        (dir.0 * 30.0, dir.1 * 30.0)
    } else {
        (50.0, 0.0) // default: north of center
    };

    for profile in &challenge.enemy_profiles {
        let level = ((profile.level_range.0 as u32 + profile.level_range.1 as u32) / 2).max(1);
        for i in 0..profile.count {
            let id = state.next_entity_id();
            let offset = i as f32 * 1.5;
            let pos = (spawn_base.0 + offset, spawn_base.1 + offset);
            let mut monster = Entity::new_monster(id, pos, level);
            // Scale stats by profile properties.
            if profile.has_siege {
                monster.attack_damage += profile.siege_damage;
            }
            if profile.can_fly || profile.can_jump {
                monster.move_speed *= 1.5;
            }
            state.entities.push(monster);
        }
    }
    state.rebuild_all_indices();
}

/// Count buildings in a settlement by scanning entities.
fn count_settlement_buildings(state: &WorldState, settlement_id: Option<u32>) -> (u32, u32) {
    let mut alive = 0u32;
    let mut total = 0u32;
    for e in &state.entities {
        if e.kind == EntityKind::Building {
            if let Some(ref bd) = e.building {
                if bd.settlement_id == settlement_id {
                    total += 1;
                    if e.alive {
                        alive += 1;
                    }
                }
            }
        }
    }
    (alive, total)
}

/// Count friendly/hostile alive entities.
fn count_teams(state: &WorldState) -> (u32, u32) {
    let mut friendly = 0u32;
    let mut hostile = 0u32;
    for e in &state.entities {
        if !e.alive {
            continue;
        }
        match e.team {
            WorldTeam::Friendly => friendly += 1,
            WorldTeam::Hostile => hostile += 1,
            WorldTeam::Neutral => {}
        }
    }
    (friendly, hostile)
}

/// Compute composite score from components using category weights.
fn compute_composite(
    def: &DefensiveScore,
    env: &EnvironmentalScore,
    econ: &EconomicScore,
    pop: &PopulationScore,
    spatial: &SpatialQualityScore,
    weights: &CategoryWeights,
) -> f32 {
    let def_score = (1.0 - def.breach_depth)
        * def.high_value_npc_survival
        * def.casualty_ratio.min(5.0) / 5.0
        * def.resource_efficiency.min(2.0) / 2.0;
    let env_score = (1.0 - env.cascading_severity)
        * env.damage_prevented.clamp(0.0, 1.0)
        * env.drainage_performance.clamp(0.0, 1.0);
    let econ_score = econ.throughput.min(1.0)
        * econ.commute_efficiency.clamp(0.0, 1.0)
        * econ.roi.clamp(0.0, 2.0) / 2.0;
    let pop_score = (1.0 - pop.housing_coverage_delta.clamp(0.0, 1.0))
        * pop.satisfaction.clamp(0.0, 1.0)
        * pop.growth_headroom.clamp(0.0, 1.0);
    let spatial_score = (spatial.connectivity_resilience
        + spatial.redundant_pathing.min(3.0) / 3.0
        + (1.0 - spatial.dead_space)
        + spatial.chokepoint_quality
        + spatial.garrison_coverage)
        / 5.0;

    let total_weight = weights.defensive
        + weights.environmental
        + weights.economic
        + weights.population
        + weights.spatial_quality;
    if total_weight < 1e-6 {
        return 0.0;
    }
    (def_score * weights.defensive
        + env_score * weights.environmental
        + econ_score * weights.economic
        + pop_score * weights.population
        + spatial_score * weights.spatial_quality)
        / total_weight
}

/// Run a challenge forward for N ticks and score the outcome.
///
/// 1. Snapshot initial state metrics.
/// 2. Inject enemies / environmental events from the challenge.
/// 3. Run tick() for the specified number of ticks.
/// 4. Measure outcomes and compute score components.
pub fn run_challenge(
    state: &mut WorldState,
    challenge: &Challenge,
    ticks: u64,
) -> OutcomeScore {
    // --- Snapshot before ---
    let (pre_friendly, _) = count_teams(state);
    let pre_building_count = state
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Building && e.alive)
        .count() as u32;
    let pre_total_hp: f32 = state
        .entities
        .iter()
        .filter(|e| e.alive && e.team == WorldTeam::Friendly)
        .map(|e| e.hp)
        .sum();

    // Find settlement ID for context (use first settlement if any).
    let settlement_id = state.settlements.first().map(|s| s.id);

    // Count high-value NPCs (level > 5 as proxy).
    let hv_npc_ids: Vec<u32> = state
        .entities
        .iter()
        .filter(|e| e.alive && e.team == WorldTeam::Friendly && e.level >= 5)
        .map(|e| e.id)
        .collect();
    let hv_count = hv_npc_ids.len().max(1) as f32;

    // --- Inject challenge ---
    let enemy_count: u32 = challenge.enemy_profiles.iter().map(|p| p.count as u32).sum();
    inject_challenge_enemies(state, challenge);

    // --- Run simulation forward ---
    let mut breach_count = 0u16;
    let mut first_breach_tick: Option<u64> = None;
    let mut defenses_held = true;

    for t in 0..ticks {
        let next = crate::world_sim::tick::tick(state);
        *state = next;

        // Detect breaches: any hostile entity within 10 units of settlement center.
        if let Some(sid) = settlement_id {
            if let Some(s) = state.settlement(sid) {
                let center = s.pos;
                for e in &state.entities {
                    if e.alive && e.team == WorldTeam::Hostile {
                        let dx = e.pos.0 - center.0;
                        let dy = e.pos.1 - center.1;
                        if (dx * dx + dy * dy).sqrt() < 10.0 {
                            if first_breach_tick.is_none() {
                                first_breach_tick = Some(state.tick);
                            }
                            breach_count += 1;
                            defenses_held = false;
                            break; // one breach per tick is enough
                        }
                    }
                }
            }
        }

        // Early termination if all hostiles dead.
        let (_, hostile_alive) = count_teams(state);
        if hostile_alive == 0 && t > 0 {
            break;
        }
    }

    // --- Measure outcomes ---
    let (post_friendly, post_hostile) = count_teams(state);
    let post_building_count = state
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Building && e.alive)
        .count() as u32;
    let post_total_hp: f32 = state
        .entities
        .iter()
        .filter(|e| e.alive && e.team == WorldTeam::Friendly)
        .map(|e| e.hp)
        .sum();

    let friendly_killed = pre_friendly.saturating_sub(post_friendly);
    let enemies_killed = enemy_count.saturating_sub(post_hostile);
    let buildings_lost = pre_building_count.saturating_sub(post_building_count);

    // Breach depth: fraction of settlement radius penetrated.
    let breach_depth = if breach_count > 0 { 0.3 + (breach_count as f32 * 0.05).min(0.7) } else { 0.0 };

    // Casualty ratio: enemy kills / friendly kills (higher = better).
    let casualty_ratio = if friendly_killed == 0 {
        enemies_killed as f32 + 1.0
    } else {
        enemies_killed as f32 / friendly_killed as f32
    };

    // High-value NPC survival.
    let hv_survived = hv_npc_ids
        .iter()
        .filter(|&&id| state.entities.iter().any(|e| e.id == id && e.alive))
        .count() as f32;
    let hv_survival = hv_survived / hv_count;

    let hold_duration = if defenses_held {
        u64::MAX
    } else {
        first_breach_tick.unwrap_or(0)
    };

    // Resource efficiency: kills per building HP invested.
    let total_building_hp: f32 = state
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Building)
        .map(|e| e.max_hp)
        .sum();
    let resource_efficiency = if total_building_hp > 0.0 {
        enemies_killed as f32 / (total_building_hp / 500.0)
    } else {
        0.0
    };

    let defensive = DefensiveScore {
        breach_count,
        breach_depth,
        casualty_ratio,
        high_value_npc_survival: hv_survival,
        response_time_mean: if breach_count > 0 { 5.0 } else { 0.0 },
        hold_duration,
        resource_efficiency,
    };

    // Environmental scoring.
    let damage_prevented = if pre_building_count > 0 {
        1.0 - (buildings_lost as f32 / pre_building_count as f32)
    } else {
        1.0
    };
    let cascading = if buildings_lost > 2 {
        (buildings_lost as f32 / pre_building_count.max(1) as f32).min(1.0)
    } else {
        0.0
    };
    let environmental = EnvironmentalScore {
        damage_prevented,
        recovery_time: buildings_lost as u64 * 50,
        cascading_severity: cascading,
        drainage_performance: if challenge.category == ChallengeCategory::Environmental {
            damage_prevented
        } else {
            1.0
        },
    };

    // Economic scoring (simple proxies).
    let economic = EconomicScore {
        throughput: if post_building_count > 0 { 0.5 } else { 0.0 },
        commute_efficiency: 0.5,
        resource_access_time: 10.0,
        roi: if total_building_hp > 0.0 {
            enemies_killed as f32 * 100.0 / total_building_hp
        } else {
            0.0
        },
    };

    // Population scoring.
    let hp_retention = if pre_total_hp > 0.0 {
        post_total_hp / pre_total_hp
    } else {
        1.0
    };
    let population = PopulationScore {
        housing_coverage_delta: buildings_lost as f32 / pre_building_count.max(1) as f32,
        satisfaction: hp_retention,
        class_accessibility: if post_building_count > 0 { 0.5 } else { 0.0 },
        growth_headroom: if buildings_lost == 0 { 0.5 } else { 0.2 },
    };

    // Spatial quality: use the dedicated scorer if settlement exists.
    let spatial_quality = if let Some(sid) = settlement_id {
        let ctx = ScoringContext {
            settlement_level: state
                .settlement(sid)
                .map(|s| s.infrastructure_level as u8)
                .unwrap_or(1),
            resource_scarcity: 0.3,
            active_challenges: vec![challenge.category],
            category_weights: CategoryWeights::default(),
        };
        score_spatial_quality(state, sid, &ctx)
    } else {
        SpatialQualityScore::default()
    };

    let weights = CategoryWeights::default();
    let composite =
        compute_composite(&defensive, &environmental, &economic, &population, &spatial_quality, &weights);

    OutcomeScore {
        defensive,
        environmental,
        economic,
        population,
        spatial_quality,
        composite,
    }
}

/// Score spatial quality of current settlement layout.
///
/// Evaluates five axes: connectivity resilience, redundant pathing, dead space,
/// chokepoint quality, and garrison coverage. Weights are scaled by settlement
/// level (hamlets get near-zero connectivity weight) and halved under scarcity.
pub fn score_spatial_quality(
    state: &WorldState,
    settlement_id: u32,
    context: &ScoringContext,
) -> SpatialQualityScore {
    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return SpatialQualityScore::default(),
    };

    // --- Settlement level weight (0.0 for hamlets, 1.0 for cities) ---
    let level_weight = (context.settlement_level as f32 / 5.0).clamp(0.0, 1.0);
    let scarcity_mult = if context.resource_scarcity > 0.5 { 0.5 } else { 1.0 };

    let building_range = state.group_index.settlement_buildings(settlement_id);

    // --- Connectivity resilience (entity-based) ---
    // Collect key building IDs (non-wall buildings).
    let key_buildings: Vec<(u32, f32, f32)> = state
        .entities
        .iter()
        .filter(|e| {
            e.alive
                && e.kind == EntityKind::Building
                && e.building
                    .as_ref()
                    .map(|bd| {
                        bd.settlement_id == Some(settlement_id)
                            && !matches!(bd.building_type, BuildingType::Wall | BuildingType::Gate)
                    })
                    .unwrap_or(false)
        })
        .map(|e| (e.id, e.pos.0, e.pos.1))
        .collect();
    let num_key = key_buildings.len().max(1);

    // Connectivity resilience: fraction of buildings within a 30-unit radius of settlement center.
    let connected_count = key_buildings
        .iter()
        .filter(|&&(_, bx, by)| {
            let dx = bx - settlement.pos.0;
            let dy = by - settlement.pos.1;
            (dx * dx + dy * dy).sqrt() < 30.0
        })
        .count();
    let connectivity_resilience = (connected_count as f32 / num_key as f32) * level_weight * scarcity_mult;

    // --- Redundant pathing (entity-based proxy) ---
    // Use building density in core area as a connectivity proxy.
    let core_buildings = key_buildings.iter().filter(|&&(_, bx, by)| {
        let dx = bx - settlement.pos.0;
        let dy = by - settlement.pos.1;
        (dx * dx + dy * dy).sqrt() < 15.0
    }).count();
    let redundant_pathing = if num_key > 1 {
        (core_buildings as f32 / num_key as f32).min(1.0) * 3.0
    } else {
        1.0
    };

    // --- Dead space (entity-based proxy) ---
    // Fraction of a 64x64 virtual area actually occupied by buildings.
    let building_count = building_range.clone().filter(|&idx| {
        idx < state.entities.len() && state.entities[idx].alive
    }).count();
    let total_cells = 64.0 * 64.0_f32;
    let dead_space = 1.0 - (building_count as f32 / total_cells).min(1.0);

    // --- Chokepoint quality (entity-based) ---
    let wall_count = state.entities.iter().filter(|e| {
        e.alive && e.kind == EntityKind::Building
            && e.building.as_ref().map(|bd| {
                bd.settlement_id == Some(settlement_id)
                    && matches!(bd.building_type, BuildingType::Wall | BuildingType::Gate)
            }).unwrap_or(false)
    }).count();
    let perimeter_estimate = 4 * 20_usize; // assume ~20-unit radius settlement
    let intentional_chokepoints = (wall_count as f32 / perimeter_estimate.max(1) as f32).min(1.0);
    let chokepoint_quality = intentional_chokepoints;

    // --- Garrison coverage (entity-based) ---
    let garrison_npcs: Vec<&Entity> = state
        .entities
        .iter()
        .filter(|e| {
            e.alive
                && e.team == WorldTeam::Friendly
                && e.kind == EntityKind::Npc
                && e.attack_damage > 5.0
        })
        .filter(|e| {
            let dx = e.pos.0 - settlement.pos.0;
            let dy = e.pos.1 - settlement.pos.1;
            (dx * dx + dy * dy).sqrt() < 40.0
        })
        .collect();
    let covered_perimeter = garrison_npcs
        .iter()
        .map(|e| (e.attack_range * 2.0) as usize)
        .sum::<usize>();
    let garrison_coverage = (covered_perimeter as f32 / perimeter_estimate.max(1) as f32).min(1.0);

    SpatialQualityScore {
        connectivity_resilience,
        redundant_pathing,
        dead_space,
        chokepoint_quality,
        garrison_coverage,
    }
}

/// Full evaluation: oracle vs. no-build vs. random baselines.
///
/// Runs counterfactual experiments to determine how much value the oracle
/// actions add over doing nothing (delta) and over random valid actions
/// (difficulty).
pub fn evaluate_scenario(
    state: &WorldState,
    obs: &BuildingObservation,
    actions: &[BuildingAction],
    num_random_baselines: usize,
) -> ScenarioResult {
    let challenge = obs.challenges.first();
    let default_challenge = Challenge {
        category: ChallengeCategory::Military,
        sub_type: 0,
        sub_type_name: String::from("default"),
        severity: 0.5,
        direction: None,
        deadline_tick: None,
        enemy_profiles: Vec::new(),
    };
    let ch = challenge.unwrap_or(&default_challenge);
    let sim_ticks = ch.deadline_tick.unwrap_or(100);

    // --- Oracle baseline: apply actions, then run challenge ---
    let mut oracle_state = state.clone();
    apply_actions(&mut oracle_state, actions);
    let oracle_score = run_challenge(&mut oracle_state, ch, sim_ticks);

    // --- No-build baseline: run challenge without any actions ---
    let mut nobuild_state = state.clone();
    let no_build_score = run_challenge(&mut nobuild_state, ch, sim_ticks);

    // --- Random baselines ---
    let mut random_scores = Vec::with_capacity(num_random_baselines);
    for i in 0..num_random_baselines {
        let mut rng_state = state.clone();
        // Generate random valid actions: shuffle placement positions.
        let random_actions = generate_random_actions(actions, i as u64);
        apply_actions(&mut rng_state, &random_actions);
        let score = run_challenge(&mut rng_state, ch, sim_ticks);
        random_scores.push(score);
    }

    // --- Compute delta and difficulty ---
    let delta = oracle_score.composite - no_build_score.composite;
    let mean_random = if random_scores.is_empty() {
        no_build_score.composite
    } else {
        random_scores.iter().map(|s| s.composite).sum::<f32>() / random_scores.len() as f32
    };
    let difficulty = oracle_score.composite - mean_random;

    // --- Quality tags ---
    let challenge_categories: Vec<ChallengeCategory> =
        obs.challenges.iter().map(|c| c.category).collect();
    let decision_types: Vec<DecisionType> = actions.iter().map(|a| a.decision_type).collect();
    let compound_depth = challenge_categories.len().min(255) as u8;

    // Garrison factor: fraction of actions that are garrison/military related.
    let garrison_actions = actions
        .iter()
        .filter(|a| {
            matches!(
                a.decision_type,
                DecisionType::DefensiveIntegration | DecisionType::Placement
            ) && matches!(
                a.action,
                ActionPayload::PlaceBuilding {
                    building_type: BuildingType::Barracks | BuildingType::Watchtower,
                    ..
                }
            )
        })
        .count();
    let garrison_factor = if actions.is_empty() {
        0.0
    } else {
        garrison_actions as f32 / actions.len() as f32
    };

    // Resource constraint from context.
    let resource_constraint = obs
        .spatial
        .economic
        .stockpiles
        .iter()
        .sum::<f32>()
        .min(1000.0)
        / 1000.0;
    let resource_constraint = 1.0 - resource_constraint; // invert: high stock = low constraint

    // Confidence: gap between oracle and best random.
    let best_random = random_scores
        .iter()
        .map(|s| s.composite)
        .fold(f32::NEG_INFINITY, f32::max);
    let confidence = if best_random.is_finite() {
        (oracle_score.composite - best_random).max(0.0).min(1.0)
    } else {
        delta.clamp(0.0, 1.0)
    };

    let quality_tags = QualityTags {
        challenge_categories,
        decision_types,
        compound_depth,
        resource_constraint,
        garrison_factor,
        confidence,
    };

    ScenarioResult {
        observation: obs.clone(),
        actions: actions.to_vec(),
        oracle_score,
        no_build_score,
        random_scores,
        delta,
        difficulty,
        quality_tags,
    }
}

// ---------------------------------------------------------------------------
// Random action generation for baselines
// ---------------------------------------------------------------------------

/// Generate random valid actions by perturbing the oracle actions.
/// Shifts placement positions and randomizes some structural choices.
fn generate_random_actions(oracle_actions: &[BuildingAction], seed: u64) -> Vec<BuildingAction> {
    oracle_actions
        .iter()
        .enumerate()
        .map(|(i, action)| {
            let mut a = action.clone();
            // Deterministic pseudo-random offset from seed + index.
            let h = ((seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) >> 16) as u16;
            match &mut a.action {
                ActionPayload::PlaceBuilding { grid_cell, .. } => {
                    // Shift position by a random offset in [-5, +5].
                    let dx = (h % 11) as i32 - 5;
                    let dy = ((h / 11) % 11) as i32 - 5;
                    grid_cell.0 = (grid_cell.0 as i32 + dx).max(0) as u16;
                    grid_cell.1 = (grid_cell.1 as i32 + dy).max(0) as u16;
                }
                ActionPayload::RouteRoad { waypoints } => {
                    for wp in waypoints.iter_mut() {
                        let dx = (h % 7) as i32 - 3;
                        let dy = ((h / 7) % 7) as i32 - 3;
                        wp.0 = (wp.0 as i32 + dx).max(0) as u16;
                        wp.1 = (wp.1 as i32 + dy).max(0) as u16;
                    }
                }
                ActionPayload::SetWallSpec {
                    height, thickness, ..
                } => {
                    *height = (*height as i32 + (h % 3) as i32 - 1).clamp(1, 10) as u8;
                    *thickness = (*thickness as i32 + (h % 3) as i32 - 1).clamp(1, 5) as u8;
                }
                _ => {} // Other actions pass through unmodified.
            }
            a
        })
        .collect()
}
