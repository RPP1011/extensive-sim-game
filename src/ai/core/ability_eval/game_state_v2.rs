use crate::ai::core::{distance, is_alive, SimState, Team, UnitState};
use crate::ai::goap::spatial::VisibilityMap;
use crate::ai::pathing::GridNav;
use super::game_state::unit_dps;
use super::game_state::{
    MAX_ENEMIES, MAX_ALLIES,
    rich_entity_features, rich_entity_features_spatial, summarize_abilities,
};
use super::game_state_threats::extract_threats_v2;
use super::game_state_positions::extract_position_tokens;
use super::game_state_zones::{extract_zone_tokens, ZoneObjective};
use super::extraction_cache::ExtractionCache;

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// V2: Variable-length game state with threats
// ---------------------------------------------------------------------------

/// Structured game state with variable-length entity and threat tokens.
/// No caps on entity count — self-attention handles arbitrary sequences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStateV2 {
    /// Per-entity feature vectors (ENTITY_FEATURE_DIM each).
    /// Order: [self, enemies sorted by distance, allies sorted by HP%]
    pub entities: Vec<Vec<f32>>,
    /// Type ID per entity: 0=self, 1=enemy, 2=ally, 3=threat
    pub entity_types: Vec<u8>,
    /// Per-threat feature vectors (THREAT_FEATURE_DIM each).
    pub threats: Vec<Vec<f32>>,
    /// Per-position feature vectors (POSITION_FEATURE_DIM each).
    /// Areas of interest: cover spots, elevated positions, safe retreats, attack positions.
    #[serde(default)]
    pub positions: Vec<Vec<f32>>,
    /// Unified zone tokens (ZONE_DIM=12 each, max 10).
    /// Replaces threats + positions for V5+ models.
    #[serde(default)]
    pub zones: Vec<Vec<f32>>,
    /// Aggregate features summarizing entities that didn't get individual slots (16 floats).
    #[serde(default)]
    pub aggregate_features: Vec<f32>,
}

/// Score all enemy and ally units by priority and return the top-K unit IDs
/// that earn individual entity slots. The rest go into the aggregate.
///
/// Priority scoring:
/// - Base: auto_dps/30 + ability_damage/50
/// - CC ready: +0.3
/// - Proximity: (1.0 - dist/20).max(0) * 0.4
/// - Low HP enemy: +0.5 if hp_pct < 0.25
/// - Low HP ally: +0.4 if hp_pct < 0.3
/// - Casting: +0.6 if is_casting
pub fn select_entity_slots(state: &SimState, unit: &UnitState, max_slots: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = Vec::new();

    for u in &state.units {
        if !is_alive(u) || u.id == unit.id {
            continue;
        }

        let dist = distance(unit.position, u.position);
        let hp_pct = u.hp as f32 / u.max_hp.max(1) as f32;
        let abil = summarize_abilities(u);

        let mut score = unit_dps(u) / 30.0 + abil.ability_damage / 50.0;

        // CC ready
        if abil.control_duration_ms > 0.0 && abil.control_cd_pct < 0.01 {
            score += 0.3;
        }

        // Proximity
        score += (1.0 - dist / 20.0).max(0.0) * 0.4;

        let is_enemy = u.team != unit.team;

        // Low HP
        if is_enemy && hp_pct < 0.25 {
            score += 0.5;
        }
        if !is_enemy && hp_pct < 0.3 {
            score += 0.4;
        }

        // Casting
        if u.casting.is_some() {
            score += 0.6;
        }

        scored.push((u.id, score));
    }

    // Sort descending by score
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top max_slots (self always gets a slot separately, so max_slots is for others)
    scored.iter().take(max_slots).map(|&(id, _)| id).collect()
}

/// Compute 16 aggregate features summarizing ALL entities that did NOT get
/// individual entity slots.
pub fn compute_aggregate_features(
    state: &SimState,
    unit: &UnitState,
    selected_ids: &[u32],
) -> Vec<f32> {
    const AGG_DIM: usize = 16;

    let mut n_enemies_total = 0usize;
    let mut n_allies_total = 0usize;
    let mut n_enemies_trunc = 0usize;
    let mut n_allies_trunc = 0usize;

    let mut enemy_sum_x = 0.0f32;
    let mut enemy_sum_y = 0.0f32;
    let mut ally_sum_x = 0.0f32;
    let mut ally_sum_y = 0.0f32;

    let mut enemy_hp_sum = 0.0f32;
    let mut min_enemy_hp_pct = 1.0f32;
    let mut max_enemy_threat = 0.0f32;

    let mut agg_enemy_dps = 0.0f32;
    let mut agg_cc_threat = 0.0f32;

    // For spread calculation
    let mut enemy_positions: Vec<(f32, f32)> = Vec::new();

    // Dominant type tracking: count melee/ranged/caster among all enemies
    let mut n_melee = 0usize;
    let mut n_ranged = 0usize;
    let mut n_caster = 0usize;

    for u in &state.units {
        if !is_alive(u) || u.id == unit.id {
            continue;
        }

        let is_enemy = u.team != unit.team;
        let hp_pct = u.hp as f32 / u.max_hp.max(1) as f32;
        let is_selected = selected_ids.contains(&u.id);

        if is_enemy {
            n_enemies_total += 1;
            enemy_sum_x += u.position.x;
            enemy_sum_y += u.position.y;
            enemy_hp_sum += hp_pct;
            if hp_pct < min_enemy_hp_pct {
                min_enemy_hp_pct = hp_pct;
            }
            enemy_positions.push((u.position.x, u.position.y));

            // Threat score for max
            let dps = unit_dps(u);
            let abil = summarize_abilities(u);
            let threat = dps / 30.0 + abil.ability_damage / 50.0;
            if threat > max_enemy_threat {
                max_enemy_threat = threat;
            }

            // Classify type based on attack range
            if u.attack_range <= 2.0 {
                n_melee += 1;
            } else if !u.abilities.is_empty() && abil.ability_damage > 0.0 {
                n_caster += 1;
            } else {
                n_ranged += 1;
            }

            if !is_selected {
                n_enemies_trunc += 1;
                agg_enemy_dps += dps;
                let cc_dur = abil.control_duration_ms;
                agg_cc_threat += cc_dur;
            }
        } else {
            n_allies_total += 1;
            ally_sum_x += u.position.x;
            ally_sum_y += u.position.y;

            if !is_selected {
                n_allies_trunc += 1;
            }
        }
    }

    let n_enemies_f = n_enemies_total.max(1) as f32;
    let n_allies_f = n_allies_total.max(1) as f32;

    // Enemy centroid
    let enemy_cx = if n_enemies_total > 0 { enemy_sum_x / n_enemies_f } else { unit.position.x };
    let enemy_cy = if n_enemies_total > 0 { enemy_sum_y / n_enemies_f } else { unit.position.y };

    // Ally centroid
    let ally_cx = if n_allies_total > 0 { ally_sum_x / n_allies_f } else { unit.position.x };
    let ally_cy = if n_allies_total > 0 { ally_sum_y / n_allies_f } else { unit.position.y };

    // Mean enemy HP
    let mean_enemy_hp = if n_enemies_total > 0 { enemy_hp_sum / n_enemies_f } else { 0.0 };

    // Enemy spread (std dev of positions)
    let enemy_spread = if enemy_positions.len() > 1 {
        let var_x: f32 = enemy_positions.iter()
            .map(|(x, _)| (x - enemy_cx) * (x - enemy_cx))
            .sum::<f32>() / n_enemies_f;
        let var_y: f32 = enemy_positions.iter()
            .map(|(_, y)| (y - enemy_cy) * (y - enemy_cy))
            .sum::<f32>() / n_enemies_f;
        (var_x + var_y).sqrt()
    } else {
        0.0
    };

    // Dominant type
    let total_typed = n_melee + n_ranged + n_caster;
    let dominant_type = if total_typed == 0 {
        0.0
    } else {
        let max_count = n_melee.max(n_ranged).max(n_caster);
        if max_count == n_melee && n_melee > n_ranged && n_melee > n_caster {
            0.0 // melee
        } else if max_count == n_ranged && n_ranged > n_melee && n_ranged > n_caster {
            0.33 // ranged
        } else if max_count == n_caster && n_caster > n_melee && n_caster > n_ranged {
            0.67 // caster
        } else {
            1.0 // mixed
        }
    };

    // Projectile count
    let n_projectiles = state.projectiles.len();

    let mut features = Vec::with_capacity(AGG_DIM);
    features.push(n_enemies_total as f32 / 20.0);          // 0
    features.push(n_allies_total as f32 / 10.0);            // 1
    features.push(n_enemies_trunc as f32 / 15.0);           // 2
    features.push(n_allies_trunc as f32 / 8.0);             // 3
    features.push(enemy_cx / 20.0);                          // 4
    features.push(enemy_cy / 20.0);                          // 5
    features.push(ally_cx / 20.0);                           // 6
    features.push(ally_cy / 20.0);                           // 7
    features.push(mean_enemy_hp);                            // 8
    features.push(min_enemy_hp_pct);                         // 9
    features.push(max_enemy_threat);                         // 10
    features.push(agg_enemy_dps / 200.0);                    // 11
    features.push(n_projectiles as f32 / 10.0);              // 12
    features.push(enemy_spread / 10.0);                      // 13
    features.push(dominant_type);                             // 14
    features.push(agg_cc_threat / 5000.0);                   // 15

    features
}

/// Extract variable-length game state for entity encoder v2.
///
/// All living enemies and allies are included (no cap).
/// Threats from projectiles, zones, and enemy casts are appended.
pub fn extract_game_state_v2(state: &SimState, unit: &UnitState) -> GameStateV2 {
    extract_game_state_v2_inner(state, unit, None, None, &[])
}

/// Extract game state v2 with spatial summary features (34-dim entities).
///
/// When vis_map and nav are provided, each entity's feature vector includes
/// 4 spatial summary features (visible corners, passage widths, corner distance).
pub fn extract_game_state_v2_spatial(
    state: &SimState,
    unit: &UnitState,
    vis_map: &VisibilityMap,
    nav: &GridNav,
) -> GameStateV2 {
    extract_game_state_v2_inner(state, unit, Some(vis_map), Some(nav), &[])
}

/// Extract game state v2 with objectives for unified zone tokens.
pub fn extract_game_state_v2_with_objectives(
    state: &SimState,
    unit: &UnitState,
    vis_map: Option<&VisibilityMap>,
    nav: Option<&GridNav>,
    objectives: &[ZoneObjective],
) -> GameStateV2 {
    extract_game_state_v2_inner(state, unit, vis_map, nav, objectives)
}

/// Extract game state using a per-tick cache. Avoids recomputing
/// `summarize_abilities()` and entity scoring for each hero.
///
/// When `vis_map` and `nav` are provided, entity features are 34-dim
/// (30 base + 4 spatial summary). Otherwise, spatial features are zero-filled.
pub fn extract_game_state_v2_cached(
    state: &SimState,
    unit: &UnitState,
    cache: &ExtractionCache,
    objectives: &[ZoneObjective],
) -> GameStateV2 {
    extract_game_state_v2_cached_spatial(state, unit, cache, objectives, None, None)
}

/// Extract game state with spatial summary features from VisibilityMap.
pub fn extract_game_state_v2_cached_spatial(
    state: &SimState,
    unit: &UnitState,
    cache: &ExtractionCache,
    objectives: &[ZoneObjective],
    vis_map: Option<&VisibilityMap>,
    nav: Option<&GridNav>,
) -> GameStateV2 {
    let max_other_slots = MAX_ENEMIES + MAX_ALLIES;
    let selected_ids = cache.select_entity_slots(unit, max_other_slots);

    let mut entities: Vec<Vec<f32>> = Vec::new();
    let mut entity_types: Vec<u8> = Vec::new();

    let entity_feats = |state: &SimState, caster: &UnitState, u: &UnitState, is_self: bool| -> Vec<f32> {
        match (vis_map, nav) {
            (Some(vm), Some(n)) => rich_entity_features_spatial(state, caster, u, is_self, vm, n).to_vec(),
            _ => rich_entity_features(state, caster, u, is_self).to_vec(),
        }
    };

    // Self entity
    entities.push(entity_feats(state, unit, unit, true));
    entity_types.push(0);

    // Selected enemies
    for &sid in &selected_ids {
        if let Some(u) = state.units.iter().find(|u| u.id == sid && is_alive(u)) {
            if u.team != unit.team {
                entities.push(entity_feats(state, unit, u, false));
                entity_types.push(1);
            }
        }
    }

    // Selected allies
    for &sid in &selected_ids {
        if let Some(u) = state.units.iter().find(|u| u.id == sid && is_alive(u)) {
            if u.team == unit.team {
                entities.push(entity_feats(state, unit, u, false));
                entity_types.push(2);
            }
        }
    }

    let threats = extract_threats_v2(state, unit);
    let positions = Vec::new(); // Skip position tokens in cached path (zones replace them)
    let zones = extract_zone_tokens(state, unit, objectives);
    let aggregate_features = compute_aggregate_features(state, unit, &selected_ids);

    GameStateV2 { entities, entity_types, threats, positions, zones, aggregate_features }
}

fn extract_game_state_v2_inner(
    state: &SimState,
    unit: &UnitState,
    vis_map: Option<&VisibilityMap>,
    nav: Option<&GridNav>,
    objectives: &[ZoneObjective],
) -> GameStateV2 {
    // Use importance-based slot selection (exclude self slot from budget)
    let max_other_slots = MAX_ENEMIES + MAX_ALLIES; // 6 slots for non-self entities
    let selected_ids = select_entity_slots(state, unit, max_other_slots);

    let entity_feats = |state: &SimState, caster: &UnitState, u: &UnitState, is_self: bool| -> Vec<f32> {
        match (vis_map, nav) {
            (Some(vm), Some(n)) => rich_entity_features_spatial(state, caster, u, is_self, vm, n).to_vec(),
            _ => rich_entity_features(state, caster, u, is_self).to_vec(),
        }
    };

    let mut entities: Vec<Vec<f32>> = Vec::new();
    let mut entity_types: Vec<u8> = Vec::new();

    // Self entity (always gets a slot)
    entities.push(entity_feats(state, unit, unit, true));
    entity_types.push(0); // self

    // Selected enemies (in priority order, matching select_entity_slots order)
    for &sid in &selected_ids {
        if let Some(u) = state.units.iter().find(|u| u.id == sid && is_alive(u)) {
            if u.team != unit.team {
                entities.push(entity_feats(state, unit, u, false));
                entity_types.push(1); // enemy
            }
        }
    }

    // Selected allies (in priority order)
    for &sid in &selected_ids {
        if let Some(u) = state.units.iter().find(|u| u.id == sid && is_alive(u)) {
            if u.team == unit.team {
                entities.push(entity_feats(state, unit, u, false));
                entity_types.push(2); // ally
            }
        }
    }

    // Threats
    let threats = extract_threats_v2(state, unit);

    // Position tokens (areas of interest)
    let positions = extract_position_tokens(state, unit);

    // Unified zone tokens (threats + opportunities + objectives + cover)
    let zones = extract_zone_tokens(state, unit, objectives);

    // Aggregate features for truncated entities
    let aggregate_features = compute_aggregate_features(state, unit, &selected_ids);

    GameStateV2 { entities, entity_types, threats, positions, zones, aggregate_features }
}

// ---------------------------------------------------------------------------
// V2 outcome dataset: variable-length entities + threats
// ---------------------------------------------------------------------------

/// V2 outcome sample with variable-length entity and threat tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeSampleV2 {
    /// Per-entity feature vectors (30-dim each), variable length.
    pub entities: Vec<Vec<f32>>,
    /// Type ID per entity: 0=self, 1=enemy, 2=ally.
    pub entity_types: Vec<u8>,
    /// Per-threat feature vectors (8-dim each), variable length.
    pub threats: Vec<Vec<f32>>,
    /// Per-position feature vectors (8-dim each), variable length.
    #[serde(default)]
    pub positions: Vec<Vec<f32>>,
    /// Unified zone tokens (12-dim each), variable length.
    #[serde(default)]
    pub zones: Vec<Vec<f32>>,
    /// 1.0 = hero team wins, 0.0 = enemy team wins.
    pub hero_wins: f32,
    /// Hero team HP fraction at fight end (0-1).
    pub hero_hp_remaining: f32,
    /// How far through the fight (0=start, 1=end).
    pub fight_progress: f32,
    /// Scenario name for debugging.
    pub scenario: String,
    pub tick: u64,
}

/// Generate v2 outcome prediction training data with variable-length entities + threats.
pub fn generate_outcome_dataset_v2(
    initial_sim: SimState,
    initial_squad_ai: crate::ai::squad::SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    sample_interval: u64,
) -> Vec<OutcomeSampleV2> {
    use crate::ai::core::{step, FIXED_TICK_MS};
    use crate::ai::squad::generate_intents;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut snapshots: Vec<(u64, Vec<GameStateV2>)> = Vec::new();

    for tick in 0..max_ticks {
        let intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        if tick % sample_interval == 0 {
            let hero_states: Vec<GameStateV2> = sim.units.iter()
                .filter(|u| u.team == Team::Hero && is_alive(u))
                .map(|u| extract_game_state_v2(&sim, u))
                .collect();
            if !hero_states.is_empty() {
                snapshots.push((tick, hero_states));
            }
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 || heroes_alive == 0 {
            break;
        }
    }

    let final_hero_hp: f32 = sim.units.iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| (u.hp.max(0) as f32) / u.max_hp.max(1) as f32)
        .sum::<f32>()
        / sim.units.iter().filter(|u| u.team == Team::Hero).count().max(1) as f32;
    let final_enemy_hp: f32 = sim.units.iter()
        .filter(|u| u.team == Team::Enemy)
        .map(|u| (u.hp.max(0) as f32) / u.max_hp.max(1) as f32)
        .sum::<f32>()
        / sim.units.iter().filter(|u| u.team == Team::Enemy).count().max(1) as f32;
    let hero_wins = if final_hero_hp > final_enemy_hp { 1.0 } else { 0.0 };
    let total_ticks = sim.tick.max(1) as f32;

    let mut samples = Vec::new();
    for (tick, hero_states) in snapshots {
        for gs in hero_states {
            samples.push(OutcomeSampleV2 {
                entities: gs.entities,
                entity_types: gs.entity_types,
                threats: gs.threats,
                positions: gs.positions,
                zones: gs.zones,
                hero_wins,
                hero_hp_remaining: final_hero_hp,
                fight_progress: tick as f32 / total_ticks,
                scenario: scenario_name.to_string(),
                tick,
            });
        }
    }

    samples
}

/// Write v2 outcome samples as JSONL.
pub fn write_outcome_dataset_v2(
    samples: &[OutcomeSampleV2],
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for sample in samples {
        serde_json::to_writer(&mut writer, sample).unwrap();
        writeln!(writer)?;
    }
    Ok(())
}

