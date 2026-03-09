use crate::ai::core::{distance, is_alive, SimState, Team, UnitState};
use crate::ai::core::types::CastKind;
use crate::ai::effects::effect_enum::Effect;
use crate::ai::effects::types::Area;
use super::features::unit_dps;

// ---------------------------------------------------------------------------
// Rich game state representation for entity encoder pre-training
// ---------------------------------------------------------------------------
//
// 7 entity slots × 30 features = 210 floats total.
// Entity order: [self, enemy0, enemy1, enemy2, ally0, ally1, ally2]
// Entity types: 0=self, 1=enemy, 2=ally
//
// Per-entity features (30):
//   -- Vitals --
//   0: hp_pct
//   1: shield_pct (shield_hp / max_hp)
//   2: resource_pct
//   3: armor / 200
//   4: magic_resist / 200
//   -- Position / terrain --
//   5: position_x / 20 (absolute, centered)
//   6: position_y / 20
//   7: distance_from_caster / 10 (0 for self)
//   8: cover_bonus
//   9: elevation / 5
//  10: n_hostile_zones_nearby / 3
//  11: n_friendly_zones_nearby / 3
//  -- Combat stats --
//  12: auto_dps / 30
//  13: attack_range / 10
//  14: attack_cd_remaining_pct (0 = ready, 1 = full CD)
//  -- Ability readiness (strongest ability) --
//  15: ability_damage / 50
//  16: ability_range / 10
//  17: ability_cd_remaining_pct
//  -- Healing --
//  18: heal_amount / 50
//  19: heal_range / 10
//  20: heal_cd_remaining_pct
//  -- CC capability --
//  21: control_range / 10
//  22: control_duration / 2000
//  23: control_cd_remaining_pct
//  -- Current state --
//  24: is_casting (0/1)
//  25: cast_progress (0-1, how far through current cast)
//  26: cc_remaining / 2000 (how long until CC wears off)
//  27: move_speed / 5
//  -- Cumulative --
//  28: total_damage_done / 1000
//  29: exists (1.0 if slot occupied, 0.0 if padding)
//
// ---------------------------------------------------------------------------
// Context features: 4 floats — situational awareness
// ---------------------------------------------------------------------------
//
//   0: n_enemies_nearby / 8  — enemies within range 5.0
//   1: n_allies_nearby / 8   — allies within range 5.0
//   2: n_enemies_total / 8   — all living enemies
//   3: n_allies_total / 8    — all living allies (excluding self)
//
// ---------------------------------------------------------------------------
// Threat slots: 8 slots × 8 features = 64 floats
// ---------------------------------------------------------------------------
//
// Threats are incoming dangers relative to the unit being encoded.
// Sources: in-flight projectiles, hostile zones, enemy casts targeting this
// unit or its position, ground-target ability casts (warning zones).
//
// Sorted by urgency (time_to_impact ascending).
// Up to 8 threats tracked (supports large team fights).
//
// Per-threat features (8):
//   0: dx / 10       — relative x offset from unit to threat impact point
//   1: dy / 10       — relative y offset from unit to threat impact point
//   2: distance / 10 — distance from unit to threat impact point
//   3: radius / 5    — AoE radius (0 for single-target projectiles)
//   4: time_to_impact / 2000 — ms until damage lands (cast remaining or travel time)
//   5: damage_ratio  — incoming_damage / unit.hp (>1.0 = lethal)
//   6: has_cc        — 0/1, whether threat includes stun/root/silence
//   7: exists        — 1.0 if slot occupied, 0.0 if padding

pub const ENTITY_FEATURE_DIM: usize = 30;
pub const MAX_ENEMIES: usize = 3;
pub const MAX_ALLIES: usize = 3;
pub const NUM_ENTITY_SLOTS: usize = 1 + MAX_ENEMIES + MAX_ALLIES; // 7

pub const THREAT_FEATURE_DIM: usize = 8;
pub const POSITION_FEATURE_DIM: usize = 8;
pub const MAX_POSITION_TOKENS: usize = 8;

/// Legacy dimension without threat slots (for backwards compatibility).
pub const GAME_STATE_DIM: usize = NUM_ENTITY_SLOTS * ENTITY_FEATURE_DIM; // 210

/// Extract rich game state features for entity encoder.
///
/// Returns a flat 210-dim vector: [self(30), enemy0(30), enemy1(30), enemy2(30),
/// ally0(30), ally1(30), ally2(30)].
///
/// Enemies sorted by distance (nearest first), allies sorted by HP% (lowest first).
pub fn extract_game_state(state: &SimState, unit: &UnitState) -> Vec<f32> {
    let mut features = Vec::with_capacity(GAME_STATE_DIM);

    // Self entity
    features.extend_from_slice(&rich_entity_features(state, unit, unit, true));

    // Enemies: sorted by distance, up to 3
    let mut enemies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    enemies.sort_by(|a, b| {
        distance(unit.position, a.position)
            .partial_cmp(&distance(unit.position, b.position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in 0..MAX_ENEMIES {
        if let Some(enemy) = enemies.get(i) {
            features.extend_from_slice(&rich_entity_features(state, unit, enemy, false));
        } else {
            features.extend_from_slice(&EMPTY_ENTITY);
        }
    }

    // Allies (excluding self): sorted by HP% ascending (most wounded first)
    let mut allies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u) && u.id != unit.id)
        .collect();
    allies.sort_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in 0..MAX_ALLIES {
        if let Some(ally) = allies.get(i) {
            features.extend_from_slice(&rich_entity_features(state, unit, ally, false));
        } else {
            features.extend_from_slice(&EMPTY_ENTITY);
        }
    }

    debug_assert_eq!(features.len(), GAME_STATE_DIM);
    features
}

const EMPTY_ENTITY: [f32; ENTITY_FEATURE_DIM] = [0.0; ENTITY_FEATURE_DIM];
const EMPTY_THREAT: [f32; THREAT_FEATURE_DIM] = [0.0; THREAT_FEATURE_DIM];

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
}

/// Extract variable-length game state for entity encoder v2.
///
/// All living enemies and allies are included (no cap).
/// Threats from projectiles, zones, and enemy casts are appended.
pub fn extract_game_state_v2(state: &SimState, unit: &UnitState) -> GameStateV2 {
    let mut entities: Vec<Vec<f32>> = Vec::new();
    let mut entity_types: Vec<u8> = Vec::new();

    // Self entity
    entities.push(rich_entity_features(state, unit, unit, true).to_vec());
    entity_types.push(0); // self

    // All enemies sorted by distance
    let mut enemies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    enemies.sort_by(|a, b| {
        distance(unit.position, a.position)
            .partial_cmp(&distance(unit.position, b.position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for enemy in &enemies {
        entities.push(rich_entity_features(state, unit, enemy, false).to_vec());
        entity_types.push(1); // enemy
    }

    // All allies (excluding self) sorted by HP% ascending
    let mut allies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u) && u.id != unit.id)
        .collect();
    allies.sort_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    for ally in &allies {
        entities.push(rich_entity_features(state, unit, ally, false).to_vec());
        entity_types.push(2); // ally
    }

    // Threats
    let threats = extract_threats_v2(state, unit);

    // Position tokens (areas of interest)
    let positions = extract_position_tokens(state, unit);

    GameStateV2 { entities, entity_types, threats, positions }
}

// ---------------------------------------------------------------------------
// Position token extraction: areas of interest for pointer-based actions
// ---------------------------------------------------------------------------

/// A candidate position with intrinsic spatial properties.
struct PositionCandidate {
    pos: crate::ai::core::SimVec2,
    /// Tactical value for sorting (higher = more interesting).
    score: f32,
}

/// Extract up to MAX_POSITION_TOKENS areas of interest as 8-dim feature vectors.
///
/// Position features (8):
///   0: dx from self / 20
///   1: dy from self / 20
///   2: path distance from self / 30 (A* distance; divergence from euclidean reveals walls)
///   3: elevation / 5
///   4: chokepoint_score / 3 (blocked cardinal neighbors)
///   5: wall_proximity / 5 (min raycast distance to nearest wall)
///   6: n_hostile_zones / 3
///   7: n_friendly_zones / 3
fn extract_position_tokens(state: &SimState, unit: &UnitState) -> Vec<Vec<f32>> {
    use crate::ai::core::sim_vec2;

    let nav = match &state.grid_nav {
        Some(nav) => nav,
        None => return Vec::new(),
    };

    let self_pos = unit.position;
    let move_range = unit.move_speed_per_sec * 1.5; // ~1.5 seconds of movement

    let mut candidates: Vec<PositionCandidate> = Vec::new();

    // Sample positions in 8 directions at 2 distances
    let directions: [(f32, f32); 8] = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707),
    ];

    for &(dx, dy) in &directions {
        for dist_mult in &[0.5, 1.0, 1.5] {
            let d = move_range * dist_mult;
            let pos = sim_vec2(self_pos.x + dx * d, self_pos.y + dy * d);

            // Skip out-of-bounds or blocked positions
            if !nav.is_walkable_pos(pos) {
                continue;
            }

            // Compute intrinsic features for scoring
            let elevation = nav.elevation_at_pos(pos);
            let cell = nav.cell_of(pos);
            let blocked_neighbors = [(1i32,0i32),(-1,0),(0,1),(0,-1)].iter()
                .filter(|(ox, oy)| nav.blocked.contains(&(cell.0 + ox, cell.1 + oy)))
                .count();

            // Score: elevation + chokepoint value (1-2 blocked = good, 3+ = cornered)
            let choke_val = match blocked_neighbors {
                1 => 1.0,
                2 => 1.5,
                _ => 0.0,
            };
            let score = elevation * 2.0 + choke_val;

            candidates.push(PositionCandidate { pos, score });
        }
    }

    // Deduplicate: remove candidates within 2.0 of a higher-scored candidate
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut filtered: Vec<PositionCandidate> = Vec::new();
    for c in candidates {
        let too_close = filtered.iter().any(|f| distance(c.pos, f.pos) < 2.0);
        if !too_close {
            filtered.push(c);
        }
        if filtered.len() >= MAX_POSITION_TOKENS {
            break;
        }
    }

    // Encode each position as 8-dim feature vector
    filtered.iter().map(|c| {
        let dx = c.pos.x - self_pos.x;
        let dy = c.pos.y - self_pos.y;
        let euclidean = (dx * dx + dy * dy).sqrt();

        // Path distance via A*: if straight line is blocked, A* will be longer
        let path_dist = {
            use crate::ai::pathing::next_waypoint;
            // Estimate path distance by checking if waypoint diverges
            let waypoint = next_waypoint(nav, self_pos, c.pos);
            let wp_dist = distance(self_pos, waypoint);
            let remaining = distance(waypoint, c.pos);
            // If waypoint == goal, path is straight; otherwise add waypoint + remaining
            if distance(waypoint, c.pos) < nav.cell_size * 1.5 {
                euclidean // straight path
            } else {
                wp_dist + remaining // approximate via first waypoint
            }
        };

        let elevation = nav.elevation_at_pos(c.pos);
        let cell = nav.cell_of(c.pos);
        let blocked_neighbors = [(1i32,0i32),(-1,0),(0,1),(0,-1)].iter()
            .filter(|(ox, oy)| nav.blocked.contains(&(cell.0 + ox, cell.1 + oy)))
            .count();

        // Wall proximity: min distance to nearest wall in 4 cardinal directions
        let wall_prox = {
            let rays = crate::ai::pathing::raycast_distances(nav, c.pos, 4, 5.0);
            rays.iter().cloned().fold(f32::INFINITY, f32::min)
        };

        // Zone counts at this position
        let hostile_zones = state.zones.iter()
            .filter(|z| z.source_team != unit.team && distance(c.pos, z.position) < 3.0)
            .count();
        let friendly_zones = state.zones.iter()
            .filter(|z| z.source_team == unit.team && distance(c.pos, z.position) < 3.0)
            .count();

        vec![
            dx / 20.0,
            dy / 20.0,
            path_dist / 30.0,
            elevation / 5.0,
            blocked_neighbors as f32 / 3.0,
            wall_prox / 5.0,
            hostile_zones as f32 / 3.0,
            friendly_zones as f32 / 3.0,
        ]
    }).collect()
}

/// A pending threat to a unit: projectile, zone, or cast in progress.
struct Threat {
    /// Position where the threat will land.
    impact_x: f32,
    impact_y: f32,
    /// AoE radius (0 for single-target).
    radius: f32,
    /// Time until damage lands (ms).
    time_to_impact_ms: f32,
    /// Total damage the threat will deal.
    damage: f32,
    /// Whether the threat includes hard CC.
    has_cc: bool,
}

/// Extract all incoming threats as variable-length 8-dim feature vectors.
fn extract_threats_v2(state: &SimState, unit: &UnitState) -> Vec<Vec<f32>> {
    let mut threats: Vec<Threat> = Vec::new();
    let unit_hp = unit.hp.max(1) as f32;

    // 1. In-flight projectiles aimed at or near this unit
    for proj in &state.projectiles {
        // Skip friendly projectiles
        if let Some(source) = state.units.iter().find(|u| u.id == proj.source_id) {
            if source.team == unit.team {
                continue;
            }
        }

        // Check if projectile is heading toward this unit (homing or close to path)
        let headed_at_unit = proj.target_id == unit.id
            || distance(proj.target_position, unit.position) < 2.0;

        if !headed_at_unit {
            // For pierce projectiles, check if unit is near the projectile path
            if proj.pierce {
                let to_x = unit.position.x - proj.position.x;
                let to_y = unit.position.y - proj.position.y;
                let cross = proj.direction.x * to_y - proj.direction.y * to_x;
                let perp_dist = cross.abs();
                let dot = proj.direction.x * to_x + proj.direction.y * to_y;
                if perp_dist > proj.width + 1.0 || dot < 0.0 {
                    continue;
                }
            } else {
                continue;
            }
        }

        let dist_to_target = distance(proj.position, unit.position);
        let time_ms = if proj.speed > 0.0 {
            (dist_to_target / proj.speed) * 1000.0
        } else {
            0.0
        };

        let (dmg, cc) = sum_effects_damage_cc(&proj.on_hit);
        let (arr_dmg, arr_cc) = sum_effects_damage_cc(&proj.on_arrival);

        let arrival_radius = proj.on_arrival.iter().find_map(|ce| {
            ce.area.as_ref().and_then(|a| area_max_radius(a))
        }).unwrap_or(0.0);

        threats.push(Threat {
            impact_x: unit.position.x,
            impact_y: unit.position.y,
            radius: arrival_radius,
            time_to_impact_ms: time_ms,
            damage: (dmg + arr_dmg) as f32,
            has_cc: cc || arr_cc,
        });
    }

    // 2. Hostile zones currently active near the unit
    for zone in &state.zones {
        if zone.source_team == unit.team {
            continue;
        }
        let zone_radius = area_max_radius(&zone.area).unwrap_or(2.0);
        let dist = distance(unit.position, zone.position);
        if dist > zone_radius + 3.0 {
            continue;
        }

        let (dmg, cc) = sum_effects_damage_cc_cond(&zone.effects);
        let time_ms = if zone.tick_interval_ms > 0 {
            (zone.tick_interval_ms - zone.tick_elapsed_ms.min(zone.tick_interval_ms)) as f32
        } else {
            0.0
        };

        threats.push(Threat {
            impact_x: zone.position.x,
            impact_y: zone.position.y,
            radius: zone_radius,
            time_to_impact_ms: time_ms,
            damage: dmg as f32,
            has_cc: cc,
        });
    }

    // 3. Enemy casts in progress — warning zones
    for enemy in &state.units {
        if enemy.team == unit.team || enemy.hp <= 0 {
            continue;
        }
        let Some(cast) = &enemy.casting else { continue };

        let (impact_pos, radius) = match cast.kind {
            CastKind::HeroAbility(idx) => {
                if let Some(slot) = enemy.abilities.get(idx) {
                    let r = ability_aoe_radius(slot);
                    if let Some(pos) = cast.target_pos {
                        (pos, r)
                    } else if cast.target_id == unit.id {
                        (unit.position, r)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            CastKind::Attack | CastKind::Ability | CastKind::Control => {
                if cast.target_id == unit.id {
                    (unit.position, 0.0)
                } else {
                    continue;
                }
            }
            CastKind::Heal => continue,
        };

        if radius > 0.0 && cast.target_pos.is_some() {
            let dist = distance(unit.position, impact_pos);
            if dist > radius + 2.0 {
                continue;
            }
        }

        let (dmg, cc) = match cast.kind {
            CastKind::HeroAbility(idx) => {
                if let Some(slot) = enemy.abilities.get(idx) {
                    sum_effects_damage_cc_cond(&slot.def.effects)
                } else {
                    (0, false)
                }
            }
            CastKind::Attack => (enemy.attack_damage, false),
            CastKind::Ability => (enemy.ability_damage, false),
            CastKind::Control => (0, true),
            CastKind::Heal => (0, false),
        };

        threats.push(Threat {
            impact_x: impact_pos.x,
            impact_y: impact_pos.y,
            radius,
            time_to_impact_ms: cast.remaining_ms as f32,
            damage: dmg as f32,
            has_cc: cc,
        });
    }

    // Sort by time_to_impact (most urgent first)
    threats.sort_by(|a, b| {
        a.time_to_impact_ms.partial_cmp(&b.time_to_impact_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Encode each threat as 8-dim feature vector
    threats.iter().map(|threat| {
        let dx = threat.impact_x - unit.position.x;
        let dy = threat.impact_y - unit.position.y;
        let dist = (dx * dx + dy * dy).sqrt();
        vec![
            dx / 10.0,
            dy / 10.0,
            dist / 10.0,
            threat.radius / 5.0,
            threat.time_to_impact_ms / 2000.0,
            threat.damage / unit_hp, // damage_ratio
            if threat.has_cc { 1.0 } else { 0.0 },
            1.0, // exists
        ]
    }).collect()
}

/// Sum instant damage and detect CC from conditional effects.
fn sum_effects_damage_cc_cond(effects: &[crate::ai::effects::types::ConditionalEffect]) -> (i32, bool) {
    let mut dmg = 0i32;
    let mut cc = false;
    for ce in effects {
        match &ce.effect {
            Effect::Damage { amount, .. } => dmg += amount,
            Effect::Stun { .. } | Effect::Root { .. } | Effect::Silence { .. } => cc = true,
            _ => {}
        }
    }
    (dmg, cc)
}

/// Sum damage and detect CC from projectile on_hit effects.
fn sum_effects_damage_cc(effects: &[crate::ai::effects::types::ConditionalEffect]) -> (i32, bool) {
    sum_effects_damage_cc_cond(effects)
}

/// Get the maximum AoE radius from an Area shape (public for token info building).
pub fn area_max_radius_pub(area: &Area) -> Option<f32> {
    area_max_radius(area)
}

/// Get the maximum AoE radius from an Area shape.
fn area_max_radius(area: &Area) -> Option<f32> {
    match area {
        Area::Circle { radius } => Some(*radius),
        Area::Cone { radius, .. } => Some(*radius),
        Area::Line { length, .. } => Some(*length),
        Area::Ring { outer_radius, .. } => Some(*outer_radius),
        Area::Spread { radius, .. } => Some(*radius),
        Area::SingleTarget | Area::SelfOnly => None,
    }
}

/// Get the AoE radius of an ability from its effects.
fn ability_aoe_radius(slot: &crate::ai::effects::AbilitySlot) -> f32 {
    // Check delivery first
    if let Some(ref delivery) = slot.def.delivery {
        match delivery {
            crate::ai::effects::Delivery::Zone { .. } => {
                // Zone radius comes from the effects' area shapes
                for ce in &slot.def.effects {
                    if let Some(r) = ce.area.as_ref().and_then(|a| area_max_radius(a)) {
                        return r;
                    }
                }
            }
            _ => {}
        }
    }
    // Check effect areas
    for ce in &slot.def.effects {
        if let Some(r) = ce.area.as_ref().and_then(|a| area_max_radius(a)) {
            return r;
        }
    }
    0.0
}

/// Summary of a unit's strongest ability/heal/CC derived from AbilitySlot DSL effects.
struct AbilitySummary {
    /// Highest single-hit damage from any damage ability
    ability_damage: f32,
    /// Range of that ability
    ability_range: f32,
    /// Cooldown fraction remaining (0 = ready) of strongest damage ability
    ability_cd_pct: f32,
    /// Highest single-hit heal from any heal ability
    heal_amount: f32,
    /// Range of that heal ability
    heal_range: f32,
    /// Cooldown fraction remaining of strongest heal ability
    heal_cd_pct: f32,
    /// Range of strongest CC ability
    control_range: f32,
    /// Duration of strongest CC (stun/root/silence)
    control_duration_ms: f32,
    /// Cooldown fraction remaining of strongest CC ability
    control_cd_pct: f32,
}

/// Scan a unit's abilities to extract strongest damage/heal/CC stats.
///
/// Falls back to legacy flat fields if the unit has no DSL abilities
/// (e.g. old-style PvE enemies).
fn summarize_abilities(unit: &UnitState) -> AbilitySummary {
    let mut best_dmg = 0i32;
    let mut dmg_range = 0.0f32;
    let mut dmg_cd_pct = 0.0f32;

    let mut best_heal = 0i32;
    let mut heal_range = 0.0f32;
    let mut heal_cd_pct = 0.0f32;

    let mut best_cc_dur = 0u32;
    let mut cc_range = 0.0f32;
    let mut cc_cd_pct = 0.0f32;

    for slot in &unit.abilities {
        let hint = slot.def.ai_hint.as_str();
        let cd_pct = if slot.def.cooldown_ms > 0 {
            slot.cooldown_remaining_ms as f32 / slot.def.cooldown_ms as f32
        } else {
            0.0
        };

        match hint {
            "damage" | "damage_unit" | "damage_aoe" => {
                let dmg = max_damage_from_effects(&slot.def.effects);
                if dmg > best_dmg {
                    best_dmg = dmg;
                    dmg_range = slot.def.range;
                    dmg_cd_pct = cd_pct;
                }
            }
            "heal" | "heal_unit" | "heal_aoe" => {
                let heal = max_heal_from_effects(&slot.def.effects);
                if heal > best_heal {
                    best_heal = heal;
                    heal_range = slot.def.range;
                    heal_cd_pct = cd_pct;
                }
            }
            "control" | "cc" | "crowd_control" => {
                let dur = max_cc_duration_from_effects(&slot.def.effects);
                if dur > best_cc_dur {
                    best_cc_dur = dur;
                    cc_range = slot.def.range;
                    cc_cd_pct = cd_pct;
                }
            }
            _ => {
                // For untagged abilities, check effects for any damage/heal/CC
                let dmg = max_damage_from_effects(&slot.def.effects);
                if dmg > best_dmg {
                    best_dmg = dmg;
                    dmg_range = slot.def.range;
                    dmg_cd_pct = cd_pct;
                }
                let heal = max_heal_from_effects(&slot.def.effects);
                if heal > best_heal {
                    best_heal = heal;
                    heal_range = slot.def.range;
                    heal_cd_pct = cd_pct;
                }
                let dur = max_cc_duration_from_effects(&slot.def.effects);
                if dur > best_cc_dur {
                    best_cc_dur = dur;
                    cc_range = slot.def.range;
                    cc_cd_pct = cd_pct;
                }
            }
        }
    }

    // Fall back to legacy flat fields if no DSL abilities found
    if unit.abilities.is_empty() {
        let ability_cd_pct = if unit.ability_cooldown_ms > 0 {
            unit.ability_cooldown_remaining_ms as f32 / unit.ability_cooldown_ms as f32
        } else { 0.0 };
        let legacy_heal_cd = if unit.heal_cooldown_ms > 0 {
            unit.heal_cooldown_remaining_ms as f32 / unit.heal_cooldown_ms as f32
        } else { 0.0 };
        let legacy_cc_cd = if unit.control_cooldown_ms > 0 {
            unit.control_cooldown_remaining_ms as f32 / unit.control_cooldown_ms as f32
        } else { 0.0 };

        return AbilitySummary {
            ability_damage: unit.ability_damage as f32,
            ability_range: unit.ability_range,
            ability_cd_pct,
            heal_amount: unit.heal_amount as f32,
            heal_range: unit.heal_range,
            heal_cd_pct: legacy_heal_cd,
            control_range: unit.control_range,
            control_duration_ms: unit.control_duration_ms as f32,
            control_cd_pct: legacy_cc_cd,
        };
    }

    AbilitySummary {
        ability_damage: best_dmg as f32,
        ability_range: dmg_range,
        ability_cd_pct: dmg_cd_pct,
        heal_amount: best_heal as f32,
        heal_range,
        heal_cd_pct,
        control_range: cc_range,
        control_duration_ms: best_cc_dur as f32,
        control_cd_pct: cc_cd_pct,
    }
}

fn max_damage_from_effects(effects: &[crate::ai::effects::types::ConditionalEffect]) -> i32 {
    effects.iter().filter_map(|ce| match &ce.effect {
        Effect::Damage { amount, .. } => Some(*amount),
        _ => None,
    }).max().unwrap_or(0)
}

fn max_heal_from_effects(effects: &[crate::ai::effects::types::ConditionalEffect]) -> i32 {
    effects.iter().filter_map(|ce| match &ce.effect {
        Effect::Heal { amount, .. } => Some(*amount),
        _ => None,
    }).max().unwrap_or(0)
}

fn max_cc_duration_from_effects(effects: &[crate::ai::effects::types::ConditionalEffect]) -> u32 {
    effects.iter().filter_map(|ce| match &ce.effect {
        Effect::Stun { duration_ms } => Some(*duration_ms),
        Effect::Root { duration_ms } => Some(*duration_ms),
        Effect::Silence { duration_ms } => Some(*duration_ms),
        _ => None,
    }).max().unwrap_or(0)
}

fn rich_entity_features(
    state: &SimState,
    caster: &UnitState,
    unit: &UnitState,
    is_self: bool,
) -> [f32; ENTITY_FEATURE_DIM] {
    let max_hp = unit.max_hp.max(1) as f32;

    // Zone proximity
    let hostile_zones = state.zones.iter()
        .filter(|z| z.source_team != unit.team && distance(unit.position, z.position) < 3.0)
        .count();
    let friendly_zones = state.zones.iter()
        .filter(|z| z.source_team == unit.team && distance(unit.position, z.position) < 3.0)
        .count();

    // Cast progress: remaining_ms normalized (lower = closer to firing)
    let (is_casting, cast_remaining_norm) = match &unit.casting {
        Some(cs) => (1.0, cs.remaining_ms as f32 / 2000.0),
        None => (0.0, 0.0),
    };

    // Cooldown remaining fractions
    let attack_cd_pct = if unit.attack_cooldown_ms > 0 {
        unit.cooldown_remaining_ms as f32 / unit.attack_cooldown_ms as f32
    } else { 0.0 };

    // Derive ability/heal/CC stats from DSL abilities (or legacy fields as fallback)
    let abil = summarize_abilities(unit);

    [
        // Vitals (0-4)
        unit.hp as f32 / max_hp,
        unit.shield_hp as f32 / max_hp,
        if unit.max_resource > 0 { unit.resource as f32 / unit.max_resource as f32 } else { 1.0 },
        unit.armor / 200.0,
        unit.magic_resist / 200.0,
        // Position / terrain (5-11)
        unit.position.x / 20.0,
        unit.position.y / 20.0,
        if is_self { 0.0 } else { distance(caster.position, unit.position) / 10.0 },
        unit.cover_bonus,
        unit.elevation / 5.0,
        hostile_zones as f32 / 3.0,
        friendly_zones as f32 / 3.0,
        // Combat stats (12-14)
        unit_dps(unit) / 30.0,
        unit.attack_range / 10.0,
        attack_cd_pct,
        // Ability (15-17)
        abil.ability_damage / 50.0,
        abil.ability_range / 10.0,
        abil.ability_cd_pct,
        // Healing (18-20)
        abil.heal_amount / 50.0,
        abil.heal_range / 10.0,
        abil.heal_cd_pct,
        // CC capability (21-23)
        abil.control_range / 10.0,
        abil.control_duration_ms / 2000.0,
        abil.control_cd_pct,
        // Current state (24-27)
        is_casting,
        cast_remaining_norm,
        unit.control_remaining_ms as f32 / 2000.0,
        unit.move_speed_per_sec / 5.0,
        // Cumulative (28-29)
        unit.total_damage_done as f32 / 1000.0,
        1.0, // exists
    ]
}

// ---------------------------------------------------------------------------
// Outcome prediction dataset
// ---------------------------------------------------------------------------

use serde::{Serialize, Deserialize};

/// A game state snapshot labeled with the final fight outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeSample {
    /// Flat entity features: [self(30), enemy0(30)..enemy2(30), ally0(30)..ally2(30)]
    pub game_state: Vec<f32>,
    /// 1.0 = hero team wins, 0.0 = enemy team wins
    pub hero_wins: f32,
    /// Hero team HP fraction at fight end (0-1, richer signal than binary)
    pub hero_hp_remaining: f32,
    /// How far through the fight (0=start, 1=end)
    pub fight_progress: f32,
    /// Scenario name for debugging
    pub scenario: String,
    pub tick: u64,
}

/// Generate outcome prediction training data from a simulation run.
///
/// Runs the sim to completion, then walks back through snapshots labeling
/// each with the final outcome. Samples every `sample_interval` ticks
/// to control dataset size.
pub fn generate_outcome_dataset(
    initial_sim: SimState,
    initial_squad_ai: crate::ai::squad::SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    sample_interval: u64,
) -> Vec<OutcomeSample> {
    use crate::ai::core::{step, FIXED_TICK_MS};
    use crate::ai::squad::generate_intents;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut snapshots: Vec<(u64, Vec<Vec<f32>>)> = Vec::new();

    // Run simulation, collecting snapshots
    for tick in 0..max_ticks {
        let intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        // Sample game state from each hero's perspective
        if tick % sample_interval == 0 {
            let hero_states: Vec<Vec<f32>> = sim.units.iter()
                .filter(|u| u.team == Team::Hero && is_alive(u))
                .map(|u| extract_game_state(&sim, u))
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

    // Determine outcome
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

    // Label each snapshot with outcome
    let mut samples = Vec::new();
    for (tick, hero_states) in snapshots {
        for gs in hero_states {
            samples.push(OutcomeSample {
                game_state: gs,
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

/// Write outcome samples as JSONL.
pub fn write_outcome_dataset(
    samples: &[OutcomeSample],
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

// ---------------------------------------------------------------------------
// Next-state prediction dataset
// ---------------------------------------------------------------------------

/// Dense game state snapshot for next-state prediction.
/// Sampled frequently; Python pairs snapshots at different deltas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextStateSample {
    /// Per-entity feature vectors (30-dim each), variable length.
    pub entities: Vec<Vec<f32>>,
    /// Type ID per entity: 0=self, 1=enemy, 2=ally.
    pub entity_types: Vec<u8>,
    /// Unit ID per entity slot (for cross-time alignment).
    pub entity_unit_ids: Vec<u32>,
    /// Per-threat feature vectors (8-dim each).
    pub threats: Vec<Vec<f32>>,
    /// Per-position feature vectors (8-dim each).
    #[serde(default)]
    pub positions: Vec<Vec<f32>>,
    /// All unit HP ratios indexed by unit ID: [[id, hp_pct], ...].
    pub unit_hps: Vec<(u32, f32)>,
    /// Scenario name.
    pub scenario: String,
    /// Tick number.
    pub tick: u64,
}

/// Per-unit ability DSL text, emitted once per scenario.
/// Maps unit_id → list of DSL strings (input to ability transformer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityRegistryEntry {
    pub unit_id: u32,
    pub abilities: Vec<String>,  // DSL text per ability
}

/// Full ability registry for a scenario, emitted before tick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityRegistry {
    pub scenario: String,
    pub entries: Vec<AbilityRegistryEntry>,
}

/// Extract game state V2 + unit IDs for next-state prediction.
fn extract_game_state_with_ids(
    state: &SimState,
    unit: &UnitState,
) -> (GameStateV2, Vec<u32>) {
    let mut entities: Vec<Vec<f32>> = Vec::new();
    let mut entity_types: Vec<u8> = Vec::new();
    let mut unit_ids: Vec<u32> = Vec::new();

    // Self entity
    entities.push(rich_entity_features(state, unit, unit, true).to_vec());
    entity_types.push(0);
    unit_ids.push(unit.id);

    // All enemies sorted by distance
    let mut enemies: Vec<&UnitState> = state
        .units
        .iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    enemies.sort_by(|a, b| {
        distance(unit.position, a.position)
            .partial_cmp(&distance(unit.position, b.position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for enemy in &enemies {
        entities.push(rich_entity_features(state, unit, enemy, false).to_vec());
        entity_types.push(1);
        unit_ids.push(enemy.id);
    }

    // All allies (excluding self) sorted by HP% ascending
    let mut allies: Vec<&UnitState> = state
        .units
        .iter()
        .filter(|u| u.team == unit.team && is_alive(u) && u.id != unit.id)
        .collect();
    allies.sort_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    for ally in &allies {
        entities.push(rich_entity_features(state, unit, ally, false).to_vec());
        entity_types.push(2);
        unit_ids.push(ally.id);
    }

    let threats = extract_threats_v2(state, unit);
    let positions = extract_position_tokens(state, unit);

    let gs = GameStateV2 {
        entities,
        entity_types,
        threats,
        positions,
    };
    (gs, unit_ids)
}

/// Generate next-state prediction dataset, streaming via callback.
///
/// Samples game state densely (every `sample_interval` ticks) with unit IDs
/// and per-unit HP snapshots. Calls `emit` for each sample instead of
/// collecting, to avoid OOM on long fights.
pub fn generate_nextstate_dataset_streaming(
    initial_sim: SimState,
    initial_squad_ai: crate::ai::squad::SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    sample_interval: u64,
    mut emit: impl FnMut(NextStateSample),
    mut emit_registry: impl FnMut(AbilityRegistry),
) -> usize {
    use crate::ai::core::{step, FIXED_TICK_MS};
    use crate::ai::squad::generate_intents;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut count = 0usize;

    // Emit ability registry once before tick loop
    use crate::ai::effects::dsl::emit::emit_ability_dsl;
    let entries: Vec<AbilityRegistryEntry> = sim.units.iter().map(|u| {
        let abilities = u.abilities.iter().map(|slot| {
            emit_ability_dsl(&slot.def)
        }).collect();
        AbilityRegistryEntry { unit_id: u.id, abilities }
    }).collect();
    emit_registry(AbilityRegistry {
        scenario: scenario_name.to_string(),
        entries,
    });

    for tick in 0..max_ticks {
        let intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        if tick % sample_interval == 0 {
            let unit_hps: Vec<(u32, f32)> = sim
                .units
                .iter()
                .map(|u| (u.id, u.hp.max(0) as f32 / u.max_hp.max(1) as f32))
                .collect();

            let heroes: Vec<&UnitState> = sim
                .units
                .iter()
                .filter(|u| u.team == Team::Hero && is_alive(u))
                .collect();

            for hero in &heroes {
                let (gs, ids) = extract_game_state_with_ids(&sim, hero);
                emit(NextStateSample {
                    entities: gs.entities,
                    entity_types: gs.entity_types,
                    entity_unit_ids: ids,
                    threats: gs.threats,
                    positions: gs.positions,
                    unit_hps: unit_hps.clone(),
                    scenario: scenario_name.to_string(),
                    tick,
                });
                count += 1;
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

    count
}
