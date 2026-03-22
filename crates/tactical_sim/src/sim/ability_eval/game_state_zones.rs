use crate::sim::{distance, SimState, UnitState};
use crate::sim::types::CastKind;
use crate::effects::effect_enum::Effect;
use super::game_state_threats::{
    area_max_radius, ability_aoe_radius, sum_effects_damage_cc_cond,
};

/// Minimal objective info needed for zone token extraction.
/// Avoids depending on the scenario module directly.
pub struct ZoneObjective {
    pub position: [f32; 2],
    pub radius: f32,
}

// ---------------------------------------------------------------------------
// Unified zone token extraction
// ---------------------------------------------------------------------------

/// Maximum number of zone tokens emitted per unit.
pub const MAX_ZONE_TOKENS: usize = 10;
/// Feature dimension per zone token.
pub const ZONE_DIM: usize = 12;

// Kind values (8 categories, linearly spaced in [0, 1])
const KIND_DAMAGE: f32 = 0.0;
const KIND_CC: f32 = 0.14;
const KIND_OBSTACLE: f32 = 0.29;
const KIND_PROJECTILE: f32 = 0.43;
const KIND_CAST: f32 = 0.57;
const KIND_HEALING: f32 = 0.71;
const KIND_COVER: f32 = 0.86;
const KIND_OBJECTIVE: f32 = 1.0;

// Hint values (6 categories, linearly spaced in [0, 1])
const HINT_AVOID: f32 = 0.0;
const HINT_DODGE: f32 = 0.2;
const HINT_APPROACH: f32 = 0.4;
const HINT_HOLD: f32 = 0.6;
const HINT_CONTEST: f32 = 0.8;
#[allow(dead_code)]
const HINT_NEUTRAL: f32 = 1.0;

/// Internal zone candidate with priority metadata.
struct ZoneCandidate {
    features: [f32; ZONE_DIM],
    /// Priority tier: lower = higher priority.
    /// 0 = imminent threats (<500ms), 1 = objectives, 2 = friendly zones,
    /// 3 = non-imminent threats, 4 = cover
    tier: u8,
    /// Secondary sort within tier: lower = more urgent/closer.
    urgency: f32,
}

/// Extract unified zone tokens for a unit, combining threats, opportunities,
/// objectives, and cover into a single token type.
///
/// Zone features (12):
///   0: dx / 20
///   1: dy / 20
///   2: distance / 20
///   3: radius / 10
///   4: intensity [0,1] — damage/hp, heal/hp, 1.0 for objectives
///   5: urgency / 2000 — time_to_impact, remaining_ms
///   6: kind [0,1] — 8 values
///   7: hint [0,1] — 6 values
///   8: friendliness {-1, 0, 1}
///   9: has_cc {0, 1}
///  10: terrain_quality [0,1]
///  11: exists {0, 1}
pub fn extract_zone_tokens(
    state: &SimState,
    unit: &UnitState,
    objectives: &[ZoneObjective],
) -> Vec<Vec<f32>> {
    let mut candidates: Vec<ZoneCandidate> = Vec::new();
    let unit_hp = unit.hp.max(1) as f32;
    let nav = state.grid_nav.as_ref();

    // --- 1. Hostile damage/CC zones ---
    for zone in &state.zones {
        if zone.source_team == unit.team {
            // Friendly zones handled below
            continue;
        }
        let zone_radius = area_max_radius(&zone.area).unwrap_or(2.0);
        let dist = distance(unit.position, zone.position);
        if dist > zone_radius + 5.0 {
            continue;
        }

        let (dmg, cc) = sum_effects_damage_cc_cond(&zone.effects);
        let time_ms = if zone.tick_interval_ms > 0 {
            (zone.tick_interval_ms - zone.tick_elapsed_ms.min(zone.tick_interval_ms)) as f32
        } else {
            0.0
        };

        let is_obstacle = !zone.blocked_cells.is_empty();
        let (kind, hint) = if is_obstacle {
            (KIND_OBSTACLE, HINT_AVOID)
        } else if cc {
            (KIND_CC, HINT_AVOID)
        } else {
            (KIND_DAMAGE, HINT_AVOID)
        };

        let intensity = (dmg as f32 / unit_hp).clamp(0.0, 1.0);
        let imminent = time_ms < 500.0;
        let tier = if imminent { 0 } else { 3 };
        let terrain_q = terrain_quality_at(nav, zone.position);

        candidates.push(ZoneCandidate {
            features: encode_zone(
                unit, zone.position.x, zone.position.y, zone_radius,
                intensity, time_ms, kind, hint, -1.0, cc, terrain_q,
            ),
            tier,
            urgency: time_ms,
        });
    }

    // --- 2. In-flight projectiles ---
    for proj in &state.projectiles {
        // Skip friendly projectiles
        if let Some(source) = state.units.iter().find(|u| u.id == proj.source_id) {
            if source.team == unit.team {
                continue;
            }
        }

        let headed_at_unit = proj.target_id == unit.id
            || distance(proj.target_position, unit.position) < 2.0;

        if !headed_at_unit {
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

        let (dmg, cc) = sum_effects_damage_cc_cond(&proj.on_hit);
        let (arr_dmg, arr_cc) = sum_effects_damage_cc_cond(&proj.on_arrival);
        let total_dmg = (dmg + arr_dmg) as f32;
        let intensity = (total_dmg / unit_hp).clamp(0.0, 1.0);
        let has_cc = cc || arr_cc;

        let arrival_radius = proj.on_arrival.iter().find_map(|ce| {
            ce.area.as_ref().and_then(|a| area_max_radius(a))
        }).unwrap_or(0.0);

        let imminent = time_ms < 500.0;
        let terrain_q = terrain_quality_at(nav, unit.position);

        candidates.push(ZoneCandidate {
            features: encode_zone(
                unit, unit.position.x, unit.position.y, arrival_radius,
                intensity, time_ms, KIND_PROJECTILE, HINT_DODGE, -1.0, has_cc, terrain_q,
            ),
            tier: if imminent { 0 } else { 3 },
            urgency: time_ms,
        });
    }

    // --- 3. Enemy casts in progress ---
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

        let intensity = (dmg as f32 / unit_hp).clamp(0.0, 1.0);
        let time_ms = cast.remaining_ms as f32;
        let imminent = time_ms < 500.0;
        let hint = if radius > 0.0 { HINT_DODGE } else { HINT_AVOID };
        let terrain_q = terrain_quality_at(nav, impact_pos);

        candidates.push(ZoneCandidate {
            features: encode_zone(
                unit, impact_pos.x, impact_pos.y, radius,
                intensity, time_ms, KIND_CAST, hint, -1.0, cc, terrain_q,
            ),
            tier: if imminent { 0 } else { 3 },
            urgency: time_ms,
        });
    }

    // --- 4. Friendly healing zones ---
    let hp_pct = unit.hp as f32 / unit.max_hp.max(1) as f32;
    for zone in &state.zones {
        if zone.source_team != unit.team {
            continue;
        }
        // Check if zone has healing effects
        let mut heal_amount = 0i32;
        for ce in &zone.effects {
            if let Effect::Heal { amount, .. } = &ce.effect {
                heal_amount += amount;
            }
        }
        if heal_amount == 0 {
            continue;
        }

        let zone_radius = area_max_radius(&zone.area).unwrap_or(2.0);
        let dist = distance(unit.position, zone.position);
        if dist > zone_radius + 5.0 {
            continue;
        }

        let intensity = (heal_amount as f32 / unit_hp).clamp(0.0, 1.0);
        let time_ms = zone.remaining_ms as f32;
        let terrain_q = terrain_quality_at(nav, zone.position);

        candidates.push(ZoneCandidate {
            features: encode_zone(
                unit, zone.position.x, zone.position.y, zone_radius,
                intensity, time_ms, KIND_HEALING, HINT_APPROACH, 1.0, false, terrain_q,
            ),
            // Only include friendly zones when HP < 80%
            tier: 2,
            urgency: -intensity, // higher heal = more urgent (lower sort key)
        });
    }

    // --- 5. Objectives ---
    for obj in objectives {
        let pos = obj.position;
        let radius = obj.radius;
        let terrain_q = terrain_quality_at(
            nav,
            crate::sim::sim_vec2(pos[0], pos[1]),
        );
        candidates.push(ZoneCandidate {
            features: encode_zone(
                unit, pos[0], pos[1], radius,
                1.0, 0.0, KIND_OBJECTIVE, HINT_CONTEST, 0.0, false, terrain_q,
            ),
            tier: 1,
            urgency: 0.0,
        });
    }

    // --- 6. Cover positions (only when nav available) ---
    if let Some(nav) = nav {
        if !nav.fully_open {
            let self_pos = unit.position;
            let move_range = unit.move_speed_per_sec * 1.5;
            let directions: [(f32, f32); 8] = [
                (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707),
            ];

            let mut cover_candidates: Vec<(f32, crate::sim::SimVec2)> = Vec::new();
            for &(dx, dy) in &directions {
                for dist_mult in &[0.5, 1.0] {
                    let d = move_range * dist_mult;
                    let pos = crate::sim::sim_vec2(
                        self_pos.x + dx * d,
                        self_pos.y + dy * d,
                    );
                    if !nav.is_walkable_pos(pos) {
                        continue;
                    }

                    let elevation = nav.elevation_at_pos(pos);
                    let blocked = nav.chokepoint_at_pos(pos) as usize;
                    let choke_val = match blocked {
                        1 => 1.0,
                        2 => 1.5,
                        _ => 0.0,
                    };
                    let score = elevation * 2.0 + choke_val;
                    if score > 0.5 {
                        cover_candidates.push((score, pos));
                    }
                }
            }

            // Deduplicate by proximity, take top candidates
            cover_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut used: Vec<crate::sim::SimVec2> = Vec::new();
            for (score, pos) in cover_candidates {
                if used.iter().any(|&u| distance(pos, u) < 2.0) {
                    continue;
                }
                used.push(pos);
                let terrain_q = terrain_quality_at(Some(nav), pos);
                candidates.push(ZoneCandidate {
                    features: encode_zone(
                        unit, pos.x, pos.y, 0.0,
                        0.0, 0.0, KIND_COVER, HINT_HOLD, 0.0, false, terrain_q,
                    ),
                    tier: 4,
                    urgency: score, // lower score = lower priority within cover tier
                });
                if used.len() >= 4 {
                    break;
                }
            }
        }
    }

    // --- Priority selection: budget per tier ---
    // Filter out friendly zones when HP >= 80% (they're low priority when healthy)
    if hp_pct >= 0.8 {
        candidates.retain(|c| c.tier != 2);
    }

    // Sort by (tier, urgency)
    candidates.sort_by(|a, b| {
        a.tier.cmp(&b.tier)
            .then(a.urgency.partial_cmp(&b.urgency).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Take up to MAX_ZONE_TOKENS with per-tier budgets
    let mut result: Vec<Vec<f32>> = Vec::new();
    let mut imminent_count = 0usize;
    let mut objective_count = 0usize;
    let mut friendly_count = 0usize;

    for c in &candidates {
        if result.len() >= MAX_ZONE_TOKENS {
            break;
        }
        match c.tier {
            0 => {
                if imminent_count >= 4 { continue; }
                imminent_count += 1;
            }
            1 => {
                if objective_count >= 2 { continue; }
                objective_count += 1;
            }
            2 => {
                if friendly_count >= 2 { continue; }
                friendly_count += 1;
            }
            _ => {} // no per-tier limit for non-imminent threats and cover
        }
        result.push(c.features.to_vec());
    }

    result
}

/// Encode a zone candidate into a 12-float feature vector.
fn encode_zone(
    unit: &UnitState,
    zone_x: f32,
    zone_y: f32,
    radius: f32,
    intensity: f32,
    time_ms: f32,
    kind: f32,
    hint: f32,
    friendliness: f32,
    has_cc: bool,
    terrain_quality: f32,
) -> [f32; ZONE_DIM] {
    let dx = zone_x - unit.position.x;
    let dy = zone_y - unit.position.y;
    let dist = (dx * dx + dy * dy).sqrt();
    [
        dx / 20.0,                                    // 0: dx
        dy / 20.0,                                    // 1: dy
        dist / 20.0,                                  // 2: distance
        radius / 10.0,                                // 3: radius
        intensity,                                     // 4: intensity [0,1]
        time_ms / 2000.0,                             // 5: urgency
        kind,                                          // 6: kind [0,1]
        hint,                                          // 7: hint [0,1]
        friendliness,                                  // 8: friendliness {-1,0,1}
        if has_cc { 1.0 } else { 0.0 },              // 9: has_cc
        terrain_quality,                               // 10: terrain_quality [0,1]
        1.0,                                           // 11: exists
    ]
}

/// Compute terrain quality at a position: combines elevation, chokepoint, and wall proximity.
fn terrain_quality_at(
    nav: Option<&crate::pathing::GridNav>,
    pos: crate::sim::SimVec2,
) -> f32 {
    let Some(nav) = nav else { return 0.5 }; // default for no-nav maps
    if nav.fully_open { return 0.5; }

    let elevation = (nav.elevation_at_pos(pos) / 5.0).clamp(0.0, 1.0);
    let choke = (nav.chokepoint_at_pos(pos) as f32 / 3.0).clamp(0.0, 1.0);
    let wall = (nav.wall_proximity_at_pos(pos) / 5.0).clamp(0.0, 1.0);

    // Weighted combination: elevation matters most, then wall coverage, then choke
    (elevation * 0.5 + wall * 0.3 + choke * 0.2).clamp(0.0, 1.0)
}
