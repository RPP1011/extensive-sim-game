use crate::sim::{distance, SimState, UnitState};
use crate::sim::types::CastKind;
use crate::effects::effect_enum::Effect;
use crate::effects::types::Area;

// ---------------------------------------------------------------------------
// Threat extraction
// ---------------------------------------------------------------------------

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
    /// Threat kind: zone=0.25, obstacle=0.5, cast_indicator=0.75, projectile=1.0
    kind: f32,
}

/// Extract all incoming threats as variable-length 8-dim feature vectors.
pub(super) fn extract_threats_v2(state: &SimState, unit: &UnitState) -> Vec<Vec<f32>> {
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
            kind: 1.0, // projectile
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

        let is_obstacle = !zone.blocked_cells.is_empty();
        threats.push(Threat {
            impact_x: zone.position.x,
            impact_y: zone.position.y,
            radius: zone_radius,
            time_to_impact_ms: time_ms,
            damage: dmg as f32,
            has_cc: cc,
            kind: if is_obstacle { 0.5 } else { 0.25 }, // obstacle or zone
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
            kind: 0.75, // cast indicator
        });
    }

    // Sort by time_to_impact (most urgent first)
    threats.sort_by(|a, b| {
        a.time_to_impact_ms.partial_cmp(&b.time_to_impact_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Encode each threat as 10-dim feature vector
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
            threat.kind, // zone=0.25, obstacle=0.5, cast=0.75, projectile=1.0
            1.0, // line_of_sight (always visible in current extraction)
        ]
    }).collect()
}

/// Sum instant damage and detect CC from conditional effects.
pub(super) fn sum_effects_damage_cc_cond(effects: &[crate::effects::types::ConditionalEffect]) -> (i32, bool) {
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
fn sum_effects_damage_cc(effects: &[crate::effects::types::ConditionalEffect]) -> (i32, bool) {
    sum_effects_damage_cc_cond(effects)
}

/// Get the maximum AoE radius from an Area shape (public for token info building).
pub fn area_max_radius_pub(area: &Area) -> Option<f32> {
    area_max_radius(area)
}

/// Get the maximum AoE radius from an Area shape.
pub(super) fn area_max_radius(area: &Area) -> Option<f32> {
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
pub(super) fn ability_aoe_radius(slot: &crate::effects::AbilitySlot) -> f32 {
    // Check delivery first
    if let Some(ref delivery) = slot.def.delivery {
        match delivery {
            crate::effects::Delivery::Zone { .. } => {
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
