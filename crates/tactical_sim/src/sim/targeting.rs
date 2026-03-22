use crate::effects::{AbilityTarget, Area, Effect, StatusKind};

use super::types::*;
use super::math::distance;

pub fn resolve_targets(
    area: Option<&Area>,
    caster_idx: usize,
    target: AbilityTarget,
    caster_team: Team,
    effect: &Effect,
    state: &SimState,
) -> Vec<u32> {
    let area = area.cloned().unwrap_or(Area::SingleTarget);
    let caster_pos = state.units[caster_idx].position;

    // Determine if this effect targets allies or enemies
    let is_heal = matches!(effect,
        Effect::Heal { .. } | Effect::Shield { .. } | Effect::Buff { .. }
        | Effect::Resurrect { .. } | Effect::OverhealShield { .. } | Effect::AbsorbToHeal { .. }
        | Effect::Lifesteal { .. } | Effect::Reflect { .. } | Effect::Immunity { .. }
        | Effect::Stealth { .. } | Effect::Redirect { .. } | Effect::Link { .. }
    );
    // Resurrect targets dead allies
    let targets_dead = matches!(effect, Effect::Resurrect { .. });

    match area {
        Area::SingleTarget => {
            match target {
                AbilityTarget::Unit(tid) => vec![tid],
                AbilityTarget::None => vec![state.units[caster_idx].id],
                AbilityTarget::Position(_) => vec![],
            }
        }
        Area::SelfOnly => {
            vec![state.units[caster_idx].id]
        }
        Area::Circle { radius } => {
            let center = match target {
                AbilityTarget::Position(p) => p,
                AbilityTarget::Unit(tid) => {
                    state.units.iter().find(|u| u.id == tid)
                        .map_or(caster_pos, |u| u.position)
                }
                AbilityTarget::None => caster_pos,
            };
            units_in_radius(state, center, radius, caster_team, is_heal)
        }
        Area::Cone { radius, angle_deg } => {
            let target_pos = match target {
                AbilityTarget::Unit(tid) => {
                    state.units.iter().find(|u| u.id == tid)
                        .map_or(caster_pos, |u| u.position)
                }
                AbilityTarget::Position(p) => p,
                AbilityTarget::None => return vec![],
            };
            let dir_x = target_pos.x - caster_pos.x;
            let dir_y = target_pos.y - caster_pos.y;
            let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
            if dir_len < f32::EPSILON {
                return vec![];
            }
            let half_angle = (angle_deg / 2.0).to_radians();
            let cos_half = half_angle.cos();

            state.units.iter()
                .filter(|u| u.hp > 0 && u.id != state.units[caster_idx].id)
                .filter(|u| if is_heal { u.team == caster_team } else { u.team != caster_team })
                .filter(|u| {
                    let d = distance(caster_pos, u.position);
                    if d > radius { return false; }
                    let ux = u.position.x - caster_pos.x;
                    let uy = u.position.y - caster_pos.y;
                    let dot = (dir_x * ux + dir_y * uy) / (dir_len * d.max(f32::EPSILON));
                    dot >= cos_half
                })
                .map(|u| u.id)
                .collect()
        }
        Area::Line { length, width } => {
            let target_pos = match target {
                AbilityTarget::Unit(tid) => {
                    state.units.iter().find(|u| u.id == tid)
                        .map_or(caster_pos, |u| u.position)
                }
                AbilityTarget::Position(p) => p,
                AbilityTarget::None => return vec![],
            };
            let dir_x = target_pos.x - caster_pos.x;
            let dir_y = target_pos.y - caster_pos.y;
            let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
            if dir_len < f32::EPSILON {
                return vec![];
            }
            let nx = dir_x / dir_len;
            let ny = dir_y / dir_len;
            let half_w = width / 2.0;

            state.units.iter()
                .filter(|u| u.hp > 0 && u.id != state.units[caster_idx].id)
                .filter(|u| if is_heal { u.team == caster_team } else { u.team != caster_team })
                .filter(|u| {
                    let ux = u.position.x - caster_pos.x;
                    let uy = u.position.y - caster_pos.y;
                    let proj = ux * nx + uy * ny;
                    if proj < 0.0 || proj > length { return false; }
                    let perp = (ux * (-ny) + uy * nx).abs();
                    perp <= half_w
                })
                .map(|u| u.id)
                .collect()
        }
        Area::Ring { inner_radius, outer_radius } => {
            let center = match target {
                AbilityTarget::Position(p) => p,
                AbilityTarget::Unit(tid) => {
                    state.units.iter().find(|u| u.id == tid)
                        .map_or(caster_pos, |u| u.position)
                }
                AbilityTarget::None => caster_pos,
            };
            state.units.iter()
                .filter(|u| (targets_dead || u.hp > 0) && u.id != state.units[caster_idx].id)
                .filter(|u| if is_heal { u.team == caster_team } else { u.team != caster_team })
                .filter(|u| {
                    let d = distance(center, u.position);
                    d >= inner_radius && d <= outer_radius
                })
                .map(|u| u.id)
                .collect()
        }
        Area::Spread { radius, max_targets } => {
            let center = match target {
                AbilityTarget::Position(p) => p,
                AbilityTarget::Unit(tid) => {
                    state.units.iter().find(|u| u.id == tid)
                        .map_or(caster_pos, |u| u.position)
                }
                AbilityTarget::None => caster_pos,
            };
            let caster_id = state.units[caster_idx].id;
            let mut targets: Vec<u32> = state.units.iter()
                .filter(|u| u.hp > 0 && u.id != caster_id)
                .filter(|u| if is_heal { u.team == caster_team } else { u.team != caster_team })
                .filter(|u| distance(center, u.position) <= radius)
                .map(|u| u.id)
                .collect();
            if max_targets > 0 && targets.len() > max_targets as usize {
                targets.truncate(max_targets as usize);
            }
            targets
        }
    }
}

pub fn units_in_radius(
    state: &SimState,
    center: SimVec2,
    radius: f32,
    caster_team: Team,
    target_allies: bool,
) -> Vec<u32> {
    state.units.iter()
        .filter(|u| u.hp > 0)
        .filter(|u| {
            if target_allies { u.team == caster_team } else { u.team != caster_team }
        })
        // Skip stealthed enemies
        .filter(|u| {
            if !target_allies && u.team != caster_team {
                !u.status_effects.iter().any(|s| matches!(s.kind, StatusKind::Stealth { .. }))
            } else { true }
        })
        // Skip banished units
        .filter(|u| !u.status_effects.iter().any(|s| matches!(s.kind, StatusKind::Banish)))
        .filter(|u| distance(center, u.position) <= radius)
        .map(|u| u.id)
        .collect()
}
