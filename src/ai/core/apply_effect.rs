use std::collections::HashMap;

use crate::ai::effects::{
    AbilityTarget, ActiveStatusEffect, Effect, Stacking, StatusKind,
};

use super::types::*;
use super::events::SimEvent;
use super::math::*;
use super::helpers::*;
use super::damage::{apply_typed_damage, apply_heal_to_unit};

pub fn apply_effect(
    effect: &Effect,
    caster_idx: usize,
    target_id: u32,
    ability_target: AbilityTarget,
    tick: u64,
    tags: &crate::ai::effects::Tags,
    stacking: Stacking,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    let caster_id = state.units[caster_idx].id;

    // Immunity check — skip if target has Immunity status covering this effect type
    if let Some(tidx) = find_unit_idx(state, target_id) {
        let effect_type_name = match effect {
            Effect::Stun { .. } => "stun",
            Effect::Slow { .. } => "slow",
            Effect::Root { .. } => "root",
            Effect::Silence { .. } => "silence",
            Effect::Fear { .. } => "fear",
            Effect::Taunt { .. } => "taunt",
            Effect::Damage { .. } => "damage",
            Effect::Blind { .. } => "blind",
            Effect::Polymorph { .. } => "polymorph",
            Effect::Banish { .. } => "banish",
            Effect::Confuse { .. } => "confuse",
            Effect::Charm { .. } => "charm",
            Effect::Suppress { .. } => "suppress",
            Effect::Grounded { .. } => "grounded",
            _ => "",
        };
        if !effect_type_name.is_empty() {
            let immune = state.units[tidx].status_effects.iter().any(|s| {
                if let StatusKind::Immunity { ref immune_to } = s.kind {
                    immune_to.iter().any(|t| t == effect_type_name)
                } else { false }
            });
            if immune {
                events.push(SimEvent::EffectResisted {
                    tick, unit_id: target_id,
                    resisted_tag: format!("immune:{}", effect_type_name),
                });
                return;
            }
        }
    }

    if !apply_effect_primary(effect, caster_idx, caster_id, target_id, ability_target, tick, tags, stacking, state, events) {
        super::apply_effect_ext::apply_effect_extended(effect, caster_idx, caster_id, target_id, ability_target, tick, tags, stacking, state, events);
    }
}

/// Handle core combat effects. Returns true if the effect was handled.
fn apply_effect_primary(
    effect: &Effect,
    caster_idx: usize,
    caster_id: u32,
    target_id: u32,
    ability_target: AbilityTarget,
    tick: u64,
    tags: &crate::ai::effects::Tags,
    stacking: Stacking,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) -> bool {
    match effect {
        Effect::Damage { amount, amount_per_tick, duration_ms, tick_interval_ms, scaling_stat, scaling_percent, damage_type, ref bonus } => {
            if *duration_ms > 0 && *amount_per_tick > 0 {
                if let Some(tidx) = find_unit_idx(state, target_id) {
                    state.units[tidx].status_effects.push(ActiveStatusEffect {
                        kind: StatusKind::Dot {
                            amount_per_tick: *amount_per_tick,
                            tick_interval_ms: *tick_interval_ms,
                            tick_elapsed_ms: 0,
                        },
                        source_id: caster_id,
                        remaining_ms: *duration_ms,
                        tags: tags.clone(),
                        stacking,
                    });
                    events.push(SimEvent::StatusEffectApplied {
                        tick,
                        unit_id: target_id,
                        effect_name: "DoT".to_string(),
                    });
                }
            } else {
                let scaled = compute_scaling(*amount, scaling_stat.as_deref(), *scaling_percent, caster_idx, target_id, state);
                let bonus_amount = resolve_bonus(bonus, caster_idx, target_id, state);
                let total = scaled + bonus_amount;
                if total > 0 {
                    apply_typed_damage(caster_idx, target_id, total, *damage_type, tick, state, events);
                }
            }
        }
        Effect::Heal { amount, amount_per_tick, duration_ms, tick_interval_ms, scaling_stat, scaling_percent, ref bonus } => {
            if *duration_ms > 0 && *amount_per_tick > 0 {
                if let Some(tidx) = find_unit_idx(state, target_id) {
                    state.units[tidx].status_effects.push(ActiveStatusEffect {
                        kind: StatusKind::Hot {
                            amount_per_tick: *amount_per_tick,
                            tick_interval_ms: *tick_interval_ms,
                            tick_elapsed_ms: 0,
                        },
                        source_id: caster_id,
                        remaining_ms: *duration_ms,
                        tags: tags.clone(),
                        stacking,
                    });
                    events.push(SimEvent::StatusEffectApplied {
                        tick,
                        unit_id: target_id,
                        effect_name: "HoT".to_string(),
                    });
                }
            } else {
                let scaled = compute_scaling(*amount, scaling_stat.as_deref(), *scaling_percent, caster_idx, target_id, state);
                let bonus_amount = resolve_bonus(bonus, caster_idx, target_id, state);
                let total = scaled + bonus_amount;
                if total > 0 {
                    apply_heal_to_unit(caster_idx, target_id, total, tick, state, events);
                }
            }
        }
        Effect::Shield { amount, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].shield_hp += amount;
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Shield { amount: *amount },
                    source_id: caster_id,
                    remaining_ms: *duration_ms,
                    tags: tags.clone(),
                    stacking,
                });
                events.push(SimEvent::ShieldApplied { tick, unit_id: target_id, amount: *amount });
            }
        }
        Effect::Stun { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                if !is_alive(&state.units[tidx]) { /* skip stun on dead unit */ }
                else {
                    state.units[tidx].control_remaining_ms =
                        state.units[tidx].control_remaining_ms.max(*duration_ms);
                    state.units[tidx].casting = None;
                    state.units[tidx].status_effects.push(ActiveStatusEffect {
                        kind: StatusKind::Stun,
                        source_id: caster_id,
                        remaining_ms: *duration_ms,
                        tags: tags.clone(),
                        stacking,
                    });
                    events.push(SimEvent::ControlApplied { tick, source_id: caster_id, target_id, duration_ms: *duration_ms });
                }
            }
        }
        Effect::Slow { factor, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Slow { factor: *factor },
                    source_id: caster_id,
                    remaining_ms: *duration_ms,
                    tags: tags.clone(),
                    stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Slow".to_string() });
            }
        }
        Effect::Knockback { distance: dist } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let caster_pos = state.units[caster_idx].position;
                let target_pos = state.units[tidx].position;
                let new_pos = move_away(target_pos, caster_pos, *dist);
                state.units[tidx].position = new_pos;
                events.push(SimEvent::KnockbackApplied { tick, source_id: caster_id, target_id, distance_x100: to_x100(*dist) });
            }
        }
        Effect::Dash { to_target, distance: dash_dist, to_position, is_blink } => {
            // Grounded check: non-blink dashes are blocked
            if !*is_blink {
                let grounded = state.units[caster_idx].status_effects.iter().any(|s|
                    matches!(s.kind, StatusKind::Grounded)
                );
                if grounded {
                    events.push(SimEvent::StatusEffectApplied {
                        tick, unit_id: caster_id,
                        effect_name: "DashBlockedByGrounded".to_string(),
                    });
                    return true;
                }
            }
            let from = state.units[caster_idx].position;
            let effective_dist = if *dash_dist > 0.0 { *dash_dist } else { 2.0 };
            let new_pos = if *to_position {
                if let AbilityTarget::Position(pos) = ability_target {
                    move_towards(from, pos, effective_dist)
                } else if let Some(tidx) = find_unit_idx(state, target_id) {
                    move_towards(from, state.units[tidx].position, effective_dist)
                } else { from }
            } else if *to_target {
                if let Some(tidx) = find_unit_idx(state, target_id) {
                    let target_pos = state.units[tidx].position;
                    position_at_range(from, target_pos, 0.5)
                } else { from }
            } else {
                if let Some(tidx) = find_unit_idx(state, target_id) {
                    move_away(from, state.units[tidx].position, effective_dist)
                } else { from }
            };
            if distance(from, new_pos) > f32::EPSILON {
                state.units[caster_idx].position = new_pos;
                events.push(SimEvent::DashPerformed {
                    tick, unit_id: caster_id,
                    from_x100: to_x100(from.x), from_y100: to_x100(from.y),
                    to_x100: to_x100(new_pos.x), to_y100: to_x100(new_pos.y),
                });
            }
        }
        Effect::Buff { stat, factor, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Buff { stat: stat.clone(), factor: *factor },
                    source_id: caster_id,
                    remaining_ms: *duration_ms,
                    tags: tags.clone(),
                    stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: format!("Buff:{}", stat) });
            }
        }
        Effect::Debuff { stat, factor, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Debuff { stat: stat.clone(), factor: *factor },
                    source_id: caster_id,
                    remaining_ms: *duration_ms,
                    tags: tags.clone(),
                    stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: format!("Debuff:{}", stat) });
            }
        }
        Effect::Duel { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let duel_tags = tags.clone();
                state.units[caster_idx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Duel { partner_id: target_id },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: duel_tags.clone(), stacking,
                });
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Duel { partner_id: caster_id },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: duel_tags, stacking,
                });
                events.push(SimEvent::DuelStarted { tick, unit_a: caster_id, unit_b: target_id });
            }
        }
        Effect::Summon { template, count, hp_percent, clone: is_clone, clone_damage_percent, directed } => {
            let caster_team = state.units[caster_idx].team;
            let caster_pos = state.units[caster_idx].position;
            let next_id = state.units.iter().map(|u| u.id).max().unwrap_or(0) + 1;
            // For directed summons with ground targeting, place at target position
            let spawn_pos = match ability_target {
                AbilityTarget::Position(p) if *directed => p,
                _ => caster_pos,
            };
            for i in 0..*count {
                let summon_id = next_id + i;
                let offset_x = if *directed { 0.0 } else { 1.0 + (i as f32 * 0.5) };
                let mut summon_unit = if *is_clone || template == "self" {
                    let mut c = state.units[caster_idx].clone();
                    c.id = summon_id;
                    c.position = sim_vec2(spawn_pos.x + offset_x, spawn_pos.y);
                    c.hp = (c.max_hp as f32 * hp_percent / 100.0) as i32;
                    if *clone_damage_percent > 0.0 && *clone_damage_percent != 100.0 {
                        c.attack_damage = (c.attack_damage as f32 * clone_damage_percent / 100.0) as i32;
                    }
                    c.status_effects.clear();
                    c.shield_hp = 0;
                    c.state_history.clear();
                    c
                } else {
                    let mut unit = build_summon_template(summon_id, caster_team, template);
                    unit.position = sim_vec2(spawn_pos.x + offset_x, spawn_pos.y);
                    unit.hp = (unit.max_hp as f32 * hp_percent / 100.0) as i32;
                    unit
                };
                summon_unit.owner_id = Some(caster_id);
                summon_unit.directed = *directed;
                if *directed {
                    summon_unit.move_speed_per_sec = 0.0; // Directed summons are stationary
                }
                state.units.push(summon_unit);
                events.push(SimEvent::UnitSummoned { tick, unit_id: summon_id, template: template.clone() });
            }
        }
        Effect::Dispel { target_tags } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let mut removed = 0u32;
                if target_tags.is_empty() {
                    state.units[tidx].status_effects.retain(|s| {
                        let is_negative = match &s.kind {
                            StatusKind::Stun | StatusKind::Slow { .. } | StatusKind::Dot { .. }
                            | StatusKind::Debuff { .. } | StatusKind::Root | StatusKind::Silence
                            | StatusKind::Fear { .. } | StatusKind::Blind { .. }
                            | StatusKind::Polymorph | StatusKind::Confuse
                            | StatusKind::DeathMark { .. } => true,
                            StatusKind::DamageModify { factor } => *factor > 1.0,
                            _ => false,
                        };
                        if is_negative { removed += 1; false } else { true }
                    });
                } else {
                    state.units[tidx].status_effects.retain(|s| {
                        let has_tag = target_tags.iter().any(|t| s.tags.contains_key(t));
                        if has_tag { removed += 1; false } else { true }
                    });
                }
                if removed > 0 {
                    state.units[tidx].control_remaining_ms = 0;
                    events.push(SimEvent::DispelApplied { tick, unit_id: target_id, removed_count: removed });
                }
            }
        }
        Effect::Root { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Root, source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Root".to_string() });
            }
        }
        Effect::Silence { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Silence, source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Silence".to_string() });
            }
        }
        Effect::Fear { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let source_pos = state.units[caster_idx].position;
                state.units[tidx].casting = None;
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Fear { source_pos }, source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Fear".to_string() });
            }
        }
        Effect::Taunt { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Taunt { taunter_id: caster_id }, source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Taunt".to_string() });
            }
        }
        Effect::Pull { distance: dist } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let caster_pos = state.units[caster_idx].position;
                let target_pos = state.units[tidx].position;
                let new_pos = move_towards(target_pos, caster_pos, *dist);
                state.units[tidx].position = new_pos;
                events.push(SimEvent::PullApplied { tick, source_id: caster_id, target_id });
            }
        }
        Effect::Swap => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let caster_pos = state.units[caster_idx].position;
                let target_pos = state.units[tidx].position;
                state.units[caster_idx].position = target_pos;
                state.units[tidx].position = caster_pos;
                events.push(SimEvent::SwapPerformed { tick, unit_a: caster_id, unit_b: target_id });
            }
        }
        _ => return false,
    }
    true
}

/// Build a summon unit with template-specific stats.
fn build_summon_template(id: u32, team: Team, template: &str) -> UnitState {
    let base = || UnitState {
        id, team, hp: 50, max_hp: 50,
        position: sim_vec2(0.0, 0.0),
        move_speed_per_sec: 2.0, attack_damage: 8, attack_range: 1.5,
        attack_cooldown_ms: 1000, attack_cast_time_ms: 300, cooldown_remaining_ms: 0,
        ability_damage: 0, ability_range: 0.0, ability_cooldown_ms: 0,
        ability_cast_time_ms: 0, ability_cooldown_remaining_ms: 0,
        heal_amount: 0, heal_range: 0.0, heal_cooldown_ms: 0,
        heal_cast_time_ms: 0, heal_cooldown_remaining_ms: 0,
        control_range: 0.0, control_duration_ms: 0, control_cooldown_ms: 0,
        control_cast_time_ms: 0, control_cooldown_remaining_ms: 0,
        control_remaining_ms: 0, casting: None,
        abilities: Vec::new(), passives: Vec::new(),
        status_effects: Vec::new(), shield_hp: 0,
        resistance_tags: HashMap::new(),
        state_history: std::collections::VecDeque::new(),
        channeling: None, resource: 0, max_resource: 0, resource_regen_per_sec: 0.0,
        owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
    };

    match template {
        // Engineer turret: stationary, high ranged DPS
        "turret" => {
            let mut u = base();
            u.hp = 40; u.max_hp = 40;
            u.move_speed_per_sec = 0.0;
            u.attack_damage = 18;
            u.attack_range = 5.0;
            u.attack_cooldown_ms = 800;
            u.attack_cast_time_ms = 200;
            u
        }
        // Engineer repair bot: stationary, heals allies
        "repair_bot" => {
            let mut u = base();
            u.hp = 35; u.max_hp = 35;
            u.move_speed_per_sec = 0.0;
            u.attack_damage = 0;
            u.attack_range = 0.0;
            u.heal_amount = 12;
            u.heal_range = 4.0;
            u.heal_cooldown_ms = 2000;
            u.heal_cast_time_ms = 300;
            u
        }
        // Druid treant: tanky melee bruiser
        "treant" => {
            let mut u = base();
            u.hp = 120; u.max_hp = 120;
            u.move_speed_per_sec = 1.8;
            u.attack_damage = 14;
            u.attack_range = 1.5;
            u.attack_cooldown_ms = 1200;
            u
        }
        // Necromancer skeleton: fragile but fast melee (comes in pairs)
        "skeleton" => {
            let mut u = base();
            u.hp = 30; u.max_hp = 30;
            u.move_speed_per_sec = 3.0;
            u.attack_damage = 10;
            u.attack_range = 1.3;
            u.attack_cooldown_ms = 800;
            u.attack_cast_time_ms = 200;
            u
        }
        // Warlock imp: ranged glass cannon
        "imp" => {
            let mut u = base();
            u.hp = 25; u.max_hp = 25;
            u.move_speed_per_sec = 3.2;
            u.attack_damage = 14;
            u.attack_range = 4.0;
            u.attack_cooldown_ms = 900;
            u.attack_cast_time_ms = 200;
            u
        }
        // Witch doctor zombie: slow tank, high HP
        "zombie" => {
            let mut u = base();
            u.hp = 80; u.max_hp = 80;
            u.move_speed_per_sec = 1.5;
            u.attack_damage = 12;
            u.attack_range = 1.3;
            u.attack_cooldown_ms = 1400;
            u.attack_cast_time_ms = 400;
            u
        }
        // Shaman spirit wolf: fast flanker
        "spirit_wolf" => {
            let mut u = base();
            u.hp = 45; u.max_hp = 45;
            u.move_speed_per_sec = 4.0;
            u.attack_damage = 12;
            u.attack_range = 1.3;
            u.attack_cooldown_ms = 700;
            u.attack_cast_time_ms = 200;
            u
        }
        // Stationary beacon (generic)
        "beacon" => {
            let mut u = base();
            u.move_speed_per_sec = 0.0;
            u.attack_damage = 0;
            u
        }
        // Unknown template: use base stats
        _ => base(),
    }
}
