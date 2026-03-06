use crate::ai::effects::{AbilityTarget, StatusKind, Trigger};

use super::types::*;
use super::events::SimEvent;
use super::helpers::{is_alive, find_unit_idx, target_in_range_for_kind, next_rand_u32, move_towards_position};
use super::hero::resolve_hero_ability;
use super::damage::apply_damage_to_unit;
use super::triggers::check_passive_triggers;

pub fn try_start_cast(
    attacker_idx: usize,
    target_id: u32,
    kind: CastKind,
    tick: u64,
    dt_ms: u32,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    if state.units[attacker_idx].cooldown_remaining_ms > 0 {
        events.push(SimEvent::AttackBlockedCooldown {
            tick,
            unit_id: state.units[attacker_idx].id,
            target_id,
            cooldown_remaining_ms: state.units[attacker_idx].cooldown_remaining_ms,
        });
        return;
    }

    let Some(target_idx) = find_unit_idx(state, target_id) else {
        events.push(SimEvent::AttackBlockedInvalidTarget {
            tick, unit_id: state.units[attacker_idx].id, target_id,
        });
        return;
    };

    if !is_alive(&state.units[target_idx]) {
        events.push(SimEvent::AttackBlockedInvalidTarget {
            tick, unit_id: state.units[attacker_idx].id, target_id,
        });
        return;
    }

    if !target_in_range_for_kind(attacker_idx, target_idx, state, kind) {
        let target_pos = state.units[target_idx].position;
        move_towards_position(attacker_idx, target_pos, tick, state, dt_ms, events);
        events.push(SimEvent::AttackRepositioned {
            tick, unit_id: state.units[attacker_idx].id, target_id,
        });
        return;
    }

    let cast = CastState {
        target_id,
        target_pos: None,
        remaining_ms: state.units[attacker_idx].attack_cast_time_ms,
        kind,
    };
    state.units[attacker_idx].casting = Some(cast);
    events.push(SimEvent::CastStarted {
        tick, unit_id: state.units[attacker_idx].id, target_id,
    });
}

pub fn resolve_cast(
    attacker_idx: usize,
    target_id: u32,
    target_pos: Option<SimVec2>,
    kind: CastKind,
    tick: u64,
    state: &mut SimState,
    events: &mut Vec<SimEvent>,
) {
    if let CastKind::HeroAbility(ability_index) = kind {
        let target = if let Some(pos) = target_pos {
            AbilityTarget::Position(pos)
        } else if target_id == 0 {
            AbilityTarget::None
        } else {
            AbilityTarget::Unit(target_id)
        };
        resolve_hero_ability(attacker_idx, ability_index, target, tick, state, events);
        return;
    }

    let Some(target_idx) = find_unit_idx(state, target_id) else {
        return;
    };
    if !is_alive(&state.units[target_idx]) {
        return;
    }

    if !target_in_range_for_kind(attacker_idx, target_idx, state, kind) {
        match kind {
            CastKind::Heal => events.push(SimEvent::HealBlockedOutOfRange {
                tick, unit_id: state.units[attacker_idx].id, target_id,
            }),
            CastKind::Control => events.push(SimEvent::ControlBlockedOutOfRange {
                tick, unit_id: state.units[attacker_idx].id, target_id,
            }),
            CastKind::HeroAbility(_) | _ => events.push(SimEvent::CastFailedOutOfRange {
                tick, unit_id: state.units[attacker_idx].id, target_id,
            }),
        }
        return;
    }

    if kind == CastKind::Heal {
        let base_heal = state.units[attacker_idx].heal_amount;
        let variance_percent = 95 + (next_rand_u32(state) % 11) as i32;
        let heal_amount = ((base_heal * variance_percent) + 99) / 100;
        let current_hp = state.units[target_idx].hp;
        let max_hp = state.units[target_idx].max_hp;
        let new_hp = (current_hp + heal_amount).min(max_hp);
        let actual_healed = new_hp - current_hp;
        state.units[target_idx].hp = new_hp;
        state.units[attacker_idx].heal_cooldown_remaining_ms =
            state.units[attacker_idx].heal_cooldown_ms;
        events.push(SimEvent::HealApplied {
            tick, source_id: state.units[attacker_idx].id, target_id,
            amount: actual_healed, target_hp_before: current_hp, target_hp_after: new_hp,
        });
        if actual_healed > 0 {
            let pct = (new_hp as f32 / max_hp as f32) * 100.0;
            check_passive_triggers(Trigger::OnHpAbove { percent: pct }, target_idx, state.units[attacker_idx].id, tick, state, events);
        }
        return;
    }
    if kind == CastKind::Control {
        let duration_ms = state.units[attacker_idx].control_duration_ms;
        state.units[target_idx].control_remaining_ms = state.units[target_idx]
            .control_remaining_ms.max(duration_ms);
        state.units[target_idx].casting = None;
        state.units[attacker_idx].control_cooldown_remaining_ms =
            state.units[attacker_idx].control_cooldown_ms;
        events.push(SimEvent::ControlApplied {
            tick, source_id: state.units[attacker_idx].id, target_id, duration_ms,
        });
        return;
    }

    let base_damage = match kind {
        CastKind::Attack => state.units[attacker_idx].attack_damage,
        CastKind::Ability => state.units[attacker_idx].ability_damage,
        CastKind::Heal => 0,
        CastKind::Control => 0,
        CastKind::HeroAbility(_) => 0,
    };

    // Set cooldowns before applying damage (apply_damage_to_unit may trigger passives)
    match kind {
        CastKind::Attack => {
            let base_cd = state.units[attacker_idx].attack_cooldown_ms;
            let atk_speed_factor: f32 = state.units[attacker_idx]
                .status_effects.iter()
                .filter_map(|se| match &se.kind {
                    StatusKind::Buff { stat, factor } if stat == "attack_speed" => Some(*factor),
                    StatusKind::Debuff { stat, factor } if stat == "attack_speed" => Some(-*factor),
                    _ => None,
                })
                .sum();
            let modified_cd = (base_cd as f32 / (1.0 + atk_speed_factor).max(0.1)) as u32;
            state.units[attacker_idx].cooldown_remaining_ms = modified_cd;
        }
        CastKind::Ability => {
            state.units[attacker_idx].ability_cooldown_remaining_ms =
                state.units[attacker_idx].ability_cooldown_ms;
        }
        CastKind::Heal => {
            state.units[attacker_idx].heal_cooldown_remaining_ms =
                state.units[attacker_idx].heal_cooldown_ms;
        }
        CastKind::Control => {
            state.units[attacker_idx].control_cooldown_remaining_ms =
                state.units[attacker_idx].control_cooldown_ms;
        }
        CastKind::HeroAbility(_) => {
            // Cooldown is managed by resolve_hero_ability (supports charges, recasts, toggles)
        }
    }

    // Use centralized damage function (handles shields, reflect, lifesteal, etc.)
    apply_damage_to_unit(attacker_idx, target_id, base_damage, tick, state, events);

    // Directed summon proxy attacks: when owner attacks, each directed summon
    // in range also attacks the target from its own position.
    if kind == CastKind::Attack {
        let attacker_id = state.units[attacker_idx].id;
        let summon_attacks: Vec<(usize, i32)> = state.units.iter()
            .enumerate()
            .filter(|(_, u)| {
                u.hp > 0
                    && u.directed
                    && u.owner_id == Some(attacker_id)
                    && super::math::distance(u.position,
                        find_unit_idx(state, target_id)
                            .map(|ti| state.units[ti].position)
                            .unwrap_or_default()
                    ) <= u.attack_range
            })
            .map(|(i, u)| (i, u.attack_damage))
            .collect();
        for (summon_idx, dmg) in summon_attacks {
            if dmg > 0 {
                apply_damage_to_unit(summon_idx, target_id, dmg, tick, state, events);
            }
        }
    }
}
