use crate::ai::effects::{
    AbilityTarget, ActiveStatusEffect, Effect, Stacking, StatusKind, Trigger,
};

use super::types::*;
use super::events::SimEvent;
use super::helpers::*;
use super::triggers::{check_passive_triggers, fire_damage_triggers};
use super::damage::apply_damage_to_unit;

/// Handle extended effects (CC, damage modifiers, complex mechanics).
pub fn apply_effect_extended(
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
) {
    match effect {
        Effect::Reflect { percent, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Reflect { percent: *percent },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Reflect".to_string() });
            }
        }
        Effect::Lifesteal { percent, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Lifesteal { percent: *percent },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Lifesteal".to_string() });
            }
        }
        Effect::DamageModify { factor, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::DamageModify { factor: *factor },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: format!("DamageModify:{}", factor) });
            }
        }
        Effect::SelfDamage { amount } => {
            apply_damage_to_unit(caster_idx, caster_id, *amount, tick, state, events);
        }
        Effect::Execute { hp_threshold_percent } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                if is_alive(&state.units[tidx]) {
                    let pct = (state.units[tidx].hp as f32 / state.units[tidx].max_hp.max(1) as f32) * 100.0;
                    if pct <= *hp_threshold_percent {
                        state.units[tidx].hp = 0;
                        events.push(SimEvent::ExecuteTriggered { tick, source_id: caster_id, target_id });
                        events.push(SimEvent::UnitDied { tick, unit_id: target_id });
                        fire_damage_triggers(caster_idx, tidx, target_id, 0, false, tick, state, events);
                    }
                }
            }
        }
        Effect::Blind { miss_chance, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Blind { miss_chance: *miss_chance },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Blind".to_string() });
            }
        }
        Effect::OnHitBuff { duration_ms, on_hit_effects } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::OnHitBuff { effects: on_hit_effects.clone() },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "OnHitBuff".to_string() });
            }
        }
        Effect::Resurrect { hp_percent } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                if state.units[tidx].hp <= 0 {
                    let max_hp = state.units[tidx].max_hp;
                    state.units[tidx].hp = (max_hp as f32 * hp_percent / 100.0).max(1.0) as i32;
                    state.units[tidx].status_effects.clear();
                    state.units[tidx].shield_hp = 0;
                    state.units[tidx].control_remaining_ms = 0;
                    state.units[tidx].casting = None;
                    events.push(SimEvent::UnitResurrected { tick, unit_id: target_id });
                }
            }
        }
        Effect::OverhealShield { duration_ms, conversion_percent } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::OverhealShield { conversion_percent: *conversion_percent },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "OverhealShield".to_string() });
            }
        }
        Effect::AbsorbToHeal { shield_amount, duration_ms, heal_percent } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].shield_hp += shield_amount;
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::AbsorbShield { amount: *shield_amount, heal_percent: *heal_percent },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::ShieldApplied { tick, unit_id: target_id, amount: *shield_amount });
            }
        }
        Effect::ShieldSteal { amount } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let stolen = (*amount).min(state.units[tidx].shield_hp);
                state.units[tidx].shield_hp -= stolen;
                state.units[caster_idx].shield_hp += stolen;
            }
        }
        Effect::StatusClone { max_count } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let caster_team = state.units[caster_idx].team;
                let target_team = state.units[tidx].team;
                if target_team == caster_team {
                    let buffs: Vec<ActiveStatusEffect> = state.units[caster_idx].status_effects.iter()
                        .filter(|s| matches!(s.kind, StatusKind::Buff { .. } | StatusKind::Shield { .. } | StatusKind::Hot { .. }))
                        .take(*max_count as usize).cloned().collect();
                    for b in buffs { state.units[tidx].status_effects.push(b); }
                } else {
                    let mut stolen = 0u32;
                    let mut to_move = Vec::new();
                    state.units[tidx].status_effects.retain(|s| {
                        if stolen >= *max_count { return true; }
                        if matches!(s.kind, StatusKind::Buff { .. } | StatusKind::Shield { .. } | StatusKind::Hot { .. }) {
                            to_move.push(s.clone()); stolen += 1; false
                        } else { true }
                    });
                    for s in to_move { state.units[caster_idx].status_effects.push(s); }
                }
            }
        }
        Effect::Immunity { immune_to, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Immunity { immune_to: immune_to.clone() },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Immunity".to_string() });
            }
        }
        Effect::Detonate { damage_multiplier } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let mut total_dot_remaining = 0i32;
                state.units[tidx].status_effects.retain(|s| {
                    if let StatusKind::Dot { amount_per_tick, tick_interval_ms, .. } = &s.kind {
                        let ticks_left = if *tick_interval_ms > 0 { s.remaining_ms / tick_interval_ms } else { 0 };
                        total_dot_remaining += *amount_per_tick * ticks_left as i32;
                        false
                    } else { true }
                });
                if total_dot_remaining > 0 {
                    let det_damage = (total_dot_remaining as f32 * damage_multiplier) as i32;
                    apply_damage_to_unit(caster_idx, target_id, det_damage, tick, state, events);
                }
            }
        }
        Effect::StatusTransfer { steal_buffs } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                if *steal_buffs {
                    let mut to_move = Vec::new();
                    state.units[tidx].status_effects.retain(|s| {
                        if matches!(s.kind, StatusKind::Buff { .. } | StatusKind::Shield { .. } | StatusKind::Hot { .. }) {
                            to_move.push(s.clone()); false
                        } else { true }
                    });
                    for s in to_move { state.units[caster_idx].status_effects.push(s); }
                } else {
                    let mut to_move = Vec::new();
                    state.units[caster_idx].status_effects.retain(|s| {
                        if matches!(s.kind, StatusKind::Debuff { .. } | StatusKind::Dot { .. } | StatusKind::Slow { .. }
                            | StatusKind::Stun | StatusKind::Root | StatusKind::Silence) {
                            to_move.push(s.clone()); false
                        } else { true }
                    });
                    for s in to_move { state.units[tidx].status_effects.push(s); }
                }
            }
        }
        Effect::DeathMark { duration_ms, damage_percent } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::DeathMark { accumulated_damage: 0, damage_percent: *damage_percent },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "DeathMark".to_string() });
            }
        }
        Effect::Polymorph { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].casting = None;
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Polymorph, source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Polymorph".to_string() });
            }
        }
        Effect::Banish { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].casting = None;
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Banish, source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Banish".to_string() });
            }
        }
        Effect::Confuse { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Confuse, source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Confuse".to_string() });
            }
        }
        Effect::Charm { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let original_team = state.units[tidx].team;
                state.units[tidx].team = state.units[caster_idx].team;
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Charm { original_team }, source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Charm".to_string() });
            }
        }
        Effect::Stealth { duration_ms, break_on_damage, break_on_ability } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Stealth { break_on_damage: *break_on_damage, break_on_ability: *break_on_ability },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Stealth".to_string() });
            }
        }
        Effect::Leash { max_range, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let anchor_pos = state.units[tidx].position;
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Leash { anchor_pos, max_range: *max_range },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Leash".to_string() });
            }
        }
        Effect::Link { duration_ms, share_percent } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[caster_idx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Link { partner_id: target_id, share_percent: *share_percent },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Link { partner_id: caster_id, share_percent: *share_percent },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Link".to_string() });
            }
        }
        Effect::Redirect { duration_ms, charges } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Redirect { protector_id: caster_id, charges: *charges },
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Redirect".to_string() });
            }
        }
        Effect::Rewind { lookback_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let current_tick = state.tick as u32;
                let target_tick = current_tick.saturating_sub(lookback_ms / FIXED_TICK_MS.max(1));
                if let Some(entry) = state.units[tidx].state_history.iter().rev().find(|(t, _, _)| *t <= target_tick) {
                    let (_, pos, hp) = *entry;
                    state.units[tidx].position = pos;
                    state.units[tidx].hp = hp.min(state.units[tidx].max_hp);
                    events.push(SimEvent::RewindApplied { tick, unit_id: target_id });
                }
            }
        }
        Effect::CooldownModify { amount_ms, ref ability_name } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                for slot in &mut state.units[tidx].abilities {
                    if let Some(ref name) = ability_name {
                        if slot.def.name != *name { continue; }
                    }
                    if *amount_ms < 0 {
                        slot.cooldown_remaining_ms = slot.cooldown_remaining_ms.saturating_sub((-*amount_ms) as u32);
                    } else {
                        slot.cooldown_remaining_ms += *amount_ms as u32;
                    }
                }
            }
        }
        Effect::ApplyStacks { ref name, count, max_stacks, duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let existing = state.units[tidx].status_effects.iter_mut().find(|s| {
                    matches!(&s.kind, StatusKind::Stacks { name: n, .. } if n == name)
                });
                let new_count = if let Some(se) = existing {
                    if let StatusKind::Stacks { count: ref mut c, max_stacks: ms, .. } = se.kind {
                        *c = (*c + count).min(ms);
                        se.remaining_ms = *duration_ms;
                        *c
                    } else { 0 }
                } else {
                    let initial = (*count).min(*max_stacks);
                    state.units[tidx].status_effects.push(ActiveStatusEffect {
                        kind: StatusKind::Stacks { name: name.clone(), count: initial, max_stacks: *max_stacks },
                        source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                    });
                    initial
                };
                events.push(SimEvent::StacksApplied { tick, unit_id: target_id, name: name.clone(), count: new_count });
                check_passive_triggers(
                    Trigger::OnStackReached { name: name.clone(), count: new_count },
                    caster_idx, target_id, tick, state, events,
                );
            }
        }
        // Obstacle effects are handled at zone creation time, not per-unit.
        Effect::Obstacle { .. } => {}

        // --- LoL Coverage: New Effects ---
        Effect::Suppress { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].control_remaining_ms =
                    state.units[tidx].control_remaining_ms.max(*duration_ms);
                state.units[tidx].casting = None;
                state.units[tidx].channeling = None;
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Suppress,
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::ControlApplied { tick, source_id: caster_id, target_id, duration_ms: *duration_ms });
            }
        }
        Effect::Grounded { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                state.units[tidx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Grounded,
                    source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
                });
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: target_id, effect_name: "Grounded".to_string() });
            }
        }
        Effect::ProjectileBlock { duration_ms } => {
            // Creates a zone-like effect that blocks projectiles. For now, apply as a status
            // on caster that causes projectile collision checks to fail in advance_projectiles.
            state.units[caster_idx].status_effects.push(ActiveStatusEffect {
                kind: StatusKind::Immunity { immune_to: vec!["projectile".to_string()] },
                source_id: caster_id, remaining_ms: *duration_ms, tags: tags.clone(), stacking,
            });
            events.push(SimEvent::StatusEffectApplied { tick, unit_id: caster_id, effect_name: "ProjectileBlock".to_string() });
        }
        Effect::Attach { duration_ms } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                // Caster becomes untargetable and moves with target
                let dur = if *duration_ms == 0 { 60000 } else { *duration_ms }; // 0 = indefinite
                state.units[caster_idx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Attached { host_id: target_id },
                    source_id: caster_id, remaining_ms: dur, tags: tags.clone(), stacking,
                });
                // Also banish caster so they can't be targeted
                state.units[caster_idx].status_effects.push(ActiveStatusEffect {
                    kind: StatusKind::Banish,
                    source_id: caster_id, remaining_ms: dur, tags: tags.clone(), stacking,
                });
                // Snap to host position
                state.units[caster_idx].position = state.units[tidx].position;
                events.push(SimEvent::StatusEffectApplied { tick, unit_id: caster_id, effect_name: "Attached".to_string() });
            }
        }

        Effect::EvolveAbility { ability_index } => {
            // Permanently replace the ability with its evolve_into variant
            if let Some(slot) = state.units[caster_idx].abilities.get_mut(*ability_index) {
                if let Some(evolved) = slot.def.evolve_into.take() {
                    slot.def = *evolved;
                    slot.cooldown_remaining_ms = 0;
                    slot.charges = slot.def.max_charges;
                    events.push(SimEvent::AbilityUsed {
                        tick, unit_id: caster_id,
                        ability_index: *ability_index,
                        ability_name: format!("Evolved: {}", slot.def.name),
                    });
                }
            }
        }

        Effect::CommandSummons { speed: _ } => {
            // Move all directed summons owned by caster toward the target position
            let target_pos = match ability_target {
                AbilityTarget::Position(p) => p,
                AbilityTarget::Unit(tid) => {
                    find_unit_idx(state, tid)
                        .map(|i| state.units[i].position)
                        .unwrap_or(state.units[caster_idx].position)
                }
                AbilityTarget::None => state.units[caster_idx].position,
            };
            // Collect summon indices to avoid borrow issues
            let summon_indices: Vec<usize> = state.units.iter()
                .enumerate()
                .filter(|(_, u)| u.owner_id == Some(caster_id) && u.directed && u.hp > 0)
                .map(|(i, _)| i)
                .collect();
            for idx in summon_indices {
                // Instant reposition (like Azir Q — soldiers dash to location)
                state.units[idx].position = target_pos;
            }
        }

        // --- Expressiveness: Percent-Based HP Effects ---
        Effect::PercentHpDamage { percent, damage_type, max_damage } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let max_hp = state.units[tidx].max_hp;
                let mut dmg = (max_hp as f32 * percent / 100.0) as i32;
                if *max_damage > 0 {
                    dmg = dmg.min(*max_damage);
                }
                super::damage::apply_typed_damage(caster_idx, target_id, dmg, *damage_type, tick, state, events);
            }
        }
        Effect::PercentMissingHpHeal { percent } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let missing_hp = state.units[tidx].max_hp - state.units[tidx].hp;
                let heal = (missing_hp as f32 * percent / 100.0).max(0.0) as i32;
                super::damage::apply_heal_to_unit(caster_idx, target_id, heal, tick, state, events);
            }
        }
        Effect::PercentMaxHpHeal { percent } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let max_hp = state.units[tidx].max_hp;
                let heal = (max_hp as f32 * percent / 100.0) as i32;
                super::damage::apply_heal_to_unit(caster_idx, target_id, heal, tick, state, events);
            }
        }

        Effect::DamagePerStack { base, per_stack, ref stack_name, damage_type, consume } => {
            if let Some(tidx) = find_unit_idx(state, target_id) {
                let stack_count = state.units[tidx].status_effects.iter()
                    .find_map(|s| match &s.kind {
                        StatusKind::Stacks { name, count, .. } if name == stack_name => Some(*count),
                        _ => None,
                    })
                    .unwrap_or(0);
                let dmg = *base + *per_stack * stack_count as i32;
                if dmg > 0 {
                    super::damage::apply_typed_damage(caster_idx, target_id, dmg, *damage_type, tick, state, events);
                }
                if *consume {
                    state.units[tidx].status_effects.retain(|s| {
                        !matches!(&s.kind, StatusKind::Stacks { name, .. } if name == stack_name)
                    });
                }
            }
        }

        // Effects already handled by apply_effect_primary
        _ => {}
    }
}
