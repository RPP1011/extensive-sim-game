use std::cmp::Ordering;

use crate::sim::{
    distance, move_towards, position_at_range, IntentAction, SimState, SimVec2, UnitIntent,
    UnitState,
};
use crate::effects::{AbilityTarget, AbilityTargeting, Effect};

use crate::squad::state::{alive_by_team, FormationMode};

pub(in crate::squad) fn healer_intent(
    state: &SimState,
    healer_id: u32,
    mode: FormationMode,
    dt_ms: u32,
) -> Option<UnitIntent> {
    let healer = state.units.iter().find(|u| u.id == healer_id)?;
    let has_hero_heal = healer
        .abilities
        .iter()
        .any(|slot| slot.def.ai_hint == "heal");
    if healer.heal_amount <= 0 && !has_hero_heal {
        return None;
    }

    let hero_heal_range = healer
        .abilities
        .iter()
        .filter(|slot| slot.def.ai_hint == "heal")
        .map(|slot| slot.def.range)
        .fold(0.0f32, f32::max);
    let effective_heal_range = if hero_heal_range > 0.0 {
        hero_heal_range
    } else {
        healer.heal_range
    };

    let allies = alive_by_team(state, healer.team);
    let triage = allies
        .iter()
        .filter_map(|ally_id| {
            let ally = state.units.iter().find(|u| u.id == *ally_id)?;
            let missing = ally.max_hp - ally.hp;
            if missing <= 0 {
                return None;
            }
            let hp_pct = ally.hp as f32 / ally.max_hp.max(1) as f32;
            Some((*ally_id, missing, hp_pct))
        })
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    let healer_hp_pct = healer.hp as f32 / healer.max_hp.max(1) as f32;
    if has_hero_heal && healer_hp_pct < 0.7 {
        if let Some(action) = evaluate_hero_heal_ability(state, healer_id, healer_id) {
            return Some(UnitIntent {
                unit_id: healer_id,
                action,
            });
        }
    }

    let Some((ally_id, _missing, hp_pct)) = triage else {
        return None;
    };

    if !has_hero_heal {
        let threshold = match mode {
            FormationMode::Advance => 0.38,
            FormationMode::Hold => 0.48,
            FormationMode::Retreat => 0.62,
        };
        if hp_pct > threshold {
            return None;
        }
    }

    let ally = state.units.iter().find(|u| u.id == ally_id)?;
    let dist = distance(healer.position, ally.position);

    if let Some(action) = evaluate_hero_heal_ability(state, healer_id, ally_id) {
        return Some(UnitIntent {
            unit_id: healer_id,
            action,
        });
    }

    if healer.heal_cooldown_remaining_ms == 0 && dist <= healer.heal_range {
        return Some(UnitIntent {
            unit_id: healer_id,
            action: IntentAction::CastHeal { target_id: ally_id },
        });
    }

    if dist > effective_heal_range {
        let max_step = healer.move_speed_per_sec * (dt_ms as f32 / 1000.0);
        let desired_pos =
            position_at_range(healer.position, ally.position, effective_heal_range * 0.9);
        let next_pos = move_towards(healer.position, desired_pos, max_step);
        return Some(UnitIntent {
            unit_id: healer_id,
            action: IntentAction::MoveTo { position: next_pos },
        });
    }

    None
}

pub(in crate::squad) fn healer_backline_position(
    state: &SimState,
    healer: &UnitState,
    effective_heal_range: f32,
) -> SimVec2 {
    let allies: Vec<&UnitState> = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == healer.team && u.id != healer.id)
        .collect();
    let enemies: Vec<&UnitState> = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != healer.team)
        .collect();

    if allies.is_empty() || enemies.is_empty() {
        return healer.position;
    }

    let ally_cx = allies.iter().map(|u| u.position.x).sum::<f32>() / allies.len() as f32;
    let ally_cy = allies.iter().map(|u| u.position.y).sum::<f32>() / allies.len() as f32;

    let enemy_cx = enemies.iter().map(|u| u.position.x).sum::<f32>() / enemies.len() as f32;
    let enemy_cy = enemies.iter().map(|u| u.position.y).sum::<f32>() / enemies.len() as f32;

    let dx = ally_cx - enemy_cx;
    let dy = ally_cy - enemy_cy;
    let len = (dx * dx + dy * dy).sqrt().max(f32::EPSILON);

    let standoff = effective_heal_range * 0.7;
    SimVec2 {
        x: ally_cx + (dx / len) * standoff,
        y: ally_cy + (dy / len) * standoff,
    }
}

pub(in crate::squad) fn evaluate_hero_heal_ability(
    state: &SimState,
    unit_id: u32,
    ally_id: u32,
) -> Option<IntentAction> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    let ally = state.units.iter().find(|u| u.id == ally_id)?;
    let dist = distance(unit.position, ally.position);
    let healer_hp_pct = unit.hp as f32 / unit.max_hp.max(1) as f32;

    let damaged_allies = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == unit.team && u.hp < u.max_hp)
        .count();

    let has_dead_ally = state
        .units
        .iter()
        .any(|u| u.hp <= 0 && u.team == unit.team && u.id != unit.id);

    let mut best: Option<(usize, f32, AbilityTarget)> = None;

    for (i, slot) in unit.abilities.iter().enumerate() {
        if slot.cooldown_remaining_ms > 0 {
            continue;
        }
        if slot.def.resource_cost > 0 && unit.resource < slot.def.resource_cost {
            continue;
        }

        let has_heal = slot.def.effects.iter().any(|ce| matches!(ce.effect, Effect::Heal { .. }));
        let has_shield = slot.def.effects.iter().any(|ce| matches!(ce.effect, Effect::Shield { .. }));
        let has_resurrect = slot.def.effects.iter().any(|ce| matches!(ce.effect, Effect::Resurrect { .. }));

        if !has_heal && !has_shield && !has_resurrect {
            continue;
        }

        let targeting = &slot.def.targeting;
        let is_self_aoe = matches!(targeting, AbilityTargeting::SelfAoe | AbilityTargeting::SelfCast);
        let is_target_ally = matches!(targeting, AbilityTargeting::TargetAlly);

        if !is_self_aoe && !is_target_ally {
            continue;
        }

        if is_target_ally && slot.def.range > 0.0 && dist > slot.def.range {
            continue;
        }

        if has_resurrect {
            if !has_dead_ally {
                continue;
            }
            let dead_ally = state
                .units
                .iter()
                .find(|u| u.hp <= 0 && u.team == unit.team && u.id != unit.id);
            if let Some(dead) = dead_ally {
                let d = distance(unit.position, dead.position);
                if slot.def.range > 0.0 && d > slot.def.range {
                    continue;
                }
                let score = 12.0;
                if best.map_or(true, |(_, bs, _)| score > bs) {
                    best = Some((i, score, AbilityTarget::Unit(dead.id)));
                }
            }
            continue;
        }

        let mut score = 0.0f32;

        if has_heal {
            score += 5.0;
            let ally_hp_pct = ally.hp as f32 / ally.max_hp.max(1) as f32;
            score += (1.0 - ally_hp_pct) * 5.0;
        }

        if has_shield {
            score += 4.0;
            if is_self_aoe && healer_hp_pct < 0.5 {
                score += 4.0;
            }
        }

        if is_self_aoe && damaged_allies > 1 {
            score += damaged_allies as f32 * 1.5;
        }

        let ability_target = if is_target_ally {
            AbilityTarget::Unit(ally_id)
        } else {
            AbilityTarget::None
        };

        if score > 0.0 {
            if best.map_or(true, |(_, bs, _)| score > bs) {
                best = Some((i, score, ability_target));
            }
        }
    }

    let (ability_index, _score, target) = best?;
    Some(IntentAction::UseAbility {
        ability_index,
        target,
    })
}
