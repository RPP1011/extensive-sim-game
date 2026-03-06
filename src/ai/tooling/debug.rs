use std::cmp::Ordering;
use std::collections::HashMap;

use crate::ai::core::{distance, move_away, position_at_range, IntentAction, SimState, UnitState};
use crate::ai::personality::{
    default_personalities, generate_scripted_intents, sample_phase5_party_state, PersonalityProfile,
    UnitMode,
};

use super::types::{ActionScoreDebug, TickDecisionDebug, UnitDecisionDebug};

pub fn build_phase5_debug(seed: u64, ticks: u32, top_k: usize) -> Vec<TickDecisionDebug> {
    let initial = sample_phase5_party_state(seed);
    let personalities = default_personalities();
    let roles = crate::ai::roles::default_roles();
    let (script, mode_history) = generate_scripted_intents(
        &initial,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
        roles,
        personalities.clone(),
    );

    let mut state = initial;
    let mut out = Vec::with_capacity(ticks as usize);

    for tick in 0..ticks as usize {
        let intents = script.get(tick).cloned().unwrap_or_default();
        let modes = mode_history.get(tick).cloned().unwrap_or_default();

        let mut mode_by_unit = HashMap::new();
        for (unit_id, mode) in modes {
            mode_by_unit.insert(unit_id, mode);
        }

        let mut decisions = Vec::new();
        for intent in &intents {
            let unit_id = intent.unit_id;
            let Some(unit) = state.units.iter().find(|u| u.id == unit_id && u.hp > 0) else {
                continue;
            };
            let mode = *mode_by_unit.get(&unit_id).unwrap_or(&UnitMode::Aggressive);
            let p = *personalities
                .get(&unit_id)
                .unwrap_or(&PersonalityProfile::vanguard());
            let mut candidates = score_candidates(&state, unit, mode, p);
            candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| format!("{:?}", b.action).cmp(&format!("{:?}", a.action)))
            });
            candidates.truncate(top_k);
            decisions.push(UnitDecisionDebug {
                unit_id,
                mode,
                chosen: intent.action,
                top_k: candidates,
            });
        }

        out.push(TickDecisionDebug {
            tick: state.tick,
            decisions,
        });

        let (new_state, _) = crate::ai::core::step(state, &intents, crate::ai::core::FIXED_TICK_MS);
        state = new_state;
    }

    out
}

fn score_candidates(
    state: &SimState,
    unit: &UnitState,
    mode: UnitMode,
    p: PersonalityProfile,
) -> Vec<ActionScoreDebug> {
    let mut out = Vec::new();

    out.push(ActionScoreDebug {
        action: IntentAction::Hold,
        score: -2.0 + p.patience * 2.0,
        reason: "idle_patience".to_string(),
    });

    let nearest_enemy = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != unit.team)
        .min_by(|a, b| {
            distance(unit.position, a.position)
                .partial_cmp(&distance(unit.position, b.position))
                .unwrap_or(Ordering::Equal)
        });

    if let Some(enemy) = nearest_enemy {
        let dist = distance(unit.position, enemy.position);
        let attack_score = 10.0 + p.aggression * 8.0 - dist * 2.5;
        out.push(ActionScoreDebug {
            action: IntentAction::Attack {
                target_id: enemy.id,
            },
            score: attack_score,
            reason: "attack_pressure".to_string(),
        });

        let ability_ready = unit.ability_cooldown_remaining_ms == 0 && unit.ability_damage > 0;
        let ability_score = 12.0 + p.control_bias * 7.0 - dist * 1.8;
        if ability_ready {
            out.push(ActionScoreDebug {
                action: IntentAction::CastAbility {
                    target_id: enemy.id,
                },
                score: ability_score,
                reason: "ability_ready".to_string(),
            });
        }

        let max_step = unit.move_speed_per_sec * 0.1;
        let move_pos = match mode {
            UnitMode::Defensive => move_away(unit.position, enemy.position, max_step),
            _ => position_at_range(unit.position, enemy.position, unit.attack_range * 0.9),
        };
        let mode_bias = match mode {
            UnitMode::Aggressive => 3.5,
            UnitMode::Defensive => 4.5,
            UnitMode::Protector => 2.0,
            UnitMode::Controller => 2.8,
        };
        out.push(ActionScoreDebug {
            action: IntentAction::MoveTo { position: move_pos },
            score: 6.0 + mode_bias,
            reason: "reposition_mode".to_string(),
        });
    }

    if unit.heal_amount > 0 {
        let weak_ally = state
            .units
            .iter()
            .filter(|u| u.hp > 0 && u.team == unit.team)
            .map(|ally| {
                (
                    ally,
                    ally.hp as f32 / ally.max_hp.max(1) as f32,
                    distance(unit.position, ally.position),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if let Some((ally, hp_pct, distance)) = weak_ally {
            let score = 14.0 + p.altruism * 10.0 - hp_pct * 8.0 - distance;
            out.push(ActionScoreDebug {
                action: IntentAction::CastHeal { target_id: ally.id },
                score,
                reason: "triage_low_hp".to_string(),
            });
        }
    }

    out
}
