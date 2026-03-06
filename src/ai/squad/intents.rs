use std::cmp::Ordering;

use crate::ai::core::{
    distance, move_away, move_towards, position_at_range, IntentAction, SimState, SimVec2, Team,
    UnitIntent,
};
use crate::ai::pathing::GridNav;

use super::combat::{
    choose_action, choose_target, healer_backline_position, healer_intent,
};
use super::forces::{compute_raw_forces, dominant_force, weighted_forces, DominantForce};
use super::state::{
    personality_movement_profile, FormationMode, SquadAiState, TickContext,
};

// ---------------------------------------------------------------------------
// Core intent generation -- force-based steering
// ---------------------------------------------------------------------------

pub fn generate_intents(state: &SimState, ai: &mut SquadAiState, dt_ms: u32) -> Vec<UnitIntent> {
    generate_intents_with_terrain(state, ai, dt_ms, None)
}

pub fn generate_intents_with_terrain(state: &SimState, ai: &mut SquadAiState, dt_ms: u32, nav: Option<&GridNav>) -> Vec<UnitIntent> {
    ai.evaluate_blackboards_if_needed(state);

    // Pre-compute enemy centroid for terrain bias
    let enemy_centroid = compute_enemy_centroid(state, Team::Enemy);

    let ctx = TickContext::new(state);
    let mut intents = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(state.units.len());
    ids.extend(ctx.allies(Team::Hero));
    ids.extend(ctx.allies(Team::Enemy));
    ids.sort_unstable();

    for unit_id in ids {
        let Some(unit) = ctx.unit(state, unit_id) else {
            continue;
        };
        let personality = ai.personality_for(unit_id);
        let profile = personality_movement_profile(&personality, unit);
        let board = ai.blackboard(unit.team);

        let (anchor, sticky_target, lock_ticks) = {
            let mem = ai.memory.get_mut(&unit_id);
            if let Some(mem) = mem {
                if board.mode == FormationMode::Advance {
                    mem.anchor_position = unit.position;
                }
                (mem.anchor_position, mem.sticky_target, mem.lock_ticks)
            } else {
                (SimVec2::default(), None, 0)
            }
        };

        let leash = match board.mode {
            FormationMode::Advance => profile.leash_distance + 3.0,
            FormationMode::Retreat => profile.leash_distance - 2.0,
            FormationMode::Hold => profile.leash_distance,
        };
        let anchor_dist = distance(unit.position, anchor);
        if anchor_dist > leash {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::MoveTo { position: anchor },
            });
            continue;
        }

        // --- Ability evaluator interrupt (hero units only) ---
        // If trained weights are loaded, check if any ability should fire RIGHT NOW.
        // This runs before all force/personality logic — abilities are interrupts.
        if unit.team == Team::Hero {
            if let Some(ref weights) = ai.ability_eval_weights {
                if unit.casting.is_none() && unit.control_remaining_ms == 0 {
                    if let Some((action, _urgency)) =
                        crate::ai::core::ability_eval::evaluate_abilities_with_encoder(
                            state, ai, unit_id, weights, ai.ability_encoder.as_ref())
                    {
                        intents.push(UnitIntent { unit_id, action });
                        continue;
                    }
                }
            }
        }

        let effective_mode = if board.mode == FormationMode::Retreat
            && anchor_dist > leash * 0.75
        {
            FormationMode::Hold
        } else {
            board.mode
        };

        // Compute forces and steer
        let raw = compute_raw_forces(state, unit, &board, &ctx, &personality);
        let weighted = weighted_forces(&raw, &personality);
        let force = dominant_force(&weighted);

        match force {
            DominantForce::Heal => {
                if let Some(heal_intent) = healer_intent(state, unit_id, effective_mode, dt_ms) {
                    intents.push(heal_intent);
                    continue;
                }
                // Dedicated healer backline behavior
                let heal_ability_count = unit.abilities.iter().filter(|s| s.def.ai_hint == "heal").count();
                if heal_ability_count >= 2 {
                    let hero_heal_range = unit
                        .abilities
                        .iter()
                        .filter(|s| s.def.ai_hint == "heal")
                        .map(|s| s.def.range)
                        .fold(0.0f32, f32::max);
                    let effective_heal_range = if hero_heal_range > 0.0 { hero_heal_range } else { 4.0 };

                    let backline = healer_backline_position(state, unit, effective_heal_range);
                    let dist_to_backline = distance(unit.position, backline);

                    if dist_to_backline > 1.0 {
                        let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                        let next_pos = move_towards(unit.position, backline, max_step);
                        let biased = bias_hero_move(nav, unit, next_pos, max_step, enemy_centroid);
                        intents.push(UnitIntent {
                            unit_id,
                            action: IntentAction::MoveTo { position: biased },
                        });
                        continue;
                    }

                    let enemies = ctx.enemies_of(unit.team);
                    if let Some(&enemy_id) = enemies.first() {
                        let action = choose_action(state, unit_id, enemy_id, &personality, effective_mode, dt_ms, &ctx);
                        intents.push(UnitIntent { unit_id, action });
                    } else {
                        intents.push(UnitIntent { unit_id, action: IntentAction::Hold });
                    }
                    continue;
                }
                // Fall through to standard combat if can't heal
            }
            DominantForce::Protect => {
                let allies = ctx.allies(unit.team);
                let enemies = ctx.enemies_of(unit.team);
                let mut best_target: Option<(u32, f32)> = None;
                for &aid in allies {
                    if aid == unit_id { continue; }
                    let Some(ally) = ctx.unit(state, aid) else { continue };
                    let ally_hp_pct = ally.hp as f32 / ally.max_hp.max(1) as f32;
                    if ally_hp_pct > 0.6 { continue; }
                    for &eid in enemies {
                        let Some(enemy) = ctx.unit(state, eid) else { continue };
                        let threat_dist = distance(ally.position, enemy.position);
                        if threat_dist < 4.0 {
                            let score = (1.0 - ally_hp_pct) * 10.0 - threat_dist;
                            if best_target.map_or(true, |(_, bs)| score > bs) {
                                best_target = Some((eid, score));
                            }
                        }
                    }
                }
                if let Some((target_id, _)) = best_target {
                    let action = choose_action(state, unit_id, target_id, &personality, effective_mode, dt_ms, &ctx);
                    intents.push(UnitIntent { unit_id, action });
                    continue;
                }
            }
            DominantForce::Retreat => {
                let enemies = ctx.enemies_of(unit.team);
                if let Some(nearest) = enemies.iter()
                    .filter_map(|&eid| ctx.unit(state, eid))
                    .min_by(|a, b| distance(unit.position, a.position)
                        .partial_cmp(&distance(unit.position, b.position))
                        .unwrap_or(Ordering::Equal))
                {
                    let base_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                    let away = move_away(unit.position, nearest.position, 3.0);
                    let next_pos = move_towards(unit.position, away, base_step * 0.6);
                    let biased = bias_hero_move(nav, unit, next_pos, base_step, enemy_centroid);
                    intents.push(UnitIntent {
                        unit_id,
                        action: IntentAction::MoveTo { position: biased },
                    });
                    continue;
                }
            }
            DominantForce::Regroup => {
                let allies = ctx.allies(unit.team);
                if allies.len() > 1 {
                    let mut cx = 0.0f32;
                    let mut cy = 0.0f32;
                    let mut count = 0u32;
                    for &aid in allies {
                        if let Some(a) = ctx.unit(state, aid) {
                            cx += a.position.x;
                            cy += a.position.y;
                            count += 1;
                        }
                    }
                    let ally_cx = if count > 0 { cx / count as f32 } else { unit.position.x };
                    let ally_cy = if count > 0 { cy / count as f32 } else { unit.position.y };
                    let centroid = SimVec2 { x: ally_cx, y: ally_cy };
                    let base_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                    let next_pos = move_towards(unit.position, centroid, base_step);
                    let biased = bias_hero_move(nav, unit, next_pos, base_step, enemy_centroid);
                    intents.push(UnitIntent {
                        unit_id,
                        action: IntentAction::MoveTo { position: biased },
                    });
                    continue;
                }
            }
            DominantForce::Position => {
                let enemies = ctx.enemies_of(unit.team);
                if let Some(nearest) = enemies.iter()
                    .filter_map(|&eid| ctx.unit(state, eid))
                    .min_by(|a, b| distance(unit.position, a.position)
                        .partial_cmp(&distance(unit.position, b.position))
                        .unwrap_or(Ordering::Equal))
                {
                    let range_center = (profile.preferred_range_min + profile.preferred_range_max) * 0.5;
                    let desired = position_at_range(unit.position, nearest.position, range_center);
                    let base_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                    let next_pos = move_towards(unit.position, desired, base_step);
                    let biased = bias_hero_move(nav, unit, next_pos, base_step, enemy_centroid);
                    intents.push(UnitIntent {
                        unit_id,
                        action: IntentAction::MoveTo { position: biased },
                    });
                    continue;
                }
            }
            DominantForce::Pursue => {
                let enemies = ctx.enemies_of(unit.team);
                if let Some(target) = enemies.iter()
                    .filter_map(|&eid| ctx.unit(state, eid))
                    .min_by(|a, b| {
                        let a_pct = a.hp as f32 / a.max_hp.max(1) as f32;
                        let b_pct = b.hp as f32 / b.max_hp.max(1) as f32;
                        a_pct.partial_cmp(&b_pct).unwrap_or(Ordering::Equal)
                    })
                {
                    let action = choose_action(state, unit_id, target.id, &personality, effective_mode, dt_ms, &ctx);
                    if let Some(mem) = ai.memory.get_mut(&unit_id) {
                        if sticky_target == Some(target.id) {
                            mem.lock_ticks = lock_ticks.saturating_sub(1);
                        } else {
                            mem.sticky_target = Some(target.id);
                            mem.lock_ticks = 4;
                        }
                    }
                    intents.push(UnitIntent { unit_id, action });
                    continue;
                }
            }
            DominantForce::Control => {
                let enemies = ctx.enemies_of(unit.team);
                if let Some(target) = enemies.iter()
                    .filter_map(|&eid| ctx.unit(state, eid))
                    .filter(|e| e.control_remaining_ms == 0)
                    .min_by(|a, b| {
                        let a_pct = a.hp as f32 / a.max_hp.max(1) as f32;
                        let b_pct = b.hp as f32 / b.max_hp.max(1) as f32;
                        a_pct.partial_cmp(&b_pct).unwrap_or(Ordering::Equal)
                    })
                {
                    let action = choose_action(state, unit_id, target.id, &personality, effective_mode, dt_ms, &ctx);
                    if let Some(mem) = ai.memory.get_mut(&unit_id) {
                        if sticky_target == Some(target.id) {
                            mem.lock_ticks = lock_ticks.saturating_sub(1);
                        } else {
                            mem.sticky_target = Some(target.id);
                            mem.lock_ticks = 4;
                        }
                    }
                    intents.push(UnitIntent { unit_id, action });
                    continue;
                }
            }
            DominantForce::Focus | DominantForce::Attack => {
                // Standard target selection + combat -- handled below
            }
        }

        // Standard combat path (Attack, Focus, or fallthrough from other forces)
        let enemies = ctx.enemies_of(unit.team);
        if enemies.is_empty() {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::Hold,
            });
            continue;
        }

        let target = choose_target(
            state,
            unit_id,
            &personality,
            board,
            sticky_target,
            lock_ticks,
            enemies,
            &ctx,
        );
        let Some(target_id) = target else {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::Hold,
            });
            continue;
        };

        if let Some(mem) = ai.memory.get_mut(&unit_id) {
            if sticky_target == Some(target_id) {
                mem.lock_ticks = lock_ticks.saturating_sub(1);
            } else {
                mem.sticky_target = Some(target_id);
                mem.lock_ticks = 4;
            }
        }

        let action = choose_action(state, unit_id, target_id, &personality, effective_mode, dt_ms, &ctx);
        intents.push(UnitIntent { unit_id, action });
    }

    intents
}

fn compute_enemy_centroid(state: &SimState, enemy_team: Team) -> Option<SimVec2> {
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut count = 0u32;
    for u in &state.units {
        if u.hp > 0 && u.team == enemy_team {
            cx += u.position.x;
            cy += u.position.y;
            count += 1;
        }
    }
    if count > 0 {
        Some(SimVec2 { x: cx / count as f32, y: cy / count as f32 })
    } else {
        None
    }
}

/// Apply terrain bias to a hero's repositioning move (not attack approach).
fn bias_hero_move(nav: Option<&GridNav>, unit: &crate::ai::core::UnitState, next_pos: SimVec2, max_step: f32, enemy_centroid: Option<SimVec2>) -> SimVec2 {
    match nav {
        Some(n) if unit.team == Team::Hero => {
            crate::ai::pathing::terrain_biased_step(n, unit.position, next_pos, max_step, enemy_centroid)
        }
        _ => next_pos,
    }
}
