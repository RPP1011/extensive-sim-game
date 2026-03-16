use crate::ai::core::{
    distance, is_alive,
    IntentAction, SimState, SimVec2,
};

use super::interpreter_actions::{resolve_target, find_unit, convert_action};
use super::types::*;

/// Evaluate a behavior tree against the current sim state for a given unit.
/// Returns the `IntentAction` from the first matching rule, or `Hold` if none match.
pub fn evaluate_behavior(
    tree: &BehaviorTree,
    state: &SimState,
    unit_id: u32,
    tick: u64,
) -> IntentAction {
    let unit_idx = match state.units.iter().position(|u| u.id == unit_id) {
        Some(i) => i,
        None => return IntentAction::Hold,
    };
    let unit = &state.units[unit_idx];
    if !is_alive(unit) {
        return IntentAction::Hold;
    }

    // Evaluate rules in declaration order. Priority ordering is expected to be
    // handled by the DSL author (priority rules listed first, then default, then fallback).
    for rule in &tree.rules {
        let cond_met = match &rule.condition {
            None => true,
            Some(cond) => eval_condition(cond, state, unit_idx, tick),
        };
        if cond_met {
            if let Some(action) = convert_action(&rule.action, state, unit_idx, tick) {
                return action;
            }
            // If action conversion failed (e.g. no valid target), continue to next rule
        }
    }

    IntentAction::Hold
}

// ---------------------------------------------------------------------------
// Condition evaluation
// ---------------------------------------------------------------------------

fn eval_condition(cond: &Condition, state: &SimState, unit_idx: usize, tick: u64) -> bool {
    match cond {
        Condition::And(a, b) => {
            eval_condition(a, state, unit_idx, tick)
                && eval_condition(b, state, unit_idx, tick)
        }
        Condition::Or(a, b) => {
            eval_condition(a, state, unit_idx, tick)
                || eval_condition(b, state, unit_idx, tick)
        }
        Condition::Not(inner) => !eval_condition(inner, state, unit_idx, tick),
        Condition::Compare(lhs, op, rhs) => {
            let l = eval_value(lhs, state, unit_idx, tick);
            let r = eval_value(rhs, state, unit_idx, tick);
            match op {
                CompOp::Lt => l < r,
                CompOp::Gt => l > r,
                CompOp::Lte => l <= r,
                CompOp::Gte => l >= r,
                CompOp::Eq => (l - r).abs() < f32::EPSILON,
                CompOp::Neq => (l - r).abs() >= f32::EPSILON,
            }
        }
        Condition::StateCheck(sc) => eval_state_check(sc, state, unit_idx, tick),
    }
}

fn eval_value(val: &Value, state: &SimState, unit_idx: usize, tick: u64) -> f32 {
    let unit = &state.units[unit_idx];
    match val {
        Value::Number(n) => *n,
        Value::SelfHp => unit.hp as f32,
        Value::SelfHpPct => unit.hp as f32 / unit.max_hp.max(1) as f32,
        Value::TargetHp(target) => {
            resolve_target(target, state, unit_idx)
                .and_then(|id| find_unit(state, id))
                .map_or(0.0, |u| u.hp as f32)
        }
        Value::TargetHpPct(target) => {
            resolve_target(target, state, unit_idx)
                .and_then(|id| find_unit(state, id))
                .map_or(0.0, |u| u.hp as f32 / u.max_hp.max(1) as f32)
        }
        Value::TargetDistance(target) => {
            resolve_target(target, state, unit_idx)
                .and_then(|id| find_unit(state, id))
                .map_or(f32::MAX, |u| distance(unit.position, u.position))
        }
        Value::TargetDps(target) => {
            resolve_target(target, state, unit_idx)
                .and_then(|id| find_unit(state, id))
                .map_or(0.0, |u| {
                    if u.attack_cooldown_ms > 0 {
                        u.attack_damage as f32 / (u.attack_cooldown_ms as f32 / 1000.0)
                    } else {
                        0.0
                    }
                })
        }
        Value::TargetCcRemaining(target) => {
            resolve_target(target, state, unit_idx)
                .and_then(|id| find_unit(state, id))
                .map_or(0.0, |u| u.control_remaining_ms as f32)
        }
        Value::TargetCastProgress(target) => {
            resolve_target(target, state, unit_idx)
                .and_then(|id| find_unit(state, id))
                .map_or(0.0, |u| {
                    u.casting.map_or(0.0, |c| c.remaining_ms as f32)
                })
        }
        Value::EnemyCount => {
            let team = unit.team;
            state.units.iter().filter(|u| is_alive(u) && u.team != team).count() as f32
        }
        Value::AllyCount => {
            let team = unit.team;
            state
                .units
                .iter()
                .filter(|u| is_alive(u) && u.team == team && u.id != unit.id)
                .count() as f32
        }
        Value::EnemyCountInRange(range) => {
            let team = unit.team;
            let pos = unit.position;
            state
                .units
                .iter()
                .filter(|u| {
                    is_alive(u) && u.team != team && distance(pos, u.position) <= *range
                })
                .count() as f32
        }
        Value::AllyCountBelowHp(threshold) => {
            let team = unit.team;
            state
                .units
                .iter()
                .filter(|u| {
                    is_alive(u)
                        && u.team == team
                        && u.id != unit.id
                        && (u.hp as f32 / u.max_hp.max(1) as f32) < *threshold
                })
                .count() as f32
        }
        Value::Tick => tick as f32,
        Value::AbilityCooldownPct(slot) => {
            unit.abilities.get(*slot).map_or(1.0, |a| {
                if a.def.cooldown_ms == 0 {
                    0.0
                } else {
                    a.cooldown_remaining_ms as f32 / a.def.cooldown_ms as f32
                }
            })
        }
        Value::BestAbilityUrgency => {
            // Stub — would need ability eval integration. Return 0 for now.
            0.0
        }
    }
}

fn eval_state_check(sc: &StateCheck, state: &SimState, unit_idx: usize, tick: u64) -> bool {
    let unit = &state.units[unit_idx];
    match sc {
        StateCheck::HealReady => unit.heal_cooldown_remaining_ms == 0 && unit.heal_amount > 0,
        StateCheck::CcReady => {
            unit.control_cooldown_remaining_ms == 0 && unit.control_duration_ms > 0
        }
        StateCheck::AoeReady => {
            unit.abilities.iter().any(|a| {
                a.cooldown_remaining_ms == 0
                    && matches!(
                        a.def.targeting,
                        crate::ai::effects::AbilityTargeting::SelfAoe
                            | crate::ai::effects::AbilityTargeting::GroundTarget
                    )
            })
        }
        StateCheck::AbilityReady(slot) => {
            unit.abilities
                .get(*slot)
                .map_or(false, |a| a.cooldown_remaining_ms == 0)
        }
        StateCheck::IsCasting => unit.casting.is_some(),
        StateCheck::IsCcd => unit.control_remaining_ms > 0,
        StateCheck::TargetIsCasting(target) => {
            resolve_target(target, state, unit_idx)
                .and_then(|id| find_unit(state, id))
                .map_or(false, |u| u.casting.is_some())
        }
        StateCheck::TargetIsCcd(target) => {
            resolve_target(target, state, unit_idx)
                .and_then(|id| find_unit(state, id))
                .map_or(false, |u| u.control_remaining_ms > 0)
        }
        StateCheck::CanAttack => {
            if unit.cooldown_remaining_ms > 0 {
                return false;
            }
            let team = unit.team;
            state.units.iter().any(|u| {
                is_alive(u)
                    && u.team != team
                    && distance(unit.position, u.position) <= unit.attack_range
            })
        }
        StateCheck::InDangerZone => {
            state.zones.iter().any(|z| {
                z.source_team != unit.team && zone_contains(z, unit.position)
            })
        }
        StateCheck::AllyInDanger => {
            let team = unit.team;
            state.units.iter().any(|u| {
                is_alive(u)
                    && u.team == team
                    && u.id != unit.id
                    && ((u.hp as f32 / u.max_hp.max(1) as f32) < 0.3
                        || state
                            .zones
                            .iter()
                            .any(|z| z.source_team != team && zone_contains(z, u.position)))
            })
        }
        StateCheck::NearWall(_dist) => {
            // Would need grid nav integration. Stub false for now.
            false
        }
        StateCheck::Every(n) => {
            if *n == 0 {
                true
            } else {
                tick % n == 0
            }
        }
    }
}

/// Simple zone containment check (circular approximation).
fn zone_contains(zone: &crate::ai::core::ActiveZone, pos: SimVec2) -> bool {
    let radius = match &zone.area {
        crate::ai::effects::Area::Circle { radius } => *radius,
        crate::ai::effects::Area::Line { length, .. } => *length,
        crate::ai::effects::Area::Cone { radius, .. } => *radius,
        crate::ai::effects::Area::SingleTarget
        | crate::ai::effects::Area::SelfOnly => 0.0,
        crate::ai::effects::Area::Ring { outer_radius, .. } => *outer_radius,
        crate::ai::effects::Area::Spread { radius, .. } => *radius,
    };
    distance(zone.position, pos) <= radius
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::{SimState, UnitState, Team, SimVec2, UnitStore};

    fn make_state(hero_pos: SimVec2, enemy_pos: SimVec2) -> SimState {
        SimState {
            tick: 10,
            rng_state: 42,
            units: UnitStore::new(vec![
                UnitState {
                    id: 1,
                    team: Team::Hero,
                    hp: 100,
                    max_hp: 100,
                    position: hero_pos,
                    move_speed_per_sec: 5.0,
                    attack_damage: 10,
                    attack_range: 1.5,
                    attack_cooldown_ms: 1000,
                    attack_cast_time_ms: 0,
                    cooldown_remaining_ms: 0,
                    ability_damage: 0,
                    ability_range: 0.0,
                    ability_cooldown_ms: 0,
                    ability_cast_time_ms: 0,
                    ability_cooldown_remaining_ms: 0,
                    heal_amount: 0,
                    heal_range: 0.0,
                    heal_cooldown_ms: 0,
                    heal_cast_time_ms: 0,
                    heal_cooldown_remaining_ms: 0,
                    control_range: 0.0,
                    control_duration_ms: 0,
                    control_cooldown_ms: 0,
                    control_cast_time_ms: 0,
                    control_cooldown_remaining_ms: 0,
                    control_remaining_ms: 0,
                    casting: None,
                    abilities: vec![],
                    passives: vec![],
                    status_effects: vec![],
                    shield_hp: 0,
                    resistance_tags: Default::default(),
                    state_history: Default::default(),
                    channeling: None,
                    resource: 0,
                    max_resource: 0,
                    resource_regen_per_sec: 0.0,
                    owner_id: None,
                    directed: false,
                    total_healing_done: 0,
                    total_damage_done: 0,
                    armor: 0.0,
                    magic_resist: 0.0,
                    cover_bonus: 0.0,
                    elevation: 0.0,
                },
                UnitState {
                    id: 2,
                    team: Team::Enemy,
                    hp: 80,
                    max_hp: 100,
                    position: enemy_pos,
                    move_speed_per_sec: 5.0,
                    attack_damage: 10,
                    attack_range: 1.5,
                    attack_cooldown_ms: 1000,
                    attack_cast_time_ms: 0,
                    cooldown_remaining_ms: 0,
                    ability_damage: 0,
                    ability_range: 0.0,
                    ability_cooldown_ms: 0,
                    ability_cast_time_ms: 0,
                    ability_cooldown_remaining_ms: 0,
                    heal_amount: 0,
                    heal_range: 0.0,
                    heal_cooldown_ms: 0,
                    heal_cast_time_ms: 0,
                    heal_cooldown_remaining_ms: 0,
                    control_range: 0.0,
                    control_duration_ms: 0,
                    control_cooldown_ms: 0,
                    control_cast_time_ms: 0,
                    control_cooldown_remaining_ms: 0,
                    control_remaining_ms: 0,
                    casting: None,
                    abilities: vec![],
                    passives: vec![],
                    status_effects: vec![],
                    shield_hp: 0,
                    resistance_tags: Default::default(),
                    state_history: Default::default(),
                    channeling: None,
                    resource: 0,
                    max_resource: 0,
                    resource_regen_per_sec: 0.0,
                    owner_id: None,
                    directed: false,
                    total_healing_done: 0,
                    total_damage_done: 0,
                    armor: 0.0,
                    magic_resist: 0.0,
                    cover_bonus: 0.0,
                    elevation: 0.0,
                },
            ]),
            projectiles: vec![],
            passive_trigger_depth: 0,
            zones: vec![],
            tethers: vec![],
            grid_nav: None,
        }
    }

    #[test]
    fn test_stationary_dummy() {
        let tree = crate::ai::behavior::parse_behavior(
            r#"behavior "dummy" { fallback hold }"#,
        )
        .unwrap();
        let state = make_state(SimVec2 { x: 0.0, y: 0.0 }, SimVec2 { x: 5.0, y: 0.0 });
        let action = evaluate_behavior(&tree, &state, 1, 10);
        assert!(matches!(action, IntentAction::Hold));
    }

    #[test]
    fn test_melee_chaser_attacks_in_range() {
        let tree = crate::ai::behavior::parse_behavior(
            r#"behavior "melee" {
                priority attack nearest_enemy when nearest_enemy.distance < 1.5
                default chase nearest_enemy
                fallback hold
            }"#,
        )
        .unwrap();
        let state = make_state(SimVec2 { x: 0.0, y: 0.0 }, SimVec2 { x: 1.0, y: 0.0 });
        let action = evaluate_behavior(&tree, &state, 1, 10);
        assert!(matches!(action, IntentAction::Attack { target_id: 2 }));
    }

    #[test]
    fn test_melee_chaser_chases_out_of_range() {
        let tree = crate::ai::behavior::parse_behavior(
            r#"behavior "melee" {
                priority attack nearest_enemy when nearest_enemy.distance < 1.5
                default chase nearest_enemy
                fallback hold
            }"#,
        )
        .unwrap();
        let state = make_state(SimVec2 { x: 0.0, y: 0.0 }, SimVec2 { x: 5.0, y: 0.0 });
        let action = evaluate_behavior(&tree, &state, 1, 10);
        assert!(matches!(action, IntentAction::MoveTo { .. }));
    }

    #[test]
    fn test_flee_moves_away() {
        let tree = crate::ai::behavior::parse_behavior(
            r#"behavior "flee" {
                default flee nearest_enemy
                fallback hold
            }"#,
        )
        .unwrap();
        let state = make_state(SimVec2 { x: 0.0, y: 0.0 }, SimVec2 { x: 5.0, y: 0.0 });
        let action = evaluate_behavior(&tree, &state, 1, 10);
        match action {
            IntentAction::MoveTo { position } => {
                assert!(position.x < 0.0, "expected flee to move away, got x={}", position.x);
            }
            other => panic!("expected MoveTo, got {:?}", other),
        }
    }

    #[test]
    fn test_maintain_distance_holds_at_range() {
        let tree = crate::ai::behavior::parse_behavior(
            r#"behavior "kite" {
                default maintain_distance nearest_enemy 5
                fallback hold
            }"#,
        )
        .unwrap();
        let state = make_state(SimVec2 { x: 0.0, y: 0.0 }, SimVec2 { x: 5.0, y: 0.0 });
        let action = evaluate_behavior(&tree, &state, 1, 10);
        assert!(matches!(action, IntentAction::Hold));
    }

    #[test]
    fn test_maintain_distance_moves_away_when_close() {
        let tree = crate::ai::behavior::parse_behavior(
            r#"behavior "kite" {
                default maintain_distance nearest_enemy 5
                fallback hold
            }"#,
        )
        .unwrap();
        let state = make_state(SimVec2 { x: 0.0, y: 0.0 }, SimVec2 { x: 2.0, y: 0.0 });
        let action = evaluate_behavior(&tree, &state, 1, 10);
        match action {
            IntentAction::MoveTo { position } => {
                assert!(position.x < 0.0, "expected move away, got x={}", position.x);
            }
            other => panic!("expected MoveTo, got {:?}", other),
        }
    }
}
