use std::collections::HashMap;

use bevy_game::ai::core::{SimEvent, SimState, Team};
use bevy_game::ai::personality::PersonalityProfile;
use bevy_game::ai::roles::Role;

use super::types::*;

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

pub fn team_str(team: Team) -> &'static str {
    match team {
        Team::Hero => "Hero",
        Team::Enemy => "Enemy",
    }
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::Tank => "Tank",
        Role::Dps => "Dps",
        Role::Healer => "Healer",
    }
}

#[allow(dead_code)]
fn formation_str(mode: bevy_game::ai::squad::FormationMode) -> &'static str {
    match mode {
        bevy_game::ai::squad::FormationMode::Hold => "Hold",
        bevy_game::ai::squad::FormationMode::Advance => "Advance",
        bevy_game::ai::squad::FormationMode::Retreat => "Retreat",
    }
}

// ---------------------------------------------------------------------------
// Condensing
// ---------------------------------------------------------------------------

pub fn condense_event(event: &SimEvent) -> CondensedEvent {
    match *event {
        SimEvent::DamageApplied {
            tick,
            source_id,
            target_id,
            amount,
            ..
        } => CondensedEvent {
            kind: "damage".into(),
            tick,
            unit_id: Some(source_id),
            target_id: Some(target_id),
            amount: Some(amount),
        },
        SimEvent::HealApplied {
            tick,
            source_id,
            target_id,
            amount,
            ..
        } => CondensedEvent {
            kind: "heal".into(),
            tick,
            unit_id: Some(source_id),
            target_id: Some(target_id),
            amount: Some(amount),
        },
        SimEvent::UnitDied { tick, unit_id } => CondensedEvent {
            kind: "death".into(),
            tick,
            unit_id: Some(unit_id),
            target_id: None,
            amount: None,
        },
        SimEvent::ControlApplied {
            tick,
            source_id,
            target_id,
            duration_ms,
        } => CondensedEvent {
            kind: "control".into(),
            tick,
            unit_id: Some(source_id),
            target_id: Some(target_id),
            amount: Some(duration_ms as i32),
        },
        _ => CondensedEvent {
            kind: "other".into(),
            tick: 0,
            unit_id: None,
            target_id: None,
            amount: None,
        },
    }
}

pub fn condense_state(
    state: &SimState,
    roles: &HashMap<u32, Role>,
    personalities: &HashMap<u32, PersonalityProfile>,
    recent_events: &[SimEvent],
    room_width: f32,
    room_depth: f32,
) -> StateMessage {
    let units = state
        .units
        .iter()
        .filter(|u| u.hp > 0)
        .map(|u| {
            let role = roles.get(&u.id).copied().unwrap_or(Role::Dps);
            CondensedUnit {
                id: u.id,
                team: team_str(u.team).into(),
                role: role_str(role).into(),
                hp: u.hp,
                max_hp: u.max_hp,
                hp_pct: u.hp as f32 / u.max_hp.max(1) as f32,
                position: [u.position.x, u.position.y],
                attack_cd_remaining_ms: u.cooldown_remaining_ms,
                ability_cd_remaining_ms: u.ability_cooldown_remaining_ms,
                heal_cd_remaining_ms: u.heal_cooldown_remaining_ms,
                control_cd_remaining_ms: u.control_cooldown_remaining_ms,
                control_remaining_ms: u.control_remaining_ms,
                is_casting: u.casting.is_some(),
            }
        })
        .collect();

    let squads = HashMap::new(); // filled by caller if needed

    let personality = personalities
        .iter()
        .map(|(id, p)| {
            (
                id.to_string(),
                CondensedPersonality {
                    aggression: p.aggression,
                    risk_tolerance: p.risk_tolerance,
                    discipline: p.discipline,
                    control_bias: p.control_bias,
                    altruism: p.altruism,
                    patience: p.patience,
                },
            )
        })
        .collect();

    let events = recent_events
        .iter()
        .filter(|e| !matches!(e, SimEvent::Moved { .. } | SimEvent::UnitControlled { .. }))
        .map(condense_event)
        .filter(|e| e.kind != "other")
        .collect();

    StateMessage {
        r#type: "state".into(),
        tick: state.tick,
        units,
        squads,
        personality,
        recent_events: events,
        room_width,
        room_depth,
    }
}

/// Assign roles based on unit stats (simple heuristic).
pub fn assign_roles(state: &SimState) -> HashMap<u32, Role> {
    let mut roles = HashMap::new();
    for unit in &state.units {
        let role = if unit.heal_amount > 0 {
            Role::Healer
        } else if unit.max_hp >= 120 {
            Role::Tank
        } else {
            Role::Dps
        };
        roles.insert(unit.id, role);
    }
    roles
}
