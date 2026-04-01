use std::collections::HashMap;
use std::path::Path;

use game::ai::core::{SimEvent, SimState, Team};
use game::ai::goap::dsl::{load_goap_file, GoapDef};
use game::ai::goap::GoapAiState;
use game::ai::personality::PersonalityProfile;
use game::ai::roles::Role;

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
fn formation_str(mode: game::ai::squad::FormationMode) -> &'static str {
    match mode {
        game::ai::squad::FormationMode::Hold => "Hold",
        game::ai::squad::FormationMode::Advance => "Advance",
        game::ai::squad::FormationMode::Retreat => "Retreat",
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

/// Map a hero template name to a .goap behavior file.
/// Falls back to role-based assignment if no template-specific behavior exists.
fn goap_file_for_template(template: &str, role: Role, behaviors_dir: &Path) -> &'static str {
    // Check for template-specific .goap file first
    let specific = behaviors_dir.join(format!("{}.goap", template));
    if specific.exists() {
        // Can't return a dynamic string as &'static str, so we use the
        // template-to-archetype mapping below instead
    }

    // Map known templates to archetypes (or template-specific .goap files)
    match template {
        // Template-specific behaviors
        "engineer" => "engineer.goap",
        // Melee / tank archetypes
        "warrior" | "knight" | "paladin" | "berserker" => "frontline.goap",
        // Ranged DPS
        "mage" | "arcanist" | "pyromancer" | "ranger" | "assassin" => "striker.goap",
        // Support / healer
        "cleric" | "druid" | "bard" | "shaman" => "medic.goap",
        // CC / controller
        "warlock" | "necromancer" => "controller.goap",
        // Utility / skirmisher
        "rogue" | "monk" => "skirmisher.goap",
        // Unknown template: fall back to role
        _ => match role {
            Role::Tank => "frontline.goap",
            Role::Healer => "medic.goap",
            Role::Dps => "striker.goap",
        },
    }
}

/// Build a GoapAiState by assigning .goap behaviors to hero units.
/// Uses hero template names when available, falling back to role-based assignment.
pub fn build_goap_for_heroes(
    state: &SimState,
    roles: &HashMap<u32, Role>,
    game_root: &Path,
    hero_templates: &[String],
) -> Option<GoapAiState> {
    let behaviors_dir = game_root.join("assets").join("behaviors");
    let mut defs: HashMap<u32, GoapDef> = HashMap::new();

    let heroes: Vec<&game::ai::core::UnitState> = state
        .units
        .iter()
        .filter(|u| u.team == Team::Hero && u.hp > 0)
        .collect();

    for (i, unit) in heroes.iter().enumerate() {
        let role = roles.get(&unit.id).copied().unwrap_or(Role::Dps);
        let goap_file = if i < hero_templates.len() {
            goap_file_for_template(&hero_templates[i], role, &behaviors_dir)
        } else {
            match role {
                Role::Tank => "frontline.goap",
                Role::Healer => "medic.goap",
                Role::Dps => "striker.goap",
            }
        };

        let path = behaviors_dir.join(goap_file);
        match load_goap_file(path.to_str().unwrap_or("")) {
            Ok(def) => {
                let template_name = if i < hero_templates.len() {
                    &hero_templates[i]
                } else {
                    "unknown"
                };
                eprintln!("[sim_bridge] GOAP: unit {} ({}) → {}", unit.id, template_name, goap_file);
                defs.insert(unit.id, def);
            }
            Err(e) => {
                eprintln!("[sim_bridge] GOAP load error for {}: {}", goap_file, e);
            }
        }
    }

    if defs.is_empty() {
        None
    } else {
        Some(GoapAiState::new(defs, None))
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
