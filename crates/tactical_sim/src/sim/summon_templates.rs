use std::collections::HashMap;

use super::types::*;
use super::types::sim_vec2;

/// Build a summon unit with template-specific stats.
pub(crate) fn build_summon_template(id: u32, team: Team, template: &str) -> UnitState {
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
