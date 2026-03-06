use crate::ai::core::{SimVec2, Team, UnitState};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Template enum
// ---------------------------------------------------------------------------

/// Identifies a pre-defined enemy archetype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnemyTemplate {
    // Phase 1
    /// Medium HP, short range, aggressive DPS.
    Grunt,
    /// High HP, melee only, protective Tank.
    Brute,
    /// Low HP, high control range, defensive Healer/Controller.
    Mystic,
    // Phase 4
    /// Stationary ranged attacker — high damage, low HP, holds position.
    Sentinel,
    /// Charger — enters aggressive mode at <50% HP (via personality).
    Berserker,
    /// Spawns one extra Grunt unit on first tick (flag-based).
    Summoner,
}

// ---------------------------------------------------------------------------
// Cast-time helpers (mirrors build_default_sim pattern: 300 ms for attacks)
// ---------------------------------------------------------------------------

pub(crate) const DEFAULT_ATTACK_CAST_MS: u32 = 300;
pub(crate) const DEFAULT_HEAL_CAST_MS: u32 = 400;
pub(crate) const DEFAULT_CONTROL_CAST_MS: u32 = 350;

// ---------------------------------------------------------------------------
// Core builder
// ---------------------------------------------------------------------------

/// Constructs a fully initialised `UnitState` for the given template.
///
/// Every field in `UnitState` is explicitly set — no defaults are relied upon
/// so that future struct additions are caught at compile time.
pub fn build_enemy_unit(template: EnemyTemplate, id: u32, position: SimVec2) -> UnitState {
    match template {
        // ---------------------------------------------------------------
        // Phase 1
        // ---------------------------------------------------------------

        EnemyTemplate::Grunt => UnitState {
            id,
            team: Team::Enemy,
            hp: 80,
            max_hp: 80,
            position,
            move_speed_per_sec: 2.5,
            attack_damage: 20,
            attack_range: 1.5,
            attack_cooldown_ms: 900,
            attack_cast_time_ms: DEFAULT_ATTACK_CAST_MS,
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
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },

        EnemyTemplate::Brute => UnitState {
            id,
            team: Team::Enemy,
            hp: 160,
            max_hp: 160,
            position,
            move_speed_per_sec: 1.8,
            attack_damage: 30,
            attack_range: 1.2,
            attack_cooldown_ms: 1400,
            attack_cast_time_ms: DEFAULT_ATTACK_CAST_MS,
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
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },

        EnemyTemplate::Mystic => UnitState {
            id,
            team: Team::Enemy,
            hp: 60,
            max_hp: 60,
            position,
            move_speed_per_sec: 2.0,
            attack_damage: 10,
            attack_range: 1.0,
            attack_cooldown_ms: 1200,
            attack_cast_time_ms: DEFAULT_ATTACK_CAST_MS,
            cooldown_remaining_ms: 0,
            ability_damage: 0,
            ability_range: 0.0,
            ability_cooldown_ms: 0,
            ability_cast_time_ms: 0,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 25,
            heal_range: 3.5,
            heal_cooldown_ms: 2000,
            heal_cast_time_ms: DEFAULT_HEAL_CAST_MS,
            heal_cooldown_remaining_ms: 0,
            control_range: 3.5,
            control_duration_ms: 2000,
            control_cooldown_ms: 4000,
            control_cast_time_ms: DEFAULT_CONTROL_CAST_MS,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },

        // ---------------------------------------------------------------
        // Phase 4
        // ---------------------------------------------------------------

        EnemyTemplate::Sentinel => UnitState {
            id,
            team: Team::Enemy,
            hp: 55,
            max_hp: 55,
            position,
            move_speed_per_sec: 0.8,
            attack_damage: 35,
            attack_range: 6.0,
            attack_cooldown_ms: 1100,
            attack_cast_time_ms: DEFAULT_ATTACK_CAST_MS,
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
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },

        EnemyTemplate::Berserker => UnitState {
            id,
            team: Team::Enemy,
            hp: 90,
            max_hp: 90,
            position,
            move_speed_per_sec: 3.5,
            attack_damage: 25,
            attack_range: 1.3,
            attack_cooldown_ms: 800,
            attack_cast_time_ms: DEFAULT_ATTACK_CAST_MS,
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
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },

        EnemyTemplate::Summoner => UnitState {
            id,
            team: Team::Enemy,
            hp: 50,
            max_hp: 50,
            position,
            move_speed_per_sec: 1.5,
            attack_damage: 8,
            attack_range: 1.0,
            attack_cooldown_ms: 1500,
            attack_cast_time_ms: DEFAULT_ATTACK_CAST_MS,
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
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        },
    }
}
