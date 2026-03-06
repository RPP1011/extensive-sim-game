pub(super) use std::collections::HashMap;
pub(super) use std::collections::VecDeque;
pub(super) use crate::ai::core::*;
pub(super) use crate::ai::core::helpers::check_tags_resisted;
pub(super) use crate::ai::effects::{
    AbilityDef, AbilityTarget, AbilityTargeting, ConditionalEffect, DamageType, Effect,
    AbilitySlot, PassiveSlot, ActiveStatusEffect, StatusKind, Stacking,
    PassiveDef, Trigger, Area, Delivery,
};

mod determinism;
mod mechanics;
mod abilities;

pub(super) fn hero_unit(id: u32, team: Team, pos: (f32, f32)) -> UnitState {
    UnitState {
        id, team, hp: 100, max_hp: 100,
        position: sim_vec2(pos.0, pos.1),
        move_speed_per_sec: 3.0, attack_damage: 10,
        attack_range: 1.4, attack_cooldown_ms: 700, attack_cast_time_ms: 300,
        cooldown_remaining_ms: 0, ability_damage: 0, ability_range: 0.0,
        ability_cooldown_ms: 0, ability_cast_time_ms: 0, ability_cooldown_remaining_ms: 0,
        heal_amount: 0, heal_range: 0.0, heal_cooldown_ms: 0, heal_cast_time_ms: 0,
        heal_cooldown_remaining_ms: 0, control_range: 0.0, control_duration_ms: 0,
        control_cooldown_ms: 0, control_cast_time_ms: 0, control_cooldown_remaining_ms: 0,
        control_remaining_ms: 0, casting: None, abilities: Vec::new(), passives: Vec::new(),
        status_effects: Vec::new(), shield_hp: 0, resistance_tags: HashMap::new(),
        state_history: VecDeque::new(), channeling: None, resource: 0, max_resource: 0,
        resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
    }
}

pub(super) fn make_state(units: Vec<UnitState>, seed: u64) -> SimState {
    SimState {
        tick: 0, rng_state: seed, units,
        projectiles: Vec::new(), passive_trigger_depth: 0,
        zones: Vec::new(), tethers: Vec::new(), grid_nav: None,
    }
}
