use contracts::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::ai::effects::{
    AbilitySlot, AbilityTarget, ActiveStatusEffect, Area, ConditionalEffect,
    PassiveSlot, Projectile, Tags,
};
use super::events::SimEvent;

pub const FIXED_TICK_MS: u32 = 100;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Team {
    Hero,
    Enemy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitState {
    pub id: u32,
    pub team: Team,
    pub hp: i32,
    pub max_hp: i32,
    pub position: SimVec2,
    pub move_speed_per_sec: f32,
    pub attack_damage: i32,
    pub attack_range: f32,
    pub attack_cooldown_ms: u32,
    pub attack_cast_time_ms: u32,
    pub cooldown_remaining_ms: u32,
    pub ability_damage: i32,
    pub ability_range: f32,
    pub ability_cooldown_ms: u32,
    pub ability_cast_time_ms: u32,
    pub ability_cooldown_remaining_ms: u32,
    pub heal_amount: i32,
    pub heal_range: f32,
    pub heal_cooldown_ms: u32,
    pub heal_cast_time_ms: u32,
    pub heal_cooldown_remaining_ms: u32,
    pub control_range: f32,
    pub control_duration_ms: u32,
    pub control_cooldown_ms: u32,
    pub control_cast_time_ms: u32,
    pub control_cooldown_remaining_ms: u32,
    pub control_remaining_ms: u32,
    pub casting: Option<CastState>,
    // --- Hero ability engine fields ---
    #[serde(default)]
    pub abilities: Vec<AbilitySlot>,
    #[serde(default)]
    pub passives: Vec<PassiveSlot>,
    #[serde(default)]
    pub status_effects: Vec<ActiveStatusEffect>,
    #[serde(default)]
    pub shield_hp: i32,
    #[serde(default)]
    pub resistance_tags: Tags,
    /// Ring buffer of (tick, position, hp) for Rewind effect.
    #[serde(skip, default)]
    pub state_history: VecDeque<(u32, SimVec2, i32)>,
    #[serde(default)]
    pub channeling: Option<ChannelState>,
    #[serde(default)]
    pub resource: i32,
    #[serde(default)]
    pub max_resource: i32,
    #[serde(default)]
    pub resource_regen_per_sec: f32,
    /// Owner ID for summons. If set, this unit was summoned by the owner.
    /// Directed summons (owner_id + directed) don't act independently.
    #[serde(default)]
    pub owner_id: Option<u32>,
    /// If true, this summon is directed — attacks when owner attacks.
    #[serde(default)]
    pub directed: bool,
    /// Cumulative healing output (for AI targeting priority).
    #[serde(skip, default)]
    pub total_healing_done: i32,
    /// Cumulative damage output (for AI targeting priority).
    #[serde(skip, default)]
    pub total_damage_done: i32,
    /// Armor — reduces physical damage. Formula: reduction = armor / (100 + armor).
    #[serde(default)]
    pub armor: f32,
    /// Magic resist — reduces magic damage. Formula: reduction = mr / (100 + mr).
    #[serde(default)]
    pub magic_resist: f32,
    /// Damage reduction from nearby cover (0.0 = none, up to 0.5).
    /// Updated per-tick from terrain context.
    #[serde(skip, default)]
    pub cover_bonus: f32,
    /// Unit's current elevation (metres above baseline).
    /// Updated per-tick from terrain context.
    #[serde(skip, default)]
    pub elevation: f32,

}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CastState {
    pub target_id: u32,
    #[serde(default)]
    pub target_pos: Option<SimVec2>,
    pub remaining_ms: u32,
    pub kind: CastKind,
    /// Area of effect shape. Populated from AbilityDef at cast start for hero abilities.
    #[serde(default)]
    pub area: Option<crate::ai::effects::Area>,
    /// Ability index in the caster's ability list (for HeroAbility casts).
    #[serde(default)]
    pub ability_index: Option<usize>,
    /// Simplified effect category for AI spatial awareness.
    #[serde(default)]
    pub effect_hint: CastEffectHint,
}

/// What a cast will do when it completes. Derived from ability effects at cast start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CastEffectHint {
    #[default]
    Unknown,
    Damage,
    Heal,
    CrowdControl,
    Obstacle,
    Buff,
    Summon,
    Mixed,
}

impl CastEffectHint {
    /// Classify an ability's effects into a hint category.
    pub fn from_ability_def(def: &crate::ai::effects::AbilityDef) -> Self {
        let mut has_damage = false;
        let mut has_heal = false;
        let mut has_cc = false;
        let mut has_obstacle = false;
        let mut has_buff = false;
        let mut has_summon = false;

        for ce in &def.effects {
            match &ce.effect {
                crate::ai::effects::Effect::Damage { .. } => has_damage = true,
                crate::ai::effects::Effect::Heal { .. } => has_heal = true,
                crate::ai::effects::Effect::Stun { .. }
                | crate::ai::effects::Effect::Slow { .. }
                | crate::ai::effects::Effect::Root { .. }
                | crate::ai::effects::Effect::Knockback { .. }
                | crate::ai::effects::Effect::Pull { .. }
                | crate::ai::effects::Effect::Silence { .. }
                | crate::ai::effects::Effect::Polymorph { .. }
                | crate::ai::effects::Effect::Fear { .. }
                | crate::ai::effects::Effect::Grounded { .. }
                | crate::ai::effects::Effect::Suppress { .. } => has_cc = true,
                crate::ai::effects::Effect::Obstacle { .. } => has_obstacle = true,
                crate::ai::effects::Effect::Shield { .. }
                | crate::ai::effects::Effect::Buff { .. }
                | crate::ai::effects::Effect::Stealth { .. } => has_buff = true,
                crate::ai::effects::Effect::Summon { .. } => has_summon = true,
                _ => {}
            }
        }

        let categories = [has_damage, has_heal, has_cc, has_obstacle, has_buff, has_summon];
        let count = categories.iter().filter(|&&b| b).count();

        if count > 1 { return CastEffectHint::Mixed; }
        if has_obstacle { return CastEffectHint::Obstacle; }
        if has_damage { return CastEffectHint::Damage; }
        if has_heal { return CastEffectHint::Heal; }
        if has_cc { return CastEffectHint::CrowdControl; }
        if has_buff { return CastEffectHint::Buff; }
        if has_summon { return CastEffectHint::Summon; }
        CastEffectHint::Unknown
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CastKind {
    Attack,
    Ability,
    Heal,
    Control,
    HeroAbility(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelState {
    pub ability_index: usize,
    pub target_id: u32,
    pub target_pos: Option<SimVec2>,
    pub remaining_ms: u32,
    pub tick_interval_ms: u32,
    pub tick_elapsed_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveZone {
    pub id: u32,
    pub source_id: u32,
    pub source_team: Team,
    pub position: SimVec2,
    pub area: Area,
    pub effects: Vec<ConditionalEffect>,
    pub remaining_ms: u32,
    pub tick_interval_ms: u32,
    pub tick_elapsed_ms: u32,
    pub trigger_on_enter: bool,
    pub invisible: bool,
    pub triggered: bool,
    pub arm_time_ms: u32,
    #[serde(skip, default)]
    pub blocked_cells: Vec<(i32, i32)>,
    /// Optional element tag for zone-reaction combos (e.g. "fire", "frost", "lightning").
    #[serde(default)]
    pub zone_tag: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTether {
    pub source_id: u32,
    pub target_id: u32,
    pub remaining_ms: u32,
    pub max_range: f32,
    pub tick_effects: Vec<ConditionalEffect>,
    pub on_complete: Vec<ConditionalEffect>,
    pub tick_interval_ms: u32,
    pub tick_elapsed_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimState {
    pub tick: u64,
    pub rng_state: u64,
    pub units: Vec<UnitState>,
    #[serde(default)]
    pub projectiles: Vec<Projectile>,
    #[serde(skip, default)]
    pub passive_trigger_depth: u32,
    #[serde(default)]
    pub zones: Vec<ActiveZone>,
    #[serde(default)]
    pub tethers: Vec<ActiveTether>,
    #[serde(skip, default)]
    pub grid_nav: Option<crate::ai::pathing::GridNav>,
}

#[invariant(self.units.iter().all(|u| u.hp <= u.max_hp))]
#[invariant(self.units.iter().all(|u| u.position.x.is_finite()))]
impl SimState {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IntentAction {
    Attack { target_id: u32 },
    CastAbility { target_id: u32 },
    CastHeal { target_id: u32 },
    CastControl { target_id: u32 },
    UseAbility { ability_index: usize, target: AbilityTarget },
    MoveTo { position: SimVec2 },
    Hold,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UnitIntent {
    pub unit_id: u32,
    pub action: IntentAction,
}
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SimVec2 {
    pub x: f32,
    pub y: f32,
}

pub const fn sim_vec2(x: f32, y: f32) -> SimVec2 {
    SimVec2 { x, y }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimMetrics {
    pub ticks_elapsed: u32,
    pub seconds_elapsed: f32,
    pub winner: Option<Team>,
    pub tick_to_first_death: Option<u64>,
    pub final_hp_by_unit: Vec<(u32, i32)>,
    pub total_damage_by_unit: Vec<(u32, i32)>,
    pub damage_taken_by_unit: Vec<(u32, i32)>,
    pub dps_by_unit: Vec<(u32, f32)>,
    pub overkill_damage_total: i32,
    pub casts_started: u32,
    pub casts_completed: u32,
    pub casts_failed_out_of_range: u32,
    pub avg_cast_delay_ms: f32,
    pub heals_started: u32,
    pub heals_completed: u32,
    pub total_healing_by_unit: Vec<(u32, i32)>,
    pub attack_intents: u32,
    pub executed_attack_intents: u32,
    pub blocked_cooldown_intents: u32,
    pub blocked_invalid_target_intents: u32,
    pub dead_source_attack_intents: u32,
    pub reposition_for_range_events: u32,
    pub focus_fire_ticks: u32,
    pub max_targeters_on_single_target: u32,
    pub target_switches_by_unit: Vec<(u32, u32)>,
    pub movement_distance_x100_by_unit: Vec<(u32, i32)>,
    pub in_range_ticks_by_unit: Vec<(u32, u32)>,
    pub out_of_range_ticks_by_unit: Vec<(u32, u32)>,
    pub chase_ticks_by_unit: Vec<(u32, u32)>,
    pub invariant_violations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    pub final_state: SimState,
    pub events: Vec<SimEvent>,
    pub event_log_hash: u64,
    pub final_state_hash: u64,
    pub per_tick_state_hashes: Vec<u64>,
    pub metrics: SimMetrics,
}
