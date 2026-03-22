use serde::{Deserialize, Serialize};

use super::types::*;
use crate::sim::SimVec2;

// ---------------------------------------------------------------------------
// AbilityDef — data definition for an active ability
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbilityTargeting {
    TargetEnemy,
    TargetAlly,
    SelfCast,
    SelfAoe,
    GroundTarget,
    Direction,
    /// Click-drag vector targeting (start point + direction). Used by Rumble R, Viktor E.
    Vector,
    /// Hits all enemies on the map regardless of range. Used by Karthus R, Soraka R.
    Global,
}

impl Default for AbilityTargeting {
    fn default() -> Self {
        AbilityTargeting::TargetEnemy
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbilityDef {
    pub name: String,
    #[serde(default)]
    pub targeting: AbilityTargeting,
    #[serde(default)]
    pub range: f32,
    #[serde(default)]
    pub cooldown_ms: u32,
    #[serde(default)]
    pub cast_time_ms: u32,
    #[serde(default)]
    pub ai_hint: String,
    #[serde(default)]
    pub effects: Vec<ConditionalEffect>,
    #[serde(default)]
    pub delivery: Option<Delivery>,
    #[serde(default)]
    pub resource_cost: i32,
    #[serde(default)]
    pub morph_into: Option<Box<AbilityDef>>,
    #[serde(default)]
    pub morph_duration_ms: u32,
    /// Element tag for zone-reaction combos (e.g. "fire", "frost", "lightning").
    #[serde(default)]
    pub zone_tag: Option<String>,

    // --- LoL Coverage: New Fields ---
    /// Max charges (ammo system). 0 = normal single-use cooldown.
    #[serde(default)]
    pub max_charges: u32,
    /// Time in ms for one charge to regenerate.
    #[serde(default)]
    pub charge_recharge_ms: u32,
    /// If true, ability is a toggle (on/off, drains resource per second while active).
    #[serde(default)]
    pub is_toggle: bool,
    /// Resource cost per second while toggled on.
    #[serde(default)]
    pub toggle_cost_per_sec: f32,
    /// Number of times this ability can be recast before going on cooldown.
    #[serde(default)]
    pub recast_count: u32,
    /// Window in ms to recast before the ability goes on cooldown.
    #[serde(default)]
    pub recast_window_ms: u32,
    /// Effects for subsequent recasts (index 0 = 2nd cast, index 1 = 3rd cast, etc.).
    #[serde(default)]
    pub recast_effects: Vec<Vec<ConditionalEffect>>,
    /// If true, caster is immune to CC during this ability's cast time.
    #[serde(default)]
    pub unstoppable: bool,
    /// Form tag — when this ability is cast, all abilities with a matching
    /// `swap_form` tag swap to their `morph_into` variant.
    #[serde(default)]
    pub swap_form: Option<String>,
    /// Form tag on this ability — identifies which form group it belongs to.
    #[serde(default)]
    pub form: Option<String>,
    /// Evolution: permanently replace this ability's def when evolved.
    #[serde(default)]
    pub evolve_into: Option<Box<AbilityDef>>,
}

// ---------------------------------------------------------------------------
// PassiveDef — data definition for a passive ability
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassiveDef {
    pub name: String,
    pub trigger: Trigger,
    #[serde(default)]
    pub cooldown_ms: u32,
    #[serde(default)]
    pub effects: Vec<ConditionalEffect>,
    #[serde(default)]
    pub range: f32,
}

// ---------------------------------------------------------------------------
// Runtime slots — track cooldowns for abilities/passives on a unit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbilitySlot {
    pub def: AbilityDef,
    pub cooldown_remaining_ms: u32,
    #[serde(default)]
    pub base_def: Option<Box<AbilityDef>>,
    #[serde(default)]
    pub morph_remaining_ms: u32,
    /// Current charges available (for ammo-system abilities).
    #[serde(default)]
    pub charges: u32,
    /// Time until next charge regenerates.
    #[serde(default)]
    pub charge_recharge_remaining_ms: u32,
    /// Whether this toggle ability is currently active.
    #[serde(default)]
    pub toggled_on: bool,
    /// Remaining recasts before cooldown starts.
    #[serde(default)]
    pub recasts_remaining: u32,
    /// Window timer for recasting (counts down after first cast).
    #[serde(default)]
    pub recast_window_remaining_ms: u32,
}

impl AbilitySlot {
    pub fn new(def: AbilityDef) -> Self {
        let charges = def.max_charges;
        Self {
            def,
            cooldown_remaining_ms: 0,
            base_def: None,
            morph_remaining_ms: 0,
            charges,
            charge_recharge_remaining_ms: 0,
            toggled_on: false,
            recasts_remaining: 0,
            recast_window_remaining_ms: 0,
        }
    }

    pub fn is_ready(&self) -> bool {
        if self.def.max_charges > 0 {
            return self.charges > 0;
        }
        if self.def.is_toggle {
            return true; // toggles are always "ready" (on/off)
        }
        if self.recasts_remaining > 0 && self.recast_window_remaining_ms > 0 {
            return true; // mid-recast
        }
        self.cooldown_remaining_ms == 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassiveSlot {
    pub def: PassiveDef,
    pub cooldown_remaining_ms: u32,
    /// For periodic triggers: tracks time since last fire.
    pub periodic_elapsed_ms: u32,
}

impl PassiveSlot {
    pub fn new(def: PassiveDef) -> Self {
        Self {
            def,
            cooldown_remaining_ms: 0,
            periodic_elapsed_ms: 0,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.cooldown_remaining_ms == 0
    }
}

// ---------------------------------------------------------------------------
// ActiveStatusEffect — live status on a unit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveStatusEffect {
    pub kind: StatusKind,
    pub source_id: u32,
    pub remaining_ms: u32,
    pub tags: Tags,
    pub stacking: Stacking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatusKind {
    // --- Existing ---
    Stun,
    Slow { factor: f32 },
    Dot {
        amount_per_tick: i32,
        tick_interval_ms: u32,
        tick_elapsed_ms: u32,
    },
    Hot {
        amount_per_tick: i32,
        tick_interval_ms: u32,
        tick_elapsed_ms: u32,
    },
    Shield { amount: i32 },
    Buff { stat: String, factor: f32 },
    Debuff { stat: String, factor: f32 },
    Duel { partner_id: u32 },

    // --- Phase 2: CC ---
    Root,
    Silence,
    Fear { source_pos: SimVec2 },
    Taunt { taunter_id: u32 },

    // --- Phase 3: Damage Modifiers ---
    Reflect { percent: f32 },
    Lifesteal { percent: f32 },
    DamageModify { factor: f32 },
    Blind { miss_chance: f32 },
    OnHitBuff { effects: Vec<ConditionalEffect> },

    // --- Phase 4: Healing & Shield ---
    OverhealShield { conversion_percent: f32 },
    AbsorbShield { amount: i32, heal_percent: f32 },

    // --- Phase 5: Status Interaction ---
    Immunity { immune_to: Vec<String> },
    DeathMark { accumulated_damage: i32, damage_percent: f32 },

    // --- Phase 6: Control ---
    Polymorph,
    Banish,
    Confuse,
    Charm { original_team: crate::sim::Team },

    // --- Phase 7: Complex ---
    Stealth { break_on_damage: bool, break_on_ability: bool },
    Leash { anchor_pos: SimVec2, max_range: f32 },
    Link { partner_id: u32, share_percent: f32 },
    Redirect { protector_id: u32, charges: u32 },

    // --- Stacks ---
    Stacks { name: String, count: u32, max_stacks: u32 },

    // --- LoL Coverage ---
    Suppress,
    Grounded,
    Attached { host_id: u32 },
}

// ---------------------------------------------------------------------------
// Projectile — in-flight entity tracked by SimState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projectile {
    pub source_id: u32,
    pub target_id: u32,
    pub position: SimVec2,
    pub direction: SimVec2,
    pub speed: f32,
    pub pierce: bool,
    pub width: f32,
    pub on_hit: Vec<ConditionalEffect>,
    pub on_arrival: Vec<ConditionalEffect>,
    /// IDs of units already hit (for pierce tracking).
    pub already_hit: Vec<u32>,
    /// Target position at time of firing (for arrival detection).
    pub target_position: SimVec2,
    /// Max travel distance for skillshots. 0.0 = homing (existing behavior).
    #[serde(default)]
    pub max_travel_distance: f32,
    /// Distance traveled so far (for skillshot expiry).
    #[serde(default)]
    pub distance_traveled: f32,
}

// ---------------------------------------------------------------------------
// AbilityTarget — targeting union for UseAbility intent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AbilityTarget {
    Unit(u32),
    Position(SimVec2),
    None,
}

// ---------------------------------------------------------------------------
// HeroToml — top-level serde struct for hero TOML files
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroToml {
    pub hero: HeroMeta,
    #[serde(default)]
    pub stats: HeroStats,
    #[serde(default)]
    pub attack: Option<AttackStats>,
    #[serde(default)]
    pub abilities: Vec<AbilityDef>,
    #[serde(default)]
    pub passives: Vec<PassiveDef>,
    /// Optional path to a `.ability` DSL file (relative to hero_templates dir).
    /// When set, abilities and passives are loaded from this file instead of inline TOML.
    #[serde(default)]
    pub abilities_file: Option<String>,
    /// References to abilities defined in an AbilityManifest.
    /// These are resolved and appended to the abilities list.
    #[serde(default)]
    pub ability_refs: Vec<String>,
    /// References to passives defined in an AbilityManifest.
    #[serde(default)]
    pub passive_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroMeta {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroStats {
    #[serde(default = "default_hero_hp")]
    pub hp: i32,
    #[serde(default = "default_move_speed")]
    pub move_speed: f32,
    #[serde(default)]
    pub tags: Tags,
    #[serde(default)]
    pub resource: i32,
    #[serde(default)]
    pub max_resource: i32,
    #[serde(default)]
    pub resource_regen_per_sec: f32,
    #[serde(default)]
    pub armor: f32,
    #[serde(default)]
    pub magic_resist: f32,
}

impl Default for HeroStats {
    fn default() -> Self {
        Self {
            hp: 100,
            move_speed: 3.0,
            tags: Tags::new(),
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0,
            armor: 0.0,
            magic_resist: 0.0,
        }
    }
}

fn default_hero_hp() -> i32 {
    100
}
fn default_move_speed() -> f32 {
    3.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackStats {
    #[serde(default = "default_attack_damage")]
    pub damage: i32,
    #[serde(default = "default_attack_range")]
    pub range: f32,
    #[serde(default = "default_attack_cooldown")]
    pub cooldown: u32,
    #[serde(default = "default_cast_time")]
    pub cast_time: u32,
}

fn default_attack_damage() -> i32 {
    15
}
fn default_attack_range() -> f32 {
    1.5
}
fn default_attack_cooldown() -> u32 {
    1000
}
fn default_cast_time() -> u32 {
    300
}

impl Default for AttackStats {
    fn default() -> Self {
        Self {
            damage: 15,
            range: 1.5,
            cooldown: 1000,
            cast_time: 300,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

