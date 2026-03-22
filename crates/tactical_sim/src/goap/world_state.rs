//! Fixed-size world state for GOAP planning. Extracted from SimState once per tick per unit.

use crate::sim::{distance, is_alive, SimState, UnitState};

// --- Property indices (compile-time usize constants) ---
pub const SELF_HP_PCT: usize = 0;
pub const SELF_IS_CASTING: usize = 1;
pub const SELF_IS_CCD: usize = 2;
pub const IN_ATTACK_RANGE: usize = 3;
pub const IN_ABILITY_RANGE: usize = 4;
pub const NEAREST_ENEMY_DISTANCE: usize = 5;
pub const NEAREST_ENEMY_HP_PCT: usize = 6;
pub const LOWEST_ALLY_HP_PCT: usize = 7;
pub const HAS_HEAL_TARGET: usize = 8;
pub const ABILITY_0_READY: usize = 9;
pub const ABILITY_1_READY: usize = 10;
pub const ABILITY_2_READY: usize = 11;
pub const ABILITY_3_READY: usize = 12;
pub const ABILITY_4_READY: usize = 13;
pub const ABILITY_5_READY: usize = 14;
pub const ABILITY_6_READY: usize = 15;
pub const ABILITY_7_READY: usize = 16;
pub const TARGET_ID: usize = 17;
pub const TARGET_IS_ALIVE: usize = 18;
pub const TARGET_DISTANCE: usize = 19;
pub const TARGET_HP_PCT: usize = 20;
pub const ENEMY_COUNT: usize = 21;
pub const ALLY_COUNT: usize = 22;
pub const TEAM_HP_ADVANTAGE: usize = 23;
pub const ENEMY_IS_CASTING: usize = 24;
pub const IN_DANGER_ZONE: usize = 25;
/// Team focus target ID set by squad blackboard. When set, target resolution
/// prefers this unit over nearest/lowest HP selection.
pub const TEAM_FOCUS_TARGET: usize = 26;
/// Distance from this unit to the team focus target (f32::MAX if none).
pub const FOCUS_TARGET_DISTANCE: usize = 27;
/// Whether this unit is currently targeting the team focus target.
pub const ON_FOCUS_TARGET: usize = 28;
pub const PROP_COUNT: usize = 32;

/// Property value — no heap allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropValue {
    Bool(bool),
    Float(f32),
    Id(Option<u32>),
}

impl PropValue {
    pub fn as_bool(self) -> bool {
        match self {
            PropValue::Bool(b) => b,
            PropValue::Float(f) => f > 0.0,
            PropValue::Id(id) => id.is_some(),
        }
    }

    pub fn as_float(self) -> f32 {
        match self {
            PropValue::Float(f) => f,
            PropValue::Bool(b) => if b { 1.0 } else { 0.0 },
            PropValue::Id(_) => 0.0,
        }
    }
}

impl Default for PropValue {
    fn default() -> Self {
        PropValue::Float(0.0)
    }
}

/// Fixed-size world state snapshot. No heap allocation.
#[derive(Debug, Clone)]
pub struct WorldState {
    pub props: [PropValue; PROP_COUNT],
}

impl Default for WorldState {
    fn default() -> Self {
        Self {
            props: [PropValue::default(); PROP_COUNT],
        }
    }
}

impl WorldState {
    pub fn get(&self, idx: usize) -> PropValue {
        self.props[idx]
    }

    pub fn set(&mut self, idx: usize, val: PropValue) {
        self.props[idx] = val;
    }

    /// Extract world state for a unit, with an optional team focus target
    /// from the squad blackboard.
    pub fn extract_with_focus(state: &SimState, unit_idx: usize, focus_target: Option<u32>) -> Self {
        let mut ws = Self::extract(state, unit_idx);
        let unit = &state.units[unit_idx];

        if let Some(focus_id) = focus_target {
            if let Some(target) = state.units.iter().find(|u| u.id == focus_id && is_alive(u)) {
                let dist = distance(unit.position, target.position);
                let hp_pct = if target.max_hp > 0 { target.hp as f32 / target.max_hp as f32 } else { 0.0 };

                ws.set(TEAM_FOCUS_TARGET, PropValue::Id(Some(focus_id)));
                ws.set(FOCUS_TARGET_DISTANCE, PropValue::Float(dist));

                // Override TARGET_ID to the focus target so planner actions
                // naturally target the team's priority
                ws.set(TARGET_ID, PropValue::Id(Some(focus_id)));
                ws.set(TARGET_IS_ALIVE, PropValue::Bool(true));
                ws.set(TARGET_DISTANCE, PropValue::Float(dist));
                ws.set(TARGET_HP_PCT, PropValue::Float(hp_pct));

                // Update range checks against focus target
                ws.set(IN_ATTACK_RANGE, PropValue::Bool(dist <= unit.attack_range));
                let ability_range = best_ability_range(unit);
                ws.set(IN_ABILITY_RANGE, PropValue::Bool(dist <= ability_range));
            }
        }

        ws
    }

    /// Extract world state for a unit from SimState.
    pub fn extract(state: &SimState, unit_idx: usize) -> Self {
        let mut ws = WorldState::default();
        let unit = &state.units[unit_idx];
        let team = unit.team;

        // Self vitals
        let hp_pct = if unit.max_hp > 0 {
            unit.hp as f32 / unit.max_hp as f32
        } else {
            0.0
        };
        ws.set(SELF_HP_PCT, PropValue::Float(hp_pct));
        ws.set(SELF_IS_CASTING, PropValue::Bool(unit.casting.is_some() || unit.channeling.is_some()));
        ws.set(SELF_IS_CCD, PropValue::Bool(unit.control_remaining_ms > 0));

        // Find nearest enemy
        let (nearest_enemy_dist, nearest_enemy_hp_pct, nearest_enemy_id) =
            find_nearest_enemy(state, unit);
        ws.set(NEAREST_ENEMY_DISTANCE, PropValue::Float(nearest_enemy_dist));
        ws.set(NEAREST_ENEMY_HP_PCT, PropValue::Float(nearest_enemy_hp_pct));

        // Range checks
        ws.set(IN_ATTACK_RANGE, PropValue::Bool(nearest_enemy_dist <= unit.attack_range));
        let ability_range = best_ability_range(unit);
        ws.set(IN_ABILITY_RANGE, PropValue::Bool(nearest_enemy_dist <= ability_range));

        // Ally info
        let (lowest_ally_hp_pct, has_heal_target) = find_ally_info(state, unit);
        ws.set(LOWEST_ALLY_HP_PCT, PropValue::Float(lowest_ally_hp_pct));
        ws.set(HAS_HEAL_TARGET, PropValue::Bool(has_heal_target));

        // Ability cooldowns
        for i in 0..8 {
            let ready = if i < unit.abilities.len() {
                let ab = &unit.abilities[i];
                if ab.def.max_charges > 0 {
                    ab.charges > 0
                } else {
                    ab.cooldown_remaining_ms == 0
                }
            } else {
                false
            };
            ws.set(ABILITY_0_READY + i, PropValue::Bool(ready));
        }

        // Target info (use nearest enemy as default target)
        ws.set(TARGET_ID, PropValue::Id(nearest_enemy_id));
        ws.set(TARGET_IS_ALIVE, PropValue::Bool(nearest_enemy_id.is_some()));
        ws.set(TARGET_DISTANCE, PropValue::Float(nearest_enemy_dist));
        ws.set(TARGET_HP_PCT, PropValue::Float(nearest_enemy_hp_pct));

        // Team state
        let (enemy_count, ally_count, team_hp_adv) = team_state(state, team);
        ws.set(ENEMY_COUNT, PropValue::Float(enemy_count as f32));
        ws.set(ALLY_COUNT, PropValue::Float(ally_count as f32));
        ws.set(TEAM_HP_ADVANTAGE, PropValue::Float(team_hp_adv));

        // Tactical awareness
        let enemy_casting = state.units.iter().any(|u| {
            is_alive(u) && u.team != team && u.casting.is_some()
        });
        ws.set(ENEMY_IS_CASTING, PropValue::Bool(enemy_casting));

        let in_danger = state.zones.iter().any(|z| {
            z.source_team != team && distance(unit.position, z.position) < 3.0
        });
        ws.set(IN_DANGER_ZONE, PropValue::Bool(in_danger));

        ws
    }
}

fn find_nearest_enemy(state: &SimState, unit: &UnitState) -> (f32, f32, Option<u32>) {
    let mut best_dist = f32::MAX;
    let mut best_hp_pct = 0.0;
    let mut best_id = None;
    for u in &state.units {
        if !is_alive(u) || u.team == unit.team {
            continue;
        }
        let d = distance(unit.position, u.position);
        if d < best_dist {
            best_dist = d;
            best_hp_pct = if u.max_hp > 0 { u.hp as f32 / u.max_hp as f32 } else { 0.0 };
            best_id = Some(u.id);
        }
    }
    if best_id.is_none() {
        best_dist = 0.0;
    }
    (best_dist, best_hp_pct, best_id)
}

fn find_ally_info(state: &SimState, unit: &UnitState) -> (f32, bool) {
    let mut lowest_pct = 1.0_f32;
    let mut has_heal_target = false;
    for u in &state.units {
        if !is_alive(u) || u.team != unit.team || u.id == unit.id {
            continue;
        }
        let pct = if u.max_hp > 0 { u.hp as f32 / u.max_hp as f32 } else { 1.0 };
        if pct < lowest_pct {
            lowest_pct = pct;
        }
        if pct < 0.7 {
            has_heal_target = true;
        }
    }
    (lowest_pct, has_heal_target)
}

fn best_ability_range(unit: &UnitState) -> f32 {
    let mut range = unit.attack_range;
    for ab in &unit.abilities {
        if ab.def.range > range {
            range = ab.def.range;
        }
    }
    if unit.ability_range > range {
        range = unit.ability_range;
    }
    range
}

fn team_state(state: &SimState, team: crate::sim::Team) -> (u32, u32, f32) {
    let mut enemy_count = 0u32;
    let mut ally_count = 0u32;
    let mut ally_hp_total = 0f32;
    let mut enemy_hp_total = 0f32;
    for u in &state.units {
        if !is_alive(u) {
            continue;
        }
        let pct = if u.max_hp > 0 { u.hp as f32 / u.max_hp as f32 } else { 0.0 };
        if u.team == team {
            ally_count += 1;
            ally_hp_total += pct;
        } else {
            enemy_count += 1;
            enemy_hp_total += pct;
        }
    }
    let adv = if enemy_count > 0 {
        (ally_hp_total / ally_count.max(1) as f32) - (enemy_hp_total / enemy_count as f32)
    } else {
        1.0
    };
    (enemy_count, ally_count, adv)
}

/// Map a property name string to its index. Used by the DSL parser.
pub fn prop_index(name: &str) -> Option<usize> {
    match name {
        "self_hp_pct" | "hp_pct" => Some(SELF_HP_PCT),
        "self_is_casting" | "is_casting" => Some(SELF_IS_CASTING),
        "self_is_ccd" | "is_ccd" => Some(SELF_IS_CCD),
        "in_attack_range" => Some(IN_ATTACK_RANGE),
        "in_ability_range" => Some(IN_ABILITY_RANGE),
        "nearest_enemy_distance" => Some(NEAREST_ENEMY_DISTANCE),
        "nearest_enemy_hp_pct" => Some(NEAREST_ENEMY_HP_PCT),
        "lowest_ally_hp_pct" => Some(LOWEST_ALLY_HP_PCT),
        "has_heal_target" => Some(HAS_HEAL_TARGET),
        "ability0_ready" => Some(ABILITY_0_READY),
        "ability1_ready" => Some(ABILITY_1_READY),
        "ability2_ready" => Some(ABILITY_2_READY),
        "ability3_ready" => Some(ABILITY_3_READY),
        "ability4_ready" => Some(ABILITY_4_READY),
        "ability5_ready" => Some(ABILITY_5_READY),
        "ability6_ready" => Some(ABILITY_6_READY),
        "ability7_ready" => Some(ABILITY_7_READY),
        "target_id" => Some(TARGET_ID),
        "target_is_alive" => Some(TARGET_IS_ALIVE),
        "target_distance" => Some(TARGET_DISTANCE),
        "target_hp_pct" => Some(TARGET_HP_PCT),
        "enemy_count" => Some(ENEMY_COUNT),
        "ally_count" => Some(ALLY_COUNT),
        "team_hp_advantage" => Some(TEAM_HP_ADVANTAGE),
        "enemy_is_casting" => Some(ENEMY_IS_CASTING),
        "in_danger_zone" => Some(IN_DANGER_ZONE),
        "team_focus_target" => Some(TEAM_FOCUS_TARGET),
        "focus_target_distance" => Some(FOCUS_TARGET_DISTANCE),
        "on_focus_target" => Some(ON_FOCUS_TARGET),
        _ => None,
    }
}
