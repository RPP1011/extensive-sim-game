//! Morale + Culture system for combat-level AI behavior.
//!
//! Morale drives AI behavior, not stats. Culture profiles (loaded from TOML)
//! filter morale inputs through weighted multipliers. Routing units cascade
//! morale penalties to nearby allies based on `cascade_susceptibility`.

pub mod culture;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use crate::sim::{SimState, SimVec2, UnitState};

// ---------------------------------------------------------------------------
// Morale levels
// ---------------------------------------------------------------------------

/// Discrete morale states that drive AI behavior overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoraleLevel {
    /// High morale — aggression boost, willingness to press advantage.
    FiredUp,
    /// Normal morale — standard behavior.
    Steady,
    /// Faltering morale — increased caution, tendency to disengage.
    Wavering,
    /// Broken morale — forced retreat, disobedience.
    Routing,
}

impl Default for MoraleLevel {
    fn default() -> Self {
        MoraleLevel::Steady
    }
}

// ---------------------------------------------------------------------------
// Unit morale state
// ---------------------------------------------------------------------------

/// Per-unit morale tracking.
#[derive(Debug, Clone)]
pub struct UnitMorale {
    /// Current morale value in [0.0, 1.0].
    pub value: f32,
    /// Derived morale level based on culture thresholds.
    pub level: MoraleLevel,
    /// Name of the culture profile applied to this unit.
    pub culture_name: String,
}

impl Default for UnitMorale {
    fn default() -> Self {
        Self {
            value: 0.65,
            level: MoraleLevel::Steady,
            culture_name: "Disciplined".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Raw morale inputs (5 categories)
// ---------------------------------------------------------------------------

/// Raw morale input values computed from simulation state.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoraleInputs {
    /// Self: HP ratio, recent damage taken.
    pub self_input: f32,
    /// Allies: nearby ally HP, ally deaths.
    pub allies_input: f32,
    /// Threats: enemy count, outnumbered ratio.
    pub threats_input: f32,
    /// Leadership: leader alive and healthy.
    pub leadership_input: f32,
    /// Situation: team average HP advantage.
    pub situation_input: f32,
}

// ---------------------------------------------------------------------------
// Morale state container
// ---------------------------------------------------------------------------

/// Container for all unit morale states within a combat.
#[derive(Debug, Clone, Default)]
pub struct MoraleState {
    pub morale_by_unit: HashMap<u32, UnitMorale>,
}

impl MoraleState {
    /// Initialize morale for all alive units with the given culture assignment.
    pub fn init(
        state: &SimState,
        culture_assignments: &HashMap<u32, String>,
        default_culture: &str,
    ) -> Self {
        let mut morale_by_unit = HashMap::new();
        for unit in state.units.iter().filter(|u| u.hp > 0) {
            let culture_name = culture_assignments
                .get(&unit.id)
                .cloned()
                .unwrap_or_else(|| default_culture.to_string());
            morale_by_unit.insert(
                unit.id,
                UnitMorale {
                    value: 0.65,
                    level: MoraleLevel::Steady,
                    culture_name,
                },
            );
        }
        MoraleState { morale_by_unit }
    }

    /// Get morale level for a unit.
    pub fn level(&self, unit_id: u32) -> MoraleLevel {
        self.morale_by_unit
            .get(&unit_id)
            .map(|m| m.level)
            .unwrap_or(MoraleLevel::Steady)
    }

    /// Get morale value for a unit.
    pub fn value(&self, unit_id: u32) -> f32 {
        self.morale_by_unit
            .get(&unit_id)
            .map(|m| m.value)
            .unwrap_or(0.65)
    }
}

// ---------------------------------------------------------------------------
// Input computation
// ---------------------------------------------------------------------------

/// Compute raw morale inputs for a given unit from the simulation state.
pub fn compute_morale_inputs(state: &SimState, unit: &UnitState) -> MoraleInputs {
    let hp_ratio = unit.hp.max(0) as f32 / unit.max_hp.max(1) as f32;

    // Self input: based on HP ratio (high HP → positive, low HP → negative)
    let self_input = hp_ratio * 2.0 - 1.0; // maps [0,1] → [-1, 1]

    // Allies input: average HP of nearby allies (within 15 units distance)
    let mut ally_hp_sum = 0.0f32;
    let mut ally_count = 0u32;
    let mut alive_allies = 0u32;
    for other in state.units.iter() {
        if other.id == unit.id || other.team != unit.team {
            continue;
        }
        if other.hp > 0 {
            alive_allies += 1;
            let dist = distance(&unit.position, &other.position);
            if dist < 15.0 {
                ally_hp_sum += other.hp.max(0) as f32 / other.max_hp.max(1) as f32;
                ally_count += 1;
            }
        }
    }
    let allies_input = if ally_count > 0 {
        (ally_hp_sum / ally_count as f32) * 2.0 - 1.0
    } else {
        -0.5 // No nearby allies is bad
    };

    // Threats input: ratio of enemies to allies (more enemies → more negative)
    let enemy_count = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != unit.team)
        .count() as f32;
    let total_allies = alive_allies.max(1) as f32;
    let ratio = total_allies / (total_allies + enemy_count).max(1.0);
    let threats_input = ratio * 2.0 - 1.0; // even → 0, outnumbered → negative

    // Leadership input: is the highest-HP ally on the team alive and healthy?
    let leader_hp = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == unit.team)
        .map(|u| u.max_hp)
        .max()
        .unwrap_or(0);
    let leader_alive = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == unit.team && u.max_hp == leader_hp)
        .map(|u| u.hp as f32 / u.max_hp.max(1) as f32)
        .next()
        .unwrap_or(0.0);
    let leadership_input = leader_alive * 2.0 - 1.0;

    // Situation input: team average HP vs enemy average HP
    let team_avg_hp = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == unit.team)
        .map(|u| u.hp.max(0) as f32 / u.max_hp.max(1) as f32)
        .sum::<f32>()
        / total_allies;
    let enemy_avg_hp = if enemy_count > 0.0 {
        state
            .units
            .iter()
            .filter(|u| u.hp > 0 && u.team != unit.team)
            .map(|u| u.hp.max(0) as f32 / u.max_hp.max(1) as f32)
            .sum::<f32>()
            / enemy_count
    } else {
        0.0
    };
    let situation_input = (team_avg_hp - enemy_avg_hp).clamp(-1.0, 1.0);

    MoraleInputs {
        self_input,
        allies_input,
        threats_input,
        leadership_input,
        situation_input,
    }
}

/// Compute the cascade penalty for a unit based on nearby routing allies.
pub fn compute_cascade_penalty(
    state: &SimState,
    unit: &UnitState,
    morale_state: &MoraleState,
) -> f32 {
    let mut routing_nearby = 0u32;
    for other in state.units.iter() {
        if other.id == unit.id || other.team != unit.team || other.hp <= 0 {
            continue;
        }
        let dist = distance(&unit.position, &other.position);
        if dist < 12.0 {
            if morale_state.level(other.id) == MoraleLevel::Routing {
                routing_nearby += 1;
            }
        }
    }
    // Each routing neighbor contributes -0.15 penalty (before culture scaling)
    routing_nearby as f32 * -0.15
}

/// Update morale for all units in the combat.
pub fn update_morale(
    state: &SimState,
    morale_state: &mut MoraleState,
    cultures: &culture::MoraleCultureRegistry,
) {
    // Collect unit IDs to avoid borrow conflicts
    let unit_ids: Vec<u32> = morale_state.morale_by_unit.keys().copied().collect();

    for unit_id in unit_ids {
        let Some(unit) = state.units.iter().find(|u| u.id == unit_id) else {
            continue;
        };
        if unit.hp <= 0 {
            continue;
        }

        let culture_name = morale_state
            .morale_by_unit
            .get(&unit_id)
            .map(|m| m.culture_name.clone())
            .unwrap_or_default();
        let profile = cultures.get(&culture_name).or_else(|| cultures.default_profile());

        let Some(profile) = profile else { continue };

        // Compute raw inputs
        let inputs = compute_morale_inputs(state, unit);

        // Apply culture weights
        let weighted = inputs.self_input * profile.input_weights.self_weight
            + inputs.allies_input * profile.input_weights.allies_weight
            + inputs.threats_input * profile.input_weights.threats_weight
            + inputs.leadership_input * profile.input_weights.leadership_weight
            + inputs.situation_input * profile.input_weights.situation_weight;

        // Cascade penalty
        let cascade = compute_cascade_penalty(state, unit, morale_state);
        let cascade_scaled = cascade * profile.cascade_susceptibility;

        // Compute delta (positive = morale gain, negative = morale loss)
        let delta = (weighted * 0.05 + cascade_scaled * 0.1) * profile.volatility;

        // Update morale value
        if let Some(morale) = morale_state.morale_by_unit.get_mut(&unit_id) {
            morale.value = (morale.value + delta).clamp(0.0, 1.0);

            // Derive level from thresholds
            morale.level = if morale.value >= profile.fired_up_threshold {
                MoraleLevel::FiredUp
            } else if morale.value <= profile.routing_threshold {
                MoraleLevel::Routing
            } else if morale.value <= profile.wavering_threshold {
                MoraleLevel::Wavering
            } else {
                MoraleLevel::Steady
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn distance(a: &SimVec2, b: &SimVec2) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}
