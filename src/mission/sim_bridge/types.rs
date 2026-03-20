use bevy::prelude::*;

use crate::ai::core::{SimState, SimVec2, UnitIntent, UnitState};
use crate::ai::pathing::GridNav;

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionOutcome {
    Victory,
    Defeat,
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Accumulates all `SimEvent`s produced during the current frame's fixed steps.
/// Cleared at the start of each `advance_sim_system`; consumed by downstream
/// systems (`apply_vfx_from_sim_events_system`, `apply_audio_sfx_from_sim_events_system`).
#[derive(Resource, Default)]
pub struct SimEventBuffer {
    pub events: Vec<crate::ai::core::SimEvent>,
}

/// Persists all `SimEvent`s for the full mission duration (unlike `SimEventBuffer`
/// which clears each frame). Used by the narrative progression system.
#[derive(Resource, Default)]
pub struct MissionEventLog {
    pub all_events: Vec<crate::ai::core::SimEvent>,
}

/// The live combat simulation state.
#[derive(Resource)]
pub struct MissionSimState {
    pub sim: SimState,
    /// Leftover ms from last frame (for fixed-step accumulation).
    pub tick_remainder_ms: u32,
    /// Set when the mission has ended.
    pub outcome: Option<MissionOutcome>,
    /// Per-tick AI state for the enemy team.
    pub enemy_ai: EnemyAiState,
    /// Player-issued intents that override AI for hero units.
    pub hero_intents: Vec<UnitIntent>,
    /// Navigation grid for terrain-aware combat (cover, elevation, pathfinding).
    pub grid_nav: Option<GridNav>,
}

/// Enemy AI state: uses Phase 3 SquadAiState with personality-driven,
/// force-based steering for formation-aware movement and targeting.
#[derive(Debug, Clone)]
pub struct EnemyAiState {
    pub squad_state: crate::ai::squad::SquadAiState,
}

impl EnemyAiState {
    pub fn new(sim: &SimState) -> Self {
        Self {
            squad_state: crate::ai::squad::SquadAiState::new_inferred(sim),
        }
    }

    /// Generate intents for all living enemy units using the Phase 3 squad AI.
    pub fn generate_intents(&mut self, sim: &SimState, dt_ms: u32) -> Vec<UnitIntent> {
        crate::ai::squad::generate_intents(sim, &mut self.squad_state, dt_ms)
    }
}

/// Records sim frames during a mission for post-combat replay.
#[derive(Resource, Default)]
pub struct MissionCombatRecording {
    pub frames: Vec<SimState>,
    pub events_per_frame: Vec<Vec<crate::ai::core::SimEvent>>,
    pub active: bool,
}

/// Stores the replay data from the most recently completed mission.
/// Persists across screen transitions so it can be viewed from the hub.
#[derive(Resource)]
pub struct LastMissionReplay {
    pub name: String,
    pub frames: Vec<SimState>,
    pub events_per_frame: Vec<Vec<crate::ai::core::SimEvent>>,
    /// Navigation grid for ASCII rendering of the combat terrain.
    pub grid_nav: Option<crate::ai::pathing::GridNav>,
}

/// Player input and selection state.
#[derive(Resource, Default)]
pub struct PlayerOrderState {
    pub selected_unit_ids: Vec<u32>,
    /// Pending move target in sim-space (x, y) — set by a ground click.
    pub pending_move: Option<SimVec2>,
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/// Marks a Bevy entity as a player-controlled unit visual.
#[derive(Component)]
pub struct PlayerUnitMarker {
    pub sim_unit_id: u32,
}

// ---------------------------------------------------------------------------
// Difficulty scaling helpers
// ---------------------------------------------------------------------------

/// Scales an enemy unit's stats based on the global campaign turn.
/// Scale factor: 1.0 at turn 0, up to 3.0 at turn 20+.
pub fn scale_enemy_stats(unit: &mut UnitState, global_turn: u32) {
    let scale = 1.0 + (global_turn as f32 / 10.0).min(2.0); // max 3x at turn 20
    unit.hp = ((unit.hp as f32) * scale) as i32;
    unit.max_hp = ((unit.max_hp as f32) * scale) as i32;
    unit.attack_damage = ((unit.attack_damage as f32) * scale) as i32;
}

/// Returns a threat level 1-5 based on the global campaign turn.
pub fn threat_level(global_turn: u32) -> u32 {
    match global_turn {
        0..=4   => 1,
        5..=9   => 2,
        10..=14 => 3,
        15..=19 => 4,
        _       => 5,
    }
}

/// Converts a numeric threat level (1-5) to its roman numeral string.
pub fn threat_level_roman(level: u32) -> &'static str {
    match level {
        1 => "I",
        2 => "II",
        3 => "III",
        4 => "IV",
        _ => "V",
    }
}
