use crate::ai::core::{SimState, UnitIntent, UnitState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionOutcome {
    Victory,
    Defeat,
}

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

    pub fn generate_intents(&mut self, sim: &SimState, dt_ms: u32) -> Vec<UnitIntent> {
        crate::ai::squad::generate_intents(sim, &mut self.squad_state, dt_ms)
    }
}

pub fn scale_enemy_stats(unit: &mut UnitState, global_turn: u32) {
    let scale = 1.0 + (global_turn as f32 / 10.0).min(2.0);
    unit.hp = ((unit.hp as f32) * scale) as i32;
    unit.max_hp = ((unit.max_hp as f32) * scale) as i32;
    unit.attack_damage = ((unit.attack_damage as f32) * scale) as i32;
}

pub fn threat_level(global_turn: u32) -> u32 {
    match global_turn {
        0..=4   => 1,
        5..=9   => 2,
        10..=14 => 3,
        15..=19 => 4,
        _       => 5,
    }
}

pub fn threat_level_roman(level: u32) -> &'static str {
    match level {
        1 => "I",
        2 => "II",
        3 => "III",
        4 => "IV",
        _ => "V",
    }
}
