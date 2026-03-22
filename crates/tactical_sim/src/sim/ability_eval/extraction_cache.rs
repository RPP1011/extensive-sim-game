//! Per-tick extraction cache: precomputes expensive shared data once per tick,
//! reused across all hero extractions.
//!
//! The main wins:
//! 1. `summarize_abilities()` called once per unit (not 2× per hero extraction)
//! 2. Entity scoring computed once (not once per hero)
//! 3. Zone/threat base data computed once per team (dx/dy adjusted per hero)
//! 4. Unit lookup by ID via HashMap instead of linear scan

use std::collections::HashMap;
use crate::sim::{distance, is_alive, SimState, Team, UnitState, SimVec2};
use super::game_state::{AbilitySummary, summarize_abilities, ENTITY_FEATURE_DIM_LEGACY};
use super::game_state::unit_dps;

/// Precomputed per-unit data that doesn't depend on the observing hero.
pub struct UnitCache {
    pub id: u32,
    pub team: Team,
    pub hp: i32,
    pub max_hp: i32,
    pub position: SimVec2,
    pub hp_pct: f32,
    pub dps: f32,
    pub is_casting: bool,
    pub ability_summary: AbilitySummary,
    /// Priority score component that doesn't depend on observer distance.
    /// observer adds: (1.0 - dist/20).max(0) * 0.4
    pub base_priority_score: f32,
}

/// Per-tick cache built once, shared across all hero extractions in the same tick.
pub struct ExtractionCache {
    /// Per-unit precomputed data, indexed by unit ID.
    pub units: HashMap<u32, UnitCache>,
    /// All alive unit IDs sorted by team.
    pub alive_enemy_ids: Vec<u32>,
    pub alive_ally_ids_by_team: HashMap<Team, Vec<u32>>,
}

impl ExtractionCache {
    /// Build cache from current sim state. Called once per tick.
    pub fn build(state: &SimState) -> Self {
        let mut units = HashMap::with_capacity(state.units.len());
        let mut alive_enemy_ids = Vec::new();
        let mut alive_ally_ids_by_team: HashMap<Team, Vec<u32>> = HashMap::new();

        for u in &state.units {
            if !is_alive(u) { continue; }

            let hp_pct = u.hp as f32 / u.max_hp.max(1) as f32;
            let dps = unit_dps(u);
            let abil = summarize_abilities(u);
            let is_casting = u.casting.is_some();

            // Base priority score (distance-independent part)
            let mut base_score = dps / 30.0 + abil.ability_damage / 50.0;
            if abil.control_duration_ms > 0.0 && abil.control_cd_pct < 0.01 {
                base_score += 0.3;
            }
            if is_casting {
                base_score += 0.6;
            }

            units.insert(u.id, UnitCache {
                id: u.id,
                team: u.team,
                hp: u.hp,
                max_hp: u.max_hp,
                position: u.position,
                hp_pct,
                dps,
                is_casting,
                ability_summary: abil,
                base_priority_score: base_score,
            });

            alive_ally_ids_by_team.entry(u.team).or_default().push(u.id);
        }

        ExtractionCache { units, alive_enemy_ids, alive_ally_ids_by_team }
    }

    /// Select top-K entity slots for a specific observer unit.
    /// Uses cached base scores + observer-relative distance.
    pub fn select_entity_slots(&self, observer: &UnitState, max_slots: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = Vec::with_capacity(self.units.len());

        for uc in self.units.values() {
            if uc.id == observer.id { continue; }

            let dist = distance(observer.position, uc.position);
            let mut score = uc.base_priority_score;

            // Distance-dependent scoring
            score += (1.0 - dist / 20.0).max(0.0) * 0.4;

            let is_enemy = uc.team != observer.team;
            if is_enemy && uc.hp_pct < 0.25 { score += 0.5; }
            if !is_enemy && uc.hp_pct < 0.3 { score += 0.4; }

            scored.push((uc.id, score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(max_slots).map(|&(id, _)| id).collect()
    }

    /// Get cached unit data by ID (O(1) instead of linear scan).
    pub fn get(&self, id: u32) -> Option<&UnitCache> {
        self.units.get(&id)
    }
}
