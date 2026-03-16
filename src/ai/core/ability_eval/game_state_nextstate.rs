use crate::ai::core::{distance, is_alive, SimState, Team, UnitState};
use super::game_state::rich_entity_features;
use super::game_state_v2::{GameStateV2, compute_aggregate_features};
use super::game_state_threats::extract_threats_v2;
use super::game_state_positions::extract_position_tokens;

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Next-state prediction dataset
// ---------------------------------------------------------------------------

/// Dense game state snapshot for next-state prediction.
/// Sampled frequently; Python pairs snapshots at different deltas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextStateSample {
    /// Per-entity feature vectors (30-dim each), variable length.
    pub entities: Vec<Vec<f32>>,
    /// Type ID per entity: 0=self, 1=enemy, 2=ally.
    pub entity_types: Vec<u8>,
    /// Unit ID per entity slot (for cross-time alignment).
    pub entity_unit_ids: Vec<u32>,
    /// Per-threat feature vectors (8-dim each).
    pub threats: Vec<Vec<f32>>,
    /// Per-position feature vectors (8-dim each).
    #[serde(default)]
    pub positions: Vec<Vec<f32>>,
    /// All unit HP ratios indexed by unit ID: [[id, hp_pct], ...].
    pub unit_hps: Vec<(u32, f32)>,
    /// Scenario name.
    pub scenario: String,
    /// Tick number.
    pub tick: u64,
}

/// Per-unit ability DSL text, emitted once per scenario.
/// Maps unit_id -> list of DSL strings (input to ability transformer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityRegistryEntry {
    pub unit_id: u32,
    pub abilities: Vec<String>,  // DSL text per ability
}

/// Full ability registry for a scenario, emitted before tick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityRegistry {
    pub scenario: String,
    pub entries: Vec<AbilityRegistryEntry>,
}

/// Extract game state V2 + unit IDs for next-state prediction.
pub fn extract_game_state_with_ids(
    state: &SimState,
    unit: &UnitState,
) -> (GameStateV2, Vec<u32>) {
    let mut entities: Vec<Vec<f32>> = Vec::new();
    let mut entity_types: Vec<u8> = Vec::new();
    let mut unit_ids: Vec<u32> = Vec::new();

    // Self entity
    entities.push(rich_entity_features(state, unit, unit, true).to_vec());
    entity_types.push(0);
    unit_ids.push(unit.id);

    // All enemies sorted by distance
    let mut enemies: Vec<&UnitState> = state
        .units
        .iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    enemies.sort_by(|a, b| {
        distance(unit.position, a.position)
            .partial_cmp(&distance(unit.position, b.position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for enemy in &enemies {
        entities.push(rich_entity_features(state, unit, enemy, false).to_vec());
        entity_types.push(1);
        unit_ids.push(enemy.id);
    }

    // All allies (excluding self) sorted by HP% ascending
    let mut allies: Vec<&UnitState> = state
        .units
        .iter()
        .filter(|u| u.team == unit.team && is_alive(u) && u.id != unit.id)
        .collect();
    allies.sort_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    for ally in &allies {
        entities.push(rich_entity_features(state, unit, ally, false).to_vec());
        entity_types.push(2);
        unit_ids.push(ally.id);
    }

    let threats = extract_threats_v2(state, unit);
    let positions = extract_position_tokens(state, unit);

    let aggregate_features = compute_aggregate_features(state, unit, &unit_ids);
    let gs = GameStateV2 {
        entities,
        entity_types,
        threats,
        positions,
        zones: Vec::new(),
        aggregate_features,
    };
    (gs, unit_ids)
}

/// Generate next-state prediction dataset, streaming via callback.
///
/// Samples game state densely (every `sample_interval` ticks) with unit IDs
/// and per-unit HP snapshots. Calls `emit` for each sample instead of
/// collecting, to avoid OOM on long fights.
pub fn generate_nextstate_dataset_streaming(
    initial_sim: SimState,
    initial_squad_ai: crate::ai::squad::SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    sample_interval: u64,
    mut emit: impl FnMut(NextStateSample),
    mut emit_registry: impl FnMut(AbilityRegistry),
) -> usize {
    use crate::ai::core::{step, FIXED_TICK_MS};
    use crate::ai::squad::generate_intents;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut count = 0usize;

    // Emit ability registry once before tick loop
    use crate::ai::effects::dsl::emit::emit_ability_dsl;
    let entries: Vec<AbilityRegistryEntry> = sim.units.iter().map(|u| {
        let abilities = u.abilities.iter().map(|slot| {
            emit_ability_dsl(&slot.def)
        }).collect();
        AbilityRegistryEntry { unit_id: u.id, abilities }
    }).collect();
    emit_registry(AbilityRegistry {
        scenario: scenario_name.to_string(),
        entries,
    });

    for tick in 0..max_ticks {
        let intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        if tick % sample_interval == 0 {
            let unit_hps: Vec<(u32, f32)> = sim
                .units
                .iter()
                .map(|u| (u.id, u.hp.max(0) as f32 / u.max_hp.max(1) as f32))
                .collect();

            let heroes: Vec<&UnitState> = sim
                .units
                .iter()
                .filter(|u| u.team == Team::Hero && is_alive(u))
                .collect();

            for hero in &heroes {
                let (gs, ids) = extract_game_state_with_ids(&sim, hero);
                emit(NextStateSample {
                    entities: gs.entities,
                    entity_types: gs.entity_types,
                    entity_unit_ids: ids,
                    threats: gs.threats,
                    positions: gs.positions,
                    unit_hps: unit_hps.clone(),
                    scenario: scenario_name.to_string(),
                    tick,
                });
                count += 1;
            }
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 || heroes_alive == 0 {
            break;
        }
    }

    count
}
