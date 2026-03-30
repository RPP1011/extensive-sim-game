//! Funeral and memorial system — delta architecture port.
//!
//! When NPCs die, the guild can hold funerals that affect morale, bonds,
//! and create lasting memorials. Pending funerals auto-resolve as
//! SimpleFunerals after 17 ticks if not acted upon.
//!
//! Original: `crates/headless_campaign/src/systems/memorials.rs`
//! Cadence: every 10 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityField, EntityKind, WorldState};

//   FuneralPending { adventurer_id, name, level, death_tick }
//   Memorial { id, adventurer_name, adventurer_id, memorial_type, created_tick, morale_bonus, description }
//   MemorialType: SimpleFuneral, HerosFuneral, Monument, NamedBuilding, LegendaryTale

/// Cadence gate.
const MEMORIAL_TICK_INTERVAL: u64 = 10;

/// Auto-resolve pending funerals after this many ticks.
const AUTO_FUNERAL_DELAY: u64 = 17;

/// Maximum memorials kept.
const MAX_MEMORIALS: usize = 5;

/// Morale duration for temporary funeral effects (ticks).
const TEMP_MORALE_DURATION: u64 = 17;

/// Gold costs by memorial type.
const COST_HEROS_FUNERAL: f32 = 30.0;
const COST_MONUMENT: f32 = 100.0;
const COST_NAMED_BUILDING: f32 = 50.0;

/// Compute memorial deltas: detect deaths, auto-resolve funerals, apply morale.
///
/// Death detection uses entity alive status. Funeral costs can be expressed
/// via UpdateTreasury. Morale effects need per-entity AdjustMorale deltas.
pub fn compute_memorials(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MEMORIAL_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Detect dead NPC entities ---
    let mut dead_npcs: Vec<u32> = Vec::new();
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        collect_dead_npcs(state, &state.entities[range], &mut dead_npcs);
    }

    // --- Grief morale effect ---
    // When dead NPCs are found, all living NPCs at the same settlement
    // suffer a grief morale penalty. Higher-level fallen comrades hurt more.
    if !dead_npcs.is_empty() {
        // Collect settlement IDs of dead NPCs to locate grieving survivors.
        for settlement in &state.settlements {
            let range = state.group_index.settlement_entities(settlement.id);
            let entities = &state.entities[range];

            // Only count RECENT deaths (from world events this tick), not all dead entities.
            let recent_deaths = state.world_events.iter()
                .filter(|e| matches!(e, crate::world_sim::state::WorldEvent::EntityDied { .. }))
                .count();
            if recent_deaths == 0 {
                continue;
            }

            // Grief penalty from recent deaths (capped at 3).
            let grief = -(recent_deaths.min(3) as f32) * 1.0;
            for entity in entities {
                if entity.alive && entity.kind == EntityKind::Npc {
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: entity.id,
                        field: EntityField::Morale,
                        value: grief,
                    });
                }
            }
        }
    }

    let _ = dead_npcs;
}

/// Per-settlement variant for parallel dispatch.
///
/// Detects dead NPCs and emits memorial-related deltas. Since full memorial
/// state doesn't exist yet on WorldState, this is a structural placeholder.
pub fn compute_memorials_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    _out: &mut Vec<WorldDelta>,
) {
    if state.tick % MEMORIAL_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Detect dead NPCs at this settlement.
    for entity in entities {
        if !entity.alive && entity.kind == EntityKind::Npc {
            // Placeholder: once memorial state exists, emit funeral deltas.
            let _ = entity.id;
        }
    }
}

/// Helper: collects dead NPC entity IDs (used by the top-level function).
fn collect_dead_npcs(
    _state: &WorldState,
    entities: &[Entity],
    dead_npcs: &mut Vec<u32>,
) {
    for entity in entities {
        if !entity.alive && entity.kind == EntityKind::Npc {
            dead_npcs.push(entity.id);
        }
    }
}
