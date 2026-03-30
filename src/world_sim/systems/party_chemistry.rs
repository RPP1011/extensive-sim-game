//! Party chemistry — fires every 7 ticks.
//!
//! NPCs who share the same grid build "chemistry" which improves their
//! combat effectiveness. In the delta architecture, chemistry bonuses are
//! expressed as Buff status effects on co-located NPCs.
//!
//! Original: `crates/headless_campaign/src/systems/party_chemistry.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};

/// Chemistry tick interval.
const CHEMISTRY_INTERVAL: u64 = 7;

/// Buff factor for co-located NPCs (synergy).
const CHEMISTRY_BUFF_FACTOR: f32 = 1.05;

/// Buff duration.
const CHEMISTRY_BUFF_MS: u32 = 7_000;

pub fn compute_party_chemistry(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CHEMISTRY_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_party_chemistry_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_party_chemistry_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % CHEMISTRY_INTERVAL != 0 {
        return;
    }

    // For settlements with multiple friendly NPCs, apply a chemistry buff.
    let npc_entities: Vec<&Entity> = entities
        .iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .collect();

    if npc_entities.len() < 2 {
        return;
    }

    for entity in &npc_entities {
        // Don't stack if already buffed
        let already_buffed = entity.status_effects.iter().any(|s| {
            matches!(&s.kind, StatusEffectKind::Buff { stat, .. } if stat == "chemistry")
        });
        if already_buffed {
            continue;
        }

        out.push(WorldDelta::ApplyStatus {
            target_id: entity.id,
            status: StatusEffect {
                kind: StatusEffectKind::Buff {
                    stat: "chemistry".to_string(),
                    factor: CHEMISTRY_BUFF_FACTOR,
                },
                source_id: entity.id,
                remaining_ms: CHEMISTRY_BUFF_MS,
            },
        });
    }
}
