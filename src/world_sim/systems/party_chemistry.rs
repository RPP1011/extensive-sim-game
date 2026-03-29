#![allow(unused)]
//! Party chemistry — fires every 7 ticks.
//!
//! NPCs who share the same grid build "chemistry" which improves their
//! combat effectiveness. In the delta architecture, chemistry bonuses are
//! expressed as Buff status effects on co-located NPCs.
//!
//! Original: `crates/headless_campaign/src/systems/party_chemistry.rs`
//!
//! NEEDS STATE: `party_chemistry: HashMap<(u32,u32), f32>` on WorldState
//! NEEDS DELTA: ModifyChemistry

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, StatusEffect, StatusEffectKind, WorldState};

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

    // For each grid with multiple friendly NPCs, apply a chemistry buff
    // to all NPCs in that grid.

    for grid in &state.grids {
        let npc_ids: Vec<u32> = grid
            .entity_ids
            .iter()
            .copied()
            .filter(|&id| {
                state
                    .entity(id)
                    .map(|e| e.alive && e.kind == EntityKind::Npc)
                    .unwrap_or(false)
            })
            .collect();

        if npc_ids.len() < 2 {
            continue;
        }

        for &npc_id in &npc_ids {
            // Don't stack if already buffed
            let already_buffed = state.entity(npc_id).map_or(true, |e| {
                e.status_effects.iter().any(|s| {
                    matches!(&s.kind, StatusEffectKind::Buff { stat, .. } if stat == "chemistry")
                })
            });
            if already_buffed {
                continue;
            }

            out.push(WorldDelta::ApplyStatus {
                target_id: npc_id,
                status: StatusEffect {
                    kind: StatusEffectKind::Buff {
                        stat: "chemistry".to_string(),
                        factor: CHEMISTRY_BUFF_FACTOR,
                    },
                    source_id: npc_id,
                    remaining_ms: CHEMISTRY_BUFF_MS,
                },
            });
        }
    }
}
