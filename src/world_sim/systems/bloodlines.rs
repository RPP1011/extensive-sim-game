#![allow(unused)]
//! Bloodline legacy system — fires every 33 ticks.
//!
//! When high-level NPCs die, their bloodline can produce descendants with
//! inherited stat bonuses. In the delta architecture, this is modeled as
//! Heal (prestige recovery) and Buff (inherited power) effects.
//!
//! Original: `crates/headless_campaign/src/systems/bloodlines.rs`
//!
//! NEEDS STATE: `bloodlines: Vec<Bloodline>` on WorldState
//! NEEDS DELTA: SpawnDescendant (new entity creation)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, StatusEffect, StatusEffectKind, WorldState};

/// Bloodline check interval.
const BLOODLINE_INTERVAL: u64 = 33;

/// Buff factor for bloodline descendants.
const BLOODLINE_BUFF_FACTOR: f32 = 1.10;

/// Buff duration for bloodline power.
const BLOODLINE_BUFF_MS: u32 = 30_000;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_bloodlines(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BLOODLINE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without bloodline tracking, we apply a proxy: high-level NPCs (level >= 8)
    // occasionally receive a "bloodline power" buff representing inherited strength.

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        if entity.level < 8 {
            continue;
        }

        let has_bloodline = entity.status_effects.iter().any(|s| {
            matches!(&s.kind, StatusEffectKind::Buff { stat, .. } if stat == "bloodline")
        });
        if has_bloodline {
            continue;
        }

        let roll = tick_hash(state.tick, entity.id as u64 ^ 0xB100D);
        if roll < 0.05 {
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: StatusEffect {
                    kind: StatusEffectKind::Buff {
                        stat: "bloodline".to_string(),
                        factor: BLOODLINE_BUFF_FACTOR,
                    },
                    source_id: entity.id,
                    remaining_ms: BLOODLINE_BUFF_MS,
                },
            });
        }
    }
}
