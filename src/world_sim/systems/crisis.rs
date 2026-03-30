#![allow(unused)]
//! Endgame crisis system — fires every tick.
//!
//! Crises escalate global threat, damage settlements, spawn powerful
//! hostile entities, and drain resources. Multiple crisis types are
//! modeled via region threat levels and entity interactions.
//!
//! Original: `crates/headless_campaign/src/systems/crisis.rs`
//!
//! Crisis types (mapped to delta-producible effects):
//! - SleepingKing: escalate fidelity in king's region, buff hostiles
//! - Breach: damage entities near breach location, drain stockpiles
//! - Corruption: damage settlements, consume commodities
//! - Decline: drain treasury and stockpiles globally
//! - Unifier: buff hostile entities, drain friendly resources
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EntityKind, StatusEffect, StatusEffectKind, WorldState, WorldTeam};
use crate::world_sim::state::{entity_hash_f32};
use crate::world_sim::NUM_COMMODITIES;


/// Threat threshold above which crisis effects intensify.
const CRISIS_THREAT_THRESHOLD: f32 = 70.0;

/// Per-tick treasury drain during decline crisis.
const DECLINE_GOLD_DRAIN: f32 = 1.0;

/// Per-tick commodity drain during corruption crisis.
const CORRUPTION_COMMODITY_DRAIN: f32 = 0.5;

/// Interval (ticks) at which crisis effects pulse.
const CRISIS_PULSE_INTERVAL: u64 = 13;

pub fn compute_crisis(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Without active_crises on WorldState, we derive crisis-like effects
    // from the region threat levels. Regions with extreme threat (> 70)
    // are treated as crisis zones.

    if state.tick % CRISIS_PULSE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let has_crisis_regions = state
        .regions
        .iter()
        .any(|r| r.threat_level > CRISIS_THREAT_THRESHOLD);
    if !has_crisis_regions {
        return;
    }

    for (ri, region) in state.regions.iter().enumerate() {
        if region.threat_level <= CRISIS_THREAT_THRESHOLD {
            continue;
        }

        let severity = (region.threat_level - CRISIS_THREAT_THRESHOLD) / 30.0; // 0..1
        let crisis_type_roll = entity_hash_f32(ri as u32, state.tick, 0xC815_15EE);

        if crisis_type_roll < 0.25 {
            // --- Breach-style: damage entities near the crisis region ---
            for entity in &state.entities {
                if !entity.alive {
                    continue;
                }
                let ex = entity.pos.0 - (ri as f32 * 20.0);
                let ey = entity.pos.1 - (ri as f32 * 15.0);
                if ex * ex + ey * ey < 400.0 {
                    let damage = 3.0 * severity;
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: damage,
                        source_id: 0,
                    });
                }
            }

            // Escalate fidelity for grids in the region.
            for grid in &state.grids {
                let dx = grid.center.0 - (ri as f32 * 20.0);
                let dy = grid.center.1 - (ri as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 && grid.fidelity != Fidelity::High {
                    out.push(WorldDelta::EscalateFidelity {
                        grid_id: grid.id,
                        new_fidelity: Fidelity::High,
                    });
                }
            }
        } else if crisis_type_roll < 0.50 {
            // --- Corruption-style: drain commodities from settlements ---
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (ri as f32 * 20.0);
                let dy = settlement.pos.1 - (ri as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 {
                    for c in 0..NUM_COMMODITIES {
                        let drain = CORRUPTION_COMMODITY_DRAIN * severity;
                        if settlement.stockpile[c] > drain {
                            out.push(WorldDelta::ConsumeCommodity {
                                location_id: settlement.id,
                                commodity: c,
                                amount: drain,
                            });
                        }
                    }
                }
            }
        } else if crisis_type_roll < 0.75 {
            // --- Decline-style: drain treasury from all settlements ---
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (ri as f32 * 20.0);
                let dy = settlement.pos.1 - (ri as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 && settlement.treasury > -100.0 {
                    out.push(WorldDelta::UpdateTreasury {
                        location_id: settlement.id,
                        delta: -DECLINE_GOLD_DRAIN * severity,
                    });
                }
            }

            // Debuff friendly NPCs in the area (morale drain proxy).
            for entity in &state.entities {
                if !entity.alive
                    || entity.kind != EntityKind::Npc
                    || entity.team != WorldTeam::Friendly
                {
                    continue;
                }
                let ex = entity.pos.0 - (ri as f32 * 20.0);
                let ey = entity.pos.1 - (ri as f32 * 15.0);
                if ex * ex + ey * ey < 400.0 {
                    out.push(WorldDelta::ApplyStatus {
                        target_id: entity.id,
                        status: StatusEffect {
                            kind: StatusEffectKind::Debuff {
                                stat: "morale".to_string(),
                                factor: 0.9,
                            },
                            source_id: 0,
                            remaining_ms: CRISIS_PULSE_INTERVAL as u32 * 100,
                        },
                    });
                }
            }
        } else {
            // --- Unifier-style: buff hostile entities ---
            for entity in &state.entities {
                if !entity.alive || entity.team != WorldTeam::Hostile {
                    continue;
                }
                let ex = entity.pos.0 - (ri as f32 * 20.0);
                let ey = entity.pos.1 - (ri as f32 * 15.0);
                if ex * ex + ey * ey < 400.0 {
                    out.push(WorldDelta::ApplyStatus {
                        target_id: entity.id,
                        status: StatusEffect {
                            kind: StatusEffectKind::Buff {
                                stat: "attack".to_string(),
                                factor: 1.0 + 0.2 * severity,
                            },
                            source_id: 0,
                            remaining_ms: CRISIS_PULSE_INTERVAL as u32 * 100,
                        },
                    });
                    out.push(WorldDelta::Shield {
                        target_id: entity.id,
                        amount: 5.0 * severity,
                        source_id: 0,
                    });
                }
            }
        }
    }
}
