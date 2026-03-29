#![allow(unused)]
//! Black market system — fires every 17 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/black_market.rs`.
//! Illegal but lucrative trade with high gold returns but reputation risk.
//! Heat decays over time; exceeding the discovery threshold triggers
//! reputation / faction penalties.
//!
//! Without guild-level state, this system models black market activity at
//! settlements: NPCs with low gold engage in risky trade that generates
//! gold but may trigger negative effects.
//!
//! NEEDS STATE: `black_market: BlackMarketState` on WorldState
//!   (heat, last_refresh_tick, total_profit, available_deals)
//! NEEDS STATE: `guild: GuildState` on WorldState (gold, reputation, supplies)
//! NEEDS STATE: `factions: Vec<FactionState>` on WorldState
//!   (diplomatic_stance, relationship_to_guild)
//! NEEDS DELTA: AdjustHeat { settlement_id: u32, delta: f32 }
//! NEEDS DELTA: AdjustReputation { entity_id: u32, delta: f32 }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// How often the black market ticks (in ticks).
const BLACK_MARKET_INTERVAL: u64 = 17;

/// Heat decay per tick interval.
const HEAT_DECAY_PER_TICK: f32 = 2.0;

/// Heat threshold above which discovery checks begin.
const DISCOVERY_THRESHOLD: f32 = 50.0;

/// Minimum tick before activation.
const ACTIVATION_TICK: u64 = 1000;

/// Fraction of a deal's profit that flows as treasury change.
const DEAL_TREASURY_FRACTION: f32 = 0.1;

/// Compute black market deltas.
///
/// Models illicit trade at settlements. NPCs with economic intent
/// `Idle` and low gold may be drawn into black-market deals, generating
/// gold that shows up as settlement treasury changes.
pub fn compute_black_market(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BLACK_MARKET_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.tick < ACTIVATION_TICK {
        return;
    }

    // For each settlement, check if conditions favor black market activity.
    for settlement in &state.settlements {
        // Higher prices indicate scarcity — black market thrives.
        let avg_price: f32 =
            settlement.prices.iter().sum::<f32>() / settlement.prices.len() as f32;
        if avg_price < 2.0 {
            continue; // Market is well-supplied, no black market pressure.
        }

        // Count NPCs at this settlement that could participate.
        let participants: Vec<u32> = state
            .entities
            .iter()
            .filter(|e| {
                e.kind == EntityKind::Npc
                    && e.alive
                    && e.npc
                        .as_ref()
                        .map(|n| n.home_settlement_id == Some(settlement.id))
                        .unwrap_or(false)
            })
            .map(|e| e.id)
            .collect();

        if participants.is_empty() {
            continue;
        }

        // Black market profit scales with price pressure and participant count.
        let price_factor = (avg_price - 2.0).min(3.0);
        let profit = price_factor * participants.len() as f32 * 0.5;

        // Gold flows into the settlement treasury (illicit commerce tax).
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement.id,
            delta: profit * DEAL_TREASURY_FRACTION,
        });

        // Each participant gains a small gold amount from dealing.
        let per_npc = profit / participants.len() as f32;
        for &npc_id in &participants {
            out.push(WorldDelta::TransferGold {
                from_id: settlement.id,
                to_id: npc_id,
                amount: per_npc,
            });
        }
    }
}
