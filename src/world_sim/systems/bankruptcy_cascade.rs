#![allow(unused)]
//! Bankruptcy cascade system — every 17 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/bankruptcy_cascade.rs`.
//! When a settlement's treasury drops below zero, creditors (other
//! settlements) face liquidity shortfalls that may cascade through the
//! economic network. Expressed as UpdateTreasury deltas propagating
//! losses and TransferGold for debt service.
//!
//!   (defaults_this_cycle, credit_freeze_ticks, systemic_risk, faction_debts,
//!    defaulted_factions)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// How often to evaluate cascade risk.
const CASCADE_INTERVAL: u64 = 17;

/// Treasury threshold below which a settlement is considered insolvent.
const INSOLVENCY_THRESHOLD: f32 = -10.0;

/// Fraction of loss propagated to connected settlements.
const LOSS_PROPAGATION_FRACTION: f32 = 0.15;

/// Systemic risk decay per tick (natural recovery).
const SYSTEMIC_RISK_DECAY: f32 = 0.5;

/// Maximum cascade iterations to prevent infinite loops.
const MAX_CASCADE_ITERATIONS: usize = 5;

/// Compute bankruptcy cascade deltas.
///
/// Settlements with negative treasury propagate losses to other
/// settlements (simulating inter-entity credit exposure). Cascading
/// defaults drain treasuries across the network.
pub fn compute_bankruptcy_cascade(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CASCADE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.settlements.len() < 2 {
        return;
    }

    // --- Identify insolvent settlements ---
    let insolvent: Vec<(u32, f32)> = state
        .settlements
        .iter()
        .filter(|s| s.treasury < INSOLVENCY_THRESHOLD)
        .map(|s| (s.id, s.treasury))
        .collect();

    if insolvent.is_empty() {
        return;
    }

    // --- Propagate losses to other settlements ---
    // Each insolvent settlement's deficit is partially borne by other
    // settlements (proxy for inter-faction credit exposure).
    let healthy_count = state
        .settlements
        .iter()
        .filter(|s| s.treasury >= 0.0)
        .count();

    if healthy_count == 0 {
        return;
    }

    for (insolvent_id, deficit) in &insolvent {
        let loss = deficit.abs() * LOSS_PROPAGATION_FRACTION;
        let per_settlement_loss = loss / healthy_count as f32;

        for settlement in &state.settlements {
            if settlement.id == *insolvent_id {
                continue;
            }
            if settlement.treasury < 0.0 {
                continue; // Already insolvent, skip.
            }
            if settlement.treasury <= -100.0 {
                continue; // At treasury floor, skip.
            }

            // Propagate a fraction of the loss.
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -per_settlement_loss,
            });
        }

        // The insolvent settlement itself gets a small recovery (debt
        // restructuring / bailout) to prevent permanent negative spiral.
        let recovery = deficit.abs() * 0.05;
        out.push(WorldDelta::UpdateTreasury {
            location_id: *insolvent_id,
            delta: recovery,
        });
    }

    // --- Systemic risk effect: all settlements see price inflation ---
    // The number of insolvent settlements drives systemic risk.
    let risk_factor = insolvent.len() as f32 / state.settlements.len() as f32;
    if risk_factor > 0.2 {
        for settlement in &state.settlements {
            // Prices inflate under systemic stress (supply chain disruption).
            let mut inflated = settlement.prices;
            for p in &mut inflated {
                *p *= 1.0 + risk_factor * 0.05;
            }
            out.push(WorldDelta::UpdatePrices {
                location_id: settlement.id,
                prices: inflated,
            });
        }
    }
}
