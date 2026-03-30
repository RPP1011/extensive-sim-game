//! Insurance contracts system — fires every 7 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/insurance.rs`.
//! Entities pay premiums to protect against loss. Claims are auto-processed
//! when matching conditions are detected. Premium payments and claim
//! payouts are expressed as TransferGold and UpdateTreasury deltas.
//!
//!   (id, policy_type, holder_entity_id, premium_per_tick, coverage,
//!    started_tick, expires_tick, claims_made, max_claims)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// How often insurance ticks.
const INSURANCE_INTERVAL: u64 = 7;

/// Base premium per NPC per tick interval (scales with threat).
const BASE_PREMIUM: f32 = 0.3;

/// Fraction of NPC gold taken as premium.
const PREMIUM_FRACTION: f32 = 0.01;

/// Claim payout multiplier (multiple of premium).
const CLAIM_PAYOUT_MULTIPLIER: f32 = 10.0;

/// HP threshold below which a "claim" is triggered (took heavy damage).
const DAMAGE_CLAIM_THRESHOLD: f32 = 0.3;

/// Compute insurance deltas.
///
/// Without explicit policy records, this system models insurance as a
/// continuous premium drain on NPC entities flowing into settlement
/// treasuries, with claim payouts triggered when NPCs fall below an
/// HP threshold (proxy for adventurer death / disaster claims).
pub fn compute_insurance(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % INSURANCE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in &state.entities {
        if entity.kind != EntityKind::Npc || !entity.alive {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };
        let home_id = match npc.home_settlement_id {
            Some(id) => id,
            None => continue,
        };

        // --- Premium payment: NPC pays into settlement treasury ---
        let premium = (npc.gold * PREMIUM_FRACTION).max(BASE_PREMIUM).min(npc.gold);
        if premium > 0.01 {
            out.push(WorldDelta::TransferGold {
                from_entity: entity.id,
                to_entity: home_id,
                amount: premium,
            });
        }

        // --- Claim check: if NPC HP is below threshold, pay out ---
        let hp_ratio = entity.hp / entity.max_hp.max(1.0);
        let home_treasury = state.settlement(home_id).map(|s| s.treasury).unwrap_or(0.0);
        if hp_ratio < DAMAGE_CLAIM_THRESHOLD && hp_ratio > 0.0 && home_treasury > -100.0 {
            let payout = premium * CLAIM_PAYOUT_MULTIPLIER;
            out.push(WorldDelta::TransferGold {
                from_entity: home_id,
                to_entity: entity.id,
                amount: payout,
            });
            // Treasury absorbs the loss.
            out.push(WorldDelta::UpdateTreasury {
                settlement_id: home_id,
                delta: -payout,
            });
        }
    }
}
