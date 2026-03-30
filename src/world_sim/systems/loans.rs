#![allow(unused)]
//! Loan and debt system — every 3 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/loans.rs`.
//! Entities can borrow gold with interest. Interest accrues over time;
//! defaulting damages relations and credit rating. Expressed as
//! TransferGold deltas for payments and UpdateTreasury for interest.
//!
//!   (id, borrower_entity_id, lender_settlement_id, principal, interest_rate,
//!    amount_owed, due_tick, payments_made)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// How often loans tick.
const LOAN_INTERVAL: u64 = 3;

/// Interest rate per tick (applied to outstanding balance).
const INTEREST_RATE_PER_TICK: f32 = 0.001;

/// Auto-payment fraction of gold above a comfort threshold.
const AUTO_PAYMENT_FRACTION: f32 = 0.05;

/// Gold threshold: NPCs with more than this try to repay.
const REPAYMENT_GOLD_THRESHOLD: f32 = 50.0;

/// Compute loan system deltas.
///
/// Without explicit loan records, this system models debt service as
/// a flow from NPC entities to their home settlement: NPCs with gold
/// above a threshold transfer a fraction to the settlement treasury
/// (proxy for loan repayment), and settlements earn interest income.
pub fn compute_loans(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % LOAN_INTERVAL != 0 {
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

        // NPCs with gold above the threshold make a "payment" to their
        // home settlement (proxy for loan repayment + interest).
        if npc.gold > REPAYMENT_GOLD_THRESHOLD {
            let payment = npc.gold * AUTO_PAYMENT_FRACTION;
            let interest_portion = payment * INTEREST_RATE_PER_TICK * LOAN_INTERVAL as f32;

            // NPC pays settlement.
            out.push(WorldDelta::TransferGold {
                from_entity: entity.id,
                to_entity: home_id,
                amount: payment,
            });

            // Settlement treasury grows from interest income.
            out.push(WorldDelta::UpdateTreasury {
                settlement_id: home_id,
                delta: interest_portion,
            });
        }
    }
}
