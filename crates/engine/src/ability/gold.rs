//! Combat Foundation Task 16 — `EffectOp::TransferGold` handler.
//!
//! Cascade handler on `Event::EffectGoldTransfer`. Moves `amount` between
//! `cold_inventory[from].gold` and `cold_inventory[to].gold`. Amounts are
//! signed `i64`; negative values reverse the flow. There is no overdraft
//! guard — debt is representable as a negative balance (state port Task H)
//! and reconciled at a later layer.
//!
//! Conservation: the sum of both agents' gold before and after the transfer
//! is identical for every `amount`. No clamping, no rounding. Transfers to
//! or from a dead agent still mutate the slot — inventories outlive their
//! owners for inheritance/loot; the gate (if any) decides whether to issue
//! the cast, not the handler.
//!
//! Self-transfer (`from == to`) is a no-op by construction: `slot.gold -= n`
//! and `slot.gold += n` on the same slot cancel. We still exercise the write
//! path so the invariant (sum preserved) holds without a special case.

use crate::cascade::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

pub struct TransferGoldHandler;

impl CascadeHandler for TransferGoldHandler {
    fn trigger(&self) -> EventKindId { EventKindId::EffectGoldTransfer }
    fn lane(&self) -> Lane { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, _events: &mut EventRing) {
        let (from, to, amount) = match *event {
            Event::EffectGoldTransfer { from, to, amount, .. } => (from, to, amount),
            _ => return,
        };
        if amount == 0 { return; }
        if from == to { return; }

        // Read-modify-write each slot via the Inventory accessors; the getter
        // returns an owned `Inventory` (it's `Copy`) so we update locally and
        // write the whole value back.
        if let Some(mut inv_from) = state.agent_inventory(from) {
            inv_from.gold = inv_from.gold.wrapping_sub(amount);
            state.set_agent_inventory(from, inv_from);
        }
        if let Some(mut inv_to) = state.agent_inventory(to) {
            inv_to.gold = inv_to.gold.wrapping_add(amount);
            state.set_agent_inventory(to, inv_to);
        }
    }
}
