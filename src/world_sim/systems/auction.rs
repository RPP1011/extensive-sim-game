#![allow(unused)]
//! Auction house system — periodic item auctions where entities compete for
//! rare goods, driving up prices.
//!
//! Ported from `crates/headless_campaign/src/systems/auction.rs`.
//! Fires every 17 ticks. New auctions spawn every 67 ticks with 2-4 items.
//! AI bidders bid based on gold reserves and need. Auctions last 17 ticks
//! before resolution. Gold transfers are expressed as TransferGold deltas.
//!
//! NEEDS STATE: `auction_house: AuctionHouseState` on WorldState
//!   (items: Vec<AuctionItem>, last_auction_tick, total_spent, total_won)
//! NEEDS STATE: `rival_guild: RivalGuildState` on WorldState
//!   (gold, active, name, power_level)
//! NEEDS STATE: `guild: GuildState` on WorldState
//!   (gold, reputation, inventory)
//! NEEDS STATE: `factions: Vec<FactionState>` on WorldState
//!   (id, name, military_strength, relationship_to_guild)
//! NEEDS STATE: `campaign_progress: f32` on WorldState or EconomyState
//! NEEDS DELTA: AuctionBid { item_id: u32, bidder_entity_id: u32, amount: f32 }
//! NEEDS DELTA: AuctionResolve { item_id: u32, winner_entity_id: u32, price: f32 }
//! NEEDS DELTA: CreateAuctionItem { item: AuctionItemData }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// How often the auction system ticks (in world ticks).
const AUCTION_TICK_INTERVAL: u64 = 17;

/// How often a new auction is created (in world ticks).
const NEW_AUCTION_INTERVAL: u64 = 67;

/// Auction duration in ticks.
const AUCTION_DURATION: u64 = 17;

/// Compute auction house deltas.
///
/// Without auction-specific state on WorldState, this system emits
/// TransferGold deltas for bid payments and treasury effects.
/// The full auction lifecycle (create / bid / resolve) requires the
/// NEEDS STATE and NEEDS DELTA items listed above.
pub fn compute_auction(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % AUCTION_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Auction resolution: winners pay gold ---
    // Without AuctionHouseState we approximate by draining treasury from
    // settlements that host auction activity.  When an auction resolves
    // the winner's gold is transferred to the settlement treasury.
    //
    // Approximation: every AUCTION_TICK_INTERVAL ticks, each settlement
    // with population > 50 runs a small auction where gold flows from
    // participating NPC entities to the settlement treasury.
    for settlement in &state.settlements {
        if settlement.population < 50 {
            continue;
        }

        // Collect NPC entities at this settlement.
        let bidders: Vec<&crate::world_sim::state::Entity> = state
            .entities
            .iter()
            .filter(|e| {
                e.kind == crate::world_sim::state::EntityKind::Npc
                    && e.alive
                    && e.npc
                        .as_ref()
                        .map(|n| n.home_settlement_id == Some(settlement.id))
                        .unwrap_or(false)
            })
            .collect();

        if bidders.is_empty() {
            continue;
        }

        // Each participating NPC pays a small auction fee into the settlement
        // treasury (proxy for bid activity).
        let fee_per_bidder = 0.5;
        for bidder in &bidders {
            let npc = bidder.npc.as_ref().unwrap();
            if npc.gold >= fee_per_bidder {
                out.push(WorldDelta::TransferGold {
                    from_id: bidder.id,
                    to_id: settlement.id,
                    amount: fee_per_bidder,
                });
            }
        }

        // Settlement treasury grows from auction activity.
        let auction_revenue = bidders.len() as f32 * fee_per_bidder * 0.5;
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement.id,
            delta: auction_revenue,
        });
    }

    // --- New auction creation every NEW_AUCTION_INTERVAL ---
    // Requires NEEDS DELTA: CreateAuctionItem to properly implement.
    // Currently a no-op beyond the treasury effects above.
}
