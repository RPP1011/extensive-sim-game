#![allow(unused)]
//! Bounty board system — every 10 ticks.
//!
//! Factions and NPCs post bounties on targets. Bounty completion rewards gold
//! via TransferGold deltas. Expiry and auto-completion are tracked through
//! bounty state on WorldState.
//!
//! Ported from `crates/headless_campaign/src/systems/bounties.rs`.
//!
//! NEEDS STATE: `bounty_board: Vec<Bounty>` on WorldState
//! NEEDS STATE: `Bounty { id, poster_faction_id, target_description, target_type,
//!              reward_gold, reward_reputation, region_id, posted_tick, expires_tick,
//!              claimed, claimed_by, completed }`
//! NEEDS STATE: `BountyTarget` enum { MonsterHunt, FactionEnemy, NemesisKill,
//!              BanditClearance, ResourceDelivery, EscortMission }
//! NEEDS STATE: `next_bounty_id: u32` on WorldState
//! NEEDS STATE: `guild_gold: f32` on WorldState (or guild entity treasury)
//! NEEDS STATE: `guild_reputation: f32` on WorldState
//! NEEDS DELTA: PostBounty { bounty_id, description, reward_gold }
//! NEEDS DELTA: CompleteBounty { bounty_id, reward_gold }
//! NEEDS DELTA: ExpireBounty { bounty_id }
//! NEEDS DELTA: UpdateReputation { entity_id, delta }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

/// How often the bounty system ticks (in world sim ticks).
const BOUNTY_INTERVAL: u64 = 10;

/// Maximum active (unclaimed + claimed-but-incomplete) bounties.
const MAX_ACTIVE_BOUNTIES: usize = 6;

/// Minimum active bounties before we try to generate more.
const MIN_ACTIVE_BOUNTIES: usize = 2;

/// Bounty expiry duration in ticks from when it was posted.
const BOUNTY_EXPIRY_TICKS: u64 = 3000;

/// Guild entity ID sentinel (by convention, entity id 0 is the guild).
const GUILD_ENTITY_ID: u32 = 0;

pub fn compute_bounties(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BOUNTY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Auto-complete check: bounties targeting dead monsters ---
    // When monsters die on a grid, nearby NPCs who are bounty-eligible
    // receive gold rewards. We approximate bounty completion by detecting
    // dead/dying monsters and rewarding nearby friendly NPCs.

    for grid in &state.grids {
        let dead_monsters: Vec<&crate::world_sim::state::Entity> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| state.entity(eid))
            .filter(|e| e.kind == EntityKind::Monster && !e.alive)
            .collect();

        if dead_monsters.is_empty() {
            continue;
        }

        let friendlies: Vec<&crate::world_sim::state::Entity> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| state.entity(eid))
            .filter(|e| e.kind == EntityKind::Npc && e.alive && e.team == WorldTeam::Friendly)
            .collect();

        if friendlies.is_empty() {
            continue;
        }

        // Bounty reward: gold per dead monster based on level, split among friendlies
        for monster in &dead_monsters {
            let bounty_gold = 30.0 + monster.level as f32 * 10.0;
            let gold_each = bounty_gold / friendlies.len() as f32;

            for friendly in &friendlies {
                if gold_each > 0.0 {
                    out.push(WorldDelta::TransferGold {
                        from_id: GUILD_ENTITY_ID, // bounty paid from guild
                        to_id: friendly.id,
                        amount: gold_each,
                    });
                }
            }
        }
    }

    // --- High-threat regions generate implicit bounty pressure ---
    // Regions with high threat get fidelity escalation so encounters
    // are resolved at higher fidelity (the "bounty" incentivizes action).
    for region in &state.regions {
        if region.threat_level > 50.0 {
            // Transfer a small bounty reward to nearest settlement treasury
            // as incentive for patrols (represents standing bounty funding)
            if let Some(settlement) = state.settlements.first() {
                let funding = region.threat_level * 0.05;
                out.push(WorldDelta::UpdateTreasury {
                    location_id: settlement.id,
                    delta: funding,
                });
            }
        }
    }
}
