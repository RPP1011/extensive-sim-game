#![allow(unused)]
//! Trade guilds — merchants at a settlement form guilds that control trade.
//!
//! When 3+ NPCs with trade tags > 50 share a settlement, they form a guild.
//! Guilds:
//! - Set price floors (prevent undercutting)
//! - Fund caravans (treasury → caravan startup gold)
//! - Compete with guilds at other settlements (price wars)
//! - Generate trade/diplomacy/negotiation tags for members
//!
//! Guild data stored on SettlementState. One guild per settlement.
//!
//! Cadence: every 200 ticks.

use crate::world_sim::state::*;

const GUILD_INTERVAL: u64 = 200;
const MIN_MERCHANTS_FOR_GUILD: usize = 3;
const TRADE_TAG_THRESHOLD: f32 = 30.0;
/// Price floor: guild prevents prices from dropping below this fraction of base.
const PRICE_FLOOR_MULT: f32 = 0.7;
/// Gold allocated per caravan funded by the guild.
const CARAVAN_FUNDING: f32 = 15.0;

pub fn advance_trade_guilds(state: &mut WorldState) {
    if state.tick % GUILD_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    for si in 0..state.settlements.len() {
        let sid = state.settlements[si].id;

        // Count merchants at this settlement.
        let merchants: Vec<usize> = state.entities.iter().enumerate()
            .filter(|(_, e)| e.alive && e.kind == EntityKind::Npc)
            .filter(|(_, e)| {
                let npc = match &e.npc { Some(n) => n, None => return false };
                npc.home_settlement_id == Some(sid)
                    && (npc.behavior_value(tags::TRADE) + npc.behavior_value(tags::NEGOTIATION))
                        > TRADE_TAG_THRESHOLD
            })
            .map(|(i, _)| i)
            .collect();

        if merchants.len() < MIN_MERCHANTS_FOR_GUILD { continue; }

        // --- Guild effects ---

        // 1. Price floors: prevent any commodity from dropping below 70% of base (1.0).
        let settlement = &mut state.settlements[si];
        for c in 0..crate::world_sim::NUM_COMMODITIES {
            if settlement.prices[c] < PRICE_FLOOR_MULT {
                settlement.prices[c] = PRICE_FLOOR_MULT;
            }
        }

        // 2. Fund a caravan: if treasury allows, give gold to a merchant for trade.
        if settlement.treasury > CARAVAN_FUNDING * 2.0 {
            // Pick the merchant with the best trade tags.
            let best_merchant = merchants.iter()
                .max_by(|&&a, &&b| {
                    let ta = state.entities[a].npc.as_ref()
                        .map(|n| n.behavior_value(tags::TRADE)).unwrap_or(0.0);
                    let tb = state.entities[b].npc.as_ref()
                        .map(|n| n.behavior_value(tags::TRADE)).unwrap_or(0.0);
                    ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied();

            if let Some(mi) = best_merchant {
                // Fund the caravan.
                state.settlements[si].treasury -= CARAVAN_FUNDING;
                if let Some(npc) = state.entities[mi].npc.as_mut() {
                    npc.gold += CARAVAN_FUNDING;
                }
            }
        }

        // 3. Guild members gain trade/diplomacy tags.
        for &mi in &merchants {
            if let Some(npc) = state.entities[mi].npc.as_mut() {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::TRADE, 1.0);
                    a.add(tags::NEGOTIATION, 0.5);
                    a.add(tags::DIPLOMACY, 0.3);
                    a
                });
            }
        }

        // 4. Guild competition: settlements with guilds push prices of their
        // specialty commodity DOWN at rival settlements (undercutting).
        let specialty = state.settlements[si].specialty;
        let specialty_commodity = match specialty {
            SettlementSpecialty::FarmingVillage => Some(crate::world_sim::commodity::FOOD),
            SettlementSpecialty::MiningTown => Some(crate::world_sim::commodity::IRON),
            SettlementSpecialty::CraftingGuild => Some(crate::world_sim::commodity::EQUIPMENT),
            _ => None,
        };

        if let Some(commodity) = specialty_commodity {
            let guild_power = merchants.len() as f32 * 0.02; // more merchants = stronger guild
            for sj in 0..state.settlements.len() {
                if sj == si { continue; }
                // Undercut: reduce rival's price for this commodity.
                let rival = &mut state.settlements[sj];
                rival.prices[commodity] = (rival.prices[commodity] - guild_power).max(PRICE_FLOOR_MULT);
            }
        }

        // 5. Chronicle for first guild formation (check if already chronicled).
        if tick <= GUILD_INTERVAL * 2 { // only chronicle early formations
            let settlement_name = state.settlements[si].name.clone();
            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Achievement,
                text: format!("A merchants' guild of {} traders has formed in {}.",
                    merchants.len(), settlement_name),
                entity_ids: merchants.iter().map(|&mi| state.entities[mi].id).collect(),
            });
        }
    }
}
