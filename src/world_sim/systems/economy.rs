#![allow(unused)]
//! Economy system — every tick.
//!
//! Ported from `crates/headless_campaign/src/systems/economy.rs`.
//! Applies guild passive income, equipment maintenance/upkeep,
//! trade income from controlled settlements, and investment drains
//! as TransferGold/UpdateTreasury deltas.
//!
//! NEEDS STATE: `tax_rate: f32` on SettlementState (for trade income scaling)
//! NEEDS STATE: `infrastructure_level: f32` on SettlementState (for investment bonus)
//! NEEDS STATE: `unrest: f32` on SettlementState (for trade disruption)
//! NEEDS STATE: `faction_id: Option<u32>` on SettlementState (for ownership check)
//! NEEDS STATE: `guild_settlement_id: Option<u32>` on EconomyState (passive income source)
//! NEEDS STATE: `threat_level: f32` on RegionState (for threat reward bonus)
//! NEEDS DELTA: ApplyFatigueAndMorale { entity_id, fatigue_delta, morale_delta }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

/// Passive gold income per tick for each guild-controlled settlement.
const PASSIVE_INCOME_PER_TICK: f32 = 0.5;

/// Equipment maintenance cost per NPC per tick.
const MAINTENANCE_PER_NPC: f32 = 0.05;

/// Trade income per settlement population unit per tick (scaled by stability).
const TRADE_INCOME_PER_POP: f32 = 0.001;

/// Threat reward bonus: fraction of region threat_level added as gold per tick.
const THREAT_REWARD_RATE: f32 = 0.01;

/// Guild entity ID sentinel used as the "guild treasury" endpoint for TransferGold.
/// By convention, entity id 0 is the guild.
const GUILD_ENTITY_ID: u32 = 0;

pub fn compute_economy(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_economy_for_settlement(state, settlement.id, &state.entities[range], out);
    }

    // --- Threat reward bonus: regions with high threat yield bonus gold to nearest settlement ---
    for region in &state.regions {
        if region.threat_level > 10.0 {
            let bonus = region.threat_level * THREAT_REWARD_RATE;
            // Find nearest settlement to this region (by faction or first match)
            if let Some(settlement) = state.settlements.first() {
                out.push(WorldDelta::UpdateTreasury {
                    location_id: settlement.id,
                    delta: bonus,
                });
            }
        }
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_economy_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // --- Passive income: settlement produces treasury ---
    let income = PASSIVE_INCOME_PER_TICK * (settlement.population as f32).max(1.0).sqrt();
    if income > 0.0 {
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement_id,
            delta: income,
        });
    }

    // --- Equipment maintenance / upkeep ---
    compute_economy_maintenance_for_settlement(state, settlement_id, entities, out);

    // --- Trade income: settlement with positive stockpile generates gold ---
    let avg_stockpile: f32 =
        settlement.stockpile.iter().sum::<f32>() / settlement.stockpile.len() as f32;
    let stability = (avg_stockpile / 10.0).clamp(0.0, 1.0);
    let trade_income = settlement.population as f32 * TRADE_INCOME_PER_POP * stability;
    if trade_income > 0.0 {
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement_id,
            delta: trade_income,
        });
    }
}

/// Per-settlement variant for parallel dispatch (maintenance upkeep).
pub fn compute_economy_maintenance_for_settlement(
    _state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    for entity in entities {
        if entity.kind != EntityKind::Npc || !entity.alive {
            continue;
        }
        let _npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement_id,
            delta: -MAINTENANCE_PER_NPC,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn passive_income_scales_with_population() {
        let mut state = WorldState::new(42);
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 100;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_economy(&state, &mut deltas);

        let treasury_delta: f32 = deltas
            .iter()
            .filter_map(|d| match d {
                WorldDelta::UpdateTreasury {
                    location_id: 10,
                    delta,
                } => Some(*delta),
                _ => None,
            })
            .sum();
        assert!(
            treasury_delta > 0.0,
            "settlement should earn passive income"
        );
    }

    #[test]
    fn maintenance_drains_treasury() {
        let mut state = WorldState::new(42);
        state
            .settlements
            .push(SettlementState::new(10, "Town".into(), (0.0, 0.0)));

        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.npc.as_mut().unwrap().home_settlement_id = Some(10);
        state.entities.push(npc);
        state.rebuild_group_index();

        let mut deltas = Vec::new();
        compute_economy(&state, &mut deltas);

        let has_drain = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateTreasury { location_id: 10, delta } if *delta < 0.0
            )
        });
        assert!(has_drain, "NPC upkeep should drain settlement treasury");
    }

    #[test]
    fn empty_state_no_deltas() {
        let state = WorldState::new(42);
        let mut deltas = Vec::new();
        compute_economy(&state, &mut deltas);
        assert!(deltas.is_empty());
    }
}
