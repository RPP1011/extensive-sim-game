//! Economy system — every tick.
//!
//! Ported from `crates/headless_campaign/src/systems/economy.rs`.
//! Applies guild passive income, equipment maintenance/upkeep,
//! trade income from controlled settlements, and investment drains
//! as TransferGold/UpdateTreasury deltas.
//!

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
                    settlement_id: settlement.id,
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

    // --- NPC taxation: NPCs with gold pay a small tax to settlement treasury ---
    // This is the primary income source for settlements. No passive income.
    // Progressive tax: rich settlements have diminishing returns to prevent
    // runaway treasury accumulation (positive feedback loop).
    let tax_efficiency = 1.0 / (1.0 + settlement.treasury / 10_000.0);
    let mut tax_income = 0.0f32;
    for entity in entities {
        if !entity.alive { continue; }
        if let Some(npc) = &entity.npc {
            if npc.gold > 1.0 {
                let tax = npc.gold * 0.001 * tax_efficiency; // 0.1% base, reduced by wealth
                tax_income += tax;
                out.push(WorldDelta::TransferGold {
                    from_entity: entity.id,
                    to_entity: settlement_id, // goes to treasury via convention
                    amount: tax,
                });
            }
        }
    }
    if tax_income > 0.0 {
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: tax_income,
        });
    }

    // --- Administrative cost decay: wealthy settlements leak treasury ---
    // Settlements with treasury > 10000 lose 0.1% per tick as admin overhead.
    if settlement.treasury > 10_000.0 {
        let admin_cost = settlement.treasury * 0.001;
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: -admin_cost,
        });
    }

    // --- Equipment maintenance / upkeep (only if treasury positive) ---
    if settlement.treasury > 0.0 {
        compute_economy_maintenance_for_settlement(state, settlement_id, entities, out);
    }

    // --- Supply/demand price update ---
    // Prices inversely proportional to stockpile. Scarce goods cost more.
    // price = base_price / (1 + stockpile / (population * halflife))
    let pop = (settlement.population as f32).max(1.0);
    let halflife = 50.0;
    let base_prices = [1.0, 3.0, 2.0, 2.5, 2.0, 5.0, 10.0, 8.0]; // FOOD..MEDICINE
    let mut new_prices = [0.0f32; 8];
    for i in 0..8 {
        let supply_ticks = settlement.stockpile[i] / (pop * 0.1).max(0.01);
        new_prices[i] = base_prices[i] / (1.0 + supply_ticks / halflife);
        new_prices[i] = new_prices[i].clamp(0.1, 100.0);
    }
    out.push(WorldDelta::UpdatePrices {
        settlement_id: settlement_id,
        prices: new_prices,
    });
}

/// Per-settlement variant for parallel dispatch (maintenance upkeep).
pub fn compute_economy_maintenance_for_settlement(
    _state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    for entity in entities {
        if !entity.alive { continue; }
        let _npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: -MAINTENANCE_PER_NPC,
        });
    }
}

/// How often (in ticks) to run debt repayment.
const DEBT_REPAYMENT_INTERVAL: u64 = 50;

/// Advance debt repayment for all NPCs.
/// Runs every 50 ticks. NPCs with debt > 0 attempt partial repayment from gold.
/// Excessive debt (> 100x income) incurs morale and credit penalties.
/// Full repayment earns a credit history bonus.
pub fn advance_debt(state: &mut WorldState) {
    if state.tick % DEBT_REPAYMENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &mut entity.npc {
            Some(n) => n,
            None => continue,
        };
        if npc.debt <= 0.0 {
            continue;
        }

        // Repayment: 20% of income_rate per cycle, capped at remaining debt.
        let repayment = npc.debt.min(npc.income_rate * 0.2);
        if repayment > 0.0 && npc.gold >= repayment {
            npc.gold -= repayment;
            npc.debt -= repayment;
        }
        // If NPC can't afford the repayment (gold < repayment), skip this cycle.

        // Excessive debt penalty: debt > 100x income_rate.
        if npc.income_rate > 0.0 && npc.debt > npc.income_rate * 100.0 {
            npc.morale = (npc.morale - 5.0).max(0.0);
            npc.credit_history = npc.credit_history.saturating_sub(10);
        }

        // Debt fully repaid: credit history bonus, clear creditor.
        if npc.debt <= 0.0 {
            npc.debt = 0.0;
            npc.credit_history = npc.credit_history.saturating_add(5);
            npc.creditor_id = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn npc_taxation_generates_income() {
        let mut state = WorldState::new(42);
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 100;
        state.settlements.push(s);

        // NPCs with gold generate tax income for the settlement.
        for i in 1..=5 {
            let mut npc = Entity::new_npc(i, (0.0, 0.0));
            npc.npc.as_mut().unwrap().gold = 100.0;
            npc.npc.as_mut().unwrap().home_settlement_id = Some(10);
            state.entities.push(npc);
        }
        state.rebuild_group_index();
        state.rebuild_entity_cache();

        let mut deltas = Vec::new();
        compute_economy(&state, &mut deltas);

        let treasury_delta: f32 = deltas
            .iter()
            .filter_map(|d| match d {
                WorldDelta::UpdateTreasury {
                    settlement_id: 10,
                    delta,
                } => Some(*delta),
                _ => None,
            })
            .sum();
        assert!(
            treasury_delta > 0.0,
            "NPC taxation should generate settlement income"
        );
    }

    #[test]
    fn maintenance_drains_treasury() {
        let mut state = WorldState::new(42);
        let mut settlement = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        settlement.treasury = 100.0; // needs positive treasury for maintenance
        state.settlements.push(settlement);

        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.npc.as_mut().unwrap().home_settlement_id = Some(10);
        state.entities.push(npc);
        state.rebuild_group_index();
        state.rebuild_entity_cache();

        let mut deltas = Vec::new();
        compute_economy(&state, &mut deltas);

        let has_drain = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateTreasury { settlement_id: 10, delta } if *delta < 0.0
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
