//! Guild contracts / commissions — fires every 10 ticks (compute phase)
//! and every 20 ticks (advance phase for contract lifecycle).
//!
//! The compute phase emits treasury deltas for approximate contract flow.
//! The advance phase (post-apply) manages the full contract lifecycle with
//! a **bidding system**:
//!   1. Expire old contracts past their bidding deadline with no bids
//!   2. **Bidding phase**: eligible idle NPCs submit bids on open contracts
//!   3. **Resolution phase**: requester picks the best bid (price/skill/credit score)
//!   4. Accepted contracts: check if provider completed the service
//!   5. Completed contracts: transfer payment from requester to provider
//!
//! Bidding window is urgency-driven — each contract's `bidding_deadline` is
//! set by the poster based on their need urgency (shorter = more urgent).
//!
//! Original: `crates/headless_campaign/src/systems/contracts.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::{entity_hash_f32, ContractBid, EntityKind, WorkState, ServiceType, tags};
use crate::world_sim::commodity;

/// Contract tick interval (compute phase).
const CONTRACT_TICK_INTERVAL: u64 = 10;

/// Contract refresh interval.
const CONTRACT_REFRESH_INTERVAL: u64 = 33;

/// Reward for contract completion (distributed to settlement).
const CONTRACT_REWARD: f32 = 25.0;

/// Penalty for contract failure.
const CONTRACT_PENALTY: f32 = 10.0;

/// Advance phase tick interval.
const ADVANCE_TICK_INTERVAL: u64 = 20;

/// Estimated labor ticks for opportunity cost calculation.
const ESTIMATED_LABOR_TICKS: f32 = 25.0;

/// Bid scoring weights.
const WEIGHT_PRICE: f32 = 0.60;
const WEIGHT_SKILL: f32 = 0.25;
const WEIGHT_CREDIT: f32 = 0.15;


pub fn compute_contracts(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CONTRACT_TICK_INTERVAL != 0 {
        return;
    }

    // Without contract tracking state, we approximate:
    // Settlements periodically receive contract rewards or penalties based
    // on their overall health (treasury, population).

    for settlement in &state.settlements {
        let roll = entity_hash_f32(settlement.id, state.tick, 0xC0EEACB);

        // Well-functioning settlements complete contracts
        if settlement.treasury > 30.0 && settlement.population > 50 && roll < 0.08 {
            out.push(WorldDelta::UpdateTreasury {
                settlement_id: settlement.id,
                delta: CONTRACT_REWARD,
            });
        }

        // Struggling settlements fail contracts (don't pile on if already deep in debt)
        if settlement.treasury < 10.0 && settlement.treasury > -100.0 && roll > 0.95 {
            out.push(WorldDelta::UpdateTreasury {
                settlement_id: settlement.id,
                delta: -CONTRACT_PENALTY,
            });
        }
    }
}

/// Resolve the required skill tag for a service type.
fn skill_tag_for_service(service: &ServiceType) -> u32 {
    match service {
        ServiceType::Build(_) => tags::CONSTRUCTION,
        ServiceType::Gather(commodity_idx, _) => {
            match *commodity_idx as usize {
                commodity::FOOD => tags::FARMING,
                commodity::WOOD => tags::WOODWORK,
                commodity::IRON => tags::MINING,
                commodity::HERBS => tags::HERBALISM,
                _ => tags::LABOR,
            }
        }
        ServiceType::Craft => tags::CRAFTING,
        ServiceType::Heal => tags::MEDICINE,
        ServiceType::Guard(_) => tags::COMBAT,
        ServiceType::Haul(_, _, _) => tags::LABOR,
        ServiceType::Teach(_) => tags::TEACHING,
        ServiceType::Barter { .. } => tags::TRADE,
    }
}

/// Score a bid for the requester's selection. Higher = better.
///
/// Formula: `0.6 * (1 - bid/max_value) + 0.25 * (skill/100) + 0.15 * (credit/255)`
///
/// `max_value` is the gold-equivalent estimated value of the contract's max payment.
fn score_bid(bid: &ContractBid, max_value: f32) -> f32 {
    let price_score = if max_value > 0.0 {
        1.0 - (bid.bid_amount / max_value)
    } else {
        0.0
    };
    let skill_score = bid.skill_value / 100.0;
    let credit_score = bid.credit_history as f32 / 255.0;

    WEIGHT_PRICE * price_score + WEIGHT_SKILL * skill_score + WEIGHT_CREDIT * credit_score
}

/// Advance contract lifecycle — called post-apply from runtime.
///
/// Runs every 20 ticks. For each settlement:
///   1. Remove expired contracts (past bidding_deadline with no bids)
///   2. **Bidding phase** (tick < bidding_deadline): eligible NPCs submit bids
///   3. **Resolution phase** (tick >= bidding_deadline): pick best bid by score
///   4. Check accepted contracts for completion
///   5. Transfer payment from requester to provider on completion
pub fn advance_contracts(state: &mut WorldState) {
    if state.tick % ADVANCE_TICK_INTERVAL != 0 {
        return;
    }

    let tick = state.tick;
    let settlement_count = state.settlements.len();

    for si in 0..settlement_count {
        let contract_count = state.settlements[si].service_contracts.len();
        if contract_count == 0 {
            continue;
        }

        let settlement_id = state.settlements[si].id;

        // ---------------------------------------------------------------
        // Step 1: Identify contracts past deadline with no bids → expire
        // ---------------------------------------------------------------
        let mut expired_indices: Vec<usize> = Vec::new();
        for ci in 0..state.settlements[si].service_contracts.len() {
            let contract = &state.settlements[si].service_contracts[ci];
            if contract.provider_id.is_none()
                && !contract.completed
                && contract.accepted_bid.is_none()
                && tick >= contract.bidding_deadline
                && contract.bids.is_empty()
            {
                expired_indices.push(ci);
            }
        }

        // ---------------------------------------------------------------
        // Step 2: Bidding phase — collect bids from eligible idle NPCs
        // ---------------------------------------------------------------
        // Gather (contract_idx, skill_tag) for contracts still in bidding window.
        let mut bidding_open: Vec<(usize, u32)> = Vec::new();
        for ci in 0..state.settlements[si].service_contracts.len() {
            let contract = &state.settlements[si].service_contracts[ci];
            // Contract is open for bids: no provider yet, not completed,
            // not expired, and still before the deadline.
            if contract.provider_id.is_some() || contract.completed {
                continue;
            }
            if expired_indices.contains(&ci) {
                continue;
            }
            if tick >= contract.bidding_deadline {
                // Past deadline — will be resolved in step 3, not bidding.
                continue;
            }
            let skill_tag = skill_tag_for_service(&contract.service);
            bidding_open.push((ci, skill_tag));
        }

        // For each open contract, eligible NPCs submit bids.
        for &(ci, skill_tag) in &bidding_open {
            let max_value = state.settlements[si].service_contracts[ci].max_payment.estimated_value();

            // Collect bids from eligible NPCs (can't borrow settlement mutably
            // while iterating entities, so collect first, then push).
            let mut new_bids: Vec<ContractBid> = Vec::new();

            for ei in 0..state.entities.len() {
                let entity = &state.entities[ei];
                if !entity.alive || entity.kind != EntityKind::Npc {
                    continue;
                }
                let npc = match &entity.npc {
                    Some(n) => n,
                    None => continue,
                };
                if npc.home_settlement_id != Some(settlement_id) {
                    continue;
                }
                if !matches!(npc.work_state, WorkState::Idle) {
                    continue;
                }

                // Check if already providing another contract or already bid on this one.
                let already_busy = state.settlements[si].service_contracts.iter()
                    .any(|c| c.provider_id == Some(entity.id) && !c.completed);
                if already_busy {
                    continue;
                }

                let already_bid = state.settlements[si].service_contracts[ci].bids.iter()
                    .any(|b| b.bidder_id == entity.id);
                if already_bid {
                    continue;
                }

                let skill_value = npc.behavior_value(skill_tag);
                let credit_history = npc.credit_history;

                // Bid amount = opportunity cost + material cost estimate.
                // opportunity_cost = income_rate * estimated_labor_ticks
                // material_cost approximated as 0 for service contracts.
                let bid_amount = npc.income_rate * ESTIMATED_LABOR_TICKS;

                // Only bid if our price is below the contract's max payment value.
                if bid_amount >= max_value {
                    continue;
                }

                new_bids.push(ContractBid {
                    bidder_id: entity.id,
                    bid_amount,
                    skill_value,
                    credit_history,
                });
            }

            // Push all new bids onto the contract.
            state.settlements[si].service_contracts[ci].bids.extend(new_bids);
        }

        // ---------------------------------------------------------------
        // Step 3: Resolution phase — contracts past deadline with bids
        // ---------------------------------------------------------------
        // Collect (contract_idx) for contracts ready to resolve.
        let mut resolve_indices: Vec<usize> = Vec::new();
        for ci in 0..state.settlements[si].service_contracts.len() {
            let contract = &state.settlements[si].service_contracts[ci];
            if contract.provider_id.is_some() || contract.completed {
                continue;
            }
            if expired_indices.contains(&ci) {
                continue;
            }
            // Past deadline and has at least one bid.
            if tick >= contract.bidding_deadline && !contract.bids.is_empty() {
                resolve_indices.push(ci);
            }
        }

        for &ci in &resolve_indices {
            let max_value = state.settlements[si].service_contracts[ci].max_payment.estimated_value();

            // Score all bids, pick the best.
            let mut best_idx: Option<usize> = None;
            let mut best_score: f32 = f32::NEG_INFINITY;

            for (bi, bid) in state.settlements[si].service_contracts[ci].bids.iter().enumerate() {
                // Verify bidder is still alive and idle (they may have become
                // busy since bidding).
                let still_eligible = state.entities.iter()
                    .find(|e| e.id == bid.bidder_id && e.alive)
                    .and_then(|e| e.npc.as_ref())
                    .map(|n| matches!(n.work_state, WorkState::Idle))
                    .unwrap_or(false);
                if !still_eligible {
                    continue;
                }

                // Check bidder isn't already assigned to another active contract.
                let already_busy = state.settlements[si].service_contracts.iter()
                    .any(|c| c.provider_id == Some(bid.bidder_id) && !c.completed);
                if already_busy {
                    continue;
                }

                let score = score_bid(bid, max_value);
                if score > best_score {
                    best_score = score;
                    best_idx = Some(bi);
                }
            }

            if let Some(bi) = best_idx {
                let winning_bid = &state.settlements[si].service_contracts[ci].bids[bi];
                let provider_id = winning_bid.bidder_id;
                let agreed_payment = winning_bid.bid_amount;

                state.settlements[si].service_contracts[ci].provider_id = Some(provider_id);
                state.settlements[si].service_contracts[ci].payment = agreed_payment;
                state.settlements[si].service_contracts[ci].accepted_bid = Some(bi);
            }
            // If no eligible bids remain, the contract stays open — it will be
            // expired on the next tick since deadline has passed and we treat
            // "no eligible bids" the same as "no bids" for expiry purposes.
            // Clear bids so the expiry check picks it up next cycle.
            if best_idx.is_none() {
                state.settlements[si].service_contracts[ci].bids.clear();
            }
        }

        // ---------------------------------------------------------------
        // Step 4 & 5: Check accepted contracts for completion and pay
        // ---------------------------------------------------------------
        let mut completed_indices: Vec<usize> = Vec::new();
        for ci in 0..state.settlements[si].service_contracts.len() {
            let contract = &state.settlements[si].service_contracts[ci];
            if contract.completed {
                continue;
            }
            let provider_id = match contract.provider_id {
                Some(id) => id,
                None => continue,
            };

            // Check if provider is idle (finished their work).
            let provider_idle = state.entities.iter()
                .find(|e| e.id == provider_id && e.alive)
                .and_then(|e| e.npc.as_ref())
                .map(|n| matches!(n.work_state, WorkState::Idle))
                .unwrap_or(false);

            // Provider has accepted and is now idle = service complete.
            // (A more sophisticated check would verify the service output,
            //  but idle-after-acceptance is a reasonable proxy.)
            let age = tick.saturating_sub(contract.posted_tick);
            if provider_idle && age > ADVANCE_TICK_INTERVAL {
                completed_indices.push(ci);
            }
        }

        // Process completions: transfer agreed payment requester → provider.
        for &ci in &completed_indices {
            let contract = &state.settlements[si].service_contracts[ci];
            let payment = contract.payment;
            let requester_id = contract.requester_id;
            let provider_id = match contract.provider_id {
                Some(id) => id,
                None => continue,
            };

            // Deduct payment from requester NPC gold.
            let mut paid = 0.0_f32;
            if let Some(req_entity) = state.entities.iter_mut()
                .find(|e| e.id == requester_id && e.alive)
            {
                if let Some(req_npc) = &mut req_entity.npc {
                    let can_pay = req_npc.gold.min(payment);
                    req_npc.gold -= can_pay;
                    paid = can_pay;
                }
            }

            // Pay provider NPC.
            if paid > 0.0 {
                if let Some(prov_entity) = state.entities.iter_mut()
                    .find(|e| e.id == provider_id && e.alive)
                {
                    if let Some(prov_npc) = &mut prov_entity.npc {
                        prov_npc.gold = (prov_npc.gold + paid).min(10000.0);
                        // Successful completion improves credit history.
                        prov_npc.credit_history = prov_npc.credit_history.saturating_add(3);
                        // Update rolling average income rate.
                        prov_npc.income_rate = prov_npc.income_rate * 0.9 + paid * 0.1;
                    }
                }
            }

            // Mark completed.
            state.settlements[si].service_contracts[ci].completed = true;
        }

        // ---------------------------------------------------------------
        // Remove expired and completed contracts
        // ---------------------------------------------------------------
        let mut remove_indices: Vec<usize> = expired_indices;
        for &ci in &completed_indices {
            if !remove_indices.contains(&ci) {
                remove_indices.push(ci);
            }
        }
        remove_indices.sort_unstable();
        remove_indices.dedup();
        for &ci in remove_indices.iter().rev() {
            state.settlements[si].service_contracts.swap_remove(ci);
        }
    }
}
