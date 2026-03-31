//! Goal evaluation system — manages NPC goal stacks.
//!
//! Runs post-apply (needs `&mut WorldState`). Evaluates each NPC's needs,
//! emotions, and current situation to push/pop goals on their goal stack.
//! The goal stack drives behavior: the top goal (highest priority) determines
//! what the NPC does each tick.
//!
//! This replaces the 100-tick decision interval with continuous goal management.
//! Goals persist across ticks — an NPC working on a goal continues until it
//! completes, gets interrupted by a higher-priority goal, or becomes irrelevant.
//!
//! Cadence: every 10 ticks (same as agent_inner).

use crate::world_sim::state::*;
use crate::world_sim::commodity;
use crate::world_sim::NUM_COMMODITIES;

const GOAL_EVAL_INTERVAL: u64 = 10;

/// Snapshot of settlement economic data needed during evaluation.
struct SettlementEcon {
    id: u32,
    prices: [f32; NUM_COMMODITIES],
}

/// Evaluate and update goal stacks for all NPCs.
/// Called post-apply from runtime.rs.
pub fn evaluate_goals(state: &mut WorldState) {
    if state.tick % GOAL_EVAL_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // Collect settlement data for goal evaluation.
    let settlement_data: Vec<(u32, f32, f32, (f32, f32))> = state.settlements.iter()
        .map(|s| (s.id, s.threat_level, s.stockpile[commodity::FOOD], s.pos))
        .collect();

    // Collect settlement economic data for economic option evaluation.
    let settlement_econ: Vec<SettlementEcon> = state.settlements.iter()
        .map(|s| SettlementEcon { id: s.id, prices: s.prices })
        .collect();

    // Deferred service contracts to post after the entity loop.
    let mut deferred_contracts: Vec<(u32, ServiceContract)> = Vec::new();

    // Pre-compute food building positions per settlement: (settlement_id, pos).
    let food_buildings: Vec<(u32, (f32, f32))> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .filter_map(|e| {
            let bd = e.building.as_ref()?;
            if bd.construction_progress < 1.0 { return None; }
            if !matches!(bd.building_type, BuildingType::Inn | BuildingType::Market) { return None; }
            Some((bd.settlement_id?, e.pos))
        })
        .collect();

    // Pre-compute work building positions: (building_entity_id, pos).
    let work_buildings: Vec<(u32, (f32, f32))> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .map(|e| (e.id, e.pos))
        .collect();

    // Pre-compute home building positions.
    let home_buildings: Vec<(u32, (f32, f32))> = work_buildings.clone();

    // Pre-compute social building positions per settlement: (settlement_id, pos).
    let social_buildings: Vec<(u32, (f32, f32))> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .filter_map(|e| {
            let bd = e.building.as_ref()?;
            if bd.construction_progress < 1.0 { return None; }
            if !matches!(bd.building_type,
                BuildingType::Inn | BuildingType::Temple
                | BuildingType::Market | BuildingType::GuildHall) { return None; }
            Some((bd.settlement_id?, e.pos))
        })
        .collect();

    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        // Skip NPCs in adventuring parties — their goals are managed by the party system.
        if npc.party_id.is_some() { continue; }

        let sid = npc.home_settlement_id;
        let (threat, food_available, _settlement_pos) = sid
            .and_then(|id| settlement_data.iter().find(|(sid, _, _, _)| *sid == id))
            .map(|(_, t, f, p)| (*t, *f, *p))
            .unwrap_or((0.0, 100.0, entity.pos));

        // --- Critical interrupts: push high-priority goals ---

        // Flee: safety critically low + immediate danger.
        if npc.needs.safety < 10.0 && threat > 0.6 && !npc.goal_stack.has(&GoalKind::Flee { from: (0.0, 0.0) }) {
            let flee_from = _settlement_pos;
            npc.goal_stack.push(
                Goal::new(GoalKind::Flee { from: flee_from }, goal_priority::FLEE, tick)
            );
        }

        // Eat (critical): hunger very low.
        if npc.needs.hunger < 15.0 && food_available > 0.0
            && !npc.goal_stack.has(&GoalKind::Eat)
        {
            let food_pos = food_buildings.iter()
                .find(|(id, _)| *id == sid.unwrap_or(u32::MAX))
                .map(|(_, p)| *p);
            let mut goal = Goal::new(GoalKind::Eat, goal_priority::EAT_CRITICAL, tick);
            goal.target_pos = food_pos;
            npc.goal_stack.push(goal);
        }
        // Eat (moderate): hunger below threshold.
        else if npc.needs.hunger < 30.0 && food_available > 0.0
            && !npc.goal_stack.has(&GoalKind::Eat)
        {
            let food_pos = food_buildings.iter()
                .find(|(id, _)| *id == sid.unwrap_or(u32::MAX))
                .map(|(_, p)| *p);
            let mut goal = Goal::new(GoalKind::Eat, goal_priority::EAT, tick);
            goal.target_pos = food_pos;
            npc.goal_stack.push(goal);
        }

        // Fight: settlement under serious threat and NPC is a combatant.
        if threat > 0.5
            && npc.needs.safety < 50.0
            && !npc.goal_stack.has(&GoalKind::Fight)
            && matches!(npc.economic_intent, EconomicIntent::Idle)
        {
            npc.goal_stack.push(
                Goal::new(GoalKind::Fight, goal_priority::FIGHT, tick)
            );
        }

        // --- Background goals: push if stack has room ---

        // Work: if NPC has a work assignment and no work goal.
        if let Some(wbid) = npc.work_building_id {
            if !npc.goal_stack.has(&GoalKind::Work) && npc.goal_stack.goals.len() < 3 {
                let work_pos = work_buildings.iter()
                    .find(|(id, _)| *id == wbid)
                    .map(|(_, p)| *p);
                let mut goal = Goal::new(GoalKind::Work, goal_priority::WORK, tick);
                goal.target_pos = work_pos;
                goal.target_entity = Some(wbid);
                npc.goal_stack.push(goal);
            }
        }

        // Socialize: social need below moderate threshold.
        if npc.needs.social < 45.0
            && !npc.goal_stack.has(&GoalKind::Socialize)
            && npc.goal_stack.goals.len() < 4
        {
            let social_pos = social_buildings.iter()
                .find(|(id, _)| *id == sid.unwrap_or(u32::MAX))
                .map(|(_, p)| *p);
            let mut goal = Goal::new(GoalKind::Socialize, goal_priority::SOCIALIZE, tick);
            goal.target_pos = social_pos;
            npc.goal_stack.push(goal);
        }

        // Rest: shelter need low.
        if let Some(hbid) = npc.home_building_id {
            if npc.needs.shelter < 20.0
                && !npc.goal_stack.has(&GoalKind::Rest)
                && npc.goal_stack.goals.len() < 4
            {
                let home_pos = home_buildings.iter()
                    .find(|(id, _)| *id == hbid)
                    .map(|(_, p)| *p);
                let mut goal = Goal::new(GoalKind::Rest, goal_priority::REST, tick);
                goal.target_pos = home_pos;
                goal.target_entity = Some(hbid);
                npc.goal_stack.push(goal);
            }
        }

        // --- Economic decision evaluation ---
        // Evaluate DIY / Hire / Borrow / Postpone for shelter and food needs.
        if let Some(econ) = sid
            .and_then(|id| settlement_econ.iter().find(|se| se.id == id))
        {
            evaluate_economic_options(
                npc, entity.id, econ.id, &econ.prices, tick,
                &mut deferred_contracts,
            );
        }

        // --- Goal completion/cleanup ---

        // Remove stale Eat goals if hunger is satisfied.
        if npc.needs.hunger > 60.0 {
            npc.goal_stack.remove_kind(&GoalKind::Eat);
        }

        // Remove Fight goals if threat subsided.
        if threat < 0.2 {
            npc.goal_stack.remove_kind(&GoalKind::Fight);
        }

        // Remove Flee goals if safety recovered.
        if npc.needs.safety > 50.0 {
            npc.goal_stack.goals.retain(|g| !matches!(g.kind, GoalKind::Flee { .. }));
        }

        // Remove Socialize if social need is met.
        if npc.needs.social > 60.0 {
            npc.goal_stack.remove_kind(&GoalKind::Socialize);
        }

        // Remove Rest if shelter need is met.
        if npc.needs.shelter > 70.0 {
            npc.goal_stack.remove_kind(&GoalKind::Rest);
        }

        // Remove goals that have been active too long (stale, stuck).
        let stale_threshold = 500;
        npc.goal_stack.goals.retain(|g| {
            tick.saturating_sub(g.started_tick) < stale_threshold
                || matches!(g.kind, GoalKind::Work | GoalKind::Trade { .. } | GoalKind::Quest { .. })
        });

        // --- Sync goal stack -> EconomicIntent (backward compat) ---
        match npc.goal_stack.current_kind() {
            GoalKind::Idle => { /* leave current intent */ }
            GoalKind::Work => {
                npc.economic_intent = EconomicIntent::Produce;
            }
            GoalKind::Trade { destination_settlement_id } => {
                npc.economic_intent = EconomicIntent::Trade {
                    destination_settlement_id: *destination_settlement_id,
                };
            }
            GoalKind::Fight => {
                npc.economic_intent = EconomicIntent::Idle;
            }
            GoalKind::Quest { quest_id, destination } => {
                npc.economic_intent = EconomicIntent::Adventuring {
                    quest_id: *quest_id,
                    destination: *destination,
                };
            }
            GoalKind::Relocate { destination_settlement_id } => {
                let dest_pos = state.settlements.iter()
                    .find(|s| s.id == *destination_settlement_id)
                    .map(|s| s.pos)
                    .unwrap_or(entity.pos);
                npc.economic_intent = EconomicIntent::Travel { destination: dest_pos };
            }
            _ => {}
        }
    }

    // Apply deferred service contracts to settlements.
    for (settlement_id, contract) in deferred_contracts {
        if let Some(settlement) = state.settlements.iter_mut().find(|s| s.id == settlement_id) {
            settlement.service_contracts.push(contract);
        }
    }
}

// ---------------------------------------------------------------------------
// Economic decision evaluation -- DIY / Hire / Borrow / Postpone
// ---------------------------------------------------------------------------

/// The four possible economic decisions.
#[derive(Debug, Clone, Copy, PartialEq)]
enum EconDecision {
    Diy,
    Hire,
    Borrow,
    Postpone,
}

/// Create a service contract with urgency-based bidding deadline.
fn make_contract(
    requester_id: u32,
    service: ServiceType,
    gold_amount: f32,
    tick: u64,
    urgency: f32,
) -> ServiceContract {
    let deadline_offset: u64 = if urgency > 0.85 {
        5
    } else if urgency > 0.7 {
        15
    } else if urgency > 0.5 {
        30
    } else {
        100
    };
    ServiceContract {
        requester_id,
        service,
        max_payment: Payment::gold_only(gold_amount),
        payment: 0.0,
        provider_id: None,
        posted_tick: tick,
        completed: false,
        bidding_deadline: tick + deadline_offset,
        bids: Vec::new(),
        accepted_bid: None,
    }
}

/// Compute the credit limit for an NPC based on income and credit history.
fn credit_limit(npc: &NpcData) -> f32 {
    npc.income_rate * 50.0 * (npc.credit_history as f32 / 255.0)
}

/// Pick the best economic option by comparing utilities.
///
/// - `urgency`: 0.0-1.0, how badly the need is unmet
/// - `resource_cost`: gold cost of materials (for DIY)
/// - `labor_ticks`: estimated ticks of labor (for DIY)
/// - `income`: NPC income_rate (gold per tick, floored)
/// - `hire_cost`: total gold cost to hire someone
fn pick_best_option(
    npc: &NpcData,
    urgency: f32,
    resource_cost: f32,
    labor_ticks: f32,
    income: f32,
    hire_cost: f32,
) -> EconDecision {
    let urgency = urgency.max(0.01);

    // 1. DIY utility: can I do this myself?
    let diy_cost = resource_cost + labor_ticks * income;
    let diy_utility = if diy_cost > 0.0 { urgency / diy_cost } else { urgency };

    // 2. Hire utility: can I pay someone?
    let hire_utility = if npc.gold >= hire_cost && hire_cost > 0.0 {
        urgency / hire_cost
    } else {
        0.0
    };

    // 3. Borrow utility: can I get credit?
    let available_credit = credit_limit(npc) - npc.debt;
    let shortfall = (hire_cost - npc.gold).max(0.0);
    let borrow_utility = if available_credit >= shortfall && hire_cost > 0.0 {
        urgency / (hire_cost * 1.2)
    } else {
        0.0
    };

    // 4. Postpone utility: can I tolerate this for now?
    let need_value = (1.0 - urgency) * 100.0;
    let suffering_rate = ((100.0 - need_value) * 0.01).max(0.001);
    let ticks_until_affordable = if income > 0.0 {
        (hire_cost / income).max(1.0)
    } else {
        10000.0
    };
    let postpone_utility = 1.0 / (suffering_rate * ticks_until_affordable);

    let options = [
        (EconDecision::Diy, diy_utility),
        (EconDecision::Hire, hire_utility),
        (EconDecision::Borrow, borrow_utility),
        (EconDecision::Postpone, postpone_utility),
    ];
    options.into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(d, _)| d)
        .unwrap_or(EconDecision::Postpone)
}

/// Evaluate economic options for unmet needs: shelter and food.
///
/// For each unmet need, scores four strategies:
/// 1. **DIY** -- gather resources and do it yourself (opportunity cost = labor x income_rate)
/// 2. **Hire** -- pay someone at market price
/// 3. **Borrow** -- take on debt and then hire (interest premium)
/// 4. **Postpone** -- wait until you can afford it naturally
///
/// The highest-utility option drives goal generation.
fn evaluate_economic_options(
    npc: &mut NpcData,
    entity_id: u32,
    settlement_id: u32,
    prices: &[f32; NUM_COMMODITIES],
    tick: u64,
    deferred_contracts: &mut Vec<(u32, ServiceContract)>,
) {
    // Don't evaluate if NPC is in a party, fighting, or fleeing.
    if npc.party_id.is_some() { return; }
    if npc.goal_stack.goals.iter().any(|g| {
        matches!(g.kind, GoalKind::Fight | GoalKind::Flee { .. })
    }) {
        return;
    }

    // Income rate floor to avoid division by zero.
    let income = npc.income_rate.max(0.01);

    // --- Shelter need: build a house ---
    if npc.needs.shelter < 80.0 && npc.home_building_id.is_none() {
        let has_build = npc.goal_stack.goals.iter().any(|g| {
            matches!(g.kind, GoalKind::Build { .. })
        });
        if !has_build {
            let shelter_urgency = (100.0 - npc.needs.shelter) / 100.0;
            let (wood_cost, iron_cost) = BuildingType::House.build_cost();
            let resource_gold_cost = wood_cost * prices[commodity::WOOD]
                + iron_cost * prices[commodity::IRON];
            let labor_ticks: f32 = 200.0;
            let hire_cost = resource_gold_cost + labor_ticks * income;

            let decision = pick_best_option(
                npc, shelter_urgency, resource_gold_cost,
                labor_ticks, income, hire_cost,
            );

            match decision {
                EconDecision::Diy => {
                    // Push a single Build goal — GOAP will decompose it into
                    // Gather(WOOD) → Gather(IRON) → PlaceBuilding → Construct
                    npc.goal_stack.push(Goal::new(
                        GoalKind::Build { building_id: 0 },
                        goal_priority::BUILD, tick,
                    ));
                }
                EconDecision::Hire => {
                    if npc.gold >= hire_cost {
                        npc.gold -= hire_cost;
                        deferred_contracts.push((settlement_id, make_contract(
                            entity_id, ServiceType::Build(BuildingType::House),
                            hire_cost, tick, shelter_urgency,
                        )));
                    }
                }
                EconDecision::Borrow => {
                    let borrow_amount = (hire_cost - npc.gold).max(0.0);
                    let available_credit = credit_limit(npc) - npc.debt;
                    if available_credit >= borrow_amount {
                        npc.debt += borrow_amount;
                        npc.creditor_id = Some(settlement_id);
                        npc.gold += borrow_amount;
                        npc.gold -= hire_cost;
                        deferred_contracts.push((settlement_id, make_contract(
                            entity_id, ServiceType::Build(BuildingType::House),
                            hire_cost, tick, shelter_urgency,
                        )));
                    }
                }
                EconDecision::Postpone => {}
            }
        }
    }

    // --- Food need: buy food vs farm ---
    if npc.needs.hunger < 60.0 {
        let has_eat = npc.goal_stack.has(&GoalKind::Eat);
        let has_gather_food = npc.goal_stack.goals.iter().any(|g| {
            matches!(g.kind, GoalKind::Gather { commodity, .. } if commodity == commodity::FOOD as u8)
        });
        if !has_eat && !has_gather_food {
            let hunger_urgency = (100.0 - npc.needs.hunger) / 100.0;
            let food_price = prices[commodity::FOOD];
            let food_units_needed: f32 = 5.0;
            let buy_cost = food_units_needed * food_price;
            let farm_labor_ticks: f32 = 100.0;

            let decision = pick_best_option(
                npc, hunger_urgency, 0.0,
                farm_labor_ticks, income, buy_cost,
            );

            match decision {
                EconDecision::Diy => {
                    let mut g = Goal::new(
                        GoalKind::Gather {
                            commodity: commodity::FOOD as u8,
                            amount: food_units_needed,
                        },
                        goal_priority::WORK, tick,
                    );
                    g.target_entity = Some(settlement_id);
                    npc.goal_stack.push(g);
                }
                EconDecision::Hire => {
                    if npc.gold >= buy_cost {
                        npc.gold -= buy_cost;
                        deferred_contracts.push((settlement_id, make_contract(
                            entity_id,
                            ServiceType::Gather(commodity::FOOD as u8, food_units_needed),
                            buy_cost, tick, hunger_urgency,
                        )));
                    }
                }
                EconDecision::Borrow => {
                    let borrow_amount = (buy_cost - npc.gold).max(0.0);
                    let available_credit = credit_limit(npc) - npc.debt;
                    if available_credit >= borrow_amount {
                        npc.debt += borrow_amount;
                        npc.creditor_id = Some(settlement_id);
                        npc.gold += borrow_amount;
                        npc.gold -= buy_cost;
                        deferred_contracts.push((settlement_id, make_contract(
                            entity_id,
                            ServiceType::Gather(commodity::FOOD as u8, food_units_needed),
                            buy_cost, tick, hunger_urgency,
                        )));
                    }
                }
                EconDecision::Postpone => {}
            }
        }
    }
}
