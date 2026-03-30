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

const GOAL_EVAL_INTERVAL: u64 = 10;

/// Evaluate and update goal stacks for all NPCs.
/// Called post-apply from runtime.rs.
pub fn evaluate_goals(state: &mut WorldState) {
    if state.tick % GOAL_EVAL_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // Collect settlement data for goal evaluation.
    let settlement_data: Vec<(u32, f32, f32, (f32, f32))> = state.settlements.iter()
        .map(|s| (s.id, s.threat_level, s.stockpile[commodity::FOOD], s.pos))
        .collect();

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
            // Flee away from settlement center (toward wilderness).
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
            // Can't use remove_kind directly because Flee has a payload.
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
        let stale_threshold = 500; // 500 ticks = ~50 seconds
        npc.goal_stack.goals.retain(|g| {
            tick.saturating_sub(g.started_tick) < stale_threshold
                || matches!(g.kind, GoalKind::Work | GoalKind::Trade { .. } | GoalKind::Quest { .. })
        });

        // --- Sync goal stack → EconomicIntent (backward compat) ---
        // The current systems still read EconomicIntent. Sync from top goal.
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
                npc.economic_intent = EconomicIntent::Idle; // combat-ready
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
            // Other goals don't map to EconomicIntent — they use their own movement.
            _ => {}
        }
    }
}
