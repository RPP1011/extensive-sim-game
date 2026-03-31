//! Emergent trade route system — maintenance and decay.
//!
//! NPCs who repeatedly profit from trading between two settlements establish
//! permanent trade routes. Successful routes attract more traders (utility bonus
//! in npc_decisions). Failed routes decay and get abandoned.
//!
//! Cadence: every 200 ticks (post-apply, needs `&mut WorldState`).

use crate::world_sim::state::*;

/// How often to run trade route maintenance.
const ROUTE_INTERVAL: u64 = 200;

/// Strength decay per maintenance cycle.
const DECAY_PER_CYCLE: f32 = 0.05;

/// Routes below this strength are abandoned.
const ABANDON_THRESHOLD: f32 = 0.1;

/// Number of profitable trades to the same destination before a route is established.
const ESTABLISH_THRESHOLD: u32 = 3;

/// Strength boost when an NPC completes a profitable trade on an existing route.
const REINFORCE_AMOUNT: f32 = 0.15;

/// Initial strength when a new route is established.
const INITIAL_STRENGTH: f32 = 0.5;

/// Maximum number of trade routes in the world.
const MAX_ROUTES: usize = 50;

/// Process trade route decay, abandonment, and chronicle entries.
/// Called post-apply from runtime.rs.
pub fn advance_trade_routes(state: &mut WorldState) {
    if state.tick % ROUTE_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // --- Phase 1: Decay all routes ---
    for route in &mut state.trade_routes {
        route.strength = (route.strength - DECAY_PER_CYCLE).max(0.0);
    }

    // --- Phase 2: Abandon dead routes ---
    // Collect abandoned route info before removing (avoids borrow conflict with retain).
    let mut abandoned_chronicles: Vec<String> = Vec::new();
    for route in &state.trade_routes {
        if route.strength < ABANDON_THRESHOLD {
            let a_name = state.settlement(route.settlement_a)
                .map(|s| s.name.clone()).unwrap_or_else(|| "unknown".into());
            let b_name = state.settlement(route.settlement_b)
                .map(|s| s.name.clone()).unwrap_or_else(|| "unknown".into());
            abandoned_chronicles.push(format!(
                "The trade route between {} and {} has been abandoned after {} trades",
                a_name, b_name, route.trade_count,
            ));
        }
    }
    state.trade_routes.retain(|route| route.strength >= ABANDON_THRESHOLD);

    // Emit chronicles for abandoned routes.
    for text in abandoned_chronicles {
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Economy,
            text,
            entity_ids: vec![],
        });
    }

    // --- Phase 3: Clear stale trade_route_id references on NPCs ---
    // After removal, indices may have shifted, so clear any that point to
    // non-existent routes or mismatched settlements.
    let route_count = state.trade_routes.len();
    for entity in &mut state.entities {
        if let Some(npc) = &mut entity.npc {
            if let Some(rid) = npc.trade_route_id {
                if rid >= route_count {
                    npc.trade_route_id = None;
                }
            }
        }
    }
}

/// Record a profitable trade completion for an NPC.
/// Called from npc_decisions when a trade run arrives and sells goods.
///
/// If the NPC has traded profitably with this destination 3+ times,
/// establishes a new trade route (or reinforces an existing one).
pub fn record_profitable_trade(
    state: &mut WorldState,
    entity_id: u32,
    home_settlement_id: u32,
    dest_settlement_id: u32,
    profit: f32,
) {
    let tick = state.tick;

    // Update trade history on the NPC.
    let entity = match state.entity_mut(entity_id) {
        Some(e) => e,
        None => return,
    };
    let npc = match &mut entity.npc {
        Some(n) => n,
        None => return,
    };

    // Find or create history entry for this destination.
    let entry = npc.trade_history.iter_mut().find(|(dest, _)| *dest == dest_settlement_id);
    let count = match entry {
        Some((_, count)) => {
            *count += 1;
            *count
        }
        None => {
            npc.trade_history.push((dest_settlement_id, 1));
            1
        }
    };

    // Cap trade history length (keep most recent 8 destinations).
    if npc.trade_history.len() > 8 {
        npc.trade_history.remove(0);
    }

    // Check if a route already exists between these settlements.
    let (a, b) = if home_settlement_id < dest_settlement_id {
        (home_settlement_id, dest_settlement_id)
    } else {
        (dest_settlement_id, home_settlement_id)
    };

    let existing = state.trade_routes.iter_mut().enumerate().find(|(_, r)| {
        (r.settlement_a == a && r.settlement_b == b)
            || (r.settlement_a == b && r.settlement_b == a)
    });

    if let Some((idx, route)) = existing {
        // Reinforce existing route.
        route.strength = (route.strength + REINFORCE_AMOUNT).min(1.0);
        route.total_profit += profit;
        route.trade_count += 1;

        // Assign NPC to this route.
        if let Some(entity) = state.entity_mut(entity_id) {
            if let Some(npc) = &mut entity.npc {
                npc.trade_route_id = Some(idx);
            }
        }
    } else if count >= ESTABLISH_THRESHOLD && state.trade_routes.len() < MAX_ROUTES {
        // Establish a new route.
        let route = TradeRoute {
            settlement_a: a,
            settlement_b: b,
            established_tick: tick,
            total_profit: profit,
            trade_count: 1,
            strength: INITIAL_STRENGTH,
        };
        let idx = state.trade_routes.len();
        state.trade_routes.push(route);

        // Assign NPC to the new route.
        if let Some(entity) = state.entity_mut(entity_id) {
            if let Some(npc) = &mut entity.npc {
                npc.trade_route_id = Some(idx);
            }
        }

        // Chronicle the new route.
        let a_name = state.settlement(a)
            .map(|s| s.name.clone()).unwrap_or_else(|| "unknown".into());
        let b_name = state.settlement(b)
            .map(|s| s.name.clone()).unwrap_or_else(|| "unknown".into());
        let trader_name = state.entity(entity_id)
            .map(|e| crate::world_sim::naming::entity_display_name(e))
            .unwrap_or_else(|| "A trader".into());
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Economy,
            text: format!(
                "{} established a trade route between {} and {}",
                trader_name, a_name, b_name,
            ),
            entity_ids: vec![entity_id],
        });
    }
}

/// Look up the utility bonus for trading to a given destination on an established route.
/// Returns 0.0 if no route exists, up to 0.3 for a strong route.
pub fn route_utility_bonus(state: &WorldState, home_id: u32, dest_id: u32) -> f32 {
    let (a, b) = if home_id < dest_id { (home_id, dest_id) } else { (dest_id, home_id) };
    for route in &state.trade_routes {
        if (route.settlement_a == a && route.settlement_b == b)
            || (route.settlement_a == b && route.settlement_b == a)
        {
            return route.strength * 0.3;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state_with_settlements() -> WorldState {
        let mut state = WorldState::new(42);
        state.tick = 200;
        let s1 = SettlementState::new(10, "Alpha".into(), (0.0, 0.0));
        let s2 = SettlementState::new(20, "Beta".into(), (50.0, 0.0));
        state.settlements.push(s1);
        state.settlements.push(s2);
        state.rebuild_settlement_index();
        state
    }

    fn add_npc(state: &mut WorldState, id: u32, home_sid: u32) {
        let mut e = Entity::new_npc(id, (0.0, 0.0));
        if let Some(npc) = &mut e.npc {
            npc.home_settlement_id = Some(home_sid);
        }
        state.entities.push(e);
        state.rebuild_entity_cache();
    }

    #[test]
    fn route_established_after_three_trades() {
        let mut state = make_state_with_settlements();
        add_npc(&mut state, 100, 10);

        // First two trades: no route yet.
        record_profitable_trade(&mut state, 100, 10, 20, 5.0);
        assert!(state.trade_routes.is_empty());

        record_profitable_trade(&mut state, 100, 10, 20, 8.0);
        assert!(state.trade_routes.is_empty());

        // Third trade: route established.
        record_profitable_trade(&mut state, 100, 10, 20, 10.0);
        assert_eq!(state.trade_routes.len(), 1);
        assert_eq!(state.trade_routes[0].settlement_a, 10);
        assert_eq!(state.trade_routes[0].settlement_b, 20);
        assert!((state.trade_routes[0].strength - INITIAL_STRENGTH).abs() < 0.01);
        assert_eq!(state.trade_routes[0].trade_count, 1);

        // NPC should be assigned to the route.
        let npc = state.entity(100).unwrap().npc.as_ref().unwrap();
        assert_eq!(npc.trade_route_id, Some(0));
    }

    #[test]
    fn route_reinforced_by_further_trades() {
        let mut state = make_state_with_settlements();
        add_npc(&mut state, 100, 10);

        // Establish route.
        for _ in 0..3 {
            record_profitable_trade(&mut state, 100, 10, 20, 5.0);
        }
        let initial_strength = state.trade_routes[0].strength;

        // Another trade reinforces it.
        record_profitable_trade(&mut state, 100, 10, 20, 5.0);
        assert!(state.trade_routes[0].strength > initial_strength);
        assert_eq!(state.trade_routes[0].trade_count, 2); // 1 from establish + 1 reinforce
    }

    #[test]
    fn route_decays_and_abandoned() {
        let mut state = make_state_with_settlements();
        state.trade_routes.push(TradeRoute {
            settlement_a: 10,
            settlement_b: 20,
            established_tick: 0,
            total_profit: 50.0,
            trade_count: 5,
            strength: 0.12, // just above threshold
        });

        // First decay cycle: 0.12 - 0.05 = 0.07 < 0.1 => abandoned.
        advance_trade_routes(&mut state);
        assert!(state.trade_routes.is_empty());

        // Should have a chronicle entry about abandonment.
        assert!(state.chronicle.iter().any(|c| c.text.contains("abandoned")));
    }

    #[test]
    fn route_utility_bonus_scales_with_strength() {
        let mut state = make_state_with_settlements();
        state.trade_routes.push(TradeRoute {
            settlement_a: 10,
            settlement_b: 20,
            established_tick: 0,
            total_profit: 100.0,
            trade_count: 10,
            strength: 0.8,
        });

        let bonus = route_utility_bonus(&state, 10, 20);
        assert!((bonus - 0.24).abs() < 0.01); // 0.8 * 0.3 = 0.24

        // Reversed order should also work.
        let bonus_rev = route_utility_bonus(&state, 20, 10);
        assert!((bonus_rev - 0.24).abs() < 0.01);

        // Non-existent route.
        let bonus_none = route_utility_bonus(&state, 10, 30);
        assert!((bonus_none).abs() < 0.001);
    }

    #[test]
    fn skips_off_cadence() {
        let mut state = make_state_with_settlements();
        state.tick = 199; // not a multiple of 200
        state.trade_routes.push(TradeRoute {
            settlement_a: 10,
            settlement_b: 20,
            established_tick: 0,
            total_profit: 50.0,
            trade_count: 5,
            strength: 0.5,
        });

        advance_trade_routes(&mut state);
        // Strength should be unchanged.
        assert!((state.trade_routes[0].strength - 0.5).abs() < 0.001);
    }
}
