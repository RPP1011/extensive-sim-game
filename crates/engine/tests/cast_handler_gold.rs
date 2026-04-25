//! Combat Foundation Task 16 — `TransferGoldHandler` moves `amount` between
//! two agents' `cold_inventory[*].gold` fields.

use engine_rules::physics::dispatch_effect_gold_transfer;
use engine_rules::views::ViewRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine::state::agent_types::Inventory;
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos: Vec3::ZERO, hp: 100.0, ..Default::default() }).unwrap()
}

fn set_gold(state: &mut SimState, id: AgentId, gold: i32) {
    let mut inv = state.agent_inventory(id).unwrap_or_default();
    inv.gold = gold;
    state.set_agent_inventory(id, inv);
}

fn gold_of(state: &SimState, id: AgentId) -> i32 {
    state.agent_inventory(id).map(|i| i.gold).unwrap_or(0)
}

#[test]
fn transfer_moves_positive_amount_from_caster_to_target() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let alice = spawn(&mut state, CreatureType::Human);
    let bob   = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, alice, 100);
    set_gold(&mut state, bob,   0);

    dispatch_effect_gold_transfer(
        &Event::EffectGoldTransfer { from: alice, to: bob, amount: 30, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(gold_of(&state, alice), 70);
    assert_eq!(gold_of(&state, bob),   30);
}

#[test]
fn negative_amount_allows_debt_on_sender() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let alice = spawn(&mut state, CreatureType::Human);
    let bob   = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, alice, 100);
    set_gold(&mut state, bob,   0);

    dispatch_effect_gold_transfer(
        &Event::EffectGoldTransfer { from: alice, to: bob, amount: -50, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(gold_of(&state, alice), 150);
    assert_eq!(gold_of(&state, bob),   -50);
}

#[test]
fn conservation_sum_is_invariant_under_arbitrary_transfers() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let alice = spawn(&mut state, CreatureType::Human);
    let bob   = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, alice, 1_000);
    set_gold(&mut state, bob,   250);
    let initial_sum = gold_of(&state, alice) + gold_of(&state, bob);

    for amt in [17_i32, -123, 500, -7, 1, -1, 999_999, -1_000_000] {
        dispatch_effect_gold_transfer(
            &Event::EffectGoldTransfer { from: alice, to: bob, amount: amt, tick: 0 },
            &mut state,
            &mut views,
            &mut events,
        );
        assert_eq!(
            gold_of(&state, alice) + gold_of(&state, bob),
            initial_sum,
            "sum must be preserved after transfer of {amt}"
        );
    }
}

#[test]
fn zero_amount_is_a_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let alice = spawn(&mut state, CreatureType::Human);
    let bob   = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, alice, 100);
    set_gold(&mut state, bob,   50);

    dispatch_effect_gold_transfer(
        &Event::EffectGoldTransfer { from: alice, to: bob, amount: 0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(gold_of(&state, alice), 100);
    assert_eq!(gold_of(&state, bob),   50);
}

#[test]
fn self_transfer_is_a_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let alice = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, alice, 100);

    dispatch_effect_gold_transfer(
        &Event::EffectGoldTransfer { from: alice, to: alice, amount: 40, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(gold_of(&state, alice), 100);
}

#[test]
fn registry_dispatches_gold_transfer_via_builtins() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let cascade = engine_rules::with_engine_builtins();
    let mut views = ViewRegistry::new();
    let alice = spawn(&mut state, CreatureType::Human);
    let bob   = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, alice, 200);
    set_gold(&mut state, bob,   0);

    events.push(Event::EffectGoldTransfer { from: alice, to: bob, amount: 75, tick: 0 });
    cascade.run_fixed_point(&mut state, &mut views, &mut events);

    assert_eq!(gold_of(&state, alice), 125);
    assert_eq!(gold_of(&state, bob),   75);
}

#[test]
fn transfer_preserves_commodity_slots() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let alice = spawn(&mut state, CreatureType::Human);
    let bob   = spawn(&mut state, CreatureType::Human);
    let inv_a = Inventory { gold: 100, commodities: [1, 2, 3, 4, 5, 6, 7, 8] };
    let inv_b = Inventory { gold: 0,   commodities: [9, 8, 7, 6, 5, 4, 3, 2] };
    state.set_agent_inventory(alice, inv_a);
    state.set_agent_inventory(bob,   inv_b);

    dispatch_effect_gold_transfer(
        &Event::EffectGoldTransfer { from: alice, to: bob, amount: 40, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );

    let got_a = state.agent_inventory(alice).unwrap();
    let got_b = state.agent_inventory(bob).unwrap();
    assert_eq!(got_a.gold, 60);
    assert_eq!(got_a.commodities, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(got_b.gold, 40);
    assert_eq!(got_b.commodities, [9, 8, 7, 6, 5, 4, 3, 2]);
}
