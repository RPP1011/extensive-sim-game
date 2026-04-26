//! Combat Foundation — tax-ability regression fixture.
//!
//! Named scenario: gold-transfer ("tax") mechanic. A caster transfers gold
//! from themselves to a target (or vice-versa) via `EffectGoldTransfer`.
//! Mechanics exercised:
//!   - Gold is conserved: sender loses exactly what receiver gains.
//!   - No double-counting when the cascade converges on the same tick.
//!   - Zero-amount transfer is a no-op.
//!   - Multiple transfers in one tick are all applied independently.
//!
//! Uses `dispatch_effect_gold_transfer` and `with_engine_builtins` cascade
//! (same pattern as `cast_handler_gold.rs`).

use engine::event::EventRing;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_rules::physics::dispatch_effect_gold_transfer;
use engine_rules::views::ViewRegistry;
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: ct,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .expect("spawn agent")
}

fn set_gold(state: &mut SimState, id: AgentId, gold: i32) {
    let mut inv = state.agent_inventory(id).unwrap_or_default();
    inv.gold = gold;
    state.set_agent_inventory(id, inv);
}

fn gold_of(state: &SimState, id: AgentId) -> i32 {
    state.agent_inventory(id).map(|i| i.gold).unwrap_or(0)
}

/// Tax: caster pays target `amount` gold — sum is conserved.
#[test]
fn tax_transfers_gold_and_conserves_sum() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();

    let tax_man = spawn(&mut state, CreatureType::Human);
    let merchant = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, tax_man, 0);
    set_gold(&mut state, merchant, 200);

    let initial_sum = gold_of(&state, tax_man) + gold_of(&state, merchant);

    // Tax man collects 50 gold from merchant.
    dispatch_effect_gold_transfer(
        &Event::EffectGoldTransfer {
            from: merchant,
            to: tax_man,
            amount: 50,
            tick: 0,
        },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(gold_of(&state, merchant), 150, "merchant loses 50 gold");
    assert_eq!(gold_of(&state, tax_man), 50, "tax_man gains 50 gold");
    assert_eq!(
        gold_of(&state, tax_man) + gold_of(&state, merchant),
        initial_sum,
        "total gold must be conserved"
    );
}

/// Multiple tax events in one tick are all applied; sum remains invariant.
#[test]
fn multiple_tax_events_all_applied_sum_conserved() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();

    let treasury = spawn(&mut state, CreatureType::Human);
    let npc = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, treasury, 1000);
    set_gold(&mut state, npc, 0);

    let transfers: &[i32] = &[10, 30, 5, 100];
    let expected_total: i32 = transfers.iter().sum();

    for &amt in transfers {
        dispatch_effect_gold_transfer(
            &Event::EffectGoldTransfer {
                from: treasury,
                to: npc,
                amount: amt,
                tick: 0,
            },
            &mut state,
            &mut views,
            &mut events,
        );
    }

    assert_eq!(
        gold_of(&state, npc),
        expected_total,
        "npc gold should equal sum of all transfers"
    );
    assert_eq!(
        gold_of(&state, treasury),
        1000 - expected_total,
        "treasury should be reduced by total transferred"
    );
    // Conservation check.
    assert_eq!(
        gold_of(&state, treasury) + gold_of(&state, npc),
        1000,
        "total gold invariant"
    );
}

/// Zero-amount transfer is a no-op (no state change).
#[test]
fn zero_tax_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();

    let a = spawn(&mut state, CreatureType::Human);
    let b = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, a, 100);
    set_gold(&mut state, b, 50);

    dispatch_effect_gold_transfer(
        &Event::EffectGoldTransfer {
            from: a,
            to: b,
            amount: 0,
            tick: 0,
        },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(gold_of(&state, a), 100, "zero transfer: sender unchanged");
    assert_eq!(gold_of(&state, b), 50, "zero transfer: receiver unchanged");
}

/// Cascade-dispatched tax via `with_engine_builtins` produces identical result
/// to the direct-dispatch path — no double-counting on convergence.
#[test]
fn cascade_tax_no_double_counting() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let cascade = engine_rules::with_engine_builtins();
    let mut views = ViewRegistry::new();

    let payer = spawn(&mut state, CreatureType::Human);
    let payee = spawn(&mut state, CreatureType::Human);
    set_gold(&mut state, payer, 500);
    set_gold(&mut state, payee, 0);

    // Push one tax event into the ring; run cascade to fixed-point.
    events.push(Event::EffectGoldTransfer {
        from: payer,
        to: payee,
        amount: 75,
        tick: 0,
    });
    cascade.run_fixed_point(&mut state, &mut views, &mut events);

    // Exactly one transfer happened — no duplication.
    assert_eq!(gold_of(&state, payer), 425, "payer post-cascade: 500-75=425");
    assert_eq!(gold_of(&state, payee), 75, "payee post-cascade: 75");
    assert_eq!(
        gold_of(&state, payer) + gold_of(&state, payee),
        500,
        "total gold conserved through cascade"
    );
}
