//! CPU-only determinism: the CPU view fold path produces byte-equal results
//! regardless of the order events are pushed into the ring.
//!
//! The forest_fire fixture runs GPU-only (no serial backend exists yet), so
//! this test exercises the CPU view fold contract directly via the engine's
//! `DamageTaken` materialized view and `EventRing`, which is the same
//! code path that any future serial backend would use.
//!
//! **What this tests:**
//! The `DamageTaken::fold_since` implementation sorts by `(target, emit_seq)`
//! before reducing. This ensures that two rings whose events arrive in
//! different push orders (but with the same emit_seq assignments) produce
//! byte-identical accumulated damage totals — even when f32 addition is
//! non-associative.
//!
//! Two runs are constructed with the same logical events but pushed in
//! reversed order. Without the sort, f32 accumulation could differ in the
//! low bits. With the sort both folds visit events in the same `(target,
//! emit_seq)` order and produce identical results.

use engine::event::EventRing;
use engine::view::materialized::{DamageTaken, MaterializedView};
use engine_data::events::Event;
use engine_data::ids::AgentId;

const AGENT_CAP: usize = 16;

// ---- helpers ---------------------------------------------------------------

fn agent(raw: u32) -> AgentId {
    AgentId::new(raw).expect("AgentId must be nonzero")
}

/// Build an `AgentAttacked` event with an explicit `emit_seq` in the ring.
fn push_attacked(
    ring: &mut EventRing<Event>,
    actor: AgentId,
    target: AgentId,
    damage: f32,
    tick: u32,
    emit_seq: u32,
) {
    ring.push_with_emit_seq(
        Event::AgentAttacked { actor, target, damage, tick },
        emit_seq,
    );
}

// ---- test: fold is order-independent via sort ------------------------------

/// Push the same set of damage events in two different orders; verify that
/// `DamageTaken::fold_since` produces byte-identical results both times.
#[test]
fn cpu_fold_byte_equal_regardless_of_push_order() {
    let a1 = agent(1);
    let a2 = agent(2);
    let attacker = agent(3);

    // Run A: events pushed in order (target=1 first, then target=2).
    let mut ring_a = EventRing::<Event>::with_cap(256);
    let start_a = ring_a.push_count();
    push_attacked(&mut ring_a, attacker, a1, 10.0, 1, 0);
    push_attacked(&mut ring_a, attacker, a1, 5.0,  1, 1);
    push_attacked(&mut ring_a, attacker, a2, 3.0,  1, 2);
    push_attacked(&mut ring_a, attacker, a1, 2.5,  1, 3);
    push_attacked(&mut ring_a, attacker, a2, 7.0,  1, 4);
    let mut view_a = DamageTaken::new(AGENT_CAP);
    view_a.fold_since(&ring_a, start_a);

    // Run B: same events pushed in reversed order.
    let mut ring_b = EventRing::<Event>::with_cap(256);
    let start_b = ring_b.push_count();
    push_attacked(&mut ring_b, attacker, a2, 7.0,  1, 4);
    push_attacked(&mut ring_b, attacker, a1, 2.5,  1, 3);
    push_attacked(&mut ring_b, attacker, a2, 3.0,  1, 2);
    push_attacked(&mut ring_b, attacker, a1, 5.0,  1, 1);
    push_attacked(&mut ring_b, attacker, a1, 10.0, 1, 0);
    let mut view_b = DamageTaken::new(AGENT_CAP);
    view_b.fold_since(&ring_b, start_b);

    // Byte-equal: the sort by (target, emit_seq) should produce identical
    // fold orders and thus identical f32 accumulation results.
    let dmg_a1 = view_a.value(a1);
    let dmg_b1 = view_b.value(a1);
    let dmg_a2 = view_a.value(a2);
    let dmg_b2 = view_b.value(a2);

    assert_eq!(
        dmg_a1.to_bits(), dmg_b1.to_bits(),
        "CPU fold not bit-identical for agent 1: run_a={dmg_a1} run_b={dmg_b1}",
    );
    assert_eq!(
        dmg_a2.to_bits(), dmg_b2.to_bits(),
        "CPU fold not bit-identical for agent 2: run_a={dmg_a2} run_b={dmg_b2}",
    );
}

/// Verify the expected total damage values are correct (sanity check).
#[test]
fn cpu_fold_accumulates_expected_totals() {
    let a1 = agent(1);
    let a2 = agent(2);
    let attacker = agent(3);

    let mut ring = EventRing::<Event>::with_cap(256);
    let start = ring.push_count();
    push_attacked(&mut ring, attacker, a1, 10.0, 1, 0);
    push_attacked(&mut ring, attacker, a1, 5.0,  1, 1);
    push_attacked(&mut ring, attacker, a2, 3.0,  1, 2);
    push_attacked(&mut ring, attacker, a1, 2.5,  1, 3);
    push_attacked(&mut ring, attacker, a2, 7.0,  1, 4);

    let mut view = DamageTaken::new(AGENT_CAP);
    view.fold_since(&ring, start);

    // a1: 10.0 + 5.0 + 2.5 = 17.5
    assert_eq!(view.value(a1), 17.5, "expected 17.5 total damage on agent 1");
    // a2: 3.0 + 7.0 = 10.0
    assert_eq!(view.value(a2), 10.0, "expected 10.0 total damage on agent 2");
    // attacker never took damage
    assert_eq!(view.value(attacker), 0.0, "attacker should have 0 damage taken");
}
