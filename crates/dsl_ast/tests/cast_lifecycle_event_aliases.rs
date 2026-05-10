//! Plan G — cast lifecycle event aliases (#284 sub-item #7 prereq).
//!
//! Pins the resolver-side mapping that lets `.sim` files declare
//! `event CastBegan { ... }` / `event CastResolved { ... }` /
//! `event CastInterrupted { ... }` and have the resulting `EventIR`
//! carry the engine-defined `EventKindId` discriminants 78 / 79 / 80.
//!
//! Without this alias wiring, a per-fixture sim that emits these
//! events on the GPU consumer side (e.g. `emit CastBegan { actor:
//! self, ability_id: aid, ... }` inside `physics RecordCastBegin`)
//! would tag the chronicle record with a sequential id chosen by
//! the resolver, mismatching the engine schema and breaking
//! cross-backend parity for replay.
//!
//! Mirrors the host-side emit shape in
//! `crates/engine/src/ability/apply.rs::emit_cast_began` (and its
//! resolved/interrupted siblings), which already write kinds 78/79/80
//! to the chronicle when the CPU backend processes a CastBegin.

use dsl_ast::compile;

fn resolve_event_kind_id(src: &str, event_name: &str) -> Option<u32> {
    let comp = compile(src).unwrap_or_else(|e| {
        panic!("compile failed:\n{src}\nerror: {e}")
    });
    comp.events
        .iter()
        .find(|e| e.name == event_name)
        .unwrap_or_else(|| panic!("event `{event_name}` not in resolved compilation"))
        .engine_kind_id
}

#[test]
fn cast_began_event_resolves_to_engine_kind_78() {
    let src = "\
        event Tick { }\n\
        event CastBegan { actor: AgentId, ability_id: u16, duration_ticks: u16, target_x_q8: i16, target_y_q8: i16 }\n\
        entity Probe : Agent { }\n\
    ";
    assert_eq!(resolve_event_kind_id(src, "CastBegan"), Some(78));
}

#[test]
fn cast_resolved_event_resolves_to_engine_kind_79() {
    let src = "\
        event Tick { }\n\
        event CastResolved { actor: AgentId, ability_id: u16 }\n\
        entity Probe : Agent { }\n\
    ";
    assert_eq!(resolve_event_kind_id(src, "CastResolved"), Some(79));
}

#[test]
fn cast_interrupted_event_resolves_to_engine_kind_80() {
    let src = "\
        event Tick { }\n\
        event CastInterrupted { actor: AgentId, ability_id: u16, interrupt_kind: u8 }\n\
        entity Probe : Agent { }\n\
    ";
    assert_eq!(resolve_event_kind_id(src, "CastInterrupted"), Some(80));
}

/// Negative pin: a typoed cast lifecycle name (`CastResolve` without
/// the `d`) does NOT resolve to an engine kind id — the alias table
/// is exact-match by design. Future drift that adds a partial-match
/// fallback would silently hijack user-defined events.
#[test]
fn typoed_cast_lifecycle_name_falls_through_to_user_event_id() {
    let src = "\
        event Tick { }\n\
        event CastResolve { actor: AgentId }\n\
        entity Probe : Agent { }\n\
    ";
    // Typo gets a sequential id (None means "not in the engine alias table"),
    // which the resolver then fills with a sequential allocation downstream.
    assert_eq!(resolve_event_kind_id(src, "CastResolve"), None);
}
