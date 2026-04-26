//! Unit tests for the `beliefs(observer).observe(target) with { ... }` grammar
//! extension (Plan ToM Task 4).
//!
//! Covers:
//!  - Happy-path parse + resolve produces `IrStmt::BeliefObserve`.
//!  - Unknown field name is rejected with `UnknownBeliefField`.
//!  - Partial field lists (only `confidence`) are accepted.

use dsl_compiler::{compile, ir::IrStmt};

/// Minimal physics rule that uses the belief mutation primitive.
/// Uses `AgentMoved` which exists in `engine_data::events`.
const BASIC_SRC: &str = r#"
event AgentMoved {
    actor: AgentId,
    location: Vec3,
}

physics observe_self_move {
    on AgentMoved { actor: a, location: loc } {
        beliefs(a).observe(a) with {
            last_known_pos: loc,
            confidence: 1.0,
        }
    }
}
"#;

#[test]
fn parses_belief_mutation() {
    let comp = compile(BASIC_SRC).expect("compile");
    // Exactly one physics rule.
    assert_eq!(comp.physics.len(), 1, "expected 1 physics rule");
    let ph = &comp.physics[0];
    assert_eq!(ph.name, "observe_self_move");
    assert_eq!(ph.handlers.len(), 1);
    let body = &ph.handlers[0].body;
    // Body must contain exactly one statement.
    assert_eq!(body.len(), 1, "expected 1 statement in handler body");
    // That statement must be a BeliefObserve.
    match &body[0] {
        IrStmt::BeliefObserve { fields, .. } => {
            assert_eq!(fields.len(), 2, "expected 2 field assignments");
            assert_eq!(fields[0].name, "last_known_pos");
            assert_eq!(fields[1].name, "confidence");
        }
        other => panic!("expected IrStmt::BeliefObserve, got {:?}", other),
    }
}

#[test]
fn belief_mutation_single_field() {
    let src = r#"
event AgentMoved {
    actor: AgentId,
    location: Vec3,
}

physics set_confidence {
    on AgentMoved { actor: a, location: loc } {
        beliefs(a).observe(a) with {
            confidence: 1.0,
        }
    }
}
"#;
    let comp = compile(src).expect("compile");
    let body = &comp.physics[0].handlers[0].body;
    match &body[0] {
        IrStmt::BeliefObserve { fields, .. } => {
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].name, "confidence");
        }
        other => panic!("expected IrStmt::BeliefObserve, got {:?}", other),
    }
}

#[test]
fn belief_mutation_all_valid_fields() {
    // Verify every field in the allowlist is accepted.
    let src = r#"
event AgentMoved {
    actor: AgentId,
    location: Vec3,
    hp: f32,
    max_hp: f32,
    ctype: u32,
    tick_val: u32,
    conf: f32,
}

physics full_observation {
    on AgentMoved { actor: a, location: loc, hp: h, max_hp: mh, ctype: ct, tick_val: tv, conf: c } {
        beliefs(a).observe(a) with {
            last_known_pos: loc,
            last_known_hp: h,
            last_known_max_hp: mh,
            last_known_creature_type: ct,
            last_updated_tick: tv,
            confidence: c,
        }
    }
}
"#;
    let comp = compile(src).expect("compile all known belief fields");
    let body = &comp.physics[0].handlers[0].body;
    match &body[0] {
        IrStmt::BeliefObserve { fields, .. } => {
            assert_eq!(fields.len(), 6);
        }
        other => panic!("expected IrStmt::BeliefObserve, got {:?}", other),
    }
}

#[test]
fn unknown_belief_field_is_rejected() {
    let src = r#"
event AgentMoved {
    actor: AgentId,
    location: Vec3,
}

physics bad_field {
    on AgentMoved { actor: a, location: loc } {
        beliefs(a).observe(a) with {
            lastpos: loc,
        }
    }
}
"#;
    let err = compile(src).expect_err("should fail on unknown field");
    let msg = err.to_string();
    assert!(
        msg.contains("lastpos"),
        "error should mention the unknown field name; got: {msg}"
    );
    assert!(
        msg.contains("last_known_pos"),
        "error should suggest valid fields; got: {msg}"
    );
}
