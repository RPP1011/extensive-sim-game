//! GPU-emittable validator for `physics` rule bodies (compiler/spec.md §1.2).
//!
//! Task 155 expanded the physics surface to include `for` + `match` — a
//! richer CPU-friendly surface than the fold / mask / scoring contexts
//! carry. This file pins the GPU-emittable discipline for that surface:
//! POD values, bounded inline-array iteration, no heap, no unbounded
//! recursion, no user-defined helpers, no `String` bindings. Each test
//! names a specific forbidden construct and checks that the resolver
//! rejects it with a `NotGpuEmittable` error.
//!
//! The positive sanity check (every rule in `assets/sim/physics.sim`
//! compiles under the validator) is covered by the existing `emit_e2e`
//! test suite — if any of those passed before this file landed and fail
//! now, the validator has grown a false-positive.

use dsl_compiler::{CompileError, ResolveError};

#[test]
fn physics_for_over_user_defined_helper_rejected() {
    let src = r#"
        event Trigger { actor: AgentId, ability: AbilityId }
        physics demo @phase(event) @terminating_in(1) {
            on Trigger { actor: a, ability: ab } {
                for op in my_helper(ab) {
                    emit Trigger { actor: a, ability: ab }
                }
            }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should reject UDF `for` source");
    match err {
        CompileError::Resolve(ResolveError::NotGpuEmittable {
            physics_name,
            construct,
            ..
        }) => {
            assert_eq!(physics_name, "demo");
            assert!(construct.contains("my_helper"));
            assert!(construct.contains("for-loop"));
        }
        other => panic!("expected NotGpuEmittable(for over helper); got: {other:?}"),
    }
}

#[test]
fn physics_for_over_stdlib_source_is_accepted() {
    let src = r#"
        event Trigger { actor: AgentId, ability: AbilityId }
        physics demo @phase(event) @terminating_in(1) {
            on Trigger { actor: a, ability: ab } {
                for op in abilities.effects(ab) {
                    emit Trigger { actor: a, ability: ab }
                }
            }
        }
    "#;
    dsl_compiler::compile(src).expect("stdlib-bounded `for` must compile");
}

#[test]
fn physics_string_let_binding_rejected() {
    let src = r#"
        event Trigger { actor: AgentId }
        physics demo @phase(event) @terminating_in(1) {
            on Trigger { actor: a } {
                let note = "hello"
                emit Trigger { actor: a }
            }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should reject String let-binding");
    match err {
        CompileError::Resolve(ResolveError::NotGpuEmittable {
            physics_name,
            construct,
            ..
        }) => {
            assert_eq!(physics_name, "demo");
            assert!(construct.contains("String") && construct.contains("note"));
        }
        other => panic!("expected NotGpuEmittable(String let); got: {other:?}"),
    }
}

#[test]
fn physics_string_literal_in_emit_rejected() {
    let src = r#"
        event Trigger { actor: AgentId, tag: string }
        physics demo @phase(event) @terminating_in(1) {
            on Trigger { actor: a } {
                emit Trigger { actor: a, tag: "literal" }
            }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should reject String literal in body");
    match err {
        CompileError::Resolve(ResolveError::NotGpuEmittable {
            physics_name,
            construct,
            ..
        }) => {
            assert_eq!(physics_name, "demo");
            assert!(construct.contains("String"));
        }
        other => panic!("expected NotGpuEmittable(String literal); got: {other:?}"),
    }
}

#[test]
fn physics_body_unresolved_call_rejected() {
    let src = r#"
        event Trigger { actor: AgentId }
        physics demo @phase(event) @terminating_in(1) {
            on Trigger { actor: a } {
                let x = game_helper(a)
                emit Trigger { actor: a }
            }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should reject UDF call");
    match err {
        CompileError::Resolve(ResolveError::NotGpuEmittable {
            physics_name,
            construct,
            ..
        }) => {
            assert_eq!(physics_name, "demo");
            assert!(construct.contains("game_helper"));
        }
        other => panic!("expected NotGpuEmittable(unresolved call); got: {other:?}"),
    }
}

#[test]
fn physics_unbounded_self_recursion_rejected() {
    let src = r#"
        event Trigger { actor: AgentId }
        physics loop_rule @phase(event) {
            on Trigger { actor: a } {
                emit Trigger { actor: a }
            }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should reject unbounded recursion");
    match err {
        CompileError::Resolve(ResolveError::NotGpuEmittable {
            physics_name,
            construct,
            ..
        }) => {
            assert_eq!(physics_name, "loop_rule");
            assert!(construct.contains("recursive self-emission"));
        }
        other => panic!("expected NotGpuEmittable(self-recursion); got: {other:?}"),
    }
}

#[test]
fn physics_self_recursion_with_terminating_in_is_accepted() {
    let src = r#"
        event Trigger { actor: AgentId }
        physics loop_rule @phase(event) @terminating_in(4) {
            on Trigger { actor: a } {
                emit Trigger { actor: a }
            }
        }
    "#;
    dsl_compiler::compile(src).expect("@terminating_in bounded recursion must compile");
}

#[test]
fn physics_self_recursion_guarded_by_cascade_ceiling_is_accepted() {
    let src = r#"
        event Trigger { actor: AgentId, depth: u8 }
        physics loop_rule @phase(event) {
            on Trigger { actor: a, depth: d } {
                let new_depth = d + 1
                if new_depth >= cascade.max_iterations {
                    emit Trigger { actor: a, depth: new_depth }
                } else {
                    emit Trigger { actor: a, depth: new_depth }
                }
            }
        }
    "#;
    dsl_compiler::compile(src)
        .expect("cascade.max_iterations guarded recursion must compile");
}

#[test]
fn physics_indirect_emission_cycle_rejected() {
    let src = r#"
        event Ping { actor: AgentId }
        event Pong { actor: AgentId }
        physics a_rule @phase(event) {
            on Ping { actor: a } {
                emit Pong { actor: a }
            }
        }
        physics b_rule @phase(event) {
            on Pong { actor: a } {
                emit Ping { actor: a }
            }
        }
    "#;
    let err =
        dsl_compiler::compile(src).expect_err("should reject indirect emission cycle");
    match err {
        CompileError::Resolve(ResolveError::NotGpuEmittable {
            construct,
            ..
        }) => {
            assert!(construct.contains("indirect recursion"));
        }
        other => panic!("expected NotGpuEmittable(indirect recursion); got: {other:?}"),
    }
}
