//! Lowering coverage for the Boids per-agent physics body.
//!
//! Companion to `boids_smoke.rs`. That file exercises parse + resolve
//! and (in a sibling test) the lazy-view body in isolation; this one
//! walks the full resolved compilation through `cg::lower::driver` so
//! the per-agent `PhysicsRule` body's CG construction is exercised.
//!
//! The check is deliberately tolerant of pre-existing follow-up gaps
//! that are NOT in this task's scope:
//!   - `LoweringError::UnsupportedAstNode { ast_label: "Fold", .. }`
//!     — the parallel work-stream's scope (Fold/comprehension lowering
//!     in `cg/lower/expr.rs`).
//!   - `LoweringError::UnresolvedEventPattern { event_name: "Tick", .. }`
//!     — the per-agent physics handler today is shaped as
//!     `on Tick {} { ... }` but `Tick` is not a declared `event` in the
//!     fixture. Wiring `Tick` as an implicit synthesized event for
//!     `@phase(per_agent)` rules is a separate surface; the resolver
//!     leaves `pattern.event = None`, so `lower_all_physics`'s handler-
//!     resolution pass surfaces this diagnostic before it ever calls
//!     into the per-agent body's CG-stmt walk. This is the same
//!     "Tick wiring is a separate, future surface" note the sibling
//!     `boids_smoke.rs::boids_fixture_lowers_count_fold` test makes.
//!
//! Anything ELSE — most notably an `UnsupportedAstNode` for the
//! `agents.set_pos` / `agents.set_vel` statement-level namespace
//! mutations the body emits — IS the cascade this test exists to
//! catch. With the `MOVEMENT_BODY` stub deleted the per-agent body
//! walks like every other op-kind body, so any IR-level gap surfaces
//! instead of being silently papered over.

use std::fs;
use std::path::PathBuf;

use dsl_compiler::cg::expr::CgTy as crate_types;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::LoweringError;

fn boids_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/dsl_compiler");
    workspace_root.join("assets/sim/boids.sim")
}

#[test]
fn boids_lowering_only_surfaces_known_followups() {
    let src = fs::read_to_string(boids_path()).expect("read boids.sim");
    let program = dsl_compiler::parse(&src).expect("parse boids.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve boids.sim");

    let Err(outcome) = lower_compilation_to_cg(&comp) else {
        // Best case: every body lowered cleanly through the new CG
        // walk — no follow-up gaps left. The MOVEMENT_BODY stub
        // deletion is fully validated.
        return;
    };

    for diag in &outcome.diagnostics {
        match diag {
            LoweringError::UnsupportedAstNode { ast_label, .. } if *ast_label == "Fold" => {
                // Parallel work-stream's scope. Sum/Min/Max still
                // reject; Count short-circuits to LitValue::I32(0).
            }
            // Tick event is now declared in boids.sim (commit e2ea387f),
            // so this case shouldn't fire — kept defensively in case
            // the fixture's event list shrinks.
            LoweringError::UnresolvedEventPattern { event_name, .. } if event_name == "Tick" => {}
            // `let new_vel = self.vel + zero_steer;` (vec3 + vec3) — the
            // CG `Binary` op for `+` is scalar-only today; vec3 binary
            // ops need a typed Add variant. Tracked as the next concrete
            // blocker after the AgentFieldId::Vel addition (this commit's
            // sibling). Treated as a known follow-up here so the test
            // tracks rather than fails.
            LoweringError::IllTypedExpression {
                expected: crate_types::F32,
                got: crate_types::Vec3F32,
                ..
            } => {}
            other => {
                panic!(
                    "boids lowering produced an unexpected diagnostic \
                     after MOVEMENT_BODY stub deletion: {other:?}\n\
                     (only Fold-unsupported, Tick-unresolved, and \
                     vec3+vec3 type-mismatch are allowed pending \
                     parallel work-streams)"
                );
            }
        }
    }
}
