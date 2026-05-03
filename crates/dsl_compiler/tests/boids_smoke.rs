//! Smoke test for `assets/sim/boids.sim` — confirms the post-wolf-sim-wipe
//! Boids fixture parses and resolves through the dsl_ast surface.
//!
//! What this verifies:
//!   - `entity Boid : Agent { pos: vec3, vel: vec3 }` parses.
//!   - `spatial_query nearby_other(self, candidate) = ...` parses + resolves
//!     (Phase 7 Task 4 surface).
//!   - Lazy view declarations using `sum`/`count` comprehensions over
//!     `agents` parse + resolve.
//!   - `physics MoveBoid @phase(per_agent) { on Tick { ... } }` parses.
//!
//! What this does NOT verify:
//!   - End-to-end compilation through `cg::*` (the per-agent physics body
//!     emit is currently a hardcoded WGSL stub; lowering the body
//!     statements is a follow-up).
//!   - Runtime correctness — there's no engine crate today to consume
//!     the emitted output.

use std::fs;
use std::path::PathBuf;

fn boids_path() -> PathBuf {
    // From `crates/dsl_compiler/tests/`, walk up to the workspace root.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/dsl_compiler");
    workspace_root.join("assets/sim/boids.sim")
}

#[test]
fn boids_fixture_file_exists() {
    let path = boids_path();
    assert!(
        path.exists(),
        "expected boids fixture at {}",
        path.display()
    );
}

#[test]
fn boids_fixture_parses() {
    let src = fs::read_to_string(boids_path()).expect("read boids.sim");
    let program = dsl_compiler::parse(&src).expect("parse boids.sim");

    // Minimum: at least one entity, one spatial_query, one physics decl,
    // one config block, plus the four views (count + centroid + avg_velocity
    // + separation_force).
    use dsl_compiler::Decl;
    let entities = program
        .decls
        .iter()
        .filter(|d| matches!(d, Decl::Entity(_)))
        .count();
    let spatial_queries = program
        .decls
        .iter()
        .filter(|d| matches!(d, Decl::SpatialQuery(_)))
        .count();
    let physics = program
        .decls
        .iter()
        .filter(|d| matches!(d, Decl::Physics(_)))
        .count();
    let configs = program
        .decls
        .iter()
        .filter(|d| matches!(d, Decl::Config(_)))
        .count();
    let views = program
        .decls
        .iter()
        .filter(|d| matches!(d, Decl::View(_)))
        .count();

    assert_eq!(entities, 1, "Boid entity");
    assert_eq!(spatial_queries, 1, "nearby_other spatial_query");
    assert_eq!(physics, 1, "MoveBoid physics rule");
    assert_eq!(configs, 1, "flock config block");
    assert_eq!(views, 1, "neighbor_count view");
}

#[test]
fn boids_fixture_resolves() {
    let src = fs::read_to_string(boids_path()).expect("read boids.sim");
    let program = dsl_compiler::parse(&src).expect("parse boids.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve boids.sim");

    assert_eq!(comp.entities.len(), 1);
    assert_eq!(comp.entities[0].name, "Boid");

    assert_eq!(comp.spatial_queries.len(), 1);
    assert_eq!(comp.spatial_queries[0].name, "nearby_other");

    assert_eq!(comp.views.len(), 1);
    assert_eq!(comp.views[0].name, "neighbor_count");

    assert_eq!(comp.physics.len(), 1);
    assert_eq!(comp.physics[0].name, "MoveBoid");

    assert_eq!(comp.configs.len(), 1);
    assert_eq!(comp.configs[0].name, "flock");
}

/// Phase 7 follow-up: the `neighbor_count` lazy view body in
/// `assets/sim/boids.sim` is a `count(other in agents where ...)`
/// comprehension that the resolver shapes as
/// `IrExpr::Fold { kind: FoldKind::Count, .. }`. With the N²-fold
/// CgStmt::ForEachAgent path lit up alongside this fixture, the
/// body lowers to a real ForEachAgent stmt + ReadLocal expression.
///
/// We exercise the view body directly rather than going through the
/// full driver because the physics-handler path bounces off
/// `UnresolvedEventPattern { event_name: "Tick" }` first (Tick wiring
/// is a separate, future surface). Direct `lower_expr` invocation
/// pins the Fold contract without coupling to that unrelated layer —
/// but the body references `config.flock.perception_radius`, so we
/// register the matching ConfigConstId by hand before lowering.
#[test]
fn boids_fixture_lowers_count_fold() {
    use dsl_ast::ir::{NamespaceId, ViewBodyIR};
    use dsl_compiler::cg::data_handle::ConfigConstId;
    use dsl_compiler::cg::lower::expr::{lower_expr, LoweringCtx};
    use dsl_compiler::cg::program::CgProgramBuilder;

    let src = fs::read_to_string(boids_path()).expect("read boids.sim");
    let program = dsl_compiler::parse(&src).expect("parse boids.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve boids.sim");

    let view = comp
        .views
        .iter()
        .find(|v| v.name == "neighbor_count")
        .expect("neighbor_count view present");
    let body = match &view.body {
        ViewBodyIR::Expr(e) => e,
        ViewBodyIR::Fold { .. } => panic!("neighbor_count is a lazy view; body must be Expr"),
    };

    let mut builder = CgProgramBuilder::new();
    let mut ctx = LoweringCtx::new(&mut builder);

    // Register every config const the boids fixture references in source
    // order — the count fold's body reads `config.flock.perception_radius`,
    // and other views/rules in the fixture read other fields in the same
    // block; the standalone driver-less harness has to populate the
    // registry by hand.
    for (i, cfg) in comp.configs.iter().enumerate() {
        for (j, fld) in cfg.fields.iter().enumerate() {
            // Stable id: walk-order index across blocks. For a single-
            // block fixture (today's boids.sim) the block index is
            // zero so per-field id is `j`.
            let _ = i;
            let key = format!("{}.{}", cfg.name, fld.name);
            ctx.register_config_const(NamespaceId::Config, key, ConfigConstId(j as u32));
        }
    }

    let result = lower_expr(body, &mut ctx);

    assert!(
        result.is_ok(),
        "expected Fold-bearing view body to lower; got: {:?}",
        result.err()
    );
}

