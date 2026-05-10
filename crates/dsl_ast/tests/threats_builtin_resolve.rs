//! Plan G G3f (2026-05-09) — surface-form tests for the `threats.*`
//! scoring primitives.
//!
//! Asserts the parser → resolver pipeline rewrites the four
//! `threats.<method>(...)` shapes (and their `threats::<method>(...)`
//! parser-flattened siblings) into the corresponding
//! [`Builtin::Threats*`] variants. The CG lowering of the same
//! constructs is covered downstream by `dsl_compiler` tests; this
//! test just pins the Builtin surface — the load-bearing piece for
//! G3f. The threats materialised view (G3g, future) wires the
//! per-cell walk that produces the real aggregates.
//!
//! See `docs/plans/g3_threats_view_design.md` for the full primitive
//! catalog.

use dsl_ast::compile;
use dsl_ast::ir::{Builtin, IrExpr, IrStmt};

/// Wrap a single statement in a `physics` body so we have a complete
/// `.sim` source the parser + resolver will accept.
fn compile_physics_body(stmt_src: &str) -> dsl_ast::Compilation {
    let src = format!(
        "event Tick {{ }}\n\
         entity Probe : Agent {{ pos: vec3, }}\n\
         physics P @phase(per_agent) {{\n\
           on Tick {{}} where (self.alive) {{\n\
             {stmt_src}\n\
           }}\n\
         }}\n"
    );
    compile(&src).unwrap_or_else(|e| {
        panic!("compile failed for body `{stmt_src}`:\n{src}\nerror: {e}")
    })
}

/// Pull the first `Let` statement out of the resolved physics body
/// and return the let's value expression.
fn first_let_value(comp: &dsl_ast::Compilation) -> &IrExpr {
    let physics = comp.physics.first().expect("expected one physics rule");
    let handler = physics.handlers.first().expect("expected one handler");
    let stmt = handler.body.first().expect("expected one body stmt");
    match stmt {
        IrStmt::Let { value, .. } => &value.kind,
        other => panic!("expected first body stmt to be Let, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Dotted form: `threats.<method>(self)` — one test per method.
// ---------------------------------------------------------------------------

#[test]
fn threats_in_zone_dotted_form_resolves_to_builtin() {
    let comp = compile_physics_body("let x = threats.in_zone(self);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::ThreatsInZone, args) => {
            assert_eq!(args.len(), 1, "threats.in_zone takes one arg (self)");
        }
        other => panic!(
            "expected BuiltinCall(ThreatsInZone, _), got {other:?}"
        ),
    }
}

#[test]
fn threats_intensity_at_dotted_form_resolves_to_builtin() {
    let comp = compile_physics_body("let x = threats.intensity_at(self.pos);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::ThreatsIntensityAt, args) => {
            assert_eq!(args.len(), 1, "threats.intensity_at takes one arg (pos)");
        }
        other => panic!(
            "expected BuiltinCall(ThreatsIntensityAt, _), got {other:?}"
        ),
    }
}

#[test]
fn threats_nearest_dotted_form_resolves_to_builtin() {
    let comp = compile_physics_body("let x = threats.nearest(self);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::ThreatsNearest, args) => {
            assert_eq!(args.len(), 1, "threats.nearest takes one arg (self)");
        }
        other => panic!(
            "expected BuiltinCall(ThreatsNearest, _), got {other:?}"
        ),
    }
}

#[test]
fn threats_dir_away_from_nearest_dotted_form_resolves_to_builtin() {
    let comp =
        compile_physics_body("let x = threats.dir_away_from_nearest(self);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::ThreatsDirAwayFromNearest, args) => {
            assert_eq!(
                args.len(),
                1,
                "threats.dir_away_from_nearest takes one arg (self)"
            );
        }
        other => panic!(
            "expected BuiltinCall(ThreatsDirAwayFromNearest, _), got {other:?}"
        ),
    }
}

// ---------------------------------------------------------------------------
// `::` form mirror — `threats::<method>(self)` parses identically.
// ---------------------------------------------------------------------------

#[test]
fn threats_in_zone_double_colon_form_resolves_to_builtin() {
    let comp = compile_physics_body("let x = threats::in_zone(self);");
    match first_let_value(&comp) {
        IrExpr::BuiltinCall(Builtin::ThreatsInZone, args) => {
            assert_eq!(args.len(), 1);
        }
        other => panic!(
            "expected BuiltinCall(ThreatsInZone, _), got {other:?}"
        ),
    }
}
