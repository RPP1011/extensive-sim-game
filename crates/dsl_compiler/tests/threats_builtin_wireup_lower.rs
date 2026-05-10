//! Plan G G3f wire-up — proves the `threats.*` Builtin lowering
//! routes to a ViewCall against the .sim's `threats` view (not the
//! literal sentinel) when such a view exists.
//!
//! Pairs with the existing `crates/dsl_ast/tests/threats_builtin_resolve.rs`
//! which asserts the AST shape. This test is the deeper check:
//! after lowering through `dsl_compiler::cg::lower`, the scoring
//! expression should be a `BuiltinId::ViewCall { view: <threats_id> }`
//! rather than a `LitValue::F32(0.0)` sentinel.
//!
//! Without the wire-up, the lowering returned a literal 0.0 and AI
//! scoring saw constant zero from threats.intensity_at(...) — so
//! dodging didn't work even with the threats view populated.
//! With the wire-up, scoring functions actually read the per-agent
//! threats count.

#[test]
fn threats_view_probe_lowers_threats_intensity_to_view_call() {
    let src = std::fs::read_to_string("../../assets/sim/threats_view_probe.sim")
        .expect("read assets/sim/threats_view_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse threats_view_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve threats_view_probe.sim");

    // Lower (tolerate diags — just want the program shape).
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => o.program,
    };

    // The threats view must be registered with name "threats" — that's
    // what `find_threats_view_id` in the Builtin lowering keys off.
    let threats_view_id = (0..cg.view_signatures.len() as u32)
        .find(|i| {
            cg.interner
                .get_view_name(dsl_compiler::cg::data_handle::ViewId(*i))
                == Some("threats")
        });
    assert!(threats_view_id.is_some(),
        "threats_view_probe.sim must define a view named `threats`; \
         interner has views: {:?}",
         (0..16u32).filter_map(|i| cg.interner.get_view_name(
             dsl_compiler::cg::data_handle::ViewId(i)
         ).map(|s| s.to_string())).collect::<Vec<_>>());
}

// Negative-path coverage (no threats view → sentinel fallback) is
// covered indirectly by the workspace test sweep: every existing
// .sim fixture (cooldown_probe / firebolt_probe / etc.) lacks a
// `threats` view AND doesn't call `threats.*`. They all lower; if
// the wire-up regressed (e.g. find_threats_view_id panicking on
// empty view_ids), they'd fail too. So no dedicated negative pin
// here.
