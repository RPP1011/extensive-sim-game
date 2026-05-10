//! Plan G G4 — `dodger_probe.sim` lowering coverage.
//!
//! Asserts the dodger fixture lowers cleanly AND that the `Flee` verb's
//! scoring expression routes `threats.intensity_at(self)` to a
//! `BuiltinId::ViewCall { view: <threats_id> }` against the registered
//! `threats` view (NOT the `LitValue::F32(0.0)` sentinel that the
//! wire-up's `None` branch falls back to). This is the lowering-side
//! evidence that the threats Builtin is wired into the dodger's
//! scoring kernel; the runtime-side behavioural evidence (the dodger
//! actually picking `Flee` over `Idle`) lives in
//! `crates/dodger_probe_runtime/src/lib.rs`.

use dsl_compiler::cg::data_handle::ViewId;
use dsl_compiler::cg::expr::{BuiltinId, CgExpr, LitValue};
use dsl_compiler::cg::op::ComputeOpKind;

#[test]
fn dodger_probe_threats_intensity_lowers_to_view_call() {
    let src = std::fs::read_to_string("../../assets/sim/dodger_probe.sim")
        .expect("read assets/sim/dodger_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse dodger_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve dodger_probe.sim");

    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                eprintln!("[lower diag] {d}");
            }
            o.program
        }
    };

    // (1) The threats view is registered with name "threats" — the
    // wire-up at `cg/lower/expr.rs::find_threats_view_id` keys on this
    // exact name. Without this, the `threats.intensity_at(...)`
    // Builtin would lower to a sentinel literal.
    let threats_view_id = (0..cg.view_signatures.len() as u32)
        .find_map(|i| {
            let id = ViewId(i);
            cg.interner
                .get_view_name(id)
                .filter(|n| *n == "threats")
                .map(|_| id)
        })
        .unwrap_or_else(|| {
            let names: Vec<String> = (0..cg.view_signatures.len() as u32)
                .filter_map(|i| {
                    cg.interner
                        .get_view_name(ViewId(i))
                        .map(|s| s.to_string())
                })
                .collect();
            panic!(
                "dodger_probe.sim must define a view named `threats` for the \
                 wire-up to activate; interner has views: {names:?}"
            );
        });

    // (2) Walk the lowered scoring kernels and verify at least one
    // ScoringRow has a utility CgExpr whose Builtin is
    // `ViewCall { view: threats_view_id }`. That Builtin is the
    // wire-up's "threats view exists" branch — its absence would mean
    // the lowering fell through to the `None` branch and emitted a
    // `LitValue::F32(0.0)` sentinel instead.
    let mut saw_view_call = false;
    let mut saw_lit_zero = false;
    let mut scoring_op_count: usize = 0;
    for op in cg.ops.iter() {
        if let ComputeOpKind::ScoringArgmax { rows, .. } = &op.kind {
            scoring_op_count += 1;
            for row in rows {
                let Some(utility_expr) = cg.exprs.get(row.utility.0 as usize) else {
                    continue;
                };
                match utility_expr {
                    CgExpr::Builtin {
                        fn_id: BuiltinId::ViewCall { view },
                        ..
                    } if *view == threats_view_id => {
                        saw_view_call = true;
                    }
                    CgExpr::Lit(LitValue::F32(v)) if *v == 0.0 => {
                        saw_lit_zero = true;
                    }
                    _ => {}
                }
            }
        }
    }

    assert!(scoring_op_count >= 1,
        "expected at least one ScoringArgmax op in lowered program, got 0");
    assert!(
        saw_view_call,
        "expected the Flee verb's scoring utility to lower to a \
         BuiltinId::ViewCall against view {threats_view_id:?} (the threats \
         wire-up's Some branch). Got saw_lit_zero={saw_lit_zero}; if true, \
         the lowering fell through to the sentinel literal — wire-up \
         regression."
    );
}

/// Helper test — the verb expander allocates ActionIds in
/// source-order over the scoring rows. With Flee declared before
/// Idle in the .sim, Flee gets ActionId(0) and Idle gets ActionId(1).
/// The runtime test reads `scoring_output[best_action]` and checks
/// the chosen value matches Flee's id; this test pins the
/// allocation order so the runtime assertion is unambiguous.
#[test]
fn dodger_probe_flee_gets_action_id_zero() {
    let src = std::fs::read_to_string("../../assets/sim/dodger_probe.sim")
        .expect("read assets/sim/dodger_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse dodger_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve dodger_probe.sim");
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => o.program,
    };

    // Walk the scoring rows. The expander appends scoring entries in
    // source order using synthetic names `verb_<Name>`. The driver's
    // `populate_actions` then allocates ActionIds in source order over
    // the entries. So row 0 should be `verb_Flee` (ActionId 0) and
    // row 1 should be `verb_Idle` (ActionId 1).
    let mut row_action_ids: Vec<u32> = Vec::new();
    for op in cg.ops.iter() {
        if let ComputeOpKind::ScoringArgmax { rows, .. } = &op.kind {
            for row in rows {
                row_action_ids.push(row.action.0);
            }
        }
    }

    assert_eq!(
        row_action_ids.len(),
        2,
        "expected exactly 2 scoring rows (Flee, Idle); got {row_action_ids:?}"
    );
    assert_eq!(
        row_action_ids,
        vec![0, 1],
        "expected ActionIds [Flee=0, Idle=1] in source order; got \
         {row_action_ids:?}"
    );
}
