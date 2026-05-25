//! Probe: does a `where`-guard on a `@materialized` view fold handler
//! reach the emitted kernel?
//!
//! `crowd_navigation.sim`'s `stuck_ticks` view has TWO `on Tick {}`
//! handlers distinguished ONLY by a `where` guard on the observer's
//! field (`w.last_progress < thresh` vs `>= thresh`). If the guard is
//! dropped during resolve→lower, both handlers fold unconditionally and
//! the view is silently wrong. This test pins whether the guard's
//! comparison survives into the WGSL fold kernel.

use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};
use dsl_compiler::cg::emit::emit_cg_program;

fn compile_to_wgsl(src: &str) -> std::collections::BTreeMap<String, String> {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => o.program,
    };
    let schedule = synthesize_schedule(&cg, ScheduleStrategy::Default);
    let arts = emit_cg_program(&schedule.schedule, &cg).expect("emit");
    arts.wgsl_files
}

// Modeled on `dsl_stress_coverage.sim`'s `damage_taken` (a known-good
// keyed materialized view), but the handler carries an ADDITIONAL
// genuine guard `a > config.probe.thresh` beyond the keying restatement
// `t == target`. The keyed accumulate must still emit (proving the path
// is well-formed); the question is whether the `a > thresh` comparison
// survives into the kernel.
const GUARDED_FOLD_SIM: &str = r#"
event Damaged { source: AgentId, target: AgentId, amount: f32 }

entity Probe : Agent { }

config probe { thresh: f32 = 10.0 }

@materialized(on_event = [Damaged])
view big_hits(target: Agent) -> f32 {
  initial: 0.0,
  on Damaged { target: t, amount: a } where t == target && a > config.probe.thresh { self += 1.0 }
  clamp: [0.0, 1000.0],
}
"#;

// PENDING G1 (see docs/superpowers/specs/2026-05-24-dsl-as-engine-scorecard.md).
// This pins the silent-drop bug as a regression anchor: it currently FAILS
// (the guard is dropped). Un-ignore when the fold `where`-guard is honored
// (carry `where_clause` into FoldHandlerIR → lower → guard-wrap emit). The
// fix is non-trivial (central `ComputeOpKind::ViewFold` ripple / side-table)
// and crowd_navigation.sim's stuck_ticks depends on it being honored, so it
// is scheduled as its own close-out rather than an inline edit.
#[test]
#[ignore = "G1 pending: fold where-guard silently dropped; see scorecard"]
fn fold_where_guard_reaches_kernel() {
    let wgsl = compile_to_wgsl(GUARDED_FOLD_SIM);
    let fold = wgsl
        .get("fold_big_hits.wgsl")
        .unwrap_or_else(|| panic!("fold_big_hits kernel emitted; files: {:?}", wgsl.keys().collect::<Vec<_>>()));
    eprintln!("==== fold_big_hits.wgsl ====\n{fold}\n=========================");

    // The keyed accumulate must be present (path is well-formed).
    assert!(
        fold.contains("atomic") || fold.contains("view_storage_primary"),
        "expected a real keyed accumulate in the fold kernel:\n{fold}"
    );
    // The genuine guard `a > config.probe.thresh` (the amount read /
    // the baked 10.0 threshold) must gate the accumulate. If absent,
    // the guard was silently dropped and EVERY Damaged event counts.
    assert!(
        fold.contains("10.0") || fold.contains("amount"),
        "where-guard comparison (a > thresh) absent from fold kernel — guard silently dropped:\n{fold}"
    );
}
