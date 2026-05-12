//! Pin: `self = <expr>` (assignment) in view fold bodies lowers to
//! `atomicStore` in WGSL emit, while `self += <expr>` continues to
//! lower to its operator-specific atomic primitive (`atomicAdd` for
//! u32 add accumulator).
//!
//! Closes the gap surfaced by `dsl_stress_coverage.sim` and
//! `crowd_navigation.sim`: pre-fix the lowerer rejected `=` as
//! `LoweringError::UnsupportedFoldOperator` and the build.rs printed
//! `[<sim> lower diag] view #N self-update operator = not supported
//! by CG IR; only += is lowered today` while silently skipping the
//! handler. Post-fix `=` flows through as `ViewFoldOp::Set` and
//! becomes a `atomicStore(&view_storage_primary[idx], ...)` in the
//! emitted WGSL.
//!
//! Determinism note (P11): atomicStore is last-writer-wins. The
//! lowering admits the construct because every shipping use today is
//! either an idempotent constant set (`self = 255` on `Observed` —
//! every matching event writes the same constant) or a single-writer-
//! per-slot-per-tick recompute (`self = mean(...)` on `Tick`). The
//! DSL author owns determinism for fold-set semantics.

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::program::CgProgram;

fn compile_inline(src: &str) -> (CgProgram, EmittedArtifacts) {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(outcome) => {
            for diag in &outcome.diagnostics {
                eprintln!("[lower diagnostic] {diag}");
            }
            panic!(
                "lower_compilation_to_cg returned {} diagnostic(s) — see stderr above",
                outcome.diagnostics.len()
            );
        }
    };
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit");
    (cg, art)
}

fn fold_wgsl<'a>(art: &'a EmittedArtifacts, view_name: &str) -> &'a str {
    let needle = format!("fold_{view_name}");
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(&needle) && name.ends_with(".wgsl"))
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| {
            let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
            panic!("no kernel matched {needle:?}; emitted kernels: {names:?}")
        })
}

/// `self = <const>` on a u8 pair-keyed view (the
/// `dsl_stress_coverage.sim::confidence` shape) emits a single
/// `atomicStore` — no CAS retry loop, no atomicAdd, no atomicOr.
#[test]
fn self_assign_const_lowers_to_atomic_store() {
    let src = r#"
event Tick { }
event Observed {
  observer: AgentId,
  subject:  AgentId,
}

@phase(per_agent)
physics PingObs {
  on Tick {} where (self.alive) {
    emit Observed { observer: self, subject: self }
  }
}

@materialized(on_event = [Observed])
view confidence(observer: Agent, subject: Agent) -> u32 {
  initial: 0,
  on Observed { observer: o, subject: s } where o == observer && s == subject {
    self = 255
  }
}
"#;
    let (_cg, art) = compile_inline(src);
    let body = fold_wgsl(&art, "confidence");
    assert!(
        body.contains("atomicStore(&view_storage_primary["),
        "fold_confidence must emit `atomicStore` for `self = 255` semantics; got body:\n{body}",
    );
    assert!(
        !body.contains("atomicAdd"),
        "fold_confidence must NOT emit atomicAdd for `self =` (assignment != add accumulator); got body:\n{body}",
    );
    assert!(
        !body.contains("atomicOr"),
        "fold_confidence must NOT emit atomicOr for `self =` (assignment != bit-OR accumulator); got body:\n{body}",
    );
    assert!(
        !body.contains("atomicCompareExchangeWeak"),
        "fold_confidence must NOT emit a CAS loop for `self =` — atomicStore is the correct primitive; got body:\n{body}",
    );
    // The constant 255 must reach the emitted WGSL verbatim (literal
    // suffix `u` because the view result type is u32).
    assert!(
        body.contains("(255u)"),
        "fold_confidence must store the literal 255u; got body:\n{body}",
    );
}

/// Companion regression — confirm `+=` continues to route through
/// `atomicAdd` (u32 add accumulator) so the new `Set` arm did not
/// disturb the post-Gap-C operator branch in `wgsl_body.rs`.
#[test]
fn self_plus_equals_still_lowers_to_atomic_add_for_u32() {
    let src = r#"
event Tick { }
event Progressed {
  who: AgentId,
}

@phase(per_agent)
physics PingProgress {
  on Tick {} where (self.alive) {
    emit Progressed { who: self }
  }
}

@materialized(on_event = [Progressed])
view progress(actor: Agent) -> u32 {
  initial: 0,
  on Progressed { who: a } where a == actor {
    self += 1
  }
}
"#;
    let (_cg, art) = compile_inline(src);
    let body = fold_wgsl(&art, "progress");
    assert!(
        body.contains("atomicAdd(&view_storage_primary["),
        "fold_progress must emit `atomicAdd` for `self += 1u` (operator-aware u32 add accumulator); got body:\n{body}",
    );
    assert!(
        !body.contains("atomicStore"),
        "fold_progress must NOT emit atomicStore for `self +=` (add accumulator != assignment); got body:\n{body}",
    );
    assert!(
        !body.contains("atomicOr"),
        "fold_progress must NOT emit atomicOr for `self += 1u` post-Gap-C-fix; got body:\n{body}",
    );
}

/// `self |=` continues to route through `atomicOr` — the bit-OR
/// accumulator path used by Theory-of-Mind belief folds. Pinned next
/// to the Add and Set arms so future operator additions can't
/// silently regress the Or branch either.
#[test]
fn self_or_equals_still_lowers_to_atomic_or_for_u32() {
    let src = r#"
event Tick { }
event BeliefAcquired {
  observer: AgentId,
  fact_bit: u32,
}

@phase(per_agent)
physics PingBelief {
  on Tick {} where (self.alive) {
    emit BeliefAcquired { observer: self, fact_bit: 1 }
  }
}

@materialized(on_event = [BeliefAcquired])
view beliefs(o: Agent) -> u32 {
  initial: 0,
  on BeliefAcquired { observer: ob, fact_bit: b } where ob == o {
    self |= b
  }
}
"#;
    let (_cg, art) = compile_inline(src);
    let body = fold_wgsl(&art, "beliefs");
    assert!(
        body.contains("atomicOr(&view_storage_primary["),
        "fold_beliefs must emit `atomicOr` for `self |= b`; got body:\n{body}",
    );
    assert!(
        !body.contains("atomicStore"),
        "fold_beliefs must NOT emit atomicStore for `self |=`; got body:\n{body}",
    );
    assert!(
        !body.contains("atomicAdd"),
        "fold_beliefs must NOT emit atomicAdd for `self |=`; got body:\n{body}",
    );
}
