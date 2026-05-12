//! Pin: `self -= <expr>` (subtract accumulator) in view fold bodies
//! lowers to operator-aware atomic primitives in WGSL emit. u32 views
//! emit a single `atomicSub`; f32 views emit a CAS+sub loop (mirrors
//! the CAS+add shape used for `self += <expr>` on f32 views).
//!
//! Closes the gap surfaced as **T1** in
//! `docs/architecture/gaps_observed.md`: pre-fix the lowerer rejected
//! `-=` as `LoweringError::UnsupportedFoldOperator` and the build.rs
//! printed `[<sim> lower diag] view #N self-update operator -= not
//! supported by CG IR; only += / |= / = are lowered today` while
//! silently skipping the handler. The trade_caravans
//! `inventory(merchant, good) -> f32` view's `on Sold { ... } { self
//! -= 1.0 }` arm (`assets/sim/trade_caravans.sim`) was the first
//! shipping consumer; without this fix every Sold event silently
//! dropped its inventory decrement.
//!
//! Determinism note (P11): `atomicSub` on u32 is commutative +
//! associative under modular arithmetic — the same guarantee `+= 1u`
//! enjoys via `atomicAdd`. The f32 CAS+sub loop retries on the weak-
//! CAS failure path, matching the pre-existing CAS+add f32 path's
//! determinism contract.

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

/// `self -= <const>` on a u32 view emits a single `atomicSub` — no
/// CAS retry loop, no atomicAdd, no atomicStore. u32 atomicSub is
/// commutative + associative under modular arithmetic so P11 is
/// trivially satisfied without a CAS spin.
#[test]
fn self_sub_lowers_to_atomic_sub_for_u32() {
    let src = r#"
event Tick { }
event Withdrew {
  who: AgentId,
}

@phase(per_agent)
physics PingWithdraw {
  on Tick {} where (self.alive) {
    emit Withdrew { who: self }
  }
}

@materialized(on_event = [Withdrew])
view balance(actor: Agent) -> u32 {
  initial: 100,
  on Withdrew { who: a } where a == actor {
    self -= 1
  }
}
"#;
    let (_cg, art) = compile_inline(src);
    let body = fold_wgsl(&art, "balance");
    assert!(
        body.contains("atomicSub(&view_storage_primary["),
        "fold_balance must emit `atomicSub` for `self -= 1u` (operator-aware u32 sub accumulator); got body:\n{body}",
    );
    assert!(
        !body.contains("atomicAdd"),
        "fold_balance must NOT emit atomicAdd for `self -=` (sub != add); got body:\n{body}",
    );
    assert!(
        !body.contains("atomicStore"),
        "fold_balance must NOT emit atomicStore for `self -=` (sub != assign); got body:\n{body}",
    );
    assert!(
        !body.contains("atomicCompareExchangeWeak"),
        "fold_balance must NOT emit a CAS loop for `self -= 1u` on u32 — atomicSub is the right primitive; got body:\n{body}",
    );
}

/// `self -= <const>` on an f32 view emits a CAS+sub loop. Mirrors the
/// trade_caravans `inventory(merchant, good) -> f32` view's `self -=
/// 1.0` arm on `Sold`. WGSL has no native `atomicSub` for f32 (the
/// storage binding is `array<atomic<u32>>`), so the emit shape is the
/// CAS+add f32 fallthrough with `+` swapped for `-`.
#[test]
fn self_sub_lowers_to_cas_sub_loop_for_f32() {
    let src = r#"
event Tick { }
event Drained {
  who: AgentId,
  amount: f32,
}

@phase(per_agent)
physics PingDrain {
  on Tick {} where (self.alive) {
    emit Drained { who: self, amount: 1.0 }
  }
}

@materialized(on_event = [Drained])
view reservoir(actor: Agent) -> f32 {
  initial: 100.0,
  on Drained { who: a, amount: x } where a == actor {
    self -= x
  }
}
"#;
    let (_cg, art) = compile_inline(src);
    let body = fold_wgsl(&art, "reservoir");
    assert!(
        body.contains("atomicCompareExchangeWeak(&view_storage_primary["),
        "fold_reservoir must emit a CAS loop for `self -= x` on f32 (no native atomicSub for f32 in WGSL); got body:\n{body}",
    );
    assert!(
        body.contains("bitcast<f32>(old) -"),
        "fold_reservoir's CAS body must subtract the rhs (`bitcast<f32>(old) - (rhs)`); got body:\n{body}",
    );
    assert!(
        !body.contains("bitcast<f32>(old) +"),
        "fold_reservoir must NOT emit `+` inside the CAS body — `self -=` is subtract, not add; got body:\n{body}",
    );
    assert!(
        !body.contains("atomicSub"),
        "fold_reservoir must NOT emit native atomicSub on an f32 view (storage binding is array<atomic<u32>>); got body:\n{body}",
    );
    assert!(
        !body.contains("atomicAdd"),
        "fold_reservoir must NOT emit atomicAdd for `self -= x`; got body:\n{body}",
    );
    assert!(
        !body.contains("atomicStore"),
        "fold_reservoir must NOT emit atomicStore for `self -= x` (sub != assign); got body:\n{body}",
    );
}

/// Companion regression — confirm `+=` continues to route through
/// `atomicAdd` (u32 add accumulator) so the new `Sub` arm did not
/// disturb the operator branch in `wgsl_body.rs`. Mirrors the
/// equivalent guard in `view_fold_self_assign_emit.rs`.
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
        "fold_progress must emit `atomicAdd` for `self += 1u` post-Sub-fix; got body:\n{body}",
    );
    assert!(
        !body.contains("atomicSub"),
        "fold_progress must NOT emit atomicSub for `self += 1u`; got body:\n{body}",
    );
    assert!(
        !body.contains("atomicStore"),
        "fold_progress must NOT emit atomicStore for `self +=`; got body:\n{body}",
    );
}
