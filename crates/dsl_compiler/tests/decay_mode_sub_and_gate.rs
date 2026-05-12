//! Additive coverage for the `@decay(mode = sub|mul, by = N, gate = MaskName)`
//! grammar extension. Pins the surface contract so a future refactor of
//! `lower_decay_hint` can't silently regress backward compatibility.
//!
//! The legacy `@decay(rate = R, per = tick)` shape still resolves and lowers
//! identically (covered by `decay_rate_bounds.rs`); this file exercises only
//! the new arguments.

const TEMPLATE_NO_GATE: &str = r#"
event Tick { }

@replayable
@gpu_amenable
event Hit {
  target: AgentId,
}

entity Particle : Agent {
  pos: vec3,
  vel: vec3,
}

@materialized(on_event = [Hit])
@decay(per = tick, MODE_AND_BY_PLACEHOLDER)
view hits(p: Agent) -> f32 {
  initial: 0.0,
  on Hit { target: agent } where agent == p { self += 1.0 }
  clamp: [0.0, 1000000.0],
}

physics Tickle @phase(per_agent) {
  on Tick {} {
    emit Hit { target: self }
  }
}
"#;

/// Variant template that adds a `mask Strike when self.alive` decl
/// up-front so `gate = Strike` resolves cleanly. The mask is the
/// no-arg "self-only" shape (`ActionHeadShape::None`) so the lowering
/// accepts it without a `from` clause — that's the canonical shape
/// the v1 mask grammar reaches for here.
const TEMPLATE_WITH_MASK: &str = r#"
event Tick { }

@replayable
@gpu_amenable
event Hit {
  target: AgentId,
}

entity Particle : Agent {
  pos: vec3,
  vel: vec3,
}

mask Strike when self.alive

@materialized(on_event = [Hit])
@decay(per = tick, MODE_AND_BY_PLACEHOLDER, gate = GATE_PLACEHOLDER)
view hits(p: Agent) -> f32 {
  initial: 0.0,
  on Hit { target: agent } where agent == p { self += 1.0 }
  clamp: [0.0, 1000000.0],
}

physics Tickle @phase(per_agent) {
  on Tick {} {
    emit Hit { target: self }
  }
}
"#;

fn try_resolve_no_gate(mode_and_by: &str) -> Result<(), String> {
    let src = TEMPLATE_NO_GATE.replace("MODE_AND_BY_PLACEHOLDER", mode_and_by);
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e}"))?;
    dsl_ast::resolve::resolve(program)
        .map(|_| ())
        .map_err(|e| format!("resolve: {e}"))
}

fn try_resolve_with_gate(mode_and_by: &str, gate: &str) -> Result<(), String> {
    let src = TEMPLATE_WITH_MASK
        .replace("MODE_AND_BY_PLACEHOLDER", mode_and_by)
        .replace("GATE_PLACEHOLDER", gate);
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e}"))?;
    dsl_ast::resolve::resolve(program)
        .map(|_| ())
        .map_err(|e| format!("resolve: {e}"))
}

// -- mode = mul -----------------------------------------------------------

#[test]
fn mode_mul_with_by_resolves_when_in_range() {
    // Equivalent to `@decay(rate = 0.5, per = tick)`.
    try_resolve_no_gate("mode = mul, by = 0.5").expect("mode = mul, by = 0.5 should resolve");
}

#[test]
fn mode_mul_rejects_by_outside_range() {
    let err = try_resolve_no_gate("mode = mul, by = 1.5")
        .expect_err("by = 1.5 must be rejected (mul mode rate-bound)");
    assert!(err.contains("[0.0, 1.0)"), "expected range error; got: {err}");
}

// -- mode = sub -----------------------------------------------------------

#[test]
fn mode_sub_with_positive_int_by_resolves() {
    try_resolve_no_gate("mode = sub, by = 1").expect("mode = sub, by = 1 should resolve");
    try_resolve_no_gate("mode = sub, by = 5").expect("mode = sub, by = 5 should resolve");
}

#[test]
fn mode_sub_rejects_zero_by() {
    let err = try_resolve_no_gate("mode = sub, by = 0")
        .expect_err("mode = sub, by = 0 is a no-op; must be rejected");
    assert!(err.contains("> 0"), "expected positive-int error; got: {err}");
}

#[test]
fn mode_sub_rejects_float_by() {
    let err = try_resolve_no_gate("mode = sub, by = 1.5")
        .expect_err("mode = sub does not accept float magnitudes");
    assert!(err.contains("positive int"), "expected int-only error; got: {err}");
}

#[test]
fn mode_sub_rejects_missing_by() {
    let err = try_resolve_no_gate("mode = sub")
        .expect_err("mode = sub requires by");
    assert!(err.contains("requires `by"), "expected required-by error; got: {err}");
}

// -- mode = sub + gate ----------------------------------------------------

#[test]
fn gate_resolves_when_mask_exists() {
    try_resolve_with_gate("mode = sub, by = 1", "Strike")
        .expect("gate = Strike should resolve cleanly when the mask is declared");
}

#[test]
fn gate_rejects_unknown_mask_name() {
    let err = try_resolve_with_gate("mode = sub, by = 1", "BogusMask")
        .expect_err("gate = BogusMask must surface an unknown-mask error");
    assert!(
        err.contains("BogusMask") && err.contains("does not match"),
        "expected unknown-mask error; got: {err}"
    );
}

// -- backward compat: legacy `rate = R` keeps working ---------------------

#[test]
fn legacy_rate_form_still_resolves() {
    // The `rate = R` form is exercised exhaustively by
    // `decay_rate_bounds.rs`; this test pins that the new code path
    // doesn't regress the legacy spelling end-to-end.
    let src = TEMPLATE_NO_GATE.replace("MODE_AND_BY_PLACEHOLDER", "rate = 0.5");
    let program = dsl_compiler::parse(&src).expect("parse");
    dsl_ast::resolve::resolve(program).expect("legacy rate = 0.5 should resolve");
}

// -- mutual exclusion: rate vs mode --------------------------------------

#[test]
fn mixing_rate_and_mode_is_rejected() {
    let err = try_resolve_no_gate("rate = 0.5, mode = mul, by = 0.5")
        .expect_err("rate + mode is ambiguous; must be rejected");
    assert!(
        err.contains("rate") && err.contains("mode"),
        "expected mutual-exclusion error; got: {err}"
    );
}

// -- mode = sub gate marker lands in emitted WGSL -----------------------

/// When `gate = MaskName` is set, the emitted decay kernel surfaces a
/// `TODO(decay-gate)` marker rather than inlining the predicate. This
/// pins the documented gap-resolution: cross-binding plumbing for decay
/// kernels is not yet implemented; the runtime keeps owning the bespoke
/// decay path. Removing this marker requires landing the (a) + (b)
/// extension chain documented in `build_view_decay_wgsl_body`.
#[test]
fn gate_emits_todo_marker_in_wgsl() {
    let src = TEMPLATE_WITH_MASK
        .replace("MODE_AND_BY_PLACEHOLDER", "mode = sub, by = 1")
        .replace("GATE_PLACEHOLDER", "Strike");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("lower");
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit");
    // Find the decay kernel — its name follows the `decay_<view>` convention.
    let decay_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(k, _)| k.starts_with("decay_"))
        .map(|(_, v)| v.as_str())
        .expect("decay kernel WGSL must be emitted when @decay is set");
    assert!(
        decay_wgsl.contains("TODO(decay-gate)"),
        "gate-set decay kernel must surface the TODO marker; got:\n{decay_wgsl}"
    );
}

/// When `mode = sub, by = N` is set without `gate`, the emitted decay
/// kernel runs `select(old - by, 0u, old < by)` per slot — the saturating
/// integer subtract that subsumes tom_probe's bespoke decay arithmetic
/// (modulo the cross-binding gap).
#[test]
fn mode_sub_emits_saturating_select_in_wgsl() {
    let src = TEMPLATE_NO_GATE.replace("MODE_AND_BY_PLACEHOLDER", "mode = sub, by = 1");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("lower");
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit");
    let decay_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(k, _)| k.starts_with("decay_"))
        .map(|(_, v)| v.as_str())
        .expect("decay kernel WGSL must be emitted");
    assert!(
        decay_wgsl.contains("let by: u32 = 1u;"),
        "expected `by` const; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("select(old - by, 0u, old < by)"),
        "expected saturating-sub select; got:\n{decay_wgsl}"
    );
    assert!(
        !decay_wgsl.contains("bitcast<f32>"),
        "sub mode should NOT bitcast to f32; got:\n{decay_wgsl}"
    );
}

/// Sibling pin for `mode = mul, by = R`: the emitted body matches the
/// legacy `rate = R` form byte-for-byte (modulo whitespace) so no
/// downstream consumer needs to special-case the explicit spelling.
#[test]
fn mode_mul_emits_legacy_multiplicative_path() {
    let src = TEMPLATE_NO_GATE.replace("MODE_AND_BY_PLACEHOLDER", "mode = mul, by = 0.5");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("lower");
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit");
    let decay_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(k, _)| k.starts_with("decay_"))
        .map(|(_, v)| v.as_str())
        .expect("decay kernel WGSL must be emitted");
    assert!(
        decay_wgsl.contains("bitcast<f32>"),
        "mul mode should bitcast to f32; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("old * 0.5"),
        "expected `old * 0.5` multiplicative step; got:\n{decay_wgsl}"
    );
}
