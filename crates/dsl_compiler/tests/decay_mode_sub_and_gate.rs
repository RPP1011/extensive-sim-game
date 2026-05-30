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

// -- mode = sub gate inlining lands in emitted WGSL ---------------------

/// When `gate = MaskName` is set, the emitted decay kernel inlines the
/// mask predicate as `if (<pred>) { <decay step> }`. The predicate's
/// `self` reference resolves against the per-cell row-derived `agent_id`
/// the body builder binds (`agent_id = k / cfg.agent_cap`). For the
/// no-arg `mask Strike when self.alive` predicate, the inlined WGSL
/// reads `agent_alive[agent_id]`.
#[test]
fn gate_inlines_predicate_in_wgsl() {
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
    // 1. The TODO marker is gone — the gap chain is closed.
    assert!(
        !decay_wgsl.contains("TODO(decay-gate)"),
        "gate-inlining should retire the TODO marker; got:\n{decay_wgsl}"
    );
    // 2. The per-cell binder mapping is in place: agent_id derived from
    //    the row, per_pair_candidate from the column, tick aliased from
    //    cfg.tick (so `world.tick` references resolve).
    assert!(
        decay_wgsl.contains("let agent_id = k / cfg.agent_cap;"),
        "expected per-cell agent_id binding; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("let per_pair_candidate = k % cfg.agent_cap;"),
        "expected per-cell per_pair_candidate binding; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("let tick = cfg.tick;"),
        "expected tick preamble alias; got:\n{decay_wgsl}"
    );
    // 3. The predicate inlines as an `if` wrapping the existing step
    //    body. For `self.alive`, the lowered WGSL reads
    //    `agent_alive[agent_id]`.
    assert!(
        decay_wgsl.contains("let decay_gate_value: bool"),
        "expected gate evaluation binding; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("agent_alive[agent_id]"),
        "expected `self.alive` to lower to indexed agent_alive read; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("if (decay_gate_value)"),
        "expected gate-wrapped decay step; got:\n{decay_wgsl}"
    );
    // 4. The kernel binds the `agent_alive` storage the predicate reads
    //    (gate-mask handle walk extends the BGL).
    assert!(
        decay_wgsl.contains("var<storage, read> agent_alive"),
        "expected agent_alive read-only binding; got:\n{decay_wgsl}"
    );
    // 5. cfg moves to the LAST slot once gate-mask extras land between
    //    view_storage_primary and cfg — verify the slot 0 / cfg invariant
    //    still holds (cfg is no longer at slot 1).
    assert!(
        decay_wgsl.contains("var<uniform> cfg:"),
        "cfg uniform binding must still exist; got:\n{decay_wgsl}"
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

/// Per-cell decay-gate inlining for a predicate that reads BeliefState
/// storage (`agents.beliefs_last_seen_tick(self, target) != world.tick`).
/// This exercises the full extension chain:
///
///   * gate-mask predicate IR walk discovers the `BeliefStateColumn::
///     LastSeenTick` handle even though the DSL surface is a
///     namespace-call (not a `Read(handle)` node);
///   * `build_view_decay_bindings` extends the kernel BGL with the
///     `beliefs_tick` storage binding;
///   * the body builder lowers the predicate to
///     `agents_beliefs_last_seen_tick(agent_id, per_pair_candidate) != tick`
///     and wraps the saturating-sub step in `if (<predicate>) { ... }`;
///   * `compose_namespace_prelude` substring-detects the call form and
///     injects the `agents_beliefs_last_seen_tick` prelude function.
///
/// The fixture mirrors tom_probe's `BeliefStillFresh` shape (the only
/// real-world consumer of this slice) without requiring the .sim to opt
/// into the runtime-managed BeliefState SoA via `agents.set_beliefs_*`
/// in physics — declaring the mask alone is sufficient because the
/// gate-mask handle walk + `BeliefStateColumn` BGL extension is
/// orthogonal to the `LowerOpts.belief_state` auto-detect.
#[test]
fn gate_inlines_belief_state_predicate() {
    let src = r#"
event Tick { }

@replayable
@gpu_amenable
event Hit {
  target: AgentId,
}

entity Knower : Agent {
  pos: vec3,
  vel: vec3,
}

mask BeliefStillFresh(target: Agent) when agents.beliefs_last_seen_tick(self, target) != world.tick

@materialized(on_event = [Hit])
@decay(per = tick, mode = sub, by = 1, gate = BeliefStillFresh)
view confidence(observer: Agent, subject: Agent) -> u32 {
  initial: 0,
  on Hit { target: agent } where agent == subject { self += 1 }
  clamp: [0, 1000000],
}

physics Tickle @phase(per_agent) {
  on Tick {} {
    emit Hit { target: self }
  }
}

physics SetSomeBelief @phase(per_agent) {
  on Tick {} {
    agents.set_beliefs_last_seen_tick(self, self, world.tick);
  }
}
"#;
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg_with_opts(
        &comp,
        dsl_compiler::cg::lower::LowerOpts {
            belief_state: true,
            ..Default::default()
        },
    )
    .expect("lower");
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

    // 1. The kernel's BGL extends with the `beliefs_tick` binding
    //    (BeliefStateColumn::LastSeenTick → `beliefs_tick`). Without the
    //    handle walk surfacing this, the prelude function's
    //    `beliefs_tick[...]` access would reference an undeclared name.
    assert!(
        decay_wgsl.contains("beliefs_tick"),
        "expected beliefs_tick binding from gate-mask BeliefState handle walk; got:\n{decay_wgsl}"
    );
    // 2. The substring-driven namespace prelude injects the getter fn
    //    body (`fn agents_beliefs_last_seen_tick(observer: u32, ...)`).
    assert!(
        decay_wgsl.contains("fn agents_beliefs_last_seen_tick"),
        "expected agents_beliefs_last_seen_tick prelude fn; got:\n{decay_wgsl}"
    );
    // 3. The predicate inlines via the prelude call form, with `self` /
    //    `target` mapped to per-cell `agent_id` / `per_pair_candidate`.
    assert!(
        decay_wgsl
            .contains("agents_beliefs_last_seen_tick(agent_id, per_pair_candidate)"),
        "expected per-cell binder mapping in predicate call; got:\n{decay_wgsl}"
    );
    // 4. The decay step wraps in the gate predicate.
    assert!(
        decay_wgsl.contains("if (decay_gate_value)"),
        "expected gate-wrapped decay step; got:\n{decay_wgsl}"
    );
    // 5. The saturating-sub formula still lives in the wrapped block.
    assert!(
        decay_wgsl.contains("select(old - by, 0u, old < by)"),
        "expected saturating-sub formula inside gate-wrapped block; got:\n{decay_wgsl}"
    );
    // 6. No leftover TODO marker.
    assert!(
        !decay_wgsl.contains("TODO(decay-gate)"),
        "TODO marker should be retired; got:\n{decay_wgsl}"
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
