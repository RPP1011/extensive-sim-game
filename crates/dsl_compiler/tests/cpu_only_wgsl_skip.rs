//! Task 2.15 — `@cpu_only` physics rules must produce NO WGSL output
//! and NO entry in the GPU event-kind dispatcher table.
//!
//! Two rules bind on the same event kind (`AgentDied`): one is
//! `@cpu_only` (narrative-only, no GPU lowering) and one is a regular
//! GPU-emittable rule. The test compiles the merged source, runs both
//! WGSL emission paths, and asserts:
//!   * the per-rule WGSL body for the `@cpu_only` rule is empty
//!   * the per-rule WGSL body for the GPU rule includes its fn name
//!   * the dispatcher contains a call to the GPU rule only; the
//!     `@cpu_only` rule's handler fn is never referenced.
//!
//! Task 2.16 — regression fence for the *CPU* emit path. The CPU
//! handler emit (`emit_physics::emit_physics`) must remain unaffected
//! by the `@cpu_only` flag: both annotated and regular rules must
//! still get their CPU handler function emitted, so the narrative /
//! event-replay path continues to run them. If a future change
//! mistakenly gates CPU emit on `cpu_only` (e.g. "skip WGSL AND skip
//! CPU"), this test catches it.

use dsl_compiler::emit_physics_wgsl::{
    emit_physics_dispatcher_wgsl, emit_physics_wgsl, EmitContext,
};

const SRC: &str = r#"
@replayable event AgentDied { agent_id: AgentId }

@cpu_only physics narrative_rule @phase(event) {
    on AgentDied { agent_id: a } { }
}

physics gpu_rule @phase(event) {
    on AgentDied { agent_id: a } { }
}
"#;

#[test]
fn cpu_only_rule_has_no_wgsl_emit_and_no_dispatch_entry() {
    let comp = dsl_compiler::compile(SRC).expect("compile OK");

    // Sanity: both rules parsed + resolved, cpu_only flag set.
    let narrative = comp
        .physics
        .iter()
        .find(|p| p.name == "narrative_rule")
        .expect("narrative_rule must be present in IR");
    let gpu = comp
        .physics
        .iter()
        .find(|p| p.name == "gpu_rule")
        .expect("gpu_rule must be present in IR");
    assert!(
        narrative.cpu_only,
        "narrative_rule should carry cpu_only flag from @cpu_only annotation"
    );
    assert!(
        !gpu.cpu_only,
        "gpu_rule should not be cpu_only"
    );

    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };

    // --- Per-rule WGSL emission ---
    let narrative_wgsl =
        emit_physics_wgsl(narrative, &ctx).expect("cpu_only rule must not error on emit");
    assert!(
        narrative_wgsl.is_empty(),
        "@cpu_only narrative_rule must produce no WGSL output, got:\n{narrative_wgsl}"
    );

    let gpu_wgsl = emit_physics_wgsl(gpu, &ctx).expect("gpu_rule must emit cleanly");
    assert!(
        gpu_wgsl.contains("fn physics_gpu_rule(event_idx: u32)"),
        "gpu_rule WGSL body must contain its handler fn signature, got:\n{gpu_wgsl}"
    );
    assert!(
        !gpu_wgsl.contains("narrative_rule"),
        "gpu_rule WGSL body must not mention narrative_rule, got:\n{gpu_wgsl}"
    );

    // --- Concatenated "WGSL shader" view over all physics rules ---
    //
    // Mirrors the call pattern in `engine_gpu::physics::build_physics_shader_*`:
    // iterate every `PhysicsIR`, concatenate emitter output, then append the
    // dispatcher. Grepping this concatenation for rule names is the concrete
    // load-bearing invariant — `@cpu_only` rules must never appear anywhere
    // in the WGSL text.
    let mut all_wgsl = String::new();
    for rule in &comp.physics {
        let part = emit_physics_wgsl(rule, &ctx).expect("emit OK");
        all_wgsl.push_str(&part);
    }
    let dispatcher = emit_physics_dispatcher_wgsl(&comp.physics, &ctx);
    all_wgsl.push_str(&dispatcher);

    assert!(
        all_wgsl.contains("physics_gpu_rule"),
        "concatenated WGSL must contain gpu_rule handler, got:\n{all_wgsl}"
    );
    assert!(
        !all_wgsl.contains("narrative_rule"),
        "@cpu_only narrative_rule must not appear anywhere in concatenated WGSL, got:\n{all_wgsl}"
    );
    assert!(
        !all_wgsl.contains("physics_narrative_rule"),
        "@cpu_only narrative_rule's handler fn name must not appear in concatenated WGSL, got:\n{all_wgsl}"
    );

    // --- Dispatcher table specifically ---
    assert!(
        dispatcher.contains("physics_gpu_rule(event_idx);"),
        "dispatcher must route AgentDied to gpu_rule, got:\n{dispatcher}"
    );
    assert!(
        !dispatcher.contains("physics_narrative_rule"),
        "dispatcher must not reference the @cpu_only rule's handler fn, got:\n{dispatcher}"
    );
}

/// Regression fence — CPU handler emit must keep firing for `@cpu_only`
/// rules. Tasks 2.12-2.15 gate *WGSL* emission on the flag; the CPU
/// path in `emit_physics::emit_physics` is intentionally untouched so
/// narrative rules still get a Rust handler fn wired into the
/// per-event-kind dispatcher. If a future change (e.g. "skip every
/// backend for cpu_only") breaks this, the emit-output inspection
/// below fires.
#[test]
fn cpu_only_rule_still_emits_cpu_handler() {
    let comp = dsl_compiler::compile(SRC).expect("compile OK");
    let artifacts = dsl_compiler::emit(&comp);

    // Each physics rule lowers to its own Rust module file
    // (`<snake_case(rule_name)>.rs`) in `rust_physics_modules`, plus
    // the aggregate dispatcher lives in `rust_physics_mod`. The
    // `@cpu_only` rule must land in both.
    let filenames: Vec<&str> = artifacts
        .rust_physics_modules
        .iter()
        .map(|(f, _)| f.as_str())
        .collect();
    assert!(
        filenames.iter().any(|f| *f == "narrative_rule.rs"),
        "@cpu_only narrative_rule must still emit a Rust module; got files: {filenames:?}"
    );
    assert!(
        filenames.iter().any(|f| *f == "gpu_rule.rs"),
        "regular gpu_rule must emit a Rust module; got files: {filenames:?}"
    );

    let narrative_rs = &artifacts
        .rust_physics_modules
        .iter()
        .find(|(f, _)| f == "narrative_rule.rs")
        .expect("narrative_rule.rs must be present")
        .1;
    let gpu_rs = &artifacts
        .rust_physics_modules
        .iter()
        .find(|(f, _)| f == "gpu_rule.rs")
        .expect("gpu_rule.rs must be present")
        .1;

    // Handler fn uses the snake_case rule name as its identifier — see
    // `emit_physics::handler_fn_name`. Both rules must have one.
    assert!(
        narrative_rs.contains("pub fn narrative_rule("),
        "@cpu_only narrative_rule must still emit its CPU handler fn, got:\n{narrative_rs}"
    );
    assert!(
        gpu_rs.contains("pub fn gpu_rule("),
        "regular gpu_rule must emit its CPU handler fn, got:\n{gpu_rs}"
    );

    // The aggregate `physics/mod.rs` dispatcher routes the triggering
    // event through *every* applicable handler, `@cpu_only` included.
    let physics_mod = &artifacts.rust_physics_mod;
    assert!(
        physics_mod.contains("pub mod narrative_rule;"),
        "physics/mod.rs must `pub mod` the @cpu_only rule, got:\n{physics_mod}"
    );
    assert!(
        physics_mod.contains("pub mod gpu_rule;"),
        "physics/mod.rs must `pub mod` gpu_rule, got:\n{physics_mod}"
    );
    // Both handler names must appear in the per-kind dispatcher call
    // list. Exact call shape is `narrative_rule::narrative_rule(...)`
    // / `gpu_rule::gpu_rule(...)`.
    assert!(
        physics_mod.contains("narrative_rule::narrative_rule("),
        "dispatcher must call @cpu_only narrative_rule's CPU handler, got:\n{physics_mod}"
    );
    assert!(
        physics_mod.contains("gpu_rule::gpu_rule("),
        "dispatcher must call gpu_rule's CPU handler, got:\n{physics_mod}"
    );
}
