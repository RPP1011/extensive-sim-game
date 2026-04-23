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
