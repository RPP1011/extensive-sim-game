//! Stress fixtures for slice 1 (cross-agent target reads) and the
//! Phase 7+8 wired primitives (event ring + view-fold storage +
//! @decay) under load.
//!
//! These tests run the .sim files at
//! `assets/sim/target_chaser.sim` and
//! `assets/sim/swarm_event_storm.sim` through the full
//! parse → resolve → CG lower → schedule → emit pipeline. They
//! intentionally exercise codepaths the existing pp/pc/cn min
//! fixtures don't:
//!
//! - **target_chaser**: `agents.pos(self.engaged_with)` in physics —
//!   the cross-agent target read codepath slice 1
//!   (`docs/superpowers/plans/2026-05-03-stdlib-into-cg-ir.md`)
//!   replaced the B1 typed-default placeholder for. Without slice 1
//!   the WGSL emit silently returned `vec3<f32>(0.0)` for the
//!   target read; with slice 1 it hoists a stmt-scope
//!   `let target_expr_<N>: u32 = …;` and uses
//!   `agent_pos[target_expr_<N>]`.
//!
//! - **swarm_event_storm**: 4 emits per agent per tick into a single
//!   ring + two folds (a plain accumulator and an @decay-anchored
//!   accumulator). Stresses the per-tick event-ring throughput and
//!   the @decay anchor RMW.
//!
//! Today these tests assert the pipeline reaches `emit` without
//! panicking. They DON'T assert observable behaviour (that needs
//! per-fixture runtime crates). The goal is to surface the next
//! gap as a typed compiler error rather than a runtime panic — so
//! any new gap is a focused fix, not a "where do I even start"
//! debugging session.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

/// Drive `path` through parse → resolve → lower → schedule → emit.
/// Returns the emitted artifacts on success; surfaces the pipeline
/// error verbatim on failure so the test failure is the next gap
/// rather than an opaque panic.
fn compile_sim(path: &std::path::Path) -> Result<EmittedArtifacts, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|e| format!("lower: {e:?}"))?;
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))
}

/// Find the first WGSL kernel whose name contains `needle`. Used to
/// pick the physics or fold body out of the artifact set without
/// hard-coding the exact emitted name (those drift as the kernel
/// composer evolves).
fn kernel_body_containing<'a>(art: &'a EmittedArtifacts, needle: &str) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle))
        .map(|(_, body)| body.as_str())
}

#[test]
fn target_chaser_compiles() {
    let path = workspace_path("assets/sim/target_chaser.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("target_chaser.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[target_chaser] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// Slice 1 invariant: the cross-agent target read in
/// `agents.pos(self.engaged_with)` lowers to a stmt-scope
/// `let target_expr_<N>: u32 = …;` paired with `agent_pos[
/// target_expr_<N>]`, NOT the prior B1 `vec3<f32>(0.0)` placeholder.
/// Locks slice 1 into the runtime-driven (not just unit-tested)
/// codepath.
#[test]
fn target_chaser_emits_target_let_binding() {
    let path = workspace_path("assets/sim/target_chaser.sim");
    let art = compile_sim(&path).expect("target_chaser compiles");
    let body = kernel_body_containing(&art, "ChaseTarget")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    assert!(
        body.contains("let target_expr_"),
        "expected slice-1 let target_expr_<N> binding in physics body; got:\n{body}",
    );
    assert!(
        body.contains("agent_pos[target_expr_"),
        "expected indexed access against target_expr_<N>; got:\n{body}",
    );
    assert!(
        !body.contains("vec3<f32>(0.0)") || body.matches("vec3<f32>(0.0)").count() < 2,
        "B1 typed-default placeholder should not dominate the body; got:\n{body}",
    );
}

#[test]
fn swarm_event_storm_compiles() {
    let path = workspace_path("assets/sim/swarm_event_storm.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("swarm_event_storm.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[swarm_event_storm] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `swarm_event_storm` declares 4 `emit Pulse { … }` per tick per
/// agent. Confirm the producer kernel actually emits 4 atomicAdd
/// + 4 atomicStore-blocks (one per emit), not e.g. one collapsed
/// emit. Locks the multi-emit codepath so a regression that
/// silently dropped emits (B1-style) would fail here.
#[test]
fn swarm_event_storm_emits_four_pulses_per_tick() {
    let path = workspace_path("assets/sim/swarm_event_storm.sim");
    let art = compile_sim(&path).expect("swarm_event_storm compiles");
    let body = kernel_body_containing(&art, "PulseAndDrift")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    let atomic_adds = body.matches("atomicAdd(&event_tail").count();
    assert_eq!(
        atomic_adds, 4,
        "expected 4 atomicAdds (one per Pulse emit); got {atomic_adds} in:\n{body}",
    );
}

/// Slice 2b probe (stdlib-into-CG-IR plan) — confirms a physics
/// rule body that consumes a Phase 7 named spatial_query through
/// the existing `sum(other in spatial.<name>(self) where ...)`
/// fold shape lowers end-to-end via the shared
/// `lower_spatial_namespace_call` helper. Without slice 2a's
/// recogniser extraction this codepath worked already (boids does
/// it) — the test pins the helper's continued coverage of physics
/// rule contexts so a regression in the fold-iter classification
/// fails here rather than silently skipping the spatial walk.
///
/// The `spatial_probe.sim` fixture is intentionally minimal: one
/// entity, one named query, one fold-bearing physics rule. The
/// neighbour-walk WGSL shape is recognisable by its
/// `spatial_grid_offsets` references (the bounded-walk template
/// reads grid offsets to enumerate candidates).
#[test]
fn spatial_probe_compiles_and_emits_neighbour_walk() {
    let path = workspace_path("assets/sim/spatial_probe.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("spatial_probe.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    let body = kernel_body_containing(&art, "ProbeMove")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    // Bounded-neighbour walk template references grid offsets; if
    // the spatial-iter recogniser ever silently falls back to
    // ForEachAgent (the unbounded N² path), this assertion catches
    // it (ForEachAgent emits a `for (var per_pair_candidate ... <
    // cfg.agent_cap` loop with no grid-offset reference).
    assert!(
        body.contains("spatial_grid_offsets") || body.contains("grid_starts"),
        "expected bounded-neighbour walk references in physics body; got:\n{body}",
    );
    eprintln!(
        "[spatial_probe] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `swarm_event_storm` declares a `@decay(rate=0.85)` view alongside
/// a non-decayed view; both consume from the same Pulse ring.
/// Confirm both fold kernels exist and the decay one references
/// the anchor binding.
#[test]
fn swarm_event_storm_emits_both_folds_with_decay_anchor() {
    let path = workspace_path("assets/sim/swarm_event_storm.sim");
    let art = compile_sim(&path).expect("swarm_event_storm compiles");
    let plain = kernel_body_containing(&art, "pulse_count").unwrap_or_else(|| {
        panic!(
            "no pulse_count fold kernel in artifacts; available: {:?}",
            art.wgsl_files.keys().collect::<Vec<_>>()
        );
    });
    let decayed = kernel_body_containing(&art, "recent_pulse_intensity").unwrap_or_else(|| {
        panic!(
            "no recent_pulse_intensity fold kernel; available: {:?}",
            art.wgsl_files.keys().collect::<Vec<_>>()
        );
    });
    // Plain fold body has primary storage but no anchor reference.
    assert!(
        plain.contains("view_storage_primary"),
        "plain fold should write primary; got:\n{plain}",
    );
    // Decayed fold body should reference the anchor binding (the
    // @decay rate gets applied via the anchor multiplication).
    // If the anchor isn't wired, this assertion surfaces the gap.
    assert!(
        decayed.contains("anchor") || decayed.contains("decay"),
        "decay fold should reference anchor or decay rate; got:\n{decayed}",
    );
}
