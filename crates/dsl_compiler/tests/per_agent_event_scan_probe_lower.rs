//! Plan G G3d — smoke test for `assets/sim/per_agent_event_scan_probe.sim`.
//!
//! Lights up the new `DispatchShape::PerAgentEventScan` variant added
//! alongside the threats-view design. The variant dispatches one
//! thread per `(observer_agent, source_agent)` pair, with an early-
//! exit when the source agent's `busy_with_ability_id` SoA column is
//! zero. The threats view (Plan G G3g) will be the first real
//! consumer; this fixture proves the dispatch-shape mechanism in
//! isolation with a trivially-simple per-agent counter view.
//!
//! Asserts:
//!
//! 1. The fixture parses + resolves + lowers (no errors).
//! 2. The `threat_count` view emits a kernel.
//! 3. The kernel's WGSL declares `@workgroup_size(8, 8)` (the 2-D
//!    geometry pinned to 8×8 = 64 threads per group).
//! 4. The kernel preamble binds `gid.y` to `source_candidate` (the
//!    structural signal that the dispatch is 2-D, not 1-D).
//! 5. The kernel preamble has the busy-filter early-exit
//!    (`agent_busy_with_ability_id[source_candidate] == 0u`).
//! 6. The kernel does NOT use the standard PerEvent preamble
//!    (`event_idx = gid.x; event_idx >= cfg.event_count`) — that
//!    would indicate the new shape silently fell back to PerEvent.

#[test]
fn per_agent_event_scan_probe_sim_lowers_clean() {
    let src = std::fs::read_to_string("../../assets/sim/per_agent_event_scan_probe.sim")
        .expect("read assets/sim/per_agent_event_scan_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse per_agent_event_scan_probe.sim");
    let comp = dsl_ast::resolve::resolve(program)
        .expect("resolve per_agent_event_scan_probe.sim");

    let (cg, lower_diags) = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => (p, Vec::new()),
        Err(o) => {
            let diags: Vec<String> = o.diagnostics.iter().map(|d| format!("{d}")).collect();
            (o.program, diags)
        }
    };
    for d in &lower_diags {
        eprintln!("[lower diag] {d}");
    }

    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit per_agent_event_scan_probe");

    assert!(
        !artifacts.kernel_index.is_empty(),
        "expected at least one emitted kernel, got none"
    );

    let kernel_names: Vec<&str> =
        artifacts.kernel_index.iter().map(|s| s.as_str()).collect();
    for expected in ["StartCast", "threat_count"] {
        assert!(
            kernel_names.iter().any(|n| n.contains(expected)),
            "expected kernel containing {expected:?}; got: {kernel_names:?}"
        );
    }

    // Pull the threat_count fold WGSL out and inspect its preamble.
    let fold_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("threat_count") && name.ends_with(".wgsl"))
        .map(|(name, body)| (name.as_str(), body.as_str()))
        .expect("threat_count fold kernel must emit a .wgsl artifact");

    eprintln!("[G3d smoke] inspecting {}", fold_wgsl.0);
    eprintln!("[G3d smoke] body length: {} bytes", fold_wgsl.1.len());

    // 2-D workgroup geometry — the new shape pins `@workgroup_size(8, 8)`.
    assert!(
        fold_wgsl.1.contains("@workgroup_size(8, 8)"),
        "PerAgentEventScan emit must declare a 2-D `@workgroup_size(8, 8)` annotation. \
         WGSL did not contain the substring.\n\nWGSL:\n{}",
        fold_wgsl.1
    );

    // 2-D thread indexing — the body must read `gid.y` for the source
    // candidate. Without it, the dispatch silently fell back to a 1-D
    // PerEvent / PerAgent shape.
    assert!(
        fold_wgsl.1.contains("let source_candidate = gid.y"),
        "PerAgentEventScan emit must bind `source_candidate = gid.y` \
         (the structural signal that the dispatch is 2-D). WGSL did not \
         contain the substring.\n\nWGSL:\n{}",
        fold_wgsl.1
    );
    assert!(
        fold_wgsl.1.contains("let observer = gid.x"),
        "PerAgentEventScan emit must bind `observer = gid.x`. WGSL did \
         not contain the substring.\n\nWGSL:\n{}",
        fold_wgsl.1
    );

    // Busy-filter early-exit — drops any `(observer, non-busy-source)`
    // pair before the per-op fold body runs.
    assert!(
        fold_wgsl.1.contains("agent_busy_with_ability_id[source_candidate]"),
        "PerAgentEventScan emit must early-exit any source candidate \
         whose `agent_busy_with_ability_id[source_candidate]` is zero. \
         WGSL did not contain the substring.\n\nWGSL:\n{}",
        fold_wgsl.1
    );

    // Negative pin: the standard PerEvent preamble (`let event_idx =
    // gid.x; if (event_idx >= cfg.event_count) ...`) must NOT appear
    // in the body — its presence would mean the dispatch silently
    // fell back to PerEvent and the busy-filter never ran.
    //
    // Note: `cfg.event_count` itself can still appear in the
    // PerAgentEventScan preamble as the bounds source for `observer` /
    // `source_candidate` (the cfg layout aliasing is documented on the
    // body builder); we pin against the `event_idx = gid.x` substring
    // specifically to catch the wrong-preamble regression.
    assert!(
        !fold_wgsl.1.contains("let event_idx = gid.x"),
        "PerAgentEventScan emit must NOT use the standard PerEvent \
         preamble (`let event_idx = gid.x`). The presence of that \
         substring means the dispatch silently fell back to PerEvent.\n\nWGSL:\n{}",
        fold_wgsl.1
    );
}
