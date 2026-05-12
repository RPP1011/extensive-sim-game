//! Memory-ordering regression test (#284 sub-item #8).
//!
//! Pins the `atomicCompareExchangeWeak` CAS-loop emit for f32 SoA
//! RMW writes inside chronicle-consumer physics rules. Without this
//! shape, N events targeting the same agent slot in one dispatch
//! race on the f32 RMW (last-writer-wins on plain
//! `agent_hp[t] = agent_hp[t] - x`), and the same seed produces
//! different results across reruns — a P5 (deterministic / replay-
//! equivalent) violation.
//!
//! Closes the regression-test half of `#244` (memory entry
//! `project_f32_rmw_race.md`). The implementation half — the
//! `f32_atomic_field_writes` bitset that drives the upgrade — landed
//! earlier; this test prevents future drift.
//!
//! Pin shape (assertions below mirror the canonical CAS-loop):
//!
//! ```text
//! @group(0) @binding(N) var<storage, read_write> agent_hp: array<atomic<u32>>;
//! …
//! loop {
//!     let _old_bits_K = atomicLoad(&agent_hp[target_expr_X]);
//!     let _new_bits_K = bitcast<u32>((bitcast<f32>(atomicLoad(&agent_hp[target_expr_X])) - local_M));
//!     let _r_K = atomicCompareExchangeWeak(&agent_hp[target_expr_X], _old_bits_K, _new_bits_K);
//!     if (_r_K.exchanged) { break; }
//! }
//! ```

#[test]
fn firebolt_probe_apply_chronicle_damage_emits_cas_loop_on_agent_hp() {
    let src = std::fs::read_to_string("../../assets/sim/firebolt_probe.sim")
        .expect("read assets/sim/firebolt_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse firebolt_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve firebolt_probe.sim");

    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => o.program,
    };
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit firebolt_probe");

    // The chronicle-consumer rule (`physics ApplyChronicleDamage`) may
    // ship as its own kernel or fuse with another PerEvent rule that
    // shares the source ring (e.g. `physics RecordCastBegin` — both are
    // `PerEvent { source_ring: #0 }`, distinct event-kind subscriptions
    // make their bodies guarded by `if event_kind == X` arms inside one
    // kernel). The CAS-loop emit is body-local, so either kernel hosting
    // ApplyChronicleDamage carries the loop. Find the file by substring
    // rather than pinning a specific kernel name.
    let (wgsl_name, wgsl) = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("ApplyChronicleDamage"))
        .map(|(n, w)| (n.clone(), w.clone()))
        .unwrap_or_else(|| {
            panic!(
                "no kernel hosting ApplyChronicleDamage emitted; got files: {:?}",
                artifacts.wgsl_files.keys().collect::<Vec<_>>()
            )
        });
    let _ = wgsl_name;
    let wgsl = &wgsl;

    // 1. The agent_hp binding must be `array<atomic<u32>>`. Plain
    //    `array<f32>` would let naga accept the kernel but the CAS-loop
    //    `atomicCompareExchangeWeak` calls inside would type-error.
    assert!(
        wgsl.contains("var<storage, read_write> agent_hp: array<atomic<u32>>;"),
        "agent_hp binding must be array<atomic<u32>> — got:\n{wgsl}"
    );

    // 2. CAS-loop atomicLoad on agent_hp before the compute step.
    assert!(
        wgsl.contains("atomicLoad(&agent_hp["),
        "expected atomicLoad on agent_hp inside the CAS loop — got:\n{wgsl}"
    );

    // 3. The CAS itself — atomicCompareExchangeWeak on agent_hp.
    assert!(
        wgsl.contains("atomicCompareExchangeWeak(&agent_hp["),
        "expected atomicCompareExchangeWeak on agent_hp (the load-bearing race fix) — got:\n{wgsl}"
    );

    // 4. The retry shape — `if (result.exchanged) { break; }`. Without
    //    the retry, the CAS could silently drop a write under
    //    contention.
    assert!(
        wgsl.contains(".exchanged) { break; }"),
        "expected CAS retry pattern `if (_r_N.exchanged) {{ break; }}` — got:\n{wgsl}"
    );

    // 5. Negative pin: NO plain `agent_hp[idx] = …` assign. The whole
    //    point of the CAS upgrade is that the plain RMW would race;
    //    if the emit ever falls back to it, this test trips.
    let plain_writes: Vec<&str> = wgsl
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            // Match `agent_hp[<idx>] = …` but allow it inside the CAS
            // (where it appears as `&agent_hp[idx]` — has a leading `&`).
            trimmed.contains("agent_hp[") && trimmed.contains(" = ")
                && !trimmed.contains("&agent_hp[")
                && !trimmed.contains("atomicLoad")
                && !trimmed.contains("atomicCompareExchangeWeak")
                && !trimmed.contains("atomicStore")
        })
        .collect();
    assert!(
        plain_writes.is_empty(),
        "found plain non-atomic agent_hp writes (P5 violation — N→1 fan-in races): {plain_writes:?}\nfull WGSL:\n{wgsl}"
    );
}
