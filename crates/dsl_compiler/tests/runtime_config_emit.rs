//! Plan 1 — Task 4 lock: a `@runtime`-annotated config field must lower
//! to a per-kernel cfg-uniform read (`cfg.config_<block>_<field>`), NOT a
//! baked inline `const`. This is the compiler half of the input channel
//! the playable game depends on; the runtime/GPU half is locked by
//! `crates/sims/tests/input_probe_exec.rs`.
//!
//! Drives `assets/sim/input_probe.sim` through
//! parse -> resolve -> lower -> schedule -> emit and asserts the emitted
//! `DriveX` kernel body reads the cfg uniform and bakes no `const config_`.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

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

fn kernel_body_containing<'a>(art: &'a EmittedArtifacts, needle: &str) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle))
        .map(|(_, body)| body.as_str())
}

#[test]
fn runtime_config_reads_cfg_not_const() {
    let art = compile_sim(&workspace_path("assets/sim/input_probe.sim")).expect("compiles");
    let body = kernel_body_containing(&art, "DriveX").unwrap_or_else(|| {
        panic!(
            "no DriveX kernel; have {:?}",
            art.wgsl_files.keys().collect::<Vec<_>>()
        )
    });
    // The @runtime field must read the cfg uniform (the kernel emit
    // rewrites `config_<id>` -> `cfg.config_<block>_<field>`).
    assert!(
        body.contains("cfg.config_probe_drive"),
        "runtime field must read cfg uniform `cfg.config_probe_drive`:\n{body}"
    );
    // ...and must NOT bake an inline `const config_<id>` — that would
    // dead-code the host-writable channel (the value would be frozen at
    // compile time instead of read per-tick).
    assert!(
        !body.contains("const config_"),
        "runtime field must NOT bake a const:\n{body}"
    );
}

/// The cfg struct must carry the runtime field after the standard 4-u32
/// header — i.e. the host setter writes at byte offset 16. This pins the
/// layout the generated `set_config_probe_drive` setter relies on.
#[test]
fn runtime_field_appended_to_cfg_struct() {
    let art = compile_sim(&workspace_path("assets/sim/input_probe.sim")).expect("compiles");
    let body = kernel_body_containing(&art, "DriveX").expect("DriveX kernel");
    assert!(
        body.contains("config_probe_drive: f32"),
        "cfg struct must declare the runtime field `config_probe_drive: f32`:\n{body}"
    );
}
