//! Plan E-A1 — shared build-script helper for `crates/*_runtime/build.rs`.
//!
//! Before this helper, every `*_runtime/build.rs` was a 100-150-line
//! file structurally identical to its 60 siblings except for the
//! fixture name string. Each duplicated:
//!
//! * Workspace-root resolution + `assets/sim/<fixture>.sim` read.
//! * `dsl_compiler::parse → resolve → cg::lower → schedule → emit`.
//! * `OUT_DIR/<kernel>.wgsl` writes.
//! * `OUT_DIR/generated.rs` concatenation with the standard `pub mod`
//!   wrappers per emitted Rust file.
//! * `cargo:warning` emit-stats lines (kernel name + size + binding count).
//!
//! That sprawl violated the "runtime crates contain no behavior"
//! direction the project is moving toward — and it was the load-bearing
//! reason every `.sim` change required touching N runtime crates by hand.
//! This helper collapses the entire build script to:
//!
//! ```ignore
//! fn main() { dsl_compiler::build_helper::emit("dodger_probe"); }
//! ```
//!
//! The behaviour is exactly what each runtime did by hand. The only
//! per-fixture-knob today is the fixture name; if any fixture later
//! needs a divergent emit-stats prefix or extra `cargo:warning` shape,
//! add a parameter here rather than re-introducing per-runtime build.rs
//! sprawl.

use std::env;
use std::fs;
use std::path::PathBuf;

/// Environment-variable name that, when set to a truthy value
/// (`1`/`true`/`yes`, case-insensitive), promotes any lower-time
/// `[<fixture> lower diag]` warning to a hard build error.
///
/// Default off (unset / `0` / `false` / empty) preserves the historical
/// "warn-and-drop" behaviour every fixture relied on before Gap
/// plague_city#P-C surfaced. Fixtures that opt in with this flag get
/// fail-fast on silent rule-drops.
///
/// Used by [`emit_into`] (and therefore every public `emit*` entry
/// point) to gate the diagnostic-promotion check.
pub const REQUIRE_ALL_RULES_ENV: &str = "SIM_REQUIRE_ALL_RULES";

/// Decides whether `SIM_REQUIRE_ALL_RULES` should be treated as set.
///
/// Truthy values (case-insensitive): `1`, `true`, `yes`, `on`. Anything
/// else — including absent, empty, `0`, `false`, `no`, `off` — is
/// treated as off. Mirrors the convention `RUST_LOG` / `CARGO_TERM_*`
/// environment-variable parsing uses.
fn require_all_rules_enabled() -> bool {
    match env::var(REQUIRE_ALL_RULES_ENV) {
        Ok(v) => is_truthy_env_value(&v),
        Err(_) => false,
    }
}

fn is_truthy_env_value(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Pure helper that decides whether a set of lower diagnostics should
/// promote to a build error under the current environment settings.
///
/// Returns `Ok(())` when either (a) `diagnostics` is empty, or (b)
/// `SIM_REQUIRE_ALL_RULES` is unset / falsy. Returns `Err(message)` —
/// a multi-line, fixture-tagged compile-error string suitable for
/// `panic!`-ing in a build script — when the env var is truthy AND
/// at least one diagnostic fired.
///
/// This entry point is split out from the `emit_into` body so tests in
/// `crates/dsl_compiler/tests/` can assert the env-gate semantics
/// without faking `OUT_DIR` / `CARGO_MANIFEST_DIR` / a real `.sim`
/// file. Tests synthesise a minimal `.sim` that fails to lower, run
/// it through `parse → resolve → lower_compilation_to_cg`, then call
/// this helper with the resulting `Vec<LoweringError>`.
///
/// See Gap P-C in `docs/architecture/gaps_plague_city.md` for the
/// real-world failure mode this guards against (silent rule-drop
/// shrinks the schedule from N to N-9 stages with no compile-time
/// signal).
pub fn check_required_rules<E: std::fmt::Display>(
    fixture_name: &str,
    diagnostics: &[E],
) -> Result<(), String> {
    if diagnostics.is_empty() || !require_all_rules_enabled() {
        return Ok(());
    }
    let mut msg = format!(
        "[{fixture_name} lower diag] {} diagnostic(s) fired with {REQUIRE_ALL_RULES_ENV}=1; \
         promoting to build error (rules referenced by these diagnostics would otherwise be \
         silently dropped from the schedule):\n",
        diagnostics.len(),
    );
    for d in diagnostics {
        msg.push_str(&format!("  - {d}\n"));
    }
    Err(msg)
}

/// Standard build-script body for any per-fixture runtime crate.
///
/// `fixture_name` is the basename used to:
/// * resolve `assets/sim/<fixture_name>.sim` against the workspace root
///   (two parents above `CARGO_MANIFEST_DIR` — i.e. the standard
///   `<workspace>/crates/<x>_runtime/` layout).
/// * label `cargo:warning=[<fixture_name> ...]` lines.
///
/// Tolerates lower diagnostics — emits them as `cargo:warning` and
/// continues with the partial CG program (matches the pre-extraction
/// behaviour of every fixture that consumed `LowerOutcome::Err`).
///
/// Panics on parse, resolve, or emit failures (these were `expect()`
/// calls in every per-fixture build.rs; surface them the same way so
/// the diagnostic surface is unchanged).
///
/// Set `SIM_REQUIRE_ALL_RULES=1` in the build environment to promote
/// any lower diagnostic to a hard build error — useful for fixtures
/// where a silently-dropped physics rule would shrink the schedule
/// below the author's intent (Gap plague_city#P-C). See
/// [`REQUIRE_ALL_RULES_ENV`] / [`check_required_rules`].
pub fn emit(fixture_name: &str) {
    emit_with_strategy(fixture_name, crate::cg::schedule::ScheduleStrategy::Default)
}

/// Same as [`emit`], but writes generated artifacts into a fixture-named
/// sub-directory of `OUT_DIR` instead of `OUT_DIR` itself. Used by the
/// `sims` mega-crate (Plan E-A6 follow-up) — its `build.rs` calls this
/// once per `.sim` file in `assets/sim/` so a single crate hosts every
/// fixture as `pub mod <fixture> { include!(.../<fixture>/generated.rs)
/// include!(.../<fixture>/runtime_core.rs) }`.
pub fn emit_namespaced(fixture_name: &str) {
    emit_namespaced_with_strategy(
        fixture_name,
        crate::cg::schedule::ScheduleStrategy::Default,
    )
}

/// Same as [`emit`], but lets the caller pin a non-default
/// [`ScheduleStrategy`]. Quest-arc and village-day-cycle fixtures
/// historically used `Conservative` to disable kernel fusion the
/// fixture wasn't compatible with.
pub fn emit_with_strategy(
    fixture_name: &str,
    strategy: crate::cg::schedule::ScheduleStrategy,
) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    emit_into(fixture_name, strategy, &out_dir);
}

/// Same as [`emit_namespaced`], with a custom strategy.
pub fn emit_namespaced_with_strategy(
    fixture_name: &str,
    strategy: crate::cg::schedule::ScheduleStrategy,
) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"))
        .join(fixture_name);
    fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("create_dir_all {}: {e}", out_dir.display()));
    emit_into(fixture_name, strategy, &out_dir);
    // The emitter bakes `crate::` paths in schedule.rs / dispatch.rs /
    // kernel sub-modules. Those work in a per-fixture crate (kernel
    // modules ARE at crate root) but break in the `sims` mega-crate
    // where everything sits inside `pub mod <fixture>`. Rewriting to
    // `super::` makes the generated code work in BOTH layouts (in a
    // per-fixture crate, `super::` from a sub-module = crate root).
    for entry in fs::read_dir(&out_dir).expect("read out_dir for crate:: rewrite") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let body = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let rewritten = body.replace("crate::", "super::");
        if rewritten != body {
            fs::write(&path, rewritten)
                .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        }
    }
}

fn emit_into(
    fixture_name: &str,
    strategy: crate::cg::schedule::ScheduleStrategy,
    out_dir: &std::path::Path,
) {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| {
            panic!(
                "workspace root above {} (expected <workspace>/crates/<name>/)",
                manifest_dir.display()
            )
        });
    let sim_path = workspace_root.join(format!("assets/sim/{fixture_name}.sim"));

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    // Resolve stdlib and sandbox roots from env vars (with workspace_root defaults).
    // WORLDSIM_STDLIB_ROOT / WORLDSIM_SANDBOX_ROOT are set by test harnesses and
    // can be overridden by per-runtime build.rs scripts; absent → conventional paths
    // relative to the workspace root.
    let stdlib_root: PathBuf = match env::var_os("WORLDSIM_STDLIB_ROOT") {
        Some(s) => PathBuf::from(s),
        None    => workspace_root.join("stdlib"),
    };
    let sandbox_root: PathBuf = match env::var_os("WORLDSIM_SANDBOX_ROOT") {
        Some(s) => PathBuf::from(s),
        None    => workspace_root.to_path_buf(),
    };
    let program = crate::imports::parse_with_imports(
        &sim_path, &stdlib_root, &sandbox_root,
    ).unwrap_or_else(|e| panic!("parse {sim_path:?} with imports: {e}"));
    let mut program = program;
    crate::cg::lower::param_rules::monomorphise(&mut program)
        .unwrap_or_else(|e| panic!("monomorphise param-rules in {sim_path:?}: {e}"));
    // Gap plague_city#P-A — populate the custom-agent-field registry
    // BEFORE resolve / lower runs. Every `field <name>: <ty>` decl
    // becomes a leaked `CustomFieldDesc`; subsequent `lower_field` /
    // `agents_setter_field` calls consult the registry via
    // `lookup_by_snake`. Idempotent — re-running on the same Program
    // returns the same interned ids.
    let _custom_field_ids = crate::custom_agent_fields::populate(&program);
    // Plan E-A6 — extract `init { ... }` blocks before resolve consumes
    // the Program. The build helper carries these through to
    // synthesize_runtime_core so try_new emits create_buffer_init with
    // the right per-slot pattern instead of zero-init create_buffer.
    let init_stmts: Vec<dsl_ast::ast::InitStmt> = program
        .decls
        .iter()
        .filter_map(|d| match d {
            dsl_ast::ast::Decl::Init(i) => Some(i.stmts.clone()),
            _ => None,
        })
        .flatten()
        .collect();
    // T11 — extract `terrain { ... }` block before resolve consumes the
    // Program. `TerrainBlock` derives Clone so we take a cheap copy here;
    // `emit_into` uses it after all the CG pipeline writes to conditionally
    // emit `terrain_gen.rs` alongside `generated.rs` and `runtime_core.rs`.
    let terrain_block: Option<dsl_ast::terrain::TerrainBlock> = program.terrain.clone();
    // Compiler-debug-mode opt-in (#242 follow-up). A `.sim` file may
    // declare `debug { depth: kernel, wgsl_event_kind_histogram: true,
    // ... }` — we extract the parsed values BEFORE resolve consumes the
    // Program and thread them into LowerOpts.debug + LowerOpts.debug_wgsl
    // below. Multiple `debug { }` blocks in one file are merged: depth
    // takes the LAST one set, every wgsl_* flag is OR'd. Absent block →
    // compiler defaults (DebugDepth::Off, all wgsl_* = false), which is
    // identical to the pre-annotation emit shape.
    let mut debug_depth: Option<dsl_ast::ast::DebugDepthLit> = None;
    let mut debug_wgsl_event_kind_histogram = false;
    let mut debug_wgsl_mask_hit_rate = false;
    let mut debug_wgsl_score_kernel_visits = false;
    for d in &program.decls {
        if let dsl_ast::ast::Decl::Debug(b) = d {
            if let Some(dd) = b.depth {
                debug_depth = Some(dd);
            }
            debug_wgsl_event_kind_histogram |= b.wgsl_event_kind_histogram;
            debug_wgsl_mask_hit_rate |= b.wgsl_mask_hit_rate;
            debug_wgsl_score_kernel_visits |= b.wgsl_score_kernel_visits;
        }
    }
    let lower_debug_depth = match debug_depth {
        Some(dsl_ast::ast::DebugDepthLit::Off) | None => crate::cg::lower::DebugDepth::Off,
        Some(dsl_ast::ast::DebugDepthLit::Stage) => crate::cg::lower::DebugDepth::Stage,
        Some(dsl_ast::ast::DebugDepthLit::StageMemory) => {
            crate::cg::lower::DebugDepth::StageMemory
        }
        Some(dsl_ast::ast::DebugDepthLit::Kernel) => crate::cg::lower::DebugDepth::Kernel,
        Some(dsl_ast::ast::DebugDepthLit::DslMapped) => {
            crate::cg::lower::DebugDepth::DslMapped
        }
    };
    let lower_debug_wgsl = crate::cg::lower::DebugWgslFlags {
        event_kind_histogram: debug_wgsl_event_kind_histogram,
        mask_hit_rate: debug_wgsl_mask_hit_rate,
        score_kernel_visits: debug_wgsl_score_kernel_visits,
    };
    let comp = dsl_ast::resolve::resolve(program)
        .unwrap_or_else(|e| panic!("resolve {fixture_name}.sim: {e}"));

    // Plan E-A6 — collect every `@runtime` config field's default value
    // keyed by the kernel-side name (`config_<block>_<field>`). Threads
    // into synthesize_runtime_core so try_new can preload each runtime
    // field with its .sim default and emit a host-side setter.
    let runtime_config_defaults: std::collections::BTreeMap<String, RuntimeConfigDefault> =
        comp.configs
            .iter()
            .flat_map(|c| {
                let block = c.name.clone();
                c.fields.iter().filter(|f| f.runtime).map(move |f| {
                    let key = format!("config_{block}_{}", f.name);
                    let scalar_ty = match &f.ty {
                        dsl_ast::ir::IrType::Named(n) => n.clone(),
                        _ => "f32".to_string(),
                    };
                    let default_lit = match &f.default {
                        dsl_ast::ast::ConfigDefault::Int(n) => format!("{n}"),
                        dsl_ast::ast::ConfigDefault::Uint(n) => format!("{n}"),
                        dsl_ast::ast::ConfigDefault::Float(f) => {
                            // Always render with a decimal so the literal
                            // typechecks as f32 in the synthesized init
                            // expression (e.g. `1.0_f32`, not `1_f32`).
                            let s = format!("{f}");
                            if s.contains('.') || s.contains('e') || s.contains('E') {
                                s
                            } else {
                                format!("{s}.0")
                            }
                        }
                        dsl_ast::ast::ConfigDefault::Bool(b) => format!("{b}"),
                        dsl_ast::ast::ConfigDefault::String(s) => format!("\"{s}\""),
                    };
                    (key, RuntimeConfigDefault { scalar_ty, default_lit })
                })
            })
            .collect();

    // Plan E-A6 follow-up — auto-detect companion `.ability` files at
    // `assets/ability_test/<fixture>/` and (a) build a real registry to
    // hand to the schedule synthesizer (so producer/consumer fusion sees
    // real chronicle EventKindIds), (b) auto-set `LowerOpts.aoe_dispatch`
    // when any program uses a non-Single area shape. Closes the gap that
    // forced fixtures with `.ability` corpora into custom build.rs files
    // (wave_defense, duel_25v25, swarm_storm, etc).
    let ability_corpus_dir =
        workspace_root.join("assets/ability_test").join(fixture_name);
    let ability_files: Vec<(String, dsl_ast::AbilityFile)> = if ability_corpus_dir.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(&ability_corpus_dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", ability_corpus_dir.display()))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("ability"))
            .collect();
        entries.sort();
        entries
            .into_iter()
            .map(|p| {
                println!("cargo:rerun-if-changed={}", p.display());
                let name = p.file_name().unwrap().to_string_lossy().into_owned();
                let src = fs::read_to_string(&p)
                    .unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
                let parsed = dsl_ast::parse_ability_file(&src)
                    .unwrap_or_else(|e| panic!("parse {name}: {e:?}"));
                (name, parsed)
            })
            .collect()
    } else {
        Vec::new()
    };
    let built_registry = if !ability_files.is_empty() {
        match crate::ability_registry::build_registry(&ability_files) {
            Ok(r) => Some(r),
            Err(e) => {
                println!("cargo:warning=[{fixture_name} ability_registry] {e:?}");
                None
            }
        }
    } else {
        None
    };
    // AOE auto-detect: if any program in the registry has any effect with
    // a non-None `EffectAreaShape`, the dispatcher needs Path B (spatial
    // walk + per-target chronicle write). Mirrors the manual opt-in every
    // outlier fixture's build.rs sets explicitly. See
    // `detect_aoe_dispatch` for the partial-failure fallback (Gap
    // squad_skirmish#B).
    let aoe_dispatch = detect_aoe_dispatch(built_registry.as_ref(), &ability_files);
    // Belief-state auto-detect: if any physics rule body contains an
    // `agents.set_beliefs_<field>(observer, subject, value)` call, the
    // BGL composer must surface the matching `BeliefStateColumn` binding
    // and the WGSL emit must replace the `return true` no-op stubs with
    // real SoA stores. Mirrors the AOE auto-detect above — collapses the
    // manual `LowerOpts.belief_state = true` opt-in that tom_probe's
    // custom build.rs used to set explicitly.
    let belief_state = comp.physics.iter().any(|p| {
        p.handlers
            .iter()
            .any(|h| stmts_contain_set_beliefs_call(&h.body))
    });
    // Symbolic ability-name → 1-based AbilityId map for the
    // `apply_ability <Name> …` DSL surface (2026-05-12). Built from the
    // same `BuiltRegistry` that drives the runtime registry, so the
    // compile-time name lookup is byte-identical to the runtime slot
    // assignment. Empty when the fixture has no `.ability` corpus —
    // every `apply_ability <Name>` then surfaces as a typed
    // `UnknownAbilityName` lower diagnostic.
    let ability_names: std::collections::BTreeMap<String, u32> = built_registry
        .as_ref()
        .map(|r| {
            r.names
                .iter()
                .map(|(name, id)| (name.clone(), id.raw()))
                .collect()
        })
        .unwrap_or_default();
    let lower_opts = crate::cg::lower::LowerOpts {
        aoe_dispatch,
        belief_state,
        debug: lower_debug_depth,
        debug_wgsl: lower_debug_wgsl,
        ability_names,
        ..Default::default()
    };
    if !ability_files.is_empty() {
        println!(
            "cargo:warning=[{fixture_name} ability-corpus] {} .ability files, aoe_dispatch={}",
            ability_files.len(),
            aoe_dispatch,
        );
    }
    if belief_state {
        println!(
            "cargo:warning=[{fixture_name} belief-state] auto-detected agents.set_beliefs_* call(s); enabling LowerOpts.belief_state",
        );
    }
    if lower_debug_depth != crate::cg::lower::DebugDepth::Off || lower_debug_wgsl.any() {
        println!(
            "cargo:warning=[{fixture_name} debug-opts] depth={:?} wgsl=event_kind_histogram={} mask_hit_rate={} score_kernel_visits={}",
            lower_debug_depth,
            lower_debug_wgsl.event_kind_histogram,
            lower_debug_wgsl.mask_hit_rate,
            lower_debug_wgsl.score_kernel_visits,
        );
    }
    let cg = match crate::cg::lower::lower_compilation_to_cg_with_opts(&comp, lower_opts) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[{fixture_name} lower diag] {d}");
            }
            // Gap plague_city#P-C: promote diagnostics to a build
            // error when the fixture (or its runtime crate's build.rs
            // wrapper) sets SIM_REQUIRE_ALL_RULES=1. Default off
            // preserves the historical warn-and-drop behaviour.
            if let Err(msg) = check_required_rules(fixture_name, &o.diagnostics) {
                panic!("{msg}");
            }
            o.program
        }
    };
    let schedule_result = crate::cg::schedule::synthesize_schedule_with_registry(
        &cg,
        strategy,
        built_registry.as_ref().map(|r| &r.registry),
    );
    let mut artifacts = crate::cg::emit::emit_cg_program_with_debug(
        &schedule_result.schedule,
        &cg,
        lower_debug_depth,
    )
    .unwrap_or_else(|e| panic!("emit {fixture_name} CG program: {e:?}"));

    // Sort-kernel opt-in: inject 15 sort kernels (4 × 3 Stage A passes +
    // 3 Stage B kernels) for any fixture that has at least one f32 view
    // fold. The kernels are complete WGSL files (own bindings + entry
    // point) emitted by cg::emit::sort_kernel; they bypass the normal
    // kernel_topology_to_spec_and_body pipeline.
    let sort_layout = sort_layout_for_fixture(&cg);
    let needs_sort = sort_layout.is_some();
    if let Some(ref layout) = sort_layout {
        inject_sort_kernels(&mut artifacts, layout);
        // Rewrite f32+Add fold kernels to use per-slot serial scan instead of
        // per-event CAS loop. After sort, events are ordered by (target, seq),
        // so each slot is a single writer — the CAS is unnecessary and causes
        // non-deterministic f32 accumulation order under parallel dispatch.
        rewrite_fold_wgsl_for_post_sort(&mut artifacts.wgsl_files);
        println!(
            "cargo:warning=[{fixture_name} sort-kernels] injected 15 radix sort kernels \
             (stride={stride}, target_word={tgt})",
            stride = layout.record_stride_u32,
            tgt = sort_target_word_offset(layout),
        );
    }

    println!(
        "cargo:warning=[{fixture_name} emit-stats] {} kernels, schedule has {} stages",
        artifacts.kernel_index.len(),
        schedule_result.schedule.stages.len(),
    );
    for kernel_name in &artifacts.kernel_index {
        let key = format!("{kernel_name}.wgsl");
        let body = match artifacts.wgsl_files.get(&key) {
            Some(b) => b,
            None => continue,
        };
        let bytes = body.len();
        let bindings = body.matches("@binding(").count();
        println!(
            "cargo:warning=[{fixture_name} emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {name}: {e}"));
    }

    let mut generated = String::new();
    generated.push_str(&format!(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by {fixture_name}_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/{fixture_name}.sim and rebuilding.\n\n",
    ));
    let mut wrap_module = |name: &str, content: &str| {
        generated.push_str(
            "#[allow(non_snake_case, unused_imports, unused_variables, dead_code, clippy::all)]\n",
        );
        generated.push_str(&format!("pub mod {name} {{\n"));
        generated.push_str(content);
        generated.push_str("\n}\n\n");
    };
    for kernel_name in &artifacts.kernel_index {
        let key = format!("{kernel_name}.rs");
        let content = artifacts.rust_files.get(&key).unwrap_or_else(|| {
            panic!("missing rust file {key} for kernel {kernel_name}")
        });
        wrap_module(kernel_name, content);
    }
    for sibling in ["schedule", "dispatch", "invariants", "metrics", "probes"] {
        let key = format!("{sibling}.rs");
        if let Some(content) = artifacts.rust_files.get(&key) {
            wrap_module(sibling, content);
        }
    }
    if let Some(lib_content) = artifacts.rust_files.get("lib.rs") {
        for line in lib_content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub mod ") || trimmed.starts_with("#![") {
                continue;
            }
            generated.push_str(line);
            generated.push('\n');
        }
    }

    fs::write(out_dir.join("generated.rs"), generated)
        .unwrap_or_else(|e| panic!("write generated.rs: {e}"));

    // Plan E-A2 — emit `runtime_core.rs` placeholder. Subsequent
    // slices (A3 buffer alloc + try_new, A4 step() body) populate
    // it with the mechanical lib.rs body that today every fixture
    // hand-writes. For A2 we just prove the wiring works: file lands
    // in OUT_DIR, doesn't get included anywhere yet, doesn't break
    // any fixture's existing build.
    // Per-fixture ability-corpus file list — drives runtime registry
    // construction inside the synthesized try_new(). Each entry is the
    // bare file name (e.g. "Strike.ability"); the synthesized code
    // joins it with `<workspace>/assets/ability_test/<fixture>/` via
    // `concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/...")` so the
    // contents are baked into the binary via include_str!. Empty list
    // (no corpus) → the placeholder no-op-program path stays in effect.
    let ability_file_names: Vec<String> =
        ability_files.iter().map(|(n, _)| n.clone()).collect();
    // Pair-keyed materialized-view detection. Any `view name(p1: Agent,
    // p2: <Entity>) -> ... { ... }` declared `@materialized` (any
    // storage hint, but in practice `pair_map`) writes its fold output
    // into `view_storage_primary` at index `[p1 * cfg.second_key_pop
    // + p2]`. The backing buffer therefore needs `agent_count *
    // <second_key_population>` slots, not the per-agent default.
    //
    // T6 fix (2026-05-11): generalised from Agent×Agent only (a single
    // bool) to a `PairKeyedSecondKey` carrying the second-key kind +
    // static count. Agent second key keeps `agent_count` (per-tick
    // variable, tom_probe shape); Item / Group / Quest second keys
    // resolve to the static count of declared entities of that root
    // (trade_caravans's `view inventory(merchant: Agent, good: Item)`
    // was the in-tree trigger — 3 declared Items resolve to a slot
    // count of `agent_count * 3` instead of the previous `agent_count`
    // under-allocation). See [`detect_pair_keyed_second_key`].
    //
    // Single-agent or scalar-keyed views (e.g. dodger_probe's `view
    // threats(observer: Agent) -> f32`) keep the per-agent sizing
    // (detector returns `None`).
    let pair_keyed_second_key = detect_pair_keyed_second_key(&comp);
    // Per-view metadata (one entry per `@materialized` view) — drives
    // the per-view storage-buffer allocation in
    // `synthesize_generated_runtime_struct` so each fold/decay kernel's
    // `view_storage_primary` BGL binding routes to its OWN
    // `view_storage_<view_name>_primary_buf` instead of the shared
    // (aliased) buffer the legacy emitter produced. Closes the 6-fixture
    // aliasing gap (forest_fire/squad_skirmish/plague_city/
    // detective_investigation/palace_coup/among_us).
    let materialized_views = collect_materialized_views(&comp);
    // Voxel-binding detection. If any kernel binds `voxel_grid` then the
    // synthesized runtime needs a CPU-side `engine_voxel::VoxelTerrain`
    // + GPU-side `engine_voxel::VoxelMirror` so `KernelBindingsContext::
    // voxel_grid` is `Some(...)` instead of `None` (which panics at
    // try_new() the moment any kernel `.expect()`s the unwrap). Walking
    // every kernel's binding list by name mirrors the existing
    // `is_infra_binding` classification path — `voxel_grid` is the
    // infra binding routed through `KernelBindingsContext`, so the
    // detection is just "does any spec have a binding by that name".
    // Fixtures with no voxel use stay zero-overhead (no buffer alloc,
    // no per-tick flush).
    let binds_voxel_grid = artifacts
        .kernel_specs
        .iter()
        .any(|spec| spec.bindings.iter().any(|b| b.name == "voxel_grid"));
    // Voxel-region-indices Phase 4b — sibling of `binds_voxel_grid`.
    // Detects whether any kernel binds the `navgrid` storage emitted
    // by the `navgrid.walkable(...)` namespace surface. The runtime
    // allocates `navgrid_buf` + `navgrid_cfg_buf` when true, and the
    // KernelBindingsContext routes them via the `binds_navgrid` arm
    // of the from_context template. Zero-overhead when no fixture
    // calls navgrid.*.
    let binds_navgrid = artifacts
        .kernel_specs
        .iter()
        .any(|spec| spec.bindings.iter().any(|b| b.name == "navgrid"));

    // Walk the synthesized schedule and collect the set of kernel names
    // that classify as `KernelTopology::Indirect` consumers — the
    // chronicle-driven `apply_ability` consumers that read events out
    // of the unified ring. The synthesized step() uses this list to
    // (a) snapshot the host-side event-ring tail BEFORE clear_tail_in,
    // and (b) overwrite each Indirect consumer's cfg `event_count`
    // slot with the snapshot. Without this, the consumers would walk
    // `agent_count` slots every tick — re-firing on stale records and
    // trampling the inject-before-step lifecycle (gaps 3 + 4 from
    // commit 353527e6's Indirect-arm doc block).
    //
    // `fold_*` kernels (PerAgentEventScan ViewFolds) classify as
    // `Indirect` at the topology layer but the schedule emits them as
    // `DispatchOp::Kernel(...)` (see `classify_topology_for_schedule`
    // in `cg/emit/cross_cutting.rs`); they handle event counting in
    // their own bodies and don't need the cfg overwrite. Filter them
    // out here so the runtime list only includes "true" Indirect
    // consumers (`physics_Apply*`-shaped chronicle handlers).
    let mut indirect_consumer_kernel_names: Vec<String> = Vec::new();
    for stage in &schedule_result.schedule.stages {
        for topology in &stage.kernels {
            if !matches!(topology, crate::cg::schedule::synthesis::KernelTopology::Indirect { .. }) {
                continue;
            }
            let name = match crate::cg::emit::kernel::semantic_kernel_name_for_topology(
                topology, &cg,
            ) {
                Some(n) => n,
                None => continue,
            };
            if name.starts_with("fold_") {
                continue;
            }
            if !indirect_consumer_kernel_names.contains(&name) {
                indirect_consumer_kernel_names.push(name);
            }
        }
    }
    // Gap T2 fix (2026-05-12): count Item-rooted and Group-rooted
    // entity declarations. Used by the alloc loop to size
    // `item_<field>_buf` / `group_<field>_buf` buffers — one slot
    // per entity of that root.
    let item_entity_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, dsl_ast::ast::EntityRoot::Item))
        .count() as u32;
    let group_entity_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, dsl_ast::ast::EntityRoot::Group))
        .count() as u32;
    let runtime_core = synthesize_runtime_core_a2(
        fixture_name,
        &artifacts,
        &init_stmts,
        &runtime_config_defaults,
        &ability_file_names,
        &comp.events,
        pair_keyed_second_key,
        &materialized_views,
        binds_voxel_grid,
        binds_navgrid,
        &indirect_consumer_kernel_names,
        item_entity_count,
        group_entity_count,
        needs_sort,
        sort_layout.as_ref(),
    );
    fs::write(out_dir.join("runtime_core.rs"), runtime_core)
        .unwrap_or_else(|e| panic!("write runtime_core.rs: {e}"));

    // T11 — conditional terrain_gen.rs emit. Present only when the source
    // has a `terrain { ... }` block; skipped entirely for all existing
    // fixtures that don't declare terrain (so their OUT_DIR stays clean).
    if let Some(tb) = terrain_block {
        let ir = crate::cg::lower::lower_terrain(&tb)
            .unwrap_or_else(|e| panic!("lower terrain for `{fixture_name}`: {e:?}"));
        let body = crate::cg::emit::emit_terrain(&ir);
        fs::write(out_dir.join("terrain_gen.rs"), body)
            .unwrap_or_else(|e| panic!("write {fixture_name}/terrain_gen.rs: {e}"));
    }
}

/// Per-kernel `@runtime` config field. Carried from the resolved
/// Compilation IR through synthesize_runtime_core so try_new can
/// initialize each cfg buffer's runtime field with its .sim default
/// and the generator can emit a host-side setter.
#[derive(Clone)]
pub struct RuntimeConfigDefault {
    pub scalar_ty: String,
    pub default_lit: String,
}

/// AOE auto-detect: scan the .ability corpus for any program with a
/// non-None `EffectAreaShape` and return whether the dispatcher needs
/// AOE Path B (spatial walk + per-target chronicle write).
///
/// Two paths:
///
/// 1. **Happy path** — `built_registry` is `Some(_)`: every .ability
///    lowered cleanly AND `build_registry` succeeded (no duplicate
///    names, no unresolved cast targets, no cast cycles). Walk the
///    frozen registry's slots in id order; return `true` on the first
///    program with any per-effect AOE shape.
///
/// 2. **Partial-failure fallback** — `built_registry` is `None`:
///    EITHER there's no corpus, OR `build_registry` returned `Err(_)`
///    because at least one .ability didn't lower (canonical case:
///    `Daze.ability` with `stun 8` — see Gap squad_skirmish#A). Pre-
///    this-helper the bare `unwrap_or(false)` then silently disabled
///    AOE Path B emit for the WHOLE corpus, including .ability peers
///    that DO declare AOE shapes (`Volley.ability`'s
///    `damage 6 in spread(4.0, 8)`). Now we walk each decl
///    individually via `lower_ability_decl` (NOT `lower_ability_file`,
///    which short-circuits on the first decl error) and union the AOE
///    signal across every program that DID lower. Per-decl lowering
///    errors are treated as `no AOE in that decl` and skipped silently
///    — the registry-build path already surfaced them via
///    `cargo:warning`.
///
/// The fallback mirrors the spirit of `LowerOutcome` (#140): one bad
/// apple shouldn't poison the rest of the file, and the same principle
/// extends across the corpus.
pub fn detect_aoe_dispatch(
    built_registry: Option<&crate::ability_registry::BuiltRegistry>,
    ability_files:  &[(String, dsl_ast::AbilityFile)],
) -> bool {
    if let Some(br) = built_registry {
        let n = br.registry.len();
        // AbilityId is a NonZero* newtype starting at 1; iterate the
        // 1..=n range and skip ids the registry rejects (defensive —
        // build_registry is contiguous today).
        return (1..=n).any(|i| {
            engine::ability::AbilityId::new(i as u32)
                .and_then(|id| br.registry.get(id))
                .map(|p| p.per_effect_areas.iter().any(|a| a.is_some()))
                .unwrap_or(false)
        });
    }
    ability_files.iter().any(|(_, file)| {
        file.abilities.iter().any(|decl| {
            crate::ability_lower::lower_ability_decl(decl)
                .map(|p| p.per_effect_areas.iter().any(|a| a.is_some()))
                .unwrap_or(false)
        })
    })
}

/// Recursive walk over a physics rule body looking for any
/// `agents.set_beliefs_<field>(observer, subject, value)` call. The
/// AST surface is `IrExpr::NamespaceCall { ns: Agents, method:
/// "set_beliefs_*", .. }` matched by `lower::physics::lower_expr_stmt`
/// at line 861. Mirroring the same name-based gate here lets the
/// build helper auto-set `LowerOpts.belief_state` instead of forcing
/// every belief-state fixture to carry a hand-written build.rs that
/// flips the bit explicitly.
fn stmts_contain_set_beliefs_call(stmts: &[dsl_ast::ir::IrStmt]) -> bool {
    stmts.iter().any(stmt_contains_set_beliefs_call)
}

fn stmt_contains_set_beliefs_call(stmt: &dsl_ast::ir::IrStmt) -> bool {
    use dsl_ast::ir::IrStmt;
    match stmt {
        IrStmt::Let { value, .. } => expr_contains_set_beliefs_call(value),
        IrStmt::Emit(emit) => emit
            .fields
            .iter()
            .any(|f| expr_contains_set_beliefs_call(&f.value)),
        IrStmt::For { iter, filter, body, .. } => {
            expr_contains_set_beliefs_call(iter)
                || filter.as_ref().is_some_and(expr_contains_set_beliefs_call)
                || stmts_contain_set_beliefs_call(body)
        }
        IrStmt::ForEachAgent { body, .. } => stmts_contain_set_beliefs_call(body),
        IrStmt::If { cond, then_body, else_body, .. } => {
            expr_contains_set_beliefs_call(cond)
                || stmts_contain_set_beliefs_call(then_body)
                || else_body
                    .as_ref()
                    .is_some_and(|b| stmts_contain_set_beliefs_call(b))
        }
        IrStmt::Match { scrutinee, arms, .. } => {
            expr_contains_set_beliefs_call(scrutinee)
                || arms.iter().any(|a| stmts_contain_set_beliefs_call(&a.body))
        }
        IrStmt::SelfUpdate { value, .. } => expr_contains_set_beliefs_call(value),
        IrStmt::SelfAppend { fields, .. } => fields
            .iter()
            .any(|f| expr_contains_set_beliefs_call(&f.value)),
        IrStmt::Expr(e) => expr_contains_set_beliefs_call(e),
        IrStmt::BeliefObserve { observer, target, fields, .. } => {
            expr_contains_set_beliefs_call(observer)
                || expr_contains_set_beliefs_call(target)
                || fields
                    .iter()
                    .any(|f| expr_contains_set_beliefs_call(&f.value))
        }
        IrStmt::ApplyAbility { ability, caster, target, .. } => {
            expr_contains_set_beliefs_call(ability)
                || caster.as_ref().is_some_and(expr_contains_set_beliefs_call)
                || target.as_ref().is_some_and(expr_contains_set_beliefs_call)
        }
    }
}

fn expr_contains_set_beliefs_call(node: &dsl_ast::ir::IrExprNode) -> bool {
    use dsl_ast::ir::{IrExpr, NamespaceId};
    match &node.kind {
        IrExpr::NamespaceCall { ns: NamespaceId::Agents, method, args }
            if method.starts_with("set_beliefs_") =>
        {
            // Deep walk would also catch nested set_beliefs_* in the
            // value arg, but the immediate-call match is the only shape
            // the lower-time gate at physics.rs:861 keys off, so this
            // arm is sufficient. Still recurse into args so nested
            // composite uses are covered.
            let _ = args;
            true
        }
        IrExpr::NamespaceCall { args, .. } => args.iter().any(call_arg_contains),
        IrExpr::Field { base, .. } => expr_contains_set_beliefs_call(base),
        IrExpr::Index(a, b) => {
            expr_contains_set_beliefs_call(a) || expr_contains_set_beliefs_call(b)
        }
        IrExpr::ViewCall(_, args)
        | IrExpr::VerbCall(_, args)
        | IrExpr::BuiltinCall(_, args)
        | IrExpr::UnresolvedCall(_, args) => args.iter().any(call_arg_contains),
        IrExpr::Binary(_, a, b) => {
            expr_contains_set_beliefs_call(a) || expr_contains_set_beliefs_call(b)
        }
        IrExpr::Unary(_, a) => expr_contains_set_beliefs_call(a),
        IrExpr::In(a, b) | IrExpr::Contains(a, b) => {
            expr_contains_set_beliefs_call(a) || expr_contains_set_beliefs_call(b)
        }
        IrExpr::Quantifier { iter, body, .. } => {
            expr_contains_set_beliefs_call(iter) || expr_contains_set_beliefs_call(body)
        }
        IrExpr::Fold { iter, body, .. } => {
            iter.as_ref()
                .is_some_and(|i| expr_contains_set_beliefs_call(i))
                || expr_contains_set_beliefs_call(body)
        }
        IrExpr::List(items) | IrExpr::Tuple(items) => {
            items.iter().any(expr_contains_set_beliefs_call)
        }
        IrExpr::StructLit { fields, .. } => {
            fields.iter().any(|f| expr_contains_set_beliefs_call(&f.value))
        }
        IrExpr::Ctor { args, .. } => args.iter().any(expr_contains_set_beliefs_call),
        IrExpr::Match { scrutinee, arms } => {
            expr_contains_set_beliefs_call(scrutinee)
                || arms.iter().any(|a| expr_contains_set_beliefs_call(&a.body))
        }
        IrExpr::If { cond, then_expr, else_expr } => {
            expr_contains_set_beliefs_call(cond)
                || expr_contains_set_beliefs_call(then_expr)
                || else_expr
                    .as_ref()
                    .is_some_and(|e| expr_contains_set_beliefs_call(e))
        }
        IrExpr::PerUnit { expr, delta } => {
            expr_contains_set_beliefs_call(expr) || expr_contains_set_beliefs_call(delta)
        }
        // Leaf nodes — no nested expression to walk.
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Local(_, _)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::View(_)
        | IrExpr::Verb(_)
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. }
        | IrExpr::EnumVariant { .. }
        | IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit(_) => false,
        // Other leaf / legacy ToM-accessor variants don't surface
        // `agents.set_beliefs_*` calls — they're either pure leaves
        // (AbilityRange) or accessor reads (BeliefsAccessor /
        // BeliefsConfidence) that the resolver flags separately. A
        // catch-all keeps this walker robust to future IrExpr variants
        // — the auto-detect can over-trigger safely (the LowerOpts.
        // belief_state flag's only effect is to surface bindings that
        // would otherwise no-op, harmless on a fixture that never
        // calls set_beliefs_*).
        _ => false,
    }
}

fn call_arg_contains(arg: &dsl_ast::ir::IrCallArg) -> bool {
    expr_contains_set_beliefs_call(&arg.value)
}

/// True if `comp` declares ≥1 `@materialized` view with exactly two
/// `Agent` parameters (= `pair_map`-shaped storage). The auto-emitter
/// uses this to up-size `view_storage_primary_buf` to N² cells; see
/// [`slot_count_expr`] for the per-binding sizing rule.
///
/// Public for integration-test access (the `tests/` crate exercises
/// the full call chain `parse → resolve → detect → synthesize`).
///
/// **Generalised in T6 fix (2026-05-11):** delegates to
/// [`detect_pair_keyed_second_key`] which now accepts a non-Agent
/// second key (`Item` / `Group` / `Quest`). The bool-returning helper
/// stays for tom_probe-shape pins that only care "is there ANY
/// pair-keyed view"; sizing-aware callers should use the richer form.
pub fn detect_pair_keyed_materialized_view(comp: &dsl_ast::ir::Compilation) -> bool {
    detect_pair_keyed_second_key(comp).is_some()
}

/// Second-key shape of a fixture's pair-keyed `@materialized` view.
/// Returned by [`detect_pair_keyed_second_key`].
///
/// The first key is always `Agent` (the only shape the resolve+lower
/// pipeline currently supports for the host-emit side). The second
/// key can be any entity-rooted type — for Agent the per-tick
/// population is the runtime `agent_count`; for Item / Group / Quest
/// the population is the static count of declared entities of that
/// root in the .sim source.
///
/// `Agent` is special-cased so the slot-count expression stays
/// `agent_count * agent_count` (matches the pre-T6 sizing for
/// tom_probe). Other variants carry their static count so the slot-
/// count expression resolves to `agent_count * <count>` at compile
/// time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairKeyedSecondKey {
    /// Agent×Agent — second-key population varies per tick. Slot-
    /// count expression: `agent_count * agent_count`.
    Agent,
    /// Agent×Item — second-key population is the static count of
    /// declared `entity X : Item` decls in the .sim. trade_caravans
    /// declares 3 (Grain / Spice / Silk).
    Item(u32),
    /// Agent×Group — second-key population is the static count of
    /// declared `entity X : Group` decls in the .sim.
    Group(u32),
    /// Agent×Quest — second-key population is the static count of
    /// declared `entity X : Quest` decls in the .sim.
    Quest(u32),
    /// Plan I slice I.3b — Agent×(u8|u32|i32). Second-key population
    /// is declared explicitly via `@key_pop(K = N)` on the belief
    /// decl. Use when the belief is keyed on something other than
    /// an entity-rooted type (chunk_id, room_id, tag_bit, etc.) and
    /// the population is bounded by the .sim author's domain (e.g.
    /// `K = 64` for a 64-chunk world map). The runtime-side slot
    /// count is `agent_count * K`, so storage scales O(N×K) instead
    /// of O(N²) — the canonical pattern for "neighbourhood" beliefs
    /// where each agent only tracks a small fixed-size keyspace.
    KeyTyped(u32),
}

impl PairKeyedSecondKey {
    /// Static second-key population if known at compile time. `None`
    /// for `Agent` (the per-tick `agent_count` runtime variable).
    pub fn static_count(self) -> Option<u32> {
        match self {
            PairKeyedSecondKey::Agent => None,
            PairKeyedSecondKey::Item(n)
            | PairKeyedSecondKey::Group(n)
            | PairKeyedSecondKey::Quest(n)
            | PairKeyedSecondKey::KeyTyped(n) => Some(n),
        }
    }

    /// Rust expression (in the synthesized `try_new` / `step` body
    /// scope where `agent_count: u32` is in scope) that evaluates to
    /// the second-key population. Used by [`slot_count_expr`].
    pub fn population_u64_expr(self) -> String {
        match self {
            PairKeyedSecondKey::Agent => "(agent_count as u64)".to_string(),
            PairKeyedSecondKey::Item(n)
            | PairKeyedSecondKey::Group(n)
            | PairKeyedSecondKey::Quest(n)
            | PairKeyedSecondKey::KeyTyped(n) => format!("{n}u64"),
        }
    }
}

/// Per-`@materialized`-view metadata threaded into the runtime
/// synthesizer for per-view storage-buffer allocation. Each entry
/// pairs a view name (the `<view_name>` in `view <view_name>(...)
/// -> ... { @materialized }`) with the shape of its backing storage
/// (per-agent for single-key views; pair-keyed with a known
/// second-key population for two-key views).
///
/// Used by the per-view storage allocator to size each view's own
/// `view_storage_<view_name>_primary_buf` independently — closes the
/// 6-fixture aliasing gap (forest_fire/squad_skirmish/plague_city/
/// detective_investigation/palace_coup/among_us) where the previous
/// shared `view_storage_primary_buf` aliased every view's writes
/// into a single buffer.
#[derive(Debug, Clone)]
pub struct MaterializedViewInfo {
    /// Snake_case view name as declared in the .sim source (e.g.
    /// `damage_dealt`, `threat_taken`, `evidence`). Matches the
    /// `<view>` segment of the kernel name (`fold_<view>` /
    /// `decay_<view>`) and the per-view BGL binding (`view_storage_
    /// <view>_primary` for read accessors).
    pub name: String,
    /// Pair-keyed second-key population (`Some(_)`) for views with
    /// two key parameters where the first is `Agent`; `None` for
    /// per-agent (single-key) views. Drives per-view buffer sizing
    /// in the synthesized `try_new`.
    pub pair_keyed: Option<PairKeyedSecondKey>,
    /// Plan G G3f follow-up — `@per_entity_ring(K)` ring length, when
    /// the view is annotated as a per-entity ring. Multiplied into the
    /// per-view storage allocation alongside `cell_stride_u32` so
    /// struct-cell ring views (e.g. `threats` with the 8-field
    /// ThreatZoneCell layout) get `agent_count * K * 8 * 4` bytes
    /// instead of the scalar-default `agent_count * 4`. `None` for
    /// every other storage shape (per-agent scalar, pair-keyed map,
    /// top-K).
    pub ring_k: Option<u16>,
    /// Per-cell u32 stride for struct-cell ring views. `1` for scalar
    /// payloads (the existing `@per_entity_ring(K)` shape that
    /// per_entity_ring_probe.sim exercises), `>= 2` for views whose
    /// fold body uses `self.append(field1, field2, ...)` and the
    /// lowering registered a `ViewLayout`. Combined with `ring_k`
    /// above to size `view_storage_<view>_primary_buf`.
    pub cell_stride_u32: u32,
}

/// Walk `comp.views` and return one [`MaterializedViewInfo`] per
/// `@materialized` view declaration, in declaration order. Used by
/// the runtime synthesizer to allocate one
/// `view_storage_<view_name>_primary_buf` per view and route each
/// fold/decay kernel's `view_storage_primary` BGL binding to its
/// own backing buffer (instead of the shared
/// `view_storage_primary_buf` that aliased every view together).
pub fn collect_materialized_views(
    comp: &dsl_ast::ir::Compilation,
) -> Vec<MaterializedViewInfo> {
    use dsl_ast::ast::EntityRoot;
    use dsl_ast::ir::{IrType, StorageHint, ViewKind};
    let item_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, EntityRoot::Item))
        .count() as u32;
    let group_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, EntityRoot::Group))
        .count() as u32;
    let quest_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, EntityRoot::Quest))
        .count() as u32;
    let mut out = Vec::new();
    for v in &comp.views {
        // Plan I (2026-05-15) — `belief` decls share the per-view
        // storage allocator with materialized views; their PairMap
        // sizing comes from the (Agent, Agent) signature inference
        // rather than the kind discriminant.
        if !matches!(v.kind, ViewKind::Materialized(_) | ViewKind::Belief) {
            continue;
        }
        let pair_keyed = if v.params.len() == 2
            && matches!(v.params[0].ty, IrType::AgentId)
        {
            // Mirrors `detect_pair_keyed_second_key`'s per-view
            // classification — the second param's type picks the
            // pair-keyed second-key population.
            match &v.params[1].ty {
                IrType::AgentId => Some(PairKeyedSecondKey::Agent),
                IrType::ItemId => Some(PairKeyedSecondKey::Item(item_count.max(1))),
                IrType::GroupId => Some(PairKeyedSecondKey::Group(group_count.max(1))),
                IrType::QuestId => Some(PairKeyedSecondKey::Quest(quest_count.max(1))),
                // Plan I slice I.3b — key-typed second param keeps
                // the per-view buffer sized at `agent_cap × K` cells.
                // K comes from `@key_pop(K = N)` on the belief decl.
                IrType::U8 | IrType::U32 | IrType::I32 => v
                    .annotations
                    .iter()
                    .find(|a| a.name == "key_pop")
                    .and_then(extract_annotation_k_arg)
                    .map(|k| PairKeyedSecondKey::KeyTyped(k.max(1))),
                IrType::Named(n) => match n.as_str() {
                    "Agent" => Some(PairKeyedSecondKey::Agent),
                    "Item" => Some(PairKeyedSecondKey::Item(item_count.max(1))),
                    "Group" => Some(PairKeyedSecondKey::Group(group_count.max(1))),
                    "Quest" => Some(PairKeyedSecondKey::Quest(quest_count.max(1))),
                    _ => None,
                },
                _ => None,
            }
        } else {
            None
        };
        // Plan G G3f follow-up — inspect the storage hint + fold body
        // to surface ring-K and per-cell stride for `@per_entity_ring(K)`
        // views with struct-cell payloads. The cell stride is implied
        // by the first `self.append(field1, field2, ...)` statement in
        // the fold body — same convention the CG-side `ViewLayout`
        // registers post-lowering. Defaults: scalar payload (stride 1)
        // for either non-ring storage or a `self += <expr>` body.
        // Plan I slice I.3c — belief decls carry `@per_entity_ring(K=N)`
        // on the annotation list rather than baked into the kind
        // discriminant. Inspect both: materialized views via the
        // hint, beliefs via the annotation arg.
        let ring_k = match v.kind {
            ViewKind::Materialized(StorageHint::PerEntityRing { k }) => Some(k),
            ViewKind::Belief => v
                .annotations
                .iter()
                .find(|a| a.name == "per_entity_ring")
                .and_then(|ann| {
                    ann.args.iter().find_map(|arg| {
                        if arg.key.as_deref() == Some("K") {
                            if let dsl_ast::ast::AnnotationValue::Int(n) = &arg.value {
                                Some((*n).clamp(1, u16::MAX as i64) as u16)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                }),
            _ => None,
        };
        let cell_stride_u32: u32 = if let dsl_ast::ir::ViewBodyIR::Fold { handlers, .. } = &v.body {
            let mut stride = 1u32;
            for h in handlers {
                for s in &h.body {
                    if let dsl_ast::ir::IrStmt::SelfAppend { fields, .. } = s {
                        // First-match wins; conflicting field counts
                        // across handlers surface as a typed lowering
                        // error during CG (ViewLayout register
                        // returns the prior entry).
                        stride = fields.len() as u32;
                        break;
                    }
                }
                if stride > 1 {
                    break;
                }
            }
            stride
        } else {
            1
        };
        out.push(MaterializedViewInfo {
            name: v.name.clone(),
            pair_keyed,
            ring_k,
            cell_stride_u32,
        });
    }
    out
}

/// Returns the second-key shape if `comp` declares ≥1 `@materialized`
/// view with exactly two parameters where the first is `Agent` and
/// the second is some entity-rooted type. Returns `None` otherwise.
///
/// **First-match wins** when a fixture mixes second-key shapes (e.g.
/// one Agent×Agent view + one Agent×Item view). Today the fold body
/// indexes `view_storage_primary[k1 * cfg.second_key_pop + k2]`
/// where `cfg.second_key_pop` is uploaded per-kernel by hand (or
/// inherited from the auto-emitter's hardcoded `1u32`). The
/// auto-emitter's slot-count expression sizes the SHARED storage
/// buffer big enough for the LARGEST second-key population — picking
/// `Agent` (whose count is `agent_count`, typically the largest)
/// when both shapes coexist keeps every view safely backed.
///
/// The mixed-key case isn't exercised by any in-tree fixture today;
/// every fixture has at most one pair-keyed view. The first-match
/// fallback keeps the contract single-valued for the synthesizer.
pub fn detect_pair_keyed_second_key(
    comp: &dsl_ast::ir::Compilation,
) -> Option<PairKeyedSecondKey> {
    use dsl_ast::ast::EntityRoot;
    use dsl_ast::ir::{IrType, ViewKind};
    // Static counts — walked once and reused across all candidate
    // views. Cheap (entities is a few-element list in every fixture).
    let item_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, EntityRoot::Item))
        .count() as u32;
    let group_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, EntityRoot::Group))
        .count() as u32;
    let quest_count = comp
        .entities
        .iter()
        .filter(|e| matches!(e.root, EntityRoot::Quest))
        .count() as u32;
    // Prefer the largest second-key shape across all pair-keyed views
    // so the shared `view_storage_primary` buffer is sized for the
    // worst case. Agent (= agent_count) dominates the static counts
    // in every realistic fixture, so the order is: Agent first, then
    // the largest static count.
    let mut best: Option<PairKeyedSecondKey> = None;
    for v in &comp.views {
        // Plan I (2026-05-15) — `belief` decls share the per-view
        // storage path with `@materialized` views; their PairMap
        // hint is inferred from the (Agent, Agent) signature rather
        // than carried on the kind enum. Treat both kinds the same
        // for sizing purposes.
        if !matches!(v.kind, ViewKind::Materialized(_) | ViewKind::Belief) {
            continue;
        }
        if v.params.len() != 2 {
            continue;
        }
        // First param must be Agent — host-side sizing keys off
        // `agent_count` for the first dimension. Non-Agent first
        // params aren't supported by the storage shape today.
        if !matches!(v.params[0].ty, IrType::AgentId) {
            continue;
        }
        // The resolve pass lowers the surface keyword as follows:
        //   * `Agent` → `IrType::AgentId` (stdlib alias entry)
        //   * `Item` / `Group` / `Quest` → `IrType::Named("Item"|...)`
        //     because there's no stdlib alias; the names ARE the
        //     entity-root keywords (and not the names of declared
        //     entities). They fall through `resolve_type`'s
        //     `entities` lookup (entities are `Grain`/`Spice`/etc.,
        //     not the root name) and land in the `Named` fallback.
        // The auto-emitter therefore matches BOTH the niche-id
        // variants (`IrType::ItemId` etc., for completeness — these
        // appear when the param is typed against a concrete event-
        // field shape rather than the entity-root keyword) AND the
        // `Named("Item"|"Group"|"Quest")` variants (the surface form
        // every in-tree fixture uses).
        let candidate = match &v.params[1].ty {
            IrType::AgentId => PairKeyedSecondKey::Agent,
            IrType::ItemId => PairKeyedSecondKey::Item(item_count.max(1)),
            IrType::GroupId => PairKeyedSecondKey::Group(group_count.max(1)),
            IrType::QuestId => PairKeyedSecondKey::Quest(quest_count.max(1)),
            // Plan I slice I.3b — key-typed second param (chunk_id,
            // room_id, tag_bit, etc.). Population is declared by the
            // .sim author via `@key_pop(K = N)` on the belief decl;
            // mirrors `@per_entity_ring(K = N)`'s K argument shape.
            // No annotation → skip (the storage-hint inferrer raises
            // `UnsupportedBeliefShape` with a pointer to slice I.3b).
            IrType::U8 | IrType::U32 | IrType::I32 => {
                let k = v
                    .annotations
                    .iter()
                    .find(|a| a.name == "key_pop")
                    .and_then(|a| extract_annotation_k_arg(a));
                match k {
                    Some(k) => PairKeyedSecondKey::KeyTyped(k.max(1)),
                    None => continue,
                }
            }
            IrType::Named(n) => match n.as_str() {
                "Agent" => PairKeyedSecondKey::Agent,
                "Item" => PairKeyedSecondKey::Item(item_count.max(1)),
                "Group" => PairKeyedSecondKey::Group(group_count.max(1)),
                "Quest" => PairKeyedSecondKey::Quest(quest_count.max(1)),
                _ => continue,
            },
            _ => continue,
        };
        // Pick the variant with the largest population. Agent always
        // wins ties (its per-tick population is unbounded above). If
        // we already saw Agent, keep it.
        match (best, candidate) {
            (None, c) => best = Some(c),
            (Some(PairKeyedSecondKey::Agent), _) => {}
            (Some(_), PairKeyedSecondKey::Agent) => best = Some(candidate),
            (Some(prev), c) => {
                if c.static_count().unwrap_or(0) > prev.static_count().unwrap_or(0) {
                    best = Some(c);
                }
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Sort kernel detection + injection helpers
// ---------------------------------------------------------------------------

/// Detect whether a fixture needs the radix sort pass.
///
/// Returns the `EventLayout` of the first f32+Add view fold's source event,
/// or `None` when no such view exists. The layout carries the record stride
/// used to size the sort scratch buffer and the field map used to find the
/// `target` word offset for `SortCfg`.
fn sort_layout_for_fixture(
    cg: &crate::cg::program::CgProgram,
) -> Option<crate::cg::program::EventLayout> {
    use crate::cg::expr::CgTy;
    use crate::cg::op::ComputeOpKind;
    use crate::cg::program::ViewFoldOp;

    // Find the first ViewFold op whose view has f32 result and Add/Sub fold.
    for op in &cg.ops {
        if let ComputeOpKind::ViewFold { view, on_event, .. } = &op.kind {
            let sig = cg.view_signatures.get(&view.0)?;
            let is_f32 = matches!(sig.result, CgTy::F32);
            let is_add_or_sub = matches!(
                sig.fold_op,
                Some(ViewFoldOp::Add) | Some(ViewFoldOp::Sub)
            );
            if is_f32 && is_add_or_sub {
                // Found a qualifying fold — return the event layout.
                if let Some(layout) = cg.event_layouts.get(&on_event.0) {
                    return Some(layout.clone());
                }
            }
        }
    }
    None
}

/// Compute the `target_word_offset` for the `SortCfg` uniform from an event
/// layout.
///
/// The offset is `header_word_count + field.word_offset_in_payload` for the
/// field named `"target"`. Falls back to `header_word_count + 1` (the second
/// payload word, a reasonable default for two-AgentId events) when the
/// layout has no field named `"target"`.
fn sort_target_word_offset(layout: &crate::cg::program::EventLayout) -> u32 {
    if let Some(field) = layout.fields.get("target") {
        return layout.header_word_count + field.word_offset_in_payload;
    }
    // Fall back: second payload word after the header.
    layout.header_word_count + 1
}

/// Inject 15 sort kernel WGSL files (4 × Stage A passes × 3 phases + 3
/// Stage B phases) and minimal Rust stub modules into `artifacts`.
///
/// The 15 kernels are complete WGSL files produced by
/// `cg::emit::sort_kernel::{emit_stage_a_pass, emit_stage_b}`; they
/// include their own `@group(0) @binding(N)` declarations and
/// `@compute @workgroup_size(...)` entry points. They therefore bypass
/// the normal `compose_wgsl_file` wrapper and are inserted directly into
/// `artifacts.wgsl_files`.
///
/// Each kernel also gets a minimal Rust stub in `artifacts.rust_files` and
/// an entry in `artifacts.kernel_index` so `emit_into`'s
/// `wrap_module` loop finds the stub file.
fn inject_sort_kernels(
    artifacts: &mut crate::cg::emit::EmittedArtifacts,
    layout: &crate::cg::program::EventLayout,
) {
    use crate::cg::emit::sort_kernel::{emit_stage_a_pass, emit_stage_b};

    // Stage A: 4 passes × 3 phases = 12 kernels.
    for pass_idx in 0..4u32 {
        let (hist_wgsl, scan_wgsl, scatter_wgsl) = emit_stage_a_pass(pass_idx, layout);
        for (phase, wgsl) in [
            ("histogram", hist_wgsl),
            ("scan", scan_wgsl),
            ("scatter", scatter_wgsl),
        ] {
            let name = format!("radix_stage_a_pass{}_{}", pass_idx, phase);
            artifacts.wgsl_files.insert(format!("{name}.wgsl"), wgsl);
            artifacts.rust_files.insert(format!("{name}.rs"), sort_kernel_stub_rs(&name));
            artifacts.kernel_index.push(name);
        }
    }

    // Stage B: 3 phases = 3 kernels.
    let (count_wgsl, scan_wgsl, scatter_wgsl) = emit_stage_b(layout);
    for (phase, wgsl) in [
        ("count", count_wgsl),
        ("scan", scan_wgsl),
        ("scatter", scatter_wgsl),
    ] {
        let name = format!("radix_stage_b_{}", phase);
        artifacts.wgsl_files.insert(format!("{name}.wgsl"), wgsl);
        artifacts.rust_files.insert(format!("{name}.rs"), sort_kernel_stub_rs(&name));
        artifacts.kernel_index.push(name);
    }
}

/// Minimal Rust stub for a sort kernel module. The stub compiles cleanly
/// inside `pub mod {name} { ... }` (the `wrap_module` wrapper in `emit_into`)
/// and provides a `SHADER_SRC` const so any code scanning for that pattern
/// can find the associated WGSL file. No Kernel trait impl is emitted —
/// sort kernels are driven directly by the runtime without going through the
/// compiler-synthesized dispatch plumbing.
fn sort_kernel_stub_rs(name: &str) -> String {
    format!(
        "// Sort kernel stub — WGSL is a complete shader, not a body fragment.\n\
         // Dispatch and buffer binding are managed by the runtime directly.\n\
         pub const SHADER_SRC: &str = include_str!(\"{name}.wgsl\");\n",
    )
}

/// Rewrite f32+Add view-fold WGSL files in-place to use a per-slot serial scan
/// instead of a per-event CAS loop. Called only when `needs_sort` is true.
///
/// The pre-sort fold dispatches one thread per event (gid.x = event_idx) and
/// uses `atomicCompareExchangeWeak` to accumulate f32 values, which races when
/// multiple events target the same slot. Post-sort, events are ordered by
/// (target, seq) so each slot's events are adjacent. We switch to one thread
/// per slot (gid.x = my_slot) with a sequential scan over the full sorted ring
/// — single-writer per slot, no race, deterministic sum.
///
/// The cfg struct keeps the same layout `{ event_count, tick, second_key_pop,
/// _pad0 }`. The runtime writes `agent_count` into `_pad0` (slot 3) so the
/// per-slot bounds check `my_slot >= cfg._pad0` works correctly. The inner loop
/// uses `cfg.event_count` (set to the prev-tick tail via encoder
/// copy_buffer_to_buffer before dispatch) for the event count so it scans
/// exactly the sorted ring from the prior tick.
///
/// Only WGSL files whose body contains the f32 CAS pattern are modified; other
/// fold shapes (u32 atomicAdd, PerAgentEventScan, etc.) are left unchanged.
fn rewrite_fold_wgsl_for_post_sort(wgsl_files: &mut std::collections::BTreeMap<String, String>) {
    for (_name, body) in wgsl_files.iter_mut() {
        // Only rewrite kernels that contain the f32 CAS+add loop.
        if !body.contains("atomicCompareExchangeWeak") {
            continue;
        }
        if !body.contains("let event_idx = gid.x;") {
            continue;
        }
        // Step 1: replace per-event preamble with per-slot preamble.
        // cfg.event_count is set to the prev-tick tail via encoder copy_buffer_to_buffer
        // so it holds the correct count for the inner loop. cfg._pad0 is written at
        // runtime with agent_count so the per-slot bounds check is correct.
        // The preamble always occupies these two lines immediately after the
        // fn signature opening brace.
        *body = body.replace(
            "    let event_idx = gid.x;\n    if (event_idx >= cfg.event_count) { return; }\n    let tick = cfg.tick;\n",
            "    let my_slot = gid.x;\n    if (my_slot >= cfg._pad0) { return; }\n    let tick = cfg.tick;\n",
        );
        // Step 2: rewrite each per-op CAS block. Each block has the form:
        //   if (event_ring[event_idx * STRIDEu + 0u] == KINDu) {
        //       let local_0: u32 = event_ring[event_idx * STRIDEu + TARGETu];
        //       loop {
        //           let _idx = local_0;
        //           let old = atomicLoad(&view_storage_primary[_idx]);
        //           let new_val = bitcast<u32>(bitcast<f32>(old) + (RHS));
        //           let result = atomicCompareExchangeWeak(...);
        //           if (result.exchanged) { break; }
        //       }
        //   }
        // We extract STRIDE, KIND, TARGET, RHS and emit a serial scan.
        *body = rewrite_cas_blocks_to_serial_scan(body);
    }
}

/// Replace every CAS+add f32 block in `wgsl` with a per-slot serial scan body.
/// Returns the rewritten string.
fn rewrite_cas_blocks_to_serial_scan(wgsl: &str) -> String {
    use std::fmt::Write;

    // Pattern markers we look for:
    //   "if (event_ring[event_idx * STRIDEu + 0u] == KINDu) {"
    //   "let local_0: u32 = event_ring[event_idx * STRIDEu + TARGETu];"
    //   "let new_val = bitcast<u32>(bitcast<f32>(old) + (RHS));"
    // We do a line-by-line scan and replace the matching block.

    let lines: Vec<&str> = wgsl.lines().collect();
    let mut out = String::with_capacity(wgsl.len() + 256);
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Detect the tag-check line: "if (event_ring[event_idx * STRIDEu + 0u] == KINDu) {"
        if trimmed.starts_with("if (event_ring[event_idx * ") && trimmed.contains("+ 0u] == ") {
            // Extract STRIDE from "event_ring[event_idx * STRIDEu + 0u]"
            let stride = extract_u32_after(trimmed, "event_idx * ", "u + 0u]").unwrap_or(11);
            // Extract KIND from "== KINDu)"
            let kind = extract_u32_after(trimmed, "== ", "u)").unwrap_or(0);

            // Detect the leading indent from the original line.
            let indent = leading_spaces(line);

            // Expect the next line: "let local_0: u32 = event_ring[event_idx * STRIDEu + TARGETu];"
            let target_word = if i + 1 < lines.len() {
                let t = lines[i + 1].trim();
                extract_u32_after(t, &format!("event_idx * {stride}u + "), "u];").unwrap_or(stride - 1)
            } else {
                stride - 1
            };

            // Find the RHS in the new_val line: "let new_val = bitcast<u32>(bitcast<f32>(old) + (RHS));"
            let rhs = find_cas_rhs(&lines[i..]).unwrap_or_else(|| "1.0".to_string());

            // Count how many lines to skip (the whole tag-check + local_0 + loop block).
            // We skip until we find the closing "}" that matches the outer "if" block.
            let skip = count_block_lines(&lines[i..]);

            // Emit the replacement: per-slot serial scan.
            // cfg.event_count holds the prev-tick event tail (set via encoder
            // copy_buffer_to_buffer before dispatch) — use it for the inner
            // loop bound so we scan exactly the sorted ring from the prior tick.
            let _ = writeln!(
                out,
                "{indent}// Post-sort: per-slot serial scan over sorted ring — single writer, no CAS.\n\
                 {indent}{{\n\
                 {indent}    var _accum: f32 = bitcast<f32>(atomicLoad(&view_storage_primary[my_slot]));\n\
                 {indent}    for (var _i = 0u; _i < cfg.event_count; _i = _i + 1u) {{\n\
                 {indent}        if (event_ring[_i * {stride}u + 0u] == {kind}u) {{\n\
                 {indent}            if (event_ring[_i * {stride}u + {target_word}u] == my_slot) {{\n\
                 {indent}                _accum = _accum + ({rhs});\n\
                 {indent}            }}\n\
                 {indent}        }}\n\
                 {indent}    }}\n\
                 {indent}    atomicStore(&view_storage_primary[my_slot], bitcast<u32>(_accum));\n\
                 {indent}}}"
            );
            i += skip;
            continue;
        }

        out.push_str(line);
        out.push('\n');
        i += 1;
    }
    out
}

/// Extract a u32 literal from `s` that appears after `prefix` and before `suffix`.
fn extract_u32_after(s: &str, prefix: &str, suffix: &str) -> Option<u32> {
    let start = s.find(prefix)? + prefix.len();
    let rest = &s[start..];
    let end = rest.find(suffix)?;
    rest[..end].parse().ok()
}

/// Count the leading spaces (indent) of a line.
fn leading_spaces(s: &str) -> &str {
    let n = s.len() - s.trim_start().len();
    &s[..n]
}

/// Find the RHS expression in the CAS loop body within the given line slice.
/// Looks for `let new_val = bitcast<u32>(bitcast<f32>(old) + (RHS));`.
fn find_cas_rhs(lines: &[&str]) -> Option<String> {
    for line in lines.iter().take(20) {
        let t = line.trim();
        if t.starts_with("let new_val = bitcast<u32>(bitcast<f32>(old) + (") {
            let after = t.strip_prefix("let new_val = bitcast<u32>(bitcast<f32>(old) + (")?;
            let rhs = after.strip_suffix("));")?;
            return Some(rhs.to_string());
        }
    }
    None
}

/// Count lines to skip for the outer `if` block starting at `lines[0]`.
/// Counts from the opening `{` to the matching `}` at the same indent level,
/// including the closing `}` line itself. Returns 1 if no block is found.
fn count_block_lines(lines: &[&str]) -> usize {
    let mut depth: i32 = 0;
    for (idx, line) in lines.iter().enumerate() {
        for ch in line.chars() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    return idx + 1;
                }
            }
        }
    }
    lines.len()
}

/// Lightweight `K = <int>` extractor for build-side annotation lookup.
/// Returns `Some(k)` only on the happy path; ill-formed annotations
/// surface to the .sim author via the resolver's `annotation_k_arg`
/// (which builds rich `ResolveError`s) or via the lower-side
/// `extract_k_arg` (which builds `LoweringError::UnsupportedBeliefShape`).
/// This helper exists so `detect_pair_keyed_second_key` can stay
/// `Option`-returning without pulling either error type into
/// `build_helper`.
fn extract_annotation_k_arg(ann: &dsl_ast::ast::Annotation) -> Option<u32> {
    for arg in &ann.args {
        if arg.key.as_deref() == Some("K") {
            if let dsl_ast::ast::AnnotationValue::Int(n) = &arg.value {
                if *n > 0 && *n <= u32::MAX as i64 {
                    return Some(*n as u32);
                }
            }
        }
    }
    None
}

/// Plan E-A3.1 — placeholder generated runtime body, now with per-kernel
/// binding metadata derived from `EmittedArtifacts.kernel_specs` (added in
/// the same slice).
///
/// Today: emits a comment block listing every kernel and its
/// (slot, name, access, wgsl_ty, bg_source) bindings. No alloc / no
/// try_new yet — that's A3.2. The binding inventory is the data the
/// alloc emit will walk.
///
/// A3.2 will use this same `kernel_specs` walk to emit
/// `pub fn try_new(seed: u64, agent_count: u32) -> Option<Self>` with
/// per-binding buffer allocation. A4 layers a default `step()` body
/// that walks the SCHEDULE table and binds each kernel automatically.
///
/// `pair_keyed_second_key` is `Some(_)` if the fixture declares ≥1
/// `@materialized` view with two params where the first is `Agent`
/// (= `pair_map`-shaped storage). The variant carries the second-key
/// kind + static population. Threaded through to
/// `synthesize_generated_runtime_struct` so the alloc loop can size
/// `view_storage_primary_buf` as `agent_count *
/// <second_key_population>` u32 cells instead of the per-agent
/// default. Without this signal the per-(observer, source) fold body
/// would write past the end of the buffer at any `agent_count > 1`.
///
/// Generalised in T6 fix (2026-05-11) from `bool` (Agent×Agent only)
/// to a richer enum so non-Agent second keys (Item / Group / Quest —
/// trade_caravans's `view inventory(merchant: Agent, good: Item)` is
/// the in-tree trigger) get their proper static counts threaded
/// through.
pub fn synthesize_runtime_core_a2(
    fixture_name: &str,
    artifacts: &crate::cg::emit::EmittedArtifacts,
    init_stmts: &[dsl_ast::ast::InitStmt],
    runtime_config_defaults: &std::collections::BTreeMap<String, RuntimeConfigDefault>,
    ability_file_names: &[String],
    events: &[dsl_ast::ir::EventIR],
    pair_keyed_second_key: Option<PairKeyedSecondKey>,
    materialized_views: &[MaterializedViewInfo],
    binds_voxel_grid: bool,
    binds_navgrid: bool,
    indirect_consumer_kernel_names: &[String],
    // Gap T2 fix (2026-05-12): per-Item / per-Group field bindings
    // (named `item_<field>` / `group_<field>`) need their backing
    // buffers sized to one slot per Item-rooted (resp. Group-rooted)
    // entity, NOT the per-agent default. These counts are passed in
    // from the .sim's resolved Compilation; pre-fix the alloc loop
    // defaulted to `agent_count` for these bindings AND only emitted
    // the first per field name (entity-prefixed names collapsed in a
    // BTreeMap), silently dropping the rest.
    item_entity_count: u32,
    group_entity_count: u32,
    // Whether this fixture has at least one f32+Add view fold —
    // gates sort-scratch buffer allocation in the generated runtime.
    needs_sort: bool,
    // Event layout for the first f32+Add fold's source event. Carries
    // the record stride used to size `event_ring_sort_scratch_buf`.
    sort_event_layout: Option<&crate::cg::program::EventLayout>,
) -> String {
    let kernel_count = artifacts.kernel_index.len();
    let mut out = String::new();
    out.push_str(&format!(
        "// Plan E-A3.1 — placeholder generated runtime core for `{fixture_name}`.\n\
         //\n\
         // Generated by `dsl_compiler::build_helper::synthesize_runtime_core_a2`.\n\
         // {kernel_count} kernels in this fixture's schedule.\n\
         //\n\
         // Subsequent slices populate this file with `try_new` (A3.2 — alloc\n\
         // per-binding buffers from the manifest below) and `step()` (A4 —\n\
         // walk SCHEDULE + bind each kernel). For now the binding inventory\n\
         // is human-readable for verification.\n\
         //\n\
         // ## Binding manifest\n\
         //\n",
    ));
    for spec in &artifacts.kernel_specs {
        out.push_str(&format!(
            "// ### kernel {} ({} bindings, kind={:?})\n",
            spec.name,
            spec.bindings.len(),
            spec.kind,
        ));
        for b in &spec.bindings {
            out.push_str(&format!(
                "//   slot {:>2}  {:<48}  access={:<18}  wgsl_ty={:<24}  src={:?}\n",
                b.slot,
                b.name,
                format!("{:?}", b.access),
                b.wgsl_ty,
                b.bg_source,
            ));
        }
        out.push_str("//\n");
    }
    out.push_str(&format!(
        "\n#[allow(dead_code)]\n\
         pub const FIXTURE_NAME: &str = \"{fixture_name}\";\n\
         #[allow(dead_code)]\n\
         pub const KERNEL_COUNT: usize = {kernel_count};\n\n",
    ));

    // Plan E-A3.2 — emit GeneratedRuntime struct + try_new constructor.
    //
    // Walks every kernel's bindings, collects unique fixture-owned
    // buffers (External bindings that AREN'T standard agent columns
    // routed through AgentBuffers, AREN'T infra bindings like sim_cfg
    // / event_ring / cfg, AREN'T Transient bindings allocated by
    // engine helpers). Each gets a `<name>_buf: wgpu::Buffer` field
    // and an alloc line in `try_new` with size derived from the
    // wgsl_ty.
    //
    // Sizing today: `agent_count * elem_bytes`. Per-(observer, subject)
    // bindings need `agent_count * agent_count` cells; the up-sizing is
    // gated by [`slot_count_expr`] in two cases:
    //   1. The 6 BeliefState SoA columns surfaced from `LowerOpts.
    //      belief_state` (allow-list inside [`is_belief_state_pair_column`]).
    //   2. The materialized-view backing storage `view_storage_primary`
    //      when the fixture has a `view foo(a: Agent, b: <Entity>)` decl
    //      (gate via the `pair_keyed_second_key` parameter, derived
    //      from `comp.views` at the call site). Generalised in T6 fix
    //      from Agent×Agent to also handle Agent×Item / Agent×Group /
    //      Agent×Quest second keys.
    // A real binding-shape annotation in the AST is the proper long-term
    // fix; for now the per-binding-name heuristic + a fixture-level
    // second-key shape keep the generator working.
    out.push_str(&synthesize_generated_runtime_struct(
        fixture_name,
        artifacts,
        init_stmts,
        runtime_config_defaults,
        ability_file_names,
        events,
        pair_keyed_second_key,
        materialized_views,
        binds_voxel_grid,
        binds_navgrid,
        indirect_consumer_kernel_names,
        item_entity_count,
        group_entity_count,
        needs_sort,
        sort_event_layout,
    ));

    out
}

/// True if `binding_name` is an `agent_*` binding routed through
/// `engine::gpu::AgentBuffers` standard columns rather than allocated
/// by the per-fixture runtime.
fn is_standard_agent_column(binding_name: &str) -> bool {
    let suffix = match binding_name.strip_prefix("agent_") {
        Some(s) => s,
        None => return false,
    };
    // Mirrors `engine::gpu::bindings_context::AgentBuffers::STANDARD_COLUMNS`.
    matches!(
        suffix,
        "hp" | "max_hp" | "alive" | "pos" | "level"
            | "move_speed" | "move_speed_mult"
            | "shield_hp" | "armor" | "magic_resist"
            | "attack_damage" | "attack_range"
            | "mana" | "max_mana" | "ability_power"
    )
}

/// True if `binding_name` is shared infrastructure that the engine
/// supplies via `KernelBindingsContext::event_ring` /
/// `event_ring.sim_cfg()` / per-kernel cfg uniforms — never allocated
/// per-fixture.
fn is_infra_binding(binding_name: &str) -> bool {
    // Mirrors `cg::emit::program::classify_binding` — bindings whose
    // value comes from the shared KernelBindingsContext (event_ring,
    // event_tail, voxel_grid, the per-kernel cfg buf, sim_cfg via
    // EventRing accessor) rather than a fixture-owned buffer.
    //
    // NOT in the list: `snapshot_kick`. It looks like infrastructure
    // by name but the compiler classifies it as Extras → it's a real
    // fixture-owned buffer the runtime allocates + writes a one-shot
    // kick into. Adding it here was the bug A4.1's failed attempt
    // surfaced.
    if matches!(
        binding_name,
        "sim_cfg" | "event_ring" | "event_tail" | "cfg" | "voxel_grid"
        // Voxel-region-indices Phase 4b — `navgrid` + `navgrid_cfg`
        // route through the shared KernelBindingsContext (mirrors
        // `voxel_grid`). The runtime owns the buffers on its own
        // fields (gated by `binds_navgrid` above), so the per-binding
        // alloc loop must NOT re-declare them as fixture-owned.
        | "navgrid" | "navgrid_cfg"
    ) {
        return true;
    }
    // `ability_registry_*` columns come from the runtime's
    // `PackedAbilityRegistryGpu` field, not separate buffers. The
    // dispatch reads them as `&self.registry_gpu.<col>`.
    binding_name.starts_with("ability_registry_")
}

/// Bytes per element for a binding's `wgsl_ty`. Returns `None` for
/// types the per-agent sizing formula can't handle yet (e.g. structs
/// with nontrivial layout); the caller emits a TODO comment for
/// those.
fn elem_bytes_for_wgsl_ty(wgsl_ty: &str) -> Option<u64> {
    let inner = wgsl_ty
        .trim()
        .strip_prefix("array<")
        .and_then(|s| s.strip_suffix(">"))
        .unwrap_or(wgsl_ty.trim());
    let inner = inner
        .strip_prefix("atomic<")
        .and_then(|s| s.strip_suffix(">"))
        .unwrap_or(inner);
    match inner {
        "u32" | "f32" | "i32" => Some(4),
        "vec2<u32>" | "vec2<f32>" | "vec2<i32>" => Some(8),
        // vec3 std430-pads to vec4 — 16 bytes
        "vec3<u32>" | "vec3<f32>" | "vec3<i32>" => Some(16),
        "vec4<u32>" | "vec4<f32>" | "vec4<i32>" => Some(16),
        _ => None,
    }
}

/// True if `binding_name` is a per-(observer, subject) BeliefState SoA
/// column. The 6 columns the spec defines (`pair_map`-shaped, sized
/// `agent_count * agent_count` cells each):
///
/// * `beliefs_flags`       — bit-OR accumulator for `BeliefAcquired`
/// * `beliefs_pos`         — last-known pos (vec4-padded vec3)
/// * `beliefs_type`        — last-known creature_type (u8-q8)
/// * `beliefs_tick`        — last-seen tick (u32)
/// * `beliefs_confidence`  — q8 confidence (u8)
/// * `beliefs_suspicion`   — q8 suspicion (u8)
///
/// Anything else with a `beliefs_` prefix is fixture-specific and falls
/// through to per-agent sizing (no fixture today uses such a name; the
/// allow-list keeps the gate explicit).
///
/// Mirrors the column set the compiler surfaces from `LowerOpts.
/// belief_state` (see `cg::lower::driver::LowerOpts.belief_state` doc).
fn is_belief_state_pair_column(binding_name: &str) -> bool {
    matches!(
        binding_name,
        "beliefs_flags"
            | "beliefs_pos"
            | "beliefs_type"
            | "beliefs_tick"
            | "beliefs_confidence"
            | "beliefs_suspicion"
    )
}

/// Number of slots in the buffer for a given binding. Sizing rules:
///
/// * `agent_count * agent_count` for per-(observer, subject) BeliefState
///   SoA columns (the 6 columns enumerated in
///   [`is_belief_state_pair_column`]). These are agent-side consumer-
///   write columns the BGL composer surfaces when `LowerOpts.belief_state`
///   is set.
/// * `agent_count * <second_key_population>` for the materialized-view
///   backing storage (`view_storage_primary`) when
///   `pair_keyed_second_key` is `Some(_)`. The fold body of a
///   `view foo(observer: Agent, source: X) -> ...` writes into
///   `view_storage_primary[observer * cfg.second_key_pop + source]`,
///   so the buffer must hold `agent_count * <second_key_pop>` cells.
///   The second-key population is determined by the kind of the second
///   param:
///     * `Agent` → `agent_count` (per-tick variable; tom_probe shape)
///     * `Item` / `Group` / `Quest` → static count of declared
///       entities of that root in the .sim source (T6 fix —
///       trade_caravans declares 3 Items, so the inventory view
///       sizes as `agent_count * 3` instead of the previous
///       `agent_count * agent_count` over-allocation OR the pre-fix
///       under-allocation).
///   Pre-fix sizing fell back to `agent_count` cells, producing a
///   `<second_key_pop>×` under-allocation that silently corrupted
///   memory at runtime when the fold body wrote through indices past
///   the per-agent prefix.
/// * `item_entity_count` for `item_<field>` bindings (Gap T2 fix,
///   2026-05-12). The `items.<field>(N)` lowering produces a binding
///   keyed by FIELD NAME (shared across all Item-rooted entities
///   declaring the field); the buffer holds one slot per declared
///   Item entity, indexed by the entity's position in declaration
///   order among Item-rooted entities. trade_caravans declares
///   3 Items (Grain, Spice, Silk), so `item_base_price` is sized to
///   3 slots. Entities not declaring the field still occupy a slot
///   (zero-init by default) — the storage layout is positional, not
///   sparse, so the user index `N` directly addresses the right
///   entity. Pre-fix the alloc loop sized this binding to
///   `agent_count` and only emitted the FIRST per-Item-entity name
///   (a `BTreeMap` collapse), silently dropping the others; reads
///   for N=1, 2, ... aliased the first Item's buffer.
/// * `group_entity_count` for `group_<field>` bindings — same shape
///   as `item_<field>` but for Group-rooted entities.
/// * `agent_count` for everything else — the per-agent default.
///
/// Today's pair-keyed detection picks a SINGLE fixture-wide
/// second-key shape (largest population — see
/// [`detect_pair_keyed_second_key`] doc on the mixed-key first-match
/// fallback) because the literal binding name `view_storage_primary`
/// is shared across every ViewFold kernel in the fixture. When the
/// codebase grows a fixture with multiple distinct view-storage
/// buffers (e.g. one `view_storage_<view>_primary` per view), this
/// gate splits into per-binding lookup.
fn slot_count_expr(
    binding_name: &str,
    pair_keyed_second_key: Option<PairKeyedSecondKey>,
    item_entity_count: u32,
    group_entity_count: u32,
) -> String {
    if is_belief_state_pair_column(binding_name) {
        // Per-(observer, subject) cell — `pair_map` storage shape per
        // the BeliefState column contract. TODO: replace name-list
        // heuristic with proper binding-shape annotation in the AST.
        "(agent_count as u64) * (agent_count as u64)".to_string()
    } else if let (true, Some(skey)) = (
        binding_name == "view_storage_primary",
        pair_keyed_second_key,
    ) {
        // Legacy fixture-wide path — fires when the rename to
        // `view_storage_<view>_primary` didn't apply (defensive
        // fallback for kernels whose names don't follow the
        // `fold_*` / `decay_*` convention). Per-view sizing for the
        // renamed names is handled by the call site (see
        // `slot_count_expr_for_view_buf`).
        format!("(agent_count as u64) * {}", skey.population_u64_expr())
    } else if binding_name.starts_with("item_") {
        // Gap T2 fix (2026-05-12): per-Item field buffer. One slot
        // per declared Item-rooted entity. `.max(1)` so a fixture
        // with zero Items (a defensive case — every binding implies
        // a non-empty field catalog) still produces a non-zero
        // allocation that wgpu accepts.
        let n = item_entity_count.max(1) as u64;
        format!("{n}u64")
    } else if binding_name.starts_with("group_") {
        // Gap T2 fix (2026-05-12): per-Group field buffer. Same
        // shape as `item_<field>` but for Group-rooted entities.
        let n = group_entity_count.max(1) as u64;
        format!("{n}u64")
    } else if binding_name == "scoring_output" {
        // The scoring kernel writes 4 u32s per agent
        // (`best_action`, `best_target`, `best_utility`, _pad)
        // at `scoring_output[agent_id*4 + N]`. Pre-fix the buffer
        // was sized as `agent_count * 4 bytes` (1 u32/agent) and
        // every write past agent_id=3 silently OOB'd — scoring for
        // those slots fell back to last-tick values (or 0), which
        // surfaced as squad_skirmish "Rally fires despite mask_2=0"
        // and similar "scoring picks an action that shouldn't be
        // eligible" bugs across PerPair-scoring fixtures.
        "(agent_count as u64) * 4u64".to_string()
    } else {
        "agent_count as u64".to_string()
    }
}

/// Slot-count expression for a per-view storage buffer (one of
/// `view_storage_<view>_primary` / `view_storage_<view>_anchor` /
/// `view_storage_<view>_ids`). `pair_keyed` is `Some(_)` for views
/// with an `(Agent, X)` key pair; `None` for per-agent views.
///
/// Per-view sizing closes the aliasing gap where the legacy single
/// `view_storage_primary_buf` was sized for the LARGEST view and
/// every fold wrote into it. Each view now allocates exactly its
/// own footprint (`agent_count` for per-agent; `agent_count *
/// second_key_pop` for pair-keyed).
fn slot_count_expr_for_view_buf(pair_keyed: Option<PairKeyedSecondKey>) -> String {
    match pair_keyed {
        Some(skey) => format!("(agent_count as u64) * {}", skey.population_u64_expr()),
        None => "agent_count as u64".to_string(),
    }
}

/// Slot-count expression for a spatial-grid backing buffer.
///
/// Returns `Some(expr)` when `binding_name` is one of the three
/// spatial-grid buffers the auto-emitter writes into:
///
/// * `spatial_grid_starts` — exclusive-prefix-scan of per-cell
///   counts. Indexed by `cell` in `[0, num_cells)`, with a `_cell +
///   1u` lookahead read at every cell. Sized `num_cells + 1` u32
///   cells.
/// * `spatial_grid_offsets` — per-cell atomic counter (phase 1) +
///   per-cell write cursor (phase 3). Indexed by `cell` in
///   `[0, num_cells)`. Sized `num_cells` u32 cells.
/// * `spatial_grid_cells` — agent-id slots. Two access patterns:
///   the legacy `BuildHash` path indexes `cell * MAX_PER_CELL +
///   slot` (max `num_cells * MAX_PER_CELL`), the real counting-sort
///   path indexes `starts[cell] + local_slot` (max `agent_count`).
///   The larger of the two bounds is `num_cells * MAX_PER_CELL`, so
///   we size for that to cover both paths without per-fixture
///   discrimination.
///
/// Pre-fix sizing routed every spatial buffer through the
/// per-agent default (`agent_count * 4 bytes`). For boids' 22³ =
/// 10 648-cell grid that left `starts` and `offsets` reading out
/// of bounds at every cell past index `agent_count - 1` (silent
/// OOB → returns 0 in WGSL storage), collapsing `nearby_targets`
/// to the empty set. Gap 1 of `detective_investigation`; same
/// shape powers hill_raid's "siege didn't animate" mode.
///
/// Returns `None` for non-spatial bindings — the caller falls
/// through to [`slot_count_expr`].
fn slot_count_expr_for_spatial_grid_buffer(binding_name: &str) -> Option<String> {
    use crate::cg::emit::spatial::{grid_dim, MAX_PER_CELL};
    let num_cells = (grid_dim() as u64).pow(3);
    match binding_name {
        // Prefix-scan output. Per-cell read at index `_cell` plus
        // a lookahead at `_cell + 1u` for every in-range cell, so
        // the buffer needs one extra slot past `num_cells - 1`.
        "spatial_grid_starts" => Some(format!("{}u64", num_cells + 1)),
        // Per-cell atomic counter / write cursor. One slot per
        // cell; no lookahead.
        "spatial_grid_offsets" => Some(format!("{}u64", num_cells)),
        // Agent-id slots. Sized for the larger of the two access
        // patterns (legacy `cell * MAX_PER_CELL + slot`).
        "spatial_grid_cells" => {
            Some(format!("{}u64", num_cells * (MAX_PER_CELL as u64)))
        }
        _ => None,
    }
}

/// Extract the `<view>` name from a kernel by inspecting its
/// `BgSource::ViewHandle` binding's resident accessor. The
/// accessor format is `fold_view_<view_name>_handles` (stable
/// across all kernel kinds that touch view storage: ViewFold,
/// ViewDecay, BeliefSocialMerge). Strip the prefix + suffix and
/// what remains IS the view name — view names with internal
/// underscores (e.g. `room_known`, `damage_dealt`) round-trip
/// correctly.
///
/// Falls back to the legacy [`view_name_from_kernel_name`] string
/// parser for kernels with no ViewHandle binding (mostly defensive;
/// view-storage-touching kernels always have one).
fn view_name_from_kernel_spec(spec: &crate::kernel_binding_ir::KernelSpec) -> Option<&str> {
    use crate::kernel_binding_ir::BgSource;
    for b in &spec.bindings {
        if let BgSource::ViewHandle { accessor, .. } = &b.bg_source {
            // accessor = `fold_view_<view_name>_handles`
            if let Some(rest) = accessor.strip_prefix("fold_view_") {
                if let Some(name) = rest.strip_suffix("_handles") {
                    return Some(name);
                }
            }
        }
    }
    view_name_from_kernel_name(&spec.name)
}

/// Legacy string-parsing extractor — kept for kernels with no
/// ViewHandle binding (mostly defensive). For
/// view-storage-touching kernels the structured
/// [`view_name_from_kernel_spec`] is preferred because it
/// round-trips view names with internal underscores.
fn view_name_from_kernel_name(kernel_name: &str) -> Option<&str> {
    if let Some(rest) = kernel_name.strip_prefix("fold_") {
        // Defensive: `fold_view_<id>` is the un-named-view fallback
        // form `view_fold_fused_kernel_name` emits when the view
        // name lookup fails. Treat the trailing slug as the view
        // name (the per-view buffer name still keeps the kernel +
        // BGL in sync).
        return Some(rest);
    }
    if let Some(rest) = kernel_name.strip_prefix("decay_") {
        return Some(rest);
    }
    // Plan I slice I.4b — BeliefSocialMerge kernel name follows
    // `merge_<view>_<event_snake>_<op>`. The view-name extraction
    // here is fragile because event names may contain underscores
    // — the structured `view_name_from_kernel_spec` (preferred)
    // gets it right by reading the BgSource::ViewHandle accessor.
    // This branch stays as a defensive fallback for tests that
    // synthesise specs without a ViewHandle binding.
    if let Some(rest) = kernel_name.strip_prefix("merge_") {
        let first_us = rest.find('_')?;
        return Some(&rest[..first_us]);
    }
    None
}

/// Per-view rename of a fold/decay kernel's view-storage bindings.
/// `binding_name` is the BGL-level name as the kernel synthesiser
/// emitted it (`view_storage_primary` / `view_storage_anchor` /
/// `view_storage_ids` for fold kernels; `view_storage_primary` only
/// for decay kernels). `view_name` is `Some(<view>)` when the
/// enclosing kernel is `fold_<view>` / `decay_<view>` (per
/// [`view_name_from_kernel_name`]).
///
/// Returns the per-view-namespaced name when both inputs match the
/// rename rule (e.g. `view_storage_damage_dealt_primary`); returns
/// the original `binding_name` (unchanged) otherwise. Bindings with
/// names that don't start with `view_storage_` (or that already
/// carry a per-view name from the scoring-kernel emit path) pass
/// through unchanged so the rename is idempotent.
fn view_storage_per_view_name(binding_name: &str, view_name: Option<&str>) -> String {
    let Some(view) = view_name else {
        return binding_name.to_string();
    };
    match binding_name {
        "view_storage_primary" => format!("view_storage_{view}_primary"),
        "view_storage_anchor" => format!("view_storage_{view}_anchor"),
        "view_storage_ids" => format!("view_storage_{view}_ids"),
        _ => binding_name.to_string(),
    }
}

/// `pair_keyed_second_key` — see [`synthesize_runtime_core_a2`]'s
/// same-named param. Read by [`slot_count_expr`] to up-size
/// `view_storage_primary` to `agent_count * <second_key_population>`
/// cells when the fixture has a pair-keyed materialized view.
///
/// `materialized_views` — one entry per `@materialized` view in the
/// fixture. Used to (a) allocate one
/// `view_storage_<view_name>_primary_buf` per view (sized for THAT
/// view's own pair-keyed second-key, not the fixture-wide max), and
/// (b) route each fold/decay kernel's `view_storage_primary` /
/// `view_storage_anchor` / `view_storage_ids` BGL bindings to the
/// per-view buffer instead of a single shared one. Closes the
/// 6-fixture aliasing gap (forest_fire/squad_skirmish/plague_city/
/// detective_investigation/palace_coup/among_us) where the legacy
/// shared `view_storage_primary_buf` aggregated incoherently
/// across views.
fn synthesize_generated_runtime_struct(
    fixture_name: &str,
    artifacts: &crate::cg::emit::EmittedArtifacts,
    init_stmts: &[dsl_ast::ast::InitStmt],
    runtime_config_defaults: &std::collections::BTreeMap<String, RuntimeConfigDefault>,
    ability_file_names: &[String],
    events: &[dsl_ast::ir::EventIR],
    pair_keyed_second_key: Option<PairKeyedSecondKey>,
    materialized_views: &[MaterializedViewInfo],
    binds_voxel_grid: bool,
    binds_navgrid: bool,
    indirect_consumer_kernel_names: &[String],
    item_entity_count: u32,
    group_entity_count: u32,
    needs_sort: bool,
    sort_event_layout: Option<&crate::cg::program::EventLayout>,
) -> String {
    use crate::kernel_binding_ir::BgSource;
    use std::collections::BTreeMap;

    // Per-view sizing lookup: view_name → pair-keyed second-key shape.
    // Used by the binding-rename loop below to size each per-view
    // `view_storage_<view_name>_primary_buf` for THAT view's own
    // shape, independently of fixture-wide pair-keyed detection.
    let view_pair_keyed: BTreeMap<&str, Option<PairKeyedSecondKey>> = materialized_views
        .iter()
        .map(|v| (v.name.as_str(), v.pair_keyed))
        .collect();
    // Plan G G3f follow-up — per-view ring storage shape. K=ring length;
    // stride=u32 words per cell. Multiplied into the per-view buffer
    // size below so `@per_entity_ring(K=N)` views with struct-cell
    // payloads (e.g. threats with the 8-field ThreatZoneCell layout)
    // get `agent_count * K * stride * 4` bytes instead of the scalar
    // `agent_count * 4`.
    let view_ring_shape: BTreeMap<&str, (Option<u16>, u32)> = materialized_views
        .iter()
        .map(|v| (v.name.as_str(), (v.ring_k, v.cell_stride_u32)))
        .collect();

    // Fold consumer detection. Used to decide
    // whether to allocate the `prev_event_tail_buf` snapshot side
    // buffer + emit the snapshot stage at the top of step(). Hoisted
    // here so the struct + try_new emit can gate the side-buffer field
    // on the same signal as the step() body emit below.
    //
    // Detection criteria (must satisfy ALL):
    //
    //   1. `kernel.kind in {ViewFold, ViewDecay, PerEventEmit}` —
    //      Generic kernels (per-agent producers like `physics_Spread`)
    //      bind event_ring too because they `atomicAdd(&event_tail,
    //      1u)` to emit, but their cfg slot 0 is `agent_cap`, NOT
    //      event_count. Overwriting agent_cap with the snapshot
    //      value would short-circuit per-agent dispatch.
    //
    //   2. NOT in `indirect_consumer_kernel_names` — Indirect
    //      chronicle handlers take a per-dispatch live-tail copy
    //      (existing path), not a top-of-step snapshot.
    //
    //   3. The kernel's WGSL actually walks the event_ring as
    //      records (1-D dispatch: one thread per `event_idx`). Some
    //      ViewFold kernels are `@dispatch(per_agent_event_scan)`
    //      — they take a 2-D dispatch over (observer, source) pairs
    //      and use `cfg.event_count` as the agent-count bound, NOT
    //      as an event-ring tail. The signal: their generated
    //      kernel rust file calls `dispatch_workgroups(_, _, 1)`
    //      with two non-1 dims (a 2-D dispatch). Event-scan folds
    //      call `dispatch_workgroups(_, 1, 1)` (1-D).
    //
    //      Heuristic: scan the kernel's wgsl_files entry for
    //      `workgroup_size(8, 8)` — that's the canonical 2-D
    //      PerAgentEventScan workgroup. 1-D folds emit
    //      `workgroup_size(64)`.
    use crate::kernel_binding_ir::KernelKind;
    let fold_consumer_kernel_names_owned: Vec<String> = artifacts
        .kernel_specs
        .iter()
        .filter(|spec| {
            let binds_event_ring = spec.bindings.iter().any(|b| {
                b.name == "event_ring" || b.name == "event_tail"
            });
            let is_indirect = indirect_consumer_kernel_names
                .iter()
                .any(|n| n == &spec.name);
            let event_count_in_slot_0 = matches!(
                spec.kind,
                KernelKind::ViewFold | KernelKind::ViewDecay | KernelKind::PerEventEmit,
            );
            // Per-agent-event-scan filter: skip 2-D pair-scan folds
            // (they treat slot 0 as agent_count, not event_count).
            let is_pair_scan = artifacts
                .wgsl_files
                .get(&format!("{}.wgsl", spec.name))
                .map(|src| src.contains("workgroup_size(8, 8)"))
                .unwrap_or(false);
            binds_event_ring && !is_indirect && event_count_in_slot_0 && !is_pair_scan
        })
        .map(|spec| spec.name.clone())
        .collect();
    let has_fold_consumers_outer = !fold_consumer_kernel_names_owned.is_empty();

    // Collect unique fixture-owned bindings across all kernels.
    // BTreeMap (name → wgsl_ty) — same binding may appear in multiple
    // kernels with different wgsl_ty (e.g. one kernel sees `array<f32>`,
    // another sees `array<atomic<u32>>` after the f32 RMW upgrade). Pick
    // the first ty seen; sizing is bytes-based so the alloc is correct
    // for either ty as long as elem-size matches (it does for f32 vs u32).
    let mut owned: BTreeMap<String, String> = BTreeMap::new(); // name → wgsl_ty
    // Per-kernel cfg buffers — one wgpu::Buffer per kernel that has
    // a Cfg-source binding (which is every kernel today). Allocated
    // sized to the cfg struct's std430 footprint (16 bytes covers
    // the standard 4-u32 cfg layouts; oversize is fine).
    let mut cfg_buffer_names: Vec<String> = Vec::new();
    // Per-view-buffer name → its pair-keyed shape. Used by
    // `slot_count_expr_for_view_buf` to size each per-view storage
    // buffer independently of the (legacy) fixture-wide
    // `pair_keyed_second_key`. Populated by the binding-rename loop
    // below as renamed `view_storage_<view>_primary` names land in
    // `owned`.
    let mut owned_view_buf_pair_keyed: BTreeMap<String, Option<PairKeyedSecondKey>> =
        BTreeMap::new();
    // Plan G G3f follow-up — per-view ring shape captured during the
    // binding-rename pass and consumed by the buffer-allocation pass
    // below to size struct-cell ring views correctly. Keyed on the
    // RENAMED `view_storage_<view>_primary` binding name so the
    // buffer-sizing arm can look up shape by the same key it uses
    // for `owned_view_buf_pair_keyed`.
    let mut owned_view_buf_ring_shape: BTreeMap<String, (Option<u16>, u32)> =
        BTreeMap::new();
    for spec in &artifacts.kernel_specs {
        let mut has_cfg = false;
        // Per-fold/decay kernel view-name extraction. The BGL slot
        // names `view_storage_primary|anchor|ids` are uniform across
        // every fold/decay kernel — the kernel name encodes WHICH
        // view's storage they refer to (`fold_<view>` / `decay_<view>`).
        // Resolve the view name once here and rewrite the binding
        // names below so each fold/decay kernel allocates + binds to
        // its own per-view buffer instead of the legacy shared
        // `view_storage_primary_buf`.
        let view_name_for_kernel = view_name_from_kernel_spec(spec);
        for b in &spec.bindings {
            if matches!(b.bg_source, BgSource::Cfg) {
                has_cfg = true;
                continue;
            }
            // Skip AliasOf — handled at BindGroup time, not a real
            // separate binding.
            if matches!(b.bg_source, BgSource::AliasOf(_)) {
                continue;
            }
            // Use NAME-based classification (mirroring the compiler's
            // classify_binding) instead of bg_source. Name-based is
            // authoritative for what goes in Extras vs ctx; bg_source
            // can't tell them apart for some cases (scoring_output,
            // view_storage_* are Resident-source but Extras-bound).
            if is_infra_binding(&b.name) {
                continue;
            }
            // Per-view storage rename: in fold/decay kernels the BGL
            // slot names `view_storage_primary|anchor|ids` are
            // generic — rewrite them to per-view names so each view
            // gets its own host-side buffer. Without this every
            // fold/decay kernel allocates + binds the SAME
            // `view_storage_primary_buf` and cross-view writes
            // collide (forest_fire / squad_skirmish / plague_city /
            // detective_investigation / palace_coup / among_us all
            // surface this gap).
            let effective_name = view_storage_per_view_name(&b.name, view_name_for_kernel);
            // Track per-view-buffer pair-keyed shape (look up by view
            // name from the materialized-view list).
            if effective_name.starts_with("view_storage_") && effective_name.ends_with("_primary") {
                let view_name = effective_name
                    .strip_prefix("view_storage_")
                    .and_then(|s| s.strip_suffix("_primary"));
                if let Some(vname) = view_name {
                    let pk = view_pair_keyed.get(vname).copied().flatten();
                    owned_view_buf_pair_keyed
                        .entry(effective_name.clone())
                        .or_insert(pk);
                    if let Some(shape) = view_ring_shape.get(vname).copied() {
                        owned_view_buf_ring_shape
                            .entry(effective_name.clone())
                            .or_insert(shape);
                    }
                }
            }
            // Standard agent columns ARE allocated as fixture-owned
            // today (no shared SimState yet). The AgentBuffers
            // population in step() routes them into ctx.state for the
            // from_context_with_extras call.
            // BTreeMap insert is "first-wins" via or_insert.
            owned.entry(effective_name).or_insert_with(|| b.wgsl_ty.clone());
        }
        if has_cfg {
            cfg_buffer_names.push(spec.name.clone());
        }
    }

    let mut out = String::new();
    // Radix sort pipeline cache struct — emitted only for fixtures that opt
    // into the sort pass. Defined before GeneratedRuntime so it can appear
    // as a field type.
    if needs_sort {
        out.push_str(
            "struct SortPipelines {\n\
             \x20   // Stage A: 4 passes × (histogram, scan, scatter) = 12 pipelines\n",
        );
        for pass in 0..4u32 {
            for phase in &["histogram", "scan", "scatter"] {
                out.push_str(&format!(
                    "    stage_a_pass{pass}_{phase}: (wgpu::ComputePipeline, wgpu::BindGroupLayout),\n",
                ));
            }
        }
        out.push_str(
            "    // Stage B: count, scan, scatter = 3 pipelines\n\
             \x20   stage_b_count: (wgpu::ComputePipeline, wgpu::BindGroupLayout),\n\
             \x20   stage_b_scan:  (wgpu::ComputePipeline, wgpu::BindGroupLayout),\n\
             \x20   stage_b_scatter: (wgpu::ComputePipeline, wgpu::BindGroupLayout),\n\
             }\n\n",
        );
    }

    out.push_str(
        "// Plan E-A3.2 — fixture-owned buffer struct + try_new constructor.\n\
         //\n\
         // The struct below collects every External binding that's NOT a\n\
         // standard agent SoA column and NOT shared infrastructure. Today\n\
         // no fixture's lib.rs imports this — A5 (firebolt_probe pilot)\n\
         // will be the first runtime to switch to it.\n\
         //\n\
         // **Inclusion contract.** This file uses module-relative paths\n\
         // (`dispatch::KernelCache`, `schedule::SCHEDULE`) so it MUST be\n\
         // `include!`d at the crate root of a runtime crate where the\n\
         // sibling generated.rs is also included (it provides those\n\
         // modules). Including from a sub-module breaks path resolution.\n\
         #[allow(dead_code, non_snake_case, clippy::all)]\n\
         pub struct GeneratedRuntime {\n\
         \x20   pub gpu: engine::GpuContext,\n\
         \x20   pub agent_count: u32,\n\
         \x20   pub seed: u64,\n\
         \x20   pub tick: u64,\n\
         \x20   pub event_ring: engine::gpu::EventRing,\n\
         \x20   pub registry_gpu: engine::ability::registry_gpu::PackedAbilityRegistryGpu,\n\
         \x20   pub cache: dispatch::KernelCache,\n",
    );
    for (name, _ty) in &owned {
        out.push_str(&format!("    pub {name}_buf: wgpu::Buffer,\n"));
    }
    // Per-kernel cfg buffers (Plan E-A4). One per kernel with a
    // Cfg-source binding. Named `cfg_<kernel>_buf` to avoid
    // collisions with fixture-owned buffers.
    for kernel_name in &cfg_buffer_names {
        out.push_str(&format!("    pub cfg_{kernel_name}_buf: wgpu::Buffer,\n"));
    }
    // 4-byte side buffer holding the snapshotted
    // GPU `event_tail` value from the END of the previous tick. Only
    // allocated when the fixture has at least one fold consumer (a
    // kernel that binds `event_ring`/`event_tail` and is NOT an
    // indirect chronicle handler). step() captures event_tail into
    // this buffer via its own submit at the top of the tick (before
    // clear_tail_in or the indirect-consumer queue.write_buffer
    // overwrites the GPU tail), then in the main encoder copies this
    // value into each fold's `cfg.event_count` slot.
    if has_fold_consumers_outer {
        out.push_str("    pub prev_event_tail_buf: wgpu::Buffer,\n");
    }
    // Voxel terrain + GPU mirror — only allocated when the fixture has at
    // least one kernel that binds `voxel_grid`. Without this gate the
    // mirror's 4 * extent³ byte buffer (64 MB at the default 256³ extent)
    // would be paid by every fixture; with it, voxel-free fixtures stay
    // zero-overhead and `KernelBindingsContext::voxel_grid` keeps its
    // `None` shape on the runtimes that don't need it.
    if binds_voxel_grid {
        out.push_str("    pub voxel_terrain: engine_voxel::VoxelTerrain,\n");
        out.push_str("    pub voxel_mirror: engine_voxel::VoxelMirror,\n");
    }
    // Voxel-region-indices Phase 4b — navgrid storage + extent uniform.
    // Only allocated when a kernel binds the `navgrid` namespace (the
    // `binds_navgrid` flag). Host fills both via `upload_navgrid(&idx)`
    // emitted below; the WGSL `voxel_navgrid_walkable(cx, cz)` helper
    // reads `navgrid_cfg.x` (size_x) for the bounds check + cell index.
    // Cap is `NAVGRID_MAX_CELLS=16384` cells × 4 bytes = 64 KB.
    if binds_navgrid {
        out.push_str("    pub navgrid_buf: wgpu::Buffer,\n");
        out.push_str("    pub navgrid_cfg_buf: wgpu::Buffer,\n");
    }
    // Radix sort scratch buffers — allocated only when the fixture has
    // at least one f32+Add view fold. The sort runs in the schedule
    // between producer phase and consumer phase so fold consumers read
    // target-grouped, seq-ordered events.
    if needs_sort {
        out.push_str("    pub event_ring_sort_scratch_buf: wgpu::Buffer,\n");
        out.push_str("    pub radix_histogram_buf: wgpu::Buffer,\n");
        out.push_str("    pub radix_bucket_offsets_buf: wgpu::Buffer,\n");
        out.push_str("    pub target_histogram_buf: wgpu::Buffer,\n");
        out.push_str("    pub target_offsets_buf: wgpu::Buffer,\n");
        out.push_str("    pub sort_cfg_buf: wgpu::Buffer,\n");
        out.push_str("    sort_pipelines: Option<SortPipelines>,\n");
    }
    // Plan E-A6 — `@runtime` config field values, mirrored host-side.
    // The .sim's `flee_strength: f32 = 1.0 @runtime` lands here as
    // `pub flee_strength: f32`. Host setters write the value to every
    // kernel cfg buffer that references the field, at the per-kernel
    // offset derived from KernelSpec.runtime_cfg_fields.
    for (name, def) in runtime_config_defaults {
        out.push_str(&format!("    pub {name}: {ty},\n", ty = def.scalar_ty));
    }
    out.push_str("}\n\n");

    out.push_str(&format!(
        "#[allow(dead_code, non_snake_case, clippy::all)]\n\
         impl GeneratedRuntime {{\n\
         \x20   pub fn try_new(seed: u64, agent_count: u32) -> Option<Self> {{\n\
         \x20       let gpu = engine::GpuContext::new_blocking().ok()?;\n\
         \x20       let event_ring = engine::gpu::EventRing::new(&gpu, \"{fixture_name}\");\n",
    ));
    // Voxel terrain construction. The CPU `VoxelTerrain` starts as an
    // empty cube at the engine_voxel default extent (256³); the
    // `VoxelMirror` allocates the matching GPU storage buffer and
    // performs the initial whole-buffer upload. After this point every
    // host-side `voxel_terrain.set_cell` call must be paired with a
    // `voxel_mirror.mark_dirty(...)` so the per-tick `flush_dirty` in
    // `step()` propagates the change to GPU before the next dispatch.
    if binds_voxel_grid {
        out.push_str(
            "        let voxel_terrain = engine_voxel::VoxelTerrain::new();\n\
             \x20       let voxel_mirror = engine_voxel::VoxelMirror::new(&gpu, voxel_terrain.grid());\n",
        );
    }
    // Voxel-region-indices Phase 4b — alloc the navgrid + cfg buffers
    // at the spec's cap (NAVGRID_MAX_CELLS=16384). The host populates
    // the cells via `upload_navgrid(&NavgridIndex)` after try_new
    // returns. Both buffers stay zero-initialized — out-of-bounds reads
    // through the WGSL helper return false, so an unpopulated navgrid
    // shows everywhere as non-walkable (safe default).
    if binds_navgrid {
        out.push_str(
            "        let navgrid_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {\n\
             \x20           label: Some(\"navgrid\"),\n\
             \x20           size: (engine_voxel::NAVGRID_MAX_CELLS as u64) * (engine_voxel::NAVGRID_BYTES_PER_CELL as u64),\n\
             \x20           usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       });\n\
             \x20       let navgrid_cfg_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {\n\
             \x20           label: Some(\"navgrid_cfg\"),\n\
             \x20           size: 16, // vec4<u32> = [size_x, size_z, origin_x, origin_z]\n\
             \x20           usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       });\n",
        );
    }
    // Radix sort scratch buffers. Sizes:
    //   - sort scratch: same capacity as event_ring
    //     (EVENT_RING_CAP_SLOTS * stride * 4 bytes)
    //   - radix_histogram / radix_bucket_offsets: 256 × 4 = 1024 bytes
    //   - target_histogram / target_offsets: (agent_count + 1) × 4 bytes
    //   - sort_cfg: 16 bytes (4 × u32 = { target_word_offset, agent_cap, _pad0, _pad1 })
    if needs_sort {
        let stride = sort_event_layout
            .map(|l| l.record_stride_u32)
            .unwrap_or(10); // default engine stride
        let target_word = sort_event_layout
            .map(sort_target_word_offset)
            .unwrap_or(3);
        // engine::gpu::EVENT_RING_CAP_SLOTS = 1_048_576
        let sort_scratch_bytes = 1_048_576_u64 * (stride as u64) * 4;
        out.push_str(&format!(
            "        let event_ring_sort_scratch_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
             \x20           label: Some(\"{fixture_name}::event_ring_sort_scratch\"),\n\
             \x20           size: {sort_scratch_bytes}u64,\n\
             \x20           usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       }});\n\
             \x20       let radix_histogram_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
             \x20           label: Some(\"{fixture_name}::radix_histogram\"),\n\
             \x20           size: 256u64 * 4u64,\n\
             \x20           usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       }});\n\
             \x20       let radix_bucket_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
             \x20           label: Some(\"{fixture_name}::radix_bucket_offsets\"),\n\
             \x20           size: 256u64 * 4u64,\n\
             \x20           usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       }});\n\
             \x20       let target_histogram_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
             \x20           label: Some(\"{fixture_name}::target_histogram\"),\n\
             \x20           size: (agent_count as u64 + 1u64) * 4u64,\n\
             \x20           usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       }});\n\
             \x20       let target_offsets_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
             \x20           label: Some(\"{fixture_name}::target_offsets\"),\n\
             \x20           size: (agent_count as u64 + 1u64) * 4u64,\n\
             \x20           usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       }});\n\
             \x20       let sort_cfg_init: [u32; 4] = [{target_word}u32, agent_count, 0u32, 0u32];\n\
             \x20       let sort_cfg_buf = wgpu::util::DeviceExt::create_buffer_init(\n\
             \x20           &gpu.device,\n\
             \x20           &wgpu::util::BufferInitDescriptor {{\n\
             \x20               label: Some(\"{fixture_name}::sort_cfg\"),\n\
             \x20               contents: bytemuck::cast_slice(&sort_cfg_init),\n\
             \x20               usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,\n\
             \x20           }},\n\
             \x20       );\n",
        ));
    }
    // Registry construction. Two paths:
    // (a) Fixture has a companion `assets/ability_test/<fixture>/`
    //     directory with `.ability` files — emit real construction.
    //     Each file is `include_str!`'d with a path resolved against
    //     CARGO_MANIFEST_DIR so the contents are baked into the binary.
    //     Iteration order matches the build-script order (alphabetical),
    //     so slot ids stay stable across build + runtime registries.
    // (b) No corpus — emit the historical placeholder (one no-op
    //     program; wgpu rejects zero-sized bindings).
    if ability_file_names.is_empty() {
        out.push_str(&format!(
            "        // No companion `assets/ability_test/{fixture_name}/` directory\n\
             \x20       // exists, so the registry stays a single-no-op-program\n\
             \x20       // placeholder (wgpu rejects zero-sized bindings, so an\n\
             \x20       // empty registry would fail to upload).\n\
             \x20       let mut _registry_builder = engine::ability::AbilityRegistryBuilder::new();\n\
             \x20       let _ = _registry_builder.register(\n\
             \x20           engine::ability::AbilityProgram::new_single_target(\n\
             \x20               1.0,\n\
             \x20               engine::ability::Gate {{ cooldown_ticks: 0, hostile_only: false, line_of_sight: false }},\n\
             \x20               [],\n\
             \x20           ),\n\
             \x20       );\n\
             \x20       let registry = _registry_builder.build();\n\
             \x20       let packed = engine::ability::PackedAbilityRegistry::pack(&registry);\n\
             \x20       let registry_gpu = engine::ability::registry_gpu::PackedAbilityRegistryGpu::upload(\n\
             \x20           &packed, &gpu, \"{fixture_name}\",\n\
             \x20       );\n",
        ));
    } else {
        // Build the (file_name, include_str!(...)) tuple list. Path
        // joins workspace_root + assets/ability_test/<fixture>/<name>;
        // CARGO_MANIFEST_DIR sits at <workspace>/crates/<crate>, so
        // `../../assets/...` resolves to the workspace assets dir
        // regardless of which crate (per-fixture *_runtime or sims)
        // calls into this generator.
        out.push_str(
            "        // Real ability registry — corpus discovered at build time.\n\
             \x20       // .ability sources are baked in via include_str! and parsed\n\
             \x20       // on first try_new(). Build-script and runtime see identical\n\
             \x20       // file lists in identical (alphabetical) order, so AbilityId\n\
             \x20       // slots line up between schedule synthesis and dispatch.\n\
             \x20       let _ability_sources: &[(&str, &str)] = &[\n",
        );
        for name in ability_file_names {
            out.push_str(&format!(
                "            (\n\
                 \x20               {name_lit},\n\
                 \x20               include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/../../assets/ability_test/{fixture_name}/{name}\")),\n\
                 \x20           ),\n",
                name_lit = format!("{name:?}"),
            ));
        }
        out.push_str(&format!(
            "        ];\n\
             \x20       let _parsed: Vec<(String, dsl_ast::AbilityFile)> = _ability_sources\n\
             \x20           .iter()\n\
             \x20           .map(|(name, src)| {{\n\
             \x20               let parsed = dsl_ast::parse_ability_file(src)\n\
             \x20                   .unwrap_or_else(|e| panic!(\"parse {{}}: {{:?}}\", name, e));\n\
             \x20               ((*name).to_string(), parsed)\n\
             \x20           }})\n\
             \x20           .collect();\n\
             \x20       let _built = dsl_compiler::ability_registry::build_registry(&_parsed)\n\
             \x20           .unwrap_or_else(|e| panic!(\"build_registry({fixture_name}): {{:?}}\", e));\n\
             \x20       let registry = _built.registry;\n\
             \x20       let packed = engine::ability::PackedAbilityRegistry::pack(&registry);\n\
             \x20       let registry_gpu = engine::ability::registry_gpu::PackedAbilityRegistryGpu::upload(\n\
             \x20           &packed, &gpu, \"{fixture_name}\",\n\
             \x20       );\n",
        ));
    }
    out.push_str(
        "        let cache = dispatch::KernelCache::default();\n",
    );
    for (name, ty) in &owned {
        let elem_bytes = match elem_bytes_for_wgsl_ty(ty) {
            Some(b) => b,
            None => {
                // Unknown type — emit a panic so the build catches it
                // and a TODO is visible in the generated source.
                out.push_str(&format!(
                    "        // TODO(plan-e/a3.2): can't size binding {name:?} of wgsl_ty {ty:?} automatically.\n\
                     \x20       panic!(\"GeneratedRuntime sizing unimplemented for {name} : {ty}\");\n",
                ));
                continue;
            }
        };
        // Per-view storage buffer sizing: when the binding name is
        // a renamed `view_storage_<view>_primary|anchor|ids`, look
        // up THAT view's pair-keyed second-key shape and size for
        // it independently of the fixture-wide
        // `pair_keyed_second_key` (which was the legacy single-
        // bucket sizing path). Falls through to the per-agent /
        // legacy-shared sizing for every other binding.
        //
        // Spatial-grid buffer sizing (Gap detective#1, 2026-05-12):
        // the three `spatial_grid_*` bindings index by cell (range
        // `[0, GRID_DIM^3)`) not by agent, so the per-agent default
        // under-allocated by ~600× on boids' 22³ grid at agent_count
        // ≈ 18. Now sized via the spatial-grid constants — see
        // `slot_count_expr_for_spatial_grid_buffer`.
        let slot_expr = if let Some(expr) = slot_count_expr_for_spatial_grid_buffer(name) {
            expr
        } else if let Some(pk) = owned_view_buf_pair_keyed.get(name) {
            // Plan G G3f follow-up — for `@per_entity_ring(K)` views
            // with struct-cell payloads (stride > 1), multiply the
            // per-agent slot count by `K * stride` so the buffer
            // covers `agent_count * K * stride` u32 words. Scalar
            // ring views (stride == 1) and non-ring views fall
            // through to the existing pair-keyed sizing.
            let base = slot_count_expr_for_view_buf(*pk);
            match owned_view_buf_ring_shape.get(name) {
                Some((Some(k), stride)) if (*k as u32) * *stride > 1 => {
                    let factor = (*k as u32) * *stride;
                    format!("({base}) * {factor}u64")
                }
                _ => base,
            }
        } else if let Some(view) = name
            .strip_prefix("view_storage_")
            .and_then(|s| s.strip_suffix("_anchor").or_else(|| s.strip_suffix("_ids")))
        {
            // Anchor / ids slots inherit the SAME sizing as the view's
            // primary buffer. The fold body atomicAdds into the
            // anchor / writes ring-cursor info into ids per agent
            // slot — they share the per-(observer, source) layout of
            // the view's primary slab.
            let pk = view_pair_keyed.get(view).copied().flatten();
            slot_count_expr_for_view_buf(pk)
        } else {
            slot_count_expr(name, pair_keyed_second_key, item_entity_count, group_entity_count)
        };
        // Plan E-A6 — if `init { ... }` declared a per-slot fill for
        // this `agent_<col>` binding, switch from zero-init create_buffer
        // to create_buffer_init with the computed slice. This is how
        // fixture-owned init state lives in the .sim source instead of
        // a hand-written *_runtime/lib.rs.
        let init_match = name.strip_prefix("agent_").and_then(|col| {
            init_stmts.iter().find(|s| s.field == col).map(|s| (col, s))
        });
        if let Some((col, stmt)) = init_match {
            if elem_bytes != 4 {
                out.push_str(&format!(
                    "        // TODO(plan-e/a6): init for {name} ignored — only u32/f32 (4-byte) elem types supported today (saw {elem_bytes}-byte).\n\
                     \x20       let {name}_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
                     \x20           label: Some(\"{fixture_name}::{name}\"),\n\
                     \x20           size: ({slot_expr} * {elem_bytes}u64).max(16),\n\
                     \x20           usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,\n\
                     \x20           mapped_at_creation: false,\n\
                     \x20       }});\n",
                ));
                continue;
            }
            // Gap G fix (2026-05-11) — route the init constant by the
            // target column's primitive type. Pre-fix, every init lowered
            // to `vec![Nu32; agent_count]` regardless of whether the
            // column is f32, u32, or bool. For f32 columns (hp, max_hp,
            // mana, ...) that wrote the u32 bit-pattern of N (e.g. 100
            // == 0x64) into the f32 buffer, which reads back as 1.4e-43
            // — functionally zero. Look up the column type via
            // `AgentFieldId::from_snake` so f32 columns get an f32 init
            // slice and u32/bool columns keep the historical u32 path.
            //
            // Bool columns are represented as packed `u32` (0/1) on GPU,
            // so they share the u32 emit. I16/q8 columns currently fall
            // through `elem_bytes != 4` above (they're 2-byte) so they
            // hit the no-init path; if a future column is i16-packed at
            // 4 bytes, this match will need an explicit arm.
            //
            // Vec3 / EnumU8 / OptAgentId / OptEnumU32 are not surfaceable
            // through `init { col: <int> }` today (no fixture declares
            // them), but if one tries we emit u32 to preserve the old
            // behaviour rather than silently producing the wrong slice.
            use crate::cg::data_handle::{AgentFieldId, AgentFieldTy};
            let col_ty = AgentFieldId::from_snake(col).map(AgentFieldId::ty);
            let elem_rust_ty = match col_ty {
                Some(AgentFieldTy::F32) => "f32",
                _ => "u32",
            };
            let init_vec_expr = match (&stmt.expr, elem_rust_ty) {
                (dsl_ast::ast::InitExpr::Const(n), "f32") => {
                    // `100` → `100.0_f32` — InitExpr is i64; cast through
                    // f64 first preserves sign for the (rare) negative
                    // init values without surprising the compiler about
                    // the suffix on a literal-expr.
                    format!("vec![{n}.0_f32; agent_count as usize]")
                }
                (dsl_ast::ast::InitExpr::Const(n), _) => {
                    format!("vec![{n}u32; agent_count as usize]")
                }
                (dsl_ast::ast::InitExpr::Slot, "f32") => {
                    "(0..agent_count).map(|i| i as f32).collect::<Vec<f32>>()".to_string()
                }
                (dsl_ast::ast::InitExpr::Slot, _) => {
                    "(0..agent_count).collect::<Vec<u32>>()".to_string()
                }
            };
            out.push_str(&format!(
                "        let {name}_init: Vec<{elem_rust_ty}> = {init_vec_expr};\n\
                 \x20       let {name}_buf = wgpu::util::DeviceExt::create_buffer_init(\n\
                 \x20           &gpu.device,\n\
                 \x20           &wgpu::util::BufferInitDescriptor {{\n\
                 \x20               label: Some(\"{fixture_name}::{name}\"),\n\
                 \x20               contents: bytemuck::cast_slice(&{name}_init),\n\
                 \x20               usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,\n\
                 \x20           }},\n\
                 \x20       );\n",
            ));
            continue;
        }
        out.push_str(&format!(
            "        let {name}_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
             \x20           label: Some(\"{fixture_name}::{name}\"),\n\
             \x20           size: ({slot_expr} * {elem_bytes}u64).max(16),\n\
             \x20           usage: wgpu::BufferUsages::STORAGE\n\
             \x20               | wgpu::BufferUsages::COPY_SRC\n\
             \x20               | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       }});\n",
        ));
    }
    // Allocate per-kernel cfg buffer (uniform, sized 64 bytes — covers
    // the standard 4-u32 cfg layout with comfortable headroom for the
    // few cfg shapes that grow). Per-tick writes happen inside step().
    for kernel_name in &cfg_buffer_names {
        out.push_str(&format!(
            "        let cfg_{kernel_name}_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
             \x20           label: Some(\"{fixture_name}::cfg_{kernel_name}\"),\n\
             \x20           size: 64u64,\n\
             \x20           usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       }});\n",
        ));
    }
    // Allocate the prev_event_tail snapshot side
    // buffer when the fixture has fold consumers. 4 bytes (one u32),
    // STORAGE so it can be a copy_buffer_to_buffer source AND
    // destination across two encoders.
    if has_fold_consumers_outer {
        out.push_str(&format!(
            "        let prev_event_tail_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {{\n\
             \x20           label: Some(\"{fixture_name}::prev_event_tail\"),\n\
             \x20           size: 4u64,\n\
             \x20           usage: wgpu::BufferUsages::STORAGE\n\
             \x20               | wgpu::BufferUsages::COPY_SRC\n\
             \x20               | wgpu::BufferUsages::COPY_DST,\n\
             \x20           mapped_at_creation: false,\n\
             \x20       }});\n",
        ));
    }
    // Plan E-A6 — preload each kernel's `@runtime` config fields with
    // their .sim defaults. The standard 4-u32 cfg header (bytes 0..16)
    // lives at the start of every kernel's cfg struct; runtime fields
    // are appended after, in KernelSpec.runtime_cfg_fields order. Each
    // field is a 4-byte scalar today (u32/f32) so offset = 16 + idx*4.
    for spec in &artifacts.kernel_specs {
        if spec.runtime_cfg_fields.is_empty() {
            continue;
        }
        for (idx, (field_name, field_ty)) in spec.runtime_cfg_fields.iter().enumerate() {
            let def = match runtime_config_defaults.get(field_name) {
                Some(d) => d,
                None => continue,
            };
            let offset = 16 + idx * 4;
            let bytes_expr = match field_ty.as_str() {
                "f32" => format!("({}_f32).to_le_bytes()", def.default_lit),
                "u32" => format!("({}_u32).to_le_bytes()", def.default_lit),
                "i32" => format!("({}_i32).to_le_bytes()", def.default_lit),
                _ => continue,
            };
            out.push_str(&format!(
                "        gpu.queue.write_buffer(\n\
                 \x20           &cfg_{kernel}_buf,\n\
                 \x20           {offset}u64,\n\
                 \x20           &{bytes_expr},\n\
                 \x20       );\n",
                kernel = spec.name,
            ));
        }
    }
    out.push_str("        Some(Self {\n");
    out.push_str("            gpu,\n");
    out.push_str("            agent_count,\n");
    out.push_str("            seed,\n");
    out.push_str("            tick: 0,\n");
    out.push_str("            event_ring,\n");
    out.push_str("            registry_gpu,\n");
    out.push_str("            cache,\n");
    for (name, _) in &owned {
        out.push_str(&format!("            {name}_buf,\n"));
    }
    for kernel_name in &cfg_buffer_names {
        out.push_str(&format!("            cfg_{kernel_name}_buf,\n"));
    }
    if has_fold_consumers_outer {
        out.push_str("            prev_event_tail_buf,\n");
    }
    if binds_voxel_grid {
        out.push_str("            voxel_terrain,\n");
        out.push_str("            voxel_mirror,\n");
    }
    if binds_navgrid {
        out.push_str("            navgrid_buf,\n");
        out.push_str("            navgrid_cfg_buf,\n");
    }
    if needs_sort {
        out.push_str("            event_ring_sort_scratch_buf,\n");
        out.push_str("            radix_histogram_buf,\n");
        out.push_str("            radix_bucket_offsets_buf,\n");
        out.push_str("            target_histogram_buf,\n");
        out.push_str("            target_offsets_buf,\n");
        out.push_str("            sort_cfg_buf,\n");
        out.push_str("            sort_pipelines: None,\n");
    }
    for (name, def) in runtime_config_defaults {
        let lit_typed = match def.scalar_ty.as_str() {
            "f32" => format!("{}_f32", def.default_lit),
            "u32" => format!("{}_u32", def.default_lit),
            "i32" => format!("{}_i32", def.default_lit),
            _ => format!("{} as {}", def.default_lit, def.scalar_ty),
        };
        out.push_str(&format!("            {name}: {lit_typed},\n"));
    }
    out.push_str("        })\n");
    out.push_str("    }\n\n");

    // Radix sort dispatch method — synthesized only for fixtures that need
    // the sort pass. Lazily builds all 15 ComputePipelines on first call
    // and caches them in self.sort_pipelines. Each subsequent tick just
    // creates the per-tick bind groups and records dispatches.
    if needs_sort {
        let engine_stride: u64 = 11;
        let ring_bytes = 1_048_576_u64 * engine_stride * 4;

        // Build pipeline creation snippet for one kernel given:
        //   name = "radix_stage_a_pass0_histogram"
        //   entry = "radix_stage_a_pass0_histogram"
        //   bgl_entries = "engine::gpu::bgl_storage(0, true), ..."
        let make_pipeline = |kernel_name: &str, bgl_entries: &str| -> String {
            format!(
                "            {{\n\
                 \x20               let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {{\n\
                 \x20                   label: Some(\"{kernel_name}::wgsl\"),\n\
                 \x20                   source: wgpu::ShaderSource::Wgsl({kernel_name}::SHADER_SRC.into()),\n\
                 \x20               }});\n\
                 \x20               let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {{\n\
                 \x20                   label: Some(\"{kernel_name}::bgl\"),\n\
                 \x20                   entries: &[{bgl_entries}],\n\
                 \x20               }});\n\
                 \x20               let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{\n\
                 \x20                   label: Some(\"{kernel_name}::pl\"),\n\
                 \x20                   bind_group_layouts: &[&bgl],\n\
                 \x20                   push_constant_ranges: &[],\n\
                 \x20               }});\n\
                 \x20               let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{\n\
                 \x20                   label: Some(\"{kernel_name}::pipeline\"),\n\
                 \x20                   layout: Some(&pl),\n\
                 \x20                   module: &shader,\n\
                 \x20                   entry_point: Some(\"{kernel_name}\"),\n\
                 \x20                   compilation_options: Default::default(),\n\
                 \x20                   cache: None,\n\
                 \x20               }});\n\
                 \x20               (pipeline, bgl)\n\
                 \x20           }}",
            )
        };

        // Stage A histogram/scatter share 3 bindings:
        //   0: event_ring_in (storage read)
        //   1: event_tail (storage read)
        //   2: radix_histogram (storage read_write)
        // Stage A scan has 2 bindings:
        //   0: radix_histogram (storage read_write)
        //   1: radix_bucket_offsets (storage read_write)
        // Stage A scatter has 5 bindings:
        //   0: event_ring_in (storage read)
        //   1: event_tail (storage read)
        //   2: radix_histogram (storage read_write)
        //   3: radix_bucket_offsets (storage read)
        //   4: event_ring_out (storage read_write)
        // Stage B count:
        //   0: event_ring_in (storage read)
        //   1: event_tail (storage read)
        //   2: target_histogram (storage read_write)
        //   3: cfg (uniform)
        // Stage B scan:
        //   0: target_histogram (storage read_write)
        //   1: target_offsets (storage read_write)
        //   2: cfg (uniform)
        // Stage B scatter:
        //   0: event_ring_in (storage read)
        //   1: event_tail (storage read)
        //   2: target_histogram (storage read_write)
        //   3: target_offsets (storage read)
        //   4: event_ring_out (storage read_write)
        //   5: cfg (uniform)

        let histogram_bgl = "engine::gpu::bgl_storage(0, true), engine::gpu::bgl_storage(1, true), engine::gpu::bgl_storage(2, false)";
        let scan_bgl_a    = "engine::gpu::bgl_storage(0, false), engine::gpu::bgl_storage(1, false)";
        let scatter_bgl_a = "engine::gpu::bgl_storage(0, true), engine::gpu::bgl_storage(1, true), engine::gpu::bgl_storage(2, false), engine::gpu::bgl_storage(3, true), engine::gpu::bgl_storage(4, false)";
        let count_bgl_b   = "engine::gpu::bgl_storage(0, true), engine::gpu::bgl_storage(1, true), engine::gpu::bgl_storage(2, false), engine::gpu::bgl_uniform(3)";
        let scan_bgl_b    = "engine::gpu::bgl_storage(0, false), engine::gpu::bgl_storage(1, false), engine::gpu::bgl_uniform(2)";
        let scatter_bgl_b = "engine::gpu::bgl_storage(0, true), engine::gpu::bgl_storage(1, true), engine::gpu::bgl_storage(2, false), engine::gpu::bgl_storage(3, true), engine::gpu::bgl_storage(4, false), engine::gpu::bgl_uniform(5)";

        let mut method = String::new();
        method.push_str(
            "    fn run_radix_sort(&mut self, encoder: &mut wgpu::CommandEncoder) {\n\
             \x20       let device = &self.gpu.device;\n\
             \x20       // Lazily build + cache all 15 sort pipelines on first call.\n\
             \x20       if self.sort_pipelines.is_none() {\n\
             \x20           self.sort_pipelines = Some(SortPipelines {\n",
        );
        for pass in 0..4u32 {
            let h_name = format!("radix_stage_a_pass{pass}_histogram");
            let s_name = format!("radix_stage_a_pass{pass}_scan");
            let sc_name = format!("radix_stage_a_pass{pass}_scatter");
            method.push_str(&format!(
                "                stage_a_pass{pass}_histogram: {h_body},\n\
                 \x20               stage_a_pass{pass}_scan: {s_body},\n\
                 \x20               stage_a_pass{pass}_scatter: {sc_body},\n",
                h_body  = make_pipeline(&h_name,  histogram_bgl),
                s_body  = make_pipeline(&s_name,  scan_bgl_a),
                sc_body = make_pipeline(&sc_name, scatter_bgl_a),
            ));
        }
        method.push_str(&format!(
            "                stage_b_count:   {count_body},\n\
             \x20               stage_b_scan:    {scan_body},\n\
             \x20               stage_b_scatter: {scatter_body},\n",
            count_body   = make_pipeline("radix_stage_b_count",   count_bgl_b),
            scan_body    = make_pipeline("radix_stage_b_scan",    scan_bgl_b),
            scatter_body = make_pipeline("radix_stage_b_scatter", scatter_bgl_b),
        ));
        method.push_str(
            "            });\n\
             \x20       }\n\
             \x20       let p = self.sort_pipelines.as_ref().unwrap();\n\n\
             \x20       // Stage A: 4 LSD radix passes on the seq field (32 bits, 8 bits/pass).\n\
             \x20       // Ping-pong: even passes  read ring  → write scratch;\n\
             \x20       //            odd  passes  read scratch → write ring.\n\
             \x20       // After 4 passes (even count) data ends in event_ring (ring).\n",
        );

        // Stage A passes — ping-pong between ring and scratch.
        // .as_entire_binding() takes &self so no & prefix needed;
        // self.event_ring.ring() already returns &Buffer.
        for pass in 0..4u32 {
            let (in_buf, out_buf) = if pass % 2 == 0 {
                ("self.event_ring.ring()", "self.event_ring_sort_scratch_buf")
            } else {
                ("self.event_ring_sort_scratch_buf", "self.event_ring.ring()")
            };
            let h_field = format!("stage_a_pass{pass}_histogram");
            let s_field = format!("stage_a_pass{pass}_scan");
            let sc_field = format!("stage_a_pass{pass}_scatter");

            method.push_str(&format!(
                "\n\
                 \x20       // --- Stage A pass {pass} ---\n\
                 \x20       // Histogram\n\
                 \x20       {{\n\
                 \x20           let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{\n\
                 \x20               label: Some(\"sort::a{pass}_histogram\"),\n\
                 \x20               layout: &p.{h_field}.1,\n\
                 \x20               entries: &[\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 0, resource: {in_buf}.as_entire_binding() }},\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 1, resource: self.event_ring.tail().as_entire_binding() }},\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 2, resource: self.radix_histogram_buf.as_entire_binding() }},\n\
                 \x20               ],\n\
                 \x20           }});\n\
                 \x20           let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{ label: Some(\"sort::a{pass}_hist\"), timestamp_writes: None }});\n\
                 \x20           pass_enc.set_pipeline(&p.{h_field}.0);\n\
                 \x20           pass_enc.set_bind_group(0, &bg, &[]);\n\
                 \x20           pass_enc.dispatch_workgroups((engine::gpu::EVENT_RING_CAP_SLOTS + 63) / 64, 1, 1);\n\
                 \x20       }}\n\
                 \x20       // Scan\n\
                 \x20       {{\n\
                 \x20           let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{\n\
                 \x20               label: Some(\"sort::a{pass}_scan\"),\n\
                 \x20               layout: &p.{s_field}.1,\n\
                 \x20               entries: &[\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 0, resource: self.radix_histogram_buf.as_entire_binding() }},\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 1, resource: self.radix_bucket_offsets_buf.as_entire_binding() }},\n\
                 \x20               ],\n\
                 \x20           }});\n\
                 \x20           let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{ label: Some(\"sort::a{pass}_scan\"), timestamp_writes: None }});\n\
                 \x20           pass_enc.set_pipeline(&p.{s_field}.0);\n\
                 \x20           pass_enc.set_bind_group(0, &bg, &[]);\n\
                 \x20           pass_enc.dispatch_workgroups(1, 1, 1);\n\
                 \x20       }}\n\
                 \x20       // Scatter\n\
                 \x20       {{\n\
                 \x20           let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {{\n\
                 \x20               label: Some(\"sort::a{pass}_scatter\"),\n\
                 \x20               layout: &p.{sc_field}.1,\n\
                 \x20               entries: &[\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 0, resource: {in_buf}.as_entire_binding() }},\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 1, resource: self.event_ring.tail().as_entire_binding() }},\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 2, resource: self.radix_histogram_buf.as_entire_binding() }},\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 3, resource: self.radix_bucket_offsets_buf.as_entire_binding() }},\n\
                 \x20                   wgpu::BindGroupEntry {{ binding: 4, resource: {out_buf}.as_entire_binding() }},\n\
                 \x20               ],\n\
                 \x20           }});\n\
                 \x20           let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{ label: Some(\"sort::a{pass}_scatter\"), timestamp_writes: None }});\n\
                 \x20           pass_enc.set_pipeline(&p.{sc_field}.0);\n\
                 \x20           pass_enc.set_bind_group(0, &bg, &[]);\n\
                 \x20           pass_enc.dispatch_workgroups(1, 1, 1);\n\
                 \x20       }}\n",
            ));
        }

        // Stage B: counting sort on target_id. After stage A (4 even passes),
        // data is in event_ring. Stage B reads from ring, writes to scratch,
        // then we copy scratch → ring so fold consumers always read from ring.
        method.push_str(
            "\n\
             \x20       // --- Stage B: counting sort by target_id ---\n\
             \x20       // Count\n\
             \x20       {\n\
             \x20           let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {\n\
             \x20               label: Some(\"sort::b_count\"),\n\
             \x20               layout: &p.stage_b_count.1,\n\
             \x20               entries: &[\n\
             \x20                   wgpu::BindGroupEntry { binding: 0, resource: self.event_ring.ring().as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 1, resource: self.event_ring.tail().as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 2, resource: self.target_histogram_buf.as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 3, resource: self.sort_cfg_buf.as_entire_binding() },\n\
             \x20               ],\n\
             \x20           });\n\
             \x20           let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(\"sort::b_count\"), timestamp_writes: None });\n\
             \x20           pass_enc.set_pipeline(&p.stage_b_count.0);\n\
             \x20           pass_enc.set_bind_group(0, &bg, &[]);\n\
             \x20           pass_enc.dispatch_workgroups((engine::gpu::EVENT_RING_CAP_SLOTS + 63) / 64, 1, 1);\n\
             \x20       }\n\
             \x20       // Scan\n\
             \x20       {\n\
             \x20           let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {\n\
             \x20               label: Some(\"sort::b_scan\"),\n\
             \x20               layout: &p.stage_b_scan.1,\n\
             \x20               entries: &[\n\
             \x20                   wgpu::BindGroupEntry { binding: 0, resource: self.target_histogram_buf.as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 1, resource: self.target_offsets_buf.as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 2, resource: self.sort_cfg_buf.as_entire_binding() },\n\
             \x20               ],\n\
             \x20           });\n\
             \x20           let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(\"sort::b_scan\"), timestamp_writes: None });\n\
             \x20           pass_enc.set_pipeline(&p.stage_b_scan.0);\n\
             \x20           pass_enc.set_bind_group(0, &bg, &[]);\n\
             \x20           pass_enc.dispatch_workgroups(1, 1, 1);\n\
             \x20       }\n\
             \x20       // Scatter: ring → scratch\n\
             \x20       {\n\
             \x20           let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {\n\
             \x20               label: Some(\"sort::b_scatter\"),\n\
             \x20               layout: &p.stage_b_scatter.1,\n\
             \x20               entries: &[\n\
             \x20                   wgpu::BindGroupEntry { binding: 0, resource: self.event_ring.ring().as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 1, resource: self.event_ring.tail().as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 2, resource: self.target_histogram_buf.as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 3, resource: self.target_offsets_buf.as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 4, resource: self.event_ring_sort_scratch_buf.as_entire_binding() },\n\
             \x20                   wgpu::BindGroupEntry { binding: 5, resource: self.sort_cfg_buf.as_entire_binding() },\n\
             \x20               ],\n\
             \x20           });\n\
             \x20           let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(\"sort::b_scatter\"), timestamp_writes: None });\n\
             \x20           pass_enc.set_pipeline(&p.stage_b_scatter.0);\n\
             \x20           pass_enc.set_bind_group(0, &bg, &[]);\n\
             \x20           pass_enc.dispatch_workgroups(1, 1, 1);\n\
             \x20       }\n\
             \x20       // Copy sorted result (scratch) back to event_ring so fold consumers\n\
             \x20       // always read from the canonical ring buffer.\n",
        );
        method.push_str(&format!(
            "        encoder.copy_buffer_to_buffer(&self.event_ring_sort_scratch_buf, 0, self.event_ring.ring(), 0, {ring_bytes}u64);\n",
        ));
        method.push_str("    }\n\n");

        out.push_str(&method);
    }

    // Plan E-A6 — host-side setters for `@runtime` config fields. Each
    // setter updates the host-side mirror AND writes the value to every
    // kernel cfg buffer that references the field, at the per-kernel
    // offset within that kernel's cfg struct.
    for (name, def) in runtime_config_defaults {
        // Find every kernel that lists this field + its index there.
        let writes: Vec<(String, usize)> = artifacts
            .kernel_specs
            .iter()
            .filter_map(|spec| {
                spec.runtime_cfg_fields
                    .iter()
                    .position(|(n, _)| n == name)
                    .map(|idx| (spec.name.clone(), idx))
            })
            .collect();
        if writes.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "    pub fn set_{name}(&mut self, value: {ty}) {{\n\
             \x20       self.{name} = value;\n\
             \x20       let bytes = value.to_le_bytes();\n",
            ty = def.scalar_ty,
        ));
        for (kernel, idx) in writes {
            let offset = 16 + idx * 4;
            out.push_str(&format!(
                "        self.gpu.queue.write_buffer(&self.cfg_{kernel}_buf, {offset}u64, &bytes);\n",
            ));
        }
        out.push_str("    }\n\n");
    }

    // Voxel-region-indices Phase 4b — host-side navgrid uploader.
    // Copies a `NavgridIndex`'s cells into the storage buffer and
    // writes the matching `[size_x, size_z, origin_x, origin_z]`
    // uniform. Bounds-checks at the cap (NAVGRID_MAX_CELLS). Idempotent
    // — call once per navgrid rebuild (typically at fixture setup or
    // when terrain changes invalidate the cells).
    if binds_navgrid {
        out.push_str(
            "    pub fn upload_navgrid(&mut self, idx: &engine_voxel::NavgridIndex) {\n\
             \x20       let cell_count = idx.cells.len();\n\
             \x20       assert!(\n\
             \x20           cell_count <= engine_voxel::NAVGRID_MAX_CELLS as usize,\n\
             \x20           \"navgrid index has {} cells but the runtime buffer caps at NAVGRID_MAX_CELLS = {}\",\n\
             \x20           cell_count,\n\
             \x20           engine_voxel::NAVGRID_MAX_CELLS,\n\
             \x20       );\n\
             \x20       let cells_raw: Vec<u32> = idx.cells.iter().map(|c| c.0).collect();\n\
             \x20       self.gpu.queue.write_buffer(\n\
             \x20           &self.navgrid_buf,\n\
             \x20           0,\n\
             \x20           bytemuck::cast_slice(&cells_raw),\n\
             \x20       );\n\
             \x20       let cfg: [u32; 4] = [idx.size_x, idx.size_z, idx.origin_x as u32, idx.origin_z as u32];\n\
             \x20       self.gpu.queue.write_buffer(\n\
             \x20           &self.navgrid_cfg_buf,\n\
             \x20           0,\n\
             \x20           bytemuck::cast_slice(&cfg),\n\
             \x20       );\n\
             \x20   }\n\n",
        );
    }

    // Plan E-A4 — default step() body.
    //
    // Today: builds AgentBuffers + KernelBindingsContext + walks the
    // compiler-emitted SCHEDULE with empty match arms (every kernel
    // dispatch falls through to the catch-all `_ => {}`). Validates
    // the infrastructure scaffolding compiles end-to-end against a
    // real fixture without yet emitting per-kernel dispatch arms.
    //
    // A4.1+ will populate match arms per kernel — needs:
    //   * Per-kernel cfg struct construction (using KernelSpec.cfg_build_expr)
    //   * Per-kernel Extras struct construction (mapping bindings →
    //     fixture-owned buffers)
    //   * dispatch::dispatch_<kernel> call
    //   * Transient-source buffer alloc (mask_bitmaps, etc.) — not
    //     yet handled by the A3.2 alloc loop (External-only).
    out.push_str(
        "    /// Default step. Builds AgentBuffers + KernelBindingsContext,\n\
         \x20   /// writes per-tick cfg uniforms, and walks SCHEDULE.\n\
         \x20   /// Per-kernel dispatch arms land in A4.1.\n\
         \x20   pub fn step(&mut self) {\n",
    );
    // Voxel mirror flush — re-uploads any host-side `set_cell` writes
    // since the last step before any kernel reads `voxel_grid`. Cheap
    // when nothing's dirty (early-return on empty set inside
    // `flush_dirty`); essential when the host seeded terrain (test
    // fixture or chronicle drain) and a kernel needs the new cells.
    if binds_voxel_grid {
        out.push_str(
            "        self.voxel_mirror.flush_dirty(&self.gpu, self.voxel_terrain.grid());\n",
        );
    }
    out.push_str(
        "        // Per-tick cfg uniform write to every kernel's cfg buffer.\n\
         \x20       // Layout: [slot0, tick, seed, slot3] where slot0 is\n\
         \x20       // agent_cap (per-agent kernels) or event_count (per-event\n\
         \x20       // kernels). Today we write agent_count to slot0 — for\n\
         \x20       // an empty event_ring this over-bounds harmlessly because\n\
         \x20       // the kernel's event-kind check on each row falls through.\n\
         \x20       // Cfg layout per kernel:\n\
         \x20       //   per-agent: { agent_cap, tick, seed, _pad0 }\n\
         \x20       //   ViewFold:  { event_count, tick, second_key_pop, _pad0 }\n\
         \x20       //   PerPair:   { agent_cap, tick, seed, pair_offset }\n\
         \x20       // Slot 2 is `seed` (per-agent — most kernels ignore it,\n\
         \x20       // they key PCG off tick+agent+purpose) or `second_key_pop`\n\
         \x20       // (ViewFold — must be 1 for single-key views, otherwise\n\
         \x20       // the per-(observer, source) index calc divides by 0 or\n\
         \x20       // wraps and the fold writes to wrong slots). Writing 1\n\
         \x20       // is correct for ViewFolds and harmless for per-agent.\n\
         \x20       // Slot 3 is _pad0 for per-agent/ViewFold (unused) and\n\
         \x20       // `pair_offset` for PerPair fused-mask kernels — those\n\
         \x20       // emit `let pair = gid.x + cfg._pad0` to chunk\n\
         \x20       // agent_cap²-sized dispatches that exceed\n\
         \x20       // `max_compute_workgroups_per_dimension`. Single-shot\n\
         \x20       // PerPair dispatches MUST set pair_offset = 0 so that\n\
         \x20       // gid.x = 0..agent_cap² covers the full pair grid.\n\
         \x20       // Pre-2026-05-12 squad_skirmish fix: slot3 was being\n\
         \x20       // written as `self.agent_count`, which made every PerPair\n\
         \x20       // mask kernel skip the first `agent_cap` pairs and walk\n\
         \x20       // pairs [agent_cap..agent_cap+dispatch_threads). For\n\
         \x20       // agent_count=16 with `(agent_cap+63)/64=1` workgroup\n\
         \x20       // (64 threads), only pairs 16..79 ran — agent 0's mask bit\n\
         \x20       // never got set and damage didn't flow. The fix lands\n\
         \x20       // pair_offset=0 universally; chunked dispatch (megaswarm\n\
         \x20       // 10000) would override per-batch but uses a separate\n\
         \x20       // emit path.\n\
         \x20       // Slot 2 = seed for per-agent kernels (wires through to\n\
         \x20       // `per_agent_u32(seed, agent_id, tick, purpose)` —\n\
         \x20       // see P5). Was `1u32` historically; now uses\n\
         \x20       // `self.seed as u32` so different `try_new(seed, …)`\n\
         \x20       // calls actually produce different RNG streams.\n\
         \x20       // ViewFold kernels get a slot-2 OVERRIDE below to\n\
         \x20       // restore the `second_key_pop = K` semantic their\n\
         \x20       // pair-keyed index expression depends on.\n\
         \x20       let cfg_words: [u32; 4] = [\n\
         \x20           self.agent_count,\n\
         \x20           self.tick as u32,\n\
         \x20           self.seed as u32,\n\
         \x20           0u32,\n\
         \x20       ];\n\
         \x20       let cfg_bytes: &[u8] = bytemuck::cast_slice(&cfg_words);\n",
    );
    for kernel_name in &cfg_buffer_names {
        out.push_str(&format!(
            "        self.gpu.queue.write_buffer(&self.cfg_{kernel_name}_buf, 0, cfg_bytes);\n",
        ));
    }

    // Per-kernel slot-2 override for ViewFold kernels — their cfg
    // shape is `{ event_count, tick, second_key_pop, _pad0 }`, so
    // slot 2 needs the view's static K value (not the default `1u32`
    // baked into cfg_words above). Without this override, pair-keyed
    // ViewFolds (`(observer: Agent, subject: Agent)` or I.3b's
    // `(observer: Agent, key: u32) @key_pop(K=N)`) compose their
    // 2-D index as `local_first * 1 + local_second`, addressing the
    // wrong storage cell at any non-trivial agent_count or K. Per-
    // agent kernels keep slot 2 = seed (untouched by this override),
    // so this loop strictly fixes ViewFold; for pair-keyed views
    // whose second key is Agent (K = agent_count, runtime-variable)
    // we emit `self.agent_count.to_le_bytes()`, for everything else
    // (Item/Group/Quest/KeyTyped) the K is compile-time-constant.
    for spec in &artifacts.kernel_specs {
        if !matches!(spec.kind, crate::kernel_binding_ir::KernelKind::ViewFold) {
            continue;
        }
        let view_name = spec.name.strip_prefix("fold_").unwrap_or("");
        let Some(Some(pair_keyed)) = view_pair_keyed.get(view_name).copied() else {
            continue;
        };
        let k_expr = match pair_keyed {
            PairKeyedSecondKey::Agent => "self.agent_count".to_string(),
            PairKeyedSecondKey::Item(n)
            | PairKeyedSecondKey::Group(n)
            | PairKeyedSecondKey::Quest(n)
            | PairKeyedSecondKey::KeyTyped(n) => format!("{n}u32"),
        };
        out.push_str(&format!(
            "        // ViewFold slot-2 override — sets cfg.second_key_pop = K for view `{view_name}`.\n\
             \x20       {{\n\
             \x20           let k_bytes: [u8; 4] = ({k_expr}).to_le_bytes();\n\
             \x20           self.gpu.queue.write_buffer(&self.cfg_{kname}_buf, 8u64, &k_bytes);\n\
             \x20       }}\n",
            kname = spec.name,
        ));
    }
    // Post-sort fold slot-3 override — when events are sorted before the fold
    // (needs_sort), the fold kernels switch from per-event dispatch to per-slot
    // dispatch and use cfg._pad0 (slot 3, offset 12) as the agent_cap bound for
    // the per-slot gid.x check. Write agent_count there so the WGSL
    // `if (my_slot >= cfg._pad0) { return; }` exits threads beyond the agent
    // population. Without this, threads in workgroup tail-padding (slots
    // agent_count..ceil(agent_count/64)*64) would attempt to accumulate into
    // out-of-bounds storage slots.
    if needs_sort {
        for spec in &artifacts.kernel_specs {
            if !matches!(spec.kind, crate::kernel_binding_ir::KernelKind::ViewFold) {
                continue;
            }
            out.push_str(&format!(
                "        // ViewFold slot-3 override (post-sort) — sets cfg._pad0 = agent_count for per-slot bounds.\n\
                 \x20       {{\n\
                 \x20           let cap_bytes: [u8; 4] = self.agent_count.to_le_bytes();\n\
                 \x20           self.gpu.queue.write_buffer(&self.cfg_{kname}_buf, 12u64, &cap_bytes);\n\
                 \x20       }}\n",
                kname = spec.name,
            ));
        }
    }
    // Indirect-consumer event-ring lifecycle — closes gaps 3 + 4
    // from commit 353527e6's Indirect-arm doc block, and folds in the
    // in-step GPU producer case via an in-encoder
    // `copy_buffer_to_buffer(event_tail → cfg.event_count)` per
    // consumer dispatch.
    //
    // Background: chronicle consumers (e.g. `physics_ApplyObserveBeliefUpdate`,
    // `physics_ApplyDamage`) classify as `KernelTopology::Indirect` and
    // emit as `DispatchOp::Indirect { kernel: KernelId::X, args_buf:
    // BufferRef::ResidentIndirectArgs }` in SCHEDULE. Their bodies
    // bound on `cfg.event_count` (`if event_idx >= cfg.event_count
    // { return; }`). The default cfg write above sets `event_count =
    // agent_count` — wrong, because:
    //   * agent_count is 16..1024 typical, but the real event count is
    //     the ring tail (0..N depending on what producer kernels +
    //     host-side injects emitted).
    //   * Old slot-0 records left from a previous tick (e.g. a
    //     `effect_observe_applied()` host inject 50 ticks ago) STAY in
    //     the ring (only the tail counter resets), and the consumer's
    //     kind-tag filter would re-process them every tick with the
    //     `agent_count` upper bound. That re-fire is the
    //     `tom_probe_decay_pin::fresh_observation_resets_decay`
    //     regression the naive Indirect-arm wire-up tripped.
    //
    // Two-pronged fix:
    //
    //   (a) **Host-inject preservation.** When a fixture has Indirect
    //       consumers, replace `clear_tail_in` (encoder copy of 0 into
    //       event_tail) with a `queue.write_buffer` of the host-side
    //       `tail_value()` snapshot. Both auto-emitted host-side
    //       injects (`effect_*_applied` typed methods) and any other
    //       caller of `note_emits` / `append_chronicle_record` get
    //       their records preserved into the next step()'s producer
    //       atomicAdd window. `wgpu::Queue::write_buffer` sequences
    //       before all encoder commands, so the value lands before
    //       any producer kernel sees the buffer.
    //
    //   (b) **GPU-side tail propagation.** For each Indirect consumer,
    //       emit an `encoder.copy_buffer_to_buffer(event_tail, 0,
    //       cfg_<name>_buf, 0, 4)` IMMEDIATELY BEFORE the consumer's
    //       dispatch. This transcribes the live GPU tail (including
    //       all events emitted by in-step GPU producers AND prior
    //       Indirect consumers in this same encoder) into the
    //       consumer's `cfg.event_count` slot, so the consumer walks
    //       0..tail and stops, instead of walking 0..agent_count and
    //       reading garbage past tail. This unlocks fixtures like
    //       `hill_raid` whose `apply_ability` rules emit events via
    //       `atomicAdd(&event_tail, 1u)` from inside the kernel — the
    //       host has zero visibility into the count, but the in-encoder
    //       copy reads the live GPU value at consumer-dispatch time.
    //
    // The per-tick cfg write above still sets `event_count =
    // agent_count` for every kernel; the per-consumer
    // copy_buffer_to_buffer overwrites the Indirect kernels' slot 0
    // before their dispatch lands. The order matters — wgpu's encoder
    // commands run in submission order, so the copy lands BEFORE the
    // dispatch reads the cfg uniform.
    let has_indirect_consumers = !indirect_consumer_kernel_names.is_empty();
    if has_indirect_consumers {
        out.push_str(
            "        // Snapshot the host-side tail estimate (counts host-injected\n\
             \x20       // chronicle records pending from `effect_*_applied()` /\n\
             \x20       // `inject_chronicle_record()` calls since the last step).\n\
             \x20       // See the multi-paragraph step() comment for the 4-gap fix.\n\
             \x20       let pending_event_count: u32 = self.event_ring.tail_value();\n",
        );
    }
    // Fold consumer prior-tick tail snapshot.
    //
    // Folds bind `event_ring`/`event_tail` and consume PRIOR-tick
    // records (the ViewFold contract: a fold at tick T sees emits from
    // tick T-1). Their bodies bound on `cfg.event_count`; the per-tick
    // host write above set that slot to `agent_count` (over-bounds),
    // so folds were walking ALL `agent_count` ring slots including
    // stale records from ticks past T-1, and producer atomicAdd
    // reordering across runs flipped which fresh slot landed where
    // (forest_fire_pin observed `max |Δ| ≈ 470 / 1024` slots).
    //
    // Fix: BEFORE clear_tail_in (or the indirect-consumer
    // queue.write_buffer of `pending_event_count`) overwrites the GPU
    // event_tail for THIS tick, snapshot it into a tiny side buffer
    // `prev_event_tail_buf`. Then in the main encoder, copy that side
    // buffer into each fold's `cfg.event_count` slot — this lands
    // AFTER the per-tick `cfg_bytes` queue.write_buffer (which sets
    // slot 0 to agent_count) since encoder commands run after queue
    // writes, so slot 0 ends up holding the prior-tick tail.
    //
    // Indirect chronicle consumers (`physics_Apply*`-shape, in-tick
    // semantics) take a DIFFERENT cfg copy — at per-dispatch time
    // from the LIVE event_tail — so they pick up THIS tick's GPU
    // producer atomicAdds. Folds need prior-tick semantics; indirect
    // consumers need in-tick semantics. Mixed fixtures (folds +
    // indirect consumers) are correctly handled by the two-stage
    // pattern: snapshot encoder runs first (capturing prior-tick GPU
    // tail before the host queue.write_buffer of pending_event_count),
    // then the main encoder applies clear/preserve and dispatches.
    let fold_consumer_kernel_names: Vec<&str> = fold_consumer_kernel_names_owned
        .iter()
        .map(|s| s.as_str())
        .collect();
    let has_fold_consumers = has_fold_consumers_outer;
    if has_fold_consumers {
        // Stage 1: capture event_tail into prev_event_tail_buf via its
        // own submit. This MUST land before the host enqueues the
        // per-tick cfg_bytes write or (for indirect-consumer fixtures)
        // the queue.write_buffer of pending_event_count, both of which
        // would otherwise sequence before any subsequent encoder
        // command and clobber what event_tail holds.
        if needs_sort {
            // When the fixture has both fold consumers and a sort pass,
            // the sort runs in the snap_encoder so it sees the prior-tick
            // GPU tail value. The main encoder resets the tail to
            // pending_event_count (a queue.write_buffer that lands before
            // main-encoder commands on submit), so running the sort in the
            // main encoder would see tail=0 and sort nothing.
            out.push_str(
                "        // Capture prior-tick GPU event_tail and sort prior-tick\n\
                 \x20       // events by (target, seq) — both in the snap encoder so\n\
                 \x20       // sort kernels see the prior-tick tail before the main\n\
                 \x20       // encoder's write_buffer resets it to pending_event_count.\n\
                 \x20       {\n\
                 \x20           let mut snap_encoder = self.gpu.device.create_command_encoder(\n\
                 \x20               &wgpu::CommandEncoderDescriptor {\n\
                 \x20                   label: Some(concat!(env!(\"CARGO_PKG_NAME\"), \"::fold_tail_snapshot\")),\n\
                 \x20               },\n\
                 \x20           );\n\
                 \x20           snap_encoder.copy_buffer_to_buffer(\n\
                 \x20               self.event_ring.tail(),\n\
                 \x20               0,\n\
                 \x20               &self.prev_event_tail_buf,\n\
                 \x20               0,\n\
                 \x20               4,\n\
                 \x20           );\n\
                 \x20           self.run_radix_sort(&mut snap_encoder);\n\
                 \x20           self.gpu.queue.submit(Some(snap_encoder.finish()));\n\
                 \x20       }\n",
            );
        } else {
            out.push_str(
                "        // Capture prior-tick GPU event_tail\n\
                 \x20       // into prev_event_tail_buf via its own submit BEFORE the\n\
                 \x20       // main step encoder enqueues anything that overwrites the\n\
                 \x20       // GPU tail (clear_tail_in or pending_event_count write).\n\
                 \x20       // The captured value drives each fold consumer's\n\
                 \x20       // cfg.event_count for THIS tick (folds-at-T see T-1 emits).\n\
                 \x20       {\n\
                 \x20           let mut snap_encoder = self.gpu.device.create_command_encoder(\n\
                 \x20               &wgpu::CommandEncoderDescriptor {\n\
                 \x20                   label: Some(concat!(env!(\"CARGO_PKG_NAME\"), \"::fold_tail_snapshot\")),\n\
                 \x20               },\n\
                 \x20           );\n\
                 \x20           snap_encoder.copy_buffer_to_buffer(\n\
                 \x20               self.event_ring.tail(),\n\
                 \x20               0,\n\
                 \x20               &self.prev_event_tail_buf,\n\
                 \x20               0,\n\
                 \x20               4,\n\
                 \x20           );\n\
                 \x20           self.gpu.queue.submit(Some(snap_encoder.finish()));\n\
                 \x20       }\n",
            );
        }
    }
    // Fallback: needs_sort but no fold consumers (unusual; normally
    // needs_sort => has_fold_consumers). Submit sort in its own encoder
    // before the main encoder to preserve prior-tick tail semantics.
    if needs_sort && !has_fold_consumers {
        out.push_str(
            "        {\n\
             \x20           let mut sort_encoder = self.gpu.device.create_command_encoder(\n\
             \x20               &wgpu::CommandEncoderDescriptor {\n\
             \x20                   label: Some(concat!(env!(\"CARGO_PKG_NAME\"), \"::sort\")),\n\
             \x20               },\n\
             \x20           );\n\
             \x20           self.run_radix_sort(&mut sort_encoder);\n\
             \x20           self.gpu.queue.submit(Some(sort_encoder.finish()));\n\
             \x20       }\n",
        );
    }
    out.push_str(
        "\n\
         \x20       let mut encoder = self.gpu.device.create_command_encoder(\n\
         \x20           &wgpu::CommandEncoderDescriptor {\n\
         \x20               label: Some(concat!(env!(\"CARGO_PKG_NAME\"), \"::step\")),\n\
         \x20           },\n\
         \x20       );\n",
    );
    // Mask bitmap clear (2026-05-12 squad_skirmish gap): the
    // fused-mask kernel sets bits via `atomicOr` with no clear step,
    // so bits latched at tick 0 stay set forever. The cooldown
    // predicate `(tick % cooldown_X == 0)` evaluates true at tick 0
    // (every cooldown), and the latched bit means scoring picks the
    // verb every subsequent tick regardless of cooldown — collapsing
    // the per-tick gate. `clear_buffer(buf, 0, None)` zeroes each
    // mask bitmap right after the encoder is created so the
    // fused-mask kernel re-evaluates from a clean slate every tick.
    let mut mask_bitmap_buf_names: Vec<String> = owned
        .keys()
        .filter(|n| n.starts_with("mask_") && n.ends_with("_bitmap"))
        .cloned()
        .collect();
    mask_bitmap_buf_names.sort();
    for buf_name in &mask_bitmap_buf_names {
        out.push_str(&format!(
            "        encoder.clear_buffer(&self.{buf_name}_buf, 0, None);\n",
        ));
    }
    // Stage 2: copy prev_event_tail_buf → each fold's cfg.event_count
    // slot inside the main encoder. queue.write_buffer of cfg_bytes
    // landed earlier (host-side enqueue) and writes slot 0 to
    // agent_count; this encoder copy runs after queue effects on
    // submit, so slot 0 ends up holding the snapshot value.
    for kname in &fold_consumer_kernel_names {
        out.push_str(&format!(
            "        // Fold cfg.event_count from prev-tick snapshot.\n\
             \x20       encoder.copy_buffer_to_buffer(\n\
             \x20           &self.prev_event_tail_buf,\n\
             \x20           0,\n\
             \x20           &self.cfg_{kname}_buf,\n\
             \x20           0,\n\
             \x20           4,\n\
             \x20       );\n",
        ));
    }
    if has_indirect_consumers {
        // Replace clear_tail_in with a queue.write_buffer of the
        // pending count. queue.write_buffer is sequenced before all
        // encoder commands, so producers atomicAdd from
        // pending_event_count upward; injected slot 0..pending-1
        // records survive intact. The host estimate resets so the
        // next step() starts clean (the same guarantee
        // clear_tail_in's `tail_estimate = 0` line provided).
        out.push_str(
            "        self.gpu.queue.write_buffer(\n\
             \x20           self.event_ring.tail(),\n\
             \x20           0,\n\
             \x20           &pending_event_count.to_le_bytes(),\n\
             \x20       );\n\
             \x20       self.event_ring.reset_tail_estimate();\n",
        );
    } else {
        out.push_str("        self.event_ring.clear_tail_in(&mut encoder);\n");
    }
    out.push_str(
        "\n\
         \x20       let agent_buffers = engine::gpu::AgentBuffers {\n",
    );
    // Populate AgentBuffers' standard column fields from any matching
    // agent_<col>_buf the runtime owns. Each entry: `<col>_buf:
    // Some(&self.agent_<col>_buf),`.
    for (name, _) in &owned {
        let suffix = match name.strip_prefix("agent_") {
            Some(s) => s,
            None => continue,
        };
        if !is_standard_agent_column(name) {
            continue;
        }
        out.push_str(&format!(
            "            {suffix}_buf: Some(&self.{name}_buf),\n",
        ));
    }
    out.push_str(
        "            ..Default::default()\n\
         \x20       };\n",
    );
    // Voxel-region-indices Phase 4b — the runtime owns `navgrid_buf`
    // + `navgrid_cfg_buf` only when a fixture binds the navgrid
    // namespace (auto-detected like `binds_voxel_grid`). The
    // `binds_navgrid` flag is computed alongside, and lets the
    // generator either pass `Some(&self.navgrid_buf)` or `None`.
    if binds_voxel_grid && binds_navgrid {
        out.push_str(
            "        let _ctx = engine::gpu::KernelBindingsContext {\n\
             \x20           state: &agent_buffers,\n\
             \x20           event_ring: &self.event_ring,\n\
             \x20           registry: &self.registry_gpu,\n\
             \x20           voxel_grid: Some(self.voxel_mirror.buffer()),\n\
             \x20           navgrid: Some(&self.navgrid_buf),\n\
             \x20           navgrid_cfg: Some(&self.navgrid_cfg_buf),\n\
             \x20       };\n",
        );
    } else if binds_voxel_grid {
        out.push_str(
            "        let _ctx = engine::gpu::KernelBindingsContext {\n\
             \x20           state: &agent_buffers,\n\
             \x20           event_ring: &self.event_ring,\n\
             \x20           registry: &self.registry_gpu,\n\
             \x20           voxel_grid: Some(self.voxel_mirror.buffer()),\n\
             \x20           navgrid: None,\n\
             \x20           navgrid_cfg: None,\n\
             \x20       };\n",
        );
    } else if binds_navgrid {
        out.push_str(
            "        let _ctx = engine::gpu::KernelBindingsContext {\n\
             \x20           state: &agent_buffers,\n\
             \x20           event_ring: &self.event_ring,\n\
             \x20           registry: &self.registry_gpu,\n\
             \x20           voxel_grid: None,\n\
             \x20           navgrid: Some(&self.navgrid_buf),\n\
             \x20           navgrid_cfg: Some(&self.navgrid_cfg_buf),\n\
             \x20       };\n",
        );
    } else {
        out.push_str(
            "        let _ctx = engine::gpu::KernelBindingsContext {\n\
             \x20           state: &agent_buffers,\n\
             \x20           event_ring: &self.event_ring,\n\
             \x20           registry: &self.registry_gpu,\n\
             \x20           voxel_grid: None,\n\
             \x20           navgrid: None,\n\
             \x20           navgrid_cfg: None,\n\
             \x20       };\n",
        );
    }
    out.push_str(
        "\n\
         \x20       for op in schedule::SCHEDULE {\n\
         \x20           match op {\n",
    );
    // FixedPoint kernel detection (Gap forest_fire#F follow-up).
    //
    // The schedule synthesizer (`cg::emit::cross_cutting::synthesize_schedule`)
    // emits `DispatchOp::FixedPoint { kernel: KernelId::Pascal, max_iter: N }`
    // entries for `@cascade(max_iter=N)`-annotated physics rules (see
    // `classify_topology_for_schedule`'s FixedPoint branches). The runtime
    // arm for these is the same as `DispatchOp::Kernel(...)` except the
    // dispatch body runs inside a `for _iter in 0..N` loop.
    //
    // MVP semantics: unconditional loop, no early-break on no-change.
    // Re-running an idempotent fixed-point kernel past its convergence
    // tick is safe (the kernel observes its own writes from the previous
    // iteration via the agent buffers, and a converged state is by
    // definition stable under further application). A true early-break
    // would need a per-iteration "changed" flag read back from the GPU,
    // which is a separately-scoped follow-up tracked in the doc note
    // on `cg/program.rs::cascade_max_iter`.
    //
    // Parse the already-emitted `schedule.rs` string to discover which
    // kernels are scheduled as `FixedPoint` (keyed by pascal name, which
    // is what the SCHEDULE entry references). This avoids re-walking the
    // schedule or duplicating `classify_topology_for_schedule` logic.
    //
    // A kernel can in principle be both `KernelTopology::Indirect`
    // (placing it in `indirect_consumer_kernel_names` upstream) AND
    // scheduled as `FixedPoint` (when the topology carries a
    // `@cascade(max_iter=N)` annotation). Today's no fixture exercises
    // this overlap (the cascade tests use a per-agent physics rule which
    // lowers to a Split or Fused topology, not Indirect), so the MVP
    // treats the two paths as mutually exclusive: the FixedPoint arm
    // replaces — does not augment — the Indirect or-pattern for those
    // kernels. The cfg-copy block stays a pre-loop step in practice
    // because the only Indirect topologies that flow into FixedPoint are
    // hypothetical; if a future fixture combines them we'd want the
    // event_tail copy *inside* the loop iteration so each fixed-point
    // sweep sees fresh events emitted by the previous iteration. Flagged
    // here for that future investigation.
    let fixed_point_kernels: std::collections::BTreeMap<String, u32> = {
        let mut map = std::collections::BTreeMap::new();
        let schedule_src = artifacts
            .rust_files
            .get("schedule.rs")
            .map(|s| s.as_str())
            .unwrap_or("");
        // Match lines of the form (with arbitrary leading whitespace):
        //     DispatchOp::FixedPoint { kernel: KernelId::<Pascal>, max_iter: <N> }
        // The synthesizer emits exactly this shape per-entry; format is
        // deterministic so a simple substring scan suffices.
        for line in schedule_src.lines() {
            let trimmed = line.trim();
            let after_marker = match trimmed.strip_prefix("DispatchOp::FixedPoint { kernel: KernelId::") {
                Some(s) => s,
                None => continue,
            };
            let (pascal, rest) = match after_marker.split_once(',') {
                Some(p) => p,
                None => continue,
            };
            let pascal = pascal.trim().to_string();
            let max_iter_str = match rest.trim().strip_prefix("max_iter:") {
                Some(s) => s.trim().trim_end_matches(',').trim_end_matches('}').trim(),
                None => continue,
            };
            let max_iter: u32 = match max_iter_str.parse() {
                Ok(n) if n > 0 => n,
                _ => continue,
            };
            map.insert(pascal, max_iter);
        }
        map
    };
    // Plan E-A4.1 — per-kernel dispatch arms.
    //
    // Mirrors `cg::emit::program::classify_binding` rules EXACTLY (by
    // binding name, not bg_source — that was the bug in attempt #1).
    // For each kernel with from_context_with_extras, build the Extras
    // struct literal + dispatch.
    for spec in &artifacts.kernel_specs {
        let kernel_rs = artifacts
            .rust_files
            .get(&format!("{}.rs", spec.name))
            .map(|s| s.as_str())
            .unwrap_or("");
        if !kernel_rs.contains("from_context_with_extras") {
            // A4.2 — emit direct Bindings { ... } construction for
            // ViewFold kernels (no Extras helper). Walk bindings,
            // render each field per name-based classification.
            //
            // Per-view storage rename (mirror of the alloc loop): in
            // fold/decay kernels the BGL slot names
            // `view_storage_primary|anchor|ids` are uniform, so the
            // host needs to route each one to the THIS-kernel's
            // per-view buffer (`view_storage_<view>_<slot>_buf`).
            // Without this, every fold kernel binds the same
            // `view_storage_primary_buf` and writes alias across
            // views (the 6-fixture aliasing gap).
            let view_name_for_kernel = view_name_from_kernel_spec(spec);
            let mut binding_fields: Vec<String> = Vec::new();
            for b in &spec.bindings {
                use crate::kernel_binding_ir::BgSource;
                if matches!(b.bg_source, BgSource::AliasOf(_)) {
                    continue;
                }
                let name = b.name.as_str();
                // The struct field name (left of the colon) stays
                // the BGL-level name (`view_storage_primary` etc.) —
                // that's what the kernel module's `Bindings` struct
                // declares. The buffer expression on the RIGHT
                // routes to the per-view-named host field.
                let buf_field_name = view_storage_per_view_name(name, view_name_for_kernel);
                let value = if name == "event_ring" {
                    "self.event_ring.ring()".to_string()
                } else if name == "event_tail" {
                    "self.event_ring.tail()".to_string()
                } else if name == "sim_cfg" {
                    "self.event_ring.sim_cfg()".to_string()
                } else if name == "cfg" {
                    format!("&self.cfg_{}_buf", spec.name)
                } else if name == "voxel_grid" {
                    // ViewFold kernels don't bind voxel_grid; skip if seen.
                    continue;
                } else if let Some(suffix) = name.strip_prefix("agent_") {
                    if is_standard_agent_column(name) {
                        // Standard column from AgentBuffers.
                        format!(
                            "agent_buffers.{suffix}_buf.expect(\"kernel binds {name} but agent_buffers.{suffix}_buf is None\")"
                        )
                    } else {
                        format!("&self.{name}_buf")
                    }
                } else if name.starts_with("ability_registry_") {
                    let col = name.strip_prefix("ability_registry_").unwrap();
                    format!("&self.registry_gpu.{col}")
                } else {
                    format!("&self.{buf_field_name}_buf")
                };
                // anchor/ids are Option<&Buffer> for view fold kernels.
                let value = if name == "view_storage_anchor"
                    || name == "view_storage_ids"
                {
                    format!("Some({value})")
                } else {
                    value
                };
                binding_fields.push(format!(
                    "                        {name}: {value},"
                ));
            }
            let body = binding_fields.join("\n");
            let dispatch_fn = format!("dispatch_{}", spec.name.to_lowercase());
            // Indirect-consumer kernels match BOTH `Kernel(...)` AND
            // `Indirect { kernel, .. }` arms (the latter is what the
            // synthesized SCHEDULE actually contains for a chronicle
            // consumer). The dispatch body is identical — the consumer
            // kernel reads `cfg.event_count` (which step() wrote with
            // the host tail snapshot above) to bound its event walk.
            // Keeping `Kernel(...)` in the or-pattern is defensive: if
            // the schedule classifier ever re-routes a consumer back to
            // `Kernel`, the arm still fires.
            //
            // FixedPoint kernels (`@cascade(max_iter=N)`) get a distinct
            // arm matching `DispatchOp::FixedPoint { kernel, .. }` and
            // wrap the dispatch body in a `for _iter in 0..N` loop. See
            // the `fixed_point_kernels` doc block at the top of the loop
            // for the MVP semantics (unconditional loop, no early-break).
            let is_indirect_consumer = indirect_consumer_kernel_names
                .iter()
                .any(|n| n == &spec.name);
            let fixed_point_max_iter = fixed_point_kernels.get(&spec.pascal).copied();
            let arm_pattern = if let Some(_) = fixed_point_max_iter {
                format!(
                    "schedule::DispatchOp::FixedPoint {{ kernel: KernelId::{pascal}, .. }}",
                    pascal = spec.pascal,
                )
            } else if is_indirect_consumer {
                format!(
                    "schedule::DispatchOp::Kernel(KernelId::{pascal}) \
                     | schedule::DispatchOp::Indirect {{ kernel: KernelId::{pascal}, .. }}",
                    pascal = spec.pascal,
                )
            } else {
                format!(
                    "schedule::DispatchOp::Kernel(KernelId::{pascal})",
                    pascal = spec.pascal,
                )
            };
            // For Indirect consumers, copy the live GPU event_tail
            // value into cfg.event_count (slot 0) right before the
            // dispatch. This picks up events emitted by in-step GPU
            // producers (e.g. `apply_ability` calling
            // `atomicAdd(&event_tail, 1u)`) AND any host-side injects
            // preserved across the per-tick `queue.write_buffer` of
            // `pending_event_count`.
            //
            // Folds bind event_ring/event_tail too, but they consume
            // PRIOR-tick records (the docstring contract: "folds at
            // tick T see emits from tick T-1") — so they get their
            // event_count snapshotted at the TOP of step() before
            // clear_tail_in zeroes the GPU tail: the prev-tick
            // snapshot lives in `prev_event_tail_buf` and is copied
            // into each fold's cfg.event_count there, not here.
            //
            // FixedPoint kernels skip the cfg-copy block — today no
            // `@cascade`-annotated kernel is also an Indirect chronicle
            // consumer (the cascade fixtures use per-agent physics rules
            // which lower to Split / Fused topologies). If a future
            // fixture combines them, the copy would belong INSIDE the
            // for-loop so each iteration sees fresh events; flagged
            // in the top-of-loop doc block for that follow-up.
            let cfg_copy_block = if is_indirect_consumer && fixed_point_max_iter.is_none() {
                format!(
                    "                    encoder.copy_buffer_to_buffer(\n\
                     \x20                       self.event_ring.tail(),\n\
                     \x20                       0,\n\
                     \x20                       &self.cfg_{kname}_buf,\n\
                     \x20                       0,\n\
                     \x20                       4,\n\
                     \x20                   );\n",
                    kname = spec.name,
                )
            } else {
                String::new()
            };
            let (loop_open, loop_close) = if let Some(n) = fixed_point_max_iter {
                (
                    format!("                    for _iter in 0..{n}u32 {{\n"),
                    "                    }\n".to_string(),
                )
            } else {
                (String::new(), String::new())
            };
            out.push_str(&format!(
                "                {arm_pattern} => {{\n\
                 {cfg_copy_block}\
                 {loop_open}\
                 \x20                   let bindings = {kname}::{pascal}Bindings {{\n\
                 {body}\n\
                 \x20                   }};\n\
                 \x20                   dispatch::{dispatch_fn}(\n\
                 \x20                       &mut self.cache,\n\
                 \x20                       &bindings,\n\
                 \x20                       &self.gpu.device,\n\
                 \x20                       &mut encoder,\n\
                 \x20                       self.agent_count,\n\
                 \x20                   );\n\
                 {loop_close}\
                 \x20               }}\n",
                kname = spec.name,
                pascal = spec.pascal,
            ));
            continue;
        }
        // Walk bindings, mirror classify_binding name rules.
        // Per-view storage rename also applies here for defensive
        // symmetry — generic kernels classified as `fold_*` /
        // `decay_*` would otherwise route the BGL `view_storage_
        // primary` slot to the legacy shared buffer. In practice
        // ViewFold + ViewDecay kernels go through the Bindings
        // direct-construction path above (no `from_context_with_extras`)
        // so this rename is a defensive no-op for today's scoring /
        // physics generic kernels (whose view-storage bindings carry
        // per-view names already from the compose_view_storage_prelude
        // pass).
        let view_name_for_kernel = view_name_from_kernel_spec(spec);
        let mut extras_fields: Vec<String> = Vec::new();
        for b in &spec.bindings {
            let name = b.name.as_str();
            // ctx-routed (NOT in Extras): same name list as
            // classify_binding's special cases.
            if matches!(
                name,
                "event_ring" | "event_tail" | "voxel_grid"
                // Voxel-region-indices Phase 4b — navgrid +
                // navgrid_cfg ride the shared KernelBindingsContext
                // (same as voxel_grid), so they're not in the
                // per-kernel Extras struct.
                | "navgrid" | "navgrid_cfg"
            ) {
                continue;
            }
            if let Some(suffix) = name.strip_prefix("agent_") {
                if is_standard_agent_column(&format!("agent_{suffix}")) {
                    continue;
                }
            }
            if name.starts_with("ability_registry_") {
                continue;
            }
            // AliasOf bindings have no struct field.
            if matches!(b.bg_source, crate::kernel_binding_ir::BgSource::AliasOf(_)) {
                continue;
            }
            // Extras-bound. Render the runtime call-site expression:
            //   cfg     → &self.cfg_<kernel>_buf
            //   sim_cfg → self.event_ring.sim_cfg()
            //   else    → &self.<view-renamed name>_buf
            let buf_field_name = view_storage_per_view_name(name, view_name_for_kernel);
            let value_expr = if name == "cfg" {
                format!("&self.cfg_{}_buf", spec.name)
            } else if name == "sim_cfg" {
                "self.event_ring.sim_cfg()".to_string()
            } else {
                format!("&self.{buf_field_name}_buf")
            };
            extras_fields.push(format!(
                "                        {name}: {value_expr},"
            ));
        }
        let extras_body = extras_fields.join("\n");
        let dispatch_fn = format!("dispatch_{}", spec.name.to_lowercase());
        // Indirect-consumer or-pattern (see the parallel block above for
        // the rationale). Both `Kernel(...)` and `Indirect { kernel, .. }`
        // dispatch through the same `dispatch::dispatch_<name>` helper;
        // the consumer kernel's body bounds on `cfg.event_count` populated
        // immediately below from the live GPU event_tail value.
        //
        // FixedPoint kernels (`@cascade(max_iter=N)`) get their own arm
        // shape, wrapping the dispatch in a `for _iter in 0..N` loop.
        // See the `fixed_point_kernels` doc block above for MVP details.
        let is_indirect_consumer = indirect_consumer_kernel_names
            .iter()
            .any(|n| n == &spec.name);
        let fixed_point_max_iter = fixed_point_kernels.get(&spec.pascal).copied();
        let arm_pattern = if let Some(_) = fixed_point_max_iter {
            format!(
                "schedule::DispatchOp::FixedPoint {{ kernel: KernelId::{pascal}, .. }}",
                pascal = spec.pascal,
            )
        } else if is_indirect_consumer {
            format!(
                "schedule::DispatchOp::Kernel(KernelId::{pascal}) \
                 | schedule::DispatchOp::Indirect {{ kernel: KernelId::{pascal}, .. }}",
                pascal = spec.pascal,
            )
        } else {
            format!(
                "schedule::DispatchOp::Kernel(KernelId::{pascal})",
                pascal = spec.pascal,
            )
        };
        // For Indirect consumers, copy the live GPU event_tail into
        // cfg.event_count immediately before dispatch. See parallel
        // block above for rationale (and the fold-vs-indirect split:
        // folds get their snapshot at the TOP of step() since they
        // consume prior-tick records, not in-tick ones).
        //
        // FixedPoint kernels skip the cfg-copy (see the parallel block
        // above for why; same overlap-with-Indirect caveat applies).
        let cfg_copy_block = if is_indirect_consumer && fixed_point_max_iter.is_none() {
            format!(
                "                    encoder.copy_buffer_to_buffer(\n\
                 \x20                       self.event_ring.tail(),\n\
                 \x20                       0,\n\
                 \x20                       &self.cfg_{kname}_buf,\n\
                 \x20                       0,\n\
                 \x20                       4,\n\
                 \x20                   );\n",
                kname = spec.name,
            )
        } else {
            String::new()
        };
        let (loop_open, loop_close) = if let Some(n) = fixed_point_max_iter {
            (
                format!("                    for _iter in 0..{n}u32 {{\n"),
                "                    }\n".to_string(),
            )
        } else {
            (String::new(), String::new())
        };
        out.push_str(&format!(
            "                {arm_pattern} => {{\n\
             {cfg_copy_block}\
             {loop_open}\
             \x20                   let extras = {kname}::{pascal}Extras {{\n\
             {extras_body}\n\
             \x20                   }};\n\
             \x20                   let bindings = {kname}::{pascal}Bindings::from_context_with_extras(\n\
             \x20                       &_ctx, &extras,\n\
             \x20                   );\n\
             \x20                   dispatch::{dispatch_fn}(\n\
             \x20                       &mut self.cache,\n\
             \x20                       &bindings,\n\
             \x20                       &self.gpu.device,\n\
             \x20                       &mut encoder,\n\
             \x20                       self.agent_count,\n\
             \x20                   );\n\
             {loop_close}\
             \x20               }}\n",
            kname = spec.name,
            pascal = spec.pascal,
        ));
    }
    // Catch-all for unhandled DispatchOp variants. Status (2026-05-12):
    //
    //   * `DispatchOp::Kernel(...)` — handled by the per-kernel arms
    //     emitted in the loop above (every emitted kernel produces an
    //     arm).
    //   * `DispatchOp::Indirect { kernel: X, .. }` — handled when X is
    //     in `indirect_consumer_kernel_names`: each per-kernel arm
    //     becomes an or-pattern matching BOTH `Kernel(X)` and
    //     `Indirect { kernel: X, .. }` and runs the same dispatch
    //     body. The consumer kernel reads `cfg.event_count` (set to the
    //     host-side ring tail snapshot earlier in step()) to bound its
    //     event walk, so direct `dispatch_workgroups(agent_count, ...)`
    //     is correct (the threads past the event count early-return).
    //   * `DispatchOp::FixedPoint { kernel: X, .. }` — handled when X
    //     was parsed out of the synthesized schedule.rs into
    //     `fixed_point_kernels`. Each such kernel gets its own per-arm
    //     emission with the dispatch body wrapped in
    //     `for _iter in 0..N { ... }` where N is the authored
    //     `@cascade(max_iter=N)` value. MVP runs unconditionally to N;
    //     early-break on no-change is a follow-up (see the
    //     `fixed_point_kernels` doc block at the top of the loop).
    //   * `DispatchOp::GatedBy { kernel, gate }` — never emitted by
    //     today's `synthesize_schedule`; placeholder for future use.
    //
    // **In-step GPU producers limitation.** When chronicle events are
    // produced by IN-STEP GPU kernels (e.g. `apply_ability` dispatcher
    // calls `atomicAdd(&event_tail, 1u)`), the host-side `tail_value()`
    // estimate is NOT bumped — only `note_emits` / typed
    // `effect_*_applied` injectors update it. Such fixtures (e.g.
    // `hill_raid`'s `apply_ability` rules) get `event_count = 0` at
    // consumer dispatch and the consumer no-ops. The proper fix is a
    // true `dispatch_workgroups_indirect(args_buf, 0)` against the
    // SeedIndirectArgs output (gaps 1 + 2 from commit 353527e6's
    // doc block — kernel-emit indirect entry + schedule reorder of
    // SeedIndirect0 BEFORE its consumers). That wiring is orthogonal
    // to this slice and tracked separately.
    out.push_str(
        "                // DispatchOp::GatedBy is never emitted today;\n\
         \x20               // DispatchOp::Indirect for kernels not in\n\
         \x20               // indirect_consumer_kernel_names and any other\n\
         \x20               // unhandled variant fall through here. See\n\
         \x20               // `synthesize_generated_runtime_struct` source\n\
         \x20               // comment for status + remaining gaps (in-step\n\
         \x20               // GPU producers).\n\
         \x20               _ => {}\n\
         \x20           }\n\
         \x20       }\n\
         \x20\n\
         \x20       self.gpu.queue.submit(Some(encoder.finish()));\n\
         \x20       self.tick += 1;\n\
         \x20   }\n",
    );

    // Generic host-side chronicle injection helper. Provides a single
    // primitive every fixture can call to inject a synthetic chronicle
    // record into the unified event_ring without going through a
    // producer kernel. Mirrors the queue_clear_tail + append_chronicle_record
    // pattern that every per-fixture runtime crate previously hand-wrote.
    //
    // Steps:
    //   1. Write 0 to the GPU event_tail counter (synthetic record lands at slot 0).
    //   2. Call event_ring.reset_tail_estimate() to sync host-side tail.
    //   3. Append the record via the engine helper (which bumps the tail).
    //
    // The caller is responsible for setting record[0] = engine event kind id
    // (see dsl_ast::engine_events::engine_event_kind_id_for_name) and
    // record[1] = world tick. Slots [2..] carry per-event payload words
    // per the chronicle layout in dsl_compiler::cpu_chronicle_reference.
    //
    // Used by host-driven runtime APIs (verbs that inject chronicle
    // events from CPU outside the per-tick step encoder), e.g.
    // tom_probe_runtime's observe/scry/reveal/decoy/erase_belief/disguise
    // methods.
    out.push_str(
        "\n\
         \x20   /// Generic host-side chronicle injection helper.\n\
         \x20   /// Resets the event_ring tail and appends a single synthetic\n\
         \x20   /// chronicle record. Caller is responsible for the kind+tick+payload\n\
         \x20   /// layout of the 11-word record (slots 0..=9 + seq trailer at 10).\n\
         \x20   pub fn inject_chronicle_record(&mut self, record: &[u32; 11]) {\n\
         \x20       let zero = 0u32;\n\
         \x20       self.gpu.queue.write_buffer(self.event_ring.tail(), 0, bytemuck::bytes_of(&zero));\n\
         \x20       self.event_ring.reset_tail_estimate();\n\
         \x20       self.event_ring.append_chronicle_record(&self.gpu.queue, record);\n\
         \x20   }\n",
    );

    // Per-event typed host-side injectors for `@host_callable` events.
    // For every event in `comp.events` carrying the `@host_callable`
    // annotation, emit a typed Rust method that:
    //   1. Builds a 10-word chronicle record from typed args:
    //      slot 0 = kind id — engine alias via `EventIR.engine_kind_id`
    //               when present, else the fixture-allocated sequential
    //               `EventKindId(i)` where `i` is the event's position in
    //               `comp.events` (mirrors `cg::lower::driver::
    //               populate_event_kinds`'s allocation so the dispatcher's
    //               filter constant matches),
    //      slot 1 = self.tick as u32,
    //      slot 2..= = each declared field, packed in declaration order.
    //   2. Calls `self.inject_chronicle_record(&record)` to write it
    //      into the event_ring.
    //
    // Field type → param mapping:
    //   AgentId / U8 / U16 / U32 → `u32` (one slot, raw value)
    // Other field types are not supported in this slice; the emitter
    // bails with a `compile_error!` for the offending event so the
    // failure surfaces at build time rather than as a silent miscompile.
    //
    // Method name = snake_case(event.name). Arguments are listed in
    // declaration order with the source field names.
    //
    // The dispatch of the matching consumer kernel is NOT performed
    // by this method — that happens via the next `step()` call which
    // walks the schedule and dispatches every kernel (including the
    // PerEvent consumer over the ring). Callers that want immediate
    // synchronous dispatch (e.g. tom_probe's hand-written verbs that
    // pre-dated this codegen) layer their own dispatch code on top.
    for (event_index, ev) in events.iter().enumerate() {
        if !ev.annotations.iter().any(|a| a.name == "host_callable") {
            continue;
        }
        // Two-path kind id resolution, mirroring
        // `cg::lower::driver::populate_event_kinds`:
        //
        //   1. Engine-aliased events (e.g. `EffectDamageApplied = 26`)
        //      carry a hardcoded `engine_kind_id` populated by
        //      `dsl_ast::resolve` from `engine_events::
        //      engine_event_kind_id_for_name`. Use it directly so the
        //      injector's `record[0]` matches the dispatcher's filter
        //      constant.
        //   2. Fixture-defined events (no engine alias) get a sequential
        //      `EventKindId(i)` where `i` is the position in
        //      `comp.events` — same allocation the lowering driver
        //      performs. The dispatcher already handles arbitrary kinds
        //      via the unified event ring; the previous engine-aliased-
        //      only gate dropped fixture-defined `@host_callable` events
        //      with a `cargo:warning` (Gap plague_city#P-B).
        let kind_id = ev.engine_kind_id.unwrap_or(event_index as u32);
        let method_name = crate::snake_case(&ev.name);
        let mut params: Vec<String> = Vec::new();
        let mut slot_writes: Vec<String> = Vec::new();
        let mut bail: Option<String> = None;
        for (i, f) in ev.fields.iter().enumerate() {
            let slot = 2 + i;
            if slot >= 10 {
                bail = Some(format!(
                    "event `{}` has more than 8 fields; chronicle records only carry 10 u32 slots",
                    ev.name,
                ));
                break;
            }
            use dsl_ast::ir::IrType;
            let (rust_ty, write_expr): (&str, String) = match &f.ty {
                IrType::AgentId => ("u32", format!("{}", f.name)),
                IrType::U8 | IrType::U16 | IrType::U32 => ("u32", format!("{}", f.name)),
                other => {
                    bail = Some(format!(
                        "event `{}` field `{}` has type {:?}; @host_callable codegen only supports AgentId / U8 / U16 / U32 today",
                        ev.name, f.name, other,
                    ));
                    break;
                }
            };
            params.push(format!("{}: {}", f.name, rust_ty));
            slot_writes.push(format!(
                "        record[{slot}] = {write_expr};\n",
            ));
        }
        if let Some(msg) = bail {
            // Build-time hard error. Mirrors the panic-on-emit-failure
            // policy elsewhere in this helper.
            panic!("[{fixture_name} host_callable] {msg}");
        }
        let params_joined = if params.is_empty() {
            String::new()
        } else {
            format!(", {}", params.join(", "))
        };
        let kind_id_source = if ev.engine_kind_id.is_some() {
            "engine alias"
        } else {
            "fixture-allocated, matches `cg::lower::driver::populate_event_kinds`"
        };
        out.push_str(&format!(
            "\n\
             \x20   /// Auto-emitted from `@host_callable event {ev_name}` in the .sim source.\n\
             \x20   /// Builds a chronicle record with kind id {kind_id} ({kind_id_source}) and\n\
             \x20   /// the per-field payload in declaration order, then injects it via\n\
             \x20   /// [`Self::inject_chronicle_record`]. The matching consumer kernel\n\
             \x20   /// fires on the next [`Self::step`] call.\n\
             \x20   pub fn {method_name}(&mut self{params_joined}) {{\n\
             \x20       let mut record = [0u32; 11];\n\
             \x20       record[0] = {kind_id};\n\
             \x20       record[1] = self.tick as u32;\n\
             {writes}        self.inject_chronicle_record(&record);\n\
             \x20   }}\n",
            ev_name = ev.name,
            writes = slot_writes.concat(),
        ));
    }

    out.push_str("}\n");

    out
}

#[cfg(test)]
mod tests {
    //! Plan E-A3.2 structural verification — confirms the generated
    //! `runtime_core.rs` source has the expected shape (balanced
    //! braces, declared pub items present). The full compile gate
    //! lands when A5 pilot `include!`s the file from a real fixture
    //! crate; this test just catches obvious emit bugs without that
    //! integration cost.

    #[test]
    fn synthesize_runtime_core_minimal_fixture_emits_well_formed_struct() {
        let artifacts = crate::cg::emit::EmittedArtifacts::default();
        let out = super::synthesize_runtime_core_a2(
            "smoke_fixture",
            &artifacts,
            &[],
            &std::collections::BTreeMap::new(),
            &[],
            &[],
            None,
            &[],
            false,
            false, // binds_navgrid
            &[],
            0,
            0,
            false,
            None,
        );

        // Braces balance.
        let opens = out.matches('{').count();
        let closes = out.matches('}').count();
        assert_eq!(
            opens, closes,
            "brace mismatch in generated runtime_core: {opens} `{{` vs {closes} `}}`\n--- source ---\n{out}"
        );

        // Required public surface.
        for required in [
            "pub struct GeneratedRuntime",
            "pub gpu: engine::GpuContext",
            "pub agent_count: u32",
            "pub fn try_new(seed: u64, agent_count: u32) -> Option<Self>",
            "pub const FIXTURE_NAME: &str = \"smoke_fixture\";",
        ] {
            assert!(
                out.contains(required),
                "generated source missing required item {required:?}\n--- source ---\n{out}"
            );
        }

        // Empty fixture has no External bindings → no buffer alloc
        // lines in try_new (only the gpu init + Some(Self {{...}})).
        assert!(
            !out.contains("create_buffer"),
            "minimal fixture should not emit buffer alloc lines\n{out}"
        );
    }

    #[test]
    fn host_callable_event_emits_typed_injector_method() {
        use dsl_ast::ast::{Annotation, Span};
        use dsl_ast::ir::{EventField, EventIR, IrType};

        let span = Span::new(0, 0);
        // Mirrors `tom_probe.sim`'s `@host_callable event EffectObserveApplied`
        // — the canonical 3-field shape across the 6 ToM verbs. AgentId fields
        // become `u32` params; declaration order maps to slots 2..= (slot 0 is
        // the engine kind id 64, slot 1 is `self.tick as u32`).
        let event = EventIR {
            name: "EffectObserveApplied".into(),
            fields: vec![
                EventField { name: "actor".into(), ty: IrType::AgentId, span },
                EventField { name: "target".into(), ty: IrType::AgentId, span },
                EventField {
                    name: "target_observer".into(),
                    ty: IrType::U32,
                    span,
                },
            ],
            tags: Vec::new(),
            annotations: vec![Annotation {
                name: "host_callable".into(),
                args: Vec::new(),
                span,
            }],
            span,
            engine_kind_id: Some(64),
        };

        let artifacts = crate::cg::emit::EmittedArtifacts::default();
        let out = super::synthesize_runtime_core_a2(
            "host_callable_smoke",
            &artifacts,
            &[],
            &std::collections::BTreeMap::new(),
            &[],
            &[event],
            None,
            &[],
            false,
            false, // binds_navgrid
            &[],
            0,
            0,
            false,
            None,
        );

        // Typed signature with snake_case method name + matching params.
        assert!(
            out.contains(
                "pub fn effect_observe_applied(&mut self, actor: u32, target: u32, target_observer: u32)"
            ),
            "expected typed `effect_observe_applied` method signature\n--- source ---\n{out}"
        );
        // Engine kind id at slot 0.
        assert!(
            out.contains("record[0] = 64;"),
            "expected `record[0] = 64;` for engine kind id\n--- source ---\n{out}"
        );
        // Tick at slot 1.
        assert!(
            out.contains("record[1] = self.tick as u32;"),
            "expected tick stamp at slot 1\n--- source ---\n{out}"
        );
        // Per-field slot writes 2..=4 in declaration order.
        for (slot, field) in [(2, "actor"), (3, "target"), (4, "target_observer")] {
            let expected = format!("record[{slot}] = {field};");
            assert!(
                out.contains(&expected),
                "expected `{expected}`\n--- source ---\n{out}"
            );
        }
        // Body forwards to the generic injector helper.
        assert!(
            out.contains("self.inject_chronicle_record(&record);"),
            "expected forward to inject_chronicle_record\n--- source ---\n{out}"
        );
    }

    #[test]
    fn event_without_host_callable_annotation_emits_no_injector_method() {
        use dsl_ast::ast::Span;
        use dsl_ast::ir::{EventField, EventIR, IrType};

        let span = Span::new(0, 0);
        let event = EventIR {
            name: "BeliefAcquired".into(),
            fields: vec![EventField {
                name: "fact_bit".into(),
                ty: IrType::U32,
                span,
            }],
            tags: Vec::new(),
            annotations: Vec::new(),
            span,
            engine_kind_id: None,
        };

        let artifacts = crate::cg::emit::EmittedArtifacts::default();
        let out = super::synthesize_runtime_core_a2(
            "no_host_callable",
            &artifacts,
            &[],
            &std::collections::BTreeMap::new(),
            &[],
            &[event],
            None,
            &[],
            false,
            false, // binds_navgrid
            &[],
            0,
            0,
            false,
            None,
        );

        // The generic helper still lands.
        assert!(out.contains("pub fn inject_chronicle_record"));
        // But no typed wrapper for BeliefAcquired (no @host_callable annotation).
        assert!(
            !out.contains("pub fn belief_acquired"),
            "should not auto-emit method without @host_callable annotation\n--- source ---\n{out}"
        );
    }
}
