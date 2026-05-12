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

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = crate::parse(&src)
        .unwrap_or_else(|e| panic!("parse {fixture_name}.sim: {e:?}"));
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
    // outlier fixture's build.rs sets explicitly.
    let aoe_dispatch = built_registry
        .as_ref()
        .map(|br| {
            let n = br.registry.len();
            // AbilityId is a NonZero* newtype starting at 1; iterate the
            // 1..=n range and skip ids the registry rejects (defensive —
            // build_registry is contiguous today).
            (1..=n).any(|i| {
                engine::ability::AbilityId::new(i as u32)
                    .and_then(|id| br.registry.get(id))
                    .map(|p| p.per_effect_areas.iter().any(|a| a.is_some()))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
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
    let lower_opts = crate::cg::lower::LowerOpts {
        aoe_dispatch,
        belief_state,
        debug: lower_debug_depth,
        debug_wgsl: lower_debug_wgsl,
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
            o.program
        }
    };
    let schedule_result = crate::cg::schedule::synthesize_schedule_with_registry(
        &cg,
        strategy,
        built_registry.as_ref().map(|r| &r.registry),
    );
    let artifacts = crate::cg::emit::emit_cg_program_with_debug(
        &schedule_result.schedule,
        &cg,
        lower_debug_depth,
    )
    .unwrap_or_else(|e| panic!("emit {fixture_name} CG program: {e:?}"));

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
    let runtime_core = synthesize_runtime_core_a2(
        fixture_name,
        &artifacts,
        &init_stmts,
        &runtime_config_defaults,
        &ability_file_names,
    );
    fs::write(out_dir.join("runtime_core.rs"), runtime_core)
        .unwrap_or_else(|e| panic!("write runtime_core.rs: {e}"));
}

/// Per-kernel `@runtime` config field. Carried from the resolved
/// Compilation IR through synthesize_runtime_core so try_new can
/// initialize each cfg buffer's runtime field with its .sim default
/// and the generator can emit a host-side setter.
#[derive(Clone)]
struct RuntimeConfigDefault {
    scalar_ty: String,
    default_lit: String,
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
fn synthesize_runtime_core_a2(
    fixture_name: &str,
    artifacts: &crate::cg::emit::EmittedArtifacts,
    init_stmts: &[dsl_ast::ast::InitStmt],
    runtime_config_defaults: &std::collections::BTreeMap<String, RuntimeConfigDefault>,
    ability_file_names: &[String],
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
    // Sizing today: `agent_count * elem_bytes`. Per-(observer, source)
    // bindings (e.g. beliefs_flags = N*N u32) are heuristically
    // detected by the binding-name suffix `_flags` — the only such
    // shape today. A real binding-shape annotation in the AST is the
    // proper long-term fix; for now the heuristic + a TODO comment
    // keeps the generator working.
    out.push_str(&synthesize_generated_runtime_struct(
        fixture_name,
        artifacts,
        init_stmts,
        runtime_config_defaults,
        ability_file_names,
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

/// Number of slots in the buffer for a given binding. Heuristic:
/// `agent_count` for the common per-agent case; `agent_count *
/// agent_count` for per-(observer, subject) BeliefState SoA columns
/// (the 6 columns enumerated in [`is_belief_state_pair_column`]).
fn slot_count_expr(binding_name: &str) -> &'static str {
    if is_belief_state_pair_column(binding_name) {
        // Per-(observer, subject) cell — `pair_map` storage shape per
        // the BeliefState column contract. TODO: replace name-list
        // heuristic with proper binding-shape annotation in the AST.
        "(agent_count as u64) * (agent_count as u64)"
    } else {
        "agent_count as u64"
    }
}

fn synthesize_generated_runtime_struct(
    fixture_name: &str,
    artifacts: &crate::cg::emit::EmittedArtifacts,
    init_stmts: &[dsl_ast::ast::InitStmt],
    runtime_config_defaults: &std::collections::BTreeMap<String, RuntimeConfigDefault>,
    ability_file_names: &[String],
) -> String {
    use crate::kernel_binding_ir::BgSource;
    use std::collections::BTreeMap;

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
    for spec in &artifacts.kernel_specs {
        let mut has_cfg = false;
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
            // Standard agent columns ARE allocated as fixture-owned
            // today (no shared SimState yet). The AgentBuffers
            // population in step() routes them into ctx.state for the
            // from_context_with_extras call.
            // BTreeMap insert is "first-wins" via or_insert.
            owned.entry(b.name.clone()).or_insert_with(|| b.wgsl_ty.clone());
        }
        if has_cfg {
            cfg_buffer_names.push(spec.name.clone());
        }
    }

    let mut out = String::new();
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
        let slot_expr = slot_count_expr(name);
        // Plan E-A6 — if `init { ... }` declared a per-slot fill for
        // this `agent_<col>` binding, switch from zero-init create_buffer
        // to create_buffer_init with the computed slice. This is how
        // fixture-owned init state lives in the .sim source instead of
        // a hand-written *_runtime/lib.rs.
        let init_match = name.strip_prefix("agent_").and_then(|col| {
            init_stmts.iter().find(|s| s.field == col)
        });
        if let Some(stmt) = init_match {
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
            let init_vec_expr = match &stmt.expr {
                dsl_ast::ast::InitExpr::Const(n) => {
                    format!("vec![{n}u32; agent_count as usize]")
                }
                dsl_ast::ast::InitExpr::Slot => {
                    "(0..agent_count).collect::<Vec<u32>>()".to_string()
                }
            };
            out.push_str(&format!(
                "        let {name}_init: Vec<u32> = {init_vec_expr};\n\
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
         \x20   pub fn step(&mut self) {\n\
         \x20       // Per-tick cfg uniform write to every kernel's cfg buffer.\n\
         \x20       // Layout: [slot0, tick, seed, slot3] where slot0 is\n\
         \x20       // agent_cap (per-agent kernels) or event_count (per-event\n\
         \x20       // kernels). Today we write agent_count to both slots — for\n\
         \x20       // an empty event_ring this over-bounds harmlessly because\n\
         \x20       // the kernel's event-kind check on each row falls through.\n\
         \x20       // Cfg layout per kernel:\n\
         \x20       //   per-agent: { agent_cap, tick, seed, _pad }\n\
         \x20       //   ViewFold:  { event_count, tick, second_key_pop, _pad }\n\
         \x20       // Slot 2 is `seed` (per-agent — most kernels ignore it,\n\
         \x20       // they key PCG off tick+agent+purpose) or `second_key_pop`\n\
         \x20       // (ViewFold — must be 1 for single-key views, otherwise\n\
         \x20       // the per-(observer, source) index calc divides by 0 or\n\
         \x20       // wraps and the fold writes to wrong slots). Writing 1\n\
         \x20       // is correct for ViewFolds and harmless for per-agent.\n\
         \x20       // A later slice can detect cfg shape per kernel and\n\
         \x20       // write the proper seed for per-agent kernels — pinned\n\
         \x20       // to the cooldown_probe staggered-fire test as a regression.\n\
         \x20       let cfg_words: [u32; 4] = [\n\
         \x20           self.agent_count,\n\
         \x20           self.tick as u32,\n\
         \x20           1u32,\n\
         \x20           self.agent_count,\n\
         \x20       ];\n\
         \x20       let cfg_bytes: &[u8] = bytemuck::cast_slice(&cfg_words);\n",
    );
    for kernel_name in &cfg_buffer_names {
        out.push_str(&format!(
            "        self.gpu.queue.write_buffer(&self.cfg_{kernel_name}_buf, 0, cfg_bytes);\n",
        ));
    }
    out.push_str(
        "\n\
         \x20       let mut encoder = self.gpu.device.create_command_encoder(\n\
         \x20           &wgpu::CommandEncoderDescriptor {\n\
         \x20               label: Some(concat!(env!(\"CARGO_PKG_NAME\"), \"::step\")),\n\
         \x20           },\n\
         \x20       );\n\
         \x20       self.event_ring.clear_tail_in(&mut encoder);\n\
         \x20\n\
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
         \x20       };\n\
         \x20       let _ctx = engine::gpu::KernelBindingsContext {\n\
         \x20           state: &agent_buffers,\n\
         \x20           event_ring: &self.event_ring,\n\
         \x20           registry: &self.registry_gpu,\n\
         \x20           voxel_grid: None,\n\
         \x20       };\n\
         \x20\n\
         \x20       for op in schedule::SCHEDULE {\n\
         \x20           match op {\n",
    );
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
            let mut binding_fields: Vec<String> = Vec::new();
            for b in &spec.bindings {
                use crate::kernel_binding_ir::BgSource;
                if matches!(b.bg_source, BgSource::AliasOf(_)) {
                    continue;
                }
                let name = b.name.as_str();
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
                    format!("&self.{name}_buf")
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
            out.push_str(&format!(
                "                schedule::DispatchOp::Kernel(KernelId::{pascal}) => {{\n\
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
                 \x20               }}\n",
                kname = spec.name,
                pascal = spec.pascal,
            ));
            continue;
        }
        // Walk bindings, mirror classify_binding name rules.
        let mut extras_fields: Vec<String> = Vec::new();
        for b in &spec.bindings {
            let name = b.name.as_str();
            // ctx-routed (NOT in Extras): same name list as
            // classify_binding's special cases.
            if matches!(name, "event_ring" | "event_tail" | "voxel_grid") {
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
            //   else    → &self.<name>_buf
            let value_expr = if name == "cfg" {
                format!("&self.cfg_{}_buf", spec.name)
            } else if name == "sim_cfg" {
                "self.event_ring.sim_cfg()".to_string()
            } else {
                format!("&self.{name}_buf")
            };
            extras_fields.push(format!(
                "                        {name}: {value_expr},"
            ));
        }
        let extras_body = extras_fields.join("\n");
        let dispatch_fn = format!("dispatch_{}", spec.name.to_lowercase());
        out.push_str(&format!(
            "                schedule::DispatchOp::Kernel(KernelId::{pascal}) => {{\n\
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
             \x20               }}\n",
            kname = spec.name,
            pascal = spec.pascal,
        ));
    }
    out.push_str(
        "                _ => {}\n\
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
         \x20   /// layout of the 10-word record.\n\
         \x20   pub fn inject_chronicle_record(&mut self, record: &[u32; 10]) {\n\
         \x20       let zero = 0u32;\n\
         \x20       self.gpu.queue.write_buffer(self.event_ring.tail(), 0, bytemuck::bytes_of(&zero));\n\
         \x20       self.event_ring.reset_tail_estimate();\n\
         \x20       self.event_ring.append_chronicle_record(&self.gpu.queue, record);\n\
         \x20   }\n",
    );

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
}
