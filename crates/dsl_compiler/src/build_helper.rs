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

/// Same as [`emit`], but lets the caller pin a non-default
/// [`ScheduleStrategy`]. Quest-arc and village-day-cycle fixtures
/// historically used `Conservative` to disable kernel fusion the
/// fixture wasn't compatible with.
pub fn emit_with_strategy(
    fixture_name: &str,
    strategy: crate::cg::schedule::ScheduleStrategy,
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
    let comp = dsl_ast::resolve::resolve(program)
        .unwrap_or_else(|e| panic!("resolve {fixture_name}.sim: {e}"));
    let cg = match crate::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[{fixture_name} lower diag] {d}");
            }
            o.program
        }
    };
    let schedule_result = crate::cg::schedule::synthesize_schedule(&cg, strategy);
    let artifacts =
        crate::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
            .unwrap_or_else(|e| panic!("emit {fixture_name} CG program: {e:?}"));

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

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
    let runtime_core = synthesize_runtime_core_a2(fixture_name, &artifacts);
    fs::write(out_dir.join("runtime_core.rs"), runtime_core)
        .unwrap_or_else(|e| panic!("write runtime_core.rs: {e}"));
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
    out.push_str(&synthesize_generated_runtime_struct(fixture_name, artifacts));

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
    matches!(
        binding_name,
        "sim_cfg" | "event_ring" | "event_tail" | "cfg" | "snapshot_kick"
    )
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

/// Number of slots in the buffer for a given binding. Heuristic:
/// `agent_count` for the common per-agent case; `agent_count *
/// agent_count` for per-(observer, source) bindings detected by the
/// `_flags` suffix (today: only `beliefs_flags`).
fn slot_count_expr(binding_name: &str) -> &'static str {
    if binding_name.ends_with("_flags") {
        // Per-(observer, source) cell. TODO: replace heuristic with
        // proper binding-shape annotation in the AST.
        "(agent_count as u64) * (agent_count as u64)"
    } else {
        "agent_count as u64"
    }
}

fn synthesize_generated_runtime_struct(
    fixture_name: &str,
    artifacts: &crate::cg::emit::EmittedArtifacts,
) -> String {
    use crate::kernel_binding_ir::BgSource;
    use std::collections::BTreeSet;

    // Collect unique fixture-owned bindings across all kernels.
    // BTreeSet preserves deterministic iteration order.
    let mut owned: BTreeSet<(String, String)> = BTreeSet::new(); // (name, wgsl_ty)
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
            }
            if !matches!(b.bg_source, BgSource::External(_)) {
                continue;
            }
            if is_standard_agent_column(&b.name) || is_infra_binding(&b.name) {
                continue;
            }
            owned.insert((b.name.clone(), b.wgsl_ty.clone()));
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
         #[allow(dead_code, clippy::all)]\n\
         pub struct GeneratedRuntime {\n\
         \x20   pub gpu: engine::GpuContext,\n\
         \x20   pub agent_count: u32,\n\
         \x20   pub seed: u64,\n\
         \x20   pub tick: u64,\n",
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
    out.push_str("}\n\n");

    out.push_str(
        "#[allow(dead_code, clippy::all)]\n\
         impl GeneratedRuntime {\n\
         \x20   pub fn try_new(seed: u64, agent_count: u32) -> Option<Self> {\n\
         \x20       let gpu = engine::GpuContext::new_blocking().ok()?;\n",
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
    out.push_str("        Some(Self {\n");
    out.push_str("            gpu,\n");
    out.push_str("            agent_count,\n");
    out.push_str("            seed,\n");
    out.push_str("            tick: 0,\n");
    for (name, _) in &owned {
        out.push_str(&format!("            {name}_buf,\n"));
    }
    for kernel_name in &cfg_buffer_names {
        out.push_str(&format!("            cfg_{kernel_name}_buf,\n"));
    }
    out.push_str("        })\n");
    out.push_str("    }\n");
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
        let out = super::synthesize_runtime_core_a2("smoke_fixture", &artifacts);

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
