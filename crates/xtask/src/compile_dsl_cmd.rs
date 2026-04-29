//! `cargo run --bin xtask -- compile-dsl` — walk `assets/sim/*.sim`, parse
//! and resolve into a single `Compilation`, then emit Rust + Python +
//! schema-hash artefacts. Either writes the files (default) or compares them
//! against the committed output (`--check`, CI guard mode).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use dsl_compiler::ast::{Decl, Program};
use dsl_compiler::ir::Compilation;

use crate::cli::CompileDslArgs;

pub fn run_compile_dsl(args: CompileDslArgs) -> ExitCode {
    let sim_files = match discover_sim_files(&args.src) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }
    };
    if sim_files.is_empty() {
        eprintln!(
            "compile-dsl: no .sim files found under {}",
            args.src.display()
        );
        return ExitCode::FAILURE;
    }

    let CompileAll { combined, sources } = match compile_all(&sim_files) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let artefacts = dsl_compiler::emit_with_per_kind_sources(
        &combined,
        dsl_compiler::EmissionSources {
            events: sources.events.as_deref(),
            physics: sources.physics.as_deref(),
            masks: sources.masks.as_deref(),
            scoring: sources.scoring.as_deref(),
            entities: sources.entities.as_deref(),
            configs: sources.configs.as_deref(),
            enums: sources.enums.as_deref(),
            views: sources.views.as_deref(),
        },
    );

    let rust_events_dir = args.out_rust.join("events");
    let rust_schema = args.out_rust.join("schema.rs");
    let py_events_dir = args.out_python.join("events");
    let py_enums_dir = args.out_python.join("enums");
    let physics_dir = args.out_physics.clone();
    let mask_dir = args.out_mask.clone();
    let scoring_dir = args.out_scoring.clone();
    let entity_dir = args.out_entity.clone();
    let config_rust_dir = args.out_config_rust.clone();
    let config_toml_dir = args.out_config_toml.clone();
    let config_toml_path = config_toml_dir.join("default.toml");
    let enum_dir = args.out_enum.clone();
    let views_dir = args.out_views.clone();
    let step_file = args.out_step.clone();
    let backend_file = args.out_backend.clone();
    let mask_fill_file = args.out_mask_fill.clone();
    let cascade_reg_file = args.out_cascade_reg.clone();

    if args.check {
        let mut mismatches = Vec::new();
        check_scaffolded_kinds(
            &artefacts,
            &mask_dir,
            &scoring_dir,
            &entity_dir,
            &mut mismatches,
        );
        check_config(
            &artefacts,
            &config_rust_dir,
            &config_toml_path,
            &mut mismatches,
        );
        check_enums(&artefacts, &enum_dir, &py_enums_dir, &mut mismatches);
        check_views(&artefacts, &views_dir, &mut mismatches);

        // Rust per-event files. Pre-format the in-memory emission so the
        // comparison ignores rustfmt-driven layout differences (committed
        // files were rustfmt-formatted on the previous `compile-dsl` run).
        for (name, content) in &artefacts.rust_event_structs {
            let formatted = rustfmt_string(content).unwrap_or_else(|_| content.clone());
            check_file(&rust_events_dir.join(name), &formatted, &mut mismatches);
        }
        let mod_fmt = rustfmt_string(&artefacts.rust_events_mod).unwrap_or_else(|_| artefacts.rust_events_mod.clone());
        check_file(
            &rust_events_dir.join("mod.rs"),
            &mod_fmt,
            &mut mismatches,
        );
        let schema_fmt = rustfmt_string(&artefacts.schema_rs).unwrap_or_else(|_| artefacts.schema_rs.clone());
        check_file(&rust_schema, &schema_fmt, &mut mismatches);

        // Physics per-rule files + aggregator.
        // Use both-fmt comparison because the committed files may have hand-corrected
        // import ordering that is not rustfmt-stable; normalise both sides.
        for (name, content) in &artefacts.rust_physics_modules {
            let formatted = rustfmt_string(content).unwrap_or_else(|_| content.clone());
            check_file_both_fmt(&physics_dir.join(name), &formatted, &mut mismatches);
        }
        let physics_mod_fmt = rustfmt_string(&artefacts.rust_physics_mod).unwrap_or_else(|_| artefacts.rust_physics_mod.clone());
        check_file_both_fmt(
            &physics_dir.join("mod.rs"),
            &physics_mod_fmt,
            &mut mismatches,
        );

        // Python per-event files.
        for (name, content) in &artefacts.python_event_modules {
            check_file(&py_events_dir.join(name), content, &mut mismatches);
        }
        check_file(
            &py_events_dir.join("__init__.py"),
            &artefacts.python_events_init,
            &mut mismatches,
        );

        // engine-side EventLike impl.
        // Use both-fmt comparison because the committed file may not be rustfmt-stable
        // (hand-corrected import ordering); normalise both sides.
        check_file_both_fmt(
            &args.out_engine_event_like_impl,
            &artefacts.engine_event_like_impl,
            &mut mismatches,
        );

        // engine_rules single-file outputs (step, backend, mask_fill, cascade_reg).
        // These are static (almost entirely DSL-independent) but compiler-owned so
        // that future DSL-driven phases can grow into them.
        let step_content = dsl_compiler::emit_step::emit_step(sources.physics.as_deref());
        check_file_both_fmt(&step_file, &step_content, &mut mismatches);

        let backend_content = dsl_compiler::emit_backend::emit_backend(sources.physics.as_deref());
        check_file_both_fmt(&backend_file, &backend_content, &mut mismatches);

        let mask_fill_content = dsl_compiler::emit_mask_fill::emit_mask_fill(
            &combined.masks,
            sources.masks.as_deref(),
        );
        check_file_both_fmt(&mask_fill_file, &mask_fill_content, &mut mismatches);

        let cascade_reg_content =
            dsl_compiler::emit_cascade_register::emit_cascade_register(sources.physics.as_deref());
        check_file_both_fmt(&cascade_reg_file, &cascade_reg_content, &mut mismatches);

        // Stale file detection: committed Rust files not in the new emission.
        check_stale(&rust_events_dir, &artefacts.rust_event_structs, "rs", &mut mismatches);
        check_stale(&physics_dir, &artefacts.rust_physics_modules, "rs", &mut mismatches);
        check_stale(&py_events_dir, &artefacts.python_event_modules, "py", &mut mismatches);

        if mismatches.is_empty() {
            println!(
                "compile-dsl: check ok ({} events, {} physics, {} configs)",
                combined.events.len(),
                combined.physics.len(),
                combined.configs.len()
            );
            ExitCode::SUCCESS
        } else {
            eprintln!("compile-dsl: check FAILED ({} mismatch(es))", mismatches.len());
            for m in &mismatches {
                eprintln!("  - {m}");
            }
            eprintln!();
            eprintln!("run `cargo run --bin xtask -- compile-dsl` to regenerate");
            ExitCode::FAILURE
        }
    } else {
        if let Err(e) = write_artefacts(
            &rust_events_dir,
            &rust_schema,
            &physics_dir,
            &py_events_dir,
            &artefacts,
        ) {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }
        if let Err(e) = write_scaffolded_kinds(
            &mask_dir,
            &scoring_dir,
            &entity_dir,
            &artefacts,
        ) {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }
        if let Err(e) = write_config_output(
            &config_rust_dir,
            &config_toml_path,
            &artefacts,
        ) {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }
        if let Err(e) = write_enum_output(&enum_dir, &py_enums_dir, &artefacts) {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }
        if let Err(e) = write_views_output(&views_dir, &artefacts) {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }

        // Write the engine_rules single-file outputs (step, backend, mask_fill, cascade_reg).
        if let Err(e) = write_engine_rules_singles(
            &step_file,
            &backend_file,
            &mask_fill_file,
            &cascade_reg_file,
            &sources,
            &combined,
        ) {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }

        // Per-kernel emit accumulators. Each per-kernel block below pushes
        // the fields / schedule rows it needs; xtask flushes them once at
        // the end into the container emitters (transient_handles,
        // external_buffers, schedule).
        let mut transient_fields: Vec<dsl_compiler::emit_transient_handles::TransientField> =
            Vec::new();
        let mut external_fields: Vec<dsl_compiler::emit_external_buffers::ExternalField> =
            Vec::new();
        let mut resident_fields: Vec<dsl_compiler::emit_resident_context::ResidentField> =
            Vec::new();
        // PingPong A/B ring fields — seeded by per-kernel blocks that
        // emit cascade-physics events (apply_actions @ T7, physics @ T9).
        let mut pingpong_fields: Vec<dsl_compiler::emit_pingpong_context::PingPongField> =
            Vec::new();
        // Pool fields — Pooled-lifetime scratch buffers reused across
        // kernels within a tick. Seeded by T12 (spatial hash + per-query
        // results); future tasks add more entries (alive_pack scratch
        // etc.).
        let mut pool_fields: Vec<dsl_compiler::emit_pool::PoolField> =
            Vec::new();
        // Per-view resident-field names in `scoring_view_binding_order`
        // emit order — consumed by emit_resident_context to generate
        // `scoring_view_buffers_slice()`.
        let mut scoring_view_field_names: Vec<String> = Vec::new();
        let mut schedule_entries: Vec<dsl_compiler::emit_schedule::ScheduleEntry> = Vec::new();

        // Emit engine_gpu_rules/src/lib.rs from the per-kernel module list.
        // Each per-kernel emit block below pushes its module name; the list
        // is sorted before emission so diffs after `.sim` edits stay readable.
        let mut modules: Vec<String> = vec![
            "alive_pack".into(),
            "append_events".into(),
            "apply_actions".into(),
            "fused_agent_unpack".into(),
            "fused_mask".into(),
            "mask_unpack".into(),
            "megakernel".into(),
            "movement".into(),
            "physics".into(),
            "pick_ability".into(),
            "scoring".into(),
            "scoring_unpack".into(),
            "seed_indirect".into(),
            "spatial_engagement_query".into(),
            "spatial_hash".into(),
            "spatial_kin_query".into(),
        ];
        // T11: one `fold_<name>` module per materialized view (skip lazy).
        // Walked from `combined.views` so the module list reflects the IR
        // 1:1; sorted below for stable diffs.
        for v in &combined.views {
            if matches!(v.body, dsl_compiler::ir::ViewBodyIR::Expr(_)) {
                continue;
            }
            modules.push(format!("fold_{}", v.name));
        }
        modules.sort();
        {
            let lib_rs = dsl_compiler::emit_kernel_index::emit_lib_rs(&modules);
            let path = PathBuf::from("crates/engine_gpu_rules/src/lib.rs");
            if let Some(parent) = path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    eprintln!("compile-dsl: mkdir engine_gpu_rules/src: {e}");
                    return ExitCode::FAILURE;
                }
            }
            if let Err(e) = fs::write(&path, lib_rs) {
                eprintln!("compile-dsl: write engine_gpu_rules/src/lib.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // ----- Per-kernel emit block: FusedMaskKernel + MaskUnpackKernel (Task 4).
        {
            use dsl_compiler::emit_external_buffers::ExternalField;
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};
            use dsl_compiler::emit_transient_handles::TransientField;

            let body = dsl_compiler::emit_mask_kernel::emit_fused_mask_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/fused_mask.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write fused_mask.rs: {e}");
                return ExitCode::FAILURE;
            }

            // Filter comp.masks to those the WGSL emitter can lift, then
            // produce the fused-kernel WGSL from that subset. Masks that
            // can't lift (e.g. ability-binding shapes) are silently dropped
            // here — the runtime path retains its CPU fallback for them.
            let emittable_masks: Vec<dsl_compiler::ir::MaskIR> = combined
                .masks
                .iter()
                .filter(|m| dsl_compiler::emit_mask_wgsl::emit_mask_wgsl(m).is_ok())
                .cloned()
                .collect();
            let fused_wgsl_body = if emittable_masks.is_empty() {
                // No emittable masks: emit a stub that compiles but does
                // nothing. Real wiring lands in Task 5.
                "@compute @workgroup_size(64) fn cs_fused_masks(@builtin(global_invocation_id) gid: vec3<u32>) {}\n".to_string()
            } else {
                match dsl_compiler::emit_mask_wgsl::emit_masks_wgsl_fused(&emittable_masks) {
                    Ok(module) => module.wgsl,
                    Err(e) => {
                        eprintln!(
                            "compile-dsl: emit_masks_wgsl_fused failed: {e}; falling back to stub"
                        );
                        "@compute @workgroup_size(64) fn cs_fused_masks(@builtin(global_invocation_id) gid: vec3<u32>) {}\n".to_string()
                    }
                }
            };
            let wgsl = format!(
                "// GENERATED by dsl_compiler::emit_mask_wgsl. Do not edit by hand.\n{}",
                fused_wgsl_body,
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/fused_mask.wgsl"),
                wgsl,
            ) {
                eprintln!("compile-dsl: write fused_mask.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            let body2 = dsl_compiler::emit_mask_kernel::emit_mask_unpack_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/mask_unpack.rs"),
                body2,
            ) {
                eprintln!("compile-dsl: write mask_unpack.rs: {e}");
                return ExitCode::FAILURE;
            }
            // Stub mask-unpack WGSL — Task 5 step 4 hoists the real shader
            // body matching today's engine_gpu::mask::MaskUnpackKernel.
            let unpack_wgsl = "// GENERATED by dsl_compiler. Do not edit by hand.\n@compute @workgroup_size(64) fn cs_mask_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {}\n";
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/mask_unpack.wgsl"),
                unpack_wgsl,
            ) {
                eprintln!("compile-dsl: write mask_unpack.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // Bring the mask kernels' buffers into the binding-source
            // containers. xtask de-duplicates by name when later
            // per-kernel blocks request the same external field.
            transient_fields.push(TransientField {
                name: "mask_bitmaps".into(),
                doc: "FusedMaskKernel output: ceil(N/32) words × N masks; recycled per tick.".into(),
            });
            transient_fields.push(TransientField {
                name: "mask_unpack_agents_input".into(),
                doc: "MaskUnpackKernel scratch: source SoA before unpack.".into(),
            });
            if !external_fields.iter().any(|f| f.name == "agents") {
                external_fields.push(ExternalField {
                    name: "agents".into(),
                    doc: "Agent SoA buffer (engine-owned).".into(),
                });
            }
            if !external_fields.iter().any(|f| f.name == "sim_cfg") {
                external_fields.push(ExternalField {
                    name: "sim_cfg".into(),
                    doc: "SimCfg uniform/storage buffer (engine-owned).".into(),
                });
            }

            // SCHEDULE entries — FusedMask produces mask_bitmaps;
            // MaskUnpack produces agents_soa from mask_unpack_agents_input.
            // No diamond writes, no cross-edge to declare yet (T5+ wiring
            // grows the read/write graph as more kernels land).
            schedule_entries.push(ScheduleEntry {
                kernel: "FusedMask".into(),
                kind: DispatchOpKind::Kernel,
            });
            schedule_entries.push(ScheduleEntry {
                kernel: "MaskUnpack".into(),
                kind: DispatchOpKind::Kernel,
            });
        }

        // ----- Per-kernel emit block: SpatialHash + Spatial{Kin,Engagement}Query (Task 12).
        //
        // First Pooled-lifetime usage in `BindingSources`: the three
        // spatial scratch buffers (`spatial_grid_cells`,
        // `spatial_grid_offsets`, `spatial_query_results`) live in
        // `Pool` and are reused across the three spatial dispatches
        // within a single tick.
        //
        // The hand-written spatial pipeline in `engine_gpu::spatial_gpu`
        // uses one fused WGSL module with multiple entry points and a
        // 12-binding layout; the emitted kernels here carve out compact
        // 4-/5-binding BGLs and write a no-op WGSL stub matching the
        // emitted slot order. T16 unifies the WGSL.
        //
        // Schedule entries are pushed BEFORE Scoring (the next emit
        // block) so the spatial output is available when the scoring
        // kernel runs.
        {
            use dsl_compiler::emit_pool::PoolField;
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};

            // Emit Rust kernel wrappers.
            let hash_rs = dsl_compiler::emit_spatial_kernel::emit_spatial_hash_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/spatial_hash.rs"),
                hash_rs,
            ) {
                eprintln!("compile-dsl: write spatial_hash.rs: {e}");
                return ExitCode::FAILURE;
            }
            let kin_rs = dsl_compiler::emit_spatial_kernel::emit_kin_query_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/spatial_kin_query.rs"),
                kin_rs,
            ) {
                eprintln!("compile-dsl: write spatial_kin_query.rs: {e}");
                return ExitCode::FAILURE;
            }
            let eng_rs = dsl_compiler::emit_spatial_kernel::emit_engagement_query_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/spatial_engagement_query.rs"),
                eng_rs,
            ) {
                eprintln!("compile-dsl: write spatial_engagement_query.rs: {e}");
                return ExitCode::FAILURE;
            }

            // No-op WGSL stubs matching the emitted BGL slot-for-slot.
            // The hand-written spatial WGSL in `engine_gpu::spatial_gpu`
            // uses a fused 12-binding layout incompatible with the
            // emitted compact BGL; T16 unifies them. Until then, the
            // stubs touch every binding once so naga keeps the
            // declarations live, and the emitted dispatches stay gated
            // off behind `engine_gpu_emitted_spatial_dispatch`.
            let hash_wgsl = "// GENERATED by dsl_compiler. Do not edit by hand.\n\
@group(0) @binding(0) var<storage, read> agents: array<u32>;\n\
@group(0) @binding(1) var<storage, read_write> grid_cells: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> grid_offsets: array<u32>;\n\
@group(0) @binding(3) var<storage, read> sim_cfg: array<u32>;\n\
struct SpatialHashCfg { agent_cap: u32, radius_q: f32, _pad0: u32, _pad1: u32 };\n\
@group(0) @binding(4) var<uniform> cfg: SpatialHashCfg;\n\
@compute @workgroup_size(64)\n\
fn cs_spatial_hash(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    // Stub body: touch every binding so naga keeps them live.\n\
    let _a = agents[0];\n\
    let _gc = grid_cells[0];\n\
    let _go = grid_offsets[0];\n\
    let _sc = sim_cfg[0];\n\
    let _c = cfg.agent_cap;\n\
}\n";
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/spatial_hash.wgsl"),
                hash_wgsl,
            ) {
                eprintln!("compile-dsl: write spatial_hash.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            let kin_wgsl = "// GENERATED by dsl_compiler. Do not edit by hand.\n\
@group(0) @binding(0) var<storage, read> agents: array<u32>;\n\
@group(0) @binding(1) var<storage, read> grid_cells: array<u32>;\n\
@group(0) @binding(2) var<storage, read> grid_offsets: array<u32>;\n\
@group(0) @binding(3) var<storage, read_write> query_results: array<u32>;\n\
@group(0) @binding(4) var<storage, read> sim_cfg: array<u32>;\n\
struct SpatialKinQueryCfg { agent_cap: u32, radius_q: f32, _pad0: u32, _pad1: u32 };\n\
@group(0) @binding(5) var<uniform> cfg: SpatialKinQueryCfg;\n\
@compute @workgroup_size(64)\n\
fn cs_spatial_kin_query(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    // Stub body: touch every binding so naga keeps them live.\n\
    let _a = agents[0];\n\
    let _gc = grid_cells[0];\n\
    let _go = grid_offsets[0];\n\
    let _qr = query_results[0];\n\
    let _sc = sim_cfg[0];\n\
    let _c = cfg.agent_cap;\n\
}\n";
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/spatial_kin_query.wgsl"),
                kin_wgsl,
            ) {
                eprintln!("compile-dsl: write spatial_kin_query.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            let eng_wgsl = "// GENERATED by dsl_compiler. Do not edit by hand.\n\
@group(0) @binding(0) var<storage, read> agents: array<u32>;\n\
@group(0) @binding(1) var<storage, read> grid_cells: array<u32>;\n\
@group(0) @binding(2) var<storage, read> grid_offsets: array<u32>;\n\
@group(0) @binding(3) var<storage, read_write> query_results: array<u32>;\n\
@group(0) @binding(4) var<storage, read> sim_cfg: array<u32>;\n\
struct SpatialEngagementQueryCfg { agent_cap: u32, radius_q: f32, _pad0: u32, _pad1: u32 };\n\
@group(0) @binding(5) var<uniform> cfg: SpatialEngagementQueryCfg;\n\
@compute @workgroup_size(64)\n\
fn cs_spatial_engagement_query(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    // Stub body: touch every binding so naga keeps them live.\n\
    let _a = agents[0];\n\
    let _gc = grid_cells[0];\n\
    let _go = grid_offsets[0];\n\
    let _qr = query_results[0];\n\
    let _sc = sim_cfg[0];\n\
    let _c = cfg.agent_cap;\n\
}\n";
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/spatial_engagement_query.wgsl"),
                eng_wgsl,
            ) {
                eprintln!("compile-dsl: write spatial_engagement_query.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // Pool fields — first Pooled-lifetime entries.
            for (n, doc) in &[
                ("spatial_grid_cells",     "Spatial-hash cell-index buffer (Pooled)."),
                ("spatial_grid_offsets",   "Spatial-hash cell-offsets buffer (Pooled)."),
                ("spatial_query_results",  "Per-query result buffer (Pooled)."),
            ] {
                if !pool_fields.iter().any(|f: &PoolField| f.name == *n) {
                    pool_fields.push(PoolField {
                        name: (*n).into(),
                        doc:  (*doc).into(),
                    });
                }
            }

            // Schedule rows — placed BEFORE Scoring (which the next
            // emit block pushes). Order: hash → kin → engagement.
            schedule_entries.push(ScheduleEntry {
                kernel: "SpatialHash".into(),
                kind:   DispatchOpKind::Kernel,
            });
            schedule_entries.push(ScheduleEntry {
                kernel: "SpatialKinQuery".into(),
                kind:   DispatchOpKind::Kernel,
            });
            schedule_entries.push(ScheduleEntry {
                kernel: "SpatialEngagementQuery".into(),
                kind:   DispatchOpKind::Kernel,
            });
        }

        // ----- Per-kernel emit block: AlivePack + FusedAgentUnpack (Task 13).
        //
        // The two final hand-written kernels still owned by engine_gpu:
        // `alive_bitmap::AlivePackKernel` packs `agents[i].alive` into a
        // bitmap that mask/scoring/physics read every tick;
        // `mask::FusedAgentUnpackKernel` is the merged top-of-tick
        // dispatch that writes mask SoA + scoring AoS in one pass.
        // Both run FIRST in every tick — `FusedAgentUnpack` writes the
        // SoA fields the rest of the schedule consumes, and
        // `AlivePack` derives its bitmap from `agents[i].alive` after
        // the unpack is settled.
        //
        // The hand-written WGSL lives behind engine_gpu's `gpu` feature
        // (`engine_gpu::alive_bitmap::ALIVE_PACK_WGSL` +
        // `engine_gpu::mask::FUSED_AGENT_UNPACK_WGSL`, both promoted to
        // `pub const`) — pulling that into xtask requires the gpu
        // feature, which transitively requires wgpu/naga and a working
        // engine_gpu lib. To keep xtask cheap, the emitted .wgsl files
        // here are no-op stubs matching the emitted BGL slot-for-slot
        // (same pattern as T11/T12). The hand-written kernel calls in
        // engine_gpu's `step_batch` keep running by default; the
        // emitted dispatches are gated behind
        // `engine_gpu_emitted_alive_pack_dispatch` /
        // `engine_gpu_emitted_fused_unpack_dispatch` (both off until
        // T16 hoists the real bodies into the emit path).
        {
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};
            use dsl_compiler::emit_resident_context::ResidentField;
            use dsl_compiler::emit_transient_handles::TransientField;

            let alive_rs = dsl_compiler::emit_spatial_kernel::emit_alive_pack_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/alive_pack.rs"),
                alive_rs,
            ) {
                eprintln!("compile-dsl: write alive_pack.rs: {e}");
                return ExitCode::FAILURE;
            }
            // Real WGSL body — recovered from pre-T16 hand-written
            // ALIVE_PACK_WGSL and adapted to the post-T16 raw-u32 SoA
            // binding layout. See dsl_compiler::emit_alive_pack_wgsl.
            let alive_wgsl = format!(
                "// GENERATED by dsl_compiler::emit_alive_pack_wgsl. Do not edit by hand.\n{}",
                dsl_compiler::emit_alive_pack_wgsl::emit_alive_pack_wgsl()
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/alive_pack.wgsl"),
                alive_wgsl,
            ) {
                eprintln!("compile-dsl: write alive_pack.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            let fused_rs = dsl_compiler::emit_spatial_kernel::emit_fused_agent_unpack_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/fused_agent_unpack.rs"),
                fused_rs,
            ) {
                eprintln!("compile-dsl: write fused_agent_unpack.rs: {e}");
                return ExitCode::FAILURE;
            }
            // Stub WGSL — touches every binding once so naga keeps the
            // declarations live. T16 hoists the real FUSED_AGENT_UNPACK_WGSL.
            let fused_wgsl = "// GENERATED by dsl_compiler. Do not edit by hand.\n\
@group(0) @binding(0) var<storage, read> agents_input: array<u32>;\n\
@group(0) @binding(1) var<storage, read_write> mask_soa: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> agent_data: array<u32>;\n\
struct FusedAgentUnpackCfg { agent_cap: u32, radius_q: f32, _pad0: u32, _pad1: u32 };\n\
@group(0) @binding(3) var<uniform> cfg: FusedAgentUnpackCfg;\n\
@compute @workgroup_size(64)\n\
fn cs_fused_agent_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    // Stub body: touch every binding so naga keeps them live.\n\
    let _ai = agents_input[0];\n\
    let _ms = mask_soa[0];\n\
    let _ad = agent_data[0];\n\
    let _c = cfg.agent_cap;\n\
}\n";
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/fused_agent_unpack.wgsl"),
                fused_wgsl,
            ) {
                eprintln!("compile-dsl: write fused_agent_unpack.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // Resident: alive_bitmap target buffer (write-only here, read
            // by mask/scoring/physics later in the tick).
            if !resident_fields.iter().any(|f: &ResidentField| f.name == "alive_bitmap") {
                resident_fields.push(ResidentField {
                    name: "alive_bitmap".into(),
                    doc:  "Per-agent alive bitmap (Resident; ceil(N/32) words).".into(),
                });
            }
            // Transient: FusedAgentUnpack source + derived mask SoA.
            transient_fields.push(TransientField {
                name: "fused_agent_unpack_input".into(),
                doc:  "FusedAgentUnpackKernel scratch: source pre-unpack agent buffer.".into(),
            });
            transient_fields.push(TransientField {
                name: "fused_agent_unpack_mask_soa".into(),
                doc:  "FusedAgentUnpackKernel scratch: derived mask SoA.".into(),
            });

            // Schedule entries — these run FIRST in every tick.
            // FusedAgentUnpack at slot 0, AlivePack at slot 1.
            schedule_entries.insert(0, ScheduleEntry {
                kernel: "FusedAgentUnpack".into(),
                kind:   DispatchOpKind::Kernel,
            });
            schedule_entries.insert(1, ScheduleEntry {
                kernel: "AlivePack".into(),
                kind:   DispatchOpKind::Kernel,
            });
        }

        // ----- Per-kernel emit block: ScoringKernel + ScoringUnpackKernel (Task 6).
        {
            use dsl_compiler::emit_resident_context::ResidentField;
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};
            use dsl_compiler::emit_scoring_kernel::{
                emit_scoring_rs, emit_scoring_unpack_rs, ViewSpecForEmit,
            };
            use dsl_compiler::emit_transient_handles::TransientField;
            use dsl_compiler::emit_view_wgsl::{classify_view, ViewShape};

            // Build ViewSpecForEmit list from the IR. Mirrors the
            // hand-written engine_gpu::view_storage::build_all_specs:
            // skip Lazy views; classify the rest into the three shape
            // strings the emitter discriminates on.
            //
            // Note `classify_view` can fail for views the WGSL emitter
            // doesn't yet lift (e.g. symmetric_pair_topk dedicated
            // shapes); those are silently dropped here, matching the
            // mask emitter's approach. The hand-written ScoringKernel
            // also walks build_all_specs and skips Lazy via
            // scoring_view_binding_order, so the Vec ends up identical
            // up to ordering (we sort below).
            let mut view_specs: Vec<ViewSpecForEmit> = Vec::new();
            for v in &combined.views {
                let spec = match classify_view(v) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let (shape, topk) = match spec.shape {
                    ViewShape::Lazy => continue,
                    ViewShape::SlotMap { .. } => ("SlotMap".to_string(), false),
                    ViewShape::PairMapScalar => {
                        ("PairMapScalar".to_string(), spec.topk.is_some())
                    }
                    ViewShape::PairMapDecay { .. } => {
                        ("PairMapDecay".to_string(), spec.topk.is_some())
                    }
                };
                view_specs.push(ViewSpecForEmit {
                    name: spec.snake.clone(),
                    shape,
                    topk,
                });
            }
            // Match `scoring_view_binding_order` ordering — sort by name.
            view_specs.sort_by(|a, b| a.name.cmp(&b.name));

            // Emit Rust kernel module.
            let scoring_rs = emit_scoring_rs(&view_specs);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/scoring.rs"),
                scoring_rs,
            ) {
                eprintln!("compile-dsl: write scoring.rs: {e}");
                return ExitCode::FAILURE;
            }

            // Emit WGSL — wraps the existing emit_scoring_wgsl_atomic_views
            // emitter (which already produces a full kernel body). Pass
            // the same view specs so binding indices align with the
            // generated Rust BGL.
            let mut wgsl_specs: Vec<dsl_compiler::emit_view_wgsl::ViewStorageSpec> = Vec::new();
            for v in &combined.views {
                if let Ok(s) = classify_view(v) {
                    if !matches!(s.shape, ViewShape::Lazy) {
                        wgsl_specs.push(s);
                    }
                }
            }
            wgsl_specs.sort_by(|a, b| a.view_name.cmp(&b.view_name));
            let scoring_wgsl_body =
                dsl_compiler::emit_scoring_wgsl::emit_scoring_wgsl_atomic_views(&wgsl_specs);
            let scoring_wgsl = format!(
                "// GENERATED by dsl_compiler::emit_scoring_wgsl. Do not edit by hand.\n{scoring_wgsl_body}"
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/scoring.wgsl"),
                scoring_wgsl,
            ) {
                eprintln!("compile-dsl: write scoring.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // Emit Rust unpack kernel module.
            let scoring_unpack_rs = emit_scoring_unpack_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/scoring_unpack.rs"),
                scoring_unpack_rs,
            ) {
                eprintln!("compile-dsl: write scoring_unpack.rs: {e}");
                return ExitCode::FAILURE;
            }
            // Stub unpack WGSL — T7 hoists the real body.
            let unpack_wgsl =
                "// GENERATED by dsl_compiler. Do not edit by hand.\n@compute @workgroup_size(64) fn cs_scoring_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {}\n";
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/scoring_unpack.wgsl"),
                unpack_wgsl,
            ) {
                eprintln!("compile-dsl: write scoring_unpack.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // Resident fields — scoring_table + per-view storage.
            resident_fields.push(ResidentField {
                name: "scoring_table".into(),
                doc:  "Resident scoring table (per-action priors).".into(),
            });
            for spec in &view_specs {
                let field_name = format!("view_storage_{}", spec.name);
                resident_fields.push(ResidentField {
                    name: field_name.clone(),
                    doc:  format!("Resident view storage for `{}`.", spec.name),
                });
                scoring_view_field_names.push(field_name);
            }

            // Transient fields — scoring_out + scoring_unpack scratch.
            transient_fields.push(TransientField {
                name: "action_buf".into(),
                doc:  "ScoringKernel output (action-per-agent buffer).".into(),
            });
            transient_fields.push(TransientField {
                name: "scoring_unpack_agents_input".into(),
                doc:  "ScoringUnpackKernel scratch.".into(),
            });

            // Schedule rows.
            schedule_entries.push(ScheduleEntry {
                kernel: "Scoring".into(),
                kind:   DispatchOpKind::Kernel,
            });
            schedule_entries.push(ScheduleEntry {
                kernel: "ScoringUnpack".into(),
                kind:   DispatchOpKind::Kernel,
            });
        }

        // ----- Per-kernel emit block: ApplyActionsKernel (Task 7).
        //
        // Walks the apply-row IR for the kernel's buffer needs and
        // emits a Rust BGL + WGSL stub matching slot-for-slot. The
        // WGSL body here is a no-op compute shader that satisfies the
        // emitted BGL — the hand-written `engine_gpu::apply_actions`
        // kernel keeps doing the real damage/heal/event work until
        // T16 retires it. The engine_gpu wire-up is feature-gated
        // behind `engine_gpu_emitted_apply_actions_dispatch` (off by
        // default) for the same reason.
        {
            use dsl_compiler::emit_pingpong_context::PingPongField;
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};

            let body = dsl_compiler::emit_scoring_kernel::emit_apply_actions_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/apply_actions.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write apply_actions.rs: {e}");
                return ExitCode::FAILURE;
            }
            // No-op WGSL stub matching the emitted BGL slot order
            // (agents @ 0, scoring_out @ 1, event_ring_records @ 2,
            // event_ring_tail @ 3, sim_cfg @ 4, cfg @ 5). Each binding
            // is referenced once with a no-op read so naga keeps the
            // declarations live; T16 hoists the real `cs_apply_actions`
            // body in, replacing the hand-written
            // `engine_gpu::apply_actions::build_shader` source.
            //
            // Why a stub here rather than reusing
            // `engine_gpu::apply_actions::build_shader`'s output: that
            // builder produces a WGSL with cfg @ 2 / sim_cfg @ 5
            // (matching the hand-written Rust BGL). The emitted Rust
            // BGL puts cfg @ 5 / sim_cfg @ 4. Lifting the hand-written
            // body verbatim would mismatch the emitted BGL and fail
            // pipeline creation. The stub matches the emitted BGL, so
            // the pair stays internally consistent; the engine_gpu
            // wire-up is feature-gated (off by default) so the
            // hand-written dispatch keeps running until the WGSL
            // emit catches up in T16.
            // Real WGSL body — recovered from pre-T16 apply_actions WGSL,
            // adapted to the post-T16 raw-u32 binding layout, includes
            // runtime prelude for event emission.
            let apply_wgsl = format!(
                "// GENERATED by dsl_compiler::emit_apply_actions_wgsl. Do not edit by hand.\n{}",
                dsl_compiler::emit_apply_actions_wgsl::emit_apply_actions_wgsl()
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/apply_actions.wgsl"),
                apply_wgsl,
            ) {
                eprintln!("compile-dsl: write apply_actions.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // PingPong A/B ring buffers — apply_actions writes to
            // `events_a_*` (seed iter=0); the cascade FixedPoint loop
            // (T10) swaps A/B for subsequent iters.
            if !pingpong_fields.iter().any(|f: &PingPongField| f.name == "events_a_records") {
                pingpong_fields.push(PingPongField {
                    name: "events_a_records".into(),
                    doc:  "Cascade-physics A-ring event records (write side at iter 0).".into(),
                });
                pingpong_fields.push(PingPongField {
                    name: "events_a_tail".into(),
                    doc:  "Cascade-physics A-ring tail (atomic counter).".into(),
                });
                pingpong_fields.push(PingPongField {
                    name: "events_b_records".into(),
                    doc:  "Cascade-physics B-ring event records.".into(),
                });
                pingpong_fields.push(PingPongField {
                    name: "events_b_tail".into(),
                    doc:  "Cascade-physics B-ring tail.".into(),
                });
            }

            schedule_entries.push(ScheduleEntry {
                kernel: "ApplyActions".into(),
                kind:   DispatchOpKind::Kernel,
            });
        }

        // ----- Per-kernel emit block: PickAbilityKernel (Task 8).
        //
        // The per_ability row body emit (kernel WGSL) already landed in
        // commits `d8e196e8` (`emit_pick_ability_wgsl`) and `8f8e3582`
        // (schema-hash coverage). This block adds the wrapper Rust
        // struct emit on top — the dispatch-emit equivalent of T7's
        // ApplyActionsKernel but reusing existing IR emit work.
        //
        // BGL slot order matches `emit_pick_ability_wgsl` slot-for-slot
        // (see emit_pick_ability_kernel.rs module docs); the per_ability
        // WGSL emitter is authoritative since it shipped in T-A2.
        {
            use dsl_compiler::emit_external_buffers::ExternalField;
            use dsl_compiler::emit_resident_context::ResidentField;
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};

            let body = dsl_compiler::emit_pick_ability_kernel::emit_pick_ability_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/pick_ability.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write pick_ability.rs: {e}");
                return ExitCode::FAILURE;
            }

            // The WGSL emitter for per_ability rows takes a single
            // ScoringIR. If multiple per_ability rows are present, the
            // emitter merges them (existing behaviour landed in commit
            // `d8e196e8`). When no per_ability rows are present, emit a
            // minimal entry-point stub so the kernel module loads even
            // when unused.
            let per_ability_owners: Vec<&dsl_compiler::ir::ScoringIR> = combined
                .scoring
                .iter()
                .filter(|s| !s.per_ability_rows.is_empty())
                .collect();
            let wgsl_body = if let Some(s) = per_ability_owners.first() {
                dsl_compiler::emit_scoring_wgsl::emit_pick_ability_wgsl(s)
            } else {
                "@compute @workgroup_size(64) fn cs_pick_ability(@builtin(global_invocation_id) gid: vec3<u32>) {}\n".to_string()
            };
            let pick_wgsl = format!(
                "// GENERATED by dsl_compiler. Do not edit by hand.\n{wgsl_body}"
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/pick_ability.wgsl"),
                pick_wgsl,
            ) {
                eprintln!("compile-dsl: write pick_ability.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // Resident fields — per_slot_cooldown + chosen_ability_buf.
            // Both persist across ticks (cooldowns tick down each tick;
            // chosen_ability_buf is read by ApplyActions next tick).
            if !resident_fields.iter().any(|f| f.name == "per_slot_cooldown") {
                resident_fields.push(ResidentField {
                    name: "per_slot_cooldown".into(),
                    doc:  "Per-agent per-slot cooldown counters (Resident; persists across ticks).".into(),
                });
            }
            if !resident_fields.iter().any(|f| f.name == "chosen_ability_buf") {
                resident_fields.push(ResidentField {
                    name: "chosen_ability_buf".into(),
                    doc:  "PickAbilityKernel output (Resident; consumed by ApplyActions next tick).".into(),
                });
            }

            // External fields — engine-owned ability_registry +
            // tag_values tables.
            if !external_fields.iter().any(|f| f.name == "ability_registry") {
                external_fields.push(ExternalField {
                    name: "ability_registry".into(),
                    doc:  "AbilityRegistry buffer (engine-owned).".into(),
                });
            }
            if !external_fields.iter().any(|f| f.name == "tag_values") {
                external_fields.push(ExternalField {
                    name: "tag_values".into(),
                    doc:  "Per-tag value table (engine-owned).".into(),
                });
            }

            schedule_entries.push(ScheduleEntry {
                kernel: "PickAbility".into(),
                kind:   DispatchOpKind::Kernel,
            });
        }

        // ----- Per-kernel emit block: MovementKernel (Task 9).
        //
        // Movement is target-bound: reads scoring's chosen action +
        // per-agent target from `transient.action_buf`, advances the
        // agent SoA position by `move_speed_mps × tick_dt`, and emits
        // `AgentMoved` events into the cascade A-ring (seeded
        // alongside ApplyActions). Hand-written reference at
        // `engine_gpu/src/movement.rs:137-770` keeps running until T16.
        //
        // The emitted Rust BGL diverges from the hand-written BGL: cfg
        // sits at slot 5 (after sim_cfg @ 4); the hand-written kernel
        // keeps cfg at slot 2 (sim_cfg @ 5). The current emitted WGSL
        // is a no-op stub matching the emitted BGL slot-for-slot — T16
        // hoists the real MoveToward / Flee body. The engine_gpu wire-up
        // is feature-gated behind `engine_gpu_emitted_movement_dispatch`
        // (off by default) for the same reason.
        //
        // The DispatchOp::FixedPoint variant lit up in T1 is unused
        // here — Movement itself is a single-dispatch Kernel entry; T10
        // appends the FixedPoint entry pointing at Physics.
        {
            use dsl_compiler::emit_pingpong_context::PingPongField;
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};

            let body = dsl_compiler::emit_movement_kernel::emit_movement_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/movement.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write movement.rs: {e}");
                return ExitCode::FAILURE;
            }
            // No-op WGSL stub matching the emitted BGL slot order
            // (agents @ 0, scoring @ 1, event_ring_records @ 2,
            // event_ring_tail @ 3, sim_cfg @ 4, cfg @ 5). Each binding
            // is referenced once with a no-op read so naga keeps the
            // declarations live; T16 hoists the real `cs_movement` body
            // from `engine_gpu::movement::build_shader`.
            // Real WGSL body — recovered from pre-T16 movement WGSL,
            // adapted to the post-T16 raw-u32 binding layout, and
            // includes the runtime prelude for event emission.
            let movement_wgsl = format!(
                "// GENERATED by dsl_compiler::emit_movement_wgsl. Do not edit by hand.\n{}",
                dsl_compiler::emit_movement_wgsl::emit_movement_wgsl()
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/movement.wgsl"),
                movement_wgsl,
            ) {
                eprintln!("compile-dsl: write movement.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // Movement writes into the A-ring alongside ApplyActions
            // (see T7's pingpong_fields seed). De-dup by name in case
            // the apply_actions block already populated these.
            if !pingpong_fields.iter().any(|f: &PingPongField| f.name == "events_a_records") {
                pingpong_fields.push(PingPongField {
                    name: "events_a_records".into(),
                    doc:  "Cascade-physics A-ring event records (write side at iter 0).".into(),
                });
                pingpong_fields.push(PingPongField {
                    name: "events_a_tail".into(),
                    doc:  "Cascade-physics A-ring tail (atomic counter).".into(),
                });
                pingpong_fields.push(PingPongField {
                    name: "events_b_records".into(),
                    doc:  "Cascade-physics B-ring event records.".into(),
                });
                pingpong_fields.push(PingPongField {
                    name: "events_b_tail".into(),
                    doc:  "Cascade-physics B-ring tail.".into(),
                });
            }

            schedule_entries.push(ScheduleEntry {
                kernel: "Movement".into(),
                kind:   DispatchOpKind::Kernel,
            });
        }

        // ----- Per-kernel emit block: Physics + SeedIndirect + AppendEvents (Task 10).
        //
        // Cascade physics is the iterative kernel: each iteration drains
        // the producer event ring, produces follow-up events into the
        // next-iter ring, then `SeedIndirectKernel` writes the next
        // iteration's `dispatch_indirect` args. The Schedule expresses
        // this as `DispatchOp::FixedPoint { kernel: KernelId::Physics,
        // max_iter: 8 }` followed by `DispatchOp::Indirect { kernel:
        // KernelId::SeedIndirect, args_buf_ref: ResidentIndirectArgs }`.
        // `AppendEventsKernel` sits at the end of the cascade and
        // promotes per-iter events into the batch ring.
        //
        // The current emitted WGSL is a no-op stub for all three (the
        // hand-written kernels in `engine_gpu::physics` and
        // `engine_gpu::cascade_resident` have BGL slot orders that
        // don't match the emitted convention — sim_cfg vs cfg slot
        // positions, num_events buffer, etc.). T16 hoists the real
        // kernel bodies once the emitted BGL is authoritative. The
        // engine_gpu wire-ups for all three are feature-gated off by
        // default for the same reason.
        {
            use dsl_compiler::emit_external_buffers::ExternalField;
            use dsl_compiler::emit_resident_context::ResidentField;
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};
            use dsl_compiler::emit_transient_handles::TransientField;

            // Physics Rust + WGSL stub.
            let body = dsl_compiler::emit_movement_kernel::emit_physics_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/physics.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write physics.rs: {e}");
                return ExitCode::FAILURE;
            }
            // No-op WGSL stub matching the emitted physics BGL slot
            // order (10 slots: agents @ 0, current_event_ring @ 1,
            // current_event_tail @ 2, next_event_ring @ 3,
            // next_event_tail @ 4, gold @ 5, standing @ 6, memory @ 7,
            // sim_cfg @ 8, cfg @ 9). Each binding is touched once with
            // a no-op read so naga keeps the declarations live; T16
            // hoists the real `cs_physics` body in.
            let physics_wgsl = "// GENERATED by dsl_compiler. Do not edit by hand.\n\
@group(0) @binding(0) var<storage, read_write> agents: array<u32>;\n\
@group(0) @binding(1) var<storage, read> current_event_ring: array<u32>;\n\
@group(0) @binding(2) var<storage, read> current_event_tail: array<u32>;\n\
@group(0) @binding(3) var<storage, read_write> next_event_ring: array<u32>;\n\
@group(0) @binding(4) var<storage, read_write> next_event_tail: atomic<u32>;\n\
@group(0) @binding(5) var<storage, read_write> gold_buf: array<i32>;\n\
@group(0) @binding(6) var<storage, read_write> standing_storage: array<u32>;\n\
@group(0) @binding(7) var<storage, read_write> memory_storage: array<u32>;\n\
@group(0) @binding(8) var<storage, read> sim_cfg: array<u32>;\n\
struct PhysicsCfg { agent_cap: u32, iter_idx: u32, max_iter: u32, event_ring_capacity: u32 };\n\
@group(0) @binding(9) var<uniform> cfg: PhysicsCfg;\n\
@compute @workgroup_size(64)\n\
fn cs_physics(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    if (gid.x >= cfg.agent_cap) { return; }\n\
    // Stub body: touch every binding so naga keeps them live.\n\
    let _a = agents[0];\n\
    let _r = current_event_ring[0];\n\
    let _ct = current_event_tail[0];\n\
    let _nr = next_event_ring[0];\n\
    let _nt = atomicLoad(&next_event_tail);\n\
    let _g = gold_buf[0];\n\
    let _s = standing_storage[0];\n\
    let _m = memory_storage[0];\n\
    let _sc = sim_cfg[0];\n\
    let _c = cfg.iter_idx;\n\
}\n";
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/physics.wgsl"),
                physics_wgsl,
            ) {
                eprintln!("compile-dsl: write physics.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // SeedIndirect Rust + WGSL stub.
            let body = dsl_compiler::emit_movement_kernel::emit_seed_indirect_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/seed_indirect.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write seed_indirect.rs: {e}");
                return ExitCode::FAILURE;
            }
            // No-op WGSL stub matching the emitted seed_indirect BGL
            // (4 slots: apply_tail @ 0, indirect_args @ 1, sim_cfg @ 2,
            // cfg @ 3). The hand-written
            // `engine_gpu::cascade_resident::SEED_INDIRECT_WGSL` uses a
            // 5-slot BGL with a separate num_events buffer; T16 hoists
            // a body that matches the emitted 4-slot convention.
            // Real WGSL body — recovered from pre-T16 SEED_INDIRECT_WGSL,
            // adapted to the post-T16 raw-u32 binding layout.
            let seed_wgsl = format!(
                "// GENERATED by dsl_compiler::emit_seed_indirect_wgsl. Do not edit by hand.\n{}",
                dsl_compiler::emit_seed_indirect_wgsl::emit_seed_indirect_wgsl()
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/seed_indirect.wgsl"),
                seed_wgsl,
            ) {
                eprintln!("compile-dsl: write seed_indirect.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // AppendEvents Rust + WGSL stub.
            let body = dsl_compiler::emit_movement_kernel::emit_append_events_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/append_events.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write append_events.rs: {e}");
                return ExitCode::FAILURE;
            }
            // No-op WGSL stub matching the emitted append_events BGL
            // (5 slots: source_ring @ 0, source_tail @ 1,
            // batch_ring @ 2, batch_tail @ 3, cfg @ 4). The
            // hand-written `engine_gpu::cascade_resident::
            // APPEND_EVENTS_WGSL` uses the same shape but with
            // batch_tail before batch_ring; T16 reorders to match.
            // Real WGSL body — recovered from pre-T16 APPEND_EVENTS_WGSL,
            // adapted to the post-T16 raw-u32 binding layout.
            let append_wgsl = format!(
                "// GENERATED by dsl_compiler::emit_append_events_wgsl. Do not edit by hand.\n{}",
                dsl_compiler::emit_append_events_wgsl::emit_append_events_wgsl()
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/append_events.wgsl"),
                append_wgsl,
            ) {
                eprintln!("compile-dsl: write append_events.wgsl: {e}");
                return ExitCode::FAILURE;
            }

            // Resident: gold, standing, memory, batch events ring + tail.
            for (n, doc) in &[
                ("gold",                "Per-agent gold balance (Resident)."),
                ("standing_primary",    "Standing view storage primary buffer (Resident)."),
                ("memory_primary",      "Memory view storage primary buffer (Resident)."),
                ("batch_events_ring",   "Batch event ring records (consumed by view folds + post-batch readback)."),
                ("batch_events_tail",   "Batch event ring tail counter."),
            ] {
                if !resident_fields.iter().any(|f: &ResidentField| f.name == *n) {
                    resident_fields.push(ResidentField {
                        name: (*n).into(),
                        doc:  (*doc).into(),
                    });
                }
            }
            // Transient: per-iteration cascade ring refs (engine_gpu
            // swaps these each FixedPoint iter).
            for (n, doc) in &[
                ("cascade_current_ring",  "Cascade producer-ring records for the current iteration."),
                ("cascade_current_tail",  "Cascade producer-ring tail counter."),
                ("cascade_next_ring",     "Cascade consumer-ring records (next iteration)."),
                ("cascade_next_tail",     "Cascade consumer-ring tail counter (atomic)."),
                ("cascade_indirect_args", "dispatch_indirect args for the next iteration."),
            ] {
                if !transient_fields.iter().any(|f: &TransientField| f.name == *n) {
                    transient_fields.push(TransientField {
                        name: (*n).into(),
                        doc:  (*doc).into(),
                    });
                }
            }
            // External: ensure agents + sim_cfg present (already added
            // by earlier blocks but defensive in case ordering changes).
            if !external_fields.iter().any(|f: &ExternalField| f.name == "agents") {
                external_fields.push(ExternalField {
                    name: "agents".into(),
                    doc:  "Agent SoA buffer (engine-owned).".into(),
                });
            }
            if !external_fields.iter().any(|f: &ExternalField| f.name == "sim_cfg") {
                external_fields.push(ExternalField {
                    name: "sim_cfg".into(),
                    doc:  "SimCfg uniform/storage buffer (engine-owned).".into(),
                });
            }

            // Schedule rows: Physics is a FixedPoint loop, SeedIndirect
            // is dispatch_indirect-driven, AppendEvents is a single
            //-dispatch Kernel entry.
            schedule_entries.push(ScheduleEntry {
                kernel: "Physics".into(),
                kind:   DispatchOpKind::FixedPoint { max_iter: 8 },
            });
            schedule_entries.push(ScheduleEntry {
                kernel: "SeedIndirect".into(),
                kind:   DispatchOpKind::Indirect {
                    args_buf_ref: "ResidentIndirectArgs".into(),
                },
            });
            schedule_entries.push(ScheduleEntry {
                kernel: "AppendEvents".into(),
                kind:   DispatchOpKind::Kernel,
            });
        }

        // ----- Per-kernel emit block: per-view Fold<View>Kernel (Task 11).
        //
        // One emitter, eight outputs: walks `combined.views`, skips lazy
        // (`ViewBodyIR::Expr`), and writes a Rust + WGSL pair per
        // materialized view. The Rust side is the
        // `Fold<Pascal>Kernel` wrapper; the WGSL side is a no-op stub
        // matching the emitted BGL slot-for-slot. T16 hoists the real
        // fold body once the hand-written `engine_gpu::view_storage`
        // dispatch retires.
        //
        // The emitted kernel binds:
        //   slot 0/1: batch_events_ring + tail (post-cascade, ro)
        //   slot 2:   view_storage_primary (rw)
        //   slot 3:   view_storage_anchor   (rw, optional — falls back to primary when None)
        //   slot 4:   view_storage_ids      (rw, optional — falls back to primary when None)
        //   slot 5:   sim_cfg (ro storage)
        //   slot 6:   per-tick FoldCfg (uniform)
        //
        // The `fold_view_<name>_handles()` accessor on the resident
        // context returns the per-view (primary, anchor_opt, ids_opt)
        // triple — emitted alongside the resident-context fields by
        // `emit_resident_context_rs_with_scoring_and_folds`. See the
        // FoldViewSpec list collected below.
        let mut fold_view_specs: Vec<dsl_compiler::emit_resident_context::FoldViewSpec> =
            Vec::new();
        {
            use dsl_compiler::emit_resident_context::FoldViewSpec;
            use dsl_compiler::emit_schedule::{DispatchOpKind, ScheduleEntry};

            for v in &combined.views {
                // Skip lazy views — they have no fold kernel.
                if matches!(v.body, dsl_compiler::ir::ViewBodyIR::Expr(_)) {
                    continue;
                }
                let name = &v.name;
                // Pascal-case view name (engaged_with -> EngagedWith).
                let pascal: String = {
                    let mut out = String::new();
                    let mut up = true;
                    for c in name.chars() {
                        if c == '_' { up = true; continue; }
                        if up { out.extend(c.to_uppercase()); up = false; }
                        else { out.push(c); }
                    }
                    out
                };

                // Emit Rust kernel wrapper.
                let rs = dsl_compiler::emit_view_fold_kernel::emit_view_fold_rs(name);
                if let Err(e) = fs::write(
                    PathBuf::from(format!("crates/engine_gpu_rules/src/fold_{name}.rs")),
                    rs,
                ) {
                    eprintln!("compile-dsl: write fold_{name}.rs: {e}");
                    return ExitCode::FAILURE;
                }
                // Dispatch to the right WGSL emitter by storage hint.
                // Falls back to the no-op stub when the emitter rejects
                // the view (Lazy, decay, mismatched param count) — the
                // dispatch is gated off in lib.rs for those cases anyway.
                use dsl_compiler::ir::{StorageHint, ViewKind};
                use dsl_compiler::emit_view_wgsl::{
                    emit_per_entity_ring_fold_wgsl,
                    emit_symmetric_pair_topk_fold_wgsl,
                };
                // emit_view_fold_wgsl (PairMap/SlotMap path) is currently
                // prelude-dependent — it references undefined `view_agent_cap`
                // and other context fns. Stream B option δ: defer that
                // emitter until its prelude module exists. The two
                // self-contained shape-dedicated emitters
                // (SymmetricPairTopK, PerEntityRing) DO produce naga-valid
                // WGSL; wire those, stub the rest.
                let body_result: Result<String, String> = match v.kind {
                    ViewKind::Materialized(StorageHint::SymmetricPairTopK { .. }) => {
                        emit_symmetric_pair_topk_fold_wgsl(v).map_err(|e| format!("{e:?}"))
                    }
                    ViewKind::Materialized(StorageHint::PerEntityRing { .. }) => {
                        emit_per_entity_ring_fold_wgsl(v).map_err(|e| format!("{e:?}"))
                    }
                    _ => Err("PairMap/SlotMap fold emitter is prelude-dependent (deferred per Stream B δ)".to_string()),
                };
                let wgsl = match body_result {
                    Ok(real_body) => format!(
                        "// GENERATED by dsl_compiler::emit_view_wgsl. Do not edit by hand.\n{real_body}"
                    ),
                    Err(reason) => {
                        // Stub fallback — the view's emitter rejected it
                        // (or doesn't yet support its shape). The
                        // corresponding dispatch arm in lib.rs is gated
                        // off, so the stub is never instantiated; it
                        // just satisfies the BGL contract for the Rust
                        // wrapper.
                        eprintln!("compile-dsl: fold_{name}.wgsl falling back to stub ({reason})");
                        format!(
                            "// GENERATED by dsl_compiler. Do not edit by hand.\n\
// Stub fallback: {reason}\n\
@group(0) @binding(0) var<storage, read> event_ring: array<u32>;\n\
@group(0) @binding(1) var<storage, read> event_tail: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> view_storage_primary: array<u32>;\n\
@group(0) @binding(3) var<storage, read_write> view_storage_anchor: array<u32>;\n\
@group(0) @binding(4) var<storage, read_write> view_storage_ids: array<u32>;\n\
@group(0) @binding(5) var<storage, read> sim_cfg: array<u32>;\n\
struct Fold{pascal}Cfg {{ event_count: u32, tick: u32, _pad0: u32, _pad1: u32 }};\n\
@group(0) @binding(6) var<uniform> cfg: Fold{pascal}Cfg;\n\
@compute @workgroup_size(64)\n\
fn cs_fold_{name}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    // Stub body: touch every binding so naga keeps them live.\n\
    let _r = event_ring[0];\n\
    let _t = event_tail[0];\n\
    let _p = view_storage_primary[0];\n\
    let _a = view_storage_anchor[0];\n\
    let _i = view_storage_ids[0];\n\
    let _sc = sim_cfg[0];\n\
    let _c = cfg.tick;\n\
}}\n",
                        )
                    }
                };
                if let Err(e) = fs::write(
                    PathBuf::from(format!("crates/engine_gpu_rules/src/fold_{name}.wgsl")),
                    wgsl,
                ) {
                    eprintln!("compile-dsl: write fold_{name}.wgsl: {e}");
                    return ExitCode::FAILURE;
                }

                // Map the view name to its primary resident field. Five
                // pair-map / slot-map views land under
                // `view_storage_<name>` (emitted by the scoring block);
                // standing + memory have dedicated `*_primary` fields
                // emitted by the physics block. `engaged_with` (slot_map)
                // currently has no resident field — `classify_view`
                // rejects it because of the `Agent`/`AgentId` return-type
                // mismatch. Until that's resolved (engine work, separate
                // task), the fold kernel for `engaged_with` falls back
                // to using `standing_primary` as a safe placeholder so
                // the emitted code compiles — the dispatch is gated off
                // by default, so this never runs in production.
                let primary_field = match name.as_str() {
                    "standing" => "standing_primary".to_string(),
                    "memory"   => "memory_primary".to_string(),
                    "engaged_with" => "standing_primary".to_string(),
                    _ => format!("view_storage_{name}"),
                };
                fold_view_specs.push(FoldViewSpec {
                    name: name.clone(),
                    primary_field,
                    // Anchor + ids buffers are not yet broken out as
                    // separate resident fields — the existing emit only
                    // declares the primary buffer per view. Returning
                    // `None` here makes the fold kernel's `record()` body
                    // fall back to the primary buffer for slots 3 and 4
                    // (the BGL slot stays valid; the WGSL stub doesn't
                    // touch those storage views in any meaningful way).
                    anchor_field: None,
                    ids_field:    None,
                });

                // Schedule entry — `Fold<Pascal>` Kernel kind. Placed
                // after AppendEvents (post-cascade) because the fold
                // kernels read from `batch_events_ring`, which is only
                // populated after the cascade FixedPoint loop converges
                // and AppendEvents promotes per-iter events into the
                // batch ring.
                schedule_entries.push(ScheduleEntry {
                    kernel: format!("Fold{pascal}"),
                    kind:   DispatchOpKind::Kernel,
                });
            }
        }

        // Schedule — populated above by per-kernel emit blocks.
        {
            let schedule_rs = dsl_compiler::emit_schedule::emit_schedule_rs(&schedule_entries);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/schedule.rs"),
                &schedule_rs,
            ) {
                eprintln!("compile-dsl: write schedule.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Resident context — populated above by per-kernel emit blocks.
        // Fold view specs (T11) are emitted here too; they reference
        // resident fields by name so the per-view fold accessors compile
        // against whatever fields the earlier blocks declared.
        {
            let rc_rs = dsl_compiler::emit_resident_context::emit_resident_context_rs_with_scoring_and_folds(
                &resident_fields,
                &scoring_view_field_names,
                &fold_view_specs,
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/resident_context.rs"),
                &rc_rs,
            ) {
                eprintln!("compile-dsl: write resident_context.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Pingpong context — populated above by per-kernel emit blocks
        // (apply_actions @ T7 seeded the A/B event-ring fields).
        {
            let body =
                dsl_compiler::emit_pingpong_context::emit_pingpong_context_rs(&pingpong_fields);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/pingpong_context.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write pingpong_context.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Pool — populated by per-kernel emit blocks. T12 seeds the
        // first Pooled-lifetime fields (spatial hash + per-query
        // results); future tasks may add more.
        {
            let body = dsl_compiler::emit_pool::emit_pool_rs(&pool_fields);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/pool.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write pool.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Transient handles — populated above by per-kernel emit blocks.
        {
            let body = dsl_compiler::emit_transient_handles::emit_transient_handles_rs(
                &transient_fields,
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/transient_handles.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write transient_handles.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // External buffers — populated above by per-kernel emit blocks.
        {
            let body = dsl_compiler::emit_external_buffers::emit_external_buffers_rs(
                &external_fields,
            );
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/external_buffers.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write external_buffers.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // BindingSources<'a> — fixed shape (5 references); never re-emitted across regens.
        {
            let body = dsl_compiler::emit_binding_sources::emit_binding_sources_rs();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/binding_sources.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write binding_sources.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Megakernel — second emit pass that walks the SCHEDULE and
        // produces a single fused WGSL kernel scaffold (T14). Each
        // schedule entry becomes an inline section comment; the
        // bodies are intentional placeholders for the
        // gpu_megakernel_plan work in flight to fill in. T14 lands
        // the wiring only — the megakernel pipeline is compiled but
        // not dispatched yet (selector wiring is left to T15+).
        {
            let mk_rs =
                dsl_compiler::emit_megakernel::emit_megakernel_rs(&schedule_entries);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/megakernel.rs"),
                mk_rs,
            ) {
                eprintln!("compile-dsl: write megakernel.rs: {e}");
                return ExitCode::FAILURE;
            }
            let mk_wgsl =
                dsl_compiler::emit_megakernel::emit_megakernel_wgsl(&schedule_entries);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/megakernel.wgsl"),
                mk_wgsl,
            ) {
                eprintln!("compile-dsl: write megakernel.wgsl: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Schema hash baseline. Walk the engine_gpu_rules/src/ tree (after
        // the writes above) and write the SHA-256 hex digest to
        // crates/engine_gpu_rules/.schema_hash. The
        // engine_gpu_rules/tests/schema_hash.rs baseline test will fail
        // if the regen output diverges from the on-disk baseline.
        {
            let mut inputs: Vec<(String, Vec<u8>)> = Vec::new();
            for entry in walkdir::WalkDir::new("crates/engine_gpu_rules/src") {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("compile-dsl: walk engine_gpu_rules/src: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                if !entry.file_type().is_file() {
                    continue;
                }
                let p = entry.path();
                let ext = p.extension().and_then(|e| e.to_str());
                if !matches!(ext, Some("rs") | Some("wgsl")) {
                    continue;
                }
                let rel = match p.strip_prefix("crates/engine_gpu_rules/src") {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("compile-dsl: strip_prefix engine_gpu_rules/src: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                let bytes = match fs::read(p) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("compile-dsl: read {}: {e}", p.display());
                        return ExitCode::FAILURE;
                    }
                };
                inputs.push((rel.display().to_string(), bytes));
            }
            inputs.sort_by(|a, b| a.0.cmp(&b.0));
            let h = dsl_compiler::schema_hash::gpu_rules_hash(&inputs);
            let hex_str = h.iter().map(|b| format!("{b:02x}")).collect::<String>();
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/.schema_hash"),
                hex_str,
            ) {
                eprintln!("compile-dsl: write engine_gpu_rules/.schema_hash: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Write the engine-side `impl EventLike for Event` generated file.
        // Lives in engine (not engine_data) to avoid a dep cycle while
        // engine retains its engine_data regular dep (chronicle.rs, Plan B2).
        if let Some(parent) = args.out_engine_event_like_impl.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("compile-dsl: {e}");
                return ExitCode::FAILURE;
            }
        }
        if let Err(e) = fs::write(
            &args.out_engine_event_like_impl,
            &artefacts.engine_event_like_impl,
        ) {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }

        // Format emitted Rust so it matches the project's style. Best effort —
        // if rustfmt fails (missing toolchain, generated file has a bug we
        // want to see), surface the error.
        let mut rustfmt_targets: Vec<PathBuf> = artefacts
            .rust_event_structs
            .iter()
            .map(|(n, _)| rust_events_dir.join(n))
            .chain([rust_events_dir.join("mod.rs"), rust_schema.clone()])
            .collect();
        rustfmt_targets.extend(
            artefacts
                .rust_physics_modules
                .iter()
                .map(|(n, _)| physics_dir.join(n)),
        );
        rustfmt_targets.push(physics_dir.join("mod.rs"));
        rustfmt_targets.extend(
            artefacts
                .rust_config_modules
                .iter()
                .map(|(n, _)| config_rust_dir.join(n)),
        );
        rustfmt_targets.push(config_rust_dir.join("mod.rs"));
        rustfmt_targets.extend(
            artefacts
                .rust_enum_modules
                .iter()
                .map(|(n, _)| enum_dir.join(n)),
        );
        rustfmt_targets.push(enum_dir.join("mod.rs"));
        rustfmt_targets.extend(
            artefacts
                .rust_view_modules
                .iter()
                .map(|(n, _)| views_dir.join(n)),
        );
        rustfmt_targets.push(views_dir.join("mod.rs"));
        // Mask / scoring / entity modules were omitted from the rustfmt
        // pass prior to task 150, which left `--check` comparing raw
        // emitter output against formatted expected strings. `--check`
        // re-formats expected via `rustfmt_string`, so the write path
        // has to format on disk or the byte-comparison in `check_file`
        // always mismatches. Include them now so the two paths agree.
        rustfmt_targets.extend(
            artefacts
                .rust_mask_modules
                .iter()
                .map(|(n, _)| mask_dir.join(n)),
        );
        rustfmt_targets.push(mask_dir.join("mod.rs"));
        rustfmt_targets.extend(
            artefacts
                .rust_scoring_modules
                .iter()
                .map(|(n, _)| scoring_dir.join(n)),
        );
        rustfmt_targets.push(scoring_dir.join("mod.rs"));
        rustfmt_targets.extend(
            artefacts
                .rust_entity_modules
                .iter()
                .map(|(n, _)| entity_dir.join(n)),
        );
        rustfmt_targets.push(entity_dir.join("mod.rs"));
        rustfmt_targets.push(args.out_engine_event_like_impl.clone());
        // engine_rules single-file outputs.
        rustfmt_targets.push(step_file.clone());
        rustfmt_targets.push(backend_file.clone());
        rustfmt_targets.push(mask_fill_file.clone());
        rustfmt_targets.push(cascade_reg_file.clone());
        if let Err(e) = rustfmt(&rustfmt_targets) {
            eprintln!("compile-dsl: rustfmt failed: {e}");
            return ExitCode::FAILURE;
        }

        println!(
            "compile-dsl: wrote {} events (event_hash={}), {} physics rule(s) (rules_hash={}), {} config block(s) (config_hash={}); combined_hash={}",
            combined.events.len(),
            hex(&artefacts.event_hash),
            combined.physics.len(),
            hex(&artefacts.rules_hash),
            combined.configs.len(),
            hex(&artefacts.config_hash),
            hex(&artefacts.combined_hash),
        );
        ExitCode::SUCCESS
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn discover_sim_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !root.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("source directory does not exist: {}", root.display()),
        ));
    }
    walk(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("sim") {
            out.push(path);
        }
    }
    Ok(())
}

/// Per-declaration-kind source paths used to stamp emitted-file headers.
/// Each is `Some(path)` when every declaration of that kind came from a
/// single `.sim` file; `None` when the kind is empty OR multiple files
/// contribute (in which case we emit a generic header).
#[derive(Debug, Default)]
struct PerKindSources {
    events: Option<String>,
    physics: Option<String>,
    masks: Option<String>,
    scoring: Option<String>,
    entities: Option<String>,
    configs: Option<String>,
    enums: Option<String>,
    views: Option<String>,
}

/// Output of `compile_all`: the merged IR plus per-kind source attributions.
struct CompileAll {
    combined: Compilation,
    sources: PerKindSources,
}

/// Parse every `.sim` file, merge their declarations into one `Program`,
/// then resolve in a single pass so cross-file references work (e.g. a
/// `physics` rule in `physics.sim` that matches an `event` declared in
/// `events.sim`). Tracks which file produced each declaration kind so
/// per-kind emission headers can stamp the right source path.
fn compile_all(files: &[PathBuf]) -> Result<CompileAll, ExitCode> {
    let mut merged = Program { decls: Vec::new() };
    let mut events_source: Option<String> = None;
    let mut physics_source: Option<String> = None;
    let mut masks_source: Option<String> = None;
    let mut scoring_source: Option<String> = None;
    let mut entities_source: Option<String> = None;
    let mut configs_source: Option<String> = None;
    let mut enums_source: Option<String> = None;
    let mut views_source: Option<String> = None;
    let mut events_multi = false;
    let mut physics_multi = false;
    let mut masks_multi = false;
    let mut scoring_multi = false;
    let mut entities_multi = false;
    let mut configs_multi = false;
    let mut enums_multi = false;
    let mut views_multi = false;
    let mut seen_events: HashSet<String> = HashSet::new();
    let mut seen_physics: HashSet<String> = HashSet::new();

    for file in files {
        let src = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("compile-dsl: read {}: {e}", file.display());
                return Err(ExitCode::FAILURE);
            }
        };
        let parsed = match dsl_compiler::parse(&src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("compile-dsl: parse {}: {e}", file.display());
                return Err(ExitCode::FAILURE);
            }
        };
        let path = relative_to_repo(file);
        for decl in parsed.decls {
            match &decl {
                Decl::Event(d) => {
                    if !seen_events.insert(d.name.clone()) {
                        eprintln!(
                            "compile-dsl: duplicate event `{}` (also appears in an earlier source)",
                            d.name
                        );
                        return Err(ExitCode::FAILURE);
                    }
                    update_kind_source(&mut events_source, &mut events_multi, &path);
                }
                Decl::Physics(d) => {
                    if !seen_physics.insert(d.name.clone()) {
                        eprintln!(
                            "compile-dsl: duplicate physics rule `{}` (also appears in an earlier source)",
                            d.name
                        );
                        return Err(ExitCode::FAILURE);
                    }
                    update_kind_source(&mut physics_source, &mut physics_multi, &path);
                }
                Decl::Mask(_) => update_kind_source(&mut masks_source, &mut masks_multi, &path),
                Decl::Scoring(_) => update_kind_source(&mut scoring_source, &mut scoring_multi, &path),
                Decl::Entity(_) => update_kind_source(&mut entities_source, &mut entities_multi, &path),
                Decl::Config(_) => update_kind_source(&mut configs_source, &mut configs_multi, &path),
                Decl::Enum(_) => update_kind_source(&mut enums_source, &mut enums_multi, &path),
                Decl::View(_) => update_kind_source(&mut views_source, &mut views_multi, &path),
                // Verb/Invariant/Probe/Metric/EventTag parsed but not yet emitted.
                _ => {}
            }
            merged.decls.push(decl);
        }
    }
    let combined = match dsl_compiler::compile_ast(merged) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("compile-dsl: resolve: {e}");
            return Err(ExitCode::FAILURE);
        }
    };
    Ok(CompileAll {
        combined,
        sources: PerKindSources {
            events: if events_multi { None } else { events_source },
            physics: if physics_multi { None } else { physics_source },
            masks: if masks_multi { None } else { masks_source },
            scoring: if scoring_multi { None } else { scoring_source },
            entities: if entities_multi { None } else { entities_source },
            configs: if configs_multi { None } else { configs_source },
            enums: if enums_multi { None } else { enums_source },
            views: if views_multi { None } else { views_source },
        },
    })
}

/// Track per-kind source attribution: when a declaration of this kind
/// shows up in a new file, mark the kind as multi-source so emission
/// falls back to the generic header.
fn update_kind_source(slot: &mut Option<String>, multi: &mut bool, candidate: &str) {
    if *multi {
        return;
    }
    match slot {
        None => *slot = Some(candidate.to_string()),
        Some(existing) if existing == candidate => {}
        Some(_) => {
            *multi = true;
            *slot = None;
        }
    }
}

fn relative_to_repo(path: &Path) -> String {
    let abs = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let cwd = std::env::current_dir().unwrap_or_default();
    match abs.strip_prefix(&cwd) {
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => abs.to_string_lossy().into_owned(),
    }
}

fn write_artefacts(
    rust_events_dir: &Path,
    rust_schema: &Path,
    physics_dir: &Path,
    py_events_dir: &Path,
    artefacts: &dsl_compiler::EmittedArtifacts,
) -> std::io::Result<()> {
    fs::create_dir_all(rust_events_dir)?;
    if let Some(parent) = rust_schema.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir_all(physics_dir)?;
    fs::create_dir_all(py_events_dir)?;

    // Clear out any stale per-decl files not in the current emission; keep
    // mod.rs / __init__.py / schema.rs (they'll be overwritten below).
    prune_stale(rust_events_dir, &artefacts.rust_event_structs, "rs", &["mod.rs"])?;
    prune_stale(physics_dir, &artefacts.rust_physics_modules, "rs", &["mod.rs"])?;
    prune_stale(py_events_dir, &artefacts.python_event_modules, "py", &["__init__.py"])?;

    for (name, content) in &artefacts.rust_event_structs {
        fs::write(rust_events_dir.join(name), content)?;
    }
    fs::write(rust_events_dir.join("mod.rs"), &artefacts.rust_events_mod)?;
    fs::write(rust_schema, &artefacts.schema_rs)?;

    for (name, content) in &artefacts.rust_physics_modules {
        fs::write(physics_dir.join(name), content)?;
    }
    fs::write(physics_dir.join("mod.rs"), &artefacts.rust_physics_mod)?;

    for (name, content) in &artefacts.python_event_modules {
        fs::write(py_events_dir.join(name), content)?;
    }
    fs::write(py_events_dir.join("__init__.py"), &artefacts.python_events_init)?;
    Ok(())
}

fn prune_stale(
    dir: &Path,
    current: &[(String, String)],
    ext: &str,
    keep: &[&str],
) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let keep: HashSet<&str> = keep.iter().copied().collect();
    let current: HashSet<&str> = current.iter().map(|(n, _)| n.as_str()).collect();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(fname) = path.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        if keep.contains(fname) {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some(ext) {
            continue;
        }
        if !current.contains(fname) {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}

fn check_file(path: &Path, expected: &str, out: &mut Vec<String>) {
    match fs::read_to_string(path) {
        Ok(actual) if actual == expected => {}
        Ok(_) => out.push(format!("{} differs from expected emission", path.display())),
        Err(e) => out.push(format!("{} missing or unreadable ({})", path.display(), e)),
    }
}

/// Like [`check_file`] but normalises both the on-disk content and the
/// expected string through rustfmt before comparing. Use this when the
/// committed file may not be rustfmt-stable (e.g. hand-corrected imports)
/// but the semantic content must match the emitter output.
fn check_file_both_fmt(path: &Path, expected: &str, out: &mut Vec<String>) {
    match fs::read_to_string(path) {
        Ok(actual) => {
            let actual_fmt = rustfmt_string(&actual).unwrap_or(actual);
            let expected_fmt = rustfmt_string(expected).unwrap_or_else(|_| expected.to_string());
            if actual_fmt != expected_fmt {
                out.push(format!("{} differs from expected emission", path.display()));
            }
        }
        Err(e) => out.push(format!("{} missing or unreadable ({})", path.display(), e)),
    }
}

fn check_stale(
    dir: &Path,
    current: &[(String, String)],
    ext: &str,
    out: &mut Vec<String>,
) {
    let Ok(iter) = fs::read_dir(dir) else {
        return;
    };
    let keep_special: HashSet<&str> = ["mod.rs", "__init__.py"].into_iter().collect();
    let current: HashSet<&str> = current.iter().map(|(n, _)| n.as_str()).collect();
    for entry in iter.flatten() {
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        if keep_special.contains(fname) {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some(ext) {
            continue;
        }
        if !current.contains(fname) {
            out.push(format!("{} is stale (no matching event in source)", path.display()));
        }
    }
}

fn rustfmt(files: &[PathBuf]) -> Result<(), String> {
    if files.is_empty() {
        return Ok(());
    }
    let mut cmd = Command::new("rustfmt");
    cmd.arg("--edition=2021");
    for f in files {
        cmd.arg(f);
    }
    let output = cmd.output().map_err(|e| format!("spawn rustfmt: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustfmt exit {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

/// Run rustfmt on an in-memory string and return the formatted output.
/// Used by `--check` mode so the in-memory emission is compared against
/// the rustfmt-stable disk content. If rustfmt isn't available or fails,
/// we return the input unchanged and let the byte comparison decide.
fn rustfmt_string(src: &str) -> Result<String, String> {
    use std::io::Write;
    let mut child = Command::new("rustfmt")
        .arg("--edition=2021")
        .arg("--emit=stdout")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn rustfmt: {e}"))?;
    {
        let stdin = child.stdin.as_mut().ok_or_else(|| "no stdin".to_string())?;
        stdin
            .write_all(src.as_bytes())
            .map_err(|e| format!("write to rustfmt stdin: {e}"))?;
    }
    let output = child.wait_with_output().map_err(|e| format!("wait rustfmt: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustfmt exit {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("rustfmt stdout utf8: {e}"))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Write the mask / scoring / entity aggregator stubs (milestone-3
/// scaffolding). When the corresponding milestone lands the per-decl
/// emitters will populate real files; until then we just keep the mod.rs
/// aggregators in sync.
fn write_scaffolded_kinds(
    mask_dir: &Path,
    scoring_dir: &Path,
    entity_dir: &Path,
    artefacts: &dsl_compiler::EmittedArtifacts,
) -> std::io::Result<()> {
    fs::create_dir_all(mask_dir)?;
    fs::create_dir_all(scoring_dir)?;
    fs::create_dir_all(entity_dir)?;

    // Per-decl files (empty until each kind's emitter lands).
    prune_stale(mask_dir, &artefacts.rust_mask_modules, "rs", &["mod.rs"])?;
    prune_stale(scoring_dir, &artefacts.rust_scoring_modules, "rs", &["mod.rs"])?;
    prune_stale(entity_dir, &artefacts.rust_entity_modules, "rs", &["mod.rs"])?;

    for (name, content) in &artefacts.rust_mask_modules {
        fs::write(mask_dir.join(name), content)?;
    }
    for (name, content) in &artefacts.rust_scoring_modules {
        fs::write(scoring_dir.join(name), content)?;
    }
    for (name, content) in &artefacts.rust_entity_modules {
        fs::write(entity_dir.join(name), content)?;
    }

    fs::write(mask_dir.join("mod.rs"), &artefacts.rust_mask_mod)?;
    fs::write(scoring_dir.join("mod.rs"), &artefacts.rust_scoring_mod)?;
    fs::write(entity_dir.join("mod.rs"), &artefacts.rust_entity_mod)?;
    Ok(())
}

/// Write the per-block config Rust + the aggregator `mod.rs` + the TOML
/// defaults file. Runs in the same write-mode as every other emission kind.
fn write_config_output(
    config_rust_dir: &Path,
    config_toml_path: &Path,
    artefacts: &dsl_compiler::EmittedArtifacts,
) -> std::io::Result<()> {
    fs::create_dir_all(config_rust_dir)?;
    if let Some(parent) = config_toml_path.parent() {
        fs::create_dir_all(parent)?;
    }

    prune_stale(
        config_rust_dir,
        &artefacts.rust_config_modules,
        "rs",
        &["mod.rs"],
    )?;

    for (name, content) in &artefacts.rust_config_modules {
        fs::write(config_rust_dir.join(name), content)?;
    }
    fs::write(config_rust_dir.join("mod.rs"), &artefacts.rust_config_mod)?;
    fs::write(config_toml_path, &artefacts.config_default_toml)?;
    Ok(())
}

/// Write per-enum Rust + Python files and their aggregator mod/init.
fn write_enum_output(
    enum_rust_dir: &Path,
    py_enums_dir: &Path,
    artefacts: &dsl_compiler::EmittedArtifacts,
) -> std::io::Result<()> {
    fs::create_dir_all(enum_rust_dir)?;
    fs::create_dir_all(py_enums_dir)?;

    prune_stale(enum_rust_dir, &artefacts.rust_enum_modules, "rs", &["mod.rs"])?;
    prune_stale(py_enums_dir, &artefacts.python_enum_modules, "py", &["__init__.py"])?;

    for (name, content) in &artefacts.rust_enum_modules {
        fs::write(enum_rust_dir.join(name), content)?;
    }
    fs::write(enum_rust_dir.join("mod.rs"), &artefacts.rust_enum_mod)?;

    for (name, content) in &artefacts.python_enum_modules {
        fs::write(py_enums_dir.join(name), content)?;
    }
    fs::write(py_enums_dir.join("__init__.py"), &artefacts.python_enum_init)?;
    Ok(())
}

/// Write every per-view Rust module plus the aggregator `mod.rs` to the
/// views output directory. Stale per-view files from a previous run are
/// pruned so renames don't leave orphans behind.
fn write_views_output(
    views_dir: &Path,
    artefacts: &dsl_compiler::EmittedArtifacts,
) -> std::io::Result<()> {
    fs::create_dir_all(views_dir)?;
    prune_stale(views_dir, &artefacts.rust_view_modules, "rs", &["mod.rs"])?;
    for (name, content) in &artefacts.rust_view_modules {
        fs::write(views_dir.join(name), content)?;
    }
    fs::write(views_dir.join("mod.rs"), &artefacts.rust_view_mod)?;
    Ok(())
}

/// Write the four static-ish single-file engine_rules outputs:
/// `step.rs`, `backend.rs`, `mask_fill.rs`, `cascade_reg.rs`.
/// These are kept as compiler-owned emissions so DSL-driven phases
/// (invariant checks, future step extensions) can grow into them.
fn write_engine_rules_singles(
    step_path: &Path,
    backend_path: &Path,
    mask_fill_path: &Path,
    cascade_reg_path: &Path,
    sources: &PerKindSources,
    combined: &dsl_compiler::ir::Compilation,
) -> std::io::Result<()> {
    for p in [step_path, backend_path, mask_fill_path, cascade_reg_path] {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(
        step_path,
        dsl_compiler::emit_step::emit_step(sources.physics.as_deref()),
    )?;
    fs::write(
        backend_path,
        dsl_compiler::emit_backend::emit_backend(sources.physics.as_deref()),
    )?;
    fs::write(
        mask_fill_path,
        dsl_compiler::emit_mask_fill::emit_mask_fill(
            &combined.masks,
            sources.masks.as_deref(),
        ),
    )?;
    fs::write(
        cascade_reg_path,
        dsl_compiler::emit_cascade_register::emit_cascade_register(sources.physics.as_deref()),
    )?;
    Ok(())
}

/// `--check` counterpart to [`write_views_output`]. Verifies every per-view
/// module + the aggregator match the committed emission post-rustfmt.
fn check_views(
    artefacts: &dsl_compiler::EmittedArtifacts,
    views_dir: &Path,
    mismatches: &mut Vec<String>,
) {
    for (name, content) in &artefacts.rust_view_modules {
        check_file_both_fmt(&views_dir.join(name), content, mismatches);
    }
    check_file_both_fmt(&views_dir.join("mod.rs"), &artefacts.rust_view_mod, mismatches);
    check_stale(views_dir, &artefacts.rust_view_modules, "rs", mismatches);
}

/// `--check` counterpart to [`write_config_output`]. Verifies every per-block
/// file matches the committed emission (post-rustfmt) and that the TOML
/// defaults file is byte-identical.
fn check_config(
    artefacts: &dsl_compiler::EmittedArtifacts,
    config_rust_dir: &Path,
    config_toml_path: &Path,
    mismatches: &mut Vec<String>,
) {
    for (name, content) in &artefacts.rust_config_modules {
        let f = rustfmt_string(content).unwrap_or_else(|_| content.clone());
        check_file(&config_rust_dir.join(name), &f, mismatches);
    }
    let fmt = rustfmt_string(&artefacts.rust_config_mod)
        .unwrap_or_else(|_| artefacts.rust_config_mod.clone());
    check_file(&config_rust_dir.join("mod.rs"), &fmt, mismatches);
    check_file(config_toml_path, &artefacts.config_default_toml, mismatches);
    check_stale(
        config_rust_dir,
        &artefacts.rust_config_modules,
        "rs",
        mismatches,
    );
}

fn check_enums(
    artefacts: &dsl_compiler::EmittedArtifacts,
    enum_rust_dir: &Path,
    py_enums_dir: &Path,
    mismatches: &mut Vec<String>,
) {
    for (name, content) in &artefacts.rust_enum_modules {
        let f = rustfmt_string(content).unwrap_or_else(|_| content.clone());
        check_file(&enum_rust_dir.join(name), &f, mismatches);
    }
    let fmt = rustfmt_string(&artefacts.rust_enum_mod)
        .unwrap_or_else(|_| artefacts.rust_enum_mod.clone());
    check_file(&enum_rust_dir.join("mod.rs"), &fmt, mismatches);
    for (name, content) in &artefacts.python_enum_modules {
        check_file(&py_enums_dir.join(name), content, mismatches);
    }
    check_file(
        &py_enums_dir.join("__init__.py"),
        &artefacts.python_enum_init,
        mismatches,
    );
    check_stale(enum_rust_dir, &artefacts.rust_enum_modules, "rs", mismatches);
    check_stale(py_enums_dir, &artefacts.python_enum_modules, "py", mismatches);
}

fn check_scaffolded_kinds(
    artefacts: &dsl_compiler::EmittedArtifacts,
    mask_dir: &Path,
    scoring_dir: &Path,
    entity_dir: &Path,
    mismatches: &mut Vec<String>,
) {
    for (name, content) in &artefacts.rust_mask_modules {
        check_file_both_fmt(&mask_dir.join(name), content, mismatches);
    }
    check_file_both_fmt(&mask_dir.join("mod.rs"), &artefacts.rust_mask_mod, mismatches);

    for (name, content) in &artefacts.rust_scoring_modules {
        let f = rustfmt_string(content).unwrap_or_else(|_| content.clone());
        check_file(&scoring_dir.join(name), &f, mismatches);
    }
    let fmt = rustfmt_string(&artefacts.rust_scoring_mod).unwrap_or_else(|_| artefacts.rust_scoring_mod.clone());
    check_file(&scoring_dir.join("mod.rs"), &fmt, mismatches);

    for (name, content) in &artefacts.rust_entity_modules {
        let f = rustfmt_string(content).unwrap_or_else(|_| content.clone());
        check_file(&entity_dir.join(name), &f, mismatches);
    }
    let fmt = rustfmt_string(&artefacts.rust_entity_mod).unwrap_or_else(|_| artefacts.rust_entity_mod.clone());
    check_file(&entity_dir.join("mod.rs"), &fmt, mismatches);

    check_stale(mask_dir, &artefacts.rust_mask_modules, "rs", mismatches);
    check_stale(scoring_dir, &artefacts.rust_scoring_modules, "rs", mismatches);
    check_stale(entity_dir, &artefacts.rust_entity_modules, "rs", mismatches);
}
