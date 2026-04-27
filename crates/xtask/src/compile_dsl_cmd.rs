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

        // Emit engine_gpu_rules/src/lib.rs from the (initially empty) module
        // list. Subsequent kernel-emit tasks will populate the list.
        {
            let modules: Vec<String> = Vec::new(); // populated by per-kernel emitters in later tasks
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

        // Schedule (initially empty; populated by per-kernel emitters in later tasks).
        {
            let schedule_rs = dsl_compiler::emit_schedule::emit_schedule_rs(&[]);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/schedule.rs"),
                &schedule_rs,
            ) {
                eprintln!("compile-dsl: write schedule.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Resident context (initially empty struct).
        {
            let rc_rs = dsl_compiler::emit_resident_context::emit_resident_context_rs(&[]);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/resident_context.rs"),
                &rc_rs,
            ) {
                eprintln!("compile-dsl: write resident_context.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Pingpong context (initially empty struct; Task 9 populates).
        {
            let body = dsl_compiler::emit_pingpong_context::emit_pingpong_context_rs(&[]);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/pingpong_context.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write pingpong_context.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Pool (initially empty; Task 12 populates with spatial scratch).
        {
            let body = dsl_compiler::emit_pool::emit_pool_rs(&[]);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/pool.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write pool.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // Transient handles (initially empty; per-kernel tasks 4-13 populate).
        {
            let body = dsl_compiler::emit_transient_handles::emit_transient_handles_rs(&[]);
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/transient_handles.rs"),
                body,
            ) {
                eprintln!("compile-dsl: write transient_handles.rs: {e}");
                return ExitCode::FAILURE;
            }
        }

        // External buffers (initially empty; per-kernel tasks 4-13 populate).
        {
            let body = dsl_compiler::emit_external_buffers::emit_external_buffers_rs(&[]);
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

        // Megakernel — empty stub until Task 14.
        {
            let stub = "// GENERATED by dsl_compiler. Do not edit by hand.\n\
                        // Megakernel emit lands in Task 14; this stub keeps the module list\n\
                        // declared by emit_kernel_index resolvable.\n";
            if let Err(e) = fs::write(
                PathBuf::from("crates/engine_gpu_rules/src/megakernel.rs"),
                stub,
            ) {
                eprintln!("compile-dsl: write megakernel.rs: {e}");
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
