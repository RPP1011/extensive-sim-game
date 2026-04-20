//! `cargo run --bin xtask -- compile-dsl` — walk `assets/sim/*.sim`, parse
//! and resolve into a single `Compilation`, then emit Rust + Python +
//! schema-hash artefacts. Either writes the files (default) or compares them
//! against the committed output (`--check`, CI guard mode).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use dsl_compiler::ir::Compilation;

use super::cli::CompileDslArgs;

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

    let (combined, primary_source) = match compile_all(&sim_files) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let artefacts = dsl_compiler::emit_with_source(&combined, primary_source.as_deref());

    let rust_events_dir = args.out_rust.join("events");
    let rust_schema = args.out_rust.join("schema.rs");
    let py_events_dir = args.out_python.join("events");

    if args.check {
        let mut mismatches = Vec::new();

        // Rust per-event files.
        for (name, content) in &artefacts.rust_event_structs {
            check_file(&rust_events_dir.join(name), content, &mut mismatches);
        }
        check_file(
            &rust_events_dir.join("mod.rs"),
            &artefacts.rust_events_mod,
            &mut mismatches,
        );
        check_file(&rust_schema, &artefacts.schema_rs, &mut mismatches);

        // Python per-event files.
        for (name, content) in &artefacts.python_event_modules {
            check_file(&py_events_dir.join(name), content, &mut mismatches);
        }
        check_file(
            &py_events_dir.join("__init__.py"),
            &artefacts.python_events_init,
            &mut mismatches,
        );

        // Stale file detection: committed Rust files not in the new emission.
        check_stale(&rust_events_dir, &artefacts.rust_event_structs, "rs", &mut mismatches);
        check_stale(&py_events_dir, &artefacts.python_event_modules, "py", &mut mismatches);

        if mismatches.is_empty() {
            println!("compile-dsl: check ok ({} events)", combined.events.len());
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
            &py_events_dir,
            &artefacts,
        ) {
            eprintln!("compile-dsl: {e}");
            return ExitCode::FAILURE;
        }

        // Format emitted Rust so it matches the project's style. Best effort —
        // if rustfmt fails (missing toolchain, generated file has a bug we
        // want to see), surface the error.
        let rustfmt_targets: Vec<PathBuf> = artefacts
            .rust_event_structs
            .iter()
            .map(|(n, _)| rust_events_dir.join(n))
            .chain([rust_events_dir.join("mod.rs"), rust_schema.clone()])
            .collect();
        if let Err(e) = rustfmt(&rustfmt_targets) {
            eprintln!("compile-dsl: rustfmt failed: {e}");
            return ExitCode::FAILURE;
        }

        println!(
            "compile-dsl: wrote {} events (hash={})",
            combined.events.len(),
            hex(&artefacts.event_hash),
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

/// Parse every `.sim` file, merge events into one `Compilation`. Reject
/// duplicate event names across files. Returns the merged compilation and,
/// when every event came from a single file, the path of that file (used to
/// stamp headers).
fn compile_all(files: &[PathBuf]) -> Result<(Compilation, Option<String>), ExitCode> {
    let mut combined = Compilation::default();
    let mut seen_events: HashSet<String> = HashSet::new();
    let mut sources: HashSet<String> = HashSet::new();

    for file in files {
        let src = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("compile-dsl: read {}: {e}", file.display());
                return Err(ExitCode::FAILURE);
            }
        };
        let comp = match dsl_compiler::compile(&src) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("compile-dsl: compile {}: {e}", file.display());
                return Err(ExitCode::FAILURE);
            }
        };
        for event in comp.events {
            if !seen_events.insert(event.name.clone()) {
                eprintln!(
                    "compile-dsl: duplicate event `{}` (also appears in an earlier source)",
                    event.name
                );
                return Err(ExitCode::FAILURE);
            }
            combined.events.push(event);
            sources.insert(relative_to_repo(file));
        }
    }

    let primary = if sources.len() == 1 {
        sources.into_iter().next()
    } else {
        None
    };
    Ok((combined, primary))
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
    py_events_dir: &Path,
    artefacts: &dsl_compiler::EmittedArtifacts,
) -> std::io::Result<()> {
    fs::create_dir_all(rust_events_dir)?;
    if let Some(parent) = rust_schema.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir_all(py_events_dir)?;

    // Clear out any stale per-event files not in the current emission; keep
    // mod.rs / __init__.py / schema.rs (they'll be overwritten below).
    prune_stale(rust_events_dir, &artefacts.rust_event_structs, "rs", &["mod.rs"])?;
    prune_stale(py_events_dir, &artefacts.python_event_modules, "py", &["__init__.py"])?;

    for (name, content) in &artefacts.rust_event_structs {
        fs::write(rust_events_dir.join(name), content)?;
    }
    fs::write(rust_events_dir.join("mod.rs"), &artefacts.rust_events_mod)?;
    fs::write(rust_schema, &artefacts.schema_rs)?;

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

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
