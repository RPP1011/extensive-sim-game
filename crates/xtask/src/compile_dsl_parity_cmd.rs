//! `cargo run --bin xtask -- compile-dsl-parity` — behavioral parity
//! harness for the Compute-Graph IR (CG) pipeline.
//!
//! Reads the CG side-channel directory previously populated by
//! `compile-dsl --cg-emit-into <dir>` and reports whether the output
//! is structurally inhabited enough to overlay a working
//! `engine_gpu_rules` build. See Task 5.1 of
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`.
//!
//! # Limitations (Task 5.1, 2026-04-29)
//!
//! - Today this only verifies that the side-channel directory exists
//!   and contains some emitted files. It does NOT yet attempt to
//!   overlay-build `engine_gpu_rules` or run `parity_with_cpu`.
//!   Closing that gap is Tasks 5.2-5.7 of the reframed plan:
//!   - Task 5.2 emits the `Kernel` trait impls + cross-kernel helpers.
//!   - Task 5.4 emits the cross-cutting modules (BindingSources,
//!     transient/external/resident contexts, schedule).
//!   - Task 5.5 closes mask/scoring/physics/spatial AST coverage so
//!     the emitted output covers every legacy kernel.
//!   - Task 5.7 runs `parity_with_cpu` against the CG-built engine.
//! - **The command is EXPECTED to FAIL today** because the CG output
//!   is missing the pieces those follow-up tasks add (`Kernel` trait
//!   impls, cross-kernel helpers, Cargo.toml/lib.rs, full op
//!   coverage). The non-zero exit is the diagnostic the controller
//!   uses to decide what to wire next.

use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use crate::cli::CompileDslParityArgs;

pub fn run_compile_dsl_parity(args: CompileDslParityArgs) -> ExitCode {
    match probe_cg_overlay(&args.cg_out) {
        Ok(report) => {
            print_report(&report);
            // Today every report is "incomplete" — see the module-level
            // limitations. We still surface a non-zero exit so the
            // controller can wire up follow-up tasks against the
            // documented missing pieces.
            if report.is_complete() {
                println!("compile-dsl-parity: PASS");
                ExitCode::SUCCESS
            } else {
                eprintln!(
                    "compile-dsl-parity: FAIL (CG output is structurally incomplete; \
                     this is expected in Task 5.1 — see follow-up tasks 5.2-5.7)"
                );
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("compile-dsl-parity: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Diagnostic report on a CG side-channel directory.
///
/// `is_complete()` is the ladder Tasks 5.2-5.7 climb. The current
/// (Task 5.1) shape always returns `false` because of the documented
/// missing pieces.
///
/// When the side-channel emit failed mid-pipeline (e.g. an
/// `emit_cg_program` error in today's Task 5.1 state), `emit_cg_side_channel`
/// drops a `_diagnostics.txt` next to where the kernels would have gone.
/// That file is the truth source for what blocked the run; the static
/// "missing pieces" map below is a structural fallback when no
/// diagnostics file exists.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ParityReport {
    pub(crate) src_dir_exists: bool,
    pub(crate) wgsl_count: usize,
    pub(crate) rs_count: usize,
    pub(crate) has_lib_rs: bool,
    pub(crate) has_cargo_toml: bool,
    /// `true` when `<cg_out>/src/_diagnostics.txt` was found during the
    /// probe. The side-channel writes this file when it fails mid-pipeline
    /// — its presence means the structural "missing pieces" map is NOT
    /// the right diagnostic to surface; `diagnostics_content` is.
    pub(crate) has_diagnostics_txt: bool,
    /// The verbatim contents of `<cg_out>/src/_diagnostics.txt` when
    /// present. `None` when the file is absent OR could not be read.
    pub(crate) diagnostics_content: Option<String>,
}

impl ParityReport {
    /// "Complete" means every piece a future overlay build would
    /// need is present. Today this is structural only — mere file
    /// counts plus the existence of `lib.rs` and `Cargo.toml`. Task
    /// 5.7 will replace this with a real `cargo check` invocation.
    pub(crate) fn is_complete(&self) -> bool {
        self.src_dir_exists
            && self.wgsl_count > 0
            && self.rs_count > 0
            && self.has_lib_rs
            && self.has_cargo_toml
    }
}

/// Probe the side-channel directory `<cg_out>/src/` and count emitted
/// files. Errors out (typed string) on missing or non-directory paths.
pub(crate) fn probe_cg_overlay(cg_out: &Path) -> Result<ParityReport, String> {
    if !cg_out.exists() {
        return Err(format!(
            "--cg-out directory does not exist: {} (run \
             `cargo run --bin xtask -- compile-dsl --cg-emit-into <dir>` first)",
            cg_out.display()
        ));
    }
    if !cg_out.is_dir() {
        return Err(format!(
            "--cg-out is not a directory: {}",
            cg_out.display()
        ));
    }

    let src_dir = cg_out.join("src");
    let src_dir_exists = src_dir.is_dir();

    let mut wgsl_count = 0usize;
    let mut rs_count = 0usize;
    if src_dir_exists {
        for entry in fs::read_dir(&src_dir)
            .map_err(|e| format!("read_dir {}: {e}", src_dir.display()))?
        {
            let entry =
                entry.map_err(|e| format!("read_dir entry {}: {e}", src_dir.display()))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            match path.extension().and_then(OsStr::to_str) {
                Some("wgsl") => wgsl_count += 1,
                Some("rs") => rs_count += 1,
                _ => {}
            }
        }
    }

    let has_lib_rs = src_dir.join("lib.rs").is_file();
    let has_cargo_toml = cg_out.join("Cargo.toml").is_file();

    // Probe `<cg_out>/src/_diagnostics.txt` — the side-channel writes this
    // when `emit_cg_program` fails. Its content is the actual blocker, so
    // we capture it here and surface it ahead of the static
    // "missing pieces" map in `print_report`. If the file exists but reads
    // fail (permissions, etc.), we record `has_diagnostics_txt = true`
    // with `diagnostics_content = None` so the report still notes the
    // file's presence.
    let diag_path = src_dir.join("_diagnostics.txt");
    let has_diagnostics_txt = diag_path.is_file();
    let diagnostics_content = if has_diagnostics_txt {
        fs::read_to_string(&diag_path).ok()
    } else {
        None
    };

    Ok(ParityReport {
        src_dir_exists,
        wgsl_count,
        rs_count,
        has_lib_rs,
        has_cargo_toml,
        has_diagnostics_txt,
        diagnostics_content,
    })
}

/// Print the report. When `_diagnostics.txt` is present, surface its
/// content FIRST as the primary diagnostic — that file is the truth
/// source for what blocked the side-channel emit, and the static
/// "missing pieces" map below would otherwise mis-attribute the blocker
/// to the wrong follow-up task. The static map is kept as a structural
/// fallback for the no-diagnostics case (where the missing pieces are
/// stable structural gaps Tasks 5.2 / 5.4 / 5.5 close).
fn print_report(report: &ParityReport) {
    println!("compile-dsl-parity report:");
    println!("  src/ directory exists: {}", report.src_dir_exists);
    println!("  *.wgsl files: {}", report.wgsl_count);
    println!("  *.rs files:   {}", report.rs_count);
    println!("  lib.rs:       {}", report.has_lib_rs);
    println!("  Cargo.toml:   {}", report.has_cargo_toml);
    println!("  _diagnostics.txt: {}", report.has_diagnostics_txt);

    // Primary diagnostic: when the side-channel left a `_diagnostics.txt`,
    // print its content verbatim. This is what actually blocked the run
    // (e.g. today's `KernelNameCollision: cg_indirect_view_fold_1`,
    // closed by Task 5.2's semantic naming) — surfacing it ahead of the
    // static map keeps the controller's "what to wire next" attribution
    // honest.
    if report.has_diagnostics_txt {
        println!();
        println!("  Side-channel diagnostics (from <cg_out>/src/_diagnostics.txt):");
        match report.diagnostics_content.as_deref() {
            Some(content) => {
                for line in content.lines() {
                    println!("    {line}");
                }
            }
            None => {
                println!("    (file present but unreadable — check filesystem permissions)");
            }
        }
    }

    if !report.is_complete() {
        println!();
        // When `_diagnostics.txt` is present, demote the static map to
        // "supplementary context" — the diagnostics file already named
        // the real blocker, so the structural gaps below are downstream
        // consequences, not the thing to fix first.
        let header = if report.has_diagnostics_txt {
            "Structural gaps (supplementary; diagnostics above are the primary signal):"
        } else {
            "Missing pieces (closed by follow-up tasks):"
        };
        println!("  {header}");
        if !report.src_dir_exists {
            println!("    - src/ directory absent — re-run `compile-dsl --cg-emit-into <dir>`");
        }
        if report.wgsl_count == 0 {
            println!("    - no *.wgsl files (Task 5.5: AST coverage for mask/scoring/physics/spatial)");
        }
        if report.rs_count == 0 {
            println!("    - no *.rs files (Task 5.5: AST coverage)");
        }
        if !report.has_lib_rs {
            println!("    - lib.rs missing (Task 5.2: Kernel trait impls + lib.rs synthesis)");
        }
        if !report.has_cargo_toml {
            println!("    - Cargo.toml missing (Task 5.4: cross-cutting modules + manifest)");
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Pick a unique scratch directory under `std::env::temp_dir()`.
    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("xtask_parity_{tag}_{pid}_{nanos}"));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    /// `probe_cg_overlay` errors out when the path doesn't exist.
    #[test]
    fn probe_errors_on_missing_directory() {
        let dir = scratch_dir("missing");
        assert!(!dir.exists());
        let err = probe_cg_overlay(&dir).expect_err("missing path must error");
        assert!(
            err.contains("does not exist"),
            "unexpected error: {err}"
        );
    }

    /// An empty directory yields a non-complete report.
    #[test]
    fn probe_reports_empty_directory_as_incomplete() {
        let dir = scratch_dir("empty");
        fs::create_dir_all(&dir).unwrap();
        let report = probe_cg_overlay(&dir).expect("empty dir is a valid probe target");
        assert!(!report.is_complete(), "empty dir must be incomplete: {report:?}");
        assert_eq!(report.wgsl_count, 0);
        assert_eq!(report.rs_count, 0);
        assert!(!report.has_lib_rs);
        assert!(!report.has_cargo_toml);
        assert!(!report.has_diagnostics_txt);
        assert!(report.diagnostics_content.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    /// When `<cg_out>/src/_diagnostics.txt` exists, the probe captures
    /// its content verbatim. This is the truth-source surface the
    /// parity report prints first (ahead of the structural "missing
    /// pieces" map) — see `print_report`.
    #[test]
    fn probe_captures_diagnostics_txt_when_present() {
        let dir = scratch_dir("diag_present");
        fs::create_dir_all(dir.join("src")).unwrap();
        let diag_body =
            "CG-side-channel emit failed at the emit_cg_program step.\n\
             Error:\nKernelNameCollision: cg_indirect_view_fold_1\n";
        fs::write(dir.join("src").join("_diagnostics.txt"), diag_body).unwrap();

        let report = probe_cg_overlay(&dir).expect("probe must succeed when diag file present");
        assert!(report.has_diagnostics_txt, "diag flag must be set: {report:?}");
        assert_eq!(
            report.diagnostics_content.as_deref(),
            Some(diag_body),
            "diag content must round-trip verbatim"
        );
        // The diagnostics file alone doesn't make the report complete —
        // structural pieces still missing.
        assert!(!report.is_complete(), "diag-only dir is still incomplete: {report:?}");

        let _ = fs::remove_dir_all(&dir);
    }

    /// A directory with src/ + some files counts the right number,
    /// but is still incomplete without lib.rs + Cargo.toml.
    #[test]
    fn probe_counts_files_but_marks_incomplete_without_manifest() {
        let dir = scratch_dir("partial");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src").join("k0.wgsl"), "// stub").unwrap();
        fs::write(dir.join("src").join("k0.rs"), "// stub").unwrap();
        fs::write(dir.join("src").join("k1.rs"), "// stub").unwrap();

        let report = probe_cg_overlay(&dir).unwrap();
        assert!(report.src_dir_exists);
        assert_eq!(report.wgsl_count, 1);
        assert_eq!(report.rs_count, 2);
        assert!(!report.has_lib_rs);
        assert!(!report.has_cargo_toml);
        assert!(!report.is_complete());

        let _ = fs::remove_dir_all(&dir);
    }

    /// A directory with every structural piece (files + lib.rs +
    /// Cargo.toml) marks as complete. This is the shape Task 5.7
    /// will eventually exercise; pinning the contract here.
    #[test]
    fn probe_marks_complete_when_all_structural_pieces_present() {
        let dir = scratch_dir("complete");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src").join("k0.wgsl"), "// stub").unwrap();
        fs::write(dir.join("src").join("k0.rs"), "// stub").unwrap();
        fs::write(dir.join("src").join("lib.rs"), "// stub").unwrap();
        fs::write(dir.join("Cargo.toml"), "[package]\nname=\"x\"\nversion=\"0\"\nedition=\"2021\"\n").unwrap();

        let report = probe_cg_overlay(&dir).unwrap();
        assert!(report.is_complete(), "all structural pieces present: {report:?}");

        let _ = fs::remove_dir_all(&dir);
    }

    /// Non-directory paths fail with a typed error.
    #[test]
    fn probe_errors_when_path_is_a_file() {
        let dir = scratch_dir("not_dir_parent");
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("just_a_file");
        fs::write(&file_path, "x").unwrap();
        let err = probe_cg_overlay(&file_path).expect_err("file path must error");
        assert!(err.contains("not a directory"), "unexpected error: {err}");
        let _ = fs::remove_dir_all(&dir);
    }
}
