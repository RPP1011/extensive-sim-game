//! Diagnostic smoke run over the entire `.ability` corpus. Emits a
//! per-file pass/fail tally with the failure reason (truncated to first
//! line) so later slices can prioritise which DSL constructs to land
//! next. Marked `#[ignore]` so it doesn't count toward the test-count
//! contract; run explicitly with:
//!
//!     cargo test -p dsl_ast --test ability_corpus_smoke -- --ignored --nocapture

use dsl_ast::parse_ability_file;
use std::fs;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
    if !dir.is_dir() {
        return;
    }
    for entry in fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, files);
        } else if path.extension().map_or(false, |e| e == "ability") {
            files.push(path);
        }
    }
}

#[test]
#[ignore]
fn ability_corpus_smoke() {
    // Walk up from the crate dir to the workspace root (which holds
    // `dataset/`). The crate lives at `crates/dsl_ast`; two levels up
    // is the workspace.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
    let dataset = workspace.join("dataset");
    if !dataset.is_dir() {
        eprintln!("dataset/ not found at {}; skipping corpus smoke", dataset.display());
        return;
    }

    let mut files = Vec::new();
    walk(&dataset, &mut files);
    files.sort();

    let mut ok = 0usize;
    let mut deferred = 0usize;
    let mut other = 0usize;
    let mut samples: Vec<String> = Vec::new();

    for path in &files {
        let src = fs::read_to_string(path).unwrap();
        match parse_ability_file(&src) {
            Ok(_) => ok += 1,
            Err(e) => {
                let line = e.to_string().lines().next().unwrap_or("").to_string();
                if line.contains("passive")
                    || line.contains("template")
                    || line.contains("structure")
                {
                    deferred += 1;
                } else {
                    other += 1;
                    if samples.len() < 30 {
                        samples.push(format!("{}: {line}", path.display()));
                    }
                }
            }
        }
    }
    eprintln!("---- ability corpus smoke ----");
    eprintln!("total files: {}", files.len());
    eprintln!("parsed ok:   {ok}");
    eprintln!("deferred (passive/template/structure top-level): {deferred}");
    eprintln!("other failures: {other}");
    for s in &samples {
        eprintln!("  - {s}");
    }
}
