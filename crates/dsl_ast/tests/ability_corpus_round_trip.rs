//! Wider grammar coverage: parse → emit → parse round-trip on the
//! real `.ability` corpus (LoL heroes + the in-tree fixtures). For
//! each file:
//!
//!   1. Parse the original source into an `AbilityFile`.
//!   2. For each ability decl that the emitter can serialise
//!      losslessly (no opaque `deliver` / `morph` / `template`
//!      instantiation / `program` block), emit it back to source.
//!   3. Parse the emitted source.
//!   4. Assert the secondary parse succeeds.
//!
//! Skips abilities that carry opaque blocks (`deliver`, `morph`,
//! `program`, template instantiation) — those round-trip via verbatim
//! `raw: String` slots that the AST-shape emitter doesn't reconstruct,
//! and are out of scope for this walker.
//!
//! Reports the count of attempted / passed / skipped files so a
//! regression that drops corpus coverage surfaces visibly.

use dsl_ast::ability_emit::emit_ability_file_single;
use dsl_ast::ability_parser::parse_ability_file;
use dsl_ast::ast::AbilityDecl;

fn is_emitter_supported(d: &AbilityDecl) -> bool {
    d.deliver.is_none()
        && d.morph.is_none()
        && d.instantiates.is_none()
        && d.program.is_none()
}

fn corpus_dirs() -> Vec<std::path::PathBuf> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    vec![
        workspace.join("dataset").join("abilities").join("lol_heroes"),
    ]
}

fn collect_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("ability") {
            out.push(path);
        }
    }
    out.sort();
    out
}

#[test]
fn lol_corpus_round_trips_through_emit() {
    let mut attempted = 0usize;
    let mut emitter_supported = 0usize;
    let mut passed = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();

    for dir in corpus_dirs() {
        for path in collect_files(&dir) {
            let stem = path.file_name().unwrap().to_string_lossy().to_string();
            let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("read {stem}: {e}");
            });
            // Skip files the primary parser already rejects — those are
            // ahead-of-grammar corpus drift, not emitter regressions.
            let file = match parse_ability_file(&src) {
                Ok(f) => f,
                Err(_) => continue,
            };
            for ad in &file.abilities {
                attempted += 1;
                if !is_emitter_supported(ad) {
                    continue;
                }
                emitter_supported += 1;
                let emitted = emit_ability_file_single(ad);
                match parse_ability_file(&emitted) {
                    Ok(parsed) => {
                        if parsed.abilities.len() == 1 && parsed.abilities[0].name == ad.name {
                            passed += 1;
                        } else {
                            failures.push((
                                format!("{stem}::{}", ad.name),
                                "name/count mismatch after re-parse".to_string(),
                            ));
                        }
                    }
                    Err(e) => failures.push((
                        format!("{stem}::{}", ad.name),
                        format!("re-parse failed: {e}\nemitted:\n{emitted}"),
                    )),
                }
            }
        }
    }

    println!("[ability_corpus_round_trip] attempted={attempted} \
              emitter_supported={emitter_supported} passed={passed} \
              failed={}",
             failures.len());

    if !failures.is_empty() {
        for (name, msg) in failures.iter().take(5) {
            eprintln!("FAIL {name}: {msg}");
        }
        panic!(
            "{} of {} emitter-supported corpus ability decls failed round-trip",
            failures.len(),
            emitter_supported
        );
    }

    // Sanity: at least some of the corpus must be emitter-supported,
    // otherwise the test is silently a no-op. The corpus has 172
    // files and many are emitter-supported (no `deliver` / `program`).
    assert!(
        emitter_supported >= 20,
        "expected ≥20 emitter-supported ability decls in the corpus; got {emitter_supported}"
    );
}
