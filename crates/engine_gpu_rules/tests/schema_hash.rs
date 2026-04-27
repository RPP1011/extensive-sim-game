//! Baseline-comparison test for engine_gpu_rules' generated content.
//! On a CI failure, the fix is one of:
//!   1. Run `cargo run --bin xtask -- compile-dsl` to re-emit; commit
//!      the regen alongside the .sim change that caused it.
//!   2. If the regen is intentional, the .schema_hash baseline is
//!      already updated by the xtask — review the diff and commit.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

fn compute_current_hash() -> String {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    walk(Path::new("src"), &mut entries);
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut h = Sha256::new();
    h.update(b"engine_gpu_rules:v1");
    for (name, bytes) in &entries {
        h.update((name.len() as u32).to_le_bytes());
        h.update(name.as_bytes());
        h.update((bytes.len() as u32).to_le_bytes());
        h.update(bytes);
    }
    let bytes: [u8; 32] = h.finalize().into();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn walk(dir: &Path, out: &mut Vec<(String, Vec<u8>)>) {
    for entry in fs::read_dir(dir).expect("readable src") {
        let entry = entry.expect("entry");
        let p = entry.path();
        if entry.file_type().expect("ft").is_dir() { walk(&p, out); continue; }
        let ext = p.extension().and_then(|e| e.to_str());
        if !matches!(ext, Some("rs") | Some("wgsl")) { continue; }
        let rel = p.strip_prefix("src").unwrap();
        let bytes = fs::read(&p).expect("read");
        out.push((rel.display().to_string(), bytes));
    }
}

#[test]
fn baseline_matches_current() {
    let baseline = include_str!("../.schema_hash").trim();
    let current = compute_current_hash();
    assert_eq!(
        current, baseline,
        "engine_gpu_rules content changed.\n\
         If intentional: re-run `cargo run --bin xtask -- compile-dsl` to bump the baseline.\n\
         Current: {}", current
    );
}
