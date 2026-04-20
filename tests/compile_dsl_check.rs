//! xtask `compile-dsl --check` end-to-end test.
//!
//! Verifies that the committed generated output in `crates/engine_rules/`
//! and `generated/python/events/` matches what the compiler currently
//! emits — the CI guard. Also verifies that tampering with a committed
//! file causes the check to fail.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn run_check() -> std::process::Output {
    Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--bin", "xtask", "--", "compile-dsl", "--check"])
        .current_dir(repo_root())
        .output()
        .expect("spawn xtask")
}

#[test]
#[ignore = "invokes cargo-run of the xtask binary; expensive, enable locally with --include-ignored"]
fn check_passes_on_committed_output() {
    let out = run_check();
    assert!(
        out.status.success(),
        "expected --check to pass on committed output:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
#[ignore = "invokes cargo-run of the xtask binary; expensive, enable locally with --include-ignored"]
fn check_fails_when_file_tampered() {
    let damage = repo_root().join("crates/engine_rules/src/events/damage.rs");
    let original = fs::read_to_string(&damage).expect("read damage.rs");
    let mut tampered = original.clone();
    // Flip one byte deterministically — drop the trailing newline.
    tampered.pop();
    fs::write(&damage, &tampered).expect("write tamper");

    let out = run_check();
    let status_ok = out.status.success();

    // Restore before any assert panics so the test is safe to re-run.
    fs::write(&damage, &original).expect("restore damage.rs");

    assert!(
        !status_ok,
        "--check should have failed on tampered file:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
