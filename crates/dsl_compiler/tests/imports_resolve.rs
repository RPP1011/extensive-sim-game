use dsl_compiler::imports::{resolve_import_path, ImportError};
use tempfile::tempdir;

#[test]
fn stdlib_resolves_against_stdlib_root() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(stdlib.join("materials")).unwrap();
    let target = stdlib.join("materials/basic.sim");
    std::fs::write(&target, "// stdlib content\n").unwrap();

    let importing = tmp.path().join("a.sim");
    std::fs::write(&importing, "import std/materials/basic.sim;\n").unwrap();

    let resolved = resolve_import_path(
        "std/materials/basic.sim",
        &importing,
        &stdlib,
        tmp.path(), // sandbox root
    ).unwrap();
    assert_eq!(resolved, target.canonicalize().unwrap());
}

#[test]
fn relative_resolves_against_importing_file() {
    let tmp = tempdir().unwrap();
    let local = tmp.path().join("local.sim");
    std::fs::write(&local, "// local\n").unwrap();
    let importing = tmp.path().join("a.sim");
    std::fs::write(&importing, "").unwrap();

    let resolved = resolve_import_path(
        "./local.sim",
        &importing,
        &tmp.path().join("stdlib"), // doesn't need to exist for ./ path
        tmp.path(),
    ).unwrap();
    assert_eq!(resolved, local.canonicalize().unwrap());
}

#[test]
fn parent_traversal_resolves_against_importing_file() {
    let tmp = tempdir().unwrap();
    let sub = tmp.path().join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    let shared = tmp.path().join("shared.sim");
    std::fs::write(&shared, "").unwrap();
    let importing = sub.join("a.sim");
    std::fs::write(&importing, "").unwrap();

    let resolved = resolve_import_path(
        "../shared.sim",
        &importing,
        &tmp.path().join("stdlib"),
        tmp.path(),
    ).unwrap();
    assert_eq!(resolved, shared.canonicalize().unwrap());
}

#[test]
fn sandbox_escape_via_parent_is_rejected() {
    // Importing file is in tmp/inner/a.sim, sandbox is tmp/inner,
    // so `../outside.sim` would escape.
    let tmp = tempdir().unwrap();
    let inner = tmp.path().join("inner");
    std::fs::create_dir_all(&inner).unwrap();
    let outside = tmp.path().join("outside.sim");
    std::fs::write(&outside, "").unwrap();
    let importing = inner.join("a.sim");
    std::fs::write(&importing, "").unwrap();

    let err = resolve_import_path(
        "../outside.sim",
        &importing,
        &inner.join("stdlib"),
        &inner, // sandbox is restricted to inner/
    ).err().unwrap();
    assert!(matches!(err, ImportError::FileNotFound { .. }),
            "expected FileNotFound for sandbox escape, got: {err:?}");
}

#[test]
fn missing_file_is_file_not_found() {
    let tmp = tempdir().unwrap();
    let importing = tmp.path().join("a.sim");
    std::fs::write(&importing, "").unwrap();

    let err = resolve_import_path(
        "std/does_not_exist.sim",
        &importing,
        &tmp.path().join("stdlib"),
        tmp.path(),
    ).err().unwrap();
    assert!(matches!(err, ImportError::FileNotFound { .. }), "got: {err:?}");
}
