use dsl_compiler::{parse_with_imports, ImportError};
use tempfile::tempdir;

#[test]
fn direct_self_import_is_cycle() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let a = tmp.path().join("a.sim");
    std::fs::write(&a, "import ./a.sim;\n").unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    assert!(matches!(err, ImportError::Cycle { .. }), "got: {err:?}");
}

#[test]
fn indirect_cycle_a_b_a() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let a = tmp.path().join("a.sim");
    let b = tmp.path().join("b.sim");
    std::fs::write(&a, "import ./b.sim;\n").unwrap();
    std::fs::write(&b, "import ./a.sim;\n").unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    match err {
        ImportError::Cycle { path_chain } => {
            assert!(path_chain.len() >= 3, "chain should be a -> b -> a; got: {path_chain:?}");
        }
        other => panic!("expected Cycle, got: {other:?}"),
    }
}
