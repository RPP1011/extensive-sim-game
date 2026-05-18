use dsl_compiler::{parse_with_imports, ImportError};
use tempfile::tempdir;

#[test]
fn duplicate_apply_name_collision() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let a = tmp.path().join("a.sim");
    // Both define an apply named X.
    std::fs::write(&a, r#"
physics X = chase(a: 1.0);
physics X = chase(a: 2.0);
"#).unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    match err {
        ImportError::DuplicateDefinition { kind, name, .. } => {
            assert_eq!(kind, "physics");
            assert_eq!(name, "X");
        }
        other => panic!("expected DuplicateDefinition, got: {other:?}"),
    }
}

#[test]
fn parameterised_rule_collides_with_concrete_rule() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let a = tmp.path().join("a.sim");
    std::fs::write(&a, r#"
physics foo(x: f32) @phase(per_agent) {
  on Tick {} {}
}
physics foo @phase(per_agent) {
  on Tick {} {}
}
"#).unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    assert!(matches!(err, ImportError::DuplicateDefinition { ref kind, ref name, .. } if kind == "physics" && name == "foo"), "got: {err:?}");
}
