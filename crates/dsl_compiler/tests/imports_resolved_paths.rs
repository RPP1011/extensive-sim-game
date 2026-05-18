use dsl_compiler::parse_with_imports;
use tempfile::tempdir;

#[test]
fn resolved_paths_are_absolute_canonicalised() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&b, "entity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nentity Goblin : Agent { hp: 5.0 }\n").unwrap();

    let program = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    assert!(program.imports_resolved.len() >= 2);
    for p in &program.imports_resolved {
        assert!(p.is_absolute(), "expected absolute path, got {p:?}");
        // Canonicalised: contains no `..` segments.
        assert!(!p.components().any(|c| matches!(c, std::path::Component::ParentDir)),
                "expected canonicalised path, got {p:?}");
    }
    // The top file and the imported file both appear.
    let names: Vec<_> = program.imports_resolved.iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect();
    assert!(names.contains(&"a.sim".to_string()), "a.sim missing: {names:?}");
    assert!(names.contains(&"b.sim".to_string()), "b.sim missing: {names:?}");
}
