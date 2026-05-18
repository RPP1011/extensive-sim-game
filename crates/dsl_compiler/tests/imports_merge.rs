use dsl_compiler::parse_with_imports;
use tempfile::tempdir;

#[test]
fn import_free_file_equivalent_to_parse() {
    // Regression: a .sim file with zero imports must produce the same
    // Program shape via parse_with_imports as via parse(src).
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let a = tmp.path().join("a.sim");
    let src = "entity Wolf : Agent { hp: 10.0 }\nentity Goblin : Agent { hp: 5.0 }\n";
    std::fs::write(&a, src).unwrap();

    let via_pwi = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    let via_pure = dsl_compiler::parse(src).unwrap();

    assert_eq!(via_pwi.decls.len(), via_pure.decls.len());
    assert_eq!(via_pwi.terrain, via_pure.terrain);
    // T8: imports_resolved assertions added when Program gains that field.
}

#[test]
fn two_file_import_merges_decls() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");

    std::fs::create_dir_all(&stdlib).unwrap();
    std::fs::write(&b, "entity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\n\nentity Goblin : Agent { hp: 5.0 }\n").unwrap();

    let program = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    // Imports flatten: b's decls before a's decls.
    assert_eq!(program.decls.len(), 2);
    // First decl is from b (Wolf), second from a (Goblin).
    let debug_first = format!("{:?}", program.decls[0]);
    let debug_second = format!("{:?}", program.decls[1]);
    assert!(debug_first.contains("Wolf"), "first should be Wolf, got: {debug_first}");
    assert!(debug_second.contains("Goblin"), "second should be Goblin, got: {debug_second}");
}

#[test]
fn transitive_import_flattens_depth_first() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let c = tmp.path().join("c.sim");
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&c, "entity Rat : Agent { hp: 1.0 }\n").unwrap();
    std::fs::write(&b, "import ./c.sim;\nentity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nentity Goblin : Agent { hp: 5.0 }\n").unwrap();

    let program = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    assert_eq!(program.decls.len(), 3);
    let debugs: Vec<String> = program.decls.iter().map(|d| format!("{:?}", d)).collect();
    assert!(debugs[0].contains("Rat"),    "expected Rat first, got: {}", debugs[0]);
    assert!(debugs[1].contains("Wolf"),   "expected Wolf second, got: {}", debugs[1]);
    assert!(debugs[2].contains("Goblin"), "expected Goblin third, got: {}", debugs[2]);
}

#[test]
fn diamond_import_loads_each_file_once() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let d = tmp.path().join("d.sim");
    let b = tmp.path().join("b.sim");
    let c = tmp.path().join("c.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&d, "entity Rat : Agent { hp: 1.0 }\n").unwrap();
    std::fs::write(&b, "import ./d.sim;\nentity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&c, "import ./d.sim;\nentity Bear : Agent { hp: 20.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nimport ./c.sim;\nentity Goblin : Agent { hp: 5.0 }\n").unwrap();

    let program = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    // D contributes Rat exactly once; B contributes Wolf; C contributes Bear; A contributes Goblin.
    assert_eq!(program.decls.len(), 4);
    let rat_count = program.decls.iter().filter(|d| format!("{d:?}").contains("Rat")).count();
    assert_eq!(rat_count, 1, "Rat must appear exactly once in diamond import");
}
