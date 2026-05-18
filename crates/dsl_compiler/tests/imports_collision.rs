use dsl_compiler::{parse_with_imports, ImportError};
use tempfile::tempdir;

#[test]
fn duplicate_entity_across_files_is_error() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&b, "entity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nentity Wolf : Agent { hp: 99.0 }\n").unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    match err {
        ImportError::DuplicateDefinition { kind, name, first_seen_at, second_seen_at } => {
            assert_eq!(kind, "entity");
            assert_eq!(name, "Wolf");
            // depth-first post-order: b.sim is merged first (imported), then a.sim's own decl.
            assert_eq!(first_seen_at.file_name().unwrap(), "b.sim");
            assert_eq!(second_seen_at.file_name().unwrap(), "a.sim");
        }
        other => panic!("expected DuplicateDefinition, got: {other:?}"),
    }
}

#[test]
fn different_kinds_same_name_does_not_collide() {
    // entity Wolf and event Wolf live in different kind-namespaces;
    // both can coexist in the merged Program without error.
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&b, "event Wolf {}\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nentity Wolf : Agent { hp: 10.0 }\n").unwrap();
    // Should succeed — kind-namespaced collision check lets these coexist.
    let program = parse_with_imports(&a, &stdlib, tmp.path()).expect("kind-scoped collision allows entity+event with same name");
    assert_eq!(program.decls.len(), 2);
}

#[test]
fn two_terrain_blocks_via_imports_is_error() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let materials_block = r#"
terrain {
  extent: 4
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: grass }
}
"#;
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&b, materials_block).unwrap();
    std::fs::write(&a, format!("import ./b.sim;\n{materials_block}")).unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    assert!(matches!(err, ImportError::DuplicateDefinition { ref kind, .. } if kind == "terrain"), "got: {err:?}");
}
