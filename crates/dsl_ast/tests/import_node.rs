use dsl_ast::ast::{Import, Program};

#[test]
fn import_construct_and_field_access() {
    let imp = Import { path: "std/materials/basic.sim".into() };
    assert_eq!(imp.path, "std/materials/basic.sim");
}

#[test]
fn program_grows_imports_field_defaulting_empty() {
    let p = Program { imports: vec![], imports_resolved: vec![], decls: vec![], terrain: None, controls: None, render: None, ui: None };
    assert!(p.imports.is_empty());
}
