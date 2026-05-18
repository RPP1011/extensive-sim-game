use dsl_compiler::parse;

#[test]
fn parses_three_imports() {
    let src = r#"
import std/materials/basic.sim;
import ./local.sim;
import ../shared/foo.sim;

entity Wolf : Agent { }
"#;
    let program = parse(src).expect("parse");
    assert_eq!(program.imports.len(), 3);
    assert_eq!(program.imports[0].path, "std/materials/basic.sim");
    assert_eq!(program.imports[1].path, "./local.sim");
    assert_eq!(program.imports[2].path, "../shared/foo.sim");
}

#[test]
fn parses_zero_imports_when_absent() {
    let src = r#"entity Wolf : Agent { }"#;
    let program = parse(src).expect("parse");
    assert!(program.imports.is_empty());
}

#[test]
fn import_path_requires_dot_sim_suffix() {
    // Required extension keeps imports grep-able.
    let src = "import std/materials/basic;\n";
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("import path must end in `.sim`"), "got: {msg}");
}

#[test]
fn import_must_have_semicolon() {
    let src = "import std/materials/basic.sim\n";
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains(";"), "got: {msg}");
}
