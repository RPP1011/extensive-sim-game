use dsl_compiler::parse;

#[test]
fn rejects_import_after_agent_decl() {
    let src = r#"
entity Wolf : Agent { }
import std/materials/basic.sim;
"#;
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("import") && msg.contains("after"),
        "expected ImportAfterDecl error, got: {msg}"
    );
}
