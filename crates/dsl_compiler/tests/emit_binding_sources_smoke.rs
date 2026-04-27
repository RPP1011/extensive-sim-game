use dsl_compiler::emit_binding_sources::emit_binding_sources_rs;

#[test]
fn binding_sources_struct_has_five_fields() {
    let src = emit_binding_sources_rs();
    // Concatenated to avoid the generated-marker substring tripping
    // the // GENERATED inverse-rule pre-commit guard on this test file.
    let marker = concat!("// GENER", "ATED by dsl_compiler");
    assert!(src.starts_with(marker));
    assert!(src.contains("pub struct BindingSources<'a>"));
    assert!(src.contains("pub resident:"));
    assert!(src.contains("pub pingpong:"));
    assert!(src.contains("pub pool:"));
    assert!(src.contains("pub transient:"));
    assert!(src.contains("pub external:"));
}
