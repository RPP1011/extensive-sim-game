//! Plan G tunable cfg — parser + resolver coverage for the
//! `@runtime` annotation on `config` block fields.
//!
//! Today the annotation only flips a `runtime: bool` flag on the
//! corresponding `ConfigField` / `ConfigFieldIR`; the CG lowering
//! consumes this in a separate test (`dsl_compiler` integration
//! tests). These tests pin the surface grammar:
//!   * accepts `@runtime` as a trailing annotation after the default
//!   * preserves backwards compatibility (no `@runtime` → `runtime = false`)
//!   * rejects `@runtime` with arguments
//!   * rejects unknown annotations on config fields

use dsl_ast::ast::Decl;
use dsl_ast::compile;

fn parse_config(src: &str) -> dsl_ast::Compilation {
    compile(src).unwrap_or_else(|e| panic!("compile failed for source:\n{src}\nerror: {e}"))
}

#[test]
fn config_field_default_is_not_runtime() {
    let src = "\
        config knobs {\n\
          baseline: u32 = 7,\n\
        }\n";
    let comp = parse_config(src);
    assert_eq!(comp.configs.len(), 1, "expected one config block");
    let block = &comp.configs[0];
    assert_eq!(block.fields.len(), 1);
    assert_eq!(block.fields[0].name, "baseline");
    assert!(
        !block.fields[0].runtime,
        "field without @runtime annotation should default to runtime=false",
    );
}

#[test]
fn config_field_runtime_annotation_is_recognized() {
    let src = "\
        config knobs {\n\
          mask: u32 = 15 @runtime,\n\
          inject_amount: f32 = 10.0,\n\
        }\n";
    let comp = parse_config(src);
    assert_eq!(comp.configs.len(), 1);
    let block = &comp.configs[0];
    assert_eq!(block.fields.len(), 2);
    assert_eq!(block.fields[0].name, "mask");
    assert!(
        block.fields[0].runtime,
        "field with @runtime annotation should set runtime=true",
    );
    assert_eq!(block.fields[1].name, "inject_amount");
    assert!(
        !block.fields[1].runtime,
        "sibling field without @runtime should stay runtime=false",
    );
}

#[test]
fn config_field_runtime_annotation_works_at_end_of_block() {
    // No trailing comma; @runtime sits between the default and the
    // closing brace. Exercises the lookahead-roll-back path in the
    // parser.
    let src = "\
        config knobs {\n\
          mask: u32 = 15 @runtime\n\
        }\n";
    let comp = parse_config(src);
    let block = &comp.configs[0];
    assert!(block.fields[0].runtime);
}

#[test]
fn config_field_runtime_annotation_with_args_rejected() {
    // `@runtime(...)` should fail to parse — the annotation takes no
    // arguments today.
    let src = "\
        config knobs {\n\
          mask: u32 = 15 @runtime(extra=1),\n\
        }\n";
    let err = compile(src).err().expect("expected parse error for @runtime with args");
    let msg = err.to_string();
    assert!(
        msg.contains("@runtime") && msg.contains("no arguments"),
        "expected @runtime-takes-no-arguments error, got: {msg}",
    );
}

#[test]
fn config_field_unknown_annotation_rejected() {
    let src = "\
        config knobs {\n\
          mask: u32 = 15 @bogus,\n\
        }\n";
    let err = compile(src).err().expect("expected parse error for @bogus");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown annotation") && msg.contains("bogus"),
        "expected unknown-annotation error, got: {msg}",
    );
}

#[test]
fn config_decl_round_trip_contains_runtime_field() {
    // Sanity check: walking the parsed Program (pre-resolve) also
    // surfaces the runtime flag on the AST `ConfigField`.
    let src = "\
        config knobs {\n\
          mask: u32 = 15 @runtime,\n\
        }\n";
    let prog = dsl_ast::parser::parse_program(src).expect("parse");
    let cfg = prog
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::Config(c) => Some(c),
            _ => None,
        })
        .expect("config decl present");
    assert!(cfg.fields[0].runtime);
}
