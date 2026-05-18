#[allow(unused_imports)]
use dsl_compiler::{parse, lower::param_rules::{validate_param_rule_decls, ParamRuleError}};

fn program_from(src: &str) -> dsl_ast::ast::Program {
    parse(src).expect("parse")
}

#[test]
fn valid_param_rule_decl_passes_validation() {
    let p = program_from(r#"
physics chase(target: EntityKind, aggro: f32) @phase(per_agent) {
  on Tick {} {}
}
"#);
    validate_param_rule_decls(&p).expect("should pass");
}

#[test]
fn empty_program_passes_validation() {
    let p = program_from("");
    validate_param_rule_decls(&p).expect("empty program is valid");
}

#[test]
fn duplicate_rule_names_across_decls_caught_by_existing_collision_pass_not_here() {
    let p = program_from(r#"
physics chase(target: EntityKind) @phase(per_agent) {
  on Tick {} {}
}
"#);
    validate_param_rule_decls(&p).expect("single decl always passes");
}
