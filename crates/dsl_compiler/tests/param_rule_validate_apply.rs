use dsl_compiler::parse;
use dsl_compiler::lower::param_rules::{validate_applications, ParamRuleError};

fn run(src: &str) -> Result<(), ParamRuleError> {
    let program = parse(src).expect("parse");
    validate_applications(&program)
}

#[test]
fn valid_apply_passes() {
    run(r#"
entity Wolf : Agent {}

physics chase(target: EntityKind, aggro: f32) @phase(per_agent) {
  on Tick {} {}
}

physics HunterChase = chase(target: Wolf, aggro: 15.0);
"#).expect("should pass");
}

#[test]
fn unknown_parameterised_rule() {
    let err = run(r#"
physics X = nonexistent(a: 1.0);
"#).err().expect("must fail");
    assert!(matches!(err, ParamRuleError::UnknownParameterisedRule { ref name, .. } if name == "nonexistent"), "got: {err:?}");
}

#[test]
fn missing_arg() {
    let err = run(r#"
entity Wolf : Agent {}
physics chase(target: EntityKind, aggro: f32) @phase(per_agent) { on Tick {} {} }
physics X = chase(target: Wolf);
"#).err().expect("must fail");
    match err {
        ParamRuleError::ApplicationParamMismatch { missing, .. } => {
            assert_eq!(missing, vec!["aggro".to_string()]);
        }
        other => panic!("expected ApplicationParamMismatch, got: {other:?}"),
    }
}

#[test]
fn extra_arg() {
    let err = run(r#"
entity Wolf : Agent {}
physics chase(target: EntityKind) @phase(per_agent) { on Tick {} {} }
physics X = chase(target: Wolf, bogus: 1);
"#).err().expect("must fail");
    match err {
        ParamRuleError::ApplicationParamMismatch { extra, .. } => {
            assert_eq!(extra, vec!["bogus".to_string()]);
        }
        other => panic!("expected ApplicationParamMismatch, got: {other:?}"),
    }
}

#[test]
fn type_mismatch_bool_for_f32() {
    let err = run(r#"
entity Wolf : Agent {}
physics chase(target: EntityKind, aggro: f32) @phase(per_agent) { on Tick {} {} }
physics X = chase(target: Wolf, aggro: true);
"#).err().expect("must fail");
    assert!(matches!(err, ParamRuleError::ApplicationTypeMismatch { ref param, .. } if param == "aggro"), "got: {err:?}");
}

#[test]
fn unknown_entity_kind() {
    let err = run(r#"
physics chase(target: EntityKind, aggro: f32) @phase(per_agent) { on Tick {} {} }
physics X = chase(target: NotAnEntity, aggro: 1.0);
"#).err().expect("must fail");
    assert!(matches!(err, ParamRuleError::UnknownEntityKind { ref name, .. } if name == "NotAnEntity"), "got: {err:?}");
}
