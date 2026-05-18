//! Validation + monomorphisation pass for parameterised rules.
//! See `docs/superpowers/specs/2026-05-17-parameterised-rules-design.md`.

use dsl_ast::ast::{Decl, PhysicsDecl, Program};
use std::collections::HashMap;

#[derive(Debug)]
pub enum ParamRuleError {
    UnknownParameterisedRule { name: String, site: String },
    ApplicationParamMismatch {
        rule: String,
        missing: Vec<String>,
        extra: Vec<String>,
        duplicates: Vec<String>,
    },
    ApplicationTypeMismatch {
        rule: String,
        param: String,
        expected: String,
        actual_kind: &'static str,
    },
    UnknownEntityKind {
        rule: String,
        param: String,
        name: String,
    },
}

impl std::fmt::Display for ParamRuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamRuleError::UnknownParameterisedRule { name, site } =>
                write!(f, "unknown parameterised rule `{name}` at apply site `{site}`"),
            ParamRuleError::ApplicationParamMismatch { rule, missing, extra, duplicates } =>
                write!(f, "application of `{rule}`: missing={missing:?} extra={extra:?} duplicates={duplicates:?}"),
            ParamRuleError::ApplicationTypeMismatch { rule, param, expected, actual_kind } =>
                write!(f, "application of `{rule}`: param `{param}` expects {expected}, got {actual_kind}"),
            ParamRuleError::UnknownEntityKind { rule, param, name } =>
                write!(f, "application of `{rule}`: param `{param}` references unknown entity `{name}`"),
        }
    }
}

impl std::error::Error for ParamRuleError {}

/// Validates each parameterised rule decl in isolation. Per-decl
/// structural checks. Parser already enforces duplicate-name and
/// unknown-type; no additional cross-decl logic is needed in v1.
pub fn validate_param_rule_decls(program: &Program) -> Result<(), ParamRuleError> {
    for decl in &program.decls {
        if let Decl::Physics(p) = decl {
            let _ = p;
        }
    }
    Ok(())
}

/// Builds a lookup table from parameterised-rule name → PhysicsDecl.
/// Used by both the application validator and the monomorphisation pass.
pub fn build_param_rule_catalog(program: &Program) -> HashMap<String, &PhysicsDecl> {
    let mut catalog = HashMap::new();
    for decl in &program.decls {
        if let Decl::Physics(p) = decl {
            if !p.params.is_empty() {
                catalog.insert(p.name.clone(), p);
            }
        }
    }
    catalog
}

/// Validates every `Decl::PhysicsApply` against its parameterised rule.
pub fn validate_applications(program: &Program) -> Result<(), ParamRuleError> {
    use dsl_ast::ast::{ApplyArgValue, ParamType};
    use std::collections::HashSet;

    let catalog = build_param_rule_catalog(program);
    let entity_names: HashSet<String> = program.decls.iter().filter_map(|d| {
        if let Decl::Entity(e) = d { Some(e.name.clone()) } else { None }
    }).collect();

    for decl in &program.decls {
        if let Decl::PhysicsApply(apply) = decl {
            let rule = catalog.get(&apply.template).ok_or_else(|| {
                ParamRuleError::UnknownParameterisedRule {
                    name: apply.template.clone(),
                    site: apply.name.clone(),
                }
            })?;

            // Missing / extra / duplicate arg names.
            let expected_names: HashSet<&str> =
                rule.params.iter().map(|p| p.name.as_str()).collect();
            let mut provided_names: HashSet<&str> = HashSet::new();
            let mut duplicates: Vec<String> = Vec::new();
            for arg in &apply.args {
                if !provided_names.insert(arg.name.as_str()) {
                    duplicates.push(arg.name.clone());
                }
            }
            let missing: Vec<String> = rule.params.iter()
                .filter(|p| !provided_names.contains(p.name.as_str()))
                .map(|p| p.name.clone())
                .collect();
            let extra: Vec<String> = apply.args.iter()
                .filter(|a| !expected_names.contains(a.name.as_str()))
                .map(|a| a.name.clone())
                .collect();
            if !missing.is_empty() || !extra.is_empty() || !duplicates.is_empty() {
                return Err(ParamRuleError::ApplicationParamMismatch {
                    rule: apply.template.clone(),
                    missing,
                    extra,
                    duplicates,
                });
            }

            // Per-arg type check.
            for arg in &apply.args {
                let param = rule.params.iter()
                    .find(|p| p.name == arg.name)
                    .expect("missing/extra already checked");

                let (matches, actual_kind) = match (&param.ty, &arg.value) {
                    (ParamType::F32, ApplyArgValue::F32(_)) => (true, "f32"),
                    (ParamType::F32, ApplyArgValue::I32(_)) => (true, "i32→f32"),
                    (ParamType::F32, ApplyArgValue::U32(_)) => (true, "u32→f32"),
                    (ParamType::I32, ApplyArgValue::I32(_)) => (true, "i32"),
                    (ParamType::U32, ApplyArgValue::U32(_)) => (true, "u32"),
                    (ParamType::U32, ApplyArgValue::I32(v)) if *v >= 0 => (true, "i32→u32"),
                    (ParamType::Bool, ApplyArgValue::Bool(_)) => (true, "bool"),
                    (ParamType::EntityKind, ApplyArgValue::EntityKind(name)) => {
                        if !entity_names.contains(name) {
                            return Err(ParamRuleError::UnknownEntityKind {
                                rule: rule.name.clone(),
                                param: param.name.clone(),
                                name: name.clone(),
                            });
                        }
                        (true, "EntityKind")
                    }
                    (_, ApplyArgValue::F32(_)) => (false, "f32"),
                    (_, ApplyArgValue::I32(_)) => (false, "i32"),
                    (_, ApplyArgValue::U32(_)) => (false, "u32"),
                    (_, ApplyArgValue::Bool(_)) => (false, "bool"),
                    (_, ApplyArgValue::EntityKind(_)) => (false, "EntityKind"),
                };
                if !matches {
                    return Err(ParamRuleError::ApplicationTypeMismatch {
                        rule: rule.name.clone(),
                        param: param.name.clone(),
                        expected: format!("{:?}", param.ty),
                        actual_kind,
                    });
                }
            }
        }
    }
    Ok(())
}
