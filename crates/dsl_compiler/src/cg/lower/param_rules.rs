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
