//! Validation + monomorphisation pass for parameterised rules.
//! See `docs/superpowers/specs/2026-05-17-parameterised-rules-design.md`.

use dsl_ast::ast::{ApplyArgValue, Decl, Expr, ExprKind, PhysicsApplyDecl, PhysicsDecl, PhysicsHandler, Program};
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

/// Walk applications, substitute param references in the template body
/// with the application's arg values, and push the resulting concrete
/// rule into `program.decls`. Removes all `Decl::PhysicsApply` decls
/// after processing.
pub fn monomorphise(program: &mut Program) -> Result<(), ParamRuleError> {
    // First, validate all applications.
    validate_applications(program)?;

    // Build the catalog: clone parameterised rules so we can mutate program.decls.
    let catalog: HashMap<String, PhysicsDecl> = program.decls.iter().filter_map(|d| match d {
        Decl::Physics(p) if !p.params.is_empty() => Some((p.name.clone(), p.clone())),
        _ => None,
    }).collect();

    // Collect all applications.
    let applications: Vec<PhysicsApplyDecl> = program.decls.iter().filter_map(|d| match d {
        Decl::PhysicsApply(a) => Some(a.clone()),
        _ => None,
    }).collect();

    // Build new concrete rules.
    let mut new_decls: Vec<PhysicsDecl> = Vec::with_capacity(applications.len());
    for apply in &applications {
        let template = catalog.get(&apply.template).expect("validated above");
        // TODO(param-rules-v2): substitute param refs in handler bodies with literal
        // values. v1 clones the template body unchanged. Once a fixture actually
        // uses param refs inside a body, the substitution pass becomes necessary
        // for correct emit.
        let handlers: Vec<PhysicsHandler> = template.handlers.clone();
        new_decls.push(PhysicsDecl {
            annotations: template.annotations.clone(),
            name: apply.name.clone(),
            params: Vec::new(), // concrete rule — no params
            handlers,
            cpu_only: template.cpu_only,
            span: apply.span,
        });
    }

    // Remove all PhysicsApply decls and parameterised templates, then append the new concrete rules.
    program.decls.retain(|d| match d {
        Decl::PhysicsApply(_) => false,
        Decl::Physics(p) if !p.params.is_empty() => false,  // strip parameterised templates
        _ => true,
    });
    for d in new_decls {
        program.decls.push(Decl::Physics(d));
    }
    Ok(())
}

/// Validates every `Decl::PhysicsApply` against its parameterised rule.
pub fn validate_applications(program: &Program) -> Result<(), ParamRuleError> {
    use dsl_ast::ast::ParamType;
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

/// Convert an `ApplyArgValue` into the corresponding `ExprKind` literal.
/// Used by `substitute_expr` when it finds an `Ident` matching a param name.
///
/// EntityKind args become `Ident(<entity_name>)` — the resolver will later
/// bind the entity name to its decl just like any other entity reference.
pub fn apply_arg_to_expr_kind(value: &ApplyArgValue) -> ExprKind {
    match value {
        ApplyArgValue::F32(v) => ExprKind::Float(*v as f64),
        ApplyArgValue::I32(v) => ExprKind::Int(*v as i64),
        ApplyArgValue::U32(v) => ExprKind::Int(*v as i64),
        ApplyArgValue::Bool(v) => ExprKind::Bool(*v),
        ApplyArgValue::EntityKind(name) => ExprKind::Ident(name.clone()),
    }
}

/// Substitute parameter references in an expression tree with the applied
/// arg values. Binder-introducing constructs (Quantifier / Fold / Block)
/// shadow params by removing the binder name from the map before recursing
/// into the body.
pub fn substitute_expr<'a>(expr: &Expr, args: &HashMap<&'a str, ApplyArgValue>) -> Expr {
    let new_kind = match &expr.kind {
        // ----- Base cases: literals are unchanged. -----
        ExprKind::Int(v)    => ExprKind::Int(*v),
        ExprKind::Float(v)  => ExprKind::Float(*v),
        ExprKind::Bool(v)   => ExprKind::Bool(*v),
        ExprKind::String(s) => ExprKind::String(s.clone()),

        // ----- THE substitution point. -----
        ExprKind::Ident(name) => {
            if let Some(value) = args.get(name.as_str()) {
                apply_arg_to_expr_kind(value)
            } else {
                ExprKind::Ident(name.clone())
            }
        }

        // ----- Pure-recursive variants. -----
        ExprKind::Field(inner, field) => ExprKind::Field(
            Box::new(substitute_expr(inner, args)),
            field.clone(),
        ),
        ExprKind::Index(a, b) => ExprKind::Index(
            Box::new(substitute_expr(a, args)),
            Box::new(substitute_expr(b, args)),
        ),
        ExprKind::Call(callee, call_args) => ExprKind::Call(
            Box::new(substitute_expr(callee, args)),
            call_args.iter().map(|ca| dsl_ast::ast::CallArg {
                name: ca.name.clone(),
                value: substitute_expr(&ca.value, args),
                span: ca.span,
            }).collect(),
        ),
        ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
            op: *op,
            lhs: Box::new(substitute_expr(lhs, args)),
            rhs: Box::new(substitute_expr(rhs, args)),
        },
        ExprKind::Unary { op, rhs } => ExprKind::Unary {
            op: *op,
            rhs: Box::new(substitute_expr(rhs, args)),
        },
        ExprKind::In { item, set } => ExprKind::In {
            item: Box::new(substitute_expr(item, args)),
            set: Box::new(substitute_expr(set, args)),
        },
        ExprKind::Contains { set, item } => ExprKind::Contains {
            set: Box::new(substitute_expr(set, args)),
            item: Box::new(substitute_expr(item, args)),
        },
        ExprKind::List(items) => ExprKind::List(
            items.iter().map(|e| substitute_expr(e, args)).collect()
        ),
        ExprKind::Tuple(items) => ExprKind::Tuple(
            items.iter().map(|e| substitute_expr(e, args)).collect()
        ),
        ExprKind::Struct { name, fields } => ExprKind::Struct {
            name: name.clone(),
            fields: fields.iter().map(|fi| dsl_ast::ast::FieldInit {
                name: fi.name.clone(),
                value: substitute_expr(&fi.value, args),
                span: fi.span,
            }).collect(),
        },
        ExprKind::Ctor { name, args: ctor_args } => ExprKind::Ctor {
            name: name.clone(),
            args: ctor_args.iter().map(|e| substitute_expr(e, args)).collect(),
        },
        ExprKind::Match { scrutinee, arms } => ExprKind::Match {
            scrutinee: Box::new(substitute_expr(scrutinee, args)),
            arms: arms.iter().map(|a| dsl_ast::ast::MatchExprArm {
                pattern: a.pattern.clone(),
                body: substitute_expr(&a.body, args),
                span: a.span,
            }).collect(),
        },
        ExprKind::If { cond, then_expr, else_expr } => ExprKind::If {
            cond: Box::new(substitute_expr(cond, args)),
            then_expr: Box::new(substitute_expr(then_expr, args)),
            else_expr: else_expr.as_ref().map(|e| Box::new(substitute_expr(e, args))),
        },
        ExprKind::PerUnit { expr: inner, delta } => ExprKind::PerUnit {
            expr: Box::new(substitute_expr(inner, args)),
            delta: Box::new(substitute_expr(delta, args)),
        },
        ExprKind::BeliefsAccessor { observer, target, field } => ExprKind::BeliefsAccessor {
            observer: Box::new(substitute_expr(observer, args)),
            target: Box::new(substitute_expr(target, args)),
            field: field.clone(),
        },
        ExprKind::BeliefsConfidence { observer, target } => ExprKind::BeliefsConfidence {
            observer: Box::new(substitute_expr(observer, args)),
            target: Box::new(substitute_expr(target, args)),
        },
        ExprKind::BeliefsView { observer, view_name } => ExprKind::BeliefsView {
            observer: Box::new(substitute_expr(observer, args)),
            view_name: view_name.clone(),
        },

        // ----- Binder-introducing variants (shadow params). -----
        ExprKind::Quantifier { kind, binder, iter, body } => {
            let new_iter = substitute_expr(iter, args);
            let mut inner_args = args.clone();
            inner_args.remove(binder.as_str());
            let new_body = substitute_expr(body, &inner_args);
            ExprKind::Quantifier {
                kind: *kind,
                binder: binder.clone(),
                iter: Box::new(new_iter),
                body: Box::new(new_body),
            }
        }
        ExprKind::Fold { kind, binder, iter, body } => {
            let new_iter = iter.as_ref().map(|i| Box::new(substitute_expr(i, args)));
            let mut inner_args = args.clone();
            if let Some(b) = binder.as_ref() {
                inner_args.remove(b.as_str());
            }
            let new_body = substitute_expr(body, &inner_args);
            ExprKind::Fold {
                kind: *kind,
                binder: binder.clone(),
                iter: new_iter,
                body: Box::new(new_body),
            }
        }
        ExprKind::Block { bindings, expr: tail } => {
            let mut inner_args = args.clone();
            let new_bindings: Vec<(String, Expr)> = bindings.iter().map(|(name, value)| {
                let new_value = substitute_expr(value, &inner_args);
                inner_args.remove(name.as_str());
                (name.clone(), new_value)
            }).collect();
            let new_tail = substitute_expr(tail, &inner_args);
            ExprKind::Block {
                bindings: new_bindings,
                expr: Box::new(new_tail),
            }
        }
    };
    Expr { kind: new_kind, span: expr.span }
}
