//! Golden-output tests for the IR layer: parse → compile_ast → serialize
//! the `Compilation` to pretty JSON. Committed `.ir.json` files are the
//! expected output. Bless by setting `DSL_IR_BLESS=1`.

use std::fs;
use std::path::PathBuf;

use dsl_compiler::ResolveError;

const FIXTURES: &[&str] = &[
    "empty",
    "event_damage",
    "entity_wolf",
    "physics_damage",
    "mask_attack",
    "scoring_attack",
    "scoring_pattern",
    "view_mood",
    "verb_pray",
    "invariant_no_bigamy",
    "probe_low_hp_flees",
    "metric_cascade_iters",
    "for_filter",
    "stdlib_usage",
    "trailing_annotation",
    "enum_decl",
    "event_tag_decl",
    "physics_tagged",
    "tick_implicit",
];

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn render_ir(source: &str) -> String {
    let comp = dsl_compiler::compile(source).expect("compile failed");
    let mut s = serde_json::to_string_pretty(&comp).expect("serialize failed");
    s.push('\n');
    s
}

#[test]
fn all_fixtures_compile_to_ir() {
    let dir = fixtures_dir();
    let bless = std::env::var("DSL_IR_BLESS").is_ok();
    let mut failures: Vec<String> = Vec::new();
    for name in FIXTURES {
        let src_path = dir.join(format!("{name}.sim"));
        let golden_path = dir.join(format!("{name}.ir.json"));
        let src = fs::read_to_string(&src_path)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", src_path.display()));
        let actual = render_ir(&src);
        if bless || !golden_path.exists() {
            fs::write(&golden_path, &actual).expect("write golden");
            continue;
        }
        let expected = fs::read_to_string(&golden_path).expect("read golden");
        if expected != actual {
            let diff_path = dir.join(format!("{name}.ir.json.actual"));
            fs::write(&diff_path, &actual).expect("write diff");
            failures.push(format!(
                "ir golden mismatch for `{name}`; wrote actual to `{}`",
                diff_path.display()
            ));
        }
    }
    if !failures.is_empty() {
        panic!("{}", failures.join("\n"));
    }
}

#[test]
fn explicit_tick_field_is_a_parse_error() {
    let src = r#"
        event Bad { target: AgentId, tick: u32 }
    "#;
    let err = dsl_compiler::parse(src).expect_err("should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("tick is implicit"),
        "expected implicit-tick diagnostic, got: {msg}"
    );
}

#[test]
fn event_tag_contract_violation_errors() {
    let src = r#"
        event_tag Harmful { target: AgentId }
        @harmful event Broken { caster: AgentId }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    match err {
        dsl_compiler::CompileError::Resolve(ResolveError::EventTagContractViolated {
            ref event,
            ref tag,
            ..
        }) => {
            assert_eq!(event, "Broken");
            assert_eq!(tag, "Harmful");
        }
        other => panic!("expected EventTagContractViolated, got {other:?}"),
    }
}

#[test]
fn enum_variant_resolves_to_declared_enum() {
    let src = r#"
        enum Flavour { Salty, Sweet }
        view pick() -> bool { Flavour::Salty == Flavour::Sweet }
    "#;
    let comp = dsl_compiler::compile(src).expect("compile");
    assert_eq!(comp.enums.len(), 1);
    assert_eq!(comp.enums[0].variants, vec!["Salty".to_string(), "Sweet".to_string()]);
}

#[test]
fn duplicate_event_decl_errors() {
    let src = r#"
        event Damage { target: AgentId, amount: f32 }
        event Damage { target: AgentId, amount: f32 }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    match err {
        dsl_compiler::CompileError::Resolve(ResolveError::DuplicateDecl { kind, ref name, .. }) => {
            assert_eq!(kind, "event");
            assert_eq!(name, "Damage");
        }
        other => panic!("expected DuplicateDecl(event), got {other:?}"),
    }
}

#[test]
fn unknown_event_in_emit_becomes_unresolved() {
    // `NonexistentEvent` is not declared. The emit statement references it.
    // Resolution should succeed (emit event refs are allowed to be unresolved
    // at 1a), but `IrEmit.event` is `None`.
    let src = r#"
        physics foo @phase(event) @terminating_in(1) {
            on Tick{t: _} {
                emit NonexistentEvent { value: 1 }
            }
        }
    "#;
    let comp = dsl_compiler::compile(src).expect("compile");
    let phys = &comp.physics[0];
    let handler = &phys.handlers[0];
    match &handler.body[0] {
        dsl_compiler::ir::IrStmt::Emit(e) => {
            assert_eq!(e.event_name, "NonexistentEvent");
            assert!(e.event.is_none(), "expected unresolved emit");
        }
        other => panic!("expected emit, got {other:?}"),
    }
}

#[test]
fn self_in_metric_body_errors() {
    // Metrics have no implicit `self` — referring to it must be rejected.
    let src = r#"
        metric {
            metric bad = self.hp
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    match err {
        dsl_compiler::CompileError::Resolve(ResolveError::SelfInTopLevel { .. }) => {}
        other => panic!("expected SelfInTopLevel, got {other:?}"),
    }
}

#[test]
fn unknown_lowercase_ident_errors() {
    // Bare unknown lowercase ident (not a builtin, not a local) is an error.
    let src = r#"
        view check() -> bool {
            notarealthing < 1.0
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    match err {
        dsl_compiler::CompileError::Resolve(ResolveError::UnknownIdent { ref name, .. }) => {
            assert_eq!(name, "notarealthing");
        }
        other => panic!("expected UnknownIdent, got {other:?}"),
    }
}

#[test]
fn game_view_call_stays_unresolved() {
    // `is_hostile` is a game-level derivation, NOT part of the Rust-backed
    // stdlib. Without a user `view is_hostile(...)` declaration it must stay
    // UnresolvedCall (1a contract). A later validation pass (1b) turns this
    // into a hard error.
    let src = r#"
        mask Attack(t) when t.alive && is_hostile(self, t)
    "#;
    let comp = dsl_compiler::compile(src).expect("compile");
    let pred = &comp.masks[0].predicate.kind;
    let mut found_is_hostile = false;
    visit_unresolved(pred, &mut |name, _args| {
        if name == "is_hostile" {
            found_is_hostile = true;
        }
    });
    assert!(
        found_is_hostile,
        "expected `is_hostile` to remain an UnresolvedCall in the mask predicate IR: {pred:?}"
    );
}

fn visit_unresolved(
    kind: &dsl_compiler::ir::IrExpr,
    f: &mut dyn FnMut(&str, &[dsl_compiler::ir::IrCallArg]),
) {
    use dsl_compiler::ir::IrExpr::*;
    match kind {
        UnresolvedCall(name, args) => f(name, args),
        Binary(_, l, r) => {
            visit_unresolved(&l.kind, f);
            visit_unresolved(&r.kind, f);
        }
        Unary(_, r) => visit_unresolved(&r.kind, f),
        Field { base, .. } => visit_unresolved(&base.kind, f),
        Index(a, b) => {
            visit_unresolved(&a.kind, f);
            visit_unresolved(&b.kind, f);
        }
        BuiltinCall(_, args) | ViewCall(_, args) | VerbCall(_, args) => {
            for a in args {
                visit_unresolved(&a.value.kind, f);
            }
        }
        NamespaceCall { args, .. } => {
            for a in args {
                visit_unresolved(&a.value.kind, f);
            }
        }
        _ => {}
    }
}
