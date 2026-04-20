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
