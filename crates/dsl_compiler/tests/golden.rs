//! Golden-output tests: for each `.sim` fixture, parse it and compare the
//! serialized AST against a checked-in `.ast.json` file. Regenerate by
//! running with `DSL_GOLDEN_BLESS=1`.

use std::fs;
use std::path::{Path, PathBuf};

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

fn render_ast(source: &str) -> String {
    let program = dsl_compiler::parse(source).expect("parse failed");
    let mut s = serde_json::to_string_pretty(&program).expect("serialize failed");
    s.push('\n');
    s
}

#[test]
fn all_fixtures_parse_cleanly() {
    let dir = fixtures_dir();
    let bless = std::env::var("DSL_GOLDEN_BLESS").is_ok();
    let mut failures: Vec<String> = Vec::new();
    for name in FIXTURES {
        let src_path = dir.join(format!("{name}.sim"));
        let golden_path = dir.join(format!("{name}.ast.json"));
        let src = fs::read_to_string(&src_path)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", src_path.display()));
        let actual = render_ast(&src);
        if bless || !golden_path.exists() {
            fs::write(&golden_path, &actual).expect("write golden");
            continue;
        }
        let expected = fs::read_to_string(&golden_path).expect("read golden");
        if expected != actual {
            let diff_path = dir.join(format!("{name}.ast.json.actual"));
            fs::write(&diff_path, &actual).expect("write diff");
            failures.push(format!(
                "golden mismatch for `{name}`; wrote actual to `{}`",
                diff_path.display()
            ));
        }
    }
    if !failures.is_empty() {
        panic!("{}", failures.join("\n"));
    }
}

#[test]
fn bad_entity_errors_with_span() {
    let src = "entity Wolf : Potato { }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.message.contains("Agent"));
    assert!(err.top_context().contains("entity"));
}

#[test]
fn bad_event_missing_colon_errors() {
    let src = "event Foo { bar f32 }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.top_context().contains("field"));
}

#[test]
fn bad_view_missing_arrow() {
    let src = "view mood(a: Agent) f32 { 0.0 }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.top_context().contains("view"));
}

#[test]
fn bad_physics_missing_pattern() {
    let src = "physics foo @phase(event) { on { } }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(!err.rendered.is_empty());
}

#[test]
fn bad_mask_missing_when() {
    let src = "mask Attack(t) t.alive\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.top_context().contains("mask") || err.message.contains("when"));
}

#[test]
fn bad_verb_missing_action() {
    let src = "verb Pray(self: Agent) = Converse(target: self)\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.top_context().contains("verb") || err.message.contains("action"));
}

#[test]
fn bad_scoring_missing_eq() {
    let src = "scoring { Attack(t) 0.5 }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.top_context().contains("scoring") || err.message.contains("="));
}

#[test]
fn bad_invariant_mode() {
    let src = "invariant foo(a: Agent) @whenever { true }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.message.contains("static") || err.message.contains("runtime"));
}

#[test]
fn bad_probe_unknown_field() {
    let src = "probe Foo { garbagefield 42 }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.message.contains("garbagefield") || err.message.contains("unknown"));
}

#[test]
fn bad_metric_missing_eq() {
    let src = "metric { metric x histogram(foo) }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(!err.rendered.is_empty());
}

#[test]
fn bad_annotation_alternation_rejected() {
    let src = "@backend(cpu | gpu)\nview x() -> f32 { 0.0 }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(
        err.message.contains("| alternation"),
        "message: {}",
        err.message
    );
}

#[test]
fn error_rendering_includes_caret() {
    let src = "entity Wolf : Potato { }\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(err.rendered.contains("line 1"), "rendered: {}", err.rendered);
    assert!(err.rendered.contains("^"), "rendered: {}", err.rendered);
}

#[allow(dead_code)]
fn _unused_path_helper(_: &Path) {}
