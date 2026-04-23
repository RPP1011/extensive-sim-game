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
    "mask_coverage",
    "scoring_attack",
    "scoring_full_coverage",
    "scoring_gradient",
    "scoring_pattern",
    "view_mood",
    "view_decay",
    "verb_pray",
    "invariant_no_bigamy",
    "probe_low_hp_flees",
    "metric_cascade_iters",
    "for_filter",
    "trailing_annotation",
    "enum_decl",
    "event_tag_decl",
    "physics_tagged",
    "tick_implicit",
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

#[test]
fn trailing_annotation_attaches_to_preceding_decl() {
    use dsl_compiler::ast::Decl;
    let src = "\
event A { x: i32 } @replayable
event B { y: f32 } @replayable @traced
@replayable event C { z: u64 }
@replayable event D { a: bool } @traced
";
    let program = dsl_compiler::parse(src).expect("parse failed");
    let names: Vec<&str> = program
        .decls
        .iter()
        .map(|d| match d {
            Decl::Event(e) => e.name.as_str(),
            _ => panic!("expected only events"),
        })
        .collect();
    assert_eq!(names, vec!["A", "B", "C", "D"]);

    let anns_for = |idx: usize| -> Vec<&str> {
        match &program.decls[idx] {
            Decl::Event(e) => e.annotations.iter().map(|a| a.name.as_str()).collect(),
            _ => unreachable!(),
        }
    };
    assert_eq!(anns_for(0), vec!["replayable"]);
    assert_eq!(anns_for(1), vec!["replayable", "traced"]);
    assert_eq!(anns_for(2), vec!["replayable"]);
    // Mixed: leading first, then trailing.
    assert_eq!(anns_for(3), vec!["replayable", "traced"]);
}

#[test]
fn orphan_trailing_annotation_at_eof_errors() {
    let src = "@replayable\n";
    let err = dsl_compiler::parse(src).expect_err("should fail");
    assert!(
        err.message.contains("top-level declaration"),
        "message: {}",
        err.message
    );
}

#[test]
fn cpu_only_annotation_parses_on_physics_rule() {
    use dsl_compiler::ast::Decl;
    let src = r#"
event AgentDied { agent_id: AgentId }

@cpu_only physics narrative_rule @phase(event) {
    on AgentDied { agent_id: a } { }
}
"#;
    let program = dsl_compiler::parse(src).expect("should parse");
    let rule = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::Physics(p) if p.name == "narrative_rule" => Some(p),
            _ => None,
        })
        .expect("rule not found in parsed program");
    assert!(rule.cpu_only, "cpu_only flag should be set on narrative_rule");
}

#[test]
fn rule_without_cpu_only_defaults_to_false() {
    use dsl_compiler::ast::Decl;
    let src = r#"
event AgentDied { agent_id: AgentId }

physics gpu_rule @phase(event) {
    on AgentDied { agent_id: a } { }
}
"#;
    let program = dsl_compiler::parse(src).expect("should parse");
    let rule = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::Physics(p) if p.name == "gpu_rule" => Some(p),
            _ => None,
        })
        .expect("rule not found");
    assert!(!rule.cpu_only, "cpu_only should default to false");
}

#[test]
fn cpu_only_flag_flows_through_ir_lowering() {
    // Task 2.13 — the parser sets `cpu_only` on the AST `PhysicsDecl`; the
    // IR lowering in `resolve.rs` must copy the flag across so downstream
    // emit paths can branch on it. We check both a `@cpu_only` rule (true)
    // and a vanilla rule (false) in the same compilation to lock in both
    // directions.
    let src = r#"
event AgentDied { agent_id: AgentId }

@cpu_only physics narrative_rule @phase(event) {
    on AgentDied { agent_id: a } { }
}

physics gpu_rule @phase(event) {
    on AgentDied { agent_id: a } { }
}
"#;
    let comp = dsl_compiler::compile(src).expect("should compile");
    let narrative = comp
        .physics
        .iter()
        .find(|p| p.name == "narrative_rule")
        .expect("narrative_rule missing from IR");
    assert!(
        narrative.cpu_only,
        "IR should carry cpu_only=true for @cpu_only rule",
    );
    let gpu = comp
        .physics
        .iter()
        .find(|p| p.name == "gpu_rule")
        .expect("gpu_rule missing from IR");
    assert!(
        !gpu.cpu_only,
        "IR should carry cpu_only=false for unannotated rule",
    );
}

#[allow(dead_code)]
fn _unused_path_helper(_: &Path) {}
