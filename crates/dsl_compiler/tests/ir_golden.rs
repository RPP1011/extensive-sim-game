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
fn decay_without_materialized_errors() {
    // `@decay` is only valid on `@materialized` fold views.
    let src = r#"
        @decay(rate = 0.98, per = tick)
        view mood(a: Agent) -> f32 { 0.0 }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("@decay") && msg.contains("@materialized"),
        "message: {msg}",
    );
}

#[test]
fn decay_with_out_of_range_rate_errors() {
    let src = r#"
        @materialized(on_event = [AgentAttacked])
        @decay(rate = 1.5, per = tick)
        view threat_level(a: Agent) -> f32 {
            initial: 0.0,
            on AgentAttacked{target: a} { self += 1.0 }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("(0.0, 1.0)"),
        "expected rate-range diagnostic; got: {msg}",
    );
}

#[test]
fn decay_with_unsupported_per_unit_errors() {
    let src = r#"
        @materialized(on_event = [AgentAttacked])
        @decay(rate = 0.98, per = hour)
        view threat_level(a: Agent) -> f32 {
            initial: 0.0,
            on AgentAttacked{target: a} { self += 1.0 }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("hour") && msg.contains("tick"),
        "expected `per` diagnostic; got: {msg}",
    );
}

#[test]
fn udf_in_fold_body_rejected_with_diagnostic() {
    // `another_view(...)` inside a fold body is a cross-view call — forbidden
    // by the fold-body operator-set restriction (spec §2.3).
    let src = r#"
        @materialized(on_event = [AgentAttacked])
        view threat_level(a: Agent) -> f32 {
            initial: 0.0,
            on AgentAttacked{target: a} { self += my_helper(a) }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    match err {
        dsl_compiler::CompileError::Resolve(ResolveError::UdfInViewFoldBody {
            view_name,
            offending_construct,
            ..
        }) => {
            assert_eq!(view_name, "threat_level");
            assert!(
                offending_construct.contains("my_helper"),
                "offending construct should mention the helper name; got: {offending_construct}",
            );
        }
        other => panic!("expected UdfInViewFoldBody; got: {other:?}"),
    }
}

#[test]
fn fold_body_for_loop_rejected() {
    // Unbounded `for` inside a fold body is forbidden by spec §2.3.
    let src = r#"
        @materialized(on_event = [AgentAttacked])
        view mood(a: Agent) -> f32 {
            initial: 0.0,
            on AgentAttacked{target: a} {
                for x in agents { self += 0.1 }
            }
        }
    "#;
    let err = dsl_compiler::compile(src).expect_err("should fail");
    let msg = format!("{err}");
    assert!(msg.contains("for") && msg.contains("mood"), "message: {msg}");
}

#[test]
fn gradient_scoring_modifier_compiles() {
    let src = r#"
        scoring {
          Attack(t) = 0.5
                    + threat_level(self, t) per_unit 0.02
                    + (1.0 - t.hp_pct) per_unit 0.6
        }
    "#;
    let comp = dsl_compiler::compile(src).expect("compile failed");
    assert_eq!(comp.scoring.len(), 1);
    assert_eq!(comp.scoring[0].entries.len(), 1);
    // The entry's expression now contains PerUnit nodes as sibling sum
    // terms. The scoring emitter lifts them into KIND_GRADIENT rows —
    // exercise via the public emit() surface.
    let artefacts = dsl_compiler::emit(&comp);
    // The generated scoring table body references the gradient row.
    assert!(
        artefacts.rust_scoring_mod.contains("KIND_GRADIENT"),
        "emitted scoring_mod should reference KIND_GRADIENT",
    );
}

#[test]
fn decay_hint_lowers_to_decayhint_struct() {
    let src = r#"
        @materialized(on_event = [AgentAttacked])
        @decay(rate = 0.98, per = tick)
        view threat_level(a: Agent) -> f32 {
            initial: 0.0,
            on AgentAttacked{target: a} { self += 1.0 }
        }
    "#;
    let comp = dsl_compiler::compile(src).expect("compile failed");
    let view = comp.views.iter().find(|v| v.name == "threat_level").unwrap();
    let hint = view.decay.expect("decay hint should be populated");
    assert!((hint.rate - 0.98).abs() < 1e-6);
    assert_eq!(hint.per, dsl_compiler::ir::DecayUnit::Tick);
}

#[test]
fn decay_view_emits_anchor_pattern_skeleton() {
    // End-to-end: parse + resolve + emit the anchor-pattern Rust for a
    // `@decay` view. The emitted skeleton defines a struct with
    // `get(a, b, tick)` + `fold_event(...)` and bakes the rate in as
    // a `const`.
    let src = r#"
        @materialized(on_event = [AgentAttacked, EffectDamageApplied])
        @decay(rate = 0.98, per = tick)
        view threat_level(a: Agent, b: Agent) -> f32 {
            initial: 0.0,
            on AgentAttacked{actor: b, target: a} { self += 1.0 }
            on EffectDamageApplied{actor: b, target: a} { self += 1.0 }
            clamp: [0.0, 1000.0],
        }
    "#;
    let comp = dsl_compiler::compile(src).expect("compile failed");
    let view = comp.views.iter().find(|v| v.name == "threat_level").unwrap();
    let out = dsl_compiler::emit_view::emit_decay_view(view)
        .expect("emit_decay_view")
        .expect("decay view should emit a non-None skeleton");
    assert!(out.contains("pub struct ThreatLevel"));
    assert!(out.contains("pub const RATE: f32 = 0.98_f32;"));
    assert!(out.contains("pub fn get(&self, a: i64, b: i64, tick: u32) -> f32"));
    assert!(out.contains("\"AgentAttacked\" =>"));
    assert!(out.contains("\"EffectDamageApplied\" =>"));
    // Clamp is wired into both `get` and `fold_event`.
    assert!(out.contains(".clamp(0.0_f32, 1000.0_f32)"));
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
    // `Broken` declares neither `actor` nor `target`, so claiming `@harmful`
    // must fail the contract check.
    let src = r#"
        event_tag Harmful { actor: AgentId, target: AgentId }
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
