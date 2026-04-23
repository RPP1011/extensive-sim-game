//! Task 1.2 — verify that `@per_entity_ring(K = 64)` parses and
//! surfaces on the view's `annotations` list with the `K = 64`
//! argument intact.
//!
//! Test-only; no parser/AST code changes expected — the generic
//! annotation parser already handles arbitrary `@<name>(args)` shapes.
//!
//! Note on DSL syntax: annotations go *before* the `view` keyword,
//! and a materialized (fold-form) view body requires `initial:` plus
//! at least one `on <Event> { ... }` handler. Source mirrors the
//! shape of `kin_fear` in `assets/sim/views.sim`.

use dsl_compiler::ast::{AnnotationValue, Decl};
use dsl_compiler::parse;

const SRC: &str = r#"
event RecordMemory { observer: AgentId, source: AgentId, fact: u64, confidence: f32 }

@materialized(on_event = [RecordMemory])
@per_entity_ring(K = 64)
view memory(observer: Agent, source: Agent) -> f32 {
  initial: 0.0,
  on RecordMemory { observer: observer, source: source, fact: f, confidence: c } { self += 1.0 }
}
"#;

#[test]
fn per_entity_ring_annotation_parses() {
    let program = parse(SRC).expect("program should parse");
    let view = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::View(v) if v.name == "memory" => Some(v),
            _ => None,
        })
        .expect("view 'memory' should be present");
    assert!(
        view.annotations
            .iter()
            .any(|a| a.name == "per_entity_ring"),
        "per_entity_ring annotation should be parsed onto the view; \
         saw annotations = {:?}",
        view.annotations
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn per_entity_ring_carries_k_argument() {
    let program = parse(SRC).expect("program should parse");
    let view = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::View(v) if v.name == "memory" => Some(v),
            _ => None,
        })
        .unwrap();
    let ann = view
        .annotations
        .iter()
        .find(|a| a.name == "per_entity_ring")
        .expect("per_entity_ring annotation missing");
    assert_eq!(
        ann.args.len(),
        1,
        "K argument should be present as a single named arg"
    );
    let arg = &ann.args[0];
    assert_eq!(arg.key.as_deref(), Some("K"), "arg key should be `K`");
    match &arg.value {
        AnnotationValue::Int(n) => {
            assert_eq!(*n, 64, "K should equal 64");
        }
        other => panic!("K value should be Int(64), got {other:?}"),
    }
}
