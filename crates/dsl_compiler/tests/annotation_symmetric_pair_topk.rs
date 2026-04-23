//! Task 1.1 — verify that `@symmetric_pair_topk(K = 8)` parses and
//! surfaces on the view's `annotations` list with the `K = 8` argument
//! intact.
//!
//! Test-only; no parser/AST code changes expected — the generic
//! annotation parser already handles arbitrary `@<name>(args)` shapes
//! (same pattern `@cpu_only` and `@decay` ride through).
//!
//! Note on DSL syntax: annotations are written *before* the `view`
//! keyword, and a materialized (fold-form) view requires `initial:`
//! plus at least one `on <Event> { ... }` handler. The source below
//! follows the shape of the existing `kin_fear` view in
//! `assets/sim/views.sim`.

use dsl_compiler::ast::{AnnotationValue, Decl};
use dsl_compiler::ir::{StorageHint, ViewKind};
use dsl_compiler::parse;

const SRC: &str = r#"
event StandingDelta { a: AgentId, b: AgentId, delta: i16 }

@materialized(on_event = [StandingDelta])
@symmetric_pair_topk(K = 8)
view standing(a: Agent, b: Agent) -> f32 {
  initial: 0.0,
  on StandingDelta { a: a, b: b, delta: d } { self += 1.0 }
}
"#;

#[test]
fn symmetric_pair_topk_annotation_parses() {
    let program = parse(SRC).expect("program should parse");
    let view = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::View(v) if v.name == "standing" => Some(v),
            _ => None,
        })
        .expect("view 'standing' should be present");
    assert!(
        view.annotations
            .iter()
            .any(|a| a.name == "symmetric_pair_topk"),
        "symmetric_pair_topk annotation should be parsed onto the view; \
         saw annotations = {:?}",
        view.annotations
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn symmetric_pair_topk_carries_k_argument() {
    let program = parse(SRC).expect("program should parse");
    let view = program
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::View(v) if v.name == "standing" => Some(v),
            _ => None,
        })
        .unwrap();
    let ann = view
        .annotations
        .iter()
        .find(|a| a.name == "symmetric_pair_topk")
        .expect("symmetric_pair_topk annotation missing");
    assert_eq!(
        ann.args.len(),
        1,
        "K argument should be present as a single named arg"
    );
    let arg = &ann.args[0];
    assert_eq!(arg.key.as_deref(), Some("K"), "arg key should be `K`");
    match &arg.value {
        AnnotationValue::Int(n) => {
            assert_eq!(*n, 8, "K should equal 8");
        }
        other => panic!("K value should be Int(8), got {other:?}"),
    }
}

/// Task 1.3 — `@symmetric_pair_topk(K = 8)` lowers to the matching IR
/// `StorageHint::SymmetricPairTopK { k: 8 }` variant. The resolver
/// converts the annotation into the typed view storage hint so
/// downstream emitters (tasks 1.5-1.8) can dispatch on shape.
#[test]
fn symmetric_pair_topk_lowers_to_ir_variant() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR should exist");
    match view.kind {
        ViewKind::Materialized(StorageHint::SymmetricPairTopK { k }) => {
            assert_eq!(k, 8, "K should equal 8")
        }
        other => panic!("expected Materialized(SymmetricPairTopK {{ k: 8 }}), got {other:?}"),
    }
}
