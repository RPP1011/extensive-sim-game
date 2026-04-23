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
use dsl_compiler::ir::{StorageHint, ViewKind};
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

/// Task 1.4 — `@per_entity_ring(K = 64)` lowers to the matching IR
/// `StorageHint::PerEntityRing { k: 64 }` variant. The resolver
/// converts the annotation into the typed view storage hint so
/// downstream emitters (tasks 1.5-1.8) can dispatch on shape.
#[test]
fn per_entity_ring_lowers_to_ir_variant() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "memory")
        .expect("view IR should exist");
    match view.kind {
        ViewKind::Materialized(StorageHint::PerEntityRing { k }) => {
            assert_eq!(k, 64, "K should equal 64")
        }
        other => panic!("expected Materialized(PerEntityRing {{ k: 64 }}), got {other:?}"),
    }
}

/// Task 1.6 — the CPU emitter produces a storage struct + `push` /
/// `fold_event` impl block for `@per_entity_ring` views. Parallels
/// Task 1.5's `@symmetric_pair_topk` emitter; ring is strictly simpler
/// (FIFO, no canonicalisation, no |v|-evict).
#[test]
fn per_entity_ring_emits_cpu_storage() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "memory")
        .expect("view IR should exist");
    let rust = dsl_compiler::emit_view::emit_view(view, None).expect("emit should succeed");

    // Storage struct + per-entry slot struct.
    assert!(
        rust.contains("pub struct Memory"),
        "missing struct:\n{rust}"
    );
    assert!(
        rust.contains("pub struct MemoryEntry"),
        "missing slot struct:\n{rust}"
    );
    assert!(
        rust.contains("rings: Vec<[MemoryEntry; 64]>"),
        "storage should be Vec<[Entry; K]>:\n{rust}"
    );
    assert!(
        rust.contains("cursors: Vec<u32>"),
        "cursor array missing:\n{rust}"
    );
    assert!(
        rust.contains("pub const K: usize = 64"),
        "missing K constant:\n{rust}"
    );

    // Public accessors. `push` is the load-bearing writer, the rest are
    // reader conveniences (cursor, get, entries).
    assert!(
        rust.contains("pub fn push(&mut self, observer_raw: u32, entry: MemoryEntry)"),
        "missing push():\n{rust}"
    );
    assert!(
        rust.contains("pub fn fold_event(&mut self, event: &Event, tick: u32)"),
        "missing fold_event():\n{rust}"
    );
    assert!(
        rust.contains("pub fn cursor(&self, observer: AgentId) -> u32"),
        "missing cursor():\n{rust}"
    );
    assert!(
        rust.contains("pub fn get(&self, observer: AgentId) -> f32"),
        "missing get() for latest value:\n{rust}"
    );
}

/// Task 1.6 — the generated `fold_event` routes `RecordMemory` through
/// `self.push`, projecting the event's `observer` field onto the ring
/// owner and `source` onto the entry's `source` handle.
#[test]
fn per_entity_ring_fold_handles_record_memory() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "memory")
        .expect("view IR should exist");
    let rust = dsl_compiler::emit_view::emit_view(view, None).expect("emit should succeed");

    // Arm destructures the RecordMemory event and pushes into the ring.
    assert!(
        rust.contains("Event::RecordMemory { observer, source, .. }"),
        "fold arm should destructure RecordMemory's owner/source fields:\n{rust}"
    );
    assert!(
        rust.contains("self.push(observer.raw()"),
        "fold arm should call push with the observer's raw id:\n{rust}"
    );
    assert!(
        rust.contains("source: source.raw()"),
        "fold arm should populate entry.source from the event's source field:\n{rust}"
    );
    assert!(
        rust.contains("anchor_tick: tick"),
        "fold arm should stamp anchor_tick with the current tick:\n{rust}"
    );
}

/// Task 1.6 — `@decay` on a ring view is rejected with `Unsupported`.
/// The ring semantics (FIFO, no accumulation) don't compose with the
/// anchor-pattern decay; punting to Phase 3.
#[test]
fn per_entity_ring_rejects_decay() {
    const SRC_DECAY: &str = r#"
event RecordMemory { observer: AgentId, source: AgentId, fact: u64, confidence: f32 }

@materialized(on_event = [RecordMemory])
@per_entity_ring(K = 32)
@decay(rate = 0.9, per = tick)
view memory(observer: Agent, source: Agent) -> f32 {
  initial: 0.0,
  on RecordMemory { observer: observer, source: source, fact: f, confidence: c } { self += 1.0 }
}
"#;
    let comp = dsl_compiler::compile(SRC_DECAY).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "memory")
        .expect("view IR should exist");
    let err = dsl_compiler::emit_view::emit_view(view, None).expect_err(
        "decay on per_entity_ring should fail until Phase 3 defines its semantics",
    );
    assert!(
        err.contains("decay"),
        "error should mention `decay`, got: {err}"
    );
}
