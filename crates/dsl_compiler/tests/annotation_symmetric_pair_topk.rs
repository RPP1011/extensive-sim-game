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

/// Task 1.5 — the CPU emitter produces a storage struct plus a `get` /
/// `adjust` / `fold_event` impl block for `@symmetric_pair_topk` views.
/// This mirrors the `per_entity_topk` emitter's public surface so the
/// engine-side consumers (Phase 3 gold + standing port) can drop the
/// generated view in without hand-written glue.
#[test]
fn symmetric_pair_topk_emits_cpu_storage() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR should exist");
    let rust = dsl_compiler::emit_view::emit_view(view, None).expect("emit should succeed");

    // Storage struct + pair-edge slot struct.
    assert!(
        rust.contains("pub struct Standing"),
        "missing struct:\n{rust}"
    );
    assert!(
        rust.contains("pub struct StandingEdge"),
        "missing slot struct:\n{rust}"
    );
    assert!(
        rust.contains("slots: Vec<[StandingEdge; 8]>"),
        "storage should be Vec<[Edge; K]>:\n{rust}"
    );
    assert!(
        rust.contains("pub const K: usize = 8;"),
        "missing K constant:\n{rust}"
    );

    // Public accessors.
    assert!(
        rust.contains("pub fn get(&self, a: AgentId, b: AgentId) -> f32"),
        "missing get():\n{rust}"
    );
    assert!(
        rust.contains("pub fn adjust(&mut self, a: AgentId, b: AgentId, delta: f32, tick: u32) -> f32"),
        "missing adjust():\n{rust}"
    );
    assert!(
        rust.contains("pub fn fold_event(&mut self, event: &Event, tick: u32)"),
        "missing fold_event():\n{rust}"
    );

    // Fold arm wires the event's pair fields into adjust() with the same
    // +1.0 delta the per_entity_topk(K>=2) emitter uses for its constant
    // fold bodies.
    assert!(
        rust.contains("Event::StandingDelta { a, b, .. }"),
        "fold arm should destructure the StandingDelta event's pair fields:\n{rust}"
    );
    assert!(
        rust.contains("self.adjust(*a, *b, 1.0, tick);"),
        "fold arm should call adjust with +1.0:\n{rust}"
    );
}

/// Task 1.5 — reads and writes canonicalise the pair so `get(a, b) ==
/// get(b, a)`. The generated code either calls an explicit
/// `canonical_pair` helper or performs a `raw()` comparison before
/// indexing the slot array.
#[test]
fn symmetric_pair_topk_canonicalises_pair_reads() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR should exist");
    let rust = dsl_compiler::emit_view::emit_view(view, None).expect("emit should succeed");

    assert!(
        rust.contains("fn canonical_pair")
            || rust.contains("a.raw() <= b.raw()")
            || rust.contains("min("),
        "get/adjust should canonicalise pair order:\n{rust}"
    );
    // The canonical form is invoked from both the reader and the writer.
    assert!(
        rust.matches("Self::canonical_pair").count() >= 2,
        "canonical_pair should be called from both get() and adjust():\n{rust}"
    );
}

/// Task 1.7 — the WGSL emitter produces a fold kernel for
/// `@symmetric_pair_topk` views. The emitted snippet must declare the
/// per-edge struct, the slot + counts bindings, a compile-time `K`
/// constant, and a `@compute`-decorated entry point that canonicalises
/// the pair before folding.
#[test]
fn symmetric_pair_topk_emits_wgsl_fold_kernel() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR should exist");
    let wgsl =
        dsl_compiler::emit_view_wgsl::emit_symmetric_pair_topk_fold_wgsl(view).expect("WGSL emit");

    // Entry point + @compute decoration (one per fold handler; `standing`
    // has a single on-StandingDelta handler).
    assert!(
        wgsl.contains("@compute"),
        "emitted kernel should be @compute-decorated:\n{wgsl}"
    );
    assert!(
        wgsl.contains("fn view_standing_fold_standing_delta"),
        "expected per-handler entry-point name:\n{wgsl}"
    );

    // Per-edge struct mirrors the CPU emitter's `StandingEdge` (other +
    // value + anchor_tick + _pad).
    assert!(
        wgsl.contains("struct StandingEdge"),
        "missing StandingEdge struct:\n{wgsl}"
    );
    assert!(
        wgsl.contains("other:       u32,"),
        "edge struct should declare `other: u32`:\n{wgsl}"
    );
    assert!(
        wgsl.contains("anchor_tick: u32,"),
        "edge struct should declare `anchor_tick: u32`:\n{wgsl}"
    );
    assert!(
        wgsl.contains("_pad:        u32,"),
        "edge struct should pad to 16B alignment:\n{wgsl}"
    );

    // Compile-time K constant from symmetric_pair_topk(K = 8).
    assert!(
        wgsl.contains("const K: u32 = 8u;"),
        "missing compile-time K constant from annotation:\n{wgsl}"
    );

    // Canonical pair via min/max. Both calls must appear — one for
    // owner, one for other.
    assert!(
        wgsl.contains("min(e.first, e.second)") && wgsl.contains("max(e.first, e.second)"),
        "kernel should canonicalise the pair via min/max:\n{wgsl}"
    );

    // Atomic operations on the counts buffer: atomicLoad + atomicAdd.
    assert!(
        wgsl.contains("atomicLoad(&view_standing_counts"),
        "should atomicLoad counts:\n{wgsl}"
    );
    assert!(
        wgsl.contains("atomicAdd(&view_standing_counts"),
        "should atomicAdd to reserve a slot:\n{wgsl}"
    );

    // Bindings: slots + counts (plus per-handler events + cfg).
    assert!(
        wgsl.contains("var<storage, read_write> view_standing_slots:  array<StandingEdge>;"),
        "missing slots binding:\n{wgsl}"
    );
    assert!(
        wgsl.contains("var<storage, read_write> view_standing_counts: array<atomic<u32>>;"),
        "missing counts binding:\n{wgsl}"
    );
    assert!(
        wgsl.contains("var<uniform>         cfg_standing_delta:"),
        "missing per-handler FoldCfg uniform binding:\n{wgsl}"
    );
}

/// Task 1.7 — the emitted kernel implements the full find-or-insert-
/// else-evict-weakest pipeline the CPU emitter spec'd. Check each arm
/// leaves an observable marker in the output.
#[test]
fn symmetric_pair_topk_wgsl_implements_three_way_fold() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR should exist");
    let wgsl =
        dsl_compiler::emit_view_wgsl::emit_symmetric_pair_topk_fold_wgsl(view).expect("WGSL emit");

    // Find-existing arm: compares `other` against each slot.
    assert!(
        wgsl.contains("view_standing_slots[row_base + i].other == other_id"),
        "find-existing arm should compare `other` against query id:\n{wgsl}"
    );

    // Insert arm: the early-return `if (count < K)` path.
    assert!(
        wgsl.contains("if (count < K)"),
        "should have an insert arm gated on `count < K`:\n{wgsl}"
    );
    // Writes `other = other_id` into the fresh slot.
    assert!(
        wgsl.contains("view_standing_slots[row_base + new_idx].other       = other_id;"),
        "insert arm should install the new edge's `other`:\n{wgsl}"
    );

    // Evict arm: linear scan for smallest |value|, then replace.
    assert!(
        wgsl.contains("weakest_idx"),
        "evict arm should track weakest slot index:\n{wgsl}"
    );
    assert!(
        wgsl.contains("abs(view_standing_slots[row_base + i].value)"),
        "evict arm should scan |value| across slots:\n{wgsl}"
    );
    assert!(
        wgsl.contains("if (abs(e.delta) > weakest_mag)"),
        "evict arm should only replace when |delta| beats weakest:\n{wgsl}"
    );
}

/// Task 1.7 — concurrency caveats must surface in emitted-kernel
/// comments so Phase 3's consumer notices them. We intentionally lean
/// on documentation-through-code here; the actual race behaviour will
/// be validated when the standing port lands.
#[test]
fn symmetric_pair_topk_wgsl_documents_concurrency_caveats() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR should exist");
    let wgsl =
        dsl_compiler::emit_view_wgsl::emit_symmetric_pair_topk_fold_wgsl(view).expect("WGSL emit");

    // Header block calls out the race surface.
    assert!(
        wgsl.contains("Concurrency caveats"),
        "kernel header should document concurrency caveats:\n{wgsl}"
    );
    // Phase-3-validate TODOs embedded in the update and evict arms.
    assert!(
        wgsl.matches("TODO(phase-3)").count() >= 2,
        "expected multiple TODO(phase-3) markers for races:\n{wgsl}"
    );
}

/// Task 1.7 — reject non-symmetric storage hints. The dedicated entry
/// point is only for `symmetric_pair_topk` views — give a clear error
/// for anything else so callers don't accidentally misroute.
#[test]
fn symmetric_pair_topk_wgsl_rejects_non_symmetric_storage() {
    const PAIR_SRC: &str = r#"
event AgentAttacked { actor: AgentId, target: AgentId }

@materialized(on_event = [AgentAttacked])
view my_enemies(a: Agent, b: Agent) -> f32 {
  initial: 0.0,
  on AgentAttacked { actor: actor, target: target } { self += 1.0 }
}
"#;
    let comp = dsl_compiler::compile(PAIR_SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "my_enemies")
        .expect("view IR should exist");
    let err = dsl_compiler::emit_view_wgsl::emit_symmetric_pair_topk_fold_wgsl(view).expect_err(
        "dedicated entry point should reject non-symmetric storage hints",
    );
    match err {
        dsl_compiler::emit_view_wgsl::EmitError::Unsupported(msg) => {
            assert!(
                msg.contains("symmetric_pair_topk"),
                "error should point at the entry-point mismatch, got: {msg}"
            );
        }
    }
}

/// Task 1.7 — the emitted WGSL parses cleanly through the same naga
/// frontend engine_gpu runs. The emitted module is a standalone unit
/// (struct decls + bindings + compute entry point), so naga should
/// accept it with no prologue.
#[test]
fn symmetric_pair_topk_wgsl_parses_through_naga() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR should exist");
    let wgsl =
        dsl_compiler::emit_view_wgsl::emit_symmetric_pair_topk_fold_wgsl(view).expect("WGSL emit");

    if let Err(e) = naga::front::wgsl::parse_str(&wgsl) {
        panic!(
            "emitted symmetric_pair_topk WGSL failed naga parse:\n{e}\n--- WGSL source ---\n{wgsl}"
        );
    }
}

/// Task 1.7 — `classify_view` continues to reject `symmetric_pair_topk`
/// with an Unsupported error that points callers at the dedicated
/// entry point. Phase 3 wires reads via a similar standalone emitter.
#[test]
fn symmetric_pair_topk_generic_classify_redirects_to_dedicated_emit() {
    let comp = dsl_compiler::compile(SRC).expect("compile should succeed");
    let view = comp
        .views
        .iter()
        .find(|v| v.name == "standing")
        .expect("view IR should exist");
    let err = dsl_compiler::emit_view_wgsl::classify_view(view)
        .expect_err("generic classify pipeline doesn't handle symmetric_pair_topk yet");
    match err {
        dsl_compiler::emit_view_wgsl::EmitError::Unsupported(msg) => {
            assert!(
                msg.contains("emit_symmetric_pair_topk_fold_wgsl"),
                "classify error should point at the dedicated entry point, got: {msg}"
            );
        }
    }
}
