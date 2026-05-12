//! Pin: `view_storage_primary` backing buffer is sized as `agent_count
//! * agent_count` cells when the fixture declares a pair-keyed
//! materialized view (e.g. `view foo(a: Agent, b: Agent) -> u32`).
//!
//! ## Why this matters
//!
//! The auto-emitted `GeneratedRuntime::try_new` allocates one
//! `wgpu::Buffer` per fixture-owned binding using `slot_count_expr`.
//! For a `pair_map`-shaped view, the fold body writes into
//! `view_storage_primary[observer * agent_count + subject]` — the
//! buffer therefore needs `agent_count * agent_count` u32 cells, not
//! the per-agent default.
//!
//! Pre-fix sizing fell back to `agent_count` cells, producing a 4×
//! under-allocation at `N=4` (16 bytes for what needed 64 bytes) that
//! silently corrupted memory at runtime when the fold body wrote
//! through indices 4..15. tom_probe's `view beliefs_flags(observer:
//! Agent, subject: Agent)` is the in-tree fixture exercising the
//! shape; this test pins the upstream sizing rule directly so a
//! regression surfaces in `cargo test -p dsl_compiler` without
//! needing the GPU dispatch path.
//!
//! ## What this exercises
//!
//! 1. [`detect_pair_keyed_materialized_view`] correctly identifies a
//!    `view foo(a: Agent, b: Agent) -> u32 { @materialized(...) }`
//!    declaration and rejects single-Agent / scalar-keyed views.
//! 2. The end-to-end build helper produces a `runtime_core.rs` where
//!    `view_storage_primary_buf` is sized via `(agent_count as u64) *
//!    (agent_count as u64) * <elem_bytes>`.

use dsl_compiler::build_helper::detect_pair_keyed_materialized_view;

const TOM_PROBE_PAIR_VIEW_SNIPPET: &str = r#"
event BeliefAcquired {
  observer: AgentId,
  subject: AgentId,
  fact_bit: u32,
}

@materialized(on_event = [BeliefAcquired], storage = pair_map)
view beliefs_flags(observer: Agent, subject: Agent) -> u32 {
  initial: 0,
  on BeliefAcquired { observer: o, subject: s, fact_bit: b } { self |= b }
  clamp: [0, 4294967295],
}
"#;

const SINGLE_AGENT_VIEW_SNIPPET: &str = r#"
event Threat {
  observer: AgentId,
  amount: f32,
}

@materialized(on_event = [Threat])
view threats(observer: Agent) -> f32 {
  initial: 0.0,
  on Threat { observer: o, amount: a } { self += a }
  clamp: [0.0, 1000000.0],
}
"#;

fn parse_resolve(src: &str) -> dsl_ast::ir::Compilation {
    let program = dsl_compiler::parse(src).expect("parse");
    dsl_ast::resolve::resolve(program).expect("resolve")
}

#[test]
fn pair_keyed_materialized_view_is_detected() {
    let comp = parse_resolve(TOM_PROBE_PAIR_VIEW_SNIPPET);
    assert!(
        detect_pair_keyed_materialized_view(&comp),
        "view foo(a: Agent, b: Agent) -> u32 must be flagged as pair-keyed",
    );
}

#[test]
fn single_agent_materialized_view_is_not_pair_keyed() {
    let comp = parse_resolve(SINGLE_AGENT_VIEW_SNIPPET);
    assert!(
        !detect_pair_keyed_materialized_view(&comp),
        "view foo(a: Agent) -> f32 must not be flagged as pair-keyed",
    );
}

#[test]
fn fixture_with_no_views_is_not_pair_keyed() {
    let comp = parse_resolve("event Tick {}\n");
    assert!(
        !detect_pair_keyed_materialized_view(&comp),
        "fixture with no `view` decls must not be flagged as pair-keyed",
    );
}

#[test]
fn tom_probe_sim_is_pair_keyed() {
    // End-to-end: load the live tom_probe.sim and confirm its declared
    // `view beliefs_flags(observer: Agent, subject: Agent)` flips the
    // flag. Catches regressions where the resolve pass changes how the
    // `Agent` param type is reified (e.g. swapping `IrType::AgentId`
    // for a new `IrType::Entity`-like variant).
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/tom_probe.sim");
    let src = std::fs::read_to_string(&path).expect("read tom_probe.sim");
    let comp = parse_resolve(&src);
    assert!(
        detect_pair_keyed_materialized_view(&comp),
        "tom_probe.sim's `view beliefs_flags(observer, subject)` must \
         be flagged as pair-keyed — this gates the N² sizing of \
         view_storage_primary_buf",
    );
}

#[test]
fn dodger_probe_sim_is_not_pair_keyed() {
    // Counter-fixture: dodger_probe declares `view threats(observer:
    // Agent) -> f32`. Single-Agent param ⇒ per-agent sizing keeps the
    // buffer at N (not N²).
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/dodger_probe.sim");
    let src = std::fs::read_to_string(&path).expect("read dodger_probe.sim");
    let comp = parse_resolve(&src);
    assert!(
        !detect_pair_keyed_materialized_view(&comp),
        "dodger_probe.sim has only single-Agent views; up-sizing \
         view_storage_primary to N² would over-allocate by N×",
    );
}
