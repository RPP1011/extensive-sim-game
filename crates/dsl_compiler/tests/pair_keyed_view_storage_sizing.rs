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

use dsl_compiler::build_helper::{
    detect_pair_keyed_materialized_view, detect_pair_keyed_second_key,
    PairKeyedSecondKey,
};

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

// ---------------------------------------------------------------------------
// T6 fix (2026-05-11): pair-keyed second key is a non-Agent entity type.
//
// Pre-fix the detector was a `bool` returning true only for
// `(Agent, Agent)`. trade_caravans's `view inventory(merchant:
// Agent, good: Item)` slipped through and got per-agent sizing,
// silently aliasing the Item dimension into the Agent index space
// (3 goods × 36 merchants writing to a 36-slot buffer = catastrophic
// aliasing). Generalised to a richer
// [`PairKeyedSecondKey`] enum so Item / Group / Quest second keys
// resolve to the static count of declared entities of that root.
// ---------------------------------------------------------------------------

const AGENT_ITEM_PAIR_VIEW_SNIPPET: &str = r#"
entity Hero : Agent { hp: f32 }
entity Grain : Item { base_price: f32 }
entity Spice : Item { base_price: f32 }
entity Silk : Item { base_price: f32 }

event Bought {
  buyer: AgentId,
  good: ItemId,
  price: f32,
}

@materialized(on_event = [Bought], storage = pair_map)
view inventory(merchant: Agent, good: Item) -> f32 {
  initial: 0.0,
  on Bought { buyer: b, good: g, price: _ } where b == merchant && g == good { self += 1.0 }
  clamp: [-1000.0, 1000.0],
}
"#;

const AGENT_GROUP_PAIR_VIEW_SNIPPET: &str = r#"
entity Hero : Agent { hp: f32 }
entity Faction : Group { reputation: f32 }
entity GuildA : Group { reputation: f32 }

event ReputationGain {
  agent: AgentId,
  faction: GroupId,
  amount: f32,
}

@materialized(on_event = [ReputationGain], storage = pair_map)
view rep(observer: Agent, faction: Group) -> f32 {
  initial: 0.0,
  on ReputationGain { agent: a, faction: f, amount: x } where a == observer && f == faction { self += x }
  clamp: [-1000.0, 1000.0],
}
"#;

#[test]
fn agent_item_pair_view_resolves_to_item_count() {
    let comp = parse_resolve(AGENT_ITEM_PAIR_VIEW_SNIPPET);
    let skey = detect_pair_keyed_second_key(&comp);
    assert_eq!(
        skey,
        Some(PairKeyedSecondKey::Item(3)),
        "view inventory(merchant: Agent, good: Item) with 3 declared \
         Item entities (Grain/Spice/Silk) must resolve to \
         `PairKeyedSecondKey::Item(3)`; got {skey:?}",
    );
    // Bool-returning legacy helper keeps its tom_probe-shape contract
    // — any pair-keyed second key still returns true.
    assert!(detect_pair_keyed_materialized_view(&comp));
}

#[test]
fn agent_group_pair_view_resolves_to_group_count() {
    let comp = parse_resolve(AGENT_GROUP_PAIR_VIEW_SNIPPET);
    let skey = detect_pair_keyed_second_key(&comp);
    assert_eq!(
        skey,
        Some(PairKeyedSecondKey::Group(2)),
        "view rep(observer: Agent, faction: Group) with 2 declared \
         Group entities (Faction/GuildA) must resolve to \
         `PairKeyedSecondKey::Group(2)`; got {skey:?}",
    );
}

#[test]
fn population_u64_expr_is_static_for_non_agent_keys() {
    // The slot-count expression composes `agent_count *
    // <population_u64_expr>` — for non-Agent second keys the
    // population is a compile-time literal so the buffer alloc
    // doesn't depend on `agent_count` for the second dimension.
    assert_eq!(PairKeyedSecondKey::Item(3).population_u64_expr(), "3u64");
    assert_eq!(PairKeyedSecondKey::Group(2).population_u64_expr(), "2u64");
    assert_eq!(PairKeyedSecondKey::Quest(7).population_u64_expr(), "7u64");
    // Agent keeps the per-tick `agent_count` runtime variable —
    // matches the pre-T6 sizing for tom_probe / Agent×Agent shapes.
    assert_eq!(
        PairKeyedSecondKey::Agent.population_u64_expr(),
        "(agent_count as u64)",
    );
}

#[test]
fn tom_probe_sim_resolves_to_agent_second_key() {
    // tom_probe's `view beliefs_flags(observer: Agent, subject:
    // Agent) -> u32` should pick `PairKeyedSecondKey::Agent`. Pinned
    // here to guard against the T6 generalisation accidentally
    // demoting Agent×Agent to a non-Agent variant when both shapes
    // coexist.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/tom_probe.sim");
    let src = std::fs::read_to_string(&path).expect("read tom_probe.sim");
    let comp = parse_resolve(&src);
    assert_eq!(
        detect_pair_keyed_second_key(&comp),
        Some(PairKeyedSecondKey::Agent),
        "tom_probe.sim's pair-keyed view is Agent×Agent — the T6 \
         enum must surface this as `PairKeyedSecondKey::Agent`, not \
         a static count",
    );
}

#[test]
fn trade_caravans_sim_resolves_to_item_second_key() {
    // End-to-end: load the live trade_caravans.sim and confirm its
    // declared `view inventory(merchant: Agent, good: Item)` resolves
    // to `Item(3)` — Grain/Spice/Silk are the three Item-rooted
    // entities. This is the in-tree T6 trigger.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/trade_caravans.sim");
    let src = std::fs::read_to_string(&path).expect("read trade_caravans.sim");
    let comp = parse_resolve(&src);
    let skey = detect_pair_keyed_second_key(&comp);
    // trade_caravans declares 3 Items (Grain/Spice/Silk) — and ALSO
    // declares `view inventory(merchant: Agent, good: Item)`. With
    // the T6 fix the detector picks the LARGEST second-key
    // population across all pair-keyed views; trade_caravans has only
    // ONE pair-keyed view (the inventory) so the answer is exactly
    // `Item(3)`. (If a future revision adds an Agent×Agent view to
    // trade_caravans this assertion will need to flip to
    // `PairKeyedSecondKey::Agent` per the largest-wins fallback —
    // see [`detect_pair_keyed_second_key`] doc.)
    assert_eq!(
        skey,
        Some(PairKeyedSecondKey::Item(3)),
        "trade_caravans.sim's `view inventory(merchant, good: Item)` \
         must resolve to `PairKeyedSecondKey::Item(3)`; got {skey:?}",
    );
}
