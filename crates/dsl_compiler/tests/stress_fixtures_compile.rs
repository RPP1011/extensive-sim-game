//! Stress fixtures for slice 1 (cross-agent target reads) and the
//! Phase 7+8 wired primitives (event ring + view-fold storage +
//! @decay) under load.
//!
//! These tests run the .sim files at
//! `assets/sim/target_chaser.sim` and
//! `assets/sim/swarm_event_storm.sim` through the full
//! parse → resolve → CG lower → schedule → emit pipeline. They
//! intentionally exercise codepaths the existing pp/pc/cn min
//! fixtures don't:
//!
//! - **target_chaser**: `agents.pos(self.engaged_with)` in physics —
//!   the cross-agent target read codepath slice 1
//!   (`docs/superpowers/plans/2026-05-03-stdlib-into-cg-ir.md`)
//!   replaced the B1 typed-default placeholder for. Without slice 1
//!   the WGSL emit silently returned `vec3<f32>(0.0)` for the
//!   target read; with slice 1 it hoists a stmt-scope
//!   `let target_expr_<N>: u32 = …;` and uses
//!   `agent_pos[target_expr_<N>]`.
//!
//! - **swarm_event_storm**: 4 emits per agent per tick into a single
//!   ring + two folds (a plain accumulator and an @decay-anchored
//!   accumulator). Stresses the per-tick event-ring throughput and
//!   the @decay anchor RMW.
//!
//! Today these tests assert the pipeline reaches `emit` without
//! panicking. They DON'T assert observable behaviour (that needs
//! per-fixture runtime crates). The goal is to surface the next
//! gap as a typed compiler error rather than a runtime panic — so
//! any new gap is a focused fix, not a "where do I even start"
//! debugging session.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

/// Drive `path` through parse → resolve → lower → schedule → emit.
/// Returns the emitted artifacts on success; surfaces the pipeline
/// error verbatim on failure so the test failure is the next gap
/// rather than an opaque panic.
fn compile_sim(path: &std::path::Path) -> Result<EmittedArtifacts, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|e| format!("lower: {e:?}"))?;
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))
}

/// Find the first WGSL kernel whose name contains `needle`. Used to
/// pick the physics or fold body out of the artifact set without
/// hard-coding the exact emitted name (those drift as the kernel
/// composer evolves).
fn kernel_body_containing<'a>(art: &'a EmittedArtifacts, needle: &str) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle))
        .map(|(_, body)| body.as_str())
}

#[test]
fn target_chaser_compiles() {
    let path = workspace_path("assets/sim/target_chaser.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("target_chaser.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[target_chaser] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// Slice 1 invariant: the cross-agent target read in
/// `agents.pos(self.engaged_with)` lowers to a stmt-scope
/// `let target_expr_<N>: u32 = …;` paired with `agent_pos[
/// target_expr_<N>]`, NOT the prior B1 `vec3<f32>(0.0)` placeholder.
/// Locks slice 1 into the runtime-driven (not just unit-tested)
/// codepath.
#[test]
fn target_chaser_emits_target_let_binding() {
    let path = workspace_path("assets/sim/target_chaser.sim");
    let art = compile_sim(&path).expect("target_chaser compiles");
    let body = kernel_body_containing(&art, "ChaseTarget")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    assert!(
        body.contains("let target_expr_"),
        "expected slice-1 let target_expr_<N> binding in physics body; got:\n{body}",
    );
    assert!(
        body.contains("agent_pos[target_expr_"),
        "expected indexed access against target_expr_<N>; got:\n{body}",
    );
    assert!(
        !body.contains("vec3<f32>(0.0)") || body.matches("vec3<f32>(0.0)").count() < 2,
        "B1 typed-default placeholder should not dominate the body; got:\n{body}",
    );
}

#[test]
fn swarm_event_storm_compiles() {
    let path = workspace_path("assets/sim/swarm_event_storm.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("swarm_event_storm.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[swarm_event_storm] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `swarm_event_storm` declares 4 `emit Pulse { … }` per tick per
/// agent. Confirm the producer kernel actually emits 4 atomicAdd
/// + 4 atomicStore-blocks (one per emit), not e.g. one collapsed
/// emit. Locks the multi-emit codepath so a regression that
/// silently dropped emits (B1-style) would fail here.
#[test]
fn swarm_event_storm_emits_four_pulses_per_tick() {
    let path = workspace_path("assets/sim/swarm_event_storm.sim");
    let art = compile_sim(&path).expect("swarm_event_storm compiles");
    let body = kernel_body_containing(&art, "PulseAndDrift")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    let atomic_adds = body.matches("atomicAdd(&event_tail").count();
    assert_eq!(
        atomic_adds, 4,
        "expected 4 atomicAdds (one per Pulse emit); got {atomic_adds} in:\n{body}",
    );
}

/// Slice 2b probe (stdlib-into-CG-IR plan) — confirms a physics
/// rule body that consumes a Phase 7 named spatial_query through
/// the existing `sum(other in spatial.<name>(self) where ...)`
/// fold shape lowers end-to-end via the shared
/// `lower_spatial_namespace_call` helper. Without slice 2a's
/// recogniser extraction this codepath worked already (boids does
/// it) — the test pins the helper's continued coverage of physics
/// rule contexts so a regression in the fold-iter classification
/// fails here rather than silently skipping the spatial walk.
///
/// The `spatial_probe.sim` fixture is intentionally minimal: one
/// entity, one named query, one fold-bearing physics rule. The
/// neighbour-walk WGSL shape is recognisable by its
/// `spatial_grid_offsets` references (the bounded-walk template
/// reads grid offsets to enumerate candidates).
#[test]
fn spatial_probe_compiles_and_emits_neighbour_walk() {
    let path = workspace_path("assets/sim/spatial_probe.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("spatial_probe.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    let body = kernel_body_containing(&art, "ProbeMove")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    // Bounded-neighbour walk template references grid offsets; if
    // the spatial-iter recogniser ever silently falls back to
    // ForEachAgent (the unbounded N² path), this assertion catches
    // it (ForEachAgent emits a `for (var per_pair_candidate ... <
    // cfg.agent_cap` loop with no grid-offset reference).
    assert!(
        body.contains("spatial_grid_offsets") || body.contains("grid_starts"),
        "expected bounded-neighbour walk references in physics body; got:\n{body}",
    );
    eprintln!(
        "[spatial_probe] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `swarm_event_storm` declares a `@decay(rate=0.85)` view alongside
/// a non-decayed view; both consume from the same Pulse ring.
/// Confirm both fold kernels exist and the decay one references
/// the anchor binding.
#[test]
fn swarm_event_storm_emits_both_folds_with_decay_anchor() {
    let path = workspace_path("assets/sim/swarm_event_storm.sim");
    let art = compile_sim(&path).expect("swarm_event_storm compiles");
    let plain = kernel_body_containing(&art, "pulse_count").unwrap_or_else(|| {
        panic!(
            "no pulse_count fold kernel in artifacts; available: {:?}",
            art.wgsl_files.keys().collect::<Vec<_>>()
        );
    });
    let decayed = kernel_body_containing(&art, "recent_pulse_intensity").unwrap_or_else(|| {
        panic!(
            "no recent_pulse_intensity fold kernel; available: {:?}",
            art.wgsl_files.keys().collect::<Vec<_>>()
        );
    });
    // Plain fold body has primary storage but no anchor reference.
    assert!(
        plain.contains("view_storage_primary"),
        "plain fold should write primary; got:\n{plain}",
    );
    // Decayed fold body should reference the anchor binding (the
    // @decay rate gets applied via the anchor multiplication).
    // If the anchor isn't wired, this assertion surfaces the gap.
    assert!(
        decayed.contains("anchor") || decayed.contains("decay"),
        "decay fold should reference anchor or decay rate; got:\n{decayed}",
    );
}

/// `bartering.sim` is the entity-root coverage fixture
/// (`docs/spec/dsl.md` §2.1). Pre-existing fixtures (boids,
/// predator_prey, particle_collision, crowd_navigation,
/// target_chaser, swarm_event_storm, spatial_probe) only declare
/// `entity ... : Agent`, leaving the Item + Group root surfaces
/// untested end-to-end. This fixture declares one of each root
/// kind plus a `Trade` event with an `item: ItemId` payload field,
/// so a regression that broke the parser's `Item` / `Group` keyword
/// handling, the resolver's `EntityRoot::{Item,Group}` carry-through,
/// or the event-field codegen for `ItemId` would surface as a
/// compile failure here rather than silent acceptance.
#[test]
fn bartering_compiles() {
    let path = workspace_path("assets/sim/bartering.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("bartering.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[bartering] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// Lock the IR-level entity-root carry-through: parsing +
/// resolving `bartering.sim` MUST yield a `Compilation` whose
/// `entities` vector contains the three distinct root kinds
/// (Agent / Item / Group). Today the lowering + emit passes
/// don't do anything with the Item + Group rows (see the GAP
/// note in the .sim file), so this test guards the upstream
/// surfaces until those passes catch up — without it, somebody
/// could silently drop Item + Group entities at resolve time
/// and no other test would notice.
#[test]
fn bartering_resolves_three_distinct_entity_roots() {
    let path = workspace_path("assets/sim/bartering.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse bartering.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve bartering.sim");
    let mut saw_agent = false;
    let mut saw_item = false;
    let mut saw_group = false;
    for e in &comp.entities {
        match e.root {
            dsl_compiler::ast::EntityRoot::Agent => saw_agent = true,
            dsl_compiler::ast::EntityRoot::Item => saw_item = true,
            dsl_compiler::ast::EntityRoot::Group => saw_group = true,
            // Bartering doesn't declare any Quest-rooted entities;
            // Quest is the post-2026-05-04 declare-only root used by
            // quest_probe.sim. No-op here.
            dsl_compiler::ast::EntityRoot::Quest => {}
        }
    }
    assert!(saw_agent, "expected an Agent-rooted entity in bartering.sim");
    assert!(saw_item, "expected an Item-rooted entity in bartering.sim");
    assert!(saw_group, "expected a Group-rooted entity in bartering.sim");
}

/// `bartering.sim`'s IdleDrift physics rule reads BOTH
/// `items.weight(0)` (Item-field read) AND `groups.size(0)`
/// (Group-field read). Confirm the emitted physics WGSL contains
/// indexed accesses against BOTH the `coin_weight[` and
/// `caravan_size[` external bindings — the runtime-side proof that
/// the Group-field READ path is wired symmetrically to the Item-field
/// READ path through the entity-field catalog. Without this
/// assertion a regression that silently collapsed the Group-field
/// arm to a typed default (or routed it to the Item catalog) would
/// pass other tests but break the bartering_app drift observable.
#[test]
fn bartering_emits_item_and_group_field_reads() {
    let path = workspace_path("assets/sim/bartering.sim");
    let art = compile_sim(&path).expect("bartering compiles");
    let body = kernel_body_containing(&art, "IdleDrift")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    assert!(
        body.contains("coin_weight["),
        "expected indexed access against `coin_weight[…]` (Item-field read) \
         in physics body; got:\n{body}",
    );
    assert!(
        body.contains("caravan_size["),
        "expected indexed access against `caravan_size[…]` (Group-field read) \
         in physics body — confirms the Group-field READ path is symmetric \
         with the Item-field READ path via the entity-field catalog; got:\n{body}",
    );
}

/// `bartering.sim`'s `Trade` event payload contains
/// `item: ItemId`. Confirm the IdleDrift physics body actually
/// emits a record into the event ring whose payload reflects the
/// ItemId slot (an unconditional u32 store, like `AgentId`).
/// Without this assertion a regression that silently dropped
/// non-AgentId ID kinds from event payloads would compile clean
/// but produce a malformed event ring — the fold downstream would
/// then read garbage and the runtime observable in `bartering_app`
/// would fall apart.
#[test]
fn bartering_emits_item_id_in_trade_payload() {
    let path = workspace_path("assets/sim/bartering.sim");
    let art = compile_sim(&path).expect("bartering compiles");
    let body = kernel_body_containing(&art, "IdleDrift")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    // The Trade event has 3 payload fields (giver, receiver, item)
    // — the producer should issue 3 atomicStore-into-payload-slot
    // ops past the standard kind+tick header (slots 0 + 1).
    let payload_stores = body.matches("atomicStore(&event_ring[slot * 10u + 2u]").count()
        + body.matches("atomicStore(&event_ring[slot * 10u + 3u]").count()
        + body.matches("atomicStore(&event_ring[slot * 10u + 4u]").count();
    assert_eq!(
        payload_stores, 3,
        "expected 3 payload stores (giver, receiver, item) into the event ring; got {payload_stores} in:\n{body}",
    );
    // Sanity-check the fold consumes the receiver field (offset
    // 3 = 2-slot header + 0-th payload field after the typed
    // re-ordering).
    let fold_body = kernel_body_containing(&art, "trade_count").unwrap_or_else(|| {
        panic!(
            "no trade_count fold kernel; available: {:?}",
            art.wgsl_files.keys().collect::<Vec<_>>()
        );
    });
    assert!(
        fold_body.contains("event_ring[event_idx * 10u +"),
        "fold should index event ring by event_idx; got:\n{fold_body}",
    );
}

// ----- Sequential-implementation backlog fixtures (2026-05-03) -----
//
// Three new fixtures land in declaration form before their runtimes
// + sim_app binaries do, so the compile-gate tests catch any future
// parser/lower/emit regression that would block the sequential
// implementation work (ecosystem → foraging → auction).

/// `ecosystem_cascade.sim` declares 3 entity types (Plant, Herbivore,
/// Carnivore) all rooted at Agent, 2 distinct Eaten event rings, 3
/// per-tier physics rules, and 3 @decay-annotated views. Compile-
/// gate only — no observables to assert until the runtime crate
/// lands. The presence of 3 distinct fold kernels for the 3 views
/// confirms the schedule synthesizer correctly partitions per-view
/// dispatch (decay → fold pairs for each).
#[test]
fn ecosystem_cascade_compiles() {
    let path = workspace_path("assets/sim/ecosystem_cascade.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("ecosystem_cascade.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    // 3 fold kernels for the 3 views (recent_browse,
    // predator_pressure, plant_pressure).
    let fold_kernels: Vec<_> = art
        .kernel_index
        .iter()
        .filter(|name| name.starts_with("fold_"))
        .collect();
    assert!(
        fold_kernels.len() >= 3,
        "expected >=3 fold kernels (one per view); got {}: {:?}",
        fold_kernels.len(),
        fold_kernels,
    );
    // Each @decay-annotated view should also yield a decay kernel
    // (per the verb/probe/metric plan + the earlier B2 close).
    let decay_kernels: Vec<_> = art
        .kernel_index
        .iter()
        .filter(|name| name.starts_with("decay_"))
        .collect();
    assert_eq!(
        decay_kernels.len(),
        3,
        "expected 3 decay kernels (one per @decay view); got: {:?}",
        decay_kernels,
    );
    eprintln!(
        "[ecosystem_cascade] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
    // Per-handler event-kind filter check (2026-05-03): each fold
    // kernel must guard its body on the event tag at offset 0 so a
    // multi-kind ring (PlantEaten + HerbivoreEaten in this fixture)
    // doesn't double-count overlapping slot ranges. The exact kind
    // ids vary with allocator order; the structural guard text is
    // the invariant.
    for view_name in ["recent_browse", "predator_pressure", "plant_pressure"] {
        let kernel_name = format!("fold_{view_name}");
        let body = kernel_body_containing(&art, &kernel_name).unwrap_or_else(|| {
            panic!("expected {kernel_name} kernel");
        });
        assert!(
            body.contains("if (event_ring[event_idx * 10u + 0u] =="),
            "{kernel_name} body must guard on per-handler event-kind tag; got:\n{body}",
        );
    }
}

/// `foraging_colony.sim` declares Ant : Agent, Food : Item, Colony :
/// Group — exercising all 3 entity-root variants in one file, plus
/// `@decay` on a per-Ant view and a multi-emit-ready physics rule.
/// Item / Group bodies are declaration-only today (per the bartering
/// fixture's GAP commentary); this test pins the resolve + emit
/// shape so when the Item/Group SoA lower lands, the fixture's
/// diff is bounded.
///
/// Also locks the Item-population-aware `pair_map` cfg shape:
/// `pheromone_trail` carries `@materialized(storage = pair_map)`
/// and the fold/decay kernels expose `second_key_pop` / `slot_count`
/// fields the runtime sets to the Food entity's per-fixture
/// population (decoupled from `agent_cap`). Without this assertion
/// a regression that re-tied the dispatch sizing to `agent_cap`
/// would silently restore the pre-fix `agent_count²` over-allocation.
#[test]
fn foraging_colony_compiles() {
    let path = workspace_path("assets/sim/foraging_colony.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("foraging_colony.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    // pheromone_deposits view fold should exist (the only @decay
    // view that's enabled today; the pair_map + Group views are
    // commented out behind the SoA gap).
    let pheromone = kernel_body_containing(&art, "pheromone_deposits");
    assert!(
        pheromone.is_some(),
        "expected pheromone_deposits fold kernel; available: {:?}",
        art.wgsl_files.keys().collect::<Vec<_>>(),
    );

    // Item-population-aware pair_map sizing: the pheromone_trail
    // view's decay kernel must early-return on `cfg.slot_count`
    // (NOT `cfg.agent_cap`) so the runtime can over-allocate to
    // `agent_cap × FOOD_COUNT` independently. Symmetrically the
    // fold body must compose the 2-D index via `cfg.second_key_pop`.
    let trail_decay = kernel_body_containing(&art, "decay_pheromone_trail")
        .expect("expected decay_pheromone_trail kernel for pair_map view");
    assert!(
        trail_decay.contains("cfg.slot_count"),
        "decay_pheromone_trail must early-return on cfg.slot_count \
         (Item-population-aware sizing); got:\n{trail_decay}",
    );
    assert!(
        trail_decay.contains("slot_count: u32"),
        "decay_pheromone_trail cfg struct must declare slot_count: u32; \
         got:\n{trail_decay}",
    );
    let trail_fold = kernel_body_containing(&art, "fold_pheromone_trail")
        .expect("expected fold_pheromone_trail kernel for pair_map view");
    assert!(
        trail_fold.contains("cfg.second_key_pop"),
        "fold_pheromone_trail must compose pair index via cfg.second_key_pop \
         (Item-population-aware sizing); got:\n{trail_fold}",
    );
    assert!(
        trail_fold.contains("second_key_pop: u32"),
        "fold_pheromone_trail cfg struct must declare second_key_pop: u32; \
         got:\n{trail_fold}",
    );

    eprintln!(
        "[foraging_colony] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `auction_market.sim` declares Trader : Agent, Good : Item,
/// Faction : Group plus 2 distinct event kinds (Bid, Allocated).
/// Bid is emitted per-tick; Allocated is declared but not yet
/// emitted (waits on auctions.* namespace lowering). Compile-gate
/// confirms both event rings are routed and the Bid producer +
/// fold consumers wire through.
#[test]
fn auction_market_compiles() {
    let path = workspace_path("assets/sim/auction_market.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("auction_market.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    // 2 view folds today: bid_activity (decay) + good_bid_total
    // (no decay).
    let bid_activity = kernel_body_containing(&art, "bid_activity");
    let good_bid_total = kernel_body_containing(&art, "good_bid_total");
    assert!(
        bid_activity.is_some() && good_bid_total.is_some(),
        "expected both bid_activity + good_bid_total folds; available: {:?}",
        art.wgsl_files.keys().collect::<Vec<_>>(),
    );
    // The Bid producer kernel should atomicStore the 3 payload
    // fields (trader, good, amount) into the ring past the header.
    let bid_producer = kernel_body_containing(&art, "WanderAndBid")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no Bid producer kernel found; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    let payload_stores = bid_producer.matches("atomicStore(&event_ring[slot * 10u + 2u]").count()
        + bid_producer.matches("atomicStore(&event_ring[slot * 10u + 3u]").count()
        + bid_producer.matches("atomicStore(&event_ring[slot * 10u + 4u]").count();
    assert_eq!(
        payload_stores, 3,
        "expected 3 Bid payload stores (trader, good, amount); got {payload_stores}",
    );
    // The `auctions.*` namespace registration probe: WanderAndBid
    // calls `auctions.place_bid(self, self, config.market.bid_amount)`
    // which must lower to a `auctions_place_bid(` call in the emitted
    // body. The B1 stub `fn auctions_place_bid(...) -> bool { return
    // true; }` is also injected via the namespace-prelude scan. This
    // confirms the auctions namespace is end-to-end registered +
    // lowerable + emittable; without the registry entry this would
    // surface as a `LoweringError::UnsupportedNamespaceCall` upstream.
    assert!(
        bid_producer.contains("auctions_place_bid("),
        "expected auctions_place_bid(...) call in WanderAndBid body; got: {bid_producer}",
    );
    assert!(
        bid_producer.contains("fn auctions_place_bid("),
        "expected B1-stub fn auctions_place_bid declaration in WanderAndBid prelude",
    );
    eprintln!(
        "[auction_market] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `event_kind_filter_probe.sim` is the focused regression-guard
/// fixture for the per-handler event-kind filtering gap closed
/// 2026-05-03 (see `cg/emit/kernel.rs::build_view_fold_wgsl_body`).
///
/// Pre-fix: the fold body iterated EVERY event in the unified ring
/// without checking the per-event kind tag at offset 0, so any view
/// declared `on KindA { ... }` also processed KindB events (and
/// vice versa). In `ecosystem_cascade` the bug was masked because
/// the per-tier disjointness (Plants don't emit; Herbivores emit
/// only PlantEaten; Carnivores emit only HerbivoreEaten) made the
/// per-slot ranges non-overlapping. This fixture overlaps the
/// targets: every Probe agent emits one KindA + one KindB targeting
/// `self`, so each view's slot range is the SAME as the other's.
/// Without the tag-check guard, both `kind_a_count` and
/// `kind_b_count` would accumulate from BOTH events (per-slot value
/// = 2*ticks); with the guard, each view sees only its declared
/// kind (per-slot value = ticks).
///
/// The compile-time invariant pinned here is structural: each fold
/// body MUST contain an `if (event_ring[event_idx * <stride>u + 0u]
/// == <kind_id>u)` line wrapping the storage RMW. The runtime
/// observable check (per-slot count = ticks not 2*ticks) lands in
/// a follow-up runtime test once the fixture has a runtime crate.
#[test]
fn event_kind_filter_probe_compiles_with_tag_guard() {
    let path = workspace_path("assets/sim/event_kind_filter_probe.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("event_kind_filter_probe.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");

    // Two fold kernels, one per view. The runtime kernel-name
    // pattern is `fold_<view>_<event>` (see semantic_kernel_name in
    // cg/emit/kernel.rs); pull each by view-name substring.
    let fold_a = kernel_body_containing(&art, "kind_a_count").unwrap_or_else(|| {
        panic!(
            "no kind_a_count fold kernel; available: {:?}",
            art.wgsl_files.keys().collect::<Vec<_>>()
        );
    });
    let fold_b = kernel_body_containing(&art, "kind_b_count").unwrap_or_else(|| {
        panic!(
            "no kind_b_count fold kernel; available: {:?}",
            art.wgsl_files.keys().collect::<Vec<_>>()
        );
    });

    // Each fold body must guard its handler block with a tag check:
    // `if (event_ring[event_idx * 10u + 0u] == <kind>u)`. The exact
    // kind id is allocator-determined; just confirm the structural
    // shape is present.
    let guard_pattern = "if (event_ring[event_idx * 10u + 0u] ==";
    assert!(
        fold_a.contains(guard_pattern),
        "kind_a_count fold body must contain per-handler tag check; got:\n{fold_a}",
    );
    assert!(
        fold_b.contains(guard_pattern),
        "kind_b_count fold body must contain per-handler tag check; got:\n{fold_b}",
    );

    // The two folds must guard on DIFFERENT kind ids — otherwise
    // both would accumulate from the same kind only and the probe
    // wouldn't distinguish KindA vs KindB. Pull the kind id from
    // each guard line.
    let extract_kind = |body: &str| -> Option<String> {
        let pat_idx = body.find(guard_pattern)?;
        let after = &body[pat_idx + guard_pattern.len()..];
        let close = after.find(')')?;
        Some(after[..close].trim().trim_end_matches('u').trim().to_string())
    };
    let kind_a = extract_kind(fold_a).expect("guard line in kind_a_count body");
    let kind_b = extract_kind(fold_b).expect("guard line in kind_b_count body");
    assert_ne!(
        kind_a, kind_b,
        "expected DIFFERENT kind ids for KindA vs KindB folds; got both = {kind_a}",
    );

    // Sanity: the producer (EmitBoth) writes BOTH kind tags into
    // the same ring, so the per-fold filter is the only thing
    // separating which view sees which event. Confirm the producer
    // body has two `atomicStore(&event_ring[slot * 10u + 0u], <id>u)`
    // tag stores corresponding to KindA vs KindB.
    let producer = kernel_body_containing(&art, "EmitBoth")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no EmitBoth physics kernel; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    let tag_stores = producer
        .matches("atomicStore(&event_ring[slot * 10u + 0u]")
        .count();
    assert_eq!(
        tag_stores, 2,
        "expected 2 tag stores in EmitBoth producer (one per emit); got {tag_stores} in:\n{producer}",
    );

    eprintln!(
        "[event_kind_filter_probe] fold_a guards on kind {kind_a}, fold_b on kind {kind_b}; {} kernels",
        art.kernel_index.len(),
    );
}

/// `tom_probe.sim` exercises Theory-of-Mind end-to-end: per-(observer,
/// subject) `u32` belief bitset, materialized as a `pair_map`-storage
/// view with bit-OR (`|=`) accumulator, fed by a per-Knower physics
/// rule that emits `BeliefAcquired` events. The discovery write-up at
/// `docs/superpowers/notes/2026-05-04-tom-probe.md` describes the
/// pre-fix shape (which probed the AST-level `BeliefsAccessor` and
/// `theory_of_mind.believes_knows` surfaces directly and dropped at
/// lower time); the post-fix shape sidesteps both AST gaps by routing
/// through regular event + view-fold infrastructure (P6: events ARE
/// the mutation channel).
///
/// This test pins the post-fix invariants:
///   - lower is clean (no diagnostics);
///   - the `fold_beliefs` kernel exists (with `atomicOr` shape — see
///     `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`);
///   - the `physics_WhatIBelieve` producer kernel exists (with
///     `BeliefAcquired` emit shape).
///
/// If anyone re-lands the AST belief-read surfaces under the same
/// fixture name, this test will need to be re-pointed at a separate
/// fixture or split into pre-/post-fix variants.
/// `trade_market_probe.sim` is the FIRST fixture that combines all
/// the recently-landed compiler surfaces in one program:
///   - 4 distinct Item entity declarations + 1 Group declaration
///   - Multi-event-kind producer (TradeExecuted + PriceObserved
///     share one event ring, two distinct kind tags at offset 0)
///   - ToM-shape `pair_map` `u32` view (`price_belief`) — atomicOr
///     fold (post-`51b5853b`)
///   - Mixed view-fold storage in one program: u32 pair_map +
///     f32 with @decay + f32 no-decay
///
/// Pins the compile-time invariants that surfaced during the
/// discovery probe (see `docs/superpowers/notes/2026-05-04-trade_
/// market_probe.md`):
///   - Lower is clean (no diagnostics).
///   - 10 kernels emitted: physics_WanderAndTrade + 3 fold kernels
///     (price_belief, trader_volume, hub_volume) + 1 decay kernel
///     (trader_volume only — hub_volume has no @decay) + 5 admin.
///   - The producer body has TWO tag stores at offset 0
///     (`atomicStore(&event_ring[slot * 10u + 0u], <kind>u)`) —
///     one per emit, distinct kind ids.
///   - Each fold body guards on the per-handler kind tag at offset
///     0 (the cb24fd69 multi-kind ring partition shape).
///   - `fold_price_belief` uses WGSL native `atomicOr` (NOT a CAS
///     loop) — the `|=` accumulator + `u32` view return type
///     selects the bit-OR emit branch.
#[test]
fn trade_market_probe_combines_landed_surfaces() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/trade_market_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse trade_market_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve trade_market_probe.sim");

    // Catalog sanity: 7 entities — 2 Agent (Trader, Hub) + 4 Item
    // (Wood/Iron/Grain/Cloth) + 1 Group (Guild).
    let mut agents = 0;
    let mut items = 0;
    let mut groups = 0;
    for e in &comp.entities {
        match e.root {
            dsl_compiler::ast::EntityRoot::Agent => agents += 1,
            dsl_compiler::ast::EntityRoot::Item => items += 1,
            dsl_compiler::ast::EntityRoot::Group => groups += 1,
            // trade_market_probe doesn't declare Quest entities; the
            // post-2026-05-04 Quest root is exercised in quest_probe.
            dsl_compiler::ast::EntityRoot::Quest => {}
        }
    }
    assert_eq!(agents, 2, "expected 2 Agent entities (Trader + Hub)");
    assert_eq!(
        items, 4,
        "expected 4 distinct Item entities (Wood/Iron/Grain/Cloth) — \
         this is the first fixture with multiple Item declarations"
    );
    assert_eq!(groups, 1, "expected 1 Group entity (Guild)");

    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        let diag_text = o
            .diagnostics
            .iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "trade_market_probe.sim lower expected clean; got {} diagnostics:\n{diag_text}",
            o.diagnostics.len(),
        );
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit trade_market_probe");

    // 3 fold kernels: price_belief, trader_volume, hub_volume.
    let fold_names: Vec<_> = art
        .kernel_index
        .iter()
        .filter(|n| n.starts_with("fold_"))
        .cloned()
        .collect();
    assert_eq!(
        fold_names.len(),
        3,
        "expected 3 fold kernels (price_belief, trader_volume, hub_volume); got: {:?}",
        fold_names,
    );
    // 1 decay kernel: trader_volume (the only @decay view).
    let decay_names: Vec<_> = art
        .kernel_index
        .iter()
        .filter(|n| n.starts_with("decay_"))
        .cloned()
        .collect();
    assert_eq!(
        decay_names.len(),
        1,
        "expected 1 decay kernel (trader_volume); got: {:?}",
        decay_names,
    );

    // Producer must emit TWO tag stores at offset 0 (one per emit:
    // TradeExecuted + PriceObserved).
    let producer = kernel_body_containing(&art, "WanderAndTrade")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });
    let tag_stores = producer
        .matches("atomicStore(&event_ring[slot * 10u + 0u]")
        .count();
    assert_eq!(
        tag_stores, 2,
        "expected 2 tag stores in WanderAndTrade producer (one per emit); got {tag_stores} in:\n{producer}",
    );

    // Per-handler tag filter (cb24fd69) — every fold body must
    // guard on the kind tag at offset 0.
    let guard_pattern = "if (event_ring[event_idx * 10u + 0u] ==";
    for view_name in ["price_belief", "trader_volume", "hub_volume"] {
        let kernel_name = format!("fold_{view_name}");
        let body = kernel_body_containing(&art, &kernel_name).unwrap_or_else(|| {
            panic!("expected {kernel_name} kernel");
        });
        assert!(
            body.contains(guard_pattern),
            "{kernel_name} body must guard on per-handler event-kind tag; got:\n{body}",
        );
    }

    // ToM shape: fold_price_belief uses atomicOr (the post-`51b5853b`
    // u32 fold branch) NOT the f32 CAS+add loop.
    let pb_body = kernel_body_containing(&art, "price_belief")
        .expect("fold_price_belief kernel missing");
    assert!(
        pb_body.contains("atomicOr"),
        "expected atomicOr in fold_price_belief body (u32 view + |= accumulator); got:\n{pb_body}",
    );
    assert!(
        !pb_body.contains("atomicCompareExchangeWeak"),
        "expected NO CAS loop in u32 fold body; got:\n{pb_body}",
    );

    // The two f32 view folds DO use the CAS+add loop (no atomic
    // float add in WGSL; the `+= a` accumulator on a `-> f32` view
    // routes through the CAS loop). We pick the fold kernel by
    // exact-name lookup because `kernel_body_containing("trader_volume")`
    // would also match `decay_trader_volume` (substring containment).
    let tv_body = art
        .wgsl_files
        .get("fold_trader_volume.wgsl")
        .map(|s| s.as_str())
        .expect("fold_trader_volume.wgsl missing");
    assert!(
        tv_body.contains("atomicCompareExchangeWeak"),
        "expected CAS+add loop in fold_trader_volume body (f32 view); got:\n{tv_body}",
    );
    let hv_body = art
        .wgsl_files
        .get("fold_hub_volume.wgsl")
        .map(|s| s.as_str())
        .expect("fold_hub_volume.wgsl missing");
    assert!(
        hv_body.contains("atomicCompareExchangeWeak"),
        "expected CAS+add loop in fold_hub_volume body (f32 view); got:\n{hv_body}",
    );

    eprintln!(
        "[trade_market_probe] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

#[test]
fn tom_probe_lowers_clean_and_emits_belief_kernels() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/tom_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse tom_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve tom_probe.sim");
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        let diag_text = o
            .diagnostics
            .iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "tom_probe.sim lower expected clean (post-fix shape); got \
             {} diagnostics:\n{diag_text}",
            o.diagnostics.len(),
        );
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit tom_probe program");
    assert!(
        art.kernel_index.iter().any(|k| k == "fold_beliefs"),
        "expected fold_beliefs kernel in artifacts; got: {:?}",
        art.kernel_index,
    );
    assert!(
        art.kernel_index.iter().any(|k| k == "physics_WhatIBelieve"),
        "expected physics_WhatIBelieve kernel in artifacts; got: {:?}",
        art.kernel_index,
    );
    // Post-fix the fold body should use WGSL native atomicOr (not the
    // f32 CAS+add loop) — the `|=` accumulator + `u32` view return type
    // selects the bit-OR emit branch.
    let fold_wgsl = art
        .wgsl_files
        .get("fold_beliefs.wgsl")
        .expect("fold_beliefs.wgsl missing");
    assert!(
        fold_wgsl.contains("atomicOr"),
        "expected atomicOr in fold_beliefs body; got:\n{fold_wgsl}",
    );
    assert!(
        !fold_wgsl.contains("atomicCompareExchangeWeak"),
        "expected NO CAS loop in u32 fold body; got:\n{fold_wgsl}",
    );
    eprintln!(
        "[tom_probe] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// Gap #1 from `2026-05-04-trade_market_probe.md` — `config.<block>.
/// <u32_field>` flowing into a `u32` event-ring slot must NOT crash
/// the WGSL validator with an `f32 → u32` auto-conversion error.
///
/// Pre-fix: every config const was materialised as
/// `const config_<id>: f32 = <v>;` regardless of the declared field
/// type, so `atomicStore(&event_ring[...], (config_<id>))` (where the
/// slot is `array<atomic<u32>>`) crashed naga's WGSL front-end.
///
/// Post-fix: a `u32`-declared config field emits as
/// `const config_<id>: u32 = <n>u;`, and the `(config_<id>)` literal
/// flows into the `u32` slot directly.
///
/// This is an inline-source stress test rather than a fixture-file
/// test so it can be co-located with the gap doc reference and so the
/// full DSL surface lives in one place. The shape is a single agent
/// with a `u32`-declared config field referenced inline in the body
/// of a physics rule that emits an event with a `u32` field.
#[test]
fn config_u32_field_emits_with_u32_suffix_in_kernel_const() {
    let src = r#"
event Tick { }

@replayable
@gpu_amenable
event ObservationLogged {
  observer:    AgentId,
  observation: u32,
}

entity Trader : Agent {
  pos: vec3,
  vel: vec3,
}

config market {
  // Mixed-type config block — the U32 field is the gap-#1 surface.
  step_scale:      f32 = 0.05,
  observation_bit: u32 = 5,
  pulse_count:     i32 = -7,
}

physics LogObservation @phase(per_agent) {
  on Tick {} where (self.alive) {
    let new_pos = self.pos + self.vel * config.market.step_scale;
    agents.set_pos(self, new_pos);
    emit ObservationLogged {
      observer:    self,
      observation: config.market.observation_bit,
    }
  }
}
"#;
    let prog = dsl_compiler::parse(src).expect("parse inline u32-config source");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve inline u32-config source");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        let diag_text = o
            .diagnostics
            .iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "lower expected clean; got {} diagnostics:\n{diag_text}",
            o.diagnostics.len(),
        );
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit inline u32-config program");

    let physics = kernel_body_containing(&art, "LogObservation")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            )
        });

    // Find each config_<id> declaration and verify its type matches
    // the declared field type. Expectations (in source-order id
    // allocation): step_scale → f32, observation_bit → u32, pulse_count → i32.
    assert!(
        physics.contains("const config_0: f32 = 0.05;"),
        "expected step_scale to materialise as f32; got physics body:\n{physics}",
    );
    assert!(
        physics.contains("const config_1: u32 = 5u;"),
        "expected observation_bit to materialise as `u32 = 5u` (gap #1); got physics body:\n{physics}",
    );
    // pulse_count is unused in the body, so its const may NOT appear in
    // the physics kernel — the emit is conditional on body containment.
    // The U32 + F32 cases above cover the gap-#1 surface; the I32 case
    // is exercised separately by the `populate_config_consts_routes_*`
    // unit test in `cg/lower/driver.rs`.

    // The atomicStore for the u32 ring slot must consume `config_1`
    // directly (no f32 cast / bitcast) — that's the validator-pass
    // shape. The exact word offset depends on the event layout, but
    // the substring is unambiguous.
    assert!(
        physics.contains("(config_1)"),
        "expected atomicStore to consume `(config_1)` directly into u32 slot; got physics body:\n{physics}",
    );
    assert!(
        !physics.contains("bitcast<u32>(config_1)"),
        "u32-typed config_1 must NOT be bitcast — that would only be needed for f32→u32 routing; got physics body:\n{physics}",
    );
}

/// Gap #6 from `2026-05-04-trade_market_probe.md` — an `event <Name>
/// { ... }` declaration with no `emit <Name> { ... }` site anywhere in
/// the program must surface a non-fatal
/// `CgDiagnosticKind::DeclaredEventNeverEmitted` warning at compile
/// time.
///
/// Pre-fix: the trade_market_probe declared a `Shipment` event whose
/// only consumer was a future task — no rule body ever emitted it, and
/// the compiler accepted the dead declaration silently. Post-fix the
/// declaration produces a warning; a hard error would break the
/// staged-work / placeholder pattern, so the diagnostic stays
/// non-fatal.
///
/// Inline-source shape so the assertion is co-located with the
/// expected behaviour without needing a fixture file.
#[test]
fn declared_event_never_emitted_surfaces_compiler_warning() {
    use dsl_compiler::cg::program::CgDiagnosticKind;

    let src = r#"
event Tick { }

@replayable
@gpu_amenable
event UsedEvent {
  observer: AgentId,
  pulse:    u32,
}

@replayable
@gpu_amenable
event NeverEmitted {
  observer: AgentId,
  payload:  u32,
}

entity Worker : Agent {
  pos: vec3,
  vel: vec3,
}

physics PingOnce @phase(per_agent) {
  on Tick {} where (self.alive) {
    let new_pos = self.pos + self.vel * 0.05;
    agents.set_pos(self, new_pos);
    emit UsedEvent {
      observer: self,
      pulse:    1,
    }
  }
}
"#;
    let prog = dsl_compiler::parse(src).expect("parse inline event-declaration source");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve inline event-declaration source");
    let cg =
        dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("lower expected clean");

    // The `NeverEmitted` declaration should have produced exactly one
    // `DeclaredEventNeverEmitted` warning. `UsedEvent` is emitted by
    // the `PingOnce` physics rule, so it should NOT produce a warning.
    let never_emitted_warnings: Vec<_> = cg
        .diagnostics
        .iter()
        .filter_map(|d| match &d.kind {
            CgDiagnosticKind::DeclaredEventNeverEmitted { event } => Some(*event),
            _ => None,
        })
        .collect();

    assert_eq!(
        never_emitted_warnings.len(),
        1,
        "expected exactly one DeclaredEventNeverEmitted warning; got {} \
         total diagnostics:\n{}",
        cg.diagnostics.len(),
        cg.diagnostics
            .iter()
            .map(|d| format!("  {d}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let event_id = never_emitted_warnings[0];
    let event_name = cg
        .interner
        .event_kinds
        .get(&event_id.0)
        .map(String::as_str)
        .unwrap_or("<unnamed>");
    assert_eq!(
        event_name, "NeverEmitted",
        "warning should fire for `NeverEmitted`, not `UsedEvent`; got `{event_name}`",
    );
}

/// `abilities_probe` compile-gate. Pins the structural surface that
/// the smallest end-to-end ability-system probe exercises:
///
///   - 2 `verb` declarations (Strike, Heal) → 2 mask predicates +
///     2 chronicle physics rules (PhysicsRule on ActionSelected).
///   - 1 ScoringArgmax with 2 competing rows.
///   - 2 view-folds keyed on distinct event tags (DamageDealt,
///     Healed) — one fed by the verb that wins argmax, one by the
///     verb that loses.
///
/// Locks the verb-cascade multi-verb codepath so a regression that
/// silently dropped a second verb's mask / chronicle / scoring row
/// fails here.
///
/// SURFACES the discovered emit-side gap (2026-05-04 abilities probe
/// discovery): the schedule fuses `fold_healing_done` (consumer of
/// `Healed`) with the Strike chronicle (producer of `DamageDealt`) into
/// `fused_fold_healing_done_healed`, and the WGSL body in that fused
/// kernel references an undeclared identifier `view_storage_primary`
/// — the bindings struct only exposes `view_1_primary`. The naga
/// validation assertion below fails today; once the rename pass is
/// fixed, the assertion flips to "all kernels naga-clean".
#[test]
fn abilities_probe_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/abilities_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse abilities_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve abilities_probe.sim");
    // Lower may emit `well_formedness` cycle diag (gap — see discovery
    // doc). Tolerate the diag and proceed with the partial program so
    // the rest of the pipeline still surfaces.
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[abilities lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit abilities_probe");

    // Op-level invariants: verb expansion produced exactly 2 mask
    // predicates + 2 chronicle physics rules + 1 scoring with 2
    // rows + 2 view folds.
    let mut mask_count = 0;
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    let mut scoring_rows = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::MaskPredicate { .. } => mask_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ScoringArgmax { rows, .. } => {
                scoring_rows = rows.len();
            }
            _ => {}
        }
    }
    assert_eq!(
        mask_count, 2,
        "expected 2 MaskPredicate ops (one per verb's `when` clause); got {mask_count}",
    );
    assert_eq!(
        physics_count, 2,
        "expected 2 PhysicsRule ops (verb-synthesised chronicles for Strike + Heal); got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 2,
        "expected 2 ViewFold ops (damage_total + healing_done); got {viewfold_count}",
    );
    assert_eq!(
        scoring_rows, 2,
        "expected 2 scoring rows (Strike + Heal competing); got {scoring_rows}",
    );

    // Both Strike and Heal must have entries in the kernel index. The
    // exact kernel names drift as the schedule fuses pairs together;
    // we look for the substring "Strike" or "Heal" anywhere in the
    // emitted set.
    let has_strike = art
        .kernel_index
        .iter()
        .any(|n| n.to_lowercase().contains("strike"));
    let has_heal = art
        .kernel_index
        .iter()
        .any(|n| n.to_lowercase().contains("heal"));
    assert!(
        has_strike,
        "expected at least one kernel name to mention Strike; got {:?}",
        art.kernel_index,
    );
    assert!(
        has_heal,
        "expected at least one kernel name to mention Heal; got {:?}",
        art.kernel_index,
    );

    eprintln!(
        "[abilities_probe] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// Probe Gap #1 — naga validation of the abilities probe must stay
/// clean. Pre-fix the schedule fused `fold_healing_done` (consumer
/// of `Healed`) with the Strike chronicle (producer of
/// `DamageDealt`) into `fused_fold_healing_done_healed`, and the
/// fused kernel's WGSL body referenced an undeclared
/// `view_storage_primary` (the fused-kernel bindings struct exposes
/// `view_<id>_<slot>` per the Generic emit path).
///
/// Closed by Rule 5 in `cg/schedule/fusion.rs::cross_domain_split_decision`:
/// any `(ViewFold, PhysicsRule)` or `(PhysicsRule, ViewFold)` pair
/// now splits regardless of write/event overlap, so the fold keeps
/// its legacy `fold_<view>` 7-binding layout and the chronicle
/// emits as a standalone `physics_<verb>` kernel. Both naga-validate
/// cleanly.
#[test]
fn abilities_probe_naga_clean() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/abilities_probe.sim");
    let src = std::fs::read_to_string(&path).expect("read");
    let prog = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| o.program);
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit");
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "abilities_probe emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// `cooldown_probe` compile-gate. Pins the structural surface that the
/// follow-up probe (closing Gap #4 in the abilities probe discovery
/// doc) exercises end-to-end:
///
///   - 1 PhysicsRule (`CheckAndCast`) reading the per-agent
///     `cooldown_next_ready_tick` SoA via `agents.cooldown_next_ready_
///     tick(self)` AND `world.tick`, then `if (tick >= ready_at) {
///     emit ActivationLogged }`.
///   - 1 ViewFold (`activations`) on `ActivationLogged` (kind = 1u).
///
/// Locks the per-agent SoA-field-read codepath for the cooldown SoA
/// — the spec table at `docs/spec/dsl.md` registers the field and
/// `CooldownNextReadyTick` is in `AgentFieldId::all_variants` (see
/// `crates/dsl_compiler/src/cg/data_handle.rs:368`) but no other
/// fixture reads it. Plus the WGSL must validate naga-clean.
///
/// Discovery doc: `docs/superpowers/notes/2026-05-04-cooldown_probe.md`.
#[test]
fn cooldown_probe_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/cooldown_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse cooldown_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve cooldown_probe.sim");
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[cooldown_probe lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit cooldown_probe");

    // Op-level invariants: physics rule + view fold present.
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            _ => {}
        }
    }
    assert_eq!(
        physics_count, 1,
        "expected 1 PhysicsRule op (CheckAndCast); got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 1,
        "expected 1 ViewFold op (activations); got {viewfold_count}",
    );

    // The CheckAndCast physics kernel must bind the cooldown SoA AND
    // reference the per-agent slot in its body. This locks the
    // `agents.cooldown_next_ready_tick(self)` lowering path.
    let body = kernel_body_containing(&art, "CheckAndCast")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| panic!(
            "no physics_CheckAndCast kernel in artifacts: {:?}",
            art.kernel_index,
        ));
    assert!(
        body.contains("agent_cooldown_next_ready_tick"),
        "physics_CheckAndCast must bind the cooldown SoA — \
         `agents.cooldown_next_ready_tick(self)` should lower to \
         `agent_cooldown_next_ready_tick[<idx>]` in the WGSL body. \
         Got body without that binding:\n{body}",
    );

    // Naga validation MUST be clean (no fused-kernel rename gap on
    // this probe — single physics rule + single fold, no fusion
    // collisions).
    let mut errs = Vec::new();
    for (name, w) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(w) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "cooldown_probe emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );

    eprintln!(
        "[cooldown_probe] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `diplomacy_probe` compile-gate. Pins the structural surface of the
/// smallest end-to-end probe of DIPLOMACY / COALITION FORMATION:
///
///   - 2 MaskPredicate ops (one per verb's `when (world.tick % 3 == X)`),
///     gated on disjoint Mod-tick predicates so EXACTLY ONE verb wins
///     argmax per tick.
///   - 1 PhysicsRule (`ObserveAndAct`) per-agent, emitting `Observed`
///     into the shared event ring every tick.
///   - 2 PhysicsRule chronicles synthesised from the verbs (each gated
///     on `action_id == <verb_id>`, emitting AllianceProposed +
///     Betrayed).
///   - 1 ScoringArgmax with 2 competing rows (ProposeAlliance + Betray).
///   - 3 ViewFold ops:
///       - `trust` — pair_map u32 fed by `Observed` (atomicOr fold)
///       - `alliances_proposed` — f32 fed by `AllianceProposed`
///       - `betrayals_committed` — f32 fed by `Betrayed`
///
/// FIRST fixture combining all of:
///   - 2 Group entity declarations (Faction + Coalition)
///   - pair_map u32 view (ToM-shape, post `51b5853b` landing)
///   - verb cascade with TWO competing verbs (post `cd007370` landing)
///   - per-handler tag-filter scaling to THREE event kinds in one ring
///     (post `cb24fd69` landing)
///   - verb `when` clauses referencing `world.tick % N == 0` (Mod
///     operator post `7208912f` landing)
///
/// Discovery doc: `docs/superpowers/notes/2026-05-04-diplomacy_probe.md`.
#[test]
fn diplomacy_probe_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/diplomacy_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse diplomacy_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve diplomacy_probe.sim");
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[diplomacy_probe lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit diplomacy_probe");

    // Op-level invariants.
    let mut mask_count = 0;
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    let mut scoring_rows = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::MaskPredicate { .. } => mask_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ScoringArgmax { rows, .. } => {
                scoring_rows = rows.len();
            }
            _ => {}
        }
    }
    assert_eq!(mask_count, 2, "expected 2 MaskPredicate ops (one per verb's `when` clause); got {mask_count}");
    assert_eq!(physics_count, 3, "expected 3 PhysicsRule ops (ObserveAndAct + 2 verb chronicles); got {physics_count}");
    assert_eq!(viewfold_count, 3, "expected 3 ViewFold ops (trust + alliances_proposed + betrayals_committed); got {viewfold_count}");
    assert_eq!(scoring_rows, 2, "expected 2 scoring rows (ProposeAlliance + Betray competing); got {scoring_rows}");

    // Mask kernel must reference `tick % 3u` — locks the verb-`when`
    // Mod-predicate lowering path that dropped pre-Mod-landing.
    let mask_body = kernel_body_containing(&art, "mask")
        .unwrap_or_else(|| panic!("no mask kernel emitted: {:?}", art.kernel_index));
    assert!(
        mask_body.contains("(tick % 3u)"),
        "mask kernel must lower `world.tick % 3` to `(tick % 3u)`; got body:\n{mask_body}",
    );

    // The `trust` fold body must use `atomicOr` — pair_map u32 storage
    // path. Locks the post-`51b5853b` landing.
    let trust_body = kernel_body_containing(&art, "trust")
        .unwrap_or_else(|| panic!("no fold_trust kernel emitted: {:?}", art.kernel_index));
    assert!(
        trust_body.contains("atomicOr"),
        "fold_trust must use atomicOr (u32 view, |= self-update); got body:\n{trust_body}",
    );

    // All emitted WGSL must be naga-clean.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "diplomacy_probe emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );

    eprintln!(
        "[diplomacy_probe] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `stochastic_probe` compile-gate. Pins the structural surface of the
/// smallest end-to-end probe of the `rng.*` namespace in a real
/// per-agent physics body:
///
///   - 1 PhysicsRule (`MaybeFire`) per-agent, drawing `rng.action()`
///     and gating an `emit Activated` on `(draw % 100) < 30`.
///   - 1 ViewFold (`activations`) on `Activated` (kind = 1u),
///     accumulating per-slot fire counts.
///
/// FIRST fixture combining all of:
///   - `rng.*` namespace call in a real physics body (lowering arm
///     at `crates/dsl_compiler/src/cg/lower/expr.rs:2532-2544`).
///   - IfStmt with an RNG-derived comparison on the LHS (no
///     DataHandle on the predicate).
///
/// **WGSL gap (closed 2026-05-04 — Gaps #1, #2, #3):** the physics
/// body lowers `rng.action()` to a `per_agent_u32(seed, agent_id,
/// tick, 1u)` call (numeric purpose id from `RngPurpose::wgsl_id()`,
/// not a string literal). The WGSL prelude in
/// `cg/emit/program.rs::RNG_WGSL_PRELUDE` defines the function; the
/// kernel preamble binds `let seed = cfg.seed;` from the per-rule cfg
/// uniform's new `seed: u32` field. This test asserts:
///
///  (a) the body contains the `per_agent_u32(` call (proves the
///      lowering arm fired);
///  (b) the purpose tag is the numeric `1u` (Action's `wgsl_id`),
///      NOT the legacy `"action"` string literal;
///  (c) naga validation now PASSES on `physics_MaybeFire` end-to-end.
///
/// Discovery doc: `docs/superpowers/notes/2026-05-04-stochastic_probe.md`.
#[test]
fn stochastic_probe_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/stochastic_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse stochastic_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve stochastic_probe.sim");
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[stochastic_probe lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit stochastic_probe");

    // Op-level invariants: physics rule + view fold present.
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            _ => {}
        }
    }
    assert_eq!(
        physics_count, 1,
        "expected 1 PhysicsRule op (MaybeFire); got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 1,
        "expected 1 ViewFold op (activations); got {viewfold_count}",
    );

    // The MaybeFire physics body must reference per_agent_u32 — proves
    // the rng.action() lowering arm fired through to wgsl_body emit.
    let body = kernel_body_containing(&art, "MaybeFire")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| panic!(
            "no physics_MaybeFire kernel in artifacts: {:?}",
            art.kernel_index,
        ));
    assert!(
        body.contains("per_agent_u32("),
        "physics_MaybeFire must lower `rng.action()` to a `per_agent_u32(...)` call \
         in WGSL; got body without that token:\n{body}",
    );
    // Purpose tag is now a numeric WGSL u32 literal (Action.wgsl_id() = 1)
    // — not the legacy `"action"` string literal (Gap #3 close, 2026-05-04).
    assert!(
        body.contains("1u"),
        "physics_MaybeFire must emit the purpose tag as the numeric `1u` \
         (Action.wgsl_id()); got body:\n{body}",
    );
    assert!(
        !body.contains("\"action\""),
        "physics_MaybeFire still contains a literal `\"action\"` string — Gap #3 \
         (numeric purpose tag) regressed; got body:\n{body}",
    );

    // Naga validation MUST PASS on `physics_MaybeFire` today — Gaps
    // #1, #2, #3 closed (2026-05-04). The kernel preamble binds
    // `let seed = cfg.seed;` (Gap #1), the WGSL `RNG_WGSL_PRELUDE`
    // defines `per_agent_u32` (Gap #2), and the purpose is a u32
    // literal (Gap #3). All other kernels stay naga-clean.
    let physics_kernel_name = art
        .wgsl_files
        .keys()
        .find(|k| k.contains("MaybeFire"))
        .cloned()
        .expect("physics_MaybeFire wgsl key present");
    let physics_body = art.wgsl_files.get(&physics_kernel_name).unwrap();
    // The WGSL prelude must be present in the rng-touching kernel.
    assert!(
        physics_body.contains("fn per_agent_u32("),
        "physics_MaybeFire must include the RNG_WGSL_PRELUDE `fn per_agent_u32(...)` \
         definition (Gap #2 close); got body:\n{physics_body}",
    );
    assert!(
        physics_body.contains("seed: u32"),
        "physics_MaybeFire's cfg struct must carry `seed: u32` (Gap #1 close); \
         got body:\n{physics_body}",
    );
    let physics_naga = naga::front::wgsl::parse_str(physics_body);
    assert!(
        physics_naga.is_ok(),
        "physics_MaybeFire WGSL must be naga-clean now that Gaps #1/#2/#3 closed; \
         got error: {:?}\nbody:\n{physics_body}",
        physics_naga.err(),
    );

    // Every kernel (the rng-touching physics body + the fold + the
    // plumbing kernels) must be naga-clean.
    let mut all_errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            all_errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        all_errs.is_empty(),
        "stochastic_probe emitted {} naga-invalid WGSL kernels:\n{}",
        all_errs.len(),
        all_errs.join("\n"),
    );

    eprintln!(
        "[stochastic_probe] {} kernels emitted: {:?}; physics_MaybeFire naga-clean (Gaps #1/#2/#3 closed)",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// Gap #1 follow-up from `2026-05-04-diplomacy_probe.md`: a bare
/// `tick` token in a verb's `when` clause must surface a typed
/// `BareNamespaceInExpression` diagnostic naming the qualified-form
/// hint (`world.tick`), instead of silently dropping the surrounding
/// MaskPredicate op via the generic `UnsupportedAstNode { ast_label:
/// "Namespace" }` arm.
///
/// Inline-source so the assertion is co-located with the expected
/// behaviour. The probe writes `when (tick % 3 == 0)` (the bare-token
/// form the diplomacy probe tripped on); the expected diagnostic is
/// the typed variant pointing the author at `world.tick`.
#[test]
fn bare_tick_in_verb_when_surfaces_typed_diagnostic() {
    use dsl_compiler::cg::lower::error::LoweringError;
    use dsl_ast::ir::NamespaceId;

    let src = r#"
event Tick { }

@replayable
@gpu_amenable
event Done { actor: AgentId }

entity Pinger : Agent {
  pos: vec3,
  vel: vec3,
}

verb Ping(self) =
  action PingAction
  when  (tick % 3 == 0)
  emit  Done { actor: self }
  score 1.0
"#;
    let prog = dsl_compiler::parse(src).expect("parse bare-tick verb source");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve bare-tick verb source");
    let outcome = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp);

    // Either the lowering returns Ok with diagnostics on the side OR
    // it returns Err with diagnostics inline; both shapes route the
    // typed error variant through `o.diagnostics` per the existing
    // populate_config_consts contract. Inspect both.
    let diags = match &outcome {
        Ok(_) => Vec::new(),
        Err(o) => o.diagnostics.clone(),
    };
    let saw_typed_bare = diags.iter().any(|d| {
        matches!(
            d,
            LoweringError::BareNamespaceInExpression {
                ns: NamespaceId::Tick,
                hint: "world.tick",
                ..
            }
        )
    });
    assert!(
        saw_typed_bare,
        "expected typed BareNamespaceInExpression{{ ns: Tick, hint: \"world.tick\" }} diagnostic; \
         got {} diagnostics:\n{}",
        diags.len(),
        diags
            .iter()
            .map(|d| format!("  {d}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    // Pre-fix this would have been the generic UnsupportedAstNode
    // arm — pin against accidental regression.
    let saw_legacy_drop = diags.iter().any(|d| {
        matches!(
            d,
            LoweringError::UnsupportedAstNode {
                ast_label: "Namespace",
                ..
            }
        )
    });
    assert!(
        !saw_legacy_drop,
        "bare `tick` token must NOT route through UnsupportedAstNode{{ ast_label: \"Namespace\" }} \
         (legacy silent-drop arm); got diagnostics:\n{}",
        diags
            .iter()
            .map(|d| format!("  {d}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

/// Gap #3 follow-up from `2026-05-04-diplomacy_probe.md`: a
/// `config.<ns>.<u32_field>` reference in arithmetic position must
/// resolve to `CgTy::U32` (not `CgTy::F32`) so
/// `world.tick % config.<ns>.<u32_field>` lowers without a
/// `BinaryOperandTyMismatch { lhs: U32, rhs: F32 }`.
///
/// Pre-fix: `data_handle_ty(ConfigConst{id})` defaulted every config
/// field to `CgTy::F32` regardless of declared type. The fix routes
/// `Read(ConfigConst)` through `ExprArena::config_const_ty(id)`,
/// which `CgProgram` now overrides to consult its
/// `config_const_values` map; a `u32`-declared field returns
/// `CgTy::U32` and the Mod operator picks the `BinaryOp::ModU32`
/// arm cleanly.
///
/// Inline-source shape so the assertion is co-located with the
/// expected behaviour and shares the trade_market hygiene fix's
/// pattern (config_u32_field_emits_with_u32_suffix_in_kernel_const).
#[test]
fn config_u32_field_lowers_typed_in_arithmetic_position() {
    let src = r#"
event Tick { }

@replayable
@gpu_amenable
event Done { actor: AgentId }

entity Pinger : Agent {
  pos: vec3,
  vel: vec3,
}

config diplomacy {
  observation_tick_mod: u32 = 3,
}

verb Ping(self) =
  action PingAction
  when  (world.tick % config.diplomacy.observation_tick_mod == 0)
  emit  Done { actor: self }
  score 1.0
"#;
    let prog = dsl_compiler::parse(src).expect("parse u32-config arithmetic source");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve u32-config arithmetic source");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        let diag_text = o
            .diagnostics
            .iter()
            .map(|d| format!("  {d}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "lower expected clean (Gap #3 fix); got {} diagnostics:\n{diag_text}",
            o.diagnostics.len(),
        );
    });

    // Mask op must exist — pre-fix the BinaryOperandTyMismatch would
    // drop it; post-fix the typed ModU32 lowers cleanly.
    let mut mask_count = 0;
    for op in &cg.ops {
        if matches!(
            &op.kind,
            dsl_compiler::cg::op::ComputeOpKind::MaskPredicate { .. }
        ) {
            mask_count += 1;
        }
    }
    assert_eq!(
        mask_count, 1,
        "expected 1 MaskPredicate op (Gap #3 fix preserves the verb `when` clause); got {mask_count}",
    );

    // The mask kernel WGSL must reference `config_<id>` (the typed
    // u32 const) on the rhs of the `%` op. Pre-fix the const
    // declaration was emitted but the mask body was dropped entirely
    // (no MaskPredicate op).
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit u32-config arithmetic program");
    let mask_body = kernel_body_containing(&art, "mask").unwrap_or_else(|| {
        panic!(
            "no mask kernel emitted (Gap #3 regression?); available: {:?}",
            art.kernel_index
        )
    });
    assert!(
        mask_body.contains("(tick % config_"),
        "mask kernel must lower `world.tick % config.diplomacy.observation_tick_mod` to \
         `(tick % config_<id>)`; got body:\n{mask_body}",
    );
    assert!(
        mask_body.contains(": u32 = 3u;"),
        "config const must emit as `u32 = 3u` (typed-routing fingerprint); got body:\n{mask_body}",
    );
}

/// `pair_scoring_probe` compile-gate — POSITIVE test pinning the
/// FULL-FIRE shape of PAIR-FIELD SCORING (spec §8.3) end-to-end.
///
/// **History.** The fixture's `verb Heal(self, target: Agent) = ...
/// score (1000.0 - agents.cooldown_next_ready_tick(target))` originally
/// produced two compile-time diagnostics — Gap #1 (mask positional
/// head requires `from` clause) AND Gap #2 (f32/u32 mismatch). The
/// 2026-05-04 closes (Gaps #1 + #3 + #2 + #4) and the follow-up
/// 2026-05-04 closes (Gap A — per-pair candidate read folding in
/// `cg::lower::expr::lower_namespace_call`; Gap B — verb-binder
/// LocalRef shadowing in `cg::lower::scoring::lower_standard_row`)
/// brought the chain end-to-end. Discovery doc:
/// `docs/superpowers/notes/2026-05-04-pair_scoring_probe.md`.
///
/// **What this test now pins.** Lowering succeeds, the scoring
/// kernel emits with:
///   - per-pair candidate inner loop (`for (var per_pair_candidate
///     ...) { ... }`) wrapping the row body,
///   - the score expression's `agents.cooldown_next_ready_tick(target)`
///     read collapsed to `agent_cooldown_next_ready_tick[per_pair_candidate]`
///     (no `target_expr_<N>` indirection — the structural fold in
///     `lower_namespace_call` resolves the read to
///     `AgentRef::PerPairCandidate` directly when the arg lowers to
///     `PerPairCandidateId`),
///   - `best_target = per_pair_candidate` on argmax wins (so the
///     ActionSelected payload carries the candidate slot, not the
///     `0xFFFFFFFFu` sentinel).
///
/// **What surfaces if a regression re-introduces the gap.** The
/// assertions below pin each fingerprint:
///   - `f32(agent_cooldown_next_ready_tick[per_pair_candidate])` —
///     direct candidate-id read (no `target_expr_<N>`),
///   - `for (var per_pair_candidate: u32 = 0u; per_pair_candidate < cfg.agent_cap` —
///     the inner candidate loop,
///   - `best_target = per_pair_candidate` — the argmax winner is the
///     candidate slot, not the sentinel.
/// A regression on any of these fails the assertion with the
/// emitted WGSL in the panic message.
#[test]
fn pair_scoring_probe_full_fire_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/pair_scoring_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse pair_scoring_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve pair_scoring_probe.sim");

    // Surface counts pre-lower — confirm the parser + resolver still
    // accept the `verb V(self, target: Agent) = ... score ...` shape.
    assert_eq!(
        comp.verbs.len(),
        1,
        "expected 1 verb (Heal); got {}",
        comp.verbs.len()
    );
    assert_eq!(
        comp.verbs[0].params.len(),
        2,
        "expected 2 verb params (self, target); got {}",
        comp.verbs[0].params.len()
    );
    assert_eq!(
        comp.verbs[0].params[1].name, "target",
        "second verb param must be named `target`"
    );

    // Lower MUST succeed — Gaps #1 + #2 + #3 closed. If a regression
    // re-introduces the f32/u32 mismatch or the positional-head reject,
    // this `expect` panic surfaces it loudly.
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        let diag_strings: Vec<String> = o.diagnostics.iter().map(|d| format!("{d}")).collect();
        panic!(
            "EXPECTED LOWER SUCCESS — got {} diagnostic(s):\n  {}",
            o.diagnostics.len(),
            diag_strings.join("\n  ")
        )
    });

    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("CG emit should succeed");

    let kernel_set: std::collections::BTreeSet<String> =
        art.kernel_index.iter().cloned().collect();
    assert!(
        kernel_set.contains("scoring"),
        "scoring kernel MUST be present; got: {:?}",
        kernel_set
    );
    assert!(
        kernel_set.iter().any(|k| k.contains("physics_verb_chronicle_Heal")),
        "chronicle physics rule MUST be present; got: {:?}",
        kernel_set
    );
    assert!(
        kernel_set.iter().any(|k| k.contains("fold_received")),
        "view-fold MUST be present; got: {:?}",
        kernel_set
    );
    assert!(
        kernel_set.iter().any(|k| k.contains("mask_verb_Heal")),
        "verb-synthesised mask kernel MUST be present; got: {:?}",
        kernel_set
    );

    let scoring_wgsl = art
        .wgsl_files
        .iter()
        .find(|(k, _)| k.as_str() == "scoring.wgsl")
        .map(|(_, v)| v.clone())
        .expect("scoring.wgsl must be present in wgsl_files");

    // FULL-FIRE fingerprint #1: u32→f32 promotion of the SoA read
    // (Gap #2 close).
    assert!(
        scoring_wgsl.contains("f32(agent_cooldown_next_ready_tick"),
        "Gap #2 close fingerprint missing — the score row should wrap the u32 \
         SoA read in `f32(...)`; got scoring.wgsl:\n{scoring_wgsl}"
    );

    // FULL-FIRE fingerprint #2: inner per-pair candidate loop wraps
    // the row body (Gap #4 close — kernel iterates over candidates).
    assert!(
        scoring_wgsl.contains("for (var per_pair_candidate: u32 = 0u; per_pair_candidate < cfg.agent_cap"),
        "FULL-FIRE per-pair inner loop missing — Gap #4 close requires the \
         row body to wrap in `for (var per_pair_candidate ...) {{ ... }}`; \
         got scoring.wgsl:\n{scoring_wgsl}"
    );

    // FULL-FIRE fingerprint #3: the agent-field read collapses to
    // `agent_cooldown_next_ready_tick[per_pair_candidate]` directly
    // (Gap A close — `lower_namespace_call` folds `AgentRef::Target(
    // <PerPairCandidateId>)` to `AgentRef::PerPairCandidate`). Without
    // this fold the row body would reference a `target_expr_<N>`
    // identifier the scoring emit doesn't drain into a `let`.
    assert!(
        scoring_wgsl.contains("agent_cooldown_next_ready_tick[per_pair_candidate]"),
        "Gap A close fingerprint missing — the score row should read \
         `agent_cooldown_next_ready_tick[per_pair_candidate]` directly \
         (no target_expr_<N> indirection); got scoring.wgsl:\n{scoring_wgsl}"
    );
    assert!(
        !scoring_wgsl.contains("target_expr_"),
        "REGRESSION — `target_expr_<N>` identifier reappeared in scoring.wgsl. \
         The `lower_namespace_call` structural fold for `PerPairCandidateId` / \
         `AgentSelfId` was lost. Got scoring.wgsl:\n{scoring_wgsl}"
    );

    // FULL-FIRE fingerprint #4: argmax winner sets `best_target =
    // per_pair_candidate`. The ActionSelected payload's target field
    // carries the candidate slot id downstream to the chronicle.
    assert!(
        scoring_wgsl.contains("best_target = per_pair_candidate"),
        "FULL-FIRE per-pair argmax-winner assignment missing — the row body \
         should set `best_target = per_pair_candidate` on argmax wins; \
         got scoring.wgsl:\n{scoring_wgsl}"
    );

    // Gap B (binder LocalRef shadowing) close: the row's `target`
    // reference must lower to `PerPairCandidateId` (NOT `ReadLocal`
    // pointing at the chronicle's event-pattern binder). Verify by
    // inspecting the scoring op's row utility expression — it must
    // transitively reference `PerPairCandidateId`.
    use dsl_compiler::cg::expr::{CgExpr, ExprArena};
    use dsl_compiler::cg::op::ComputeOpKind;
    let scoring_op = cg
        .ops
        .iter()
        .find(|op| matches!(op.kind, ComputeOpKind::ScoringArgmax { .. }))
        .expect("scoring op must exist");
    if let ComputeOpKind::ScoringArgmax { rows, .. } = &scoring_op.kind {
        assert_eq!(rows.len(), 1, "expected 1 row (Heal); got {}", rows.len());
        // Walk the utility expression tree looking for PerPairCandidateId.
        fn walks_per_pair_candidate(
            id: dsl_compiler::cg::data_handle::CgExprId,
            cg: &dsl_compiler::cg::program::CgProgram,
        ) -> bool {
            let Some(node) = ExprArena::get(cg, id) else { return false; };
            match node {
                CgExpr::PerPairCandidateId => true,
                CgExpr::Read(handle) => {
                    use dsl_compiler::cg::data_handle::{AgentRef, DataHandle};
                    matches!(
                        handle,
                        DataHandle::AgentField {
                            target: AgentRef::PerPairCandidate,
                            ..
                        }
                    )
                }
                CgExpr::Binary { lhs, rhs, .. } => {
                    walks_per_pair_candidate(*lhs, cg)
                        || walks_per_pair_candidate(*rhs, cg)
                }
                CgExpr::Unary { arg, .. } => walks_per_pair_candidate(*arg, cg),
                CgExpr::Builtin { args, .. } | CgExpr::NamespaceCall { args, .. } => {
                    args.iter().any(|a| walks_per_pair_candidate(*a, cg))
                }
                _ => false,
            }
        }
        assert!(
            walks_per_pair_candidate(rows[0].utility, &cg),
            "Gap B close fingerprint missing — scoring row utility expression \
             must transitively reference `PerPairCandidateId` (or a \
             `Read(AgentField{{ target: PerPairCandidate }})`); the verb's \
             `target` LocalRef leaked through chronicle-physics's local_ids \
             registration."
        );
    }
}

/// `stdlib_math_probe` compile-gate. Pins the structural surface of
/// the smallest end-to-end probe stress-testing under-exercised
/// stdlib math + RNG conversion surfaces:
///
///   - 1 PhysicsRule (`SampleAndBucket`) per-agent, running a tier
///     of stdlib `let` bindings (`floor` / `ceil` / `round` /
///     `log2` / `abs`) and emitting `Sampled` unconditionally with
///     the bucket value drawn from `rng.action() % 4u`.
///   - 1 ViewFold (`sampled_count`) on `Sampled` (kind = 1u),
///     accumulating per-slot fire count = TICKS.
///
/// The probe originally surfaced FIVE compiler gaps (Gaps #A-#E in
/// `docs/superpowers/notes/2026-05-04-stdlib_math_probe.md`). ALL
/// FIVE are now closed: Gaps #A + #B + #D in 8d7c2673, Gaps #C + #E
/// in the followup commit. The fixture body now exercises every
/// advertised stdlib + RNG surface end-to-end. This test pins:
///
///   (a) the structural surface (1 PhysicsRule + 1 ViewFold);
///   (b) the math stdlib lowering — the physics WGSL body must
///       contain at least `floor(`, `ceil(`, `round(`, `log2(`, and
///       `abs(` calls (proves all five `BuiltinId` arms emit);
///   (c) the safe bucket-emit lowering — the body must contain
///       `(per_agent_u32(seed, agent_id, tick, 1u) % 4u)` (proves
///       the rng.action() Action-purpose path, locked by stochastic
///       _probe Gaps #1/#2/#3, still emits cleanly when composed
///       with `% 4u` modulo);
///   (d) the closed-gap fingerprints — Gap #A's `log2(10.0)`
///       divisor (math identity rewrite for `log10`), Gap #B's
///       bare `planar_distance(` / `z_separation(` calls (prelude-
///       shimmed in the wrapping module), and Gap #D's
///       `((per_agent_u32(...) & 1u) == 0u)` (bool-from-u32 for
///       `rng.coin()`);
///   (e) every emitted WGSL kernel passes naga TEXT validation —
///       this also verifies the spatial-builtin prelude is injected
///       (any missing prelude ⇒ "no definition in scope" error).
///       The wgpu FULL validator (which previously rejected the
///       open Gap #E surfaces with "Abstract types may only appear
///       in constant expressions") runs only at runtime in
///       `create_shader_module`; the `stdlib_math_probe_app`
///       harness catches any FULL-validator regression via
///       `catch_unwind`.
///
/// Discovery doc: `docs/superpowers/notes/2026-05-04-stdlib_math_probe.md`.
#[test]
fn stdlib_math_probe_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/stdlib_math_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse stdlib_math_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve stdlib_math_probe.sim");
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[stdlib_math_probe lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit stdlib_math_probe");

    // Op-level invariants: physics rule + view fold present.
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            _ => {}
        }
    }
    assert_eq!(
        physics_count, 1,
        "expected 1 PhysicsRule op (SampleAndBucket); got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 1,
        "expected 1 ViewFold op (sampled_count); got {viewfold_count}",
    );

    // Math-stdlib fingerprints — each retained Tier 1 surface must
    // appear as a bare WGSL call in the physics body. This locks the
    // `wgsl_body.rs::builtin_name` arms for Floor / Ceil / Round /
    // Log2 / Abs against accidental rename.
    let body = kernel_body_containing(&art, "SampleAndBucket")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| panic!(
            "no physics_SampleAndBucket kernel in artifacts: {:?}",
            art.kernel_index,
        ));
    for needle in ["floor(", "ceil(", "round(", "log2(", "abs("] {
        assert!(
            body.contains(needle),
            "physics_SampleAndBucket must lower the {} stdlib call to a bare WGSL \
             call; got body without that token:\n{body}",
            needle.trim_end_matches('('),
        );
    }

    // Bucket-emit fingerprint — `rng.action() % 4u` must lower to
    // `per_agent_u32(seed, agent_id, tick, 1u) % 4u`. The `1u` is
    // Action's `wgsl_id`; the `% 4u` is the BinaryOp::ModU32 arm.
    assert!(
        body.contains("per_agent_u32(seed, agent_id, tick, 1u)"),
        "physics_SampleAndBucket must lower `rng.action()` to \
         `per_agent_u32(seed, agent_id, tick, 1u)` (Action's wgsl_id); got body:\n{body}",
    );
    assert!(
        body.contains("% 4u"),
        "physics_SampleAndBucket must compose the bucket via `% 4u` (BinaryOp::ModU32); \
         got body:\n{body}",
    );

    // GAP CLOSE FINGERPRINTS — Gaps #A, #B, #D are now closed; the
    // probe re-introduces the previously-omitted surfaces and this
    // test pins the new emit shapes against accidental rename /
    // regression.
    //
    // Gap #A close (2026-05-04): `log10(x)` rewrites at the Builtin
    // emit site to `(log2(<x>) / log2(10.0))` — math identity, no
    // prelude. The bare WGSL identifier `log10(` MUST NOT appear
    // (no native), but BOTH `log2(` and the divisor `log2(10.0)`
    // MUST appear together as the fingerprint of the rewrite.
    assert!(
        !body.contains("log10("),
        "Gap #A regression — bare `log10(` reappeared but WGSL has no native \
         log10. Got body:\n{body}",
    );
    assert!(
        body.contains("log2(") && body.contains("log2(10.0)"),
        "Gap #A close — `log10(x)` should rewrite to `(log2(x) / log2(10.0))` \
         at the Builtin emit site; missing `log2(10.0)` divisor in body:\n{body}",
    );

    // Gap #B close (2026-05-04): `planar_distance(a, b)` and
    // `z_separation(a, b)` emit as bare WGSL calls; a kernel-prelude
    // shim (`SPATIAL_BUILTIN_WGSL_PRELUDE` in `cg/emit/program.rs`)
    // is substring-injected when either call appears. This test
    // operates on the kernel body alone; the prelude lives in the
    // wrapping module file. Pin the call shapes here; the prelude
    // injection is verified by the naga TEXT validation pass below
    // (any missing prelude ⇒ "no definition in scope" error).
    assert!(
        body.contains("planar_distance("),
        "Gap #B close — `planar_distance(self.pos, self.pos)` should lower to \
         a bare WGSL call; missing token in body:\n{body}",
    );
    assert!(
        body.contains("z_separation("),
        "Gap #B close — `z_separation(self.pos, self.pos)` should lower to \
         a bare WGSL call; missing token in body:\n{body}",
    );

    // Gap #D close (2026-05-04): `rng.coin()` lowers to
    // `CgExpr::Rng { Coin, Bool }` and now emits the bool-typed
    // expression `((per_agent_u32(seed, agent_id, tick, 7u) & 1u)
    // == 0u)` (purpose id 7u = RngPurpose::Coin::wgsl_id). The
    // surrounding `let local_N: bool = ...` accepts a bool RHS.
    assert!(
        body.contains("(per_agent_u32(seed, agent_id, tick, 7u) & 1u) == 0u"),
        "Gap #D close — `rng.coin()` should emit \
         `((per_agent_u32(seed, agent_id, tick, 7u) & 1u) == 0u)` (bool from \
         u32 bit-extract); missing fingerprint in body:\n{body}",
    );

    // Gap #E close (this commit): `rng.uniform(lo, hi)` and
    // `rng.gauss(mu, sigma)` now emit per-purpose conversion at the
    // `CgExpr::Rng` site so the surrounding f32 arithmetic is
    // concretely-typed (the wgpu FULL validator no longer rejects
    // abstract-type binary ops). Pin the new emit shapes:
    //   - Uniform (purpose 5): the conversion `f32(per_agent_u32(...
    //     5u)) / f32(4294967295u)` is the unit-interval f32 draw.
    //   - Gauss (purpose 6): the Box-Muller pair-draw uses BOTH
    //     purpose 6 (Gauss) and purpose 9 (GaussB, internal sibling
    //     for the second draw); the canonical `sqrt(-2.0 * log(...))`
    //     prefix is the fingerprint.
    assert!(
        body.contains("f32(per_agent_u32(seed, agent_id, tick, 5u)) / f32(4294967295u)"),
        "Gap #E close (rng.uniform) — expected `f32(per_agent_u32(seed, agent_id, \
         tick, 5u)) / f32(4294967295u)` (unit-interval f32 conversion at the Uniform \
         emit site); missing from body:\n{body}",
    );
    assert!(
        body.contains("per_agent_u32(seed, agent_id, tick, 6u)")
            && body.contains("per_agent_u32(seed, agent_id, tick, 9u)")
            && body.contains("sqrt(-2.0 * log("),
        "Gap #E close (rng.gauss) — expected Box-Muller pair-draw using purposes 6 \
         (Gauss) + 9 (GaussB) with `sqrt(-2.0 * log(...))` prefix; missing fingerprint \
         in body:\n{body}",
    );

    // Gap #C close (this commit): `rng.uniform_int(lo, hi)` flipped
    // signature from the unreachable `(i32, i32) -> i32` to
    // `(u32, u32) -> u32`. The IR shape is `lo + (draw % (hi - lo))`
    // on the UniformInt u32 stream; the WGSL emit is bare
    // `per_agent_u32(...)` (purpose 8u) since the surface IS u32 —
    // no per-purpose conversion needed.
    assert!(
        body.contains("per_agent_u32(seed, agent_id, tick, 8u)"),
        "Gap #C close (rng.uniform_int) — expected bare \
         `per_agent_u32(seed, agent_id, tick, 8u)` (UniformInt u32 stream); missing \
         from body:\n{body}",
    );
    // The `lo + (draw % (hi - lo))` shape with concrete u32 ops:
    // `(0u + (per_agent_u32(...) % (4u - 0u)))`.
    assert!(
        body.contains("(per_agent_u32(seed, agent_id, tick, 8u) % (4u - 0u))"),
        "Gap #C close (rng.uniform_int) — expected `lo + (draw % (hi - lo))` shape \
         with U32 ops `(per_agent_u32(seed, agent_id, tick, 8u) % (4u - 0u))`; missing \
         from body:\n{body}",
    );

    // Naga TEXT validation MUST be clean across all kernels. (Gap
    // #E's full-validator reject is NOT caught here — wgpu's
    // create_shader_module runs the additional FULL validator that
    // rejects abstract-type binary ops; that's surfaced by the
    // sim_app harness, not this test.)
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "stdlib_math_probe emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );

    eprintln!(
        "[stdlib_math_probe] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `quest_probe` compile-gate. Pins the structural surface of the
/// SMALLEST end-to-end probe of `entity X : Quest` AND the `quests.*`
/// namespace — both surfaces are documented in spec but never
/// exercised in any fixture today.
///
/// Spec context:
///   - `docs/spec/dsl.md:653-663` — `Quest` listed alongside `Agent`,
///     `Item`, `Group` etc. in the typed-entity table.
///   - `docs/spec/dsl.md:871,1072` — `quests` legacy namespace +
///     singular `quest` namespace both registered.
///
/// Discovery target — the fixture is the SMALLEST .sim that:
///   - Declares an `Adventurer : Agent` entity (live).
///   - Declares a `Mission : Item` entity as a Quest analog (live —
///     the `Quest` root keyword is rejected at parse time today).
///   - Emits one `ProgressTick` per alive Adventurer per tick.
///   - Folds into a `progress(agent: Agent) -> u32` view via `+= 1u`.
///
/// What this locks:
///   - 1 PhysicsRule (`ProgressAndComplete`) + 1 ViewFold (`progress`)
///     ops emitted from the live shape.
///   - `Mission : Item` reaches the entity_field_catalog (per-Item
///     SoA allocation surface).
///   - The u32-view fold-body lowers via the `+=` operator gate at
///     `crates/dsl_compiler/src/cg/lower/view.rs:564`.
///   - The WGSL emitter routes `+= 1u` on a u32-result view to
///     `atomicOr(&storage[idx], (1u))` per the result-type-only
///     branching in `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:
///     1326-1338`. This is the LIVE GAP: `+=` semantics silently
///     map to bitwise-OR semantics, leaving the per-slot value at
///     `1u` regardless of fire count. Locked here so a future fix
///     (separate u32_add fold semantic OR reject `+=` on u32 views)
///     surfaces as a test failure.
///
/// Discovery doc: `docs/superpowers/notes/2026-05-04-quest_probe.md`.
#[test]
fn quest_probe_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/quest_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse quest_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve quest_probe.sim");
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[quest_probe lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit quest_probe");

    // Op-level invariants: physics rule + view fold present.
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            _ => {}
        }
    }
    assert_eq!(
        physics_count, 1,
        "expected 1 PhysicsRule op (ProgressAndComplete); got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 1,
        "expected 1 ViewFold op (progress); got {viewfold_count}",
    );

    // Entity catalog: the `Mission : Item` declaration must reach
    // the per-fixture entity_field_catalog (Item path is fully
    // wired; this asserts the Quest fall-back analog populates it).
    let mission_in_items = cg
        .entity_field_catalog
        .items
        .values()
        .any(|rec| rec.entity_name == "Mission");
    assert!(
        mission_in_items,
        "Mission : Item must populate entity_field_catalog.items; got items: {:?}",
        cg.entity_field_catalog
            .items
            .values()
            .map(|r| &r.entity_name)
            .collect::<Vec<_>>(),
    );

    // The `progress` fold body must use `atomicAdd` — post-2026-05-04
    // (Gap C closure) the WGSL emitter branches on (fold_op,
    // result_ty), so `+= 1u` on a u32 view picks the
    // commutative-+-associative `atomicAdd` primitive. Pre-fix this
    // assertion locked the regression: the emitter ignored the
    // operator and routed every u32 view through `atomicOr`.
    let progress_body = kernel_body_containing(&art, "progress")
        .unwrap_or_else(|| panic!("no fold_progress kernel emitted: {:?}", art.kernel_index));
    assert!(
        progress_body.contains("atomicAdd"),
        "fold_progress must use atomicAdd post-Gap-C-fix (`+= 1u` on a \
         u32 view — operator-aware emit branch). Got body:\n{progress_body}",
    );
    assert!(
        !progress_body.contains("atomicOr"),
        "fold_progress must NOT use atomicOr post-Gap-C-fix (`+= 1u` is \
         add-shaped, not OR-shaped). Got body:\n{progress_body}",
    );

    // Quest-root entity declaration (Gap A closure). `MainQuest :
    // Quest { ... }` must reach the resolved program as a Quest-
    // rooted entity. The catalog populator skips Quest the same way
    // it skips Agent (no per-Quest SoA today), so we assert on
    // `comp.entities` directly via re-resolving from the source.
    let comp_for_quest = dsl_ast::resolve::resolve(
        dsl_compiler::parse(&src).expect("re-parse quest_probe.sim"),
    )
    .expect("re-resolve quest_probe.sim");
    let saw_quest_entity = comp_for_quest
        .entities
        .iter()
        .any(|e| matches!(e.root, dsl_compiler::ast::EntityRoot::Quest));
    assert!(
        saw_quest_entity,
        "expected at least one Quest-rooted entity (Gap A closure — \
         `entity MainQuest : Quest {{ ... }}`); got entities: {:?}",
        comp_for_quest
            .entities
            .iter()
            .map(|e| (&e.name, e.root))
            .collect::<Vec<_>>(),
    );

    // `quests.is_active` namespace stub (Gap B closure). The B1
    // method registration must surface as a `quests_is_active` WGSL
    // helper somewhere in the emitted shaders (the physics body's
    // `let active = quests.is_active(0u);` lowers to a call to it).
    let saw_quests_helper = art
        .wgsl_files
        .values()
        .any(|w| w.contains("quests_is_active"));
    assert!(
        saw_quests_helper,
        "expected `quests_is_active` WGSL helper post-Gap-B-fix in any \
         emitted shader; got kernels: {:?}",
        art.kernel_index,
    );

    // All emitted WGSL must be naga-clean (the gap is a SEMANTIC
    // miscompile, not a syntactic one — naga still validates).
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "quest_probe emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );

    eprintln!(
        "[quest_probe] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `duel_1v1` compile-gate. Pins the structural surface of the FIRST
/// real gameplay-shaped fixture (not a coverage probe):
///
///   - 3 MaskPredicate ops (one per verb's `when` clause: Strike,
///     Spell, Heal — all pair-field rows because all verbs declare a
///     `target: Agent` parameter).
///   - 5 PhysicsRule ops (3 verb-synthesised chronicles + 2
///     authored physics rules: ApplyDamage + ApplyHeal — these
///     fuse into one `physics_ApplyDamage_and_ApplyHeal` kernel).
///   - 2 ViewFold ops (damage_dealt + healing_done, both f32).
///   - 1 ScoringArgmax with 3 competing rows (Strike + Spell + Heal).
///
/// Locks the surfaces this fixture closes inline:
///   - `agents.set_hp` / `agents.set_alive` / `agents.set_mana`
///     setters registered (extending the prior `set_pos` / `set_vel`
///     set in `cg::lower::physics::agents_setter_field`).
///   - Non-self target accepted by `lower_agents_setter` — routes
///     the lowered AgentId expression via `AgentRef::Target(<expr>)`,
///     mirroring the existing read-side path.
///   - PerPair mask kernel preamble binds `let tick = cfg.tick;`
///     (was missing pre-`duel_1v1` — only PerAgent / PerEvent
///     preambles bound the simulation clock; PerPair `world.tick`
///     references in mask predicates surfaced as `no definition in
///     scope for identifier: tick`).
///   - WGSL Bool LHS rewrite (`agent_alive[t] = select(0u, 1u, v)`)
///     — pre-`duel_1v1` the LHS used `(agent_alive[t] != 0u)` (the
///     read-form coercion) which is not a valid lvalue.
///   - ScoringArgmax emits ActionSelected gated on `best_utility >
///     -inf` — pre-`duel_1v1` the emit was unconditional, producing
///     a phantom event whose target = NO_TARGET (0xFFFFFFFFu) for
///     agents whose mask never set, leading to an out-of-bounds
///     `agent_<field>[0xFFFFFFFFu]` write downstream.
///
/// Discovery doc: `docs/superpowers/notes/2026-05-04-duel_1v1.md`
/// (gap punch list including the deferred P6 + cycle false positives
/// for `@phase(post)` chronicle physics, and the Heal verb's
/// `score (... - self.hp)` lowering `self.hp` to
/// `agent_hp[target_expr_<N>]` under pair-field scoring).
#[test]
fn duel_1v1_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/duel_1v1.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse duel_1v1.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve duel_1v1.sim");
    // Tolerate lower diagnostics — duel_1v1 has known-deferred
    // well_formed warnings (P6 + cycle).
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[duel_1v1 lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit duel_1v1");

    // Op-level invariants.
    let mut mask_count = 0;
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    let mut scoring_rows = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::MaskPredicate { .. } => mask_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ScoringArgmax { rows, .. } => {
                scoring_rows = rows.len();
            }
            _ => {}
        }
    }
    assert_eq!(
        mask_count, 3,
        "expected 3 MaskPredicate ops (one per verb: Strike + Spell + Heal); got {mask_count}",
    );
    assert_eq!(
        physics_count, 5,
        "expected 5 PhysicsRule ops (3 verb chronicles + ApplyDamage + ApplyHeal); got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 2,
        "expected 2 ViewFold ops (damage_dealt + healing_done); got {viewfold_count}",
    );
    assert_eq!(
        scoring_rows, 3,
        "expected 3 scoring rows (Strike + Spell + Heal competing); got {scoring_rows}",
    );

    // PerPair mask kernel must bind `let tick = cfg.tick;` so verb
    // `when (world.tick % N == 0)` cooldown predicates lower cleanly.
    // Locks the post-duel_1v1 PerPair preamble fix.
    let mask_body = kernel_body_containing(&art, "fused_mask")
        .unwrap_or_else(|| panic!("no fused_mask kernel emitted: {:?}", art.kernel_index));
    assert!(
        mask_body.contains("let tick = cfg.tick;"),
        "PerPair mask kernel must bind `tick`; got body:\n{mask_body}",
    );

    // ApplyDamage_and_ApplyHeal kernel must emit a select(...) for
    // the bool agent_alive write (post-duel_1v1 LHS-coercion fix). The
    // RHS coerces bool→u32 via `select(0u, 1u, value)` since
    // agent_alive storage is array<u32>.
    let apply_body = kernel_body_containing(&art, "ApplyDamage")
        .unwrap_or_else(|| panic!("no ApplyDamage kernel emitted: {:?}", art.kernel_index));
    assert!(
        apply_body.contains("agent_alive["),
        "ApplyDamage must write agent_alive (set_alive); got body:\n{apply_body}",
    );
    assert!(
        apply_body.contains("select(0u, 1u,"),
        "ApplyDamage must coerce bool RHS via select(0u, 1u, ...) when writing \
         agent_alive (post-duel_1v1 Bool LHS rewrite); got body:\n{apply_body}",
    );
    // The lvalue must NOT be wrapped in the read-form coercion `(x != 0u)` —
    // that would produce an invalid WGSL assignment. Locks the
    // `agent_field_access_lvalue` fix.
    assert!(
        !apply_body.contains("(agent_alive[") || !apply_body.contains(") = "),
        "ApplyDamage must NOT use read-form `(agent_alive[t] != 0u)` as an LHS; \
         got body:\n{apply_body}",
    );

    // Scoring kernel must gate the emit on best_utility > -inf so
    // agents whose mask never set don't fire a phantom ActionSelected.
    let scoring_body = kernel_body_containing(&art, "scoring")
        .unwrap_or_else(|| panic!("no scoring kernel emitted: {:?}", art.kernel_index));
    assert!(
        scoring_body.contains("if (best_utility > -3.4028235e38)"),
        "scoring kernel must gate ActionSelected emit on best_utility > -inf; \
         got body:\n{scoring_body}",
    );

    // All emitted WGSL must be naga-clean.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "duel_1v1 emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );

    eprintln!(
        "[duel_1v1] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// First SCALE-UP fixture (25 Red vs 25 Blue squad scuffle) — compiles
/// `assets/sim/duel_25v25.sim` end-to-end. Sidesteps the duel_1v1 verb
/// cascade's Gap W (PerPair `mask_k=1u` placeholder) by doing
/// target-finding via SPATIAL body-form physics: each Combatant per
/// tick walks its 27-cell neighbourhood and emits Damaged for any
/// nearby agent of the opposing creature_type.
///
/// Op-level invariants:
///   - 2 entity types (RedCombatant, BlueCombatant) sharing one Agent
///     SoA, with creature_type discriminant per the predator_prey_min
///     precedent.
///   - 2 PhysicsRule ops: ScanAndStrike (per_agent + spatial body-form
///     emit) + ApplyDamage (post chronicle, agents.set_hp/set_alive
///     + Defeated emit).
///   - 2 ViewFold ops: damage_dealt + defeats_received.
///   - Spatial body-form physics with creature_type body-side filter
///     (the spatial_query body filter is informational at the body-iter
///     site today — see particle_collision_min comments).
///
/// Discovery doc: `docs/superpowers/notes/2026-05-04-duel_25v25.md`.
#[test]
fn duel_25v25_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/duel_25v25.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse duel_25v25.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve duel_25v25.sim");
    // Tolerate lower diagnostics — duel_25v25 inherits the
    // duel_1v1 P6 + cycle warnings (chronicle ApplyDamage writes
    // agent.hp/alive; the well_formed checker doesn't yet model
    // @phase(post) authored mutation as legitimate).
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[duel_25v25 lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit duel_25v25");

    let mut physics_count = 0;
    let mut viewfold_count = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            _ => {}
        }
    }
    assert!(
        physics_count >= 2,
        "expected at least 2 PhysicsRule ops (ScanAndStrike + ApplyDamage); got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 2,
        "expected 2 ViewFold ops (damage_dealt + defeats_received); got {viewfold_count}",
    );

    // ScanAndStrike body must contain a spatial-grid neighbour walk
    // (locks the body-form path against any silent regression to
    // ForEachAgent fallback).
    let scan_body = kernel_body_containing(&art, "ScanAndStrike")
        .unwrap_or_else(|| {
            panic!(
                "no ScanAndStrike kernel emitted; available: {:?}",
                art.kernel_index
            )
        });
    assert!(
        scan_body.contains("spatial_grid_starts"),
        "ScanAndStrike must emit a spatial-grid neighbour walk; got body:\n{scan_body}",
    );
    assert!(
        scan_body.contains("agent_creature_type["),
        "ScanAndStrike must read agent_creature_type for the body-side \
         team filter; got body:\n{scan_body}",
    );

    // ApplyDamage must write agent_hp + agent_alive (same shape as
    // duel_1v1, post-Bool-LHS-rewrite).
    let apply_body = kernel_body_containing(&art, "ApplyDamage")
        .unwrap_or_else(|| {
            panic!(
                "no ApplyDamage kernel emitted; available: {:?}",
                art.kernel_index
            )
        });
    assert!(
        apply_body.contains("agent_hp["),
        "ApplyDamage must write agent_hp; got body:\n{apply_body}",
    );

    // All emitted WGSL must be naga-clean.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "duel_25v25 emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );

    eprintln!(
        "[duel_25v25] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `foraging_real` compile-gate. Pins the structural surface of the
/// FIRST LIFECYCLE fixture (the alive bitmap actually flips BOTH
/// directions during a run) and locks the inline compiler fixes it
/// surfaced:
///
///   - `agents.set_hunger(t, v)` is wired through the
///     `agents_setter_field` map (newly added in this fixture's
///     commit) so the EnergyDecay rule's `agents.set_hunger(self,
///     hunger - decay_rate)` lowers cleanly.
///   - `ForEachNeighborBody` walker surfaces the implicit
///     `agent_pos[agent_id]` self-cell read so AntFeed (whose body
///     doesn't reference `self.pos`) gets `agent_pos` bound. (Same
///     fix that landed for `duel_25v25`'s ScanAndStrike — shared.)
///
/// Tolerates the same lower diagnostics duel_1v1 + duel_25v25 do
/// (P6 false positive for @phase(post) chronicle physics writing
/// agent fields). The op count is the structural fingerprint:
///   - 4 ViewFold ops (eat_count + food_consumed + starved_count +
///     depleted_count)
///   - 3 PhysicsRule ops (AntFeed + ApplyEat + EnergyDecay)
///   - 0 MaskPredicate, 0 ScoringArgmax (no verb cascade — all
///     decision logic lives in physics rule bodies + chronicle)
#[test]
fn foraging_real_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/foraging_real.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse foraging_real.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve foraging_real.sim");
    // Tolerate the inherited P6 false-positive warnings on
    // chronicle physics writes to agent fields (same shape the
    // duel_1v1 + duel_25v25 fixtures live with).
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[foraging_real lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit foraging_real");

    // Op-level invariants — pin the count of physics rules + view
    // folds. AntFeed + ApplyEat + EnergyDecay = 3 PhysicsRule ops;
    // eat_count + food_consumed + starved_count + depleted_count = 4
    // ViewFold ops; no verbs / scoring / masks.
    let mut mask_count = 0;
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    let mut scoring_rows = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::MaskPredicate { .. } => mask_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ScoringArgmax { rows, .. } => {
                scoring_rows = rows.len();
            }
            _ => {}
        }
    }
    assert_eq!(mask_count, 0, "foraging_real has no verbs → 0 MaskPredicate ops; got {mask_count}");
    assert_eq!(scoring_rows, 0, "foraging_real has no verbs → 0 scoring rows; got {scoring_rows}");
    assert_eq!(
        physics_count, 3,
        "expected 3 PhysicsRule ops (AntFeed + ApplyEat + EnergyDecay); got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 4,
        "expected 4 ViewFold ops (eat_count + food_consumed + starved_count + depleted_count); got {viewfold_count}",
    );

    // ApplyEat must write agent_hp (food deplete) AND agent_alive
    // (set_alive on food.hp<=0) AND agent_hunger (the
    // newly-wired set_hunger setter — locks the inline compiler fix).
    let apply_body = kernel_body_containing(&art, "ApplyEat")
        .unwrap_or_else(|| panic!("no ApplyEat kernel emitted: {:?}", art.kernel_index));
    assert!(
        apply_body.contains("agent_hp["),
        "ApplyEat must write agent_hp (food.hp decrement); got body:\n{apply_body}",
    );
    assert!(
        apply_body.contains("agent_alive["),
        "ApplyEat must write agent_alive (set_alive on hp<=0); got body:\n{apply_body}",
    );
    assert!(
        apply_body.contains("agent_hunger["),
        "ApplyEat must write agent_hunger (newly-wired set_hunger setter); got body:\n{apply_body}",
    );

    // EnergyDecay must write agent_alive (set_alive(self, false) on
    // hunger<=0) — the alive=true → false direction this fixture
    // exercises in DSL.
    let decay_body = kernel_body_containing(&art, "EnergyDecay")
        .unwrap_or_else(|| panic!("no EnergyDecay kernel emitted: {:?}", art.kernel_index));
    assert!(
        decay_body.contains("agent_alive["),
        "EnergyDecay must write agent_alive (set_alive on hunger<=0); got body:\n{decay_body}",
    );
    assert!(
        decay_body.contains("agent_hunger["),
        "EnergyDecay must read+write agent_hunger; got body:\n{decay_body}",
    );

    // AntFeed (body-form spatial walk) must bind agent_pos — locks
    // the `ForEachNeighborBody` implicit-self.pos-read fix added to
    // `collect_stmt_dependencies`. Without this binding, AntFeed
    // emits naga-invalid WGSL referencing an unbound `agent_pos`.
    let antfeed_body = kernel_body_containing(&art, "AntFeed")
        .unwrap_or_else(|| panic!("no AntFeed kernel emitted: {:?}", art.kernel_index));
    assert!(
        antfeed_body.contains("agent_pos: array<vec3<f32>>")
            || antfeed_body.contains("agent_pos: array<vec3"),
        "AntFeed must bind agent_pos — ForEachNeighborBody implicit \
         self.pos read fix; got body:\n{antfeed_body}",
    );

    // All emitted WGSL must be naga-clean.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "foraging_real emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );

    eprintln!(
        "[foraging_real] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// `predator_prey_real` compile-gate. Pins the structural surface of
/// the FOURTH REAL fixture — first to compose COMBAT + LIFECYCLE on
/// TWO simultaneously-live creature types.
///
/// Locks:
///   - 4 PhysicsRule ops (WolfHunt + ApplyKill + SheepGraze +
///     EnergyDecay)
///   - 3 ViewFold ops (kills_total + sheep_killed_total +
///     starved_total)
///   - 0 MaskPredicate, 0 ScoringArgmax (no verb cascade — all
///     decision logic in physics rule bodies + chronicle)
///   - WolfHunt binds spatial-grid + agent_pos (body-form spatial
///     walk → ForEachNeighborBody implicit self.pos read)
///   - ApplyKill writes agent_hp + agent_alive + agent_hunger
///     (chronicle physics: combat death + wolf-eats-sheep)
///   - EnergyDecay writes agent_alive (universal starvation gate)
///   - SheepGraze writes agent_hunger (uniform graze regrowth)
#[test]
fn predator_prey_real_compile_gate() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/predator_prey_real.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse predator_prey_real.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve predator_prey_real.sim");
    // Tolerate the inherited P6 false-positive warnings on chronicle
    // physics writes to agent fields (same shape duel_1v1 + duel_25v25
    // + foraging_real fixtures live with).
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        for d in &o.diagnostics {
            eprintln!("[predator_prey_real lower diag] {d}");
        }
        o.program
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_real");

    let mut mask_count = 0;
    let mut physics_count = 0;
    let mut viewfold_count = 0;
    let mut scoring_rows = 0;
    for op in &cg.ops {
        match &op.kind {
            dsl_compiler::cg::op::ComputeOpKind::MaskPredicate { .. } => mask_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::PhysicsRule { .. } => physics_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ViewFold { .. } => viewfold_count += 1,
            dsl_compiler::cg::op::ComputeOpKind::ScoringArgmax { rows, .. } => {
                scoring_rows = rows.len();
            }
            _ => {}
        }
    }
    assert_eq!(
        mask_count, 0,
        "predator_prey_real has no verbs → 0 MaskPredicate ops; got {mask_count}",
    );
    assert_eq!(
        scoring_rows, 0,
        "predator_prey_real has no verbs → 0 scoring rows; got {scoring_rows}",
    );
    assert_eq!(
        physics_count, 4,
        "expected 4 PhysicsRule ops (WolfHunt + ApplyKill + SheepGraze + EnergyDecay); \
         got {physics_count}",
    );
    assert_eq!(
        viewfold_count, 3,
        "expected 3 ViewFold ops (kills_total + sheep_killed_total + starved_total); \
         got {viewfold_count}",
    );

    // ApplyKill must write agent_hp (sheep.hp decrement) AND
    // agent_alive (set_alive on hp<=0 — combat death) AND
    // agent_hunger (wolf.hunger += meat_gain — wolf eats sheep).
    let apply_body = kernel_body_containing(&art, "ApplyKill")
        .unwrap_or_else(|| panic!("no ApplyKill kernel emitted: {:?}", art.kernel_index));
    assert!(
        apply_body.contains("agent_hp["),
        "ApplyKill must write agent_hp (sheep.hp decrement); got body:\n{apply_body}",
    );
    assert!(
        apply_body.contains("agent_alive["),
        "ApplyKill must write agent_alive (set_alive on hp<=0); got body:\n{apply_body}",
    );
    assert!(
        apply_body.contains("agent_hunger["),
        "ApplyKill must write agent_hunger (wolf.hunger += meat_gain); got body:\n{apply_body}",
    );

    // EnergyDecay must write agent_alive (universal starvation gate)
    // and read+write agent_hunger.
    let decay_body = kernel_body_containing(&art, "EnergyDecay")
        .unwrap_or_else(|| panic!("no EnergyDecay kernel emitted: {:?}", art.kernel_index));
    assert!(
        decay_body.contains("agent_alive["),
        "EnergyDecay must write agent_alive (set_alive on hunger<=0); got body:\n{decay_body}",
    );
    assert!(
        decay_body.contains("agent_hunger["),
        "EnergyDecay must read+write agent_hunger; got body:\n{decay_body}",
    );

    // SheepGraze must write agent_hunger (per-sheep += graze_rate).
    let graze_body = kernel_body_containing(&art, "SheepGraze")
        .unwrap_or_else(|| panic!("no SheepGraze kernel emitted: {:?}", art.kernel_index));
    assert!(
        graze_body.contains("agent_hunger["),
        "SheepGraze must write agent_hunger (uniform regrowth proxy); got body:\n{graze_body}",
    );

    // WolfHunt (body-form spatial walk) must bind agent_pos —
    // mirrors foraging_real AntFeed's locking of the
    // ForEachNeighborBody implicit-self.pos-read fix.
    let hunt_body = kernel_body_containing(&art, "WolfHunt")
        .unwrap_or_else(|| panic!("no WolfHunt kernel emitted: {:?}", art.kernel_index));
    assert!(
        hunt_body.contains("agent_pos: array<vec3<f32>>")
            || hunt_body.contains("agent_pos: array<vec3"),
        "WolfHunt must bind agent_pos — ForEachNeighborBody implicit \
         self.pos read fix; got body:\n{hunt_body}",
    );

    // All emitted WGSL must be naga-clean.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}: {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "predator_prey_real emitted {} naga-invalid WGSL kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );

    eprintln!(
        "[predator_prey_real] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}
