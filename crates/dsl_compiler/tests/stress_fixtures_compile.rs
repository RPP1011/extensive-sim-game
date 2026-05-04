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
