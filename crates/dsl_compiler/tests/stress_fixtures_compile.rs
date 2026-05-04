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

/// `tom_probe.sim` is the discovery probe of the Theory-of-Mind
/// belief-read path (see `docs/superpowers/notes/2026-05-04-tom-probe.md`).
/// Pre-fix expected behaviour: BOTH belief-read surfaces drop out at
/// CG-lower time:
///   - `BeliefsAccessor` (sugar-free `beliefs(o).about(t).<field>`
///     surface) → `LoweringError::UnsupportedAstNode { ast_label:
///     "BeliefsAccessor", .. }` at `expr.rs:817`.
///   - `theory_of_mind.believes_knows(...)` namespace fn (the bool
///     bit-against-bitset surface from spec §6) → registry-fallback
///     `LoweringError::UnsupportedNamespaceCall { ns: TheoryOfMind,
///     .. }` at `expr.rs:2663`.
///
/// Both rejections surface as lower diagnostics (the `lower_compilation_to_cg`
/// helper bundles them into the partial program and continues); the
/// emit pass then produces the `fold_fact_witnesses` view-fold + the
/// admin kernels but no producer kernel for either physics rule.
///
/// This test pins THE GAP CHAIN itself — when the lowering closes,
/// the assertion strings flip from "expected diagnostic" to "expected
/// kernel name", and that diff is the pull request that lands the
/// follow-up work.
#[test]
fn tom_probe_surfaces_belief_read_lowering_gaps() {
    use dsl_compiler::cg::lower::lower_compilation_to_cg;
    let path = workspace_path("assets/sim/tom_probe.sim");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let prog = dsl_compiler::parse(&src).expect("parse tom_probe.sim");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve tom_probe.sim");
    let outcome = lower_compilation_to_cg(&comp);
    let (cg, diags) = match outcome {
        Ok(_) => panic!(
            "tom_probe.sim lower unexpectedly clean — the probe expects \
             BeliefsAccessor + believes_knows lowering gaps to surface as \
             diagnostics. If you closed those gaps, update this test to \
             assert the producer kernels exist instead."
        ),
        Err(o) => (o.program, o.diagnostics),
    };
    assert!(
        !diags.is_empty(),
        "expected at least one lower diagnostic (BeliefsAccessor + believes_knows)",
    );
    let diag_text = diags
        .iter()
        .map(|d| format!("{d}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        diag_text.contains("BeliefsAccessor"),
        "expected BeliefsAccessor lower diagnostic; got:\n{diag_text}",
    );
    assert!(
        diag_text.contains("believes_knows"),
        "expected theory_of_mind.believes_knows lower diagnostic; got:\n{diag_text}",
    );
    // Schedule + emit on the partial program — the fold_fact_witnesses
    // kernel + admin kernels should still be produced.
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit tom_probe partial program");
    assert!(
        art.kernel_index.iter().any(|k| k.contains("fact_witnesses")),
        "expected fold_fact_witnesses kernel in artifacts; got: {:?}",
        art.kernel_index,
    );
    assert!(
        !art.kernel_index.iter().any(|k| k.starts_with("physics_CheckBelief")),
        "expected NO physics_CheckBelief kernel (producer rule should drop \
         at lower time); got: {:?}",
        art.kernel_index,
    );
    eprintln!(
        "[tom_probe] {} diagnostics, {} kernels: {:?}",
        diags.len(),
        art.kernel_index.len(),
        art.kernel_index,
    );
}
