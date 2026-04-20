# Project Table of Contents — 2026-04-19

> Single-page reference: what this project is, what's built, what's
> planned, what's not yet planned. For scope questions and sequencing
> decisions. Updated per major milestone.
>
> Cross-refs (don't reproduce): `README.md`, `docs/engine/spec.md`,
> `docs/engine/status.md`, `docs/compiler/spec.md`,
> `docs/superpowers/roadmap.md`, `docs/superpowers/plans/`,
> `docs/audit_2026-04-19.md`,
> `docs/engine/{verification,stub}_audit_2026-04-19.md`.

---

## 1. What this project is

A **deterministic tactical orchestration RPG** built in Rust. The
player manages a hero roster across a contested overworld, resolves
flashpoint crises through multi-room missions, and commands squads in
real-time deterministic combat. A Dwarf-Fortress-style world
simulation (zero-player, ~167 historical systems in the legacy stack)
runs underneath to produce emergent narrative with no scripted events.

The architecture is **three nested layers**: a turn-based **Campaign**
overworld that spawns **Missions** (multi-room dungeons / sieges /
flashpoints) that drop into a 100 ms fixed-tick **Combat** sim where
squads of heroes fight squads of enemies. The Combat layer is the
core of the AI/ML work and the substrate everything else composes on
top of. Combat must be bit-deterministic across replays and across
both compute backends (CPU-scalar reference + GPU-Vulkan performance
path).

The active engineering effort is a **ground-up rewrite of the Combat
layer** as a new crate (`crates/engine/`) plus a **DSL + compiler**
that emits engine code from a high-level rules language. The legacy
tactical_sim crate (and its ML pipeline — combat student, ability
transformer, actor-critic V3, etc.) remains in-tree for reference and
training-data harvest, but new code targets the new engine. The
unified runtime ships **two first-class backends** — `SerialBackend`
(host-scalar Rust, the reference) and `GpuBackend` (SPIR-V kernels via
`voxel_engine::compute::GpuHarness`, performance path) — that share
the same public API and must produce byte-identical
`replayable_sha256()` on the same seed.

---

## 2. Current snapshot — 2026-04-19

| Metric | Value |
|---|--:|
| Engine commits today (2026-04-19) | **149** |
| Engine tests (release) | **333** |
| Engine tests (debug) | **334** |
| Viz tests | 10 |
| `cargo-fuzz` targets (compile-verified) | 2 |
| Engine test files (`crates/engine/tests/*.rs`) | 88 |
| Engine source dirs | `ability/`, `aggregate/`, `cascade/`, `event/`, `invariant/`, `policy/`, `state/`, `telemetry/`, `view/` + `step.rs`, `mask.rs`, `spatial.rs`, `pool.rs`, `rng.rs`, `schema_hash.rs`, `trajectory.rs`, `channel.rs`, `creature.rs`, `ids.rs`, `lib.rs` |
| Schema hash baseline | `7bf05a9ff5aafa7d5cf6f2ba30a00ac0353a906a1660b1b4ff993ecf69af5006` |
| What landed today | Engine MVP → Plan 1 → Plan 2 → Plan 2.75 → Plan 3.0 viz harness → Plan 3.1 viz fixups → State-port (13 SoA tasks) → **Combat Foundation** (engagement + abilities + 8 effect ops + recursion) → verification audit (HIGH+MED resolved) → stub audit (3 CRITICAL + 5 HIGH + 3 MED + 6 LOW resolved). |
| Outstanding draft | Plan 3 (persistence + obs packer + probes), 1027 LoC, 12 tasks, never executed. |
| Outstanding subsystem backlog | 14 deferred subsystems indexed in `docs/superpowers/roadmap.md`. |
| Plans not yet written | Plan 4 (debug+trace), Plan 5 (ComputeBackend trait), Plan 6 (GpuBackend foundation), Plan 7+ (per-kernel GPU ports), 14 deferred-subsystem plans, all DSL compiler plans, all ML reconnect plans, all game-layer plans, all production/CI plans. |

---

## 3. Plan inventory

Status: ✅ complete / 🟡 in-flight / ⚠️ drafted (not executed) / ❌ needed-but-not-written.

### 3.1 Engine plans (drafted or executed)

| # | Plan file | Status | Tasks | Commits landed | Notes |
|---|---|---|---|---|---|
| 1 | `2026-04-19-world-sim-engine-mvp.md` | ✅ | multi-phase | `24c7a7eb`..`95b3c36a` | Pools, EventRing, spatial index, PCG RNG, SoA agent, mask buffer, UtilityBackend, MicroKind enum, MaterializedView+DamageTaken, trajectory writer, schema-hash baseline. |
| 2 | `2026-04-19-engine-plan-1-action-space.md` | ✅ | 19 (T1–T18) | `64675559`..`155a51df` + Announce/cascade extensions | 18 MicroKinds, 5 MacroKinds, Announce primary+overhear cascade, AggregatePool, EventId cause sidecar, CascadeRegistry + run_fixed_point. |
| 3 | `2026-04-19-engine-plan-2-pipeline-traits.md` | ✅ | 9 (T1–T5) | `fe6e2a1d`..`53f2b6d5` | 6-phase tick pipeline (`step_full`), `LazyView`+`TopKView` traits, `InvariantRegistry`, `TelemetrySink` (Null/Vec/File). |
| 4 | `2026-04-19-engine-plan-2_75-verification-infra.md` | ✅ | 7 | `ccc31271`..`86576f19` | proptest suite (event-hash, pool, spatial, mask validity, cascade bound, baseline), `#[contracts]` on Pool + step_full, 2 cargo-fuzz targets. |
| 5 | `2026-04-19-engine-plan-3_0-viz-harness.md` | ✅ | 8 (T1–T5 + 3.1 fixups) | `c50ee255` (rewrite of `db758a63`) → `0fa26452`..`d1b2c1df` | `crates/viz/`: winit + `voxel_engine` window, scenario loader, agent voxel render, event overlays, controls, agent stacking workaround for missing collision. |
| 6 | `2026-04-19-engine-plan-state-port.md` | ✅ | 13 (Tasks A–L + L-rebase) | `56e586d5`..`c83de370` + reconcile `da8aa9ce` | Ported full `docs/dsl/state.md` agent catalogue: 28 hot SoA + 9 cold SoA + 9 supporting types. Storage-only, no behaviour. Schema-hash rebased once. |
| 7 | `2026-04-19-combat-foundation.md` | ✅ | 24 | `babb8ec0`..`a379f3df` | Engagement + ZoC + ability runtime: `hot_engaged_with`, `cold_standing` (SparseStandings), 5 hot timing fields, `ability/` module (id, program, registry, cast, expire, gate + 8 EffectOp handler files), `OpportunityAttackHandler`, 13 new event variants, mask cast-gate + cooldown + engagement. Stub audit (3 CRITICAL + 5 HIGH + 3 MED + 6 LOW) hunted + fully resolved post-merge. |
| 8 | `2026-04-19-engine-plan-3-persistence-obs-probes.md` | ⚠️ | 12 | none | Snapshot format + migration + `ObsPacker` + `Probe` harness. Drafted at `7bcfdbe2`, never executed. Plan 3 must execute **after** all SoA-adding plans (current SoA = 28 hot + 9 cold + ability state) so the snapshot writer covers the full layout in one pass. |

### 3.2 Engine plans needed-but-not-written

| # | Plan file | Status | Tasks (est.) | Notes |
|---|---|---|---|---|
| 9 | — (TBD) Plan 4 — Debug & trace runtime | ❌ | 6–8 | Spec §24: `trace_mask`, `causal_tree`, `tick_stepper`, `tick_profile`, `agent_history`, `snapshot`. Inspector tooling for the engine; depends on Plan 3 snapshot infra. |
| 10 | — (TBD) Plan 5 — `ComputeBackend` trait extraction | ❌ | 4–6 | Spec §3+§25: factor backend-agnostic API out of `step_full`. Today, `Serial` is implicit (no trait). This plan introduces the trait but adds zero kernels. Prerequisite for any GPU work. |
| 11 | — (TBD) Plan 6 — `GpuBackend` foundation | ❌ | 8–12 | Spec §8+§25: ship `voxel_engine`-backed backend. Embed engine-universal SPIR-V kernels (mask predicates, apply kernels for Move/Flee/Attack/Needs, view reductions, spatial hash). Cross-backend parity test in CI (mandatory, spec §2). |
| 12 | — (TBD) Plan 7+ — per-kernel GPU ports (one per subsystem) | ❌ | 4–6 plans × 6–10 kernels | Each subsystem port: SPIR-V kernel + engine wiring + parity test against `SerialBackend`. Estimated kernels: spatial hash, mask predicates (×6+), apply kernels (×7+), policy utility argmax, materialized-view reductions (×3+). |
| 13 | — (TBD) Cascade-handler GPU emitter (post-Plan 6) | ❌ | 2–3 plans | Each compiler-emitted cascade handler in the deferred-subsystem layer needs a SPIR-V counterpart for `GpuBackend`. Cross-cutting plan family, runs alongside per-subsystem ports. |
| 14 | — (TBD) `FailureMode::Rollback` rewind machinery | ❌ | 1 | Variant was removed in `e18b6635` (stub audit HIGH #8). If real rewind is ever wanted, it lands as its own plan with state-snapshot/restore at tick boundaries. |

### 3.3 Deferred-subsystem plans (engine domain layer)

The 14 subsystems indexed in `docs/superpowers/roadmap.md`. Per-row scope is one paragraph in that doc; not reproduced here. **All 14 are ❌ needed-but-not-written.**

| # | Plan file | Status | Tasks (est.) | Roadmap §, blocks |
|---|---|---|---|---|
| 15 | — (TBD) Items runtime | ❌ | ~14 | §3.4. Bifurcated **commodities** (`Inventory.commodities: [u16; 8]`) vs **items** (`AggregatePool<ItemInstance>`, max-stack=1). Blocks Buildings, Quests rewards. |
| 16 | — (TBD) Groups runtime | ❌ | ~12 | §3.7. `AggregatePool<Group>` instance data. Highest leverage — blocks Memberships, Factions, Settlements, Quests. |
| 17 | — (TBD) Memberships runtime | ❌ | ~10 | §3.1. Per-agent group pointers + standing_q8. Blocks Factions, Settlements, Quests. |
| 18 | — (TBD) Memory ring runtime | ❌ | ~16 | §3.2. Folds `Event::RecordMemory` + `Source` enum + decay. Blocks Relationships, Theory-of-Mind. |
| 19 | — (TBD) Relationships runtime | ❌ | ~10 | §3.3. Per-pair `valence_q8` + alpha-learning from memory. Blocks Theory-of-Mind. Resolves `cold_standing` naming via Model B (all-independent layers). |
| 20 | — (TBD) Personality-influenced utility | ❌ | ~6 | §3.5. Smallest plan; reads `hot_risk_tolerance` etc. into UtilityBackend score formula. No new state. |
| 21 | — (TBD) Theory-of-mind / `believed_knowledge` | ❌ | ~10 | §3.6. 32-bit per-pair domain bitset; Deceive predicate. |
| 22 | — (TBD) Factions | ❌ | ~14 | §3.8. `Group { kind=Faction }` with diplomacy cascade, war/peace/alliance. |
| 23 | — (TBD) Buildings + Rooms | ❌ | ~18 | §3.9. `AggregatePool<BuildingData>` + nested `Room` + tile-imprint cascade. Blocks Settlements, Interior nav. |
| 24 | — (TBD) Settlements | ❌ | ~14 | §3.10. `Group { kind=Settlement }` with housing alloc + treasury + service contracts. |
| 25 | — (TBD) Regions | ❌ | ~10 | §3.11. `WorldState.regions: Vec<RegionState>` + control + threat + unrest. |
| 26 | — (TBD) Quests runtime | ❌ | ~12 | §3.12. `AggregatePool<Quest>` instance data + `QuestPosting` lifecycle. |
| 27 | — (TBD) Terrain + voxel collision | ❌ | ~22 | §3.13. Largest subsystem — `voxel_engine` integration beyond viz, NavGrid, walk_collision_check phase. Blocks Buildings, Interior nav. |
| 28 | — (TBD) Interior navigation | ❌ | ~10 | §3.14. `cold_grid_id` / `cold_local_pos` activation, A* over InteriorNavGrid, door traversal events. |

### 3.4 DSL compiler plans

| # | Plan file | Status | Tasks (est.) | Notes |
|---|---|---|---|---|
| 29 | — (TBD) Compiler scaffold + winnow parser | ❌ | ~8 | `dsl/spec.md` §1–2. Crate skeleton, lexer, AST. |
| 30 | — (TBD) Lower verb decls | ❌ | ~4 | Compiler spec §3 verb desugaring. |
| 31 | — (TBD) Lower `Read → Ask` | ❌ | ~2 | Compiler spec §3 sugar lowering. |
| 32 | — (TBD) Type checker / structural type relationships | ❌ | ~10 | DSL spec §5. |
| 33 | — (TBD) `entity` decl → SoA emitter | ❌ | ~8 | DSL spec §2.1. Hot/cold split per `@hot`/`@cold` annotations (compiler spec §1.3). |
| 34 | — (TBD) `event` decl → byte-stable Event variant emitter | ❌ | ~6 | DSL §2.2. Bridges to engine `EventRing` byte-format. |
| 35 | — (TBD) `view` decl → MaterializedView/LazyView/TopKView codegen | ❌ | ~10 | DSL §2.3 + storage-hint selection (`pair_map`, `per_entity_topk`, `lazy_cached`). |
| 36 | — (TBD) `physics` rule → cascade handler codegen (Serial, scalar Rust closures) | ❌ | ~10 | DSL §2.4. Lane assignment + cycle detection. |
| 37 | — (TBD) `physics` rule → SPIR-V cascade kernel emitter | ❌ | ~12 | Engine spec §11 GPU side. Pairs with #36. |
| 38 | — (TBD) `mask` predicate → mask kernel emitter (Serial + GPU) | ❌ | ~8 | DSL §2.5. |
| 39 | — (TBD) `policy` decl → PolicyBackend codegen | ❌ | ~8 | DSL §2.7. Includes `UtilityBackend` recipes + `NeuralBackend` GPU matmul codegen. |
| 40 | — (TBD) `invariant` decl → Invariant impl codegen | ❌ | ~4 | DSL §2.8. |
| 41 | — (TBD) `probe` decl → Probe harness codegen | ❌ | ~4 | DSL §2.9. Pairs with engine Plan 3 probes. |
| 42 | — (TBD) `curriculum` decl → Python emitter | ❌ | ~6 | DSL §2.10. ML-side artefact. |
| 43 | — (TBD) `telemetry` decl → metric registration | ❌ | ~3 | DSL §2.11. |
| 44 | — (TBD) Schema-hash emitter (4 sub-hashes + combined) | ❌ | ~4 | Compiler spec §2. Saves checkpoint diff + git remediation hint. |
| 45 | — (TBD) Schema-drift CI guard | ❌ | ~3 | Compiler spec §2 last paragraph. Append-only changes pass; reorder/remove blocks. |
| 46 | — (TBD) Reward-block / advantage / value codegen | ❌ | ~6 | DSL §3.4. |
| 47 | — (TBD) Cross-entity walk codegen for GPU `AggregatePool<T>` | ❌ | ~8 | Compiler spec §1.2 last paragraph. Fixed-size inline-array iteration in SPIR-V. |
| 48 | — (TBD) `cargo xtask compile-shaders` integration | ❌ | ~3 | Compiler spec non-goal §10 `shaderc` integration is the one allowed exception. |
| 49 | — (TBD) Asset pipeline: `.ability` (legacy) → DSL bridge | ❌ | ~6 | 199 hero/LoL ability files in `assets/`; bridge so we don't lose content during migration. |

### 3.5 ML reconnection plans

| # | Plan file | Status | Tasks (est.) | Notes |
|---|---|---|---|---|
| 50 | — (TBD) Trajectory schema → training adapter | ❌ | ~4 | Engine emits safetensors today (`src/trajectory.rs`); training scripts in `training/` consume the legacy `tactical_sim` shape. Bridge layer. |
| 51 | — (TBD) `NeuralBackend` for `PolicyBackend` (Serial stub + GPU forward) | ❌ | ~10 | Engine spec §13. GPU matmul kernels for fused GEMM + activation. |
| 52 | — (TBD) Distillation: existing combat-student / ability-transformer / actor-critic-v3 → new engine | ❌ | ~8 | Decide: port the existing checkpoints (`generated/*.json`) or retrain fresh. |
| 53 | — (TBD) Behavioral-embedding (operator) integration | ❌ | ~4 | `crates/ability_operator` + `crates/ability-vae` produce 128-d behavioral embeddings; wire as `external_cls_proj` for the new neural policy. |
| 54 | — (TBD) Curriculum runner over new engine | ❌ | ~6 | Pairs with DSL `curriculum` decl (#42). Port `bench_throughput.py` + `bench_compare.py` to drive new-engine sim_bridge. |

### 3.6 Game-layer plans

| # | Plan file | Status | Tasks (est.) | Notes |
|---|---|---|---|---|
| 55 | — (TBD) Campaign overworld layer | ❌ | ~14 | Turn-based; spawns Missions. Per CLAUDE.md three-layer arch. Owns roster, calendar, world map. Likely a separate engine-instance-per-layer. |
| 56 | — (TBD) Mission layer (multi-room dungeon / siege / flashpoint) | ❌ | ~12 | Glue between Campaign and Combat. Manages room transitions, party state across encounters, loot/XP carry-over. |
| 57 | — (TBD) Hero roster + class/role definitions on new engine | ❌ | ~10 | 27 base heroes + 172 LoL imports in legacy `assets/`; needs porting to new ability DSL. |
| 58 | — (TBD) Player UI / hero command panel | ❌ | ~12 | Real-time squad orders, ability targeting, pause-with-orders. Bevy or egui on top of `voxel_engine` window. |
| 59 | — (TBD) Chronicle prose generator | ❌ | ~8 | Engine spec §26 non-goal — host-side template expansion from event log. Likely a separate `crates/chronicle/` crate. |
| 60 | — (TBD) Worldgen pipeline (regions + settlements + history) | ❌ | ~14 | DF-style zero-player history pre-game-start. Composes deferred subsystems #15–#28 over fast-forwarded ticks. |
| 61 | — (TBD) Save/load at game level (Campaign+Mission+Combat snapshot bundle) | ❌ | ~6 | Builds on engine Plan 3 (per-instance snapshots) plus campaign/mission state files. |
| 62 | — (TBD) Difficulty / scenario authoring tools | ❌ | ~6 | Replaces legacy `scenarios/*.toml` runner with new-engine equivalent. |

### 3.7 Production / CI / distribution plans

| # | Plan file | Status | Tasks (est.) | Notes |
|---|---|---|---|---|
| 63 | — (TBD) GitHub Actions workflow rebuild for new engine | ❌ | ~4 | Determinism tests in debug+release; clippy `-D warnings`; schema-hash drift guard. |
| 64 | — (TBD) Cross-backend parity CI (Vulkan via Mesa lavapipe) | ❌ | ~6 | Engine spec §25 mandates parity test in CI. Software Vulkan in container for non-GPU runners. |
| 65 | — (TBD) GPU-perf regression CI (real GPU) | ❌ | ~4 | Per-kernel timing baseline; alert on N% regression. |
| 66 | — (TBD) Fuzz-farm scaling (cargo-fuzz nightly + corpus expansion) | ❌ | ~3 | Today: 2 fuzz targets compile-verified, nightly run wired. Expand corpus, persist findings. |
| 67 | — (TBD) Distribution: build pipeline for player-facing binary | ❌ | ~6 | Once game layers exist. Per-platform packaging, asset embedding, voxel_engine vendoring. |

---

## 4. Deferred-subsystem plans (engine-layer)

Indexed at `docs/superpowers/roadmap.md`. Per-subsystem detail (state touched, events needed, cascade handlers, mask predicates, dependencies, blockers, design questions) is in §3 of that doc — **not reproduced here**. Summary table only:

| # | Subsystem | Roadmap rank | Tasks (est.) | Direct downstream blocks |
|---|---|---:|---:|---|
| 1 | Memberships | 3 | 10 | Factions, Settlements |
| 2 | Memory ring | 4 | 16 | Relationships, Theory-of-Mind |
| 3 | Relationships | 5 | 10 | Theory-of-Mind |
| 4 | Items | 1 | 14 | Buildings, Settlements (via Buildings), Quests |
| 5 | Personality utility | 14 | 6 | (none) |
| 6 | Theory-of-mind | 11 | 10 | (none) |
| 7 | Groups | 2 | 12 | Memberships, Factions, Settlements, Quests |
| 8 | Factions | 6 | 14 | (Quests partially) |
| 9 | Buildings + Rooms | 8 | 18 | Settlements, Interior nav |
| 10 | Settlements | 9 | 14 | Regions |
| 11 | Regions | 12 | 10 | (none) |
| 12 | Quests | 10 | 12 | (none) |
| 13 | Terrain + voxel collision | 7 | 22 | Buildings, Interior nav |
| 14 | Interior nav | 13 | 10 | (none) |

**Open design questions resolved 2026-04-19** (per roadmap §6):
1. Group-membership conflict — faction-relation-score-mediated.
2. Item-stack semantics — two-pool split (commodities + items).
3. Memory-vs-standing — Model B, all-independent layers.

---

## 5. Horizon layers

| Horizon | What's in it | Plans est. | Status |
|---|---|--:|---|
| Near-term — finish engine Combat layer | Plan 3 (persistence + obs + probes) + first 3–5 deferred subsystems (Items → Groups → Memberships → Memory → Relationships) | 5–6 plans | 1 drafted (Plan 3); 4–5 not written |
| Near-term — engine discipline | Plan 4 debug/trace runtime, Plan 5 `ComputeBackend` trait, Plan 6 `GpuBackend` foundation | 3 plans | All not written |
| Medium-term — engine domain features | Remaining 9 deferred subsystems (Theory-of-Mind, Factions, Buildings+Rooms, Settlements, Regions, Quests, Terrain, Interior nav, Personality utility) | 9 plans | All not written |
| Medium-term — GPU porting | Plans 7+ — per-kernel ports with cross-backend parity tests; cascade-handler SPIR-V emitter; per-subsystem GPU completion plans | 4–6 plans | All not written |
| Long-term — DSL compiler | Compiler crate scaffold, lexer/parser, type checker, entity/event/view/physics/mask/policy/invariant/probe/curriculum/telemetry codegen, schema-hash emitter, drift CI guard, asset bridge | 20+ plans | All not written; almost not started |
| Long-term — ML reconnect | Trajectory adapter, `NeuralBackend`, distillation of existing checkpoints, behavioural-embedding integration, curriculum runner | 3–5 plans | All not written |
| Long-term — game layers | Campaign overworld, Mission layer, hero roster port, player UI, Chronicle prose, worldgen pipeline, game-level save/load, scenario authoring | 8–12 plans | All not written |
| Long-term — production / distribution | CI rebuild, cross-backend parity CI, GPU-perf regression CI, fuzz-farm scaling, distribution pipeline | 3–5 plans | All not written |

---

## 6. Subsystem inventory

Aggregate of every subsystem the project touches. Spec § references are to `docs/engine/spec.md` unless noted. Status: ✅ done / 🟡 partial / ⚠️ stub / ❌ not started. Code paths in `crates/` are relative to repo root.

### 6.1 Engine — core infrastructure

| Subsystem | Spec § | Status | Code | Plan |
|---|---|---|---|---|
| `Pool<T>` (generic, `NonZeroU32`, freelist) | §3 | ✅ | `crates/engine/src/pool.rs` | MVP + P1 T1, P2.75 |
| SoA agent state (28 hot + 9 cold fields, full state.md catalogue) | §3 | ✅ | `crates/engine/src/state/` | MVP + P1 T2 + state-port A–L |
| Typed IDs (`AgentId, GroupId, QuestId, ItemId, AuctionId, InviteId, SettlementId, AbilityId, EventId`) | §4 | ✅ | `crates/engine/src/ids.rs`, `ability/id.rs` | MVP + P1 + Combat Foundation |
| `EventRing` + byte-stable hash | §5 | ✅ | `crates/engine/src/event/ring.rs`, `event/mod.rs` | MVP + P1 + P2.75 + fuzz |
| Spatial index (2D-column BTreeMap + z-sort + movement-mode sidecar) + integration | §6 | ✅ | `crates/engine/src/spatial.rs` (wired in `state/mod.rs`, `mask.rs`, `step.rs`, `ability/expire.rs` after stub-audit CRITICAL #1) | MVP + P2.75 + audit fix `20fc5a26` |
| RNG (PCG-XSH-RR + per-agent stream + golden tests) | §7 | ✅ | `crates/engine/src/rng.rs` | MVP |
| `MicroKind` enum (18 variants) | §9 | ✅ | `crates/engine/src/mask.rs` | P1 T4 |
| MicroKind execution: Hold / MoveToward / Flee / Attack / Eat / Drink / Rest | §9 | ✅ | `crates/engine/src/step.rs` | P1 T6–T9 + audit fixes (boundary pinning) |
| 11 event-only micros (Cast, UseItem, Harvest, PlaceTile/Voxel, HarvestVoxel, Converse, ShareStory, Communicate, Ask, Remember) | §9 | ✅ | `crates/engine/src/step.rs` | P1 T10 |
| `MacroKind` (PostQuest, AcceptQuest, Bid, Announce, NoOp) | §10 | ✅ | `crates/engine/src/policy/macro_kind.rs`, `policy/query.rs` | P1 T5, T12 |
| Announce cascade (audience + primary + overhear, channel-gated) | §10 | ✅ | `crates/engine/src/step.rs::429-507` | P1 T14, T15 + audit fixes (channel_range wired `53fe6214`) |
| `CascadeRegistry` + `CascadeHandler` trait + Lane (Validation→Effect→Reaction→Audit) | §11 | ✅ | `crates/engine/src/cascade/` | P1 T11 |
| Cascade `run_fixed_point` (`MAX_CASCADE_ITERATIONS=8`) + telemetry | §11 | ✅ | `crates/engine/src/cascade/dispatch.rs` | P1 T13 + P2.75 + audit fix `c6996271` (CASCADE_ITERATIONS metric wired) |
| `MaskBuffer` (18-bit-per-agent head) + predicates (hold, move, flee, attack, needs, domain hooks, **cast gate** integration) | §12 | ✅ | `crates/engine/src/mask.rs` | MVP + P2.75 + Combat Foundation + audit fix `711c0023` (cast-gate wired) |
| `PolicyBackend` trait + `UtilityBackend` scalar | §13 | ✅ | `crates/engine/src/policy/` | MVP |
| 6-phase tick pipeline (`step_full`) + contracts | §14 | ✅ | `crates/engine/src/step.rs::step_full` | P2 T5 + P2.75 |
| `MaterializedView` + `DamageTaken` example | §15 | ✅ | `crates/engine/src/view/materialized.rs` | MVP |
| `LazyView` + `NearestEnemyLazy` (trait shape only — **NOT wired into `step_full`**) | §15 | ⚠️ | `crates/engine/src/view/lazy.rs` | P2 T1; integration deferred (canary test `#[ignore]`d) |
| `TopKView` + `MostHostileTopK` (cumulative-damage heap, K=8) | §15 | ✅ | `crates/engine/src/view/topk.rs` | P2 T2 |
| `AggregatePool<T>` + Pod-shaped `Quest` / `Group` | §16 | ✅ | `crates/engine/src/aggregate/` | P1 T16 |
| Trajectory emission (safetensors) | §17 | ✅ | `crates/engine/src/trajectory.rs` | MVP |
| `Invariant` trait + `Violation` + `FailureMode` (Panic/Log; Rollback **removed** in `e18b6635`) | §19 | ✅ | `crates/engine/src/invariant/trait_.rs`, `invariant/registry.rs` | P2 T3 + audit fix HIGH #8 |
| Built-in invariants: `MaskValidityInvariant`, `PoolNonOverlapInvariant` | §19 | ✅ | `crates/engine/src/invariant/builtins.rs` | P2 T3 + audit fix `bc6fac31` (real PoolNonOverlap impl) |
| Schema hash + `.schema_hash` baseline | §22 | ✅ | `crates/engine/src/schema_hash.rs`, `crates/engine/.schema_hash` | MVP + P1 T4 + P2 T4 + post-state-port + post-Rollback removal |
| `TelemetrySink` trait + `Null/Vec/File` sinks + metric name consts | §22 | ✅ | `crates/engine/src/telemetry/` | P2 T4 |

### 6.2 Engine — Combat Foundation (today)

| Subsystem | Spec § | Status | Code | Plan |
|---|---|---|---|---|
| `hot_engaged_with` SoA + bidirectional engagement update | (combat foundation) | ✅ | `crates/engine/src/state/mod.rs`, `step.rs` (tick-start phase) | Combat Foundation T1–T3 |
| `cold_standing` (`SparseStandings`, agent-pair, clamped `[-1000, 1000]`) | (combat foundation) | ✅ | `crates/engine/src/state/mod.rs` | Combat Foundation T4 + state-port |
| Hot timing fields: `stun_remaining_ticks`, `slow_remaining_ticks`, `slow_factor_q8`, `cooldown_next_ready_tick`, `shield_hp` | (combat foundation) | ✅ | `crates/engine/src/state/mod.rs` | Combat Foundation T4 + state-port Tasks B–C |
| `ability/` module: `AbilityId`, `AbilityProgram`, `EffectOp`, `Delivery`, `Area`, `Gate`, `TargetSelector`, `AbilityRegistry` (append-only, slot-stable, builder) | (combat foundation) | ✅ | `crates/engine/src/ability/{id,program,registry}.rs` | Combat Foundation T5–T8 |
| `CastHandler` (single dispatch on `EffectOp`) + `evaluate_cast_gate` (engagement-aware, cooldown-aware, mask-consulted) | (combat foundation) | ✅ | `crates/engine/src/ability/{cast,gate}.rs` | Combat Foundation T9–T10 + audit fix CRITICAL #2 |
| 8 EffectOp handlers: `Damage` (shield-first absorb + death cascade), `Heal` (max_hp cap), `Shield` (additive layer), `Stun`, `Slow` (factor_q8 + duration), `TransferGold` (signed i64, debt allowed), `ModifyStanding` (symmetric clamp), `CastAbility` (recursion bounded by `MAX_CASCADE_ITERATIONS`) | (combat foundation) | ✅ | `crates/engine/src/ability/{damage,heal,shield,stun,slow,gold,standing,cast}.rs` | Combat Foundation T11–T18 |
| `RecordMemoryHandler` (writes `MemoryEvent` to `cold_memory[observer]` on Announce primary + overhear) | (combat foundation, audit) | ✅ | `crates/engine/src/ability/record_memory.rs` | Audit fix HIGH #4 (`57cb4bc7`) |
| Tick-start unified phase (engagement update + stun/slow/cooldown decrement + expiry events) | (combat foundation) | ✅ | `crates/engine/src/ability/expire.rs` | Combat Foundation T19 |
| `OpportunityAttackHandler` (engagement-aware MoveToward/Flee + slow factor) | (combat foundation) | ✅ | `crates/engine/src/ability/expire.rs` | Combat Foundation T20–T22 |
| 13+ new event variants (AgentCast, EffectDamageApplied, EffectHealApplied, ShieldGranted/Expired, StunApplied/Expired, SlowApplied/Expired, GoldTransferred, StandingChanged, CastDepthExceeded, GoldInsufficient, OpportunityAttackTriggered, ExpiryEvents) | §5 | ✅ | `crates/engine/src/event/mod.rs`, `event/ring.rs` | Combat Foundation T23 |
| Per-agent combat stats (`hot_attack_damage`, `hot_attack_range`) **read by Attack kernel** + `OpportunityAttackHandler` | §9 | ✅ | `crates/engine/src/step.rs`, `ability/expire.rs` | Audit fix MEDIUM #10 (`642848d7`) |

### 6.3 Engine — auxiliary / debug

| Subsystem | Spec § | Status | Code | Plan |
|---|---|---|---|---|
| Save/load (snapshot format + migration registry) | §18 | ❌ | — | Plan 3 (drafted) |
| Probe harness | §20 | ❌ | — | Plan 3 (drafted) |
| Observation packer (`ObsPacker` + `FeatureSource` + Vitals/Position/Neighbor sources) | §23 | ❌ | — | Plan 3 (drafted) |
| Debug & trace runtime (`trace_mask`, `causal_tree`, `tick_stepper`, `tick_profile`, `agent_history`, `snapshot`) | §24 | ❌ | — | Plan 4 (TBD) |

### 6.4 Engine — GPU + cross-backend (all not started)

| Subsystem | Spec § | Status | Plan |
|---|---|---|---|
| `ComputeBackend` trait extraction | §3, §25 | ❌ | Plan 5 (TBD) |
| `SerialBackend` formal split (today is implicit) | §3, §4 | ⚠️ (de facto exists; not behind a trait) | Plan 5 (TBD) |
| `GpuBackend` foundation (init, kernel catalog, SPIR-V loader) | §8 | ❌ | Plan 6 (TBD) |
| Engine-universal SPIR-V kernels: mask predicates (×6), apply (×7+), policy utility argmax, view reductions (×3+), spatial hash (×2) | §8 | ❌ | Plan 6 + 7+ (TBD) |
| Cross-backend parity test (mandatory CI) | §2 | ❌ | Plan 6 (TBD) |
| RNG cross-backend golden test (host vs shader) | §7 | ❌ | Plan 6 (TBD) |
| GPU event-ring drain + sorted hashing | §5 | ❌ | Plan 6 (TBD) |
| GPU-eligible aggregates (`AggregatePool<T: Pod>`) | §16 | ⚠️ (Pod shapes exist; no GPU storage path) | Plan 6 (TBD) |
| Cascade handler GPU emitter (per-`physics`-rule SPIR-V) | §11 | ❌ | Plan 7+ + DSL compiler (TBD) |

### 6.5 Engine — domain (deferred subsystems #15–#28 from §3.3)

See §4 above. Every row ❌. State catalogue partially staged via state-port (cold collections allocated; no behaviour wired).

### 6.6 Compiler

| Subsystem | Compiler spec § | Status | Plan |
|---|---|---|---|
| Crate scaffold (`crates/compiler/`?) | §1 | ❌ | TBD |
| Lexer + winnow parser | §1 | ❌ | TBD |
| AST + type checker | §1 | ❌ | TBD |
| `entity` decl → SoA + `@hot`/`@cold` partition | §1.3 | ❌ | TBD |
| `event` decl → byte-stable variant emitter | §2 | ❌ | TBD |
| `view` decl → MaterializedView/LazyView/TopKView codegen + storage-hint selection | §3 | ❌ | TBD |
| `physics` rule → cascade handler codegen (Serial closures + SPIR-V kernels) | §3 | ❌ | TBD |
| `mask` predicate codegen (Serial + GPU) | §3 | ❌ | TBD |
| `verb` desugaring | §3 | ❌ | TBD |
| `Read → Ask` lowering | §3 | ❌ | TBD |
| `policy` decl → PolicyBackend codegen (Utility + Neural) | §1.2 | ❌ | TBD |
| `invariant` decl → `Invariant` impl codegen | §1.1 | ❌ | TBD |
| `probe` decl → Probe harness codegen | §1.1 | ❌ | TBD |
| `curriculum` decl → Python emitter | §1.1 | ❌ | TBD |
| `telemetry` decl → metric registration | §1.1 | ❌ | TBD |
| Schema hash emitter (4 sub-hashes + combined) + drift CI guard | §2 | ❌ | TBD |
| Reward/value/advantage codegen | §1.1 | ❌ | TBD |
| Cross-entity walk codegen for GPU `AggregatePool<T>` | §1.2 | ❌ | TBD |
| `cargo xtask compile-shaders` integration (`shaderc`) | §10 non-goal exception | ❌ | TBD |

### 6.7 ML

| Subsystem | Status | Plan | Notes |
|---|---|---|---|
| Trajectory writer + reader (safetensors) | ✅ | MVP (engine) | `src/trajectory.rs`. Schema = engine-emitted; training side TBD. |
| Trajectory schema → training adapter | ❌ | TBD #50 | Bridge to legacy training shape. |
| `NeuralBackend` for `PolicyBackend` (Serial stub + GPU forward) | ❌ | TBD #51 | Engine spec §13 stubs it. |
| Distillation: existing combat-student / ability-transformer / actor-critic-v3 | ❌ | TBD #52 | Decide port-vs-retrain. |
| Behavioural-embedding (operator) integration | ❌ | TBD #53 | `crates/ability_operator` + `crates/ability-vae` produce 128-d embeddings. |
| Curriculum runner over new engine | ❌ | TBD #54 | Pairs with DSL `curriculum`. |
| Legacy ML pipeline (combat-student, ability-eval v2/v3, ability-transformer, actor-critic, pointer-V3, next-state-prediction d128) | 🟡 (works on legacy `tactical_sim`; not yet on new engine) | — | Stays in tree for reference + training-data harvest. |

### 6.8 Game layers

| Subsystem | Status | Plan |
|---|---|---|
| Campaign overworld | ❌ | TBD #55 |
| Mission layer | ❌ | TBD #56 |
| Hero roster + class/role on new engine | ❌ | TBD #57 |
| Player UI / hero command panel | ❌ | TBD #58 |
| Chronicle prose generator | ❌ | TBD #59 |
| Worldgen pipeline (zero-player history) | ❌ | TBD #60 |
| Game-level save/load (Campaign + Mission + Combat snapshot bundle) | ❌ | TBD #61 |
| Scenario authoring tools (replacement for `scenarios/*.toml` runner) | ❌ | TBD #62 |

### 6.9 Production

| Subsystem | Status | Plan |
|---|---|---|
| GitHub Actions workflow rebuild for new engine | ❌ | TBD #63 |
| Cross-backend parity CI (Mesa lavapipe) | ❌ | TBD #64 |
| GPU-perf regression CI (real GPU) | ❌ | TBD #65 |
| Fuzz-farm scaling (cargo-fuzz nightly + corpus) | 🟡 | 2 targets compile-verified; nightly wired (`86576f19`); needs corpus growth |
| Distribution / packaging | ❌ | TBD #67 |

---

## 7. Dependency graph (top-level)

```
                    Engine MVP ✅
                          │
        ┌─────────────────┼──────────────────────────────┐
        ▼                 ▼                              ▼
   Plan 1 (action) ✅   Plan 2 (pipeline) ✅       Plan 2.75 (verif) ✅
        │                 │
        └─────────┬───────┘
                  ▼
        Plan 3.0 viz harness ✅ + Plan 3.1 fixups ✅
                  │
                  ▼
          State-port (28 hot + 9 cold) ✅
                  │
                  ▼
      Combat Foundation (engagement + abilities) ✅
                  │
       ┌──────────┼─────────────────────────────────────┐
       ▼          ▼                                     ▼
  Plan 3      14 deferred subsystems          Plan 4 debug+trace ❌
  persistence ❌                              (depends on Plan 3 snapshots)
  obs+probes ❌    ┌─────┐
  (drafted)        │ #4  │ Items (independent, ranked #1)
                   │ #7  │ Groups (blocks #1, #8, #10, #12)
                   │ #1  │ Memberships (after #7)
                   │ #2  │ Memory (independent)
                   │ #3  │ Relationships (after #2)
                   │ #5  │ Personality utility (orthogonal, smallest)
                   │ #6  │ Theory-of-Mind (after #2 + #3)
                   │ #8  │ Factions (after #7 + #1 + #3)
                   │ #13 │ Terrain + voxel (independent, biggest)
                   │ #9  │ Buildings + Rooms (after #13 + #4)
                   │ #10 │ Settlements (after #9 + #1 + #4)
                   │ #12 │ Quests (after #7 + #8 + #10)
                   │ #11 │ Regions (after #10 + #8 + #13)
                   │ #14 │ Interior nav (after #9 + #13)
                   └─────┘
                  │
                  ▼
      Plan 5 ComputeBackend trait ❌
                  │
                  ▼
      Plan 6 GpuBackend foundation ❌
                  │
                  ▼
      Plan 7+ per-kernel GPU ports ❌
                  │
                  ▼
      DSL compiler stream (~20 plans) ❌
        - parser → AST → type-check
        - entity / event / view / physics / mask / policy emitters
        - schema-hash emitter + drift CI
        - asset bridge from legacy `.ability` files
                  │
                  ▼
      ML reconnect ❌
        - trajectory adapter
        - NeuralBackend (Serial stub + GPU forward)
        - distillation OR retrain
                  │
                  ▼
      Game layers ❌
        - Campaign overworld
        - Mission layer
        - hero roster port
        - player UI
        - Chronicle prose
        - worldgen pipeline
                  │
                  ▼
      Production / distribution ❌
        - CI rebuild + parity + perf-regression
        - fuzz-farm scaling
        - packaging
```

---

## 8. Effort estimates

**Today's data point (2026-04-19):** one calendar day produced **149 commits** that took the engine from MVP to full Combat Foundation: the entire 6-phase pipeline, 18-MicroKind action space, cascade runtime, full state.md SoA port (28 hot + 9 cold), engagement / ZoC, ability runtime with 8 EffectOp handlers, recursion bounding, mask-cast-gate integration, plus two audits (verification audit on Plans 1+2 and stub audit on the merged Combat Foundation) all resolved. **334 debug / 333 release tests** ship green. Subagent concurrency was the key force multiplier — many plans + audits ran in parallel against isolated worktrees, then merged back to `main`.

| Horizon | Plan count | Per-plan effort estimate | Wall-clock estimate (with subagent concurrency) |
|---|--:|---|---|
| Near-term — finish engine (Plan 3 + 3–5 deferred subsystems) | 5–6 | Plan 3: ~12 tasks, ~1000 LoC. Subsystems: ~10–18 tasks, ~900–1800 LoC each. | 2–3 days |
| Near-term — engine discipline (Plans 4, 5, 6) | 3 | Plan 4: ~6–8 tasks. Plan 5: ~4–6 tasks (trait extraction, additive). Plan 6: ~8–12 tasks (largest — first GPU integration). | 2–3 days |
| Medium-term — remaining 9 deferred subsystems | 9 | ~6–22 tasks each (avg ~12). Terrain (#13) is the outlier at 22 tasks. | 5–8 days |
| Medium-term — GPU per-kernel ports | 4–6 | ~6–10 kernels per plan, each kernel = SPIR-V + Rust dispatch + parity test. | 4–7 days |
| Long-term — DSL compiler (~20 plans) | 20+ | Codegen passes ~6–10 tasks each; type checker + parser are larger. | 10–15 days |
| Long-term — ML reconnect (~3–5 plans) | 3–5 | NeuralBackend GPU integration is the unknown. | 3–5 days |
| Long-term — game layers (~8–12 plans) | 8–12 | Campaign + Mission + UI are large each. | 10–15 days |
| Long-term — production (~3–5 plans) | 3–5 | CI work is mostly YAML + harness scripts. | 2–4 days |
| **Total estimated calendar-day budget (subagent-concurrent)** | ~55–75 plans | — | **~38–60 days** of focused, well-organised work |

**Caveat:** wall-clock compresses with concurrency, but cumulative complexity does not. Each new subsystem bumps the schema hash, expands the event-ring budget, and (post Plan 6) requires a SPIR-V kernel counterpart with parity test. By the time the deferred 14 land, every subsequent change crosses a much wider verification surface than today.

---

## 9. Open strategic questions (need user input)

1. **Priority after Plan 3 + first deferred subsystems.** Roadmap recommends Items (#4) first for blast radius. After Items + Groups + Memberships land, do we (a) continue the deferred subsystem stream (Memory → Relationships → Theory-of-Mind), (b) pivot to **DSL compiler** so future subsystems are written in the language not in Rust, (c) pivot to **GPU port** to bank performance before the engine grows further, (d) pivot to **ML reconnect** so training pipelines stop being on the legacy stack, or (e) pivot to **game layers** so there's something playable to demo? Sequencing here changes the look of the next 3 months.

2. **GPU port timing.** Wait until all 14 deferred subsystems stabilize before kicking off Plans 5+6+7 (clean port, minimum churn) — OR intersperse (every ~3 subsystems → 1 GPU kernel parity port) to keep both backends in sync from day one (more churn, but no big-bang divergence)?

3. **ML reconnect scope.** Do we (a) port the existing tactical_sim training scripts (combat-student, ability-eval, ability-transformer, actor-critic-V3, next-state-prediction) to the new engine's trajectory format and `NeuralBackend`, (b) leave the legacy stack running as a separate research lane and build new training from scratch on the new engine, or (c) freeze ML work entirely until the DSL compiler exists so policies are language-emitted?

4. **Game-layer ownership.** Are Campaign and Mission layers built on top of the same engine (one engine instance per layer, with custom MicroKind/MacroKind sets per layer) or each its own subsystem with bespoke runtime? CLAUDE.md states the three-layer model but doesn't pin implementation strategy.

5. **Public release shape.** Early-access on Steam, open-source on GitHub, private/closed alpha, or commercial with public-facing dev log only? Affects CI + documentation + license decisions, and changes Plan #67 (distribution) substantially.

6. **Chronicle prose generator location.** Engine spec §26 punts text generation as non-goal — *something* downstream produces narrative prose from the event log. Which layer owns it? Options: (a) a separate `crates/chronicle/` crate that consumes event logs offline, (b) part of the Mission/Campaign layer's recap UI, (c) an LLM-driven external tool (e.g., the existing LFM agent) that subscribes to sim_bridge.

7. **Legacy `tactical_sim` retirement.** When (if ever) do we delete the legacy crate? It carries 27 base heroes, 172 LoL hero imports, the full ability DSL parser, and the existing AI pipeline. Asset bridge (#49) is the prerequisite for retiring it; after that, deletion is a workspace member removal but loses the historical training infrastructure.

---

## 10. Notes / caveats

- **Schema hash stability.** As the engine acquires more types + events + kernels, the schema hash rebases. Checkpoint compat requires version-pinning + migration story (already spec'd in §18, lands with Plan 3). Today's baseline: `7bf05a9ff5aafa7d5cf6f2ba30a00ac0353a906a1660b1b4ff993ecf69af5006` (post-Rollback-removal).
- **Cross-backend parity overhead.** Once `ComputeBackend` trait + `GpuBackend` lands (Plans 5+6), every cascade handler / mask predicate / view reduction needs a SPIR-V counterpart with a parity test. Plans 7+ and the per-subsystem plans must include "GPU-portability checklist" sections per roadmap §5b.
- **Compiler scope scales with subsystem count.** Each deferred subsystem's cascade handlers + mask predicates + view declarations become emission targets for the DSL compiler when that lands. Compiler-layer complexity is a function of (subsystems shipped × language features wanted).
- **Event-ring budget.** Combat Foundation alone added 13+ event variants. Roadmap §5c notes another ~50+ are coming across the 14 deferred subsystems. Either widen the ring, partition per-lane, or apply event-kind priority eviction (decision deferred).
- **`MicroKind` enum closed at 18.** Roadmap §5e: prefer expressing new actions as `MacroKind` cascades that lower into existing micros, rather than expanding the enum (each new variant bumps the schema hash and forces mask predicate work).
- **Aggregate pool count.** Today: `Quest` + `Group` (Pod-shaped, no instance data). Roadmap will add `BuildingData`, `RegionState`, `ItemInstance`, possibly `Room`, `ServiceContract`, `TradeRoute`. 4–6 new pools each with a `rebuild_index` story on load.
- **`LazyView` integration gap.** Trait shipped; integration into `step_full` deferred. Canary test `lazy_view_wired_into_step_full` is `#[ignore]`d until wired. Tracked in `status.md` weak-test-risks #2.
- **Legacy tactical_sim coexistence.** Workspace members: `bevy_game` (root, with legacy AI pipeline), `crates/tactical_sim`, `crates/engine`, `crates/viz`, `crates/ability_operator`, `crates/ability-vae`, `crates/combat-trainer`, `crates/world_sim_bench` (excluded). New engine is `crates/engine/`; old combat sim is in root `src/ai/core/`. Migration story per question 7 above.
- **Spec drift caveat.** `docs/dsl/state.md` is "target spec" — the engine's state-port covered the agent top-level catalogue, but Aggregate / World / detailed AgentData (~60 fields) sub-sections are mostly aspirational and land via deferred subsystems #15–#28. Reviewers comparing engine code to state.md should expect "✅ for top-level Agent / ❌ for everything else" until each subsystem lands.

---

## 11. Document freshness

Last updated **2026-04-19**. Update after every major plan's execution. Cross-references that may go stale:

- §2 commit count + test count — refreshes per plan.
- §3.1 commit ranges — refresh per plan execution.
- §6 — flip ⚠️ / ❌ → ✅ rows as plans land.
- §3.2–§3.7 — drop rows from "needed-but-not-written" as plan files get drafted.

When this doc needs rewriting from scratch (e.g., after a major architecture pivot), source-of-truth order is: `README.md` → `docs/engine/spec.md` → `docs/engine/status.md` → `docs/superpowers/roadmap.md` → `docs/superpowers/plans/*.md` → `docs/audit_2026-04-19.md` and the two engine audits.
