# Roadmap

> Comprehensive index of future work. Two lines per item (title + scope/reference).
> Stable across day-to-day churn; update when items change tier (deferred → drafted → active → done) or when a new tier-2 item is identified.
>
> Canonical specs live under `docs/spec/`; in-flight plans + brainstorms under `docs/superpowers/`.
> Live engine status (per-subsystem ✅/⚠️/❌): `docs/engine/status.md`.
> Project overview (architectural intro): `docs/overview.md`.

---

## Active (plan written, in flight)

- **GPU sim state (Subsystem 1)**
  CPU `SimState` → GPU-resident mirror with `@cpu_only` opt-out. `plans/2026-04-22-gpu-sim-state.md`.
- **GPU cold-state replay — Phase 3 (gold + standing)**
  Gold + standing as GPU-resident `@materialized` views. `plans/2026-04-23-cold-state-phase-3-gold-standing.md`.
- **GPU megakernel**
  Single-dispatch tick path; perf-sweep harness at N=200k. `plans/gpu_megakernel_plan.md`, `plans/gpu_megakernel_perf.md`.

## Drafted (spec exists, plan does not)

- **Ability DSL implementation**
  Voxel ops, control verbs (root/silence/fear/taunt), AI-state manipulation, structures, materials. `spec/ability.md`.
- **Economic depth implementation**
  Recipes, contracts, labor, heterogeneity, information asymmetry, market structure, macro dynamics. 3 phases. `spec/economy.md`.
- **GPU ability evaluation (Subsystem 3)**
  Move ability evaluator kernel from CPU to GPU. `spec/gpu.md` §5.

## Engine plans not yet written

- **Plan 4 — debug & trace runtime (§23)**
  `trace_mask`, `causal_tree`, `tick_stepper`, `tick_profile`, `agent_history`, snapshot repro bundle.
- **Plan 5 — `ComputeBackend` trait extraction (§24)**
  Abstract Serial vs GPU under one trait; cross-backend parity-test infrastructure.
- **Plan 6 — `GpuBackend` foundation**
  Bridge from `ComputeBackend` trait to `engine_gpu` primitives. Prerequisite: Plan 5.
- **Plan 7+ — per-kernel GPU porting under the trait**
  Each kernel migrates with parity tests against `SerialBackend` reference.

## Deferred world-sim subsystems

Storage stubs are already in `SimState` (state-port plan, in git history); behaviour attachment is the remaining work. Sequencing + dependency graph in `docs/superpowers/roadmap.md`.

- **Memberships** — group lookup + role-gated cascades.
- **Memory** — `@per_entity_ring(K=64)` view; @materialized version retired 2026-04-23.
- **Relationships** — directional pair-wise sentiment + decay.
- **Items** — `Pool<ItemTag>` + ownership + transfer.
- **Groups** — entity_pool aggregate; standing view.
- **Factions** — multi-group emergent standing on top of Groups.
- **Buildings** — derived view over voxels + harvest events.
- **Settlements** — building-cluster aggregation.
- **Regions** — spatial-region aggregation: climate, biome, history.
- **Quests** — `AggregatePool<Quest>` + lifecycle handlers + posting/accepting cascades.
- **Theory-of-mind** — `believed_knowledge` with per-bit volatility (Short/Medium/Long half-lives).
- **Personality utility** — `risk_tolerance` / `social_drive` / `ambition` / `altruism` / `curiosity` drive scoring.
- **Terrain** — 3D voxel-aware navigation, height, walkability, line-of-sight.
- **Interior nav** — room-graph, doors, line-of-sight inside structures.

## Game / UX layer (player-facing)

Documented at a high level in `docs/overview.md`; not yet plan-decomposed. Most depend on world-sim and engine plans landing first.

- **Campaign overworld** — turn-based hero roster + flashpoint dispatch.
- **Mission system** — multi-room dungeons / sieges / flashpoints; drops into combat sim.
- **Viz harness extensions** — scenario authoring UI, replay debugger, in-window HUD (currently stdout-only).
- **ML training pipelines** — external pytorch; engine emits dataclasses + Dataset over trace format. Spec §10.
- **Voxel backend** — `crates/engine_voxel/` placeholder; ability DSL voxel ops depend on it.

## Open technical debt / verification questions

Tracked in `docs/engine/status.md` "Open verification questions" + "Top weak-test risks". Lifted here for visibility.

- **4 named Combat Foundation regression fixtures** — 2v2-cast, tax-ability, meteor-swarm, tank-wall as standalone tests. Mechanics done; named scenarios live implicitly in other suites.
- **LazyView wiring into `step_full`** — trait + impl exist; tick-pipeline integration is `#[ignore]`'d. `tests/view_lazy.rs::lazy_view_wired_into_step_full`.
- **Collision detection** — agents can co-occupy a `Vec3`; viz works around via vertical voxel stacking. status.md open question #11.
- **Announce 3D vs planar distance** — spec §10 silent; impl uses `Vec3::distance` (3D). status.md open question #1.
- **Per-tick alloc in `NeighborSource<K>`** — tolerated for MVP; needs a `SimScratch` slot.
- **`PolicyBackend::evaluate` zero-alloc budget** — currently `≤16` blocks; should be 0.

## How this doc stays current

- Items move down the tiers (Active → Drafted → Engine plans → Deferred) when their plan is **drafted** or their spec is **landed**, not when they're merely "planned in conversation".
- Items leave the doc entirely when **fully merged** — git history is the record. Don't archive done items here.
- One commit per tier change keeps `git log -- docs/ROADMAP.md` legible as a status timeline.
