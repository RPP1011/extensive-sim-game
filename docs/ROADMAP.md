# Roadmap

> Comprehensive index of future work. Two lines per item (title + scope/reference).
> Stable across day-to-day churn; update when items change tier or when a new
> tier-2 item is identified.
>
> Canonical specs live under `docs/spec/`; in-flight plans + brainstorms under `docs/superpowers/`.
> Live engine status (per-subsystem ✅/⚠️/❌): `docs/engine/status.md`.
> Project overview (architectural intro): `docs/overview.md`.

---

## Active (plan written, in flight)

- **GPU megakernel**
  Single-dispatch tick path; perf-sweep done at N=200k, optimization ongoing. `plans/gpu_megakernel_plan.md`, `plans/gpu_megakernel_perf.md`.
- **GPU cold-state replay (Subsystem 2) — Phases 2–4**
  Phase 1 (DSL-compiler view annotations) done; Phase 2 (chronicle) done; Phase 3 (gold + standing) done; Phases 2–4 umbrella plan kept for future per-phase plans. `plans/2026-04-22-gpu-cold-state-replay.md`.
- **DSL Authoring Engine — IR interpreter (P1a + P1b)**
  Visitor-pattern interpreter on top of an extracted `dsl_ast` crate; `interpreted-rules` cargo feature gates the runtime path. Unlocks fast `.sim` iteration without regen-and-recompile. Porting from `wsb-engine-viz` branch (16 commits, 2026-04-26).

## Drafted (spec exists, plan does not)

- **Ability DSL implementation**
  Voxel ops, control verbs (root/silence/fear/taunt), AI-state manipulation, structures, materials, passive triggers. `spec/ability.md`.
- **Economic depth implementation**
  Recipes, contracts, labor, heterogeneity, information asymmetry, market structure, macro dynamics. 3 phases. `spec/economy.md`.
- **GPU ability evaluation (Subsystem 3)**
  Move ability evaluator kernel from CPU to GPU. `spec/engine.md` §11.

## Engine plans not yet written

- **Plan 5b–e — Remaining ComputeBackend phases**
  Phase 2: GPU impl bodies. Phase 3: PolicyBackend extraction. Phase 4-5: full kernel-dispatch surface + cross-backend parity tests. Prerequisite: Plan 5a complete.
- **Plan 6 — `GpuBackend` foundation**
  Bridge from `ComputeBackend` trait to `engine_gpu` primitives. Prerequisite: Plan 5 complete.
- **Plan 7+ — per-kernel GPU porting under the trait**
  Each kernel migrates with parity tests against `SerialBackend` reference.

## Partially landed (DSL stubs / MVP seam done; behaviour attachment pending)

Subsystems where parser/resolver/storage scaffolding has landed but cascade rules + behaviour are still being attached. Full per-subsystem detail in `docs/superpowers/roadmap.md`.

- **Memberships** — DSL stubs landed (commit `5ebe80ce`); group lookup + role-gated cascades pending.
- **Memory** — `@per_entity_ring(K=64)` view + GPU driver landed (commit `70b6cc84`); memory-folded behaviour pending.
- **Relationships** — DSL stubs landed (commit `76dcdbdc`); pair-wise sentiment + decay pending.
- **Groups** — DSL stubs landed (commit `2b882c59`); aggregate pool + standing view pending.
- **Quests** — DSL stubs landed (commit `16a857ae`); `AggregatePool<Quest>` + lifecycle handlers pending.
- **Theory-of-mind** — DSL stubs landed (commit `63fa6b64`); Phase 1 (full belief state) landed 2026-04-26 across plan `plans/2026-04-25-theory-of-mind-impl.md`; Phase 2 (second-order, terrain LOS, lying, trust) deferred.
- **Terrain** — MVP `TerrainQuery` trait seam in `crates/engine/src/terrain.rs` (commit `856fe171`); voxel adapter + 3D walkability pending.

## Deferred world-sim subsystems (no work yet)

Storage stubs are already in `SimState`; behaviour attachment is the remaining work. Sequencing in `docs/superpowers/roadmap.md`.

- **Items** — `Pool<ItemTag>` + ownership + transfer.
- **Factions** — multi-group emergent standing on top of Groups.
- **Buildings** — derived view over voxels + harvest events.
- **Settlements** — building-cluster aggregation.
- **Regions** — spatial-region aggregation: climate, biome, history.
- **Personality utility** — `risk_tolerance` / `social_drive` / `ambition` / `altruism` / `curiosity` drive scoring.
- **Interior nav** — room-graph, doors, line-of-sight inside structures.

## Game / UX layer (player-facing)

Documented at a high level in `docs/overview.md`; not yet plan-decomposed. Most depend on world-sim and engine plans landing first.

- **Campaign overworld** — turn-based hero roster + flashpoint dispatch.
- **Mission system** — multi-room dungeons / sieges / flashpoints; drops into combat sim.
- **Viz harness extensions** — scenario authoring UI, replay debugger, in-window HUD (currently stdout-only).
- **ML training pipelines** — external pytorch; engine emits dataclasses + Dataset over trace format. Spec §10.
- **Voxel backend** — `crates/engine_voxel/` placeholder (does not yet exist); ability DSL voxel ops depend on it.

## Open technical debt / verification questions

Tracked in `docs/engine/status.md` "Open verification questions" + "Top weak-test risks". Lifted here for visibility.

- **4 named Combat Foundation regression fixtures** — 2v2-cast, tax-ability, meteor-swarm, tank-wall as standalone tests. Mechanics done; named scenarios live implicitly in other suites.
- **LazyView wiring into `step_full`** — trait + impl exist; tick-pipeline integration is `#[ignore]`'d. `tests/view_lazy.rs::lazy_view_wired_into_step_full`.
- **Collision detection** — agents can co-occupy a `Vec3`; viz works around via vertical voxel stacking. status.md open question #11.
- **Announce 3D vs planar distance** — spec §10 silent; impl uses 3D via `spatial.within_radius`. status.md open question #1.
- **Per-tick alloc in `NeighborSource<K>`** — tolerated for MVP; needs a `SimScratch` slot.
- **`PolicyBackend::evaluate` zero-alloc budget** — currently `≤16` blocks; should be 0.
- **Passive triggers spec mismatch** — `spec/ability.md` §6/§23.1 marks `passive` block + on_damage_dealt/on_hp_below/etc as `runs-today`, but no Trigger AST node or handler exists. Spec is overclaiming; either implement or downgrade the markers to `planned`.

## How this doc stays current

- Items move down the tiers (Active → Drafted → Engine plans → Partially landed → Deferred) when their plan is **drafted** or their spec is **landed**, not when they're merely "planned in conversation".
- Items leave the doc entirely when **fully merged** — git history is the record. Don't archive done items here.
- One commit per tier change keeps `git log -- docs/ROADMAP.md` legible as a status timeline.
