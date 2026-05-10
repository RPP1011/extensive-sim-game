# Viewer Ability Visualization

> Goal: turn the viewer into a debugging-grade visualization of ability use that's also rich enough to read as game-feel. Treat *debugging* as the **superset** — a debugger needs every signal a game-feel viewer would have, plus information density layers on top. One viewer, switchable overlays — not two systems.

## Goal

Today the viewer shows agents moving + dying + per-tick HP halos derived from HP-delta detection. That's enough to know "something happened," but it can't:

- Distinguish ability identity (Strike vs Snipe vs ConcussiveBlow render identically).
- Show source attribution (which hero healed which ally?).
- Visualise misses (a cast that resolves to zero records leaves zero halos).
- Render AOE shape (a Spread{r=8, count=16} looks identical to a single-target hit when only one cell happens to fall in range).
- Show overhealing (HP capped at max → invisible delta).
- Show telegraphs (cast → impact rhythm).
- Smoothly animate at the render rate (visuals are pinned to the 10 Hz sim tick).

These gaps matter for both debugging *and* game-feel. The fix is a single layered architecture where each tier serves both modes; debug-only features (chronicle log, attribution arrows, decision explainer) are toggleable overlays on top of a shared rendering core.

## Layered architecture

```
┌──────────────────────────────────────────────────────────┐
│  5. Debug overlays  (toggleable via F-keys)              │
│     • Chronicle log panel                                │
│     • Source→target attribution arrows                   │
│     • Per-ability decision explainer                     │
│     • Hitbox / range visualisation                       │
│     • Numeric magnitude readouts                         │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│  4. Animation timeline  (viewer-side, render rate)       │
│     • Per-cast state machine: cast_tick → resolve_tick   │
│     • Interpolates at 60 Hz between 10 Hz sim ticks      │
│     • Multiple overlapping casts coexist                 │
│     • Sim stays deterministic; only visuals interpolate  │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│  3. DSL visualization extensions                         │
│     • New `visual { … }` block in .ability files         │
│     • Compiler emits ability metadata table              │
│     • Per-ability author specifies icon, telegraph,      │
│       projectile speed/color, animation_ticks            │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│  2. Visual primitives  (always on, populate both modes)  │
│     • AOE ring/cone/line on ground at cast position      │
│     • Beam/projectile source → target                    │
│     • Impact halo at target on resolve                   │
│     • Telegraph wedge for cast-time > 0 abilities        │
│     • HP bars + status icons (debug shows precision)     │
│     • Cooldown rings (debug shows numeric remaining)     │
│     • Per-ability identity icon over caster              │
│     • Floating damage / heal numbers                     │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│  1. Data layer                                           │
│     • Cast event stream  (intent: actor, ability_id,     │
│       target, position, cast_tick, expected_resolve_tick)│
│     • Effect event stream  (resolved: kind, magnitude,   │
│       actor, target, resolve_tick)                       │
│     • Ability metadata table  (id → name, area, range,   │
│       cooldown, plus DSL-extension viz fields)           │
│     • Joinable: (actor, ability_id, cast_tick) keys both │
│       streams so a cast knows its resolved effects (or   │
│       that it had none → miss)                           │
└──────────────────────────────────────────────────────────┘
```

The animation timeline (tier 4) is the central abstraction — it makes debug-mode-as-snapshot and game-feel-mode-as-smooth-animation possible from the same data. Without it, debug is per-tick frozen frames and game-feel is impossible.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/engine/src/ability/program.rs::EffectOp` (variant set; informs cast-event payload shape).
  - `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` — the `apply_ability` lowering writes verb-chronicle records before any effect resolves; that's the cast-intent emission point.
  - `crates/dsl_ast/src/ability_parser.rs` — current .ability parser; needs the new `visual { … }` block.
  - `crates/dsl_ast/src/ast.rs::AbilityDef` — struct holding the parsed ability; needs an optional `visual: Option<VisualMeta>` field.
  - `crates/engine/src/ability/PackedAbilityRegistry` — runtime ability table; needs to surface viz metadata to the viewer alongside gameplay metadata.
  - `crates/engine/src/gpu/event_ring.rs` — the chronicle ring infrastructure; viewer needs a host-readback path that doesn't currently exist.
  - `crates/viewer_runtime/src/lib.rs::ViewerApp` — current pilot wraps `BossFightState`; will grow to consume the cast/effect streams + metadata table from any fixture.
  - `~/Projects/voxel_engine/src/ui/mod.rs::EguiState` — paint surface for HUD overlays (Phase C of the viewer plan).

  Search method: `rg`, `grep -nE`, direct `Read`.

- **Decision:** new — full-stack addition spanning DSL parser/AST/lowering, engine ability metadata exposure, runtime chronicle readback, and viewer rendering primitives. Each tier is additive (no existing surface changes its semantics).

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `assets/ability_test/**/*.ability` get optional `visual { … }` blocks added per-ability where authored. Existing files without the block keep working (defaults synthesised from gameplay fields per Tier 3 fallbacks).
  - Generated outputs re-emitted: every per-runtime `OUT_DIR/generated.rs` gets the new ability-metadata table additions. Per-runtime build.rs auto-regenerates.

- **Hand-written downstream code:** YES, intentionally — the viewer crate is presentation glue (Tier 5 overlays, Tier 4 animation timeline, Tier 2 primitives are all viewer-side). Engine extensions go through the DSL emitter per P1; viewer code is hand-written because dsl_compiler is not positioned to emit windowing / egui code.

- **Constitution check:**
  - **P1 (Compiler-First Engine Extension):** PASS. The new `AbilityCast` chronicle event variant (Tier 1) is emitted by the dsl_compiler's `apply_ability` lowering, not hand-rolled in `engine/src/handlers/`. The new `visual { … }` block is parsed by `dsl_ast::ability_parser` and threaded through the existing emit pipeline.
  - **P2 (Schema-Hash on Layout):** REQUIRES BUMP. Adding the `AbilityCast` event variant changes the event variant set; `crates/engine/.schema_hash` must regenerate. Adding viz fields to ability metadata is additive to the registry layout; also bumps.
  - **P3 (Cross-Backend Parity):** N/A. Viewer is presentation-only; sim semantics unchanged. The new `AbilityCast` event needs to fire on both backends (it's compiler-emitted from the same lowering path that already runs cross-backend), so parity is automatic.
  - **P4 (`EffectOp` Size Budget):** N/A. No EffectOp changes; viz metadata lives on the registry, not on EffectOp.
  - **P5 (Determinism via Keyed PCG):** PASS. Viewer is read-only WRT sim state. No new RNG draws.
  - **P6 (Events Are the Mutation Channel):** PASS. The new `AbilityCast` event is informational (no fold consumes it); viewer reads chronicle events from the host-readback path, never writes back.
  - **P7 (Replayability Flagged):** REQUIRES DECISION. `AbilityCast` should be `replayable: true` (cast events are part of the deterministic sim record, and viewing a replay should show the same casts). `EffectStarted` / `CastStarted` / telegraph events introduced in later phases need explicit replayable flags per declaration.
  - **P8 (AIS Required):** PASS — this section satisfies it.
  - **P9 (Tasks Close With Verified Commit):** PASS — every task closes with a SHA on the active branch.
  - **P10 (No Runtime Panic):** PASS. Viewer must `Result`-bubble GPU errors and chronicle-readback failures, not `.unwrap()` on the deterministic path.
  - **P11 (Reduction Determinism):** N/A. Viewer reads, doesn't reduce.

- **Runtime gate:** every phase has a runtime test that actually exercises the new code path and asserts an observable post-condition.
  - Phase 1: `viewer_runtime::tests::reads_recent_casts_from_chronicle` — drive 10 ticks of boss_fight, assert `app.recent_casts()` contains entries for at least one of (BossStrike, HeroAttack).
  - Phase 2: `viewer_runtime::tests::aoe_shape_painted_at_cast_position` — drive a fixture with a Spread{8,16} ability, assert the bridge's CPU grid contains the AOE-ring material at `cast_center ± radius`.
  - Phase 3: `dsl_compiler::tests::ability_visual_block_parses_and_emits` — load a `.ability` with `visual { telegraph: 5t, animation: 3t, icon: "strike" }`, assert the emitted registry exposes those fields.
  - Phase 4: `viewer_runtime::tests::projectile_interpolation_advances` — between two sim ticks, drive 10 render frames, assert projectile interpolated position monotonically advances source → target.
  - Phase 5: `viewer_runtime::tests::debug_overlay_toggle` — synthesise an F1 keypress, assert the chronicle-log overlay's visibility flag flips.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

## Phasing — five PRs

Each phase is independently shippable; phase N's deliverable is usable on its own. Later phases extend earlier ones additively.

| # | Phase | Deliverable |
|---|---|---|
| **1** | **Data layer**: cast + effect event readback, ability metadata exposure | Per-fixture `read_recent_casts() -> &[AbilityCast]`, `read_recent_effects() -> &[ResolvedEffect]`, `ability_metadata() -> &AbilityMetadataTable`. New `AbilityCast` chronicle event variant emitted by the dsl_compiler from every `apply_ability` lowering. ViewerApp consumes both streams; can `eprintln!` per-cast traces in the console driver. No visual changes yet. |
| **2** | **Primitive library** (snapshot, no interpolation) | Bridge gains `paint_aoe_ring(center, radius, material)`, `paint_beam(source, target, material)`, `paint_identity_icon(over, ability_id)`. egui HUD gains floating-magnitude widget (world→screen project). All primitives drive from cast + effect events; metadata-aware where possible (AOE ring uses ability's spread radius). Per-tick snapshot — visible at 10 Hz cadence. |
| **3** | **DSL visualization extensions** | New `visual { telegraph: <ticks>, animation: <ticks>, projectile: { speed: f32, color: rgba } | none, icon: "<name>", aoe_ring_color: rgba | inherit }` block in `.ability` files. Parser + AST + lowering changes. Per-fixture metadata picks up new fields. Phase 2 primitives query metadata where present, fall back to defaults synthesised from gameplay fields. |
| **4** | **Animation timeline** | Viewer-side state machine. Per cast event, spawn timeline entry tracking `cast_tick / resolve_tick / ability_id`. Render-frame interpolation produces position(t), alpha(t), scale(t) per primitive. Projectile travels source → target over `animation_ticks * (1/sim_rate)`. AOE telegraph fades over `telegraph_ticks`. Multiple overlapping casts coexist. Sim cadence unchanged; only visuals smooth. |
| **5** | **Debug overlays** | F1 = chronicle log panel (scrolling list of recent events with filter UI). F2 = source→target attribution arrows. F3 = hitbox/range visualisation (faint disc around each agent showing their attack range). F4 = decision explainer (which `where` clause + scoring row picked this ability this tick — requires reading the per-fixture decision data; per-fixture). F5+ = future. Each overlay independently toggleable via egui state + keypress handler. |

Phase 1 unblocks 2-5. Phase 2 + 3 unblock 4 (animation timeline needs primitives to interpolate AND DSL fields to know the animation_ticks). Phase 5 sits on top of 1+2 (doesn't need 3 or 4).

Suggested merge order: **1 → 2 → 5 → 3 → 4**. Rationale: 5 (overlays) gives the most debugging value per LOC and depends only on 1+2; 3 (DSL extensions) is heavier and unblocks 4 (animation), so we ship visible-debug-value first, polish-and-game-feel last.

## Decisions to lock in BEFORE starting

These need a sign-off before Phase 1 starts because they ripple downstream:

### D1: Cast event source — read existing verb-chronicle ring vs add new `AbilityCast` event variant?

**Option A** (read existing): The `apply_ability` lowering already writes per-verb chronicle records (`physics_verb_chronicle_BossStrike` etc.) before any effect resolves. These records contain `actor`, `target`, position. Add a host-readback path; no new event variant. **Pros**: zero schema-hash bump, no compiler changes. **Cons**: per-verb event types are fragmented (one variant per ability); viewer needs to deserialise N variants and unify them; ability identity is implicit in the variant tag, not an explicit `ability_id` field.

**Option B** (new variant): Add a single `AbilityCast { actor, ability_id, target_slot, cast_position }` event the compiler emits unconditionally from the `apply_ability` lowering, alongside the per-verb records. **Pros**: one event variant for the viewer to read, explicit ability_id field, future-proof. **Cons**: schema-hash bump (P2); slightly more chronicle bandwidth (~16-24 bytes/cast, negligible at typical cast rates).

**Recommendation**: B. The slight cost is worth the simpler downstream contract — every viewer/inspector/replay tool sees one consistent surface.

### D2: Where does `AbilityMetadata` live?

Today each fixture builds its own `AbilityRegistry` from .ability files via the dsl_compiler. The viewer needs to query `(ability_id, kind, area_shape, range, cooldown, viz_fields)` per cast.

**Option A** (per-fixture exposure): each `*_runtime` adds a `pub fn ability_metadata() -> &AbilityMetadataTable` accessor that proxies its internal registry. **Pros**: matches current pattern, no engine changes. **Cons**: duplicated boilerplate in every runtime; viewer needs to know which fixture's accessor to call (works for our hardcoded pilot, brittle for Phase E multi-fixture).

**Option B** (shared engine helper): add `engine::ability::PackedAbilityRegistry::metadata_table() -> AbilityMetadataTable` and have every runtime expose its internal registry through a uniform `pub fn registry() -> &PackedAbilityRegistry` method. Viewer queries via the shared type. **Pros**: one viewer-side path; matches the existing `PackedAbilityRegistry` shape. **Cons**: cross-cutting change to every runtime crate (~40 crates).

**Recommendation**: B, but defer the per-runtime sweep — Phase 1 does the engine-side helper, the pilot fixture (boss_fight) opts in, other runtimes opt in lazily as Phase E lands them in the viewer.

### D3: Animation timing — fixed budget per ability or per-instance scriptable?

**Option A** (fixed budget): one `animation_ticks` field per ability. All instances of a Strike take the same time to play. **Pros**: simple; matches deterministic sim cadence. **Cons**: no in-flight variation (e.g. travel-time scales with distance for a projectile).

**Option B** (per-instance): cast event includes `expected_resolve_tick` based on dynamic factors (distance, speed). Animation timeline interpolates over that span. **Pros**: realistic projectile travel; supports cast-time variation. **Cons**: requires the dispatcher to compute resolve time at cast emission — small but real lowering change.

**Recommendation**: B for projectiles, A for everything else. The cast event carries `expected_resolve_tick` (= cast_tick + travel_ticks for projectiles, = cast_tick + 1 for instant abilities), giving each cast a precise timeline. Travel ticks computed in the lowering as `ceil(distance / projectile_speed * sim_rate)` when the ability has a projectile spec; defaults to instant otherwise.

### D4: How "smooth" does the animation need to be?

**Option A** (snap to ticks): visuals advance only on sim tick boundaries. 10 Hz visual rate. Cheap.

**Option B** (interpolated): viewer-side timeline state machine produces interpolated position(t) per render frame at 60 Hz. ~6× smoother apparent motion. More code.

**Recommendation**: B. Without interpolation, projectile travel reads as a stutter-step rather than motion. The state machine is small (~150 LOC); the win is qualitative.

### D5: Debug overlay UX — egui panels vs in-world overlays vs both?

**Option A** (egui only): all debug info in the egui panel layer (chronicle log, magnitude numbers, etc.). World stays "game" looking.

**Option B** (in-world only): attribution arrows, hitboxes, decision overlays drawn as voxel splats / lines in the world.

**Option C** (both): per-overlay choice — chronicle log in egui, attribution arrows in world, magnitude numbers in egui floating above target.

**Recommendation**: C. Overlay placement matches its information character — log scrolls in egui, position-tied annotations live in the world.

## Cross-cutting concerns

- **AbilityId stability**: viewer keys metadata by ability_id. Today each fixture's registry numbers abilities 0..N in registration order. As fixtures grow, ids shift. The viewer must read the metadata table at construction time and keep references stable per-session; a fixture restart re-reads. No global ability id namespace today; this plan doesn't add one (each fixture's table is self-consistent).

- **Cast → Effect joining**: a cast event with `(actor, ability_id, cast_tick=N)` should associate with effect events from the same actor in tick range `[N, N + max_resolve_delay]`. For instant abilities `max_resolve_delay = 1`. For projectiles up to `expected_resolve_tick - cast_tick`. Misses surface as casts with no matched effects after the window closes.

- **Chronicle ring buffering**: the engine's chronicle ring is a fixed-capacity append-only buffer cleared per tick. Viewer-side host readback drains the most recent tick's contents into a ring of (cast_event, resolved_at_tick) pairs the viewer maintains for ~30 ticks (long enough to display fade-out animations). Don't unbound — debug-overlay chronicle log can scroll a much larger viewer-side history (~10k events) but the per-tick join window stays small.

- **Replay determinism**: chronicle is read-only on the viewer side; cast events don't mutate sim state. Viewer is purely a consumer. A snapshot-replay of the sim produces identical chronicle → identical viewer output. Good for golden-image testing.

- **Performance budget**: cast events at typical fixtures are < 100/tick (boss_fight: ~6 casts/tick max). Per-tick chronicle drain + viewer-side processing is microseconds. Animation timeline holds < 200 active casts in worst case (10 Hz × 20 second animation per cast); per-render-frame state-machine update is microseconds. No performance concerns at fixture scale; revisit if a fixture with > 10k agents opts in.

- **Per-fixture vs shared**: chronicle event format is engine-shared. Cast event, ability metadata table — engine-shared via D1+D2 recommendations. Viewer primitives consume the engine surface uniformly across fixtures. Each fixture can author its own `.ability` files with arbitrary `visual { … }` blocks; viewer renders without knowing about the fixture.

## Critical files

- `crates/dsl_ast/src/ability_parser.rs` — parse new `visual { … }` block.
- `crates/dsl_ast/src/ast.rs::AbilityDef` — add `visual: Option<VisualMeta>` field.
- `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` — emit `AbilityCast` chronicle write inside the `apply_ability` lowering (Phase 1 / D1B).
- `crates/dsl_compiler/src/cg/emit/program.rs` — extend ability registry emit to include viz fields (Phase 3).
- `crates/engine/src/ability/program.rs` — add `AbilityCast` event variant; bump P2 schema hash.
- `crates/engine/src/ability/registry.rs` — `PackedAbilityRegistry` gains viz metadata fields + `metadata_table()` accessor (D2B).
- `crates/engine/src/gpu/event_ring.rs` — host-readback API for the most recent N events (Phase 1).
- `crates/viewer_runtime/src/lib.rs` — `ViewerApp` reads cast/effect streams + metadata; spawns animation-timeline entries.
- `crates/viewer_runtime/src/voxel_bridge.rs` — new primitive paints (AOE ring, beam, projectile, telegraph).
- `crates/viewer_runtime/src/animation.rs` (new) — per-cast state machine, render-rate interpolation.
- `crates/viewer_runtime/src/debug_overlays.rs` (new) — egui overlay state + per-overlay paint.
- `crates/viewer_runtime/src/bin/viewer_window.rs` — wire keypress handlers for F1-F5 overlay toggles; pass animation timeline state through `paint_hud`.

## Out of scope (explicitly)

- **Full particle system** — sparks, smoke, screen shake. Hand-rolled in voxel_engine if/when game-feel becomes the primary mode.
- **Audio** — sound effects per cast; defer.
- **3D models for abilities** — voxel_engine renders voxels; ability animation primitives stay voxel-cube-shaped (rings, beams, splats). Imported meshes are a separate concern.
- **Camera follow / cinematic cameras** — top-down stays.
- **Network / multiplayer cast prediction** — single-process sim.
- **Cross-fixture ability inspector UI** — Phase 5 debug overlays show per-fixture; a "compare ability X across fixtures" UI is out of scope.
- **DSL extensions for FX beyond visualization** — `visual { … }` is purely about how an ability is *displayed*, not *what it does*. Anything that changes effect semantics belongs in the existing `effects { … }` block.

## Verification end-to-end

After all 5 phases land:

1. `cargo build --release` — clean across the workspace.
2. `cargo test -p dsl_compiler --release` — DSL parser tests pass (including new `visual { … }` parsing).
3. `cargo test -p engine --test schema_hash --release` — schema hash regenerated to include `AbilityCast` variant.
4. `cargo test -p viewer_runtime --release` — all five phase runtime gates pass.
5. Manual: `cargo run -p viewer_runtime --bin viewer_window --release` against boss_fight:
   - Visible AOE ring on the ground when an AOE ability fires (Phase 2).
   - Source → target beam for ~3 frames after a cast (Phase 2).
   - Smooth projectile travel from caster to target across ~3-5 render frames (Phase 4).
   - Telegraph wedge fades during cast time before impact (Phase 4 + Phase 3).
   - Floating "−12" damage / "+8" heal numbers above target on resolve (Phase 2).
   - F1 toggles a scrolling chronicle log panel (Phase 5).
   - Per-ability identity icon visible above caster on cast (Phase 2 + Phase 3).
6. Manual: edit a `.ability` file's `visual { … }` block (e.g. change `icon: "strike"` to `icon: "snipe"`), `cargo run` again — icon changes without source-code touch (Phase 3).

## Why path C (viewer_engine integration) is still the right base

This plan extends the path-C viewer architecture (per `docs/superpowers/plans/2026-05-09-viewer-runtime.md`) — voxel_engine-as-renderer with egui overlays. The animation timeline + debug overlays + DSL viz extensions all build on the existing `EguiState` + `VoxelBridge` + `ViewerApp` scaffold. No re-architecture; pure additive layering.

If at the end of Phase 4 the animation feels constrained by voxel_engine's render model (single-color-per-object, no native sprite/billboard support), that's the trigger to consider a Bevy migration *for the viewer specifically* — sim stays put. Not a current concern.
