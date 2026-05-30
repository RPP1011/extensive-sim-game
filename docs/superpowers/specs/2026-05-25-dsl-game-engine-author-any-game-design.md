# DSL Game Engine #1 — Author Any Game From the `.sim` (Design)

> Status: design, awaiting review. Next: `writing-plans` → implementation plan with AIS (P8).
> This is **direction #1 of a 4-part engine roadmap** the user set (2026-05-25): (1) author any game from the DSL [this spec], (2) editor + fast iteration, (3) real rendering + game feel, (4) distributable runtime. Built in order; each builds on the prior. This spec is the foundation — it defines the runtime seam (#2–#4 all build on it).
> Builds on the playable-vampire-survivors work (`docs/superpowers/specs/2026-05-25-interactive-runtime-playable-vs-design.md`, Plans 1–4 landed on branch): the runtime input channel, `engine_ui`, and the input-driven VS gameplay. **This spec absorbs the queued Plan 5** (the `ui {}` block) into a broader declarative trio.

## 1. Goal

Make a new playable game **a `.sim` file, not a Rust project.** Today, making `vampire_survivors` playable took a hand-written `vs_viewer` binary, a `VsBridge` renderer, a `vs_ui` host state machine, WASD wiring, and a mana-band role hack — all bespoke. This spec lifts that glue into a **declarative layer in the DSL + one generic player binary**, so a second (and Nth) game runs through the same path with **zero new Rust**.

Success = `cargo run -p engine_play --bin play vampire_survivors` and `... play predator_prey` both open playable windows, driven entirely by `render {}` / `controls {}` / `ui {}` blocks in their `.sim` files — and the hand-written `vs_*` files are deleted.

## 2. Background: the two layers we built

The playable-VS work produced two distinct kinds of code:

- **Already general (keep, build on):** the `.sim` → WGSL compiler, the deterministic event/fold model + keyed RNG, the `@runtime` input channel (Plan 1), `engine_ui` (Plan 2, sim-agnostic), the summon/ability system, spatial queries, materialized views.
- **vampire-survivors-specific glue (the target of this spec):** `crates/viewer_runtime/src/bin/vs_viewer.rs` (window/loop/camera/controls), `vs.rs::VsBridge` (arena floor, palette, world→voxel, role→color, the hand-coded bolt/nova/garlic/whip VFX), `vs_ui.rs` (host progress/menu/death state machine + WASD→`set_config_ctl_*`), the mana-band role discriminant, and the per-fixture `GeneratedRuntime` shape that forces a per-game binary.

"Make it an engine" = make that second layer **declarative + generic**.

## 3. The crux: a uniform runtime seam so one binary runs any game

**Decision: Approach A — a generated `PlayableRuntime` trait + a name→runtime registry, consumed by a generic player.**

Today each fixture is its own `GeneratedRuntime` struct (own `try_new`/`step`/buffer fields), so no binary can run "any" `.sim`. The emitter already generates `step()`, `set_config_*` setters, `ui_descriptor()` (Plan 5). Generalize that into a uniform trait every fixture's runtime implements, plus the two new descriptor accessors:

```rust
// crate: engine_play_api (leaf crate — trait + descriptor structs, no heavy deps)
pub trait PlayableRuntime {
    fn tick(&self) -> u64;
    fn step(&mut self);                                  // includes the summon drain
    fn set_input(&mut self, field: &str, value: f32);    // dispatch by @runtime field name
    fn agent_snapshot(&mut self) -> Vec<AgentView>;      // pos + alive + the columns render keys on
    fn view_value(&mut self, view: &str, slot: u32) -> f32; // materialized-view readback (e.g. xp)
    fn render_descriptor(&self)   -> &'static str;       // JSON, emitted from render {}
    fn controls_descriptor(&self) -> &'static str;       // JSON, emitted from controls {}
    fn ui_descriptor(&self)       -> &'static str;       // JSON, emitted from ui {} (Plan 5)
}
pub struct AgentView { pub pos: [f32; 3], pub alive: bool, pub hp: f32, pub mana: f32, pub move_speed: f32, pub creature_type: u32 }
```

- **Crate topology (avoids a dependency cycle):** `engine_play_api` (leaf: trait + descriptor types) ← `sims` (generates the trait impls + a registry `pub fn make_playable(name: &str) -> Option<Box<dyn PlayableRuntime>>`) ← `engine_play` (the generic viewer/input/UI loop + the `play` binary; also depends on `engine_ui` + `voxel_engine`).
- **`set_input` by name** dispatches to the generated `set_config_ctl_*` setters (the impl matches the field string; u32 fields like `bolt_rate_level` cast from the f32 arg).
- **`agent_snapshot`** exposes a fixed set of common columns (pos/alive/hp/mana/move_speed/creature_type) that `render {}` keys on — the same columns `VsBridge` already reads. (Arbitrary-field keying is a future refinement.)

**Rejected — B (per-game TOML config outside the `.sim`):** abandons the DSL-as-engine premise; declarations belong in the `.sim`. **Rejected — C (reusable Rust viewer lib, per-game Rust descriptor):** "new game = write Rust" is not an engine.

## 4. The declarative trio (DSL → JSON descriptors)

Three new top-level `.sim` blocks, each lowered by the compiler to a `&'static str` JSON descriptor on the generated runtime (the exact mechanism Plan 5 defined for `ui {}`: hand-emit JSON, parse in the consumer crate with serde). `engine_play` deserializes them.

- **`render {}`** — replaces `VsBridge`:
  - `arena { radius | bounds }`, `camera: follow(<agent-selector>) | observer`, top-down/iso.
  - per-agent visuals **by field range**: `agent where mana in [0.5,1.5] { color: cyan }` … (reuses the mana-band/field discriminant `VsBridge` already uses — no new runtime support; the snapshot exposes the columns).
  - `vfx on <Rule> { kind: ring|beam, radius: <expr/const>, color }` — generalizes the hand-coded bolt-beam / nova-/garlic-/whip-rings. The descriptor names a rule + a periodicity the player times against `tick`.
- **`controls {}`** — replaces the WASD wiring: `key W -> ctl.move_y: 1.0`, `key S -> ctl.move_y: -1.0`, … maps held/pressed keys to `@runtime` input-field writes (`set_input`). Normalization (e.g. diagonal movement) handled generically by the player.
- **`ui {}`** — exactly Plan 5: HUD widgets + menu/end screens, bound to `view_value`/snapshot fields and `set_input` actions. The host menu→input loop becomes generic (engine_ui already returns `UiAction`; the generic player applies it via `set_input`).

## 5. Phases (one spec, internally gated — like the playable-VS spec)

1. **`engine_play_api` + `PlayableRuntime` trait + generated impl + registry.** Emitter generates the trait impl per fixture (wrapping existing `step`/setters/buffers) + `sims::make_playable`. Verify: a headless test constructs `make_playable("vampire_survivors")`, steps, reads a snapshot, sets an input — no per-fixture code.
2. **`render {}` block + descriptor + generic bridge.** New grammar + JSON emit; a generic `EngineBridge` in `engine_play` paints the voxel grid from the descriptor + snapshot (port `VsBridge`'s arena/floor/splat/VFX logic to be descriptor-driven). Verify: compile-gate (descriptor JSON) + a headless bridge unit test.
3. **`controls {}` block + descriptor + generic input.** Grammar + emit; generic key→`set_input` mapping in the player. Verify: compile-gate + a pure unit test of the key-map → input-write resolution.
4. **`ui {}` block (absorbs Plan 5).** Grammar + emit + `engine_ui::UiModel::from_json`. Verify: Plan 5's compile-gate + round-trip tests.
5. **`engine_play` crate + generic `play <fixture>` binary.** The window/loop/camera consuming the trait + all three descriptors + `engine_ui`. Verify: `cargo run ... play vampire_survivors` (manual, user-side) + a headless construct-and-step smoke.
6. **Migrate VS to the generic path + delete the glue.** Add `render{}`/`controls{}`/`ui{}` to `vampire_survivors.sim`; confirm `play vampire_survivors` reaches parity with the old `vs_viewer`; **delete `vs_viewer.rs`, `vs.rs::VsBridge`, `vs_ui.rs`** (keep only what the generic path can't yet express, documented).
7. **Convert predator_prey (play as the Hare) — the generality proof.** Add `render{}`/`controls{}`/`ui{}` to `predator_prey.sim`; a `@runtime ctl` block + a `HareControl` rule (drive a Hare by input, replacing autonomous `MoveHare` for the player's Hare); `play predator_prey` is playable (evade wolves, survive timer/score HUD) with **zero new Rust**. This is the test that the abstractions aren't VS-in-disguise.

## 6. Constitution check (for the plan's AIS, P8)

- **P1 (Compiler-First):** PASS — strengthens it: render/controls/UI become compiler-lowered data, not hand-written Rust. The generic player + `EngineBridge` are presentation/runtime (not engine-rule behavior); the `PlayableRuntime` trait is a sanctioned runtime seam beside the existing `Backend` trait.
- **P2 (Schema-Hash):** N/A — descriptors are inert strings; no SoA/event-layout change. (predator_prey's `HareControl` adds `@runtime` cfg fields, not Agent SoA columns.)
- **P3 (Parity):** PASS — descriptors/`set_input` lower identically across backends; host input write is the determinism boundary.
- **P5 (Keyed PCG):** PASS — RNG paths unchanged.
- **P6 (Events):** PASS — input read-only into rules; mutations stay event/kernel.
- **P10 (No panic):** descriptor parse is fallible with a fallback (per Plan 5); headless smokes gate the runtime paths.
- **P8:** the plan carries the full AIS.

## 7. Top risks

- **Generic snapshot vs arbitrary fields.** Starting with fixed columns (pos/hp/mana/move_speed/creature_type) covers VS + predator_prey, but a 3rd game may need a field not exposed. Mitigation: the column set is extensible; document the limit.
- **Crate-dependency cycle.** The `engine_play_api` leaf crate is the explicit fix; the plan must not let `sims` and `engine_play` depend on each other.
- **`render {}` expressiveness.** Generalizing VFX (beam-to-nearest, periodic rings) into a declarative form without an explosion of cases — keep the VFX vocabulary minimal (ring/beam + a rule trigger), accept that exotic effects stay future work.
- **Two new DSL blocks (render/controls) + grammar churn.** Parser/emitter work comparable to the `ui {}` block; keep each block minimal.
- **predator_prey shape gaps.** "Play as the Hare" may surface a control/render need VS didn't (e.g. no waves, a survive-timer win condition). That's the point — but budget for 1–2 small declarative-layer extensions it forces.
- **Deleting `vs_*` prematurely.** Only delete after `play vampire_survivors` demonstrably reaches parity (Phase 6 gate), else keep the old binary until it does.

## 8. File map

- `crates/engine_play_api/` — `PlayableRuntime` trait + `AgentView` + descriptor structs (new leaf crate).
- `crates/dsl_ast/src/{ast,parser}.rs` — `render {}`, `controls {}`, `ui {}` grammar (extend).
- `crates/dsl_compiler/src/cg/emit/` + `build_helper.rs` — emit the three descriptors + the `PlayableRuntime` impl + `make_playable` registry (extend).
- `crates/sims/` — generated impls + registry (regen).
- `crates/engine_play/` — generic `EngineBridge`, input mapper, player loop, `play` binary; depends on `sims`, `engine_play_api`, `engine_ui`, `voxel_engine` (new crate).
- `assets/sim/vampire_survivors.sim` — add render/controls/ui blocks (Phase 6).
- `assets/sim/predator_prey.sim` — add render/controls/ui + `HareControl` (Phase 7).
- **Delete (Phase 6):** `crates/viewer_runtime/src/bin/vs_viewer.rs`, `vs.rs::VsBridge`, `vs_ui.rs` (and trim `viewer_runtime` to its remaining dungeon path).

## 9. Out of scope (the later directions + per-game limits)

- **Directions #2–#4** (editor/hot-reload, real sprite/mesh rendering + audio, distributable runtime/SDK) — each its own spec, built after #1. The `PlayableRuntime` trait + descriptors are the seam they extend.
- Arbitrary-field render keying (beyond the fixed snapshot columns).
- Rich VFX/animation/particles (that's direction #3); this spec keeps voxel splats + ring/beam VFX.
- Hot-reload of descriptors (direction #2); descriptors are baked at compile time here.
- The dungeon_horde viewer is not migrated in this spec (it stays on its own path; migrating it is optional future cleanup).
