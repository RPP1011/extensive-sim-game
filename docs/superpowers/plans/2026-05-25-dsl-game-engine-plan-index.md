# DSL Game Engine #1 — Plan Index & Parallelization DAG

> Spec: `docs/superpowers/specs/2026-05-25-dsl-game-engine-author-any-game-design.md`.
> Decomposes the spec into **seven independently-landable plans** structured for **maximum parallelism**. The unlock is a **frozen contract** (the `engine_play_api` crate) landed first; then three tracks build in parallel against it on disjoint files. Each plan is its own file with an AIS preamble (P8).

## The frozen contract (defines all parallelism)

Everything keys off `crates/engine_play_api` (Plan 0) — the trait + descriptor schemas the compiler emits to and the player consumes. Frozen here so the compiler side (Plan A), player side (Plan B), and UI side (Plan C) build in parallel without sharing files:

```rust
// crates/engine_play_api/src/lib.rs
pub trait PlayableRuntime {
    fn tick(&self) -> u64;
    fn step(&mut self);                                   // includes the summon drain
    fn set_input(&mut self, field: &str, value: f32);     // dispatch by @runtime field name
    fn agent_snapshot(&mut self) -> Vec<AgentView>;
    fn view_value(&mut self, view: &str, slot: u32) -> f32;
    fn render_descriptor(&self) -> &'static str;
    fn controls_descriptor(&self) -> &'static str;
    fn ui_descriptor(&self) -> &'static str;
}
#[derive(Clone, Copy, Debug)]
pub struct AgentView { pub pos: [f32; 3], pub alive: bool, pub hp: f32, pub mana: f32, pub move_speed: f32, pub creature_type: u32 }

// Descriptor schemas (serde Serialize+Deserialize). Compiler emits JSON matching these; player parses them.
pub struct RenderDescriptor { pub arena_radius: f32, pub camera: CameraSpec, pub agents: Vec<AgentVisual>, pub vfx: Vec<VfxSpec> }
pub struct FieldRange { pub field: String, pub lo: f32, pub hi: f32 } // field ∈ {"mana","hp","move_speed","creature_type"}
pub enum CameraSpec { Follow(FieldRange), Observer }
pub struct AgentVisual { pub when: FieldRange, pub color: [u8; 3] }
pub struct VfxSpec { pub on_rule: String, pub period: u32, pub kind: VfxKind, pub radius: f32, pub color: [u8; 3] }
pub enum VfxKind { Ring, BeamToNearest { target: FieldRange } }
pub struct ControlsDescriptor { pub bindings: Vec<KeyBinding> }
pub struct KeyBinding { pub key: String, pub field: String, pub value: f32, pub mode: BindMode } // mode normalizes diagonals
pub enum BindMode { Hold, Press }
// UI descriptor reuses engine_ui::UiModel (Plan C adds from_json).
```

Plan 0 also ships **hand-written sample descriptor JSON** for vampire_survivors (`crates/engine_play_api/fixtures/vs_render.json`, `vs_controls.json`) so Plan B can build + test the generic player before the compiler (Plan A) emits anything.

## Plans & file ownership

| # | Plan file | Depends on | Owns (files) |
|---|-----------|-----------|--------------|
| 0 | `…-engine-play-api-impl.md` | — | `crates/engine_play_api/**` (new); root `Cargo.toml` members |
| A | `…-engine-compiler-emit-impl.md` | 0 | `dsl_ast/src/{ast,parser}.rs`; `dsl_compiler/src/cg/emit/{render,controls,ui,playable}.rs` + `build_helper.rs`; `sims` regen |
| B | `…-engine-generic-player-impl.md` | 0 | `crates/engine_play/**` (new) — bridge/input/loop/bin |
| C | `…-engine-ui-descriptor-impl.md` | 0 | `crates/engine_ui/{Cargo.toml,src/model.rs,src/data.rs}` (serde + from_json) |
| D | `…-engine-integration-impl.md` | A,B,C | `crates/engine_play/src/registry-wire` + `sims::make_playable` consume; `play` end-to-end on VS |
| E | `…-engine-migrate-vs-impl.md` | D | `assets/sim/vampire_survivors.sim` (+blocks); **delete** `viewer_runtime/src/{bin/vs_viewer.rs,vs.rs::VsBridge,vs_ui.rs}` |
| F | `…-engine-predator-prey-impl.md` | D | `assets/sim/predator_prey.sim` (+blocks +HareControl) |

## Execution waves

```
Wave 0  (1 plan — the frozen contract):
  └── Plan 0: engine_play_api (trait + descriptor schemas + sample JSON fixtures)

Wave 1  (3 plans, FULLY PARALLEL — disjoint files, all build against the Wave-0 contract):
  ├── Plan A: compiler emit  [dsl_ast + dsl_compiler + sims]   — emits descriptors + PlayableRuntime impl + registry
  ├── Plan B: generic player [engine_play new crate]           — bridge/input/loop/bin vs SAMPLE json + a mock runtime
  └── Plan C: engine_ui      [engine_ui crate]                 — UiModel serde + from_json

Wave 2  (integration, on-branch):
  ├── Plan D: integration — wire real descriptors + registry; `play vampire_survivors` end-to-end  [needs A+B+C]
  └── then Plan E ∥ Plan F (parallel — different .sim files):
        ├── Plan E: migrate VS to the generic path + DELETE vs_* glue
        └── Plan F: convert predator_prey (play as Hare), zero new Rust  (the generality proof)
```

**Merge-tax guard:** the three DSL blocks (render/controls/ui) all touch `dsl_ast` parser/AST + `dsl_compiler` emit — so they are **all in Plan A** (one plan, internally serial), never raced across plans. Plans B and C touch entirely separate crates. Plan A↔B↔C share zero files.

## Execution note (the worktree-isolation gotcha)

Agent `isolation:"worktree"` branches from **origin/main**, not this feature branch — and `engine_ui` + the `config_ctl` setters exist **only on this branch**. So Plans B (depends on engine_play_api) and C (edits engine_ui) **cannot** run as origin/main-isolated agents until the branch is on main. Two execution options:
1. **Merge the current branch to origin/main first** (the playable-VS work is complete + tested) → isolated agents branch from a base that has everything → run Wave 1's three plans as parallel isolated agents cleanly.
2. **Run on-branch:** Plan 0 inline, then Wave 1 via manual worktrees created from the branch HEAD (`git worktree add -b <track> <path> HEAD`) or sequentially. The plan *structure* is parallel regardless; only the agent-dispatch mechanism is constrained.

## Done = all of
- `cargo build` clean; `cargo test -p engine_play_api -p engine_play -p engine_ui -p dsl_compiler -p sims` green.
- `cargo run -p engine_play --bin play vampire_survivors` and `… play predator_prey` both open playable windows driven entirely by `.sim` `render{}`/`controls{}`/`ui{}` blocks.
- `vs_viewer.rs` / `VsBridge` / `vs_ui.rs` deleted; no per-game Rust remains for either game.
