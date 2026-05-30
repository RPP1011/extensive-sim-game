# engine_play_api — Contract Crate Implementation Plan (Plan 0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Land the frozen contract crate — the `PlayableRuntime` trait, `AgentView`, and the serde descriptor schemas (`RenderDescriptor`/`ControlsDescriptor`) — plus hand-written sample descriptor JSON, so the compiler (Plan A), generic player (Plan B), and engine_ui (Plan C) build in parallel against it.

**Architecture:** A leaf crate with no heavy deps (only `serde`/`serde_json`). It defines the seam between compiled `.sim` runtimes and the generic player. The UI descriptor reuses `engine_ui::UiModel` (Plan C), so this crate does NOT define UI types — only render + controls + the trait.

**Tech Stack:** Rust, `serde`, `serde_json`.

---

## Architectural Impact Statement

- **Existing primitives searched:** `engine::backend::Backend` trait at `crates/engine/src/backend.rs` (the CPU/GPU seam — `PlayableRuntime` sits beside it as the *authoring/player* seam); `engine_ui::UiModel` (Plan 2). Method: `rg`/`Read`.
- **Decision:** NEW leaf crate `engine_play_api` — needed to break the `sims ↔ engine_play` cycle (sims impls the trait; engine_play consumes it; both depend on this leaf).
- **Rule-compiler touchpoints:** none — pure types crate.
- **Hand-written downstream code:** `engine_play_api/**` — justified: a runtime-seam trait + serde DTOs, not engine-rule behavior (P1 governs rules).
- **Constitution check:** P1 PASS (not rule behavior); P2–P7/P11 N/A; P10 PASS (serde parse is `Result`); P8 PASS (this section).
- **Runtime gate:** `descriptors_roundtrip` + `sample_fixtures_parse` (test mod) — every descriptor serializes→deserializes equal, and the shipped sample JSON parses.
- **Re-evaluation:** [x] design-phase. [ ] post-design.

---

### Task 1: Crate scaffold + trait + AgentView

**Files:** Create `crates/engine_play_api/Cargo.toml`, `crates/engine_play_api/src/lib.rs`; Modify root `Cargo.toml` (append member).

- [ ] **Step 1: Cargo.toml + member.**
```toml
# crates/engine_play_api/Cargo.toml
[package]
name = "engine_play_api"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```
Append `"crates/engine_play_api",` to the root `Cargo.toml` `[workspace] members`.

- [ ] **Step 2: Trait + AgentView in `src/lib.rs`.**
```rust
pub mod render;
pub mod controls;

/// Uniform interface a compiled `.sim` runtime exposes so one binary can play any game.
/// Sits beside `engine::backend::Backend` (CPU/GPU seam) as the authoring/player seam.
pub trait PlayableRuntime {
    fn tick(&self) -> u64;
    fn step(&mut self);
    /// Write an `@runtime` config field by name (dispatches to set_config_<block>_<field>).
    fn set_input(&mut self, field: &str, value: f32);
    fn agent_snapshot(&mut self) -> Vec<AgentView>;
    /// Materialized-view value at an agent slot (e.g. "xp"); 0.0 if unknown.
    fn view_value(&mut self, view: &str, slot: u32) -> f32;
    fn render_descriptor(&self) -> &'static str;
    fn controls_descriptor(&self) -> &'static str;
    fn ui_descriptor(&self) -> &'static str;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgentView {
    pub pos: [f32; 3],
    pub alive: bool,
    pub hp: f32,
    pub mana: f32,
    pub move_speed: f32,
    pub creature_type: u32,
}
```

- [ ] **Step 3:** `cargo build -p engine_play_api` → expect a "file not found: render/controls" error (modules not yet created). Proceed to Task 2.

### Task 2: Render + controls descriptor schemas (serde)

**Files:** Create `crates/engine_play_api/src/render.rs`, `crates/engine_play_api/src/controls.rs`.

- [ ] **Step 1: `render.rs`.**
```rust
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FieldRange { pub field: String, pub lo: f32, pub hi: f32 }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CameraSpec { Follow(FieldRange), Observer }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentVisual { pub when: FieldRange, pub color: [u8; 3] }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VfxKind { Ring, BeamToNearest { target: FieldRange } }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VfxSpec { pub on_rule: String, pub period: u32, pub kind: VfxKind, pub radius: f32, pub color: [u8; 3] }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RenderDescriptor {
    pub arena_radius: f32,
    pub camera: CameraSpec,
    pub agents: Vec<AgentVisual>,
    pub vfx: Vec<VfxSpec>,
}
impl RenderDescriptor {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
}
```

- [ ] **Step 2: `controls.rs`.**
```rust
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum BindMode { Hold, Press }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KeyBinding { pub key: String, pub field: String, pub value: f32, pub mode: BindMode }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ControlsDescriptor { pub bindings: Vec<KeyBinding> }
impl ControlsDescriptor {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
}
```

- [ ] **Step 3: re-export in `lib.rs`** — add `pub use render::*; pub use controls::*;` after the `pub mod` lines.

- [ ] **Step 4: round-trip test** (in `lib.rs` `#[cfg(test)]`):
```rust
#[test] fn descriptors_roundtrip() {
    let r = render::RenderDescriptor {
        arena_radius: 42.0,
        camera: render::CameraSpec::Follow(render::FieldRange{ field:"mana".into(), lo:0.5, hi:1.5 }),
        agents: vec![render::AgentVisual{ when: render::FieldRange{field:"mana".into(),lo:0.5,hi:1.5}, color:[0,220,220] }],
        vfx: vec![render::VfxSpec{ on_rule:"NovaFire".into(), period:40, kind:render::VfxKind::Ring, radius:6.0, color:[255,255,120] }],
    };
    let j = serde_json::to_string(&r).unwrap();
    assert_eq!(render::RenderDescriptor::from_json(&j).unwrap(), r);
}
```
Run `cargo test -p engine_play_api`. Expect PASS. Commit.
```bash
git add crates/engine_play_api Cargo.toml
git commit -m "feat(engine_play_api): PlayableRuntime trait + render/controls descriptor schemas"
```

### Task 3: Sample descriptor fixtures (so Plan B builds before Plan A)

**Files:** Create `crates/engine_play_api/fixtures/vs_render.json`, `vs_controls.json`; Test in `lib.rs`.

- [ ] **Step 1: `fixtures/vs_render.json`** — a hand-written VS render descriptor (arena 42; player cyan by mana [0.5,1.5]; enemy orange [1.5,2.5]; nova ring vfx; bolt beam vfx):
```json
{ "arena_radius": 42.0,
  "camera": { "Follow": { "field": "mana", "lo": 0.5, "hi": 1.5 } },
  "agents": [
    { "when": { "field": "mana", "lo": 0.5, "hi": 1.5 }, "color": [0,220,220] },
    { "when": { "field": "mana", "lo": 1.5, "hi": 2.5 }, "color": [230,100,20] } ],
  "vfx": [
    { "on_rule": "NovaFire", "period": 40, "kind": "Ring", "radius": 6.0, "color": [255,255,120] },
    { "on_rule": "BoltFire", "period": 12, "kind": { "BeamToNearest": { "target": { "field":"mana","lo":1.5,"hi":2.5 } } }, "radius": 0.0, "color": [200,255,255] } ] }
```

- [ ] **Step 2: `fixtures/vs_controls.json`.**
```json
{ "bindings": [
  { "key": "w", "field": "ctl.move_y", "value":  1.0, "mode": "Hold" },
  { "key": "s", "field": "ctl.move_y", "value": -1.0, "mode": "Hold" },
  { "key": "d", "field": "ctl.move_x", "value":  1.0, "mode": "Hold" },
  { "key": "a", "field": "ctl.move_x", "value": -1.0, "mode": "Hold" } ] }
```

- [ ] **Step 3: fixtures-parse test.**
```rust
#[test] fn sample_fixtures_parse() {
    let r = include_str!("../fixtures/vs_render.json");
    let c = include_str!("../fixtures/vs_controls.json");
    assert_eq!(render::RenderDescriptor::from_json(r).unwrap().agents.len(), 2);
    assert_eq!(controls::ControlsDescriptor::from_json(c).unwrap().bindings.len(), 4);
}
```
Run `cargo test -p engine_play_api`. Expect PASS. Commit.
```bash
git add crates/engine_play_api/fixtures crates/engine_play_api/src/lib.rs
git commit -m "feat(engine_play_api): sample VS descriptor fixtures for parallel player dev"
```

## Self-review note
This crate is the frozen contract. Plan A emits JSON matching these exact serde shapes (externally-tagged enums: `{"Follow":{...}}`, `"Observer"`, `"Ring"`, `{"BeamToNearest":{...}}`, `"Hold"`). Plan B builds the player against the sample fixtures + a mock `PlayableRuntime`. Do not change these shapes after Wave 1 starts without updating all three tracks.
