# engine_ui Framework Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reusable, sim-agnostic Rust UI framework — a declarative `UiModel` (HUD widgets + modal screens), a per-frame `UiData` value snapshot, and `UiAction` results — rendered via egui, consumable by any fixture's viewer.

**Architecture:** Pure data + a thin egui adapter. `UiModel` describes *what* to show with string bind-keys; `UiData` is a `name → f32` snapshot the caller fills from sim readback; `draw(ctx, model, data, active_screen)` paints the HUD always and one optional modal screen, returning any `UiAction` the user triggered (card click / restart). The framework owns **no** game state and never reads the sim — the caller resolves bindings and applies actions. This makes everything except the egui paint call unit-testable without a GPU.

**Tech Stack:** Rust, `egui = "0.33"` (voxel_engine pins this; it does NOT re-export egui, so depend on it directly). No GPU in this crate's tests.

**Runs fully in parallel with Plan 1 — zero shared files except an append to the root `Cargo.toml` members list.**

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `voxel_engine::EguiState` at `/home/ricky/Projects/voxel_engine/src/ui/mod.rs:45` (egui-over-Vulkan; consumed by Plan 4, not here)
  - no existing reusable UI model/framework in the workspace (`rg "struct UiModel" crates/` → none)
  - Search method: `rg` + direct `Read`.
- **Decision:** NEW crate `crates/engine_ui` because no reusable UI layer exists; it must be sim-agnostic and depend only on `egui` (Vulkan wiring stays in the viewer via `voxel_engine`).
- **Rule-compiler touchpoints:** none — pure runtime/presentation crate.
- **Hand-written downstream code:** `crates/engine_ui/**` — justified: this is presentation code, not engine-rule behavior; P1 governs sim rules, not UI rendering. (Plan 5 later lets the DSL *declare* a `UiModel`, but the renderer itself is legitimately hand-written Rust.)
- **Constitution check:**
  - P1 (Compiler-First): PASS — not engine-rule behavior; no `impl Rule`, no handler code.
  - P2 / P3 / P4 / P5 / P6 / P7 / P11: N/A — touches no sim state, events, RNG, or kernels.
  - P10 (No Runtime Panic): PASS — headless egui smoke test asserts `draw` does not panic.
  - P8 (AIS Required): PASS — this section.
- **Runtime gate:**
  - `draw_hud_headless_no_panic` at `crates/engine_ui/src/render.rs` (test mod) — running `draw` inside a headless `egui::Context` produces non-empty tessellated output and does not panic.
- **Re-evaluation:** [x] AIS reviewed at design phase.  [ ] AIS reviewed post-design.

---

### Task 1: Crate scaffold + model/data types + pure unit tests

**Files:**
- Create: `crates/engine_ui/Cargo.toml`, `crates/engine_ui/src/lib.rs`, `crates/engine_ui/src/model.rs`, `crates/engine_ui/src/data.rs`
- Modify: root `Cargo.toml` (append `"crates/engine_ui"` to `[workspace] members`)

- [ ] **Step 1: Cargo.toml + workspace member.**
```toml
# crates/engine_ui/Cargo.toml
[package]
name = "engine_ui"
version = "0.1.0"
edition = "2021"

[dependencies]
egui = "0.33"
```
Append `"crates/engine_ui",` to the root `Cargo.toml` `members` list (alphabetical-ish; this is the only shared-file touch — additive line).

- [ ] **Step 2: Model + data types.**
```rust
// crates/engine_ui/src/model.rs
/// A value lookup key into UiData (a named sim-readback value).
pub type BindKey = String;

#[derive(Clone, Debug)]
pub enum Widget {
    /// Horizontal bar: value/max fraction, labelled, RGB color.
    Bar { label: String, value: BindKey, max: BindKey, color: [u8; 3] },
    /// Text with `{key}` placeholders substituted from UiData (formatted as ints).
    Text { template: String },
}

#[derive(Clone, Debug)]
pub enum UiAction {
    /// Increment a named host-side counter (e.g. "bolt_level"). Applied by the caller.
    Increment(String),
    /// Restart the run.
    Restart,
}

#[derive(Clone, Debug)]
pub struct Card { pub label: String, pub action: UiAction }

#[derive(Clone, Debug)]
pub enum Screen {
    /// Modal upgrade menu — pauses; cards are buttons returning their action.
    Menu { title: String, cards: Vec<Card> },
    /// Modal end screen — summary rows (label, BindKey) + a restart button.
    End { title: String, summary: Vec<(String, BindKey)>, restart_label: String },
}

#[derive(Clone, Debug)]
pub struct NamedScreen { pub name: String, pub screen: Screen }

#[derive(Clone, Debug, Default)]
pub struct UiModel { pub hud: Vec<Widget>, pub screens: Vec<NamedScreen> }

impl UiModel {
    pub fn screen(&self, name: &str) -> Option<&Screen> {
        self.screens.iter().find(|s| s.name == name).map(|s| &s.screen)
    }
}
```
```rust
// crates/engine_ui/src/data.rs
use std::collections::HashMap;
#[derive(Clone, Debug, Default)]
pub struct UiData { vals: HashMap<String, f32> }
impl UiData {
    pub fn new() -> Self { Self::default() }
    pub fn set(&mut self, key: &str, v: f32) -> &mut Self { self.vals.insert(key.to_string(), v); self }
    pub fn get(&self, key: &str) -> f32 { self.vals.get(key).copied().unwrap_or(0.0) }
    /// Substitute `{key}` placeholders (rendered as rounded ints) in a template.
    pub fn fill(&self, template: &str) -> String {
        let mut out = String::new();
        let mut rest = template;
        while let Some(open) = rest.find('{') {
            out.push_str(&rest[..open]);
            if let Some(close) = rest[open..].find('}') {
                let key = &rest[open + 1..open + close];
                out.push_str(&format!("{}", self.get(key).round() as i64));
                rest = &rest[open + close + 1..];
            } else { out.push_str(&rest[open..]); rest = ""; }
        }
        out.push_str(rest);
        out
    }
}
```

- [ ] **Step 3: lib.rs re-exports.**
```rust
// crates/engine_ui/src/lib.rs
pub mod model;
pub mod data;
pub mod render;
pub use model::{UiModel, Widget, Screen, NamedScreen, Card, UiAction, BindKey};
pub use data::UiData;
```

- [ ] **Step 4: Pure unit tests for data binding.**
```rust
// in data.rs #[cfg(test)] mod
#[test] fn fill_substitutes_int_keys() {
    let mut d = super::UiData::new(); d.set("level", 3.0).set("kills", 42.7);
    assert_eq!(d.fill("Lv {level}  Kills {kills}"), "Lv 3  Kills 43");
}
#[test] fn fill_missing_key_is_zero() {
    assert_eq!(super::UiData::new().fill("hp {hp}"), "hp 0");
}
```

- [ ] **Step 5: Run + commit.** Run: `cargo test -p engine_ui`. Expected: PASS.
```bash
git add crates/engine_ui/ Cargo.toml
git commit -m "feat(engine_ui): scaffold crate + UiModel/UiData/UiAction + binding tests"
```

### Task 2: egui render adapter (HUD + modal screens)

**Files:**
- Create: `crates/engine_ui/src/render.rs`

- [ ] **Step 1: Implement `draw`.** Paints the HUD always; if `active` names a screen, paints it modally and returns any triggered action.
```rust
// crates/engine_ui/src/render.rs
use crate::{UiModel, UiData, Widget, Screen, UiAction};

/// Draw the HUD + optional modal screen. Returns the action the user
/// triggered this frame (card click / restart), if any. Caller applies it.
pub fn draw(ctx: &egui::Context, model: &UiModel, data: &UiData, active: Option<&str>) -> Option<UiAction> {
    // HUD — top-left, non-interactive overlay.
    egui::Area::new(egui::Id::new("engine_ui_hud"))
        .fixed_pos(egui::pos2(12.0, 12.0))
        .show(ctx, |ui| {
            for w in &model.hud {
                match w {
                    Widget::Text { template } => { ui.label(data.fill(template)); }
                    Widget::Bar { label, value, max, color } => {
                        let v = data.get(value);
                        let m = data.get(max).max(1e-3);
                        let frac = (v / m).clamp(0.0, 1.0);
                        let col = egui::Color32::from_rgb(color[0], color[1], color[2]);
                        ui.horizontal(|ui| {
                            ui.label(label);
                            ui.add(egui::ProgressBar::new(frac).fill(col).desired_width(180.0));
                        });
                    }
                }
            }
        });

    // Modal screen.
    let mut action = None;
    if let Some(name) = active {
        if let Some(screen) = model.screen(name) {
            egui::Window::new("engine_ui_modal")
                .title_bar(false).collapsible(false).resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    match screen {
                        Screen::Menu { title, cards } => {
                            ui.heading(title);
                            for c in cards {
                                if ui.button(&c.label).clicked() { action = Some(c.action.clone()); }
                            }
                        }
                        Screen::End { title, summary, restart_label } => {
                            ui.heading(title);
                            for (label, key) in summary {
                                ui.label(format!("{label}: {}", data.fill(&format!("{{{key}}}"))));
                            }
                            if ui.button(restart_label).clicked() { action = Some(UiAction::Restart); }
                        }
                    }
                });
        }
    }
    action
}
```

- [ ] **Step 2: Headless no-panic smoke test.**
```rust
// in render.rs #[cfg(test)] mod
#[test] fn draw_hud_headless_no_panic() {
    use crate::*;
    let model = UiModel {
        hud: vec![
            Widget::Bar { label: "HP".into(), value: "hp".into(), max: "hp_max".into(), color: [220,40,40] },
            Widget::Text { template: "Lv {level}  Kills {kills}".into() },
        ],
        screens: vec![NamedScreen { name: "level_up".into(), screen: Screen::Menu {
            title: "Level Up".into(),
            cards: vec![Card { label: "Bolt +".into(), action: UiAction::Increment("bolt_level".into()) }],
        }}],
    };
    let mut data = UiData::new();
    data.set("hp", 50.0).set("hp_max", 100.0).set("level", 2.0).set("kills", 7.0);
    let ctx = egui::Context::default();
    let out = ctx.run(egui::RawInput::default(), |ctx| { let _ = draw(ctx, &model, &data, Some("level_up")); });
    assert!(!out.shapes.is_empty(), "draw produced no shapes");
}
```

- [ ] **Step 3: Run + commit.** Run: `cargo test -p engine_ui`. Expected: PASS.
```bash
git add crates/engine_ui/src/render.rs
git commit -m "feat(engine_ui): egui render adapter for HUD bars/text + modal menu/end screens"
```

## Self-review note
`draw` is the only egui-touching surface; everything else (`UiModel`, `UiData::fill`) is pure and unit-tested. Plan 4 wires `draw` into `vs_viewer` via `voxel_engine::EguiState`; Plan 5 makes the compiler emit a `UiModel`. The `UiModel`/`UiData`/`UiAction` shapes are the frozen contract those plans build against.
