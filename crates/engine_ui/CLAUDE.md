# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this crate is

`engine_ui` is a small, sim-agnostic egui presentation framework: a declarative `UiModel` (HUD widgets + modal menu/end screens), a per-frame `UiData` value snapshot, and a `draw()` adapter that paints them. It owns no game state and never reads a sim directly — the caller (`engine_play`, or a legacy `*_runtime` viewer) resolves bind-keys into `UiData` and applies any `UiAction` the player triggered.

`UiModel` doubles as the deserialization target for the compiler-emitted `ui_descriptor()` JSON: a `ui {}` block in a `.sim` file lowers to JSON (`dsl_compiler::cg::emit::ui_model`, called from `build_helper.rs`) that `UiModel::from_json` parses back at runtime. So this crate's serde shapes are a **frozen contract with the compiler**, not just an internal representation — see the doc comment atop `src/model.rs`.

Four files total:
- `src/lib.rs` — module wiring + re-exports.
- `src/model.rs` — `UiModel`/`Widget`/`Screen`/`NamedScreen`/`Card`/`UiAction`/`BindKey`, all `Serialize`/`Deserialize`, plus `UiModel::from_json`.
- `src/data.rs` — `UiData`: a `name → f32` map (`vals`) plus a `name → String` map (`texts`) that the caller fills from sim readback each frame, and `fill()` for `{key}` template substitution.
- `src/render.rs` — `draw(ctx, model, data, active_screen) -> Option<UiAction>`, the only egui-touching surface in the crate.

## Commands

```bash
cargo build -p engine_ui
cargo test -p engine_ui     # pure model/data unit tests + one headless egui smoke test
```

## Architecture: who consumes this

`engine_ui` has no dependency on `engine_play_api` or `sims` — it's a leaf presentation crate depended on directly by:
- `crates/dsl_compiler` — compile-time only, to keep the compiler's hand-emitted `ui_model` JSON shape in lockstep with these serde derives (not a runtime dependency).
- `crates/engine_play` — the generic player; `player.rs` builds a `UiData` each frame and calls `engine_ui::draw` for the HUD/menu/death overlay.
- `crates/viewer_runtime` — the legacy per-fixture viewer (`vs_ui.rs`, `bin/vs_viewer.rs`) uses it the same way, pre-dating `engine_play`.
- `crates/sims` — **dev-dependency only** (see its `Cargo.toml` comment): the `playable_registry` test parses a generated fixture's emitted `ui_descriptor()` through `UiModel::from_json` to verify round-tripping. The `sims` lib itself has zero UI/egui dependency; this is test-only.

`engine_play_api` does **not** depend on this crate in Rust (`crates/engine_play_api/Cargo.toml` has no `engine_ui` entry) despite its `src/lib.rs` doc comment saying "The UI descriptor reuses `engine_ui::UiModel`" — the reuse is at the **JSON-shape** level only: `PlayableRuntime::ui_descriptor()` returns a `&'static str`, and callers (`engine_play`) independently choose to parse it with `engine_ui::UiModel::from_json`. There's no compiled coupling between the two crates.

## Non-obvious things

- **The serde enum representation is load-bearing.** All enums (`Widget`, `UiAction`, `Screen`) use serde's default *externally-tagged* form (`{"Bar":{...}}`, `"Restart"`, `{"Increment":"bolt_level"}`). `dsl_compiler::cg::emit::ui_model` hand-writes JSON in exactly this shape rather than deriving it — changing the derive attributes here (e.g. adding `#[serde(tag = "type")]`) silently breaks the compiler's emitted JSON without a compile error; only `from_json_roundtrip` (this crate) and the compiler's own emit tests would catch it.
- **`UiData` has two parallel channels, not one.** It started as `f32`-only (`vals`); `texts` (a `name → String` map) was added later so a HUD can print words, not just numbers (a day name, a colonist's name, a petition's ask) — see the doc comment in `data.rs`. In `fill()`, a `texts` entry for a key wins over `vals` for that same key. `get()` still only reads `vals` (returns `0.0` for a text-only key) — use `get_text()` for the string channel.
- **`draw()` renders the HUD unconditionally and at most one modal screen.** The modal (`active: Option<&str>`) is looked up by name via `UiModel::screen()`; an unknown name is silently a no-op (no screen drawn), not an error — `UiModel::from_json` is the fallible boundary (P10), not screen lookup.
- **No GPU needed for most of this crate.** Only `render.rs`'s `draw_hud_headless_no_panic` test touches egui, and it does so through a headless `egui::Context` (no window/Vulkan surface) — `model.rs`/`data.rs` tests are plain Rust unit tests.
