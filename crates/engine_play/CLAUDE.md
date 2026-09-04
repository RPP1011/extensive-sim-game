# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this crate is

`engine_play` is the generic, descriptor-driven windowed player: one binary (`play`) that can play *any* compiled `.sim` fixture. It hosts a running sim in an interactive `winit` window, rendering the world through `voxel_engine`'s Vulkan renderer with an `egui` HUD/menu overlay on top. It replaces the old per-game `vs_*` viewer files — `bridge.rs`'s `EngineBridge` is explicitly "a generalized port of `viewer_runtime::vs::VsBridge`."

Everything a fixture needs to be playable — how agents map to voxel colors/VFX (`RenderDescriptor`), how keys map to `@runtime` input writes (`ControlsDescriptor`), and the HUD/menu/death-screen layout (`engine_ui::UiModel`) — is consumed entirely through the `engine_play_api::PlayableRuntime` trait. This crate has zero dependency on the DSL compiler emitting anything; it's tested against a headless `MockRuntime` (`src/mock.rs`) plus `engine_play_api`'s sample fixture JSON.

Modules (only 6 files total):
- `src/lib.rs` — module wiring + re-exports.
- `src/bridge.rs` — `EngineBridge`: descriptor-driven flat-arena voxel renderer. CPU painting logic (`EngineBridge::paint`) is factored out behind a `PaintGrid` trait so it's headlessly unit-testable; GPU resources (`bridge_gpu` submodule) only compile in when the bridge is built with a `VulkanContext`.
- `src/input.rs` — `ControlsMapper`: pure keys+descriptor → `(field, value)` writes. No window/GPU dependency.
- `src/player.rs` — the per-frame `update()` function (controls → `set_input` → `step` → snapshot → paint → HUD `UiData` → `engine_ui::draw`), deliberately separable from the winit/GPU shell so it's testable headlessly; and the windowed `Player` (`winit::ApplicationHandler`) that wraps `update` with the Vulkan/egui surface.
- `src/mock.rs` — `MockRuntime`, a headless in-memory `PlayableRuntime` impl for tests (fixed 2-agent roster, records `set_input` writes).
- `src/bin/play.rs` — the `play` binary entry point.

## Commands

```bash
cargo build -p engine_play
cargo test -p engine_play          # headless: bridge/input/player `update()` tests + play_probe_end_to_end
```

Running the binary (requires a display — the windowed path creates a Vulkan surface and will fail to construct one on a headless host):

```bash
cargo run -p engine_play --bin play <fixture> [seed] [agents]   # e.g. `play play_probe`
cargo run -p engine_play --bin play -- --mock                   # MockRuntime fallback, no fixture/GPU-adapter construction needed
```

With no fixture argument, `play` prints the known fixtures from `sims::PLAYABLE_FIXTURES` and exits 2. `make_playable` returning `None` (unknown name, or GPU adapter unavailable) also prints that list and exits 2.

## Architecture: relation to `engine_play_api` and `sims`

Dependency direction is acyclic and load-bearing — don't introduce a cycle:

```
engine_play_api (the frozen trait + descriptor DTOs, no deps on either side)
      ↑
    sims (compiled-.sim registry; implements PlayableRuntime; depends on the contract)
      ↑
engine_play (this crate; the binary depends on the registry)
```

- `engine_play_api` (`crates/engine_play_api/src/lib.rs`) defines `PlayableRuntime` (`tick`, `step`, `set_input`, `agent_snapshot`, `view_value`, `view_text`, `render_descriptor`/`controls_descriptor`/`ui_descriptor`), `AgentView`, and the `RenderDescriptor`/`ControlsDescriptor` serde schemas. This crate never defines its own runtime-facing types — it only consumes that trait.
- `sims::make_playable(name, seed, agents)` is the real registry `bin/play.rs` calls to construct a `Box<dyn PlayableRuntime>` for a named compiled fixture; `sims::PLAYABLE_FIXTURES` lists what's registered.
- `engine_ui::UiModel`/`UiData`/`UiAction` (a separate crate, Plan C) is what `player.rs` drives for the HUD/menu/death overlay; the `ui_descriptor()` JSON a runtime serves is parsed via `UiModel::from_json`.
- Full history/rationale for this split lives in `docs/superpowers/plans/2026-05-25-engine-play-api-impl.md` (the contract-crate plan, Plan 0) and `docs/superpowers/plans/2026-05-25-engine-integration-impl.md` (Plan D — wiring the real `sims` registry into `play`, replacing `MockRuntime`, and the `play_probe` end-to-end fixture/test). Don't restate their content here; read them for design intent if touching the registry wiring or the probe fixture.

## Non-obvious things

- **`voxel_engine` is a git path-shaped dependency, not local.** Despite the root `Cargo.toml` comment ("voxel_engine path dep with `app-harness` enabled") this crate's own `Cargo.toml` pulls it via `git = "https://github.com/RPP1011/voxel_engine", rev = "..."` with `features = ["app-harness"]` (winit + egui support baked into the renderer crate). The `egui`/`winit`/`ash`/`glam` versions in this crate's `[dependencies]` are pinned to match what `voxel_engine`'s `app-harness` feature pulls in — bumping one without checking `voxel_engine`'s lockstep versions will break the build. `viewer_runtime` depends on the same `voxel_engine` rev so cargo unifies the build; keep both crates' revs in sync if bumping.
- **`update()` is split from the windowed shell on purpose.** All per-frame logic (controls resolution, `set_input`, `step`, snapshot, level/death-modal edge detection, HUD `UiData`) lives in the free function `player::update`, taking a `&mut dyn PlayableRuntime` and a `&mut dyn PaintGrid` — no `winit`/Vulkan types in its signature. The windowed `Player::redraw` calls `update` with a throwaway `Painted` sink for the CPU logic, then separately re-paints the real GPU grid via `EngineBridge::refresh` from the post-step snapshot, to avoid double-stepping the sim. Prefer testing new per-frame behavior through `update()` headlessly rather than through the windowed shell.
- **`Observer` vs `Follow` camera changes HUD/modal behavior.** `RenderDescriptor::camera` being `CameraSpec::Observer` (no player agent — a colony/ecology/crowd sim) suppresses the level-up/death modal machine entirely (`update`'s `has_player` gate) and falls back to `player::observer_ui_model()` (just `tick`/`alive`/`agents` — the only generic numbers available) instead of `mock_ui_model()`'s HP/XP-bar HUD, which would otherwise lie for a fixture with no player. This substitution also happens in `bin/play.rs` whenever a fixture's `ui_descriptor()` comes back empty (`{"hud":[],"screens":[]}`, the compiler's default when no `ui {}` block is authored).
- **`PlayerConfig::hud_views`/`hud_texts`** are the fixture-agnostic seam for putting real per-fixture numbers/prose on the HUD: each name is queried once per frame via `PlayableRuntime::view_value`/`view_text` and published into `UiData` under the same name, so a `ui {}` block (or a host like a future `webband_play`) can reference `{wb_day}` etc. Both are empty by default (no behavior change for existing callers).
- **egui HUD action is captured twice per windowed frame** — once from a throwaway headless `egui::Context` inside `update()` (for the action, ignored for painting) and once from the real visible `EguiState::run` in `Player::redraw` — with the visible one's `UiAction` taking priority (`ui_action.or(out.action)`). This is intentional (avoids needing a visible surface for the headless `update()` tests) but means UI action logic must not assume it only fires once per frame.
- **`--mock` exists as a windowless-independent smoke path** that sidesteps the `sims` registry (and thus doesn't need a GPU adapter to construct the runtime) — useful for verifying the player shell itself when a fixture/GPU isn't available, distinct from the headless `cargo test` path which exercises `update()` without any window at all.
