# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this crate is

`engine_play_api` is the frozen contract between a compiled `.sim` runtime and the generic player — a leaf crate with no heavy deps (`serde`/`serde_json` only) so both sides can depend on it without a cycle. It defines:
- the `PlayableRuntime` trait — the uniform interface any compiled `.sim` runtime exposes so one binary can play any game (sits beside `engine::backend::Backend`, the CPU/GPU seam, as the *authoring/player* seam);
- `AgentView` — the live-agent snapshot shape the renderer reads;
- `RenderDescriptor`/`ControlsDescriptor` (and their nested types) — serde DTOs for "how agents map to visuals" and "how keys map to `@runtime` field writes," each with a `from_json` parse entry point.

It does **not** define its own UI descriptor type. `PlayableRuntime::ui_descriptor()` returns a bare `&'static str` of JSON; the doc comment in `src/lib.rs` notes this reuses `engine_ui::UiModel`'s shape — but that's a JSON-shape convention only, not a Rust dependency. Check: `crates/engine_play_api/Cargo.toml` has no `engine_ui` entry. Callers (`engine_play`) independently choose to parse the string with `engine_ui::UiModel::from_json`.

Three source files:
- `src/lib.rs` — `PlayableRuntime` trait, `AgentView`, re-exports.
- `src/render.rs` — `FieldRange`, `CameraSpec`, `AgentVisual`, `VfxKind`, `VfxSpec`, `RenderDescriptor`.
- `src/controls.rs` — `BindMode`, `KeyBinding`, `ControlsDescriptor`.
- `fixtures/vs_render.json`, `fixtures/vs_controls.json` — hand-written sample descriptors (from the original Vampire-Survivors-style fixture) that exercise `from_json` in tests and let dependent crates build/test before a real compiler emitter exists.

## Commands

```bash
cargo build -p engine_play_api
cargo test -p engine_play_api   # descriptors_roundtrip + sample_fixtures_parse
```

## Architecture: who depends on this, and why the shape is here

This is the seam that breaks what would otherwise be a `sims ↔ engine_play` cycle:

```
engine_play_api (this crate — trait + descriptor DTOs, no deps on either side)
      ↑                              ↑
    sims                        engine_play
(impls PlayableRuntime      (consumes the trait via
 for GeneratedRuntime)       sims::make_playable)
```

- `crates/sims` depends on it to `impl engine_play_api::PlayableRuntime for GeneratedRuntime` (every compiled `.sim` fixture) and to expose `make_playable()`/`PLAYABLE_FIXTURES`.
- `crates/engine_play` depends on it as the only runtime-facing type surface the generic player (`play` binary) consumes — it never defines its own runtime types, only calls through this trait.
- `crates/dsl_compiler` depends on it too (compile-time emitter target: the compiler's hand-written JSON for `render_descriptor()`/`controls_descriptor()` must match these serde shapes exactly).

## Non-obvious things

- **The serde shapes are frozen once dependents build against them.** Enums are the default *externally-tagged* form (`{"Follow":{...}}`, `"Observer"`, `{"BeamToNearest":{...}}`, `"Hold"`) — the compiler hand-emits JSON in this exact shape rather than deriving it, so a derive-attribute change here (e.g. adding `#[serde(tag=...)]`) silently desyncs the emitter without a compile error. Per the plan's self-review note (`docs/superpowers/plans/2026-05-25-engine-play-api-impl.md`): don't change these shapes without updating the compiler emitter, `engine_play`, and `engine_ui` in lockstep.
- **`view_text` is a defaulted trait method, added later (S13) alongside `engine_ui::UiData`'s text channel.** It returns `None` by default so every existing `PlayableRuntime` implementor — including every generated `.sim` runtime — is source-compatible without changes; only a host carrying real prose (a petition, a colonist's name) needs to override it.
- **`fixtures/*.json` exist so dependent crates can build in parallel without a real compiler emitter.** They were hand-written against a specific (now-migrated) Vampire-Survivors-style fixture's shape, not generated — don't expect them to track the current state of any specific `.sim` file in `assets/sim/`; they're pinned test fixtures, not living samples.
- **This crate has zero knowledge of `egui`, `wgpu`, `voxel_engine`, or the DSL/compiler pipeline** — by design, per its own doc comment ("leaf crate, no heavy deps"). If a change here would require pulling in one of those, it almost certainly belongs in `engine_play` or `sims` instead.
