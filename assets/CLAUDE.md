# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

Read the workspace-root `F:\Game\extensive-sim-game\CLAUDE.md` first — it already covers `assets/sim/*.sim` auto-discovery via `crates/sims/build.rs`. This file adds depth specific to `assets/`.

## What lives here

- **`assets/sim/`** — 107 `.sim` files (no `rules/` subdirectory currently exists on disk, despite `docs/game/overview.md`'s layer diagram mentioning `assets/sim/rules/*.sim` as a hypothetical path). This is the DSL source-of-truth for every compiled sim fixture in the workspace — per `docs/game/overview.md`, "DSL source files describing this specific sim." Nothing under `crates/` hand-writes game logic; every cascade handler, mask, scoring row, and entity spawn template is compiler output from one of these files (`docs/game/compiler_progress.md`, `docs/game/feature_flow.md`).
- **`assets/ability_test/`** — one directory per fixture stem, holding that fixture's `.ability` corpus (the ability-DSL, a separate mini-language from the `.sim` physics/mask/scoring DSL). See §5.
- **`assets/config/default.toml`** — hand-maintained balance tunables, loaded via `Config::from_toml` / defaulted via `Config::default()`. Milestone 13 in `docs/game/compiler_progress.md` moved 16 balance consts out of `step.rs` / `mask.rs` / `ability/expire.rs` / `channel.rs` into this file. See §4.
- **`assets/models/`** — the only non-DSL, non-config content here: `kenney_mini-characters` glTF character meshes (27 `.glb` files under `Models/GLB format/`, plus a `Textures/` dir and the pack's own `License.txt`/`Overview.html`), consumed by `crates/viewer_runtime` for rendering agents.

## `.sim` filename to consumer mapping

The root CLAUDE.md and `crates/sims/CLAUDE.md` already cover the `crates/sims/build.rs` allowlist (a `matches!(stem.as_str(), "a" | "b" | ...)` gate — a `.sim` file on disk is not automatically compiled just by existing). As of this writing, 7 stems exist under `assets/sim/` but are **not** in that allowlist and so never become `sims::<stem>::GeneratedRuntime`:

```
crowd_navigation_min   event_kind_filter_probe   particle_collision_min
per_agent_event_scan_probe   per_entity_ring_struct_probe   predator_prey_min
spatial_probe
```

Of these, `per_agent_event_scan_probe` and `per_entity_ring_struct_probe` aren't consumed by any runtime crate at all — they exist purely as fixtures for `crates/dsl_compiler`'s own lowering/golden tests (e.g. `crates/dsl_compiler/tests/per_agent_event_scan_probe_lower.rs`, `.../per_entity_ring_struct_probe_lower.rs`). The other five are referenced from `crates/dsl_compiler/tests/stress_fixtures_compile.rs` (parse/compile-only smoke coverage) and/or superseded by a listed sibling (e.g. `predator_prey_min` vs. the allow-listed `predator_prey` / `predator_prey_real`).

Two crates still compile a `.sim` themselves outside the `sims` mega-crate, per the workspace root's "legacy `crates/*_runtime`" note:

- **`crates/tom_probe_runtime`** has its own `build.rs`, calling `dsl_compiler::build_helper::emit("tom_probe")` directly — it compiles `assets/sim/tom_probe.sim` independently of (and in addition to) `sims`, which *also* has `"tom_probe"` in its allowlist. Don't confuse `crates/sims/tests/tom_probe_*_pin.rs` (ported off this crate, lives in `sims`) with `tom_probe_runtime` itself still being live.
- **`crates/viewer_runtime`** has **no `build.rs`** as of this writing — it does not compile a `.sim` on its own. It depends on the `sims` crate and drives `sims::dungeon_horde::GeneratedRuntime` (see the crate-level doc comment in `crates/viewer_runtime/Cargo.toml`), so `dungeon_horde.sim` reaches it indirectly through the mega-crate. `viewer_runtime`'s direct asset dependency is `assets/models/` (glTF meshes via the `gltf` crate), not a `.sim` file.

## Naming conventions across `assets/sim/*.sim`

No enforced schema, but clear patterns recur:

- **`_probe` suffix** — a small, single-feature isolation fixture proving one compiler capability works (`cooldown_probe.sim`, `spatial_probe.sim`, `belief_smoke_probe.sim`, `threats_view_probe.sim`). These are compiler/engine regression fixtures more than "game content."
- **`_smoke` suffix** — minimal compiles-and-runs sanity check (`apply_ability_smoke.sim`, `param_rule_smoke.sim`).
- **`_min` / `_real` pairs** — a stripped-down variant vs. a fuller one of the same scenario (`predator_prey_min.sim` / `predator_prey_real.sim` / plain `predator_prey.sim`; `particle_collision_min.sim`; `crowd_navigation_min.sim`; `foraging_real.sim`; `trade_market_probe.sim` / `trade_market_real.sim`; `quest_probe.sim` / `quest_arc_real.sim`).
- **Scale suffixes** — numeric team/agent-count in the name signals stress scale, not just gameplay shape: `duel_1v1.sim` vs `duel_25v25.sim`, `tactical_squad_5v5.sim` vs `mass_battle_100v100.sim` vs `objective_capture_10v10.sim`, `megaswarm_1000.sim` vs `megaswarm_10000.sim`.
- **`stress_*` / `*_stresstest` naming** — deliberate perf/throughput fixtures rather than gameplay fixtures (`stress_agent_count.sim`, `stress_cast_density.sim`, `threat_stresstest.sim`, `threat_horizon_stresstest.sim`, `threat_scoring_stresstest.sim`, `swarm_event_storm.sim`).
- **Genre/theme clustering by name prefix** — dungeon-crawl family (`dungeon_crawl`, `dungeon_horde`, `dungeon_layout`, `dungeon_stealth`), maze family (`maze_explorer`, `maze_explorer_belief_smart`, `maze_explorer_multi`, `maze_explorer_smart`, `maze_explorer_visited` — each isolating one AI capability layered onto the same base maze), trade/economy family (`bartering`, `auction_market`, `trade_caravans`, `trade_market_probe`, `trade_market_real`, `village_economy`), social/political family (`diplomacy_probe`, `palace_coup`, `spy_network`, `detective_investigation`, `assassination_threat_test`), belief-system family (`belief_key_typed_probe`, `belief_merge_ops_probe`, `belief_merge_propagation_probe`, `belief_smoke_probe`).
- A one- or two-line header comment block at the top of nearly every file states the fixture's purpose, what compiler surfaces it exercises, and often cites the commit(s) that landed the feature it's proving out (see `duel_1v1.sim`, `diplomacy_probe.sim`, `dsl_stress_coverage.sim` for representative examples) — read that comment before the body; it's usually more informative than the DSL itself for "why does this fixture exist."

## `assets/config/default.toml`

Five `[block]` sections — `[belief]`, `[combat]`, `[communication]`, `[movement]`, `[needs]` — each backing one Rust struct in `crates/engine_data/src/config/` (`BeliefConfig`, `CombatConfig`, `CommunicationConfig`, `MovementConfig`, `NeedsConfig`), aggregated into `engine_data::config::Config` and embedded in `SimState` as `state.config.<block>.<field>`. `Config::from_toml(path)` reads + `toml::from_str`s a file at this shape (missing fields fall back to `#[serde(default)]` per-block); `Config::default()` bakes in the DSL's `= <default>` clauses directly (no file I/O) and is what most tests use today.

The file's own header is load-bearing: it was originally compiler-emitted (`xtask compile-dsl`, retired Phase 7 / 2026-05-02) but is now **hand-maintained**. Editing *values* here is the normal way to tune balance. Do **not** add, remove, or rename fields/sections — that changes the config schema and bumps `CONFIG_HASH` (`crates/engine_data/src/schema.rs`), which is a breaking change requiring compiler-side updates in lockstep, not just a TOML edit.

## `assets/ability_test/`

This is the corpus for the **ability DSL** — a separate, smaller language from the `.sim` physics/mask/scoring/entity DSL, parsed by the same `crates/dsl_compiler` (see `dsl_ast::AbilityFile`) but with its own grammar for `ability <Name> { target: ...; range: ...; cooldown: ...; hint: ...; damage ... }`-shaped declarations (e.g. `assets/ability_test/among_us/Kill.ability`).

**The pairing convention is directory-name == fixture stem.** `crates/dsl_compiler/src/build_helper.rs` auto-detects a companion corpus by checking whether `assets/ability_test/<fixture_name>/` exists (`workspace_root.join("assets/ability_test").join(fixture_name)`); if it does, every `.ability` file in it is `include_str!`'d at compile time (path resolved via `concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/ability_test/<fixture>/<file>")`) and a real `AbilityRegistryBuilder` registry is constructed inside the generated `try_new()` — no runtime file I/O, but real ability parsing happens on first call. No corpus directory → the generated code falls back to a single no-op placeholder program (wgpu rejects zero-sized bindings). This is true whether the caller is the `sims` mega-crate or a standalone `*_runtime` crate.

Two directories are **stale, not migrated**: `webband_bench/` and `webband_bench_nopair/` are leftovers from the `webband_*` fixtures, which left the workspace entirely for their own repo (`RPP1011/webband`, 2026-07-23, per `crates/sims/build.rs` and `crates/sims/Cargo.toml` comments) — there is no `webband_bench.sim` / `webband_bench_nopair.sim` under `assets/sim/` to pair with them; don't treat their presence as evidence a matching fixture exists.

One directory has **no fixture pairing by design**: `dsl_coverage/` is walked directly by `crates/dsl_compiler/tests/dsl_coverage_corpus.rs` (a dedicated "every `.ability` in this directory must lower clean" regression test), not via the `build_helper` fixture-name match — note its name doesn't match any `.sim` stem (`dsl_stress_coverage.sim` is a different, unrelated fixture despite the similar name).

## `assets/models/`

Third-party asset pack (Kenney "mini-characters", see the pack's own `License.txt`) — 27 `.glb` meshes (playable characters plus accessibility props like canes/wheelchairs) under `Models/GLB format/`, with a `Textures/` subfolder. The only consumer is `crates/viewer_runtime`, which loads these via the `gltf` crate (`import`, `names`, `utils` features, `extras` disabled) to render agent characters over the voxel scene it mirrors from `sims::dungeon_horde::GeneratedRuntime`'s GPU state each tick. This directory has no relationship to the DSL compiler or any `.sim`/`.ability` file — it's pure render content.
