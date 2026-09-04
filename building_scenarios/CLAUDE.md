# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this is

TOML data for a "building AI" scenario system: `enemy_profiles/` (flat enemy-type stat blocks — `type_name`, `level_range`, movement capabilities like `can_jump`/`can_climb`/`can_tunnel`/`can_fly`, siege stats), `scenarios/` (+ `scenarios/eval/`) (a `[meta]` block with tags/matrix_cells, a `[seed]` pointing at a settlement template, `[[challenges]]` with severity ranges and `[[challenges.enemies]]` referencing an enemy profile by path, plus `sim_ticks`/`num_random_baselines`), and `seeds/` (settlement templates: population/tech_tier/terrain, `[stockpiles]`, repeated `[[buildings]]` and `[[npcs]]` blocks). Scenario files cross-reference other files by relative path string (`template = "seeds/town_hill.toml"`, `profile = "enemy_profiles/orcs.toml"`) — nothing enforces those paths at parse time, they're just conventions a loader would resolve. Two loose `visualize_*.html` files sit at the top level (standalone viewers, not consumed by any crate either — see below).

## What consumes it

**Nothing in the current `crates/` workspace.** `Grep`-ing `crates/` for `building_scenarios`, `enemy_profiles`, `scenarios/eval`, or any of the TOML field names above (`matrix_cells`, `sim_ticks`, `num_random_baselines`, `settlement_level`) returns zero hits — no `include_str!`, no path literal, no config-loader struct.

Git history explains why: this directory was added in commit `05ed5b8c` ("feat: add building AI pipeline with reactive sim integration") along with a `building-ai run/validate/generate` CLI and a TOML-driven scenario/oracle/validation pipeline living under the old pre-workspace `src/` tree (`src/ai`, `src/world_sim`, `src/scenario`, `src/mission`, etc.). That entire `src/` tree — the only code that ever read these files — was deleted wholesale in commit `78aa1578` ("chore: delete legacy src/{ai,world_sim,scenario,mission,game_core,content,narrative,model_backend,ascii_gen} + casualty xtask sources (Plan B3 Task 3)", 2026-04-25). The data survived that deletion; its reader did not.

`docs/spec/state.md` still has a couple of fossil references to the concept (`skip_resource_init` field noted as "Skip resource node spawning (building-AI scenarios)"; `construction_memory` noted as read by "building_ai pattern learning") — these describe engine-side hooks that predate/anticipate the deleted pipeline, not evidence of a live consumer.

## Status

**Orphaned.** No crate in the workspace reads any file under this directory. Do not assume changes here affect simulation behavior, and do not build new features on the assumption that a loader exists — it doesn't, currently. If you're asked to revive building-AI scenario evaluation, the prior implementation (and the shape of the loader/oracle/validator it expects) is only recoverable from git history before `78aa1578`, not from anything presently in `crates/`.

## Non-obvious

- The `matrix_cells` field in scenario `[meta]` blocks (pairs like `{ challenge = "military", decision = "placement" }`) suggests these scenarios were meant to tile an eval matrix (challenge type × decision type) for a coverage-style eval harness — that harness is part of what got deleted.
- `visualize_all_evals.html` / `visualize_layout.html` at the directory root are standalone/static viewers (not referenced by any crate either); check them directly before assuming they still work against the current TOML shapes.
