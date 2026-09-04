# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What it does

A standalone PPO trainer for a tactical-combat AI policy, built on the
`rl4burn` RL framework. `src/main.rs` loads hero templates (TOML +
companion `.ability` DSL files) from `dataset/abilities/hero_templates/`
and `dataset/abilities/lol_heroes/`, wraps the tactical combat simulator as
an `rl4burn::Env` (`src/env.rs`'s `CombatEnv`), and trains a shared
actor-critic MLP (`src/model.rs`'s `CombatPolicy`, 210 → 128 → 64 →
{14 actor logits, 1 critic value}) via masked PPO
(`masked_ppo_collect`/`masked_ppo_update`) on 32 vectorized envs. It's
parameter-shared independent PPO (IPPO): `CombatEnv` steps through heroes
round-robin, one observation/action per hero per call, and only advances the
sim tick once every hero on the team has acted; reward is shaped from
ally/enemy HP and kill deltas. Checkpointing is a stub — the "save model
weights to disk" TODO in `main.rs` is never implemented.

## Status: currently broken — cannot even parse

`Cargo.toml` declares `tactical_sim = { path = "../tactical_sim" }`, but
**`crates/tactical_sim` does not exist** — it was deleted in the Phase 7
wolf-sim wipe (2026-05-02; explicitly named in root `Cargo.toml`'s wipe
comment). It also declares `rl4burn = { workspace = true }`, but the root
`Cargo.toml` has no `[workspace.dependencies]` table at all, and because
this crate is workspace-`exclude`d, cargo won't even associate it with the
root workspace to inherit from. The manifest fails to parse before dependency
resolution gets anywhere near `tactical_sim`:

```
error: failed to parse manifest at .../combat-trainer/Cargo.toml
Caused by: error inheriting `rl4burn` from workspace root manifest's `workspace.dependencies.rl4burn`
Caused by: failed to find a workspace root
```

This is an orphaned pre-wipe artifact left in the workspace `exclude` list,
not a crate anyone can currently build. `src/env.rs` also has an unresolved
symbol mismatch baked in regardless of the missing dependency: it imports
`extract_game_state_v2` from `tactical_sim::sim::ability_eval` but calls a
function named `extract_game_state` (not imported), and references
`GAME_STATE_DIM` for `OBS_DIM` without importing it (only `model.rs` imports
`GAME_STATE_DIM`) — so this file would still fail to compile even with a
working `tactical_sim` path and a real `rl4burn` version pinned.

Reviving this crate means, at minimum: reintroducing or replacing
`tactical_sim` (check git history around the Phase 7 wipe for what it
contained), adding a concrete `rl4burn` version/source (it's an external dep,
not in `crates/`), and fixing the `env.rs` symbol references before touching
training logic.

## Commands (once/if it's revived)

Excluded from the workspace, so `-p combat-trainer` from the workspace root
does not resolve it ("did not match any packages"). `cd
crates/combat-trainer` and run cargo from inside the crate directory —
`--manifest-path` from outside fails identically.

```bash
cd crates/combat-trainer
cargo check
cargo run --release            # long-running: 10M steps, no CLI args/flags today
```

No `rust-toolchain.toml` here (unlike `world_sim_bench`) — it uses whatever
toolchain is ambient.

## Dependencies (Cargo.toml)

- `rl4burn` (workspace-inherited version — currently unresolvable, see above)
  — the PPO/env framework (`Env`, `Space`, `Step`, `SyncVecEnv`,
  `masked_ppo_collect`, `masked_ppo_update`, `PpoConfig`,
  `MaskedActorCritic`, `ActionDist`). Not vendored in this repo.
- `tactical_sim` (path `../tactical_sim`) — does not exist; supplies
  `SimState`, `UnitState`, `UnitStore`, `step`, `Team`, hero/ability types
  (`HeroToml`, `AbilitySlot`, `PassiveSlot`), the DSL ability parser
  (`effects::dsl::parse_abilities`), squad AI (`generate_intents`,
  `SquadAiState`, `Personality`), and the observation extractor
  (`ability_eval::{extract_game_state*, GAME_STATE_DIM, ...}`).
- `burn = "0.20"` (`ndarray`, `autodiff` features) — the tensor/NN framework
  `CombatPolicy` and training are built on.
- `rand = "0.10"` — env RNG (`SmallRng`).
- `toml = "0.8"` — parses hero template `.toml` files.

No dependency on `engine`, `sims`, `dsl_compiler`, `ability_operator`, or
`ability-vae` — this crate predates the current `engine`/`sims`
architecture and targets the now-deleted `tactical_sim` crate instead.
