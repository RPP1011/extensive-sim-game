# Deterministic Tactical Orchestration Game

A tactical crisis-management RPG built with **Rust** and the **Bevy 0.13** ECS engine. Players manage a hero roster across a contested overworld, resolve flashpoint crises through multi-room missions, and command squads in deterministic real-time combat.

> **Status:** Active development -- Rust-native architecture using Bevy ECS.

---

## Highlights

- **Three-layer simulation** -- Turn-based campaign overworld, multi-room mission runs, and 100ms fixed-tick deterministic combat
- **27 hero archetypes** (Alchemist, Assassin, Berserker, Paladin, Witch, ...) defined in TOML with 8-9 abilities each
- **Data-driven ability system** -- 32+ effects, 7 area shapes, 7 delivery methods, 20+ conditions, passive triggers
- **Autonomous AI factions** that pursue territory, diplomacy, and military operations
- **Flashpoint crisis chains** (Recon -> Sabotage -> Decisive) with player-chosen risk/reward intents
- **Companion story arcs** woven into campaign consequences
- **Deterministic seeding** throughout for reproducible simulations and testing
- **Optional Gemini AI integration** for procedural map art and backstory cinematics

## Quick Start

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (edition 2021)
- Optional: `GEMINI_API_KEY` environment variable for AI art generation features

### Run the Game

```bash
cargo run
```

### Run Tests

```bash
cargo test
```

### Load a Saved Campaign

```bash
cargo run -- --load-campaign generated/saves/campaign_save.json
```

## Architecture

```
Campaign (turn-based overworld)
  -> Mission (multi-room dungeon runs)
    -> Combat (100ms fixed-tick, deterministic)
```

| Layer | Description |
|-------|-------------|
| **Campaign** | Overworld map with faction ownership, unrest, diplomacy, border pressure, and flashpoint triggers |
| **Mission** | Multi-room sequences (Entry -> Pressure -> Pivot -> Setpiece -> Recovery -> Climax) with procedural encounters |
| **Combat** | Squad-based real-time simulation with personality-driven AI, role assignment, and phase-based tactics |

### Key Modules

| Directory | Purpose |
|-----------|---------|
| `src/game_core/` | Campaign state, mission types, roster management |
| `src/ai/` | Combat simulation engine, ability system, squad AI, pathing |
| `src/mission/` | Mission execution, objectives, room generation, enemy templates |
| `src/hub_ui_draw/` | Egui-based hub UI rendering |
| `src/campaign_ops/` | Save/load, campaign initialization, entity snapshots |
| `src/runtime_assets/` | Async Gemini-powered environment art generation |
| `src/bin/xtask/` | Task runner for map generation, screenshot capture, simulation phases |
| `assets/hero_templates/` | 27 hero definitions in TOML |

## Controls

### Campaign Save/Load

| Action | Keys |
|--------|------|
| Save/Load slot 1 | `F5` / `F9` |
| Save/Load slot 2 | `Shift+F5` / `Shift+F9` |
| Save/Load slot 3 | `Ctrl+F5` / `Ctrl+F9` |
| Save panel | `F6` (open), `Up/Down` (select), `S` (save), `G` (preview), `Enter` (confirm load) |

Autosave triggers every 10 turns. Save writes are atomic with `.bak` backups. Save files are versioned and auto-migrated on load. Slot metadata lives in `generated/saves/campaign_index.json`.

### Flashpoint Intents

During active flashpoint missions, press `1` (Stealth Push), `2` (Direct Assault), or `3` (Civilian First) to adjust risk/reward.

## Screenshot Capture

Capture UI frames for visual regression testing. Each capture produces a `.png` and a `.json` sidecar with simulation state.

```bash
# Single snapshot
cargo run -- --screenshot generated/screenshots/ui_baseline

# Multi-frame sequence
cargo run -- --steps 30 --screenshot-sequence generated/screenshots/run_a
```

Optional flags: `--screenshot-every N`, `--screenshot-warmup-frames N`.

### Windows Native Capture

For environments where WSL graphics support is limited, run captures from a native Windows shell:

```powershell
# Single capture (hub scene)
cargo run --bin xtask -- capture windows --mode single --hub --out-dir generated/screenshots/windows_hub

# Sequence capture
cargo run --bin xtask -- capture windows --mode sequence --steps 40 --every 1 --out-dir generated/screenshots/windows_seq

# Hub stage walkthrough (Start Menu -> Character Creation -> Overworld -> Region View -> Local Intro)
cargo run --bin xtask -- capture windows --mode hub-stages --out-dir generated/screenshots/windows_hub_stages

# Safe sequence (one process per frame, avoids swapchain issues)
cargo run --bin xtask -- capture windows --mode safe-sequence --steps 20 --out-dir generated/screenshots/windows_safe_seq
```

Captures are ephemeral by default. Add `--persist` to keep artifacts. Run `cargo run --bin xtask -- capture dedupe --out-dir <dir>` to remove exact visual duplicates.

## AI-Assisted Content Generation

Requires the `gemini` feature flag and a `GEMINI_API_KEY`.

### Map Concept Art

```bash
# From inline prompt
cargo run --bin xtask -- map gemini \
  --prompt "Top-down fantasy dungeon map, guild branch mission, 3 lanes, ritual boss chamber" \
  --out generated/maps/guild_mission_map.png --save-text

# From prompt template
cargo run --bin xtask -- map gemini \
  --prompt-file scripts/map_prompt_template.txt \
  --out generated/maps/template_map.png
```

Model selection: default `gemini-3-pro-image-preview`, alternate `--model gemini-2.5-flash-image`.

### Procedural Overworld (Voronoi -> Gemini)

Build a weighted Voronoi map prompt from a campaign save, then generate art:

```bash
# Generate prompt + spec
cargo run --bin xtask -- map voronoi \
  --save generated/saves/campaign_autosave.json \
  --out-prompt generated/maps/overworld_voronoi_prompt.txt \
  --out-spec generated/maps/overworld_voronoi_spec.json

# Generate image directly
cargo run --bin xtask -- map voronoi \
  --save generated/saves/campaign_autosave.json \
  --run-gemini --gemini-out generated/maps/overworld_voronoi_map.png
```

### Environment Art Pipeline (Gemini + HNSW)

Semantic search over an environment prompt corpus, then generate fantasy environment art:

```bash
# Build index
cargo run --bin xtask -- map env-art build-index \
  --corpus scripts/ai/fantasy_env_prompt_corpus.json \
  --index generated/hnsw/env_art_index.json

# Query
cargo run --bin xtask -- map env-art query \
  --corpus scripts/ai/fantasy_env_prompt_corpus.json \
  --index generated/hnsw/env_art_index.json \
  --query "stormy mountain fortress with vertical traversal lanes" --top-k 8

# Generate batch art
cargo run --bin xtask -- map env-art generate \
  --corpus scripts/ai/fantasy_env_prompt_corpus.json \
  --index generated/hnsw/env_art_index.json \
  --query "misty flooded ruins with lantern lighting" \
  --top-k 12 --count 8 --style concept \
  --out-dir generated/maps/fantasy_env
```

Style options: `concept`, `matte`, `illustration`. Use `--refresh-index` to force index rebuild.

### Runtime Asset Generation

During gameplay the runtime asset service queues environment concept-art jobs asynchronously. Live status appears in the `Runtime Asset Gen` overlay. Outputs go to `generated/maps/runtime_env`.

### Backstory Cinematic

After backstory confirmation, a cinematic plays 4 AI-generated beats (Origin, Crisis, Decision, Vow) as a Ken Burns-style slideshow before transitioning to the overworld.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System layers, module layout, design decisions |
| [docs/SYSTEMS.md](docs/SYSTEMS.md) | Complete reference for campaign, mission, AI, and UI systems |
| [docs/ABILITY_SYSTEM.md](docs/ABILITY_SYSTEM.md) | Ability engine reference (effects, shapes, delivery, conditions) |
| [docs/ABILITY_PRIORITY_SYSTEM.md](docs/ABILITY_PRIORITY_SYSTEM.md) | Squad ability rotation and prioritization |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Execution plan and priorities |
| [docs/CAMPAIGN_FLOW_PLAN.md](docs/CAMPAIGN_FLOW_PLAN.md) | Target player journey and phased implementation |

## Project Management

Use GitHub issues and milestones for active planning.

## License

See repository for license details.
