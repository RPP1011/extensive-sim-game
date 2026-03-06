# Deterministic Tactical Orchestration Game

This project is a deterministic tactical orchestration game where the player manages parallel crises.

## Project Status

The project is currently undergoing a migration to a Rust-Native architecture using the Bevy engine.

## Documentation

*   **Roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md) - Current execution plan and priorities.
*   **Campaign Flow Plan:** [docs/CAMPAIGN_FLOW_PLAN.md](docs/CAMPAIGN_FLOW_PLAN.md) - Target player journey, gaps, and phased implementation plan.
*   **Changelog:** [CHANGELOG.md](CHANGELOG.md) - Chronological change history.

## Project Management

*   Use issues and milestones for active planning.

## Building and Running

Run the game:

```bash
cargo run
```

Run tests:

```bash
cargo test
```

Campaign save/load controls:
- `F5` / `F9`: save/load slot 1 (`generated/saves/campaign_save.json`)
- `Shift+F5` / `Shift+F9`: save/load slot 2 (`generated/saves/campaign_slot_2.json`)
- `Ctrl+F5` / `Ctrl+F9`: save/load slot 3 (`generated/saves/campaign_slot_3.json`)
- Autosave: every 10 turns to `generated/saves/campaign_autosave.json`
- Save panel: `F6` to open, `Up/Down` select, `S` save selected slot, `G` preview load, `Enter` confirm load.
- Save writes are atomic and keep a backup (`.bak`) when overwriting existing slot files.
- Save files are versioned and migrated on load when supported.
- Slot metadata is tracked in `generated/saves/campaign_index.json` (version, turn, seed, timestamp, compatibility).
- Vertical-slice overworld crisis loop: high border pressure can open a 3-stage `Flashpoint` mission chain (`Recon -> Sabotage -> Decisive`) that can shift region ownership and alter recruit access.
- Flashpoint stage intents: press `1` (`Stealth Push`), `2` (`Direct Assault`), or `3` (`Civilian First`) to retune active flashpoint mission risk/reward.
- Flashpoints now surface companion hook objectives and projected consequences (`win=>border+recruit`, `lose=>hold-line`) directly in mission queue entries.

Startup load:

```bash
cargo run -- --load-campaign generated/saves/campaign_save.json
```

UI screenshot capture:

- Single snapshot directory (captures one frame then exits):

```bash
cargo run -- --screenshot generated/screenshots/ui_baseline
```

- Snapshot sequence (good for regression tests over multiple ticks):

```bash
cargo run -- --steps 30 --screenshot-sequence generated/screenshots/run_a
```

Optional capture tuning:
- `--screenshot-every N`: write one image every `N` rendered frames in sequence mode.
- `--screenshot-warmup-frames N`: number of rendered frames to wait before first capture.
- Each capture writes both `frame_000NN.png` and `frame_000NN.json`.
- JSON sidecars include global turn plus mission-level state so visual output can be validated against internal simulation state.

Windows-native Rust-only capture (outside WSL):

- If WSL graphics backend support is limited, run screenshot captures from a native Windows shell using Cargo directly.
- Preferred entrypoint: `cargo run --bin xtask -- capture ...`
- Underlying wrappers remain available:
  - PowerShell: `scripts/capture_windows.ps1`
  - CMD: `scripts/capture_windows.cmd`
- Capture is ephemeral by default (auto-cleans screenshot artifacts after run).
- Add `--persist` to keep screenshots in `--out-dir`.

Single capture (hub scene):

```powershell
cargo run --bin xtask -- capture windows --mode single --hub --out-dir generated/screenshots/windows_hub
```

Sequence capture (regression run):

```powershell
cargo run --bin xtask -- capture windows --mode sequence --steps 40 --every 1 --out-dir generated/screenshots/windows_seq
```

Note: `--hub` with `--mode sequence` is intentionally blocked because hub start-menu gating currently bypasses step-based auto-exit.

Safe sequence fallback (one process per frame; slower but avoids swapchain instability on some systems):

```powershell
cargo run --bin xtask -- capture windows --mode safe-sequence --steps 20 --out-dir generated/screenshots/windows_safe_seq
```

Deterministic hub stage capture (Start Menu -> Character Creation -> Overworld -> Region View -> Local Intro, one frame each, auto-exit):

```powershell
cargo run --bin xtask -- capture windows --mode hub-stages --out-dir generated/screenshots/windows_hub_stages
```

Keep artifacts for inspection:

```powershell
cargo run --bin xtask -- capture windows --mode hub-stages --out-dir generated/screenshots/windows_hub_stages --persist
```

Post-process dedupe (exact visual duplicates + orphan JSON cleanup):

```powershell
cargo run --bin xtask -- capture dedupe --out-dir generated/screenshots/windows_seq
```

CMD equivalent:

```cmd
scripts\capture_windows.cmd -Mode sequence -Steps 40 -Every 1 -OutDir generated\screenshots\windows_seq
```

## Gemini Map Image Generation

You can generate map concept images with Gemini (including Pro image model support).

Prerequisites:
- Export an API key: `export GEMINI_API_KEY=...`

Generate from inline prompt:

```bash
cargo run --bin xtask -- map gemini \
  --prompt "Top-down fantasy dungeon map, guild branch mission, 3 lanes, ritual boss chamber" \
  --out generated/maps/guild_mission_map.png \
  --save-text
```

Generate from the included prompt template:

```bash
cargo run --bin xtask -- map gemini \
  --prompt-file scripts/map_prompt_template.txt \
  --out generated/maps/template_map.png
```

Model selection:
- Default model: `gemini-3-pro-image-preview`
- Alternate: `--model gemini-2.5-flash-image`

## Procedural Overworld Voronoi -> Gemini

Build a weighted Voronoi campaign-map prompt/spec from the current overworld save (territory size scaling by faction strength, plus organic edge metadata):

```bash
cargo run --bin xtask -- map voronoi \
  --save generated/saves/campaign_autosave.json \
  --out-prompt generated/maps/overworld_voronoi_prompt.txt \
  --out-spec generated/maps/overworld_voronoi_spec.json
```

Then run Gemini directly from that generated prompt:

```bash
cargo run --bin xtask -- map voronoi \
  --save generated/saves/campaign_autosave.json \
  --run-gemini \
  --gemini-out generated/maps/overworld_voronoi_map.png
```

## Gemini + hnsw_rs Environment Art Pipeline

Use Gemini embeddings with Rust `hnsw_rs` ANN search for semantic term queries, then generate
detailed fantasy environment background art (including concept-art style outputs).

Prerequisites:
- `GEMINI_API_KEY` in environment or `.env`

Query the environment prompt corpus with semantic search:

```bash
cargo run --bin xtask -- map env-art build-index \
  --corpus scripts/ai/fantasy_env_prompt_corpus.json \
  --index generated/hnsw/env_art_index.json
```

Then query the persisted index:

```bash
cargo run --bin xtask -- map env-art query \
  --corpus scripts/ai/fantasy_env_prompt_corpus.json \
  --index generated/hnsw/env_art_index.json \
  --query "stormy mountain fortress with vertical traversal lanes" \
  --top-k 8
```

Generate environment-only batch art from top semantic matches:

```bash
cargo run --bin xtask -- map env-art generate \
  --corpus scripts/ai/fantasy_env_prompt_corpus.json \
  --index generated/hnsw/env_art_index.json \
  --query "misty flooded ruins with lantern lighting and broken bridges" \
  --top-k 12 \
  --count 8 \
  --style concept \
  --out-dir generated/maps/fantasy_env
```

Notes:
- Style options: `concept`, `matte`, `illustration`.
- Outputs include `.png`, prompt text files, Gemini response text, and `manifest.json`.
- The generator enforces environment-only constraints (no characters).
- Use `--refresh-index` on `query`/`generate` to force index rebuild.

## Runtime Asset Generation (In-Game)

When running hub mode (`cargo run` default), the game now includes a runtime asset generation
service:

- Queues environment concept-art jobs while the game is running.
- Executes generation asynchronously (non-blocking game loop).
- Shows live status + recent outputs in the `Runtime Asset Gen` overlay window.
- Writes outputs to `generated/maps/runtime_env`.

Current provider is Gemini; the service is intentionally provider-abstracted in code so a local
1-2B model backend can be swapped in later.

### Backstory Cinematic Flow

After confirming backstory, the hub now transitions to a `BackstoryCinematic` screen:

- Queues 4 beat images (`Origin`, `Crisis`, `Decision`, `Vow`) with high priority.
- Shows a loading screen until the first beat image is ready.
- Plays a Ken Burns-style slideshow (pan/zoom over stills) while remaining beats continue generating.
- Automatically enters `OverworldMap` when the cinematic finishes.
