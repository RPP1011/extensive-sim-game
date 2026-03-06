# Deterministic Tactical Orchestration Game

This project is a deterministic tactical orchestration game where the player manages parallel crises.

## Project Status

The project is currently undergoing a migration to a Rust-Native architecture using the Bevy engine.

## Documentation

*   **Roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md) - Current execution plan and priorities.
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

## Gemini Map Image Generation

You can generate map concept images with Gemini (including Pro image model support).

Prerequisites:
- Export an API key: `export GEMINI_API_KEY=...`

Generate from inline prompt:

```bash
python3 scripts/gemini_mapgen.py \
  --prompt "Top-down fantasy dungeon map, guild branch mission, 3 lanes, ritual boss chamber" \
  --out generated/maps/guild_mission_map.png \
  --save-text
```

Generate from the included prompt template:

```bash
python3 scripts/gemini_mapgen.py \
  --prompt-file scripts/map_prompt_template.txt \
  --out generated/maps/template_map.png
```

Model selection:
- Default model: `gemini-3-pro-image-preview`
- Alternate: `--model gemini-2.5-flash-image`
