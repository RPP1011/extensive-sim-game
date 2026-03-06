# Deterministic Tactical Orchestration Game

A deterministic tactical orchestration game built with Bevy where the player manages parallel crises across a procedurally generated overworld.

## Overview

The player operates from an Adventurer's Guild hub, dispatching heroes across a seeded hex overworld, managing faction diplomacy, and intervening directly in concurrent missions. All simulation is deterministic and fully saveable.

## Architecture

- **Engine:** Bevy 0.13 (Rust)
- **Mission state:** ECS components per mission entity (`MissionData`, `MissionProgress`, `MissionTactics`, `AssignedHero`), with `ActiveMission` marking the focused entity
- **AI layers:** Utility, Role, Squad, Control, and Personality AI systems with semantic naming
- **Persistence:** JSON campaign saves with versioned schema, atomic writes, and `.bak` backups

## Building and Running

```bash
cargo run
```

Run tests (159 passing):

```bash
cargo test
```

Load a saved campaign at startup:

```bash
cargo run -- --load-campaign generated/saves/campaign_save.json
```

Specify an overworld seed:

```bash
cargo run -- --map-seed 12345
```

## Controls

### Hub / Overworld

| Key | Action |
|-----|--------|
| `Up` / `Down` | Select hub action or overworld region |
| `Enter` / `Space` | Confirm selection |
| `J` / `L` | Cycle travel route |
| `T` | Commit travel to selected region |
| `1` / `2` / `3` | Set flashpoint intent (Stealth Push / Direct Assault / Civilian First) |
| `U` / `O` | Cycle diplomacy offers |
| `Y` / `N` | Accept / decline offer |
| `Escape` | Open settings |

### Save / Load

| Key | Action |
|-----|--------|
| `F5` / `F9` | Save / load slot 1 (`generated/saves/campaign_save.json`) |
| `Shift+F5` / `Shift+F9` | Save / load slot 2 |
| `Ctrl+F5` / `Ctrl+F9` | Save / load slot 3 |
| `F6` | Open save panel |
| `Up` / `Down` (panel) | Select slot |
| `S` (panel) | Save to selected slot |
| `G` (panel) | Preview load |
| `Enter` (panel) | Confirm load |
| `Escape` (panel) | Close panel |

Autosave fires every 10 turns to `generated/saves/campaign_autosave.json`. Slot metadata is tracked in `generated/saves/campaign_index.json`.

### 3D Scene / Camera

| Key | Action |
|-----|--------|
| `W` / `A` / `S` / `D` | Orbit camera |
| `F` | Focus camera |
| `Space` | Pause / resume scenario replay |
| `Left` / `Right` | Step replay frame |

## Campaign Features

- **Overworld:** Seeded hex map with faction boundaries, regional unrest/control, and travel cooldown/energy rules
- **Factions:** AI commanders generate crisis intents; border friction drives war-goal computation and AI-vs-AI pressure simulation
- **Flashpoints:** Sustained regional pressure spawns 3-stage crisis chains (Recon → Sabotage → Decisive) with campaign consequences on decisive resolution
- **Roster:** Procedural hero generation with faction/region backstory, companion assignment, injury/loyalty/stress drift, and story quests
- **Diplomacy:** Joint missions, raids, training loans, and recruit borrowing offers from NPC factions
- **Intel:** Region fog with decay, mission-signal discovery, and travel-based reveals

## Gemini Map Image Generation

Generate map concept images using the Gemini API.

Prerequisites:

```bash
export GEMINI_API_KEY=...
```

Generate from an inline prompt:

```bash
python3 scripts/gemini_mapgen.py \
  --prompt "Top-down fantasy dungeon map, guild branch mission, 3 lanes, ritual boss chamber" \
  --out generated/maps/guild_mission_map.png \
  --save-text
```

Generate from the included template:

```bash
python3 scripts/gemini_mapgen.py \
  --prompt-file scripts/map_prompt_template.txt \
  --out generated/maps/template_map.png
```

Model selection:
- Default: `gemini-3-pro-image-preview`
- Alternate: `--model gemini-2.5-flash-image`

## Documentation

- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
