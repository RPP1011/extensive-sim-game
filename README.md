# Deterministic Tactical Orchestration Game

A tactical crisis-management RPG built in **Rust**. Players manage a hero roster across a contested overworld, resolve flashpoint crises through multi-room missions, and command squads in deterministic real-time combat. A DF-style world simulation runs 167 systems to produce emergent narratives with zero scripted events.

> **Status:** Active development — Rust-native architecture. Not ready to play.

## Nifty Diagram for illustrating the perils of AI native development.
<img src="spaget.svg">

## Documentation

All documentation lives in `docs/` as standalone HTML pages served via GitHub Pages. The entry point is [`docs/index.html`](docs/index.html).

### Structure

```
docs/
├── index.html                  # Landing page — links to everything
├── architecture.html           # Three-layer sim, workspace, module map
├── simulation.html             # Core step(), state, determinism, events
├── effects.html                # Effect system & five composable dimensions
├── dsl.html                    # Ability DSL syntax, parser pipeline
├── ai.html                     # AI decision pipeline (squad, GOAP, transformer)
├── game.html                   # Campaign, factions, missions, room gen
├── ml.html                     # ML training pipeline, entity encoder, RL
├── heroes.html                 # Hero templates, LoL imports
├── tooling.html                # CLI, scenario runner, sim_bridge
├── development.html            # Testing, contributing
├── roadmap.html                # Project roadmap
├── mechanics/                  # Game mechanics wiki (combat, effects, heroes)
│   └── index.html
├── blog_*.html                 # Dev log posts
└── OLD/                        # Archived posts & ML concept pages
```

### Adding new documentation

Documentation pages are self-contained HTML files with inline CSS. There is no build step or static site generator.

**To add a new doc page:**

1. Copy an existing page (e.g. `docs/architecture.html`) as a starting template
2. The shared style lives inline in each file's `<style>` block — keep the CSS variables (`:root { --bg, --surface, ... }`) and font imports consistent
3. Include a `<a href="index.html">← Back</a>` link at the top
4. Add a card entry in `docs/index.html` under the appropriate section (Project Documentation, Dev Log, or ML Concepts)

**To add a new blog/dev log post:**

1. Copy `docs/blog_burn_migration.html` as a template — it has the full article style (header, subtitle, stat cards, code blocks, tables, callouts)
2. Name the file `docs/blog_<slug>.html`
3. Add a card to the "Dev Log" section in `docs/index.html` with date, title, description, and tags

**Style reference:**

| Element | Class/Pattern |
|---------|--------------|
| Info callout | `<div class="callout">` |
| Error/bug callout | `<div class="callout bug">` |
| Success callout | `<div class="callout success">` |
| Green metric | `<td class="good">` |
| Red metric | `<td class="bad">` |
| Warning metric | `<td class="warn">` |
| Inline code | `<code>` (auto-styled in `p`, `li`, `td`) |
| Code block | `<pre><code>` |
| Stat cards (header) | `.stats > .stat > .value + .label` |

## Build & Run

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # All tests
```

### CLI (xtask)

```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
cargo run --bin xtask -- scenario bench scenarios/
cargo run --bin xtask -- scenario generate dataset/scenarios/
```

## Project Management

Use GitHub issues and milestones for active planning.
