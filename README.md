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
cargo test                     # All tests (workspace)
cargo test -p engine           # World-sim engine tests only (~210)
cargo test -p engine --release # Release build; required for 2s-budget acceptance tests
```

### Running the world-sim viz

The world-sim engine ships with a visualization harness at `crates/viz/` that renders a running `SimState` via `voxel_engine`'s Vulkan path. Three sample scenarios under `crates/viz/scenarios/`:

```bash
cargo run -p viz -- crates/viz/scenarios/viz_basic.toml     # 5 humans vs 1 wolf
cargo run -p viz -- crates/viz/scenarios/viz_attack.toml    # targeted melee
cargo run -p viz -- crates/viz/scenarios/viz_announce.toml  # announce cascade (no macros from default backend yet)
```

**Controls:**

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `.` (Period) | Single step (while paused) |
| `R` | Reset to initial scenario |
| `[` / `]` | Slow down / speed up tick rate |
| `W` `A` `S` `D` | Pan camera in XY plane |
| `Q` / `E` | Raise / lower camera |
| Middle-mouse drag | Orbit around scene centroid |
| Right-mouse drag | Look (mouselook) |
| Scroll | Zoom in / out |
| `Esc` | Exit |

**Voxel palette:**

| Color | Meaning |
|---|---|
| Gray floor | Ground plane |
| **Blue** | Alive human |
| **Red** (dark) | Alive wolf |
| Tan | Alive deer |
| Orange | Alive dragon |
| **Bright red / salmon** | Attack overlay (5-tick TTL from `AgentAttacked`) |
| **Black** | Death marker (persistent from `AgentDied`) |
| **White** ring | Announce propagation (expanding over 3 ticks) |

When multiple agents share a voxel cell (the engine has no geometric body collision by design — see `docs/engine/status.md`), agents are **vertically stacked** so all are visible.

Stdout HUD prints `tick=N alive=A/T overlays=O fps=F eye=(x,y,z) lookAt=(x,y,z)` every second.

### CLI (xtask — legacy tactical sim)

```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
cargo run --bin xtask -- scenario bench scenarios/
cargo run --bin xtask -- scenario generate dataset/scenarios/
```

### Engine spec + status

- `docs/engine/spec.md` — 26-section runtime contract (Serial + GPU backends)
- `docs/engine/status.md` — **start here**: per-subsystem implementation state, known weak tests, visual-check criteria, open verification questions
- `docs/engine/verification_audit_2026-04-19.md` — prior audit (HIGH + MEDIUM findings all resolved)
- `docs/audit_2026-04-19.md` — repository-wide audit (plans queue + state-catalog gap + consolidation proposals)
- `docs/superpowers/plans/` — per-plan implementation intent (MVP through Combat Foundation)

### Throughput baseline

From `cargo bench -p engine --bench tick_throughput -- --quick` (single-threaded; GPU backend not yet built):

| Policy | 100 agents × 1000 ticks | Steps/sec |
|---|---:|---:|
| `UtilityBackend` | ~5.7 ms | ~175k |
| `MixedPolicy` (Drink/Rest/Attack/Move/Communicate) | ~11.1 ms | ~90k |

At n=500 the mixed-action workload lands ~8.6k steps/sec — bound by single-threaded mask predicates and apply-action dispatch. `ComputeBackend` trait extraction + rayon + GPU kernel porting are planned (see `docs/engine/spec.md` §25, Plans 5 + 6+).

## Project Management

Use GitHub issues and milestones for active planning. Implementation plans live in `docs/superpowers/plans/`; per-subsystem status lives in `docs/engine/status.md`.
