# Introduction

Welcome to **The Bevy Game Book** — the definitive guide to an ambitious deterministic
tactical RPG built in Rust. This project sits at the intersection of game development,
AI research, and machine learning, using the [Bevy](https://bevyengine.org/) engine as
its foundation and a custom deterministic combat simulation as its core.

## What This Project Is

At its heart, this is a tactical squad-based RPG with three layers of gameplay:

1. **Campaign** — a turn-based overworld with faction diplomacy, flashpoints, and
   narrative progression
2. **Mission** — multi-room procedurally generated dungeons with objectives and
   enemy encounters
3. **Combat** — a 100ms fixed-tick deterministic simulation where squads of heroes
   fight enemies using a rich, data-driven ability system

The combat layer is where the most interesting engineering lives. It powers:
- A **composable ability DSL** that defines hundreds of unique hero abilities
- A **layered AI decision pipeline** from squad-level personality down to neural
  ability evaluation
- A **machine learning training pipeline** that teaches agents to play through
  self-play, behavior cloning, and reinforcement learning

## Who This Book Is For

- **Contributors** who want to understand the codebase before diving in
- **Researchers** interested in the AI/ML architecture and training pipeline
- **Game designers** exploring the ability system and hero template format
- **Rustaceans** curious about a non-trivial Bevy + ML project structure

## How to Read This Book

The book is organized from core systems outward:

- Start with the **Architecture Overview** to understand the three-layer model and
  module map
- Read **The Simulation Engine** to understand the deterministic `step()` function
  that drives everything
- Explore **The Effect System** and **Ability DSL** to understand how abilities are
  defined and resolved
- Study the **AI Decision Pipeline** to see how units choose what to do each tick
- The remaining sections cover game systems, ML training, hero content, tooling, and
  development practices

## Building the Book

This book is built with [mdBook](https://rust-lang.github.io/mdBook/). To build
locally:

```bash
cargo install mdbook
cd book
mdbook build    # outputs to book/book/
mdbook serve    # live preview at http://localhost:3000
```

## Building the Project

```bash
cargo build                    # debug build
cargo build --release          # optimized build
cargo test                     # run all tests
cargo run --bin xtask -- --help  # CLI task runner
```

## Other Resources

This book is part of a larger documentation site. See also:

- [**Dev Log**](../blog_burn_migration.html) — engineering blog posts about the
  training infrastructure and migration work
- [**ML Concepts**](../OLD/concepts_foundations.html) — a six-part learning path
  covering the mathematical foundations used in the ML pipeline

Let's begin.
