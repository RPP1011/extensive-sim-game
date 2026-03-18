# Architecture Overview

This project is a tactical RPG built on three core principles:

1. **Determinism** — identical inputs always produce identical outputs, enabling
   reproducible AI training and replay verification
2. **Data-driven design** — abilities, behaviors, and hero definitions live in
   external files, not hardcoded logic
3. **Layered AI** — decision-making flows through composable layers, from squad
   personality down to neural network evaluation

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Bevy App                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Hub UI   │  │ Camera   │  │  Audio / VFX     │  │
│  │ (egui)   │  │ (orbit)  │  │  (Bevy systems)  │  │
│  └────┬─────┘  └──────────┘  └──────────────────┘  │
│       │                                             │
│  ┌────▼──────────────────────────────────────────┐  │
│  │               Game Loop                       │  │
│  │  Campaign ──▶ Mission ──▶ Combat              │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                             │
│  ┌────────────────────▼──────────────────────────┐  │
│  │            AI Decision Pipeline               │  │
│  │  Squad → GOAP → Behavior → Neural Eval        │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                             │
│  ┌────────────────────▼──────────────────────────┐  │
│  │          Deterministic Simulation             │  │
│  │  step(state, intents, dt_ms) → (state, events)│  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Pure Functional Simulation Core

The simulation engine is designed as a pure function:

```rust
pub fn step(state: SimState, intents: &[UnitIntent], dt_ms: u32)
    -> (SimState, Vec<SimEvent>)
```

`SimState` is taken by value, ensuring the caller cannot hold stale references.
The function returns a new state plus a log of events. This makes the simulation
trivially reproducible and replay-friendly.

### Effects as Plain Data

The entire ability system uses plain data structs dispatched via pattern matching.
There are no closures, no trait objects, no dynamic dispatch in the effect pipeline.
This keeps the effect system serializable, inspectable, and cache-friendly.

### AI as Intent Generation

AI systems never directly mutate game state. Instead, they produce `Vec<UnitIntent>`,
which the simulation engine consumes. This separation means:
- AI can be swapped, layered, or disabled without touching simulation code
- Training pipelines can inject their own intents
- Replays can substitute recorded intents for live AI

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Game engine | Bevy 0.13 |
| UI | bevy_egui |
| Parsing | winnow 0.7 |
| ML (Rust) | Burn 0.20, ndarray |
| ML (Python) | PyTorch, NumPy |
| Serialization | serde + serde_json, TOML |
| Parallelism | rayon, crossbeam |
| CLI | clap 4.5 |
| Testing | proptest (property-based) |
