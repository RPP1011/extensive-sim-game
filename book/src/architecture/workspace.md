# Workspace & Crate Map

The project is a Cargo workspace with two member crates.

## Workspace Layout

```toml
# Cargo.toml (root)
[workspace]
members = [".", "crates/ability_operator"]
```

### `bevy_game` (root crate)

The main crate containing the game binary, simulation engine, AI systems, and all
game logic. It compiles as both a library (for binary targets to link against) and
a binary (the main game application).

**Feature flags:**
- `app` — application mode (enables Bevy rendering and UI)
- `stream-monitor` — enables the real-time monitoring system
- `burn-gpu` — GPU-accelerated ML via Burn + tch backend
- `burn-cpu` — CPU ML via Burn + ndarray backend

### `crates/ability_operator`

A standalone crate for **behavioral embeddings** of abilities. It trains a small
neural network to map ability definitions to embedding vectors, enabling the AI to
reason about unfamiliar abilities by their behavioral similarity to known ones.

```
crates/ability_operator/
├── Cargo.toml
└── src/
    ├── lib.rs           # library root
    ├── bin/train.rs     # training binary
    ├── model.rs         # Burn model architecture
    ├── train.rs         # training loop
    ├── loss.rs          # loss functions
    ├── data.rs          # dataset handling
    └── grokfast.rs      # grokking acceleration
```

## Binary Targets

The workspace produces several binary targets:

| Binary | Source | Purpose |
|--------|--------|---------|
| `bevy_game` | `src/main.rs` | Main game application |
| `xtask` | `src/bin/xtask/main.rs` | CLI task runner (scenarios, training, maps) |
| `sim_bridge` | `src/bin/sim_bridge/main.rs` | Headless simulator with NDJSON protocol |
| `gen_scenarios` | `src/bin/gen_scenarios.rs` | Batch scenario generation |
| `room_preview` | `src/bin/room_preview.rs` | Room visualization tool |
| `train_operator` | `crates/ability_operator/src/bin/train.rs` | Ability operator training |

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `bevy` | 0.13.2 | Game engine (ECS, rendering, input) |
| `bevy_egui` | 0.27 | Immediate-mode UI |
| `winnow` | 0.7 | Parser combinators for the ability DSL |
| `serde` | 1.0 | Serialization framework |
| `toml` | 0.8 | TOML config parsing |
| `ndarray` | 0.16 | N-dimensional arrays for ML features |
| `burn` | 0.20 | ML framework (optional, behind feature flags) |
| `rayon` | 1.10 | Data parallelism |
| `clap` | 4.5 | CLI argument parsing |
| `contracts` | 0.6 | Design-by-contract assertions |
| `proptest` | 1.6 | Property-based testing (dev only) |
