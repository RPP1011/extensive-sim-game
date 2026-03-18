# CLI & xtask

The project uses the [xtask pattern](https://github.com/matklad/cargo-xtask) for
project-specific CLI commands. The `xtask` binary provides scenario running,
benchmarking, oracle evaluation, map generation, and training utilities.

## Running xtask

```bash
cargo run --bin xtask -- <subcommand> [options]
```

## Subcommands

### `scenario run`
Run a single scenario to completion:
```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
```

### `scenario bench`
Benchmark a directory of scenarios:
```bash
cargo run --bin xtask -- scenario bench scenarios/
```

### `scenario oracle eval`
Evaluate the oracle policy on scenarios:
```bash
cargo run --bin xtask -- scenario oracle eval scenarios/
```

### `scenario oracle dataset`
Generate training datasets from oracle play:
```bash
cargo run --bin xtask -- scenario oracle dataset scenarios/ --output generated/oracle.jsonl
```

### `scenario oracle transformer-play`
Run the transformer model on scenarios:
```bash
cargo run --bin xtask -- scenario oracle transformer-play model.npz --weights weights/
```

### `scenario oracle transformer-rl generate`
Generate self-play RL training data:
```bash
cargo run --bin xtask -- scenario oracle transformer-rl generate scenarios/
```

### `map`
Map generation utilities:
```bash
cargo run --bin xtask -- map generate --seed 42 --output generated/map.json
```

### `roomgen`
Room generation and preview:
```bash
cargo run --bin xtask -- roomgen --seed 42 --theme dungeon
```

## Module: `src/bin/xtask/`

```
xtask/
├── main.rs           # CLI entry point (clap)
├── cli/
│   ├── mod.rs        # Subcommand definitions
│   ├── scenario.rs   # Scenario subcommands
│   └── map.rs        # Map subcommands
├── scenario_cmd.rs   # Scenario execution
├── map.rs            # Map generation
├── roomgen_cmd.rs    # Room generation
├── train_v6.rs       # V6 Burn training
├── capture.rs        # Screenshot capture
└── oracle_cmd/
    ├── mod.rs        # Oracle subcommands
    ├── rl_generate.rs
    ├── rl_train.rs
    └── impala_train.rs
```
