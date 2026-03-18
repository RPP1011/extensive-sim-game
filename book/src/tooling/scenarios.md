# Scenario Runner

Scenarios are TOML-defined combat setups used for testing, benchmarking, and
training data generation. The scenario system provides reproducible combat
encounters without going through the full campaign/mission pipeline.

## Scenario Format

```toml
[scenario]
name = "Basic 4v4"
seed = 42
max_ticks = 500

[[heroes]]
template = "warrior"
position = [2.0, 5.0]

[[heroes]]
template = "mage"
position = [1.0, 7.0]

[[heroes]]
template = "cleric"
position = [0.0, 6.0]

[[heroes]]
template = "ranger"
position = [3.0, 7.0]

[[enemies]]
template = "berserker"
position = [8.0, 5.0]

[[enemies]]
template = "assassin"
position = [9.0, 7.0]

[[enemies]]
template = "necromancer"
position = [10.0, 6.0]

[[enemies]]
template = "elementalist"
position = [7.0, 7.0]
```

## Module: `src/scenario/`

```
scenario/
├── types.rs        # ScenarioConfig
├── runner.rs       # run_scenario_to_state
├── simulation.rs   # run_scenario, check_assertions
└── gen/            # Coverage-driven generation
```

## Running Scenarios

### Programmatic
```rust
let config = load_scenario_file("scenarios/basic_4v4.toml")?;
let (final_state, events) = run_scenario(&config)?;
check_assertions(&config, &final_state)?;
```

### CLI
```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
```

## Coverage-Driven Generation

The `gen/` module automatically generates scenarios to maximize coverage of:
- Hero combinations
- Ability interactions
- Team compositions (tank-heavy, mage-heavy, healer-heavy)
- Difficulty ranges

```bash
cargo run --bin gen_scenarios -- --count 100 --output generated/scenarios/
```

## Assertions

Scenarios can include assertions to verify expected behavior:

```toml
[assertions]
hero_team_wins = true
max_ticks = 300
min_damage_dealt = 500
```

These are checked by `check_assertions()` after simulation completes, making
scenarios usable as regression tests.
