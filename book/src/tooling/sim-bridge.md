# sim_bridge Protocol

The `sim_bridge` binary exposes the combat simulation as a headless process
communicating over stdin/stdout using the NDJSON (newline-delimited JSON)
protocol. This enables external agents — written in any language — to play
the game.

## Module: `src/bin/sim_bridge/`

```
sim_bridge/
├── main.rs      # Process entry point
├── helpers.rs   # State serialization
└── types.rs     # Protocol message types
```

## Protocol

### Initialization
The bridge sends the initial state as the first message:

```json
{"type": "state", "tick": 0, "units": [...], "done": false}
```

### Agent Response
The external agent sends intents:

```json
{"type": "intents", "actions": [
    {"unit_id": 1, "action": {"Attack": {"target_id": 5}}},
    {"unit_id": 2, "action": {"MoveTo": {"position": [3.0, 4.0]}}},
    {"unit_id": 3, "action": "Hold"}
]}
```

### Step Result
The bridge runs `step()` and returns the new state plus events:

```json
{"type": "state", "tick": 1, "units": [...], "events": [...], "done": false}
```

### Termination
When one team is eliminated or max ticks reached:

```json
{"type": "state", "tick": 150, "units": [...], "done": true, "winner": "Hero"}
```

## Running

```bash
# Start the bridge
cargo run --bin sim_bridge -- --scenario scenarios/basic_4v4.toml

# Pipe to/from an external agent
cargo run --bin sim_bridge -- --scenario scenarios/basic_4v4.toml \
    | python my_agent.py \
    | cargo run --bin sim_bridge -- --scenario scenarios/basic_4v4.toml
```

## Use Cases

- **Python RL training** — agents written in Python generate experience through
  the bridge, avoiding the need to reimplement the simulation
- **External tools** — visualization, analysis, or AI competition frameworks
- **Testing** — scripted test agents that verify specific behaviors
