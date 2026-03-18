# Behavior DSL

The behavior DSL defines simple behavior trees that drive unit AI when GOAP is
not loaded. These are lighter-weight than GOAP plans — suitable for simple enemy
archetypes that don't need goal reasoning.

## Module: `src/ai/behavior/`

```
behavior/
├── parser.rs       # .behavior file parser
├── interpreter.rs  # Tree execution
├── types.rs        # Node types
└── ...
```

## File Format

Behavior files use a `.behavior` extension and live in `assets/behaviors/`:

```
# healer_bot.behavior

selector {
    sequence {
        condition ally_hp_below 50%
        action heal lowest_hp_ally
    }
    sequence {
        condition enemy_in_range 4.0
        action attack nearest_enemy
    }
    action hold
}
```

## Node Types

| Node | Description |
|------|------------|
| `selector` | Tries children in order, returns first success |
| `sequence` | Runs children in order, fails on first failure |
| `condition` | Checks a world predicate |
| `action` | Produces an intent |

## When to Use Behavior Trees vs GOAP

| Behavior Trees | GOAP |
|---------------|------|
| Simple, predictable enemies | Complex hero AI |
| Few decision points | Many goals and actions |
| No planning needed | Plans span multiple ticks |
| Reactive (stimulus → response) | Proactive (goal → plan → action) |

In practice, most enemy archetypes use behavior trees, while hero squads use
the full GOAP + neural pipeline.
