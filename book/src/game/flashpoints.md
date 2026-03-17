# Flashpoints

Flashpoints are dynamic crisis events that create urgency and choice on the
overworld map. They spawn semi-randomly based on faction tensions and narrative
state, and they escalate if not addressed.

## Lifecycle

```
Spawn → Active → (Player responds) → Resolution → Consequences
         │
         └─▶ (Ignored) → Escalation → Worse consequences
```

## Flashpoint Types

Flashpoints can represent:
- **Border skirmishes** between rival factions
- **Monster outbreaks** threatening a region
- **Political crises** requiring diplomatic intervention
- **Resource shortages** affecting multiple factions
- **Companion-related events** tied to story arcs

## Player Choices

When responding to a flashpoint, the player typically has multiple options:
- **Intervene militarily** — start a mission to resolve by combat
- **Negotiate** — attempt diplomacy (success depends on reputation)
- **Exploit** — take advantage of the chaos for personal gain
- **Ignore** — let events unfold (escalation risk)

## Modules

- `src/game_core/flashpoint_spawn.rs` — generation logic and spawn conditions
- `src/game_core/flashpoint_progression.rs` — escalation and resolution
