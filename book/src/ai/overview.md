# AI Overview

The AI system is a layered pipeline where each layer can generate, modify, or
override the intents produced by previous layers. This architecture allows
mixing hand-crafted tactics with learned behaviors.

## The Intent Pipeline

```
┌─────────────┐
│  Squad AI   │  Team-wide personality & formation
└──────┬──────┘
       │ Vec<UnitIntent>
┌──────▼──────┐
│  GOAP       │  Goal-oriented planning per unit
└──────┬──────┘
       │ Vec<UnitIntent> (may override)
┌──────▼──────┐
│ Ability     │  Neural urgency evaluation
│ Evaluator   │  (fires when urgency > 0.4)
└──────┬──────┘
       │ Vec<UnitIntent> (may override)
┌──────▼──────┐
│ Transformer │  Cross-attention decision head
│ (optional)  │  over entity + ability tokens
└──────┬──────┘
       │ Vec<UnitIntent> (may override)
┌──────▼──────┐
│ Control AI  │  Hard CC timing coordination
│ (optional)  │
└──────┬──────┘
       │ Final Vec<UnitIntent>
       ▼
   step(state, intents, dt_ms)
```

Each layer receives the current `SimState` and the intents from the previous
layer. It can:
- **Pass through** — leave intents unchanged for units it doesn't handle
- **Override** — replace an intent for a specific unit
- **Generate** — create intents for units that don't have one yet

## Layer Responsibilities

| Layer | Scope | Timing | Input |
|-------|-------|--------|-------|
| Squad AI | Team | Every tick | SimState + personality |
| GOAP | Per-unit | Replans on change | SimState + goal defs |
| Ability Evaluator | Per-unit | Urgency > 0.4 | SimState + neural net |
| Transformer | Per-unit | Every tick | Entity + ability tokens |
| Control AI | Per-unit | CC available | SimState + CC cooldowns |

## When to Use Which Layer

- **Squad AI** handles the "big picture" — should the team advance, retreat,
  focus fire, or protect the healer?
- **GOAP** handles individual unit goals — "I need to heal the tank" or "I need
  to chase the fleeing enemy"
- **Ability Evaluator** handles tactical interrupts — "I should use my stun NOW
  because the enemy is casting a big spell"
- **Transformer** handles optimal ability selection from the full ability set
- **Control AI** handles precise CC chain timing — "Don't stun yet, wait for the
  other stun to expire first"

## AI Modules

```
src/ai/
├── squad/          # Team-level AI
├── goap/           # Goal-oriented planning
├── behavior/       # Behavior tree DSL
├── core/
│   ├── ability_eval/       # Neural urgency evaluator
│   ├── ability_transformer/ # Cross-attention transformer
│   └── self_play/          # RL training loop
├── roles/          # Tank/DPS/Support role system
├── personality.rs  # Personality types
├── control.rs      # CC coordination
├── phase.rs        # Combat phase detection
├── utility.rs      # Utility evaluation functions
├── advanced/       # Horde mode, tactical reasoning
└── student/        # Oracle learning
```
