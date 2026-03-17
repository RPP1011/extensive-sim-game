# Summary

[Introduction](./introduction.md)

# Architecture

- [Architecture Overview](./architecture/overview.md)
    - [Three-Layer Simulation](./architecture/three-layers.md)
    - [Workspace & Crate Map](./architecture/workspace.md)
    - [Module Map](./architecture/module-map.md)

# The Simulation Engine

- [Core Simulation](./simulation/core.md)
    - [The `step()` Function](./simulation/step.md)
    - [State & Types](./simulation/state.md)
    - [Determinism Contract](./simulation/determinism.md)
    - [Events & Logging](./simulation/events.md)

# The Effect System

- [Effect System](./effects/overview.md)
    - [Ability Definitions](./effects/ability-defs.md)
    - [The Five Dimensions](./effects/five-dimensions.md)
    - [Effect Dispatch & Resolution](./effects/dispatch.md)

# The Ability DSL

- [Ability DSL](./dsl/overview.md)
    - [Syntax Reference](./dsl/syntax.md)
    - [Parser Pipeline](./dsl/parser-pipeline.md)
    - [Writing New Abilities](./dsl/writing-abilities.md)

# AI Decision Pipeline

- [AI Overview](./ai/overview.md)
    - [Squad AI](./ai/squad.md)
    - [GOAP Planner](./ai/goap.md)
    - [Behavior DSL](./ai/behavior.md)
    - [Ability Evaluator (Neural)](./ai/ability-evaluator.md)
    - [Ability Transformer](./ai/ability-transformer.md)
    - [Roles & Personality](./ai/roles.md)

# Game Systems

- [Campaign & Overworld](./game/campaign.md)
    - [Factions & Diplomacy](./game/factions.md)
    - [Flashpoints](./game/flashpoints.md)
    - [Save System](./game/save.md)
- [Mission System](./game/mission.md)
    - [Room Generation](./game/room-gen.md)
    - [Objectives](./game/objectives.md)

# Machine Learning

- [ML Training Pipeline](./ml/overview.md)
    - [Entity Encoder](./ml/entity-encoder.md)
    - [Self-Play & RL](./ml/self-play.md)
    - [Burn Model Integration](./ml/burn.md)
    - [Python Training Scripts](./ml/python-training.md)

# Hero Content

- [Hero Templates](./heroes/overview.md)
    - [Template File Format](./heroes/format.md)
    - [LoL Champion Imports](./heroes/lol-imports.md)

# Tooling

- [CLI & xtask](./tooling/cli.md)
    - [Scenario Runner](./tooling/scenarios.md)
    - [sim_bridge Protocol](./tooling/sim-bridge.md)
    - [Map & Room Generation](./tooling/map-gen.md)

# Development

- [Testing](./development/testing.md)
- [Contributing](./development/contributing.md)
