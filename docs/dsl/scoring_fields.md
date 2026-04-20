# Scoring field-id mapping

The compiler-emitted scoring table (`engine_rules::scoring`) encodes every
predicate as a `PredicateDescriptor` with a `field_id: u16`. This document
pins the mapping from DSL field reference (e.g. `self.hp_pct`) to the
integer id the scorer's `read_field` function dispatches on.

Changing this table breaks the committed `SCORING_TABLE` constants and bumps
`SCORING_HASH`. Treat every row as a stable contract.

## Agent-local fields (`self.*`)

| `field_id` | DSL reference       | Engine accessor                              | Notes |
|-----------:|---------------------|----------------------------------------------|-------|
|          0 | `self.hp`           | `state.agent_hp(agent).unwrap_or(0.0)`       | raw hit points |
|          1 | `self.max_hp`       | `state.agent_max_hp(agent).unwrap_or(1.0)`   | defaults to 1.0 so `hp_pct` is well-defined |
|          2 | `self.hp_pct`       | `hp / max_hp` (derived)                      | 0.0..=1.0 normalised |
|          3 | `self.shield_hp`    | `state.agent_shield_hp(agent).unwrap_or(0.0)`| |
|          4 | `self.attack_range` | `state.agent_attack_range(agent).unwrap_or(2.0)` | |
|          5 | `self.hunger`       | `state.agent_hunger(agent).unwrap_or(0.0)`   | 0.0 sated … 1.0 starving |
|          6 | `self.thirst`       | `state.agent_thirst(agent).unwrap_or(0.0)`   | 0.0 sated … 1.0 parched |
|          7 | `self.fatigue`      | `state.agent_rest_timer(agent).unwrap_or(0.0)` | Engine slot is named `rest_timer`; DSL surface calls it `fatigue` because that matches the urge to Rest. 0.0 rested … 1.0 exhausted. |
|          8 | `self.personality.aggression`    | placeholder `0.0` (task 141) | Personality SoA not yet wired; reads return `0.0`. Reserving the id lets scoring rows reference the field today and pick up live values without a schema bump. |
|          9 | `self.personality.social_drive`  | placeholder `0.0` (task 141) | |
|         10 | `self.personality.ambition`      | placeholder `0.0` (task 141) | |
|         11 | `self.personality.altruism`      | placeholder `0.0` (task 141) | |
|         12 | `self.personality.curiosity`     | placeholder `0.0` (task 141) | |

## Pair fields (PairField predicates)

Reserved range `field_id >= 0x8000`. Not emitted at milestone 5 — pair
predicates (`self.pos vs other.pos`, `self.team != other.team`, …) land
with the verb / mask expansion.

## Reserved

`field_id == u16::MAX` is reserved for the "invalid field" sentinel. The
scorer returns `f32::NAN` for it so mistakes show up as always-false
numeric comparisons rather than silent zero reads.
