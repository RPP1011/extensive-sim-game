# Factions & Diplomacy

The diplomacy system models relationships between factions on the overworld map.
Player actions and narrative events shift these relationships, affecting available
missions, trade, and alliances.

## Diplomacy State

Faction relationships are tracked as a matrix of sentiment values:

- **Hostile** — factions actively oppose each other
- **Unfriendly** — tensions exist, trade restricted
- **Neutral** — default state
- **Friendly** — trade bonuses, shared intelligence
- **Allied** — joint missions, military support

## Relationship Modifiers

Actions that affect diplomacy:

| Action | Effect |
|--------|--------|
| Complete mission for faction | +reputation with that faction |
| Attack faction territory | -reputation, may trigger hostility |
| Trade with faction | Small +reputation |
| Resolve flashpoint in faction's favor | +reputation |
| Resolve flashpoint against faction | -reputation |
| Companion story choices | Targeted reputation shifts |

## Faction AI

NPC factions have simple strategic AI:
- Expand into unclaimed regions
- Defend threatened territory
- Form alliances against dominant factions
- Spawn flashpoints at contested borders

## Module: `src/game_core/diplomacy_systems.rs`

The diplomacy system runs each campaign turn, updating relationships and
checking for state transitions (peace → war, alliance → betrayal, etc.).
