# Roles & Personality

The role and personality systems provide high-level behavioral archetypes that
inform the lower-level AI layers.

## Roles

The role system (`src/ai/roles/`) classifies units into tactical roles:

| Role | Behavior |
|------|----------|
| **Tank** | Engage enemies, absorb damage, protect backline |
| **Melee DPS** | Focus priority targets, maximize damage output |
| **Ranged DPS** | Maintain distance, kite, focus fire |
| **Healer** | Prioritize ally survival, position safely |
| **Support** | Buff allies, debuff enemies, CC priority targets |
| **Assassin** | Flank, burst squishy targets, escape |

Roles influence:
- **Target selection** — tanks taunt nearest enemies; assassins seek squishy
  backline targets
- **Positioning** — healers stay behind the frontline; tanks stay in front
- **Ability priority** — healers prioritize healing abilities; DPS prioritize
  damage

## Personality Types

From `src/ai/personality.rs`:

Personality affects team-level behavior through the squad AI:

| Trait | Low (0.0) | High (1.0) |
|-------|-----------|------------|
| Aggression | Defensive, wait for openings | All-in, chase kills |
| Focus Fire | Spread damage | Concentrate on one target |
| Protect Healer | Ignore healer safety | Peel for healer aggressively |
| Ability Usage | Prefer basic attacks | Use abilities eagerly |

## Combat Phase Detection

The `phase.rs` module detects the current phase of combat:

- **Opening** — teams approaching, abilities not yet used
- **Engage** — first contact, big cooldowns being used
- **Sustained** — mid-fight, trading damage and healing
- **Cleanup** — one side has clear advantage, chasing stragglers
- **Desperate** — losing badly, emergency measures

Phase detection feeds into personality-based decision making. An aggressive
personality might initiate the engage phase earlier; a defensive personality
might extend the opening phase by poking at range.
