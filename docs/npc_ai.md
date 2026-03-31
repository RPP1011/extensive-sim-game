# NPC AI System

## Architecture: Flat Action-Space Utility Evaluator

Every 5 ticks, each NPC evaluates all available primitive actions, scores them by utility, and picks the best one. No planning, no goal stacks, no pre-authored behavior chains. Multi-step behavior emerges from repeated single-action decisions.

## Action Space

10 primitive actions an NPC can take each tick:

| Action | Requires | Effect |
|--------|----------|--------|
| **Idle** | nothing | Do nothing |
| **MoveTo(pos)** | nothing | Set move_target toward position |
| **Harvest(resource)** | Adjacent to resource node | Extract commodity into inventory |
| **Eat** | Food in inventory | Consume 0.1 food, restore 60 hunger |
| **Build(blueprint)** | Adjacent to incomplete building + materials | Deposit materials, advance progress |
| **PlaceBlueprint** | 1+ wood in inventory, no home | Create building shell at current position |
| **Work** | Assigned work building nearby | Enter WorkState, produce commodities |
| **Attack(target)** | Hostile within 20 units | Move toward and engage |
| **Flee** | Low HP + nearby hostile | Move away from danger |
| **Trade(target)** | Adjacent friendly NPC | Exchange commodities |

## Utility Scoring

Each action is scored:

```
utility = need_satisfaction × accessibility × personality_mod × emotion_mod
```

### Need Satisfaction

Driven by 6 needs (0-100 each, higher = more satisfied):
- **Hunger**: food consumption rate, restored by Eat action
- **Safety**: proximity to hostiles, threat level
- **Shelter**: requires home_building_id, decays without home
- **Social**: proximity to other NPCs
- **Purpose**: meaningful work/contribution
- **Esteem**: achievements, prosperity

`urgency = (100 - need_value) / 100` — lower need = higher urgency.

### Accessibility

- **Distance decay**: `1.0 / (1.0 + distance / 10.0)` — closer targets score higher
- **Binary checks**: has food? has wood? adjacent to resource? within harvest range (5 units)?
- **MoveTo inherits utility**: if Harvest(bush) scores 0.7 but bush is 30 units away, MoveTo(bush) scores `0.7 × 0.8 × distance_factor`

### Personality Modifiers

5 traits (0.0-1.0 each):
- **Ambition**: boosts Work, Build (+20% if > 0.6)
- **Risk tolerance**: boosts Attack, reduces Flee weight
- **Compassion**: boosts helping others build
- **Social drive**: boosts Trade, social interactions
- **Curiosity**: boosts Harvest exploration

### Emotion Modifiers

6 emotions (0.0-1.0 each):
- **Anger** > 0.3: boosts Attack utility
- **Fear** > 0.3: boosts Flee utility
- **Grief** > 0.3: dampens ALL action utilities (depression)
- **Anxiety**: reduces risk-taking
- **Joy**: no direct modifier (increases need satisfaction indirectly)
- **Pride**: boosts Work/Build

## Example: Hungry NPC Near Berry Bush

| Action | need_sat | access | personality | emotion | total |
|--------|----------|--------|-------------|---------|-------|
| Eat (has food) | 0.9 | 1.0 | 1.0 | 1.0 | **0.90** |
| Harvest(bush) | 0.7 | 0.8 | 1.0 | 1.0 | 0.56 |
| MoveTo(bush) | 0.5 | 1.0 | 1.0 | 1.0 | 0.50 |
| Work | 0.3 | 0.2 | 1.1 | 1.0 | 0.07 |
| Idle | 0.0 | 1.0 | 1.0 | 1.0 | 0.00 |

NPC picks **Eat**. Next tick, hunger satisfied, shelter becomes most urgent need → NPC starts gathering wood.

## Emergent Behavior

No behavior is hand-coded. These emerge from utility:

1. **Survival loop**: Gather food → eat → gather wood → build house
2. **Specialization**: NPCs near mines harvest iron, NPCs near forests harvest wood
3. **Community building**: Housed NPCs help build others' blueprints (purpose utility)
4. **Migration**: NPCs move toward resource-rich areas (distance × need scoring)
5. **Combat**: Hostile entities trigger Attack when safety is low + anger is high
6. **Flee**: Low HP + fear → flee utility exceeds attack utility

## Construction System

### Blueprints

1. NPC with 1+ wood and no home places a **building shell** (0% progress)
2. The shell is a real Entity with `BuildingData.construction_progress = 0.0`
3. Recipe: `BuildingType::House.build_cost() = (5.0 wood, 0.0 iron)`

### Multi-Worker Contribution

1. Any NPC near an incomplete building with matching materials can contribute
2. Each tick: NPC deposits up to 1.0 of each needed commodity from their inventory
3. `progress = total_deposited / recipe_cost`
4. Multiple workers deposit in parallel → faster construction
5. On completion: first finisher becomes owner, homeless builders get assigned

### Recipe Examples

| Building | Wood | Iron | Residential Cap |
|----------|------|------|-----------------|
| House | 5 | 0 | 4 |
| Longhouse | 10 | 2 | 8 |
| Forge | 5 | 5 | 0 |
| Temple | 12 | 5 | 2 |
| GuildHall | 15 | 8 | 2 |

## Movement System

Single unified mover: `advance_movement()` in `systems/movement.rs`.

- All action systems set `entity.move_target = Some(pos)`
- Movement system moves entity toward target at `move_speed × move_speed_mult × dt`
- Clears target on arrival (within 1.5 units)
- CC-aware: Stun/Root status effects block movement
- No other system touches `entity.pos` directly (except combat Move deltas)

## Monster AI

Same action space, simpler evaluation:
- **Flee**: HP < 20% of max → utility 0.9, move away from nearest hostile
- **Attack**: NPC within aggro range (20 units) → utility 0.8
- **Idle**: default → utility 0.3

## Needs Drift (systems/agent_inner.rs)

Every 10 ticks:
- **Hunger**: -1.5/tick at settlement, -3.0 in wilderness
- **Shelter**: +5.0/tick with home building, -1.0 homeless at settlement, -3.0 in wilderness
- **Safety**: driven by threat level + grid hostility
- **Social**: -0.2/tick when isolated
- **Esteem**: +0.5 if prosperous settlement

## Files

| File | Purpose |
|------|---------|
| `systems/action_eval.rs` | Action evaluator + executor |
| `systems/movement.rs` | Unified movement system |
| `systems/agent_inner.rs` | Needs drift, emotion decay, morale recovery |
| `systems/work.rs` | Work state machine (produce commodities at buildings) |
| `systems/buildings.rs` | Building construction, assignment |
| `systems/resource_nodes.rs` | Resource spawning, regrowth |
| `state.rs` | NpcData, Needs, Personality, Emotions, NpcAction |
