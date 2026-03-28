# Ability DSL & Class DSL Reference

## Overview

The Ability DSL is a text-based language for defining hero abilities, passive effects, and campaign actions. It provides a concise, readable way to express complex game mechanics as plain data, avoiding closures or imperative code.

### Pipeline

```
.ability file (DSL text)
    --> parser (winnow-based, parser.rs + parse_effects.rs + parse_delivery.rs)
    --> AST (ast.rs: AbilityNode, PassiveNode, EffectNode, ...)
    --> lowering (lower.rs + lower_effects.rs + lower_delivery.rs)
    --> AbilityDef / PassiveDef (defs.rs)
    --> runtime dispatch (apply_effect.rs)
```

### Five Composable Dimensions

Every ability is described along five orthogonal axes:

| Dimension | What it controls | Examples |
|-----------|-----------------|----------|
| **Effect** (WHAT) | The gameplay action | `damage`, `heal`, `stun`, `buff` |
| **Area** (WHERE) | Spatial extent | `in circle(3.0)`, `in cone(4.0, 60)` |
| **Delivery** (HOW) | How effects reach targets | `deliver projectile`, `deliver zone` |
| **Trigger/Condition** (WHEN) | When/if effects fire | `when target_hp_below(30%)`, trigger: `on_kill` |
| **Tags** | Power levels and element typing | `[FIRE: 60]`, `[PHYSICAL: 50]` |

---

## Ability DSL Reference

### File Structure

A `.ability` file contains one or more top-level blocks:

```
ability Name { ... }
passive Name { ... }
```

Comments use `//` or `#`. Both are stripped before parsing.

### Ability Block Syntax

```
ability Name {
    // Header properties (key: value pairs, comma-separated on one line)
    target: enemy, range: 5.0
    cooldown: 8s, cast: 300ms
    hint: damage

    // Effect lines (one per line)
    damage 55 in circle(3.0) [FIRE: 60] when target_hp_below(30%)

    // Optional delivery block
    deliver projectile { speed: 8.0 } {
        on_hit {
            damage 40 [MAGIC: 50]
        }
    }

    // Optional morph block
    morph into {
        damage 80 [FIRE: 80]
    } for 5s

    // Optional recast blocks
    recast 1 {
        damage 30
    }
}
```

### Header Properties

| Property | Aliases | Type | Description |
|----------|---------|------|-------------|
| `target` | `targeting` | identifier | Targeting mode (see below) |
| `range` | | number | Ability range in units |
| `cooldown` | | duration | Time between uses |
| `cast` | | duration | Cast time before effects apply |
| `hint` | `ai_hint` | identifier | AI categorization hint |
| `cost` | `resource_cost` | number | Resource cost to cast |
| `zone_tag` | | string | Element tag for zone-reaction combos |
| `charges` | | number | Max charges (ammo system). 0 = normal cooldown |
| `recharge` | | duration | Time per charge to regenerate |
| `toggle` | | flag | Ability is a toggle (on/off) |
| `toggle_cost` | | number | Resource cost per second while toggled on |
| `recast` | | number | Number of recasts before cooldown |
| `recast_window` | | duration | Window to recast before cooldown starts |
| `unstoppable` | | flag | Caster is CC-immune during cast |
| `form` | | string | Form group this ability belongs to |
| `swap_form` | | string | Casting swaps all abilities with matching form |
| `requires` | `requires_participants` | number | Number of participants (2+ = combination ability) |
| `requires_class` | | string | Required class for at least one participant |

Flag properties (`unstoppable`, `toggle`) can be written without a value:
```
unstoppable
toggle
```

### Targeting Modes

#### Combat Targeting
| Mode | Description |
|------|-------------|
| `enemy` | Target a single enemy unit |
| `ally` | Target a single ally unit |
| `self` | Target the caster |
| `self_aoe` | AoE centered on caster |
| `ground` | Target a ground position |
| `direction` | Target a direction (skillshot) |
| `vector` | Click-drag vector (start + direction) |
| `global` | All enemies on map, ignores range |

#### Campaign Targeting
| Mode | Description |
|------|-------------|
| `faction` | Target a faction |
| `region` | Target a region |
| `market` | Target a market |
| `party` | Target the party |
| `guild` | Target the guild |
| `adventurer` | Target an adventurer |
| `location` | Target a location |

### AI Hint Values

Combat: `damage`, `heal`, `crowd_control`, `defense`, `utility`

Campaign: `economy`, `diplomacy`, `stealth`, `leadership`, `utility`, `defense`, `heal`

### Duration Syntax

| Format | Meaning | Example |
|--------|---------|---------|
| `Nms` | Milliseconds | `300ms` |
| `Ns` | Seconds (converted to ms internally) | `5s` = 5000ms |
| `Nt` | Campaign ticks (raw count, not converted) | `500t` |
| bare number | Defaults to ms | `3000` = 3000ms |
| `while_alive` | Permanent while unit lives (`u32::MAX` ms) | `for while_alive` |

---

### Effect Line Syntax

```
effect_name [args...] [for duration] [in area] [tags] [when condition] [else effect] [targeting filter] [scales_with stat] [+ N% stat] [stacking mode] [chance N]
```

All modifiers are optional and can appear in any order (after the effect name and its positional args).

### Effect Types: Combat

#### Direct Damage & Healing

| Effect | Args | Description |
|--------|------|-------------|
| `damage` | `amount` | Instant damage |
| `damage` | `amount/interval` `for duration` | Damage over time (DoT). Interval: `N/tick` (1s), `N/Ns`, `N/Nms` |
| `heal` | `amount` | Instant heal |
| `heal` | `amount/interval` `for duration` | Heal over time (HoT) |
| `shield` | `amount` `for duration` | Absorb shield |
| `self_damage` | `amount` | Damage the caster |
| `execute` | `threshold%` | Kill target if HP below threshold (default 15%) |

#### Crowd Control (CC)

| Effect | Args | Description |
|--------|------|-------------|
| `stun` | `duration` | Cannot act |
| `root` | `duration` | Cannot move, can still act |
| `silence` | `duration` | Cannot use abilities |
| `slow` | `factor` `for duration` | Reduce move speed (factor: 0-1 proportion) |
| `fear` | `duration` | Forced to flee from caster |
| `taunt` | `duration` | Forced to attack the caster |
| `charm` | `duration` | Switches team temporarily |
| `confuse` | `duration` | Random behavior |
| `polymorph` | `duration` | Cannot act, reduced stats |
| `banish` | `duration` | Untargetable + cannot act |
| `suppress` | `duration` | Hard CC, cannot be cleansed |
| `grounded` | `duration` | Prevents dashes/blinks/movement abilities |
| `blind` | `miss_chance` `for duration` | Chance to miss auto attacks |
| `knockback` | `distance` | Push target away from caster |
| `pull` | `distance` | Pull target toward caster |
| `swap` | | Swap positions with target |

#### Positioning & Mobility

| Effect | Args | Description |
|--------|------|-------------|
| `dash` | `distance` | Dash in current direction |
| `dash` | `to_target` | Dash toward ability target |
| `dash` | `to_position` | Dash toward target position |
| `blink` | `distance` | Instant teleport (ignores terrain/grounded) |

#### Buffs & Debuffs

| Effect | Args | Description |
|--------|------|-------------|
| `buff` | `stat` `factor` `for duration` | Increase a stat multiplicatively |
| `debuff` | `stat` `factor` `for duration` | Decrease a stat multiplicatively |
| `damage_modify` | `factor` `for duration` | Multiply all damage dealt |
| `lifesteal` | `percent` `for duration` | Heal for % of damage dealt |
| `reflect` | `percent` `for duration` | Reflect % of damage taken back to attacker |

Common stat names: `damage_output`, `move_speed`, `attack_speed`, `armor`, `magic_resist`, `cooldown_reduction`

#### Shields & Healing Modifiers

| Effect | Args | Description |
|--------|------|-------------|
| `overheal_shield` | `for duration` | Overhealing becomes a shield (default 100% conversion) |
| `absorb_to_heal` | `shield_amount` `for duration` | Shield that heals the target when it expires |
| `shield_steal` | `amount` | Steal shield from target |
| `resurrect` | `hp_percent` | Revive a dead ally at % HP (default 50%) |

#### Status Interaction

| Effect | Args | Description |
|--------|------|-------------|
| `dispel` | `[TAG: ...]` | Remove effects matching given tags |
| `immunity` | `"status"...` `for duration` | Immune to named statuses |
| `detonate` | `multiplier` | Consume and detonate all DoTs on target (default 1.0x) |
| `status_transfer` | `[steal]` | Transfer debuffs to target (or steal buffs) |
| `status_clone` | `[max N]` | Copy buffs from target (max_count default 3) |
| `death_mark` | `duration` `damage_percent` | Mark: detonates for % of damage taken during mark (default 50%) |

#### Stacks

| Effect | Args | Description |
|--------|------|-------------|
| `apply_stacks` | `"name"` `count` `[max N]` `[for duration]` | Apply named stacks on target |

#### Complex Mechanics

| Effect | Args | Description |
|--------|------|-------------|
| `stealth` | `duration` `[break_on_damage]` | Become invisible |
| `leash` | `max_range` `for duration` | Tether target to current position |
| `link` | `share_percent` `for duration` | Link two units (share damage, default 50%) |
| `redirect` | `duration` `[charges N]` | Redirect damage from target to caster (default 3 charges) |
| `rewind` | `lookback_ms` | Restore target to state N ms ago (default 3000) |
| `cooldown_modify` | `amount_ms` `["ability_name"]` | Reduce cooldown by N ms (negative = reduce) |
| `duel` | `duration` | Force 1v1 duel, others cannot interfere |
| `on_hit_buff` | `duration` `{ children }` | Next auto attacks apply child effects |
| `summon` | `"template"` `[xN]` | Summon unit(s) from template. `"clone"` = copy caster |
| `command_summons` | `[speed]` | Move directed summons toward target (default speed 8.0) |
| `obstacle` | `width` `x` `height` | Create terrain obstacle |
| `projectile_block` | `duration` | Block enemy projectiles in area |
| `attach` | `duration` | Attach to ally (become untargetable, move with them) |
| `evolve_ability` | `index` | Permanently evolve an ability by index |

#### Meta-Effects

| Effect | Args | Description |
|--------|------|-------------|
| `refresh_cooldowns` | | Reset all ability cooldowns to 0 |
| `refresh_cooldown` | `index` | Reset a specific ability's cooldown by index |
| `amplify` | `multiplier` `charges` | Next N casts have multiplied effectiveness (default 1.5x, 1 charge) |
| `echo` | `charges` | Next ability fires twice (default 1 charge) |
| `extend_durations` | `amount_ms` | Extend all active buff/debuff durations |
| `instant_cast` | `charges` | Next ability has 0 cast time (default 1 charge) |
| `free_cast` | `charges` | Next ability costs 0 resources (default 1 charge) |
| `spell_shield` | `charges` | Block the next enemy ability entirely (default 1 charge) |
| `mana_burn` | `cost_multiplier` `for duration` | Increase target's ability costs (default 2.0x) |
| `cooldown_lock` | `duration` | Lock target out of abilities (uncleansable silence) |

#### Recursive Effects

| Effect | Args | Description |
|--------|------|-------------|
| `on_hit_cast` | `"ability_name"` | Trigger a named ability when this effect lands |
| `grant_ability` | `"ability_name"` `for duration` | Give target a temporary ability (default 10s) |
| `cast_copy` | `[last_used]` | Copy the last ability used by the target |
| `evolve_after` | `cast_count` | After N casts, permanently evolve this ability (default 5) |

### Effect Types: Campaign

Campaign effects no-op in the combat sim and are dispatched by the campaign layer.

#### Economy

| Effect | Args | Description |
|--------|------|-------------|
| `corner_market` | `"commodity"` `for Nt` | Monopolize a commodity |
| `forge_trade_route` | `income_per_tick` `for Nt` | Create trade route |
| `appraise` | | Identify item values |
| `golden_touch` | `for Nt` | Increased gold from all sources |
| `trade_embargo` | `for Nt` | Block trade with target faction |
| `silver_tongue` | | Better deal terms |
| `market_maker` | `for Nt` | Control market prices |
| `trade_empire` | `income_per_tick` | Passive income generation |

#### Diplomacy

| Effect | Args | Description |
|--------|------|-------------|
| `demand_audience` | | Force faction interaction |
| `ceasefire` | `for Nt` | Temporary peace |
| `destabilize` | `instability` `for Nt` | Reduce faction stability (default 0.3) |
| `broker_alliance` | `for Nt` | Create alliance |
| `subvert_loyalty` | | Undermine faction loyalty |
| `treaty_breaker` | | Break existing treaties |
| `shatter_alliance` | | Destroy an alliance |

#### Information

| Effect | Args | Description |
|--------|------|-------------|
| `reveal` | `count` | Reveal hidden threats |
| `prophecy` | `count` | Reveal upcoming events |
| `beast_lore` | | Identify creature weaknesses |
| `read_the_room` | | Read faction intentions |
| `all_seeing_eye` | | Reveal everything |
| `decipher` | | Decode encrypted information |
| `trap_sense` | | Detect nearby traps |
| `sapper_eye` | | Detect fortification weaknesses |

#### Leadership

| Effect | Args | Description |
|--------|------|-------------|
| `rally` | `morale_restore` | Restore party morale (default 0.5) |
| `rallying_cry` | `morale_restore` | Restore party morale (default 0.5) |
| `inspire` | `morale_boost` `for Nt` | Temporary morale increase (default 0.2) |
| `field_command` | `for Nt` | Enhanced party coordination |
| `coordinated_strike` | | All party members attack together |
| `war_cry` | `for Nt` | Buff party combat stats |

#### Stealth / Movement

| Effect | Args | Description |
|--------|------|-------------|
| `ghost_walk` | `for Nt` | Phase through obstacles |
| `shadow_step` | `for Nt` | Teleport short distance |
| `silent_movement` | | Move without detection |
| `hidden_camp` | `for Nt` | Invisible camp |
| `vanish` | | Immediate stealth |
| `distraction` | `for Nt` | Misdirect enemies |

#### Territory

| Effect | Args | Description |
|--------|------|-------------|
| `claim_territory` | | Claim unclaimed territory |
| `fortify` | `for Nt` | Strengthen defenses |
| `sanctuary` | `for Nt` | Create safe zone |
| `plague_ward` | `for Nt` | Area denial ward |
| `safe_house` | `for Nt` | Hidden safe location |

#### Supernatural / Body

| Effect | Args | Description |
|--------|------|-------------|
| `blood_oath` | `stat_bonus` | Permanent stat boost at cost (default 0.15) |
| `unbreakable` | | Cannot be killed for one lethal hit |
| `life_eternal` | | Permanent immortality effect |
| `purify` | | Cleanse all negative effects |
| `name_the_nameless` | | Name an unknown entity |
| `forbidden_knowledge` | | Gain forbidden lore |

#### Passive / Skill-State

| Effect | Args | Description |
|--------|------|-------------|
| `field_triage` | `heal_rate_multiplier` | Improved healing rate (default 1.5) |
| `inspiring_presence` | `morale_boost` | Passive morale aura (default 0.2) |
| `battle_instinct` | | Combat awareness bonus |
| `quick_study` | | Faster learning |
| `forage` | `supply_per_tick` | Passive supply generation (default 0.1) |
| `track_prey` | | Track creatures |
| `field_repair` | | Repair equipment in field |
| `stabilize_ally` | | Prevent ally from dying |

#### Higher Tier

| Effect | Args | Description |
|--------|------|-------------|
| `disinformation` | `for Nt` | Spread false information |
| `accelerated_study` | `for Nt` | Faster research |
| `take_the_blow` | `for Nt` | Absorb damage for ally |
| `hold_the_line` | | Party cannot be routed |
| `forgery` | | Create counterfeit documents |
| `masterwork_craft` | | Create masterwork items |
| `intel_gathering` | | Continuous intelligence |
| `master_armorer` | | Superior equipment |
| `forge_artifact` | | Create an artifact |

#### Legendary / Mythic

| Effect | Description |
|--------|-------------|
| `living_legend` | Permanent fame and influence |
| `rewrite_history` | Alter past events |
| `the_last_word` | Final decisive action |
| `wealth_of_nations` | Control entire economies |
| `omniscience` | Know everything |
| `immortal_moment` | Freeze a moment in time |
| `claim_by_right` | Claim any territory by authority |
| `rewrite_the_record` | Alter official records |

### Campaign Primitives

Generic composable building blocks for campaign effects:

| Effect | Args | Description |
|--------|------|-------------|
| `modify_stat` | `"entity"` `"property"` `amount` `[for Nt]` | Modify a numeric property. Entities: `adventurer`, `party`, `guild`, `faction`, `region`. Ops: add/multiply/set |
| `set_flag` | `"entity"` `"flag"` `[for Nt]` | Set a boolean flag on an entity |
| `reveal_info` | `"target_type"` `"scope"` | Reveal hidden information. Scopes: `faction_stance`, `enemy_weaknesses`, `upcoming_events`, `map`, `all` |
| `create_entity` | `"entity_type"` `"subtype"` `[for Nt]` | Create a new entity. Types: `location`, `trade_route`, `agreement`, `item`, `region_modifier` |
| `destroy_entity` | `"target_type"` | Remove/destroy something. Types: `disease`, `agreement`, `war_state`, `buff`, `event` |
| `transfer` | `"from_entity"` `"to_entity"` `"property"` `amount` | Transfer value between entities |

---

### Delivery Blocks

Delivery blocks control how effects reach their targets. Syntax:

```
deliver method { params } {
    on_hit {
        // effects applied when delivery hits
    }
    on_arrival {
        // effects on projectile arrival at position
    }
    on_complete {
        // effects when delivery completes (tethers)
    }
}
```

If no hooks are specified, the outer `{ }` can be omitted.

#### Delivery Methods

| Method | Params | Description |
|--------|--------|-------------|
| `projectile` | `speed`, `pierce`, `width` | Fires a projectile at the target. `pierce` = passes through units |
| `chain` | `bounces`, `range`, `falloff` | Bounces between targets. `falloff` = damage reduction per bounce |
| `zone` | `duration`, `tick` | Persistent ground area, ticks periodically |
| `trap` | `duration`, `trigger_radius`, `arm_time` | Placed on ground, activates when enemy enters radius |
| `channel` | `duration`, `tick` | Channeled over time (caster is stationary) |
| `tether` | `max_range`, `tick` | Connects caster to target, breaks if range exceeded |

Param values use the same duration syntax (e.g. `duration: 4s`, `tick: 1s`).

---

### Area Modifiers

Area modifiers follow the `in` keyword and control the spatial extent of an effect:

| Area | Syntax | Args | Description |
|------|--------|------|-------------|
| Circle | `in circle(radius)` | 1 float | AoE circle |
| Cone | `in cone(radius, angle_deg)` | 2 floats | Cone in facing direction |
| Line | `in line(length, width)` | 2 floats | Rectangular line |
| Ring | `in ring(inner_radius, outer_radius)` | 2 floats | Donut shape |
| Spread | `in spread(radius, max_targets)` | 1-2 | Spread to nearby targets (with optional cap) |
| Self | `in self` | none | Apply to caster only (within AoE context) |

---

### Tags

Tags assign element/power metadata to effects. Syntax:

```
[ELEMENT: power_level]
[FIRE: 60, MAGIC: 40]
```

Tags serve two purposes:
1. **Element typing** -- classify the damage type thematically
2. **Power budgeting** -- effect tags assert power; unit resistance tags resist if resistance >= effect's tag value

Common tags: `PHYSICAL`, `MAGIC`, `FIRE`, `ICE`, `DARK`, `HOLY`, `POISON`, `CROWD_CONTROL`, `KNOCKBACK`, `FEAR`

---

### Conditions

Conditions gate an effect with `when condition`:

```
damage 80 when target_hp_below(30%)
heal 50 when caster_hp_below(25%)
```

#### Combat Conditions

| Condition | Args | Description |
|-----------|------|-------------|
| `target_hp_below(N%)` | percent | Target HP below threshold |
| `target_hp_above(N%)` | percent | Target HP above threshold |
| `caster_hp_below(N%)` | percent | Caster HP below threshold |
| `caster_hp_above(N%)` | percent | Caster HP above threshold |
| `target_is_stunned` | | Target is stunned |
| `target_is_slowed` | | Target is slowed |
| `target_is_rooted` | | Target is rooted |
| `target_is_silenced` | | Target is silenced |
| `target_is_feared` | | Target is feared |
| `target_is_taunted` | | Target is taunted |
| `target_is_banished` | | Target is banished |
| `target_is_stealthed` | | Target is stealthed |
| `target_is_charmed` | | Target is charmed |
| `target_is_polymorphed` | | Target is polymorphed |
| `target_has_tag("tag")` | string | Target has named tag |
| `hit_count_above(N)` | number | Ability has hit N times |

#### Campaign Conditions

| Condition | Args | Description |
|-----------|------|-------------|
| `faction_hostile` | | Target faction is hostile |
| `at_war` | | Currently at war |
| `crisis_active` | | A crisis is underway |
| `gold_above(N)` | number | Gold above threshold |
| `gold_below(N)` | number | Gold below threshold |
| `outnumbered` | | Party is outnumbered |
| `ally_injured` | | An ally is injured |
| `alone` | | Unit is alone |
| `near_death` | | Unit is near death |

#### Compound Conditions

```
when and(target_hp_below(50%), target_is_stunned)
when or(caster_hp_below(20%), ally_injured)
when not(target_is_stealthed)
```

#### Else Clause

Effects can have an `else` branch:

```
damage 80 when target_hp_below(30%) else damage 30
```

---

### Targeting Filters

Targeting filters narrow which entities within an area are affected. Syntax:

```
buff damage_output 0.3 for 10s in circle(8.0) targeting under_command
heal 50 in circle(10.0) targeting injured
buff armor 0.3 for 10s in circle(6.0) targeting has_class("knight")
```

| Filter | Args | Description |
|--------|------|-------------|
| `under_command` | | Units under the caster's command |
| `loyalty_above(N)` | number | Units with loyalty above threshold |
| `loyalty_below(N)` | number | Units with loyalty below threshold |
| `has_class("name")` | string | Units with a specific class |
| `level_above(N)` | number | Units above a level |
| `level_below(N)` | number | Units below a level |
| `has_status("name")` | string | Units with a specific status |
| `faction("name")` | string | Units belonging to a faction |
| `injured` | | Units with injury > 20% |
| `healthy` | | Units with injury < 20% |

---

### Scaling

#### Inline Scaling (+ N% stat)

```
damage 50 + 10% target_max_hp
heal 30 + 15% caster_missing_hp consume cap 100
```

Multiple scaling terms stack additively:
```
damage 20 + 5% caster_attack_damage + 3% target_missing_hp
```

Options after each scaling term:
- `consume` -- remove the referenced stacks after reading
- `cap N` -- hard cap on this term's contribution

#### scales_with Keyword

Shorthand for `+ 100% stat`:
```
buff damage_output 0.01 for 10s scales_with kingdom_size
```

#### Available Stat References

| Stat Reference | Description |
|----------------|-------------|
| `target_max_hp` | Target's maximum HP |
| `target_current_hp` | Target's current HP |
| `target_missing_hp` | Target's missing HP (max - current) |
| `caster_max_hp` | Caster's maximum HP |
| `caster_current_hp` | Caster's current HP |
| `caster_missing_hp` | Caster's missing HP |
| `caster_attack_damage` | Caster's attack damage |
| `target_stacks("name")` | Stack count on target |
| `caster_stacks("name")` | Stack count on caster |
| `kingdom_size` | Campaign: kingdom size |
| `army_size` | Campaign: army size |
| `faction_territory` | Campaign: faction territory |
| `guild_reputation` | Campaign: guild reputation |
| `adventurer_count` | Campaign: number of adventurers |
| `loyalty_average` | Campaign: average loyalty |
| `party_size` | Campaign: party size |
| `guild_gold` / `gold` | Campaign: guild gold |
| `caster_level` / `level` | Campaign: caster level |

---

### Stacking Modes

Controls behavior when the same effect is applied multiple times:

```
buff damage_output 0.2 for 5s stacking refresh
```

| Mode | Description |
|------|-------------|
| `refresh` | Reset duration to full (default) |
| `extend` | Add duration to remaining |
| `strongest` | Keep the stronger effect |
| `stack` | Stack independently |

---

### Chance

Probability that an effect fires when its condition passes (0.0-1.0):

```
damage 50 chance 0.3
```

A value of `0.0` (default) is treated as always-fires.

---

### Passive Block Syntax

```
passive Name {
    trigger: trigger_type
    cooldown: 5s
    range: 5.0

    // Effect lines
    heal 12
}
```

#### Passive Properties

| Property | Type | Description |
|----------|------|-------------|
| `trigger` | identifier | What causes the passive to fire (required) |
| `cooldown` | duration | Internal cooldown between activations |
| `range` | number | Range for range-dependent triggers |

#### Triggers

##### Combat Triggers

| Trigger | Args | Description |
|---------|------|-------------|
| `on_damage_dealt` | | When this unit deals damage |
| `on_damage_taken` | | When this unit takes damage |
| `on_kill` | | When this unit kills an enemy |
| `on_death` | | When this unit dies |
| `on_ability_used` | | When this unit uses an ability |
| `on_shield_broken` | | When this unit's shield breaks |
| `on_stun_expire` | | When a stun on this unit expires |
| `on_heal_received` | | When this unit is healed |
| `on_status_applied` | | When any status is applied to this unit |
| `on_status_expired` | | When a status expires on this unit |
| `on_resurrect` | | When this unit is resurrected |
| `on_dodge` | | When this unit dodges |
| `on_reflect` | | When this unit reflects damage |
| `on_auto_attack` | | When this unit auto attacks |
| `on_hp_below(N%)` | percent | When HP drops below threshold |
| `on_hp_above(N%)` | percent | When HP rises above threshold |
| `on_ally_damaged(range: N)` | float | When a nearby ally takes damage |
| `on_ally_killed(range: N)` | float | When a nearby ally dies |
| `periodic(Ns)` | duration | Fires every N (e.g. `periodic(3s)`) |
| `on_stack_reached("name", N)` | string, count | When named stacks reach count |

##### Campaign Triggers

| Trigger | Args | Description |
|---------|------|-------------|
| `on_trade` | | On completing a trade |
| `on_quest_complete` | | On quest completion |
| `on_faction_change` | | On faction relation change |
| `on_level_up` | | On leveling up |
| `on_crisis_start` | | On crisis beginning |
| `on_ability_use` | | On any ability use |
| `periodic_tick(Nt)` | ticks | Fires every N campaign ticks |

---

### Morph Block

Temporarily replace an ability with a different set of effects:

```
ability Transform {
    target: self
    cooldown: 20s, cast: 500ms
    hint: utility

    buff attack_speed 0.3 for 10s

    morph into {
        target: enemy, range: 1.5
        damage 80 [FIRE: 60]
    } for 10s
}
```

The morph inner block supports the same properties and effects as a regular ability. `for duration` controls how long the morph lasts (0 = permanent until base cooldown).

### Recast Block

Define effects for subsequent recasts of the same ability:

```
ability TripleSlash {
    target: enemy, range: 1.5
    cooldown: 8s, cast: 100ms
    recast: 2
    recast_window: 3s
    hint: damage

    damage 20 [PHYSICAL: 40]

    recast 1 {
        damage 25 [PHYSICAL: 45]
    }
    recast 2 {
        damage 40 [PHYSICAL: 55]
    }
}
```

---

### Damage Types

Used internally by the `Damage` effect:

| Type | Description |
|------|-------------|
| `physical` | Physical damage (reduced by armor) |
| `magic` | Magic damage (reduced by magic resist) |
| `true` | True damage (ignores resistances) |

---

## Class DSL Reference

### File Structure

`.class` files define character classes with stat growth, scaling rules, abilities, and requirements.

```
class Name {
    stat_growth: +N stat, +N stat, ... per level

    tags: tag1, tag2, tag3

    scaling source_name {
        when condition: bonus
        always: bonus
    }

    abilities {
        level N: ability_name "description"
    }

    requirements: req1, req2
    consolidates_at: N
}
```

Comments use `#`.

### Properties

| Property | Description |
|----------|-------------|
| `stat_growth` | Per-level stat gains. Syntax: `+N stat, +N stat, ... per level` |
| `tags` | Comma-separated role tags for contextual ability pools |
| `scaling` | Conditional bonus blocks (see below) |
| `abilities` | Guaranteed ability unlocks at specific levels |
| `requirements` | Prerequisites to unlock this class |
| `consolidates_at` | Level at which this class merges with primary class |

### Stats

#### Combat Stats

| Stat | Aliases | Description |
|------|---------|-------------|
| `attack` | | Attack damage |
| `defense` | | Damage reduction |
| `speed` | | Movement/action speed |
| `max_hp` | `hp` | Hit points |
| `ability_power` | `ap` | Spell power |

#### Non-Combat Stats

| Stat | Aliases | Description |
|------|---------|-------------|
| `diplomacy` | | Diplomatic skill |
| `commerce` | `trade` | Trading/economic skill |
| `crafting` | `craft` | Item creation |
| `medicine` | `healing` | Healing outside combat |
| `scholarship` | `lore` | Research/knowledge |
| `stealth` | `subterfuge` | Stealth/infiltration |
| `leadership` | `command` | Command ability |

The keyword `all` sets every stat to the same value:
```
stat_growth: +5 all per level
```

### Scaling Blocks

Scaling blocks define conditional bonuses from a named source:

```
scaling party_alive_count {
    when party_members > 0: +10% defense
    when party_members >= 3: tenacity 0.5
    always: aura morale +2
}
```

#### Valid Scaling Sources

`party_alive_count`, `party_size`, `faction_strength`, `coalition_strength`, `crisis_severity`, `fame`, `territory_control`, `adventurer_count`, `gold`, `reputation`, `threat_level`, `trade_income`, `diplomatic_relations`, `crafting_output`, `research_progress`, `guild_morale`, `supply_level`

#### Condition Operators

`>`, `>=`, `<`, `<=`, `==`

Simple flag conditions (no operator) are treated as `> 0`.

#### Bonus Types

| Syntax | Description |
|--------|-------------|
| `+N% stat` | Percentage stat increase |
| `+N stat` | Flat stat increase |
| `aura stat +N` | Buff to nearby allies |
| `tenacity N` | Combat mechanic (named value) |
| `name below/above N%` | Conditional mechanic with trigger |
| `trade +N%` | Trade bonus percentage |
| `diplomacy +N%` | Diplomacy bonus |
| `crafting +N%` | Crafting bonus |
| `healing +N%` | Healing effectiveness bonus |
| `research +N%` | Research speed bonus |

### Abilities Block

```
abilities {
    level 1: shield_wall "Reduces incoming damage to party"
    level 5: taunt "Forces enemies to target this unit"
    level 10: iron_will "Immune to morale effects"
}
```

Each line: `level N: ability_name "optional description"`

### Requirements

```
requirements: level 1, fame 2000, active_crisis
```

| Requirement | Syntax | Description |
|-------------|--------|-------------|
| Level | `level N` | Minimum level |
| Fame | `fame N` or `fame>=N` | Minimum fame |
| Trait | `trait name` | Must have named trait |
| Quests | `quests N` | Quests completed |
| Crisis | `active_crisis` | Must be during a crisis |
| Gold | `gold N` | Gold invested |
| Group | `group_size N` | Minimum group size |
| Allies | `allies N` | Number of allied factions |

### Validation Rules

- Total stat growth per level must be <= 50
- Total stat growth must be >= 0
- Scaling sources must be from the valid list
- Ability levels must be in ascending order and <= 100
- Consolidation level should be <= 20

---

## Grammar Space

The grammar space (`crates/ability-vae/src/grammar_space.rs`) provides an invertible mapping between a 48-dimensional unit hypercube `[0,1]^48` and valid DSL programs. Every point in the hypercube maps to exactly one syntactically valid ability, and every valid ability maps back to exactly one point.

### Dimension Layout (48 total)

#### Header (dims 0-7)

| Dim | Name | Range / Encoding |
|-----|------|-----------------|
| 0 | `D_TYPE` | 0 = ability, 1 = passive |
| 1 | `D_DOMAIN` | 0 = combat, 1 = campaign |
| 2 | `D_TARGET` | Index into targeting mode table |
| 3 | `D_RANGE` | Linear: 0.5 - 10.0 units |
| 4 | `D_COOLDOWN` | Log scale. Combat: 1s-60s. Campaign: 150s-30000s |
| 5 | `D_CAST` | Linear: 0 - 2000ms (combat only) |
| 6 | `D_HINT` | Index into hint table |
| 7 | `D_COST` | Linear: 0 - 30 resource cost |

#### Delivery (dims 8-11)

| Dim | Name | Range / Encoding |
|-----|------|-----------------|
| 8 | `D_DELIVERY` | Index into: `[none, none, none, projectile, chain, zone, trap]` (weighted 50% toward none) |
| 9 | `D_DELIV_P0` | Linear: 1-20. Speed (projectile), bounces (chain), duration (zone/trap) |
| 10 | `D_DELIV_P1` | Linear: 0.5-10. Range (chain), tick interval (zone), width (projectile) |
| 11 | `D_N_EFFECTS` | 1-4 effects per ability |

#### Effect Blocks (dims 12-43, stride of 8)

Each of the 4 possible effects occupies 8 dimensions:

| Offset | Name | Encoding |
|--------|------|----------|
| +0 | `TYPE` | Index into effect type pool (domain-dependent). See below |
| +1 | `PARAM` | Primary parameter (amount/factor/charges). Range depends on effect |
| +2 | `DUR` | Duration. Combat: log 500ms-5000ms. Campaign: log 50t-5000t |
| +3 | `AREA` | Index into: `[none(x4), circle, cone, line]` (weighted toward none) |
| +4 | `AREA_P` | Area parameter (radius). Linear: 1.0-5.0 |
| +5 | `TAG` | Index into: `[none(x3), PHYSICAL, MAGIC, FIRE, ICE, DARK, HOLY, POISON]` |
| +6 | `TAG_PWR` | Tag power level. Linear: 20-80 |
| +7 | `COND` | Index into: `[none(x6), target_hp_below, target_hp_above, caster_hp_below, hit_count_above]` |

Effect 0: dims 12-19. Effect 1: dims 20-27. Effect 2: dims 28-35. Effect 3: dims 36-43.

#### Passive / Scaling (dims 44-47)

| Dim | Name | Range / Encoding |
|-----|------|-----------------|
| 44 | `D_TRIGGER` | Index into trigger table |
| 45 | `D_TRIGGER_P` | Trigger parameter |
| 46 | `D_SCALING_STAT` | Index into: `[none(x4), target_max_hp, caster_max_hp, caster_attack_damage, target_missing_hp]` |
| 47 | `D_SCALING_PCT` | Scaling percentage. Linear: 5-50% |

### Combat Effect Type Pool

Effects are organized by parameter pattern:

- **Instant** (no params): `dispel`, `swap`, `refresh_cooldowns`
- **Amount**: `damage`, `heal`, `shield`, `knockback`, `pull`, `dash`, `blink`
- **Duration**: `stun`, `root`, `silence`, `fear`, `taunt`, `stealth`, `blind`, `charm`, `suppress`, `confuse`, `polymorph`, `banish`, `cooldown_lock`
- **Amount + Duration**: `slow`, `damage_modify`, `lifesteal`, `mana_burn`
- **Charges**: `amplify`, `echo`, `instant_cast`, `free_cast`, `spell_shield`
- **Buff** (5 stats): `damage_output`, `move_speed`, `attack_speed`, `armor`, `magic_resist`
- **Debuff** (5 stats): same stat list
- **DoT**: `damage N/1s for Ns`
- **HoT**: `heal N/1s for Ns`

### Campaign Effect Type Pool

- **Instant**: 34 effects (e.g. `appraise`, `beast_lore`, `claim_territory`, ...)
- **Amount**: `rally`, `blood_oath`, `forage`, `field_triage`, `trade_empire`
- **Duration**: 19 effects (e.g. `ghost_walk`, `fortify`, `ceasefire`, ...)
- **Count**: `reveal`, `prophecy`

---

## Quality Scoring

The quality scorer (`crates/ability-vae/src/quality.rs`) evaluates grammar space vectors on four axes, returning a score from 0.0 (garbage) to 1.0 (excellent):

### Coherence

Penalizes combinations that do not make logical sense:
- Heal targeting enemy (-0.3)
- Damage targeting ally (-0.3)
- Campaign effects with combat delivery (-0.4)
- Element tags on campaign effects (-0.1)
- Ground target without area and without delivery (-0.15)

### Balance

Penalizes over/underpowered abilities:
- High damage + low cooldown (-0.2)
- Low damage + high cooldown on a single effect (-0.15)
- Many effects with short cooldown (-0.1)
- Single simple effect with very long cooldown (-0.1)
- Rewards power proportional to cooldown (+0.15 scaled)

### Purpose

Rewards clear identity:
- Hint matches effect type (+0.2)
- Passive trigger relates to effects (+0.1)
- Campaign triggers on combat abilities penalized (-0.3)
- Hard CC on low-cooldown passives penalized (-0.2)

### Variety

Bonuses for underrepresented ability types:
- Non-damage combat abilities (+0.1)
- Campaign abilities (+0.05)
- Passives (+0.05)
- Element tags (+0.05)
- Area effects (+0.05)
- Multi-effect combos (+0.05)
- Delivery methods (+0.05)

---

## Tokenizer Vocabulary

The ability tokenizer (`tokenizer_vocab.rs`) maps DSL text to a 397-token vocabulary for the transformer model. Key token groups:

- **Special tokens** (0-7): `[PAD]`, `[CLS]`, `[MASK]`, `[SEP]`, `[UNK]`, `[NAME]`, `[STR]`, `[TAG]`
- **Punctuation** (8-17): `{`, `}`, `(`, `)`, `[`, `]`, `:`, `,`, `+`, `%`
- **Keywords** (18-226): All DSL keywords sorted alphabetically (`ability`, `damage`, `heal`, `when`, etc.)
- **Number buckets** (227-246): `NUM_0` through `NUM_10`, `NUM_SMALL` (11-50), `NUM_MED` (51-200), `NUM_LARGE` (201-1000), `NUM_HUGE` (1001+), `FRAC_TINY` (0-0.2), `FRAC_LOW`, `FRAC_MID`, `FRAC_HIGH`, `FRAC_MAX` (0.8-1.0)
- **Duration buckets** (247-251): `DUR_INSTANT` (<200ms), `DUR_SHORT` (200ms-2s), `DUR_MED` (2s-8s), `DUR_LONG` (8s-30s), `DUR_VLONG` (30s+)
- **Campaign keywords** (252-299): Campaign-specific effect names and conditions
- **Campaign tick buckets** (348-352): `TICK_SHORT` (1-50t), `TICK_MED` (51-200t), `TICK_LONG` (201-500t), `TICK_VLONG` (501-1000t), `TICK_EPIC` (1001t+)
- **Meta/recursive** (353+): Meta-effect keywords, combination skill tokens, targeting filter keywords, aura/scaling keywords
- Max sequence length: 256 tokens

---

## Examples

### 1. Simple Damage Ability

```
ability Strike {
    target: enemy, range: 1.5
    cooldown: 3s, cast: 200ms
    hint: damage

    damage 20 [PHYSICAL: 40]
}
```

### 2. AoE Heal with Zone Delivery

```
ability ConsecratedGround {
    target: ground, range: 5.0
    cooldown: 10s, cast: 300ms
    hint: heal

    deliver zone { duration: 5s, tick: 1s } {
        on_hit {
            heal 12 in circle(3.0) [HOLY: 35]
        }
    }
}
```

### 3. Passive with Trigger

```
passive Bloodthirst {
    trigger: on_damage_dealt
    cooldown: 5s

    heal 12
}
```

### 4. Campaign Ability

```
ability CornerTheMarket {
    target: market
    cooldown: 5000t
    hint: economy

    corner_market "grain" for 500t
}
```

### 5. Meta-Effect (Amplify + Echo)

```
ability TripleThreat {
    target: self
    cooldown: 30s, cast: 0ms
    hint: damage

    amplify 2.0 1
    echo 1
    instant_cast 1
}
```

### 6. Recursive (On-Hit Cast + Grant Ability)

```
ability MentorStrike {
    target: enemy, range: 1.5
    cooldown: 6s, cast: 200ms
    hint: damage

    damage 35 [PHYSICAL: 50]
    grant_ability "StudentCounter" for 8s
}
```

### 7. Aura (while_alive)

```
passive HealingPresence {
    trigger: periodic(3s)
    cooldown: 3s

    heal 8 in circle(5.0) while_alive [HOLY: 25]
}
```

### 8. Targeting Filter

```
ability ClassicFormation {
    target: self_aoe
    cooldown: 25s, cast: 400ms
    hint: defense

    buff armor 0.3 for 10s in circle(6.0) targeting has_class("knight")
    buff damage_output 0.2 for 10s in circle(6.0) targeting has_class("ranger")
}
```

### 9. Scaling Ability

```
ability LegionStrike {
    target: enemy, range: 3.0
    cooldown: 12s, cast: 300ms
    hint: damage

    damage 5 [PHYSICAL: 40] scales_with army_size
}
```

### 10. Combination Ability (Ultimate)

```
ability BlackHole {
    target: ground, range: 6.0
    cooldown: 60s, cast: 600ms
    hint: crowd_control

    pull 4.0 in circle(4.0)
    stun 2500ms in circle(3.0) [CROWD_CONTROL: 65]
    damage 80 in circle(3.0) [DARK: 55]
}
```

### 11. DoT with Chain Delivery

```
ability ChainLightning {
    target: enemy, range: 6.0
    cooldown: 8s, cast: 300ms
    hint: damage

    deliver chain { bounces: 3, range: 4.0, falloff: 0.3 } {
        on_hit {
            damage 40 [MAGIC: 50]
            on_hit_cast "ChainBolt"
        }
    }
}
```

### 12. Conditional with Else

```
damage 80 when target_hp_below(30%) else damage 30
```

### 13. Hybrid Combat + Campaign Ability

```
ability WarCry {
    target: self_aoe
    cooldown: 15s
    hint: leadership

    buff damage_output 0.2 for 5s
    rally 0.3
    taunt 2s in circle(4.0)
}
```
