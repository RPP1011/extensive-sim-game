# Syntax Reference

This chapter is the complete syntax reference for the `.ability` DSL.

## Ability Declaration

```
ability <Name> {
    <header>
    <effects and delivery>
}
```

### Header Fields

| Field | Syntax | Example | Required |
|-------|--------|---------|----------|
| Target | `target: <mode>` | `target: enemy` | Yes |
| Range | `range: <float>` | `range: 5.0` | No (default 0) |
| Cooldown | `cooldown: <duration>` | `cooldown: 10s` | Yes |
| Cast time | `cast: <duration>` | `cast: 300ms` | No (default 0) |
| AI hint | `hint: <category>` | `hint: damage` | No |
| Resource cost | `cost: <int>` | `cost: 50` | No |

**Target modes:** `enemy`, `ally`, `self`, `self_aoe`, `ground`, `direction`,
`vector`, `global`

**Duration formats:** `5s` (seconds), `300ms` (milliseconds), `1.5s`

**AI hint values:** `damage`, `crowd_control`, `defense`, `utility`, `heal`

Multiple header fields can appear on the same line, separated by commas:
```
target: enemy, range: 5.0
cooldown: 5s, cast: 300ms
```

## Effect Statements

Effects appear in the body of an ability or inside delivery blocks:

### Damage
```
damage <amount>
damage <amount> [TAG: value, ...]
damage <amount> in <area>
damage <amount> in <area> when <condition>
```

### Healing
```
heal <amount>
heal <amount> in <area>
```

### Shields
```
shield <amount> for <duration>
```

### Crowd Control
```
stun <duration>
stun <duration> [CROWD_CONTROL: value]
root <duration>
silence <duration>
fear <duration>
polymorph <duration>
slow <factor> for <duration>
```

### Knockback & Movement
```
knockback <distance>
pull <distance>
dash <distance>
dash to_target
```

### Buffs & Debuffs
```
buff <stat> <factor> for <duration>
buff <stat> <factor> for <duration> in <area>
damage_modify <factor> for <duration>
```

**Buff stats:** `damage_output`, `move_speed`, `attack_speed`, `cooldown_reduction`,
`armor`, `magic_resist`

### Damage Over Time / Heal Over Time
```
dot <amount> for <duration>
hot <amount> for <duration>
```

### Summon
```
summon <template_name>
```

### Utility
```
revive
rewind <duration>
steal_buff
cleanse
```

## Area Shapes

```
in circle(<radius>)
in line(<width>)
in cone(<angle>)
in ring(<inner_radius>, <outer_radius>)
in rectangle(<width>, <height>)
```

## Conditions

```
when hit_count_above(<n>)
when target_is_stunned
when target_hp_below(<percent>%)
when caster_hp_below(<percent>%)
when target_has_buff
when target_has_debuff
```

## Tags

Tags follow an effect, enclosed in square brackets:

```
damage 55 [FIRE: 60, MAGIC: 40]
stun 2s [CROWD_CONTROL: 80, ICE: 60]
```

## Delivery Blocks

```
deliver <type> { <params> } {
    on_hit { <effects> }
    on_arrival { <effects> }
    on_tick { <effects> }
}
```

### Projectile
```
deliver projectile { speed: 8.0, width: 0.3 } {
    on_hit { damage 55 }
}
```

### Chain
```
deliver chain { bounces: 3, range: 3.0, falloff: 0.8 } {
    on_hit { damage 35 }
}
```

### Zone
```
deliver zone { duration: 4s, tick: 1s } {
    on_hit { damage 15 in circle(3.0) }
}
```

### Tether
```
deliver tether { duration: 3s, range: 6.0 } {
    on_tick { damage 10 }
}
```

### Boomerang
```
deliver boomerang { speed: 6.0, return_speed: 8.0 } {
    on_hit { damage 30 }
}
```

### Wall
```
deliver wall { segments: 5, spacing: 1.0 } {
    on_hit { stun 1s }
}
```

## Passive Declaration

```
passive <Name> {
    trigger: <trigger_type>
    cooldown: <duration>

    <effects>
}
```

### Trigger Types

```
trigger: on_damage_taken
trigger: on_hp_below(50%)
trigger: on_ability_used
trigger: on_kill
trigger: on_death
trigger: on_ally_death
```

## Comments

Line comments start with `//`:

```
// Mage abilities
ability Fireball {
    // Main damage spell
    target: enemy, range: 5.0
    ...
}
```

## Advanced Features

### Charges
```
ability Mushroom {
    target: ground, range: 4.0
    cooldown: 1s, cast: 200ms
    charges: 3, recharge: 20s
    ...
}
```

### Toggle
```
ability PoisonTrail {
    target: self
    toggle: true
    toggle_cost: 10/s
    ...
}
```

### Recast
```
ability SpiritRush {
    target: direction
    cooldown: 12s, cast: 0ms
    recast: 2, window: 10s
    ...
}
```

### Unstoppable
```
ability UnstoppableForce {
    target: enemy, range: 5.0
    cooldown: 15s, cast: 500ms
    unstoppable: true
    ...
}
```
