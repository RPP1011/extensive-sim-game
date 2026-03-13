# /ability — Generate Hero Ability DSL

Generate a hero ability or passive definition in .ability DSL format from a natural-language description.
The DSL is parsed by the ability parser in `src/ai/effects/dsl/` and is the canonical format for ability definitions.

## Instructions

1. Read the user's description of the ability/passive they want.
2. Generate valid DSL following the syntax reference below.
3. Output ONLY the ability/passive block(s) in a code fence.
4. Validate against the checklist at the bottom.

---

## DSL Syntax Reference

### Ability Structure
```
ability AbilityName {
    target: <targeting>, range: <float>      // range omitted for self/self_aoe
    cooldown: <duration>, cast: <duration>
    hint: <ai_hint>
    resource_cost: <int>                      // optional, default 0

    <effects...>
    <delivery...>                             // optional
}
```

### Passive Structure
```
passive PassiveName {
    trigger: <trigger_type>(<args>)
    cooldown: <duration>
    range: <float>                            // only for ally/kill triggers

    <effects...>
}
```

### Duration Format
- `Xs` for seconds: `3s`, `1.5s`
- `Xms` for milliseconds: `1500ms`, `300ms`

### Targeting
- `enemy` — single enemy (`target: enemy, range: 5.0`)
- `ally` — single ally (`target: ally, range: 4.0`)
- `self` — caster only (`target: self`)
- `self_aoe` — AoE centered on caster (`target: self_aoe`)
- `ground` — ground-targeted (`target: ground, range: 6.0`)
- `direction` — skillshot (`target: direction, range: 8.0`)

### AI Hints
`damage`, `heal`, `crowd_control`, `defense`, `utility`

---

## Effect Types

### Damage & Healing
```
damage <amount> [TAG: value, ...]                     // instant
damage <amount>/<interval> for <duration> [TAGS]      // DoT (e.g. damage 10/1s for 4s)
heal <amount> [TAGS]                                  // instant
heal <amount>/<interval> for <duration> [TAGS]        // HoT
shield <amount> for <duration>
self_damage <amount>
```

### Crowd Control
```
stun <duration> [TAGS]
slow <factor> for <duration> [TAGS]          // factor 0.0–1.0
root <duration> [TAGS]
silence <duration> [TAGS]
fear <duration> [TAGS]
taunt <duration> [TAGS]
polymorph <duration> [TAGS]
banish <duration> [TAGS]
confuse <duration> [TAGS]
charm <duration> [TAGS]
blind <miss_chance> for <duration> [TAGS]    // miss_chance 0.0–1.0
```

### Positioning
```
dash to_target                    // dash to ability target
dash <distance>                   // dash forward (self)
knockback <distance> [TAGS]
pull <distance>
swap
```

### Buffs & Modifiers
```
buff <stat> <factor> for <duration>
debuff <stat> <factor> for <duration> [TAGS]
damage_modify <factor> for <duration>         // >1 = amplify incoming, <1 = reduce
reflect <percent> for <duration>
lifesteal <percent> for <duration>
stealth for <duration>
```

Stat names: `damage`, `move_speed`, `attack_speed`, `cooldown_reduction`, `heal_power`, `damage_output`

### Advanced
```
execute <threshold>%
death_mark for <duration> <damage_percent>%
detonate <multiplier>x
resurrect <hp_percent>%
dispel [TAG1, TAG2, ...]
summon "<template>" x<count>
```

---

## Area Modifiers

Append `in <shape>(<args>)` to any effect:
```
damage 40 in circle(2.5)
damage 35 in cone(3.0, 60)           // radius, angle_deg
damage 30 in line(5.0, 1.0)          // length, width
damage 25 in spread(4.0, 3)          // radius, max_targets
damage 30 in ring(1.5, 3.5)          // inner_radius, outer_radius
heal 15 in self                       // caster only (for self-heal on damage abilities)
```

## Tags

Append `[TAG: value, ...]` to any effect line:
```
damage 50 [PHYSICAL: 45, FIRE: 30]
stun 2s [CROWD_CONTROL: 60]
```
Common: `PHYSICAL`, `MAGIC`, `FIRE`, `ICE`, `HOLY`, `DARK`, `POISON`, `CROWD_CONTROL`, `SLOW`, `KNOCKBACK`, `SILENCE`, `BLEED`, `FEAR`

## Conditions

Append `when <condition>` to any effect:
```
damage 30 when target_hp_below(25%)
damage 25 when target_is_stunned
heal 20 when caster_hp_below(50%)
damage 10 in circle(2.5) when hit_count_above(2)
```

Conditions: `target_hp_below(<pct>)`, `target_hp_above(<pct>)`, `caster_hp_below(<pct>)`, `caster_hp_above(<pct>)`, `target_is_stunned`, `target_is_slowed`, `target_is_rooted`, `target_is_silenced`, `target_is_feared`, `target_is_taunted`, `target_is_banished`, `target_is_stealthed`, `target_is_charmed`, `target_is_polymorphed`, `hit_count_above(<n>)`, `target_has_tag(<TAG>)`, `caster_has_status(<status>)`, `target_has_status(<status>)`, `target_debuff_count(<n>)`, `caster_buff_count(<n>)`, `ally_count_below(<n>)`, `enemy_count_below(<n>)`, `target_stack_count(<name>, <n>)`

---

## Delivery Methods

### Projectile
```
deliver projectile { speed: <f32>, width: <f32>, pierce } {
    on_hit {
        <effects...>
    }
    on_arrival {
        <effects...>       // splash effects, usually with area
    }
}
```

### Chain
```
deliver chain { bounces: <u32>, range: <f32>, falloff: <f32> } {
    on_hit {
        <effects...>
    }
}
```

### Zone (persistent ground)
```
deliver zone { duration: <dur>, tick: <dur> } {
    on_hit {
        <effects...>       // applied each tick to units in area
    }
}
```

### Trap
```
deliver trap { duration: <dur>, trigger_radius: <f32>, arm_time: <dur> } {
    on_hit {
        <effects...>       // applied once when triggered
    }
}
```

### Tether
```
deliver tether { max_range: <f32>, tick: <dur> } {
    on_hit {
        <effects...>       // applied each tick
    }
    on_complete {
        <effects...>       // fires when tether expires naturally
    }
}
```

### Channel
```
deliver channel { duration: <dur>, tick: <dur> } {
    on_hit {
        <effects...>       // applied each tick
    }
}
```

---

## Triggers (passives only)

```
trigger: on_damage_dealt
trigger: on_damage_taken
trigger: on_kill
trigger: on_ally_damaged, range: <f32>
trigger: on_death
trigger: on_ability_used
trigger: on_hp_below(<pct>%)
trigger: on_hp_above(<pct>%)
trigger: on_shield_broken
trigger: periodic(<interval>)
trigger: on_heal_received
trigger: on_ally_killed, range: <f32>
trigger: on_auto_attack
trigger: on_stack_reached(<name>, <count>)
```

---

## Examples

### Single target damage
```
ability Zap {
    target: enemy, range: 5.0
    cooldown: 4s, cast: 200ms
    hint: damage

    damage 35 [MAGIC: 45]
}
```

### AoE with CC
```
ability FrostNova {
    target: self_aoe
    cooldown: 10s, cast: 300ms
    hint: crowd_control

    damage 20 in circle(3.0)
    stun 2s in circle(3.0) [CROWD_CONTROL: 80, ICE: 60]
}
```

### Dash + damage
```
ability ChargeStrike {
    target: enemy, range: 5.0
    cooldown: 6s, cast: 0ms
    hint: damage

    dash to_target
    damage 35 [PHYSICAL: 45]
}
```

### DoT
```
ability Ignite {
    target: enemy, range: 4.0
    cooldown: 6s, cast: 200ms
    hint: damage

    damage 8/1s for 3s [FIRE: 50]
}

```

### Projectile with splash
```
ability Fireball {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    deliver projectile { speed: 8.0, width: 0.3 } {
        on_hit {
            damage 55 [FIRE: 60]
        }
        on_arrival {
            damage 15 in circle(2.0)
        }
    }
}
```

### Chain bounce
```
ability ChainLightning {
    target: enemy, range: 5.0
    cooldown: 8s, cast: 300ms
    hint: damage

    damage 30 [MAGIC: 55]

    deliver chain { bounces: 3, range: 3.0, falloff: 0.15 } {
        on_hit {
            damage 30 [MAGIC: 55]
        }
    }
}
```

### Persistent zone
```
ability Blizzard {
    target: ground, range: 6.0
    cooldown: 12s, cast: 400ms
    hint: damage

    deliver zone { duration: 4s, tick: 1s } {
        on_hit {
            damage 15 in circle(3.0) [ICE: 50]
            slow 0.3 for 1500ms in circle(3.0)
        }
    }
}
```

### Damage + self-heal
```
ability SoulDrain {
    target: enemy, range: 4.0
    cooldown: 4s, cast: 300ms
    hint: damage

    damage 40 [DARK: 55]
    heal 20
}
```

### Conditional bonus
```
ability Shatter {
    target: enemy, range: 1.5
    cooldown: 6s, cast: 200ms
    hint: damage

    damage 35 [PHYSICAL: 45]
    damage 25 when target_is_stunned [PHYSICAL: 50]
}
```

### Passive — reactive shield
```
passive IronSkin {
    trigger: on_damage_taken
    cooldown: 5s

    shield 20 for 3s
}
```

### Passive — on-kill buff
```
passive Predator {
    trigger: on_kill
    cooldown: 5s

    buff damage_output 0.2 for 4s
}
```

---

## Hero Template (full file)

When generating a complete hero `.ability` file, include stats as comments and use this structure:
```
// HeroName abilities
// Stats: hp=100, speed=3.0, atk=15/1.5range

ability AbilityOne {
    ...
}

ability AbilityTwo {
    ...
}

passive PassiveOne {
    ...
}
```

---

## Validation Checklist

Before outputting, verify:

- [ ] All effect names are valid (damage, heal, stun, slow, root, silence, fear, etc.)
- [ ] Durations use `Xs` or `Xms` format consistently
- [ ] DoT/HoT uses `amount/interval for duration` syntax (e.g. `damage 10/1s for 4s`)
- [ ] Instant damage/heal uses just `amount` (no interval/duration)
- [ ] Targeting matches effect intent (heal → ally, damage → enemy or self_aoe)
- [ ] `hint` accurately reflects the ability's primary purpose
- [ ] Projectile/chain abilities put effects inside `deliver ... { on_hit { ... } }`
- [ ] Tags use `[UPPER_CASE: value]` format with values 30–100
- [ ] Cooldowns are reasonable (2–30s for abilities, 3–60s for passives)
- [ ] Range > 0 for targeted abilities, omitted for self/self_aoe
- [ ] Buff/debuff stat is valid: damage, move_speed, attack_speed, cooldown_reduction, heal_power, damage_output
- [ ] Braces are balanced and properly nested

$ARGUMENTS
