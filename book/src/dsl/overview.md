# Ability DSL

The ability DSL is a domain-specific language for defining hero abilities. It
replaces verbose TOML arrays with a compact, readable syntax that game designers
can author without touching Rust code.

## Motivation

Compare the TOML representation of a simple ability:

```toml
[[abilities]]
name = "Fireball"
targeting = "target_enemy"
range = 5.0
cooldown_ms = 5000
cast_time_ms = 500
ai_hint = "damage"

[[abilities.effects]]
effect = { Damage = 55 }
tags = { FIRE = 60 }

[[abilities.delivery]]
type = "Projectile"
speed = 8.0
width = 0.3
```

With the DSL version:

```
ability Fireball {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 500ms
    hint: damage

    deliver projectile { speed: 8.0, width: 0.3 } {
        on_hit { damage 55 [FIRE: 60] }
    }
}
```

The DSL is more readable, more concise, and catches errors at parse time with
line/column information.

## File Convention

- Active abilities use the `ability Name { ... }` syntax
- Passive abilities use the `passive Name { ... }` syntax
- Files use the `.ability` extension
- Hero templates pair a `.toml` file (stats) with an `.ability` file (abilities)
- Multiple abilities and passives can be defined in a single file

## Quick Example

A complete hero's abilities (`warrior.ability`):

```
ability Whirlwind {
    target: self_aoe
    cooldown: 8s, cast: 400ms
    hint: damage

    damage 40 in circle(2.5) [PHYSICAL: 50]
    damage 10 in circle(2.5) when hit_count_above(2) [PHYSICAL: 50]
}

ability ShieldWall {
    target: self
    cooldown: 15s, cast: 200ms
    hint: defense

    shield 60 for 5s
}

ability HeroicCharge {
    target: enemy, range: 5.0
    cooldown: 10s, cast: 0ms
    hint: crowd_control

    dash to_target
    damage 35 [PHYSICAL: 50]
    stun 1500ms [CROWD_CONTROL: 70]
}

passive IronSkin {
    trigger: on_damage_taken
    cooldown: 5s

    shield 20 for 3s
}

passive LastStand {
    trigger: on_hp_below(25%)
    cooldown: 45s

    heal 50
    buff damage_output 0.3 for 5s
}
```

## Module Location

The DSL parser lives in `src/ai/effects/dsl/` with 17 source files covering
parsing, AST representation, lowering, emission, and testing.
