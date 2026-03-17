# The Five Dimensions

Every ability in the game can be described along five composable dimensions.
These dimensions are orthogonal — any combination is valid, creating an
exponentially large design space from a small set of primitives.

## 1. Effect — *What* Happens

The core of what an ability does to the world:

| Effect | Description |
|--------|------------|
| `damage N` | Deal N damage |
| `heal N` | Restore N HP |
| `shield N for Ts` | Grant N temporary HP for T seconds |
| `stun Ts` | Prevent actions for T seconds |
| `root Ts` | Prevent movement for T seconds |
| `slow F for Ts` | Reduce move speed by factor F |
| `silence Ts` | Prevent ability usage |
| `fear Ts` | Force random movement |
| `polymorph Ts` | Transform (prevents all actions) |
| `knockback D` | Push away D units |
| `pull D` | Pull toward caster D units |
| `dash D` | Move caster D units |
| `buff STAT F for Ts` | Modify a stat by factor F |
| `damage_modify F for Ts` | Increase damage taken |
| `dot N for Ts` | Damage over time |
| `hot N for Ts` | Heal over time |
| `summon NAME` | Create a unit |
| `revive` | Resurrect a dead ally |
| `rewind Ts` | Restore position/HP from T seconds ago |
| `steal_buff` | Transfer a buff from target |
| `cleanse` | Remove negative status effects |

## 2. Area — *Where* It Hits

Defines the spatial shape of the effect:

| Area | Description |
|------|------------|
| (none) | Single target |
| `circle(R)` | Circle with radius R around target |
| `line(W)` | Line from caster to target with width W |
| `cone(A)` | Cone in front of caster with angle A |
| `ring(Rmin, Rmax)` | Ring between two radii |
| `rectangle(W, H)` | Rectangle ahead of caster |

Example: `damage 50 in circle(3.0)` deals 50 damage to all enemies in a 3-unit
radius circle.

## 3. Delivery — *How* It Gets There

Defines how the effect reaches its target:

| Delivery | Description |
|----------|------------|
| (none) | Instant — effect applies immediately |
| `projectile { speed, width }` | Travels in a line, hits first target |
| `chain { bounces, range, falloff }` | Bounces between targets |
| `zone { duration, tick }` | Persists on the ground, ticking periodically |
| `tether { duration, range }` | Beam connecting caster to target |
| `boomerang { speed, return_speed }` | Travels out and returns |
| `wall { segments, spacing }` | Creates a wall of effects |
| `aura { range }` | Continuous effect around caster |

Each delivery wraps effects in `on_hit`, `on_arrival`, or `on_tick` blocks
that fire at different points in the delivery lifecycle.

## 4. Trigger — *When* It Fires

For passives and conditional effects:

| Trigger | Description |
|---------|------------|
| `on_damage_taken` | When this unit takes damage |
| `on_hp_below(N%)` | When HP drops below N% |
| `on_ability_used` | After using any ability |
| `on_kill` | After killing an enemy |
| `on_death` | When this unit dies |
| `on_ally_death` | When a teammate dies |
| `when hit_count_above(N)` | When ability hits N+ targets |
| `when target_is_stunned` | When target has a stun effect |
| `when target_hp_below(N%)` | When target HP is below N% |

## 5. Tags — *Power Levels*

Tags are key-value pairs that describe the nature and intensity of effects:

```
[FIRE: 60, CROWD_CONTROL: 80]
```

Tags serve multiple purposes:
- **Elemental typing** — `FIRE`, `ICE`, `LIGHTNING`, `MAGIC`, `PHYSICAL`
- **Effect intensity** — a `CROWD_CONTROL: 80` stun is harder to resist than `50`
- **AI reasoning** — the neural evaluator uses tags to estimate ability value
- **Resistance matching** — units with `resistance_tags` reduce matching effects

## Composition Example

Here's a real ability showing all five dimensions:

```
ability Fireball {
    target: enemy, range: 5.0        # targeting
    cooldown: 5s, cast: 500ms        # timing

    deliver projectile { speed: 8.0, width: 0.3 } {    # 3. Delivery
        on_hit {
            damage 55 [FIRE: 60]                         # 1. Effect + 5. Tags
        }
        on_arrival {
            damage 15 in circle(2.0)                     # 1. Effect + 2. Area
        }
    }
}
```

This Fireball:
- **Effect:** Deals 55 single-target damage on hit, plus 15 AoE on arrival
- **Area:** The on-arrival damage hits a circle of radius 2.0
- **Delivery:** Travels as a projectile at speed 8.0
- **Tags:** FIRE: 60 (strong fire element)
- **Trigger:** None (it's an active ability, not a passive)
