# Template File Format

## TOML Stats File

The `.toml` file defines a hero's base statistics:

```toml
[hero]
name = "Mage"
role = "ranged_dps"

[stats]
hp = 500
armor = 15
magic_resist = 25
attack_damage = 35
attack_range = 4.5
attack_cooldown_ms = 1200
attack_cast_time_ms = 200
move_speed_per_sec = 3.2
ability_damage = 60
ability_range = 5.0
ability_cooldown_ms = 5000
ability_cast_time_ms = 500
heal_amount = 0
control_range = 4.0
control_duration_ms = 2000
control_cooldown_ms = 15000
resource = 100
max_resource = 100
resource_regen_per_sec = 5.0
```

### Stat Descriptions

| Stat | Description |
|------|------------|
| `hp` | Maximum hit points |
| `armor` | Physical damage reduction (reduction = armor / (100 + armor)) |
| `magic_resist` | Magic damage reduction (same formula as armor) |
| `attack_damage` | Base auto-attack damage |
| `attack_range` | Auto-attack range (in grid units) |
| `attack_cooldown_ms` | Time between auto-attacks |
| `attack_cast_time_ms` | Wind-up time for auto-attacks |
| `move_speed_per_sec` | Movement speed (grid units per second) |
| `ability_damage` | Legacy: base ability damage (used by simple AI) |
| `ability_range` | Legacy: base ability range |
| `heal_amount` | Legacy: base heal amount |
| `control_range` | Legacy: CC range |
| `control_duration_ms` | Legacy: CC duration |
| `resource` / `max_resource` | Resource pool (mana, energy, etc.) |
| `resource_regen_per_sec` | Resource regeneration rate |

> **Note:** The `ability_damage`, `heal_amount`, and `control_*` fields are legacy
> fields used by the simple AI before the hero ability engine existed. They are still
> used as fallbacks when a hero has no `.ability` file.

## Ability File

The `.ability` file uses the [Ability DSL](../dsl/overview.md):

```
ability Fireball {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 500ms
    hint: damage

    deliver projectile { speed: 8.0, width: 0.3 } {
        on_hit { damage 55 [FIRE: 60] }
        on_arrival { damage 15 in circle(2.0) }
    }
}

// ... more abilities

passive ArcaneShield {
    trigger: on_hp_below(50%)
    cooldown: 30s

    shield 40 for 4s
}
```

See the [Syntax Reference](../dsl/syntax.md) for the full DSL specification.

## File Naming Convention

- Stats: `<hero_name>.toml`
- Abilities: `<hero_name>.ability`
- Both files must share the same base name
- Names use snake_case: `blood_mage.toml`, `blood_mage.ability`
