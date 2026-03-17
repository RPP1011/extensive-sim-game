# Writing New Abilities

This chapter walks through creating a new hero with abilities from scratch.

## Step 1: Create the Stats File

Create `assets/hero_templates/pyromancer.toml`:

```toml
[hero]
name = "Pyromancer"
role = "mage"

[stats]
hp = 450
armor = 15
magic_resist = 20
attack_damage = 30
attack_range = 4.0
attack_cooldown_ms = 1200
move_speed_per_sec = 3.0
resource = 100
max_resource = 100
resource_regen_per_sec = 5.0
```

## Step 2: Create the Ability File

Create `assets/hero_templates/pyromancer.ability`:

```
// Pyromancer — fire mage specializing in AoE and DoT

ability Ignite {
    target: enemy, range: 5.0
    cooldown: 6s, cast: 200ms
    hint: damage

    damage 30 [FIRE: 50]
    dot 40 for 4s [FIRE: 40]
}

ability Inferno {
    target: ground, range: 6.0
    cooldown: 14s, cast: 600ms
    hint: damage

    deliver zone { duration: 5s, tick: 1s } {
        on_hit {
            damage 20 in circle(2.5) [FIRE: 60]
        }
    }
}

ability FlameBarrier {
    target: self
    cooldown: 18s, cast: 300ms
    hint: defense

    shield 40 for 4s
    damage 15 in circle(2.0) [FIRE: 40]
}

ability Pyroblast {
    target: enemy, range: 5.0
    cooldown: 20s, cast: 1200ms
    hint: damage
    cost: 40

    deliver projectile { speed: 6.0, width: 0.4 } {
        on_hit {
            damage 120 [FIRE: 80]
        }
        on_arrival {
            damage 40 in circle(2.0) [FIRE: 60]
        }
    }
}

passive Cauterize {
    trigger: on_ability_used
    cooldown: 10s

    hot 20 for 3s
}
```

## Step 3: Verify Parsing

Run the parser on your new file to check for syntax errors:

```bash
cargo test -- --nocapture parse_abilities 2>&1 | head -50
```

Or use the scenario runner to load and simulate the hero:

```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
```

## Design Guidelines

### Ability Count
Each hero should have **4-8 active abilities** and **1-3 passives**. More than
8 actives overwhelms the AI evaluator; fewer than 4 feels flat.

### Cooldown Ranges
- **Short** (3-6s): bread-and-butter abilities, spammable
- **Medium** (8-15s): impactful abilities with tactical timing
- **Long** (18-30s+): ultimates, fight-defining cooldowns

### Cast Times
- **0ms**: instant (dashes, emergency buttons)
- **200-400ms**: fast (most abilities)
- **500-800ms**: deliberate (powerful effects)
- **1000ms+**: telegraphed (very powerful, interruptible)

### AI Hints Matter
The `hint` field directly affects how the squad AI prioritizes abilities:
- `damage` hints are used when pursuing kills
- `crowd_control` hints are used against high-priority targets
- `heal` hints are used when allies are injured
- `defense` hints are used when under heavy pressure
- `utility` hints are used for mobility and positioning

### Tag Values
Tag values range from 0 to 100:
- **0-30**: Minor effect
- **40-60**: Standard effect
- **70-80**: Strong effect
- **90-100**: Extreme (use sparingly)

### Balance Through Tradeoffs
Good abilities have clear tradeoffs:
- High damage → long cooldown or cast time
- AoE → lower per-target damage
- Crowd control → no damage
- Mobility → no offensive effect
- Instant cast → short range or weaker effect
