# Ability System Reference

Complete reference for the data-driven ability engine, the LoL champion dataset, and the gap analysis for full coverage.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Hero TOML Format](#hero-toml-format)
3. [Effect Types (45)](#effect-types)
4. [Area Shapes (7)](#area-shapes)
5. [Delivery Methods (7)](#delivery-methods)
6. [Conditions (24)](#conditions)
7. [Passive Triggers (19)](#passive-triggers)
8. [Targeting Modes (8)](#targeting-modes)
9. [Damage Types](#damage-types)
10. [Status Effects](#status-effects)
11. [LoL Champion Dataset](#lol-champion-dataset)
12. [Gap Analysis](#gap-analysis)

---

## System Architecture

Five composable dimensions define every ability:

```
WHAT (Effect) x WHERE (Area) x HOW (Delivery) x WHEN (Condition/Trigger) + Tags
```

### Key Files

| File | Purpose |
|------|---------|
| `src/ai/effects/effect_enum.rs` | `Effect` enum with all 45 variants and serde defaults |
| `src/ai/effects/types.rs` | Area, Delivery, Condition, Trigger enums; `ConditionalEffect` wrapper; `DamageType`; `Stacking` |
| `src/ai/effects/defs.rs` | `AbilityDef`, `PassiveDef`, `AbilitySlot`, `PassiveSlot`, `HeroToml`, `AbilityTargeting`, `StatusKind`, `ActiveStatusEffect`, `Projectile`, `AbilityTarget` |
| `src/ai/core/apply_effect.rs` | Main effect dispatcher (handles all effect types) |
| `src/ai/core/apply_effect_ext.rs` | Extended effect application (phases 4-7) |
| `src/ai/core/damage.rs` | Damage/heal application, chain delivery, stat scaling |
| `src/ai/core/triggers.rs` | Passive trigger system (check + fire) |
| `src/ai/core/targeting.rs` | Target resolution, area queries |
| `src/ai/core/hero/resolution.rs` | Ability resolution, cooldown/charge mechanics |
| `src/ai/core/hero/reactions.rs` | Morph, form swap, zone reactions |
| `src/ai/core/tick_systems.rs` | Cooldown ticks, status effects, projectile advancement |
| `src/ai/core/tick_world.rs` | Zone, channel, tether tick updates |
| `src/mission/hero_templates.rs` | TOML loading, `parse_hero_toml`, `hero_toml_to_unit` |
| `assets/hero_templates/*.toml` | 27 hero definitions |

---

## Hero TOML Format

Every hero is a TOML file in `assets/hero_templates/`. The struct is `HeroToml` in `src/ai/effects/defs.rs`.

```toml
[hero]
name = "Paladin"

[stats]
hp = 160
move_speed = 2.4
resource = 100         # optional mana/energy pool
max_resource = 100
resource_regen_per_sec = 5.0
armor = 30.0           # physical damage reduction
magic_resist = 25.0    # magic damage reduction

[stats.tags]           # resistance tags (higher = more resistant)
CROWD_CONTROL = 50.0
HOLY = 40.0

[attack]
damage = 14
range = 1.5
cooldown = 1000        # ms between auto-attacks
cast_time = 300        # ms wind-up

# --- Active abilities (typically 8-9 per hero) ---
[[abilities]]
name = "HolyStrike"
targeting = "target_enemy"    # see Targeting Modes
range = 1.8
cooldown_ms = 3500
cast_time_ms = 300
ai_hint = "damage"           # damage | heal | defense | crowd_control | utility
resource_cost = 20            # optional

[[abilities.effects]]         # instant effects
type = "damage"
amount = 30
damage_type = "magic"         # physical (default) | magic | true

[abilities.effects.tags]      # effect power tags
HOLY = 50.0

# --- Delivery-based abilities ---
[[abilities]]
name = "Pyroblast"
targeting = "target_enemy"
range = 5.0
cooldown_ms = 15000
cast_time_ms = 800
ai_hint = "damage"

[abilities.delivery]          # how the effect travels
method = "projectile"
speed = 6.0
pierce = false
width = 0.5

[[abilities.delivery.on_hit]] # effects applied on projectile hit
type = "damage"
amount = 75

# --- Charge/Ammo abilities ---
[[abilities]]
name = "ShieldBash"
targeting = "target_enemy"
range = 1.5
cooldown_ms = 0              # charges handle availability
max_charges = 2              # starts with 2 charges
charge_recharge_ms = 8000    # one charge every 8s
ai_hint = "damage"

# --- Toggle abilities ---
[[abilities]]
name = "AuraOfFaith"
targeting = "self_cast"
is_toggle = true
toggle_cost_per_sec = 8.0    # drains resource while active
ai_hint = "utility"

# --- Recast abilities ---
[[abilities]]
name = "TripleSlash"
targeting = "target_enemy"
range = 1.8
cooldown_ms = 10000
cast_time_ms = 200
ai_hint = "damage"
recast_count = 3             # 3 casts total
recast_window_ms = 4000      # must recast within 4s

[[abilities.effects]]
type = "damage"
amount = 20

# recast_effects[0] = 2nd cast effects, recast_effects[1] = 3rd cast effects
# (defined as arrays of ConditionalEffect)

# --- Unstoppable abilities ---
[[abilities]]
name = "UnstoppableCharge"
targeting = "target_enemy"
range = 6.0
cooldown_ms = 20000
cast_time_ms = 1000
ai_hint = "crowd_control"
unstoppable = true           # immune to CC during cast

# --- Form swap abilities ---
[[abilities]]
name = "StanceSwitch"
targeting = "self_cast"
cooldown_ms = 3000
swap_form = "cougar"         # swaps all abilities tagged with this form

[[abilities]]
name = "Pounce"
targeting = "target_enemy"
range = 2.0
form = "cougar"              # belongs to the "cougar" form group
ai_hint = "damage"

# --- Passive abilities (typically 2 per hero) ---
[[passives]]
name = "RetributionAura"
cooldown_ms = 0
range = 3.5

[passives.trigger]
type = "periodic"
interval_ms = 2000

[[passives.effects]]
type = "reflect"
percent = 18.0
duration_ms = 2500

[passives.effects.area]
shape = "circle"
radius = 3.5
```

### Morph System

Abilities can temporarily transform into another ability via `morph_into`:

```toml
[[abilities]]
name = "StanceA"
morph_into = { name = "StanceB", targeting = "self_cast", ... }
morph_duration_ms = 5000
```

Defined in `AbilityDef` at `src/ai/effects/defs.rs`. Applied in `src/ai/core/hero/reactions.rs`.

### Zone Reactions

Abilities with a `zone_tag` can trigger combo reactions when zones overlap:

```toml
[[abilities]]
name = "FireWall"
zone_tag = "fire"            # fire + frost, fire + lightning, frost + lightning
```

Combo logic in `src/ai/core/hero/reactions.rs:check_zone_reactions`.

### Evolution System

Abilities can permanently upgrade via `evolve_into`:

```toml
[[abilities]]
name = "BasicSlash"
evolve_into = { name = "EmpoweredSlash", ... }
```

Triggered by the `EvolveAbility` effect.

---

## Effect Types

45 effect types defined in `src/ai/effects/effect_enum.rs`. Each is a variant of the `Effect` enum.

### Core Combat

| Effect | Fields | Description |
|--------|--------|-------------|
| `damage` | `amount`, `amount_per_tick`, `duration_ms`, `tick_interval_ms`, `scaling_stat`, `scaling_percent`, `damage_type` | Direct damage or DoT. Supports physical/magic/true via `DamageType`. |
| `heal` | `amount`, `amount_per_tick`, `duration_ms`, `tick_interval_ms`, `scaling_stat`, `scaling_percent` | Direct heal or HoT |
| `shield` | `amount`, `duration_ms` | Temporary HP |
| `self_damage` | `amount` | HP cost abilities |
| `execute` | `hp_threshold_percent` | Kill below %HP |
| `lifesteal` | `percent`, `duration_ms` | Damage-to-heal conversion |
| `reflect` | `percent`, `duration_ms` | Return damage to attacker |
| `damage_modify` | `factor`, `duration_ms` | Increase/decrease damage taken |

### Crowd Control

| Effect | Fields | Description |
|--------|--------|-------------|
| `stun` | `duration_ms` | Cannot act |
| `root` | `duration_ms` | Cannot move |
| `silence` | `duration_ms` | Cannot cast abilities |
| `slow` | `factor`, `duration_ms` | Reduced move speed |
| `fear` | `duration_ms` | Run away from source |
| `taunt` | `duration_ms` | Forced to attack taunter |
| `blind` | `miss_chance`, `duration_ms` | Attacks can miss |
| `polymorph` | `duration_ms` | Cannot act, reduced stats |
| `banish` | `duration_ms` | Untargetable + cannot act |
| `confuse` | `duration_ms` | Random targeting |
| `charm` | `duration_ms` | Walk toward source |
| `suppress` | `duration_ms` | Hard CC, cannot be cleansed by normal means |
| `grounded` | `duration_ms` | Prevents dashes, blinks, and movement abilities |

### Positioning

| Effect | Fields | Description |
|--------|--------|-------------|
| `dash` | `to_target`, `distance`, `to_position`, `is_blink` | Self-movement. `is_blink = true` for instant teleport (ignores terrain/grounded). Default distance: 2.0. |
| `knockback` | `distance` | Push enemy away |
| `pull` | `distance` | Pull enemy closer |
| `swap` | (none) | Swap positions with target |

### Buffs & Debuffs

| Effect | Fields | Description |
|--------|--------|-------------|
| `buff` | `stat`, `factor`, `duration_ms` | Stat increase (stats: `damage_output`, `move_speed`) |
| `debuff` | `stat`, `factor`, `duration_ms` | Stat decrease |
| `on_hit_buff` | `duration_ms`, `on_hit_effects[]` | Add effects to auto-attacks |

### Summoning

| Effect | Fields | Description |
|--------|--------|-------------|
| `summon` | `template`, `count`, `hp_percent`, `clone`, `clone_damage_percent`, `directed` | Spawn ally units. `clone = true` copies caster stats/abilities. `directed = true` means summon attacks when owner attacks (doesn't act independently). Defaults: count=1, hp_percent=100, clone_damage_percent=75. |
| `command_summons` | `speed` | Move all owned directed summons toward a target position. Default speed: 8.0. |

### Healing & Shield

| Effect | Fields | Description |
|--------|--------|-------------|
| `resurrect` | `hp_percent` | Revive dead ally |
| `overheal_shield` | `duration_ms`, `conversion_percent` | Excess heal to shield. Default conversion: 100%. |
| `absorb_to_heal` | `shield_amount`, `duration_ms`, `heal_percent` | Absorb shield heals when it expires. Default heal: 50%. |
| `shield_steal` | `amount` | Take enemy shield |
| `status_clone` | `max_count` | Copy statuses. Default max: 3. |

### Status Interaction

| Effect | Fields | Description |
|--------|--------|-------------|
| `immunity` | `immune_to[]`, `duration_ms` | Status immunity |
| `death_mark` | `duration_ms`, `damage_percent` | Deferred damage. Default damage: 50%. |
| `detonate` | `damage_multiplier` | Pop accumulated status. Default multiplier: 1.0. |
| `status_transfer` | `steal_buffs` | Move statuses between units |
| `dispel` | `target_tags[]` | Remove status effects matching tags |

### Complex Mechanics

| Effect | Fields | Description |
|--------|--------|-------------|
| `duel` | `duration_ms` | Force 1v1 |
| `stealth` | `duration_ms`, `break_on_damage`, `break_on_ability` | Invisibility |
| `leash` | `max_range`, `duration_ms` | Tether to position |
| `link` | `duration_ms`, `share_percent` | Damage sharing. Default share: 50%. |
| `redirect` | `duration_ms`, `charges` | Redirect damage to protector. Default charges: 3. |
| `rewind` | `lookback_ms` | Restore previous state. Default lookback: 3000ms. |
| `cooldown_modify` | `amount_ms`, `ability_name` | Change ability cooldowns |
| `apply_stacks` | `name`, `count`, `max_stacks`, `duration_ms` | Stack system. Default count: 1, max: 4. |
| `obstacle` | `width`, `height` | Terrain creation |
| `projectile_block` | `duration_ms` | Blocks enemy projectiles in an area |
| `attach` | `duration_ms` | Attach to an ally — become untargetable and move with them. 0 = until recast. |
| `evolve_ability` | `ability_index` | Permanently replace an ability with its `evolve_into` variant |

---

## Area Shapes

7 shapes defined in `src/ai/effects/types.rs`.

| Shape | Fields | Usage |
|-------|--------|-------|
| `single_target` | (none) | Default, hits one unit |
| `circle` | `radius` | AoE around target/self |
| `cone` | `radius`, `angle_deg` | Fan in facing direction |
| `line` | `length`, `width` | Rectangular strip |
| `ring` | `inner_radius`, `outer_radius` | Donut shape |
| `self` | (none) | Only affects caster |
| `spread` | `radius`, `max_targets` | Bounces to nearby targets |

---

## Delivery Methods

7 delivery methods defined in `src/ai/effects/types.rs`.

| Method | Fields | Usage |
|--------|--------|-------|
| `instant` | (none) | Default, immediate application |
| `projectile` | `speed`, `pierce`, `width`, `on_hit[]`, `on_arrival[]` | Traveling missile. Supports skillshots via `max_travel_distance` on the runtime `Projectile` struct. |
| `channel` | `duration_ms`, `tick_interval_ms` | Sustained cast, interruptible |
| `zone` | `duration_ms`, `tick_interval_ms` | Persistent ground area |
| `tether` | `max_range`, `tick_interval_ms`, `on_complete[]` | Unit-to-unit link |
| `trap` | `duration_ms`, `trigger_radius`, `arm_time_ms` | Placed mine/trap |
| `chain` | `bounces`, `bounce_range`, `falloff`, `on_hit[]` | Bouncing projectile |

---

## Conditions

24 conditions defined in `src/ai/effects/types.rs`. Evaluated per-effect to conditionally apply.

| Condition | Fields | Usage |
|-----------|--------|-------|
| `always` | (none) | Default |
| `target_hp_below` | `percent` | Execute range |
| `target_hp_above` | `percent` | Full HP bonus |
| `target_is_stunned` | (none) | Combo bonus |
| `target_is_slowed` | (none) | Combo bonus |
| `target_is_rooted` | (none) | Combo bonus |
| `target_is_silenced` | (none) | Combo bonus |
| `target_is_feared` | (none) | Combo bonus |
| `target_is_taunted` | (none) | Combo bonus |
| `target_is_banished` | (none) | Combo bonus |
| `target_is_stealthed` | (none) | Reveal bonus |
| `target_is_charmed` | (none) | Combo bonus |
| `target_is_polymorphed` | (none) | Combo bonus |
| `caster_hp_below` | `percent` | Low HP bonus |
| `caster_hp_above` | `percent` | High HP bonus |
| `hit_count_above` | `count` | Multi-hit scaling |
| `target_has_tag` | `tag` | Tag interaction |
| `caster_has_status` | `status` | Requires self-status |
| `target_has_status` | `status` | Requires target status |
| `target_debuff_count` | `min_count` | Debuff stacking |
| `caster_buff_count` | `min_count` | Buff stacking |
| `ally_count_below` | `count` | Last stand |
| `enemy_count_below` | `count` | Cleanup |
| `target_stack_count` | `name`, `min_count` | Stack threshold |

---

## Passive Triggers

19 triggers defined in `src/ai/effects/types.rs`. Drive passive ability activation.

| Trigger | Fields | Usage |
|---------|--------|-------|
| `on_damage_dealt` | (none) | After dealing damage |
| `on_damage_taken` | (none) | After taking damage |
| `on_kill` | (none) | After killing a unit |
| `on_ally_damaged` | `range` | Nearby ally takes damage. Default range: 5.0. |
| `on_death` | (none) | On unit death |
| `on_ability_used` | (none) | After casting any ability |
| `on_hp_below` | `percent` | HP drops below threshold |
| `on_hp_above` | `percent` | HP rises above threshold |
| `on_shield_broken` | (none) | Shield expires/breaks |
| `on_stun_expire` | (none) | Stun wears off |
| `periodic` | `interval_ms` | Every N ms |
| `on_heal_received` | (none) | After receiving heal |
| `on_status_applied` | (none) | After gaining a status |
| `on_status_expired` | (none) | After losing a status |
| `on_resurrect` | (none) | After being revived |
| `on_dodge` | (none) | After dodging an attack |
| `on_reflect` | (none) | After reflecting damage |
| `on_ally_killed` | `range` | Nearby ally dies. Default range: 5.0. |
| `on_auto_attack` | (none) | On each auto-attack |
| `on_stack_reached` | `name`, `count` | Stack count threshold |

---

## Targeting Modes

8 modes defined in `src/ai/effects/defs.rs`.

| Mode | TOML value | Behavior |
|------|------------|----------|
| Target Enemy | `target_enemy` | Click enemy unit (default) |
| Target Ally | `target_ally` | Click ally unit |
| Self Cast | `self_cast` | No target needed |
| Self AoE | `self_aoe` | AoE centered on caster |
| Ground Target | `ground_target` | Click position on ground |
| Direction | `direction` | Fire in a direction (skillshot) |
| Vector | `vector` | Click-drag vector targeting (start point + direction). Used by Rumble R, Viktor E. |
| Global | `global` | Hits all enemies on the map regardless of range. Used by Karthus R, Soraka R. |

---

## Damage Types

3 damage types defined in `src/ai/effects/types.rs` as the `DamageType` enum. Applied on the `Damage` effect variant.

| Type | Interaction |
|------|-------------|
| `physical` | Reduced by `armor` stat (default) |
| `magic` | Reduced by `magic_resist` stat |
| `true` | Ignores all damage reduction |

`HeroStats` includes `armor: f32` and `magic_resist: f32` fields for damage type resolution.

---

## Status Effects

Live status tracking via `ActiveStatusEffect` in `src/ai/effects/defs.rs`. Each status has a `StatusKind`, source unit, remaining duration, tags, and stacking mode.

### Stacking Modes

| Mode | Behavior |
|------|----------|
| `refresh` | Reset duration (default) |
| `extend` | Add duration to existing |
| `strongest` | Keep only the highest-value instance |
| `stack` | Allow multiple independent instances |

### StatusKind Variants

Core: `Stun`, `Slow { factor }`, `Dot { amount_per_tick, tick_interval_ms }`, `Hot { amount_per_tick, tick_interval_ms }`, `Shield { amount }`, `Buff { stat, factor }`, `Debuff { stat, factor }`, `Duel { partner_id }`

CC: `Root`, `Silence`, `Fear { source_pos }`, `Taunt { taunter_id }`, `Polymorph`, `Banish`, `Confuse`, `Charm { original_team }`, `Suppress`, `Grounded`

Damage Modifiers: `Reflect { percent }`, `Lifesteal { percent }`, `DamageModify { factor }`, `Blind { miss_chance }`, `OnHitBuff { effects }`

Healing/Shield: `OverhealShield { conversion_percent }`, `AbsorbShield { amount, heal_percent }`

Status Interaction: `Immunity { immune_to }`, `DeathMark { accumulated_damage, damage_percent }`

Complex: `Stealth { break_on_damage, break_on_ability }`, `Leash { anchor_pos, max_range }`, `Link { partner_id, share_percent }`, `Redirect { protector_id, charges }`, `Attached { host_id }`

Stacks: `Stacks { name, count, max_stacks }`

---

## LoL Champion Dataset

172 champions scraped from Riot Data Dragon + LoL Wiki. Each champion has 5 abilities (Passive, Q, W, E, R).

### Location

```
assets/lol_champions/*.json     # 172 files, 3.5 MB total
scripts/fetch_lol_champions.py  # fetch script (resumable, parallel)
```

### JSON Structure

```json
{
  "id": "Jinx",
  "name": "Jinx",
  "title": "the Loose Cannon",
  "resource_type": "Mana",
  "tags": ["Marksman"],
  "abilities": {
    "passive": {
      "name": "Get Excited!",
      "description": "..."
    },
    "Q": {
      "id": "JinxQ",
      "name": "Switcheroo!",
      "description": "...",
      "tooltip": "...",
      "cooldown": "0.9",
      "cost": "20",
      "cost_type": "Mana Per Rocket",
      "range": "600",
      "max_rank": 5,
      "effect_burn": [...],
      "leveltip": {...},
      "resource": "...",
      "wiki_detail": {
        "champion": "Jinx",
        "targeting": "Auto",
        "damagetype": "Physical",
        "description": "...(detailed wikitext)...",
        "leveling": "...(exact values/ratios)...",
        "cast time": "...",
        "cooldown": "...",
        "cost": "...",
        "variables": {"b1": "10", "r1": "60", ...}
      }
    }
  }
}
```

### Data Dragon Fields (every ability)

| Field | Description |
|-------|-------------|
| `name` | Ability name |
| `description` | Clean-text description |
| `tooltip` | Template tooltip with variable placeholders |
| `cooldown` | Cooldown per rank, slash-separated (e.g. `"14/12/10/8/6"`) |
| `cost` | Cost per rank |
| `cost_type` | Mana, Energy, etc. |
| `range` | Cast range |
| `max_rank` | Number of ranks (usually 5) |

### Wiki Detail Fields (167/172 champions, 796/860 abilities)

| Field | Count | Description |
|-------|-------|-------------|
| `targeting` | 772 | Auto, Direction, Location, Unit, Passive, Vector |
| `damagetype` | 672 | Physical, Magic, True |
| `description` + `description2-6` | 794 | Full mechanical description with exact values |
| `leveling` + `leveling2-3` | 459 | Exact damage/scaling per rank |
| `variables` | varies | Extracted base values and ratios |
| `cooldown` | 607 | Exact cooldown values |
| `cost` / `costtype` | 547 | Resource costs |
| `cast time` | 613 | Cast time in seconds |
| `effect radius` | 354 | AoE radius |
| `target range` | 333 | Targeting range |
| `speed` | 277 | Projectile/missile speed |
| `width` | 189 | Projectile/skillshot width |
| `spellshield` | 565 | Whether spell shield blocks it |
| `spelleffects` | 652 | Spell effect classification |
| `notes` | 679 | Detailed mechanical notes |

### LoL Mechanic Frequency (across all 860 abilities)

| Mechanic | Count | System Support |
|----------|-------|----------------|
| Heal | 151 | `effect.heal` |
| Slow | 144 | `effect.slow` |
| Knockback/up | 65 | `effect.knockback` |
| Shield | 64 | `effect.shield` |
| Stun | 59 | `effect.stun` |
| Dash/leap | 82 | `effect.dash` (includes `is_blink` for teleports) |
| Recast | 114 | `recast_count` + `recast_window_ms` + `recast_effects` on `AbilityDef` |
| Charge/ammo | 112 | `max_charges` + `charge_recharge_ms` on `AbilityDef` |
| Stack-based | 40 | `effect.apply_stacks` |
| %HP damage | 76 | `scaling_stat: "target_max_hp"` / `"target_missing_hp"` on `Damage` |
| On-hit effects | 33 | `effect.on_hit_buff` |
| Summon/pet | 27 | `effect.summon` with `directed` flag + `effect.command_summons` |
| Root/snare | 25 | `effect.root` |
| Wall/terrain | 19 | `effect.obstacle` |
| True damage | 17 | `damage_type = "true"` on `Damage` effect |
| Channel | 16 | `delivery.channel` |
| Toggle | 15 | `is_toggle` + `toggle_cost_per_sec` on `AbilityDef` |
| Terrain creation | 15 | `effect.obstacle` |
| Suppress | 17 | `effect.suppress` |
| Untargetable | 12 | `effect.banish` |
| Transform/stance | 13 | `swap_form` + `form` on `AbilityDef` |
| Teleport | 12 | `effect.dash { is_blink = true }` |
| Unstoppable | 12 | `unstoppable = true` on `AbilityDef` |
| Execute (%HP) | 11 | `effect.execute` |
| Invisible | 9 | `effect.stealth` |
| Bounce | 9 | `delivery.chain` |
| Trap | 9 | `delivery.trap` |
| Clone | 8 | `effect.summon { clone = true }` |
| Evolve | 8 | `evolve_into` on `AbilityDef` + `effect.evolve_ability` |
| Global range | 42 | `targeting = "global"` |
| Attach | 7 | `effect.attach` |
| Fear | 7 | `effect.fear` |
| Vector targeting | 4 | `targeting = "vector"` |
| Grounded | 31 | `effect.grounded` |
| Projectile blocking | ~5 | `effect.projectile_block` |

---

## Gap Analysis

Most previously identified gaps have been implemented. Remaining gaps are limited to advanced AI behaviors that require specialized handling beyond the data-driven ability system.

### Implemented (formerly gaps)

| Mechanic | Implementation |
|----------|---------------|
| Recast system | `recast_count`, `recast_window_ms`, `recast_effects` on `AbilityDef`; `recasts_remaining`, `recast_window_remaining_ms` on `AbilitySlot` |
| Charge/ammo system | `max_charges`, `charge_recharge_ms` on `AbilityDef`; `charges`, `charge_recharge_remaining_ms` on `AbilitySlot` |
| Damage types (Phys/Magic/True) | `DamageType` enum on `Damage` effect; `armor`, `magic_resist` on `HeroStats` |
| Toggle abilities | `is_toggle`, `toggle_cost_per_sec` on `AbilityDef`; `toggled_on` on `AbilitySlot` |
| Suppress CC | `Effect::Suppress`, `StatusKind::Suppress` |
| Teleport/Blink | `is_blink` field on `Effect::Dash` |
| Unstoppable / CC immunity during cast | `unstoppable` flag on `AbilityDef` |
| Transform/stance swap | `swap_form` + `form` fields on `AbilityDef` |
| Clone mechanic | `clone`, `clone_damage_percent` fields on `Effect::Summon` |
| Ability evolution | `evolve_into` on `AbilityDef` + `Effect::EvolveAbility` |
| Vector targeting | `AbilityTargeting::Vector` |
| Global targeting | `AbilityTargeting::Global` |
| Attach mechanic | `Effect::Attach`, `StatusKind::Attached` |
| Projectile blocking | `Effect::ProjectileBlock` |
| Grounded effect | `Effect::Grounded`, `StatusKind::Grounded` |
| Directed summons | `directed` flag on `Effect::Summon` + `Effect::CommandSummons` |
| %HP damage | `scaling_stat` supports `"target_max_hp"` and `"target_missing_hp"` |

### Remaining Gaps

| Mechanic | Impact | Notes |
|----------|--------|-------|
| Pet AI | Low | Directed summons cover most cases; fully autonomous pet AI (Tibbers, Daisy) would need separate intent generation |
| Ability combos / Marks | Low | Some champions (Akali, Zed) apply marks then detonate. Partially expressible via `apply_stacks` + `detonate` but exact mark-target tracking is limited. |

---

## Translation Rules: LoL -> Hero TOML

When converting LoL champion data to TOML:

### Stat Scaling

LoL HP is ~2000, game HP is ~65-180. Divide all values by ~15-20:

| LoL Value | Game Value | Multiplier |
|-----------|-----------|------------|
| HP (2000) | ~130 | /15 |
| Damage (300) | ~20 | /15 |
| Shield (500) | ~33 | /15 |
| Heal (400) | ~27 | /15 |
| Range (550 units) | ~3.5 | /160 |
| Move speed (340) | ~2.8 | /120 |

### Cooldowns

LoL cooldowns in seconds -> game in ms. Use rank 3 (mid) value x 1000.

### Abilities

LoL has 5 (P/Q/W/E/R), game expects ~8 active + 2 passive.

- LoL Passive -> game passive
- LoL Q/W/E/R -> 4 active abilities
- Multi-part abilities (Aatrox Q 3 casts) -> use `recast_count` + `recast_effects`
- Transform champions (Jayce, Nidalee) -> use `swap_form` + `form` tags
- Remaining slots: derive from passive interactions or leave empty

### AI Hints

| LoL targeting | wiki `damagetype` | Suggested `ai_hint` |
|---------------|-------------------|---------------------|
| Direction + Physical/Magic | high damage | `"damage"` |
| Unit + has CC | stun/root/slow | `"crowd_control"` |
| Self/Auto + shield/heal | - | `"defense"` / `"heal"` |
| Self + buff/stealth | - | `"utility"` |

### Resource Types

| LoL Resource | Game Mapping |
|-------------|-------------|
| Mana | `resource = 100-500`, `resource_regen_per_sec = 5-15` |
| Energy | `resource = 200`, `resource_regen_per_sec = 50` (fast regen, small pool) |
| No cost | omit resource fields |
| Health cost | `effect.self_damage` |
