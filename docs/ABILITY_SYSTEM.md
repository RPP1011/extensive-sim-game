# Ability System Reference

Complete reference for the data-driven ability engine, the LoL champion dataset, and the gap analysis for full coverage.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Hero TOML Format](#hero-toml-format)
3. [Effect Types (32)](#effect-types)
4. [Area Shapes (7)](#area-shapes)
5. [Delivery Methods (7)](#delivery-methods)
6. [Conditions (20)](#conditions)
7. [Passive Triggers (17)](#passive-triggers)
8. [Targeting Modes (6)](#targeting-modes)
9. [LoL Champion Dataset](#lol-champion-dataset)
10. [Gap Analysis](#gap-analysis)

---

## System Architecture

Five composable dimensions define every ability:

```
WHAT (Effect) x WHERE (Area) x HOW (Delivery) x WHEN (Condition/Trigger) + Tags
```

### Key Files

| File | Purpose |
|------|---------|
| `src/ai/effects/types.rs` | Effect, Area, Delivery, Condition, Trigger enums |
| `src/ai/effects/defs.rs` | AbilityDef, PassiveDef, HeroToml, AbilitySlot, runtime types |
| `src/ai/core/apply_effect.rs` | Main effect dispatcher (handles all 32 effect types) |
| `src/ai/core/apply_effect_ext.rs` | Extended effect application (phases 4-7) |
| `src/ai/core/damage.rs` | Damage/heal application, chain delivery, stat scaling |
| `src/ai/core/triggers.rs` | Passive trigger system (check + fire) |
| `src/ai/core/targeting.rs` | Target resolution, area queries |
| `src/ai/core/hero.rs` | Ability resolution, morph system |
| `src/ai/core/tick_systems.rs` | Cooldown ticks, status effects, projectile advancement |
| `src/ai/core/tick_world.rs` | Zone, channel, tether tick updates |
| `src/mission/hero_templates.rs` | TOML loading, `parse_hero_toml`, `hero_toml_to_unit` |
| `assets/hero_templates/*.toml` | 27 hero definitions |

---

## Hero TOML Format

Every hero is a TOML file in `assets/hero_templates/`. The struct is `HeroToml` in `src/ai/effects/defs.rs:236`.

```toml
[hero]
name = "Paladin"

[stats]
hp = 160
move_speed = 2.4
resource = 100         # optional mana/energy pool
max_resource = 100
resource_regen_per_sec = 5.0

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

Defined in `AbilityDef` at `src/ai/effects/defs.rs:48-50`. Applied in `src/ai/core/hero.rs`.

---

## Effect Types

32 effect types defined in `src/ai/effects/types.rs:47-273`. Each is a variant of the `Effect` enum.

### Core Combat

| Effect | Fields | Example |
|--------|--------|---------|
| `damage` | `amount`, `amount_per_tick`, `duration_ms`, `tick_interval_ms`, `scaling_stat`, `scaling_percent` | Direct damage or DoT |
| `heal` | `amount`, `amount_per_tick`, `duration_ms`, `tick_interval_ms`, `scaling_stat`, `scaling_percent` | Direct heal or HoT |
| `shield` | `amount`, `duration_ms` | Temporary HP |
| `self_damage` | `amount` | HP cost abilities |
| `execute` | `hp_threshold_percent` | Kill below %HP |
| `lifesteal` | `percent`, `duration_ms` | Damage-to-heal conversion |
| `reflect` | `percent`, `duration_ms` | Return damage to attacker |
| `damage_modify` | `factor`, `duration_ms` | Increase/decrease damage taken |

### Crowd Control

| Effect | Fields | Example |
|--------|--------|---------|
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

### Positioning

| Effect | Fields | Example |
|--------|--------|---------|
| `dash` | `to_target`, `distance`, `to_position` | Self-movement |
| `knockback` | `distance` | Push enemy away |
| `pull` | `distance` | Pull enemy closer |
| `swap` | (none) | Swap positions with target |

### Buffs & Debuffs

| Effect | Fields | Example |
|--------|--------|---------|
| `buff` | `stat`, `factor`, `duration_ms` | Stat increase (stats: `damage_output`, `move_speed`) |
| `debuff` | `stat`, `factor`, `duration_ms` | Stat decrease |
| `on_hit_buff` | `duration_ms`, `on_hit_effects[]` | Add effects to auto-attacks |

### Advanced

| Effect | Fields | Example |
|--------|--------|---------|
| `summon` | `template`, `count`, `hp_percent` | Spawn ally units |
| `dispel` | `target_tags[]` | Remove status effects |
| `duel` | `duration_ms` | Force 1v1 |
| `stealth` | `duration_ms`, `break_on_damage`, `break_on_ability` | Invisibility |
| `leash` | `max_range`, `duration_ms` | Tether to position |
| `link` | `duration_ms`, `share_percent` | Damage sharing |
| `redirect` | `duration_ms`, `charges` | Redirect damage to protector |
| `rewind` | `lookback_ms` | Restore previous state |
| `cooldown_modify` | `amount_ms`, `ability_name` | Change ability cooldowns |
| `apply_stacks` | `name`, `count`, `max_stacks`, `duration_ms` | Stack system |
| `immunity` | `immune_to[]`, `duration_ms` | Status immunity |
| `death_mark` | `duration_ms`, `damage_percent` | Deferred damage |
| `detonate` | `damage_multiplier` | Pop accumulated status |
| `status_transfer` | `steal_buffs` | Move statuses between units |
| `resurrect` | `hp_percent` | Revive dead ally |
| `overheal_shield` | `duration_ms`, `conversion_percent` | Excess heal to shield |
| `absorb_to_heal` | `shield_amount`, `duration_ms`, `heal_percent` | Absorb shield heals |
| `shield_steal` | `amount` | Take enemy shield |
| `status_clone` | `max_count` | Copy statuses |
| `obstacle` | `width`, `height` | Terrain creation |

---

## Area Shapes

7 shapes defined in `src/ai/effects/types.rs:319-345`.

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

7 delivery methods defined in `src/ai/effects/types.rs:357-401`.

| Method | Fields | Usage |
|--------|--------|-------|
| `instant` | (none) | Default, immediate application |
| `projectile` | `speed`, `pierce`, `width`, `on_hit[]`, `on_arrival[]` | Traveling missile |
| `channel` | `duration_ms`, `tick_interval_ms` | Sustained cast, interruptible |
| `zone` | `duration_ms`, `tick_interval_ms` | Persistent ground area |
| `tether` | `max_range`, `tick_interval_ms`, `on_complete[]` | Unit-to-unit link |
| `trap` | `duration_ms`, `trigger_radius`, `arm_time_ms` | Placed mine/trap |
| `chain` | `bounces`, `bounce_range`, `falloff`, `on_hit[]` | Bouncing projectile |

---

## Conditions

20 conditions defined in `src/ai/effects/types.rs:413-441`. Evaluated per-effect to conditionally apply.

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

17 triggers defined in `src/ai/effects/types.rs:449-482`. Drive passive ability activation.

| Trigger | Fields | Usage |
|---------|--------|-------|
| `on_damage_dealt` | (none) | After dealing damage |
| `on_damage_taken` | (none) | After taking damage |
| `on_kill` | (none) | After killing a unit |
| `on_ally_damaged` | `range` | Nearby ally takes damage |
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
| `on_ally_killed` | `range` | Nearby ally dies |
| `on_auto_attack` | (none) | On each auto-attack |
| `on_stack_reached` | `name`, `count` | Stack count threshold |

---

## Targeting Modes

6 modes defined in `src/ai/effects/defs.rs:13-20`.

| Mode | TOML value | Behavior |
|------|------------|----------|
| Target Enemy | `target_enemy` | Click enemy unit |
| Target Ally | `target_ally` | Click ally unit |
| Self Cast | `self_cast` | No target needed |
| Self AoE | `self_aoe` | AoE centered on caster |
| Ground Target | `ground_target` | Click position on ground |
| Direction | `direction` | Fire in a direction (skillshot) |

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

| Mechanic | Count | Current System Support |
|----------|-------|----------------------|
| Heal | 151 | `effect.heal` |
| Slow | 144 | `effect.slow` |
| Knockback/up | 65 | `effect.knockback` |
| Shield | 64 | `effect.shield` |
| Stun | 59 | `effect.stun` |
| Dash/leap | 82 | `effect.dash` |
| Recast | 114 | **GAP** — `morph_into` is partial |
| Charge/ammo | 112 | **GAP** — no charge system |
| Stack-based | 40 | `effect.apply_stacks` (partial) |
| %HP damage | 76 | **GAP** — no %HP scaling |
| On-hit effects | 33 | `effect.on_hit_buff` |
| Summon/pet | 27 | `effect.summon` (no pet AI) |
| Root/snare | 25 | `effect.root` |
| Wall/terrain | 19 | `effect.obstacle` (basic) |
| True damage | 17 | **GAP** — no damage type system |
| Channel | 16 | `delivery.channel` |
| Toggle | 15 | **GAP** — no toggle abilities |
| Terrain creation | 15 | `effect.obstacle` (partial) |
| Suppress | 17 | **GAP** — no suppress CC |
| Untargetable | 12 | `effect.banish` (partial) |
| Transform/stance | 13 | `morph_into` (partial) |
| Teleport | 12 | **GAP** — no teleport effect |
| Unstoppable | 12 | **GAP** — no CC immunity during cast |
| Execute (%HP) | 11 | `effect.execute` (threshold only, no scaling) |
| Invisible | 9 | `effect.stealth` |
| Bounce | 9 | `delivery.chain` |
| Trap | 9 | `delivery.trap` |
| Clone | 8 | **GAP** — no clone mechanic |
| Evolve | 8 | **GAP** — no ability evolution |
| Global range | 42 | Works (just set large range) |
| Attach | 7 | **GAP** — no attach mechanic |
| Fear | 7 | `effect.fear` |
| Vector targeting | 4 | **GAP** — no vector targeting mode |

---

## Gap Analysis

Mechanics needed to fully express all 172 LoL champions. Ordered by impact (how many champions use the mechanic).

### GAP 1: Recast System (114 abilities)

**Problem:** Many LoL abilities can be recast 1-3 times with different effects per cast (Aatrox Q, Ahri R, Riven Q). Current `morph_into` only supports one transformation with a timer.

**Examples:** Aatrox Q (3 casts, different hitboxes), Ahri R (3 dashes), Lee Sin Q (dash to marked target), Akali R (2 casts with different effects).

**Proposal:** Add a `recast` field to `AbilityDef`:
```toml
[[abilities]]
name = "TheDarkinBlade"
recast_count = 3
recast_window_ms = 4000
recast_effects = [...]   # different effects per cast
```

**Files to modify:** `src/ai/effects/defs.rs` (AbilityDef), `src/ai/core/hero.rs` (resolve_hero_ability)

### GAP 2: Charge/Ammo System (112 abilities)

**Problem:** Many abilities have multiple charges that regenerate independently (Ammu Q has 2 charges, Akali R has 2 charges, Teemo R stores mushrooms). No charge system exists.

**Examples:** Teemo R (stores up to 3 mushrooms), Corki R (stores up to 7 missiles), Riven E (1 charge with independent CD).

**Proposal:** Add `max_charges` and `charge_recharge_ms` to `AbilityDef`:
```toml
[[abilities]]
name = "NoxianFervor"
max_charges = 3
charge_recharge_ms = 12000
```

**Files to modify:** `src/ai/effects/defs.rs` (AbilityDef), `src/ai/effects/defs.rs` (AbilitySlot — add `charges: u32`), `src/ai/core/tick_systems.rs` (charge regeneration)

### GAP 3: %HP Damage Scaling (76 abilities)

**Problem:** Many abilities deal damage based on target's max/current/missing HP. Current `damage` only supports flat amounts and stat-based scaling.

**Examples:** Aatrox passive (% max HP), Vayne W (% max HP true damage), Lee Sin Q recast (% missing HP), Elise Q (% current HP).

**Proposal:** Add `hp_scaling` fields to the `Damage` effect:
```toml
[[abilities.effects]]
type = "damage"
target_max_hp_percent = 8.0     # deals 8% of target's max HP
target_current_hp_percent = 0.0
target_missing_hp_percent = 0.0
```

**Files to modify:** `src/ai/effects/types.rs` (Effect::Damage), `src/ai/core/damage.rs` (apply_damage_to_unit)

### GAP 4: Damage Types (Physical/Magic/True) (all abilities)

**Problem:** LoL has three distinct damage types (physical, magic, true) that interact with armor/MR. Current system uses flat tags but no damage type resolution — there's no armor/MR stat or damage type reduction.

**Note:** The tag system (`FIRE = 50.0`, `HOLY = 40.0`) already provides a resistance framework. Damage types could be implemented as special tags or as a first-class field.

**Proposal:** Add optional `damage_type` field to `Damage` effect:
```toml
[[abilities.effects]]
type = "damage"
amount = 30
damage_type = "magic"   # physical | magic | true
```

**Files to modify:** `src/ai/effects/types.rs` (Effect::Damage), `src/ai/core/damage.rs`, `src/ai/effects/defs.rs` (HeroStats — add `armor`, `magic_resist`)

### GAP 5: Toggle Abilities (15 abilities)

**Problem:** Some abilities toggle on/off with per-second costs (Amumu W, Anivia R, Singed Q). No toggle state exists.

**Examples:** Amumu W (persistent AoE drain), Anivia R (channeled blizzard), Karthus E (AoE damage aura).

**Proposal:** Add `is_toggle` flag to `AbilityDef`:
```toml
[[abilities]]
name = "Despair"
is_toggle = true
resource_cost_per_sec = 8
```

**Files to modify:** `src/ai/effects/defs.rs` (AbilityDef), `src/ai/effects/defs.rs` (AbilitySlot — add `toggled_on: bool`), `src/ai/core/tick_systems.rs` (per-tick drain)

### GAP 6: Suppress CC (17 abilities)

**Problem:** Suppress is a hard CC that also prevents summoner spells and cannot be cleansed by normal means. Currently no suppress effect.

**Examples:** Malzahar R, Warwick R, Skarner R, Urgot R.

**Proposal:** Add `Suppress` variant to `Effect`:
```toml
[[abilities.effects]]
type = "suppress"
duration_ms = 2500
```

**Files to modify:** `src/ai/effects/types.rs` (Effect, StatusKind), `src/ai/core/apply_effect.rs`

### GAP 7: Teleport/Blink Effect (12 abilities)

**Problem:** `Dash` moves along a path (interruptible, can be blocked). Blink/teleport is instant repositioning. Several champions need this (Ezreal E, Kassadin R, Flash).

**Proposal:** Add `blink` field to `Dash` or new `Teleport` effect:
```toml
[[abilities.effects]]
type = "dash"
distance = 4.0
is_blink = true    # instant, ignores terrain/units
```

**Files to modify:** `src/ai/effects/types.rs` (Effect::Dash), `src/ai/core/apply_effect.rs`

### GAP 8: Unstoppable / CC Immunity During Cast (12 abilities)

**Problem:** Some abilities grant CC immunity during their cast (Malphite R, Sion R, Vi R). No way to express this.

**Proposal:** Add `unstoppable` flag to `AbilityDef`:
```toml
[[abilities]]
name = "UnstoppableForce"
unstoppable = true   # immune to CC during cast
```

**Files to modify:** `src/ai/effects/defs.rs` (AbilityDef), `src/ai/core/resolve.rs` (skip CC checks during unstoppable casts)

### GAP 9: Transform/Stance Swap (13 abilities)

**Problem:** Champions like Jayce, Nidalee, and Elise swap their entire ability kit. Current `morph_into` only transforms one ability at a time.

**Examples:** Jayce R (swap all 3 basic abilities), Nidalee R (human/cougar), Elise R (human/spider).

**Proposal:** Add `form` system — ability sets that swap as a group:
```toml
[[abilities]]
name = "Transform"
targeting = "self_cast"
swap_form = "cougar"   # swaps all abilities tagged with this form
```

**Files to modify:** `src/ai/effects/defs.rs` (AbilityDef — add `form` tag), `src/ai/core/hero.rs` (form swap logic)

### GAP 10: Clone Mechanic (8 abilities)

**Problem:** Shaco R, LeBlanc passive, Wukong W create controllable/uncontrollable clones. `Summon` exists but clones are supposed to copy the caster's appearance/stats.

**Proposal:** Extend `Summon` with a `clone` flag:
```toml
[[abilities.effects]]
type = "summon"
template = "self_clone"
clone = true           # copies caster stats/appearance
clone_damage_percent = 75
clone_damage_taken_percent = 150
```

**Files to modify:** `src/ai/effects/types.rs` (Effect::Summon), `src/ai/core/apply_effect.rs`

### GAP 11: Ability Evolution (8 abilities)

**Problem:** Kha'Zix and Kai'Sa can evolve abilities mid-game, permanently changing their effects. No upgrade/evolution system exists.

**Proposal:** Add `evolve_into` to `AbilityDef`:
```toml
[[abilities]]
name = "TasteTheirFear"
evolve_into = { name = "EvolvedTasteTheirFear", ... }
evolve_condition = { type = "target_stack_count", name = "evolution_points", min_count = 1 }
```

**Files to modify:** `src/ai/effects/defs.rs` (AbilityDef), `src/ai/core/hero.rs`

### GAP 12: Vector Targeting (4 abilities)

**Problem:** Rumble R, Viktor E, Taliyah W use click-drag vector targeting (start point + direction). Only 4 abilities use this.

**Proposal:** Add `vector` targeting mode:
```toml
[[abilities]]
name = "TheEqualizer"
targeting = "vector"
range = 5.0
vector_length = 6.0
```

**Files to modify:** `src/ai/effects/defs.rs` (AbilityTargeting), `src/ai/core/targeting.rs`

### GAP 13: Attach Mechanic (7 abilities)

**Problem:** Yuumi W attaches to an ally, becoming untargetable and moving with them. No attach system.

**Proposal:** Add `Attach` effect:
```toml
[[abilities.effects]]
type = "attach"
duration_ms = 0    # 0 = until recast
```

**Files to modify:** `src/ai/effects/types.rs` (Effect), `src/ai/core/apply_effect.rs`, `src/ai/core/tick_systems.rs` (position sync)

### GAP 14: Projectile Blocking (Yasuo Wind Wall)

**Problem:** Yasuo W and Braum E block/destroy enemy projectiles in an area. No projectile interaction system.

**Proposal:** Add `ProjectileBlock` effect that creates a zone destroying incoming projectiles:
```toml
[[abilities.effects]]
type = "projectile_block"
duration_ms = 4000

[abilities.effects.area]
shape = "line"
length = 4.0
width = 0.5
```

**Files to modify:** `src/ai/effects/types.rs` (Effect), `src/ai/core/tick_systems.rs` (projectile advancement — check for blocking zones)

### GAP 15: Grounded Effect (31 abilities mention it)

**Problem:** LoL's "Grounded" status prevents dashes, blinks, and movement abilities. Several champions apply it (Cassiopeia W, Singed W, Poppy W zone).

**Proposal:** Add `Grounded` effect/status:
```toml
[[abilities.effects]]
type = "grounded"
duration_ms = 3000
```

**Files to modify:** `src/ai/effects/types.rs` (Effect, StatusKind), `src/ai/core/apply_effect.rs` (block dashes when grounded)

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
- Multi-part abilities (Aatrox Q 3 casts) -> split into separate slots OR use recast system
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
