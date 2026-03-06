# /ability — Generate Hero Ability TOML

Generate a hero ability or passive definition in TOML format from a natural-language description.
The output must be valid TOML that can be parsed by `parse_hero_toml()` in `src/mission/hero_templates.rs`.

## Instructions

1. Read the user's description of the ability/passive they want.
2. Generate valid TOML following the type reference below.
3. Output ONLY the TOML block(s) — either `[[abilities]]` or `[[passives]]` sections.
4. Validate against the checklist at the bottom.

---

## Type Reference

### Effect Types (`type` field, snake_case)

#### Damage & Healing
```
damage        — amount (instant), OR amount_per_tick + tick_interval_ms + duration_ms (DoT)
                Optional: scaling_stat (string), scaling_percent (f32) for HP-scaling
heal          — amount (instant), OR amount_per_tick + tick_interval_ms + duration_ms (HoT)
                Optional: scaling_stat (string), scaling_percent (f32) for HP-scaling
shield        — amount (i32), duration_ms (u32)
self_damage   — amount (i32): deal damage to caster
```

#### Crowd Control
```
stun          — duration_ms
slow          — factor (0.0–1.0), duration_ms
root          — duration_ms: prevents movement, can still attack/cast
silence       — duration_ms: prevents ability use, can move/auto-attack
fear          — duration_ms: forced flee from source, cancels casting
taunt         — duration_ms: force target to attack caster
polymorph     — duration_ms: can only move, no attack/cast
banish        — duration_ms: untargetable, immune, can't act
confuse       — duration_ms: random targeting (any team)
charm         — duration_ms: swap team temporarily
blind         — miss_chance (f32, 0.0–1.0), duration_ms
```

#### Positioning
```
knockback     — distance (f32): push target away from caster
pull          — distance (f32): pull target toward caster
dash          — to_target (bool), distance (f32, default 2.0), to_position (bool)
                to_target=true → dash toward target; to_position=true → dash toward ability position
swap          — (no fields): swap caster and target positions
```

#### Buffs & Modifiers
```
buff          — stat (string), factor (f32), duration_ms
debuff        — stat (string), factor (f32), duration_ms
reflect       — percent (f32), duration_ms: reflect % of incoming damage
lifesteal     — percent (f32), duration_ms: heal for % of damage dealt
damage_modify — factor (f32), duration_ms: multiply incoming damage (>1 = amp, <1 = reduce)
on_hit_buff   — duration_ms: next attack applies stored effects to target
immunity      — immune_to (list of strings), duration_ms: block specific effect types
stealth       — duration_ms, break_on_damage (bool), break_on_ability (bool)
```

#### Advanced Damage
```
execute       — hp_threshold_percent (f32): instant kill if target HP% <= threshold
death_mark    — duration_ms, damage_percent (f32): tracks damage, detonates on expire
detonate      — damage_multiplier (f32): consume all DoTs as instant burst
```

#### Healing & Shields
```
resurrect     — hp_percent (f32): revive dead ally at % HP
overheal_shield — duration_ms, conversion_percent (f32): overflow healing → shield
absorb_to_heal — shield_amount (i32), duration_ms, heal_percent (f32): shield that heals on expire
shield_steal  — amount (i32): transfer shield from target to caster
```

#### Status Interaction
```
dispel        — target_tags (list of strings): remove matching statuses
status_clone  — max_count (u32): copy status effects between units
status_transfer — steal_buffs (bool): move buffs/debuffs between caster and target
```

#### Stacks
```
apply_stacks  — name (string), count (u32, default 1), max_stacks (u32, default 4),
                duration_ms (u32): apply/increment named stacks on target
```

#### Complex
```
duel          — duration_ms
summon        — template (string), count (u32), hp_percent (f32, default 100.0)
link          — duration_ms, share_percent (f32): share damage with linked unit
redirect      — duration_ms, charges (u32): redirect attacks to protector
leash         — max_range (f32), duration_ms: restrict movement range
rewind        — lookback_ms (u32): restore position/HP from history
cooldown_modify — amount_ms (i32), ability_name (string, optional): adjust ability cooldowns (+/-)
                  If ability_name is set, only affects that specific ability
```

#### Scaling Stats (for `scaling_stat` on damage/heal)
```
"caster_current_hp", "caster_missing_hp", "caster_max_hp"
"target_current_hp", "target_missing_hp", "target_max_hp"
```

### Buff/Debuff Stat Names
```
"damage"             — multiply attack damage by (1.0 + factor)
"move_speed"         — multiply move speed by (1.0 + factor)
"attack_speed"       — multiply attack cooldown by 1.0 / (1.0 + factor)
"cooldown_reduction" — multiply ability cooldowns by 1.0 / (1.0 + factor)
"heal_power"         — multiply healing output by (1.0 + factor)
"damage_output"      — multiply outgoing damage by factor (0.0 = pacify)
```

### Area Shapes (`[*.area]` section)

```
shape = "single_target"     — affects only the target
shape = "circle"            — radius (f32)
shape = "cone"              — radius (f32), angle_deg (f32)
shape = "line"              — length (f32), width (f32)
shape = "ring"              — inner_radius (f32), outer_radius (f32)
shape = "self"              — caster only
shape = "spread"            — radius (f32), max_targets (u32, 0=unlimited)
                              like circle but skips targets already affected
```

### Delivery Methods (`[abilities.delivery]` section)

```
method = "instant"          — default, no delivery section needed
method = "projectile"       — speed (f32), pierce (bool), width (f32)
                              on_hit = [{effect}], on_arrival = [{effect}]
                              Skillshots: use targeting = "direction" to fire in a direction
                              Projectile tracks max_travel_distance (= ability range for skillshots)
method = "channel"          — duration_ms (u32), tick_interval_ms (u32)
                              Locks caster in place, applies effects each tick interval
                              Interrupted by stun/silence/fear/polymorph/banish/death
method = "zone"             — duration_ms (u32), tick_interval_ms (u32)
                              Persistent ground zone, applies effects to units inside at each tick
method = "tether"           — max_range (f32), tick_interval_ms (u32, default 0),
                              on_complete = [{effect}]
                              Maintained link between caster and target; breaks if range exceeded
                              tick effects applied at interval; on_complete fires when tether expires naturally
method = "trap"             — duration_ms (u32), trigger_radius (f32), arm_time_ms (u32, default 0)
                              Invisible ground zone that triggers once when enemy enters radius
                              arm_time_ms = delay before trap becomes active
method = "chain"            — bounces (u32), bounce_range (f32), falloff (f32)
                              on_hit = [{effect}]
                              Falloff is proportional: all values reduce by (1-falloff)^bounce
```

### Targeting (`targeting` field)

```
"target_enemy"    — single enemy unit
"target_ally"     — single allied unit
"self_cast"       — caster only, no target needed
"self_aoe"        — AoE centered on caster
"target_position" — ground-targeted (use with area)
"direction"       — fire toward a position for full range (skillshots)
                    AI resolves this the same as target_position
```

### Conditions (`[*.condition]` section)

```
type = "always"               — (default, omit section)
type = "target_hp_below"      — percent (f32)
type = "target_hp_above"      — percent (f32)
type = "target_is_stunned"
type = "target_is_slowed"
type = "target_is_rooted"
type = "target_is_silenced"
type = "target_is_feared"
type = "target_is_taunted"
type = "target_is_banished"
type = "target_is_stealthed"
type = "target_is_charmed"
type = "target_is_polymorphed"
type = "caster_hp_below"      — percent (f32)
type = "caster_hp_above"      — percent (f32)
type = "hit_count_above"      — count (u32)
type = "target_has_tag"       — tag (string)
type = "caster_has_status"    — status (string)
type = "target_has_status"    — status (string)
type = "target_debuff_count"  — min_count (u32)
type = "caster_buff_count"    — min_count (u32)
type = "ally_count_below"     — count (u32)
type = "enemy_count_below"    — count (u32)
type = "target_stack_count"   — name (string), min_count (u32): check named stacks on target
```

### Triggers (passives only, `[passives.trigger]` section)

```
type = "on_damage_dealt"
type = "on_damage_taken"
type = "on_kill"
type = "on_ally_damaged"     — range (f32)
type = "on_death"
type = "on_ability_used"
type = "on_hp_below"         — percent (f32)
type = "on_hp_above"         — percent (f32)
type = "on_shield_broken"
type = "on_stun_expire"
type = "periodic"            — interval_ms (u32)
type = "on_heal_received"
type = "on_status_applied"
type = "on_status_expired"
type = "on_resurrect"
type = "on_dodge"
type = "on_reflect"
type = "on_ally_killed"      — range (f32)
type = "on_auto_attack"
type = "on_stack_reached"    — name (string), count (u32): fires when named stacks reach count
```

### Stacking (`stacking` field on effects)

```
"refresh"    — reset duration (default)
"extend"     — add to remaining duration
"strongest"  — keep highest value
"stack"      — accumulate
```

### Tags (`[*.tags]` section on effects)

Tags are `HashMap<String, f32>`. Higher value = harder to resist.
Common tags: `PHYSICAL`, `MAGIC`, `FIRE`, `ICE`, `HOLY`, `DARK`, `POISON`,
`CROWD_CONTROL`, `SLOW`, `KNOCKBACK`, `SILENCE`, `BLEED`, `ANTI_HEAL`, `FEAR`.
Range: 30–100. New tags can be invented freely.

### AI Hint (`ai_hint` field on abilities)

```
"damage"         — offensive ability
"heal"           — healing ability
"crowd_control"  — CC ability
"defense"        — defensive/shield ability
"utility"        — other
```

---

## Example Abilities

### Whirlwind (Warrior — self AoE damage)
```toml
[[abilities]]
name = "Whirlwind"
targeting = "self_aoe"
range = 0.0
cooldown_ms = 8000
cast_time_ms = 300
ai_hint = "damage"

[[abilities.effects]]
type = "damage"
amount = 40

[abilities.effects.area]
shape = "circle"
radius = 2.5

[abilities.effects.tags]
PHYSICAL = 50.0

[[abilities.effects]]
type = "damage"
amount = 10

[abilities.effects.condition]
type = "hit_count_above"
count = 2

[abilities.effects.area]
shape = "circle"
radius = 2.5

[abilities.effects.tags]
PHYSICAL = 50.0
```

### Fireball (Mage — projectile with splash)
```toml
[[abilities]]
name = "Fireball"
targeting = "target_enemy"
range = 5.0
cooldown_ms = 5000
cast_time_ms = 300
ai_hint = "damage"

[abilities.delivery]
method = "projectile"
speed = 8.0
pierce = false
width = 0.3

[[abilities.delivery.on_hit]]
type = "damage"
amount = 55

[abilities.delivery.on_hit.tags]
FIRE = 60.0

[[abilities.delivery.on_arrival]]
type = "damage"
amount = 15

[abilities.delivery.on_arrival.area]
shape = "circle"
radius = 2.0
```

### Chain Lightning (Elementalist — chain delivery)
```toml
[[abilities]]
name = "ChainLightning"
targeting = "target_enemy"
range = 5.0
cooldown_ms = 8000
cast_time_ms = 300
ai_hint = "damage"

[[abilities.effects]]
type = "damage"
amount = 30

[abilities.effects.tags]
MAGIC = 55.0

[abilities.delivery]
method = "chain"
bounces = 3
bounce_range = 3.0
falloff = 0.15

[[abilities.delivery.on_hit]]
type = "damage"
amount = 30

[abilities.delivery.on_hit.tags]
MAGIC = 55.0
```

### Holy Light (Cleric — targeted heal with bonus)
```toml
[[abilities]]
name = "Holy Light"
targeting = "target_ally"
range = 4.0
cooldown_ms = 2500
cast_time_ms = 300
ai_hint = "heal"

[[abilities.effects]]
type = "heal"
amount = 35

[abilities.effects.tags]
HOLY = 30.0

[[abilities.effects]]
type = "heal"
amount = 15

[abilities.effects.tags]
HOLY = 30.0

[abilities.effects.condition]
type = "target_hp_below"
percent = 30.0
```

### Backstab (Rogue — dash + conditional bonus)
```toml
[[abilities]]
name = "Backstab"
targeting = "target_enemy"
range = 4.0
cooldown_ms = 3000
cast_time_ms = 0
ai_hint = "damage"

[[abilities.effects]]
type = "dash"
to_target = true

[[abilities.effects]]
type = "damage"
amount = 50

[abilities.effects.tags]
PHYSICAL = 40.0

[[abilities.effects]]
type = "damage"
amount = 25

[abilities.effects.condition]
type = "target_is_stunned"
```

### Blood Pact (Blood Mage — self damage + buff)
```toml
[[abilities]]
name = "BloodPact"
targeting = "self_cast"
range = 0.0
cooldown_ms = 15000
cast_time_ms = 300
ai_hint = "utility"

[[abilities.effects]]
type = "self_damage"
amount = 30

[[abilities.effects]]
type = "lifesteal"
percent = 40.0
duration_ms = 6000

[[abilities.effects]]
type = "buff"
stat = "damage"
factor = 0.3
duration_ms = 6000
```

### Terrify (Warlock — AoE fear)
```toml
[[abilities]]
name = "Terrify"
targeting = "self_aoe"
range = 0.0
cooldown_ms = 12000
cast_time_ms = 400
ai_hint = "crowd_control"

[[abilities.effects]]
type = "fear"
duration_ms = 2500

[abilities.effects.area]
shape = "circle"
radius = 3.0

[abilities.effects.tags]
FEAR = 55.0
CROWD_CONTROL = 50.0
```

### Thorns (Warden — reflect passive)
```toml
[[passives]]
name = "Thorns"
cooldown_ms = 0
range = 0.0

[passives.trigger]
type = "on_damage_taken"

[[passives.effects]]
type = "reflect"
percent = 25.0
duration_ms = 3000
```

---

## Example Passives

### Iron Skin (Warrior — reactive shield)
```toml
[[passives]]
name = "Iron Skin"
cooldown_ms = 5000
range = 0.0

[passives.trigger]
type = "on_damage_taken"

[[passives.effects]]
type = "shield"
amount = 20
duration_ms = 3000
```

### Retribution (Paladin — ally-reactive damage)
```toml
[[passives]]
name = "Retribution"
cooldown_ms = 4000
range = 3.0

[passives.trigger]
type = "on_ally_damaged"
range = 3.0

[[passives.effects]]
type = "damage"
amount = 12
```

### Arcane Shield (Mage — HP threshold trigger)
```toml
[[passives]]
name = "Arcane Shield"
cooldown_ms = 30000
range = 0.0

[passives.trigger]
type = "on_hp_below"
percent = 50.0

[[passives.effects]]
type = "shield"
amount = 40
duration_ms = 4000
```

### Vengeance (Paladin — on-ally-killed buff)
```toml
[[passives]]
name = "Vengeance"
cooldown_ms = 10000
range = 5.0

[passives.trigger]
type = "on_ally_killed"
range = 5.0

[[passives.effects]]
type = "buff"
stat = "damage"
factor = 0.4
duration_ms = 6000

[[passives.effects]]
type = "buff"
stat = "attack_speed"
factor = 0.3
duration_ms = 6000
```

---

## Validation Checklist

Before outputting, verify:

- [ ] All `type` values are valid snake_case enum variants from the list above
- [ ] All durations use `_ms` suffix and are in milliseconds (e.g., 2000 for 2 seconds)
- [ ] DoT/HoT effects have all three fields: `amount_per_tick`, `tick_interval_ms`, `duration_ms`
- [ ] Instant damage/heal uses `amount` field only (no tick fields)
- [ ] `targeting` matches effect intent (heal → target_ally, damage → target_enemy or self_aoe)
- [ ] `ai_hint` accurately reflects the ability's primary purpose
- [ ] Projectile abilities put damage in `on_hit`/`on_arrival`, NOT in `abilities.effects`
- [ ] Chain abilities put effects in `delivery.on_hit`, with optional `abilities.effects` for primary target
- [ ] Tags use UPPER_CASE names with float values (e.g., `FIRE = 60.0`)
- [ ] `cooldown_ms` is reasonable (2000–30000 for abilities, 3000–60000 for passives)
- [ ] `range` > 0 for targeted abilities, 0.0 for self-cast/self_aoe
- [ ] Passive `range` field = 0.0 unless trigger is `on_ally_damaged`, `on_ally_killed`, or `periodic`
- [ ] Buff/debuff `stat` is one of: damage, move_speed, attack_speed, cooldown_reduction, heal_power, damage_output
- [ ] HP scaling uses valid `scaling_stat` values (caster_current_hp, target_missing_hp, etc.)

### Resource Cost (`resource_cost` field on abilities)

```
resource_cost = 30    — deducted from unit's resource pool on cast (default 0)
                        Ability blocked if unit.resource < resource_cost
```

### Ability Morphing (`morph_into` section on abilities)

```
[abilities.morph_into]
name = "..."          — after using the ability, it transforms into this ability
                        Use morph_duration_ms on the outer ability to set a timed revert
                        morph_duration_ms = 0 means morph persists until the morphed ability is used
                        Using the morphed ability reverts back to the original
```

### Hero Stats — Resource Fields

```
[stats]
resource = 100            — starting resource (default 0)
max_resource = 100        — resource cap (default 0, 0 = no resource system)
resource_regen_per_sec = 5.0  — passive regen rate (default 0.0)
```

## Output Format

Output the TOML block(s) in a code fence. If generating a complete hero, include all sections:
```toml
[hero]
name = "..."

[stats]
hp = ...
move_speed = ...
resource = ...           # optional, default 0
max_resource = ...       # optional, default 0
resource_regen_per_sec = ...  # optional, default 0.0

[stats.tags]
# resistance tags

[attack]
damage = ...
range = ...

# Then [[abilities]] and [[passives]] blocks
```

$ARGUMENTS
