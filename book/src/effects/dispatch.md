# Effect Dispatch & Resolution

When an ability fires, its effects must be applied to the simulation state. This
chapter covers how effects flow from definition to state mutation.

## The Resolution Pipeline

```
UnitIntent::UseAbility
       │
       ▼
   resolve_cast()          # validate targeting, start cast timer
       │
       ▼
   (cast_time elapses)     # unit is in CastState for N ticks
       │
       ▼
   complete_cast()         # cast finishes, effects fire
       │
       ▼
   apply_delivery()        # if delivery exists, create projectile/zone/etc
       │                   # otherwise, apply effects immediately
       ▼
   apply_effects()         # iterate ConditionalEffect list
       │
       ▼
   apply_effect()          # pattern match on each Effect variant
       │
       ▼
   SimState mutation       # HP changes, status applied, etc.
   + SimEvent emission     # events logged
```

## Conditional Effects

Effects are wrapped in `ConditionalEffect`:

```rust
pub struct ConditionalEffect {
    pub effect: Effect,
    pub condition: Option<Condition>,
    pub area: Option<Area>,
    pub tags: Tags,
}
```

Before applying an effect, its condition (if any) is checked against the current
state. This is how abilities have conditional bonuses:

```
damage 60 [PHYSICAL: 50]
damage 30 when target_is_stunned [PHYSICAL: 50]
```

The second line only fires if the target is stunned.

## Effect Application

`apply_effect.rs` and `apply_effect_ext.rs` contain the main dispatch:

```rust
fn apply_effect(
    state: &mut SimState,
    source_idx: usize,
    target_idx: usize,
    effect: &Effect,
    tags: &Tags,
    events: &mut Vec<SimEvent>,
) {
    match effect {
        Effect::Damage(amount) => { /* reduce HP, emit DamageDealt */ }
        Effect::Heal(amount) => { /* increase HP, emit HealApplied */ }
        Effect::Shield { amount, duration_ms } => { /* add shield */ }
        Effect::Stun(duration_ms) => { /* apply stun status */ }
        Effect::Dash(distance) => { /* move caster */ }
        Effect::Summon(template) => { /* create new unit */ }
        // ... 30+ variants
    }
}
```

The pattern matching is exhaustive — the compiler ensures every effect type is
handled. Adding a new effect variant requires adding a match arm, which prevents
silent omissions.

## Area Resolution

When an effect has an `Area`, the engine finds all valid targets within the shape:

```rust
fn targets_in_area(
    state: &SimState,
    origin: SimVec2,
    area: &Area,
    source_team: Team,
    friendly: bool,
) -> Vec<usize>
```

The `friendly` flag determines whether the area selects allies or enemies. Most
damage effects select enemies; most healing effects select allies.

## Delivery Resolution

Deliveries add a time delay or movement phase between cast completion and effect
application:

### Projectile
Creates a `Projectile` entry in `SimState.projectiles`. Each tick, projectiles
advance by `speed * dt`. On collision (distance < `width`), `on_hit` effects
fire. On reaching the target position, `on_arrival` effects fire.

### Zone
Creates a `Zone` entry in `SimState.zones`. Each tick interval, `on_tick`
effects are applied to all units inside the zone area.

### Chain
Immediately applies `on_hit` effects to the primary target, then bounces to
the nearest valid target within `range`, reducing damage by `falloff` per bounce.

### Tether
Creates a `Tether` linking source and target. Effects apply each tick while the
tether holds. If the target moves beyond `range`, the tether breaks.

## Damage Calculation

The damage formula in `damage.rs`:

```
effective_damage = base_damage
    * (1.0 - armor_reduction)     // armor / (100 + armor)
    * (1.0 - resistance_reduction) // from matching resistance tags
    * (1.0 + damage_modify)       // damage amplification debuffs
    * (1.0 - cover_bonus)         // terrain cover
```

Damage is applied to shield first, then HP:

```
if target.shield_hp > 0 {
    let absorbed = min(effective_damage, target.shield_hp);
    target.shield_hp -= absorbed;
    remaining = effective_damage - absorbed;
    emit ShieldAbsorbed event
}
target.hp -= remaining;
```
