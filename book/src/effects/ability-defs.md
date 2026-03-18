# Ability Definitions

This chapter covers the full anatomy of an `AbilityDef` — all the fields and
systems that make abilities work at runtime.

## Runtime Slots

Abilities live on units as `AbilitySlot` wrappers:

```rust
pub struct AbilitySlot {
    pub def: AbilityDef,
    pub cooldown_remaining_ms: u32,
    pub base_def: Option<Box<AbilityDef>>,  // for morphed abilities
    pub charges: u32,
    pub charge_timer_ms: u32,
    pub is_toggled: bool,
    pub recast_remaining: u32,
    pub recast_timer_ms: u32,
}
```

The slot tracks mutable runtime state (cooldowns, charges, toggle state) while
the `def` holds the immutable definition.

### Cooldown System

Standard cooldowns: when an ability is used, `cooldown_remaining_ms` is set to
`def.cooldown_ms` and decremented each tick.

### Charge System

Some abilities (like Teemo's mushrooms) use an ammo system:
- `max_charges` — maximum stored charges
- `charge_recharge_ms` — time per charge regeneration
- The ability is usable as long as `charges > 0`

### Toggle System

Toggle abilities (like Singed's Poison Trail) flip on/off:
- `is_toggle = true` — marks this ability as a toggle
- `toggle_cost_per_sec` — resource drain while active
- No cooldown; the toggle is the "cooldown"

### Recast System

Abilities with multiple phases (like Ahri's R):
- `recast_count` — number of additional casts
- `recast_window_ms` — time to recast before cooldown starts
- `recast_effects` — different effects for each recast

### Form Swap

Abilities that change based on form (like Nidalee, Jayce):
- `form` — which form group this ability belongs to
- `swap_form` — casting this ability swaps all abilities in this form group
- `morph_into` — the alternate definition to swap to

## AI Hints

The `ai_hint` field tells the AI what category of ability this is:

- `"damage"` — primarily deals damage
- `"crowd_control"` — primarily applies CC
- `"defense"` — shields, damage reduction
- `"utility"` — dashes, buffs, vision
- `"heal"` — restores HP

The squad AI uses hints to prioritize abilities without parsing the full effect
list.

## Passive Slots

```rust
pub struct PassiveSlot {
    pub def: PassiveDef,
    pub cooldown_remaining_ms: u32,
}
```

Simpler than ability slots — passives just track their cooldown.

## Loading from Files

Hero templates in `assets/hero_templates/` define abilities in `.ability` files.
These are parsed at load time by the DSL parser (see [Ability DSL](../dsl/overview.md))
and converted to `AbilityDef`/`PassiveDef` structs, which are then attached to
units as `AbilitySlot`/`PassiveSlot`.
