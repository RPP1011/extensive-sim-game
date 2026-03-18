# The `step()` Function

The `step()` function in `src/ai/core/simulation.rs` is the most important function
in the codebase. Every combat tick passes through it.

## Signature

```rust
#[requires(state.units.iter().all(|u|
    u.position.x.is_finite() && u.position.y.is_finite()))]
#[requires(state.units.iter().all(|u| u.hp <= u.max_hp))]
#[ensures(ret.0.tick == old(state.tick) + 1)]
#[ensures(ret.0.units.len() >= old(state.units.len()))]
pub fn step(
    mut state: SimState,
    intents: &[UnitIntent],
    dt_ms: u32,
) -> (SimState, Vec<SimEvent>)
```

Note the contract annotations:
- **Preconditions:** all unit positions must be finite; HP cannot exceed max HP
- **Postconditions:** tick always advances by exactly 1; unit count never decreases
  (summons can add units, but nothing removes them from the array — dead units
  remain with `hp <= 0`)

## Taking State by Value

`SimState` is taken **by value**, not by mutable reference. This is a deliberate
design choice:

```rust
// The caller gives up ownership:
let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
// `state` is consumed — only `new_state` exists now
```

This prevents accidental state aliasing and makes it clear that `step()` produces
a new state. The compiler enforces that no one holds a stale reference to the
previous state.

## Unit Processing Order

A critical detail: unit processing order is **shuffled every tick** using the
simulation's seeded RNG:

```rust
let mut unit_order: Vec<usize> = (0..state.units.len()).collect();
for i in (1..unit_order.len()).rev() {
    let j = (next_rand_u32(&mut state) as usize) % (i + 1);
    unit_order.swap(i, j);
}
```

This Fisher-Yates shuffle prevents first-mover advantage — without it, the unit
at index 0 would always act first, creating a systematic bias in training data.
Because the shuffle uses the seeded RNG, it is still deterministic.

## Intent Resolution

For each unit (in shuffled order), the function checks:

1. Is the unit alive? (skip dead units)
2. Is this a directed summon? (skip — directed summons act when their owner acts)
3. Decrement cooldown timers
4. Process crowd control (stun, root, etc.)
5. Handle casting state (wind-up animations)
6. Resolve the unit's intent action:
   - `Hold` — do nothing
   - `MoveTo { position }` — move toward a position
   - `Attack { target_id }` — basic attack
   - `CastHeal { target_id }` — use heal ability
   - `CastControl { target_id }` — use CC ability
   - `UseAbility { ability_index, target }` — use a hero ability
   - `Dash` / `Retreat` / `Skulk` — movement abilities

## The Fixed Tick Constant

```rust
pub const FIXED_TICK_MS: u32 = 100;
```

All timing in the simulation is based on this 100ms tick. A 5-second cooldown is
50 ticks. A 300ms cast time is 3 ticks. This quantization simplifies the simulation
and ensures perfect determinism (no floating-point time accumulation errors).
