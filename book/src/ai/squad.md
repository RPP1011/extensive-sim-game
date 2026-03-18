# Squad AI

The squad AI is the first layer in the decision pipeline. It operates at the
**team level**, making strategic decisions about formation, focus targets, and
overall behavior based on a personality profile.

## Module: `src/ai/squad/`

```
squad/
├── personality.rs  # Personality profiles
├── state.rs        # SquadAiState, FormationMode, SquadBlackboard
├── intents.rs      # Intent generation
├── forces.rs       # Force-based steering
├── combat/
│   ├── abilities.rs  # Ability evaluation
│   ├── healer.rs     # Healer-specific targeting
│   └── targeting.rs  # Focus target selection
├── replay.rs       # Replay integration
└── tests.rs
```

## Personality Profiles

Each squad has a `Personality` that biases its behavior:

```rust
pub struct Personality {
    pub aggression: f32,    // 0.0 (defensive) to 1.0 (all-in)
    pub focus_fire: f32,    // tendency to concentrate on one target
    pub protect_healer: f32, // priority on defending support units
    pub ability_usage: f32,  // eagerness to use abilities vs basic attacks
}
```

Personality values shift the weights used in intent generation. An aggressive
squad focuses on killing priority targets; a defensive squad protects its healer
and plays safe.

## Formation Modes

```rust
pub enum FormationMode {
    Advance,    // move toward enemies
    Hold,       // maintain position
    Retreat,    // fall back
    Flank,      // circle to the side
    Protect,    // collapse on the healer
}
```

The squad AI switches formation modes based on combat state — team HP ratios,
number of kills, enemy positioning, etc.

## SquadBlackboard

The `SquadBlackboard` is a shared state visible to all units on a team:

```rust
pub struct SquadBlackboard {
    pub focus_target: Option<u32>,    // team-wide focus fire target
    pub formation_mode: FormationMode,
    pub threat_level: f32,
    // ...
}
```

Individual AI layers (GOAP, ability evaluator) can read the blackboard to align
with team strategy. For example, GOAP units override their target to the
blackboard's focus target when set.

## Intent Generation

The main entry point:

```rust
pub fn generate_intents(
    state: &SimState,
    squad_state: &mut SquadAiState,
) -> Vec<UnitIntent>
```

For each unit, this function:

1. Updates the blackboard (focus target, formation mode)
2. Evaluates the unit's role (tank, DPS, healer)
3. Generates an appropriate intent based on role + personality:
   - **Tanks** prioritize engaging the nearest enemy or peeling for allies
   - **DPS** prioritize the focus target or lowest-HP enemy
   - **Healers** prioritize the lowest-HP ally

## Ability Evaluation

`combat/abilities.rs` contains the hero ability evaluation logic:

```rust
pub fn evaluate_hero_ability(
    state: &SimState,
    unit_id: u32,
    target_id: u32,
    formation: FormationMode,
    ctx: &TickContext,
) -> Option<IntentAction>
```

This function checks each of the unit's abilities:
- Is it off cooldown?
- Is the target in range?
- Does the AI hint match the current tactical need?
- Is this a better option than a basic attack?

If an ability is worth using, it returns `UseAbility` instead of `Attack`.

## Deconfliction

After all intents are generated, a deconfliction pass prevents wasteful
duplication:

- If two healers target the same ally, the second switches to attacking
- If two units CC the same enemy, the second switches to attacking

This is handled by `deconflict_intents()` in the GOAP module, which runs
as a post-processing step.
