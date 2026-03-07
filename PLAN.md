# Ability DSL Design Plan

## Overview

Replace the TOML format used for ability/passive definitions with a brace-based DSL
that parses directly into `AbilityDef` / `PassiveDef` Rust structs. Hero-level
definitions (stats, attack, hero name) remain in TOML — only the `[[abilities]]` and
`[[passives]]` arrays move to the new `.ability` format.

---

## 1. DSL Syntax Specification

### 1.1 File Structure

An `.ability` file contains one or more `ability` and/or `passive` blocks.
Hero TOML files gain an optional `abilities_file = "mage.ability"` field; when present,
the loader reads abilities/passives from that file instead of inline TOML arrays.

### 1.2 Ability Block

```
ability Fireball {
    target: enemy
    range: 5.0
    cooldown: 5s
    cast: 300ms
    hint: damage

    deliver projectile { speed: 8.0, width: 0.3 } {
        on_hit {
            damage 55 [FIRE: 60]
        }
        on_arrival {
            damage 15 in circle(2.0)
        }
    }
}
```

### 1.3 Passive Block

```
passive ArcaneShield {
    trigger: on_hp_below(50%)
    cooldown: 30s

    shield 40 for 4s
}
```

### 1.4 Syntax Rules

#### Header Properties (key-value pairs)
```
target: <targeting>        # enemy | ally | self | self_aoe | ground | direction | vector | global
range: <float>
cooldown: <duration>       # 5s | 5000ms | 5000 (defaults to ms)
cast: <duration>
hint: <ai_hint>            # damage | crowd_control | defense | utility | heal
cost: <int>                # resource_cost
zone_tag: <string>         # for zone-reaction combos
```

#### Duration Shorthand
- `5s` → 5000ms
- `300ms` → 300ms
- `5000` → 5000ms (bare number = ms)

#### Advanced Ability Properties
```
charges: 3, recharge: 10s          # ammo system
toggle, toggle_cost: 5.0           # toggle ability
recast: 2, recast_window: 3s      # recast system
unstoppable                        # immune to CC during cast
form: "bear"                       # form group tag
swap_form: "bear"                  # triggers form swap
```

#### Effect Syntax

Effects are written as compact one-liners with optional modifiers:

```
# Basic effects
damage 55                          # simple damage
damage 55 [FIRE: 60, MAGIC: 40]   # with tags
heal 30                            # heal
shield 40 for 4s                   # shield with duration
stun 2s                            # CC with duration
slow 0.3 for 1.5s                  # slow factor + duration
root 2s
silence 1.5s
fear 2s
taunt 2s
charm 1.5s
polymorph 3s
banish 2s
confuse 1s
suppress 2s
grounded 3s

# Movement
dash 3.0                           # dash distance
dash to_target 2.0                 # dash toward target
dash to_position                   # dash to cursor
blink 5.0                          # instant teleport
knockback 3.0
pull 2.0
swap                               # swap positions

# Buffs/Debuffs
buff damage_output 0.15 for 5s
debuff move_speed 0.3 for 3s
damage_modify 1.5 for 4s
reflect 0.3 for 3s
lifesteal 0.2 for 5s
blind 0.5 for 2s

# Advanced
summon "skeleton" x3               # summon 3 skeletons
summon clone                       # clone of caster
stealth 5s break_on_damage
leash 4.0 for 3s
link 0.5 for 4s                    # share_percent for duration
redirect 3s charges 2
rewind 3s                          # lookback
cooldown_modify -2s                # reduce cooldowns
cooldown_modify -1s "Fireball"     # specific ability
apply_stacks "poison" 2 max 5 for 4s
execute 15%                        # hp threshold
self_damage 50
dispel [CROWD_CONTROL]
immunity [stun, root] for 3s
death_mark 4s 0.5                  # duration, damage_percent
resurrect 50%
overheal_shield 5s
absorb_to_heal 30 for 4s heal 0.5
shield_steal 20
status_clone max 3
detonate 1.5                       # damage_multiplier
status_transfer steal              # steal_buffs
on_hit_buff 5s { damage 10 }       # on-hit effects
obstacle 2.0 x 1.0                 # width x height
projectile_block 3s
attach 4s
evolve_ability 2                   # ability_index
command_summons speed 8.0
```

#### Area Modifiers

Appended to any effect with `in`:

```
damage 50 in circle(3.0)
damage 30 in cone(4.0, 60)         # radius, angle_deg
damage 20 in line(6.0, 1.0)        # length, width
heal 15 in ring(2.0, 5.0)          # inner, outer
damage 25 in spread(3.0, 4)        # radius, max_targets
```

#### Delivery Block

```
deliver projectile { speed: 8.0, pierce, width: 0.3 } {
    on_hit {
        damage 55 [FIRE: 60]
    }
    on_arrival {
        damage 15 in circle(2.0)
    }
}

deliver chain { bounces: 3, range: 3.0, falloff: 0.8 } {
    on_hit {
        damage 35 [MAGIC: 50]
    }
}

deliver zone { duration: 4s, tick: 1s } {
    on_hit {
        damage 15 in circle(3.0) [ICE: 50]
        slow 0.3 for 1.5s in circle(3.0)
    }
}

deliver channel { duration: 3s, tick: 500ms }

deliver tether { max_range: 5.0, tick: 500ms } {
    on_complete {
        stun 1.5s
    }
}

deliver trap { duration: 60s, trigger_radius: 1.5, arm_time: 1s }
```

#### Conditions (the `when` keyword)

```
# Simple condition
damage 100 when target_hp_below(30%) [FIRE: 70]

# Else branch
damage 50 when target_is_stunned else damage 25

# Compound conditions
stun 2s when and(target_hp_below(50%), caster_hp_above(30%))
damage 80 when or(target_is_stunned, target_is_rooted)
heal 30 when not(caster_hp_above(80%))

# Probability
damage 30 chance 0.25              # 25% chance
```

#### Stacking
```
buff damage_output 0.08 for 4s stacking stack
buff armor 0.1 for 3s stacking refresh    # default
buff speed 0.05 for 5s stacking extend
```

#### Scaling
```
damage 50 + 8% target_max_hp
damage 0 + 15% caster_attack_damage + 5% caster_max_hp
heal 20 + 10% caster_missing_hp
damage 0 + 20% target_stacks("poison") consume cap 100
```

#### Morph
```
ability StanceSwitch {
    target: self
    cooldown: 1s
    hint: utility
    swap_form: "bear"

    morph into {
        # full ability def for morphed version
        target: self
        cooldown: 1s
        hint: utility
        swap_form: "human"
    } for 0ms   # 0 = permanent until recast
}
```

#### Recast Effects
```
ability Ahri_R {
    target: self
    cooldown: 10s
    cast: 0ms
    hint: utility
    recast: 2, recast_window: 10s

    dash 5.0

    recast 1 {
        dash 5.0
    }
    recast 2 {
        dash 5.0
        damage 40 in circle(2.0)
    }
}
```

#### Passive Triggers
```
passive Name {
    trigger: on_damage_dealt
    trigger: on_damage_taken
    trigger: on_kill
    trigger: on_ally_damaged(range: 5.0)
    trigger: on_death
    trigger: on_ability_used
    trigger: on_hp_below(50%)
    trigger: on_hp_above(80%)
    trigger: on_shield_broken
    trigger: on_stun_expire
    trigger: periodic(3s)
    trigger: on_heal_received
    trigger: on_status_applied
    trigger: on_status_expired
    trigger: on_resurrect
    trigger: on_dodge
    trigger: on_reflect
    trigger: on_ally_killed(range: 5.0)
    trigger: on_auto_attack
    trigger: on_stack_reached("poison", 4)
    cooldown: 8s
    range: 5.0

    # effects here
}
```

### 1.5 Comments
```
// single-line comment
# also single-line comment (TOML compatibility)
```

---

## 2. Full Examples

### 2.1 Mage Abilities (currently 231 lines of TOML → ~90 lines of DSL)

```
// Mage abilities

ability Fireball {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    deliver projectile { speed: 8.0, width: 0.3 } {
        on_hit {
            damage 55 [FIRE: 60]
        }
        on_arrival {
            damage 15 in circle(2.0)
        }
    }
}

ability FrostNova {
    target: self_aoe
    cooldown: 10s, cast: 300ms
    hint: crowd_control

    damage 20 in circle(3.0)
    stun 2s in circle(3.0) [CROWD_CONTROL: 80, ICE: 60]
}

ability ArcaneMissiles {
    target: enemy, range: 5.0
    cooldown: 4s, cast: 200ms
    hint: damage

    deliver chain { bounces: 3, range: 3.0, falloff: 0.8 } {
        on_hit {
            damage 35 [MAGIC: 50]
        }
    }
}

ability Blizzard {
    target: ground, range: 6.0
    cooldown: 12s, cast: 400ms
    hint: damage

    deliver zone { duration: 4s, tick: 1s } {
        on_hit {
            damage 15 in circle(3.0) [ICE: 50]
            slow 0.3 for 1.5s in circle(3.0)
        }
    }
}

ability Meteor {
    target: ground, range: 6.0
    cooldown: 18s, cast: 500ms
    hint: damage

    damage 80 in circle(2.5) [FIRE: 70]
}

ability Blink {
    target: self
    cooldown: 10s, cast: 0ms
    hint: utility

    dash 3.0
    damage 15 in circle(2.0) [MAGIC: 40]
}

ability Polymorph {
    target: enemy, range: 5.0
    cooldown: 15s, cast: 300ms
    hint: crowd_control

    polymorph 3s [CROWD_CONTROL: 80]
}

ability ManaShield {
    target: self
    cooldown: 14s, cast: 200ms
    hint: defense

    shield 50 for 5s
}

passive ArcaneShield {
    trigger: on_hp_below(50%)
    cooldown: 30s

    shield 40 for 4s
}

passive ArcaneMastery {
    trigger: on_ability_used
    cooldown: 8s

    buff cooldown_reduction 0.15 for 3s
}
```

### 2.2 Complex Example with Conditions and Scaling

```
ability VampiricStrike {
    target: enemy, range: 2.0
    cooldown: 8s, cast: 200ms
    hint: damage

    damage 40 + 10% caster_attack_damage
    lifesteal 0.3 for 3s when caster_hp_below(50%)
    heal 20 when caster_hp_below(30%) else damage 15 in circle(2.0)
}

passive VengeanceAura {
    trigger: on_ally_killed(range: 6.0)
    cooldown: 15s

    buff damage_output 0.25 for 5s
    buff move_speed 0.15 for 5s
    damage 30 in circle(4.0) [MAGIC: 50]
}
```

---

## 3. Implementation Plan

### 3.1 Parser Architecture

**Crate**: Use `winnow` (successor to `nom`, better error messages, maintained)
as the parser combinator library. It produces excellent error messages and is
well-suited for custom DSL parsing.

**Pipeline**:
```
.ability file (text)
    → winnow parser
    → AST (intermediate representation)
    → lowering pass
    → Vec<AbilityDef> + Vec<PassiveDef>
```

### 3.2 New Files

```
src/ai/effects/
    dsl/
        mod.rs          — public API: parse_abilities(input) -> Result<(Vec<AbilityDef>, Vec<PassiveDef>)>
        ast.rs          — AST types (AbilityNode, PassiveNode, EffectNode, etc.)
        parser.rs       — winnow parser combinators
        lower.rs        — AST → AbilityDef/PassiveDef conversion
        error.rs        — user-friendly parse error formatting
        tests.rs        — unit tests
```

### 3.3 AST Types (ast.rs)

```rust
pub struct AbilityFile {
    pub abilities: Vec<AbilityNode>,
    pub passives: Vec<PassiveNode>,
}

pub struct AbilityNode {
    pub name: String,
    pub props: Vec<Property>,          // key-value header properties
    pub effects: Vec<EffectNode>,      // direct effects
    pub delivery: Option<DeliveryNode>,
    pub morph: Option<MorphNode>,
    pub recasts: Vec<RecastNode>,
}

pub struct PassiveNode {
    pub name: String,
    pub trigger: TriggerNode,
    pub cooldown: Option<Duration>,
    pub range: Option<f32>,
    pub effects: Vec<EffectNode>,
}

pub struct EffectNode {
    pub effect_type: String,           // "damage", "heal", "stun", etc.
    pub args: Vec<Arg>,                // positional args (amount, duration, etc.)
    pub area: Option<AreaNode>,        // in circle(3.0)
    pub tags: Vec<(String, f32)>,      // [FIRE: 60]
    pub condition: Option<ConditionNode>,
    pub else_effects: Vec<EffectNode>,
    pub stacking: Option<String>,
    pub chance: Option<f32>,
    pub scaling: Vec<ScalingNode>,     // + 10% stat
    pub duration: Option<Duration>,    // for Xs (shield, buff, etc.)
    pub children: Vec<EffectNode>,     // nested effects (on_hit_buff)
}

pub struct DeliveryNode {
    pub method: String,
    pub params: Vec<(String, Arg)>,
    pub on_hit: Vec<EffectNode>,
    pub on_arrival: Vec<EffectNode>,
    pub on_complete: Vec<EffectNode>,
}
```

### 3.4 Integration with Hero TOML

Modify `HeroToml` to support an optional `abilities_file` field:

```rust
// In defs.rs
pub struct HeroToml {
    pub hero: HeroMeta,
    pub stats: HeroStats,
    pub attack: Option<AttackStats>,
    pub abilities: Vec<AbilityDef>,        // from inline TOML OR .ability file
    pub passives: Vec<PassiveDef>,         // from inline TOML OR .ability file
    #[serde(default)]
    pub abilities_file: Option<String>,    // NEW: path to .ability file
}
```

In `hero_templates.rs`, after parsing the TOML:
```rust
if let Some(file) = &toml.abilities_file {
    let dsl_text = std::fs::read_to_string(format!("assets/hero_templates/{file}"))?;
    let (abilities, passives) = dsl::parse_abilities(&dsl_text)?;
    toml.abilities = abilities;
    toml.passives = passives;
}
```

### 3.5 Implementation Steps

| Step | What | Details |
|------|------|---------|
| 1 | **Scaffold** | Create `src/ai/effects/dsl/` module with `mod.rs`, `ast.rs`, `parser.rs`, `lower.rs`, `error.rs` |
| 2 | **AST types** | Define all AST node types in `ast.rs` |
| 3 | **Core parser** | Implement winnow parsers: identifiers, numbers, durations, strings, comments, whitespace |
| 4 | **Effect parser** | Parse effect one-liners: `damage 55 in circle(3.0) [FIRE: 60] when cond` |
| 5 | **Delivery parser** | Parse `deliver method { params } { on_hit { ... } }` blocks |
| 6 | **Ability parser** | Parse full `ability Name { ... }` blocks |
| 7 | **Passive parser** | Parse `passive Name { ... }` blocks |
| 8 | **Lowering** | Convert AST → `AbilityDef` / `PassiveDef` structs in `lower.rs` |
| 9 | **Error reporting** | Line/column numbers, context, suggestions |
| 10 | **Integration** | Add `abilities_file` support to `HeroToml` and loader |
| 11 | **Tests** | Unit tests for parser + round-trip tests (parse DSL → struct → compare with TOML parse) |
| 12 | **Convert Mage** | Write `mage.ability`, update `mage.toml` to use `abilities_file = "mage.ability"` |
| 13 | **Update /ability command** | Modify `.claude/commands/ability.md` to generate DSL instead of TOML |
| 14 | **Migrate remaining heroes** | Write a converter script or manually convert all 27 heroes |

### 3.6 Dependencies

Add to `Cargo.toml`:
```toml
winnow = "0.7"
```

No other new dependencies needed.

---

## 4. Key Design Decisions

1. **Abilities-only DSL** — Hero definitions (stats, attack) remain TOML. The DSL
   only covers `ability` and `passive` blocks.

2. **Direct parsing** — No TOML transpilation. The DSL parser produces
   `AbilityDef`/`PassiveDef` directly, avoiding double-indirection.

3. **Both formats supported** — Hero TOML files can use inline `[[abilities]]`
   (backwards-compatible) OR `abilities_file` to reference a `.ability` file. Gradual migration.

4. **Brace-based syntax** — Familiar to Rust developers, unambiguous block boundaries,
   no whitespace-sensitivity issues.

5. **Effects as one-liners** — The biggest verbosity win. `damage 55 in circle(3.0) [FIRE: 60]`
   replaces 8+ lines of TOML.

6. **Duration shorthand** — `5s` instead of `5000` everywhere. Huge readability win.

7. **Inline tags** — `[FIRE: 60, MAGIC: 40]` on the same line as the effect.

8. **Conditions inline** — `when target_hp_below(30%)` on the effect line.

---

## 5. Templates (Sub-Turing Composition)

The DSL supports parameterized templates for reuse across abilities. Templates are
**pure substitution** — no recursion, no loops, no conditionals at the template level.
This keeps the language sub-Turing (guaranteed termination) while enabling composition.

### 5.1 Template Syntax

```
template aoe_damage(amount, radius, tag, power) {
    damage $amount in circle($radius) [$tag: $power]
}

template cc_slow(factor, dur, radius) {
    slow $factor for $dur in circle($radius)
}

template projectile_nuke(speed, amount, tag, power) {
    deliver projectile { speed: $speed, width: 0.3 } {
        on_hit {
            damage $amount [$tag: $power]
        }
    }
}
```

### 5.2 Template Usage

```
ability Blizzard {
    target: ground, range: 6.0
    cooldown: 12s, cast: 400ms
    hint: damage

    deliver zone { duration: 4s, tick: 1s } {
        on_hit {
            aoe_damage(15, 3.0, ICE, 50)
            cc_slow(0.3, 1.5s, 3.0)
        }
    }
}

ability Meteor {
    target: ground, range: 6.0
    cooldown: 18s, cast: 500ms
    hint: damage

    aoe_damage(80, 2.5, FIRE, 70)
}
```

### 5.3 Why Sub-Turing Is Intentional

- **Every valid program terminates** — no halting problem
- **Every generated ability is finite** — bounded by grammar depth
- **Templates are statically expanded** — no runtime evaluation
- **The grammar remains context-free** after expansion — critical for the
  generative grammar and transformer tokenization (see sections 6 and 7)

---

## 6. Generative Grammar

The DSL's grammar can be used **bidirectionally**: parsing (text → AST) and
generation (random walk → AST → ability). This enables procedural ability creation
without an LLM.

### 6.1 Grammar as Production Rules

```
ability      → name × targeting × timing × (delivery | effects+)
targeting    → enemy | ally | self | self_aoe | ground | direction | vector | global
timing       → cooldown × cast_time
delivery     → projectile(params) × hooks | chain(params) × hooks | zone(params) × hooks | ...
hooks        → on_hit(effects) | on_arrival(effects) | on_complete(effects)
effect       → effect_type × area? × tags? × condition? × scaling?
effect_type  → damage(amount) | heal(amount) | stun(dur) | slow(factor, dur) | ...
area         → circle(r) | cone(r, angle) | line(len, w) | ring(inner, outer) | ...
condition    → target_hp_below(%) | caster_hp_below(%) | target_is_stunned | ...
```

### 6.2 Weighted Sampling

Each production rule has a weight and parameter constraints:

```
Rule                          Weight    Constraint
─────────────────────────────────────────────────────────
targeting = self_aoe          0.15      → range = 0, area required
targeting = enemy             0.35      → range ∈ [2.0, 6.0]
delivery = projectile         0.30      → requires targeting ∈ {enemy, ground, direction}
effect = damage               0.40      → amount ∈ [15, 80], scales with cooldown
effect = stun                 0.10      → duration ∈ [0.5s, 2.5s], needs CC tag
area = circle                 0.50      → radius ∈ [1.5, 4.0]
add_condition                 0.15      → more likely on high-damage effects
add_scaling                   0.10      → weighted toward execute-style at low hp
```

### 6.3 Balance Constraints

The generator enforces **power budget** rules during sampling:

- **Damage budget**: `total_damage × (1 / cooldown_s)` ≈ target DPS range
- **CC budget**: `cc_duration_s × (1 / cooldown_s)` ≤ max CC uptime
- **Healing budget**: `total_heal × (1 / cooldown_s)` ≈ target HPS range
- **Complexity cap**: max 4 effects per ability, max 2 conditions
- **Tag coherence**: fire abilities shouldn't also have ICE tags (unless intentional)

### 6.4 Use Cases

1. **Procedural ability generation** — Roguelike/ARPG random abilities via grammar walk
2. **Ability fuzzing** — Generate thousands of valid abilities to stress-test the sim
3. **Balance parameter sweeps** — Fix structure, sweep numeric parameters
4. **Ability mutation** — Swap one subtree in an existing ability's parse tree
5. **Training data generation** — Create synthetic abilities for transformer pre-training

### 6.5 Implementation

```
src/ai/effects/dsl/
    generate.rs       — grammar + weights → AST (generation)
    constraints.rs    — balance rules, validity checks, power budget
```

`generate.rs` walks the same AST types as the parser but in reverse — producing
nodes from weighted random choices. Then `lower.rs` converts them to `AbilityDef`.

---

## 7. Transformer Architecture for Ability Evaluation

Replace the current hand-crafted feature extraction + per-category MLP pipeline with
a transformer that takes tokenized DSL ability definitions as input.

### 7.1 Current Pipeline (What We Replace)

```
AbilityDef → extract_ability_properties() → 80-dim float vector
                                                    ↓
                                            MLP autoencoder (80 → 32)
                                                    ↓
                                            32-dim ability embedding
                                                    ↓
Game State → hand-crafted features ───→ concat with embedding
                                                    ↓
                                        Per-category MLP (×9 categories)
                                                    ↓
                                        urgency + target_scores
```

**Problems with current approach:**
- `extract_ability_properties()` is 200 lines of hand-crafted normalization
- Adding a new effect requires updating properties.rs, retraining autoencoder, retraining 9 MLPs
- 9 separate per-category models (DamageUnit, CcUnit, HealUnit, etc.) — no cross-category learning
- Fixed 80-dim representation can't capture compositional ability interactions

### 7.2 Token Vocabulary

The DSL grammar's terminals define a natural vocabulary (~160 tokens):

| Category | Tokens | Count |
|----------|--------|-------|
| Structure | `ABILITY`, `PASSIVE`, `DELIVER`, `ON_HIT`, `ON_ARRIVAL`, `RECAST`, `MORPH` | ~10 |
| Targeting | `ENEMY`, `ALLY`, `SELF`, `SELF_AOE`, `GROUND`, `DIRECTION`, `VECTOR`, `GLOBAL` | 8 |
| Effects | `DAMAGE`, `HEAL`, `SHIELD`, `STUN`, `SLOW`, `DASH`, `BUFF`, ... (all 52) | 52 |
| Areas | `CIRCLE`, `CONE`, `LINE`, `RING`, `SPREAD` | 5 |
| Delivery | `PROJECTILE`, `CHAIN`, `ZONE`, `CHANNEL`, `TETHER`, `TRAP` | 6 |
| Conditions | `WHEN`, `TARGET_HP_BELOW`, `CASTER_HP_ABOVE`, `AND`, `OR`, `NOT`, ... | ~30 |
| Triggers | `ON_DAMAGE_DEALT`, `ON_KILL`, `ON_HP_BELOW`, ... | ~18 |
| Tags | `FIRE`, `ICE`, `MAGIC`, `CROWD_CONTROL`, `PHYSICAL`, ... | open set |
| Hints | `HINT_DAMAGE`, `HINT_HEAL`, `HINT_CC`, `HINT_DEFENSE`, `HINT_UTILITY` | 5 |
| Numbers | Quantized buckets (see §7.3) | ~20 |
| Duration | `DUR_SHORT` (<1s), `DUR_MED` (1-3s), `DUR_LONG` (3-8s), `DUR_VLONG` (8s+) | 4 |
| Special | `PAD`, `SEP`, `CLS`, `GAME_STATE` | 4 |
| **Total** | | **~160** |

### 7.3 Number Quantization

Numeric values are bucketed to keep vocabulary small:

```
Damage/Heal amounts:  [0, 10, 20, 30, 40, 50, 65, 80, 100, 150]  → 10 tokens
Percentages:          [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]   → 10 tokens
Distances/Radii:      [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0] → 10 tokens
Factors (0-1):        [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]   → 9 tokens
Speeds:               [2, 4, 6, 8, 10, 12, 15, 20]               → 8 tokens
```

Each bucket maps to the nearest value. The slight quantization loss is acceptable
because the model learns to work with bucketed values, and exact numeric precision
matters less than structural understanding for decision-making.

### 7.4 Tokenized Examples

**Fireball:**
```
[CLS] ABILITY ENEMY RANGE_5 CD_MED CAST_SHORT HINT_DAMAGE
      DELIVER PROJECTILE SPEED_8 WIDTH_03
      ON_HIT DAMAGE AMT_55 TAG FIRE POW_60
      ON_ARRIVAL DAMAGE AMT_15 CIRCLE RAD_2 [SEP]
```
(~20 tokens)

**Frost Nova:**
```
[CLS] ABILITY SELF_AOE CD_LONG CAST_SHORT HINT_CC
      DAMAGE AMT_20 CIRCLE RAD_3
      STUN DUR_MED CIRCLE RAD_3 TAG CROWD_CONTROL POW_80 TAG ICE POW_60 [SEP]
```
(~18 tokens)

**Full hero kit**: 8 abilities + 2 passives ≈ 200 tokens (well within transformer capacity).

### 7.5 Model Architecture

```
                    Ability Token Sequence (~20 tokens)
                           ↓
               ┌───────────────────────┐
               │  Token Embedding      │  160 vocab → 64-dim
               │  + Positional Enc     │  learnable positions
               └───────────┬───────────┘
                           ↓
               ┌───────────────────────┐
               │  Transformer Encoder  │  2-3 layers, 4 heads
               │  (self-attention)     │  d_model=64, d_ff=128
               │                       │  ~50K-100K params
               └───────────┬───────────┘
                           ↓
                    [CLS] embedding (64-dim)
                           ↓
               ┌───────────────────────┐
               │  Cross-Attention      │  ability embedding attends
               │  with Game State      │  to game state features
               │                       │  (same features as today:
               │                       │   HP%, distance, team, etc.)
               └───────────┬───────────┘
                           ↓
               ┌───────────────────────┐
               │  Decision Head MLP    │  64 → 32 → (1 + N_targets)
               │                       │  → urgency + target scores
               └───────────────────────┘
```

**Key architectural choices:**
- **[CLS] token** pools the entire ability representation via self-attention
- **Cross-attention** lets the ability representation query game state features
  (replaces the current concat approach)
- **Single model** replaces all 9 per-category MLPs — the token sequence itself
  encodes the category
- **Same game state features** as the current system (HP%, distance, team balance, etc.)
  — we're replacing the ability encoding, not the game state encoding

### 7.6 What Self-Attention Learns

The transformer's attention mechanism can discover compositional patterns that
the hand-crafted 80-dim vector cannot represent:

- `DAMAGE + CIRCLE + ZONE` = sustained AoE → high urgency when enemies cluster
- `DAMAGE + PROJECTILE + ON_HIT + STUN` = single-target CC chain → high urgency on priority target
- `HEAL + SELF_AOE + CIRCLE` = AoE heal → urgency scales with team HP deficit
- `STUN + WHEN + TARGET_HP_BELOW` = conditional execute setup → urgency spikes at threshold
- `DELIVER + CHAIN + BOUNCES_3 + DAMAGE` = multi-target → urgency scales with enemy count

The current system can't compose these because each is a separate float in a fixed vector.

### 7.7 Training Pipeline

```
Phase 1: Pre-train ability encoder (self-supervised, masked token modeling)
  ┌─────────────────────────────────────────────────────────┐
  │ Input:  [CLS] ABILITY ENEMY RANGE_5 [MASK] CAST_SHORT  │
  │         DELIVER PROJECTILE SPEED_8 [MASK] ...           │
  │ Target: predict masked tokens                           │
  │ Data:   27 heroes × ~10 abilities = ~270 real           │
  │         + 5000-10000 grammar-generated synthetic         │
  │ Loss:   cross-entropy on masked token predictions       │
  └─────────────────────────────────────────────────────────┘

  This teaches the transformer the structure of valid abilities.
  The generative grammar (§6) is critical here for creating enough
  training data.

Phase 2: Fine-tune for decision-making (supervised, oracle ground truth)
  ┌─────────────────────────────────────────────────────────┐
  │ Input:  [CLS] ability_tokens [SEP] game_state_tokens    │
  │ Target: urgency (sigmoid) + target_scores (softmax)     │
  │ Data:   oracle rollout ground truth (existing pipeline)  │
  │ Loss:   BCE(urgency) + CE(target_selection)             │
  │ Replaces: autoencoder + 9 per-category MLPs             │
  └─────────────────────────────────────────────────────────┘

Phase 3: Fine-tune for generation (autoregressive, optional)
  ┌─────────────────────────────────────────────────────────┐
  │ Input:  game_state + desired_hint + power_budget        │
  │ Target: valid ability token sequence                    │
  │ Decoding: grammar-constrained (only valid next tokens)  │
  │ Use:    procedural ability creation at runtime          │
  └─────────────────────────────────────────────────────────┘
```

### 7.8 Grammar-Constrained Decoding (Phase 3)

When generating abilities, the transformer's output distribution is **masked** at each
step to only allow tokens that are valid according to the grammar. This guarantees
every generated sequence is a syntactically valid ability:

```
State: [CLS] ABILITY ENEMY RANGE_5 CD_MED CAST_SHORT HINT_DAMAGE DELIVER _
                                                                          ↑
Valid next tokens: { PROJECTILE, CHAIN, ZONE, CHANNEL, TETHER, TRAP }
Invalid tokens:   { DAMAGE, HEAL, STUN, ENEMY, ... } ← masked to -∞
```

This is where the formal grammar and the transformer meet: the grammar provides
the constraint mask, the transformer provides the probability distribution.

### 7.9 Performance Budget

| Component | Params | Inference (est.) | Notes |
|-----------|--------|-------------------|-------|
| **Current** | | | |
| Autoencoder (80→32) | ~5K | ~1μs | Per ability, cached |
| Per-category MLP (×9) | ~25K total | ~2μs | Per ability eval |
| **Proposed** | | | |
| Token embedding | ~10K | negligible | 160 × 64 |
| 2-layer transformer | ~35K | ~5μs | 4 heads, d=64 |
| Cross-attention + head | ~5K | ~2μs | |
| **Total proposed** | ~50K | ~7μs | Per ability eval |

With 8 abilities × 10 units = 80 evals/tick at 7μs = **0.56ms/tick**. Current sim
runs at ~1ms/tick, so this is well within budget for 100Hz.

**Optimization path if needed:**
- Cache [CLS] embeddings (ability definitions don't change mid-fight) → amortize to ~2μs
- SIMD vectorization (existing AVX2+FMA pattern in student model) → 2-4x speedup
- Quantize to int8 → 2x memory, ~1.5x speed

### 7.10 Advantages Over Current System

| Dimension | Current (MLP) | Proposed (Transformer) |
|-----------|---------------|----------------------|
| New effect type | Update properties.rs (80-dim), retrain autoencoder + 9 MLPs | Add token to vocab, fine-tune one model |
| Cross-category learning | None (9 separate models) | Shared attention layers learn universal patterns |
| Compositionality | Flat vector, no interaction | Self-attention discovers effect combinations |
| Feature engineering | Manual (200 lines of normalization) | Learned from token sequences |
| Ability generation | Separate system (grammar only) | Same model, autoregressive mode |
| Embedding quality | Fixed 80→32 bottleneck | Contextual, variable-length input |

### 7.11 Implementation Steps

| Step | What | Details |
|------|------|---------|
| 1 | **Tokenizer** | `src/ai/core/ability_encoding/tokenizer.rs` — AbilityDef → token sequence |
| 2 | **Vocabulary** | Define token enum, quantization buckets, special tokens |
| 3 | **Transformer** | `src/ai/core/ability_encoding/transformer.rs` — inference-only, SIMD |
| 4 | **Cross-attention** | Fuse ability embedding with existing game state features |
| 5 | **Decision head** | Replace per-category dispatch in `eval.rs` with single head |
| 6 | **Training script** | Python script: generate training data → train → export JSON weights |
| 7 | **Pre-training** | Masked token modeling on real + grammar-generated abilities |
| 8 | **Fine-tuning** | Oracle ground truth → urgency + target prediction |
| 9 | **A/B comparison** | Run MLP vs transformer eval side-by-side, compare oracle agreement |
| 10 | **Migrate** | Swap default eval to transformer, keep MLP as fallback |

### 7.12 New Files

```
src/ai/core/ability_encoding/
    tokenizer.rs      — AbilityDef → token IDs (using DSL grammar vocabulary)
    vocab.rs          — token enum, quantization, special tokens
    transformer.rs    — inference-only transformer encoder (SIMD-optimized)

src/ai/core/ability_eval/
    eval_transformer.rs  — transformer-based evaluate_abilities() replacing eval.rs
    cross_attention.rs   — ability × game_state cross-attention fusion

training/
    pretrain_ability_transformer.py   — masked token modeling
    finetune_decision.py              — oracle-supervised fine-tuning
    export_weights.py                 — PyTorch → JSON weight export
```

---

## 8. Estimated Reduction

| Hero | TOML lines | DSL lines (est.) | Reduction |
|------|-----------|-------------------|-----------|
| Mage | 231 | ~90 | ~61% |
| Elementalist | 240 | ~95 | ~60% |
| Average hero | ~200 | ~80 | ~60% |
