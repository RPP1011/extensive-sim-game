# TWI DSL Extensions — 5 Missing Primitives

Based on analysis of 3768 TWI skills from the wiki, these 5 primitives cover ~80% of unexpressible skills.

## 1. Auras (persistent radius effects)

TWI examples:
- [Aura of the Brave] — allies resist fear in radius
- [Aura of Command] — NPCs obey in radius
- [Aura of Disarming] — enemies drop weapons in radius
- [Aegis of my Contempt] — wards allies against remote skills within 1000 paces

**DSL syntax:**
```dsl
// New duration type: `while_alive` — persists as long as caster lives
// Auras are just effects with `while_alive` duration and an area

ability AuraOfTheBrave {
    target: self_aoe
    cooldown: 0s
    hint: defense

    immunity "fear" in circle(10.0) while_alive
}

ability AuraOfDisarming {
    target: self_aoe
    cooldown: 0s
    hint: crowd_control

    disarm for 1s in circle(5.0) while_alive tick 2s
}

// `while_alive` = persistent, re-applied every tick
// `tick Ns` on the effect = how often it pulses (for non-instant effects)
```

**Implementation:**
- New duration keyword: `while_alive` → `duration_ms = u32::MAX` (or a sentinel like 0xFFFFFFFF)
- New effect modifier: `tick Ns` → sets re-application interval
- Runtime: aura effects register as persistent status on caster, pulse every tick interval to units in area
- On caster death: remove all aura effects from affected units

## 2. Targeting Filters

TWI examples:
- [Army of the King] — all sworn soldiers get temp levels
- [A Fraction of My Experience] — soldiers under command get knowledge
- [Army: Aspect of Mithril] — army gets mithril coating
- [Reward of the Loyal] — permanent blessings to those who fight for user

**DSL syntax:**
```dsl
// `targeting PREDICATE` after an effect filters which entities are affected
// Predicates: loyalty_above(N), class("X"), level_above(N), status("X"),
//             under_command, faction("X"), has_skill("X"), injured, healthy

ability ArmyOfTheKing {
    target: global
    cooldown: 720000s, cast: 0ms
    hint: leadership

    modify_stat "unit" "effective_level" +10 for 86400s
        targeting under_command
        + 0.5% caster_level
    buff damage_output 0.3 for 86400s in global
        targeting under_command
    buff armor 20 for 86400s in global
        targeting under_command
}

ability RewardOfTheLoyal {
    target: global
    cooldown: 0s
    hint: leadership

    buff damage_output 0.15 permanent
        targeting loyalty_above(80)
}
```

**Implementation:**
- New optional field on `ConditionalEffect`: `targeting_filter: Option<TargetFilter>`
- `TargetFilter` enum: `UnderCommand`, `LoyaltyAbove(f32)`, `LoyaltyBelow(f32)`, `HasClass(String)`, `LevelAbove(u32)`, `LevelBelow(u32)`, `HasStatus(String)`, `FactionMember(String)`, `Injured`, `Healthy`
- Parser: `targeting KEYWORD(args)` after effect line, before tags
- At apply-time: iterate all entities matching area, filter by predicate, apply effect to matches

## 3. External Scaling

TWI examples:
- [A Kingdom's Strength] — strength proportional to kingdom size
- [Army: Duplicate Projectiles] — double projectiles for army
- [Undying Loyalty] — power scales with followers' devotion

**DSL syntax:**
```dsl
// `scales_with STAT_REF` — multiplies effect by external game state
// Stat refs: kingdom_size, army_size, faction_territory, guild_reputation,
//            adventurer_count, loyalty_average, party_size, gold

ability KingdomsStrength {
    target: self
    cooldown: 0s
    hint: utility

    buff damage_output 0.01 while_alive
        scales_with kingdom_size
}

ability StrengthInNumbers {
    target: self_aoe
    cooldown: 10s
    hint: defense

    buff armor 2 for 10s in circle(8.0)
        scales_with party_size
}
```

**Implementation:**
- Extend `ScalingTerm` (already exists for `+ N% target_max_hp`) with campaign stat references
- New `StatRef` variants: `KingdomSize`, `ArmySize`, `FactionTerritory`, `GuildReputation`, `AdventurerCount`, `LoyaltyAverage`, `PartySize`, `Gold`
- Parser: `scales_with KEYWORD` after effect, maps to a `ScalingTerm` with the external stat
- At apply-time: resolve the stat ref against campaign state, multiply effect amount

## 4. Charge/Daily System

TWI examples:
- [A Bottle A Day: Ink] — once a day, fills a bottle with ink
- [A Bullet A Day: Armor Piercing Round] — once a day, conjures AP bullet
- [A Magical Gift] — once a week at lv30, once a day at lv40
- [Copy Recipe] — one dish per day

**DSL syntax:**
```dsl
// `charges N recharge Ns` already exists in AbilityDef
// Extend with `recharge daily` and `recharge weekly` for real-time gates

ability BottleADayInk {
    target: self
    charges: 1, recharge: daily
    hint: utility

    create_entity "item" "ink_bottle"
}

ability MagicalGift {
    target: ally, range: 10.0
    charges: 1, recharge: daily
    hint: utility

    grant_ability "random_useful" for 86400s
}

// `recharge: daily` = 86400 seconds (24h real time)
// `recharge: weekly` = 604800 seconds
// Could also do `recharge: per_quest` or `recharge: per_battle`
```

**Implementation:**
- New `recharge` property values in parser: `daily`, `weekly`, `per_quest`, `per_battle`
- Maps to: `daily` → `charge_recharge_ms = 86400000`, `weekly` → `604800000`
- `per_quest` and `per_battle` need event-based recharge (charges restore on quest_complete / battle_end)
- AbilityDef already has `max_charges` and `charge_recharge_ms` — just needs the parser keywords and event-based variant

## 5. Combination Skills

TWI examples:
- [Meteor Guardbreaker Shot] — 3 people working together
- [Bolt from the Heavens] — requires all participants each time
- [Combined Skill: Barrier of Ego] — multiple nobles create barrier
- [Lightning Tempest] — alchemist + lightning bottles + amplifier

**DSL syntax:**
```dsl
// `requires N participants` — skill only fires when N units use it together
// `requires_class "X"` — at least one participant must have class X
// `requires_skill "X"` — at least one participant must have skill X

ability MeteorGuardbreakerShot {
    target: enemy, range: 8.0
    cooldown: 30s, cast: 1s
    hint: damage
    requires: 3 participants

    damage 300 [PHYSICAL: 80]
    knockback 5.0
    stun 2s
}

ability BoltFromTheHeavens {
    target: ground, range: 10.0
    cooldown: 45s, cast: 2s
    hint: damage
    requires: 2 participants
    requires_class: "mage"

    damage 500 [MAGIC: 90]
    stun 3s in circle(3.0)
}

// When a unit tries to cast a `requires: N` skill:
// 1. Check if N-1 other allies within range also have this skill
// 2. If yes, all N units go on cooldown, combined effect fires
// 3. If no, skill fails / queues waiting for others
```

**Implementation:**
- New AbilityDef fields: `requires_participants: u32`, `requires_class: Option<String>`, `requires_skill: Option<String>`
- Parser: `requires: N participants`, `requires_class: "X"`
- Combat sim: when ability cast is initiated with requires > 1, check nearby allies for matching ability. If found, trigger combined effect. If not, fail/queue.
- Combination damage/effects can be `amplify` scaled by participant count

## Implementation Priority

1. **Targeting filters** — enables army-wide skills, most impactful for campaign
2. **Auras** — `while_alive` duration + tick interval, enables persistent buffs
3. **Charge system** — `daily`/`weekly`/`per_battle` recharge keywords, low effort
4. **External scaling** — extend existing `ScalingTerm` with campaign stats
5. **Combination skills** — most complex, can defer

## Files to Modify

All 5 features touch:
- `effect_enum.rs` or `types.rs` — new types/fields
- `defs.rs` — new AbilityDef fields (charges, requires)
- `parser.rs` — new keywords
- `lower.rs` / `lower_effects.rs` — new lowering
- `emit.rs` / `emit_effects.rs` — new emission
- `tokenizer_vocab.rs` — new tokens
- `grammar_space.rs` — new dimensions
