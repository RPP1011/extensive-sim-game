# LoL Champion -> Hero TOML Conversion Plan

## Overview
Convert all 172 LoL champions into playable hero TOML files using an automated
pipeline with manual review for complex cases.

**Champion tiers**: 28 simple, 107 medium, 37 complex

## Phase 0: Last Two Mechanic Gaps (1 session)

### 0a. Directed Summons (Azir, Zyra, Yorick, Heimerdinger)
Files: `effects/types.rs`, `effects/defs.rs`, `core/types.rs`, `core/apply_effect.rs`, `core/simulation.rs`

- `Effect::Summon` — add `directed: bool` (unit doesn't generate its own intents)
- `UnitState` — add `owner_id: Option<u32>` (links summon to caster)
- `simulation.rs` step() — skip intent for units with `owner_id` where directed;
  instead when owner attacks, each directed summon in attack range also attacks from
  its position
- New `Effect::CommandSummons` — move all owned directed summons toward target pos
  (for Azir Q). Fields: `{ speed: f32 }` — summons dash to location

### 0b. Global Targeting
- Add `AbilityTargeting::Global` — targets all enemies on the map
- In `oracle.rs` + `combat.rs` match arms, treat like SelfAoe but hits all enemies
- In `resolve_targets`, return all enemies regardless of position

## Phase 1: Converter Script (`scripts/lol_to_toml.py`)

### Input
- `assets/lol_champions/*.json` (172 files with Data Dragon + wiki data)
- `docs/ABILITY_SYSTEM.md` (effect/area/delivery reference)

### Stat Normalization
LoL stats -> our scale:
| LoL stat       | Our stat      | Formula                          |
|----------------|---------------|----------------------------------|
| HP (600-2500)  | hp (80-250)   | `hp = lol_hp / 10`               |
| AD (50-130)    | attack.damage | `dmg = lol_ad / 4`               |
| AS (0.6-1.0)   | attack.cooldown| `cd = 1000 / lol_as`             |
| Range (125-650)| attack.range  | `range = lol_range / 100`        |
| MS (325-355)   | move_speed    | `ms = lol_ms / 100`              |
| Armor (25-45)  | armor         | `armor = lol_armor`              |
| MR (30-53)     | magic_resist  | `mr = lol_mr`                    |
| Mana (250-500) | max_resource  | `resource = lol_mana / 5`        |

### Ability Mapping Rules
The converter maps LoL abilities using keyword detection + wiki targeting data:

```
DAMAGE TYPE:
  wiki_detail.damagetype == "Physical" -> damage_type = "physical"
  wiki_detail.damagetype == "Magic"    -> damage_type = "magic"
  wiki_detail.damagetype == "True"     -> damage_type = "true"

TARGETING:
  wiki "Location"         -> targeting = "ground_target"
  wiki "Direction"        -> targeting = "direction"
  wiki "Unit" + enemy     -> targeting = "target_enemy"
  wiki "Unit" + ally      -> targeting = "target_ally"
  wiki "Auto"/"Passive"   -> targeting = "self_cast"

DELIVERY (from description keywords):
  "fires a projectile/skillshot/bolt" -> Projectile
  "channels for X seconds"           -> Channel
  "creates a zone/field/area"        -> Zone
  "places a trap/mine/ward"          -> Trap
  "tethers to/leashes"               -> Tether
  "bounces to/chains to"             -> Chain
  default                            -> Instant

EFFECTS (from description keywords):
  "deals X damage"        -> Damage { amount: X_normalized }
  "heals for X"           -> Heal { amount: X_normalized }
  "shields for X"         -> Shield { amount: X_normalized }
  "stuns for X seconds"   -> Stun { duration_ms: X*1000 }
  "slows by X%"           -> Slow { factor: X/100 }
  "roots/snares"          -> Root
  "knocks back/up"        -> Knockback
  "dashes/leaps/lunges"   -> Dash
  "silences"              -> Silence
  "fears/terrifies"       -> Fear
  "taunts"                -> Taunt
  "suppresses"            -> Suppress
  "grounds"               -> Grounded
  "charms"                -> Charm
  "stealths/invisible"    -> Stealth
  "% max health"          -> scaling_stat: "target_max_hp"
  "% missing health"      -> scaling_stat: "target_missing_hp"

AI_HINT:
  Primary damage ability      -> "damage"
  CC ability                  -> "crowd_control"
  Heal/shield ability         -> "heal" / "defense"
  Mobility ability            -> "utility"
  Buff/steroid ability        -> "buff"
```

### Converter Architecture
```
lol_to_toml.py
├── parse_champion(json_path) -> ChampionData
├── normalize_stats(lol_stats) -> HeroStats
├── map_ability(ability_data, slot) -> AbilityDef
│   ├── detect_targeting(wiki_detail, description)
│   ├── detect_delivery(description, wiki_detail)
│   ├── extract_effects(description, wiki_detail)
│   ├── extract_cooldown(cooldown_data)
│   └── detect_special_mechanics(description)
│       ├── check_recast()
│       ├── check_charges()
│       ├── check_toggle()
│       ├── check_transform()
│       └── check_passive_component()
├── map_passive(passive_data) -> PassiveDef
├── generate_toml(champion_data) -> str
└── validate_toml(toml_str) -> bool
```

### Output
- `assets/lol_heroes/{champion_name}.toml` — one file per champion
- `assets/lol_heroes/_manifest.json` — conversion metadata (warnings, manual flags)

## Phase 2: Batch Generate (automated)

1. Run `python3 scripts/lol_to_toml.py --all`
2. Generates 172 TOML files in `assets/lol_heroes/`
3. Manifest tracks per-champion: conversion confidence (high/medium/low),
   flagged abilities (couldn't auto-map), missing wiki data

## Phase 3: Validate (automated)

`scripts/validate_lol_heroes.py` or `xtask lol validate`:
1. **Parse check** — every TOML loads without error
2. **Ability check** — every ability has at least one effect
3. **Stat check** — HP > 0, move_speed > 0, cooldowns > 0
4. **Sim check** — spawn each champion in a 1v1 duel, run 200 ticks,
   verify no panics and abilities fire
5. Report: `X/172 pass, Y warnings, Z failures`

## Phase 4: Manual Review (complex champions)

The 37 complex champions get manual review:
- Transform champions (Jayce, Nidalee, Elise, Gnar): verify form swap works
- Directed summons (Azir, Zyra, Heimerdinger): verify soldier mechanics
- Multi-recast (Riven, Aatrox, Samira): verify recast chains
- Clone champions (Shaco, LeBlanc, Wukong): verify clone behavior
- Unique passives (Viego, Sylas): may need approximation

## Phase 5: Integration

1. Register all lol_heroes in `hero_templates.rs` (or dynamic loader)
2. Create `scenarios/lol/` benchmark scenarios (mirror matchups, team fights)
3. Run AI effectiveness tests across all champions
4. Add to oracle training pipeline for student model

## File Structure
```
assets/
  lol_champions/     # 172 raw JSON (already exists)
  lol_heroes/        # 172 generated TOML (Phase 2)
scripts/
  lol_to_toml.py     # Converter (Phase 1)
  validate_lol_heroes.py  # Validator (Phase 3)
```

## Key Design Decisions

1. **Approximate, don't replicate** — we're mapping to our system, not cloning LoL.
   Passives that reference LoL-specific UI (minimap, shop) get dropped or simplified.

2. **One TOML per champion** — even multi-form champions use a single file with
   `swap_form`/`morph_into` for alternate ability sets.

3. **Damage values are relative** — normalized to our HP pool (80-250).
   A LoL ability doing 300 damage to a 2000 HP target ≈ 15% HP,
   so in our system: `amount = target_hp * 0.15`.

4. **Missing data defaults** — if wiki doesn't have exact values, use sensible
   defaults from the ability description (e.g., "briefly stuns" = 1000ms).

5. **AI hints are critical** — every ability needs a correct `ai_hint` so the
   squad AI uses it properly. Damage abilities → "damage", CC → "crowd_control", etc.
