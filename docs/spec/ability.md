# Ability DSL — Language Reference

> ⚠️ **Spec audit (2026-05-08) — `[CRITICAL]` items:** The §22.2 `EffectOp`
> ordinal table (variants 8–26) is **wrong** — the engine shipped a different
> ordering during Wave 2 / Wave 3 / Lift A–D. Codegen against the documented
> ordinals would produce silently-mismatched chronicle records (e.g. `8` is
> `Root` in the engine, not `PlaceVoxels`). See the appended "Spec audit
> 2026-05-08" callout in §22.2 and the per-row reconciliation table in §8.11.
> The companion event ordinal table (§22.4) carried the same shape mismatch
> and is replaced with the shipped chronicle ordinals 26–76.
>
> Counts: **CRITICAL 1 / STATUS-FLIP 28 / UNDOCUMENTED 16 / MISSING 13 /
> STALE 2.**

> ⚠️ **Audit 2026-04-26 — two-stack reality:** This spec describes a future-state engine path. There are **two parallel ability stacks** in the codebase:
> - **Engine** (`crates/engine`) — what this spec targets. Hand-built `AbilityProgram`. `Delivery` enum has only `Instant`. `Area` enum has only `SingleTarget`. `EffectOp` has 8 variants (0-7). No passive triggers. The §23 status matrix describes only this stack.
> - **TacticalSim** (`crates/tactical_sim/src/effects/dsl/`) — the running implementation for campaign play. Winnow-based parser, full `Delivery` (Projectile/Chain/Zone/Channel), full `Area` (all 5 disc shapes), 17+ passive trigger kinds, control/movement/buff verbs (charm, polymorph, dash, knockback, buff, debuff, …). TacticalSim is **1–2 tracks ahead** of the spec's `runs-today` claims, and implements several constructs the spec marks `reserved` (charm, polymorph, banish, reflect, summon, leash, redirect, …).
>
> All `.ability` files in `dataset/hero_templates/` and `assets/hero_templates/` are loaded by the **TacticalSim** parser, **not** by the engine or `dsl_compiler`. Cross-check spec claims against TacticalSim's implementation as a reference.
> See `docs/superpowers/notes/2026-04-26-audit-ability.md` for the per-§23-row status (48 rows: ~10 ✅ engine / ~8 🤔 TacticalSim-only / ~15 ⚠️ partial / ~12 ❌).
>
> **NOTE (2026-05-08):** The `tactical_sim` crate has been retired
> (Phase 7 wolf-sim wipe, 2026-05-02). The "two-stack reality" callout is
> retained for historical context but the second stack no longer exists —
> the only `.ability` consumer today is `crates/dsl_compiler` →
> `crates/engine`. The `assets/hero_templates/` directory was retired and
> is not coming back; the `.ability` corpus lives at
> `dataset/abilities/lol_heroes/` and is only loaded for parser regression
> coverage.

> **Status:** Design spec (2026-04-24). Canonical syntax, semantics, and
> IR-lowering contract for `.ability` files. Authoritative for language
> surface; defers runtime cascade semantics to `assets/sim/physics.sim`
> and broader design rationale / training pipelines to `PLAN.md`.
>
> Companion to:
> - `docs/spec/language.md` — world-sim DSL.
> - `PLAN.md` — ability DSL design rationale + training pipelines.
> - `docs/superpowers/specs/2026-04-22-theory-of-mind-design.md` — Phase 1 belief state.
> - `docs/spec/economy.md` — economic system design,
>   recipes-as-abilities anchor, and the IR additions for variants 17–26
>   (Recipe, WearTool, TransferProperty, ForcibleTransfer, CreateObligation,
>   DischargeObligation, DefaultObligation, EstablishRoute, JoinCaravan,
>   TransferObligation). The economic spec extends this DSL's `EffectOp`
>   catalog and registers new entities (Tool, Component, Property, Obligation,
>   ResourceNode), registries (Recipe, ToolKind, Skill, ComponentKind,
>   Commodity, Route), and the `Economic` / `Financial` ability hints.

---

## §1 Scope & non-goals

### 1.1 Scope

This document is the canonical language reference for `.ability` files.
It defines the lexical grammar, formal grammar, surface semantics, and
lowering contract to the `AbilityProgram` IR. Every construct the DSL
can author is covered, whether or not the current engine runs it.

Each construct carries one of three status markers:

- **`runs-today`** — engine currently executes the construct.
- **`planned`** — engine will implement; parser accepts and lowers; runtime
  may reject with a documented error until the engine catches up.
- **`reserved`** — parser accepts; lowering is a stub error
  (`EmitError::Unsupported`) until the prerequisite engine work lands.

Status changes are engine-side PRs, not spec edits.

### 1.2 Non-goals

- **Runtime cascade semantics.** Owned by `assets/sim/physics.sim`.
- **Generative grammar weights, transformer tokenization, training
  pipelines.** Owned by `PLAN.md`.
- **Hero stats, attack stats, hero metadata.** Owned by hero `.toml`.
- **Voxel backend implementation details.** Owned by `crates/engine_voxel/*`
  when it lands.
- **`.sim` DSL grammar.** Owned by `docs/spec/language.md` and `docs/spec/compiler.md`.

### 1.3 Dependencies & layering

- Hero TOML files gain `abilities_file = "<name>.ability"`.
- The set of compiled `.ability` files produces one `AbilityRegistry`,
  frozen before tick 0.
- Material names resolve against a `.sim`-declared catalog (see §17).
- Voxel-aware conditions and writes share the terrain backend via
  `TerrainQuery` (`crates/engine/src/terrain.rs`).
- AI-state manipulation verbs emit events folded by `@materialized` views
  in `assets/sim/views.sim`; the DSL spec does not own the fold rules.

---

## §2 Lexical grammar

| Token | Regex / form | Notes |
|---|---|---|
| `IDENT` | `[A-Za-z_][A-Za-z0-9_]*` | case-sensitive |
| `TAG_SYMBOL` | `[A-Z][A-Z0-9_]*` | e.g. `FIRE`, `CROWD_CONTROL` |
| `INT` | `-?[0-9]+` \| `0x[0-9A-Fa-f_]+` `[ui](8\|16\|32\|64)?` | hex + Rust-style int suffix shipped 2026-05-08 (PR #38, `b3326fec`); suffix is consumed and discarded at the lex layer |
| `FLOAT` | `-?[0-9]+\.[0-9]+` | no exponent form in v1 |
| `DURATION` | `INT (s\|ms)` \| `INT` (bare = ms) \| `FLOAT (s\|ms)` | `5s`, `300ms`, `5000`, `1.5s` |
| `PERCENT` | `(INT\|FLOAT) %` | `30%`, `12.5%` |
| `STRING` | `"..."` with `\"` `\\` `\n` escapes | |
| `PARAM_REF` | `\$IDENT` | only inside `template` / `structure` bodies |
| `COMMENT` | `//...$` \| `#...$` | both accepted |
| `WS` | `[ \t\r\n]+` | significant only as separator |
| Punctuation | `{ } [ ] ( ) , : ;` | `;` is optional statement terminator |
| Operators | `+` `-` `*` `/` `%` `->` `==` `!=` `<` `>` `<=` `>=` | arithmetic + comparison |

### 2.1 Rules

- Keywords (`ability`, `passive`, `template`, `structure`, `material`,
  `deliver`, `in`, `when`, `else`, `for`, `stacking`, `chance`, `consume`,
  `cap`, `recast`, `morph`, `into`, `x`, `at`, `rotate`, `scale`, `if`,
  `place`, `harvest`, `transform`, `include`, `union`, `diff`, `intersect`,
  `min`, `max`, `clamp`, `abs`, `until_caster_dies`, `damageable_hp`,
  `drop_as`, `return_to_caster`, `symmetry`, `bounds`, `origin`,
  `rotatable`, `palette`, `from`, `to`, `of`, `as`, `with`, `by`, `range`,
  `through`, `channel`, `facing`, `thickness`) are reserved and not valid
  `IDENT`s at grammar positions that expect them.
- `x` is a keyword only in multiplicity position (`summon "skeleton" x3`).
- Numeric literals never carry thousands separators.
- Comments strip to newline; not preserved in AST.

---

## §3 Formal grammar (EBNF)

The complete grammar. Every later section references a production defined
here. Ambiguity notes follow.

```ebnf
(* ============================================================ *)
(* Top level                                                    *)
(* ============================================================ *)

file            = { top_item } ;
top_item        = ability | passive | template | structure ;
(* `material` is declared in `.sim`, not `.ability` (see §17).  *)

(* ============================================================ *)
(* Ability / Passive                                            *)
(* ============================================================ *)

ability         = "ability" IDENT "{" ability_body "}" ;
ability_body    = { header_prop ("," | NL) }
                  { body_item } ;
body_item       = effect_stmt
                | deliver_block
                | recast_block
                | morph_block ;

passive         = "passive" IDENT "{" passive_body "}" ;
passive_body    = trigger_decl
                  { header_prop ("," | NL) }
                  { effect_stmt } ;
trigger_decl    = "trigger" ":" trigger_expr ;

(* ============================================================ *)
(* Header properties                                            *)
(* ============================================================ *)

header_prop     = "target"       ":" targeting
                | "range"        ":" number
                | "cooldown"     ":" duration
                | "cast"         ":" duration
                | "hint"         ":" hint_name
                | "cost"         ":" INT
                | "zone_tag"     ":" STRING
                | "charges"      ":" INT
                | "recharge"     ":" duration
                | "toggle"
                | "toggle_cost"  ":" number
                | "recast"       ":" INT
                | "recast_window" ":" duration
                | "unstoppable"
                | "form"         ":" STRING
                | "swap_form"    ":" STRING ;

targeting       = "enemy" | "ally" | "self" | "self_aoe"
                | "ground" | "direction" | "vector" | "global" ;
hint_name       = "damage" | "defense" | "crowd_control"
                | "utility" | "heal" ;

(* ============================================================ *)
(* Effect statements — the one-liner                           *)
(* ============================================================ *)

effect_stmt     = effect_verb effect_args
                  [ shape_mod ]
                  [ tag_list ]
                  [ duration_mod ]
                  [ condition_mod ]
                  [ chance_mod ]
                  [ stacking_mod ]
                  [ scaling_mod ]
                  [ lifetime_mod ]
                  [ nested_block ]
                  [ ";" ] ;

effect_verb     = IDENT ;  (* §8 catalog validates at lowering *)
effect_args     = { effect_arg } ;
effect_arg      = number | duration | percent | STRING
                | material_ref | structure_ref | agent_ref
                | "x" INT
                | shape_expr
                | "to_target" number
                | "to_position"
                | material_transform ;

material_transform = material_ref "->" material_ref ;

shape_mod       = "in" shape_expr ;
shape_expr      = shape_atom { shape_op shape_atom } ;
shape_op        = "union" | "diff" | "intersect" ;
shape_atom      = "circle"   "(" number ")"
                | "cone"     "(" number "," number ")"
                | "line"     "(" number "," number ")"
                | "ring"     "(" number "," number ")"
                | "spread"   "(" number "," INT ")"
                | "box"      "(" number "," number "," number ")"
                | "sphere"   "(" number ")"
                | "column"   "(" number "," number ")"
                | "wall"     "(" number "," number "," number ")"
                                [ "facing" ":" number ]
                | "cylinder" "(" number "," number ")"
                | "dome"     "(" number ")"
                | "hull"     "(" number ")"
                | "(" shape_expr ")" ;

tag_list        = "[" tag_entry { "," tag_entry } "]" ;
tag_entry       = TAG_SYMBOL ":" number ;

duration_mod    = "for" duration ;

condition_mod   = "when" cond_expr [ "else" effect_stmt ] ;
cond_expr       = cond_atom
                | "and" "(" cond_expr "," cond_expr ")"
                | "or"  "(" cond_expr "," cond_expr ")"
                | "not" "(" cond_expr ")" ;
cond_atom       = cond_ident [ "(" cond_arg_list ")" ]
                | cond_ident comparison cond_value ;
comparison      = "==" | "!=" | "<" | ">" | "<=" | ">=" ;

chance_mod      = "chance" number ;
stacking_mod    = "stacking" ("stack" | "refresh" | "extend") ;

scaling_mod     = "+" percent stat_ref
                  [ "consume" ]
                  [ "cap" number ]
                  { "+" percent stat_ref } ;

lifetime_mod    = "until_caster_dies"
                | "damageable_hp" number ;

nested_block    = "{" { effect_stmt } "}" ;

(* ============================================================ *)
(* Delivery                                                     *)
(* ============================================================ *)

deliver_block   = "deliver" delivery_method
                  "{" delivery_params "}"
                  "{" { hook_block } "}" ;
delivery_method = "projectile" | "chain" | "zone"
                | "channel"    | "tether" | "trap" ;
delivery_params = [ param_entry { "," param_entry } ] ;
param_entry     = IDENT [ ":" param_value ]
                | IDENT ;
param_value     = number | duration | STRING ;

hook_block      = hook_name "{" { effect_stmt } "}" ;
hook_name       = "on_hit" | "on_arrival" | "on_complete"
                | "on_tick" | "on_expire" ;

(* ============================================================ *)
(* Recast / Morph                                               *)
(* ============================================================ *)

recast_block    = "recast" INT "{" { body_item } "}" ;
morph_block     = "morph" "into" "{" ability_body "}" "for" duration ;

(* ============================================================ *)
(* Passive triggers                                             *)
(* ============================================================ *)

trigger_expr    = trigger_name [ "(" trigger_arg_list ")" ] ;
trigger_name    = "on_damage_dealt" | "on_damage_taken"
                | "on_kill" | "on_death" | "on_ability_used"
                | "on_ally_damaged" | "on_ally_killed"
                | "on_hp_below" | "on_hp_above"
                | "on_shield_broken" | "on_stun_expire"
                | "periodic" | "on_heal_received"
                | "on_status_applied" | "on_status_expired"
                | "on_resurrect" | "on_dodge" | "on_reflect"
                | "on_auto_attack" | "on_stack_reached"
                | "on_voxel_placed" | "on_voxel_harvested"
                | "on_voxel_transformed" | "on_voxel_reverted"
                | "on_structure_placed" ;

(* ============================================================ *)
(* Template                                                     *)
(* ============================================================ *)

template        = "template" IDENT "(" [ param_list ] ")" "{"
                    { effect_stmt | deliver_block }
                  "}" ;
param_list      = typed_param { "," typed_param } ;
typed_param     = IDENT [ ":" type_name [ "=" default_val ] ] ;
type_name       = "int" | "float" | "bool" | "Material" | "Structure" ;
default_val     = number | bool_lit | STRING | material_ref | structure_ref ;
bool_lit        = "true" | "false" ;
template_call   = IDENT "(" [ arg_list ] ")" ;
arg_list        = template_arg { "," template_arg } ;
template_arg    = number | duration | percent | STRING
                | TAG_SYMBOL | material_ref | structure_ref | bool_lit
                | arith_expr ;

arith_expr      = arith_atom { arith_op arith_atom } ;
arith_op        = "+" | "-" | "*" | "/" | "%" ;
arith_atom      = number | PARAM_REF | "(" arith_expr ")"
                | builtin_call ;
builtin_call    = ("min" | "max" | "clamp" | "abs")
                  "(" arith_expr { "," arith_expr } ")" ;

(* ============================================================ *)
(* Structure (voxel blueprint)                                  *)
(* ============================================================ *)

structure       = "structure" IDENT
                    [ "(" param_list ")" ]
                    "{" structure_body "}" ;
structure_body  = { structure_prop ("," | NL) }
                  { voxel_stmt } ;
structure_prop  = "bounds"   ":" shape_expr
                | "origin"   ":" offset
                | "material" ":" material_ref
                | "rotatable"
                | "symmetry" ":" symmetry_kind ;
symmetry_kind   = "none" | "x" | "y" | "xy" | "radial" "(" INT ")" ;

voxel_stmt      = place_stmt | harvest_stmt | transform_stmt
                | compose_stmt | if_stmt ;
place_stmt      = "place" material_or_ref "in" shape_expr
                  [ "at" offset ] [ "rotate" INT ] ;
harvest_stmt    = "harvest" "in" shape_expr [ "at" offset ] ;
transform_stmt  = "transform" material_or_ref "->" material_or_ref
                  "in" shape_expr [ "at" offset ] ;
compose_stmt    = "include" structure_ref
                  [ "(" arg_list ")" ]
                  [ "at" offset ] [ "rotate" INT ]
                  [ "scale" INT ] ;
if_stmt         = "if" cond_expr "{" { voxel_stmt } "}"
                  [ "else" "{" { voxel_stmt } "}" ] ;

offset          = "(" arith_expr "," arith_expr "," arith_expr ")" ;
material_ref    = IDENT ;
material_or_ref = material_ref | PARAM_REF ;
structure_ref   = IDENT ;
agent_ref       = IDENT ;   (* keywords: caster, target, self, other *)
stat_ref        = IDENT ;
cond_ident      = IDENT ;
cond_value      = number | STRING | material_ref | TAG_SYMBOL ;

number          = INT | FLOAT ;
duration        = DURATION ;
percent         = PERCENT ;
```

### 3.1 Ambiguity resolution

- `effect_verb` is an open `IDENT` at grammar level; the lowerer validates
  against the catalog in §8. Unknown verbs → lower-error (status `reserved`).
- `area_mod` vs `nested_block`: `nested_block` only applies to verbs that
  opt in (`on_hit_buff`, etc., catalogued in §8).
- `template_call` vs `effect_verb`: one-pass resolution — if `IDENT` matches
  a defined `template`, it lowers via substitution; otherwise it's an effect.
- `IDENT` in an `arith_expr` context (inside templates / structures) is
  always a `PARAM_REF` in practice; compiler rejects bare `IDENT` in
  arithmetic slots outside parameter scope.

---

## §4 File structure & top-level forms

### 4.1 Files

Ability files live in `dataset/hero_templates/<hero>.ability` (convention).
Any `.ability` file under the asset roots is compiled together.
Per-file contents: any mix of `ability`, `passive`, `template`, `structure`
blocks.

### 4.2 Hero TOML binding

Hero TOML grows one optional field:

```toml
[hero]
name = "Mage"
# ...stats...
abilities_file = "mage.ability"
```

Resolution order:
1. If `abilities_file` is present, parse that `.ability` file.
2. Else, fall back to inline `[[abilities]]` / `[[passives]]` arrays (legacy).
3. Both present → parse error.

### 4.3 Compilation unit

All compiled `.ability` files are processed together to build one
`AbilityRegistry`. Name collisions:

- Two `ability` blocks with the same name across any files → hard error.
- `template` and `structure` names share a namespace with abilities.
- `material` names live in the `.sim` catalog (§17), do not collide with
  ability / template / structure namespaces.

### 4.4 Ability slots

Each hero references at most `MAX_ABILITIES = 8` ability names.
Passives do not consume ability slots. Exceeding the slot budget is a
lower-error at registry freeze time.

### 4.5 Registry wiring

The `AbilityRegistry` is frozen before tick 0 and holds
`AbilityId → AbilityProgram`. `AbilityId` is assigned in compile order
for determinism. The schema hash covers:

- `AbilityProgram` shape (delivery, area, gate, effects, hint, tags).
- The set of compiled ability, template, and structure names in order.
- Each structure's rasterized 3D voxel mask (after CSG folding).
- Each voxel mask registry entry used by `EffectOp::PlaceVoxels` et al.
- The material catalog (names + property values).

---

## §5 The `ability` block

### 5.0 Overview

**What it is.** An `ability` declares one active ability: a unit-castable
action with targeting, cooldown, cast time, optional delivery, and
effects.

**Grammar.**
```
ability = "ability" IDENT "{" ability_body "}" ;
ability_body = { header_prop ("," | NL) } { body_item } ;
body_item = effect_stmt | deliver_block | recast_block | morph_block ;
```

**Semantics.**
- Exactly one `AbilityProgram` per `ability`, stored in the registry at
  a stable `AbilityId`.
- Header properties set `Gate` fields and metadata slots.
- Body items compose the `effects` smallvec (`MAX_EFFECTS_PER_PROGRAM = 4`)
  and `delivery` / `area`.
- Header property order is free; each property appears at most once.
- A body holds **either** a `deliver` block **or** one-or-more bare
  `effect_stmt`s, not both. `recast` and `morph` combine with either.
- Ability names are globally unique across the compilation.

**Status.** `runs-today` for `ability` block parsing. Individual header
properties and body items carry their own status (below). The `passive`
block (§6) is `planned` — see §23.1 for the full status matrix.

**Errors.**
- Duplicate header → `duplicate header 'cooldown'`.
- `deliver` + bare effects → `ability 'X' mixes deliver{} and top-level effects`.
- Effects overflow → `ability 'X' exceeds MAX_EFFECTS_PER_PROGRAM (4)`.
- Duplicate name → `duplicate ability name 'X'`.

### 5.1 `target:` — targeting mode

Decides who/where the cast aims at and what the cast packet carries.

| Mode | Payload | Mask predicate | Status |
|---|---|---|---|
| `enemy` | `AgentId` | alive, hostile, in range | `runs-today` |
| `ally` | `AgentId` | alive, friendly, in range | `planned` |
| `self` | caster | always legal | `runs-today` |
| `self_aoe` | caster position | always legal | `planned` |
| `ground` | `Vec3` | in range, reachable | `planned` |
| `direction` | unit `Vec2` | always legal | `planned` |
| `vector` | `Vec3` offset | always legal | `reserved` |
| `global` | none | always legal | `reserved` |

**IR mapping.** `enemy` / `self` → `Area::SingleTarget { range }`, with
`gate.hostile_only = (mode == enemy)`. Others need new `Area` variants.

### 5.2 `range:` — maximum cast distance

Horizontal range cap (metres) from caster to target. Checked by
`mask::inferred_cast_target`. For `self`, lowered as `0.0`. Negative →
parse error.

**Status.** `runs-today`.

### 5.3 `cooldown:` — recharge

Duration between casts. Enforced via `mask Cast(ability)`.
`gate.cooldown_ticks = ceil(dur_ms / 100)` at the default 10 Hz tick
rate. Bare integer is treated as ms.

**Status.** `runs-today`.

### 5.4 `cast:` — wind-up

Duration between cast-start and resolution. `0ms` = instant (today's
only behavior). Non-zero introduces a cast-in-progress state consumed
by `unstoppable` and by `on_cast_interrupted` (future passive).

**Status.** `planned`. Requires `cast_ticks: u32` on `AbilityProgram`.

### 5.5 `hint:` — AI category

Coarse category consumed by scoring DSL (`ability::hint`) and ML.
Lowers to `AbilityProgram.hint = Some(AbilityHint::*)`. The enum today
has **six** variants: `Damage`, `Defense`, `CrowdControl`, `Utility`,
`Heal` (added per #142), and `Buff` (added per #142 as a distinct
duration-bound stat-boost category). `Economic` is a parser-AST hint
that is rejected at lower time (`HintReserved`) and remains
`reserved`.

> **Spec audit (2026-05-08) — STATUS-FLIP:** `heal` is `runs-today`
> (engine `AbilityHint::Heal = 4`); the previous "lowers to `Utility`
> at a documented loss" claim is stale. `buff` is **UNDOCUMENTED** in
> this spec — it is a real engine variant (`AbilityHint::Buff = 5`)
> that scoring already consumes; the lowering is `runs-today`.

**Status.** `runs-today` for `damage` / `defense` / `crowd_control` /
`utility` / `heal` / `buff`; `reserved` for `economic`.

### 5.6 `cost:` — resource cost

Integer resource pool cost, deducted at cast-start. **Status:** `reserved`
— no IR field today.

### 5.7 `zone_tag:` — combo anchor

Intern-table STRING tag for zone-reaction combos.
**Status:** `reserved`.

### 5.8 `charges:` / `recharge:` — ammo

Independent refill timer, separate from `cooldown`. Requires
`charges_max: u8`, `recharge_ticks: u32` on `AbilityProgram`.
**Status:** `planned`.

### 5.9 `toggle` / `toggle_cost:` — sustained

Bare flag + per-tick drain. First cast = on, second = off.
**Status:** `reserved`.

### 5.10 `unstoppable` — CC immunity during cast

Bare flag. While `cast` is in progress, caster ignores newly-applied CC.
**Status:** `planned`.

### 5.11 `form:` / `swap_form:` — form groups

Per-caster form tag + optional swap directive. Reserves a per-agent
form slot. **Status:** `reserved`.

### 5.12 `recast N { … }` — multi-stage

Sub-block redefining the `N`th cast within `recast_window`. Requires
`recast:` + `recast_window:` on the parent. **Status:** `reserved`.

### 5.13 `morph into { … } for <dur>` — temporary replacement

Replaces the caster's own slot for this ability for `<dur>` (0ms =
permanent until recast). Requires per-agent ability-slot override.
**Status:** `reserved`.

---

## §6 The `passive` block

**What it is.** A passive declares a trigger + effects that fire when
the trigger condition is met on the owning agent.

**Grammar.**
```
passive = "passive" IDENT "{" passive_body "}" ;
passive_body = trigger_decl { header_prop } { effect_stmt } ;
trigger_decl = "trigger" ":" trigger_expr ;
```

**Semantics.**
- Exactly one trigger per passive; compound triggers use `and`/`or` over
  condition atoms inside the passive body's `when` modifier.
- `cooldown:` and `range:` are the only ability-style headers supported.
- Effects body follows the same rules as `ability_body`'s effect body
  (no `deliver`, no `recast`, no `morph`).

### 6.1 Trigger catalog

All triggers support optional modifiers in the same parens: `range:`,
`by:` (caster kind: `enemy`/`ally`/`self`), and trigger-specific filters.

| Trigger | Arguments | Semantics | Status |
|---|---|---|---|
| `on_damage_dealt` | `[range:, by:]` | After any `EffectDamageApplied` event | `planned` |
| `on_damage_taken` | `[range:, by:]` | After damage lands on owner | `planned` |
| `on_kill` | `[range:]` | After kill event attributed to owner | `planned` |
| `on_death` | — | Owner dies | `planned` |
| `on_ability_used` | `[ability:]` | Any ability cast by owner | `planned` |
| `on_ally_damaged` | `range:` required | Ally within range took damage | `planned` |
| `on_ally_killed` | `range:` required | Ally within range died | `planned` |
| `on_hp_below` | `<percent>` | Owner HP crossed threshold | `planned` |
| `on_hp_above` | `<percent>` | Owner HP crossed threshold | `planned` |
| `on_shield_broken` | — | Shield reduced to 0 | `planned` |
| `on_stun_expire` | — | Owner exits stun | `planned` |
| `periodic` | `<dur>` | Every `dur` while passive is active | `planned` |
| `on_heal_received` | `[range:]` | Any `EffectHealApplied` targeting owner | `planned` |
| `on_status_applied` | `[status:]` | Any status effect lands | `planned` |
| `on_status_expired` | `[status:]` | Any status expires | `planned` |
| `on_resurrect` | — | Reserved for resurrection cascade | `reserved` |
| `on_dodge` | — | Dodge event (reserved cascade) | `reserved` |
| `on_reflect` | — | Reflect event | `reserved` |
| `on_auto_attack` | `[on_hit:]` | Owner auto-attacks | `planned` |
| `on_stack_reached` | `<stack_name> <n>` | Stack count crossed threshold | `planned` |
| `on_voxel_placed` | `[range:, by:, material:]` | `EffectVoxelsPlaced` nearby | `reserved` |
| `on_voxel_harvested` | `[range:, by:, material:]` | `EffectVoxelsHarvested` nearby | `reserved` |
| `on_voxel_transformed` | `[from:, to:, range:]` | `EffectVoxelsTransformed` nearby | `reserved` |
| `on_voxel_reverted` | `[range:]` | Lifetime-scheduled revert fires | `reserved` |
| `on_structure_placed` | `[structure:, range:]` | `EffectStructurePlaced` nearby | `reserved` |

---

## §7 Effect statements

The one-liner grammar. The canonical compact form for an effect atom.

**Grammar.** See §3 `effect_stmt`. Every field after `effect_verb
effect_args` is optional and independently useful.

**Modifier precedence (left to right in source):**
1. `effect_verb effect_args` — mandatory.
2. `in <shape_expr>` — scope of the effect.
3. `[TAG: value, …]` — power ratings per tag.
4. `for <dur>` — effect duration (shield, buff, slow, etc.).
5. `when <cond> [else <eff>]` — conditional gate.
6. `chance <p>` — probabilistic gate (0..=1).
7. `stacking <mode>` — see §12.
8. `+ N% <stat>` — scaling terms (see §13).
9. `until_caster_dies` / `damageable_hp N` — lifetime modifiers (voxel only).
10. `{ … }` — nested effects (where verb opts in).

**Evaluation order at runtime:**
1. `when` + `chance` evaluated first; failing either → no effect.
2. Scaling resolved (stat lookups against caster/target at cast-resolve).
3. Shape rasterized (compile-time for static shapes; see §9).
4. Tags aggregated into `AbilityProgram.tags` (compile-time, §14).
5. Effect applied.
6. `stacking` decided by the cascade handler for the affected target.

**Example (parsed and lowered in isolation):**
```
damage 50 + 8% target_max_hp in circle(3.0) [FIRE: 60] when target_hp_below(30%) chance 0.8 stacking refresh
```
Lowers to:
```
EffectOp::Damage { amount: compute(50 + 0.08 * target.max_hp) }
Area::SingleTarget { range } is extended by the shape modifier
(future: Area::Circle { radius: 3.0 } for self-centered casts;
 Area::CircleAt { center, radius } for ground-targeted)
Stacking mode 'refresh' → handler uses existing behavior (max-wins)
Tags contribute (Fire, 60) to AbilityProgram.tags
Gate: when cast-resolve sees target hp_pct >= 0.30, effect is skipped
Probabilistic: RNG gate with 0.8 probability via next_rand_u32
```

---

## §8 Effect catalog

**Conventions.** Each entry lists: `verb <args>`, surface semantics, IR
mapping (which `EffectOp` and field), status, notes. `α` denotes
arity-positional arguments; named modifiers from §7 are universal and
omitted.

### 8.1 Combat core

| Verb | Args | IR mapping | Status |
|---|---|---|---|
| `damage` | amount:f32 | `EffectOp::Damage { amount }` | `runs-today` |
| `heal` | amount:f32 | `EffectOp::Heal { amount }` | `runs-today` |
| `shield` | amount:f32 | `EffectOp::Shield { amount }` | `runs-today` |
| `stun` | duration | `EffectOp::Stun { duration_ticks }` | `runs-today` |
| `slow` | factor:f32, duration | `EffectOp::Slow { duration_ticks, factor_q8 }` | `runs-today` |

### 8.2 Control

| Verb | Args | IR mapping | Status |
|---|---|---|---|
| `root` | duration | `EffectOp::Root` (variant 8) | `runs-today` |
| `silence` | duration | `EffectOp::Silence` (variant 9) | `runs-today` |
| `fear` | duration | `EffectOp::Fear` (variant 10) | `runs-today` |
| `taunt` | duration | `EffectOp::Taunt` (variant 11) | `runs-today` |
| `charm` | duration | `EffectOp::Charm` (variant 28) | `runs-today` |
| `polymorph` | duration | `EffectOp::Polymorph` + shape override | `reserved` |
| `banish` | duration | `EffectOp::Banish` (target removed from sim) | `reserved` |
| `confuse` | duration | intent-randomization modifier | `reserved` |
| `suppress` | duration | `EffectOp::Suppress` (variant 30) | `runs-today` |
| `grounded` | duration | `EffectOp::Grounded` (variant 29) | `runs-today` |

> **Spec audit (2026-05-08) — STATUS-FLIP:** `root` / `silence` / `fear`
> / `taunt` / `charm` / `grounded` / `suppress` lower today (Wave 2
> pieces 1, 8). Apply-handler runtime gating for some of these is still
> deferred — IR variant + DSL lowering are shipped; the cascade-fold
> consumer rules land per fixture. The lower-time test is the source of
> truth (the IR ordinal is pinned).

### 8.3 Movement

| Verb | Args | IR mapping | Status |
|---|---|---|---|
| `dash` | distance:f32 | `EffectOp::Dash { distance }` (variant 12) | `runs-today` |
| `dash to_target` | — | `Dash { distance: f32::INFINITY }` (sentinel) | `runs-today` |
| `dash to_position` | — | `Dash { to: CursorPos }` | `reserved` |
| `blink` | distance:f32 | `EffectOp::Blink` (variant 13) | `runs-today` |
| `knockback` | distance:f32 | `EffectOp::Knockback` (variant 14) | `runs-today` |
| `pull` | distance:f32 | `EffectOp::Pull` (variant 15) | `runs-today` |
| `swap` | — | caster ↔ target positions | `reserved` |
| `travel_to` | dest_x:f32, dest_y:f32, [dest_z:f32], `for` duration | `EffectOp::TravelTo { dest_x_q8, dest_y_q8, eta_ticks }` (variant 39) | `runs-today` |

> **Spec audit (2026-05-08) — STATUS-FLIP:** `dash` / `blink` /
> `knockback` / `pull` lower today (Wave 2 piece 2). The `dash to_target`
> alias encodes "travel until the cast's target is reached" via a
> `f32::INFINITY` sentinel in the same single-f32 payload — the spec
> previously described it as a separate IR shape.
>
> **UNDOCUMENTED:** `travel_to` is a Lift A multi-tick movement verb
> (`7bb0929c`). Self-cast — caster initiates a multi-tick walk to a 2D
> destination over `eta_ticks` (with an optional ignored Z arg for
> forward compatibility). Destination packs as q8 (`256 = 1.0`); range
> ±127.99 cells. Distinct from `blink` (instant teleport) and `dash`
> (single-tick travel along facing).

### 8.4 Buffs / Debuffs

| Verb | Args | IR mapping | Status |
|---|---|---|---|
| `buff` | stat:id, magnitude:f32, `for` duration | `EffectOp::Buff { stat, magnitude_q8, duration_ticks }` (variant 23) | `runs-today` |
| `debuff` | stat:id, delta:f32 | `EffectOp::Buff` (negative magnitude) | `planned` |
| `damage_modify` | factor:f32, duration | `EffectOp::DamageModify { duration_ticks, multiplier_q8 }` (variant 19) | `runs-today` |
| `reflect` | fraction:f32, `for` duration | `EffectOp::Reflect { duration_ticks, fraction_q8 }` (variant 31) | `runs-today` |
| `lifesteal` | fraction:f32, duration | `EffectOp::LifeSteal { duration_ticks, fraction_q8 }` (variant 18) | `runs-today` |
| `blind` | factor:f32 | hit-chance reduction | `reserved` |

> **Spec audit (2026-05-08) — STATUS-FLIP:** `buff` / `damage_modify` /
> `lifesteal` / `reflect` all lower today. `buff`'s stat vocabulary is
> currently two values (`move_speed`, `attack_speed`); the `BuffStat`
> enum (`crates/engine/src/ability/program.rs`) extends as more apply
> paths land. Magnitudes pack q8 (`1.0 → 256`).

### 8.5 Advanced / narrative

| Verb | Args | IR mapping | Status |
|---|---|---|---|
| `summon` | name:STRING, [count:INT], [`for` duration] | `EffectOp::Summon { template_hash, count, lifetime_ticks }` (variant 24) | `runs-today` (lower-only — apply handler deferred) |
| `summon clone` | — | clone caster | `reserved` |
| `stealth` | `for` duration, [`break_on_damage`] | `EffectOp::Stealth { duration_ticks }` (variant 27) | `runs-today` |
| `leash` | distance:f32 | enforce max distance to target | `reserved` |
| `link` | fraction:f32 | share-damage link | `reserved` |
| `redirect` | duration, `charges` n:INT | intercept incoming | `reserved` |
| `rewind` | duration | snapshot-restore | `reserved` |
| `cooldown_modify` | delta, [name] | cooldown delta (all or one) | `reserved` |
| `apply_stacks` | name:STRING, n, `max`, duration | stack application | `reserved` |
| `execute` | hp_threshold:f32 | `EffectOp::Execute { hp_threshold }` (variant 16) | `runs-today` |
| `self_damage` | amount:f32 | `EffectOp::SelfDamage { amount }` (variant 17) | `runs-today` |
| `dispel` | tag_list | remove status matching tags | `reserved` |
| `immunity` | tag_list, duration | grant status-family immunity | `reserved` |
| `death_mark` | duration, damage_pct | on-expire damage | `reserved` |
| `resurrect` | hp_pct:PERCENT | revive target | `reserved` |
| `overheal_shield` | duration | convert overheal to shield | `reserved` |
| `absorb_to_heal` | amount, duration, ratio | damage → heal conversion | `reserved` |
| `shield_steal` | amount:f32 | transfer shield | `reserved` |
| `status_clone` | `max` n:INT | copy status stacks | `reserved` |
| `detonate` | damage_multiplier:f32 | detonate zone-tagged effects | `reserved` |
| `status_transfer steal` | — | steal target's buffs | `reserved` |
| `on_hit_buff` | duration, `{ <eff>… }` | buffs on next-hit | `reserved` |
| `obstacle` | width, height | temporary barrier | `reserved` |
| `projectile_block` | duration | intercept projectiles | `reserved` |
| `attach` | duration | attach to target (AoE origin) | `reserved` |
| `evolve_ability` | ability_index:INT | runtime ability swap | `reserved` |
| `command_summons` | property, value | direct summoned units | `reserved` |

### 8.6 World — voxel ops (see §18)

| Verb | Args | IR mapping | Status |
|---|---|---|---|
| `place_voxels` | material, `in` shape | `EffectOp::PlaceVoxels` | `planned` |
| `harvest_voxels` | `in` shape, `drop`? | `EffectOp::HarvestVoxels` | `planned` |
| `transform_voxels` | from→to, `in` shape | `EffectOp::TransformVoxels` | `planned` |
| `place_structure` | name(args?) | `EffectOp::PlaceStructure` | `planned` |

### 8.7 World — wealth & economy

See the economic depth spec (`spec/economy.md`) for the
full economic verb catalog, contestation semantics, and reactive-passive
machinery. Verbs cross-listed here for completeness.

| Verb | Args | IR mapping | Status |
|---|---|---|---|
| `transfer_gold` | amount:i32 | `EffectOp::TransferGold` (variant 5) | `runs-today` |
| `modify_standing` | delta:i16 | `EffectOp::ModifyStanding { delta }` (variant 6, legacy single-i16 shape) | `runs-today` |
| `cast_recipe` | recipe_id:u16, [`target` tool_slot:u8] | `EffectOp::Recipe { recipe_id, target_tool }` (variant **40**) | `runs-today` |
| `wear_tool` | tool_kind:u8, amount:u16 (q8) | `EffectOp::WearTool { tool_kind, amount }` (variant **41**) | `runs-today` |
| `propose` | contract_kind:u8, [`expires_at` tick:u32] | `EffectOp::Propose { contract_kind, expires_at_tick }` (variant **42**) | `runs-today` |
| `announce` | announcement_kind:u8, `radius` cells:f32 (q8) | `EffectOp::Announce { announcement_kind, radius_q8 }` (variant **43**) | `runs-today` |
| `gain_skill` | skill_id:u8, amount:u16 (q8) | `EffectOp::GainSkill { skill_id, amount_q8 }` (variant **44**) | `runs-today` |
| `create_obligation` | obligation_id:u16, kind:u8 | `EffectOp::CreateObligation { obligation_id, kind }` (variant **45**) | `runs-today` |
| `transfer_property` | property_id, target | `EffectOp::TransferProperty` | `reserved` |
| `pickpocket` / `demand` | target, item, contest_kind | `EffectOp::ForcibleTransfer` | `reserved` |
| `discharge_obligation` | obligation_id | `EffectOp::DischargeObligation` | `reserved` |
| `establish_route` | from, to | `EffectOp::EstablishRoute` | `reserved` |
| `join_caravan` | caravan: GroupId | `EffectOp::JoinCaravan` | `reserved` |
| `transfer_obligation` | obligation_id, target | `EffectOp::TransferObligation` | `reserved` |

> **Spec audit (2026-05-08) — STATUS-FLIP + ordinal correction:** Lift B
> (`cast_recipe` / `wear_tool`), Lift C (`propose` / `announce`), and
> Lift D (`gain_skill` / `create_obligation`) all lower today. The
> previous economy table claimed ordinals 17–26; the actual shipped
> ordinals are **40–45** (engine `EffectOp` 17–22 are
> `SelfDamage`/`LifeSteal`/`DamageModify`/`DoT`/`HoT`/`TimedShield`;
> engine ordinals 23–27 are `Buff`/`Summon`/`Harvest`/`PlaceVoxel`/
> `Stealth`). The `transfer_property` / `pickpocket` /
> `discharge_obligation` / `establish_route` / `join_caravan` /
> `transfer_obligation` verbs are still `reserved` (no engine variant,
> no DSL arm). See §8.11 for the full IR-vs-spec ordinal reconciliation.

### 8.8 AI-state manipulation (see §20)

All verbs in §20 H/J/K/L/M/N appear in this catalog; exact IR mapping
and status listed in their section.

### 8.9 Meta

| Verb | Args | IR mapping | Status |
|---|---|---|---|
| `cast` | ability_name | `EffectOp::CastAbility { ability, selector }` | `runs-today` |

Nested `cast` participates in `MAX_CASCADE_ITERATIONS = 8` depth budget.

### 8.10 Opt-in nested blocks

Only these verbs accept a trailing `{ <effect_stmt>… }` block:
- `on_hit_buff` — the nested effects fire on the caster's next attack.
- future: `on_tick_of`, `on_hit`, `on_arrival` at effect-statement scope
  (today they live inside `deliver` hook blocks; §10).

Other verbs with a trailing `{ … }` → parse error.

### 8.11 Spec-audit verb additions (2026-05-08)

The following verbs are shipped in the `crates/dsl_compiler` ability
lowerer + `crates/engine::EffectOp` enum but were never (or only
partially) catalogued in §8.1–8.9. Listed here for completeness; the
section structure above stays unchanged.

#### 8.11.1 Multi-tick / over-time variants (UNDOCUMENTED)

These are `for`-modifier swaps on already-catalogued verbs, but the
modifier flips them to a different `EffectOp` variant — they are not
just lifetime-tagged versions of the base op.

| Surface | IR mapping | Variant | Status |
|---|---|---|---|
| `damage <amount> for <dur>` | `EffectOp::DamageOverTime { amount, duration_ticks }` | 20 | `runs-today` |
| `heal <amount> for <dur>` | `EffectOp::HealOverTime { amount, duration_ticks }` | 21 | `runs-today` |
| `shield <amount> for <dur>` | `EffectOp::TimedShield { amount, duration_ticks }` | 22 | `runs-today` |

The base `Damage` / `Heal` / `Shield` ops (variants 0–2) carry no
duration. Apply-side scheduler that re-emits per-tick events is
deferred — IR + lowering ship today.

#### 8.11.2 World primitives — `harvest` / `mine` / `place_voxel` (UNDOCUMENTED)

Bridge verbs into the existing non-combat event surface
(`AgentHarvested = 9`, `AgentHarvestedVoxel = 12`,
`AgentPlacedVoxel = 11`).

| Surface | IR mapping | Variant | Status |
|---|---|---|---|
| `harvest "<kind>" [amount]` | `EffectOp::Harvest { kind_hash, amount }` | 25 | `runs-today` |
| `mine "<kind>" [amount]` | alias for `harvest` (lowers identically) | 25 | `runs-today` |
| `place_voxel "<kind>"` | `EffectOp::PlaceVoxel { kind_hash }` | 26 | `runs-today` |

`kind_hash` is FxHash of the resource ident — apply handlers map the
hash to a concrete resource via the runtime registry. Voxel resources
emit `AgentHarvestedVoxel`; surface resources emit `AgentHarvested`. The
spec's §18 voxel-shape verbs (`place_voxels` / `harvest_voxels` /
`transform_voxels` / `place_structure`) remain `planned` — `place_voxel`
(singular) is a separate single-cell primitive, not the area-stamp
verb.

#### 8.11.3 Theory-of-Mind verbs (UNDOCUMENTED — Wave 3)

Seven ToM verbs shipped between Wave 3 Phase 1 and Phase 4. The §20.2
table catalogs the design surface; this table reflects the actual
shipped IR ordinals.

| Surface | IR mapping | Variant | Chronicle event | Status |
|---|---|---|---|---|
| `plant_belief <subject_idx> bit <fact_bit>` | `EffectOp::PlantBelief { subject_idx, fact_bit }` | 32 | `EffectPlantBeliefApplied = 63` | `runs-today` |
| `observe <id>` | `EffectOp::Observe { target_observer }` | 33 | `EffectObserveApplied = 64` | `runs-today` |
| `scry <observer> <subject>` | `EffectOp::Scry { target_observer, subject_idx }` | 34 | `EffectScryApplied = 65` | `runs-today` |
| `reveal <subject_idx>` | `EffectOp::Reveal { subject_idx }` | 35 | `EffectRevealApplied = 66` | `runs-today` |
| `disguise <fake_type> for <dur>` | `EffectOp::Disguise { fake_type, duration_ticks }` | 36 | `EffectDisguiseApplied = 67` | `runs-today` |
| `decoy <subject_idx> <fake_pos>` | `EffectOp::Decoy { subject_idx, fake_pos }` | 37 | `EffectDecoyApplied = 68` | `runs-today` |
| `erase_belief <subject_idx> <fields>` | `EffectOp::EraseBelief { subject_idx, fields }` | 38 | `EffectEraseBeliefApplied = 69` | `runs-today` |

All seven are bit-flag / payload-only chronicle writes; the BeliefState
SoA mutation lives in compiler-emitted `physics @phase(post)` consumer
rules in `tom_probe.sim`. Cast shape:
- `plant_belief` / `observe` / `scry` / `reveal` / `decoy` /
  `erase_belief` — target the `cast.target` agent (writes into target's
  belief row about `subject_idx`).
- `disguise` — self-cast (caster declares fake type).

> **Spec audit (2026-05-08):** §20.2 documents these under design
> identifiers (`AgentSel`, `SubjectSel`, `BeliefFieldMask`, `PlantBelief
> { observer_sel, subject_sel, fields }`). The actual shipped shapes
> are simpler — the IR carries a `subject_idx: u32` directly (no
> selector enum) and the consumer rules in `.sim` resolve the
> observer = caster, target = cast.target convention. The richer
> selector model is `reserved` — current spy_network tests only need
> the simpler shape.

#### 8.11.4 Production / consent / knowledge verbs (Lifts A–D)

Catalogued in §8.3 (movement: `travel_to`) and §8.7 (economic surface:
`cast_recipe`/`wear_tool`/`propose`/`announce`/`gain_skill`/
`create_obligation`). Variants 39–45; chronicle events 70–76. Full
ordinal reconciliation in §22.2.1 / §22.4.1.

#### 8.11.5 P4 (≤16-byte EffectOp) compliance

Per project constitution P4, every `EffectOp` variant must fit ≤16
bytes including the discriminant tag. The Wave 2+ / Wave 3 / Lift A–D
additions all comply (the in-source comments in
`crates/engine/src/ability/program.rs` annotate each variant with its
payload+tag size). Tightest end of the range:

| Variant | Payload | Tag | Total |
|---|---|---|---|
| `Observe { target_observer: u8 }` (33) | 1B | 1B | **2B** |
| `Recipe { recipe_id: u16, target_tool: u8 }` (40) | 3B | 1B | **4B** |
| `WearTool { tool_kind: u8, amount: u16 }` (41) | 3B | 1B | **4B** |
| `Announce { announcement_kind: u8, radius_q8: u16 }` (43) | 3B | 1B | **4B** |
| `GainSkill { skill_id: u8, amount_q8: u16 }` (44) | 3B | 1B | **4B** |
| `CreateObligation { obligation_id: u16, kind: u8 }` (45) | 3B | 1B | **4B** |
| `PlantBelief { subject_idx: u32, fact_bit: u8 }` (32) | 5B | 1B | **6B** |
| `Disguise { fake_type: u8, duration_ticks: u32 }` (36) | 5B | 1B | **6B** |
| `EraseBelief { subject_idx: u32, fields: u8 }` (38) | 5B | 1B | **6B** |
| `Scry { target_observer: u8, subject_idx: u32 }` (34) | 5B | 1B | **6B** |
| `Reveal { subject_idx: u32 }` (35) | 4B | 1B | **5B** |
| `Propose { contract_kind: u8, expires_at_tick: u32 }` (42) | 5B | 1B | **6B** |
| `Decoy { subject_idx: u32, fake_pos: u32 }` (37) | 8B | 1B | **9B** |
| `TravelTo { dest_x_q8: i16, dest_y_q8: i16, eta_ticks: u32 }` (39) | 8B | 1B | **9B** |
| `Summon { template_hash: u32, count: u8, lifetime_ticks: u32 }` (24) | 9B | 1B | **10B** |

All shipped variants (0–45) sit at ≤10B with tag — well below the 16B
ceiling. The `Reflect` / `Buff` family (signed q8 magnitudes packed in
i16) is the widest at 12B with tag.

---

## §9 Shapes (unified areas & volumes)

All shapes are 3D primitives. The "2D" family (`circle`, `cone`, `line`,
`ring`, `spread`) are 3D primitives with a default disc thickness; the
"3D" family (`box`, `sphere`, `column`, `wall`, `cylinder`, `dome`,
`hull`) are inherently 3D. Any verb may use any shape.

### 9.1 Disc family (default thickness = 1 voxel above reference)

| Shape | Grammar | Semantics | Status |
|---|---|---|---|
| `circle` | `circle(radius)` | Disc of radius `r` on XZ plane | `planned` |
| `cone` | `cone(radius, angle_deg)` | Disc arc of `angle` degrees | `planned` |
| `line` | `line(length, width)` | Rectangular strip from origin | `planned` |
| `ring` | `ring(inner, outer)` | Annulus | `planned` |
| `spread` | `spread(radius, max_targets)` | Disc with per-target cap | `reserved` |

Thickness override: `circle(3.0) thickness 2` → 2-voxel-thick disc.

### 9.2 3D volume family

| Shape | Grammar | Semantics | Status |
|---|---|---|---|
| `box` | `box(wx, wy, wz)` | Axis-aligned box | `planned` |
| `sphere` | `sphere(r)` | Sphere | `planned` |
| `column` | `column(r, h)` | Upright cylinder | `planned` |
| `wall` | `wall(len, h, thick) [facing: deg]` | Oriented slab | `planned` |
| `cylinder` | `cylinder(r, h)` | Oriented cylinder | `planned` |
| `dome` | `dome(r)` | Hemisphere | `planned` |
| `hull` | `hull(r)` | Castle-footprint (cuboid w/ beveled corners) | `planned` |

**`hull(r)` predicate (pinned 2026-05-08, Task #231):** intersection of an
axis-aligned cube of half-extent `r` with a sphere of radius `r·√2`. A
candidate is in-hull when **both** gates hold:

1. `|dx| ≤ r ∧ |dy| ≤ r ∧ |dz| ≤ r`  (cube of half-extent `r`)
2. `dx² + dy² + dz² ≤ 2·r²`           (bevel sphere — `r·√2` radius)

Geometric meaning: a cube with its 8 corners chamfered off. The face
centers (`dist = r`) are in, the edge midpoints (`dist = r·√2`) are on
the bevel boundary (inclusive), the corners (`dist = r·√3`) are clipped.
This produces a victim set strictly between `sphere(r)` (smaller — only
within `r`) and `box(r, r, r)` (larger — includes corners up to `r·√3`).
Was previously aliased to Sphere; both backends now implement the
distinct predicate (CPU oracle: `apply_program_aoe_hull_filter`; GPU:
`area_kind == 11u` branch in `wgsl_body.rs`).

### 9.3 Orientation

- Default: inferred from caster → cast direction. For `target: ground`,
  orientation is the caster→target vector projected horizontally.
- Explicit override via `facing: <degrees>` on `wall`, `cylinder`,
  `box`. Degree values are grid-snapped to `{0, 90, 180, 270}` per §9.5.

### 9.4 CSG composition

Shapes combine with `union`, `diff`, `intersect`. Parens group; standard
precedence is `intersect > diff > union`.

```
sphere(5) diff sphere(4)                          # thick shell
sphere(5) diff box(10, 3, 10)                     # dome with flat cut
(sphere(5) diff sphere(4)) union column(1, 8)    # shell + pillar
```

CSG is **compiled at lowering time** to a 3D bitmask stored in the voxel
mask registry. Runtime cost is O(voxels-in-bbox), not O(CSG-ops). Schema
hash covers the rasterized bitmask; edits invalidate prior replays.
**Status:** `planned`.

### 9.5 Grid snapping

- Shape origins snap to the voxel grid. Default cell size is `1.0m`;
  tunable via `config.sim::voxel_cell_size` without a schema-hash bump
  as long as rasterization is re-run against the same cell size.
- Rotations snap to `{0, 90, 180, 270}` degrees.
- Shape dimensions accept `FLOAT` at the grammar level; rasterization
  rounds to nearest cell with tie-break toward `+x`.

---

## §10 Delivery methods

Delivery blocks package effects into hooks that fire at stages of a
delivered cast.

**Grammar.** See §3 `deliver_block`.

### 10.1 Methods

| Method | Params | Hooks | IR mapping | Status |
|---|---|---|---|---|
| `projectile` | `speed`, `width`, `pierce`? | `on_hit`, `on_arrival` | `Delivery::Projectile` | `planned` |
| `chain` | `bounces`, `range`, `falloff` | `on_hit` | `Delivery::Chain` | `planned` |
| `zone` | `duration`, `tick` | `on_hit`, `on_tick`, `on_expire` | `Delivery::Zone` | `planned` |
| `channel` | `duration`, `tick` | `on_tick`, `on_expire` | `Delivery::Channel` | `planned` |
| `tether` | `max_range`, `tick` | `on_tick`, `on_complete` | `Delivery::Tether` | `reserved` |
| `trap` | `duration`, `trigger_radius`, `arm_time` | `on_trigger` | `Delivery::Trap` | `reserved` |

Today only `Delivery::Instant` ships. Every method above is a new variant
of the `Delivery` enum plus resolver code.

### 10.2 Hook semantics

- `on_hit` — fires once per impact. For `chain`, fires per bounce.
- `on_arrival` — fires once when the projectile reaches its max range
  without impact.
- `on_tick` — fires every `tick` duration inside a zone / channel /
  tether.
- `on_complete` — fires once when the tether expires cleanly.
- `on_expire` — fires when zone / channel timer reaches 0.
- `on_trigger` — trap armed, detonated.

Effects inside a hook obey all §7 modifiers and lower against the hook's
scope (per-hit, per-tick, etc.) with separate `cascade_id`s per invocation.

---

## §11 Conditions

The `when <cond_expr>` modifier (§7) gates an effect on a predicate.
`cond_expr` supports `and`/`or`/`not` as prefix forms.

### 11.1 Condition atom catalog

Atoms come from five families, expanded in dedicated sections:

- **Physical state** — `target_hp_below(X)`, `caster_hp_above(X)`,
  `target_is_stunned`, `target_is_rooted`, `target_is_shielded`, etc.
  Status mostly `runs-today` / `planned`.
- **Voxel-aware** — §19.
- **AI-state** — §21.
- **Tag-aware** — `target_has_tag(TAG)`, `caster_has_buff(name)`.
- **Temporal** — `tick_after(dur)`, `cast_within(dur)`.

### 11.2 Boolean composition

```
when and(target_hp_below(30%), caster_hp_above(50%))
when or(target_is_stunned, target_is_rooted)
when not(caster_has_buff("shielded"))
when and(target_hp_below(30%), not(target_has_tag(BOSS)))
```

Evaluation is short-circuiting left-to-right. Authors order expensive
atoms (e.g., Class-2 memory-ring scans from §21) last.

### 11.3 `else` branches

```
damage 50 when target_is_stunned else damage 25
```

If `when` fails, `else` fires. `else` chains a full `effect_stmt` (with
its own modifiers, including its own `when`).

### 11.4 `chance <p>`

Probabilistic gate, `0.0 <= p <= 1.0`. RNG via `next_rand_u32()` —
deterministic. Composable with `when`: both must pass.

---

## §12 Stacking modes

Apply to effects with a `for <dur>` modifier. Decide what happens when
the same effect from the same caster (or the same named stack group)
is re-applied to a target that already has it active.

| Mode | Semantics | Default for |
|---|---|---|
| `stack` | New application stacks independently (N separate timers) | `apply_stacks` verb |
| `refresh` | Existing timer resets; duration = new duration | Generic `for <dur>` buffs |
| `extend` | New duration adds to remaining | — |

The IR-level rule per effect is encoded on the relevant cascade
handler; see `assets/sim/physics.sim`. This spec only describes the
surface contract.

**Status.** `refresh` is today's default for existing `stun` / `slow`
handlers; `stack` and `extend` are `planned`.

---

## §13 Scaling terms

Scale an effect's amount by a percentage of a runtime stat.

**Grammar.** `+ <percent> <stat_ref> [consume] [cap <number>]`.
Chainable: `+ 10% ATK + 5% MAX_HP`.

### 13.1 Stat catalog

| Stat | Source | Status |
|---|---|---|
| `caster_attack_damage` | `Agent.atk` | `planned` |
| `caster_max_hp` | `Agent.max_hp` | `planned` |
| `caster_missing_hp` | `Agent.max_hp - Agent.hp` | `planned` |
| `caster_hp` | `Agent.hp` | `planned` |
| `target_max_hp` | Target agent | `planned` |
| `target_current_hp` | Target agent | `planned` |
| `target_missing_hp` | Target agent | `planned` |
| `target_shield` | Target shield_hp | `planned` |
| `caster_stacks("<name>")` | Stack group count | `reserved` |
| `target_stacks("<name>")` | Stack group count | `reserved` |

### 13.2 Modifiers

- `consume` — deducts stacks consumed during scaling (only meaningful
  with `caster_stacks` / `target_stacks`).
- `cap <number>` — caps the additive contribution from that term.

### 13.3 Evaluation

Resolved at cast-resolve time against current state. Scaling does not
re-evaluate per tick inside a delivered zone; the computed amount is
baked at resolve and re-used per hook fire.

---

## §14 Tags

Tags aggregate into per-ability power ratings used by scoring / ML.

**Grammar.** `[TAG_SYMBOL: number, …]` on an effect line.

Engine-side: `AbilityTag` enum (fixed 6: `Physical`, `Magical`,
`CrowdControl`, `Heal`, `Defense`, `Utility`) — packed as column index
into `PackedAbilityRegistry.tag_values`.

### 14.1 Tag resolution

- Author writes symbolic tags (`FIRE`, `ICE`, `MAGIC`, `LIGHTNING`).
- Lowering maps author tags onto engine `AbilityTag`s via a mapping
  table (see §17 analogy for materials). Unmapped author tags → lower
  warning (not fatal — they're stripped).
- Per-effect tag values are summed across the ability into
  `AbilityProgram.tags: SmallVec<[(AbilityTag, f32); 6]>`.

### 14.2 Open-set vs fixed-enum

Current engine: fixed 6-enum. Author-facing surface is open-set.
**Status of open-set:** `planned` — widening requires either growing
the engine enum (schema bump) or introducing a runtime tag symbol
table. Author tags not in the current mapping lower with a warning.

### 14.3 Example

```
damage 55 [FIRE: 60, MAGIC: 40]
stun 2s [CROWD_CONTROL: 80, ICE: 60]
```
Aggregated: `(Magical, 60+40=100)` if `FIRE`→`Magical`, `ICE`→`Magical`;
`(CrowdControl, 80)`.

---

## §15 Templates

**What it is.** Parameterized, pure-substitution reusable bodies for
effect sequences or delivery blocks. Sub-Turing: no recursion, no loops,
no runtime evaluation.

**Grammar.** See §3 `template`.

### 15.1 Parameter types

- `int`, `float`, `bool` — primitives.
- `Material` — resolves against §17 catalog.
- `Structure` — resolves against compiled `structure` table.

Parameters may have defaults: `template fireball(dmg: int = 55, radius: float = 2.0) { … }`.

### 15.2 Parameter references

Inside a template body, parameters appear as `$name`:

```
template aoe_damage(amount: int, radius: float, tag: Material, power: int) {
    damage $amount in circle($radius) [$tag: $power]
}
```

### 15.3 Arithmetic in arguments

Template calls accept arithmetic expressions over constants and caller-
scope parameters: `aoe_damage($base + 10, 3.0, FIRE, 60)`. Full operator
set + `min`/`max`/`clamp`/`abs` built-ins; no loops, no recursion.

### 15.4 Substitution rules

- Templates expand at AST lowering time, before effect catalog validation.
- No hygiene concerns — names are positional; no captured identifiers.
- Circular references (template A calls B calls A) → lower error.
- Max expansion depth: 16 (guards against unexpected blow-up).

**Status.** `planned`.

---

## §16 Structure blueprints

Structure blocks define voxel blueprints — `place_structure Castle(...)`
references them. Parameterized via the same mechanism as templates.

**Grammar.** See §3 `structure`.

### 16.1 Structure body

A structure contains:
- Optional `bounds:` — bounding volume for placement clipping.
- Optional `origin:` — anchor offset inside the structure's local space.
  Default anchor is baseplate center (XZ center of bbox at y=0).
- Optional `material:` — default material for `place` statements without
  an explicit material.
- Optional `symmetry:` — mirrors the body across axes at compile time.
- Sequence of voxel statements: `place`, `harvest`, `transform`, `include`,
  `if`.

### 16.2 Voxel statements

```
place <material> in <shape> [at <offset>] [rotate <deg>]
harvest in <shape> [at <offset>]
transform <mat_from> -> <mat_to> in <shape> [at <offset>]
include <structure>(<args>?) [at <offset>] [rotate <deg>] [scale <n>]
if <cond> { <voxel_stmts> } [else { <voxel_stmts> }]
```

### 16.3 Composition

`include` inserts another structure at an offset, optionally rotated
(`{0, 90, 180, 270}` only) and scaled (`{1, 2, 4}` only — no arbitrary
scale). Nested `include`s expand fully at compile time; depth capped at
16.

### 16.4 Symmetry

`symmetry: <kind>` values:

| Kind | Effect |
|---|---|
| `none` | No mirroring (default) |
| `x` | Mirror along X axis |
| `y` | Mirror along Y axis |
| `xy` | Mirror along both X and Y |
| `radial(n)` | N-fold radial symmetry around Y axis |

Authors draw one quadrant; the compiler unrolls the mirrors.

### 16.5 Compile-time expression evaluation

Structure bodies support full arithmetic + conditional placement (via
`if`). All resolved at structure-instantiation time (i.e., at
`place_structure` lowering), before CSG rasterization.

Example (parameterized castle):
```
structure Castle(
    wall_mat: Material = stone,
    height: int = 8,
    ornate: bool = true,
) {
    bounds: box(30, height + 5, 30)
    origin: (15, 0, 15)
    symmetry: radial(4)

    place $wall_mat in wall(30, $height, 2)
    place $wall_mat in column(3, $height + 3) at (14, 0, 14)
    if $ornate {
        place $wall_mat in dome(2) at (14, $height + 3, 14)
    }
}
```

### 16.6 IR mapping

Each `structure` lowers to a `StructureId` entry in the structure
registry. The entry carries:
- Rasterized 3D mask per parameter instantiation (compiled on-demand
  for each unique parameter set seen in `place_structure` calls).
- Anchor offset.
- Parameter type signature.
- Default palette.

Schema hash covers the full expanded set.

**Status.** `planned`.

---

## §17 Materials

Materials are declared in `.sim`-layer `enums.sim` (or a dedicated
`materials.sim`), **not** in `.ability` files. The ability DSL only
references them by name.

### 17.1 Declaration

```
material Ice {
    solid:         true,
    liquid:        false,
    destructible:  true,
    flammable:     false,
    hp:            30,
}

material Water {
    solid:         false,
    liquid:        true,
    destructible:  false,
    flammable:     false,
    hp:            0,
}
```

Fixed property set: `solid`, `liquid`, `destructible`, `flammable`, `hp`.
Adding a property = edit `materials.sim` + regenerate + schema-hash bump.

### 17.2 Resolution

- `.ability` / `structure` references to `IDENT` in material-slot
  position resolve against the declared catalog at lowering.
- Unknown name → lower-error.
- Catalog order pins material discriminants (`MaterialId: u8`).

### 17.3 IR mapping

- `MaterialId(u8)` — index into packed property buffer.
- Property lookups in DSL conditions (`floor_is(solid)`) read the packed
  buffer at `MaterialId * property_stride`.

### 17.4 Palette remapping (via templates)

Materials pass through templates and structures as `Material`-typed
parameters (§15.1, §16.5). A single structure authored in `stone` can
be instantiated in `ice` by passing the material parameter. No separate
"palette" mechanism — this is just substitution.

**Status.** `planned` (requires `materials.sim` catalog wired to the
DSL compiler's enum emitter).

---

## §18 Voxel effect verbs

### 18.1 Verb set

Four verbs:

#### `place_voxels <material> in <shape>`

Stamps the shape's rasterized voxel cells with `material`. Modifiers:
- `for <dur>` — timed revert (lifetime).
- `until_caster_dies` — reverts on caster death.
- `damageable_hp <n>` — voxels have HP; enemies can destroy them.
- `drop_as <item>` / `return_to_caster` — (for `harvest`; invalid here).

IR: `EffectOp::PlaceVoxels { mask, material, lifetime, damageable_hp }`.

#### `harvest_voxels in <shape> [<drop_mode>]`

Removes voxels in shape. Drop modes:
- default → destroy only.
- `drop_as <item>` → spawn world item per voxel (or bulk, impl choice).
- `return_to_caster` → add material count to caster's inventory.

IR: `EffectOp::HarvestVoxels { mask, drop_mode }`.

#### `transform_voxels <from> -> <to> in <shape>`

Per-voxel conditional — only voxels matching `from` become `to`. Others
untouched. Supports lifetime modifiers.

IR: `EffectOp::TransformVoxels { mask, from, to, lifetime }`.

#### `place_structure <name>[(args)]`

Instantiates a structure blueprint from §16. Args fill the structure's
parameters.

IR: `EffectOp::PlaceStructure { structure, lifetime }`.

### 18.2 Composability

Voxel effects share the 4-slot `effects` smallvec with combat effects.
An Earthshatter example:

```
ability Earthshatter {
    target: ground, range: 5.0
    cooldown: 20s, cast: 600ms
    hint: damage

    damage 80 in sphere(3.0) [PHYSICAL: 80]
    harvest_voxels in sphere(3.0) drop_as stone_chunk
}
```

Resolution order: combat effects resolve first in declaration order;
voxel effects resolve after. Both fire within one cast-resolve tick.

### 18.3 Material-conditional placement

Use the `when` modifier:

```
place_voxels ice in column(1.5, 4.0) when floor_material == water for 8s
```

See §19 for the voxel-aware condition atoms available.

### 18.4 Events

Each verb fires one bulk event per cast:
- `EffectVoxelsPlaced { cast_id, origin, mask, material, count, lifetime, tick }`
- `EffectVoxelsHarvested { cast_id, origin, mask, drop, count, tick }`
- `EffectVoxelsTransformed { cast_id, origin, mask, from, to, count, lifetime, tick }`
- `EffectStructurePlaced { cast_id, origin, structure, rotation, count, lifetime, tick }`

All `@replayable @bulk`. `count` is precomputed at compile time.

The existing `AgentPlacedVoxel` / `AgentHarvestedVoxel` events are
**preserved** for the non-ability `MicroKind::PlaceVoxel` / `HarvestVoxel`
path (agent worker placing a block by hand) — those remain per-voxel.

### 18.5 Side state

Three side tables keyed by `cast_id`:
- `VoxelRevertQueue` — heap of pending reverts, popped each tick.
- `DamageableVoxelSet` — hp-per-cast for damageable placements.
- `CasterLinkedCasts` — reverts on caster death.

**Status.** All voxel verbs `planned`; engine-side revert queue not wired today.

### 18.6 Worldgen interaction

Voxel ops interact with the world-extent / chunk-clip system already in
engine. Voxels falling outside world_extent are silently clipped at
rasterization (consistent with other placement primitives).

---

## §19 Voxel-aware conditions & triggers

### 19.1 Single-voxel conditions

Six atoms, each O(1):

| Atom | Semantics |
|---|---|
| `floor_material == <mat>` | Voxel below reference point matches |
| `floor_is(<prop>)` | Floor voxel has material property |
| `above_material == <mat>` | Voxel above reference matches |
| `above_is(<prop>)` | Above voxel has property |
| `voxel_at(dx, dy, dz) == <mat>` | Cell offset from reference matches |
| `voxel_at(dx, dy, dz) is <prop>` | Cell offset has property |
| `occupied(dx, dy, dz)` | Non-air at offset |
| `depth_below_surface < <n>` | Cells below height-map surface |

### 19.2 Volume-aggregate conditions

Two aggregates over compile-time shapes:

| Atom | Semantics |
|---|---|
| `any_voxel_in(<shape>, <mat>)` | At least one voxel matches |
| `any_voxel_in(<shape>, <prop>)` | At least one voxel has property |
| `voxel_count(<shape>, <mat>) <op> N` | Counting predicate |
| `voxel_count(<shape>, <prop>) <op> N` | Property-counting predicate |

Bounded by the shape's voxel count (compile-time known).

### 19.3 Reference frame

Default reference is determined by the ability's `target:`:
- `enemy` / `ally` → target's voxel.
- `self` / `self_aoe` → caster's voxel.
- `ground` → targeted position.

Explicit prefix overrides:
- `target_floor_material == X` — target's floor.
- `caster_floor_material == X` — caster's floor.
- `cast_floor_material == X` — ground-target cast point's floor (errors
  on non-ground modes).

Same prefix system applies to all voxel-aware atoms.

### 19.4 Passive triggers

Five voxel triggers (from §6):
- `on_voxel_placed(range:, by:, material:)`
- `on_voxel_harvested(range:, by:, material:)`
- `on_voxel_transformed(from:, to:, range:)`
- `on_voxel_reverted(range:)`
- `on_structure_placed(structure:, range:)`

All bulk-event-based — one trigger fire per verb invocation, not per
voxel.

**Status.** All voxel atoms + triggers `reserved` until the voxel
storage + bulk events are wired.

---

## §20 AI-state manipulation verbs

This section catalogs the ability-writable AI-state. All verbs emit
events folded by `@materialized` views in `assets/sim/views.sim` unless
noted. Status markers reference dependencies in §23.

### 20.1 Standing / relationship writes (H)

| Verb | Semantics | IR | Status |
|---|---|---|---|
| `befriend <n>` | `standing(caster, target) += n` | `ModifyStanding { a_sel: Caster, b_sel: Target, delta: +n }` | `runs-today` |
| `enmity <n>` | `standing(caster, target) -= n` | `ModifyStanding { delta: -n }` | `runs-today` |
| `endear <n>` | `standing(target, caster) += n` | `ModifyStanding { a_sel: Target, b_sel: Caster, delta: +n }` | `planned` |
| `charm <n> for <dur>` | timed enmity + auto-revert | `ModifyStanding` + revert queue | `planned` |
| `duel_challenge` | `standing(caster, target) = MIN` until death | `ModifyStanding { delta: MIN }` | `planned` |
| `slander <third> by <n>` | `standing(target, third) -= n` | `ModifyStanding { a_sel: Target, b_sel: OtherAgent(third), delta: -n }` | `planned` |
| `rally_cry <n> in <vol>` | AoE befriend within volume | N × `ModifyStanding` | `planned` |

Default magnitudes: see H4.γ — verbs have argument defaults but accept
explicit values.

The existing `EffectOp::ModifyStanding { delta: i16 }` evolves to
`{ a_sel: PairSelector, b_sel: PairSelector, delta: i16 }` where
`PairSelector ∈ {Caster, Target, OtherAgent(AgentId)}`. Schema hash bump.

### 20.2 Theory-of-mind manipulation (J)

Built against Phase 1 TOM (`BeliefState` snapshot). Phase 2 extensions
listed for completeness; all `reserved`.

#### Phase 1 writes

> **Spec audit (2026-05-08) — STATUS-FLIP + shape correction:** All
> seven verbs below shipped (Wave 3 phases 1, 3, 3.5, 4) but with
> simpler IR shapes than the original `observer_sel` / `subject_sel` /
> `BeliefFieldMask` design. The actual shipped `EffectOp` ordinals are
> 32–38 (not 12–14), each carrying a flat `subject_idx: u32` rather
> than a selector enum. `observe` shipped as a self-observe verb
> (`Observe { target_observer: u8 }`, variant 33) — refreshes caster's
> own belief row about cast target. `scry` shipped as cross-observer
> (`Scry { target_observer, subject_idx }`, variant 34). The
> "observers in volume" / range-gated fan-out for `reveal` collapses to
> "all observers" today (range gating awaits spatial-query infra).
> `stealth` ships as variant 27 (`runs-today`) but is a self-cast
> invisibility flag, NOT the "observers' beliefs of caster retain
> stale pos" semantic — the observer-side stale-pos behavior is
> downstream consumer-rule territory.

| Verb | Semantics | IR | Status |
|---|---|---|---|
| `observe <id>` (NEW — replaces synthetic `AgentObserved` plan) | Caster's own belief row about cast target refreshes from ground truth | `EffectOp::Observe { target_observer: u8 }` (variant 33) | `runs-today` |
| `scry <observer> <subject>` | Caster reads `observer`'s beliefs about `subject`, folds into caster's | `EffectOp::Scry { target_observer, subject_idx }` (variant 34) | `runs-today` |
| `reveal <subject>` | Caster broadcasts beliefs about subject to all observers (no range gate yet) | `EffectOp::Reveal { subject_idx }` (variant 35) | `runs-today` |
| `stealth for <dur>` | Self-cast invisibility flag (binary state) | `EffectOp::Stealth { duration_ticks }` (variant 27) | `runs-today` (lower-only — apply handler deferred) |
| `disguise <fake_type> for <dur>` | Caster publicly poses as `fake_type` for the duration | `EffectOp::Disguise { fake_type, duration_ticks }` (variant 36) | `runs-today` |
| `decoy <subject_idx> <fake_pos>` | Caster plants attacker-controlled BeliefState row in target's belief map about `subject_idx` | `EffectOp::Decoy { subject_idx, fake_pos }` (variant 37) | `runs-today` |
| `erase_belief <subject_idx> <fields>` | Caster clears specific fields of target's beliefs about `subject_idx` (bitset: pos/type/tick/confidence/suspicion/flags) | `EffectOp::EraseBelief { subject_idx, fields }` (variant 38) | `runs-today` |
| `plant_belief <subject_idx> bit <fact_bit>` | Bit-flag primitive: target's belief map for `subject_idx` gains `1u << fact_bit` via atomic-OR | `EffectOp::PlantBelief { subject_idx, fact_bit }` (variant 32) | `runs-today` |

Shipped `EffectOp`s (source: `crates/engine/src/ability/program.rs`):
```rust
PlantBelief { subject_idx: u32, fact_bit: u8 } = 32
Observe     { target_observer: u8 }            = 33
Scry        { target_observer: u8, subject_idx: u32 } = 34
Reveal      { subject_idx: u32 }                = 35
Disguise    { fake_type: u8, duration_ticks: u32 } = 36
Decoy       { subject_idx: u32, fake_pos: u32 } = 37
EraseBelief { subject_idx: u32, fields: u8 }    = 38
```
Shipped events (source: `crates/engine/src/cascade/handler.rs`):
`EffectPlantBeliefApplied = 63`, `EffectObserveApplied = 64`,
`EffectScryApplied = 65`, `EffectRevealApplied = 66`,
`EffectDisguiseApplied = 67`, `EffectDecoyApplied = 68`,
`EffectEraseBeliefApplied = 69`.

The original-design `ClearBelief` / `RefreshBelief` selector-typed
ops are `reserved` — the shipped `EraseBelief` (with fields bitset)
and `Observe` (refresh) cover the same use cases with simpler shapes.

#### Phase 2 extensions (reserved)

Two-layer TOM: Phase 1 `BeliefState` (physical snapshot) + new
`BeliefKnowledge { believed_knows: Bitset<32>, refreshed_at: [u32;32] }`.

| Verb | Semantics | Status |
|---|---|---|
| `teach_domain <target> <domain>` | Set bit in target's BeliefKnowledge about caster | `reserved` |
| `reveal_domain <domain> in <vol>` | Broadcast bits | `reserved` |
| `hide_domain <domain>` | Clear bit across observers | `reserved` |
| `forge_testimony <target> of <doc>` | source=Testimony(doc), trust=author.trust | `reserved` |
| `spread_rumor <subj> <fields>` | source=Hearsay(caster), trust=low | `reserved` |
| `verify_rumor <target> about <subj>` | Raises trust on matching belief | `reserved` |

Second-order beliefs (`beliefs(a).beliefs_of(b).about(c)`) — out of
scope entirely for this spec.

### 20.3 Communication manipulation (M)

Capability-side writes and event-attribution rewrites.

| Verb | Framing | Status |
|---|---|---|
| `silence <target> [channel <ch>] for <dur>` | Capability gate | `planned` |
| `mute <target> for <dur>` | Sugar for all linguistic channels | `planned` |
| `jam_channel <ch> in <vol> for <dur>` | AoE capability override | `planned` |
| `translate <target> to <lang> for <dur>` | Add LanguageId to capabilities | `planned` |
| `comprehend_languages <target> for <dur>` | Add every language | `planned` |
| `amplify_voice <target> by <factor> for <dur>` | Next Speech events get extended hearing_range | `planned` |
| `grant_telepathy <target> for <dur>` | Add Telepathy channel | `planned` |
| `speak_as <other> via <ch> "<msg>"` | Event-attribution rewrite | `reserved` |
| `impersonate <other> for <dur>` | All caster's AgentCommunicated events rewrite speaker | `reserved` |
| `forge_document as <other> with facts <list>` | Spawn Document item with forged author | `reserved` |

`AgentCommunicated` event gains a `real_caster: AgentId` field for
impersonation; `Document` gains `forger_id: Option<AgentId>`. Both are
schema-hash bumps.

### 20.4 Group / faction manipulation (K)

| Verb | Semantics | Status |
|---|---|---|
| `induct <target> [into <group>]` | Add membership (leader-only) | `reserved` |
| `exile <target> [from <group>]` | Remove membership (leader-only) | `reserved` |
| `defect [from <group>]` | Self-exile | `reserved` |
| `recruit <target>` | Alias for induct with implicit group | `reserved` |
| `crown <target>` | Set group leader (current leader only) | `reserved` |
| `abdicate` | Relinquish leadership | `reserved` |
| `depose <target>` | Forcibly remove leader (special gate) | `reserved` |
| `found_group <kind> [with members <list>]` | Spawn new group | `reserved` |
| `declare_war <group>` | `ModifyGroupStanding` to war-threshold | `reserved` |
| `form_alliance <group>` | `ModifyGroupStanding` to alliance | `reserved` |
| `break_treaty <group>` | Revert group standing to neutral | `reserved` |
| `send_tribute <group> <amount>` | TransferGold + ModifyGroupStanding | `reserved` |

All depend on `cold_memberships` + `Group` pool instance data (Plan 1
T16 shipped Pod shape; instance-data path pending).

### 20.5 Quest valuation manipulation (L)

Abilities do **not** touch quest creation, acceptance, completion, or
bidding (those are `MicroKind` economic acts). The ability DSL affects
how *observers perceive* quests — same structural model as TOM, but
subject-type=Quest.

Storage: `QuestBeliefState` alongside Phase 1 `BeliefState`.

```rust
struct QuestBeliefState {
    believed_reward:     f32,
    believed_difficulty: f32,
    believed_urgency:    f32,
    trust_in_poster:     f32,
    last_updated_tick:   u32,
    confidence:          f32,
}
```

| Verb | Semantics | Status |
|---|---|---|
| `inflate_bounty <quest> [for <obs>] [range: R] by <n>` | Raise believed_reward | `reserved` |
| `deflate_bounty <quest> [for <obs>] by <n>` | Lower believed_reward | `reserved` |
| `intimidate_takers <quest> by <n>` | Raise believed_difficulty | `reserved` |
| `embolden_takers <quest> by <n>` | Lower believed_difficulty | `reserved` |
| `urgent_call <quest> [range: R]` | Set believed_urgency to max | `reserved` |
| `false_lull <quest> [for <obs>]` | Zero believed_urgency | `reserved` |
| `endorse_quest <quest>` | Raise trust_in_poster | `reserved` |
| `slander_quest <quest>` | Lower trust_in_poster | `reserved` |
| `broadcast_quest <quest> [range: R]` | Fresh belief to all observers in range | `reserved` |
| `conceal_quest <quest> [from <obs>]` | Clear belief entry | `reserved` |

TOM `EffectOp`s (J6) generalize over `Subject: AgentId | QuestId` via
`SubjectSel` enum. Events gain `EffectQuestBelief{Planted,Cleared,Refreshed}`.

All `reserved` — depend on Phase 2 TOM extended to quest subjects.

### 20.6 Engagement / coordination writes (N)

Event-emission only. All verbs emit events that existing `@materialized`
views in `views.sim` already fold.

| Verb | Event emitted | Status |
|---|---|---|
| `force_engage <target> with <other>` | `EngagementCommitted { actor: target, engaged_with: other }` | `planned` |
| `break_engagement <target>` | `EngagementBroken { actor: target }` | `planned` |
| `taunt <target>` | `EngagementCommitted { actor: target, engaged_with: caster }` | `planned` |
| `scatter [in <vol>]` | `EngagementBroken` per agent in volume | `planned` |
| `set_pack_focus <target> for <obs>` | `PackAssist { observer: obs, target }` | `planned` |
| `rally_call <wounded_kin>` | `RallyCall { observer: caster, wounded_kin }` | `planned` |
| `rally_cry in <vol> around <wounded>` | `RallyCall` per observer in vol | `planned` |
| `pack_call <target> in <vol>` | `PackAssist` per observer in vol | `planned` |
| `incite_fear <toward> for <obs>` | `FearSpread { observer, dead_kin: toward }` | `planned` |
| `panic_wave in <vol>` | `FearSpread` per observer in vol | `planned` |

### 20.7 Fold/decay migration note

All coordination-view fold and decay logic (for `kin_fear`, `rally_boost`,
`pack_focus`, `threat_level`, `my_enemies`) MUST be expressed in
`.sim` files. Any fold or decay currently in hand-written engine Rust
is a migration item; this spec treats the `.sim` body as authoritative.

---

## §21 AI-state conditions (unified surface)

### 21.1 Grammar integration

AI-state atoms plug into the same `cond_expr` grammar (§11). Single
boolean algebra across physical / voxel / AI-state / tag / temporal
atoms.

### 21.2 Default reference frame

Bare atoms (no prefix) target `caster, target` by default:
- `standing_below(X)` → `standing(caster, target) < X`.
- `target_confidence_below(X)` → target's belief of caster, confidence
  field.

Explicit prefixes:
- `caster_standing_below(X)` — from caster's side.
- `pair_standing_below(<a>, <b>) < X` — three-party read.
- `believes(<target>, <subject>).<field> <op> <val>` — explicit triple.

### 21.3 Cost classes

Every atom carries a cost class. Scoring / `when` evaluators can
budget them per row.

| Class | Examples | Cost |
|---|---|---|
| 0 — O(1) scalar | `standing()`, `engaged_with()`, `has_channel()` | Cheap |
| 1 — O(K) top-K scan | `pack_focus_of`, `kin_fear`, `rally_boost_of` | K=8 scan |
| 2 — O(M) memory-ring | `target_spoke_within()`, `remembers()` | 64 + 5 entries |
| 3 — O(K²) cross-ref | Second-order beliefs | Reserved |

### 21.4 Catalog

Per §20's topics — the reads mirror the writes.

**Standing (H5, §20.1):**
- `standing_below(X)` / `standing_above(X)` — Class 0 — `planned`.
- `standing_between(lo, hi)` — Class 0 — `planned`.
- `is_enemy` / `is_friend` — threshold sugar — `planned`.
- `pair_standing_below(<a>, <b>)` — Class 0 — `planned`.

**TOM snapshot (J5):**
- `target_confidence_below(X)` — Class 0 — `planned`.
- `target_last_saw_me_within(<dur>)` — Class 0 — `planned`.
- `believes(<target>, <subject>).<field> <op> <val>` — Class 0 — `planned`.
- `target_has_belief_of(<subject>)` — Class 0 — `planned`.
- `confidence_of(<observer>, <subject>) <op> <val>` — Class 0 — `planned`.

**TOM knowledge-domain (reserved):**
- `target_believes_domain(<subject>, <domain>)` — Class 0 — `reserved`.

**Group / faction (K6):**
- `is_group_member`, `is_group_leader`, `can_join_group`, `is_outcast`
  — Class 0 — `reserved` (stubs).
- `is_enemy_of(<group>)`, `member_count(<group>) <op> N`,
  `leader_of(<group>) == <t>`, `standing_between(<g1>, <g2>) <op> N`
  — sugar over stubs — `reserved`.

**Quest valuation (L4):**
- `can_accept`, `is_target`, `party_near_destination` — stubs — `reserved`.
- `quest_active(<id>)`, `quest_poster(<id>) == <t>`,
  `quest_kind(<id>) == <kind>`, `has_pending_quest`,
  `party_size(<id>) <op> N` — sugar — `reserved`.
- `believes_quest(<t>, <quest>).<field> <op> <val>`,
  `target_values_quest_above(<quest>, <val>)`,
  `confidence_about(<t>, <quest>) <op> <val>` — `reserved`.

**Communication (M7):**
- `target_has_channel(<ch>)` — Class 0 — `planned`.
- `target_has_language(<lang>)` — Class 0 — `planned`.
- `channel_jammed_here(<ch>)` — Class 0 — `planned`.
- `target_spoke_within(<dur>)` — Class 2 — `planned`.
- `target_last_communicated_with(<other>)` — Class 2 — `planned`.
- `in_conversation(<a>, <b>)` — Class 0 — `planned`.

**Engagement / coordination (N5):**
- `engaged_elsewhere` — Class 0 — `runs-today`.
- `engaged_with(<other>)` — Class 0 — `runs-today`.
- `target_engaged_with(<other>)` — Class 0 — `runs-today`.
- `pack_focused_on(<me>)` — Class 1 — `planned`.
- `pack_focus_of(<observer>, <target>) <op> N` — Class 1 — `planned`.
- `fear_level_above(<N>)` — Class 1 — `planned`.
- `rally_boost_of(<observer>) <op> N` — Class 1 — `planned`.
- `threat_level(<observer>, <target>) <op> N` — Class 1 — `planned`.

### 21.5 Mixed example

```
damage 80 in sphere(3.0) when and(
    floor_material == water,          // voxel
    target_confidence_below(0.3),     // TOM
    standing_below(-500),             // standing
    engaged_elsewhere                 // engagement
) [FIRE: 60]
```

Short-circuit evaluation, left-to-right. Authors order expensive
(Class-2) atoms last.

---

## §22 IR lowering contract

### 22.1 `AbilityProgram` anatomy

After all locked decisions, `AbilityProgram`:

```rust
struct AbilityProgram {
    delivery: Delivery,             // Instant | Projectile | Chain | Zone | Channel | Tether | Trap
    area:     Area,                 // SingleTarget | Circle | Cone | … (ground-anchored variants)
    gate:     Gate,                 // cooldown_ticks, hostile_only, line_of_sight, cast_ticks, charges, unstoppable
    effects:  SmallVec<[EffectOp; 4]>,
    hint:     Option<AbilityHint>,
    tags:     SmallVec<[(AbilityTag, f32); 6]>,
    // Added by this spec:
    delivery_params: DeliveryParams,  // per-method config (speed, bounces, etc.)
    stage:           Option<StageDescriptor>,  // for recast multi-stage
    form:            Option<FormTag>,
    swap_form:       Option<FormTag>,
    zone_tag:        Option<InternedString>,
}
```

### 22.2 `EffectOp` variants

> ⚠️ **Spec audit (2026-05-08) — `[CRITICAL]`:** The previous ordinal
> assignments below (8–26) were a design proposal that the engine
> shipped a **different** layout for. The engine's actual
> `crates/engine/src/ability/program.rs::EffectOp` enum is the source
> of truth — replace the design proposal with the shipped table. Any
> code compiled against the old (8 = `PlaceVoxels`, 12 = `ClearBelief`,
> …) ordinals would produce silently-mismatched chronicle records.
> The shipped engine reserves variants 8–11 for `Root`/`Silence`/`Fear`/
> `Taunt` (Wave 2 piece 1), 12–15 for movement (Dash/Blink/Knockback/
> Pull), 16–17 for Execute/SelfDamage, 18–19 for LifeSteal/DamageModify,
> 20–22 for the over-time triplet, 23–24 for Buff/Summon, 25–26 for
> Harvest/PlaceVoxel, 27 for Stealth, 28–31 for the extended-CC
> vocabulary, 32–38 for the seven ToM verbs, 39 for TravelTo, and
> 40–45 for Lifts B/C/D (Recipe/WearTool/Propose/Announce/GainSkill/
> CreateObligation). The original design's voxel-area ops
> (`PlaceVoxels`/`HarvestVoxels`/`TransformVoxels`/`PlaceStructure`),
> belief-selector ops (`ClearBelief`/`RefreshBelief`/`PlantBelief` with
> selectors), the `EmitEvent` escape hatch, the
> `ModifyGroupStanding` op, and the rest of the economy ops
> (`TransferProperty`/`ForcibleTransfer`/`DischargeObligation`/
> `DefaultObligation`/`EstablishRoute`/`JoinCaravan`/
> `TransferObligation`) are all **MISSING** — no engine variant, no
> DSL arm. They remain `reserved`.

#### 22.2.1 Shipped layout (source of truth: `crates/engine/src/ability/program.rs`)

| Ordinal | Variant | Wave / Lift | Status |
|---|---|---|---|
| 0 | `Damage { amount: f32 }` | Combat | `runs-today` |
| 1 | `Heal { amount: f32 }` | Combat | `runs-today` |
| 2 | `Shield { amount: f32 }` | Combat | `runs-today` |
| 3 | `Stun { duration_ticks: u32 }` | Combat | `runs-today` |
| 4 | `Slow { duration_ticks, factor_q8: i16 }` | Combat | `runs-today` |
| 5 | `TransferGold { amount: i32 }` | World | `runs-today` |
| 6 | `ModifyStanding { delta: i16 }` (legacy single-i16 shape) | World | `runs-today` |
| 7 | `CastAbility { ability, selector }` | Meta | `runs-today` |
| 8 | `Root { duration_ticks }` | Wave 2 piece 1 | `runs-today` |
| 9 | `Silence { duration_ticks }` | Wave 2 piece 1 | `runs-today` |
| 10 | `Fear { duration_ticks }` | Wave 2 piece 1 | `runs-today` |
| 11 | `Taunt { duration_ticks }` | Wave 2 piece 1 | `runs-today` |
| 12 | `Dash { distance: f32 }` (∞ = `to_target` sentinel) | Wave 2 piece 2 | `runs-today` |
| 13 | `Blink { distance: f32 }` | Wave 2 piece 2 | `runs-today` |
| 14 | `Knockback { distance: f32 }` | Wave 2 piece 2 | `runs-today` |
| 15 | `Pull { distance: f32 }` | Wave 2 piece 2 | `runs-today` |
| 16 | `Execute { hp_threshold: f32 }` | Wave 2 piece 3 | `runs-today` |
| 17 | `SelfDamage { amount: f32 }` | Wave 2 piece 3 | `runs-today` |
| 18 | `LifeSteal { duration_ticks, fraction_q8: i16 }` | Wave 2 piece 4 | `runs-today` |
| 19 | `DamageModify { duration_ticks, multiplier_q8: i16 }` | Wave 2 piece 4 | `runs-today` |
| 20 | `DamageOverTime { amount: f32, duration_ticks }` | Wave 1.5+ | `runs-today` |
| 21 | `HealOverTime { amount: f32, duration_ticks }` | Wave 1.5+ | `runs-today` |
| 22 | `TimedShield { amount: f32, duration_ticks }` | Wave 1.5+ | `runs-today` |
| 23 | `Buff { stat: BuffStat, magnitude_q8: i16, duration_ticks }` | Wave 2 piece 4 | `runs-today` |
| 24 | `Summon { template_hash: u32, count: u8, lifetime_ticks }` | Wave 2 | `runs-today` (lower-only) |
| 25 | `Harvest { kind_hash: u32, amount: u16 }` | Non-combat ph1 | `runs-today` |
| 26 | `PlaceVoxel { kind_hash: u32 }` | Non-combat ph1 | `runs-today` |
| 27 | `Stealth { duration_ticks }` | Wave 2 piece 7 | `runs-today` |
| 28 | `Charm { duration_ticks }` | Wave 2 piece 8 | `runs-today` |
| 29 | `Grounded { duration_ticks }` | Wave 2 piece 8 | `runs-today` |
| 30 | `Suppress { duration_ticks }` | Wave 2 piece 8 | `runs-today` |
| 31 | `Reflect { duration_ticks, fraction_q8: i16 }` | Wave 2 piece 8 | `runs-today` |
| 32 | `PlantBelief { subject_idx: u32, fact_bit: u8 }` | Wave 3 phase 1 (ToM) | `runs-today` |
| 33 | `Observe { target_observer: u8 }` | Wave 3 phase 3 (ToM) | `runs-today` |
| 34 | `Scry { target_observer: u8, subject_idx: u32 }` | Wave 3 phase 3.5 (ToM) | `runs-today` |
| 35 | `Reveal { subject_idx: u32 }` | Wave 3 phase 3.5 (ToM) | `runs-today` |
| 36 | `Disguise { fake_type: u8, duration_ticks }` | Wave 3 phase 4 (ToM) | `runs-today` |
| 37 | `Decoy { subject_idx: u32, fake_pos: u32 }` | Wave 3 phase 4 (ToM) | `runs-today` |
| 38 | `EraseBelief { subject_idx: u32, fields: u8 }` | Wave 3 phase 4 (ToM) | `runs-today` |
| 39 | `TravelTo { dest_x_q8: i16, dest_y_q8: i16, eta_ticks }` | Lift A | `runs-today` |
| 40 | `Recipe { recipe_id: u16, target_tool: u8 }` | Lift B | `runs-today` |
| 41 | `WearTool { tool_kind: u8, amount: u16 }` | Lift B | `runs-today` |
| 42 | `Propose { contract_kind: u8, expires_at_tick: u32 }` | Lift C | `runs-today` |
| 43 | `Announce { announcement_kind: u8, radius_q8: u16 }` | Lift C | `runs-today` |
| 44 | `GainSkill { skill_id: u8, amount_q8: u16 }` | Lift D | `runs-today` |
| 45 | `CreateObligation { obligation_id: u16, kind: u8 }` | Lift D | `runs-today` |

#### 22.2.2 Reserved (planned / not shipped)

The original spec proposed several IR shapes that the engine has not
adopted. Listed here for traceability:

| Variant | Source | Status |
|---|---|---|
| `PlaceVoxels { mask, material, lifetime, damageable_hp }` | §18.1 (area-stamp) | `reserved` (no engine variant) |
| `HarvestVoxels { mask, drop }` | §18.1 (area-harvest) | `reserved` (single-cell `Harvest`/`PlaceVoxel` ship today; area shapes await mask registry) |
| `TransformVoxels { mask, from, to, lifetime }` | §18.1 | `reserved` |
| `PlaceStructure { structure, lifetime }` | §18.1 | `reserved` |
| `ClearBelief { observer_sel, subject_sel }` | §20.2 | `reserved` (use `EraseBelief { subject_idx, fields }` for fields-bitset; selector-typed clear is unimplemented) |
| `RefreshBelief { observer_sel, subject_sel }` | §20.2 | `reserved` (use `Observe { target_observer }`) |
| `EmitEvent { kind: EventKindId, payload_sel: PayloadSel }` | §20.3, §20.6 | `reserved` (each verb gets its own packed EffectOp instead) |
| `ModifyGroupStanding { group_sel, other_group, delta: i16 }` | §20.4 | `reserved` |
| `TransferProperty { property_id, target_sel }` | econ §6.1 | `reserved` |
| `ForcibleTransfer { subject, target_sel, contest_kind, detection_threshold }` | econ §6.3 | `reserved` |
| `DischargeObligation { obligation_id }` | econ §7.5 | `reserved` |
| `DefaultObligation { obligation_id }` | econ §7.5 | `reserved` |
| `EstablishRoute { from, to }` | econ §5.6 | `reserved` |
| `JoinCaravan { caravan: GroupId }` | econ §5.6 | `reserved` |
| `TransferObligation { obligation_id, target_sel }` | econ §12.1 | `reserved` |

The promised `ModifyStanding` evolution to `{ a_sel, b_sel, delta }`
(three-party standing writes) is also `reserved` — the shipped variant
6 still carries only `{ delta: i16 }` and writes the symmetric edge
between caster and cast target.

Proxy types: `VoxelMaskId`, `StructureId`, `MaterialId` are `NonZeroU16` / `u8`.

#### Selector enums

```rust
enum PairSelector {       // used by ModifyStanding
    Caster,
    Target,
    OtherAgent(AgentId),  // named at compile time from scope
}

enum AgentSel {           // used by belief ops (observer / subject roles)
    Caster,
    Target,
    OtherAgent(AgentId),
}

enum SubjectSel {         // used when a belief op's subject can be agent or quest
    Agent(AgentSel),
    Quest(QuestSel),
}

enum QuestSel {
    ThisQuest,            // quest referenced in the ability's ground target
    NamedQuest(QuestId),
}

enum GroupSel {
    CasterGroup,          // first membership of caster
    TargetGroup,
    NamedGroup(GroupId),
}

enum PayloadSel {         // for EffectOp::EmitEvent
    Preset(u8),           // baked payload per verb (compile-time resolved)
    FromScope,            // event fields resolved from cast scope
                          // (caster, target, radius, etc.)
}
```

`EmitEvent` is the escape hatch used by engagement / communication /
group verbs that write to `@materialized` views: the compiler knows
the event's `EventKindId` and the payload shape (from `events.sim`),
and `PayloadSel::FromScope` pulls the required fields from the cast
packet (actor = caster, target = cast target, etc.). `Preset(u8)` is
for verbs whose payloads are literal constants (e.g., `rally_call`
always fires with the caster as `observer` and the caster's closest
wounded kin as `wounded_kin`).

### 22.3 Registries

Three registries, each schema-hashed:

- `AbilityRegistry` — `AbilityId → AbilityProgram`.
- `VoxelMaskRegistry` — `VoxelMaskId → (rasterized 3D bitmask, bbox, voxel_count)`.
- `StructureRegistry` — `StructureId → (rasterized mask per parameter
  instantiation, origin, symmetry, parameter sig, default palette)`.

Material catalog is in `.sim`; referenced by `MaterialId(u8)` index.

### 22.4 Event additions

> **Spec audit (2026-05-08) — STATUS-FLIP + UNDOCUMENTED:** The events
> below were the design proposal. The engine actually shipped a
> different set under `EventKindId` in
> `crates/engine/src/cascade/handler.rs` (chronicle ordinals 26–76).
> The voxel / belief shapes documented below are NOT the shipped shapes —
> the shipped variants carry caster + target slots + packed payload
> words, not symbolic `cast_id, origin, mask, …` fields.

#### 22.4.1 Shipped chronicle events (source of truth: `crates/engine/src/cascade/handler.rs::EventKindId`)

| Ordinal | Event | EffectOp source | Status |
|---|---|---|---|
| 26 | `EffectDamageApplied` | `Damage` (0) | `runs-today` |
| 27 | `EffectHealApplied` | `Heal` (1) | `runs-today` |
| 28 | `EffectShieldApplied` | `Shield` (2) | `runs-today` |
| 29 | `EffectStunApplied` | `Stun` (3) | `runs-today` |
| 30 | `EffectSlowApplied` | `Slow` (4) | `runs-today` |
| 31 | `EffectGoldTransfer` | `TransferGold` (5) | `runs-today` |
| 32 | `EffectStandingDelta` | `ModifyStanding` (6) | `runs-today` |
| 33 | `CastDepthExceeded` | (cascade depth bound) | `runs-today` |
| 34 | `EngagementCommitted` | (mask trigger) | `runs-today` |
| 35 | `EngagementBroken` | (mask trigger) | `runs-today` |
| 36 | `FearSpread` | `AgentDied` fan-out | `runs-today` |
| 37 | `PackAssist` | `EngagementCommitted` fan-out | `runs-today` |
| 38 | `RallyCall` | wounded-victim fan-out | `runs-today` |
| 39 | `EffectSelfDamageApplied` | `SelfDamage` (17) | `runs-today` |
| 40 | `EffectLifeStealApplied` | `LifeSteal` (18) | `runs-today` |
| 41 | `EffectDamageModifyApplied` | `DamageModify` (19) | `runs-today` |
| 42 | `EffectExecuteApplied` | `Execute` (16) | `runs-today` |
| 43 | `EffectRootApplied` | `Root` (8) | `runs-today` |
| 44 | `EffectSilenceApplied` | `Silence` (9) | `runs-today` |
| 45 | `EffectFearApplied` | `Fear` (10) | `runs-today` |
| 46 | `EffectTauntApplied` | `Taunt` (11) | `runs-today` |
| 47 | `EffectDashApplied` | `Dash` (12) | `runs-today` |
| 48 | `EffectBlinkApplied` | `Blink` (13) | `runs-today` |
| 49 | `EffectKnockbackApplied` | `Knockback` (14) | `runs-today` |
| 50 | `EffectPullApplied` | `Pull` (15) | `runs-today` |
| 51 | `EffectDamageOverTimeApplied` | `DamageOverTime` (20) | `runs-today` |
| 52 | `EffectHealOverTimeApplied` | `HealOverTime` (21) | `runs-today` |
| 53 | `EffectTimedShieldApplied` | `TimedShield` (22) | `runs-today` |
| 54 | `EffectStealthApplied` | `Stealth` (27) | `runs-today` |
| 55 | `EffectCharmApplied` | `Charm` (28) | `runs-today` |
| 56 | `EffectGroundedApplied` | `Grounded` (29) | `runs-today` |
| 57 | `EffectSuppressApplied` | `Suppress` (30) | `runs-today` |
| 58 | `EffectBuffApplied` | `Buff` (23) | `runs-today` |
| 59 | `EffectHarvestApplied` | `Harvest` (25) | `runs-today` |
| 60 | `EffectPlaceVoxelApplied` | `PlaceVoxel` (26) | `runs-today` |
| 61 | `EffectReflectApplied` | `Reflect` (31) | `runs-today` |
| 62 | `EffectSummonApplied` | `Summon` (24) | `runs-today` |
| 63 | `EffectPlantBeliefApplied` | `PlantBelief` (32) | `runs-today` |
| 64 | `EffectObserveApplied` | `Observe` (33) | `runs-today` |
| 65 | `EffectScryApplied` | `Scry` (34) | `runs-today` |
| 66 | `EffectRevealApplied` | `Reveal` (35) | `runs-today` |
| 67 | `EffectDisguiseApplied` | `Disguise` (36) | `runs-today` |
| 68 | `EffectDecoyApplied` | `Decoy` (37) | `runs-today` |
| 69 | `EffectEraseBeliefApplied` | `EraseBelief` (38) | `runs-today` |
| 70 | `EffectTravelToApplied` | `TravelTo` (39) | `runs-today` |
| 71 | `EffectRecipeApplied` | `Recipe` (40) | `runs-today` |
| 72 | `EffectWearToolApplied` | `WearTool` (41) | `runs-today` |
| 73 | `EffectProposeApplied` | `Propose` (42) | `runs-today` |
| 74 | `EffectAnnounceApplied` | `Announce` (43) | `runs-today` |
| 75 | `EffectGainSkillApplied` | `GainSkill` (44) | `runs-today` |
| 76 | `EffectCreateObligationApplied` | `CreateObligation` (45) | `runs-today` |
| 128 | `ChronicleEntry` | `chronicle <event> { ... }` records | `runs-today` |

Slots 23–24 retired in task 143 along with `StunExpired` / `SlowExpired`
(stun/slow expiry is now a synthetic boundary read off
`stun_expires_at_tick` / `slow_expires_at_tick`).

#### 22.4.2 Reserved (planned / not shipped)

The original event-shape proposals — `EffectVoxelsPlaced`,
`EffectVoxelsHarvested`, `EffectVoxelsTransformed`,
`EffectStructurePlaced`, `EffectVoxelDamage`, `EffectVoxelReverted`,
`EffectBeliefCleared`, `EffectBeliefPlanted` (with selector enums),
`EffectBeliefRefreshed`, `EffectQuestBelief{Cleared,Planted,Refreshed}` —
are all `reserved`. The shipped Wave 3 ToM events use the much simpler
`subject_idx: u32 + payload bits` shape; the area-stamp voxel events
are deferred until the voxel-mask registry lands.

Schema hash covers all new events, `EffectOp` ordinals, registry
layouts.

### 22.5 Side state

- `VoxelRevertQueue` — heap of pending reverts.
- `DamageableVoxelSet` — hp map per placement cast.
- `CasterLinkedCasts` — caster → {cast_id} for on-death reverts.
- `EffectRevertQueue` — generic queue for timed standing / TOM reverts
  (extends the voxel-only queue at a later phase).

---

## §23 Capability status matrix

> ⚠️ **Audit 2026-04-26 — matrix overclaims TacticalSim parity:** Status markers describe **only the engine stack**. Constructs marked `planned` for engine are often `runs-today` in TacticalSim:
> - `passive` block: TacticalSim runs full trigger system (17+ kinds, depth-guarded recursion, cooldowns).
> - `deliver projectile/chain/zone/channel`: TacticalSim parsed and dispatched; engine has only `Delivery::Instant`.
> - 2D shapes (circle/cone/line/ring/spread): TacticalSim full impl; engine has only `SingleTarget`.
> - Control verbs (root, silence, fear, taunt) + reserved verbs (charm, polymorph, banish, reflect, summon, leash, …): TacticalSim implements; spec says `planned`/`reserved`.
> - Movement verbs (dash, knockback, pull, blink, swap), buffs/debuffs (lifesteal, reflect, blind), advanced (execute, self_damage, summon, dispel, immunity, death_mark, …): all TacticalSim-only.
>
> Spec also marks `hint: heal` as `planned` but engine `AbilityHint` enum has no `Heal` variant — no documented fallback to `Utility` exists either (overclaim).
>
> §22.2 `ModifyStanding` evolution to `{ a_sel, b_sel, delta }` not done; engine's `ModifyStanding` still carries only `delta: i16` — three-party standing writes (`endear`, `slander`, `rally_cry`) are unimplementable.
> See `docs/superpowers/notes/2026-04-26-audit-ability.md` for per-row detail and the four implementation tracks proposed.

Consolidated table. Per construct: status, required engine work, linked
ticket (when opened). Construct IDs reference sections above.

### 23.1 Core (§5, §6, §7, §8.1)

| Construct | Status |
|---|---|
| `ability` block | `runs-today` |
| `passive` block | `planned` (no Trigger AST or handler in engine yet) |
| `target: enemy` / `self` | `runs-today` |
| `target: ally` / `self_aoe` / `ground` / `direction` | `planned` |
| `target: vector` / `global` | `reserved` |
| `range:`, `cooldown:`, `hint:` | `runs-today` |
| `hint: heal` / `hint: buff` (variants 4, 5) | `runs-today` |
| `cast:`, `charges:`, `unstoppable` | `planned` |
| `cost:`, `zone_tag:`, `toggle`, `form:`, `swap_form:`, `recast`, `morph` | `reserved` |
| Combat effects (`damage`, `heal`, `shield`, `stun`, `slow`) | `runs-today` |
| `damage <a> for <d>` / `heal <a> for <d>` / `shield <a> for <d>` (DoT/HoT/TimedShield, variants 20–22) | `runs-today` |
| Movement verbs (`dash` / `blink` / `knockback` / `pull`) | `runs-today` |
| `dash to_target` (∞-distance sentinel) | `runs-today` |
| `dash to_position`, `swap` | `reserved` |
| `travel_to` (Lift A multi-tick) | `runs-today` |
| Control verbs (`root` / `silence` / `fear` / `taunt` / `charm` / `grounded` / `suppress`) | `runs-today` |
| Control verbs (`polymorph` / `banish` / `confuse`) | `reserved` |
| Buff / debuff verbs (`buff` / `damage_modify` / `lifesteal` / `reflect`) | `runs-today` |
| `debuff` (negative-magnitude Buff), `blind` | `planned` / `reserved` |
| Advanced verbs (`execute` / `self_damage` / `summon` / `stealth`) | `runs-today` (some lower-only — apply handlers per fixture) |
| Other advanced verbs (§8.5) | `reserved` |
| ToM verbs (`plant_belief` / `observe` / `scry` / `reveal` / `disguise` / `decoy` / `erase_belief`) | `runs-today` |
| Production / consent (`cast_recipe` / `wear_tool` / `propose` / `announce` / `gain_skill` / `create_obligation`) | `runs-today` |
| World primitives (`harvest` / `mine` / `place_voxel`) | `runs-today` |
| Tags: fixed 6 enum | `runs-today` |
| Tags: open-set mapping | `planned` |
| Scaling: stat terms | `planned` |
| Conditions: physical atoms | mixed `runs-today` / `planned` |
| Delivery: `Instant` | `runs-today` |
| Delivery: all other methods | `planned` or `reserved` |
| Lexical: hex literals (`0xFF`) + integer suffixes (`32u8`) | `runs-today` (PR #38, 2026-05-08) |

### 23.2 Shapes & volumes (§9)

| Construct | Status |
|---|---|
| 2D shapes (circle/cone/line/ring/spread) | `planned` |
| 3D volumes (box/sphere/column/wall/cylinder/dome/hull) | `planned` |
| CSG composition (union/diff/intersect) | `planned` |
| Grid snapping & 90° rotation | `planned` |
| Shape rasterization + mask registry | `planned` |

### 23.3 Templates & structures (§15, §16, §17)

| Construct | Status |
|---|---|
| `template` block with typed parameters | `planned` |
| Compile-time arithmetic (+−×÷%, min/max/clamp/abs) | `planned` |
| `structure` block with parameters | `planned` |
| `include` composition + symmetry | `planned` |
| Anchor (baseplate-center default + `origin:` override) | `planned` |
| Material declaration in `.sim` | `planned` |
| Material property catalog (solid/liquid/destructible/flammable/hp) | `planned` |

### 23.4 Voxel ops (§18, §19)

| Construct | Status |
|---|---|
| `place_voxels`, `harvest_voxels`, `transform_voxels`, `place_structure` | `planned` |
| Lifetime modifiers (`for`, `until_caster_dies`, `damageable_hp`) | `planned` |
| Drop modes (`drop_as`, `return_to_caster`) | `planned` |
| New `EffectOp` variants 8–11 | `planned` |
| Bulk voxel events | `planned` |
| Side tables (revert queue, damageable set, caster-linked) | `planned` |
| Visually-phased rise (cosmetic) | `planned` |
| Voxel single-voxel conditions (§19.1) | `reserved` |
| Voxel aggregate conditions (§19.2) | `reserved` |
| Voxel passive triggers (§19.4) | `reserved` |

### 23.5 AI-state manipulation (§20)

| Subsystem | Writes | Reads |
|---|---|---|
| Standing (H) | `runs-today` for basic, `planned` for three-party | `planned` |
| TOM Phase 1 (J) | `planned` (feature-gated) | `planned` (feature-gated) |
| TOM Phase 2 bitset (J2.γ) | `reserved` | `reserved` |
| Communication capability (M1–M3) | `planned` | `planned` |
| Communication attribution (M4–M5) | `reserved` | `reserved` |
| Group / faction (K) | `reserved` | `reserved` |
| Quest valuation (L) | `reserved` | `reserved` |
| Engagement / coordination (N) | `planned` | `planned` |

### 23.6 IR & events

> **Spec audit (2026-05-08) — STATUS-FLIP:** The ordinal claims below
> were the design proposal. The shipped engine ships variants 8–11 as
> `Root`/`Silence`/`Fear`/`Taunt`, 12–15 as movement, 16–17 as
> `Execute`/`SelfDamage`, 18–19 as `LifeSteal`/`DamageModify`, 20–22 as
> over-time triplet, 23 as `Buff`, 24 as `Summon`, 25–26 as
> `Harvest`/`PlaceVoxel`, 27 as `Stealth`, 28–31 as
> `Charm`/`Grounded`/`Suppress`/`Reflect`, 32–38 as the seven ToM verbs,
> 39 as `TravelTo`, and 40–45 as Lifts B/C/D. The voxel-area /
> EmitEvent / GroupStanding / quest-belief proposals are all
> `reserved` (no engine variant). See §22.2 for the full reconciliation.

| Construct | Status |
|---|---|
| `EffectOp` 8–11 (Root/Silence/Fear/Taunt — Wave 2) | `runs-today` |
| `EffectOp` 12–15 (Dash/Blink/Knockback/Pull) | `runs-today` |
| `EffectOp` 16–17 (Execute/SelfDamage) | `runs-today` |
| `EffectOp` 18–19 (LifeSteal/DamageModify) | `runs-today` |
| `EffectOp` 20–22 (DoT/HoT/TimedShield) | `runs-today` |
| `EffectOp` 23–24 (Buff/Summon) | `runs-today` |
| `EffectOp` 25–26 (Harvest/PlaceVoxel — single-cell) | `runs-today` |
| `EffectOp` 27 (Stealth) | `runs-today` |
| `EffectOp` 28–31 (Charm/Grounded/Suppress/Reflect) | `runs-today` |
| `EffectOp` 32–38 (ToM verbs) | `runs-today` |
| `EffectOp` 39 (TravelTo — Lift A) | `runs-today` |
| `EffectOp` 40–45 (Recipe/WearTool/Propose/Announce/GainSkill/CreateObligation — Lifts B/C/D) | `runs-today` |
| `EffectOp` voxel-area ops (`PlaceVoxels`/`HarvestVoxels`/`TransformVoxels`/`PlaceStructure`) | `reserved` |
| `EffectOp` `EmitEvent` / `ModifyGroupStanding` / quest-belief shapes | `reserved` |
| `PairSelector` / `SubjectSel` / `AgentSel` enums | `reserved` (shipped ToM uses `subject_idx: u32`) |
| `ModifyStanding { a_sel, b_sel, delta }` selector evolution | `reserved` (variant 6 still `{ delta: i16 }`) |
| Voxel mask registry + schema hash | `planned` |
| Structure registry + schema hash | `planned` |
| Chronicle events 26–76 (per §22.4.1) | `runs-today` |
| Bulk events (area-stamp voxel, structure) | `reserved` |
| Belief events (agent — `EffectPlantBeliefApplied=63` … `EffectEraseBeliefApplied=69`) | `runs-today` |
| Quest-belief events | `reserved` |
| Communication attribution fields (`real_caster`, `forger_id`) | `reserved` |

### 23.7 Budget & determinism

| Constant | Value |
|---|---|
| `MAX_ABILITIES` (per agent) | 8 (ships today) |
| `MAX_EFFECTS_PER_PROGRAM` | 4 (ships today) |
| `MAX_TAGS_PER_PROGRAM` | 6 (ships today) |
| `MAX_CASCADE_ITERATIONS` | 8 (ships today) |
| `MAX_VOXELS_PER_ABILITY` | 16384 (planned) |
| `MAX_VOXELS_PER_TICK` | 65536 (planned, spill queue) |
| `MAX_ABILITY_CASTS_PER_TICK` | 1024 (planned, spill queue) |
| Template expansion depth | 16 (planned) |
| Structure include depth | 16 (planned) |

---

## §24 Budget & determinism

### 24.1 Per-ability caps

- `MAX_EFFECTS_PER_PROGRAM = 4` — hard static cap. Including voxel ops.
- `MAX_TAGS_PER_PROGRAM = 6` — `AbilityTag::COUNT`.
- `MAX_VOXELS_PER_ABILITY = 16384` — compile-time sum across all 4
  effect slots after structure expansion + CSG rasterization.

Exceeding any cap → lower-error.

### 24.2 Per-tick caps

- `MAX_VOXELS_PER_TICK = 65536` — total voxel mutations per tick.
  Overflow casts spill into the next tick's queue, ordered by
  `(cast_start_tick, cast_id)`.
- `MAX_ABILITY_CASTS_PER_TICK = 1024` — tunable via `config.sim`.
  Same spill mechanism.

Both caps checked independently; hitting either triggers spill. Cast's
`AgentCast` event carries the actual resolve tick, not the intent tick.

### 24.3 Determinism contract

- **Mask rasterization** is bit-identical across CPU and GPU. `wgsl`
  and CPU share the same fused formula; no platform-dependent rounding.
- **Structure compilation** (expression folding + symmetry unroll +
  include composition) is a single-pass, declaration-order iteration.
  Same inputs → byte-identical registry.
- **Revert queue** pops in `(revert_tick, cast_id)` order; ties broken
  by `cast_id` (globally unique).
- **Bulk event emission** inside a tick is `cast_id`-ascending.
- **Overlap rule:** later `cast_id` wins on conflicting voxel writes.
  `TransformVoxels` fired after `PlaceVoxels` in the same tick uses the
  post-place state.
- **World-extent clip** is silent — voxels outside `world_extent` are
  dropped at rasterization. Chronicle & event count reflect
  actually-placed voxels.
- **Cascade depth** is a hard bound; per-tick voxel budget is a soft
  spill (cascades are not rewound on over-budget).

### 24.4 Replay invalidation

Any of the following bumps the schema hash and invalidates prior
replays:

- Material catalog additions / edits.
- Voxel mask CSG edits.
- Structure body / parameter / symmetry edits.
- `EffectOp` ordinal changes.
- New event kinds.
- Selector enum changes.

---

## §25 Error model

### 25.1 Parse errors

Reported with file, line, column, span. Categories:

- **Lexical** — unknown token, bad duration (`5xs`), unterminated string.
- **Grammar** — unexpected token at position, missing `{` / `(`, etc.
- **Duplicate header** — `duplicate 'cooldown:' in ability 'Fireball'`.
- **Mixed body** — `ability 'X' mixes deliver{} and top-level effects`.
- **Unknown reserved keyword in identifier position** — e.g., using
  `deliver` as an effect verb name.

### 25.2 Lowering errors

Reported with offending source span + IR context. Categories:

- **Unknown verb** — `unknown effect verb 'teleport'` (and a suggestion
  list if close match exists).
- **Unknown material** — `unknown material 'sandstone' (not in .sim catalog)`.
- **Unknown structure** — `unknown structure 'OrcKeep'`.
- **Arity mismatch** — `damage expects 1 argument, got 2`.
- **Budget exceeded** — `ability 'X' has 5 effects (max 4)`, `ability 'X'
  places 20000 voxels (max 16384)`.
- **Status reserved** — `verb 'polymorph' is reserved; engine does not
  implement it yet`.
- **Structure parameter mismatch** — `Castle expects wall_mat: Material,
  got int`.
- **Template recursion** — `template 'a' calls 'b' calls 'a' (cycle)`.
- **Duplicate name** — `duplicate ability / template / structure name 'X'`.

### 25.3 Runtime errors

The engine never panics on a cast-time invariant; violations emit
diagnostic events and fail the cast cleanly. Examples:

- `CastDepthExceeded` — already ships for `MAX_CASCADE_ITERATIONS`.
- `CastBudgetSpilled` — cast spilled into later tick.
- `VoxelMaskUnloaded` — registry miss (should never happen in a clean
  build; diagnostic only).

### 25.4 Unhandled-construct policy

- Author uses a `reserved` construct → lower-error (build fails).
- Author uses a `planned` construct whose engine side isn't yet wired
  → lower warning + runtime no-op (build succeeds; cast does nothing
  instead of the expected effect).
- Author uses a `runs-today` construct → zero friction.

---

## §26 Appendix: delta vs `PLAN.md`

Constructs this spec reshapes from `PLAN.md`:

### 26.1 Newly introduced

- `structure` top-level form (§16) — not in `PLAN.md`.
- `material` declaration in `.sim` (§17) — not in `PLAN.md`.
- 3D volume primitives (§9.2) — `PLAN.md` had 2D areas only.
- CSG composition (§9.4) — new.
- Voxel effect verbs + events (§18) — new.
- Voxel-aware conditions + triggers (§19) — new.
- AI-state manipulation verbs (§20) — new; `PLAN.md` only covered
  combat effect verbs.
- Unified AI-state condition surface (§21) — new.

### 26.2 Refined

- Shape unification (§9) — `PLAN.md` implied 2D; this spec makes every
  shape 3D with default disc thickness for the 2D family.
- Templates typed (§15) — `PLAN.md` had positional substitution only;
  this spec adds typed parameters with defaults.
- Template arithmetic (§15.3) — `PLAN.md` was pure substitution; this
  spec adds bounded arithmetic for compile-time evaluation inside
  structures (§16.5) and permits it in template args.
- Status markers (§23) — `PLAN.md` had no capability status; this spec
  makes it a first-class artifact.
- Tag system surface (§14) — `PLAN.md` had open-set; this spec
  acknowledges the engine's fixed 6-enum and marks widening as
  `planned` via a mapping table.

### 26.3 Deferred to `PLAN.md` (not owned here)

- Generative grammar + weighted sampling (`PLAN.md` §6).
- Transformer architecture + tokenization (`PLAN.md` §7).
- Training pipelines (pretrain / finetune) (`PLAN.md` §7.7).
- Grammar-constrained decoding (`PLAN.md` §7.8).
- Line-count / ML advantages discussion (`PLAN.md` §8, §10).

### 26.4 Out of scope entirely

- Quest creation / acceptance / completion / bidding as ability verbs —
  the quest subsystem is the economy; ability writes bypass economic
  deliberation. Abilities can only affect *valuation* (§20.5).
- Second-order beliefs — deferred until TOM Phase 3+.
- Arbitrary-scale structure placement — grid snap + `{1, 2, 4}`
  only.
- Arbitrary-rotation volume placement — 90° snap only for determinism.
- In-engine cooperative CSG evaluation — all CSG resolves at compile time.

---

*End of spec.*
