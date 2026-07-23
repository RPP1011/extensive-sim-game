# Webband ability subset — S5-prep port notes

Ported 2026-07-21 from `F:\MB\src\battle\abilities\catalog.ts` (IR:
`F:\MB\src\battle\abilities\ir.ts`), the 10-spec representative subset listed
in `docs/superpowers/plans/webband-port-data.md` §10. Verified to parse and
lower by `crates/dsl_compiler/tests/webband_abilities.rs` (walks this
directory; asserts per-ability op counts, shapes, cooldown ticks, and LF-only
files).

Files:

- `webband_catalog.ability` — all 10 abilities (Webband's catalog is one file;
  keeping one file keeps the names one compilation unit, like the TS).
- `raider.class` / `elite_leader.class` / `warlord.class` — the KITS enemy
  identities (see "Kits" below).

## The clock conversion: 1 round = 2 seconds

Webband battles are turn-based; cooldowns and status durations are ROUNDS
(`ir.ts:9-10`), with a 40-round cap. This engine is real-time at 10 Hz
(`gate.cooldown_ticks = ceil(ms/100)`). Declared conversion: **1 round = 2 s**
(a 40-round Webband fight ≈ an 80 s engagement; ability tempo relative to the
fight length is preserved). Ranges/areas stay meters, verbatim — Webband kept
them metric on purpose (never re-quantized to hexes).

| Webband quantity | rounds | seconds | ticks (10 Hz) |
|---|---|---|---|
| power_strike cooldown | 3 | 6s | 60 |
| crippling_shot cooldown | 2 | 4s | 40 |
| crippling_shot slow dur | 2 | 4s | 40 |
| field_dressing cooldown | 2 | 4s | 40 |
| shield_wall cooldown | 4 | 8s | 80 |
| shield_wall shield dur | 4 | 8s | 80 |
| whirlwind cooldown | 4 | 8s | 80 |
| cleaving_blow cooldown | 3 | 6s | 60 |
| lunge cooldown | 3 | 6s | 60 |
| second_wind cooldown | 6 | 12s | 120 |
| second_wind shield dur | 3 | 6s | 60 |
| warlord_sweep cooldown | 3 | 6s | 60 |
| ballista_bolt cooldown | 3 | 6s | 60 |

## Per-spec mapping (Webband op → engine verb)

Original numbers in brackets; everything not bracketed is verbatim.

| Webband spec | ability name | mapping |
|---|---|---|
| `power_strike` | `WebbandPowerStrike` | `damage 46` → `Damage`(0). enemy/range 2.6 → `Area::SingleTarget`. |
| `crippling_shot` | `WebbandCripplingShot` | `deliver projectile { on_hit { damage 28; slow 0.5 for 4s } }` → `Delivery::Method{Projectile}` with one `OnHit` hook: `Damage`(0) + `Slow`(4). Webband `when:'on_hit'` = the hook. `speed`/`width` params have no Webband IR counterpart (projectile speed was visual-only, `ir.ts:41`); 12 m/s per corpus convention. Slow factor is a speed MULTIPLIER in both engines (Webband `slowMult = amount`, engine `factor_q8` "51 ≈ 0.2× speed") — 0.5 ports unchanged. |
| `field_dressing` | `WebbandFieldDressing` | `heal 40` → `Heal`(1). `target: ally` lowers (status `planned` for the runtime mask, per spec §5.1). |
| `shield_wall` | `WebbandShieldWall` | `shield 35 for 8s` → `TimedShield`(22) [orig dur 4r]. The timed variant is the faithful one — Webband shields expire; bare `Shield`(2) never decays. |
| `whirlwind` | `WebbandWhirlwind` | `damage 30 in circle(3.2)` + `knockback 2.0 in circle(3.2)` → `Damage`(0) + `Knockback`(14), each with `per_effect_areas` Circle r=3.2. Webband's `knockback ... when:'on_hit'` gated on the block roll leaving dmg > 0 — this engine has no block roll, the gate is vacuously true, ported unconditional. |
| `cleaving_blow` | `WebbandCleavingBlow` | `damage 38 in cone(50.0, 2.8)` → `Damage`(0) + Cone shape. **Arg order**: the shipped cone apply reads `args[0]` = HALF-angle deg, `args[1]` = range (`crates/engine/src/ability/apply.rs:955-958`), despite spec §9.1's `cone(radius, angle)` surface — authored to the apply convention. Webband areaDeg 100 is the full opening (it also halves: `cosHalf`, `engine.ts:1041`) → half-angle 50. |
| `lunge` | `WebbandLunge` | `dash 7.0` + `damage 34` → `Dash`(12) + `Damage`(0), order preserved (Webband resolves the dash pre-strike, `engine.ts:1096`). `dash to_target` (INFINITY sentinel) is the intent-first alternative; the literal magnitude (= cast range) was kept. |
| `second_wind` | `WebbandSecondWind` | `heal 45 when self.hp < 50` + `shield 25 for 6s` → `Heal`(1) with a when-predicate + `TimedShield`(22) [orig dur 3r]. **Approximation**: Webband's `caster_hp_below` is fractional — hp < 0.5·maxHp (`SELF_CAST_HP_FRAC = 0.5`, `engine.ts:94`); this engine's when-vocab is `<binder>.<field> <op> <literal>` with absolute literals only → hp < 50, i.e. 50% of the ~100-hp companion frame (Webband companion hp envelopes 68-150; warlord 340 — for a warlord port the threshold should be re-derived). See gaps. |
| `warlord_sweep` | `WebbandWarlordSweep` | as whirlwind at 40 dmg / r 3.5. |
| `ballista_bolt` | `WebbandBallistaBolt` | `damage 32 in line(11.0, 1.8)` → `Damage`(0) + Line shape, INSTANT. The projectile flavor was deliberately dropped: Webband's delivery is visual-only, the line AoE is the load-bearing part, and hook-stmt `in <shape>` modifiers are silently dropped by the lowerer today (`program.rs` `Delivery::Method` doc) — a projectile port would lose the line. Width 1.8 = Webband's fixed half-width 0.9 doubled (`engine.ts:1056`; engine line `args[1]` is full width, `apply.rs:1190-1191`). |

Choices applied uniformly:

- **No tags** (`[PHYSICAL: n]` etc.): Webband's IR has no tag concept; `hint:`
  carries the AI category (damage/heal/defense), matching Webband's own
  classifier predicates (`isOffensive`/`isSelfSupport`/`isAllySupport`).
- **No `cast:` header**: Webband casts are instant (turn-based); omitted rather
  than inventing wind-ups.
- **All `chance` = 1** in the subset — no `chance` modifiers emitted.
- Names carry a `Webband` prefix: `ShieldWall`, `Whirlwind`, `Lunge`,
  `SecondWind` already exist elsewhere in `dataset/abilities/` and §4.3 makes
  duplicate names in one compilation unit a hard error.

## Kits — the enemy identities (KITS, catalog.ts:101-105)

Webband's `KITS` is a flat `Record<string, string[]>` of spec ids:

| identity | Webband kit | ported ability names |
|---|---|---|
| `raider` | `['cleaving_blow']` | `WebbandCleavingBlow` |
| `elite_leader` | `['whirlwind', 'power_strike']` | `WebbandWhirlwind`, `WebbandPowerStrike` |
| `warlord` | `['warlord_sweep', 'second_wind']` | `WebbandWarlordSweep`, `WebbandSecondWind` |

The `.ability` DSL has no kit-grouping construct; the engine-native grouping is
a hero/agent binding to ≤ `MAX_ABILITIES = 8` ability names (spec §4.4) — a kit
IS a name list, exactly Webband's shape. The `.class` format
(`dataset/abilities/classes/`) does group abilities, so the three identities are
mirrored as `raider.class` / `elite_leader.class` / `warlord.class` here — but
note **no crate parses `.class` files today** (verified: no reference to
`dataset/abilities/classes` or a `.class` loader anywhere in `crates/`), so
those files are corpus-convention documentation and THIS TABLE is the
authoritative grouping for the S5 fixture, which will bind the names at
agent-declaration time. Webband troop stats ride along for S5:
looter hp 42/dmg 11/power 1, bandit 72/17/2, raider 105/24/3.5,
warlord 340/34/12 (`F:\MB\src\campaign\data.ts`); `elite_leader` is a KITS
identity with no TROOPS row.

## Gap list — Webband IR vs the engine EffectOp catalog

Checked against `crates/engine/src/ability/program.rs` (pinned ordinals 0-45:
0 Damage, 1 Heal, 2 Shield, 3 Stun, 4 Slow, 5 TransferGold, 6 ModifyStanding,
7 CastAbility, 8 Root, 9 Silence, 10 Fear, 11 Taunt, 12 Dash, 13 Blink,
14 Knockback, 15 Pull, 16 Execute, 17 SelfDamage, 18 LifeSteal,
19 DamageModify, 20 DamageOverTime, 21 HealOverTime, 22 TimedShield, 23 Buff,
24 Summon, 25 Harvest, 26 PlaceVoxel, 27 Stealth, 28 Charm, 29 Grounded,
30 Suppress, 31 Reflect, 32-38 ToM (PlantBelief/Observe/Scry/Reveal/Disguise/
Decoy/EraseBelief), 39 TravelTo, 40 Recipe, 41 WearTool, 42 Propose,
43 Announce, 44 GainSkill, 45 CreateObligation).

Webband's 10 IR ops (`ir.ts:12-22`) and their fates:

| Webband op | engine mapping | status |
|---|---|---|
| `damage` | `Damage`(0) / hook | clean |
| `heal` | `Heal`(1) | clean |
| `stun` | `Stun`(3) | clean (not in the 10-subset) |
| `slow` | `Slow`(4) — same multiplier semantics | clean |
| `knockback` | `Knockback`(14) — meters both sides | clean |
| `dash` | `Dash`(12) (+ `to_target` sentinel) | clean |
| `shield` | `Shield`(2) / `TimedShield`(22) with dur | clean |
| `expose` (damage-taken amp, `exposeMult`) | `DamageModify`(19) is the same intent (duration + multiplier) | equivalent exists; not in the 10-subset (expose_weakness) |
| `riposte` (counterattack stance: answer melee swings taken with a counter-strike at `amount` mult, dur rounds) | **NO equivalent.** `Reflect`(31) reflects a FRACTION of damage taken — a passive mirror, not a counter-attack that rolls the striker's own dmg pipeline. Closest composite would lose the stance semantics. | **gap** — affects sisters_knife/rally-era specs when the full catalog ports; not in the 10-subset |
| `drain` (damage; half the dealt harm returns to caster as hp) | **NO single equivalent.** `LifeSteal`(18) is a timed buff on future damage, not a per-cast drain; `damage + heal` pair loses the dealt-amount coupling (heal would be flat, not half-of-dealt). | **gap** — not in the 10-subset |

Trigger/`when` gaps (Webband `ir.ts:32` `Trigger`):

| Webband trigger | engine mapping | status |
|---|---|---|
| `on_hit` (projectile) | deliver `on_hit` hook | clean |
| `on_hit` (instant — "not fully blocked") | none; engine has no block roll, gate vacuous | dropped, semantics preserved by vacuity |
| `caster_hp_below` (fractional: hp < 0.5·maxHp) | `when self.hp < <literal>` — ABSOLUTE only. No `hp_pct` in the `ScalingStatRef` 8-field predicate subset; field-vs-field (`self.hp < 0.5 * self.max_hp`) rejected (`WhenConditionUnsupported`, `ability_lower.rs` restricted leaf vocab). NOTE: the corpus's `caster_hp_below(25%)` atom syntax (tier5) PARSES but does NOT lower. | **approximated** at 50 absolute; per-body re-derivation needed when kits bind to real stat frames |
| `target_hp_below` | `when target.hp < <literal>` — same absolute-only caveat | expressible (approximated), not in the 10-subset |

Other engine-side facts that shaped the port:

- `MAX_EFFECTS_PER_PROGRAM = 4` — all Webband specs have ≤ 2 effects; fine
  (full-catalog note: every spec fits, LIMITS.effects = 6 in Webband but no
  shipped spec exceeds 4... verify at full-catalog time).
- Hook-stmt inner modifiers (in-shape/chance/stacking/when) are silently
  dropped at lowering (`program.rs` Method doc) — why `ballista_bolt` is
  instant+line and `crippling_shot` keeps its slow OUT of an `in` shape (the
  `for` duration is NOT a dropped modifier; verb dispatch consumes it).
- Delivery runtime semantics (projectile travel) are deferred engine work
  (#124/#125) — the hook IR is structured and lowers today, which is all
  S5-prep asserts.
- `.ability` files must be LF — the webband test asserts no `\r` in this dir
  (existing corpus dirs are CRLF and predate the rule; new files comply).
