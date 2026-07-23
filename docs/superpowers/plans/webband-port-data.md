# Webband port — the load-bearing data slice (verbatim, with citations)

Extracted 2026-07-21 from `F:\MB` (read-only). Every number below is copied verbatim from the
TS sources with `file:line` citations, so porting agents do not each re-read the originals.
Line numbers refer to the files as of this extraction.

---

## 1. Colony catalogs — `F:\MB\src\colony\defs.ts`

### ITEMS (defs.ts:80-90)

| id | name | stackMax | value | nutrition | meal | spoilDays | medicinal |
|---|---|---|---|---|---|---|---|
| berries | Berries | 75 | 1 | 0.5 | — | 5 | — |
| grain | Grain | 75 | 1 | 0.25 | — | 40 | — |
| meal | Meals | 40 | 3 | 1 | true | 3 | — |
| timber | Timber | 75 | 2 | — | — | — | — |
| plank | Planks | 75 | 3 | — | — | — | — |
| herbs | Herbs | 50 | 2 | — | — | 10 | — |
| poultice | Poultices | 25 | 8 | — | — | — | true |
| hide | Hides | 50 | 4 | — | — | — | — |
| venison | Venison | 50 | 2 | 0.5 | — | 3 | — |

(colors/icons are render-only: berries 0x7a4a66 '❈', grain 0xc9ab5e '፨', meal 0xb3874a '◉',
timber 0x6b5138 '≡', plank 0x8a6a45 '▤', herbs 0x5e7a4a '❦', poultice 0x9aa06a '✚',
hide 0x8a6a52 '◗', venison 0x9a4f42 '◆'.)

### RECIPES (defs.ts:92-113)

| id | station | inputs | outputs | work (min) | job | keep |
|---|---|---|---|---|---|---|
| meal_from_grain | hearth | grain: 3 | meal: 2 | 30 | cook | 12 |
| meal_from_berries | hearth | berries: 4 | meal: 2 | 30 | cook | 12 |
| meal_from_venison | hearth | venison: 2 | meal: 2 | 30 | cook | 12 |
| saw_planks | workbench | timber: 1 | plank: 2 | 20 | craft | 16 |
| brew_poultice | workbench | herbs: 2 | poultice: 1 | 25 | craft | 4 |

`keep` semantics (defs.ts:37): "Standing bill: keep making while total stock of the primary
output < this." The primary output is `Object.keys(recipe.outputs)[0]` (jobs.ts:254).

### CROPS (defs.ts:119-123)

Comment at defs.ts:115-118: "Subsistence is SLOW by design (guild economy, 2026-07-20) …
One full-time farmhand should feed roughly a mouth and a half, not four."

| id | item | growDays | yield (units/cell) |
|---|---|---|---|
| grain | grain | 12 | 3 |
| berries | berries | 9 | 2 |
| herbs | herbs | 14 | 2 |

### Footprint primitives (defs.ts:142-149)

```ts
export const FOOT_1: { q: number; r: number }[] = [{ q: 0, r: 0 }];
export const FOOT_3: { q: number; r: number }[] = [
  { q: 0, r: 0 }, { q: 1, r: 0 }, { q: 0, r: 1 },
];
export const FOOT_7: { q: number; r: number }[] = [
  { q: 0, r: 0 }, { q: 1, r: 0 }, { q: 1, r: -1 }, { q: 0, r: -1 },
  { q: -1, r: 0 }, { q: -1, r: 1 }, { q: 0, r: 1 },
];
```

Rotation (defs.ts:283): "Rotate the axial offset 60° clockwise `rot` times: (q, r) → (-r, q+r)."
Validation (defs.ts:204-222): footprint must be contiguous and include its anchor `(0,0)`.
The anchor is NOT the centre — measure/draw from footprint cells (`footprintOf`), never `b.q/b.r`.

### BUILDINGS (defs.ts:151-170)

| kind | name | cost | work (min) | footprint | passable | station | joins | effects |
|---|---|---|---|---|---|---|---|---|
| hearth | Hearth | timber: 4 | 40 | FOOT_1 (default) | false | true | — | — |
| workbench | Workbench | timber: 6 | 60 | FOOT_1 | false | true | — | — |
| bed | Bed | plank: 2 | 40 | FOOT_1 | false | — | — | bed: true |
| cot | Infirmary cot | plank: 3 | 50 | FOOT_1 | false | — | — | cot: true |
| mess_table | Mess table | plank: 6 | 90 | FOOT_3 | false | — | — | table: true |
| store_shed | Store shed | plank: 9 | 120 | FOOT_3 | false | — | — | spoilMult: 2 |
| muster_ground | Muster ground | timber: 4, plank: 2 | 90 | FOOT_3 | false | — | — | drill: true |
| wall | Wall | timber: 1 | 15 | FOOT_1 | false | — | true | — |
| door | Door | plank: 2 | 30 | FOOT_1 | **true** | — | true | — |

- `SHED_REACH = 2` hexes (defs.ts:76) — store-shed preserving reach, measured from the
  NEAREST footprint cell (defs.ts:302-308). `spoilMult: 2` means stacks in reach keep 2× longer.
- Colony wealth (defs.ts:312-322): `gold + roster.length * 40 + Σ(item.value × count over stacks)
  + Σ(cost item value × n × 2 over BUILT buildings)`, rounded.

---

## 2. Needs & mood — `F:\MB\src\colony\needs.ts`

### THOUGHTS table (needs.ts:16-28)

| key | label | delta | days |
|---|---|---|---|
| ate_raw | Ate raw scraps | -4 | 1 |
| slept_rough | Slept on the ground | -6 | 1 |
| starving | Starving | -18 | 1 |
| hungry_road | Hungry on the road | -12 | 2 |
| came_home_to_ashes | Came home to ashes | -14 | 4 |
| festival | A festival day | +15 | 3 |
| victory | We drove them off | +8 | 2 |
| defeat | We were plundered | -10 | 3 |
| goal_served | Our purpose is served | +12 | 4 |
| home_served | We stood by my home ground | +10 | 3 |
| home_refused | We turned our backs on my home ground | -9 | 3 |

Dedupe rule (needs.ts:36): `addThought` skips if an event with the same `key` AND same `day`
already exists on the member. A thought is active while `day < e.day + e.days` (needs.ts:47).
Validation (needs.ts:29-31): every entry must have a label, `days > 0`, `delta !== 0`.

### moodWord ladder (needs.ts:42-45)

```ts
mood >= 75 ? 'bright' : mood >= 55 ? 'steady'
  : mood >= 40 ? 'weathered' : mood >= 25 ? 'grim' : 'breaking';
```

### Mood formula (needs.ts:121-132)

```
mood = clamp(0, 100, round(
  40 + 25*needs.food + 15*needs.rest + 10*needs.comfort + 10*needs.cheer
  + Σ(active thought deltas) + company))
company = 8 * (Σ standingOf(m, other) / (roster.length - 1))   // roster > 1 only
```

### Eating chain and drains (needs.ts:79-115, per dawn resolveNeeds; skips non-`ready` members, needs.ts:78)

- Eat priority: 1 meal (nutrition 1) → else raw: 2 berries × 0.5 → 2 venison × 0.5 →
  grain × 0.25 up to 1.0 (needs.ts:81-91). Any raw eating adds `ate_raw` (needs.ts:92).
- Fed: `needs.food = min(1, nutrition)`, `starvingDays = 0` (needs.ts:95-96).
- Nothing eaten: `needs.food -= 0.4` (floor 0), `hpFrac -= 0.15` (floor **0.05**),
  `starvingDays += 1`, add `starving` thought (needs.ts:98-101).
- Rest: bed owner → `rest = 1`; else `rest = 0.7` + `slept_rough` thought (needs.ts:107-112).
- Comfort: `0.4 + (bed ? 0.3 : 0) + (mess_table built ? 0.3 : 0)` (needs.ts:113).
- Cheer: `clamp(0,1, cheer - 0.15 + (supper ? 0.2 : 0))` (needs.ts:114-115).
- Recent breaks kept 7 days (needs.ts:119).

### healColonists (needs.ts:140-155)

Doc comment (needs.ts:137-139): "base rate 1, +1 with a surgeon home, +1 with a built cot,
+1 when a healer tended them today, +1 for a poultice (consumed, one per injured colonist per
night)". Applied as `injuryDays -= rate` (floor 0); on reaching 0, `hpFrac = max(hpFrac, 0.8)`
(needs.ts:150). Uninjured members heal `hpFrac += 0.25` per day, cap 1 (needs.ts:152).
Skips non-`ready` members (needs.ts:144).

---

## 3. Jobs & gathering — `F:\MB\src\colony\jobs.ts` (+ grid.ts)

### Constants (jobs.ts:24-52)

| const | value | meaning |
|---|---|---|
| WORK_DAY | 600 | jobs.ts:24 (one work unit = one minute; day = 600) |
| MOVE | 1.5 | min per hex step; "rough ground already doubled by the path" (jobs.ts:25-26) |
| HAUL_CAP | 30 | jobs.ts:27 |
| PICKUP | 5 | jobs.ts:28 |
| DROP | 5 | jobs.ts:29 |
| FETCH | 15 | flat ingredient-fetch cost added to recipe work (jobs.ts:31, used jobs.ts:227) |
| CHOP_WORK | 60 | jobs.ts:32 |
| CHOP_YIELD | 2 | timber per tree (jobs.ts:33) |
| TREE_REGROW | 25 | days (jobs.ts:34) |
| FORAGE_WORK | 80 | jobs.ts:35 |
| FORAGE_YIELD | 3 | berries per bush (jobs.ts:37) |
| BUSH_REGROW | 6 | days (jobs.ts:38) |
| HUNT_WORK | 100 | jobs.ts:40 |
| HUNT_MEAT | 4 | venison per kill (jobs.ts:41) |
| HUNT_HIDES | 1 | hide per kill (jobs.ts:42) |
| GAME_REGROW | 8 | days (jobs.ts:43) |
| KEEP_HIDES | 6 | hunt fires while hides < this even when fed (jobs.ts:45) |
| TEND_WORK | 30 | heal-tend job (jobs.ts:46) |
| SOW_WORK | 8 | jobs.ts:47 |
| TEND_CROP_WORK | 10 | once per day per sown cell (jobs.ts:48, gate jobs.ts:342) |
| HARVEST_WORK | 15 | jobs.ts:49 |
| KEEP_TIMBER | 30 | chop bill gate (jobs.ts:51) |
| KEEP_FOOD_UNITS | 30 | forage/hunt bill gate (jobs.ts:52) |

### Job work as priced in offers

- chop: `work: CHOP_WORK` (60), drops 2 timber at cell, stamps `felled[cell] = day + 25` (jobs.ts:184-193).
- forage: `work: FORAGE_WORK` (80), drops 3 berries, `foraged[cell] = day + 6` (jobs.ts:195-204).
- hunt: `work: HUNT_WORK` (100), drops 4 venison + 1 hide, `hunted[cell] = day + 8` (jobs.ts:206-216).
- recipe: `work: recipe.work + FETCH` at the built station (jobs.ts:227).
- build: `work: b.workLeft`, continuous, applied per-minute (jobs.ts:264-276).
- supply haul: `work: PICKUP + DROP` (10), count `min(need, HAUL_CAP)`; apply re-checks `b.needs`
  at landing (jobs.ts:281-295).
- stray-stack haul: `work: PICKUP + DROP + hexDist(src, dest) * MOVE` (jobs.ts:308).
- sow 8 / tend-crop 10 (once per day: `sown.tendedDay !== g.day`) / harvest 15 dropping
  `crop.yield` of `crop.item` (jobs.ts:326-347).
- heal-tend: `work: TEND_WORK` (30) at CACHE_GROUND until cots exist (jobs.ts:244-250).

### Bill gates (jobs.ts:352-372)

- Chop offered while `stockOf(timber) < 30`, at the 6 nearest tree cells.
- Food units = `Σ nutrition × count` over ALL stacks (generic over ITEMS, jobs.ts:358-359).
  Forage offered while `foodUnits < 30`, 6 nearest bushes.
- Hunt offered while `foodUnits < 30 OR stockOf(hide) < 6`, 4 nearest game cells.
- Recipe standing bills: offered while `stockOf(primary output) < recipe.keep` (jobs.ts:253-258).

### Stack mutations

- `takeItems` (jobs.ts:73-89): nearest stacks first (hexDist, then id order); removes empties.
- `dropItems` (jobs.ts:93-129): merge into same-item stack ≤ stackMax, **earlier spoilDay wins**
  (`min` merge, jobs.ts:107); else new stack; else deterministic ring spill out to ring 4.
  `spoilDay = g.day + def.spoilDays` at drop time (jobs.ts:98).

### Grid derivation constants (`F:\MB\src\colony\grid.ts`)

- `CLEARING_R = 6` — cells within this of the founding ground stay clear of derived growth (grid.ts:27).
- treeAt (grid.ts:72-80): needs grove fbm `fbm(q*0.13+31, r*0.13-17, 2) >= 0.56` AND
  `hash(q*3.7+11.3, r*3.1-7.9) > 0.5`; suppressed by felled-until day, water, rough, city, clearing.
- bushAt (grid.ts:85-93): allowed from `CLEARING_R - 2`; `hash(q*5.3-3.1, r*4.7+13.7) > 0.955`.
- gameAt (grid.ts:100-110): allowed from `CLEARING_R + 3`; grove fbm in `[0.45, 0.56)`
  (the wood's-edge band, same fbm as trees) AND `hash(q*7.1+23.9, r*6.3-11.1) > 0.93`.
- isRough: `groundSlope > 0.55` (grid.ts:65). Rough ground doubles MOVE via the path cost.
- Trees, bushes, and game cells are passable-as-ground but NOT standable (worked from beside;
  adjacency is `hexDist === 1`, never `<= 1`) (grid.ts:156-166).

### workSpeedFor coupling (`F:\MB\src\campaign\classes.ts:169-176`)

```ts
export function workSpeedFor(prog: CharProgress | undefined, job: JobKind): number {
  if (!prog) return 1;
  let mult = 1;
  for (const id of prog.learned) {
    if (WORK_PERKS[id]?.job === job) mult += 0.15;
  }
  return mult;
}
```

Applied in tick.ts:169-171: `spend = min(left, job.work / speed); done = spend * speed` —
i.e. a perk-holder pays fewer minutes for the same work.

---

## 4. Raids — `F:\MB\src\colony\raids.ts`

### raidBudget / raidTier (raids.ts:64-70) — verbatim

```ts
export function raidBudget(wealth: number, colonists: number, day: number): number {
  return Math.round(2 + colonists * 2 + wealth * 0.005 + day * 0.25);
}

export function raidTier(wealth: number, day: number): number {
  return Math.max(1, Math.min(7, 1 + Math.floor((wealth + day * 12) / 600)));
}
```

Measured tuning table in the doc comment (raids.ts:49-51), win rate for three unwalled founders:
budget 6/8/10/12/14/16/20 → foes 3.1/4.5/5.2/5.9/7.0/7.5/9.6 → win% 100/100/100/75/33/13/0.

### rollComp (raids.ts:26-38)

Greedy spend of `budget` power: while `left >= TROOPS.looter.power` (guard 200 iterations),
pick uniformly (seeded rng) among affordable of `['looter', 'bandit', 'raider']`, subtract its power.

Troop powers (`F:\MB\src\campaign\data.ts:26-46`): looter power **1** (hp 42, dmg 11, loot 9),
bandit power **2** (hp 72, dmg 17, loot 20), raider power **3.5** (hp 105, dmg 24, loot 38),
warlord power **12** (hp 340, dmg 34, loot 0). (`elite_leader` is a KITS identity, not a TROOPS row.)

### spawnRaid (raids.ts:74-104)

- `tier = min(7, raidTier(wealth, day) + tierBump)`.
- `comp = rollComp(g, raidBudget(wealth, roster.length, day) * (1 + tier * 0.15))` (raids.ts:83).
- `eliteName`: 50% chance of a generated named elite when not supplied (raids.ts:85-86).
- `arrivesDay = g.day + 1` — one night's warning (raids.ts:87).
- `entryDir = rngInt(0,5)`; city scenario remaps 2→1 and 3→4 (roll always happens — seeded
  stream stays byte-identical across scenarios) (raids.ts:92-96).

### Timeout / withdraw (`F:\MB\src\battle\engine.ts`)

- `ROUND_CAP = 40` (engine.ts:93). On `round > ROUND_CAP`:
  `finished = timeoutWins ? 'victory' : 'defeat'` (engine.ts:324-325, 494-495).
- Colony raids are staged with `{ fieldR: g.colony.radius, blocked: blockedHexes(g),
  timeoutWins: true }` — timeout means the raiders withdraw, defender-favorable
  (colony-screen.ts:474; headless-sim.ts:342).

### resolveRaid victory (raids.ts:137-164)

- `gold += result.gold`; `renown += 4 + tier * 2` (raids.ts:138-139).
- threatRef removed from `g.threats`; bandRef camped band is signed via cause + goal satisfied.
- Hostile author faction: latch cleared, `moveStanding(g, factionId, 20)` (raids.ts:149-153).
- Fielded members of signed bands: `patience = min(100, patience + 8)` (raids.ts:131-134).
- Injury on homecoming: `hpFrac < 0.4 → injuryDays = max(injuryDays, 2)` (raids.ts:117-118).
- Everyone gets the `victory` thought (raids.ts:160).

### plunder (raids.ts:179-228) — defeat ('beaten') or nobody home ('undefended')

```ts
const heavy = severity === 'undefended';
const take = (10 + raid.tier * 8) * (heavy ? 2 : 1);   // units of stock taken, by value desc
const spare = g.roster.length * 7;                      // a week of food floored per roster head
```

- Stacks sorted by item value descending; per-stack food floor
  `ceil(spare / nutrition / max(1, stacks.length))` for edibles (raids.ts:190-196).
- Burns the `heavy ? 3 : 2` outermost BUILT buildings (sorted by `|q|+|r|` descending, raids.ts:199-203).
- `renown -= heavy ? 6 : 4` (floor 0) (raids.ts:205).
- Threat removed either way; raiders always MOVE ON (no siege loop) (raids.ts:206-209).
- `director.points -= 20` (floor 0); **relief window** `director.reliefUntil = day + (heavy ? 4 : 3)`
  (raids.ts:210-211).
- Thoughts: present members get `defeat`, away members `came_home_to_ashes` (raids.ts:215-217).

---

## 5. Minds tuning — `F:\MB\src\minds\tuning.ts` (entire constant table)

**These are PER-DAY rates** (the clock ticks once per day via `MindPort.onDayEnd`;
decay is lazy on read: `effConf = baseConf * ATTR_DECAY[attr]^(days elapsed)`, tuning.ts:20-24).
The port must convert to per-tick — see Conversion notes at the end. Do not port the numbers
as per-tick values.

```ts
// ── capacities ──
export const SUBJECT_CAP = 24;   // distinct subjects per mind (source: BELIEF_CAP 25)
export const FACT_CAP = 160;     // facts per mind; evict lowest effective confidence
export const MEMORY_CAP = 12;    // episode ring (source: 8; battles are coarser than ticks)
export const SAGA_CAP = 64;
export const CLOSED_SAGA_TTL = 15; // DAYS a closed saga lingers (was 6 battles pre-pivot)
export const CHRONICLE_CAP = 200;

// ── per-DAY lazy decay ── (tuning.ts:25-32)
export const ATTR_DECAY: number[] = new Array(N_ATTRS).fill(0.97);
ATTR_DECAY[A_ACQUAINT] = 0.976;
ATTR_DECAY[A_STANDING] = 1.0; // standing decays via its own retention curve below
ATTR_DECAY[A_HOSTILE] = 1.0;
ATTR_DECAY[A_THREAT] = 0.959;
ATTR_DECAY[A_OWES] = 1.0;
ATTR_DECAY[A_NOTORIETY] = 0.97;
ATTR_DECAY[A_INTENT] = 1.0;

// Standing value retention per DAY: base + span·|st|/32768 (tuning.ts:34-37)
export const STANDING_RETAIN_BASE = 0.982;
export const STANDING_RETAIN_SPAN = 0.018;

// ── gossip ── (tuning.ts:39-44)
export const RUMOUR_FADE = 0.85; // confidence multiplier per retelling
export const MAX_HOPS = 3;       // beyond thirdhand it's noise
export const AFFINITY_GAIN = 600; // standing per peaceful camp phase…
export const AFFINITY_CAP = 8192; // …capped well short of BOND_BAR: camp time never forges bonds

// ── witness-fold magnitudes ── (tuning.ts:46-52)
export const SOUR_VICTIM = 3500;    // toward the one who felled me (AVENGER_SOUR)
export const SOUR_WITNESS = 2000;   // toward the one who felled a comrade
export const WARM_SAVED = 4000;     // toward the one who pulled me back
export const WARM_SAVE_WITNESS = 1500;
export const WARM_SLEW_FOE = 800;   // admiration for the one who dropped the named foe
export const THREAT_PER_KILL = 1500;
export const NOTORIETY_PER_LEVEL = 1200;

// ── episode salience ── (tuning.ts:54-61)
export const SAL_STRUCK_DOWN = 55000;
export const SAL_SUCCOURED = 50000;
export const SAL_WITNESSED_FALL = 48000;
export const SAL_SLEW = 45000;
export const SAL_AVENGED = 52000;
export const SAL_SAVED = 40000;
export const SAL_WINDFALL = 20000;

// ── thresholds ── (tuning.ts:63-69)
export const CLUTCH_HP_FRAC = 0.25; // a heal landing below this is a rescue
export const WINDFALL_GOLD = 120;
export const BOND_BAR = 12000;      // mutual standing that opens a Comrades saga
export const FEUD_BAR = -10000;
export const CONF_WITNESSED = 60000;
```

Module-load invariant asserts (tuning.ts:72-83), port these as validation:

1. Every `ATTR_DECAY[i]` in `(0, 1]`.
2. `ATTR_DECAY[A_OWES] === 1 && ATTR_DECAY[A_HOSTILE] === 1`
   ("debts and grudges are settled by deeds, not forgotten by time").
3. `AFFINITY_CAP < BOND_BAR` ("camp affinity alone must never cross BOND_BAR").

Design law (tuning.ts:1-6): bounded forgetting is a survival mechanism — belief cap ~25 was a
measured optimum; caps ≥ 100 made hostility metastasize. Retention is keyed on STANDING,
never on the hostile latch.

---

## 6. Witness folds — `F:\MB\src\minds\witness.ts`

### foldBattle deltas (witness.ts:56-191)

| event | who | effect |
|---|---|---|
| ally felled | victim | memory salience SAL_STRUCK_DOWN, `sour(actor, SOUR_VICTIM=3500, latch=namedActor)` (witness.ts:89-91) |
| ally felled | still-standing witnesses | memory SAL_WITNESSED_FALL, `sour(actor, SOUR_WITNESS=2000, latch=namedActor)` (witness.ts:92-96) |
| ally felled by `foe:` actor | victim | vendetta saga opened (witness.ts:98-100) |
| ally kills enemy | witnesses | `bump(actor, A_THREAT, THREAT_PER_KILL=1500)` (witness.ts:104-106) |
| named foe slain | slayer + witnesses | slayer memory SAL_SLEW; witnesses `warm(actor, WARM_SLEW_FOE=800)`; vendettas close, `closeSubject(foe)` (witness.ts:107-124) |
| clutch heal (targetHpBefore < CLUTCH_HP_FRAC=0.25, actor≠target) | saved | memory SAL_SUCCOURED, `warm(actor, WARM_SAVED=4000)`, `addDebt(actor, 1)`, life_debt saga (witness.ts:127-147) |
| clutch heal | witnesses | `warm(actor, WARM_SAVE_WITNESS=1500)` (witness.ts:133-135) |
| reciprocal save (savior owed target) | savior | memory SAL_SAVED, debt settled, life_debt saga closed (witness.ts:136-142) |
| victory with `gold >= WINDFALL_GOLD=120` | all participants | memory SAL_WINDFALL (witness.ts:152-156) |
| mutual standing ≥ BOND_BAR both ways | pair | comrades saga + bond_forged beat (witness.ts:159-169) |
| class level-up | other participants | `bump(key, A_NOTORIETY, NOTORIETY_PER_LEVEL=1200 * level)` (witness.ts:172-177) |

**Latch rule**: the hostile latch is passed as the third arg of `sour(...)` and is
`namedActor = d.actor.startsWith('foe:')` (witness.ts:84) — grudges latch ONLY for named foes.

### foldIncident (witness.ts:211-255) — the non-combat bridge

| kind | deltas |
|---|---|
| brawl | both parties: memory `SAL_STRUCK_DOWN * 0.6`, `sour(other, SOUR_VICTIM * 0.6, **false**)`; witnesses `sour(INSTIGATOR=actor, SOUR_WITNESS * 0.5, **false**)` (witness.ts:220-230) |
| rescue (sickbed tend) | saved: SAL_SUCCOURED memory, `warm(actor, WARM_SAVED)`, `addDebt(actor, 1)`, life_debt saga; witnesses `warm(actor, WARM_SAVE_WITNESS)` (witness.ts:232-240) |
| windfall | every witness: SAL_WINDFALL memory; 'feast' beat (witness.ts:242-246) |
| prowess | witnesses: `bump(actor, A_NOTORIETY, NOTORIETY_PER_LEVEL * (magnitude ?? 1))` (witness.ts:248-252) |

**Colonist-on-colonist grudges never latch**: every `sour(...)` in foldIncident passes
literal `false` for the latch flag (witness.ts:225-228) — only battle deeds by `foe:` actors latch.

---

## 7. Storyteller — `F:\MB\src\guild\director.ts`

### Budget & cadence

- `POINT_CAP = 120` (director.ts:39); `COOLDOWN_DAYS = 3` (director.ts:40).
- Accrual, once per day (director.ts:84-86):
  ```ts
  d.points = Math.min(POINT_CAP,
    d.points + 2 + Math.ceil(g.roster.length / 2) + Math.floor(wealth / 800)
    + (mood > 60 ? 2 : 0));
  ```
- Fire gates (director.ts:88-90): no event while `day < reliefUntil`, while
  `day - lastEventDay < COOLDOWN_DAYS`, or while a raid is pending ("one storm at a time").

### Mercy gate (director.ts:95)

```ts
const mercied = (g.roster.length <= 2 || mood < 30) && wealth <= 2000;
```

### Trope table — COSTS (director.ts:42-54) and WEIGHT (director.ts:132-135)

| trope | cost | weight | eligibility (director.ts:97-121) |
|---|---|---|---|
| raid | 60 | 3 | `!mercied` |
| petition | 20 | 3 | no open petition, `!mercied`, petitioners exist, `availableHands >= 2` |
| warband | 50 | 2 | `!mercied` and no existing threats |
| wanderer | 35 | 1 | no current guest and guest pool non-empty (canRecruit) |
| feud | 30 | 3 | open vendettas exist and `!mercied` |
| cause_raid | 30 | 3 | a band's cause was requested |
| refugee_band | 30 | 1 | canRecruit and a camped band exists |
| festival | 30 | 1 | always |
| blight | 25 | 1 | ≥ 8 sown growth cells |
| windfall | 25 | 1 | always |
| caravan | 25 | 2 | no caravan camped; mercantile power not hostile |

### The plan (DirectorState.plan, director.ts:126-147)

Priority overrides first: `cause_raid` if eligible and affordable, else `feud` if feuds exist,
eligible and affordable (director.ts:128-129). Otherwise, if no plan or the plan went
ineligible, draw a new plan from the WEIGHTED pool (weight = pool repetitions) with the seeded
rng (director.ts:130-142). Then: `if (d.points < COSTS[d.plan]) return null; // saving up`
(director.ts:143) — the storyteller SAVES TOWARD the committed plan. On firing: plan cleared,
`points -= cost`, `lastEventDay = day` (director.ts:144-147).
ALL rng draws happen inside the committed trope's case, never in eligibility (director.ts:284-287).

### Trope payloads

- raid: author = hostile faction ?? wild power; `spawnRaid({factionId})`; `reliefUntil = day + 4` (director.ts:154-159).
- feud: `spawnRaid({eliteName: foe, sagaRef: 'vendetta:<a>:<foe>'})`; `reliefUntil = day + 4` (director.ts:172-174).
- cause_raid: `spawnRaid({eliteName: goal.foeName, bandRef})`; `reliefUntil = day + 4` (director.ts:188-191).
- warband: spawns at radius 85 (random angle), walks owPath to the hall;
  `comp = rollComp(g, 20 + colonyWealth * 0.02 + day * 0.3)`, tier 1; drums file a first
  (inexact) threat report (director.ts:203-231). Advances ~2 tiles per day (`stepIdx += 2`,
  director.ts:376); pillages settled landmarks within 12 world units of its step, once per band
  (director.ts:380-390); arrival = `spawnRaid({threatRef, tierBump: 1})` (director.ts:391-396);
  a warband engaged by the current raid holds its ground (director.ts:375).
- wanderer: guest with `leavesDay = day + 2` (director.ts:235).
- refugee_band: `desperateUntil = day + 3` (director.ts:249); (sign price ×0.4 lives in goals.ts).
- blight: kills `max(1, round(sownCells * 0.4))` sown beds (director.ts:258-263).
- festival: everyone gets the `festival` thought and `cheer = 1` (director.ts:272-275).
- caravan: 3 wares drawn without replacement from
  `[grain 18-30, timber 10-20, meal 6-10, herbs 5-10, poultice 2-4, plank 8-14]`;
  purse `gold: 60 + rngInt(0, 80)`; `leavesDay = day + 2` (director.ts:323-337).
- windfall: drops 6 meals at (10,-10) and 8 timber at (11,-10), `gold += 40` (director.ts:355-359).

### Starvation exodus (in `F:\MB\src\colony\colony-day.ts:197-217`)

Non-founders walk after **3** hungry days (`starvingDays`); founders hold to **6**
(colony-day.ts:198-208: `const bar = isFounder ? 6 : 3`). A signed non-founder band walks
TOGETHER (`departBand`); members afield are never removed mid-journey (colony-day.ts:205).
Empty roster = the guild fall.

---

## 8. Trade & scenarios — `F:\MB\src\colony\trade.ts` + `F:\MB\src\guild\scenario.ts`

### Price spread (trade.ts:20-31)

```ts
export const SELL_MULT = 0.6;
export const BUY_MULT = 1.5;
sellPrice = Math.max(1, Math.floor(value * SELL_MULT));   // what the trader pays you
buyPrice  = Math.max(1, Math.ceil(value * BUY_MULT));     // what the trader charges you
```

Home-stall selling uses the same `floor(value * 0.6)` spread (trade.ts:117); home buying uses
the market's posted price (trade.ts:135). Bought goods land AT THE CAMP (`caravanSpot`,
preferred cell `{q: 8, r: -2}` NE of the cache ground, ring-scanned for standable, trade.ts:37-51);
home purchases land at CACHE_GROUND (trade.ts:142).

### Caravan purse/duration (director.ts:334-337)

`gold: 60 + rngInt(g, 0, 80)`, camped `arrivedDay = day`, `leavesDay = day + 2`.
Departure with business done bumps standing +4 with its power (colony-day sweep — see
CLAUDE.md trade.ts section). Sell capped by the trader's purse, buy by their packs and your
gold (verdicts, trade.ts:55-70).

### Scenario specs (scenario.ts:68-126) — verbatim

| field | village | town | wilderness | city |
|---|---|---|---|---|
| recruiting | true | false | false | true |
| gold | 90 | 420 | 25 | 200 |
| stock | meal 14, timber 10, plank 4 | meal 26, timber 18, plank 12, herbs 6 | meal 6, timber 4 | meal 12, timber 8 |
| standing | +10 | -30 | 0 | 0 |
| tradePerDay | 2 | 14 | 0 | 0 |
| mealPrice | 3 | 3 | 0 (no market) | 3 |
| signDiscount | 0.55 | 1 | 1 | 0.8 |
| rentPerDay | 0 | 0 | 0 | 8 |

- `DEFAULT_SCENARIO = 'village'` (scenario.ts:136).
- applyScenario runs LAST after all founding rolls and takes ZERO draws (comparability law,
  scenario.ts:138-161). Stock is dropped as stray stacks at CACHE_GROUND (the resource cache).
  Default provisioning order: `g.provisioning = mealPrice > 0 ? 4 : 0` (scenario.ts:160).
- Rent (scenario.ts:182-197): paid silently while the purse holds; unpaid →
  `moveStanding(holder, -3)` + chronicle 'Rent unpaid'. Never a scripted eviction.
- tradeIncome (scenario.ts:201-207): `gold += tradePerDay` at dawn; 0 if home settlement sacked.
- canRecruit (scenario.ts:171-176): scenario flag AND home settlement not sacked.

---

## 9. Cast generation — `F:\MB\src\campaign\castgen.ts`

### Name constraints (castgen.ts:49-55, asserted castgen.ts:695-701)

`acceptName` — a name is rejected if, against every taken name (case-insensitive):
- shares a 4-char prefix (`n.slice(0, 4) === s.slice(0, 4)`) — banter recovers speakers by 4-char prefix;
- nests (`n.includes(s) || s.includes(n)`) — prose name-lookup does substring replaceAll;
- is in `RESERVED = {looter, bandit, raider, warlord, elite_leader, hero, founding}` (castgen.ts:19)
  or `BANNED_NAMES = {alien, marys}` (castgen.ts:22).

Ids are the lowercased name and must match `/^[a-z][a-z0-9-]*$/` (DOM-safe, castgen.ts:684).
Band fore-names share the same taken pool (a companion name inside a band name would linkify
mid-word). Roll: 40 seeded retries, then a deterministic fixed-order scan (castgen.ts:60-73).

### Band structure (castgen.ts:446-458)

```ts
const nBands = rngInt(g, 4, 5);
const sizes = [rngInt(g, 3, 4)];                       // founders: 3-4
for (let i = 1; i < nBands; i++) sizes.push(rngPick(g, [2, 3, 3, 4]));
let nFree = rngInt(g, 2, 3);                            // freelancers
// Deterministic clamp (no draws) to total 14-18.
```

Asserted (castgen.ts:722-733): every band ≥ 2 members; founding band 3-4; cast total 14-18;
exactly one `founders` band; founders hold `goalKind: 'guild'`.

### Goals & wants

- Band goal kinds: founders always `'guild'`; others `rngPick(['deed', 'deed', 'prosperity', 'debt'])`
  (castgen.ts:474); ≥ 1 non-founding deed band forced (castgen.ts:527, asserted :731).
- ≤ 1 band carries a poach want targeting a freelancer, rolled at p=0.35 per non-founding band
  until one exists (castgen.ts:475-477, resolved :526); target must be a freelancer (asserted :726-728).
- Freelancer wants (castgen.ts:515-519): renown `rngInt(4,8) * 25`, or ground tag, or a
  non-founding band signed. Freelancer hireCost `rngPick([150, 200, 250, 300])` (castgen.ts:514);
  band member cost `b === 0 ? 100 : 100 + b * 50 + rngPick([0, 50])` (castgen.ts:500).

### Look-kit uniqueness (castgen.ts:602-670, asserted :703-716)

- `HUE_BAR = 25` degrees (castgen.ts:602): every companion pair must differ by ≥ 25° hue OR
  headwear family OR build ("look-twins" assert, castgen.ts:703-711).
- Headwear never repeats within a band; freelancers count as one group (castgen.ts:629-641,
  asserted :713-716). Headwear preference is id-hash derived (`featureRoll(idSeed(id), 5)`), the
  pass is pure and DRAW-FREE — the seeded rng stream is untouched.
- Build from hp: `hp < 88 ? 0 : hp < 112 ? 1 : 2` (castgen.ts:604).
- Band hues fan `±(m - (size-1)/2) * 24°` around band hue; band hues spaced `360/(nBands+1)`
  (castgen.ts:474 area: castgen.ts:473, :499); freelancer hues at `+f * 24°` past the band fan (:507, :513).

### Founders guarantees (castgen.ts:492-497, :531-545; asserted :717-725)

- Exactly one healer aboard: one founder slot is pinned to the `healer` archetype
  (`healerSlot = rngInt(0, size-1)`), the rest exclude healer (castgen.ts:493-497); assert
  requires `field_dressing` in a founder kit (castgen.ts:725).
- Perks: founders get one of `['tracker', 'trader', 'surgeon']` + one other perk
  (castgen.ts:542-545); remaining perks scatter archetype-fitted at p=0.7 each over non-founders
  (castgen.ts:546-549). An `engineer` with a 1-spec kit gains `ballista_bolt` (castgen.ts:540).
- Per-band dedup: no repeated temper or backstory at one fire (castgen.ts:412-426).
- Kit spec ids must exist in CATALOG; hooks must be ⊆ worldgen KIND_TAGS (castgen.ts:688-689).
- Archetype cap: ≤ 2 repeats of an archetype per band (castgen.ts:392-398).
- Cultures: member rolls band culture at p=0.7, else any (castgen.ts:495).

Archetype stat envelopes (castgen.ts:89-132), rolled uniformly then rounded (hp step 5,
speed/reach step 0.1, block step 0.05 — castgen.ts:381-389):

| arch | hp | damage | speed | reach | block |
|---|---|---|---|---|---|
| skirmisher | 80-95 | 20-24 | 6.0-6.4 | 1.9-2.1 | 0.25-0.4 |
| archer | 80-95 | 22-26 | 5.8-6.2 | 2.0-2.1 | 0.25-0.35 |
| bruiser | 125-150 | 24-30 | 5.5-5.9 | 2.2-2.4 | 0.45-0.55 |
| duelist | 105-125 | 26-30 | 5.8-6.2 | 2.2-2.4 | 0.4-0.55 |
| warden | 130-150 | 22-26 | 5.4-5.7 | 2.2-2.3 | 0.5-0.55 |
| healer | 78-100 | 16-19 | 5.4-5.6 | 2.0-2.1 | 0.2-0.35 |
| assassin | 68-80 | 28-32 | 6.2-6.5 | 1.9-1.9 | 0.25-0.35 |

---

## 10. Ability catalog (representative subset) — `F:\MB\src\battle\abilities\catalog.ts`

Spec defaults (catalog.ts:12-18): `area: 'self', areaR: 0, areaDeg: 0, delivery: 'instant'`.
Units: cooldowns are ROUNDS; ranges/areas are METERS (metric between hex centers — the catalog
was deliberately NOT re-quantized to hexes). Verbatim specs:

```ts
spec('power_strike', 'Power Strike', 'winds up a crushing blow', {
  target: 'enemy', range: 2.6, cooldown: 3,
  effects: [effect('damage', 46)],
}),                                                          // catalog.ts:24-27 (basic melee nuke)
spec('crippling_shot', 'Crippling Shot', 'puts an arrow through the knee', {
  target: 'enemy', range: 10, cooldown: 2, delivery: 'projectile',
  effects: [effect('damage', 28), effect('slow', 0.5, { dur: 2, when: 'on_hit' })],
}),                                                          // catalog.ts:50-53 (projectile)
spec('field_dressing', 'Field Dressing', 'binds wounds with practiced hands', {
  target: 'ally', range: 6, cooldown: 2,
  effects: [effect('heal', 40)],
}),                                                          // catalog.ts:54-57 (heal; the founders' healer marker)
spec('shield_wall', 'Shield Wall', 'plants her shield like a gate', {
  target: 'self', range: 0, cooldown: 4,
  effects: [effect('shield', 35, { dur: 4 })],
}),                                                          // catalog.ts:62-65 (shield)
spec('whirlwind', 'Whirlwind', 'whirls through the press', {
  target: 'enemy', range: 3.2, cooldown: 4, area: 'circle', areaR: 3.2,
  effects: [effect('damage', 30), effect('knockback', 2, { when: 'on_hit' })],
}),                                                          // catalog.ts:32-35 (circle AoE)
spec('cleaving_blow', 'Cleaving Blow', 'cleaves in a wide arc', {
  target: 'enemy', range: 2.8, cooldown: 3, area: 'cone', areaR: 2.8, areaDeg: 100,
  effects: [effect('damage', 38)],
}),                                                          // catalog.ts:40-43 (cone AoE)
spec('lunge', 'Lunge', 'lunges', {
  target: 'enemy', range: 7, cooldown: 3,
  effects: [effect('dash', 7), effect('damage', 34)],
}),                                                          // catalog.ts:28-31 (dash + strike)
spec('second_wind', 'Second Wind', 'finds a second wind', {
  target: 'self', range: 0, cooldown: 6,
  effects: [effect('heal', 45, { when: 'caster_hp_below' }), effect('shield', 25, { dur: 3 })],
}),                                                          // catalog.ts:36-39 (conditional self-heal)
spec('warlord_sweep', "Warlord's Sweep", 'sweeps his greataxe through the line', {
  target: 'enemy', range: 3.5, cooldown: 3, area: 'circle', areaR: 3.5,
  effects: [effect('damage', 40), effect('knockback', 2, { when: 'on_hit' })],
}),                                                          // catalog.ts:90-93 (warlord signature)
spec('ballista_bolt', 'Ballista Bolt', 'looses a bolt that argues with geometry', {
  target: 'enemy', range: 11, cooldown: 3, area: 'line', areaR: 11, delivery: 'projectile',
  effects: [effect('damage', 32)],
}),                                                          // catalog.ts:82-85 (line projectile, engineer grant)
```

(Not copied here but present in catalog.ts: expose_weakness, pocket_sand, quartermasters_brew,
sisters_knife, rally, steppe_charge, heroic_strike — same IR, same field set.)

### KITS — enemy identities (catalog.ts:101-105) — verbatim

```ts
export const KITS: Record<string, string[]> = {
  raider: ['cleaving_blow'],
  elite_leader: ['whirlwind', 'power_strike'],
  warlord: ['warlord_sweep', 'second_wind'],
};
```

`kitFor(key)` resolves companions first, then KITS keys; unknown spec ids degrade to no
ability, never a crash (catalog.ts:110-113).

---

## 11. Clock constants

- `Game.SPEEDS = [0, 6, 24, 90]` — colony minutes per real second by speed setting, 0 = paused
  (`F:\MB\src\core\game.ts:48`). `?calm` starts paused (game.ts:49).
- `MINUTES_PER_DAY = 600` — "A colony day is ten hours of working light"
  (`F:\MB\src\colony\tick.ts:29`).
- `STEP_MINUTES = 0.5` — the sim always advances in whole steps of this size (the determinism
  contract) (tick.ts:32). So a day is **1200 sim steps** of 0.5 min each.
- `WORK_DAY = 600` (jobs.ts:24) — same number, the jobs-layer alias.
- `COLONY_R = 60` — playable board hex radius, shared by the jobs sim, raid engine fieldR,
  and the map render (`F:\MB\src\guild\state.ts:516`).
- `CACHE_GROUND = { q: 0, r: 0 }` (state.ts:520) — the founding ground / muster / ward.
- timeOfDay word thresholds (fraction of day): < 0.22 early morning, < 0.45 morning,
  < 0.6 midday, < 0.82 afternoon, else evening (tick.ts:225-231).
- Dawn-fold work→progression divisor: worked minutes fold into the deed profile at `/120`
  (~5 profile points for a full 600-minute day, battle-comparable — CLAUDE.md, train directive;
  the fold lives in colony-day.ts).

---

## Conversion notes for the Rust port

1. **The day mapping.** One colony day = 600 minutes = 1200 fixed steps of `STEP_MINUTES = 0.5`.
   Decide the port's tick unit FIRST: if a tick = 1 minute, day = 600 ticks; if a tick = one
   0.5-min step, day = 1200 ticks. All jobs.ts work numbers are denominated in MINUTES
   (1 work unit = 1 minute), so a 1-minute tick ports them unchanged.

2. **Per-day → per-tick decay.** Every rate in `minds/tuning.ts` (ATTR_DECAY,
   STANDING_RETAIN_BASE/SPAN) and the asymmetric standing drift in petitions
   (0.5/day up from negative, 0.25/day down from positive) is PER-DAY. If the port decays
   per tick instead of lazily per day, convert multiplicative rates as
   `rate_tick = rate_day^(1/600)` (or `^(1/1200)` for 0.5-min ticks), and additive drifts as
   `delta_tick = delta_day / 600`. NOTE: the TS implementation never runs a decay pass at all —
   decay is **lazy at read** (`eff = base * rate^(days elapsed)`), which is both cheaper and
   exactly equivalent; consider porting the lazy idiom instead of converting.

3. **Three distinct clocks, do not conflate them:**
   - Colony sim: minutes (600/day), fixed 0.5-min steps.
   - Minds: whole DAYS (`MindsState.clock`, ticked once per day by `MindPort.onDayEnd`;
     battles stamp the current day, never increment).
   - Battle: ROUNDS (cooldowns/status durations are rounds; `ROUND_CAP = 40`).
   Day-resolved systems (needs, healing, spoilage, growth, storyteller, regrow days) fold ONCE
   at dawn regardless of tick rate.

4. **Unit mismatches noticed:**
   - Battle ability ranges/areas are METERS (true distance between hex centers) while movement
     is HEX-native (`moveBudget = round(speed/2)` hexes) — the catalog was deliberately not
     re-quantized. Port both units as-is or re-derive the tile display from metric truth.
   - Colony MOVE cost is minutes-per-hex (1.5, rough ×2 via the path); battle movement is
     hexes-per-round. Same board geometry, different currencies.
   - `raidBudget` is in TROOP POWER units (looter 1 / bandit 2 / raider 3.5 / warlord 12);
     `plunder`'s `take` is in STACK UNITS (item counts, taken by value); `colonyWealth` is in
     GOLD-equivalents. Three different scales flowing through the same difficulty axis.
   - Minds magnitudes are i16-scale integers (standing ±32768 range; STANDING_RETAIN uses
     `|st|/32768`); mood is 0-100; needs are 0-1 fractions; hpFrac is 0-1. Keep the i16 scale
     for minds or re-derive BOND_BAR/FEUD_BAR/AFFINITY_CAP proportionally.
   - Spoilage: `spoilDay` is an absolute day stamped at DROP time (`g.day + spoilDays`), and
     merges keep the EARLIER spoilDay; shed reach multiplies the remaining window ×2. Not a
     per-tick freshness decay.
   - `workSpeedFor` divides minutes (work done per minute = speed), not the job's work total —
     `spend = work/speed` minutes of wall-clock, `done = spend * speed` work landed (tick.ts:169-171).
