# Engine Expressiveness Gaps — Mechanics That Cannot Be Expressed Today

> **Why this exists:** a session of "being more creative with the engine"
> made it obvious that the hard part isn't composing scenes, it's spotting
> which mechanics require engine work and which are hacks on top of the
> existing substrate. This doc catalogs the former — canonical game
> mechanics the engine's current IR / traits / physics cannot represent
> without genuine extension. Each entry lists (a) a concrete mechanic
> that needs it, (b) why the current state can't express it, and (c) the
> minimal-surface extension.
>
> Scope: substantive gaps. Skips "the thing exists as a field but isn't
> used yet" items where the work is wiring, not schema.

---

## 1. Area-of-effect abilities (Cone / Circle / Line)

**Canonical mechanic:** fire breath hits every hostile in a 90° cone. A
meteor strike damages everything in a 4m circle. A line-of-effect spell
hits agents in a 1m-wide beam between caster and point.

**Current state:** `crates/engine/src/ability/program.rs` defines
`Area` as a single-variant enum: `SingleTarget { range: f32 }`. Doc
comment explicitly says "Plan-2 adds `Cone`, `Circle`, and full AoE;
those variants land alongside their resolver code." Today every cast
hits exactly one target. I can hack it from the policy layer by issuing
N casts on N nearest hostiles, but each cast emits a separate
`AgentCast` event and consumes a separate cooldown — observable
differences from true AoE.

**Extension surface:**
- `Area::Cone { range: f32, half_angle: f32, facing: Vec3 }`
- `Area::Circle { radius: f32, center: Vec3 }`
- New target-resolution step in `CastHandler` that expands one Cast
  event into N EffectDamageApplied events (one per in-area hostile).
- Possibly a new `Event::EffectAreaResolved` for telemetry.

**Why it matters:** AoE is the defining primitive of every RTS / MOBA /
rogue-like. Not having it means "wizard archetype" isn't expressible.

---

## 2. Projectile delivery (travel time)

**Canonical mechanic:** an arrow / fireball takes N ticks to cross
distance D. If the target moves during flight, the projectile might
miss. An interceptor can destroy the projectile mid-flight.

**Current state:** `Delivery::Instant` is the only variant. Comment:
"Plan-2 ability work adds `Projectile` (travel-time) and `Zone`."

**Extension surface:**
- `Delivery::Projectile { speed: f32 }` — spawns an in-flight entity
  or a deferred resolution queue entry.
- A per-tick "in-flight projectiles" list on `SimState`, advanced each
  tick, resolving on arrival (or on intercept / miss).
- New `Event::ProjectileLaunched` / `ProjectileArrived` / `ProjectileIntercepted`.
- Mask primitives: `can_intercept(projectile_id)`.

**Why it matters:** turn-based projectiles collapse to instant because
you can't model a bullet that's in-flight across ticks. Counter-play
to ranged attacks requires this.

---

## 3. Persistent zones / aura effects

**Canonical mechanic:** a flame patch on the ground damages everyone
standing on it each tick for 10 ticks. A healing aura buffs kin within
5m. A poison cloud moves with wind.

**Current state:** no persistent spatial effects. EffectOp applies once
at cast resolution. `Delivery::Zone` is reserved but not implemented.

**Extension surface:**
- New entity kind `AggregatePool<ZoneEffect>` with `pos`, `radius`,
  `remaining_ticks`, `tick_effect: EffectOp` fields.
- Per-tick pipeline phase that iterates zones, applies effects to
  agents in radius.
- New `Event::ZoneSpawned` / `ZoneExpired` / `ZoneTicked`.

**Why it matters:** DoT ground effects are a staple of every combat
sandbox; without them you can't model fire/ice/poison terrain.

---

## 4. Movement modes (Fly / Swim / Climb / Fall)

**Canonical mechanic:** a flying unit can cross a river a ground unit
cannot. A climber scales a wall. A falling unit takes impact damage.

**Current state:** every agent moves the same way, as a Vec3 translation
toward a target point. Terrain doesn't exist at the sim level, so
"flying" today is just "initial z > 0" and the unit walks in 3D. The
DSL spec at `docs/dsl/spec.md` mentions a "movement-mode sidecar
(Walk | Climb | Fly | Swim | Fall)" and a "movement_mode ≠ Walk"
spatial-index partition, but the engine's `step.rs` move logic doesn't
branch on mode — there's no terrain to disagree about.

**Extension surface:**
- `enum MovementMode { Walk, Fly, Swim, Climb, Fall }` — SoA cold field.
- Per-tile terrain schema (currently absent): traversable by which modes?
- Movement resolver that consults tile × mode to decide legal deltas.
- Ability to set mode per tick (e.g., on landing, transition Fly → Walk).

**Why it matters:** the canonical fantasy / Starcraft altitude
distinction. Currently "flying dragon" is a narrator trick — the
engine can't enforce "ground melee can't reach the flyer." My
altitude scenes in `ecosystem.rs` and `starcraft_tribute.rs` work
only because distance > 2m, not because the engine knows about flight.

---

## 5. Per-unit attack range / attack damage

**Canonical mechanic:** Marine has 6m rifle range, Zergling has 2m
melee. Dragon bite does 40 dmg, human sword does 10.

**Current state:** `config.combat.attack_range: 2.0` and
`config.combat.attack_damage: 10.0` are GLOBAL. Every agent uses the
same `Attack` mask with the same range. Every melee deals the same 10.

HP is per-spawn, but attack capability is species-agnostic.

**Extension surface:**
- Per-entity `base_attack_range: f32` + `base_attack_damage: f32`
  in `entity` declarations in `entities.sim`, lowered to SoA fields.
- Mask's `can_attack(target)` reads per-agent range; Attack EffectOp
  reads per-agent damage.
- Inventory-modified variants later (weapons bump range/damage).

**Why it matters:** "different units with different reach/punch" is
the core of unit-variety games. My Starcraft tribute faked this with
HP, but the Marine can't actually shoot at a distance.

---

## 6. Damage types + resistances

**Canonical mechanic:** fire-resistant dragon takes ½ fire damage.
Steel plate armor blocks 50% of piercing. Ethereal enemies immune to
physical.

**Current state:** `EffectOp::Damage { amount: f32 }` is a single
scalar. No type tag, no per-species mod.

**Extension surface:**
- Tag damage: `Damage { amount, kind: DamageKind }` where
  `DamageKind ∈ { Physical, Fire, Frost, Poison, Arcane, ... }`.
- Per-creature `resistances: [f32; NUM_DAMAGE_KINDS]` from entities.sim.
- Damage application multiplies amount × (1 - resistance[kind]).

**Why it matters:** non-trivial combat strategy (counter-picking) needs
this. Fire breath against a fire dragon should tickle, not burn.

---

## 7. Damage-over-time + healing-over-time

**Canonical mechanic:** bleed applies 2 dmg/tick for 10 ticks. Regen
heals 1 hp/tick while active.

**Current state:** `EffectOp::Stun { duration_ticks }` and
`EffectOp::Slow { duration_ticks, factor_q8 }` have durations, but
they're state flags, not per-tick effect applications. No EffectOp
emits damage or healing over time.

**Extension surface:**
- `EffectOp::DamageOverTime { amount_per_tick: f32, duration_ticks: u32, kind: DamageKind }`
- Per-tick pipeline phase that iterates active DoTs, applies damage,
  decrements remaining ticks, emits expiration event at 0.
- Symmetric `HealOverTime`.

**Why it matters:** bleeds, burns, poisons, regens, shields-that-decay
are all canonical.

---

## 8. Line-of-sight / cover / occlusion

**Canonical mechanic:** archer can't shoot through a wall. Stealth unit
hidden behind an obstacle isn't targetable. Cover reduces incoming
damage.

**Current state:** `Gate.line_of_sight: bool` exists but docstring says
"MVP: unused." No raycast, no terrain, no occluders.

**Extension surface:**
- Terrain data (tile map or voxel-based — connects to `voxel_engine`
  crate).
- `line_of_sight(a: AgentId, b: AgentId) -> bool` raycast.
- Mask clauses consult it; abilities' gate enforces it.

**Why it matters:** stealth, cover, chokepoints — the basis of
tactical games — are absent.

---

## 9. Targeted friendly abilities (Heal ally, buff kin)

**Canonical mechanic:** medic heals nearby wounded allies. Priest
buffs the tank. Bard inspires the squad.

**Current state:** `Gate.hostile_only: bool` defaults true and the
mask enforces hostile-only targeting. There's no "friendly-only" or
"kin-only" targeting selector. `EffectOp::Heal` exists but its
current plumbing is single-target-hostile-by-default.

**Extension surface:**
- `Gate.friendly_only: bool` + matching mask predicate.
- Target-selection primitive `TargetSelector::FriendlyKin` in scoring
  so the scorer picks the wounded ally to heal.

**Why it matters:** support roles (medic, healer, bard) are entire
class archetypes in RPGs/MMOs. Currently every ability is adversarial.

---

## 10. Unit spawning / production (mid-sim)

**Canonical mechanic:** Zerg queen lays an egg that hatches into a
zergling. Necromancer raises a skeleton from a corpse. Mitosis.

**Current state:** agents are spawned ONLY at sim init via
`SimState::spawn_agent`. There's no EffectOp that creates a new agent.
No mechanism for runtime spawning. Agent slots are fixed-cap at init.

**Extension surface:**
- `EffectOp::Spawn { creature_type: CreatureType, relative_pos: Vec3, hp: f32 }`
- Agent-slot pool management for mid-sim births (free-slot pick,
  determinism-preserving id assignment).
- `Event::AgentSpawned` so cascades can react (grief, welcome, etc.).

**Why it matters:** without this you can't model reproduction,
summoning, necromancy, siege engines producing units, starcraft-style
production queues.

---

## 11. Equipment / inventory-driven stats

**Canonical mechanic:** agent equips a sword, attack_damage goes up by
5. Armor reduces incoming damage. Hat grants +10 max HP.

**Current state:** `cold_inventory` exists on SoA (commodities +
items) but no mechanism connects inventory to combat stats. No
"equipped slot." Weapons don't modify attack.

**Extension surface:**
- `ItemInstance` already has shape; needs an `equipped_bonuses` side
  vector per agent.
- Per-tick resolver that recomputes effective stats from inventory.
- UI/DSL for equipping in cascade handlers.

**Why it matters:** RPG progression, loot, crafting all hinge on this.

---

## 12. Structures / destructible environment

**Canonical mechanic:** siege wall blocks movement. Gate can be
destroyed to open a path. Dragon's breath sets trees on fire.

**Current state:** no static entities at all. The only "things" in the
sim are agents. Voxel/terrain integration mentioned in `docs/` but
the agent layer ignores it.

**Extension surface:**
- Static entity pool (like AggregatePool<Item> but for stationary
  world objects with health + destruction events).
- Pathfinding / spatial queries that respect them.
- Event::StructureDestroyed cascades.

**Why it matters:** base defense, fortifications, destructible cover,
siege warfare all need static, damageable world state.

---

## 13. Non-circular hit detection

**Canonical mechanic:** a greatsword has a forward arc. A shield blocks
frontal damage but not flanks. A backstab bonus needs facing.

**Current state:** agents have `pos: Vec3` but no facing. All distance
checks are symmetric Euclidean. No concept of "which way is an agent
looking."

**Extension surface:**
- `hot_facing: Vec3` (unit vector) on agent SoA.
- Facing updates when movement changes direction.
- Mask predicates like `target_in_front(target, half_angle)`.
- Damage modifiers based on attack angle.

**Why it matters:** flanking, positioning, "dance" combat rely on
facing.

---

## 14. Macro-scale time (day / night / season)

**Canonical mechanic:** wolves hunt at night. Dragon sleeps in
daylight. Crops grow over seasons.

**Current state:** `world.tick` is monotonically increasing u32; no
wall-clock semantics. DSL spec mentions a season system
(TICKS_PER_SEASON) for the zero-player world sim, but combat doesn't
read it for scoring.

**Extension surface:**
- `ReadContext::world_phase` (dawn / day / dusk / night).
- Scoring predicates that branch on phase.
- Time-of-day effects: vision range shrinks at night, etc.

**Why it matters:** day/night cycles shape behavior in most simulation
games.

---

## 15. Multi-agent formations / coordinated action

**Canonical mechanic:** phalanx moves as one unit. Archers hold the
back while tanks engage. A wolfpack encircles prey in a specific
pattern.

**Current state:** `pack_focus` and `kin_fear` are the only
coordination primitives — both are aggregate-over-kin view reads.
There's no "leader / follower" relationship, no shared goal, no
formation slot assignment.

**Extension surface:**
- Per-agent `hot_formation_slot: Option<(FormationId, u8)>`.
- Formation declaration in DSL (positions relative to a leader).
- Scoring predicates that penalize breaking formation.

**Why it matters:** tactical squads are a staple; currently agents
are atomized scorers.

---

## What this session delivered vs. these gaps

Everything built this session composes on top of the existing substrate:

- `scenes.rs`, `ecosystem.rs`, `starcraft_tribute.rs` — scene
  authoring + narration over existing primitives.
- `dragon_fire.rs` — custom `PolicyBackend` + `AbilityRegistry`
  population. Not an extension; these are explicit plug points the
  engine already provides.
- The Flee-tuning experiment and the dispatch-gate fix — tuning +
  bugfix, not expression-surface expansion.

None of it touched the IR types (`EffectOp`, `Area`, `Delivery`,
`Gate`), added a new MicroKind, added an entity-level field, or
changed the physics pipeline. That's why "cyborgs," "true fire breath
AoE," "Starcraft," etc. were unreachable — they need items 1, 2, 4,
5, 6, 10 here.

## Which to land first

Ranking by leverage-per-cost:

| Gap | Cost | Leverage | Rec |
|---|---|---|---|
| **5. Per-unit attack range/damage** | Low — entity field + mask read | High — unlocks unit variety | 🥇 |
| **1. AoE abilities (Cone)** | Medium — Area variant + cast-resolver | High — wizard archetype | 🥈 |
| **6. Damage types + resistances** | Medium — EffectOp variant + per-creature resistances | High — counter-pick depth | 🥉 |
| 7. DoT / HoT | Medium | Medium | |
| 2. Projectile delivery | High | High (novel dynamics) | |
| 10. Unit spawning | High | High (Starcraft requires this) | |
| 4. Movement modes | High (terrain dependency) | High | |
| 9. Friendly abilities | Low | Medium | |
| 13. Facing / non-circular hits | Medium | Medium | |
| 3. Persistent zones | Medium | Medium | |

5 is the cheapest first move that meaningfully widens the design space.
After that, 1 + 6 + 9 together get you a real spellcaster / support
archetype family.
