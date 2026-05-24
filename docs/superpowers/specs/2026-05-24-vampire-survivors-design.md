# Vampire Survivors — DSL-as-Engine Benchmark Fixture (Design)

> Status: design, awaiting review. Next: `writing-plans` → implementation plan with AIS (P8).
> Predecessor probe: `tower_defense` (`assets/sim/tower_defense.sim`). Genre sibling already in-tree: `predator_prey` (`assets/sim/predator_prey.sim`).

## 1. Goal & deliverable

A recognizable Vampire-Survivors-shaped fixture, authored as `assets/sim/vampire_survivors.sim`, run through the interpreted-rules path. It is the next DSL-as-engine benchmark after `tower_defense` in the **DSL → full engine** progression: new game-shaped fixtures are deliberate probes that surface where the DSL falls short of "real engine" expressivity.

**Two co-equal outputs:**
1. A deterministic fixture that runs to player-death (the vehicle / the game).
2. **A structured DSL-gap ledger** — every mechanic we tried to express in the DSL, whether it lowered cleanly, and if not, the minimal candidate primitive that would close it. This is the product; it feeds the next round of DSL extensions.

**Honest framing (post-investigation).** A survey of the existing sim corpus showed the VS *combat + power-ramp loop is almost entirely expressible with shipped primitives* — `predator_prey` is essentially "VS minus the build loop." So this fixture does **not** aim to produce a long gap list. It aims to (a) confirm the combat loop composes, and (b) drive deliberately at the two hardest remaining gaps — **discrete leveling math** and **branching upgrade selection** — and pin them precisely. A short, accurate ledger beats an inflated one.

## 2. Agent model

Entity subkinds (the `predator_prey` idiom — `entity Hare : Agent { creature_type: Hare }`, gated by `self.creature_type == …`; subkinds are first-class). **No mana-band role hack** (that was a `tower_defense` workaround we don't need):

- `entity Player : Agent { pos: vec3, vel: vec3 }` — one agent, slot 0. Has HP; moves by kiting.
- `entity Enemy : Agent { pos: vec3, vel: vec3 }` — slots 1..N; `alive = 0` until the runtime spawns them into a slot.

Agents target each other via **spatial queries / neighbor reductions**, not slot references — so the `tower_defense` "AgentId-literal-0 coercion" workaround does not arise here (`predator_prey` proves agents hunt/flee each other fine without it).

## 3. Score & termination

```
score        = death_tick                    (ticks the player survived)
termination  = player.alive == false
enemy ramp   = wave_size(t) grows unboundedly with t   (wave_defense precedent)
```

Same seed → same swarm → same death tick → same score. Any engine/DSL change that alters behavior moves the death tick. Secondary trace metrics: total kills, max level reached, cumulative XP, upgrades taken.

## 4. Per-tick mechanics — the combat loop (proven primitives)

Every row below reuses a pattern already exercised in-tree. Citations are to `assets/sim/predator_prey.sim` and `assets/sim/cooldown_probe.sim`.

| VS mechanic | Expressed as | Precedent | Gap? |
|---|---|---|---|
| Player kiting (flee swarm) | per-agent physics; role-filtered separation sum over enemy neighbors | `MoveHare` (predator_prey 189–204) | No |
| Enemy chase moving player | per-agent physics; nearest-of-role neighbor reduction toward player | `MoveWolf` (predator_prey 210–223) | No |
| Bolt — nearest enemy, periodic | `@phase(event) physics`, gate `world.tick % bolt_period == 0`, `for e in closest_enemy(self) { emit Damaged }` | `StrikePrey` + `cooldown_probe` tick-gate | No |
| Nova — AOE around player, periodic | same, but iterate `enemies_in_radius(self, r)` — AOE *is* the spatial-query iteration | `@spatial` query, no `@top_k` cap | No (no `EffectOp` AOE needed on the interp path) |
| XP on kill, attributed to player | `emit Killed { by: self }` → `view xp(by)` fold | `Killed` / `kill_count` (predator_prey 33–167) | No |
| Player death / termination | enemy contact `emit Damaged { target: player }` → `ApplyDamage` flips `alive=false` at ≤0 | `tower_defense` `ApplyDamage` | No |
| Power ramp (damage scales with progress) | read `xp(self)` / `upgrade_count(self, …)` in the damage amount | `score 1.0 + 0.5*kill_count(self)` (predator_prey 282) | **Verify** — proven in *scoring*, unverified inside a *physics emit amount* |

**Cooldown note.** The cooldown SoA is a *single* per-agent `cooldown_next_ready_tick` slot (`cooldown_probe`), and the `abilities.*` per-ability cooldown namespace is registered-but-unlowered (abilities_probe Gap #4). Two independent auto-fire periods on one player do **not** need it: `world.tick % period == 0` works (`Mod` is a supported arithmetic op; `world.tick` reads fine in a physics body — `cooldown_probe` proves both). Weapons are modeled as periodic `@phase(event)` physics rules (deterministic auto-fire), not utility-scored verbs (the player is not *choosing* between weapons — both fire on their own timers).

## 5. Discrete leveling — probe: `floor` / integer math on a view

`view xp(player) -> f32` folds `Killed { by: player }`. Level is `floor(xp / xp_per_level)`. The observed builtin set is `min / max / saturating_add / distance` — **no `floor` / int-cast / integer-division observed** — so this is a real probe, attempted in order:

- **Attempt A (interp-friendly):** `floor(xp(self) / k)` directly in a body/scoring expression. If `floor` does not lower → logged gap; candidate primitive: `floor` / `trunc` builtin, or `//` integer division.
- **Fallback B (schema change → full rebuild):** a `level: u32` field on `entity Player`, bumped by a `LevelUp` event whenever `xp >= (level+1)*k` (comparison + increment, no `floor`). Probes whether entity-subkind-declared scalar accumulator fields work, at the cost of leaving the interp-only path (rule class F).

We try A first; whichever holds is the finding. (If B is needed, the `level` field is a routine schema-hash regen — per project convention not an AIS impact line.)

## 6. Upgrade choice — probe: branching selection among options (the big gap)

The genuinely hard, previously-deferred gap. Modeled minimally but truthfully:

- `enum UpgradeKind { BoltDamage, NovaRadius, BoltRate, MoveSpeed }`.
- Per-`(player, kind)` tally via a **keyed view**: `view upgrade_count(player, kind)` — the pair-keyed-view pattern (`predator_focus(a, b)`, predator_prey 175–181). Weapon bodies read it: e.g. `bolt_damage = base + upgrade_count(self, BoltDamage) * step`.
- On each `LevelUp`, **select one upgrade by fixed priority among eligible options** (eligible = below its per-kind cap). The selected `UpgradeChosen { player, kind }` bumps the tally.
- **The probe:** expressing "pick the highest-priority upgrade not yet at cap" requires selecting among a fixed option set with a state-dependent filter. If the DSL has no `choose` / `match`-argmax-over-literal-options construct, that's the gap confirmed → candidate primitive: a priority-cascade `select` expression. We attempt it; the wall (if any) is the headline benchmark result.

## 7. DSL-owned vs runtime-owned split

- **DSL (`.sim`) owns** all per-tick steady state: kiting, both weapons, enemy chase, contact, damage, XP fold, level math, upgrade selection, the observability views.
- **Runtime owns** (the `tower_defense` precedent): initial state writes; **wave spawning** (every K ticks, fill `alive==0` enemy slots with escalating count + `per_agent_u32`-seeded spawn positions); reading `death_tick` / score out of state. Lifecycle/spawning staying out of the DSL is itself a standing, *known* gap (no DSL spawn primitive) — closed in Slice 2 below.

## 8. Gap ledger (the deliverable — updated during implementation)

| # | Probe | Expectation | Candidate primitive if it walls |
|---|---|---|---|
| 1 | View-value read inside a physics-body emit amount | Likely OK (proven in scoring) | extend body-expr view reads |
| 2 | `floor` / int math on a view → discrete level | **Likely gap** | `floor` / `trunc` / `//` |
| 3 | Branching selection among upgrade options | **Likely gap (the big one)** | `select` / priority-`match` expr |
| 4 | Spawn / infinite ramp | Known gap (runtime-owned today) | DSL `summon`/spawn primitive |
| 5 | Entity-subkind scalar accumulator field (`level`) — leveling fallback B | Unknown | — |

## 9. Verification & determinism pin

- Run via the interpreted-rules path for fast iteration. **Open implementation question:** the exact harness entry point for a crate-less standalone fixture — `predator_prey` and `tower_defense` are the templates; pin this first in the plan.
- **Determinism pin:** same seed → same `death_tick` and same per-tick `Killed`-count trace. A behavioral `probe` block (à la `predator_prey`'s probes) asserts the death tick and that level/upgrade milestones fire at fixed ticks.
- All RNG (spawn positions, spawn-timing jitter) flows through `per_agent_u32` (P5).

## 10. Constitution check (for the implementation plan's AIS / P8)

- **P1 compiler-first** ✅ — all behavior in `.sim`; no hand-written rule logic.
- **P5 keyed-PCG** ✅ — spawns via `per_agent_u32`.
- **P6 / P7 events** ✅ — mutation via flagged events (mirrors `predator_prey`).
- **P2 schema-hash** — N/A unless leveling fallback B adds a `level` field (then a routine regen, not an AIS line).
- **P3 parity** — fixture targets the interp path first; cross-backend deferred (like `wave_defense`) until the GPU-runtime slice.
- **P8 AIS** — the implementation plan carries the full statement.

---

## 11. Roadmap — from gap-probe to a small Vampire Survivors game in truth

Each slice is *both* a step toward a real, recognizable VS game *and* a DSL coverage probe (the project's standing principle). Each gets its own spec → plan → implementation cycle. Slices 1–4 are the minimum that feels like VS; 5–8 make it a real small game; 9–10 make it fast and watchable.

**Slice 1 — Foundation (this spec).**
Kiting player, bolt + nova auto-fire, swarm chase, XP-on-kill, discrete leveling (floor probe), minimal fixed-priority upgrade choice (branching probe), contact-death, runtime-driven ramp. *Game feel:* a lone survivor auto-fights an endless escalating swarm and slowly powers up until overwhelmed. *Closes/probes:* gaps #1–#3, #5.

**Slice 2 — In-engine spawning & infinite ramp.**
Move wave spawning out of the runtime into the DSL via a `summon`/spawn effect (`EffectOp::Summon` exists; `wave_defense` referenced it). Edge-spawning, escalating cadence, enemy-type mix, timed "swarm spike" events. *Game feel:* the world drives itself; no CPU babysitter. *Closes:* gap #4 (DSL spawn primitive).

**Slice 3 — Physical XP gems + magnet pickup.**
Killed enemies drop an `entity Gem : Agent` that the player collects by proximity (magnet radius growing with an upgrade). XP comes from *collecting*, not from the kill directly. *Game feel:* the authentic "hoover up the gem carpet" loop. *Probes:* third entity type + lifecycle, proximity-gated resource transfer, gem despawn.

**Slice 4 — Weapon variety + firing patterns.**
Add 2–4 more weapons with distinct geometry: whip (forward arc), orbiting projectiles (rotating around the player), lightning (random in-radius strike), garlic aura (persistent damage zone). Each is a new dispatch shape. *Game feel:* real build diversity. *Probes:* arc / rotational / persistent-zone targeting expressivity; per-weapon independent timers at scale.

**Slice 5 — Full level-up menu with real choice.**
Replace fixed-priority auto-pick with a proper level-up that offers N candidate upgrades and a *policy* selects (deterministic-but-stateful, or wired to the project's RL/utility pipeline so the fixture doubles as a policy benchmark). Reroll/banish. Weapon **evolution** (max-level weapon + matching passive → evolved weapon). *Game feel:* the VS build-craft core. *Probes:* the full branching-selection surface (beyond Slice 1's minimal version); cross-item state interaction.

**Slice 6 — Enemy variety, elites & bosses.**
Enemy archetypes (fast/swarmer, tanky/bruiser, ranged/spitter), elite spawns with modifiers, timed mini-bosses with larger HP and telegraphed attacks. *Game feel:* threat texture instead of one undifferentiated tide. *Probes:* heterogeneous AI in one tick, telegraph/cast-state (cast-state work exists in-tree), HP-bar-scale entities.

**Slice 7 — Survivability & pickups.**
Player max-HP / regen / armor / dodge, i-frames after a hit, revives. World pickups: hearts (heal), gold, bombs (screen clear), chests (on elite kill → upgrade roll). *Game feel:* the moment-to-moment "do I have enough defense" tension. *Probes:* timed status windows (i-frames), pickup taxonomy, instantaneous AOE clear.

**Slice 8 — Run structure & meta-progression.**
A timed run (e.g. survive 30 minutes → final "Death" reaper boss spawns and the run is meant to end), a gold economy, persistent between-run meta-upgrades bought with gold, and character select with different starting weapons/stats. *Game feel:* a complete loop you can "win" or "lose" and return to stronger — **this is the point where it is a small Vampire Survivors game in truth.** *Probes:* run-scoped vs persistent state, timed phase transitions, parameterized characters (the parameterised-rules work feeds this).

**Slice 9 — GPU runtime crate + perf signature.**
Graduate `assets/sim/vampire_survivors.sim` into `crates/vampire_survivors_runtime/`, compiled to WGSL + Rust, GPU-only, thousands of enemies and hundreds of projectiles. Death-tick becomes a real perf benchmark; cross-backend parity (P3). *Game feel:* true bullet-heaven scale. *Probes:* the full emit pipeline + reduction determinism (P11) at swarm scale.

**Slice 10 — Viewer / watchable build.**
Hook into `viewer_runtime` to render terrain + agents + projectiles + gems so the game can actually be watched and felt, not just scored. *Game feel:* you can sit and play/watch a run. *Probes:* viz-JSON extension surface for projectile/pickup/health-bar overlays.

**Definition of "a small VS game in truth":** Slices 1–8 complete — endless escalating swarm, multiple evolving auto-weapons, gem-collection leveling with a real upgrade menu, enemy/elite/boss variety, survivability systems, and a timed run with meta-progression. Slices 9–10 make it fast and watchable.
