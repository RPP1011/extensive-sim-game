# Vampire Survivors — DSL-as-Engine Benchmark Fixture (Design)

> Status: design, awaiting review. Next: `writing-plans` → implementation plan with AIS (P8).
> Predecessor probe: `tower_defense` (`assets/sim/tower_defense.sim`). Genre sibling already in-tree: `predator_prey` (`assets/sim/predator_prey.sim`).

## 1. Goal & deliverable

A recognizable Vampire-Survivors-shaped fixture, authored as `assets/sim/vampire_survivors.sim`, gated by `dsl_compiler` compile-tests (see §9). It is the next DSL-as-engine benchmark after `tower_defense` in the **DSL → full engine** progression: new game-shaped fixtures are deliberate probes that surface where the DSL falls short of "real engine" expressivity.

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
| Nova — AOE around player, periodic | same, but iterate `enemies_in_radius(self, r)` — AOE *is* the spatial-query iteration | `@spatial` query, no `@top_k` cap | No (no `EffectOp` AOE needed) |
| XP on kill, attributed to player | `emit Killed { by: self }` → `view xp(by)` fold | `Killed` / `kill_count` (predator_prey 33–167) | No |
| Player death / termination | enemy contact `emit Damaged { target: player }` → `ApplyDamage` flips `alive=false` at ≤0 | `tower_defense` `ApplyDamage` | No |
| Power ramp (damage scales with progress) | read `xp(self)` / `upgrade_count(self, …)` in the damage amount | `score 1.0 + 0.5*kill_count(self)` (predator_prey 282) | **Verify** — proven in *scoring*, unverified inside a *physics emit amount* |

**Cooldown note.** The cooldown SoA is a *single* per-agent `cooldown_next_ready_tick` slot (`cooldown_probe`), and the `abilities.*` per-ability cooldown namespace is registered-but-unlowered (abilities_probe Gap #4). Two independent auto-fire periods on one player do **not** need it: `world.tick % period == 0` works (`Mod` is a supported arithmetic op; `world.tick` reads fine in a physics body — `cooldown_probe` proves both). Weapons are modeled as periodic `@phase(event)` physics rules (deterministic auto-fire), not utility-scored verbs (the player is not *choosing* between weapons — both fire on their own timers).

## 5. Discrete leveling — probe: `floor` / integer math on a view

`view xp(player) -> f32` folds `Killed { by: player }`. Level is `floor(xp / xp_per_level)`. The observed builtin set is `min / max / saturating_add / distance` — **no `floor` / int-cast / integer-division observed** — so this is a real probe, attempted in order:

- **Attempt A:** `floor(xp(self) / k)` directly in a body/scoring expression. `Builtin::Floor`/`Ceil`/`Round` *do* lower in the compiled path (`crates/dsl_compiler/src/cg/lower/expr.rs:2021` → `BuiltinId::Floor`), so via the compile-gate harness (§9) this is expected to **pass**. The narrower finding: the interpreter's `eval_numeric_builtin` (`crates/dsl_ast/src/eval/builtins.rs`, known set `min/max/saturating_add/distance`) lacks a `floor` arm — so if/when the fixture is run on the *interpreted-rules* path, that arm is the gap (a one-line add per the world-sim-dsl coverage discipline).
- **Fallback B (only if A surprises us):** a `level: u32` field on `entity Player`, bumped by a `LevelUp` event whenever `xp >= (level+1)*k` (comparison + increment, no `floor`). Probes whether entity-subkind-declared scalar accumulator fields work, at the cost of a schema-hash regen (rule class F).

We try A first; whichever holds is the finding.

## 6. Upgrade choice — probe: branching selection among options (the big gap)

The genuinely hard, previously-deferred gap. Modeled minimally but truthfully. **Outcome (Task 8): the gap was narrower than feared — runtime-varying priority selection lowers; only two narrow walls remain.** See §8 rows #3/#3a/#3b/#3c for the precise results.

- **Sub-probe 3a — enum surface:** attempted `enum UpgradeKind { BoltDamage, NovaRadius, BoltRate, MoveSpeed }`. **PASSES** — parses and resolves with no corpus precedent. (Left declared but unused; the emit uses `u32` kind ids.)
- **The headline probe (3) — branching selection:** "pick the highest-priority upgrade not yet at cap" was expressed as a value-returning `if/else` chain producing a `u32`. **PASSES** via the *nested* form `if c0 { 0u } else { if c1 { 1u } else { 2u } }`, lowering to nested WGSL `select`. The only sub-wall is the chained `else if` *syntax* (parser requires `{` straight after `else`), which the nested form sidesteps losslessly. So there IS a usable priority-cascade `select` surface today — the candidate primitive (a dedicated `select`/`match`-argmax) would only be ergonomic sugar, not a missing capability.
- **Sub-probe 3b — view read in the `if` condition:** reading `upgrades_total(self) < cap` inside the selection condition (not just an emit amount) **PASSES**.
- **Sub-probe 3c — per-kind tally via a fold `where` guard:** the intended design (a tally view per kind, gated by `where k == 0u` on the `UpgradeChosen.kind` payload, à la `predator_focus`'s `where`) **WALLS** — `FoldHandlerIR` has no `where_clause` field and the resolver silently drops the guard, so the per-kind split is unachievable by fold guard. Landed fallback: a single un-split `upgrades_total(player)` tally.

## 7. DSL-owned vs runtime-owned split

- **DSL (`.sim`) owns** all per-tick steady state: kiting, both weapons, enemy chase, contact, damage, XP fold, level math, upgrade selection, the observability views.
- **Runtime owns** (the `tower_defense` precedent): initial state writes; **wave spawning** (every K ticks, fill `alive==0` enemy slots with escalating count + `per_agent_u32`-seeded spawn positions); reading `death_tick` / score out of state. Lifecycle/spawning staying out of the DSL is itself a standing, *known* gap (no DSL spawn primitive) — closed in Slice 2 below.

## 8. Gap ledger (the deliverable — updated during implementation)

| # | Probe | Result | Candidate primitive if it walls |
|---|---|---|---|
| 1 | View-value read inside a physics-body emit amount | **PASS** — lowers; BoltFire amount reads xp view storage | — |
| 2 | `floor` on a view → discrete level | **PASS** — `floor(xp(self)/k)` lowers to WGSL `floor(` in NovaFire emit amount (compiled path); interp arm in `eval/builtins.rs` still absent | `floor` arm in `eval/builtins.rs` (interp only) |
| 3 | Branching selection among upgrade options (value-returning `if/else → u32`) | **PASS (narrower than feared)** — the *nested* `if cond { 0u } else { if cond { 1u } else { 2u } }` lowers to nested WGSL `select(select(2u, 1u, …), 0u, …)`. Runtime-varying, state-dependent priority selection over a fixed option set IS expressible. **One sub-wall:** chained `else if` does **not** parse — the parser's if-expr `else` form (`crates/dsl_ast/src/parser.rs:4742`) requires `{` directly after `else`, so `else if …` fails with `parse error: expected `{` … parsing `else` expr `{``. Workaround is the equivalent nested `else { if … }` (no semantic loss). | `else if` sugar in the if-expr parser (else: nest manually) |
| 3a | Standalone `enum` surface (`UpgradeKind`) | **PASS** — `enum UpgradeKind { BoltDamage, NovaRadius, BoltRate, MoveSpeed }` parses and resolves cleanly with no corpus precedent. (Declared but unused in the landed fixture; kinds are `u32` ids in the emit.) | — |
| 3b | View read in an `if` *condition* in a physics body (`upgrades_total(self) < cap`) | **PASS** — lowers to `view_1_get(agent_id) < config_N` inside the `select`; the view storage read inside a condition (not just an amount) lowers fine. | — |
| 3c | Event-fold `where` guard on a u32 payload field (`on UpgradeChosen { kind: k } where k == 0u`) | **WALL (silent drop, no error)** — `FoldHandlerIR` (`crates/dsl_ast/src/ir.rs:1113`) has **no `where_clause` field**; the resolver (`crates/dsl_ast/src/resolve.rs:1771`) builds `FoldHandlerIR { pattern, body, span }` and **silently discards `h.where_clause`**. The guard never lowers — every matching event folds identically, so per-kind tallies are unachievable via a fold guard. (The `predator_prey` `where predator == a && victim == b` "precedent" only appears to work because those are key-param equalities aligned with pair-map indexing, not because the guard is evaluated.) Landed fallback: a single un-split `upgrades_total(player)` view (drop the per-kind split). | `where_clause` field on `FoldHandlerIR` + a guard arm in `lower_one_handler` (`crates/dsl_compiler/src/cg/lower/view.rs:764`) |
| 4 | Spawn / infinite ramp | Known gap (runtime-owned today) | DSL `summon`/spawn primitive |
| 5 | Entity-subkind scalar accumulator field (`level`) — leveling fallback B | Unknown | — |

**Landed shape (Task 8).** `enum UpgradeKind` declared (3a PASS, unused). One `event UpgradeChosen { player, kind: u32 }`, one `view upgrades_total(player)` fold (un-split because the per-kind `where` guard walls — 3c), and a `physics ChooseUpgrade` whose body is a nested value-returning `if/else` over view-gated conditions selecting a `u32` kind (3 + 3b PASS). Net: **fixed-priority, runtime-varying upgrade selection IS expressible today**; the only true walls are the `else if` parser sugar (cosmetic — nest manually) and the fold `where`-guard drop (substantive — blocks per-kind tally-by-fold-guard).

## 9. Verification & determinism pin

- **Harness (resolved):** crate-less fixtures like `predator_prey`/`tower_defense` are exercised by **`dsl_compiler` compile-gate tests** — a `compile_sim()` helper drives `parse → resolve → lower → schedule → emit` and asserts emitted-kernel shapes (see `crates/dsl_compiler/tests/stress_fixtures_compile.rs`). "Did it lower cleanly?" *is* the gap test; an unsupported construct surfaces as a typed compiler error, not a silent pass. The engine `interpreted-rules` path is a secondary, behavior-level runner coupled to the wolves+humans engine runtime and is **not** how these fixtures gate — so it is out of scope for this slice (and is where the `floor` interp-arm finding (§5) would land if pursued later).
- **Determinism pin:** the compile-gate is deterministic by construction (same source → same kernels). A behavioral death-tick pin (à la `predator_prey`'s `probe` blocks) is deferred to the runtime-crate slice (Slice 9), since there is no runtime executing this `.sim` until then.
- All RNG (spawn positions, spawn-timing jitter) flows through `per_agent_u32` (P5).

## 10. Constitution check (for the implementation plan's AIS / P8)

- **P1 compiler-first** ✅ — all behavior in `.sim`; no hand-written rule logic.
- **P5 keyed-PCG** ✅ — spawns via `per_agent_u32`.
- **P6 / P7 events** ✅ — mutation via flagged events (mirrors `predator_prey`).
- **P2 schema-hash** — N/A unless leveling fallback B adds a `level` field (then a routine regen, not an AIS line).
- **P3 parity** — compile-gate slice emits kernels but executes nothing; cross-backend behavioral parity deferred (like `wave_defense`) until the GPU-runtime slice (Slice 9).
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
