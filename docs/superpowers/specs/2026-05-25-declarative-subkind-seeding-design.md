# Declarative Subkind Seeding + Subkind Gating (Design)

> Status: design, awaiting review. Next: `writing-plans` → implementation plan with AIS (P8).
> Closes the seeding gap that the DSL-game-engine #1 generality proof surfaced: `make_playable` zero-inits agents, so `play vampire_survivors` / `play predator_prey` start empty (no seeded player, enemies, or wolves). Builds on `docs/superpowers/specs/2026-05-25-dsl-game-engine-author-any-game-design.md` (engine #1, Waves 0–2 on `main`).

## 1. Goal

Let a `.sim` declare its **initial population by entity subkind**, with zero Rust and **no slot indices** — the agent-slot array is an implementation detail the compiler owns. Seeding flows through `try_new`, which `make_playable` already calls, so the generic `play <fixture>` binary gets real games for free.

Concurrently, make the **entity subkind the role discriminant** everywhere (rules, render, controls), retiring the mana-band idiom (`mana ∈ [0.5,1.5]` = player) in vampire_survivors + predator_prey. Subkinds are already first-class (`self.creature_type == Hare` works today); this makes them the authoring-level "role."

Success = `cargo run -p engine_play --bin play vampire_survivors` opens a real game (a seeded player kiting a wave-spawned swarm) and `… play predator_prey` opens a real game (a player Hare evading seeded Wolves), both authored entirely in `.sim` — which then unblocks deleting the bespoke `vs_*` viewer (Plan E of the engine spec).

## 2. Background: the gap + the slot leak

- `make_playable(name, seed, agents)` calls `GeneratedRuntime::try_new`, which zero-inits agent buffers. The wave drain fills enemies, but **nothing seeds the player** (VS relied on a Rust `seed_initial_state` that `make_playable` never calls) or the autonomous population (predator_prey's Hares/Wolves). `play_probe` only works because it self-seeds via the existing `init {}` block.
- Today's `init {}` is **uniform and slot-leaky**: `field: <int|slot>` applied to every agent (`crates/dsl_ast/src/parser.rs:1072`). It can't express "1 player + a pool" without `slot` arithmetic hacks (predator_prey's `mana: slot` + `mana < 0.5` player band), which exposes the slot array — the wrong abstraction.
- The current role discriminant is the **mana-band hack** (`mana ∈ [lo,hi]`), used by every rule guard, every spatial-query filter, and the `render`/`controls` blocks. It's a workaround for "no per-agent role" — but agents *do* carry `creature_type`, and subkinds set it.

## 3. Design

### 3.1 Seeding: `spawn <Subkind> count <N> { … }`

Replace the uniform `init {}` body with **per-subkind population blocks**:
```
init {
  spawn Player count 1    { hp: 100.0, pos: origin }
  spawn Enemy  count 511  { alive: 0 }            // pool the wave drain fills
}
```
```
init {
  spawn PlayerHare count 1    { pos: origin }
  spawn Hare       count 199  { pos: scatter(40.0) }
  spawn Wolf       count 8    { pos: scatter(40.0) }
}
```
- **`<Subkind>`** is a declared `entity X : Agent`. **`count`** is a literal or `config.*` value. Counts sum ≤ `agent_count`.
- The compiler assigns contiguous slots per block (skipping the slot-0 `AgentId` sentinel), and for each seeded agent stamps **`creature_type` = the subkind's ordinal**, defaults **`alive: 1`** (overridable — a pool is `alive: 0`), applies the block's field values, and seeds `pos`.
- **Field values:** int, **`f32`** (new — `hp: 100.0`), `slot` (kept for niche uses), and **position builtins** for `pos`: `origin`, `scatter(r)` (uniform in a radius-`r` disc), `ring(r)` (on a radius-`r` circle) — both seeded via `per_agent_u32(seed, slot, 0, purpose)` (P5-deterministic).
- The flat `init { field: v }` form remains valid (sugar for "all agents"), so existing fixtures (`play_probe`, others) are unaffected.

### 3.2 Subkind gating (retire mana-band in the two games)

With every agent carrying its subkind, replace mana-band guards with subkind checks:
- **Rule guards / spatial filters:** `self.mana >= player_mana_min && … <= player_mana_max` → `self.creature_type == Player`; `candidate.mana ∈ enemy_band` → `candidate.creature_type == Enemy`. (predator_prey: `PlayerHare` / `Hare` / `Wolf`.)
- **`render {}`:** `agent when mana in [0.5,1.5] { color … }` → `agent when creature_type is Player { color … }` (the render block gains a `creature_type is <Subkind>` selector; `controls` is unchanged — key→input).
- **The wave drain** (`crates/sims/src/summon_alloc.rs`) sets spawned enemies' **`creature_type = Enemy`** (instead of the enemy mana band) so they gate correctly. (Sanctioned runtime lifecycle, like the existing field writes.)
- The mana-band `config` fields + mana seeding are dropped from the two games; `mana` is freed as plain game data (unused for now).

### 3.3 Codegen

Extend `build_helper`'s existing `init {}` → `create_buffer_init` path (the "Plan E-A6" per-slot pattern, `build_helper.rs:233`) to: walk the `spawn` blocks, compute each subkind's slot range, and emit per-range buffer fills for `creature_type`, `alive`, the declared fields (int/f32 bit-patterns), and `pos` (constant for `origin`, `per_agent_u32`-seeded for `scatter`/`ring`). No new binding or registry change — `make_playable`→`try_new` already runs this.

## 4. Phases (one spec; each verifiable)

1. **`f32` init values** — the parser/codegen accept float fills (today int-only). Verify: a probe `.sim` seeds `hp: 100.0`, readback confirms.
2. **`spawn <Subkind> count N { … }` grammar + slot assignment** — parse the population blocks; codegen assigns slot ranges + stamps `creature_type` + `alive` + fields. Verify (headless): `make_playable` a probe with two subkinds, snapshot shows the right per-subkind counts + `creature_type`s.
3. **Seeded positions** — `origin`/`scatter(r)`/`ring(r)` via `per_agent_u32`. Verify: seeded agents' positions are within the radius + deterministic across runs (same seed → same positions; P5).
4. **VS subkind gating + seeding** — `spawn Player count 1` + `Enemy` pool; switch VS rule guards + `render` to subkind; drain sets `creature_type = Enemy`. Verify: `make_playable("vampire_survivors")` snapshot shows a live player + a growing enemy count under the drain; existing `vampire_survivors_exec` GPU tests adapted + green. **Then `play vampire_survivors` is a real game → unblocks engine-spec Plan E (parity + delete `vs_*`).**
5. **predator_prey subkind gating + seeding** — `spawn PlayerHare/Hare/Wolf`; switch guards + render to subkind; scattered positions. Verify: `predator_prey_playable` shows a player Hare + seeded Wolves; `play predator_prey` is a real game (the zero-Rust generality proof completes).

## 5. Constitution check (for the plan's AIS, P8)
- **P5 (keyed PCG)** ✅ — `scatter`/`ring` positions via `per_agent_u32`; deterministic across runs.
- **P2 (schema-hash)** N/A — seeding writes existing Agent SoA columns at `try_new`; subkind gating reads existing `creature_type`; no layout/event change.
- **P1 (compiler-first)** ✅ — seeding + role discrimination become compiler-lowered DSL data; the drain's `creature_type` write is sanctioned runtime lifecycle (like its existing field writes).
- **P3 (parity)** ✅ — `creature_type` guards lower identically on both backends.
- **P10 (no panic)** — phase headless drivers assert seeded construction + T ticks without panic.

## 6. Top risks
- **Slot-assignment vs the slot-0 `AgentId` sentinel** — the compiler must skip slot 0 (NonZeroU32 "absent"); off-by-one here mis-seeds. Pin with a snapshot test.
- **The wave drain must set `creature_type`** — miss it and spawned enemies don't gate (invisible/inert). Covered by VS's "enemy count grows + weapons cull" test.
- **`count` summing > agent_count** — overflow the buffer; the codegen must validate counts ≤ `agent_count` at compile time (a typed error).
- **Mana-band removal ripple** — every guard + spatial filter + render selector in two `.sim`s changes; do it per-game (Phase 4 then 5), keep the GPU tests green at each step.
- **`creature_type is <Subkind>` render selector** is new render-block grammar (small extension to the engine spec's `render {}`); keep it minimal.

## 7. File map
- `crates/dsl_ast/src/{ast,parser}.rs` — `spawn <Subkind> count N { … }` grammar; `f32`/position-builtin init values (modify).
- `crates/dsl_compiler/src/build_helper.rs` — slot assignment + per-range `create_buffer_init` (creature_type/alive/fields/seeded pos) (modify).
- `crates/dsl_compiler/src/cg/emit/render.rs` — `creature_type is <Subkind>` selector → descriptor (modify).
- `crates/sims/src/summon_alloc.rs` — drain sets `creature_type = Enemy` (modify).
- `assets/sim/vampire_survivors.sim` — `spawn` block + subkind guards + render-by-subkind (modify).
- `assets/sim/predator_prey.sim` — `spawn` blocks + subkind guards + render-by-subkind (modify).
- `crates/sims/tests/` — seeding + per-game playable headless tests (modify/create).

## 8. Out of scope
- A general subkind-gating migration of *all* fixtures (only VS + predator_prey here).
- Per-agent seeded *non-position* randomness (e.g. random hp); only positions get `scatter`/`ring`. Other fields are per-subkind constants.
- The engine-spec Plan E deletion itself (this spec *unblocks* it; the deletion lands under the engine spec).
- Runtime-resizable populations / spawning new subkinds at runtime beyond the existing wave drain.
