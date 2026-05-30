# Vampire Survivors — Foundation Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author `assets/sim/vampire_survivors.sim` — a Vampire-Survivors-shaped DSL fixture (kiting player, bolt+nova auto-fire, swarm chase, XP-on-kill, discrete leveling, minimal upgrade choice) — gated by `dsl_compiler` compile-tests, producing a compiling fixture plus an accurate DSL-gap ledger.

**Architecture:** The fixture is pure `.sim` source. It is exercised by a single `dsl_compiler` integration test file using the established `compile_sim()` helper (`parse → resolve → lower → schedule → emit`, then assert emitted-kernel shapes). "Did it lower cleanly?" is the test. Each mechanic is added incrementally behind its own assertion; the two hard probes (view-read-in-physics-amount; upgrade-choice/enum) have explicit wall-handling branches that update the spec's gap ledger and apply an in-fixture workaround so the fixture stays green.

**Tech Stack:** Rust, the World Sim DSL, `crates/dsl_compiler` test harness. Reference fixtures: `assets/sim/predator_prey.sim` (hunt/flee/views/scoring), `assets/sim/cooldown_probe.sim` (tick-gated emit, `init` block), `assets/sim/tower_defense.sim` (ApplyDamage chronicle).

**Spec:** `docs/superpowers/specs/2026-05-24-vampire-survivors-design.md`. This plan implements **Slice 1 only**; Slices 2–10 (in-engine spawning, gems, weapon variety, full upgrade menu, bosses, survivability, run/meta, GPU crate, viewer) each get their own spec→plan cycle.

---

## Architectural Impact Statement (P8)

- **Existing primitives searched:** `predator_prey.sim` (spatial `@top_k` queries, `sum(... where if ...)` neighbour reductions in physics, `@materialized` view-folds, view reads in `verb score`, `Killed { by }` killer attribution), `cooldown_probe.sim` (`world.tick`-gated `emit` in a physics body, `init { ... }` fixture-owned initial state), `tower_defense.sim` (`ApplyDamage` HP-subtract chronicle, role-discriminated `where` gates). Search method: `rg` over `assets/sim/` + `crates/dsl_compiler/tests/`.
- **Decision:** new `.sim` fixture only, plus one `dsl_compiler` integration-test file. No new runtime crate, no hand-written rule logic, no new `EffectOp` variants. Mirrors the crate-less compile-gate pattern (`stress_fixtures_compile.rs`).
- **Rule-compiler touchpoints:** DSL input added: `assets/sim/vampire_survivors.sim`. No emitter changes planned. *If* a probe walls and the fix is a one-line interpreter/builtin arm, it is logged in the spec ledger and deferred — language extensions are out of scope for this fixture slice (P1: they go through the normal compiler-extension path, not a fixture).
- **Hand-written downstream code:** NONE beyond the test file (test code, not engine rule logic).
- **Constitution check:** P1 ✅ (all behaviour in `.sim`). P2 N/A unless leveling fallback B adds a `level` field (routine regen, not an AIS line — `feedback_schema_hash_auto`). P3 N/A (compile-gate emits but executes nothing; behavioural parity deferred to Slice 9). P4 N/A (no new variants). P5 N/A this slice (spawning/RNG is runtime-owned, deferred to Slice 2). P6/P7 ✅ (events flagged `@replayable @gpu_amenable`, mirrors `predator_prey`). P8 ✅ (this section). P10/P11 N/A (no runtime execution this slice).

---

## File Structure

- **Create:** `assets/sim/vampire_survivors.sim` — the fixture (built up across Tasks 1–9).
- **Create:** `crates/dsl_compiler/tests/vampire_survivors_compile.rs` — compile-gate tests (one assertion group per mechanic).
- **Modify (docs):** `docs/superpowers/specs/2026-05-24-vampire-survivors-design.md` §8 gap ledger — updated with actual probe outcomes in Tasks 6–9.

---

## A note on probes (read before starting)

This is a gap-discovery fixture. For most tasks, the compile-test goes **red → green** like normal TDD. But Tasks 1, 6, 8 are *probes*: the construct may not lower. When a probe step fails to lower:

1. **Do not bash on it.** Copy the exact compiler error (`parse:`/`resolve:`/`lower:`/`emit:` prefix from `compile_sim`) into the spec's §8 gap ledger row.
2. Apply the **documented fallback** in that task to keep the fixture compiling.
3. The finding (wall + workaround) is a deliverable — a passing test on the *fallback* shape is success.

---

## Task 1: Harness + skeleton (probe: subkind discriminant)

**Files:**
- Create: `assets/sim/vampire_survivors.sim`
- Create: `crates/dsl_compiler/tests/vampire_survivors_compile.rs`

- [ ] **Step 1: Write the skeleton `.sim`**

Create `assets/sim/vampire_survivors.sim`. Player + Enemy as Agent subkinds; events for the full slice (declared now, emitted later — `predator_prey` declares ahead of emit). A trivial per-agent physics rule so a kernel is emitted. **Probe:** whether two bare Agent subkinds are distinguishable in a `where` gate. We start with a `role` discriminant via the proven `mana`-band pattern from `tower_defense` (slot 0 = player, `mana` band tags), because subkind-identity gating in `where` has no corpus precedent — declaring the subkinds AND tagging by mana keeps Task 1 green while we note the finding.

```
// vampire_survivors — DSL-as-engine benchmark fixture (Slice 1 foundation).
// See docs/superpowers/specs/2026-05-24-vampire-survivors-design.md.
//
// Roles are mana-band tagged (tower_defense precedent): player ∈ [0.5,1.5],
// enemy ∈ [1.5,2.5]. Subkind-identity gating in `where` has no corpus
// precedent (Gap: see spec §2); mana-band is the proven discriminant.

event Tick { }

@replayable @gpu_amenable
event Damaged { source: AgentId, target: AgentId, amount: f32 }

@replayable @gpu_amenable
event Killed { by: AgentId, prey: AgentId, pos: vec3 }

entity Player : Agent { pos: vec3, vel: vec3 }
entity Enemy  : Agent { pos: vec3, vel: vec3 }

config vs {
  player_mana_min: f32 = 0.5,
  player_mana_max: f32 = 1.5,
  enemy_mana_min:  f32 = 1.5,
  enemy_mana_max:  f32 = 2.5,
  drift_speed:     f32 = 0.01,
}

// Trivial seed rule so the pipeline emits at least one kernel.
physics IdleDrift @phase(per_agent) {
  on Tick {} where (self.alive) {
    let new_pos = self.pos + self.vel * config.vs.drift_speed;
    agents.set_pos(self, new_pos);
  }
}
```

- [ ] **Step 2: Write the compile-gate harness + first test**

Create `crates/dsl_compiler/tests/vampire_survivors_compile.rs`, copying the proven helper shape from `crates/dsl_compiler/tests/stress_fixtures_compile.rs:33-71`.

```rust
//! Compile-gate tests for the vampire_survivors DSL benchmark fixture.
//! Drives assets/sim/vampire_survivors.sim through
//! parse -> resolve -> lower -> schedule -> emit and asserts emitted
//! kernel shapes. A failing lower IS the gap signal (spec §8 ledger).

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn compile_sim(path: &std::path::Path) -> Result<EmittedArtifacts, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|e| format!("lower: {e:?}"))?;
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))
}

fn kernel_body_containing<'a>(art: &'a EmittedArtifacts, needle: &str) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle))
        .map(|(_, body)| body.as_str())
}

#[test]
fn vampire_survivors_compiles() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| panic!("vampire_survivors.sim failed at: {e}"));
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[vampire_survivors] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`
Expected: PASS, with an `[vampire_survivors]` line listing at least one kernel. If it fails at `parse:`/`resolve:`/`lower:`, the error message names the construct that didn't lower — if it's the entity/config/event shape, reconcile against `predator_prey.sim`/`tower_defense.sim` syntax. **Probe note:** if you instead tried subkind-identity gating (`where self is Player`) and it failed, that confirms the Gap §2 finding — keep the mana-band tagging.

- [ ] **Step 4: Commit**

```bash
git add assets/sim/vampire_survivors.sim crates/dsl_compiler/tests/vampire_survivors_compile.rs
git commit -m "feat(vampire_survivors): skeleton fixture + compile-gate harness"
```

---

## Task 2: Enemy chase (MoveWolf pattern)

**Files:**
- Modify: `assets/sim/vampire_survivors.sim`
- Modify: `crates/dsl_compiler/tests/vampire_survivors_compile.rs`

- [ ] **Step 1: Add the chase test (write it first)**

Append to `vampire_survivors_compile.rs`:

```rust
#[test]
fn enemy_chase_emits_neighbour_walk() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let body = kernel_body_containing(&art, "ChasePlayer")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| panic!("no chase kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        body.contains("spatial_grid_offsets") || body.contains("grid_starts"),
        "expected bounded-neighbour walk in ChasePlayer body; got:\n{body}",
    );
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile enemy_chase -- --nocapture`
Expected: FAIL — `no chase kernel` (the rule doesn't exist yet).

- [ ] **Step 3: Add the chase query + physics to the `.sim`**

Add a perception spatial query and an enemy-chase rule, modelled on `predator_prey.sim:112-117` (`nearby_agents`) and `predator_prey.sim:206-223` (`MoveWolf`). Add to `config vs`: `perception_radius: f32 = 12.0`, `enemy_speed: f32 = 0.1`. Then:

```
@spatial(radius = config.vs.perception_radius, kind = [Agent])
@top_k(8)
query nearby_agents(self: Agent, radius: f32) -> [Agent]
sort_by distance(self, _) limit 8 {
  candidate != self
}

// Enemies steer toward the nearest player-band agent (the moving player).
@phase(post)
physics ChasePlayer {
  on Tick {} where (self.alive
                    && self.mana >= config.vs.enemy_mana_min
                    && self.mana <= config.vs.enemy_mana_max) {
    let toward = sum(other in spatial.nearby_agents(self, config.vs.perception_radius) where
      if other.mana >= config.vs.player_mana_min
         && other.mana <= config.vs.player_mana_max
         && other.alive {
        other.pos - self.pos
      } else {
        vec3(0.0, 0.0, 0.0)
      });
    let new_vel = self.vel + toward * config.vs.enemy_speed;
    let new_pos = self.pos + new_vel;
    agents.set_vel(self, new_vel);
    agents.set_pos(self, new_pos);
  }
}
```

- [ ] **Step 4: Run both tests**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`
Expected: PASS (both `vampire_survivors_compiles` and `enemy_chase_emits_neighbour_walk`). If `lower:` errors on the `sum(... where if ...)` shape, compare byte-for-byte against `MoveWolf` — the most likely mismatch is the predicate/`if`/`else` arms or the `spatial.<query>(self, radius)` call form.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(vampire_survivors): enemy chase via nearest-player neighbour reduction"
```

---

## Task 3: Player kiting (MoveHare pattern)

**Files:**
- Modify: `assets/sim/vampire_survivors.sim`
- Modify: `crates/dsl_compiler/tests/vampire_survivors_compile.rs`

- [ ] **Step 1: Add the kiting test**

```rust
#[test]
fn player_kite_emits_neighbour_walk() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let body = kernel_body_containing(&art, "KitePlayer")
        .unwrap_or_else(|| panic!("no KitePlayer kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        body.contains("spatial_grid_offsets") || body.contains("grid_starts"),
        "expected bounded-neighbour walk in KitePlayer body; got:\n{body}",
    );
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile player_kite -- --nocapture`
Expected: FAIL — `no KitePlayer kernel`.

- [ ] **Step 3: Add the player kiting rule**

Add to `config vs`: `flee_radius: f32 = 8.0`, `player_speed: f32 = 0.12`. The player (player-band) flees enemy-band neighbours (separation), mirroring `MoveHare` (`predator_prey.sim:188-204`):

```
@phase(post)
physics KitePlayer {
  on Tick {} where (self.alive
                    && self.mana >= config.vs.player_mana_min
                    && self.mana <= config.vs.player_mana_max) {
    let flee = sum(other in spatial.nearby_agents(self, config.vs.flee_radius) where
      if other.mana >= config.vs.enemy_mana_min
         && other.mana <= config.vs.enemy_mana_max
         && other.alive
         && distance(self.pos, other.pos) < config.vs.flee_radius {
        self.pos - other.pos
      } else {
        vec3(0.0, 0.0, 0.0)
      });
    let new_vel = self.vel + flee * config.vs.player_speed;
    let new_pos = self.pos + new_vel;
    agents.set_vel(self, new_vel);
    agents.set_pos(self, new_pos);
  }
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`
Expected: PASS (three tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(vampire_survivors): player kiting via enemy-filtered separation"
```

---

## Task 4: Bolt weapon + damage application

**Files:**
- Modify: `assets/sim/vampire_survivors.sim`
- Modify: `crates/dsl_compiler/tests/vampire_survivors_compile.rs`

- [ ] **Step 1: Add the bolt + ApplyDamage test**

```rust
#[test]
fn bolt_fires_and_damage_applies() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    // Bolt producer emits into the event ring.
    let bolt = kernel_body_containing(&art, "BoltFire")
        .unwrap_or_else(|| panic!("no BoltFire kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        bolt.contains("atomicStore(&event_ring") || bolt.contains("atomicAdd(&event_tail"),
        "BoltFire should emit a Damaged event; got:\n{bolt}",
    );
    // ApplyDamage consumes Damaged and writes hp.
    let apply = kernel_body_containing(&art, "ApplyDamage")
        .unwrap_or_else(|| panic!("no ApplyDamage kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(apply.contains("agent_hp"), "ApplyDamage should write agent_hp; got:\n{apply}");
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile bolt_fires -- --nocapture`
Expected: FAIL — `no BoltFire kernel`.

- [ ] **Step 3: Add the closest-enemy query, BoltFire, and ApplyDamage**

Add to `config vs`: `bolt_range: f32 = 18.0`, `bolt_period: u32 = 12`, `bolt_damage: f32 = 6.0`. The `closest_enemy` query mirrors `closest_prey` (`predator_prey.sim:123-130`); `BoltFire` mirrors `StrikePrey` (`predator_prey.sim:228-244`) with a `world.tick % period` gate (`cooldown_probe.sim:97-104` proves `world.tick` in a physics body; `%` is supported — `eval/builtins.rs:137`); `ApplyDamage` mirrors `tower_defense.sim:132-142`.

```
@spatial(radius = config.vs.bolt_range, kind = [Agent])
@top_k(1)
query closest_enemy(self: Agent) -> [Agent]
sort_by distance(self, _) limit 1 {
  candidate != self
  && candidate.alive
  && candidate.mana >= config.vs.enemy_mana_min
  && candidate.mana <= config.vs.enemy_mana_max
}

// Bolt: every bolt_period ticks, the player strikes the nearest enemy.
@phase(event)
physics BoltFire {
  on Tick {} where (self.alive
                    && self.mana >= config.vs.player_mana_min
                    && self.mana <= config.vs.player_mana_max
                    && world.tick % config.vs.bolt_period == 0) {
    for target in spatial.closest_enemy(self) {
      emit Damaged { source: self, target: target, amount: config.vs.bolt_damage }
    }
  }
}

@phase(post)
physics ApplyDamage {
  on Damaged { source: _, target: t, amount: a } {
    let new_hp = agents.hp(t) - a;
    agents.set_hp(t, new_hp);
    if (new_hp <= 0.0) {
      agents.set_alive(t, false);
      emit Killed { by: self, prey: t, pos: agents.pos(t) }
    }
  }
}
```

**Probe note:** `world.tick % config.vs.bolt_period == 0` in a `verb`/physics `where` is the auto-fire gate. If `world.tick` is not allowed in a `where` clause (only in the body, as in `cooldown_probe`), the fallback is to move the gate into the body: wrap the `for` loop in `if (world.tick % config.vs.bolt_period == 0u) { ... }`. Use whichever lowers; if neither does, log to ledger #2-adjacent and gate on a per-agent cooldown slot instead. Also note: `ApplyDamage`'s `emit Killed { by: self }` — `self` here is the *damaged target's* handler scope, not the killer; if `source` is needed as `by`, bind it: `on Damaged { source: s, target: t, amount: a }` and use `by: s`. Prefer `by: s` (correct killer attribution).

- [ ] **Step 4: Run all tests**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`
Expected: PASS (four tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(vampire_survivors): bolt auto-fire (tick-gated) + damage chronicle"
```

---

## Task 5: Nova weapon (periodic AOE) — mixed dispatch milestone

**Files:**
- Modify: `assets/sim/vampire_survivors.sim`
- Modify: `crates/dsl_compiler/tests/vampire_survivors_compile.rs`

- [ ] **Step 1: Add the nova test**

```rust
#[test]
fn nova_fires_aoe_neighbour_walk() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let nova = kernel_body_containing(&art, "NovaFire")
        .unwrap_or_else(|| panic!("no NovaFire kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        nova.contains("spatial_grid_offsets") || nova.contains("grid_starts"),
        "NovaFire should iterate enemies in radius via neighbour walk; got:\n{nova}",
    );
    assert!(
        nova.contains("atomicStore(&event_ring") || nova.contains("atomicAdd(&event_tail"),
        "NovaFire should emit Damaged per enemy in radius; got:\n{nova}",
    );
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile nova_fires -- --nocapture`
Expected: FAIL — `no NovaFire kernel`.

- [ ] **Step 3: Add an in-radius enemy query + NovaFire**

Add to `config vs`: `nova_radius: f32 = 6.0`, `nova_period: u32 = 40`, `nova_damage: f32 = 3.0`. Unlike `closest_enemy`, `enemies_in_radius` has no `@top_k` cap — it enumerates all enemies in radius (AOE *is* the spatial iteration), then emits one `Damaged` per enemy in a `for` loop (multi-emit, proven by `swarm_event_storm`):

```
@spatial(radius = config.vs.nova_radius, kind = [Agent])
@top_k(32)
query enemies_in_radius(self: Agent) -> [Agent]
sort_by distance(self, _) limit 32 {
  candidate != self
  && candidate.alive
  && candidate.mana >= config.vs.enemy_mana_min
  && candidate.mana <= config.vs.enemy_mana_max
}

// Nova: every nova_period ticks, burst-damage every enemy in radius.
@phase(event)
physics NovaFire {
  on Tick {} where (self.alive
                    && self.mana >= config.vs.player_mana_min
                    && self.mana <= config.vs.player_mana_max
                    && world.tick % config.vs.nova_period == 0) {
    for target in spatial.enemies_in_radius(self) {
      emit Damaged { source: self, target: target, amount: config.vs.nova_damage }
    }
  }
}
```

**Probe note:** if a `@top_k(32)`-capped query is the only available "all in radius" form, that's the AOE shape (capped at 32 hits/burst — fine for the fixture). If an *uncapped* radius enumeration is desired and unsupported, log "AOE = capped top_k only" as a minor finding. Reuse the same body-vs-where gate decision from Task 4.

- [ ] **Step 4: Run all tests**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`
Expected: PASS (five tests). Both `BoltFire` (single-target) and `NovaFire` (AOE) emit in the same tick — the mixed-dispatch milestone.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(vampire_survivors): nova AOE auto-fire (mixed dispatch with bolt)"
```

---

## Task 6: XP fold + power ramp (PROBE #1 — view read in physics emit amount)

**Files:**
- Modify: `assets/sim/vampire_survivors.sim`
- Modify: `crates/dsl_compiler/tests/vampire_survivors_compile.rs`
- Modify: `docs/superpowers/specs/2026-05-24-vampire-survivors-design.md` (ledger, if it walls)

- [ ] **Step 1: Add the XP-fold test (always-true part)**

```rust
#[test]
fn xp_view_folds_kills() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let xp = kernel_body_containing(&art, "xp")
        .unwrap_or_else(|| panic!("no xp fold kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(xp.contains("view_storage"), "xp fold should write view storage; got:\n{xp}");
}
```

- [ ] **Step 2: Run to confirm fail**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile xp_view -- --nocapture`
Expected: FAIL — `no xp fold kernel`.

- [ ] **Step 3: Add the XP view (folds Killed by killer)**

Mirrors `kill_count` (`predator_prey.sim:161-167`) / `defender_damage_dealt` (`tower_defense.sim:189-194`):

```
@materialized(on_event = [Killed])
view xp(by: Agent) -> f32 {
  initial: 0.0,
  on Killed { by: p, prey: _, pos: _ } { self += 1.0 }
  clamp: [0.0, 1000000.0],
}
```

- [ ] **Step 4: Run the XP-fold test**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile xp_view -- --nocapture`
Expected: PASS.

- [ ] **Step 5: PROBE — read `xp(self)` inside the BoltFire emit amount**

Change `BoltFire`'s emit amount from the constant to a ramped value:

```
emit Damaged { source: self, target: target, amount: config.vs.bolt_damage + xp(self) * config.vs.bolt_ramp }
```

Add `bolt_ramp: f32 = 0.5` to `config vs`.

- [ ] **Step 6: Run the full suite and classify the outcome**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`

- **If it PASSES:** view-value reads lower inside a physics emit amount. Update spec §8 ledger row #1 to "PASS — confirmed". Add an assertion that the `BoltFire` body references the xp view storage:
  ```rust
  // inside bolt_fires_and_damage_applies, after the existing asserts:
  assert!(bolt.contains("view_storage"), "BoltFire amount should read the xp view; got:\n{bolt}");
  ```
  Re-run; expected PASS.
- **If it WALLS** (a `lower:`/`emit:` error): copy the verbatim error into spec §8 ledger row #1 ("WALL — <error>; candidate primitive: view read in physics emit amount"). Apply the **fallback**: revert the amount to the constant `config.vs.bolt_damage` (the ramp moves to Slice 5's upgrade system). Re-run; expected PASS on the constant form.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(vampire_survivors): XP-on-kill view; probe XP-ramped bolt damage (ledger #1)"
```

---

## Task 7: Discrete leveling (PROBE #2 — `floor` on a view)

**Files:**
- Modify: `assets/sim/vampire_survivors.sim`
- Modify: `crates/dsl_compiler/tests/vampire_survivors_compile.rs`
- Modify: spec §8 ledger

- [ ] **Step 1: PROBE — derive level via `floor` and gate a stat tier**

`floor`/`ceil`/`round` lower in the compiled path (`crates/dsl_compiler/src/cg/lower/expr.rs:2021`), so this is expected to pass via the compile-gate. Add `xp_per_level: f32 = 5.0`, `nova_radius_step: f32 = 1.0` to `config vs`, and make NovaFire's radius level-scaled. Since the query radius is a config literal, scale the *damage* by level inside the emit instead (radius scaling waits on parameterised queries — note it):

```
emit Damaged { source: self, target: target, amount: config.vs.nova_damage + floor(xp(self) / config.vs.xp_per_level) * config.vs.nova_radius_step }
```

- [ ] **Step 2: Add the leveling test**

```rust
#[test]
fn nova_damage_scales_with_floor_level() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let nova = kernel_body_containing(&art, "NovaFire").expect("NovaFire kernel");
    // floor lowers to a WGSL floor() call in the compiled path.
    assert!(nova.contains("floor("), "NovaFire amount should contain floor(...); got:\n{nova}");
}
```

- [ ] **Step 3: Run and classify**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile nova_damage_scales -- --nocapture`
- **If PASS:** update spec §8 ledger #2 to "PASS in compiled path (interp arm still missing — `eval/builtins.rs`)".
- **If WALL:** copy the error to ledger #2; fallback = drop the `floor(...)` term (constant nova damage) and record the wall. Re-run expecting PASS on the fallback.

Note: this step depends on Task 6's XP-ramp branch. If Task 6 walled (xp-read-in-amount unsupported), then reading `xp(self)` here also walls for the same reason — in that case record "blocked by ledger #1" for #2 and use the fallback (constant damage).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(vampire_survivors): floor-based discrete leveling probe (ledger #2)"
```

---

## Task 8: Upgrade choice (PROBE #3 + #3a — the big gap)

**Files:**
- Modify: `assets/sim/vampire_survivors.sim`
- Modify: `crates/dsl_compiler/tests/vampire_survivors_compile.rs`
- Modify: spec §8 ledger

- [ ] **Step 1: PROBE #3a — attempt a standalone `enum`**

Add to the top of the `.sim` (no corpus precedent — this may not parse):

```
enum UpgradeKind { BoltDamage, NovaRadius, BoltRate, MoveSpeed }
```

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile vampire_survivors_compiles -- --nocapture`
- **If `parse:`/`resolve:` error:** remove the `enum`. Record spec §8 ledger #3a "WALL — no user `enum` surface: <error>". Model upgrade kinds as **distinct events** instead (next step uses `UpgradeBolt` / `UpgradeNova`).
- **If PASS:** record #3a "PASS". You may key the tally view on the enum.

- [ ] **Step 2: PROBE #3 — level-up emits an upgrade chosen by fixed priority**

Add a `LevelUp`-style rule that, when the player crosses a level boundary, selects ONE upgrade by fixed priority among eligible (below-cap) options. Attempt the natural expression first:

Add `bolt_cap: f32 = 5.0` and `nova_cap: f32 = 5.0` to the existing `config vs` block (a single config block per fixture is the corpus norm — do **not** add a second block). Then:

```
@replayable @gpu_amenable
event UpgradeChosen { player: AgentId, kind: u32 }

// Per-tick: pick the highest-priority not-yet-capped upgrade.
// PROBE: requires selecting among a fixed option set with a state filter.
@phase(event)
physics ChooseUpgrade {
  on Tick {} where (self.alive
                    && self.mana >= config.vs.player_mana_min
                    && self.mana <= config.vs.player_mana_max) {
    let chosen =
      if bolt_upgrades(self) < config.vs.bolt_cap { 0u }
      else if nova_upgrades(self) < config.vs.nova_cap { 1u }
      else { 2u };
    emit UpgradeChosen { player: self, kind: chosen }
  }
}
```

with two tally views:

```
@materialized(on_event = [UpgradeChosen])
view bolt_upgrades(player: Agent) -> f32 {
  initial: 0.0,
  on UpgradeChosen { player: p, kind: k } where k == 0u { self += 1.0 }
  clamp: [0.0, 100.0],
}
@materialized(on_event = [UpgradeChosen])
view nova_upgrades(player: Agent) -> f32 {
  initial: 0.0,
  on UpgradeChosen { player: p, kind: k } where k == 1u { self += 1.0 }
  clamp: [0.0, 100.0],
}
```

This deliberately probes several surfaces at once: a value-returning `if/else if/else` chain selecting a `u32` (the "choice"); reading a view (`bolt_upgrades(self)`) inside a physics `if` condition; and an event-fold `where` guard on a `u32` payload field (`k == 0u`).

- [ ] **Step 3: Add a test for whatever shape lands**

```rust
#[test]
fn upgrade_choice_compiles() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    assert!(
        kernel_body_containing(&art, "ChooseUpgrade").is_some()
            || kernel_body_containing(&art, "bolt_upgrades").is_some(),
        "expected an upgrade-selection or tally kernel; have {:?}",
        art.wgsl_files.keys().collect::<Vec<_>>(),
    );
}
```

- [ ] **Step 4: Run and classify (the headline benchmark result)**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`
- **If PASS:** the value-returning `if/else` selection lowers — record spec §8 ledger #3 "PASS — fixed-priority selection expressible via value-returning if/else over view-gated conditions". This is a strong positive finding (the branching-selection gap is narrower than feared).
- **If WALL on the `if/else`-returns-value selection:** copy the error to ledger #3 ("candidate primitive: value-returning multi-arm `select`/`match` expression"). Fallback: replace the choice with a **fixed level-indexed unlock** — gate each weapon's stat term directly on `floor(xp(self)/k) >= threshold` (no runtime selection), which still gives a deterministic upgrade curve. Re-run expecting PASS on the fallback.
- **If WALL only on reading a view inside an `if` condition:** record that narrower finding; fallback = compute the choice from `floor(xp/k)` parity (e.g., `kind = level % 3`) instead of from tallies.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(vampire_survivors): upgrade-choice probe — enum + priority selection (ledger #3/#3a)"
```

---

## Task 9: Finalize ledger + fixture header

**Files:**
- Modify: `assets/sim/vampire_survivors.sim` (header comment)
- Modify: `docs/superpowers/specs/2026-05-24-vampire-survivors-design.md` (§8 ledger final state)

- [ ] **Step 1: Write the fixture header comment**

Replace the top comment of `vampire_survivors.sim` with a composition + findings summary in the style of `predator_prey.sim:1-21` — list the rules (IdleDrift removed if now unused; ChasePlayer, KitePlayer, BoltFire, NovaFire, ApplyDamage, ChooseUpgrade), the views (xp, bolt_upgrades, nova_upgrades), and a one-line-per-probe outcome pointer to the spec ledger.

- [ ] **Step 2: Finalize the spec gap ledger**

Update spec §8 so every row (#1–#5, #3a) reads PASS / WALL with the verbatim outcome observed in Tasks 6–8. Add a one-paragraph "Findings summary" under the table stating, in plain terms, how much of VS the DSL expressed and which (if any) primitives are the next extensions.

- [ ] **Step 3: Remove the now-redundant IdleDrift seed rule (if unused)**

If `ChasePlayer`/`KitePlayer` now guarantee kernels for both roles, delete `IdleDrift` from the `.sim`. Run the full suite to confirm still green:

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`
Expected: PASS (all tests).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs(vampire_survivors): finalize gap ledger + fixture header"
```

---

## Final verification

- [ ] Run the full fixture suite: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture` → all PASS.
- [ ] Confirm no regression in the broader compiler suite: `cargo test -p dsl_compiler` → PASS.
- [ ] Confirm the spec §8 ledger reflects actual observed outcomes (no "Likely"/"Expected" left — every probe is PASS or WALL with evidence).
- [ ] Confirm `git log --oneline` shows one commit per task, all on `worktree-floating-jumping-ladybug`.
