# Edgeworld Phase 1 — Predators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `Wolf` predators that hunt the survivor band, turning survival into a food-vs-safety trade-off — and providing a genuine, position-dependent cull that doesn't depend on the (broken) food-depletion race.

**Architecture:** Extend `assets/sim/edgeworld.sim` with a `Wolf : Agent` entity, wolf hunger/starvation (predators are under survival pressure too, so they don't overrun), a WolfHunt rule (steer toward nearest survivor; kill within range via direct neighbour `set_alive`; killing resets wolf hunger), and a survivor Flee rule (steer away from nearby wolves). Render adds red wolf dots.

**Tech Stack:** `.sim` DSL, the Phase 0 runtime (`sims::edgeworld::GeneratedRuntime`), `image` PNG render. All patterns below are already proven in Phase 0.

---

## Architectural Impact Statement (P8)

- **P1:** All behavior in `edgeworld.sim`; no hand-written engine/WGSL. ✅
- **P2:** No new SoA columns. Wolf hunger rides on the shared `hunger` column (rules gate by `creature_type`). No schema-hash bump. ✅
- **P3/P5/P11:** Deterministic; RNG (if any) via the engine's keyed surface; kill is a neighbour `set_alive` (boolean store, idempotent — no float-reduction race). Wolf-on-survivor kill is N-to-1 but the target is a boolean set to a constant `false`, so last-writer-wins is harmless (all writers write the same value). ✅
- **P7:** No new replayable events needed (kills are direct `set_alive`); if an event is introduced, flag `@replayable`. ✅
- **Carried risk:** the `set_pos`-only DCE drop ([[project_edgeworld_dsl_findings]] finding 1) applies to WolfHunt and Flee — both must anchor their `set_pos` with a scalar self-write. The `@spatial(radius)` ~6-unit cap (finding 2) applies — keep the world compact.

## Proven Phase 0 patterns to reuse (don't re-derive)

- Per-agent hunger fold: `physics Hunger @phase(per_agent) { on Tick {} where (self.alive && self.creature_type == 1) { let p = agents.hunger(self); agents.set_hunger(self, p + config.edgeworld.hunger_rate); } }`.
- Death: `agents.set_alive(self, false)` (self) — and neighbour `set_alive(candidate, false)` is the same neighbour-write path as the proven `set_mana(candidate, …)`.
- Spatial loop reading neighbour cols + `candidate.pos`: `for candidate in spatial.q(self) { if (candidate.alive && candidate.creature_type == K) { … candidate.pos … } }`.
- Movement integrator: `agents.set_pos(self, self.pos + dir * speed)` — **must be anchored** by a scalar self-write in the same rule or the kernel is silently dropped.
- `distance(a, b)` is a numeric builtin — usable to gate kill/flee on real proximity within the perception ring.
- creature_type discriminants are alphabetical decl order. **Add `Wolf` (not `Predator`/`Hunter`) so it sorts after `Survivor`: FoodNode=0, Survivor=1, Wolf=2 — existing `== 1` guards stay valid.**

---

## File structure

| File | Change |
|---|---|
| `assets/sim/edgeworld.sim` | Add `Wolf` entity, wolf config, WolfHunt, Flee; generalize hunger/starvation to wolves | Modify |
| `crates/sims/tests/edgeworld_common/mod.rs` | Add `CT_WOLF=2`; extend `seed_world` to place wolves; render color | Modify |
| `crates/sims/tests/edgeworld_pin.rs` | Add wolf-hunt + flee + Phase-1 dynamics pins | Modify |
| `crates/sims/tests/edgeworld_render.rs` | Render wolves as red dots; tune scenario | Modify |

---

## Task 1: Add the Wolf entity (no renumbering) + wolf hunger/starvation

**Files:** Modify `assets/sim/edgeworld.sim`, `crates/sims/tests/edgeworld_pin.rs`, `crates/sims/tests/edgeworld_common/mod.rs`.

- [ ] **Step 1: Declare the Wolf entity** — add AFTER `Survivor` (so alphabetical sort keeps Survivor=1, Wolf=2; VERIFY by checking entity order — `FoodNode`, `Survivor`, `Wolf`):
```
entity Wolf : Agent {
  pos: vec3,
  vel: vec3,
}
```

- [ ] **Step 2: Generalize hunger + starvation to wolves.** Wolves are under survival pressure too (or they overrun). Change the `Hunger` and `Starvation` rule guards from `self.creature_type == 1` to `self.creature_type != 0` (everything that isn't food gets hungry and starves). Add wolf-specific config so wolves can be tuned independently if needed:
```
  // wolves
  wolf_hunger_rate: f32 = 0.03,   // wolves get hungry slower than survivors
  wolf_move_speed:  f32 = 0.18,
  kill_range:       f32 = 1.2,
  flee_speed:       f32 = 0.25,   // > survivor move_speed so fleeing dominates
  flee_range:       f32 = 5.0,
```
For Phase 1 minimal, keep a SINGLE shared `hunger_rate` for the starve threshold logic but apply the wolf rate in a wolf-only hunger rule. Concretely: keep `Hunger` gated to survivors (`== 1`), add a parallel `WolfHunger` rule gated to wolves (`== 2`) using `wolf_hunger_rate`, and change `Starvation` to `self.creature_type != 0` so both starve at `hunger_max`. (This keeps survivor tuning independent of wolf tuning.)
```
physics WolfHunger @phase(per_agent) {
  on Tick {} where (self.alive && self.creature_type == 2) {
    let p = agents.hunger(self);
    agents.set_hunger(self, p + config.edgeworld.wolf_hunger_rate);
  }
}
```

- [ ] **Step 3: Add `CT_WOLF` + seed wolves.** In `edgeworld_common/mod.rs` add `pub const CT_WOLF: u32 = 2;`. Extend `seed_world(state, n_survivors, n_food, world_half)` to a new signature `seed_world(state, n_survivors, n_food, n_wolves, world_half)` placing wolves at the world edges (away from the central survivor cluster) with `hunger=0`, `alive=1`, `creature_type=2`. Update existing callers. Keep deterministic seeding.

- [ ] **Step 4: Smoke test** — seed e.g. 6 survivors + 2 food + 2 wolves, step 5 ticks, assert all three types are alive and counted correctly (a wolf-presence smoke test). Run `cargo test -p sims --test edgeworld_pin --release -- --nocapture`; fix DSL compile errors against Phase 0 syntax; confirm existing pins still pass (the `!= 0` guard change must not break survivor starvation).

- [ ] **Step 5: Commit** `git commit -m "feat(edgeworld): add Wolf predator entity + wolf hunger/starvation"`.

---

## Task 2: Wolves hunt and kill survivors

**Files:** Modify `assets/sim/edgeworld.sim`, `crates/sims/tests/edgeworld_pin.rs`.

- [ ] **Step 1: WolfHunt rule** — wolves steer toward the nearest survivor in range and kill within `kill_range`, resetting their own hunger on a kill. Anchor the `set_pos` with the hunger self-write (which is also the kill-feed, so it doubles as the DCE anchor):
```
@spatial(radius = 6.0, kind = [Agent])
spatial_query prey_in_sight(self: AgentId, candidate: AgentId) =
  candidate != self

physics WolfHunt @phase(per_agent) {
  on Tick {} where (self.alive && self.creature_type == 2) {
    for candidate in spatial.prey_in_sight(self) {
      if (candidate.alive && candidate.creature_type == 1) {
        let d = distance(self.pos, candidate.pos);
        if (d < config.edgeworld.kill_range) {
          // kill the prey + feed (reset wolf hunger)
          agents.set_alive(candidate, false);
          agents.set_hunger(self, 0.0);
        } else {
          // steer toward the prey; the set_hunger below anchors the
          // set_pos kernel (DCE-drop workaround) with an identity write
          let dir = candidate.pos - self.pos;
          agents.set_pos(self, self.pos + dir * config.edgeworld.wolf_move_speed);
          agents.set_hunger(self, agents.hunger(self));
        }
      }
    }
  }
}
```
NOTE: confirm `distance(self.pos, candidate.pos)` lowers (it's a documented numeric builtin; if the signature differs, check `stdlib_math_probe.sim` for the exact form). If `set_alive(candidate, false)` neighbour-write is rejected (it shouldn't be — same path as `set_mana(candidate,…)`), fall back to neighbour hp-damage. Confirm the two branches of the `if/else` both lower (mixed neighbour-write + self-write in a spatial loop).

- [ ] **Step 2: Kill test** — seed 1 wolf adjacent to 1 survivor (within kill_range), no food, step ~3 ticks, assert the survivor is dead and the wolf is alive with hunger reset to ~0. Then seed 1 wolf ~4 units from a survivor and assert the wolf moved toward it (distance decreased) over 10 ticks.

- [ ] **Step 3: Run all pins green, commit** `git commit -m "feat(edgeworld): wolves hunt + kill survivors"`.

---

## Task 3: Survivors flee from wolves (the food-vs-safety trade-off)

**Files:** Modify `assets/sim/edgeworld.sim`, `crates/sims/tests/edgeworld_pin.rs`.

- [ ] **Step 1: Flee rule** — survivors with a wolf within `flee_range` step directly away from the nearest wolf, at `flee_speed` (> `move_speed` so flight dominates the SeekFood nudge when both fire). Anchor `set_pos` with an identity hunger self-write:
```
@spatial(radius = 6.0, kind = [Agent])
spatial_query threat_in_sight(self: AgentId, candidate: AgentId) =
  candidate != self

physics Flee @phase(per_agent) {
  on Tick {} where (self.alive && self.creature_type == 1) {
    for candidate in spatial.threat_in_sight(self) {
      if (candidate.alive && candidate.creature_type == 2) {
        let d = distance(self.pos, candidate.pos);
        if (d < config.edgeworld.flee_range) {
          // away vector = self.pos - wolf.pos
          let away = self.pos - candidate.pos;
          agents.set_pos(self, self.pos + away * config.edgeworld.flee_speed);
          agents.set_hunger(self, agents.hunger(self)); // DCE anchor
        }
      }
    }
  }
}
```
NOTE: Flee and SeekFood both write `set_pos` per tick; they compose (net vector). Flee's higher speed makes it dominate when a wolf is in range — the intended food-vs-safety tension (a starving survivor near food but also near a wolf gets pulled both ways). If the two `set_pos` rules conflict at the scheduler level (both per-agent writing pos), confirm both kernels emit and the writes compose as expected; if one shadows the other, sequence them (Flee after SeekFood) so flight is the last word.

- [ ] **Step 2: Flee test** — seed 1 survivor between food and a wolf (wolf within flee_range), step ~8 ticks, assert the survivor's distance to the wolf increased (it fled) even with food nearby.

- [ ] **Step 3: Run all pins green, commit** `git commit -m "feat(edgeworld): survivors flee from wolves"`.

---

## Task 4: Render wolves + tune for a real predator-driven dynamic

**Files:** Modify `crates/sims/tests/edgeworld_render.rs`, `crates/sims/tests/edgeworld_common/mod.rs`, `crates/sims/tests/edgeworld_pin.rs`, and `assets/sim/edgeworld.sim` (config tuning only).

- [ ] **Step 1: Render wolves red** — in the render loop, draw `CT_WOLF` agents as a red blob (e.g. `Rgb([220,40,40])`), survivors amber, food green. Update `seed_world` call to include wolves.

- [ ] **Step 2: Phase-1 dynamics pin** — a pin asserting predators matter: run two 600-tick scenarios from the same seed, one with wolves and one without (n_wolves=0), and assert the surviving-survivor remnant is **lower with wolves** than without (predation culls), AND at least one wolf is still alive at the end (wolves sustain by feeding — they don't all starve). This proves the predator-prey coupling is real, not cosmetic.
```rust
#[test]
fn edgeworld_predators_reduce_remnant() {
    // ... seed_world(.., n_wolves=0) -> remnant_no_wolves
    // ... seed_world(.., n_wolves=4) -> remnant_with_wolves, wolves_alive
    assert!(remnant_with_wolves < remnant_no_wolves, "wolves should cull survivors");
    assert!(wolves_alive >= 1, "wolves should sustain by feeding, not all starve");
}
```

- [ ] **Step 3: Tune** `wolf_hunger_rate`, `kill_range`, `flee_*`, and wolf count until the pin passes AND the rendered frames legibly show wolves chasing/culling survivors (red dots pursuing amber, amber scattering, fewer amber over time). **Self-verify visually from the PNG frames each pass — don't wait for human review.** Keep the world compact (~`WORLD_HALF=8`) per the perception cap.

- [ ] **Step 4: Run all pins + render green, commit** `git commit -m "feat(edgeworld): render wolves + predator-driven culling dynamics; Phase 1 complete"`.

- [ ] **Step 5: Human sign-off checkpoint** — surface early/mid/late frames showing the hunt for the Phase 1 "is this a saga worth watching" sign-off.

---

## Self-review notes

- **Spec coverage:** Phase 1 of the design spec (predators that hunt the band, Flee enters behavior, food-vs-safety trade-off, borrows predator_prey) is covered by Tasks 1–4. The optional day/cold abiotic sub-step is deferred (spec says "later sub-step"). Huntable-prey-the-band-eats is deferred — wolves-hunt-survivors is the headline; survivor-hunts-prey can be a Phase 1.5 if desired.
- **Renumbering guard:** `Wolf` chosen specifically to sort after `Survivor` so discriminants don't shift. Task 1 Step 1 verifies entity order. CT_FOOD=0/CT_SURVIVOR=1/CT_WOLF=2 consistent across .sim and tests.
- **DCE anchor:** every `set_pos` rule (WolfHunt, Flee) pairs with a scalar self-write per [[project_edgeworld_dsl_findings]] finding 1.
- **Kill race is benign:** N wolves killing 1 survivor all write `set_alive(_, false)` — same constant, last-writer-wins harmless (unlike the float food race).
- **Genuine dynamics:** unlike Phase 0's staged cull, predation is position-dependent and self-limiting (wolves starve if they don't catch prey), so the Phase-1 dynamics pin tests real coupling, not a seeded artifact.
