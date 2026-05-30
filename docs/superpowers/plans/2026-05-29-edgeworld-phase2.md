# Edgeworld Phase 2 — Belief-Gated Flee (Imperfect Perception) Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Survivors flee based on a *decaying threat belief*, not ground truth — so they forage bravely until fear builds (reaction lag), keep fleeing after a wolf leaves (lingering fear), and sometimes mis-judge. This showcases the engine's differentiator (first-class `belief` primitive) and softens the panic-flee dynamic.

**Architecture:** Add a `belief threats(observer: Agent) -> f32` with `@decay` (the `dodger_probe`/`threats_view_probe` pattern), fed by a `WolfSpotted` perception event a survivor emits when it sees a wolf in range. Gate the Flee/forage behavior on the belief level instead of direct wolf-visibility.

**Tech stack:** `.sim` DSL belief primitive + `@decay`; the Phase 0/1 edgeworld runtime.

---

## AIS (P8)
- P1: behavior in DSL only. P2: belief storage is view-allocated, no engine SoA column add. P3/P5/P11: belief fold is `+=` accumulate + `@decay` (commutative); RNG (if mis-perception uses chance) via keyed surface. P7: `WolfSpotted` flagged `@replayable`.
- **Key risk:** Edgeworld movement is `@phase(per_agent)` physics rules (direct `set_pos`), but the proven belief-READ wire-up (`threats.intensity_at(self)`) is in the *scoring/verb* path (`dodger_probe`). Whether a physics rule can read the threat belief is the central unknown — Task 1 resolves it, with two fallbacks (materialize belief→column; or convert flee/forage to verbs).

## Reference fixtures
- `assets/sim/dodger_probe.sim` — `belief threats(observer)->f32`, `threats.intensity_at(self)` read in a verb `score`, Flee-vs-Idle argmax. THE template.
- `assets/sim/threats_with_decay_probe.sim` — `@decay` on the threats belief.
- `crates/sims/tests/threat_stresstest_pin.rs`, `tom_probe_belief_gated_threat_pin.rs` — runtime readback of belief/threat storage.
- `assets/sim/edgeworld.sim` — current Flee rule (`@phase(per_agent)`, away-from-nearest-wolf within flee_range, inline world-clamp, DCE anchor).

---

## Task 1: Threat belief that accumulates on perception and decays

Add the belief + perception event; prove it rises when a survivor sees a wolf and decays otherwise (runtime readback). No behavior change yet.

**Files:** `assets/sim/edgeworld.sim`, `crates/sims/tests/edgeworld_pin.rs`, maybe `edgeworld_common/mod.rs` (a belief-storage readback helper).

- [ ] **Step 1:** Add a `@replayable @gpu_amenable event WolfSpotted { observer: AgentId }`. Add a `Perceive` physics rule: a survivor (`creature_type==1`) that has a wolf (`==2`) within a perception spatial query emits `WolfSpotted { observer: self }`. Model the spatial loop on the existing Flee rule.
- [ ] **Step 2:** Add the belief (model exactly on `dodger_probe.sim` + `threats_with_decay_probe.sim`):
```
@dispatch(per_agent_event_scan)
@decay(rate = 0.90, per = tick)
belief threats(observer: Agent) -> f32 {
  initial: 0.0,
  on WolfSpotted { observer: o } where (o == observer) { self += 1.0 }
  clamp: [0.0, 100.0],
}
```
Confirm exact `@decay`/belief syntax against the reference fixtures; the view name MUST be `threats` for the read wire-up. Iterate against the compiler.
- [ ] **Step 3:** Test: seed 1 survivor + 1 wolf within perception. Step ~10 ticks, read the threats belief storage (find the buffer field, e.g. `view_storage_threats_primary_buf`, readback like detective's view reads), assert the survivor's threat level ROSE above 0. Then move the wolf far away (or kill it) and step ~20 more ticks; assert the threat level DECAYED back down. This proves rise + decay.
- [ ] **Step 4:** Run all pins green (existing 12 + new), commit `feat(edgeworld): decaying threat belief fed by wolf perception`.

## Task 2: Gate flee + forage on the belief (resolve the read-location fork)

Make behavior depend on the belief, not direct visibility. **Resolve whether a physics rule can read the belief.**

**Files:** `assets/sim/edgeworld.sim`, `crates/sims/tests/edgeworld_pin.rs`.

- [ ] **Step 1 — probe the read:** Try reading the threat belief inside the `Flee` physics rule's guard/body — e.g. gate flee on `threats.intensity_at(self) > config.edgeworld.fear_threshold`, or read the belief storage value. Build. 
  - If a physics rule CAN read it: gate Flee on `belief > fear_threshold` (so the survivor only flees once fear has built — reaction lag), and gate SeekFood to keep foraging while `belief <= fear_threshold` (brave foraging). Direction of flee stays "away from nearest visible wolf"; when belief is high but no wolf is visible (lingering fear), the survivor stays wary (suppress SeekFood) so fear has a foraging cost.
  - **Fallback A (belief not readable in physics):** add a `@phase(post)` fold that writes the per-observer threat level into a repurposed survivor SoA column (e.g. `shield_hp`), and have the physics Flee/SeekFood rules read THAT column (proven column-read pattern). The belief still owns the decay.
  - **Fallback B:** convert Flee/forage to the verb/scoring system (the dodger pattern, where belief-read is proven) — heavier; only if A fails.
  Document which path worked.
- [ ] **Step 2 — config:** add `fear_threshold: f32 = 1.5` (tunable hysteresis point).
- [ ] **Step 3 — tests:** (a) **reaction lag** — survivor next to a wolf does NOT flee on tick 1 (belief below threshold) but DOES by tick ~3 (belief accrued past threshold). (b) **lingering fear** — after the wolf leaves perception, the survivor stays wary/fleeing for several ticks until the belief decays below threshold, then resumes foraging.
- [ ] **Step 4:** Run all pins green, commit `feat(edgeworld): belief-gated flee — reaction lag + lingering fear`.

## Task 3: Tune + render the imperfect-perception dynamic

**Files:** `assets/sim/edgeworld.sim` (tuning), `crates/sims/tests/edgeworld_render.rs`, `edgeworld_pin.rs`.

- [ ] **Step 1:** Color survivors by fear in the render (e.g. amber→bright-red tint scaled by threat belief) so you can SEE who's scared. Read the belief in the render loop and map it.
- [ ] **Step 2:** Tune `@decay` rate, `fear_threshold`, perception so the dynamic is legible: survivors forage near food, spook and scatter when a wolf approaches (with a slight lag), and re-settle to foraging after it passes (with lingering wariness). Keep coexistence (both populations alive at 600) — belief-gated flee should let survivors forage MORE than the always-flee Phase 1, so the remnant may be larger/healthier. Update/extend the coexistence pin.
- [ ] **Step 3 — VISUAL self-verify** (don't wait for human review): open early/mid/late frames, confirm fear-tinting reacts to wolves with lag/linger and foraging resumes. Iterate.
- [ ] **Step 4:** All pins + render green, commit `feat(edgeworld): tuned belief-gated perception dynamics; Phase 2 complete`. Report 3 sign-off frames.

## Self-review
- Differentiator: uses the real `belief` primitive with `@decay` (not a hand-rolled column) wherever the read-location allows; fallback A only if physics-read is blocked.
- Coexistence pin must stay green; belief-gated foraging should improve survivor health vs Phase 1.
- The view name `threats` is load-bearing for the read wire-up — don't rename.
