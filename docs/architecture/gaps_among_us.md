# Gaps surfaced by `among_us` fixture (2026-05-11)

The `among_us` fixture (assets/sim/among_us.sim, pin
crates/sims/tests/among_us_pin.rs) is a social-deduction adversarial
fixture stacking asymmetric perception, cross-pair belief writes,
LoS-gated witness rules, and per-Crew vote scoring. It surfaced 5
gaps during construction. The fixture is structurally clean (21
kernels emit, all 500 ticks step without panic) but **behaviourally
inert** — same chronicle-consumer-not-firing gap as `hill_raid`.

Pin output: 17 Crew + 3 Imposter, 500 ticks, mean position radius
8.25 (movement works), task indices reach 5/8 (task waypoint cycling
works), but **0 kills, 0 votes, 0 publicly accused**.

---

## Gap #1 — Chronicle-consumer Indirect-dispatch from synthesized `step()`

**Same gap hill_raid surfaced (commit 78ad8a77).** apply_ability
records enqueue into the unified event ring, but the consumer
kernels (ApplyDamageFromChronicle, ApplyDamage, ApplyWitness,
ApplyVoteFromChronicle) don't fire from the auto-emitted `step()`.

**Source location**: `crates/dsl_compiler/src/build_helper.rs`
catch-all match arm (the `_ => {}` in `synthesize_generated_runtime_struct`
that silently drops `DispatchOp::Indirect` schedule entries).

**Symptom in pin**: `total_event_load=0`, `imposter_publicly_accused=0`,
no agents flip alive=0 despite 500 ticks of ImposterKill firing
inside spatial range.

**Workaround precedent**: `crates/sims/tests/tom_probe_helpers/mod.rs`
hand-rolls the inject + dispatch in synchronous `dispatch_observe` /
`dispatch_scry` / etc. wrappers. The Among Us pin DELIBERATELY does
not adopt this — the gap is the load-bearing finding.

**Fix-blocker**: Documented as a four-gap coordinated change at the
catch-all arm; not in scope for this fixture.

---

## Gap #2 — `terrain.line_of_sight` in `@phase(post)` chronicle consumer

**Status**: investigated 2026-05-12 — **phantom gap**, fix already
in place; regression locked in.

**Original design**: ApplyWitness was supposed to gate witness writes
on `terrain.line_of_sight(self.pos, killer_pos)` so witnesses behind
cover wouldn't update their belief. This is the LoS-gated belief-
write surface the user specifically called out as a likely gap.

**Investigation**: the voxel_grid binding synthesis in
`cg/emit/kernel.rs` (the substring scan around `terrain_line_of_sight(`
≈line 810) runs after `wgsl_body` is composed — kernel-kind agnostic.
PerEventEmit (chronicle-consumer) kernels go through the same generic
binding pipeline as PerAgent physics, so the scan fires for both. The
host-side dispatch (`build_helper.rs` line 2348+) wires
`KernelBindingsContext::voxel_grid: Some(...)` whenever ANY kernel's
spec binds `voxel_grid` — chronicle-consumer kernels included.

Verified end-to-end by temporarily adding `&& terrain.line_of_sight(
slot_pos, killer_pos)` to `ApplyWitness`'s `for_each_agent` body
gate: `cargo build -p sims --release` succeeded, the emitted
`physics_ApplyDamage_and_ApplyWitness.wgsl` declared `voxel_grid` at
slot 9 with a clean `cs_*` entry point, naga validated every kernel,
and the 500-tick `among_us_pin` ran green. Among_us .sim itself NOT
modified in this commit — the LoS gate is a behavioural change held
back as a follow-up so the binding-side fix lands without coupling.

**Regression**: `crates/dsl_compiler/tests/voxel_query_in_chronicle_consumer.rs`
pins both the positive (consumer with LoS → `voxel_grid` binding,
`ctx.voxel_grid.expect(...)` routing, naga-clean WGSL) and negative
(consumer without any terrain call → NO `voxel_grid` binding) shapes.

---

## Gap #3 — WGSL emit ordering for fused `set_pos` + `set_mana` in same physics body

**Status**: surfaced + worked around.

**Symptom**: declaring AgentTaskSteer with both `agents.set_pos(self,
new_pos)` and `agents.set_mana(self, new_task)` in a single body
produced WGSL referencing `local_9` BEFORE its declaration:

```wgsl
agent_pos[agent_id] = local_9;     // <-- USE
var local_9: vec3<f32>;            // <-- DECLARATION (later)
loop {
    ...
    local_9 = select(agent_pos[agent_id], ...);  // <-- ASSIGNMENT
    ...
}
```

The `mana` write triggers an f32-RMW CAS loop because mana is stored
as `array<atomic<u32>>` (engine-wide convention to make atomic
mutators applicable to f32 values via bitcast). The CAS loop is
emitted SECOND, but the emitter floats the `pos` write to BEFORE
the loop — past the `local_9` definition that the loop introduces.

**Source location**: `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`
(the f32-on-atomic-u32 emit path that builds the CAS loop). Likely
related to commit `123e8cd6` (view-fold `self = <expr>` lowers to
atomicStore).

**Workaround**: split AgentTaskSteer into two separate physics rules
— AgentTaskSteer (writes pos only) and AgentTaskAdvance (writes mana
only). Same source-text logic, separate per-rule SSA isolation.
After the split, the fused kernel
`physics_AgentTaskSteer_and_AgentTaskAdvance` emits cleanly because
the per-rule body has only one mutator.

**Conjectured root cause**: the emitter computes the use site for
`local_9` from the consuming statement (`set_pos`), but the CAS-loop
construction inserts a fresh `var` declaration that masks the
already-emitted use. Fix would either:
  (a) emit `pos` write inside the CAS loop alongside the `mana` write,
      treating both as part of the same atomic transaction, OR
  (b) hoist the `local_9 = select(...)` computation OUT of the loop
      so the use site sees a defined value.

---

## Gap #4 — Multi-view aliasing in `view_storage_primary`

**Status**: surfaced + worked around.

**Symptom**: declaring two per-agent `@materialized` views in the
same fixture (`kills_by_source(source)` + `votes_against(target)`)
silently aliases their writes — both fold kernels bind to the same
`view_storage_primary` array, indexing by their respective key
field. With overlapping key spaces (e.g. an Imposter slot is BOTH a
kill source AND a vote target), the per-slot byte at
`view_storage_primary[i]` accumulates BOTH counts.

**Detection**: pin's `read_view_votes_against` computed an offset of
`N_TOTAL * 4` bytes (assuming contiguous-slab layout), tried to
`copy_buffer_to_buffer` 160 bytes from an 80-byte source —
validation error: `Copy of 0..160 would end up overrunning the bounds
of the Source buffer of size 80`. This is the bug-revealing failure
mode: the build_helper sized the buffer for ONE view, not two.

**Source location**: `crates/dsl_compiler/src/build_helper.rs`
view-storage allocation logic (around line 880-940). The "primary"
buffer is sized as `max(per_view_storage_words)` rather than `sum`;
all per-agent views share the buffer with no per-view offset
multiplexing.

**Workaround**: collapsed two views into one (`event_load`) with a
disjunctive gate covering both event types. Per-slot tally is the
SUM of (kills authored, votes received) — pin host-side decomposes
via per-slot `creature_type` + `hunger`-flag inspection.

**Fix shape**: build_helper should allocate one buffer per
materialized view (or one slab with per-view byte-offset bindings).
Each fold kernel's BGL would bind its own view-specific slot.

---

## Gap #5 — Cross-pair belief read in scoring expression

**Status**: known-gap precedent; design REROUTED to public-flag stand-in.

**Original design**: Vote score = argmax over `agents.beliefs_suspicion(self,
candidate)` per the canonical Among Us rule (each Crew picks the
suspect THEY believe most). This requires reading a per-pair belief
column from the SCORING kernel's BGL, addressed by `[self_slot *
agent_cap + candidate_slot]`.

**Why blocked**: spy_network and tom_probe both READ per-pair beliefs
in CHRONICLE CONSUMER bodies (`agents.beliefs_*(o, s)` reads in
`@phase(post)` physics rules), never in scoring expressions. The
scoring kernel BGL composer in
`crates/dsl_compiler/src/cg/emit/scoring.rs` doesn't currently bind
the `BeliefStateColumn::Suspicion` storage handle — only the
chronicle-consumer BGL composer does.

**Replacement (collective-suspicion stand-in)**: ApplyWitness writes
a global per-killer flag (`agents.set_hunger(killer, 1.0)`) in
addition to the per-pair `beliefs_suspicion(witness, killer) = 255`.
The Vote score reads `target.hunger` (a per-agent f32 column already
wired through scoring kernel BGLs). All Crew vote against the SAME
publicly-accused agent — collective suspicion rather than per-Crew
belief.

**Fix shape**: extend the scoring-kernel BGL composer to surface
BeliefStateColumn bindings when the score expression reads
`agents.beliefs_<field>(observer, subject)` — same shape as the
chronicle-consumer BGL composer's existing handling. This unblocks
authentic per-Crew vote scoring across the entire ToM-bearing fixture
class.

**Witness consumer DOES write per-pair beliefs**: the gap is in the
SCORING surface, not the chronicle-consumer surface.
ApplyWitness's `agents.set_beliefs_suspicion(slot, killer, 255)`
write inside `for_each_agent` lowers cleanly (the kernel emits a
fused `physics_ApplyDamage_and_ApplyWitness` with 10 bindings
including `beliefs_suspicion` and `beliefs_flags`). The per-pair
data IS being written every tick a kill fires — it's just not
readable from the scoring kernel.

---

## Adjacent observations (non-blocking)

**A. `if`-expression body must be single expression.** Original
attempt at `let new_task = if (...) { let next = self.mana + 1.0;
... }` failed with `expected '}' while parsing if expr '}'`. The DSL
parser today rejects `let`-prefixed if-expr arms. Worked around by
hoisting the `let` outside the if-expression.

**B. `cos`/`sin` builtins missing.** `Builtin::Length`, `Sqrt`, `Dot`
exist; trig functions do not as of 2026-05-11. Worked around with a
hand-rolled 8-way if-cascade for compass-pattern waypoint placement
(0.7071068 ≈ sqrt(2)/2 hardcoded as the diagonal scalar). For
fixtures that need richer pathing geometry, this becomes painful;
adding `Cos`/`Sin`/`Atan2` to the `Builtin` enum is a small slice
that would unblock cleaner agent-distribution patterns.

**C. `set_max_mana` / `set_armor` not in agents-setter map.**
Re-purposing standard SoA columns as observability flags requires
the column to have a registered `set_<name>` lower path. Today only
the columns enumerated in `agents_setter_field` (in
`crates/dsl_compiler/src/cg/lower/physics.rs` line 916+) are
writable. We chose `set_hunger` (already wired for foraging_real's
energy-counter repurpose). Adding setters for the ~4 standard
columns missing from this list is mechanical follow-up work.

**D. `plant_belief` accepts `bit` keyword in proper form.** Vote.ability
v1 used `plant_belief 0 0` (matching spy_network/Slander.ability's
warning-but-works form). Corrected to `plant_belief 0 bit 0` — the
`bit` keyword is the in-band separator the lowering expects per
`crates/dsl_compiler/src/ability_lower.rs:1857`. spy_network's
shipped form is also non-canonical and emits the same lower
warning; both are non-blocking but should standardise.

---

## What's verified vs what's deferred

**Verified end-to-end (compile + step + readback)**:
- 21 kernels emit (3 PerAgent, 4 chronicle consumers, 1 verb chronicle,
  1 mask, 1 fold view, 1 scoring, 5 spatial build, 5 infrastructure).
- Movement (PerAgent kernel `physics_AgentTaskSteer`) — agents
  cycle through 8 task waypoints. Mean task_idx after 500 ticks =
  3.15, max = 5. Pathfinding-via-steering works as designed.
- `physics_ApplyDamage_and_ApplyWitness` lowers cleanly with 10
  bindings — the for_each_agent body inside @phase(post), cross-pair
  belief writes from chronicle consumer, agents.creature_type read,
  and per-pair beliefs_suspicion + beliefs_flags writes ALL compose
  end-to-end without surfacing a lowering gap. This is the single
  most adversarial-axis-stacked rule body in the fixture and it
  emits structurally; only the chronicle dispatch (Gap #1) blocks
  observation.

**Deferred until upstream gaps close**:
- Behavioural validation of Imposter Kill firing → Crew survival
  reduction (blocked on Gap #1).
- Witness consumer per-pair belief writes producing observable
  `beliefs_suspicion` cells (blocked on Gap #1; the consumer kernel
  exists but doesn't dispatch from `step()`).
- Vote scoring picking the publicly-accused Imposter (blocked on
  Gap #1 + Gap #5; the scoring kernel exists but the
  `hunger`-stand-in flag is never written because ApplyWitness
  doesn't fire).
- Per-Crew belief-driven voting (blocked on Gap #5 even after Gap
  #1 closes).

**To re-run pin once Gap #1 closes**:
```
RUST_MIN_STACK=33554432 cargo test -p sims --release --test among_us_pin -- --nocapture
```

Expected behaviour shift after Gap #1 fix (rough prediction):
- 3 Imposters reach crew within ~30 ticks (1.5 unit strike radius
  + 0.20 imposter step / closure, average distance ~5).
- ~5-15 kills per Imposter over 500 ticks (cooldown 8 ticks gates
  rate; Crew dispersal slows close-in-again loops).
- `imposter_publicly_accused` should hit 3/3 (every Imposter that
  performs a kill triggers ApplyWitness → hunger=1.0).
- Crew should SHRINK below 17 if Imposter kills land before votes
  eject them.
- Vote should fire every 50 ticks for every alive Crew, all voting
  against the same publicly-accused Imposter slot.
