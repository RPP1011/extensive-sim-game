# Combat Foundation — Abilities + Engagement + World Effects + Recursion

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** after this plan, agents `Cast` abilities with recursion-bounded effect dispatch; 8 `EffectOp` variants (5 combat + 2 world + 1 meta) round-trip through the cascade; timed debuffs expire at tick boundaries; **engagement** locks hostile agents in melee (tank-wall formations without body collision); masks gate invalid casts AND free moves when engaged.

**Architecture:** IR-level abilities are lowered programs held in an append-only `AbilityRegistry`; `CastHandler` (one cascade handler) decomposes a cast into subordinate events per `EffectOp`; each effect event has its own handler that folds into `SimState`. Engagement state lives in a new `hot_engaged_with` SoA field, updated at tick start (bidirectional invariant). Cast-gate and move-gate mask predicates respect engagement + cooldown. One unified tick-start phase handles (a) engagement update, (b) stun/slow/cooldown decrement, and (c) status-effect expiry event emission. Recursion depth bounded by the engine's existing `MAX_CASCADE_ITERATIONS = 8`.

**Tech stack:** Rust 2021, `glam`, `smallvec`, `ahash`, `sha2`. No new deps.

**Prerequisites:** Engine MVP + Plans 1, 2, 2.75, 3.0, 3.1, + state port (`da8aa9ce`). Specifically:
- `MicroKind::Cast` exists in the enum
- `CascadeRegistry` + lanes exist (`155a51df`)
- `EventId` + cause sidecar on `EventRing` (`31e45d16`)
- `AggregatePool<T>` is available (`c438f249`) — reused for `AbilityRegistry`
- State port landed (`c83de370..da8aa9ce`): `hot_shield_hp`, `StatusEffect`/`StatusEffectKind` types, `cold_status_effects`, `Inventory { gold: i64, commodities: [u16; 8] }`, `cold_inventory`, `Relationship`, `cold_relationships`, `Membership`, `cold_memberships`, `MemoryEvent`, `cold_memory`, `cold_creditor_ledger`, combat extras (`hot_armor`, `hot_magic_resist`, `hot_attack_damage`, `hot_attack_range`, `hot_mana`, `hot_max_mana`), psychological needs (`hot_safety`, `hot_shelter`, `hot_social`, `hot_purpose`, `hot_esteem`), personality 5-dim (`hot_risk_tolerance`, `hot_social_drive`, `hot_ambition`, `hot_altruism`, `hot_curiosity`). Inventory.gold is i64 — signed; TransferGold reads/writes via `cold_inventory[slot].gold`, not a hot field.

**Prerequisites delta (merger audit):** the following items from the original Plan 3.5 and Ability Plan 1 drafts are **already done** and are NO-OPs here:

| Original task | Superseded by |
|---|---|
| Ability P1 T4.1: add `hot_shield` | state port Task B (landed as `hot_shield_hp`) |
| Ability P1 T4.2: `StatusEffect` + `StatusEffectKind` types | state port Task C |
| Ability P1 T4.3: add `hot_gold` | state port Task H → gold lives in `cold_inventory.gold: i64` |
| Ability P1 T4.4: `Membership` + `Relationship` types | state port Tasks G, J |
| Ability P1 T4.5: `cold_status_effects` SoA slot | state port Task C |
| Plan 3.5 T2.partial: personality dims | state port Task E |

**Backend scope:** Serial-first per the 2026-04-19 spec rewrite. Per-`EffectOp` SPIR-V kernels land in Plan 6+ GPU porting — their dispatch signatures are designed to be GPU-friendly (POD args, no trait objects in the kernel body).

---

## Acceptance criteria (plan-level)

1. **Cast dispatch.** `Action::Cast { ability, target }` with valid mask bit → `CastHandler` fires → effect-specific events → `SimState` mutation.
2. **Eight effects live.** Damage / Heal / Shield / Stun / Slow / TransferGold / ModifyStanding / CastAbility all round-trip.
3. **Timed effects decrement deterministically.** Stun and Slow durations decrement at tick start; each expires exactly once, emitting `StunExpired` / `SlowExpired`.
4. **Mask gating.** `can_cast(agent, ability)` is false when caster is stunned OR cooldown not ready OR target not alive OR target out of range. `UtilityBackend` never emits a masked-off `Cast`.
5. **Engagement bidirectional + symmetric.** `engaged_with[a] == Some(b)` iff `engaged_with[b] == Some(a)`. Both forms proptest-verified.
6. **MoveToward slow + opportunity attack.** While engaged, `MoveToward` toward a non-engager target is slowed by `ENGAGEMENT_SLOW_FACTOR = 0.3` and triggers an opportunity attack from the engager.
7. **Recursion bounded.** A chain of 8 `CastAbility` effects resolves; a 9th emits `CastDepthExceeded` and is dropped. Self-recursive ability self-terminates at 8 iterations with no state corruption.
8. **World effects mutate state.** `TransferGold` debits source, credits target (signed i64, debt allowed); `ModifyStanding` shifts per-pair standing clamped to `[-1000, 1000]`.
9. **Determinism.** Four acceptance fixtures (combat 2v2 with abilities; tax-ability world; meteor-swarm recursive; tank-wall formation) each produce bit-exact `replayable_sha256` hashes across two runs and across debug/release builds, seed 42.
10. **Zero per-tick heap churn.** Existing `determinism_no_alloc` pattern passes with the new tick-start phase added.
11. **Schema hash re-baselined.** `crates/engine/.schema_hash` regenerated; CI test passes.

If any of (1)–(11) fails, the plan is not done.

---

## Files overview

**New:**
- `crates/engine/src/ability/mod.rs` — module root, re-exports
- `crates/engine/src/ability/id.rs` — `AbilityId` newtype (`NonZeroU32`)
- `crates/engine/src/ability/program.rs` — `AbilityProgram`, `EffectOp`, `Delivery`, `Area`, `Gate`, `TargetSelector`
- `crates/engine/src/ability/registry.rs` — `AbilityRegistry` + builder (reuses `AggregatePool`)
- `crates/engine/src/ability/cast.rs` — `CastHandler` (one cascade handler that branches on `EffectOp`)
- `crates/engine/src/ability/expire.rs` — tick-start unified pass (decrement + expire + engagement update)
- `crates/engine/src/ability/gate.rs` — `evaluate_cast_gate()` for mask predicate

**Modified:**
- `crates/engine/src/state/mod.rs` — 5 new hot fields: `hot_stun_remaining_ticks`, `hot_slow_remaining_ticks`, `hot_slow_factor_q8`, `hot_cooldown_next_ready_tick`, `hot_engaged_with` + `cold_standing: SparseStandings`
- `crates/engine/src/state/agent.rs` — `AgentSpawn` gets no new params (all default)
- `crates/engine/src/creature.rs` — `CreatureType::is_hostile_to(other) -> bool`
- `crates/engine/src/mask.rs` — cast-valid head; move-allowed-when-engaged predicate; cooldown gate
- `crates/engine/src/policy/mod.rs` — `MicroTarget::Cast { ability, target }` variant surface (if not present), `Action` carries it
- `crates/engine/src/policy/utility.rs` — score `Cast` when valid
- `crates/engine/src/step.rs` — unified tick-start phase; integrate engagement update + status decrement + cooldown decrement; route `Cast` into cascade via `AgentCast` event
- `crates/engine/src/event/mod.rs` + `ring.rs` — 12+ new replayable event variants (combat + world + engagement + audit)
- `crates/engine/src/cascade/handler.rs` — new `EventKindId` ordinals
- `crates/engine/src/schema_hash.rs` — fingerprint additions
- `crates/engine/.schema_hash` — regenerated baseline
- `crates/engine/src/lib.rs` — register `ability` module
- `crates/engine/src/ids.rs` — re-export `AbilityId`

**Tests (new):**
- `crates/engine/tests/engagement_field.rs`
- `crates/engine/tests/combat_state_fields.rs`
- `crates/engine/tests/engagement_tick_start.rs`
- `crates/engine/tests/engagement_move_slow.rs`
- `crates/engine/tests/proptest_engagement.rs`
- `crates/engine/tests/ability_registry.rs`
- `crates/engine/tests/ability_program_shape.rs`
- `crates/engine/tests/action_cast_emits_agentcast.rs`
- `crates/engine/tests/cast_handler_damage.rs`
- `crates/engine/tests/cast_handler_heal.rs`
- `crates/engine/tests/cast_handler_shield_absorb.rs`
- `crates/engine/tests/cast_handler_stun.rs`
- `crates/engine/tests/cast_handler_slow.rs`
- `crates/engine/tests/cast_handler_gold.rs`
- `crates/engine/tests/cast_handler_standing.rs`
- `crates/engine/tests/cast_handler_recursive.rs`
- `crates/engine/tests/cast_recursion_depth.rs`
- `crates/engine/tests/stun_expiry.rs`
- `crates/engine/tests/slow_expiry.rs`
- `crates/engine/tests/mask_can_cast.rs`
- `crates/engine/tests/cooldown_blocks_recast.rs`
- `crates/engine/tests/ability_no_alloc.rs`
- `crates/engine/tests/acceptance_2v2_cast.rs`
- `crates/engine/tests/acceptance_world_tax.rs`
- `crates/engine/tests/acceptance_meteor_swarm.rs`
- `crates/engine/tests/acceptance_wall_formation.rs`

---

## Task 1 — `hot_engaged_with` + `cold_standing` + `is_hostile_to` stub

**Files:** `crates/engine/src/state/mod.rs`, `crates/engine/src/creature.rs`, `crates/engine/tests/engagement_field.rs`

Adds the engagement SoA field and the hostility predicate. **Bidirectional invariant doc'd on the field:** `engaged_with[a] == Some(b)` iff `engaged_with[b] == Some(a)` after `update_engagements` runs (Task 3 enforces; this task just adds storage).

**Storage:**
```rust
// state/mod.rs
hot_engaged_with: Vec<Option<AgentId>>,
cold_standing: SparseStandings,          // per-pair (i16_q8 -1000..1000)
```

`SparseStandings` is a `BTreeMap<(AgentId, AgentId), i16>` keyed on `(min(a,b), max(a,b))` for symmetry. Small default API: `get(a, b) -> i16`, `set(a, b, v)`, `adjust(a, b, delta) -> i16_clamped`.

**Accessors:**
```rust
pub fn agent_engaged_with(&self, id: AgentId) -> Option<AgentId> { ... }
pub fn set_agent_engaged_with(&mut self, id: AgentId, other: Option<AgentId>) { ... }
pub fn hot_engaged_with(&self) -> &[Option<AgentId>] { ... }
pub fn standing(&self, a: AgentId, b: AgentId) -> i16 { ... }
pub fn adjust_standing(&mut self, a: AgentId, b: AgentId, delta: i16) -> i16 { ... }
```

**CreatureType::is_hostile_to (stub):**
```rust
pub fn is_hostile_to(self, other: CreatureType) -> bool {
    use CreatureType::*;
    match (self, other) {
        (Wolf, Human) | (Human, Wolf) => true,
        (Wolf, Deer)  | (Deer, Wolf)  => true,
        (Dragon, _)   | (_, Dragon)   => true,   // dragons hostile to all
        _ => false,
    }
}
```
Document as "stub superseded by per-pair relationship standing when the Memory/Relationships plan lands."

**Test** (`engagement_field.rs`):
- Spawn agent → `agent_engaged_with(id) == None`.
- Set engaged → read back.
- Kill agent → engagement slot stays (zombie state), cleared by update_engagements in Task 3.
- Pairwise hostility matrix check (Human↔Wolf hostile; Human↔Human friendly; Dragon universally hostile).
- `standing(a, b) == 0` default; `adjust_standing(a, b, 50) == 50`; `adjust_standing(a, b, 2000)` clamps to 1000.

**Why this test isn't circular:** the hostility matrix is a pure data table — if the implementer swapped any pair the matrix assertion fails. The engagement field test pins `None`-as-default separately from `Some(id)`-after-set.

**Commit:**
```bash
git add crates/engine/src/state/ crates/engine/src/creature.rs crates/engine/tests/engagement_field.rs
git commit -m "feat(engine): hot_engaged_with + cold_standing + CreatureType::is_hostile_to stub"
```

---

## Task 2 — stun / slow / cooldown hot fields

**Files:** `crates/engine/src/state/mod.rs`, `crates/engine/tests/combat_state_fields.rs`

Adds the 4 remaining combat-state fields the state port didn't cover. All `u32` or `i16` hot SoA.

```rust
hot_stun_remaining_ticks:    Vec<u32>,   // 0 = not stunned
hot_slow_remaining_ticks:    Vec<u32>,   // 0 = not slowed
hot_slow_factor_q8:          Vec<i16>,   // q8 fixed-point slow multiplier (e.g. 51 = 0.2× speed)
hot_cooldown_next_ready_tick: Vec<u32>,  // the tick after which the agent's global cast cooldown ends
```

Accessors follow the state-port pattern (per-agent getter/setter + bulk slice). `AgentSpawn` carries none; all default to 0.

**Test:** spawn → all four read 0; set → read back; bulk slices length == cap.

**Why this test isn't circular:** same-shape as state-port's Task B tests, which proved the pattern works for 7 combat fields. This task adds 4 more under the same accessor pattern — a bug in any one breaks symmetric spawn/set/read.

**Commit:**
```bash
git commit -m "feat(engine): hot_stun/slow/cooldown SoA fields for combat timing"
```

---

## Task 3 — Unified tick-start phase (engagement + decrement + expiry)

**Files:** `crates/engine/src/ability/expire.rs` (new), `crates/engine/src/step.rs`, `crates/engine/tests/engagement_tick_start.rs`, `crates/engine/tests/stun_expiry.rs`, `crates/engine/tests/slow_expiry.rs`

**One function, three jobs:**

```rust
// ability/expire.rs
pub fn tick_start(state: &mut SimState, events: &mut EventRing) {
    // 1. Decrement stun / slow / cooldown, emit expiry events on transition to 0.
    for id in state.agents_alive() {
        let slot = (id.raw() - 1) as usize;

        let stun = state.hot_stun_remaining_ticks()[slot];
        if stun > 0 {
            let new_stun = stun - 1;
            state.set_agent_stun_remaining(id, new_stun);
            if new_stun == 0 {
                events.push(Event::StunExpired { agent_id: id, tick: state.tick });
            }
        }

        let slow = state.hot_slow_remaining_ticks()[slot];
        if slow > 0 {
            let new_slow = slow - 1;
            state.set_agent_slow_remaining(id, new_slow);
            if new_slow == 0 {
                events.push(Event::SlowExpired { agent_id: id, tick: state.tick });
                state.set_agent_slow_factor_q8(id, 0);  // clear factor too
            }
        }
        // cooldown_next_ready_tick is absolute — no decrement needed; it's
        // compared against state.tick by the cooldown mask predicate.
    }

    // 2. Update engagements (bidirectional): each alive agent's nearest hostile
    //    within ENGAGEMENT_RANGE. Tentative-pick-then-commit to preserve
    //    symmetry (see proptest in Task 5 for the 3-agent case this handles).
    let mut tentative: Vec<Option<AgentId>> = vec![None; state.agent_cap() as usize];
    for id in state.agents_alive() {
        let pos = match state.agent_pos(id) { Some(p) => p, None => continue };
        let ct  = match state.agent_creature_type(id) { Some(c) => c, None => continue };
        let mut best: Option<(AgentId, f32)> = None;
        for other in state.agents_alive() {
            if other == id { continue; }
            let op = match state.agent_pos(other) { Some(p) => p, None => continue };
            let oc = match state.agent_creature_type(other) { Some(c) => c, None => continue };
            if !ct.is_hostile_to(oc) { continue; }
            let d = pos.distance(op);
            if d > ENGAGEMENT_RANGE { continue; }
            if best.map_or(true, |(_, bd)| d < bd) { best = Some((other, d)); }
        }
        tentative[(id.raw() - 1) as usize] = best.map(|(a, _)| a);
    }
    // Commit only mutual pairings.
    for id in state.agents_alive() {
        let slot = (id.raw() - 1) as usize;
        let candidate = tentative[slot];
        let committed = match candidate {
            Some(other) => {
                let other_slot = (other.raw() - 1) as usize;
                if tentative[other_slot] == Some(id) { Some(other) } else { None }
            }
            None => None,
        };
        state.set_agent_engaged_with(id, committed);
    }
}
```

`step_full` now calls `ability::expire::tick_start(state, events)` BEFORE `scratch.mask.reset()`.

**Tests:**

`engagement_tick_start.rs`:
- Two hostile agents at 1.5m → both become engaged with each other after one tick.
- Same-species agents → stay `None`.
- Agents at 3m (outside ENGAGEMENT_RANGE=2.0) → stay `None`.
- Previously engaged agents move to 5m apart → next tick both are `None`.
- **3-agent symmetry case:** A, B hostile at 1m; C hostile at 0.5m from B. B's nearest hostile is C, not A. After commit: B engaged with C; C engaged with B; A engaged with None. (The tentative-commit logic catches this.)

`stun_expiry.rs`:
- Set stun=3; step 3 times; on the 3rd step, `StunExpired` event emitted and `hot_stun_remaining_ticks[slot] == 0`.
- Verify only ONE expiry event across many ticks (no double-fire).

`slow_expiry.rs`:
- Similar but with slow+factor; verify slow_factor_q8 is zeroed on expiry.

**Why these tests aren't circular:** the 3-agent symmetry case is a specific counterexample — a naive "A picks B; B is therefore engaged with A" commit would pass the 2-agent test but fail the 3-agent test because B actually closer to C. Forces the real tentative-commit logic.

**Commit:**
```bash
git commit -m "feat(engine): unified tick-start phase — engagement update + stun/slow decrement + expiry events"
```

---

## Task 4 — Engagement-aware MoveToward/Flee + opportunity attack cascade

**Files:** `crates/engine/src/step.rs`, `crates/engine/src/event/mod.rs`, `crates/engine/tests/engagement_move_slow.rs`

**Constants** (`step.rs`):
```rust
pub const ENGAGEMENT_RANGE: f32 = 2.0;       // matches ATTACK_RANGE
pub const ENGAGEMENT_SLOW_FACTOR: f32 = 0.3;
```

**New event:**
```rust
Event::OpportunityAttackTriggered { attacker: AgentId, target: AgentId, tick: u32 },
```
EventKindId ordinal: next slot. Byte-packing: tag, attacker u32, target u32, tick u32.

**Apply-actions change** (pseudo-diff on MoveToward + Flee branches):

```rust
ActionKind::Micro { kind: MicroKind::MoveToward, target: MicroTarget::Position(dest) } => {
    let sp = state.agent_pos(action.agent).unwrap_or(Vec3::ZERO);
    let dir = (dest - sp).normalize_or_zero();
    if dir.length_squared() == 0.0 { /* same as before */ }
    else {
        let base_speed = MOVE_SPEED_MPS;

        let mut speed = base_speed;
        if let Some(engager) = state.agent_engaged_with(action.agent) {
            // Engaged. Moving toward engager is fine (closing melee). Moving
            // anywhere else is slowed + triggers opportunity attack.
            let engager_pos = state.agent_pos(engager).unwrap_or(sp);
            let toward_engager = (engager_pos - sp).dot(dir) > 0.0;
            if !toward_engager {
                speed *= ENGAGEMENT_SLOW_FACTOR;
                events.push(Event::OpportunityAttackTriggered {
                    attacker: engager, target: action.agent, tick: state.tick,
                });
            }
        }

        let new_pos = sp + dir * speed;
        state.set_agent_pos(action.agent, new_pos);
        events.push(Event::AgentMoved { agent_id: action.agent, from: sp, to: new_pos, tick: state.tick });
    }
}
ActionKind::Micro { kind: MicroKind::Flee, target: MicroTarget::Agent(threat) } => {
    // Flee intentionally disengages. Always triggers OA if engaged with threat or someone else.
    let sp = state.agent_pos(action.agent).unwrap_or(Vec3::ZERO);
    let tp = state.agent_pos(threat).unwrap_or(sp);
    let dir = (sp - tp).normalize_or_zero();
    if dir.length_squared() > 0.0 {
        if let Some(engager) = state.agent_engaged_with(action.agent) {
            events.push(Event::OpportunityAttackTriggered {
                attacker: engager, target: action.agent, tick: state.tick,
            });
        }
        let new_pos = sp + dir * MOVE_SPEED_MPS;   // full-speed disengage
        state.set_agent_pos(action.agent, new_pos);
        events.push(Event::AgentFled { agent_id: action.agent, from: sp, to: new_pos, tick: state.tick });
    }
}
```

**Opportunity attack cascade handler** (engine-registered by default):

```rust
// Registered in CascadeRegistry::register_engine_builtins()
pub struct OpportunityAttackHandler;
impl CascadeHandler for OpportunityAttackHandler {
    fn trigger(&self) -> EventKindId { EventKindId::OpportunityAttackTriggered }
    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        if let Event::OpportunityAttackTriggered { attacker, target, tick } = *event {
            if !state.agent_alive(target) { return; }
            let new_hp = (state.agent_hp(target).unwrap_or(0.0) - ATTACK_DAMAGE).max(0.0);
            state.set_agent_hp(target, new_hp);
            events.push(Event::AgentAttacked { attacker, target, damage: ATTACK_DAMAGE, tick });
            if new_hp <= 0.0 {
                events.push(Event::AgentDied { agent_id: target, tick });
                state.kill_agent(target);
            }
        }
    }
}
```

**Tests** (`engagement_move_slow.rs`):
- Engaged A moves toward non-engager point: displacement == `ENGAGEMENT_SLOW_FACTOR * MOVE_SPEED_MPS = 0.3m` exactly. Assert `(delta - 0.3).abs() < 1e-5`.
- Engaged A moves toward engager: full 1.0m/tick. Assert `(delta - 1.0).abs() < 1e-5`.
- Opportunity attack fires: engaged A flees → `OpportunityAttackTriggered` event emitted + `AgentAttacked` cascade + A takes 10 HP damage.
- Kill via OA: engaged A with hp=5 flees → A takes 10 damage → hp=0 → `AgentDied` + kill.

**Why these tests aren't circular:** the `(delta - 0.3).abs() < 1e-5` assertion pins the slow factor to a specific value. Changing `ENGAGEMENT_SLOW_FACTOR` to 0.4 would fail. The OA death test confirms the cascade fires and applies ATTACK_DAMAGE via the shared constant.

**Commit:**
```bash
git commit -m "feat(engine): engagement-aware MoveToward/Flee + OpportunityAttackTriggered cascade"
```

---

## Task 5 — Engagement symmetry proptest

**Files:** `crates/engine/tests/proptest_engagement.rs`

Proptest: generate random populations of 3-10 agents with random CreatureType + random position in `[-10, 10]³`. After `tick_start`:

1. `engaged_with[a] == Some(b) ⇔ engaged_with[b] == Some(a)` (bidirectional symmetry)
2. No engagement between non-hostile pairs.
3. No engagement at distance > `ENGAGEMENT_RANGE`.
4. Two instances of SimState built from the same population tuple produce byte-identical `engaged_with` slices (cross-instance determinism).
5. After perturbation (move one agent by random `[-5, 5]` in each axis), re-run tick_start; previously-engaged pair that's now out-of-range is cleared.

**Why this isn't circular:** proptest generates adversarial inputs — cases the implementer wouldn't hand-pick. Property 1 shrinks to the 3-agent case (covered in Task 3 but here adversarially generated). Property 4 catches any hashmap iteration or float-comparison nondeterminism that our `replayable_sha256` test wouldn't see.

**Commit:**
```bash
git commit -m "test(engine): proptest — engagement symmetry + range respect + cross-instance determinism"
```

---

## Task 6 — `AbilityId` + module scaffold

**Files:** `crates/engine/src/ability/mod.rs`, `crates/engine/src/ability/id.rs`, `crates/engine/src/lib.rs`, `crates/engine/src/ids.rs`, `crates/engine/tests/ability_registry.rs` (scaffold)

```rust
// ability/id.rs
use std::num::NonZeroU32;
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AbilityId(NonZeroU32);
impl AbilityId {
    pub fn new(raw: u32) -> Option<Self> { NonZeroU32::new(raw).map(Self) }
    pub fn raw(self) -> u32 { self.0.get() }
    pub fn slot(self) -> usize { (self.0.get() - 1) as usize }
}
```

`ability/mod.rs`:
```rust
mod id;      pub use id::AbilityId;
pub mod program;
pub mod registry;
pub mod cast;
pub mod expire;      // Task 3 already created this
pub mod gate;

pub use program::{AbilityProgram, Area, Delivery, EffectOp, Gate, TargetSelector};
pub use registry::{AbilityRegistry, AbilityRegistryBuilder};
pub use cast::CastHandler;
```

Sub-modules start as stubs (`pub fn _placeholder() {}`) so the tree compiles; Tasks 7–9 fill them.

Register `pub mod ability;` in `crates/engine/src/lib.rs`; re-export `AbilityId` from `ids.rs`.

**Test:** `ability_id_round_trips_through_new()`, `ability_id_rejects_zero()`. Minimal.

**Commit:** `feat(engine): ability module scaffold + AbilityId newtype`

---

## Task 7 — `AbilityProgram` IR

**Files:** `crates/engine/src/ability/program.rs`, `crates/engine/tests/ability_program_shape.rs`

```rust
use smallvec::SmallVec;
use crate::ids::{AbilityId, AgentId};

pub const MAX_EFFECTS_PER_PROGRAM: usize = 4;

#[derive(Clone, Debug)]
pub struct AbilityProgram {
    pub delivery: Delivery,
    pub area:     Area,
    pub gate:     Gate,
    pub effects:  SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Delivery { Instant }       // Plan 2 adds Projectile / Zone

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Area { SingleTarget { range: f32 } }  // Plan 2 adds Cone / Circle / AoE

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Gate {
    pub cooldown_ticks: u32,
    pub hostile_only:   bool,
    pub line_of_sight:  bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TargetSelector { Target, Caster }

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EffectOp {
    // Combat (5)
    Damage { amount: f32 }                                = 0,
    Heal   { amount: f32 }                                = 1,
    Shield { amount: f32 }                                = 2,
    Stun   { duration_ticks: u32 }                        = 3,
    Slow   { duration_ticks: u32, factor_q8: i16 }        = 4,
    // World (2)
    TransferGold   { amount: i64 }                        = 5,  // signed, debt-allowed
    ModifyStanding { delta: i16 }                         = 6,
    // Meta (1)
    CastAbility { ability: AbilityId, selector: TargetSelector } = 7,
}
```

**Size budget:** `EffectOp` ≤ 16 bytes. The largest variant is `TransferGold { amount: i64 }` = 8 bytes payload + 1 discriminant + padding = 16. Pinned in tests:

```rust
#[test]
fn effect_op_size_budget() {
    assert!(std::mem::size_of::<EffectOp>() <= 16,
        "EffectOp grew: {}", std::mem::size_of::<EffectOp>());
}
```

Other tests: construct one-effect Damage program; two-effect Damage+Stun program; world-effect TransferGold + ModifyStanding program; recursion-chain of two CastAbility effects.

**Why test isn't circular:** size budget blocks silent growth. Construction tests pin the field shapes — renaming `amount` → `value` on any variant fails the test.

**Commit:** `feat(engine): AbilityProgram IR — 8 EffectOp variants, Delivery + Area + Gate, size ≤16B`

---

## Task 8 — `AbilityRegistry` + builder

**Files:** `crates/engine/src/ability/registry.rs`, `crates/engine/tests/ability_registry.rs` (extend)

```rust
use super::{AbilityProgram, AbilityId};

pub struct AbilityRegistry {
    programs: Vec<AbilityProgram>,
}

impl AbilityRegistry {
    pub fn new() -> Self { Self { programs: Vec::new() } }

    pub fn get(&self, id: AbilityId) -> Option<&AbilityProgram> {
        self.programs.get(id.slot())
    }

    pub fn len(&self) -> usize { self.programs.len() }
    pub fn is_empty(&self) -> bool { self.programs.is_empty() }
}

pub struct AbilityRegistryBuilder {
    programs: Vec<AbilityProgram>,
}

impl AbilityRegistryBuilder {
    pub fn new() -> Self { Self { programs: Vec::new() } }

    pub fn register(&mut self, program: AbilityProgram) -> AbilityId {
        self.programs.push(program);
        AbilityId::new(self.programs.len() as u32).expect("len > 0 after push")
    }

    pub fn build(self) -> AbilityRegistry { AbilityRegistry { programs: self.programs } }
}
```

Append-only. Registry IDs stable (start at 1, monotonic).

**Test:** register two programs, build, lookup by AbilityId round-trips; out-of-range AbilityId returns None.

**Commit:** `feat(engine): AbilityRegistry + builder — append-only, slot-stable AbilityIds`

---

## Task 9 — `CastHandler` dispatch + engagement-respecting cast-gate

**Files:** `crates/engine/src/ability/cast.rs`, `crates/engine/src/ability/gate.rs`, `crates/engine/src/mask.rs`, `crates/engine/src/step.rs`, `crates/engine/tests/action_cast_emits_agentcast.rs`, `crates/engine/tests/mask_can_cast.rs`

Two concerns: **action dispatch** (Cast → AgentCast event → CastHandler runs) and **mask gating** (cast invalid when stunned / cooldown / out-of-range / engaged-and-target-not-engager).

**gate.rs:**
```rust
pub fn evaluate_cast_gate(
    state: &SimState, registry: &AbilityRegistry,
    caster: AgentId, ability: AbilityId, target: AgentId,
) -> bool {
    // 1. Caster alive, not stunned.
    if !state.agent_alive(caster) { return false; }
    if state.agent_stun_remaining(caster).unwrap_or(0) > 0 { return false; }
    // 2. Cooldown ready.
    if state.tick < state.agent_cooldown_next_ready(caster).unwrap_or(0) { return false; }
    // 3. Ability exists.
    let prog = match registry.get(ability) { Some(p) => p, None => return false };
    // 4. Target alive + in range.
    if !state.agent_alive(target) { return false; }
    let dist = match (state.agent_pos(caster), state.agent_pos(target)) {
        (Some(a), Some(b)) => a.distance(b), _ => return false,
    };
    let range = match prog.area { Area::SingleTarget { range } => range };
    if dist > range { return false; }
    // 5. Hostile-only check (gate.hostile_only → check creature-type hostility).
    if prog.gate.hostile_only {
        let ct = state.agent_creature_type(caster).unwrap();
        let tc = state.agent_creature_type(target).unwrap();
        if !ct.is_hostile_to(tc) { return false; }
    }
    // 6. Engagement respect: if caster is engaged with someone ELSE, forbid
    //    casting (caster is locked in melee with their engager).
    if let Some(engager) = state.agent_engaged_with(caster) {
        if engager != target { return false; }
    }
    true
}
```

**cast.rs:** `CastHandler` is triggered by `Event::AgentCast { caster, ability, target }`. It looks up the program, iterates effects, emits per-effect events.

```rust
pub struct CastHandler {
    registry: Arc<AbilityRegistry>,
}

impl CastHandler {
    pub fn new(registry: Arc<AbilityRegistry>) -> Self { Self { registry } }
}

impl CascadeHandler for CastHandler {
    fn trigger(&self) -> EventKindId { EventKindId::AgentCast }
    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        if let Event::AgentCast { caster, ability, target, tick } = *event {
            let prog = match self.registry.get(ability) { Some(p) => p, None => return };
            for op in prog.effects.iter() {
                emit_effect_event(*op, caster, target, state, events, tick);
            }
            // Start cooldown.
            state.set_agent_cooldown_next_ready(caster, tick + prog.gate.cooldown_ticks);
        }
    }
}

fn emit_effect_event(op: EffectOp, caster: AgentId, target: AgentId,
                     state: &mut SimState, events: &mut EventRing, tick: u32) {
    match op {
        EffectOp::Damage { amount } => events.push(Event::EffectDamageApplied { caster, target, amount, tick }),
        EffectOp::Heal   { amount } => events.push(Event::EffectHealApplied   { caster, target, amount, tick }),
        EffectOp::Shield { amount } => events.push(Event::EffectShieldApplied { caster, target, amount, tick }),
        EffectOp::Stun { duration_ticks } => events.push(Event::EffectStunApplied { caster, target, duration_ticks, tick }),
        EffectOp::Slow { duration_ticks, factor_q8 } => events.push(Event::EffectSlowApplied { caster, target, duration_ticks, factor_q8, tick }),
        EffectOp::TransferGold { amount }   => events.push(Event::EffectGoldTransfer  { from: caster, to: target, amount, tick }),
        EffectOp::ModifyStanding { delta }  => events.push(Event::EffectStandingDelta { a: caster, b: target, delta, tick }),
        EffectOp::CastAbility { ability, selector } => {
            let effective_target = match selector {
                TargetSelector::Target => target,
                TargetSelector::Caster => caster,
            };
            // Recursion: re-emit AgentCast; cascade loop bounds depth at MAX_CASCADE_ITERATIONS.
            events.push(Event::AgentCast { caster, ability, target: effective_target, tick });
        }
    }
}
```

**Mask predicate** (`mask.rs::mark_cast_valid`): one bit per agent — compute via `evaluate_cast_gate` for the agent's current best-target heuristic (nearest hostile within the widest ability range available in its registry). MVP: per-agent single ability slot; the mask checks that one ability.

**step.rs:** `apply_actions` branch for `Cast`:
```rust
ActionKind::Micro { kind: MicroKind::Cast, target: MicroTarget::Ability { id, target } } => {
    events.push(Event::AgentCast { caster: action.agent, ability: id, target, tick: state.tick });
    // CastHandler picks it up through cascade dispatch.
}
```

(If `MicroTarget::Ability` variant doesn't exist, add it in this task.)

**Tests** (`action_cast_emits_agentcast.rs` and `mask_can_cast.rs`):
- Emit Cast action → AgentCast event present → CastHandler fires → Damage event emitted (with 1-effect Damage program).
- Stunned caster → mask bit false → UtilityBackend skips.
- Cooldown pending → mask bit false.
- Engaged-with-other-than-target → mask bit false.
- Target out of range → mask bit false.

**Why tests aren't circular:** each predicate branch has its own failing-case test; the four-way conjunction in `evaluate_cast_gate` needs all four to pass for a `true`. Adversarial inputs (stunned+target-in-range; alive+out-of-range; etc.) exercise each independently.

**Commit:** `feat(engine): CastHandler dispatch + evaluate_cast_gate (respects engagement)`

---

## Tasks 10–14 — Combat EffectOp handlers

Each is a separate cascade handler keyed on its effect event. All follow the same shape.

### Task 10 — `EffectOp::Damage`

Handler on `Event::EffectDamageApplied { caster, target, amount }`. Reduces target hp; if hp ≤ 0, emits AgentDied + kills. Respects `hot_shield_hp` — shield absorbs first, overflow hits hp.

Test: Damage 30 with shield 10 → shield goes to 0, hp goes down by 20. Damage 100 on 50-hp target → dies, event chain AgentAttacked → AgentDied present (reuses the existing Attack death cascade semantics).

**Commit:** `feat(engine): EffectOp::Damage handler — shield-first absorb + death cascade`

### Task 11 — `EffectOp::Heal`

Handler on `Event::EffectHealApplied`. Raises target hp clamped at max_hp. Emits `EffectHealed { delta_applied }` with the real applied delta (like Task 1's Eat pattern).

Test: heal 20 on 40-hp (max 100) → hp = 60, delta = 20. Heal 20 on 95-hp → hp = 100, delta = 5.

**Commit:** `feat(engine): EffectOp::Heal — cap at max_hp + emit real applied delta`

### Task 12 — `EffectOp::Shield` (uses `hot_shield_hp`)

Handler on `Event::EffectShieldApplied`. Adds to target's `hot_shield_hp`. No cap for MVP; stacking allowed. Document as "stackable; no decay" (Ability Plan 2 adds timed shields).

Test: Shield 30 + Shield 20 → hot_shield_hp = 50; then Damage 40 → shield = 10, hp unchanged. Damage 20 → shield = 0, hp -= 10 (the overflow).

**Commit:** `feat(engine): EffectOp::Shield — additive absorb layer on hot_shield_hp`

### Task 13 — `EffectOp::Stun`

Handler on `Event::EffectStunApplied`. Sets `hot_stun_remaining_ticks[target] = max(existing, duration)` (longer stun wins).

Test: Stun duration 10 → agent's stun_remaining = 10; agent can't cast (mask false for 10 ticks). After 10 `tick_start` calls, `StunExpired` event fires; mask flips true.

**Commit:** `feat(engine): EffectOp::Stun — sets hot_stun_remaining_ticks, blocks Cast via mask`

### Task 14 — `EffectOp::Slow`

Handler on `Event::EffectSlowApplied`. Sets `hot_slow_remaining_ticks` + `hot_slow_factor_q8` (longer + stronger overrides). Apply order: if new factor_q8 > current OR new duration > current, replace.

Test: Slow { duration: 5, factor: 51 } (q8 for 0.2×) → subsequent MoveToward moves at 0.2 * MOVE_SPEED_MPS = 0.2m/tick. After 5 ticks, `SlowExpired`; speed back to 1.0.

**Commit:** `feat(engine): EffectOp::Slow — factor_q8 + duration; slowed move + expiry`

---

## Task 15 — Cooldown mask gate

**Files:** `crates/engine/src/mask.rs`, `crates/engine/tests/cooldown_blocks_recast.rs`

Already implemented inline in Task 9's `evaluate_cast_gate`. This task adds the **regression test**: cast → cooldown_next_ready_tick set → subsequent casts blocked for `cooldown_ticks` ticks → mask re-opens after.

Test:
- Register ability with gate.cooldown_ticks = 10.
- Tick 0: cast → cooldown_next_ready_tick = 10.
- Ticks 1..9: mask bit false.
- Tick 10: mask bit true; cast fires again.

**Commit:** `test(engine): cooldown mask gate — blocks recast until cooldown_next_ready_tick`

---

## Task 16 — `EffectOp::TransferGold`

**Files:** `crates/engine/src/ability/cast.rs` (extend) or a small handler module, `crates/engine/tests/cast_handler_gold.rs`

Handler on `Event::EffectGoldTransfer { from, to, amount: i64 }`. Reads `cold_inventory[from_slot].gold` and `cold_inventory[to_slot].gold`, debits + credits. Signed `i64` — no overdraft check (debt allowed). Emits `GoldInsufficient` audit event if `from.gold < amount` AND some `strict_mode` flag — for MVP we just allow negative.

Test: Alice gold=100, Bob gold=0; TransferGold 30 → Alice=70, Bob=30. TransferGold -50 (Bob pays Alice) → Alice=120, Bob=-20.

**Commit:** `feat(engine): EffectOp::TransferGold — signed i64 via cold_inventory.gold`

---

## Task 17 — `EffectOp::ModifyStanding`

**Files:** handler + `crates/engine/tests/cast_handler_standing.rs`

Handler on `Event::EffectStandingDelta { a, b, delta }`. Calls `state.adjust_standing(a, b, delta)`, clamped to `[-1000, 1000]`.

Test: adjust(A, B, 50) → 50. adjust(A, B, 2000) → 1000 (clamp). adjust(B, A, -100) → -100 (symmetric key — same pair). adjust(A, B, -50) → 950 (after previous 1000).

**Commit:** `feat(engine): EffectOp::ModifyStanding — clamped [-1000,1000] per-pair adjustment`

---

## Task 18 — `EffectOp::CastAbility` (recursion, bounded)

**Files:** already handled by Task 9's dispatch; this adds the recursion-depth test.

Test (`cast_recursion_depth.rs`):
- Chain: ability A casts ability A on target (self-recursion).
- Agent casts A → cascade fires A again → again → ... bounded by `MAX_CASCADE_ITERATIONS = 8`.
- Count `Event::EffectDamageApplied` (or any downstream effect) emissions: exactly 8 in release, panics in debug when 9th would have fired (per existing cascade-bounded release-vs-debug behavior).

Test (`cast_handler_recursive.rs`):
- Chain: ability A → casts B → which casts C. A 3-step chain. All 3 damage events fire.

**Commit:** `test(engine): EffectOp::CastAbility recursion bounded by MAX_CASCADE_ITERATIONS`

---

## Task 19 — Acceptance: combat 2v2

**Files:** `crates/engine/tests/acceptance_2v2_cast.rs`

2 wolves vs 2 humans, each human has a Damage ability (50 dmg, 20-tick cooldown, range 6m) + a Shield ability (30 absorb, 30-tick cooldown, range 3m). Humans alternate Damage and Shield; wolves melee.

Seed 42, 100 ticks. Assert:
- `replayable_sha256` matches a pinned golden hash (first run captures; subsequent runs verify).
- At tick 100, at least 1 human alive AND at least 1 wolf dead (emergent: humans survive via shield rotation).
- `EffectShieldApplied` events ≥ 3 per human over the run.

**Commit:** `test(engine): acceptance — 2v2 combat with Damage + Shield abilities`

---

## Task 20 — Acceptance: tax-ability world

**Files:** `crates/engine/tests/acceptance_world_tax.rs`

1 tax-collector Human with `Tax` ability (TransferGold -50 from target to caster + ModifyStanding -20, range 3m, cooldown 100 ticks). 3 other humans with gold=100 each. Collector stands at origin; others wander.

Seed 42, 500 ticks. Assert:
- Collector gold > 100 after 500 ticks (has collected).
- At least one target has standing with collector < -50 (ModifyStanding fired).
- Total gold conservation: sum(all humans' gold) = initial_total (gold transfers are zero-sum, no creation or destruction).

**Commit:** `test(engine): acceptance — tax-ability world (TransferGold + ModifyStanding)`

---

## Task 21 — Acceptance: meteor-swarm recursive

**Files:** `crates/engine/tests/acceptance_meteor_swarm.rs`

Ability `MeteorSwarm` has 3 effects: Damage 20, CastAbility(Meteor, Target), CastAbility(Meteor, Target). Ability `Meteor` has 1 effect: Damage 10.

So one cast of MeteorSwarm → 1 direct hit (20) + 2 Meteor casts (10 each) = 40 total on target.

Seed 42, 1 caster + 1 target, 1 tick. Assert:
- `EffectDamageApplied` events = 3 (one Damage 20, two Damage 10).
- Target hp reduced by exactly 40.
- No `CastDepthExceeded` event (we're at depth 2, well under 8).

Secondary scenario: ability `InfiniteLoop` casts itself recursively. Assert exactly 8 `EffectDamageApplied` + 1 `CastDepthExceeded` after one tick.

**Commit:** `test(engine): acceptance — meteor-swarm recursion + depth-bound verification`

---

## Task 22 — Acceptance: tank-wall formation

**Files:** `crates/engine/tests/acceptance_wall_formation.rs`

3 tanks (humans, hp=200) at `x ∈ {-1, 0, 1}, y=0`. 1 wizard (human, hp=30) at `y=-2`. 1 wolf at `y=10`.

Seed 42, 50 ticks. Assert:
- Wolf engages the middle tank within ≤ 12 ticks.
- Wolf's `y` advances `< 4.0` between tick 12 and tick 25 (slowed by engagement + OA).
- Wizard survives past tick 30.
- At least 2 `OpportunityAttackTriggered` events fired.

Secondary baseline (`acceptance_no_wall_baseline.rs`): no tanks; wolf reaches wizard and kills within 15 ticks. Pin the differential ≥ 2×.

**Commit:** `test(engine): acceptance — tank-wall + baseline differential ≥ 2×`

---

## Task 23 — No-alloc steady-state test

**Files:** `crates/engine/tests/ability_no_alloc.rs`

Same shape as existing `determinism_no_alloc`: run the three acceptance scenarios under `dhat`, assert zero allocations after warmup. Ensures the new tick-start phase, cascade handlers, and gate evaluation don't churn the heap.

**Commit:** `test(engine): no-alloc steady-state across Combat Foundation scenarios`

---

## Task 24 — Schema-hash rebaseline + status.md + cleanup

**Files:** `crates/engine/src/schema_hash.rs`, `crates/engine/.schema_hash`, `docs/engine/status.md`, delete the two source plan files.

Schema hash additions:
```rust
h.update(b"hot_stun_remaining_ticks=u32,hot_slow_remaining_ticks=u32,hot_slow_factor_q8=i16,hot_cooldown_next_ready_tick=u32,hot_engaged_with=OptionAgentId,cold_standing=SparseBTreeMap");
h.update(b"EffectOp:Damage=0,Heal=1,Shield=2,Stun=3,Slow=4,TransferGold=5,ModifyStanding=6,CastAbility=7");
h.update(b"Delivery:Instant;Area:SingleTarget;TargetSelector:Target,Caster");
h.update(b"ENGAGEMENT_RANGE=2.0,ENGAGEMENT_SLOW_FACTOR=0.3");
h.update(b"MAX_EFFECTS_PER_PROGRAM=4");
```

Regenerate baseline. All 11 acceptance criteria pass.

**status.md updates:**
- Flip "Combat" rows from ⚠️ partial → ✅ with Combat Foundation commit refs.
- Flip "Engagement / ZoC" row from ⚠️ drafted → ✅ with commit ref.
- Plans-index: remove the "Plan 3.5" and "Ability Plan 1" rows; add one row: `| Combat Foundation | 2026-04-19-combat-foundation.md | ✅ executed (24 tasks) |`.
- Add Visual-check items: V10 "hostile wolf slows visibly in tank range", V11 "disengage triggers red-flash OA on disengaging agent", V12 "Cast ability emits EffectXXApplied visible via cascade trail".

**Cleanup:**
```bash
git rm docs/superpowers/plans/2026-04-19-engine-plan-3_5-engagement-zoc.md
git rm docs/superpowers/plans/2026-04-19-ability-plan-1-foundation.md
```

**Commit:**
```
chore(engine): Combat Foundation schema rebaseline + docs + source-plan cleanup

All 24 tasks committed. Schema hash regenerated (covers 5 new hot fields +
8 EffectOps + new constants). Status.md flipped. Original Plan 3.5 and
Ability Plan 1 source files removed — their content is subsumed by this
merged plan and consolidated acceptance suite.
```

---

## Self-review checklist

- [ ] **Spec coverage.** All 11 plan-level acceptance criteria have at least one test that demonstrably exercises them.
- [ ] **Engagement is bidirectional.** Proptest (Task 5) shrinks to the 3-agent counterexample.
- [ ] **Cast-gate respects engagement.** Task 9 mask predicate tests cover engaged-with-non-target rejection.
- [ ] **Recursion bounded.** Task 18 tests confirm exactly 8 fires in release, panics in debug (reusing existing `cascade_bounded` behavior).
- [ ] **World effects signed-clean.** Task 16 verifies negative gold (debt) round-trips.
- [ ] **No missed prerequisites.** Every field referenced above exists post-state-port or is added here.
- [ ] **Schema hash covers the delta.** Task 24's fingerprint additions include every new type, constant, and enum variant.

---

## Execution handoff

Same-session subagent-driven-development recommended. 24 tasks; ~2 weeks of deliberate execution if running one task per session. After completion:
- **Plan 3** (persistence + obs + probes) executes next — its snapshot writer covers the full SoA including the 5 new combat/engagement fields in one atomic schema-hash bump.
- **Plan 4** (debug & trace runtime) follows.
- **Plan 5** (`ComputeBackend` trait) + **Plan 6+** (GPU kernel porting) per original roadmap.
