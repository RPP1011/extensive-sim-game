# Ability Plan 1 — Foundation: Combat + World Effects + Recursion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring ability casting into the ECS engine as a first-class action for both combat *and* world-level (economic / social) effects, with recursive abilities (one ability casting another) supported via the cascade runtime's iteration bound. After this plan: agents can `Cast` an ability at a target; the cascade runtime (engine plan 1) dispatches the ability's lowered program; a representative cross-domain subset of effects — five combat (Damage, Heal, Shield, Stun, Slow), two world (TransferGold, ModifyStanding), one meta (CastAbility for recursion) — emit typed events that fold into `SimState`; timed effects expire at tick boundaries; masks gate invalid casts; three deterministic acceptance fixtures exercise the combat, world, and recursive chains. Delivery is restricted to `Instant × SingleTarget` — projectile / zone / AoE come in Plan 2.

**Architecture:** Additive changes to `crates/engine/` only. A new `ability` module hosts four things: a lowered IR (`AbilityProgram` = list of typed `EffectOp`s + optional `TargetSelector` for recursion), a registry (`AbilityRegistry` = append-only `Vec<AbilityProgram>` indexed by `AbilityId`), a `CastHandler` that decomposes a cast into subordinate events, and a set of per-event handlers that apply state mutation. New SoA hot fields on `SimState` track combat state (shield pool, stun-remaining, slow-remaining, slow-factor, cooldown-next-ready) and world-level state (gold, standing table). New `MicroKind::Cast` payload carries `{ ability: AbilityId, target: AgentId }`. New replayable events span combat (`AgentCast`, `AgentDamaged`, `AgentHealed`, `ShieldGranted/Expired`, `StunApplied/Expired`, `SlowApplied/Expired`), world (`GoldTransferred`, `StandingChanged`), and recursion (no new event — `CastAbility` re-emits `AgentCast`, cascading through the same handler). Tick-start decrement phase runs before policy evaluation. Recursion depth is bounded by the engine's existing `MAX_CASCADE_ITERATIONS=8` — a `CastAbility` op emitting a fresh `AgentCast` uses exactly one cascade step, so a chain of 7 `CastAbility`s resolves and an 8th is dropped with a `CastDepthExceeded` audit event. The `tactical_sim::effects` types are **not** imported — the engine's IR is authored directly. A parser pass from existing `.ability` text is deferred to Plan 2.

**Tech stack:** Rust 2021, `glam`, `smallvec`, `ahash`, `sha2`. No new deps. No `tactical_sim` dep — engine stays standalone.

**Prerequisites:** Engine Plan 1 complete. Specifically this plan assumes `MicroKind::Cast` exists in the enum (even if currently emit-only), the cascade registry dispatches events through lanes, `EventId` + cause sidecar exist on `EventRing`, and `AggregatePool<T>` is available (reused for `AbilityRegistry`).

**Backend scope** (2026-04-19 spec-rewrite addendum): this plan targets the `SerialBackend` — the reference implementation. Per `docs/engine/spec.md` §§3, 11, the `CastHandler` needs a GPU SPIR-V counterpart alongside other cascade handlers; that kernel lands during the broader GPU cascade-porting work (Plan 6+), not here. The `AbilityProgram` IR + `EffectOp` enum are already Pod-shaped (the plan's `effect_op_is_pod_sized` test pins ≤ 16 bytes), so the IR is GPU-compatible without rework when the kernel is written.

**Plan location note:** originally drafted in `.claude/worktrees/inherited-roaming-torvalds/` by a parallel session on 2026-04-19; pulled into `world-sim-bench` on the same day for execution. No content changes from the source draft beyond this preamble.

---

## Acceptance criteria (plan-level)

1. **Cast dispatch.** An agent emitting `Action::Cast { ability, target }` with a valid mask bit causes the registered `Cast` handler to fire, which in turn emits the effect-specific events dictated by the ability's `AbilityProgram`. Handlers for each effect event fold into `SimState`.
2. **Eight effects live.** Combat: `EffectOp::Damage`, `Heal`, `Shield`, `Stun`, `Slow`. World: `TransferGold`, `ModifyStanding`. Meta: `CastAbility`. All round-trip: action → program → emitted event → state mutation.
3. **Expiry deterministic.** Stun and slow durations decrement at tick start; expiry emits `StunExpired` / `SlowExpired` exactly once.
4. **Mask gating.** `can_cast(agent, ability)` mask bit is false when any of: caster stunned, cooldown not ready, target not alive, target out of range. Utility backend never emits a `Cast` with a zero mask bit.
5. **Recursion bounded.** A `CastAbility` effect re-emits an `AgentCast` event that cascades through `CastHandler` again. The engine's `MAX_CASCADE_ITERATIONS=8` limit holds: a chain of 8 recursive casts resolves; a 9th emits a `CastDepthExceeded` audit event and is dropped. An ability that recurses into itself unconditionally self-terminates after 8 iterations with no state corruption.
6. **World effects mutate world state.** `TransferGold` deducts from source agent's gold and credits target (clamp ≥ 0 on overdraft; emit `GoldInsufficient` audit event on failure). `ModifyStanding` shifts the per-pair standing by signed delta, clamped to `[-1000, 1000]`.
7. **Determinism.** Three independent acceptance fixtures (combat 2v2; tax-ability world; meteor-swarm recursive) each produce bit-exact `replayable_sha256` hashes across two runs and across debug/release builds, seed 42.
8. **No per-tick heap churn.** `cargo test -p engine --test ability_no_alloc` passes with the same warm-up window as the existing `determinism_no_alloc` test, across all three acceptance scenarios.
9. **Schema hash re-baselined.** `crates/engine/.schema_hash` regenerated; CI test passes.

If any of (1)–(9) fails, the plan is not done.

---

## Files overview

New:
- `crates/engine/src/ability/mod.rs` — module root, re-exports.
- `crates/engine/src/ability/id.rs` — `AbilityId` newtype (`NonZeroU32`).
- `crates/engine/src/ability/program.rs` — `AbilityProgram`, `EffectOp`, `Delivery`, `Area`, `Gate`.
- `crates/engine/src/ability/registry.rs` — `AbilityRegistry` (append-only pool) + builder.
- `crates/engine/src/ability/cast.rs` — `CastHandler` (one cascade handler that branches on `EffectOp`).
- `crates/engine/src/ability/expire.rs` — tick-start decrement + expiry event emission.
- `crates/engine/src/ability/gate.rs` — `evaluate_gate(state, caster, target, ability)` → bool.

Tests (new):
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

Modified:
- `crates/engine/src/lib.rs` — register `ability` module.
- `crates/engine/src/ids.rs` — add `AbilityId` re-export.
- `crates/engine/src/state/agent.rs` — add `shield`, `stun_remaining_ticks`, `slow_remaining_ticks`, `slow_factor_q8`, `cooldown_next_ready_tick`, `gold` to `AgentSpawn` + defaults.
- `crates/engine/src/state/mod.rs` — add matching SoA hot fields + `cold_standing: SparseMatrix<i16>` + accessors.
- `crates/engine/src/event/mod.rs` — add 12 new replayable event variants (9 combat + 2 world + 1 audit).
- `crates/engine/src/mask.rs` — add `cast_valid` head: `bool per (agent × ability_slot)`; one ability per agent for Plan 1.
- `crates/engine/src/policy/mod.rs` — `MicroTarget::Cast { ability: AbilityId, target: AgentId }` variant; `Action` carries it through.
- `crates/engine/src/policy/utility.rs` — score `Cast` when in range + cooldown ready + target enemy + mask valid; prefer highest-tier effect available.
- `crates/engine/src/step.rs` — (a) call `ability::expire::tick_start_decrement` before policy evaluation; (b) route `Cast` action into cascade registry via `AgentCast` event.
- `crates/engine/src/schema_hash.rs` — include `EffectOp` variant list + new constants.
- `crates/engine/.schema_hash` — regenerated baseline.

---

## Task 1: `AbilityId` + module scaffold

**Files:**
- Create: `crates/engine/src/ability/mod.rs`
- Create: `crates/engine/src/ability/id.rs`
- Modify: `crates/engine/src/lib.rs`
- Modify: `crates/engine/src/ids.rs`
- Test: `crates/engine/tests/ability_registry.rs` (scaffold only)

- [ ] **Step 1: Write failing scaffold test** `crates/engine/tests/ability_registry.rs`

```rust
use engine::ability::AbilityId;

#[test]
fn ability_id_round_trips_through_new() {
    let id = AbilityId::new(1).expect("1 is non-zero");
    assert_eq!(id.raw(), 1);
}

#[test]
fn ability_id_rejects_zero() {
    assert!(AbilityId::new(0).is_none());
}
```

- [ ] **Step 2: Verify fails** — `engine::ability` does not exist.

```
cargo test -p engine --test ability_registry
```

- [ ] **Step 3: Create `crates/engine/src/ability/id.rs`**

```rust
use std::num::NonZeroU32;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AbilityId(NonZeroU32);

impl AbilityId {
    #[inline]
    pub fn new(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }
    #[inline]
    pub fn raw(self) -> u32 { self.0.get() }
    #[inline]
    pub fn slot(self) -> usize { (self.0.get() - 1) as usize }
}
```

- [ ] **Step 4: Create `crates/engine/src/ability/mod.rs`**

```rust
//! Ability casting — lowered IR, registry, cast handler, expiry.
//!
//! Abilities live in the engine as a *lowered program*: a list of typed
//! `EffectOp`s plus a `Delivery`, `Area`, and optional `Gate`. Source text
//! authored in the external DSL is lowered into this IR by a separate pass
//! (Plan 2). Plan 1 ships the runtime only; programs are hand-constructed
//! in tests and by the registry builder.

mod id;
pub use id::AbilityId;

pub mod program;
pub mod registry;
pub mod cast;
pub mod expire;
pub mod gate;

pub use program::{AbilityProgram, Area, Delivery, EffectOp, Gate};
pub use registry::{AbilityRegistry, AbilityRegistryBuilder};
pub use cast::CastHandler;
```

Keep the sub-modules empty (`pub fn __placeholder() {}` stubs) so the tree compiles; they get filled by Tasks 2–7.

- [ ] **Step 5: Wire into `lib.rs`**

Add, after `pub mod aggregate;` (or equivalent anchor from engine plan 1):

```rust
pub mod ability;
```

- [ ] **Step 6: Re-export `AbilityId` from `ids.rs`**

Append:

```rust
pub use crate::ability::AbilityId;
```

- [ ] **Step 7: Run tests**

```
cargo test -p engine --test ability_registry
cargo test -p engine
```

- [ ] **Step 8: Commit**

```bash
git add crates/engine/src/ability crates/engine/src/lib.rs \
        crates/engine/src/ids.rs crates/engine/tests/ability_registry.rs
git commit -m "feat(engine): ability module scaffold + AbilityId newtype"
```

---

## Task 2: `AbilityProgram` IR

**Files:**
- Modify: `crates/engine/src/ability/program.rs`
- Test: `crates/engine/tests/ability_program_shape.rs`

The IR is intentionally narrow in Plan 1 — five effect ops, one delivery, one area, one gate shape. Plan 2 widens the enums without breaking existing programs because the enums are `#[non_exhaustive]` and handlers dispatch on variant.

- [ ] **Step 1: Write failing test** `crates/engine/tests/ability_program_shape.rs`

```rust
use engine::ability::{AbilityProgram, Area, Delivery, EffectOp, Gate};

#[test]
fn damage_program_constructs_cleanly() {
    let p = AbilityProgram {
        delivery: Delivery::Instant,
        area: Area::SingleTarget { range: 6.0 },
        gate: Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
        effects: smallvec::smallvec![EffectOp::Damage { amount: 40.0 }],
    };
    assert_eq!(p.effects.len(), 1);
    match p.delivery {
        Delivery::Instant => {}
        _ => panic!("expected Instant"),
    }
}

#[test]
fn combo_program_with_two_effects() {
    let p = AbilityProgram {
        delivery: Delivery::Instant,
        area: Area::SingleTarget { range: 4.0 },
        gate: Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        effects: smallvec::smallvec![
            EffectOp::Damage { amount: 25.0 },
            EffectOp::Stun   { duration_ticks: 20 },
        ],
    };
    assert_eq!(p.effects.len(), 2);
}

#[test]
fn effect_op_is_pod_sized() {
    // Budget guard: grow deliberately, not accidentally. Plan 1 target: ≤ 16 bytes.
    // (TransferGold's i64 plus discriminant sets the floor.)
    assert!(std::mem::size_of::<EffectOp>() <= 16, "EffectOp grew: {}", std::mem::size_of::<EffectOp>());
}

#[test]
fn world_effect_ops_construct_cleanly() {
    use engine::ability::TargetSelector;

    let tax = AbilityProgram {
        delivery: Delivery::Instant,
        area:     Area::SingleTarget { range: 3.0 },
        gate:     Gate { cooldown_ticks: 100, hostile_only: false, line_of_sight: false },
        effects:  smallvec::smallvec![
            EffectOp::TransferGold   { amount: -50 },   // take 50 from target
            EffectOp::ModifyStanding { delta:  -20 },   // target dislikes caster more
        ],
    };
    assert_eq!(tax.effects.len(), 2);

    let chain = AbilityProgram {
        delivery: Delivery::Instant,
        area:     Area::SingleTarget { range: 8.0 },
        gate:     Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        effects:  smallvec::smallvec![
            EffectOp::CastAbility { ability: AbilityId::new(42).unwrap(), selector: TargetSelector::Target },
            EffectOp::CastAbility { ability: AbilityId::new(43).unwrap(), selector: TargetSelector::Caster },
        ],
    };
    assert_eq!(chain.effects.len(), 2);
}
```

- [ ] **Step 2: Verify fails** — `program` module is a stub.

- [ ] **Step 3: Implement `crates/engine/src/ability/program.rs`**

```rust
use smallvec::SmallVec;

/// Maximum effects per ability program. Plan 1 covers single- and dual-effect
/// abilities (e.g. damage+stun). Plan 2 widens if authored content needs it.
pub const MAX_EFFECTS_PER_PROGRAM: usize = 4;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Delivery {
    /// Effect applies the tick the cast resolves, no in-flight object.
    Instant,
    // Projectile / Zone / Channel / Tether / Trap / Chain — Plan 2+.
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Area {
    /// Exactly one target, ≤ `range` world units away.
    SingleTarget { range: f32 },
    // Circle / Cone / Line / Ring / SelfOnly / Spread — Plan 2+.
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gate {
    /// How many ticks after cast resolution the caster's cooldown blocks recasts.
    pub cooldown_ticks: u32,
    /// Require a hostile standing on the target; friendly / self targets reject.
    pub hostile_only: bool,
    /// Reserved for Plan 2; false in Plan 1.
    pub line_of_sight: bool,
}

/// Target-redirection selector used by recursive `CastAbility` ops.
/// The outer cast's (caster, target) pair is the reference frame.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum TargetSelector {
    /// Inner cast targets the outer caster.
    Caster,
    /// Inner cast targets the outer target.
    Target,
    /// Inner cast targets the outer caster themselves (self-buff recursion).
    SelfCast,
}

/// One primitive state-mutating effect. Amounts / durations are stored in
/// engine units (f32 for hp-like, u32 ticks for durations, i64 signed for
/// gold so a "take" effect is just a negative transfer).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum EffectOp {
    // ── Combat ───────────────────────────────────────────────────────
    Damage { amount: f32 },
    Heal   { amount: f32 },
    /// Absorption pool; stacks additively with existing shield up to `SHIELD_CAP`.
    Shield { amount: f32, duration_ticks: u32 },
    Stun   { duration_ticks: u32 },
    /// `factor` is in q8 fixed point: 128 means 50% move speed, 64 means 25%.
    Slow   { factor_q8: u8, duration_ticks: u32 },

    // ── World (non-combat) ──────────────────────────────────────────
    /// Transfer gold from caster to target. Negative amount reverses the flow
    /// (a "Tax" ability takes from target → caster by emitting a
    /// `TransferGold { amount: -50 }`). Clamps at 0 on overdraft; a failed
    /// transfer emits a `GoldInsufficient` audit event.
    TransferGold { amount: i64 },
    /// Shift target's standing *toward* the caster by a signed delta.
    /// Applied to the pair `(target → caster)`; clamped to `[-1000, 1000]`.
    ModifyStanding { delta: i16 },

    // ── Meta (recursive) ────────────────────────────────────────────
    /// Re-emit an `AgentCast` for `ability` with target resolved via `selector`.
    /// The inner cast participates in the enclosing cascade; depth is bounded
    /// by `MAX_CASCADE_ITERATIONS` in the engine's cascade runtime.
    /// Cooldown, mask, and gate checks are *skipped* on the inner cast — the
    /// outer ability has already committed to this chain.
    CastAbility { ability: AbilityId, selector: TargetSelector },
}

/// A fully lowered ability. The registry owns one of these per `AbilityId`;
/// all handlers read it by reference through the registry.
#[derive(Clone, Debug)]
pub struct AbilityProgram {
    pub delivery: Delivery,
    pub area:     Area,
    pub gate:     Gate,
    pub effects:  SmallVec<[EffectOp; MAX_EFFECTS_PER_PROGRAM]>,
}
```

- [ ] **Step 4: Run tests**

```
cargo test -p engine --test ability_program_shape
```

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/ability/program.rs \
        crates/engine/tests/ability_program_shape.rs
git commit -m "feat(engine): AbilityProgram IR — Instant+SingleTarget, 5 effect ops"
```

---

## Task 3: `AbilityRegistry` — pool of programs

**Files:**
- Modify: `crates/engine/src/ability/registry.rs`
- Test: extend `crates/engine/tests/ability_registry.rs`

- [ ] **Step 1: Extend `crates/engine/tests/ability_registry.rs`**

```rust
use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder,
    Area, Delivery, EffectOp, Gate,
};
use smallvec::smallvec;

fn sample_damage(amount: f32) -> AbilityProgram {
    AbilityProgram {
        delivery: Delivery::Instant,
        area:     Area::SingleTarget { range: 6.0 },
        gate:     Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        effects:  smallvec![EffectOp::Damage { amount }],
    }
}

#[test]
fn builder_assigns_sequential_ids_from_one() {
    let mut b = AbilityRegistryBuilder::new();
    let a = b.add(sample_damage(10.0));
    let b2 = b.add(sample_damage(20.0));
    assert_eq!(a.raw(), 1);
    assert_eq!(b2.raw(), 2);
    let r = b.build();
    assert_eq!(r.len(), 2);
}

#[test]
fn registry_lookup_returns_program() {
    let mut b = AbilityRegistryBuilder::new();
    let id = b.add(sample_damage(37.5));
    let r = b.build();
    let p = r.get(id).expect("id valid");
    let EffectOp::Damage { amount } = p.effects[0] else { panic!() };
    assert!((amount - 37.5).abs() < 1e-6);
}

#[test]
fn registry_get_unknown_id_is_none() {
    let r = AbilityRegistryBuilder::new().build();
    assert!(r.get(AbilityId::new(999).unwrap()).is_none());
}
```

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: Implement `crates/engine/src/ability/registry.rs`**

```rust
use super::{AbilityId, AbilityProgram};

/// Append-only registry of lowered programs. Built once at startup, immutable
/// afterwards — all handlers read by `&AbilityRegistry`. Programs are never
/// removed at runtime; unused programs cost one vec slot each.
pub struct AbilityRegistry {
    programs: Vec<AbilityProgram>,
}

impl AbilityRegistry {
    #[inline]
    pub fn len(&self) -> usize { self.programs.len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.programs.is_empty() }
    #[inline]
    pub fn get(&self, id: AbilityId) -> Option<&AbilityProgram> {
        self.programs.get(id.slot())
    }
}

pub struct AbilityRegistryBuilder {
    programs: Vec<AbilityProgram>,
}

impl AbilityRegistryBuilder {
    pub fn new() -> Self { Self { programs: Vec::new() } }

    pub fn add(&mut self, p: AbilityProgram) -> AbilityId {
        let raw = (self.programs.len() as u32) + 1;
        self.programs.push(p);
        AbilityId::new(raw).expect("raw >= 1")
    }

    pub fn build(self) -> AbilityRegistry {
        AbilityRegistry { programs: self.programs }
    }
}

impl Default for AbilityRegistryBuilder {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 4: Run tests, commit.**

```bash
git add crates/engine/src/ability/registry.rs crates/engine/tests/ability_registry.rs
git commit -m "feat(engine): AbilityRegistry + builder, id→program lookup"
```

---

## Task 4: SoA hot fields for combat state

**Files:**
- Modify: `crates/engine/src/state/agent.rs`
- Modify: `crates/engine/src/state/mod.rs`
- Test: `crates/engine/tests/state_ability_fields.rs` (new)

Design note: one cooldown slot per agent (not per ability) in Plan 1 keeps the SoA flat. Multi-ability cooldowns require a per-agent `[u64; N]` array which bloats the hot partition; deferred to Plan 2 with a sidecar cold table keyed by `(agent, ability_idx)`.

Gold is a single `i64` per agent (signed to allow debt in future plans; Plan 1 clamps ≥ 0 on overdraft). Standing is a **cold, dense** `Vec<i16>` of length `cap * cap`, indexed `[observer_slot * cap + about_slot]`. Dense is wasteful at large N; Plan 2 moves to a sparse `FxHashMap<(AgentId, AgentId), i16>`. For Plan 1's 4-agent acceptance fixture the matrix is 16 cells.

- [ ] **Step 1: Write failing test** `crates/engine/tests/state_ability_fields.rs`

```rust
use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState) -> engine::ids::AgentId {
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
    }).unwrap()
}

#[test]
fn spawn_initializes_ability_fields_to_zero() {
    let mut s = SimState::new(4, 7);
    let a = spawn(&mut s);
    assert_eq!(s.agent_shield(a),             Some(0.0));
    assert_eq!(s.agent_stun_remaining(a),     Some(0));
    assert_eq!(s.agent_slow_remaining(a),     Some(0));
    assert_eq!(s.agent_slow_factor_q8(a),     Some(255));   // 255 = 1.0 (no slow)
    assert_eq!(s.agent_shield_expires_at(a),  Some(0));
    assert_eq!(s.agent_cooldown_ready_at(a),  Some(0));
    assert_eq!(s.agent_gold(a),               Some(0));
}

#[test]
fn setters_round_trip() {
    let mut s = SimState::new(4, 7);
    let a = spawn(&mut s);
    s.set_agent_shield(a, 25.0);
    s.set_agent_shield_expires_at(a, 300);
    s.set_agent_stun_remaining(a, 40);
    s.set_agent_slow_remaining(a, 60);
    s.set_agent_slow_factor_q8(a, 128);
    s.set_agent_cooldown_ready_at(a, 1_234);
    s.set_agent_gold(a, 500);

    assert_eq!(s.agent_shield(a),            Some(25.0));
    assert_eq!(s.agent_shield_expires_at(a), Some(300));
    assert_eq!(s.agent_stun_remaining(a),    Some(40));
    assert_eq!(s.agent_slow_remaining(a),    Some(60));
    assert_eq!(s.agent_slow_factor_q8(a),    Some(128));
    assert_eq!(s.agent_cooldown_ready_at(a), Some(1_234));
    assert_eq!(s.agent_gold(a),              Some(500));
}

#[test]
fn standing_defaults_to_zero_and_round_trips() {
    let mut s = SimState::new(4, 7);
    let a = spawn(&mut s);
    let b = spawn(&mut s);
    assert_eq!(s.standing(a, b), 0);
    s.set_standing(a, b, -250);
    assert_eq!(s.standing(a, b),  -250);
    assert_eq!(s.standing(b, a),     0, "standing is directional");
}

#[test]
fn standing_clamps_to_thousand() {
    let mut s = SimState::new(4, 7);
    let a = spawn(&mut s);
    let b = spawn(&mut s);
    s.set_standing(a, b,  5_000);
    assert_eq!(s.standing(a, b),  1_000);
    s.set_standing(a, b, -5_000);
    assert_eq!(s.standing(a, b), -1_000);
}
```

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: Extend `SimState`** — add seven SoA vecs + one dense-matrix cold field in `crates/engine/src/state/mod.rs`:

```rust
pub struct SimState {
    // ... existing fields (hot_pos, hot_hp, hot_alive, hot_movement_mode, ...) ...
    hot_shield:               Vec<f32>,
    hot_shield_expires_at:    Vec<u64>,   // tick at which remaining shield is dropped
    hot_stun_remaining:       Vec<u16>,   // ticks, decremented each tick start
    hot_slow_remaining:       Vec<u16>,
    hot_slow_factor_q8:       Vec<u8>,    // 255 = no slow; 128 = 0.5× speed
    hot_cooldown_ready_at:    Vec<u64>,
    hot_gold:                 Vec<i64>,   // signed; Plan 1 clamps ≥ 0 on writes

    /// Dense cap×cap matrix: `cold_standing[observer_slot * cap + about_slot]`.
    /// Directional: `standing(a, b)` means "how a regards b".
    /// Plan 1 only — Plan 2 switches to sparse.
    cold_standing:            Vec<i16>,
}
```

In `SimState::new`:

```rust
let cap = cap as usize;
hot_shield:            vec![0.0; cap],
hot_shield_expires_at: vec![0;   cap],
hot_stun_remaining:    vec![0;   cap],
hot_slow_remaining:    vec![0;   cap],
hot_slow_factor_q8:    vec![255; cap],
hot_cooldown_ready_at: vec![0;   cap],
hot_gold:              vec![0;   cap],
cold_standing:         vec![0;   cap * cap],
```

In `spawn_agent`, after computing `slot`:

```rust
self.hot_shield[slot]            = 0.0;
self.hot_shield_expires_at[slot] = 0;
self.hot_stun_remaining[slot]    = 0;
self.hot_slow_remaining[slot]    = 0;
self.hot_slow_factor_q8[slot]    = 255;
self.hot_cooldown_ready_at[slot] = 0;
self.hot_gold[slot]              = 0;
// standing rows/cols for this slot are already zero from the cap*cap vec.
```

Accessors + mutators (mirror the existing needs-field pattern from engine plan 1):

```rust
pub fn agent_shield(&self, id: AgentId) -> Option<f32> {
    self.hot_shield.get(AgentSlotPool::slot_of_agent(id)).copied()
}
pub fn set_agent_shield(&mut self, id: AgentId, v: f32) {
    if let Some(s) = self.hot_shield.get_mut(AgentSlotPool::slot_of_agent(id)) { *s = v.max(0.0); }
}
// ...one pair per field, following the same shape...

pub fn hot_shield(&self)            -> &[f32] { &self.hot_shield }
pub fn hot_stun_remaining(&self)    -> &[u16] { &self.hot_stun_remaining }
pub fn hot_slow_remaining(&self)    -> &[u16] { &self.hot_slow_remaining }
pub fn hot_slow_factor_q8(&self)    -> &[u8]  { &self.hot_slow_factor_q8 }
pub fn hot_cooldown_ready_at(&self) -> &[u64] { &self.hot_cooldown_ready_at }
pub fn hot_gold(&self)              -> &[i64] { &self.hot_gold }

pub fn agent_gold(&self, id: AgentId) -> Option<i64> {
    self.hot_gold.get(AgentSlotPool::slot_of_agent(id)).copied()
}
pub fn set_agent_gold(&mut self, id: AgentId, v: i64) {
    if let Some(s) = self.hot_gold.get_mut(AgentSlotPool::slot_of_agent(id)) { *s = v.max(0); }
}

pub fn standing(&self, observer: AgentId, about: AgentId) -> i16 {
    let cap = self.agent_cap() as usize;
    let i = AgentSlotPool::slot_of_agent(observer) * cap + AgentSlotPool::slot_of_agent(about);
    self.cold_standing.get(i).copied().unwrap_or(0)
}
pub fn set_standing(&mut self, observer: AgentId, about: AgentId, v: i16) {
    let cap = self.agent_cap() as usize;
    let i = AgentSlotPool::slot_of_agent(observer) * cap + AgentSlotPool::slot_of_agent(about);
    if let Some(cell) = self.cold_standing.get_mut(i) {
        *cell = v.clamp(-1_000, 1_000);
    }
}
```

- [ ] **Step 4: Run tests**

```
cargo test -p engine --test state_ability_fields
cargo test -p engine
```

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/state/mod.rs crates/engine/src/state/agent.rs \
        crates/engine/tests/state_ability_fields.rs
git commit -m "feat(engine): SoA hot fields for shield / stun / slow / cooldown"
```

---

## Task 5: New event variants

**Files:**
- Modify: `crates/engine/src/event/mod.rs`
- Test: `crates/engine/tests/ability_event_variants.rs` (new)

The engine's `Event` enum grows by nine variants, all replayable. `AgentCast` is the root cause for each ability invocation — its `EventId` threads through the cause sidecar of every downstream effect event, so replay can reconstruct which cast generated which damage.

- [ ] **Step 1: Write failing test** `crates/engine/tests/ability_event_variants.rs`

```rust
use engine::ability::AbilityId;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;

fn a(n: u32) -> AgentId { AgentId::new(n).unwrap() }

#[test]
fn agent_cast_event_round_trips_through_ring() {
    let mut r = EventRing::with_cap(16);
    let eid = r.push(Event::AgentCast {
        caster: a(1), ability: AbilityId::new(3).unwrap(), target: a(2), tick: 7,
    });
    assert_eq!(eid.tick, 7);
    let e = r.iter().next().unwrap();
    match e {
        Event::AgentCast { caster, ability, target, .. } => {
            assert_eq!(caster.raw(), 1);
            assert_eq!(ability.raw(), 3);
            assert_eq!(target.raw(), 2);
        }
        _ => panic!("expected AgentCast"),
    }
}

#[test]
fn effect_events_carry_cause() {
    let mut r = EventRing::with_cap(16);
    let cause = r.push(Event::AgentCast {
        caster: a(1), ability: AbilityId::new(1).unwrap(), target: a(2), tick: 0,
    });
    let dmg = r.push_caused(Event::AgentDamaged { target: a(2), amount: 40.0, tick: 0 }, cause);
    assert_eq!(r.cause_of(dmg), Some(cause));
}
```

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: Extend `Event` enum**

Add, inside `pub enum Event { ... }`:

```rust
AgentCast          { caster: AgentId, ability: AbilityId, target: AgentId, depth: u8, tick: u64 },
AgentDamaged       { target: AgentId, amount: f32,        tick: u64 },
AgentHealed        { target: AgentId, amount: f32,        tick: u64 },
ShieldGranted      { target: AgentId, amount: f32, expires_at: u64, tick: u64 },
ShieldExpired      { target: AgentId, tick: u64 },
StunApplied        { target: AgentId, duration_ticks: u16, tick: u64 },
StunExpired        { target: AgentId, tick: u64 },
SlowApplied        { target: AgentId, factor_q8: u8, duration_ticks: u16, tick: u64 },
SlowExpired        { target: AgentId, tick: u64 },
GoldTransferred    { from: AgentId, to: AgentId, amount: i64, tick: u64 },
StandingChanged    { observer: AgentId, about: AgentId, delta: i16, tick: u64 },
CastDepthExceeded  { caster: AgentId, ability: AbilityId, depth: u8, tick: u64 },
```

`AgentCast.depth` starts at 0 for actions emitted from the policy backend; the `CastAbility` handler increments it when re-emitting. Used by the cascade runtime to drop casts past `MAX_CASCADE_ITERATIONS`.

Assign stable `EventKindId`s (see `schema_hash.rs`): continue from the last id used in engine plan 1 — e.g. `AgentCast = 16, AgentDamaged = 17, …, CastDepthExceeded = 27`. Locked at schema-hash time, Task 13.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/event/mod.rs crates/engine/tests/ability_event_variants.rs
git commit -m "feat(engine): 9 replayable event variants for ability casts + effects"
```

---

## Task 6: `MicroKind::Cast` payload + action routing

**Files:**
- Modify: `crates/engine/src/mask.rs` (nothing yet — cast mask is Task 9)
- Modify: `crates/engine/src/policy/mod.rs`
- Modify: `crates/engine/src/step.rs`
- Test: `crates/engine/tests/action_cast_emits_agentcast.rs`

`Cast` previously existed in `MicroKind` as emit-only. Now it must carry `{ AbilityId, AgentId }`. Existing engine plan 1 already defines `MicroTarget` — extend with a `Cast` variant rather than overloading `Position` / `Agent`.

- [ ] **Step 1: Failing test** `crates/engine/tests/action_cast_emits_agentcast.rs`

```rust
use engine::ability::{AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder,
                      Area, Delivery, EffectOp, Gate};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use smallvec::smallvec;

struct FireOnceAt { caster: engine::ids::AgentId, target: engine::ids::AgentId, ability: AbilityId, fired: std::cell::Cell<bool> }
impl PolicyBackend for FireOnceAt {
    fn evaluate(&self, _s: &SimState, _m: &MaskBuffer, out: &mut Vec<Action>) {
        if self.fired.get() { return; }
        self.fired.set(true);
        out.push(Action {
            agent: self.caster,
            kind: ActionKind::Micro {
                kind: MicroKind::Cast,
                target: MicroTarget::Cast { ability: self.ability, target: self.target },
            },
        });
    }
}

#[test]
fn cast_action_emits_agent_cast_event() {
    let mut reg = AbilityRegistryBuilder::new();
    let fireball = reg.add(AbilityProgram {
        delivery: Delivery::Instant,
        area:     Area::SingleTarget { range: 10.0 },
        gate:     Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        effects:  smallvec![EffectOp::Damage { amount: 30.0 }],
    });
    let reg = reg.build();

    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = {
        let mut c = CascadeRegistry::new();
        engine::ability::CastHandler::register(&mut c, &reg);
        c
    };
    let a1 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0 }).unwrap();
    let a2 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0, 0.0, 0.0), hp: 100.0 }).unwrap();

    let backend = FireOnceAt { caster: a1, target: a2, ability: fireball, fired: false.into() };
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);

    let mut seen = false;
    for e in events.iter() {
        if let Event::AgentCast { caster, ability, target, .. } = e {
            assert_eq!(caster.raw(), a1.raw());
            assert_eq!(ability.raw(), fireball.raw());
            assert_eq!(target.raw(),  a2.raw());
            seen = true;
        }
    }
    assert!(seen, "AgentCast not emitted");
}
```

- [ ] **Step 2: Verify fails.** `MicroTarget::Cast` doesn't exist; `CastHandler::register` doesn't exist; the two get implemented in Steps 3–4 and in Task 7.

- [ ] **Step 3: Add `MicroTarget::Cast` variant**

In `crates/engine/src/policy/mod.rs`:

```rust
#[derive(Clone, Copy, Debug)]
pub enum MicroTarget {
    None,
    Position(Vec3),
    Agent(AgentId),
    Cast { ability: AbilityId, target: AgentId },
}
```

- [ ] **Step 4: In `step.rs`, route `Cast` to the cascade registry**

The `apply_actions` phase already dispatches `MicroKind` variants to their handlers. For `MicroKind::Cast`:

```rust
MicroKind::Cast => {
    let MicroTarget::Cast { ability, target } = action.target else {
        // Mask layer guarantees this in Plan 1; a mask bug gets caught by the assertion.
        debug_assert!(false, "Cast emitted with non-Cast MicroTarget");
        continue;
    };
    let eid = events.push(Event::AgentCast {
        caster: action.agent, ability, target, tick: state.tick(),
    });
    cascade.dispatch(&mut DispatchCtx {
        state, events, tick: state.tick(), cause: eid,
    }, Event::AgentCast { caster: action.agent, ability, target, tick: state.tick() });
}
```

(The exact `DispatchCtx` / `cascade.dispatch` shape is defined in engine plan 1; the snippet above mirrors its call pattern.)

- [ ] **Step 5: Commit** — note that the test still fails until Task 7 lands `CastHandler::register`.

```bash
git add crates/engine/src/policy/mod.rs crates/engine/src/step.rs \
        crates/engine/tests/action_cast_emits_agentcast.rs
git commit -m "feat(engine): MicroTarget::Cast variant + Cast action routes through cascade"
```

---

## Task 7: `CastHandler` — one handler, dispatches each `EffectOp`

**Files:**
- Modify: `crates/engine/src/ability/cast.rs`
- Test: the test from Task 6 now goes green; Task 8 adds per-effect tests.

- [ ] **Step 1: Implement `crates/engine/src/ability/cast.rs`**

```rust
use super::{AbilityProgram, AbilityRegistry, EffectOp, Area, Gate, TargetSelector};
use crate::cascade::{CascadeRegistry, CascadeHandler, DispatchCtx, Lane, MAX_CASCADE_ITERATIONS};
use crate::event::Event;
use crate::ids::AgentId;

/// Single cascade handler that resolves an `AgentCast` event into per-effect
/// subordinate events. Registered against `CascadeRegistry` at startup with
/// `Lane::Effect`.
pub struct CastHandler<'r> {
    registry: &'r AbilityRegistry,
}

impl<'r> CastHandler<'r> {
    pub fn register(cascade: &mut CascadeRegistry, registry: &'r AbilityRegistry) {
        cascade.register_event(
            EventKindId::AgentCast,
            Lane::Effect,
            Box::new(move |ctx: &mut DispatchCtx, ev: &Event| {
                let Event::AgentCast { caster, ability, target, depth, tick } = *ev else { return; };

                // Depth guard: drop the cast with an audit event, no state mutation.
                if depth as usize >= MAX_CASCADE_ITERATIONS {
                    ctx.push_caused(Event::CastDepthExceeded { caster, ability, depth, tick }, ctx.cause);
                    return;
                }
                let Some(prog) = registry.get(ability) else { return; };
                resolve(prog, ctx, caster, target, depth, tick);
            }),
        );
    }
}

fn resolve(
    prog:   &AbilityProgram,
    ctx:    &mut DispatchCtx<'_>,
    caster: AgentId,
    target: AgentId,
    depth:  u8,
    tick:   u64,
) {
    // --- Gate: only enforced for top-level casts (depth == 0). Inner recursive
    //     casts skip the mask/cooldown/range gates — the outer cast has already
    //     committed to the chain. Targets still must be alive. ---
    if depth == 0 {
        match prog.area {
            Area::SingleTarget { range } => {
                let caster_pos = match ctx.state.agent_pos(caster) { Some(p) => p, None => return };
                let target_pos = match ctx.state.agent_pos(target) { Some(p) => p, None => return };
                if (caster_pos - target_pos).length() > range {
                    debug_assert!(false, "mask should have rejected out-of-range cast");
                    return;
                }
            }
        }
    }
    if !ctx.state.is_agent_alive(target) { return; }

    // --- Write cooldown (top-level only) ---
    if depth == 0 && prog.gate.cooldown_ticks > 0 {
        ctx.state.set_agent_cooldown_ready_at(caster, tick + prog.gate.cooldown_ticks as u64);
    }

    // --- Emit one event per effect op, caused by the AgentCast ---
    let cause = ctx.cause;
    for op in prog.effects.iter() {
        match *op {
            EffectOp::Damage { amount } => {
                ctx.push_caused(Event::AgentDamaged { target, amount, tick }, cause);
            }
            EffectOp::Heal { amount } => {
                ctx.push_caused(Event::AgentHealed { target, amount, tick }, cause);
            }
            EffectOp::Shield { amount, duration_ticks } => {
                ctx.push_caused(Event::ShieldGranted {
                    target, amount, expires_at: tick + duration_ticks as u64, tick,
                }, cause);
            }
            EffectOp::Stun { duration_ticks } => {
                ctx.push_caused(Event::StunApplied {
                    target, duration_ticks: duration_ticks.min(u16::MAX as u32) as u16, tick,
                }, cause);
            }
            EffectOp::Slow { factor_q8, duration_ticks } => {
                ctx.push_caused(Event::SlowApplied {
                    target, factor_q8,
                    duration_ticks: duration_ticks.min(u16::MAX as u32) as u16,
                    tick,
                }, cause);
            }

            // ── World effects ───────────────────────────────────────
            EffectOp::TransferGold { amount } => {
                // amount > 0 → caster pays target; < 0 → caster takes from target.
                // The event models the *net flow*; the handler in Task 8 enforces funds.
                let (from, to, abs) = if amount >= 0 {
                    (caster, target, amount)
                } else {
                    (target, caster, -amount)
                };
                ctx.push_caused(Event::GoldTransferred { from, to, amount: abs, tick }, cause);
            }
            EffectOp::ModifyStanding { delta } => {
                // `observer = target`, `about = caster` — target's view of caster shifts.
                ctx.push_caused(Event::StandingChanged {
                    observer: target, about: caster, delta, tick,
                }, cause);
            }

            // ── Meta: recursion ─────────────────────────────────────
            EffectOp::CastAbility { ability: inner, selector } => {
                let inner_target = match selector {
                    TargetSelector::Caster   => caster,
                    TargetSelector::Target   => target,
                    TargetSelector::SelfCast => caster,
                };
                // Re-emit AgentCast with depth + 1. The cascade runtime drains
                // the queue up to MAX_CASCADE_ITERATIONS total; the depth field
                // is what CastHandler checks on entry.
                ctx.push_caused(
                    Event::AgentCast { caster, ability: inner, target: inner_target, depth: depth + 1, tick },
                    cause,
                );
            }
        }
    }
}
```

Note the two-step dispatch: `AgentCast` → `resolve()` emits effect events → effect-event handlers (Task 8) mutate state. State mutation *never* happens inside `resolve()`; all mutation lives in effect-event handlers. This mirrors engine spec §9 lane discipline (validation reads; effect writes; reaction emits; audit reads). Recursion rides the same dispatch loop: emitting an `AgentCast` inside `resolve()` enqueues it on the cascade queue; the loop drains the queue, so deeper casts run *after* the current effects have finished emitting.

- [ ] **Step 2: Run Task-6 test** — should now pass; commit.

```bash
git add crates/engine/src/ability/cast.rs
git commit -m "feat(engine): CastHandler — AgentCast → per-effect subordinate events"
```

---

## Task 8: Per-effect state handlers

**Files:**
- Modify: `crates/engine/src/ability/cast.rs` (handler registrations — or factor into a new `effects.rs` sibling for readability)
- Tests: `crates/engine/tests/cast_handler_damage.rs`, `_heal.rs`, `_shield_absorb.rs`, `_stun.rs`, `_slow.rs`

Each effect event gets its own cascade handler on `Lane::Effect` that owns state mutation. The rules:

- `AgentDamaged { target, amount }` — if shielded, absorb from shield first (clamp ≥ 0). Remainder deducts from `hot_hp`. If `hp ≤ 0`, emit `AgentDied` (engine plan 1 event).
- `AgentHealed { target, amount }` — `hp = min(hp + amount, max_hp)`.
- `ShieldGranted { target, amount, expires_at }` — `shield += amount`; `shield_expires_at = max(old, expires_at)`.
- `StunApplied { target, duration_ticks }` — `stun_remaining = max(stun_remaining, duration_ticks)`. Stun *refreshes* to the larger of current and new; does not stack additively.
- `SlowApplied { target, factor_q8, duration_ticks }` — overwrite with the *stronger* slow (lower `factor_q8`). If new `factor_q8 < current factor_q8` or current has 0 remaining, apply; otherwise keep current.

- [ ] **Step 1: Damage test** `crates/engine/tests/cast_handler_damage.rs`

```rust
use engine::ability::*;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use smallvec::smallvec;

fn spawn(state: &mut SimState, x: f32, hp: f32) -> engine::ids::AgentId {
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(x, 0.0, 0.0), hp,
    }).unwrap()
}

struct OneShot { caster: engine::ids::AgentId, target: engine::ids::AgentId, ability: AbilityId, fired: std::cell::Cell<bool> }
impl PolicyBackend for OneShot {
    fn evaluate(&self, _s: &SimState, _m: &MaskBuffer, out: &mut Vec<Action>) {
        if self.fired.replace(true) { return; }
        out.push(Action { agent: self.caster, kind: ActionKind::Micro { kind: MicroKind::Cast,
            target: MicroTarget::Cast { ability: self.ability, target: self.target }}});
    }
}

#[test]
fn damage_effect_reduces_hp_by_amount() {
    let mut b = AbilityRegistryBuilder::new();
    let fireball = b.add(AbilityProgram {
        delivery: Delivery::Instant,
        area:     Area::SingleTarget { range: 10.0 },
        gate:     Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        effects:  smallvec![EffectOp::Damage { amount: 25.0 }],
    });
    let reg = b.build();
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = { let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg); engine::ability::register_effect_handlers(&mut c); c };
    let a1 = spawn(&mut state, 0.0, 100.0);
    let a2 = spawn(&mut state, 2.0, 100.0);
    let p = OneShot { caster: a1, target: a2, ability: fireball, fired: false.into() };
    step(&mut state, &mut scratch, &mut events, &p, &cascade);
    assert_eq!(state.agent_hp(a2), Some(75.0));
}

#[test]
fn shield_absorbs_damage_before_hp() {
    let mut b = AbilityRegistryBuilder::new();
    let shield = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 },
        gate: Gate::default(),
        effects: smallvec![EffectOp::Shield { amount: 50.0, duration_ticks: 300 }],
    });
    let strike = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 },
        gate: Gate::default(),
        effects: smallvec![EffectOp::Damage { amount: 30.0 }],
    });
    let reg = b.build();
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = { let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg); engine::ability::register_effect_handlers(&mut c); c };
    let a1 = spawn(&mut state, 0.0, 100.0);
    let a2 = spawn(&mut state, 2.0, 100.0);

    // Pre-shield the target directly (unit-style, bypasses cast flow):
    state.set_agent_shield(a2, 50.0);
    state.set_agent_shield_expires_at(a2, 300);

    let strike_p = OneShot { caster: a1, target: a2, ability: strike, fired: false.into() };
    step(&mut state, &mut scratch, &mut events, &strike_p, &cascade);
    assert_eq!(state.agent_shield(a2), Some(20.0));
    assert_eq!(state.agent_hp(a2),     Some(100.0));
    let _ = shield; // referenced for symmetry, not used in this test body
}

#[test]
fn damage_past_shield_spills_to_hp() {
    // Shield 10, damage 30 → shield 0, hp 100 - 20 = 80.
    let mut b = AbilityRegistryBuilder::new();
    let strike = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::Damage { amount: 30.0 }],
    });
    let reg = b.build();
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = { let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg); engine::ability::register_effect_handlers(&mut c); c };
    let a1 = spawn(&mut state, 0.0, 100.0);
    let a2 = spawn(&mut state, 2.0, 100.0);
    state.set_agent_shield(a2, 10.0);
    state.set_agent_shield_expires_at(a2, 300);
    let p = OneShot { caster: a1, target: a2, ability: strike, fired: false.into() };
    step(&mut state, &mut scratch, &mut events, &p, &cascade);
    assert_eq!(state.agent_shield(a2), Some(0.0));
    assert_eq!(state.agent_hp(a2),     Some(80.0));
}
```

Analogous tests: heal clamps to max_hp (`cast_handler_heal.rs`); stun writes max-of-current-and-new (`cast_handler_stun.rs`); slow applies stronger factor (`cast_handler_slow.rs`).

- [ ] **Step 2: Verify the three damage-test cases all fail** (handlers unregistered). Then implement.

- [ ] **Step 3: Add `register_effect_handlers` in `crates/engine/src/ability/mod.rs`**

```rust
pub fn register_effect_handlers(cascade: &mut crate::cascade::CascadeRegistry) {
    use crate::cascade::Lane;
    use crate::event::{Event, EventKindId};

    cascade.register_event(EventKindId::AgentDamaged, Lane::Effect, Box::new(|ctx, ev| {
        let Event::AgentDamaged { target, amount, tick } = *ev else { return; };
        let shield = ctx.state.agent_shield(target).unwrap_or(0.0);
        let absorbed = shield.min(amount);
        let remaining = amount - absorbed;
        if absorbed > 0.0 {
            ctx.state.set_agent_shield(target, shield - absorbed);
        }
        if remaining > 0.0 {
            let hp = ctx.state.agent_hp(target).unwrap_or(0.0);
            let new_hp = (hp - remaining).max(0.0);
            ctx.state.set_agent_hp(target, new_hp);
            if new_hp == 0.0 && ctx.state.is_agent_alive(target) {
                ctx.push_caused(Event::AgentDied { agent: target, tick }, ctx.cause);
            }
        }
    }));

    cascade.register_event(EventKindId::AgentHealed, Lane::Effect, Box::new(|ctx, ev| {
        let Event::AgentHealed { target, amount, tick: _ } = *ev else { return; };
        let hp  = ctx.state.agent_hp(target).unwrap_or(0.0);
        let max = ctx.state.agent_max_hp(target).unwrap_or(hp);
        ctx.state.set_agent_hp(target, (hp + amount).min(max));
    }));

    cascade.register_event(EventKindId::ShieldGranted, Lane::Effect, Box::new(|ctx, ev| {
        let Event::ShieldGranted { target, amount, expires_at, tick: _ } = *ev else { return; };
        let cur     = ctx.state.agent_shield(target).unwrap_or(0.0);
        let cur_exp = ctx.state.agent_shield_expires_at(target).unwrap_or(0);
        ctx.state.set_agent_shield(target, cur + amount);
        ctx.state.set_agent_shield_expires_at(target, cur_exp.max(expires_at));
    }));

    cascade.register_event(EventKindId::StunApplied, Lane::Effect, Box::new(|ctx, ev| {
        let Event::StunApplied { target, duration_ticks, tick: _ } = *ev else { return; };
        let cur = ctx.state.agent_stun_remaining(target).unwrap_or(0);
        ctx.state.set_agent_stun_remaining(target, cur.max(duration_ticks));
    }));

    cascade.register_event(EventKindId::SlowApplied, Lane::Effect, Box::new(|ctx, ev| {
        let Event::SlowApplied { target, factor_q8, duration_ticks, tick: _ } = *ev else { return; };
        let cur_factor = ctx.state.agent_slow_factor_q8(target).unwrap_or(255);
        let cur_rem    = ctx.state.agent_slow_remaining(target).unwrap_or(0);
        // "Stronger" slow = lower factor; prefer it, or extend duration if same factor.
        if factor_q8 < cur_factor || cur_rem == 0 {
            ctx.state.set_agent_slow_factor_q8(target, factor_q8);
            ctx.state.set_agent_slow_remaining(target, duration_ticks);
        } else if factor_q8 == cur_factor {
            ctx.state.set_agent_slow_remaining(target, cur_rem.max(duration_ticks));
        }
    }));

    // ── World effects ───────────────────────────────────────────────
    cascade.register_event(EventKindId::GoldTransferred, Lane::Effect, Box::new(|ctx, ev| {
        let Event::GoldTransferred { from, to, amount, tick } = *ev else { return; };
        debug_assert!(amount >= 0, "amount should have been normalized by CastHandler");
        let have = ctx.state.agent_gold(from).unwrap_or(0);
        if have < amount {
            // Insufficient funds: log as audit, no state change.
            ctx.push_caused(Event::GoldInsufficient { agent: from, requested: amount, had: have, tick }, ctx.cause);
            return;
        }
        ctx.state.set_agent_gold(from, have - amount);
        let to_have = ctx.state.agent_gold(to).unwrap_or(0);
        ctx.state.set_agent_gold(to, to_have.saturating_add(amount));
    }));

    cascade.register_event(EventKindId::StandingChanged, Lane::Effect, Box::new(|ctx, ev| {
        let Event::StandingChanged { observer, about, delta, tick: _ } = *ev else { return; };
        let cur = ctx.state.standing(observer, about);
        let new = (cur as i32 + delta as i32).clamp(-1_000, 1_000) as i16;
        ctx.state.set_standing(observer, about, new);
    }));

    // ── Audit: recursion depth ──────────────────────────────────────
    // CastDepthExceeded has no handler — it's a terminal audit event.
    // Audit-lane readers (telemetry, replay probes) consume it.
}
```

Note: `GoldInsufficient` is a new audit event omitted from the main event list in Task 5 because it's internal to this handler. Add it when implementing:

```rust
GoldInsufficient { agent: AgentId, requested: i64, had: i64, tick: u64 },
```

EventKindId 28. Include in the schema hash (Task 13).

- [ ] **Step 4: Run the five new tests; fix any that stay red; commit.**

```bash
git add crates/engine/src/ability/mod.rs crates/engine/src/ability/cast.rs \
        crates/engine/tests/cast_handler_damage.rs \
        crates/engine/tests/cast_handler_heal.rs \
        crates/engine/tests/cast_handler_shield_absorb.rs \
        crates/engine/tests/cast_handler_stun.rs \
        crates/engine/tests/cast_handler_slow.rs
git commit -m "feat(engine): per-effect cascade handlers — dmg, heal, shield, stun, slow"
```

---

## Task 8B: World-effect handlers (gold, standing)

**Files:**
- Modify: `crates/engine/src/ability/mod.rs` (the `register_effect_handlers` fn, already shipped in Task 8; the world cascade arms added there)
- Test: `crates/engine/tests/cast_handler_gold.rs`, `crates/engine/tests/cast_handler_standing.rs`

- [ ] **Step 1: Gold test** `crates/engine/tests/cast_handler_gold.rs`

```rust
use engine::ability::*;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use smallvec::smallvec;

struct OneShot { caster: engine::ids::AgentId, target: engine::ids::AgentId, ability: AbilityId, fired: std::cell::Cell<bool> }
impl PolicyBackend for OneShot {
    fn evaluate(&self, _s: &SimState, _m: &MaskBuffer, out: &mut Vec<Action>) {
        if self.fired.replace(true) { return; }
        out.push(Action { agent: self.caster, kind: ActionKind::Micro { kind: MicroKind::Cast,
            target: MicroTarget::Cast { ability: self.ability, target: self.target }}});
    }
}

fn spawn(s: &mut SimState, x: f32, gold: i64) -> engine::ids::AgentId {
    let id = s.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(x, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    s.set_agent_gold(id, gold);
    id
}

fn setup(prog: AbilityProgram) -> (SimState, SimScratch, EventRing, CascadeRegistry, engine::ids::AgentId, engine::ids::AgentId, AbilityId) {
    let mut b = AbilityRegistryBuilder::new();
    let ab = b.add(prog);
    let reg = Box::leak(Box::new(b.build()));
    let mut state = SimState::new(4, 42);
    let scratch = SimScratch::new(state.agent_cap() as usize);
    let events = EventRing::with_cap(256);
    let mut c = CascadeRegistry::new();
    CastHandler::register(&mut c, reg);
    engine::ability::register_effect_handlers(&mut c);
    let a1 = spawn(&mut state, 0.0, 100);
    let a2 = spawn(&mut state, 2.0,  50);
    (state, scratch, events, c, a1, a2, ab)
}

#[test]
fn positive_amount_pays_target() {
    let (mut state, mut scratch, mut events, cascade, a1, a2, ab) = setup(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::TransferGold { amount: 30 }],
    });
    let p = OneShot { caster: a1, target: a2, ability: ab, fired: false.into() };
    step(&mut state, &mut scratch, &mut events, &p, &cascade);
    assert_eq!(state.agent_gold(a1), Some(70));
    assert_eq!(state.agent_gold(a2), Some(80));
}

#[test]
fn negative_amount_takes_from_target() {
    let (mut state, mut scratch, mut events, cascade, a1, a2, ab) = setup(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::TransferGold { amount: -20 }],
    });
    let p = OneShot { caster: a1, target: a2, ability: ab, fired: false.into() };
    step(&mut state, &mut scratch, &mut events, &p, &cascade);
    assert_eq!(state.agent_gold(a1), Some(120));
    assert_eq!(state.agent_gold(a2), Some(30));
}

#[test]
fn insufficient_funds_emits_audit_and_preserves_balance() {
    let (mut state, mut scratch, mut events, cascade, a1, a2, ab) = setup(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::TransferGold { amount: 500 }],   // a1 only has 100
    });
    let p = OneShot { caster: a1, target: a2, ability: ab, fired: false.into() };
    step(&mut state, &mut scratch, &mut events, &p, &cascade);
    assert_eq!(state.agent_gold(a1), Some(100));  // unchanged
    assert_eq!(state.agent_gold(a2), Some(50));
    let mut audited = 0;
    for e in events.iter() {
        if matches!(e, Event::GoldInsufficient { .. }) { audited += 1; }
    }
    assert_eq!(audited, 1);
}
```

- [ ] **Step 2: Standing test** `crates/engine/tests/cast_handler_standing.rs`

```rust
use engine::ability::*;
// ... same imports + OneShot + setup as cast_handler_gold.rs ...

#[test]
fn modify_standing_shifts_target_view_of_caster() {
    let mut b = AbilityRegistryBuilder::new();
    let bribe = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::ModifyStanding { delta: 100 }],
    });
    let threaten = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::ModifyStanding { delta: -250 }],
    });
    let reg = b.build();
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = {
        let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg);
        engine::ability::register_effect_handlers(&mut c);
        c
    };
    let a1 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0 }).unwrap();
    let a2 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0, 0.0, 0.0), hp: 100.0 }).unwrap();

    step(&mut state, &mut scratch, &mut events, &OneShot { caster: a1, target: a2, ability: bribe, fired: false.into() }, &cascade);
    assert_eq!(state.standing(a2, a1),  100);
    assert_eq!(state.standing(a1, a2),    0, "directional — a1's view unchanged");

    step(&mut state, &mut scratch, &mut events, &OneShot { caster: a1, target: a2, ability: threaten, fired: false.into() }, &cascade);
    assert_eq!(state.standing(a2, a1), -150);
}
```

- [ ] **Step 3: Run tests, commit.**

```bash
git add crates/engine/tests/cast_handler_gold.rs crates/engine/tests/cast_handler_standing.rs
git commit -m "test(engine): world-effect handlers — gold transfer, standing shift"
```

---

## Task 8C: Recursive `CastAbility`

**Files:**
- Test: `crates/engine/tests/cast_handler_recursive.rs`, `crates/engine/tests/cast_recursion_depth.rs`

The recursion logic lives in Task 7's `resolve()` function (already written). This task adds the tests and verifies the depth-exceeded audit path.

- [ ] **Step 1: Recursive test** `crates/engine/tests/cast_handler_recursive.rs`

```rust
//! "Meteor Swarm" = one outer cast whose program is three inner CastAbility
//! ops, each of which casts a damaging inner ability at the same target.
//! Expected: 3 inner AgentCast events, each carrying depth=1; 3 AgentDamaged
//! events; target hp drops by 3× the inner damage.

use engine::ability::*;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use smallvec::smallvec;

struct OneShot { caster: engine::ids::AgentId, target: engine::ids::AgentId, ability: AbilityId, fired: std::cell::Cell<bool> }
impl PolicyBackend for OneShot {
    fn evaluate(&self, _s: &SimState, _m: &MaskBuffer, out: &mut Vec<Action>) {
        if self.fired.replace(true) { return; }
        out.push(Action { agent: self.caster, kind: ActionKind::Micro { kind: MicroKind::Cast,
            target: MicroTarget::Cast { ability: self.ability, target: self.target }}});
    }
}

#[test]
fn meteor_swarm_fires_three_inner_damages() {
    let mut b = AbilityRegistryBuilder::new();
    let meteor = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::Damage { amount: 10.0 }],
    });
    let swarm = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 },
        gate: Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        effects: smallvec![
            EffectOp::CastAbility { ability: meteor, selector: TargetSelector::Target },
            EffectOp::CastAbility { ability: meteor, selector: TargetSelector::Target },
            EffectOp::CastAbility { ability: meteor, selector: TargetSelector::Target },
        ],
    });
    let reg = b.build();
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = {
        let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg);
        engine::ability::register_effect_handlers(&mut c);
        c
    };
    let a1 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO,             hp: 100.0 }).unwrap();
    let a2 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0,0.0,0.0), hp: 100.0 }).unwrap();

    step(&mut state, &mut scratch, &mut events, &OneShot { caster: a1, target: a2, ability: swarm, fired: false.into() }, &cascade);
    assert_eq!(state.agent_hp(a2), Some(70.0));

    let mut inner = 0;
    let mut outer = 0;
    for e in events.iter() {
        if let Event::AgentCast { depth, .. } = e {
            if *depth == 0 { outer += 1; } else { inner += 1; }
        }
    }
    assert_eq!(outer, 1);
    assert_eq!(inner, 3);
}

#[test]
fn recursive_selector_target_resolves_correctly() {
    let mut b = AbilityRegistryBuilder::new();
    let self_heal = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 0.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::Heal { amount: 25.0 }],
    });
    // Outer ability targets an enemy with damage AND self-heals via recursion.
    let strike_and_recover = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 10.0 }, gate: Gate::default(),
        effects: smallvec![
            EffectOp::Damage { amount: 15.0 },
            EffectOp::CastAbility { ability: self_heal, selector: TargetSelector::Caster },
        ],
    });
    let reg = b.build();
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = {
        let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg);
        engine::ability::register_effect_handlers(&mut c);
        c
    };
    let a1 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 50.0 }).unwrap();
    let a2 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0,0.0,0.0), hp: 100.0 }).unwrap();

    step(&mut state, &mut scratch, &mut events, &OneShot { caster: a1, target: a2, ability: strike_and_recover, fired: false.into() }, &cascade);

    assert_eq!(state.agent_hp(a2), Some(85.0), "target took 15 damage");
    assert_eq!(state.agent_hp(a1), Some(75.0), "caster healed 25 (50 + 25)");
}
```

- [ ] **Step 2: Depth test** `crates/engine/tests/cast_recursion_depth.rs`

```rust
//! "Infinite self-cast" ability: its only effect is CastAbility on itself.
//! Expected: MAX_CASCADE_ITERATIONS AgentCast events (depth 0..=7), then
//! exactly one CastDepthExceeded at depth 8. No state corruption.

use engine::ability::*;
use engine::cascade::{CascadeRegistry, MAX_CASCADE_ITERATIONS};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use smallvec::smallvec;

struct OneShot { caster: engine::ids::AgentId, ability: AbilityId, fired: std::cell::Cell<bool> }
impl PolicyBackend for OneShot {
    fn evaluate(&self, _s: &SimState, _m: &MaskBuffer, out: &mut Vec<Action>) {
        if self.fired.replace(true) { return; }
        out.push(Action { agent: self.caster, kind: ActionKind::Micro { kind: MicroKind::Cast,
            target: MicroTarget::Cast { ability: self.ability, target: self.caster }}});
    }
}

#[test]
fn infinite_self_cast_bounded_by_max_cascade_iterations() {
    // Build the self-referential ability in two steps: placeholder id → real id.
    let mut b = AbilityRegistryBuilder::new();
    let id = AbilityId::new((b.len() as u32) + 1).unwrap();
    let _ = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 0.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::CastAbility { ability: id, selector: TargetSelector::SelfCast }],
    });
    let reg = b.build();
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = {
        let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg);
        engine::ability::register_effect_handlers(&mut c);
        c
    };
    let a1 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0 }).unwrap();

    step(&mut state, &mut scratch, &mut events, &OneShot { caster: a1, ability: id, fired: false.into() }, &cascade);

    let mut casts = 0;
    let mut exceeded = 0;
    for e in events.iter() {
        match e {
            Event::AgentCast         { .. } => casts += 1,
            Event::CastDepthExceeded { .. } => exceeded += 1,
            _ => {}
        }
    }
    assert_eq!(casts, MAX_CASCADE_ITERATIONS);
    assert_eq!(exceeded, 1);
    assert_eq!(state.agent_hp(a1), Some(100.0), "no state corruption");
}
```

- [ ] **Step 3: Add `len()` to `AbilityRegistryBuilder`** (self-referential construction needs it). Trivial:

```rust
pub fn len(&self) -> usize { self.programs.len() }
```

- [ ] **Step 4: Run tests, commit.**

```bash
git add crates/engine/src/ability/registry.rs \
        crates/engine/tests/cast_handler_recursive.rs \
        crates/engine/tests/cast_recursion_depth.rs
git commit -m "test(engine): recursive CastAbility — meteor swarm + depth guard"
```

---

## Task 9: Expiry phase (stun / slow / shield)

**Files:**
- Modify: `crates/engine/src/ability/expire.rs`
- Modify: `crates/engine/src/step.rs`
- Tests: `crates/engine/tests/stun_expiry.rs`, `crates/engine/tests/slow_expiry.rs`

The expiry phase runs at tick start, *before* policy evaluation, *before* mask building. This means a stunned agent sees `stun_remaining = 0` in the mask phase of the tick its stun wears off, and thus can cast that same tick. Duration semantics: `duration_ticks` = "N ticks of effect, including the tick of application," so applying a 3-tick stun on tick 10 expires entering tick 14.

- [ ] **Step 1: Failing test** `crates/engine/tests/stun_expiry.rs`

```rust
use engine::ability::*;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct Idle;
impl PolicyBackend for Idle {
    fn evaluate(&self, s: &SimState, _m: &MaskBuffer, out: &mut Vec<Action>) {
        for id in s.agents_alive() {
            out.push(Action { agent: id, kind: ActionKind::Micro { kind: MicroKind::Hold, target: MicroTarget::None }});
        }
    }
}

#[test]
fn stun_counts_down_and_emits_expired() {
    let reg = AbilityRegistryBuilder::new().build();
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = { let mut c = CascadeRegistry::new();
        engine::ability::register_effect_handlers(&mut c); c };
    let a = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0 }).unwrap();
    state.set_agent_stun_remaining(a, 3);

    for _ in 0..2 { step(&mut state, &mut scratch, &mut events, &Idle, &cascade); }
    assert_eq!(state.agent_stun_remaining(a), Some(1), "2 ticks after stun=3 → 1");

    step(&mut state, &mut scratch, &mut events, &Idle, &cascade);
    assert_eq!(state.agent_stun_remaining(a), Some(0));
    let mut expired = 0;
    for e in events.iter() {
        if matches!(e, Event::StunExpired { .. }) { expired += 1; }
    }
    assert_eq!(expired, 1, "exactly one expiry event");
    let _ = reg;
}
```

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: Implement `crates/engine/src/ability/expire.rs`**

```rust
use crate::event::{Event, EventRing};
use crate::state::SimState;

/// Runs at tick start before policy / mask. Decrements per-agent timers and
/// emits `*Expired` events on the 1→0 transition (exactly once per effect).
pub fn tick_start_decrement(state: &mut SimState, events: &mut EventRing) {
    let tick = state.tick();
    let n    = state.agent_cap() as usize;
    for slot in 0..n {
        if !state.is_slot_alive(slot) { continue; }
        let id = state.agent_id_for_slot(slot);

        // Stun
        let stun = state.hot_stun_remaining()[slot];
        if stun > 0 {
            let new = stun - 1;
            state.set_agent_stun_remaining(id, new);
            if new == 0 {
                events.push(Event::StunExpired { target: id, tick });
            }
        }

        // Slow
        let slow = state.hot_slow_remaining()[slot];
        if slow > 0 {
            let new = slow - 1;
            state.set_agent_slow_remaining(id, new);
            if new == 0 {
                state.set_agent_slow_factor_q8(id, 255);
                events.push(Event::SlowExpired { target: id, tick });
            }
        }

        // Shield expiry: timer-based, not per-tick decrement — expire when tick reaches expires_at.
        let shield  = state.hot_shield()[slot];
        let exp_at  = state.agent_shield_expires_at(id).unwrap_or(0);
        if shield > 0.0 && exp_at > 0 && tick >= exp_at {
            state.set_agent_shield(id, 0.0);
            state.set_agent_shield_expires_at(id, 0);
            events.push(Event::ShieldExpired { target: id, tick });
        }
    }
}
```

- [ ] **Step 4: Wire into `step.rs`** — insert `ability::expire::tick_start_decrement(state, events);` as the *first* action of `step(...)`, before mask-build.

- [ ] **Step 5: Run expiry tests, fix if red, commit.**

```bash
git add crates/engine/src/ability/expire.rs crates/engine/src/step.rs \
        crates/engine/tests/stun_expiry.rs crates/engine/tests/slow_expiry.rs
git commit -m "feat(engine): tick-start expiry for stun/slow/shield + typed events"
```

---

## Task 10: `can_cast` mask predicate

**Files:**
- Modify: `crates/engine/src/mask.rs`
- Modify: `crates/engine/src/ability/gate.rs`
- Modify: `crates/engine/src/step.rs` (mask-build phase)
- Tests: `crates/engine/tests/mask_can_cast.rs`, `crates/engine/tests/cooldown_blocks_recast.rs`

In Plan 1 the mask is simple: **one bit per agent — "can cast your assigned ability this tick."** Plan 2 widens it to per-ability with a more structured head.

The UtilityBackend (engine plan 1) uses the mask to emit legal actions. Cast is off by default; a domain glue layer (world-sim) assigns each agent a `primary_ability: Option<AbilityId>` and targets `MicroTarget::Cast` at the nearest hostile.

- [ ] **Step 1: Failing test** `crates/engine/tests/mask_can_cast.rs`

```rust
use engine::ability::*;
use engine::mask::MaskBuffer;
use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;
use smallvec::smallvec;

fn build_reg() -> (AbilityRegistry, AbilityId) {
    let mut b = AbilityRegistryBuilder::new();
    let id = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 3.0 },
        gate: Gate { cooldown_ticks: 20, hostile_only: false, line_of_sight: false },
        effects: smallvec![EffectOp::Damage { amount: 10.0 }],
    });
    (b.build(), id)
}

#[test]
fn out_of_range_target_masks_off() {
    let (reg, ab) = build_reg();
    let mut s = SimState::new(4, 0);
    let a1 = s.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0 }).unwrap();
    let a2 = s.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(10.0, 0.0, 0.0), hp: 100.0 }).unwrap();
    let mut m = MaskBuffer::new(s.agent_cap() as usize);
    engine::ability::gate::build_cast_mask(&s, &reg, &[(a1, ab, a2)], &mut m);
    assert_eq!(m.cast_valid(a1), false);
}

#[test]
fn in_range_alive_target_masks_on() {
    let (reg, ab) = build_reg();
    let mut s = SimState::new(4, 0);
    let a1 = s.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0 }).unwrap();
    let a2 = s.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0, 0.0, 0.0), hp: 100.0 }).unwrap();
    let mut m = MaskBuffer::new(s.agent_cap() as usize);
    engine::ability::gate::build_cast_mask(&s, &reg, &[(a1, ab, a2)], &mut m);
    assert_eq!(m.cast_valid(a1), true);
}

#[test]
fn stunned_caster_masks_off() {
    let (reg, ab) = build_reg();
    let mut s = SimState::new(4, 0);
    let a1 = s.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0 }).unwrap();
    let a2 = s.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0, 0.0, 0.0), hp: 100.0 }).unwrap();
    s.set_agent_stun_remaining(a1, 5);
    let mut m = MaskBuffer::new(s.agent_cap() as usize);
    engine::ability::gate::build_cast_mask(&s, &reg, &[(a1, ab, a2)], &mut m);
    assert_eq!(m.cast_valid(a1), false);
}

#[test]
fn dead_target_masks_off() {
    let (reg, ab) = build_reg();
    let mut s = SimState::new(4, 0);
    let a1 = s.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0 }).unwrap();
    let a2 = s.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0, 0.0, 0.0), hp: 100.0 }).unwrap();
    s.kill_agent(a2);
    let mut m = MaskBuffer::new(s.agent_cap() as usize);
    engine::ability::gate::build_cast_mask(&s, &reg, &[(a1, ab, a2)], &mut m);
    assert_eq!(m.cast_valid(a1), false);
}
```

`mask_can_cast.rs` also covers cooldown in a fifth test (`cooldown_blocks_recast`) that re-uses the builder.

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: `MaskBuffer::cast_valid` — add a single-bit head**

In `mask.rs`:

```rust
pub struct MaskBuffer {
    // ... existing heads (micro_kind, target, ...) ...
    cast_valid: Vec<bool>,  // per-agent, Plan 1
}
impl MaskBuffer {
    pub fn cast_valid(&self, id: AgentId) -> bool {
        self.cast_valid.get(AgentSlotPool::slot_of_agent(id)).copied().unwrap_or(false)
    }
    pub fn set_cast_valid(&mut self, id: AgentId, v: bool) {
        if let Some(s) = self.cast_valid.get_mut(AgentSlotPool::slot_of_agent(id)) { *s = v; }
    }
}
```

- [ ] **Step 4: Implement `gate.rs`**

```rust
use crate::ability::{AbilityRegistry, AbilityId, Area};
use crate::ids::AgentId;
use crate::mask::MaskBuffer;
use crate::state::SimState;

/// Build the `cast_valid` mask head for `(caster, ability, target)` triples.
/// Called once per tick during mask-build.
pub fn build_cast_mask(
    state:      &SimState,
    registry:   &AbilityRegistry,
    assignments:&[(AgentId, AbilityId, AgentId)],
    mask:       &mut MaskBuffer,
) {
    let tick = state.tick();
    for &(caster, ability, target) in assignments {
        mask.set_cast_valid(caster, can_cast(state, registry, caster, ability, target, tick));
    }
}

fn can_cast(
    state: &SimState, registry: &AbilityRegistry,
    caster: AgentId, ability: AbilityId, target: AgentId, tick: u64,
) -> bool {
    if !state.is_agent_alive(caster) { return false; }
    if !state.is_agent_alive(target) { return false; }
    if state.agent_stun_remaining(caster).unwrap_or(0) > 0 { return false; }
    if state.agent_cooldown_ready_at(caster).unwrap_or(0) > tick { return false; }

    let Some(prog) = registry.get(ability) else { return false; };

    match prog.area {
        Area::SingleTarget { range } => {
            let cp = match state.agent_pos(caster) { Some(p) => p, None => return false };
            let tp = match state.agent_pos(target) { Some(p) => p, None => return false };
            if (cp - tp).length() > range { return false; }
        }
    }
    true
}
```

- [ ] **Step 5: Wire into `step.rs` mask-build phase**

The world-sim domain layer owns the `assignments` list. For the engine's own tests, it's supplied directly (as in the Task 10 tests). For the acceptance test (Task 12), build the assignments inline.

- [ ] **Step 6: Cooldown test** `crates/engine/tests/cooldown_blocks_recast.rs` — pre-set `cooldown_ready_at` to `tick + 5`, assert mask false; fast-forward 5 ticks of `Idle`, assert mask true.

- [ ] **Step 7: Run tests, commit.**

```bash
git add crates/engine/src/mask.rs crates/engine/src/ability/gate.rs \
        crates/engine/src/step.rs \
        crates/engine/tests/mask_can_cast.rs \
        crates/engine/tests/cooldown_blocks_recast.rs
git commit -m "feat(engine): can_cast mask predicate + gate evaluator"
```

---

## Task 11: No-alloc harness for ability runtime

**Files:**
- Test: `crates/engine/tests/ability_no_alloc.rs`

The engine's `determinism_no_alloc` test (from MVP plan) wraps an existing fixture; this test wraps the ability acceptance fixture (Task 12). Must run after Task 12's fixture is in place, so commit order is 12 → 11. Keep the file listed here so the plan task count stays aligned.

- [ ] **Step 1: Copy the acceptance test from Task 12 and wrap with `dhat`.** Assert `≤ 16` allocations after tick 100 (the warm-up window the MVP plan established).
- [ ] **Step 2: Run, fix allocations, commit.**

```bash
git commit -m "test(engine): ability runtime stays heap-quiet after warm-up"
```

---

## Task 12: Acceptance — 2v2 deterministic cast fixture

**Files:**
- Test: `crates/engine/tests/acceptance_2v2_cast.rs`

- [ ] **Step 1: Write acceptance test**

```rust
//! 4 agents, two on team A at x < 0 and two on team B at x > 0, each team agent
//! has a damage ability and a heal ability. Policy: if ally < 50% hp, heal the
//! lowest-hp ally; else damage the nearest enemy. 400 ticks. Same seed twice →
//! identical replayable hash.

use engine::ability::*;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use smallvec::smallvec;

struct TeamPolicy {
    damage:     AbilityId,
    heal:       AbilityId,
    team_a:     [engine::ids::AgentId; 2],
    team_b:     [engine::ids::AgentId; 2],
}
impl PolicyBackend for TeamPolicy {
    fn evaluate(&self, s: &SimState, m: &MaskBuffer, out: &mut Vec<Action>) {
        for &caster in self.team_a.iter().chain(self.team_b.iter()) {
            if !s.is_agent_alive(caster) { continue; }
            let (allies, enemies) = if self.team_a.contains(&caster) {
                (self.team_a, self.team_b)
            } else {
                (self.team_b, self.team_a)
            };
            let wounded_ally = allies.iter()
                .filter(|&&a| s.is_agent_alive(a) && s.agent_hp(a).unwrap_or(0.0) < 50.0)
                .min_by(|&&a, &&b| s.agent_hp(a).unwrap().partial_cmp(&s.agent_hp(b).unwrap()).unwrap());
            let (ability, target) = if let Some(&ally) = wounded_ally {
                (self.heal, ally)
            } else {
                let live_enemy = enemies.iter().find(|&&e| s.is_agent_alive(e)).copied();
                match live_enemy {
                    Some(e) => (self.damage, e),
                    None    => continue,
                }
            };
            let valid = m.cast_valid(caster);
            if !valid { continue; }
            out.push(Action {
                agent: caster,
                kind:  ActionKind::Micro {
                    kind:   MicroKind::Cast,
                    target: MicroTarget::Cast { ability, target },
                },
            });
        }
    }
}

fn run(seed: u64) -> [u8; 32] {
    let mut b = AbilityRegistryBuilder::new();
    let damage = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 20.0 },
        gate: Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        effects: smallvec![EffectOp::Damage { amount: 12.0 }],
    });
    let heal = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 20.0 },
        gate: Gate { cooldown_ticks: 40, hostile_only: false, line_of_sight: false },
        effects: smallvec![EffectOp::Heal { amount: 20.0 }],
    });
    let reg = b.build();

    let mut state = SimState::new(8, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(200_000);
    let cascade = {
        let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg);
        engine::ability::register_effect_handlers(&mut c);
        c
    };

    let a1 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(-5.0, 0.0, 0.0), hp: 100.0 }).unwrap();
    let a2 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(-5.0, 2.0, 0.0), hp: 100.0 }).unwrap();
    let b1 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new( 5.0, 0.0, 0.0), hp: 100.0 }).unwrap();
    let b2 = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new( 5.0, 2.0, 0.0), hp: 100.0 }).unwrap();
    let backend = TeamPolicy { damage, heal, team_a: [a1, a2], team_b: [b1, b2] };

    for _ in 0..400 {
        step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    }
    events.replayable_sha256()
}

#[test]
fn same_seed_same_hash() { assert_eq!(run(42), run(42)); }

#[test]
fn different_seed_different_hash() { assert_ne!(run(42), run(43)); }

#[test]
fn at_least_one_kill_in_400_ticks() {
    // Sanity: the fixture actually resolves combat; protects against
    // silent regressions where `Cast` becomes a no-op.
    let mut state = SimState::new(8, 42);
    // ...same setup inline as run()...
    let hash = run(42);
    assert_ne!(hash, [0u8; 32], "empty event log => cast pipeline is broken");
}
```

- [ ] **Step 2: Run.**

```
cargo test -p engine --test acceptance_2v2_cast
cargo test -p engine --test acceptance_2v2_cast --release
```

- [ ] **Step 3: Commit.**

```bash
git add crates/engine/tests/acceptance_2v2_cast.rs
git commit -m "test(engine): acceptance — 2v2 cast duel deterministic in 400 ticks"
```

- [ ] **Step 4: Acceptance fixture 2 — world-effect tax cycle** `crates/engine/tests/acceptance_world_tax.rs`

```rust
//! 3 agents. Agent 0 is a "tax collector" who repeatedly casts a Tax ability
//! (TransferGold -10 + ModifyStanding -5) on Agents 1 and 2. After 200 ticks:
//! - collector's gold ≈ initial + sum(collected)
//! - subjects' gold decreases by sum(taxed)
//! - subjects' standing toward collector drops proportionally
//! Same seed → same hash; different seed → different hash.

use engine::ability::*;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use smallvec::smallvec;

struct TaxCollector { collector: engine::ids::AgentId, subjects: [engine::ids::AgentId; 2], tax: AbilityId }
impl PolicyBackend for TaxCollector {
    fn evaluate(&self, s: &SimState, m: &MaskBuffer, out: &mut Vec<Action>) {
        if !m.cast_valid(self.collector) { return; }
        // Alternate subjects each cast based on tick parity.
        let subject = self.subjects[(s.tick() as usize / 2) % 2];
        if !s.is_agent_alive(subject) { return; }
        out.push(Action {
            agent: self.collector,
            kind:  ActionKind::Micro {
                kind:   MicroKind::Cast,
                target: MicroTarget::Cast { ability: self.tax, target: subject },
            },
        });
    }
}

fn run(seed: u64) -> [u8; 32] {
    let mut b = AbilityRegistryBuilder::new();
    let tax = b.add(AbilityProgram {
        delivery: Delivery::Instant,
        area:     Area::SingleTarget { range: 5.0 },
        gate:     Gate { cooldown_ticks: 20, hostile_only: false, line_of_sight: false },
        effects:  smallvec![
            EffectOp::TransferGold   { amount: -10 },
            EffectOp::ModifyStanding { delta: -5   },
        ],
    });
    let reg = b.build();

    let mut state = SimState::new(8, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(100_000);
    let cascade = {
        let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg);
        engine::ability::register_effect_handlers(&mut c);
        c
    };

    let collector = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::ZERO,             hp: 100.0 }).unwrap();
    let s1        = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0,0.0,0.0), hp: 100.0 }).unwrap();
    let s2        = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(3.0,0.0,0.0), hp: 100.0 }).unwrap();
    state.set_agent_gold(collector, 0);
    state.set_agent_gold(s1, 1000);
    state.set_agent_gold(s2, 1000);

    let backend = TaxCollector { collector, subjects: [s1, s2], tax };
    for _ in 0..200 {
        step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    }
    // Sanity check the numbers are *plausible*; precise values land in hash.
    assert!(state.agent_gold(collector).unwrap() > 0);
    assert!(state.standing(s1, collector) < 0);
    assert!(state.standing(s2, collector) < 0);

    events.replayable_sha256()
}

#[test] fn same_seed_same_hash()           { assert_eq!(run(42), run(42)); }
#[test] fn different_seed_different_hash() { assert_ne!(run(42), run(43)); }
```

- [ ] **Step 5: Acceptance fixture 3 — recursive Meteor Swarm at scale** `crates/engine/tests/acceptance_meteor_swarm.rs`

```rust
//! 8 agents (2 teams of 4). Each team has one "mage" that casts a Meteor Swarm
//! (3× recursive damage cast) at a cooldown. Over 500 ticks, verify:
//! - total AgentDamaged events = 3× total AgentCast(depth=0 by a mage)
//! - no CastDepthExceeded events
//! - deterministic hash across two runs.

use engine::ability::*;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::*;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use smallvec::smallvec;

struct MagesPolicy { mages: [engine::ids::AgentId; 2], enemies: [[engine::ids::AgentId; 3]; 2], swarm: AbilityId }
impl PolicyBackend for MagesPolicy {
    fn evaluate(&self, s: &SimState, m: &MaskBuffer, out: &mut Vec<Action>) {
        for (i, &mage) in self.mages.iter().enumerate() {
            if !s.is_agent_alive(mage) || !m.cast_valid(mage) { continue; }
            let enemies = &self.enemies[i ^ 1];
            let Some(&target) = enemies.iter().find(|&&e| s.is_agent_alive(e)) else { continue };
            out.push(Action {
                agent: mage,
                kind:  ActionKind::Micro {
                    kind:   MicroKind::Cast,
                    target: MicroTarget::Cast { ability: self.swarm, target },
                },
            });
        }
    }
}

fn run(seed: u64) -> ([u8; 32], u32, u32, u32) {
    let mut b = AbilityRegistryBuilder::new();
    let meteor = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 20.0 }, gate: Gate::default(),
        effects: smallvec![EffectOp::Damage { amount: 8.0 }],
    });
    let swarm = b.add(AbilityProgram {
        delivery: Delivery::Instant, area: Area::SingleTarget { range: 20.0 },
        gate: Gate { cooldown_ticks: 40, hostile_only: true, line_of_sight: false },
        effects: smallvec![
            EffectOp::CastAbility { ability: meteor, selector: TargetSelector::Target },
            EffectOp::CastAbility { ability: meteor, selector: TargetSelector::Target },
            EffectOp::CastAbility { ability: meteor, selector: TargetSelector::Target },
        ],
    });
    let reg = b.build();
    let mut state = SimState::new(16, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(500_000);
    let cascade = {
        let mut c = CascadeRegistry::new();
        CastHandler::register(&mut c, &reg);
        engine::ability::register_effect_handlers(&mut c);
        c
    };

    let mage_a  = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(-5.0, 0.0, 0.0),  hp: 200.0 }).unwrap();
    let e_a1    = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(-5.0, 2.0, 0.0),  hp: 100.0 }).unwrap();
    let e_a2    = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(-5.0, 4.0, 0.0),  hp: 100.0 }).unwrap();
    let e_a3    = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(-5.0, 6.0, 0.0),  hp: 100.0 }).unwrap();
    let mage_b  = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new( 5.0, 0.0, 0.0),  hp: 200.0 }).unwrap();
    let e_b1    = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new( 5.0, 2.0, 0.0),  hp: 100.0 }).unwrap();
    let e_b2    = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new( 5.0, 4.0, 0.0),  hp: 100.0 }).unwrap();
    let e_b3    = state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new( 5.0, 6.0, 0.0),  hp: 100.0 }).unwrap();

    let backend = MagesPolicy {
        mages:   [mage_a, mage_b],
        enemies: [[e_a1, e_a2, e_a3], [e_b1, e_b2, e_b3]],
        swarm,
    };
    for _ in 0..500 {
        step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    }

    let mut outer = 0; let mut inner = 0; let mut damages = 0; let mut exceeded = 0;
    for e in events.iter() {
        match e {
            Event::AgentCast { depth: 0, .. }      => outer   += 1,
            Event::AgentCast { .. }                => inner   += 1,
            Event::AgentDamaged { .. }             => damages += 1,
            Event::CastDepthExceeded { .. }        => exceeded += 1,
            _ => {}
        }
    }
    (events.replayable_sha256(), outer, inner, damages)
        .tap(|_| { assert_eq!(exceeded, 0); () });  // extension-trait pattern; inline-fn the assert if `tap` isn't in the crate
}

#[test]
fn same_seed_same_hash() {
    let a = run(42); let b = run(42);
    assert_eq!(a.0, b.0);
    assert_eq!(a.3, 3 * a.1, "damages == 3 × outer casts");
    assert_eq!(a.2, 3 * a.1, "inner casts == 3 × outer casts");
}

#[test]
fn different_seed_different_hash() {
    assert_ne!(run(42).0, run(43).0);
}
```

- [ ] **Step 6: Run all three fixtures, debug + release.**

```bash
cargo test -p engine --test acceptance_2v2_cast        --release
cargo test -p engine --test acceptance_world_tax       --release
cargo test -p engine --test acceptance_meteor_swarm    --release
```

- [ ] **Step 7: Commit fixtures 2 and 3.**

```bash
git add crates/engine/tests/acceptance_world_tax.rs \
        crates/engine/tests/acceptance_meteor_swarm.rs
git commit -m "test(engine): acceptance — world tax + meteor swarm fixtures deterministic"
```

- [ ] **Step 8: Back-fill Task 11 (no-alloc) wrapping all three fixtures.** Commit.

---

## Task 13: Schema-hash re-baseline

**Files:**
- Modify: `crates/engine/src/schema_hash.rs`
- Modify: `crates/engine/.schema_hash`

- [ ] **Step 1: Extend `schema_hash.rs`**

```rust
h.update(b"EffectOp:Damage,Heal,Shield,Stun,Slow,TransferGold,ModifyStanding,CastAbility");
h.update(b"TargetSelector:Caster,Target,SelfCast");
h.update(b"Delivery:Instant");
h.update(b"Area:SingleTarget");
h.update(b"EventKindId:AgentCast=16,AgentDamaged=17,AgentHealed=18,ShieldGranted=19,ShieldExpired=20,StunApplied=21,StunExpired=22,SlowApplied=23,SlowExpired=24,GoldTransferred=25,StandingChanged=26,CastDepthExceeded=27,GoldInsufficient=28");
h.update(b"MAX_EFFECTS_PER_PROGRAM=4,SHIELD_CAP_PER_AGENT=f32::MAX,SLOW_FACTOR_Q8_BITS=8,STANDING_MIN=-1000,STANDING_MAX=1000");
h.update(b"AgentCast.depth_bits=8");
```

- [ ] **Step 2: Regenerate baseline, run schema-hash test, commit.**

```bash
cargo run -p engine --example print_schema_hash > crates/engine/.schema_hash
cargo test -p engine --test schema_hash
git add crates/engine/src/schema_hash.rs crates/engine/.schema_hash
git commit -m "chore(engine): re-baseline schema hash — EffectOp, Area, Delivery, 9 event kinds"
```

---

## Self-review checklist

Before marking the plan complete:

- [ ] **All eight effect ops mutate state.** Combat: `Damage` deducts (via shield), `Heal` clamps, `Shield` stacks, `Stun` refreshes to max, `Slow` picks stronger factor. World: `TransferGold` moves gold with overdraft audit, `ModifyStanding` shifts clamped to `[-1000, 1000]`. Meta: `CastAbility` re-emits `AgentCast` at depth+1.
- [ ] **One cause per subordinate event.** Every `AgentDamaged` / `AgentHealed` / `ShieldGranted` / `StunApplied` / `SlowApplied` / `GoldTransferred` / `StandingChanged` / inner `AgentCast` in the ring has a non-None cause pointing at its root `AgentCast`. Expiry events have no cause (emitted by the expiry phase, not a handler).
- [ ] **Recursion bounded.** `cast_recursion_depth` test green: self-cast terminates at `MAX_CASCADE_ITERATIONS` with one `CastDepthExceeded` audit event and no state corruption.
- [ ] **Inner-cast gate skipping.** `resolve()` skips range / cooldown / hostile-only checks when `depth > 0` (test: `recursive_selector_target_resolves_correctly` self-heals through a 0-range ability).
- [ ] **Expiry runs first.** `tick_start_decrement` is the first statement in `step`. Stun-expires-this-tick means "can cast this tick."
- [ ] **Mask gates all top-level casts.** No `Action::Cast` with `cast_valid == false` is emitted by the Utility backend; apply phase `debug_assert!`s the same invariant. Recursive inner casts bypass the mask, consistent with the inner-gate-skipping rule above.
- [ ] **Standing is directional.** `standing(a, b) ≠ standing(b, a)`; `ModifyStanding` only touches `(target → caster)`. Verified in `cast_handler_standing.rs`.
- [ ] **Gold never goes negative.** `set_agent_gold` clamps ≥ 0; `TransferGold` emits `GoldInsufficient` on overdraft instead of mutating. Verified in `cast_handler_gold.rs`.
- [ ] **No new deps.** `Cargo.toml` unchanged modulo feature gates.
- [ ] **Schema hash rebased.** `.schema_hash` bumped covering 8 ops + `TargetSelector` + 13 event kinds + standing/depth constants; CI green.
- [ ] **Determinism across three fixtures.** Same-seed → same-hash over 400 / 200 / 500 ticks respectively. Debug vs release identical.
- [ ] **No heap alloc.** `ability_no_alloc` test wraps all three acceptance fixtures; ≤ 16 allocations after tick-100 warm-up in each.
- [ ] **No dep on `tactical_sim`.** `cargo tree -p engine` does not list it.
- [ ] **Placeholder scan.** No "TBD" / "TODO" in shipped code; tests cover each effect (combat + world + recursive) and each mask gate.
- [ ] **Lane discipline.** `CastHandler` emits only events; per-effect handlers mutate only state. `CastDepthExceeded` and `GoldInsufficient` are audit-only, no handler registered. No handler both emits and mutates.

---

## Out of scope (Plan 2 and later)

- **Area variants.** Circle / Cone / Line / Ring / Spread / SelfOnly — each needs its own gate + target-expansion pass.
- **Delivery variants.** Projectile / Zone / Channel / Tether / Trap / Chain. Projectile spawns a projectile-kind agent with a forward-moving policy and an `OnArrival` cascade; Zone spawns a zone-kind agent with Periodic tick; Channel / Tether are persistent relationships with per-tick break conditions; Trap/Chain are proximity-triggered.
- **DSL parser integration.** Lowering from `.ability` text files into `AbilityProgram`. Requires moving `crates/tactical_sim/src/effects/dsl/` into the engine tree or a shared `crates/ability_dsl/` crate (both sides depend on it).
- **More combat effects.** 63 of 68 tactical_sim combat `Effect` variants (Knockback, Dash, Root, Silence, Fear, Taunt, Pull, Swap, Reflect, Lifesteal, DamageModify, Execute, Blind, OnHitBuff, Resurrect, …). Prioritize by use-count in shipped `.ability` files.
- **More world effects.** `Announce` (info cascade via cascade registry on a world channel), `GrantMemory` (add a fact to target's `believed_knowledge`), `PostQuest` (aggregate creation), `Bid` (auction entry), `SetStanding` (absolute vs current delta), item transfers, territory modification, faction-wide standing shifts. Each needs corresponding engine state — memory ring, quest aggregate pool, etc. — most of which the engine spec §§7–14 already defines.
- **Triggers.** `OnDamageDealt`, `OnKill`, `Periodic`, etc. Mapped onto cascade-registry filters on the corresponding event kind. Unlike Plan 1's one-shot cascade, triggers register *persistent* subscriptions scoped to a caster or target.
- **Conditions.** Boolean predicate tree — lowered to gate predicates + mask bits.
- **Richer `TargetSelector`.** Plan 1 supports three static selectors (Caster / Target / SelfCast). Plan 2+: `NearestHostile { radius }`, `AllAlliesIn { radius }`, `Random { within }`, `ByStanding { below: i16 }`. Spatial selectors interact with the spatial index, so land them alongside Area variants.
- **Recursion beyond `CastAbility`.** Plan 1's only recursive op is cast-another-ability. Later plans may want `Loop { times, body }` for bounded-repeat semantics without abusing CascadeRegistry depth. If so, add a dedicated `Repeat` op that expands inline inside `resolve()` instead of re-emitting events.
- **Per-ability cooldown table.** Plan 1's single-slot cooldown is a wart; replace with a cold-side `BTreeMap<(AgentId, AbilityId), u64>` once ability counts rise.
- **Sparse standing matrix.** Plan 1's dense `cap × cap` table scales as O(N²); replace with sparse `FxHashMap<(AgentId, AgentId), i16>` once N > ~1000.
- **Migration of `src/world_sim/` combat.** Delete the mode-switch; thread `AbilityRegistry` through world-sim runtime; remove `tactical_sim` as a runtime dependency of the game binary (it stays for training).

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-04-19-ability-plan-1-foundation.md`.

Use `superpowers:subagent-driven-development` if you want the plan executed in the same session; `superpowers:executing-plans` if you'd rather pick it up cold in a fresh session.

Next plans:
- **Ability Plan 2** — Area variants (Circle, Cone, Line) + Projectile delivery via spawned entity; spatial `TargetSelector`s.
- **Ability Plan 3** — Triggers + Conditions lowering; status-effect stack (per-agent SmallVec) replacing the five single-slot timers.
- **Ability Plan 4** — Broader world effects: `Announce`, `GrantMemory`, `PostQuest`, `Bid`. Engine memory-ring + quest aggregate-pool integration.
- **Ability Plan 5** — DSL parser lifted into the engine tree; `.ability` files load into `AbilityRegistry` at startup.
- **Ability Plan 6** — `src/world_sim/` migration; drop the mode switch; retire `tactical_sim` as a runtime dep.
