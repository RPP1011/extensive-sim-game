//! Context traits for the DSL IR interpreter.
//!
//! ## Coverage source
//!
//! All methods here are derived directly from the wolves+humans interpreter
//! coverage survey in
//! `docs/superpowers/notes/2026-04-22-wolves-humans-interp-coverage.md`.
//! No method is included speculatively — if it is not in the survey it is
//! not here.  Methods for unsupported primitives (RNG, voxel, quantifiers,
//! aggregation builtins) are explicitly out of scope for P1b.
//!
//! ## Design: three traits instead of one
//!
//! The four rule classes split cleanly across a **mutation axis**:
//!
//! | Class    | Agent reads | Agent writes | Event emission | View-cell mutation |
//! |----------|-------------|--------------|----------------|--------------------|
//! | Mask     | yes         | no           | no             | no                 |
//! | Scoring  | yes         | no           | no             | no                 |
//! | Physics  | yes         | yes          | yes            | no                 |
//! | View     | yes         | no           | no             | yes (fold `self +=`) |
//!
//! A single `RuleContext` trait would expose mutable agent-write methods
//! (e.g. `set_hp`) to mask and scoring interpreters that must never call them,
//! which would be a footgun.  A split into one read-only base + two mutating
//! extensions keeps each contract narrow:
//!
//! - `ReadContext`    — pure reads; used by masks, scoring, and lazy views.
//! - `CascadeContext: ReadContext` — adds agent mutation + event emission; used
//!   by physics cascade handlers.
//! - `ViewContext: ReadContext`    — adds fold-accumulator mutation; used by
//!   materialized view fold handlers.
//!
//! **Naming deviation from the plan (acceptance criterion #6):** The plan names
//! four traits `MaskContext` / `ScoringContext` / `CascadeContext` / `ViewContext`.
//! Masks and scoring share an identical read-only surface, so they collapse into
//! one `ReadContext`.  Mask interpreters (Task 3) and scoring interpreters
//! (Task 4) both bind `C: ReadContext`.  `CascadeContext` and `ViewContext` keep
//! their names unchanged.
//!
//! The `engine` crate (Task 7) provides the `SimState`-backed impls.  Until
//! then the traits compile with zero impls.
//!
//! ## IDs and engine-agnostic types
//!
//! `dsl_ast` must not depend on `engine`.  This module defines minimal
//! runtime ID newtypes (`AgentId`, `AbilityId`) and a `Vec3` positional type
//! that mirror the engine's shapes exactly (the engine impls just cast).  The
//! `EffectOp` enum is a local mirror of `engine::ability::EffectOp` used by
//! the `abilities.effects` iterator in physics rules; the variants and field
//! types must stay byte-compatible with the engine type.
//!
//! **TODO (Task 7):** Add a compile-time assertion in
//! `crates/engine/src/evaluator/context.rs` verifying discriminant and layout
//! match between `engine::ability::program::EffectOp` and
//! `dsl_ast::eval::EffectOp`.  Something like:
//! ```ignore
//! const _: () = assert!(
//!     std::mem::size_of::<engine::ability::program::EffectOp>()
//!         == std::mem::size_of::<dsl_ast::eval::EffectOp>()
//! );
//! ```
//! plus variant-count / discriminant checks.  No implementation needed before
//! Task 7 — this note records the intent.

// ---------------------------------------------------------------------------
// Runtime ID newtypes (engine-agnostic)
// ---------------------------------------------------------------------------

/// Identifies a live agent in the simulation.
///
/// The engine uses `NonZeroU32` internally, making `Option<AgentId>::None` the
/// canonical zero-cost "absent" representation.  `dsl_ast` mirrors this via the
/// `new()` constructor + `Option<AgentId>` at trait boundaries; the inner field
/// is private to prevent accidental sentinel construction.
///
/// Mirrors `engine_generated::ids::AgentId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AgentId(u32);

impl AgentId {
    /// Construct from a raw non-zero value.  Returns `None` for zero (the
    /// `Option<AgentId>::None` niche).
    #[inline]
    pub fn new(raw: u32) -> Option<Self> {
        if raw == 0 { None } else { Some(AgentId(raw)) }
    }

    /// Return the underlying raw `u32`.  Use to bridge to the engine's
    /// `NonZeroU32::get()` in Task 7 Context impls.
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// Identifies a registered ability program in the ability registry.
///
/// Same sentinel discipline as [`AgentId`]: the inner field is private;
/// construct via `AbilityId::new(raw)` and test absence via `Option<AbilityId>`.
///
/// Mirrors `engine_generated::ids::AbilityId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbilityId(u32);

impl AbilityId {
    /// Construct from a raw non-zero value.  Returns `None` for zero.
    #[inline]
    pub fn new(raw: u32) -> Option<Self> {
        if raw == 0 { None } else { Some(AbilityId(raw)) }
    }

    /// Return the underlying raw `u32`.  Use to bridge to the engine's
    /// `NonZeroU32::get()` in Task 7 Context impls.
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// 3-component position / direction vector.  Using `[f32; 3]` keeps this
/// trivially `Copy` and avoids a dependency on `glam`.
pub type Vec3 = [f32; 3];

// ---------------------------------------------------------------------------
// EffectOp mirror
// ---------------------------------------------------------------------------

/// Mirror of `engine::ability::EffectOp` for use in the `dsl_ast` interpreter.
///
/// The variants and field types must remain in sync with the canonical
/// definition in `crates/engine/src/ability/program.rs`.  The engine's Task 7
/// impl converts between the two by transmitting via a match arm (no unsafe).
///
/// Only the variants reachable in the wolves+humans fixture are required for
/// P1b; all variants are included here for completeness so the mirror stays
/// accurate and doesn't silently drop variants from `abilities.effects`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectOp {
    Damage { amount: f32 },
    Heal { amount: f32 },
    Shield { amount: f32 },
    Stun { duration_ticks: u32 },
    Slow { duration_ticks: u32, factor_q8: i16 },
    TransferGold { amount: i64 },
    ModifyStanding { delta: i16 },
    CastAbility { ability: AbilityId, selector: TargetSelector },
}

/// Target selection for `CastAbility` effects.  Mirrors
/// `engine::ability::program::TargetSelector`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSelector {
    /// Use the caster as the nested target.
    Caster,
    /// Use the original target as the nested target.
    Target,
}

// ---------------------------------------------------------------------------
// ReadContext — pure state reads (masks, scoring, lazy views)
// ---------------------------------------------------------------------------

/// Pure read-only access to simulation state.  All four rule classes use
/// this surface; masks and scoring use only this trait.
///
/// Methods correspond 1-to-1 with the stdlib functions listed in the
/// coverage survey (§§1.2, 2.3, 3.2, 4.2).  Return types use owned
/// primitives / `Copy` types to avoid lifetime gymnastics.
///
/// Implementors: `engine::evaluator::context` (Task 7, under
/// `#[cfg(feature = "interpreted-rules")]`).
pub trait ReadContext {
    // ---- world ----

    /// Current simulation tick counter.
    fn world_tick(&self) -> u32;

    // ---- agents (read-only field accessors) ----

    /// Returns `true` if the agent exists and is alive.
    fn agents_alive(&self, agent: AgentId) -> bool;

    /// Current world-space position of `agent`.
    fn agents_pos(&self, agent: AgentId) -> Vec3;

    /// Current HP of `agent`.
    fn agents_hp(&self, agent: AgentId) -> f32;

    /// Maximum HP cap of `agent`.
    fn agents_max_hp(&self, agent: AgentId) -> f32;

    /// HP as a ratio `hp / max_hp` in `[0.0, 1.0]`.
    ///
    /// The interpreter computes `agents_hp / agents_max_hp` for `IrExpr::Field`
    /// nodes referencing `hp_pct`; this convenience method lets the impl
    /// provide a single read instead of two.  Returns `0.0` when max_hp is 0.
    fn agents_hp_pct(&self, agent: AgentId) -> f32;

    /// Current shield buffer HP.
    fn agents_shield_hp(&self, agent: AgentId) -> f32;

    /// Absolute tick at which the agent's stun expires (`0` = not stunned).
    fn agents_stun_expires_at_tick(&self, agent: AgentId) -> u32;

    /// Absolute tick at which the agent's slow expires (`0` = not slowed).
    fn agents_slow_expires_at_tick(&self, agent: AgentId) -> u32;

    /// Slow factor in q8 fixed-point (`0` = not slowed).
    fn agents_slow_factor_q8(&self, agent: AgentId) -> i16;

    /// Base melee attack damage for `agent`.
    fn agents_attack_damage(&self, agent: AgentId) -> f32;

    /// The agent `agent` is currently engaged with, if any.
    ///
    /// Also serves the DSL's `engaged_with_or(x, sentinel)` form — the physics
    /// interpreter unwraps `None` to the sentinel at the call site.
    fn agents_engaged_with(&self, agent: AgentId) -> Option<AgentId>;

    /// Hostility predicate: does `a` treat `b` as a hostile?
    fn agents_is_hostile_to(&self, a: AgentId, b: AgentId) -> bool;

    /// Current gold balance of `agent`.  Used by the `transfer_gold` physics
    /// rule (not in wolves+humans, but included because `agents.gold` is in the
    /// coverage survey §3.2).
    fn agents_gold(&self, agent: AgentId) -> i64;

    // ---- query (spatial / collection) ----

    /// Enumerate all agents within `radius` of `center`, calling `f` for each.
    ///
    /// Chose a closure-passing shape instead of `impl Iterator` to avoid
    /// lifetime constraints on the implementor's borrow of the agent table.
    fn query_nearby_agents(&self, center: Vec3, radius: f32, f: &mut dyn FnMut(AgentId));

    /// Enumerate all agents within `radius` of `center` that are the same
    /// species / kin-group as `origin`, calling `f` for each.
    fn query_nearby_kin(&self, origin: AgentId, center: Vec3, radius: f32, f: &mut dyn FnMut(AgentId));

    /// Return the nearest hostile agent to `agent` within `radius`.
    /// Returns `None` when no hostile is within range.
    fn query_nearest_hostile_to(&self, agent: AgentId, radius: f32) -> Option<AgentId>;

    // ---- abilities (registry reads) ----

    /// `abilities.is_known(ab)` — does the registry contain an entry for `ab`?
    fn abilities_is_known(&self, ab: AbilityId) -> bool;

    /// `abilities.known(self, ab)` — does `agent` have ability `ab`?
    fn abilities_known(&self, agent: AgentId, ab: AbilityId) -> bool;

    /// `abilities.cooldown_ready(self, ab)` — is the cooldown for `ab` ready
    /// for `agent`?
    fn abilities_cooldown_ready(&self, agent: AgentId, ab: AbilityId) -> bool;

    /// Cooldown duration of ability `ab` in ticks.
    fn abilities_cooldown_ticks(&self, ab: AbilityId) -> u32;

    /// Iterate the effect ops of ability `ab`, calling `f` for each.
    fn abilities_effects(&self, ab: AbilityId, f: &mut dyn FnMut(EffectOp));

    // ---- config (tunable constants) ----

    /// `config.combat.attack_range` — melee attack search radius in metres.
    fn config_combat_attack_range(&self) -> f32;

    /// `config.combat.engagement_range` — engagement radius in metres.
    fn config_combat_engagement_range(&self) -> f32;

    /// `config.movement.max_move_radius` — movement search radius in metres.
    fn config_movement_max_move_radius(&self) -> f32;

    /// `config.cascade.max_iterations` — maximum cascade depth.
    fn config_cascade_max_iterations(&self) -> u32;

    // ---- views (computed predicates, called from masks and scoring) ----

    /// `view::is_hostile(a, b)` — hostility matrix lookup (lazy view).
    fn view_is_hostile(&self, a: AgentId, b: AgentId) -> bool;

    /// `view::is_stunned(x)` — stun predicate based on tick vs expiry.
    fn view_is_stunned(&self, agent: AgentId) -> bool;

    /// `view::threat_level(self, target)` — accumulated threat value of
    /// `target` against `observer` (materialized view).
    fn view_threat_level(&self, observer: AgentId, target: AgentId) -> f32;

    /// `view::my_enemies(self, target)` — grudge flag: has `target` attacked
    /// `observer`?  Returns `1.0` if yes, `0.0` otherwise.
    fn view_my_enemies(&self, observer: AgentId, target: AgentId) -> f32;

    /// `view::pack_focus(self, target)` — pack-hunt engagement beacon for
    /// `target` from the perspective of `observer`.
    fn view_pack_focus(&self, observer: AgentId, target: AgentId) -> f32;

    /// `view::kin_fear(self, _)` — accumulated rout-fear signal for `observer`
    /// (wildcard target — sums over all sources).
    fn view_kin_fear(&self, observer: AgentId) -> f32;

    /// `view::rally_boost(self, _)` — accumulated rally signal for `observer`
    /// (wildcard target — sums over all sources).
    fn view_rally_boost(&self, observer: AgentId) -> f32;

    /// `view::slow_factor(a)` — current slow multiplier for `agent` as a
    /// normalized float in `[0.0, 1.0]`.  From the lazy `slow_factor` view.
    fn view_slow_factor(&self, agent: AgentId) -> f32;
}

// ---------------------------------------------------------------------------
// CascadeContext — mutable agent state + event emission (physics)
// ---------------------------------------------------------------------------

/// Extends `ReadContext` with the agent mutation and event-emission surface
/// required by physics cascade handlers.
///
/// Methods correspond to the write-side stdlib functions in the coverage
/// survey §3.2 (`agents.set_*`, `agents.kill`, `agents.add_gold`, etc.) plus
/// the `emit` statement surface.
pub trait CascadeContext: ReadContext {
    // ---- mutable agent writes ----

    fn agents_set_hp(&mut self, agent: AgentId, hp: f32);
    fn agents_set_shield_hp(&mut self, agent: AgentId, shield_hp: f32);
    fn agents_set_stun_expires_at_tick(&mut self, agent: AgentId, expires_at: u32);
    fn agents_set_slow_expires_at_tick(&mut self, agent: AgentId, expires_at: u32);
    fn agents_set_slow_factor_q8(&mut self, agent: AgentId, factor: i16);
    fn agents_set_engaged_with(&mut self, a: AgentId, b: AgentId);
    fn agents_clear_engaged_with(&mut self, agent: AgentId);

    /// Mark `agent` as dead.  Called by the damage handler when HP drops to 0.
    fn agents_kill(&mut self, agent: AgentId);

    fn agents_add_gold(&mut self, agent: AgentId, amount: i64);
    fn agents_sub_gold(&mut self, agent: AgentId, amount: i64);

    /// Adjust faction standing between `a` and `b` by `delta`.  Clamped to
    /// `[-1000, 1000]` inside the engine impl.
    fn agents_adjust_standing(&mut self, a: AgentId, b: AgentId, delta: i16);

    /// Record a memory event for `observer` about `subject`.
    /// Params mirror `agents.record_memory(observer, subject, feeling, context, tick)`.
    fn agents_record_memory(
        &mut self,
        observer: AgentId,
        subject: AgentId,
        feeling: f32,
        context: u32,
        tick: u32,
    );

    /// Mark that `agent`'s cooldown for ability `ab` starts now; the ability
    /// will be ready again at `world_tick() + cooldown_ticks(ab)`.
    ///
    /// Writes to the per-`(agent, ability)` cooldown slot — not to agent state
    /// itself.  This is why the method name uses the `abilities_` prefix rather
    /// than `agents_`, making the naming asymmetry explicit.
    fn abilities_set_cooldown_next_ready(&mut self, agent: AgentId, ab: AbilityId, ready_at: u32);

    // ---- cascade event emission ----

    /// Emit a named event with a flat key-value payload.
    ///
    /// The physics interpreter dispatches `IrStmt::Emit` through this method.
    /// The event name is the DSL event name (e.g. `"EffectDamageApplied"`); the
    /// fields slice carries field-name/value pairs as `EvalValue`.
    fn emit(&mut self, event_name: &str, fields: &[(&str, EvalValue)]);
}

// ---------------------------------------------------------------------------
// ViewContext — fold-accumulator mutation (materialized views)
// ---------------------------------------------------------------------------

/// Extends `ReadContext` with the fold-accumulator mutation surface required
/// by materialized view fold handlers.
///
/// Materialized views accumulate via `self += delta` in their fold body.
/// The interpreter translates `IrStmt::SelfUpdate` into calls on this trait.
pub trait ViewContext: ReadContext {
    /// Apply a `+= delta` update to the view cell identified by `(view_name, key)`.
    ///
    /// `key` is a 1- or 2-element slice: `[a]` for per-agent views, `[a, b]`
    /// for pair-keyed views.  The engine impl routes this to the correct
    /// storage shape.
    fn view_self_add(&mut self, view_name: &str, key: &[AgentId], delta: f32);

    /// Apply a `+= delta` update to an integer view cell.  Used by the
    /// `engaged_with` materialized view (initial `0`, increments by `1`).
    fn view_self_add_int(&mut self, view_name: &str, key: &[AgentId], delta: i64);
}

// ---------------------------------------------------------------------------
// EvalValue — dynamically typed value for emit payloads
// ---------------------------------------------------------------------------

/// A dynamically typed scalar value used in `CascadeContext::emit` field payloads.
///
/// Kept deliberately small — only the types reachable in wolves+humans event
/// emissions.  The engine impl matches on this to populate event structs.
///
/// The variant set is intentionally restricted to what wolves+humans exercises.
/// Future rule classes may need `Str(&'static str)` or similar; resist adding
/// variants speculatively — add them when a concrete rule class requires them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvalValue {
    Bool(bool),
    I32(i32),
    I64(i64),
    U32(u32),
    F32(f32),
    Agent(AgentId),
    Ability(AbilityId),
}

impl From<bool> for EvalValue {
    fn from(v: bool) -> Self { EvalValue::Bool(v) }
}
impl From<i32> for EvalValue {
    fn from(v: i32) -> Self { EvalValue::I32(v) }
}
impl From<i64> for EvalValue {
    fn from(v: i64) -> Self { EvalValue::I64(v) }
}
impl From<u32> for EvalValue {
    fn from(v: u32) -> Self { EvalValue::U32(v) }
}
impl From<f32> for EvalValue {
    fn from(v: f32) -> Self { EvalValue::F32(v) }
}
impl From<AgentId> for EvalValue {
    fn from(v: AgentId) -> Self { EvalValue::Agent(v) }
}
impl From<AbilityId> for EvalValue {
    fn from(v: AbilityId) -> Self { EvalValue::Ability(v) }
}

// ---------------------------------------------------------------------------
// Interpreter sub-modules (one per rule class)
// ---------------------------------------------------------------------------

pub mod mask;
