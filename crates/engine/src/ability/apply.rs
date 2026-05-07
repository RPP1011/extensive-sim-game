//! Registry-driven apply dispatch (#125, MVP slice).
//!
//! Translates `AbilityProgram` IR into a stream of typed `ApplyEvent`s
//! that downstream sims can drain into their existing event rings.
//! Honors the per-effect chance gate (`program.chances[i]`) using
//! `per_agent_u32_pcg_with_extra` per the P5 keyed-PCG contract +
//! P11 cross-backend bit-equality — same WGSL prelude and host
//! mixer chain so a chance-gated effect produces a byte-identical
//! chronicle record across CPU and GPU dispatchers.
//!
//! # Status
//!
//! MVP scope is the per-(caster, target) translation pass. Out of
//! scope for this slice:
//!   * scaling lookup against caster stat SoA (needs sim-side stat
//!     resolver — pass via callback or context struct in a follow-up)
//!   * GPU-side multi-target AOE expansion (Path B for #121 — the
//!     CPU oracle path lives in `apply_program_aoe`, but the GPU
//!     dispatcher's chronicle write loop still consumes a single
//!     explicit target slot per cast). **BGL opt-in landed
//!     2026-05-07** (commit message: `feat(dsl): BGL opt-in for
//!     spatial-bind in chronicle dispatcher`): a per-fixture
//!     [`dsl_compiler::cg::lower::LowerOpts::aoe_dispatch`] flag
//!     stamps `with_aoe_dispatch: true` onto every
//!     `CgStmt::ApplyAbility` lowered under the opting-in
//!     `Compilation`. Production runtimes (`duel_abilities_runtime`,
//!     `tactical_squad_5v5_runtime`, `boss_fight_runtime`,
//!     `duel_25v25_runtime`, `mass_battle_100v100_runtime`) keep
//!     the default `false` so their existing single-target
//!     dispatcher emit + zero-spatial-overhead BGL are preserved.
//!     **Path B emit remains TODO** — the WGSL dispatcher's
//!     `CgStmt::ApplyAbility` arm reads the flag (currently as
//!     `_with_aoe_dispatch`) but unconditionally emits the
//!     single-target chain. Wiring the AOE walk shape (gate on
//!     `area_kinds[effect_base + i] == 0u` for Circle, walk the
//!     27-cell neighborhood around `agent_pos[target_slot]`, gate
//!     each candidate on `dist² <= radius²`, then re-execute the
//!     chronicle arm chain with the candidate slot shadowing
//!     `target_slot`) and surfacing the matching reads via a new
//!     `wire_apply_ability_aoe_reads` helper (sibling to
//!     `wire_ability_registry_column_reads`) is the next slice.
//!   * Non-AOE-Path-B AOE shapes (Line, Sphere, Box, etc.) on CPU.
//!     `apply_program_aoe` expands `Circle` (#121 follow-on) and
//!     `Cone` (#178) slots; other shapes fall back to single-target
//!     dispatch on `primary_target`. The cone math lives in
//!     `apply_program_aoe_cone_filter` (range² gate ∧ angular
//!     half-angle gate, mirroring the GPU kernel's WGSL bit-for-bit
//!     for P11 byte-equal parity). The other 10 shapes still need
//!     additional geometry kernels (capsule, AABB, …).
//!   * delivery method scheduling (Projectile travel, Channel hold —
//!     #124 IR done; runtime not wired)
//!
//! **Wave 1.5#9 nested-effect dispatch (2026-05-06).** After the
//! primary effect's ApplyEvent is emitted for slot `i`, the dispatch
//! walks `program.nested_per_effect[i]` and emits an ApplyEvent per
//! nested op. Nested ops apply to the SAME target as the primary
//! (the .ability source's `<verb> <args> { <inner_stmt>; ... }` shape
//! treats inner stmts as auxiliary effects riding on the primary's
//! cast). Nested ops carry no chance gate or scaling slot today —
//! inner-stmt modifiers were silently dropped at lowering (see
//! `program.nested_per_effect` doc; recursive aggregator capture is
//! later infrastructure). Closes the documented gap surfaced by the
//! Reap verb swap (commit `72a35307`): Reap's `{ stun 1s }` now
//! produces an ApplyEvent::Stun in addition to ApplyEvent::Execute.
//!
//! # Contract with sims
//!
//! Sims that opt into registry-driven dispatch:
//!   1. Bind their event vocabulary to engine's `ApplyEvent` (or
//!      provide a translator at the boundary).
//!   2. Replace per-verb hand-mirrored emit blocks with a single
//!      generic `apply_program(ability_id, caster, target, …)` call
//!      from each verb body.
//!   3. Keep their existing apply-physics chronicles
//!      (ApplyDamage / ApplyHeal / etc.) — those drain `ApplyEvent`s
//!      into SoA mutations exactly the same way.
//!
//! With this slice landed, a sim with N hand-mirrored verbs collapses
//! to one generic dispatcher; adding a new ability becomes a pure
//! .ability-file change.

use crate::ability::program::{
    AbilityProgram, BuffStat, CasterStats, EffectOp, EffectPredicate,
    EffectPredicateBinder, EffectPredicateOp, ShapeKind,
};
use crate::ids::AgentId;
use crate::rng::per_agent_u32_pcg_with_extra;
use smallvec::SmallVec;

/// Typed apply-event vocabulary. Each variant matches an `EffectOp`
/// shape exactly, expanded with the caster/target context resolved at
/// dispatch time. Sims consume these via their existing apply-physics
/// chronicles (`on Damaged { … }`, `on Healed { … }`, etc.).
///
/// `source = u32::MAX` for self-only effects (Stun/Buff target the
/// passed-in target; the source field is unused but kept for shape
/// uniformity). Apply-physics handlers should match on the variant
/// they care about and ignore unused fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApplyEvent {
    Damage     { source: AgentId, target: AgentId, amount: f32 },
    Heal       { source: AgentId, target: AgentId, amount: f32 },
    Shield     { source: AgentId, target: AgentId, amount: f32 },
    Stun       { target: AgentId, duration_ticks: u32 },
    Slow       { target: AgentId, duration_ticks: u32, factor_q8: i16 },
    Root       { target: AgentId, duration_ticks: u32 },
    Silence    { target: AgentId, duration_ticks: u32 },
    Fear       { target: AgentId, duration_ticks: u32 },
    Taunt      { target: AgentId, duration_ticks: u32 },
    Dash       { source: AgentId, distance: f32 },
    Blink      { source: AgentId, distance: f32 },
    Knockback  { source: AgentId, target: AgentId, distance: f32 },
    Pull       { source: AgentId, target: AgentId, distance: f32 },
    Execute    { target: AgentId, hp_threshold: f32 },
    SelfDamage { source: AgentId, amount: f32 },
    LifeSteal  { target: AgentId, duration_ticks: u32, fraction_q8: i16 },
    DamageModify { target: AgentId, duration_ticks: u32, multiplier_q8: i16 },
    DamageOverTime { source: AgentId, target: AgentId, amount: f32, duration_ticks: u32 },
    HealOverTime   { source: AgentId, target: AgentId, amount: f32, duration_ticks: u32 },
    TimedShield    { source: AgentId, target: AgentId, amount: f32, duration_ticks: u32 },
    Buff           { target: AgentId, stat: BuffStat, magnitude_q8: i16, duration_ticks: u32 },
    /// `summon "<template>" [N] [for <duration>]` — caster spawns
    /// `count` minions of `template_hash` for `lifetime_ticks`. The
    /// template hash is the FxHash of the template ident from the
    /// .ability source (deferred resolution — apply handlers map the
    /// hash to a spawner via a registry follow-up). Captured here so
    /// downstream sims can drain the event when the spawner wires up;
    /// no runtime sim consumes it today (deferred infra mirroring the
    /// CastAbility/TransferGold/ModifyStanding fall-through pattern).
    Summon         { source: AgentId, template_hash: u32, count: u8, lifetime_ticks: u32 },
    /// `harvest "<kind>" [<amount>]` — caster gathers `amount` units of
    /// the named resource. `kind_hash` is the FxHash of the resource
    /// ident from the .ability source (deferred resolution — apply
    /// handlers map the hash to a concrete resource via a registry
    /// follow-up). Apply handlers route to AgentHarvested for organic /
    /// surface resources or AgentHarvestedVoxel for voxel-backed
    /// resources, distinguished by the registry lookup. No runtime sim
    /// consumes it today (deferred infra mirroring the Summon
    /// fall-through pattern).
    Harvest        { source: AgentId, kind_hash: u32, amount: u16 },
    /// `place_voxel "<kind>"` — caster places one voxel of `kind_hash`
    /// at the cast target's position. Apply handlers emit
    /// AgentPlacedVoxel and write the voxel into world state; deferred
    /// infrastructure today (no runtime sim consumes the event yet).
    PlaceVoxel     { source: AgentId, kind_hash: u32 },
    /// `stealth for <duration>` — self-cast invisibility for
    /// `duration_ticks`. The LoL idiom is `stealth for 3s
    /// break_on_damage` — the lifetime modifier rides the per-effect
    /// lifetime SoA and isn't reflected here. Apply handlers will
    /// gate target selection by the caster's stealth flag. No runtime
    /// sim consumes the event today (deferred — same fall-through as
    /// Summon / Harvest / PlaceVoxel).
    Stealth        { source: AgentId, duration_ticks: u32 },
    /// Wave 2 piece 8 CC verbs. Same shape as Stun (target + duration);
    /// apply handlers wire per-agent expiry tick-stamps. No runtime
    /// sim consumes them today.
    Charm          { target: AgentId, duration_ticks: u32 },
    Grounded       { target: AgentId, duration_ticks: u32 },
    Suppress       { target: AgentId, duration_ticks: u32 },
    /// `reflect <fraction> for <duration>` — fraction-of-damage
    /// bounce. Mirrors DamageModify's payload shape.
    Reflect        { target: AgentId, duration_ticks: u32, fraction_q8: i16 },
    /// `transfer_gold <amount>` — caster moves `amount` gold to
    /// target. The world-state effect (debiting caster's purse,
    /// crediting target's purse) is downstream of apply_program;
    /// this variant signals **the cast occurred** for chronicle /
    /// reaction-handler consumers. Pairs with
    /// `EventKindId::EffectGoldTransfer = 31` on the chronicle side.
    TransferGold   { source: AgentId, target: AgentId, amount: i32 },
    /// `modify_standing <delta>` — caster changes their standing
    /// with target by `delta` (i16 signed delta in standing's
    /// internal units). Same world-state-deferred shape as
    /// TransferGold. Pairs with `EventKindId::EffectStandingDelta =
    /// 32` on the chronicle side.
    ModifyStanding { source: AgentId, target: AgentId, delta: i16 },
}

/// Inline budget — most abilities have ≤4 effects (P4 says
/// `MAX_EFFECTS_PER_PROGRAM = 6` today). Heap-spill is fine for
/// the rare 5+ ult.
const APPLY_INLINE: usize = 4;

/// Translate one cast of `program` (caster → target at `tick`) into a
/// stream of ApplyEvents. Honors the per-effect chance gate AND the
/// per-effect `scalings_per_effect` modifier (`+ N% stat_ref`).
///
/// `world_seed` and `tick` together with `caster` derive the RNG
/// stream per P5 — replay equivalence holds because the same cast at
/// the same tick produces the same gate decisions.
///
/// `caster_stats` is the caster's stat snapshot at cast-decide time.
/// For each amount-bearing variant (Damage / Heal / Shield / SelfDamage
/// / DamageOverTime / HealOverTime / TimedShield), the dispatcher
/// computes `scaled = base + Σ percent * stat` from
/// `program.scalings_per_effect[i]` before emitting the event. Pass
/// `&CasterStats::default()` for legacy / non-scaling call sites —
/// all-zero stats project to a `0.0` contribution per scaling slot, so
/// the output is byte-identical to the pre-scaling apply path when the
/// program carries no scalings (or when the caster has no relevant
/// stats).
///
/// `target_stats` is the target's stat snapshot at cast-decide time.
/// Threaded so the per-effect when-predicate evaluator (Wave 1.5#7
/// — this slice) can read `target.<field>` for predicates like
/// `when target.hp < 20`. Pass `&CasterStats::default()` for call
/// sites that don't carry when-predicates — when no slot fires the
/// predicate eval, the column is dead and the snapshot is unread.
///
/// `Some(amount) = 0xFFFF` chance slot fires deterministically (max
/// q16 value — apply handlers treat as "always"); `None` slot also
/// fires deterministically (no gate authored). The runtime gate
/// compares `(per_agent_u32_pcg_with_extra(seed, caster_slot, tick,
/// RngPurpose::Chance, slot_idx) & 0xFFFF) < q16` — when q16=65534
/// (canonical "100%") this is true 65534/65536 ≈ 99.997% of draws
/// (indistinguishable from "always" at 16-bit RNG resolution).
pub fn apply_program(
    program:      &AbilityProgram,
    caster:       AgentId,
    target:       AgentId,
    tick:         u64,
    world_seed:   u64,
    caster_stats: &CasterStats,
    target_stats: &CasterStats,
) -> SmallVec<[ApplyEvent; APPLY_INLINE]> {
    let mut out: SmallVec<[ApplyEvent; APPLY_INLINE]> = SmallVec::new();

    for (i, op) in program.effects.iter().enumerate() {
        // -- Wave 1.5#5 chance gate (P11 GPU-parity mixer). --
        // The chances slice is either empty (no effect carried the
        // modifier — fire all) or per-effect Option<u16>. None within
        // a populated slice = no gate on that slot.
        //
        // P11: switched from ahash `per_agent_u32` to the GPU-parity
        // PCG primitive `per_agent_u32_pcg_with_extra` so the GPU
        // dispatcher's chance gate produces a byte-identical chronicle
        // record under the same inputs. Caster's 0-based slot index
        // (`caster.raw() - 1`) keys agent_id — mirrors the WGSL
        // dispatcher's `caster_slot` (= `gid.x`, 0-based). The
        // `RngPurpose::Chance` (id 10) tag is shared across all
        // chance-gated effects; the per-effect slot index `i` is
        // mixed in via `extra` so multi-effect abilities don't share
        // a draw across slots.
        if let Some(Some(q16)) = program.chances.get(i).copied() {
            let caster_slot = caster.raw().saturating_sub(1);
            let draw = per_agent_u32_pcg_with_extra(
                world_seed as u32,
                caster_slot,
                tick as u32,
                /* purpose_id = RngPurpose::Chance.wgsl_id() */ 10,
                i as u32,
            ) & 0xFFFF;
            if (draw as u16) >= q16 {
                continue; // gate fails — skip this effect
            }
        }
        // -- Wave 1.5#7 when-predicate gate (this slice). --
        // Per-effect `when <binder>.<field> <op> <literal>` predicate.
        // Empty `when_per_effect` slice = no slot carried the modifier.
        // `Some(EffectWhenCondition { when_compiled: Some(p), .. })`
        // evaluates the structured predicate against the appropriate
        // stat snapshot (caster vs target). When the predicate fails,
        // skip BOTH the primary AND any nested ops on this slot —
        // matches the chance-gate semantic (auxiliary effects ride on
        // the primary's success).
        //
        // `when_compiled = None` (or out-of-vocab field at runtime)
        // evaluates as false defensively, so a malformed slot does not
        // silently fire — the lower path errors loudly when the
        // predicate fails to compile, so this branch is unreachable in
        // practice for `.ability`-sourced programs.
        if let Some(Some(when)) = program.when_per_effect.get(i) {
            if let Some(pred) = when.when_compiled.as_ref() {
                if !evaluate_predicate(pred, caster_stats, target_stats) {
                    continue; // predicate fails — skip primary + nested
                }
            }
        }
        // -- Wave 1.5#4 scaling — compute additive `Σ percent * stat`
        // bonus from `scalings_per_effect[i]`. Empty/missing slot ⇒ 0.0
        // (output bit-identical to pre-scaling behavior). Apply only to
        // amount-bearing variants in the dispatch arms below.
        let scale_bonus: f32 = program
            .scalings_per_effect
            .get(i)
            .map(|inner| {
                inner
                    .iter()
                    .map(|s| s.percent * caster_stats.get(s.stat_ref))
                    .sum::<f32>()
            })
            .unwrap_or(0.0);
        // -- Per-EffectOp dispatch. Mirrors pack_effect's variant
        // walk. Primary effect carries the slot's scaling bonus; nested
        // ops below are emitted with `scale_bonus = 0.0` because nested
        // ops have no scaling slot in the registry today (inner-stmt
        // modifiers were silently dropped at lowering — see
        // `program.nested_per_effect` doc).
        push_effect_event(&mut out, op, caster, target, scale_bonus);

        // -- Wave 1.5#9 nested-effect dispatch. After the primary's
        // ApplyEvent is emitted for slot `i`, walk
        // `program.nested_per_effect[i]` and emit one ApplyEvent per
        // nested op. Nested ops:
        //   * apply to the SAME target as the primary (auxiliary
        //     effects riding on the primary's cast — `execute ... {
        //     stun 1s }` stuns the target the execute hit),
        //   * have no chance gate (no slot in `program.chances` —
        //     inner-stmt modifiers dropped at lowering),
        //   * have no scaling (no slot in `scalings_per_effect` —
        //     inner-stmt modifiers dropped at lowering).
        // Closes the gap surfaced by the Reap verb swap
        // (commit `72a35307`): Reap's `{ stun 1s }` now produces an
        // ApplyEvent::Stun alongside ApplyEvent::Execute.
        if let Some(nested) = program.nested_per_effect.get(i) {
            for nested_op in nested {
                push_effect_event(&mut out, nested_op, caster, target, 0.0);
            }
        }
    }
    out
}

/// Translate one `EffectOp` into the matching `ApplyEvent` and push
/// it onto `out`. Shared between the primary-effect dispatch and the
/// nested-effect dispatch in `apply_program`. `scale_bonus` is added
/// to amount-bearing variants; pass `0.0` for nested ops (no scaling
/// slot in the registry today).
///
/// Mirrors `pack_effect`'s variant walk in
/// `crates/engine/src/ability/packed.rs` 1:1 — both must enumerate
/// the same set of `EffectOp` variants in the same order.
fn push_effect_event(
    out: &mut SmallVec<[ApplyEvent; APPLY_INLINE]>,
    op: &EffectOp,
    caster: AgentId,
    target: AgentId,
    scale_bonus: f32,
) {
    match *op {
        EffectOp::Damage    { amount } => out.push(ApplyEvent::Damage { source: caster, target, amount: amount + scale_bonus }),
        EffectOp::Heal      { amount } => out.push(ApplyEvent::Heal   { source: caster, target, amount: amount + scale_bonus }),
        EffectOp::Shield    { amount } => out.push(ApplyEvent::Shield { source: caster, target, amount: amount + scale_bonus }),
        EffectOp::Stun      { duration_ticks } => out.push(ApplyEvent::Stun    { target, duration_ticks }),
        EffectOp::Slow      { duration_ticks, factor_q8 } =>
            out.push(ApplyEvent::Slow { target, duration_ticks, factor_q8 }),
        EffectOp::Root      { duration_ticks } => out.push(ApplyEvent::Root    { target, duration_ticks }),
        EffectOp::Silence   { duration_ticks } => out.push(ApplyEvent::Silence { target, duration_ticks }),
        EffectOp::Fear      { duration_ticks } => out.push(ApplyEvent::Fear    { target, duration_ticks }),
        EffectOp::Taunt     { duration_ticks } => out.push(ApplyEvent::Taunt   { target, duration_ticks }),
        EffectOp::Dash      { distance } => out.push(ApplyEvent::Dash  { source: caster, distance }),
        EffectOp::Blink     { distance } => out.push(ApplyEvent::Blink { source: caster, distance }),
        EffectOp::Knockback { distance } => out.push(ApplyEvent::Knockback { source: caster, target, distance }),
        EffectOp::Pull      { distance } => out.push(ApplyEvent::Pull      { source: caster, target, distance }),
        EffectOp::Execute   { hp_threshold } => out.push(ApplyEvent::Execute { target, hp_threshold }),
        EffectOp::SelfDamage{ amount } => out.push(ApplyEvent::SelfDamage { source: caster, amount: amount + scale_bonus }),
        EffectOp::LifeSteal { duration_ticks, fraction_q8 } =>
            out.push(ApplyEvent::LifeSteal { target: caster, duration_ticks, fraction_q8 }),
        EffectOp::DamageModify { duration_ticks, multiplier_q8 } =>
            out.push(ApplyEvent::DamageModify { target, duration_ticks, multiplier_q8 }),
        EffectOp::DamageOverTime { amount, duration_ticks } =>
            out.push(ApplyEvent::DamageOverTime { source: caster, target, amount: amount + scale_bonus, duration_ticks }),
        EffectOp::HealOverTime   { amount, duration_ticks } =>
            out.push(ApplyEvent::HealOverTime   { source: caster, target, amount: amount + scale_bonus, duration_ticks }),
        EffectOp::TimedShield    { amount, duration_ticks } =>
            out.push(ApplyEvent::TimedShield    { source: caster, target, amount: amount + scale_bonus, duration_ticks }),
        EffectOp::Buff { stat, magnitude_q8, duration_ticks } =>
            out.push(ApplyEvent::Buff { target, stat, magnitude_q8, duration_ticks }),
        EffectOp::Summon { template_hash, count, lifetime_ticks } =>
            out.push(ApplyEvent::Summon { source: caster, template_hash, count, lifetime_ticks }),
        // Non-combat verbs phase 1 — world primitives. No scaling
        // applies (these aren't amount-bearing in the combat sense
        // — `amount` is a resource quantity, not an HP delta).
        EffectOp::Harvest    { kind_hash, amount } =>
            out.push(ApplyEvent::Harvest    { source: caster, kind_hash, amount }),
        EffectOp::PlaceVoxel { kind_hash } =>
            out.push(ApplyEvent::PlaceVoxel { source: caster, kind_hash }),
        // Wave 2 piece 7: stealth is self-cast (apply handler
        // gates target selection by caster's stealth flag).
        EffectOp::Stealth    { duration_ticks } =>
            out.push(ApplyEvent::Stealth { source: caster, duration_ticks }),
        // Wave 2 piece 8 CC verbs — target-cast, single duration.
        EffectOp::Charm      { duration_ticks } =>
            out.push(ApplyEvent::Charm    { target, duration_ticks }),
        EffectOp::Grounded   { duration_ticks } =>
            out.push(ApplyEvent::Grounded { target, duration_ticks }),
        EffectOp::Suppress   { duration_ticks } =>
            out.push(ApplyEvent::Suppress { target, duration_ticks }),
        EffectOp::Reflect    { duration_ticks, fraction_q8 } =>
            out.push(ApplyEvent::Reflect  { target, duration_ticks, fraction_q8 }),
        // TransferGold / ModifyStanding emit chronicle-bearing
        // ApplyEvents that signal "the cast happened". The
        // world-state effects (debiting/crediting purses, mutating
        // standing tables) are downstream of apply_program — kept
        // intentionally separate so the chronicle stream stays a
        // pure function of the cast inputs (P5/P11) regardless of
        // when the world-state side-effects land. Pairs with
        // `EventKindId::EffectGoldTransfer = 31` and
        // `EffectStandingDelta = 32` respectively.
        EffectOp::TransferGold { amount } =>
            out.push(ApplyEvent::TransferGold { source: caster, target, amount }),
        EffectOp::ModifyStanding { delta } =>
            out.push(ApplyEvent::ModifyStanding { source: caster, target, delta }),
        // CastAbility is recursive (needs cascade-style
        // re-dispatch); deferred to slice δ. Skip for now.
        EffectOp::CastAbility { .. } => {}
    }
}

/// Wave 1.5#7: evaluate one structured `EffectPredicate` against the
/// caster + target stat snapshots. The binder picks which snapshot the
/// LHS reads from; the field discriminant resolves to a f32 stat value
/// via [`CasterStats::get_by_field_id`]; the op + literal complete the
/// comparison.
///
/// Returns `false` defensively when the field discriminant is
/// out-of-range (no `ScalingStatRef` slot maps to it). The lowering
/// pass guards the in-vocab subset, so this branch is unreachable for
/// `.ability`-sourced programs.
#[inline]
pub(crate) fn evaluate_predicate(
    pred:   &EffectPredicate,
    caster: &CasterStats,
    target: &CasterStats,
) -> bool {
    let stats = match pred.binder {
        EffectPredicateBinder::SelfBinder => caster,
        EffectPredicateBinder::Target     => target,
    };
    let lhs = match stats.get_by_field_id(pred.field) {
        Some(v) => v,
        None    => return false,
    };
    let rhs = pred.literal;
    match pred.op {
        EffectPredicateOp::Lt => lhs <  rhs,
        EffectPredicateOp::Le => lhs <= rhs,
        EffectPredicateOp::Gt => lhs >  rhs,
        EffectPredicateOp::Ge => lhs >= rhs,
        EffectPredicateOp::Eq => lhs == rhs,
        EffectPredicateOp::Ne => lhs != rhs,
    }
}

/// Task #121 (Path A — CPU-only): multi-target AOE dispatch.
///
/// Translate one cast of `program` (caster → primary_target at `tick`)
/// into a stream of ApplyEvents, expanding per-effect AOE slots over
/// `aoe_targets`.
///
/// Per-effect slot dispatch:
///   * If `program.per_effect_areas[i]` is `Some(EffectAreaShape{
///     kind: Circle, .. })` or `Some(EffectAreaShape{ kind: Cone, .. })`,
///     the slot fires once per `aoe_targets` entry — one ApplyEvent
///     per target. The caller is responsible for performing the
///     spatial + geometric filter (Circle: `state.spatial().within_radius(
///     state, target_pos, args[0])`; Cone: range² gate ∧ angular gate,
///     see `apply_program_aoe_cone_filter` for the canonical CPU
///     filter that mirrors the GPU kernel's WGSL math) and passing
///     the result here. The slice MUST be sorted ascending by raw
///     `AgentId` (P11 — `SpatialHash::within_radius` does this by
///     construction; the cone helper sorts as its final step).
///   * Any other `Some(...)` shape (Line, Sphere, Box, etc.) —
///     deferred. The slot falls back to single-target dispatch on
///     `primary_target`. The other shapes need additional geometry
///     kernels (capsule, AABB, etc.) which Path A defers.
///   * `None` slot (single-target, default) — fires once on
///     `primary_target`, identical to `apply_program`'s behavior.
///
/// **Chance gate semantic.** The chance gate's RNG draw is keyed by
/// `(world_seed, caster, tick, slot_index)` — NOT per target. This
/// makes the AOE slot all-or-nothing: when the gate fires, every
/// target in `aoe_targets` receives the event; when it fails, none
/// of them do. Per-target independent procs would require a different
/// purpose tag and are out of scope for this slice.
///
/// **When-predicate semantic.** The when-predicate evaluates against
/// `target_stats` (the primary target's snapshot) — same as the
/// single-target dispatcher. Per-target predicate evaluation across
/// the AOE expansion is deferred (would require a per-target stat
/// snapshot slice; this slice ships the simplest semantic — the
/// predicate gates the AOE *slot* as a whole, then the slot fires
/// across all targets if it passes).
///
/// **Nested-effect semantic.** When the primary effect on a slot is
/// expanded across `aoe_targets`, each target receives the nested
/// ops too — same target as the primary on that iteration.
///
/// **GPU parity.** GPU dispatcher is single-target only today
/// (consumes only the explicit cast target slot). Multi-target AOE
/// dispatch on GPU is deferred (Path B): the kernel would need a
/// spatial-query loop inside the apply path. Until then, AOE
/// expansion is a CPU-only oracle path.
pub fn apply_program_aoe(
    program:        &AbilityProgram,
    caster:         AgentId,
    primary_target: AgentId,
    aoe_targets:    &[AgentId],
    tick:           u64,
    world_seed:     u64,
    caster_stats:   &CasterStats,
    target_stats:   &CasterStats,
) -> SmallVec<[ApplyEvent; APPLY_INLINE]> {
    let mut out: SmallVec<[ApplyEvent; APPLY_INLINE]> = SmallVec::new();

    for (i, op) in program.effects.iter().enumerate() {
        // Chance gate (same shape as apply_program — slot-keyed,
        // all-or-nothing across the AOE expansion). P11: PCG mixer
        // for cross-backend parity; see the doc comment in
        // `apply_program`.
        if let Some(Some(q16)) = program.chances.get(i).copied() {
            let caster_slot = caster.raw().saturating_sub(1);
            let draw = per_agent_u32_pcg_with_extra(
                world_seed as u32,
                caster_slot,
                tick as u32,
                /* purpose_id = RngPurpose::Chance.wgsl_id() */ 10,
                i as u32,
            ) & 0xFFFF;
            if (draw as u16) >= q16 {
                continue;
            }
        }
        // When-predicate gate (slot-keyed against `target_stats` —
        // primary target's snapshot; per-target eval deferred).
        if let Some(Some(when)) = program.when_per_effect.get(i) {
            if let Some(pred) = when.when_compiled.as_ref() {
                if !evaluate_predicate(pred, caster_stats, target_stats) {
                    continue;
                }
            }
        }
        // Scaling bonus (same shape as apply_program).
        let scale_bonus: f32 = program
            .scalings_per_effect
            .get(i)
            .map(|inner| {
                inner
                    .iter()
                    .map(|s| s.percent * caster_stats.get(s.stat_ref))
                    .sum::<f32>()
            })
            .unwrap_or(0.0);

        // Choose target list for this slot. Circle and Cone expand
        // across aoe_targets (caller pre-filters for both — Circle by
        // `within_radius`, Cone by `apply_program_aoe_cone_filter`);
        // everything else (None, or unrecognised shape) is
        // single-target on primary_target.
        let is_aoe_shape = matches!(
            program.per_effect_areas.get(i).copied().flatten(),
            Some(shape) if shape.kind == ShapeKind::Circle
                        || shape.kind == ShapeKind::Cone
        );
        let targets_for_slot: &[AgentId] = if is_aoe_shape {
            aoe_targets
        } else {
            std::slice::from_ref(&primary_target)
        };

        for &t in targets_for_slot {
            push_effect_event(&mut out, op, caster, t, scale_bonus);
            // Nested ops ride on each target (auxiliary effects fire
            // on the same target the primary just hit).
            if let Some(nested) = program.nested_per_effect.get(i) {
                for nested_op in nested {
                    push_effect_event(&mut out, nested_op, caster, t, 0.0);
                }
            }
        }
    }
    out
}

/// CPU-side cone-filter oracle for AOE Path B Cone (mirrors the GPU
/// kernel's WGSL math bit-for-bit so callers feeding `apply_program_aoe`
/// produce a chronicle record set byte-equal to the GPU dispatcher).
///
/// Inputs:
///   * `apex` — caster's position (cone origin).
///   * `target_pos` — explicit cast target's position. The cone faces
///     `direction = normalize(target_pos - apex)`.
///   * `half_angle_deg` — `args[0]` from the `EffectAreaShape` (cone's
///     half-angle in degrees; the cone's full opening is `2 *
///     half_angle_deg`).
///   * `range` — `args[1]` from the `EffectAreaShape` (max distance
///     from apex along the cone axis).
///   * `candidates` — slice of `(AgentId, position)` pairs the caller
///     pre-collected from the spatial grid (the GPU walks 27 cells
///     around `apex`; the CPU helper accepts any superset and filters
///     down — typical caller is `state.spatial().within_radius(state,
///     apex, range)` mapped to (id, pos) pairs).
///
/// Output: in-cone `AgentId`s sorted ascending by raw id (P11). The
/// apex itself (caster) is excluded — the cone never targets its own
/// origin (degenerate `to_cand = (0,0,0)` would produce an undefined
/// direction; the GPU kernel skips the candidate explicitly when its
/// position equals the apex).
///
/// Edge case: if `target_pos == apex` (caster targets self), the
/// cone's direction is degenerate (zero-vector). Both backends MUST
/// return an empty in-cone set in this case (the cone is undefined,
/// and emitting any subset would create CPU↔GPU drift). The CPU
/// short-circuits here; the GPU's `dir_len_sq < 1e-6` branch matches.
///
/// In-cone predicate (per candidate, identical to the WGSL kernel):
///   1. `cand_pos != apex` (position-equality, not id-equality —
///      handles two agents stacked at the same world-coord).
///   2. `dist² ≤ range²` where `dist² = dot(to_cand, to_cand)`.
///   3. `dot(normalize(to_cand), direction) ≥ cos(half_angle_rad)`.
///
/// **P11.** Both backends evaluate the predicate identically (same
/// f32 ops in the same order: subtract, dot, inverseSqrt, dot). The
/// final sort makes the post-filter set deterministic — GPU's atomic
/// ring claim doesn't preserve order, but the parity sweep sorts
/// post-readback (canonicalize) and the sets agree.
pub fn apply_program_aoe_cone_filter(
    apex:           glam::Vec3,
    target_pos:     glam::Vec3,
    half_angle_deg: f32,
    range:          f32,
    candidates:     &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    // Degenerate cone — caster targets self. The direction is
    // undefined; emit no targets to match the GPU kernel.
    let direction_raw = target_pos - apex;
    let dir_len_sq = direction_raw.dot(direction_raw);
    if dir_len_sq < 1e-6 {
        return Vec::new();
    }
    let direction = direction_raw * dir_len_sq.recip().sqrt();
    let half_angle_rad = half_angle_deg * std::f32::consts::PI / 180.0;
    let cos_half_angle = half_angle_rad.cos();
    let range_sq = range * range;

    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let to_cand = cand_pos - apex;
        let dist_sq = to_cand.dot(to_cand);
        // Apex exclusion: candidate at the cone origin.
        if dist_sq < 1e-6 {
            continue;
        }
        if dist_sq > range_sq {
            continue;
        }
        let cand_dir = to_cand * dist_sq.recip().sqrt();
        if cand_dir.dot(direction) < cos_half_angle {
            continue;
        }
        hits.push(id);
    }
    // P11 reduction-determinism: sort ascending by raw AgentId so the
    // CPU oracle and GPU dispatcher (after canonicalize) agree on
    // emit order.
    hits.sort_by_key(|id| id.raw());
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ability::program::{
        EffectAreaShape, EffectScaling, EffectWhenCondition, Gate, ScalingStatRef,
    };
    use crate::ability::AbilityId;
    use smallvec::smallvec;

    fn caster() -> AgentId { AgentId::new(1).unwrap() }
    fn target() -> AgentId { AgentId::new(2).unwrap() }

    #[test]
    fn apply_strike_emits_damage_event() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            ApplyEvent::Damage { source, target: t, amount }
            if source == caster() && t == target() && amount == 30.0
        ));
    }

    #[test]
    fn apply_multi_effect_program_emits_in_order() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 20.0 },
                EffectOp::Stun   { duration_ticks: 10 },
            ],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ApplyEvent::Damage { .. }));
        assert!(matches!(events[1], ApplyEvent::Stun { .. }));
    }

    #[test]
    fn chance_zero_gates_out() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Stun { duration_ticks: 10 }],
        );
        // q16 = 0 → no draw can be < 0; effect never fires.
        prog.chances.push(Some(0));
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 0, "chance=0 must gate the effect out");
    }

    #[test]
    fn chance_max_always_fires() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Stun { duration_ticks: 10 }],
        );
        // q16 = 0xFFFE (canonical 100% per the chance lowering's
        // clamp(0..=65534)) — fires for any draw < 65534, i.e. all
        // but 1/65536 of draws. Try a fixed seed/tick combination to
        // verify the expected fire (deterministic).
        prog.chances.push(Some(0xFFFE));
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 1, "chance=0xFFFE must fire deterministically at this seed/tick");
    }

    #[test]
    fn chance_deterministic_replay() {
        // Same (program, caster, target, tick, seed) must produce the
        // same gate decision across calls — P5 replay equivalence.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Stun { duration_ticks: 10 }],
        );
        prog.chances.push(Some(32768)); // 50%
        let a = apply_program(&prog, caster(), target(), 42, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        let b = apply_program(&prog, caster(), target(), 42, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(a.len(), b.len(), "same inputs → same gate decisions");
    }

    #[test]
    fn cast_ability_falls_through() {
        // CastAbility is recursive cascade — out of MVP scope. Apply
        // skips it without panicking (P10 — no panic on hot path).
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::CastAbility {
                ability:  AbilityId::new(1).unwrap(),
                selector: crate::ability::program::TargetSelector::Target,
            }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 0, "CastAbility falls through (deferred)");
    }

    #[test]
    fn transfer_gold_emits_apply_event_with_amount() {
        // EffectOp::TransferGold packs source=caster, target=target,
        // amount=raw i32. World-state effects (purse debit/credit)
        // are downstream of apply_program.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::TransferGold { amount: 42 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 1, "TransferGold emits exactly one ApplyEvent");
        match events[0] {
            ApplyEvent::TransferGold { source, target: t, amount } => {
                assert_eq!(source, caster());
                assert_eq!(t, target());
                assert_eq!(amount, 42, "amount round-trips from EffectOp");
            }
            other => panic!("expected TransferGold, got {other:?}"),
        }
    }

    #[test]
    fn transfer_gold_preserves_negative_amount() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::TransferGold { amount: -7 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        match events[0] {
            ApplyEvent::TransferGold { amount, .. } =>
                assert_eq!(amount, -7, "negative amount preserved (sign isn't lost)"),
            other => panic!("expected TransferGold, got {other:?}"),
        }
    }

    #[test]
    fn modify_standing_emits_apply_event_with_delta() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::ModifyStanding { delta: -25 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 1, "ModifyStanding emits exactly one ApplyEvent");
        match events[0] {
            ApplyEvent::ModifyStanding { source, target: t, delta } => {
                assert_eq!(source, caster());
                assert_eq!(t, target());
                assert_eq!(delta, -25, "delta round-trips from EffectOp (sign preserved)");
            }
            other => panic!("expected ModifyStanding, got {other:?}"),
        }
    }

    // -- Caster-stat scaling --------------------------------------------------

    #[test]
    fn apply_strike_with_attack_damage_scaling_adds_to_amount() {
        // Damage 30 + 50% AD; caster has 100 AD ⇒ emit 30 + 50 = 80.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.scalings_per_effect.push(smallvec![EffectScaling {
            stat_ref: ScalingStatRef::AttackDamage,
            percent:  0.50,
        }]);
        let stats = CasterStats { attack_damage: 100.0, ..Default::default() };
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &stats, &CasterStats::default());
        assert_eq!(events.len(), 1);
        match events[0] {
            ApplyEvent::Damage { amount, .. } => {
                assert!((amount - 80.0).abs() < 1e-5, "expected 80.0, got {amount}");
            }
            other => panic!("expected Damage, got {other:?}"),
        }
    }

    #[test]
    fn apply_skipped_effect_doesnt_scale() {
        // chance=0 gates the effect out — no event emitted, scaling math
        // must not run (and certainly must not produce a side-effect).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.chances.push(Some(0));
        prog.scalings_per_effect.push(smallvec![EffectScaling {
            stat_ref: ScalingStatRef::AttackDamage,
            percent:  10.0, // would be huge if it ran
        }]);
        let stats = CasterStats { attack_damage: 1000.0, ..Default::default() };
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &stats, &CasterStats::default());
        assert_eq!(events.len(), 0, "chance=0 must gate the effect out before scaling");
    }

    #[test]
    fn apply_no_scaling_is_bit_stable() {
        // Empty `scalings_per_effect` ⇒ output identical to the
        // pre-scaling apply path (regression guard for the B slice).
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 30.0 },
                EffectOp::Heal   { amount: 12.5 },
                EffectOp::Shield { amount:  7.0 },
            ],
        );
        // Even with massive caster stats, an empty scalings vec must
        // contribute zero — output is bit-identical to default-stats.
        let stats = CasterStats {
            attack_damage: 9999.0,
            ability_power: 9999.0,
            ..Default::default()
        };
        let with    = apply_program(&prog, caster(), target(), 0, 0xCAFE, &stats, &CasterStats::default());
        let without = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(with.len(), without.len());
        for (a, b) in with.iter().zip(without.iter()) {
            assert_eq!(a, b, "with-stats vs default-stats diverged with no scalings");
        }
        // Spot-check the absolute values.
        assert!(matches!(with[0], ApplyEvent::Damage { amount, .. } if amount == 30.0));
        assert!(matches!(with[1], ApplyEvent::Heal   { amount, .. } if amount == 12.5));
        assert!(matches!(with[2], ApplyEvent::Shield { amount, .. } if amount == 7.0));
    }

    #[test]
    fn apply_summon_emits_summon_event() {
        // Verify the new EffectOp::Summon arm produces an ApplyEvent::Summon
        // with the template_hash threaded through.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
            [EffectOp::Summon { template_hash: 0xDEADBEEF, count: 3, lifetime_ticks: 80 }],
        );
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            ApplyEvent::Summon { source, template_hash, count, lifetime_ticks }
            if source == caster() && template_hash == 0xDEADBEEF && count == 3 && lifetime_ticks == 80
        ));
    }

    // -- Wave 1.5#9 nested-effect dispatch ----------------------------------

    #[test]
    fn nested_stun_on_damage_emits_two_events_in_order() {
        // Reap-shape: primary Damage + nested `{ stun 1s }`. apply_program
        // emits Damage first, then Stun, both targeting the same target.
        // Closes the gap surfaced by the Reap verb swap (commit
        // `72a35307`).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 20.0 }],
        );
        prog.nested_per_effect.push(smallvec![EffectOp::Stun { duration_ticks: 10 }]);
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 2, "primary + nested → 2 ApplyEvents");
        assert!(
            matches!(events[0], ApplyEvent::Damage { source, target: t, amount }
                if source == caster() && t == target() && amount == 20.0),
            "first event must be the primary Damage",
        );
        assert!(
            matches!(events[1], ApplyEvent::Stun { target: t, duration_ticks }
                if t == target() && duration_ticks == 10),
            "second event must be the nested Stun targeting same target",
        );
    }

    #[test]
    fn nested_two_ops_emit_in_declaration_order() {
        // MAX_NESTED_PER_EFFECT == 2 — both inner ops fire after the
        // primary, in the order they appear in the source.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.nested_per_effect.push(smallvec![
            EffectOp::Stun { duration_ticks: 5 },
            EffectOp::Slow { duration_ticks: 20, factor_q8: -64 },
        ]);
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 3, "primary + 2 nested → 3 ApplyEvents");
        assert!(matches!(events[0], ApplyEvent::Damage { .. }));
        assert!(matches!(events[1], ApplyEvent::Stun { duration_ticks: 5, .. }));
        assert!(matches!(events[2], ApplyEvent::Slow { duration_ticks: 20, factor_q8: -64, .. }));
    }

    #[test]
    fn empty_nested_slot_emits_only_primary() {
        // Outer slice populated but inner slot empty → only the primary
        // event fires (back-compat with abilities whose source has no
        // `{ ... }` block).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        );
        prog.nested_per_effect.push(smallvec![]); // empty inner — no nested op
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 1, "empty nested slot → only primary fires");
        assert!(matches!(events[0], ApplyEvent::Damage { .. }));
    }

    #[test]
    fn chance_gate_skips_primary_and_nested() {
        // The chance gate checks the OUTER slot — when it fails, the
        // primary AND nested are skipped together. This matches the
        // "auxiliary effect riding on the primary" semantic: if the
        // primary doesn't fire, neither does the nested.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        );
        prog.chances.push(Some(0)); // gate-out the primary
        prog.nested_per_effect.push(smallvec![EffectOp::Stun { duration_ticks: 10 }]);
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &CasterStats::default(), &CasterStats::default());
        assert_eq!(events.len(), 0, "chance=0 must skip both primary and nested");
    }

    #[test]
    fn nested_op_does_not_carry_scaling_bonus() {
        // The slot's scaling applies only to the primary effect's
        // amount; nested ops emit with `scale_bonus = 0.0`. Pin this
        // because nested ops don't have their own scaling slot in the
        // registry today (inner-stmt modifiers dropped at lowering).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        // Outer slot scales: +50% AD on the Damage.
        prog.scalings_per_effect.push(smallvec![EffectScaling {
            stat_ref: ScalingStatRef::AttackDamage,
            percent:  0.50,
        }]);
        // Nested Damage 5 — should emit with amount=5.0, NOT 5.0+50=55.
        prog.nested_per_effect.push(smallvec![EffectOp::Damage { amount: 5.0 }]);
        let stats = CasterStats { attack_damage: 100.0, ..Default::default() };
        let events = apply_program(&prog, caster(), target(), 0, 0xCAFE, &stats, &CasterStats::default());
        assert_eq!(events.len(), 2);
        // Primary scaled.
        assert!(matches!(events[0], ApplyEvent::Damage { amount, .. } if (amount - 80.0).abs() < 1e-5),
            "primary amount must include scaling bonus (30 + 50 = 80), got {events:?}");
        // Nested NOT scaled.
        assert!(matches!(events[1], ApplyEvent::Damage { amount, .. } if (amount - 5.0).abs() < 1e-5),
            "nested amount must NOT carry the slot's scaling bonus (5.0 expected), got {events:?}");
    }

    // -- Wave 1.5#7 when-predicate gate ------------------------------------

    /// Helper: build a one-effect program with a when-predicate.
    fn prog_with_when(op: EffectOp, pred: EffectPredicate) -> AbilityProgram {
        let mut p = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [op],
        );
        p.when_per_effect.push(Some(EffectWhenCondition {
            when_cond:     "<test>".to_string(),
            else_cond:     None,
            when_compiled: Some(pred),
        }));
        p
    }

    #[test]
    fn apply_program_with_when_predicate_true_fires_effect() {
        // when target.hp < 20 ; target.hp = 5 → fires
        let pred = EffectPredicate {
            binder:  EffectPredicateBinder::Target,
            field:   ScalingStatRef::Hp.discriminant(),
            op:      EffectPredicateOp::Lt,
            literal: 20.0,
        };
        let prog = prog_with_when(EffectOp::Damage { amount: 50.0 }, pred);
        let target_stats = CasterStats { hp: 5.0, ..Default::default() };
        let events = apply_program(
            &prog, caster(), target(), 0, 0xCAFE,
            &CasterStats::default(), &target_stats,
        );
        assert_eq!(events.len(), 1, "predicate true must fire effect");
        assert!(matches!(events[0], ApplyEvent::Damage { amount, .. } if amount == 50.0));
    }

    #[test]
    fn apply_program_with_when_predicate_false_skips_effect() {
        // when target.hp < 20 ; target.hp = 50 → skip
        let pred = EffectPredicate {
            binder:  EffectPredicateBinder::Target,
            field:   ScalingStatRef::Hp.discriminant(),
            op:      EffectPredicateOp::Lt,
            literal: 20.0,
        };
        let prog = prog_with_when(EffectOp::Damage { amount: 50.0 }, pred);
        let target_stats = CasterStats { hp: 50.0, ..Default::default() };
        let events = apply_program(
            &prog, caster(), target(), 0, 0xCAFE,
            &CasterStats::default(), &target_stats,
        );
        assert_eq!(events.len(), 0, "predicate false must skip effect");
    }

    #[test]
    fn apply_program_when_target_hp_lt_20_executes_correctly() {
        // Reap-shape: when target.hp < 20, execute(20). Verifies the
        // semantic that drives the duel_abilities Reap registry-driven
        // gate (the .sim verb's redundant gate is dropped in this slice).
        let pred = EffectPredicate {
            binder:  EffectPredicateBinder::Target,
            field:   ScalingStatRef::Hp.discriminant(),
            op:      EffectPredicateOp::Lt,
            literal: 20.0,
        };
        let prog = prog_with_when(
            EffectOp::Execute { hp_threshold: 20.0 },
            pred,
        );
        // Below threshold → Execute fires.
        let low = CasterStats { hp: 10.0, ..Default::default() };
        let evs_low = apply_program(
            &prog, caster(), target(), 0, 0xCAFE,
            &CasterStats::default(), &low,
        );
        assert_eq!(evs_low.len(), 1, "hp=10 < 20 → Execute fires");
        assert!(matches!(evs_low[0], ApplyEvent::Execute { .. }));

        // At threshold (Lt is strict) → Execute skipped.
        let at = CasterStats { hp: 20.0, ..Default::default() };
        let evs_at = apply_program(
            &prog, caster(), target(), 0, 0xCAFE,
            &CasterStats::default(), &at,
        );
        assert_eq!(evs_at.len(), 0, "hp=20 NOT < 20 → Execute skipped");
    }

    #[test]
    fn apply_program_when_predicate_false_skips_nested_too() {
        // When the primary's predicate fails, BOTH primary and nested
        // ops are skipped — auxiliary nested effects ride on the
        // primary's success (mirrors the chance-gate semantic).
        let pred = EffectPredicate {
            binder:  EffectPredicateBinder::Target,
            field:   ScalingStatRef::Hp.discriminant(),
            op:      EffectPredicateOp::Lt,
            literal: 20.0,
        };
        let mut prog = prog_with_when(EffectOp::Damage { amount: 50.0 }, pred);
        prog.nested_per_effect.push(smallvec![EffectOp::Stun { duration_ticks: 10 }]);
        let target_stats = CasterStats { hp: 50.0, ..Default::default() };
        let events = apply_program(
            &prog, caster(), target(), 0, 0xCAFE,
            &CasterStats::default(), &target_stats,
        );
        assert_eq!(events.len(), 0, "predicate false must skip primary AND nested");
    }

    #[test]
    fn apply_program_when_self_binder_reads_caster_stats() {
        // when self.hp < 50 ; caster.hp = 30 → fires; caster.hp = 60 → skipped.
        let pred = EffectPredicate {
            binder:  EffectPredicateBinder::SelfBinder,
            field:   ScalingStatRef::Hp.discriminant(),
            op:      EffectPredicateOp::Lt,
            literal: 50.0,
        };
        let prog = prog_with_when(EffectOp::Heal { amount: 25.0 }, pred);
        let low = CasterStats { hp: 30.0, ..Default::default() };
        let evs_low = apply_program(
            &prog, caster(), target(), 0, 0xCAFE, &low, &CasterStats::default(),
        );
        assert_eq!(evs_low.len(), 1, "self.hp=30 < 50 → fires");
        let high = CasterStats { hp: 60.0, ..Default::default() };
        let evs_high = apply_program(
            &prog, caster(), target(), 0, 0xCAFE, &high, &CasterStats::default(),
        );
        assert_eq!(evs_high.len(), 0, "self.hp=60 NOT < 50 → skipped");
    }

    // -- Task #121 (Path A): multi-target AOE behavioral E2E ---------------

    /// Helpers for the AOE behavioral fixture. Three agents in a row;
    /// the AOE pre-query returns ids 2 and 3 (the targets within the
    /// caster's circle), sorted ascending.
    fn agent_n(n: u32) -> AgentId { AgentId::new(n).unwrap() }

    #[test]
    fn aoe_circle_expands_across_multiple_targets() {
        // Strike-shape: damage 30 in circle(0.5). Caster id=1, primary
        // target id=2; aoe_targets = [2, 3] (both inside the circle —
        // the spatial query is the caller's responsibility; here we
        // simulate its output directly). Expect 2 ApplyEvents (one
        // Damage per target) — pinning the multi-target AOE expansion.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Circle,
            args: [0.5, 0.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),                 // caster
            agent_n(2),                 // primary target (irrelevant for Circle slot)
            &aoe_targets,
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 2, "Circle AOE → one event per aoe_target");
        // Sorted-by-AgentId-ascending preservation: emitted events
        // mirror the input slice order (P11 — caller pre-sorts the
        // spatial query, dispatch preserves the order).
        assert!(matches!(
            events[0],
            ApplyEvent::Damage { source, target, amount }
            if source == agent_n(1) && target == agent_n(2) && amount == 30.0
        ), "first event must target lowest AgentId, got {events:?}");
        assert!(matches!(
            events[1],
            ApplyEvent::Damage { source, target, amount }
            if source == agent_n(1) && target == agent_n(3) && amount == 30.0
        ), "second event must target next AgentId, got {events:?}");
    }

    #[test]
    fn aoe_circle_with_three_targets_emits_three_records() {
        // ≥3 agents in a row: caster id=1; AOE targets ids [2, 3, 4].
        // Pins the "≥3 chronicle records produced from 1 cast"
        // requirement of the task's behavioral E2E.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Circle,
            args: [2.0, 0.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3), agent_n(4)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &aoe_targets,
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 3, "3 targets in AOE → 3 chronicle records from 1 cast");
        for (i, target) in [agent_n(2), agent_n(3), agent_n(4)].iter().enumerate() {
            assert!(matches!(
                events[i],
                ApplyEvent::Damage { target: t, amount, .. }
                if t == *target && amount == 30.0
            ), "event {i} must target {target:?}, got {events:?}");
        }
    }

    #[test]
    fn aoe_non_expandable_shape_falls_back_to_single_target() {
        // Line (and every shape other than Circle/Cone) is deferred —
        // the slot fires once on `primary_target` even when
        // `aoe_targets` is populated. Pin this so the deferral is
        // explicit (regression guard for future shape additions).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Line,
            args: [5.0, 1.0, 0.0, 0.0],
        }));
        // aoe_targets is non-empty, but Line isn't expanded yet.
        let aoe_targets = [agent_n(2), agent_n(3), agent_n(4)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &aoe_targets,
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 1, "Line shape → single-target fallback (Path A defers)");
        assert!(matches!(
            events[0],
            ApplyEvent::Damage { target: t, .. } if t == agent_n(2)
        ));
    }

    #[test]
    fn aoe_cone_expands_across_pre_filtered_targets() {
        // Cone shape: same expansion contract as Circle — the caller
        // pre-filters and passes the in-cone set, the dispatcher emits
        // one ApplyEvent per target. The cone math itself is exercised
        // by `aoe_cone_filter_*` tests below; this test pins the
        // dispatch leg.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 25.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Cone,
            args: [60.0, 4.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &aoe_targets,
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 2, "Cone AOE → one event per pre-filtered target");
        assert!(matches!(
            events[0],
            ApplyEvent::Damage { target: t, .. } if t == agent_n(2)
        ));
        assert!(matches!(
            events[1],
            ApplyEvent::Damage { target: t, .. } if t == agent_n(3)
        ));
    }

    #[test]
    fn aoe_cone_filter_5_agents_in_arc() {
        // Fan layout: agent 0 (caster) at origin, agents 1-4 in an arc
        // in front of agent 0 facing +X.
        //
        //   agent 4 (out-of-cone — wide angle):  pos = (1, 5, 0)        ~78.7° off axis
        //   agent 3 (in-cone, top):               pos = (3, 1, 0)       ~18.4° off axis
        //   agent 2 (in-cone, on-axis target):    pos = (4, 0, 0)        0° off axis
        //   agent 1 (in-cone, bottom):            pos = (3, -1, 0)      ~18.4° off axis
        //   agent 5 (out-of-cone — past range):   pos = (10, 0, 0)      0° off axis but range>5
        //
        // cone(60°, 5) from agent 0 facing agent 2 hits {1, 2, 3} but
        // not 4 (outside angle) or 5 (past range). agent 0 (the caster
        // itself, at apex) is excluded.
        let apex = glam::Vec3::new(0.0, 0.0, 0.0);
        let target_pos = glam::Vec3::new(4.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), apex),                              // caster — apex exclusion
            (agent_n(2), glam::Vec3::new(3.0, -1.0, 0.0)),   // in-cone bottom
            (agent_n(3), target_pos),                         // on-axis target
            (agent_n(4), glam::Vec3::new(3.0, 1.0, 0.0)),    // in-cone top
            (agent_n(5), glam::Vec3::new(1.0, 5.0, 0.0)),    // out-of-cone (wide)
            (agent_n(6), glam::Vec3::new(10.0, 0.0, 0.0)),   // out-of-cone (range)
        ];
        let hits = apply_program_aoe_cone_filter(
            apex,
            target_pos,
            /*half_angle_deg*/ 60.0,
            /*range*/ 5.0,
            &candidates,
        );
        assert_eq!(
            hits,
            vec![agent_n(2), agent_n(3), agent_n(4)],
            "cone(60°, 5) facing +X must hit agents 2/3/4 only \
             (caster apex excluded, agent 5 outside angle, agent 6 \
             past range); got {hits:?}",
        );
    }

    #[test]
    fn aoe_cone_filter_caster_targets_self_returns_empty() {
        // Edge case: caster targets self ⇒ direction = (0,0,0) is
        // degenerate. CPU oracle returns empty; GPU's `dir_len_sq <
        // 1e-6` branch matches by going to no-op. Pinning this so a
        // future "caster auto-faces nearest enemy on degenerate
        // direction" change doesn't silently flip CPU↔GPU parity.
        let apex = glam::Vec3::new(2.0, 3.0, 4.0);
        let target_pos = apex;
        let candidates = vec![
            (agent_n(2), glam::Vec3::new(3.0, 3.0, 4.0)),
            (agent_n(3), glam::Vec3::new(2.0, 4.0, 4.0)),
        ];
        let hits = apply_program_aoe_cone_filter(
            apex, target_pos, 60.0, 5.0, &candidates,
        );
        assert!(
            hits.is_empty(),
            "degenerate cone (caster targets self) → empty in-cone set, got {hits:?}",
        );
    }

    #[test]
    fn aoe_cone_filter_apex_exclusion() {
        // Two candidates at the apex (e.g. caster + a co-located ally
        // at the same world coord). Both must be excluded from the
        // in-cone set — the GPU kernel's `dist_sq < 1e-6` branch skips
        // them, and the CPU oracle matches.
        let apex = glam::Vec3::new(0.0, 0.0, 0.0);
        let target_pos = glam::Vec3::new(5.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), apex),                              // caster at apex
            (agent_n(2), apex),                              // co-located ally at apex
            (agent_n(3), glam::Vec3::new(2.0, 0.0, 0.0)),    // in-cone, NOT at apex
        ];
        let hits = apply_program_aoe_cone_filter(
            apex, target_pos, 60.0, 5.0, &candidates,
        );
        assert_eq!(
            hits,
            vec![agent_n(3)],
            "apex-coincident candidates must be excluded; got {hits:?}",
        );
    }

    #[test]
    fn aoe_no_shape_slot_is_single_target() {
        // No per_effect_areas slot at all — single-target by default,
        // identical to apply_program. aoe_targets is ignored.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 25.0 }],
        );
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &aoe_targets,
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 1, "no shape slot → single-target");
        assert!(matches!(
            events[0],
            ApplyEvent::Damage { target: t, amount, .. }
            if t == agent_n(2) && amount == 25.0
        ));
    }

    #[test]
    fn aoe_circle_with_empty_target_list_emits_nothing() {
        // Caller's spatial query found no targets in the circle —
        // the slot fires zero times. Important guard: empty AOE must
        // not panic, must not fall back to primary_target (the AOE
        // slot's contract is "only the spatial-query result hits";
        // primary_target is for non-Circle slots only).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Circle,
            args: [0.5, 0.0, 0.0, 0.0],
        }));
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &[],
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 0, "empty AOE target list → zero events");
    }

    #[test]
    fn aoe_chance_gate_is_all_or_nothing() {
        // Chance gate is keyed by slot, NOT per target — so when it
        // fails, all aoe_targets are skipped together; when it fires,
        // all are hit. Pins the "AOE proc'd or didn't" semantic.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.chances.push(Some(0)); // gate-out
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Circle,
            args: [0.5, 0.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3), agent_n(4)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &aoe_targets,
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 0, "chance=0 must skip ALL AOE targets");
    }

    #[test]
    fn aoe_nested_op_fires_per_target() {
        // damage 30 in circle, with nested { stun 1s }. Each target
        // in the AOE receives the Damage AND a Stun (nested ops ride
        // on each target the primary hits).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Circle,
            args: [0.5, 0.0, 0.0, 0.0],
        }));
        prog.nested_per_effect.push(smallvec![EffectOp::Stun { duration_ticks: 10 }]);
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &aoe_targets,
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        // 2 targets × (1 primary + 1 nested) = 4 events, in
        // (primary_t1, nested_t1, primary_t2, nested_t2) order.
        assert_eq!(events.len(), 4, "2 targets × (Damage + Stun) → 4 events");
        assert!(matches!(events[0], ApplyEvent::Damage { target: t, .. } if t == agent_n(2)));
        assert!(matches!(events[1], ApplyEvent::Stun   { target: t, .. } if t == agent_n(2)));
        assert!(matches!(events[2], ApplyEvent::Damage { target: t, .. } if t == agent_n(3)));
        assert!(matches!(events[3], ApplyEvent::Stun   { target: t, .. } if t == agent_n(3)));
    }

    #[test]
    fn aoe_scaling_applies_to_each_target() {
        // damage 30 + 50% AD with caster.attack_damage = 100 → each
        // target receives 80.0 (= 30 + 50). Scaling is computed once
        // from caster_stats and applied uniformly across the AOE
        // expansion (same caster, same scaling).
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Circle,
            args: [0.5, 0.0, 0.0, 0.0],
        }));
        prog.scalings_per_effect.push(smallvec![EffectScaling {
            stat_ref: ScalingStatRef::AttackDamage,
            percent:  0.50,
        }]);
        let stats = CasterStats { attack_damage: 100.0, ..Default::default() };
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &aoe_targets,
            0, 0xCAFE,
            &stats,
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 2);
        for ev in &events {
            assert!(matches!(*ev, ApplyEvent::Damage { amount, .. }
                if (amount - 80.0).abs() < 1e-5),
                "each AOE target must get scaled amount 80.0, got {ev:?}");
        }
    }
}
