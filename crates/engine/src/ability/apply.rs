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
    /// Wave 3 ToM Phase 1 bit-flag belief primitive. Caster causes
    /// `target`'s belief map for `subject_idx` to gain
    /// `1u << fact_bit` via atomic-OR fold. The chronicle dispatcher
    /// writes a record (kind=63) carrying caster + target +
    /// subject_idx + fact_bit_mask; downstream view consumers
    /// (`view <name>(target: Agent, subject: Agent) -> u32 { on
    /// EffectPlantBeliefApplied { ... } { self |= b } }`) fold the
    /// mask into the pair_map cell. Pairs with
    /// `EventKindId::EffectPlantBeliefApplied = 63` on the chronicle
    /// side. The full Wave 3 multi-field BeliefState (creature_type
    /// / decay phase / disguise verbs / slander cascade) is deferred.
    PlantBelief    { source: AgentId, target: AgentId, subject_idx: u32, fact_bit: u8 },
    /// Wave 3 ToM Phase 3 — `observe` self-observe-target verb. Caster
    /// refreshes its own belief row about `target`: the consumer reads
    /// target's CURRENT pos / creature_type from the agent SoA at
    /// consume tick and writes into the BeliefState SoA's 6 columns at
    /// `[caster_slot * agent_cap + target_slot]`. Pairs with
    /// `EventKindId::EffectObserveApplied = 64` on the chronicle side.
    /// The `target_observer` byte is a future-extension hook (only the
    /// self-observe shape `0` is wired today).
    Observe        { source: AgentId, target: AgentId, target_observer: u8 },
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
        // Wave 3 ToM Phase 1 — `plant_belief` bit-flag primitive. The
        // dispatcher records the cast as a chronicle event (kind=63);
        // the actual atomic-OR write into the pair_map cell happens in
        // a downstream view consumer (the existing `tom_probe.sim`
        // fold-body shape: `on EffectPlantBeliefApplied { ... } { self
        // |= b }`). Same separation of concerns as Damage/Heal/etc.,
        // where ApplyEvent emission is the cast record and the
        // world-state mutation lives in the cascade consumer.
        EffectOp::PlantBelief { subject_idx, fact_bit } =>
            out.push(ApplyEvent::PlantBelief { source: caster, target, subject_idx, fact_bit }),
        // Wave 3 ToM Phase 3 — `observe` self-observe-target verb. The
        // dispatcher records the cast as a chronicle event (kind=64);
        // the actual writeback into the BeliefState SoA's 6 columns
        // happens in a downstream runtime consumer that reads target's
        // current pos / creature_type from the agent SoA at consume
        // tick. Same separation of concerns as PlantBelief, where
        // ApplyEvent emission is the cast record and the world-state
        // mutation lives in the cascade consumer.
        EffectOp::Observe { target_observer } =>
            out.push(ApplyEvent::Observe { source: caster, target, target_observer }),
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
///     kind: Circle | Cone | Box | Sphere | Ring | Line, .. })`, the
///     slot fires once per `aoe_targets` entry — one ApplyEvent per
///     target. The caller is responsible for performing the spatial +
///     geometric filter:
///     - Circle / Sphere: `state.spatial().within_radius(state,
///       target_pos, args[0])` — see `apply_program_aoe_sphere_filter`
///       (Sphere is mathematically equivalent to Circle; see #180);
///     - Cone: range² gate ∧ angular gate (#178), see
///       `apply_program_aoe_cone_filter`;
///     - Box: per-axis `|d.<axis>| <= w<axis>` AABB containment (#179),
///       see `apply_program_aoe_box_filter`;
///     - Ring (#180): annulus gate `inner² <= dist² <= outer²`, see
///       `apply_program_aoe_ring_filter`;
///     - Line (#180): forward rectangle along `normalize(target -
///       caster)`, see `apply_program_aoe_line_filter`.
///     The slice MUST be sorted ascending by raw `AgentId` (P11 —
///     `SpatialHash::within_radius` does this by construction; every
///     filter helper sorts as its final step).
///   * Any other `Some(...)` shape (Spread/Column/Wall/Cylinder/Dome/
///     Hull) — deferred. The slot falls back to single-target dispatch
///     on `primary_target`. The remaining shapes need additional
///     geometry kernels (capsule, vertical column, etc.) which Path B
///     defers.
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

        // Choose target list for this slot. Circle, Cone, Box, Sphere,
        // Ring, and Line expand across aoe_targets (caller pre-filters
        // — Circle/Sphere by `within_radius`, Cone by
        // `apply_program_aoe_cone_filter`, Box by
        // `apply_program_aoe_box_filter`, Ring by
        // `apply_program_aoe_ring_filter`, Line by
        // `apply_program_aoe_line_filter`); everything else (None, or
        // unrecognised shape) is single-target on primary_target.
        let is_aoe_shape = matches!(
            program.per_effect_areas.get(i).copied().flatten(),
            Some(shape) if shape.kind == ShapeKind::Circle
                        || shape.kind == ShapeKind::Cone
                        || shape.kind == ShapeKind::Box
                        || shape.kind == ShapeKind::Sphere
                        || shape.kind == ShapeKind::Ring
                        || shape.kind == ShapeKind::Line
                        // #181 AOE Path B remaining shapes: Spread, Column,
                        // Wall, Cylinder, Dome, Hull. Hull is a Sphere alias
                        // (no separate filter helper today; see
                        // `apply_program_aoe_sphere_filter` doc).
                        || shape.kind == ShapeKind::Spread
                        || shape.kind == ShapeKind::Column
                        || shape.kind == ShapeKind::Wall
                        || shape.kind == ShapeKind::Cylinder
                        || shape.kind == ShapeKind::Dome
                        || shape.kind == ShapeKind::Hull
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

/// CPU-side box-filter oracle for AOE Path B Box (mirrors the GPU
/// kernel's WGSL math bit-for-bit so callers feeding `apply_program_aoe`
/// produce a chronicle record set byte-equal to the GPU dispatcher).
///
/// Inputs:
///   * `center` — explicit cast target's position (same convention as
///     Circle: `aoe_center = agent_pos[target_slot]`).
///   * `wx`, `wy`, `wz` — `args[0..=2]` from the `EffectAreaShape` (box
///     half-extents along each world axis). `args[3]` is unused.
///   * `candidates` — slice of `(AgentId, position)` pairs the caller
///     pre-collected from the spatial grid (the GPU walks 27 cells
///     around `center`; the CPU helper accepts any superset and filters
///     down).
///
/// Output: in-box `AgentId`s sorted ascending by raw id (P11).
///
/// In-box predicate (per candidate, identical to the WGSL kernel):
///   1. `abs(cand.x - center.x) <= wx`
///   2. `abs(cand.y - center.y) <= wy`
///   3. `abs(cand.z - center.z) <= wz`
///
/// Edge case: any half-extent of 0 collapses that axis to a strict
/// equality (only candidates exactly at `center.<axis>` on that axis
/// match). Same closed-AABB semantic as Circle's `<=` on radius² —
/// candidates exactly at the wall are inside.
///
/// **Spatial walk limitation.** The GPU dispatcher walks the 27-cell
/// neighborhood around the center; if any of `wx`, `wy`, `wz` exceed
/// the spatial cell size (`SPATIAL_CELL_SIZE = 6.0`), candidates beyond
/// the 27-cell ring are missed by the GPU walk. The CPU helper does
/// not impose this constraint — callers are responsible for sizing the
/// candidate superset accordingly (typical caller passes
/// `state.spatial().within_radius(state, center, max(wx, wy, wz) * √3)`
/// or similar). Tests + parity sweeps must keep extents ≤ cell size to
/// stay byte-equal across backends.
///
/// **P11.** Both backends evaluate the predicate identically (three
/// abs/sub/cmp ops in the same order). The final sort makes the
/// post-filter set deterministic — GPU's atomic ring claim doesn't
/// preserve order, but the parity sweep sorts post-readback
/// (canonicalize) and the sets agree.
pub fn apply_program_aoe_box_filter(
    center:     glam::Vec3,
    wx:         f32,
    wy:         f32,
    wz:         f32,
    candidates: &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let dvec = cand_pos - center;
        if dvec.x.abs() <= wx && dvec.y.abs() <= wy && dvec.z.abs() <= wz {
            hits.push(id);
        }
    }
    // P11 reduction-determinism: sort ascending by raw AgentId so the
    // CPU oracle and GPU dispatcher (after canonicalize) agree on
    // emit order.
    hits.sort_by_key(|id| id.raw());
    hits
}

/// CPU-side sphere-filter oracle for AOE Path B Sphere (mirrors the GPU
/// kernel's WGSL math bit-for-bit). Sphere is mathematically equivalent
/// to Circle today (3D distance check, `dot(d, d) <= radius²`); the
/// separate filter exists for code clarity + so callers can pass the
/// shape's args slot without a Circle/Sphere alias mapping.
///
/// Inputs:
///   * `center` — explicit cast target's position (same convention as
///     Circle: `aoe_center = agent_pos[target_slot]`).
///   * `radius` — `args[0]` from the `EffectAreaShape` (sphere radius).
///   * `candidates` — slice of `(AgentId, position)` pairs the caller
///     pre-collected from the spatial grid.
///
/// Output: in-sphere `AgentId`s sorted ascending by raw id (P11).
///
/// **Equivalence with Circle.** The Circle/Sphere split is a contract
/// on the shape kind only — both compute `dist² <= radius²` over a 3D
/// position. A future divergence (e.g. a flat-disk Circle vs a true 3D
/// Sphere) would update both filters; today they share semantics. The
/// GPU branch comment documents the equivalence the same way.
pub fn apply_program_aoe_sphere_filter(
    center:     glam::Vec3,
    radius:     f32,
    candidates: &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    let radius_sq = radius * radius;
    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let dvec = cand_pos - center;
        if dvec.dot(dvec) <= radius_sq {
            hits.push(id);
        }
    }
    // P11 reduction-determinism: sort ascending by raw AgentId.
    hits.sort_by_key(|id| id.raw());
    hits
}

/// CPU-side ring-filter oracle for AOE Path B Ring (mirrors the GPU
/// kernel's WGSL math bit-for-bit so callers feeding `apply_program_aoe`
/// produce a chronicle record set byte-equal to the GPU dispatcher).
///
/// Inputs:
///   * `center` — explicit cast target's position (same convention as
///     Circle).
///   * `inner_radius` — `args[0]` (inner edge of the annulus).
///   * `outer_radius` — `args[1]` (outer edge of the annulus).
///   * `candidates` — slice of `(AgentId, position)` pairs the caller
///     pre-collected from the spatial grid.
///
/// Output: in-ring `AgentId`s sorted ascending by raw id (P11).
///
/// In-ring predicate (per candidate, identical to the WGSL kernel):
///   `inner² <= dist² <= outer²` where `dist² = dot(cand-center,
///   cand-center)`. Closed on both edges (≤ semantic on both bounds —
///   candidates exactly at either wall are in-ring).
///
/// **Edge case: `inner_radius > outer_radius`.** The bounds invert, so
/// the predicate `inner² <= dist² <= outer²` can never be satisfied
/// (lhs > rhs). Result: empty in-ring set. Both backends agree.
///
/// **Spatial walk limitation.** The GPU dispatcher walks 27 cells
/// around the center; if `outer_radius > SPATIAL_CELL_SIZE`, candidates
/// beyond the 27-cell ring are missed. Same caveat as Circle/Sphere/
/// Box (#179) — fixtures must keep the outer radius ≤ cell size to
/// stay byte-equal across backends.
pub fn apply_program_aoe_ring_filter(
    center:       glam::Vec3,
    inner_radius: f32,
    outer_radius: f32,
    candidates:   &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    let inner_sq = inner_radius * inner_radius;
    let outer_sq = outer_radius * outer_radius;
    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let dvec = cand_pos - center;
        let dist_sq = dvec.dot(dvec);
        if dist_sq >= inner_sq && dist_sq <= outer_sq {
            hits.push(id);
        }
    }
    // P11 reduction-determinism.
    hits.sort_by_key(|id| id.raw());
    hits
}

/// CPU-side line-filter oracle for AOE Path B Line (mirrors the GPU
/// kernel's WGSL math bit-for-bit so callers feeding `apply_program_aoe`
/// produce a chronicle record set byte-equal to the GPU dispatcher).
///
/// Inputs:
///   * `apex` — caster's position (line origin; the rectangle starts
///     here and extends `length` along `direction`).
///   * `target_pos` — explicit cast target's position. The line faces
///     `direction = normalize(target_pos - apex)`.
///   * `length` — `args[0]` (rectangle length along the direction).
///   * `width` — `args[1]` (rectangle full width perpendicular to the
///     direction; the gate uses half-width = width/2).
///   * `candidates` — slice of `(AgentId, position)` pairs the caller
///     pre-collected from the spatial grid.
///
/// Output: in-line `AgentId`s sorted ascending by raw id (P11).
///
/// In-line predicate (per candidate, identical to the WGSL kernel):
///   1. Let `to_cand = cand_pos - apex`,
///      `along = dot(to_cand, direction)` (signed distance along axis).
///   2. `along >= 0` (in front of caster).
///   3. `along <= length` (within length).
///   4. `perp_sq = dot(to_cand, to_cand) - along*along <= (width/2)²`
///      (within half-width perpendicular). Pythagoras avoids a 3D
///      cross-product, matching the GPU kernel.
///
/// Edge case: if `target_pos == apex` (caster targets self), the
/// line's direction is degenerate (zero-vector). Both backends MUST
/// return an empty set (the line is undefined; emitting any subset
/// would create CPU↔GPU drift). The CPU short-circuits here; the
/// GPU's `dir_len_sq < 1e-6` branch matches.
///
/// **Spatial walk limitation.** The GPU dispatcher walks 27 cells
/// around the apex; if `length > SPATIAL_CELL_SIZE` or the line
/// extends past the 27-cell ring, candidates beyond are missed. Same
/// caveat as Cone (#178) — fixtures must keep `length ≤ cell size` to
/// stay byte-equal.
///
/// **P11.** Both backends evaluate the predicate with identical f32
/// op order (subtract, dot, inverseSqrt, dot, sub, compare). The final
/// sort makes the post-filter set deterministic.
pub fn apply_program_aoe_line_filter(
    apex:       glam::Vec3,
    target_pos: glam::Vec3,
    length:     f32,
    width:      f32,
    candidates: &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    // Degenerate line — caster targets self. Direction is undefined;
    // emit no targets to match the GPU kernel.
    let direction_raw = target_pos - apex;
    let dir_len_sq = direction_raw.dot(direction_raw);
    if dir_len_sq < 1e-6 {
        return Vec::new();
    }
    let direction = direction_raw * dir_len_sq.recip().sqrt();
    let half_width = width * 0.5;
    let half_width_sq = half_width * half_width;

    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let to_cand = cand_pos - apex;
        let along = to_cand.dot(direction);
        if along < 0.0 || along > length {
            continue;
        }
        let dist_sq = to_cand.dot(to_cand);
        let perp_sq = dist_sq - along * along;
        if perp_sq > half_width_sq {
            continue;
        }
        hits.push(id);
    }
    // P11 reduction-determinism.
    hits.sort_by_key(|id| id.raw());
    hits
}

/// CPU-side spread-filter oracle for AOE Path B Spread (#181 — count-
/// capped Circle). Mirrors the GPU kernel's WGSL math bit-for-bit so
/// callers feeding `apply_program_aoe` produce a chronicle record set
/// byte-equal to the GPU dispatcher.
///
/// Inputs:
///   * `center` — explicit cast target's position (same convention as
///     Circle).
///   * `radius` — `args[0]` (same `dist² ≤ radius²` gate as Circle).
///   * `max_targets` — `args[1]` rounded to integer (`as u32`). Caps the
///     emitted set after the in-radius filter + AgentId sort. `0` ⇒
///     empty set.
///   * `candidates` — slice of `(AgentId, position)` pairs.
///
/// Output: in-radius `AgentId`s sorted ascending by raw id, then
/// truncated to the first `max_targets` entries (P11). Determinism is
/// preserved because both the in-radius set and the cap selection are
/// AgentId-stable.
///
/// **Spread vs Circle.** Spread reuses Circle's geometric gate and adds
/// a hard count cap — the lowest-AgentId K targets within the radius
/// fire. Useful for "single-target with overflow", "chain-style" hit
/// caps, and similar bounded multi-hit semantics.
///
/// **Spatial walk limitation.** 27-cell walk; if `radius >
/// SPATIAL_CELL_SIZE`, candidates beyond the 27-cell ring are missed.
/// Same caveat as Circle/Sphere/Box (#179).
pub fn apply_program_aoe_spread_filter(
    center:      glam::Vec3,
    radius:      f32,
    max_targets: u32,
    candidates:  &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    let radius_sq = radius * radius;
    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let dvec = cand_pos - center;
        if dvec.dot(dvec) <= radius_sq {
            hits.push(id);
        }
    }
    // P11: sort ascending, then truncate to the cap so the kept set is
    // the lowest-AgentId K hits — both backends agree on the slice
    // (after canonicalize sort on the GPU side).
    hits.sort_by_key(|id| id.raw());
    hits.truncate(max_targets as usize);
    hits
}

/// CPU-side column-filter oracle for AOE Path B Column (#181 — vertical
/// cylinder extending UP from the cast center). Mirrors the GPU kernel's
/// WGSL math bit-for-bit.
///
/// Inputs:
///   * `center` — explicit cast target's position. The column starts at
///     `center.y` and extends upward to `center.y + height`.
///   * `radius` — `args[0]` (horizontal radius in the XZ plane).
///   * `height` — `args[1]` (vertical extent above `center.y`).
///   * `candidates` — slice of `(AgentId, position)` pairs.
///
/// Output: in-column `AgentId`s sorted ascending by raw id (P11).
///
/// In-column predicate (per candidate, identical to the WGSL kernel):
///   1. `dist_xz_sq ≤ radius²` where `dist_xz_sq = dx*dx + dz*dz`
///      (Y is the vertical axis, ignored in the horizontal gate).
///   2. `0 ≤ dy ≤ height` where `dy = cand.y - center.y`. The column
///      extends UP only — candidates below `center.y` are excluded.
///
/// **Column vs Cylinder.** Column extends UP only (like a one-sided
/// pillar from the ground); Cylinder is symmetric (`|dy| ≤ height/2`).
/// Pick the one matching the spec'd intent.
///
/// **Spatial walk limitation.** 27-cell walk around the center; if
/// `radius` or `height` exceeds `SPATIAL_CELL_SIZE`, candidates beyond
/// the 27-cell ring are missed. Same caveat as Circle/Sphere/Box.
pub fn apply_program_aoe_column_filter(
    center:     glam::Vec3,
    radius:     f32,
    height:     f32,
    candidates: &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    let radius_sq = radius * radius;
    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let dvec = cand_pos - center;
        let dist_xz_sq = dvec.x * dvec.x + dvec.z * dvec.z;
        if dist_xz_sq > radius_sq {
            continue;
        }
        if dvec.y < 0.0 || dvec.y > height {
            continue;
        }
        hits.push(id);
    }
    hits.sort_by_key(|id| id.raw());
    hits
}

/// CPU-side wall-filter oracle for AOE Path B Wall (#181 — facing-
/// bearing rectangular slab). Mirrors the GPU kernel's WGSL math bit-
/// for-bit.
///
/// Inputs:
///   * `center` — wall origin (the cast target's position).
///   * `length` — `args[0]` (width of the slab perpendicular to facing,
///     symmetric — `lateral ≤ length/2`).
///   * `height` — `args[1]` (vertical extent — `0 ≤ dy ≤ height`,
///     extends UP from the center, matching Column's vertical convention).
///   * `thickness` — `args[2]` (depth in the facing direction —
///     `0 ≤ forward ≤ thickness`).
///   * `facing_deg` — `args[3]` (yaw angle in degrees in the XZ plane,
///     `0deg = +X`, increasing CCW toward `+Z`). The facing direction
///     is `dir_xz = (cos θ, 0, sin θ)`. The lateral axis is `perp_xz =
///     (-sin θ, 0, cos θ)` (90°-rotated CCW).
///   * `candidates` — slice of `(AgentId, position)` pairs.
///
/// Output: in-wall `AgentId`s sorted ascending by raw id (P11).
///
/// In-wall predicate (per candidate, identical to the WGSL kernel):
///   1. Compute `to_cand = cand_pos - center`.
///   2. Forward projection: `forward = to_cand.x * cos(θ) + to_cand.z *
///      sin(θ)`. Gate `0 ≤ forward ≤ thickness` (slab in front of
///      center, within thickness depth).
///   3. Lateral projection: `lateral = -to_cand.x * sin(θ) + to_cand.z *
///      cos(θ)`. Gate `|lateral| ≤ length/2` (within slab width).
///   4. Vertical: `0 ≤ to_cand.y ≤ height` (extends up from center).
///
/// **Convention chosen.** Wall faces outward from the cast center toward
/// `facing_deg`, with thickness extending in front (NOT centered on the
/// center — the slab starts AT the center and extends `thickness` units
/// forward). Length is perpendicular width (symmetric). Height is
/// vertical (extends UP, matching Column). The facing arg in degrees
/// converts to a unit XZ direction via `(cos, 0, sin)` — `0deg` = `+X`.
///
/// **Spatial walk limitation.** 27-cell walk; if `length`, `height`, or
/// `thickness` exceeds `SPATIAL_CELL_SIZE`, candidates beyond the 27-
/// cell ring are missed. Wall is the only 4-arg shape (the others use
/// 1-3 args + zero-padding); the schema_hash already pins this layout.
pub fn apply_program_aoe_wall_filter(
    center:     glam::Vec3,
    length:     f32,
    height:     f32,
    thickness:  f32,
    facing_deg: f32,
    candidates: &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    let half_length = length * 0.5;
    let theta_rad = facing_deg * std::f32::consts::PI / 180.0;
    let dir_x = theta_rad.cos();
    let dir_z = theta_rad.sin();
    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let to_cand = cand_pos - center;
        let forward = to_cand.x * dir_x + to_cand.z * dir_z;
        if forward < 0.0 || forward > thickness {
            continue;
        }
        let lateral = -to_cand.x * dir_z + to_cand.z * dir_x;
        if lateral.abs() > half_length {
            continue;
        }
        if to_cand.y < 0.0 || to_cand.y > height {
            continue;
        }
        hits.push(id);
    }
    hits.sort_by_key(|id| id.raw());
    hits
}

/// CPU-side cylinder-filter oracle for AOE Path B Cylinder (#181 — 3D
/// cylinder centered on the cast target, vertical-symmetric). Mirrors
/// the GPU kernel's WGSL math bit-for-bit.
///
/// Inputs:
///   * `center` — explicit cast target's position.
///   * `radius` — `args[0]` (horizontal radius in the XZ plane).
///   * `height` — `args[1]` (full vertical extent — symmetric, gate is
///     `|dy| ≤ height/2`).
///   * `candidates` — slice of `(AgentId, position)` pairs.
///
/// Output: in-cylinder `AgentId`s sorted ascending by raw id (P11).
///
/// In-cylinder predicate (per candidate, identical to the WGSL kernel):
///   1. `dist_xz_sq ≤ radius²` (horizontal gate, ignores Y).
///   2. `|cand.y - center.y| ≤ height/2` (vertical, symmetric).
///
/// **Cylinder vs Column.** Cylinder is symmetric vertically (extends
/// `height/2` above AND below the center); Column extends UP only.
pub fn apply_program_aoe_cylinder_filter(
    center:     glam::Vec3,
    radius:     f32,
    height:     f32,
    candidates: &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    let radius_sq = radius * radius;
    let half_height = height * 0.5;
    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let dvec = cand_pos - center;
        let dist_xz_sq = dvec.x * dvec.x + dvec.z * dvec.z;
        if dist_xz_sq > radius_sq {
            continue;
        }
        if dvec.y.abs() > half_height {
            continue;
        }
        hits.push(id);
    }
    hits.sort_by_key(|id| id.raw());
    hits
}

/// CPU-side dome-filter oracle for AOE Path B Dome (#181 — half-sphere
/// above the cast center's horizontal plane). Mirrors the GPU kernel's
/// WGSL math bit-for-bit.
///
/// Inputs:
///   * `center` — explicit cast target's position. The dome covers the
///     hemisphere where `cand.y ≥ center.y`.
///   * `radius` — `args[0]` (3D distance gate, same as Sphere).
///   * `candidates` — slice of `(AgentId, position)` pairs.
///
/// Output: in-dome `AgentId`s sorted ascending by raw id (P11).
///
/// In-dome predicate:
///   1. `dist_sq ≤ radius²` (3D distance, same as Sphere).
///   2. `cand.y ≥ center.y` (above the horizontal plane). The boundary
///      is inclusive — candidates exactly at `center.y` are in-dome.
pub fn apply_program_aoe_dome_filter(
    center:     glam::Vec3,
    radius:     f32,
    candidates: &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    let radius_sq = radius * radius;
    let mut hits: Vec<AgentId> = Vec::with_capacity(candidates.len());
    for &(id, cand_pos) in candidates {
        let dvec = cand_pos - center;
        if dvec.dot(dvec) > radius_sq {
            continue;
        }
        if dvec.y < 0.0 {
            continue;
        }
        hits.push(id);
    }
    hits.sort_by_key(|id| id.raw());
    hits
}

/// CPU-side hull-filter oracle for AOE Path B Hull (#181). The Hull
/// shape's spec semantics are not nailed down today — the only place
/// it's mentioned in the codebase is `ShapeKind::Hull = 11` in
/// `program.rs` (no doc-comment, no spec text under
/// `dataset/abilities/`). Without a spec, we ship Hull as an **alias to
/// Sphere** (3D `dist² ≤ radius²`) so author intent matching the most
/// common reading ("hull around me") works, and a future spec change
/// can refine the gate without an API break (the args slot already
/// reserves 4 f32, only `args[0]` is consumed today).
///
/// **NOTE: Hull is a Sphere alias.** When/if the spec defines Hull as
/// a distinct shape (convex hull around an entity group? equipment-
/// blob hitbox? something else), update both this filter and the GPU
/// branch in `wgsl_body.rs` together.
///
/// Inputs:
///   * `center` — explicit cast target's position.
///   * `radius` — `args[0]` (sphere radius).
///   * `candidates` — slice of `(AgentId, position)` pairs.
///
/// Output: in-hull `AgentId`s sorted ascending by raw id (P11).
pub fn apply_program_aoe_hull_filter(
    center:     glam::Vec3,
    radius:     f32,
    candidates: &[(AgentId, glam::Vec3)],
) -> Vec<AgentId> {
    // Hull is a Sphere alias today (see doc-comment NOTE above).
    apply_program_aoe_sphere_filter(center, radius, candidates)
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
    fn aoe_spread_caps_target_count_at_max() {
        // #181 AOE Path B Spread — count-capped Circle. The dispatcher
        // expands Spread across `aoe_targets` (the caller pre-filters
        // by Circle's geometric gate AND truncates to `max_targets`
        // via `apply_program_aoe_spread_filter`). This test pins the
        // dispatch leg: when the caller passes a single-element slice,
        // the slot fires once. The geometric + count-cap math is
        // exercised by `aoe_spread_filter_*` tests below.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Spread,
            args: [5.0, 1.0, 0.0, 0.0],
        }));
        // Caller has already capped to max_targets=1 (the lowest-id
        // agent in radius). Dispatcher fires once on that target.
        let aoe_targets = [agent_n(2)];
        let events = apply_program_aoe(
            &prog,
            agent_n(1),
            agent_n(2),
            &aoe_targets,
            0, 0xCAFE,
            &CasterStats::default(),
            &CasterStats::default(),
        );
        assert_eq!(events.len(), 1, "Spread (max=1) → single in-radius target receives event");
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
    fn aoe_box_expands_across_pre_filtered_targets() {
        // Box shape: same expansion contract as Circle/Cone — the caller
        // pre-filters and passes the in-box set, the dispatcher emits
        // one ApplyEvent per target. The AABB math itself is exercised
        // by `aoe_box_filter_*` tests below; this test pins the
        // dispatch leg.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 20.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Box,
            args: [1.5, 1.5, 1.5, 0.0],
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
        assert_eq!(events.len(), 2, "Box AOE → one event per pre-filtered target");
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
    fn aoe_box_filter_non_uniform_extents() {
        // Non-uniform extents (wx=2, wy=0.5, wz=2) — narrow band along
        // the y-axis. Candidates at varying y must be filtered down to
        // only those with |y - center.y| ≤ 0.5; x and z extents are
        // wide enough to admit the full row.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),     // in-box (origin)
            (agent_n(2), glam::Vec3::new(1.0, 0.4, 0.5)),     // in-box (|y|=0.4 ≤ 0.5)
            (agent_n(3), glam::Vec3::new(-1.5, 0.5, -1.0)),   // in-box (|y|=0.5 — wall, ≤ semantic)
            (agent_n(4), glam::Vec3::new(0.0, 0.6, 0.0)),     // out (|y|=0.6 > 0.5)
            (agent_n(5), glam::Vec3::new(0.0, -0.7, 0.0)),    // out (|y|=0.7 > 0.5)
            (agent_n(6), glam::Vec3::new(2.5, 0.0, 0.0)),     // out (|x|=2.5 > 2.0)
        ];
        let hits = apply_program_aoe_box_filter(center, 2.0, 0.5, 2.0, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2), agent_n(3)],
            "narrow-band box must filter to {{1,2,3}} (sorted ascending by id); got {hits:?}"
        );
    }

    #[test]
    fn aoe_box_filter_wall_inclusive_edge_case() {
        // A candidate at exactly `center + (wx, 0, 0)` is in-box (≤
        // semantic, not <). Pin so a future "open AABB" change surfaces.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let wx = 1.5;
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(wx, 0.0, 0.0)),       // at +x wall
            (agent_n(2), glam::Vec3::new(-wx, 0.0, 0.0)),      // at -x wall
            (agent_n(3), glam::Vec3::new(wx + 0.001, 0.0, 0.0)), // just past +x wall — out
        ];
        let hits = apply_program_aoe_box_filter(center, wx, 1.5, 1.5, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2)],
            "candidates at the AABB walls (|d|=wx) are in-box (closed AABB), \
             but |d|=wx+ε is out; got {hits:?}"
        );
    }

    #[test]
    fn aoe_box_filter_empty_set_outside_extents() {
        // No candidate inside the box → empty in-box set.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(2.0, 0.0, 0.0)),  // out (|x|=2 > 1.5)
            (agent_n(2), glam::Vec3::new(0.0, 5.0, 0.0)),  // out (|y|=5 > 1.5)
            (agent_n(3), glam::Vec3::new(0.0, 0.0, -10.0)), // out (|z|=10 > 1.5)
        ];
        let hits = apply_program_aoe_box_filter(center, 1.5, 1.5, 1.5, &candidates);
        assert!(
            hits.is_empty(),
            "all candidates outside extents → empty in-box set; got {hits:?}"
        );
    }

    // -- AOE Path B Sphere (#180) -----------------------------------

    #[test]
    fn aoe_sphere_expands_across_pre_filtered_targets() {
        // Sphere shape: same expansion contract as Circle/Cone/Box —
        // dispatcher emits one ApplyEvent per pre-filtered target.
        // Trivial mirror of the Circle dispatch test.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 18.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Sphere,
            args: [2.0, 0.0, 0.0, 0.0],
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
        assert_eq!(events.len(), 2, "Sphere AOE → one event per pre-filtered target");
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
    fn aoe_sphere_filter_3d_distance_check() {
        // Sphere is mathematically equivalent to Circle today (3D
        // dist² ≤ radius²). Pin the equivalence here so a future
        // divergence (e.g. flat-disk Circle vs true 3D Sphere) is
        // visible.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),    // d=0   in
            (agent_n(2), glam::Vec3::new(1.5, 0.0, 0.0)),    // d=1.5 in
            (agent_n(3), glam::Vec3::new(0.0, 2.0, 0.0)),    // d=2.0 wall, in (≤)
            (agent_n(4), glam::Vec3::new(0.0, 0.0, 2.001)),  // d>2.0 out
        ];
        let hits = apply_program_aoe_sphere_filter(center, 2.0, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2), agent_n(3)],
            "sphere(2.0) at origin must hit 1/2/3 (3rd at wall ≤ semantic); got {hits:?}"
        );
    }

    // -- AOE Path B Ring (#180) -------------------------------------

    #[test]
    fn aoe_ring_expands_across_pre_filtered_targets() {
        // Ring shape: dispatcher contract is identical to Circle/Box.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 14.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Ring,
            args: [0.5, 2.0, 0.0, 0.0],
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
        assert_eq!(events.len(), 2, "Ring AOE → one event per pre-filtered target");
    }

    #[test]
    fn aoe_ring_filter_excludes_inner_radius() {
        // Annulus: inner=0.5, outer=2.0. Agents at d=0 and d=0.4 are
        // INSIDE the inner radius and are excluded; agents at d=0.5
        // (inner wall, in), d=1.5 (in), d=2.0 (outer wall, in), d=2.1
        // (out).
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),    // d=0   inner-excluded
            (agent_n(2), glam::Vec3::new(0.4, 0.0, 0.0)),    // d=0.4 inner-excluded
            (agent_n(3), glam::Vec3::new(0.5, 0.0, 0.0)),    // d=0.5 inner wall — in (≤)
            (agent_n(4), glam::Vec3::new(1.5, 0.0, 0.0)),    // d=1.5 in
            (agent_n(5), glam::Vec3::new(2.0, 0.0, 0.0)),    // d=2.0 outer wall — in (≤)
            (agent_n(6), glam::Vec3::new(2.1, 0.0, 0.0)),    // d=2.1 out
        ];
        let hits = apply_program_aoe_ring_filter(center, 0.5, 2.0, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(3), agent_n(4), agent_n(5)],
            "ring(0.5, 2.0) excludes d<inner; both walls are inclusive; got {hits:?}"
        );
    }

    #[test]
    fn aoe_ring_filter_inverted_bounds_returns_empty() {
        // inner > outer ⇒ predicate `inner² ≤ d² ≤ outer²` is
        // unsatisfiable. Empty in-ring set on both backends.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(1.0, 0.0, 0.0)),
            (agent_n(2), glam::Vec3::new(0.5, 0.0, 0.0)),
        ];
        let hits = apply_program_aoe_ring_filter(center, 5.0, 2.0, &candidates);
        assert!(
            hits.is_empty(),
            "inner > outer must yield empty in-ring set; got {hits:?}"
        );
    }

    // -- AOE Path B Line (#180) -------------------------------------

    #[test]
    fn aoe_line_expands_across_pre_filtered_targets() {
        // Line shape: dispatcher contract is identical to Circle/Cone/
        // Box/Sphere/Ring.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 22.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Line,
            args: [5.0, 1.0, 0.0, 0.0],
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
        assert_eq!(events.len(), 2, "Line AOE → one event per pre-filtered target");
    }

    #[test]
    fn aoe_line_filter_degenerate_self_target_returns_empty() {
        // Caster targets self ⇒ direction = (0,0,0) is degenerate.
        // Must return empty set on both backends (matches the cone's
        // degenerate semantic).
        let apex = glam::Vec3::new(2.0, 3.0, 4.0);
        let target_pos = apex;
        let candidates = vec![
            (agent_n(2), glam::Vec3::new(3.0, 3.0, 4.0)),
            (agent_n(3), glam::Vec3::new(2.0, 4.0, 4.0)),
        ];
        let hits = apply_program_aoe_line_filter(
            apex, target_pos, 5.0, 1.0, &candidates,
        );
        assert!(
            hits.is_empty(),
            "degenerate line (caster targets self) → empty set; got {hits:?}"
        );
    }

    #[test]
    fn aoe_line_filter_lateral_outside_width_excluded() {
        // Apex at origin facing +X; length=5, width=1 ⇒ half_width=0.5.
        // Candidates:
        //   ( 1.0,  0.0, 0.0): along=1, perp=0           → in
        //   ( 2.0,  0.5, 0.0): along=2, perp=0.5 (=hw)    → in (≤)
        //   ( 3.0, -0.4, 0.0): along=3, perp=0.4         → in
        //   ( 1.0,  0.6, 0.0): along=1, perp=0.6 (>hw)    → out (lateral)
        //   ( 5.0,  0.0, 0.0): along=5 (=length)         → in (≤)
        //   ( 6.0,  0.0, 0.0): along=6 (>length)         → out
        //   (-1.0,  0.0, 0.0): along=-1 (behind apex)    → out
        let apex = glam::Vec3::new(0.0, 0.0, 0.0);
        let target_pos = glam::Vec3::new(5.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(1.0, 0.0, 0.0)),
            (agent_n(2), glam::Vec3::new(2.0, 0.5, 0.0)),
            (agent_n(3), glam::Vec3::new(3.0, -0.4, 0.0)),
            (agent_n(4), glam::Vec3::new(1.0, 0.6, 0.0)),
            (agent_n(5), glam::Vec3::new(5.0, 0.0, 0.0)),
            (agent_n(6), glam::Vec3::new(6.0, 0.0, 0.0)),
            (agent_n(7), glam::Vec3::new(-1.0, 0.0, 0.0)),
        ];
        let hits = apply_program_aoe_line_filter(
            apex, target_pos, /*length*/ 5.0, /*width*/ 1.0, &candidates,
        );
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2), agent_n(3), agent_n(5)],
            "line(5, 1) facing +X must include along∈[0,length] ∧ perp ≤ \
             half_width only; got {hits:?}"
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

    // -- AOE Path B Spread / Column / Wall / Cylinder / Dome / Hull (#181) --

    #[test]
    fn aoe_spread_filter_caps_to_max_targets() {
        // 4 candidates, all within radius=2.0; max_targets=2 → keep
        // the lowest-AgentId 2 (sorted ascending by id, then truncate).
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),     // d=0
            (agent_n(2), glam::Vec3::new(1.0, 0.0, 0.0)),     // d=1
            (agent_n(3), glam::Vec3::new(0.0, 1.5, 0.0)),     // d=1.5
            (agent_n(4), glam::Vec3::new(2.0, 0.0, 0.0)),     // d=2.0 wall, in
        ];
        let hits = apply_program_aoe_spread_filter(center, /*radius*/ 2.0, /*max*/ 2, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2)],
            "spread(r=2, max=2) must keep lowest-AgentId 2 in-radius hits; got {hits:?}"
        );
    }

    #[test]
    fn aoe_spread_filter_zero_max_returns_empty() {
        // max_targets=0 ⇒ empty set even when in-radius candidates exist.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),
            (agent_n(2), glam::Vec3::new(1.0, 0.0, 0.0)),
        ];
        let hits = apply_program_aoe_spread_filter(center, 5.0, 0, &candidates);
        assert!(hits.is_empty(), "max=0 → empty set, got {hits:?}");
    }

    #[test]
    fn aoe_spread_filter_max_exceeds_in_radius_keeps_all() {
        // max_targets larger than in-radius count → all in-radius kept.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),
            (agent_n(2), glam::Vec3::new(0.5, 0.0, 0.0)),
            (agent_n(3), glam::Vec3::new(10.0, 0.0, 0.0)),    // out of radius
        ];
        let hits = apply_program_aoe_spread_filter(center, 1.0, 100, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2)],
            "max≫in-radius keeps all in-radius hits, got {hits:?}"
        );
    }

    #[test]
    fn aoe_column_filter_extends_up_only() {
        // Column at origin, radius=1.0, height=3.0. XZ disc at any y in
        // [0, 3] is in. Below y=0 or above y=3 is out. Outside XZ radius
        // is out.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),     // origin: in
            (agent_n(2), glam::Vec3::new(0.5, 1.5, 0.5)),     // mid-column: in (XZ d=0.71<1, y=1.5)
            (agent_n(3), glam::Vec3::new(0.0, 3.0, 0.0)),     // top wall: in (y=3 ≤ 3)
            (agent_n(4), glam::Vec3::new(0.0, -0.1, 0.0)),    // below: out (y<0)
            (agent_n(5), glam::Vec3::new(0.0, 3.1, 0.0)),     // above: out (y>3)
            (agent_n(6), glam::Vec3::new(1.5, 1.0, 0.0)),     // outside XZ: out (XZ d=1.5>1)
        ];
        let hits = apply_program_aoe_column_filter(center, /*r*/ 1.0, /*h*/ 3.0, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2), agent_n(3)],
            "column(r=1, h=3) extends UP only; got {hits:?}"
        );
    }

    #[test]
    fn aoe_column_filter_ignores_y_in_xz_distance() {
        // Tall column: a candidate far above on the cylinder axis is
        // in-column even though 3D distance is large. Pin the "Y is
        // ignored in XZ gate" semantic.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.5, 5.0, 0.0)),     // XZ=0.5 in, y=5 in
            (agent_n(2), glam::Vec3::new(0.0, 9.99, 0.0)),    // XZ=0 in, y=9.99 in
        ];
        let hits = apply_program_aoe_column_filter(center, /*r*/ 1.0, /*h*/ 10.0, &candidates);
        assert_eq!(hits, vec![agent_n(1), agent_n(2)]);
    }

    #[test]
    fn aoe_wall_filter_facing_plus_x_basic_slab() {
        // Wall facing +X (facing_deg=0), length=4 (half=2 lateral),
        // height=2, thickness=1. Slab covers x∈[0,1], z∈[-2,2], y∈[0,2].
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),     // at center: in
            (agent_n(2), glam::Vec3::new(0.5, 1.0, 1.5)),     // forward=0.5, lateral=1.5, y=1: in
            (agent_n(3), glam::Vec3::new(1.0, 2.0, -2.0)),    // wall corner: forward=1, lateral=-2, y=2: in (≤)
            (agent_n(4), glam::Vec3::new(1.5, 0.0, 0.0)),     // forward=1.5 > 1: out
            (agent_n(5), glam::Vec3::new(-0.5, 0.0, 0.0)),    // forward=-0.5 < 0: out (behind)
            (agent_n(6), glam::Vec3::new(0.5, 0.0, 2.5)),     // lateral=2.5 > 2: out
            (agent_n(7), glam::Vec3::new(0.5, 2.5, 0.0)),     // y=2.5 > 2: out
        ];
        let hits = apply_program_aoe_wall_filter(
            center, /*length*/ 4.0, /*height*/ 2.0, /*thickness*/ 1.0,
            /*facing_deg*/ 0.0, &candidates,
        );
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2), agent_n(3)],
            "wall(len=4, h=2, thick=1, +X) covers x∈[0,1], z∈[-2,2], y∈[0,2]; got {hits:?}"
        );
    }

    #[test]
    fn aoe_wall_filter_facing_plus_z_rotates_slab() {
        // Wall facing +Z (facing_deg=90). Direction is now (cos 90, 0,
        // sin 90) = (0, 0, 1). Forward is +Z; lateral is -X (perp =
        // (-sin 90, 0, cos 90) = (-1, 0, 0)). Slab covers z∈[0,1],
        // x∈[-2,2], y∈[0,2] under the same args as the +X test.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.5)),     // forward=0.5 in slab
            (agent_n(2), glam::Vec3::new(1.5, 1.0, 1.0)),     // forward=1 wall, lateral=-1.5 in (|≤2|)
            (agent_n(3), glam::Vec3::new(0.0, 0.0, -0.1)),    // forward=-0.1: out (behind)
            (agent_n(4), glam::Vec3::new(2.5, 0.0, 0.5)),     // lateral=-2.5: out
        ];
        let hits = apply_program_aoe_wall_filter(
            center, 4.0, 2.0, 1.0, /*facing_deg*/ 90.0, &candidates,
        );
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2)],
            "wall facing +Z must rotate the slab axes accordingly; got {hits:?}"
        );
    }

    #[test]
    fn aoe_cylinder_filter_symmetric_vertical() {
        // Cylinder at origin, radius=1.0, height=2.0 (half=1). Covers
        // XZ disc at any y in [-1, 1]. Symmetric vertically — distinct
        // from Column.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),     // center: in
            (agent_n(2), glam::Vec3::new(0.5, 1.0, 0.0)),     // top wall: in (y=1 ≤ 1)
            (agent_n(3), glam::Vec3::new(0.5, -1.0, 0.0)),    // bottom wall: in (|y|=1 ≤ 1)
            (agent_n(4), glam::Vec3::new(0.0, 1.5, 0.0)),     // y>half: out
            (agent_n(5), glam::Vec3::new(0.0, -1.5, 0.0)),    // y<-half: out
            (agent_n(6), glam::Vec3::new(1.5, 0.0, 0.0)),     // XZ outside: out
        ];
        let hits = apply_program_aoe_cylinder_filter(center, /*r*/ 1.0, /*h*/ 2.0, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2), agent_n(3)],
            "cylinder(r=1, h=2) symmetric vertically; got {hits:?}"
        );
    }

    #[test]
    fn aoe_dome_filter_above_plane_only() {
        // Dome at origin, radius=2.0. Sphere gate + y≥0 plane gate.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),     // y=0 plane (boundary): in
            (agent_n(2), glam::Vec3::new(1.0, 1.0, 0.0)),     // d=1.41, y=1: in
            (agent_n(3), glam::Vec3::new(0.0, 2.0, 0.0)),     // d=2 wall, y=2: in (≤)
            (agent_n(4), glam::Vec3::new(0.0, -0.1, 0.0)),    // y<0: out (below plane)
            (agent_n(5), glam::Vec3::new(0.0, 2.1, 0.0)),     // d>radius: out
        ];
        let hits = apply_program_aoe_dome_filter(center, /*r*/ 2.0, &candidates);
        assert_eq!(
            hits,
            vec![agent_n(1), agent_n(2), agent_n(3)],
            "dome(r=2) covers upper hemisphere; got {hits:?}"
        );
    }

    #[test]
    fn aoe_hull_filter_aliases_sphere() {
        // Hull is a Sphere alias today (see filter doc-comment NOTE).
        // Pin equivalence so any future spec change surfaces.
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let candidates = vec![
            (agent_n(1), glam::Vec3::new(0.0, 0.0, 0.0)),
            (agent_n(2), glam::Vec3::new(1.0, 1.0, 1.0)),
            (agent_n(3), glam::Vec3::new(0.0, 0.0, 2.001)),
        ];
        let hull_hits = apply_program_aoe_hull_filter(center, 2.0, &candidates);
        let sphere_hits = apply_program_aoe_sphere_filter(center, 2.0, &candidates);
        assert_eq!(
            hull_hits, sphere_hits,
            "hull must alias sphere today; got hull={hull_hits:?} sphere={sphere_hits:?}"
        );
    }

    #[test]
    fn aoe_spread_dispatch_one_event_per_target_after_cap() {
        // Dispatcher contract: same as Circle/Cone/Box/Sphere/Ring/Line.
        // Caller passes the post-cap target slice.
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 12.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Spread,
            args: [5.0, 2.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog, agent_n(1), agent_n(2), &aoe_targets, 0, 0xCAFE,
            &CasterStats::default(), &CasterStats::default(),
        );
        assert_eq!(events.len(), 2, "Spread dispatch → one event per target slot");
    }

    #[test]
    fn aoe_column_dispatch_one_event_per_target() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 16.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Column,
            args: [1.5, 4.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog, agent_n(1), agent_n(2), &aoe_targets, 0, 0xCAFE,
            &CasterStats::default(), &CasterStats::default(),
        );
        assert_eq!(events.len(), 2, "Column dispatch → one event per target slot");
    }

    #[test]
    fn aoe_wall_dispatch_one_event_per_target() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 16.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Wall,
            args: [4.0, 2.0, 1.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog, agent_n(1), agent_n(2), &aoe_targets, 0, 0xCAFE,
            &CasterStats::default(), &CasterStats::default(),
        );
        assert_eq!(events.len(), 2, "Wall dispatch → one event per target slot");
    }

    #[test]
    fn aoe_cylinder_dispatch_one_event_per_target() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 16.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Cylinder,
            args: [1.5, 2.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog, agent_n(1), agent_n(2), &aoe_targets, 0, 0xCAFE,
            &CasterStats::default(), &CasterStats::default(),
        );
        assert_eq!(events.len(), 2, "Cylinder dispatch → one event per target slot");
    }

    #[test]
    fn aoe_dome_dispatch_one_event_per_target() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 16.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Dome,
            args: [2.0, 0.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog, agent_n(1), agent_n(2), &aoe_targets, 0, 0xCAFE,
            &CasterStats::default(), &CasterStats::default(),
        );
        assert_eq!(events.len(), 2, "Dome dispatch → one event per target slot");
    }

    #[test]
    fn aoe_hull_dispatch_one_event_per_target() {
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 16.0 }],
        );
        prog.per_effect_areas.push(Some(EffectAreaShape {
            kind: ShapeKind::Hull,
            args: [2.0, 0.0, 0.0, 0.0],
        }));
        let aoe_targets = [agent_n(2), agent_n(3)];
        let events = apply_program_aoe(
            &prog, agent_n(1), agent_n(2), &aoe_targets, 0, 0xCAFE,
            &CasterStats::default(), &CasterStats::default(),
        );
        assert_eq!(events.len(), 2, "Hull dispatch → one event per target slot");
    }
}
