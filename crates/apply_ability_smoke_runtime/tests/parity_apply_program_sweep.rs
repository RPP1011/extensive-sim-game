//! P3 cross-backend parity sweep for `apply_program` (#133, #138).
//!
//! Closes the open half of `apply_program`'s P3 contract: every
//! ApplyEvent the CPU oracle (`engine::ability::apply::apply_program`)
//! emits for a given (caster, target, ability_id, tick) tuple must have
//! a byte-identical chronicle record on the GPU dispatcher's
//! `event_ring`. The four existing apply_ability_*_runtime device tests
//! pin parity per-fixture, but each hand-writes its expected record;
//! this test drives the full matrix end-to-end through the CPU oracle
//! so coverage tracks as new ApplyEvent variants get wired (instead of
//! every new variant needing its own bespoke device test).
//!
//! ## Matrix
//!
//! N abilities × M caster×target × T ticks = K casts. The registered
//! abilities cover every effect-modifier combination the GPU dispatcher
//! handles end-to-end today:
//!
//!   1. `Damage(30)`                                — Strike-shape (no modifiers)
//!   2. `Heal(25)`                                  — Mend-shape
//!   3. `Shield(50)`                                — ShieldUp-shape
//!   4. `SelfDamage(5) + 5% MaxHp`                  — Bleed-shape (scaling on MaxHp)
//!   5. `Execute(20) when target.hp < 20`           — Reap-shape (when-predicate gate)
//!   6. `Damage(10) chance=0xFFFE (always-fires)`   — Daze-shape, near-deterministic (1/65536 miss)
//!   7. `Damage(40) { stun 1s }`                    — Reap+nested-stun shape
//!   8. `Heal(8) + 50% AbilityPower`                — AP scaling (=0.0 on both backends)
//!   9. `LifeSteal(0.5, 5s)`                        — Vampirize-shape (q8 fraction)
//!  10. `DamageModify(0.5, 5s)`                     — Fortify-shape (q8 multiplier)
//!  11. `Root(3s)`                                  — Stop-shape (Wave 2 piece 1)
//!  12. `Silence(3s)`                               — Mute-shape (Wave 2 piece 1)
//!  13. `Fear(3s)`                                  — Terrify-shape (Wave 2 piece 1)
//!  14. `Taunt(3s)`                                 — Provoke-shape (Wave 2 piece 1)
//!  15. `Dash(12)`                                  — Lunge-shape (Wave 2 piece 2)
//!  16. `Blink(8)`                                  — Phase-shape (Wave 2 piece 2)
//!  17. `Knockback(5)`                              — Smash-shape (Wave 2 piece 2)
//!  18. `Pull(3)`                                   — Hook-shape (Wave 2 piece 2)
//!  19. `DamageOverTime(6, 30)`                     — Burn-shape (Wave 1.5+)
//!  20. `HealOverTime(4, 50)`                       — Regen-shape (Wave 1.5+)
//!  21. `TimedShield(25, 100)`                      — Aegis-shape (Wave 1.5+)
//!  22. `Stealth(50)`                               — Vanish-shape (extended status)
//!  23. `Charm(30)`                                 — Allure-shape (extended status)
//!  24. `Grounded(25)`                              — Tether-shape (extended status)
//!  25. `Suppress(40)`                              — Hush-shape (extended status)
//!  26. `Buff(AttackSpeed, -64, 50)`                — Empower-shape (slice γ tail, packed signed magnitude)
//!  27. `Reflect(50, -64)`                          — Mirror-shape (slice γ tail, packed signed fraction)
//!  28. `Harvest(0xCAFEBABE, 5)`                    — Reap_Ore-shape (slice γ tail, caster-self)
//!  29. `PlaceVoxel(0xFACEFEED)`                    — Drop_Stone-shape (slice γ tail, caster-self)
//!  30. `Damage(10) chance=0x4000 (~25%)`           — DazeMid-shape, P11 chance-gate parity (intermediate q16)
//!
//! M = 1 caster×target permutation in this fixture: `(c=0, t=0)`
//! self-cast. The smoke fixture's `apply_ability` source uses the
//! implicit-target rule (no `by ... target ...` clause), so the
//! dispatcher writes `caster_slot == target_slot == agent_id` per the
//! per-agent loop. Distinct caster≠target requires an explicit-target
//! verb body — deferred to a follow-on fixture (see N_AGENTS docs).
//!
//! T = 5 ticks (0, 17, 100, 1000, 65500) — varied across the u32 range
//! to surface any wraparound bug in expires_at_tick computations.
//!
//! K = 30 × 1 × 5 = **150 casts**, producing ≈ 150 chronicle records
//! (145 primary always-fire + 5 nested-Stun follow-ups from the
//! Reap+Stun-shape arm + ~1.25 expected DazeMid fires under chance ≈ 25%
//! across 5 ticks, draws determined by the PCG mixer).
//!
//! ## P11 chance gate parity (slice ζ, 2026-05-07)
//!
//! Both backends now key the chance gate's draw on the same PCG mixer
//! (`per_agent_u32_pcg_with_extra` host / `per_agent_u32_with_extra`
//! WGSL prelude — bit-equal under shared inputs). The CPU oracle and
//! GPU dispatcher therefore agree on EVERY (seed, caster_slot, tick,
//! slot_idx) tuple's gate decision, so the sweep can pin intermediate
//! q16 thresholds. The DazeMid arm above (q16 = 0x4000) exercises the
//! divergent-fires branch — some draws fire and some don't across the
//! 5 sweep ticks; both backends must produce identical record sets.
//!
//! ## What's deferred
//!
//! - **Distinct caster≠target casts.** This fixture's source uses the
//!   implicit-target rule (`apply_ability AID by self`), so the
//!   dispatcher always writes `caster_slot == target_slot`. To exercise
//!   the explicit-target slice ε plumbing (`apply_ability AID by self
//!   target <expr>`) we'd need a parallel fixture compiled from a
//!   `.sim` source that lowers to `CgStmt::ApplyAbility { target:
//!   distinct }`. The CPU oracle path (apply_program with distinct
//!   `caster` / `target` AgentIds) is fully exercised by the unit
//!   tests in `engine::ability::apply::tests`; only the GPU↔CPU
//!   record-byte comparison for that path is deferred.
//!
//! ## Skip path
//!
//! The test gates on adapter availability — when no wgpu adapter is
//! present (CI without a software-rendering fallback), prints a skip
//! message and returns Ok. The compile-time path still validated the
//! registry construction + the dispatcher kernel emit.

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use dsl_compiler::cpu_chronicle_reference::apply_event_to_chronicle_record;
use engine::ability::apply::apply_program;
use engine::ability::program::{
    BuffStat, CasterStats, EffectOp, EffectPredicate, EffectPredicateBinder, EffectPredicateOp,
    EffectScaling, EffectWhenCondition, Gate, ScalingStatRef,
};
use engine::ability::{AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder};
use engine::ids::AgentId;

/// Pinned RNG seed for the sweep — `world_seed` flows into the chance
/// gate's `per_agent_u32` derivation. Pin same value on both backends
/// (the chance modifier is documented-deferred above; in practice the
/// only chance-gated arm in this matrix uses q16=0xFFFE which fires
/// deterministically regardless of the draw).
const WORLD_SEED: u64 = 0xC0FFEE_u64;

/// Sweep ticks. Five values across the u32 range surface any
/// wraparound bug in `expires_at_tick = tick + duration` for the
/// duration-bearing arms (Stun / LifeSteal / DamageModify).
const TICKS: &[u32] = &[0, 17, 100, 1000, 65500];

// (caster, target) pairs are restricted to self-cast only in this
// fixture — see the N_AGENTS doc-comment below for the explanation
// of why distinct caster≠target requires a different fixture surface.

/// Number of agents per dispatch invocation.
///
/// The smoke fixture's `apply_ability` source uses the implicit-target
/// rule (no `by ... target ...` operands), so the dispatcher writes
/// `caster_slot == target_slot == agent_id` for every alive agent in
/// the workgroup — every cast is structurally a self-cast. We
/// therefore use a 1-agent SoA per dispatch (`caster_slot = 0`,
/// `target_slot = 0`) and rebuild the fixture once per ability ×
/// tick, rewriting `agent_level[0]` to the ability under test before
/// each `step()`.
///
/// Distinct caster≠target requires an explicit-target lowering
/// (`CgStmt::ApplyAbility { target: distinct_expr }`) — not present
/// in the smoke fixture's compiled .sim source today. Deferred to a
/// follow-on fixture; see the module docs' "What's deferred" section.
const N_AGENTS: u32 = 1;

/// Total cast count printed in the coverage report. With self-cast
/// only and N_AGENTS = 1, K = N_ABILITIES × T (one cast per ability
/// per tick, all agent 0 self-cast).
fn coverage_k(n_abilities: usize) -> usize {
    n_abilities * TICKS.len()
}

/// Build the 10-ability sweep registry. Each entry returns
/// `(name, AbilityProgram, CasterStats)` — the stat snapshot is the
/// caster's stats for THIS ability's apply_program call. Same snapshot
/// is uploaded into the GPU's per-stat SoA for the cast.
///
/// Some abilities require non-default stats (e.g. Bleed-shape needs
/// `caster.max_hp = 100` to make the `+ 5% MaxHp` scaling produce a
/// non-zero bonus; Reap-shape needs `target.hp = 5` to pass the
/// `target.hp < 20` predicate — and since the smoke fixture uses
/// self-cast, target == caster ⇒ caster.hp = 5).
fn build_sweep() -> Vec<(&'static str, AbilityProgram, CasterStats)> {
    let mut out: Vec<(&'static str, AbilityProgram, CasterStats)> = Vec::new();

    // 1. Strike-shape — no modifiers.
    out.push((
        "Strike",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        ),
        CasterStats::default(),
    ));

    // 2. Mend-shape — Heal.
    out.push((
        "Mend",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 30, hostile_only: false, line_of_sight: false },
            [EffectOp::Heal { amount: 25.0 }],
        ),
        CasterStats::default(),
    ));

    // 3. ShieldUp-shape — Shield.
    out.push((
        "ShieldUp",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 40, hostile_only: false, line_of_sight: false },
            [EffectOp::Shield { amount: 50.0 }],
        ),
        CasterStats::default(),
    ));

    // 4. Bleed-shape — SelfDamage(5) + 5% MaxHp scaling. Caster has
    //    max_hp=100 → scale_bonus = 0.05 * 100 = 5.0 → emitted
    //    SelfDamage amount = 5.0 + 5.0 = 10.0.
    let mut bleed = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 50, hostile_only: false, line_of_sight: false },
        [EffectOp::SelfDamage { amount: 5.0 }],
    );
    let mut bleed_inner: smallvec::SmallVec<[EffectScaling; 2]> = smallvec::SmallVec::new();
    bleed_inner.push(EffectScaling { stat_ref: ScalingStatRef::MaxHp, percent: 0.05 });
    bleed.scalings_per_effect.push(bleed_inner);
    out.push((
        "Bleed",
        bleed,
        CasterStats { max_hp: 100.0, ..Default::default() },
    ));

    // 5. Reap-shape — Execute(20) when target.hp < 20. Self-cast
    //    fixture: caster == target, so caster.hp = 5 makes the
    //    predicate fire. The CPU oracle reads `target_stats.hp`; we
    //    pass the same CasterStats for both caster and target slots
    //    in the apply_program call (mirrors GPU's `pred_agent =
    //    target_slot` reading the same agent_hp column for self-cast).
    let mut reap = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
        [EffectOp::Execute { hp_threshold: 20.0 }],
    );
    reap.when_per_effect.push(Some(EffectWhenCondition {
        when_cond:     "target.hp < 20".to_string(),
        else_cond:     None,
        when_compiled: Some(EffectPredicate {
            binder:  EffectPredicateBinder::Target,
            field:   ScalingStatRef::Hp.discriminant(),
            op:      EffectPredicateOp::Lt,
            literal: 20.0,
        }),
    }));
    out.push((
        "Reap",
        reap,
        CasterStats { hp: 5.0, ..Default::default() },
    ));

    // 6. Daze-shape — chance=0xFFFE near-deterministic. Damage(10)
    //    gated on always-fires q16. Both backends use the same PCG
    //    mixer (`per_agent_u32_pcg_with_extra` / WGSL prelude) keyed
    //    on `(seed, caster_slot, tick, RngPurpose::Chance=10,
    //    slot_idx=0)`, so draw values match bit-for-bit. The 1/65536
    //    miss case (draw == 0xFFFE or 0xFFFF) is identical on both
    //    sides — the count would still match (both miss together).
    let mut daze = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 40, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 10.0 }],
    );
    daze.chances.push(Some(0xFFFE));
    out.push((
        "Daze",
        daze,
        CasterStats::default(),
    ));

    // 7. Reap+Stun-shape — Damage(40) { stun 1s } nested. Primary
    //    Damage emits one record, nested Stun emits another with
    //    expires_at_tick = tick + 10.
    let mut nested_dmg = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 40.0 }],
    );
    let mut nested_inner: smallvec::SmallVec<[EffectOp; 2]> = smallvec::SmallVec::new();
    nested_inner.push(EffectOp::Stun { duration_ticks: 10 });
    nested_dmg.nested_per_effect.push(nested_inner);
    out.push((
        "DamageWithStun",
        nested_dmg,
        CasterStats::default(),
    ));

    // 8. Heal + 50% AbilityPower. AbilityPower has no agent SoA on
    //    GPU — scale_bonus = 0.0 there. CPU's PerAgentStats default
    //    sets ability_power = 0.0 → 0.5 * 0.0 = 0.0. Both backends
    //    must produce Heal(8.0) — pin via this arm.
    let mut heal_ap = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: false, line_of_sight: false },
        [EffectOp::Heal { amount: 8.0 }],
    );
    let mut ap_inner: smallvec::SmallVec<[EffectScaling; 2]> = smallvec::SmallVec::new();
    ap_inner.push(EffectScaling { stat_ref: ScalingStatRef::AbilityPower, percent: 0.50 });
    heal_ap.scalings_per_effect.push(ap_inner);
    out.push((
        "HealAP",
        heal_ap,
        CasterStats::default(),
    ));

    // 9. Vampirize-shape — LifeSteal(0.5 q8=128 / 5s = 50 ticks).
    out.push((
        "Vampirize",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 80, hostile_only: false, line_of_sight: false },
            [EffectOp::LifeSteal { duration_ticks: 50, fraction_q8: 128 }],
        ),
        CasterStats::default(),
    ));

    // 10. Fortify-shape — DamageModify(0.5 q8=128 / 5s = 50 ticks).
    out.push((
        "Fortify",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 80, hostile_only: false, line_of_sight: false },
            [EffectOp::DamageModify { duration_ticks: 50, multiplier_q8: 128 }],
        ),
        CasterStats::default(),
    ));

    // Wave 2 piece 1 — control statuses (Root/Silence/Fear/Taunt).
    // Each shares Stun's shape: 3-payload-word chronicle record
    // (actor + target + expires_at_tick = tick + duration). Adding all
    // four extends the sweep matrix to 14 abilities so the four new
    // GPU dispatcher arms get parity-pinned alongside the existing
    // Stun arm.

    // 11. Stop-shape — Root.
    out.push((
        "Stop",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::Root { duration_ticks: 30 }],
        ),
        CasterStats::default(),
    ));

    // 12. Mute-shape — Silence.
    out.push((
        "Mute",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::Silence { duration_ticks: 30 }],
        ),
        CasterStats::default(),
    ));

    // 13. Terrify-shape — Fear.
    out.push((
        "Terrify",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::Fear { duration_ticks: 30 }],
        ),
        CasterStats::default(),
    ));

    // 14. Provoke-shape — Taunt.
    out.push((
        "Provoke",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::Taunt { duration_ticks: 30 }],
        ),
        CasterStats::default(),
    ));

    // Wave 2 piece 2 — movement EffectOps (Dash/Blink/Knockback/Pull).
    // Two distinct shapes:
    //   - Dash/Blink: caster-self motion. Engine event has no target
    //     field; the chronicle record stores distance at payload word
    //     1 (= ring slot offset 3).
    //   - Knockback/Pull: forced motion on a target. 3-payload-word
    //     record: actor + target + distance at ring slot offset 4.
    // Adding all four extends the sweep matrix to 18 abilities so the
    // four new GPU dispatcher arms get parity-pinned.

    // 15. Lunge-shape — Dash (caster-self motion).
    out.push((
        "Lunge",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 30, hostile_only: false, line_of_sight: false },
            [EffectOp::Dash { distance: 12.0 }],
        ),
        CasterStats::default(),
    ));

    // 16. Phase-shape — Blink (caster-self motion).
    out.push((
        "Phase",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 30, hostile_only: false, line_of_sight: false },
            [EffectOp::Blink { distance: 8.0 }],
        ),
        CasterStats::default(),
    ));

    // 17. Smash-shape — Knockback (forced motion on target).
    out.push((
        "Smash",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
            [EffectOp::Knockback { distance: 5.0 }],
        ),
        CasterStats::default(),
    ));

    // 18. Hook-shape — Pull (forced motion on target).
    out.push((
        "Hook",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
            [EffectOp::Pull { distance: 3.0 }],
        ),
        CasterStats::default(),
    ));

    // Wave 1.5+ — multi-tick effects (DoT/HoT/TimedShield). All three
    // share the same 5-payload-word chronicle shape: actor + target +
    // amount (bitcast f32 → u32 at slot 4) + duration_ticks (raw u32
    // at slot 5). Adding all three extends the sweep matrix to 21
    // abilities so the three new GPU dispatcher arms get parity-pinned.

    // 19. Burn-shape — DamageOverTime (per-tick damage over a window).
    out.push((
        "Burn",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::DamageOverTime { amount: 6.0, duration_ticks: 30 }],
        ),
        CasterStats::default(),
    ));

    // 20. Regen-shape — HealOverTime (per-tick healing over a window).
    out.push((
        "Regen",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
            [EffectOp::HealOverTime { amount: 4.0, duration_ticks: 50 }],
        ),
        CasterStats::default(),
    ));

    // 21. Aegis-shape — TimedShield (one-shot shield amount over a
    // window). Same payload shape as DoT/HoT but `amount` is the
    // one-shot magnitude rather than per-tick.
    out.push((
        "Aegis",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
            [EffectOp::TimedShield { amount: 25.0, duration_ticks: 100 }],
        ),
        CasterStats::default(),
    ));

    // Extended-corpus statuses (Stealth/Charm/Grounded/Suppress). Two
    // distinct shapes:
    //   - Stealth: caster-self status. Engine event has no target field;
    //     the chronicle record stores duration_ticks at payload word 1
    //     (= ring slot offset 3). Same family as Dash/Blink.
    //   - Charm/Grounded/Suppress: target-cast statuses. 3-payload-word
    //     record: actor + target + duration_ticks at ring slot offset 4.
    // Adding all four extends the sweep matrix to 25 abilities so the
    // four new GPU dispatcher arms get parity-pinned.

    // 22. Vanish-shape — Stealth (caster-self status).
    out.push((
        "Vanish",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
            [EffectOp::Stealth { duration_ticks: 50 }],
        ),
        CasterStats::default(),
    ));

    // 23. Allure-shape — Charm (target-cast status).
    out.push((
        "Allure",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::Charm { duration_ticks: 30 }],
        ),
        CasterStats::default(),
    ));

    // 24. Tether-shape — Grounded (target-cast status).
    out.push((
        "Tether",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::Grounded { duration_ticks: 25 }],
        ),
        CasterStats::default(),
    ));

    // 25. Hush-shape — Suppress (target-cast status).
    out.push((
        "Hush",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::Suppress { duration_ticks: 40 }],
        ),
        CasterStats::default(),
    ));

    // Slice γ tail — Buff/Harvest/PlaceVoxel/Reflect. Four distinct
    // shapes:
    //   - Buff: target-cast with packed payload (stat | mag_q8 << 8 | duration).
    //     Negative magnitude_q8 exercises the i16 → i32 → u32 sign-cast.
    //   - Reflect: target-cast with packed payload (duration | fraction_q8 in
    //     payload_b's low 16 bits). Negative fraction_q8 exercises the
    //     i16 → u16 → u32 zero-extend through low 16 bits.
    //   - Harvest: caster-self (kind_hash + amount). No target field.
    //   - PlaceVoxel: caster-self (kind_hash). Position implicit.
    // Adding all four extends the sweep matrix to 29 abilities.

    // 26. Empower-shape — Buff (target-cast, packed signed magnitude).
    //     Negative magnitude_q8 exercises the sign-cast path on both
    //     CPU and GPU sides — the chronicle bytes round-trip iff
    //     `pack_effect`'s OR of `(stat as u32) | ((mag as i32 as u32) << 8)`
    //     matches the dispatcher's raw payload_a write.
    out.push((
        "Empower",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
            [EffectOp::Buff {
                stat: BuffStat::AttackSpeed,
                magnitude_q8: -64,
                duration_ticks: 50,
            }],
        ),
        CasterStats::default(),
    ));

    // 27. Mirror-shape — Reflect (target-cast, packed signed fraction).
    //     Negative fraction_q8 exercises the i16 → u16 → u32 zero-extend
    //     path; consumer rules sign-extend low 16 bits to recover.
    out.push((
        "Mirror",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
            [EffectOp::Reflect { duration_ticks: 50, fraction_q8: -64 }],
        ),
        CasterStats::default(),
    ));

    // 28. Reap_Ore-shape — Harvest (caster-self).
    out.push((
        "Reap_Ore",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
            [EffectOp::Harvest { kind_hash: 0xCAFEBABE, amount: 5 }],
        ),
        CasterStats::default(),
    ));

    // 29. Drop_Stone-shape — PlaceVoxel (caster-self).
    out.push((
        "Drop_Stone",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
            [EffectOp::PlaceVoxel { kind_hash: 0xFACEFEED }],
        ),
        CasterStats::default(),
    ));

    // 30. DazeMid-shape — Damage(10) chance=0x4000 (~25%). P11
    //     intermediate-chance arm: the CPU oracle's
    //     `per_agent_u32_pcg_with_extra` draw and the GPU
    //     dispatcher's `per_agent_u32_with_extra` draw must agree
    //     bit-for-bit on whether the chronicle record fires for
    //     each (seed, caster_slot, tick, slot_idx=0) tuple. With
    //     5 sweep ticks under uniform draws, expected fires ≈ 1.25
    //     and the count for our pinned WORLD_SEED is exactly the
    //     count both backends reach (any divergence surfaces as a
    //     count mismatch panic from the byte-equal check).
    let mut daze_mid = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 40, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 10.0 }],
    );
    daze_mid.chances.push(Some(0x4000));
    out.push((
        "DazeMid",
        daze_mid,
        CasterStats::default(),
    ));

    // 31. Cleave-shape — `Damage(30) in circle(2.0)` (#121 AOE Path B).
    //     The single AOE entry in the sweep, exercising the Path B
    //     27-cell spatial walk + per-target chronicle write the
    //     dispatcher emits when `with_aoe_dispatch == true`. Driven by
    //     a 4-agent fixture (positions in a row at x=0, 1.5, 3.0, 4.5)
    //     with only agent 0 alive — the dispatcher fires once on
    //     caster=slot 0, the spatial walk reads `agent_pos[0]` as the
    //     center, and the in-radius set (≤2.0) is {slot 0 (d=0),
    //     slot 1 (d=1.5)}. CPU oracle calls `apply_program_aoe` with
    //     the same {0, 1} set; both backends emit 2 chronicle records
    //     (kind=26 EffectDamageApplied, target=0 + target=1).
    //
    //     The test runner detects the "Cleave" name and routes through
    //     the 4-agent fixture path (`run_cleave_parity_iteration`)
    //     instead of the default 1-agent self-cast path. The CPU
    //     oracle for Cleave goes through `apply_program_aoe` with the
    //     CPU-determined in-radius set, mirroring what the GPU walk
    //     produces structurally (P11 sort handles atomicAdd-induced
    //     ring order non-determinism on readback).
    let mut cleave = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 30.0 }],
    );
    cleave.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Circle,
        args: [2.0, 0.0, 0.0, 0.0],
    }));
    out.push((
        "Cleave",
        cleave,
        CasterStats::default(),
    ));

    out
}

/// Pack the registry once — assigns AbilityId(1..=N) in registration
/// order. Mirrors `build_duel_abilities_registry`'s shape but for the
/// sweep test's bespoke synthetic abilities (no .ability files).
fn build_registry(sweep: &[(&'static str, AbilityProgram, CasterStats)]) -> AbilityRegistry {
    let mut builder = AbilityRegistryBuilder::new();
    for (name, prog, _stats) in sweep {
        let id = builder.register(prog.clone());
        debug_assert_ne!(id.raw(), 0, "AbilityId is 1-based: {name}");
    }
    builder.build()
}

/// Convert CasterStats → PerAgentStats. Re-uploads the CPU oracle's
/// snapshot to the GPU SoA so both backends read the same f32 values
/// at `agent_<stat>[caster_slot]` (GPU) vs `caster_stats.<stat>` (CPU).
fn per_agent_from_caster_stats(s: &CasterStats) -> PerAgentStats {
    PerAgentStats {
        attack_damage: s.attack_damage,
        ability_power: s.ability_power,
        max_hp:        s.max_hp,
        hp:            s.hp,
        armor:         s.armor,
        magic_resist:  s.magic_resist,
        move_speed:    s.move_speed,
        mana:          s.mana,
    }
}

/// Run apply_program on CPU for a single self-cast, return the
/// expected chronicle records (in CPU emit order).
fn cpu_records_for_cast(
    program:      &AbilityProgram,
    caster_slot:  u32,
    target_slot:  u32,
    tick:         u32,
    caster_stats: &CasterStats,
    target_stats: &CasterStats,
) -> Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> {
    let caster = AgentId::new(caster_slot + 1).expect("caster_slot+1 non-zero");
    let target = AgentId::new(target_slot + 1).expect("target_slot+1 non-zero");
    let events = apply_program(
        program,
        caster,
        target,
        tick as u64,
        WORLD_SEED,
        caster_stats,
        target_stats,
    );
    let mut out = Vec::with_capacity(events.len());
    for ev in events {
        if let Some(rec) =
            apply_event_to_chronicle_record(ev, tick, caster_slot, target_slot)
        {
            out.push(rec);
        }
    }
    out
}

/// CPU oracle for the AOE Cleave entry. Calls
/// `apply_program_aoe(caster=slot+1, primary_target=slot+1,
/// aoe_targets=<id-set>)` and emits one chronicle record per
/// in-circle target. Mirrors the smoke fixture's implicit-target rule
/// (caster == primary_target == slot 0); the AOE expansion happens
/// inside `apply_program_aoe`'s per-target loop. The host pre-computes
/// the in-radius set (the dispatcher walks the spatial grid; the test
/// pins the expected set against agent positions), so this helper
/// receives the slot ids as `aoe_target_slots`.
///
/// **P11 sort.** Each chronicle record carries the ApplyEvent's
/// `target` field (1-based AgentId) → 0-based slot via `target_id =
/// raw - 1`. The canonicalize sort happens at the test runner; this
/// helper just emits records in `aoe_target_slots` order.
fn cleave_cpu_records_for_cast(
    program:           &AbilityProgram,
    caster_slot:       u32,
    aoe_target_slots:  &[u32],
    tick:              u32,
    caster_stats:      &CasterStats,
) -> Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> {
    use engine::ability::apply::apply_program_aoe;

    let caster = AgentId::new(caster_slot + 1).expect("caster_slot+1 non-zero");
    let primary_target = caster; // self-cast convention (smoke fixture)
    let aoe_targets: Vec<AgentId> = aoe_target_slots
        .iter()
        .map(|s| AgentId::new(s + 1).expect("aoe_target_slot+1 non-zero"))
        .collect();

    let events = apply_program_aoe(
        program,
        caster,
        primary_target,
        &aoe_targets,
        tick as u64,
        WORLD_SEED,
        caster_stats,
        /*target_stats*/ caster_stats,
    );
    let mut out = Vec::with_capacity(events.len());
    for ev in events {
        // Each ApplyEvent carries the per-target AgentId; pull it back
        // out as a slot for the chronicle record's target field. The
        // ApplyEvent kinds we care about (`Damage`) carry `target`
        // explicitly; pattern-match exhaustively so a future variant
        // gets the per-target conversion treatment.
        let target_slot = match &ev {
            engine::ability::apply::ApplyEvent::Damage { target, .. } => target.raw() - 1,
            engine::ability::apply::ApplyEvent::Heal { target, .. } => target.raw() - 1,
            engine::ability::apply::ApplyEvent::Shield { target, .. } => target.raw() - 1,
            engine::ability::apply::ApplyEvent::Stun { target, .. } => target.raw() - 1,
            engine::ability::apply::ApplyEvent::Slow { target, .. } => target.raw() - 1,
            // Other variants stay un-AOE'd today (Cleave is the only
            // AOE entry); fall through to the caster_slot if a future
            // variant lands without an explicit target field.
            _ => caster_slot,
        };
        if let Some(rec) =
            apply_event_to_chronicle_record(ev, tick, caster_slot, target_slot)
        {
            out.push(rec);
        }
    }
    out
}

/// Sort records by `(kind, payload_a, payload_b, payload_c, payload_d)`
/// for order-stable comparison. The GPU dispatcher uses atomicAdd to
/// claim ring slots — record order in `event_ring` is workgroup-
/// schedule-dependent, so canonicalising both sides by content gives
/// us the byte-equality property without depending on slot ordering.
///
/// `payload_a..d` cover slots 4..7 of the 10-word record (header is
/// kind + tick at slots 0+1, then actor + target at slots 2+3, then
/// payload at 4..). Including 4 payload words in the sort key keeps
/// records distinct for variants that share kind+actor+target+amount
/// but differ in expires_at_tick or fraction_q8 (e.g. two Slows on
/// the same target with different durations would otherwise collide).
fn canonicalize(records: &mut Vec<[u32; CHRONICLE_STRIDE_U32 as usize]>) {
    records.sort_by_key(|r| (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]));
}

/// Pretty-print a record for diagnostic output.
fn fmt_record(r: &[u32; CHRONICLE_STRIDE_U32 as usize]) -> String {
    format!(
        "[kind={} tick={} actor={} target={} p4=0x{:08x} p5=0x{:08x} p6=0x{:08x} p7=0x{:08x} p8=0x{:08x} p9=0x{:08x}]",
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
    )
}

#[test]
fn cpu_gpu_apply_program_byte_equal_across_modifier_matrix() {
    let sweep = build_sweep();
    let registry = build_registry(&sweep);
    let n_abilities = sweep.len();
    let k_total = coverage_k(n_abilities);

    let mut total_cpu_records = 0usize;
    let mut total_gpu_records = 0usize;
    let mut adapter_skipped = false;

    for (ability_idx, (name, _prog, caster_stats)) in sweep.iter().enumerate() {
        let ability_id = (ability_idx + 1) as u32; // 1-based
        let program = registry
            .get(AbilityId::new(ability_id).expect("non-zero id"))
            .unwrap_or_else(|| panic!("ability {name} not registered"));

        // #121 AOE Path B: the Cleave entry exercises the
        // dispatcher's spatial walk + per-target chronicle write,
        // which requires N≥2 agents in the spatial grid (one caster
        // + one in-circle target). Route through the dedicated
        // 4-agent fixture path; every other entry uses the default
        // 1-agent self-cast path.
        let n_agents_for_ability = if *name == "Cleave" { 4 } else { N_AGENTS };

        // GPU side seeds slot 0 with this ability's per-agent stats.
        // Self-cast convention: target_stats = caster_stats (CPU
        // oracle reads target_stats for `target.<field>` predicates;
        // the smoke fixture's implicit-target rule means target_slot
        // == caster_slot ⇒ same SoA row ⇒ same f32 stat value).
        let per_agent_stats =
            vec![per_agent_from_caster_stats(caster_stats); n_agents_for_ability as usize];
        let per_agent_levels = vec![ability_id; n_agents_for_ability as usize];

        let state = match ApplyAbilitySmokeState::try_new_with_registry(
            n_agents_for_ability,
            &registry,
            &per_agent_levels,
            &per_agent_stats,
        ) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[parity_apply_program_sweep] skipping: no wgpu adapter \
                     available on this host. The compile path still validated \
                     registry construction + dispatcher kernel emit."
                );
                adapter_skipped = true;
                break;
            }
        };
        let mut state = state;

        // For the Cleave entry, configure the 4-agent fixture: row of
        // positions (0, 1.5, 3.0, 4.5) on the x-axis; only agent 0
        // alive (it's the caster); the spatial grid was pre-populated
        // by the constructor (every slot in cell 0). The caster's AOE
        // walk reads `agent_pos[0] = (0,0,0)` as the center and
        // collects every agent within the 2.0 radius — agents 0 (d=0)
        // and 1 (d=1.5) under our positions; agents 2 (d=3.0) and 3
        // (d=4.5) are out of range.
        if *name == "Cleave" {
            state.set_agent_alive(&[1, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
            ]);
        }

        for &tick in TICKS {
            // -- CPU oracle.
            let mut cpu = if *name == "Cleave" {
                cleave_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[0, 1],
                    tick,
                    caster_stats,
                )
            } else {
                cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*target_slot*/ 0,
                    tick,
                    caster_stats,
                    /*target_stats*/ caster_stats,
                )
            };
            canonicalize(&mut cpu);

            // -- GPU dispatch. Pin the GPU's cfg.seed to the same
            //    `world_seed as u32` the CPU oracle keys on, so the
            //    chance-gate PCG draws agree bit-for-bit (P11).
            state.step_with_seed(tick, WORLD_SEED as u32);
            let tail = state.read_event_tail();
            let mut gpu = state.read_event_ring(tail);
            canonicalize(&mut gpu);

            total_cpu_records += cpu.len();
            total_gpu_records += gpu.len();

            // -- Byte-equal assert.
            if cpu.len() != gpu.len() {
                panic!(
                    "[parity {name} tick={tick}] record count mismatch: \
                     cpu={} gpu={}\n  cpu records: {}\n  gpu records: {}",
                    cpu.len(),
                    gpu.len(),
                    cpu.iter().map(fmt_record).collect::<Vec<_>>().join("\n    "),
                    gpu.iter().map(fmt_record).collect::<Vec<_>>().join("\n    "),
                );
            }
            for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                if c != g {
                    panic!(
                        "[parity {name} tick={tick}] record {i} bytes diverged:\n  \
                         CPU: {}\n  GPU: {}\n  \
                         (kind cpu={} gpu={}, payload[4] cpu=0x{:08x} gpu=0x{:08x}, \
                         payload[5] cpu=0x{:08x} gpu=0x{:08x})",
                        fmt_record(c),
                        fmt_record(g),
                        c[0], g[0], c[4], g[4], c[5], g[5],
                    );
                }
            }
        }
    }

    if adapter_skipped {
        return;
    }

    eprintln!(
        "Apply parity sweep: {n_abilities} abilities × {pairs} caster×target × \
         {ticks} ticks = {k_total} casts, {total_gpu_records} chronicle records, \
         all byte-equal (vs CPU oracle {total_cpu_records} records).",
        pairs = 1, // self-cast only — distinct caster/target deferred (see module docs)
        ticks = TICKS.len(),
    );

    assert_eq!(
        total_cpu_records, total_gpu_records,
        "global record count must match across all sweep iterations",
    );
}
