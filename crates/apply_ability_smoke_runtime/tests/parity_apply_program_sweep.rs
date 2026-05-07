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
//!  31. `Damage(30) in circle(2.0)`                  — Cleave-shape (#121 AOE Path B Circle, 4-agent row)
//!  32. `Damage(25) in cone(60°, 4)`                 — Slash-shape (#178 AOE Path B Cone, 5-agent fan, degenerate self-cast)
//!  33. `Damage(20) in box(1.5, 1.5, 1.5)`           — Pulverize-shape (#179 AOE Path B Box, 4-agent row, wall-inclusive)
//!  34. `Damage(18) in sphere(2.0)`                  — BlastSphere-shape (#180 AOE Path B Sphere, 4-agent row, alias of Circle)
//!  35. `Damage(14) in ring(0.5, 2.0)`               — ShockwaveRing-shape (#180 AOE Path B Ring, 4-agent row, inner-excludes slot 0)
//!  36. `Damage(22) in line(5.0, 1.0)`               — PiercingLine-shape (#180 AOE Path B Line, 4-agent row, degenerate self-cast)
//!  43. `Damage(25) in cone(45°, 5)`                  — NonDegSlash-shape (#182 explicit-target Cone, 5-agent fan, +X direction)
//!  44. `Damage(22) in line(5.0, 1.0)`                — NonDegPiercingLine-shape (#182 explicit-target Line, 4-agent fixture, +X direction)
//!  45. `Damage(11) in wall(4, 2, 2, 0°)`             — NonDegShieldWall-shape (#182 explicit-target Wall, 4-agent fixture, slab at target)
//!  46. `Summon(0xDEADBEEF, 3, 120)`                  — EvocationSummon-shape (slice γ closer, caster-self packed payload)
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

    // 32. Slash-shape — `Damage(25) in cone(60°, 4)` (#178 AOE Path B Cone).
    //     Exercises the dispatcher's cone branch (`area_kind == 1u`):
    //     27-cell walk around the apex (caster), per-candidate range²
    //     gate + angular dot gate, shadowed `target_slot = candidate`.
    //
    //     **Smoke fixture self-cast caveat.** The smoke fixture's
    //     implicit-target rule writes `caster_slot == target_slot ==
    //     agent_id` per the per-agent loop, so `agent_pos[caster_slot]
    //     == agent_pos[target_slot]` and the cone's `direction_raw =
    //     target - apex` is always (0,0,0) → degenerate. The GPU
    //     kernel's `dir_len_sq < 1e-6` branch skips the spatial walk;
    //     the CPU oracle's `apply_program_aoe_cone_filter` returns
    //     empty in the same condition. Both backends emit 0 chronicle
    //     records — byte-equal at the trivial-zero level.
    //
    //     The non-degenerate cone math is exercised on the CPU via
    //     `engine::tests::aoe_multi_agent_e2e::aoe_cone_hits_three_in_fan_*`
    //     and the unit tests in `engine::ability::apply::tests` (5-agent
    //     arc, apex exclusion, range cutoff). The GPU emit is validated
    //     here at the WGSL-compile level (the kernel must parse and
    //     dispatch without error even when the cone branch runs); a
    //     follow-on fixture with an explicit-target `apply_ability AID
    //     by self target other` lowering would unlock N>0 GPU cone
    //     records (deferred — same surface as the Cleave-shape note
    //     above and the module-level "What's deferred" section).
    let mut slash = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 25.0 }],
    );
    slash.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Cone,
        args: [60.0, 4.0, 0.0, 0.0],
    }));
    out.push((
        "Slash",
        slash,
        CasterStats::default(),
    ));

    // 33. Pulverize-shape — `Damage(20) in box(1.5, 1.5, 1.5)` (#179 AOE
    //     Path B Box). Exercises the dispatcher's box branch
    //     (`area_kind == 5u`): 27-cell walk around the center
    //     (`agent_pos[target_slot]` — same convention as Circle), per-
    //     candidate AABB containment gate (`|d.<axis>| ≤ w<axis>`),
    //     shadowed `target_slot = candidate`.
    //
    //     Driven by the same 4-agent fixture as Cleave (positions in a
    //     row at x=0, 1.5, 3.0, 4.5). With caster at slot 0
    //     (target_slot==caster_slot under the smoke fixture's
    //     implicit-target rule), the box is centered at (0,0,0) with
    //     half-extents (1.5, 1.5, 1.5). The closed-AABB semantic (≤,
    //     not <) means agent 1 at (1.5, 0, 0) is exactly at the +x
    //     wall and IS in-box; agents 2 (3.0) and 3 (4.5) are out.
    //
    //     **Spatial walk constraint.** wx/wy/wz = 1.5 ≤ cell_size = 6.0
    //     so the 27-cell walk visits every cell that could hold an
    //     in-box candidate. Larger extents would require additional
    //     cell rings — kept out of scope for this fixture (see the
    //     box branch's WGSL doc-comment for the limitation).
    let mut pulverize = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 20.0 }],
    );
    pulverize.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Box,
        args: [1.5, 1.5, 1.5, 0.0],
    }));
    out.push((
        "Pulverize",
        pulverize,
        CasterStats::default(),
    ));

    // 34. BlastSphere-shape — `Damage(18) in sphere(2.0)` (#180 AOE
    //     Path B Sphere). Sphere is mathematically equivalent to
    //     Circle today (3D dist² ≤ radius²), but routes through the
    //     dispatcher's dedicated Sphere branch (`area_kind == 6u`).
    //     Same 4-agent row fixture as Cleave; in-radius set under the
    //     smoke fixture's self-cast = {slot 0 (d=0), slot 1 (d=1.5)}
    //     under radius=2.0. Both backends emit 2 chronicle records.
    let mut blast_sphere = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 18.0 }],
    );
    blast_sphere.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Sphere,
        args: [2.0, 0.0, 0.0, 0.0],
    }));
    out.push((
        "BlastSphere",
        blast_sphere,
        CasterStats::default(),
    ));

    // 35. ShockwaveRing-shape — `Damage(14) in ring(0.5, 2.0)` (#180
    //     AOE Path B Ring). Annulus gate excludes the inner radius:
    //     under the 4-agent row (x = 0, 1.5, 3.0, 4.5) with caster at
    //     slot 0, distances are 0, 1.5, 3.0, 4.5. Inner=0.5 excludes
    //     slot 0 (d=0 < 0.5); outer=2.0 admits slot 1 (d=1.5 ≤ 2.0)
    //     and excludes slots 2/3 (d > 2.0). Expected hit set = {slot
    //     1} — one chronicle record. Validates the inner-radius-
    //     exclusion semantic byte-equal across CPU/GPU.
    let mut shockwave_ring = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 14.0 }],
    );
    shockwave_ring.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Ring,
        args: [0.5, 2.0, 0.0, 0.0],
    }));
    out.push((
        "ShockwaveRing",
        shockwave_ring,
        CasterStats::default(),
    ));

    // 36. PiercingLine-shape — `Damage(22) in line(5.0, 1.0)` (#180 AOE
    //     Path B Line). Smoke-fixture self-cast invariant: `caster_slot
    //     == target_slot`, so `apex == target_pos == agent_pos[0]`.
    //     Direction = target - apex = (0,0,0) is degenerate; the WGSL
    //     kernel's `dir_len_sq < 1e-6 → no-op` branch skips the spatial
    //     walk, the CPU oracle's `apply_program_aoe_line_filter`
    //     returns empty in the same condition. Both backends emit 0
    //     chronicle records — byte-equal at the trivial-zero level
    //     (mirrors Slash's degenerate-cone semantic in entry 32).
    //
    //     Non-degenerate line math is exercised on CPU via
    //     `engine::tests::aoe_multi_agent_e2e::aoe_line_hits_along_corridor_*`
    //     and the unit tests in `engine::ability::apply::tests`.
    let mut piercing_line = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 22.0 }],
    );
    piercing_line.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Line,
        args: [5.0, 1.0, 0.0, 0.0],
    }));
    out.push((
        "PiercingLine",
        piercing_line,
        CasterStats::default(),
    ));

    // 37. PickFew-shape — `Damage(15) in spread(2.0, 2)` (#181 AOE Path B
    //     Spread, GPU-emitted Wave 1.6 #183). Spread is Circle gate +
    //     sort by AgentId ascending + truncate to max_targets. With the
    //     4-agent row fixture (x=0, 1.5, 3.0, 4.5) self-cast at slot 0,
    //     the in-radius set under radius=2.0 is {slot 0 (d=0), slot 1
    //     (d=1.5)} — slots 2/3 are outside. max_targets=2 keeps both
    //     after sort, so the GPU's per-thread sort+truncate must agree
    //     with the CPU oracle's `apply_program_aoe_spread_filter`
    //     post-cap list `[slot 0, slot 1]`. Two chronicle records,
    //     byte-equal across backends.
    //
    //     The `args[1] = 2.0` literal is stored as f32 in the registry
    //     (per-effect area_args is `[f32; 4]`); the WGSL emit casts it
    //     to u32 via `u32(area_args[base + 1])`. CPU oracle reads the
    //     `EffectAreaShape::args[1]` slot as `u32` directly via
    //     `as u32` (matching the WGSL truncation semantics for
    //     non-negative finite f32s — see `apply_program_aoe_spread_filter`
    //     callers).
    //
    //     16-slot cap caveat: this fixture's 4 candidates fit comfortably
    //     under the per-thread `array<u32, 16>` collection buffer; no
    //     pre-sort overflow is exercised here. Fixtures targeting > 16
    //     simultaneous in-radius candidates per cast must keep
    //     n_in_radius ≤ 16 to stay byte-equal across backends.
    let mut pick_few = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 15.0 }],
    );
    pick_few.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Spread,
        args: [2.0, 2.0, 0.0, 0.0],
    }));
    out.push((
        "PickFew",
        pick_few,
        CasterStats::default(),
    ));

    // 38. TallStomp-shape — `Damage(13) in column(2.0, 4.0)` (#181 AOE
    //     Path B Column). Vertical cylinder extending UP from the cast
    //     center (XZ disc + `0 ≤ dy ≤ height` gate). Under the smoke
    //     fixture's 4-agent row at (0, 1.5, 3.0, 4.5) on the X-axis (all
    //     at y=0), Column(2, 4) at caster slot 0 hits slot 0 (origin,
    //     y=0 on plane) and slot 1 (XZ d=1.5, y=0 on plane). Slot 2/3
    //     are outside the XZ radius. Both backends emit 2 chronicle
    //     records.
    let mut tall_stomp = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 13.0 }],
    );
    tall_stomp.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Column,
        args: [2.0, 4.0, 0.0, 0.0],
    }));
    out.push((
        "TallStomp",
        tall_stomp,
        CasterStats::default(),
    ));

    // 39. ShieldWall-shape — `Damage(11) in wall(4, 2, 2, 0)` (#181 AOE
    //     Path B Wall). Rectangular slab facing +X (facing_deg=0): slab
    //     covers x∈[0, 2], z∈[-2, 2], y∈[0, 2]. Under the smoke fixture's
    //     4-agent row at (0, 1.5, 3.0, 4.5) on the X-axis (all at y=0),
    //     Wall hits slot 0 (origin: forward=0, lateral=0, y=0 → in) and
    //     slot 1 (forward=1.5 ≤ 2, lateral=0, y=0 → in). Slots 2/3 are
    //     forward > 2 → out. Both backends emit 2 chronicle records.
    let mut shield_wall = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 11.0 }],
    );
    shield_wall.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Wall,
        args: [4.0, 2.0, 2.0, 0.0],
    }));
    out.push((
        "ShieldWall",
        shield_wall,
        CasterStats::default(),
    ));

    // 40. Dropzone-shape — `Damage(9) in cylinder(2, 4)` (#181 AOE Path
    //     B Cylinder). Symmetric vertical cylinder. Under the 4-agent
    //     row, hits slot 0 + slot 1 (XZ d=1.5 ≤ 2, |y|=0 ≤ 2). Both
    //     backends emit 2 chronicle records.
    let mut dropzone = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 9.0 }],
    );
    dropzone.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Cylinder,
        args: [2.0, 4.0, 0.0, 0.0],
    }));
    out.push((
        "Dropzone",
        dropzone,
        CasterStats::default(),
    ));

    // 41. Aegis-shape — `Damage(7) in dome(2)` (#181 AOE Path B Dome).
    //     Half-sphere covering the upper hemisphere (`dist² ≤ radius²`
    //     ∧ `dy ≥ 0`). Under the 4-agent row at y=0, Dome(2) at caster
    //     slot 0: slot 0 (d=0, dy=0 → in via inclusive-plane), slot 1
    //     (d=1.5, dy=0 → in), slots 2/3 out (d > radius). Both backends
    //     emit 2 chronicle records. The y=0 boundary inclusivity pins
    //     the `>=` vs `>` semantic for the plane gate.
    let mut aegis = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 7.0 }],
    );
    aegis.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Dome,
        args: [2.0, 0.0, 0.0, 0.0],
    }));
    out.push((
        "Aegis",
        aegis,
        CasterStats::default(),
    ));

    // 42. Bulwark-shape — `Damage(5) in hull(2)` (#181 AOE Path B Hull).
    //     **Hull is a Sphere alias today** (no spec semantics defined —
    //     see `apply_program_aoe_hull_filter` doc-comment NOTE). With
    //     radius=2, behaves identically to BlastSphere: under the 4-agent
    //     row, hits slot 0 + slot 1. Pin the alias so a future spec
    //     change surfaces here. Both backends emit 2 chronicle records.
    let mut bulwark = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 5.0 }],
    );
    bulwark.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Hull,
        args: [2.0, 0.0, 0.0, 0.0],
    }));
    out.push((
        "Bulwark",
        bulwark,
        CasterStats::default(),
    ));

    // ---- Non-degenerate direction-bearing AOE shapes (#182). ----
    //
    // The Slash/PiercingLine/ShieldWall entries above all dispatch
    // through the `DispatchAbility` (target=self) physics rule; under
    // self-cast, Cone and Line collapse to the `dir_len_sq < 1e-6 →
    // no-op` branch and Wall centers its slab at the caster (covering
    // the same agents as a target=self cast). The CPU oracle has full
    // direction-aware coverage in unit tests, but the GPU branches that
    // gate on a non-zero apex→target direction were unexercised in the
    // sweep matrix before #182. The three entries below dispatch through
    // the new `DispatchAbilityToOther` physics rule (target =
    // agents.engaged_with(self)); the test runner seeds engaged_with[0]
    // = 1 so target_slot = 1, the cone faces the target, and the GPU
    // walk's angular / corridor / closed-AABB gates fire on real
    // candidates. Both backends compute the in-shape set on the same
    // (apex, target_pos) → byte-equal chronicle records.
    //
    // Position layouts mirror the explicit-target pins in
    // `aoe_chronicle_pin.rs::aoe_*_non_degenerate_*` so the same
    // CPU-oracle slot-set drives both pin shapes.

    // 43. NonDegSlash-shape — `Damage(25) in cone(45°, 5)` (#182). Apex
    //     = caster slot 0 at (0,0,0); target slot 1 at (4,0,0) drives
    //     the +X direction. 5-agent fan layout. Expected hit set =
    //     {slot 1 (on-axis), slot 2 (in-cone, dot ≈ 0.949)}; slot 0
    //     apex-excluded, slot 3 off-axis (dot ≈ 0.316), slot 4 out-of-
    //     range (dist=6 > 5). Both backends emit 2 chronicle records.
    let mut non_deg_slash = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 25.0 }],
    );
    non_deg_slash.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Cone,
        args: [45.0, 5.0, 0.0, 0.0],
    }));
    out.push((
        "NonDegSlash",
        non_deg_slash,
        CasterStats::default(),
    ));

    // 44. NonDegPiercingLine-shape — `Damage(22) in line(5.0, 1.0)`
    //     (#182). Apex = caster slot 0 at (0,0,0); target slot 1 at
    //     (4,0,0) drives +X direction. 4-agent layout. Expected hit
    //     set = {slot 0 (apex, along=0), slot 1 (target, along=4),
    //     slot 2 (along=2, perp²=0.16 ≤ 0.25)}; slot 3 perp²=0.36 >
    //     0.25 outside corridor. Both backends emit 3 chronicle
    //     records. Note Line has NO apex-exclusion (unlike Cone) — the
    //     caster is in-corridor at along=0.
    let mut non_deg_piercing_line = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 22.0 }],
    );
    non_deg_piercing_line.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Line,
        args: [5.0, 1.0, 0.0, 0.0],
    }));
    out.push((
        "NonDegPiercingLine",
        non_deg_piercing_line,
        CasterStats::default(),
    ));

    // 45. NonDegShieldWall-shape — `Damage(11) in wall(4, 2, 2, 0°)`
    //     (#182). Wall is centered at agent_pos[target_slot] (slot 1
    //     at (3,0,0)) with fixed +X facing. 4-agent layout. Expected
    //     hit set = {slot 1 (slab origin), slot 2 (forward=1.5 ≤ 2)};
    //     slot 0 forward=-3 (behind), slot 3 forward=2.5 > thickness=2
    //     out. Both backends emit 2 chronicle records. Distinguishes
    //     "wall centered at target" from "wall centered at caster" —
    //     the existing Slot 0+1 hit-set under self-cast moves to slot
    //     1+2 when the explicit-target rule shifts the slab.
    let mut non_deg_shield_wall = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 11.0 }],
    );
    non_deg_shield_wall.per_effect_areas.push(Some(engine::ability::program::EffectAreaShape {
        kind: engine::ability::program::ShapeKind::Wall,
        args: [4.0, 2.0, 2.0, 0.0],
    }));
    out.push((
        "NonDegShieldWall",
        non_deg_shield_wall,
        CasterStats::default(),
    ));

    // 46. EvocationSummon-shape — Summon(template_hash=0xDEADBEEF,
    //     count=3, lifetime=120) (slice γ closer). Caster-self with
    //     packed payload — the dispatcher splits the packed `payload_b`
    //     (= count<<24 | lifetime in low 24 bits) into distinct ring
    //     slots (slot 4 = count widened u8→u32, slot 5 = lifetime_ticks
    //     raw u32) so consumers don't have to redo the bit-unpack on
    //     read. The CPU oracle calls `apply_program`, which emits ONE
    //     `ApplyEvent::Summon` carrying the typed (count, lifetime);
    //     `apply_event_to_chronicle_record` reconstructs the same
    //     5-payload-word record (kind=62 + actor + template_hash +
    //     count + lifetime). Both backends emit byte-equal records;
    //     the parity sweep's sort + memcmp pin the equality.
    //     Adding this entry extends the sweep matrix from 45 → 46
    //     abilities and closes the last `// TODO slice γ` arm.
    out.push((
        "EvocationSummon",
        AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
            [EffectOp::Summon {
                template_hash: 0xDEADBEEF,
                count: 3,
                lifetime_ticks: 120,
            }],
        ),
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
    aoe_cpu_records_for_cast(program, caster_slot, aoe_target_slots, tick, caster_stats)
}

/// CPU oracle for the AOE Slash entry (#178 Cone). The smoke fixture's
/// implicit-target rule means `caster_slot == target_slot ==
/// agent_id` for every alive agent's cast; the cone's `direction =
/// target_pos - apex = (0,0,0)` is structurally degenerate. The CPU
/// helper passes an empty `aoe_target_slots` to mirror the GPU
/// kernel's `dir_len_sq < 1e-6 → no-op` branch — both backends emit 0
/// chronicle records under this fixture topology.
///
/// The shape of this helper is identical to `cleave_cpu_records_for_cast`
/// (both go through `apply_program_aoe`); the alias name pins the
/// "Slash routes through the cone CPU oracle path" semantic in the
/// test runner. A future explicit-target fixture would replace the
/// empty slot list with the in-cone candidate set computed via
/// `apply_program_aoe_cone_filter`.
fn slash_cpu_records_for_cast(
    program:           &AbilityProgram,
    caster_slot:       u32,
    aoe_target_slots:  &[u32],
    tick:              u32,
    caster_stats:      &CasterStats,
) -> Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> {
    aoe_cpu_records_for_cast(program, caster_slot, aoe_target_slots, tick, caster_stats)
}

/// CPU oracle for the AOE Pulverize entry (#179 Box). Same dispatch
/// shape as Cleave/Slash (all three route through `apply_program_aoe`).
/// The caller pre-filters by AABB containment around
/// `agent_pos[target_slot]`; this helper just dispatches and packs.
///
/// Under the smoke fixture's implicit-target rule
/// (caster_slot==target_slot), the box is centered at the caster's
/// position; with the 4-agent row fixture (x=0, 1.5, 3.0, 4.5) and
/// half-extents (1.5, 1.5, 1.5), in-box = {slot 0 (origin), slot 1
/// (at +x wall)}. The alias name pins the "Pulverize routes through
/// the box CPU oracle path" semantic in the test runner.
fn pulverize_cpu_records_for_cast(
    program:           &AbilityProgram,
    caster_slot:       u32,
    aoe_target_slots:  &[u32],
    tick:              u32,
    caster_stats:      &CasterStats,
) -> Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> {
    aoe_cpu_records_for_cast(program, caster_slot, aoe_target_slots, tick, caster_stats)
}

/// CPU oracle for the AOE BlastSphere entry (#180 Sphere). Same
/// dispatch shape as Cleave (Sphere is mathematically equivalent to
/// Circle today; both route through `apply_program_aoe` and emit per-
/// target chronicle records). Under the smoke fixture's 4-agent row
/// (x=0, 1.5, 3.0, 4.5) with caster at slot 0 (self-cast convention),
/// the sphere is centered at the caster; with radius=2.0, in-sphere =
/// {slot 0 (d=0), slot 1 (d=1.5)}. The alias name pins the "BlastSphere
/// routes through the sphere CPU oracle path" semantic in the test
/// runner.
fn blast_sphere_cpu_records_for_cast(
    program:           &AbilityProgram,
    caster_slot:       u32,
    aoe_target_slots:  &[u32],
    tick:              u32,
    caster_stats:      &CasterStats,
) -> Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> {
    aoe_cpu_records_for_cast(program, caster_slot, aoe_target_slots, tick, caster_stats)
}

/// CPU oracle for the AOE ShockwaveRing entry (#180 Ring). Same
/// dispatch shape as Cleave/BlastSphere (all route through
/// `apply_program_aoe`). Under the smoke fixture's 4-agent row (x=0,
/// 1.5, 3.0, 4.5) with caster at slot 0 (self-cast), the ring is
/// centered at the caster; with inner=0.5 / outer=2.0, distances 0,
/// 1.5, 3.0, 4.5 yield in-ring = {slot 1 (d=1.5 ∈ [0.5, 2.0])} only —
/// slot 0 is inner-excluded, slots 2/3 outer-excluded.
fn shockwave_ring_cpu_records_for_cast(
    program:           &AbilityProgram,
    caster_slot:       u32,
    aoe_target_slots:  &[u32],
    tick:              u32,
    caster_stats:      &CasterStats,
) -> Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> {
    aoe_cpu_records_for_cast(program, caster_slot, aoe_target_slots, tick, caster_stats)
}

/// CPU oracle for the AOE PiercingLine entry (#180 Line). The smoke
/// fixture's implicit-target rule means `caster_slot == target_slot ==
/// agent_id`; the line's `direction = target_pos - apex = (0,0,0)` is
/// structurally degenerate. The CPU helper passes an empty
/// `aoe_target_slots` to mirror the GPU kernel's `dir_len_sq < 1e-6 →
/// no-op` branch — both backends emit 0 chronicle records under this
/// fixture topology (same shape as Slash for the cone's degenerate
/// case).
fn piercing_line_cpu_records_for_cast(
    program:           &AbilityProgram,
    caster_slot:       u32,
    aoe_target_slots:  &[u32],
    tick:              u32,
    caster_stats:      &CasterStats,
) -> Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> {
    aoe_cpu_records_for_cast(program, caster_slot, aoe_target_slots, tick, caster_stats)
}

/// Shared CPU oracle for AOE Path B entries (Circle Cleave / Cone
/// Slash / Box Pulverize / Sphere BlastSphere / Ring ShockwaveRing /
/// Line PiercingLine). Calls `apply_program_aoe` with the caller-
/// supplied pre-filtered target slot list, then maps each emitted
/// ApplyEvent to a chronicle record. The caller does the geometric
/// filtering (Circle/Sphere: spatial walk; Cone:
/// `apply_program_aoe_cone_filter`; Box: `apply_program_aoe_box_filter`;
/// Ring: `apply_program_aoe_ring_filter`; Line:
/// `apply_program_aoe_line_filter`); the helper just dispatches and
/// packs.
fn aoe_cpu_records_for_cast(
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
            // Other variants stay un-AOE'd today (Cleave/Slash are
            // the only AOE entries); fall through to the caster_slot
            // if a future variant lands without an explicit target
            // field.
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
        // 1-agent self-cast path. The Slash entry (#178 cone) uses
        // a 5-agent fan layout so the cone walk has a meaningful
        // candidate set even though the smoke fixture's self-cast
        // rule degenerates the cone direction (see Slash's entry
        // doc-comment for the byte-equal-at-zero contract).
        let n_agents_for_ability = match *name {
            "Cleave"        => 4,
            "Slash"         => 5,
            "Pulverize"     => 4,
            "BlastSphere"   => 4,
            "ShockwaveRing" => 4,
            "PiercingLine"  => 4,
            // #181 AOE Path B remaining shapes — 4-agent row fixtures
            // (same x=0/1.5/3.0/4.5 layout as Cleave) for every shape
            // that expects a non-trivial in-shape set under self-cast.
            "PickFew"       => 4,
            "TallStomp"     => 4,
            "ShieldWall"    => 4,
            "Dropzone"      => 4,
            "Aegis"         => 4,
            "Bulwark"       => 4,
            // #182 non-degenerate direction-bearing shapes — caster +
            // target + extra candidates, dispatched through the new
            // `DispatchAbilityToOther` physics rule (target =
            // agents.engaged_with(self)).
            "NonDegSlash"        => 5,
            "NonDegPiercingLine" => 4,
            "NonDegShieldWall"   => 4,
            _               => N_AGENTS,
        };

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
        if *name == "Pulverize" {
            // 4-agent fixture for the box AOE: same row layout as Cleave
            // (x=0, 1.5, 3.0, 4.5). With box(1.5, 1.5, 1.5) centered at
            // caster slot 0 (under self-cast), in-box = {slot 0 (origin,
            // |d|=0), slot 1 (+x wall, |d.x|=1.5)}. Slot 2 (|d.x|=3.0)
            // and slot 3 (|d.x|=4.5) are outside extents. Agent 1 at
            // exactly the wall validates the closed-AABB (≤) semantic
            // matches between CPU oracle and GPU dispatcher.
            state.set_agent_alive(&[1, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
            ]);
        }
        if *name == "Slash" {
            // 5-agent fan layout. Caster at slot 0 (origin); other
            // slots sit in an arc the cone WOULD hit if the dispatch
            // were explicit-target (caster≠target):
            //   slot 1: ( 3, -1, 0) — in-cone bottom (≈18.4° below +X)
            //   slot 2: ( 4,  0, 0) — on-axis target candidate
            //   slot 3: ( 3,  1, 0) — in-cone top (≈18.4° above +X)
            //   slot 4: ( 1,  5, 0) — outside-cone (≈78.7° off-axis)
            // Only slot 0 alive → one cast per tick from slot 0; the
            // smoke fixture's implicit-target rule sets target_slot=0,
            // so apex==target_pos==(0,0,0) → degenerate cone → 0
            // records emitted. Both CPU oracle and GPU dispatcher
            // observe the same degenerate result; byte-equal at zero.
            state.set_agent_alive(&[1, 0, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],
                [3.0, -1.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [1.0, 5.0, 0.0],
            ]);
        }
        if *name == "BlastSphere" {
            // 4-agent fixture for the sphere AOE: same row layout as
            // Cleave (x=0, 1.5, 3.0, 4.5). Sphere is mathematically
            // equivalent to Circle today (3D dist² ≤ radius²); with
            // radius=2.0 centered at caster slot 0, in-sphere = {slot
            // 0 (d=0), slot 1 (d=1.5)}. Both backends emit 2 chronicle
            // records — same byte-set as Cleave but routed through the
            // dedicated Sphere branch (`area_kind == 6u`).
            state.set_agent_alive(&[1, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
            ]);
        }
        if *name == "ShockwaveRing" {
            // 4-agent fixture for the ring AOE: same row layout (x=0,
            // 1.5, 3.0, 4.5). Ring(0.5, 2.0) centered at caster slot 0:
            //   - slot 0 (d=0):   d² < inner² ⇒ inner-excluded
            //   - slot 1 (d=1.5): inner² ≤ d² ≤ outer² ⇒ in
            //   - slot 2 (d=3.0): d² > outer² ⇒ outer-excluded
            //   - slot 3 (d=4.5): d² > outer² ⇒ outer-excluded
            // Expected hit set = {slot 1} → 1 chronicle record. The
            // inner-radius exclusion is the distinguishing feature
            // vs. Cleave/BlastSphere — both backends agree byte-equal.
            state.set_agent_alive(&[1, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
            ]);
        }
        if *name == "PiercingLine" {
            // 4-agent fixture for the line AOE: same row layout. Under
            // the smoke fixture's self-cast rule (caster_slot ==
            // target_slot), apex == target_pos == (0,0,0) ⇒
            // direction_raw = (0,0,0) ⇒ degenerate. The WGSL kernel's
            // `dir_len_sq < 1e-6` branch skips the spatial walk; the
            // CPU oracle's `apply_program_aoe_line_filter` returns
            // empty in the same condition. Both backends emit 0
            // chronicle records — byte-equal at zero (mirrors Slash's
            // degenerate-cone semantic).
            state.set_agent_alive(&[1, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
            ]);
        }
        // #181 AOE Path B remaining shapes share the same 4-agent row
        // layout (x=0, 1.5, 3.0, 4.5) as Cleave/Pulverize/BlastSphere/
        // etc. The per-shape gates determine which slot ids end up in
        // the in-shape set — see each entry's doc-comment in
        // `build_sweep` for the expected hit set.
        if matches!(*name, "PickFew" | "TallStomp" | "ShieldWall" | "Dropzone" | "Aegis" | "Bulwark") {
            state.set_agent_alive(&[1, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
            ]);
        }
        // #182 non-degenerate direction-bearing AOE setup. Each fixture
        // seeds caster=slot 0 alive, target=slot 1 alive=0 but at the
        // direction-driving position, plus the rest of the fan/row at
        // shape-specific positions. engaged_with[0] = 1 so the dispatcher
        // computes target_slot = 1 from `agent_engaged_with[caster_slot]`.
        if *name == "NonDegSlash" {
            // Cone fan layout, mirrors the explicit-target pin in
            // `aoe_chronicle_pin.rs::aoe_cone_non_degenerate_*`.
            state.set_agent_alive(&[1, 0, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],   // slot 0: caster apex
                [4.0, 0.0, 0.0],   // slot 1: target (drives +X direction)
                [3.0, 1.0, 0.0],   // slot 2: in-cone (dot ≈ 0.949)
                [1.0, 3.0, 0.0],   // slot 3: off-axis (dot ≈ 0.316)
                [6.0, 0.0, 0.0],   // slot 4: out of range (d=6 > 5)
            ]);
            state.set_agent_engaged_with(&[1, 1, 2, 3, 4]);
        }
        if *name == "NonDegPiercingLine" {
            state.set_agent_alive(&[1, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],   // slot 0: caster apex (along=0, perp=0)
                [4.0, 0.0, 0.0],   // slot 1: target (along=4)
                [2.0, 0.4, 0.0],   // slot 2: in-corridor (perp²=0.16)
                [2.0, 0.6, 0.0],   // slot 3: outside corridor (perp²=0.36)
            ]);
            state.set_agent_engaged_with(&[1, 1, 2, 3]);
        }
        if *name == "NonDegShieldWall" {
            state.set_agent_alive(&[1, 0, 0, 0]);
            state.set_agent_positions(&[
                [0.0, 0.0, 0.0],   // slot 0: caster (forward=-3, behind slab)
                [3.0, 0.0, 0.0],   // slot 1: target (slab origin → slab x ∈ [3,5])
                [4.5, 0.0, 0.0],   // slot 2: forward=1.5 ≤ thickness=2 → in
                [5.5, 0.0, 0.0],   // slot 3: forward=2.5 > thickness → out
            ]);
            state.set_agent_engaged_with(&[1, 1, 2, 3]);
        }

        for &tick in TICKS {
            // -- CPU oracle.
            let mut cpu = match *name {
                "Cleave" => cleave_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[0, 1],
                    tick,
                    caster_stats,
                ),
                "Slash" => slash_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[],
                    tick,
                    caster_stats,
                ),
                "Pulverize" => pulverize_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[0, 1],
                    tick,
                    caster_stats,
                ),
                "BlastSphere" => blast_sphere_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[0, 1],
                    tick,
                    caster_stats,
                ),
                "ShockwaveRing" => shockwave_ring_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[1],
                    tick,
                    caster_stats,
                ),
                "PiercingLine" => piercing_line_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[],
                    tick,
                    caster_stats,
                ),
                // #181 AOE Path B remaining shapes (Spread/Column/Wall/
                // Cylinder/Dome/Hull). Hit sets under the 4-agent row
                // fixture (x=0, 1.5, 3.0, 4.5 at y=0):
                //
                //   PickFew   (Spread, r=2.0, max=2) — Wave 1.6 #183:
                //     Circle gate hits slot 0 (d=0) and slot 1 (d=1.5);
                //     slots 2/3 are outside r=2. After sort by AgentId
                //     ascending + truncate to 2, kept set = [slot 0,
                //     slot 1]. CPU oracle's `apply_program_aoe_spread_filter`
                //     produces the same list; GPU's per-thread insertion
                //     sort + cap matches. Both produce 2 records.
                //
                //   TallStomp (Column, r=2.0, h=4.0):
                //     XZ disc gate hits slot 0 (d=0) and slot 1 (d=1.5);
                //     all at y=0, dy=0 ∈ [0, 4]. → [slot 0, slot 1].
                //
                //   ShieldWall (Wall, len=4, h=2, thick=2, +X):
                //     Slab covers x∈[0,2], z∈[-2,2], y∈[0,2]. Hits slot
                //     0 (forward=0) and slot 1 (forward=1.5 ≤ 2). Slots
                //     2/3 forward > 2 ⇒ out. → [slot 0, slot 1].
                //
                //   Dropzone (Cylinder, r=2.0, h=4.0):
                //     XZ disc + |dy| ≤ 2. Hits slot 0 + slot 1. → [slot 0, slot 1].
                //
                //   Aegis (Dome, r=2.0):
                //     Sphere gate + dy ≥ 0 (all at dy=0 boundary). Hits
                //     slot 0 + slot 1. → [slot 0, slot 1].
                //
                //   Bulwark (Hull, r=2.0):
                //     Sphere alias today. Hits slot 0 + slot 1. →
                //     [slot 0, slot 1].
                "PickFew" => aoe_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[0, 1],
                    tick,
                    caster_stats,
                ),
                "TallStomp" | "ShieldWall" | "Dropzone" | "Aegis" | "Bulwark" => {
                    aoe_cpu_records_for_cast(
                        program,
                        /*caster_slot*/ 0,
                        /*aoe_target_slots*/ &[0, 1],
                        tick,
                        caster_stats,
                    )
                }
                // #182 non-degenerate direction-bearing AOE: the
                // pre-filtered slot lists below mirror the GPU-side
                // candidate sets computed by the cone/line/wall WGSL
                // walks. The CPU oracle's `apply_program_aoe` iterates
                // these slot lists and emits one ApplyEvent per (op,
                // target) pair; both backends produce byte-equal
                // chronicle records when keyed on (actor=caster_slot,
                // target=per-target slot).
                //
                // For Cone:  apex-excluded caster + slot 1 (on-axis) +
                //            slot 2 (in-cone). Slot 3 off-axis, slot 4
                //            out-of-range.
                // For Line:  no apex-exclusion; slots 0+1+2 in-corridor;
                //            slot 3 perp²=0.36 > 0.25 outside.
                // For Wall:  centered at slot 1's position; slot 1 +
                //            slot 2 in-slab; slot 0 behind, slot 3
                //            past thickness.
                "NonDegSlash" => aoe_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[1, 2],
                    tick,
                    caster_stats,
                ),
                "NonDegPiercingLine" => aoe_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[0, 1, 2],
                    tick,
                    caster_stats,
                ),
                "NonDegShieldWall" => aoe_cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*aoe_target_slots*/ &[1, 2],
                    tick,
                    caster_stats,
                ),
                _ => cpu_records_for_cast(
                    program,
                    /*caster_slot*/ 0,
                    /*target_slot*/ 0,
                    tick,
                    caster_stats,
                    /*target_stats*/ caster_stats,
                ),
            };
            canonicalize(&mut cpu);

            // -- GPU dispatch. Pin the GPU's cfg.seed to the same
            //    `world_seed as u32` the CPU oracle keys on, so the
            //    chance-gate PCG draws agree bit-for-bit (P11).
            //
            // #182: direction-bearing non-degenerate entries dispatch
            // through the third physics rule (`DispatchAbilityToOther`,
            // target = agents.engaged_with(self)) so the dispatcher
            // computes target_slot from `agent_engaged_with[caster_slot]`
            // (seeded to slot 1 above). All other entries continue to
            // use the self-cast `DispatchAbility` rule.
            match *name {
                "NonDegSlash" | "NonDegPiercingLine" | "NonDegShieldWall" => {
                    state.step_explicit_target_with_seed(tick, WORLD_SEED as u32);
                }
                _ => {
                    state.step_with_seed(tick, WORLD_SEED as u32);
                }
            }
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
