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
//!   6. `Damage(10) chance=0xFFFE (always-fires)`   — Daze-shape, deterministic
//!   7. `Damage(40) { stun 1s }`                    — Reap+nested-stun shape
//!   8. `Heal(8) + 50% AbilityPower`                — AP scaling (=0.0 on both backends)
//!   9. `LifeSteal(0.5, 5s)`                        — Vampirize-shape (q8 fraction)
//!  10. `DamageModify(0.5, 5s)`                     — Fortify-shape (q8 multiplier)
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
//! K = 10 × 1 × 5 = **50 casts**, producing 55 chronicle records (50
//! primary + 5 nested-Stun follow-ups from the Reap+Stun-shape arm).
//!
//! ## What's deferred
//!
//! - **Intermediate chance values (0 < q16 < 0xFFFE).** The CPU's chance
//!   gate uses `per_agent_u32` (ahash-based, P5 contract). The GPU
//!   dispatcher today does NOT gate on the chance modifier at all (it
//!   honors the `when` predicate but not `chances[i]` — scoping decision
//!   in the slice that landed). Until P11 wires chance-gating onto the
//!   GPU side, intermediate q16 values would diverge: CPU gates per
//!   PCG draw, GPU fires unconditionally. We sidestep with chance =
//!   0xFFFE (CPU "always fires" — and GPU also fires unconditionally,
//!   so both produce the same record). Add an intermediate-chance arm
//!   when GPU chance-gate-on-GPU lands.
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
    CasterStats, EffectOp, EffectPredicate, EffectPredicateBinder, EffectPredicateOp,
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

    // 6. Daze-shape — chance=0xFFFE deterministic. Damage(10) gated
    //    on always-fires q16. CPU's per_agent_u32 derivation produces
    //    a draw `& 0xFFFF` < 0xFFFE for every (seed, caster, tick)
    //    other than the 1/65536 case where the draw is exactly 0xFFFE
    //    or 0xFFFF — for our pinned seed/tick set we verify offline
    //    that the gate fires for every entry. The GPU dispatcher
    //    doesn't gate on `chances[i]` today, so it fires unconditionally
    //    — both backends produce the same record IF the CPU draw also
    //    fires. (If we hit the 1-in-65k miss for a specific tick, the
    //    test would fail loudly with a count mismatch — that's the
    //    documented deferral above.)
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

        // GPU side seeds slot 0 with this ability's per-agent stats.
        // Self-cast convention: target_stats = caster_stats (CPU
        // oracle reads target_stats for `target.<field>` predicates;
        // the smoke fixture's implicit-target rule means target_slot
        // == caster_slot ⇒ same SoA row ⇒ same f32 stat value).
        let per_agent_stats =
            vec![per_agent_from_caster_stats(caster_stats); N_AGENTS as usize];
        let per_agent_levels = vec![ability_id; N_AGENTS as usize];

        let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
            N_AGENTS,
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

        for &tick in TICKS {
            // -- CPU oracle: self-cast at slot 0.
            let mut cpu = cpu_records_for_cast(
                program,
                /*caster_slot*/ 0,
                /*target_slot*/ 0,
                tick,
                caster_stats,
                /*target_stats*/ caster_stats,
            );
            canonicalize(&mut cpu);

            // -- GPU dispatch.
            state.step(tick);
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
