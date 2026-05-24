//! Smoke test for `assets/sim/apply_ability_smoke.sim` — task #138.
//!
//! Drives the full `dsl_compiler` pipeline (parse → resolve → CG lower
//! → schedule → emit) against the apply_ability fixture and asserts the
//! WGSL kernel composer wires up the dispatcher correctly:
//!
//!   - the `CgStmt::ApplyAbility` arm in `cg::emit::wgsl_body` runs to
//!     completion (no `UnsupportedPhysicsStmt` regression),
//!   - the dispatcher loop scaffolding (`for (var i: u32 = 0u; i < 6u`,
//!     `EFFECT_KIND_EMPTY = 0xFFu` continue) lands in the kernel body,
//!   - **slice γ — chronicle-bearing arms** emit real `atomicStore`
//!     writes against `event_ring` with the runtime EventKindIds
//!     (Damage=26, Heal=27, Shield=28, Stun=29, Slow=30, TransferGold=31,
//!     ModifyStanding=32). The unit tests in `wgsl_body.rs` pin the
//!     same fact at the format-string level; this test pins it at the
//!     kernel-body level (i.e. after the binding composer + thread
//!     preamble + cfg uniform have been wrapped around it).
//!
//! Without this test, the dispatcher is exercised only by the inline
//! tests in `wgsl_body.rs` against a hand-built `CgProgram` — the full
//! pipeline (binding composer / cfg uniform / EventRing(Append) write
//! recording / `agent_id` preamble) goes uncovered, so a regression in
//! any of those layers would surface only when the first runtime crate
//! finally references `apply_ability`. That's a worse failure surface.
//!
//! Mirror sim file shape: `target_chaser_compiles` in
//! `stress_fixtures_compile.rs` — same `compile_sim` driver,
//! same kernel-body fishing pattern.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn compile_sim(path: &std::path::Path) -> Result<EmittedArtifacts, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|e| format!("lower: {e:?}"))?;
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))
}

/// Find the first WGSL kernel whose name contains `needle`.
fn kernel_body_containing<'a>(art: &'a EmittedArtifacts, needle: &str) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle))
        .map(|(_, body)| body.as_str())
}

#[test]
fn apply_ability_smoke_compiles() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("apply_ability_smoke.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[apply_ability_smoke] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// Like `compile_sim` but tolerates well_formed diagnostics — mirrors
/// duel_abilities_runtime's build.rs (which extracts `o.program` from
/// the `DriverOutcome::Err` path on P6 violations and emits anyway).
/// Returns the artifacts even when lower reports diagnostics.
fn compile_sim_tolerating_diagnostics(
    path: &std::path::Path,
) -> Result<(EmittedArtifacts, Vec<String>), String> {
    let src = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let (cg, diags) = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => (p, Vec::new()),
        Err(o) => {
            let diags: Vec<String> = o.diagnostics.iter().map(|d| format!("{d:?}")).collect();
            (o.program, diags)
        }
    };
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))?;
    Ok((art, diags))
}

/// Closed-loop pin: apply_ability writes EffectDamageApplied chronicle
/// records → a PerEvent physics rule reads them and decrements target
/// hp. Proves the chronicle records can drive sim state mutation —
/// the missing link between dispatcher writes and game state for #138.
///
/// Trips the same P6 well_formed diagnostic that
/// duel_abilities.sim::ApplyDamage used to trip (PerEvent +
/// agents.set_hp). Gap X (2026-05-04) extended the P6 check to
/// recognize the authored `@phase(post)` annotation as the spec'd
/// chronicle / telemetry channel — agent mutation is exactly the
/// point of `@phase(post)` rules. The fixture now annotates the
/// chronicle consumer with `@phase(post)` and the diagnostic no
/// longer fires; the test pins clean lowering.
#[test]
fn apply_ability_chronicle_consumer_compiles_with_tolerated_p6() {
    let path = workspace_path("assets/sim/apply_ability_chronicle_consumer.sim");
    let (art, diags) = compile_sim_tolerating_diagnostics(&path).unwrap_or_else(|e| {
        panic!("apply_ability_chronicle_consumer.sim hard-failed at: {e}");
    });

    // Pin: NO diagnostics post-Gap-X. The `@phase(post)` annotation
    // on `ApplyChronicleDamage` opts into the P6 exemption for
    // authored chronicle physics. Any diagnostic — P6 or otherwise —
    // is a regression.
    assert!(
        diags.is_empty(),
        "unexpected diagnostics from chronicle consumer post-Gap-X:\n{:?}",
        diags,
    );

    // Half A: dispatcher kernel emits.
    let dispatch = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("DispatchAbility"))
        .map(|(_, b)| b.as_str())
        .expect("DispatchAbility kernel missing");
    assert!(
        dispatch.contains("for (var i: u32 = 0u; i < 6u;"),
        "DispatchAbility kernel must carry the dispatcher loop;\n{dispatch}"
    );

    // Half B: consumer kernel emits and writes agent_hp.
    let consumer = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("ApplyChronicleDamage"))
        .map(|(_, b)| b.as_str())
        .expect("ApplyChronicleDamage kernel missing");
    assert!(
        consumer.contains("agent_hp"),
        "ApplyChronicleDamage must touch agent_hp;\n{consumer}"
    );
}

/// Regression pin for the engine-event kind aliasing fix
/// (2026-05-06). Pre-fix, the consumer kernel filtered records via
/// `if (kind == 1u)` because `EffectDamageApplied` was the second
/// `event` decl in the .sim source order (after `Tick`). The
/// dispatcher writes records with the engine's hardcoded
/// `EventKindId::EffectDamageApplied = 26`, so the consumer's filter
/// never matched. The runtime
/// `apply_ability_chronicle_consumer_runtime/build.rs` used to
/// sed-rewrite `== 1u` to `== 26u` to close the loop.
///
/// Post-fix, `dsl_ast::engine_events::engine_event_kind_id_for_name`
/// aliases `EffectDamageApplied` to discriminant 26 at resolve time,
/// and both `populate_event_kinds` (driver) and `resolve_event_ref`
/// (driver) mirror that assignment. The consumer kernel emits
/// `if (kind == 26u)` directly. This test asserts the post-fix shape
/// — if it ever regresses to `== 1u`, the closed loop in
/// `apply_ability_chronicle_consumer_runtime` silently breaks again.
#[test]
fn chronicle_consumer_filter_uses_engine_discriminant() {
    let path = workspace_path("assets/sim/apply_ability_chronicle_consumer.sim");
    let (art, _diags) = compile_sim_tolerating_diagnostics(&path).unwrap_or_else(|e| {
        panic!("apply_ability_chronicle_consumer.sim hard-failed at: {e}");
    });

    let consumer = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("ApplyChronicleDamage"))
        .map(|(_, b)| b.as_str())
        .expect("ApplyChronicleDamage kernel missing");

    // Engine discriminant for EffectDamageApplied is 26 — see
    // `EventKindId` in `crates/engine/src/cascade/handler.rs` and the
    // mirroring `EFFECT_KIND_TO_EVENT_KIND_ID` table in
    // `cg/emit/wgsl_body.rs`. The consumer kernel must filter on 26u
    // for the closed loop to fire.
    assert!(
        consumer.contains("== 26u)"),
        "consumer kernel must filter on engine discriminant=26 for \
         EffectDamageApplied; raw filter constants should NOT be the \
         .sim declaration-index (=1 for this fixture). Pre-fix this \
         test would fail with `== 1u)` and the runtime crate had to \
         sed-rewrite the WGSL. Post-fix, the compiler aliases known \
         engine event names to their hardcoded EventKindId at \
         resolve time. Kernel:\n{consumer}"
    );

    // Belt-and-braces pin against the pre-fix shape: the .sim-local
    // declaration-index for EffectDamageApplied is 1 (after `Tick` at
    // 0). If the kernel ever emits `== 1u)` again the regression has
    // returned. Whitelist the inner agent loop's `i < 1u` (it doesn't
    // exist today; this is a future-proof guard against a pattern
    // that COULD include `1u` for unrelated reasons — adjust if a
    // legitimate `1u)` literal lands in the consumer body).
    assert!(
        !consumer.contains("== 1u)"),
        "consumer kernel still emits the .sim-local index `== 1u)` — \
         engine-event aliasing did not take effect. Kernel:\n{consumer}"
    );
}

/// Corpus-level pin for the verb-body apply_ability surface. The
/// inline-source tests (`verb_body_with_apply_ability_lowers_cleanly`)
/// already prove parse → resolve → emit; this one pins the same path
/// at the .sim file level — symmetric with apply_ability_smoke.sim
/// covering the physics-scope surface.
#[test]
fn apply_ability_verb_smoke_compiles() {
    let path = workspace_path("assets/sim/apply_ability_verb_smoke.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("apply_ability_verb_smoke.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");

    // The verb expander synthesises `verb_chronicle_Cast` from the
    // verb body; its kernel must contain the dispatcher slot loop +
    // both operand lets (mirroring the inline-source assertions in
    // verb_body_with_apply_ability_lowers_cleanly, but at the
    // corpus-file level).
    let body = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("verb_chronicle_Cast"))
        .map(|(_, b)| b.as_str())
        .unwrap_or_else(|| {
            panic!(
                "no verb_chronicle_Cast kernel in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>(),
            );
        });
    assert!(
        body.contains("for (var i: u32 = 0u; i < 6u;"),
        "verb_chronicle_Cast must carry the dispatcher slot loop;\n\
         body:\n{body}"
    );
    assert!(
        body.contains("let caster_slot: u32"),
        "verb_chronicle_Cast must emit caster_slot from `by self`;\n\
         body:\n{body}"
    );
    assert!(
        body.contains("let target_slot: u32"),
        "verb_chronicle_Cast must emit target_slot from `target self`;\n\
         body:\n{body}"
    );
}

/// Combined fixture pin: verb-body apply_ability dispatcher (the
/// apply_ability_verb_smoke shape) together with a chronicle consumer
/// rule (the apply_ability_chronicle_consumer shape). Structural
/// template for task #138's Strike swap — once this works, swapping
/// duel_abilities Strike onto the same shape is a mechanical port.
///
/// Used to trip the same P6 well_formed diagnostic that
/// apply_ability_chronicle_consumer.sim tripped (PerEvent +
/// agents.set_hp). Gap X (2026-05-04) added `@phase(post)`
/// recognition to the P6 check — the fixture now annotates the
/// consumer rule and lowers cleanly.
#[test]
fn apply_ability_verb_chronicle_consumer_compiles_with_tolerated_p6() {
    let path = workspace_path("assets/sim/apply_ability_verb_chronicle_consumer.sim");
    let (art, diags) = compile_sim_tolerating_diagnostics(&path).unwrap_or_else(|e| {
        panic!("apply_ability_verb_chronicle_consumer.sim hard-failed at: {e}");
    });

    // Pin: NO diagnostics post-Gap-X. Authored `@phase(post)`
    // chronicle physics is the spec'd channel for agent mutation.
    assert!(
        diags.is_empty(),
        "unexpected diagnostics from verb+chronicle-consumer fixture post-Gap-X:\n{:?}",
        diags,
    );

    // Half A: verb-body dispatcher kernel emits.
    let dispatch = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("verb_chronicle_Cast"))
        .map(|(_, b)| b.as_str())
        .unwrap_or_else(|| {
            panic!(
                "no verb_chronicle_Cast kernel in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>(),
            );
        });
    assert!(
        dispatch.contains("for (var i: u32 = 0u; i < 6u;"),
        "verb_chronicle_Cast must carry the dispatcher slot loop;\n\
         body:\n{dispatch}"
    );
    assert!(
        dispatch.contains("let caster_slot: u32"),
        "verb_chronicle_Cast must emit caster_slot from `by self`;\n\
         body:\n{dispatch}"
    );
    assert!(
        dispatch.contains("let target_slot: u32"),
        "verb_chronicle_Cast must emit target_slot from `target self`;\n\
         body:\n{dispatch}"
    );

    // Half B: chronicle consumer kernel emits and writes agent_hp.
    // Note: the scheduler FUSES the consumer (op#1: physics
    // ApplyChronicleDamage) with the verb-body dispatcher (op#2:
    // physics_verb_chronicle_Cast) into a single kernel named
    // `physics_ApplyChronicleDamage_and_verb_chronicle_Cast` because
    // both are PerEvent-shape rules over event_ring. We pull the
    // fused kernel here and assert on the consumer-half emit.
    let consumer = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("ApplyChronicleDamage"))
        .map(|(_, b)| b.as_str())
        .expect("ApplyChronicleDamage kernel missing");
    assert!(
        consumer.contains("agent_hp"),
        "ApplyChronicleDamage must touch agent_hp;\n{consumer}"
    );

    // Engine-event kind aliasing: consumer must filter on engine
    // discriminant=26 for EffectDamageApplied, NOT the .sim-local
    // declaration index. Mirrors `chronicle_consumer_filter_uses_engine_discriminant`
    // for apply_ability_chronicle_consumer.sim.
    assert!(
        consumer.contains("== 26u)"),
        "consumer kernel must filter on engine discriminant=26 for \
         EffectDamageApplied. Pre-fix the runtime had to sed-rewrite \
         the WGSL; post-fix `dsl_ast::engine_events` aliases known \
         engine event names to their hardcoded EventKindId at resolve \
         time. Kernel:\n{consumer}"
    );
    // Belt-and-braces guard against the pre-fix `== 1u)` filter on the
    // consumer's atomicLoad of `event_ring[event_idx * 11u + 0u]`
    // (kind word). The fused kernel contains many `== 1u)` instances
    // for unrelated reasons (Heal effect kind, action_id branch, etc.),
    // so a flat `!contains("== 1u)")` is too aggressive. Instead pin
    // the consumer's specific filter shape: the op#1 emit reads
    // `event_ring[event_idx * 11u + 0u]` and gates on `== 26u)`. If
    // engine-event aliasing breaks, that filter would emit `== 1u)`.
    // (stride 10→11: P11 seq trailer word added at record offset 10)
    assert!(
        consumer.contains("event_ring[event_idx * 11u + 0u]) == 26u)"),
        "consumer's kind-filter must be `event_ring[event_idx * 11u + 0u]) \
         == 26u)` — exact pre-fix regression target. Kernel:\n{consumer}"
    );
}

#[test]
fn apply_ability_smoke_emits_dispatcher_loop_in_kernel_body() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    // Pick the kernel that hosts the DispatchAbility physics rule.
    // The kernel composer's naming may evolve; fall back to a
    // generic "physics" search so the fixture pin doesn't drift on
    // composer renames.
    let body = kernel_body_containing(&art, "DispatchAbility")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });

    // Dispatcher scaffolding (slice β step 2):
    assert!(
        body.contains("for (var i: u32 = 0u; i < 6u;"),
        "expected dispatcher slot loop (MAX_EFFECTS_PER_PROGRAM = 6) in kernel body;\n\
         got body:\n{body}"
    );
    assert!(
        body.contains("if (kind == 0xFFu)"),
        "expected EFFECT_KIND_EMPTY skip in kernel body;\n\
         got body:\n{body}"
    );
    assert!(
        body.contains("ability_registry_effect_kinds[effect_base + i]"),
        "expected effect-kinds SoA read indexed by effect_base + i;\n\
         got body:\n{body}"
    );

    // Slice γ — every chronicle-bearing arm emits a kind-tag header
    // store against `event_ring`. The dispatcher's `let _slot: u32 =
    // atomicAdd(&event_tail[0], 1u);` slot acquisition appears once
    // per chronicle-bearing arm, so the body should carry exactly 11
    // copies after the binding composer wraps it (Bleed verb swap
    // added SelfDamage=39 (2026-05-06); Vampirize verb swap added
    // LifeSteal=40, mirror of Bleed; Fortify verb swap added
    // DamageModify=41, mirror of Vampirize; Reap verb swap added
    // Execute=42, mirror of Fortify — closes all 8 duel_abilities verbs).
    for (variant_label, expected_kind_tag) in &[
        ("Damage",          26u32),
        ("Heal",            27u32),
        ("Shield",          28u32),
        ("Stun",            29u32),
        ("Slow",            30u32),
        ("TransferGold",    31u32),
        ("ModifyStanding",  32u32),
        ("SelfDamage",      39u32),
        ("LifeSteal",       40u32),
        ("DamageModify",    41u32),
        ("Execute",         42u32),
    ] {
        let needle = format!(
            "atomicStore(&event_ring[_slot * 11u + 0u], {expected_kind_tag}u);"
        );
        assert!(
            body.contains(&needle),
            "post-pipeline kernel body should still carry the {variant_label} \
             arm's chronicle write (kind={expected_kind_tag}u);\n\
             got body:\n{body}"
        );
    }

    // Slot acquisition appears at least 11 times — once per chronicle-
    // bearing arm. Use `>= 11` rather than `== 11` because future arms
    // may grow chronicle counterparts (the test stays correct as long
    // as the eleven slice-γ wirings remain).
    let slot_acquisitions = body
        .matches("let _slot: u32 = atomicAdd(&event_tail[0], 1u);")
        .count();
    assert!(
        slot_acquisitions >= 11,
        "expected ≥11 chronicle slot acquisitions (one per slice-γ arm); \
         got {slot_acquisitions};\n\
         body:\n{body}"
    );

    // Wave 1.5#4 GPU wire-up (2026-05-07): the dispatcher computes
    // `scale_bonus = Σ percent * caster_stat` from the per-slot
    // `scaling_stat_refs` / `scaling_percents` SoA + per-stat agent SoA
    // reads at `caster_slot`, then folds it into amount-bearing arms via
    // `bitcast<f32>(payload_a) + scale_bonus`. Pin three structural
    // facts so any future drift surfaces here:
    //   1. The scaling SoA reads land at the slot's `scaling_base + i*2 + k`
    //      offset (k=0..1 covers MAX_SCALINGS_PER_EFFECT=2).
    //   2. The per-stat switch reaches every non-AbilityPower variant
    //      (the AbilityPower=1 tag returns 0.0 — no agent SoA slot).
    //   3. The Damage arm now writes `payload_a + scale_bonus` (NOT just
    //      `payload_a`).
    assert!(
        body.contains("ability_registry_scaling_stat_refs[s_off]"),
        "expected dispatcher to read the scaling_stat_refs SoA at \
         per-slot offset; got body:\n{body}"
    );
    assert!(
        body.contains("ability_registry_scaling_percents[s_off]"),
        "expected dispatcher to read the scaling_percents SoA at \
         per-slot offset; got body:\n{body}"
    );
    assert!(
        body.contains("agent_max_hp[caster_slot]"),
        "expected agent_stat switch to read agent_max_hp at caster_slot \
         (the MaxHp branch is the load-bearing case for the duel \
         Bleed verb's `+5% max_hp` scaling); got body:\n{body}"
    );
    assert!(
        body.contains("agent_attack_damage[caster_slot]"),
        "expected agent_stat switch to cover agent_attack_damage; \
         got body:\n{body}"
    );
    assert!(
        body.contains("scale_bonus = scale_bonus + s_pct * stat_v"),
        "expected scale_bonus accumulator to fold each percent * stat \
         contribution; got body:\n{body}"
    );
    assert!(
        body.contains("bitcast<f32>(payload_a) + scale_bonus"),
        "expected amount-bearing arms (Damage / Heal / Shield / \
         SelfDamage / DoT / HoT / TimedShield) to fold scale_bonus into \
         the f32 amount; got body:\n{body}"
    );
    assert!(
        body.contains("let nested_scale_bonus: f32 = 0.0;"),
        "expected the nested-effect walk to force scale_bonus = 0.0 \
         (mirrors apply.rs's `push_effect_event(... 0.0)` for nested \
         ops — they have no scaling slot in the registry); \
         got body:\n{body}"
    );
}

/// Wave 1.5#7 GPU eval pin (updated for task #227 — compound predicates):
/// dispatcher emits an RPN-walking when-predicate evaluator guarded by
/// `if (when_passes)`. Each effect slot owns 12 RPN nodes
/// (MAX_PRED_NODES_PER_EFFECT); per-ability stride is 6*12 = 72.
/// Operator markers: 0xFE=AND, 0xFD=OR, 0xFC=NOT; sentinel 0xFF
/// terminates the walk.
#[test]
fn dispatcher_emits_when_predicate_eval_block() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");
    let body = kernel_body_containing(&art, "DispatchAbility")
        .expect("dispatcher kernel must exist");
    // Per-effect node base = ability_slot * 72 + i * 12.
    assert!(
        body.contains("pred_node_base: u32 = ability_slot * 72u + i * 12u"),
        "expected per-effect RPN base = ability_slot*72 + i*12 \
         (per-ability stride MAX_EFFECTS_PER_PROGRAM*MAX_PRED_NODES_PER_EFFECT); \
         got body:\n{body}"
    );
    // Walk the 12-node node array.
    assert!(
        body.contains("for (var pi: u32 = 0u; pi < 12u"),
        "expected RPN walk loop bounded at MAX_PRED_NODES_PER_EFFECT=12; \
         got body:\n{body}"
    );
    // End-of-nodes sentinel.
    assert!(
        body.contains("if (pn_binder == 0xFFu) { break; }"),
        "expected RPN walk to break on WHEN_PRED_NONE_SENTINEL (0xFFu); \
         got body:\n{body}"
    );
    // Operator markers.
    assert!(
        body.contains("if (pn_binder == 0xFEu)"),
        "expected AND-operator branch (0xFEu); got body:\n{body}"
    );
    assert!(
        body.contains("if (pn_binder == 0xFDu)"),
        "expected OR-operator branch (0xFDu); got body:\n{body}"
    );
    assert!(
        body.contains("if (pn_binder == 0xFCu)"),
        "expected NOT-operator branch (0xFCu); got body:\n{body}"
    );
    // Atom evaluator's per-op switch shape.
    assert!(
        body.contains("atom_v = pred_lhs <  pred_literal"),
        "expected atom_v assignment for Lt op (case 0u); got body:\n{body}"
    );
    assert!(
        body.contains("atom_v = pred_lhs == pred_literal"),
        "expected atom_v assignment for Eq op (case 4u); got body:\n{body}"
    );
    // Wraps the chronicle arm chain in `if (when_passes)`.
    assert!(
        body.contains("if (when_passes && chance_passes)"),
        "expected the chronicle arm chain to be wrapped in \
         `if (when_passes && chance_passes) {{ ... }}`; got body:\n{body}"
    );
    // Predicate uses the same agent SoA bindings as the scale_bonus
    // computation — pin agent_hp[pred_agent] so the load-bearing
    // `target.hp` Reap path stays wired.
    assert!(
        body.contains("agent_hp[pred_agent]"),
        "expected predicate switch to read agent_hp at pred_agent slot \
         (Reap-shape `when target.hp < 20`); got body:\n{body}"
    );
}

/// Naga-validate the dispatcher's emitted WGSL. Format-string
/// assertions catch missing arms / wrong constants, but they DON'T
/// catch syntax errors the WGSL frontend rejects (mismatched braces,
/// undeclared identifiers, type errors, atomic-handle misuse, etc.).
/// Without this gate, a malformed dispatcher could ship green at the
/// dsl_compiler test boundary and only blow up when a runtime crate
/// tries to feed the kernel into wgpu.
///
/// Naga 26.0.0 is the WGSL frontend that wgpu 26.0.1 uses (see
/// `Cargo.toml` pinning), so passing naga here is a strong proxy for
/// "this kernel will compile on a real GPU device". The actual
/// device-driving GPU parity test (#133) layers on top of this gate
/// once a runtime crate consumes the dispatcher.
#[test]
fn apply_ability_smoke_kernel_parses_through_naga() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}:\n    {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "apply_ability_smoke emitted {} naga-invalid WGSL kernels — \
         the dispatcher's chronicle writes (slice-γ self-cast or \
         per-arm atomicStore layout) likely produced WGSL the frontend \
         rejects:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Variant: `apply_ability` nested inside an `if` body. Exercises
/// `list_contains_apply_ability`'s recursion through `CgStmt::If` —
/// the helper that drives `wire_ability_registry_column_reads` must
/// descend into both `then` and `else_` arms; if it stops at the
/// top-level statement list, the BGL composer never wires the
/// `ability_registry_*` bindings and naga rejects the kernel
/// (same shape of bug as commit `f447d3eb`).
///
/// Source-string-compiled rather than file-loaded: a transient
/// fixture variant doesn't justify a new `.sim` file in the corpus.
#[test]
fn apply_ability_nested_in_if_body_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }

        entity Hero : Agent { }

        physics ConditionalDispatch @phase(per_agent) {
          on Tick {} where (self.alive) {
            if (agents.level(self) > 0) {
              apply_ability agents.level(self)
            }
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "ConditionalDispatch")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");
    // Sanity: dispatcher loop reached the if-arm.
    assert!(
        body.contains("for (var i: u32 = 0u; i < 6u;"),
        "dispatcher loop must land in nested if arm;\n{body}"
    );
    // Sanity: ability_registry bindings declared (proves
    // wire_ability_registry_column_reads recursed into the if).
    assert!(
        body.contains("ability_registry_effect_kinds"),
        "BGL composer must wire ability_registry_effect_kinds even \
         when ApplyAbility is nested in an if body;\n{body}"
    );

    // Naga validator over every emitted kernel.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse failed: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate failed: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "nested-if variant emitted {} kernels naga rejects:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Variant: two PerAgent physics rules sharing `on Tick` — one uses
/// `apply_ability`, one doesn't. The schedule fusion pass combines
/// them into a single kernel (`cs_physics_DispatcherRule_and_PlainRule`)
/// because both share the same dispatch shape, event source, and
/// have no read/write conflicts.
///
/// The fused kernel's binding set is the UNION of both rules' reads
/// and writes — so `ability_registry_*` is present (DispatcherRule
/// needs it) and `event_ring` is present (DispatcherRule's chronicle
/// writes), even though PlainRule individually doesn't reference
/// either. This is correct behavior; the test pins it so a future
/// fusion-pass change that disables this case surfaces deliberately
/// rather than as a confusing kernel-naming change.
///
/// What this test guards: the post-fusion kernel still passes naga
/// validation. The dispatcher's binding-recording logic
/// (`wire_ability_registry_column_reads`) survives fusion correctly
/// — a fusion-time bug would either drop the recorded reads (kernel
/// references undeclared identifiers) or duplicate them (double
/// binding declarations).
#[test]
fn apply_ability_in_fused_kernel_with_plain_rule_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }

        entity Hero : Agent { }

        physics DispatcherRule @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self)
          }
        }

        physics PlainRule @phase(per_agent) {
          on Tick {} where (self.alive) {
            agents.set_hp(self, agents.hp(self) + 1.0)
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    // The fused kernel name is composer-driven; look for either rule
    // name, taking whichever the composer picked.
    let fused_body = kernel_body_containing(&art, "DispatcherRule")
        .or_else(|| kernel_body_containing(&art, "PlainRule"))
        .expect("fused physics kernel emitted");

    // Bindings present (union of both rules):
    assert!(
        fused_body.contains("ability_registry_effect_kinds"),
        "fused kernel must carry ability_registry_effect_kinds (from \
         DispatcherRule's apply_ability);\n{fused_body}"
    );
    assert!(
        fused_body.contains("agent_hp"),
        "fused kernel must carry agent_hp (from PlainRule's HP write);\n{fused_body}"
    );

    // Each binding declared exactly once (dedup correct under fusion).
    let kinds_decls = fused_body
        .lines()
        .filter(|l| l.contains("var<storage") && l.contains("ability_registry_effect_kinds"))
        .count();
    assert_eq!(
        kinds_decls, 1,
        "ability_registry_effect_kinds declared exactly once even under fusion;\n{fused_body}"
    );

    // Naga validator confirms the fused kernel is well-formed.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "fused-rule variant emitted {} naga-rejected kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Variant: `apply_ability` inside a PerEvent rule (driven by a
/// custom event with payload binding). Different kernel shape than
/// the PerAgent fixtures — the kernel iterates `event_count` rather
/// than `agent_cap`, and the cfg uniform carries event-count, not
/// agent-count.
///
/// **Slice δ part 2 (#161): now errors cleanly at lowering time.**
///
/// Was previously `#[ignore]` because the dispatcher hardcoded
/// `agent_id` (PerAgent's per-thread var) and PerEvent kernels
/// produced broken WGSL. Slice δ part 1 made caster an explicit
/// operand; part 2 (this iteration) gates the lowering: PerAgent
/// rules lower caster to `AgentSelfId` as before, PerEvent rules
/// surface a typed `UnsupportedPhysicsStmt` so the user sees the
/// gap at compile time instead of getting an opaque naga error
/// from the runtime far from the design context.
///
/// **Still pending (slice δ part 3)**: actually synthesize the
/// caster from the event payload's actor field for PerEvent rules
/// — replace the typed-error branch with a real
/// `Read(EventField{actor})` lowering. That needs a convention
/// for which event field is the actor (or per-event opt-in) plus
/// resolution against the event-field-index registry.
///
/// This test now PASSES — it asserts the lowering fails with the
/// expected typed error rather than producing broken WGSL.
#[test]
fn apply_ability_in_per_event_rule_errors_at_lowering() {
    let src = "
        event Tick { }
        event Triggered { who: AgentId, ability_id: u32 }

        entity Hero : Agent { }

        physics DispatchOnTrigger {
          on Triggered { who: w, ability_id: a } {
            apply_ability a
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    // Lowering must surface a typed UnsupportedPhysicsStmt with
    // ast_label="ApplyAbility/PerEvent" — NOT proceed to emit and
    // produce broken WGSL.
    let outcome = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp);
    let err = outcome.expect_err(
        "PerEvent ApplyAbility must error at lowering time \
         (slice δ part 2 typed-error gate)",
    );
    let diags = format!("{err:?}");
    assert!(
        diags.contains("UnsupportedPhysicsStmt"),
        "expected UnsupportedPhysicsStmt diagnostic, got: {diags}"
    );
    assert!(
        diags.contains("ApplyAbility/PerEvent/no-caster"),
        "expected ast_label=\"ApplyAbility/PerEvent/no-caster\", got: {diags}"
    );
}

/// Slice ε resolver-coverage pin: a non-GPU-emittable expression
/// inside `by <caster>` (e.g. a `String` literal) must be rejected
/// at resolve time, not silently passed through to CG lowering
/// where the failure mode is harder to trace. Pins commit `c500eba7`.
///
/// Without that fix, `validate_physics_expr` walked only the
/// `ability` operand — a typo'd or non-emittable expression in
/// `caster`/`target` would slip through to the CG layer and
/// surface as a downstream type error far from the source location.
#[test]
fn apply_ability_by_non_emittable_caster_rejected_at_resolve() {
    let src = r#"
        event Tick { }
        entity Hero : Agent { }

        physics Bad @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self) by "not-a-real-caster"
          }
        }
    "#;
    let program = dsl_compiler::parse(src).expect("parse");
    let outcome = dsl_ast::resolve::resolve(program);
    let err = outcome.expect_err(
        "non-GPU-emittable expression in `by <caster>` must surface \
         as a typed resolve error (slice ε resolver coverage)",
    );
    let diags = format!("{err:?}");
    assert!(
        diags.contains("NotGpuEmittable") || diags.contains("String"),
        "expected NotGpuEmittable diagnostic mentioning String literal, got: {diags}"
    );
}

/// Slice ε part 1: explicit `target <expr>` syntax. The dispatcher
/// writes the caster slot into chronicle payload word 2 (actor) and
/// the target slot into payload word 3 — distinct values when the
/// source supplies them. This unblocks chronicle records where the
/// caster ≠ target (the slice-γ self-cast default coalesces them).
///
/// Pinned by:
///   - kernel body has BOTH `caster_slot` AND `target_slot` lets,
///   - chronicle write at payload word 3 references `(target_slot)`,
///     not `(caster_slot)` (slice-ε behavior),
///   - naga full validator passes the kernel.
#[test]
fn apply_ability_per_event_with_target_lowers_distinctly() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }
        event Triggered { who: AgentId, victim: AgentId, ability_id: u32 }

        entity Hero : Agent { }

        physics DispatchOnTrigger {
          on Triggered { who: w, victim: v, ability_id: a } {
            apply_ability a by w target v
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("PerEvent ApplyAbility with `by w target v` lowers");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "DispatchOnTrigger")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");
    // Both let-bindings present.
    assert!(
        body.contains("let caster_slot: u32"),
        "dispatcher must emit `let caster_slot` from the by-operand;\n{body}"
    );
    assert!(
        body.contains("let target_slot: u32"),
        "dispatcher must emit `let target_slot` from the target-operand;\n{body}"
    );
    // Chronicle payload word 3 (target slot) reads target_slot.
    assert!(
        body.contains("atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));"),
        "chronicle word 3 must write target_slot when distinct target is supplied;\n{body}"
    );

    // Naga full validator over every emitted kernel.
    let mut errs = Vec::new();
    for (name, kernel_body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(kernel_body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "PerEvent + `by w target v` variant emitted {} naga-rejected kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Slice δ part 3 (#161): explicit `by <caster>` syntax unblocks
/// PerEvent ApplyAbility. The same rule that errored above (no
/// caster context for PerEvent kernel) now lowers cleanly when the
/// source supplies an explicit caster from the event-pattern
/// destructuring (`who: w` → caster = `w`).
///
/// Pinned by:
///   - lowering reaches emit (no UnsupportedPhysicsStmt error),
///   - emitted WGSL passes naga's full validator (the caster local
///     read lowers through the same expression path as any other
///     event-pattern binding, so naga sees a well-formed reference
///     to the kernel-bound payload field rather than the prior
///     undeclared `agent_id`).
#[test]
fn apply_ability_per_event_with_by_caster_lowers_cleanly() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }
        event Triggered { who: AgentId, ability_id: u32 }

        entity Hero : Agent { }

        physics DispatchOnTrigger {
          on Triggered { who: w, ability_id: a } {
            apply_ability a by w
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("PerEvent ApplyAbility with `by w` lowers cleanly");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    // Naga validator over every emitted kernel — proves the kernel
    // is well-formed end-to-end, not just that lowering didn't error.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "PerEvent + `by w` variant emitted {} naga-rejected kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Variant: `apply_ability` + regular `emit` in the same rule body.
/// Both write to `event_ring` — the dispatcher's chronicle writes
/// (slice-γ kinds 26..=32) and the user-declared `emit Pinged { ... }`
/// share the same SoA buffer. The BGL composer must wire `event_ring`
/// + `event_tail` exactly once even when both producers contribute
/// to the same kernel.
///
/// Without correct binding deduplication, the kernel would either
/// declare `event_ring` twice (WGSL rejects) or once with the wrong
/// access mode (atomic vs non-atomic). Both surface here as naga
/// errors; if neither fires, the dedup is correct.
#[test]
fn apply_ability_alongside_regular_emit_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }
        event Pinged { who: AgentId, when: u32 }

        entity Hero : Agent { }

        physics MixedDispatch @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self)
            emit Pinged { who: self, when: world.tick }
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "MixedDispatch")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");
    // event_ring should appear exactly once as a binding declaration
    // (the BGL composer dedups). Counting `var<storage` declarations
    // mentioning `event_ring` is a coarse-but-sufficient proxy.
    let event_ring_decls = body
        .lines()
        .filter(|l| l.contains("var<storage") && l.contains("event_ring"))
        .count();
    assert_eq!(
        event_ring_decls, 1,
        "event_ring binding must dedup to exactly one declaration \
         even when ApplyAbility (chronicle writes 26..=32) and regular \
         emit (Pinged) both target it; found {event_ring_decls};\n{body}"
    );

    // Naga validator over every emitted kernel.
    let mut errs = Vec::new();
    for (name, kernel_body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(kernel_body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse failed: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate failed: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "mixed-emit variant emitted {} kernels naga rejects:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Stronger gate than `..._parses_through_naga` — runs naga's full
/// validator over the parsed module. Catches type errors, missing
/// `@binding(N) @group(0)` annotations, atomic-handle misuse,
/// access-mode mismatches, etc. that pure parsing misses.
///
/// `Capabilities::default()` matches WebGPU's baseline; the dispatcher
/// uses no extension features (atomic u32 ops on storage are baseline).
#[test]
fn apply_ability_smoke_kernel_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse failed: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate failed: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "apply_ability_smoke emitted {} kernels that fail naga \
         validation (parse OK, validator NOT OK — type / binding / \
         atomic-handle issues):\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Slice ε pin for PerEvent + `by w` + no `target`: lowering
/// defaults target to caster (slice-γ self-cast), so the
/// dispatcher emits both `let caster_slot` and `let target_slot`
/// from the same source expression. Confirms the implicit-target
/// default works in PerEvent shape, not just PerAgent.
///
/// Without this test, a regression that fixed the implicit-target
/// default for PerAgent only (broke PerEvent — which previously
/// errored at lowering rather than reaching emit) would slip
/// through to silently broken WGSL when the user later adds
/// PerEvent rules.
#[test]
fn per_event_by_caster_without_explicit_target_emits_both_slots() {
    let src = "
        event Tick { }
        event Triggered { who: AgentId, ability_id: u32 }

        entity Hero : Agent { }

        physics DispatchOnTrigger {
          on Triggered { who: w, ability_id: a } {
            apply_ability a by w
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("PerEvent ApplyAbility with `by w` lowers");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "DispatchOnTrigger")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");

    // Both let-bindings present (implicit target = caster).
    assert!(
        body.contains("let caster_slot: u32"),
        "PerEvent + `by w` must emit `let caster_slot` from the by-operand;\n{body}"
    );
    assert!(
        body.contains("let target_slot: u32"),
        "PerEvent + `by w` (no explicit target) must still emit `let \
         target_slot` — implicit default target = caster (slice-γ \
         self-cast preserved);\n{body}"
    );
}

/// Slice ε regression pin: even when the source omits `target
/// <expr>`, lowering must still populate the target operand
/// (defaulting to caster) so the dispatcher emit produces BOTH
/// `let caster_slot` AND `let target_slot` lets. Without this,
/// a regression that left `target = None` in the lowered IR would
/// produce a kernel missing the target_slot binding (silent — the
/// payload word 3 would reference an undeclared identifier).
///
/// The smoke fixture's first rule (`DispatchAbility`) uses implicit
/// target; this test pins that the implicit-target default lands
/// correctly through to emit.
#[test]
fn smoke_fixture_implicit_target_still_emits_target_slot_let() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    let body = kernel_body_containing(&art, "DispatchAbility")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");

    // Both lets present — implicit target defaults to caster, so
    // the dispatcher emit produces both bindings (with the same
    // resolved expression). A regression that left target = None
    // would break the format-string injection.
    assert!(
        body.contains("let caster_slot: u32"),
        "dispatcher must emit `let caster_slot`;\n{body}"
    );
    assert!(
        body.contains("let target_slot: u32"),
        "dispatcher must emit `let target_slot` even when source \
         omits `target <expr>` (implicit-target default = caster);\n{body}"
    );
}

/// The smoke fixture has TWO PerAgent rules (`DispatchAbility` with
/// implicit target + `DispatchAbilityExplicit` with `by self target
/// self`). They emit as two separate kernels (`physics_DispatchAbility`
/// + `physics_DispatchAbilityExplicit`), each carrying its own
/// dispatcher block.
///
/// Pins that the explicit-clause rule:
///   1. produces its own kernel (not silently dropped during scheduling),
///   2. carries a complete 7-arm dispatcher block in that kernel,
///   3. emits both `caster_slot` + `target_slot` lets like the
///      implicit-target rule.
///
/// Without (3), the slice-ε explicit-operand surface in .sim corpus
/// could regress at lowering and ship blank dispatcher arms.
#[test]
fn smoke_fixture_explicit_rule_kernel_has_full_dispatcher() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    let explicit_body = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("DispatchAbilityExplicit"))
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| {
            panic!(
                "expected `physics_DispatchAbilityExplicit` kernel in artifacts; \
                 a missing kernel here means the second rule was dropped \
                 during scheduling. Available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });

    // Full 32-arm chronicle dispatch in the explicit-clause kernel,
    // emitted twice (primary effect walk + Wave 1.5#9 nested-effect
    // walk = 64 chronicle slot acquisitions per `apply_ability` stmt).
    // (Was 7 pre-Bleed-swap; SelfDamage=39 added 2026-05-06; LifeSteal=40
    // added by Vampirize verb swap, mirror of Bleed; DamageModify=41
    // added by Fortify verb swap, mirror of Vampirize; Execute=42 added
    // by Reap verb swap, mirror of Fortify — closes the slice across
    // all 8 duel_abilities verbs. Wave 1.5#9 nested-effect dispatch
    // doubles that, 2026-05-06. Wave 2 piece 1 adds Root=43, Silence=44,
    // Fear=45, Taunt=46 — count goes 22 → 30. Wave 2 piece 2 adds
    // Dash=47, Blink=48, Knockback=49, Pull=50 — count goes 30 → 38.
    // Wave 1.5+ adds DamageOverTime=51, HealOverTime=52, TimedShield=53
    // — count goes 38 → 44. Extended-status slice adds Stealth=54,
    // Charm=55, Grounded=56, Suppress=57 — count goes 44 → 52.
    // Slice γ tail adds Buff=58, Harvest=59, PlaceVoxel=60, Reflect=61
    // — count goes 52 → 60. Slice γ closer adds Summon=62 — count
    // goes 60 → 62. Wave 3 ToM Phase 1 adds PlantBelief=63 — count
    // goes 62 → 64. Wave 3 ToM Phase 3 adds Observe=64 — count goes
    // 64 → 66. Wave 3 ToM Phase 3.5 adds Scry=65 + Reveal=66 — count
    // goes 66 → 70. Wave 3 ToM Phase 4 adds Disguise=67 + Decoy=68 +
    // EraseBelief=69 — count goes 70 → 76. Lift A adds TravelTo=70 —
    // count goes 76 → 78. Lift B adds Recipe=71 + WearTool=72 — count
    // goes 78 → 82. Lift C adds Propose=73 + Announce=74 — count goes
    // 82 → 86. Lift D adds GainSkill=75 + CreateObligation=76 — count
    // goes 86 → 90.
    // NO `// TODO slice γ` arms remain; the slice is closed.)
    let slot_acquisitions = explicit_body
        .matches("let _slot: u32 = atomicAdd(&event_tail[0], 1u);")
        .count();
    assert_eq!(
        slot_acquisitions, 92,
        "DispatchAbilityExplicit kernel must carry all 92 chronicle slot \
         acquisitions (46 chronicle-bearing variants × {{primary, nested}} \
         walk — Plan G added CastBegin=46); got {slot_acquisitions}\nbody:\n{explicit_body}"
    );

    // Slice ε surface: explicit `by self target self` lowers to both
    // operand lets — same shape as the implicit default but proves
    // the explicit-clause path through parser → resolve → lower → emit
    // doesn't regress on the .sim corpus surface.
    assert!(
        explicit_body.contains("let caster_slot: u32"),
        "explicit-clause kernel must emit `let caster_slot`;\n{explicit_body}"
    );
    assert!(
        explicit_body.contains("let target_slot: u32"),
        "explicit-clause kernel must emit `let target_slot` (the \
         `target self` clause should produce a let, not be dropped);\n\
         {explicit_body}"
    );
}

/// Two `apply_ability` statements with identical operands inside a
/// single rule body must each emit their own dispatcher block. CSE or
/// statement deduplication on `CgStmt::ApplyAbility` would silently
/// halve the chronicle write throughput — same lowered IR ≠ same
/// runtime semantics, since each statement represents a separate
/// dispatch operation.
///
/// Pinned by inline source (the smoke .sim file's two statements live
/// in different rules; this exercises the same-rule case).
#[test]
fn back_to_back_apply_ability_in_one_rule_emits_two_dispatcher_blocks() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics DoubleDispatch @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self) by self target self
            apply_ability agents.level(self) by self target self
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("two back-to-back ApplyAbility lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "DoubleDispatch")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");

    // Two dispatcher loops, one per statement.
    let dispatcher_loops = body.matches("for (var i: u32 = 0u; i < 6u;").count();
    assert_eq!(
        dispatcher_loops, 2,
        "two back-to-back apply_ability statements must each emit \
         their own dispatcher slot loop. Got {dispatcher_loops} — a \
         value of 1 means CSE/dedup collapsed the second statement, \
         halving chronicle throughput;\nbody:\n{body}"
    );

    // 128 slot acquisitions (32 chronicle arms × 2 statements ×
    // {primary, nested} — Wave 1.5#9 doubled this with the nested
    // walk, 2026-05-06; Wave 2 piece 1 added 4 control-status arms,
    // bumping 44 → 60; Wave 2 piece 2 added 4 movement arms, bumping
    // 60 → 76; Wave 1.5+ added 3 multi-tick arms, bumping 76 → 88;
    // extended-status slice added 4 status arms, bumping 88 → 104;
    // slice γ tail added 4 arms (Buff/Harvest/PlaceVoxel/Reflect),
    // bumping 104 → 120; slice γ closer added 1 arm (Summon),
    // bumping 120 → 124; Wave 3 ToM Phase 1 added 1 arm
    // (PlantBelief), bumping 124 → 128; Wave 3 ToM Phase 3 added 1
    // arm (Observe), bumping 128 → 132; Wave 3 ToM Phase 3.5 added 2
    // arms (Scry + Reveal), bumping 132 → 140; Wave 3 ToM Phase 4
    // added 3 arms (Disguise + Decoy + EraseBelief), bumping 140 → 152.
    // Lift A added 1 arm (TravelTo), bumping 152 → 156. Lift B added
    // 2 arms (Recipe + WearTool), bumping 156 → 164. Lift C added 2
    // arms (Propose + Announce), bumping 164 → 172. Lift D added 2
    // arms (GainSkill + CreateObligation), bumping 172 → 180. Plan G
    // added 1 arm (CastBegin), bumping 180 → 184.
    let slot_acquisitions = body
        .matches("let _slot: u32 = atomicAdd(&event_tail[0], 1u);")
        .count();
    assert_eq!(
        slot_acquisitions, 184,
        "expected 184 slot acquisitions (46 chronicle arms × 2 statements × \
         {{primary, nested}} walks); got {slot_acquisitions}\nbody:\n{body}"
    );

    // Naga sanity: the duplicated dispatcher emit doesn't introduce
    // duplicate-identifier conflicts (e.g. two `let _slot` shadowing
    // would still parse — check anyway since let-rebinding rules can
    // be subtle in WGSL).
    let module = naga::front::wgsl::parse_str(body)
        .unwrap_or_else(|e| panic!("naga parse failed for back-to-back kernel: {e}"));
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("naga validate failed: {e:?}"));
}

/// Companion to back_to_back: two `apply_ability` statements with
/// DIFFERENT operands in one body must also each emit their own
/// dispatcher block. Catches a different regression class than the
/// identical-operand pin: an operand-aware dedup that only collapsed
/// structurally-identical statements would pass that test but could
/// still fail if the IDs of the operands were what got hashed (a
/// regression where ExprIds rather than CgExpr structure was the
/// dedup key).
#[test]
fn back_to_back_apply_ability_with_distinct_operands_each_emit() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics DoubleDistinctDispatch @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self) by self target self
            apply_ability agents.level(self) by self target agents.level(self)
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("two distinct-operand ApplyAbility lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "DoubleDistinctDispatch")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");

    // 2 dispatcher loops + 20 slot acquisitions (same totals as the
    // identical-operand case — the structural shape is identical).
    let dispatcher_loops = body.matches("for (var i: u32 = 0u; i < 6u;").count();
    assert_eq!(
        dispatcher_loops, 2,
        "two distinct-operand apply_ability statements must each emit \
         their own dispatcher block; got {dispatcher_loops}\nbody:\n{body}"
    );

    let slot_acquisitions = body
        .matches("let _slot: u32 = atomicAdd(&event_tail[0], 1u);")
        .count();
    // Wave 1.5#9 + Wave 2 piece 1 + Wave 2 piece 2 + Wave 1.5+ +
    // extended-status slice + slice γ tail + slice γ closer + Wave 3
    // ToM Phase 1 (PlantBelief) + Wave 3 ToM Phase 3 (Observe) + Wave 3
    // ToM Phase 3.5 (Scry + Reveal) + Wave 3 ToM Phase 4 (Disguise +
    // Decoy + EraseBelief) + Lift A (TravelTo) + Lift B (Recipe +
    // WearTool) + Lift C (Propose + Announce) + Lift D (GainSkill +
    // CreateObligation) + Plan G (CastBegin): 46 chronicle arms × 2
    // statements × {primary, nested} walks = 184.
    assert_eq!(slot_acquisitions, 184);

    // Naga validates — different target_slot expressions in the two
    // dispatch blocks shouldn't introduce binding conflicts.
    let module = naga::front::wgsl::parse_str(body)
        .unwrap_or_else(|e| panic!("naga parse failed: {e}"));
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("naga validate failed: {e:?}"));
}

/// PerAgent slice-ε surface where `target` is a non-trivial SoA-field
/// read (`agents.level(self)`) instead of `self`. Confirms lowering
/// accepts any expression the physics-scope `lower_expr` supports as
/// the target operand — not just `AgentSelfId`. Same shape extends to
/// future `agents.attack_target(self)` style fields once a real u32
/// SoA column carries an agent id.
///
/// Pinned by:
///   - lowering reaches emit (no UnsupportedPhysicsStmt or expression-
///     scope rejection),
///   - emitted WGSL parses and validates through naga,
///   - chronicle word 3 reads `target_slot` (which itself was
///     bound from the SoA-field read, not from `agent_id`).
#[test]
fn per_agent_apply_ability_with_soa_field_target_validates() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics DispatchToFieldTarget @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self) by self target agents.level(self)
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("PerAgent ApplyAbility with SoA-field target should lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "DispatchToFieldTarget")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");

    // The target_slot binding must be present and the chronicle write
    // must read from it (not from agent_id directly — that would
    // indicate the SoA-field expression was discarded and target
    // collapsed back to caster).
    assert!(
        body.contains("let target_slot: u32"),
        "dispatcher must emit `let target_slot` from the SoA-field \
         target operand;\n{body}"
    );
    assert!(
        body.contains("atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));"),
        "chronicle word 3 must write target_slot (not agent_id) when \
         the source supplies a non-self target;\n{body}"
    );

    let mut errs = Vec::new();
    for (name, kernel_body) in &art.wgsl_files {
        match naga::front::wgsl::parse_str(kernel_body) {
            Ok(m) => {
                let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
                if let Err(e) = validator.validate(&m) {
                    errs.push(format!("  {name}: validate: {e:?}"));
                }
            }
            Err(e) => {
                errs.push(format!("  {name}: parse: {e}"));
            }
        }
    }
    assert!(
        errs.is_empty(),
        "SoA-field-target variant emitted {} naga-rejected kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Slice ε type-check (`well_formed.rs` ApplyAbility arm): the
/// target operand must lower to `CgTy::U32` or `CgTy::AgentId`.
/// `target 3.14` (an f32 literal) used to silently truncate through
/// the dispatcher's `u32(...)` coercion in the emit format string,
/// producing garbage agent ids in chronicle records (`u32(3.14) = 3`
/// in WGSL — naga accepted it). This pin asserts the well-formed
/// pass now rejects it at lowering, before emit.
///
/// Catches regressions where the type-check return values get
/// silently discarded again (the original well_formed code had
/// `let _ = type_check(...)` for this arm).
#[test]
fn apply_ability_rejects_f32_target_at_lowering() {
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics BadTarget @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self) by self target 3.14
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let err = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect_err(
            "lowering must reject f32 target — silent acceptance \
             produces garbage agent ids in chronicle records via \
             WGSL `u32(3.14)` truncation",
        );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("TypeMismatch") || msg.contains("F32"),
        "expected TypeMismatch / F32 mention in error; got: {msg}"
    );
}

/// Slice ε type-check: ability operand is AbilityId (NonZeroU32
/// wrapper), NOT AgentId. Mechanically both types fit in 32 bits and
/// the dispatcher's `u32(...)` coercion would happily accept either,
/// but semantically passing an agent slot as the ability id is a
/// typo (`apply_ability self` would treat the caster's slot index
/// as a registry slot, looking up effects from a wrong/garbage row).
///
/// Tighter than caster/target's check (which accepts both U32 and
/// AgentId, since u32 SoA fields can semantically carry agent ids).
#[test]
fn apply_ability_rejects_agentid_as_ability_operand() {
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics WeirdAbility @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability self by self target self
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let err = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect_err("lowering must reject AgentId as ability operand");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("TypeMismatch") || msg.contains("AgentId"),
        "expected TypeMismatch / AgentId mention; got: {msg}"
    );
}

/// Probe: literal `0` as ability id. AbilityId wraps NonZeroU32, so
/// runtime `AbilityId::new(0)` returns None and dispatch silently
/// skips. Compile-time accepts because 0u32 is a valid u32 literal —
/// but it's a guaranteed runtime no-op + dead code.
///
/// Pinned as documentation; if the well-formed pass grows constant-
/// folding for this case (rejecting literal 0 ability), this test
/// flips to `expect_err`.
#[test]
fn probe_apply_ability_with_zero_literal() {
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics ZeroAbility @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability 0 by self target self
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(_) => eprintln!(
            "[probe] literal 0 accepted as ability id. Runtime \
             AbilityId::new(0) returns None — dispatch silently \
             skips. A constant-fold check at well-formed could \
             reject this at compile time as guaranteed dead code."
        ),
        Err(e) => eprintln!(
            "[probe] lowering rejected literal 0 ability (good): {e:?}"
        ),
    }
}

/// `world.tick` is a valid u32 source under the namespace registry's
/// type tag (`NamespaceField { ty: U32 }`), even though `CgTy::Tick`
/// exists for the scoring/anchor case. The slice-ε type-check
/// permits it as target — semantically dubious (using the tick value
/// as an agent slot index), but mechanically u32. This pin documents
/// that.
///
/// If the namespace registry ever upgrades `world.tick` to
/// `CgTy::Tick` (more accurate type tag), this test flips and the
/// type-check would need updating to reject Tick — different decision
/// to make at that point.
#[test]
fn apply_ability_accepts_world_tick_as_target_today() {
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics TickAsTarget @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self) by self target world.tick
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("world.tick is registered as CgTy::U32 in NamespaceRegistry — \
                 acceptable as target operand under the slice-ε type-check");
}

/// Bool target (e.g., a typo where the user wrote `target self.alive`
/// instead of `target self`). Without the type-check fix, this would
/// also silently coerce through `u32(bool)` in emit. Asserts the
/// fix catches non-numeric types beyond just F32.
#[test]
fn apply_ability_rejects_bool_target_at_lowering() {
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics BadBoolTarget @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self) by self target self.alive
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let err = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect_err(
            "lowering must reject Bool target — `self.alive` is a bool \
             SoA field; coercing through u32() would produce 0 or 1, \
             not an agent id"
        );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("TypeMismatch") || msg.contains("Bool"),
        "expected TypeMismatch / Bool mention; got: {msg}"
    );
}

/// Same regression pin for the caster operand. Symmetric to the
/// target check — both operands route through the same `u32(...)`
/// coercion in emit.
#[test]
fn apply_ability_rejects_f32_caster_at_lowering() {
    let src = "
        event Tick { }
        entity Hero : Agent { }

        physics BadCaster @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self) by 3.14 target self
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let err = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect_err("lowering must reject f32 caster");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("TypeMismatch") || msg.contains("F32"),
        "expected TypeMismatch / F32 mention in error; got: {msg}"
    );
}

/// Pin the BGL composer's wiring of `event_ring` + `event_tail` into
/// the dispatcher kernel. Without these bindings, the chronicle writes
/// emitted by the dispatcher arms would reference undeclared identifiers
/// at WGSL compile time. Recording an `EventRing(Append)` write on
/// ApplyAbility-bearing ops (commit `1779b0e6`) is what hooks the
/// composer; this assertion tests that the hook still fires after the
/// rest of the pipeline runs.
#[test]
fn apply_ability_smoke_kernel_binds_event_ring_and_event_tail() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    let body = kernel_body_containing(&art, "DispatchAbility")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });

    // The composer emits `var<storage, ...> event_ring : array<...>;`
    // and `var<storage, ...> event_tail : array<...>;` declarations
    // after running the EventRing(Append)+sibling-event_tail synthesis
    // path in `cg::emit::kernel`. Match the bare `event_ring` /
    // `event_tail` identifier rather than the full type signature
    // (`array<atomic<u32>>` vs `array<u32>` may evolve as the binding
    // metadata refines), so the assertion is robust to wgsl-ty drift.
    assert!(
        body.contains("event_ring"),
        "dispatcher kernel must bind event_ring (the chronicle writes \
         in the slice-γ arms reference it);\n\
         got body:\n{body}"
    );
    assert!(
        body.contains("event_tail"),
        "dispatcher kernel must bind event_tail (the dispatcher's \
         atomicAdd slot acquisition references it);\n\
         got body:\n{body}"
    );

    // The two bindings appear as WGSL `var<storage, ...>` declarations
    // (one each). At least one declaration per identifier must be
    // present — multiple references in the chronicle writes are fine
    // but the binding declaration itself is what the BGL composer
    // emits exactly once.
    assert!(
        body.matches("var<storage").count() >= 2,
        "expected ≥2 storage binding declarations (event_ring + \
         event_tail at minimum);\n\
         got body:\n{body}"
    );
}

/// Wave 1.7 / task #138: a `verb` body accepts `apply_ability <expr>
/// [by <c>] [target <t>]` as an alternative to `emit <Event> { ... }`.
///
/// This test pins the parser → resolve → CG-lower wiring for the new
/// surface — once it lands, the next bite (Strike's verb body in
/// `assets/sim/duel_abilities.sim`) can swap from
/// `emit Damaged { ... }` to `apply_ability self.action_ability` /
/// equivalent without a separate parser change.
///
/// What this proves:
///   - The verb-body parser accepts `apply_ability` (not just `emit`).
///   - The resolver lifts it into `IrStmt::ApplyAbility { ability,
///     caster, target }` inside the verb's `body` vec.
///   - The verb expander injects the cascade physics handler whose
///     gated body carries the `IrStmt::ApplyAbility` verbatim.
///   - The CG lowering pipeline (`lower_compilation_to_cg` →
///     `synthesize_schedule` → `emit_cg_program`) succeeds end-to-end
///     against that synthesised handler — i.e. the verb-body
///     ApplyAbility flows through the same well-formed type-check +
///     dispatcher arms the physics-scope ApplyAbility uses (see
///     `crates/dsl_compiler/src/cg/well_formed.rs::type_check_op`).
///
/// We use `agents.level(self)` for the ability operand (a u32 SoA
/// accessor; same pattern `assets/sim/apply_ability_smoke.sim` uses).
/// `self` for caster / target satisfies the CgTy::AgentId arm of the
/// well-formed slot type-check.
#[test]
fn verb_body_with_apply_ability_lowers_cleanly() {
    let src = r#"
event Tick { }

entity Hero : Agent { }

scoring {
  Cast = 1.0
}

verb Cast(self) =
  action Cast
  when  self.alive
  apply_ability agents.level(self) by self target self
  score 1.0
"#;
    let program = dsl_compiler::parse(src).expect("verb body with apply_ability parses");
    let comp = dsl_ast::resolve::resolve(program)
        .expect("verb body with apply_ability resolves");

    // Surface assertion — the resolver populated the verb's body with
    // exactly one `IrStmt::ApplyAbility`. This pins the new resolve
    // path (not just "lowering didn't blow up").
    assert_eq!(comp.verbs.len(), 1, "expected one verb");
    let verb = &comp.verbs[0];
    assert_eq!(verb.name, "Cast");
    assert_eq!(
        verb.body.len(),
        1,
        "expected verb body to carry one ApplyAbility stmt; got {} stmts",
        verb.body.len(),
    );
    match &verb.body[0] {
        dsl_ast::ir::IrStmt::ApplyAbility { caster, target, .. } => {
            assert!(
                caster.is_some(),
                "expected `by self` to populate the caster operand"
            );
            assert!(
                target.is_some(),
                "expected `target self` to populate the target operand"
            );
        }
        other => panic!(
            "expected verb body[0] = IrStmt::ApplyAbility; got {other:?}",
        ),
    }

    // Lower → schedule → emit must succeed AND the synthesised
    // verb_chronicle_Cast kernel must carry the dispatcher block —
    // proves the verb-body ApplyAbility reaches GPU emit, not just
    // that lowering didn't blow up. The verb expander wraps
    // the ApplyAbility in the synthesised `verb_chronicle_Cast`
    // physics handler and the standard physics-lowering pipeline
    // takes it from there.
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("verb body with apply_ability lowers to CG");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit succeeds for verb-body apply_ability");

    // The verb expander wraps the body into a `verb_chronicle_<Name>`
    // physics handler. That handler must contain the dispatcher loop
    // scaffolding — proves the verb-body ApplyAbility actually reached
    // GPU emit, not just that lowering didn't blow up.
    let body = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("verb_chronicle_Cast"))
        .map(|(_, b)| b.as_str())
        .unwrap_or_else(|| {
            panic!(
                "no verb_chronicle_Cast kernel in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>(),
            );
        });
    assert!(
        body.contains("for (var i: u32 = 0u; i < 6u;"),
        "verb_chronicle_Cast must carry the dispatcher slot loop \
         (MAX_EFFECTS_PER_PROGRAM = 6);\nbody:\n{body}"
    );
    assert!(
        body.contains("let caster_slot: u32"),
        "verb_chronicle_Cast must emit the caster_slot let from the \
         `by self` clause;\nbody:\n{body}"
    );
    assert!(
        body.contains("let target_slot: u32"),
        "verb_chronicle_Cast must emit the target_slot let from the \
         `target self` clause;\nbody:\n{body}"
    );
}

/// Companion to [`verb_body_with_apply_ability_lowers_cleanly`]: the
/// shorthand surface (no `by` / `target`) parses + resolves cleanly
/// into an `IrStmt::ApplyAbility { caster: None, target: None }`. We
/// don't drive the full lower here — the implicit-caster default for
/// the synthesised cascade handler depends on the physics-scope arm
/// that resolves `caster: None` → per-thread agent. The parse +
/// resolve gate is enough to pin the source surface.
#[test]
fn verb_body_apply_ability_without_by_target_parses_and_resolves() {
    let src = r#"
event Tick { }

entity Hero : Agent { }

verb Cast(self) =
  action Cast
  when  self.alive
  apply_ability agents.level(self)
"#;
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    assert_eq!(comp.verbs.len(), 1);
    match &comp.verbs[0].body[0] {
        dsl_ast::ir::IrStmt::ApplyAbility { caster, target, .. } => {
            assert!(caster.is_none(), "no `by` clause → caster: None");
            assert!(target.is_none(), "no `target` clause → target: None");
        }
        other => panic!("expected ApplyAbility; got {other:?}"),
    }
}
