//! Gap #3 follow-up coverage — `config.<ns>.<u32_field>` in arithmetic
//! position lowers as `u32` not `f32`.
//!
//! diplomacy_probe Gap #3 was first closed by commit `16905527`; the
//! fix routes `Read(ConfigConst)` through
//! `ExprArena::config_const_ty(id)`, which `CgProgram` overrides to
//! consult `config_const_values`. The existing pin lives in
//! `stress_fixtures_compile.rs::config_u32_field_lowers_typed_in_arithmetic_position`
//! and exercises the verb-`when` mask path against an inline source
//! with one config field.
//!
//! This test extends coverage to the canonical "duel-style" cooldown
//! shape — multiple `cooldown_*: u32` config fields, multiple verbs
//! with `when (world.tick % config.combat.cooldown_<verb> == 0)` mod
//! gates — that the `duel_1v1.sim` / `duel_25v25.sim` /
//! `mass_battle_100v100.sim` / `boss_fight.sim` /
//! `duel_abilities.sim` fixtures use after the workaround
//! restoration. A future regression that broke any one cadence would
//! drop the corresponding MaskPredicate op and trip the count
//! assertion below.
//!
//! WGSL fingerprint: every u32-declared cooldown const must emit with
//! the `u`-suffix (`: u32 = <n>u;`) — matching the trade_market
//! hygiene pattern. Each verb's mask body must contain a
//! `(tick % config_<id>)` substring (the typed-Mod arm's WGSL shape).

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::op::ComputeOpKind;
use dsl_compiler::cg::program::CgProgram;

fn compile_inline(src: &str) -> (CgProgram, EmittedArtifacts) {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = lower_compilation_to_cg(&comp).unwrap_or_else(|o| {
        let diag_text = o
            .diagnostics
            .iter()
            .map(|d| format!("  {d}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "lower expected clean (Gap #3 fix); got {} diagnostics:\n{diag_text}",
            o.diagnostics.len(),
        );
    });
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art =
        dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit");
    (cg, art)
}

#[test]
fn duel_style_three_u32_cooldowns_lower_clean() {
    // Mirrors duel_1v1.sim's three-verb cascade: each verb gates on a
    // distinct u32 cooldown config field. Pre-Gap-#3-fix the rhs of
    // the Mod op resolved to f32 and every mask predicate failed
    // BinaryOperandTyMismatch — silently dropping the verb.
    let src = r#"
event Tick { }
@replayable @gpu_amenable
event Damaged { source: AgentId, target: AgentId, amount: f32 }
@replayable @gpu_amenable
event Healed { source: AgentId, target: AgentId, amount: f32 }
entity Combatant : Agent { pos: vec3, vel: vec3, }
config combat {
  strike_damage:    f32 = 12.0,
  spell_damage:     f32 = 25.0,
  heal_amount:      f32 = 18.0,
  cooldown_strike:  u32 = 2,
  cooldown_spell:   u32 = 5,
  cooldown_heal:    u32 = 4,
}
verb Strike(self, target: Agent) =
  action StrikeAction
  when (self.alive
        && target.alive
        && target != self
        && (world.tick % config.combat.cooldown_strike == 0))
  emit Damaged { source: self, target: target, amount: config.combat.strike_damage }
  score 1.0
verb Spell(self, target: Agent) =
  action SpellAction
  when (self.alive
        && target.alive
        && target != self
        && (world.tick % config.combat.cooldown_spell == 0))
  emit Damaged { source: self, target: target, amount: config.combat.spell_damage }
  score (config.combat.spell_damage * 2.0 - target.hp)
verb Heal(self, target: Agent) =
  action HealAction
  when (self.alive
        && target == self
        && (world.tick % config.combat.cooldown_heal == 0))
  emit Healed { source: self, target: self, amount: config.combat.heal_amount }
  score 1.0
"#;
    let (cg, art) = compile_inline(src);

    // Three verbs → three MaskPredicate ops survive lowering. Pre-fix
    // BinaryOperandTyMismatch would have dropped each one.
    let mask_count = cg
        .ops
        .iter()
        .filter(|op| matches!(&op.kind, ComputeOpKind::MaskPredicate { .. }))
        .count();
    assert_eq!(
        mask_count, 3,
        "expected 3 MaskPredicate ops (one per verb); got {mask_count}",
    );

    // The three u32 cooldown constants must emit with the `u`
    // suffix (typed-routing fingerprint, mirrors the trade_market
    // hygiene pattern). Search across all kernel bodies.
    let all_bodies: String = art
        .wgsl_files
        .values()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    for (n, label) in [
        (2u32, "cooldown_strike"),
        (5u32, "cooldown_spell"),
        (4u32, "cooldown_heal"),
    ] {
        assert!(
            all_bodies.contains(&format!(": u32 = {n}u;")),
            "expected `: u32 = {n}u;` constant for {label}; not found in any kernel body",
        );
    }

    // The typed Mod arm emits `(tick % config_<id>)`. There must be
    // one occurrence per verb (three total) somewhere in the WGSL —
    // pre-fix the f32 promotion would have either failed lowering or
    // emitted a `bitcast<f32>` indirection. The exact fused-kernel
    // home of each occurrence drifts as the schedule composer
    // evolves, so we count globally across all bodies.
    let total_typed_mods = all_bodies.matches("(tick % config_").count();
    assert_eq!(
        total_typed_mods, 3,
        "expected exactly 3 `(tick % config_<id>)` occurrences (one per verb); got {total_typed_mods}",
    );
}
