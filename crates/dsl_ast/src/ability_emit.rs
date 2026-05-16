//! AbilityDecl → source-text emitter.
//!
//! Inverse of `ability_parser.rs`: given an in-memory `AbilityDecl`,
//! produce a `.ability` source string the parser will round-trip back
//! into an equivalent AST. Used by:
//!
//!   * `tests/ability_grammar_walker.rs` — generative grammar coverage
//!     (every header / effect-arg / target-mode / lifetime variant gets
//!     emitted → parsed → asserted equivalent).
//!   * Future hot-reload paths (programmatic ability mutation cycle).
//!
//! Scope: emit only the surface the parser today admits. Opaque blocks
//! (`deliver { … }`, `morph { … }`, `template <N>(…)`,
//! `structure <N>(…)`, `program { cast { … } effect { … } }`) are out
//! of scope — they live as `raw: String` slots and round-trip
//! verbatim; the grammar walker leaves them at `None`.

use crate::ast::*;
use std::fmt::Write;

/// Emit a full `.ability` file containing one `AbilityDecl`. The
/// emitted source is whitespace-canonical (4-space indent, one
/// statement per line) so byte-for-byte comparison of a parse →
/// emit → parse cycle is meaningful.
pub fn emit_ability_file_single(d: &AbilityDecl) -> String {
    let mut out = String::new();
    emit_ability_decl(d, &mut out);
    out
}

fn emit_ability_decl(d: &AbilityDecl, out: &mut String) {
    writeln!(out, "ability {} {{", d.name).unwrap();
    for h in &d.headers {
        emit_header(h, out);
    }
    if !d.headers.is_empty() && !d.effects.is_empty() {
        writeln!(out).unwrap();
    }
    for e in &d.effects {
        emit_effect_stmt(e, 1, out);
    }
    writeln!(out, "}}").unwrap();
}

fn emit_header(h: &AbilityHeader, out: &mut String) {
    match h {
        AbilityHeader::Target(t) => writeln!(out, "    target: {}", target_str(*t)).unwrap(),
        AbilityHeader::Range(r) => writeln!(out, "    range: {}", fmt_f32(*r)).unwrap(),
        AbilityHeader::Cooldown(d, phase) => {
            let phase_suffix = match phase {
                None => String::new(),
                Some(CooldownPhase::Cast) => " @ cast".to_string(),
                Some(CooldownPhase::Resolve) => " @ resolve".to_string(),
                Some(CooldownPhase::Interrupt) => " @ interrupt".to_string(),
            };
            writeln!(out, "    cooldown: {}{}", duration_str(*d), phase_suffix).unwrap();
        }
        AbilityHeader::Cast(d) => writeln!(out, "    cast: {}", duration_str(*d)).unwrap(),
        AbilityHeader::Hint(h) => writeln!(out, "    hint: {}", hint_str(*h)).unwrap(),
        AbilityHeader::Cost(cs) => writeln!(out, "    cost: {}", cost_str(cs)).unwrap(),
        AbilityHeader::Charges(n) => writeln!(out, "    charges: {n}").unwrap(),
        AbilityHeader::Recharge(d) => writeln!(out, "    recharge: {}", duration_str(*d)).unwrap(),
        AbilityHeader::Toggle => writeln!(out, "    toggle").unwrap(),
        AbilityHeader::Recast(r) => {
            let v = match r {
                RecastValue::Count(n) => format!("{n}"),
                RecastValue::Duration(d) => duration_str(*d),
            };
            writeln!(out, "    recast: {v}").unwrap();
        }
        AbilityHeader::RecastWindow(d) => {
            writeln!(out, "    recast_window: {}", duration_str(*d)).unwrap()
        }
    }
}

fn emit_effect_stmt(e: &EffectStmt, indent: usize, out: &mut String) {
    let pad = "    ".repeat(indent);
    out.push_str(&pad);
    out.push_str(&e.verb);
    for arg in &e.args {
        out.push(' ');
        out.push_str(&effect_arg_str(arg));
    }
    if let Some(a) = &e.area {
        write!(out, " in {}", a.shape).unwrap();
        if !a.args.is_empty() {
            out.push('(');
            for (i, v) in a.args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&fmt_f32(*v));
            }
            out.push(')');
        }
    }
    for tag in &e.tags {
        write!(out, " [{}: {}]", tag.name, fmt_f32(tag.value)).unwrap();
    }
    if let Some(d) = &e.duration {
        write!(out, " for {}", duration_str(d.duration)).unwrap();
    }
    if let Some(c) = &e.chance {
        write!(out, " chance {}%", fmt_f32(c.p * 100.0)).unwrap();
    }
    if let Some(stk) = &e.stacking {
        let s = match stk {
            StackingMode::Refresh => "refresh",
            StackingMode::Stack => "stack",
            StackingMode::Extend => "extend",
        };
        write!(out, " stacking {s}").unwrap();
    }
    for sc in &e.scalings {
        write!(out, " + {}% {}", fmt_f32(sc.percent), sc.stat_ref).unwrap();
    }
    if let Some(lt) = &e.lifetime {
        match lt {
            EffectLifetime::UntilCasterDies { .. } => out.push_str(" until_caster_dies"),
            EffectLifetime::DamageableHp { hp, .. } => {
                write!(out, " damageable_hp({})", fmt_f32(*hp)).unwrap()
            }
            EffectLifetime::BreakOnDamage { .. } => out.push_str(" break_on_damage"),
        }
    }
    if let Some(cond) = &e.condition {
        write!(out, " when {}", cond.when_cond).unwrap();
        if let Some(els) = &cond.else_cond {
            write!(out, " else {els}").unwrap();
        }
    }
    if e.nested.is_empty() {
        writeln!(out).unwrap();
    } else {
        out.push_str(" {\n");
        for n in &e.nested {
            emit_effect_stmt(n, indent + 1, out);
        }
        writeln!(out, "{}}}", pad).unwrap();
    }
}

fn target_str(t: TargetMode) -> &'static str {
    match t {
        TargetMode::Enemy => "enemy",
        TargetMode::Self_ => "self",
        TargetMode::Ally => "ally",
        TargetMode::SelfAoe => "self_aoe",
        TargetMode::Ground => "ground",
        TargetMode::Direction => "direction",
        TargetMode::Vector => "vector",
        TargetMode::Global => "global",
    }
}

fn hint_str(h: HintName) -> &'static str {
    match h {
        HintName::Damage => "damage",
        HintName::Defense => "defense",
        HintName::CrowdControl => "crowd_control",
        HintName::Utility => "utility",
        HintName::Heal => "heal",
        HintName::Economic => "economic",
        HintName::Buff => "buff",
    }
}

fn cost_str(c: &CostSpec) -> String {
    let resource = match c.resource {
        CostResource::Mana => "mana",
        CostResource::Stamina => "stamina",
        CostResource::Hp => "hp",
        CostResource::Gold => "gold",
    };
    match c.amount {
        CostAmount::Flat(n) => format!("{} {}", fmt_f32(n), resource),
        CostAmount::PercentOfMax(p) => format!("{}% {}", fmt_f32(p), resource),
    }
}

fn duration_str(d: Duration) -> String {
    // The parser accepts both `Ns`/`Nms`. Emit the most human form:
    // whole seconds if evenly divisible, otherwise ms.
    if d.millis % 1000 == 0 {
        format!("{}s", d.millis / 1000)
    } else {
        format!("{}ms", d.millis)
    }
}

fn effect_arg_str(a: &EffectArg) -> String {
    match a {
        EffectArg::Number(n) => fmt_f32(*n),
        EffectArg::Duration(d) => duration_str(*d),
        EffectArg::Percent(p) => format!("{}%", fmt_f32(*p)),
        EffectArg::String(s) => format!("\"{s}\""),
        EffectArg::Ident(s) => s.clone(),
    }
}

fn fmt_f32(v: f32) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}
