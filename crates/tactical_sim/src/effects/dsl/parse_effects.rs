//! Effect type parsing (damage, heal, status, conditions, areas, scaling, etc.)

use winnow::prelude::*;
use winnow::combinator::{alt, delimited, fail, opt, preceded, separated, terminated};
use winnow::ascii::{multispace1};

use super::ast::*;
use super::parser::{ws, hws, ident, number, string_lit, duration_or_number, try_parse_property, duration_or_ticks};

// ---------------------------------------------------------------------------
// Tags: [FIRE: 60, MAGIC: 40]
// ---------------------------------------------------------------------------

fn tag_pair(input: &mut &str) -> ModalResult<(String, f64)> {
    ws.parse_next(input)?;
    let name: String = ident.parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = ':'.parse_next(input)?;
    ws.parse_next(input)?;
    let val: f64 = number.parse_next(input)?;
    Ok((name, val))
}

fn tag_list(input: &mut &str) -> ModalResult<Vec<(String, f64)>> {
    let _: char = '['.parse_next(input)?;
    let tags: Vec<(String, f64)> = separated(0.., tag_pair, (ws, ',')).parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = ']'.parse_next(input)?;
    Ok(tags)
}

// ---------------------------------------------------------------------------
// Area: in circle(3.0)
// ---------------------------------------------------------------------------

fn area_modifier(input: &mut &str) -> ModalResult<AreaNode> {
    let _: &str = "in".parse_next(input)?;
    multispace1.parse_next(input)?;
    let shape: String = ident.parse_next(input)?;
    // "in self" means apply to caster (no parenthesized args)
    if shape == "self" {
        return Ok(AreaNode { shape, args: vec![] });
    }
    let _: char = '('.parse_next(input)?;
    let args: Vec<f64> = separated(0.., (ws, number).map(|(_, n)| n), (ws, ',')).parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = ')'.parse_next(input)?;
    Ok(AreaNode { shape, args })
}

// ---------------------------------------------------------------------------
// Conditions: when target_hp_below(30%)
// ---------------------------------------------------------------------------

fn cond_arg(input: &mut &str) -> ModalResult<CondArg> {
    ws.parse_next(input)?;
    alt((
        string_lit.map(CondArg::StringLit),
        super::parser::percent.map(CondArg::Percent),
        number.map(CondArg::Number),
        ident.map(CondArg::Ident),
    )).parse_next(input)
}

fn condition_atom(input: &mut &str) -> ModalResult<ConditionNode> {
    ws.parse_next(input)?;
    let name: String = ident.parse_next(input)?;

    match name.as_str() {
        "and" | "or" => {
            let _: char = '('.parse_next(input)?;
            let subs: Vec<ConditionNode> = separated(1.., condition_atom, (ws, ',')).parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = ')'.parse_next(input)?;
            if name == "and" {
                Ok(ConditionNode::And(subs))
            } else {
                Ok(ConditionNode::Or(subs))
            }
        }
        "not" => {
            let _: char = '('.parse_next(input)?;
            ws.parse_next(input)?;
            let inner: ConditionNode = condition_atom.parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = ')'.parse_next(input)?;
            Ok(ConditionNode::Not(Box::new(inner)))
        }
        _ => {
            let args: Option<Vec<CondArg>> = opt(delimited('(', separated(0.., cond_arg, (ws, ',')), ')')).parse_next(input)?;
            Ok(ConditionNode::Simple {
                name,
                args: args.unwrap_or_default(),
            })
        }
    }
}

fn condition(input: &mut &str) -> ModalResult<ConditionNode> {
    let _: &str = "when".parse_next(input)?;
    multispace1.parse_next(input)?;
    condition_atom.parse_next(input)
}

// ---------------------------------------------------------------------------
// Scaling: + 10% target_max_hp [consume] [cap 100]
// ---------------------------------------------------------------------------

fn scaling_term(input: &mut &str) -> ModalResult<ScalingNode> {
    ws.parse_next(input)?;
    let _: char = '+'.parse_next(input)?;
    ws.parse_next(input)?;
    let pct: f64 = number.parse_next(input)?;
    let _: char = '%'.parse_next(input)?;
    multispace1.parse_next(input)?;
    let stat: String = ident.parse_next(input)?;

    ws.parse_next(input)?;
    let consume: bool = opt("consume").parse_next(input)?.is_some();

    ws.parse_next(input)?;
    let cap: Option<i32> = opt(preceded(terminated("cap", multispace1), number.map(|n: f64| n as i32))).parse_next(input)?;

    Ok(ScalingNode { percent: pct, stat, consume, cap })
}

// ---------------------------------------------------------------------------
// Effect line
// ---------------------------------------------------------------------------

/// Parse a single effect line like `damage 55 in circle(3.0) [FIRE: 60] when cond`.
pub fn effect_node(input: &mut &str) -> ModalResult<EffectNode> {
    ws.parse_next(input)?;
    let effect_type: String = ident.parse_next(input)?;

    let mut args = Vec::new();
    let mut area = None;
    let mut tags = Vec::new();
    let mut cond = None;
    let mut else_effects = Vec::new();
    let mut stacking = None;
    let mut chance = None;
    let mut scaling = Vec::new();
    let mut dur = None;
    let mut dur_t = None;
    let mut children = Vec::new();
    let mut targeting_filter = None;
    let mut targeting_args = Vec::new();

    loop {
        hws.parse_next(input)?;

        if input.is_empty() || input.starts_with('}') || input.starts_with('\n') {
            break;
        }

        // Also break on comment start (effect is done, rest of line is comment)
        if input.starts_with("//") || input.starts_with('#') {
            break;
        }

        if input.starts_with("in ") && !input.starts_with("into") {
            area = Some(area_modifier.parse_next(input)?);
            continue;
        }

        if input.starts_with('[') {
            tags = tag_list.parse_next(input)?;
            continue;
        }

        if input.starts_with("when ") {
            cond = Some(condition.parse_next(input)?);
            ws.parse_next(input)?;
            if input.starts_with("else ") {
                let _: &str = "else".parse_next(input)?;
                multispace1.parse_next(input)?;
                let else_eff: EffectNode = effect_node.parse_next(input)?;
                else_effects.push(else_eff);
            }
            continue;
        }

        if input.starts_with("while_alive") && (input.len() == 11 || !input[11..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')) {
            let _: &str = "while_alive".parse_next(input)?;
            dur = Some(u32::MAX);
            continue;
        }

        if input.starts_with("for ") {
            let _: &str = "for".parse_next(input)?;
            multispace1.parse_next(input)?;
            // Check for `for while_alive` (emitter produces `for while_alive` via fmt_duration)
            if input.starts_with("while_alive") && (input.len() == 11 || !input[11..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')) {
                let _: &str = "while_alive".parse_next(input)?;
                dur = Some(u32::MAX);
            } else {
                let (ms, ticks) = duration_or_ticks.parse_next(input)?;
                dur = ms;
                dur_t = ticks;
            }
            continue;
        }

        if input.starts_with("stacking ") {
            let _: &str = "stacking".parse_next(input)?;
            multispace1.parse_next(input)?;
            stacking = Some(ident.parse_next(input)?);
            continue;
        }

        if input.starts_with("chance ") {
            let _: &str = "chance".parse_next(input)?;
            multispace1.parse_next(input)?;
            chance = Some(number.parse_next(input)?);
            continue;
        }

        if input.starts_with("targeting ") {
            let _: &str = "targeting".parse_next(input)?;
            multispace1.parse_next(input)?;
            let pred_name: String = ident.parse_next(input)?;
            targeting_filter = Some(pred_name);
            // Parse optional parenthesized args
            hws.parse_next(input)?;
            if input.starts_with('(') {
                let _: char = '('.parse_next(input)?;
                let targs: Vec<Arg> = separated(0.., |i: &mut &str| {
                    ws.parse_next(i)?;
                    alt((
                        string_lit.map(Arg::StringLit),
                        duration_or_number,
                        ident.map(Arg::Ident),
                    )).parse_next(i)
                }, (ws, ',')).parse_next(input)?;
                ws.parse_next(input)?;
                let _: char = ')'.parse_next(input)?;
                targeting_args = targs;
            }
            continue;
        }

        if input.starts_with('+') {
            scaling.push(scaling_term.parse_next(input)?);
            continue;
        }

        if input.starts_with("scales_with") && (input.len() == 11 || !input[11..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')) {
            let _: &str = "scales_with".parse_next(input)?;
            multispace1.parse_next(input)?;
            let stat_name: String = ident.parse_next(input)?;
            scaling.push(ScalingNode {
                percent: 100.0,
                stat: stat_name,
                consume: false,
                cap: None,
            });
            continue;
        }

        if input.starts_with('{') {
            let _: char = '{'.parse_next(input)?;
            children = effect_list.parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '}'.parse_next(input)?;
            continue;
        }

        // Try positional argument
        let checkpoint = *input;
        if let Ok(arg) = try_parse_arg(input) {
            args.push(arg);
            continue;
        }
        *input = checkpoint;

        break;
    }

    Ok(EffectNode {
        effect_type,
        args,
        area,
        tags,
        condition: cond,
        else_effects,
        stacking,
        chance,
        scaling,
        duration: dur,
        duration_ticks: dur_t,
        children,
        targeting_filter,
        targeting_args,
    })
}

/// Try to parse an argument.
fn try_parse_arg(input: &mut &str) -> ModalResult<Arg> {
    ws.parse_next(input)?;
    alt((
        string_lit.map(Arg::StringLit),
        duration_or_number,
        |i: &mut &str| {
            let checkpoint = *i;
            let id: String = ident.parse_next(i)?;
            match id.as_str() {
                "in" | "when" | "for" | "stacking" | "chance" | "else"
                | "deliver" | "ability" | "passive" | "morph" | "recast"
                | "on_hit" | "on_arrival" | "on_complete" | "targeting"
                | "while_alive" | "scales_with" => {
                    *i = checkpoint;
                    fail.parse_next(i)
                }
                _ => Ok(Arg::Ident(id)),
            }
        },
    )).parse_next(input)
}

/// Parse a list of effect lines.
pub fn effect_list(input: &mut &str) -> ModalResult<Vec<EffectNode>> {
    let mut effects = Vec::new();
    loop {
        ws.parse_next(input)?;
        if input.is_empty() || input.starts_with('}') {
            break;
        }
        if input.starts_with("deliver ")
            || input.starts_with("morph ")
            || input.starts_with("recast ")
            || input.starts_with("on_hit")
            || input.starts_with("on_arrival")
            || input.starts_with("on_complete")
        {
            break;
        }
        let checkpoint = *input;
        if try_parse_property(input).is_ok() {
            *input = checkpoint;
            break;
        }
        *input = checkpoint;

        let eff: EffectNode = effect_node.parse_next(input)?;
        effects.push(eff);
    }
    Ok(effects)
}
