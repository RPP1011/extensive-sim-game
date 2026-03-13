//! Winnow-based parser for the ability DSL.
//!
//! Parses `.ability` files into AST nodes defined in `ast.rs`.

use winnow::prelude::*;
use winnow::combinator::{alt, delimited, fail, opt, preceded, separated, terminated};
use winnow::token::{any, take_while};
use winnow::ascii::{digit1, multispace0, multispace1};

use super::ast::*;

// ---------------------------------------------------------------------------
// Whitespace & comments
// ---------------------------------------------------------------------------

/// Skip whitespace and comments (// and #).
fn ws(input: &mut &str) -> ModalResult<()> {
    loop {
        let _: &str = multispace0.parse_next(input)?;
        if input.starts_with("//") || input.starts_with('#') {
            let _: &str = take_while(0.., |c: char| c != '\n').parse_next(input)?;
        } else {
            break;
        }
    }
    Ok(())
}

/// Skip only horizontal whitespace (spaces and tabs), not newlines.
fn hws(input: &mut &str) -> ModalResult<()> {
    let _: &str = take_while(0.., |c: char| c == ' ' || c == '\t').parse_next(input)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*
fn ident(input: &mut &str) -> ModalResult<String> {
    let first: char = any.verify(|c: &char| c.is_ascii_alphabetic() || *c == '_').parse_next(input)?;
    let rest: &str = take_while(0.., |c: char| c.is_ascii_alphanumeric() || c == '_').parse_next(input)?;
    let mut s = String::with_capacity(1 + rest.len());
    s.push(first);
    s.push_str(rest);
    Ok(s)
}

/// Parse a number (integer or float, possibly negative).
fn number(input: &mut &str) -> ModalResult<f64> {
    let neg: Option<char> = opt('-').parse_next(input)?;
    let int_part: &str = digit1.parse_next(input)?;
    let frac_part: Option<&str> = opt(preceded('.', digit1)).parse_next(input)?;

    let mut s = String::new();
    if neg.is_some() {
        s.push('-');
    }
    s.push_str(int_part);
    if let Some(frac) = frac_part {
        s.push('.');
        s.push_str(frac);
    }
    match s.parse::<f64>() {
        Ok(n) => Ok(n),
        Err(_) => fail.parse_next(input),
    }
}

/// Parse a duration: `5s`, `300ms`, or bare number (defaults to ms).
fn duration(input: &mut &str) -> ModalResult<u32> {
    let n: f64 = number.parse_next(input)?;
    if input.starts_with("ms") {
        let _: &str = "ms".parse_next(input)?;
        Ok(n as u32)
    } else if input.starts_with('s') && (input.len() == 1 || !input[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        let _: char = 's'.parse_next(input)?;
        Ok((n * 1000.0) as u32)
    } else {
        Ok(n as u32)
    }
}

/// Parse a string literal: `"..."`.
fn string_lit(input: &mut &str) -> ModalResult<String> {
    let _: char = '"'.parse_next(input)?;
    let content: &str = take_while(0.., |c: char| c != '"').parse_next(input)?;
    let _: char = '"'.parse_next(input)?;
    Ok(content.to_string())
}

/// Parse a percentage like `50%` → returns the number (50.0).
fn percent(input: &mut &str) -> ModalResult<f64> {
    let n: f64 = number.parse_next(input)?;
    let _: char = '%'.parse_next(input)?;
    Ok(n)
}

/// Parse a duration or number argument value.
/// Also handles `X/tick` and `X/Ns` for periodic (DoT/HoT) amounts.
fn duration_or_number(input: &mut &str) -> ModalResult<Arg> {
    let n: f64 = number.parse_next(input)?;
    // Check for per-tick syntax: X/tick or X/500ms or X/1s
    if input.starts_with('/') {
        let _: char = '/'.parse_next(input)?;
        if input.starts_with("tick") {
            let _: &str = "tick".parse_next(input)?;
            return Ok(Arg::PerTick { amount: n as i32, interval_ms: 1000 });
        }
        // Parse interval duration: /500ms or /1s
        let interval = duration.parse_next(input)?;
        return Ok(Arg::PerTick { amount: n as i32, interval_ms: interval });
    }
    // Check for duration suffix or percent
    if input.starts_with("ms") {
        let _: &str = "ms".parse_next(input)?;
        Ok(Arg::Duration(n as u32))
    } else if input.starts_with('%') {
        let _: char = '%'.parse_next(input)?;
        Ok(Arg::Number(n))
    } else if input.starts_with('s') && (input.len() == 1 || !input[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        let _: char = 's'.parse_next(input)?;
        Ok(Arg::Duration((n * 1000.0) as u32))
    } else {
        Ok(Arg::Number(n))
    }
}

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
        percent.map(CondArg::Percent),
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
    let mut children = Vec::new();

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

        if input.starts_with("for ") {
            let _: &str = "for".parse_next(input)?;
            multispace1.parse_next(input)?;
            dur = Some(duration.parse_next(input)?);
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

        if input.starts_with('+') {
            scaling.push(scaling_term.parse_next(input)?);
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
        children,
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
                | "on_hit" | "on_arrival" | "on_complete" => {
                    *i = checkpoint;
                    fail.parse_next(i)
                }
                _ => Ok(Arg::Ident(id)),
            }
        },
    )).parse_next(input)
}

/// Parse a list of effect lines.
fn effect_list(input: &mut &str) -> ModalResult<Vec<EffectNode>> {
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

// ---------------------------------------------------------------------------
// Delivery block
// ---------------------------------------------------------------------------

fn delivery_param(input: &mut &str) -> ModalResult<(String, Arg)> {
    ws.parse_next(input)?;
    let key: String = ident.parse_next(input)?;
    ws.parse_next(input)?;
    if input.starts_with(':') {
        let _: char = ':'.parse_next(input)?;
        ws.parse_next(input)?;
        let val: Arg = alt((
            string_lit.map(Arg::StringLit),
            duration_or_number,
            ident.map(Arg::Ident),
        )).parse_next(input)?;
        Ok((key, val))
    } else {
        Ok((key, Arg::Ident("true".into())))
    }
}

fn delivery_params(input: &mut &str) -> ModalResult<Vec<(String, Arg)>> {
    let _: char = '{'.parse_next(input)?;
    let params: Vec<(String, Arg)> = separated(0.., delivery_param, (ws, ',')).parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = '}'.parse_next(input)?;
    Ok(params)
}

fn delivery_hooks(input: &mut &str) -> ModalResult<(Vec<EffectNode>, Vec<EffectNode>, Vec<EffectNode>)> {
    let mut on_hit = Vec::new();
    let mut on_arrival = Vec::new();
    let mut on_complete = Vec::new();

    let _: char = '{'.parse_next(input)?;
    loop {
        ws.parse_next(input)?;
        if input.starts_with('}') {
            break;
        }
        if input.starts_with("on_hit") {
            let _: &str = "on_hit".parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '{'.parse_next(input)?;
            on_hit = effect_list.parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '}'.parse_next(input)?;
        } else if input.starts_with("on_arrival") {
            let _: &str = "on_arrival".parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '{'.parse_next(input)?;
            on_arrival = effect_list.parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '}'.parse_next(input)?;
        } else if input.starts_with("on_complete") {
            let _: &str = "on_complete".parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '{'.parse_next(input)?;
            on_complete = effect_list.parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '}'.parse_next(input)?;
        } else {
            let eff: EffectNode = effect_node.parse_next(input)?;
            on_hit.push(eff);
        }
    }
    let _: char = '}'.parse_next(input)?;

    Ok((on_hit, on_arrival, on_complete))
}

fn delivery_block(input: &mut &str) -> ModalResult<DeliveryNode> {
    let _: &str = "deliver".parse_next(input)?;
    multispace1.parse_next(input)?;
    let method: String = ident.parse_next(input)?;
    ws.parse_next(input)?;

    let params: Vec<(String, Arg)> = delivery_params.parse_next(input)?;
    ws.parse_next(input)?;

    let (on_hit, on_arrival, on_complete) = if input.starts_with('{') {
        delivery_hooks.parse_next(input)?
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };

    Ok(DeliveryNode {
        method,
        params,
        on_hit,
        on_arrival,
        on_complete,
    })
}

// ---------------------------------------------------------------------------
// Properties (key: value pairs)
// ---------------------------------------------------------------------------

fn try_parse_property(input: &mut &str) -> ModalResult<Property> {
    ws.parse_next(input)?;
    let key: String = ident.parse_next(input)?;

    ws.parse_next(input)?;

    if input.starts_with(':') {
        let _: char = ':'.parse_next(input)?;
        ws.parse_next(input)?;
        let value: PropValue = prop_value.parse_next(input)?;
        Ok(Property { key, value })
    } else {
        match key.as_str() {
            "unstoppable" | "toggle" => Ok(Property {
                key,
                value: PropValue::Bool(true),
            }),
            _ => fail.parse_next(input),
        }
    }
}

fn prop_value(input: &mut &str) -> ModalResult<PropValue> {
    alt((
        string_lit.map(PropValue::StringLit),
        |i: &mut &str| {
            let n: f64 = number.parse_next(i)?;
            if i.starts_with("ms") {
                let _: &str = "ms".parse_next(i)?;
                Ok(PropValue::Duration(n as u32))
            } else if i.starts_with('s') && (i.len() == 1 || !i[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
                let _: char = 's'.parse_next(i)?;
                Ok(PropValue::Duration((n * 1000.0) as u32))
            } else {
                Ok(PropValue::Number(n))
            }
        },
        // Identifier, possibly with parenthesized args: on_hp_below(50%)
        |i: &mut &str| {
            let name: String = ident.parse_next(i)?;
            if i.starts_with('(') {
                let _: char = '('.parse_next(i)?;
                let inner: &str = take_while(0.., |c: char| c != ')').parse_next(i)?;
                let _: char = ')'.parse_next(i)?;
                Ok(PropValue::Ident(format!("{name}({inner})")))
            } else {
                Ok(PropValue::Ident(name))
            }
        },
    )).parse_next(input)
}

fn ability_body(input: &mut &str) -> ModalResult<(Vec<Property>, Vec<EffectNode>, Option<DeliveryNode>, Option<Box<MorphNode>>, Vec<RecastNode>)> {
    let mut props = Vec::new();
    let mut effects = Vec::new();
    let mut delivery = None;
    let mut morph = None;
    let mut recasts = Vec::new();

    loop {
        ws.parse_next(input)?;
        if input.is_empty() || input.starts_with('}') {
            break;
        }

        if input.starts_with("deliver ") {
            delivery = Some(delivery_block.parse_next(input)?);
            continue;
        }

        if input.starts_with("morph ") {
            morph = Some(Box::new(morph_block.parse_next(input)?));
            continue;
        }

        if input.starts_with("recast ") {
            recasts.push(recast_block.parse_next(input)?);
            continue;
        }

        let checkpoint = *input;
        if let Ok(prop) = try_parse_property(input) {
            props.push(prop);
            loop {
                ws.parse_next(input)?;
                if input.starts_with(',') {
                    let _: char = ','.parse_next(input)?;
                    ws.parse_next(input)?;
                    if let Ok(prop) = try_parse_property(input) {
                        props.push(prop);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            continue;
        }
        *input = checkpoint;

        let eff: EffectNode = effect_node.parse_next(input)?;
        effects.push(eff);
    }

    Ok((props, effects, delivery, morph, recasts))
}

// ---------------------------------------------------------------------------
// Morph and Recast blocks
// ---------------------------------------------------------------------------

fn morph_block(input: &mut &str) -> ModalResult<MorphNode> {
    let _: &str = "morph".parse_next(input)?;
    multispace1.parse_next(input)?;
    let _: &str = "into".parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = '{'.parse_next(input)?;

    let (props, effects, delivery, inner_morph, recasts) = ability_body.parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = '}'.parse_next(input)?;

    ws.parse_next(input)?;
    let dur: u32 = if input.starts_with("for ") {
        let _: &str = "for".parse_next(input)?;
        multispace1.parse_next(input)?;
        duration.parse_next(input)?
    } else {
        0
    };

    Ok(MorphNode {
        inner: AbilityNode {
            name: String::new(),
            props,
            effects,
            delivery,
            morph: inner_morph,
            recasts,
        },
        duration_ms: dur,
    })
}

fn recast_block(input: &mut &str) -> ModalResult<RecastNode> {
    let _: &str = "recast".parse_next(input)?;
    multispace1.parse_next(input)?;
    let idx: f64 = number.parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = '{'.parse_next(input)?;
    let effects: Vec<EffectNode> = effect_list.parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = '}'.parse_next(input)?;
    Ok(RecastNode { index: idx as u32, effects })
}

// ---------------------------------------------------------------------------
// Top-level: ability and passive blocks
// ---------------------------------------------------------------------------

fn ability_block(input: &mut &str) -> ModalResult<AbilityNode> {
    let _: &str = "ability".parse_next(input)?;
    multispace1.parse_next(input)?;
    let name: String = ident.parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = '{'.parse_next(input)?;

    let (props, effects, delivery, morph, recasts) = ability_body.parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = '}'.parse_next(input)?;

    Ok(AbilityNode {
        name,
        props,
        effects,
        delivery,
        morph,
        recasts,
    })
}

fn passive_block(input: &mut &str) -> ModalResult<PassiveNode> {
    let _: &str = "passive".parse_next(input)?;
    multispace1.parse_next(input)?;
    let name: String = ident.parse_next(input)?;
    ws.parse_next(input)?;
    let _: char = '{'.parse_next(input)?;

    let mut props = Vec::new();
    let mut effects = Vec::new();

    loop {
        ws.parse_next(input)?;
        if input.is_empty() || input.starts_with('}') {
            break;
        }

        let checkpoint = *input;
        if let Ok(prop) = try_parse_property(input) {
            props.push(prop);
            loop {
                ws.parse_next(input)?;
                if input.starts_with(',') {
                    let _: char = ','.parse_next(input)?;
                    ws.parse_next(input)?;
                    if let Ok(prop) = try_parse_property(input) {
                        props.push(prop);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            continue;
        }
        *input = checkpoint;

        let eff: EffectNode = effect_node.parse_next(input)?;
        effects.push(eff);
    }

    ws.parse_next(input)?;
    let _: char = '}'.parse_next(input)?;

    Ok(PassiveNode {
        name,
        props,
        effects,
    })
}

fn top_level_item(input: &mut &str) -> ModalResult<TopLevel> {
    ws.parse_next(input)?;
    alt((
        ability_block.map(TopLevel::Ability),
        passive_block.map(TopLevel::Passive),
    )).parse_next(input)
}

/// Parse an entire `.ability` file into an `AbilityFile` AST.
pub fn parse_ability_file(input: &mut &str) -> ModalResult<AbilityFile> {
    let mut items = Vec::new();
    loop {
        ws.parse_next(input)?;
        if input.is_empty() {
            break;
        }
        let item: TopLevel = top_level_item.parse_next(input)?;
        items.push(item);
    }
    Ok(AbilityFile { items })
}
