//! Winnow-based parser for the ability DSL.
//!
//! Parses `.ability` files into AST nodes defined in `ast.rs`.

use winnow::prelude::*;
use winnow::combinator::{alt, fail, opt, preceded};
use winnow::token::{any, take_while};
use winnow::ascii::{digit1, multispace0, multispace1};

use super::ast::*;

// Re-export sub-module parsers for use from this module
pub(super) use super::parse_effects::{effect_node, effect_list};
pub(super) use super::parse_delivery::delivery_block;

// ---------------------------------------------------------------------------
// Whitespace & comments
// ---------------------------------------------------------------------------

/// Skip whitespace and comments (// and #).
pub(super) fn ws(input: &mut &str) -> ModalResult<()> {
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
pub(super) fn hws(input: &mut &str) -> ModalResult<()> {
    let _: &str = take_while(0.., |c: char| c == ' ' || c == '\t').parse_next(input)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*
pub(super) fn ident(input: &mut &str) -> ModalResult<String> {
    let first: char = any.verify(|c: &char| c.is_ascii_alphabetic() || *c == '_').parse_next(input)?;
    let rest: &str = take_while(0.., |c: char| c.is_ascii_alphanumeric() || c == '_').parse_next(input)?;
    let mut s = String::with_capacity(1 + rest.len());
    s.push(first);
    s.push_str(rest);
    Ok(s)
}

/// Parse a number (integer or float, possibly negative).
pub(super) fn number(input: &mut &str) -> ModalResult<f64> {
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
pub(super) fn duration(input: &mut &str) -> ModalResult<u32> {
    let n: f64 = number.parse_next(input)?;
    if input.starts_with("ms") {
        let _: &str = "ms".parse_next(input)?;
        Ok(n as u32)
    } else if input.starts_with('s') && (input.len() == 1 || !input[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        let _: char = 's'.parse_next(input)?;
        Ok((n * 1000.0) as u32)
    } else if input.starts_with('t') && (input.len() == 1 || !input[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        // Campaign tick duration: Nt — return raw tick count
        let _: char = 't'.parse_next(input)?;
        Ok(n as u32)
    } else {
        Ok(n as u32)
    }
}

/// Parse a duration that might be tick-based (`500t`).
/// Returns (ms_duration, tick_duration) — exactly one will be Some.
pub(super) fn duration_or_ticks(input: &mut &str) -> ModalResult<(Option<u32>, Option<u32>)> {
    let n: f64 = number.parse_next(input)?;
    if input.starts_with("ms") {
        let _: &str = "ms".parse_next(input)?;
        Ok((Some(n as u32), None))
    } else if input.starts_with('s') && (input.len() == 1 || !input[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        let _: char = 's'.parse_next(input)?;
        Ok((Some((n * 1000.0) as u32), None))
    } else if input.starts_with('t') && (input.len() == 1 || !input[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        let _: char = 't'.parse_next(input)?;
        Ok((None, Some(n as u32)))
    } else {
        Ok((Some(n as u32), None))
    }
}

/// Parse a string literal: `"..."`.
pub(super) fn string_lit(input: &mut &str) -> ModalResult<String> {
    let _: char = '"'.parse_next(input)?;
    let content: &str = take_while(0.., |c: char| c != '"').parse_next(input)?;
    let _: char = '"'.parse_next(input)?;
    Ok(content.to_string())
}

/// Parse a percentage like `50%` → returns the number (50.0).
pub(super) fn percent(input: &mut &str) -> ModalResult<f64> {
    let n: f64 = number.parse_next(input)?;
    let _: char = '%'.parse_next(input)?;
    Ok(n)
}

/// Parse a duration or number argument value.
/// Also handles `X/tick` and `X/Ns` for periodic (DoT/HoT) amounts.
pub(super) fn duration_or_number(input: &mut &str) -> ModalResult<Arg> {
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
    } else if input.starts_with('t') && (input.len() == 1 || !input[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        // Campaign tick duration: Nt
        let _: char = 't'.parse_next(input)?;
        Ok(Arg::TickDuration(n as u32))
    } else {
        Ok(Arg::Number(n))
    }
}

// ---------------------------------------------------------------------------
// Properties (key: value pairs)
// ---------------------------------------------------------------------------

pub(super) fn try_parse_property(input: &mut &str) -> ModalResult<Property> {
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
