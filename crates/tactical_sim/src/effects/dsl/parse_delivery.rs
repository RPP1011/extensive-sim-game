//! Delivery method parsing (projectile, chain, zone, tether, trap, channel).

use winnow::prelude::*;
use winnow::combinator::{alt, separated};
use winnow::ascii::multispace1;

use super::ast::*;
use super::parser::{effect_node, ws, ident, string_lit, duration_or_number};

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

pub(super) fn delivery_hooks(input: &mut &str) -> ModalResult<(Vec<EffectNode>, Vec<EffectNode>, Vec<EffectNode>)> {
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
            on_hit = super::parser::effect_list.parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '}'.parse_next(input)?;
        } else if input.starts_with("on_arrival") {
            let _: &str = "on_arrival".parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '{'.parse_next(input)?;
            on_arrival = super::parser::effect_list.parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '}'.parse_next(input)?;
        } else if input.starts_with("on_complete") {
            let _: &str = "on_complete".parse_next(input)?;
            ws.parse_next(input)?;
            let _: char = '{'.parse_next(input)?;
            on_complete = super::parser::effect_list.parse_next(input)?;
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

pub(super) fn delivery_block(input: &mut &str) -> ModalResult<DeliveryNode> {
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
