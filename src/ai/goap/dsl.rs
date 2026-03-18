//! Parser for `.goap` files.

use super::action::{GoapAction, IntentTemplate};
use super::goal::{CompOp, Goal, InsistenceFn, Precondition};
use super::target::{parse_target, Target};
use super::world_state::{self, prop_index, PropValue};

/// Role hint for threat/formation systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleHint {
    MeleeEngage,
    RangedDamage,
    Support,
    Controller,
    Skirmisher,
}

/// A complete GOAP definition parsed from a `.goap` or `.behavior` file.
#[derive(Debug, Clone)]
pub struct GoapDef {
    pub name: String,
    pub role_hint: Option<RoleHint>,
    pub goals: Vec<Goal>,
    pub actions: Vec<GoapAction>,
}

/// Load a `.goap` file from disk.
pub fn load_goap_file(path: &str) -> Result<GoapDef, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    parse_goap(&content)
}

fn flip_op(op: CompOp) -> CompOp {
    match op {
        CompOp::Lt => CompOp::Gt,
        CompOp::Gt => CompOp::Lt,
        CompOp::Lte => CompOp::Gte,
        CompOp::Gte => CompOp::Lte,
        CompOp::Eq => CompOp::Eq,
        CompOp::Neq => CompOp::Neq,
    }
}

// ===========================================================================
// Native GOAP parser
// ===========================================================================

/// Parse a `.goap` file from text.
pub fn parse_goap(input: &str) -> Result<GoapDef, String> {
    let lines: Vec<&str> = input.lines().collect();
    let mut idx = 0;

    skip_blank(&lines, &mut idx);

    let header_line = lines.get(idx).ok_or("Expected 'goap' block")?;
    let tokens = tokenize(header_line);
    if tokens.first().map(|s| s.as_str()) != Some("goap") {
        return Err(format!("Expected 'goap', got: {:?}", tokens.first()));
    }
    let name = unquote(tokens.get(1).ok_or("Expected goap name")?);
    expect_token(&tokens, 2, "{")?;
    idx += 1;

    let mut role_hint = None;
    let mut goals = Vec::new();
    let mut actions = Vec::new();

    while idx < lines.len() {
        skip_blank(&lines, &mut idx);
        if idx >= lines.len() {
            break;
        }

        let line = lines[idx].trim();
        if line == "}" {
            break;
        }

        let tokens = tokenize(line);
        if tokens.is_empty() {
            idx += 1;
            continue;
        }

        match tokens[0].as_str() {
            "role_hint" => {
                role_hint = Some(parse_role_hint(
                    tokens.get(1).ok_or("Expected role hint value")?,
                )?);
                idx += 1;
            }
            "goal" => {
                let (goal, new_idx) = parse_goal(&lines, idx)?;
                goals.push(goal);
                idx = new_idx;
            }
            "action" => {
                let (action, new_idx) = parse_action_block(&lines, idx)?;
                actions.push(action);
                idx = new_idx;
            }
            other => {
                return Err(format!("Unexpected token in goap block: '{}'", other));
            }
        }
    }

    Ok(GoapDef {
        name,
        role_hint,
        goals,
        actions,
    })
}

fn parse_goal(lines: &[&str], start: usize) -> Result<(Goal, usize), String> {
    let header = tokenize(lines[start]);
    let name = unquote(header.get(1).ok_or("Expected goal name")?);
    expect_token(&header, 2, "{")?;

    let mut idx = start + 1;
    let mut desired = Vec::new();
    let mut insistence = InsistenceFn::Fixed(0.5);

    while idx < lines.len() {
        let line = lines[idx].trim();
        if line.is_empty() || line.starts_with('#') {
            idx += 1;
            continue;
        }
        if line == "}" {
            idx += 1;
            break;
        }

        let tokens = tokenize(line);
        match tokens[0].as_str() {
            "desire" => {
                let (prop, pre) = parse_precondition(&tokens[1..])?;
                desired.push((prop, pre));
            }
            "insistence" => {
                insistence = parse_insistence(&tokens[1..])?;
            }
            other => return Err(format!("Unknown goal keyword: '{}'", other)),
        }
        idx += 1;
    }

    Ok((Goal { name, desired, insistence }, idx))
}

fn parse_action_block(lines: &[&str], start: usize) -> Result<(GoapAction, usize), String> {
    let header = tokenize(lines[start]);
    let name = unquote(header.get(1).ok_or("Expected action name")?);
    expect_token(&header, 2, "{")?;

    let mut idx = start + 1;
    let mut cost = 1.0_f32;
    let mut preconditions = Vec::new();
    let mut effects = Vec::new();
    let mut intent = IntentTemplate::Hold;
    let mut duration = 1u32;

    while idx < lines.len() {
        let line = lines[idx].trim();
        if line.is_empty() || line.starts_with('#') {
            idx += 1;
            continue;
        }
        if line == "}" {
            idx += 1;
            break;
        }

        let tokens = tokenize(line);
        match tokens[0].as_str() {
            "cost" => {
                cost = tokens[1].parse().map_err(|_| "Invalid cost value")?;
            }
            "precondition" => {
                let (prop, pre) = parse_precondition(&tokens[1..])?;
                preconditions.push((prop, pre));
            }
            "effect" => {
                let (prop, val) = parse_effect(&tokens[1..])?;
                effects.push((prop, val));
            }
            "intent" => {
                intent = parse_intent(&tokens[1..])?;
            }
            "duration" => {
                duration = tokens[1].parse().map_err(|_| "Invalid duration")?;
            }
            other => return Err(format!("Unknown action keyword: '{}'", other)),
        }
        idx += 1;
    }

    Ok((
        GoapAction {
            name,
            cost,
            preconditions,
            effects,
            intent,
            duration_ticks: duration,
        },
        idx,
    ))
}

fn parse_precondition(tokens: &[String]) -> Result<(usize, Precondition), String> {
    if tokens.len() < 3 {
        return Err(format!("Precondition needs 3 tokens, got: {:?}", tokens));
    }
    let prop = prop_index(&tokens[0])
        .ok_or_else(|| format!("Unknown property: '{}'", tokens[0]))?;
    let op = parse_comp_op(&tokens[1])?;
    let value = parse_prop_value(&tokens[2])?;
    Ok((prop, Precondition { op, value }))
}

fn parse_effect(tokens: &[String]) -> Result<(usize, PropValue), String> {
    if tokens.len() < 3 {
        return Err(format!("Effect needs 3 tokens, got: {:?}", tokens));
    }
    let prop = prop_index(&tokens[0])
        .ok_or_else(|| format!("Unknown property: '{}'", tokens[0]))?;
    let value = parse_prop_value(&tokens[2])?;
    Ok((prop, value))
}

fn parse_insistence(tokens: &[String]) -> Result<InsistenceFn, String> {
    match tokens[0].as_str() {
        "fixed" => {
            let val: f32 = tokens[1]
                .parse()
                .map_err(|_| "Invalid fixed insistence")?;
            Ok(InsistenceFn::Fixed(val))
        }
        "linear" => {
            let prop = prop_index(&tokens[1])
                .ok_or_else(|| format!("Unknown property: '{}'", tokens[1]))?;
            let mut scale = 1.0_f32;
            let mut offset = 0.0_f32;
            let mut i = 2;
            while i < tokens.len() {
                match tokens[i].as_str() {
                    "scale" => {
                        scale = tokens[i + 1].parse().map_err(|_| "Invalid scale")?;
                        i += 2;
                    }
                    "offset" => {
                        offset = tokens[i + 1].parse().map_err(|_| "Invalid offset")?;
                        i += 2;
                    }
                    _ => return Err(format!("Unknown insistence modifier: '{}'", tokens[i])),
                }
            }
            Ok(InsistenceFn::Linear {
                prop,
                scale,
                offset,
            })
        }
        "threshold" => {
            let prop = prop_index(&tokens[1])
                .ok_or_else(|| format!("Unknown property: '{}'", tokens[1]))?;
            let op = parse_comp_op(&tokens[2])?;
            let threshold: f32 = tokens[3].parse().map_err(|_| "Invalid threshold")?;
            if tokens.get(4).map(|s| s.as_str()) != Some("value") {
                return Err("Expected 'value' keyword in threshold insistence".to_string());
            }
            let value: f32 = tokens[5].parse().map_err(|_| "Invalid threshold value")?;
            Ok(InsistenceFn::Threshold {
                prop,
                op,
                threshold,
                value,
            })
        }
        other => Err(format!("Unknown insistence type: '{}'", other)),
    }
}

fn parse_intent(tokens: &[String]) -> Result<IntentTemplate, String> {
    match tokens[0].as_str() {
        "attack" => {
            let target = parse_target(&tokens[1..])?;
            Ok(IntentTemplate::AttackTarget(target))
        }
        "chase" => {
            let target = parse_target(&tokens[1..])?;
            Ok(IntentTemplate::ChaseTarget(target))
        }
        "flee" => {
            let target = parse_target(&tokens[1..])?;
            Ok(IntentTemplate::FleeTarget(target))
        }
        "maintain_distance" => {
            let target = parse_target(&tokens[1..tokens.len() - 1])?;
            let range: f32 = tokens
                .last()
                .ok_or("Missing range")?
                .parse()
                .map_err(|_| "Invalid range")?;
            Ok(IntentTemplate::MaintainDistance(target, range))
        }
        "cast_if_ready" => {
            let slot = parse_ability_slot(&tokens[1])?;
            let target_start = if tokens.get(2).map(|s| s.as_str()) == Some("on") {
                3
            } else {
                2
            };
            let target = parse_target(&tokens[target_start..])?;
            Ok(IntentTemplate::CastIfReady(slot, target))
        }
        "hold" => Ok(IntentTemplate::Hold),
        other => Err(format!("Unknown intent: '{}'", other)),
    }
}

fn parse_ability_slot(s: &str) -> Result<usize, String> {
    if let Some(rest) = s.strip_prefix("ability") {
        rest.parse()
            .map_err(|_| format!("Invalid ability slot: '{}'", s))
    } else {
        Err(format!("Expected 'abilityN', got '{}'", s))
    }
}

fn parse_comp_op(s: &str) -> Result<CompOp, String> {
    match s {
        "==" => Ok(CompOp::Eq),
        "!=" => Ok(CompOp::Neq),
        "<" => Ok(CompOp::Lt),
        ">" => Ok(CompOp::Gt),
        "<=" => Ok(CompOp::Lte),
        ">=" => Ok(CompOp::Gte),
        _ => Err(format!("Unknown comparison operator: '{}'", s)),
    }
}

fn parse_prop_value(s: &str) -> Result<PropValue, String> {
    match s {
        "true" => Ok(PropValue::Bool(true)),
        "false" => Ok(PropValue::Bool(false)),
        _ => {
            let f: f32 = s
                .parse()
                .map_err(|_| format!("Invalid value: '{}'", s))?;
            Ok(PropValue::Float(f))
        }
    }
}

fn parse_role_hint(s: &str) -> Result<RoleHint, String> {
    match s {
        "melee_engage" => Ok(RoleHint::MeleeEngage),
        "ranged_damage" => Ok(RoleHint::RangedDamage),
        "support" => Ok(RoleHint::Support),
        "controller" => Ok(RoleHint::Controller),
        "skirmisher" => Ok(RoleHint::Skirmisher),
        _ => Err(format!("Unknown role hint: '{}'", s)),
    }
}

// ===========================================================================
// Shared utilities
// ===========================================================================

fn tokenize(line: &str) -> Vec<String> {
    let line = line.trim();
    let line = if let Some(idx) = line.find('#') {
        &line[..idx]
    } else {
        line
    };
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();
    let mut current = String::new();

    while let Some(&ch) = chars.peek() {
        if ch == '"' {
            chars.next();
            let mut quoted = String::new();
            while let Some(&c) = chars.peek() {
                if c == '"' {
                    chars.next();
                    break;
                }
                quoted.push(c);
                chars.next();
            }
            tokens.push(quoted);
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            chars.next();
        } else {
            current.push(ch);
            chars.next();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn unquote(s: &str) -> String {
    s.trim_matches('"').to_string()
}

fn expect_token(tokens: &[String], idx: usize, expected: &str) -> Result<(), String> {
    match tokens.get(idx) {
        Some(t) if t == expected => Ok(()),
        Some(t) => Err(format!("Expected '{}', got '{}'", expected, t)),
        None => Err(format!("Expected '{}', got end of line", expected)),
    }
}

fn skip_blank(lines: &[&str], idx: &mut usize) {
    while *idx < lines.len() {
        let line = lines[*idx].trim();
        if line.is_empty() || line.starts_with('#') {
            *idx += 1;
        } else {
            break;
        }
    }
}
