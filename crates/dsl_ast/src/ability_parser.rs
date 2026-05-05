//! Recursive-descent parser for `.ability` files.
//!
//! Wave 1.0 (per `docs/spec/ability_dsl_unified.md` §3.2 / §4) parsed
//! `ability <Name> { headers... effects... }` blocks only with five
//! header keys (`target`, `range`, `cooldown`, `cast`, `hint`).
//!
//! Wave 1.1 extends the surface (spec §4.2 + §5):
//! * Four new ability-block headers — `cost`, `charges`, `recharge`,
//!   `toggle` (the latter is a marker with no value).
//! * `passive <Name> { headers... effects... }` top-level blocks with
//!   their own header set (`trigger`, `cooldown`, `range`, `hint` per
//!   spec §5).
//!
//! Effect statements are parsed as `verb arg* modifier*`: the verb name
//! is preserved verbatim (validation against the verb catalog lives in
//! lowering, Wave 1.6) and the first run of simple positional arguments
//! (numbers / durations / percents / strings / idents) is collected.
//!
//! Wave 1.5 lifts the nine modifier slots in spec §6.1 into typed
//! `EffectStmt` fields (`area`, `tags`, `duration`, `condition`,
//! `chance`, `stacking`, `scalings`, `lifetime`, `nested`). Modifiers
//! may appear in any order after the positional args; each maps to a
//! distinct slot; duplicates on a single-value slot are a parse error.
//! The condition body inside `when ... [else ...]` is captured as an
//! opaque source slice (the ~80-atom condition grammar in spec §10
//! lands in a later wave).
//!
//! Out of scope for this slice (handled in later Waves):
//! - `template` / `structure` top-level blocks (Waves 1.2 / 1.3)
//! - `deliver` / `recast` / `morph` body blocks (Wave 1.4)
//! - The condition grammar inside `when … [else …]` — Wave 2.
//! - Shape vocabulary validation (12 primitives in spec §8) — Wave 2.
//! - Tag-name vocabulary lookup against `AbilityTag` — Wave 2.
//! - `cast_on <selector>` modifier (separate from these 9) — Wave 2.
//! - Lowering of the new headers / `passive` blocks / modifier slots
//!   (`crates/dsl_compiler/src/ability_lower.rs` errors with
//!   `HeaderNotImplemented` / `PassiveBlockNotImplemented` /
//!   `ModifierNotImplemented` for them pending Wave 2+ engine schema
//!   changes).
//!
//! The parser shares the lexer infrastructure (`Cursor`, `is_ident_start`,
//! `is_ident_cont`) with the `.sim` parser at `parser.rs`.

use crate::ast::*;
use crate::error::ParseError;
use crate::tokens::{is_ident_cont, is_ident_start, Cursor};

type PResult<T> = Result<T, ParseErr>;

#[derive(Debug, Clone)]
struct ParseErr {
    span: Span,
    context: Vec<String>,
    message: String,
}

impl ParseErr {
    fn at(span: Span, msg: impl Into<String>) -> Self {
        ParseErr { span, context: Vec::new(), message: msg.into() }
    }
    fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context.push(ctx.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Parse a `.ability` source string into an `AbilityFile` AST.
///
/// Errors carry a context chain (top-level → ability → header/effect)
/// and a rendered caret pointer matching the `.sim` parser's style.
pub fn parse_ability_file(source: &str) -> Result<AbilityFile, ParseError> {
    let mut c = Cursor::new(source);
    c.skip_ws();
    let mut abilities = Vec::new();
    let mut passives = Vec::new();
    while !c.eof() {
        let kw = peek_ident(&c);
        match kw.as_deref() {
            Some("ability") => match ability_decl(&mut c) {
                Ok(decl) => abilities.push(decl),
                Err(e) => return Err(ParseError::new(source, e.span, e.context, e.message)),
            },
            Some("passive") => match passive_decl(&mut c) {
                Ok(decl) => passives.push(decl),
                Err(e) => return Err(ParseError::new(source, e.span, e.context, e.message)),
            },
            Some("template") | Some("structure") => {
                // Out of scope for Wave 1.1; report a clean parse error so
                // hero templates that include these blocks fail loudly
                // rather than silently producing a partial AST.
                let kw = kw.unwrap();
                return Err(ParseError::new(
                    source,
                    here(&c),
                    vec!["parsing top-level `.ability` decl".to_string()],
                    format!("`{kw}` blocks are not supported in this parser slice (Wave 1.1); only `ability` and `passive` blocks are recognised"),
                ));
            }
            Some(other) => {
                return Err(ParseError::new(
                    source,
                    here(&c),
                    Vec::new(),
                    format!("expected `ability` or `passive` (top-level); got `{other}`"),
                ));
            }
            None => {
                return Err(ParseError::new(
                    source,
                    here(&c),
                    Vec::new(),
                    "expected `ability` or `passive` (top-level)".to_string(),
                ));
            }
        }
        c.skip_ws();
    }
    Ok(AbilityFile { abilities, passives })
}

// ---------------------------------------------------------------------------
// `ability <Name> { ... }` block
// ---------------------------------------------------------------------------

fn ability_decl(c: &mut Cursor) -> PResult<AbilityDecl> {
    let start = c.pos;
    expect_keyword(c, "ability")
        .map_err(|e| e.with_context("parsing `ability` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing ability name"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing ability body (expected `{`)"))?;
    let mut headers: Vec<AbilityHeader> = Vec::new();
    let mut effects: Vec<EffectStmt> = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        if c.eof() {
            return Err(ParseErr::at(here(c), "unexpected end of input inside ability body")
                .with_context(format!("parsing ability `{name}`")));
        }
        // Decide: header (`<ident>:`) vs marker header (`toggle`) vs
        // effect (`<ident> ...`). We peek an ident and look for a `:`
        // after it — same heuristic the spec implies (header keys are
        // `key: value`, effect verbs are bare). The Wave 1.1 marker
        // header `toggle` (spec §4.2 lists it as a flag) is a special
        // case: bare keyword, no colon, no value.
        if is_header_start(c) {
            let header = parse_header(c)
                .map_err(|e| e.with_context(format!("parsing header in ability `{name}`")))?;
            check_duplicate_header(&headers, &header)
                .map_err(|e| e.with_context(format!("parsing header in ability `{name}`")))?;
            headers.push(header);
            // Optional comma separator (allows `cooldown: 8s, cast: 400ms`
            // on a single line).
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
            }
        } else if is_marker_header(c, "toggle") {
            // Consume the bare `toggle` keyword.
            let _ = ident(c);
            let header = AbilityHeader::Toggle;
            check_duplicate_header(&headers, &header)
                .map_err(|e| e.with_context(format!("parsing header in ability `{name}`")))?;
            headers.push(header);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
            }
        } else {
            let effect = parse_effect(c)
                .map_err(|e| e.with_context(format!("parsing effect in ability `{name}`")))?;
            effects.push(effect);
        }
    }
    if headers.is_empty() && effects.is_empty() {
        return Err(ParseErr::at(
            Span::new(start, c.pos),
            format!("ability `{name}` has empty body — at least one header or effect is required"),
        ));
    }
    Ok(AbilityDecl { name, headers, effects, span: Span::new(start, c.pos) })
}

/// Return `true` if the cursor sits at the start of a header line —
/// i.e. an ident followed (after optional whitespace) by `:`. This
/// disambiguates header lines (`target: enemy`) from effect lines
/// (`damage 40 ...`) without consuming input.
fn is_header_start(c: &Cursor) -> bool {
    let Some(name) = peek_ident(c) else { return false };
    let after = c.pos + name.len();
    let mut look = Cursor { src: c.src, pos: after };
    // Skip inline whitespace (NOT comments; an unrelated `:` after a
    // comment-only line is irrelevant since headers are terse).
    while let Some(ch) = look.peek_char() {
        if ch == ' ' || ch == '\t' {
            look.bump(ch.len_utf8());
        } else {
            break;
        }
    }
    look.starts_with_char(':')
}

// ---------------------------------------------------------------------------
// Wave 1.1 — `passive <Name> { ... }` block (spec §5)
// ---------------------------------------------------------------------------

fn passive_decl(c: &mut Cursor) -> PResult<PassiveDecl> {
    let start = c.pos;
    expect_keyword(c, "passive")
        .map_err(|e| e.with_context("parsing `passive` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing passive name"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing passive body (expected `{`)"))?;
    let mut headers: Vec<PassiveHeader> = Vec::new();
    let mut effects: Vec<EffectStmt> = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        if c.eof() {
            return Err(ParseErr::at(here(c), "unexpected end of input inside passive body")
                .with_context(format!("parsing passive `{name}`")));
        }
        if is_header_start(c) {
            let header = parse_passive_header(c)
                .map_err(|e| e.with_context(format!("parsing header in passive `{name}`")))?;
            check_duplicate_passive_header(&headers, &header)
                .map_err(|e| e.with_context(format!("parsing header in passive `{name}`")))?;
            headers.push(header);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
            }
        } else {
            let effect = parse_effect(c)
                .map_err(|e| e.with_context(format!("parsing effect in passive `{name}`")))?;
            effects.push(effect);
        }
    }
    if headers.is_empty() && effects.is_empty() {
        return Err(ParseErr::at(
            Span::new(start, c.pos),
            format!("passive `{name}` has empty body — at least one header or effect is required"),
        ));
    }
    Ok(PassiveDecl { name, headers, effects, span: Span::new(start, c.pos) })
}

fn parse_passive_header(c: &mut Cursor) -> PResult<PassiveHeader> {
    c.skip_ws();
    let key_start = c.pos;
    let key = ident(c).map_err(|e| e.with_context("parsing passive header key"))?;
    c.skip_ws();
    expect_char(c, ':').map_err(|e| {
        e.with_context(format!("parsing passive header `{key}` (expected `:`)"))
    })?;
    c.skip_ws();
    let header = match key.as_str() {
        "trigger" => {
            // Trigger event names are open-ended (24+ kinds in §5.2 plus
            // `periodic`); we store the bare ident verbatim so lowering
            // can validate against the catalog without a parser re-rev
            // when new triggers land.
            let evt = ident(c).map_err(|e| e.with_context("parsing `trigger:` event name"))?;
            PassiveHeader::Trigger(evt)
        }
        "cooldown" => {
            let d = parse_duration(c)
                .map_err(|e| e.with_context("parsing passive `cooldown:` value"))?;
            PassiveHeader::Cooldown(d)
        }
        "range" => {
            let v = parse_number(c)
                .map_err(|e| e.with_context("parsing passive `range:` value"))?;
            PassiveHeader::Range(v as f32)
        }
        "hint" => {
            let h = parse_hint(c)
                .map_err(|e| e.with_context("parsing passive `hint:` value"))?;
            PassiveHeader::Hint(h)
        }
        other => {
            return Err(ParseErr::at(
                Span::new(key_start, key_start + other.len()),
                format!(
                    "unsupported passive header `{other}` — Wave 1.1 supports `trigger`, `cooldown`, `range`, `hint`"
                ),
            ));
        }
    };
    Ok(header)
}

fn check_duplicate_passive_header(
    headers: &[PassiveHeader],
    new: &PassiveHeader,
) -> PResult<()> {
    let same_kind = |a: &PassiveHeader, b: &PassiveHeader| {
        matches!(
            (a, b),
            (PassiveHeader::Trigger(_), PassiveHeader::Trigger(_))
                | (PassiveHeader::Cooldown(_), PassiveHeader::Cooldown(_))
                | (PassiveHeader::Range(_), PassiveHeader::Range(_))
                | (PassiveHeader::Hint(_), PassiveHeader::Hint(_))
        )
    };
    if headers.iter().any(|h| same_kind(h, new)) {
        let key = match new {
            PassiveHeader::Trigger(_) => "trigger",
            PassiveHeader::Cooldown(_) => "cooldown",
            PassiveHeader::Range(_) => "range",
            PassiveHeader::Hint(_) => "hint",
        };
        return Err(ParseErr::at(
            Span::new(0, 0),
            format!("duplicate `{key}:` header — each header may appear at most once"),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Headers
// ---------------------------------------------------------------------------

fn parse_header(c: &mut Cursor) -> PResult<AbilityHeader> {
    c.skip_ws();
    let key_start = c.pos;
    let key = ident(c).map_err(|e| e.with_context("parsing header key"))?;
    c.skip_ws();
    expect_char(c, ':').map_err(|e| {
        e.with_context(format!("parsing header `{key}` (expected `:`)"))
    })?;
    c.skip_ws();
    let header = match key.as_str() {
        "target" => {
            let mode = parse_target_mode(c)
                .map_err(|e| e.with_context("parsing `target:` value"))?;
            AbilityHeader::Target(mode)
        }
        "range" => {
            let v = parse_number(c)
                .map_err(|e| e.with_context("parsing `range:` value"))?;
            AbilityHeader::Range(v as f32)
        }
        "cooldown" => {
            let d = parse_duration(c)
                .map_err(|e| e.with_context("parsing `cooldown:` value"))?;
            AbilityHeader::Cooldown(d)
        }
        "cast" => {
            let d = parse_duration(c)
                .map_err(|e| e.with_context("parsing `cast:` value"))?;
            AbilityHeader::Cast(d)
        }
        "hint" => {
            let h = parse_hint(c).map_err(|e| e.with_context("parsing `hint:` value"))?;
            AbilityHeader::Hint(h)
        }
        "cost" => {
            let cost = parse_cost(c, key_start)
                .map_err(|e| e.with_context("parsing `cost:` value"))?;
            AbilityHeader::Cost(cost)
        }
        "charges" => {
            let n = parse_unsigned_int(c)
                .map_err(|e| e.with_context("parsing `charges:` value"))?;
            AbilityHeader::Charges(n)
        }
        "recharge" => {
            let d = parse_duration(c)
                .map_err(|e| e.with_context("parsing `recharge:` value"))?;
            AbilityHeader::Recharge(d)
        }
        other => {
            return Err(ParseErr::at(
                Span::new(key_start, key_start + other.len()),
                format!(
                    "unsupported header `{other}` — Wave 1.1 supports `target`, `range`, `cooldown`, `cast`, `hint`, `cost`, `charges`, `recharge`, `toggle`"
                ),
            ));
        }
    };
    Ok(header)
}

fn check_duplicate_header(headers: &[AbilityHeader], new: &AbilityHeader) -> PResult<()> {
    let same_kind = |a: &AbilityHeader, b: &AbilityHeader| {
        matches!(
            (a, b),
            (AbilityHeader::Target(_), AbilityHeader::Target(_))
                | (AbilityHeader::Range(_), AbilityHeader::Range(_))
                | (AbilityHeader::Cooldown(_), AbilityHeader::Cooldown(_))
                | (AbilityHeader::Cast(_), AbilityHeader::Cast(_))
                | (AbilityHeader::Hint(_), AbilityHeader::Hint(_))
                | (AbilityHeader::Cost(_), AbilityHeader::Cost(_))
                | (AbilityHeader::Charges(_), AbilityHeader::Charges(_))
                | (AbilityHeader::Recharge(_), AbilityHeader::Recharge(_))
                | (AbilityHeader::Toggle, AbilityHeader::Toggle)
        )
    };
    if headers.iter().any(|h| same_kind(h, new)) {
        // `toggle` is a marker (no `:`); the rest are `key: value`.
        // Render the diagnostic to match the source spelling so authors
        // see exactly the form the parser tripped on.
        let key_with_punct = match new {
            AbilityHeader::Target(_) => "target:",
            AbilityHeader::Range(_) => "range:",
            AbilityHeader::Cooldown(_) => "cooldown:",
            AbilityHeader::Cast(_) => "cast:",
            AbilityHeader::Hint(_) => "hint:",
            AbilityHeader::Cost(_) => "cost:",
            AbilityHeader::Charges(_) => "charges:",
            AbilityHeader::Recharge(_) => "recharge:",
            AbilityHeader::Toggle => "toggle",
        };
        return Err(ParseErr::at(
            Span::new(0, 0),
            format!("duplicate `{key_with_punct}` header — each header may appear at most once"),
        ));
    }
    Ok(())
}

/// True iff the cursor sits at the bare keyword `kw`, not followed by a
/// `:` (i.e. a marker header like `toggle`, not a `key: value` pair) and
/// not followed by an identifier-continuation char (so we don't misread
/// `toggle_cost` as `toggle`).
fn is_marker_header(c: &Cursor, kw: &str) -> bool {
    if !starts_with_keyword(c, kw) {
        return false;
    }
    // Walk past the kw + inline whitespace; if the next non-ws char is
    // `:` then this is a regular header line, not a marker.
    let after = c.pos + kw.len();
    let look = Cursor { src: c.src, pos: after };
    let mut probe = look;
    while let Some(ch) = probe.peek_char() {
        if ch == ' ' || ch == '\t' {
            probe.bump(ch.len_utf8());
        } else {
            break;
        }
    }
    !probe.starts_with_char(':')
}

fn parse_target_mode(c: &mut Cursor) -> PResult<TargetMode> {
    let name_span_start = c.pos;
    let name = ident(c)?;
    let mode = match name.as_str() {
        "enemy" => TargetMode::Enemy,
        "self" => TargetMode::Self_,
        "ally" => TargetMode::Ally,
        "self_aoe" => TargetMode::SelfAoe,
        "ground" => TargetMode::Ground,
        "direction" => TargetMode::Direction,
        "vector" => TargetMode::Vector,
        "global" => TargetMode::Global,
        other => {
            return Err(ParseErr::at(
                Span::new(name_span_start, name_span_start + other.len()),
                format!(
                    "unknown target mode `{other}` — expected one of: enemy, self, ally, self_aoe, ground, direction, vector, global"
                ),
            ));
        }
    };
    Ok(mode)
}

fn parse_hint(c: &mut Cursor) -> PResult<HintName> {
    let name_span_start = c.pos;
    let name = ident(c)?;
    let hint = match name.as_str() {
        "damage" => HintName::Damage,
        "defense" => HintName::Defense,
        "crowd_control" => HintName::CrowdControl,
        "utility" => HintName::Utility,
        "heal" => HintName::Heal,
        "economic" => HintName::Economic,
        "buff" => HintName::Buff,
        other => {
            return Err(ParseErr::at(
                Span::new(name_span_start, name_span_start + other.len()),
                format!(
                    "unknown hint `{other}` — expected one of: damage, defense, crowd_control, utility, heal, economic, buff"
                ),
            ));
        }
    };
    Ok(hint)
}

/// Parse a `cost:` value. Spec §4.2 lists `cost: int` ("mana/resource
/// gate"). Wave 1.1 accepts three surface forms:
///
/// 1. `cost: <amount>`              -> `Flat(amount), resource = Mana`
///    (matches the existing LoL hero corpus, where `cost: 12` means 12
///    mana by convention).
/// 2. `cost: <amount> <resource>`   -> `Flat(amount), resource = …`
/// 3. `cost: <amount>% <resource>`  -> `PercentOfMax(amount), resource = …`
///
/// Negative amounts are rejected at parse time; resource names that
/// fall outside the four spec'd values (mana / stamina / hp / gold)
/// produce a clean error pointing at the unknown token.
///
/// `key_start` is the byte offset of the `cost` key, used to assemble
/// the span.
fn parse_cost(c: &mut Cursor, key_start: usize) -> PResult<CostSpec> {
    let value_start = c.pos;
    // Lex the numeric magnitude (no unit suffix). We accept ints and
    // floats (LoL corpus uses ints; spec §4.2 says `int` but the four
    // resources may grow fractional in future without a parser
    // re-rev — the AST already stores `f32`).
    let (val, _is_float) = number_literal(c).map_err(|e| e.with_context("parsing cost amount"))?;
    if val < 0.0 {
        return Err(ParseErr::at(
            Span::new(value_start, c.pos),
            format!("cost amount must be >= 0; got {val}"),
        ));
    }
    // Optional `%` -> percent form.
    let is_percent = c.starts_with_char('%');
    if is_percent {
        c.bump(1);
    }
    // Optional resource keyword.
    skip_inline_ws_only(c);
    let resource = if let Some(name) = peek_ident(c) {
        // Three cases for the ident sitting after `cost: <num>[%]`:
        //   (a) a known resource keyword → consume + classify
        //   (b) an unknown ident NOT followed by `:` → reject (clearly
        //       intended as a resource, but the resource is unknown)
        //   (c) any ident followed by `:` → it's the next header; do
        //       not consume; default to Mana (per §4.2 mana/resource).
        // The percent form `cost: 25% hp` REQUIRES case (a) — falling
        // through to (c) would produce silent surprise (PercentOfMax
        // with no explicit resource).
        match name.as_str() {
            "mana" => {
                c.bump(name.len());
                CostResource::Mana
            }
            "stamina" => {
                c.bump(name.len());
                CostResource::Stamina
            }
            "hp" => {
                c.bump(name.len());
                CostResource::Hp
            }
            "gold" => {
                c.bump(name.len());
                CostResource::Gold
            }
            other => {
                // Look ahead past the ident to see if a `:` follows
                // (= next header, leave it). Anything else is a bad
                // resource keyword — error out.
                let after_ident = c.pos + other.len();
                let mut probe = Cursor { src: c.src, pos: after_ident };
                while let Some(ch) = probe.peek_char() {
                    if ch == ' ' || ch == '\t' {
                        probe.bump(ch.len_utf8());
                    } else {
                        break;
                    }
                }
                if probe.starts_with_char(':') && !is_percent {
                    // Next header — don't consume; default to Mana.
                    // (Only safe in the bare-number form; the percent
                    // form mandates an explicit resource.)
                    CostResource::Mana
                } else {
                    return Err(ParseErr::at(
                        Span::new(c.pos, c.pos + other.len()),
                        format!(
                            "unknown cost resource `{other}` — expected one of: mana, stamina, hp, gold (or omit for default mana)"
                        ),
                    ));
                }
            }
        }
    } else {
        // No identifier follows — default to Mana per §4.2. The
        // percent form should not reach here in well-formed input
        // (a `}` or newline immediately after `25%` is the only way
        // for `peek_ident` to be `None`); accept the default Mana.
        CostResource::Mana
    };
    let amount = if is_percent {
        CostAmount::PercentOfMax(val as f32)
    } else {
        CostAmount::Flat(val as f32)
    };
    Ok(CostSpec {
        resource,
        amount,
        span: Span::new(key_start, c.pos),
    })
}

/// Parse a non-negative integer (used by `charges:`). Floats with a
/// non-zero fractional part are rejected so `charges: 3.5` is a clean
/// parse error rather than a silent truncation.
fn parse_unsigned_int(c: &mut Cursor) -> PResult<u32> {
    c.skip_ws();
    let start = c.pos;
    let (val, is_float) = number_literal(c)?;
    if val < 0.0 || (is_float && val.fract() != 0.0) {
        return Err(ParseErr::at(
            Span::new(start, c.pos),
            format!("expected a non-negative integer; got {val}"),
        ));
    }
    if val > u32::MAX as f64 {
        return Err(ParseErr::at(
            Span::new(start, c.pos),
            format!("integer literal {val} overflows u32"),
        ));
    }
    Ok(val as u32)
}

// ---------------------------------------------------------------------------
// Effect statements
// ---------------------------------------------------------------------------

fn parse_effect(c: &mut Cursor) -> PResult<EffectStmt> {
    let start = c.pos;
    let verb = ident(c).map_err(|e| e.with_context("parsing effect verb"))?;
    let mut args: Vec<EffectArg> = Vec::new();
    // Wave 1.5 modifier slots — collected into the EffectStmt below.
    let mut area: Option<EffectArea> = None;
    let mut tags: Vec<EffectTag> = Vec::new();
    let mut duration: Option<EffectDuration> = None;
    let mut condition: Option<EffectCondition> = None;
    let mut chance: Option<EffectChance> = None;
    let mut stacking: Option<StackingMode> = None;
    let mut scalings: Vec<EffectScaling> = Vec::new();
    let mut lifetime: Option<EffectLifetime> = None;
    let mut nested: Vec<EffectStmt> = Vec::new();

    // Phase 1: collect leading positional args (numbers / durations /
    // percents / strings / idents). Stops at the first modifier token.
    loop {
        // Skip ONLY inline whitespace — a newline ends the statement.
        skip_inline_ws_only(c);
        if c.eof() || c.starts_with_char('\n') || c.starts_with_char('}') {
            break;
        }
        // Comments terminate the line.
        if c.starts_with("//") || c.starts_with_char('#') {
            consume_to_eol(c);
            break;
        }
        if is_modifier_start(c) {
            break;
        }
        // Trailing `,` is unusual on effect lines but we accept it as
        // a no-op to mirror header tolerance (defensive).
        if c.starts_with_char(',') {
            c.bump(1);
            continue;
        }
        let arg = parse_effect_arg(c)?;
        args.push(arg);
    }

    // Phase 2: dispatch modifiers. Modifiers may appear in any order;
    // every keyword maps to a distinct slot; duplicates on single-value
    // slots are a parse error.
    loop {
        // Modifiers may span lines: `damage 50` then on the next line
        // `+ 30% AP`. We do NOT cross a blank line however — a blank
        // line ends the statement. The skip_inline_ws_only + newline
        // probe below mirrors the phase-1 termination rule but allows
        // wrapping if the next non-ws char IS a modifier start.
        skip_inline_ws_only(c);
        if c.eof() || c.starts_with_char('}') {
            break;
        }
        if c.starts_with("//") || c.starts_with_char('#') {
            consume_to_eol(c);
            break;
        }
        if c.starts_with_char('\n') {
            // Look ahead past the newline + leading whitespace to see
            // if the next token continues this statement (only `+`,
            // `[`, `{`, or one of the modifier keywords). Otherwise
            // the statement ends.
            if !modifier_continues_after_newline(c) {
                break;
            }
            // Consume the newline + any inline whitespace; the next
            // iteration parses the modifier.
            while let Some(ch) = c.peek_char() {
                if ch.is_whitespace() {
                    c.bump(ch.len_utf8());
                } else {
                    break;
                }
            }
            continue;
        }
        if !is_modifier_start(c) {
            // Anything else here is junk after the positional args
            // that wasn't a recognised modifier — error loudly so the
            // author sees the offending token.
            return Err(ParseErr::at(
                here(c),
                format!(
                    "unknown modifier or trailing token `{}` in effect `{verb}`; expected one of: in / [TAG: …] / for / when / chance / stacking / + N% stat / until_caster_dies / damageable_hp / {{ nested }}",
                    peek_word_for_error(c)
                ),
            ));
        }
        let mod_start = c.pos;
        if c.starts_with_char('[') {
            let tag = parse_tag(c)?;
            tags.push(tag);
            continue;
        }
        if c.starts_with_char('{') {
            let block = parse_nested_block(c)?;
            // Spec §6.1 doesn't constrain the count of nested-block
            // modifiers per effect; accept any number, append in order.
            nested.extend(block);
            continue;
        }
        if c.starts_with_char('+') {
            let s = parse_scaling(c)?;
            scalings.push(s);
            continue;
        }
        if starts_with_keyword(c, "in") {
            if area.is_some() {
                return Err(ParseErr::at(
                    Span::new(mod_start, c.pos),
                    format!("duplicate `in <shape>` modifier on effect `{verb}` — at most one allowed"),
                ));
            }
            area = Some(parse_area(c)?);
            continue;
        }
        if starts_with_keyword(c, "for") {
            if duration.is_some() {
                return Err(ParseErr::at(
                    Span::new(mod_start, c.pos),
                    format!("duplicate `for <duration>` modifier on effect `{verb}` — at most one allowed"),
                ));
            }
            duration = Some(parse_for_duration(c)?);
            continue;
        }
        if starts_with_keyword(c, "when") {
            if condition.is_some() {
                return Err(ParseErr::at(
                    Span::new(mod_start, c.pos),
                    format!("duplicate `when <cond>` modifier on effect `{verb}` — at most one allowed"),
                ));
            }
            condition = Some(parse_condition(c)?);
            continue;
        }
        if starts_with_keyword(c, "chance") {
            if chance.is_some() {
                return Err(ParseErr::at(
                    Span::new(mod_start, c.pos),
                    format!("duplicate `chance` modifier on effect `{verb}` — at most one allowed"),
                ));
            }
            chance = Some(parse_chance(c)?);
            continue;
        }
        if starts_with_keyword(c, "stacking") {
            if stacking.is_some() {
                return Err(ParseErr::at(
                    Span::new(mod_start, c.pos),
                    format!("duplicate `stacking` modifier on effect `{verb}` — at most one allowed"),
                ));
            }
            stacking = Some(parse_stacking(c)?);
            continue;
        }
        if starts_with_keyword(c, "until_caster_dies") {
            if lifetime.is_some() {
                return Err(ParseErr::at(
                    Span::new(mod_start, c.pos),
                    format!("duplicate lifetime modifier on effect `{verb}` — at most one of `until_caster_dies` / `damageable_hp` allowed"),
                ));
            }
            let s = c.pos;
            c.bump("until_caster_dies".len());
            lifetime = Some(EffectLifetime::UntilCasterDies { span: Span::new(s, c.pos) });
            continue;
        }
        if starts_with_keyword(c, "damageable_hp") {
            if lifetime.is_some() {
                return Err(ParseErr::at(
                    Span::new(mod_start, c.pos),
                    format!("duplicate lifetime modifier on effect `{verb}` — at most one of `until_caster_dies` / `damageable_hp` allowed"),
                ));
            }
            lifetime = Some(parse_damageable_hp(c)?);
            continue;
        }
        // Defensive: shouldn't be reachable because is_modifier_start
        // above returned true. Belt + braces.
        return Err(ParseErr::at(
            here(c),
            format!(
                "unknown modifier `{}` in effect `{verb}`",
                peek_word_for_error(c)
            ),
        ));
    }

    Ok(EffectStmt {
        verb,
        args,
        span: Span::new(start, c.pos),
        area,
        tags,
        duration,
        condition,
        chance,
        stacking,
        scalings,
        lifetime,
        nested,
    })
}

/// Return `true` iff the cursor sits at the start of a known modifier
/// token. Used by `parse_effect` to decide between "another positional
/// arg" and "first modifier slot". The set of triggers is the disjoint
/// union of:
///  * `[`            — `[TAG: value]`
///  * `{`            — nested block
///  * `+`            — `+ N% stat_ref`
///  * `in` / `for` / `when` / `chance` / `stacking`
///  * `until_caster_dies` / `damageable_hp` (lifetime keywords)
fn is_modifier_start(c: &Cursor) -> bool {
    if c.starts_with_char('[') || c.starts_with_char('{') || c.starts_with_char('+') {
        return true;
    }
    starts_with_keyword(c, "in")
        || starts_with_keyword(c, "for")
        || starts_with_keyword(c, "when")
        || starts_with_keyword(c, "chance")
        || starts_with_keyword(c, "stacking")
        || starts_with_keyword(c, "until_caster_dies")
        || starts_with_keyword(c, "damageable_hp")
}

/// After hitting a `\n` inside an effect statement, peek past the
/// whitespace and decide whether the next non-blank line continues the
/// modifier list. Returns `true` only when the next token is a clear
/// modifier start AND it isn't a header line for the next effect /
/// ability body.
///
/// We deliberately do NOT continue across newlines for `in` / `for` /
/// `when` / `chance` / `stacking` / `until_caster_dies` /
/// `damageable_hp` because those keywords could be the start of an
/// effect verb on the next line in a malformed program (e.g. an author
/// who omitted a verb). Continuation is only safe for the purely
/// punctuation-led modifier syntaxes: `+`, `[`, `{`. Any keyword-led
/// modifier on a continuation line should be on the same line as the
/// effect verb.
fn modifier_continues_after_newline(c: &Cursor) -> bool {
    let mut probe = Cursor { src: c.src, pos: c.pos };
    while let Some(ch) = probe.peek_char() {
        if ch.is_whitespace() {
            probe.bump(ch.len_utf8());
        } else {
            break;
        }
    }
    if probe.eof() {
        return false;
    }
    probe.starts_with_char('+') || probe.starts_with_char('[') || probe.starts_with_char('{')
}

fn consume_to_eol(c: &mut Cursor) {
    while let Some(ch) = c.peek_char() {
        if ch == '\n' {
            break;
        }
        c.bump(ch.len_utf8());
    }
}

// ---------------------------------------------------------------------------
// Wave 1.5 — per-modifier sub-parsers
// ---------------------------------------------------------------------------

/// Parse `in <shape>(args…)`. The cursor starts at `in`; on success it
/// sits past the closing `)`.
///
/// Per spec §8 there are 12 shape primitives plus CSG composition. Wave
/// 1.5 stores the shape name verbatim and a flat `Vec<f32>` of args;
/// the lowering pass validates the name + arity. CSG composition
/// (`union` / `diff` / `intersect`) is NOT recognised here — the corpus
/// post-Wave-1.5 sticks to a single shape per effect (Aatrox-style
/// `in circle(2.5)`); CSG-bearing effects will need a follow-up parse
/// pass when they show up in real source.
fn parse_area(c: &mut Cursor) -> PResult<EffectArea> {
    let start = c.pos;
    expect_keyword(c, "in").map_err(|e| e.with_context("parsing `in <shape>` modifier"))?;
    skip_inline_ws_only(c);
    let shape = ident(c).map_err(|e| e.with_context("parsing shape name after `in`"))?;
    skip_inline_ws_only(c);
    expect_char(c, '(').map_err(|e| {
        e.with_context(format!("parsing shape `{shape}(…)` arg list (expected `(`)"))
    })?;
    let mut args: Vec<f32> = Vec::new();
    loop {
        skip_inline_ws_only(c);
        if c.starts_with_char(')') {
            c.bump(1);
            break;
        }
        if c.eof() {
            return Err(ParseErr::at(here(c), format!("unexpected end of input inside `{shape}(…)`")));
        }
        let (val, _is_float) = number_literal(c)
            .map_err(|e| e.with_context(format!("parsing arg in `{shape}(…)`")))?;
        // Shapes accept `5deg` for cone angles per spec §8 (`cone(r,
        // angle_deg)` — the corpus uses bare numbers but `45deg` is a
        // documented form). Accept the suffix and store the bare
        // number (degrees) — Wave 1.5 stores opaquely; lowering owns
        // unit normalisation.
        if c.starts_with("deg") {
            // Match only on a bare unit token (not part of a longer
            // ident).
            let after = c.pos + 3;
            let next = c.src[after..].chars().next();
            if next.map_or(true, |ch| !is_ident_cont(ch)) {
                c.bump(3);
            }
        }
        args.push(val as f32);
        skip_inline_ws_only(c);
        if c.starts_with_char(',') {
            c.bump(1);
            continue;
        }
        if c.starts_with_char(')') {
            c.bump(1);
            break;
        }
        return Err(ParseErr::at(
            here(c),
            format!("expected `,` or `)` inside `{shape}(…)`; got `{}`", peek_word_for_error(c)),
        ));
    }
    Ok(EffectArea { shape, args, span: Span::new(start, c.pos) })
}

/// Parse one `[TAG: value]` power tag. The cursor sits at `[`; on
/// success it sits past the matching `]`.
fn parse_tag(c: &mut Cursor) -> PResult<EffectTag> {
    let start = c.pos;
    expect_char(c, '[').map_err(|e| e.with_context("parsing `[TAG: value]` modifier"))?;
    skip_inline_ws_only(c);
    let name = ident(c).map_err(|e| e.with_context("parsing tag name inside `[…]`"))?;
    skip_inline_ws_only(c);
    expect_char(c, ':').map_err(|e| {
        e.with_context(format!("parsing tag `[{name}: …]` (expected `:`)"))
    })?;
    skip_inline_ws_only(c);
    let (val, _is_float) = number_literal(c)
        .map_err(|e| e.with_context(format!("parsing value of tag `[{name}: …]`")))?;
    skip_inline_ws_only(c);
    expect_char(c, ']').map_err(|e| {
        e.with_context(format!("parsing tag `[{name}: …]` (expected `]`)"))
    })?;
    Ok(EffectTag { name, value: val as f32, span: Span::new(start, c.pos) })
}

/// Parse `for <duration>`. Cursor starts at `for`.
fn parse_for_duration(c: &mut Cursor) -> PResult<EffectDuration> {
    let start = c.pos;
    expect_keyword(c, "for").map_err(|e| e.with_context("parsing `for <duration>` modifier"))?;
    let d = parse_duration(c).map_err(|e| e.with_context("parsing `for <duration>` value"))?;
    Ok(EffectDuration { duration: d, span: Span::new(start, c.pos) })
}

/// Parse `when <cond> [else <cond>]`. The condition language (~80
/// atoms in spec §10) is owned by the expression parser; Wave 1.5
/// captures opaque source slices terminated by the next modifier
/// keyword / EOL / `}`. Inside parens / brackets / braces we follow
/// nesting so `when (a or b)` doesn't terminate at the inner `)`.
fn parse_condition(c: &mut Cursor) -> PResult<EffectCondition> {
    let start = c.pos;
    expect_keyword(c, "when").map_err(|e| e.with_context("parsing `when <cond>` modifier"))?;
    let when_cond = capture_cond_text(c, /* stop_on_else = */ true)?;
    if when_cond.is_empty() {
        return Err(ParseErr::at(
            here(c),
            "empty condition body after `when`".to_string(),
        ));
    }
    let else_cond = if starts_with_keyword(c, "else") {
        c.bump("else".len());
        let body = capture_cond_text(c, /* stop_on_else = */ false)?;
        if body.is_empty() {
            return Err(ParseErr::at(
                here(c),
                "empty condition body after `else`".to_string(),
            ));
        }
        Some(body)
    } else {
        None
    };
    Ok(EffectCondition {
        when_cond,
        else_cond,
        span: Span::new(start, c.pos),
    })
}

/// Capture the verbatim source slice of a `when` / `else` condition
/// body. Stops at:
///  - end of statement (`\n`, `;`, EOF, `}`)
///  - any other modifier keyword (`in`/`for`/`chance`/`stacking`/
///    `until_caster_dies`/`damageable_hp`)
///  - a `[…]` power tag start (`[`) or nested block (`{`) at top level
///  - if `stop_on_else` is true: the `else` keyword (so the caller can
///    consume it and recurse for the alternative branch)
///
/// Balances `()` so `when (a or b)` doesn't terminate at `)`.
fn capture_cond_text(c: &mut Cursor, stop_on_else: bool) -> PResult<String> {
    skip_inline_ws_only(c);
    let start = c.pos;
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;
    while let Some(ch) = c.peek_char() {
        // Termination conditions only apply at outer nesting depth.
        if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 {
            if ch == '\n' || ch == ';' || ch == '}' {
                break;
            }
            // Comment ends the condition.
            if c.starts_with("//") || c.starts_with_char('#') {
                break;
            }
            if stop_on_else && starts_with_keyword(c, "else") {
                break;
            }
            // Power tag / nested block / scaling all end the condition
            // body (the caller's modifier loop will pick them up).
            if ch == '[' || ch == '{' || ch == '+' {
                break;
            }
            if starts_with_keyword(c, "in")
                || starts_with_keyword(c, "for")
                || starts_with_keyword(c, "chance")
                || starts_with_keyword(c, "stacking")
                || starts_with_keyword(c, "until_caster_dies")
                || starts_with_keyword(c, "damageable_hp")
            {
                break;
            }
            // Top-level `when` after a `when` body terminates that
            // body — the caller's modifier loop then sees the second
            // `when` and produces a duplicate-slot error. Inside an
            // `else` body (stop_on_else=false), `when` is allowed: the
            // brief's example `... when X else when Y` reads "else
            // when Y" as the alternative branch's body, not a fresh
            // outer-level `when` modifier.
            if stop_on_else && starts_with_keyword(c, "when") {
                break;
            }
        }
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth = (paren_depth - 1).max(0),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = (bracket_depth - 1).max(0),
            '{' => brace_depth += 1,
            '}' => brace_depth -= 1,
            '"' => {
                // Skip past the matching quote (cheap escape handling).
                c.bump(1);
                while let Some(qc) = c.peek_char() {
                    if qc == '\\' {
                        c.bump(1);
                        if let Some(esc) = c.peek_char() {
                            c.bump(esc.len_utf8());
                        }
                        continue;
                    }
                    if qc == '"' {
                        c.bump(1);
                        break;
                    }
                    c.bump(qc.len_utf8());
                }
                continue;
            }
            _ => {}
        }
        c.bump(ch.len_utf8());
    }
    Ok(c.src[start..c.pos].trim().to_string())
}

/// Parse `chance N%`. The trailing `%` is mandatory per the brief
/// (`chance 30%` -> `EffectChance { p: 0.30 }`). A bare number form
/// (`chance 0.30`) is rejected so authors don't accidentally type
/// `chance 30` and silently get a 30x amplification.
fn parse_chance(c: &mut Cursor) -> PResult<EffectChance> {
    let start = c.pos;
    expect_keyword(c, "chance").map_err(|e| e.with_context("parsing `chance N%` modifier"))?;
    skip_inline_ws_only(c);
    let val_start = c.pos;
    let (val, _is_float) = number_literal(c)
        .map_err(|e| e.with_context("parsing `chance` value"))?;
    if !c.starts_with_char('%') {
        return Err(ParseErr::at(
            Span::new(val_start, c.pos),
            "`chance` value must carry a `%` suffix (e.g. `chance 25%`)".to_string(),
        ));
    }
    c.bump(1);
    let p = (val as f32) / 100.0;
    if !p.is_finite() || p < 0.0 || p > 1.0 {
        return Err(ParseErr::at(
            Span::new(val_start, c.pos),
            format!("`chance {val}%` is out of range; expected 0%..=100%"),
        ));
    }
    Ok(EffectChance { p, span: Span::new(start, c.pos) })
}

/// Parse `stacking refresh|stack|extend`.
fn parse_stacking(c: &mut Cursor) -> PResult<StackingMode> {
    expect_keyword(c, "stacking").map_err(|e| e.with_context("parsing `stacking <mode>` modifier"))?;
    skip_inline_ws_only(c);
    let mode_start = c.pos;
    let name = ident(c).map_err(|e| e.with_context("parsing stacking mode"))?;
    match name.as_str() {
        "refresh" => Ok(StackingMode::Refresh),
        "stack" => Ok(StackingMode::Stack),
        "extend" => Ok(StackingMode::Extend),
        other => Err(ParseErr::at(
            Span::new(mode_start, mode_start + other.len()),
            format!(
                "unknown stacking mode `{other}` — expected one of: refresh, stack, extend"
            ),
        )),
    }
}

/// Parse `+ N% stat_ref`. The cursor starts at `+`. `stat_ref` is the
/// next ident-like token; we accept `AP`, `AD`, `self.hp` and similar
/// dotted refs (the dot-segment is captured verbatim — lowering owns
/// resolution).
fn parse_scaling(c: &mut Cursor) -> PResult<EffectScaling> {
    let start = c.pos;
    expect_char(c, '+').map_err(|e| e.with_context("parsing `+ N% stat_ref` modifier"))?;
    skip_inline_ws_only(c);
    let val_start = c.pos;
    let (val, _is_float) = number_literal(c)
        .map_err(|e| e.with_context("parsing scaling percent"))?;
    if !c.starts_with_char('%') {
        return Err(ParseErr::at(
            Span::new(val_start, c.pos),
            "scaling value must carry a `%` suffix (e.g. `+ 30% AP`)".to_string(),
        ));
    }
    c.bump(1);
    skip_inline_ws_only(c);
    let stat_ref = parse_stat_ref(c).map_err(|e| e.with_context("parsing `+ N% stat_ref` stat reference"))?;
    Ok(EffectScaling {
        percent: val as f32,
        stat_ref,
        span: Span::new(start, c.pos),
    })
}

/// Parse a stat reference: an ident followed by zero or more
/// `.segment` continuations. Examples: `AP`, `AD`, `self.hp`,
/// `target.max_hp`. Stops at the first non-ident / non-dot char.
fn parse_stat_ref(c: &mut Cursor) -> PResult<String> {
    let start = c.pos;
    let _first = ident(c)?;
    loop {
        if !c.starts_with_char('.') {
            break;
        }
        c.bump(1);
        // Tolerate a missing segment (`AP.`) by erroring.
        let next = c.peek_char();
        if next.map_or(true, |ch| !is_ident_start(ch)) {
            return Err(ParseErr::at(
                here(c),
                "expected identifier after `.` in stat reference".to_string(),
            ));
        }
        let _seg = ident(c)?;
    }
    Ok(c.src[start..c.pos].to_string())
}

/// Parse `damageable_hp(N)`. Cursor starts at `damageable_hp`.
fn parse_damageable_hp(c: &mut Cursor) -> PResult<EffectLifetime> {
    let start = c.pos;
    expect_keyword(c, "damageable_hp").map_err(|e| e.with_context("parsing `damageable_hp(N)` modifier"))?;
    skip_inline_ws_only(c);
    expect_char(c, '(').map_err(|e| e.with_context("parsing `damageable_hp(N)` (expected `(`)"))?;
    skip_inline_ws_only(c);
    let (val, _is_float) = number_literal(c)
        .map_err(|e| e.with_context("parsing `damageable_hp(N)` HP value"))?;
    skip_inline_ws_only(c);
    expect_char(c, ')').map_err(|e| e.with_context("parsing `damageable_hp(N)` (expected `)`)"))?;
    Ok(EffectLifetime::DamageableHp { hp: val as f32, span: Span::new(start, c.pos) })
}

/// Parse a nested `{ … }` block of effect statements. The cursor sits
/// at `{`; on success it sits past the matching `}`. Returns the
/// flattened list of inner `EffectStmt`s in source order. Inner
/// effects parse with the full Wave 1.5 modifier surface (recursion is
/// allowed — `heal 50 { stun 1s when target.alive }` works).
fn parse_nested_block(c: &mut Cursor) -> PResult<Vec<EffectStmt>> {
    expect_char(c, '{').map_err(|e| e.with_context("parsing nested `{ … }` block modifier"))?;
    let mut out: Vec<EffectStmt> = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        if c.eof() {
            return Err(ParseErr::at(here(c), "unexpected end of input inside nested `{ … }`".to_string()));
        }
        let stmt = parse_effect(c)
            .map_err(|e| e.with_context("parsing effect inside nested `{ … }` block"))?;
        out.push(stmt);
    }
    Ok(out)
}

fn parse_effect_arg(c: &mut Cursor) -> PResult<EffectArg> {
    c.skip_ws();
    if c.starts_with_char('"') {
        let s = string_lit(c)?;
        return Ok(EffectArg::String(s));
    }
    if peek_number_or_sign(c) {
        // Could be a duration (`5s`/`300ms`/`1.5s`), a percent (`30%`),
        // or a plain number. Lex the numeric, then check the suffix.
        let (val, _is_float) = number_literal(c)?;
        // Inline whitespace is NOT allowed between the number and the
        // unit suffix per the spec lex rule (`5s`, not `5 s`).
        if c.starts_with("ms") {
            c.bump(2);
            let millis = (val.round() as i64).max(0) as u32;
            return Ok(EffectArg::Duration(Duration { millis }));
        }
        if c.starts_with_char('s') {
            // Check it's not part of a longer ident (e.g. `5stacks`).
            let next = c.src[c.pos + 1..].chars().next();
            let is_unit = next.map_or(true, |ch| !is_ident_cont(ch));
            if is_unit {
                c.bump(1);
                let millis = (val * 1000.0).round().max(0.0) as u32;
                return Ok(EffectArg::Duration(Duration { millis }));
            }
        }
        if c.starts_with_char('%') {
            c.bump(1);
            return Ok(EffectArg::Percent(val as f32));
        }
        return Ok(EffectArg::Number(val as f32));
    }
    if let Some(name) = peek_ident(c) {
        // Bump past the ident.
        c.bump(name.len());
        return Ok(EffectArg::Ident(name));
    }
    Err(ParseErr::at(
        here(c),
        format!("expected effect argument; got `{}`", peek_word_for_error(c)),
    ))
}

// ---------------------------------------------------------------------------
// Number / duration / percent lex helpers
// ---------------------------------------------------------------------------

fn parse_number(c: &mut Cursor) -> PResult<f64> {
    c.skip_ws();
    let (v, _is_float) = number_literal(c)?;
    Ok(v)
}

fn parse_duration(c: &mut Cursor) -> PResult<Duration> {
    c.skip_ws();
    let start = c.pos;
    let (val, _is_float) = number_literal(c)?;
    if c.starts_with("ms") {
        c.bump(2);
        let millis = (val.round() as i64).max(0) as u32;
        return Ok(Duration { millis });
    }
    if c.starts_with_char('s') {
        let next = c.src[c.pos + 1..].chars().next();
        let is_unit = next.map_or(true, |ch| !is_ident_cont(ch));
        if is_unit {
            c.bump(1);
            let millis = (val * 1000.0).round().max(0.0) as u32;
            return Ok(Duration { millis });
        }
    }
    // Bare integer = milliseconds (per spec §6 / lex rules in this
    // file's module doc). `cast: 0` is a common form.
    let millis = (val.round() as i64).max(0) as u32;
    let _ = start;
    Ok(Duration { millis })
}

fn peek_number_or_sign(c: &Cursor) -> bool {
    match c.peek_char() {
        Some(ch) if ch.is_ascii_digit() => true,
        Some('-') => {
            // Look at the next char.
            let next = c.src[c.pos + 1..].chars().next();
            matches!(next, Some(n) if n.is_ascii_digit())
        }
        Some('.') => {
            let next = c.src[c.pos + 1..].chars().next();
            matches!(next, Some(n) if n.is_ascii_digit())
        }
        _ => false,
    }
}

/// Parse a numeric literal — duplicated from `parser.rs` so we can keep
/// the .ability lexer self-contained without exporting internals from
/// the .sim parser. Returns `(value, is_float)`. Accepts a leading `-`,
/// digits with `_` separators, optional fractional part, optional
/// exponent.
fn number_literal(c: &mut Cursor) -> PResult<(f64, bool)> {
    c.skip_ws();
    let start = c.pos;
    if c.starts_with_char('-') {
        c.bump(1);
    }
    while let Some(ch) = c.peek_char() {
        if ch.is_ascii_digit() || ch == '_' {
            c.bump(1);
        } else {
            break;
        }
    }
    let mut is_float = false;
    if c.starts_with_char('.') {
        let after = c.pos + 1;
        let next = c.src[after..].chars().next();
        if let Some(n) = next {
            if n.is_ascii_digit() {
                c.bump(1);
                while let Some(ch) = c.peek_char() {
                    if ch.is_ascii_digit() || ch == '_' {
                        c.bump(1);
                    } else {
                        break;
                    }
                }
                is_float = true;
            }
        }
    }
    if c.starts_with_char('e') || c.starts_with_char('E') {
        c.bump(1);
        if c.starts_with_char('+') || c.starts_with_char('-') {
            c.bump(1);
        }
        while let Some(ch) = c.peek_char() {
            if ch.is_ascii_digit() {
                c.bump(1);
            } else {
                break;
            }
        }
        is_float = true;
    }
    let raw = c.src[start..c.pos].replace('_', "");
    if raw.is_empty() || raw == "-" {
        return Err(ParseErr::at(here(c), "expected numeric literal"));
    }
    let v = raw.parse::<f64>().map_err(|_| {
        ParseErr::at(Span::new(start, c.pos), format!("invalid numeric literal `{raw}`"))
    })?;
    Ok((v, is_float))
}

fn string_lit(c: &mut Cursor) -> PResult<String> {
    c.skip_ws();
    expect_char(c, '"').map_err(|e| e.with_context("parsing string literal"))?;
    let start = c.pos;
    while let Some(ch) = c.peek_char() {
        if ch == '"' {
            break;
        }
        if ch == '\\' {
            c.bump(1);
            if !c.eof() {
                let esc = c.peek_char().unwrap();
                c.bump(esc.len_utf8());
            }
            continue;
        }
        c.bump(ch.len_utf8());
    }
    let raw = c.src[start..c.pos].to_string();
    expect_char(c, '"')?;
    let mut out = String::with_capacity(raw.len());
    let mut it = raw.chars();
    while let Some(ch) = it.next() {
        if ch == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Token helpers (mirrors of the .sim parser's primitives)
// ---------------------------------------------------------------------------

fn ident(c: &mut Cursor) -> PResult<String> {
    c.skip_ws();
    let start = c.pos;
    let first = c.peek_char().ok_or_else(|| ParseErr::at(here(c), "expected identifier"))?;
    if !is_ident_start(first) {
        return Err(ParseErr::at(here(c), format!("expected identifier; got `{first}`")));
    }
    c.bump(first.len_utf8());
    while let Some(ch) = c.peek_char() {
        if !is_ident_cont(ch) {
            break;
        }
        c.bump(ch.len_utf8());
    }
    Ok(c.src[start..c.pos].to_string())
}

fn peek_ident(c: &Cursor) -> Option<String> {
    let rem = c.remaining();
    let mut it = rem.chars();
    let first = it.next()?;
    if !is_ident_start(first) {
        return None;
    }
    let mut end = first.len_utf8();
    for ch in it {
        if !is_ident_cont(ch) {
            break;
        }
        end += ch.len_utf8();
    }
    Some(rem[..end].to_string())
}

fn starts_with_keyword(c: &Cursor, kw: &str) -> bool {
    let rem = c.remaining();
    if !rem.starts_with(kw) {
        return false;
    }
    let next = rem[kw.len()..].chars().next();
    next.map_or(true, |ch| !is_ident_cont(ch))
}

fn expect_keyword(c: &mut Cursor, kw: &str) -> PResult<()> {
    c.skip_ws();
    if starts_with_keyword(c, kw) {
        c.bump(kw.len());
        Ok(())
    } else {
        Err(ParseErr::at(here(c), format!("expected keyword `{kw}`")))
    }
}

fn expect_char(c: &mut Cursor, ch: char) -> PResult<()> {
    c.skip_ws();
    if c.starts_with_char(ch) {
        c.bump(ch.len_utf8());
        Ok(())
    } else {
        Err(ParseErr::at(here(c), format!("expected `{ch}`")))
    }
}

fn skip_inline_ws_only(c: &mut Cursor) {
    while let Some(ch) = c.peek_char() {
        if ch == ' ' || ch == '\t' {
            c.bump(ch.len_utf8());
        } else {
            break;
        }
    }
}

fn here(c: &Cursor) -> Span {
    Span::new(c.pos, c.pos + c.peek_char().map_or(1, |ch| ch.len_utf8()))
}

fn peek_word_for_error(c: &Cursor) -> String {
    let rem = c.remaining();
    let end = rem
        .find(|ch: char| ch.is_whitespace() || ch == '{' || ch == '(' || ch == ';')
        .unwrap_or(rem.len());
    rem[..end].to_string()
}
