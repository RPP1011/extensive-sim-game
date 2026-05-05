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
//! Effect statements are parsed as `verb arg* <ignored-modifier-tail>`:
//! the verb name is preserved verbatim (validation against the verb
//! catalog lives in lowering, Wave 1.6) and the first run of simple
//! positional arguments (numbers / durations / percents / strings /
//! idents) is collected. When the parser sees a modifier keyword (`in`,
//! `for`, `when`, `chance`, `stacking`, `+`), a bracketed power-tag list
//! (`[FIRE: 60]`), or a nested-effects block (`{ ... }`), it stops
//! collecting args and skips to the next newline (balancing `()` /
//! `[]` / `{}` along the way).
//!
//! Out of scope for this slice (handled in later Waves):
//! - `template` / `structure` top-level blocks (Waves 1.2 / 1.3)
//! - `deliver` / `recast` / `morph` body blocks (Wave 1.4)
//! - Modifier slot capture into the `EffectStmt` AST (Wave 1.5)
//! - Lowering of the new headers / `passive` blocks
//!   (`crates/dsl_compiler/src/ability_lower.rs` errors with
//!   `HeaderNotImplemented` / `PassiveBlockNotImplemented` for them
//!   pending Wave 2+ engine schema changes).
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
        other => {
            return Err(ParseErr::at(
                Span::new(name_span_start, name_span_start + other.len()),
                format!(
                    "unknown hint `{other}` — expected one of: damage, defense, crowd_control, utility, heal, economic"
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
    loop {
        // Skip ONLY inline whitespace — a newline ends the statement.
        skip_inline_ws_only(c);
        if c.eof() || c.starts_with_char('\n') || c.starts_with_char('}') {
            break;
        }
        // Comments terminate the line.
        if c.starts_with("//") || c.starts_with_char('#') {
            // Consume to end of line.
            while let Some(ch) = c.peek_char() {
                if ch == '\n' {
                    break;
                }
                c.bump(ch.len_utf8());
            }
            break;
        }
        // Modifier-tail tokens — Wave 1.0 stops collecting args and
        // skips them. Power tags `[…]`, scoped fan-out `in <shape>`,
        // duration `for <dur>`, conditional `when <cond>`, probability
        // `chance <p>`, stacking, scaling `+ N% stat`, nested
        // `{ … }` — all deferred.
        if c.starts_with_char('[')
            || c.starts_with_char('{')
            || c.starts_with_char('+')
            || starts_with_keyword(c, "in")
            || starts_with_keyword(c, "for")
            || starts_with_keyword(c, "when")
            || starts_with_keyword(c, "chance")
            || starts_with_keyword(c, "stacking")
        {
            skip_modifier_tail(c);
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
    Ok(EffectStmt { verb, args, span: Span::new(start, c.pos) })
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

/// Skip the trailing modifier portion of an effect statement to the next
/// newline (or to the closing `}` of the ability body), balancing `()`
/// / `[]` / `{}` along the way. Wave 1.5 will replace this with a real
/// modifier parser.
fn skip_modifier_tail(c: &mut Cursor) {
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;
    while let Some(ch) = c.peek_char() {
        match ch {
            '\n' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => break,
            '}' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                // Ability body closer; do NOT consume.
                break;
            }
            '(' => {
                paren_depth += 1;
                c.bump(1);
            }
            ')' => {
                paren_depth = (paren_depth - 1).max(0);
                c.bump(1);
            }
            '[' => {
                bracket_depth += 1;
                c.bump(1);
            }
            ']' => {
                bracket_depth = (bracket_depth - 1).max(0);
                c.bump(1);
            }
            '{' => {
                brace_depth += 1;
                c.bump(1);
            }
            '}' => {
                brace_depth -= 1;
                c.bump(1);
                if brace_depth <= 0 {
                    brace_depth = 0;
                }
            }
            '"' => {
                // Skip string literal (don't try to parse, just advance
                // past matching quote, honouring backslash escape).
                c.bump(1);
                while let Some(ch) = c.peek_char() {
                    if ch == '\\' {
                        c.bump(1);
                        if let Some(esc) = c.peek_char() {
                            c.bump(esc.len_utf8());
                        }
                        continue;
                    }
                    if ch == '"' {
                        c.bump(1);
                        break;
                    }
                    c.bump(ch.len_utf8());
                }
            }
            _ => {
                c.bump(ch.len_utf8());
            }
        }
    }
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
