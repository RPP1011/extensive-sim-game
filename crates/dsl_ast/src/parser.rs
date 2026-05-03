//! Recursive-descent parser for the World Sim DSL.
//!
//! The parser walks a `Cursor` over the source, producing AST nodes with
//! byte-spans into the original input. Errors carry a context chain (outer
//! rule → inner rule) and a rendered caret pointer.
//!
//! Grammar coverage: `entity`, `event`, `view`, `query`, `physics`, `mask`,
//! `verb`, `scoring`, `invariant`, `probe`, `metric`. See `docs/dsl/spec.md`
//! §2 for the canonical grammar.

use crate::ast::*;
use crate::error::ParseError;
use crate::tokens::{is_ident_cont, is_ident_start, unicode_op_ascii, Cursor};

type PResult<T> = Result<T, ParseErr>;

/// Internal error type; converted to `ParseError` at the top level with the
/// full source attached for rendering.
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

pub fn parse_program(source: &str) -> Result<Program, ParseError> {
    let mut c = Cursor::new(source);
    c.skip_ws();
    let mut decls = Vec::new();
    while !c.eof() {
        match decl(&mut c) {
            Ok(mut d) => {
                if let Err(e) = absorb_trailing_annotations(&mut c, &mut d) {
                    return Err(ParseError::new(source, e.span, e.context, e.message));
                }
                decls.push(d);
            }
            Err(e) => {
                return Err(ParseError::new(source, e.span, e.context, e.message));
            }
        }
        c.skip_ws();
    }
    Ok(Program { decls })
}

/// Gather `@annotation`s that follow a just-parsed decl. Trailing annotations
/// must sit on the same source line as the decl's closing token; an `@` that
/// only appears after a newline is treated as the *next* decl's leading
/// annotation. This matches `event Foo { ... } @replayable` (trailing) versus
/// `event Foo { ... }\n@replayable\nevent Bar { ... }` (leading on `Bar`).
fn absorb_trailing_annotations(c: &mut Cursor, d: &mut Decl) -> PResult<()> {
    loop {
        // Same-line check: skip only spaces/tabs (NOT newlines or comments).
        let save = c.pos;
        skip_inline_ws(c);
        if !c.starts_with_char('@') {
            c.pos = save;
            return Ok(());
        }
        let ann = parse_annotation(c)?;
        if let Some(anns) = decl_annotations_mut(d) {
            anns.push(ann);
            let span = decl_span_mut(d);
            span.end = c.pos;
        }
        // Keep typed flags on decls in sync with the annotation vec. Today
        // only `PhysicsDecl::cpu_only` derives from a trailing annotation.
        if let Decl::Physics(p) = d {
            p.cpu_only = p.annotations.iter().any(|a| a.name == "cpu_only");
        }
    }
}

/// Skip spaces and tabs but NOT newlines — used by trailing-annotation
/// disambiguation to keep the trailing run constrained to the source line.
fn skip_inline_ws(c: &mut Cursor) {
    while let Some(ch) = c.peek_char() {
        if ch == ' ' || ch == '\t' {
            c.bump(ch.len_utf8());
        } else {
            break;
        }
    }
}

fn decl_annotations_mut(d: &mut Decl) -> Option<&mut Vec<Annotation>> {
    Some(match d {
        Decl::Entity(x) => &mut x.annotations,
        Decl::Event(x) => &mut x.annotations,
        Decl::EventTag(x) => &mut x.annotations,
        Decl::Enum(x) => &mut x.annotations,
        Decl::View(x) => &mut x.annotations,
        Decl::Physics(x) => &mut x.annotations,
        Decl::Mask(x) => &mut x.annotations,
        Decl::Verb(x) => &mut x.annotations,
        Decl::Scoring(x) => &mut x.annotations,
        Decl::Invariant(x) => &mut x.annotations,
        Decl::Probe(x) => &mut x.annotations,
        Decl::Metric(x) => &mut x.annotations,
        Decl::Config(x) => &mut x.annotations,
        Decl::SpatialQuery(x) => &mut x.annotations,
        // `query` does not currently accept annotations on the decl; trailing
        // `@`s after a `query` will fall through to the orphan-annotation
        // error path on the next iteration.
        Decl::Query(_) => return None,
    })
}

fn decl_span_mut(d: &mut Decl) -> &mut Span {
    match d {
        Decl::Entity(x) => &mut x.span,
        Decl::Event(x) => &mut x.span,
        Decl::EventTag(x) => &mut x.span,
        Decl::Enum(x) => &mut x.span,
        Decl::View(x) => &mut x.span,
        Decl::Physics(x) => &mut x.span,
        Decl::Mask(x) => &mut x.span,
        Decl::Verb(x) => &mut x.span,
        Decl::Scoring(x) => &mut x.span,
        Decl::Invariant(x) => &mut x.span,
        Decl::Probe(x) => &mut x.span,
        Decl::Metric(x) => &mut x.span,
        Decl::Config(x) => &mut x.span,
        Decl::Query(x) => &mut x.span,
        Decl::SpatialQuery(x) => &mut x.span,
    }
}

// ---------------------------------------------------------------------------
// Top-level declaration dispatch
// ---------------------------------------------------------------------------

fn decl(c: &mut Cursor) -> PResult<Decl> {
    c.skip_ws();
    let annotations = parse_annotations(c)?;
    c.skip_ws();
    let start = c.pos;
    let kw = peek_ident(c);
    match kw.as_deref() {
        Some("entity") => entity_decl(c, annotations, start).map(Decl::Entity),
        Some("event_tag") => event_tag_decl(c, annotations, start).map(Decl::EventTag),
        Some("event") => event_decl(c, annotations, start).map(Decl::Event),
        Some("enum") => enum_decl(c, annotations, start).map(Decl::Enum),
        Some("view") => view_decl(c, annotations, start).map(Decl::View),
        Some("query") => query_decl(c, annotations, start).map(Decl::Query),
        Some("physics") => physics_decl(c, annotations, start).map(Decl::Physics),
        Some("mask") => mask_decl(c, annotations, start).map(Decl::Mask),
        Some("verb") => verb_decl(c, annotations, start).map(Decl::Verb),
        Some("scoring") => scoring_decl(c, annotations, start).map(Decl::Scoring),
        Some("invariant") => invariant_decl(c, annotations, start).map(Decl::Invariant),
        Some("probe") => probe_decl(c, annotations, start).map(Decl::Probe),
        Some("metric") => metric_block(c, annotations, start).map(Decl::Metric),
        Some("config") => config_decl(c, annotations, start).map(Decl::Config),
        Some("spatial_query") => {
            spatial_query_decl(c, annotations, start).map(Decl::SpatialQuery)
        }
        _ => Err(ParseErr::at(
            here(c),
            format!(
                "expected top-level declaration (entity, event, event_tag, enum, view, query, physics, mask, verb, scoring, invariant, probe, metric, config, spatial_query); got `{}`",
                peek_word_for_error(c)
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// Annotations
// ---------------------------------------------------------------------------

fn parse_annotations(c: &mut Cursor) -> PResult<Vec<Annotation>> {
    let mut anns = Vec::new();
    loop {
        c.skip_ws();
        if !c.starts_with_char('@') {
            break;
        }
        anns.push(parse_annotation(c)?);
    }
    Ok(anns)
}

fn parse_annotation(c: &mut Cursor) -> PResult<Annotation> {
    let start = c.pos;
    expect_char(c, '@').map_err(|e| e.with_context("parsing annotation"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing annotation name"))?;
    let mut args = Vec::new();
    let after_name = c.pos;
    c.skip_ws();
    if c.starts_with_char('(') {
        c.bump(1);
        loop {
            c.skip_ws();
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            let arg = parse_annotation_arg(c)?;
            args.push(arg);
            c.skip_ws();
            // `@name(a | b)` rejected; spec prose uses `|` as "author picks one," not grammar.
            if c.starts_with_char('|') {
                return Err(ParseErr::at(
                    here(c),
                    "annotation arguments do not support | alternation; use a single value or comma-separated args",
                ));
            }
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `)` in annotation args"));
        }
    } else {
        // No args — roll back the lookahead whitespace so the cursor sits
        // exactly after the annotation name. Trailing-annotation gathering
        // relies on this to detect end-of-line accurately.
        c.pos = after_name;
    }
    let span = Span::new(start, c.pos);
    Ok(Annotation { name, args, span })
}

fn parse_annotation_arg(c: &mut Cursor) -> PResult<AnnotationArg> {
    let start = c.pos;
    // Peek: `<ident> =` or `<ident>(...)`?
    let save = c.pos;
    let key = if let Some(name) = peek_ident(c) {
        // Save a checkpoint, tentatively consume, check for =
        let after_name = c.pos + name.len();
        let mut look = Cursor { src: c.src, pos: after_name };
        look.skip_ws();
        if look.starts_with_char('=') {
            c.pos = after_name;
            c.skip_ws();
            c.bump(1); // consume `=`
            Some(name)
        } else {
            c.pos = save;
            None
        }
    } else {
        None
    };
    c.skip_ws();
    let value = parse_annotation_value(c)?;
    Ok(AnnotationArg { key, value, span: Span::new(start, c.pos) })
}

fn parse_annotation_value(c: &mut Cursor) -> PResult<AnnotationValue> {
    c.skip_ws();
    // Comparator form: `>= Medium`, `< 0.5`, `== X`.
    if let Some(op) = try_comparator(c) {
        c.skip_ws();
        let inner = parse_annotation_value(c)?;
        return Ok(AnnotationValue::Comparator { op: op.to_string(), value: Box::new(inner) });
    }
    if c.starts_with_char('[') {
        c.bump(1);
        let mut items = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char(']') {
                c.bump(1);
                break;
            }
            items.push(parse_annotation_value(c)?);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char(']') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `]` in annotation list"));
        }
        return Ok(AnnotationValue::List(items));
    }
    if c.starts_with_char('"') {
        return Ok(AnnotationValue::String(string_lit(c)?));
    }
    if peek_number(c) {
        let (n, is_float) = number_literal(c)?;
        return Ok(if is_float {
            AnnotationValue::Float(n as f64)
        } else {
            AnnotationValue::Int(n as i64)
        });
    }
    if let Some(name) = peek_ident(c) {
        c.bump(name.len());
        // `per_entity_topk(K = 8)` — an ident followed by `(` opens a
        // Call form. The inner args reuse `parse_annotation_arg` so
        // `key = value` and bare positional args parse identically to
        // the top-level annotation grammar.
        let save = c.pos;
        c.skip_ws();
        if c.starts_with_char('(') {
            c.bump(1);
            let mut args = Vec::new();
            loop {
                c.skip_ws();
                if c.starts_with_char(')') {
                    c.bump(1);
                    break;
                }
                args.push(parse_annotation_arg(c)?);
                c.skip_ws();
                if c.starts_with_char(',') {
                    c.bump(1);
                    continue;
                }
                if c.starts_with_char(')') {
                    c.bump(1);
                    break;
                }
                return Err(ParseErr::at(here(c), "expected `,` or `)` in annotation call args"));
            }
            return Ok(AnnotationValue::Call { name, args });
        }
        // No `(` — rewind past the whitespace we consumed to keep the
        // cursor exactly after the bare ident. The caller's trailing-
        // annotation lookahead relies on the end-of-value position being
        // the end of the ident, not the end of any whitespace after it.
        c.pos = save;
        return Ok(AnnotationValue::Ident(name));
    }
    Err(ParseErr::at(here(c), "expected annotation value"))
}

fn try_comparator(c: &mut Cursor) -> Option<&'static str> {
    for op in [">=", "<=", "==", "!=", ">", "<"] {
        if c.starts_with(op) {
            c.bump(op.len());
            return Some(op);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 2.1 entity
// ---------------------------------------------------------------------------

fn entity_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<EntityDecl> {
    expect_keyword(c, "entity").map_err(|e| e.with_context("parsing `entity` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing entity name"))?;
    c.skip_ws();
    expect_char(c, ':')
        .map_err(|e| e.with_context("parsing entity root kind (expected `:`)"))?;
    c.skip_ws();
    let root_ident = ident(c).map_err(|e| e.with_context("parsing entity root kind"))?;
    let root = match root_ident.as_str() {
        "Agent" => EntityRoot::Agent,
        "Item" => EntityRoot::Item,
        "Group" => EntityRoot::Group,
        other => {
            return Err(ParseErr::at(
                here_back(c, other.len()),
                format!("expected `Agent`, `Item`, or `Group`; got `{other}`"),
            )
            .with_context("parsing entity root kind"))
        }
    };
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing entity body (expected `{`)"))?;
    let fields = parse_entity_fields(c)?;
    c.skip_ws();
    expect_char(c, '}')
        .map_err(|e| e.with_context("parsing entity body (expected `}`)"))?;
    Ok(EntityDecl { annotations, name, root, fields, span: Span::new(start, c.pos) })
}

fn parse_entity_fields(c: &mut Cursor) -> PResult<Vec<EntityField>> {
    let mut fields = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            break;
        }
        fields.push(parse_entity_field(c)?);
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    Ok(fields)
}

fn parse_entity_field(c: &mut Cursor) -> PResult<EntityField> {
    let start = c.pos;
    let annotations = parse_annotations(c)?;
    c.skip_ws();
    let name = ident(c).map_err(|e| e.with_context("parsing field name"))?;
    c.skip_ws();
    expect_char(c, ':')
        .map_err(|e| e.with_context("parsing field (expected `:` after name)"))?;
    c.skip_ws();
    let value = parse_entity_field_value(c)?;
    Ok(EntityField { annotations, name, value, span: Span::new(start, c.pos) })
}

fn parse_entity_field_value(c: &mut Cursor) -> PResult<EntityFieldValue> {
    c.skip_ws();
    // List literal
    if c.starts_with_char('[') {
        c.bump(1);
        let mut items = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char(']') {
                c.bump(1);
                break;
            }
            // Handle trailing `...` pseudo-tokens ("[HungerDriveKind, ...]")
            if c.starts_with("...") {
                c.bump(3);
                c.skip_ws();
                if c.starts_with_char(']') {
                    c.bump(1);
                    break;
                }
                continue;
            }
            items.push(parse_expr(c)?);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char(']') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `]` in list"));
        }
        return Ok(EntityFieldValue::List(items));
    }
    // Struct literal? (typename followed by `{`)
    let save = c.pos;
    if let Some(name) = peek_ident(c) {
        let after = c.pos + name.len();
        let mut look = Cursor { src: c.src, pos: after };
        look.skip_ws();
        // capture optional type args: `Foo<A, B> { ... }`
        if look.starts_with_char('<') {
            // attempt to parse type args and then a `{`
            let mut la = Cursor { src: c.src, pos: after };
            let _ = type_ref(&mut la);
            la.skip_ws();
            if la.starts_with_char('{') {
                // re-parse as type_ref from original position
                let ty = type_ref(c)?;
                c.skip_ws();
                c.bump(1); // `{`
                let fields = parse_entity_fields(c)?;
                c.skip_ws();
                expect_char(c, '}')
                    .map_err(|e| e.with_context("parsing struct literal `}`"))?;
                return Ok(EntityFieldValue::StructLiteral { ty, fields });
            }
        }
        if look.starts_with_char('{') {
            let ty = type_ref(c)?;
            c.skip_ws();
            c.bump(1); // `{`
            let fields = parse_entity_fields(c)?;
            c.skip_ws();
            expect_char(c, '}')
                .map_err(|e| e.with_context("parsing struct literal `}`"))?;
            return Ok(EntityFieldValue::StructLiteral { ty, fields });
        }
    }
    c.pos = save;
    // Try type_ref followed by a non-expression continuation.
    // Strategy: parse a type_ref tentatively; if what follows is `,` / `}` /
    // EOF we consumed a type; otherwise treat as an expression.
    let ck = c.clone();
    match type_ref(c) {
        Ok(ty) => {
            let mut la = c.clone();
            la.skip_ws();
            if la.eof() || la.starts_with_char(',') || la.starts_with_char('}') {
                return Ok(EntityFieldValue::Type(ty));
            }
            // Otherwise fall back to expr, rewinding.
            *c = ck;
        }
        Err(_) => {
            *c = ck;
        }
    }
    Ok(EntityFieldValue::Expr(parse_expr(c)?))
}

// ---------------------------------------------------------------------------
// 2.2 event
// ---------------------------------------------------------------------------

fn event_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<EventDecl> {
    expect_keyword(c, "event").map_err(|e| e.with_context("parsing `event` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing event name"))?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing event body (expected `{`)"))?;
    let fields = parse_field_decls(c)?;
    // Implicit tick: authors must not declare a `tick` field. Every emitted
    // Event variant receives `tick: u32` automatically (see emit_rust.rs).
    if let Some(f) = fields.iter().find(|f| f.name == "tick") {
        return Err(ParseErr::at(
            f.span,
            "tick is implicit on every event; remove this field.",
        ));
    }
    c.skip_ws();
    expect_char(c, '}').map_err(|e| e.with_context("parsing event body (expected `}`)"))?;
    Ok(EventDecl {
        annotations,
        name,
        fields,
        tags: Vec::new(),
        span: Span::new(start, c.pos),
    })
}

fn event_tag_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<EventTagDecl> {
    expect_keyword(c, "event_tag")
        .map_err(|e| e.with_context("parsing `event_tag` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing event_tag name"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing event_tag body (expected `{`)"))?;
    let fields = parse_field_decls(c)?;
    if let Some(f) = fields.iter().find(|f| f.name == "tick") {
        return Err(ParseErr::at(
            f.span,
            "tick is implicit on every event; remove this field from the tag.",
        ));
    }
    c.skip_ws();
    expect_char(c, '}')
        .map_err(|e| e.with_context("parsing event_tag body (expected `}`)"))?;
    Ok(EventTagDecl { annotations, name, fields, span: Span::new(start, c.pos) })
}

fn enum_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<EnumDecl> {
    expect_keyword(c, "enum").map_err(|e| e.with_context("parsing `enum` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing enum name"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing enum body (expected `{`)"))?;
    let mut variants = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let vstart = c.pos;
        let vname = ident(c).map_err(|e| e.with_context("parsing enum variant name"))?;
        variants.push(EnumVariant { name: vname, span: Span::new(vstart, c.pos) });
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
            continue;
        }
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        return Err(ParseErr::at(here(c), "expected `,` or `}` in enum body"));
    }
    Ok(EnumDecl { annotations, name, variants, span: Span::new(start, c.pos) })
}

fn parse_field_decls(c: &mut Cursor) -> PResult<Vec<FieldDecl>> {
    let mut fields = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            break;
        }
        let fstart = c.pos;
        let name = ident(c).map_err(|e| e.with_context("parsing field name"))?;
        c.skip_ws();
        expect_char(c, ':').map_err(|e| e.with_context("parsing field `:`"))?;
        c.skip_ws();
        let ty = type_ref(c)?;
        fields.push(FieldDecl { name, ty, span: Span::new(fstart, c.pos) });
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    Ok(fields)
}

// ---------------------------------------------------------------------------
// 2.3 view / query
// ---------------------------------------------------------------------------

fn view_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<ViewDecl> {
    expect_keyword(c, "view").map_err(|e| e.with_context("parsing `view` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing view name"))?;
    let params = parse_params(c)?;
    c.skip_ws();
    expect_str(c, "->").map_err(|e| e.with_context("parsing view return-type arrow `->`"))?;
    c.skip_ws();
    let return_ty = type_ref(c)?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing view body `{`"))?;
    let body = parse_view_body(c)?;
    c.skip_ws();
    expect_char(c, '}').map_err(|e| e.with_context("parsing view body `}`"))?;
    Ok(ViewDecl { annotations, name, params, return_ty, body, span: Span::new(start, c.pos) })
}

/// Parse `spatial_query <name>(<params>) = <filter_expr>`.
///
/// Mirrors `verb_decl` (the closest sibling: also `name(params) =
/// <body>`); the body is a single expression (Bool — well_formed
/// gates it once lowered to CG). Phase 7 Task 4.
fn spatial_query_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<SpatialQueryDecl> {
    expect_keyword(c, "spatial_query")
        .map_err(|e| e.with_context("parsing `spatial_query` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing spatial_query name"))?;
    let params = parse_params(c)?;
    c.skip_ws();
    expect_char(c, '=').map_err(|e| e.with_context("parsing spatial_query `=`"))?;
    c.skip_ws();
    let filter = parse_expr(c)?;
    Ok(SpatialQueryDecl {
        annotations,
        name,
        params,
        filter,
        span: Span::new(start, c.pos),
    })
}

fn parse_view_body(c: &mut Cursor) -> PResult<ViewBody> {
    c.skip_ws();
    // Detect fold form via `initial:` keyword as first token.
    if c.starts_with("initial") {
        let after = c.pos + "initial".len();
        let mut look = Cursor { src: c.src, pos: after };
        look.skip_ws();
        if look.starts_with_char(':') {
            c.bump("initial".len());
            c.skip_ws();
            expect_char(c, ':')?;
            c.skip_ws();
            let initial = parse_expr(c)?;
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
            }
            let mut handlers = Vec::new();
            let mut clamp = None;
            loop {
                c.skip_ws();
                if c.starts_with_char('}') {
                    break;
                }
                if c.starts_with("clamp") {
                    let save = c.pos;
                    c.bump("clamp".len());
                    c.skip_ws();
                    expect_char(c, ':').map_err(|e| e.with_context("parsing `clamp:`"))?;
                    c.skip_ws();
                    expect_char(c, '[').map_err(|e| e.with_context("parsing clamp bounds `[`"))?;
                    c.skip_ws();
                    let lo = parse_expr(c)?;
                    c.skip_ws();
                    expect_char(c, ',')
                        .map_err(|e| e.with_context("parsing clamp bounds `,`"))?;
                    c.skip_ws();
                    let hi = parse_expr(c)?;
                    c.skip_ws();
                    expect_char(c, ']').map_err(|e| e.with_context("parsing clamp bounds `]`"))?;
                    c.skip_ws();
                    if c.starts_with_char(',') {
                        c.bump(1);
                    }
                    clamp = Some((lo, hi));
                    let _ = save;
                    continue;
                }
                if c.starts_with("on ") || c.starts_with("on\t") || c.starts_with("on\n") {
                    handlers.push(parse_fold_handler(c)?);
                    continue;
                }
                break;
            }
            return Ok(ViewBody::Fold { initial, handlers, clamp });
        }
    }
    let expr = parse_expr(c)?;
    Ok(ViewBody::Expr(expr))
}

fn parse_fold_handler(c: &mut Cursor) -> PResult<FoldHandler> {
    let start = c.pos;
    expect_keyword(c, "on").map_err(|e| e.with_context("parsing fold `on` handler"))?;
    c.skip_ws();
    let pattern = parse_event_pattern(c)?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing fold handler body `{`"))?;
    let body = parse_stmt_block_until_close(c)?;
    c.skip_ws();
    expect_char(c, '}').map_err(|e| e.with_context("parsing fold handler body `}`"))?;
    Ok(FoldHandler { pattern, body, span: Span::new(start, c.pos) })
}

fn query_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<QueryDecl> {
    expect_keyword(c, "query").map_err(|e| e.with_context("parsing `query` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing query name"))?;
    let params = parse_params(c)?;
    c.skip_ws();
    expect_str(c, "->").map_err(|e| e.with_context("parsing query return-type arrow `->`"))?;
    c.skip_ws();
    let return_ty = type_ref(c)?;
    c.skip_ws();
    let mut sort_by = None;
    if c.starts_with("sort_by") {
        c.bump("sort_by".len());
        c.skip_ws();
        sort_by = Some(parse_expr(c)?);
        c.skip_ws();
    }
    let mut limit = None;
    if c.starts_with("limit") {
        c.bump("limit".len());
        c.skip_ws();
        limit = Some(parse_expr(c)?);
        c.skip_ws();
    }
    let mut body = None;
    if c.starts_with_char('{') {
        c.bump(1);
        c.skip_ws();
        if !c.starts_with_char('}') {
            body = Some(parse_expr(c)?);
        }
        c.skip_ws();
        expect_char(c, '}').map_err(|e| e.with_context("parsing query body `}`"))?;
    }
    Ok(QueryDecl {
        annotations,
        name,
        params,
        return_ty,
        sort_by,
        limit,
        body,
        span: Span::new(start, c.pos),
    })
}

fn parse_params(c: &mut Cursor) -> PResult<Vec<Param>> {
    c.skip_ws();
    expect_char(c, '(').map_err(|e| e.with_context("parsing parameter list `(`"))?;
    let mut params = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char(')') {
            c.bump(1);
            break;
        }
        let pstart = c.pos;
        let name = ident(c).map_err(|e| e.with_context("parsing parameter name"))?;
        c.skip_ws();
        // Allow untyped `self` (spec §2.6 verb example). Any other untyped
        // parameter is still a hard error.
        let ty = if c.starts_with_char(':') {
            c.bump(1);
            c.skip_ws();
            type_ref(c)?
        } else if name == "self" {
            TypeRef { kind: TypeKind::Named("Self".to_string()), span: Span::new(pstart, c.pos) }
        } else {
            return Err(ParseErr::at(here(c), "expected `:` after parameter name")
                .with_context("parsing parameter `:`"));
        };
        params.push(Param { name, ty, span: Span::new(pstart, c.pos) });
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
            continue;
        }
        if c.starts_with_char(')') {
            c.bump(1);
            break;
        }
        return Err(ParseErr::at(here(c), "expected `,` or `)` in parameter list"));
    }
    Ok(params)
}

// ---------------------------------------------------------------------------
// 2.4 physics
// ---------------------------------------------------------------------------

fn physics_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<PhysicsDecl> {
    expect_keyword(c, "physics").map_err(|e| e.with_context("parsing `physics` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing physics name"))?;
    // Annotations may appear after the name: `physics foo @phase(event) { ... }`.
    c.skip_ws();
    let mut extra_ann = parse_annotations(c)?;
    let mut all = annotations;
    all.append(&mut extra_ann);
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing physics body `{`"))?;
    let mut handlers = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        handlers.push(parse_physics_handler(c)?);
    }
    let cpu_only = all.iter().any(|a| a.name == "cpu_only");
    Ok(PhysicsDecl {
        annotations: all,
        name,
        handlers,
        cpu_only,
        span: Span::new(start, c.pos),
    })
}

fn parse_physics_handler(c: &mut Cursor) -> PResult<PhysicsHandler> {
    let start = c.pos;
    expect_keyword(c, "on").map_err(|e| e.with_context("parsing physics handler `on`"))?;
    c.skip_ws();
    let pattern = if c.starts_with_char('@') {
        let pstart = c.pos;
        c.bump(1);
        let name = ident(c).map_err(|e| e.with_context("parsing physics tag name"))?;
        c.skip_ws();
        let mut bindings = Vec::new();
        if c.starts_with_char('{') {
            c.bump(1);
            loop {
                c.skip_ws();
                if c.starts_with_char('}') {
                    c.bump(1);
                    break;
                }
                bindings.push(parse_pattern_binding(c)?);
                c.skip_ws();
                if c.starts_with_char(',') {
                    c.bump(1);
                    continue;
                }
                if c.starts_with_char('}') {
                    c.bump(1);
                    break;
                }
                return Err(ParseErr::at(here(c), "expected `,` or `}` in tag pattern"));
            }
        }
        PhysicsPattern::Tag { name, bindings, span: Span::new(pstart, c.pos) }
    } else {
        PhysicsPattern::Kind(parse_event_pattern(c)?)
    };
    c.skip_ws();
    let mut where_clause = None;
    if c.starts_with("where") {
        c.bump("where".len());
        c.skip_ws();
        where_clause = Some(parse_expr(c)?);
        c.skip_ws();
    }
    expect_char(c, '{').map_err(|e| e.with_context("parsing physics handler body `{`"))?;
    let body = parse_stmt_block_until_close(c)?;
    c.skip_ws();
    expect_char(c, '}').map_err(|e| e.with_context("parsing physics handler body `}`"))?;
    Ok(PhysicsHandler { pattern, where_clause, body, span: Span::new(start, c.pos) })
}

// ---------------------------------------------------------------------------
// Event patterns (shared by physics + fold)
// ---------------------------------------------------------------------------

fn parse_event_pattern(c: &mut Cursor) -> PResult<EventPattern> {
    let start = c.pos;
    let name = ident(c).map_err(|e| e.with_context("parsing event pattern name"))?;
    c.skip_ws();
    let mut bindings = Vec::new();
    if c.starts_with_char('{') {
        c.bump(1);
        loop {
            c.skip_ws();
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            bindings.push(parse_pattern_binding(c)?);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `}` in event pattern"));
        }
    }
    Ok(EventPattern { name, bindings, span: Span::new(start, c.pos) })
}

fn parse_pattern_binding(c: &mut Cursor) -> PResult<PatternBinding> {
    let start = c.pos;
    let field = ident(c).map_err(|e| e.with_context("parsing pattern field name"))?;
    c.skip_ws();
    expect_char(c, ':').map_err(|e| e.with_context("parsing pattern `:`"))?;
    c.skip_ws();
    let value = parse_pattern_value(c)?;
    Ok(PatternBinding { field, value, span: Span::new(start, c.pos) })
}

fn parse_pattern_value(c: &mut Cursor) -> PResult<PatternValue> {
    c.skip_ws();
    if c.starts_with_char('_') {
        let after = c.pos + 1;
        let next = c.src[after..].chars().next();
        if next.map_or(true, |ch| !is_ident_cont(ch)) {
            c.bump(1);
            return Ok(PatternValue::Wildcard);
        }
    }
    // Try to parse: Ident, or Ident(...) (ctor), or Ident { ... } (struct
    // pattern over an enum variant), or a literal expression.
    let save = c.pos;
    if let Some(name) = peek_ident(c) {
        let after = c.pos + name.len();
        let mut look = Cursor { src: c.src, pos: after };
        look.skip_ws();
        if look.starts_with_char('(') {
            // ctor-wrap: `Agent(inner_bind)`.
            c.bump(name.len());
            c.skip_ws();
            c.bump(1); // `(`
            let mut inner = Vec::new();
            loop {
                c.skip_ws();
                if c.starts_with_char(')') {
                    c.bump(1);
                    break;
                }
                inner.push(parse_pattern_value(c)?);
                c.skip_ws();
                if c.starts_with_char(',') {
                    c.bump(1);
                    continue;
                }
                if c.starts_with_char(')') {
                    c.bump(1);
                    break;
                }
                return Err(ParseErr::at(here(c), "expected `,` or `)` in pattern ctor"));
            }
            return Ok(PatternValue::Ctor { name, inner });
        }
        // Struct-shaped pattern: `Damage { amount }` or
        // `Slow { duration_ticks, factor_q8: f }`. Only capitalized names
        // trigger this shape (lowercase names followed by `{` would be a
        // block/struct literal in an expression context, but in pattern
        // position we only accept the PascalCase enum-variant form).
        if look.starts_with_char('{')
            && name.chars().next().map_or(false, |c0| c0.is_ascii_uppercase())
        {
            c.bump(name.len());
            c.skip_ws();
            c.bump(1); // `{`
            let mut bindings = Vec::new();
            loop {
                c.skip_ws();
                if c.starts_with_char('}') {
                    c.bump(1);
                    break;
                }
                let bstart = c.pos;
                let field = ident(c).map_err(|e| e.with_context("parsing struct-pattern field name"))?;
                c.skip_ws();
                // Either `field` (shorthand bind) or `field: <inner-pattern>`.
                let value = if c.starts_with_char(':') {
                    c.bump(1);
                    c.skip_ws();
                    parse_pattern_value(c)?
                } else {
                    PatternValue::Bind(field.clone())
                };
                bindings.push(PatternBinding {
                    field,
                    value,
                    span: Span::new(bstart, c.pos),
                });
                c.skip_ws();
                if c.starts_with_char(',') {
                    c.bump(1);
                    continue;
                }
                if c.starts_with_char('}') {
                    c.bump(1);
                    break;
                }
                return Err(ParseErr::at(
                    here(c),
                    "expected `,` or `}` in struct-pattern",
                ));
            }
            return Ok(PatternValue::Struct { name, bindings });
        }
        // Bare ident: decide "simple bind" vs "expression" by looking at
        // what follows. If `,`, `}`, or `)`, treat as a bind.
        if look.eof()
            || look.starts_with_char(',')
            || look.starts_with_char('}')
            || look.starts_with_char(')')
        {
            c.bump(name.len());
            return Ok(PatternValue::Bind(name));
        }
    }
    c.pos = save;
    // Fall through to general expression.
    Ok(PatternValue::Expr(parse_expr(c)?))
}

// ---------------------------------------------------------------------------
// 2.5 mask
// ---------------------------------------------------------------------------

fn mask_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<MaskDecl> {
    expect_keyword(c, "mask").map_err(|e| e.with_context("parsing `mask` declaration"))?;
    let head = parse_action_head(c)?;
    c.skip_ws();
    // Optional `from <candidate_source_expr>` clause. Task 138 —
    // target-bound masks enumerate candidates from this source and
    // filter each through the `when` predicate. Self-masks (Hold,
    // Eat, …) omit `from` entirely.
    let candidate_source = if starts_with_keyword(c, "from") {
        expect_keyword(c, "from").map_err(|e| e.with_context("parsing mask `from`"))?;
        c.skip_ws();
        let expr = parse_expr(c)?;
        c.skip_ws();
        Some(expr)
    } else {
        None
    };
    expect_keyword(c, "when").map_err(|e| e.with_context("parsing mask `when`"))?;
    c.skip_ws();
    let predicate = parse_expr(c)?;
    Ok(MaskDecl { annotations, head, candidate_source, predicate, span: Span::new(start, c.pos) })
}

fn parse_action_head(c: &mut Cursor) -> PResult<ActionHead> {
    let start = c.pos;
    let name = ident(c).map_err(|e| e.with_context("parsing action head name"))?;
    c.skip_ws();
    if c.starts_with_char('(') {
        c.bump(1);
        let mut ids: Vec<(String, Option<TypeRef>)> = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            if c.starts_with_char('_') {
                let after = c.pos + 1;
                let next = c.src[after..].chars().next();
                if next.map_or(true, |ch| !is_ident_cont(ch)) {
                    c.bump(1);
                    ids.push(("_".to_string(), None));
                    c.skip_ws();
                    if c.starts_with_char(',') {
                        c.bump(1);
                        continue;
                    }
                    if c.starts_with_char(')') {
                        c.bump(1);
                        break;
                    }
                    return Err(ParseErr::at(here(c), "expected `,` or `)` in action head"));
                }
            }
            let name = ident(c)?;
            c.skip_ws();
            // Optional `: Type` annotation. Task 157 — lets `mask
            // Cast(ability: AbilityId)` type its head param without
            // forcing every existing `Attack(target)` / `MoveToward(target)`
            // to grow a `: AgentId` suffix.
            let ty = if c.starts_with_char(':') {
                c.bump(1);
                c.skip_ws();
                Some(type_ref(c).map_err(|e| e.with_context("parsing action head type annotation"))?)
            } else {
                None
            };
            ids.push((name, ty));
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `)` in action head"));
        }
        return Ok(ActionHead {
            name,
            shape: ActionHeadShape::Positional(ids),
            span: Span::new(start, c.pos),
        });
    }
    if c.starts_with_char('{') {
        c.bump(1);
        let mut bindings = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            bindings.push(parse_pattern_binding(c)?);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `}` in action head"));
        }
        return Ok(ActionHead {
            name,
            shape: ActionHeadShape::Named(bindings),
            span: Span::new(start, c.pos),
        });
    }
    Ok(ActionHead { name, shape: ActionHeadShape::None, span: Span::new(start, c.pos) })
}

// ---------------------------------------------------------------------------
// 2.6 verb
// ---------------------------------------------------------------------------

fn verb_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<VerbDecl> {
    expect_keyword(c, "verb").map_err(|e| e.with_context("parsing `verb` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing verb name"))?;
    let params = parse_params(c)?;
    c.skip_ws();
    expect_char(c, '=').map_err(|e| e.with_context("parsing verb `=`"))?;
    c.skip_ws();
    expect_keyword(c, "action").map_err(|e| e.with_context("parsing verb `action` keyword"))?;
    let action = parse_verb_action(c)?;
    c.skip_ws();
    let mut when = None;
    if c.starts_with("when") {
        c.bump("when".len());
        c.skip_ws();
        when = Some(parse_expr(c)?);
        c.skip_ws();
    }
    let mut emits = Vec::new();
    while c.starts_with("emit ") || c.starts_with("emit\t") || c.starts_with("emit\n") {
        emits.push(parse_emit_stmt(c)?);
        c.skip_ws();
    }
    let mut scoring = None;
    if c.starts_with("scoring") {
        c.bump("scoring".len());
        c.skip_ws();
        scoring = Some(parse_expr(c)?);
    } else if c.starts_with("score") {
        c.bump("score".len());
        c.skip_ws();
        scoring = Some(parse_expr(c)?);
    }
    Ok(VerbDecl { annotations, name, params, action, when, emits, scoring, span: Span::new(start, c.pos) })
}

fn parse_verb_action(c: &mut Cursor) -> PResult<VerbAction> {
    let start = c.pos;
    c.skip_ws();
    let name = ident(c).map_err(|e| e.with_context("parsing verb action name"))?;
    c.skip_ws();
    let mut args = Vec::new();
    if c.starts_with_char('(') {
        c.bump(1);
        loop {
            c.skip_ws();
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            args.push(parse_call_arg(c)?);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `)` in verb action args"));
        }
    }
    Ok(VerbAction { name, args, span: Span::new(start, c.pos) })
}

// ---------------------------------------------------------------------------
// 3.4 scoring
// ---------------------------------------------------------------------------

fn scoring_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<ScoringDecl> {
    expect_keyword(c, "scoring").map_err(|e| e.with_context("parsing `scoring` block"))?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing scoring block `{`"))?;
    let mut entries = Vec::new();
    let mut per_ability_rows = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        // `row <name> per_ability { ... }` — new per-ability scoring
        // row (GPU ability evaluation Phase 2). A leading `row`
        // keyword is the disambiguator vs. the legacy `Head = expr`
        // entry shape. Existing scoring files never use `row` as an
        // action head (action heads start with uppercase letters), so
        // the keyword check has no ambiguity.
        if starts_with_keyword(c, "row") {
            per_ability_rows.push(parse_per_ability_row(c)?);
            continue;
        }
        entries.push(parse_scoring_entry(c)?);
    }
    Ok(ScoringDecl {
        annotations,
        entries,
        per_ability_rows,
        span: Span::new(start, c.pos),
    })
}

fn parse_scoring_entry(c: &mut Cursor) -> PResult<ScoringEntry> {
    let start = c.pos;
    let head = parse_action_head(c)?;
    c.skip_ws();
    expect_char(c, '=').map_err(|e| e.with_context("parsing scoring entry `=`"))?;
    c.skip_ws();
    let expr = parse_expr(c)?;
    Ok(ScoringEntry { head, expr, span: Span::new(start, c.pos) })
}

/// Parse a `row <name> per_ability { guard: ..., score: ..., target: ... }`
/// row. Three clauses — `guard` and `target` are optional, `score` is
/// required. Order is not pinned; each clause is keyed by its identifier
/// and followed by `:`.
///
/// Added 2026-04-23 (GPU ability evaluation subsystem Phase 2). See
/// `docs/spec/engine.md §11`.
fn parse_per_ability_row(c: &mut Cursor) -> PResult<PerAbilityRow> {
    let start = c.pos;
    expect_keyword(c, "row").map_err(|e| e.with_context("parsing `row` keyword"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing per_ability row name"))?;
    c.skip_ws();
    expect_keyword(c, "per_ability")
        .map_err(|e| e.with_context("parsing `per_ability` row kind"))?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing per_ability row `{`"))?;
    let mut guard: Option<Expr> = None;
    let mut score: Option<Expr> = None;
    let mut target: Option<Expr> = None;
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let clause = ident(c).map_err(|e| e.with_context("parsing per_ability clause name"))?;
        c.skip_ws();
        expect_char(c, ':').map_err(|e| e.with_context("parsing per_ability clause `:`"))?;
        c.skip_ws();
        let expr = parse_expr(c)?;
        match clause.as_str() {
            "guard" => {
                if guard.is_some() {
                    return Err(ParseErr::at(
                        here(c),
                        "duplicate `guard:` clause in per_ability row",
                    ));
                }
                guard = Some(expr);
            }
            "score" => {
                if score.is_some() {
                    return Err(ParseErr::at(
                        here(c),
                        "duplicate `score:` clause in per_ability row",
                    ));
                }
                score = Some(expr);
            }
            "target" => {
                if target.is_some() {
                    return Err(ParseErr::at(
                        here(c),
                        "duplicate `target:` clause in per_ability row",
                    ));
                }
                target = Some(expr);
            }
            other => {
                return Err(ParseErr::at(
                    here(c),
                    format!(
                        "unknown per_ability clause `{other}`; \
                         expected `guard`, `score`, or `target`"
                    ),
                ));
            }
        }
        // Optional trailing comma between clauses.
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    let Some(score) = score else {
        return Err(ParseErr::at(
            Span::new(start, c.pos),
            "per_ability row must include a `score:` clause",
        ));
    };
    Ok(PerAbilityRow {
        name,
        guard,
        score,
        target,
        span: Span::new(start, c.pos),
    })
}

// ---------------------------------------------------------------------------
// 2.8 invariant
// ---------------------------------------------------------------------------

fn invariant_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<InvariantDecl> {
    expect_keyword(c, "invariant").map_err(|e| e.with_context("parsing `invariant` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing invariant name"))?;
    c.skip_ws();
    let scope = if c.starts_with_char('(') { parse_params(c)? } else { Vec::new() };
    c.skip_ws();
    expect_char(c, '@').map_err(|e| e.with_context("parsing invariant mode (expected `@`)"))?;
    let mode_ident = ident(c).map_err(|e| e.with_context("parsing invariant mode"))?;
    let mode = match mode_ident.as_str() {
        "static" => InvariantMode::Static,
        "runtime" => InvariantMode::Runtime,
        "debug_only" => InvariantMode::DebugOnly,
        other => {
            return Err(ParseErr::at(
                here_back(c, other.len()),
                format!("expected `@static`, `@runtime`, or `@debug_only`; got `@{other}`"),
            ))
        }
    };
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing invariant body `{`"))?;
    c.skip_ws();
    let predicate = parse_expr(c)?;
    c.skip_ws();
    expect_char(c, '}').map_err(|e| e.with_context("parsing invariant body `}`"))?;
    Ok(InvariantDecl { annotations, name, scope, mode, predicate, span: Span::new(start, c.pos) })
}

// ---------------------------------------------------------------------------
// 2.9 probe
// ---------------------------------------------------------------------------

fn probe_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<ProbeDecl> {
    expect_keyword(c, "probe").map_err(|e| e.with_context("parsing `probe` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing probe name"))?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing probe body `{`"))?;
    let mut scenario = None;
    let mut seed = None;
    let mut seeds = None;
    let mut ticks = None;
    let mut tolerance = None;
    let mut asserts = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let kw = ident(c).map_err(|e| e.with_context("parsing probe field name"))?;
        c.skip_ws();
        match kw.as_str() {
            "scenario" => {
                scenario = Some(string_lit(c)?);
            }
            "seed" => {
                let (n, _) = number_literal(c)?;
                seed = Some(n as u64);
            }
            "seeds" => {
                expect_char(c, '[')
                    .map_err(|e| e.with_context("parsing `seeds [...]`"))?;
                let mut out = Vec::new();
                loop {
                    c.skip_ws();
                    if c.starts_with_char(']') {
                        c.bump(1);
                        break;
                    }
                    let (n, _) = number_literal(c)?;
                    out.push(n as u64);
                    c.skip_ws();
                    if c.starts_with_char(',') {
                        c.bump(1);
                    }
                }
                seeds = Some(out);
            }
            "ticks" => {
                let (n, _) = number_literal(c)?;
                ticks = Some(n as u32);
            }
            "tolerance" => {
                let (n, _) = number_literal(c)?;
                tolerance = Some(n);
            }
            "assert" => {
                c.skip_ws();
                expect_char(c, '{')
                    .map_err(|e| e.with_context("parsing `assert {` block"))?;
                loop {
                    c.skip_ws();
                    if c.starts_with_char('}') {
                        c.bump(1);
                        break;
                    }
                    asserts.push(parse_assert_expr(c)?);
                    c.skip_ws();
                    if c.starts_with_char(',') {
                        c.bump(1);
                    }
                }
            }
            other => {
                return Err(ParseErr::at(
                    here_back(c, other.len()),
                    format!("unknown probe field `{other}`"),
                ))
            }
        }
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    Ok(ProbeDecl {
        annotations,
        name,
        scenario,
        seed,
        seeds,
        ticks,
        tolerance,
        asserts,
        span: Span::new(start, c.pos),
    })
}

fn parse_assert_expr(c: &mut Cursor) -> PResult<AssertExpr> {
    let start = c.pos;
    let head = ident(c).map_err(|e| e.with_context("parsing assert head (count|pr|mean)"))?;
    c.skip_ws();
    expect_char(c, '[').map_err(|e| e.with_context("parsing assert `[`"))?;
    c.skip_ws();
    let span_fn = |c: &mut Cursor, _op: &str, _value: &Expr| Span::new(start, c.pos);
    match head.as_str() {
        "count" => {
            let filter = parse_expr_until_pipe_or_close(c)?;
            c.skip_ws();
            expect_char(c, ']').map_err(|e| e.with_context("parsing assert `]`"))?;
            c.skip_ws();
            let op = expect_comparator(c)?;
            c.skip_ws();
            let value = parse_expr(c)?;
            let s = span_fn(c, &op, &value);
            Ok(AssertExpr::Count { filter, op, value, span: s })
        }
        "pr" => {
            let action_filter = parse_expr_until_pipe_or_close(c)?;
            c.skip_ws();
            expect_char(c, '|').map_err(|e| e.with_context("parsing `pr[a | b]`"))?;
            c.skip_ws();
            let obs_filter = parse_expr_until_pipe_or_close(c)?;
            c.skip_ws();
            expect_char(c, ']').map_err(|e| e.with_context("parsing assert `]`"))?;
            c.skip_ws();
            let op = expect_comparator(c)?;
            c.skip_ws();
            let value = parse_expr(c)?;
            let s = span_fn(c, &op, &value);
            Ok(AssertExpr::Pr { action_filter, obs_filter, op, value, span: s })
        }
        "mean" => {
            let scalar = parse_expr_until_pipe_or_close(c)?;
            c.skip_ws();
            expect_char(c, '|').map_err(|e| e.with_context("parsing `mean[e | filter]`"))?;
            c.skip_ws();
            let filter = parse_expr_until_pipe_or_close(c)?;
            c.skip_ws();
            expect_char(c, ']').map_err(|e| e.with_context("parsing assert `]`"))?;
            c.skip_ws();
            let op = expect_comparator(c)?;
            c.skip_ws();
            let value = parse_expr(c)?;
            let s = span_fn(c, &op, &value);
            Ok(AssertExpr::Mean { scalar, filter, op, value, span: s })
        }
        other => Err(ParseErr::at(
            here_back(c, other.len()),
            format!("expected `count`, `pr`, or `mean`; got `{other}`"),
        )),
    }
}

fn expect_comparator(c: &mut Cursor) -> PResult<String> {
    c.skip_ws();
    match try_comparator(c) {
        Some(op) => Ok(op.to_string()),
        None => Err(ParseErr::at(here(c), "expected comparator (>=, <=, ==, !=, >, <)")),
    }
}

/// Parse an expression stopping at the first un-nested `|` or `]`.
fn parse_expr_until_pipe_or_close(c: &mut Cursor) -> PResult<Expr> {
    parse_expr_bounded(c, |ck| ck.starts_with_char('|') || ck.starts_with_char(']'))
}

// ---------------------------------------------------------------------------
// 2.11 metric
// ---------------------------------------------------------------------------

fn metric_block(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<MetricBlock> {
    expect_keyword(c, "metric").map_err(|e| e.with_context("parsing `metric` block"))?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing metric block `{`"))?;
    let mut metrics = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        metrics.push(parse_metric_decl(c)?);
    }
    Ok(MetricBlock { annotations, metrics, span: Span::new(start, c.pos) })
}

fn parse_metric_decl(c: &mut Cursor) -> PResult<MetricDecl> {
    let start = c.pos;
    expect_keyword(c, "metric").map_err(|e| e.with_context("parsing `metric <name>`"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing metric name"))?;
    c.skip_ws();
    expect_char(c, '=').map_err(|e| e.with_context("parsing metric `=`"))?;
    c.skip_ws();
    // Parse the primary value expression, stopping at one of the clause
    // keywords or at the end of the metric.
    let value = parse_expr_bounded(c, |ck| {
        ck.starts_with("window")
            || ck.starts_with("emit_every")
            || ck.starts_with("conditioned_on")
            || ck.starts_with("alert")
            || ck.starts_with_char('}')
            || ck.eof()
            || (starts_with_keyword(ck, "metric"))
    })?;
    let mut window = None;
    let mut emit_every = None;
    let mut conditioned_on = None;
    let mut alert_when = None;
    loop {
        c.skip_ws();
        if c.starts_with("window") {
            c.bump("window".len());
            c.skip_ws();
            let (n, _) = number_literal(c)?;
            window = Some(n as u64);
            continue;
        }
        if c.starts_with("emit_every") {
            c.bump("emit_every".len());
            c.skip_ws();
            let (n, _) = number_literal(c)?;
            emit_every = Some(n as u64);
            continue;
        }
        if c.starts_with("conditioned_on") {
            c.bump("conditioned_on".len());
            c.skip_ws();
            conditioned_on = Some(parse_expr_bounded(c, |ck| {
                ck.starts_with("alert")
                    || ck.starts_with("window")
                    || ck.starts_with("emit_every")
                    || ck.starts_with_char('}')
                    || ck.eof()
                    || starts_with_keyword(ck, "metric")
            })?);
            continue;
        }
        if c.starts_with("alert") {
            c.bump("alert".len());
            c.skip_ws();
            expect_keyword(c, "when").map_err(|e| e.with_context("parsing `alert when ...`"))?;
            c.skip_ws();
            alert_when = Some(parse_expr_bounded(c, |ck| {
                ck.starts_with("window")
                    || ck.starts_with("emit_every")
                    || ck.starts_with("conditioned_on")
                    || ck.starts_with_char('}')
                    || ck.eof()
                    || starts_with_keyword(ck, "metric")
            })?);
            continue;
        }
        break;
    }
    Ok(MetricDecl {
        name,
        value,
        window,
        emit_every,
        conditioned_on,
        alert_when,
        span: Span::new(start, c.pos),
    })
}

// ---------------------------------------------------------------------------
// 2.12 config (balance tunables)
// ---------------------------------------------------------------------------

fn config_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<ConfigDecl> {
    expect_keyword(c, "config").map_err(|e| e.with_context("parsing `config` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing config block name"))?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing config body `{`"))?;
    let mut fields = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            break;
        }
        fields.push(parse_config_field(c)?);
        c.skip_ws();
        // Comma is optional between fields; newlines terminate by virtue of
        // `skip_ws` on the next iteration matching the `}`.
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    expect_char(c, '}').map_err(|e| e.with_context("parsing config body `}`"))?;
    Ok(ConfigDecl { annotations, name, fields, span: Span::new(start, c.pos) })
}

fn parse_config_field(c: &mut Cursor) -> PResult<ConfigField> {
    let fstart = c.pos;
    let name = ident(c).map_err(|e| e.with_context("parsing config field name"))?;
    c.skip_ws();
    expect_char(c, ':').map_err(|e| e.with_context("parsing config field `:`"))?;
    c.skip_ws();
    let ty = type_ref(c)?;
    c.skip_ws();
    expect_char(c, '=').map_err(|e| e.with_context("parsing config field `=` for default value"))?;
    c.skip_ws();
    let default = parse_config_default(c)?;
    Ok(ConfigField { name, ty, default, span: Span::new(fstart, c.pos) })
}

/// Parse the RHS of `<field>: <type> = <literal>`. Accepts one of:
///   - a decimal integer (possibly signed by a leading `-`)
///   - a float (same lexer as the rest of the grammar)
///   - `true` / `false`
///   - a double-quoted string
///
/// The type tag returned here is informational; lowering reconciles it with
/// the declared `<type>` so `u32 = 10` is accepted.
fn parse_config_default(c: &mut Cursor) -> PResult<ConfigDefault> {
    c.skip_ws();
    // String literal.
    if c.starts_with_char('"') {
        let s = string_lit(c)?;
        return Ok(ConfigDefault::String(s));
    }
    // Bool literal — look ahead for a bare `true` / `false` that isn't part
    // of a longer identifier.
    if starts_with_keyword(c, "true") {
        c.bump("true".len());
        return Ok(ConfigDefault::Bool(true));
    }
    if starts_with_keyword(c, "false") {
        c.bump("false".len());
        return Ok(ConfigDefault::Bool(false));
    }
    // Signed numeric.
    let negative = if c.starts_with_char('-') {
        c.bump(1);
        c.skip_ws();
        true
    } else {
        false
    };
    let (v, is_float) = number_literal(c)?;
    let v = if negative { -v } else { v };
    if is_float {
        Ok(ConfigDefault::Float(v))
    } else if negative {
        // A negative integer literal must fit in i64.
        Ok(ConfigDefault::Int(v as i64))
    } else {
        // Unsigned by default; the type declaration decides how it's emitted.
        Ok(ConfigDefault::Uint(v as u64))
    }
}

// ---------------------------------------------------------------------------
// Statements (physics body, fold body)
// ---------------------------------------------------------------------------

fn parse_stmt_block_until_close(c: &mut Cursor) -> PResult<Vec<Stmt>> {
    let mut stmts = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            break;
        }
        stmts.push(parse_stmt(c)?);
        c.skip_ws();
        if c.starts_with_char(';') {
            c.bump(1);
        }
    }
    Ok(stmts)
}

fn parse_stmt(c: &mut Cursor) -> PResult<Stmt> {
    c.skip_ws();
    let start = c.pos;
    if c.starts_with("let ") {
        c.bump("let".len());
        c.skip_ws();
        let name = ident(c)?;
        c.skip_ws();
        expect_char(c, '=').map_err(|e| e.with_context("parsing `let name = ...`"))?;
        c.skip_ws();
        let value = parse_expr(c)?;
        c.skip_ws();
        if c.starts_with_char(';') {
            c.bump(1);
        }
        return Ok(Stmt::Let { name, value, span: Span::new(start, c.pos) });
    }
    if c.starts_with("emit ") || c.starts_with("emit\t") || c.starts_with("emit\n") {
        return Ok(Stmt::Emit(parse_emit_stmt(c)?));
    }
    if c.starts_with("for ") {
        c.bump("for".len());
        c.skip_ws();
        let binder = ident(c)?;
        c.skip_ws();
        expect_keyword(c, "in").map_err(|e| e.with_context("parsing `for x in ...`"))?;
        c.skip_ws();
        let iter = parse_expr_bounded(c, |ck| {
            ck.starts_with_char('{') || starts_with_keyword(ck, "where")
        })?;
        c.skip_ws();
        let filter = if starts_with_keyword(c, "where") {
            c.bump("where".len());
            c.skip_ws();
            Some(parse_expr_bounded(c, |ck| ck.starts_with_char('{'))?)
        } else {
            None
        };
        c.skip_ws();
        expect_char(c, '{').map_err(|e| e.with_context("parsing `for` body `{`"))?;
        let body = parse_stmt_block_until_close(c)?;
        c.skip_ws();
        expect_char(c, '}').map_err(|e| e.with_context("parsing `for` body `}`"))?;
        return Ok(Stmt::For { binder, iter, filter, body, span: Span::new(start, c.pos) });
    }
    if c.starts_with("if ") {
        c.bump("if".len());
        c.skip_ws();
        let cond = parse_expr_bounded(c, |ck| ck.starts_with_char('{'))?;
        c.skip_ws();
        expect_char(c, '{').map_err(|e| e.with_context("parsing `if` body `{`"))?;
        let then_body = parse_stmt_block_until_close(c)?;
        c.skip_ws();
        expect_char(c, '}').map_err(|e| e.with_context("parsing `if` body `}`"))?;
        c.skip_ws();
        let mut else_body = None;
        if c.starts_with("else") {
            c.bump("else".len());
            c.skip_ws();
            expect_char(c, '{').map_err(|e| e.with_context("parsing `else` body `{`"))?;
            else_body = Some(parse_stmt_block_until_close(c)?);
            c.skip_ws();
            expect_char(c, '}').map_err(|e| e.with_context("parsing `else` body `}`"))?;
        }
        return Ok(Stmt::If { cond, then_body, else_body, span: Span::new(start, c.pos) });
    }
    if c.starts_with("match ") {
        c.bump("match".len());
        c.skip_ws();
        let scrutinee = parse_expr_bounded(c, |ck| ck.starts_with_char('{'))?;
        c.skip_ws();
        expect_char(c, '{').map_err(|e| e.with_context("parsing `match` body `{`"))?;
        let mut arms = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            let arm_start = c.pos;
            let pattern = parse_pattern_value(c)?;
            c.skip_ws();
            expect_str(c, "=>").map_err(|e| e.with_context("parsing match arm `=>`"))?;
            c.skip_ws();
            let body = if c.starts_with_char('{') {
                c.bump(1);
                let b = parse_stmt_block_until_close(c)?;
                c.skip_ws();
                expect_char(c, '}').map_err(|e| e.with_context("parsing match arm `}`"))?;
                b
            } else {
                vec![Stmt::Expr(parse_expr(c)?)]
            };
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
            }
            arms.push(MatchArm { pattern, body, span: Span::new(arm_start, c.pos) });
        }
        return Ok(Stmt::Match { scrutinee, arms, span: Span::new(start, c.pos) });
    }
    // `beliefs(<ident>).observe(<ident>) with { ... }` — statement form.
    // `beliefs(expr).about(expr).<field>` / `.confidence(expr)` / `.<view>(_)` — expression form.
    // Disambiguate via lookahead: scan for `.observe` after `beliefs(...)`.
    if starts_with_keyword(c, "beliefs") && is_belief_observe_stmt(c) {
        return Ok(Stmt::BeliefObserve(parse_belief_observe_stmt(c)?));
    }
    if c.starts_with("self") {
        // Check for `self += / -= / *=` operators.
        let save = c.pos;
        c.bump("self".len());
        c.skip_ws();
        for op in ["+=", "-=", "*=", "/=", "="] {
            if c.starts_with(op) {
                c.bump(op.len());
                c.skip_ws();
                let value = parse_expr(c)?;
                return Ok(Stmt::SelfUpdate { op: op.to_string(), value, span: Span::new(start, c.pos) });
            }
        }
        c.pos = save;
    }
    // Fallback: a bare expression statement.
    let e = parse_expr(c)?;
    Ok(Stmt::Expr(e))
}

fn parse_emit_stmt(c: &mut Cursor) -> PResult<EmitStmt> {
    let start = c.pos;
    expect_keyword(c, "emit").map_err(|e| e.with_context("parsing `emit` statement"))?;
    c.skip_ws();
    let event_name = ident(c).map_err(|e| e.with_context("parsing emit event name"))?;
    c.skip_ws();
    let mut fields = Vec::new();
    if c.starts_with_char('{') {
        c.bump(1);
        loop {
            c.skip_ws();
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            let fstart = c.pos;
            let name = ident(c).map_err(|e| e.with_context("parsing emit field name"))?;
            c.skip_ws();
            expect_char(c, ':').map_err(|e| e.with_context("parsing emit field `:`"))?;
            c.skip_ws();
            let value = parse_expr(c)?;
            fields.push(FieldInit { name, value, span: Span::new(fstart, c.pos) });
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `}` in emit body"));
        }
    }
    Ok(EmitStmt { event_name, fields, span: Span::new(start, c.pos) })
}

fn parse_belief_observe_stmt(c: &mut Cursor) -> PResult<BeliefObserveStmt> {
    let start = c.pos;
    expect_keyword(c, "beliefs")
        .map_err(|e| e.with_context("parsing `beliefs(...)` statement"))?;
    c.skip_ws();
    expect_char(c, '(').map_err(|e| e.with_context("parsing `beliefs(` open paren"))?;
    c.skip_ws();
    let observer =
        ident(c).map_err(|e| e.with_context("parsing `beliefs(observer` identifier"))?;
    c.skip_ws();
    expect_char(c, ')').map_err(|e| e.with_context("parsing `beliefs(...)` close paren"))?;
    c.skip_ws();
    expect_char(c, '.').map_err(|e| e.with_context("parsing `.` in `beliefs(...).observe`"))?;
    expect_keyword(c, "observe")
        .map_err(|e| e.with_context("parsing `.observe` method"))?;
    c.skip_ws();
    expect_char(c, '(').map_err(|e| e.with_context("parsing `observe(` open paren"))?;
    c.skip_ws();
    let target =
        ident(c).map_err(|e| e.with_context("parsing `observe(target` identifier"))?;
    c.skip_ws();
    expect_char(c, ')').map_err(|e| e.with_context("parsing `observe(...)` close paren"))?;
    c.skip_ws();
    expect_keyword(c, "with")
        .map_err(|e| e.with_context("parsing `with` keyword in belief mutation"))?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing `{` in belief mutation body"))?;
    let mut fields = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let fstart = c.pos;
        let name = ident(c).map_err(|e| e.with_context("parsing belief field name"))?;
        c.skip_ws();
        expect_char(c, ':').map_err(|e| e.with_context("parsing `:` after belief field name"))?;
        c.skip_ws();
        let value = parse_expr(c)?;
        fields.push(FieldInit { name, value, span: Span::new(fstart, c.pos) });
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
            continue;
        }
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        return Err(ParseErr::at(here(c), "expected `,` or `}` in belief mutation body"));
    }
    Ok(BeliefObserveStmt { observer, target, fields, span: Span::new(start, c.pos) })
}

/// Returns `true` when the cursor is positioned at `beliefs(...)` followed
/// (after optional whitespace) by `.observe` — indicating the statement form
/// `beliefs(o).observe(t) with { ... }`.  Returns `false` for the expression
/// read form `beliefs(o).about(t).<field>`, `.confidence(t)`, or `.<view>(_)`.
///
/// Uses a probe cursor so the real cursor is never advanced.
fn is_belief_observe_stmt(c: &Cursor) -> bool {
    let mut probe = Cursor { src: c.src, pos: c.pos };
    // Consume `beliefs`
    if !starts_with_keyword(&probe, "beliefs") {
        return false;
    }
    probe.bump("beliefs".len());
    probe.skip_ws();
    // Consume `(`
    if !probe.starts_with_char('(') {
        return false;
    }
    probe.bump(1);
    // Scan past the argument (balanced parens) to find the closing `)`.
    let mut depth = 1usize;
    while depth > 0 {
        if probe.eof() {
            return false;
        }
        match probe.peek_char() {
            Some('(') => { depth += 1; probe.bump(1); }
            Some(')') => {
                depth -= 1;
                probe.bump(1);
            }
            Some(ch) => { probe.bump(ch.len_utf8()); }
            None => return false,
        }
    }
    probe.skip_ws();
    // Must have a `.` next.
    if !probe.starts_with_char('.') {
        return false;
    }
    probe.bump(1);
    probe.skip_ws();
    // Observe-form iff the method name is `observe`.
    starts_with_keyword(&probe, "observe")
}

/// Parse a `beliefs(observer)` expression primary and the trailing tail:
/// - `.about(target).<field>`   → `ExprKind::BeliefsAccessor`
/// - `.confidence(target)`      → `ExprKind::BeliefsConfidence`
/// - `.<view_name>(_)`          → `ExprKind::BeliefsView`
fn parse_belief_expr(
    c: &mut Cursor,
    stop: &dyn Fn(&Cursor) -> bool,
) -> PResult<Expr> {
    let start = c.pos;
    expect_keyword(c, "beliefs")
        .map_err(|e| e.with_context("parsing `beliefs(...)` expression"))?;
    c.skip_ws();
    expect_char(c, '(').map_err(|e| e.with_context("parsing `beliefs(` open paren"))?;
    c.skip_ws();
    let observer = parse_expr(c)?;
    c.skip_ws();
    expect_char(c, ')').map_err(|e| e.with_context("parsing `beliefs(...)` close paren"))?;
    c.skip_ws();
    expect_char(c, '.').map_err(|e| e.with_context("parsing `.` in beliefs expression"))?;
    c.skip_ws();
    // Peek the method/field name that follows.
    let method = ident(c).map_err(|e| e.with_context("parsing beliefs method/field name"))?;
    c.skip_ws();
    match method.as_str() {
        "about" => {
            // `.about(target).<field>`
            expect_char(c, '(').map_err(|e| e.with_context("parsing `about(` open paren"))?;
            c.skip_ws();
            let target = parse_expr(c)?;
            c.skip_ws();
            expect_char(c, ')').map_err(|e| e.with_context("parsing `about(...)` close paren"))?;
            c.skip_ws();
            expect_char(c, '.').map_err(|e| e.with_context("parsing `.` after `about(...)`"))?;
            c.skip_ws();
            let field = ident(c).map_err(|e| e.with_context("parsing belief field name"))?;
            let span = Span::new(start, c.pos);
            let expr = Expr {
                kind: ExprKind::BeliefsAccessor {
                    observer: Box::new(observer),
                    target: Box::new(target),
                    field,
                },
                span,
            };
            parse_postfix(c, expr, stop)
        }
        "confidence" => {
            // `.confidence(target)`
            expect_char(c, '(').map_err(|e| e.with_context("parsing `confidence(` open paren"))?;
            c.skip_ws();
            let target = parse_expr(c)?;
            c.skip_ws();
            expect_char(c, ')').map_err(|e| e.with_context("parsing `confidence(...)` close paren"))?;
            let span = Span::new(start, c.pos);
            let expr = Expr {
                kind: ExprKind::BeliefsConfidence {
                    observer: Box::new(observer),
                    target: Box::new(target),
                },
                span,
            };
            parse_postfix(c, expr, stop)
        }
        view_name => {
            // `.<view_name>(_)` — aggregate view form.
            let view_name = view_name.to_string();
            expect_char(c, '(').map_err(|e| e.with_context("parsing beliefs view `(` open paren"))?;
            c.skip_ws();
            // Accept `_` as the wildcard argument (required by grammar).
            if c.starts_with_char('_') {
                c.bump(1);
            } else {
                return Err(ParseErr::at(
                    here(c),
                    "beliefs view argument must be `_`; e.g. `beliefs(o).all_known(_)`",
                ));
            }
            c.skip_ws();
            expect_char(c, ')').map_err(|e| e.with_context("parsing beliefs view `)` close paren"))?;
            let span = Span::new(start, c.pos);
            let expr = Expr {
                kind: ExprKind::BeliefsView {
                    observer: Box::new(observer),
                    view_name,
                },
                span,
            };
            parse_postfix(c, expr, stop)
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

fn type_ref(c: &mut Cursor) -> PResult<TypeRef> {
    let start = c.pos;
    c.skip_ws();
    if c.starts_with_char('[') {
        c.bump(1);
        c.skip_ws();
        let inner = type_ref(c)?;
        c.skip_ws();
        expect_char(c, ']').map_err(|e| e.with_context("parsing list type `]`"))?;
        return Ok(TypeRef { kind: TypeKind::List(Box::new(inner)), span: Span::new(start, c.pos) });
    }
    if c.starts_with_char('(') {
        c.bump(1);
        let mut elems = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            elems.push(type_ref(c)?);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `)` in tuple type"));
        }
        return Ok(TypeRef { kind: TypeKind::Tuple(elems), span: Span::new(start, c.pos) });
    }
    let name = ident(c).map_err(|e| e.with_context("parsing type name"))?;
    c.skip_ws();
    if c.starts_with_char('<') {
        c.bump(1);
        let mut args = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char('>') {
                c.bump(1);
                break;
            }
            if peek_number(c) {
                let (n, _) = number_literal(c)?;
                args.push(TypeArg::Const(n as i64));
            } else {
                args.push(TypeArg::Type(type_ref(c)?));
            }
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char('>') {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(here(c), "expected `,` or `>` in type arguments"));
        }
        if name == "Option" {
            if let [TypeArg::Type(t)] = args.as_slice() {
                return Ok(TypeRef { kind: TypeKind::Option(Box::new(t.clone())), span: Span::new(start, c.pos) });
            }
        }
        return Ok(TypeRef { kind: TypeKind::Generic { name, args }, span: Span::new(start, c.pos) });
    }
    Ok(TypeRef { kind: TypeKind::Named(name), span: Span::new(start, c.pos) })
}

// ---------------------------------------------------------------------------
// Expressions — Pratt-style parser for operator precedence.
// ---------------------------------------------------------------------------

fn parse_expr(c: &mut Cursor) -> PResult<Expr> {
    parse_expr_bounded(c, |_| false)
}

/// Parse an expression, stopping (before consuming) as soon as `stop` returns
/// true at a top-level (non-nested) position.
fn parse_expr_bounded(c: &mut Cursor, stop: impl Fn(&Cursor) -> bool) -> PResult<Expr> {
    parse_binary(c, 0, &stop)
}

fn parse_binary(c: &mut Cursor, min_prec: u8, stop: &dyn Fn(&Cursor) -> bool) -> PResult<Expr> {
    let mut lhs = parse_unary(c, stop)?;
    loop {
        c.skip_ws();
        if stop(c) {
            break;
        }
        // Normalize Unicode operators lazily.
        if let Some(ch) = c.peek_char() {
            if let Some(ascii) = unicode_op_ascii(ch) {
                // pretend we see ascii; we'll match on it below by bumping
                // the UTF-8 bytes of ch.
                let op_len_in_src = ch.len_utf8();
                if let Some(info) = bin_op_info(ascii) {
                    if info.prec < min_prec {
                        break;
                    }
                    c.bump(op_len_in_src);
                    c.skip_ws();
                    let rhs = parse_binary(c, info.prec + 1, stop)?;
                    let span = Span::new(lhs.span.start, rhs.span.end);
                    lhs = Expr { kind: ExprKind::Binary { op: info.op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, span };
                    continue;
                }
            }
        }
        // Try ASCII two-char or one-char ops, plus keyword ops.
        let ascii_op = if c.starts_with("&&") { Some("&&") }
            else if c.starts_with("||") { Some("||") }
            else if c.starts_with("==") { Some("==") }
            else if c.starts_with("!=") { Some("!=") }
            else if c.starts_with(">=") { Some(">=") }
            else if c.starts_with("<=") { Some("<=") }
            else if c.starts_with_char('<') { Some("<") }
            else if c.starts_with_char('>') { Some(">") }
            else if c.starts_with_char('+') { Some("+") }
            else if c.starts_with_char('-') { Some("-") }
            else if c.starts_with_char('*') { Some("*") }
            else if c.starts_with_char('/') && !c.starts_with("//") { Some("/") }
            else if c.starts_with_char('%') { Some("%") }
            else { None };
        if let Some(ascii) = ascii_op {
            if let Some(info) = bin_op_info(ascii) {
                if info.prec < min_prec {
                    break;
                }
                c.bump(ascii.len());
                c.skip_ws();
                let rhs = parse_binary(c, info.prec + 1, stop)?;
                let span = Span::new(lhs.span.start, rhs.span.end);
                lhs = Expr { kind: ExprKind::Binary { op: info.op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, span };
                continue;
            }
        }
        // Keyword ops: `in`, `contains`.
        if starts_with_keyword(c, "in") {
            // Treat `in` with the same precedence as comparison.
            const IN_PREC: u8 = 4;
            if IN_PREC < min_prec { break; }
            c.bump("in".len());
            c.skip_ws();
            let rhs = parse_binary(c, IN_PREC + 1, stop)?;
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expr { kind: ExprKind::In { item: Box::new(lhs), set: Box::new(rhs) }, span };
            continue;
        }
        if starts_with_keyword(c, "contains") {
            const C_PREC: u8 = 4;
            if C_PREC < min_prec { break; }
            c.bump("contains".len());
            c.skip_ws();
            let rhs = parse_binary(c, C_PREC + 1, stop)?;
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expr { kind: ExprKind::Contains { set: Box::new(lhs), item: Box::new(rhs) }, span };
            continue;
        }
        // `per_unit` — gradient modifier marker. Binds tighter than `+`/`-`
        // so `foo per_unit 0.4 + bar per_unit 0.2` parses as two sibling
        // modifier terms in the scoring sum. Right-associative to reject
        // the ambiguous `a per_unit b per_unit c` rather than silently
        // picking one side (the resolver rejects `per_unit` in the delta
        // slot, but we also avoid a surprising grammar parse here).
        if starts_with_keyword(c, "per_unit") {
            // Precedence between `+` (5) and `*` (6) — `expr * k per_unit d`
            // reads as `(expr * k) per_unit d`, which is the natural shape.
            const PER_UNIT_PREC: u8 = 5;
            if PER_UNIT_PREC < min_prec { break; }
            c.bump("per_unit".len());
            c.skip_ws();
            // Right-bind at PER_UNIT_PREC+1 so a nested `per_unit` on the
            // right-hand side is a grammar error rather than an accidental
            // chain.
            let rhs = parse_binary(c, PER_UNIT_PREC + 1, stop)?;
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Expr {
                kind: ExprKind::PerUnit { expr: Box::new(lhs), delta: Box::new(rhs) },
                span,
            };
            continue;
        }
        break;
    }
    Ok(lhs)
}

fn parse_unary(c: &mut Cursor, stop: &dyn Fn(&Cursor) -> bool) -> PResult<Expr> {
    c.skip_ws();
    let start = c.pos;
    if c.starts_with_char('!') && !c.starts_with("!=") {
        c.bump(1);
        let rhs = parse_unary(c, stop)?;
        let span = Span::new(start, rhs.span.end);
        return Ok(Expr { kind: ExprKind::Unary { op: UnOp::Not, rhs: Box::new(rhs) }, span });
    }
    if let Some(ch) = c.peek_char() {
        if ch == '¬' {
            c.bump(ch.len_utf8());
            let rhs = parse_unary(c, stop)?;
            let span = Span::new(start, rhs.span.end);
            return Ok(Expr { kind: ExprKind::Unary { op: UnOp::Not, rhs: Box::new(rhs) }, span });
        }
    }
    if c.starts_with_char('-') {
        // unary minus — but only if not a binary op (we know it's at start of
        // an atom since `parse_binary` already consumed any LHS).
        c.bump(1);
        let rhs = parse_unary(c, stop)?;
        let span = Span::new(start, rhs.span.end);
        return Ok(Expr { kind: ExprKind::Unary { op: UnOp::Neg, rhs: Box::new(rhs) }, span });
    }
    parse_atom(c, stop)
}

fn parse_atom(c: &mut Cursor, stop: &dyn Fn(&Cursor) -> bool) -> PResult<Expr> {
    c.skip_ws();
    let start = c.pos;
    if c.starts_with_char('(') {
        c.bump(1);
        c.skip_ws();
        // empty paren? not valid; at least one expr.
        let first = parse_expr(c)?;
        c.skip_ws();
        if c.starts_with_char(',') {
            let mut items = vec![first];
            while c.starts_with_char(',') {
                c.bump(1);
                c.skip_ws();
                if c.starts_with_char(')') {
                    break;
                }
                items.push(parse_expr(c)?);
                c.skip_ws();
            }
            expect_char(c, ')').map_err(|e| e.with_context("parsing tuple `)`"))?;
            return Ok(Expr { kind: ExprKind::Tuple(items), span: Span::new(start, c.pos) });
        }
        expect_char(c, ')').map_err(|e| e.with_context("parsing parenthesized expr `)`"))?;
        return parse_postfix(c, first, stop);
    }
    if c.starts_with_char('[') || c.starts_with_char('{') {
        let open = c.peek_char().unwrap();
        let close = if open == '[' { ']' } else { '}' };
        c.bump(1);
        let mut items = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char(close) {
                c.bump(1);
                break;
            }
            items.push(parse_expr(c)?);
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            if c.starts_with_char(close) {
                c.bump(1);
                break;
            }
            return Err(ParseErr::at(
                here(c),
                format!("expected `,` or `{close}` in literal"),
            ));
        }
        return Ok(Expr { kind: ExprKind::List(items), span: Span::new(start, c.pos) });
    }
    if c.starts_with_char('"') {
        let s = string_lit(c)?;
        return parse_postfix(c, Expr { kind: ExprKind::String(s), span: Span::new(start, c.pos) }, stop);
    }
    if peek_number(c) {
        let (n, is_float) = number_literal(c)?;
        let kind = if is_float { ExprKind::Float(n as f64) } else { ExprKind::Int(n as i64) };
        return parse_postfix(c, Expr { kind, span: Span::new(start, c.pos) }, stop);
    }
    // Keyword-driven atoms
    if starts_with_keyword(c, "true") { c.bump(4); return parse_postfix(c, Expr { kind: ExprKind::Bool(true), span: Span::new(start, c.pos) }, stop); }
    if starts_with_keyword(c, "false") { c.bump(5); return parse_postfix(c, Expr { kind: ExprKind::Bool(false), span: Span::new(start, c.pos) }, stop); }
    if starts_with_keyword(c, "forall") || starts_with_keyword(c, "exists") {
        let kind = if c.starts_with("forall") { QuantKind::Forall } else { QuantKind::Exists };
        c.bump(if kind == QuantKind::Forall { 6 } else { 6 });
        c.skip_ws();
        let binder = ident(c)?;
        c.skip_ws();
        expect_keyword(c, "in").map_err(|e| e.with_context("parsing quantifier `in`"))?;
        c.skip_ws();
        let iter = parse_expr_bounded(c, |ck| ck.starts_with_char(':'))?;
        c.skip_ws();
        expect_char(c, ':').map_err(|e| e.with_context("parsing quantifier `:`"))?;
        c.skip_ws();
        let body = parse_expr_bounded(c, stop)?;
        let span = Span::new(start, body.span.end);
        return Ok(Expr {
            kind: ExprKind::Quantifier { kind, binder, iter: Box::new(iter), body: Box::new(body) },
            span,
        });
    }
    for (kw, fk) in [
        ("count", FoldKind::Count),
        ("sum", FoldKind::Sum),
        ("max", FoldKind::Max),
        ("min", FoldKind::Min),
    ] {
        if starts_with_keyword(c, kw) {
            // Save the cursor BEFORE eating the keyword so we can fall back
            // to a regular function call if this turns out to be a pairwise
            // `min(a, b)` / `max(a, b)` call, not a fold expression.
            let pre_kw = c.pos;
            c.bump(kw.len());
            c.skip_ws();
            if c.starts_with_char('(') {
                c.bump(1);
                c.skip_ws();
                // Accept: `binder in iter where body` OR just `expr`.
                let save = c.pos;
                let try_bind = peek_ident(c);
                if let Some(bname) = try_bind.clone() {
                    let after = c.pos + bname.len();
                    let mut look = Cursor { src: c.src, pos: after };
                    look.skip_ws();
                    if look.starts_with("in ") || look.starts_with("in\t") || look.starts_with("in\n") {
                        c.bump(bname.len());
                        c.skip_ws();
                        c.bump(2); // `in`
                        c.skip_ws();
                        let iter = parse_expr_bounded(c, |ck| ck.starts_with(" where ") || ck.starts_with("where ") || ck.starts_with_char(')'))?;
                        c.skip_ws();
                        let body = if c.starts_with("where") {
                            c.bump("where".len());
                            c.skip_ws();
                            parse_expr_bounded(c, |ck| ck.starts_with_char(')'))?
                        } else {
                            Expr { kind: ExprKind::Bool(true), span: Span::new(c.pos, c.pos) }
                        };
                        c.skip_ws();
                        expect_char(c, ')').map_err(|e| e.with_context("parsing fold `)`"))?;
                        let span = Span::new(start, c.pos);
                        return Ok(Expr {
                            kind: ExprKind::Fold { kind: fk, binder: Some(bname), iter: Some(Box::new(iter)), body: Box::new(body) },
                            span,
                        });
                    }
                }
                c.pos = save;
                // Treat as single-expression fold argument. If parsing the
                // single expression succeeds but we don't see `)` next — most
                // commonly because we're really looking at a pairwise call
                // `min(a, b)` whose `,` the bounded-expr parse stopped at —
                // backtrack to before the keyword and let the generic
                // primary-expression parse pick it up as a `Call`. The
                // builtin name itself resolves to a `Min`/`Max`/`Count`/
                // `Sum` Builtin during name resolution, so the call form
                // ends up as `IrExpr::BuiltinCall(Builtin::Min, args)` and
                // emission dispatches on arity (see `docs/dsl/stdlib.md`).
                let mut probe = Cursor { src: c.src, pos: c.pos };
                if parse_expr_bounded(&mut probe, |ck| ck.starts_with_char(')')).is_ok() {
                    probe.skip_ws();
                    if probe.starts_with_char(')') {
                        // Real fold form: re-parse using the real cursor.
                        let body = parse_expr_bounded(c, |ck| ck.starts_with_char(')'))?;
                        c.skip_ws();
                        expect_char(c, ')').map_err(|e| e.with_context("parsing fold `)`"))?;
                        let span = Span::new(start, c.pos);
                        return Ok(Expr {
                            kind: ExprKind::Fold { kind: fk, binder: None, iter: None, body: Box::new(body) },
                            span,
                        });
                    }
                }
                // Backtrack to before the keyword. Fall through to normal
                // primary parsing — the keyword becomes an Ident and the
                // following `(...)` becomes a `Call`.
                c.pos = pre_kw;
                break;
            }
            // Keyword not followed by `(`: not a fold, fall through to the
            // ident path. Backtrack to before the keyword.
            c.pos = pre_kw;
            break;
        }
    }
    // `beliefs(observer).about(target).<field>` / `.confidence(target)` / `.<view>(_)`
    // expression read form (Plan ToM Task 8).  The statement form
    // `beliefs(o).observe(t) with { ... }` is handled in `parse_stmt` before
    // we ever reach here, so when `parse_atom` sees `beliefs` it is always the
    // expression form.
    if starts_with_keyword(c, "beliefs") {
        return parse_belief_expr(c, stop);
    }
    if starts_with_keyword(c, "if") {
        c.bump(2);
        c.skip_ws();
        let cond = parse_expr_bounded(c, |ck| ck.starts_with_char('{'))?;
        c.skip_ws();
        expect_char(c, '{').map_err(|e| e.with_context("parsing `if` expr `{`"))?;
        c.skip_ws();
        let then_expr = parse_expr(c)?;
        c.skip_ws();
        expect_char(c, '}').map_err(|e| e.with_context("parsing `if` expr `}`"))?;
        c.skip_ws();
        let mut else_expr = None;
        if c.starts_with("else") {
            c.bump(4);
            c.skip_ws();
            expect_char(c, '{').map_err(|e| e.with_context("parsing `else` expr `{`"))?;
            c.skip_ws();
            else_expr = Some(Box::new(parse_expr(c)?));
            c.skip_ws();
            expect_char(c, '}').map_err(|e| e.with_context("parsing `else` expr `}`"))?;
        }
        return Ok(Expr {
            kind: ExprKind::If { cond: Box::new(cond), then_expr: Box::new(then_expr), else_expr },
            span: Span::new(start, c.pos),
        });
    }
    if starts_with_keyword(c, "match") {
        c.bump(5);
        c.skip_ws();
        let scrutinee = parse_expr_bounded(c, |ck| ck.starts_with_char('{'))?;
        c.skip_ws();
        expect_char(c, '{').map_err(|e| e.with_context("parsing `match` expr `{`"))?;
        let mut arms = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            let astart = c.pos;
            let pattern = parse_pattern_value(c)?;
            c.skip_ws();
            expect_str(c, "=>").map_err(|e| e.with_context("parsing match arm `=>`"))?;
            c.skip_ws();
            let body = parse_expr_bounded(c, |ck| ck.starts_with_char(',') || ck.starts_with_char('}'))?;
            let end = c.pos;
            arms.push(MatchExprArm { pattern, body, span: Span::new(astart, end) });
            c.skip_ws();
            if c.starts_with_char(',') { c.bump(1); }
        }
        return Ok(Expr {
            kind: ExprKind::Match { scrutinee: Box::new(scrutinee), arms },
            span: Span::new(start, c.pos),
        });
    }
    // Identifier-based atom.
    let name = ident(c)?;
    // Path segments like `view::channel_range` (colon-colon) get flattened
    // into a single ident with `::` preserved.
    let mut name = name;
    while c.starts_with("::") {
        c.bump(2);
        let next = ident(c)?;
        name.push_str("::");
        name.push_str(&next);
    }
    // Ctor-style call with `(`, or record-style with `{`.
    c.skip_ws();
    if c.starts_with_char('(') {
        c.bump(1);
        let mut args = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char(')') {
                c.bump(1);
                break;
            }
            args.push(parse_expr(c)?);
            c.skip_ws();
            if c.starts_with_char(',') { c.bump(1); continue; }
            if c.starts_with_char(')') { c.bump(1); break; }
            return Err(ParseErr::at(here(c), "expected `,` or `)` in call"));
        }
        // Distinguish ctor (Capitalized single-arg or multi-arg constructor like `Agent(x)`)
        // from generic function call. For now, always emit `Call` so lowering can
        // reclassify — except when the callee name starts uppercase and is a single
        // arg, we emit `Ctor` so mask/pattern contexts get a uniform AST.
        let callee_span = Span::new(start, start + name.chars().count());
        let is_ctor = name.chars().next().map_or(false, |c0| c0.is_ascii_uppercase());
        let node = if is_ctor {
            Expr {
                kind: ExprKind::Ctor { name: name.clone(), args: args.into_iter().collect() },
                span: Span::new(start, c.pos),
            }
        } else {
            let callee = Expr { kind: ExprKind::Ident(name.clone()), span: callee_span };
            let call_args: Vec<CallArg> = args.into_iter().map(|e| CallArg { name: None, value: e.clone(), span: e.span }).collect();
            Expr {
                kind: ExprKind::Call(Box::new(callee), call_args),
                span: Span::new(start, c.pos),
            }
        };
        return parse_postfix(c, node, stop);
    }
    if c.starts_with_char('{') && !stop(c) {
        // Struct literal: `EventName { f: 1, g: 2 }`.
        // Only treat as struct-literal if the name is capitalized — prevents
        // confusion with block expressions. Skip if the caller's `stop`
        // predicate is already triggered on `{` (e.g. `for x in iter where p { .. }`
        // should not eat the for-body as a struct literal).
        if name.chars().next().map_or(false, |c0| c0.is_ascii_uppercase()) {
            c.bump(1);
            let mut fields = Vec::new();
            loop {
                c.skip_ws();
                if c.starts_with_char('}') {
                    c.bump(1);
                    break;
                }
                let fstart = c.pos;
                let fname = ident(c)?;
                c.skip_ws();
                expect_char(c, ':').map_err(|e| e.with_context("parsing struct-literal field `:`"))?;
                c.skip_ws();
                let value = parse_expr(c)?;
                fields.push(FieldInit { name: fname, value, span: Span::new(fstart, c.pos) });
                c.skip_ws();
                if c.starts_with_char(',') { c.bump(1); continue; }
                if c.starts_with_char('}') { c.bump(1); break; }
                return Err(ParseErr::at(here(c), "expected `,` or `}` in struct literal"));
            }
            let node = Expr { kind: ExprKind::Struct { name, fields }, span: Span::new(start, c.pos) };
            return parse_postfix(c, node, stop);
        }
    }
    let node = Expr { kind: ExprKind::Ident(name), span: Span::new(start, c.pos) };
    parse_postfix(c, node, stop)
}

fn parse_postfix(c: &mut Cursor, mut expr: Expr, stop: &dyn Fn(&Cursor) -> bool) -> PResult<Expr> {
    loop {
        c.skip_ws();
        if stop(c) {
            break;
        }
        if c.starts_with_char('.') {
            c.bump(1);
            let field = ident(c)?;
            let span = Span::new(expr.span.start, c.pos);
            expr = Expr { kind: ExprKind::Field(Box::new(expr), field), span };
            continue;
        }
        if c.starts_with_char('[') {
            c.bump(1);
            c.skip_ws();
            let idx = parse_expr(c)?;
            c.skip_ws();
            expect_char(c, ']').map_err(|e| e.with_context("parsing index `]`"))?;
            let span = Span::new(expr.span.start, c.pos);
            expr = Expr { kind: ExprKind::Index(Box::new(expr), Box::new(idx)), span };
            continue;
        }
        if c.starts_with_char('(') {
            c.bump(1);
            let mut args = Vec::new();
            loop {
                c.skip_ws();
                if c.starts_with_char(')') { c.bump(1); break; }
                args.push(parse_call_arg(c)?);
                c.skip_ws();
                if c.starts_with_char(',') { c.bump(1); continue; }
                if c.starts_with_char(')') { c.bump(1); break; }
                return Err(ParseErr::at(here(c), "expected `,` or `)` in call args"));
            }
            let span = Span::new(expr.span.start, c.pos);
            expr = Expr { kind: ExprKind::Call(Box::new(expr), args), span };
            continue;
        }
        break;
    }
    Ok(expr)
}

fn parse_call_arg(c: &mut Cursor) -> PResult<CallArg> {
    let start = c.pos;
    c.skip_ws();
    // Named arg: `ident: expr`.
    let save = c.pos;
    if let Some(name) = peek_ident(c) {
        let after = c.pos + name.len();
        let mut look = Cursor { src: c.src, pos: after };
        look.skip_ws();
        if look.starts_with_char(':') && !look.starts_with("::") {
            c.bump(name.len());
            c.skip_ws();
            c.bump(1); // `:`
            c.skip_ws();
            let value = parse_expr(c)?;
            return Ok(CallArg { name: Some(name), value, span: Span::new(start, c.pos) });
        }
    }
    c.pos = save;
    let value = parse_expr(c)?;
    Ok(CallArg { name: None, value, span: Span::new(start, c.pos) })
}

// ---------------------------------------------------------------------------
// Operator precedence table
// ---------------------------------------------------------------------------

struct BinOpInfo {
    op: BinOp,
    prec: u8,
}

fn bin_op_info(s: &str) -> Option<BinOpInfo> {
    Some(match s {
        "||" => BinOpInfo { op: BinOp::Or, prec: 1 },
        "&&" => BinOpInfo { op: BinOp::And, prec: 2 },
        "==" => BinOpInfo { op: BinOp::Eq, prec: 3 },
        "!=" => BinOpInfo { op: BinOp::NotEq, prec: 3 },
        "<" => BinOpInfo { op: BinOp::Lt, prec: 4 },
        "<=" => BinOpInfo { op: BinOp::LtEq, prec: 4 },
        ">" => BinOpInfo { op: BinOp::Gt, prec: 4 },
        ">=" => BinOpInfo { op: BinOp::GtEq, prec: 4 },
        "+" => BinOpInfo { op: BinOp::Add, prec: 5 },
        "-" => BinOpInfo { op: BinOp::Sub, prec: 5 },
        "*" => BinOpInfo { op: BinOp::Mul, prec: 6 },
        "/" => BinOpInfo { op: BinOp::Div, prec: 6 },
        "%" => BinOpInfo { op: BinOp::Mod, prec: 6 },
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Primitives: identifiers, numbers, strings
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

fn expect_str(c: &mut Cursor, s: &str) -> PResult<()> {
    c.skip_ws();
    if c.starts_with(s) {
        c.bump(s.len());
        Ok(())
    } else {
        Err(ParseErr::at(here(c), format!("expected `{s}`")))
    }
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
    // Minimal unescape for `\"` `\\` `\n`.
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
                Some(other) => { out.push('\\'); out.push(other); }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

fn peek_number(c: &Cursor) -> bool {
    c.peek_char().map_or(false, |ch| ch.is_ascii_digit())
}

/// Parse a numeric literal. Returns `(value, is_float)`.
fn number_literal(c: &mut Cursor) -> PResult<(f64, bool)> {
    c.skip_ws();
    let start = c.pos;
    while let Some(ch) = c.peek_char() {
        if ch.is_ascii_digit() || ch == '_' {
            c.bump(1);
        } else {
            break;
        }
    }
    let mut is_float = false;
    if c.starts_with_char('.') {
        // Not a method chain: require a digit after `.`.
        let after = c.pos + 1;
        let next = c.src[after..].chars().next();
        if let Some(n) = next {
            if n.is_ascii_digit() {
                c.bump(1);
                while let Some(ch) = c.peek_char() {
                    if ch.is_ascii_digit() || ch == '_' { c.bump(1); }
                    else { break; }
                }
                is_float = true;
            }
        }
    }
    // exponent
    if c.starts_with_char('e') || c.starts_with_char('E') {
        c.bump(1);
        if c.starts_with_char('+') || c.starts_with_char('-') { c.bump(1); }
        while let Some(ch) = c.peek_char() {
            if ch.is_ascii_digit() { c.bump(1); } else { break; }
        }
        is_float = true;
    }
    let raw = c.src[start..c.pos].replace('_', "");
    if raw.is_empty() {
        return Err(ParseErr::at(here(c), "expected numeric literal"));
    }
    let v = raw.parse::<f64>().map_err(|_| ParseErr::at(Span::new(start, c.pos), format!("invalid numeric literal `{raw}`")))?;
    Ok((v, is_float))
}

// ---------------------------------------------------------------------------
// Helpers for error reporting
// ---------------------------------------------------------------------------

fn here(c: &Cursor) -> Span {
    Span::new(c.pos, c.pos + c.peek_char().map_or(1, |ch| ch.len_utf8()))
}

fn here_back(c: &Cursor, n: usize) -> Span {
    let start = c.pos.saturating_sub(n);
    Span::new(start, c.pos)
}

fn peek_word_for_error(c: &Cursor) -> String {
    let rem = c.remaining();
    let end = rem.find(|ch: char| ch.is_whitespace() || ch == '{' || ch == '(' || ch == ';').unwrap_or(rem.len());
    rem[..end].to_string()
}
