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
use crate::tokens::{consume_int_suffix, is_ident_cont, is_ident_start, unicode_op_ascii, Cursor};

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

    // Parse leading `import <path>;` directives before any other declarations.
    let mut imports: Vec<crate::ast::Import> = Vec::new();
    loop {
        if c.eof() {
            break;
        }
        if !starts_with_keyword(&c, "import") {
            break;
        }
        // Consume `import` keyword.
        if let Err(e) = expect_keyword(&mut c, "import") {
            return Err(ParseError::new(source, e.span, e.context, e.message));
        }
        c.skip_ws();
        // Consume the bare path: all non-whitespace, non-`;` characters.
        let path_start = c.pos;
        loop {
            match c.peek_char() {
                None => break,
                Some(ch) if ch.is_whitespace() || ch == ';' => break,
                Some(ch) => c.bump(ch.len_utf8()),
            }
        }
        let path = c.src[path_start..c.pos].to_string();
        if path.is_empty() {
            return Err(ParseError::new(
                source,
                here(&c),
                vec!["parsing `import` directive".to_string()],
                "expected import path after `import`",
            ));
        }
        if !path.ends_with(".sim") {
            return Err(ParseError::new(
                source,
                here(&c),
                vec!["parsing `import` directive".to_string()],
                "import path must end in `.sim`",
            ));
        }
        // Consume `;`.
        if let Err(e) = expect_char(&mut c, ';') {
            return Err(ParseError::new(source, e.span, e.context, e.message));
        }
        imports.push(crate::ast::Import { path });
        c.skip_ws();
    }

    let mut decls = Vec::new();
    let mut terrain: Option<crate::terrain::TerrainBlock> = None;
    let mut controls: Option<crate::ast::ControlsDecl> = None;
    let mut render: Option<crate::ast::RenderDecl> = None;
    let mut ui: Option<crate::ast::UiDecl> = None;
    while !c.eof() {
        // Reject `import` that appears after a non-import top-level decl.
        if starts_with_keyword(&c, "import") {
            return Err(ParseError::new(
                source,
                here(&c),
                vec!["parsing top-level declarations".to_string()],
                "`import` statements must appear before any other top-level decl; found `import` after a decl",
            ));
        }
        // `terrain { ... }` is a singleton top-level block, not a Decl variant.
        // Handle it here before the general Decl dispatcher.
        if peek_ident(&c).as_deref() == Some("terrain") {
            match parse_terrain(&mut c) {
                Ok(t) => {
                    if terrain.is_some() {
                        return Err(ParseError::new(
                            source,
                            here(&c),
                            vec!["parsing `terrain` block".to_string()],
                            "duplicate `terrain` block; only one is allowed per file",
                        ));
                    }
                    terrain = Some(t);
                }
                Err(e) => {
                    return Err(ParseError::new(source, e.span, e.context, e.message));
                }
            }
            c.skip_ws();
            continue;
        }
        // Plan A — player-facing descriptor blocks are singleton top-level
        // blocks (like `terrain`), parsed onto the Program rather than into
        // a `Decl`. The resolver never sees them.
        match peek_ident(&c).as_deref() {
            Some("controls") => {
                let start = c.pos;
                match controls_decl(&mut c, start) {
                    Ok(d) => {
                        if controls.is_some() {
                            return Err(ParseError::new(source, here(&c),
                                vec!["parsing `controls` block".to_string()],
                                "duplicate `controls` block; only one is allowed per file"));
                        }
                        controls = Some(d);
                    }
                    Err(e) => return Err(ParseError::new(source, e.span, e.context, e.message)),
                }
                c.skip_ws();
                continue;
            }
            Some("render") => {
                let start = c.pos;
                match render_decl(&mut c, start) {
                    Ok(d) => {
                        if render.is_some() {
                            return Err(ParseError::new(source, here(&c),
                                vec!["parsing `render` block".to_string()],
                                "duplicate `render` block; only one is allowed per file"));
                        }
                        render = Some(d);
                    }
                    Err(e) => return Err(ParseError::new(source, e.span, e.context, e.message)),
                }
                c.skip_ws();
                continue;
            }
            Some("ui") => {
                let start = c.pos;
                match ui_decl(&mut c, start) {
                    Ok(d) => {
                        if ui.is_some() {
                            return Err(ParseError::new(source, here(&c),
                                vec!["parsing `ui` block".to_string()],
                                "duplicate `ui` block; only one is allowed per file"));
                        }
                        ui = Some(d);
                    }
                    Err(e) => return Err(ParseError::new(source, e.span, e.context, e.message)),
                }
                c.skip_ws();
                continue;
            }
            _ => {}
        }
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
    Ok(Program { imports, imports_resolved: vec![], decls, terrain, controls, render, ui })
}

/// Parse a single expression from a free-floating source string. Used
/// by downstream validators (e.g. dsl_compiler's when-condition
/// re-parse #143) that captured an expression as raw text and need to
/// turn it back into an `Expr` for syntactic / semantic checking.
///
/// The expression must consume the entire input — trailing junk is
/// rejected with a ParseError so a half-consumed predicate like
/// `target.hp < 30 then` doesn't silently lower as just
/// `target.hp < 30`.
pub fn parse_expression(source: &str) -> Result<Expr, ParseError> {
    let mut c = Cursor::new(source);
    c.skip_ws();
    let expr = match parse_expr(&mut c) {
        Ok(e) => e,
        Err(e) => return Err(ParseError::new(source, e.span, e.context, e.message)),
    };
    c.skip_ws();
    if !c.eof() {
        let here = c.pos;
        return Err(ParseError::new(
            source,
            crate::ast::Span::new(here, source.len()),
            vec!["parsing standalone expression".to_string()],
            "trailing input after expression".to_string(),
        ));
    }
    Ok(expr)
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
        Decl::Init(x) => &mut x.annotations,
        Decl::Debug(x) => &mut x.annotations,
        Decl::AgentField(x) => &mut x.annotations,
        Decl::SpatialQuery(x) => &mut x.annotations,
        Decl::Belief(x) => &mut x.annotations,
        Decl::Table(x) => &mut x.annotations,
        Decl::RegionKind(x) => &mut x.annotations,
        Decl::RegionIndices(x) => &mut x.annotations,
        Decl::Index(x) => &mut x.annotations,
        // `query` does not currently accept annotations on the decl; trailing
        // `@`s after a `query` will fall through to the orphan-annotation
        // error path on the next iteration.
        Decl::Query(_) => return None,
        // apply-form: annotations not yet surfaced in parser; fall through.
        Decl::PhysicsApply(x) => &mut x.annotations,
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
        Decl::Init(x) => &mut x.span,
        Decl::Debug(x) => &mut x.span,
        Decl::AgentField(x) => &mut x.span,
        Decl::Belief(x) => &mut x.span,
        Decl::Table(x) => &mut x.span,
        Decl::RegionKind(x) => &mut x.span,
        Decl::RegionIndices(x) => &mut x.span,
        Decl::Index(x) => &mut x.span,
        Decl::PhysicsApply(x) => &mut x.span,
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
        // Plan I — belief declaration. Mirrors view shape (signature
        // + fold-body) plus `merge from <agent>: <op>` clauses
        // interleaved with the fold handlers in the body.
        Some("belief") => belief_decl(c, annotations, start).map(Decl::Belief),
        Some("query") => query_decl(c, annotations, start).map(Decl::Query),
        Some("physics") => physics_any_decl(c, annotations, start),
        Some("mask") => mask_decl(c, annotations, start).map(Decl::Mask),
        Some("verb") => verb_decl(c, annotations, start).map(Decl::Verb),
        Some("scoring") => scoring_decl(c, annotations, start).map(Decl::Scoring),
        Some("invariant") => invariant_decl(c, annotations, start).map(Decl::Invariant),
        Some("probe") => probe_decl(c, annotations, start).map(Decl::Probe),
        Some("metric") => metric_block(c, annotations, start).map(Decl::Metric),
        Some("config") => config_decl(c, annotations, start).map(Decl::Config),
        Some("init") => init_decl(c, annotations, start).map(Decl::Init),
        Some("debug") => debug_decl(c, annotations, start).map(Decl::Debug),
        Some("field") => agent_field_decl(c, annotations, start).map(Decl::AgentField),
        Some("table") => table_decl(c, annotations, start).map(Decl::Table),
        Some("region_kind") => region_kind_decl(c, annotations, start).map(Decl::RegionKind),
        Some("region_indices") => region_indices_decl(c, annotations, start).map(Decl::RegionIndices),
        Some("index") => index_decl(c, annotations, start).map(Decl::Index),
        Some("spatial_query") => {
            spatial_query_decl(c, annotations, start).map(Decl::SpatialQuery)
        }
        _ => Err(ParseErr::at(
            here(c),
            format!(
                "expected top-level declaration (entity, event, event_tag, enum, view, query, physics, mask, verb, scoring, invariant, probe, metric, config, init, debug, field, spatial_query); got `{}`",
                peek_word_for_error(c)
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// 2.18 field (Gap plague_city#P-A — custom per-agent SoA column registry)
//
// Grammar:
//   field <name>: <type>
//
// Where `<type>` is one of `u32`, `f32`, `bool`, `vec3`. Trailing semicolon
// optional. Multiple `field` decls allowed per file; each registers a
// new column the rest of the .sim source can read via `self.<name>` /
// write via `agents.set_<name>(target, value)`. See `AgentFieldDecl`.
// ---------------------------------------------------------------------------

fn agent_field_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<AgentFieldDecl> {
    expect_keyword(c, "field")
        .map_err(|e| e.with_context("parsing `field` declaration"))?;
    c.skip_ws();
    let name = ident(c).map_err(|e| e.with_context("parsing field name"))?;
    c.skip_ws();
    expect_char(c, ':')
        .map_err(|e| e.with_context("parsing field decl (expected `:` after name)"))?;
    c.skip_ws();
    let ty_name = ident(c).map_err(|e| e.with_context("parsing field type"))?;
    c.skip_ws();
    // Optional trailing semicolon for visual symmetry with `let`
    // statements; not required.
    if c.starts_with_char(';') {
        c.bump(1);
    }
    Ok(AgentFieldDecl {
        annotations,
        name,
        ty_name,
        span: Span::new(start, c.pos),
    })
}

// ---------------------------------------------------------------------------
// 2.18b table — top-level static lookup table
//
// Grammar:
//   table <name>: <element_ty>[<length>] = [<v1>, <v2>, ..., <vN>]
//
// Where `<element_ty>` is `u32` (first cut) and `<length>` must equal
// the initializer's element count (resolver-validated). Trailing
// semicolon optional. Read in physics/view bodies as
// `tables.<name>(<idx_expr>)`.
// ---------------------------------------------------------------------------

fn table_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<TableDecl> {
    expect_keyword(c, "table")
        .map_err(|e| e.with_context("parsing `table` declaration"))?;
    c.skip_ws();
    let name = ident(c).map_err(|e| e.with_context("parsing table name"))?;
    c.skip_ws();
    expect_char(c, ':')
        .map_err(|e| e.with_context("parsing table decl (expected `:` after name)"))?;
    c.skip_ws();
    let element_ty_name = ident(c)
        .map_err(|e| e.with_context("parsing table element type"))?;
    c.skip_ws();
    expect_char(c, '[')
        .map_err(|e| e.with_context("parsing table decl (expected `[<length>]`)"))?;
    c.skip_ws();
    let (length_f, is_float) = number_literal(c)?;
    if is_float || length_f < 1.0 || length_f > (u32::MAX as f64) {
        return Err(ParseErr::at(
            here(c),
            format!(
                "table length must be a positive integer ≤ u32::MAX; got {length_f}"
            ),
        ));
    }
    let length = length_f as u32;
    c.skip_ws();
    expect_char(c, ']')
        .map_err(|e| e.with_context("parsing table decl (expected `]` after length)"))?;
    c.skip_ws();
    expect_char(c, '=')
        .map_err(|e| e.with_context("parsing table decl (expected `=` before initializer)"))?;
    c.skip_ws();
    expect_char(c, '[')
        .map_err(|e| e.with_context("parsing table initializer (expected `[`)"))?;
    let mut values: Vec<i64> = Vec::with_capacity(length as usize);
    loop {
        c.skip_ws();
        if c.starts_with_char(']') {
            c.bump(1);
            break;
        }
        // Optional leading minus for signed values (future-friendly;
        // u32-only first cut rejects negatives at resolve time).
        let negate = if c.starts_with_char('-') {
            c.bump(1);
            true
        } else {
            false
        };
        let (n, is_f) = number_literal(c)?;
        if is_f {
            return Err(ParseErr::at(
                here(c),
                "table initializer must be integer literals only (first cut)",
            ));
        }
        let mut v = n as i64;
        if negate {
            v = -v;
        }
        values.push(v);
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    c.skip_ws();
    // Optional trailing semicolon.
    if c.starts_with_char(';') {
        c.bump(1);
    }
    Ok(TableDecl {
        annotations,
        name,
        element_ty_name,
        length,
        values,
        span: Span::new(start, c.pos),
    })
}

// ---------------------------------------------------------------------------
// 2.18c region_kind / region_indices (voxel-region-indices spec §6.1.2)
//
// Grammar:
//   region_kind <Name> { max_active = <N> }
//   region_indices <Name> { <IndexKind>, <IndexKind>, ... }
//
// Pairs by `<Name>` — every `region_indices` body's kind name must
// match a declared `region_kind`. The resolver enforces the cross-
// decl link. Index-kind names defer to Phase 2's `index` decl
// resolution.
// ---------------------------------------------------------------------------

fn region_kind_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<RegionKindDecl> {
    expect_keyword(c, "region_kind")
        .map_err(|e| e.with_context("parsing `region_kind` declaration"))?;
    c.skip_ws();
    let name = ident(c).map_err(|e| e.with_context("parsing region_kind name"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing region_kind body (expected `{`)"))?;
    c.skip_ws();
    expect_keyword(c, "max_active")
        .map_err(|e| e.with_context("parsing region_kind body (expected `max_active = N`)"))?;
    c.skip_ws();
    expect_char(c, '=')
        .map_err(|e| e.with_context("parsing region_kind body (expected `=` after `max_active`)"))?;
    c.skip_ws();
    let (n, is_float) = number_literal(c)?;
    if is_float || n < 1.0 || n > (u32::MAX as f64) {
        return Err(ParseErr::at(
            here(c),
            format!("`max_active` must be a positive integer ≤ u32::MAX; got {n}"),
        ));
    }
    let max_active = n as u32;
    c.skip_ws();
    // Optional trailing comma — visual symmetry with config field
    // syntax (`name: type = N,`).
    if c.starts_with_char(',') {
        c.bump(1);
        c.skip_ws();
    }
    expect_char(c, '}')
        .map_err(|e| e.with_context("parsing region_kind body (expected `}`)"))?;
    Ok(RegionKindDecl {
        annotations,
        name,
        max_active,
        span: Span::new(start, c.pos),
    })
}

fn region_indices_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<RegionIndicesDecl> {
    expect_keyword(c, "region_indices")
        .map_err(|e| e.with_context("parsing `region_indices` declaration"))?;
    c.skip_ws();
    let name = ident(c).map_err(|e| e.with_context("parsing region_indices kind name"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing region_indices body (expected `{`)"))?;
    let mut index_kinds: Vec<String> = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let kind = ident(c)
            .map_err(|e| e.with_context("parsing index-kind name in region_indices body"))?;
        index_kinds.push(kind);
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    Ok(RegionIndicesDecl {
        annotations,
        name,
        index_kinds,
        span: Span::new(start, c.pos),
    })
}

// ---------------------------------------------------------------------------
// 2.18d index (voxel-region-indices spec §7.2) — Phase 2a
//
// Grammar:
//   index <name>(region: VoxelRegion) -> <Output> {
//       storage: <shape>(<args...>),
//       cost_class: <Cheap|Medium|Heavy>,
//       rebuild_on: chunk_epoch_advance(region.<field>) | manual,
//       build { <raw body> }
//   }
//
// Phase 2a captures the decl SHELL — body is preserved as raw
// text (between matched braces), Phase 2b parses it into an
// expression tree.
// ---------------------------------------------------------------------------

fn index_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<IndexDecl> {
    expect_keyword(c, "index")
        .map_err(|e| e.with_context("parsing `index` declaration"))?;
    c.skip_ws();
    let name = ident(c).map_err(|e| e.with_context("parsing index name"))?;
    c.skip_ws();
    expect_char(c, '(')
        .map_err(|e| e.with_context("parsing index params (expected `(`)"))?;
    c.skip_ws();
    let region_param_name = ident(c)
        .map_err(|e| e.with_context("parsing index region-param name"))?;
    c.skip_ws();
    expect_char(c, ':')
        .map_err(|e| e.with_context("parsing index params (expected `:` after region param)"))?;
    c.skip_ws();
    let ty = ident(c).map_err(|e| e.with_context("parsing index region-param type"))?;
    if ty != "VoxelRegion" {
        return Err(ParseErr::at(
            here(c),
            format!("index region param type must be `VoxelRegion`; got `{ty}`"),
        ));
    }
    c.skip_ws();
    expect_char(c, ')')
        .map_err(|e| e.with_context("parsing index params (expected `)`)"))?;
    c.skip_ws();
    expect_char(c, '-')
        .map_err(|e| e.with_context("parsing index return arrow (expected `->`)"))?;
    expect_char(c, '>')
        .map_err(|e| e.with_context("parsing index return arrow (expected `->`)"))?;
    c.skip_ws();
    let output_type_name = ident(c)
        .map_err(|e| e.with_context("parsing index return type"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing index body (expected `{`)"))?;
    c.skip_ws();

    // Field order is enforced: storage → cost_class → rebuild_on →
    // build. Single occurrence each. Keeps the grammar predictable
    // and the resolver simple; a future cut could relax to any
    // order if real fixtures want it.
    let storage = parse_index_storage_clause(c)?;
    c.skip_ws();
    let cost_class = parse_index_cost_class_clause(c)?;
    c.skip_ws();
    let rebuild_on = parse_index_rebuild_trigger_clause(c)?;
    c.skip_ws();
    let (build_body, build_body_ast) = parse_index_build_clause(c)?;
    c.skip_ws();
    expect_char(c, '}')
        .map_err(|e| e.with_context("parsing index body (expected `}` after build)"))?;

    Ok(IndexDecl {
        annotations,
        name,
        region_param_name,
        output_type_name,
        storage,
        cost_class,
        rebuild_on,
        build_body,
        build_body_ast,
        span: Span::new(start, c.pos),
    })
}

fn parse_index_storage_clause(c: &mut Cursor) -> PResult<IndexStorageShape> {
    expect_keyword(c, "storage")
        .map_err(|e| e.with_context("parsing index storage clause (expected `storage:`)"))?;
    c.skip_ws();
    expect_char(c, ':')
        .map_err(|e| e.with_context("parsing index storage clause (expected `:`)"))?;
    c.skip_ws();
    let shape_name = ident(c).map_err(|e| e.with_context("parsing index storage shape name"))?;
    c.skip_ws();
    expect_char(c, '(')
        .map_err(|e| e.with_context("parsing index storage shape args (expected `(`)"))?;
    let kv: Vec<(String, u32)> = parse_named_u32_args(c)?;
    c.skip_ws();
    expect_char(c, ')')
        .map_err(|e| e.with_context("parsing index storage shape args (expected `)`)"))?;
    // Optional trailing comma.
    c.skip_ws();
    if c.starts_with_char(',') {
        c.bump(1);
    }
    let lookup = |k: &str| -> PResult<u32> {
        kv.iter()
            .find(|(name, _)| name == k)
            .map(|(_, v)| *v)
            .ok_or_else(|| {
                ParseErr::at(
                    here(c),
                    format!("index storage shape missing arg `{k}`"),
                )
            })
    };
    let shape = match shape_name.as_str() {
        "per_cell_2d" => IndexStorageShape::PerCell2d {
            max_cells: lookup("max_cells")?,
            bytes_per_cell: lookup("bytes_per_cell")?,
        },
        "per_cell_3d" => IndexStorageShape::PerCell3d {
            max_cells: lookup("max_cells")?,
            bytes_per_cell: lookup("bytes_per_cell")?,
        },
        "bitset_pairs" => IndexStorageShape::BitsetPairs {
            max_cells: lookup("max_cells")?,
        },
        "mesh_buffer" => IndexStorageShape::MeshBuffer {
            max_vertices: lookup("max_vertices")?,
            max_indices: lookup("max_indices")?,
        },
        "sparse_grid" => IndexStorageShape::SparseGrid {
            max_cells: lookup("max_cells")?,
            bytes_per_cell: lookup("bytes_per_cell")?,
        },
        other => {
            return Err(ParseErr::at(
                here(c),
                format!(
                    "unknown index storage shape `{other}` (expected per_cell_2d / per_cell_3d / bitset_pairs / mesh_buffer / sparse_grid)"
                ),
            ));
        }
    };
    Ok(shape)
}

fn parse_named_u32_args(c: &mut Cursor) -> PResult<Vec<(String, u32)>> {
    let mut out: Vec<(String, u32)> = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char(')') {
            break;
        }
        let key = ident(c).map_err(|e| e.with_context("parsing arg name"))?;
        c.skip_ws();
        expect_char(c, '=')
            .map_err(|e| e.with_context("parsing arg (expected `=`)"))?;
        c.skip_ws();
        let (n, is_float) = number_literal(c)?;
        if is_float || n < 0.0 || n > (u32::MAX as f64) {
            return Err(ParseErr::at(
                here(c),
                format!("arg `{key}` must be a non-negative integer ≤ u32::MAX; got {n}"),
            ));
        }
        out.push((key, n as u32));
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    Ok(out)
}

fn parse_index_cost_class_clause(c: &mut Cursor) -> PResult<IndexCostClass> {
    expect_keyword(c, "cost_class")
        .map_err(|e| e.with_context("parsing index cost_class clause (expected `cost_class:`)"))?;
    c.skip_ws();
    expect_char(c, ':')
        .map_err(|e| e.with_context("parsing index cost_class clause (expected `:`)"))?;
    c.skip_ws();
    let cls = ident(c).map_err(|e| e.with_context("parsing index cost class"))?;
    c.skip_ws();
    if c.starts_with_char(',') {
        c.bump(1);
    }
    match cls.as_str() {
        "Cheap" => Ok(IndexCostClass::Cheap),
        "Medium" => Ok(IndexCostClass::Medium),
        "Heavy" => Ok(IndexCostClass::Heavy),
        other => Err(ParseErr::at(
            here(c),
            format!("unknown cost class `{other}` (expected Cheap / Medium / Heavy)"),
        )),
    }
}

fn parse_index_rebuild_trigger_clause(c: &mut Cursor) -> PResult<IndexRebuildTrigger> {
    expect_keyword(c, "rebuild_on")
        .map_err(|e| e.with_context("parsing index rebuild_on clause (expected `rebuild_on:`)"))?;
    c.skip_ws();
    expect_char(c, ':')
        .map_err(|e| e.with_context("parsing index rebuild_on clause (expected `:`)"))?;
    c.skip_ws();
    let trigger = ident(c).map_err(|e| e.with_context("parsing rebuild trigger name"))?;
    let result = match trigger.as_str() {
        "chunk_epoch_advance" => {
            c.skip_ws();
            expect_char(c, '(')
                .map_err(|e| e.with_context("parsing `chunk_epoch_advance(region.<field>)` (expected `(`)"))?;
            c.skip_ws();
            // Expect `region.<field>` — the `region` token must match
            // the index decl's region-param name; not validated at
            // parse time (deferred to resolve).
            let head = ident(c)
                .map_err(|e| e.with_context("parsing rebuild trigger arg (expected `region.<field>`)"))?;
            c.skip_ws();
            expect_char(c, '.')
                .map_err(|e| e.with_context("parsing rebuild trigger arg (expected `.`)"))?;
            c.skip_ws();
            let field = ident(c)
                .map_err(|e| e.with_context("parsing rebuild trigger field name"))?;
            // Use `head` to construct the qualified name; the actual
            // resolver-time check happens elsewhere.
            let _ = head; // accept any identifier for the head; checked at resolve
            c.skip_ws();
            expect_char(c, ')')
                .map_err(|e| e.with_context("parsing rebuild trigger arg (expected `)`)"))?;
            IndexRebuildTrigger::ChunkEpochAdvance { region_field: field }
        }
        "manual" => IndexRebuildTrigger::Manual,
        other => {
            return Err(ParseErr::at(
                here(c),
                format!(
                    "unknown rebuild trigger `{other}` (expected `chunk_epoch_advance(region.chunks)` or `manual`)"
                ),
            ));
        }
    };
    c.skip_ws();
    if c.starts_with_char(',') {
        c.bump(1);
    }
    Ok(result)
}

fn parse_index_build_clause(c: &mut Cursor) -> PResult<(String, IndexBuildBody)> {
    expect_keyword(c, "build")
        .map_err(|e| e.with_context("parsing index build clause (expected `build {`)"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing index build clause (expected `{`)"))?;
    // Capture raw text + parse AST in one pass. The raw text scans
    // to matching brace; the AST parser runs over the same range
    // (rewinding the cursor first so position tracking stays
    // accurate inside the body).
    let body_start = c.pos;
    let mut depth = 1usize;
    let src = c.src;
    let mut scan_pos = c.pos;
    while depth > 0 {
        if scan_pos >= src.len() {
            return Err(ParseErr::at(
                here(c),
                "unterminated `build { ... }` body".to_string(),
            ));
        }
        let b = src.as_bytes()[scan_pos];
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        scan_pos += 1;
    }
    let body_end = scan_pos;
    let body_raw = src[body_start..body_end].to_string();
    // Parse the body AST. Cursor is at body_start; parser advances
    // it through the body. After parsing we assert the cursor lands
    // at body_end (modulo trailing whitespace).
    let body_ast = parse_index_build_body(c, body_end)?;
    // Consume the closing `}` of the build block.
    expect_char(c, '}')
        .map_err(|e| e.with_context("parsing index build clause (expected `}`)"))?;
    // Optional trailing comma after the build block.
    c.skip_ws();
    if c.starts_with_char(',') {
        c.bump(1);
    }
    Ok((body_raw, body_ast))
}

/// Parse the body AST. Called with cursor positioned immediately
/// after the opening `{`; consumes characters up to (but not
/// including) the closing `}`.
///
/// Grammar:
///   body  := stmt* (return_expr)?
///   stmt  := `let` IDENT `=` expr `;`
///   return_expr := expr
///   expr  := IDENT (`.` IDENT)? | IDENT `(` expr_list `)` | `engine::` IDENT `(` expr_list `)` | INT
fn parse_index_build_body(c: &mut Cursor, body_end: usize) -> PResult<IndexBuildBody> {
    let mut stmts: Vec<IndexBuildStmt> = Vec::new();
    loop {
        c.skip_ws();
        if c.pos >= body_end {
            break;
        }
        let stmt_start = c.pos;
        // Detect `let` keyword vs trailing expression.
        if starts_with_keyword(c, "let") {
            c.bump(3);
            c.skip_ws();
            let name = ident(c)
                .map_err(|e| e.with_context("parsing index build `let` name"))?;
            c.skip_ws();
            expect_char(c, '=')
                .map_err(|e| e.with_context("parsing index build `let` (expected `=`)"))?;
            c.skip_ws();
            let value = parse_index_build_expr(c, body_end)?;
            c.skip_ws();
            expect_char(c, ';')
                .map_err(|e| e.with_context("parsing index build `let` (expected `;`)"))?;
            stmts.push(IndexBuildStmt::Let {
                name,
                value,
                span: Span::new(stmt_start, c.pos),
            });
        } else {
            // Trailing expression — must be the last thing in the body.
            let value = parse_index_build_expr(c, body_end)?;
            stmts.push(IndexBuildStmt::Return {
                value,
                span: Span::new(stmt_start, c.pos),
            });
            // After the return expr, only whitespace + optional
            // trailing semicolon should remain before body_end.
            c.skip_ws();
            if c.starts_with_char(';') {
                c.bump(1);
                c.skip_ws();
            }
            if c.pos < body_end {
                return Err(ParseErr::at(
                    here(c),
                    "trailing tokens after build body return expression".to_string(),
                ));
            }
            break;
        }
    }
    Ok(IndexBuildBody { stmts })
}

fn parse_index_build_expr(c: &mut Cursor, body_end: usize) -> PResult<IndexBuildExpr> {
    let start = c.pos;
    c.skip_ws();
    // Integer literal?
    if c.pos < body_end && (c.starts_with_char('-') || peek_number(c)) {
        let negate = if c.starts_with_char('-') {
            c.bump(1);
            true
        } else {
            false
        };
        let (n, is_float) = number_literal(c)?;
        if is_float {
            return Err(ParseErr::at(
                here(c),
                "index build body integer literal only (float NYI)".to_string(),
            ));
        }
        let mut v = n as i64;
        if negate {
            v = -v;
        }
        return Ok(IndexBuildExpr::Int { value: v, span: Span::new(start, c.pos) });
    }
    // Identifier — could be a bare var, a member access, a call,
    // or `engine::<helper>`.
    let head = ident(c).map_err(|e| e.with_context("parsing index build expr head"))?;
    // `engine::<name>(...)` — engine helper call.
    if head == "engine" && c.starts_with_char(':') && c.src.as_bytes().get(c.pos + 1) == Some(&b':') {
        c.bump(2);
        let helper = ident(c)
            .map_err(|e| e.with_context("parsing index build `engine::` helper name"))?;
        c.skip_ws();
        expect_char(c, '(')
            .map_err(|e| e.with_context("parsing engine call (expected `(`)"))?;
        let args = parse_index_build_arg_list(c, body_end)?;
        c.skip_ws();
        expect_char(c, ')')
            .map_err(|e| e.with_context("parsing engine call (expected `)`)"))?;
        return Ok(IndexBuildExpr::EngineCall {
            name: helper,
            args,
            span: Span::new(start, c.pos),
        });
    }
    // Bare ident followed by `(` — a user-defined call (rejected
    // today; the only call form is `engine::<name>(...)`). Allow
    // it as a syntactic shape but surface a typed error.
    if c.starts_with_char('(') {
        return Err(ParseErr::at(
            here(c),
            format!(
                "bare-name call `{head}(...)` not allowed in index build body — \
                 use `engine::{head}(...)` for engine helpers; user-defined \
                 helpers are not yet a thing"
            ),
        ));
    }
    // `region.chunks`-style member access. Single level only.
    if c.starts_with_char('.') {
        c.bump(1);
        let field = ident(c)
            .map_err(|e| e.with_context("parsing index build member access"))?;
        return Ok(IndexBuildExpr::Member {
            base: head,
            field,
            span: Span::new(start, c.pos),
        });
    }
    // Bare identifier — local binding or top-level constant.
    Ok(IndexBuildExpr::Var {
        name: head,
        span: Span::new(start, c.pos),
    })
}

fn parse_index_build_arg_list(
    c: &mut Cursor,
    body_end: usize,
) -> PResult<Vec<IndexBuildExpr>> {
    let mut args: Vec<IndexBuildExpr> = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char(')') {
            break;
        }
        let expr = parse_index_build_expr(c, body_end)?;
        args.push(expr);
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    Ok(args)
}

// ---------------------------------------------------------------------------
// 2.16 init (Plan E-A6 — fixture-owned initial buffer state in the DSL)
// ---------------------------------------------------------------------------

fn init_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<InitDecl> {
    expect_keyword(c, "init")
        .map_err(|e| e.with_context("parsing `init` declaration"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing init body (expected `{`)"))?;
    let mut stmts = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let stmt_start = c.pos;
        let field = ident(c).map_err(|e| e.with_context("parsing init field name"))?;
        c.skip_ws();
        expect_char(c, ':')
            .map_err(|e| e.with_context("parsing init stmt (expected `:` after field)"))?;
        c.skip_ws();
        let expr = if let Some(name) = peek_ident(c) {
            if name == "slot" {
                c.bump(name.len());
                InitExpr::Slot
            } else {
                return Err(ParseErr::at(
                    here(c),
                    format!(
                        "expected `slot` or integer literal as init expression; got `{name}`"
                    ),
                ));
            }
        } else if peek_number(c) {
            let (n, is_float) = number_literal(c)?;
            if is_float {
                return Err(ParseErr::at(
                    here(c),
                    "init expression must be an integer (no float fills)",
                ));
            }
            InitExpr::Const(n as i64)
        } else {
            return Err(ParseErr::at(
                here(c),
                "expected `slot` or integer literal as init expression",
            ));
        };
        stmts.push(InitStmt { field, expr, span: Span::new(stmt_start, c.pos) });
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    Ok(InitDecl { annotations, stmts, span: Span::new(start, c.pos) })
}

// ---------------------------------------------------------------------------
// 2.17 debug (compiler-debug-mode opt-in: surfaces LowerOpts.debug +
// LowerOpts.debug_wgsl from the .sim source instead of a hand-written
// build.rs)
//
// Grammar:
//   debug {
//     depth: <off|stage|stage_memory|kernel|dsl_mapped>,
//     wgsl_event_kind_histogram: true,
//     wgsl_mask_hit_rate: true,
//     wgsl_score_kernel_visits: true,
//   }
//
// All fields are optional; an empty `debug { }` is equivalent to omitting
// the block entirely. Trailing commas allowed. Unknown field names raise
// a parse error (catches typos early).
// ---------------------------------------------------------------------------

fn debug_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<DebugDecl> {
    expect_keyword(c, "debug")
        .map_err(|e| e.with_context("parsing `debug` declaration"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing debug body (expected `{`)"))?;
    let mut depth: Option<DebugDepthLit> = None;
    let mut wgsl_event_kind_histogram = false;
    let mut wgsl_mask_hit_rate = false;
    let mut wgsl_score_kernel_visits = false;
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let field = ident(c).map_err(|e| e.with_context("parsing debug field name"))?;
        c.skip_ws();
        expect_char(c, ':')
            .map_err(|e| e.with_context("parsing debug field (expected `:` after name)"))?;
        c.skip_ws();
        match field.as_str() {
            "depth" => {
                let level = ident(c).map_err(|e| e.with_context("parsing debug depth level"))?;
                depth = Some(match level.as_str() {
                    "off" => DebugDepthLit::Off,
                    "stage" => DebugDepthLit::Stage,
                    "stage_memory" => DebugDepthLit::StageMemory,
                    "kernel" => DebugDepthLit::Kernel,
                    "dsl_mapped" => DebugDepthLit::DslMapped,
                    other => {
                        return Err(ParseErr::at(
                            here(c),
                            format!(
                                "unknown debug depth `{other}`; expected one of off, stage, stage_memory, kernel, dsl_mapped"
                            ),
                        ));
                    }
                });
            }
            "wgsl_event_kind_histogram"
            | "wgsl_mask_hit_rate"
            | "wgsl_score_kernel_visits" => {
                let val = ident(c)
                    .map_err(|e| e.with_context("parsing debug bool value (true/false)"))?;
                let b = match val.as_str() {
                    "true" => true,
                    "false" => false,
                    other => {
                        return Err(ParseErr::at(
                            here(c),
                            format!("expected `true` or `false`; got `{other}`"),
                        ));
                    }
                };
                match field.as_str() {
                    "wgsl_event_kind_histogram" => wgsl_event_kind_histogram = b,
                    "wgsl_mask_hit_rate" => wgsl_mask_hit_rate = b,
                    "wgsl_score_kernel_visits" => wgsl_score_kernel_visits = b,
                    _ => unreachable!(),
                }
            }
            other => {
                return Err(ParseErr::at(
                    here(c),
                    format!(
                        "unknown debug field `{other}`; expected one of depth, wgsl_event_kind_histogram, wgsl_mask_hit_rate, wgsl_score_kernel_visits"
                    ),
                ));
            }
        }
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    Ok(DebugDecl {
        annotations,
        depth,
        wgsl_event_kind_histogram,
        wgsl_mask_hit_rate,
        wgsl_score_kernel_visits,
        span: Span::new(start, c.pos),
    })
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
        // Extend bare ident to a dotted path (`config.nav.radius`). The
        // path text is stored as a single `Ident(String)` value to keep
        // the AnnotationValue surface narrow — consumers that care
        // about the head/tail split it themselves.
        let mut full = name.clone();
        while c.starts_with_char('.') {
            // Look-ahead: only consume `.<ident>` (not `.5` floats etc.).
            let after_dot = c.pos + 1;
            let probe = Cursor { src: c.src, pos: after_dot };
            if probe.peek_char().map_or(false, is_ident_start) {
                c.bump(1);
                let segment = ident(c)?;
                full.push('.');
                full.push_str(&segment);
            } else {
                break;
            }
        }
        let name = full;
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
        // Spec `docs/spec/dsl.md:653-663` lists `Quest` as a typed
        // entity root alongside the three above. Accepted at parse
        // time as a declare-only root (no per-Quest SoA today; the
        // catalog populator at `cg/lower/driver.rs::populate_entity_
        // field_catalog` skips Quest the same way it skips Agent).
        "Quest" => EntityRoot::Quest,
        other => {
            return Err(ParseErr::at(
                here_back(c, other.len()),
                format!("expected `Agent`, `Item`, `Group`, or `Quest`; got `{other}`"),
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
    // Anonymous struct literal: bare `{ ... }` with no leading typename.
    // The shape is implicit from the field's declared type — used for
    // `predator_prey: { prey_of: [], preys_on: [] }` in entity bodies.
    if c.starts_with_char('{') {
        c.bump(1);
        let fields = parse_entity_fields(c)?;
        c.skip_ws();
        expect_char(c, '}')
            .map_err(|e| e.with_context("parsing anonymous struct literal `}`"))?;
        return Ok(EntityFieldValue::AnonStruct(fields));
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

/// Plan I — `belief <name>(observer: Agent[, key]) -> T { ... }`.
/// Same surface as `view_decl` for the signature + fold-body, plus
/// recognition of `merge from <agent>: <op>` clauses interleaved
/// with the body's `on <Event> {...}` propagation handlers. The
/// social-merge clauses split out into a separate list on
/// [`BeliefDecl`]; the propagation handlers stay in [`ViewBody::Fold`].
fn belief_decl(
    c: &mut Cursor,
    annotations: Vec<Annotation>,
    start: usize,
) -> PResult<BeliefDecl> {
    expect_keyword(c, "belief").map_err(|e| e.with_context("parsing `belief` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing belief name"))?;
    let params = parse_params(c)?;
    c.skip_ws();
    expect_str(c, "->").map_err(|e| e.with_context("parsing belief return-type arrow `->`"))?;
    c.skip_ws();
    let return_ty = type_ref(c)?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing belief body `{`"))?;
    // Custom body parser — mirrors parse_view_body's fold path but
    // also recognises `merge from <agent>: <op>` clauses. We require
    // beliefs to use the fold form (initial: + on/merge handlers);
    // the lazy-expr form isn't meaningful for belief storage.
    let (body, social_merges) = parse_belief_body(c)?;
    c.skip_ws();
    expect_char(c, '}').map_err(|e| e.with_context("parsing belief body `}`"))?;
    Ok(BeliefDecl {
        annotations,
        name,
        params,
        return_ty,
        body,
        social_merges,
        span: Span::new(start, c.pos),
    })
}

/// Parse a belief body: `initial: <expr>` followed by zero or more
/// `on <Event> { ... }` (propagation, lowered as fold handlers) AND
/// `on <Event> { <pattern> } [where <expr>] merge from <agent>: <op>`
/// (social-merge clauses, split out separately). Optional `decay:`
/// + `clamp:` clauses are recognised by the same code paths as views.
fn parse_belief_body(c: &mut Cursor) -> PResult<(ViewBody, Vec<SocialMergeClause>)> {
    c.skip_ws();
    expect_keyword(c, "initial")
        .map_err(|e| e.with_context("parsing belief body — must start with `initial:`"))?;
    c.skip_ws();
    expect_char(c, ':').map_err(|e| e.with_context("parsing belief `initial:`"))?;
    c.skip_ws();
    let initial = parse_expr(c)?;
    c.skip_ws();
    if c.starts_with_char(',') {
        c.bump(1);
    }
    let mut handlers = Vec::new();
    let mut clamp = None;
    let mut social_merges = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            break;
        }
        // clamp clause — same shape as view body.
        if c.starts_with("clamp")
            && c.src[c.pos + "clamp".len()..]
                .chars()
                .next()
                .map_or(true, |ch| !is_ident_cont(ch))
        {
            c.bump("clamp".len());
            c.skip_ws();
            expect_char(c, ':').map_err(|e| e.with_context("parsing belief `clamp:`"))?;
            c.skip_ws();
            expect_char(c, '[').map_err(|e| e.with_context("parsing clamp bounds `[`"))?;
            c.skip_ws();
            let lo = parse_expr(c)?;
            c.skip_ws();
            expect_char(c, ',').map_err(|e| e.with_context("parsing clamp bounds `,`"))?;
            c.skip_ws();
            let hi = parse_expr(c)?;
            c.skip_ws();
            expect_char(c, ']').map_err(|e| e.with_context("parsing clamp bounds `]`"))?;
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
            }
            clamp = Some((lo, hi));
            continue;
        }
        // on-handler — could be a propagation handler (body is `{...}`)
        // or a social-merge clause (body is `merge from <agent>: <op>`).
        // Distinguish by peeking past the pattern.
        if c.starts_with("on ") || c.starts_with("on\t") || c.starts_with("on\n") {
            // Try social-merge first; if the post-pattern token isn't
            // `merge` (or `where ... merge`), fall back to fold handler.
            // To do this without backtracking, we parse a generalised
            // belief handler that returns either variant.
            let handler_or_merge = parse_belief_handler(c)?;
            match handler_or_merge {
                BeliefHandler::Propagation(h) => handlers.push(h),
                BeliefHandler::SocialMerge(m) => social_merges.push(m),
            }
            continue;
        }
        // Fixture-extension fields like view body's parse-and-discard
        // shape. Keep parity with parse_view_body.
        if let Some(name) = peek_ident(c) {
            let after = c.pos + name.len();
            let mut look = Cursor { src: c.src, pos: after };
            look.skip_ws();
            if look.starts_with_char(':') {
                c.bump(name.len());
                c.skip_ws();
                c.bump(1);
                c.skip_ws();
                let _ = parse_expr(c)?;
                c.skip_ws();
                if c.starts_with_char(',') {
                    c.bump(1);
                }
                continue;
            }
        }
        break;
    }
    Ok((ViewBody::Fold { initial, handlers, clamp }, social_merges))
}

enum BeliefHandler {
    Propagation(FoldHandler),
    SocialMerge(SocialMergeClause),
}

/// Parse one `on <Event> { <pattern> } [where <expr>] (body | merge from <agent>: <op>)`
/// into either a propagation handler or a social-merge clause. The
/// pattern + optional where-clause are shared; the trailing form
/// (`{` body vs `merge from`) disambiguates. Mirrors `parse_fold_handler`'s
/// where-clause + body shape so propagation handlers parse identically
/// to view fold handlers.
fn parse_belief_handler(c: &mut Cursor) -> PResult<BeliefHandler> {
    let start = c.pos;
    expect_keyword(c, "on").map_err(|e| e.with_context("parsing belief `on` handler"))?;
    c.skip_ws();
    let pattern = parse_event_pattern(c)?;
    c.skip_ws();
    // where-clause, terminated at `{` OR at `merge` keyword (so the
    // bare expr parser doesn't consume the merge clause).
    let where_clause = if c.starts_with("where") {
        let after = c.pos + "where".len();
        let next = c.src[after..].chars().next();
        if next.map_or(true, |ch| !is_ident_cont(ch)) {
            c.bump("where".len());
            c.skip_ws();
            // Bound at `{` (propagation body) OR `merge` (social-merge
            // tail) so the predicate doesn't gobble the trailing
            // form.
            let expr = parse_expr_bounded(c, |ck| {
                ck.starts_with_char('{')
                    || (ck.starts_with("merge")
                        && ck.src[ck.pos + "merge".len()..]
                            .chars()
                            .next()
                            .map_or(true, |ch| !is_ident_cont(ch)))
            })?;
            c.skip_ws();
            Some(expr)
        } else {
            None
        }
    } else {
        None
    };
    c.skip_ws();
    // Disambiguate: `merge from <agent>: <op>` is a social-merge;
    // `{ ... }` is a propagation handler body.
    if c.starts_with("merge")
        && c.src[c.pos + "merge".len()..]
            .chars()
            .next()
            .map_or(true, |ch| !is_ident_cont(ch))
    {
        c.bump("merge".len());
        c.skip_ws();
        expect_keyword(c, "from")
            .map_err(|e| e.with_context("parsing `merge from` clause"))?;
        c.skip_ws();
        let source_agent_name = ident(c)
            .map_err(|e| e.with_context("parsing source-agent identifier in `merge from`"))?;
        c.skip_ws();
        expect_char(c, ':')
            .map_err(|e| e.with_context("parsing `:` in `merge from <agent>:`"))?;
        c.skip_ws();
        let op_name = ident(c)
            .map_err(|e| e.with_context("parsing merge op name (bit_or / max / min / replace)"))?;
        let op = match op_name.as_str() {
            "bit_or" => SocialMergeOpName::BitOr,
            "max" => SocialMergeOpName::Max,
            "min" => SocialMergeOpName::Min,
            "replace" => SocialMergeOpName::Replace,
            other => {
                return Err(ParseErr::at(
                    Span::new(start, c.pos),
                    format!(
                        "unknown merge op `{other}` — expected one of bit_or / max / min / replace"
                    ),
                ));
            }
        };
        Ok(BeliefHandler::SocialMerge(SocialMergeClause {
            pattern,
            where_clause,
            source_agent_name,
            op,
            span: Span::new(start, c.pos),
        }))
    } else {
        // Propagation body — same shape as a view fold handler.
        expect_char(c, '{').map_err(|e| e.with_context("parsing belief handler body `{`"))?;
        let body = parse_stmt_block_until_close(c)?;
        expect_char(c, '}').map_err(|e| e.with_context("parsing belief handler body `}`"))?;
        Ok(BeliefHandler::Propagation(FoldHandler {
            pattern,
            where_clause,
            body,
            span: Span::new(start, c.pos),
        }))
    }
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
                if c.starts_with("clamp")
                    && c.src[c.pos + "clamp".len()..]
                        .chars()
                        .next()
                        .map_or(true, |ch| !is_ident_cont(ch))
                {
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
                // Fixture-extension fields like `clamp_norm: 100.0` —
                // the standard view-body grammar (initial/on/clamp)
                // doesn't recognise them. Parse-and-discard rather than
                // failing so the .sim still lowers; semantic adoption
                // is a future grammar slice.
                if let Some(name) = peek_ident(c) {
                    let after = c.pos + name.len();
                    let mut look = Cursor { src: c.src, pos: after };
                    look.skip_ws();
                    if look.starts_with_char(':') {
                        c.bump(name.len());
                        c.skip_ws();
                        c.bump(1); // `:`
                        c.skip_ws();
                        let _ = parse_expr(c)?;
                        c.skip_ws();
                        if c.starts_with_char(',') {
                            c.bump(1);
                        }
                        continue;
                    }
                }
                break;
            }
            return Ok(ViewBody::Fold { initial, handlers, clamp });
        }
    }
    // Allow a prelude of `let <name> = <expr>;` bindings before the
    // final expression. Wrapped into a single `ExprKind::Block` so
    // the resolver sees the bindings in scope when it walks the
    // final expression. Used by `@lazy` view bodies in the design-
    // target `crowd_navigation.sim`.
    let bstart = c.pos;
    let mut bindings: Vec<(String, Expr)> = Vec::new();
    while c.starts_with("let ") || c.starts_with("let\t") {
        c.bump("let".len());
        c.skip_ws();
        let bname = ident(c)?;
        c.skip_ws();
        expect_char(c, '=').map_err(|e| e.with_context("parsing view-body `let =`"))?;
        c.skip_ws();
        let value = parse_expr_bounded(
            c,
            |ck| ck.starts_with_char(';') || ck.starts_with_char('\n'),
        )?;
        bindings.push((bname, value));
        c.skip_ws();
        if c.starts_with_char(';') {
            c.bump(1);
        }
        c.skip_ws();
    }
    let final_expr = parse_expr(c)?;
    let expr = if bindings.is_empty() {
        final_expr
    } else {
        let span = Span::new(bstart, final_expr.span.end);
        Expr {
            kind: ExprKind::Block { bindings, expr: Box::new(final_expr) },
            span,
        }
    };
    Ok(ViewBody::Expr(expr))
}

fn parse_fold_handler(c: &mut Cursor) -> PResult<FoldHandler> {
    let start = c.pos;
    expect_keyword(c, "on").map_err(|e| e.with_context("parsing fold `on` handler"))?;
    c.skip_ws();
    let pattern = parse_event_pattern(c)?;
    c.skip_ws();
    // Optional `where <predicate>` between pattern and body — same
    // surface physics handlers already accept. Resolver gates the
    // fold's write on this when present.
    let where_clause = if c.starts_with("where") {
        // Boundary check: `where` only counts as a keyword if not part
        // of a longer identifier (e.g. `where_house`).
        let after = c.pos + "where".len();
        let next = c.src[after..].chars().next();
        if next.map_or(true, |ch| !is_ident_cont(ch)) {
            c.bump("where".len());
            c.skip_ws();
            // Same body-`{` boundary as physics handler — see comment
            // on `parse_physics_handler::where_clause` for the rationale.
            let expr = parse_expr_bounded(c, |ck| ck.starts_with_char('{'))?;
            c.skip_ws();
            Some(expr)
        } else {
            None
        }
    } else {
        None
    };
    expect_char(c, '{').map_err(|e| e.with_context("parsing fold handler body `{`"))?;
    let body = parse_stmt_block_until_close(c)?;
    c.skip_ws();
    expect_char(c, '}').map_err(|e| e.with_context("parsing fold handler body `}`"))?;
    Ok(FoldHandler { pattern, where_clause, body, span: Span::new(start, c.pos) })
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

fn parse_param_type(c: &mut Cursor) -> PResult<crate::ast::ParamType> {
    let pstart = here(c);
    let type_ident = ident(c).map_err(|e| e.with_context("parsing parameter type"))?;
    match type_ident.as_str() {
        "f32" => Ok(crate::ast::ParamType::F32),
        "i32" => Ok(crate::ast::ParamType::I32),
        "u32" => Ok(crate::ast::ParamType::U32),
        "bool" => Ok(crate::ast::ParamType::Bool),
        "EntityKind" => Ok(crate::ast::ParamType::EntityKind),
        other => Err(ParseErr::at(
            pstart,
            format!("unknown parameter type `{other}`; expected one of f32, i32, u32, bool, EntityKind"),
        )),
    }
}

fn parse_rule_params(c: &mut Cursor) -> PResult<Vec<crate::ast::ParamDecl>> {
    c.skip_ws();
    if !c.starts_with_char('(') {
        return Ok(Vec::new());
    }
    c.bump(1); // consume '('
    let mut params: Vec<crate::ast::ParamDecl> = Vec::new();
    let mut seen_names = std::collections::HashSet::new();
    loop {
        c.skip_ws();
        if c.starts_with_char(')') {
            c.bump(1);
            break;
        }
        let pstart = c.pos;
        let pname = ident(c).map_err(|e| e.with_context("parsing rule parameter name"))?;
        if !seen_names.insert(pname.clone()) {
            return Err(ParseErr::at(
                Span::new(pstart, c.pos),
                format!("duplicate parameter name `{pname}`"),
            ));
        }
        c.skip_ws();
        expect_char(c, ':').map_err(|e| e.with_context("parsing rule parameter `:`"))?;
        c.skip_ws();
        let ty = parse_param_type(c)?;
        params.push(crate::ast::ParamDecl { name: pname, ty, span: Span::new(pstart, c.pos) });
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
            continue;
        }
        if c.starts_with_char(')') {
            c.bump(1);
            break;
        }
        return Err(ParseErr::at(here(c), "expected `,` or `)` in rule parameter list"));
    }
    Ok(params)
}

/// Dispatch: returns either `Decl::Physics` (template or concrete rule) or
/// `Decl::PhysicsApply` (the `physics X = template(args);` apply form).
fn physics_any_decl(c: &mut Cursor, annotations: Vec<Annotation>, start: usize) -> PResult<Decl> {
    expect_keyword(c, "physics").map_err(|e| e.with_context("parsing `physics` declaration"))?;
    let name = ident(c).map_err(|e| e.with_context("parsing physics name"))?;
    c.skip_ws();
    // Mutually exclusive: `=` → apply form; `(` → param list; else → concrete rule.
    if c.starts_with_char('=') {
        c.bump(1); // consume `=`
        c.skip_ws();
        let template = ident(c).map_err(|e| e.with_context("parsing apply template name"))?;
        expect_char(c, '(').map_err(|e| e.with_context("parsing apply arg list `(`"))?;
        let mut args: Vec<crate::ast::ApplyArg> = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char(')') {
                break;
            }
            let astart = c.pos;
            let arg_name = ident(c).map_err(|e| e.with_context("parsing apply arg name"))?;
            expect_char(c, ':').map_err(|e| e.with_context("parsing `:` after apply arg name"))?;
            c.skip_ws();
            let value = parse_apply_arg_value(c)?;
            args.push(crate::ast::ApplyArg {
                name: arg_name,
                value,
                span: Span::new(astart, c.pos),
            });
            c.skip_ws();
            if c.starts_with_char(',') {
                c.bump(1);
                continue;
            }
            break;
        }
        expect_char(c, ')').map_err(|e| e.with_context("parsing apply arg list `)`"))?;
        expect_char(c, ';').map_err(|e| e.with_context("parsing apply `;`"))?;
        return Ok(Decl::PhysicsApply(crate::ast::PhysicsApplyDecl {
            annotations,
            name,
            template,
            args,
            span: Span::new(start, c.pos),
        }));
    }
    // Regular physics (template with params, or concrete rule without params).
    // Reconstruct the cursor state as if we entered physics_decl normally but
    // already consumed `physics <name>`.  We pass the already-parsed name
    // through to a helper that continues from after the name.
    physics_decl_after_name(c, annotations, name, start).map(Decl::Physics)
}

fn parse_apply_arg_value(c: &mut Cursor) -> PResult<crate::ast::ApplyArgValue> {
    c.skip_ws();
    // Numeric literal (f32, i32, u32).
    if peek_number(c) {
        let (n, is_float) = number_literal(c)?;
        if is_float {
            return Ok(crate::ast::ApplyArgValue::F32(n as f32));
        }
        // Integer: prefer I32, fall back to U32 for large values.
        if n >= 0.0 && n <= (i32::MAX as f64) {
            return Ok(crate::ast::ApplyArgValue::I32(n as i32));
        }
        if n >= 0.0 && n <= (u32::MAX as f64) {
            return Ok(crate::ast::ApplyArgValue::U32(n as u32));
        }
        return Err(ParseErr::at(here(c), format!("integer literal {n} out of range for i32/u32")));
    }
    // Boolean literals.
    if starts_with_keyword(c, "true") {
        expect_keyword(c, "true")?;
        return Ok(crate::ast::ApplyArgValue::Bool(true));
    }
    if starts_with_keyword(c, "false") {
        expect_keyword(c, "false")?;
        return Ok(crate::ast::ApplyArgValue::Bool(false));
    }
    // Bare identifier → EntityKind reference.
    let id = ident(c).map_err(|e| e.with_context("parsing apply arg value (expected literal or identifier)"))?;
    Ok(crate::ast::ApplyArgValue::EntityKind(id))
}

fn physics_decl_after_name(c: &mut Cursor, annotations: Vec<Annotation>, name: String, start: usize) -> PResult<PhysicsDecl> {
    // Parse optional parameterised rule params: `physics chase(target: EntityKind, ...) { ... }`.
    c.skip_ws();
    let params = parse_rule_params(c)?;
    // Annotations may appear after the name (or params): `physics foo @phase(event) { ... }`.
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
        params,
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
        // Bound the where-predicate at the body's opening `{` — without
        // this, a tail like `where self.creature_type == Hare {` parses
        // `Hare {...}` as a struct literal and swallows the entire body.
        // Mirrors the for/if/match parsers that also stop at `{`.
        where_clause = Some(parse_expr_bounded(c, |ck| ck.starts_with_char('{'))?);
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
    // Verb body: any mix of `emit <Event> { ... }` and
    // `apply_ability <expr> [by <c>] [target <t>]` statements, in
    // source order. Either form fires as part of the synthesised
    // cascade physics handler when the verb's action wins the
    // per-agent argmax. Task #138 (Wave 1.7) added the apply_ability
    // alternative so a verb can dispatch through the
    // PackedAbilityRegistry rather than hand-mirror per-effect
    // chronicle events.
    let mut body = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with("emit ") || c.starts_with("emit\t") || c.starts_with("emit\n") {
            let stmt = parse_emit_stmt(c)?;
            // Skip the parse-and-discarded empty `emit` form (used when
            // a verb declares `emit  // (none)` to document an action
            // that emits nothing). Real emits with an event name flow
            // through to lowering as before.
            if !stmt.event_name.is_empty() {
                body.push(VerbBodyStmt::Emit(stmt));
            }
            continue;
        }
        if c.starts_with("apply_ability ")
            || c.starts_with("apply_ability\t")
            || c.starts_with("apply_ability\n")
        {
            body.push(VerbBodyStmt::ApplyAbility(parse_apply_ability_stmt(c)?));
            continue;
        }
        break;
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
    Ok(VerbDecl { annotations, name, params, action, when, body, scoring, span: Span::new(start, c.pos) })
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
    // Optional entity-binding name: `scoring Wolf {` scopes the table
    // to a specific entity. Resolver doesn't currently use the name —
    // accepted at parse time so fixtures with multi-entity scoring
    // (predator_prey, etc.) parse cleanly.
    if let Some(name) = peek_ident(c) {
        if name.chars().next().map_or(false, |c0| c0.is_ascii_uppercase()) {
            c.bump(name.len());
            c.skip_ws();
        }
    }
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
    // Optional row-kind keyword. The standard form is `per_ability`,
    // but design-target fixtures (predator_prey, crowd_navigation) use
    // `per_target` — and a no-kind shape (`row Wait { ... }`) for
    // verbs that score once per agent without target enumeration. The
    // kind is parse-and-discarded today; semantic adoption is a future
    // grammar slice. Whichever shape we see, the body is the same
    // free-form clause-list below.
    if c.starts_with("per_ability") || c.starts_with("per_target") {
        // Detect-and-skip whichever keyword is present.
        if c.starts_with("per_ability") {
            c.bump("per_ability".len());
        } else {
            c.bump("per_target".len());
        }
        c.skip_ws();
    }
    expect_char(c, '{').map_err(|e| e.with_context("parsing per_ability row `{`"))?;
    let mut guard: Option<Expr> = None;
    let mut score: Option<Expr> = None;
    let mut target: Option<Expr> = None;
    let mut weights: Option<Expr> = None;
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
            // Design-target fixtures use `base:` + `weights:` clauses
            // for the utility-table form of scoring rows. The lowerer
            // composes the row's utility as `base + weights` (both
            // F32). `base:` doubles as the score field; `weights:` is
            // captured into a sibling `weights` slot the IR / lowerer
            // consume. Closes Gap C from `gaps_observed.md` (2026-05-11):
            // pre-fix the parser parse-and-discarded `weights:` so the
            // personality-weighted scoring rows contributed nothing to
            // argmax. See `cg::lower::scoring::lower_per_ability_row`.
            "base" => {
                if score.is_some() {
                    return Err(ParseErr::at(
                        here(c),
                        "duplicate `base:` clause in per_ability row \
                         (also conflicts with `score:`)",
                    ));
                }
                score = Some(expr);
            }
            "weights" => {
                if weights.is_some() {
                    return Err(ParseErr::at(
                        here(c),
                        "duplicate `weights:` clause in per_ability row",
                    ));
                }
                weights = Some(expr);
            }
            // Any other identifier-keyed clause is parse-and-discarded.
            // Design-target rows carry fixture-specific fields like
            // `cooldown:`, `range:`, `targeting:` that the utility
            // surface will adopt later — for now the parser just
            // needs to round-trip them cleanly.
            _other => {
                let _ = expr;
            }
        }
        // Optional trailing comma between clauses.
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    // If no `score:` and no `base:` was present, synthesise a 0.0
    // placeholder so per-row lowering doesn't choke. Design-target
    // rows often omit explicit scoring while their `weights` clause
    // carries the real signal — reconcile when the utility-scoring
    // surface lands.
    let score = score.unwrap_or_else(|| Expr {
        kind: ExprKind::Float(0.0),
        span: Span::new(start, c.pos),
    });
    Ok(PerAbilityRow {
        name,
        guard,
        score,
        target,
        weights,
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
        // Optional `:` between field name and value. Older .sim files
        // omit it (`scenario "foo"`); newer design-target .sim files
        // include it (`scenario: "foo"`). Accept both.
        if c.starts_with_char(':') {
            c.bump(1);
            c.skip_ws();
        }
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
                if c.starts_with_char('{') {
                    c.bump(1);
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
                } else {
                    // Single inline assert: `assert: <expr>` (no block).
                    // Newer probe grammar uses this shape — accept it
                    // alongside the existing `assert { … }` block form.
                    asserts.push(parse_assert_expr(c)?);
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
    // Detect the legacy `count[...]` / `pr[...]` / `mean[...]` head
    // shapes by peeking the next ident + `[`. If the head is one of
    // those three AND followed by `[`, use the closed-form parser.
    // Otherwise treat the whole assert as a generic predicate
    // expression (Raw) — supports design-target shapes like
    // `forall g in groups: …`, `events.kind_count(…) > 0`,
    // `count(…) < 0.05 * count(…)`, etc.
    let try_head = peek_ident(c);
    let is_legacy = match try_head.as_deref() {
        Some(h) if matches!(h, "count" | "pr" | "mean") => {
            let after = c.pos + h.len();
            let mut look = Cursor { src: c.src, pos: after };
            look.skip_ws();
            look.starts_with_char('[')
        }
        _ => false,
    };
    if !is_legacy {
        let expr = parse_expr(c)?;
        let span = Span::new(start, c.pos);
        return Ok(AssertExpr::Raw { expr, span });
    }
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
    // Two shapes:
    //
    //   1. Legacy block:   `metric { metric name = value, ... }`  (multi-decl)
    //   2. Design-target:  `metric <name> { value: <expr>, emit_every: <n>, ... }`
    //
    // The disambiguator is whether the next token is `{` (legacy) or
    // an identifier (design-target single-metric form).
    if c.starts_with_char('{') {
        c.bump(1);
        let mut metrics = Vec::new();
        loop {
            c.skip_ws();
            if c.starts_with_char('}') {
                c.bump(1);
                break;
            }
            metrics.push(parse_metric_decl(c)?);
        }
        return Ok(MetricBlock { annotations, metrics, span: Span::new(start, c.pos) });
    }
    // Design-target single-metric form. Parse one MetricDecl with the
    // `value:` / `emit_every:` / `alert_when:` field syntax and wrap
    // it in a MetricBlock so downstream consumers see the same shape.
    let m = parse_metric_decl_field_form(c, start)?;
    Ok(MetricBlock { annotations, metrics: vec![m], span: Span::new(start, c.pos) })
}

/// Design-target metric form:
///   `metric <name> { value: <expr>, emit_every: <n>, alert_when: <expr> }`
/// Mirrors the legacy `metric <name> = <expr> [...clauses...]` parser
/// but with `: <value>` separators and trailing-comma blocks.
fn parse_metric_decl_field_form(c: &mut Cursor, start: usize) -> PResult<MetricDecl> {
    let name = ident(c).map_err(|e| e.with_context("parsing metric name"))?;
    c.skip_ws();
    expect_char(c, '{').map_err(|e| e.with_context("parsing metric body `{`"))?;
    let mut value: Option<Expr> = None;
    let mut window = None;
    let mut emit_every = None;
    let mut conditioned_on: Option<Expr> = None;
    let mut alert_when: Option<Expr> = None;
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let field = ident(c).map_err(|e| e.with_context("parsing metric field name"))?;
        c.skip_ws();
        expect_char(c, ':').map_err(|e| e.with_context("parsing metric field `:`"))?;
        c.skip_ws();
        match field.as_str() {
            "value" => {
                value = Some(parse_expr_bounded(c, |ck| {
                    ck.starts_with_char(',') || ck.starts_with_char('}')
                })?);
            }
            "emit_every" => {
                let (n, _) = number_literal(c)?;
                emit_every = Some(n as u64);
            }
            "window" => {
                let (n, _) = number_literal(c)?;
                window = Some(n as u64);
            }
            "conditioned_on" => {
                conditioned_on = Some(parse_expr_bounded(c, |ck| {
                    ck.starts_with_char(',') || ck.starts_with_char('}')
                })?);
            }
            "alert_when" => {
                alert_when = Some(parse_expr_bounded(c, |ck| {
                    ck.starts_with_char(',') || ck.starts_with_char('}')
                })?);
            }
            other => {
                return Err(ParseErr::at(
                    here_back(c, other.len()),
                    format!(
                        "unknown metric field `{other}`; expected `value`, `emit_every`, `window`, `conditioned_on`, or `alert_when`"
                    ),
                ));
            }
        }
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    let value = value.ok_or_else(|| {
        ParseErr::at(
            Span::new(start, c.pos),
            "metric is missing required `value:` field",
        )
    })?;
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
    // Optional trailing `@runtime` annotation (Plan G tunable cfg).
    // Consumed here — NOT via the generic `parse_annotations` driver —
    // because config fields don't carry the full annotation surface
    // today; only this single boolean knob is wired through. Other
    // annotations after the default value are reported as a parse error
    // so a future extension lands as a typed grammar change rather than
    // a silent ignore.
    //
    // The lookahead is lookahead-safe: `parse_config_default` leaves the
    // cursor exactly after the literal, and the outer `config_decl` loop
    // skips whitespace before checking for `,` / `}`. We peek `@` after
    // a whitespace skip; a non-`@` char rolls back to the post-default
    // position so the comma / brace gating still works.
    let mut runtime = false;
    let after_default = c.pos;
    c.skip_ws();
    if c.starts_with_char('@') {
        let ann = parse_annotation(c)?;
        if ann.name == "runtime" {
            if !ann.args.is_empty() {
                return Err(ParseErr::at(
                    here(c),
                    "@runtime annotation on config fields takes no arguments",
                ));
            }
            runtime = true;
        } else {
            return Err(ParseErr::at(
                here(c),
                format!(
                    "unknown annotation `@{}` on config field — only `@runtime` is supported",
                    ann.name
                ),
            ));
        }
    } else {
        // Roll back the whitespace skip so the outer config-decl loop's
        // `skip_ws + comma/brace` gating starts from the same position
        // it would have without this lookahead.
        c.pos = after_default;
    }
    Ok(ConfigField { name, ty, default, runtime, span: Span::new(fstart, c.pos) })
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
// Plan A — player-facing descriptor blocks (`controls` / `render` / `ui`).
// Each lowers to a `&'static str` JSON descriptor on the generated runtime.
// ---------------------------------------------------------------------------

/// Parse a non-negative or signed f64 number (no `@runtime`-style suffix).
fn parse_f64(c: &mut Cursor) -> PResult<f64> {
    c.skip_ws();
    let negative = if c.starts_with_char('-') {
        c.bump(1);
        c.skip_ws();
        true
    } else {
        false
    };
    let (v, _is_float) = number_literal(c)?;
    Ok(if negative { -v } else { v })
}

/// Parse a `u32` literal (rejects negatives / out-of-range).
fn parse_u32(c: &mut Cursor) -> PResult<u32> {
    c.skip_ws();
    let (v, is_float) = number_literal(c)?;
    if is_float || v < 0.0 || v > u32::MAX as f64 {
        return Err(ParseErr::at(here(c), "expected a u32 integer literal"));
    }
    Ok(v as u32)
}

/// Parse a `u8` color channel (0..=255).
fn parse_u8_channel(c: &mut Cursor) -> PResult<u8> {
    c.skip_ws();
    let (v, is_float) = number_literal(c)?;
    if is_float || v < 0.0 || v > 255.0 {
        return Err(ParseErr::at(here(c), "color channel must be an integer 0..=255"));
    }
    Ok(v as u8)
}

/// Parse `(r, g, b)` into `[u8; 3]`.
fn parse_color(c: &mut Cursor) -> PResult<[u8; 3]> {
    expect_char(c, '(').map_err(|e| e.with_context("parsing color `(`"))?;
    let r = parse_u8_channel(c)?;
    expect_char(c, ',').map_err(|e| e.with_context("parsing color `,`"))?;
    let g = parse_u8_channel(c)?;
    expect_char(c, ',').map_err(|e| e.with_context("parsing color `,`"))?;
    let b = parse_u8_channel(c)?;
    expect_char(c, ')').map_err(|e| e.with_context("parsing color `)`"))?;
    Ok([r, g, b])
}

/// Parse `when <field> in [lo, hi]` into a `FieldRangeDecl`.
fn parse_field_range(c: &mut Cursor) -> PResult<crate::ast::FieldRangeDecl> {
    expect_keyword(c, "when").map_err(|e| e.with_context("parsing field range `when`"))?;
    let field = ident(c).map_err(|e| e.with_context("parsing field range column name"))?;
    expect_keyword(c, "in").map_err(|e| e.with_context("parsing field range `in`"))?;
    expect_char(c, '[').map_err(|e| e.with_context("parsing field range `[`"))?;
    let lo = parse_f64(c)?;
    expect_char(c, ',').map_err(|e| e.with_context("parsing field range `,`"))?;
    let hi = parse_f64(c)?;
    expect_char(c, ']').map_err(|e| e.with_context("parsing field range `]`"))?;
    Ok(crate::ast::FieldRangeDecl { field, lo, hi })
}

/// ```text
/// controls {
///   key "w" -> ctl.move_y: 1.0
///   key "space" -> ctl.bolt_rate_level: 1.0 press
/// }
/// ```
fn controls_decl(c: &mut Cursor, start: usize) -> PResult<crate::ast::ControlsDecl> {
    expect_keyword(c, "controls").map_err(|e| e.with_context("parsing `controls` block"))?;
    expect_char(c, '{').map_err(|e| e.with_context("parsing controls body `{`"))?;
    let mut bindings = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            break;
        }
        let bstart = c.pos;
        expect_keyword(c, "key").map_err(|e| e.with_context("parsing controls binding `key`"))?;
        c.skip_ws();
        // Key name: quoted string or bare identifier.
        let key = if c.starts_with_char('"') {
            string_lit(c)?
        } else {
            ident(c).map_err(|e| e.with_context("parsing controls key name"))?
        };
        expect_str(c, "->").map_err(|e| e.with_context("parsing controls binding `->`"))?;
        let block = ident(c).map_err(|e| e.with_context("parsing controls target block name"))?;
        expect_char(c, '.').map_err(|e| e.with_context("parsing controls target `.`"))?;
        let field = ident(c).map_err(|e| e.with_context("parsing controls target field name"))?;
        expect_char(c, ':').map_err(|e| e.with_context("parsing controls value `:`"))?;
        let value = parse_f64(c)?;
        // Optional trailing `press` keyword → Press mode; else Hold.
        let after_value = c.pos;
        c.skip_ws();
        let press = if starts_with_keyword(c, "press") {
            c.bump("press".len());
            true
        } else {
            c.pos = after_value;
            false
        };
        bindings.push(crate::ast::ControlBinding {
            key,
            block,
            field,
            value,
            press,
            span: Span::new(bstart, c.pos),
        });
        c.skip_ws();
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }
    expect_char(c, '}').map_err(|e| e.with_context("parsing controls body `}`"))?;
    Ok(crate::ast::ControlsDecl { bindings, span: Span::new(start, c.pos) })
}

/// ```text
/// render {
///   arena_radius 120.0
///   camera follow when mana in [0.5, 1.5]
///   agent when mana in [0.5, 1.5] { color (0, 220, 220) }
///   vfx on NovaFire period 40 { ring radius 6.0 color (255, 255, 120) }
///   vfx on Bolt period 12 { beam_to_nearest when mana in [1.5, 9.9] color (120, 200, 255) }
/// }
/// ```
fn render_decl(c: &mut Cursor, start: usize) -> PResult<crate::ast::RenderDecl> {
    use crate::ast::{AgentVisualDecl, CameraDecl, VfxDecl, VfxKindDecl};
    expect_keyword(c, "render").map_err(|e| e.with_context("parsing `render` block"))?;
    expect_char(c, '{').map_err(|e| e.with_context("parsing render body `{`"))?;
    let mut arena_radius: Option<f64> = None;
    let mut camera: Option<CameraDecl> = None;
    let mut agents: Vec<AgentVisualDecl> = Vec::new();
    let mut vfx: Vec<VfxDecl> = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            break;
        }
        match peek_ident(c).as_deref() {
            Some("arena_radius") => {
                expect_keyword(c, "arena_radius")?;
                arena_radius = Some(parse_f64(c)?);
            }
            Some("camera") => {
                expect_keyword(c, "camera")?;
                c.skip_ws();
                if starts_with_keyword(c, "follow") {
                    c.bump("follow".len());
                    let range = parse_field_range(c)?;
                    camera = Some(CameraDecl::Follow(range));
                } else if starts_with_keyword(c, "observer") {
                    c.bump("observer".len());
                    camera = Some(CameraDecl::Observer);
                } else {
                    return Err(ParseErr::at(here(c),
                        "expected `follow when …` or `observer` after `camera`"));
                }
            }
            Some("agent") => {
                expect_keyword(c, "agent")?;
                let when = parse_field_range(c)?;
                expect_char(c, '{').map_err(|e| e.with_context("parsing agent visual `{`"))?;
                expect_keyword(c, "color").map_err(|e| e.with_context("parsing agent visual `color`"))?;
                let color = parse_color(c)?;
                expect_char(c, '}').map_err(|e| e.with_context("parsing agent visual `}`"))?;
                agents.push(AgentVisualDecl { when, color });
            }
            Some("vfx") => {
                expect_keyword(c, "vfx")?;
                expect_keyword(c, "on").map_err(|e| e.with_context("parsing vfx `on`"))?;
                let on_rule = ident(c).map_err(|e| e.with_context("parsing vfx rule name"))?;
                expect_keyword(c, "period").map_err(|e| e.with_context("parsing vfx `period`"))?;
                let period = parse_u32(c)?;
                expect_char(c, '{').map_err(|e| e.with_context("parsing vfx body `{`"))?;
                c.skip_ws();
                let (kind, radius, color) = if starts_with_keyword(c, "ring") {
                    c.bump("ring".len());
                    expect_keyword(c, "radius").map_err(|e| e.with_context("parsing vfx ring `radius`"))?;
                    let radius = parse_f64(c)?;
                    expect_keyword(c, "color").map_err(|e| e.with_context("parsing vfx ring `color`"))?;
                    let color = parse_color(c)?;
                    (VfxKindDecl::Ring, radius, color)
                } else if starts_with_keyword(c, "beam_to_nearest") {
                    c.bump("beam_to_nearest".len());
                    let target = parse_field_range(c)?;
                    expect_keyword(c, "color").map_err(|e| e.with_context("parsing vfx beam `color`"))?;
                    let color = parse_color(c)?;
                    (VfxKindDecl::BeamToNearest { target }, 0.0, color)
                } else {
                    return Err(ParseErr::at(here(c),
                        "expected `ring radius … color …` or `beam_to_nearest when … color …` in vfx body"));
                };
                expect_char(c, '}').map_err(|e| e.with_context("parsing vfx body `}`"))?;
                vfx.push(VfxDecl { on_rule, period, kind, radius, color });
            }
            other => {
                return Err(ParseErr::at(here(c), format!(
                    "unexpected `{}` in render block; expected arena_radius / camera / agent / vfx",
                    other.unwrap_or("<eof>"),
                )));
            }
        }
    }
    expect_char(c, '}').map_err(|e| e.with_context("parsing render body `}`"))?;
    let arena_radius = arena_radius.ok_or_else(|| {
        ParseErr::at(Span::new(start, c.pos), "render block missing required `arena_radius <n>`")
    })?;
    let camera = camera.ok_or_else(|| {
        ParseErr::at(Span::new(start, c.pos),
            "render block missing required `camera follow when … | observer`")
    })?;
    Ok(crate::ast::RenderDecl { arena_radius, camera, agents, vfx, span: Span::new(start, c.pos) })
}

/// ```text
/// ui {
///   hud {
///     bar "HP" value hp max hp_max color (220, 40, 40)
///     text "Lv {level}  Kills {kills}"
///   }
///   menu level_up "Level Up!" {
///     card "Bolt Damage +" -> bolt_level
///   }
///   screen dead "You Died" { summary time level kills  restart "Restart (R)" }
/// }
/// ```
fn ui_decl(c: &mut Cursor, start: usize) -> PResult<crate::ast::UiDecl> {
    use crate::ast::{UiCard, UiScreen, UiWidget};
    expect_keyword(c, "ui").map_err(|e| e.with_context("parsing `ui` block"))?;
    expect_char(c, '{').map_err(|e| e.with_context("parsing ui body `{`"))?;
    let mut hud: Vec<UiWidget> = Vec::new();
    let mut screens: Vec<UiScreen> = Vec::new();
    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            break;
        }
        match peek_ident(c).as_deref() {
            Some("hud") => {
                expect_keyword(c, "hud")?;
                expect_char(c, '{').map_err(|e| e.with_context("parsing hud body `{`"))?;
                loop {
                    c.skip_ws();
                    if c.starts_with_char('}') {
                        break;
                    }
                    match peek_ident(c).as_deref() {
                        Some("bar") => {
                            expect_keyword(c, "bar")?;
                            let label = string_lit(c).map_err(|e| e.with_context("parsing bar label"))?;
                            expect_keyword(c, "value").map_err(|e| e.with_context("parsing bar `value`"))?;
                            let value = ident(c).map_err(|e| e.with_context("parsing bar value key"))?;
                            expect_keyword(c, "max").map_err(|e| e.with_context("parsing bar `max`"))?;
                            let max = ident(c).map_err(|e| e.with_context("parsing bar max key"))?;
                            expect_keyword(c, "color").map_err(|e| e.with_context("parsing bar `color`"))?;
                            let color = parse_color(c)?;
                            hud.push(UiWidget::Bar { label, value, max, color });
                        }
                        Some("text") => {
                            expect_keyword(c, "text")?;
                            let template = string_lit(c).map_err(|e| e.with_context("parsing text template"))?;
                            hud.push(UiWidget::Text { template });
                        }
                        other => {
                            return Err(ParseErr::at(here(c), format!(
                                "unexpected `{}` in hud block; expected bar / text",
                                other.unwrap_or("<eof>"),
                            )));
                        }
                    }
                }
                expect_char(c, '}').map_err(|e| e.with_context("parsing hud body `}`"))?;
            }
            Some("menu") => {
                expect_keyword(c, "menu")?;
                let name = ident(c).map_err(|e| e.with_context("parsing menu name"))?;
                let title = string_lit(c).map_err(|e| e.with_context("parsing menu title"))?;
                expect_char(c, '{').map_err(|e| e.with_context("parsing menu body `{`"))?;
                let mut cards = Vec::new();
                loop {
                    c.skip_ws();
                    if c.starts_with_char('}') {
                        break;
                    }
                    expect_keyword(c, "card").map_err(|e| e.with_context("parsing menu `card`"))?;
                    let label = string_lit(c).map_err(|e| e.with_context("parsing card label"))?;
                    expect_str(c, "->").map_err(|e| e.with_context("parsing card `->`"))?;
                    let action_field = ident(c).map_err(|e| e.with_context("parsing card action field"))?;
                    cards.push(UiCard { label, action_field });
                }
                expect_char(c, '}').map_err(|e| e.with_context("parsing menu body `}`"))?;
                screens.push(UiScreen::Menu { name, title, cards });
            }
            Some("screen") => {
                expect_keyword(c, "screen")?;
                let name = ident(c).map_err(|e| e.with_context("parsing screen name"))?;
                let title = string_lit(c).map_err(|e| e.with_context("parsing screen title"))?;
                expect_char(c, '{').map_err(|e| e.with_context("parsing screen body `{`"))?;
                let mut summary: Vec<(String, String)> = Vec::new();
                let mut restart_label: Option<String> = None;
                loop {
                    c.skip_ws();
                    if c.starts_with_char('}') {
                        break;
                    }
                    if starts_with_keyword(c, "summary") {
                        c.bump("summary".len());
                        // One or more bare-ident summary keys until `restart` or `}`.
                        loop {
                            c.skip_ws();
                            if c.starts_with_char('}') || starts_with_keyword(c, "restart") {
                                break;
                            }
                            let key = ident(c).map_err(|e| e.with_context("parsing summary key"))?;
                            // Summary row label defaults to the key (host substitutes the value).
                            summary.push((key.clone(), key));
                        }
                    } else if starts_with_keyword(c, "restart") {
                        c.bump("restart".len());
                        restart_label = Some(string_lit(c).map_err(|e| e.with_context("parsing restart label"))?);
                    } else {
                        return Err(ParseErr::at(here(c),
                            "expected `summary <keys…>` or `restart \"label\"` in screen body"));
                    }
                }
                expect_char(c, '}').map_err(|e| e.with_context("parsing screen body `}`"))?;
                let restart_label = restart_label.unwrap_or_else(|| "Restart".to_string());
                screens.push(UiScreen::End { name, title, summary, restart_label });
            }
            other => {
                return Err(ParseErr::at(here(c), format!(
                    "unexpected `{}` in ui block; expected hud / menu / screen",
                    other.unwrap_or("<eof>"),
                )));
            }
        }
    }
    expect_char(c, '}').map_err(|e| e.with_context("parsing ui body `}`"))?;
    Ok(crate::ast::UiDecl { hud, screens, span: Span::new(start, c.pos) })
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
    // `apply_ability <expr>` — registry-driven dispatch (#132). The
    // expression resolves at runtime to an AbilityId; the WGSL emitter
    // expands this into a per-effect-slot dispatch loop reading from
    // PackedAbilityRegistry SoA columns.
    if c.starts_with("apply_ability ")
        || c.starts_with("apply_ability\t")
        || c.starts_with("apply_ability\n")
    {
        return Ok(Stmt::ApplyAbility(parse_apply_ability_stmt(c)?));
    }
    // `for_each_agent <binder> { <body> }` — body-shape primitive that
    // walks every alive agent slot in deterministic linear order. Must
    // be checked BEFORE `for ` (the keyword `for_each_agent` shares the
    // `for` prefix; without the longest-match-first ordering the bare
    // `for` arm would consume `for` and then choke on `_each_agent`).
    if c.starts_with("for_each_agent ")
        || c.starts_with("for_each_agent\t")
        || c.starts_with("for_each_agent\n")
    {
        c.bump("for_each_agent".len());
        c.skip_ws();
        let binder = ident(c)?;
        c.skip_ws();
        expect_char(c, '{')
            .map_err(|e| e.with_context("parsing `for_each_agent` body `{`"))?;
        let body = parse_stmt_block_until_close(c)?;
        c.skip_ws();
        expect_char(c, '}')
            .map_err(|e| e.with_context("parsing `for_each_agent` body `}`"))?;
        return Ok(Stmt::ForEachAgent {
            binder,
            body,
            span: Span::new(start, c.pos),
        });
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
            // `else if (cond) { ... }` sugar (Gap dungeon_stealth#4,
            // 2026-05-12): fold a bare `if` after `else` into the
            // equivalent `else { if (cond) { ... } [else ...] }`. The
            // recursive `parse_stmt` call consumes the entire if-stmt
            // (including any nested else/else-if tail), so chains of
            // arbitrary length compose without further special-casing.
            if c.starts_with("if ") {
                let nested_if = parse_stmt(c)?;
                else_body = Some(vec![nested_if]);
            } else {
                expect_char(c, '{').map_err(|e| e.with_context("parsing `else` body `{`"))?;
                else_body = Some(parse_stmt_block_until_close(c)?);
                c.skip_ws();
                expect_char(c, '}').map_err(|e| e.with_context("parsing `else` body `}`"))?;
            }
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
    if starts_with_keyword(c, "self") {
        // Check for `self += / -= / *= / |=` operators, OR
        // `self.append( field: expr, ... )` (Plan G G3b/G3c — struct
        // ring append).
        let save = c.pos;
        c.bump("self".len());
        c.skip_ws();
        // `|=` ordered before `=` so the lookahead `|` consumes both bytes
        // (a bare `=` would also match `|=`'s tail otherwise).
        for op in ["+=", "-=", "*=", "/=", "|=", "="] {
            if c.starts_with(op) {
                c.bump(op.len());
                c.skip_ws();
                let value = parse_expr(c)?;
                return Ok(Stmt::SelfUpdate { op: op.to_string(), value, span: Span::new(start, c.pos) });
            }
        }
        // `self.append(...)` — struct-payload ring append. Each arg is
        // a `name: <expr>` pair; the name list defines the per-cell
        // struct layout in declaration order.
        if c.starts_with_char('.') {
            let dot_save = c.pos;
            c.bump(1);
            c.skip_ws();
            if starts_with_keyword(c, "append") {
                c.bump("append".len());
                c.skip_ws();
                expect_char(c, '(')
                    .map_err(|e| e.with_context("parsing `self.append(` open paren"))?;
                let mut fields: Vec<FieldInit> = Vec::new();
                loop {
                    c.skip_ws();
                    if c.starts_with_char(')') {
                        c.bump(1);
                        break;
                    }
                    let fstart = c.pos;
                    let name = ident(c)
                        .map_err(|e| e.with_context("parsing self.append field name"))?;
                    c.skip_ws();
                    expect_char(c, ':').map_err(|e| {
                        e.with_context("parsing `:` after self.append field name")
                    })?;
                    c.skip_ws();
                    let value = parse_expr(c)?;
                    fields.push(FieldInit { name, value, span: Span::new(fstart, c.pos) });
                    c.skip_ws();
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
                        "expected `,` or `)` in self.append(...)",
                    ));
                }
                return Ok(Stmt::SelfAppend {
                    fields,
                    span: Span::new(start, c.pos),
                });
            }
            c.pos = dot_save;
        }
        c.pos = save;
    }
    // Fallback: a bare expression statement.
    let e = parse_expr(c)?;
    Ok(Stmt::Expr(e))
}

/// Parse an `apply_ability <expr> [by <c>] [target <t>]` statement
/// starting at the `apply_ability` keyword. Shared by physics-body
/// statements (via [`parse_stmt`]) and verb-body statements (via
/// [`parse_verb_body_stmt`]) so the surface stays in lock-step.
fn parse_apply_ability_stmt(c: &mut Cursor) -> PResult<crate::ast::ApplyAbilityStmt> {
    let start = c.pos;
    expect_keyword(c, "apply_ability")
        .map_err(|e| e.with_context("parsing `apply_ability` statement"))?;
    c.skip_ws();
    // Slice δ part 3 (#161): optional `by <caster_expr>` syntax for the
    // explicit caster operand. The ability operand parses up to the
    // first `by` keyword OR statement terminator.
    let ability = parse_expr_bounded(c, |ck| {
        ck.starts_with_char(';')
            || ck.starts_with_char('}')
            || ck.starts_with_char('\n')
            || ck.starts_with("by ")
            || ck.starts_with("by\t")
            || ck.starts_with("by\n")
            || ck.starts_with("target ")
            || ck.starts_with("target\t")
            || ck.starts_with("target\n")
    })?;
    // Symbolic ability-name surface (2026-05-12): when the parsed ability
    // operand is a bare PascalCase identifier (e.g. `Strike`, `Volley`),
    // capture it as a name for the lowerer to resolve against the
    // fixture's ability-name registry. Numeric (`apply_ability 3`),
    // lowercase locals (`apply_ability a` — local from `on Event { …: a }`),
    // and complex expressions (`agents.level(self)`, `self.action_ability`)
    // keep their existing expression-only path.
    //
    // The PascalCase gate matches the existing DSL convention:
    //   - Ability filenames are PascalCase (`Strike.ability`).
    //   - Lowercase bare-ident bindings name pattern locals
    //     (`on Triggered { who: w, ability_id: a } { apply_ability a … }`).
    //   - Reserved namespaces (`agents`, `world`, `config`, …) are lowercase
    //     so they never collide with the PascalCase gate.
    //
    // Closes the silent-mis-dispatch footgun documented in commit
    // 08cc223e (squad_skirmish): the registry sorts filenames
    // alphabetically, so `apply_ability 1` is NOT necessarily the
    // verb-author's "first" ability. With this surface, authors write
    // `apply_ability Strike` and the lowerer resolves the slot for them
    // (or surfaces a typed `UnknownAbilityName` error).
    let ability_name = if let ExprKind::Ident(name) = &ability.kind {
        if name
            .chars()
            .next()
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false)
        {
            Some(name.clone())
        } else {
            None
        }
    } else {
        None
    };
    c.skip_ws();
    let caster = if c.starts_with("by ") || c.starts_with("by\t") || c.starts_with("by\n") {
        c.bump("by".len());
        c.skip_ws();
        let caster_expr = parse_expr_bounded(c, |ck| {
            ck.starts_with_char(';')
                || ck.starts_with_char('}')
                || ck.starts_with_char('\n')
                || ck.starts_with("target ")
                || ck.starts_with("target\t")
                || ck.starts_with("target\n")
        })?;
        c.skip_ws();
        Some(caster_expr)
    } else {
        None
    };
    // Slice ε part 1: optional `target <expr>` clause — explicit target
    // operand for chronicle records that distinguish actor from target.
    let target = if c.starts_with("target ")
        || c.starts_with("target\t")
        || c.starts_with("target\n")
    {
        c.bump("target".len());
        c.skip_ws();
        let target_expr = parse_expr_bounded(c, |ck| {
            ck.starts_with_char(';') || ck.starts_with_char('}') || ck.starts_with_char('\n')
        })?;
        c.skip_ws();
        Some(target_expr)
    } else {
        None
    };
    if c.starts_with_char(';') {
        c.bump(1);
    }
    Ok(crate::ast::ApplyAbilityStmt {
        ability,
        ability_name,
        caster,
        target,
        span: Span::new(start, c.pos),
    })
}

fn parse_emit_stmt(c: &mut Cursor) -> PResult<EmitStmt> {
    let start = c.pos;
    expect_keyword(c, "emit").map_err(|e| e.with_context("parsing `emit` statement"))?;
    c.skip_ws();
    // Allow an empty `emit` clause inside verb bodies — e.g.
    //   verb Wait(self) =
    //     action Hold
    //     emit  // (none — passive action)
    //     score 0.5 + ...
    // Detect by looking at the next token; if it's a verb-clause
    // keyword (`score`, `scoring`, `when`) or another control keyword
    // that follows an `emit` slot, treat the emit as a no-op.
    if c.starts_with("score")
        || c.starts_with("scoring")
        || c.starts_with("when")
        || c.starts_with("apply_ability")
        || c.starts_with("emit")
    {
        // No event name; trailing keyword belongs to the next clause.
        return Ok(EmitStmt {
            event_name: String::new(),
            fields: Vec::new(),
            span: Span::new(start, c.pos),
        });
    }
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
        // Checkpoint BEFORE skip_ws so we can rewind when no operator
        // follows. Otherwise `parse_binary` for `candidate != self`
        // (in a `spatial_query <name>(...) = <expr>` filter) leaves
        // the cursor past the trailing newline + comments at the
        // start of the NEXT decl's `@annotation`, and the top-level
        // `absorb_trailing_annotations` then mis-attaches that
        // annotation to the spatial_query (it skips only inline ws,
        // so it sees `@` right at cursor and absorbs it as trailing).
        // Rewinding restores the cursor to right-after-LHS so the
        // top-level decl loop's parse_annotations picks up the @ as
        // a leading annotation on the next decl. Same fix applies to
        // every parse_expr caller whose decl boundary is line-based.
        let pre_ws = c.pos;
        c.skip_ws();
        if stop(c) {
            c.pos = pre_ws;
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
        // `&&`/`||` MUST be checked before single-char `&`/`|` so the
        // bitwise-op #159 lookups don't shadow the logical ops.
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
            else if c.starts_with_char('|') { Some("|") }
            else if c.starts_with_char('&') { Some("&") }
            else if c.starts_with_char('^') { Some("^") }
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
            // Treat `in` with the same precedence as comparison
            // (`<`/`<=`/`>`/`>=` — prec 4). #159 left this number
            // unchanged: bitwise ops were slotted between comparison
            // and arithmetic, not between logical and equality.
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
            // Comparison precedence; matches `in` (prec 4).
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
        // No operator matched at this level — rewind the leading
        // skip_ws so the cursor sits right after the LHS rather than
        // at the next non-op token. Critical for line-bounded callers
        // like `spatial_query <name>(...) = <expr>` where the next
        // line starts a fresh decl: leaving the cursor on the next
        // decl's `@annotation` triggers the trailing-annotation
        // absorber to mis-attach it. (The precedence-mismatch breaks
        // above do NOT rewind — a real operator IS at the cursor
        // and the outer recursion needs to see it.)
        c.pos = pre_ws;
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
        // `{ let <name> = <expr>; ... <final_expr> }` — block expression
        // form used by match-arm bodies and (future) view-body lets that
        // need local intermediates. The let bindings are parse-and-
        // discarded; the parser does no name resolution so the final
        // expression's references to bound names go through cleanly.
        // Lowering will see only the final expression. Semantic adoption
        // (carrying the bindings in the AST and inlining them) is a
        // future grammar slice.
        if open == '{' {
            let probe_pos = c.pos + 1;
            let mut probe = Cursor { src: c.src, pos: probe_pos };
            probe.skip_ws();
            if probe.starts_with("let ") || probe.starts_with("let\t") {
                c.bump(1); // consume `{`
                let mut bindings: Vec<(String, Expr)> = Vec::new();
                while {
                    c.skip_ws();
                    c.starts_with("let ") || c.starts_with("let\t")
                } {
                    c.bump("let".len());
                    c.skip_ws();
                    let bname = ident(c)?;
                    c.skip_ws();
                    expect_char(c, '=')
                        .map_err(|e| e.with_context("parsing block `let =`"))?;
                    c.skip_ws();
                    let value = parse_expr_bounded(
                        c,
                        |ck| ck.starts_with_char(';') || ck.starts_with_char('\n'),
                    )?;
                    bindings.push((bname, value));
                    c.skip_ws();
                    if c.starts_with_char(';') {
                        c.bump(1);
                    }
                    c.skip_ws();
                }
                // Final expression closing the block.
                let final_expr = parse_expr_bounded(
                    c,
                    |ck| ck.starts_with_char('}'),
                )?;
                c.skip_ws();
                let end = c.pos;
                expect_char(c, '}')
                    .map_err(|e| e.with_context("parsing block `}`"))?;
                let block = Expr {
                    kind: ExprKind::Block {
                        bindings,
                        expr: Box::new(final_expr),
                    },
                    span: Span::new(start, end),
                };
                return parse_postfix(c, block, stop);
            }
        }
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
        // Accept either a single binder ident or a parenthesised
        // tuple binder `(a, b, ...)`. Tuple binders are flattened to
        // their first element today (parse-and-discard the rest) —
        // good enough for the design-target probe form
        // `forall (run_a, run_b) in [(seed, seed)]: …` whose body
        // sits in the parse-and-discarded Raw assert path.
        let binder = if c.starts_with_char('(') {
            c.bump(1);
            c.skip_ws();
            let first = ident(c)?;
            c.skip_ws();
            while c.starts_with_char(',') {
                c.bump(1);
                c.skip_ws();
                let _next = ident(c)?;
                c.skip_ws();
            }
            expect_char(c, ')')
                .map_err(|e| e.with_context("parsing quantifier tuple-binder `)`"))?;
            first
        } else {
            ident(c)?
        };
        c.skip_ws();
        expect_keyword(c, "in").map_err(|e| e.with_context("parsing quantifier `in`"))?;
        c.skip_ws();
        let iter = parse_expr_bounded(c, |ck| {
            ck.starts_with_char(':') || ck.starts_with("where")
        })?;
        c.skip_ws();
        // Optional `where <pred>` between iter and body — parsed and
        // ANDed into the body. `forall e in xs where p(e): q(e)` is
        // semantically `forall e in xs: !p(e) || q(e)`. For the
        // parse-only path we just discard the predicate; semantic
        // adoption happens when invariants land in the GeneratedRuntime
        // path.
        if c.starts_with("where")
            && c.src[c.pos + "where".len()..]
                .chars()
                .next()
                .map_or(true, |ch| !is_ident_cont(ch))
        {
            c.bump("where".len());
            c.skip_ws();
            let _pred = parse_expr_bounded(c, |ck| ck.starts_with_char(':'))?;
            c.skip_ws();
        }
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
        ("mean", FoldKind::Mean),
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
                // Accept three forms:
                //   1. `<binder> in <iter> [where <body>]`        (filter-fold)
                //   2. `<expr> for <binder> in <iter>`            (generator-comprehension)
                //   3. `<expr>`                                   (single-arg fold)
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
                // Try generator-comprehension form: `<expr> for <binder> in <iter>`.
                // We probe by parsing an expression bounded at ` for ` / `for `.
                // If we land on `for <ident> in`, treat it as the comprehension form;
                // otherwise fall through to single-expr / call backtrack paths.
                let probe_save = c.pos;
                let mut probe = Cursor { src: c.src, pos: c.pos };
                let body_ok = parse_expr_bounded(
                    &mut probe,
                    |ck| ck.starts_with(" for ") || ck.starts_with("\nfor ") || ck.starts_with(")"),
                ).is_ok();
                if body_ok {
                    probe.skip_ws();
                    // Be careful: bare identifier `for` could collide with
                    // `for_each_agent` or other ident prefixes. Require it to
                    // be the keyword and followed by an ident + ` in `.
                    let is_for = probe.starts_with("for ") || probe.starts_with("for\t") || probe.starts_with("for\n");
                    if is_for {
                        // Re-parse the body with the real cursor up to ` for `.
                        let body = parse_expr_bounded(
                            c,
                            |ck| ck.starts_with(" for ") || ck.starts_with("\nfor "),
                        )?;
                        c.skip_ws();
                        // Consume `for`.
                        c.bump(3);
                        c.skip_ws();
                        let bname = ident(c)?;
                        c.skip_ws();
                        expect_keyword(c, "in").map_err(|e| e.with_context("parsing comprehension `in`"))?;
                        c.skip_ws();
                        let iter = parse_expr_bounded(c, |ck| {
                            ck.starts_with_char(')')
                                || ck.starts_with(" where ")
                                || ck.starts_with("\nwhere ")
                        })?;
                        c.skip_ws();
                        // Optional `where <pred>` filter on the
                        // comprehension's iter — design-target metric
                        // bodies use it (`mean(x for a in agents
                        // where a.kind == Wolf)`). Today we keep the
                        // body as the projection expression and
                        // discard the filter; the legacy `binder in
                        // iter where body` form already exists for
                        // when the filter IS the body.
                        if c.starts_with("where")
                            && c.src[c.pos + "where".len()..]
                                .chars()
                                .next()
                                .map_or(true, |ch| !is_ident_cont(ch))
                        {
                            c.bump("where".len());
                            c.skip_ws();
                            let _filter = parse_expr_bounded(c, |ck| ck.starts_with_char(')'))?;
                            c.skip_ws();
                        }
                        expect_char(c, ')').map_err(|e| e.with_context("parsing fold `)`"))?;
                        let span = Span::new(start, c.pos);
                        return Ok(Expr {
                            kind: ExprKind::Fold {
                                kind: fk,
                                binder: Some(bname),
                                iter: Some(Box::new(iter)),
                                body: Box::new(body),
                            },
                            span,
                        });
                    }
                }
                c.pos = probe_save;
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
    // Ctor-style call with `(`, or record-style with `{`. Rewind
    // the leading skip_ws when neither shape matches so a bare
    // identifier doesn't leave the cursor past trailing whitespace
    // + comments. Critical when the next decl starts on a new line
    // with `@annotation` — `absorb_trailing_annotations` would
    // otherwise mis-attach it as trailing on the current decl.
    let post_ident = c.pos;
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
                // `..` rest pattern in struct literal — the design-target
                // probe shape `NewGroupGoal { group: g, .. }` skips the
                // remaining fields. Parse-and-discarded today; the
                // surface just needs to round-trip cleanly.
                if c.starts_with("..") {
                    c.bump(2);
                    c.skip_ws();
                    if c.starts_with_char(',') {
                        c.bump(1);
                        continue;
                    }
                    if c.starts_with_char('}') {
                        c.bump(1);
                        break;
                    }
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
    // Bare identifier — no Ctor/Struct postfix matched. Restore the
    // cursor to the position right after the identifier (rewinding
    // the speculative skip_ws above) so the span ends cleanly and
    // any trailing whitespace/newline/comment is left for the outer
    // decl boundary to handle.
    c.pos = post_ident;
    let node = Expr { kind: ExprKind::Ident(name), span: Span::new(start, c.pos) };
    parse_postfix(c, node, stop)
}

fn parse_postfix(c: &mut Cursor, mut expr: Expr, stop: &dyn Fn(&Cursor) -> bool) -> PResult<Expr> {
    loop {
        // Same rewind discipline as `parse_binary`: checkpoint
        // before skip_ws so when no postfix matches we don't leave
        // the cursor past trailing newlines + comments. Otherwise
        // `parse_atom("self")` → parse_postfix skips past the
        // newline at the end of the expression and lands on the
        // next decl's `@annotation`, which then gets misattached
        // by `absorb_trailing_annotations`.
        let pre_ws = c.pos;
        c.skip_ws();
        if stop(c) {
            c.pos = pre_ws;
            break;
        }
        // `..` range operator (`0..500`, `start..end`). The result is
        // a Tuple holding `(lo, hi)` — the parser does no semantic
        // adoption today; consumers (`exists t in 0..500: …`) live in
        // the parse-and-discarded Raw assert path.
        if c.starts_with("..") {
            c.bump(2);
            c.skip_ws();
            let hi = parse_expr_bounded(c, |ck| {
                ck.starts_with_char(':')
                    || ck.starts_with_char(',')
                    || ck.starts_with_char(')')
                    || ck.starts_with_char(']')
                    || ck.starts_with_char('}')
            })?;
            let span = Span::new(expr.span.start, c.pos);
            expr = Expr {
                kind: ExprKind::Tuple(vec![expr, hi]),
                span,
            };
            continue;
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
        // `expr as <type>` cast — design-target invariant uses
        // `(config.hunt.predator_max_kills as f32)` to widen a u32
        // config field. Parsed as a no-op (the cast is dropped) so
        // the surrounding expression keeps its shape; semantic
        // adoption when explicit cast typing lands in the lowering
        // pipeline. Type names are restricted to identifiers so we
        // don't accidentally swallow following keywords.
        if c.starts_with("as ") || c.starts_with("as\t") || c.starts_with("as\n") {
            // Require the keyword to be a standalone token, not the
            // prefix of `assert` or similar.
            let after_as = c.pos + 2;
            let is_real_kw = c.src[after_as..]
                .chars()
                .next()
                .map_or(false, |ch| ch.is_whitespace());
            if is_real_kw {
                c.bump(2);
                c.skip_ws();
                let _ty = ident(c)?;
                continue;
            }
        }
        // `expr at <tick_expr>` — time-indexed view read used by
        // design-target probes (`a.alive at t`, `a.alive at t-1`).
        // Parse-and-discard the tick index; the surrounding assert
        // body sits in the Raw assert path so semantics are deferred.
        if c.starts_with("at ") || c.starts_with("at\t") || c.starts_with("at\n") {
            let after_at = c.pos + 2;
            let is_real_kw = c.src[after_at..]
                .chars()
                .next()
                .map_or(false, |ch| ch.is_whitespace());
            if is_real_kw {
                c.bump(2);
                c.skip_ws();
                // Stop at end-of-expression markers used by the
                // outer parsers — same boundary set as the `..`
                // postfix above.
                let _tick = parse_expr_bounded(c, |ck| {
                    ck.starts_with_char(':')
                        || ck.starts_with_char(',')
                        || ck.starts_with_char(')')
                        || ck.starts_with_char(']')
                        || ck.starts_with_char('}')
                        || ck.starts_with("&&")
                        || ck.starts_with("||")
                        || ck.starts_with("<=")
                        || ck.starts_with(">=")
                        || ck.starts_with("==")
                        || ck.starts_with("!=")
                        || ck.starts_with_char('<')
                        || ck.starts_with_char('>')
                })?;
                continue;
            }
        }
        // No postfix matched — rewind so the cursor doesn't sit
        // past trailing whitespace + comments. See the loop-top
        // comment for the rationale.
        c.pos = pre_ws;
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
    // Precedence ladder (low → high binding). Mirrors Rust's
    // operator-precedence table — bitwise ops slot BETWEEN comparison
    // (`==`/`<` etc., looser) and arithmetic (`+`/`*`, tighter), with
    // `&` tighter than `^` tighter than `|`. So:
    //   `a & MASK != 0`   parses as  `(a & MASK) != 0`
    //   `a & b && c`      parses as  `(a & b) && c`
    //   `a + b & MASK`    parses as  `(a + b) & MASK`
    // No need for parens around the bitset membership test, matching
    // the Rust convention. (#159)
    Some(match s {
        "||" => BinOpInfo { op: BinOp::Or,    prec: 1 },
        "&&" => BinOpInfo { op: BinOp::And,   prec: 2 },
        "==" => BinOpInfo { op: BinOp::Eq,    prec: 3 },
        "!=" => BinOpInfo { op: BinOp::NotEq, prec: 3 },
        "<"  => BinOpInfo { op: BinOp::Lt,    prec: 4 },
        "<=" => BinOpInfo { op: BinOp::LtEq,  prec: 4 },
        ">"  => BinOpInfo { op: BinOp::Gt,    prec: 4 },
        ">=" => BinOpInfo { op: BinOp::GtEq,  prec: 4 },
        "|"  => BinOpInfo { op: BinOp::BitOr,  prec: 5 },
        "^"  => BinOpInfo { op: BinOp::BitXor, prec: 6 },
        "&"  => BinOpInfo { op: BinOp::BitAnd, prec: 7 },
        "+"  => BinOpInfo { op: BinOp::Add,   prec: 8 },
        "-"  => BinOpInfo { op: BinOp::Sub,   prec: 8 },
        "*"  => BinOpInfo { op: BinOp::Mul,   prec: 9 },
        "/"  => BinOpInfo { op: BinOp::Div,   prec: 9 },
        "%"  => BinOpInfo { op: BinOp::Mod,   prec: 9 },
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
///
/// Surface forms (task #225 added hex + suffixes — 2026-05-08):
/// - Decimal: `123`, `1_000_000`, `1.5`, `2.5e-3`
/// - Hex: `0xFF`, `0xDEAD_BEEF` (any case, `_` separators allowed)
/// - Integer suffix: trailing `(u|i)(8|16|32|64)?` is consumed and
///   discarded — purely informational at this stage; lower assigns the
///   real type from the verb signature.
fn number_literal(c: &mut Cursor) -> PResult<(f64, bool)> {
    c.skip_ws();
    let start = c.pos;
    // Hex literal: `0x…`. Lex digits + `_`; no fractional / exponent /
    // suffix-other-than-int. Integer suffix still allowed at the tail.
    if c.starts_with("0x") || c.starts_with("0X") {
        c.bump(2);
        let digits_start = c.pos;
        while let Some(ch) = c.peek_char() {
            if ch.is_ascii_hexdigit() || ch == '_' { c.bump(1); } else { break; }
        }
        let raw = c.src[digits_start..c.pos].replace('_', "");
        if raw.is_empty() {
            return Err(ParseErr::at(
                Span::new(start, c.pos),
                "expected hex digits after `0x`",
            ));
        }
        let v = u64::from_str_radix(&raw, 16).map_err(|_| {
            ParseErr::at(
                Span::new(start, c.pos),
                format!("invalid hex literal `0x{raw}`"),
            )
        })?;
        consume_int_suffix(c);
        return Ok((v as f64, false));
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
    // Optional Rust-style integer suffix (`u8`, `i32`, bare `u`, …). Only
    // consumed for non-float lexemes; `1.0u8` is rejected by leaving the
    // suffix unconsumed (downstream sees it as garbage).
    if !is_float { consume_int_suffix(c); }
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

// ---------------------------------------------------------------------------
// terrain { extent, cell_size, seed_purpose, materials { ... }, layers { ... } }
// ---------------------------------------------------------------------------

fn parse_material_decl(c: &mut Cursor) -> PResult<crate::terrain::MaterialDecl> {
    c.skip_ws();
    let name_span = here(c);
    let name = ident(c).map_err(|e| e.with_context("parsing material entry name"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing material entry body (expected `{`)"))?;

    let mut id: Option<u8> = None;
    let mut walkable: Option<bool> = None;
    let mut hardness: Option<u8> = None;
    let mut biome_tag: Option<String> = None;
    let mut color: Option<u32> = None;
    let mut movement_cost: Option<f32> = None;

    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let prop_span = here(c);
        let prop = ident(c).map_err(|e| e.with_context("parsing material property name"))?;
        c.skip_ws();
        expect_char(c, ':')
            .map_err(|e| e.with_context("parsing material property (expected `:` after name)"))?;
        c.skip_ws();
        match prop.as_str() {
            "id" => {
                let num_span = here(c);
                let (v, is_float) = number_literal(c)
                    .map_err(|e| e.with_context("parsing material `id` value"))?;
                if is_float {
                    return Err(ParseErr::at(num_span, "material `id` must be an integer"));
                }
                if v < 0.0 || v > 255.0 {
                    return Err(ParseErr::at(num_span, "material id must be 1..=255"));
                }
                id = Some(v as u8);
            }
            "walkable" => {
                let bool_span = here(c);
                if starts_with_keyword(c, "true") {
                    c.bump("true".len());
                    walkable = Some(true);
                } else if starts_with_keyword(c, "false") {
                    c.bump("false".len());
                    walkable = Some(false);
                } else {
                    return Err(ParseErr::at(bool_span, "expected `true` or `false` for `walkable`"));
                }
            }
            "hardness" => {
                let num_span = here(c);
                let (v, is_float) = number_literal(c)
                    .map_err(|e| e.with_context("parsing material `hardness` value"))?;
                if is_float {
                    return Err(ParseErr::at(num_span, "material `hardness` must be an integer"));
                }
                if v < 0.0 || v > 255.0 {
                    return Err(ParseErr::at(num_span, "material `hardness` out of u8 range (0..=255)"));
                }
                hardness = Some(v as u8);
            }
            "biome_tag" => {
                let tag = ident(c).map_err(|e| e.with_context("parsing material `biome_tag` value"))?;
                biome_tag = Some(tag);
            }
            "color" => {
                let num_span = here(c);
                let (v, is_float) = number_literal(c)
                    .map_err(|e| e.with_context("parsing material `color` value"))?;
                if is_float {
                    return Err(ParseErr::at(num_span, "material `color` must be an integer"));
                }
                if v < 0.0 || v > u32::MAX as f64 {
                    return Err(ParseErr::at(num_span, "material `color` out of u32 range"));
                }
                color = Some(v as u32);
            }
            "movement_cost" => {
                let (v, _) = number_literal(c)
                    .map_err(|e| e.with_context("parsing material `movement_cost` value"))?;
                movement_cost = Some(v as f32);
            }
            other => {
                return Err(ParseErr::at(
                    prop_span,
                    format!(
                        "unknown material property `{other}`; expected one of id, walkable, hardness, biome_tag, color, movement_cost"
                    ),
                ));
            }
        }
        c.skip_ws();
        // optional trailing comma between properties
        if c.starts_with_char(',') {
            c.bump(1);
        }
    }

    let id_val = id.ok_or_else(|| {
        ParseErr::at(name_span, format!("material `{name}` missing required field `id`"))
    })?;
    // Validate id >= 1 (0 is reserved/invalid)
    if id_val == 0 {
        return Err(ParseErr::at(name_span, "material id must be 1..=255"));
    }
    let walkable_val = walkable.ok_or_else(|| {
        ParseErr::at(name_span, format!("material `{name}` missing required field `walkable`"))
    })?;
    let hardness_val = hardness.ok_or_else(|| {
        ParseErr::at(name_span, format!("material `{name}` missing required field `hardness`"))
    })?;
    let color_val = color.ok_or_else(|| {
        ParseErr::at(name_span, format!("material `{name}` missing required field `color`"))
    })?;

    Ok(crate::terrain::MaterialDecl {
        name,
        id: id_val,
        walkable: walkable_val,
        hardness: hardness_val,
        biome_tag,
        color: color_val,
        movement_cost: movement_cost.unwrap_or(1.0),
    })
}

fn parse_terrain(c: &mut Cursor) -> PResult<crate::terrain::TerrainBlock> {
    expect_keyword(c, "terrain")
        .map_err(|e| e.with_context("parsing `terrain` block"))?;
    c.skip_ws();
    expect_char(c, '{')
        .map_err(|e| e.with_context("parsing terrain body (expected `{`)"))?;

    let mut extent: Option<u32> = None;
    let mut cell_size: Option<f32> = None;
    let mut seed_purpose: Option<u32> = None;
    let mut materials: Vec<crate::terrain::MaterialDecl> = Vec::new();
    let mut layers: Vec<crate::terrain::LayerDecl> = Vec::new();

    loop {
        c.skip_ws();
        if c.starts_with_char('}') {
            c.bump(1);
            break;
        }
        let field_span = here(c);
        let field = ident(c).map_err(|e| e.with_context("parsing terrain field name"))?;
        c.skip_ws();
        match field.as_str() {
            "materials" => {
                // Sub-block: materials { <ident> { ... }* }
                expect_char(c, '{')
                    .map_err(|e| e.with_context("parsing `materials` block (expected `{`)"))?;
                loop {
                    c.skip_ws();
                    if c.starts_with_char('}') {
                        c.bump(1);
                        break;
                    }
                    let mat = parse_material_decl(c)
                        .map_err(|e| e.with_context("parsing material entry"))?;
                    materials.push(mat);
                    c.skip_ws();
                    // optional comma between entries
                    if c.starts_with_char(',') {
                        c.bump(1);
                    }
                }
                // Validate: no duplicate ids
                let mut seen_ids = std::collections::HashSet::new();
                for mat in &materials {
                    if !seen_ids.insert(mat.id) {
                        return Err(ParseErr::at(
                            field_span,
                            format!("duplicate material id {} in `materials` block", mat.id),
                        ));
                    }
                }
            }
            "layer" => {
                // `layer <kind> { ... }` — kind ident follows without colon
                let kind_span = here(c);
                let kind_name = ident(c).map_err(|e| e.with_context("parsing layer kind"))?;
                c.skip_ws();
                let kind = match kind_name.as_str() {
                    "fill" => {
                        expect_char(c, '{')
                            .map_err(|e| e.with_context("parsing `layer fill` body (expected `{`)"))?;
                        c.skip_ws();
                        // Only property: `material: <ident>`
                        let prop_span = here(c);
                        let prop = ident(c).map_err(|e| e.with_context("parsing `layer fill` property name"))?;
                        if prop != "material" {
                            return Err(ParseErr::at(
                                prop_span,
                                format!("unknown property `{prop}` in `layer fill`; expected `material`"),
                            ));
                        }
                        c.skip_ws();
                        expect_char(c, ':')
                            .map_err(|e| e.with_context("parsing `layer fill` material (expected `:`)"))?;
                        c.skip_ws();
                        let material = ident(c)
                            .map_err(|e| e.with_context("parsing `layer fill` material name"))?;
                        c.skip_ws();
                        // optional trailing comma inside block
                        if c.starts_with_char(',') {
                            c.bump(1);
                        }
                        c.skip_ws();
                        expect_char(c, '}')
                            .map_err(|e| e.with_context("parsing `layer fill` body (expected `}`)"))?;
                        crate::terrain::LayerKind::Fill { material }
                    }
                    other => {
                        return Err(ParseErr::at(
                            kind_span,
                            format!("unknown layer kind: {other}"),
                        ));
                    }
                };
                let index = layers.len() as u32 + 1;
                layers.push(crate::terrain::LayerDecl { index, kind });
            }
            _ => {
                // All scalar fields require `:` after the name
                expect_char(c, ':')
                    .map_err(|e| e.with_context("parsing terrain field (expected `:` after name)"))?;
                c.skip_ws();
                match field.as_str() {
                    "extent" => {
                        let num_span = here(c);
                        let (v, is_float) = number_literal(c)
                            .map_err(|e| e.with_context("parsing terrain extent value"))?;
                        if is_float {
                            return Err(ParseErr::at(num_span, "`extent` must be an integer"));
                        }
                        if v < 0.0 || v > u32::MAX as f64 {
                            return Err(ParseErr::at(num_span, "`extent` out of u32 range"));
                        }
                        extent = Some(v as u32);
                    }
                    "cell_size" => {
                        let (v, _) = number_literal(c)
                            .map_err(|e| e.with_context("parsing terrain cell_size value"))?;
                        cell_size = Some(v as f32);
                    }
                    "seed_purpose" => {
                        let num_span = here(c);
                        let (v, is_float) = number_literal(c)
                            .map_err(|e| e.with_context("parsing terrain seed_purpose value"))?;
                        if is_float {
                            return Err(ParseErr::at(num_span, "`seed_purpose` must be an integer"));
                        }
                        if v < 0.0 || v > u32::MAX as f64 {
                            return Err(ParseErr::at(num_span, "`seed_purpose` out of u32 range"));
                        }
                        let val = v as u32;
                        if val == 0 {
                            return Err(ParseErr::at(
                                num_span,
                                "`seed_purpose` must be non-zero",
                            ));
                        }
                        seed_purpose = Some(val);
                    }
                    other => {
                        return Err(ParseErr::at(
                            field_span,
                            format!(
                                "unknown terrain field `{other}`; expected one of extent, cell_size, seed_purpose, materials"
                            ),
                        ));
                    }
                }
                c.skip_ws();
                // optional trailing comma or newline-separated (no comma needed)
                if c.starts_with_char(',') {
                    c.bump(1);
                }
            }
        }
    }

    let extent = extent.ok_or_else(|| {
        ParseErr::at(here(c), "terrain block missing required field `extent`")
    })?;
    let cell_size = cell_size.ok_or_else(|| {
        ParseErr::at(here(c), "terrain block missing required field `cell_size`")
    })?;
    let seed_purpose = seed_purpose.ok_or_else(|| {
        ParseErr::at(here(c), "terrain block missing required field `seed_purpose`")
    })?;

    Ok(crate::terrain::TerrainBlock {
        extent,
        cell_size,
        seed_purpose,
        materials,
        layers,
    })
}
