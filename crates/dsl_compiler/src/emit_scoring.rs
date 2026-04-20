//! Scoring-table emission (milestone 5).
//!
//! Each `scoring { <entries> }` block in DSL becomes one or more rows in a
//! compiler-emitted `SCORING_TABLE` constant. Rows are POD `#[repr(C)]` so
//! the CPU scorer (`engine::policy::utility::UtilityBackend`) and the
//! future GPU kernel share the exact same layout. The engine's runtime
//! `ScoringEntry` / `ModifierRow` / `PredicateDescriptor` types live in
//! `engine_rules::scoring::types`; the emitter only produces the table
//! body.
//!
//! ## Emission target
//!
//! Everything lands in one file: `<out>/mod.rs`. The file carries both
//! the POD type definitions (hand-written Rust, but emitted literally so
//! mod.rs is the sole generated artefact — avoids an xtask
//! stale-pruning corner) and the `SCORING_TABLE` constant built from
//! DSL entries. Call sites import `engine_rules::scoring::{
//! ScoringEntry, ModifierRow, PredicateDescriptor, SCORING_TABLE}`.
//!
//! ## Supported expression shape
//!
//! At milestone 5 the emitter recognises:
//!
//! - Literal RHS: `<head> = 0.1` → base 0.1, no modifiers.
//! - Sum of literal + `if`: `0.0 + (if <pred> { <lit> } else { 0.0 })` →
//!   base 0.0, one modifier (pred → lit).
//! - Chained `+`: each trailing `if` adds a modifier row.
//!
//! Each predicate must be a `ScalarCompare`-shaped expression:
//! `<field_ref> <op> <lit>` where `<field_ref>` is `self.<ident>` and
//! `<op>` ∈ {`<`, `<=`, `==`, `>=`, `>`, `!=`}. Anything else raises
//! `EmitError::UnsupportedPredicate` and is out of scope for milestone 5.
//!
//! ## Field-id mapping
//!
//! `self.<ident>` → `field_id` — table in `docs/dsl/scoring_fields.md`.
//! The compiler and the engine's `read_field` dispatcher must agree; any
//! change bumps `SCORING_HASH`.

use std::fmt::Write;

use crate::ast::BinOp;
use crate::ir::{IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, ScoringEntryIR, ScoringIR, ViewIR, ViewRef};

/// Errors the emitter can raise. Propagated up to xtask which stamps a
/// diagnostic with the offending expression span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// The action head's name does not map to a known `MicroKind`. Extend
    /// the mapping (and the engine's `MicroKind` enum) if you need a new
    /// head.
    UnknownActionHead(String),
    /// The RHS expression doesn't match `lit | sum_of_lit_and_ifs`.
    UnsupportedExprShape(String),
    /// A modifier's condition doesn't match a `ScalarCompare` shape.
    UnsupportedPredicate(String),
    /// A modifier's delta (then-branch) isn't a float literal, or the
    /// else-branch isn't 0.0.
    UnsupportedModifierBody(String),
    /// Too many modifiers on a single entry; bump `MAX_MODIFIERS` if
    /// legitimately needed.
    TooManyModifiers {
        head: String,
        count: usize,
        max: usize,
    },
    /// View referenced by a scoring predicate has no VIEW_ID mapping.
    /// Only `@materialized` views with a compile-time `VIEW_ID_*`
    /// constant may be called from scoring predicates.
    UnsupportedView(String),
    /// View-call argument could not be lowered to an arg-slot (`self`,
    /// target binding, or `_` wildcard).
    UnsupportedViewArg(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::UnknownActionHead(n) => {
                write!(
                    f,
                    "scoring entry action head `{n}` does not map to a MicroKind"
                )
            }
            EmitError::UnsupportedExprShape(s) => {
                write!(f, "scoring expression shape not supported: {s}")
            }
            EmitError::UnsupportedPredicate(s) => {
                write!(f, "scoring predicate not supported: {s}")
            }
            EmitError::UnsupportedModifierBody(s) => {
                write!(f, "scoring modifier body not supported: {s}")
            }
            EmitError::TooManyModifiers { head, count, max } => {
                write!(
                    f,
                    "scoring entry `{head}` has {count} modifiers but MAX_MODIFIERS is {max}"
                )
            }
            EmitError::UnsupportedView(n) => {
                write!(
                    f,
                    "scoring predicate references view `{n}` which has no VIEW_ID mapping; only @materialized views with an assigned runtime id may be used in scoring predicates"
                )
            }
            EmitError::UnsupportedViewArg(desc) => {
                write!(
                    f,
                    "scoring view-call argument not supported: {desc}; expected `self`, the head's target binding, or the `_` sum-wildcard"
                )
            }
        }
    }
}

impl std::error::Error for EmitError {}

// ---------------------------------------------------------------------------
// Constants that mirror `engine_rules::scoring::types`.
// ---------------------------------------------------------------------------
//
// Duplicated as `usize` / `u8` literals here so the compiler doesn't take a
// dev-dep on `engine_rules`. Any drift between the two sides is caught by
// `engine_rules::scoring::types::MAX_MODIFIERS` being referenced in the
// emitted code — a rustc error at build time.

const MAX_MODIFIERS: usize = 8;
const PERSONALITY_DIMS: usize = 5;

// PredicateDescriptor kind discriminants.
const KIND_ALWAYS: u8 = 0;
const KIND_SCALAR_COMPARE: u8 = 1;
/// Gradient modifier: `score += expr * delta`. Emitted for the DSL form
/// `<expr> per_unit <delta>` (spec §3.4). The engine-side decoder
/// evaluates the gradient expression (compiled into a side-table
/// keyed by `field_id`), multiplies by `delta`, and adds the result to
/// the score.
const KIND_GRADIENT: u8 = 6;
/// View-call scalar compare: `score += (view_call <op> <lit>) ? delta : 0`.
/// `field_id` holds the VIEW_ID; payload[0..4] = threshold (f32 LE);
/// payload[4] / [5] = arg-slot codes; payload[6] = arg_count.
const KIND_VIEW_SCALAR_COMPARE: u8 = 7;
/// View-call gradient: `score += view_call * delta`. Same arg-slot
/// layout as `KIND_VIEW_SCALAR_COMPARE`; threshold bytes unused.
const KIND_VIEW_GRADIENT: u8 = 8;

// View-call arg-slot codes.
const ARG_SELF: u8 = 0;
const ARG_TARGET: u8 = 1;
const ARG_WILDCARD: u8 = 0xFE;
const ARG_NONE: u8 = 0xFF;

/// Runtime VIEW_ID mapping — must match the engine-side
/// `eval_view_call` dispatch in `crates/engine/src/policy/utility.rs`.
const VIEW_ID_THREAT_LEVEL: u16 = 0;
/// Memory-driven scoring — per-pair "this attacker has hit this observer"
/// grudge flag. See `assets/sim/views.sim::my_enemies`.
const VIEW_ID_MY_ENEMIES: u16 = 1;
/// Rout mechanic (task 167) — per-(observer, dead_kin) decayed fear bump
/// fed by `FearSpread` emitted when a same-species neighbour dies. See
/// `assets/sim/views.sim::kin_fear`.
const VIEW_ID_KIN_FEAR: u16 = 2;
/// Pack-hunt focus (task 169) — per-(observer, target) decayed beacon
/// fed by `PackAssist` emitted when a same-species neighbour commits an
/// engagement. See `assets/sim/views.sim::pack_focus`.
const VIEW_ID_PACK_FOCUS: u16 = 3;

// ScalarCompare operator discriminants. Keep aligned with
// `PredicateDescriptor::OP_*` in engine_rules.
const OP_LT: u8 = 0;
const OP_LE: u8 = 1;
const OP_EQ: u8 = 2;
const OP_GE: u8 = 3;
const OP_GT: u8 = 4;
const OP_NE: u8 = 5;

// ---------------------------------------------------------------------------
// Public emission entry points
// ---------------------------------------------------------------------------

/// Emit a single `scoring` block. Currently just flattens the block's
/// entries; the aggregate table is built by `emit_scoring_mod` which
/// concatenates every block's entries in declaration order.
///
/// We still produce a per-block file so the emission shape matches the
/// other kinds (physics, mask, entity). The per-block file just lists
/// comments noting which entries feed into the aggregate; the actual
/// `SCORING_TABLE` constant lives in the mod-level output.
pub fn emit_scoring(
    scoring: &ScoringIR,
    views: &[ViewIR],
    source_file: Option<&str>,
) -> Result<String, EmitError> {
    let mut out = String::new();
    emit_header(&mut out, source_file);
    writeln!(out, "// Per-block placeholder — the aggregate table in").unwrap();
    writeln!(
        out,
        "// `mod.rs` owns every entry. Keeping this file around"
    )
    .unwrap();
    writeln!(out, "// mirrors the other kinds' emission layout (physics,").unwrap();
    writeln!(
        out,
        "// mask, entity) so the xtask's file-bookkeeping logic"
    )
    .unwrap();
    writeln!(out, "// doesn't need a scoring-specific branch.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "// Entries contributed by this block:").unwrap();
    // Pre-validate every entry so errors surface here with a precise
    // diagnostic rather than at aggregation time. We drop the output; the
    // mod-level emitter re-runs the walk.
    for (i, entry) in scoring.entries.iter().enumerate() {
        let rows = lower_entry(entry, views)?;
        writeln!(
            out,
            "// - [{i}] head=`{}` base={} modifiers={}",
            entry.head.name,
            format_float(rows.base as f64),
            rows.modifier_count,
        )
        .unwrap();
    }
    Ok(out)
}

/// Emit the aggregate `<out>/mod.rs`. Reads every entry across every
/// `ScoringIR` block and emits one `SCORING_TABLE` constant plus
/// per-block module declarations so rustfmt has something to chew on.
pub fn emit_scoring_mod(blocks: &[ScoringIR], views: &[ViewIR]) -> String {
    // Fallible; panic on error — matches the style of the physics emitter.
    // xtask catches resolve-time errors before we get here, so an emission
    // failure is a compiler bug.
    let mut entries: Vec<LoweredEntry> = Vec::new();
    for (block_idx, block) in blocks.iter().enumerate() {
        for (entry_idx, e) in block.entries.iter().enumerate() {
            match lower_entry(e, views) {
                Ok(r) => entries.push(r),
                Err(err) => panic!(
                    "scoring emission failed for block {block_idx} entry {entry_idx} (`{}`): {err}",
                    e.head.name
                ),
            }
        }
    }

    let mut out = String::new();
    writeln!(out, "// GENERATED by dsl_compiler. Do not edit by hand.").unwrap();
    writeln!(
        out,
        "// Regenerate with `cargo run --bin xtask -- compile-dsl`."
    )
    .unwrap();
    writeln!(out).unwrap();

    // Per-block modules, so they show up on disk alongside mod.rs. Pruning
    // + rustfmt in xtask cleans up stale entries.
    for i in 0..blocks.len() {
        writeln!(out, "pub mod scoring_{i:03};").unwrap();
    }
    if !blocks.is_empty() {
        writeln!(out).unwrap();
    }

    emit_types_prelude(&mut out);

    if entries.is_empty() {
        writeln!(out).unwrap();
        writeln!(
            out,
            "/// Empty scoring table — no `scoring` declarations in scope."
        )
        .unwrap();
        writeln!(out, "pub const SCORING_TABLE: &[ScoringEntry] = &[];").unwrap();
        return out;
    }

    writeln!(out).unwrap();
    writeln!(
        out,
        "/// Compiler-emitted scoring table. One row per `scoring <head>`"
    )
    .unwrap();
    writeln!(
        out,
        "/// entry in DSL source. The CPU scorer iterates this in order;"
    )
    .unwrap();
    writeln!(
        out,
        "/// the future GPU kernel uploads it as a buffer verbatim."
    )
    .unwrap();
    writeln!(out, "pub const SCORING_TABLE: &[ScoringEntry] = &[").unwrap();
    for e in &entries {
        emit_entry_literal(&mut out, e);
    }
    writeln!(out, "];").unwrap();
    out
}

/// Emit the POD type definitions that the table uses. The types are
/// stable hand-written Rust; keeping them in the generated `mod.rs`
/// means there's exactly one file under `engine_rules/src/scoring/`
/// that changes on a compile-dsl run. See `docs/dsl/scoring_fields.md`
/// for the field-id mapping the descriptors reference.
fn emit_types_prelude(out: &mut String) {
    out.push_str(TYPES_PRELUDE);
}

const TYPES_PRELUDE: &str = r#"/// Maximum number of modifier rows per scoring entry. Fixed so the row is
/// `#[repr(C)]` with a known size.
pub const MAX_MODIFIERS: usize = 8;

/// Number of personality dimensions dot-producted with `personality_weights`.
pub const PERSONALITY_DIMS: usize = 5;

/// One row of the emitted scoring table. POD; laid out for CPU + GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ScoringEntry {
    /// Discriminator into `MicroKind` (engine-side).
    pub action_head: u16,
    /// Unconditional base score.
    pub base: f32,
    /// Dot-producted with the agent's personality vector.
    pub personality_weights: [f32; PERSONALITY_DIMS],
    /// Number of valid entries in `modifiers`.
    pub modifier_count: u8,
    /// Per-modifier predicate + delta, evaluated in order.
    pub modifiers: [ModifierRow; MAX_MODIFIERS],
}

impl ScoringEntry {
    pub const EMPTY: ScoringEntry = ScoringEntry {
        action_head: 0,
        base: 0.0,
        personality_weights: [0.0; PERSONALITY_DIMS],
        modifier_count: 0,
        modifiers: [ModifierRow::EMPTY; MAX_MODIFIERS],
    };
}

/// One modifier: predicate descriptor + delta.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ModifierRow {
    pub predicate: PredicateDescriptor,
    pub delta: f32,
}

impl ModifierRow {
    pub const EMPTY: ModifierRow = ModifierRow {
        predicate: PredicateDescriptor::ALWAYS,
        delta: 0.0,
    };
}

/// POD descriptor for a predicate. `kind` selects the shape; `payload`
/// is op-specific bytes (little-endian where numeric). See
/// `docs/dsl/scoring_fields.md` for the kind and op discriminants.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PredicateDescriptor {
    pub kind: u8,
    pub op: u8,
    pub field_id: u16,
    pub payload: [u8; 12],
}

impl PredicateDescriptor {
    pub const KIND_ALWAYS: u8 = 0;
    pub const KIND_SCALAR_COMPARE: u8 = 1;
    pub const KIND_ENUM_EQ: u8 = 2;
    pub const KIND_BIT_TEST: u8 = 3;
    pub const KIND_SET_MEMBERSHIP: u8 = 4;
    pub const KIND_PAIR_FIELD: u8 = 5;
    /// Gradient modifier — `score += expr * delta` rather than
    /// `score += (pred ? delta : 0)`. Opaque expression handle in
    /// `field_id`; the engine-side decoder reads a compiled scalar expr
    /// from a side-table keyed by `field_id`.
    pub const KIND_GRADIENT: u8 = 6;
    /// View-call scalar compare — `score += (view_call <op> threshold) ? delta : 0`.
    /// `field_id` holds the runtime VIEW_ID; `payload[0..4]` is the
    /// threshold (f32 LE), `payload[4]` / `[5]` are arg-slot codes
    /// (`ARG_SELF=0`, `ARG_TARGET=1`, `ARG_WILDCARD=0xFE`,
    /// `ARG_NONE=0xFF`), `payload[6]` is arg_count. Dispatched by
    /// `eval_view_call` in the engine-side scorer.
    pub const KIND_VIEW_SCALAR_COMPARE: u8 = 7;
    /// View-call gradient — `score += view_call * delta`. Same arg-slot
    /// layout as `KIND_VIEW_SCALAR_COMPARE`; `delta` is on the enclosing
    /// `ModifierRow.delta`. `payload[0..4]` is reserved (zeros).
    pub const KIND_VIEW_GRADIENT: u8 = 8;

    /// View-call arg-slot codes. Mirrored on the compiler side so a
    /// drift between the two lowerings is a rustc type error, not a
    /// silent-semantics regression.
    pub const ARG_SELF: u8 = 0;
    pub const ARG_TARGET: u8 = 1;
    pub const ARG_WILDCARD: u8 = 0xFE;
    pub const ARG_NONE: u8 = 0xFF;

    /// Runtime VIEW_IDs. Extend by adding a VIEW_ID_* constant + an
    /// engine-side `eval_view_call` arm + a VIEW_NAME_* entry in the
    /// compiler's emitter. @materialized views only; @lazy views are
    /// called inline at emit time and don't flow through the runtime
    /// dispatcher.
    pub const VIEW_ID_THREAT_LEVEL: u16 = 0;
    /// Memory-driven scoring — per-pair "this attacker has hit this
    /// observer" grudge flag. See `assets/sim/views.sim::my_enemies`.
    pub const VIEW_ID_MY_ENEMIES: u16 = 1;
    /// Rout mechanic (task 167) — per-(observer, dead_kin) decayed fear
    /// bump fed by `FearSpread` emitted when a same-species neighbour
    /// dies. See `assets/sim/views.sim::kin_fear`.
    pub const VIEW_ID_KIN_FEAR: u16 = 2;
    /// Pack-hunt focus (task 169) — per-(observer, target) decayed
    /// beacon fed by `PackAssist` emitted when a same-species neighbour
    /// commits an engagement. See `assets/sim/views.sim::pack_focus`.
    pub const VIEW_ID_PACK_FOCUS: u16 = 3;

    pub const OP_LT: u8 = 0;
    pub const OP_LE: u8 = 1;
    pub const OP_EQ: u8 = 2;
    pub const OP_GE: u8 = 3;
    pub const OP_GT: u8 = 4;
    pub const OP_NE: u8 = 5;

    pub const ALWAYS: PredicateDescriptor = PredicateDescriptor {
        kind: Self::KIND_ALWAYS,
        op: 0,
        field_id: 0,
        payload: [0; 12],
    };

    /// Build a `ScalarCompare` descriptor. Encodes the f32 threshold into
    /// the payload in little-endian so the GPU side decodes the same way.
    pub const fn scalar_compare(field_id: u16, op: u8, threshold: f32) -> Self {
        let bytes = threshold.to_le_bytes();
        let mut payload = [0u8; 12];
        payload[0] = bytes[0];
        payload[1] = bytes[1];
        payload[2] = bytes[2];
        payload[3] = bytes[3];
        PredicateDescriptor {
            kind: Self::KIND_SCALAR_COMPARE,
            op,
            field_id,
            payload,
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Header helper
// ---------------------------------------------------------------------------

fn emit_header(out: &mut String, source_file: Option<&str>) {
    match source_file {
        Some(path) => writeln!(out, "// GENERATED by dsl_compiler from {path}.").unwrap(),
        None => writeln!(out, "// GENERATED by dsl_compiler.").unwrap(),
    }
    writeln!(
        out,
        "// Edit the .sim source; rerun `cargo run --bin xtask -- compile-dsl`."
    )
    .unwrap();
    writeln!(out, "// Do not edit by hand.").unwrap();
    writeln!(out).unwrap();
}

// ---------------------------------------------------------------------------
// Lowered row representation (compiler-local; not the engine's struct)
// ---------------------------------------------------------------------------

/// Intermediate representation the emitter produces before stringifying.
/// Decoupled from `engine_rules::scoring::ScoringEntry` so the compiler
/// doesn't depend on engine_rules.
#[derive(Debug)]
pub(crate) struct LoweredEntry {
    head_name: String,
    action_head: u16,
    base: f32,
    personality_weights: [f32; PERSONALITY_DIMS],
    modifier_count: u8,
    modifiers: Vec<LoweredModifier>,
}

#[derive(Debug)]
pub(crate) struct LoweredModifier {
    kind: u8,
    op: u8,
    field_id: u16,
    /// Little-endian payload — mirrors the POD layout.
    payload: [u8; 12],
    delta: f32,
}

// ---------------------------------------------------------------------------
// Entry lowering
// ---------------------------------------------------------------------------

fn lower_entry(entry: &ScoringEntryIR, views: &[ViewIR]) -> Result<LoweredEntry, EmitError> {
    let action_head = action_head_discriminant(&entry.head)?;
    // Target-bound heads (Attack / MoveToward) expose the candidate target as
    // a positional local — `Attack(t)` binds `t`, `Attack(target)` binds
    // `target`. The first positional name becomes the recognised
    // target-reference prefix in modifier predicates (`<binding>.<field>`),
    // emitted with the `0x4000 | self_field_id` target-side range.
    let target_binding: Option<&str> = match &entry.head.shape {
        IrActionHeadShape::Positional(binds) if !binds.is_empty() => {
            Some(binds[0].0.as_str())
        }
        _ => None,
    };
    let (base, terms) = flatten_sum(&entry.expr)?;

    if terms.len() > MAX_MODIFIERS {
        return Err(EmitError::TooManyModifiers {
            head: entry.head.name.clone(),
            count: terms.len(),
            max: MAX_MODIFIERS,
        });
    }

    let mut modifiers = Vec::with_capacity(terms.len());
    for term in terms {
        match term {
            SumTerm::BoolIf { cond, delta } => {
                modifiers.push(lower_modifier(cond, delta, target_binding, views)?);
            }
            SumTerm::Gradient { expr, delta } => {
                // Recognise `view::<name>(args...)` gradients — they emit
                // `KIND_VIEW_GRADIENT` so the runtime evaluates the view
                // to produce the scalar the delta multiplies. Any other
                // gradient-expression shape falls through to the v1
                // `KIND_GRADIENT` placeholder (side-table emitter is a
                // future milestone).
                if let IrExpr::ViewCall(vref, ir_args) = &expr.kind {
                    let vname = view_name(*vref, views);
                    let view_id = view_id_for(vname)?;
                    let (slots, count) = lower_view_args(ir_args, target_binding)?;
                    let mut payload = [0u8; 12];
                    payload[4] = slots[0];
                    payload[5] = slots[1];
                    payload[6] = count;
                    modifiers.push(LoweredModifier {
                        kind: KIND_VIEW_GRADIENT,
                        op: 0,
                        field_id: view_id,
                        payload,
                        delta,
                    });
                } else {
                    modifiers.push(LoweredModifier {
                        kind: KIND_GRADIENT,
                        op: 0,
                        field_id: 0,
                        payload: [0u8; 12],
                        delta,
                    });
                }
            }
        }
    }

    let modifier_count = modifiers.len() as u8;

    Ok(LoweredEntry {
        head_name: entry.head.name.clone(),
        action_head,
        base,
        personality_weights: [0.0; PERSONALITY_DIMS],
        modifier_count,
        modifiers,
    })
}

/// Map a scoring action head name onto its `MicroKind` discriminant value.
/// The emitter owns this mapping so the engine doesn't need a string-keyed
/// dispatch table. Kept in sync with `engine::mask::MicroKind` by hand —
/// adding a new head requires both a `MicroKind` variant and a row here.
fn action_head_discriminant(head: &IrActionHead) -> Result<u16, EmitError> {
    // MicroKind discriminants — values match `crates/engine/src/mask.rs`.
    // Keep this match exhaustive so unrecognised heads can't silently
    // become `Hold`.
    let v: u16 = match head.name.as_str() {
        "Hold" => 0,
        "MoveToward" => 1,
        "Flee" => 2,
        "Attack" => 3,
        "Cast" => 4,
        "UseItem" => 5,
        "Harvest" => 6,
        "Eat" => 7,
        "Drink" => 8,
        "Rest" => 9,
        "PlaceTile" => 10,
        "PlaceVoxel" => 11,
        "HarvestVoxel" => 12,
        "Converse" => 13,
        "ShareStory" => 14,
        "Communicate" => 15,
        "Ask" => 16,
        "Remember" => 17,
        other => return Err(EmitError::UnknownActionHead(other.to_string())),
    };
    Ok(v)
}

// ---------------------------------------------------------------------------
// Expression-shape recognition
// ---------------------------------------------------------------------------

/// One term of a flattened scoring expression sum. `Attack(t) =
/// 0.5 + (if ... { ... } else { 0.0 }) + (expr per_unit 0.3)` decomposes
/// into a base literal and a list of these.
enum SumTerm<'a> {
    /// `if <pred> { <lit> } else { 0.0 }` — boolean modifier (existing shape).
    BoolIf { cond: &'a IrExprNode, delta: f32 },
    /// `<expr> per_unit <delta>` — gradient modifier (spec §3.4). `expr`
    /// is an opaque handle at v1; the engine-side reads the compiled
    /// scalar from a future side-table keyed by the entry's row index.
    Gradient {
        #[allow(dead_code)]
        expr: &'a IrExprNode,
        delta: f32,
    },
}

/// Flatten a top-level `+` tree into a base literal plus a list of sum
/// terms. Any leaf that isn't a literal, an `if <cond> { lit } else { 0.0 }`,
/// or an `<expr> per_unit <delta>` is unsupported.
fn flatten_sum<'a>(expr: &'a IrExprNode) -> Result<(f32, Vec<SumTerm<'a>>), EmitError> {
    let mut base: f32 = 0.0;
    let mut terms: Vec<SumTerm<'a>> = Vec::new();
    let mut seen_base = false;

    collect_sum(expr, &mut base, &mut seen_base, &mut terms)?;
    Ok((base, terms))
}

fn collect_sum<'a>(
    expr: &'a IrExprNode,
    base: &mut f32,
    seen_base: &mut bool,
    terms: &mut Vec<SumTerm<'a>>,
) -> Result<(), EmitError> {
    match &expr.kind {
        IrExpr::Binary(BinOp::Add, lhs, rhs) => {
            collect_sum(lhs, base, seen_base, terms)?;
            collect_sum(rhs, base, seen_base, terms)?;
            Ok(())
        }
        IrExpr::If { cond, then_expr, else_expr } => {
            // Modifier: `if cond { lit } else { 0.0 }`.
            let then_v = lit_float(then_expr).ok_or_else(|| {
                EmitError::UnsupportedModifierBody(
                    "then-branch of conditional modifier must be a float literal".into(),
                )
            })?;
            match else_expr.as_deref() {
                Some(eb) => {
                    let ev = lit_float(eb).ok_or_else(|| {
                        EmitError::UnsupportedModifierBody(
                            "else-branch of conditional modifier must be a float literal".into(),
                        )
                    })?;
                    if ev != 0.0 {
                        return Err(EmitError::UnsupportedModifierBody(
                            "else-branch must be 0.0 (non-zero base belongs in the sum's base term)".into(),
                        ));
                    }
                }
                None => {
                    return Err(EmitError::UnsupportedModifierBody(
                        "`if` modifier requires an explicit `else { 0.0 }` arm".into(),
                    ));
                }
            }
            terms.push(SumTerm::BoolIf { cond, delta: then_v });
            Ok(())
        }
        IrExpr::PerUnit { expr: gradient_expr, delta } => {
            // Gradient modifier: `<expr> per_unit <delta>`. Delta must be
            // a float literal at v1 — variable deltas are a later milestone.
            let d = lit_float(delta).ok_or_else(|| {
                EmitError::UnsupportedModifierBody(
                    "`per_unit <delta>` requires a float literal delta".into(),
                )
            })?;
            terms.push(SumTerm::Gradient { expr: gradient_expr, delta: d });
            Ok(())
        }
        IrExpr::LitFloat(v) => {
            // Base literal. We add so multiple bases (unusual but legal) sum
            // together; `seen_base` just guards diagnostics.
            *base += *v as f32;
            *seen_base = true;
            Ok(())
        }
        IrExpr::LitInt(v) => {
            *base += *v as f32;
            *seen_base = true;
            Ok(())
        }
        other => Err(EmitError::UnsupportedExprShape(format!(
            "expected literal, `if <pred> {{ <lit> }} else {{ 0.0 }}`, or `<expr> per_unit <delta>` (or `+` of those); got {other:?}"
        ))),
    }
}

fn lit_float(expr: &IrExprNode) -> Option<f32> {
    match &expr.kind {
        IrExpr::LitFloat(v) => Some(*v as f32),
        IrExpr::LitInt(v) => Some(*v as f32),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Modifier lowering: predicate descriptor + delta
// ---------------------------------------------------------------------------

fn lower_modifier(
    cond: &IrExprNode,
    delta: f32,
    target_binding: Option<&str>,
    views: &[ViewIR],
) -> Result<LoweredModifier, EmitError> {
    // Recognise:
    // - `<self.<field>> <op> <lit>` — classic scalar compare.
    // - `<target.<field>> <op> <lit>` — target-side scalar compare on
    //   target-bound heads (`[0x4000, 0x8000)` id range).
    // - `view::<name>(args...) <op> <lit>` — view-call scalar compare.
    //   Runtime evaluates the view, compares against the literal.
    let (op, lhs, rhs) = match &cond.kind {
        IrExpr::Binary(op, lhs, rhs) => {
            let op_tag = binop_scalar_compare(*op).ok_or_else(|| {
                EmitError::UnsupportedPredicate(format!(
                    "operator {op:?} is not a scalar-compare operator"
                ))
            })?;
            (op_tag, lhs.as_ref(), rhs.as_ref())
        }
        _ => {
            return Err(EmitError::UnsupportedPredicate(format!(
                "only `<field> <op> <lit>` or `view::<name>(args) <op> <lit>` shape is recognised; got {:?}",
                cond.kind
            )))
        }
    };

    // View-call scalar compare. Runs before the field-ref branch so a
    // view call never gets mis-classified as an unsupported field.
    if let Some((view_id, slots, count, op_tag, threshold)) =
        try_view_compare(op, lhs, rhs, target_binding, views)?
    {
        let mut payload = [0u8; 12];
        let tb = threshold.to_le_bytes();
        payload[0..4].copy_from_slice(&tb);
        payload[4] = slots[0];
        payload[5] = slots[1];
        payload[6] = count;
        return Ok(LoweredModifier {
            kind: KIND_VIEW_SCALAR_COMPARE,
            op: op_tag,
            field_id: view_id,
            payload,
            delta,
        });
    }

    // Classic field scalar compare.
    let (final_op, field_id, threshold) = if let Some(fid) = try_field_ref(lhs, target_binding) {
        let t = lit_float(rhs).ok_or_else(|| {
            EmitError::UnsupportedPredicate(
                "RHS of scalar compare must be a float literal".into(),
            )
        })?;
        (op, fid, t)
    } else if let Some(fid) = try_field_ref(rhs, target_binding) {
        let t = lit_float(lhs).ok_or_else(|| {
            EmitError::UnsupportedPredicate(
                "one side of scalar compare must be a float literal".into(),
            )
        })?;
        (flip_op(op), fid, t)
    } else {
        return Err(EmitError::UnsupportedPredicate(
            "scalar compare must reference `self.<field>`, the head's target binding, or a `view::<name>(args)` call on one side".into(),
        ));
    };

    let mut payload = [0u8; 12];
    let bytes = threshold.to_le_bytes();
    payload[0..4].copy_from_slice(&bytes);

    Ok(LoweredModifier {
        kind: KIND_SCALAR_COMPARE,
        op: final_op,
        field_id,
        payload,
        delta,
    })
}

/// Try to recognise `view::<name>(args) <op> <lit>` (or mirror). Returns
/// `Ok(None)` when neither side of the compare is a view call; `Ok(Some(_))`
/// when one side is a view call and the other is a literal; `Err(_)` when
/// the view is not in the runtime VIEW_ID table or an arg-slot cannot be
/// resolved.
fn try_view_compare(
    op: u8,
    lhs: &IrExprNode,
    rhs: &IrExprNode,
    target_binding: Option<&str>,
    views: &[ViewIR],
) -> Result<Option<(u16, [u8; 2], u8, u8, f32)>, EmitError> {
    if let IrExpr::ViewCall(vref, ir_args) = &lhs.kind {
        let threshold = lit_float(rhs).ok_or_else(|| {
            EmitError::UnsupportedPredicate(
                "RHS of view-call scalar compare must be a float literal".into(),
            )
        })?;
        let vname = view_name(*vref, views);
        let view_id = view_id_for(vname)?;
        let (slots, count) = lower_view_args(ir_args, target_binding)?;
        return Ok(Some((view_id, slots, count, op, threshold)));
    }
    if let IrExpr::ViewCall(vref, ir_args) = &rhs.kind {
        let threshold = lit_float(lhs).ok_or_else(|| {
            EmitError::UnsupportedPredicate(
                "LHS of view-call scalar compare must be a float literal when the view call is on the right".into(),
            )
        })?;
        let vname = view_name(*vref, views);
        let view_id = view_id_for(vname)?;
        let (slots, count) = lower_view_args(ir_args, target_binding)?;
        return Ok(Some((view_id, slots, count, flip_op(op), threshold)));
    }
    Ok(None)
}

/// Look up the source-level view name for a `ViewRef`.
fn view_name(view_ref: ViewRef, views: &[ViewIR]) -> &str {
    views
        .get(view_ref.0 as usize)
        .map(|v| v.name.as_str())
        .unwrap_or("?")
}

/// Map a view name to its runtime VIEW_ID. Only `@materialized` views
/// with an assigned id are callable from scoring.
fn view_id_for(name: &str) -> Result<u16, EmitError> {
    match name {
        "threat_level" => Ok(VIEW_ID_THREAT_LEVEL),
        "my_enemies" => Ok(VIEW_ID_MY_ENEMIES),
        "kin_fear" => Ok(VIEW_ID_KIN_FEAR),
        "pack_focus" => Ok(VIEW_ID_PACK_FOCUS),
        other => Err(EmitError::UnsupportedView(other.to_string())),
    }
}

/// Lower a view call's positional args (max 2) into arg-slot codes.
/// Accepted arg forms: bare local resolving to `self`, the head's
/// target binding, or `_` (sum-wildcard).
fn lower_view_args(
    ir_args: &[IrCallArg],
    target_binding: Option<&str>,
) -> Result<([u8; 2], u8), EmitError> {
    if ir_args.len() > 2 {
        return Err(EmitError::UnsupportedViewArg(format!(
            "view call has {} args; only 0-2 positional args supported",
            ir_args.len()
        )));
    }
    let mut slots = [ARG_NONE; 2];
    for (i, arg) in ir_args.iter().enumerate() {
        slots[i] = view_arg_slot(&arg.value, target_binding)?;
    }
    Ok((slots, ir_args.len() as u8))
}

fn view_arg_slot(
    expr: &IrExprNode,
    target_binding: Option<&str>,
) -> Result<u8, EmitError> {
    match &expr.kind {
        IrExpr::Local(_, name) if name == "self" => Ok(ARG_SELF),
        IrExpr::Local(_, name) if name == "_" => Ok(ARG_WILDCARD),
        IrExpr::Local(_, name) => match target_binding {
            Some(tb) if name == tb => Ok(ARG_TARGET),
            _ => Err(EmitError::UnsupportedViewArg(format!(
                "bare local `{name}` is not `self`, the head's target binding, or `_`"
            ))),
        },
        other => Err(EmitError::UnsupportedViewArg(format!(
            "expected bare local (`self`, target binding, or `_`); got {other:?}"
        ))),
    }
}

/// Reserved `field_id` range for target-side fields. Mirrors the engine's
/// `read_field` dispatch (`crates/engine/src/policy/utility.rs`) and the
/// reservation spelled out in `docs/dsl/scoring_fields.md`. Self-side ids
/// use the low range (0..8); OR-ing `TARGET_FIELD_BASE` promotes a
/// self-side id to its target-side counterpart.
const TARGET_FIELD_BASE: u16 = 0x4000;

/// Return `Some(field_id)` when `expr` is a field access on either `self`
/// or the action head's target binding. For self references we return the
/// low self-side id (`scoring_field_id`); for target references we OR in
/// `TARGET_FIELD_BASE` so the engine dispatches through the target-side
/// `read_field` branch. `target_binding` is `None` on self-only heads
/// (Hold, Flee, Eat, …) — any `<binding>.<field>` on those raises the
/// standard "must reference self" error up the stack.
fn try_field_ref(expr: &IrExprNode, target_binding: Option<&str>) -> Option<u16> {
    match &expr.kind {
        IrExpr::Field {
            base, field_name, ..
        } => match &base.kind {
            IrExpr::Local(_, name) if name == "self" => scoring_field_id(field_name),
            IrExpr::Local(_, name) => match target_binding {
                Some(tb) if name == tb => {
                    scoring_field_id(field_name).map(|fid| fid | TARGET_FIELD_BASE)
                }
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// DSL field name → numeric id. See `docs/dsl/scoring_fields.md` for the
/// canonical table. The mapping is shared between self-side (raw id) and
/// target-side reads (id OR-ed with `TARGET_FIELD_BASE`). Only the fields
/// the engine's target-side branch also dispatches on (hp, max_hp, hp_pct,
/// shield_hp) should be used with a target reference today; the other ids
/// (attack_range, hunger, thirst, fatigue) are self-only until the engine
/// grows matching target-side accessors.
fn scoring_field_id(name: &str) -> Option<u16> {
    match name {
        "hp" => Some(0),
        "max_hp" => Some(1),
        "hp_pct" => Some(2),
        "shield_hp" => Some(3),
        "attack_range" => Some(4),
        "hunger" => Some(5),
        "thirst" => Some(6),
        "fatigue" => Some(7),
        _ => None,
    }
}

fn binop_scalar_compare(op: BinOp) -> Option<u8> {
    Some(match op {
        BinOp::Lt => OP_LT,
        BinOp::LtEq => OP_LE,
        BinOp::Eq => OP_EQ,
        BinOp::GtEq => OP_GE,
        BinOp::Gt => OP_GT,
        BinOp::NotEq => OP_NE,
        _ => return None,
    })
}

/// Swap the operator when the comparison was written field-on-the-right.
fn flip_op(op: u8) -> u8 {
    match op {
        OP_LT => OP_GT,
        OP_LE => OP_GE,
        OP_GE => OP_LE,
        OP_GT => OP_LT,
        OP_EQ => OP_EQ,
        OP_NE => OP_NE,
        _ => op,
    }
}

// ---------------------------------------------------------------------------
// Rust-source emission
// ---------------------------------------------------------------------------

fn emit_entry_literal(out: &mut String, e: &LoweredEntry) {
    writeln!(
        out,
        "    // head=`{}` (MicroKind discriminant {})",
        e.head_name, e.action_head
    )
    .unwrap();
    writeln!(out, "    ScoringEntry {{").unwrap();
    writeln!(out, "        action_head: {},", e.action_head).unwrap();
    writeln!(out, "        base: {},", format_float(e.base as f64)).unwrap();
    writeln!(
        out,
        "        personality_weights: [{}],",
        e.personality_weights
            .iter()
            .map(|v| format_float(*v as f64))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(out, "        modifier_count: {},", e.modifier_count).unwrap();
    writeln!(out, "        modifiers: [").unwrap();
    for m in &e.modifiers {
        emit_modifier_literal(out, m);
    }
    // Pad to MAX_MODIFIERS with EMPTY.
    for _ in e.modifiers.len()..MAX_MODIFIERS {
        writeln!(out, "            ModifierRow::EMPTY,").unwrap();
    }
    writeln!(out, "        ],").unwrap();
    writeln!(out, "    }},").unwrap();
}

fn emit_modifier_literal(out: &mut String, m: &LoweredModifier) {
    if m.kind == KIND_ALWAYS {
        // Short form fits within rustfmt's default width budget.
        writeln!(
            out,
            "            ModifierRow {{ predicate: PredicateDescriptor::ALWAYS, delta: {} }},",
            format_float(m.delta as f64)
        )
        .unwrap();
        return;
    }
    if m.kind == KIND_SCALAR_COMPARE {
        // Decode the threshold from payload so the emitted source reads
        // like `scalar_compare(field, op, 0.5)` rather than a byte array.
        let mut tbytes = [0u8; 4];
        tbytes.copy_from_slice(&m.payload[0..4]);
        let threshold = f32::from_le_bytes(tbytes);
        // Multi-line form so rustfmt doesn't want to reshape it (the
        // single-line form blows past the 100-col budget once predicate
        // descriptors carry realistic delta / field-id bytes).
        writeln!(out, "            ModifierRow {{").unwrap();
        writeln!(
            out,
            "                predicate: PredicateDescriptor::scalar_compare({}, PredicateDescriptor::{}, {}),",
            m.field_id,
            op_const_name(m.op),
            format_float(threshold as f64),
        )
        .unwrap();
        writeln!(
            out,
            "                delta: {},",
            format_float(m.delta as f64)
        )
        .unwrap();
        writeln!(out, "            }},").unwrap();
        return;
    }
    if m.kind == KIND_GRADIENT {
        // Gradient modifier row. The `field_id` is a future side-table
        // index; v1 emits a placeholder `0` descriptor so the engine-side
        // `KIND_GRADIENT` dispatcher can load the compiled gradient expr
        // via `field_id` once the side-table lands.
        writeln!(out, "            ModifierRow {{").unwrap();
        writeln!(
            out,
            "                predicate: PredicateDescriptor {{ kind: PredicateDescriptor::KIND_GRADIENT, op: 0, field_id: {}, payload: [0; 12] }},",
            m.field_id
        )
        .unwrap();
        writeln!(
            out,
            "                delta: {},",
            format_float(m.delta as f64)
        )
        .unwrap();
        writeln!(out, "            }},").unwrap();
        return;
    }
    if m.kind == KIND_VIEW_SCALAR_COMPARE {
        let payload_str = format!(
            "[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]",
            m.payload[0], m.payload[1], m.payload[2], m.payload[3],
            m.payload[4], m.payload[5], m.payload[6], m.payload[7],
            m.payload[8], m.payload[9], m.payload[10], m.payload[11],
        );
        writeln!(out, "            ModifierRow {{").unwrap();
        writeln!(
            out,
            "                predicate: PredicateDescriptor {{ kind: PredicateDescriptor::KIND_VIEW_SCALAR_COMPARE, op: PredicateDescriptor::{}, field_id: {}, payload: {} }},",
            op_const_name(m.op),
            m.field_id,
            payload_str,
        )
        .unwrap();
        writeln!(
            out,
            "                delta: {},",
            format_float(m.delta as f64)
        )
        .unwrap();
        writeln!(out, "            }},").unwrap();
        return;
    }
    if m.kind == KIND_VIEW_GRADIENT {
        let payload_str = format!(
            "[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]",
            m.payload[0], m.payload[1], m.payload[2], m.payload[3],
            m.payload[4], m.payload[5], m.payload[6], m.payload[7],
            m.payload[8], m.payload[9], m.payload[10], m.payload[11],
        );
        writeln!(out, "            ModifierRow {{").unwrap();
        writeln!(
            out,
            "                predicate: PredicateDescriptor {{ kind: PredicateDescriptor::KIND_VIEW_GRADIENT, op: 0, field_id: {}, payload: {} }},",
            m.field_id,
            payload_str,
        )
        .unwrap();
        writeln!(
            out,
            "                delta: {},",
            format_float(m.delta as f64)
        )
        .unwrap();
        writeln!(out, "            }},").unwrap();
        return;
    }
    // Other kinds land when the predicate taxonomy grows. Emitter coverage
    // lives in `lower_modifier`; unreachable here.
    unreachable!("emit_modifier_literal: unsupported kind {}", m.kind);
}

fn op_const_name(op: u8) -> &'static str {
    match op {
        OP_LT => "OP_LT",
        OP_LE => "OP_LE",
        OP_EQ => "OP_EQ",
        OP_GE => "OP_GE",
        OP_GT => "OP_GT",
        OP_NE => "OP_NE",
        _ => "OP_EQ", // unreachable via lower_modifier
    }
}

/// Render a float so rustc parses it as `f32` (no integer inference drift).
/// Values that round-trip cleanly from f32 through a short f64 repr render
/// short (`0.1` not `0.10000000149011612`); otherwise we fall back to the
/// full f64 form so precision is preserved.
fn format_float(v: f64) -> String {
    let f = v as f32;
    // Try 1-9 decimal places and pick the shortest that round-trips back
    // to the same f32 bits. Rustc treats `0.5_f32` and `0.5f32` identically,
    // so plain decimal is fine.
    for digits in 1..=9 {
        let candidate = format!("{f:.digits$}", digits = digits);
        if let Ok(parsed) = candidate.parse::<f32>() {
            if parsed.to_bits() == f.to_bits() {
                // Trim redundant trailing zeros past one fractional digit
                // (so `0.500` becomes `0.5`, but `1.0` stays `1.0`).
                return trim_trailing_zeros(&candidate);
            }
        }
    }
    // Fall back to plain `{v}` for non-finite or truly awkward values.
    let s = format!("{v}");
    if s.contains('.')
        || s.contains('e')
        || s.contains('E')
        || s == "inf"
        || s == "-inf"
        || s == "NaN"
    {
        s
    } else {
        format!("{s}.0")
    }
}

fn trim_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let trimmed = s.trim_end_matches('0');
    if trimmed.ends_with('.') {
        format!("{trimmed}0")
    } else {
        trimmed.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::ir::{
        IrActionHead, IrActionHeadShape, IrExpr, IrExprNode, LocalRef, ScoringEntryIR, ScoringIR,
    };

    fn span() -> Span {
        Span::dummy()
    }

    fn lit_float_node(v: f64) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::LitFloat(v),
            span: span(),
        }
    }

    fn self_field(name: &str) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::Field {
                base: Box::new(IrExprNode {
                    kind: IrExpr::Local(LocalRef(0), "self".into()),
                    span: span(),
                }),
                field_name: name.into(),
                field: None,
            },
            span: span(),
        }
    }

    /// Field access against a named head-local (e.g. `t.hp_pct` on
    /// `Attack(t)`). LocalRef id is irrelevant to the emitter's shape
    /// check — only the binding name matters.
    fn local_field(local_name: &str, field_name: &str) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::Field {
                base: Box::new(IrExprNode {
                    kind: IrExpr::Local(LocalRef(1), local_name.into()),
                    span: span(),
                }),
                field_name: field_name.into(),
                field: None,
            },
            span: span(),
        }
    }

    fn binop(op: BinOp, lhs: IrExprNode, rhs: IrExprNode) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::Binary(op, Box::new(lhs), Box::new(rhs)),
            span: span(),
        }
    }

    fn if_expr(cond: IrExprNode, then_v: f64, else_v: f64) -> IrExprNode {
        IrExprNode {
            kind: IrExpr::If {
                cond: Box::new(cond),
                then_expr: Box::new(lit_float_node(then_v)),
                else_expr: Some(Box::new(lit_float_node(else_v))),
            },
            span: span(),
        }
    }

    fn action_head(name: &str) -> IrActionHead {
        IrActionHead {
            name: name.into(),
            shape: IrActionHeadShape::None,
            span: span(),
        }
    }

    fn attack_head() -> IrActionHead {
        IrActionHead {
            name: "Attack".into(),
            shape: IrActionHeadShape::Positional(vec![(
                "t".into(),
                LocalRef(0),
                crate::ir::IrType::AgentId,
            )]),
            span: span(),
        }
    }

    #[test]
    fn hold_bare_literal_lowers_to_zero_modifier_row() {
        let entry = ScoringEntryIR {
            head: action_head("Hold"),
            expr: lit_float_node(0.1),
            span: span(),
        };
        let row = lower_entry(&entry, &[]).unwrap();
        assert_eq!(row.action_head, 0);
        assert!((row.base - 0.1).abs() < 1e-6);
        assert_eq!(row.modifier_count, 0);
        assert_eq!(row.modifiers.len(), 0);
    }

    #[test]
    fn attack_with_hp_pct_guard_lowers_to_one_modifier() {
        // Attack(t) = 0.0 + (if self.hp_pct >= 0.5 { 0.5 } else { 0.0 })
        let cond = binop(BinOp::GtEq, self_field("hp_pct"), lit_float_node(0.5));
        let expr = binop(BinOp::Add, lit_float_node(0.0), if_expr(cond, 0.5, 0.0));
        let entry = ScoringEntryIR {
            head: attack_head(),
            expr,
            span: span(),
        };
        let row = lower_entry(&entry, &[]).unwrap();
        assert_eq!(row.action_head, 3, "Attack → MicroKind::Attack = 3");
        assert!((row.base - 0.0).abs() < 1e-6);
        assert_eq!(row.modifier_count, 1);
        assert_eq!(row.modifiers.len(), 1);
        let m = &row.modifiers[0];
        assert_eq!(m.kind, KIND_SCALAR_COMPARE);
        assert_eq!(m.op, OP_GE);
        assert_eq!(m.field_id, 2, "hp_pct → field_id 2");
        assert!((m.delta - 0.5).abs() < 1e-6);
        // Payload decodes to 0.5.
        let mut tb = [0u8; 4];
        tb.copy_from_slice(&m.payload[0..4]);
        assert!((f32::from_le_bytes(tb) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn flipped_operator_when_field_is_on_rhs() {
        // 0.5 <= self.hp_pct  → canonicalises to hp_pct >= 0.5
        let cond = binop(BinOp::LtEq, lit_float_node(0.5), self_field("hp_pct"));
        let expr = binop(BinOp::Add, lit_float_node(0.0), if_expr(cond, 0.5, 0.0));
        let entry = ScoringEntryIR {
            head: attack_head(),
            expr,
            span: span(),
        };
        let row = lower_entry(&entry, &[]).unwrap();
        assert_eq!(row.modifiers[0].op, OP_GE);
    }

    #[test]
    fn target_side_field_ref_uses_reserved_range() {
        // Attack(t) = 0.0 + (if t.hp_pct < 0.3 { 0.4 } else { 0.0 })
        // — `t` is the action head's target binding, so `t.hp_pct` emits
        // `0x4000 | 2 = 0x4002` (the engine's target-side hp_pct id).
        let cond = binop(BinOp::Lt, local_field("t", "hp_pct"), lit_float_node(0.3));
        let expr = binop(BinOp::Add, lit_float_node(0.0), if_expr(cond, 0.4, 0.0));
        let entry = ScoringEntryIR {
            head: attack_head(),
            expr,
            span: span(),
        };
        let row = lower_entry(&entry, &[]).unwrap();
        assert_eq!(row.modifier_count, 1);
        let m = &row.modifiers[0];
        assert_eq!(m.kind, KIND_SCALAR_COMPARE);
        assert_eq!(m.op, OP_LT);
        assert_eq!(
            m.field_id, 0x4002,
            "t.hp_pct on Attack(t) → 0x4002 (target-side hp_pct)"
        );
        assert!((m.delta - 0.4).abs() < 1e-6);
    }

    #[test]
    fn target_binding_rename_still_resolves() {
        // Rename the head's positional param from `t` to `target`. The
        // emitter recognises any first positional name as the
        // target-binding prefix — Attack(target) with `target.hp_pct`
        // lowers the same as Attack(t) with `t.hp_pct`.
        let head = IrActionHead {
            name: "Attack".into(),
            shape: IrActionHeadShape::Positional(vec![(
                "target".into(),
                LocalRef(1),
                crate::ir::IrType::AgentId,
            )]),
            span: span(),
        };
        let cond = binop(
            BinOp::Lt,
            local_field("target", "hp_pct"),
            lit_float_node(0.5),
        );
        let expr = binop(BinOp::Add, lit_float_node(0.0), if_expr(cond, 0.2, 0.0));
        let entry = ScoringEntryIR { head, expr, span: span() };
        let row = lower_entry(&entry, &[]).unwrap();
        assert_eq!(row.modifiers[0].field_id, 0x4002);
    }

    #[test]
    fn target_ref_on_self_only_head_is_an_error() {
        // Hold has no positional binding — referencing `t.hp_pct`
        // (or any non-self local) must raise UnsupportedPredicate rather
        // than silently miscompile.
        let cond = binop(BinOp::Lt, local_field("t", "hp_pct"), lit_float_node(0.3));
        let expr = binop(BinOp::Add, lit_float_node(0.0), if_expr(cond, 0.4, 0.0));
        let entry = ScoringEntryIR {
            head: action_head("Hold"),
            expr,
            span: span(),
        };
        let err = lower_entry(&entry, &[]).unwrap_err();
        assert!(matches!(err, EmitError::UnsupportedPredicate(_)));
    }

    #[test]
    fn unknown_action_head_is_an_error() {
        let entry = ScoringEntryIR {
            head: action_head("Teleport"),
            expr: lit_float_node(0.1),
            span: span(),
        };
        let err = lower_entry(&entry, &[]).unwrap_err();
        assert!(matches!(err, EmitError::UnknownActionHead(ref n) if n == "Teleport"));
    }

    #[test]
    fn non_zero_else_branch_is_an_error() {
        // if self.hp_pct >= 0.5 { 0.5 } else { 0.2 }  — disallowed: non-zero
        // else belongs in the sum's base, not the modifier.
        let cond = binop(BinOp::GtEq, self_field("hp_pct"), lit_float_node(0.5));
        let expr = if_expr(cond, 0.5, 0.2);
        let entry = ScoringEntryIR {
            head: attack_head(),
            expr,
            span: span(),
        };
        let err = lower_entry(&entry, &[]).unwrap_err();
        assert!(matches!(err, EmitError::UnsupportedModifierBody(_)));
    }

    #[test]
    fn emitted_table_header_and_rows_are_rust_shaped() {
        let cond = binop(BinOp::GtEq, self_field("hp_pct"), lit_float_node(0.5));
        let expr = binop(BinOp::Add, lit_float_node(0.0), if_expr(cond, 0.5, 0.0));
        let block = ScoringIR {
            entries: vec![
                ScoringEntryIR {
                    head: action_head("Hold"),
                    expr: lit_float_node(0.1),
                    span: span(),
                },
                ScoringEntryIR {
                    head: attack_head(),
                    expr,
                    span: span(),
                },
            ],
            annotations: vec![],
            span: span(),
        };
        let out = emit_scoring_mod(&[block], &[]);
        assert!(out.contains("pub const SCORING_TABLE: &[ScoringEntry] = &["));
        assert!(out.contains("action_head: 0,"));
        assert!(out.contains("action_head: 3,"));
        // 0.1 round-trips through f32 → 0.1 (rustc still formats with short
        // notation); guard on the prefix so an f64→f32 rounding wobble
        // doesn't flip the test.
        assert!(out.contains("base: 0.1"), "emitted output: {out}");
        assert!(
            out.contains("PredicateDescriptor::scalar_compare(2, PredicateDescriptor::OP_GE, 0.5)")
        );
        // Padding rows present (MAX_MODIFIERS total).
        let padding_count = out.matches("ModifierRow::EMPTY,").count();
        // Hold: 8 empties; Attack: 7 empties (one modifier fills slot 0).
        assert_eq!(padding_count, MAX_MODIFIERS + (MAX_MODIFIERS - 1));
    }

    #[test]
    fn empty_blocks_emit_empty_table() {
        let out = emit_scoring_mod(&[], &[]);
        assert!(out.contains("pub const SCORING_TABLE: &[ScoringEntry] = &[];"));
    }

    #[test]
    fn float_formatter_preserves_fractional_digit() {
        assert_eq!(format_float(0.0), "0.0");
        assert_eq!(format_float(1.0), "1.0");
        assert_eq!(format_float(0.5), "0.5");
        assert_eq!(format_float(-0.3), "-0.3");
    }
}
