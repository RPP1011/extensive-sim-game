//! Inner-expression and inner-statement WGSL emission.
//!
//! Walks a [`CgExpr`] / [`CgStmt`] tree and produces a WGSL source
//! fragment — never a complete kernel, never a binding declaration.
//! Composing fragments into kernel bodies is Task 4.2's job; assembling
//! the kernel module is Task 4.3.
//!
//! # Limitations
//!
//! - **Naming strategy.** Today only [`HandleNamingStrategy::Structural`]
//!   is implemented. Each [`DataHandle`] prints as a deterministic
//!   identifier-shaped name (`agent_self_hp`, `view_3_primary`,
//!   `mask_2_bitmap`, …) — useful for snapshot tests and as a
//!   placeholder until BGL slot assignment lands. Task 4.2 will plug in
//!   a slot-aware strategy that emits the actual buffer access form
//!   (e.g. `agents.hp[gid.x]` or `view_3_primary[a]`).
//! - **`AgentRef::Target(expr_id)`.** A target reference is a per-thread
//!   runtime value — no structural name can name it. The Structural
//!   strategy emits the placeholder `agent_target_expr_<N>_<field>`
//!   (where `<N>` is the [`CgExprId`]'s numeric value); Task 4.2
//!   replaces this with a runtime-resolved buffer access using the
//!   target expression's lowered value.
//! - **Custom builtins.** [`BuiltinId::PlanarDistance`],
//!   [`BuiltinId::ZSeparation`], [`BuiltinId::SaturatingAdd`],
//!   `is_hostile`, `kin_count_within`, etc. are emitted as direct
//!   function calls (`planar_distance(a, b)`, `saturating_add(x, y)`).
//!   Task 4.3 wires the WGSL prelude that provides these helpers.
//! - **`Match` lowering.** Lowered as an `if`-chain over each arm's
//!   variant tag (`if (scrutinee_tag == VARIANT_<N>) { ... }`). WGSL
//!   does support `switch`, but the IR's variant ids are not yet
//!   resolved to compact case constants — `if`-chain is the honest
//!   placeholder until the prelude lands. Arm-binding locals
//!   (`MatchArmBinding::local`) are not yet referenced from arm bodies
//!   (the IR errors on local reads in expression lowering today).
//! - **Event emit shape.** The emit form here is a placeholder
//!   `emit_event_<N>(field0: ..., field1: ...);` — Task 4.2 wires the
//!   actual ring-append form once event-ring slot assignment is known.
//! - **Vec3 swizzles.** Writes to a `Vec3` field as a whole are
//!   supported; per-component writes are an emit-time concern not yet
//!   surfaced in the IR.
//!
//! # Reuse from prior layers
//!
//! [`crate::cg::CgExpr`], [`crate::cg::CgStmt`], [`DataHandle`],
//! [`crate::cg::BinaryOp`], [`crate::cg::UnaryOp`], [`BuiltinId`] are
//! consumed read-only — no IR shapes are added by Task 4.1. New
//! lowerings of those types extend the match arms here exhaustively
//! (no `_ =>` fallthroughs in production code).

use std::fmt;

use crate::cg::data_handle::{
    AgentRef, AgentScratchKind, CgExprId, DataHandle, EventRingAccess, RngPurpose,
    SpatialStorageKind, ViewStorageSlot,
};
use crate::cg::expr::{BinaryOp, BuiltinId, CgExpr, CgTy, ExprArena, LitValue, NumericTy, UnaryOp};
use crate::cg::program::CgProgram;
use crate::cg::stmt::{
    CgMatchArm, CgStmt, CgStmtId, CgStmtListId, EventField, MatchArmBinding, StmtArena,
    StmtListArena,
};

// ---------------------------------------------------------------------------
// EmitCtx
// ---------------------------------------------------------------------------

/// Strategy for naming a [`DataHandle`] when it appears as the bare
/// operand of a `Read` / `Assign`. Task 4.1 ships only the
/// [`Structural`] strategy; future tasks add a slot-aware variant.
///
/// [`Structural`]: HandleNamingStrategy::Structural
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum HandleNamingStrategy {
    /// Each handle prints as a deterministic identifier-shaped name
    /// (`agent_self_hp`, `view_3_primary`, `mask_2_bitmap`,
    /// `event_ring_5_read`, `rng_action`, …). The shape mirrors
    /// [`DataHandle::Display`]'s output but stripped down to
    /// WGSL-valid identifier characters (`[A-Za-z0-9_]` only). Used by
    /// snapshot tests and as the Task-4.1 placeholder before BGL slot
    /// assignment lands.
    Structural,
}

/// Context carried through the inner WGSL walks. Holds just the
/// program (for arena lookups) and the active handle naming strategy.
///
/// Constructed by Task 4.2's kernel-body composer; Task 4.1's tests
/// build it directly.
pub struct EmitCtx<'a> {
    /// The program — every [`CgExprId`] / [`CgStmtId`] / [`CgStmtListId`]
    /// is resolved against this program's arenas via the
    /// [`ExprArena`] / [`StmtArena`] / [`StmtListArena`] trait impls.
    pub prog: &'a CgProgram,
    /// Strategy for printing a [`DataHandle`] as a WGSL identifier.
    pub naming: HandleNamingStrategy,
}

impl<'a> EmitCtx<'a> {
    /// Construct an emit context with the [`HandleNamingStrategy::Structural`]
    /// strategy — the only one Task 4.1 ships.
    pub fn structural(prog: &'a CgProgram) -> Self {
        Self {
            prog,
            naming: HandleNamingStrategy::Structural,
        }
    }

    /// Render `handle` as a WGSL identifier per the active naming
    /// strategy.
    ///
    /// # Limitations
    ///
    /// - With [`HandleNamingStrategy::Structural`], every variant
    ///   produces a deterministic identifier; [`AgentRef::Target`] is
    ///   rendered as a placeholder (`agent_target_expr_<N>_<field>`)
    ///   that Task 4.2 will replace with a runtime-resolved access.
    /// - Plumbing-only handles ([`DataHandle::AliveBitmap`],
    ///   [`DataHandle::IndirectArgs`], [`DataHandle::AgentScratch`],
    ///   [`DataHandle::SimCfgBuffer`], [`DataHandle::SnapshotKick`])
    ///   never appear inside an expression body in a well-formed
    ///   program (they live on `PlumbingKind` ops). The Structural
    ///   strategy still gives them a deterministic name so error
    ///   diagnostics on a malformed IR remain readable.
    pub fn handle_name(&self, h: &DataHandle) -> String {
        match self.naming {
            HandleNamingStrategy::Structural => structural_handle_name(h),
        }
    }
}

// ---------------------------------------------------------------------------
// Structural handle naming
// ---------------------------------------------------------------------------

/// Render `handle` as a deterministic WGSL identifier — the
/// [`HandleNamingStrategy::Structural`] form. Stable across runs.
fn structural_handle_name(h: &DataHandle) -> String {
    match h {
        DataHandle::AgentField { field, target } => {
            format!("agent_{}_{}", agent_ref_token(target), field.snake())
        }
        DataHandle::ViewStorage { view, slot } => {
            format!("view_{}_{}", view.0, view_slot_token(*slot))
        }
        DataHandle::EventRing { ring, kind } => {
            format!("event_ring_{}_{}", ring.0, event_ring_access_token(*kind))
        }
        DataHandle::ConfigConst { id } => format!("config_{}", id.0),
        DataHandle::MaskBitmap { mask } => format!("mask_{}_bitmap", mask.0),
        DataHandle::ScoringOutput => "scoring_output".to_string(),
        DataHandle::SpatialStorage { kind } => {
            format!("spatial_{}", spatial_storage_token(*kind))
        }
        DataHandle::Rng { purpose } => format!("rng_{}", rng_purpose_token(*purpose)),
        DataHandle::AliveBitmap => "alive_bitmap".to_string(),
        DataHandle::IndirectArgs { ring } => format!("indirect_args_{}", ring.0),
        DataHandle::AgentScratch { kind } => {
            format!("agent_scratch_{}", agent_scratch_token(*kind))
        }
        DataHandle::SimCfgBuffer => "sim_cfg_buffer".to_string(),
        DataHandle::SnapshotKick => "snapshot_kick".to_string(),
    }
}

/// Identifier token for an [`AgentRef`]. `Target(expr_id)` maps to the
/// placeholder `target_expr_<N>` per the module-level limitations note.
fn agent_ref_token(target: &AgentRef) -> String {
    match target {
        AgentRef::Self_ => "self".to_string(),
        AgentRef::Actor => "actor".to_string(),
        AgentRef::EventTarget => "event_target".to_string(),
        AgentRef::Target(id) => format!("target_expr_{}", id.0),
    }
}

fn view_slot_token(slot: ViewStorageSlot) -> &'static str {
    match slot {
        ViewStorageSlot::Primary => "primary",
        ViewStorageSlot::Anchor => "anchor",
        ViewStorageSlot::Ids => "ids",
        ViewStorageSlot::Counts => "counts",
        ViewStorageSlot::Cursors => "cursors",
    }
}

fn event_ring_access_token(kind: EventRingAccess) -> &'static str {
    match kind {
        EventRingAccess::Read => "read",
        EventRingAccess::Append => "append",
        EventRingAccess::Drain => "drain",
    }
}

fn spatial_storage_token(kind: SpatialStorageKind) -> &'static str {
    match kind {
        SpatialStorageKind::GridCells => "grid_cells",
        SpatialStorageKind::GridOffsets => "grid_offsets",
        SpatialStorageKind::QueryResults => "query_results",
    }
}

fn rng_purpose_token(purpose: RngPurpose) -> &'static str {
    // Routes through the canonical snake-case label so adding a new
    // RngPurpose variant requires only one update site (the enum impl).
    purpose.snake()
}

fn agent_scratch_token(kind: AgentScratchKind) -> &'static str {
    match kind {
        AgentScratchKind::Packed => "packed",
    }
}

// ---------------------------------------------------------------------------
// EmitError
// ---------------------------------------------------------------------------

/// Errors a Task-4.1 lowering can raise. Every variant names a typed
/// id — no free-form `String` reasons — so callers can match on the
/// shape of the failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// A [`CgExprId`] reference was past the end of the program's
    /// expression arena.
    ExprIdOutOfRange { id: CgExprId, arena_len: u32 },
    /// A [`CgStmtId`] reference was past the end of the program's
    /// statement arena.
    StmtIdOutOfRange { id: CgStmtId, arena_len: u32 },
    /// A [`CgStmtListId`] reference was past the end of the program's
    /// statement-list arena.
    StmtListIdOutOfRange {
        id: CgStmtListId,
        arena_len: u32,
    },
    /// The active [`HandleNamingStrategy`] does not produce a WGSL name
    /// for `handle`. Today nothing raises this — Task 4.2's slot-aware
    /// strategy will use it for handles that have no slot assignment.
    UnsupportedHandle {
        handle: DataHandle,
        reason: &'static str,
    },
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmitError::ExprIdOutOfRange { id, arena_len } => write!(
                f,
                "CgExprId(#{}) out of range (expr arena holds {} entries)",
                id.0, arena_len
            ),
            EmitError::StmtIdOutOfRange { id, arena_len } => write!(
                f,
                "CgStmtId(#{}) out of range (stmt arena holds {} entries)",
                id.0, arena_len
            ),
            EmitError::StmtListIdOutOfRange { id, arena_len } => write!(
                f,
                "CgStmtListId(#{}) out of range (stmt-list arena holds {} entries)",
                id.0, arena_len
            ),
            EmitError::UnsupportedHandle { handle, reason } => {
                write!(f, "unsupported handle {handle}: {reason}")
            }
        }
    }
}

impl std::error::Error for EmitError {}

// ---------------------------------------------------------------------------
// Op-symbol mappings
// ---------------------------------------------------------------------------

/// WGSL infix symbol for a [`BinaryOp`]. Per-variant exhaustive — no
/// fallthrough — so adding a new `BinaryOp` variant forces a decision
/// here.
fn binary_op_to_wgsl(op: BinaryOp) -> &'static str {
    use BinaryOp::*;
    match op {
        AddF32 | AddU32 | AddI32 => "+",
        SubF32 | SubU32 | SubI32 => "-",
        MulF32 | MulU32 | MulI32 => "*",
        DivF32 | DivU32 | DivI32 => "/",
        LtF32 | LtU32 | LtI32 => "<",
        LeF32 | LeU32 | LeI32 => "<=",
        GtF32 | GtU32 | GtI32 => ">",
        GeF32 | GeU32 | GeI32 => ">=",
        EqBool | EqU32 | EqI32 | EqF32 | EqAgentId => "==",
        NeBool | NeU32 | NeI32 | NeF32 | NeAgentId => "!=",
        And => "&&",
        Or => "||",
    }
}

/// Render `op(arg)` for unary ops. Some unaries are prefix operators
/// (`-x`, `!x`); others are call-form (`abs(x)`, `sqrt(x)`,
/// `normalize(x)`). Returned tag selects the shape so the caller can
/// build the right string.
enum UnaryShape {
    /// `<symbol><arg>` — prefix operator.
    Prefix(&'static str),
    /// `<name>(<arg>)` — function call.
    Call(&'static str),
}

fn unary_op_shape(op: UnaryOp) -> UnaryShape {
    use UnaryOp::*;
    match op {
        NotBool => UnaryShape::Prefix("!"),
        NegF32 | NegI32 => UnaryShape::Prefix("-"),
        AbsF32 | AbsI32 => UnaryShape::Call("abs"),
        SqrtF32 => UnaryShape::Call("sqrt"),
        NormalizeVec3F32 => UnaryShape::Call("normalize"),
    }
}

/// WGSL function name for a [`BuiltinId`]. View calls embed the view
/// id structurally so each view's getter has a stable, distinct name.
fn builtin_name(id: BuiltinId) -> String {
    use BuiltinId::*;
    match id {
        Distance => "distance".to_string(),
        PlanarDistance => "planar_distance".to_string(),
        ZSeparation => "z_separation".to_string(),
        Min(t) => format!("min_{}", numeric_ty_token(t)),
        Max(t) => format!("max_{}", numeric_ty_token(t)),
        Clamp(t) => format!("clamp_{}", numeric_ty_token(t)),
        SaturatingAdd(t) => format!("saturating_add_{}", numeric_ty_token(t)),
        Floor => "floor".to_string(),
        Ceil => "ceil".to_string(),
        Round => "round".to_string(),
        Ln => "log".to_string(),
        Log2 => "log2".to_string(),
        Log10 => "log10".to_string(),
        Entity => "entity".to_string(),
        ViewCall { view } => format!("view_{}_get", view.0),
    }
}

fn numeric_ty_token(t: NumericTy) -> &'static str {
    match t {
        NumericTy::F32 => "f32",
        NumericTy::U32 => "u32",
        NumericTy::I32 => "i32",
    }
}

// ---------------------------------------------------------------------------
// Literal emission
// ---------------------------------------------------------------------------

/// Render an `f32` as a WGSL float literal, matching the legacy
/// `emit_view::format_f32_lit` convention so Phase-5 byte-for-byte
/// parity with the legacy emit path holds.
///
/// Convention (ported locally — does **not** depend on `emit_view.rs`,
/// which is slated for retirement in Task 5.2):
/// 1. Format via `Display` (`{v}`) — gives `"1"` for `1.0`, `"1.5"` for
///    `1.5`, `"0.00001"` for `1e-5`, `"1000000000000000000000000000000"`
///    for `1e30`, and the fully-expanded decimal for sub-normals.
/// 2. If the result already contains `.`, `e`, or `E`, return as-is.
/// 3. Otherwise append `".0"` so WGSL parses the literal as `f32`,
///    not an abstract integer.
///
/// # WGSL syntax notes
///
/// - Integer-valued: `1.0` → `"1.0"`. Round-trip safe.
/// - Sub-unit: `0.5` → `"0.5"`, `-0.5` → `"-0.5"`. Both retain the dot.
/// - Very large: `1e30` → `"1000…0.0"` — a 31-digit literal. Legal WGSL,
///   but ugly; well-formed sim programs do not use literals this large.
/// - Very small: `1e-30` → `"0.000…01"` — a 32-digit literal. Same caveat.
/// - `f32::MIN_POSITIVE` (`~1.175e-38`) — the fully-expanded decimal is
///   45+ characters; well-formed sim programs do not embed it as a literal.
fn format_f32_lit(v: f32) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Render a [`LitValue`] as a WGSL constant fragment. `f32` and the
/// three components of `Vec3F32` route through [`format_f32_lit`] so
/// output is byte-identical to the legacy emit path.
fn lower_literal(lit: &LitValue) -> String {
    match lit {
        LitValue::Bool(true) => "true".to_string(),
        LitValue::Bool(false) => "false".to_string(),
        LitValue::U32(v) => format!("{}u", v),
        LitValue::I32(v) => format!("{}i", v),
        LitValue::F32(v) => format_f32_lit(*v),
        // Tick is u32 at the WGSL level — see `CgTy::Tick` doc.
        LitValue::Tick(v) => format!("{}u", v),
        // AgentId is a u32 slot index at the WGSL level.
        LitValue::AgentId(v) => format!("{}u", v),
        LitValue::Vec3F32 { x, y, z } => {
            format!(
                "vec3<f32>({}, {}, {})",
                format_f32_lit(*x),
                format_f32_lit(*y),
                format_f32_lit(*z)
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Expression emission
// ---------------------------------------------------------------------------

/// Lower a single [`CgExpr`] (resolved by id from `ctx.prog`) into a
/// WGSL source fragment.
///
/// # Limitations
///
/// - Walks are pure: no decisions, no kernel boilerplate, no new
///   bindings. Each variant maps to a fixed WGSL form.
/// - `Read` produces the bare handle name (Task 4.2 wraps with the
///   actual buffer indexing form).
/// - `Rng` produces a structural call `per_agent_u32(seed, agent_id, tick, "<purpose>")`;
///   the actual seed/agent/tick arguments are wired by Task 4.2.
/// - `Builtin` emits the WGSL function name from [`builtin_name`];
///   custom helpers (`planar_distance`, `saturating_add_<ty>`,
///   `view_<id>_get`) are assumed to live in the prelude (Task 4.3).
/// - `Select` emits WGSL's `select(false_val, true_val, cond)` shape —
///   note the false-value-first ordering.
///
/// # Errors
///
/// Returns [`EmitError::ExprIdOutOfRange`] if any descendant id is past
/// the end of `ctx.prog.exprs`.
pub fn lower_cg_expr_to_wgsl(expr_id: CgExprId, ctx: &EmitCtx) -> Result<String, EmitError> {
    let arena_len = ctx.prog.exprs.len() as u32;
    let node = <CgProgram as ExprArena>::get(ctx.prog, expr_id).ok_or(
        EmitError::ExprIdOutOfRange {
            id: expr_id,
            arena_len,
        },
    )?;
    match node {
        CgExpr::Read(handle) => Ok(ctx.handle_name(handle)),
        CgExpr::Lit(v) => Ok(lower_literal(v)),
        CgExpr::Binary { op, lhs, rhs, ty: _ } => {
            let l = lower_cg_expr_to_wgsl(*lhs, ctx)?;
            let r = lower_cg_expr_to_wgsl(*rhs, ctx)?;
            Ok(format!("({} {} {})", l, binary_op_to_wgsl(*op), r))
        }
        CgExpr::Unary { op, arg, ty: _ } => {
            let a = lower_cg_expr_to_wgsl(*arg, ctx)?;
            match unary_op_shape(*op) {
                UnaryShape::Prefix(sym) => Ok(format!("({}{})", sym, a)),
                UnaryShape::Call(name) => Ok(format!("{}({})", name, a)),
            }
        }
        CgExpr::Builtin { fn_id, args, ty: _ } => {
            let mut parts = Vec::with_capacity(args.len());
            for a in args {
                parts.push(lower_cg_expr_to_wgsl(*a, ctx)?);
            }
            Ok(format!("{}({})", builtin_name(*fn_id), parts.join(", ")))
        }
        CgExpr::Rng { purpose, ty: _ } => {
            // `per_agent_u32(seed, agent_id, tick, "<purpose>")` —
            // matches the engine RNG primitive named in
            // `engine::rng::per_agent_u32`. The seed/agent/tick names
            // are placeholders for Task 4.2, which knows the kernel's
            // local variable bindings.
            Ok(format!(
                "per_agent_u32(seed, agent_id, tick, \"{}\")",
                rng_purpose_token(*purpose)
            ))
        }
        CgExpr::Select {
            cond,
            then,
            else_,
            ty: _,
        } => {
            let c = lower_cg_expr_to_wgsl(*cond, ctx)?;
            let t = lower_cg_expr_to_wgsl(*then, ctx)?;
            let e = lower_cg_expr_to_wgsl(*else_, ctx)?;
            // WGSL's `select(false_val, true_val, cond)` — note the
            // false-value-first order.
            Ok(format!("select({}, {}, {})", e, t, c))
        }
    }
}

// ---------------------------------------------------------------------------
// Statement emission
// ---------------------------------------------------------------------------

/// Indent every line of `s` by `indent` four-space levels — matches
/// the convention used throughout the legacy emit path
/// (`emit_view_wgsl.rs`, etc.) so Phase-5 parity holds without
/// whitespace drift.
fn indent_block(s: &str, indent: usize) -> String {
    let prefix: String = "    ".repeat(indent);
    s.lines()
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{}{}", prefix, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Lower a single [`CgStmt`] into a WGSL source fragment. The output
/// contains no leading indentation — the caller composes it with its
/// surrounding context.
///
/// # Limitations
///
/// - `Assign` produces `<target> = <value>;` using the active naming
///   strategy for the target.
/// - `Emit` produces a placeholder call form
///   `emit_event_<N>(field_<I>: <expr>, ...);`. Task 4.2 wires the
///   actual ring-append shape.
/// - `If` emits `if (...) { ... }` (or `if (...) { ... } else { ... }`)
///   using brace-and-newline structure.
/// - `Match` emits an `if`-chain over each arm's variant tag — see
///   the module-level limitations note.
///
/// # Errors
///
/// Returns one of [`EmitError::ExprIdOutOfRange`],
/// [`EmitError::StmtIdOutOfRange`], or
/// [`EmitError::StmtListIdOutOfRange`] for any dangling id.
pub fn lower_cg_stmt_to_wgsl(stmt_id: CgStmtId, ctx: &EmitCtx) -> Result<String, EmitError> {
    let arena_len = ctx.prog.stmts.len() as u32;
    let node = <CgProgram as StmtArena>::get(ctx.prog, stmt_id).ok_or(
        EmitError::StmtIdOutOfRange {
            id: stmt_id,
            arena_len,
        },
    )?;
    match node {
        CgStmt::Assign { target, value } => {
            let lhs = ctx.handle_name(target);
            let rhs = lower_cg_expr_to_wgsl(*value, ctx)?;
            Ok(format!("{} = {};", lhs, rhs))
        }
        CgStmt::Emit { event, fields } => lower_emit_to_wgsl(event.0, fields, ctx),
        CgStmt::If { cond, then, else_ } => {
            let c = lower_cg_expr_to_wgsl(*cond, ctx)?;
            let then_body = lower_cg_stmt_list_to_wgsl(*then, ctx)?;
            match else_ {
                Some(else_id) => {
                    let else_body = lower_cg_stmt_list_to_wgsl(*else_id, ctx)?;
                    Ok(format!(
                        "if ({}) {{\n{}\n}} else {{\n{}\n}}",
                        c,
                        indent_block(&then_body, 1),
                        indent_block(&else_body, 1)
                    ))
                }
                None => Ok(format!(
                    "if ({}) {{\n{}\n}}",
                    c,
                    indent_block(&then_body, 1)
                )),
            }
        }
        CgStmt::Match { scrutinee, arms } => lower_match_to_wgsl(*scrutinee, arms, ctx),
    }
}

/// Lower a [`CgStmt::Emit`] body. Placeholder shape per the module
/// limitations note — Task 4.2 wires the real ring-append form.
fn lower_emit_to_wgsl(
    event_id: u32,
    fields: &[(EventField, CgExprId)],
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    let mut parts = Vec::with_capacity(fields.len());
    for (field, expr_id) in fields {
        let v = lower_cg_expr_to_wgsl(*expr_id, ctx)?;
        parts.push(format!("field_{}: {}", field.index, v));
    }
    Ok(format!("emit_event_{}({});", event_id, parts.join(", ")))
}

/// Lower a [`CgStmt::Match`] as a scrutinee-bound `if`-chain. WGSL's
/// `switch` would be a future-tense option; today the chain is the
/// honest placeholder.
///
/// The scrutinee is bound to a local variable `_scrut_<N>` *before* the
/// chain so non-identifier scrutinees (e.g. a `Binary { ... }` node
/// lowered to `(x + 1)`) produce valid WGSL — `((x + 1)_tag)` is
/// nonsense, `_scrut_<N>.tag` is fine. `<N>` is the scrutinee's
/// [`CgExprId`] (the only id this function has access to — `CgStmtId` /
/// `CgStmtListId` are not threaded through). Since each `Match`
/// statement has a distinct scrutinee expression node in the arena, the
/// id is unique-per-match-site within a program.
///
/// Arm-binding locals are still emitted as a comment for now, but the
/// comment references `_scrut_<N>.<field>` so a future Task 4.x can
/// flip the comment into a real `let local_<N>: <ty> = _scrut_<N>.<field>;`
/// without changing the surrounding shape.
fn lower_match_to_wgsl(
    scrutinee: CgExprId,
    arms: &[CgMatchArm],
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    let s = lower_cg_expr_to_wgsl(scrutinee, ctx)?;
    if arms.is_empty() {
        // Empty match body — emit a comment so the generated WGSL is
        // still syntactically inert. (Should not occur in well-formed
        // programs.)
        return Ok(format!("// match {} {{ /* no arms */ }}", s));
    }
    let scrut_name = format!("_scrut_{}", scrutinee.0);
    let mut out = format!("let {} = {};\n", scrut_name, s);
    for (i, arm) in arms.iter().enumerate() {
        let body = lower_cg_stmt_list_to_wgsl(arm.body, ctx)?;
        let bindings_comment = if arm.bindings.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = arm
                .bindings
                .iter()
                .map(|b: &MatchArmBinding| {
                    format!(
                        "{name}=local_{lid} from {scrut}.{name}",
                        name = b.field_name,
                        lid = b.local.0,
                        scrut = scrut_name,
                    )
                })
                .collect();
            format!(" /* bindings: {} */", pairs.join(", "))
        };
        if i == 0 {
            out.push_str(&format!(
                "if ({}.tag == VARIANT_{}u) {{{}\n{}\n}}",
                scrut_name,
                arm.variant.0,
                bindings_comment,
                indent_block(&body, 1)
            ));
        } else {
            out.push_str(&format!(
                " else if ({}.tag == VARIANT_{}u) {{{}\n{}\n}}",
                scrut_name,
                arm.variant.0,
                bindings_comment,
                indent_block(&body, 1)
            ));
        }
    }
    Ok(out)
}

/// Lower a [`crate::cg::CgStmtList`] as a sequence of statements,
/// joined with `\n`. Empty lists produce the empty string.
///
/// # Limitations
///
/// Same as [`lower_cg_stmt_to_wgsl`].
pub fn lower_cg_stmt_list_to_wgsl(
    list_id: CgStmtListId,
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    let arena_len = ctx.prog.stmt_lists.len() as u32;
    let list = <CgProgram as StmtListArena>::get(ctx.prog, list_id).ok_or(
        EmitError::StmtListIdOutOfRange {
            id: list_id,
            arena_len,
        },
    )?;
    let mut parts = Vec::with_capacity(list.stmts.len());
    for stmt_id in &list.stmts {
        parts.push(lower_cg_stmt_to_wgsl(*stmt_id, ctx)?);
    }
    Ok(parts.join("\n"))
}

// ---------------------------------------------------------------------------
// CgTy → WGSL type name (used by snapshot-style harnesses; not the
// public surface but kept here so the mapping has one home).
// ---------------------------------------------------------------------------

/// WGSL type name for a [`CgTy`]. Useful in tests + future kernel
/// emission. Exhaustive — adding a CgTy variant forces a decision.
pub fn cg_ty_to_wgsl(ty: CgTy) -> String {
    match ty {
        CgTy::Bool => "bool".to_string(),
        CgTy::U32 => "u32".to_string(),
        CgTy::I32 => "i32".to_string(),
        CgTy::F32 => "f32".to_string(),
        CgTy::Vec3F32 => "vec3<f32>".to_string(),
        // AgentId, Tick both lower to u32 at the WGSL boundary — the
        // engine narrows ticks (u64 → u32) and represents agent slot
        // ids as u32 indices.
        CgTy::AgentId | CgTy::Tick => "u32".to_string(),
        // ViewKey is a phantom u32 at the WGSL level — its semantic
        // payload is whatever the view's primary storage carries.
        CgTy::ViewKey { .. } => "u32".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{
        AgentFieldId, ConfigConstId, EventRingId, MaskId, ViewId,
    };
    use crate::cg::op::EventKindId;
    use crate::cg::stmt::{
        CgMatchArm, CgStmt, CgStmtId, CgStmtList, CgStmtListId, EventField, LocalId,
        MatchArmBinding, VariantId,
    };

    /// Build a fresh `CgProgram` and populate it directly via the
    /// `pub` arena fields. Task 4.1 tests don't need a full builder
    /// pass — they only need to wire ids that resolve.
    fn empty_prog() -> CgProgram {
        CgProgram::default()
    }

    fn push_expr(prog: &mut CgProgram, e: CgExpr) -> CgExprId {
        let id = CgExprId(prog.exprs.len() as u32);
        prog.exprs.push(e);
        id
    }

    fn push_stmt(prog: &mut CgProgram, s: CgStmt) -> CgStmtId {
        let id = CgStmtId(prog.stmts.len() as u32);
        prog.stmts.push(s);
        id
    }

    fn push_list(prog: &mut CgProgram, l: CgStmtList) -> CgStmtListId {
        let id = CgStmtListId(prog.stmt_lists.len() as u32);
        prog.stmt_lists.push(l);
        id
    }

    // ---- 1. LitValue per-variant ----

    #[test]
    fn lower_lit_each_variant() {
        let mut prog = empty_prog();
        let cases: Vec<(LitValue, &'static str)> = vec![
            (LitValue::Bool(true), "true"),
            (LitValue::Bool(false), "false"),
            (LitValue::U32(7), "7u"),
            (LitValue::I32(-3), "-3i"),
            (LitValue::F32(1.5), "1.5"),
            (LitValue::Tick(42), "42u"),
            (LitValue::AgentId(11), "11u"),
        ];
        for (lit, expected) in cases {
            let id = push_expr(&mut prog, CgExpr::Lit(lit));
            let ctx = EmitCtx::structural(&prog);
            assert_eq!(lower_cg_expr_to_wgsl(id, &ctx).unwrap(), expected);
        }

        // Vec3F32 separately — `{:?}` on f32 → "1.0", "2.0", "3.0".
        let id = push_expr(
            &mut prog,
            CgExpr::Lit(LitValue::Vec3F32 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            }),
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(id, &ctx).unwrap(),
            "vec3<f32>(1.0, 2.0, 3.0)"
        );
    }

    // ---- 2. BinaryOp class coverage (arith, comparison, logical) ----

    #[test]
    fn lower_binary_arith_comparison_logical() {
        // (hp + 1.0)
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let add = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: hp,
                rhs: one,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(add, &ctx).unwrap(),
            "(agent_self_hp + 1.0)"
        );

        // (hp < 5.0)
        let five = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(5.0)));
        let lt = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: five,
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(lt, &ctx).unwrap(),
            "(agent_self_hp < 5.0)"
        );

        // (true && false)
        let t = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
        let f = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(false)));
        let and = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::And,
                lhs: t,
                rhs: f,
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(lower_cg_expr_to_wgsl(and, &ctx).unwrap(), "(true && false)");
    }

    /// Spot-check every `BinaryOp` symbol mapping (smoke test for the
    /// exhaustive match).
    #[test]
    fn binary_op_to_wgsl_covers_each_class() {
        // Arithmetic
        assert_eq!(binary_op_to_wgsl(BinaryOp::AddF32), "+");
        assert_eq!(binary_op_to_wgsl(BinaryOp::SubU32), "-");
        assert_eq!(binary_op_to_wgsl(BinaryOp::MulI32), "*");
        assert_eq!(binary_op_to_wgsl(BinaryOp::DivF32), "/");
        // Comparisons
        assert_eq!(binary_op_to_wgsl(BinaryOp::LtF32), "<");
        assert_eq!(binary_op_to_wgsl(BinaryOp::LeU32), "<=");
        assert_eq!(binary_op_to_wgsl(BinaryOp::GtI32), ">");
        assert_eq!(binary_op_to_wgsl(BinaryOp::GeF32), ">=");
        // Equality
        assert_eq!(binary_op_to_wgsl(BinaryOp::EqU32), "==");
        assert_eq!(binary_op_to_wgsl(BinaryOp::EqAgentId), "==");
        assert_eq!(binary_op_to_wgsl(BinaryOp::NeF32), "!=");
        // Logical
        assert_eq!(binary_op_to_wgsl(BinaryOp::And), "&&");
        assert_eq!(binary_op_to_wgsl(BinaryOp::Or), "||");
    }

    // ---- 3. UnaryOp class coverage ----

    #[test]
    fn lower_unary_neg_not_abs_sqrt_normalize() {
        let mut prog = empty_prog();
        // -hp
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let neg = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::NegF32,
                arg: hp,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(lower_cg_expr_to_wgsl(neg, &ctx).unwrap(), "(-agent_self_hp)");

        // !alive
        let alive = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::Self_,
            }),
        );
        let not_alive = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::NotBool,
                arg: alive,
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(not_alive, &ctx).unwrap(),
            "(!agent_self_alive)"
        );

        // abs(slow_factor_q8)
        let sf = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::SlowFactorQ8,
                target: AgentRef::Self_,
            }),
        );
        let abs = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::AbsI32,
                arg: sf,
                ty: CgTy::I32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(abs, &ctx).unwrap(),
            "abs(agent_self_slow_factor_q8)"
        );

        // sqrt(2.0)
        let two = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.0)));
        let sq = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::SqrtF32,
                arg: two,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(lower_cg_expr_to_wgsl(sq, &ctx).unwrap(), "sqrt(2.0)");

        // normalize(pos)
        let pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let norm = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::NormalizeVec3F32,
                arg: pos,
                ty: CgTy::Vec3F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(norm, &ctx).unwrap(),
            "normalize(agent_self_pos)"
        );
    }

    // ---- 4. Builtin coverage ----

    #[test]
    fn lower_builtin_distance_min_clamp_view_call() {
        let mut prog = empty_prog();
        let pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let actor_pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Actor,
            }),
        );
        // distance(self.pos, actor.pos)
        let dist = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Distance,
                args: vec![pos, actor_pos],
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(dist, &ctx).unwrap(),
            "distance(agent_self_pos, agent_actor_pos)"
        );

        // min_f32(1.0, 2.0)
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let two = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.0)));
        let min = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Min(NumericTy::F32),
                args: vec![one, two],
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(min, &ctx).unwrap(),
            "min_f32(1.0, 2.0)"
        );

        // clamp_u32(level, 1, 99)
        let level = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Level,
                target: AgentRef::Self_,
            }),
        );
        let lo = push_expr(&mut prog, CgExpr::Lit(LitValue::U32(1)));
        let hi = push_expr(&mut prog, CgExpr::Lit(LitValue::U32(99)));
        let cl = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Clamp(NumericTy::U32),
                args: vec![level, lo, hi],
                ty: CgTy::U32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(cl, &ctx).unwrap(),
            "clamp_u32(agent_self_level, 1u, 99u)"
        );

        // view_2_get(self_pos)
        let vc = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::ViewCall { view: ViewId(2) },
                args: vec![pos],
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(vc, &ctx).unwrap(),
            "view_2_get(agent_self_pos)"
        );

        // saturating_add_u32 spot-check
        assert_eq!(
            builtin_name(BuiltinId::SaturatingAdd(NumericTy::U32)),
            "saturating_add_u32"
        );
        // log/log2/log10/floor/ceil/round + planar_distance + z_separation + entity
        assert_eq!(builtin_name(BuiltinId::Floor), "floor");
        assert_eq!(builtin_name(BuiltinId::Ceil), "ceil");
        assert_eq!(builtin_name(BuiltinId::Round), "round");
        assert_eq!(builtin_name(BuiltinId::Ln), "log");
        assert_eq!(builtin_name(BuiltinId::Log2), "log2");
        assert_eq!(builtin_name(BuiltinId::Log10), "log10");
        assert_eq!(builtin_name(BuiltinId::PlanarDistance), "planar_distance");
        assert_eq!(builtin_name(BuiltinId::ZSeparation), "z_separation");
        assert_eq!(builtin_name(BuiltinId::Entity), "entity");
    }

    // ---- 5. DataHandle Read coverage (each variant) ----

    #[test]
    fn lower_read_each_data_handle_variant() {
        let mut prog = empty_prog();
        // AgentField — Self_ / Actor / EventTarget / Target(expr_id)
        let target_expr_id = push_expr(&mut prog, CgExpr::Lit(LitValue::AgentId(0)));
        let cases: Vec<(DataHandle, &str)> = vec![
            (
                DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                "agent_self_hp",
            ),
            (
                DataHandle::AgentField {
                    field: AgentFieldId::Pos,
                    target: AgentRef::Actor,
                },
                "agent_actor_pos",
            ),
            (
                DataHandle::AgentField {
                    field: AgentFieldId::Alive,
                    target: AgentRef::EventTarget,
                },
                "agent_event_target_alive",
            ),
            (
                DataHandle::AgentField {
                    field: AgentFieldId::Pos,
                    target: AgentRef::Target(target_expr_id),
                },
                "agent_target_expr_0_pos",
            ),
            (
                DataHandle::ViewStorage {
                    view: ViewId(2),
                    slot: ViewStorageSlot::Primary,
                },
                "view_2_primary",
            ),
            (
                DataHandle::EventRing {
                    ring: EventRingId(5),
                    kind: EventRingAccess::Read,
                },
                "event_ring_5_read",
            ),
            (
                DataHandle::ConfigConst {
                    id: ConfigConstId(11),
                },
                "config_11",
            ),
            (
                DataHandle::MaskBitmap { mask: MaskId(3) },
                "mask_3_bitmap",
            ),
            (DataHandle::ScoringOutput, "scoring_output"),
            (
                DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::GridCells,
                },
                "spatial_grid_cells",
            ),
            (
                DataHandle::Rng {
                    purpose: RngPurpose::Action,
                },
                "rng_action",
            ),
        ];
        for (h, expected) in cases {
            let id = push_expr(&mut prog, CgExpr::Read(h));
            let ctx = EmitCtx::structural(&prog);
            assert_eq!(
                lower_cg_expr_to_wgsl(id, &ctx).unwrap(),
                expected,
                "naming for variant {expected}"
            );
        }

        // Plumbing handles still get a structural name (defense-in-
        // depth — they should not appear in expressions but the strategy
        // must round-trip every variant).
        assert_eq!(structural_handle_name(&DataHandle::AliveBitmap), "alive_bitmap");
        assert_eq!(
            structural_handle_name(&DataHandle::IndirectArgs {
                ring: EventRingId(7)
            }),
            "indirect_args_7"
        );
        assert_eq!(
            structural_handle_name(&DataHandle::AgentScratch {
                kind: AgentScratchKind::Packed
            }),
            "agent_scratch_packed"
        );
        assert_eq!(structural_handle_name(&DataHandle::SimCfgBuffer), "sim_cfg_buffer");
        assert_eq!(structural_handle_name(&DataHandle::SnapshotKick), "snapshot_kick");
    }

    // ---- 6. Rng — every purpose ----

    #[test]
    fn lower_rng_every_purpose() {
        let mut prog = empty_prog();
        let cases = [
            (
                RngPurpose::Action,
                "per_agent_u32(seed, agent_id, tick, \"action\")",
            ),
            (
                RngPurpose::Sample,
                "per_agent_u32(seed, agent_id, tick, \"sample\")",
            ),
            (
                RngPurpose::Shuffle,
                "per_agent_u32(seed, agent_id, tick, \"shuffle\")",
            ),
            (
                RngPurpose::Conception,
                "per_agent_u32(seed, agent_id, tick, \"conception\")",
            ),
        ];
        for (purpose, expected) in cases {
            let id = push_expr(
                &mut prog,
                CgExpr::Rng {
                    purpose,
                    ty: CgTy::U32,
                },
            );
            let ctx = EmitCtx::structural(&prog);
            assert_eq!(lower_cg_expr_to_wgsl(id, &ctx).unwrap(), expected);
        }
    }

    // ---- 7. Select ----

    #[test]
    fn lower_select_emits_wgsl_select_with_false_first_order() {
        // select(true, hp, 0.0)
        // → WGSL: select(0.0, agent_self_hp, true)  -- false_val FIRST.
        let mut prog = empty_prog();
        let cond = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let sel = push_expr(
            &mut prog,
            CgExpr::Select {
                cond,
                then: hp,
                else_: zero,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(sel, &ctx).unwrap(),
            "select(0.0, agent_self_hp, true)"
        );
    }

    // ---- 8. Statement coverage ----

    #[test]
    fn lower_assign_stmt() {
        // assign(hp <- (hp + 1.0))
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let add = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: hp,
                rhs: one,
                ty: CgTy::F32,
            },
        );
        let s = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: add,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_stmt_to_wgsl(s, &ctx).unwrap(),
            "agent_self_hp = (agent_self_hp + 1.0);"
        );
    }

    #[test]
    fn lower_emit_stmt() {
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let s = push_stmt(
            &mut prog,
            CgStmt::Emit {
                event: EventKindId(7),
                fields: vec![
                    (
                        EventField {
                            event: EventKindId(7),
                            index: 0,
                        },
                        hp,
                    ),
                    (
                        EventField {
                            event: EventKindId(7),
                            index: 1,
                        },
                        zero,
                    ),
                ],
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_stmt_to_wgsl(s, &ctx).unwrap(),
            "emit_event_7(field_0: agent_self_hp, field_1: 0.0);"
        );
    }

    #[test]
    fn lower_if_with_and_without_else() {
        let mut prog = empty_prog();
        // assign hp <- 1.0
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let assign_one = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: one,
            },
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let assign_zero = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: zero,
            },
        );
        let then_list = push_list(&mut prog, CgStmtList::new(vec![assign_one]));
        let else_list = push_list(&mut prog, CgStmtList::new(vec![assign_zero]));
        let cond_lit = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));

        let if_with_else = push_stmt(
            &mut prog,
            CgStmt::If {
                cond: cond_lit,
                then: then_list,
                else_: Some(else_list),
            },
        );
        let if_no_else = push_stmt(
            &mut prog,
            CgStmt::If {
                cond: cond_lit,
                then: then_list,
                else_: None,
            },
        );

        let ctx = EmitCtx::structural(&prog);
        let with_else = lower_cg_stmt_to_wgsl(if_with_else, &ctx).unwrap();
        assert_eq!(
            with_else,
            "if (true) {\n    agent_self_hp = 1.0;\n} else {\n    agent_self_hp = 0.0;\n}"
        );

        let no_else = lower_cg_stmt_to_wgsl(if_no_else, &ctx).unwrap();
        assert_eq!(no_else, "if (true) {\n    agent_self_hp = 1.0;\n}");
    }

    #[test]
    fn lower_match_stmt_emits_if_chain() {
        // match hp { variant#0 { amount=local#0 } => assign(hp <- 1.0),
        //            variant#1 => assign(hp <- 0.0) }
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let arm0_assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: one,
            },
        );
        let arm1_assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: zero,
            },
        );
        let arm0_body = push_list(&mut prog, CgStmtList::new(vec![arm0_assign]));
        let arm1_body = push_list(&mut prog, CgStmtList::new(vec![arm1_assign]));
        let match_stmt = push_stmt(
            &mut prog,
            CgStmt::Match {
                scrutinee: hp,
                arms: vec![
                    CgMatchArm {
                        variant: VariantId(0),
                        bindings: vec![MatchArmBinding {
                            field_name: "amount".to_string(),
                            local: LocalId(0),
                        }],
                        body: arm0_body,
                    },
                    CgMatchArm {
                        variant: VariantId(1),
                        bindings: vec![],
                        body: arm1_body,
                    },
                ],
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let out = lower_cg_stmt_to_wgsl(match_stmt, &ctx).unwrap();
        // Scrutinee `hp` has CgExprId(0) → binding name `_scrut_0`.
        let expected = "let _scrut_0 = agent_self_hp;\n\
                        if (_scrut_0.tag == VARIANT_0u) { /* bindings: amount=local_0 from _scrut_0.amount */\n\
                        \x20\x20\x20\x20agent_self_hp = 1.0;\n\
                        } else if (_scrut_0.tag == VARIANT_1u) {\n\
                        \x20\x20\x20\x20agent_self_hp = 0.0;\n\
                        }";
        assert_eq!(out, expected);
    }

    /// Non-identifier scrutinee — verify the `let _scrut_<N> = (...);`
    /// binding makes the emission valid even when the scrutinee lowers
    /// to a parenthesised expression like `(agent_self_hp + 1.0)`.
    /// Without the binding, the old shape produced
    /// `((agent_self_hp + 1.0)_tag) == ...` which is invalid WGSL.
    #[test]
    fn lower_match_with_non_identifier_scrutinee_binds_local() {
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        // Scrutinee is `hp + 1.0` — lowers to `(agent_self_hp + 1.0)`.
        let scrutinee_expr = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: hp,
                rhs: one,
                ty: CgTy::F32,
            },
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let arm_assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: zero,
            },
        );
        let arm_body = push_list(&mut prog, CgStmtList::new(vec![arm_assign]));
        let match_stmt = push_stmt(
            &mut prog,
            CgStmt::Match {
                scrutinee: scrutinee_expr,
                arms: vec![CgMatchArm {
                    variant: VariantId(0),
                    bindings: vec![],
                    body: arm_body,
                }],
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let out = lower_cg_stmt_to_wgsl(match_stmt, &ctx).unwrap();
        // scrutinee_expr is the third pushed expression → CgExprId(2).
        let expected = "let _scrut_2 = (agent_self_hp + 1.0);\n\
                        if (_scrut_2.tag == VARIANT_0u) {\n\
                        \x20\x20\x20\x20agent_self_hp = 0.0;\n\
                        }";
        assert_eq!(out, expected);
    }

    // ---- 9. Snapshot test on a non-trivial expression ----

    /// Pin the lowered string of a non-trivial expression to detect
    /// drift in any of: literal formatting, infix bracketing, builtin
    /// naming, handle naming, select arg ordering.
    #[test]
    fn snapshot_select_clamp_distance_expression() {
        // select(
        //     hp < 5.0,
        //     clamp_f32(distance(self.pos, actor.pos), 0.0, 100.0),
        //     0.0,
        // )
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let five = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(5.0)));
        let cond = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: five,
                ty: CgTy::Bool,
            },
        );
        let pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let actor_pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Actor,
            }),
        );
        let dist = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Distance,
                args: vec![pos, actor_pos],
                ty: CgTy::F32,
            },
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let hundred = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(100.0)));
        let cl = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Clamp(NumericTy::F32),
                args: vec![dist, zero, hundred],
                ty: CgTy::F32,
            },
        );
        let zero2 = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let sel = push_expr(
            &mut prog,
            CgExpr::Select {
                cond,
                then: cl,
                else_: zero2,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(sel, &ctx).unwrap(),
            "select(0.0, \
             clamp_f32(distance(agent_self_pos, agent_actor_pos), 0.0, 100.0), \
             (agent_self_hp < 5.0))"
        );
    }

    // ---- 10. Determinism ----

    /// The same program must produce the same lowered string on every
    /// invocation — no `HashMap` ordering, no float locale, no random
    /// padding.
    #[test]
    fn wgsl_emit_is_deterministic() {
        let mut prog = empty_prog();
        let pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let normalize = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::NormalizeVec3F32,
                arg: pos,
                ty: CgTy::Vec3F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let first = lower_cg_expr_to_wgsl(normalize, &ctx).unwrap();
        for _ in 0..32 {
            assert_eq!(lower_cg_expr_to_wgsl(normalize, &ctx).unwrap(), first);
        }
    }

    /// Edge-case coverage for `format_f32_lit` — pin the legacy
    /// (`emit_view::format_f32_lit`) convention's output for the values
    /// most likely to surface differences with `{:?}` / `{}` alone.
    /// A regression here breaks Phase-5 byte-for-byte parity.
    #[test]
    fn format_f32_lit_edge_cases() {
        // Integer-valued: Display gives "1", we append ".0".
        assert_eq!(format_f32_lit(1.0), "1.0");
        assert_eq!(format_f32_lit(0.0), "0.0");
        assert_eq!(format_f32_lit(-1.0), "-1.0");
        assert_eq!(format_f32_lit(100.0), "100.0");
        // Sub-unit: Display already contains '.', return as-is.
        assert_eq!(format_f32_lit(0.5), "0.5");
        assert_eq!(format_f32_lit(-0.5), "-0.5");
        assert_eq!(format_f32_lit(1.5), "1.5");
        // Very large: Display fully expands, no '.' / 'e', append ".0".
        // Well-formed sim programs do not embed literals this large, but
        // the lowering must not panic on them.
        assert_eq!(
            format_f32_lit(1e30),
            "1000000000000000000000000000000.0"
        );
        // Very small (denormal-adjacent): Display contains '.', return
        // as-is — the literal's enormous length is a known caveat for
        // pathological inputs, not for well-formed programs.
        assert!(format_f32_lit(1e-30).contains('.'));
        assert!(format_f32_lit(1e-5).starts_with("0."));
        // f32::MIN_POSITIVE — sub-normal-adjacent. Same caveat.
        assert!(format_f32_lit(f32::MIN_POSITIVE).contains('.'));
    }

    // ---- 11. Error cases ----

    #[test]
    fn dangling_expr_id_returns_out_of_range() {
        let prog = empty_prog();
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_expr_to_wgsl(CgExprId(0), &ctx).unwrap_err();
        assert_eq!(
            err,
            EmitError::ExprIdOutOfRange {
                id: CgExprId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn dangling_stmt_id_returns_out_of_range() {
        let prog = empty_prog();
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_stmt_to_wgsl(CgStmtId(0), &ctx).unwrap_err();
        assert_eq!(
            err,
            EmitError::StmtIdOutOfRange {
                id: CgStmtId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn dangling_stmt_list_id_returns_out_of_range() {
        let prog = empty_prog();
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_stmt_list_to_wgsl(CgStmtListId(3), &ctx).unwrap_err();
        assert_eq!(
            err,
            EmitError::StmtListIdOutOfRange {
                id: CgStmtListId(3),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn nested_dangling_expr_inside_stmt_propagates() {
        // assign(hp <- expr#9) where expr#9 doesn't exist.
        let mut prog = empty_prog();
        let s = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(9),
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_stmt_to_wgsl(s, &ctx).unwrap_err();
        match err {
            EmitError::ExprIdOutOfRange { id, .. } => assert_eq!(id, CgExprId(9)),
            other => panic!("expected ExprIdOutOfRange, got {other:?}"),
        }
    }

    // ---- 12. Display impl on EmitError ----

    #[test]
    fn emit_error_display_each_variant() {
        let e1 = EmitError::ExprIdOutOfRange {
            id: CgExprId(7),
            arena_len: 3,
        };
        assert_eq!(
            format!("{}", e1),
            "CgExprId(#7) out of range (expr arena holds 3 entries)"
        );
        let e2 = EmitError::StmtIdOutOfRange {
            id: CgStmtId(1),
            arena_len: 0,
        };
        assert_eq!(
            format!("{}", e2),
            "CgStmtId(#1) out of range (stmt arena holds 0 entries)"
        );
        let e3 = EmitError::StmtListIdOutOfRange {
            id: CgStmtListId(4),
            arena_len: 2,
        };
        assert_eq!(
            format!("{}", e3),
            "CgStmtListId(#4) out of range (stmt-list arena holds 2 entries)"
        );
        let e4 = EmitError::UnsupportedHandle {
            handle: DataHandle::ScoringOutput,
            reason: "no slot",
        };
        assert_eq!(
            format!("{}", e4),
            "unsupported handle scoring.output: no slot"
        );
    }

    // ---- 13. Statement-list joining ----

    #[test]
    fn stmt_list_emits_newline_joined() {
        let mut prog = empty_prog();
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let two = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.0)));
        let s0 = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: one,
            },
        );
        let s1 = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::ShieldHp,
                    target: AgentRef::Self_,
                },
                value: two,
            },
        );
        let list = push_list(&mut prog, CgStmtList::new(vec![s0, s1]));
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_stmt_list_to_wgsl(list, &ctx).unwrap(),
            "agent_self_hp = 1.0;\nagent_self_shield_hp = 2.0;"
        );
    }

    #[test]
    fn stmt_list_empty_emits_empty_string() {
        let mut prog = empty_prog();
        let list = push_list(&mut prog, CgStmtList::new(vec![]));
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(lower_cg_stmt_list_to_wgsl(list, &ctx).unwrap(), "");
    }

    // ---- 14. cg_ty_to_wgsl spot-check ----

    #[test]
    fn cg_ty_to_wgsl_each_variant() {
        assert_eq!(cg_ty_to_wgsl(CgTy::Bool), "bool");
        assert_eq!(cg_ty_to_wgsl(CgTy::U32), "u32");
        assert_eq!(cg_ty_to_wgsl(CgTy::I32), "i32");
        assert_eq!(cg_ty_to_wgsl(CgTy::F32), "f32");
        assert_eq!(cg_ty_to_wgsl(CgTy::Vec3F32), "vec3<f32>");
        assert_eq!(cg_ty_to_wgsl(CgTy::AgentId), "u32");
        assert_eq!(cg_ty_to_wgsl(CgTy::Tick), "u32");
        assert_eq!(cg_ty_to_wgsl(CgTy::ViewKey { view: ViewId(2) }), "u32");
    }
}
