//! `CgExpr` — the compute-graph expression tree.
//!
//! `CgExpr` is the lowered form of every DSL expression — mask
//! predicates, scoring utilities, fold-body computations. It is
//! type-checked, side-effect-free, and reads simulation state through
//! the typed [`DataHandle`] vocabulary defined in Task 1.1.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 1.2, for the design rationale.
//!
//! # Arena vs node
//!
//! Expressions form a DAG; children are referenced by [`CgExprId`],
//! never embedded inline. The actual `Vec<CgExpr>` arena lives in
//! `CgProgram` (Task 1.5). This file defines only the *node* enum
//! and its supporting type / op enums — type-checking and
//! pretty-printing accept an external `id → expr` resolver so they
//! can run today against tiny test arenas before `CgProgram` exists.
//!
//! # `f32` and `Eq`
//!
//! `LitValue::F32` and `LitValue::Vec3F32` carry raw `f32` payloads.
//! `f32` is not `Eq` (`NaN != NaN`), so neither `LitValue` nor any
//! enum that contains it derives `Eq`. We derive `PartialEq` only;
//! the contract for tests is that two values compare equal iff every
//! field's bit pattern matches modulo IEEE-754 NaN. All literals used
//! in fixtures are non-NaN, so `PartialEq` suffices.

use std::fmt;

use serde::{Deserialize, Serialize};

use dsl_ast::ir::NamespaceId;

use super::data_handle::{CgExprId, DataHandle, RngPurpose, ViewId};
use super::op::EventKindId;

// ---------------------------------------------------------------------------
// CgTy — compute-graph types
// ---------------------------------------------------------------------------

/// The compute-graph type universe. Every `CgExpr` node carries a
/// `CgTy` (directly via the `ty:` field on the variants that have one,
/// or implicitly via the `LitValue` / `DataHandle` it wraps). Type
/// checking is `(operand types) → result type` per node; the
/// `type_check` walker enforces it.
///
/// The variant set covers the primitives the DSL surface produces
/// today (sourced from `docs/spec/dsl.md` §5.1 and `dsl_ast::IrType`).
/// New surface types add a variant here.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum CgTy {
    /// Boolean — mask predicates, conditional guards.
    Bool,
    /// Unsigned 32-bit integer — agent levels, monotonic counters.
    U32,
    /// Signed 32-bit integer — q8 fixed-point widened, signed math.
    I32,
    /// Single-precision float — vitals, ranges, multipliers, scores.
    F32,
    /// 3-component float vector — positions, displacements.
    Vec3F32,
    /// Opaque agent id (32-bit slot index, sentinel `0xFFFF_FFFF`).
    AgentId,
    /// Tick stamp — `world.tick`, expiry stamps. Wider than U32 in
    /// the engine (u64) but narrowed here because the GPU side uses
    /// u32 ticks; the IR carries the narrowed form because that's
    /// what every emit consumes. Surfaces as a *type tag* via
    /// `LitValue::Tick` and `ViewStorageSlot::Anchor`'s
    /// `data_handle_ty`; comparisons themselves use `BinaryOp::*U32`
    /// (the engine compares ticks as u32).
    Tick,
    /// "Key into view N" — the result type of `view::<name>(self, _)`
    /// reads. `view` records which materialized view this key
    /// references so later passes can resolve the storage layer.
    ViewKey { view: ViewId },
}

impl fmt::Display for CgTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CgTy::Bool => f.write_str("bool"),
            CgTy::U32 => f.write_str("u32"),
            CgTy::I32 => f.write_str("i32"),
            CgTy::F32 => f.write_str("f32"),
            CgTy::Vec3F32 => f.write_str("vec3<f32>"),
            CgTy::AgentId => f.write_str("agent_id"),
            CgTy::Tick => f.write_str("tick"),
            CgTy::ViewKey { view } => write!(f, "view_key<#{}>", view.0),
        }
    }
}

// ---------------------------------------------------------------------------
// LitValue — typed literal payload
// ---------------------------------------------------------------------------

/// Typed literal. Each variant pins its `CgTy` so a literal node can
/// carry its type without a separate `ty:` field. `f32`-carrying
/// variants block deriving `Eq`; we derive only `PartialEq`. See the
/// module-level `f32 and Eq` note.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LitValue {
    Bool(bool),
    U32(u32),
    I32(i32),
    F32(f32),
    Tick(u32),
    Vec3F32 { x: f32, y: f32, z: f32 },
    AgentId(u32),
}

impl LitValue {
    /// Type of this literal — pinned by the variant.
    pub fn ty(&self) -> CgTy {
        match self {
            LitValue::Bool(_) => CgTy::Bool,
            LitValue::U32(_) => CgTy::U32,
            LitValue::I32(_) => CgTy::I32,
            LitValue::F32(_) => CgTy::F32,
            LitValue::Tick(_) => CgTy::Tick,
            LitValue::Vec3F32 { .. } => CgTy::Vec3F32,
            LitValue::AgentId(_) => CgTy::AgentId,
        }
    }
}

impl fmt::Display for LitValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LitValue::Bool(b) => write!(f, "{}", b),
            LitValue::U32(v) => write!(f, "{}u32", v),
            LitValue::I32(v) => write!(f, "{}i32", v),
            LitValue::F32(v) => write!(f, "{:?}f32", v),
            LitValue::Tick(v) => write!(f, "{}tick", v),
            LitValue::Vec3F32 { x, y, z } => write!(f, "vec3<f32>({:?}, {:?}, {:?})", x, y, z),
            LitValue::AgentId(v) => write!(f, "agent#{}", v),
        }
    }
}

// ---------------------------------------------------------------------------
// BinaryOp — per-type binary variants
// ---------------------------------------------------------------------------

/// Binary operator. Per-type variants encode operand + result types
/// structurally — `AddF32` is `(F32, F32) -> F32`, `LtU32` is
/// `(U32, U32) -> Bool`, etc. The two helpers `operand_ty()` and
/// `result_ty()` are the single source of truth for those types and
/// are consulted by the type checker.
///
/// Numeric op variants cover `{F32, U32, I32}` — the types observed
/// in the actual DSL surface. Comparison ops additionally cover
/// `Tick` (for tick-stamp comparisons) and `AgentId` (`Eq`/`Ne` only).
/// Logical ops are `Bool`-only.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    // --- Arithmetic ---
    AddF32,
    SubF32,
    MulF32,
    DivF32,
    AddU32,
    SubU32,
    MulU32,
    DivU32,
    AddI32,
    SubI32,
    MulI32,
    DivI32,
    /// Modulo (remainder). Lowered from `BinOp::Mod` for f32/u32/i32
    /// operands. WGSL emits the native `%` operator. Used by
    /// cooldown-style mask gates (`tick % cooldown_ticks == 0u`) and
    /// the natural ability-system shape — see the abilities-probe
    /// discovery doc (`docs/superpowers/notes/2026-05-04-abilities_probe.md`,
    /// Gap #3).
    ModF32,
    ModU32,
    ModI32,

    // --- Vec3 arithmetic (Phase 7 boids fixture) ---
    //
    // Componentwise vec3+vec3 → vec3 and vec3-vec3 → vec3 — the two
    // shapes the boids steering math needs (`self.vel + zero_steer`,
    // `centroid - self.pos`). Scalar×vec3 is deferred until a fixture
    // actually needs weighted steering deltas.
    AddVec3,
    SubVec3,
    /// Asymmetric: lhs is `vec3<f32>`, rhs is `f32`; result is the
    /// per-component product `lhs * rhs`. WGSL emits this natively as
    /// `(vec3 * f32)`. Used by boids steering for weighted force
    /// composition (`alignment_force * alignment_weight`, etc.). The
    /// type-checker reads the asymmetric `(Vec3F32, F32)` operand
    /// pair from [`Self::operand_tys`].
    MulVec3ByF32,
    /// Asymmetric: lhs is `vec3<f32>`, rhs is `f32`; result is the
    /// per-component quotient `lhs / rhs`. WGSL emits this natively
    /// as `(vec3 / f32)`. Used by boids cohesion / alignment for
    /// "average position / velocity over neighborhood" — sum / count.
    DivVec3ByF32,

    // --- Ordered comparisons ---
    //
    // No `*Tick` variants — tick stamps are represented as `u32` once
    // they reach the IR (`AgentFieldTy::U32` for stamp fields,
    // `data_handle_ty` resolves `ViewStorageSlot::Anchor` to
    // `CgTy::Tick` but lowering coerces the comparison to `*U32` since
    // the engine compares ticks as u32). Tick comparisons use the
    // `*U32` variants.
    LtF32,
    LeF32,
    GtF32,
    GeF32,
    LtU32,
    LeU32,
    GtU32,
    GeU32,
    LtI32,
    LeI32,
    GtI32,
    GeI32,

    // --- Equality ---
    EqBool,
    EqU32,
    EqI32,
    EqF32,
    EqAgentId,
    NeBool,
    NeU32,
    NeI32,
    NeF32,
    NeAgentId,

    // --- Logical ---
    And,
    Or,
}

impl BinaryOp {
    /// Operand type — both `lhs` and `rhs` must have this type.
    pub fn operand_ty(self) -> CgTy {
        use BinaryOp::*;
        match self {
            AddF32 | SubF32 | MulF32 | DivF32 | ModF32 | LtF32 | LeF32 | GtF32 | GeF32 | EqF32
            | NeF32 => CgTy::F32,
            AddU32 | SubU32 | MulU32 | DivU32 | ModU32 | LtU32 | LeU32 | GtU32 | GeU32 | EqU32
            | NeU32 => CgTy::U32,
            AddI32 | SubI32 | MulI32 | DivI32 | ModI32 | LtI32 | LeI32 | GtI32 | GeI32 | EqI32
            | NeI32 => CgTy::I32,
            AddVec3 | SubVec3 | MulVec3ByF32 | DivVec3ByF32 => CgTy::Vec3F32,
            EqAgentId | NeAgentId => CgTy::AgentId,
            EqBool | NeBool | And | Or => CgTy::Bool,
        }
    }

    /// Operand-pair types — `(lhs_ty, rhs_ty)`. For symmetric ops
    /// returns `(operand_ty(), operand_ty())`; for the asymmetric
    /// vec3-by-scalar ops returns `(Vec3F32, F32)`. The type checker
    /// (and any future rewriter) consults this rather than
    /// [`Self::operand_ty`] when it needs each operand's expected
    /// type independently.
    pub fn operand_tys(self) -> (CgTy, CgTy) {
        use BinaryOp::*;
        match self {
            MulVec3ByF32 | DivVec3ByF32 => (CgTy::Vec3F32, CgTy::F32),
            _ => {
                let t = self.operand_ty();
                (t, t)
            }
        }
    }

    /// Result type — derived purely from the variant.
    pub fn result_ty(self) -> CgTy {
        use BinaryOp::*;
        match self {
            AddF32 | SubF32 | MulF32 | DivF32 | ModF32 => CgTy::F32,
            AddU32 | SubU32 | MulU32 | DivU32 | ModU32 => CgTy::U32,
            AddI32 | SubI32 | MulI32 | DivI32 | ModI32 => CgTy::I32,
            AddVec3 | SubVec3 | MulVec3ByF32 | DivVec3ByF32 => CgTy::Vec3F32,
            // Every comparison and logical op produces `Bool`.
            LtF32 | LeF32 | GtF32 | GeF32 | EqF32 | NeF32 | LtU32 | LeU32 | GtU32 | GeU32
            | EqU32 | NeU32 | LtI32 | LeI32 | GtI32 | GeI32 | EqI32 | NeI32 | EqAgentId
            | NeAgentId | EqBool | NeBool | And | Or => CgTy::Bool,
        }
    }

    /// Stable snake_case label for pretty-printing (`add.f32`,
    /// `lt.u32`, `eq.agent_id`, …). Decomposes into op-stem +
    /// type-suffix so the output reads consistently.
    pub fn label(self) -> &'static str {
        use BinaryOp::*;
        match self {
            AddF32 => "add.f32",
            SubF32 => "sub.f32",
            MulF32 => "mul.f32",
            DivF32 => "div.f32",
            AddU32 => "add.u32",
            SubU32 => "sub.u32",
            MulU32 => "mul.u32",
            DivU32 => "div.u32",
            AddI32 => "add.i32",
            SubI32 => "sub.i32",
            MulI32 => "mul.i32",
            DivI32 => "div.i32",
            ModF32 => "mod.f32",
            ModU32 => "mod.u32",
            ModI32 => "mod.i32",
            AddVec3 => "add.vec3",
            SubVec3 => "sub.vec3",
            MulVec3ByF32 => "mul.vec3.f32",
            DivVec3ByF32 => "div.vec3.f32",
            LtF32 => "lt.f32",
            LeF32 => "le.f32",
            GtF32 => "gt.f32",
            GeF32 => "ge.f32",
            LtU32 => "lt.u32",
            LeU32 => "le.u32",
            GtU32 => "gt.u32",
            GeU32 => "ge.u32",
            LtI32 => "lt.i32",
            LeI32 => "le.i32",
            GtI32 => "gt.i32",
            GeI32 => "ge.i32",
            EqBool => "eq.bool",
            EqU32 => "eq.u32",
            EqI32 => "eq.i32",
            EqF32 => "eq.f32",
            EqAgentId => "eq.agent_id",
            NeBool => "ne.bool",
            NeU32 => "ne.u32",
            NeI32 => "ne.i32",
            NeF32 => "ne.f32",
            NeAgentId => "ne.agent_id",
            And => "and",
            Or => "or",
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// UnaryOp — per-type unary variants
// ---------------------------------------------------------------------------

/// Unary operator. Per-type variants encode operand + result types
/// structurally. `NotBool` is the only logical unary; the rest are
/// numeric. `Normalize` operates on `Vec3F32` only — used by the
/// movement-direction lowering.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Boolean negation.
    NotBool,
    /// Numeric negation.
    NegF32,
    NegI32,
    /// Absolute value.
    AbsF32,
    AbsI32,
    /// Square root (f32 only — the DSL spec restricts `sqrt` to f32).
    SqrtF32,
    /// `vec3 / length(vec3)` — produces a unit vector. Length 0
    /// becomes `(0,0,0)` (the lowering inserts the safe form, matching
    /// the existing `normalize_or_zero` emit pattern).
    NormalizeVec3F32,
}

impl UnaryOp {
    pub fn operand_ty(self) -> CgTy {
        use UnaryOp::*;
        match self {
            NotBool => CgTy::Bool,
            NegF32 | AbsF32 | SqrtF32 => CgTy::F32,
            NegI32 | AbsI32 => CgTy::I32,
            NormalizeVec3F32 => CgTy::Vec3F32,
        }
    }

    pub fn result_ty(self) -> CgTy {
        // For every unary op the DSL surfaces, the result type matches
        // the operand type. (Any future variant whose result differs —
        // e.g. `length(vec3) -> f32` — should land as a Builtin call
        // instead of a unary; binary/unary stay shape-pure.)
        self.operand_ty()
    }

    pub fn label(self) -> &'static str {
        use UnaryOp::*;
        match self {
            NotBool => "not.bool",
            NegF32 => "neg.f32",
            NegI32 => "neg.i32",
            AbsF32 => "abs.f32",
            AbsI32 => "abs.i32",
            SqrtF32 => "sqrt.f32",
            NormalizeVec3F32 => "normalize.vec3<f32>",
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// BuiltinId — typed enumeration of every callable builtin
// ---------------------------------------------------------------------------

/// Element type for the polymorphic numeric builtins (`min`, `max`,
/// `clamp`, `saturating_add`). Restricted to the concrete numeric
/// types the DSL surface uses today — extending the set is a matter
/// of adding a variant here and a per-type `BuiltinId` variant below.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NumericTy {
    F32,
    U32,
    I32,
}

impl NumericTy {
    pub fn cg_ty(self) -> CgTy {
        match self {
            NumericTy::F32 => CgTy::F32,
            NumericTy::U32 => CgTy::U32,
            NumericTy::I32 => CgTy::I32,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            NumericTy::F32 => "f32",
            NumericTy::U32 => "u32",
            NumericTy::I32 => "i32",
        }
    }
}

impl fmt::Display for NumericTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Typed call-form builtins.
///
/// The variant set mirrors `dsl_ast::Builtin` (the canonical resolved
/// list — see `crates/dsl_ast/src/ir.rs::Builtin`) restricted to the
/// *call* shapes:
///
/// - Aggregations / quantifiers (`Count`, `Sum`, `Forall`, `Exists`)
///   are NOT call-form in the DSL parser — they're dedicated AST
///   nodes (`Fold`, `Quantifier`). They lower to compute-op-level
///   constructs (Task 1.3), not to `CgExpr::Builtin`.
/// - `Abs`, `Sqrt` are surfaced as [`UnaryOp`] variants; lowering
///   rewrites the AST `BuiltinCall(Abs, _)` form to `Unary { AbsF32 }`.
/// - `Min`/`Max`/`Clamp` are pairwise here (the iterator-fold form
///   becomes a `Fold` node, again lowered separately).
///
/// `ViewCall` is parametric over `ViewId`: each materialized view call
/// like `view::is_hostile(self, target)` resolves to a concrete view
/// at lowering time and becomes `Builtin { fn_id: ViewCall { view },
/// args: [...], ty: ViewKey { view } | <view's return ty> }`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum BuiltinId {
    // --- Spatial ---
    /// `(Vec3, Vec3) -> F32` — Euclidean.
    Distance,
    /// `(Vec3, Vec3) -> F32` — XY-plane.
    PlanarDistance,
    /// `(Vec3, Vec3) -> F32` — `abs(a.z - b.z)`.
    ZSeparation,

    // --- Numeric (pairwise) ---
    Min(NumericTy),
    Max(NumericTy),
    Clamp(NumericTy),
    SaturatingAdd(NumericTy),

    // --- Numeric (unary fns that don't fit the UnaryOp shape pure-shape rule) ---
    Floor,
    Ceil,
    Round,
    Ln,
    Log2,
    Log10,

    // --- ID dereference ---
    /// `(AgentId) -> AgentRow` — but at the CG layer the returned row
    /// is opaque; the only use is feeding back into `AgentField`
    /// reads via an enclosing dotted access. Kept as a typed builtin
    /// so the lowering can recognise it and rewrite to a direct
    /// agent-slot reference.
    Entity,

    // --- Materialized view call ---
    /// `view::<name>(self, target)` — `view` identifies which
    /// materialized view this references. The signature is "looked up
    /// by view id" at type-check time (Task 1.5's program holds the
    /// view table); the per-instance return type is recorded on the
    /// enclosing `CgExpr::Builtin { ty }` field.
    ViewCall { view: ViewId },

    // --- Constructors ---
    /// `vec3(x, y, z)` — pack three F32 components into a Vec3F32.
    /// Lowers to WGSL `vec3<f32>(x, y, z)`. Added 2026-05-02 to give
    /// the Boids fixture a vec3 literal form without going through
    /// `agents.pos(...)`. Phase-7-post-nuke unlock #1.
    Vec3Ctor,

    // --- Casts ---
    /// `f32(x)` where `x: U32 | I32` — promote an integer scalar to
    /// f32. Lowers to WGSL `f32(<arg>)`. Inserted implicitly by
    /// `lower_binary` when one operand is `F32` and the peer is `U32`
    /// or `I32` so mixed-type arith like `1000.0 - hp_u32` lowers as
    /// `1000.0 - f32(hp_u32)`. The carried [`NumericTy`] is the
    /// SOURCE type (`U32` or `I32`); `F32` is rejected (no-op cast)
    /// at construction by the only call site (the lowering helper).
    /// Closes Gap #2 from
    /// `docs/superpowers/notes/2026-05-04-pair_scoring_probe.md`.
    AsF32(NumericTy),
}

/// Typed signature of a builtin call. `args` is the list of expected
/// operand types in order; `result` is the result type. For most
/// builtins the signature is fixed once `BuiltinId` is known; for
/// `ViewCall` the signature is "look it up in the program's view
/// table" — we model this as `Signature::ViewCall { view }` so the
/// type checker can defer to the program context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuiltinSignature {
    /// Concrete signature: each arg has a fixed type, result has a
    /// fixed type.
    Fixed { args: Vec<CgTy>, result: CgTy },
    /// Deferred: the type checker should consult the program's view
    /// table (Task 1.5) for argument and return types of view N.
    ViewCall { view: ViewId },
}

impl BuiltinId {
    /// Typed signature. The type checker walks operand types against
    /// `signature().args` and validates the claimed result against
    /// `signature().result`. `ViewCall` returns the deferred form.
    pub fn signature(self) -> BuiltinSignature {
        use BuiltinId::*;
        match self {
            Distance | PlanarDistance | ZSeparation => BuiltinSignature::Fixed {
                args: vec![CgTy::Vec3F32, CgTy::Vec3F32],
                result: CgTy::F32,
            },
            Min(t) | Max(t) => BuiltinSignature::Fixed {
                args: vec![t.cg_ty(), t.cg_ty()],
                result: t.cg_ty(),
            },
            Clamp(t) => BuiltinSignature::Fixed {
                args: vec![t.cg_ty(), t.cg_ty(), t.cg_ty()],
                result: t.cg_ty(),
            },
            SaturatingAdd(t) => BuiltinSignature::Fixed {
                args: vec![t.cg_ty(), t.cg_ty()],
                result: t.cg_ty(),
            },
            Floor | Ceil | Round | Ln | Log2 | Log10 => BuiltinSignature::Fixed {
                args: vec![CgTy::F32],
                result: CgTy::F32,
            },
            Entity => BuiltinSignature::Fixed {
                args: vec![CgTy::AgentId],
                result: CgTy::AgentId,
            },
            ViewCall { view } => BuiltinSignature::ViewCall { view },
            Vec3Ctor => BuiltinSignature::Fixed {
                args: vec![CgTy::F32, CgTy::F32, CgTy::F32],
                result: CgTy::Vec3F32,
            },
            AsF32(t) => BuiltinSignature::Fixed {
                args: vec![t.cg_ty()],
                result: CgTy::F32,
            },
        }
    }

    /// Stable label for pretty-printing.
    pub fn label(self) -> String {
        use BuiltinId::*;
        match self {
            Distance => "distance".to_string(),
            PlanarDistance => "planar_distance".to_string(),
            ZSeparation => "z_separation".to_string(),
            Min(t) => format!("min.{}", t.label()),
            Max(t) => format!("max.{}", t.label()),
            Clamp(t) => format!("clamp.{}", t.label()),
            SaturatingAdd(t) => format!("saturating_add.{}", t.label()),
            Floor => "floor".to_string(),
            Ceil => "ceil".to_string(),
            Round => "round".to_string(),
            Ln => "ln".to_string(),
            Log2 => "log2".to_string(),
            Log10 => "log10".to_string(),
            Entity => "entity".to_string(),
            ViewCall { view } => format!("view_call.#{}", view.0),
            Vec3Ctor => "vec3".to_string(),
            AsF32(t) => format!("as_f32.{}", t.label()),
        }
    }
}

impl fmt::Display for BuiltinId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

// ---------------------------------------------------------------------------
// CgExpr — the tree node enum
// ---------------------------------------------------------------------------

/// Node in the compute-graph expression tree. Nodes reference their
/// children via [`CgExprId`] (resolved against an external arena);
/// they are flat structures.
///
/// Every variant whose result type is not pinned by its payload (i.e.
/// every variant except `Read` and `Lit`) carries an explicit `ty`
/// field. The type checker validates that the claimed `ty` matches
/// the operands' types under the operator's signature.
///
/// `Read`'s type is read off the [`DataHandle`] via
/// [`data_handle_ty`]; `Lit`'s type is read off the [`LitValue`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CgExpr {
    /// Read a piece of state.
    Read(DataHandle),
    /// Numeric / boolean literal.
    Lit(LitValue),
    /// Binary op. `op` carries the type of operands + result.
    Binary {
        op: BinaryOp,
        lhs: CgExprId,
        rhs: CgExprId,
        ty: CgTy,
    },
    /// Unary op (negate, not, abs, sqrt, normalize).
    Unary {
        op: UnaryOp,
        arg: CgExprId,
        ty: CgTy,
    },
    /// Built-in call — distance, dot, cross, normalize_or_zero,
    /// is_hostile, can_attack, etc. Each builtin is a typed enum
    /// variant (not a name string).
    Builtin {
        fn_id: BuiltinId,
        args: Vec<CgExprId>,
        ty: CgTy,
    },
    /// Per-agent RNG draw. `purpose` differentiates streams.
    Rng { purpose: RngPurpose, ty: CgTy },
    /// Conditional select (if-then-else as expression, not stmt).
    Select {
        cond: CgExprId,
        then: CgExprId,
        else_: CgExprId,
        ty: CgTy,
    },
    /// The current dispatch's actor as an `AgentId` value. Surfaces in
    /// the surface DSL as a bare `self` reference (e.g.,
    /// `agents.alive(self)`, `target != self`). Resolved by lowering
    /// `IrExpr::Local(_, "self")` when no `.field` access is present
    /// (the `self.<field>` path goes through
    /// `Read(AgentField { target: AgentRef::Self_, ... })`). Always
    /// typed `CgTy::AgentId`. Distinct from
    /// `Read(AgentField { target: AgentRef::Self_, ... })`: this
    /// variant carries the actor's *id*, not the read of any field.
    AgentSelfId,
    /// The per-pair candidate's `AgentId` value. Surfaces in
    /// pair-bound dispatch contexts (today only
    /// `mask <Name>(target) from query.nearby_agents(...)`
    /// predicates) where bare `target` appears
    /// (`agents.alive(target)`, `target != self`). Lowering only
    /// constructs this variant when `LoweringCtx::target_local` is
    /// `true`; outside pair-bound contexts the bare `target`
    /// reference surfaces as `LoweringError::UnsupportedLocalBinding`.
    /// Mirrors `AgentRef::PerPairCandidate`'s contract: the
    /// candidate's slot id is implicit in the surrounding
    /// `DispatchShape::PerPair { source }`; the IR layer just tags
    /// the read.
    PerPairCandidateId,
    /// Read a let-bound local. `local` resolves through the
    /// surrounding op's body — either an `IrStmt::Let` lowered to
    /// `CgStmt::Let { local, value, ty }` (Task 5.5b), or a future
    /// match-arm binding once those wire in. `ty` mirrors the
    /// binding's declared CG type so the type checker has the result
    /// type without needing to walk the binder.
    ///
    /// Lowering only constructs this variant when
    /// `IrExpr::Local(_, name)` resolves through
    /// `LoweringCtx::local_ids` (the typed `LocalRef → LocalId`
    /// registry). The read carries no `LocalRef` — once the lowering
    /// binds, only the typed `LocalId` flows in the IR.
    ReadLocal {
        local: crate::cg::stmt::LocalId,
        ty: CgTy,
    },
    /// Read a typed field from the current event's payload. Surfaces in
    /// PerEvent-shaped op bodies (physics-rule + view-fold handlers)
    /// where event-pattern bindings (`on EffectDamageApplied { actor: c,
    /// target: t, amount: a }`) introduce locals whose values come from
    /// the event record being processed by the current dispatch.
    ///
    /// Schema-driven: `event_kind` keys into
    /// [`super::program::CgProgram::event_layouts`] to resolve
    /// `(record_stride_u32, header_word_count, buffer_name,
    /// field_offset)`. Today every kind shares one ring with stride 10
    /// (2 header + 8 payload, sized for `AgentMoved` / `AgentFled`);
    /// future per-kind ring fanout returns per-kind buffer + stride
    /// without any IR shape change. The `word_offset_in_payload` is
    /// the field's 0-based u32-word offset within the event's payload
    /// (NOT including the 2-word header). For an event like
    /// `event Foo { a: Vec3, b: AgentId }`, the AgentId's
    /// `word_offset_in_payload` is `3` (Vec3 occupies 3 u32 words),
    /// not `1` — the value is a *word offset*, not a logical field
    /// index. Mirrors the source-of-truth field name on
    /// [`super::program::FieldLayout::word_offset_in_payload`] so the
    /// IR carries the same vocabulary the layout schema uses.
    ///
    /// Lowering only constructs this variant inside
    /// [`super::lower::physics::lower_one_handler`] (and the analogous
    /// fold-handler lowering) when synthesizing a `CgStmt::Let` per
    /// pattern binder. The well-formed pass flags `EventField` reads
    /// in non-PerEvent op bodies as
    /// [`super::well_formed::CgError::EventFieldInNonPerEventBody`].
    EventField {
        event_kind: EventKindId,
        word_offset_in_payload: u32,
        ty: CgTy,
    },
    /// Stdlib namespace-method call (e.g. `agents.is_hostile_to(target)`,
    /// `agents.engaged_with_or(target, fallback)`,
    /// `query.nearest_hostile_to_or(...)`). Schema-driven: `(ns,
    /// method)` keys into [`super::program::CgProgram::namespace_registry`]
    /// to resolve `(return_ty, arg_tys, wgsl_fn_name)`. Today's WGSL
    /// emit produces `<wgsl_fn_name>(<arg1>, <arg2>, ...)`; the kernel
    /// composer prepends a B1-stub prelude function for each distinct
    /// `(ns, method)` referenced by the kernel. Real implementations
    /// land in Task 9-11 territory (runtime-format work).
    ///
    /// Adding a new namespace method/field is a registry edit, not an IR
    /// change. The lowering surface and emitter walks consult the
    /// registry; only the registry's source-of-truth contents change.
    NamespaceCall {
        ns: NamespaceId,
        method: String,
        args: Vec<CgExprId>,
        ty: CgTy,
    },
    /// Stdlib namespace-field read (e.g. `world.tick`). Schema-driven:
    /// `(ns, field)` keys into
    /// [`super::program::CgProgram::namespace_registry`] to resolve
    /// `(ty, wgsl_access)`. Today's only entry is `world.tick`, which
    /// resolves to a kernel-preamble local (`tick`) bound by the fold
    /// kernel's preamble. The same shape supports
    /// `WgslAccessForm::UniformField` for cfg-bound fields without an
    /// IR change. Distinct from
    /// [`Self::Read`] of [`DataHandle::ConfigConst`] (which is the
    /// `config.<block>.<field>` shape and routes through a separate
    /// interner).
    NamespaceField {
        ns: NamespaceId,
        field: String,
        ty: CgTy,
    },
}

/// Compute the type a `DataHandle` reads at. `Read(h)` has the type
/// returned here; the type checker uses it to resolve operand types.
///
/// `DataHandle::ScoringOutput` and `DataHandle::SpatialStorage` carry
/// composite payloads (action+target+score, packed agent ids); their
/// "type" depends on which sub-field a later op extracts. Until the
/// op layer (Task 1.3) makes that decomposition explicit, we report
/// these as `U32` — the storage-element type. This is the same call
/// the existing emitters make.
///
/// # Panics
///
/// Plumbing-only handles (`AliveBitmap`, `IndirectArgs`,
/// `AgentScratch`, `SimCfgBuffer`, `SnapshotKick`) are not
/// expression-readable: they appear only on
/// [`crate::cg::op::PlumbingKind`] ops, whose reads/writes are
/// recorded structurally via
/// [`crate::cg::op::PlumbingKind::dependencies`] and never via embedded
/// `CgExpr::Read` nodes. `data_handle_ty` is an expression-typing
/// helper; reaching one of these arms means an `CgExpr::Read` was
/// constructed naming a plumbing handle, which violates the IR's
/// invariants. We `unreachable!` rather than coerce silently to U32 —
/// the storage element type would be a lie (e.g., `AliveBitmap` is a
/// packed bit array). This is a compile-time helper, not a
/// deterministic-runtime path, so the panic is the correct contract
/// per P10.
pub fn data_handle_ty(h: &DataHandle) -> CgTy {
    use crate::cg::data_handle::{
        DataHandle as H, EventRingAccess, SpatialStorageKind, ViewStorageSlot,
    };
    match h {
        H::AgentField { field, .. } => agent_field_ty_to_cg(field.ty()),
        H::ItemField { field, .. } => agent_field_ty_to_cg(field.ty),
        H::GroupField { field, .. } => agent_field_ty_to_cg(field.ty),
        H::ViewStorage { view, slot } => match slot {
            // The "primary" storage of any view is opaque at the IR
            // layer — its element type is view-shape-specific. The
            // ViewKey form lets later passes resolve via the program
            // table; until then it's the right phantom type.
            ViewStorageSlot::Primary => CgTy::ViewKey { view: *view },
            // Anchor slots are tick stamps; counts/cursors are u32;
            // ids are AgentId.
            ViewStorageSlot::Anchor => CgTy::Tick,
            ViewStorageSlot::Counts | ViewStorageSlot::Cursors => CgTy::U32,
            ViewStorageSlot::Ids => CgTy::AgentId,
        },
        H::EventRing { kind, .. } => match kind {
            // Reads pull a typed event record — opaque at this layer
            // (decomposition into fields is a later concern); appends
            // emit a record. We represent all three as U32 (the
            // underlying ring element type). `Drain` is the consumer
            // mode introduced by Task 2.7 plumbing.
            EventRingAccess::Read | EventRingAccess::Append | EventRingAccess::Drain => CgTy::U32,
        },
        H::ConfigConst { .. } => CgTy::F32,
        H::MaskBitmap { .. } => CgTy::Bool,
        H::ScoringOutput => CgTy::U32,
        H::SpatialStorage { kind } => match kind {
            SpatialStorageKind::GridCells | SpatialStorageKind::QueryResults => CgTy::AgentId,
            SpatialStorageKind::GridOffsets => CgTy::U32,
            // NonemptyCells holds compact cell indices, NonemptyCellsIndirectArgs
            // holds dispatch tuples, GridStarts holds prefix-scan
            // outputs — all u32 from the DSL's perspective. None of
            // these is read via DSL surface (the tiled MoveBoid emit
            // reads them through fixed WGSL templates, not via
            // `read(SpatialStorage{kind})`); the type annotation here
            // keeps the closed-set match exhaustive.
            SpatialStorageKind::NonemptyCells
            | SpatialStorageKind::NonemptyCellsIndirectArgs
            | SpatialStorageKind::GridStarts
            | SpatialStorageKind::ChunkSums => CgTy::U32,
        },
        H::Rng { purpose } => purpose.result_ty(),
        // Plumbing-only handles. These are touched only by
        // [`crate::cg::op::PlumbingKind`] ops, which carry no embedded
        // `CgExpr` reads (their reads/writes are sourced structurally
        // from `PlumbingKind::dependencies()`); the IR layer never
        // type-checks an expression that names one of these handles.
        // Per the doc comment's "Panics" section: returning a synthetic
        // type here would silently mask an invariant violation, so each
        // arm panics with a per-variant message instead.
        H::AliveBitmap => unreachable!(
            "plumbing handle 'AliveBitmap' is not expression-readable; \
             data_handle_ty should never see this in practice"
        ),
        H::IndirectArgs { .. } => unreachable!(
            "plumbing handle 'IndirectArgs' is not expression-readable; \
             data_handle_ty should never see this in practice"
        ),
        H::AgentScratch { .. } => unreachable!(
            "plumbing handle 'AgentScratch' is not expression-readable; \
             data_handle_ty should never see this in practice"
        ),
        H::SimCfgBuffer => unreachable!(
            "plumbing handle 'SimCfgBuffer' is not expression-readable; \
             data_handle_ty should never see this in practice"
        ),
        H::SnapshotKick => unreachable!(
            "plumbing handle 'SnapshotKick' is not expression-readable; \
             data_handle_ty should never see this in practice"
        ),
    }
}

/// Map an [`AgentFieldTy`] (the per-field primitive tag shared between
/// `AgentFieldId`, `ItemFieldId`, and `GroupFieldId`) to the
/// expression-typing [`CgTy`] surfaced by [`data_handle_ty`].
fn agent_field_ty_to_cg(ty: crate::cg::data_handle::AgentFieldTy) -> CgTy {
    use crate::cg::data_handle::AgentFieldTy;
    match ty {
        AgentFieldTy::F32 => CgTy::F32,
        AgentFieldTy::U32 => CgTy::U32,
        AgentFieldTy::I16 => CgTy::I32,
        AgentFieldTy::Bool => CgTy::Bool,
        AgentFieldTy::Vec3 => CgTy::Vec3F32,
        // Packed enum tags read as u32 on the GPU side; widening
        // happens at the binding boundary.
        AgentFieldTy::EnumU8 | AgentFieldTy::OptEnumU32 => CgTy::U32,
        // OptAgentId reads as AgentId at the IR level.
        AgentFieldTy::OptAgentId => CgTy::AgentId,
    }
}

impl CgExpr {
    /// The type a node evaluates to. For variants that carry an
    /// explicit `ty`, returns that. For `Read` and `Lit`, derives it
    /// from the payload.
    pub fn ty(&self) -> CgTy {
        match self {
            CgExpr::Read(h) => data_handle_ty(h),
            CgExpr::Lit(v) => v.ty(),
            CgExpr::Binary { ty, .. } => *ty,
            CgExpr::Unary { ty, .. } => *ty,
            CgExpr::Builtin { ty, .. } => *ty,
            CgExpr::Rng { ty, .. } => *ty,
            CgExpr::Select { ty, .. } => *ty,
            CgExpr::AgentSelfId => CgTy::AgentId,
            CgExpr::PerPairCandidateId => CgTy::AgentId,
            CgExpr::ReadLocal { ty, .. } => *ty,
            CgExpr::EventField { ty, .. } => *ty,
            CgExpr::NamespaceCall { ty, .. } => *ty,
            CgExpr::NamespaceField { ty, .. } => *ty,
        }
    }
}

// ---------------------------------------------------------------------------
// Pretty-printer
// ---------------------------------------------------------------------------

/// Resolver from a [`CgExprId`] to the underlying [`CgExpr`]. The
/// prog-arena (Task 1.5) implements this; tests pass a closure over a
/// `&[CgExpr]` slice.
///
/// The `Option<&CgExpr>` return surfaces out-of-range ids without
/// panicking — every caller (type checker, pretty-printer, well-formed
/// pass) handles the `None` arm explicitly. Closing the P10 panic gap
/// at the API layer keeps later passes (Phase 2 lowering) safe even on
/// partially-constructed arenas.
pub trait ExprArena {
    fn get(&self, id: CgExprId) -> Option<&CgExpr>;

    /// Optional refinement: surface the typed scalar for a
    /// [`DataHandle::ConfigConst`] read so the type checker can pick
    /// `CgTy::U32` / `CgTy::I32` / `CgTy::F32` based on the config
    /// field's declared type instead of the `data_handle_ty` default
    /// (which is always [`CgTy::F32`] because the bare
    /// [`crate::cg::data_handle::ConfigConstId`] doesn't carry a type
    /// tag). The default impl returns `None` so a slice / vec arena
    /// stays inert; [`crate::cg::program::CgProgram`] overrides to
    /// consult its `config_const_values` map. Closes Gap #3 from
    /// `docs/superpowers/notes/2026-05-04-diplomacy_probe.md` —
    /// `world.tick % config.<ns>.<u32_field>` typed correctly without
    /// a `BinaryOperandTyMismatch`.
    fn config_const_ty(
        &self,
        _id: crate::cg::data_handle::ConfigConstId,
    ) -> Option<CgTy> {
        None
    }
}

impl ExprArena for [CgExpr] {
    fn get(&self, id: CgExprId) -> Option<&CgExpr> {
        <[CgExpr]>::get(self, id.0 as usize)
    }
}

impl ExprArena for Vec<CgExpr> {
    fn get(&self, id: CgExprId) -> Option<&CgExpr> {
        <[CgExpr]>::get(self.as_slice(), id.0 as usize)
    }
}

/// Render `expr` as an s-expression, recursively resolving child ids
/// via `arena`. Output is deterministic and parseable-shaped:
///
/// ```text
/// (add.f32 (read agent.self.hp) (lit 1.0f32))
/// (builtin.distance (read agent.self.pos) (read agent.target(#0).pos))
/// (select (lit true) (lit 1u32) (lit 0u32))
/// ```
pub fn pretty(expr: &CgExpr, arena: &dyn ExprArena) -> String {
    let mut s = String::new();
    pretty_into(expr, arena, &mut s).expect("write to String never fails");
    s
}

/// Pretty-print a sub-expression by id. If the id is out-of-range
/// (corrupted arena), emit a `<oor:#N>` token rather than panicking —
/// the pretty-printer is a debugging aid and must tolerate the same
/// adversarial inputs the well-formed pass tolerates.
fn pretty_child(id: CgExprId, arena: &dyn ExprArena, out: &mut String) -> fmt::Result {
    use std::fmt::Write;
    match arena.get(id) {
        Some(child) => pretty_into(child, arena, out),
        None => write!(out, "<oor:#{}>", id.0),
    }
}

fn pretty_into(expr: &CgExpr, arena: &dyn ExprArena, out: &mut String) -> fmt::Result {
    use std::fmt::Write;
    match expr {
        CgExpr::Read(h) => write!(out, "(read {})", h),
        CgExpr::Lit(v) => write!(out, "(lit {})", v),
        CgExpr::Binary { op, lhs, rhs, .. } => {
            write!(out, "({} ", op)?;
            pretty_child(*lhs, arena, out)?;
            out.push(' ');
            pretty_child(*rhs, arena, out)?;
            out.push(')');
            Ok(())
        }
        CgExpr::Unary { op, arg, .. } => {
            write!(out, "({} ", op)?;
            pretty_child(*arg, arena, out)?;
            out.push(')');
            Ok(())
        }
        CgExpr::Builtin { fn_id, args, .. } => {
            write!(out, "(builtin.{}", fn_id)?;
            for a in args {
                out.push(' ');
                pretty_child(*a, arena, out)?;
            }
            out.push(')');
            Ok(())
        }
        CgExpr::Rng { purpose, .. } => write!(out, "(rng {})", purpose),
        CgExpr::Select {
            cond, then, else_, ..
        } => {
            out.push_str("(select ");
            pretty_child(*cond, arena, out)?;
            out.push(' ');
            pretty_child(*then, arena, out)?;
            out.push(' ');
            pretty_child(*else_, arena, out)?;
            out.push(')');
            Ok(())
        }
        CgExpr::AgentSelfId => write!(out, "(agent self_id)"),
        CgExpr::PerPairCandidateId => write!(out, "(agent per_pair_candidate_id)"),
        CgExpr::ReadLocal { local, ty } => write!(out, "(read_local {} {})", local, ty),
        CgExpr::EventField {
            event_kind,
            word_offset_in_payload,
            ty,
        } => write!(
            out,
            "(event_field event#{} word_off#{} {})",
            event_kind.0, word_offset_in_payload, ty
        ),
        CgExpr::NamespaceCall {
            ns,
            method,
            args,
            ty,
        } => {
            write!(out, "(ns_call {:?}.{} {}", ns, method, ty)?;
            for a in args {
                out.push(' ');
                pretty_child(*a, arena, out)?;
            }
            out.push(')');
            Ok(())
        }
        CgExpr::NamespaceField { ns, field, ty } => {
            write!(out, "(ns_field {:?}.{} {})", ns, field, ty)
        }
    }
}

// ---------------------------------------------------------------------------
// Type-check
// ---------------------------------------------------------------------------

/// Typed reasons a `CgExpr` tree can be ill-typed. Every variant names
/// the offending node and the structural mismatch in typed fields —
/// no `String` reasons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    /// A binary op's claimed result type doesn't match its variant's
    /// `result_ty()`.
    ClaimedResultMismatch {
        node: CgExprId,
        expected: CgTy,
        got: CgTy,
    },
    /// An operand's type doesn't match the op's required operand type.
    OperandMismatch {
        node: CgExprId,
        operand_index: u8,
        expected: CgTy,
        got: CgTy,
    },
    /// A builtin call had the wrong number of arguments.
    ArityMismatch {
        node: CgExprId,
        builtin: BuiltinId,
        expected: u8,
        got: u8,
    },
    /// `Select`'s `cond` operand was not `Bool`.
    SelectCondNotBool { node: CgExprId, got: CgTy },
    /// `Select`'s `then` and `else` arms had different types.
    SelectArmsMismatch {
        node: CgExprId,
        then_ty: CgTy,
        else_ty: CgTy,
    },
    /// A `ViewCall` builtin needs the program-level view table to
    /// resolve. The standalone type checker can't resolve it; tests
    /// that exercise `ViewCall` should use a `TypeCheckCtx` whose
    /// `view_signature` resolver is wired up.
    ViewSignatureUnresolved { node: CgExprId, view: ViewId },
    /// A child id referenced by `node` does not resolve in the arena —
    /// the recursive descent encountered an out-of-range id. Surfaced
    /// as a typed error rather than a panic so the well-formed pass
    /// (and any future caller running `type_check` on a partially-
    /// constructed program) handles arena corruption uniformly.
    DanglingExprId { node: CgExprId, referenced: CgExprId },
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeError::ClaimedResultMismatch {
                node,
                expected,
                got,
            } => write!(
                f,
                "expr#{} claims result {} but operands require {}",
                node.0, got, expected
            ),
            TypeError::OperandMismatch {
                node,
                operand_index,
                expected,
                got,
            } => write!(
                f,
                "expr#{} operand[{}] expected {}, got {}",
                node.0, operand_index, expected, got
            ),
            TypeError::ArityMismatch {
                node,
                builtin,
                expected,
                got,
            } => write!(
                f,
                "expr#{} builtin {} expected {} argument(s), got {}",
                node.0, builtin, expected, got
            ),
            TypeError::SelectCondNotBool { node, got } => write!(
                f,
                "expr#{} select cond expected bool, got {}",
                node.0, got
            ),
            TypeError::SelectArmsMismatch {
                node,
                then_ty,
                else_ty,
            } => write!(
                f,
                "expr#{} select arms mismatch — then is {}, else is {}",
                node.0, then_ty, else_ty
            ),
            TypeError::ViewSignatureUnresolved { node, view } => write!(
                f,
                "expr#{} view_call.#{} signature unresolved (no resolver wired)",
                node.0, view.0
            ),
            TypeError::DanglingExprId { node, referenced } => write!(
                f,
                "expr#{} references dangling expr#{}",
                node.0, referenced.0
            ),
        }
    }
}

/// Resolves a view's `(args, result)` signature. The standalone type
/// checker passes `None` here and gets `ViewSignatureUnresolved` for
/// any `ViewCall` it encounters; the program-level type-check (Task
/// 1.5) wires this up against the view table.
pub type ViewSignatureResolver<'a> = &'a dyn Fn(ViewId) -> Option<(Vec<CgTy>, CgTy)>;

/// Context passed to `type_check`. Bundles the arena (so the checker
/// can recurse into child ids) and the optional view resolver.
pub struct TypeCheckCtx<'a> {
    pub arena: &'a dyn ExprArena,
    pub view_signature: Option<ViewSignatureResolver<'a>>,
}

impl<'a> TypeCheckCtx<'a> {
    pub fn new(arena: &'a dyn ExprArena) -> Self {
        Self {
            arena,
            view_signature: None,
        }
    }

    pub fn with_view_signature(arena: &'a dyn ExprArena, resolver: ViewSignatureResolver<'a>) -> Self {
        Self {
            arena,
            view_signature: Some(resolver),
        }
    }
}

/// Shared arity / per-operand / claimed-result check used by both the
/// `Fixed` and `ViewCall` builtin paths. Pulled out so adding a new
/// builtin signature shape only adds one resolver branch — the
/// validation logic itself is single-source.
fn check_against_signature(
    args: &[CgExprId],
    want_args: &[CgTy],
    result: CgTy,
    fn_id: BuiltinId,
    claimed_ty: CgTy,
    node_id: CgExprId,
    ctx: &TypeCheckCtx<'_>,
) -> Result<CgTy, TypeError> {
    if args.len() != want_args.len() {
        return Err(TypeError::ArityMismatch {
            node: node_id,
            builtin: fn_id,
            expected: want_args.len() as u8,
            got: args.len() as u8,
        });
    }
    for (i, (arg_id, want)) in args.iter().zip(want_args.iter()).enumerate() {
        let arg_node = ctx.arena.get(*arg_id).ok_or(TypeError::DanglingExprId {
            node: node_id,
            referenced: *arg_id,
        })?;
        let arg_ty = type_check(arg_node, *arg_id, ctx)?;
        if arg_ty != *want {
            return Err(TypeError::OperandMismatch {
                node: node_id,
                operand_index: i as u8,
                expected: *want,
                got: arg_ty,
            });
        }
    }
    if claimed_ty != result {
        return Err(TypeError::ClaimedResultMismatch {
            node: node_id,
            expected: result,
            got: claimed_ty,
        });
    }
    Ok(result)
}

/// Type-check `expr` with `node_id` as its identity (used in error
/// reports). Returns `Ok(ty)` if the tree is well-typed, else a
/// `TypeError` naming the offending node + the typed mismatch.
///
/// The check is recursive: child nodes are themselves type-checked
/// before their type is consulted. A failure deep in the tree
/// surfaces as the deepest mismatch encountered.
///
/// **Arena tolerance.** If a child id resolves to `None` in the arena
/// (a dangling reference), the checker returns
/// [`TypeError::DanglingExprId`] rather than panicking. Callers no
/// longer need a separate pre-pass to gate `type_check` against
/// out-of-range ids — the typed-error form makes the API panic-free.
pub fn type_check(
    expr: &CgExpr,
    node_id: CgExprId,
    ctx: &TypeCheckCtx<'_>,
) -> Result<CgTy, TypeError> {
    match expr {
        CgExpr::Read(h) => {
            // Refine ConfigConst reads via the arena's typed registry
            // (CgProgram-side config_const_values). The bare
            // `data_handle_ty(ConfigConst { id })` defaults every
            // config field to F32 because the id alone has no type
            // tag; the arena-aware path consults the registered
            // ConfigConstValue variant and returns U32 / I32 / F32
            // accordingly. Closes Gap #3 from
            // `docs/superpowers/notes/2026-05-04-diplomacy_probe.md`.
            if let crate::cg::data_handle::DataHandle::ConfigConst { id } = h {
                if let Some(refined) = ctx.arena.config_const_ty(*id) {
                    return Ok(refined);
                }
            }
            Ok(data_handle_ty(h))
        }
        CgExpr::Lit(v) => Ok(v.ty()),

        CgExpr::Binary { op, lhs, rhs, ty } => {
            let lhs_node = ctx.arena.get(*lhs).ok_or(TypeError::DanglingExprId {
                node: node_id,
                referenced: *lhs,
            })?;
            let rhs_node = ctx.arena.get(*rhs).ok_or(TypeError::DanglingExprId {
                node: node_id,
                referenced: *rhs,
            })?;
            let lhs_ty = type_check(lhs_node, *lhs, ctx)?;
            let rhs_ty = type_check(rhs_node, *rhs, ctx)?;
            let (want_lhs, want_rhs) = op.operand_tys();
            if lhs_ty != want_lhs {
                return Err(TypeError::OperandMismatch {
                    node: node_id,
                    operand_index: 0,
                    expected: want_lhs,
                    got: lhs_ty,
                });
            }
            if rhs_ty != want_rhs {
                return Err(TypeError::OperandMismatch {
                    node: node_id,
                    operand_index: 1,
                    expected: want_rhs,
                    got: rhs_ty,
                });
            }
            let result = op.result_ty();
            if *ty != result {
                return Err(TypeError::ClaimedResultMismatch {
                    node: node_id,
                    expected: result,
                    got: *ty,
                });
            }
            Ok(result)
        }

        CgExpr::Unary { op, arg, ty } => {
            let arg_node = ctx.arena.get(*arg).ok_or(TypeError::DanglingExprId {
                node: node_id,
                referenced: *arg,
            })?;
            let arg_ty = type_check(arg_node, *arg, ctx)?;
            let want = op.operand_ty();
            if arg_ty != want {
                return Err(TypeError::OperandMismatch {
                    node: node_id,
                    operand_index: 0,
                    expected: want,
                    got: arg_ty,
                });
            }
            let result = op.result_ty();
            if *ty != result {
                return Err(TypeError::ClaimedResultMismatch {
                    node: node_id,
                    expected: result,
                    got: *ty,
                });
            }
            Ok(result)
        }

        CgExpr::Builtin { fn_id, args, ty } => match fn_id.signature() {
            BuiltinSignature::Fixed {
                args: want_args,
                result,
            } => {
                if args.len() != want_args.len() {
                    return Err(TypeError::ArityMismatch {
                        node: node_id,
                        builtin: *fn_id,
                        expected: want_args.len() as u8,
                        got: args.len() as u8,
                    });
                }
                check_against_signature(args, &want_args, result, *fn_id, *ty, node_id, ctx)
            }
            BuiltinSignature::ViewCall { view } => {
                let resolver = ctx.view_signature.ok_or(TypeError::ViewSignatureUnresolved {
                    node: node_id,
                    view,
                })?;
                let (want_args, result) =
                    resolver(view).ok_or(TypeError::ViewSignatureUnresolved {
                        node: node_id,
                        view,
                    })?;
                check_against_signature(args, &want_args, result, *fn_id, *ty, node_id, ctx)
            }
        },

        CgExpr::Rng { purpose, ty } => {
            // Each `RngPurpose` has a natural result type:
            //  - `Action` / `Sample` / `Shuffle` / `Conception` →
            //    `U32` (the raw `per_agent_u32` draw, exposed by the
            //    lower-level surface).
            //  - `Uniform` / `Gauss` → `F32` (unit interval / standard
            //    normal — the WGSL emit performs the conversion from
            //    the underlying `u32` draw).
            //  - `Coin` → `Bool`.
            //  - `UniformInt` → `I32` (raw bits bitcast).
            //
            // The lowering pass is responsible for setting the claimed
            // `ty` to `purpose.result_ty()`; this check enforces the
            // invariant.
            let expected = purpose.result_ty();
            if *ty != expected {
                return Err(TypeError::ClaimedResultMismatch {
                    node: node_id,
                    expected,
                    got: *ty,
                });
            }
            Ok(expected)
        }

        CgExpr::Select {
            cond,
            then,
            else_,
            ty,
        } => {
            let cond_node = ctx.arena.get(*cond).ok_or(TypeError::DanglingExprId {
                node: node_id,
                referenced: *cond,
            })?;
            let cond_ty = type_check(cond_node, *cond, ctx)?;
            if cond_ty != CgTy::Bool {
                return Err(TypeError::SelectCondNotBool {
                    node: node_id,
                    got: cond_ty,
                });
            }
            let then_node = ctx.arena.get(*then).ok_or(TypeError::DanglingExprId {
                node: node_id,
                referenced: *then,
            })?;
            let then_ty = type_check(then_node, *then, ctx)?;
            let else_node = ctx.arena.get(*else_).ok_or(TypeError::DanglingExprId {
                node: node_id,
                referenced: *else_,
            })?;
            let else_ty = type_check(else_node, *else_, ctx)?;
            if then_ty != else_ty {
                return Err(TypeError::SelectArmsMismatch {
                    node: node_id,
                    then_ty,
                    else_ty,
                });
            }
            if *ty != then_ty {
                return Err(TypeError::ClaimedResultMismatch {
                    node: node_id,
                    expected: then_ty,
                    got: *ty,
                });
            }
            Ok(then_ty)
        }

        CgExpr::AgentSelfId => Ok(CgTy::AgentId),
        CgExpr::PerPairCandidateId => Ok(CgTy::AgentId),
        CgExpr::ReadLocal { ty, .. } => Ok(*ty),
        // EventField carries its own claimed type; the lowering pins
        // this from the event's `FieldLayout::ty`. The well-formed pass
        // separately verifies the schema entry exists for `(event_kind,
        // word_offset_in_payload)`, so this arm just trusts the
        // claimed type — a fabricated mismatch is a builder defect,
        // not a typing one.
        CgExpr::EventField { ty, .. } => Ok(*ty),
        // NamespaceCall / NamespaceField carry claimed types pinned by
        // the registry-driven lowering. Operand types are not validated
        // here — the lowering already type-checked each argument's
        // expression; the registry's `arg_tys` schema is structural
        // metadata for downstream emit, not a typing-rule input. A
        // fabricated mismatch (claimed return type ≠ registry return
        // type) is a builder defect, not a typing one.
        CgExpr::NamespaceCall { ty, .. } => Ok(*ty),
        CgExpr::NamespaceField { ty, .. } => Ok(*ty),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{AgentFieldId, AgentRef, MaskId};

    // ---- helpers ----

    /// Round-trip a value through serde JSON and assert structural
    /// equality. Equality means `PartialEq` for types containing
    /// `f32`; full `Eq` otherwise.
    fn assert_roundtrip<T>(v: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let json = serde_json::to_string(v).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, v, "round-trip changed value (json was {json})");
    }

    fn read_self_hp() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        })
    }

    fn read_self_pos() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Pos,
            target: AgentRef::Self_,
        })
    }

    // ---- CgTy ----

    #[test]
    fn cg_ty_display_distinct_per_variant() {
        let cases = [
            (CgTy::Bool, "bool"),
            (CgTy::U32, "u32"),
            (CgTy::I32, "i32"),
            (CgTy::F32, "f32"),
            (CgTy::Vec3F32, "vec3<f32>"),
            (CgTy::AgentId, "agent_id"),
            (CgTy::Tick, "tick"),
            (CgTy::ViewKey { view: ViewId(7) }, "view_key<#7>"),
        ];
        for (ty, expected) in cases {
            assert_eq!(format!("{}", ty), expected);
            assert_roundtrip(&ty);
        }
    }

    // ---- LitValue ----

    #[test]
    fn lit_value_ty_and_display_match() {
        let cases = [
            (LitValue::Bool(true), CgTy::Bool, "true"),
            (LitValue::U32(7), CgTy::U32, "7u32"),
            (LitValue::I32(-3), CgTy::I32, "-3i32"),
            (LitValue::Tick(42), CgTy::Tick, "42tick"),
            (LitValue::AgentId(0xFF), CgTy::AgentId, "agent#255"),
        ];
        for (lit, ty, label) in cases {
            assert_eq!(lit.ty(), ty);
            assert_eq!(format!("{}", lit), label);
            assert_roundtrip(&lit);
        }
    }

    #[test]
    fn lit_value_f32_display_uses_debug_form() {
        // 1.5 prints as "1.5f32" via Debug formatting; a NaN-free
        // round-trip is the contract.
        let lit = LitValue::F32(1.5);
        assert_eq!(lit.ty(), CgTy::F32);
        assert_eq!(format!("{}", lit), "1.5f32");
        assert_roundtrip(&lit);
    }

    #[test]
    fn lit_value_vec3_display_and_roundtrip() {
        let lit = LitValue::Vec3F32 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        assert_eq!(lit.ty(), CgTy::Vec3F32);
        assert_eq!(format!("{}", lit), "vec3<f32>(1.0, 2.0, 3.0)");
        assert_roundtrip(&lit);
    }

    // ---- BinaryOp ----

    #[test]
    fn binary_op_signatures_consistent() {
        // Arithmetic: same operand and result type.
        assert_eq!(BinaryOp::AddF32.operand_ty(), CgTy::F32);
        assert_eq!(BinaryOp::AddF32.result_ty(), CgTy::F32);
        assert_eq!(BinaryOp::AddU32.operand_ty(), CgTy::U32);
        assert_eq!(BinaryOp::AddU32.result_ty(), CgTy::U32);
        assert_eq!(BinaryOp::SubI32.operand_ty(), CgTy::I32);
        assert_eq!(BinaryOp::SubI32.result_ty(), CgTy::I32);

        // Comparison: operand-typed, result Bool.
        assert_eq!(BinaryOp::LtF32.operand_ty(), CgTy::F32);
        assert_eq!(BinaryOp::LtF32.result_ty(), CgTy::Bool);
        assert_eq!(BinaryOp::EqAgentId.operand_ty(), CgTy::AgentId);
        assert_eq!(BinaryOp::EqAgentId.result_ty(), CgTy::Bool);
        // Tick comparisons reuse the U32 variants — see the BinaryOp
        // doc comment for the rationale.
        assert_eq!(BinaryOp::GtU32.operand_ty(), CgTy::U32);

        // Logical: Bool/Bool/Bool.
        assert_eq!(BinaryOp::And.operand_ty(), CgTy::Bool);
        assert_eq!(BinaryOp::And.result_ty(), CgTy::Bool);
        assert_eq!(BinaryOp::Or.result_ty(), CgTy::Bool);
    }

    #[test]
    fn binary_op_display_labels_distinct() {
        let ops = [
            BinaryOp::AddF32,
            BinaryOp::SubF32,
            BinaryOp::MulF32,
            BinaryOp::DivF32,
            BinaryOp::AddU32,
            BinaryOp::SubU32,
            BinaryOp::MulU32,
            BinaryOp::DivU32,
            BinaryOp::AddI32,
            BinaryOp::SubI32,
            BinaryOp::MulI32,
            BinaryOp::DivI32,
            BinaryOp::ModF32,
            BinaryOp::ModU32,
            BinaryOp::ModI32,
            BinaryOp::LtF32,
            BinaryOp::LeF32,
            BinaryOp::GtF32,
            BinaryOp::GeF32,
            BinaryOp::LtU32,
            BinaryOp::LeU32,
            BinaryOp::GtU32,
            BinaryOp::GeU32,
            BinaryOp::LtI32,
            BinaryOp::LeI32,
            BinaryOp::GtI32,
            BinaryOp::GeI32,
            BinaryOp::EqBool,
            BinaryOp::EqU32,
            BinaryOp::EqI32,
            BinaryOp::EqF32,
            BinaryOp::EqAgentId,
            BinaryOp::NeBool,
            BinaryOp::NeU32,
            BinaryOp::NeI32,
            BinaryOp::NeF32,
            BinaryOp::NeAgentId,
            BinaryOp::And,
            BinaryOp::Or,
        ];
        let mut seen = std::collections::HashSet::new();
        for op in ops {
            let label = op.label();
            assert!(seen.insert(label), "duplicate BinaryOp label: {label}");
            assert_eq!(format!("{}", op), label);
            assert_roundtrip(&op);
        }
    }

    // ---- UnaryOp ----

    #[test]
    fn unary_op_signatures_consistent() {
        assert_eq!(UnaryOp::NotBool.operand_ty(), CgTy::Bool);
        assert_eq!(UnaryOp::NotBool.result_ty(), CgTy::Bool);
        assert_eq!(UnaryOp::NegF32.operand_ty(), CgTy::F32);
        assert_eq!(UnaryOp::NegF32.result_ty(), CgTy::F32);
        assert_eq!(UnaryOp::AbsI32.operand_ty(), CgTy::I32);
        assert_eq!(UnaryOp::AbsI32.result_ty(), CgTy::I32);
        assert_eq!(UnaryOp::SqrtF32.result_ty(), CgTy::F32);
        assert_eq!(UnaryOp::NormalizeVec3F32.operand_ty(), CgTy::Vec3F32);
        assert_eq!(UnaryOp::NormalizeVec3F32.result_ty(), CgTy::Vec3F32);
    }

    #[test]
    fn unary_op_display_labels_distinct() {
        let ops = [
            UnaryOp::NotBool,
            UnaryOp::NegF32,
            UnaryOp::NegI32,
            UnaryOp::AbsF32,
            UnaryOp::AbsI32,
            UnaryOp::SqrtF32,
            UnaryOp::NormalizeVec3F32,
        ];
        let mut seen = std::collections::HashSet::new();
        for op in ops {
            let label = op.label();
            assert!(seen.insert(label), "duplicate UnaryOp label: {label}");
            assert_eq!(format!("{}", op), label);
            assert_roundtrip(&op);
        }
    }

    // ---- BuiltinId ----

    #[test]
    fn builtin_signatures() {
        let cases: &[(BuiltinId, BuiltinSignature)] = &[
            (
                BuiltinId::Distance,
                BuiltinSignature::Fixed {
                    args: vec![CgTy::Vec3F32, CgTy::Vec3F32],
                    result: CgTy::F32,
                },
            ),
            (
                BuiltinId::PlanarDistance,
                BuiltinSignature::Fixed {
                    args: vec![CgTy::Vec3F32, CgTy::Vec3F32],
                    result: CgTy::F32,
                },
            ),
            (
                BuiltinId::ZSeparation,
                BuiltinSignature::Fixed {
                    args: vec![CgTy::Vec3F32, CgTy::Vec3F32],
                    result: CgTy::F32,
                },
            ),
            (
                BuiltinId::Min(NumericTy::F32),
                BuiltinSignature::Fixed {
                    args: vec![CgTy::F32, CgTy::F32],
                    result: CgTy::F32,
                },
            ),
            (
                BuiltinId::Max(NumericTy::U32),
                BuiltinSignature::Fixed {
                    args: vec![CgTy::U32, CgTy::U32],
                    result: CgTy::U32,
                },
            ),
            (
                BuiltinId::Clamp(NumericTy::I32),
                BuiltinSignature::Fixed {
                    args: vec![CgTy::I32, CgTy::I32, CgTy::I32],
                    result: CgTy::I32,
                },
            ),
            (
                BuiltinId::SaturatingAdd(NumericTy::U32),
                BuiltinSignature::Fixed {
                    args: vec![CgTy::U32, CgTy::U32],
                    result: CgTy::U32,
                },
            ),
            (
                BuiltinId::Floor,
                BuiltinSignature::Fixed {
                    args: vec![CgTy::F32],
                    result: CgTy::F32,
                },
            ),
            (
                BuiltinId::Ln,
                BuiltinSignature::Fixed {
                    args: vec![CgTy::F32],
                    result: CgTy::F32,
                },
            ),
            (
                BuiltinId::Entity,
                BuiltinSignature::Fixed {
                    args: vec![CgTy::AgentId],
                    result: CgTy::AgentId,
                },
            ),
            (
                BuiltinId::ViewCall { view: ViewId(2) },
                BuiltinSignature::ViewCall { view: ViewId(2) },
            ),
        ];
        for (id, sig) in cases {
            assert_eq!(&id.signature(), sig);
            assert_roundtrip(id);
        }
    }

    #[test]
    fn builtin_display_labels() {
        assert_eq!(format!("{}", BuiltinId::Distance), "distance");
        assert_eq!(format!("{}", BuiltinId::PlanarDistance), "planar_distance");
        assert_eq!(format!("{}", BuiltinId::ZSeparation), "z_separation");
        assert_eq!(format!("{}", BuiltinId::Min(NumericTy::F32)), "min.f32");
        assert_eq!(format!("{}", BuiltinId::Max(NumericTy::U32)), "max.u32");
        assert_eq!(format!("{}", BuiltinId::Clamp(NumericTy::I32)), "clamp.i32");
        assert_eq!(
            format!("{}", BuiltinId::SaturatingAdd(NumericTy::U32)),
            "saturating_add.u32"
        );
        assert_eq!(format!("{}", BuiltinId::Floor), "floor");
        assert_eq!(format!("{}", BuiltinId::Ceil), "ceil");
        assert_eq!(format!("{}", BuiltinId::Round), "round");
        assert_eq!(format!("{}", BuiltinId::Ln), "ln");
        assert_eq!(format!("{}", BuiltinId::Log2), "log2");
        assert_eq!(format!("{}", BuiltinId::Log10), "log10");
        assert_eq!(format!("{}", BuiltinId::Entity), "entity");
        assert_eq!(
            format!("{}", BuiltinId::ViewCall { view: ViewId(11) }),
            "view_call.#11"
        );
    }

    // ---- CgExpr round-trip ----

    #[test]
    fn cg_expr_read_roundtrip() {
        let e = read_self_hp();
        assert_eq!(e.ty(), CgTy::F32);
        assert_roundtrip(&e);
    }

    #[test]
    fn cg_expr_lit_roundtrip() {
        let e = CgExpr::Lit(LitValue::F32(2.5));
        assert_eq!(e.ty(), CgTy::F32);
        assert_roundtrip(&e);
    }

    #[test]
    fn cg_expr_binary_roundtrip_and_ty() {
        let e = CgExpr::Binary {
            op: BinaryOp::AddF32,
            lhs: CgExprId(0),
            rhs: CgExprId(1),
            ty: CgTy::F32,
        };
        assert_eq!(e.ty(), CgTy::F32);
        assert_roundtrip(&e);
    }

    #[test]
    fn cg_expr_unary_roundtrip_and_ty() {
        let e = CgExpr::Unary {
            op: UnaryOp::NegF32,
            arg: CgExprId(0),
            ty: CgTy::F32,
        };
        assert_eq!(e.ty(), CgTy::F32);
        assert_roundtrip(&e);
    }

    #[test]
    fn cg_expr_builtin_roundtrip_and_ty() {
        let e = CgExpr::Builtin {
            fn_id: BuiltinId::Distance,
            args: vec![CgExprId(0), CgExprId(1)],
            ty: CgTy::F32,
        };
        assert_eq!(e.ty(), CgTy::F32);
        assert_roundtrip(&e);
    }

    #[test]
    fn cg_expr_rng_roundtrip_and_ty() {
        let e = CgExpr::Rng {
            purpose: RngPurpose::Action,
            ty: CgTy::U32,
        };
        assert_eq!(e.ty(), CgTy::U32);
        assert_roundtrip(&e);
    }

    #[test]
    fn cg_expr_select_roundtrip_and_ty() {
        let e = CgExpr::Select {
            cond: CgExprId(0),
            then: CgExprId(1),
            else_: CgExprId(2),
            ty: CgTy::F32,
        };
        assert_eq!(e.ty(), CgTy::F32);
        assert_roundtrip(&e);
    }

    // ---- Pretty-print ----

    #[test]
    fn pretty_read_lit_binary() {
        // `agent.self.hp + 1.0`
        let arena: Vec<CgExpr> = vec![
            read_self_hp(),
            CgExpr::Lit(LitValue::F32(1.0)),
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            },
        ];
        let printed = pretty(&arena[2], &arena);
        assert_eq!(printed, "(add.f32 (read agent.self.hp) (lit 1.0f32))");
    }

    #[test]
    fn pretty_unary() {
        let arena: Vec<CgExpr> = vec![
            CgExpr::Read(DataHandle::MaskBitmap { mask: MaskId(3) }),
            CgExpr::Unary {
                op: UnaryOp::NotBool,
                arg: CgExprId(0),
                ty: CgTy::Bool,
            },
        ];
        let printed = pretty(&arena[1], &arena);
        assert_eq!(printed, "(not.bool (read mask[#3].bitmap))");
    }

    #[test]
    fn pretty_builtin_distance() {
        // `distance(agent.self.pos, agent.self.pos)` (a degenerate
        // call; we only care about the printed form here).
        let arena: Vec<CgExpr> = vec![
            read_self_pos(),
            read_self_pos(),
            CgExpr::Builtin {
                fn_id: BuiltinId::Distance,
                args: vec![CgExprId(0), CgExprId(1)],
                ty: CgTy::F32,
            },
        ];
        let printed = pretty(&arena[2], &arena);
        assert_eq!(
            printed,
            "(builtin.distance (read agent.self.pos) (read agent.self.pos))"
        );
    }

    #[test]
    fn pretty_rng() {
        let e = CgExpr::Rng {
            purpose: RngPurpose::Sample,
            ty: CgTy::U32,
        };
        let arena: Vec<CgExpr> = vec![e.clone()];
        assert_eq!(pretty(&e, &arena), "(rng sample)");
    }

    #[test]
    fn pretty_select() {
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::Bool(true)),
            CgExpr::Lit(LitValue::U32(1)),
            CgExpr::Lit(LitValue::U32(0)),
            CgExpr::Select {
                cond: CgExprId(0),
                then: CgExprId(1),
                else_: CgExprId(2),
                ty: CgTy::U32,
            },
        ];
        assert_eq!(
            pretty(&arena[3], &arena),
            "(select (lit true) (lit 1u32) (lit 0u32))"
        );
    }

    // ---- Type-check happy path ----

    #[test]
    fn type_check_self_hp_plus_one_is_f32() {
        // agent.self.hp + 1.0 — F32 + F32 -> F32.
        let arena: Vec<CgExpr> = vec![
            read_self_hp(),
            CgExpr::Lit(LitValue::F32(1.0)),
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let ty = type_check(&arena[2], CgExprId(2), &ctx).expect("well-typed");
        assert_eq!(ty, CgTy::F32);
    }

    #[test]
    fn type_check_distance_call_is_f32() {
        let arena: Vec<CgExpr> = vec![
            read_self_pos(),
            read_self_pos(),
            CgExpr::Builtin {
                fn_id: BuiltinId::Distance,
                args: vec![CgExprId(0), CgExprId(1)],
                ty: CgTy::F32,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let ty = type_check(&arena[2], CgExprId(2), &ctx).expect("well-typed");
        assert_eq!(ty, CgTy::F32);
    }

    #[test]
    fn type_check_select_picks_arm_type() {
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::Bool(true)),
            CgExpr::Lit(LitValue::U32(1)),
            CgExpr::Lit(LitValue::U32(0)),
            CgExpr::Select {
                cond: CgExprId(0),
                then: CgExprId(1),
                else_: CgExprId(2),
                ty: CgTy::U32,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let ty = type_check(&arena[3], CgExprId(3), &ctx).expect("well-typed");
        assert_eq!(ty, CgTy::U32);
    }

    #[test]
    fn type_check_unary_not_on_bool_is_bool() {
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::Bool(false)),
            CgExpr::Unary {
                op: UnaryOp::NotBool,
                arg: CgExprId(0),
                ty: CgTy::Bool,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let ty = type_check(&arena[1], CgExprId(1), &ctx).expect("well-typed");
        assert_eq!(ty, CgTy::Bool);
    }

    #[test]
    fn type_check_view_call_with_resolver() {
        // ViewCall { view: 5 } — resolver says
        // (AgentId, AgentId) -> F32.
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::AgentId(0)),
            CgExpr::Lit(LitValue::AgentId(1)),
            CgExpr::Builtin {
                fn_id: BuiltinId::ViewCall { view: ViewId(5) },
                args: vec![CgExprId(0), CgExprId(1)],
                ty: CgTy::F32,
            },
        ];
        let resolver = |v: ViewId| -> Option<(Vec<CgTy>, CgTy)> {
            if v == ViewId(5) {
                Some((vec![CgTy::AgentId, CgTy::AgentId], CgTy::F32))
            } else {
                None
            }
        };
        let ctx = TypeCheckCtx::with_view_signature(&arena, &resolver);
        let ty = type_check(&arena[2], CgExprId(2), &ctx).expect("well-typed");
        assert_eq!(ty, CgTy::F32);
    }

    // ---- Type-check rejection ----

    #[test]
    fn type_check_rejects_bool_plus_f32() {
        // `true + 1.0` — `AddF32` requires both operands F32; lhs is
        // Bool. The mismatch surfaces on operand index 0.
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::Bool(true)),
            CgExpr::Lit(LitValue::F32(1.0)),
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let err = type_check(&arena[2], CgExprId(2), &ctx).expect_err("ill-typed");
        assert_eq!(
            err,
            TypeError::OperandMismatch {
                node: CgExprId(2),
                operand_index: 0,
                expected: CgTy::F32,
                got: CgTy::Bool,
            }
        );
    }

    #[test]
    fn type_check_rejects_claimed_result_mismatch() {
        // F32 + F32 -> Bool (claimed). The op's `result_ty` is F32;
        // claiming Bool is a `ClaimedResultMismatch`.
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::F32(1.0)),
            CgExpr::Lit(LitValue::F32(2.0)),
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::Bool,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let err = type_check(&arena[2], CgExprId(2), &ctx).expect_err("ill-typed");
        assert_eq!(
            err,
            TypeError::ClaimedResultMismatch {
                node: CgExprId(2),
                expected: CgTy::F32,
                got: CgTy::Bool,
            }
        );
    }

    #[test]
    fn type_check_rejects_arity_mismatch() {
        // distance() with 1 arg — expected 2.
        let arena: Vec<CgExpr> = vec![
            read_self_pos(),
            CgExpr::Builtin {
                fn_id: BuiltinId::Distance,
                args: vec![CgExprId(0)],
                ty: CgTy::F32,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let err = type_check(&arena[1], CgExprId(1), &ctx).expect_err("ill-typed");
        assert_eq!(
            err,
            TypeError::ArityMismatch {
                node: CgExprId(1),
                builtin: BuiltinId::Distance,
                expected: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn type_check_rejects_select_cond_not_bool() {
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::U32(1)),
            CgExpr::Lit(LitValue::F32(1.0)),
            CgExpr::Lit(LitValue::F32(2.0)),
            CgExpr::Select {
                cond: CgExprId(0),
                then: CgExprId(1),
                else_: CgExprId(2),
                ty: CgTy::F32,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let err = type_check(&arena[3], CgExprId(3), &ctx).expect_err("ill-typed");
        assert_eq!(
            err,
            TypeError::SelectCondNotBool {
                node: CgExprId(3),
                got: CgTy::U32,
            }
        );
    }

    #[test]
    fn type_check_rejects_select_arms_mismatch() {
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::Bool(true)),
            CgExpr::Lit(LitValue::F32(1.0)),
            CgExpr::Lit(LitValue::U32(0)),
            CgExpr::Select {
                cond: CgExprId(0),
                then: CgExprId(1),
                else_: CgExprId(2),
                ty: CgTy::F32,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let err = type_check(&arena[3], CgExprId(3), &ctx).expect_err("ill-typed");
        assert_eq!(
            err,
            TypeError::SelectArmsMismatch {
                node: CgExprId(3),
                then_ty: CgTy::F32,
                else_ty: CgTy::U32,
            }
        );
    }

    #[test]
    fn type_check_rejects_view_call_without_resolver() {
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::AgentId(0)),
            CgExpr::Lit(LitValue::AgentId(1)),
            CgExpr::Builtin {
                fn_id: BuiltinId::ViewCall { view: ViewId(9) },
                args: vec![CgExprId(0), CgExprId(1)],
                ty: CgTy::F32,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let err = type_check(&arena[2], CgExprId(2), &ctx).expect_err("unresolved view");
        assert_eq!(
            err,
            TypeError::ViewSignatureUnresolved {
                node: CgExprId(2),
                view: ViewId(9),
            }
        );
    }

    #[test]
    fn type_check_rejects_unary_operand_mismatch() {
        // not.bool applied to F32 — operand mismatch on index 0.
        let arena: Vec<CgExpr> = vec![
            CgExpr::Lit(LitValue::F32(1.0)),
            CgExpr::Unary {
                op: UnaryOp::NotBool,
                arg: CgExprId(0),
                ty: CgTy::Bool,
            },
        ];
        let ctx = TypeCheckCtx::new(&arena);
        let err = type_check(&arena[1], CgExprId(1), &ctx).expect_err("ill-typed");
        assert_eq!(
            err,
            TypeError::OperandMismatch {
                node: CgExprId(1),
                operand_index: 0,
                expected: CgTy::Bool,
                got: CgTy::F32,
            }
        );
    }

    #[test]
    fn type_check_rejects_rng_claimed_non_u32() {
        let arena: Vec<CgExpr> = vec![CgExpr::Rng {
            purpose: RngPurpose::Action,
            ty: CgTy::F32,
        }];
        let ctx = TypeCheckCtx::new(&arena);
        let err = type_check(&arena[0], CgExprId(0), &ctx).expect_err("ill-typed");
        assert_eq!(
            err,
            TypeError::ClaimedResultMismatch {
                node: CgExprId(0),
                expected: CgTy::U32,
                got: CgTy::F32,
            }
        );
    }

    #[test]
    fn type_check_rng_purpose_natural_ty_pairing() {
        // Each `RngPurpose` carries a per-purpose natural result type
        // (`purpose.result_ty()`); the type checker's
        // `ClaimedResultMismatch` arm enforces that the claimed `ty`
        // on `CgExpr::Rng { purpose, ty }` matches it. Internal
        // purposes return U32; spec-named purposes return the type
        // their surface method advertises.
        let cases = [
            (RngPurpose::Action, CgTy::U32),
            (RngPurpose::Sample, CgTy::U32),
            (RngPurpose::Shuffle, CgTy::U32),
            (RngPurpose::Conception, CgTy::U32),
            (RngPurpose::Uniform, CgTy::F32),
            (RngPurpose::Gauss, CgTy::F32),
            (RngPurpose::Coin, CgTy::Bool),
            (RngPurpose::UniformInt, CgTy::I32),
        ];
        for (purpose, expected_ty) in cases {
            // Positive: claimed ty matches purpose.result_ty().
            let arena: Vec<CgExpr> = vec![CgExpr::Rng {
                purpose,
                ty: expected_ty,
            }];
            let ctx = TypeCheckCtx::new(&arena);
            let ok = type_check(&arena[0], CgExprId(0), &ctx)
                .unwrap_or_else(|e| panic!("expected ok for {purpose:?}: {e:?}"));
            assert_eq!(ok, expected_ty);
            // Negative: claimed ty mismatches purpose.result_ty().
            let wrong_ty = if expected_ty == CgTy::U32 {
                CgTy::F32
            } else {
                CgTy::U32
            };
            let arena: Vec<CgExpr> = vec![CgExpr::Rng {
                purpose,
                ty: wrong_ty,
            }];
            let ctx = TypeCheckCtx::new(&arena);
            let err = type_check(&arena[0], CgExprId(0), &ctx)
                .expect_err("type-check should reject mismatched claimed ty");
            assert!(matches!(
                err,
                TypeError::ClaimedResultMismatch { expected, got, .. }
                    if expected == expected_ty && got == wrong_ty
            ));
        }
    }

    // ---- Task 5.5d: AgentSelfId / PerPairCandidateId / ReadLocal ----

    #[test]
    fn agent_self_id_ty_is_agent_id() {
        let e = CgExpr::AgentSelfId;
        assert_eq!(e.ty(), CgTy::AgentId);
        let arena: Vec<CgExpr> = vec![e.clone()];
        assert_eq!(pretty(&arena[0], &arena), "(agent self_id)");
        // Serde round-trip.
        let json = serde_json::to_string(&e).unwrap();
        let back: CgExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn per_pair_candidate_id_ty_is_agent_id() {
        let e = CgExpr::PerPairCandidateId;
        assert_eq!(e.ty(), CgTy::AgentId);
        let arena: Vec<CgExpr> = vec![e.clone()];
        assert_eq!(pretty(&arena[0], &arena), "(agent per_pair_candidate_id)");
        let json = serde_json::to_string(&e).unwrap();
        let back: CgExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn type_check_passes_for_self_id_and_pair_candidate_id() {
        let arena: Vec<CgExpr> = vec![CgExpr::AgentSelfId, CgExpr::PerPairCandidateId];
        let ctx = TypeCheckCtx::new(&arena);
        assert_eq!(
            type_check(&arena[0], CgExprId(0), &ctx).unwrap(),
            CgTy::AgentId
        );
        assert_eq!(
            type_check(&arena[1], CgExprId(1), &ctx).unwrap(),
            CgTy::AgentId
        );
    }

    #[test]
    fn read_local_ty_pinned_from_field() {
        use crate::cg::stmt::LocalId;
        let e = CgExpr::ReadLocal {
            local: LocalId(3),
            ty: CgTy::F32,
        };
        assert_eq!(e.ty(), CgTy::F32);
        let arena: Vec<CgExpr> = vec![e.clone()];
        assert_eq!(pretty(&arena[0], &arena), "(read_local local#3 f32)");
        let json = serde_json::to_string(&e).unwrap();
        let back: CgExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn type_check_passes_for_read_local() {
        use crate::cg::stmt::LocalId;
        let arena: Vec<CgExpr> = vec![CgExpr::ReadLocal {
            local: LocalId(3),
            ty: CgTy::F32,
        }];
        let ctx = TypeCheckCtx::new(&arena);
        assert_eq!(
            type_check(&arena[0], CgExprId(0), &ctx).unwrap(),
            CgTy::F32
        );
    }
}
