//! Typed IR for the World Sim DSL. Built by `resolve.rs` from a parsed AST.
//!
//! The IR is a flat catalog: one `Vec<*IR>` per declaration kind, with typed
//! references (`*Ref` newtypes) everywhere a cross-declaration name is used.
//! Source-level names are preserved on every IR node for diagnostics and
//! emission debugging.
//!
//! Non-goals at this layer: validation (cycle / race / arity), desugaring,
//! schema hashing, full type inference. See `docs/compiler/spec.md` §3.

use serde::Serialize;

use crate::ast::{self, Annotation, BinOp, QuantKind, Span, UnOp};

// ---------------------------------------------------------------------------
// Typed reference IDs
// ---------------------------------------------------------------------------

macro_rules! ref_newtype {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
        pub struct $name(pub u16);
    };
}

ref_newtype!(EventRef);
ref_newtype!(EntityRef);
ref_newtype!(PhysicsRef);
ref_newtype!(MaskRef);
ref_newtype!(ScoringRef);
ref_newtype!(ViewRef);
ref_newtype!(VerbRef);
ref_newtype!(InvariantRef);
ref_newtype!(ProbeRef);
ref_newtype!(MetricRef);
ref_newtype!(LocalRef);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct FieldRef {
    pub entity: EntityRef,
    pub field_idx: u16,
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IrType {
    // Primitives
    Bool,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Vec3,
    String,
    // Niche IDs
    AgentId,
    ItemId,
    GroupId,
    QuestId,
    AuctionId,
    EventId,
    AbilityId,
    // Collections with element type + capacity
    SortedVec(Box<IrType>, u16),
    RingBuffer(Box<IrType>, u16),
    SmallVec(Box<IrType>, u16),
    Array(Box<IrType>, u16),
    Optional(Box<IrType>),
    Tuple(Vec<IrType>),
    List(Box<IrType>),
    // Resolved references
    EntityRef(EntityRef),
    EventRef(EventRef),
    // Named enum (stdlib or user-declared variant group)
    Enum { name: String, variants: Vec<String> },
    // Unresolved — left to a later pass (1b) to type-check.
    Unknown,
    /// Fallback: type name we could not resolve. Kept as a string so
    /// diagnostics can reference it by the source name.
    Named(String),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrExprNode {
    pub kind: IrExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IrExpr {
    // Literals
    LitBool(bool),
    LitInt(i64),
    LitFloat(f64),
    LitString(String),
    // Name references — resolved
    Local(LocalRef, String),
    Event(EventRef),
    Entity(EntityRef),
    View(ViewRef),
    Verb(VerbRef),
    /// Stdlib namespace / sim-wide accessor: `cascade`, `event`, `agents`,
    /// `mask`, `action`. Meaning is resolved at a later pass.
    Namespace(String),
    // Enum variant (e.g. `Conquest`, `Family`, `true`, `false`).
    EnumVariant { ty: String, variant: String },
    // Field access. When we can resolve, `field` is `Some`. Otherwise we keep
    // the source-level name in `field_name` and defer to 1b.
    Field {
        base: Box<IrExprNode>,
        field_name: String,
        field: Option<FieldRef>,
    },
    Index(Box<IrExprNode>, Box<IrExprNode>),
    // Function calls split by resolved callee kind.
    ViewCall(ViewRef, Vec<IrCallArg>),
    VerbCall(VerbRef, Vec<IrCallArg>),
    BuiltinCall(Builtin, Vec<IrCallArg>),
    /// Call whose callee couldn't be resolved at this pass — kept for 1b.
    UnresolvedCall(String, Vec<IrCallArg>),
    // Operators
    Binary(BinOp, Box<IrExprNode>, Box<IrExprNode>),
    Unary(UnOp, Box<IrExprNode>),
    // Set membership / quantifiers
    In(Box<IrExprNode>, Box<IrExprNode>),
    Contains(Box<IrExprNode>, Box<IrExprNode>),
    Quantifier {
        kind: QuantKind,
        binder: LocalRef,
        binder_name: String,
        iter: Box<IrExprNode>,
        body: Box<IrExprNode>,
    },
    Fold {
        kind: ast::FoldKind,
        binder: Option<LocalRef>,
        binder_name: Option<String>,
        iter: Option<Box<IrExprNode>>,
        body: Box<IrExprNode>,
    },
    List(Vec<IrExprNode>),
    Tuple(Vec<IrExprNode>),
    /// `EventName { a: 1, b: 2 }` — if `ctor` resolves to an event or entity,
    /// it's recorded; otherwise we keep the name only.
    StructLit {
        name: String,
        ctor: Option<CtorRef>,
        fields: Vec<IrFieldInit>,
    },
    /// `Agent(x)` / `Some(x)` — constructor-style call. `ctor` is set when
    /// the name resolves (e.g. to an entity).
    Ctor {
        name: String,
        ctor: Option<CtorRef>,
        args: Vec<IrExprNode>,
    },
    Match {
        scrutinee: Box<IrExprNode>,
        arms: Vec<IrMatchArm>,
    },
    If {
        cond: Box<IrExprNode>,
        then_expr: Box<IrExprNode>,
        else_expr: Option<Box<IrExprNode>>,
    },
    /// Retained original AST shape for anything we can't lower meaningfully.
    Raw(Box<ast::Expr>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CtorRef {
    Event(EventRef),
    Entity(EntityRef),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrCallArg {
    pub name: Option<String>,
    pub value: IrExprNode,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrFieldInit {
    pub name: String,
    pub value: IrExprNode,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrMatchArm {
    pub pattern: IrPattern,
    pub body: IrExprNode,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Patterns (event-pattern bindings, match arms)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IrPattern {
    /// Binds a name into the local scope.
    Bind { name: String, local: LocalRef },
    /// Ctor-style: `Agent(x)`, `Some(y)`. `ctor` is set when resolvable.
    Ctor { name: String, ctor: Option<CtorRef>, inner: Vec<IrPattern> },
    /// Literal / expression pattern.
    Expr(IrExprNode),
    Wildcard,
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IrStmt {
    Let {
        name: String,
        local: LocalRef,
        value: IrExprNode,
        span: Span,
    },
    Emit(IrEmit),
    For {
        binder: LocalRef,
        binder_name: String,
        iter: IrExprNode,
        filter: Option<IrExprNode>,
        body: Vec<IrStmt>,
        span: Span,
    },
    If {
        cond: IrExprNode,
        then_body: Vec<IrStmt>,
        else_body: Option<Vec<IrStmt>>,
        span: Span,
    },
    Match {
        scrutinee: IrExprNode,
        arms: Vec<IrStmtMatchArm>,
        span: Span,
    },
    SelfUpdate {
        op: String,
        value: IrExprNode,
        span: Span,
    },
    Expr(IrExprNode),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrStmtMatchArm {
    pub pattern: IrPattern,
    pub body: Vec<IrStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrEmit {
    pub event_name: String,
    pub event: Option<EventRef>,
    pub fields: Vec<IrFieldInit>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Per-decl IR structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventIR {
    pub name: String,
    pub fields: Vec<EventField>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventField {
    pub name: String,
    pub ty: IrType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityIR {
    pub name: String,
    pub root: ast::EntityRoot,
    pub fields: Vec<EntityFieldIR>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

/// Entity fields are preserved mostly verbatim — the RHS can be a type, a
/// struct literal (nested fields), a list literal, or a bare expression. 1a
/// resolves the types it can; values are resolved as `IrExprNode` where
/// applicable.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityFieldIR {
    pub name: String,
    pub value: EntityFieldValueIR,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum EntityFieldValueIR {
    /// `creature_type: CreatureType` — bare type.
    Type(IrType),
    /// `capabilities: Capabilities { ... }`.
    StructLiteral {
        ty: IrType,
        fields: Vec<EntityFieldIR>,
    },
    /// A list of expressions.
    List(Vec<IrExprNode>),
    /// `eligibility_predicate: <expr>`.
    Expr(IrExprNode),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicsIR {
    pub name: String,
    pub handlers: Vec<PhysicsHandlerIR>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicsHandlerIR {
    pub pattern: IrEventPattern,
    pub where_clause: Option<IrExprNode>,
    pub body: Vec<IrStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrEventPattern {
    pub name: String,
    pub event: Option<EventRef>,
    pub bindings: Vec<IrPatternBinding>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrPatternBinding {
    pub field: String,
    pub value: IrPattern,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MaskIR {
    pub head: IrActionHead,
    pub predicate: IrExprNode,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrActionHead {
    pub name: String,
    pub shape: IrActionHeadShape,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IrActionHeadShape {
    Positional(Vec<(String, LocalRef)>),
    Named(Vec<IrPatternBinding>),
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoringIR {
    pub entries: Vec<ScoringEntryIR>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoringEntryIR {
    pub head: IrActionHead,
    pub expr: IrExprNode,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ViewIR {
    pub name: String,
    pub params: Vec<IrParam>,
    pub return_ty: IrType,
    pub body: ViewBodyIR,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IrParam {
    pub name: String,
    pub local: LocalRef,
    pub ty: IrType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ViewBodyIR {
    Expr(IrExprNode),
    Fold {
        initial: IrExprNode,
        handlers: Vec<FoldHandlerIR>,
        clamp: Option<(IrExprNode, IrExprNode)>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FoldHandlerIR {
    pub pattern: IrEventPattern,
    pub body: Vec<IrStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerbIR {
    pub name: String,
    pub params: Vec<IrParam>,
    pub action: VerbActionIR,
    pub when: Option<IrExprNode>,
    pub emits: Vec<IrEmit>,
    pub scoring: Option<IrExprNode>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerbActionIR {
    pub name: String,
    pub args: Vec<IrCallArg>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InvariantIR {
    pub name: String,
    pub scope: Vec<IrParam>,
    pub mode: ast::InvariantMode,
    pub predicate: IrExprNode,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProbeIR {
    pub name: String,
    pub scenario: Option<String>,
    pub seed: Option<u64>,
    pub seeds: Option<Vec<u64>>,
    pub ticks: Option<u32>,
    pub tolerance: Option<f64>,
    pub asserts: Vec<IrAssertExpr>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IrAssertExpr {
    Count { filter: IrExprNode, op: String, value: IrExprNode, span: Span },
    Pr { action_filter: IrExprNode, obs_filter: IrExprNode, op: String, value: IrExprNode, span: Span },
    Mean { scalar: IrExprNode, filter: IrExprNode, op: String, value: IrExprNode, span: Span },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricIR {
    pub name: String,
    pub value: IrExprNode,
    pub window: Option<u64>,
    pub emit_every: Option<u64>,
    pub conditioned_on: Option<IrExprNode>,
    pub alert_when: Option<IrExprNode>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Builtins
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Builtin {
    Count,
    Sum,
    Min,
    Max,
    Distance,
    PlanarDistance,
    ZSeparation,
    Entity,
    Forall,
    Exists,
}

impl Builtin {
    pub fn name(&self) -> &'static str {
        match self {
            Builtin::Count => "count",
            Builtin::Sum => "sum",
            Builtin::Min => "min",
            Builtin::Max => "max",
            Builtin::Distance => "distance",
            Builtin::PlanarDistance => "planar_distance",
            Builtin::ZSeparation => "z_separation",
            Builtin::Entity => "entity",
            Builtin::Forall => "forall",
            Builtin::Exists => "exists",
        }
    }
}

// ---------------------------------------------------------------------------
// Span table
// ---------------------------------------------------------------------------

/// Maps IR node IDs to source spans. For now we keep spans on IR nodes
/// directly and expose this table as a simple flat list for future use.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SpanTable {
    pub entries: Vec<Span>,
}

// ---------------------------------------------------------------------------
// Compilation unit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Compilation {
    pub events: Vec<EventIR>,
    pub entities: Vec<EntityIR>,
    pub physics: Vec<PhysicsIR>,
    pub masks: Vec<MaskIR>,
    pub scoring: Vec<ScoringIR>,
    pub views: Vec<ViewIR>,
    pub verbs: Vec<VerbIR>,
    pub invariants: Vec<InvariantIR>,
    pub probes: Vec<ProbeIR>,
    pub metrics: Vec<MetricIR>,
    pub spans: SpanTable,
}
