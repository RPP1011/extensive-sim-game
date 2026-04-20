//! AST for the World Sim DSL. All nodes carry byte-spans into the source.
//!
//! Lowering lives in a later milestone; this AST is deliberately verbose and
//! one-variant-per-shape.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
    pub fn dummy() -> Self {
        Span { start: 0, end: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Spanned { node, span }
    }
}

// ---------------------------------------------------------------------------
// Program / top-level declarations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    pub decls: Vec<Decl>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Decl {
    Entity(EntityDecl),
    Event(EventDecl),
    View(ViewDecl),
    Query(QueryDecl),
    Physics(PhysicsDecl),
    Mask(MaskDecl),
    Verb(VerbDecl),
    Scoring(ScoringDecl),
    Invariant(InvariantDecl),
    Probe(ProbeDecl),
    Metric(MetricBlock),
    Config(ConfigDecl),
}

// ---------------------------------------------------------------------------
// Annotations (shared)
// ---------------------------------------------------------------------------

/// A generic annotation like `@materialized(on_event=[X, Y], storage=pair_map)`.
/// All semantic interpretation is deferred to lowering.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Annotation {
    pub name: String,
    pub args: Vec<AnnotationArg>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnnotationArg {
    /// `Some("on_event")` for `on_event = [X, Y]`; `None` for a bare positional arg.
    pub key: Option<String>,
    pub value: AnnotationValue,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AnnotationValue {
    Ident(String),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<AnnotationValue>),
    /// `>= Medium`, `< 0.5`, etc — a comparator followed by a value.
    Comparator { op: String, value: Box<AnnotationValue> },
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A type expression. Covers primitives, generic bounded collections, tuples,
/// arrays, user-defined type names.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TypeRef {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum TypeKind {
    /// `AgentId`, `f32`, `MyStruct`.
    Named(String),
    /// `SortedVec<AgentId, 4>`, `Map<K, V, Cap>`, `Bitset<32>`.
    Generic { name: String, args: Vec<TypeArg> },
    /// `[Agent]` or `[Agent, ...]`.
    List(Box<TypeRef>),
    /// `(A, B)`.
    Tuple(Vec<TypeRef>),
    /// `Option<T>`.
    Option(Box<TypeRef>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum TypeArg {
    Type(TypeRef),
    Const(i64),
}

// ---------------------------------------------------------------------------
// 2.1 entity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub root: EntityRoot,
    pub fields: Vec<EntityField>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum EntityRoot {
    Agent,
    Item,
    Group,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityField {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub value: EntityFieldValue,
    pub span: Span,
}

/// Right-hand side of an entity field. Often a bare type (`CreatureType`) or
/// a nested struct literal (`{ channels: ..., can_fly: true }`) or a list
/// literal.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum EntityFieldValue {
    /// `creature_type: CreatureType` — just a type name.
    Type(TypeRef),
    /// `capabilities: Capabilities { ... }` — a type name followed by a struct body.
    StructLiteral { ty: TypeRef, fields: Vec<EntityField> },
    /// A list literal of values (expressions).
    List(Vec<Expr>),
    /// An expression (used for `eligibility_predicate: <predicate>`).
    Expr(Expr),
}

// ---------------------------------------------------------------------------
// 2.2 event
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub fields: Vec<FieldDecl>,
    pub span: Span,
}

/// `<name>: <type>` as used in event / struct-literal contexts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldDecl {
    pub name: String,
    pub ty: TypeRef,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// 2.3 view / query
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ViewDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: TypeRef,
    pub body: ViewBody,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Param {
    pub name: String,
    pub ty: TypeRef,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ViewBody {
    /// `@lazy` view: `{ <expression> }`.
    Expr(Expr),
    /// `@materialized` event-fold body.
    Fold {
        initial: Expr,
        handlers: Vec<FoldHandler>,
        clamp: Option<(Expr, Expr)>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FoldHandler {
    pub pattern: EventPattern,
    pub body: Vec<Stmt>,
    pub span: Span,
}

/// `query <name>(...) -> <type> sort_by <expr> limit <k> { <body> }`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: TypeRef,
    pub sort_by: Option<Expr>,
    pub limit: Option<Expr>,
    pub body: Option<Expr>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// 2.4 physics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicsDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub handlers: Vec<PhysicsHandler>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicsHandler {
    pub pattern: EventPattern,
    pub where_clause: Option<Expr>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

/// `<EventName>{f1: bind1, f2: bind2, ...}`, or the bare name.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventPattern {
    pub name: String,
    pub bindings: Vec<PatternBinding>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PatternBinding {
    pub field: String,
    /// The capture pattern: e.g. `a` (ident), `Agent(a)` (ctor-wrap), or a
    /// literal expression to match against.
    pub value: PatternValue,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PatternValue {
    /// `field: bind_name`.
    Bind(String),
    /// `field: Agent(inner_bind)` or `field: Some(x)`.
    Ctor { name: String, inner: Vec<PatternValue> },
    /// `field: <literal>` or `field: <expr>` to match against.
    Expr(Expr),
    /// `field: _`.
    Wildcard,
}

// ---------------------------------------------------------------------------
// 2.5 mask
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MaskDecl {
    pub annotations: Vec<Annotation>,
    pub head: ActionHead,
    pub predicate: Expr,
    pub span: Span,
}

/// `Attack(t)` or `PostQuest{type: Conquest, party: Group(g)}`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActionHead {
    pub name: String,
    pub shape: ActionHeadShape,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ActionHeadShape {
    /// `Attack(t)` — positional params, each an identifier binding.
    Positional(Vec<String>),
    /// `PostQuest{type: Conquest, party: Group(g)}` — named param patterns.
    Named(Vec<PatternBinding>),
    /// `Eat` — no params.
    None,
}

// ---------------------------------------------------------------------------
// 2.6 verb
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerbDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub params: Vec<Param>,
    pub action: VerbAction,
    pub when: Option<Expr>,
    pub emits: Vec<EmitStmt>,
    pub scoring: Option<Expr>,
    pub span: Span,
}

/// `action Converse(target: shrine.patron_agent_id)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerbAction {
    pub name: String,
    pub args: Vec<CallArg>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// 3.4 scoring
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoringDecl {
    pub annotations: Vec<Annotation>,
    pub entries: Vec<ScoringEntry>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoringEntry {
    pub head: ActionHead,
    pub expr: Expr,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// 2.8 invariant
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InvariantDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    /// Zero or more scope parameters: `(a: Agent)`, `(q: Quest)`, or empty.
    pub scope: Vec<Param>,
    pub mode: InvariantMode,
    pub predicate: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum InvariantMode {
    Static,
    Runtime,
    DebugOnly,
}

// ---------------------------------------------------------------------------
// 2.9 probe
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProbeDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub scenario: Option<String>,
    pub seed: Option<u64>,
    pub seeds: Option<Vec<u64>>,
    pub ticks: Option<u32>,
    pub tolerance: Option<f64>,
    pub asserts: Vec<AssertExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AssertExpr {
    /// `count[<filter>] <op> <scalar>`
    Count { filter: Expr, op: String, value: Expr, span: Span },
    /// `pr[<action_filter> | <obs_filter>] <op> <prob>`
    Pr { action_filter: Expr, obs_filter: Expr, op: String, value: Expr, span: Span },
    /// `mean[<scalar_expr> | <filter>] <op> <scalar>`
    Mean { scalar: Expr, filter: Expr, op: String, value: Expr, span: Span },
}

// ---------------------------------------------------------------------------
// 2.11 metric
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricBlock {
    pub annotations: Vec<Annotation>,
    pub metrics: Vec<MetricDecl>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// 2.12 config (tunable balance constants)
// ---------------------------------------------------------------------------

/// `config <Name> { <field>: <type> = <default>, ... }` — a named block of
/// scalar tunables whose default values are baked into an emitted Rust struct
/// and written out as `assets/config/default.toml` for runtime tuning.
/// Block names must be unique per compilation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConfigDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub fields: Vec<ConfigField>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConfigField {
    pub name: String,
    pub ty: TypeRef,
    pub default: ConfigDefault,
    pub span: Span,
}

/// Parsed default literal for a `config` field. The type tag is informational
/// — lowering pairs this with the field's declared `ty` to pick a canonical
/// emission form. String defaults carry the already-unescaped literal body.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ConfigDefault {
    Int(i64),
    Uint(u64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricDecl {
    pub name: String,
    pub value: Expr,
    pub window: Option<u64>,
    pub emit_every: Option<u64>,
    pub conditioned_on: Option<Expr>,
    pub alert_when: Option<Expr>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Statements (physics / fold handler bodies)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Stmt {
    Let { name: String, value: Expr, span: Span },
    Emit(EmitStmt),
    /// `for x in <iter> { <body> }` or `for x in <iter> where <filter> { <body> }`.
    For { binder: String, iter: Expr, filter: Option<Expr>, body: Vec<Stmt>, span: Span },
    /// `if <cond> { <body> } else { <body> }` / `match <scrut> { ... }`.
    If { cond: Expr, then_body: Vec<Stmt>, else_body: Option<Vec<Stmt>>, span: Span },
    Match { scrutinee: Expr, arms: Vec<MatchArm>, span: Span },
    /// Self-delta in fold bodies: `self -= 0.1 * e.damage`, `self += 0.3`.
    SelfUpdate { op: String, value: Expr, span: Span },
    /// Bare expression (for fold bodies that set self).
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmitStmt {
    pub event_name: String,
    pub fields: Vec<FieldInit>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldInit {
    pub name: String,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MatchArm {
    pub pattern: PatternValue,
    pub body: Vec<Stmt>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    /// Bare identifier reference.
    Ident(String),
    /// `a.b`.
    Field(Box<Expr>, String),
    /// `a[b]`.
    Index(Box<Expr>, Box<Expr>),
    /// `f(x, y)` or `view::call(x)`.
    Call(Box<Expr>, Vec<CallArg>),
    /// Infix binary operator.
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    /// Prefix unary operator.
    Unary { op: UnOp, rhs: Box<Expr> },
    /// `x in set`.
    In { item: Box<Expr>, set: Box<Expr> },
    /// `set contains x`.
    Contains { set: Box<Expr>, item: Box<Expr> },
    /// `forall x in set: <body>` / `exists x in set: <body>`.
    Quantifier { kind: QuantKind, binder: String, iter: Box<Expr>, body: Box<Expr> },
    /// `count(x in set where <body>)` / `count[<filter>]` / `sum(...)`, etc.
    Fold { kind: FoldKind, binder: Option<String>, iter: Option<Box<Expr>>, body: Box<Expr> },
    /// `{ ... }` list / set literal.
    List(Vec<Expr>),
    /// `(a, b, c)` tuple literal.
    Tuple(Vec<Expr>),
    /// `EventName { a: 1, b: 2 }` struct-ish literal (used inline in expressions).
    Struct { name: String, fields: Vec<FieldInit> },
    /// `Agent(x)` / `Some(x)` / `Group(g)` constructor-style call.
    Ctor { name: String, args: Vec<Expr> },
    /// `match <scrut> { <arm> => <body>, ... }` used as an expression.
    Match { scrutinee: Box<Expr>, arms: Vec<MatchExprArm> },
    /// `if <c> { e1 } else { e2 }` used as an expression.
    If { cond: Box<Expr>, then_expr: Box<Expr>, else_expr: Option<Box<Expr>> },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MatchExprArm {
    pub pattern: PatternValue,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CallArg {
    /// `Some("target")` for `target: x`, `None` for positional.
    pub name: Option<String>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinOp {
    And,
    Or,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UnOp {
    Not,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QuantKind {
    Forall,
    Exists,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FoldKind {
    Count,
    Sum,
    Max,
    Min,
}
