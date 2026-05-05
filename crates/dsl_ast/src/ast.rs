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
    EventTag(EventTagDecl),
    Enum(EnumDecl),
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
    SpatialQuery(SpatialQueryDecl),
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
    /// `per_entity_topk(K = 8)` — an identifier followed by a parenthesised
    /// argument list. Used by storage hints that carry tuning knobs
    /// (task 196 added the `K = N` parameter to `per_entity_topk`). The
    /// inner args re-use the same `AnnotationArg` shape as top-level
    /// annotations — each arg is `key = value` or a bare positional.
    Call { name: String, args: Vec<AnnotationArg> },
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
    /// Quest-rooted entity. Spec table at `docs/spec/dsl.md:653-663`
    /// lists `Quest` alongside `Agent`/`Item`/`Group`. Today the
    /// declaration parses + lowers as a declare-only entity (no
    /// per-Quest SoA storage, no `quests.field(idx)` accessor) — the
    /// `populate_entity_field_catalog` driver skips Quest entries the
    /// same way it skips Agent ones. Closes Gap A from
    /// `docs/superpowers/notes/2026-05-04-quest_probe.md`. Future
    /// extension: add `EntityFieldCatalog::quests` when a fixture
    /// surfaces a `quests.<field>(idx)` call site.
    Quest,
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
    /// Named tags attached to this event via `@tag_name` annotations. Stored
    /// as lowercased tag-annotation names (the matching `event_tag
    /// <PascalName>` declaration has its name lowercased for lookup).
    pub tags: Vec<Spanned<String>>,
    pub span: Span,
}

/// `event_tag <Name> { <field>: <type>, ... }` — a compile-time contract
/// declaring a set of required fields an event claims via `@<name>`
/// annotation. No runtime type is emitted; tags are enforced at emit time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventTagDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub fields: Vec<FieldDecl>,
    pub span: Span,
}

/// `enum <Name> { <Variant>, ... }` — a named list of variants emitted as a
/// `#[repr(u8)]` Rust enum and a Python `IntEnum`. Variants are assigned
/// sequential ordinals starting at 0 in source order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnumDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnumVariant {
    pub name: String,
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

/// `spatial_query <name>(self, candidate, <typed-args>) = <filter_expr>`.
///
/// Declares a named per-candidate filter for spatial walks. The first
/// two positional binders MUST be `self` (the querying agent) and
/// `candidate` (the per-pair neighbour under inspection); the
/// resolver enforces the convention. Remaining params are typed
/// value args substituted at the call site (e.g.
/// `from spatial.nearby_in_radius(self, config.movement.max_move_radius)`).
///
/// Note the call-site arity convention (Phase 7 Task 4 Adjustment A):
/// the from-clause passes `(self, value_args...)` only — `candidate`
/// is implicit and binds positionally to the per-iteration spatial-walk
/// neighbour at lowering time. Mask action-head binders such as
/// `target` cannot be passed as call-site arguments because they are
/// bound AFTER the from-clause is resolved.
///
/// The filter is a single expression (Bool — well_formed gate from
/// Phase 7 Task 3 enforces the type once lowered to CG). No `{}`
/// block; mirrors the verb `name(...) = action ...` shape.
///
/// Lowering produces a `CgExprId` filter for the IR's
/// `SpatialQueryKind::FilteredWalk`. See
/// `docs/superpowers/plans/2026-05-01-phase-7-general-spatial-queries.md`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpatialQueryDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub params: Vec<Param>,
    pub filter: Expr,
    pub span: Span,
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
    /// Intentionally-CPU-only rule. Set when the source carries the
    /// `@cpu_only` annotation. The compiler emits the CPU handler but
    /// skips WGSL emission and the GPU event-kind dispatcher entry.
    /// Bypasses the GPU-emittable validator so string-formatting / heap
    /// allocation / other non-WGSL primitives in the body don't fail the
    /// build.
    pub cpu_only: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicsHandler {
    pub pattern: PhysicsPattern,
    pub where_clause: Option<Expr>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

/// Physics `on` pattern — either a concrete event kind (`on Foo { ... }`) or
/// a tag (`on @harmful { ... }`). Tag-matched handlers run against every
/// event that declares the tag via `@tag_name`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PhysicsPattern {
    Kind(EventPattern),
    Tag {
        /// Lowercased tag name (matches `event_tag` decl's lowercased name).
        name: String,
        bindings: Vec<PatternBinding>,
        span: Span,
    },
}

impl PhysicsPattern {
    pub fn span(&self) -> Span {
        match self {
            PhysicsPattern::Kind(p) => p.span,
            PhysicsPattern::Tag { span, .. } => *span,
        }
    }
    pub fn bindings(&self) -> &[PatternBinding] {
        match self {
            PhysicsPattern::Kind(p) => &p.bindings,
            PhysicsPattern::Tag { bindings, .. } => bindings,
        }
    }
    pub fn display_name(&self) -> &str {
        match self {
            PhysicsPattern::Kind(p) => &p.name,
            PhysicsPattern::Tag { name, .. } => name,
        }
    }
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
    /// `Damage { amount }` or `Slow { duration_ticks, factor_q8: f }` —
    /// struct-shaped variant pattern. Each binding names a variant field and
    /// either introduces a shorthand bind with the same name (`amount`) or an
    /// aliased nested pattern (`factor_q8: f`). Used to destructure enum
    /// variants carrying named fields; the emitter lowers this to Rust's
    /// `Name { field, field: inner }` pattern syntax.
    Struct { name: String, bindings: Vec<PatternBinding> },
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
    /// Optional `from <expression>` clause — the candidate source for
    /// target-bound masks. When present, the emitted mask fn enumerates
    /// candidates from this expression (typically a `query.nearby_agents`
    /// call) and filters each through the `when` predicate. Task 138 —
    /// retire `nearest_other` in favour of scoring-argmax over masked
    /// candidates.
    pub candidate_source: Option<Expr>,
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
    /// `Attack(t)` — positional params. Each entry is `(name,
    /// optional type annotation)`. Untyped params (`Attack(t)`) preserve
    /// the implicit-`AgentId` contract every v1 mask head relies on;
    /// typed params (`Cast(ability: AbilityId)`) let a mask head carry
    /// non-agent IDs (cast targets an ability slot). Task 157.
    Positional(Vec<(String, Option<TypeRef>)>),
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
    /// Standard per-agent rows: `Head = expression`. Each entry scores
    /// one (agent, action) pair.
    pub entries: Vec<ScoringEntry>,
    /// `row <name> per_ability { guard: ..., score: ..., target: ... }`
    /// rows. The scoring kernel iterates each agent's ability slots and
    /// produces one score per (agent, ability) pair. Added 2026-04-23
    /// (GPU ability evaluation Phase 2). Kept as a sibling list rather
    /// than folded into `entries` so legacy emitters that walk `entries`
    /// stay untouched — Phase 3 wires a dedicated CPU lowering for the
    /// `PerAbilityRow` shape.
    pub per_ability_rows: Vec<PerAbilityRow>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoringEntry {
    pub head: ActionHead,
    pub expr: Expr,
    pub span: Span,
}

/// A `per_ability` scoring row: `row <name> per_ability { ... }`.
///
/// The row's three clauses:
/// * `guard:` — boolean predicate evaluated per (agent, ability). When
///   the guard is false the ability is skipped (does not compete for
///   argmax). Optional; default is `true`.
/// * `score:` — f32 scoring expression. The argmax over every ability
///   whose guard passes is the ability the agent casts this tick.
/// * `target:` — agent-id expression resolving to the cast target for
///   the selected ability. Optional at parse time; Phase 3 may require
///   it when lowering.
///
/// See `docs/spec/engine.md §11`
/// §Architecture.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerAbilityRow {
    pub name: String,
    pub guard: Option<Expr>,
    pub score: Expr,
    pub target: Option<Expr>,
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
    /// `beliefs(observer).observe(target) with { field: expr, ... }` — belief
    /// mutation primitive (Plan ToM Task 4). Mutates a single `BeliefState`
    /// cell in `SimState::cold_beliefs` for the observer/target pair.
    BeliefObserve(BeliefObserveStmt),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BeliefObserveStmt {
    pub observer: String,
    pub target: String,
    pub fields: Vec<FieldInit>,
    pub span: Span,
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
    /// Gradient modifier: `<expr> per_unit <delta>`. Usable as a top-level
    /// term inside a `scoring` entry's sum; the scoring emitter recognises
    /// it as a gradient modifier row rather than a plain multiplication.
    /// Semantically identical to `expr * delta` if the scoring lowering
    /// didn't promote it to a dedicated modifier kind. See spec §3.4.
    PerUnit { expr: Box<Expr>, delta: Box<Expr> },
    /// `beliefs(observer).about(target).<field>` — read a single field from
    /// the belief cell for an observer/target pair. `field` must be one of the
    /// `BELIEF_FIELDS` allowlist (validated in the resolver, Plan ToM Task 8).
    BeliefsAccessor { observer: Box<Expr>, target: Box<Expr>, field: String },
    /// `beliefs(observer).confidence(target)` — read the `confidence` field
    /// from the belief cell. Syntactic sugar for
    /// `beliefs(o).about(t).confidence`.
    BeliefsConfidence { observer: Box<Expr>, target: Box<Expr> },
    /// `beliefs(observer).<view_name>(_)` — aggregate view over the set of
    /// targets the observer currently believes in (Plan ToM Task 8).
    BeliefsView { observer: Box<Expr>, view_name: String },
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

// ---------------------------------------------------------------------------
// `.ability` file AST (Wave 1.0 subset of `docs/spec/ability_dsl_unified.md`)
//
// Parses the surface from spec §4 (`ability` blocks) — header properties
// (target / range / cooldown / cast / hint) plus zero-or-more bare effect
// statements. Modifier slots (in / for / when / chance / [TAGS] / scaling /
// nested), `passive` / `template` / `structure` blocks, and `deliver`
// blocks are deliberately deferred to later slices (Waves 1.1-1.5). When
// the parser sees a modifier token mid-effect it records the simple
// positional arguments collected so far and skips the rest of the line.
// See `crates/dsl_ast/src/ability_parser.rs` for the parser.
// ---------------------------------------------------------------------------

/// A single `.ability` source file. Currently holds only `ability` decls;
/// future slices add `passive`, `template`, and `structure`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AbilityFile {
    pub abilities: Vec<AbilityDecl>,
    // Future (Wave 1.1+): passives, templates, structures.
}

/// A parsed `ability <Name> { headers... effects... }` block.
///
/// `headers` is the list of header properties in source order. Duplicate
/// header keys are rejected at parse time. `effects` is the list of bare
/// effect statements in source order. The Wave 1.0 parser does NOT support
/// `deliver` / `recast` / `morph` blocks — those are parse errors today
/// (deferred to Wave 1.4).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AbilityDecl {
    pub name: String,
    pub headers: Vec<AbilityHeader>,
    pub effects: Vec<EffectStmt>,
    pub span: Span,
}

/// One header property line inside an `ability` block. Header keys cap out
/// at the five Wave 1.0 properties (target / range / cooldown / cast /
/// hint). Other spec-listed headers (cost / charges / recharge / toggle /
/// recast / morph / form / require_skill / require_tool / zone_tag /
/// unstoppable) are deferred — encountering one is a parse error today.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AbilityHeader {
    Target(TargetMode),
    Range(f32),
    Cooldown(Duration),
    Cast(Duration),
    Hint(HintName),
}

/// One effect statement: a verb name plus zero-or-more positional args.
///
/// Wave 1.0 captures only the leading positional args (numbers / durations
/// / percents / strings / idents). When the parser encounters a modifier
/// keyword (`in`, `for`, `when`, `chance`, `stacking`, `+`) or a
/// bracketed power-tag list (`[FIRE: 60]`) or a nested-effects block
/// (`{ ... }`), it stops collecting args and skips to the end of the
/// statement. Modifier capture lands in Wave 1.5. `args` therefore
/// reflects only the verb's required scalar arguments — sufficient for
/// the five combat-core verbs (damage / heal / shield / stun / slow)
/// plus the simple control verbs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectStmt {
    pub verb: String,
    pub args: Vec<EffectArg>,
    pub span: Span,
    // Future (Wave 1.5): modifier slots (in / for / when / chance /
    // tags / stacking / scaling / lifetime / nested).
}

/// One positional argument in an effect statement. The Wave 1.0 parser
/// records the literal kind it saw; lowering (Wave 1.6) is responsible
/// for verb-specific type checking (e.g. `damage` wants `Number`,
/// `stun` wants `Duration`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum EffectArg {
    Number(f32),
    Duration(Duration),
    Percent(f32),
    String(String),
    Ident(String),
}

/// Target mode for an ability — sets the dispatch shape (PerAgent /
/// PerPair) per spec §4.3. Variant set matches the eight modes the spec
/// table lists; `Self_` uses a trailing underscore because `self` is a
/// Rust keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TargetMode {
    Enemy,
    Self_,
    Ally,
    SelfAoe,
    Ground,
    Direction,
    Vector,
    Global,
}

/// Hint enum used for scoring metadata only (spec §4.2). Six variants
/// matching the existing `.ability` corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HintName {
    Damage,
    Defense,
    CrowdControl,
    Utility,
    Heal,
    Economic,
}

/// A normalized duration in milliseconds. The lexer accepts `5s`,
/// `300ms`, `1.5s`, and bare `5000` (interpreted as ms per the spec
/// §6 lowering note that durations on the GPU side are tick-quantized
/// from millis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Duration {
    pub millis: u32,
}
