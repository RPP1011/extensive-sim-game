//! Typed IR for the World Sim DSL. Built by `resolve.rs` from a parsed AST.
//!
//! The IR is a flat catalog: one `Vec<*IR>` per declaration kind, with typed
//! references (`*Ref` newtypes) everywhere a cross-declaration name is used.
//! Source-level names are preserved on every IR node for diagnostics and
//! emission debugging.
//!
//! Non-goals at this layer: validation (cycle / race / arity), desugaring,
//! schema hashing, full type inference. See `docs/compiler/spec.md` §3.

use serde::{Deserialize, Serialize};

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
ref_newtype!(EventTagRef);
ref_newtype!(EnumRef);
ref_newtype!(EntityRef);
ref_newtype!(PhysicsRef);
ref_newtype!(MaskRef);
ref_newtype!(ScoringRef);
ref_newtype!(ViewRef);
ref_newtype!(VerbRef);
ref_newtype!(InvariantRef);
ref_newtype!(ProbeRef);
ref_newtype!(MetricRef);
ref_newtype!(ConfigRef);
ref_newtype!(LocalRef);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct FieldRef {
    pub entity: EntityRef,
    pub field_idx: u16,
}

// ---------------------------------------------------------------------------
// Ability-evaluation primitives (GPU ability evaluation Phase 2)
// ---------------------------------------------------------------------------
//
// Mirrors of the engine's `AbilityTag` / `AbilityHint` enums from
// `crates/engine/src/ability/program.rs`. The dsl_compiler crate is
// intentionally independent of engine (it runs in the xtask build, not
// the game binary), so the enums are duplicated here with pinned
// discriminants. Renaming or reordering variants requires a coordinated
// change on both sides (and a schema-hash bump).
//
// Per the spec's "Open questions" §"Tag registry shape": fixed enum for
// v1. Extensibility deferred until real-world tag usage demands it.

/// Per-effect power-rating tag surfaced via `.ability` DSL's
/// `[TAG: value]` syntax. Mirrored from `engine::ability::AbilityTag`.
///
/// Discriminants are pinned to match the engine-side enum so GPU
/// packing (`PackedAbilityRegistry::tag_values`) and WGSL `const`
/// comparisons align without a runtime lookup. Column index into
/// `tag_values` is the `#[repr(u8)]` ordinal.
///
/// Spec: `docs/spec/engine.md §11`
/// §"Open questions" (fixed enum for v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[repr(u8)]
pub enum AbilityTag {
    Physical = 0,
    Magical = 1,
    CrowdControl = 2,
    Heal = 3,
    Defense = 4,
    Utility = 5,
}

impl AbilityTag {
    /// Total variant count. Pinned to match the engine-side
    /// `AbilityTag::COUNT` + the `NUM_ABILITY_TAGS` stride used by
    /// `PackedAbilityRegistry::tag_values`. Bump in lockstep with any
    /// enum addition.
    pub const COUNT: usize = 6;

    /// Parse the tag from its DSL token form (identifier-case:
    /// `PHYSICAL`, `MAGICAL`, `CROWD_CONTROL`, `HEAL`, `DEFENSE`,
    /// `UTILITY`). Returns `None` for an unknown spelling so upstream
    /// can surface the original token in its error.
    pub fn parse_ident(s: &str) -> Option<Self> {
        match s {
            "PHYSICAL" => Some(Self::Physical),
            "MAGICAL" => Some(Self::Magical),
            "CROWD_CONTROL" => Some(Self::CrowdControl),
            "HEAL" => Some(Self::Heal),
            "DEFENSE" => Some(Self::Defense),
            "UTILITY" => Some(Self::Utility),
            _ => None,
        }
    }

    /// Canonical DSL token for this tag (identifier-case).
    pub fn as_ident(self) -> &'static str {
        match self {
            Self::Physical => "PHYSICAL",
            Self::Magical => "MAGICAL",
            Self::CrowdControl => "CROWD_CONTROL",
            Self::Heal => "HEAL",
            Self::Defense => "DEFENSE",
            Self::Utility => "UTILITY",
        }
    }
}

/// Coarse ability-category hint. Mirrored from
/// `engine::ability::AbilityHint`. One hint per ability; rows can
/// compare via `ability::hint == damage` (DSL uses lowercase
/// identifier form; parser flattens `::` into a single identifier).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[repr(u8)]
pub enum AbilityHint {
    Damage = 0,
    Defense = 1,
    CrowdControl = 2,
    Utility = 3,
}

impl AbilityHint {
    /// Parse the hint from its DSL token form (lowercase:
    /// `damage`, `defense`, `crowd_control`, `utility`). Returns
    /// `None` for an unknown spelling.
    pub fn parse_ident(s: &str) -> Option<Self> {
        match s {
            "damage" => Some(Self::Damage),
            "defense" => Some(Self::Defense),
            "crowd_control" => Some(Self::CrowdControl),
            "utility" => Some(Self::Utility),
            _ => None,
        }
    }

    /// Canonical DSL token for this hint (lowercase).
    pub fn as_ident(self) -> &'static str {
        match self {
            Self::Damage => "damage",
            Self::Defense => "defense",
            Self::CrowdControl => "crowd_control",
            Self::Utility => "utility",
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IrType {
    // Primitives
    Bool,
    I8,
    U8,
    I16,
    U16,
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
    /// Stdlib namespace / sim-wide accessor: `world`, `cascade`, `event`,
    /// `mask`, `action`, `rng`, `query`, `voxel`, plus the legacy collection
    /// accessors (`agents`, `items`, `groups`, `quests`, `auctions`, `tick`).
    /// Fields and methods hanging off a typed namespace resolve to
    /// `NamespaceField` / `NamespaceCall`; legacy collections stay loose.
    Namespace(NamespaceId),
    /// `world.tick`, `cascade.iterations`, etc. Resolved with a stdlib-typed
    /// field signature.
    NamespaceField {
        ns: NamespaceId,
        field: String,
        ty: IrType,
    },
    /// `rng.uniform(0.0, 1.0)`, `query.nearby_agents(pos, 20.0)`, etc.
    /// Resolved against a stdlib-declared method signature.
    NamespaceCall {
        ns: NamespaceId,
        method: String,
        args: Vec<IrCallArg>,
    },
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
    /// Gradient modifier: `<expr> per_unit <delta>`. See `ast::ExprKind::PerUnit`.
    /// Lowered as a distinct modifier row by the scoring emitter; outside
    /// scoring contexts it is semantically `expr * delta`.
    PerUnit {
        expr: Box<IrExprNode>,
        delta: Box<IrExprNode>,
    },
    /// `ability::tag(TAG_NAME)` — reads the named tag's power rating
    /// off the currently-scored ability. Returns f32; 0.0 if the
    /// ability has no entry for the tag. Only meaningful inside a
    /// `per_ability` scoring row (where "the currently-scored
    /// ability" has a binding) — Phase 3 enforces this at emit time.
    ///
    /// Added 2026-04-23 (GPU ability evaluation Phase 2).
    AbilityTag { tag: AbilityTag },
    /// `ability::hint` — reads the coarse hint category of the
    /// currently-scored ability. Compared for equality against a
    /// hint literal (see `AbilityHintLit`).
    ///
    /// Phase 3 lowers this to an `Option<AbilityHint>` read off the
    /// packed registry — the sentinel case (no hint) compares as
    /// "not a match" against every hint literal.
    AbilityHint,
    /// Literal ability-hint value, produced on the RHS of an
    /// `ability::hint == <ident>` comparison. The DSL spelling uses
    /// lowercase identifiers (`damage`, `defense`, `crowd_control`,
    /// `utility`) so this is context-sensitive: the resolver only
    /// promotes a bare lowercase ident to `AbilityHintLit` when the
    /// opposite side of the `==` is `AbilityHint`.
    AbilityHintLit(AbilityHint),
    /// `ability::range` — reads the currently-scored ability's
    /// `Area::SingleTarget { range }` as f32. Naked accessor (no
    /// `()` suffix); only meaningful inside a `per_ability` row.
    AbilityRange,
    /// `ability::on_cooldown(<slot_expr>)` — returns `true` when the
    /// given ability slot is still on cooldown for the scoring
    /// agent. Inside a `per_ability` row the slot expression is
    /// typically the implicit `ability` local (the row's iterator).
    ///
    /// Kept as a distinct variant (rather than a generic namespace
    /// call) so Phase 3's CPU / Phase 4's GPU emitters can dispatch
    /// directly onto the per-(agent, slot) cooldown buffer from the
    /// ability-cooldowns micro-subsystem without re-shaping through
    /// the `NamespaceCall` path.
    AbilityOnCooldown(Box<IrExprNode>),
    /// Retained original AST shape for anything we can't lower meaningfully.
    Raw(Box<ast::Expr>),
    /// `beliefs(observer).about(target).<field>` — read a single field from
    /// the belief cell for an observer/target pair. `field` is validated
    /// against the `BELIEF_FIELDS` allowlist in the resolver (Plan ToM T8).
    /// CPU/GPU lowering deferred to T9.
    BeliefsAccessor {
        observer: Box<IrExprNode>,
        target: Box<IrExprNode>,
        field: String,
    },
    /// `beliefs(observer).confidence(target)` — read the `confidence` scalar.
    /// Syntactic sugar for `BeliefsAccessor { field: "confidence" }`.
    /// CPU/GPU lowering deferred to T9.
    BeliefsConfidence {
        observer: Box<IrExprNode>,
        target: Box<IrExprNode>,
    },
    /// `beliefs(observer).<view_name>(_)` — aggregate view over the believed
    /// target set. CPU/GPU lowering deferred to T9.
    BeliefsView {
        observer: Box<IrExprNode>,
        view_name: String,
    },
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
    /// Struct-shaped variant pattern: `Damage { amount }` or
    /// `Slow { duration_ticks, factor_q8: f }`. `ctor` is set when the
    /// variant name resolves (e.g. to an `EffectOp` variant — currently a
    /// stdlib-known sum type; the emitter hardcodes the enum prefix). Each
    /// binding names a variant field and either introduces a shorthand bind
    /// (same local name as the field) or a nested aliased pattern.
    Struct { name: String, ctor: Option<CtorRef>, bindings: Vec<IrPatternBinding> },
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
    /// `beliefs(observer).observe(target) with { field: expr, ... }` — belief
    /// mutation primitive resolved from the DSL surface (Plan ToM Task 4).
    ///
    /// `observer` and `target` are local-scope references (typically event
    /// bindings like `actor`). `fields` is the validated list of
    /// `BeliefState` field assignments; each name has been checked against
    /// the `BELIEF_FIELDS` allowlist in the resolver.
    ///
    /// Lowering to emitted Rust is deferred to T5 (emit_physics).
    BeliefObserve {
        observer: IrExprNode,
        target: IrExprNode,
        fields: Vec<IrFieldInit>,
        span: Span,
    },
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
    /// Tag references this event claims. Resolved from `@tag_name`
    /// annotations in pass 1. A claimed tag implies the event declares
    /// every field in the tag with matching name + type (validated in
    /// pass 2).
    pub tags: Vec<EventTagRef>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

/// `event_tag <Name>` declaration. The listed fields are the contract every
/// event claiming this tag must satisfy (same name + matching type).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventTagIR {
    pub name: String,
    pub fields: Vec<EventField>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

/// `enum <Name> { <Variant>, ... }` — user-declared enum surface. Emitted as
/// `#[repr(u8)]` Rust + Python `IntEnum`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnumIR {
    pub name: String,
    pub variants: Vec<String>,
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
    /// Intentionally-CPU-only rule (from `@cpu_only` annotation). Emit
    /// paths check this to skip WGSL emission + GPU dispatcher entry;
    /// validator uses it to bypass GPU-emittable checks.
    pub cpu_only: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicsHandlerIR {
    pub pattern: IrPhysicsPattern,
    pub where_clause: Option<IrExprNode>,
    pub body: Vec<IrStmt>,
    pub span: Span,
}

/// A physics `on` pattern at the IR layer — either a kind match or a tag
/// match. Tag matches resolve against the compiler's `EventTagIR` catalog.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IrPhysicsPattern {
    Kind(IrEventPattern),
    Tag {
        name: String,
        tag: Option<EventTagRef>,
        bindings: Vec<IrPatternBinding>,
        span: Span,
    },
}

impl IrPhysicsPattern {
    pub fn span(&self) -> Span {
        match self {
            IrPhysicsPattern::Kind(p) => p.span,
            IrPhysicsPattern::Tag { span, .. } => *span,
        }
    }
    pub fn bindings(&self) -> &[IrPatternBinding] {
        match self {
            IrPhysicsPattern::Kind(p) => &p.bindings,
            IrPhysicsPattern::Tag { bindings, .. } => bindings,
        }
    }
    pub fn display_name(&self) -> &str {
        match self {
            IrPhysicsPattern::Kind(p) => &p.name,
            IrPhysicsPattern::Tag { name, .. } => name,
        }
    }
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
    /// Optional `from <expression>` clause — the candidate source for
    /// target-bound masks. When `Some`, the emitted mask enumerator
    /// walks this expression (typically `query.nearby_agents(...)`)
    /// and filters each candidate through `predicate`. When `None`, the
    /// mask emits the legacy per-pair / self-predicate shape. Task 138.
    pub candidate_source: Option<IrExprNode>,
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
    /// Positional params: `(name, local slot, resolved type)`. Untyped
    /// params default to `IrType::AgentId` so every v1 target-bound
    /// mask (`Attack(target)`, `MoveToward(target)`) preserves the
    /// implicit-agent contract. Typed params (`Cast(ability:
    /// AbilityId)`) surface non-agent heads without rewriting every
    /// caller. Task 157.
    Positional(Vec<(String, LocalRef, IrType)>),
    Named(Vec<IrPatternBinding>),
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoringIR {
    /// Standard per-agent rows: `Head = expression`.
    pub entries: Vec<ScoringEntryIR>,
    /// `row <name> per_ability { ... }` rows. See `PerAbilityRowIR`.
    /// Added 2026-04-23 (GPU ability evaluation subsystem Phase 2).
    pub per_ability_rows: Vec<PerAbilityRowIR>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

/// Discriminant for the two scoring-row shapes currently supported.
///
/// * `Standard` — the existing per-agent `Head = expr` row. Scored once
///   per agent; emitted into `SCORING_TABLE` as a `ScoringEntry`.
/// * `PerAbility` — new `row <name> per_ability { ... }` row. Scored
///   once per (agent, ability) pair. See `PerAbilityRowIR`.
///
/// Kept as a bare enum rather than folded into `ScoringEntryIR` so
/// downstream emitters (Phase 3 CPU, Phase 4 GPU) can dispatch on row
/// shape without re-deriving it from the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ScoringRowKind {
    Standard,
    PerAbility,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoringEntryIR {
    pub head: IrActionHead,
    pub expr: IrExprNode,
    pub span: Span,
}

/// IR for a `per_ability` scoring row. The scoring kernel iterates each
/// agent's ability slots and scores every (agent, ability) pair; the
/// argmax over passing `guard`s is the ability the agent casts this
/// tick.
///
/// * `guard` — optional boolean predicate. `None` parses as `true`.
/// * `score` — f32 scoring expression (required).
/// * `target` — optional agent-id expression picking the cast target.
///
/// Phase 2 of the GPU ability evaluation subsystem. See
/// `docs/spec/engine.md §11`
/// §Architecture.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerAbilityRowIR {
    pub name: String,
    pub guard: Option<IrExprNode>,
    pub score: IrExprNode,
    pub target: Option<IrExprNode>,
    pub span: Span,
}

impl PerAbilityRowIR {
    /// Dispatch helper for downstream emitters that walk scoring
    /// entries and branch on row shape.
    #[inline]
    pub fn kind(&self) -> ScoringRowKind {
        ScoringRowKind::PerAbility
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ViewIR {
    pub name: String,
    pub params: Vec<IrParam>,
    pub return_ty: IrType,
    pub body: ViewBodyIR,
    pub annotations: Vec<Annotation>,
    /// View kind resolved from `@lazy` / `@materialized` annotations. Spec §2.3.
    pub kind: ViewKind,
    /// Parsed, validated form of `@decay(rate=R, per=tick)` if present.
    /// `None` when the view has no decay annotation. Only valid on
    /// `@materialized` views with a `Fold` body — enforced in resolve.
    pub decay: Option<DecayHint>,
    pub span: Span,
}

/// Top-level view kind — lazy (pure fn, evaluated at read) or materialized
/// (event-fold with persistent storage). Spec §2.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ViewKind {
    Lazy,
    Materialized(StorageHint),
}

/// Storage hint for `@materialized` views. Parsed from
/// `@materialized(storage = <hint>)` or from sibling view-shape
/// annotations (`@symmetric_pair_topk(...)`, `@per_entity_ring(...)`).
/// Spec §9 D31 + GPU cold-state replay plan (2026-04-22).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StorageHint {
    /// Dense pair-keyed map. Backed by `HashMap<(K1, K2), V>`. Default when
    /// `storage` is omitted; compiler rejects `pair_map` over
    /// `(AgentId, AgentId)` at N=200K as infeasible (spec §9 D31).
    PairMap,
    /// Bounded per-entity top-K. Backed by
    /// `HashMap<KeyedOn, SortedVec<V, K>>`.
    PerEntityTopK { k: u16, keyed_on: u8 },
    /// Symmetric pair-keyed per-entity storage. Each agent keeps up to
    /// `k` edges; reads dedupe by ordered-pair key so `(a, b)` and
    /// `(b, a)` resolve to the same entry. Bounded at K per agent with
    /// weakest-evicted policy. Gated by the
    /// `@symmetric_pair_topk(K = <n>)` view annotation.
    SymmetricPairTopK { k: u16 },
    /// Per-entity FIFO ring of fixed size K. Atomic write cursor
    /// increments mod K; oldest record evicted. Gated by the
    /// `@per_entity_ring(K = <n>)` view annotation.
    PerEntityRing { k: u16 },
    /// Compute-on-demand with per-tick cache. Backed by
    /// `RefCell<HashMap<Args, (V, tick)>>`.
    LazyCached,
}

/// Parsed `@decay(rate=R, per=tick)` annotation. The anchor-pattern
/// emitter reads `rate` to generate `base * rate.powi(tick - anchor)`.
/// Rate is a compile-time constant in `(0.0, 1.0)`. Variable decay
/// rates are not supported in v1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DecayHint {
    pub rate: f32,
    pub per: DecayUnit,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DecayUnit {
    /// `per = tick` — the only supported unit in v1.
    Tick,
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

/// Lowered `config <Name> { <field>: <type> = <default>, ... }` block. Each
/// field becomes one emitted Rust struct field + one TOML row; defaults
/// bake into `Default::default()`. The default literal is carried verbatim
/// from the AST so the TOML emitter can render it with the right shape for
/// each scalar type.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConfigIR {
    pub name: String,
    pub fields: Vec<ConfigFieldIR>,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConfigFieldIR {
    pub name: String,
    pub ty: IrType,
    pub default: ast::ConfigDefault,
    pub span: Span,
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
// Stdlib namespaces
// ---------------------------------------------------------------------------

/// Identifier for a Rust-backed stdlib namespace. The typed namespaces
/// (`World`, `Cascade`, `Event`, `Mask`, `Action`, `Rng`, `Query`, `Voxel`)
/// carry declared field and method schemas; the legacy collection
/// namespaces (`Agents`, `Items`, `Groups`, `Quests`, `Auctions`, `Tick`)
/// are iterables / sim-wide accessors whose per-field schema is not yet
/// spelled out in the compiler — they carry through unchanged for 1a.
///
/// See `docs/dsl/stdlib.md` for the canonical reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NamespaceId {
    World,
    Cascade,
    Event,
    Mask,
    Action,
    Rng,
    Query,
    Voxel,
    /// Runtime-tunable constants declared via `config <Name> { ... }` blocks.
    /// `config.<block>.<field>` is a two-hop lookup against `ConfigIR`;
    /// the resolver collapses it into a single `NamespaceField` whose
    /// `field` string is `"<block>.<field>"`.
    Config,
    /// `view::<name>(...)` — disambiguation syntax for calling a declared
    /// `view`. Equivalent to a bare `<name>(...)` when the callee resolves
    /// to a `ViewRef`; the resolver rewrites `NamespaceCall { ns: View,
    /// method, args }` into `IrExpr::ViewCall(ref, args)` as a convenience.
    /// Spec §2.3 (view) + emission path in `emit_view.rs`.
    View,
    // Legacy collection / accessor namespaces. Kept typed so the IR stays
    // closed; their fields are not yet declared.
    Agents,
    Items,
    Groups,
    Quests,
    Auctions,
    Tick,
    /// `abilities.*` — sim-wide accessor for the `AbilityRegistry` living on
    /// `SimState`. Methods: `abilities.is_known(id) -> bool`,
    /// `abilities.cooldown_ticks(id) -> u32`, `abilities.effects(id)` yields
    /// the program's `SmallVec<[EffectOp; N]>` as an iterable. Added so the
    /// `cast` physics rule can iterate and dispatch a cast's effect list
    /// without a hand-written cascade handler.
    Abilities,
    /// `terrain.*` — sim-wide accessor for the `TerrainQuery` backend
    /// living on `SimState`. Methods: `terrain.line_of_sight(from, to)
    /// -> bool`. Registered 2026-04-23 (Task 81 terrain integration
    /// MVP). Deliberately tiny surface — MVP slice only exposes LOS;
    /// `height_at` / `walkable` stay engine-internal for now and can be
    /// lifted into the DSL in a follow-up once a concrete scoring /
    /// mask row needs them.
    Terrain,
    /// `membership::*` — Subsystem §1 (roadmap.md:161-211). Predicates on
    /// per-agent `cold_memberships` SmallVec. Methods are `is_group_member`,
    /// `is_group_leader`, `can_join_group`, `is_outcast`. All return bool.
    /// Grammar stub only — emitters return `EmitError::Unsupported` until
    /// the memberships runtime lands.
    Membership,
    /// `relationship::*` — Subsystem §3 (roadmap.md:279-311). Predicates on
    /// per-agent `cold_relationships` SmallVec. Methods are `is_hostile`,
    /// `is_friendly`, `knows_well`. All return bool. Grammar stub only —
    /// emitters return `EmitError::Unsupported` until the relationships
    /// runtime lands (replaces Combat Foundation's stub `is_hostile_to`
    /// with valence-based friendship / hostility thresholds).
    Relationship,
    /// `theory_of_mind::*` — Subsystem §6 (roadmap.md:447-506). Predicates
    /// over `Relationship.believed_knowledge: Bitset<32>`. Methods are
    /// `believes_knows`, `can_deceive`, `is_surprised_by`. All return bool.
    /// Grammar stub only — emitters return `EmitError::Unsupported` until
    /// the theory-of-mind runtime lands (gossip / belief-tracking fold).
    TheoryOfMind,
    /// `group::*` — Subsystem §7 (roadmap.md:510-574). Predicates on the
    /// `AggregatePool<Group>` pool. Methods are `exists`, `is_active`,
    /// `has_leader`, `can_afford_from_treasury`. All return bool.
    /// Grammar stub only — emitters return `EmitError::Unsupported` until
    /// the groups runtime lands (Plan 1 T16 shipped the Pod shape; this
    /// subsystem populates the instance data).
    ///
    /// Singular name `group` chosen to match the roadmap spelling; the
    /// pre-existing plural `groups` namespace (legacy collection accessor)
    /// is unchanged and continues to resolve independently.
    Group,
    /// `quest::*` — Subsystem §12 (roadmap.md:811-872). Predicates on the
    /// `AggregatePool<Quest>` pool. Methods are `can_accept`, `is_target`,
    /// `party_near_destination`. All return bool. Grammar stub only —
    /// emitters return `EmitError::Unsupported` until the quests runtime
    /// lands (Pod shape shipped by Plan 1 T16; instance data pending).
    ///
    /// Singular `quest` is distinct from the pre-existing plural
    /// `quests` (legacy collection accessor).
    Quest,
}

impl NamespaceId {
    pub fn name(&self) -> &'static str {
        match self {
            NamespaceId::World => "world",
            NamespaceId::Cascade => "cascade",
            NamespaceId::Event => "event",
            NamespaceId::Mask => "mask",
            NamespaceId::Action => "action",
            NamespaceId::Rng => "rng",
            NamespaceId::Query => "query",
            NamespaceId::Voxel => "voxel",
            NamespaceId::Config => "config",
            NamespaceId::View => "view",
            NamespaceId::Agents => "agents",
            NamespaceId::Items => "items",
            NamespaceId::Groups => "groups",
            NamespaceId::Quests => "quests",
            NamespaceId::Auctions => "auctions",
            NamespaceId::Tick => "tick",
            NamespaceId::Abilities => "abilities",
            NamespaceId::Terrain => "terrain",
            NamespaceId::Membership => "membership",
            NamespaceId::Relationship => "relationship",
            NamespaceId::TheoryOfMind => "theory_of_mind",
            NamespaceId::Group => "group",
            NamespaceId::Quest => "quest",
        }
    }
}

// ---------------------------------------------------------------------------
// Builtins
// ---------------------------------------------------------------------------

/// Rust-backed stdlib primitive functions. These are the engine-intrinsic
/// callables the compiler recognises without requiring a DSL declaration.
/// See `docs/dsl/stdlib.md` for the complete signature reference.
///
/// Note: the enum is intentionally flat (no separate `StdlibFn` sister
/// enum). All stdlib primitives share the same emitter dispatch as the
/// aggregation / spatial builtins that were here before this milestone, so
/// keeping them in one enum avoids a second match in every consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Builtin {
    // Aggregations / quantifiers (legacy).
    Count,
    Sum,
    Forall,
    Exists,
    // Spatial.
    Distance,
    PlanarDistance,
    ZSeparation,
    // ID dereference.
    Entity,
    // Numeric. `Min`/`Max` double as fold aggregators in existing use; the
    // runtime dispatches on arity (one arg over an iterable = aggregation,
    // two args = pairwise min/max).
    Min,
    Max,
    Clamp,
    Abs,
    Floor,
    Ceil,
    Round,
    Ln,
    Log2,
    Log10,
    Sqrt,
    /// `saturating_add(a, b)` — saturating addition on integer scalars.
    /// Clamps to the type's MAX on overflow instead of wrapping or
    /// panicking. Used by the `cast` physics rule to compute absolute
    /// expiry ticks (`tick + duration_ticks`) without reaching for a
    /// method-call syntax the DSL doesn't otherwise expose.
    SaturatingAdd,
}

impl Builtin {
    pub fn name(&self) -> &'static str {
        match self {
            Builtin::Count => "count",
            Builtin::Sum => "sum",
            Builtin::Forall => "forall",
            Builtin::Exists => "exists",
            Builtin::Distance => "distance",
            Builtin::PlanarDistance => "planar_distance",
            Builtin::ZSeparation => "z_separation",
            Builtin::Entity => "entity",
            Builtin::Min => "min",
            Builtin::Max => "max",
            Builtin::Clamp => "clamp",
            Builtin::Abs => "abs",
            Builtin::Floor => "floor",
            Builtin::Ceil => "ceil",
            Builtin::Round => "round",
            Builtin::Ln => "ln",
            Builtin::Log2 => "log2",
            Builtin::Log10 => "log10",
            Builtin::Sqrt => "sqrt",
            Builtin::SaturatingAdd => "saturating_add",
        }
    }

    /// Fixed arity for primitives whose call shape is pinned. `None` means
    /// the call may vary (e.g. `min`/`max` can be pairwise or fold-over-iter).
    pub fn fixed_arity(&self) -> Option<usize> {
        match self {
            Builtin::Distance | Builtin::PlanarDistance | Builtin::ZSeparation => Some(2),
            Builtin::Entity => Some(1),
            Builtin::Clamp => Some(3),
            Builtin::Abs
            | Builtin::Floor
            | Builtin::Ceil
            | Builtin::Round
            | Builtin::Ln
            | Builtin::Log2
            | Builtin::Log10
            | Builtin::Sqrt => Some(1),
            Builtin::SaturatingAdd => Some(2),
            // Quantifiers are parsed as a dedicated AST node, not a call; this
            // entry is for completeness only.
            Builtin::Forall | Builtin::Exists => None,
            Builtin::Count | Builtin::Sum | Builtin::Min | Builtin::Max => None,
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
    pub event_tags: Vec<EventTagIR>,
    pub enums: Vec<EnumIR>,
    pub entities: Vec<EntityIR>,
    pub physics: Vec<PhysicsIR>,
    pub masks: Vec<MaskIR>,
    pub scoring: Vec<ScoringIR>,
    pub views: Vec<ViewIR>,
    pub verbs: Vec<VerbIR>,
    pub invariants: Vec<InvariantIR>,
    pub probes: Vec<ProbeIR>,
    pub metrics: Vec<MetricIR>,
    pub configs: Vec<ConfigIR>,
    pub spans: SpanTable,
}
