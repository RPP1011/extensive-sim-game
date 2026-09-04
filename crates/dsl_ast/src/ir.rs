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
ref_newtype!(SpatialQueryRef);

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
    /// `per_ability` scoring row (where "the currently-scored ability"
    /// has a binding). Emit time enforces this at Phase 3.
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
    /// `for_each_agent <binder> { <body> }` — body-shape primitive that
    /// walks every alive agent slot in deterministic linear order
    /// (slot 0 → slot agent_cap-1). Resolved sibling of `Stmt::ForEachAgent`.
    /// Lowering produces a `CgStmt::ForEachAgentBody` and (for per-agent
    /// rules) retags dispatch to `OneShot` so a single thread executes the
    /// linear scan once per tick.
    ForEachAgent {
        binder: LocalRef,
        binder_name: String,
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
    /// Plan G G3b/G3c — `self.append(field1: expr1, field2: expr2, ...)` —
    /// struct-payload ring append in a `@per_entity_ring(...)` view fold
    /// body. Each cell of the ring stores a multi-field struct whose
    /// layout is implied by the field list (declaration order, types
    /// inferred from the bound exprs at lowering time). The ring index
    /// is allocated via the per-agent cursor counter; the per-field
    /// stores write into `primary[ring_idx * field_count + field_idx]`.
    ///
    /// Distinct from `SelfUpdate { op = "+=" }` — that scalar accumulate
    /// path stays in place for `recent_damages` style folds; this new
    /// shape unblocks the threats view's struct-cell payload.
    SelfAppend {
        fields: Vec<IrFieldInit>,
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
    /// `apply_ability <ability_expr>` — registry-driven dispatch (#132).
    /// Resolved sibling of `Stmt::ApplyAbility`; the WGSL emitter expands
    /// this into a per-effect-slot dispatch loop reading from
    /// `PackedAbilityRegistry` SoA columns. Apply handler reads
    /// `ability_id` from the resolved expression at runtime; caster
    /// defaults to `self`, target to the verb's `target` binder.
    ApplyAbility {
        ability: IrExprNode,
        /// Symbolic ability-name from the `apply_ability <Name> …`
        /// surface (2026-05-12). `Some(name)` when the parser captured a
        /// bare identifier as the ability operand; the lowerer resolves
        /// it against `LoweringCtx::ability_name_to_id` (populated from
        /// `LowerOpts::ability_names`) and replaces the dispatched
        /// AbilityId. `None` for numeric / general-expression surfaces
        /// (the existing path lowers the `ability` expression unchanged).
        ability_name: Option<String>,
        /// Optional explicit caster operand from `apply_ability <a> by
        /// <caster>` source-level syntax (slice δ part 3, #161). When
        /// `None`, lowering uses the per-thread agent (PerAgent rules)
        /// or surfaces a typed error (PerEvent rules).
        caster:  Option<IrExprNode>,
        /// Optional explicit target operand from `apply_ability <a>
        /// [by <c>] target <t>` (slice ε part 1). When `None`, the
        /// dispatcher writes the caster into the target chronicle
        /// slot (slice-γ self-cast convention).
        target:  Option<IrExprNode>,
        span:    Span,
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
    /// Engine-aliased `EventKindId` discriminant for events whose name
    /// matches a hardcoded engine event (`EffectDamageApplied = 26`,
    /// etc.). `None` for user-declared events — those are allocated
    /// sequentially by `crate::engine_events::assign_event_kind_ids`,
    /// which SKIPS every reserved engine discriminant to avoid collisions
    /// with the dispatcher's hardcoded event tags.
    ///
    /// Populated by `dsl_ast::resolve` from
    /// `crate::engine_events::engine_event_kind_id_for_name`. The
    /// lowering driver's `populate_event_kinds` mirrors this assignment
    /// so the kernel's filter constant matches the dispatcher's
    /// hardcoded write tag (closed loop for the chronicle pipeline).
    /// See `assets/sim/apply_ability_chronicle_consumer.sim` for the
    /// motivating fixture and `assets/sim/many_events_ability.sim` for
    /// the >25-event collision pin.
    pub engine_kind_id: Option<u32>,
}

impl EventIR {
    /// Returns `true` when this event carries the `@traced` annotation —
    /// the surface that marks an event as a non-replayable observability /
    /// diagnostics record (sibling to `@non_replayable`, inverse-intent
    /// of `@replayable`). Fold consumers that produce the deterministic
    /// trace hash should skip events for which this returns `true`;
    /// trace consumers (histograms, per-tick debug logs, chronicle
    /// renderers) should include them.
    ///
    /// **Surface status (Gap forest_fire#E, 2026-05-12):** the
    /// annotation parses + resolves through the generic annotation
    /// surface today (no special parser arm). Pinned by
    /// `crates/dsl_compiler/tests/traced_annotation_parses.rs` and
    /// `crates/dsl_compiler/tests/predator_prey_non_replayable.rs`.
    /// Wiring `is_traced` into the per-kind [`EventLayout`] (so the
    /// schedule synthesizer can route traced events to a separate ring
    /// and the host fold can filter on it without re-walking the
    /// `EventIR.annotations` vec) is a follow-up scoped behind the same
    /// gap entry.
    ///
    /// [`EventLayout`]: ../../dsl_compiler/cg/program/struct.EventLayout.html
    pub fn is_traced(&self) -> bool {
        self.annotations.iter().any(|a| a.name == "traced")
    }

    /// Returns `true` when this event carries the `@non_replayable`
    /// annotation — the surface that routes the event off the
    /// deterministic replay ring and onto the host-side chronicle ring
    /// (string payloads allowed, no `Pod` round-trip). The matching
    /// resolver path lives in
    /// [`crate::resolve`] (`non_replayable_events` collection +
    /// `validate_physics_body`).
    pub fn is_non_replayable(&self) -> bool {
        self.annotations.iter().any(|a| a.name == "non_replayable")
    }
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
    /// `predator_prey: { prey_of: [], preys_on: [] }` — anonymous
    /// struct body, shape implicit from the field's declared type.
    AnonStruct {
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
/// * `weights` — optional f32 utility-table addend. When present, the
///   lowerer composes the row's utility as `score + weights` (both
///   F32). Mirrors the AST `weights:` clause; design-target fixtures
///   that use the `base: <const>, weights: <expr>` shape route through
///   this field. See `crates/dsl_ast/src/ast.rs::PerAbilityRow.weights`.
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
    pub weights: Option<IrExprNode>,
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
    /// `@belief_gated` annotation present? When `true`, a
    /// PerAgentEventScan fold's source-candidate gate switches from
    /// `agent_busy_with_ability_id[source]` (omniscient — every
    /// observer sees every busy source) to
    /// `beliefs_flags[observer * agent_cap + source] & (1u << bit)`
    /// (observation-shaped — only observers whose belief bit is set
    /// see the source). The bit position is `BELIEF_BIT_OBSERVED_BUSY = 7`
    /// by convention; chosen to avoid collision with the live
    /// `physics_WhatIBelieve` self-stamp at bit 0. Opt-in per-view
    /// so existing fixtures (dodger_probe / threats_view_probe) keep
    /// the omniscient gate by default.
    pub belief_gated: bool,
    /// `@storage(<name>)` annotation. Defaults to `Packing::None` (one
    /// logical cell per u32 word — the legacy shape every existing
    /// view uses). `Packing::Q8` switches the per-cell storage to
    /// 4 packed u8 cells per u32 word; the decay + fold WGSL emit
    /// branch on this to compose byte-shift unpack / repack steps.
    /// Plumbed from the AST `@storage(packed_q8)` annotation —
    /// see `lower_storage_annotation` in `resolve.rs`.
    pub storage_packing: Packing,
    /// Plan I — social-merge handlers for the `belief` declaration
    /// shape. Each entry encodes a `merge from <agent>: <op>` clause
    /// — receiver bitwise-merges the named agent's belief storage
    /// into their own when the event fires + the predicate passes.
    /// Empty for every `view` declaration (only beliefs populate it).
    /// Lowered by the (future) belief-arm of `lower_view` to a
    /// `ComputeOpKind::BeliefSocialMerge` op kind.
    pub social_merges: Vec<SocialMergeHandler>,
    pub span: Span,
}

/// Plan I — one `merge from <agent>: <op>` clause inside a `belief`
/// declaration. The pattern + predicate gate when the merge fires;
/// `source_agent` is the LocalRef of the event field carrying the
/// belief-giver agent (e.g. `dead` in `on AllyDied { dead: d, ... }`).
/// `op` picks the per-cell merge operation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SocialMergeHandler {
    pub pattern: IrEventPattern,
    pub where_clause: Option<IrExprNode>,
    pub source_agent: LocalRef,
    pub op: MergeOp,
    pub span: Span,
}

/// Plan I — per-cell merge operation for a social-merge handler. All
/// variants are commutative + associative under the per-event ordering
/// the chronicle's `seq` field provides, so cross-backend determinism
/// (P11) is bit-exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MergeOp {
    /// Bitwise OR — receiver gains every bit the giver has set.
    /// Used for bitmap-shape beliefs (e.g. room knowledge: receiver
    /// learns every room the giver knew).
    BitOr,
    /// Per-cell max — receiver takes the higher confidence/value
    /// between their own and the giver's. Used for scalar beliefs
    /// where higher = stronger evidence.
    Max,
    /// Per-cell min — symmetric to Max, lower = stronger evidence.
    Min,
    /// Per-cell replace — receiver overwrites their cell with the
    /// giver's. Used when freshness matters more than aggregation
    /// (the most recent observer wins).
    Replace,
}

/// View-storage packing discriminator. Drives the WGSL emit's per-cell
/// vs per-word arithmetic in the decay + fold kernels — `Packing::None`
/// keeps the legacy "one logical cell per u32" path; `Packing::Q8`
/// switches to "4 packed u8 cells per u32 word" with byte-shift
/// unpack / repack arithmetic. Future packings (`Packing::Q4`,
/// `Packing::Q16`, …) extend the same surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Packing {
    /// Default: one logical cell per u32 word. Buffer sizing is
    /// `cells * 4 bytes`; per-cell decay reads `view_storage_primary[k]`
    /// directly.
    None,
    /// q8: 4 packed u8 cells per u32 word, little-endian byte order.
    /// Buffer sizing is `(cells + 3) / 4 * 4 bytes` (round up to a full
    /// word). Per-cell decay reads the WORD, decomposes to 4 bytes,
    /// applies the per-cell step (and gate predicate, if any) to each
    /// byte, recomposes, atomic-stores. Mirrors the bespoke
    /// `belief_decay_wgsl::decay_kernel_wgsl` shape that this
    /// annotation subsumes.
    Q8,
}

/// Top-level view kind — lazy (pure fn, evaluated at read) or materialized
/// (event-fold with persistent storage). Spec §2.3.
///
/// **Plan I (`docs/superpowers/plans/2026-05-15-belief-primitive.md`)** —
/// `Belief` is a marker variant for the new `belief <name>(observer:
/// Agent[, key]) -> T { ... }` declaration. It stays a marker (no
/// payload) so `ViewKind` keeps its `Copy` impl; the social-merge
/// handlers + storage-shape inference inputs live on dedicated
/// [`ViewIR`] fields (`social_merges`, future inference flags).
/// Storage hint is filled in by the lowering's signature-based
/// inference table — author doesn't pick it for beliefs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ViewKind {
    Lazy,
    Materialized(StorageHint),
    Belief,
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
/// Rate is a compile-time constant in `[0.0, 1.0)` — `0.0` is the
/// "full reset every tick" idiom (decay multiplier zeroes prior
/// storage before the fold runs); `1.0` is rejected as a no-op.
/// Variable decay rates are not supported in v1.
///
/// **Wave-N extension (mode = sub).** The annotation now carries a
/// `mode` discriminator + `by` magnitude that subsume the legacy
/// `rate = R` form:
///
///   * `mode = mul, by = R` (also spelled `rate = R`) — per-tick
///     `cell = old * R` (legacy anchor-pattern).
///   * `mode = sub, by = N` — per-tick `cell = saturating_sub(old, N)`
///     (linear decay; tom_probe's belief-confidence shape).
///
/// `rate` is preserved as the compile-time constant scalar for the
/// `mul` mode (in `[0.0, 1.0)`). For `sub` mode, `sub_by` carries the
/// positive integer step. The two share the `span` + `per` fields.
///
/// **Optional `gate = MaskName`.** When set, the emitted decay kernel
/// wraps its per-cell body in `if (<mask_predicate>) { ... }`, so cells
/// where the predicate evaluates FALSE are left untouched. The mask is
/// resolved against the compilation's `masks` vec by name. Today the
/// runtime stops at this IR field — the WGSL emit reports an
/// unsupported-gate diagnostic when the gate mask's predicate
/// references view-storage cells (cross-binding plumbing not yet
/// available for decay kernels).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DecayHint {
    pub rate: f32,
    pub per: DecayUnit,
    pub mode: DecayMode,
    /// For `mode = sub`: the integer step magnitude (positive). Unused
    /// in `mode = mul` (the multiplier travels via `rate`).
    pub sub_by: u32,
    /// Resolved `MaskRef` for the optional `gate = <MaskName>` argument.
    /// `None` when no gate is specified. Resolution happens in
    /// `resolve.rs::lower_decay_hint` once the global symbol table is
    /// populated. The `MaskRef`'s u16 indexes `Compilation.masks`.
    pub gate: Option<MaskRef>,
    pub span: Span,
}

/// Discriminator for the `@decay(mode = ...)` argument. Defaults to
/// `Mul` for the legacy `rate = R` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DecayMode {
    /// Per-tick `cell = old * rate` (legacy anchor-pattern).
    Mul,
    /// Per-tick `cell = saturating_sub(old, sub_by)`. Targets integer
    /// storage (u8/u16/u32). The decay kernel's per-cell body emits
    /// `let new_val = select(old - by, 0, old < by);` (or the
    /// equivalent u32 saturating-sub).
    Sub,
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

/// Resolved `spatial_query <name>(self, candidate, <args>) = <filter>`
/// declaration (Phase 7 Task 4).
///
/// Mirrors `ViewIR` (name + LocalRef-bound params + body), with two
/// shape narrowings:
///   - No `return_ty`: the filter is implicitly Bool. The CG
///     well_formed gate (Phase 7 Task 3) enforces the type once the
///     filter expression lowers into the `SpatialQueryKind::FilteredWalk`
///     op.
///   - No `kind` discriminator: every spatial_query is structurally a
///     `FilteredWalk` once constructed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpatialQueryIR {
    pub name: String,
    pub params: Vec<IrParam>,
    pub filter: IrExprNode,
    pub annotations: Vec<Annotation>,
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
    /// Resolved verb body statements — a sequence of `IrStmt::Emit`
    /// (`emit <Event> { ... }` source) and/or `IrStmt::ApplyAbility`
    /// (`apply_ability <a> [by <c>] [target <t>]` source). Lifted
    /// verbatim by the verb expander into the synthesised
    /// `verb_chronicle_<name>` cascade physics handler. Wave 1.7 /
    /// task #138 generalised this from `Vec<IrEmit>` so apply_ability
    /// can ride the same cascade pipeline.
    pub body: Vec<IrStmt>,
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
    /// Mirrors [`ast::ConfigField::runtime`] — `true` when the source
    /// field carried the `@runtime` annotation. Threaded through resolve
    /// so the CG-lowering driver can mark the resulting `ConfigConstId`
    /// as runtime-tunable (per-kernel cfg-uniform field) instead of the
    /// default baked-WGSL-const path.
    pub runtime: bool,
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
    /// `spatial::<name>(...)` / `spatial.<name>(...)` — references to
    /// declared `spatial_query` filters. Resolved against
    /// `Compilation::spatial_queries` by the resolver; lowering
    /// (Phase 7 Task 5) substitutes the filter expression into the
    /// owning mask's `SpatialQueryKind::FilteredWalk`.
    Spatial,
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
    /// -> bool`. Deliberately tiny MVP surface — only LOS is exposed;
    /// `height_at` / `walkable` stay engine-internal and can be lifted into
    /// the DSL in a follow-up once a concrete scoring / mask row needs them.
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
    /// `threats::*` — Plan G G3f (2026-05-09). Per-agent scoring
    /// primitives over the threats materialised view (G3g, future).
    /// Methods `in_zone`, `intensity_at`, `nearest`,
    /// `dir_away_from_nearest` resolve to the corresponding
    /// `Builtin::Threats*` variants in the resolver — the threats
    /// fold's per-cell walk is invisible at the DSL surface.
    /// See `docs/plans/g3_threats_view_design.md`.
    Threats,
    /// `events::*` — Sim-wide event-trace accessors used by metrics,
    /// invariants, and probes (NOT physics / view bodies). Surfaces
    /// the per-tick event ring + history-window queries that
    /// trace-consumer hooks (`metric { value: count(e in
    /// events.this_tick where e.kind == X) }`) read against.
    ///
    /// Methods/fields:
    ///   * `events.this_tick` — accessor returning the current-tick
    ///     events vec (iter source for `count(e in events.this_tick
    ///     where ...)` folds).
    ///   * `events.at_tick(tick: u32) -> [Event]` — events recorded at
    ///     a specific tick.
    ///   * `events.range(from: u32, to: u32) -> [Event]` — range
    ///     query over the trace history.
    ///   * `events.kind_count(kind: EventKindId) -> u32` — count of
    ///     events of a given kind in the current tick.
    ///
    /// The events namespace is META-LEVEL: resolves cleanly into
    /// `IrExpr::NamespaceField` / `IrExpr::NamespaceCall` so
    /// metric/invariant/probe shape classifiers see a structured node
    /// and emit per-name SKIP setters (the runtime fills in the actual
    /// count). Physics / view bodies cannot use `events.*` (not yet
    /// GPU-side); lowering surfaces `UnsupportedNamespaceCall`.
    ///
    /// Distinct from the singular `event` namespace which carries
    /// the per-handler currently-firing event accessors (`event.kind`,
    /// `event.tick`); `events` is the plural accessor over the trace
    /// stream as a whole.
    Events,
    /// `tables::*` — static lookup tables declared at top level via
    /// `table <name>: <ty>[N] = […]`. `tables.<name>(<idx_expr>)`
    /// resolves to a typed `IrExpr::TableLookup { table, index }`
    /// during the resolver pass; lowering emits a WGSL `const`
    /// array prepended to each kernel that references the table
    /// plus an array-index expression in the body. Read-only world
    /// data — distinct from `Agents` (per-instance per-tick state).
    Tables,
    /// `navgrid::*` — voxel-region-indices spec Phase 4b
    /// (`docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`).
    /// Per-region 2D walkability + cell-height index built by
    /// `engine_voxel::build_navgrid` at host time and uploaded to a
    /// `navgrid` storage binding. Sole method today:
    /// `navgrid.walkable(cx: u32, cz: u32) -> bool` (the registry
    /// holds a single global navgrid; multi-region dispatch
    /// `navgrid.walkable(region, cx, cz)` lands when fixtures need
    /// it). Cells are u32-packed (low 8 bits walkable flag, mid 16
    /// bits height, top 8 reserved) per
    /// `engine_voxel::navgrid::NavgridCell`.
    Navgrid,
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
            NamespaceId::Spatial => "spatial",
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
            NamespaceId::Threats => "threats",
            NamespaceId::Events => "events",
            NamespaceId::Tables => "tables",
            NamespaceId::Navgrid => "navgrid",
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
    /// `normalize(v)` — vec3 unit-vector. Length 0 returns `(0,0,0)`
    /// (the lowering uses [`UnaryOp::NormalizeVec3F32`] which the WGSL
    /// emitter renders as a `normalize(...)` call). Used by movement /
    /// flocking rules that need a direction from a delta. Spec-compliant
    /// vector primitive.
    Normalize,
    /// `length(v)` — Euclidean norm of a `vec3<f32>`. Returns `f32`.
    /// Used by speed-cap clamps and invariants. Lowers to the WGSL
    /// `length(...)` builtin via [`BuiltinId::LengthVec3F32`].
    Length,
    /// `dot(a, b)` — dot product of two `vec3<f32>` operands. Returns
    /// `f32`. Used by direction-filter masks (only count obstructions
    /// in front of self) and projection arithmetic. Lowers to the WGSL
    /// `dot(...)` builtin via [`BuiltinId::DotVec3F32`].
    Dot,
    /// `saturating_add(a, b)` — saturating addition on integer scalars.
    /// Clamps to the type's MAX on overflow instead of wrapping or
    /// panicking. Used by the `cast` physics rule to compute absolute
    /// expiry ticks (`tick + duration_ticks`) without reaching for a
    /// method-call syntax the DSL doesn't otherwise expose.
    SaturatingAdd,
    /// `vec3(x, y, z)` — construct a Vec3 from three scalar components.
    /// All three operands are F32; result is Vec3F32. The lone vec3
    /// literal form supported by the DSL today.
    Vec3,
    /// `f32(x)` — explicit cast of a numeric scalar to `f32`. Source
    /// must be `i32` or `u32`; a same-type cast is rejected at lowering
    /// so authors can't accidentally mask a type-inference mistake.
    /// Lowers to `BuiltinId::AsF32(<src>)` which emits WGSL `f32(<arg>)`.
    F32Cast,
    /// `u32(x)` — explicit cast of a numeric scalar to `u32`. Source
    /// must be `f32` (lossy truncation toward zero) or `i32` (lossy
    /// when negative). Lowers to `BuiltinId::AsU32(<src>)` which emits
    /// WGSL `u32(<arg>)`.
    U32Cast,
    /// `i32(x)` — explicit cast of a numeric scalar to `i32`. Source
    /// must be `f32` (truncation toward zero) or `u32` (high bit may
    /// flip sign). Lowers to `BuiltinId::AsI32(<src>)` which emits
    /// WGSL `i32(<arg>)`.
    I32Cast,

    // -- Threats scoring primitives (Plan G G3f, 2026-05-09). --
    //
    // Resolved from `threats.<method>(...)` / `threats::<method>(...)`
    // by the resolver — the `threats` namespace dispatches the four
    // method names to these `Builtin` variants directly (rather than
    // routing through `IrExpr::NamespaceCall`) so the lowering pass
    // sees a single closed-set enum to dispatch on.
    //
    // Today's CG lowering emits sentinel values (`false` / `0.0` /
    // `AgentId(0)` / `vec3(0)`) — the threats materialised view
    // (G3g, future) wires the per-cell walk over the agent's
    // threat-zone ring. The Builtin surface is the load-bearing
    // piece; the WGSL behaviour is downstream.
    //
    // Argument shapes per the design doc's "Scoring primitives" table:
    // * `threats.in_zone(self) -> bool` — any live cell where self.pos
    //   is in the projected zone.
    // * `threats.intensity_at(self.pos) -> f32` — sum of (radius -
    //   distance) over live cells.
    // * `threats.nearest(self) -> AgentId` — source of the closest
    //   live zone.
    // * `threats.dir_away_from_nearest(self) -> Vec3` — unit vector
    //   from nearest threat's centre toward self.

    /// `threats.in_zone(self) -> bool` — true when the agent's current
    /// position falls inside any live threat zone. Lowers to a per-
    /// cell walk over the agent's threat-zone ring (K=4 cells today),
    /// OR-reducing the per-cell `in_zone` test.
    ThreatsInZone,
    /// `threats.intensity_at(pos) -> f32` — sum of `(radius - dist)`
    /// over every live cell whose zone covers `pos`. Negative
    /// distances are clamped at zero so distant threats contribute
    /// nothing. Lowers to a per-cell walk over the agent's threat-
    /// zone ring with a sum-reduce.
    ThreatsIntensityAt,
    /// `threats.nearest(self) -> AgentId` — caster id of the live
    /// threat zone whose centre is nearest to self.pos. Returns
    /// `AgentId::SENTINEL` (0) when no threats are active. Lowers to
    /// a per-cell argmin walk.
    ThreatsNearest,
    /// `threats.dir_away_from_nearest(self) -> Vec3` — unit vector
    /// pointing FROM the nearest live threat zone's centre TO self.
    /// Returns `Vec3::ZERO` when no threats are active. Lowers to
    /// the same per-cell argmin walk as `ThreatsNearest`, then
    /// derives the unit direction from the winning cell's centre +
    /// self.pos.
    ThreatsDirAwayFromNearest,

    /// `next_waypoint(group) -> Vec3` — returns the next destination
    /// waypoint for a group. Today this is a placeholder (sentinel
    /// `Vec3::ZERO`) so design-target fixtures (`crowd_navigation.sim`)
    /// can lower without requiring a quest/landmark runtime. The real
    /// implementation would consult quest state or random landmarks per
    /// the `crowd_navigation.sim` author note. Arity 1 (the group id).
    NextWaypoint,
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
            Builtin::Normalize => "normalize",
            Builtin::Length => "length",
            Builtin::Dot => "dot",
            Builtin::SaturatingAdd => "saturating_add",
            Builtin::Vec3 => "vec3",
            Builtin::F32Cast => "f32",
            Builtin::U32Cast => "u32",
            Builtin::I32Cast => "i32",
            Builtin::ThreatsInZone => "threats.in_zone",
            Builtin::ThreatsIntensityAt => "threats.intensity_at",
            Builtin::ThreatsNearest => "threats.nearest",
            Builtin::ThreatsDirAwayFromNearest => "threats.dir_away_from_nearest",
            Builtin::NextWaypoint => "next_waypoint",
        }
    }

    /// Fixed arity for primitives whose call shape is pinned. `None` means
    /// the call may vary (e.g. `min`/`max` can be pairwise or fold-over-iter).
    pub fn fixed_arity(&self) -> Option<usize> {
        match self {
            Builtin::Distance | Builtin::PlanarDistance | Builtin::ZSeparation => Some(2),
            Builtin::Entity => Some(1),
            Builtin::Clamp => Some(3),
            Builtin::Vec3 => Some(3),
            Builtin::F32Cast | Builtin::U32Cast | Builtin::I32Cast => Some(1),
            Builtin::Abs
            | Builtin::Floor
            | Builtin::Ceil
            | Builtin::Round
            | Builtin::Ln
            | Builtin::Log2
            | Builtin::Log10
            | Builtin::Sqrt
            | Builtin::Normalize
            | Builtin::Length => Some(1),
            Builtin::Dot => Some(2),
            Builtin::SaturatingAdd => Some(2),
            // Threats scoring primitives — every variant is arity 1
            // (a single agent / pos arg). See the per-variant doc.
            Builtin::ThreatsInZone
            | Builtin::ThreatsIntensityAt
            | Builtin::ThreatsNearest
            | Builtin::ThreatsDirAwayFromNearest => Some(1),
            // `next_waypoint(group)` placeholder — single arg.
            Builtin::NextWaypoint => Some(1),
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
    /// Resolved `spatial_query <name>(self, candidate, …) = <filter>`
    /// declarations (Phase 7 Task 4). Filter expressions are consumed
    /// by `cg/lower/mask.rs` once the from-clause refactor (Task 5)
    /// lands. `Vec<_>` to match the `views` convention; the symbol
    /// table carries the name → index lookup.
    ///
    /// `skip_serializing_if` keeps the JSON snapshot stable for
    /// fixtures that don't declare any spatial_query — Task 4 lands
    /// the surface; Task 5 refactors fixtures to use it. The IR
    /// shape stays additive on the snapshot side.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spatial_queries: Vec<SpatialQueryIR>,
    /// Static lookup tables declared by `table <name>: <ty>[N] = […]`
    /// at the top level. World-data primitive — read-only at run
    /// time, lowers to a WGSL `const` array prepended to every
    /// kernel that references it via `tables.<name>(<idx_expr>)`.
    ///
    /// Right home for "rooms / terrain costs / faction stances /
    /// item-tier loot tables" — data the simulation reads but does
    /// not mutate. Skip-if-empty preserves JSON snapshot stability
    /// for fixtures that don't declare any tables.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<TableIR>,
    /// Voxel-region kinds declared via `region_kind <Name> {
    /// max_active = N }`. Per spec
    /// `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
    /// §6.1.2 — each kind gets its name + max-active count
    /// resolved into a typed [`VoxelRegionKindId`]. The matching
    /// `region_indices` decl is paired into the same slot via name
    /// match during pass-2.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub region_kinds: Vec<RegionKindIR>,
    /// Region-attached indices declared via `index <name>(region:
    /// VoxelRegion) -> <Output> { … }`. Per spec
    /// `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
    /// §7.2 — each index gets bounded storage, cost class, rebuild
    /// trigger, and (post-Phase-2b) a build expression tree. Phase
    /// 2a stores the build body as raw text; Phase 2b will resolve
    /// it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indices: Vec<IndexIR>,
    pub spans: SpanTable,
}

/// Resolved `index <name>(region: VoxelRegion) -> <Output> { … }`
/// declaration. Mirrors [`ast::IndexDecl`] field-for-field; future
/// cuts may resolve `output_type_name` into a typed handle once
/// the index-output-type registry lands.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndexIR {
    pub name: String,
    pub region_param_name: String,
    pub output_type_name: String,
    pub storage: ast::IndexStorageShape,
    pub cost_class: ast::IndexCostClass,
    pub rebuild_on: ast::IndexRebuildTrigger,
    /// Raw text of the `build { … }` body — preserved for error
    /// reporting alongside the parsed AST.
    pub build_body: String,
    /// Phase 2b — parsed build body. Phase 4 lowers this to a
    /// build kernel.
    pub build_body_ast: ast::IndexBuildBody,
    pub span: Span,
}

/// Typed handle for [`IndexIR`] entries in
/// [`Compilation::indices`]. Stored in the symbol table by name;
/// downstream code (region-indices cross-validation, Phase 4 build
/// kernel emit) routes through this id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct IndexId(pub u32);

/// Resolved `region_kind <Name> { max_active = N } + region_indices
/// <Name> { …idx kinds… }` pair. Two source-level decls collapse
/// into one IR entry keyed by `name` after pass-2 cross-decl
/// validation.
///
/// `index_kind_names` are kept as `String`s for now — Phase 2 will
/// resolve them into typed [`IndexId`] handles once the `index`
/// decl grammar lands. Pre-Phase-2 the resolver leaves them
/// un-validated (any identifier is accepted) so Phase 1 can ship +
/// be exercised by a smoke probe.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RegionKindIR {
    pub name: String,
    pub max_active: u32,
    pub index_kind_names: Vec<String>,
    pub span: Span,
}

/// Typed handle for [`RegionKindIR`] entries in
/// [`Compilation::region_kinds`]. Stored in the symbol table by
/// name; downstream code (Phase 3+) routes through this id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct VoxelRegionKindId(pub u32);

/// Resolved `table <name>: <ty>[N] = […]` declaration. Mirrors
/// `ast::TableDecl` but with the element type validated against
/// the supported set + a typed `TableId` registered in the symbol
/// table for downstream lookup.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TableIR {
    pub name: String,
    /// Element type — `u32` for the first cut, future cuts may
    /// extend to `i32` / `f32`.
    pub element_ty: IrType,
    /// Declared length (validated == values.len() at resolve).
    pub length: u32,
    /// Element values as `i64` (resolver bounds-checks against
    /// `element_ty` at registration).
    pub values: Vec<i64>,
    pub span: Span,
}

/// Typed handle for a [`TableIR`] entry within
/// [`Compilation::tables`]. Stored in the symbol table by name; the
/// CG lowering looks up via this handle when resolving
/// `tables.<name>(idx)` namespace calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct TableId(pub u32);
