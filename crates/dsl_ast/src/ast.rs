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

impl Default for Span {
    fn default() -> Self {
        Span::dummy()
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

/// A single `import "<path>";` statement at the top of a `.sim` file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Import {
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    pub imports: Vec<Import>,
    /// Canonicalised absolute paths of every `.sim` file that contributed to
    /// this `Program`, including the top-level file itself.  Populated by
    /// `parse_with_imports`; always empty when constructed directly by the
    /// parser (`parse(src)`).
    #[serde(skip)]
    pub imports_resolved: Vec<std::path::PathBuf>,
    pub decls: Vec<Decl>,
    /// Optional singleton `terrain { ... }` block. `None` if the source
    /// does not contain a `terrain` block.
    pub terrain: Option<crate::terrain::TerrainBlock>,
    /// Plan A — optional singleton player-facing descriptor blocks. Each is
    /// parsed onto the Program (not a `Decl`), so the resolver ignores them;
    /// the build helper lowers each to a `&'static str` JSON descriptor.
    pub controls: Option<ControlsDecl>,
    pub render: Option<RenderDecl>,
    pub ui: Option<UiDecl>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Decl {
    Entity(EntityDecl),
    Event(EventDecl),
    EventTag(EventTagDecl),
    Enum(EnumDecl),
    View(ViewDecl),
    /// Plan I — `belief <name>(observer: Agent[, key]) -> T { ... }`
    /// declaration. Mirrors `View` shape (params + return_ty + fold-body
    /// propagation handlers + decay + clamp) plus a `social_merges`
    /// list of `merge from <agent>: <op>` clauses. Resolved into a
    /// `ViewIR` with `kind: ViewKind::Belief` and `social_merges`
    /// populated; lowering at slice I.3 picks the storage hint from
    /// the signature shape.
    Belief(BeliefDecl),
    Query(QueryDecl),
    Physics(PhysicsDecl),
    PhysicsApply(PhysicsApplyDecl),
    Mask(MaskDecl),
    Verb(VerbDecl),
    Scoring(ScoringDecl),
    Invariant(InvariantDecl),
    Probe(ProbeDecl),
    Metric(MetricBlock),
    Config(ConfigDecl),
    SpatialQuery(SpatialQueryDecl),
    Init(InitDecl),
    Debug(DebugDecl),
    /// Top-level `field <name>: <type>` — declares a custom per-agent
    /// SoA column shared across every entity rooted at `Agent`. See
    /// [`AgentFieldDecl`] for the semantics; the resolver passes the
    /// decl through to `dsl_compiler::custom_agent_fields::intern`
    /// before lowering touches `self.<name>` reads or
    /// `agents.set_<name>(...)` writes (Gap plague_city#P-A).
    AgentField(AgentFieldDecl),
    /// Top-level `table <name>: <element_ty>[<N>] = [<v1>, …, <vN>]` —
    /// declares a static lookup table of fixed length and element
    /// type. Lowers to a WGSL `const <name>: array<T, N> = …;`
    /// declaration prepended to every kernel that references it,
    /// readable as `tables.<name>(idx)` from physics / view bodies.
    /// Right home for "world map data the simulation reads but does
    /// not mutate" (room door bitmaps, terrain costs, faction
    /// stances, etc.) — rooms ARE NOT agents.
    Table(TableDecl),
    /// Per spec `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
    /// §6.1.2 — declares a `VoxelRegionKind` tag + its `max_active`
    /// instance cap.
    RegionKind(RegionKindDecl),
    /// Per spec §6.1.2 — maps a `VoxelRegionKind` to its default set
    /// of region-attached indices (Navgrid, Vismap, ...).
    RegionIndices(RegionIndicesDecl),
    /// Per spec §7.2 — `index <name>(region: VoxelRegion) -> Output
    /// { storage, cost_class, rebuild_on, build { … } }`.
    Index(IndexDecl),
}

/// Per-fixture initial-state declaration. Plan E-A6 escape hatch: lets
/// a `.sim` author express "fill agent SoA column `<col>` with this
/// per-slot value before tick 0" without smuggling code into a
/// hand-written `*_runtime/lib.rs`. The build helper consumes this
/// when synthesising the GeneratedRuntime's `try_new` and emits
/// `create_buffer_init` (instead of zero-init `create_buffer`) for
/// every standard or fixture-owned agent column with a matching stmt.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InitDecl {
    pub annotations: Vec<Annotation>,
    /// Flat uniform form: `field: <value>` applied to every agent slot.
    /// Retained for back-compat (`play_probe`, etc.).
    pub stmts: Vec<InitStmt>,
    /// Per-subkind population blocks: `spawn <Subkind> count <N> { … }`.
    /// The compiler assigns contiguous slot ranges (skipping slot 0, the
    /// AgentId NonZeroU32 sentinel) and stamps each seeded agent's
    /// `creature_type` = the subkind ordinal + `alive: 1` (overridable).
    pub spawns: Vec<SpawnBlock>,
    pub span: Span,
}

/// A `spawn <Subkind> count <N> { <field: value,>* }` population block.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpawnBlock {
    /// The `entity X : Agent` subkind name; resolved to its declaration-order
    /// `creature_type` ordinal when stamping seeded agents.
    pub subkind: String,
    pub count: CountExpr,
    /// Per-block field fills (int/f32/slot/pos), applied to the block's
    /// slot range on top of the auto-stamped `creature_type` + `alive`.
    pub fields: Vec<InitStmt>,
    pub span: Span,
}

/// `count` value for a `spawn` block — a literal or a `config.*` reference.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CountExpr {
    /// `count 16` — literal slot count.
    Lit(u32),
    /// `count config.waves.cap` — dotted `config.<block>.<field>` reference,
    /// stored as the joined name (`waves.cap`). Resolved to the runtime
    /// config default at codegen.
    Config(String),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InitStmt {
    /// Bare field name, e.g. `alive` or `cooldown_next_ready_tick`. The
    /// build helper resolves this against `agent_<field>_buf` allocs.
    pub field: String,
    pub expr: InitExpr,
    pub span: Span,
}

/// Tiny init-expression vocabulary. Intentionally minimal; the user
/// directive is "get init state into the DSL, refactor later".
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum InitExpr {
    /// Constant fill: every slot gets this value.
    Const(i64),
    /// Float constant fill: every slot gets this value. Written into an
    /// f32 column as its bit pattern (`(v as f32).to_bits()`); written
    /// into a u32/bool column as `(v as f32) as u32` (truncated).
    Float(f64),
    /// Staggered fill: slot N gets value N (per-agent slot index).
    Slot,
    /// Position builtin for the `pos` field — seeded at `try_new` via
    /// `per_agent_u32` (P5-deterministic per `(seed, slot)`).
    Pos(PosBuiltin),
}

/// Initial-position builtins for the `pos:` init field. Seeded host-side
/// at `try_new` so positions are deterministic for a given `(seed, slot)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum PosBuiltin {
    /// `origin` — every seeded slot at `[0, 0, 0]`.
    Origin,
    /// `scatter(r)` — uniform point in a radius-`r` disc (XY plane).
    Scatter(f64),
    /// `ring(r)` — on the radius-`r` circle (XY plane).
    Ring(f64),
}

/// Per-fixture compiler-debug-mode opt-in. Lets a `.sim` author surface
/// the `LowerOpts.debug` (per-stage / per-kernel timestamp depth) and
/// `LowerOpts.debug_wgsl` (atomic-counter axes) knobs that previously
/// required a hand-written `*_runtime/build.rs`. Mirrors the `init`
/// block precedent: build_helper extracts the parsed values directly
/// from the Program; no Compilation IR slot.
///
/// Today three fixtures opt in (debug_probe, stress_agent_count,
/// stress_cast_density) — every other fixture omits the block and the
/// build_helper falls back to `LowerOpts::default()`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DebugDecl {
    pub annotations: Vec<Annotation>,
    /// Per-stage/per-kernel timestamp depth. `None` = compiler default
    /// (`DebugDepth::Off`).
    pub depth: Option<DebugDepthLit>,
    /// WGSL-side atomic-counter axes. Defaults all-false; setting any
    /// `true` flag in the block flips it on.
    pub wgsl_event_kind_histogram: bool,
    pub wgsl_mask_hit_rate: bool,
    pub wgsl_score_kernel_visits: bool,
    pub span: Span,
}

/// Mirror of `dsl_compiler::cg::lower::DebugDepth`. Kept in `dsl_ast`
/// so the parser can reject invalid level names without depending on
/// the compiler crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DebugDepthLit {
    Off,
    Stage,
    StageMemory,
    Kernel,
    DslMapped,
}

// ---------------------------------------------------------------------------
// Plan A — player-facing descriptor blocks (`controls {}` / `render {}` /
// `ui {}`). Each is a singleton top-level block (like `terrain {}`), parsed
// onto `Program` rather than into a `Decl` variant (the resolver never sees
// them). The build helper extracts them before `resolve` consumes the Program
// and lowers each to a `&'static str` JSON descriptor on the generated
// runtime, matching the `engine_play_api` / `engine_ui` serde shapes.
// ---------------------------------------------------------------------------

/// `controls { key "w" -> ctl.move_y: 1.0  press? … }`. Each binding maps a
/// keyboard key to a write of `value` into the `@runtime` field `<block>.<field>`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ControlsDecl {
    pub bindings: Vec<ControlBinding>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ControlBinding {
    /// Lowercased key name (e.g. `"w"`, `"space"`).
    pub key: String,
    /// `@runtime` block name (e.g. `ctl`).
    pub block: String,
    /// `@runtime` field name within the block (e.g. `move_y`).
    pub field: String,
    pub value: f64,
    /// `true` = fire once on key-down (`BindMode::Press`); `false` = apply
    /// every frame held (`BindMode::Hold`).
    pub press: bool,
    pub span: Span,
}

/// `render { arena_radius <r>  camera …  agent when … { color … }  vfx … }`.
/// Mirrors the `engine_play_api::RenderDescriptor` schema.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RenderDecl {
    pub arena_radius: f64,
    pub camera: CameraDecl,
    pub agents: Vec<AgentVisualDecl>,
    pub vfx: Vec<VfxDecl>,
    pub span: Span,
}

/// `[lo, hi]` range over a named agent column. Two surface forms:
///   * `when <field> in [lo, hi]` — explicit numeric range.
///   * `when creature_type is <Subkind>` — subkind selector; `field` is
///     `"creature_type"`, `subkind` carries the name, and `lo`/`hi` are
///     filled in at JSON-emit time with the subkind's declaration-order
///     ordinal (`lo == hi == ordinal`) — no new descriptor variant.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldRangeDecl {
    pub field: String,
    pub lo: f64,
    pub hi: f64,
    /// `Some(name)` for the `creature_type is <Subkind>` selector; the
    /// compiler resolves `name` → ordinal and stamps `lo == hi == ordinal`.
    /// `None` for the numeric `in [lo, hi]` form.
    #[serde(default)]
    pub subkind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CameraDecl {
    /// `camera follow when <field> in [lo, hi]`.
    Follow(FieldRangeDecl),
    /// `camera observer`.
    Observer,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgentVisualDecl {
    /// `agent when <field> in [lo, hi] { color (r,g,b) }`.
    pub when: FieldRangeDecl,
    pub color: [u8; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum VfxKindDecl {
    /// `ring radius <r> color (r,g,b)`.
    Ring,
    /// `beam_to_nearest when <field> in [lo, hi] color (r,g,b)`.
    BeamToNearest { target: FieldRangeDecl },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VfxDecl {
    /// `vfx on <RuleName> period <n> { … }`.
    pub on_rule: String,
    pub period: u32,
    pub kind: VfxKindDecl,
    pub radius: f64,
    pub color: [u8; 3],
}

/// `ui { hud { … }  menu <name> "title" { … }  screen <name> "title" { … } }`.
/// Mirrors the `engine_ui::UiModel` serde shape.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UiDecl {
    pub hud: Vec<UiWidget>,
    pub screens: Vec<UiScreen>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum UiWidget {
    /// `bar "HP" value hp max hp_max color (220,40,40)`.
    Bar {
        label: String,
        value: String,
        max: String,
        color: [u8; 3],
    },
    /// `text "Lv {level}  Kills {kills}"`.
    Text { template: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UiCard {
    pub label: String,
    /// `card "Bolt +" -> bolt_level` increments host-side counter `bolt_level`.
    pub action_field: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum UiScreen {
    /// `menu <name> "title" { card … }` — modal upgrade menu.
    Menu {
        name: String,
        title: String,
        cards: Vec<UiCard>,
    },
    /// `screen <name> "title" { summary <key> … restart "label" }` — modal end
    /// screen with summary rows + a restart button.
    End {
        name: String,
        title: String,
        summary: Vec<(String, String)>,
        restart_label: String,
    },
}

/// Top-level `field <name>: <type>` declaration — registers a custom
/// per-agent SoA column (Gap plague_city#P-A). The declared column
/// behaves identically to a built-in `AgentFieldId` variant for the
/// duration of the compile: reads via `self.<name>` /
/// `<binder>.<name>` lower to `agent_<name>[idx]`; writes via
/// `agents.set_<name>(target, value)` lower to the standard setter
/// path; the build helper auto-allocates `agent_<name>_buf` sized
/// `agent_count * <elem_bytes>`.
///
/// Today's supported primitives mirror the q8-free arms of
/// `AgentFieldTy`: `u32`, `f32`, `bool`, `vec3` (Gap
/// dungeon_stealth#3-vec3, 2026-05-12). `i16` / `enum_u8` / option
/// types are deferred — fixtures needing those should add the
/// necessary `parse_field_ty` arm + sizing/init coverage in a
/// follow-up. Mirrors the `init` block precedent: the decl is
/// extracted by `dsl_compiler::build_helper` directly from the
/// parsed Program; no Compilation IR slot.
///
/// Storage shape per primitive (set by the auto-allocated
/// `agent_<name>_buf` in build_helper):
///   * `u32` / `f32` / `bool` — `agent_count * 4` bytes (`array<...>`)
///   * `vec3` — `agent_count * 16` bytes (`array<vec3<f32>>`,
///     std430-padded)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgentFieldDecl {
    pub annotations: Vec<Annotation>,
    /// Bare snake_case column name (`infected`, `cult_loyalty`, …).
    /// Validated against the closed built-in set + duplicate
    /// `field` decls in the same Program by the resolver.
    pub name: String,
    /// `"u32"` / `"f32"` / `"bool"` / `"vec3"` — checked by the
    /// compiler-side interner against the `AgentFieldTy` allowlist.
    pub ty_name: String,
    pub span: Span,
}

/// Top-level `table <name>: <ty>[<N>] = [<v1>, …, <vN>]`.
/// Static lookup table. Lowers to a WGSL `const <name>: array<T, N> = …;`
/// declaration that the kernel emit prepends to every kernel body
/// that references the table; the read surface is
/// `tables.<name>(<idx_expr>)` in physics / view rules.
///
/// Element type is restricted to `u32` for the first cut — the
/// surface generalises naturally to `i32`/`f32` when a fixture
/// needs them. `length` is bounded by `u32::MAX` (the WGSL spec's
/// `array<T, N>` length limit is implementation-defined; consumer
/// GPUs commonly allow ≥ 64K which is well past any plausible
/// world-table size).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TableDecl {
    pub annotations: Vec<Annotation>,
    /// Bare snake_case table name (`maze_doors`, `terrain_cost`, …).
    pub name: String,
    /// Element type name as written in source (`"u32"` for the
    /// first-cut surface; future cuts may extend to `i32`/`f32`).
    pub element_ty_name: String,
    /// Declared length (must equal `values.len()` — checked at
    /// resolve time, surfaces a typed error otherwise).
    pub length: u32,
    /// Element values in declaration order. Held as `i64` to fit
    /// `u32` / `i32` / future signed shapes without lossy
    /// conversion; the resolver bounds-checks against the declared
    /// element type at registration time.
    pub values: Vec<i64>,
    pub span: Span,
}

/// Top-level `region_kind <Name> { max_active = N }` — per spec
/// §6.1.2 of `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`.
/// Declares a named `VoxelRegionKind` tag that physics rules use when
/// emitting `VoxelRegionRegistered` events. `max_active` sizes the
/// per-region storage pool at compile time (the registry refuses
/// further registrations once `count(kind) == max_active`).
///
/// Pairs with [`RegionIndicesDecl`] (same kind name) which maps the
/// kind to its default index set.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RegionKindDecl {
    pub annotations: Vec<Annotation>,
    /// PascalCase kind name (`Settlement`, `Building`, `BattleSite`,
    /// `WildernessTile`). Validated for non-duplication by the
    /// resolver; the same name must appear in exactly one
    /// `region_indices` decl.
    pub name: String,
    /// Compile-time upper bound on simultaneously-active regions of
    /// this kind. Drives static pool sizing
    /// (`max_active × per_region_storage`) for the indices the kind
    /// declares.
    pub max_active: u32,
    pub span: Span,
}

/// Top-level `index <name>(region: VoxelRegion) -> <Output> {
/// storage: SHAPE, cost_class: CLASS, rebuild_on: TRIGGER,
/// build { BODY } }` — per spec
/// `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
/// §7.2. Declares a region-attached index with bounded storage
/// and a deterministic build/rebuild pipeline.
///
/// Phase 2a (this decl shape) stores `build_body` as raw source
/// text — Phase 2b parses it into an expression tree, Phase 4
/// lowers it to a build kernel.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndexDecl {
    pub annotations: Vec<Annotation>,
    /// PascalCase index-kind name as it appears in `region_indices`
    /// bodies (`Navgrid`, `Vismap`, `CoverMap`, `SurfaceMesh`). By
    /// convention, the declared identifier is the kind's name in
    /// PascalCase; the spec example writes it lowercase
    /// (`index navgrid(...)`), which we treat as a stylistic choice
    /// — the resolver name-matches case-insensitively against
    /// `region_indices` entries (with a non-strict TODO marker
    /// pending tightening once a real fixture forces the
    /// canonicalisation).
    pub name: String,
    /// The region-parameter identifier — by convention `region`,
    /// surfaces in the build body as `region.chunks`, etc. Stored
    /// so the build-body resolver can bind it correctly when
    /// Phase 2b lands.
    pub region_param_name: String,
    /// Return type, e.g. `Walkable` for Navgrid, `Vismap` for
    /// PVS. Today opaque — Phase 2b registers known output types.
    pub output_type_name: String,
    pub storage: IndexStorageShape,
    pub cost_class: IndexCostClass,
    pub rebuild_on: IndexRebuildTrigger,
    /// Raw text of the `build { ... }` body, captured between
    /// matched braces. Preserved alongside the parsed form
    /// (`build_body_ast`) so error reports can quote the original
    /// source verbatim.
    pub build_body: String,
    /// Phase 2b — parsed expression tree for the build body. Phase
    /// 4 lowers this to a build kernel.
    pub build_body_ast: IndexBuildBody,
    pub span: Span,
}

/// Mini-AST for the `build { ... }` body of an `index` decl. Each
/// statement is either a `let` binding or the trailing return
/// expression. Engine helpers (`engine::column_reduce_xz`, etc.)
/// are first-class call shapes; bindings flow as locals; integer
/// literals carry their own variant.
///
/// **Design decision**: dedicated mini-AST rather than reusing
/// `IrExpr` from view-fold bodies — the build body operates over
/// voxel/region space (with `region.<field>`, `engine::<helper>`,
/// per-cell maps), not per-agent SoA. Reusing the agent-IR would
/// either muddle scoping (`self` doesn't bind here) or require a
/// parallel set of agent-typed wrappers. A purpose-built AST is
/// ~150 LoC and matches the spec's worked examples exactly.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IndexBuildStmt {
    /// `let <name> = <value>;` — bindings flow into the surrounding
    /// scope for subsequent stmts.
    Let { name: String, value: IndexBuildExpr, span: Span },
    /// Trailing expression — the index's output. Per spec §7.2 the
    /// last expression in the body is the return value.
    Return { value: IndexBuildExpr, span: Span },
}

/// Expression shapes for the build body. Closed set per spec
/// §7.2's worked examples; the resolver validates that called
/// engine helpers + referenced identifiers resolve.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum IndexBuildExpr {
    /// `engine::<helper>(<arg>, <arg>, ...)` — built-in engine
    /// primitive (column_reduce_xz, per_cell_classify,
    /// connect_neighbors, etc.).
    EngineCall { name: String, args: Vec<IndexBuildExpr>, span: Span },
    /// Bare identifier: refers to either a `let`-bound local
    /// (within this build body), the region parameter (default
    /// name `region`), or a top-level constant (e.g.
    /// `AGENT_STEP_HEIGHT` — host-side constants the engine
    /// exposes).
    Var { name: String, span: Span },
    /// Member access on an identifier: `region.chunks`. Limited
    /// to single-level paths — nested member access would require
    /// a full expression grammar.
    Member { base: String, field: String, span: Span },
    /// Integer literal — passed to helpers that take counts /
    /// dimensions.
    Int { value: i64, span: Span },
}

/// Top-level container for the build body — a sequence of stmts.
/// Validated at resolve time to end in exactly one
/// [`IndexBuildStmt::Return`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndexBuildBody {
    pub stmts: Vec<IndexBuildStmt>,
}

/// Storage shape per spec §7.2. Each variant carries its own
/// bound arguments; the resolver does compile-time arithmetic on
/// these to enforce the per-kind memory budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum IndexStorageShape {
    /// `per_cell_2d(max_cells = N, bytes_per_cell = M)` — 2D
    /// texture, peak `N × M` bytes per region instance.
    PerCell2d { max_cells: u32, bytes_per_cell: u32 },
    /// `per_cell_3d(max_cells = N, bytes_per_cell = M)` — 3D
    /// texture.
    PerCell3d { max_cells: u32, bytes_per_cell: u32 },
    /// `bitset_pairs(max_cells = N)` — pair-membership bitset,
    /// peak `N² / 8` bytes.
    BitsetPairs { max_cells: u32 },
    /// `mesh_buffer(max_vertices = V, max_indices = I)` —
    /// vertex/index buffer pair.
    MeshBuffer { max_vertices: u32, max_indices: u32 },
    /// `sparse_grid(max_cells = N, bytes_per_cell = M)` —
    /// hash-table-backed sparse storage.
    SparseGrid { max_cells: u32, bytes_per_cell: u32 },
}

/// Cost class per spec §7.2. Drives scheduling priority + budget
/// allocation in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum IndexCostClass {
    Cheap,
    Medium,
    Heavy,
}

/// What triggers a rebuild of this index. Per spec §7.2 today's
/// only documented trigger is `chunk_epoch_advance(region.chunks)`
/// — the region's covering chunks bumped their epoch (voxel write).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum IndexRebuildTrigger {
    /// `chunk_epoch_advance(region.<field>)` — the named field on
    /// the region (typically `chunks`) had its epoch advance.
    ChunkEpochAdvance { region_field: String },
    /// `manual` — rebuild only on explicit `rebuild_index` event.
    /// Future shape; Phase 2a parses but doesn't validate.
    Manual,
}

/// Top-level `region_indices <Name> { Navgrid, Vismap, ... }` — per
/// spec §6.1.2. Maps a `VoxelRegionKind` (declared via
/// [`RegionKindDecl`] with the same `name`) to its default index
/// set. The compiler emits a `static REGION_INDEX_MAP` so the
/// registry can schedule the appropriate index builds at
/// `register_region` time.
///
/// All instances of a given kind get the same indices by design —
/// per-instance variation is deferred (spec §6.1.2 migration note).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RegionIndicesDecl {
    pub annotations: Vec<Annotation>,
    /// PascalCase kind name — must match a declared `region_kind`.
    /// Cross-decl name resolution happens in the resolver.
    pub name: String,
    /// Index-kind names in declaration order (`Navgrid`, `Vismap`,
    /// `CoverMap`, `SurfaceMesh`). Each must resolve to a declared
    /// `index <name>(region: VoxelRegion)` decl from Phase 2;
    /// today's parser accepts any identifier and defers validation
    /// to the resolver (which will fire `ResolveError::UnknownIndexKind`
    /// once `index` decls land).
    pub index_kinds: Vec<String>,
    pub span: Span,
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
    /// `predator_prey: { prey_of: [], preys_on: [] }` — anonymous
    /// struct body with no leading typename. Shape is implicit from the
    /// field's declared type. The resolver passes this through; the
    /// GeneratedRuntime path doesn't interpret entity field values, so
    /// this just needs to parse cleanly.
    AnonStruct(Vec<EntityField>),
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

/// Plan I — AST shape for the new `belief` declaration. Carries the
/// same fields as [`ViewDecl`] plus a list of `social_merges` parsed
/// from `merge from <agent>: <op>` clauses interleaved with the
/// fold-body's `on <Event> {...}` handlers. The body still uses
/// [`ViewBody::Fold`] for the propagation handlers — only the
/// social-merge clauses split out into their own list.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BeliefDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: TypeRef,
    pub body: ViewBody,
    pub social_merges: Vec<SocialMergeClause>,
    pub span: Span,
}

/// Plan I — one `merge from <agent_field>: <op>` clause inside a
/// belief body. Parser-side AST; the resolver maps `source_agent_name`
/// to a `LocalRef` once the event pattern's bindings are in scope.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SocialMergeClause {
    pub pattern: EventPattern,
    pub where_clause: Option<Expr>,
    /// The event-binder identifier naming the giver agent
    /// (e.g. `dead` in `merge from dead: bit_or`).
    pub source_agent_name: String,
    pub op: SocialMergeOpName,
    pub span: Span,
}

/// Plan I — parser-side spelling of [`crate::ir::MergeOp`]. Kept as a
/// separate enum so the AST stays decoupled from the IR; resolver
/// converts via a 1:1 mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SocialMergeOpName {
    BitOr,
    Max,
    Min,
    Replace,
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
    /// Optional `where <predicate>` clause between the pattern and the
    /// body, e.g. `on Killed { by: predator } where predator == by { ... }`.
    /// Same surface as physics-handler `where`. Resolver gates the
    /// fold-write on this when present.
    pub where_clause: Option<Expr>,
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
pub enum ParamType {
    F32,
    I32,
    U32,
    Bool,
    EntityKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ParamDecl {
    pub name: String,
    pub ty: ParamType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicsApplyDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,        // the new concrete-rule name (e.g. "HunterChase")
    pub template: String,    // the parameterised-rule name (e.g. "chase")
    pub args: Vec<ApplyArg>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ApplyArg {
    pub name: String,        // by-name args; positional not supported in v1
    pub value: ApplyArgValue,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ApplyArgValue {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
    EntityKind(String),  // identifier; resolved to a known entity decl in validation
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicsDecl {
    pub annotations: Vec<Annotation>,
    pub name: String,
    pub params: Vec<ParamDecl>,
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
    /// Verb body — a sequence of `emit <Event> { ... }` and/or
    /// `apply_ability <expr> [by <c>] [target <t>]` statements. Each
    /// fires when the verb's action wins the per-agent argmax for the
    /// tick. The verb expander lifts these into the synthesised
    /// `verb_chronicle_<name>` cascade physics handler. See task #138
    /// (Wave 1.7) — `apply_ability` was added as an alternative to
    /// `emit` so the verb body can dispatch through the
    /// `PackedAbilityRegistry` rather than hand-mirror per-effect
    /// chronicle events.
    pub body: Vec<VerbBodyStmt>,
    pub scoring: Option<Expr>,
    pub span: Span,
}

/// One statement inside a `verb`'s body. Either an `emit <Event>{...}`
/// (the legacy / pre-#138 form) or an `apply_ability <expr> [by <c>]
/// [target <t>]` registry-driven dispatch (#138 / Wave 1.7).
///
/// Both variants are lifted into the synthesised `verb_chronicle_<name>`
/// cascade physics handler by the verb expander; lowering to CG happens
/// through the standard physics-handler path.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum VerbBodyStmt {
    Emit(EmitStmt),
    ApplyAbility(ApplyAbilityStmt),
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
/// The row's clauses:
/// * `guard:` — boolean predicate evaluated per (agent, ability). When
///   the guard is false the ability is skipped (does not compete for
///   argmax). Optional; default is `true`.
/// * `score:` — f32 scoring expression. The argmax over every ability
///   whose guard passes is the ability the agent casts this tick.
/// * `target:` — agent-id expression resolving to the cast target for
///   the selected ability. Optional at parse time; Phase 3 may require
///   it when lowering.
/// * `weights:` — utility-table form addend that is summed into the
///   row's score expression. The lowerer composes the utility as
///   `score + weights` (both F32). Optional; design-target fixtures
///   (predator_prey, crowd_navigation, squad_skirmish) use the
///   `base: <const>, weights: <expr>` shape where `base:` doubles as
///   the score field. Closes Gap C from `docs/architecture/gaps_observed.md`
///   (2026-05-11): pre-fix the parser parse-and-discarded `weights:`
///   so personality-weighted scoring contributed nothing to argmax.
///
/// See `docs/spec/engine.md §11`
/// §Architecture.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerAbilityRow {
    pub name: String,
    pub guard: Option<Expr>,
    pub score: Expr,
    pub target: Option<Expr>,
    pub weights: Option<Expr>,
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
    /// Generic predicate form — used by design-target probes whose
    /// assert clause is a free-form quantifier or comparison
    /// (`forall g in groups: ...`, `events.kind_count(...) > 0`, etc.).
    /// Parse-and-discarded today; semantic adoption when probe runner
    /// grows a generic-expr evaluator.
    Raw { expr: Expr, span: Span },
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
    /// `@runtime` annotation flag (Plan G tunable cfg). When `true`, the
    /// field lowers to a per-kernel cfg-uniform field (host-tunable per
    /// tick) instead of a baked-at-compile WGSL `const`. The default
    /// (`false`) preserves the existing const-baked behaviour for every
    /// field that doesn't opt in. Source surface:
    ///   `mask: u32 = 15 @runtime,`
    /// (the `@runtime` token sits AFTER the default value, matching the
    /// trailing-annotation idiom used elsewhere in the grammar).
    pub runtime: bool,
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
    /// `for_each_agent <binder> { <body> }` — the body executes once per alive
    /// agent slot in deterministic linear order (slot 0 → slot agent_cap-1).
    /// `<binder>` is the per-iteration variable bound to the visited slot's
    /// `AgentId`. Reads of `<binder>` and `<binder>.<field>` inside the body
    /// resolve to the candidate-side AgentRef just like the per-pair body
    /// forms (`for x in spatial.…`).
    ///
    /// Source-level shape:
    ///
    /// ```text
    /// for_each_agent a {
    ///   agents.set_mana(a, agents.mana(a) + 1.0)
    /// }
    /// ```
    ///
    /// Lowers to `IrStmt::ForEachAgent` → `CgStmt::ForEachAgentBody` →
    /// a per-thread linear scan in WGSL. The containing per-agent rule
    /// retags its dispatch to `OneShot` so a single thread executes the
    /// scan once per tick (otherwise N threads each scan all N slots,
    /// producing O(N²) writes per tick — pathological).
    ForEachAgent { binder: String, body: Vec<Stmt>, span: Span },
    /// `if <cond> { <body> } else { <body> }` / `match <scrut> { ... }`.
    If { cond: Expr, then_body: Vec<Stmt>, else_body: Option<Vec<Stmt>>, span: Span },
    Match { scrutinee: Expr, arms: Vec<MatchArm>, span: Span },
    /// Self-delta in fold bodies: `self -= 0.1 * e.damage`, `self += 0.3`.
    SelfUpdate { op: String, value: Expr, span: Span },
    /// Plan G G3b/G3c — `self.append(field1: expr1, field2: expr2, ...)` —
    /// struct-payload ring append in a `@per_entity_ring(...)` view fold
    /// body. The field list defines the per-cell struct layout (in
    /// declaration order); types are inferred from the bound exprs at
    /// lowering time. The ring index is allocated via the per-agent
    /// cursor counter; per-field stores write into
    /// `primary[ring_idx * field_count + field_idx]`.
    SelfAppend { fields: Vec<FieldInit>, span: Span },
    /// Bare expression (for fold bodies that set self).
    Expr(Expr),
    /// `beliefs(observer).observe(target) with { field: expr, ... }` — belief
    /// mutation primitive (Plan ToM Task 4). Mutates a single `BeliefState`
    /// cell in `SimState::cold_beliefs` for the observer/target pair.
    BeliefObserve(BeliefObserveStmt),
    /// `apply_ability <ability_expr> [by <caster>] [target <target>]` —
    /// registry-driven dispatch (#125 / #132 / slice δ #161 / slice ε).
    /// Replaces hand-mirrored `emit Damaged{...}` / `emit Healed{...}`
    /// patterns with a single statement that the WGSL emitter expands into
    /// a per-effect-slot dispatch loop reading from `PackedAbilityRegistry`
    /// SoA columns. The ability_expr resolves at runtime to an `AbilityId`
    /// (typically `self.action_ability`); the dispatcher honors per-effect
    /// chance gates and emits the matching chronicle event for each
    /// non-empty effect slot.
    ///
    /// **Slice δ + ε surface (post-`d0bc37fd`):** `caster` defaults to
    /// `self` for PerAgent rules and surfaces a typed
    /// `UnsupportedPhysicsStmt` for PerEvent rules without explicit
    /// `by <caster>`. `target` defaults to the resolved caster
    /// expression (slice-γ self-cast convention). Both can be supplied
    /// explicitly: `apply_ability a by w target v` distinguishes
    /// chronicle actor from target slots.
    ///
    /// Lowering: `IrStmt::ApplyAbility` → `CgStmt::ApplyAbility { ability,
    /// caster, target }` → WGSL dispatcher (#132C+ / `92572af8` / `d0bc37fd`).
    ApplyAbility(ApplyAbilityStmt),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ApplyAbilityStmt {
    /// Expression resolving to the ability_id at runtime. Most call sites
    /// pass `self.action_ability` — the AbilityId field already living on
    /// agent SoA, written by the action-selection scoring kernel.
    ///
    /// **Symbolic ability-name surface (2026-05-12):** when the source
    /// reads `apply_ability Strike by self target target`, the parser
    /// captures the identifier in [`Self::ability_name`] and uses a
    /// placeholder `Int(0)` here. The lowerer resolves the name through
    /// the fixture's registry (sorted-filename → 1-based AbilityId map
    /// in `LowerOpts::ability_names`) and substitutes the resolved id at
    /// the dispatch boundary. Mismatched names surface as a typed
    /// `LowerError::UnknownAbilityName` instead of dispatching whatever
    /// numeric slot happened to be in this field. The numeric surface
    /// (`apply_ability 3 by …`) still parses and lowers unchanged.
    pub ability: Expr,
    /// Symbolic ability-name from the `apply_ability <Name> …` surface
    /// (2026-05-12). `Some(name)` when the parser observed a bare
    /// identifier as the ability operand; `None` for the numeric / general-
    /// expression surface. Resolved to an AbilityId at lower time against
    /// `LowerOpts::ability_names`. Closes the silent-mis-dispatch footgun
    /// that took ~3h to debug for squad_skirmish (`apply_ability 1` was
    /// Daze, not Strike, because the registry sorts filenames
    /// alphabetically).
    pub ability_name: Option<String>,
    /// Optional explicit caster expression (`apply_ability <ability> by
    /// <caster>`). When `None`, lowering defaults to the per-thread
    /// agent (`self`) for PerAgent rules and surfaces a typed error
    /// for PerEvent rules where there's no implicit per-thread agent.
    /// When `Some(expr)`, lowering uses the resolved expression as the
    /// caster slot — typically `e.actor` for PerEvent rules destructuring
    /// an event payload's actor field via the `on EventName { actor: e.actor }`
    /// pattern. Slice δ part 3 (#161).
    pub caster: Option<Expr>,
    /// Optional explicit target expression (`apply_ability <a> [by <c>]
    /// target <t>`). When `None`, the dispatcher writes the caster
    /// into the target chronicle slot (slice-γ self-cast convention —
    /// preserves prior behavior). When `Some(expr)`, lowering uses the
    /// expression as a distinct target slot, so chronicle records
    /// distinguish actor from target. Slice ε part 1.
    pub target: Option<Expr>,
    pub span: Span,
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
    /// `{ let a = e1; let b = e2; final_expr }` — sequential
    /// let-binding block. Used by @lazy view bodies and match arms
    /// that need an intermediate name. Each binding extends the
    /// scope visible to subsequent bindings + the final expression.
    /// Lowering inlines bindings substitution-style; the resolver
    /// simply binds each name into the local scope.
    Block { bindings: Vec<(String, Expr)>, expr: Box<Expr> },
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
    /// Bitwise OR (`|`). Unsigned-int only at the CG layer; the type
    /// checker rejects f32/i32/bool/Vec3 operands. Used for bitset
    /// merges (e.g. `recipes_known | RECIPE_BREAD`).
    BitOr,
    /// Bitwise XOR (`^`).
    BitXor,
    /// Bitwise AND (`&`). Used for bitset membership tests
    /// (e.g. `(recipes_known & RECIPE_BREAD) != 0`).
    BitAnd,
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
    /// Arithmetic mean. Parser accepts `mean(<binder> in <iter>)` and
    /// `mean(<expr> for <binder> in <iter>)`. Lowering currently treats
    /// Mean the same as Sum (no /N divide) — the design-target fixtures
    /// that use it (`crowd_navigation.sim`) ride a future semantic-
    /// adoption slice; today the symbol just has to round-trip.
    Mean,
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

/// A single `.ability` source file. Wave 1.0 only held `ability` decls;
/// Wave 1.1 added `passive` blocks; Wave 1.2 added `template` blocks;
/// Wave 1.3 adds `structure` blocks (body captured opaquely — per
/// spec §12 the body holds 5 statement kinds whose GPU rasterization
/// + StructureRegistry wiring (§12.2) is Wave 2+ work).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AbilityFile {
    pub abilities: Vec<AbilityDecl>,
    /// Wave 1.1: top-level `passive <Name> { ... }` blocks per spec §5.
    /// The parser populates this in source order; lowering of passives
    /// is deferred to Wave 2+ (`dsl_compiler::ability_lower` errors with
    /// `PassiveBlockNotImplemented` if this vec is non-empty).
    pub passives: Vec<PassiveDecl>,
    /// Wave 1.2: top-level `template <Name>(<params>) { ... }` blocks
    /// per spec §11. Parser populates in source order. Template
    /// expansion (parameter substitution into `$ident` references in
    /// the body) lands at Wave 2+ — this slice only stores the parsed
    /// surface. Lowering of a non-empty `templates` vec surfaces
    /// `LowerError::TemplateBlockNotImplemented` so authors don't run
    /// with silently-dropped template definitions.
    pub templates: Vec<TemplateDecl>,
    /// Wave 1.3: top-level `structure <Name>(<params>) { ... }` blocks
    /// per spec §12. Parser populates in source order. The body is
    /// captured OPAQUELY (verbatim source slice) — per-statement
    /// parsing of the 5 body kinds (`place` / `harvest` / `transform`
    /// / `include` / `if`) plus the optional headers (`bounds:` /
    /// `origin:` / `rotatable` / `symmetry:`) lands when voxel
    /// storage + rasterization (spec §12.2 GPU work) exists. Lowering
    /// of a non-empty `structures` vec surfaces
    /// `LowerError::StructureBlockNotImplemented` so authors don't run
    /// with silently-dropped structure definitions.
    pub structures: Vec<StructureDecl>,
}

/// A parsed `ability <Name> { headers... effects... }` block.
///
/// `headers` is the list of header properties in source order. Duplicate
/// header keys are rejected at parse time. `effects` is the list of bare
/// effect statements in source order.
///
/// Wave 1.4 added two optional body-block fields:
/// * `deliver` — a `deliver <method> { params } { body }` block; captured
///   opaquely (verbatim source slice) because the inner delivery-method
///   params + on_hit/on_arrival/on_tick hooks belong to spec §9 and are
///   wave-2+ work. Storing it here lets lowering surface a clean
///   "deliver block not implemented" error and lets downstream tooling
///   round-trip the source.
/// * `morph` — a `morph { effects } into <Other>` block.
///
/// Spec §4.4 / §23.1 says deliver and bare `effects` are mutually
/// exclusive, but a portion of the LoL corpus (e.g. Ahri.SpiritRush)
/// pairs `deliver projectile { … } { on_hit { … } }` with a trailing
/// `dash to_target`. To maximise the corpus parse rate Wave 1.4 admits
/// both at parse time; the lowering layer (`dsl_compiler`) is the one
/// that enforces the mutual-exclusion via `LowerError::MixedBody`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AbilityDecl {
    pub name: String,
    pub headers: Vec<AbilityHeader>,
    pub effects: Vec<EffectStmt>,
    /// Wave 1.4: optional `deliver <method> { params } { body }` block.
    /// `None` for ability bodies that use only bare effects (the Wave 1
    /// corpus). See `DeliverBlock` for the opaque-capture rationale.
    pub deliver: Option<DeliverBlock>,
    /// Wave 1.4: optional `morph { effects } into <Other>` block. `None`
    /// for the LoL corpus (no morph usage); ships for spec coverage.
    pub morph:   Option<MorphBlock>,
    /// Wave 1.2: optional `: TemplateName(arg1, arg2, ...)` clause sitting
    /// between the ability name and the `{` body brace. Per spec §11 this
    /// instantiates a template, supplying positional args; the body block
    /// can still hold headers / effects that the template lowering layer
    /// (Wave 2+) merges with the template's expanded effects. `None` for
    /// the Wave 1 corpus (no instantiation usage). Lowering of a
    /// `Some(_)` value surfaces `TemplateInstantiationNotImplemented`.
    pub instantiates: Option<TemplateInstantiation>,
    /// Plan G (2026-05-09): optional cast/effect program — a sequence
    /// of `cast { … } effect { … }` blocks giving the ability
    /// deferred-resolution semantics with telegraphs, interrupts,
    /// and threat zones. `None` = legacy single-step ability that
    /// resolves immediately from the `effects` field above. `Some`
    /// = the ability uses the new program shape; `effects` is
    /// empty in that case (the program's `Effects(...)` step
    /// holds the effect statements instead).
    ///
    /// Backwards compat: every existing `.ability` file parses
    /// with `program: None`. Lowering inspects `program.is_some()`
    /// to pick the deferred path vs the legacy immediate path.
    pub program: Option<Vec<AbilityProgramStep>>,
    pub span: Span,
}

/// `deliver <method> { params } { body }` — projectile / channel / zone /
/// chain / tether / trap delivery wrapper (spec §9, six methods).
///
/// Wave 1.4 captures the entire deliver invocation as a verbatim source
/// slice (`raw`) — the inner `{ key: val, … }` params block and the
/// `{ on_hit { … } | on_arrival { … } | on_tick { … } | … }` body
/// block both belong to spec §9 hook grammar (Wave 2+). Storing the
/// opaque slice lets:
///   1. The parser succeed on the 110+ LoL files that use `deliver`.
///   2. Lowering surface a clean
///      `LowerError::DeliverBlockNotImplemented` instead of a parse
///      error.
///   3. Downstream tooling (formatter, schema-hash, IR diff) recover
///      the original text without re-traversing the source.
///
/// `method` is the delivery-method ident immediately following
/// `deliver` (`projectile`, `channel`, `zone`, `chain`, `tether`,
/// `trap`). It's pulled out of the slice so callers can reason about
/// the delivery shape without re-parsing `raw`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeliverBlock {
    /// Delivery method ident (`projectile`, `channel`, `zone`, `chain`,
    /// `tether`, `trap`). Verbatim source spelling; spec validation is
    /// lowering's job.
    pub method: String,
    /// Verbatim source slice from the `deliver` keyword to (and
    /// including) the closing `}` of the body block. Multi-line; trims
    /// no whitespace. Retained for backward compatibility — the
    /// structured `hooks` field is preferred for new lowering paths.
    pub raw:    String,
    /// Wave 2 piece 5/6 follow-up (#139): structured parse of the body
    /// block's hook stanzas. The body is a sequence of
    /// `<hook_ident> { <effect stmts> }` entries (e.g. `on_hit { … }`,
    /// `on_tick { … }`, `on_arrival { … }`). Each entry's effect
    /// statements use the regular `EffectStmt` grammar so all the
    /// Wave 1.5 modifier slots compose inside hooks the same way they
    /// do at the ability's top level. Empty when the body has no
    /// recognizable hook stanzas.
    pub hooks:  Vec<DeliverHook>,
    pub span:   Span,
}

/// One `<hook_ident> { … }` entry inside a `deliver` body block.
/// `kind` is the verbatim hook identifier (e.g. `"on_hit"`); the
/// engine validates the vocabulary at lowering time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeliverHook {
    pub kind:    String,
    pub effects: Vec<EffectStmt>,
    pub span:    Span,
}

/// `morph { effects } into <Other>` — temporary form-swap (spec §4.4
/// reserved keyword + §6.4.4 body-item grammar).
///
/// Inner `effects` re-uses the regular `EffectStmt` grammar (parser
/// recursion is allowed). `into` carries the name of the morphed-into
/// ability — semantic resolution against the `AbilityRegistry` is
/// lowering's job.
///
/// Wave 1.4 ships this surface even though the LoL corpus has no
/// `morph` usage today: the spec calls it out as one of the three
/// body-block forms, and shipping it now keeps the AST forward-stable
/// when authors begin using it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MorphBlock {
    pub effects: Vec<EffectStmt>,
    /// Ident of the ability to morph into (e.g. `Heatseeker`). Resolution
    /// against the registry happens in lowering.
    pub into:    String,
    pub span:    Span,
}

/// One header property line inside an `ability` block. Wave 1.0 covered
/// the five core properties (`target`, `range`, `cooldown`, `cast`,
/// `hint`); Wave 1.1 added `cost`, `charges`, `recharge`, `toggle` per
/// spec §4.2. Still deferred: `recast` / `morph` / `form` /
/// `require_skill` / `require_tool` / `zone_tag` / `unstoppable`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AbilityHeader {
    Target(TargetMode),
    Range(f32),
    /// Cooldown duration. Plan G (2026-05-09) added the optional
    /// `@ phase` qualifier — when does the cooldown timer start?
    /// Defaults to `CooldownPhase::Cast` (cast initiation) so
    /// spam-cancel can't bypass the cooldown gate. `None` (the
    /// pre-Plan-G shape) means the parser didn't see an explicit
    /// `@` qualifier; lowering treats `None` and `Some(Cast)`
    /// identically.
    Cooldown(Duration, Option<CooldownPhase>),
    Cast(Duration),
    Hint(HintName),
    /// Wave 1.1: resource cost (mana / stamina / hp / gold). Spec §4.2
    /// describes it as `cost: int` with the "mana/resource" predicate.
    /// We accept either a bare number (default resource = mana, matching
    /// the existing LoL hero corpus) or `cost: <amount> <resource>` /
    /// `cost: <amount>% <resource>` for the full form. Item costs are
    /// reserved for Wave 4.
    Cost(CostSpec),
    /// Wave 1.1: max stored charges (per-agent SoA in the future). Spec
    /// §4.2 lists `charges: int`.
    Charges(u32),
    /// Wave 1.1: per-charge regen time. Spec §4.2 lists `recharge:
    /// duration` separately from `cooldown:`.
    Recharge(Duration),
    /// Wave 1.1: marker (no value) — declares this ability as a toggle.
    /// Spec §4.2 lists `toggle / toggle_cost` (flag / f32); the
    /// `toggle_cost` companion field is deferred to Wave 2+ (its
    /// per-tick drain semantics need engine-side accounting we don't
    /// have yet).
    Toggle,
    /// Wave 1.4: `recast: <int|dur>` — multi-stage cast state (spec
    /// §4.2: "recast / recast_window | int / dur | multi-stage cast
    /// state"). The corpus uses bare ints (`recast: 1`, `recast: 3`)
    /// to mean "max consecutive recasts", and durations (`recast: 4s`)
    /// to mean "recast cooldown". Both forms are accepted; lowering
    /// (Wave 2+) interprets per the full spec semantics.
    Recast(RecastValue),
    /// Wave 1.4: `recast_window: <duration>` — how long after the
    /// initial cast the recast window stays open before the recast
    /// state is dropped. Spec §4.2 fixes the type to duration only.
    RecastWindow(Duration),
}

/// `recast:` value — int (count of allowed recasts) or duration
/// (recast cooldown). Spec §4.2 lists `recast: int / dur`. The corpus
/// has both forms — `recast: 1` (int) on Aatrox.TheDarkinBlade and
/// `recast: 4s` would be a valid duration form. Wave 1.4 stores the
/// shape parsed; lowering (Wave 2+) owns the semantic distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RecastValue {
    /// `recast: N` — integer count.
    Count(u32),
    /// `recast: <duration>` — recast cooldown.
    Duration(Duration),
}

/// Resource cost expression for the `cost:` header.
///
/// Spec §4.2 lists `cost: int` as a "mana/resource gate in mask
/// predicate". This struct generalises the surface to four resources
/// with either a flat amount or a percent of max.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CostSpec {
    pub resource: CostResource,
    pub amount:   CostAmount,
    pub span:     Span,
}

/// The resource a `cost:` header debits from.
///
/// Per spec §4.2 the listed resources are mana / stamina / hp / gold.
/// Item costs (`consume <item> <n>`) live in their own effect verb and
/// are not exposed via `cost:`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CostResource {
    Mana,
    Stamina,
    Hp,
    Gold,
}

/// Cost magnitude — either a flat scalar or a percent of the resource's
/// max. The percent form preserves the percentage-scalar convention the
/// Wave 1.0 parser already uses for `EffectArg::Percent` (e.g. `25%`
/// stores `25.0`, NOT `0.25`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum CostAmount {
    Flat(f32),
    /// Percentage scalar, matching `EffectArg::Percent`. `25% mana`
    /// stores `25.0`.
    PercentOfMax(f32),
}

/// One effect statement: a verb name plus zero-or-more positional args
/// plus the nine optional modifier slots described in spec §6.1.
///
/// Wave 1.0 captured only the leading positional args. Wave 1.5 (this
/// version) lifts the nine modifier slots into typed AST fields:
///
/// 1. `area`      — `in <shape>(args…)`         (spec §8 shape vocab)
/// 2. `tags`      — `[TAG: value]` (multiple)   (spec §6.1)
/// 3. `duration`  — `for <duration>`            (spec §6.1)
/// 4. `condition` — `when <cond> [else <cond>]` (spec §10, opaque)
/// 5. `chance`    — `chance N%`                 (spec §6.1)
/// 6. `stacking`  — `stacking refresh|stack|extend`
/// 7. `scalings`  — `+ N% stat_ref` (multiple)
/// 8. `lifetime`  — `until_caster_dies` / `damageable_hp(N)` (voxel)
/// 9. `nested`    — `{ … }` block of follow-up effects
///
/// All slots are optional so terse verbs (`damage 50`) stay terse. The
/// modifier order on the source line is NOT semantically meaningful at
/// parse time — every keyword maps to a distinct slot. Lowering (Wave
/// 2+) is responsible for actually consuming each slot; until then
/// `dsl_compiler::ability_lower::lower_effect_stmt` returns
/// `LowerError::ModifierNotImplemented` for each populated slot.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectStmt {
    pub verb: String,
    pub args: Vec<EffectArg>,
    pub span: Span,
    /// `in <shape>(args)` — area expansion (spec §6.1 slot 2).
    pub area: Option<EffectArea>,
    /// `[TAG: value]` power tags. Multiple allowed; the LoL corpus uses
    /// entries like `[FIRE: 60]`, `[CROWD_CONTROL: 30]`.
    pub tags: Vec<EffectTag>,
    /// `for <duration>` — how long the effect persists.
    pub duration: Option<EffectDuration>,
    /// `when <cond> [else <otherwise>]` — conditional gate. The
    /// condition language (spec §10, ~80 atoms) is owned by the
    /// expression parser; Wave 1.5 stores opaque source slices.
    pub condition: Option<EffectCondition>,
    /// `chance N%` — Bernoulli gate. Stored as 0.0..=1.0.
    pub chance: Option<EffectChance>,
    /// `stacking refresh|stack|extend` — repeat-application policy.
    pub stacking: Option<StackingMode>,
    /// `+ N% stat_ref` — additive scaling terms. Multiple allowed
    /// (e.g. damage scales with both AP and AD).
    pub scalings: Vec<EffectScaling>,
    /// `until_caster_dies` / `damageable_hp(N)` — alternative to
    /// `for <duration>` for effects bound to caster state or a damage
    /// budget.
    pub lifetime: Option<EffectLifetime>,
    /// `{ … }` — nested follow-up effects (verb opts in).
    pub nested: Vec<EffectStmt>,
}

/// `in <shape>` modifier — area expansion. The shape vocabulary is the
/// 12 primitives listed in spec §8 (circle, sphere, cone, line, etc.)
/// — Wave 1.5 stores them as a flat (name, args) tuple; the lowering
/// pass will validate the shape name + arity later.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectArea {
    /// Shape primitive name verbatim from source (e.g. "circle",
    /// "cone", "sphere"). Lowering (later wave) validates against the
    /// 12-shape vocab in spec §8.
    pub shape: String,
    /// Shape parameters in source order (radius, angle, length…).
    /// Arity is shape-specific; lowering enforces it.
    pub args: Vec<f32>,
    pub span: Span,
}

/// `[TAG: value]` power tag. Multiple tags allowed per effect.
/// `value` is `f32` per spec §6.1 (corpus has both int and float forms).
/// Tag-name vocabulary lookup against `AbilityTag` is lowering's job
/// (Wave 2); Wave 1.5 stores the verbatim source spelling.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectTag {
    /// Verbatim source spelling — UPPERCASE by convention.
    pub name: String,
    pub value: f32,
    pub span: Span,
}

/// `for <duration>` — how long the effect persists.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EffectDuration {
    pub duration: Duration,
    pub span: Span,
}

/// `when <cond> [else <otherwise>]` — conditional effect application.
/// Wave 1.5 stores condition expressions as opaque source slices; the
/// condition language itself (spec §10, ~80 atoms) is owned by the
/// expression parser, not this slot. Lowering re-parses against the
/// condition grammar in a later wave.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectCondition {
    /// Verbatim source slice for the `when` clause (whitespace-trimmed).
    pub when_cond: String,
    /// Verbatim source slice for the optional `else` clause.
    pub else_cond: Option<String>,
    pub span: Span,
}

/// `chance N%` — Bernoulli gate. `25%` source becomes
/// `EffectChance { p: 0.25 }`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EffectChance {
    /// Probability in `0.0..=1.0`.
    pub p: f32,
    pub span: Span,
}

/// `stacking <mode>` — repeat-application policy (spec §6.1 slot 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StackingMode {
    /// `stacking refresh` — re-application resets duration to full.
    Refresh,
    /// `stacking stack` — each application increments a counter.
    Stack,
    /// `stacking extend` — new duration = remaining + new.
    Extend,
}

/// `+ N% stat_ref` — additive scaling (spec §6.1 slot 8). Multiple
/// allowed (e.g. `damage 50 + 30% AP + 20% AD`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectScaling {
    /// 50.0 means +50%.
    pub percent: f32,
    /// Verbatim source token, e.g. "AP", "AD", "self.hp". Stat-ref
    /// resolution is lowering's job in a later wave.
    pub stat_ref: String,
    pub span: Span,
}

/// Effect lifetime modifier (spec §6.1 slot 9) — alternative to `for
/// <duration>` for effects bound to caster state or a damage budget.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum EffectLifetime {
    /// `until_caster_dies` — effect persists until the caster dies.
    UntilCasterDies { span: Span },
    /// `damageable_hp(N)` — voxel-style damage budget; effect dies
    /// when this HP pool is depleted.
    DamageableHp { hp: f32, span: Span },
    /// `break_on_damage` — effect ends when caster takes damage.
    /// Used by stealth-style abilities (LoL: Akali/Elise/MonkeyKing
    /// stealth-for-3s-break_on_damage). Spec drift: not in spec §6.1
    /// originally; surface added 2026-05-04 to close the LoL-corpus
    /// long tail (#85 follow-up — last 25/172 files used this token).
    BreakOnDamage { span: Span },
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

/// Hint enum used for scoring metadata only (spec §4.2). Seven variants
/// matching the existing `.ability` corpus + the LoL-corpus `buff` token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HintName {
    Damage,
    Defense,
    CrowdControl,
    Utility,
    Heal,
    Economic,
    /// LoL corpus uses `hint: buff` for ally-empowering abilities (e.g.
    /// haste, damage amp). Lowering routes Buff → AbilityHint::Utility
    /// today; if the engine grows a dedicated `Buff` hint variant
    /// (schema-hash bump), update both arms.
    Buff,
}

/// A normalized duration in milliseconds. The lexer accepts `5s`,
/// `300ms`, `1.5s`, and bare `5000` (interpreted as ms per the spec
/// §6 lowering note that durations on the GPU side are tick-quantized
/// from millis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Duration {
    pub millis: u32,
}

// ---------------------------------------------------------------------------
// Wave 1.1: passive top-level form (spec §5).
// ---------------------------------------------------------------------------

/// A parsed `passive <Name> { headers... effects... }` block. The body
/// shape mirrors `AbilityDecl` — passives reuse `EffectStmt` (spec §5
/// states the body is a regular effect block). The four trigger event
/// kinds in §5.2 are kept as a string for now (24+ values; a finite enum
/// would lock us in too early — Wave 2 lowering will catalog them).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PassiveDecl {
    pub name:    String,
    pub headers: Vec<PassiveHeader>,
    pub effects: Vec<EffectStmt>,
    pub span:    Span,
}

/// One header property line inside a `passive` block.
///
/// Spec §5 lists `trigger:` (event kind), an optional `cooldown:`
/// (between successive trigger fires), and an optional `range:`
/// (modifier on the trigger predicate). `hint:` reuses ability §4.2
/// shape. Spec §5.2 also mentions optional modifiers in parens
/// (`by:`, type filters); those are not parsed in Wave 1.1 — they fall
/// through `skip_modifier_tail` like the effect-line modifiers do.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PassiveHeader {
    /// `trigger: on_damage_taken | on_kill | on_ability_use | periodic
    /// | on_voxel_placed | …` (24 kinds in §5.2 plus the `periodic`
    /// special-case). Stored as a string until lowering catalogs them.
    Trigger(String),
    /// `cooldown:` between successive trigger fires. Optional —
    /// triggerless passives have no cooldown.
    Cooldown(Duration),
    /// `range:` filter on the trigger predicate (per §5.3 "by-agent /
    /// range filters compile to mask predicate clauses"). Optional.
    Range(f32),
    /// Tag/category — same shape and semantics as ability `hint:` per
    /// spec §4.2.
    Hint(HintName),
}

// ---------------------------------------------------------------------------
// Wave 1.2: template top-level form (spec §11) + ability instantiation.
// ---------------------------------------------------------------------------

/// A parsed `template <Name>(<params>) { <effects> }` block per spec §11.
///
/// Body re-uses the existing `EffectStmt` vocabulary (Wave 1.5 modifier
/// slots included). Parameter substitution (`$ident` references in the
/// body) happens at expansion time, not parse time. This slice stores
/// effects with `$ident` tokens parsed as Ident-shaped EffectArgs;
/// expansion (Wave 2+) replaces them with the bound `TemplateArg`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemplateDecl {
    pub name:    String,
    pub params:  Vec<TemplateParam>,
    pub effects: Vec<EffectStmt>,
    pub span:    Span,
}

/// One `(<param>, <param>, ...)` entry in a template's parameter list.
///
/// Spec §11.1 grammar:
/// ```text
/// template_param = IDENT [ ":" type_name [ "=" default_val ] ] ;
/// ```
/// The type and default are independently optional; an unbound,
/// non-default param is required at instantiation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemplateParam {
    pub name:    String,
    pub ty:      Option<TemplateParamTy>,
    pub default: Option<TemplateArg>,
    pub span:    Span,
}

/// Parameter type-tag from spec §11.1 `type_name` — the closed set of
/// names the spec admits at the type slot of a template parameter.
///
/// `Material` and `Structure` reference enum / decl shapes spec'd
/// elsewhere (`.sim` materials, .ability §12 structures). The set is
/// intentionally narrow — anything outside this list is a parse error,
/// to keep the surface tight as more decl kinds land.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum TemplateParamTy {
    Int,
    Float,
    Bool,
    Material,
    Structure,
}

/// One positional argument supplied to a template instantiation, or one
/// default value attached to a template parameter.
///
/// Stored shape-tagged because the grammar (§11.1) admits four literal
/// forms; semantic resolution (e.g. matching a `Material` ident against
/// the `.sim` material catalog, or coercing `Number(3)` to a typed
/// `Int`) is template-expansion's job (Wave 2+).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum TemplateArg {
    /// Numeric literal — int or float; lowering will coerce to template
    /// param type.
    Number(f32),
    /// Bare identifier — usually a Material name (`fire`, `frost`) or a
    /// Structure. Stored verbatim; semantic resolution at
    /// template-expansion time.
    Ident(String),
    /// String literal `"…"`.
    String(String),
    /// Boolean literal `true` / `false`.
    Bool(bool),
}

/// `: TemplateName(arg1, arg2, ...)` clause attached to an ability
/// declaration per spec §11. Stored on the `AbilityDecl` rather than
/// inlined into the body so callers can detect "this ability is a
/// template instance" without walking the effect list.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemplateInstantiation {
    /// Name of the template being instantiated.
    pub name: String,
    /// Positional args; arity / type checking lives at expansion time.
    pub args: Vec<TemplateArg>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Wave 1.3: structure top-level form (spec §12) — voxel blueprint.
// ---------------------------------------------------------------------------

/// `structure <Name>(<params>) { <body> }` — voxel-template top-level
/// per spec §12. Body holds 5 statement types (place / harvest /
/// transform / include / if) plus optional headers (bounds: / origin:
/// / rotatable / symmetry:). Wave 1.3 captures the body OPAQUELY as a
/// verbatim source slice — per-statement parsing is later work
/// (lowering needs voxel storage + rasterization, all spec §12.2 GPU
/// work). Parameters reuse `TemplateParam` exactly (same int / float
/// / bool / Material / Structure typed grammar from Wave 1.2).
///
/// The opaque-capture pattern mirrors Wave 1.4's `DeliverBlock` — it
/// lets the parser succeed on every well-formed structure definition
/// in author-written `.ability` files while reserving a clean
/// diagnostic (`LowerError::StructureBlockNotImplemented`) for the
/// lowering layer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StructureDecl {
    pub name:     String,
    /// Parameter list. Reuses `TemplateParam` exactly — same int /
    /// float / bool / Material / Structure typed grammar from Wave
    /// 1.2. Empty for `structure Empty() { … }` and for
    /// `structure Wall { … }` (parens omitted entirely — accepted as
    /// shorthand for `()`).
    pub params:   Vec<TemplateParam>,
    /// Verbatim text between the outer `{` and the matching `}` of
    /// the structure body. Excludes the braces themselves. Multi-line;
    /// no whitespace trimming. Per-statement parsing is deferred to
    /// Wave 2+ — until then this slice is opaque to all consumers.
    pub body_raw: String,
    pub span:     Span,
}

// =============================================================
// Plan G — Cast state, interrupts, threat zones (2026-05-09)
// =============================================================
//
// AST types for the new `cast { duration; telegraph; interrupts }`
// block form, the `interrupts:` set syntax, and the `cooldown @
// phase` qualifier. The parser additions land in a follow-up
// commit; this commit defines the types so downstream lowering
// (Plan G phases G2+) can be sketched against a stable shape.
//
// See `docs/superpowers/plans/2026-05-09-cast-state-and-threat-zones.md`.

/// One step of an ability's deferred-resolution program. New
/// `.ability` files can author a sequence:
///
/// ```text
/// ability Firebolt {
///     cast { duration: 3t; ... }    // → AbilityProgramStep::Cast(...)
///     effect { damage 25 in line(...) }   // → AbilityProgramStep::Effects(...)
/// }
/// ```
///
/// Multi-stage abilities chain `cast` / `effect` / `cast` / `effect`
/// for telegraph-then-impact-then-recovery sequences. Backwards
/// compatible: an ability with no `cast` block parses with NO
/// program steps; the legacy `effects: Vec<EffectStmt>` field on
/// `AbilityDecl` still drives lowering for that case.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AbilityProgramStep {
    /// A windup phase. Sets the agent's busy state for `duration`
    /// ticks; populates the threat-zone view if `telegraph` is set;
    /// listens for interrupt sources per the `interrupts` set.
    Cast(CastSpec),
    /// A resolution phase — the same effect-statement vocabulary
    /// the legacy `effects:` field uses. Fires immediately when the
    /// preceding `cast` (if any) elapses without interruption.
    Effects(Vec<EffectStmt>),
}

/// Body of a `cast { … }` block.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CastSpec {
    /// `duration: 3t` — number of fixed-step sim ticks the cast
    /// occupies before resolving (or being interrupted). `1t` =
    /// 100 ms at the standard 10 Hz tick rate.
    pub duration_ticks: u32,
    /// `telegraph: line(self.pos, target.pos, width: 2)` — optional
    /// shape projected into the threats view during the cast. `None`
    /// = silent cast (no AOE warning visible to AI or viewer).
    /// Captured opaquely as a verbatim source slice for now;
    /// per-shape parsing lands in Plan G's lowering phase (G3) when
    /// the threats view fold needs the shape vocabulary.
    pub telegraph: Option<String>,
    /// `interrupts: standard | { … } | none + set ops`. See
    /// [`InterruptSet`] for the vocabulary.
    pub interrupts: InterruptSet,
    pub span: Span,
}

/// What can interrupt an in-progress cast or busy activity.
///
/// Authored at the call-site (per cast block) using a small set
/// algebra:
///
/// - `interrupts: standard`           → [`InterruptSet::Standard`]
/// - `interrupts: { damage, stun }`   → [`InterruptSet::Subset`]
/// - `interrupts: standard + { mvmt }`→ [`InterruptSet::StandardPlus`]
/// - `interrupts: standard - { dmg }` → [`InterruptSet::StandardMinus`]
/// - `interrupts: none`               → [`InterruptSet::None`]
///
/// `standard` is the engine-wide named default declared once via
/// `set standard = { damage, stun, caster_died, target_died }`
/// (see [`StandardSetDecl`]). Lowering resolves named sets at
/// compile time, so the kernel sees a fixed bitmask per ability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum InterruptSet {
    /// Resolves to whatever `set standard = { … }` declared.
    Standard,
    /// Explicit subset — exactly these kinds, nothing else.
    Subset(Vec<InterruptKind>),
    /// `standard + { … }` — standard plus extras.
    StandardPlus(Vec<InterruptKind>),
    /// `standard - { … }` — standard minus exclusions.
    StandardMinus(Vec<InterruptKind>),
    /// `interrupts: none` — uninterruptible (Bind Soul, etc.).
    None,
}

/// Sources that can interrupt a cast / busy activity. The named
/// `standard` set is `{ Damage, Stun, CasterDied, TargetDied }`
/// per design 2026-05-09. `Movement` is opt-in only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum InterruptKind {
    /// Caster took a `Damaged` event this tick.
    Damage,
    /// Caster received an `EffectStunApplied` event this tick.
    Stun,
    /// Caster died (`Defeated` event).
    CasterDied,
    /// The cast's target died — no-op for AOE / self-cast / position-target casts.
    TargetDied,
    /// Caster moved (pre-tick / post-tick `agent_pos` differ).
    Movement,
}

/// Phase qualifier on the `cooldown:` header — when does the
/// cooldown start counting?
///
/// ```text
/// cooldown: 5s              # defaults to `@ cast`
/// cooldown: 5s @ resolve    # cooldown begins when the cast lands
/// cooldown: 5s @ interrupt  # only consumed on interrupt (rare)
/// ```
///
/// Default `@ cast` prevents spam-canceling from bypassing the
/// cooldown gate. `@ resolve` and `@ interrupt` are parsed +
/// stored but only `@ cast` is implemented in Plan G's MVP; the
/// other two surface a "phase qualifier not yet supported" error
/// at lowering until later slices wire them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CooldownPhase {
    /// Cooldown starts when the cast / activity begins.
    Cast,
    /// Cooldown starts when the cast resolves successfully.
    Resolve,
    /// Cooldown only consumed on interruption.
    Interrupt,
}

/// Engine-wide named-set declaration. One `set standard = { … }`
/// in the engine baseline declares the contents of `standard` for
/// every ability that references it via [`InterruptSet::Standard`]
/// / [`InterruptSet::StandardPlus`] / [`InterruptSet::StandardMinus`].
///
/// Per-fixture re-declaration is allowed but takes effect only
/// within that fixture's compilation unit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StandardSetDecl {
    pub members: Vec<InterruptKind>,
    pub span: Span,
}
