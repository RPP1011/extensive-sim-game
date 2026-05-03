//! `DataHandle` — typed references to simulation state.
//!
//! Every read or write in the Compute-Graph IR refers to a
//! `DataHandle` rather than a string. The handle captures *what kind*
//! of data this is and *which* instance, so naming becomes an
//! emit-time concern rather than an IR-level one. This file defines
//! the handle enum, the typed sub-enums it composes, the newtype IDs
//! it relies on, and `Display` + serde impls for round-trippable
//! debugging snapshots.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 1.1, for the design rationale.

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Stable opaque newtype IDs
// ---------------------------------------------------------------------------
//
// These are assigned during AST → HIR lowering. They must not be
// mixed up: a `MaskId(0)` is not the same as a `ViewId(0)`. Each
// type is its own newtype around `u32` so the type system enforces
// the distinction. `Copy` is fine — they are 4-byte values with no
// indirection.

/// Stable id for a `view` declaration. Indexes the program's view
/// table; assignment order matches resolved IR order so snapshots are
/// reproducible.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ViewId(pub u32);

/// Stable id for a mask predicate. One per `mask` declaration.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MaskId(pub u32);

/// Stable id for an event-ring data channel. The IR distinguishes
/// rings by id so producer/consumer relationships are tracked even
/// when emit-time chooses to fuse multiple rings into one buffer.
///
/// Concrete current rings: the apply-path A-ring (resolved
/// `batch_events`) and the cascade A/B ring used by the cascade
/// pingpong context. Lowering passes resolve a ring id to a buffer
/// name; the IR itself does not name buffers.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EventRingId(pub u32);

/// Stable id for a configuration constant — `config.combat.attack_range`,
/// `config.movement.move_speed_mps`, ability-registry slots, etc.
/// The id is resolved to a struct-field path at emit time via the
/// `emit_config` pipeline; the IR carries only the id.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConfigConstId(pub u32);

/// Stable id for a node in the compute-graph expression tree.
/// `CgExpr` itself is defined in Task 1.2; this id type is hosted
/// here so `AgentRef::Target` can reference it without a forward
/// dependency.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CgExprId(pub u32);

// ---------------------------------------------------------------------------
// Per-agent field id
// ---------------------------------------------------------------------------

/// Static type information for an agent SoA field. The variants here
/// are the primitive shapes the DSL surfaces — every WGSL/Rust
/// emission walks one of these. `Vec3` is `(f32, f32, f32)` packed in
/// the same field order the engine SoA uses.
///
/// This is metadata carried by `AgentFieldId`; it is *not* the IR's
/// general expression type system (Task 1.2 owns `CgTy`). Keeping
/// the field-type tag tiny here avoids forcing every later layer
/// that reads `AgentFieldId` to thread a full type universe just to
/// know "is this a u32?".
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AgentFieldTy {
    /// Plain `f32` — vitals, ranges, multipliers, needs.
    F32,
    /// Plain `u32` — tick stamps, monotonic counters.
    U32,
    /// `i16` — q8 fixed-point slow factor (the only signed-int field).
    I16,
    /// `bool` on CPU; packed `u32` (0/1) on GPU.
    Bool,
    /// 3-component `f32` position vector.
    Vec3,
    /// 8-bit packed enum tag (`MovementMode` today). Stored as `u8`
    /// on CPU, widened to `u32` on GPU.
    EnumU8,
    /// Optional `AgentId` — `Some(id)` on CPU, `0xFFFF_FFFF`
    /// sentinel on GPU. `engaged_with` is the canonical example.
    OptAgentId,
    /// Optional `CreatureType` enum tag.
    OptEnumU32,
}

impl fmt::Display for AgentFieldTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentFieldTy::F32 => write!(f, "f32"),
            AgentFieldTy::U32 => write!(f, "u32"),
            AgentFieldTy::I16 => write!(f, "i16"),
            AgentFieldTy::Bool => write!(f, "bool"),
            AgentFieldTy::Vec3 => write!(f, "vec3"),
            AgentFieldTy::EnumU8 => write!(f, "enum_u8"),
            AgentFieldTy::OptAgentId => write!(f, "opt_agent_id"),
            AgentFieldTy::OptEnumU32 => write!(f, "opt_enum_u32"),
        }
    }
}

/// Every per-agent SoA field the DSL can read or write.
///
/// Sourced from the SoA layout in `crates/engine/src/state/mod.rs`
/// (`SimState::hot_*` and `cold_*` fields) and cross-checked against
/// the GPU mirror in `crates/engine_gpu/src/sync_helpers.rs`
/// (`GpuAgentSlot`). The list is exhaustive over the fields the DSL
/// surface can name today; adding a new field to the SoA without
/// adding a variant here is a structural mismatch the
/// well-formedness pass (Task 1.6) is responsible for catching.
///
/// Each variant carries an `AgentFieldTy` payload via the
/// [`AgentFieldId::ty`] method below — that method is the single
/// source of truth for "which primitive does this field carry?"
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AgentFieldId {
    // --- Hot SoA: physical state, vitals, combat ---
    Pos,
    Hp,
    MaxHp,
    Alive,
    MovementMode,
    Level,
    MoveSpeed,
    MoveSpeedMult,
    ShieldHp,
    Armor,
    MagicResist,
    AttackDamage,
    AttackRange,
    Mana,
    MaxMana,

    // --- Hot SoA: needs (physiological + psychological) ---
    Hunger,
    Thirst,
    RestTimer,
    Safety,
    Shelter,
    Social,
    Purpose,
    Esteem,

    // --- Hot SoA: personality ---
    RiskTolerance,
    SocialDrive,
    Ambition,
    Altruism,
    Curiosity,

    // --- Hot SoA: combat foundation (engagement + statuses) ---
    EngagedWith,
    StunExpiresAtTick,
    SlowExpiresAtTick,
    SlowFactorQ8,
    CooldownNextReadyTick,

    // --- Cold SoA: identity and lifecycle ---
    CreatureType,
    SpawnTick,

    // --- Cold SoA: spatial extras ---
    GridId,
    LocalPos,
    MoveTarget,

    // --- Phase 7 boids fixture ---
    /// Per-agent velocity (vec3). Added 2026-05-02 so `boids.sim`'s
    /// `entity Boid : Agent { pos: vec3, vel: vec3 }` declaration can
    /// resolve `self.vel` reads through the `IrExpr::Field` lowering.
    /// The wolf-sim AgentSoA has no velocity slot today; per-fixture
    /// runtime crates that don't need engine's SoA at all (e.g.
    /// `boids_runtime`'s own `Vec<Vec3>` storage) ignore this entry.
    /// When/if the wolf-sim runtime ever wants velocity, this is the
    /// canonical name.
    Vel,
}

impl AgentFieldId {
    /// Primitive type carried by this field. Single source of truth
    /// shared by every consumer (lowering, codegen, well-formed
    /// checks).
    pub fn ty(self) -> AgentFieldTy {
        use AgentFieldId::*;
        match self {
            // f32 — vitals + scalar combat stats + needs + personality
            Hp | MaxHp | ShieldHp | Armor | MagicResist | AttackDamage | AttackRange | Mana
            | MaxMana | MoveSpeed | MoveSpeedMult | Hunger | Thirst | RestTimer | Safety
            | Shelter | Social | Purpose | Esteem | RiskTolerance | SocialDrive | Ambition
            | Altruism | Curiosity => AgentFieldTy::F32,

            // u32 — monotonic counters + tick stamps + level
            Level | StunExpiresAtTick | SlowExpiresAtTick | CooldownNextReadyTick => {
                AgentFieldTy::U32
            }

            // i16 — q8 fixed-point slow factor
            SlowFactorQ8 => AgentFieldTy::I16,

            // bool — alive flag
            Alive => AgentFieldTy::Bool,

            // Vec3 — position + velocity
            Pos | Vel => AgentFieldTy::Vec3,

            // 8-bit packed enum tag
            MovementMode => AgentFieldTy::EnumU8,

            // Option<AgentId> sentinel-encoded on GPU
            EngagedWith => AgentFieldTy::OptAgentId,

            // Cold + spatial extras
            CreatureType => AgentFieldTy::OptEnumU32,
            SpawnTick => AgentFieldTy::U32,
            GridId => AgentFieldTy::U32,
            LocalPos | MoveTarget => AgentFieldTy::Vec3,
        }
    }

    // (Vel is included in the `Pos | Vel => Vec3` arm above; no
    // additional entry needed here.)

    /// Stable snake_case name, used by `Display` and by external tools
    /// that need a human-readable handle. Matches the SoA field name
    /// modulo the `hot_` / `cold_` prefix and the `_tick` suffix on
    /// expiry stamps where the DSL convention drops it.
    pub fn snake(self) -> &'static str {
        use AgentFieldId::*;
        match self {
            Pos => "pos",
            Hp => "hp",
            MaxHp => "max_hp",
            Alive => "alive",
            MovementMode => "movement_mode",
            Level => "level",
            MoveSpeed => "move_speed",
            MoveSpeedMult => "move_speed_mult",
            ShieldHp => "shield_hp",
            Armor => "armor",
            MagicResist => "magic_resist",
            AttackDamage => "attack_damage",
            AttackRange => "attack_range",
            Mana => "mana",
            MaxMana => "max_mana",
            Hunger => "hunger",
            Thirst => "thirst",
            RestTimer => "rest_timer",
            Safety => "safety",
            Shelter => "shelter",
            Social => "social",
            Purpose => "purpose",
            Esteem => "esteem",
            RiskTolerance => "risk_tolerance",
            SocialDrive => "social_drive",
            Ambition => "ambition",
            Altruism => "altruism",
            Curiosity => "curiosity",
            EngagedWith => "engaged_with",
            StunExpiresAtTick => "stun_expires_at_tick",
            SlowExpiresAtTick => "slow_expires_at_tick",
            SlowFactorQ8 => "slow_factor_q8",
            CooldownNextReadyTick => "cooldown_next_ready_tick",
            CreatureType => "creature_type",
            SpawnTick => "spawn_tick",
            GridId => "grid_id",
            LocalPos => "local_pos",
            MoveTarget => "move_target",
            Vel => "vel",
        }
    }

    /// Every declared [`AgentFieldId`] variant, in declaration order.
    /// Used by [`crate::cg::op::PlumbingKind::PackAgents`] /
    /// [`crate::cg::op::PlumbingKind::UnpackAgents`] to enumerate the
    /// full read/write set without forcing a separate
    /// `DataHandle::AllAgentFields` aggregate variant.
    ///
    /// Adding a new [`AgentFieldId`] variant requires extending this
    /// list — exhaustively checked by the
    /// `all_agent_field_variants_round_trip_through_snake` test in
    /// this module (any miss surfaces as a missing snake-name).
    pub fn all_variants() -> &'static [AgentFieldId] {
        use AgentFieldId::*;
        &[
            Pos,
            Hp,
            MaxHp,
            Alive,
            MovementMode,
            Level,
            MoveSpeed,
            MoveSpeedMult,
            ShieldHp,
            Armor,
            MagicResist,
            AttackDamage,
            AttackRange,
            Mana,
            MaxMana,
            Hunger,
            Thirst,
            RestTimer,
            Safety,
            Shelter,
            Social,
            Purpose,
            Esteem,
            RiskTolerance,
            SocialDrive,
            Ambition,
            Altruism,
            Curiosity,
            EngagedWith,
            StunExpiresAtTick,
            SlowExpiresAtTick,
            SlowFactorQ8,
            CooldownNextReadyTick,
            CreatureType,
            SpawnTick,
            GridId,
            LocalPos,
            MoveTarget,
            Vel,
        ]
    }

    /// Convenience: every [`AgentFieldId`] variant wrapped in a
    /// `DataHandle::AgentField { field, target: AgentRef::Self_ }`.
    /// Used by plumbing lowering's pack / unpack reads/writes.
    pub fn all_agent_field_handles() -> Vec<DataHandle> {
        Self::all_variants()
            .iter()
            .map(|f| DataHandle::AgentField {
                field: *f,
                target: AgentRef::Self_,
            })
            .collect()
    }

    /// Inverse of [`AgentFieldId::snake`]. Returns `Some(variant)` when
    /// `s` is the snake_case name of a declared variant, `None`
    /// otherwise. Used by the expression lowering to map DSL field
    /// identifiers (`self.<name>`) back to the typed [`AgentFieldId`].
    ///
    /// Colocated with `snake()` so the two cannot drift if a new
    /// variant lands; the
    /// `agent_field_id_from_snake_round_trips_every_variant` test
    /// guards round-trip exhaustively.
    pub fn from_snake(s: &str) -> Option<Self> {
        use AgentFieldId::*;
        Some(match s {
            "pos" => Pos,
            "hp" => Hp,
            "max_hp" => MaxHp,
            "alive" => Alive,
            "movement_mode" => MovementMode,
            "level" => Level,
            "move_speed" => MoveSpeed,
            "move_speed_mult" => MoveSpeedMult,
            "shield_hp" => ShieldHp,
            "armor" => Armor,
            "magic_resist" => MagicResist,
            "attack_damage" => AttackDamage,
            "attack_range" => AttackRange,
            "mana" => Mana,
            "max_mana" => MaxMana,
            "hunger" => Hunger,
            "thirst" => Thirst,
            "rest_timer" => RestTimer,
            "safety" => Safety,
            "shelter" => Shelter,
            "social" => Social,
            "purpose" => Purpose,
            "esteem" => Esteem,
            "risk_tolerance" => RiskTolerance,
            "social_drive" => SocialDrive,
            "ambition" => Ambition,
            "altruism" => Altruism,
            "curiosity" => Curiosity,
            "engaged_with" => EngagedWith,
            "stun_expires_at_tick" => StunExpiresAtTick,
            "slow_expires_at_tick" => SlowExpiresAtTick,
            "slow_factor_q8" => SlowFactorQ8,
            "cooldown_next_ready_tick" => CooldownNextReadyTick,
            "creature_type" => CreatureType,
            "spawn_tick" => SpawnTick,
            "grid_id" => GridId,
            "local_pos" => LocalPos,
            "move_target" => MoveTarget,
            "vel" => Vel,
            _ => return None,
        })
    }
}

impl fmt::Display for AgentFieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.snake())
    }
}

// ---------------------------------------------------------------------------
// AgentRef
// ---------------------------------------------------------------------------

/// Which agent does the field reference?
///
/// The dispatch shape implies the "current" actor — `Self_` for
/// `PerAgent`, `Actor` and `EventTarget` for `PerEvent`. `Target` is
/// computed: it carries a CgExpr id whose evaluation produces the
/// agent slot to read or write (typically read out of a candidate
/// buffer or scoring output). `PerPairCandidate` names the implicit
/// candidate side of a per-pair dispatch — see its docstring for the
/// resolution contract.
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AgentRef {
    /// The dispatch's current agent (PerAgent shape).
    Self_,
    /// A target identified by an expression (typically read from a
    /// candidate buffer or scoring output).
    Target(CgExprId),
    /// The actor of the current event (PerEvent shape).
    Actor,
    /// The target of the current event.
    EventTarget,
    /// The candidate side of a per-pair dispatch — `target` in DSL
    /// surface (e.g., `mask MoveToward(target) from
    /// query.nearby_agents(...)`'s predicate referencing
    /// `target.alive`, `target.pos`, etc.).
    ///
    /// Carries no payload: the candidate's slot id is implicit in the
    /// surrounding [`crate::cg::dispatch::DispatchShape::PerPair`]
    /// shape's `source` (today
    /// [`crate::cg::dispatch::PerPairSource::SpatialQuery`]). The IR
    /// layer doesn't know which buffer / offset holds the candidate —
    /// the emit layer (Task 4.x) resolves this from the dispatch shape
    /// at codegen time. Compare to `Target(CgExprId)`, which carries a
    /// concrete expression that must already evaluate to an `AgentId`
    /// in the surrounding op's expression arena.
    PerPairCandidate,
}

impl fmt::Display for AgentRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentRef::Self_ => f.write_str("self"),
            AgentRef::Target(id) => write!(f, "target(#{})", id.0),
            AgentRef::Actor => f.write_str("actor"),
            AgentRef::EventTarget => f.write_str("event_target"),
            AgentRef::PerPairCandidate => f.write_str("per_pair_candidate"),
        }
    }
}

// ---------------------------------------------------------------------------
// View storage layers
// ---------------------------------------------------------------------------

/// Storage layers a materialized view exposes. The layer set depends
/// on the view's storage hint — encoded here so the IR can refer to
/// "which layer of which view" without a string lookup.
///
/// Mapping (sourced from `crates/dsl_compiler/src/emit_view_fold_kernel.rs`
/// and `emit_view_wgsl.rs`):
///
/// - `PairMap` (no decay) → `Primary`
/// - `PairMap` + `@decay`  → `Primary`, `Anchor`
/// - `PerEntityTopK { k }` (`k == 1`, slot_map collapse) → `Primary`
/// - `PerEntityTopK { k }` (`k >= 2`, sparse layout)     → `Primary`, `Ids`
/// - `SymmetricPairTopK { k }` → `Primary` (slots), `Counts`
/// - `PerEntityRing { k }`     → `Primary` (rings), `Cursors`
/// - `LazyCached`              → `Primary` (the per-tick result cache)
///
/// The plan calls these "primary, anchor, ids, …" generically; the
/// shape-specific layers (Counts, Cursors) keep the IR honest about
/// which buffers a given view actually has.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViewStorageSlot {
    /// The main storage backing the view — values for pair_map,
    /// slots for sym_pair_topk, rings for per_entity_ring.
    Primary,
    /// Anchor-tick layer for `@decay` pair_maps (one u32 per cell).
    Anchor,
    /// "Other end" id layer for sparse top-K storage (per-entity
    /// agent ids that pair with the primary value column).
    Ids,
    /// Atomic per-row count for symmetric_pair_topk (how many
    /// neighbors are populated for an owner).
    Counts,
    /// Atomic per-owner cursor for per_entity_ring (write head,
    /// taken mod K).
    Cursors,
}

impl fmt::Display for ViewStorageSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViewStorageSlot::Primary => f.write_str("primary"),
            ViewStorageSlot::Anchor => f.write_str("anchor"),
            ViewStorageSlot::Ids => f.write_str("ids"),
            ViewStorageSlot::Counts => f.write_str("counts"),
            ViewStorageSlot::Cursors => f.write_str("cursors"),
        }
    }
}

// ---------------------------------------------------------------------------
// Event-ring access mode
// ---------------------------------------------------------------------------

/// How an event ring is being touched in this op. The IR tracks both
/// directions (read vs append) explicitly so dependency analysis can
/// build a proper producer/consumer graph and the lowering knows
/// which atomics to insert.
///
/// `Drain` is a third mode introduced by Task 2.7 plumbing: the apply-
/// path consumes events from the ring and zeroes the ring's tail
/// counter so the next iteration starts empty. Drain is structurally
/// distinct from `Read` because read-only consumers can run alongside
/// each other; a drain mutates the ring's tail and must be exclusive.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventRingAccess {
    /// Read events out of the ring (a fold or apply pass).
    Read,
    /// Append events into the ring (an emitter pass).
    Append,
    /// Drain the ring — read every entry and reset the tail counter.
    /// Used by [`crate::cg::op::PlumbingKind::DrainEvents`]; the IR
    /// tracks drain as a third mode so schedule synthesis can sequence
    /// it after every reader/writer of the ring.
    Drain,
}

impl fmt::Display for EventRingAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventRingAccess::Read => f.write_str("read"),
            EventRingAccess::Append => f.write_str("append"),
            EventRingAccess::Drain => f.write_str("drain"),
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial-grid storage kinds
// ---------------------------------------------------------------------------

/// Pieces of the uniform spatial grid the DSL can read or write.
/// Sourced from `BindingSources`'s `pool` group (see
/// `crates/dsl_compiler/src/emit_spatial_kernel.rs`).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpatialStorageKind {
    /// `spatial_grid_cells` — packed agent ids, indexed by cell
    /// offset.
    GridCells,
    /// `spatial_grid_offsets` — per-cell starting offset into
    /// `GridCells`.
    GridOffsets,
    /// `spatial_query_results` — scratch buffer holding the result
    /// of one nearest-neighbor / nearby-agents query.
    QueryResults,
}

impl fmt::Display for SpatialStorageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpatialStorageKind::GridCells => f.write_str("grid_cells"),
            SpatialStorageKind::GridOffsets => f.write_str("grid_offsets"),
            SpatialStorageKind::QueryResults => f.write_str("query_results"),
        }
    }
}

// ---------------------------------------------------------------------------
// Agent scratch buffers (Task 2.7 plumbing)
// ---------------------------------------------------------------------------

/// Kinds of per-agent scratch buffer the plumbing layer uses. Sourced
/// from the existing emit-time agent pack/unpack pipeline (today only
/// the canonical packed-SoA scratch exists; future packings — e.g.,
/// per-field shadow buffers — accrete here as variants).
///
/// The variant set is intentionally tight: a generic "scratch" with a
/// free-form id would defeat structural typing. Each named kind maps
/// to one well-known buffer at emit time; the lowering pass selects
/// the kind by [`crate::cg::op::PlumbingKind`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AgentScratchKind {
    /// Packed agent SoA buffer used by `PackAgents` / `UnpackAgents` —
    /// one record per agent slot with every hot/cold field laid out in
    /// the engine's GPU `GpuAgentSlot` order. Reads land into here
    /// before a GPU dispatch reads the agents; writes from this buffer
    /// land back into per-field [`AgentFieldId`] storage on unpack.
    Packed,
}

impl fmt::Display for AgentScratchKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentScratchKind::Packed => f.write_str("packed"),
        }
    }
}

// ---------------------------------------------------------------------------
// RNG purpose
// ---------------------------------------------------------------------------

/// Stream identifier for the deterministic per-agent RNG.
///
/// The runtime primitive is
/// `engine::rng::per_agent_u32(world_seed, agent_id, tick, purpose)`
/// where `purpose` is a free-form byte string. The IR pins the
/// canonical purposes the engine + DSL surface today as typed
/// variants so emitters can produce stable byte literals; future
/// purposes are added by extending this enum (no string keys).
///
/// Variants mirror the documented purposes in
/// `crates/engine/src/rng.rs` (`b"action"`, `b"sample"`,
/// `b"shuffle"`, `b"conception"`).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RngPurpose {
    /// `b"action"` — action selection / tiebreak.
    Action,
    /// `b"sample"` — generic sampling stream.
    Sample,
    /// `b"shuffle"` — agent-order shuffles within a step.
    Shuffle,
    /// `b"conception"` — reproduction roll.
    Conception,
}

impl RngPurpose {
    /// Byte-literal form passed to `per_agent_u32` at emit time.
    pub fn as_bytes(self) -> &'static [u8] {
        match self {
            RngPurpose::Action => b"action",
            RngPurpose::Sample => b"sample",
            RngPurpose::Shuffle => b"shuffle",
            RngPurpose::Conception => b"conception",
        }
    }

    /// Snake-case label used by `Display` and structured logs.
    pub fn snake(self) -> &'static str {
        match self {
            RngPurpose::Action => "action",
            RngPurpose::Sample => "sample",
            RngPurpose::Shuffle => "shuffle",
            RngPurpose::Conception => "conception",
        }
    }
}

impl fmt::Display for RngPurpose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.snake())
    }
}

// ---------------------------------------------------------------------------
// DataHandle
// ---------------------------------------------------------------------------

/// Typed reference to a piece of simulation state. The compiler
/// tracks every "where does this data live" via these handles —
/// never via raw names. Naming becomes an emit-time concern: the
/// lowering decides what binding slot or struct field corresponds
/// to a given handle.
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DataHandle {
    /// Per-agent SoA field. `field` is a typed enum that names every
    /// field the DSL can read or write (hp, pos, alive, shield_hp,
    /// attack_damage, …). The agent itself is selected by `target`,
    /// which combines with the dispatch shape (PerAgent → "current
    /// agent slot").
    AgentField {
        field: AgentFieldId,
        target: AgentRef,
    },

    /// Materialized view storage. `view` is the view's stable id;
    /// `slot` is which of the view's storage layers (primary,
    /// anchor, ids, counts, cursors — depends on storage hint).
    ViewStorage { view: ViewId, slot: ViewStorageSlot },

    /// Event ring — either the apply-path A-ring or the cascade-
    /// physics ring, identified by `ring`. `kind` carries whether
    /// this access reads or appends. The IR tracks producer/
    /// consumer relationships through these handles.
    EventRing {
        ring: EventRingId,
        kind: EventRingAccess,
    },

    /// Configuration constant — `sim_cfg.tick`, `sim_cfg.move_speed`,
    /// ability registry slots, etc. `id` resolves through the
    /// config emit pipeline.
    ConfigConst { id: ConfigConstId },

    /// Mask bitmap output (one bit per agent per mask). `mask` is
    /// the mask's stable id.
    MaskBitmap { mask: MaskId },

    /// Scoring output — per-agent (action, target, score) tuple.
    ScoringOutput,

    /// Spatial-grid storage (cells, offsets, query results).
    SpatialStorage { kind: SpatialStorageKind },

    /// The deterministic RNG primitive: `per_agent_u32(seed, agent,
    /// tick, purpose)`. Emit-time becomes a function call — but at
    /// the IR level it's a typed read.
    Rng { purpose: RngPurpose },

    /// Per-agent alive-status bitmap (1 bit per agent). One global
    /// resource — the IR carries no id field. Refreshed by
    /// [`crate::cg::op::PlumbingKind::AliveBitmap`] every tick after
    /// any op that mutates [`AgentFieldId::Alive`]. Distinct from
    /// [`MaskBitmap`] (which is per-mask user-authored predicate
    /// output) because the alive bitmap is a runtime-system invariant,
    /// not a DSL-surface predicate.
    AliveBitmap,

    /// Indirect-args buffer for one cascade ring's per-iteration
    /// dispatch. Seeded by
    /// [`crate::cg::op::PlumbingKind::SeedIndirectArgs`] from the
    /// ring's tail count so the next [`crate::cg::dispatch::DispatchShape::PerEvent`]
    /// dispatch reads the correct workgroup count. One buffer per
    /// ring; the typed [`EventRingId`] keeps producer/consumer edges
    /// honest at the IR level.
    IndirectArgs { ring: EventRingId },

    /// Per-agent scratch buffer for the GPU pack/unpack pipeline.
    /// `kind` selects which scratch buffer is being touched (today
    /// only [`AgentScratchKind::Packed`]). [`PackAgents`] writes the
    /// packed buffer; [`UnpackAgents`] reads it back into per-field
    /// storage.
    ///
    /// [`PackAgents`]: crate::cg::op::PlumbingKind::PackAgents
    /// [`UnpackAgents`]: crate::cg::op::PlumbingKind::UnpackAgents
    AgentScratch { kind: AgentScratchKind },

    /// The sim_cfg uniform buffer — a whole-buffer write performed by
    /// [`crate::cg::op::PlumbingKind::UploadSimCfg`] every tick. The
    /// per-key [`ConfigConst`] handles cover individual reads of
    /// configuration constants once the buffer has been uploaded; the
    /// upload itself touches the buffer as one indivisible region, so
    /// it gets its own variant rather than synthesising a sentinel
    /// `ConfigConst` id.
    SimCfgBuffer,

    /// The snapshot dump trigger — written by
    /// [`crate::cg::op::PlumbingKind::KickSnapshot`] at end-of-tick to
    /// signal the runtime that the snapshot ring should advance.
    /// Single global resource, no id field.
    SnapshotKick,
}

/// Kind tag for an interned id surfaced by [`DataHandle::fmt_with`].
/// Closed set — adding a new id-kind to a [`DataHandle`] variant adds a
/// member here so the `DataHandleNameResolver` impls handle every
/// shape exhaustively.
///
/// Today only the four kinds actually embedded in a [`DataHandle`]
/// (`view`, `mask`, `event_ring`, `config_const`) are listed. Other id
/// kinds the IR has (event-kind, action, scoring, physics-rule) live on
/// op variants (`ComputeOpKind::*`), not on data handles, and are
/// rendered through the program-level pretty-printer rather than
/// through `DataHandle::Display`.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum IdKind {
    View,
    Mask,
    EventRing,
    ConfigConst,
}

/// Resolves an interned id to its human-readable name. The
/// program-level interner implements this in `program.rs`; tests pass
/// the unit `()` resolver below for the always-`None` shape (every id
/// renders as `#N`).
///
/// Borrowed-return shape: `Option<&str>` with the lifetime tied to
/// `&self`. The closure-based form runs into `for<'a>` higher-rank
/// trouble; a trait object sidesteps it cleanly while keeping the
/// "single-source rendering" guarantee.
pub trait DataHandleNameResolver {
    fn name_for(&self, kind: IdKind, id: u32) -> Option<&str>;
}

/// The unit resolver: always returns `None`. Used by
/// [`fmt::Display for DataHandle`] so the default rendering keeps the
/// opaque `#N` form. Tests for the unnamed path of `fmt_with` use this
/// resolver directly.
impl DataHandleNameResolver for () {
    fn name_for(&self, _kind: IdKind, _id: u32) -> Option<&str> {
        None
    }
}

impl DataHandle {
    /// Format the handle, consulting `names` for human-readable id
    /// substitutions. `names.name_for(kind, id)` returning `Some(name)`
    /// replaces the opaque `#N` form with `name`; returning `None`
    /// keeps the `#N` form (Task 1.1's contract).
    ///
    /// Both [`fmt::Display`] and the program-level
    /// `display_with_names` route through this single method, so the
    /// shape of the rendered output is single-source: adding a new
    /// `DataHandle` variant updates both surfaces simultaneously.
    pub fn fmt_with(
        &self,
        f: &mut fmt::Formatter<'_>,
        names: &dyn DataHandleNameResolver,
    ) -> fmt::Result {
        match self {
            DataHandle::AgentField { field, target } => {
                write!(f, "agent.{}.{}", target, field)
            }
            DataHandle::ViewStorage { view, slot } => {
                write!(f, "view[")?;
                write_named_id(f, names.name_for(IdKind::View, view.0), view.0)?;
                write!(f, "].{}", slot)
            }
            DataHandle::EventRing { ring, kind } => {
                write!(f, "event_ring[")?;
                write_named_id(f, names.name_for(IdKind::EventRing, ring.0), ring.0)?;
                write!(f, "].{}", kind)
            }
            DataHandle::ConfigConst { id } => {
                write!(f, "config[")?;
                write_named_id(f, names.name_for(IdKind::ConfigConst, id.0), id.0)?;
                write!(f, "]")
            }
            DataHandle::MaskBitmap { mask } => {
                write!(f, "mask[")?;
                write_named_id(f, names.name_for(IdKind::Mask, mask.0), mask.0)?;
                write!(f, "].bitmap")
            }
            DataHandle::ScoringOutput => f.write_str("scoring.output"),
            DataHandle::SpatialStorage { kind } => write!(f, "spatial.{}", kind),
            DataHandle::Rng { purpose } => write!(f, "rng({})", purpose),
            DataHandle::AliveBitmap => f.write_str("alive_bitmap"),
            DataHandle::IndirectArgs { ring } => {
                write!(f, "indirect_args[")?;
                write_named_id(f, names.name_for(IdKind::EventRing, ring.0), ring.0)?;
                write!(f, "]")
            }
            DataHandle::AgentScratch { kind } => write!(f, "agent_scratch.{}", kind),
            DataHandle::SimCfgBuffer => f.write_str("sim_cfg_buffer"),
            DataHandle::SnapshotKick => f.write_str("snapshot_kick"),
        }
    }
}

/// Format an id token as either its interned `name` (when present) or
/// the opaque `#N` form. Shared between `DataHandle::fmt_with` and
/// any future caller that needs the same rendering rule.
fn write_named_id(f: &mut fmt::Formatter<'_>, name: Option<&str>, id: u32) -> fmt::Result {
    match name {
        Some(n) => f.write_str(n),
        None => write!(f, "#{}", id),
    }
}

impl fmt::Display for DataHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Default rendering: never substitute names — every id renders
        // as `#N`. Programs that want named substitution route through
        // `DataHandle::fmt_with` directly (or via
        // `CgProgram::display_with_names`).
        self.fmt_with(f, &())
    }
}

// ---------------------------------------------------------------------------
// Cycle-edge projection
// ---------------------------------------------------------------------------

/// Projection of a [`DataHandle`] used as the key when matching writers
/// to readers during cycle detection (see
/// `crate::cg::well_formed::detect_cycles`).
///
/// Most variants are projected as-is — full identity equality is what
/// "this storage location aliases that one" means for cycle detection.
/// The one exception is [`DataHandle::EventRing`]: its
/// [`EventRingAccess`] discriminant (`Read` | `Append` | `Drain`)
/// reflects the **producer/consumer kind** of the access, not a
/// distinct resource. A producer's `Append` and a consumer's `Read` on
/// the same ring form a real dependency edge — the cycle gate must
/// close that edge regardless of access mode. So `EventRing` collapses
/// to its [`EventRingId`] alone, and every other variant keeps its
/// full identity by being wrapped in [`CycleEdgeKey::Other`].
///
/// This projection is **additive**: it does not change
/// [`DataHandle`]'s `Eq`/`Hash`/`Ord` semantics. Other consumers of
/// `DataHandle` (the writers/readers maps in op metadata, error
/// rendering, etc.) continue to use full-identity equality.
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum CycleEdgeKey {
    /// An event ring keyed by ring identity alone — access mode
    /// (Read/Append/Drain) is intentionally collapsed.
    Ring(EventRingId),
    /// Any non-`EventRing` `DataHandle`, keyed by its full identity.
    Other(DataHandle),
}

impl DataHandle {
    /// Project this [`DataHandle`] to its [`CycleEdgeKey`] for cycle
    /// detection.
    ///
    /// `EventRing { ring, kind: _ }` → `Ring(ring)` so that producer
    /// (`Append`) and consumer (`Read`) accesses on the same ring map
    /// to the same key. Every other variant is returned as
    /// `Other(self.clone())` — full identity is preserved.
    ///
    /// The `kind` field on `EventRing` is matched explicitly (not via
    /// `..`) so adding a new field to `EventRing` will force a compile
    /// error here, prompting an explicit decision about whether the
    /// new field participates in the cycle-edge identity.
    pub fn cycle_edge_key(&self) -> CycleEdgeKey {
        match self {
            // EventRing: collapse the access kind. Producer/consumer
            // edges must close regardless of which side the read or
            // write claims.
            DataHandle::EventRing { ring, kind: _ } => CycleEdgeKey::Ring(*ring),

            // Every other variant keeps full-identity equality. List
            // exhaustively so adding a new `DataHandle` variant forces
            // an explicit decision here rather than silently falling
            // into the `Other` bucket via a wildcard.
            DataHandle::AgentField { .. }
            | DataHandle::ViewStorage { .. }
            | DataHandle::ConfigConst { .. }
            | DataHandle::MaskBitmap { .. }
            | DataHandle::ScoringOutput
            | DataHandle::SpatialStorage { .. }
            | DataHandle::Rng { .. }
            | DataHandle::AliveBitmap
            | DataHandle::IndirectArgs { .. }
            | DataHandle::AgentScratch { .. }
            | DataHandle::SimCfgBuffer
            | DataHandle::SnapshotKick => CycleEdgeKey::Other(self.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip via serde JSON and assert the value compares equal.
    fn assert_roundtrip(handle: &DataHandle) {
        let json = serde_json::to_string(handle).expect("serialize");
        let back: DataHandle = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, handle, "round-trip changed value (json was {json})");
    }

    // ---- AgentField ----

    #[test]
    fn agent_field_self_hp_display_and_roundtrip() {
        let h = DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        };
        assert_eq!(format!("{}", h), "agent.self.hp");
        assert_roundtrip(&h);
    }

    #[test]
    fn agent_field_actor_pos_display() {
        let h = DataHandle::AgentField {
            field: AgentFieldId::Pos,
            target: AgentRef::Actor,
        };
        assert_eq!(format!("{}", h), "agent.actor.pos");
        assert_roundtrip(&h);
    }

    #[test]
    fn agent_field_event_target_alive_display() {
        let h = DataHandle::AgentField {
            field: AgentFieldId::Alive,
            target: AgentRef::EventTarget,
        };
        assert_eq!(format!("{}", h), "agent.event_target.alive");
        assert_roundtrip(&h);
    }

    #[test]
    fn agent_field_target_expr_display_and_roundtrip() {
        let h = DataHandle::AgentField {
            field: AgentFieldId::ShieldHp,
            target: AgentRef::Target(CgExprId(7)),
        };
        assert_eq!(format!("{}", h), "agent.target(#7).shield_hp");
        assert_roundtrip(&h);
    }

    #[test]
    fn agent_field_per_pair_candidate_display_and_roundtrip() {
        // Pair-bound mask predicates surface `target.<field>` reads as
        // `AgentField { target: AgentRef::PerPairCandidate, .. }`. Pin
        // the rendered form + serde round-trip so structural snapshots
        // and structured logs see the stable wire format.
        let h = DataHandle::AgentField {
            field: AgentFieldId::Alive,
            target: AgentRef::PerPairCandidate,
        };
        assert_eq!(format!("{}", h), "agent.per_pair_candidate.alive");
        assert_roundtrip(&h);
    }

    #[test]
    fn agent_field_id_ty_classification() {
        // Sanity — every f32 vital is f32.
        assert_eq!(AgentFieldId::Hp.ty(), AgentFieldTy::F32);
        assert_eq!(AgentFieldId::AttackDamage.ty(), AgentFieldTy::F32);
        // Position is Vec3.
        assert_eq!(AgentFieldId::Pos.ty(), AgentFieldTy::Vec3);
        // Tick stamps are u32.
        assert_eq!(AgentFieldId::StunExpiresAtTick.ty(), AgentFieldTy::U32);
        assert_eq!(AgentFieldId::CooldownNextReadyTick.ty(), AgentFieldTy::U32);
        // i16 q8 fixed-point.
        assert_eq!(AgentFieldId::SlowFactorQ8.ty(), AgentFieldTy::I16);
        // bool alive.
        assert_eq!(AgentFieldId::Alive.ty(), AgentFieldTy::Bool);
        // movement_mode is an 8-bit packed enum tag.
        assert_eq!(AgentFieldId::MovementMode.ty(), AgentFieldTy::EnumU8);
        // engaged_with carries the optional-agent-id sentinel.
        assert_eq!(AgentFieldId::EngagedWith.ty(), AgentFieldTy::OptAgentId);
        // creature_type is the optional u32 enum tag.
        assert_eq!(AgentFieldId::CreatureType.ty(), AgentFieldTy::OptEnumU32);
    }

    #[test]
    fn agent_field_id_snake_names_are_unique() {
        // Quick guard against typos that would silently overlap.
        // Build the full set of declared variants by exhaustively
        // exercising `snake()` on a representative list. If a new
        // variant is added without updating `snake()`, `match` would
        // refuse to compile, so this test only needs to verify
        // uniqueness across the names produced.
        let names: Vec<&'static str> = [
            AgentFieldId::Pos,
            AgentFieldId::Hp,
            AgentFieldId::MaxHp,
            AgentFieldId::Alive,
            AgentFieldId::MovementMode,
            AgentFieldId::Level,
            AgentFieldId::MoveSpeed,
            AgentFieldId::MoveSpeedMult,
            AgentFieldId::ShieldHp,
            AgentFieldId::Armor,
            AgentFieldId::MagicResist,
            AgentFieldId::AttackDamage,
            AgentFieldId::AttackRange,
            AgentFieldId::Mana,
            AgentFieldId::MaxMana,
            AgentFieldId::Hunger,
            AgentFieldId::Thirst,
            AgentFieldId::RestTimer,
            AgentFieldId::Safety,
            AgentFieldId::Shelter,
            AgentFieldId::Social,
            AgentFieldId::Purpose,
            AgentFieldId::Esteem,
            AgentFieldId::RiskTolerance,
            AgentFieldId::SocialDrive,
            AgentFieldId::Ambition,
            AgentFieldId::Altruism,
            AgentFieldId::Curiosity,
            AgentFieldId::EngagedWith,
            AgentFieldId::StunExpiresAtTick,
            AgentFieldId::SlowExpiresAtTick,
            AgentFieldId::SlowFactorQ8,
            AgentFieldId::CooldownNextReadyTick,
            AgentFieldId::CreatureType,
            AgentFieldId::SpawnTick,
            AgentFieldId::GridId,
            AgentFieldId::LocalPos,
            AgentFieldId::MoveTarget,
        ]
        .iter()
        .map(|f| f.snake())
        .collect();

        let mut seen = std::collections::HashSet::new();
        for n in &names {
            assert!(seen.insert(*n), "duplicate AgentFieldId snake name: {n}");
        }
    }

    #[test]
    fn agent_field_id_from_snake_round_trips_every_variant() {
        // Property: for every declared `AgentFieldId` variant,
        // `from_snake(snake()) == Some(variant)`. Mirrors the variant
        // list in `agent_field_id_snake_names_are_unique` — both will
        // miss a new variant the same way (compile-time `match`
        // exhaustiveness on `snake()` plus this hand-written list keeps
        // them in lockstep).
        //
        // Also asserts `from_snake` rejects an obviously-bogus name —
        // keeps the `_ => None` arm honest.
        let variants = [
            AgentFieldId::Pos,
            AgentFieldId::Hp,
            AgentFieldId::MaxHp,
            AgentFieldId::Alive,
            AgentFieldId::MovementMode,
            AgentFieldId::Level,
            AgentFieldId::MoveSpeed,
            AgentFieldId::MoveSpeedMult,
            AgentFieldId::ShieldHp,
            AgentFieldId::Armor,
            AgentFieldId::MagicResist,
            AgentFieldId::AttackDamage,
            AgentFieldId::AttackRange,
            AgentFieldId::Mana,
            AgentFieldId::MaxMana,
            AgentFieldId::Hunger,
            AgentFieldId::Thirst,
            AgentFieldId::RestTimer,
            AgentFieldId::Safety,
            AgentFieldId::Shelter,
            AgentFieldId::Social,
            AgentFieldId::Purpose,
            AgentFieldId::Esteem,
            AgentFieldId::RiskTolerance,
            AgentFieldId::SocialDrive,
            AgentFieldId::Ambition,
            AgentFieldId::Altruism,
            AgentFieldId::Curiosity,
            AgentFieldId::EngagedWith,
            AgentFieldId::StunExpiresAtTick,
            AgentFieldId::SlowExpiresAtTick,
            AgentFieldId::SlowFactorQ8,
            AgentFieldId::CooldownNextReadyTick,
            AgentFieldId::CreatureType,
            AgentFieldId::SpawnTick,
            AgentFieldId::GridId,
            AgentFieldId::LocalPos,
            AgentFieldId::MoveTarget,
        ];
        for v in variants {
            let name = v.snake();
            assert_eq!(
                AgentFieldId::from_snake(name),
                Some(v),
                "from_snake({name:?}) did not round-trip to {v:?}"
            );
        }
        assert_eq!(AgentFieldId::from_snake("hp_pct"), None);
        assert_eq!(AgentFieldId::from_snake(""), None);
    }

    // ---- ViewStorage ----

    #[test]
    fn view_storage_primary_display_and_roundtrip() {
        let h = DataHandle::ViewStorage {
            view: ViewId(3),
            slot: ViewStorageSlot::Primary,
        };
        assert_eq!(format!("{}", h), "view[#3].primary");
        assert_roundtrip(&h);
    }

    #[test]
    fn view_storage_all_slots_display_distinct() {
        let view = ViewId(0);
        let slots = [
            (ViewStorageSlot::Primary, "view[#0].primary"),
            (ViewStorageSlot::Anchor, "view[#0].anchor"),
            (ViewStorageSlot::Ids, "view[#0].ids"),
            (ViewStorageSlot::Counts, "view[#0].counts"),
            (ViewStorageSlot::Cursors, "view[#0].cursors"),
        ];
        for (slot, expected) in slots {
            let h = DataHandle::ViewStorage { view, slot };
            assert_eq!(format!("{}", h), expected);
            assert_roundtrip(&h);
        }
    }

    // ---- EventRing ----

    #[test]
    fn event_ring_read_display_and_roundtrip() {
        let h = DataHandle::EventRing {
            ring: EventRingId(1),
            kind: EventRingAccess::Read,
        };
        assert_eq!(format!("{}", h), "event_ring[#1].read");
        assert_roundtrip(&h);
    }

    #[test]
    fn event_ring_append_display_and_roundtrip() {
        let h = DataHandle::EventRing {
            ring: EventRingId(2),
            kind: EventRingAccess::Append,
        };
        assert_eq!(format!("{}", h), "event_ring[#2].append");
        assert_roundtrip(&h);
    }

    // ---- ConfigConst ----

    #[test]
    fn config_const_display_and_roundtrip() {
        let h = DataHandle::ConfigConst {
            id: ConfigConstId(42),
        };
        assert_eq!(format!("{}", h), "config[#42]");
        assert_roundtrip(&h);
    }

    // ---- MaskBitmap ----

    #[test]
    fn mask_bitmap_display_and_roundtrip() {
        let h = DataHandle::MaskBitmap { mask: MaskId(5) };
        assert_eq!(format!("{}", h), "mask[#5].bitmap");
        assert_roundtrip(&h);
    }

    // ---- ScoringOutput ----

    #[test]
    fn scoring_output_display_and_roundtrip() {
        let h = DataHandle::ScoringOutput;
        assert_eq!(format!("{}", h), "scoring.output");
        assert_roundtrip(&h);
    }

    // ---- SpatialStorage ----

    #[test]
    fn spatial_storage_all_kinds_display_and_roundtrip() {
        let kinds = [
            (SpatialStorageKind::GridCells, "spatial.grid_cells"),
            (SpatialStorageKind::GridOffsets, "spatial.grid_offsets"),
            (SpatialStorageKind::QueryResults, "spatial.query_results"),
        ];
        for (kind, expected) in kinds {
            let h = DataHandle::SpatialStorage { kind };
            assert_eq!(format!("{}", h), expected);
            assert_roundtrip(&h);
        }
    }

    // ---- Rng ----

    #[test]
    fn rng_purpose_display_and_roundtrip() {
        let purposes = [
            (RngPurpose::Action, "rng(action)", b"action".as_slice()),
            (RngPurpose::Sample, "rng(sample)", b"sample".as_slice()),
            (RngPurpose::Shuffle, "rng(shuffle)", b"shuffle".as_slice()),
            (
                RngPurpose::Conception,
                "rng(conception)",
                b"conception".as_slice(),
            ),
        ];
        for (purpose, expected_display, expected_bytes) in purposes {
            let h = DataHandle::Rng { purpose };
            assert_eq!(format!("{}", h), expected_display);
            assert_eq!(purpose.as_bytes(), expected_bytes);
            assert_roundtrip(&h);
        }
    }

    // ---- Equality + Hash ----

    #[test]
    fn structural_equality_is_field_by_field() {
        let a = DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        };
        let b = DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        };
        let c = DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Actor,
        };
        let d = DataHandle::AgentField {
            field: AgentFieldId::Mana,
            target: AgentRef::Self_,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);

        // Hash-based containers should treat structurally-equal
        // handles as one entry.
        let mut set = std::collections::HashSet::new();
        set.insert(a.clone());
        set.insert(b.clone());
        set.insert(c.clone());
        assert_eq!(set.len(), 2, "a == b should collapse to one set entry");
    }

    // ---- New plumbing-related variants (Task 2.7) ---------------------

    #[test]
    fn alive_bitmap_display_and_roundtrip() {
        let h = DataHandle::AliveBitmap;
        assert_eq!(format!("{}", h), "alive_bitmap");
        assert_roundtrip(&h);
    }

    #[test]
    fn indirect_args_display_and_roundtrip() {
        let h = DataHandle::IndirectArgs {
            ring: EventRingId(2),
        };
        assert_eq!(format!("{}", h), "indirect_args[#2]");
        assert_roundtrip(&h);
    }

    #[test]
    fn agent_scratch_packed_display_and_roundtrip() {
        let h = DataHandle::AgentScratch {
            kind: AgentScratchKind::Packed,
        };
        assert_eq!(format!("{}", h), "agent_scratch.packed");
        assert_roundtrip(&h);
    }

    #[test]
    fn sim_cfg_buffer_display_and_roundtrip() {
        let h = DataHandle::SimCfgBuffer;
        assert_eq!(format!("{}", h), "sim_cfg_buffer");
        assert_roundtrip(&h);
    }

    #[test]
    fn snapshot_kick_display_and_roundtrip() {
        let h = DataHandle::SnapshotKick;
        assert_eq!(format!("{}", h), "snapshot_kick");
        assert_roundtrip(&h);
    }

    #[test]
    fn event_ring_drain_display_and_roundtrip() {
        let h = DataHandle::EventRing {
            ring: EventRingId(4),
            kind: EventRingAccess::Drain,
        };
        assert_eq!(format!("{}", h), "event_ring[#4].drain");
        assert_roundtrip(&h);
    }

    #[test]
    fn agent_field_id_all_variants_matches_snake_name_count() {
        // Every entry returned by `all_variants` must round-trip
        // through `snake()` / `from_snake()` — guards against a future
        // variant being added to `AgentFieldId` without extending the
        // `all_variants` enumeration.
        let all = AgentFieldId::all_variants();
        for v in all {
            assert_eq!(
                AgentFieldId::from_snake(v.snake()),
                Some(*v),
                "all_variants entry {v:?} did not round-trip through snake"
            );
        }
        // A spot-check on count — the enum has 39 variants today (38
        // wolf-sim baseline + Vel added 2026-05-02 for the Boids
        // fixture); if a new variant lands and `all_variants` isn't
        // updated, this assertion fails before the round-trip loop
        // above can.
        assert_eq!(all.len(), 39);
    }

    #[test]
    fn all_agent_field_handles_returns_one_self_handle_per_variant() {
        let handles = AgentFieldId::all_agent_field_handles();
        assert_eq!(handles.len(), AgentFieldId::all_variants().len());
        for (h, v) in handles.iter().zip(AgentFieldId::all_variants().iter()) {
            match h {
                DataHandle::AgentField { field, target } => {
                    assert_eq!(field, v);
                    assert_eq!(target, &AgentRef::Self_);
                }
                other => panic!("unexpected handle {other:?}"),
            }
        }
    }

    #[test]
    fn newtype_ids_are_distinct_per_type_at_the_type_level() {
        // This is a compile-time guarantee, not a runtime one — the
        // assertion here is that the snippet below would NOT compile
        // if the newtypes were aliased. We sanity-check serialization
        // behaviour: the wire form for `ViewId(7)` and `MaskId(7)`
        // must NOT round-trip across types.
        let view = ViewId(7);
        let mask_json = serde_json::to_string(&MaskId(7)).unwrap();
        let view_json = serde_json::to_string(&view).unwrap();
        // Both serialize their inner u32, but the surrounding handles
        // disambiguate (see `mask_bitmap_display_and_roundtrip` /
        // `view_storage_primary_display_and_roundtrip`).
        assert_eq!(view_json, mask_json);
        // A handle wrapping each gets distinct DataHandle JSON.
        let view_handle = DataHandle::ViewStorage {
            view,
            slot: ViewStorageSlot::Primary,
        };
        let mask_handle = DataHandle::MaskBitmap { mask: MaskId(7) };
        assert_ne!(
            serde_json::to_string(&view_handle).unwrap(),
            serde_json::to_string(&mask_handle).unwrap()
        );
    }
}
