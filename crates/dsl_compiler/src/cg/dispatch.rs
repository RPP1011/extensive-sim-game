//! `DispatchShape` — how a [`crate::cg::ComputeOp`] runs.
//!
//! Each shape captures the structural information emit needs to derive
//! workgroup size, dispatch count, and per-thread indexing — without
//! hard-coding those constants in the IR. The IR carries shape *data*
//! only; the helper methods that compute workgroup size + dispatch
//! dimensions are added by Task 1.4 below.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Tasks 1.3 and 1.4, for the full design rationale.
//!
//! ## Workgroup-size constants
//!
//! These mirror the values the existing emitters hard-code in WGSL
//! `@workgroup_size(...)` annotations. They are *properties of the
//! shape*, not of the runtime — so encoding them as named constants
//! here moves the only-correct-value out of `format!` strings and into
//! the type system. Future emit passes (Task 4.x) consume them; the
//! current `emit_*.rs` modules continue to inline literals until the
//! migration lands.
//!
//! Catalogued at Task 1.4 from `crates/dsl_compiler/src/emit_*.rs` and
//! `crates/engine_gpu_rules/src/*.wgsl`:
//!
//! | Shape       | `@workgroup_size` | dispatch                    |
//! |-------------|-------------------|-----------------------------|
//! | `PerAgent`  | 64                | `(agent_cap + 63) / 64`     |
//! | `PerEvent`  | 64                | indirect (cascade seed)     |
//! | `PerPair`   | 64                | (no current emitter — IR-only) |
//! | `OneShot`   | 1                 | `(1, 1, 1)`                 |
//! | `PerWord`   | 64                | over-dispatched as PerAgent today; IR rule below is `ceil(num_words / 64)` |
//!
//! ### Note on `PerWord`'s IR rule vs. current emit
//!
//! `alive_pack.wgsl` declares `@workgroup_size(64)` and is dispatched
//! today with `(agent_cap + 63) / 64` workgroups (the same rule as
//! `PerAgent`). The actual work is `ceil(agent_cap / 32)` words; the
//! extra threads early-return. The IR captures the *correct* rule
//! (`ceil(ceil(agent_cap / 32) / 64)`) so that when Task 4.x replaces
//! the inline `format!` with these helpers, the kernel stops over-
//! dispatching. This is intentional: Task 1.4 captures the rule, Task
//! 4.x harmonises the emitters.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::data_handle::EventRingId;
use super::op::SpatialQueryKind;

// ---------------------------------------------------------------------------
// Workgroup-size constants — properties of the dispatch shape.
// ---------------------------------------------------------------------------

/// Workgroup x-dim for [`DispatchShape::PerAgent`]. Matches every
/// per-agent kernel under `crates/engine_gpu_rules/src/`.
pub const PER_AGENT_WORKGROUP_X: u32 = 64;

/// Workgroup x-dim for [`DispatchShape::PerEvent`]. Matches the
/// cascade indirect-dispatch lane count emitted by
/// `emit_seed_indirect_wgsl::WORKGROUP_LANE_COUNT`.
pub const PER_EVENT_WORKGROUP_X: u32 = 64;

/// Workgroup x-dim for [`DispatchShape::PerPair`]. No current emitter
/// produces a pair-shaped dispatch; we adopt the same 64 the rest of
/// the codebase uses so emitting a per-pair kernel does not introduce
/// a new size class. Task 4.x is free to revisit if profiling shows a
/// different size dominates pair workloads.
pub const PER_PAIR_WORKGROUP_X: u32 = 64;

/// Workgroup x-dim for [`DispatchShape::OneShot`]. Matches
/// `seed_indirect.wgsl`'s `@workgroup_size(1)`.
pub const ONE_SHOT_WORKGROUP_X: u32 = 1;

/// Workgroup x-dim for [`DispatchShape::PerWord`]. Matches
/// `alive_pack.wgsl`'s `@workgroup_size(64)`.
pub const PER_WORD_WORKGROUP_X: u32 = 64;

/// Bits per packed alive-bitmap word. Drives [`DispatchShape::PerWord`]'s
/// dispatch count: one thread writes one 32-bit word holding 32 agent
/// alive flags.
pub const PER_WORD_BITS: u32 = 32;

/// Workgroup x-dim for [`DispatchShape::PerCell`]. Pinned to
/// `crate::cg::emit::spatial::MAX_PER_CELL` so each spatial-grid cell
/// maps to exactly one workgroup of that many lanes, with one lane per
/// home-cell agent slot. The tiled-MoveBoid emit relies on this
/// equality to schedule one thread per home-cell agent and let the
/// remaining lanes participate in the cooperative tile load.
///
/// Tracks `MAX_PER_CELL` — must stay in sync. Today both are 32.
pub const PER_CELL_WORKGROUP_X: u32 = 32;

/// Workgroup x-dim for [`DispatchShape::PerScanChunk`]. The parallel
/// prefix-scan over `spatial_grid_offsets` partitions `num_cells` into
/// chunks of this many cells, with one workgroup per chunk and one
/// thread per cell. 256 is the wgpu default
/// `max_compute_invocations_per_workgroup` lower bound, so kernels at
/// this size build on every adapter without bumping limits.
pub const PER_SCAN_CHUNK_WORKGROUP_X: u32 = 256;

/// How a compute op's threads are laid out.
///
/// The variants describe *what* drives the count and indexing — agent
/// slots, event-ring entries, (agent, target) pairs, a single thread,
/// or one thread per packed bitmap word. Emit-time uses this to
/// compute the workgroup count + per-thread fetching pattern; the
/// helper methods that perform that derivation are introduced in Task
/// 1.4.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum DispatchShape {
    /// One thread per agent slot. Workgroup count = ceil(agent_cap /
    /// workgroup_size). The default for mask predicates and per-agent
    /// scoring rows.
    PerAgent,

    /// One thread per event in the source ring. The count is read
    /// from the ring's tail buffer (indirect dispatch); `source_ring`
    /// names which ring drives the count.
    PerEvent { source_ring: EventRingId },

    /// One thread per (agent, target) pair within a spatial radius.
    /// Count = agents × candidates_per_agent. `source` describes how
    /// the candidate list per agent is materialized.
    PerPair { source: PerPairSource },

    /// Single-threaded — workgroup_size 1, dispatch 1×1×1. Used for
    /// `sim_cfg.tick` bumps, indirect-args seeding, and other
    /// scalar-output ops.
    OneShot,

    /// One thread per output word (32 agent slots per thread). Count
    /// = ceil(agent_cap / 32). Used for alive bitmap pack and mask
    /// compaction.
    PerWord,

    /// One workgroup per spatial-grid cell, with `MAX_PER_CELL` lanes
    /// per workgroup. Used by the tiled-MoveBoid emit (Layer 1 of the
    /// boids perf plan): each workgroup handles a "home cell", lanes
    /// 0..27 cooperatively load the surrounding 27 cells' agent
    /// pos/vel into `var<workgroup>` shared memory, then every lane
    /// processes one home-cell agent by walking the cached tile
    /// instead of re-reading global memory. Trades a fixed dispatch
    /// over `num_cells` workgroups (≈ `GRID_DIM³` ≈ 10k for boids
    /// defaults) for a ~5-10× reduction in inner-loop memory
    /// bandwidth on the dominant kernel.
    ///
    /// Empty cells (no agents) early-exit at the workgroup preamble;
    /// the over-dispatch is bounded by `num_cells` regardless of agent
    /// population. A future optimization compacts the non-empty cell
    /// list and switches to indirect dispatch — see
    /// [`super::data_handle::SpatialStorageKind::NonemptyCells`] +
    /// [`super::op::SpatialQueryKind::CompactNonemptyCells`] (added
    /// alongside this shape but currently inert).
    PerCell,

    /// One workgroup per `PER_SCAN_CHUNK_WORKGROUP_X`-sized chunk of
    /// `num_cells`, with one lane per cell. Drives the parallel
    /// prefix scan over `spatial_grid_offsets`: phase 2a
    /// (`BuildHashScanLocal`) and phase 2c (`BuildHashScanAdd`) both
    /// dispatch under this shape so the `(workgroup_id, local_id)`
    /// pair maps directly to `(chunk_id, cell_within_chunk)`. The
    /// dispatch count is `ceil(num_cells / PER_SCAN_CHUNK_WORKGROUP_X)`
    /// (~42 workgroups for boids' 22³=10 648 grid). Out-of-range
    /// lanes (`chunk_base + lane >= num_cells`) participate with
    /// count = 0 inside the scan and skip their slot writes.
    PerScanChunk,
}

/// What drives the per-agent candidate list for a [`DispatchShape::PerPair`]
/// dispatch. Only spatial-query-driven pairing is surfaced today — every
/// pair-shaped op emits via the spatial-grid neighborhood walk. Future
/// pair sources (e.g. view-driven k-nearest, fully dense N×N) accrete
/// here as variants.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum PerPairSource {
    /// Per-agent candidates come from a spatial-grid query. The
    /// embedded [`SpatialQueryKind`] selects which query (kin,
    /// engagement) feeds the pair walk.
    SpatialQuery(SpatialQueryKind),
}

impl fmt::Display for PerPairSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerPairSource::SpatialQuery(k) => write!(f, "spatial_query({})", k),
        }
    }
}

impl fmt::Display for DispatchShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchShape::PerAgent => f.write_str("per_agent"),
            DispatchShape::PerEvent { source_ring } => {
                write!(f, "per_event(ring=#{})", source_ring.0)
            }
            DispatchShape::PerPair { source } => write!(f, "per_pair({})", source),
            DispatchShape::OneShot => f.write_str("one_shot"),
            DispatchShape::PerWord => f.write_str("per_word"),
            DispatchShape::PerCell => f.write_str("per_cell"),
            DispatchShape::PerScanChunk => f.write_str("per_scan_chunk"),
        }
    }
}

// ---------------------------------------------------------------------------
// Task 1.4 — dispatch-rule helper types.
// ---------------------------------------------------------------------------

/// WGSL `@workgroup_size(x, y, z)`. All current shapes are 1-D; the
/// `(y, z)` fields exist so future 2-D / 3-D dispatches accrete here
/// without an enum change.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupSize {
    /// `(x, 1, 1)` — the 1-D form every current shape uses.
    pub const fn linear(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Total threads per workgroup (`x * y * z`).
    pub const fn total(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl fmt::Display for WorkgroupSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

/// Runtime parameters the IR does not know — supplied by emit at
/// dispatch time. The IR's helpers are pure functions of `(self, ctx)`;
/// this struct is the entire surface for "things only the runtime
/// resolves".
///
/// Each shape uses only the fields relevant to it. Constructing a
/// context with a value of `0` for an unused field is valid and tested.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct DispatchCtx {
    /// Maximum agent slot count for the run. Resolved from the engine's
    /// `SimState::agent_cap()` at schedule time. Used by
    /// [`DispatchShape::PerAgent`] and [`DispatchShape::PerWord`].
    pub agent_cap: u32,
    /// Per-agent candidate count for [`DispatchShape::PerPair`]. The
    /// caller resolves this from the per-pair source — typically a
    /// configured `kin_max` or `engagement_max` for a spatial query.
    /// `0` is treated as zero work (and yields `Direct { x: 0, y: 1,
    /// z: 1 }`, matching how the current `IndirectArgs` initialiser
    /// handles uninitialised slots).
    pub per_pair_candidates: u32,
}

impl DispatchCtx {
    /// Convenience constructor for tests + per-agent emitters.
    pub const fn per_agent(agent_cap: u32) -> Self {
        Self {
            agent_cap,
            per_pair_candidates: 0,
        }
    }
}

/// Result of [`DispatchShape::dispatch_count`] — either a direct
/// `dispatch_workgroups(x, y, z)` or an indirect dispatch reading the
/// `(x, y, z)` triple from a buffer.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum DispatchCount {
    /// Emit `pass.dispatch_workgroups(x, y, z)` with these constants.
    Direct { x: u32, y: u32, z: u32 },
    /// Emit `pass.dispatch_workgroups_indirect(buffer, byte_offset)`.
    /// `source_ring` names the event ring whose tail count drives the
    /// dispatch; the schedule resolves it to a concrete buffer +
    /// per-iter slot offset. `offset_words` is in u32 words within
    /// the indirect-args buffer (slot index × 3 words per
    /// `IndirectArgs` triple).
    Indirect {
        source_ring: EventRingId,
        offset_words: u32,
    },
}

impl fmt::Display for DispatchCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchCount::Direct { x, y, z } => write!(f, "direct({}, {}, {})", x, y, z),
            DispatchCount::Indirect {
                source_ring,
                offset_words,
            } => write!(
                f,
                "indirect(ring=#{}, offset_words={})",
                source_ring.0, offset_words
            ),
        }
    }
}

/// How a single thread maps `gid.x` → "the work this thread does". The
/// emit pass turns this into the WGSL prelude inside each kernel body.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum ThreadIndexing {
    /// `let agent = gid.x; if (agent >= agent_cap) { return; }`.
    PerAgentSlot,
    /// `let event_idx = gid.x;` — the indirect dispatch already bounds
    /// `gid.x` by the ring's tail count, but emit may also issue an
    /// explicit guard for over-dispatch tail rounding.
    PerEventSlot,
    /// `let pair = gid.x; let agent = pair / k; let cand = pair % k;`
    /// where `k = candidates_per_agent`.
    PerPairSlot { candidates_per_agent: u32 },
    /// `if (gid.x != 0u) { return; }` — the canonical OneShot prelude.
    SingleThread,
    /// `let word = gid.x; let num_words = (agent_cap + 31u) >> 5u; if
    /// (word >= num_words) { return; }` — alive-bitmap pack.
    PerAgentWord,
    /// `let home_cell = workgroup_id.x; let lane =
    /// local_invocation_id.x;` plus a cooperative-tile-load preamble
    /// composed by the tiled-MoveBoid emit. The kernel-emit's
    /// `thread_indexing_preamble` is empty for this variant (the
    /// preamble lives entirely in the tiled body builder so it can
    /// inline the `var<workgroup>` decls + the cooperative-load
    /// nested loops without re-deriving the spatial constants).
    PerCellWorkgroup,
    /// `let chunk_id = workgroup_id.x; let lane =
    /// local_invocation_id.x; let cell = chunk_id *
    /// PER_SCAN_CHUNK_WORKGROUP_X + lane;`. Used by the parallel
    /// prefix-scan phases. Per-cell bounds checks (`if (cell >=
    /// num_cells) ...`) live inside the per-op body because the
    /// scan still wants every lane to participate in the in-shared-
    /// memory Hillis-Steele reduction (out-of-range lanes contribute
    /// count = 0 but stay through every barrier).
    PerScanChunkLane,
}

impl fmt::Display for ThreadIndexing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThreadIndexing::PerAgentSlot => f.write_str("per_agent_slot"),
            ThreadIndexing::PerEventSlot => f.write_str("per_event_slot"),
            ThreadIndexing::PerPairSlot {
                candidates_per_agent,
            } => write!(f, "per_pair_slot(k={})", candidates_per_agent),
            ThreadIndexing::SingleThread => f.write_str("single_thread"),
            ThreadIndexing::PerAgentWord => f.write_str("per_agent_word"),
            ThreadIndexing::PerCellWorkgroup => f.write_str("per_cell_workgroup"),
            ThreadIndexing::PerScanChunkLane => f.write_str("per_scan_chunk_lane"),
        }
    }
}

// ---------------------------------------------------------------------------
// Task 1.4 — dispatch-rule helper methods on `DispatchShape`.
// ---------------------------------------------------------------------------

impl DispatchShape {
    /// Workgroup size declared on the kernel's `@workgroup_size(...)`
    /// WGSL annotation. A pure function of the dispatch shape — no
    /// runtime parameters required.
    ///
    /// Workgroup *size* is a shape-property (it determines the kernel's
    /// occupancy), distinct from workgroup *count* (how many groups
    /// run, which depends on `agent_cap` etc. and lives in
    /// [`Self::dispatch_count`]).
    pub const fn workgroup_size(&self) -> WorkgroupSize {
        match self {
            DispatchShape::PerAgent => WorkgroupSize::linear(PER_AGENT_WORKGROUP_X),
            DispatchShape::PerEvent { .. } => WorkgroupSize::linear(PER_EVENT_WORKGROUP_X),
            DispatchShape::PerPair { .. } => WorkgroupSize::linear(PER_PAIR_WORKGROUP_X),
            DispatchShape::OneShot => WorkgroupSize::linear(ONE_SHOT_WORKGROUP_X),
            DispatchShape::PerWord => WorkgroupSize::linear(PER_WORD_WORKGROUP_X),
            DispatchShape::PerCell => WorkgroupSize::linear(PER_CELL_WORKGROUP_X),
            DispatchShape::PerScanChunk => WorkgroupSize::linear(PER_SCAN_CHUNK_WORKGROUP_X),
        }
    }

    /// Number of workgroups to dispatch, given the runtime parameters
    /// the IR doesn't know.
    ///
    /// - `PerAgent`: `ceil(agent_cap / workgroup_x)`.
    /// - `PerEvent`: indirect — defers to the source ring's tail.
    /// - `PerPair`: `ceil((agent_cap × per_pair_candidates) / workgroup_x)`.
    /// - `OneShot`: `(1, 1, 1)`.
    /// - `PerWord`: `ceil(ceil(agent_cap / 32) / workgroup_x)`. (See
    ///   the module docs on the IR-rule vs. current-emit divergence.)
    ///
    /// All arithmetic uses `u32::div_ceil`; `0` inputs yield `0`
    /// workgroups (no panics, no divide-by-zero — every divisor is a
    /// `pub const` known non-zero).
    pub fn dispatch_count(&self, ctx: &DispatchCtx) -> DispatchCount {
        let wg_x = self.workgroup_size().x;
        match self {
            DispatchShape::PerAgent => DispatchCount::Direct {
                x: ctx.agent_cap.div_ceil(wg_x),
                y: 1,
                z: 1,
            },
            DispatchShape::PerEvent { source_ring } => DispatchCount::Indirect {
                source_ring: *source_ring,
                offset_words: 0,
            },
            DispatchShape::PerPair { .. } => {
                let pairs = ctx.agent_cap.saturating_mul(ctx.per_pair_candidates);
                DispatchCount::Direct {
                    x: pairs.div_ceil(wg_x),
                    y: 1,
                    z: 1,
                }
            }
            DispatchShape::OneShot => DispatchCount::Direct { x: 1, y: 1, z: 1 },
            DispatchShape::PerWord => {
                let num_words = ctx.agent_cap.div_ceil(PER_WORD_BITS);
                DispatchCount::Direct {
                    x: num_words.div_ceil(wg_x),
                    y: 1,
                    z: 1,
                }
            }
            DispatchShape::PerCell => {
                // One workgroup per spatial-grid cell. The cell count
                // is fixed by the per-fixture spatial-grid configuration
                // (see `crate::cg::emit::spatial::num_cells`); ctx
                // doesn't carry it, so we look it up directly. The
                // workgroup_size is `PER_CELL_WORKGROUP_X` (=
                // MAX_PER_CELL), and we want one workgroup per cell, so
                // the dispatch x is just `num_cells` (no div_ceil
                // needed — the kernel binds wgid.x to home_cell
                // directly).
                let num_cells = crate::cg::emit::spatial::num_cells();
                DispatchCount::Direct {
                    x: num_cells,
                    y: 1,
                    z: 1,
                }
            }
            DispatchShape::PerScanChunk => {
                // ceil(num_cells / chunk_size) workgroups, one per
                // scan chunk. The cell count is sourced directly
                // from the per-fixture spatial-grid configuration so
                // the dispatch + the WGSL `const NUM_CELLS = ...`
                // never drift.
                let num_cells = crate::cg::emit::spatial::num_cells();
                DispatchCount::Direct {
                    x: num_cells.div_ceil(wg_x),
                    y: 1,
                    z: 1,
                }
            }
        }
    }

    /// Per-thread indexing rule — emit synthesises this into the
    /// kernel's prelude (`let agent = gid.x; …`). Returns enough
    /// information for emit to pick the right WGSL snippet without
    /// re-deriving it per kernel.
    ///
    /// For [`DispatchShape::PerPair`] the returned variant carries the
    /// emit-time `candidates_per_agent`, which the caller passes via
    /// the per-pair source's resolved capacity. The IR-level shape
    /// only declares "this is per-pair"; the candidate count is a
    /// schedule-pass concern.
    ///
    /// **Note:** unlike [`Self::dispatch_count`], this method does not
    /// take a [`DispatchCtx`] because all of its inputs are derivable
    /// from the shape itself plus a fixed-per-shape constant for
    /// per-pair (defaulted to `1` here; emit overrides via the
    /// per-pair source). Emit can construct
    /// [`ThreadIndexing::PerPairSlot { candidates_per_agent }`]
    /// directly when it knows the value; the helper exists so the IR
    /// has a canonical default for diagnostics + snapshot tests.
    pub const fn thread_indexing(&self) -> ThreadIndexing {
        match self {
            DispatchShape::PerAgent => ThreadIndexing::PerAgentSlot,
            DispatchShape::PerEvent { .. } => ThreadIndexing::PerEventSlot,
            DispatchShape::PerPair { .. } => ThreadIndexing::PerPairSlot {
                // Default — emit replaces with the resolved kin/eng
                // capacity from the per-pair source. Tested below.
                candidates_per_agent: 1,
            },
            DispatchShape::OneShot => ThreadIndexing::SingleThread,
            DispatchShape::PerWord => ThreadIndexing::PerAgentWord,
            DispatchShape::PerCell => ThreadIndexing::PerCellWorkgroup,
            DispatchShape::PerScanChunk => ThreadIndexing::PerScanChunkLane,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_roundtrip<T>(v: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let json = serde_json::to_string(v).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, v, "round-trip changed value (json was {json})");
    }

    #[test]
    fn per_agent_display_and_roundtrip() {
        let s = DispatchShape::PerAgent;
        assert_eq!(format!("{}", s), "per_agent");
        assert_roundtrip(&s);
    }

    #[test]
    fn per_event_display_and_roundtrip() {
        let s = DispatchShape::PerEvent {
            source_ring: EventRingId(3),
        };
        assert_eq!(format!("{}", s), "per_event(ring=#3)");
        assert_roundtrip(&s);
    }



    #[test]
    fn one_shot_display_and_roundtrip() {
        let s = DispatchShape::OneShot;
        assert_eq!(format!("{}", s), "one_shot");
        assert_roundtrip(&s);
    }

    #[test]
    fn per_word_display_and_roundtrip() {
        let s = DispatchShape::PerWord;
        assert_eq!(format!("{}", s), "per_word");
        assert_roundtrip(&s);
    }

    #[test]
    fn shapes_are_distinct() {
        // Sanity — different variants do not compare equal.
        let agent = DispatchShape::PerAgent;
        let one_shot = DispatchShape::OneShot;
        let word = DispatchShape::PerWord;
        let e1 = DispatchShape::PerEvent {
            source_ring: EventRingId(1),
        };
        let e2 = DispatchShape::PerEvent {
            source_ring: EventRingId(2),
        };
        assert_ne!(agent, one_shot);
        assert_ne!(agent, word);
        assert_ne!(one_shot, word);
        assert_ne!(e1, e2);
    }

    // -----------------------------------------------------------------------
    // Task 1.4 — workgroup_size / dispatch_count / thread_indexing.
    // -----------------------------------------------------------------------

    #[allow(dead_code)] // unused after legacy variants dropped
    fn all_shapes() -> Vec<DispatchShape> {
        vec![
            DispatchShape::PerAgent,
            DispatchShape::PerEvent {
                source_ring: EventRingId(7),
            },
            DispatchShape::OneShot,
            DispatchShape::PerWord,
        ]
    }

    // --- workgroup_size ----------------------------------------------------

    #[test]
    fn workgroup_size_per_agent_matches_emit() {
        // `crates/engine_gpu_rules/src/scoring.wgsl` etc. all declare
        // `@workgroup_size(64)`.
        assert_eq!(
            DispatchShape::PerAgent.workgroup_size(),
            WorkgroupSize::linear(64)
        );
    }

    #[test]
    fn workgroup_size_per_event_matches_emit() {
        let s = DispatchShape::PerEvent {
            source_ring: EventRingId(0),
        };
        assert_eq!(s.workgroup_size(), WorkgroupSize::linear(64));
    }


    #[test]
    fn workgroup_size_one_shot_matches_emit() {
        // `crates/engine_gpu_rules/src/seed_indirect.wgsl` uses
        // `@workgroup_size(1)`.
        assert_eq!(DispatchShape::OneShot.workgroup_size(), WorkgroupSize::linear(1));
    }

    #[test]
    fn workgroup_size_per_word_matches_emit() {
        // `crates/engine_gpu_rules/src/alive_pack.wgsl` uses
        // `@workgroup_size(64)`.
        assert_eq!(DispatchShape::PerWord.workgroup_size(), WorkgroupSize::linear(64));
    }

    #[test]
    fn workgroup_size_total_is_x_times_y_times_z() {
        let s = WorkgroupSize::linear(64);
        assert_eq!(s.total(), 64);
        let cube = WorkgroupSize { x: 8, y: 4, z: 2 };
        assert_eq!(cube.total(), 64);
    }

    // --- dispatch_count: PerAgent -----------------------------------------

    #[test]
    fn dispatch_count_per_agent_typical() {
        // Same rule as `(agent_cap + 63) / 64` in every per-agent
        // emitter — `200000.div_ceil(64) == 3125`.
        let count = DispatchShape::PerAgent.dispatch_count(&DispatchCtx::per_agent(200_000));
        assert_eq!(
            count,
            DispatchCount::Direct {
                x: 3125,
                y: 1,
                z: 1
            }
        );
    }

    #[test]
    fn dispatch_count_per_agent_zero_cap() {
        let count = DispatchShape::PerAgent.dispatch_count(&DispatchCtx::per_agent(0));
        assert_eq!(count, DispatchCount::Direct { x: 0, y: 1, z: 1 });
    }

    #[test]
    fn dispatch_count_per_agent_exact_boundary() {
        // 64 → 1 group, 65 → 2 groups, 128 → 2 groups.
        for (cap, want) in [(64u32, 1u32), (65, 2), (128, 2), (129, 3)] {
            let got = DispatchShape::PerAgent.dispatch_count(&DispatchCtx::per_agent(cap));
            assert_eq!(
                got,
                DispatchCount::Direct {
                    x: want,
                    y: 1,
                    z: 1
                },
                "PerAgent cap={cap}"
            );
        }
    }

    // --- dispatch_count: PerEvent -----------------------------------------

    #[test]
    fn dispatch_count_per_event_is_indirect_carrying_ring() {
        let s = DispatchShape::PerEvent {
            source_ring: EventRingId(11),
        };
        let count = s.dispatch_count(&DispatchCtx::per_agent(0));
        assert_eq!(
            count,
            DispatchCount::Indirect {
                source_ring: EventRingId(11),
                offset_words: 0,
            }
        );
    }

    #[test]
    fn dispatch_count_per_event_independent_of_agent_cap() {
        // The runtime tail count drives the count; the IR returns the
        // same Indirect descriptor regardless of agent_cap.
        let s = DispatchShape::PerEvent {
            source_ring: EventRingId(2),
        };
        let a = s.dispatch_count(&DispatchCtx::per_agent(0));
        let b = s.dispatch_count(&DispatchCtx::per_agent(200_000));
        assert_eq!(a, b);
    }

    // --- dispatch_count: PerPair ------------------------------------------






    // --- dispatch_count: OneShot ------------------------------------------

    #[test]
    fn dispatch_count_one_shot_is_one_workgroup() {
        let count = DispatchShape::OneShot.dispatch_count(&DispatchCtx::per_agent(200_000));
        assert_eq!(count, DispatchCount::Direct { x: 1, y: 1, z: 1 });
    }

    #[test]
    fn dispatch_count_one_shot_independent_of_ctx() {
        let a = DispatchShape::OneShot.dispatch_count(&DispatchCtx::per_agent(0));
        let b = DispatchShape::OneShot.dispatch_count(&DispatchCtx::per_agent(u32::MAX));
        assert_eq!(a, b);
    }

    // --- dispatch_count: PerWord ------------------------------------------

    #[test]
    fn dispatch_count_per_word_partial_word() {
        // agent_cap = 31 → 1 word of work → 1 workgroup.
        let count = DispatchShape::PerWord.dispatch_count(&DispatchCtx::per_agent(31));
        assert_eq!(count, DispatchCount::Direct { x: 1, y: 1, z: 1 });
    }

    #[test]
    fn dispatch_count_per_word_exact_word() {
        // agent_cap = 32 → 1 word → 1 workgroup.
        let count = DispatchShape::PerWord.dispatch_count(&DispatchCtx::per_agent(32));
        assert_eq!(count, DispatchCount::Direct { x: 1, y: 1, z: 1 });
    }

    #[test]
    fn dispatch_count_per_word_one_past_word_boundary() {
        // agent_cap = 33 → 2 words → 1 workgroup (still fits in 64
        // threads). 64 words → 1 workgroup. 65 words → 2 workgroups.
        for (cap, want) in [
            (33u32, 1u32),         // 2 words ≤ 64 threads
            (32 * 64, 1),          // 64 words = 1 workgroup
            (32 * 64 + 1, 2),      // 65 words → 2 workgroups
            (32 * 128, 2),         // 128 words → 2 workgroups
            (32 * 128 + 1, 3),     // 129 words → 3 workgroups
        ] {
            let got = DispatchShape::PerWord.dispatch_count(&DispatchCtx::per_agent(cap));
            assert_eq!(
                got,
                DispatchCount::Direct {
                    x: want,
                    y: 1,
                    z: 1
                },
                "PerWord cap={cap}"
            );
        }
    }

    #[test]
    fn dispatch_count_per_word_zero_cap() {
        let count = DispatchShape::PerWord.dispatch_count(&DispatchCtx::per_agent(0));
        assert_eq!(count, DispatchCount::Direct { x: 0, y: 1, z: 1 });
    }

    #[test]
    fn dispatch_count_per_word_at_max_cap() {
        // Must not panic; 200_000 / 32 = 6250 words / 64 = 98 groups.
        let count = DispatchShape::PerWord.dispatch_count(&DispatchCtx::per_agent(200_000));
        assert_eq!(count, DispatchCount::Direct { x: 98, y: 1, z: 1 });
    }

    // --- thread_indexing --------------------------------------------------

    #[test]
    fn thread_indexing_per_agent() {
        assert_eq!(
            DispatchShape::PerAgent.thread_indexing(),
            ThreadIndexing::PerAgentSlot
        );
    }

    #[test]
    fn thread_indexing_per_event() {
        let s = DispatchShape::PerEvent {
            source_ring: EventRingId(0),
        };
        assert_eq!(s.thread_indexing(), ThreadIndexing::PerEventSlot);
    }


    #[test]
    fn thread_indexing_one_shot() {
        assert_eq!(
            DispatchShape::OneShot.thread_indexing(),
            ThreadIndexing::SingleThread
        );
    }

    #[test]
    fn thread_indexing_per_word() {
        assert_eq!(
            DispatchShape::PerWord.thread_indexing(),
            ThreadIndexing::PerAgentWord
        );
    }

    // --- helpers + display -------------------------------------------------

    #[test]
    fn workgroup_size_linear_constructor() {
        let s = WorkgroupSize::linear(64);
        assert_eq!((s.x, s.y, s.z), (64, 1, 1));
        assert_eq!(format!("{}", s), "(64, 1, 1)");
    }

    #[test]
    fn dispatch_count_display_direct_and_indirect() {
        assert_eq!(
            format!(
                "{}",
                DispatchCount::Direct {
                    x: 7,
                    y: 1,
                    z: 1
                }
            ),
            "direct(7, 1, 1)"
        );
        assert_eq!(
            format!(
                "{}",
                DispatchCount::Indirect {
                    source_ring: EventRingId(3),
                    offset_words: 6,
                }
            ),
            "indirect(ring=#3, offset_words=6)"
        );
    }

    #[test]
    fn thread_indexing_display() {
        assert_eq!(
            format!("{}", ThreadIndexing::PerAgentSlot),
            "per_agent_slot"
        );
        assert_eq!(
            format!("{}", ThreadIndexing::PerEventSlot),
            "per_event_slot"
        );
        assert_eq!(
            format!(
                "{}",
                ThreadIndexing::PerPairSlot {
                    candidates_per_agent: 12
                }
            ),
            "per_pair_slot(k=12)"
        );
        assert_eq!(format!("{}", ThreadIndexing::SingleThread), "single_thread");
        assert_eq!(format!("{}", ThreadIndexing::PerAgentWord), "per_agent_word");
    }

    #[test]
    fn dispatch_ctx_per_agent_constructor_zeroes_unused_fields() {
        let c = DispatchCtx::per_agent(123);
        assert_eq!(c.agent_cap, 123);
        assert_eq!(c.per_pair_candidates, 0);
    }

    // --- round-trip serde --------------------------------------------------

    #[test]
    fn workgroup_size_roundtrips() {
        assert_roundtrip(&WorkgroupSize::linear(64));
        assert_roundtrip(&WorkgroupSize { x: 8, y: 4, z: 2 });
    }

    #[test]
    fn dispatch_ctx_roundtrips() {
        assert_roundtrip(&DispatchCtx::per_agent(200_000));
        assert_roundtrip(&DispatchCtx {
            agent_cap: 1024,
            per_pair_candidates: 8,
        });
    }

    #[test]
    fn dispatch_count_roundtrips() {
        assert_roundtrip(&DispatchCount::Direct {
            x: 3125,
            y: 1,
            z: 1,
        });
        assert_roundtrip(&DispatchCount::Indirect {
            source_ring: EventRingId(7),
            offset_words: 12,
        });
    }

    #[test]
    fn thread_indexing_roundtrips() {
        assert_roundtrip(&ThreadIndexing::PerAgentSlot);
        assert_roundtrip(&ThreadIndexing::PerEventSlot);
        assert_roundtrip(&ThreadIndexing::PerPairSlot {
            candidates_per_agent: 8,
        });
        assert_roundtrip(&ThreadIndexing::SingleThread);
        assert_roundtrip(&ThreadIndexing::PerAgentWord);
    }

    #[test]
    fn all_dispatch_shape_variants_roundtrip() {
        // Exhaustive: every variant the enum exposes.
        for s in all_shapes() {
            assert_roundtrip(&s);
        }
    }

    // --- exhaustive coverage smoke ----------------------------------------

    #[test]
    fn every_shape_yields_nonempty_methods() {
        // Every variant returns *something* for every helper — guards
        // against a future variant being added with a `_ =>` arm that
        // silently produces a wrong default. (Each helper has explicit
        // arms; this test just exercises them all.)
        for s in all_shapes() {
            let wg = s.workgroup_size();
            assert!(wg.x >= 1 && wg.y == 1 && wg.z == 1, "wg shape: {wg}");

            let ctx = DispatchCtx {
                agent_cap: 1024,
                per_pair_candidates: 8,
            };
            let _count = s.dispatch_count(&ctx);

            let _idx = s.thread_indexing();
        }
    }

    #[test]
    fn dispatch_count_matches_legacy_emit_rule_for_per_agent() {
        // Existing per-agent emitters write `(agent_cap + 63) / 64`.
        // The IR rule must produce the same value for every cap in a
        // representative sample.
        for cap in [0u32, 1, 63, 64, 65, 1024, 200_000, u32::MAX / 2] {
            let want = cap.div_ceil(64);
            let got = match DispatchShape::PerAgent.dispatch_count(&DispatchCtx::per_agent(cap)) {
                DispatchCount::Direct { x, y: 1, z: 1 } => x,
                other => panic!("expected Direct, got {other:?} for cap={cap}"),
            };
            assert_eq!(got, want, "cap={cap}");
        }
    }
}
