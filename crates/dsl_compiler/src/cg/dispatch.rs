//! `DispatchShape` — how a [`crate::cg::ComputeOp`] runs.
//!
//! Each shape captures the structural information emit needs to derive
//! workgroup size, dispatch count, and per-thread indexing — without
//! hard-coding those constants in the IR. The IR carries shape *data*
//! only; the helper methods that compute workgroup size + dispatch
//! dimensions land in Task 1.4. Until then, the variants and their
//! payloads are first-class so [`crate::cg::ComputeOp`] (Task 1.3) can
//! reference them.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Tasks 1.3 and 1.4, for the full design rationale. The variant set
//! here is the one specified in the plan (lines 324–344); the
//! dispatch-rule helper methods are the deferred work.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::data_handle::EventRingId;
use super::op::SpatialQueryKind;

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
    fn per_pair_kin_display_and_roundtrip() {
        let s = DispatchShape::PerPair {
            source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
        };
        assert_eq!(format!("{}", s), "per_pair(spatial_query(kin_query))");
        assert_roundtrip(&s);
    }

    #[test]
    fn per_pair_engagement_display_and_roundtrip() {
        let s = DispatchShape::PerPair {
            source: PerPairSource::SpatialQuery(SpatialQueryKind::EngagementQuery),
        };
        assert_eq!(
            format!("{}", s),
            "per_pair(spatial_query(engagement_query))"
        );
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
}
