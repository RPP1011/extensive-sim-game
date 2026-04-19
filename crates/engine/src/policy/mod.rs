// crates/engine/src/policy/mod.rs
pub mod macro_kind;
pub mod query;
pub mod utility;

use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use crate::state::SimState;
use glam::Vec3;

pub use macro_kind::{
    AnnounceAudience, MacroAction, MacroKind, QuestCategory, Resolution,
};
pub use query::{EntityQueryRef, MemoryKind, QueryKind};
pub use utility::UtilityBackend;

/// Action emitted by a `PolicyBackend`. Either a micro primitive (the 18-kind
/// universal set from Appendix A) or a parameterised macro (post-quest, bid,
/// announce, …). See `docs/engine/runtime_contract.md` §3 for the split.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Action {
    pub agent: AgentId,
    pub kind:  ActionKind,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ActionKind {
    Micro { kind: MicroKind, target: MicroTarget },
    Macro(MacroAction),
}

/// Universal target parameter for a micro action. `Opaque(u64)` is the
/// extension hatch — compiler-registered tables decode handles for
/// domain-specific targets (UseItem slot, Harvest node, PlaceVoxel cell,
/// Document id, …). The engine itself only knows the universal branches.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MicroTarget {
    None,
    Agent(AgentId),
    Position(Vec3),
    ItemSlot(u8),
    AbilityIdx(u8),
    Query(QueryKind),
    /// Extension hatch — UseItem/Harvest/PlaceVoxel/Document targets decode
    /// via compiler-registered tables keyed on an opaque u64 handle.
    Opaque(u64),
}

impl Action {
    /// Hold in place — no target.
    pub fn hold(agent: AgentId) -> Self {
        Self {
            agent,
            kind: ActionKind::Micro {
                kind:   MicroKind::Hold,
                target: MicroTarget::None,
            },
        }
    }

    /// Move toward a world-space position. Backends resolve the destination
    /// (e.g. nearest-other lookup) at emission time.
    pub fn move_toward(agent: AgentId, pos: Vec3) -> Self {
        Self {
            agent,
            kind: ActionKind::Micro {
                kind:   MicroKind::MoveToward,
                target: MicroTarget::Position(pos),
            },
        }
    }

    /// Attack a specific agent.
    pub fn attack(agent: AgentId, target: AgentId) -> Self {
        Self {
            agent,
            kind: ActionKind::Micro {
                kind:   MicroKind::Attack,
                target: MicroTarget::Agent(target),
            },
        }
    }

    /// Eat from own inventory — no target (item slot resolution in later tasks).
    pub fn eat(agent: AgentId) -> Self {
        Self {
            agent,
            kind: ActionKind::Micro {
                kind:   MicroKind::Eat,
                target: MicroTarget::None,
            },
        }
    }

    /// Convenience: extract the micro kind if this is a micro action.
    pub fn micro_kind(&self) -> Option<MicroKind> {
        match self.kind {
            ActionKind::Micro { kind, .. } => Some(kind),
            ActionKind::Macro(_) => None,
        }
    }
}

pub trait PolicyBackend {
    /// Evaluate policy for every alive agent; append to `out`.
    /// Caller resets `out` before calling (zero-alloc contract).
    fn evaluate(&self, state: &SimState, mask: &MaskBuffer, out: &mut Vec<Action>);
}
