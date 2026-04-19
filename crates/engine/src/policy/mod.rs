// crates/engine/src/policy/mod.rs
pub mod utility;

use crate::ids::AgentId;
use crate::mask::MaskBuffer;
use crate::mask::MicroKind;
use crate::state::SimState;
pub use utility::UtilityBackend;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Action {
    pub agent:      AgentId,
    pub micro_kind: MicroKind,
    pub target:     Option<AgentId>,
}

pub trait PolicyBackend {
    /// Evaluate policy for every alive agent; append to `out`.
    /// Caller resets `out` before calling (zero-alloc contract).
    fn evaluate(&self, state: &SimState, mask: &MaskBuffer, out: &mut Vec<Action>);
}
