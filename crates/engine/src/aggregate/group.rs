use crate::ids::AgentId;
use smallvec::SmallVec;

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub kind_tag: u32,
    pub members:  SmallVec<[AgentId; 8]>,
    pub leader:   Option<AgentId>,
}

impl Group {
    pub fn empty(kind_tag: u32) -> Self {
        Self {
            kind_tag,
            members: SmallVec::new(),
            leader: None,
        }
    }
}
