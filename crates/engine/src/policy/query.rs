//! Query kinds for the `Ask` micro primitive.

use crate::ids::{AgentId, GroupId};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum QueryKind {
    AboutEntity(EntityQueryRef),
    AboutKind(MemoryKind),
    /// All facts — used by the `Read(doc)` sugar lowering.
    AboutAll,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EntityQueryRef {
    Agent(AgentId),
    Group(GroupId),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryKind {
    Combat    = 0,
    Trade     = 1,
    Social    = 2,
    Political = 3,
    Other     = 4,
}
