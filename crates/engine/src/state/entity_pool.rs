//! Historical location of `AgentSlotPool`. Now a type alias over `Pool<AgentTag>`.
//! The `alloc_agent` / `kill_agent` / `slot_of_agent` shims preserve the
//! existing call-site shape (`AgentSlotPool::alloc() -> Option<AgentId>`).

use crate::ids::AgentId;
use crate::pool::{Pool, PoolId};

pub struct AgentTag;
pub type AgentSlotPool = Pool<AgentTag>;

pub trait AgentPoolOps {
    fn alloc_agent(&mut self) -> Option<AgentId>;
    fn kill_agent(&mut self, id: AgentId);
}

impl AgentPoolOps for AgentSlotPool {
    fn alloc_agent(&mut self) -> Option<AgentId> {
        let pid: PoolId<AgentTag> = self.alloc()?;
        AgentId::new(pid.raw())
    }
    fn kill_agent(&mut self, id: AgentId) {
        if let Some(p) = PoolId::<AgentTag>::new(id.raw()) {
            self.kill(p);
        }
    }
}

// Inherent impl so existing call sites `AgentSlotPool::slot_of_agent(id)` compile.
impl AgentSlotPool {
    #[inline]
    pub fn slot_of_agent(id: AgentId) -> usize {
        (id.raw() - 1) as usize
    }
}
