use crate::ids::AgentId;

pub struct AgentSlotPool {
    cap:      u32,
    next:     u32,
    pub alive: Vec<bool>,
}

impl AgentSlotPool {
    pub fn new(cap: u32) -> Self {
        Self { cap, next: 1, alive: vec![false; cap as usize] }
    }
    pub fn alloc(&mut self) -> Option<AgentId> {
        if self.next > self.cap { return None; }
        let id = AgentId::new(self.next)?;
        let slot = (self.next - 1) as usize;
        self.alive[slot] = true;
        self.next += 1;
        Some(id)
    }
    pub fn kill(&mut self, id: AgentId) {
        let slot = (id.raw() - 1) as usize;
        if slot < self.cap as usize {
            self.alive[slot] = false;
        }
    }
    #[inline] pub fn slot_of(id: AgentId) -> usize { (id.raw() - 1) as usize }
}
