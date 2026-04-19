// crates/engine/src/mask.rs
use crate::state::SimState;

pub const TARGET_SLOTS: usize = 12;  // matches nearby_actors K=12 per spec §9 D5

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MicroKind {
    Hold        = 0,
    MoveToward  = 1,
    Attack      = 2,
    Eat         = 3,
}

impl MicroKind {
    pub const ALL: &'static [MicroKind] = &[
        MicroKind::Hold, MicroKind::MoveToward, MicroKind::Attack, MicroKind::Eat,
    ];
}

pub struct MaskBuffer {
    pub micro_kind: Vec<bool>,     // [N_agents × NUM_MICRO]
    pub target:     Vec<bool>,     // [N_agents × TARGET_SLOTS]
}

impl MaskBuffer {
    pub fn new(n_agents: usize) -> Self {
        Self {
            micro_kind: vec![false; n_agents * MicroKind::ALL.len()],
            target:     vec![false; n_agents * TARGET_SLOTS],
        }
    }
    pub fn reset(&mut self) {
        self.micro_kind.iter_mut().for_each(|b| *b = false);
        self.target.iter_mut().for_each(|b| *b = false);
    }
    pub fn mark_hold_allowed(&mut self, state: &SimState) {
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let offset = slot * MicroKind::ALL.len() + MicroKind::Hold as usize;
            self.micro_kind[offset] = true;
        }
    }
}
