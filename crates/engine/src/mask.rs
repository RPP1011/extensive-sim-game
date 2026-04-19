// crates/engine/src/mask.rs
use crate::state::SimState;

pub const TARGET_SLOTS: usize = 12;  // matches nearby_actors K=12 per spec §9 D5

/// Radius within which another agent counts as a threat for flee-mask purposes.
const AGGRO_RANGE: f32 = 50.0;

/// Mirrors `ATTACK_RANGE` in `step.rs`. Both constants MUST move together —
/// mask permissiveness vs resolution cutoff must agree or the policy will
/// choose moves the kernel silently drops.
const ATTACK_RANGE_FOR_MASK: f32 = 2.0;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum MicroKind {
    // Movement (3)
    Hold        = 0,
    MoveToward  = 1,
    Flee        = 2,
    // Combat (3)
    Attack      = 3,
    Cast        = 4,
    UseItem     = 5,
    // Resource (4)
    Harvest     = 6,
    Eat         = 7,
    Drink       = 8,
    Rest        = 9,
    // Construction (3)
    PlaceTile    = 10,
    PlaceVoxel   = 11,
    HarvestVoxel = 12,
    // Social (2)
    Converse     = 13,
    ShareStory   = 14,
    // Info push + pull (2)
    Communicate  = 15,
    Ask          = 16,
    // Memory (1)
    Remember     = 17,
}

impl MicroKind {
    pub const ALL: &'static [MicroKind] = &[
        MicroKind::Hold,         MicroKind::MoveToward,   MicroKind::Flee,
        MicroKind::Attack,       MicroKind::Cast,         MicroKind::UseItem,
        MicroKind::Harvest,      MicroKind::Eat,          MicroKind::Drink,
        MicroKind::Rest,         MicroKind::PlaceTile,    MicroKind::PlaceVoxel,
        MicroKind::HarvestVoxel, MicroKind::Converse,     MicroKind::ShareStory,
        MicroKind::Communicate,  MicroKind::Ask,          MicroKind::Remember,
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
    pub fn mark_move_allowed_if_others_exist(&mut self, state: &SimState) {
        let n_alive = state.agents_alive().count();
        if n_alive < 2 { return; }
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let offset = slot * MicroKind::ALL.len() + MicroKind::MoveToward as usize;
            self.micro_kind[offset] = true;
        }
    }

    /// Mark `Flee` as allowed for every alive agent that has at least one
    /// other alive agent within `AGGRO_RANGE`. No threat → no flee.
    pub fn mark_flee_allowed_if_threat_exists(&mut self, state: &SimState) {
        let n_kinds = MicroKind::ALL.len();
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let self_pos = match state.agent_pos(id) {
                Some(p) => p,
                None    => continue,
            };
            let has_threat = state.agents_alive().any(|other| {
                other != id && state.agent_pos(other)
                    .map(|op| op.distance(self_pos) <= AGGRO_RANGE)
                    .unwrap_or(false)
            });
            if has_threat {
                let offset = slot * n_kinds + MicroKind::Flee as usize;
                self.micro_kind[offset] = true;
            }
        }
    }

    /// Mark `Eat`, `Drink`, and `Rest` as always-allowed for every alive agent.
    /// MVP: no world-state preconditions (e.g. food availability, rest site) —
    /// those land when the inventory/site systems arrive.
    pub fn mark_needs_allowed(&mut self, state: &SimState) {
        let n_kinds = MicroKind::ALL.len();
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            self.micro_kind[slot * n_kinds + MicroKind::Eat   as usize] = true;
            self.micro_kind[slot * n_kinds + MicroKind::Drink as usize] = true;
            self.micro_kind[slot * n_kinds + MicroKind::Rest  as usize] = true;
        }
    }

    /// Mark `Attack` as allowed for every alive agent that has at least one
    /// other alive agent within `ATTACK_RANGE_FOR_MASK`. No target in range →
    /// no attack.
    pub fn mark_attack_allowed_if_target_in_range(&mut self, state: &SimState) {
        let n_kinds = MicroKind::ALL.len();
        for id in state.agents_alive() {
            let slot = (id.raw() - 1) as usize;
            let self_pos = match state.agent_pos(id) {
                Some(p) => p,
                None    => continue,
            };
            let has_target = state.agents_alive().any(|other| {
                other != id && state.agent_pos(other)
                    .map(|op| op.distance(self_pos) <= ATTACK_RANGE_FOR_MASK)
                    .unwrap_or(false)
            });
            if has_target {
                let offset = slot * n_kinds + MicroKind::Attack as usize;
                self.micro_kind[offset] = true;
            }
        }
    }
}
