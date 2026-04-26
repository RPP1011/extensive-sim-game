// crates/engine/src/mask.rs
//
// Storage primitives for the action mask: MicroKind enum, TargetMask, and
// MaskBuffer. Rule-aware mask-build methods (mark_hold_allowed,
// mark_move_allowed_from_candidates, mark_attack_allowed_from_candidates,
// mark_flee_allowed_if_threat_exists, mark_needs_allowed,
// mark_domain_hook_micros_allowed) and the inferred_cast_target helper are
// DELETED in Plan B1' Task 11 — they called into generated mask functions and
// belong to the rule layer. `engine_rules::step` (emitted in Task 11) will
// own those calls. Only storage primitives remain here.
use crate::ids::AgentId;
use smallvec::SmallVec;

pub const TARGET_SLOTS: usize = 12;  // matches nearby_actors K=12 per spec §9 D5

// `AGGRO_RANGE` (flee-threat radius) and the attack spatial iterator floor
// now live in `assets/sim/config.sim` as `config.combat.aggro_range` and
// `config.combat.attack_range` respectively. Callers below read them off
// `state.config.combat.*`.

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

    /// Target-bound kinds that consume a `TargetMask` candidate list (task
    /// 138). Self-only kinds (Hold / Flee / Eat / Drink / Rest / … the
    /// zero-arg domain hooks) read no candidates and are scored once per
    /// agent. Attack and MoveToward are the only target-bound kinds at
    /// v1 — `Cast`'s target is still inferred via `inferred_cast_target`
    /// in mask-build, and the other targeted heads (Converse /
    /// Communicate / Ask / …) stay permissive until their DSL mask
    /// declarations land.
    pub const TARGET_BOUND: &'static [MicroKind] = &[
        MicroKind::Attack,
        MicroKind::MoveToward,
    ];

    /// Dense 0-based index into `TARGET_BOUND` for this kind, or `None` for
    /// self-only kinds. Used by `TargetMask` to address per-kind candidate
    /// lists.
    pub fn target_slot(self) -> Option<usize> {
        match self {
            MicroKind::Attack => Some(0),
            MicroKind::MoveToward => Some(1),
            _ => None,
        }
    }
}

/// Per-agent, per-target-bound-kind list of valid targets. Task 138 — the
/// scorer argmaxes over these candidate lists for every targeted
/// `MicroKind`, replacing the old `nearest_other` heuristic. Self-only
/// kinds (Hold / Flee / Eat / …) read no candidates and are scored once
/// per agent.
///
/// Storage shape: flat `Vec<SmallVec<...>>` indexed by
/// `agent_slot * TARGET_BOUND.len() + kind.target_slot()`. Inline
/// capacity is 8 per (agent, kind) pair — enough for a small melee mob
/// without spilling to the heap in the common case.
pub struct TargetMask {
    pub candidates: Vec<SmallVec<[AgentId; 8]>>,
}

impl TargetMask {
    pub fn new(n_agents: usize) -> Self {
        let slots = n_agents * MicroKind::TARGET_BOUND.len();
        let mut candidates = Vec::with_capacity(slots);
        for _ in 0..slots {
            candidates.push(SmallVec::new());
        }
        Self { candidates }
    }

    /// Clear every per-agent candidate list; called once per tick before
    /// the target-mask-build phase re-populates them.
    pub fn reset(&mut self) {
        for c in self.candidates.iter_mut() {
            c.clear();
        }
    }

    fn slot(&self, agent: AgentId, kind: MicroKind) -> Option<usize> {
        let agent_slot = (agent.raw() - 1) as usize;
        let kind_slot = kind.target_slot()?;
        Some(agent_slot * MicroKind::TARGET_BOUND.len() + kind_slot)
    }

    /// Push a candidate target into the list for `(agent, kind)`. Called
    /// by the compiler-emitted `mask_<name>_candidates` fns.
    pub fn push(&mut self, agent: AgentId, kind: MicroKind, target: AgentId) {
        if let Some(i) = self.slot(agent, kind) {
            if let Some(list) = self.candidates.get_mut(i) {
                list.push(target);
            }
        }
    }

    /// Read the candidate list for `(agent, kind)`. Empty for self-only
    /// kinds and for (agent, targeted-kind) pairs with no hits.
    pub fn candidates_for(&self, agent: AgentId, kind: MicroKind) -> &[AgentId] {
        match self.slot(agent, kind) {
            Some(i) => self.candidates.get(i).map_or(&[], |v| v.as_slice()),
            None => &[],
        }
    }
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

    /// Set the mask bit for `(agent_slot, kind)` directly. Used by
    /// `engine_rules` mask-build code (Task 11) and low-level tests that
    /// need to write a specific bit without the rule-aware mark_* helpers
    /// (which are deleted — they called generated mask fns from engine_rules).
    pub fn set(&mut self, agent_slot: usize, kind: MicroKind, value: bool) {
        let offset = agent_slot * MicroKind::ALL.len() + kind as usize;
        if let Some(b) = self.micro_kind.get_mut(offset) {
            *b = value;
        }
    }

    /// Read the mask bit for `(agent_slot, kind)`.
    pub fn get(&self, agent_slot: usize, kind: MicroKind) -> bool {
        let offset = agent_slot * MicroKind::ALL.len() + kind as usize;
        self.micro_kind.get(offset).copied().unwrap_or(false)
    }

    /// Number of agents this buffer was sized for.
    pub fn n_agents(&self) -> u32 {
        (self.micro_kind.len() / MicroKind::ALL.len()) as u32
    }

    /// Number of micro-kind slots per agent.
    pub fn n_kinds(&self) -> u32 {
        MicroKind::ALL.len() as u32
    }

    /// Raw read-only view of the micro-kind bit matrix
    /// (length = n_agents × n_kinds, row-major by agent).
    pub fn bits(&self) -> &[bool] {
        &self.micro_kind
    }
}
