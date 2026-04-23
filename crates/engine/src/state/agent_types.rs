//! Stub types for `SimState` cold-collection fields. These are minimal
//! Pod-friendly shells that reserve the shape state.md commits to; subsequent
//! plans attach typed payloads and cascade handlers. See
//! `docs/superpowers/plans/2026-04-19-engine-plan-state-port.md` for the
//! full inventory and rationale.

use crate::ids::{AgentId, GroupId};

// ---- Task C: StatusEffect ------------------------------------------------

/// Kind discriminator for a `StatusEffect`. state.md §StatusEffect commits to
/// the set {Stun, Slow, Root, Silence, Dot, Hot, Buff, Debuff}; kind-specific
/// payloads live in `StatusEffect::payload_q8` (an opaque i16 slot reserved
/// for a later, typed encoding).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum StatusEffectKind {
    Stun    = 0,
    Slow    = 1,
    Root    = 2,
    Silence = 3,
    Dot     = 4,
    Hot     = 5,
    Buff    = 6,
    Debuff  = 7,
}

/// A single status-effect entry on an agent. `payload_q8` is a
/// kind-dispatched opaque slot (q8 fixed-point in most kinds — e.g. slow
/// factor, buff magnitude); the decoder is a later plan's concern.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StatusEffect {
    pub kind:            StatusEffectKind,
    pub source:          AgentId,
    pub remaining_ticks: u32,
    pub payload_q8:      i16,
}

// ---- Task G: Membership --------------------------------------------------

/// Role an agent holds inside a `Group`. state.md §Membership enumerates
/// `{Member, Officer, Leader, Founder, Apprentice, Outcast}`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum GroupRole {
    Member     = 0,
    Officer    = 1,
    Leader     = 2,
    Founder    = 3,
    Apprentice = 4,
    Outcast    = 5,
}

/// One group membership. An agent can hold multiple (§Memberships: faction,
/// family, guild, religion, party, pack, settlement — emergent loyalty
/// conflicts).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Membership {
    pub group:       GroupId,
    pub role:        GroupRole,
    pub joined_tick: u32,
    pub standing_q8: i16,
}

// ---- Task H: Inventory ---------------------------------------------------

/// Portable commodity storage, one per agent. `gold` is signed (`i32`) so debt
/// is representable as a negative balance. Narrowed from i64 on 2026-04-22 so
/// `transfer_gold` can run on GPU (WGSL has no atomic i64). i32's ±2.1B range
/// is well above any practical economic-sim value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Inventory {
    pub gold:        i32,
    pub commodities: [u16; 8],
}

// ---- Task I: MemoryEvent -------------------------------------------------

/// Opaque-payload memory entry. state.md §MemoryEvent commits to richer
/// fields (tick, MemEventType, location, entity_ids, emotional_impact,
/// Source, confidence). This is a Pod-friendly shell; the compiler attaches
/// typed payloads via `kind` + `payload` later.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryEvent {
    pub source:        AgentId,
    pub kind:          u8,
    pub payload:       u64,
    pub confidence_q8: u8,
    pub tick:          u32,
}

// ---- Task J: Relationship ------------------------------------------------

/// Directional pair-wise sentiment. state.md §Relationship carries richer
/// fields (trust, familiarity, last_interaction, perceived_personality,
/// believed_knowledge); this is the MVP shell — a q8 valence plus tenure.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Relationship {
    pub other:         AgentId,
    pub valence_q8:    i16,
    pub tenure_ticks:  u32,
}

// ---- Task K: Misc cold ---------------------------------------------------

/// A single class slot (`classes`, state.md §AgentData.Skill & Class).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ClassSlot {
    pub class_tag: u32,
    pub level:     u8,
}

/// One creditor on the debt ledger (§AgentData.Economic `creditor_id`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Creditor {
    pub creditor: AgentId,
    pub amount:   u32,
}

/// One mentor link (§AgentData.Relationships `mentor_lineage`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MentorLink {
    pub mentor:     AgentId,
    pub discipline: u8,
}

// ---- Combat Foundation Task 1: SparseStandings --------------------------

/// Per-pair standing ledger. Keyed by the ordered tuple `(min(a, b), max(a, b))`
/// so lookups and updates are symmetric (standing is a mutual quantity). Value
/// is an `i16` clamped to `[-1000, 1000]` by `adjust(..)`.
///
/// Cold storage: read only when a pair-wise interaction references it (e.g.
/// `EffectOp::ModifyStanding`, gate predicates that defer to standing).
#[derive(Clone, Debug, Default)]
pub struct SparseStandings {
    pairs: std::collections::BTreeMap<(AgentId, AgentId), i16>,
}

impl SparseStandings {
    pub fn new() -> Self { Self { pairs: std::collections::BTreeMap::new() } }

    #[inline]
    fn key(a: AgentId, b: AgentId) -> (AgentId, AgentId) {
        if a <= b { (a, b) } else { (b, a) }
    }

    /// Current standing for the pair. Returns 0 when no entry has been set.
    pub fn get(&self, a: AgentId, b: AgentId) -> i16 {
        self.pairs.get(&Self::key(a, b)).copied().unwrap_or(0)
    }

    /// Replace the standing for the pair.
    pub fn set(&mut self, a: AgentId, b: AgentId, v: i16) {
        self.pairs.insert(Self::key(a, b), v);
    }

    /// Adjust by `delta`, saturating at `[-1000, 1000]`. Returns the new
    /// clamped value.
    pub fn adjust(&mut self, a: AgentId, b: AgentId, delta: i16) -> i16 {
        let k = Self::key(a, b);
        let cur = self.pairs.get(&k).copied().unwrap_or(0) as i32;
        let new = (cur + delta as i32).clamp(-1000, 1000) as i16;
        self.pairs.insert(k, new);
        new
    }

    pub fn is_empty(&self) -> bool { self.pairs.is_empty() }
    pub fn len(&self) -> usize { self.pairs.len() }
}
