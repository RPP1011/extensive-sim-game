//! Universal macro mechanisms. Language-level macros are the 4 variants
//! `PostQuest`, `AcceptQuest`, `Bid`, `Announce`; `NoOp` occupies slot 0 so
//! `macro_kind != NoOp` cleanly distinguishes macro from micro emission.

use crate::ids::{AgentId, GroupId, QuestId};
use glam::Vec3;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum MacroKind {
    NoOp        = 0,
    PostQuest   = 1,
    AcceptQuest = 2,
    Bid         = 3,
    Announce    = 4,
}

impl MacroKind {
    pub const ALL: &'static [MacroKind] = &[
        MacroKind::NoOp, MacroKind::PostQuest, MacroKind::AcceptQuest,
        MacroKind::Bid,  MacroKind::Announce,
    ];
}

/// Recipient scope for `Announce`. Matches `dsl/spec.md` §3.2.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AnnounceAudience {
    Group(GroupId),
    Area(Vec3, f32),
    Anyone,
}

impl AnnounceAudience {
    pub fn tag(&self) -> u8 {
        match self {
            AnnounceAudience::Group(_)   => 0,
            AnnounceAudience::Area(_, _) => 1,
            AnnounceAudience::Anyone     => 2,
        }
    }
}

// `Resolution` and `QuestCategory` live in `engine_data::types`.
// Task 4 (Plan B1') dropped the re-export shims; callers import directly
// from `engine_data::types`.
use engine_data::types::{QuestCategory, Resolution};

/// Parameterised macro action emitted by a policy.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MacroAction {
    NoOp,
    PostQuest {
        quest_id:   QuestId,
        category:   QuestCategory,
        resolution: Resolution,
    },
    AcceptQuest {
        quest_id: QuestId,
        acceptor: AgentId,
    },
    Bid {
        auction_id: QuestId,
        bidder:     AgentId,
        amount:     f32,
    },
    Announce {
        speaker:      AgentId,
        audience:     AnnounceAudience,
        fact_payload: u64,
    },
}

impl MacroAction {
    pub fn kind(&self) -> MacroKind {
        match self {
            MacroAction::NoOp => MacroKind::NoOp,
            MacroAction::PostQuest   { .. } => MacroKind::PostQuest,
            MacroAction::AcceptQuest { .. } => MacroKind::AcceptQuest,
            MacroAction::Bid         { .. } => MacroKind::Bid,
            MacroAction::Announce    { .. } => MacroKind::Announce,
        }
    }
}
