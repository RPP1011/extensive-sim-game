use crate::channel::{ChannelSet, CommunicationChannel};
use smallvec::{smallvec, SmallVec};
use std::num::NonZeroU16;

#[derive(Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(u8)]
pub enum CreatureType {
    #[default]
    Human = 0,
    Wolf = 1,
    Deer = 2,
    Dragon = 3,
}

/// Niche-optimised language id. `Option<LanguageId>` is two bytes.
/// state.md §56 commits to per-capability `languages`; the concrete catalogue
/// (Common, Draconic, Elven, ...) is a later-plan concern; two built-in
/// constants seed the MVP species defaults.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(transparent)]
pub struct LanguageId(NonZeroU16);

impl LanguageId {
    pub const COMMON:   LanguageId = LanguageId(NonZeroU16::new(1).unwrap());
    pub const DRACONIC: LanguageId = LanguageId(NonZeroU16::new(2).unwrap());

    #[inline]
    pub fn new(raw: u16) -> Option<Self> {
        NonZeroU16::new(raw).map(Self)
    }

    #[inline]
    pub fn raw(self) -> u16 {
        self.0.get()
    }
}

/// Per-agent capability bundle — what the agent *can* do, structurally.
/// state.md §56 documents all fields. `channels` drives communication
/// eligibility; `can_fly` gates z-separation combat predicates; `languages`
/// is the set of tongues this agent understands (sorted, capped at 4
/// per state.md). Other booleans unlock specific action types
/// (`Build`/`Trade`/`Climb`/`Tunnel`/`Marry`).
#[derive(Clone, Debug)]
pub struct Capabilities {
    pub channels:    ChannelSet,
    pub languages:   SmallVec<[LanguageId; 4]>,
    pub can_fly:     bool,
    pub can_build:   bool,
    pub can_trade:   bool,
    pub can_climb:   bool,
    pub can_tunnel:  bool,
    pub can_marry:   bool,
    pub max_spouses: u8,
}

impl Capabilities {
    pub fn for_creature(ct: CreatureType) -> Self {
        use CommunicationChannel as CC;
        use CreatureType as Ct;
        match ct {
            Ct::Human => Self {
                channels:    smallvec![CC::Speech, CC::Testimony],
                languages:   smallvec![LanguageId::COMMON],
                can_fly:     false,
                can_build:   true,
                can_trade:   true,
                can_climb:   true,
                can_tunnel:  true,
                can_marry:   true,
                max_spouses: 1,
            },
            Ct::Wolf => Self {
                channels:    smallvec![CC::PackSignal],
                languages:   SmallVec::new(),
                can_fly:     false,
                can_build:   false,
                can_trade:   false,
                can_climb:   false,
                can_tunnel:  true,
                can_marry:   false,
                max_spouses: 0,
            },
            Ct::Deer => Self {
                channels:    smallvec![CC::PackSignal],
                languages:   SmallVec::new(),
                can_fly:     false,
                can_build:   false,
                can_trade:   false,
                can_climb:   false,
                can_tunnel:  false,
                can_marry:   false,
                max_spouses: 0,
            },
            Ct::Dragon => Self {
                channels:    smallvec![CC::Speech, CC::Song],
                languages:   smallvec![LanguageId::DRACONIC],
                can_fly:     true,
                can_build:   false,
                can_trade:   false,
                can_climb:   true,
                can_tunnel:  false,
                can_marry:   false,
                max_spouses: 0,
            },
        }
    }
}
