use crate::channel::{ChannelSet, CommunicationChannel};
use smallvec::smallvec;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(u8)]
pub enum CreatureType {
    Human  = 0,
    Wolf   = 1,
    Deer   = 2,
    Dragon = 3,
}

#[derive(Clone, Debug)]
pub struct Capabilities {
    pub channels:    ChannelSet,
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
                can_fly:     false, can_build: true, can_trade: true,
                can_climb:   true,  can_tunnel: true, can_marry: true,
                max_spouses: 1,
            },
            Ct::Wolf => Self {
                channels:    smallvec![CC::PackSignal],
                can_fly:     false, can_build: false, can_trade: false,
                can_climb:   false, can_tunnel: true, can_marry: false,
                max_spouses: 0,
            },
            Ct::Deer => Self {
                channels:    smallvec![CC::PackSignal],
                can_fly:     false, can_build: false, can_trade: false,
                can_climb:   false, can_tunnel: false, can_marry: false,
                max_spouses: 0,
            },
            Ct::Dragon => Self {
                channels:    smallvec![CC::Speech, CC::Song],
                can_fly:     true,  can_build: false, can_trade: false,
                can_climb:   true,  can_tunnel: false, can_marry: false,
                max_spouses: 0,
            },
        }
    }
}
