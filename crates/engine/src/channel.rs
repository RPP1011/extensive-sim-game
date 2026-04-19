use smallvec::SmallVec;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(u8)]
pub enum CommunicationChannel {
    Speech      = 0,
    PackSignal  = 1,
    Pheromone   = 2,
    Song        = 3,
    Telepathy   = 4,
    Testimony   = 5,
}

pub type ChannelSet = SmallVec<[CommunicationChannel; 4]>;

pub fn channel_range(channel: CommunicationChannel, vocal_strength: f32) -> f32 {
    const SPEECH_RANGE: f32 = 30.0;
    const PACK_RANGE: f32 = 20.0;
    const PHEROMONE_RANGE: f32 = 40.0;
    const LONG_RANGE_VOCAL: f32 = 200.0;

    match channel {
        CommunicationChannel::Speech     => SPEECH_RANGE * vocal_strength,
        CommunicationChannel::PackSignal => PACK_RANGE,
        CommunicationChannel::Pheromone  => PHEROMONE_RANGE,
        CommunicationChannel::Song       => LONG_RANGE_VOCAL,
        CommunicationChannel::Telepathy  => f32::INFINITY,
        CommunicationChannel::Testimony  => 0.0,
    }
}
