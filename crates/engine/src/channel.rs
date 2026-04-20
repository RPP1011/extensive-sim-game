//! Communication channels. Type definitions moved to
//! `engine_rules::types` at milestone 6 (2026-04-19) so the compiler-emitted
//! `Capabilities` struct can reference them without inverting the
//! `engine → engine_rules` dependency direction. The re-exports below keep
//! every `use engine::channel::{CommunicationChannel, ChannelSet}` call site
//! compiling unchanged.
//!
//! `channel_range` stays here: it's a hand-written engine primitive (range
//! per channel given vocal strength), not part of the compiled rule surface.

pub use engine_rules::types::{ChannelSet, CommunicationChannel};

pub fn channel_range(channel: CommunicationChannel, vocal_strength: f32) -> f32 {
    const SPEECH_RANGE: f32 = 30.0;
    const PACK_RANGE: f32 = 20.0;
    const PHEROMONE_RANGE: f32 = 40.0;
    const LONG_RANGE_VOCAL: f32 = 200.0;

    match channel {
        CommunicationChannel::Speech => SPEECH_RANGE * vocal_strength,
        CommunicationChannel::PackSignal => PACK_RANGE,
        CommunicationChannel::Pheromone => PHEROMONE_RANGE,
        CommunicationChannel::Song => LONG_RANGE_VOCAL,
        CommunicationChannel::Telepathy => f32::INFINITY,
        CommunicationChannel::Testimony => 0.0,
    }
}
