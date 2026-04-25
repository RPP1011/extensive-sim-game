//! Communication channels. Type definitions moved to
//! `engine_data::types` at milestone 6 (2026-04-19) so the compiler-emitted
//! `Capabilities` struct can reference them without inverting the
//! `engine → engine_rules` dependency direction. The re-exports below keep
//! every `use engine::channel::{CommunicationChannel, ChannelSet}` call site
//! compiling unchanged.
//!
//! `channel_range` is the per-channel effective-range formula. Task 142
//! migrated the per-channel base distances (speech, pack, pheromone,
//! long-range-vocal) into `assets/sim/config.sim` as
//! `config.communication.channel_*_range`; the dispatch logic (which
//! channel gets vocal-strength scaling, which is unbounded, which is
//! silent) stays engine-primitive because it's a dispatch table keyed on
//! an engine enum, not a balance knob.

use engine_data::config::CommunicationConfig;
pub use engine_data::types::{ChannelSet, CommunicationChannel};

pub fn channel_range(
    channel: CommunicationChannel,
    vocal_strength: f32,
    cfg: &CommunicationConfig,
) -> f32 {
    match channel {
        CommunicationChannel::Speech => cfg.channel_speech_range * vocal_strength,
        CommunicationChannel::PackSignal => cfg.channel_pack_range,
        CommunicationChannel::Pheromone => cfg.channel_pheromone_range,
        CommunicationChannel::Song => cfg.channel_long_range_vocal,
        CommunicationChannel::Telepathy => f32::INFINITY,
        CommunicationChannel::Testimony => 0.0,
    }
}
