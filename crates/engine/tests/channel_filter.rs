use engine::channel::channel_range;
use engine_data::types::CommunicationChannel;
use engine_data::entities::{Capabilities, CreatureType};
use engine_data::config::CommunicationConfig;

#[test]
fn wolves_share_packsignal_not_speech() {
    let wolf = Capabilities::for_creature(CreatureType::Wolf);
    let human = Capabilities::for_creature(CreatureType::Human);
    assert!(wolf.channels.contains(&CommunicationChannel::PackSignal));
    assert!(!wolf.channels.contains(&CommunicationChannel::Speech));
    assert!(human.channels.contains(&CommunicationChannel::Speech));
    assert!(!human.channels.contains(&CommunicationChannel::PackSignal));

    let shared = wolf.channels.iter().any(|c| human.channels.contains(c));
    assert!(
        !shared,
        "wolves and humans must not share a channel by default"
    );
}

#[test]
fn speech_range_is_vocal_strength_scaled() {
    let cfg = CommunicationConfig::default();
    let base = channel_range(CommunicationChannel::Speech, 1.0, &cfg);
    let loud = channel_range(CommunicationChannel::Speech, 2.0, &cfg);
    assert_eq!(loud, 2.0 * base);
}

#[test]
fn telepathy_is_unbounded() {
    let cfg = CommunicationConfig::default();
    let r = channel_range(CommunicationChannel::Telepathy, 1.0, &cfg);
    assert!(r.is_infinite());
}
