use engine::channel::{channel_range, CommunicationChannel};
use engine::creature::{Capabilities, CreatureType};

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
    let base = channel_range(CommunicationChannel::Speech, 1.0);
    let loud = channel_range(CommunicationChannel::Speech, 2.0);
    assert_eq!(loud, 2.0 * base);
}

#[test]
fn telepathy_is_unbounded() {
    let r = channel_range(CommunicationChannel::Telepathy, 1.0);
    assert!(r.is_infinite());
}
