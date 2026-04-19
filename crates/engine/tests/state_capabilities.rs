//! Task F: `Capabilities::languages` + per-creature defaults. state.md §56
//! commits to `languages` as part of the capability bundle beside channels.
//!
//! The other `Capabilities` fields (can_fly, can_build, can_trade, can_climb,
//! can_tunnel, can_marry, max_spouses) already exist — this test keeps them
//! pinned so a refactor doesn't erase them.

use engine::channel::CommunicationChannel;
use engine::creature::{Capabilities, CreatureType, LanguageId};

#[test]
fn human_capabilities_include_common_language_and_marry() {
    let caps = Capabilities::for_creature(CreatureType::Human);
    assert!(caps.can_build);
    assert!(caps.can_trade);
    assert!(caps.can_climb);
    assert!(caps.can_marry);
    assert_eq!(caps.max_spouses, 1);
    assert!(!caps.can_fly);
    // Humans start with one shared language (Common).
    assert_eq!(caps.languages.len(), 1);
    assert!(caps.languages.contains(&LanguageId::COMMON));
    // Channels unchanged by Task F.
    assert!(caps.channels.contains(&CommunicationChannel::Speech));
    assert!(caps.channels.contains(&CommunicationChannel::Testimony));
}

#[test]
fn wolf_capabilities_have_pack_channel_and_no_languages() {
    let caps = Capabilities::for_creature(CreatureType::Wolf);
    assert!(!caps.can_build);
    assert!(!caps.can_fly);
    assert_eq!(caps.max_spouses, 0);
    assert!(caps.channels.contains(&CommunicationChannel::PackSignal));
    // Wolves don't speak a language — empty.
    assert!(caps.languages.is_empty());
}

#[test]
fn dragon_capabilities_can_fly_zero_spouses() {
    let caps = Capabilities::for_creature(CreatureType::Dragon);
    assert!(caps.can_fly);
    assert!(caps.can_climb);
    assert_eq!(caps.max_spouses, 0);
    assert!(!caps.can_marry);
    // Dragons speak in a tongue of their own — dragon language.
    assert_eq!(caps.languages.len(), 1);
    assert!(caps.languages.contains(&LanguageId::DRACONIC));
}

#[test]
fn language_ids_are_nonzero_and_ordered() {
    // A u16 niche-optimised id so Option<LanguageId> is still two bytes.
    assert_eq!(std::mem::size_of::<LanguageId>(), 2);
    assert_eq!(std::mem::size_of::<Option<LanguageId>>(), 2);
    assert!(LanguageId::COMMON.raw() > 0);
    assert_ne!(LanguageId::COMMON, LanguageId::DRACONIC);
}
