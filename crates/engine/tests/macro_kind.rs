use engine::ids::{AgentId, GroupId, QuestId};
use engine::policy::macro_kind::{
    AnnounceAudience, MacroAction, MacroKind,
};
use engine_data::types::{QuestCategory, Resolution};
use glam::Vec3;

#[test]
fn macro_kind_has_five_variants_including_noop() {
    assert_eq!(MacroKind::NoOp as u8, 0);
    assert_eq!(MacroKind::PostQuest as u8, 1);
    assert_eq!(MacroKind::AcceptQuest as u8, 2);
    assert_eq!(MacroKind::Bid as u8, 3);
    assert_eq!(MacroKind::Announce as u8, 4);
    assert_eq!(MacroKind::ALL.len(), 5);
}

#[test]
fn announce_audience_variants_disambiguate() {
    let g = GroupId::new(1).unwrap();
    let a = AnnounceAudience::Group(g);
    let b = AnnounceAudience::Area(Vec3::ZERO, 30.0);
    let c = AnnounceAudience::Anyone;
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_ne!(a, c);
}

#[test]
fn resolution_coalition_carries_min_parties() {
    let r = Resolution::Coalition { min_parties: 3 };
    match r {
        Resolution::Coalition { min_parties } => assert_eq!(min_parties, 3),
        _ => panic!("wrong variant"),
    }
}

#[test]
fn macro_action_kind_matches_variant() {
    let a = AgentId::new(1).unwrap();
    let q = QuestId::new(1).unwrap();
    let act = MacroAction::PostQuest {
        quest_id: q,
        category: QuestCategory::Physical,
        resolution: Resolution::HighestBid,
    };
    assert_eq!(act.kind(), MacroKind::PostQuest);

    let noop = MacroAction::NoOp;
    assert_eq!(noop.kind(), MacroKind::NoOp);

    let _ = a;
}

#[test]
fn quest_category_ordering_stable() {
    assert_eq!(QuestCategory::Physical  as u8, 0);
    assert_eq!(QuestCategory::Political as u8, 1);
    assert_eq!(QuestCategory::Personal  as u8, 2);
    assert_eq!(QuestCategory::Economic  as u8, 3);
    assert_eq!(QuestCategory::Narrative as u8, 4);
}
