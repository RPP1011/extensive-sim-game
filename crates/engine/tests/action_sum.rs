use engine::ids::{AgentId, QuestId};
use engine::mask::MicroKind;
use engine::policy::{
    Action, ActionKind, MacroAction, MicroTarget, QueryKind, QuestCategory, Resolution,
};
use glam::Vec3;

#[test]
fn construct_micro_variant() {
    let a = AgentId::new(1).unwrap();
    let act = Action {
        agent: a,
        kind: ActionKind::Micro {
            kind:   MicroKind::Attack,
            target: MicroTarget::Agent(a),
        },
    };
    match act.kind {
        ActionKind::Micro { kind: MicroKind::Attack, target: MicroTarget::Agent(_) } => (),
        _ => panic!(),
    }
}

#[test]
fn construct_macro_variant() {
    let a = AgentId::new(1).unwrap();
    let q = QuestId::new(1).unwrap();
    let act = Action {
        agent: a,
        kind:  ActionKind::Macro(MacroAction::PostQuest {
            quest_id:   q,
            category:   QuestCategory::Physical,
            resolution: Resolution::HighestBid,
        }),
    };
    assert_eq!(act.agent.raw(), 1);
    match act.kind {
        ActionKind::Macro(MacroAction::PostQuest { .. }) => (),
        _ => panic!(),
    }
}

#[test]
fn micro_target_covers_all_universal_branches() {
    let _ = MicroTarget::None;
    let _ = MicroTarget::Agent(AgentId::new(1).unwrap());
    let _ = MicroTarget::Position(Vec3::ZERO);
    let _ = MicroTarget::ItemSlot(0);
    let _ = MicroTarget::AbilityIdx(0);
    let _ = MicroTarget::Query(QueryKind::AboutAll);
    let _ = MicroTarget::Opaque(0xDEADBEEFu64);
}

#[test]
fn convenience_constructors() {
    let a = AgentId::new(1).unwrap();
    let h = Action::hold(a);
    assert!(matches!(
        h.kind,
        ActionKind::Micro { kind: MicroKind::Hold, target: MicroTarget::None }
    ));

    let mv = Action::move_toward(a, Vec3::new(5.0, 0.0, 10.0));
    assert!(matches!(
        mv.kind,
        ActionKind::Micro { kind: MicroKind::MoveToward, target: MicroTarget::Position(_) }
    ));

    let at = Action::attack(a, a);
    assert!(matches!(
        at.kind,
        ActionKind::Micro { kind: MicroKind::Attack, target: MicroTarget::Agent(_) }
    ));
}
