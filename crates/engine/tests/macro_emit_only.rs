use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::{AgentId, QuestId};
use engine::mask::MaskBuffer;
use engine::policy::{Action, ActionKind, MacroAction, PolicyBackend, QuestCategory, Resolution};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

fn make() -> (SimState, SimScratch, EventRing, CascadeRegistry, AgentId) {
    let mut state = SimState::new(4, 42);
    let scratch = SimScratch::new(state.agent_cap() as usize);
    let events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    (state, scratch, events, cascade, a)
}

struct EmitMacro(AgentId, MacroAction);
impl PolicyBackend for EmitMacro {
    fn evaluate(&self, _s: &SimState, _m: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        out.push(Action {
            agent: self.0,
            kind: ActionKind::Macro(self.1),
        });
    }
}

#[test]
fn postquest_emits_quest_posted_event() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let q = QuestId::new(1).unwrap();
    let b = EmitMacro(
        a,
        MacroAction::PostQuest {
            quest_id: q,
            category: QuestCategory::Physical,
            resolution: Resolution::HighestBid,
        },
    );
    step(&mut state, &mut scratch, &mut events, &b, &cascade);
    let found = events.iter().any(|e| match e {
        Event::QuestPosted { poster, quest_id, category, resolution, .. } => {
            *poster == a
                && *quest_id == q
                && *category == QuestCategory::Physical
                && *resolution == Resolution::HighestBid
        }
        _ => false,
    });
    assert!(found, "QuestPosted emitted");
}

#[test]
fn acceptquest_emits_quest_accepted_event() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let q = QuestId::new(7).unwrap();
    let b = EmitMacro(
        a,
        MacroAction::AcceptQuest {
            quest_id: q,
            acceptor: a,
        },
    );
    step(&mut state, &mut scratch, &mut events, &b, &cascade);
    assert!(events.iter().any(|e| matches!(e,
        Event::QuestAccepted { acceptor, quest_id, .. }
            if *acceptor == a && *quest_id == q)));
}

#[test]
fn bid_emits_bid_placed_event() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let q = QuestId::new(3).unwrap();
    let b = EmitMacro(
        a,
        MacroAction::Bid {
            auction_id: q,
            bidder: a,
            amount: 42.5,
        },
    );
    step(&mut state, &mut scratch, &mut events, &b, &cascade);
    let found = events.iter().any(|e| match e {
        Event::BidPlaced { bidder, auction_id, amount, .. } => {
            *bidder == a && *auction_id == q && (*amount - 42.5).abs() < 1e-6
        }
        _ => false,
    });
    assert!(found);
}

#[test]
fn noop_macro_emits_nothing() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let b = EmitMacro(a, MacroAction::NoOp);
    step(&mut state, &mut scratch, &mut events, &b, &cascade);
    let count_macro = events
        .iter()
        .filter(|e| matches!(e,
            Event::QuestPosted { .. } | Event::QuestAccepted { .. } | Event::BidPlaced { .. }
        ))
        .count();
    assert_eq!(count_macro, 0);
}
