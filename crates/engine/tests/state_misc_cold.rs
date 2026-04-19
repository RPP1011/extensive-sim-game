//! Task K: misc cold SoA — class definitions, creditor ledger, mentor
//! lineage. state.md §AgentData.{Skill&Class, Economic, Relationships}.

use engine::ids::AgentId;
use engine::state::agent_types::{ClassSlot, Creditor, MentorLink};
use engine::state::{AgentSpawn, SimState};
use smallvec::SmallVec;

#[test]
fn spawn_defaults_classes_to_empty_slots() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let classes = state.agent_classes(a).unwrap();
    // Fixed [ClassSlot; 4] — all slots zeroed (class_tag=0, level=0 → unused).
    assert_eq!(classes.len(), 4);
    assert!(classes.iter().all(|c| c.class_tag == 0 && c.level == 0));
}

#[test]
fn set_and_read_classes() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_classes(
        a,
        [
            ClassSlot { class_tag: 100, level: 3 },
            ClassSlot { class_tag: 200, level: 1 },
            ClassSlot::default(),
            ClassSlot::default(),
        ],
    );
    let classes = state.agent_classes(a).unwrap();
    assert_eq!(classes[0].class_tag, 100);
    assert_eq!(classes[0].level, 3);
    assert_eq!(classes[1].level, 1);
    assert_eq!(classes[3].class_tag, 0);
}

#[test]
fn creditor_ledger_starts_empty_and_supports_push() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert!(state.agent_creditors(a).unwrap().is_empty());
    let c = Creditor {
        creditor: AgentId::new(7).unwrap(),
        amount:   500,
    };
    state.push_agent_creditor(a, c);
    assert_eq!(state.agent_creditors(a).unwrap(), &[c]);
}

#[test]
fn mentor_lineage_defaults_none() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let lineage = state.agent_mentor_lineage(a).unwrap();
    assert_eq!(lineage.len(), 8);
    assert!(lineage.iter().all(|x| x.is_none()));
}

#[test]
fn set_mentor_lineage() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let m1 = AgentId::new(10).unwrap();
    let mut lineage = [None; 8];
    lineage[0] = Some(m1);
    state.set_agent_mentor_lineage(a, lineage);
    assert_eq!(state.agent_mentor_lineage(a).unwrap()[0], Some(m1));
    assert_eq!(state.agent_mentor_lineage(a).unwrap()[1], None);
}

#[test]
fn bulk_slices_have_cap_length() {
    let state = SimState::new(8, 42);
    assert_eq!(state.cold_class_definitions().len(), 8);
    let ledger: &[SmallVec<[Creditor; 16]>] = state.cold_creditor_ledger();
    assert_eq!(ledger.len(), 8);
    assert_eq!(state.cold_mentor_lineage().len(), 8);
    // MentorLink stub — we reserve a placeholder but lineage is Option<AgentId>
    // per prompt. The type is still available for future use.
    let _m = MentorLink {
        mentor:     AgentId::new(1).unwrap(),
        discipline: 0,
    };
}
