use engine::ids::{AgentId, GroupId};
use engine::policy::query::{EntityQueryRef, MemoryKind, QueryKind};

#[test]
fn about_all_is_the_sugar_target() {
    // Read(doc) lowers to Ask(doc, QueryKind::AboutAll).
    let q = QueryKind::AboutAll;
    match q {
        QueryKind::AboutAll => (),
        _ => panic!("AboutAll variant missing"),
    }
}

#[test]
fn query_kind_about_entity_holds_ref() {
    let a = AgentId::new(1).unwrap();
    let q = QueryKind::AboutEntity(EntityQueryRef::Agent(a));
    let g = GroupId::new(1).unwrap();
    let q2 = QueryKind::AboutEntity(EntityQueryRef::Group(g));
    assert_ne!(q, q2);
}

#[test]
fn memory_kind_ordinals() {
    assert_eq!(MemoryKind::Combat    as u8, 0);
    assert_eq!(MemoryKind::Trade     as u8, 1);
    assert_eq!(MemoryKind::Social    as u8, 2);
    assert_eq!(MemoryKind::Political as u8, 3);
    assert_eq!(MemoryKind::Other     as u8, 4);
}
