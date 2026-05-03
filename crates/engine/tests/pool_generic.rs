use engine::pool::{Pool, PoolId};

struct AgentTag;

#[test]
fn alloc_gives_sequential_ids_from_one() {
    let mut p: Pool<AgentTag> = Pool::new(4);
    assert_eq!(p.alloc().map(|i| i.raw()), Some(1));
    assert_eq!(p.alloc().map(|i| i.raw()), Some(2));
    assert_eq!(p.alloc().map(|i| i.raw()), Some(3));
}

#[test]
fn alloc_returns_none_at_capacity() {
    let mut p: Pool<AgentTag> = Pool::new(2);
    assert!(p.alloc().is_some());
    assert!(p.alloc().is_some());
    assert!(p.alloc().is_none());
}

#[test]
fn kill_then_alloc_reuses_slot() {
    let mut p: Pool<AgentTag> = Pool::new(4);
    let a = p.alloc().unwrap();
    let b = p.alloc().unwrap();
    p.kill(a);
    let c = p.alloc().unwrap();
    assert_eq!(c.raw(), a.raw(), "freelist popped, slot reused");
    assert_ne!(b.raw(), c.raw());
}

#[test]
fn is_alive_tracks_state() {
    let mut p: Pool<AgentTag> = Pool::new(4);
    let a = p.alloc().unwrap();
    assert!(p.is_alive(a));
    p.kill(a);
    assert!(!p.is_alive(a));
}

// Suppress unused warnings for the tag type
#[allow(dead_code)]
fn _keep_tag_used(_: PoolId<AgentTag>) {}
