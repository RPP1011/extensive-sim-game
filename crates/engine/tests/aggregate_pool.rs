use engine::aggregate::{AggregatePool, Group, Quest};
use engine::policy::{QuestCategory, Resolution};

#[test]
fn alloc_returns_pool_id() {
    let mut pool: AggregatePool<Quest> = AggregatePool::new(16);
    let id = pool.alloc(Quest::stub(42)).unwrap();
    assert_eq!(id.raw(), 1);
    assert_eq!(pool.get(id).map(|q| q.seq), Some(42));
}

#[test]
fn alloc_sequential_ids() {
    let mut pool: AggregatePool<Quest> = AggregatePool::new(4);
    let a = pool.alloc(Quest::stub(1)).unwrap();
    let b = pool.alloc(Quest::stub(2)).unwrap();
    assert_eq!(a.raw(), 1);
    assert_eq!(b.raw(), 2);
}

#[test]
fn alloc_returns_none_at_capacity() {
    let mut pool: AggregatePool<Quest> = AggregatePool::new(2);
    assert!(pool.alloc(Quest::stub(1)).is_some());
    assert!(pool.alloc(Quest::stub(2)).is_some());
    assert!(pool.alloc(Quest::stub(3)).is_none());
}

#[test]
fn kill_then_alloc_reuses_slot_and_clears_contents() {
    let mut pool: AggregatePool<Quest> = AggregatePool::new(4);
    let a = pool.alloc(Quest::stub(1)).unwrap();
    pool.kill(a);
    assert!(pool.get(a).is_none(), "slot cleared on kill");
    let b = pool.alloc(Quest::stub(99)).unwrap();
    assert_eq!(a.raw(), b.raw(), "freelist reuse");
    assert_eq!(pool.get(b).map(|q| q.seq), Some(99));
}

#[test]
fn get_mut_allows_in_place_update() {
    let mut pool: AggregatePool<Quest> = AggregatePool::new(4);
    let id = pool.alloc(Quest::stub(5)).unwrap();
    pool.get_mut(id).unwrap().seq = 50;
    assert_eq!(pool.get(id).map(|q| q.seq), Some(50));
}

#[test]
fn quest_stub_defaults_match_spec() {
    let q = Quest::stub(0);
    assert_eq!(q.category, QuestCategory::Physical);
    assert_eq!(q.resolution, Resolution::HighestBid);
    assert!(q.poster.is_none());
    assert!(q.acceptors.is_empty());
    assert_eq!(q.posted_tick, 0);
}

#[test]
fn group_empty_constructor() {
    let g = Group::empty(0xF00DCAFE);
    assert_eq!(g.kind_tag, 0xF00DCAFE);
    assert!(g.members.is_empty());
    assert!(g.leader.is_none());
}
