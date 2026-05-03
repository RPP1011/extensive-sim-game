use engine::pool::BoundedMap;

#[test]
fn upsert_within_capacity() {
    let mut m: BoundedMap<u32, i32, 4> = BoundedMap::new();
    assert_eq!(m.get(&1), None);
    m.upsert(1, 10);
    m.upsert(2, 20);
    assert_eq!(m.get(&1), Some(&10));
    assert_eq!(m.get(&2), Some(&20));
    m.upsert(1, 100);
    assert_eq!(m.get(&1), Some(&100));
}

#[test]
fn lru_evicts_oldest_when_full() {
    let mut m: BoundedMap<u32, (i32, u32), 3> = BoundedMap::new();
    m.upsert(1, (10, 1));
    m.upsert(2, (20, 2));
    m.upsert(3, (30, 3));
    m.upsert(4, (40, 4));
    assert_eq!(m.get(&1), None);
    assert_eq!(m.get(&4), Some(&(40, 4)));
}

#[test]
fn retain_drops_filtered_entries() {
    let mut m: BoundedMap<u32, i32, 4> = BoundedMap::new();
    m.upsert(1, 10);
    m.upsert(2, 20);
    m.upsert(3, 30);
    m.retain(|_, v| *v >= 20);
    assert_eq!(m.get(&1), None);
    assert_eq!(m.get(&2), Some(&20));
    assert_eq!(m.get(&3), Some(&30));
}
