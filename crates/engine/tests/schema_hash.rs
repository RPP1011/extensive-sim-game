use engine::schema_hash::schema_hash;

#[test]
fn schema_hash_is_stable() {
    let h1 = schema_hash();
    let h2 = schema_hash();
    assert_eq!(h1, h2, "hash is a pure function of compile-time layout");
}

#[test]
fn schema_hash_matches_baseline() {
    let hash = schema_hash();
    let baseline = include_str!("../.schema_hash").trim();
    let actual = hex::encode(hash);
    assert_eq!(
        actual, baseline,
        "Schema hash changed. If intentional, update crates/engine/.schema_hash with the new hash. Current: {}",
        actual
    );
}
