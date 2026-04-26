//! Plan 3 Task 4: single-step migration registry.

use engine::event::EventRing;
use engine_data::events::Event;
use engine::snapshot::{
    load_snapshot, load_snapshot_with_migrations, save_snapshot, MigrationRegistry, SnapshotError,
};
use engine::state::SimState;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn tmp_path(name: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("engine_snap_{}_{}_{}.bin", name, pid, nonce))
}

/// Corrupt the snapshot's schema_hash to simulate a file written by an older
/// engine. The returned `[u8; 32]` is the forged "old" hash; the body is
/// untouched (still conforms to the current layout) so our migration
/// closure's job is simply to splice the current hash back into the header.
fn fake_old_hash_file(path: &std::path::Path) -> [u8; 32] {
    let mut buf = std::fs::read(path).unwrap();
    // XOR the first four hash bytes to flip the fingerprint.
    for byte in &mut buf[8..12] {
        *byte ^= 0xA5;
    }
    let mut old_hash = [0u8; 32];
    old_hash.copy_from_slice(&buf[8..40]);
    std::fs::write(path, &buf).unwrap();
    old_hash
}

#[test]
fn migration_from_fake_old_hash_runs_on_load() {
    let state = SimState::new(4, 42);
    let events = EventRing::<Event>::with_cap(16);
    let path = tmp_path("mig");
    save_snapshot(&state, &events, &path).unwrap();

    let old_hash = fake_old_hash_file(&path);

    // Sanity: regular load fails with SchemaMismatch.
    match load_snapshot::<Event>(&path) {
        Err(SnapshotError::SchemaMismatch { .. }) => (),
        Err(other) => panic!("expected SchemaMismatch, got {:?}", other),
        Ok(_) => panic!("expected error, got Ok"),
    }

    // Register a migration from the forged old hash to the current hash.
    // The migration splices the current hash into offset 8..40 of the
    // header — no body changes because our forgery only touched the
    // header bytes.
    let current = engine::schema_hash::schema_hash();
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = called.clone();

    let mut reg = MigrationRegistry::new();
    reg.register(old_hash, current, move |bytes| {
        called_clone.store(true, Ordering::SeqCst);
        let mut out = bytes.to_vec();
        out[8..40].copy_from_slice(&current);
        Ok(out)
    });

    // Loading via the migration path succeeds.
    let (state2, _ring) = load_snapshot_with_migrations::<Event>(&path, &reg).unwrap();
    assert!(
        called.load(Ordering::SeqCst),
        "migration closure was not invoked"
    );
    assert_eq!(state2.seed, 42);
    assert_eq!(state2.tick, 0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn migration_registry_returns_error_when_no_path_registered() {
    let state = SimState::new(4, 42);
    let events = EventRing::<Event>::with_cap(16);
    let path = tmp_path("mig_none");
    save_snapshot(&state, &events, &path).unwrap();
    let _old_hash = fake_old_hash_file(&path);

    let reg = MigrationRegistry::new();
    match load_snapshot_with_migrations::<Event>(&path, &reg) {
        Err(SnapshotError::MigrationFailed(_)) => (),
        Err(other) => panic!("expected MigrationFailed, got {:?}", other),
        Ok(_) => panic!("expected error, got Ok"),
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn matching_hash_skips_migration_registry_entirely() {
    let state = SimState::new(4, 42);
    let events = EventRing::<Event>::with_cap(16);
    let path = tmp_path("mig_skip");
    save_snapshot(&state, &events, &path).unwrap();

    // Registry is empty — but since the hash already matches, load should
    // succeed without touching the registry.
    let reg = MigrationRegistry::new();
    let (state2, _) = load_snapshot_with_migrations::<Event>(&path, &reg).unwrap();
    assert_eq!(state2.seed, 42);

    std::fs::remove_file(&path).ok();
}
