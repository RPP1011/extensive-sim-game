use engine::event::EventRing;
use engine_data::events::Event;
use engine::snapshot::{load_snapshot, save_snapshot, SnapshotError};
use engine::state::SimState;

fn tmp_path(name: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("engine_snap_{}_{}_{}.bin", name, pid, nonce))
}

#[test]
fn load_rejects_snapshot_with_wrong_schema_hash() {
    let state = SimState::new(4, 42);
    let events = EventRing::<Event>::with_cap(16);
    let path = tmp_path("sm");
    save_snapshot(&state, &events, &path).unwrap();

    // Corrupt the schema_hash bytes (offset 8..40 in the header).
    let mut buf = std::fs::read(&path).unwrap();
    buf[8] ^= 0xFF;
    std::fs::write(&path, &buf).unwrap();

    match load_snapshot::<Event>(&path) {
        Err(SnapshotError::SchemaMismatch { .. }) => (),
        Err(other) => panic!("expected SchemaMismatch, got {:?}", other),
        Ok(_) => panic!("expected error, got Ok"),
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn load_rejects_short_file() {
    let path = tmp_path("short");
    std::fs::write(&path, b"short").unwrap();
    match load_snapshot::<Event>(&path) {
        Err(SnapshotError::ShortHeader) | Err(SnapshotError::Io(_)) => (),
        Err(other) => panic!("expected ShortHeader or Io, got {:?}", other),
        Ok(_) => panic!("expected error, got Ok"),
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn load_rejects_bad_magic() {
    let path = tmp_path("magic");
    // 64 bytes of zeros — passes length check, fails magic.
    std::fs::write(&path, vec![0u8; 64]).unwrap();
    match load_snapshot::<Event>(&path) {
        Err(SnapshotError::BadMagic) => (),
        Err(other) => panic!("expected BadMagic, got {:?}", other),
        Ok(_) => panic!("expected error, got Ok"),
    }
    std::fs::remove_file(&path).ok();
}
