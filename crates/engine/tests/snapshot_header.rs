use engine::snapshot::{SnapshotHeader, FORMAT_VERSION, MAGIC};

#[test]
fn header_serializes_to_64_bytes() {
    let h = SnapshotHeader {
        magic: *MAGIC,
        schema_hash: [0xAB; 32],
        tick: 42,
        seed: 0xDEADBEEF_CAFEF00D,
        format_version: FORMAT_VERSION,
        reserved: [0; 10],
    };
    let bytes = h.to_bytes();
    assert_eq!(bytes.len(), 64);
    let h2 = SnapshotHeader::from_bytes(&bytes).unwrap();
    assert_eq!(h.schema_hash, h2.schema_hash);
    assert_eq!(h.tick, h2.tick);
    assert_eq!(h.seed, h2.seed);
    assert_eq!(h.format_version, h2.format_version);
}

#[test]
fn magic_is_wsimsv01_ascii() {
    assert_eq!(MAGIC, b"WSIMSV01");
}

#[test]
fn from_bytes_rejects_bad_magic() {
    let mut bytes = vec![0u8; 64];
    bytes[..8].copy_from_slice(b"NOPEXXXX");
    assert!(SnapshotHeader::from_bytes(&bytes).is_err());
}

#[test]
fn from_bytes_rejects_short_input() {
    let bytes = vec![0u8; 10];
    assert!(SnapshotHeader::from_bytes(&bytes).is_err());
}
