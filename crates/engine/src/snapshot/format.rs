//! Snapshot file format. 64-byte header + field blocks.
//!
//! Layout:
//! - magic            : [u8; 8]   offset 0..8
//! - schema_hash      : [u8; 32]  offset 8..40
//! - tick             : u32 LE    offset 40..44
//! - seed             : u64 LE    offset 44..52
//! - format_version   : u16 LE    offset 52..54
//! - reserved         : [u8; 10]  offset 54..64  (future fields)

pub const MAGIC: &[u8; 8] = b"WSIMSV01";
pub const FORMAT_VERSION: u16 = 1;

pub const HEADER_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotHeader {
    pub magic: [u8; 8],
    pub schema_hash: [u8; 32],
    pub tick: u32,
    pub seed: u64,
    pub format_version: u16,
    pub reserved: [u8; 10],
}

impl SnapshotHeader {
    pub fn to_bytes(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        out[0..8].copy_from_slice(&self.magic);
        out[8..40].copy_from_slice(&self.schema_hash);
        out[40..44].copy_from_slice(&self.tick.to_le_bytes());
        out[44..52].copy_from_slice(&self.seed.to_le_bytes());
        out[52..54].copy_from_slice(&self.format_version.to_le_bytes());
        out[54..64].copy_from_slice(&self.reserved);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.len() < HEADER_BYTES {
            return Err(SnapshotError::ShortHeader);
        }
        if &bytes[0..8] != MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        let mut schema_hash = [0u8; 32];
        schema_hash.copy_from_slice(&bytes[8..40]);
        let tick = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        let seed = u64::from_le_bytes(bytes[44..52].try_into().unwrap());
        let format_version = u16::from_le_bytes(bytes[52..54].try_into().unwrap());
        let mut reserved = [0u8; 10];
        reserved.copy_from_slice(&bytes[54..64]);
        Ok(Self {
            magic,
            schema_hash,
            tick,
            seed,
            format_version,
            reserved,
        })
    }
}

#[derive(Debug)]
pub enum SnapshotError {
    BadMagic,
    ShortHeader,
    SchemaMismatch {
        expected: [u8; 32],
        found: [u8; 32],
    },
    UnsupportedFormatVersion {
        found: u16,
        expected: u16,
    },
    Truncated(&'static str),
    Io(std::io::Error),
    /// Migration registered but failed, or no migration path exists.
    MigrationFailed(&'static str),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad magic bytes — not a snapshot file"),
            Self::ShortHeader => write!(f, "file shorter than 64-byte header"),
            Self::SchemaMismatch { expected, found } => write!(
                f,
                "schema hash mismatch — expected {:02x?}…, found {:02x?}…",
                &expected[..4],
                &found[..4],
            ),
            Self::UnsupportedFormatVersion { found, expected } => write!(
                f,
                "unsupported snapshot format version {} (this engine writes v{})",
                found, expected,
            ),
            Self::Truncated(what) => write!(f, "snapshot truncated while reading {}", what),
            Self::Io(e) => write!(f, "io error: {}", e),
            Self::MigrationFailed(msg) => write!(f, "migration failed: {}", msg),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        SnapshotError::Io(e)
    }
}
