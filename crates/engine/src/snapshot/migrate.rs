//! Snapshot migration registry. One-step hop from an older schema hash to
//! the current one; chain composition is future work.

use super::format::SnapshotError;

pub type MigrationFn =
    Box<dyn Fn(&[u8]) -> Result<Vec<u8>, SnapshotError> + Send + Sync + 'static>;

pub struct MigrationRegistry {
    migrations: Vec<([u8; 32], [u8; 32], MigrationFn)>,
}

impl MigrationRegistry {
    pub fn new() -> Self {
        Self {
            migrations: Vec::new(),
        }
    }

    /// Register a migration from `from` schema hash to `to` schema hash.
    pub fn register<F>(&mut self, from: [u8; 32], to: [u8; 32], f: F)
    where
        F: Fn(&[u8]) -> Result<Vec<u8>, SnapshotError> + Send + Sync + 'static,
    {
        self.migrations.push((from, to, Box::new(f)));
    }

    /// Apply a one-step migration if one exists from `from` → `to`.
    ///
    /// Chain composition (from → mid → to) is intentionally not implemented;
    /// add it when multi-step migrations are needed.
    pub fn apply(
        &self,
        from: [u8; 32],
        to: [u8; 32],
        bytes: &[u8],
    ) -> Result<Vec<u8>, SnapshotError> {
        for (f_from, f_to, f) in &self.migrations {
            if *f_from == from && *f_to == to {
                return f(bytes);
            }
        }
        Err(SnapshotError::MigrationFailed(
            "no one-step migration registered for this schema-hash pair",
        ))
    }
}

impl Default for MigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Load a snapshot, applying a one-step migration if the file's schema hash
/// doesn't match the current engine hash but a registered migration covers
/// the gap. The migration closure is handed the **entire file bytes**
/// (header + body) and must return a new byte buffer whose header carries
/// the current schema hash — usually by splicing the current hash into the
/// header while rewriting the body to match the current layout.
///
/// On hash match, falls through to [`crate::snapshot::load_snapshot`].
/// On hash mismatch with no registered migration, returns the same
/// `SchemaMismatch` error the non-migration path would.
pub fn load_snapshot_with_migrations<E: crate::event::EventLike>(
    path: &std::path::Path,
    reg: &MigrationRegistry,
) -> Result<(crate::state::SimState, crate::event::EventRing<E>), SnapshotError> {
    let bytes = std::fs::read(path)?;
    let header = super::format::SnapshotHeader::from_bytes(&bytes)?;
    let current = crate::schema_hash::schema_hash();
    if header.schema_hash == current {
        return super::format::load_from_bytes(&bytes);
    }
    let migrated = reg.apply(header.schema_hash, current, &bytes)?;
    super::format::load_from_bytes(&migrated)
}
